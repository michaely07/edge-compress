"""Fused knowledge-distillation loss in Triton: forward and backward.

For one row, with p = softmax(s/T) (student) and q = softmax(t/T) (teacher):

    L = T^2/B * sum_rows sum_j q_j (log q_j - log p_j)        dL/ds_j = T/B * (p_j - q_j)

The gradient needs only the two per-row log-sum-exps saved by the forward pass.

Two forward kernels:
  v1  the straightforward version: three loops per row (online LSE over s, online LSE over t, then
      the KL sum). Every input element is read twice.
  v2  one loop per row. It uses  sum_j q_j (t_j - s_j)/T = W / Z_t  with
      W = sum_j exp(t_j/T - m) (t_j - s_j)/T  and  Z_t = sum_j exp(t_j/T - m),
      where W is rescaled online exactly like Z_t when the running max m moves. Then
      KL = W/Z_t - lse_t + lse_s, so every input element is read once. (Merging just the two LSE
      loops would not cut bytes: the KL loop still needs a second pass.)

Backward (shared): one pass that reads s and t, writes the gradient directly in the input dtype,
and reads the upstream gradient from device memory, so there is no host sync and no extra
`grad * g` pass over B x V.

Develop without a GPU: TRITON_INTERPRET=1 runs these kernels on CPU tensors.
Assumes finite logits (a -inf teacher logit would make (t - s) undefined).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _online_lse(ptr, V, inv_temp, BLOCK: tl.constexpr):
    # Per-lane online log-sum-exp over a row of length V, read in BLOCK-sized tiles
    m = tl.full([BLOCK], float("-inf"), tl.float32)
    d = tl.zeros([BLOCK], tl.float32)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        x = tl.load(ptr + cols, mask=cols < V, other=float("-inf")).to(tl.float32) * inv_temp
        m_new = tl.maximum(m, x)
        live = m_new != float("-inf")  # lanes that have seen no data stay 0
        d = tl.where(live, d * tl.exp(m - m_new) + tl.exp(x - m_new), 0.0)
        m = m_new
    M = tl.max(m, 0)
    return M + tl.log(tl.sum(d * tl.exp(m - M), 0))


@triton.jit
def kd_fwd_kernel(S, Tt, LOSS, LSE_S, LSE_T, stride_s, stride_t, V, inv_temp,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    s_ptr = S + row * stride_s
    t_ptr = Tt + row * stride_t
    lse_s = _online_lse(s_ptr, V, inv_temp, BLOCK)
    lse_t = _online_lse(t_ptr, V, inv_temp, BLOCK)
    acc = tl.zeros([BLOCK], tl.float32)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        ls = tl.load(s_ptr + cols, mask=mask, other=0.0).to(tl.float32) * inv_temp - lse_s
        lt = tl.load(t_ptr + cols, mask=mask, other=0.0).to(tl.float32) * inv_temp - lse_t
        acc += tl.where(mask, tl.exp(lt) * (lt - ls), 0.0)
    tl.store(LOSS + row, tl.sum(acc, 0))
    tl.store(LSE_S + row, lse_s)
    tl.store(LSE_T + row, lse_t)


@triton.jit
def kd_fwd_kernel_v2(S, Tt, LOSS, LSE_S, LSE_T, stride_s, stride_t, V, inv_temp,
                     BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    s_ptr = S + row * stride_s
    t_ptr = Tt + row * stride_t
    ninf = float("-inf")
    m_s = tl.full([BLOCK], ninf, tl.float32)
    d_s = tl.zeros([BLOCK], tl.float32)
    m_t = tl.full([BLOCK], ninf, tl.float32)
    d_t = tl.zeros([BLOCK], tl.float32)
    w = tl.zeros([BLOCK], tl.float32)
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        s = tl.load(s_ptr + cols, mask=mask, other=ninf).to(tl.float32) * inv_temp
        t = tl.load(t_ptr + cols, mask=mask, other=ninf).to(tl.float32) * inv_temp
        # student: online log-sum-exp
        ms_new = tl.maximum(m_s, s)
        d_s = tl.where(ms_new != ninf, d_s * tl.exp(m_s - ms_new) + tl.exp(s - ms_new), 0.0)
        m_s = ms_new
        # teacher: online log-sum-exp and the q-weighted (t - s) sum, rescaled together
        mt_new = tl.maximum(m_t, t)
        live = mt_new != ninf
        rescale = tl.exp(m_t - mt_new)
        e = tl.exp(t - mt_new)
        diff = tl.where(mask, t - s, 0.0)
        d_t = tl.where(live, d_t * rescale + e, 0.0)
        w = tl.where(live, w * rescale + e * diff, 0.0)
        m_t = mt_new
    Ms = tl.max(m_s, 0)
    lse_s = Ms + tl.log(tl.sum(d_s * tl.exp(m_s - Ms), 0))
    Mt = tl.max(m_t, 0)
    lane = tl.exp(m_t - Mt)
    z_t = tl.sum(d_t * lane, 0)
    lse_t = Mt + tl.log(z_t)
    kl = tl.sum(w * lane, 0) / z_t - lse_t + lse_s
    tl.store(LOSS + row, kl)
    tl.store(LSE_S + row, lse_s)
    tl.store(LSE_T + row, lse_t)


@triton.jit
def kd_bwd_kernel(S, Tt, LSE_S, LSE_T, GRAD, G, stride_s, stride_t, stride_g, V,
                  inv_temp, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    lse_s = tl.load(LSE_S + row)
    lse_t = tl.load(LSE_T + row)
    g = tl.load(G).to(tl.float32) * scale  # upstream grad read on device: no host sync
    for off in range(0, V, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < V
        s = tl.load(S + row * stride_s + cols, mask=mask, other=0.0).to(tl.float32)
        t = tl.load(Tt + row * stride_t + cols, mask=mask, other=0.0).to(tl.float32)
        p = tl.exp(s * inv_temp - lse_s)
        q = tl.exp(t * inv_temp - lse_t)
        tl.store(GRAD + row * stride_g + cols, (p - q) * g, mask=mask)  # cast to GRAD's dtype


def _launch_cfg(V):
    BLOCK = min(triton.next_power_of_2(V), 4096)
    return BLOCK, (4 if BLOCK <= 1024 else 8)


class FusedKDLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, t, T, version):
        s, t = s.contiguous(), t.contiguous()
        B, V = s.shape
        loss = torch.empty(B, device=s.device, dtype=torch.float32)
        lse_s, lse_t = torch.empty_like(loss), torch.empty_like(loss)
        BLOCK, warps = _launch_cfg(V)
        kern = kd_fwd_kernel_v2 if version == 2 else kd_fwd_kernel
        kern[(B,)](s, t, loss, lse_s, lse_t, s.stride(0), t.stride(0), V, 1.0 / T,
                   BLOCK=BLOCK, num_warps=warps)
        ctx.save_for_backward(s, t, lse_s, lse_t)
        ctx.T = T
        return loss.mean() * T * T

    @staticmethod
    def backward(ctx, g):
        s, t, lse_s, lse_t = ctx.saved_tensors
        B, V = s.shape
        grad = torch.empty_like(s)
        BLOCK, warps = _launch_cfg(V)
        kd_bwd_kernel[(B,)](s, t, lse_s, lse_t, grad, g.contiguous(), s.stride(0), t.stride(0),
                            grad.stride(0), V, 1.0 / ctx.T, ctx.T / B, BLOCK=BLOCK, num_warps=warps)
        return grad, None, None, None


def fused_kd_loss(student_logits, teacher_logits, T, version=2):
    """Drop-in for train.kd_loss: KL(teacher || student) at temperature T, times T^2, batch mean."""
    return FusedKDLoss.apply(student_logits, teacher_logits.detach(), float(T), version)


def fused_kd_loss_v1(student_logits, teacher_logits, T):
    return fused_kd_loss(student_logits, teacher_logits, T, version=1)
