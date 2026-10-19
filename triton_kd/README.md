# Fused knowledge-distillation loss in Triton

One Triton kernel pair (forward + backward) for the KD loss used in `train.py`, checked against
PyTorch, benchmarked against PyTorch eager and `torch.compile`, and read back through every
compiler stage.

<!-- Headline sentence goes here once benchmarks exist, e.g. "At a 128k vocabulary the fused loss reaches
     X GB/s (Y% of peak) and uses Z× less peak memory than PyTorch eager." -->

## Why this kernel

PyTorch's eager KD loss materializes two full `B×V` log-softmax tensors (plus the elementwise KL)
before reducing to a scalar. A fused kernel streams each row and writes one scalar per row. At 100
classes that is noise; at LLM vocabulary sizes (32k–128k), where distillation of language models
actually happens, it is memory traffic and peak memory saved.

## The math

For one row, with `p = softmax(s/T)` (student) and `q = softmax(t/T)` (teacher):

```
L = T²/B · Σ_rows Σ_j q_j (log q_j − log p_j)

log p_j = s_j/T − lse(s/T)      ⇒   ∂ log p_k / ∂ s_j = (1/T)(δ_jk − p_j)

∂L/∂s_j = T²/B · Σ_k q_k · (−1/T)(δ_jk − p_j)
        = T²/B · (−1/T)(q_j − p_j Σ_k q_k)
        = T/B · (p_j − q_j)                     (Σ_k q_k = 1)
```

The gradient is a scaled difference of two softmaxes, so backward needs only the two per-row
log-sum-exps saved by the forward pass. Nothing of size `B×V` is saved except the inputs.

## Kernels (`kernel.py`)

| | Forward reads per input element | Notes |
|---|---|---|
| **v1** | 2 | Three loops per row: online LSE(s), online LSE(t), then the KL sum |
| **v2** | **1** | One loop per row: both online LSEs **and** the KL numerator, rescaled online together |

**v2 idea.** Because `Σ_j q_j = 1`,

```
KL = Σ_j q_j (t_j − s_j)/T − lse_t + lse_s,     Σ_j q_j (t_j − s_j)/T = W / Z_t
W  = Σ_j e^{t_j/T − m} (t_j − s_j)/T,           Z_t = Σ_j e^{t_j/T − m}
```

`W` is rescaled by `e^{m_old − m_new}` whenever the running max `m` moves, exactly like `Z_t` in the
online softmax, so the whole loss falls out of a single pass. Merging only the two LSE loops into one
would *not* reduce bytes, because the KL loop would still need a second read.

**Backward** (shared by v1 and v2) reads `s` and `t` once, writes the gradient directly in the input
dtype, and loads the upstream gradient from device memory, so there is no host sync and no extra
`grad * g` pass over `B×V`.

## Correctness (`test_kernel.py`)

Compared with the reference `kd_loss` from `train.py`, for both the loss and the gradient:

- shapes `(B, V)` = (1, 7), (64, 100), (32, 1000), (8, 32000), (4, 128256) — `V = 7` exercises the masked tail of a single tile
- `T ∈ {1, 4}`, fp32 and fp16 inputs, v1 and v2
- stability: logits scaled by 1e3 and 1e4 (the online max must prevent overflow), all-equal rows,
  identical student/teacher (loss and gradient exactly 0), a non-contiguous transposed slice
- the upstream gradient is applied (`(loss * 3).backward()`)

Tolerances: fp32 `rtol=1e-4`. fp16 `rtol=atol=2e-3`: the kernel upcasts to fp32 internally, so the
loss stays tight, but the gradient is rounded to fp16 on store (relative spacing ≈ 1e-3) and the
reference rounds at different points. Large-logit cases use `atol = 1e-6 × scale`, because the loss
is a difference of terms of size `scale/T`.

```bash
pytest triton_kd -q                        # on a GPU
TRITON_INTERPRET=1 pytest triton_kd -q     # on CPU (interpreter): 44 passed, 8 large-fp16 cases skipped
```

## Benchmarks (`bench.py`)

`triton.testing.do_bench`, B = 2048 rows, V ∈ {100, 1k, 8k, 32k, 128k}, forward and forward+backward
separately, for PyTorch eager, `torch.compile(kd_loss)`, fused v1 and fused v2.

- **Effective bandwidth** uses the minimum bytes any implementation must move (fwd: read `s` and `t`
  once = `2·B·V·elt`; fwd+bwd: `5·B·V·elt`), so implementations are comparable. v1's real forward
  traffic is twice that and is reported separately (`fwd_actual_gbps`).
- **Peak memory**: `torch.cuda.max_memory_allocated` above the inputs, for one fwd+bwd step.

<!-- RESULTS:triton -->
_Benchmark results will appear here once `python triton_kd/bench.py --update-readme` has run on a GPU._
<!-- /RESULTS:triton -->

`torch.compile` is the strong baseline here: Inductor generates its own fused Triton reduction kernels for
this pattern. Where it matches the hand-written kernel, the tables show that as measured.

## Reading the compiler's output

```bash
bash triton_kd/dump_ir.sh        # TRITON_KERNEL_DUMP=1 → triton_kd/ir_dump/<hash>/*.{ttir,ttgir,llir,ptx}
```

What each stage shows:

- **TTIR** — hardware-independent tile ops: the `scf.for` over tiles, `tt.load` with the mask, the
  `tt.reduce` for max/sum.
- **TTGIR** — where the `BLOCK` tile gets a layout: the `#ttg.blocked<{sizePerThread = [...],
  threadsPerWarp = [32], warpsPerCTA = [...]}>` encoding records how tile elements map to threads and warps.
- **LLVM IR** — the reductions become warp shuffles (`llvm.nvvm.shfl.sync.bfly`) plus shared memory
  across warps.
- **PTX** — vectorized global loads (`ld.global.v4.b32`) and the `shfl.sync.bfly` butterfly.

<!-- Add short excerpts from triton_kd/ir_dump/ for each stage after running dump_ir.sh on a GPU. -->

Intel maintains a Triton backend for its GPUs (`intel-xpu-backend-for-triton`) that shares these
front-end stages (TTIR/TTGIR) and lowers to Intel hardware (via SPIR-V) instead of PTX.

## Bridge into training

`python train.py --model resnet18 --teacher r50_ce --seed 0 --fused-kd --name r18_kd_fused` swaps
`kd_loss` for the fused loss; `python plots.py --curves r18_kd r18_kd_fused` overlays the validation
curves. At 100 classes the loss is a negligible part of the step, so no speedup is expected. This
shows that the kernel works inside real mixed-precision training, while the large-vocabulary benchmarks
show where it pays off.
