"""Correctness of the fused KD loss against the PyTorch reference (loss and gradient).

GPU:  pytest triton_kd -q
CPU:  TRITON_INTERPRET=1 pytest triton_kd -q      (interpreter; slower, no GPU needed)
"""
import os
import sys
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")
triton = pytest.importorskip("triton")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import kd_loss  # noqa: E402  (the reference from Project 1)
from triton_kd.kernel import fused_kd_loss  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
INTERP = os.environ.get("TRITON_INTERPRET") == "1"
SHAPES = [(1, 7), (64, 100), (32, 1000), (8, 32000), (4, 128256)]  # last two: LLM vocabularies
# fp32: rtol ~1e-4. fp16: the kernel upcasts to fp32 internally, so the loss stays tight, but the
# gradient is rounded to fp16 on store (~1e-3 relative), and the reference rounds differently.
TOL = {torch.float32: dict(rtol=1e-4, atol=1e-5), torch.float16: dict(rtol=2e-3, atol=2e-3)}


def run_both(s, t, T, version):
    s1 = s.detach().clone().requires_grad_(True)
    s2 = s.detach().clone().requires_grad_(True)
    ref = kd_loss(s1, t, T)
    ref.backward()
    out = fused_kd_loss(s2, t, T, version=version)
    out.backward()
    return ref, out, s1.grad, s2.grad


def check(s, t, T, version, tol=None):
    ref, out, g_ref, g_out = run_both(s, t, T, version)
    tol = tol or TOL[s.dtype]
    torch.testing.assert_close(out.float(), ref.float(), **tol)
    assert g_out.dtype == s.dtype
    torch.testing.assert_close(g_out.float(), g_ref.float(), **tol)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("T", [1.0, 4.0])
@pytest.mark.parametrize("B,V", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_matches_reference(B, V, T, dtype, version):
    if INTERP and B * V > 200_000 and dtype == torch.float16:
        pytest.skip("large fp16 cases are slow in the interpreter; covered on GPU")
    g = torch.Generator().manual_seed(B * 7 + V)
    s = (torch.randn(B, V, generator=g) * 3).to(DEV, dtype)
    t = (torch.randn(B, V, generator=g) * 3).to(DEV, dtype)
    check(s, t, T, version)


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("scale", [1e3, 1e4])
def test_large_logits_do_not_overflow(scale, version):
    # exp(1e4) overflows fp32; the online max must keep everything finite.
    g = torch.Generator().manual_seed(0)
    s = (torch.randn(16, 1000, generator=g) * scale).to(DEV)
    t = (torch.randn(16, 1000, generator=g) * scale).to(DEV)
    ref, out, g_ref, g_out = run_both(s, t, 4.0, version)
    assert torch.isfinite(out) and torch.isfinite(g_out).all()
    # The loss is a difference of terms of size ~scale/T: relative accuracy is what fp32 allows.
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-6 * scale)
    torch.testing.assert_close(g_out, g_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("version", [1, 2])
def test_all_equal_row(version):
    s = torch.randn(4, 300).to(DEV)
    t = torch.randn(4, 300).to(DEV)
    s[1] = 2.5   # uniform student row
    t[2] = -1.0  # uniform teacher row
    check(s, t, 2.0, version)


@pytest.mark.parametrize("version", [1, 2])
def test_identical_inputs_give_zero(version):
    s = torch.randn(8, 500).to(DEV)
    ref, out, _, g = run_both(s, s.clone(), 3.0, version)
    assert abs(out.item()) < 1e-4
    assert g.abs().max().item() < 1e-6


@pytest.mark.parametrize("version", [1, 2])
def test_non_contiguous_input(version):
    base_s = torch.randn(1000, 64).to(DEV)
    base_t = torch.randn(1000, 64).to(DEV)
    s, t = base_s.t()[:, :777], base_t.t()[:, :777]  # transposed slice: stride(1) != 1
    assert not s.is_contiguous()
    check(s, t, 4.0, version)


def test_v1_and_v2_agree():
    g = torch.Generator().manual_seed(3)
    s = torch.randn(32, 5000, generator=g).to(DEV)
    t = torch.randn(32, 5000, generator=g).to(DEV)
    a = fused_kd_loss(s, t, 4.0, version=1)
    b = fused_kd_loss(s, t, 4.0, version=2)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_upstream_gradient_is_applied():
    s = torch.randn(8, 100, device=DEV, requires_grad=True)
    t = torch.randn(8, 100, device=DEV)
    (fused_kd_loss(s, t, 4.0) * 3.0).backward()
    s2 = s.detach().clone().requires_grad_(True)
    (kd_loss(s2, t, 4.0) * 3.0).backward()
    torch.testing.assert_close(s.grad, s2.grad, rtol=1e-4, atol=1e-6)
