"""Compression paths on CPU with tiny inputs: pruning, PTQ, QAT, save/load of INT8 graphs."""
from types import SimpleNamespace

import pytest
import torch

import quantize as Q
from models import build_model
from prune import prune_channels
from train import count_macs_params


class Loader(list):
    def __init__(self, items, batch_size):
        super().__init__(items)
        self.batch_size = batch_size


def batches(n, bs=8, img=32, classes=100):
    g = torch.Generator().manual_seed(0)
    return Loader([(torch.randn(bs, 3, img, img, generator=g), torch.randint(0, classes, (bs,), generator=g))
                   for _ in range(n)], bs)


def warm_bn(model, img=32):
    """Random-init BN stats make activations ~1e-10; give them realistic statistics."""
    model.train()
    with torch.no_grad():
        for _ in range(5):
            model(torch.randn(32, 3, img, img))
    return model.eval()


def test_structured_pruning_halves_channels_and_runs():
    m = build_model("resnet18")
    macs0, params0 = count_macs_params(m, 32)
    m = prune_channels(m, 32, 0.5)
    macs1, params1 = count_macs_params(m, 32)
    assert m.conv1.out_channels == 32
    assert m.fc.out_features == 100
    assert params1 < 0.35 * params0 and macs1 < 0.35 * macs0  # ~quadratic in width
    with torch.no_grad():
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, 100)


@pytest.mark.parametrize("arch", ["resnet18", "mobilenetv2"])
def test_ptq_close_to_fp32_and_dynamic_batch(arch, tmp_path):
    m = warm_bn(build_model(arch))
    q = Q.ptq(m, batches(3), 32, n_batches=3)
    x = torch.randn(16, 3, 32, 32)
    with torch.no_grad():
        ref, out = m(x), q(x)
        assert ((ref - out).norm() / ref.norm()) < 0.1
        for b in (1, 5):
            assert q(torch.randn(b, 3, 32, 32)).shape == (b, 100)
    Q.save_quantized(q, tmp_path / "q.pt2", 32)
    q2 = Q.load_quantized(tmp_path / "q.pt2")
    with torch.no_grad():
        assert torch.allclose(q2(x), out)


def test_ptq_after_pruning():
    m = warm_bn(prune_channels(build_model("resnet18"), 32, 0.5))
    q = Q.ptq(m, batches(2), 32, n_batches=2)
    with torch.no_grad():
        assert q(torch.randn(3, 3, 32, 32)).shape == (3, 100)


def test_qat_runs_and_converts():
    student = warm_bn(build_model("mobilenetv2", 10))
    teacher = warm_bn(build_model("resnet18", 10))
    data = SimpleNamespace(train=batches(2, classes=10), val=batches(1, classes=10), img_size=32)
    q, hist = Q.qat(student, teacher, data, epochs=1, lr=1e-3, device="cpu", log=lambda *_: None)
    assert len(hist) == 1 and "val_acc" in hist[0]
    with torch.no_grad():
        assert q(torch.randn(4, 3, 32, 32)).shape == (4, 10)
