"""Fast CPU tests: models, KD loss, calibration, statistics, Pareto frontier."""
import math

import pytest
import torch
import torch.nn.functional as F

from calib import ece, fit_temperature
from models import build_model
from plots import pareto_frontier
from report import mean_ci, paired_ci
from train import kd_loss


@pytest.mark.parametrize("arch", ["resnet18", "resnet50", "mobilenetv2"])
@pytest.mark.parametrize("img", [32, 64])
def test_models_shapes(arch, img):
    m = build_model(arch, 100).eval()
    with torch.no_grad():
        assert m(torch.randn(2, 3, img, img)).shape == (2, 100)


def test_resnet_keeps_spatial_resolution():
    # The CIFAR stem (3x3 stride 1, no maxpool) keeps 32x32 into layer1.
    m = build_model("resnet18").eval()
    x = m.relu(m.bn1(m.conv1(torch.randn(1, 3, 32, 32))))
    assert m.layer1(m.maxpool(x)).shape[-1] == 32


def test_resnet_state_dict_matches_torchvision():
    import torchvision.models as tvm
    for depth, ctor in ((18, tvm.resnet18), (50, tvm.resnet50)):
        ours = build_model(f"resnet{depth}").state_dict()
        ref = ctor(num_classes=100).state_dict()
        assert ours.keys() == ref.keys()


def test_kd_loss_matches_manual_kl():
    s, t, T = torch.randn(8, 10), torch.randn(8, 10), 3.0
    p, q = F.softmax(s / T, -1), F.softmax(t / T, -1)
    manual = (q * (q.log() - p.log())).sum(-1).mean() * T * T
    assert torch.allclose(kd_loss(s, t, T), manual, atol=1e-6)


def test_kd_gradient_is_scaled_softmax_difference():
    # dL/ds = T/B (p - q): the T^2 factor restores the 1/T^2 shrinkage from softening.
    B, T = 4, 4.0
    s = torch.randn(B, 10, requires_grad=True)
    t = torch.randn(B, 10)
    kd_loss(s, t, T).backward()
    expected = T / B * (F.softmax(s / T, -1) - F.softmax(t / T, -1))
    assert torch.allclose(s.grad, expected.detach(), atol=1e-6)


def test_ece_zero_for_perfect_calibration_and_positive_for_overconfident():
    labels = torch.zeros(1000, dtype=torch.long)
    probs = torch.zeros(1000, 2)
    probs[:, 0] = 1.0
    assert ece(probs, labels) == pytest.approx(0.0, abs=1e-6)
    probs = torch.tensor([[0.99, 0.01]] * 1000)
    labels = torch.tensor([0] * 600 + [1] * 400)  # 60% accurate at 99% confidence
    assert ece(probs, labels) == pytest.approx(0.39, abs=1e-3)


def test_temperature_scaling_recovers_known_temperature():
    torch.manual_seed(0)
    true_logits = torch.randn(5000, 10) * 2
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    T = fit_temperature(true_logits * 3.0, labels)  # 3x overconfident
    assert T == pytest.approx(3.0, rel=0.1)


def test_mean_ci_uses_t_distribution():
    m, h, n = mean_ci([70.0, 71.0, 72.0])
    assert (m, n) == (71.0, 3)
    assert h == pytest.approx(4.303 * 1.0 / math.sqrt(3), rel=1e-6)
    assert mean_ci([5.0])[1] is None


def test_paired_ci_uses_shared_seeds():
    r = paired_ci({0: 72.0, 1: 73.0, 2: 71.5}, {0: 70.0, 1: 71.5, 2: 70.0, 3: 99.0})
    assert r["seeds"] == [0, 1, 2]
    assert r["diffs"] == [2.0, 1.5, 1.5]
    assert r["mean"] == pytest.approx(5 / 3)


def test_pareto_frontier():
    pts = [(1.0, 60, "a"), (2.0, 70, "b"), (3.0, 65, "c"), (4.0, 75, "d"), (0.5, 50, "e")]
    assert [p[2] for p in pareto_frontier(pts)] == ["e", "a", "b", "d"]
