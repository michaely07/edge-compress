"""Calibration: ECE, reliability-diagram data, and temperature scaling.

A single temperature is fit on the validation split (LBFGS on NLL) and applied to the test logits.
Works from the logits saved by train/prune/quantize, so no model is re-run.

  python calib.py --dataset cifar100
"""
import argparse
import re

import torch
import torch.nn.functional as F

from common import Paths, log_number, update_result


def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    acc = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    e = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - acc[m].mean()).abs()
    return e.item()


def reliability(probs, labels, n_bins=15):
    """Per-bin confidence, accuracy and count, for the reliability diagram."""
    conf, pred = probs.max(1)
    acc = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        n = int(m.sum())
        bins.append({"lo": lo.item(), "hi": hi.item(), "count": n,
                     "conf": conf[m].mean().item() if n else None,
                     "acc": acc[m].mean().item() if n else None})
    return bins


def fit_temperature(logits, labels, iters=100):
    """Minimize validation NLL over one scalar. Optimizes log T so T stays positive."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def calibrate(val, test, n_bins=15):
    vl, vy = val["logits"].float(), val["labels"]
    tl, ty = test["logits"].float(), test["labels"]
    T = fit_temperature(vl, vy)
    p0, p1 = tl.softmax(1), (tl / T).softmax(1)
    return {
        "temperature": T,
        "ece": ece(p0, ty, n_bins), "ece_ts": ece(p1, ty, n_bins),
        "nll": F.cross_entropy(tl, ty).item(), "nll_ts": F.cross_entropy(tl / T, ty).item(),
        "n_bins": n_bins,
        "reliability": reliability(p0, ty, n_bins), "reliability_ts": reliability(p1, ty, n_bins),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "drone", "fake"])
    p.add_argument("--out", default=".")
    p.add_argument("--bins", type=int, default=15)
    args = p.parse_args()
    paths = Paths(args.out, args.dataset)

    for f in sorted(paths.logits.glob("*_test.pt")):
        m = re.match(r"(.+)_seed(\d+)_test$", f.stem)
        config, seed = m.group(1), int(m.group(2))
        vf = paths.logit_file(config, seed, "val")
        if not vf.exists():
            continue
        c = calibrate(torch.load(vf), torch.load(f), args.bins)
        update_result(paths.result(config, seed), {"calibration": c})
        log_number(args.out, f"{args.dataset} {config} seed{seed}: ECE {c['ece']*100:.2f}% -> "
                             f"{c['ece_ts']*100:.2f}% after temperature scaling (T={c['temperature']:.3f})")


if __name__ == "__main__":
    main()
