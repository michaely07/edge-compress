"""Structured channel pruning with Torch-Pruning, then a KD fine-tune.

Torch-Pruning builds a dependency graph, so removing a channel also removes everything coupled to
it (the next conv's input channel, BN, and every layer tied through a residual add). The model
really gets smaller and faster. torch.nn.utils.prune only masks weights to zero, so the dense
kernels do the same work and nothing speeds up.

Example
  python prune.py --base r18_kd --ratio 0.5 --epochs 15 --teacher r50_ce --seed 0   # -> r18_kd_prune50
"""
import argparse
import time

import torch
import torch_pruning as tp

from common import Paths, add_common_args, load_checkpoint, read_result, set_seed
from data import get_data
from models import classifier_layer
from train import count_macs_params, finish_run, fit, resolve_teacher


def prune_channels(model, img_size, ratio=0.5):
    """Uniform L2-magnitude group pruning; the classifier is left intact."""
    model = model.eval().cpu()
    ex = torch.randn(1, 3, img_size, img_size)
    Importance = getattr(tp.importance, "GroupMagnitudeImportance", None) or tp.importance.MagnitudeImportance
    Pruner = getattr(tp.pruner, "BasePruner", None) or tp.pruner.MetaPruner  # name moved between releases
    pruner = Pruner(model, ex, importance=Importance(p=2), pruning_ratio=ratio,
                    ignored_layers=[classifier_layer(model)])
    pruner.step()
    return model


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--base", default="r18_kd")
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--teacher", default="r50_ce")
    p.add_argument("--teacher-seed", type=int, default=0)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    paths = Paths(args.out, args.dataset)
    config = f"{args.base}_prune{round(args.ratio * 100)}"
    data = get_data(args.dataset, args.data_dir, args.batch_size, num_workers=args.num_workers,
                    seed=args.seed, subset=args.subset)
    model, ck = load_checkpoint(paths.checkpoint(args.base, args.seed))
    macs0, params0 = count_macs_params(model, data.img_size)
    model = prune_channels(model, data.img_size, args.ratio)
    macs1, params1 = count_macs_params(model, data.img_size)
    print(f"MACs {macs0/1e6:.1f}M -> {macs1/1e6:.1f}M, params {params0/1e6:.2f}M -> {params1/1e6:.2f}M")

    teacher, _ = load_checkpoint(resolve_teacher(args.teacher, paths, args.teacher_seed))
    t0 = time.time()
    history, _ = fit(model, data.train, args.epochs, val_loader=data.val, teacher=teacher, T=args.T,
                     alpha=args.alpha, lr=args.lr, device=args.device)
    base = read_result(paths.result(args.base, args.seed))
    record = {
        "arch": ck["arch"], "training": base.get("training", "kd"), "base": args.base,
        "compression": f"prune{round(args.ratio * 100)}",
        "hparams": {"ratio": args.ratio, "epochs": args.epochs, "lr": args.lr, "T": args.T,
                    "alpha": args.alpha, "subset": args.subset},
        "pruning": {"macs_before": macs0, "params_before": params0, "macs_after": macs1,
                    "params_after": params1},
        "history": history, "train_time_s": time.time() - t0,
    }
    finish_run(model, data, paths, config, args.seed, record, args.device, args.out, whole_module=True)


if __name__ == "__main__":
    main()
