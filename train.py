"""Training loop: cross-entropy or knowledge distillation, AMP, seeds, resumable.

Examples
  python train.py --model resnet50 --epochs 60                       # teacher  -> r50_ce
  python train.py --model resnet18 --epochs 40 --seed 1              # baseline -> r18_ce
  python train.py --model resnet18 --epochs 40 --teacher r50_ce      # KD       -> r18_kd
  python train.py --model resnet18 --epochs 40 --teacher r50_ce --fused-kd --name r18_kd_fused
"""
import argparse
import copy
import time

import torch
import torch.nn.functional as F

from common import (ARCH_ABBR, Paths, add_common_args, collect_logits, env_info, load_checkpoint,
                    log_number, metrics_from_logits, save_checkpoint, save_logits, set_seed,
                    state_dict_size_mb, update_result)
from data import get_data
from models import build_model


def kd_loss(s_logits, t_logits, T):
    # KL(teacher || student) on temperature-softened distributions, scaled by T^2.
    # Softening scales the soft-target gradients by roughly 1/T^2; the factor keeps them comparable
    # to the cross-entropy term as T changes.
    return F.kl_div(F.log_softmax(s_logits.float() / T, -1),
                    F.log_softmax(t_logits.float() / T, -1),
                    reduction="batchmean", log_target=True) * T * T


def count_macs_params(model, img_size):
    try:
        import torch_pruning as tp
        was_training = model.training
        model.eval()
        p = next(model.parameters())
        macs, params = tp.utils.count_ops_and_params(model, torch.randn(1, 3, img_size, img_size,
                                                                        device=p.device))
        model.train(was_training)
        return int(macs), int(params)
    except Exception as e:  # torch-pruning missing or API drift; params are still exact
        print(f"[warn] MAC count failed: {e}")
        return None, sum(p.numel() for p in model.parameters())


def fit(model, loader, epochs, *, val_loader=None, teacher=None, T=4.0, alpha=0.9, lr=0.1,
        wd=5e-4, device="cuda", kd_fn=kd_loss, state_path=None, log=print):
    """Train with SGD + OneCycle. Returns (history, best_val_acc); loads the best-val weights."""
    if len(loader) == 0:
        raise ValueError("training set is smaller than one batch (drop_last=True); lower --batch-size")
    amp = device == "cuda"
    model.to(device)
    if amp:
        model.to(memory_format=torch.channels_last)
    opt =torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if teacher is not None:
        teacher.to(device).eval()
        if amp:
            teacher.to(memory_format=torch.channels_last)

    history, best_acc, best_state, start = [], -1.0, None, 0
    if state_path is not None and state_path.exists():  # resume an interrupted run
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        history, best_acc, best_state, start = st["history"], st["best_acc"], st["best_state"], st["epoch"]
        log(f"resumed from epoch {start}")

    for ep in range(start, epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if amp:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(x)
                    loss = (1 - alpha) * loss + alpha * kd_fn(logits, t_logits, T)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        row = {"epoch": ep + 1, "train_loss": tot / n, "time_s": time.time() - t0,
               "lr": sched.get_last_lr()[0]}
        if val_loader is not None:
            model.eval()
            vl, vy = collect_logits(model, val_loader, device, amp)
            row["val_acc"] = metrics_from_logits(vl, vy)["top1"]
            if row["val_acc"] > best_acc:
                best_acc = row["val_acc"]
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        history.append(row)
        log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if state_path is not None:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "history": history, "best_acc": best_acc,
                        "best_state": best_state, "epoch": ep + 1}, state_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(memory_format=torch.contiguous_format)
    return history, best_acc


def finish_run(model, data, paths, config, seed, record, device, out_root, whole_module=False):
    """Final val/test evaluation, logits, checkpoint and JSON row for a trained FP32 model."""
    model.to(device).eval()
    amp = device == "cuda"
    summary = {}
    for split, loader in (("val", data.val), ("test", data.test)):
        lg, lb = collect_logits(model, loader, device, amp)
        save_logits(paths.logit_file(config, seed, split), lg, lb)
        summary[split] = metrics_from_logits(lg, lb)
    model.cpu()
    macs, params = count_macs_params(model, data.img_size)
    ckpt = paths.checkpoint(config, seed)
    save_checkpoint(ckpt, model, record["arch"], data.num_classes, extra={"config": config},
                    whole_module=whole_module)
    record.update({
        "config": config, "seed": seed, "dataset": paths.dataset, "precision": "fp32",
        "val_acc": summary["val"]["top1"], "test_acc": summary["test"]["top1"],
        "test_top5": summary["test"]["top5"], "test_nll": summary["test"]["nll"],
        "macs": macs, "params": params,
        "size_mb": state_dict_size_mb(model, str(ckpt) + ".sizetmp"),
        "checkpoint": str(ckpt), "env": env_info(),
    })
    update_result(paths.result(config, seed), record)
    log_number(out_root, f"{paths.dataset} {config} seed{seed}: test top-1 {summary['test']['top1']:.2f}% "
                         f"(val {summary['val']['top1']:.2f}%, NLL {summary['test']['nll']:.3f})")
    return summary


def resolve_teacher(spec, paths, teacher_seed):
    """`--teacher r50_ce` resolves to that config's checkpoint; a path is used as is."""
    from pathlib import Path
    p = Path(spec)
    return p if p.suffix == ".pt" and p.exists() else paths.checkpoint(spec, teacher_seed)


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--model", required=True, choices=list(ARCH_ABBR))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--teacher", default=None, help="teacher config name (e.g. r50_ce) or .pt path")
    p.add_argument("--teacher-seed", type=int, default=0)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--fused-kd", action="store_true", help="use the Triton fused KD loss")
    p.add_argument("--name", default=None, help="config name (default: <arch>_<ce|kd>)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--fresh", action="store_true", help="ignore a resumable .last.pt state")
    args = p.parse_args()

    training = "kd" if args.teacher else "ce"
    config = args.name or f"{ARCH_ABBR[args.model]}_{training}"
    paths = Paths(args.out, args.dataset)
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    data = get_data(args.dataset, args.data_dir, args.batch_size, num_workers=args.num_workers,
                    seed=args.seed, subset=args.subset)
    model = build_model(args.model, data.num_classes)

    teacher, kd_fn = None, kd_loss
    if args.teacher:
        tpath = resolve_teacher(args.teacher, paths, args.teacher_seed)
        teacher, _ = load_checkpoint(tpath)
        print(f"teacher: {tpath}")
        if args.fused_kd:
            from triton_kd.kernel import fused_kd_loss
            kd_fn = fused_kd_loss

    state_path = paths.checkpoint(config, args.seed, ".last.pt")
    if args.fresh and state_path.exists():
        state_path.unlink()
    t0 = time.time()
    history, best_val = fit(model, data.train, args.epochs, val_loader=data.val, teacher=teacher,
                            T=args.T, alpha=args.alpha, lr=args.lr, wd=args.wd, device=args.device,
                            kd_fn=kd_fn, state_path=state_path)
    record = {
        "arch": args.model, "training": training, "compression": "none",
        "teacher": str(resolve_teacher(args.teacher, paths, args.teacher_seed)) if args.teacher else None,
        "hparams": {"epochs": args.epochs, "lr": args.lr, "wd": args.wd, "batch_size": args.batch_size,
                    "T": args.T, "alpha": args.alpha, "fused_kd": args.fused_kd, "subset": args.subset},
        "history": history, "train_time_s": time.time() - t0, "device": args.device,
    }
    finish_run(model, data, paths, config, args.seed, record, args.device, args.out)
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
