"""INT8 quantization with the PT2E flow in torchao (X86InductorQuantizer).

The old torch.ao.quantization (eager/FX) APIs are deprecated, so everything here goes through
torch.export -> prepare_pt2e / prepare_qat_pt2e -> convert_pt2e, then lowers to oneDNN INT8
kernels with torch.compile (Inductor, freezing on).

Examples
  python quantize.py --mode ptq --base r18_kd --seed 0                 # -> r18_kd_ptq
  python quantize.py --mode ptq --base r18_kd_prune50 --seed 0         # -> r18_kd_prune50_ptq
  python quantize.py --mode qat --base mbv2_kd --teacher r50_ce --seed 0 --epochs 5   # -> mbv2_kd_qat
"""
import argparse
import copy
import os
import time
import warnings

os.environ.setdefault("TORCHINDUCTOR_FREEZING", "1")

import torch
import torch.nn.functional as F
from torch.export import Dim

import torchao
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e
import torchao.quantization.pt2e.quantizer.x86_inductor_quantizer as xiq
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer

from common import (Paths, add_common_args, collect_logits, env_info, file_size_mb, load_checkpoint,
                    log_number, metrics_from_logits, read_result, save_logits, set_seed, update_result)
from data import get_data
from train import kd_loss, resolve_teacher


def export_graph(model, img_size, batch=100, dynamic=True):
    """torch.export with a symbolic batch dimension, so one graph serves batch 1, 32 and 100."""
    dev = next(model.parameters()).device
    ex = (torch.randn(batch, 3, img_size, img_size, device=dev),)
    shapes = ({0: Dim("batch", min=1, max=1024)},) if dynamic else None
    return torch.export.export(model, ex, dynamic_shapes=shapes).module()


def make_quantizer(is_qat=False):
    q = X86InductorQuantizer()
    q.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=is_qat))
    return q


def ptq(model, calib_loader, img_size, n_batches=10):
    """Post-training static INT8 quantization, calibrated on training images only."""
    model = model.eval().cpu()
    gm = export_graph(model, img_size, batch=calib_loader.batch_size)
    prepared = prepare_pt2e(gm, make_quantizer())  # folds BN into conv, inserts observers
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):  # TRAINING images, never test
            if i == n_batches:
                break
            prepared(x)
    return convert_pt2e(prepared)


def qat(model, teacher, data, epochs=5, lr=1e-3, device="cpu", T=4.0, alpha=0.9, log=print):
    """Quantization-aware fine-tuning with the same KD loss. Returns (int8 graph, history)."""
    pt2e = torchao.quantization.pt2e
    model = model.to(device).train()
    gm = export_graph(model, data.img_size, batch=data.train.batch_size)
    prepared = prepare_qat_pt2e(gm, make_quantizer(is_qat=True))
    teacher = teacher.to(device).eval() if teacher is not None else None
    opt = torch.optim.SGD(prepared.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(data.train))
    history, best_acc, best_state = [], -1.0, None
    for ep in range(epochs):
        pt2e.move_exported_model_to_train(prepared)  # exported graphs don't support .train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            logits = prepared(x)
            loss = F.cross_entropy(logits, y)
            if teacher is not None:
                with torch.no_grad():
                    t_logits = teacher(x)
                loss = (1 - alpha) * loss + alpha * kd_loss(logits, t_logits, T)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        pt2e.move_exported_model_to_eval(prepared)  # not .eval()
        vl, vy = collect_logits(prepared, data.val, device)
        row = {"epoch": ep + 1, "train_loss": tot / n, "val_acc": metrics_from_logits(vl, vy)["top1"],
               "time_s": time.time() - t0}
        history.append(row)
        log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if row["val_acc"] > best_acc:
            best_acc, best_state = row["val_acc"], copy.deepcopy(prepared.state_dict())
    prepared.load_state_dict(best_state)
    prepared = prepared.cpu()
    pt2e.move_exported_model_to_eval(prepared)
    return convert_pt2e(prepared), history


def lower(qgm, example):
    """Lower the q/dq graph to oneDNN INT8 kernels. Falls back to the eager reference graph
    (fp32 math around q/dq ops) when Inductor can't compile here, e.g. Windows without MSVC."""
    import torch._inductor.config as ind
    ind.freezing = True
    compiled = torch.compile(qgm)
    try:
        with torch.no_grad():
            compiled(example)
        return compiled, "inductor"
    except Exception as e:
        warnings.warn(f"torch.compile failed ({type(e).__name__}: {str(e)[:200]}); "
                      "evaluating the eager reference graph instead")
        torch._dynamo.reset()
        return qgm, "eager-reference"


def strip_unused_tensors(gm):
    """convert_pt2e leaves the original FP32 weights attached (unused) next to the folded INT8 ones;
    re-exporting would serialize both and make the 'INT8' file bigger than the FP32 one."""
    used = {n.target for n in gm.graph.nodes if n.op == "get_attr"}
    for name, _ in list(gm.named_parameters()) + list(gm.named_buffers()):
        if name in used:
            continue
        *path, attr = name.split(".")
        owner = gm.get_submodule(".".join(path)) if path else gm
        delattr(owner, attr)
    gm.delete_all_unused_submodules()
    return gm


def save_quantized(qgm, path, img_size):
    qgm = strip_unused_tensors(qgm)
    ex = (torch.randn(2, 3, img_size, img_size),)
    ep = torch.export.export(qgm, ex, dynamic_shapes=({0: Dim("batch", min=1, max=1024)},))
    torch.export.save(ep, str(path))


def load_quantized(path):
    return torch.export.load(str(path)).module()


def main():
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--mode", choices=["ptq", "qat"], required=True)
    p.add_argument("--base", required=True, help="FP32 config to quantize, e.g. r18_kd")
    p.add_argument("--calib-batches", type=int, default=10)
    p.add_argument("--teacher", default="r50_ce")
    p.add_argument("--teacher-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="QAT training device (PTQ always runs on CPU)")
    p.add_argument("--no-compile", action="store_true", help="evaluate the eager reference graph")
    args = p.parse_args()

    set_seed(args.seed)
    paths = Paths(args.out, args.dataset)
    config = f"{args.base}_{args.mode}"
    data = get_data(args.dataset, args.data_dir, args.batch_size, num_workers=args.num_workers,
                    seed=args.seed, subset=args.subset)
    model, ck = load_checkpoint(paths.checkpoint(args.base, args.seed))
    base_rec = read_result(paths.result(args.base, args.seed))

    t0 = time.time()
    record = {"config": config, "seed": args.seed, "dataset": args.dataset, "arch": ck["arch"],
              "training": base_rec.get("training"), "base": args.base, "precision": "int8",
              "compression": (base_rec.get("compression", "none") + "+" if base_rec.get("compression")
                              not in (None, "none") else "") + f"int8-{args.mode}",
              "macs": base_rec.get("macs"), "params": base_rec.get("params")}
    if args.mode == "ptq":
        qgm = ptq(model, data.calib, data.img_size, args.calib_batches)
        record["hparams"] = {"calib_batches": args.calib_batches, "calib_batch_size": data.calib.batch_size}
    else:
        teacher, _ = load_checkpoint(resolve_teacher(args.teacher, paths, args.teacher_seed))
        qgm, hist = qat(model, teacher, data, args.epochs, args.lr, args.device, args.T, args.alpha)
        record["history"] = hist
        record["hparams"] = {"epochs": args.epochs, "lr": args.lr, "T": args.T, "alpha": args.alpha,
                             "device": args.device}
    record["quantize_time_s"] = time.time() - t0

    qpath = paths.checkpoint(config, args.seed, ".pt2")
    save_quantized(qgm, qpath, data.img_size)
    example = torch.randn(data.val.batch_size, 3, data.img_size, data.img_size)
    runner, backend = (qgm, "eager-reference") if args.no_compile else lower(qgm, example)
    summary = {}
    for split, loader in (("val", data.val), ("test", data.test)):
        lg, lb = collect_logits(runner, loader, "cpu")
        save_logits(paths.logit_file(config, args.seed, split), lg, lb)
        summary[split] = metrics_from_logits(lg, lb)
    record.update({"val_acc": summary["val"]["top1"], "test_acc": summary["test"]["top1"],
                   "test_top5": summary["test"]["top5"], "test_nll": summary["test"]["nll"],
                   "int8_eval_backend": backend, "size_mb": file_size_mb(qpath),
                   "checkpoint": str(qpath), "env": env_info()})
    update_result(paths.result(config, args.seed), record)
    delta = summary["test"]["top1"] - base_rec.get("test_acc", float("nan"))
    log_number(args.out, f"{args.dataset} {config} seed{args.seed}: test top-1 {summary['test']['top1']:.2f}% "
                         f"({delta:+.2f} vs {args.base}, eval={backend})")


if __name__ == "__main__":
    main()
