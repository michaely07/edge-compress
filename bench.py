"""CPU latency and throughput harness.

All configs are measured on one fixed machine (a laptop CPU as the edge proxy; shared cloud CPUs
are too noisy). Threads are pinned and the CPU model is recorded. FP32 baselines are compiled with
torch.compile too (freezing on), so the comparison is FP32 vs INT8, not eager vs compiled.

Latency depends on the graph, not the weights, so each config is measured once (on --seed) and
the numbers are written into every seed's JSON with `measured_on_seed`.

Examples
  python bench.py --dataset cifar100                       # every config that has a checkpoint
  python bench.py --configs r18_kd r18_kd_ptq --profile    # + torch.profiler op tables
  python bench.py --configs r18_kd mbv2_kd --onnx          # + ONNX Runtime latency for FP32 models
"""
import argparse
import os
import statistics
import time
import warnings

os.environ.setdefault("TORCHINDUCTOR_FREEZING", "1")

import torch

from common import Paths, cpu_model, env_info, file_size_mb, load_checkpoint, log_number, update_result


def bench(fn, x, warmup=20, iters=200):
    with torch.no_grad():
        for _ in range(warmup):
            fn(x)  # first calls trigger torch.compile
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"median_ms": statistics.median(ts), "p90_ms": ts[int(0.9 * len(ts))]}


def load_any(paths, config, seed):
    """FP32/pruned checkpoints are .pt; INT8 graphs are exported .pt2 programs."""
    pt2 = paths.checkpoint(config, seed, ".pt2")
    if pt2.exists():
        from quantize import load_quantized
        return load_quantized(pt2), "int8", pt2
    pt = paths.checkpoint(config, seed)
    model, _ = load_checkpoint(pt)
    return model, "fp32", pt


def compiled(model, x, use_compile=True, allow_eager=False):
    """Fresh, shape-specialized compile per batch size. Without Inductor, INT8 graphs run as eager
    q/dq reference code that is SLOWER than FP32, so falling back is refused unless allowed."""
    torch._dynamo.reset()
    if not use_compile:
        return model, False
    import torch._inductor.config as ind
    ind.freezing = True
    fn = torch.compile(model, dynamic=False)
    try:
        with torch.no_grad():
            fn(x)
        return fn, True
    except Exception as e:
        msg = (f"torch.compile failed ({type(e).__name__}: {str(e)[:200]}). On Windows install the MSVC "
               "build tools or run under WSL2/Linux.")
        if not allow_eager:
            raise SystemExit(msg + " Refusing to record eager numbers (pass --allow-eager to override).")
        warnings.warn(msg + " Falling back to eager.")
        torch._dynamo.reset()
        return model, False


def profile_ops(fn, x, out_file, iters=50):
    from torch.profiler import ProfilerActivity, profile
    with torch.no_grad(), profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(iters):
            fn(x)
    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
    out_file.write_text(table)
    return table


def onnx_latency(model, img, path, threads, iters=200):
    import onnxruntime as ort
    x = torch.randn(1, 3, img, img)
    torch.onnx.export(model, (x,), str(path), input_names=["x"], output_names=["logits"])
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    xn = x.numpy()
    r = bench(lambda a: sess.run(None, {"x": a}), xn, iters=iters)
    r["size_mb"] = file_size_mb(path)
    return r


def configs_with_checkpoints(paths, seed):
    out = set()
    for f in paths.ckpt.glob(f"*_seed{seed}.pt*"):
        if f.name.endswith(".last.pt"):
            continue
        out.add(f.name.rsplit(f"_seed{seed}", 1)[0])
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "drone", "fake"])
    p.add_argument("--out", default=".")
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--seed", type=int, default=0, help="checkpoint seed to measure")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--allow-eager", action="store_true", help="record eager numbers if torch.compile fails")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--onnx", action="store_true")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    paths = Paths(args.out, args.dataset)
    img = 64 if args.dataset == "drone" else 32
    cpu = cpu_model()
    configs = args.configs or configs_with_checkpoints(paths, args.seed)
    print(f"CPU: {cpu} | threads={args.threads} | configs={configs}")
    (paths.results / "profiles").mkdir(exist_ok=True)

    for config in configs:
        model, precision, ckpt = load_any(paths, config, args.seed)
        if precision == "fp32":
            model.eval()
        lat = {"cpu": cpu, "threads": args.threads, "measured_on_seed": args.seed, "torch": torch.__version__}

        x1 = torch.randn(1, 3, img, img)
        fn, ok = compiled(model, x1, not args.no_compile, args.allow_eager)
        b1 = bench(fn, x1, args.warmup, args.iters)
        lat.update({"compiled": ok, "b1_median_ms": b1["median_ms"], "b1_p90_ms": b1["p90_ms"]})
        if args.profile:
            profile_ops(fn, x1, paths.results / "profiles" / f"{config}_b1.txt")

        x32 = torch.randn(32, 3, img, img)
        fn, ok32 = compiled(model, x32, not args.no_compile, args.allow_eager)
        b32 = bench(fn, x32, max(3, args.warmup // 4), max(20, args.iters // 4))
        lat.update({"b32_median_ms": b32["median_ms"], "b32_throughput_ips": 32e3 / b32["median_ms"],
                    "compiled_b32": ok32})
        if args.profile:
            profile_ops(fn, x32, paths.results / "profiles" / f"{config}_b32.txt", iters=10)

        if args.onnx and precision == "fp32":
            try:
                lat["ort"] = onnx_latency(model, img, paths.ckpt / f"{config}_seed{args.seed}.onnx", args.threads)
            except Exception as e:
                warnings.warn(f"ONNX export/run failed for {config}: {e}")

        seeds = sorted(int(f.stem.rsplit("_seed", 1)[1]) for f in paths.results.glob(f"{config}_seed*.json")
                       if f.stem.rsplit("_seed", 1)[0] == config)
        for s in seeds or [args.seed]:
            update_result(paths.result(config, s), {"latency": lat, "bench_env": env_info()})
        log_number(args.out, f"{args.dataset} {config} [{precision}] on {cpu} ({args.threads} thr, "
                             f"compiled={ok}): b1 median {b1['median_ms']:.3f} ms / p90 {b1['p90_ms']:.3f} ms, "
                             f"b32 {lat['b32_throughput_ips']:.0f} img/s")


if __name__ == "__main__":
    main()
