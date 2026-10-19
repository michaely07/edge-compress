"""Benchmark the fused KD loss against PyTorch eager and torch.compile (GPU required).

Per vocabulary size V (B rows fixed), for forward and forward+backward separately:
  time (ms, triton.testing.do_bench), effective bandwidth (GB/s), peak memory (MB).

Effective bandwidth uses the *minimum* bytes any implementation must move, so the numbers are
comparable across implementations:  fwd = 2*B*V*elt (read s and t once),
fwd+bwd = 5*B*V*elt (fwd reads, bwd reads s and t again and writes the gradient).
v1's actual forward traffic is 2x the minimum (each input read twice); that is reported as
`fwd_actual_gbps` for v1.

  python triton_kd/bench.py                       # B=2048, V in {100, 1k, 8k, 32k, 128k}, fp32
  python triton_kd/bench.py --dtype fp16 --B 4096
  python triton_kd/bench.py --update-readme       # also fill the results block in triton_kd/README.md
  TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./ir_dump python triton_kd/bench.py --Vs 32768 --impls fused_v2
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import kd_loss  # noqa: E402
from triton_kd.kernel import fused_kd_loss  # noqa: E402

PEAK_GBPS = {"T4": 320, "L4": 300, "A10": 600, "A100-SXM4-40GB": 1555, "A100-SXM4-80GB": 2039,
             "A100-PCIE-40GB": 1555, "A100 80GB PCIe": 1935, "H100": 3350, "RTX 4090": 1008, "RTX 3090": 936}


def peak_bw(name):
    for k, v in PEAK_GBPS.items():
        if k in name:
            return v
    return None


def make_impls():
    torch._dynamo.config.cache_size_limit = 64
    return {
        "eager": kd_loss,
        "compile": torch.compile(kd_loss, dynamic=False),
        "fused_v1": lambda s, t, T: fused_kd_loss(s, t, T, version=1),
        "fused_v2": lambda s, t, T: fused_kd_loss(s, t, T, version=2),
    }


def measure(fn, B, V, dtype, T=4.0):
    elt = torch.tensor([], dtype=dtype).element_size()
    s = torch.randn(B, V, device="cuda", dtype=dtype)
    t = torch.randn(B, V, device="cuda", dtype=dtype)
    sg = s.clone().requires_grad_(True)
    fn(s, t, T)  # compile / autotune outside the timed region
    fwd = triton.testing.do_bench(lambda: fn(s, t, T))

    def step():
        fn(sg, t, T).backward()
    step()
    fb = triton.testing.do_bench(step, grad_to_none=[sg])

    torch.cuda.synchronize()
    sg.grad = None
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    step()
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    return {"fwd_ms": fwd, "fwd_bwd_ms": fb, "fwd_gbps": 2 * B * V * elt / fwd / 1e6,
            "fwd_bwd_gbps": 5 * B * V * elt / fb / 1e6, "peak_mem_mb": peak,
            "inputs_mb": 2 * B * V * elt / 2**20}


def plot(rows, out, gpu, peak):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"eager": "#2a78d6", "compile": "#eb6834", "fused_v1": "#1baf7a", "fused_v2": "#4a3aa7"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor="#fcfcfb")
    for ax, key, label in ((axes[0], "fwd_gbps", "Forward effective bandwidth (GB/s)"),
                           (axes[1], "peak_mem_mb", "Peak extra memory, fwd+bwd (MB)")):
        ax.set_facecolor("#fcfcfb")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(True, color="#e4e3df")
        ax.set_axisbelow(True)
        for impl, col in colors.items():
            pts = sorted((r["V"], r[key]) for r in rows if r["impl"] == impl and key in r)
            if pts:
                ax.plot(*zip(*pts), marker="o", ms=6, lw=2, color=col, label=impl)
        if key == "fwd_gbps" and peak:
            ax.axhline(peak, color="#52514e", ls="--", lw=1)
            ax.text(min(r["V"] for r in rows), peak * 1.02, f"{gpu} spec peak {peak} GB/s", fontsize=8, color="#52514e")
        ax.set_xscale("log")
        ax.set_yscale("log" if key == "peak_mem_mb" else "linear")
        ax.set_xlabel("Vocabulary size V (log scale)")
        ax.set_title(label, loc="left", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--B", type=int, default=2048)
    p.add_argument("--Vs", type=int, nargs="+", default=[100, 1000, 8192, 32768, 131072])
    p.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    p.add_argument("--impls", nargs="+", default=["eager", "compile", "fused_v1", "fused_v2"])
    p.add_argument("--out", default=str(Path(__file__).parent / "results"))
    p.add_argument("--update-readme", action="store_true", help="write the table + plot into triton_kd/README.md")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU (correctness tests run on CPU with TRITON_INTERPRET=1)")

    dtype = {"fp32": torch.float32, "fp16": torch.float16}[args.dtype]
    gpu = torch.cuda.get_device_name(0)
    peak = peak_bw(gpu)
    impls = {k: v for k, v in make_impls().items() if k in args.impls}
    rows = []
    for V in args.Vs:
        for name, fn in impls.items():
            try:
                r = measure(fn, args.B, V, dtype)
            except torch.cuda.OutOfMemoryError:
                r = {"oom": True}
            torch.cuda.empty_cache()
            r.update({"impl": name, "V": V, "B": args.B, "dtype": args.dtype})
            if name == "fused_v1" and "fwd_gbps" in r:
                r["fwd_actual_gbps"] = 2 * r["fwd_gbps"]
            if peak and "fwd_gbps" in r:
                r["fwd_pct_peak"] = 100 * r.get("fwd_actual_gbps", r["fwd_gbps"]) / peak
            rows.append(r)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{gpu.replace(' ', '_')}_{args.dtype}_B{args.B}"
    (out / f"bench_{tag}.json").write_text(json.dumps(
        {"gpu": gpu, "peak_gbps": peak, "torch": torch.__version__, "triton": triton.__version__, "rows": rows}, indent=2))
    plot([r for r in rows if "oom" not in r], out / f"bench_{tag}.png", gpu, peak)

    lines = [f"GPU {gpu}, B={args.B}, {args.dtype}", "",
             "| V | impl | fwd ms | fwd+bwd ms | fwd GB/s (min bytes) | fwd+bwd GB/s | peak MB |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "oom" in r:
            lines.append(f"| {r['V']} | {r['impl']} | OOM | | | | |")
        else:
            lines.append(f"| {r['V']} | {r['impl']} | {r['fwd_ms']:.3f} | {r['fwd_bwd_ms']:.3f} | {r['fwd_gbps']:.0f} | "
                         f"{r['fwd_bwd_gbps']:.0f} | {r['peak_mem_mb']:.0f} |")
    (out / f"bench_{tag}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.update_readme:
        from report import inject
        readme = Path(__file__).parent / "README.md"
        fig = (out / f"bench_{tag}.png").resolve().relative_to(readme.parent.resolve()).as_posix()
        inject(readme, "triton", "\n".join(lines) + f"\n\n![Bandwidth and peak memory vs V]({fig})")


if __name__ == "__main__":
    main()
