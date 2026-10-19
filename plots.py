"""README figures, generated from results/<dataset>/*.json.

1. Pareto plot: CPU latency at batch 1 (log x) vs top-1 (y), seed CIs as error bars, frontier highlighted
2. Reliability diagrams: FP32 vs INT8 for one student
3. Drone confusion matrix and accuracy by object size
4. Validation curves (e.g. reference vs fused Triton KD loss on the same seed)

  python plots.py --dataset cifar100             # (report.py --update-readme also runs this)
  python plots.py --dataset drone
  python plots.py --dataset cifar100 --curves r18_kd r18_kd_fused --curve-seed 0
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import Paths, load_results
from report import aggregate, by_config, config_order, mean_ci, SIZE_BUCKETS

# Colorblind-safe categorical palette (the three slots stay distinguishable under common color-vision
# deficiencies); architecture is also never encoded by color alone (see the legend in plot_pareto).
ARCH_COLOR = {"resnet50": "#2a78d6", "resnet18": "#eb6834", "mobilenetv2": "#1baf7a"}
ARCH_LABEL = {"resnet50": "ResNet-50", "resnet18": "ResNet-18", "mobilenetv2": "MobileNetV2"}
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_BAR = "#2a78d6"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def marker_for(compression):
    c = compression or "none"
    if "prune" in c and "int8" in c:
        return "P"
    if "prune" in c:
        return "s"
    if "qat" in c:
        return "D"
    if "ptq" in c:
        return "^"
    return "o"


MARKER_LABEL = {"o": "FP32", "^": "INT8 PTQ", "D": "INT8 QAT", "s": "Pruned 50% (FP32)", "P": "Pruned + INT8 PTQ"}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def pareto_frontier(points):
    """points: (latency, acc, name). Non-dominated: no other point is both faster and more accurate."""
    pts = sorted(points)
    front, best = [], -np.inf
    for lat, acc, name in pts:
        if acc > best:
            front.append((lat, acc, name))
            best = acc
    return front


def place_labels(ax, items, fig):
    """Greedy non-overlapping labels: try a few offsets around each marker, keep the first free one."""
    offsets = [(7, -12, "left"), (7, 5, "left"), (-7, 5, "right"), (-7, -12, "right"),
               (7, 16, "left"), (7, -24, "left"), (-7, 16, "right"), (-7, -24, "right")]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed = [ax.get_legend().get_window_extent(renderer)] if ax.get_legend() else []
    for x, y, text, color in items:
        for dx, dy, ha in offsets:
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8,
                            color=color, ha=ha)
            bb = t.get_window_extent(renderer).expanded(1.05, 1.15)
            if not any(bb.overlaps(o) for o in placed) and ax.get_window_extent(renderer).contains(bb.x0, bb.y0):
                placed.append(bb)
                break
            t.remove()
        else:  # no free spot: keep the default position
            ax.annotate(text, (x, y), xytext=offsets[0][:2], textcoords="offset points", fontsize=8, color=color)


def plot_pareto(groups, dataset, out):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    pts, fig = [], plt.figure(figsize=(9, 5.2), facecolor=SURFACE)
    ax = fig.add_axes([0.08, 0.11, 0.68, 0.8])
    style(ax)
    rows = []
    for c in config_order(groups):
        a = aggregate(groups[c])
        m, h, _ = a["acc"]
        if a["b1_ms"] is None or m is None:
            continue
        pts.append((a["b1_ms"], m, c))
        rows.append((c, a, m, h))
    if not pts:
        plt.close(fig)
        print("[pareto] no configs with both latency and accuracy yet")
        return
    front = pareto_frontier(pts)
    on_front = {p[2] for p in front}
    ax.step([p[0] for p in front], [p[1] for p in front], where="post", color=INK, lw=1.5, ls="--", zorder=2)
    for c, a, m, h in rows:
        ax.errorbar(a["b1_ms"], m, yerr=h if h is not None else None, fmt=marker_for(a["compression"]), ms=8,
                    color=ARCH_COLOR.get(a["arch"], INK2), mec=INK if c in on_front else SURFACE,
                    mew=1.5, ecolor=INK2, elinewidth=1, capsize=3, zorder=3)
    ax.set_xscale("log")
    fmt_ms = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt_ms)
    ax.xaxis.set_minor_formatter(fmt_ms)
    ax.tick_params(axis="x", which="minor", labelsize=8, colors=INK2)
    ax.set_xlabel("CPU latency, batch 1, median (ms, log scale)", color=INK, fontsize=10)
    ax.set_ylabel("Top-1 accuracy (%), mean ± 95% CI over seeds", color=INK, fontsize=10)
    cpu = next((aggregate(g)["cpu"] for g in groups.values() if aggregate(g)["cpu"]), "")
    ax.set_title(f"{dataset}: accuracy vs latency ({cpu})", color=INK, fontsize=11, loc="left")
    # Legend outside the plot: color = architecture, shape = compression (identity is never color-alone).
    archs = sorted({aggregate(groups[c])["arch"] for c in groups}, key=list(ARCH_COLOR).index)
    marks = sorted({marker_for(aggregate(groups[c])["compression"]) for c in groups}, key=list(MARKER_LABEL).index)
    handles = [Line2D([], [], marker="o", ls="", color=ARCH_COLOR[a], ms=8, label=ARCH_LABEL[a]) for a in archs]
    handles += [Line2D([], [], marker=m, ls="", color=INK2, ms=7, label=MARKER_LABEL[m]) for m in marks]
    handles += [Line2D([], [], color=INK, ls="--", lw=1.5, marker="o", mfc="none", mec=INK, label="Pareto frontier")]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    place_labels(ax, [(a["b1_ms"], m, c, INK if c in on_front else INK2) for c, a, m, h in rows], fig)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def plot_reliability(groups, fp32, int8, seed, out):
    cals = []
    for c in (fp32, int8):
        cal = groups.get(c, {}).get(seed, {}).get("calibration")
        if not cal:
            print(f"[reliability] no calibration for {c} seed {seed}; run calib.py first")
            return
        cals.append((c, cal))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True, facecolor=SURFACE)
    for ax, (c, cal) in zip(axes, cals):
        style(ax)
        bins = cal["reliability"]
        width = bins[0]["hi"] - bins[0]["lo"]
        centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
        accs = [b["acc"] if b["acc"] is not None else 0 for b in bins]
        ax.bar(centers, accs, width=width * 0.9, color=SEQ_BAR, label="Accuracy in bin")
        ax.plot([0, 1], [0, 1], color=INK2, ls="--", lw=1, label="Perfect calibration")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", color=INK)
        ax.set_title(f"{c}  ECE {cal['ece']*100:.2f}% (→ {cal['ece_ts']*100:.2f}% after TS)",
                     fontsize=9.5, color=INK, loc="left")
    axes[0].set_ylabel("Accuracy", color=INK)
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def plot_drone(groups, config, out_cm, out_size):
    runs = [r for r in groups.get(config, {}).values() if r.get("drone")]
    if not runs:
        print(f"[drone] no drone metrics for {config}; run report.py --dataset drone first")
        return
    cm = np.sum([np.array(r["drone"]["confusion"]) for r in runs], axis=0)
    names = runs[0]["drone"]["class_names"]
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    from matplotlib.colors import LinearSegmentedColormap
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor=SURFACE)
    ax.imshow(norm, cmap=LinearSegmentedColormap.from_list("seq", SEQ), vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{norm[i, j]*100:.1f}%\n({cm[i, j]})", ha="center", va="center", fontsize=8.5,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(names)), names); ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted", color=INK); ax.set_ylabel("True", color=INK)
    ax.set_title(f"{config}: row-normalized, {len(runs)} seed(s) pooled", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(out_cm, dpi=160); plt.close(fig)
    print(f"wrote {out_cm}")

    fig, ax = plt.subplots(figsize=(5.5, 3.6), facecolor=SURFACE)
    style(ax)
    labels, means, cis, ns = [], [], [], []
    for name, _, _ in SIZE_BUCKETS:
        m, h, _ = mean_ci([r["drone"]["by_size"][name]["acc"] for r in runs])
        labels.append(name); means.append(m or 0); cis.append(h or 0)
        ns.append(runs[0]["drone"]["by_size"][name]["n"])
    ax.bar(labels, means, yerr=cis, color=SEQ_BAR, width=0.55, ecolor=INK2, capsize=3)
    for i, (m, n) in enumerate(zip(means, ns)):
        ax.text(i, m + 1, f"{m:.1f}%\nn={n}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Test accuracy, objects only (%)", color=INK)
    ax.set_xlabel("Original box size (max side)", color=INK)
    ax.set_title(f"{config}: accuracy by object size", fontsize=9.5, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(out_size, dpi=160); plt.close(fig)
    print(f"wrote {out_size}")
    return out_cm, out_size


def plot_curves(groups, configs, seed, out):
    fig, ax = plt.subplots(figsize=(6.5, 4), facecolor=SURFACE)
    style(ax)
    colors = ["#2a78d6", "#eb6834", "#1baf7a"]
    for c, col in zip(configs, colors):
        h = groups.get(c, {}).get(seed, {}).get("history")
        if not h:
            print(f"[curves] no history for {c} seed {seed}")
            continue
        ax.plot([r["epoch"] for r in h], [r["val_acc"] for r in h], color=col, lw=2, label=c)
    ax.set_xlabel("Epoch", color=INK); ax.set_ylabel("Validation top-1 (%)", color=INK)
    ax.set_title(f"Validation curves, seed {seed}", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    print(f"wrote {out}")
    return out


DEFAULT_RELIABILITY = {"cifar100": ("mbv2_kd", "mbv2_kd_ptq"), "drone": ("r18_kd", "r18_kd_ptq"),
                       "fake": ("mbv2_kd", "mbv2_kd_ptq")}


def make_figures(out_root, dataset, reliability=None, drone_config="r18_kd", seed=0, curves=None, curve_seed=0):
    """Write every figure the results support; returns {name: path} for the ones that were written."""
    paths = Paths(out_root, dataset)
    groups = by_config(load_results(paths.results))
    if not groups:
        raise SystemExit(f"no results in {paths.results}")
    rel = reliability or DEFAULT_RELIABILITY[dataset]
    figs = {"pareto": plot_pareto(groups, dataset, paths.figures / f"pareto_{dataset}.png"),
            "reliability": plot_reliability(groups, *rel, seed, paths.figures / f"reliability_{dataset}_{rel[0]}.png")}
    if dataset == "drone":
        cm_size = plot_drone(groups, drone_config, paths.figures / "drone_confusion.png",
                             paths.figures / "drone_acc_by_size.png")
        if cm_size:
            figs["drone_confusion"], figs["drone_size"] = cm_size
    if curves:
        figs["curves"] = plot_curves(groups, curves, curve_seed, paths.figures / f"curves_{dataset}.png")
    return {k: v for k, v in figs.items() if v}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "drone", "fake"])
    p.add_argument("--out", default=".")
    p.add_argument("--reliability", nargs=2, default=None, metavar=("FP32", "INT8"),
                   help="default: mbv2_kd/mbv2_kd_ptq on CIFAR-100, r18_kd/r18_kd_ptq on drone")
    p.add_argument("--drone-config", default="r18_kd")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--curves", nargs="*", default=None)
    p.add_argument("--curve-seed", type=int, default=0)
    args = p.parse_args()

    make_figures(args.out, args.dataset, args.reliability, args.drone_config, args.seed,
                 args.curves, args.curve_seed)

if __name__ == "__main__":
    main()
