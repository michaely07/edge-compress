"""Statistics and the results table, generated from results/<dataset>/*.json (never typed by hand).

- mean ± 95% CI over seeds with the t-distribution (n=3 -> t=4.303, so intervals are wide)
- paired comparisons on the same seed (e.g. KD vs CE), CI of the per-seed differences
- drone case study: per-class recall, bird->drone false-alarm rate, accuracy by object size

  python report.py --dataset cifar100 --update-readme     # tables + figures + README block
  python report.py --dataset drone --data-dir data --update-readme
"""
import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import torch

from common import Paths, load_results, update_result

# Two-sided 95% critical values of Student's t, df = 1..30.
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
       2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
       2.052, 2.048, 2.045, 2.042]

# The experiment matrix, in README order.
CONFIG_ORDER = ["r50_ce", "r18_ce", "r18_kd", "mbv2_ce", "mbv2_kd", "r18_kd_ptq", "mbv2_kd_ptq",
                "mbv2_kd_qat", "r18_kd_prune50", "r18_kd_prune50_ptq"]
PAIRS = [("r18_kd", "r18_ce"), ("mbv2_kd", "mbv2_ce"), ("r18_kd_ptq", "r18_kd"),
         ("mbv2_kd_ptq", "mbv2_kd"), ("mbv2_kd_qat", "mbv2_kd_ptq"), ("mbv2_kd_qat", "mbv2_kd"),
         ("r18_kd_prune50", "r18_kd"), ("r18_kd_prune50_ptq", "r18_kd_prune50")]
SIZE_BUCKETS = [("<16px", 0, 16), ("16-32px", 16, 32), (">32px", 32, math.inf)]


def t_crit(df):
    return T95[df - 1] if df <= len(T95) else 1.96


def mean_ci(xs):
    """(mean, 95% CI half-width, n). Half-width is None for a single run."""
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return None, None, 0
    m = sum(xs) / n
    if n == 1:
        return m, None, 1
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, t_crit(n - 1) * sd / math.sqrt(n), n


def paired_ci(a, b):
    """a, b: {seed: value}. CI of the per-seed differences a - b over the shared seeds."""
    seeds = sorted(set(a) & set(b))
    diffs = [a[s] - b[s] for s in seeds]
    m, h, n = mean_ci(diffs)
    return {"mean": m, "ci": h, "n": n, "diffs": diffs, "seeds": seeds}


def by_config(records):
    out = defaultdict(dict)
    for r in records:
        out[r["config"]][r["seed"]] = r
    return out


def config_order(configs):
    known = [c for c in CONFIG_ORDER if c in configs]
    return known + sorted(c for c in configs if c not in CONFIG_ORDER)


def aggregate(runs):
    """runs: {seed: record} for one config -> summary row."""
    recs = list(runs.values())
    first = recs[0]
    acc = mean_ci([r.get("test_acc") for r in recs])
    lat = first.get("latency") or {}
    cal = [r.get("calibration") or {} for r in recs]
    return {
        "precision": first.get("precision"), "training": first.get("training"),
        "compression": first.get("compression"), "arch": first.get("arch"),
        "acc": acc, "seeds": sorted(runs),
        "ece": mean_ci([c.get("ece") for c in cal])[0], "ece_ts": mean_ci([c.get("ece_ts") for c in cal])[0],
        "b1_ms": lat.get("b1_median_ms"), "b1_p90_ms": lat.get("b1_p90_ms"),
        "ips": lat.get("b32_throughput_ips"), "macs": first.get("macs"), "params": first.get("params"),
        "size_mb": first.get("size_mb"), "cpu": lat.get("cpu"),
    }


def fmt(x, f="{:.2f}", none="–"):
    return none if x is None else f.format(x)


def results_table(groups):
    lines = ["| Config | Arch | Training | Compression | Top-1 (%) | Seeds | b1 median (ms) | b1 p90 (ms) "
             "| b32 (img/s) | MMACs | Params (M) | Size (MB) | ECE % → TS |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in config_order(groups):
        a = aggregate(groups[c])
        m, h, n = a["acc"]
        acc = fmt(m) + (f" ± {h:.2f}" if h is not None else "")
        ece = (f"{a['ece']*100:.2f} → {a['ece_ts']*100:.2f}" if a["ece"] is not None else "–")
        lines.append(f"| `{c}` | {a['arch']} | {a['training']} | {a['compression']} | {acc} | {n} | "
                     f"{fmt(a['b1_ms'], '{:.3f}')} | {fmt(a['b1_p90_ms'], '{:.3f}')} | {fmt(a['ips'], '{:.0f}')} | "
                     f"{fmt(a['macs'] and a['macs'] / 1e6, '{:.0f}')} | {fmt(a['params'] and a['params'] / 1e6)} | "
                     f"{fmt(a['size_mb'])} | {ece} |")
    return "\n".join(lines)


def paired_table(groups):
    lines = ["| Comparison (same seed) | Δ top-1 (pts) | 95% CI | Seeds | Per-seed Δ |", "|---|---|---|---|---|"]
    for a, b in PAIRS:
        if a not in groups or b not in groups:
            continue
        r = paired_ci({s: x["test_acc"] for s, x in groups[a].items()},
                      {s: x["test_acc"] for s, x in groups[b].items()})
        if not r["n"]:
            continue
        ci = f"[{r['mean'] - r['ci']:+.2f}, {r['mean'] + r['ci']:+.2f}]" if r["ci"] is not None else "–"
        lines.append(f"| `{a}` − `{b}` | {r['mean']:+.2f} | {ci} | {r['n']} | "
                     f"{', '.join(f'{d:+.2f}' for d in r['diffs'])} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- drone case study

def drone_metrics(logits, labels, meta, class_names):
    pred = logits.argmax(1)
    k = len(class_names)
    cm = torch.zeros(k, k, dtype=torch.long)
    for t, p in zip(labels.tolist(), pred.tolist()):
        cm[t, p] += 1
    recall = {class_names[i]: (cm[i, i] / cm[i].sum()).item() if cm[i].sum() else None for i in range(k)}
    bird, drone = class_names.index("bird"), class_names.index("drone")
    out = {"accuracy": pred.eq(labels).float().mean().item() * 100, "recall": recall,
           "bird_to_drone_rate": (cm[bird, drone] / cm[bird].sum()).item() if cm[bird].sum() else None,
           "confusion": cm.tolist(), "class_names": class_names, "by_size": {}}
    sizes = torch.tensor([float(r["box_px"]) if r.get("box_px") else math.nan for r in meta])
    correct = pred.eq(labels)
    for name, lo, hi in SIZE_BUCKETS:
        m = (sizes >= lo) & (sizes < hi)
        out["by_size"][name] = {"n": int(m.sum()), "acc": correct[m].float().mean().item() * 100 if m.any() else None}
    if meta and "near_dup" in meta[0]:
        clean = torch.tensor([r["near_dup"] in ("0", "False", "") for r in meta])
        out["acc_dedup_test"] = correct[clean].float().mean().item() * 100 if clean.any() else None
        out["n_dedup_test"] = int(clean.sum())
    return out


def drone_analysis(paths, data_dir):
    from data import DRONE_CLASSES, read_manifest
    meta = [r for r in read_manifest(Path(data_dir) / "drone_crops") if r["split"] == "test"]
    lines = ["| Config | Acc (%) | Recall drone | Recall bird | Recall bg | Bird→drone false alarms | "
             + " | ".join(f"Acc {b[0]}" for b in SIZE_BUCKETS) + " | Acc, dedup test |",
             "|---|---|---|---|---|---|" + "---|" * len(SIZE_BUCKETS) + "---|"]
    per_config = defaultdict(list)
    for f in sorted(paths.logits.glob("*_test.pt")):
        m = re.match(r"(.+)_seed(\d+)_test$", f.stem)
        d = torch.load(f)
        dm = drone_metrics(d["logits"], d["labels"], meta, DRONE_CLASSES)
        update_result(paths.result(m.group(1), int(m.group(2))), {"drone": dm})
        per_config[m.group(1)].append(dm)
    for c in config_order(per_config):
        ms = per_config[c]
        avg = lambda f: mean_ci([f(x) for x in ms])[0]
        r = lambda name: avg(lambda x: x["recall"][name] and x["recall"][name] * 100)
        sizes = " | ".join(fmt(avg(lambda x, b=b[0]: x["by_size"][b]["acc"])) for b in SIZE_BUCKETS)
        lines.append(f"| `{c}` | {fmt(avg(lambda x: x['accuracy']))} | {fmt(r('drone'))} | {fmt(r('bird'))} | "
                     f"{fmt(r('background'))} | {fmt(avg(lambda x: x['bird_to_drone_rate'] and x['bird_to_drone_rate'] * 100))}% | "
                     f"{sizes} | {fmt(avg(lambda x: x.get('acc_dedup_test')))} |")
    return "\n".join(lines)


def inject(readme, tag, text):
    """Replace the block between <!-- RESULTS:tag --> and <!-- /RESULTS:tag --> in the README."""
    p = Path(readme)
    s = p.read_text(encoding="utf-8")
    start, end = f"<!-- RESULTS:{tag} -->", f"<!-- /RESULTS:{tag} -->"
    if start not in s:
        print(f"[warn] no {start} marker in {readme}")
        return
    pre, rest = s.split(start, 1)
    post = rest.split(end, 1)[1]
    p.write_text(f"{pre}{start}\n{text}\n{end}{post}", encoding="utf-8")


DATASET_TITLE = {"cifar100": "CIFAR-100", "drone": "Drone / bird / background (64×64 crops)", "fake": "Smoke test (fake data)"}


def setup_line(groups):
    """Hardware and library versions, read from the runs themselves."""
    recs = [r for g in groups.values() for r in g.values()]
    gpus = sorted({(r.get("env") or {}).get("gpu") for r in recs if r.get("device") == "cuda"} - {None})
    lats = [r["latency"] for r in recs if r.get("latency")]
    env = next((r["env"] for r in recs if r.get("env")), {})
    names = {"torch": "PyTorch", "torchvision": "torchvision", "torchao": "torchao",
             "torch_pruning": "Torch-Pruning", "triton": "Triton"}
    versions = ", ".join(f"{n} {env[k]}" for k, n in names.items() if env.get(k))
    lat = (f"{lats[0]['cpu']}, {lats[0]['threads']} threads, torch.compile "
           f"{'on' if lats[0].get('compiled') else 'OFF (eager fallback)'}") if lats else "not yet measured"
    return (f"Training: {', '.join(gpus) or 'CPU'}. Latency (batch 1): {lat}. "
            f"CPU VNNI: { {True: 'yes', False: 'no'}.get(env.get('cpu_vnni'), 'unknown')}. {versions}.")


def leakage_summary(data_dir):
    f = Path(data_dir) / "drone_crops" / "leakage.json"
    if not f.exists():
        return None
    rep = json.loads(f.read_text())
    fr, cr = rep["frames"]["test"], rep["crops"]["test"]
    return (f"**Train/test leakage** (perceptual hash, ≤{rep['max_dist_bits']} of 64 bits): "
            f"{fr['near_dup']} / {fr['n']} test frames ({fr['frac'] * 100:.1f}%) and {cr['near_dup']} / {cr['n']} "
            f"test crops ({cr['frac'] * 100:.1f}%) near-duplicate a training item ({fr['exact_dup']} frames are "
            f"exact duplicates). The last column of the table below re-scores accuracy with those crops removed.")


def figure(figs, key, alt, root):
    return [f"![{alt}]({Path(figs[key]).relative_to(root).as_posix()})", ""] if key in figs else []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "drone", "fake"])
    p.add_argument("--out", default=".")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--update-readme", action="store_true", help="also draw the figures and fill the README block")
    args = p.parse_args()

    paths = Paths(args.out, args.dataset)
    groups = by_config(load_results(paths.results))
    if not groups:
        raise SystemExit(f"no results in {paths.results}")
    drone = args.dataset == "drone"
    drone_text = drone_analysis(paths, args.data_dir) if drone else None  # writes per-run drone metrics first
    figs = {}
    if args.update_readme:
        from plots import make_figures
        figs = make_figures(args.out, args.dataset)
    root = Path(args.out)

    parts = [f"### {DATASET_TITLE[args.dataset]}", "", setup_line(groups), ""]
    parts += figure(figs, "pareto", "Accuracy vs CPU latency", root)
    parts += ["Top-1 is the mean ± 95% CI over seeds (t-distribution; with 3 seeds t = 4.303, so intervals "
              "are wide). ECE is before → after temperature scaling.", "", results_table(groups), "",
              "**Paired comparisons** (difference on the same seed, which removes seed-to-seed variance):", "",
              paired_table(groups), ""]
    parts += figure(figs, "reliability", "Reliability diagrams, FP32 vs INT8", root)
    if drone:
        leak = leakage_summary(args.data_dir)
        parts += ["**Counter-drone metrics** (test split, mean over seeds):", ""] + ([leak, ""] if leak else [])
        parts += [drone_text, ""]
        parts += figure(figs, "drone_confusion", "Drone confusion matrix", root)
        parts += figure(figs, "drone_size", "Accuracy by object size", root)
    text = "\n".join(parts).rstrip()
    (paths.results / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.update_readme:
        inject(root / "README.md", args.dataset, text)


if __name__ == "__main__":
    main()
