"""Run the experiment matrix as a dependency-aware job queue over the available GPUs.

Each GPU runs one job at a time (CUDA_VISIBLE_DEVICES=<i>), CPU-only jobs (PTQ) run in their own
slot, and a job starts as soon as its dependencies finish. Jobs whose result JSON already has a
test accuracy are skipped, so an interrupted run picks up where it stopped.

  python run_matrix.py --dataset cifar100                 # full matrix, 3 seeds
  python run_matrix.py --dataset cifar100 --dry-run       # print the plan
  python run_matrix.py --dataset cifar100 --ce-seeds-mbv2 1   # one seed for the mbv2_ce baseline
  python run_matrix.py --dataset drone --data-dir data
  python run_matrix.py --smoke                            # tiny CPU run on fake data, end to end
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PY = sys.executable
os.environ.setdefault("PYTHONIOENCODING", "utf-8")  # report tables print ± and → on Windows consoles too


@dataclass
class Job:
    name: str
    cmd: list
    device: str = "gpu"          # gpu | cpu
    deps: list = field(default_factory=list)
    result: str = ""             # results JSON that marks the job done
    always: bool = False         # rerun even if a result exists (analysis steps)


def plan(a):
    common = ["--dataset", a.dataset, "--data-dir", a.data_dir, "--out", a.out,
              "--num-workers", str(a.num_workers)] + (["--subset", str(a.subset)] if a.subset else [])
    res = lambda config, s: str(Path(a.out) / "results" / a.dataset / f"{config}_seed{s}.json")
    jobs = []

    def add(name, cmd, device="gpu", deps=(), result="", always=False):
        jobs.append(Job(name, cmd, device, list(deps), result, always))

    add("data", [PY, "data.py", "--dataset", a.dataset, "--data-dir", a.data_dir], "cpu", always=True)
    add("r50_ce_s0", [PY, "train.py", "--model", "resnet50", "--epochs", str(a.teacher_epochs), "--seed", "0"] + common,
        result=res("r50_ce", 0))
    drone = a.dataset == "drone"
    for s in a.seeds:
        seed = ["--seed", str(s)]
        ep = ["--epochs", str(a.student_epochs)]
        if not drone:
            add(f"r18_ce_s{s}", [PY, "train.py", "--model", "resnet18"] + ep + seed + common, result=res("r18_ce", s))
            if s in a.seeds[:a.ce_seeds_mbv2]:
                add(f"mbv2_ce_s{s}", [PY, "train.py", "--model", "mobilenetv2"] + ep + seed + common,
                    result=res("mbv2_ce", s))
        add(f"r18_kd_s{s}", [PY, "train.py", "--model", "resnet18", "--teacher", "r50_ce"] + ep + seed + common,
            deps=["r50_ce_s0"], result=res("r18_kd", s))
        add(f"mbv2_kd_s{s}", [PY, "train.py", "--model", "mobilenetv2", "--teacher", "r50_ce"] + ep + seed + common,
            deps=["r50_ce_s0"], result=res("mbv2_kd", s))
        add(f"r18_kd_ptq_s{s}", [PY, "quantize.py", "--mode", "ptq", "--base", "r18_kd"] + seed + common,
            device="cpu", deps=[f"r18_kd_s{s}"], result=res("r18_kd_ptq", s))
        if not drone:
            add(f"mbv2_kd_ptq_s{s}", [PY, "quantize.py", "--mode", "ptq", "--base", "mbv2_kd"] + seed + common,
                device="cpu", deps=[f"mbv2_kd_s{s}"], result=res("mbv2_kd_ptq", s))
        add(f"mbv2_kd_qat_s{s}", [PY, "quantize.py", "--mode", "qat", "--base", "mbv2_kd", "--teacher", "r50_ce",
                                  "--epochs", str(a.qat_epochs)] + seed + common,
            deps=[f"mbv2_kd_s{s}", "r50_ce_s0"], result=res("mbv2_kd_qat", s))
        add(f"r18_kd_prune50_s{s}", [PY, "prune.py", "--base", "r18_kd", "--ratio", "0.5", "--teacher", "r50_ce",
                                     "--epochs", str(a.prune_epochs)] + seed + common,
            deps=[f"r18_kd_s{s}", "r50_ce_s0"], result=res("r18_kd_prune50", s))
        if not drone:
            add(f"r18_kd_prune50_ptq_s{s}", [PY, "quantize.py", "--mode", "ptq", "--base", "r18_kd_prune50"] + seed + common,
                device="cpu", deps=[f"r18_kd_prune50_s{s}"], result=res("r18_kd_prune50_ptq", s))

    for j in jobs[1:]:
        j.deps.insert(0, "data")
    everything = [j.name for j in jobs]
    out = ["--dataset", a.dataset, "--out", a.out]
    add("calib", [PY, "calib.py"] + out, "cpu", everything, always=True)
    last = ["calib"]
    if a.bench:
        add("bench", [PY, "bench.py"] + out + a.bench_args, "cpu", ["calib"], always=True)
        last = ["bench"]
    add("report", [PY, "report.py", "--data-dir", a.data_dir] + out, "cpu", last, always=True)
    add("plots", [PY, "plots.py"] + out, "cpu", ["report"], always=True)
    return jobs


def is_done(job):
    if job.always or not job.result or not Path(job.result).exists():
        return False
    return "test_acc" in json.loads(Path(job.result).read_text())


def run(jobs, n_gpus, log_dir, dry_run=False):
    log_dir.mkdir(parents=True, exist_ok=True)
    status = {j.name: ("done" if is_done(j) else "todo") for j in jobs}
    for j in jobs:
        print(f"[{status[j.name]:4}] {j.device:3} {j.name:24} deps={j.deps}\n        {' '.join(j.cmd)}")
    if dry_run:
        return status
    gpu_slots = {f"gpu{i}": None for i in range(n_gpus)} if n_gpus else {}
    slots = {**gpu_slots, "cpu0": None}
    running = {}  # slot -> (job, Popen, t0)
    t_start = time.time()
    while True:
        for slot, (job, proc, t0) in list(running.items()):
            if proc.poll() is not None:
                ok = proc.returncode == 0
                status[job.name] = "done" if ok else "failed"
                print(f"{'ok  ' if ok else 'FAIL'} {job.name} on {slot} in {(time.time() - t0) / 60:.1f} min"
                      + ("" if ok else f" (exit {proc.returncode}; see {log_dir / (job.name + '.log')})"))
                del running[slot]
        for j in jobs:  # propagate failures to dependents
            if status[j.name] == "todo" and any(status[d] in ("failed", "skipped") for d in j.deps):
                status[j.name] = "skipped"
                print(f"- skip {j.name} (dependency failed)")
        ready = [j for j in jobs if status[j.name] == "todo" and all(status[d] == "done" for d in j.deps)]
        for j in ready:
            # GPU jobs need a GPU slot; with no GPU at all, everything shares the CPU slot.
            pool = [s for s in slots if s.startswith("gpu")] if (j.device == "gpu" and n_gpus) else ["cpu0"]
            free = [s for s in pool if s not in running]
            if not free:
                continue
            slot = free[0]
            env = dict(os.environ)
            if slot.startswith("gpu"):
                env["CUDA_VISIBLE_DEVICES"] = slot[3:]
            elif n_gpus:
                env["CUDA_VISIBLE_DEVICES"] = ""
            log = open(log_dir / f"{j.name}.log", "w")
            print(f"start {j.name} on {slot}")
            running[slot] = (j, subprocess.Popen(j.cmd, stdout=log, stderr=subprocess.STDOUT, env=env), time.time())
            status[j.name] = "running"
        if not running and not any(s == "todo" for s in status.values()):
            break
        if not running and not ready:
            break
        time.sleep(2)
    print(f"finished in {(time.time() - t_start) / 60:.1f} min: "
          + ", ".join(f"{k}={sum(v == k for v in status.values())}" for k in ("done", "failed", "skipped")))
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "drone", "fake"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=".")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--ce-seeds-mbv2", type=int, default=3, help="seeds for config 4 (the least important)")
    p.add_argument("--teacher-epochs", type=int, default=60)
    p.add_argument("--student-epochs", type=int, default=40)
    p.add_argument("--qat-epochs", type=int, default=5)
    p.add_argument("--prune-epochs", type=int, default=15)
    p.add_argument("--num-workers", type=int, default=4 if os.name != "nt" else 0)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--gpus", type=int, default=None, help="default: all visible GPUs")
    p.add_argument("--bench", action="store_true", help="also run bench.py (latency should come from one fixed benchmark machine)")
    p.add_argument("--bench-args", nargs="*", default=[])
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end CPU run on fake data")
    a = p.parse_args()
    if a.smoke:
        a.dataset, a.out, a.seeds, a.subset = "fake", "runs/smoke", [0, 1], 256
        a.teacher_epochs = a.student_epochs = a.qat_epochs = a.prune_epochs = 1
        a.bench, a.bench_args = True, ["--iters", "20", "--warmup", "3", "--allow-eager"]
    if a.gpus is None:
        try:
            import torch
            a.gpus = torch.cuda.device_count()
        except Exception:
            a.gpus = 0
    if a.dataset == "drone" and a.teacher_epochs == 60:
        a.teacher_epochs, a.student_epochs = 30, 30
    Path(a.out).mkdir(parents=True, exist_ok=True)
    status = run(plan(a), a.gpus, Path(a.out) / "logs" / a.dataset, a.dry_run)
    sys.exit(1 if any(v == "failed" for v in status.values()) else 0)


if __name__ == "__main__":
    main()
