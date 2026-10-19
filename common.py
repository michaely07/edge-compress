"""Shared helpers: output paths, per-run JSON records, checkpoints, evaluation, environment info."""
import datetime
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Short names used in config strings, e.g. r18_kd_ptq.
ARCH_ABBR = {"resnet18": "r18", "resnet50": "r50", "mobilenetv2": "mbv2"}
ABBR_ARCH = {v: k for k, v in ARCH_ABBR.items()}

if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252; tables print ± and →
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------- paths

class Paths:
    """All outputs live under one root, so a smoke run never mixes with real results."""

    def __init__(self, root=".", dataset="cifar100"):
        self.root = Path(root)
        self.dataset = dataset
        self.results = self.root / "results" / dataset
        self.ckpt = self.root / "checkpoints" / dataset
        self.logits = self.root / "logits" / dataset
        self.figures = self.root / "figures"
        for d in (self.results, self.ckpt, self.logits, self.figures):
            d.mkdir(parents=True, exist_ok=True)

    def result(self, config, seed):
        return self.results / f"{config}_seed{seed}.json"

    def checkpoint(self, config, seed, ext=".pt"):
        return self.ckpt / f"{config}_seed{seed}{ext}"

    def logit_file(self, config, seed, split):
        return self.logits / f"{config}_seed{seed}_{split}.pt"


def add_common_args(p):
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "drone", "fake"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default=".", help="root for results/, checkpoints/, logits/, figures/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4 if os.name != "nt" else 0)
    p.add_argument("--subset", type=int, default=0, help="debug: use only N training images")
    return p


# ---------------------------------------------------------------- run records

def update_result(path, record):
    """Merge `record` into the run's JSON so train/quantize/bench/calib each add their own fields."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    for k, v in record.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k].update(v)
        else:
            data[k] = v
    data["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)
    return data


def read_result(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def load_results(results_dir):
    out = []
    for f in sorted(Path(results_dir).glob("*_seed*.json")):
        out.append(json.loads(f.read_text()))
    return out


def log_number(out_root, text):
    """Append a timestamped line to the RESULTS.md experiment log as soon as a number is produced."""
    f = Path(out_root) / "RESULTS.md"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(f"- `{stamp}` {text}\n")
    print(f"[RESULTS] {text}")


# ---------------------------------------------------------------- environment

def cpu_model():
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Windows":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20)
            if out.stdout.strip():
                return out.stdout.strip()
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
            return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def cpu_has_vnni():
    """VNNI matters for INT8: without it oneDNN can saturate INT8 math (see README)."""
    try:
        if platform.system() == "Linux":
            flags = open("/proc/cpuinfo").read()
            return "vnni" in flags
    except Exception:
        pass
    return None  # not detected outside Linux


def env_info():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "cpu": cpu_model(),
        "cpu_vnni": cpu_has_vnni(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for mod in ("torchvision", "torchao", "torch_pruning", "triton"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    return info


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- checkpoints

def save_checkpoint(path, model, arch, num_classes, extra=None, whole_module=False):
    """FP32 models store a state_dict; pruned models store the whole module (shapes changed)."""
    payload = {"arch": arch, "num_classes": num_classes, "extra": extra or {}}
    model = model.cpu() if whole_module else model
    if whole_module:
        payload["module"] = model
    else:
        payload["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(payload, path)


def load_checkpoint(path):
    from models import build_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "module" in ck:
        model = ck["module"]
    else:
        model = build_model(ck["arch"], ck["num_classes"])
        model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def file_size_mb(path):
    return os.path.getsize(path) / 2**20


def state_dict_size_mb(model, tmp_path):
    torch.save(model.state_dict(), tmp_path)
    size = file_size_mb(tmp_path)
    os.remove(tmp_path)
    return size


# ---------------------------------------------------------------- evaluation

@torch.no_grad()
def collect_logits(model, loader, device="cpu", amp=False):
    """Run a model over a loader; return float32 logits and labels on CPU."""
    logits, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
        logits.append(out.float().cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def metrics_from_logits(logits, labels):
    top5 = logits.topk(min(5, logits.shape[1]), 1).indices
    return {
        "top1": logits.argmax(1).eq(labels).float().mean().item() * 100,
        "top5": top5.eq(labels[:, None]).any(1).float().mean().item() * 100,
        "nll": F.cross_entropy(logits, labels).item(),
    }


def save_logits(path, logits, labels):
    torch.save({"logits": logits, "labels": labels}, path)
