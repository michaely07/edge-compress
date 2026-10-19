"""Build the drone / bird / background crop dataset for the counter-drone case study, and check
train-test leakage.

The source is a *detection* dataset: Kaggle "Bird vs Drone" (~20.9k 640x640 video frames with
YOLO-format boxes, already split into train/valid/test). The models in this repo are image
classifiers, so each labeled box is turned into a classification sample. CIFAR-100 needs no such step
because torchvision provides it ready to use. Derived crops are not committed to this repo; the
dataset's own license governs redistribution.

1. Crops: each box is cropped as a square around its center with `--pad` (20%) padding on each
   side, clamped to the frame, and resized to 64x64. The original box size (max side, px) is
   kept for the small-object analysis.
2. Background: 64x64 patches sampled at native resolution that don't overlap any (padded) box.
3. Leakage: the frames come from videos, so near-duplicates can land in both train and test.
   Every source frame and every crop is perceptually hashed (imagehash.phash); test items within
   `--max-dist` bits of any train item are counted, reported, and flagged in the manifest
   (`near_dup`) so accuracy can also be reported on a de-duplicated test set.

  python drone_data.py --src /path/to/bird-vs-drone --out data/drone_crops
"""
import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["drone", "bird", "background"]
SPLITS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


# ---------------------------------------------------------------- discovery

def read_yaml_names(src):
    """Class names from the dataset's data.yaml (`names: [...]` or a `names:` block)."""
    for y in sorted(Path(src).rglob("*.y*ml")):
        text = y.read_text(errors="ignore")
        m = re.search(r"names\s*:\s*\[(.*?)\]", text, re.S)
        if m:
            return [s.strip().strip("'\"") for s in m.group(1).split(",") if s.strip()]
        m = re.search(r"names\s*:\s*\n((?:\s+.*\n?)+)", text)
        if m:
            names = []
            for line in m.group(1).splitlines():
                line = line.strip()
                if not line:
                    break
                names.append(re.sub(r"^(-|\d+\s*:)\s*", "", line).strip("'\""))
            return names
    return None


def class_map(names, override=None):
    """YOLO class id -> our label index (drone=0, bird=1)."""
    if override:
        return {int(k): CLASSES.index(v) for k, v in (kv.split("=") for kv in override.split(","))}
    if not names:
        raise SystemExit("no data.yaml with class names found; pass --class-map e.g. 0=bird,1=drone")
    out = {}
    for i, n in enumerate(names):
        n = n.lower()
        if "drone" in n or "uav" in n:
            out[i] = 0
        elif "bird" in n:
            out[i] = 1
    if sorted(out.values()) != [0, 1]:
        raise SystemExit(f"could not map class names {names}; pass --class-map")
    return out


def find_frames(src):
    """Yield (split, image_path, label_path) for every frame under */images/*."""
    for img in sorted(Path(src).rglob("*")):
        if img.suffix.lower() not in IMG_EXT or img.parent.name != "images":
            continue
        split = next((SPLITS[p.lower()] for p in reversed(img.parts[:-2]) if p.lower() in SPLITS), None)
        if split is None:
            continue
        lab = img.parent.parent / "labels" / (img.stem + ".txt")
        yield split, img, lab


def read_boxes(label_path, W, H):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            boxes.append((c, cx * W, cy * H, w * W, h * H))
    return boxes


# ---------------------------------------------------------------- cropping

def square_crop(img, cx, cy, w, h, pad, out=64):
    W, H = img.size
    side = max(w, h) * (1 + 2 * pad)
    side = max(4.0, min(side, W, H))
    x0 = min(max(cx - side / 2, 0), W - side)
    y0 = min(max(cy - side / 2, 0), H - side)
    return img.crop((round(x0), round(y0), round(x0 + side), round(y0 + side))).resize((out, out), Image.BILINEAR)


def overlaps(a, b):
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def sample_background(img, boxes, k, rng, size=64, pad=0.2, tries=50):
    W, H = img.size
    if W < size or H < size:
        return []
    keep = [(cx - w / 2 - pad * w, cy - h / 2 - pad * h, cx + w / 2 + pad * w, cy + h / 2 + pad * h)
            for _, cx, cy, w, h in boxes]
    out = []
    for _ in range(tries):
        if len(out) == k:
            break
        x0, y0 = rng.randint(0, W - size), rng.randint(0, H - size)
        r = (x0, y0, x0 + size, y0 + size)
        if not any(overlaps(r, b) for b in keep) and not any(overlaps(r, o) for o in out):
            out.append(r)
    return [img.crop(r) for r in out]


# ---------------------------------------------------------------- leakage

def phash_int(img):
    import imagehash
    h = imagehash.phash(img)  # 64-bit perceptual hash
    return int(str(h), 16)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def min_hamming(queries, refs, chunk=256):
    """For each query hash, the minimum Hamming distance to any reference hash (64-bit)."""
    if len(refs) == 0 or len(queries) == 0:
        return np.full(len(queries), 64)
    r = np.array(refs, dtype=np.uint64)
    out = []
    for i in range(0, len(queries), chunk):
        q = np.array(queries[i:i + chunk], dtype=np.uint64)
        x = (q[:, None] ^ r[None, :]).view(np.uint8).reshape(len(q), len(r), 8)
        out.append(_POP[x].sum(-1).min(1))
    return np.concatenate(out)


def leakage_report(items, max_dist):
    """items: dicts with split, hash. Count test (and valid) items near-duplicating a train item."""
    train = [it["hash"] for it in items if it["split"] == "train"]
    rep = {}
    for split in ("valid", "test"):
        sel = [it for it in items if it["split"] == split]
        d = min_hamming([it["hash"] for it in sel], train)
        for it, di in zip(sel, d):
            it["min_dist"] = int(di)
        n_dup = int((d <= max_dist).sum())
        rep[split] = {"n": len(sel), "near_dup": n_dup, "frac": n_dup / max(1, len(sel)),
                      "exact_dup": int((d == 0).sum()),
                      "dist_hist": {str(k): int(v) for k, v in sorted(Counter(d.tolist()).items()) if k <= 16}}
    return rep


# ---------------------------------------------------------------- main

def build(src, out, pad=0.2, size=64, bg_per_frame=1, seed=0, override=None, max_dist=4, limit=0):
    rng = random.Random(seed)
    out = Path(out)
    for s in ("train", "valid", "test"):
        for c in CLASSES:
            (out / s / c).mkdir(parents=True, exist_ok=True)
    cmap = class_map(read_yaml_names(src), override)
    print(f"YOLO class id -> label: { {k: CLASSES[v] for k, v in cmap.items()} }")

    rows, frames = [], []
    for n, (split, img_path, lab_path) in enumerate(find_frames(src)):
        if limit and n >= limit:
            break
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        boxes = [b for b in read_boxes(lab_path, W, H) if b[0] in cmap]
        frames.append({"split": split, "src": str(img_path), "hash": phash_int(img)})
        stem = f"{img_path.stem}"
        for i, (c, cx, cy, w, h) in enumerate(boxes):
            crop = square_crop(img, cx, cy, w, h, pad, size)
            cls = CLASSES[cmap[c]]
            rel = f"{split}/{cls}/{stem}_{i}.png"
            crop.save(out / rel)
            rows.append({"path": rel, "split": split, "label": cmap[c], "cls": cls,
                         "box_px": f"{max(w, h):.1f}", "src": str(img_path), "hash": phash_int(crop)})
        for j, patch in enumerate(sample_background(img, boxes, bg_per_frame, rng, size, pad)):
            rel = f"{split}/background/{stem}_bg{j}.png"
            patch.save(out / rel)
            rows.append({"path": rel, "split": split, "label": 2, "cls": "background", "box_px": "",
                         "src": str(img_path), "hash": phash_int(patch)})
        if n % 1000 == 0:
            print(f"{n} frames, {len(rows)} crops")

    report = {"frames": leakage_report(frames, max_dist), "crops": leakage_report(rows, max_dist),
              "max_dist_bits": max_dist, "pad": pad, "size": size,
              "counts": {s: dict(Counter(r["cls"] for r in rows if r["split"] == s)) for s in ("train", "valid", "test")}}
    for r in rows:
        r["near_dup"] = int(r.get("min_dist", 64) <= max_dist) if r["split"] != "train" else 0
        r.pop("min_dist", None)
        r["hash"] = f"{r['hash']:016x}"
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "split", "label", "cls", "box_px", "src", "hash", "near_dup"])
        w.writeheader()
        w.writerows(rows)
    (out / "leakage.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="root of the Bird vs Drone dataset")
    p.add_argument("--out", default="data/drone_crops")
    p.add_argument("--pad", type=float, default=0.2)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--bg-per-frame", type=int, default=1)
    p.add_argument("--class-map", default=None, help="override, e.g. 0=bird,1=drone")
    p.add_argument("--max-dist", type=int, default=4, help="phash Hamming distance counted as near-duplicate")
    p.add_argument("--limit", type=int, default=0, help="debug: only the first N frames")
    p.add_argument("--results-root", default=".", help="where RESULTS.md lives")
    args = p.parse_args()
    rep = build(args.src, args.out, args.pad, args.size, args.bg_per_frame, 0, args.class_map,
                args.max_dist, args.limit)
    print(json.dumps({k: rep[k] for k in ("counts", "max_dist_bits")}, indent=2))
    from common import log_number
    for level in ("frames", "crops"):
        t = rep[level]["test"]
        log_number(args.results_root, f"drone leakage ({level}, phash ≤{args.max_dist} bits): {t['near_dup']}/{t['n']} "
                                      f"test {level} ({t['frac']*100:.1f}%) near-duplicate a train {level[:-1]}; "
                                      f"{t['exact_dup']} exact")


if __name__ == "__main__":
    main()
