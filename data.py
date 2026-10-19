"""CIFAR-100 and drone-crop loaders with a fixed train/val split.

The 5,000-image CIFAR validation split is fixed (generator seed 0) and shared by every training
seed. It is used for model selection and temperature scaling; the test set is touched only for
final numbers.
"""
import csv
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as Tr
from torchvision import datasets

CIFAR_MEAN, CIFAR_STD = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DRONE_CLASSES = ["drone", "bird", "background"]
VAL_SIZE = 5000
SPLIT_SEED = 0


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    calib: DataLoader          # training images, eval transform, fixed order; for PTQ calibration
    num_classes: int
    img_size: int
    class_names: list = field(default_factory=list)
    test_meta: list = field(default_factory=list)   # drone: per-test-sample manifest rows


def _transforms(img_size, mean, std):
    pad = img_size // 8  # 4px at 32px, as in the standard CIFAR recipe
    train = Tr.Compose([Tr.RandomCrop(img_size, padding=pad), Tr.RandomHorizontalFlip(),
                        Tr.ToTensor(), Tr.Normalize(mean, std)])
    test = Tr.Compose([Tr.ToTensor(), Tr.Normalize(mean, std)])
    return train, test


class ManifestDataset(Dataset):
    """Drone crops listed in manifest.csv (built by drone_data.py)."""

    def __init__(self, root, rows, transform):
        self.root, self.rows, self.transform = Path(root), rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(self.root / r["path"]).convert("RGB")
        return self.transform(img), int(r["label"])


def read_manifest(root):
    with open(Path(root) / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def _split_indices(n, val_size):
    g = torch.Generator().manual_seed(SPLIT_SEED)
    perm = torch.randperm(n, generator=g).tolist()
    return perm[val_size:], perm[:val_size]


def get_data(dataset="cifar100", data_dir="data", batch_size=128, eval_batch_size=100,
             num_workers=4, seed=0, subset=0, calib_batch_size=100):
    test_meta, class_names = [], []
    if dataset == "cifar100":
        img_size, num_classes = 32, 100
        tr_tf, te_tf = _transforms(32, CIFAR_MEAN, CIFAR_STD)
        full_tr = datasets.CIFAR100(data_dir, train=True, download=True, transform=tr_tf)
        full_te = datasets.CIFAR100(data_dir, train=True, download=False, transform=te_tf)
        test = datasets.CIFAR100(data_dir, train=False, download=True, transform=te_tf)
        class_names = full_tr.classes
        tr_idx, val_idx = _split_indices(len(full_tr), VAL_SIZE)
        train, val, calib = Subset(full_tr, tr_idx), Subset(full_te, val_idx), Subset(full_te, tr_idx)
    elif dataset == "fake":
        # Offline stand-in with CIFAR shapes, for smoke tests only.
        img_size, num_classes = 32, 100
        tr_tf, te_tf = _transforms(32, CIFAR_MEAN, CIFAR_STD)
        n = 2000
        full_tr = datasets.FakeData(n, (3, 32, 32), num_classes, transform=tr_tf, random_offset=0)
        full_te = datasets.FakeData(n, (3, 32, 32), num_classes, transform=te_tf, random_offset=0)
        test = datasets.FakeData(500, (3, 32, 32), num_classes, transform=te_tf, random_offset=10**6)
        tr_idx, val_idx = _split_indices(n, 200)
        train, val, calib = Subset(full_tr, tr_idx), Subset(full_te, val_idx), Subset(full_te, tr_idx)
    elif dataset == "drone":
        img_size, num_classes, class_names = 64, 3, DRONE_CLASSES
        root = Path(data_dir) / "drone_crops"
        tr_tf, te_tf = _transforms(64, IMAGENET_MEAN, IMAGENET_STD)
        rows = read_manifest(root)
        by = {s: [r for r in rows if r["split"] == s] for s in ("train", "valid", "test")}
        train = ManifestDataset(root, by["train"], tr_tf)
        calib = ManifestDataset(root, by["train"], te_tf)
        val = ManifestDataset(root, by["valid"], te_tf)
        test = ManifestDataset(root, by["test"], te_tf)
        test_meta = by["test"]
    else:
        raise ValueError(dataset)

    if subset:
        train = Subset(train, list(range(min(subset, len(train)))))

    g = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    kw = dict(num_workers=num_workers, pin_memory=pin, persistent_workers=num_workers > 0)
    train_l = DataLoader(train, batch_size, shuffle=True, drop_last=True, generator=g, **kw)
    val_l = DataLoader(val, eval_batch_size, shuffle=False, **kw)
    test_l = DataLoader(test, eval_batch_size, shuffle=False, **kw)
    # Calibration: TRAINING images (never test), normalized, fixed order, static batch shape.
    calib_l = DataLoader(calib, calib_batch_size, shuffle=True, drop_last=True,
                         generator=torch.Generator().manual_seed(1234), num_workers=num_workers)
    return DataBundle(train_l, val_l, test_l, calib_l, num_classes, img_size, class_names, test_meta)


if __name__ == "__main__":
    # Download / verify once before parallel jobs start (two jobs extracting into data/ at once corrupt it).
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar100", choices=["cifar100", "drone", "fake"])
    p.add_argument("--data-dir", default="data")
    a = p.parse_args()
    d = get_data(a.dataset, a.data_dir, num_workers=0)
    print(f"{a.dataset}: train {len(d.train.dataset)}, val {len(d.val.dataset)}, test {len(d.test.dataset)}")
