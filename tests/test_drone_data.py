"""Drone crop builder on a synthetic YOLO dataset: crops, background sampling, leakage detection."""
import csv
import json

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("imagehash")
import drone_data as D  # noqa: E402


def make_frame(path, seed, boxes, label_path):
    rng = np.random.default_rng(seed)
    img = (rng.random((640, 640, 3)) * 255).astype(np.uint8)
    for _, cx, cy, w, h in boxes:
        x0, y0 = int((cx - w / 2) * 640), int((cy - h / 2) * 640)
        img[y0:y0 + int(h * 640), x0:x0 + int(w * 640)] = 255
    Image.fromarray(img).save(path)
    label_path.write_text("\n".join(f"{c} {cx} {cy} {w} {h}" for c, cx, cy, w, h in boxes))


@pytest.fixture
def yolo(tmp_path):
    src = tmp_path / "src"
    (src / "data.yaml").parent.mkdir(parents=True)
    (src / "data.yaml").write_text("nc: 2\nnames: ['Bird', 'Drone']\n")
    for split in ("train", "valid", "test"):
        (src / split / "images").mkdir(parents=True)
        (src / split / "labels").mkdir(parents=True)
    boxes = [(0, 0.3, 0.3, 0.02, 0.02), (1, 0.7, 0.7, 0.1, 0.08)]
    for i in range(3):
        make_frame(src / "train" / "images" / f"f{i}.jpg", i, boxes, src / "train" / "labels" / f"f{i}.txt")
    make_frame(src / "valid" / "images" / "v0.jpg", 10, boxes, src / "valid" / "labels" / "v0.txt")
    # test: one exact copy of a train frame (leak) and one new frame
    make_frame(src / "test" / "images" / "leak.jpg", 0, boxes, src / "test" / "labels" / "leak.txt")
    make_frame(src / "test" / "images" / "new.jpg", 99, boxes, src / "test" / "labels" / "new.txt")
    return src


def test_build_crops_and_detect_leak(yolo, tmp_path):
    out = tmp_path / "crops"
    rep = D.build(yolo, out, pad=0.2, size=64, max_dist=4)
    rows = list(csv.DictReader(open(out / "manifest.csv")))
    assert {r["cls"] for r in rows} == {"drone", "bird", "background"}
    for r in rows:
        assert Image.open(out / r["path"]).size == (64, 64)
    bird = next(r for r in rows if r["cls"] == "bird")
    assert float(bird["box_px"]) == pytest.approx(12.8, abs=0.1)  # 0.02 * 640
    assert rep["frames"]["test"]["n"] == 2
    assert rep["frames"]["test"]["exact_dup"] >= 1  # the copied frame is caught
    # object crops of the copied frame are flagged; background patches are resampled at random spots
    leak_rows = [r for r in rows if r["split"] == "test" and "leak" in r["path"] and r["cls"] != "background"]
    assert leak_rows and all(r["near_dup"] == "1" for r in leak_rows)
    assert json.loads((out / "leakage.json").read_text())["counts"]["train"]["drone"] == 3


def test_class_map_from_names():
    assert D.class_map(["Bird", "Drone"]) == {0: 1, 1: 0}
    assert D.class_map(None, "0=drone,1=bird") == {0: 0, 1: 1}


def test_background_never_overlaps_boxes():
    import random
    img = Image.new("RGB", (640, 640))
    img.paste((255, 255, 255), (270, 270, 370, 370))  # a 100px white object at the center
    boxes = [(0, 320, 320, 100, 100)]
    patches = D.sample_background(img, boxes, 5, random.Random(0), 64, 0.2, tries=500)
    assert len(patches) == 5
    assert all(max(p.getextrema()[0]) == 0 for p in patches)  # no white pixels: no overlap


def test_min_hamming():
    d = D.min_hamming([0b1111, 0], [0b0111, 1 << 40])
    assert d.tolist() == [1, 1]


def test_drone_loader_and_metrics(yolo, tmp_path):
    import torch
    from data import DRONE_CLASSES, get_data
    from models import build_model
    from report import drone_metrics
    D.build(yolo, tmp_path / "drone_crops", max_dist=4)
    data = get_data("drone", str(tmp_path), batch_size=4, eval_batch_size=4, num_workers=0)
    assert (data.num_classes, data.img_size) == (3, 64)
    x, y = next(iter(data.test))
    assert x.shape[1:] == (3, 64, 64)
    with torch.no_grad():
        assert build_model("resnet18", 3).eval()(x).shape == (len(x), 3)
    labels = torch.tensor([int(r["label"]) for r in data.test_meta])
    logits = torch.nn.functional.one_hot(labels, 3).float()
    bird = DRONE_CLASSES.index("bird")
    logits[labels == bird] = torch.tensor([5.0, 0, 0])  # every bird predicted as drone
    m = drone_metrics(logits, labels, data.test_meta, DRONE_CLASSES)
    assert m["bird_to_drone_rate"] == 1.0
    assert m["recall"]["drone"] == 1.0 and m["recall"]["bird"] == 0.0
    assert m["by_size"]["<16px"]["acc"] == 0.0     # the 12.8px objects are the birds
    assert m["by_size"][">32px"]["acc"] == 100.0   # the 64px objects are the drones
