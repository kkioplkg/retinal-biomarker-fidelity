"""E1 -- the measurement scale of each dataset: FOV diameter, not frame size.

Why this script exists
----------------------
Fundus datasets differ in how much black surround they keep around the retina,
so the frame size is a misleading proxy for how many pixels the measurement
chain actually has to work with.  What the measurement-fidelity mechanism
depends on is the **number of pixels across the retina**, i.e. the equivalent
diameter of the field-of-view mask, ``2 sqrt(A / pi)``.  The gap is large: a
MESSIDOR-derived frame at 1440x960 has a retina only ~909 px across (frame/FOV
1.58), below CHASE\\_DB1, while a tightly cropped 1280x1280 frame has ~1238 px
(frame/FOV 1.03).

This script measures it uniformly for every dataset with reference masks, so
the datasets table in the manuscript quotes a measured number rather than a
frame size.

Where the numbers come from
---------------------------
For the four audit datasets the FOV area is already measured per image in
``results/pivot/bio_master.csv`` (``feat_fov_area_px``, computed at native
resolution by the same loader), so no image is re-read.  For STARE and for the
two replication cohorts the FOV masks are read through
``src.data.datasets.load_dataset`` and measured here.

CLI
---
    cd exp
    python -m src.pivot.e1_fov_scale
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR

FROM_MASTER = ("drive", "chasedb1", "hrf", "fives")
FROM_LOADER = ("stare", "fundusavseg", "maplesdr")


def _diam(area: np.ndarray) -> np.ndarray:
    return 2.0 * np.sqrt(np.asarray(area, float) / np.pi)


def from_master(path: str) -> List[dict]:
    m = pd.read_csv(path, low_memory=False)
    m = m[m["source"] == "gt"]
    rows = []
    for ds, g in m.groupby("dataset"):
        if ds not in FROM_MASTER:
            continue
        a = g["feat_fov_area_px"].to_numpy(float)
        d = _diam(a)
        h = g["feat_h"].to_numpy(float)
        w = g["feat_w"].to_numpy(float)
        rows.append(_row(str(ds), h, w, d, len(g),
                         "results/pivot/bio_master.csv (source=gt, "
                         "feat_fov_area_px at native resolution)"))
    return rows


def from_loader(name: str) -> dict:
    import cv2

    from src.data.datasets import load_dataset
    from src.pivot.common import binarize, read_gray

    recs = load_dataset(name)
    hs, ws, ds = [], [], []
    for r in recs:
        f = binarize(read_gray(str(r["fov_path"]))).astype(np.uint8)
        img_hw = None
        try:
            img_hw = read_gray(str(r["image_path"])).shape[:2]
        except Exception:
            pass
        if img_hw and f.shape[:2] != tuple(img_hw):
            f = (cv2.resize(f * 255, (img_hw[1], img_hw[0]),
                            interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
        h, w = (img_hw if img_hw else f.shape[:2])
        hs.append(h)
        ws.append(w)
        ds.append(_diam(f.sum()))
    return _row(name, np.array(hs, float), np.array(ws, float),
                np.array(ds, float), len(recs),
                f"src.data.datasets.load_dataset('{name}') FOV masks, "
                "equivalent diameter 2*sqrt(area/pi) at native resolution")


def _row(ds: str, h: np.ndarray, w: np.ndarray, d: np.ndarray, n: int,
         source: str) -> dict:
    frames = sorted({f"{int(b)}x{int(a)}" for a, b in zip(h, w)},
                    key=lambda s: -int(s.split("x")[0]))
    return dict(
        dataset=ds, n=int(n),
        frames="; ".join(frames[:3]) + ("; ..." if len(frames) > 3 else ""),
        frame_h_median=float(np.median(h)), frame_w_median=float(np.median(w)),
        fov_diam_median=float(np.median(d)),
        fov_diam_min=float(np.min(d)), fov_diam_max=float(np.max(d)),
        frame_over_fov=float(np.median(np.maximum(h, w) / d)),
        source=source)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    ap.add_argument("--master", default=os.path.join(PIVOT_DIR, "bio_master.csv"))
    args = ap.parse_args(argv)

    rows: List[dict] = from_master(args.master)
    for name in FROM_LOADER:
        try:
            rows.append(from_loader(name))
        except Exception as e:                     # pragma: no cover
            print(f"[fov][skip] {name}: {e}")

    order = {d: i for i, d in enumerate(
        ("drive", "stare", "chasedb1", "fundusavseg", "maplesdr", "fives", "hrf"))}
    out = pd.DataFrame(rows).sort_values(
        "dataset", key=lambda s: s.map(lambda v: order.get(v, 99)))
    p = os.path.join(args.out_dir, "dataset_scale.csv")
    out.to_csv(p, index=False)
    print("wrote", p, out.shape)
    with pd.option_context("display.width", 200):
        print(out.drop(columns=["source"]).round(1).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
