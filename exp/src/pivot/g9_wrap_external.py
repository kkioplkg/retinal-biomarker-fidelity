"""Review point 9(a) -- wrap a THIRD-PARTY vessel pipeline's output into the
layout ``src/pivot/p5_eval.py`` expects, so the C1 audit runs on it unchanged.

An external pipeline (VascX, AutoMorph, LWNet, ...) writes binary masks its own
way: its own file names, its own canvas (VascX works on a 1024x1024 fovea-
centred crop), sometimes its own polarity.  This module does exactly one thing:
it maps those files onto **this project's test split**, resamples each mask back
to the image's **native resolution**, intersects it with the project's FOV mask
(the same one every other arm uses), and writes

    <out>/mask/<key>.png        uint8 {0,255}, native resolution
    <out>/manifest.csv          the p5_eval / seg.infer schema
    <out>/pixel_metrics.csv     Dice + clDice in FOV, per image
    <out>/infer_meta.json       provenance: pipeline, weights, geometry

after which the usual CPU stage

    python -m src.pivot.p5_eval bio --dataset <ds> --tag <tag> --root <out>

produces ``bio.csv`` with the identical estimator path as every internal arm.

No probability maps are written: third-party pipelines generally ship a
thresholded mask, and the C1 audit only consumes the binary mask.

Matching rule
-------------
A prediction file is matched to a test record by **file stem**, after stripping
a configurable list of suffixes (``_vessels``, ``_pred``, ...).  FIVES numbers
its train and test splits independently, so a bare stem such as ``100_D`` is
ambiguous across splits; only the *test* records are considered here, which
makes the stem unique again.  A record with no prediction file is an error, not
a silent drop -- a partial arm would bias every pooled statistic.

CLI
---
    python -m src.pivot.g9_wrap_external \\
        --dataset drive --pipeline lwnet \\
        --src E:/.../third_party/lwnet/out/drive --ext png \\
        --out runs/pivot/r2/lwnet/drive/pred \\
        --note "DRIVE-trained W-Net checkpoint, applied zero-shot"
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

cv2.setNumThreads(0)

#: suffixes an external tool commonly appends to the image stem
STRIP_SUFFIXES = ("_vessels", "_vessel", "_pred", "_prediction", "_mask",
                  "_seg", "_segmentation", "_bin", "_binary", "_out",
                  "_artery_vein", "_av")


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(str(path)))[0]


def _canon_stem(s: str) -> str:
    s = str(s)
    changed = True
    while changed:
        changed = False
        for suf in STRIP_SUFFIXES:
            if s.lower().endswith(suf):
                s = s[: -len(suf)]
                changed = True
    return s


def index_predictions(src: str, ext: str) -> Dict[str, str]:
    """{canonical stem -> file path} over ``src`` (recursively)."""
    exts = tuple("." + e.strip().lower().lstrip(".") for e in ext.split(",") if e.strip())
    out: Dict[str, str] = {}
    for root, _dirs, files in os.walk(src):
        for fn in files:
            if not fn.lower().endswith(exts):
                continue
            p = os.path.join(root, fn)
            for key in {_stem(fn), _canon_stem(_stem(fn))}:
                out.setdefault(key, p)
                # tolerate zero padding differences (01 vs 1)
                if key.isdigit():
                    out.setdefault(str(int(key)), p)
    return out


def read_mask(path: str, invert: bool, thresh: int) -> np.ndarray:
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError("unreadable prediction %s" % path)
    if a.ndim == 3:
        # a colour artery/vein map: vessel = any non-background channel
        a = a[..., :3].max(axis=2)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        m = float(a.max()) if a.size else 1.0
        a = (a / m * 255.0).astype(np.uint8) if m > 0 else a.astype(np.uint8)
    m = (a > int(thresh)).astype(np.uint8)
    if invert:
        m = 1 - m
    return m


def to_native(mask: np.ndarray, native_hw) -> np.ndarray:
    if mask.shape[:2] == tuple(native_hw):
        return mask
    return (cv2.resize(mask * 255, (native_hw[1], native_hw[0]),
                       interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pipeline", required=True,
                    help="e.g. vascx / automorph / lwnet -- goes into the tag")
    ap.add_argument("--src", required=True,
                    help="directory holding the pipeline's mask files")
    ap.add_argument("--ext", default="png,tif,tiff,jpg,gif,bmp")
    ap.add_argument("--out", default=None,
                    help="default runs/pivot/r2/<pipeline>/<dataset>/pred")
    ap.add_argument("--split", default="test", choices=["test", "train", "val"])
    ap.add_argument("--threshold", type=int, default=127)
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--no-fov", action="store_true",
                    help="do NOT intersect with the project FOV mask "
                         "(default is to intersect, as every internal arm does)")
    ap.add_argument("--weights", default="", help="weights id, for provenance")
    ap.add_argument("--note", default="")
    ap.add_argument("--allow-missing", action="store_true",
                    help="write what matched instead of failing (diagnostics only)")
    a = ap.parse_args(argv)

    os.chdir(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "..")))
    import pandas as pd

    from src.seg import data as segdata
    from src.seg.evaluate import read_fov, read_gt
    from src.seg.infer import load_native_fov
    from src.topo.metrics import cldice as cldice_fn
    from src.topo.metrics import dice as dice_fn

    ds = segdata.canon(a.dataset)
    recs = segdata.get_records(ds)
    tr, va, te, _ = segdata.make_splits(recs, segdata.DEFAULT_SPLIT_SEED)
    subset = {"test": te, "train": tr, "val": va}[a.split]

    preds = index_predictions(a.src, a.ext)
    print("[wrap] %s: %d %s records, %d prediction files indexed under %s"
          % (ds, len(subset), a.split, len(set(preds.values())), a.src), flush=True)

    od = a.out or os.path.join("runs", "pivot", "r2", a.pipeline, ds, "pred")
    mask_dir = os.path.join(od, "mask")
    os.makedirs(mask_dir, exist_ok=True)

    rows: List[dict] = []
    px: List[dict] = []
    missing: List[str] = []
    for rec in subset:
        key = _stem(rec["image_path"])
        cand = None
        for probe in (key, _canon_stem(key), str(rec.get("image_id", "")),
                      _canon_stem(str(rec.get("image_id", "")))):
            if probe and probe in preds:
                cand = preds[probe]
                break
        if cand is None:
            missing.append(key)
            continue

        img = cv2.imread(str(rec["image_path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("unreadable image %s" % rec["image_path"])
        native_hw = (int(img.shape[0]), int(img.shape[1]))
        m = to_native(read_mask(cand, a.invert, a.threshold), native_hw)
        fov = load_native_fov({"fov_path": rec.get("fov_path"),
                               "image_path": rec["image_path"]}, native_hw)
        if not a.no_fov:
            m = (m & (fov > 0)).astype(np.uint8)

        mp = os.path.join(mask_dir, key + ".png")
        cv2.imwrite(mp, m * 255)
        row = {"dataset": ds, "seed": a.pipeline, "split": a.split, "image": key,
               "subject_id": rec.get("subject_id"),
               "image_path": os.path.abspath(str(rec["image_path"])),
               "label_path": str(rec.get("label_path") or ""),
               "label2_path": str(rec.get("label2_path") or ""),
               "fov_path": str(rec.get("fov_path") or ""),
               "prob_path": "", "mask_path": os.path.abspath(mp),
               "native_h": native_hw[0], "native_w": native_hw[1],
               "infer_hw": "external", "patch": -1, "stride": -1,
               "threshold": a.threshold,
               "pred_fg_frac": float(m.sum()) / max(1.0, float(fov.sum())),
               "src_pred": os.path.abspath(cand), "seconds": 0.0}
        rows.append(row)

        gt = read_gt(row, native_hw)
        fv = read_fov(row, native_hw)
        px.append({"image": key,
                   "dice": float(dice_fn(m, gt, fv)),
                   "cldice": float(cldice_fn(m, gt, fv)),
                   "pred_fg_frac": row["pred_fg_frac"]})
        print("  %s dice=%.4f cldice=%.4f fg=%.4f  <- %s"
              % (key, px[-1]["dice"], px[-1]["cldice"], row["pred_fg_frac"],
                 os.path.basename(cand)), flush=True)

    if missing:
        msg = ("[wrap] %d of %d %s records have NO prediction file: %s"
               % (len(missing), len(subset), a.split, missing[:12]))
        if not a.allow_missing:
            print(msg, flush=True)
            raise SystemExit("refusing to write a partial arm; fix the source "
                             "directory or pass --allow-missing for a probe")
        print(msg + "  (continuing: --allow-missing)", flush=True)
    if not rows:
        raise SystemExit("nothing matched")

    with open(os.path.join(od, "manifest.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    pxdf = pd.DataFrame(px)
    pxdf.to_csv(os.path.join(od, "pixel_metrics.csv"), index=False)

    meta = {"review_point": "9a-external-pipeline-zero-shot",
            "dataset": ds, "pipeline": a.pipeline, "weights": a.weights,
            "split": a.split, "n_images": len(rows),
            "source_dir": os.path.abspath(a.src),
            "threshold": a.threshold, "inverted": bool(a.invert),
            "fov_intersected": not a.no_fov,
            "resampled_to_native": "nearest neighbour where the pipeline's "
                                   "canvas differs from the native frame",
            "mean_dice": float(pxdf["dice"].mean()),
            "mean_cldice": float(pxdf["cldice"].mean()),
            "note": a.note}
    with open(os.path.join(od, "infer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
