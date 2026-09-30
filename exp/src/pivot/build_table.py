"""Build the master biomarker table the three pivot probes share.

One row per (dataset, image_id, source) where ``source`` is

    gt         biomarkers of the reference (observer-1) mask
    pred_test  biomarkers of ``runs/seg/<ds>/seed<S>/pred/mask`` (held-out test)
    pred_oof   biomarkers of ``runs/seg_oof/<ds>/pred/mask``    (out-of-fold train)

Everything is recomputed here rather than lifted from the repair runs'
``bio_before_*`` columns, because those were produced with a different
``fd_rotations`` and only cover 60 of the 200 FIVES test images.  One
``fd_rotations`` (default 5) for every row keeps FD comparable across sources.

The optic disc is detected **once per image from the fundus image alone**
(``vessel_mask=None``), so the disc -- and therefore zone B -- is identical for
the GT row and the prediction row of the same image and never leaks the
reference mask.

Per-(image, source) results are cached as JSON under
``results/pivot/cache/``; re-running only computes what is missing.

CLI
---
    python -m src.pivot.build_table --procs 6
    python -m src.pivot.build_table --datasets fives --sources gt,pred_test
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.pivot.common import (CACHE_DIR, DATASETS, PIVOT_DIR, binarize,
                              image_features, json_dump, load_fov, pred_paths,
                              prob_features, read_gray, records)

FD_ROTATIONS_DEFAULT = 5


# --------------------------------------------------------------------------
def fives_train_subset(recs: List[dict], n_per_class: int) -> List[dict]:
    """Deterministic class-balanced subset of the FIVES training split."""
    by: Dict[str, List[dict]] = {}
    for r in recs:
        if r["split"] != "train":
            continue
        by.setdefault(r["disease"], []).append(r)
    out = []
    for cls in sorted(by):
        out.extend(sorted(by[cls], key=lambda r: r["image_id"])[:n_per_class])
    return out


def plan(dataset: str, sources: Tuple[str, ...], seed: int,
         fives_train_n: int) -> List[Tuple[dict, Tuple[str, ...]]]:
    """[(record, sources to compute for it)] for one dataset."""
    recs = records(dataset)
    jobs: List[Tuple[dict, Tuple[str, ...]]] = []
    for r in recs:
        if dataset == "fives" and r["split"] == "train":
            continue
        want = []
        for s in sources:
            if s == "gt":
                want.append(s)
            elif s == "pred_test" and r["split"] == "test":
                want.append(s)
            elif s == "pred_oof" and r["split"] == "train":
                want.append(s)
        if want:
            jobs.append((r, tuple(want)))
    if dataset == "fives" and fives_train_n > 0:
        for r in fives_train_subset(recs, fives_train_n):
            want = [s for s in sources if s in ("gt", "pred_oof")]
            if want:
                jobs.append((r, tuple(want)))
    # drop jobs whose prediction file does not exist
    out = []
    for r, want in jobs:
        keep = []
        for s in want:
            if s == "gt":
                keep.append(s)
                continue
            pp, mp = pred_paths(dataset, r["image_id"], s, seed)
            if os.path.exists(pp) and os.path.exists(mp):
                keep.append(s)
        if keep:
            out.append((r, tuple(keep)))
    return out


# --------------------------------------------------------------------------
def cache_path(dataset: str, image_id: str, source: str, seed: int,
               fd_rot: int) -> str:
    tag = f"seed{seed}" if source == "pred_test" else "oof" if source == "pred_oof" else "gt"
    return os.path.join(CACHE_DIR, f"fd{fd_rot}", dataset, f"{source}_{tag}",
                        image_id + ".json")


def _one_image(args) -> Tuple[str, int, str]:
    dataset, rec, sources, seed, fd_rot = args
    from src.bio.biomarkers import compute_all
    from src.bio.disc import locate_optic_disc
    from src.data.datasets import read_image

    todo = [s for s in sources
            if not os.path.exists(cache_path(dataset, rec["image_id"], s, seed, fd_rot))]
    if not todo:
        return (rec["image_id"], 0, "cached")

    t0 = time.perf_counter()
    image = read_image(rec["image_path"])
    native_hw = image.shape[:2]
    fov = load_fov(rec, native_hw)
    ifeat = image_features(image, fov)

    try:
        disc = locate_optic_disc(image, fov, vessel_mask=None)
    except Exception:                                            # noqa: BLE001
        disc = None

    base = dict(dataset=dataset, image_id=rec["image_id"], split=rec["split"],
                subject_id=rec.get("subject_id"), disease=rec.get("disease"),
                fd_rotations=fd_rot, **ifeat)
    if disc is not None:
        base.update(disc_cx=float(disc.cx), disc_cy=float(disc.cy),
                    disc_r=float(disc.r), disc_confident=int(bool(disc.confident)),
                    disc_fov_diameter=float(disc.fov_diameter),
                    disc_r_rel=float(disc.r) / max(float(disc.fov_diameter), 1e-6))

    for s in todo:
        row = dict(base, source=s, seed=(seed if s == "pred_test" else -1))
        if s == "gt":
            mask = binarize(read_gray(rec["label_path"]))
            if mask.shape != tuple(native_hw):
                import cv2
                mask = cv2.resize(mask.astype(np.uint8) * 255,
                                  (native_hw[1], native_hw[0]),
                                  interpolation=cv2.INTER_NEAREST) > 127
            mask = mask & (fov > 0)
        else:
            pp, mp = pred_paths(dataset, rec["image_id"], s, seed)
            mask = binarize(read_gray(mp)) & (fov > 0)
            prob = np.load(pp).astype(np.float32)
            row.update(prob_features(prob, mask, fov, image))
        row["fg_px"] = int(mask.sum())
        bio = compute_all(mask, fov, image=None, disc=disc,
                          fd_rotations=fd_rot)
        bio.pop("pvbm_version", None)
        row.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in bio.items()})
        json_dump(row, cache_path(dataset, rec["image_id"], s, seed, fd_rot))

    return (rec["image_id"], len(todo), "%.1fs" % (time.perf_counter() - t0))


# --------------------------------------------------------------------------
def assemble(fd_rot: int, out_csv: str) -> "object":
    import pandas as pd

    root = os.path.join(CACHE_DIR, f"fd{fd_rot}")
    rows = []
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            import json as _json
            with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                rows.append(_json.load(fh))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["dataset", "source", "image_id"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--sources", default="gt,pred_test,pred_oof")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fives_train_n", type=int, default=50,
                    help="images per FIVES disease class taken from the 600 "
                         "training images (0 = none)")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--fd_rotations", type=int, default=FD_ROTATIONS_DEFAULT)
    ap.add_argument("--out", default=os.path.join(PIVOT_DIR, "bio_master.csv"))
    ap.add_argument("--assemble_only", action="store_true")
    args = ap.parse_args(argv)

    fd = int(args.fd_rotations)
    if not args.assemble_only:
        srcs = tuple(s.strip() for s in args.sources.split(",") if s.strip())
        tasks = []
        for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
            for rec, want in plan(ds, srcs, args.seed, args.fives_train_n):
                tasks.append((ds, rec, want, args.seed, fd))
        n_pending = sum(
            1 for ds, rec, want, sd, f in tasks for s in want
            if not os.path.exists(cache_path(ds, rec["image_id"], s, sd, f)))
        print("[plan] %d images, %d (image,source) still to compute"
              % (len(tasks), n_pending), flush=True)

        from concurrent.futures import ProcessPoolExecutor, as_completed

        t0, done = time.perf_counter(), 0
        with ProcessPoolExecutor(max_workers=int(args.procs)) as ex:
            futs = [ex.submit(_one_image, t) for t in tasks]
            for fu in as_completed(futs):
                iid, n, note = fu.result()
                done += 1
                if done % 20 == 0 or n:
                    print("[%4d/%4d] %-16s +%d %s  (%.0f s elapsed)"
                          % (done, len(tasks), iid, n, note,
                             time.perf_counter() - t0), flush=True)

    df = assemble(fd, args.out)
    print("[out] %s  rows=%d" % (args.out, len(df)), flush=True)
    try:
        print(df.groupby(["dataset", "source"]).size())
    except Exception:                                            # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
