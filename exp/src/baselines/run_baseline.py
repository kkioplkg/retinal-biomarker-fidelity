"""Run one S6 repair baseline over a dataset and score it.

    python -m src.baselines.run_baseline --method geometric \
        --dataset drive --pred_dir runs/seg/drive/seed0/pred \
        --out runs/repair/drive/seed0/geometric

Writes
------
``<out>/mask/<image>.png``      the repaired mask (uint8, 0 / 255)
``<out>/edges/<image>.npz``     the accepted edges (path pixels + radius), so
                                TRR / FCR can be recomputed without re-running
``<out>/per_image.csv``         one row per image, in the **same schema**
                                ``src.rigr.run_rigr`` writes (schema item 1):
                                column ``image`` (not ``image_id``), metric
                                prefixes ``before_*`` / ``after_*`` /
                                ``delta_*`` (not ``base_*`` / ``rep_*`` /
                                ``d_*``), plus ``dataset, seed, method,
                                TRR_recall, TRR_precision, FCR, n_should,
                                n_accepted, n_true, n_matched`` and, when
                                ``--bio`` is not ``none``, the same
                                ``bio_*`` / ``macro_mae_before`` /
                                ``macro_mae_after`` / ``macro_mae_delta``
                                biomarker columns ``run_rigr.py`` writes
                                (schema item 2; both call the shared
                                ``src.eval.biomarker_eval`` helper so the two
                                producers compute biomarker error identically)
``<out>/summary.json``          dataset-level means and the run configuration

Inputs
------
``--pred_dir`` points at an S2 prediction directory laid out as
``pred/prob/<key>.npy`` and ``pred/mask/<key>.png``, where ``<key>`` is the
image path's basename without its extension (see :func:`pred_key`) -- not
necessarily ``rec["image_id"]``.  While S2 is
still training, ``--synth`` builds the inputs instead: ``M_hat`` = GT with
``--n_cuts`` random capsule severances, ``P`` = blurred ``M_hat``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.data.datasets import load_dataset, read_binary, read_image
from src.topo import skeleton as sk
from src.topo.metrics import evaluate_all
from src.eval.trr_fcr import CandidateEdge, count_repairable_events, trr_fcr
from src.eval.biomarker_eval import (
    auto_disc, bio_columns, biomarker_row, load_gt_scales, pipes_for,
)
from src.baselines.common import METHODS, get_method
from src.baselines.synth import synthesize

__all__ = ["run_dataset", "main"]


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------


def _save_mask(path: Path, mask: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((sk.as_bool(mask).astype(np.uint8) * 255)).save(path)


def _save_edges(path: Path, edges: List[CandidateEdge]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not edges:
        np.savez_compressed(path, n=0)
        return
    np.savez_compressed(
        path,
        n=len(edges),
        paths=np.concatenate([e.path for e in edges], axis=0),
        lengths=np.asarray([len(e.path) for e in edges], dtype=np.int64),
        radius=np.asarray([e.radius for e in edges], dtype=np.float64),
        score=np.asarray([e.score for e in edges], dtype=np.float64),
    )


def pred_key(rec: Dict[str, Any]) -> str:
    """The key ``src.seg.infer`` names its output files by.

    Every other consumer of a prediction directory in this project
    (``src.seg.infer``, ``src.seg.data.load_sample``,
    ``src.rigr.run_rigr.load_case``, ``src.rigr.build_data``,
    ``src.rigr.synth_cuts.load_oof_pairs``) derives the file key as the image
    path's basename without its extension.  ``rec["image_id"]`` is *not* always
    the same string: FIVES prefixes its ids with the split (``test_100_D``)
    because its ``train/``, ``val/`` and ``test/`` folders reuse the same
    numbers (see ``exp/DECISIONS.md``), while the prediction files are named
    ``100_D.png``.  Keying on ``image_id`` therefore found nothing for FIVES.
    """
    return os.path.splitext(os.path.basename(str(rec["image_path"])))[0]


def _load_pred(pred_dir: Path, image_id, shape):
    """Read ``(prob, mask)`` for one image from an S2 prediction directory.

    ``image_id`` may be a single key or an ordered list of candidate keys (see
    :func:`pred_key`); the first one that resolves wins.
    """
    keys = [image_id] if isinstance(image_id, str) else list(image_id)
    keys = [k for i, k in enumerate(keys) if k and k not in keys[:i]]

    prob = None
    mask = None
    for key in keys:
        for ext in (".npy", ".npz"):
            p = pred_dir / "prob" / (key + ext)
            if p.exists():
                arr = np.load(p)
                prob = np.asarray(arr["prob"] if ext == ".npz" else arr,
                                  dtype=np.float32)
                break
        if prob is None:
            for ext in (".png", ".tif"):
                p = pred_dir / "prob" / (key + ext)
                if p.exists():
                    a = np.asarray(read_image(p), dtype=np.float32)
                    prob = a / (255.0 if a.max() > 1.0 else 1.0)
                    break

        for ext in (".png", ".tif", ".npy"):
            p = pred_dir / "mask" / (key + ext)
            if p.exists():
                mask = np.load(p) > 0.5 if ext == ".npy" else read_binary(p)
                break
        if mask is not None or prob is not None:
            break

    if mask is None and prob is not None:
        mask = prob >= 0.5
    if mask is None:
        raise FileNotFoundError(
            f"no prediction for any of {keys!r} under {pred_dir}")
    return prob, sk.as_bool(mask)


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------


def run_dataset(
    method: str,
    dataset: str,
    out_dir: str,
    pred_dir: Optional[str] = None,
    split: Optional[str] = "test",
    synth: bool = False,
    n_cuts: int = 12,
    seed: int = 0,
    limit: Optional[int] = None,
    image_ids: Optional[List[str]] = None,
    save_masks: bool = True,
    method_kw: Optional[Dict[str, Any]] = None,
    bio: str = "both",
    fd_rotations: int = 8,
) -> "Any":
    """Repair every image of one dataset split and return the per-image table.

    ``bio`` (``"none"/"skan"/"pvbm"/"both"``) controls the biomarker macro-MAE
    columns (schema item 2): when not ``"none"``, GT / pre-repair / post-repair
    biomarkers are computed with the exact same
    ``src.eval.biomarker_eval.biomarker_row`` call ``src.rigr.run_rigr`` makes,
    and standardised against the same single-source-of-truth sigma
    (``results/gateA_biomarker_scales.csv``, schema item 3) -- see that
    module's docstring for the native-resolution contract this relies on.
    """
    import pandas as pd

    fn = get_method(method)
    kw = dict(method_kw or {})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pipes = pipes_for(bio)
    sigma = load_gt_scales(dataset) if bio != "none" else {}

    recs = load_dataset(dataset, split=split)
    if image_ids:
        wanted = set(image_ids)
        # accept either the dataset's own id or the prediction-file key
        recs = [r for r in recs
                if r["image_id"] in wanted or pred_key(r) in wanted]
    if limit:
        recs = recs[: int(limit)]
    if not recs:
        raise SystemExit(f"no images selected for {dataset}/{split}")

    pdir = Path(pred_dir) if pred_dir else None
    if pdir is None and not synth:
        raise SystemExit("give --pred_dir, or --synth to build inputs from the GT")

    rows: List[Dict[str, Any]] = []
    for k, r in enumerate(recs):
        # ``iid`` is the dataset's own id (kept for traceability); ``key`` is
        # the prediction-file key, which is also the ``image`` column every
        # other producer writes -- so Tab.2's paired per-image merge against
        # ``run_rigr``'s per_image.csv and ``results/seg_per_image.csv`` lines
        # up on FIVES too.  For every other dataset the two are identical.
        iid = r["image_id"]
        key = pred_key(r)
        image = np.asarray(read_image(r["image_path"]))
        gt = read_binary(r["label_path"])
        fov = read_binary(r["fov_path"]) if r.get("fov_path") else np.ones(gt.shape, bool)

        if synth:
            mhat, prob, cuts = synthesize(gt, fov, n_cuts=n_cuts, seed=seed + k)
            n_verified = int(sum(bool(c.get("verified")) for c in cuts))
        else:
            prob, mhat = _load_pred(pdir, [key, iid], gt.shape)
            n_verified = -1

        t0 = time.time()
        res = fn(image, prob, mhat, fov, **kw)
        dt = time.time() - t0

        base = evaluate_all(prob, mhat, gt, fov)
        rep = evaluate_all(prob, res.mask, gt, fov)
        events = count_repairable_events(mhat, gt, fov)
        tf = trr_fcr(res.edges, mhat, gt, fov, events=events)

        # schema item 1: column ``image`` (not ``image_id``), and
        # before_*/after_*/delta_* metric prefixes -- the same per_image.csv
        # schema src.rigr.run_rigr writes.
        row: Dict[str, Any] = dict(
            dataset=dataset, split=split or "all", image=key, image_id=iid,
            subject_id=r.get("subject_id"), method=method,
            input="synth" if synth else "pred", seed=seed,
            n_cuts=n_cuts if synth else -1, n_cuts_verified=n_verified,
            seconds=round(dt, 3), n_accepted_edges=len(res.edges),
        )
        row.update({f"before_{m}": v for m, v in base.items()})
        row.update({f"after_{m}": v for m, v in rep.items()})
        row.update({f"delta_{m}": rep[m] - base[m] for m in base})
        row.update({k2: v for k2, v in tf.items()})
        row.update({f"info_{k2}": v for k2, v in res.info.items()
                    if isinstance(v, (int, float, str, bool))})

        # schema item 2: the same src.bio.biomarkers pipeline run_rigr.py
        # calls, via the shared src.eval.biomarker_eval helper.
        if bio != "none":
            disc = auto_disc(image, fov, vessel_mask=mhat)
            g = biomarker_row(gt, fov, image, disc=disc, pipelines=bio,
                              fd_rotations=fd_rotations)
            b = biomarker_row(mhat, fov, image, disc=disc, pipelines=bio,
                              fd_rotations=fd_rotations)
            a = biomarker_row(res.mask, fov, image, disc=disc, pipelines=bio,
                              fd_rotations=fd_rotations)
            row.update(bio_columns(g, b, a, sigma, pipes))

        rows.append(row)

        if save_masks:
            _save_mask(out / "mask" / f"{key}.png", res.mask)
            _save_edges(out / "edges" / f"{key}.npz", res.edges)

        print(f"[{k+1}/{len(recs)}] {key:>14}  clDice {base['cldice']:.4f}->{rep['cldice']:.4f}"
              f"  TRRrec {tf['TRR_recall']:.3f}  FCR {tf['FCR']:.3f}"
              f"  edges {len(res.edges):3d}  {dt:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    csv = out / "per_image.csv"
    df.to_csv(csv, index=False)

    num = df.select_dtypes("number")
    summary = dict(
        method=method, dataset=dataset, split=split or "all",
        input="synth" if synth else "pred", pred_dir=str(pdir) if pdir else None,
        n_images=len(df), seed=seed, n_cuts=n_cuts if synth else None,
        method_kw=kw, csv=str(csv),
        means={c: (None if not np.isfinite(num[c]).any() else float(np.nanmean(num[c])))
               for c in num.columns},
    )
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    m = summary["means"]
    print(f"\n== {method} / {dataset} ({len(df)} images) ==")
    print(f"  clDice   {m['before_cldice']:.4f} -> {m['after_cldice']:.4f}  (d {m['delta_cldice']:+.4f})")
    print(f"  Dice     {m['before_dice']:.4f} -> {m['after_dice']:.4f}  (d {m['delta_dice']:+.4f})")
    print(f"  BCS      {m['before_bcs']:.4f} -> {m['after_bcs']:.4f}  (d {m['delta_bcs']:+.4f})")
    print(f"  beta0err {m['before_beta0_err']:.2f} -> {m['after_beta0_err']:.2f}")
    if "macro_mae_before" in m and m["macro_mae_before"] is not None:
        print(f"  macroMAE {m['macro_mae_before']:.4f} -> {m['macro_mae_after']:.4f}"
              f"  (d {m['macro_mae_delta']:+.4f})")
    print(f"  TRR_recall {m['TRR_recall']:.4f}   TRR_precision {m['TRR_precision']:.4f}"
          f"   FCR {m['FCR']:.4f}")
    print(f"  edges/img  {m['n_accepted_edges']:.1f}   events/img {m['n_should']:.1f}")
    print(f"  -> {csv}")
    return df


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.baselines.run_baseline",
        description="Run a repair baseline over a dataset and score it.")
    ap.add_argument("--method", required=True,
                    choices=list(METHODS) + ["evapore"],
                    help="evapore = deprecated alias for evapore_e2e")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred_dir", default=None,
                    help="S2 prediction dir with prob/ and mask/ subdirectories")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--synth", action="store_true",
                    help="build M_hat/P from the GT with synthetic capsule cuts")
    ap.add_argument("--n_cuts", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--image_ids", default=None,
                    help="comma-separated subset of image ids")
    ap.add_argument("--no_save_masks", action="store_true")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint for the learned baselines (rnca / evapore)")
    ap.add_argument("--device", default="cpu",
                    help="torch device for the learned baselines, e.g. cuda:1")
    ap.add_argument("--opt", default=None,
                    help="extra method kwargs as k=v[,k=v...] (ints/floats/bools parsed)")
    ap.add_argument("--bio", default="both", choices=["none", "skan", "pvbm", "both"],
                    help="biomarker macro-MAE pipelines (schema item 2); "
                         "'none' skips biomarker computation entirely")
    ap.add_argument("--fd_rotations", type=int, default=8)
    a = ap.parse_args(argv)

    kw: Dict[str, Any] = {}
    if a.ckpt:
        kw["ckpt"] = a.ckpt
    if a.method.startswith(("rnca", "evapore")):
        kw["device"] = a.device
    if a.opt:
        for item in a.opt.split(","):
            if not item.strip():
                continue
            k, _, v = item.partition("=")
            k, v = k.strip(), v.strip()
            if v.lower() in ("true", "false"):
                kw[k] = v.lower() == "true"
            else:
                try:
                    kw[k] = int(v)
                except ValueError:
                    try:
                        kw[k] = float(v)
                    except ValueError:
                        kw[k] = v

    run_dataset(
        method=a.method, dataset=a.dataset, out_dir=a.out, pred_dir=a.pred_dir,
        split=a.split, synth=a.synth, n_cuts=a.n_cuts, seed=a.seed,
        limit=a.limit,
        image_ids=[s.strip() for s in a.image_ids.split(",")] if a.image_ids else None,
        save_masks=not a.no_save_masks, method_kw=kw,
        bio=a.bio, fd_rotations=a.fd_rotations,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
