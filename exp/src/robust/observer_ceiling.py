"""Inter-observer annotation ceiling: DRIVE (2 obs.) and CHASE_DB1 (2 obs., 28).

Plan reference: exp/EXPERIMENT_PLAN.md S5 / proposal v3 section 4.3, Fig.4C --
"annotation reproducibility: DRIVE / CHASE_DB1 two-observer biomarker
differences, and model-vs-human reproducibility comparison. Assumption:
Fig.4C gives the two-observer biomarker difference as the reference interval
for model error."

This module is entirely CPU (no torch, no GPU inference): it only reads the
two manual annotations each dataset ships and compares them to each other.

Outputs
-------
``exp/results/fig4c_observer.csv``
    one row per image: pixel Dice, clDice, Betti errors, BCS, Junction-F1
    between the two observers, plus the absolute (and MAD-standardised)
    difference of each of the 8 primary biomarker columns
    (``{FD,tortuosity,density,total_length} x {pvbm,skan}``).
``exp/results/fig4c_observer_summary.csv``
    per-dataset mean/median/sd of every column above (the "observer ceiling").

Usage
-----
    cd exp
    python -m src.robust.observer_ceiling
    python -m src.robust.observer_ceiling --datasets drive
    python -m src.robust.observer_ceiling --model-manifest runs/seg/drive/seed0/pred/manifest.csv
"""

from __future__ import annotations

import argparse
import os
import traceback
from typing import Dict, List, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")

GATE_A_GT_CSV = os.path.join(RESULTS_DIR, "gateA_biomarkers_gt.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "fig4c_observer.csv")
OUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "fig4c_observer_summary.csv")

DATASET_LOADERS = {
    # dataset key -> (load_dataset name, split filter or None for "all")
    "drive": ("drive", "test"),        # 2nd observer only ships with DRIVE test
    "chasedb1": ("chasedb1", None),    # "all 28" per plan -- both splits
}


# --------------------------------------------------------------------------- #
# per-image comparison
# --------------------------------------------------------------------------- #
def _load_gray_bool(path: str) -> np.ndarray:
    from src.data.datasets import read_binary

    return read_binary(path)


def _load_rgb(path: str) -> np.ndarray:
    from src.data.datasets import read_image

    img = read_image(path)
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return img[..., :3].astype(np.uint8)


def compare_pair(image: np.ndarray, fov: np.ndarray, obs1: np.ndarray, obs2: np.ndarray
                  ) -> Dict[str, float]:
    """All observer-vs-observer metrics + primary-biomarker diffs for one image."""
    from src.topo.metrics import evaluate_all
    from src.bio.disc import locate_optic_disc
    from src.bio.biomarkers import compute_all, primary_columns

    out: Dict[str, float] = {}
    out.update(evaluate_all(None, obs2, obs1, fov))  # obs1 = "gt" reference

    disc = locate_optic_disc(image, fov, vessel_mask=obs1)
    b1 = compute_all(obs1, fov, image=None, disc=disc)
    b2 = compute_all(obs2, fov, image=None, disc=disc)
    for col in primary_columns(zone=False):
        v1, v2 = b1.get(col, np.nan), b2.get(col, np.nan)
        out[f"obs1_{col}"] = v1
        out[f"obs2_{col}"] = v2
        out[f"absdiff_{col}"] = abs(v1 - v2) if np.isfinite(v1) and np.isfinite(v2) else np.nan
    return out


def _dataset_records(name: str, split: Optional[str]) -> List[dict]:
    from src.data.datasets import load_dataset

    recs = load_dataset(name, split=split)
    return [r for r in recs if r.get("label2_path") and os.path.exists(str(r["label2_path"]))]


def run(datasets: List[str]) -> "object":
    import pandas as pd

    rows: List[dict] = []
    for key in datasets:
        if key not in DATASET_LOADERS:
            raise ValueError(f"unknown dataset {key!r}; choices: {sorted(DATASET_LOADERS)}")
        name, split = DATASET_LOADERS[key]
        recs = _dataset_records(name, split)
        print(f"[observer_ceiling] {key}: {len(recs)} images with a 2nd observer")
        for rec in recs:
            row = dict(dataset=key, image_id=rec["image_id"], subject_id=rec.get("subject_id", ""),
                       split=rec.get("split", ""), error="")
            try:
                image = _load_rgb(rec["image_path"])
                obs1 = _load_gray_bool(rec["label_path"])
                obs2 = _load_gray_bool(rec["label2_path"])
                if rec.get("fov_path") and os.path.exists(str(rec["fov_path"])):
                    fov = _load_gray_bool(rec["fov_path"])
                else:
                    from src.seg.data import derive_fov

                    fov = derive_fov(image) > 0
                if fov.shape != obs1.shape:
                    import cv2

                    fov = cv2.resize(fov.astype(np.uint8) * 255, (obs1.shape[1], obs1.shape[0]),
                                      interpolation=cv2.INTER_NEAREST) > 127
                row.update(compare_pair(image, fov, obs1, obs2))
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc(limit=6)
                print(f"  [warn] {key}/{rec['image_id']}: {row['error']}")
            rows.append(row)

    df = pd.DataFrame(rows)
    df = _standardise_diffs(df)
    return df


# --------------------------------------------------------------------------- #
# standardisation by the Gate-A GT MAD scale (falls back to an in-run pooled
# MAD over the two observers when the Gate-A csv is not available yet)
# --------------------------------------------------------------------------- #
def _standardise_diffs(df):
    import pandas as pd
    from src.bio.biomarkers import primary_columns

    cols = primary_columns(zone=False)
    df = df.copy()

    scales: Dict[str, Dict[str, float]] = {}  # dataset -> column -> sigma
    if os.path.exists(GATE_A_GT_CSV):
        try:
            gt = pd.read_csv(GATE_A_GT_CSV)
            from src.bio.biomarkers import standardise

            gt["dataset"] = gt["dataset"].astype(str).str.lower()
            sc, _ = standardise(gt, dataset_col="dataset", columns=cols, add_z=False)
            for r in sc.itertuples():
                scales.setdefault(r.dataset, {})[r.biomarker] = r.sigma
            print(f"[observer_ceiling] standardising against {GATE_A_GT_CSV}")
        except Exception as exc:  # noqa: BLE001
            print(f"[observer_ceiling] could not read {GATE_A_GT_CSV} ({exc}); "
                  "falling back to an in-run pooled-observer MAD scale")

    # fallback / gap-fill: pooled-observer MAD, per dataset actually present
    for ds, g in df.groupby("dataset"):
        have = scales.get(ds, {})
        for c in cols:
            if have.get(c) and np.isfinite(have[c]) and have[c] > 0:
                continue
            pooled = pd.concat([
                pd.to_numeric(g.get(f"obs1_{c}"), errors="coerce"),
                pd.to_numeric(g.get(f"obs2_{c}"), errors="coerce"),
            ])
            pooled = pooled[np.isfinite(pooled)]
            if pooled.size >= 3:
                med = float(pooled.median())
                mad = float((pooled - med).abs().median())
                sigma = 1.4826 * mad
                if not np.isfinite(sigma) or sigma <= 0:
                    sigma = float(pooled.std()) if pooled.size > 1 else np.nan
            else:
                sigma = np.nan
            scales.setdefault(ds, {})[c] = sigma

    for c in cols:
        zcol = f"z_absdiff_{c}"
        vals = np.full(len(df), np.nan)
        for i, (ds, v) in enumerate(zip(df["dataset"], pd.to_numeric(df.get(f"absdiff_{c}"), errors="coerce"))):
            sig = scales.get(ds, {}).get(c, np.nan)
            if np.isfinite(v) and np.isfinite(sig) and sig > 0:
                vals[i] = v / sig
        df[zcol] = vals
    return df


# --------------------------------------------------------------------------- #
# model-vs-observer1 vs observer2-vs-observer1 (paired), optional hook
# --------------------------------------------------------------------------- #
def compare_model_vs_observer(
    obs_df, model_manifest_csv: str, image_id_col: str = "image_id",
    metrics=("dice", "cldice", "bcs", "junction_f1"),
):
    """Paired comparison of model-vs-observer1 error against observer2-vs-observer1.

    ``model_manifest_csv`` is an ``exp/src/seg/infer.py`` manifest (columns
    ``image, mask_path, label_path, fov_path``). For every image present in
    both the manifest and ``obs_df`` (i.e. it has a 2nd observer), this
    recomputes the same metrics between the model mask and observer 1, then
    runs a paired Wilcoxon signed-rank test of ``error_model`` vs
    ``error_obs2`` per metric, where ``error = 1 - metric`` for the four
    bounded-agreement metrics and the raw value for the two Betti errors.

    Returns ``(per_image_df, summary_df)``. Requires the model masks to
    already exist on disk (run ``src.seg.infer`` first) -- this is a hook, not
    exercised by the CPU smoke tests in this repo (GPUs busy at authoring
    time).
    """
    import pandas as pd
    from src.topo.metrics import evaluate_all
    from src.data.datasets import read_binary

    man = pd.read_csv(model_manifest_csv)
    man = man.set_index("image")

    rows = []
    for rec in obs_df.itertuples():
        iid = getattr(rec, image_id_col)
        if iid not in man.index:
            continue
        mrow = man.loc[iid]
        try:
            model_mask = read_binary(mrow["mask_path"])
            obs1 = read_binary(mrow["label_path"])
            fov = read_binary(mrow["fov_path"]) if isinstance(mrow.get("fov_path"), str) and mrow["fov_path"] else None
            row = {"image_id": iid, "dataset": rec.dataset}
            model_vs_obs1 = evaluate_all(None, model_mask, obs1, fov)
            for m in metrics + ("beta0_err", "beta1_err"):
                row[f"model_{m}"] = model_vs_obs1.get(m, np.nan)
                row[f"obs2_{m}"] = getattr(rec, m, np.nan)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] model-vs-observer skip {iid}: {exc}")

    per_image = pd.DataFrame(rows)
    summary_rows = []
    from scipy import stats as sstats

    for m in metrics + ("beta0_err", "beta1_err"):
        mc, oc = f"model_{m}", f"obs2_{m}"
        if mc not in per_image.columns:
            continue
        a = pd.to_numeric(per_image[mc], errors="coerce")
        b = pd.to_numeric(per_image[oc], errors="coerce")
        ok = np.isfinite(a) & np.isfinite(b)
        if m in ("beta0_err", "beta1_err"):
            err_model, err_obs2 = a[ok], b[ok]
        else:
            err_model, err_obs2 = 1.0 - a[ok], 1.0 - b[ok]
        n = int(ok.sum())
        if n >= 2 and not np.allclose(err_model, err_obs2):
            try:
                stat, p = sstats.wilcoxon(err_model, err_obs2)
            except Exception:
                stat, p = np.nan, np.nan
        else:
            stat, p = np.nan, np.nan
        summary_rows.append(dict(
            metric=m, n=n,
            mean_error_model=float(np.mean(err_model)) if n else np.nan,
            mean_error_obs2=float(np.mean(err_obs2)) if n else np.nan,
            mean_diff=float(np.mean(err_model - err_obs2)) if n else np.nan,
            wilcoxon_stat=float(stat) if np.isfinite(stat) else np.nan,
            wilcoxon_p=float(p) if np.isfinite(p) else np.nan,
        ))
    summary = pd.DataFrame(summary_rows)
    return per_image, summary


# --------------------------------------------------------------------------- #
# summary table
# --------------------------------------------------------------------------- #
_SUMMARY_METRICS = ["dice", "cldice", "beta0_err", "beta1_err", "bcs", "junction_f1"]


def summarise(df) -> "object":
    import pandas as pd
    from src.bio.biomarkers import primary_columns

    cols = _SUMMARY_METRICS + [f"absdiff_{c}" for c in primary_columns(False)] \
        + [f"z_absdiff_{c}" for c in primary_columns(False)]
    rows = []
    for ds, g in df.groupby("dataset"):
        g_ok = g[g["error"] == ""] if "error" in g.columns else g
        row = {"dataset": ds, "n_images": len(g), "n_ok": len(g_ok)}
        for c in cols:
            if c not in g_ok.columns:
                continue
            x = pd.to_numeric(g_ok[c], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            row[f"{c}_mean"] = float(np.mean(x)) if x.size else np.nan
            row[f"{c}_median"] = float(np.median(x)) if x.size else np.nan
            row[f"{c}_sd"] = float(np.std(x)) if x.size > 1 else np.nan
            row[f"{c}_n"] = int(x.size)
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="inter-observer annotation ceiling (DRIVE, CHASE_DB1)")
    ap.add_argument("--datasets", default="drive,chasedb1")
    ap.add_argument("--out", default=OUT_CSV)
    ap.add_argument("--summary-out", default=OUT_SUMMARY_CSV)
    ap.add_argument("--model-manifest", default=None,
                     help="optional exp/src/seg/infer.py manifest.csv to also compute "
                          "model-vs-observer1 vs observer2-vs-observer1 (paired); "
                          "needs a trained checkpoint's predictions on disk")
    args = ap.parse_args(argv)

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    df = run(datasets)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows -> {args.out}")

    summary = summarise(df)
    summary.to_csv(args.summary_out, index=False)
    print(f"wrote summary -> {args.summary_out}")
    import pandas as pd
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(summary[["dataset", "n_images", "n_ok", "dice_mean", "cldice_mean",
                        "beta0_err_mean", "beta1_err_mean", "bcs_mean", "junction_f1_mean"]])

    if args.model_manifest:
        per_image, cmp_summary = compare_model_vs_observer(df, args.model_manifest)
        pi_path = os.path.join(os.path.dirname(args.out), "fig4c_model_vs_observer.csv")
        cs_path = os.path.join(os.path.dirname(args.out), "fig4c_model_vs_observer_summary.csv")
        per_image.to_csv(pi_path, index=False)
        cmp_summary.to_csv(cs_path, index=False)
        print(f"wrote model-vs-observer comparison -> {pi_path}, {cs_path}")
        print(cmp_summary)
    else:
        print("[observer_ceiling] no --model-manifest given; "
              "skipping model-vs-observer1 vs observer2-vs-observer1 comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
