"""Probe P3 -- does measurement error attenuate downstream clinical validity? (D3)

For FIVES (AMD / DR / Glaucoma / Normal) and HRF (dr / g / h) we predict the
disease label from the four primary biomarkers (both pipelines = 8 columns,
optionally plus the zone-B variants) and compare three feature sources

    gt          biomarkers of the reference mask        -- the ceiling
    pred        biomarkers of the segmentation, raw     -- what a deployed
                                                            oculomics pipeline
                                                            actually measures
    cal_<model> biomarkers after the probe-P2 calibration

Every calibrated value comes from ``results/pivot/p2_predictions.csv``, where
rows of training images are cross-fitted, so no image is scored by a calibrator
that saw its own reference biomarkers.

Classifier: multinomial logistic regression (standardised) and a small GBDT,
5-fold stratified CV repeated 3x with fixed seeds; the score is the macro
one-vs-rest AUC of the pooled out-of-fold probabilities, with a 1000-sample
image-level bootstrap CI.  Differences between sources are bootstrapped
**paired** on the same resampled images.

CLI
---
    python -m src.pivot.p3_downstream
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS, PRIMARY_COLS_ZONEB

SEED = 0
N_BOOT = 1000
N_REPEAT = 3                  # repeated stratified CV
MODELS = ("offset_per_ds", "linear_per_ds", "gbdt_pooled",
          "offset_lodo", "gbdt_lodo")


def macro_auc(y: np.ndarray, proba: np.ndarray, classes: Sequence) -> float:
    from sklearn.metrics import roc_auc_score

    aucs = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        aucs.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def cv_proba(X: np.ndarray, y: np.ndarray, kind: str, classes: Sequence,
             n_splits: int = 5) -> np.ndarray:
    """Out-of-fold class probabilities, averaged over ``N_REPEAT`` CV repeats."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    acc = np.zeros((len(y), len(classes)), dtype=float)
    for rep in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED + rep)
        for tr, te in skf.split(X, y):
            if kind == "logreg":
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=1.0))
            else:
                m = HistGradientBoostingClassifier(
                    max_depth=3, max_iter=150, learning_rate=0.08,
                    min_samples_leaf=10, l2_regularization=1.0,
                    early_stopping=False, random_state=SEED + rep)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            order = [list(m.classes_).index(c) for c in classes]
            acc[te] += p[:, order]
    return acc / N_REPEAT


def boot_ci(y: np.ndarray, probas: Dict[str, np.ndarray], classes: Sequence,
            n_boot: int = N_BOOT) -> Tuple[Dict[str, Tuple[float, float]],
                                           Dict[str, Tuple[float, float]]]:
    """Paired image-level bootstrap: per-source CI and CI of (gt - source)."""
    rng = np.random.RandomState(SEED)
    n = len(y)
    keys = list(probas)
    vals: Dict[str, List[float]] = {k: [] for k in keys}
    diffs: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < len(classes):
            continue
        cur = {k: macro_auc(y[idx], probas[k][idx], classes) for k in keys}
        for k in keys:
            vals[k].append(cur[k])
            diffs[k].append(cur["gt"] - cur[k])
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
          for k, v in vals.items() if v}
    dci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
           for k, v in diffs.items() if v}
    return ci, dci


def feature_block(pred: pd.DataFrame, source: str, cols: Sequence[str]) -> np.ndarray:
    if source == "gt":
        names = [c + "__gt" for c in cols]
    elif source == "pred":
        names = [c + "__pred" for c in cols]
    else:
        names = [c + "__cal_" + source for c in cols]
    X = pred[names].to_numpy(dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def run_dataset(pred: pd.DataFrame, dataset: str, cols: Sequence[str],
                sources: Sequence[str], restrict_split: str = None,
                featureset: str = "primary8") -> List[dict]:
    d = pred[pred["dataset"] == dataset].copy()
    if restrict_split:
        d = d[d["split"] == restrict_split]
    d = d[d["disease"].notna()].reset_index(drop=True)
    if len(d) < 30:
        return []
    y = d["disease"].to_numpy()
    classes = sorted(pd.unique(y))
    rows = []
    for kind in ("logreg", "gbdt"):
        probas = {}
        for s in sources:
            X = feature_block(d, s, cols)
            if not np.isfinite(X).any() or np.allclose(X, 0):
                continue
            probas[s] = cv_proba(X, y, kind, classes)
        if "gt" not in probas:
            continue
        ci, dci = boot_ci(y, probas, classes)
        for s, p in probas.items():
            rows.append(dict(
                dataset=dataset, split=restrict_split or "all", clf=kind,
                featureset=featureset,
                source=s, n=len(d), n_classes=len(classes),
                macro_auc=macro_auc(y, p, classes),
                ci_lo=ci.get(s, (np.nan, np.nan))[0],
                ci_hi=ci.get(s, (np.nan, np.nan))[1],
                gt_minus_this=macro_auc(y, probas["gt"], classes) - macro_auc(y, p, classes),
                diff_ci_lo=dci.get(s, (np.nan, np.nan))[0],
                diff_ci_hi=dci.get(s, (np.nan, np.nan))[1],
                class_counts=";".join("%s=%d" % (c, int((y == c).sum()))
                                      for c in classes)))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(PIVOT_DIR, "p2_predictions.csv"))
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    ap.add_argument("--datasets", default="fives,hrf")
    ap.add_argument("--featuresets", default="all",
                    help="comma list of skan4,primary8,primary8+zoneB; 'all' "
                         "runs every one (the default, and what a full rebuild "
                         "does)")
    ap.add_argument("--sources", default="all",
                    help="comma list of feature sources; 'gt,pred' is enough "
                         "for the reference-mask arm and is much cheaper than "
                         "the full list, which also fits the five calibration "
                         "models")
    ap.add_argument("--merge", action="store_true",
                    help="merge the computed rows into the existing "
                         "p3_downstream.csv (replacing any row with the same "
                         "dataset/split/clf/featureset/source) instead of "
                         "overwriting the file")
    args = ap.parse_args(argv)

    pred = pd.read_csv(args.pred)
    sources = ["gt", "pred"] + list(MODELS)
    if args.sources != "all":
        want = [s.strip() for s in args.sources.split(",") if s.strip()]
        sources = [s for s in sources if s in want]
    want_fs = ("skan4", "primary8", "primary8+zoneB") if args.featuresets == "all" \
        else tuple(s.strip() for s in args.featuresets.split(",") if s.strip())
    rows: List[dict] = []
    zone = [c for c in PRIMARY_COLS_ZONEB if (c + "__gt") in pred.columns]
    # The frozen primary measurement panel (DECISIONS.md 2026-09-17 13:30): the
    # four biomarkers on the skan pipeline.  The eight-column panel duplicates
    # density exactly (both pipelines read the same foreground fraction) and
    # carries the PVBM tortuosity column that the cross-pipeline agreement rule
    # already disqualified, so eight equally weighted columns double-count.
    # It is retained as a sensitivity analysis, not dropped.
    skan4 = [c for c in PRIMARY_COLS if c.endswith("_skan")]
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        if "skan4" in want_fs:
            rows += run_dataset(pred, ds, skan4, sources, featureset="skan4")
        if "primary8" in want_fs:
            rows += run_dataset(pred, ds, PRIMARY_COLS, sources)
        if zone and "primary8+zoneB" in want_fs:
            rows += run_dataset(pred, ds, list(PRIMARY_COLS) + zone, sources,
                                featureset="primary8+zoneB")
        if ds == "fives":
            if "skan4" in want_fs:
                rows += run_dataset(pred, ds, skan4, sources,
                                    restrict_split="test", featureset="skan4")
            if "primary8" in want_fs:
                rows += run_dataset(pred, ds, PRIMARY_COLS, sources,
                                    restrict_split="test")
    res = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "p3_downstream.csv")
    if args.merge and os.path.exists(out_path):
        key = ["dataset", "split", "clf", "featureset", "source"]
        old = pd.read_csv(out_path)
        keep = old.merge(res[key].drop_duplicates(), on=key, how="left",
                         indicator=True)
        keep = keep[keep["_merge"] == "left_only"].drop(columns="_merge")
        res = pd.concat([keep, res], ignore_index=True)
        res = res.sort_values(key).reset_index(drop=True)
        print(f"[merge] {len(keep)} existing rows kept, {len(rows)} recomputed")
    res.to_csv(out_path, index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(res.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
