"""E1 -- an OUTER bootstrap around the whole classifier-fitting pipeline.

Why this script exists
----------------------
The reference gap reported by ``e1_oracle.py`` comes from an image bootstrap of
the *already fitted* cross-validated out-of-fold probabilities.  That interval is
therefore **conditional on the fitted fold models and on the fold assignment**,
and a reviewer of the manuscript pointed out, correctly, that at HRF's n = 45 it
is optimistic about the variance of the classifier fit itself.

This script prices that variance.  For each outer resample of the images the
whole protocol is re-run from scratch -- fold assignment, every fit, both arms --
and the gap is recomputed, so the interval covers fold-assignment noise and
fitting noise as well as sampling noise.

Two details that matter and are easy to get wrong:

* **Folds are assigned on unique images, not on resampled rows.** A naive
  bootstrap of a cross-validation puts duplicate copies of the same image in the
  training and the test fold, which leaks and shrinks the interval. Here the
  fold label is attached to the *image*, and every duplicate of that image
  inherits it, so no image is ever both trained on and scored in one fold.
* **Both arms share the resample and the folds.** The gap is a paired quantity;
  resampling the two arms independently would price a difference that nobody
  computes.

A draw is discarded (and counted) if the resampled set does not contain at least
``n_splits`` members of every class, which happens occasionally at n = 45.

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.e1_outer_boot --n_boot 200
"""
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS

SEED = 0
N_REPEAT = 3          # CV repeats, identical to p3_downstream / e1_oracle
N_SPLITS = 5
PANEL = [c for c in PRIMARY_COLS if c.endswith("_skan")]   # frozen primary panel


# ------------------------------------------------------------------ helpers
def macro_auc(y: np.ndarray, proba: np.ndarray, classes: Sequence) -> float:
    from sklearn.metrics import roc_auc_score

    a = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.sum() in (0, len(yy)):
            continue
        a.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(a)) if a else float("nan")


def _model(kind: str, rep: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "logreg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=5000, C=1.0))
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.08, min_samples_leaf=10,
        l2_regularization=1.0, early_stopping=False, random_state=SEED + rep)


def cv_proba_fixed_folds(X: np.ndarray, y: np.ndarray, folds: np.ndarray,
                         kind: str, classes: Sequence) -> np.ndarray:
    """Out-of-fold probabilities with an externally supplied fold label.

    ``folds`` is per row; every copy of one image carries the same label, which
    is what keeps a bootstrap duplicate out of its own training set.
    """
    acc = np.zeros((len(y), len(classes)), float)
    n_rep = folds.shape[1]
    for rep in range(n_rep):
        f = folds[:, rep]
        for k in np.unique(f):
            te = f == k
            tr = ~te
            if tr.sum() < len(classes) or len(np.unique(y[tr])) < 2:
                continue
            m = _model(kind, rep)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            order = [list(m.classes_).index(c) if c in list(m.classes_) else -1
                     for c in classes]
            for j, o in enumerate(order):
                if o >= 0:
                    acc[te, j] += p[:, o]
    return acc / n_rep


def assign_folds(y_unique: np.ndarray, rng: np.random.RandomState = None
                 ) -> np.ndarray:
    """``(n_unique, N_REPEAT)`` stratified fold labels for the unique images.

    With ``rng=None`` the fold seeds are ``SEED + rep``, i.e. exactly the ones
    ``p3_downstream.py`` uses, so the point estimate this script prints must
    reproduce ``e1_oracle_downstream.csv`` to the last digit -- which is the
    self-check that the re-implemented CV loop is the same protocol.
    """
    from sklearn.model_selection import StratifiedKFold

    out = np.zeros((len(y_unique), N_REPEAT), int)
    for rep in range(N_REPEAT):
        rs = (SEED + rep) if rng is None else int(rng.randint(0, 2 ** 31 - 1))
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=rs)
        for k, (_, te) in enumerate(skf.split(np.zeros(len(y_unique)), y_unique)):
            out[te, rep] = k
    return out


def usable(y: np.ndarray, classes: Sequence) -> bool:
    return all((y == c).sum() >= N_SPLITS for c in classes)


# --------------------------------------------------------------------- run
def run_cell(d: pd.DataFrame, cols: Sequence[str], kind: str, n_boot: int,
             seed: int = SEED) -> Dict[str, float]:
    y = d["disease"].to_numpy()
    classes = sorted(pd.unique(y))
    Xg = np.nan_to_num(d[[c + "__gt" for c in cols]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    Xp = np.nan_to_num(d[[c + "__pred" for c in cols]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    n = len(y)

    rng = np.random.RandomState(seed)

    # ---- point estimate: the full sample, the p3_downstream fold seeds
    folds = assign_folds(y)
    auc_g = macro_auc(y, cv_proba_fixed_folds(Xg, y, folds, kind, classes), classes)
    auc_p = macro_auc(y, cv_proba_fixed_folds(Xp, y, folds, kind, classes), classes)
    gap_point = auc_g - auc_p

    gaps, gts, prs = [], [], []
    skipped = 0
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yb = y[idx]
        if not usable(yb, classes):
            skipped += 1
            continue
        # fold label lives on the image, so duplicates stay together
        f_img = assign_folds(y, rng)          # labels for every original image
        fb = f_img[idx]                       # ... inherited by each copy
        pg = cv_proba_fixed_folds(Xg[idx], yb, fb, kind, classes)
        pp = cv_proba_fixed_folds(Xp[idx], yb, fb, kind, classes)
        ag, ap = macro_auc(yb, pg, classes), macro_auc(yb, pp, classes)
        if not (np.isfinite(ag) and np.isfinite(ap)):
            skipped += 1
            continue
        gts.append(ag)
        prs.append(ap)
        gaps.append(ag - ap)

    g = np.asarray(gaps, float)
    return dict(
        n=int(n), n_classes=len(classes), clf=kind,
        auc_gt=auc_g, auc_pred=auc_p, gap=gap_point,
        n_outer=int(g.size), n_skipped=int(skipped),
        gap_outer_lo=float(np.percentile(g, 2.5)) if g.size else float("nan"),
        gap_outer_hi=float(np.percentile(g, 97.5)) if g.size else float("nan"),
        gap_outer_mean=float(np.mean(g)) if g.size else float("nan"),
        gap_outer_sd=float(np.std(g, ddof=1)) if g.size > 1 else float("nan"),
        auc_gt_outer_lo=float(np.percentile(gts, 2.5)) if g.size else float("nan"),
        auc_gt_outer_hi=float(np.percentile(gts, 97.5)) if g.size else float("nan"),
        auc_pred_outer_lo=float(np.percentile(prs, 2.5)) if g.size else float("nan"),
        auc_pred_outer_hi=float(np.percentile(prs, 97.5)) if g.size else float("nan"),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(PIVOT_DIR, "p2_predictions.csv"))
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    ap.add_argument("--n_boot", type=int, default=200)
    ap.add_argument("--n_boot_gbdt", type=int, default=0,
                    help="separate draw count for the (much slower) GBDT "
                         "sensitivity arm; 0 means use --n_boot")
    ap.add_argument("--clfs", default="logreg,gbdt")
    args = ap.parse_args(argv)

    pred = pd.read_csv(args.pred, low_memory=False)
    cells: List[Tuple[str, str]] = [("fives", "all"), ("fives", "test"),
                                    ("hrf", "all")]
    rows: List[dict] = []
    for ds, split in cells:
        d = pred[pred["dataset"] == ds]
        if split != "all":
            d = d[d["split"] == split]
        d = d[d["disease"].notna()].reset_index(drop=True)
        if len(d) < 30:
            continue
        for kind in [c.strip() for c in args.clfs.split(",") if c.strip()]:
            nb = (args.n_boot_gbdt if (kind == "gbdt" and args.n_boot_gbdt)
                  else args.n_boot)
            t0 = time.time()
            r = run_cell(d, PANEL, kind, nb)
            r.update(dataset=ds, split=split, featureset="skan4",
                     n_features=len(PANEL), seconds=round(time.time() - t0, 1),
                     source=("results/pivot/p2_predictions.csv (skan4 panel); "
                             "outer bootstrap of the FULL protocol -- fold "
                             "assignment and every fit recomputed per draw, "
                             "folds attached to the image so bootstrap "
                             "duplicates cannot leak across a fold, both arms "
                             "sharing the resample and the folds; "
                             f"{nb} draws, seed {SEED}"))
            rows.append(r)
            print(f"[outer] {ds}/{split}/{kind}: gap {r['gap']:+.4f} "
                  f"inner-free outer CI [{r['gap_outer_lo']:+.4f}, "
                  f"{r['gap_outer_hi']:+.4f}] "
                  f"({r['n_outer']} draws, {r['seconds']}s)")

    out = pd.DataFrame(rows)
    cols = ["dataset", "split", "clf", "featureset", "n_features", "n",
            "n_classes", "auc_gt", "auc_pred", "gap", "gap_outer_lo",
            "gap_outer_hi", "gap_outer_mean", "gap_outer_sd",
            "auc_gt_outer_lo", "auc_gt_outer_hi", "auc_pred_outer_lo",
            "auc_pred_outer_hi", "n_outer", "n_skipped", "seconds", "source"]
    out = out[[c for c in cols if c in out.columns]]
    p = os.path.join(args.out_dir, "e1_outer_boot.csv")
    out.to_csv(p, index=False)
    print("wrote", p, out.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
