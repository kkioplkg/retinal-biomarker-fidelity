"""R2 -- the FIVES audit on all 800 images.

Reviewer point 11 (``review/user_cmig_review_20260918.md``)
----------------------------------------------------------
"FIVES has 800 images and you used a capped subset. Run the full set, or show by
stratified multi-seed sub-sampling that the headline is insensitive to the
sampling scheme."

Status of the inputs (checked, not assumed)
-------------------------------------------
``runs/seg_oof/fives/pred/{prob,mask}`` holds **600 of 600** cross-fitted
out-of-fold predictions for the FIVES training split, so nothing is missing and
the full-800 version of the audit is computable without any new inference.  The
shipped ``bio_master.csv`` capped the training split at 50 images per disease
class (200 of 600) purely to bound CPU time; ``src.pivot.build_table`` was
re-run with ``--fives_train_n 150`` to fill in the remaining 400 images, writing
``results/pivot/r2/bio_master_full.csv``.

What this script reports
------------------------
``r2_fives_audit_800.csv``
    Constant offset, residual SD, Pearson and Spearman r per biomarker on
    (a) the 200-image test split against the seed-0 held-out prediction,
    (b) the 600-image training split against the cross-fitted out-of-fold
        prediction,
    (c) all 800 pooled, with the prediction source recorded per block, and
    (d) the shipped 200-image training subset, so the change caused by lifting
        the cap is visible directly.
    The prediction source differs between (a) and (b) -- a single seed-0 model
    on held-out images versus a cross-fitted ensemble on training images -- so
    (c) is reported as a pooled description and the two blocks are never
    silently merged.

``r2_fives_downstream_800.csv``
    The reference-gap downstream arm on the same blocks: macro one-vs-rest AUC
    from reference-mask biomarkers against segmentation-derived biomarkers,
    logistic regression and HistGradientBoosting, 5-fold x 3-repeat stratified
    CV, with a 1000-draw paired image bootstrap of the difference.

``r2_fives_subsample.csv``
    The representativeness check the reviewer asked for as the fallback, run
    anyway because it is cheap and it is the thing that makes the original
    200-image number defensible in retrospect: five class-stratified random
    subsamples of the 600 training images at n = 200, each re-running the audit
    and the downstream gap, against the full-600 value.

CLI
---
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r2_fives_full
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import PIVOT_DIR, PRIMARY_COLS, RUNS_DIR, sigma_table

SEED = 0
N_BOOT = 2000
N_BOOT_AUC = 1000
N_REPEAT = 3
N_SPLITS = 5
PANEL_SKAN = [c for c in PRIMARY_COLS if c.endswith("_skan")]
PANEL = {c: ("primary" if c.endswith("_skan") else "sensitivity")
         for c in PRIMARY_COLS}


# ------------------------------------------------------------------ checks
def oof_coverage(master: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Which FIVES training images have an out-of-fold prediction on disk."""
    from src.pivot.common import pred_paths

    recs = master[(master["dataset"] == "fives") &
                  (master["source"] == "gt")]["image_id"].astype(str)
    train_ids = sorted(set(
        master[(master["dataset"] == "fives") & (master["split"] == "train")]
        ["image_id"].astype(str)))
    rows, missing = [], []
    for iid in train_ids:
        pp, mp = pred_paths("fives", iid, "pred_oof")
        ok = os.path.exists(pp) and os.path.exists(mp)
        rows.append(dict(image_id=iid, prob_path=pp, mask_path=mp,
                         prob_exists=os.path.exists(pp),
                         mask_exists=os.path.exists(mp), oof_available=ok))
        if not ok:
            missing.append(iid)
    del recs
    return pd.DataFrame(rows), missing


# ------------------------------------------------------------------ audit
def boot_ci(x: np.ndarray, fn, n_boot: int = N_BOOT, seed: int = SEED):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, x.size, size=(n_boot, x.size))
    v = np.array([fn(x[i]) for i in idx], float)
    v = v[np.isfinite(v)]
    if not v.size:
        return (float("nan"), float("nan"))
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def pairs(master: pd.DataFrame, split: str, pred_source: str,
          ids: Optional[Sequence[str]] = None) -> pd.DataFrame:
    m = master[master["dataset"] == "fives"]
    gt = m[(m["source"] == "gt") & (m["split"] == split)]
    pr = m[(m["source"] == pred_source) & (m["split"] == split)]
    cols = ["image_id", "disease"] + list(PRIMARY_COLS)
    d = gt[cols].merge(pr[["image_id"] + list(PRIMARY_COLS)], on="image_id",
                       suffixes=("__gt", "__pred"))
    if ids is not None:
        d = d[d["image_id"].astype(str).isin(set(ids))]
    return d.reset_index(drop=True)


def audit_block(d: pd.DataFrame, sig: Dict[str, float], block: str,
                pred_source: str) -> List[dict]:
    from scipy.stats import pearsonr, spearmanr

    rows: List[dict] = []
    for col in PRIMARY_COLS:
        s = sig.get(col, float("nan"))
        g = d[col + "__gt"].to_numpy(float)
        p = d[col + "__pred"].to_numpy(float)
        e = (p - g) / s
        ok = np.isfinite(e)
        if ok.sum() < 3:
            continue
        olo, ohi = boot_ci(e[ok], np.mean)
        slo, shi = boot_ci(e[ok], lambda v: np.std(v, ddof=1))
        okr = np.isfinite(g) & np.isfinite(p)
        rows.append(dict(
            block=block, pred_source=pred_source, biomarker=col,
            panel=PANEL[col], n=int(ok.sum()), sigma=s,
            offset=float(np.mean(e[ok])), offset_lo=olo, offset_hi=ohi,
            resid_sd=float(np.std(e[ok], ddof=1)), resid_sd_lo=slo,
            resid_sd_hi=shi,
            r_pearson=float(pearsonr(g[okr], p[okr])[0]) if okr.sum() > 2 else np.nan,
            r_spearman=float(spearmanr(g[okr], p[okr])[0]) if okr.sum() > 2 else np.nan,
            source=("results/pivot/r2/bio_master_full.csv; sigma frozen at "
                    "results/gateA_biomarker_scales_train.csv "
                    "(FIVES scale estimated on 120 training images); "
                    "2000-draw image bootstrap")))
    return rows


# -------------------------------------------------------------- downstream
def macro_auc(y, proba, classes) -> float:
    from sklearn.metrics import roc_auc_score
    a = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if yy.sum() in (0, len(yy)):
            continue
        a.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(a)) if a else float("nan")


def cv_proba(X, y, kind, classes):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    acc = np.zeros((len(y), len(classes)), float)
    for rep in range(N_REPEAT):
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                              random_state=SEED + rep)
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


def downstream_block(d: pd.DataFrame, block: str, pred_source: str,
                     panel_name: str, cols: Sequence[str]) -> List[dict]:
    y = d["disease"].to_numpy()
    classes = sorted(pd.unique(y))
    if len(classes) < 2 or len(y) < 30:
        return []
    Xg = np.nan_to_num(d[[c + "__gt" for c in cols]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    Xp = np.nan_to_num(d[[c + "__pred" for c in cols]].to_numpy(float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    rows: List[dict] = []
    for kind in ("logreg", "gbdt"):
        pg, pp = cv_proba(Xg, y, kind, classes), cv_proba(Xp, y, kind, classes)
        a_g, a_p = macro_auc(y, pg, classes), macro_auc(y, pp, classes)
        rng = np.random.RandomState(SEED)
        n = len(y)
        dd, gg, pv = [], [], []
        for _ in range(N_BOOT_AUC):
            idx = rng.randint(0, n, n)
            if len(np.unique(y[idx])) < len(classes):
                continue
            bg, bp = macro_auc(y[idx], pg[idx], classes), macro_auc(y[idx], pp[idx], classes)
            if np.isfinite(bg) and np.isfinite(bp):
                dd.append(bg - bp); gg.append(bg); pv.append(bp)

        def pc(v):
            v = np.asarray(v, float)
            return ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                    if v.size else (float("nan"), float("nan")))
        d_lo, d_hi = pc(dd)
        g_lo, g_hi = pc(gg)
        p_lo, p_hi = pc(pv)
        rows.append(dict(
            block=block, pred_source=pred_source, featureset=panel_name,
            clf=kind, n=n, n_classes=len(classes),
            auc_reference=a_g, auc_reference_lo=g_lo, auc_reference_hi=g_hi,
            auc_predicted=a_p, auc_predicted_lo=p_lo, auc_predicted_hi=p_hi,
            delta_auc=a_g - a_p, delta_auc_lo=d_lo, delta_auc_hi=d_hi,
            class_counts=";".join(f"{c}={int((y == c).sum())}" for c in classes),
            source=("results/pivot/r2/bio_master_full.csv; 5-fold x 3 repeats "
                    "stratified CV, macro one-vs-rest AUC; 1000-draw paired "
                    "image bootstrap, identical protocol to "
                    "src/pivot/p3_downstream.py")))
    return rows


# ------------------------------------------------------ subsample sensitivity
def stratified_subsample(d: pd.DataFrame, n_per_class: int, seed: int
                         ) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    keep = []
    for c, g in d.groupby("disease"):
        idx = np.array(g.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        keep += list(idx[:n_per_class])
    return d.loc[sorted(keep)].reset_index(drop=True)


# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=os.path.join(PIVOT_DIR, "r2",
                                                     "bio_master_full.csv"))
    ap.add_argument("--master_capped",
                    default=os.path.join(PIVOT_DIR, "bio_master.csv"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r2"))
    ap.add_argument("--n_subsample_seeds", type=int, default=5)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    master = pd.read_csv(args.master, low_memory=False)
    fives = master[master["dataset"] == "fives"]
    print("[fives rows]", fives.groupby(["split", "source"]).size().to_dict())

    cov, missing = oof_coverage(master)
    cov.to_csv(os.path.join(args.out_dir, "fives_oof_coverage.csv"), index=False)
    if missing:
        p = os.path.join(args.out_dir, "fives_oof_missing.csv")
        pd.DataFrame(dict(image_id=missing)).to_csv(p, index=False)
        print("[oof] %d of %d FIVES training images MISSING an out-of-fold "
              "prediction -- listed in %s" % (len(missing), len(cov), p))
    else:
        print("[oof] all %d FIVES training images have an out-of-fold "
              "prediction on disk" % len(cov))

    sig = sigma_table()["fives"]

    # ---- blocks
    d_test = pairs(master, "test", "pred_test")
    d_train = pairs(master, "train", "pred_oof")
    capped = pd.read_csv(args.master_capped, low_memory=False)
    capped_ids = sorted(set(
        capped[(capped["dataset"] == "fives") & (capped["split"] == "train") &
               (capped["source"] == "pred_oof")]["image_id"].astype(str)))
    d_train_capped = d_train[d_train["image_id"].astype(str).isin(
        set(capped_ids))].reset_index(drop=True)
    d_all = pd.concat([d_test.assign(_src="pred_test"),
                       d_train.assign(_src="pred_oof")], ignore_index=True)
    print("[blocks] test=%d train_full=%d train_capped=%d pooled=%d"
          % (len(d_test), len(d_train), len(d_train_capped), len(d_all)))

    rows: List[dict] = []
    rows += audit_block(d_test, sig, "test200", "pred_test (seg seed 0)")
    rows += audit_block(d_train, sig, "train600", "pred_oof (cross-fitted)")
    rows += audit_block(d_train_capped, sig, "train200_shipped_cap",
                        "pred_oof (cross-fitted)")
    rows += audit_block(d_all, sig, "all800_pooled",
                        "pred_test on test200 + pred_oof on train600")
    au = pd.DataFrame(rows)
    p = os.path.join(args.out_dir, "r2_fives_audit_800.csv")
    au.to_csv(p, index=False)
    print("wrote", p, au.shape)

    drows: List[dict] = []
    for block, dd, srcname in (("test200", d_test, "pred_test (seg seed 0)"),
                               ("train600", d_train, "pred_oof (cross-fitted)"),
                               ("train200_shipped_cap", d_train_capped,
                                "pred_oof (cross-fitted)"),
                               ("all800_pooled", d_all,
                                "pred_test + pred_oof")):
        for pname, cols in (("skan4", PANEL_SKAN), ("primary8", PRIMARY_COLS)):
            drows += downstream_block(dd, block, srcname, pname, cols)
            print("[down] %s %s done" % (block, pname), flush=True)
    dn = pd.DataFrame(drows)
    p = os.path.join(args.out_dir, "r2_fives_downstream_800.csv")
    dn.to_csv(p, index=False)
    print("wrote", p, dn.shape)

    # ---- stratified sub-sampling sensitivity on the 600 training images
    srows: List[dict] = []
    for sd in range(args.n_subsample_seeds):
        sub = stratified_subsample(d_train, 50, 1000 + sd)
        a = audit_block(sub, sig, "train200_random_seed%d" % sd,
                        "pred_oof (cross-fitted)")
        prim = [r for r in a if r["panel"] == "primary"]
        d0 = downstream_block(sub, "train200_random_seed%d" % sd,
                              "pred_oof (cross-fitted)", "skan4", PANEL_SKAN)
        for r in d0:
            srows.append(dict(
                subsample_seed=sd, n=len(sub), clf=r["clf"],
                mean_abs_offset=float(np.mean([abs(x["offset"]) for x in prim])),
                mean_resid_sd=float(np.mean([x["resid_sd"] for x in prim])),
                mean_r=float(np.mean([x["r_pearson"] for x in prim])),
                auc_reference=r["auc_reference"],
                auc_predicted=r["auc_predicted"],
                delta_auc=r["delta_auc"], delta_auc_lo=r["delta_auc_lo"],
                delta_auc_hi=r["delta_auc_hi"],
                source=("class-stratified random subsample of the 600 FIVES "
                        "training images, 50 per disease class; "
                        "results/pivot/r2/bio_master_full.csv")))
        print("[sub] seed %d done" % sd, flush=True)
    # the full-600 reference row
    a_full = [r for r in audit_block(d_train, sig, "train600",
                                     "pred_oof") if r["panel"] == "primary"]
    for r in downstream_block(d_train, "train600", "pred_oof", "skan4",
                              PANEL_SKAN):
        srows.append(dict(
            subsample_seed=-1, n=len(d_train), clf=r["clf"],
            mean_abs_offset=float(np.mean([abs(x["offset"]) for x in a_full])),
            mean_resid_sd=float(np.mean([x["resid_sd"] for x in a_full])),
            mean_r=float(np.mean([x["r_pearson"] for x in a_full])),
            auc_reference=r["auc_reference"], auc_predicted=r["auc_predicted"],
            delta_auc=r["delta_auc"], delta_auc_lo=r["delta_auc_lo"],
            delta_auc_hi=r["delta_auc_hi"],
            source="all 600 FIVES training images (reference row)"))
    sb = pd.DataFrame(srows)
    p = os.path.join(args.out_dir, "r2_fives_subsample.csv")
    sb.to_csv(p, index=False)
    print("wrote", p, sb.shape)

    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print("\n[audit, primary skan panel]")
        print(au[au["panel"] == "primary"][
            ["block", "biomarker", "n", "offset", "offset_lo", "offset_hi",
             "resid_sd", "r_pearson"]].round(4).to_string(index=False))
        print("\n[downstream]")
        print(dn[["block", "featureset", "clf", "n", "auc_reference",
                  "auc_predicted", "delta_auc", "delta_auc_lo",
                  "delta_auc_hi"]].round(4).to_string(index=False))
        print("\n[sub-sampling sensitivity, skan4]")
        print(sb.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
