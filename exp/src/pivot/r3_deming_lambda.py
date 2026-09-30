"""R3 point 3 -- sensitivity of the Deming calibration slope to the error-variance
ratio lambda.

Reviewer point 3 (``review/user_cmig_review3_20260929.md``)
---------------------------------------------------------
"Report Deming lambda sensitivity: lambda in {0.25, 0.5, 1, 2, 4} and a range
estimated from inter-observer variability; check whether the HRF
length/tortuosity and FIVES FD slope conclusions are stable."

Model and convention (identical to ``src/pivot/r2_calibration.py``)
------------------------------------------------------------------
    B_pred = alpha + beta * B_ref, both observed with error,
    lambda = var(err_pred) / var(err_ref)            (y = pred, x = ref)
    beta   = ( Syy - lam*Sxx + sqrt((Syy - lam*Sxx)^2 + 4*lam*Sxy^2) ) / (2*Sxy)

on the sigma axis (frozen per-cohort sigma_b, results/gateA_biomarker_scales_
train.csv), centred on the cohort's reference median.  lambda -> infinity is
OLS of pred on ref (reference assumed exact, slope attenuated); lambda -> 0 is
the inverse regression 1 / OLS(ref on pred).  Both limits are written as
bracketing rows (``ols_y_on_x``, ``ols_x_on_y_inv``).

Inter-observer lambda (DRIVE, CHASE_DB1 -- the only cohorts with a second
observer; per-image obs1/obs2 skan biomarkers in results/fig4c_observer.csv)
-------------------------------------------------------------------------
    d_i    = (B_obs2,i - B_obs1,i) / sigma_b              (observer discrepancy)
    e_i    = (B_pred,i - B_ref,i)  / sigma_b              (machine - reference)

  lambda_obs_mom   (primary, method of moments)
        s2_ref  = var(d) / 2        per-observer reference error variance
                                    (two exchangeable observers with
                                    independent errors: var(d) = 2 s2_ref)
        s2_pred = var(e) - s2_ref   machine error variance, since
                                    var(e) = s2_pred + s2_ref if independent
        lambda  = s2_pred / s2_ref  (floored at 0.05 if var(e) <= s2_ref)
  lambda_obs_raw   (the reviewer's literal specification)
        lambda  = var(e) / var(d)
        i.e. the whole residual variance of pred against the whole observer-
        difference variance; it ignores that var(e) contains s2_ref and that
        var(d) contains two observers' error, and is reported as a second
        reading, not preferred.

  For each cohort with observers a 2000-draw bootstrap (observer images and
  test images resampled independently) gives a 95 % range of lambda; beta and
  its 2000-draw image-bootstrap CI are then reported at the point lambda and at
  both ends of that range.

  HRF and FIVES have ONE observer: lambda cannot be estimated from their own
  data and the "own" rows are NA.  An EXTRAPOLATED range is given by
  transporting s2_ref (in sigma_b units) from DRIVE and from CHASE_DB1 and
  combining it with the cohort's own var(e); these rows are labelled
  ``extrapolated`` and carry no confirmatory weight.

Bootstrap: 2000-draw paired image bootstrap of the (ref, pred) pairs; the
resample indices are drawn exactly as in ``r2_calibration.py`` (one
RandomState(0) stream over DATASETS x PRIMARY_COLS), so the lambda = 1 and
lambda = 4 rows reproduce ``results/pivot/r2/r2_calibration.csv`` bit-for-bit
(checked in ``r3_deming_repro_check.csv``).  The same resampled images enter
every lambda for a given cell, so the curves are paired.

Sensitivity cohort ``fives_all800`` (not in the paper's main table): all 800
FIVES images (held-out test 200 with pred_test + training 600 with cross-
fitted out-of-fold predictions, results/pivot/r2/bio_master_full.csv), its own
RandomState(1) stream.

Outputs (results/pivot/r3/)
    r3_deming_lambda.csv          dataset x biomarker x lambda_setting rows
    r3_deming_lambda_obs.csv      inter-observer lambda estimates
    r3_deming_conclusions.csv     do the three slope conclusions survive?
    r3_deming_repro_check.csv     lambda 1 / 4 against r2_calibration.csv
    r3_deming_stability.csv       per cell: beta(lambda=1) [CI], beta range over
                                  the grid, lambda-stable yes/no (definition
                                  fixed in DECISIONS 2026-09-29 22:06);
                                  main_text=True marks HRF length, HRF
                                  tortuosity and FIVES FD
    figs/pivot/r3_deming_lambda.{png,pdf}

CLI
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r3_deming_lambda
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import EXP_ROOT, PIVOT_DIR, PRIMARY_COLS, RESULTS_DIR, sigma_table

SEED = 0
N_BOOT = 2000
DATASETS = ("drive", "chasedb1", "hrf", "fives")
SKAN = ["density_skan", "total_length_skan", "FD_skan", "tortuosity_skan"]
LAMBDA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
LAM_FLOOR = 0.05
OBS_COHORTS = ("drive", "chasedb1")


# ------------------------------------------------------------------ Deming
def deming_vec(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """Deming slope for each row of X, Y (shape B x n).  lam may be np.inf
    (OLS y|x) or 0 (inverse OLS x|y)."""
    n = X.shape[1]
    mx = X.mean(axis=1, keepdims=True)
    my = Y.mean(axis=1, keepdims=True)
    sxx = ((X - mx) ** 2).sum(axis=1) / (n - 1)
    syy = ((Y - my) ** 2).sum(axis=1) / (n - 1)
    sxy = ((X - mx) * (Y - my)).sum(axis=1) / (n - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isinf(lam):
            b = sxy / sxx
        elif lam == 0:
            b = syy / sxy
        else:
            disc = (syy - lam * sxx) ** 2 + 4.0 * lam * sxy ** 2
            b = (syy - lam * sxx + np.sqrt(np.maximum(disc, 0.0))) / (2.0 * sxy)
    b = np.where(sxy == 0, np.nan, b)
    return b


def deming_alpha(x, y, b):
    return float(np.mean(y) - b * np.mean(x))


# ------------------------------------------------------------------ data
def cohort_pairs(master: pd.DataFrame, ds: str) -> pd.DataFrame:
    m = master[master["dataset"] == ds]
    gt = m[(m["source"] == "gt") & (m["split"] == "test")]
    pr = m[m["source"] == "pred_test"]
    cols = ["image_id"] + list(PRIMARY_COLS)
    return gt[cols].merge(pr[cols], on="image_id", suffixes=("__gt", "__pred"))


def fives800_pairs(full: pd.DataFrame) -> pd.DataFrame:
    m = full[full["dataset"] == "fives"]
    cols = ["image_id"] + list(PRIMARY_COLS)
    out = []
    for split, src in (("test", "pred_test"), ("train", "pred_oof")):
        gt = m[(m["source"] == "gt") & (m["split"] == split)]
        pr = m[(m["source"] == src) & (m["split"] == split)]
        out.append(gt[cols].merge(pr[cols], on="image_id",
                                  suffixes=("__gt", "__pred")))
    return pd.concat(out, ignore_index=True)


def observer_lambda(obs: pd.DataFrame, pairs: Dict[str, pd.DataFrame],
                    sig: Dict[str, Dict[str, float]]
                    ) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], dict]]:
    """lambda from the inter-observer data (own cohort) and extrapolated."""
    rows: List[dict] = []
    lut: Dict[Tuple[str, str], dict] = {}
    rng = np.random.RandomState(SEED + 7)
    s2ref_by = {}
    for ds in OBS_COHORTS:
        o = obs[obs["dataset"] == ds]
        for col in SKAN:
            s = sig[ds][col]
            d = ((o["obs2_" + col] - o["obs1_" + col]) / s).to_numpy(float)
            d = d[np.isfinite(d)]
            s2ref_by[(ds, col)] = (float(np.var(d, ddof=1) / 2.0), d)
    for ds in DATASETS:
        pr = pairs[ds]
        for col in SKAN:
            s = sig[ds][col]
            e = ((pr[col + "__pred"] - pr[col + "__gt"]) / s).to_numpy(float)
            e = e[np.isfinite(e)]
            ve = float(np.var(e, ddof=1))
            sources = ([("own", ds)] if ds in OBS_COHORTS else []) + \
                [("extrapolated", c) for c in OBS_COHORTS if c != ds]
            if ds not in OBS_COHORTS:
                rows.append(dict(dataset=ds, biomarker=col, basis="own",
                                 obs_source="--", n_obs=0, n_test=int(e.size),
                                 var_e=ve, s2_ref=np.nan, var_d=np.nan,
                                 lambda_mom=np.nan, lambda_mom_lo=np.nan,
                                 lambda_mom_hi=np.nan, lambda_raw=np.nan,
                                 lambda_raw_lo=np.nan, lambda_raw_hi=np.nan,
                                 floored=False,
                                 note="NA: one observer only"))
            for basis, src in sources:
                s2r, d = s2ref_by[(src, col)]
                # transport: d is in sigma_b units of its own cohort
                vd = 2.0 * s2r
                lam_mom = (ve - s2r) / s2r
                floored = bool(lam_mom < LAM_FLOOR)
                lam_mom = max(lam_mom, LAM_FLOOR)
                lam_raw = ve / vd
                bm, br = [], []
                for _ in range(N_BOOT):
                    db = d[rng.randint(0, d.size, d.size)]
                    eb = e[rng.randint(0, e.size, e.size)]
                    s2b = float(np.var(db, ddof=1) / 2.0)
                    veb = float(np.var(eb, ddof=1))
                    if s2b <= 0:
                        continue
                    bm.append(max((veb - s2b) / s2b, LAM_FLOOR))
                    br.append(veb / (2.0 * s2b))
                row = dict(dataset=ds, biomarker=col, basis=basis,
                           obs_source=src, n_obs=int(d.size),
                           n_test=int(e.size), var_e=ve, s2_ref=s2r,
                           var_d=vd, lambda_mom=lam_mom,
                           lambda_mom_lo=float(np.percentile(bm, 2.5)),
                           lambda_mom_hi=float(np.percentile(bm, 97.5)),
                           lambda_raw=lam_raw,
                           lambda_raw_lo=float(np.percentile(br, 2.5)),
                           lambda_raw_hi=float(np.percentile(br, 97.5)),
                           floored=floored,
                           note=("own inter-observer data" if basis == "own"
                                 else "s2_ref transported from %s in sigma_b "
                                      "units; no confirmatory weight" % src))
                rows.append(row)
                lut[(ds, col, basis, src)] = row
    return pd.DataFrame(rows), lut


# ------------------------------------------------------------------ main fit
def fit_cell(x: np.ndarray, y: np.ndarray, idx: np.ndarray,
             lam: float) -> dict:
    b = float(deming_vec(x[None, :], y[None, :], lam)[0])
    X, Y = x[idx], y[idx]
    keep = np.std(X, axis=1) > 0
    bb = deming_vec(X[keep], Y[keep], lam)
    bb = bb[np.isfinite(bb)]
    lo, hi = (float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))) \
        if bb.size else (np.nan, np.nan)
    return dict(beta=b, beta_lo=lo, beta_hi=hi,
                alpha=deming_alpha(x, y, b),
                beta_excludes_1=bool(np.isfinite(lo) and (lo > 1 or hi < 1)),
                beta_below_1=bool(b < 1), n_boot_ok=int(bb.size))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=os.path.join(PIVOT_DIR, "bio_master.csv"))
    ap.add_argument("--full", default=os.path.join(PIVOT_DIR, "r2",
                                                   "bio_master_full.csv"))
    ap.add_argument("--obs", default=os.path.join(RESULTS_DIR,
                                                  "fig4c_observer.csv"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r3"))
    ap.add_argument("--fig_dir", default=os.path.join(EXP_ROOT, "figs", "pivot"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    sig = sigma_table()
    master = pd.read_csv(args.master, low_memory=False)
    pairs = {ds: cohort_pairs(master, ds) for ds in DATASETS}

    # --- resample indices, drawn exactly as r2_calibration.calibration_rows
    rng_master = np.random.RandomState(SEED)
    cells: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for ds in DATASETS:
        d = pairs[ds]
        for col in PRIMARY_COLS:
            s = sig[ds][col]
            xg = d[col + "__gt"].to_numpy(float)
            yp = d[col + "__pred"].to_numpy(float)
            ok = np.isfinite(xg) & np.isfinite(yp)
            x, y = xg[ok], yp[ok]
            if x.size < 4:
                continue
            med = float(np.median(x))
            xs, ys = (x - med) / s, (y - med) / s
            idx = rng_master.randint(0, x.size, size=(N_BOOT, x.size))
            if col in SKAN:
                cells[(ds, col)] = (xs, ys, idx, med)
    # --- the 800-image FIVES sensitivity cohort
    full = pd.read_csv(args.full, low_memory=False)
    f8 = fives800_pairs(full)
    rng8 = np.random.RandomState(SEED + 1)
    for col in SKAN:
        s = sig["fives"][col]
        xg = f8[col + "__gt"].to_numpy(float)
        yp = f8[col + "__pred"].to_numpy(float)
        ok = np.isfinite(xg) & np.isfinite(yp)
        x, y = xg[ok], yp[ok]
        med = float(np.median(x))
        cells[("fives_all800", col)] = ((x - med) / s, (y - med) / s,
                                        rng8.randint(0, x.size,
                                                     size=(N_BOOT, x.size)),
                                        med)

    obs = pd.read_csv(args.obs)
    lam_obs, lut = observer_lambda(obs, pairs, sig)
    p = os.path.join(args.out_dir, "r3_deming_lambda_obs.csv")
    lam_obs.assign(source=("results/fig4c_observer.csv (obs1/obs2 skan "
                           "biomarkers: DRIVE 20 test, CHASE_DB1 28) and "
                           "results/pivot/bio_master.csv (test pairs); sigma_b "
                           "= results/gateA_biomarker_scales_train.csv; "
                           "2000-draw bootstrap of observer and test images")
                   ).to_csv(p, index=False)
    print("wrote", p, lam_obs.shape)

    rows: List[dict] = []
    for (ds, col), (xs, ys, idx, med) in cells.items():
        base = ds if ds != "fives_all800" else "fives"
        settings: List[Tuple[str, str, float]] = [
            ("grid", "lambda=%g" % l, l) for l in LAMBDA_GRID]
        settings += [("limit", "ols_y_on_x", np.inf),
                     ("limit", "ols_x_on_y_inv", 0.0)]
        for (d_, c_, basis, src), r in lut.items():
            if d_ != base or c_ != col:
                continue
            for kind in ("mom", "raw"):
                for end in ("", "_lo", "_hi"):
                    lam = r["lambda_" + kind + end]
                    settings.append((("obs_" if basis == "own" else "extrap_")
                                     + kind,
                                     "%s[%s]%s" % (kind, src, end or "_point"),
                                     float(lam)))
        for group, label, lam in settings:
            f = fit_cell(xs, ys, idx, lam)
            rows.append(dict(dataset=ds, biomarker=col, setting_group=group,
                             setting=label, lam=lam, n=int(xs.size),
                             ref_median_raw=med, **f))
    res = pd.DataFrame(rows)
    res["source"] = ("results/pivot/bio_master.csv (gt[test] vs pred_test, seg "
                     "seed 0; fives_all800: r2/bio_master_full.csv test200 "
                     "pred_test + train600 pred_oof); sigma_b frozen; centred on "
                     "the cohort reference median; 2000-draw paired image "
                     "bootstrap, same indices across lambda")
    p = os.path.join(args.out_dir, "r3_deming_lambda.csv")
    res.to_csv(p, index=False)
    print("wrote", p, res.shape)

    # --- reproduction check against r2_calibration.csv
    r2 = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "r2_calibration.csv"))
    chk = []
    for meth, lab in (("deming_l1", "lambda=1"), ("deming_l4", "lambda=4")):
        for (ds, col) in [k for k in cells if k[0] != "fives_all800"]:
            a = r2[(r2.dataset == ds) & (r2.biomarker == col) &
                   (r2.method == meth)].iloc[0]
            b = res[(res.dataset == ds) & (res.biomarker == col) &
                    (res.setting == lab)].iloc[0]
            chk.append(dict(dataset=ds, biomarker=col, method=meth,
                            beta_r2=a.beta, beta_r3=b.beta,
                            beta_lo_r2=a.beta_lo, beta_lo_r3=b.beta_lo,
                            beta_hi_r2=a.beta_hi, beta_hi_r3=b.beta_hi,
                            max_abs_diff=float(max(abs(a.beta - b.beta),
                                                   abs(a.beta_lo - b.beta_lo),
                                                   abs(a.beta_hi - b.beta_hi)))))
    chk = pd.DataFrame(chk)
    p = os.path.join(args.out_dir, "r3_deming_repro_check.csv")
    chk.to_csv(p, index=False)
    print("wrote", p, "max |diff| vs r2_calibration = %.3g"
          % chk["max_abs_diff"].max())

    # --- the three conclusions under every lambda
    concl = []
    main_settings = [s for s in res["setting"].unique()]
    for setting in main_settings:
        def get(ds, col):
            q = res[(res.dataset == ds) & (res.biomarker == col) &
                    (res.setting == setting)]
            return q.iloc[0] if len(q) else None
        h = {c: get("hrf", c) for c in SKAN}
        fv = {c: get("fives", c) for c in SKAN}
        f8r = {c: get("fives_all800", c) for c in SKAN}
        rec = dict(setting=setting)
        if all(v is not None for v in h.values()):
            rec.update(
                lam_hrf=";".join("%s=%.3g" % (c.split("_skan")[0], h[c].lam)
                                 for c in SKAN),
                hrf_all4_below1=all(h[c].beta < 1 for c in SKAN),
                hrf_3_below1_skan3=all(h[c].beta < 1 for c in SKAN[:3]),
                hrf_length_excl1=bool(h["total_length_skan"].beta_hi < 1),
                hrf_tort_excl1=bool(h["tortuosity_skan"].beta_hi < 1),
                hrf_beta_range="%.2f-%.2f" % (min(h[c].beta for c in SKAN),
                                              max(h[c].beta for c in SKAN)),
                hrf_length="%.2f [%.2f, %.2f]" % (
                    h["total_length_skan"].beta, h["total_length_skan"].beta_lo,
                    h["total_length_skan"].beta_hi),
                hrf_tort="%.2f [%.2f, %.2f]" % (
                    h["tortuosity_skan"].beta, h["tortuosity_skan"].beta_lo,
                    h["tortuosity_skan"].beta_hi))
        if fv["FD_skan"] is not None:
            r = fv["FD_skan"]
            rec.update(lam_fives_fd=r.lam,
                       fives_fd_gt1_excl1=bool(r.beta > 1 and r.beta_lo > 1),
                       fives_fd="%.2f [%.2f, %.2f]" % (r.beta, r.beta_lo,
                                                        r.beta_hi))
            t = fv["tortuosity_skan"]
            if t is not None:
                rec.update(fives_tort="%.2f [%.2f, %.2f]" % (t.beta, t.beta_lo,
                                                              t.beta_hi),
                           fives_tort_excl1=bool(t.beta_hi < 1))
            dn, ln = fv["density_skan"], fv["total_length_skan"]
            if dn is not None and ln is not None:
                rec.update(fives_density_len_cover1=bool(
                    dn.beta_lo <= 1 <= dn.beta_hi and
                    ln.beta_lo <= 1 <= ln.beta_hi))
        if f8r["FD_skan"] is not None:
            r = f8r["FD_skan"]
            rec.update(fives800_fd="%.2f [%.2f, %.2f]" % (r.beta, r.beta_lo,
                                                           r.beta_hi),
                       fives800_fd_gt1_excl1=bool(r.beta > 1 and
                                                  r.beta_lo > 1))
        concl.append(rec)
    concl = pd.DataFrame(concl)
    p = os.path.join(args.out_dir, "r3_deming_conclusions.csv")
    concl.to_csv(p, index=False)
    print("wrote", p, concl.shape)
    with pd.option_context("display.width", 250, "display.max_columns", 40,
                           "display.max_rows", 200):
        print(lam_obs[["dataset", "biomarker", "basis", "obs_source", "var_e",
                       "s2_ref", "lambda_mom", "lambda_mom_lo", "lambda_mom_hi",
                       "lambda_raw", "lambda_raw_lo", "lambda_raw_hi",
                       "floored"]].round(3).to_string(index=False))
        print(concl.to_string(index=False))

    # --- lambda-stability, definition fixed before looking (DECISIONS
    # 2026-09-29 22:06): a cell is lambda-stable iff its lambda = 1 verdict
    # (95 % interval excludes 1, or not) is unchanged for every lambda in the
    # grid {0.25, 0.5, 1, 2, 4}.
    st = []
    for (ds, col) in [k for k in cells]:
        g = res[(res.dataset == ds) & (res.biomarker == col) &
                (res.setting_group == "grid")].sort_values("lam")
        g1 = g[g.lam == 1.0].iloc[0]
        verdicts = g["beta_excludes_1"].astype(bool).tolist()
        o = res[(res.dataset == ds) & (res.biomarker == col) &
                (res.setting_group.isin(["obs_mom", "obs_raw", "extrap_mom",
                                          "extrap_raw"]))]
        st.append(dict(
            dataset=ds, biomarker=col, n=int(g1.n),
            beta_l1=g1.beta, beta_l1_lo=g1.beta_lo, beta_l1_hi=g1.beta_hi,
            verdict_l1=("excludes 1" if g1.beta_excludes_1 else "covers 1"),
            beta_min_grid=float(g.beta.min()), beta_max_grid=float(g.beta.max()),
            ci_lo_min_grid=float(g.beta_lo.min()),
            ci_hi_max_grid=float(g.beta_hi.max()),
            lambda_where_verdict_differs=";".join(
                "%g" % l for l, v in zip(g.lam, verdicts)
                if v != bool(g1.beta_excludes_1)) or "--",
            lambda_stable=bool(all(v == bool(g1.beta_excludes_1)
                                   for v in verdicts)),
            obs_lambda_settings=int(len(o)),
            obs_verdict_same_all=(bool((o["beta_excludes_1"].astype(bool) ==
                                        bool(g1.beta_excludes_1)).all())
                                  if len(o) else None),
            obs_beta_range=("%.2f-%.2f" % (o.beta.min(), o.beta.max())
                            if len(o) else "--"),
            main_text=bool((ds, col) in {("hrf", "total_length_skan"),
                                         ("hrf", "tortuosity_skan"),
                                         ("fives", "FD_skan")}),
            definition=("lambda-stable: the lambda=1 verdict (95 % image-"
                        "bootstrap interval of beta excludes 1, or not) is "
                        "unchanged for all lambda in {0.25,0.5,1,2,4}"),
            obs_note=("obs_* settings: lambda_mom = (var(e)-var(d)/2)/(var(d)/2)"
                      " and lambda_raw = var(e)/var(d), d = obs2-obs1, e = "
                      "pred-ref, sigma_b units; point and bootstrap 2.5/97.5 "
                      "ends; own data on DRIVE/CHASE_DB1, transported "
                      "(extrapolated) elsewhere")))
    st = pd.DataFrame(st)
    p = os.path.join(args.out_dir, "r3_deming_stability.csv")
    st.to_csv(p, index=False)
    print("wrote", p)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(st.drop(columns=["definition", "obs_note"]).round(3)
              .to_string(index=False))

    make_figure(res, lam_obs, args.fig_dir)
    return 0


def make_figure(res: pd.DataFrame, lam_obs: pd.DataFrame, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval.savefig_util import save_fig

    dss = ["drive", "chasedb1", "hrf", "fives"]
    lab = {"density_skan": "density", "total_length_skan": "total length",
           "FD_skan": "fractal dim.", "tortuosity_skan": "tortuosity"}
    fig, axes = plt.subplots(len(dss), len(SKAN), figsize=(10.5, 9.0),
                             sharex=True)
    for i, ds in enumerate(dss):
        for j, col in enumerate(SKAN):
            ax = axes[i, j]
            g = res[(res.dataset == ds) & (res.biomarker == col) &
                    (res.setting_group == "grid")].sort_values("lam")
            ax.fill_between(g.lam, g.beta_lo, g.beta_hi, color="#3b6ea5",
                            alpha=0.18, lw=0)
            ax.plot(g.lam, g.beta, "o-", color="#3b6ea5", ms=3.5, lw=1.2,
                    label="test set")
            if ds == "fives":
                g8 = res[(res.dataset == "fives_all800") &
                         (res.biomarker == col) &
                         (res.setting_group == "grid")].sort_values("lam")
                ax.plot(g8.lam, g8.beta, "s--", color="#c0504d", ms=3, lw=1.0,
                        label="all 800")
                ax.fill_between(g8.lam, g8.beta_lo, g8.beta_hi,
                                color="#c0504d", alpha=0.10, lw=0)
            ax.axhline(1.0, color="0.3", lw=0.8, ls=":")
            q = lam_obs[(lam_obs.dataset == ds) & (lam_obs.biomarker == col) &
                        lam_obs.lambda_mom.notna()]
            for r in q.itertuples():
                lo = max(r.lambda_mom_lo, 0.15)
                hi = min(r.lambda_mom_hi, 6.0)
                own = r.basis == "own"
                ax.axvspan(lo, hi, color=("#9bbb59" if own else "#e8a33d"),
                           alpha=(0.22 if own else 0.12), lw=0)
                if 0.15 <= r.lambda_mom <= 6:
                    ax.axvline(r.lambda_mom, color=("#5a7d2a" if own else
                                                    "#b0701a"),
                               lw=1.0, ls=("-" if own else "--"))
            ax.set_xscale("log")
            ax.set_xlim(0.2, 5.0)
            ax.set_xticks(list(LAMBDA_GRID))
            ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"], fontsize=7)
            from matplotlib.ticker import NullFormatter, NullLocator
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            lo_all = np.nanmin(g.beta_lo)
            hi_all = np.nanmax(g.beta_hi)
            ax.set_ylim(max(lo_all - 0.1, -1.0), min(hi_all + 0.1, 3.5))
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2, lw=0.5)
            if i == 0:
                ax.set_title(lab[col], fontsize=9)
            if j == 0:
                ax.set_ylabel("%s\nDeming beta" % ds.upper(), fontsize=8)
            if i == len(dss) - 1:
                ax.set_xlabel("lambda = var(err_pred)/var(err_ref)", fontsize=7)
            if i == 3 and j == 0:
                ax.legend(fontsize=6.5, frameon=False, loc="best")
    fig.suptitle("Deming slope against the error-variance ratio (95 % image "
                 "bootstrap band). Green: lambda from own inter-observer data; "
                 "orange: transported (extrapolated)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, os.path.join(fig_dir, "r3_deming_lambda.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
