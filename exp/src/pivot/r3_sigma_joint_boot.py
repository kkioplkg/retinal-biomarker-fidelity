"""R3 point 4 -- joint bootstrap of test images AND the training reference masks
from which the scale is estimated.

Reviewer point 4 (``review/user_cmig_review3_20260929.md``)
---------------------------------------------------------
"The sampling uncertainty of sigma_b does not enter the main confidence
intervals: add a joint image + sigma_b bootstrap, or state whether including it
changes the qualitative conclusions."

What is resampled
-----------------
Per-cohort scale sigma_b (the frozen scale of results/gateA_biomarker_scales_
train.csv, 1.4826 * MAD of training-split observer-1 reference biomarkers):
    DRIVE 20, CHASE_DB1 20, HRF 15 training images  -> results/gateA_biomarkers_gt.csv
    FIVES 120 training images                        -> results/fives_native_train_biomarkers.csv
  (recomputing 1.4826*MAD on exactly these images reproduces the frozen table
  to machine precision -- checked at run time, ``sigma_repro_ratio``).
Common scale s_k (results/pivot/r2/r2_common_scale_formula.txt): weighted
median of cohort-centred |deviations| of the training reference masks of the
four cohorts (bio_master.csv source=gt split=train; FOV-normalised length),
equal cohort weight.

Independence unit.  Images are independent in DRIVE, HRF and FIVES (one image
per subject_id in every split used here).  CHASE_DB1 is NOT: its 28 images are
the left and right eyes of 14 children (test 8 = 4 children, training 20 = 10
children), so on CHASE_DB1 every resample below draws CHILDREN (both eyes
together), for the test images and for the training masks of sigma_b alike.
The published r2 intervals resampled CHASE_DB1 images; the difference is
reported (``cond_unit`` columns of r3_main_table_joint.csv).

Three bootstraps, 2000 draws each, SEED = 0:
    image   test images resampled (paired across the four biomarkers of a
            cohort), scale FIXED at its frozen value -- the published interval
    joint   test images resampled AND, independently, the training images the
            scale is estimated from (within each cohort; for s_k within every
            cohort at once, since s_k pools all four)
    sigma   only the scale resampled (the test set fixed) -- isolates the scale's
            own contribution
The test-image indices are identical in ``image`` and ``joint``, so the width
ratio isolates what the scale adds.

Per dataset x biomarker (skan panel) the table reports the sigma-scaled offset
mu = mean(pred - ref)/scale and the residual SD sd(pred - ref)/scale.

Claims re-evaluated per draw (``r3_sigma_joint_claims.csv``):
    hrf_gt_fives_offset    HRF panel mean |mu| > FIVES panel mean |mu|
    hrf_gt_fives_sd        HRF panel mean resid SD > FIVES
    order_one              the four-cohort grand mean |mu| and resid SD, and
                           every cohort's panel mean, lie in [1/3, 3] sigma
                           ("of order one": within half an order of magnitude)
    topology_ratio         max topology-specific harm h_net over the primary
                           cells (design ``nn_within_image`` and the worst single
                           stratum of results/pivot/r2/r2_topology_*.csv) divided
                           into the four-cohort grand mean |mu| -- the "17-20x"
                           claim.  h_net is itself in sigma_b units, so each cell
                           is rescaled by sigma_frozen / sigma_draw; the event-
                           level sampling of h_net is NOT resampled (its own CIs
                           are in r2_topology_matched.csv).
Each claim is evaluated on the frozen four-biomarker panel (skan4) and on the
three-biomarker primary panel proposed in review point 5 (skan3: density,
total length, FD).

Outputs (results/pivot/r3/)
    r3_main_table_joint.csv      every column of results/pivot/r2/r2_main_table.csv
                                 (all 8 columns, same keys, same point values)
                                 plus, for mu / alpha / resid_sd: joint
                                 percentile and BCa intervals, the sigma-fixed
                                 interval at the true independence unit
                                 (cond_unit), and sigma_b [95 % CI].  For DRIVE,
                                 HRF and FIVES the test-image indices are the
                                 r2_calibration.py stream, so cond_unit equals the
                                 published interval exactly and joint differs
                                 from it only by the sigma_b draw.
    r3_sigma_joint_boot.csv      per dataset x biomarker x scale x bootstrap
    r3_sigma_joint_panel.csv     panel means per cohort, per bootstrap
    r3_sigma_joint_claims.csv    the claims
    r3_sigma_boot_scale.csv      the scale's own bootstrap CI
    figs/pivot/r3_sigma_joint_boot.{png,pdf}
CLI
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r3_sigma_joint_boot
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.pivot.common import EXP_ROOT, PIVOT_DIR, RESULTS_DIR, sigma_table

SEED = 0
N_BOOT = 2000
DATASETS = ("drive", "chasedb1", "hrf", "fives")
GATEA = {"drive": "DRIVE", "chasedb1": "CHASE_DB1", "hrf": "HRF"}
SKAN = ["density_skan", "total_length_skan", "FD_skan", "tortuosity_skan"]
PANELS = {"skan4": SKAN, "skan3": SKAN[:3]}
LENGTH_LIKE = {"total_length"}
ORDER_ONE = (1.0 / 3.0, 3.0)


def mad_scale(a: np.ndarray, axis=-1) -> np.ndarray:
    med = np.nanmedian(a, axis=axis, keepdims=True)
    out = 1.4826 * np.nanmedian(np.abs(a - med), axis=axis)
    # a resample in which more than half the draws are one image has MAD = 0:
    # the scale is undefined there and the draw is dropped (counted in
    # r3_sigma_boot_scale.csv::frac_zero_draws)
    return np.where(out > 0, out, np.nan)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w)
    return float(x[int(np.searchsorted(c, 0.5 * c[-1]))])


def common_scale(vals: Dict[str, np.ndarray]) -> float:
    d, w = [], []
    for x in vals.values():
        x = x[np.isfinite(x)]
        d.append(np.abs(x - np.median(x)))
        w.append(np.full(x.size, 1.0 / x.size))
    return 1.4826 * weighted_median(np.concatenate(d), np.concatenate(w))


def fovf(df: pd.DataFrame, col: str) -> np.ndarray:
    if col.rsplit("_", 1)[0] in LENGTH_LIKE:
        d = df["disc_fov_diameter"].to_numpy(float)
        return np.where(np.isfinite(d) & (d > 0), d, np.nan)
    return np.ones(len(df))


CLUSTERED = {"chasedb1"}      # left/right eyes of the same child


def unit_labels(ds: str, df: pd.DataFrame) -> np.ndarray:
    """Independence-unit label per row: subject for CHASE_DB1, else the row."""
    if ds in CLUSTERED:
        return df["subject_id"].astype(str).to_numpy()
    return np.arange(len(df)).astype(str)


def draw_idx(labels: np.ndarray, B: int, rng: np.random.RandomState
             ) -> np.ndarray:
    """B x n row indices, resampling whole units (clusters of equal size)."""
    uniq, inv = np.unique(labels, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    sizes = {len(m) for m in members}
    if len(sizes) != 1:
        raise ValueError("unequal cluster sizes: %s" % sorted(sizes))
    M = np.stack(members)                      # nc x k
    pick = rng.randint(0, len(uniq), size=(B, len(uniq)))
    return M[pick].reshape(B, -1)


def pct(v: np.ndarray):
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) \
        if v.size else (np.nan, np.nan)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=os.path.join(PIVOT_DIR, "bio_master.csv"))
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r3"))
    ap.add_argument("--fig_dir", default=os.path.join(EXP_ROOT, "figs", "pivot"))
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    sig = sigma_table()
    master = pd.read_csv(args.master, low_memory=False)
    gA = pd.read_csv(os.path.join(RESULTS_DIR, "gateA_biomarkers_gt.csv"))
    f120 = pd.read_csv(os.path.join(RESULTS_DIR,
                                    "fives_native_train_biomarkers.csv"))

    # ---------------- data
    D, Dc, T, Tc, S0 = {}, {}, {}, {}, {}
    UD, UT, UTc = {}, {}, {}
    for ds in DATASETS:
        m = master[master["dataset"] == ds]
        gt = m[(m.source == "gt") & (m.split == "test")]
        pr = m[m.source == "pred_test"]
        cols = ["image_id", "subject_id", "disc_fov_diameter"] + SKAN
        d = gt[cols].merge(pr[["image_id"] + SKAN], on="image_id",
                           suffixes=("__gt", "__pred"))
        d = d.rename(columns={"disc_fov_diameter": "disc_fov_diameter"})
        D[ds] = np.column_stack([(d[c + "__pred"] - d[c + "__gt"]).to_numpy(float)
                                 for c in SKAN])
        Dc[ds] = np.column_stack([((d[c + "__pred"] - d[c + "__gt"]) /
                                   fovf(d, c)).to_numpy(float) for c in SKAN])
        if ds in GATEA:
            t = gA[(gA.dataset == GATEA[ds]) & (gA.observer == "obs1") &
                   (gA.split == "train")]
        else:
            t = f120
        T[ds] = t[SKAN].to_numpy(float)
        UD[ds] = unit_labels(ds, d)
        UT[ds] = unit_labels(ds, t.reset_index(drop=True))
        tr = m[(m.source == "gt") & (m.split == "train")].reset_index(drop=True)
        UTc[ds] = unit_labels(ds, tr)
        Tc[ds] = np.column_stack([tr[c].to_numpy(float) / fovf(tr, c)
                                  for c in SKAN])
        S0[ds] = np.array([sig[ds][c] for c in SKAN])
    repro = {ds: mad_scale(T[ds].T) / S0[ds] for ds in DATASETS}
    for ds in DATASETS:
        assert np.allclose(repro[ds], 1.0, rtol=1e-9), (ds, repro[ds])
    SC0 = np.array([common_scale({ds: Tc[ds][:, j] for ds in DATASETS})
                    for j in range(4)])

    # ---------------- draws
    rng = np.random.RandomState(SEED)
    I = {ds: draw_idx(UD[ds], N_BOOT, rng) for ds in DATASETS}
    J = {ds: draw_idx(UT[ds], N_BOOT, rng) for ds in DATASETS}
    Jc = {ds: draw_idx(UTc[ds], N_BOOT, rng) for ds in DATASETS}
    # scale draws: (B x 4)
    Sb = {ds: np.stack([mad_scale(T[ds][J[ds], j]) for j in range(4)], axis=1)
          for ds in DATASETS}
    SCb = np.zeros((N_BOOT, 4))
    for b in range(N_BOOT):
        for j in range(4):
            SCb[b, j] = common_scale({ds: Tc[ds][Jc[ds][b], j]
                                      for ds in DATASETS})

    # raw moments of the test differences under image resampling (B x 4)
    def moments(X, idx):
        Xb = X[idx]                          # B x n x 4
        with np.errstate(invalid="ignore"):
            return np.nanmean(Xb, axis=1), np.nanstd(Xb, axis=1, ddof=1)
    M, SD, Mc, SDc = {}, {}, {}, {}
    for ds in DATASETS:
        M[ds], SD[ds] = moments(D[ds], I[ds])
        Mc[ds], SDc[ds] = moments(Dc[ds], I[ds])
    m0 = {ds: np.nanmean(D[ds], axis=0) for ds in DATASETS}
    sd0 = {ds: np.nanstd(D[ds], axis=0, ddof=1) for ds in DATASETS}
    mc0 = {ds: np.nanmean(Dc[ds], axis=0) for ds in DATASETS}
    sdc0 = {ds: np.nanstd(Dc[ds], axis=0, ddof=1) for ds in DATASETS}

    # per (scale, boot) -> dict ds -> (mu B x 4, sd B x 4)
    draws = {}
    for ds in DATASETS:
        draws[("own", "image", ds)] = (M[ds] / S0[ds], SD[ds] / S0[ds])
        draws[("own", "joint", ds)] = (M[ds] / Sb[ds], SD[ds] / Sb[ds])
        draws[("own", "sigma", ds)] = (m0[ds][None] / Sb[ds],
                                       sd0[ds][None] / Sb[ds])
        draws[("common", "image", ds)] = (Mc[ds] / SC0, SDc[ds] / SC0)
        draws[("common", "joint", ds)] = (Mc[ds] / SCb, SDc[ds] / SCb)
        draws[("common", "sigma", ds)] = (mc0[ds][None] / SCb,
                                          sdc0[ds][None] / SCb)
    point = {}
    for ds in DATASETS:
        point[("own", ds)] = (m0[ds] / S0[ds], sd0[ds] / S0[ds])
        point[("common", ds)] = (mc0[ds] / SC0, sdc0[ds] / SC0)

    # ---------------- per-cell table
    rows: List[dict] = []
    for scale in ("own", "common"):
        for ds in DATASETS:
            mu_p, sd_p = point[(scale, ds)]
            for j, col in enumerate(SKAN):
                rec = dict(dataset=ds, biomarker=col, scale=scale,
                           scale_value=(S0[ds][j] if scale == "own" else SC0[j]),
                           n_test=int(np.isfinite(D[ds][:, j]).sum()),
                           n_scale_images=(len(T[ds]) if scale == "own"
                                           else sum(len(Tc[x]) for x in DATASETS)),
                           mu=mu_p[j], resid_sd=sd_p[j])
                for bt in ("image", "joint", "sigma"):
                    mu_b, sd_b = draws[(scale, bt, ds)]
                    lo, hi = pct(mu_b[:, j])
                    slo, shi = pct(sd_b[:, j])
                    rec.update({f"mu_lo_{bt}": lo, f"mu_hi_{bt}": hi,
                                f"sd_lo_{bt}": slo, f"sd_hi_{bt}": shi})
                rec["mu_width_ratio_joint_over_image"] = (
                    (rec["mu_hi_joint"] - rec["mu_lo_joint"]) /
                    (rec["mu_hi_image"] - rec["mu_lo_image"]))
                rec["sd_width_ratio_joint_over_image"] = (
                    (rec["sd_hi_joint"] - rec["sd_lo_joint"]) /
                    (rec["sd_hi_image"] - rec["sd_lo_image"]))
                rec["mu_excl0_image"] = bool(rec["mu_lo_image"] > 0 or
                                             rec["mu_hi_image"] < 0)
                rec["mu_excl0_joint"] = bool(rec["mu_lo_joint"] > 0 or
                                             rec["mu_hi_joint"] < 0)
                rows.append(rec)
    cell = pd.DataFrame(rows)
    cell["source"] = ("test pairs: results/pivot/bio_master.csv (gt[test] vs "
                      "pred_test, seg seed 0); sigma_b images: "
                      "results/gateA_biomarkers_gt.csv (obs1, train) and "
                      "results/fives_native_train_biomarkers.csv (120); common "
                      "scale: bio_master.csv gt[train], FOV-normalised length; "
                      "2000 draws, SEED 0")
    p = os.path.join(args.out_dir, "r3_sigma_joint_boot.csv")
    cell.to_csv(p, index=False)
    print("wrote", p, cell.shape)

    # scale's own CI
    srows = []
    for ds in DATASETS:
        for j, col in enumerate(SKAN):
            lo, hi = pct(Sb[ds][:, j])
            srows.append(dict(dataset=ds, biomarker=col, scale="own",
                              n=len(T[ds]), value=S0[ds][j], lo=lo, hi=hi,
                              rel_halfwidth=(hi - lo) / 2 / S0[ds][j],
                              frac_zero_draws=float(np.mean(~np.isfinite(Sb[ds][:, j]))),
                              sigma_repro_ratio=float(repro[ds][j])))
    for j, col in enumerate(SKAN):
        lo, hi = pct(SCb[:, j])
        srows.append(dict(dataset="all4", biomarker=col, scale="common",
                          n=sum(len(Tc[x]) for x in DATASETS), value=SC0[j],
                          lo=lo, hi=hi, rel_halfwidth=(hi - lo) / 2 / SC0[j],
                          sigma_repro_ratio=np.nan))
    sc = pd.DataFrame(srows)
    p = os.path.join(args.out_dir, "r3_sigma_boot_scale.csv")
    sc.to_csv(p, index=False)
    print("wrote", p)

    # ---------------- topology numerators
    tm = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "r2_topology_matched.csv"))
    ts = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "r2_topology_strata.csv"))
    topo = {}
    for design in ("cem_L0_edit_burden", "nn_within_image"):
        q = tm[(tm.design == design) & (tm.biomarker.isin(SKAN))]
        topo[design] = {(r.dataset, r.biomarker): abs(r.h_net)
                        for r in q.itertuples()}
    q = ts[ts.biomarker.isin(SKAN)].groupby(["dataset", "biomarker"])["h_net"] \
        .apply(lambda v: float(np.nanmax(np.abs(v))))
    topo["worst_stratum"] = q.to_dict()

    # ---------------- panel means and claims
    prow, crow = [], []
    for scale in ("own", "common"):
        for bt in ("point", "image", "joint", "sigma"):
            for pname, cols in PANELS.items():
                jj = [SKAN.index(c) for c in cols]
                pm_mu, pm_sd = {}, {}
                for ds in DATASETS:
                    if bt == "point":
                        mu, sd = point[(scale, ds)]
                        mu, sd = mu[None], sd[None]
                    else:
                        mu, sd = draws[(scale, bt, ds)]
                    pm_mu[ds] = np.nanmean(np.abs(mu[:, jj]), axis=1)
                    pm_sd[ds] = np.nanmean(sd[:, jj], axis=1)
                grand_mu = np.mean([pm_mu[d] for d in DATASETS], axis=0)
                grand_sd = np.mean([pm_sd[d] for d in DATASETS], axis=0)
                for ds in DATASETS:
                    lo, hi = pct(pm_mu[ds])
                    slo, shi = pct(pm_sd[ds])
                    prow.append(dict(scale=scale, bootstrap=bt, panel=pname,
                                     dataset=ds,
                                     mean_abs_mu=float(np.median(pm_mu[ds]))
                                     if bt != "point" else float(pm_mu[ds][0]),
                                     lo=lo, hi=hi,
                                     mean_resid_sd=float(np.median(pm_sd[ds]))
                                     if bt != "point" else float(pm_sd[ds][0]),
                                     sd_lo=slo, sd_hi=shi))
                if bt == "point":
                    continue
                ratio_mu = pm_mu["hrf"] / pm_mu["fives"]
                ratio_sd = pm_sd["hrf"] / pm_sd["fives"]
                in1 = np.ones(N_BOOT, bool)
                for arr in [grand_mu, grand_sd] + [pm_mu[d] for d in DATASETS] \
                        + [pm_sd[d] for d in DATASETS]:
                    in1 &= (arr >= ORDER_ONE[0]) & (arr <= ORDER_ONE[1])
                in1g = ((grand_mu >= ORDER_ONE[0]) & (grand_mu <= ORDER_ONE[1])
                        & (grand_sd >= ORDER_ONE[0]) & (grand_sd <= ORDER_ONE[1]))
                rec = dict(scale=scale, bootstrap=bt, panel=pname,
                           hrf_over_fives_mu_lo=pct(ratio_mu)[0],
                           hrf_over_fives_mu_hi=pct(ratio_mu)[1],
                           p_hrf_gt_fives_mu=float(np.mean(ratio_mu > 1)),
                           hrf_over_fives_sd_lo=pct(ratio_sd)[0],
                           hrf_over_fives_sd_hi=pct(ratio_sd)[1],
                           p_hrf_gt_fives_sd=float(np.mean(ratio_sd > 1)),
                           grand_mu_lo=pct(grand_mu)[0],
                           grand_mu_hi=pct(grand_mu)[1],
                           grand_sd_lo=pct(grand_sd)[0],
                           grand_sd_hi=pct(grand_sd)[1],
                           p_grand_in_order_one=float(np.mean(in1g)),
                           p_all_cohort_means_in_order_one=float(np.mean(in1)))
                if scale == "own":
                    for design, lut in topo.items():
                        num = np.full(N_BOOT, np.nan)
                        for ds in DATASETS:
                            for j in jj:
                                h = lut.get((ds, SKAN[j]), np.nan)
                                if not np.isfinite(h):
                                    continue
                                if bt in ("joint", "sigma"):
                                    hb = h * S0[ds][j] / Sb[ds][:, j]
                                else:
                                    hb = np.full(N_BOOT, h)
                                num = np.fmax(num, hb)
                        rat = grand_mu / num
                        rsd = grand_sd / num
                        rec[f"topo_{design}_ratio_mu_lo"] = pct(rat)[0]
                        rec[f"topo_{design}_ratio_mu_hi"] = pct(rat)[1]
                        rec[f"topo_{design}_ratio_sd_lo"] = pct(rsd)[0]
                        rec[f"topo_{design}_ratio_sd_hi"] = pct(rsd)[1]
                        rec[f"topo_{design}_p_ratio_ge10"] = float(np.mean(rat >= 10))
                crow.append(rec)
    # point-estimate topology ratios, for reference
    for pname, cols in PANELS.items():
        jj = [SKAN.index(c) for c in cols]
        g = np.mean([np.mean(np.abs(point[("own", d)][0][jj])) for d in DATASETS])
        gs = np.mean([np.mean(point[("own", d)][1][jj]) for d in DATASETS])
        rec = dict(scale="own", bootstrap="point", panel=pname,
                   grand_mu_lo=g, grand_mu_hi=g, grand_sd_lo=gs, grand_sd_hi=gs)
        for design, lut in topo.items():
            num = max(lut.get((d, SKAN[j]), 0.0) for d in DATASETS for j in jj)
            rec[f"topo_{design}_ratio_mu_lo"] = rec[f"topo_{design}_ratio_mu_hi"] = g / num
            rec[f"topo_{design}_ratio_sd_lo"] = rec[f"topo_{design}_ratio_sd_hi"] = gs / num
        crow.append(rec)
    panel = pd.DataFrame(prow)
    claims = pd.DataFrame(crow)
    p = os.path.join(args.out_dir, "r3_sigma_joint_panel.csv")
    panel.to_csv(p, index=False)
    print("wrote", p)
    p = os.path.join(args.out_dir, "r3_sigma_joint_claims.csv")
    claims.to_csv(p, index=False)
    print("wrote", p)

    with pd.option_context("display.width", 260, "display.max_columns", 60,
                           "display.max_rows", 200):
        print(sc.round(4).to_string(index=False))
        show = cell[cell.scale == "own"][
            ["dataset", "biomarker", "mu", "mu_lo_image", "mu_hi_image",
             "mu_lo_joint", "mu_hi_joint", "mu_width_ratio_joint_over_image",
             "resid_sd", "sd_lo_image", "sd_hi_image", "sd_lo_joint",
             "sd_hi_joint", "sd_width_ratio_joint_over_image"]]
        print(show.round(3).to_string(index=False))
        print(cell[cell.scale == "common"][
            ["dataset", "biomarker", "mu", "mu_lo_image", "mu_hi_image",
             "mu_lo_joint", "mu_hi_joint", "mu_width_ratio_joint_over_image",
             "sd_width_ratio_joint_over_image"]].round(3).to_string(index=False))
        print(panel.round(3).to_string(index=False))
        print(claims.round(3).T.to_string())

    make_figure(cell, panel, args.fig_dir)
    t4, cmp_ = main_table_joint(master, sig, gA, f120)
    p = os.path.join(args.out_dir, "r3_main_table_joint.csv")
    t4.to_csv(p, index=False)
    print("wrote", p, t4.shape)
    p = os.path.join(args.out_dir, "r3_sigma_joint_width_compare.csv")
    cmp_.to_csv(p, index=False)
    print("wrote", p, cmp_.shape)
    with pd.option_context("display.width", 260, "display.max_columns", 60,
                           "display.max_rows", 200):
        print(cmp_[cmp_.biomarker.str.endswith("_skan")].round(3)
              .to_string(index=False))
    return 0



# ------------------------------------------------ Table-4 (r2_main_table) feed
def bca(theta: float, boots: np.ndarray, jack: np.ndarray):
    """BCa 95 % interval; acceleration from the jackknife values ``jack``."""
    from scipy.stats import norm
    b = boots[np.isfinite(boots)]
    jack = jack[np.isfinite(jack)]
    if b.size < 10 or jack.size < 3 or not np.isfinite(theta):
        return (np.nan, np.nan)
    prop = np.clip(np.mean(b < theta), 1.0 / b.size, 1 - 1.0 / b.size)
    z0 = norm.ppf(prop)
    L = jack.mean() - jack
    den = 6.0 * np.sum(L ** 2) ** 1.5
    a = float(np.sum(L ** 3) / den) if den > 0 else 0.0
    out = []
    for al in (0.025, 0.975):
        z = norm.ppf(al)
        q = norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
        out.append(float(np.percentile(b, 100 * q)))
    return tuple(out)


def raw_stats(X: np.ndarray, Y: np.ndarray):
    """mu, Deming(lambda=1) alpha and residual SD in RAW centred units, per row."""
    from src.pivot.r3_deming_lambda import deming_vec
    b = deming_vec(X, Y, 1.0)
    a = Y.mean(axis=1) - b * X.mean(axis=1)
    res = Y - (a[:, None] + b[:, None] * X)
    return dict(mu=(Y - X).mean(axis=1), alpha=a,
                resid_sd=res.std(axis=1, ddof=1), beta=b)


def main_table_joint(master: pd.DataFrame, sig, gA: pd.DataFrame,
                     f120: pd.DataFrame):
    from src.pivot.common import PRIMARY_COLS
    r2 = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "r2_main_table.csv"))
    rng_master = np.random.RandomState(SEED)      # the r2_calibration stream
    rng_c = np.random.RandomState(SEED + 21)
    rng_t = np.random.RandomState(SEED + 22)
    rows, cmp_ = [], []
    for ds in DATASETS:
        m = master[master["dataset"] == ds]
        gt = m[(m.source == "gt") & (m.split == "test")]
        pr = m[m.source == "pred_test"]
        cols = ["image_id", "subject_id"] + list(PRIMARY_COLS)
        d = gt[cols].merge(pr[["image_id"] + list(PRIMARY_COLS)],
                           on="image_id", suffixes=("__gt", "__pred"))
        t = (gA[(gA.dataset == GATEA[ds]) & (gA.observer == "obs1") &
                (gA.split == "train")] if ds in GATEA else f120
             ).reset_index(drop=True)
        ut = unit_labels(ds, t)
        J = draw_idx(ut, N_BOOT, rng_t)
        tunits = np.unique(ut)
        for col in PRIMARY_COLS:
            xg = d[col + "__gt"].to_numpy(float)
            yp = d[col + "__pred"].to_numpy(float)
            ok = np.isfinite(xg) & np.isfinite(yp)
            x, y = xg[ok], yp[ok]
            if x.size < 4:
                continue
            idx_pub = rng_master.randint(0, x.size, size=(N_BOOT, x.size))
            med = float(np.median(x))
            xc, yc = x - med, y - med
            if ds in CLUSTERED:
                lab = d.loc[ok, "subject_id"].astype(str).to_numpy()
                idx_u = draw_idx(lab, N_BOOT, rng_c)
            else:
                lab = np.arange(x.size).astype(str)
                idx_u = idx_pub
            s0 = float(sig[ds][col])
            tv = t[col].to_numpy(float)
            sb = mad_scale(tv[J])                     # B
            pt = {k: float(v[0]) for k, v in
                  raw_stats(xc[None], yc[None]).items()}
            bu = raw_stats(xc[idx_u], yc[idx_u])
            bp = raw_stats(xc[idx_pub], yc[idx_pub])
            # jackknife: leave one test unit out (sigma fixed), leave one
            # training unit out (test fixed)
            jk_test = {k: [] for k in ("mu", "alpha", "resid_sd")}
            for u in np.unique(lab):
                keep = lab != u
                st = raw_stats(xc[keep][None], yc[keep][None])
                for k in jk_test:
                    jk_test[k].append(float(st[k][0]))
            s_jk = np.array([mad_scale(tv[ut != u]) for u in tunits], float)
            r0 = r2[(r2.dataset == ds) & (r2.biomarker == col)]
            rec = r0.iloc[0].to_dict() if len(r0) else dict(dataset=ds,
                                                            biomarker=col)
            sl, sh = pct(sb)
            rec.update(sigma=s0, sigma_lo=sl, sigma_hi=sh,
                       sigma_n_units=int(len(tunits)),
                       sigma_n_images=int(np.isfinite(tv).sum()),
                       resample_unit=("child (both eyes)" if ds in CLUSTERED
                                      else "image"))
            for k in ("mu", "alpha", "resid_sd"):
                th = pt[k] / s0
                cond = bu[k] / s0
                joint = bu[k] / sb
                pub = bp[k] / s0
                jack = np.concatenate([np.array(jk_test[k]) / s0,
                                       pt[k] / s_jk])
                lo_c, hi_c = pct(cond)
                lo_j, hi_j = pct(joint)
                lo_b, hi_b = bca(th, joint, jack)
                lo_p, hi_p = pct(pub)
                rec.update({f"{k}_cond_unit_lo": lo_c, f"{k}_cond_unit_hi": hi_c,
                            f"{k}_joint_lo": lo_j, f"{k}_joint_hi": hi_j,
                            f"{k}_joint_bca_lo": lo_b, f"{k}_joint_bca_hi": hi_b})
                pub_lo = rec.get(f"{k}_lo", lo_p)
                pub_hi = rec.get(f"{k}_hi", hi_p)
                w_pub = pub_hi - pub_lo
                cmp_.append(dict(
                    dataset=ds, biomarker=col, statistic=k, point=th,
                    published_lo=pub_lo, published_hi=pub_hi,
                    repro_published_lo=lo_p, repro_published_hi=hi_p,
                    cond_unit_lo=lo_c, cond_unit_hi=hi_c,
                    joint_lo=lo_j, joint_hi=hi_j,
                    joint_bca_lo=lo_b, joint_bca_hi=hi_b,
                    width_published=w_pub,
                    width_ratio_cond_unit=(hi_c - lo_c) / w_pub,
                    width_ratio_joint=(hi_j - lo_j) / w_pub,
                    width_ratio_joint_bca=(hi_b - lo_b) / w_pub,
                    excl0_published=bool(pub_lo > 0 or pub_hi < 0),
                    excl0_joint=bool(lo_j > 0 or hi_j < 0),
                    excl0_joint_bca=bool(lo_b > 0 or hi_b < 0),
                    resample_unit=rec["resample_unit"]))
            rec["joint_source"] = (
                "r3_sigma_joint_boot.py: 2000 draws; test resample = the "
                "r2_calibration.py index stream (DRIVE/HRF/FIVES) or children "
                "(CHASE_DB1); sigma_b re-estimated (1.4826*MAD) in every draw "
                "from resampled training reference masks; BCa acceleration from "
                "the joint leave-one-unit-out jackknife. beta/CCC/r are "
                "scale-free and unchanged.")
            rows.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(cmp_)


def make_figure(cell: pd.DataFrame, panel: pd.DataFrame, fig_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval.savefig_util import save_fig

    lab = {"density_skan": "density", "total_length_skan": "length",
           "FD_skan": "FD", "tortuosity_skan": "tortuosity"}
    c = cell[cell.scale == "own"].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.6), sharey=True)
    ylabels = []
    y = 0
    ys = []
    for ds in DATASETS:
        for col in SKAN:
            ys.append(y)
            ylabels.append(f"{ds.upper()} {lab[col]}")
            y += 1
        pm = panel[(panel.scale == "own") & (panel.dataset == ds) &
                   (panel.panel == "skan4")]
        ys.append(y)
        ylabels.append(f"{ds.upper()} panel mean |mu| / SD")
        y += 1.6
    for k, (ax, what) in enumerate(zip(axes, ("mu", "sd"))):
        i = 0
        for ds in DATASETS:
            for col in SKAN:
                r = c[(c.dataset == ds) & (c.biomarker == col)].iloc[0]
                pt = r["mu"] if what == "mu" else r["resid_sd"]
                for bt, off, colr in (("image", -0.15, "#3b6ea5"),
                                      ("joint", 0.15, "#c0504d")):
                    lo, hi = r[f"{what}_lo_{bt}"], r[f"{what}_hi_{bt}"]
                    ax.plot([lo, hi], [ys[i] + off] * 2, color=colr, lw=2)
                    ax.plot(pt, ys[i] + off, "o", color=colr, ms=3.5)
                i += 1
            pm = panel[(panel.scale == "own") & (panel.dataset == ds) &
                       (panel.panel == "skan4")]
            pt = pm[pm.bootstrap == "point"]
            val = float(pt["mean_abs_mu" if what == "mu" else "mean_resid_sd"].iloc[0])
            for bt, off, colr in (("image", -0.15, "#3b6ea5"),
                                  ("joint", 0.15, "#c0504d")):
                q = pm[pm.bootstrap == bt].iloc[0]
                lo, hi = (q.lo, q.hi) if what == "mu" else (q.sd_lo, q.sd_hi)
                ax.plot([lo, hi], [ys[i] + off] * 2, color=colr, lw=3)
                ax.plot(val, ys[i] + off, "D", color=colr, ms=4)
            i += 1
        ax.axvline(0 if what == "mu" else 1, color="0.4", lw=0.8, ls=":")
        ax.set_xlabel("offset mu  [sigma_b]" if what == "mu"
                      else "residual SD  [sigma_b]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2, lw=0.5)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(ylabels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].plot([], [], color="#3b6ea5", lw=2, label="image bootstrap (sigma_b fixed)")
    axes[0].plot([], [], color="#c0504d", lw=2,
                 label="joint: images + sigma_b training masks")
    axes[0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle("95 % intervals with and without the sampling uncertainty of "
                 "sigma_b (2000 draws; skan panel)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, os.path.join(fig_dir, "r3_sigma_joint_boot.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
