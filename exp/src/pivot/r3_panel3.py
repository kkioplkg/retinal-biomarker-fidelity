"""R3 point 5 -- every panel-mean summary recomputed on a three-biomarker panel
(density, total length, FD), side by side with the locked four-biomarker skan
panel, plus the reference-gap Delta AUC with a ``skan3`` featureset.

Status (DECISIONS 2026-09-29 22:06): the three-biomarker summaries are an
ADDED POST-HOC REPORTING HIERARCHY.  The locked analysis (skan4 downstream
panel, all four-biomarker outputs) is untouched and stays primary for every
downstream number; skan3 is a sensitivity analysis.  Tortuosity is not
dropped: every tortuosity number stays reported in full per biomarker; it is
only excluded from the three-biomarker MEANS.  No locked output file is
modified -- this script writes only under results/pivot/r3/.

Sections and outputs (results/pivot/r3/)
    r3_panel3_audit.csv          tab_audit panel means (offset, resid SD, r, rho,
                                 topology mean|dB| and RMS, observer, view)
                                 per cohort + four-cohort grand means;
                                 sources e1_decomposition.csv,
                                 e1_topology_image.csv (rigr_uniform, pooled)
    r3_panel3_textranges.csv     every range / count quoted in the text that
                                 is taken over the panel (topology ranges,
                                 slope/CCC ranges, proportional-bias count,
                                 topology event bracket, stress-table macro
                                 |e| change, family means, topomatch ratios,
                                 scale-variant ratios, FIVES subsample means,
                                 case-figure macro error)
    r3_panel3_family.csv         Table (family) means, skan4 vs skan3
    r3_panel3_topomatch.csv      topomatch ratios, skan4 vs skan3
    r3_panel3_scale.csv          13 scale variants: HRF/FIVES ratios and axis
                                 separation, skan4 vs skan3
    r3_panel3_fives_subsample.csv FIVES 200-image subsamples, skan4 vs skan3
    r3_panel3_gap.csv            reference gap, skan4 (reproduces the locked
                                 value) vs skan3, with a paired bootstrap of
                                 gap(skan3) - gap(skan4) on the SAME resampled
                                 images; HRF (p2_predictions, n=45), FIVES
                                 p2-400, FIVES test200 / train600 / all800
                                 (r2/bio_master_full.csv), and the two
                                 second-segmenter families (HRF, FIVES test200)
    External label-only cohorts (APTOS-2019, IDRiD, Messidor-2): not run.  They
    have no reference masks, so no reference gap exists, and
    src/pivot/e3_external.py hard-codes the four-biomarker featureset (no panel
    argument); the skan4 E3 results stand as locked.

CLI
    cd exp
    OMP_NUM_THREADS=1 python -m src.pivot.r3_panel3
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.pivot.common import (PIVOT_DIR, PRIMARY_COLS, RESULTS_DIR, RUNS_DIR,
                              sigma_table)

SEED = 0
N_BOOT_AUC = 1000
DATASETS = ("drive", "chasedb1", "hrf", "fives")
SKAN4 = ["density_skan", "total_length_skan", "FD_skan", "tortuosity_skan"]
SKAN3 = SKAN4[:3]
PANELS = {"skan4": SKAN4, "skan3": SKAN3}
#: classifier featuresets in the LOCKED column order (PRIMARY_COLS order: FD,
#: tortuosity, density, length).  Logistic regression is invariant to column
#: order, but HistGradientBoosting breaks split ties by feature index, so the
#: locked GBDT numbers are only reproduced in this order.
CLF_PANELS = {"skan4": [c for c in PRIMARY_COLS if c.endswith("_skan")],
              "skan3": [c for c in PRIMARY_COLS if c.endswith("_skan")
                        and not c.startswith("tortuosity")]}
TR: List[dict] = []          # text-range rows


def tr(item: str, skan4, skan3, source: str, paper: str = "") -> None:
    TR.append(dict(item=item, paper_value=paper, skan4=skan4, skan3=skan3,
                   source=source))


def rng_str(v: Sequence[float], d: int = 3) -> str:
    v = [x for x in v if np.isfinite(x)]
    return ("%.*f-%.*f" % (d, min(v), d, max(v))) if v else "--"


# =============================================================== 1. the audit
def audit_means() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(PIVOT_DIR, "e1_decomposition.csv"))
    wide = d.pivot_table(index=["dataset", "biomarker"], columns="quantity",
                         values="value")
    t = pd.read_csv(os.path.join(PIVOT_DIR, "e1_topology_image.csv"))
    t = t[(t.arm == "rigr_uniform") & (t.seed == "pooled")] \
        .set_index(["dataset", "biomarker"])
    rows = []
    for pname, cols in PANELS.items():
        per = []
        for ds in DATASETS:
            sub = wide.loc[ds].reindex(cols)
            ts = t.loc[ds].reindex(cols)
            rec = dict(panel=pname, dataset=ds,
                       mean_abs_offset=sub["bias_const"].abs().mean(),
                       mean_resid_sd=sub["resid_sd"].mean(),
                       mean_r=sub["r_pearson"].mean(),
                       mean_rho=sub["r_spearman"].mean(),
                       mean_topo_abs=ts["mean_abs_dB"].mean(),
                       mean_topo_rms=ts["rms_dB"].mean(),
                       mean_observer=(sub["observer"].mean()
                                      if sub["observer"].notna().any()
                                      else np.nan),
                       mean_view=sub["view_spread"].mean())
            per.append(rec)
        rows += per
        g = pd.DataFrame(per)
        rows.append(dict(panel=pname, dataset="grand_mean_4_cohorts",
                         **{k: g[k].mean() for k in g.columns
                            if k not in ("panel", "dataset")}))
    out = pd.DataFrame(rows)
    out["source"] = ("results/pivot/e1_decomposition.csv; "
                     "results/pivot/e1_topology_image.csv (arm=rigr_uniform, "
                     "seed=pooled); same arithmetic as "
                     "paper2/tools/mk_tab_audit.py::_summary_row")
    # text ranges
    for pname, cols in PANELS.items():
        pass
    tt = t.reset_index()
    for q, lab in (("mean_abs_dB", "topology repair mean |dB| range over cells"),
                   ("rms_dB", "topology repair RMS dB range over cells")):
        tr(lab, rng_str(tt[tt.biomarker.isin(SKAN4)][q]),
           rng_str(tt[tt.biomarker.isin(SKAN3)][q]),
           "e1_topology_image.csv rigr_uniform pooled",
           "0.006-0.115" if q == "mean_abs_dB" else "0.006-0.351")
    a4 = out[(out.panel == "skan4") & out.dataset.isin(DATASETS)]
    a3 = out[(out.panel == "skan3") & out.dataset.isin(DATASETS)]
    tr("view spread panel-mean range", rng_str(a4.mean_view),
       rng_str(a3.mean_view), "e1_decomposition.csv", "0.268-0.523 (sic)")
    return out


# ============================================= 2. calibration text summaries
def calib_summaries() -> None:
    mt = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "r2_main_table.csv"))
    for pname, cols in PANELS.items():
        pass
    def f(cols, ds, q):
        return mt[(mt.dataset == ds) & mt.biomarker.isin(cols)][q]
    tr("HRF Deming slope range", rng_str(f(SKAN4, "hrf", "beta"), 2),
       rng_str(f(SKAN3, "hrf", "beta"), 2), "r2_main_table.csv", "0.37-0.81")
    tr("HRF slopes below 1 (count)",
       "%d/4" % (f(SKAN4, "hrf", "beta") < 1).sum(),
       "%d/3" % (f(SKAN3, "hrf", "beta") < 1).sum(), "r2_main_table.csv",
       "all four")
    tr("HRF CCC range", rng_str(f(SKAN4, "hrf", "ccc"), 2),
       rng_str(f(SKAN3, "hrf", "ccc"), 2), "r2_main_table.csv", "0.12-0.57")
    tr("HRF slope intervals excluding 1",
       ";".join(b.split("_")[0] for b in SKAN4
                if (lambda r: r.beta_lo > 1 or r.beta_hi < 1)(
                    mt[(mt.dataset == "hrf") & (mt.biomarker == b)].iloc[0])),
       ";".join(b.split("_")[0] for b in SKAN3
                if (lambda r: r.beta_lo > 1 or r.beta_hi < 1)(
                    mt[(mt.dataset == "hrf") & (mt.biomarker == b)].iloc[0])),
       "r2_main_table.csv", "length, tortuosity")
    p4 = mt[mt.biomarker.isin(SKAN4)]
    p3 = mt[mt.biomarker.isin(SKAN3)]
    tr("proportional bias p<0.05 (cells)",
       "%d/%d" % ((p4.prop_bias_p < 0.05).sum(), len(p4)),
       "%d/%d" % ((p3.prop_bias_p < 0.05).sum(), len(p3)),
       "r2_main_table.csv", "6 of 16")
    tr("slope intervals excluding 1 (cells, all cohorts)",
       "%d/%d" % (((p4.beta_lo > 1) | (p4.beta_hi < 1)).sum(), len(p4)),
       "%d/%d" % (((p3.beta_lo > 1) | (p3.beta_hi < 1)).sum(), len(p3)),
       "r2_main_table.csv", "")


# ================================================================ 3. families
def family_means(audit: pd.DataFrame) -> pd.DataFrame:
    a = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "family2_audit.csv"))
    a = a[(a.analysis_set == "main") & (a.panel == "primary")]
    rows = []
    for ds in ("hrf", "fives"):
        for pname, cols in PANELS.items():
            f1 = audit[(audit.panel == pname) & (audit.dataset == ds)].iloc[0]
            rows.append(dict(dataset=ds, family="unet (family 1)", seed=0,
                             panel=pname, mean_abs_mu=f1.mean_abs_offset,
                             mean_resid_sd=f1.mean_resid_sd,
                             mean_r=f1.mean_r))
            for (fam, sd), q in a[a.dataset == ds].groupby(["family", "seed"]):
                q = q[q.biomarker.isin(cols)]
                rows.append(dict(dataset=ds, family=fam, seed=sd, panel=pname,
                                 mean_abs_mu=q.mu_sigma_own.abs().mean(),
                                 mean_resid_sd=q.resid_sd.mean(),
                                 mean_r=q.r_pearson.mean()))
    out = pd.DataFrame(rows)
    out["source"] = ("r2/family2_audit.csv (analysis_set=main, primary) and "
                     "e1_decomposition.csv (family 1); arithmetic of "
                     "paper2/tools/mk_tab_family.py")
    w = out.pivot_table(index=["dataset", "family", "seed"], columns="panel",
                        values=["mean_abs_mu", "mean_resid_sd"])
    for idx, r in w.iterrows():
        pass
    # does the ordering claim survive?  HRF > FIVES offset, resid SD; FIVES > HRF r
    for pname in PANELS:
        o = out[out.panel == pname]
        ok_mu = ok_sd = ok_r = 0
        tot = 0
        for (fam, sd), g in o.groupby(["family", "seed"]):
            if set(g.dataset) != {"hrf", "fives"}:
                continue
            h, f = g[g.dataset == "hrf"].iloc[0], g[g.dataset == "fives"].iloc[0]
            tot += 1
            ok_mu += h.mean_abs_mu > f.mean_abs_mu
            ok_sd += h.mean_resid_sd > f.mean_resid_sd
            ok_r += f.mean_r > h.mean_r
        TR.append(dict(item="family orderings replicated (offset HRF>FIVES / "
                            "resid SD HRF>FIVES / r FIVES>HRF), panel " + pname,
                       paper_value="4/4, 3/4 (W-Net fails), 4/4",
                       skan4=("%d/%d, %d/%d, %d/%d" % (ok_mu, tot, ok_sd, tot,
                                                       ok_r, tot))
                       if pname == "skan4" else "",
                       skan3=("%d/%d, %d/%d, %d/%d" % (ok_mu, tot, ok_sd, tot,
                                                       ok_r, tot))
                       if pname == "skan3" else "",
                       source="r3_panel3_family.csv"))
    return out


# ============================================================ 4. topomatch
def topomatch(audit: pd.DataFrame) -> pd.DataFrame:
    tm = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "r2_topology_matched.csv"))
    ts = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "r2_topology_strata.csv"))
    rows = []
    for pname, cols in PANELS.items():
        g = audit[(audit.panel == pname) &
                  (audit.dataset == "grand_mean_4_cohorts")].iloc[0]
        for design, q in tm[tm.biomarker.isin(cols)].groupby("design",
                                                              sort=False):
            mx = float(q.h_net.abs().max())
            rows.append(dict(panel=pname, design=design,
                             retained=float(q.match_rate.mean()),
                             mean_h_net=float(q.h_net.mean()), max_h_net=mx,
                             max_cell=q.loc[q.h_net.abs().idxmax(),
                                            ["dataset", "biomarker"]]
                             .str.cat(sep=":"),
                             grand_mean_abs_offset=g.mean_abs_offset,
                             grand_mean_resid_sd=g.mean_resid_sd,
                             ratio_offset=g.mean_abs_offset / mx,
                             ratio_resid_sd=g.mean_resid_sd / mx))
        q = ts[ts.biomarker.isin(cols)]
        i = q.h_net.abs().idxmax()
        mx = float(abs(q.loc[i, "h_net"]))
        rows.append(dict(panel=pname, design="worst_stratum", retained=np.nan,
                         mean_h_net=np.nan, max_h_net=mx,
                         max_cell="%s:%s:%s=%s" % (q.loc[i, "dataset"],
                                                   q.loc[i, "biomarker"],
                                                   q.loc[i, "stratum_by"],
                                                   q.loc[i, "stratum"]),
                         grand_mean_abs_offset=g.mean_abs_offset,
                         grand_mean_resid_sd=g.mean_resid_sd,
                         ratio_offset=g.mean_abs_offset / mx,
                         ratio_resid_sd=g.mean_resid_sd / mx))
    out = pd.DataFrame(rows)
    out["source"] = ("r2/r2_topology_matched.csv, r2/r2_topology_strata.csv "
                     "(panel columns only), r3_panel3_audit.csv grand means")
    for pname in PANELS:
        o = out[out.panel == pname]
    o4, o3 = out[out.panel == "skan4"], out[out.panel == "skan3"]
    tr("topomatch denominators: grand mean |offset| / resid SD",
       "%.3f / %.3f" % (o4.grand_mean_abs_offset.iloc[0],
                        o4.grand_mean_resid_sd.iloc[0]),
       "%.3f / %.3f" % (o3.grand_mean_abs_offset.iloc[0],
                        o3.grand_mean_resid_sd.iloc[0]),
       "r3_panel3_audit.csv", "0.951 / 1.139")
    tr("topomatch mean h_net range over designs",
       rng_str(o4.mean_h_net, 4), rng_str(o3.mean_h_net, 4),
       "r2_topology_matched.csv", "0.0046-0.0068")
    tr("topomatch max h_net range over the six designs",
       rng_str(o4[o4.design != "worst_stratum"].max_h_net, 4),
       rng_str(o3[o3.design != "worst_stratum"].max_h_net, 4),
       "r2_topology_matched.csv", "0.020-0.048")
    tr("topology margin vs mean |offset| (designs; worst stratum)",
       "%s; %.1f" % (rng_str(o4[o4.design != "worst_stratum"].ratio_offset, 1),
                     o4[o4.design == "worst_stratum"].ratio_offset.iloc[0]),
       "%s; %.1f" % (rng_str(o3[o3.design != "worst_stratum"].ratio_offset, 1),
                     o3[o3.design == "worst_stratum"].ratio_offset.iloc[0]),
       "r3_panel3_topomatch.csv", "20-48; 17")
    return out


# ========================================================= 5. scale variants
def scale_variants() -> pd.DataFrame:
    s = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "r2_scale_sensitivity.csv"))
    tmt = pd.read_csv(os.path.join(PIVOT_DIR, "e1_topology_matched.csv"))
    rows = []
    for pname, cols in PANELS.items():
        tq = tmt[(tmt.scope == "sever") & tmt.biomarker.isin(cols)]
        topo_max = float(tq.h_net.abs().max())
        for (co, est, rep), g in s[s.biomarker.isin(cols)].groupby(
                ["cohort", "estimator", "representation"]):
            per = g.groupby("dataset").agg(mu=("abs_offset", "mean"),
                                           sd=("resid_sd", "mean"))
            big = float(max(per.mu.max(), per.sd.max()))
            rows.append(dict(panel=pname, cohort=co, estimator=est,
                             representation=rep,
                             hrf_mu=per.loc["hrf", "mu"],
                             fives_mu=per.loc["fives", "mu"],
                             drive_mu=per.loc["drive", "mu"],
                             hrf_over_fives_mu=per.loc["hrf", "mu"] /
                             per.loc["fives", "mu"],
                             hrf_over_fives_sd=per.loc["hrf", "sd"] /
                             per.loc["fives", "sd"],
                             drive_gt_hrf_mu=bool(per.loc["drive", "mu"] >
                                                  per.loc["hrf", "mu"]),
                             axis_separation=big / topo_max,
                             topo_max=topo_max))
    out = pd.DataFrame(rows)
    out["source"] = ("r2/r2_scale_sensitivity.csv (13 variants) and "
                     "e1_topology_matched.csv (scope=sever) for the topology "
                     "maximum; arithmetic of r2_scale.py::claims")
    o4, o3 = out[out.panel == "skan4"], out[out.panel == "skan3"]
    tr("13 scale variants: n variants", len(o4), len(o3), "", "13")
    tr("13 scale variants: HRF/FIVES mean |offset| ratio range",
       rng_str(o4.hrf_over_fives_mu, 1), rng_str(o3.hrf_over_fives_mu, 1),
       "r3_panel3_scale.csv", "3.9-6.2")
    tr("13 scale variants: HRF/FIVES mean resid SD ratio range",
       rng_str(o4.hrf_over_fives_sd, 1), rng_str(o3.hrf_over_fives_sd, 1),
       "r3_panel3_scale.csv", "1.4-2.0")
    tr("13 scale variants: axis separation range",
       rng_str(o4.axis_separation, 0), rng_str(o3.axis_separation, 0),
       "r3_panel3_scale.csv", "56-102")
    c4 = o4[o4.cohort == "common"].iloc[0]
    c3 = o3[o3.cohort == "common"].iloc[0]
    tr("common scale: DRIVE vs HRF mean |offset|",
       "%.2f vs %.2f" % (c4.drive_mu, c4.hrf_mu),
       "%.2f vs %.2f" % (c3.drive_mu, c3.hrf_mu), "r3_panel3_scale.csv",
       "1.36 vs 1.26")
    return out


# ================================================= 6. topology stress rows
def topology_stress() -> None:
    tmt = pd.read_csv(os.path.join(PIVOT_DIR, "e1_topology_matched.csv"))
    tmt = tmt[tmt.scope == "sever"]
    for pname, cols in PANELS.items():
        pass
    q4, q3 = tmt[tmt.biomarker.isin(SKAN4)], tmt[tmt.biomarker.isin(SKAN3)]
    tr("per-event |dB_topo| range", rng_str(q4.h_topo), rng_str(q3.h_topo),
       "e1_topology_matched.csv sever", "0.002-0.039")
    tr("per-event paired H_net range", rng_str(q4.h_net), rng_str(q3.h_net),
       "e1_topology_matched.csv sever", "0.000-0.020")
    tr("per-event H_net CI upper end max", "%.3f" % q4.h_net_hi.max(),
       "%.3f" % q3.h_net_hi.max(), "e1_topology_matched.csv sever", "0.037")
    tr("largest per-event H_net cell",
       q4.loc[q4.h_net.idxmax(), "dataset"] + ":" +
       q4.loc[q4.h_net.idxmax(), "biomarker"],
       q3.loc[q3.h_net.idxmax(), "dataset"] + ":" +
       q3.loc[q3.h_net.idxmax(), "biomarker"], "", "CHASE_DB1 tortuosity")
    tr("additive bracket H_net x events_per_image (NOT the paper bracket, which uses n_should from runs/rigr; not reproduced -- do not quote)", rng_str(q4.h_net *
                                                        q4.events_per_image),
       rng_str(q3.h_net * q3.events_per_image),
       "e1_topology_matched.csv sever", "0.025-0.692")
    # macro |e| before/after repair from the per-image files
    sig = sigma_table()
    res = {p: [] for p in PANELS}
    for ds in DATASETS:
        frames = []
        for sd in (0, 1, 2):
            p = os.path.join(RUNS_DIR, "rigr", "uniform", ds, "seed%d" % sd,
                             "per_image.csv")
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
        if not frames:
            continue
        a = pd.concat(frames, ignore_index=True)
        for pname, cols in PANELS.items():
            eb, ea = [], []
            for c in cols:
                s = sig[ds][c]
                g = a["bio_gt_" + c].to_numpy(float)
                b = a["bio_before_" + c].to_numpy(float)
                af = a["bio_after_" + c].to_numpy(float)
                eb.append(np.abs(b - g) / s)
                ea.append(np.abs(af - g) / s)
            eb = np.nanmean(np.vstack(eb), axis=0)
            ea = np.nanmean(np.vstack(ea), axis=0)
            ok = np.isfinite(eb) & np.isfinite(ea)
            res[pname].append((ds, float(np.mean(ea[ok] - eb[ok])),
                               float(np.mean(eb[ok]))))
    img = pd.read_csv(os.path.join(PIVOT_DIR, "e1_topology_image.csv"))
    ref = img[(img.arm == "rigr_uniform") & (img.seed == "pooled") &
              (img.biomarker == "macro_primary4")].set_index("dataset")
    for ds, d4, b4 in res["skan4"]:
        assert abs(d4 - ref.loc[ds, "d_abs_err"]) < 1e-9, (ds, d4)
    tr("repair change in macro |e| (range over cohorts)",
       rng_str([r[1] for r in res["skan4"]]),
       rng_str([r[1] for r in res["skan3"]]),
       "runs/rigr/uniform/<ds>/seed{0,1,2}/per_image.csv (skan4 reproduces "
       "e1_topology_image.csv macro_primary4)", "-0.021 to -0.008")
    tr("baseline macro |e| (range over cohorts)",
       rng_str([r[2] for r in res["skan4"]], 2),
       rng_str([r[2] for r in res["skan3"]], 2), "same", "0.64-1.72")


# ====================================================== 7. case figure macro
def case_macro() -> None:
    gt = pd.read_csv(os.path.join(RESULTS_DIR, "gateA_biomarkers_gt.csv"))
    sig = pd.read_csv(os.path.join(RESULTS_DIR,
                                   "gateA_biomarker_scales_train.csv"))
    key = {"fives": "FIVES", "hrf": "HRF"}
    out = {}
    for ds in ("fives", "hrf"):
        base = os.path.join(RUNS_DIR, "pivot", ds, "seed0", "baseline", "pred")
        pred = pd.read_csv(os.path.join(base, "bio.csv"))
        px = pd.read_csv(os.path.join(base, "pixel_metrics.csv"))
        g = gt[(gt.dataset == key[ds]) & (gt.observer == "obs1") &
               (gt.split == "test")]
        px = px.rename(columns={"image": "image_id"}).copy()
        px["_key"] = px.image_id.astype(str).str.replace(
            r"^(train|test|val)_", "", regex=True)
        m = pred.merge(g, on="image_id", suffixes=("_p", "_g"))
        m["_key"] = m.image_id.astype(str).str.replace(
            r"^(train|test|val)_", "", regex=True)
        m = m.merge(px[["_key", "dice"]], on="_key")
        q = sig[sig.dataset == key[ds]]
        for pname, cols in PANELS.items():
            err = np.zeros(len(m))
            for c in cols:
                s = float(q[q.biomarker == c]["sigma"].iloc[0])
                err += np.abs(m[c + "_p"] - m[c + "_g"]) / s
            mm = m.assign(macro_abs=err / len(cols)).dropna(
                subset=["macro_abs", "dice"])
            r_d = (mm.dice - mm.dice.median()).abs().rank(method="first")
            r_e = (mm.macro_abs - mm.macro_abs.median()).abs().rank(
                method="first")
            row = mm.loc[(r_d + r_e).idxmin()]
            out[(ds, pname)] = (row.image_id, float(row.dice),
                                float(row.macro_abs))
    for pname in PANELS:
        pass
    f4, h4 = out[("fives", "skan4")], out[("hrf", "skan4")]
    f3, h3 = out[("fives", "skan3")], out[("hrf", "skan3")]
    tr("case figure: selected images (FIVES; HRF)",
       "%s; %s" % (f4[0], h4[0]), "%s; %s" % (f3[0], h3[0]),
       "paper2/tools/fig_cases.py rule re-run with the panel swapped", "")
    tr("case figure: Dice (FIVES vs HRF)", "%.3f vs %.3f" % (f4[1], h4[1]),
       "%.3f vs %.3f" % (f3[1], h3[1]), "", "0.927 vs 0.803")
    tr("case figure: macro |err| sigma (FIVES vs HRF; ratio)",
       "%.2f vs %.2f; %.1f" % (f4[2], h4[2], h4[2] / f4[2]),
       "%.2f vs %.2f; %.1f" % (f3[2], h3[2], h3[2] / f3[2]), "",
       "0.30 vs 1.65; 5.5")


# ============================================================= 8. the gap
def cv_proba(X, y, kind, classes):
    from src.pivot.r2_fives_full import cv_proba as _cv
    return _cv(X, y, kind, classes)


def macro_auc(y, p, classes):
    from src.pivot.r2_fives_full import macro_auc as _m
    return _m(y, p, classes)


def gap_pair(d: pd.DataFrame, label: dict) -> List[dict]:
    y = d["disease"].astype(str).to_numpy()
    classes = sorted(set(y))
    if len(classes) < 2 or len(y) < 30:
        return []
    rows = []
    for kind in ("logreg", "gbdt"):
        P = {}
        for pname, cols in CLF_PANELS.items():
            Xg = np.nan_to_num(d[[c + "__gt" for c in cols]].to_numpy(float),
                               nan=0.0, posinf=0.0, neginf=0.0)
            Xp = np.nan_to_num(d[[c + "__pred" for c in cols]].to_numpy(float),
                               nan=0.0, posinf=0.0, neginf=0.0)
            P[pname] = (cv_proba(Xg, y, kind, classes),
                        cv_proba(Xp, y, kind, classes))
        pt = {p: (macro_auc(y, P[p][0], classes), macro_auc(y, P[p][1], classes))
              for p in PANELS}
        rng = np.random.RandomState(SEED)
        n = len(y)
        B = {k: [] for k in ("g4", "p4", "g3", "p3")}
        for _ in range(N_BOOT_AUC):
            i = rng.randint(0, n, n)
            if len(set(y[i])) < len(classes):
                continue
            v = dict(g4=macro_auc(y[i], P["skan4"][0][i], classes),
                     p4=macro_auc(y[i], P["skan4"][1][i], classes),
                     g3=macro_auc(y[i], P["skan3"][0][i], classes),
                     p3=macro_auc(y[i], P["skan3"][1][i], classes))
            if all(np.isfinite(x) for x in v.values()):
                for k in B:
                    B[k].append(v[k])
        B = {k: np.asarray(v) for k, v in B.items()}
        gap4 = B["g4"] - B["p4"]
        gap3 = B["g3"] - B["p3"]

        def pc(v):
            return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        g4, g3, dd = pc(gap4), pc(gap3), pc(gap3 - gap4)
        rows.append(dict(
            **label, clf=kind, n=n, n_classes=len(classes),
            auc_ref_skan4=pt["skan4"][0], auc_pred_skan4=pt["skan4"][1],
            gap_skan4=pt["skan4"][0] - pt["skan4"][1], gap_skan4_lo=g4[0],
            gap_skan4_hi=g4[1],
            auc_ref_skan3=pt["skan3"][0], auc_pred_skan3=pt["skan3"][1],
            gap_skan3=pt["skan3"][0] - pt["skan3"][1], gap_skan3_lo=g3[0],
            gap_skan3_hi=g3[1],
            d_auc_ref=pt["skan3"][0] - pt["skan4"][0],
            d_auc_pred=pt["skan3"][1] - pt["skan4"][1],
            gap_diff_3_minus_4=(pt["skan3"][0] - pt["skan3"][1]) -
            (pt["skan4"][0] - pt["skan4"][1]),
            gap_diff_lo=dd[0], gap_diff_hi=dd[1],
            gap_diff_excl0=bool(dd[0] > 0 or dd[1] < 0),
            gap_skan4_excl0=bool(g4[0] > 0 or g4[1] < 0),
            gap_skan3_excl0=bool(g3[0] > 0 or g3[1] < 0),
            n_boot_ok=int(gap4.size)))
        print("[gap] %s %s %s: skan4 %+.4f  skan3 %+.4f  diff %+.4f [%+.4f,"
              " %+.4f]" % (label.get("cohort"), label.get("block"), kind,
                           rows[-1]["gap_skan4"], rows[-1]["gap_skan3"],
                           rows[-1]["gap_diff_3_minus_4"], dd[0], dd[1]),
              flush=True)
    return rows


def gaps() -> pd.DataFrame:
    from src.pivot import r2_fives_full as ff
    from src.pivot import r2_family2 as fam2
    rows: List[dict] = []
    src_proto = ("5-fold x 3 repeats stratified CV (seeds 0-2), macro one-vs-"
                 "rest AUC; 1000-draw paired image bootstrap, SEED 0 -- the "
                 "locked protocol; the same resampled images enter both panels")
    p2 = pd.read_csv(os.path.join(PIVOT_DIR, "p2_predictions.csv"),
                     low_memory=False)
    for ds in ("hrf", "fives"):
        d = p2[(p2.dataset == ds) & p2.disease.notna()].reset_index(drop=True)
        rows += gap_pair(d, dict(cohort=ds, block=("all45" if ds == "hrf"
                                                   else "p2_400"),
                                 source="results/pivot/p2_predictions.csv; "
                                        + src_proto))
    full = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "bio_master_full.csv"),
                       low_memory=False)
    d_test = ff.pairs(full, "test", "pred_test")
    d_train = ff.pairs(full, "train", "pred_oof")
    d_all = pd.concat([d_test, d_train], ignore_index=True)
    for blk, d in (("test200", d_test), ("train600", d_train),
                   ("all800", d_all)):
        rows += gap_pair(d, dict(cohort="fives", block=blk,
                                 source="results/pivot/r2/bio_master_full.csv; "
                                        + src_proto))
    # second-segmenter families
    ref = fam2.load_reference(os.path.join(PIVOT_DIR, "r2",
                                           "bio_master_full.csv"))
    for rec in fam2.discover(os.path.join(RUNS_DIR, "pivot", "r2")):
        if rec["dataset"] not in ("hrf", "fives"):
            continue
        pred = fam2.flag_masks(pd.read_csv(rec["path"], low_memory=False))
        d0 = fam2.pairs(ref, pred, rec["dataset"])
        sets = ("all_incl_empty", "main", "sens_no_degen")
        seen = {}
        for aset in sets:
            d = fam2.subset(d0, aset)
            if len(d) in seen:
                for r in seen[len(d)]:
                    rr = dict(r)
                    rr["block"] = aset
                    rr["source"] += "; identical rows to %s" % r["block"]
                    rows.append(rr)
                continue
            got = gap_pair(d, dict(cohort=rec["dataset"], block=aset,
                                   family=rec["family"], seed=rec["seed"],
                                   source=rec["path"].replace("\\", "/")
                                   + "; " + src_proto))
            seen[len(d)] = got
            rows += got
    out = pd.DataFrame(rows)
    if "family" in out:
        out["family"] = out["family"].fillna("unet (family 1)")
        out["seed"] = out["seed"].fillna(0).astype(int)
    return out


# =================================================== 9. FIVES subsample
def fives_subsample() -> pd.DataFrame:
    from src.pivot import r2_fives_full as ff
    full = pd.read_csv(os.path.join(PIVOT_DIR, "r2", "bio_master_full.csv"),
                       low_memory=False)
    sig = sigma_table()["fives"]
    d_train = ff.pairs(full, "train", "pred_oof")
    rows = []
    for sd in list(range(5)) + [-1]:
        sub = (ff.stratified_subsample(d_train, 50, 1000 + sd) if sd >= 0
               else d_train)
        a = ff.audit_block(sub, sig, "x", "x")
        for pname, cols in PANELS.items():
            q = [r for r in a if r["biomarker"] in cols]
            dn = ff.downstream_block(sub, "x", "x", pname, CLF_PANELS[pname])
            for r in dn:
                rows.append(dict(subsample_seed=sd, n=len(sub), panel=pname,
                                 clf=r["clf"],
                                 mean_abs_offset=float(np.mean(
                                     [abs(x["offset"]) for x in q])),
                                 mean_resid_sd=float(np.mean(
                                     [x["resid_sd"] for x in q])),
                                 mean_r=float(np.mean([x["r_pearson"]
                                                       for x in q])),
                                 delta_auc=r["delta_auc"],
                                 delta_auc_lo=r["delta_auc_lo"],
                                 delta_auc_hi=r["delta_auc_hi"]))
    out = pd.DataFrame(rows)
    out["source"] = ("src/pivot/r2_fives_full.py stratified_subsample(50/class,"
                     " seeds 1000-1004) / audit_block / downstream_block; "
                     "subsample_seed=-1 is all 600 training images")
    for pname in PANELS:
        o = out[(out.panel == pname) & (out.clf == "logreg")]
        s, f = o[o.subsample_seed >= 0], o[o.subsample_seed < 0].iloc[0]
        TR.append(dict(item="FIVES 200-subsamples: mean |offset| range (full "
                            "600), panel " + pname,
                       paper_value="0.186-0.223 (0.214)",
                       skan4="%s (%.3f)" % (rng_str(s.mean_abs_offset),
                                            f.mean_abs_offset)
                       if pname == "skan4" else "",
                       skan3="%s (%.3f)" % (rng_str(s.mean_abs_offset),
                                            f.mean_abs_offset)
                       if pname == "skan3" else "",
                       source="r3_panel3_fives_subsample.csv"))
        TR.append(dict(item="FIVES 200-subsamples: mean r range (full 600), "
                            "panel " + pname,
                       paper_value="0.848-0.878 (0.861)",
                       skan4="%s (%.3f)" % (rng_str(s.mean_r), f.mean_r)
                       if pname == "skan4" else "",
                       skan3="%s (%.3f)" % (rng_str(s.mean_r), f.mean_r)
                       if pname == "skan3" else "",
                       source="r3_panel3_fives_subsample.csv"))
        TR.append(dict(item="FIVES 200-subsamples: logreg reference gap "
                            "range, panel " + pname,
                       paper_value="-0.019 to +0.031",
                       skan4=rng_str(s.delta_auc) if pname == "skan4" else "",
                       skan3=rng_str(s.delta_auc) if pname == "skan3" else "",
                       source="r3_panel3_fives_subsample.csv"))
    return out


# ============================================================== main
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=os.path.join(PIVOT_DIR, "r3"))
    ap.add_argument("--skip_gap", action="store_true")
    ap.add_argument("--only_gap", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    w = lambda df, n: (df.to_csv(os.path.join(args.out_dir, n), index=False),
                       print("wrote", n, df.shape))
    if args.only_gap:
        g = gaps()
        w(g, "r3_panel3_gap.csv")
        return 0

    audit = audit_means()
    w(audit, "r3_panel3_audit.csv")
    calib_summaries()
    fam = family_means(audit)
    w(fam, "r3_panel3_family.csv")
    tm = topomatch(audit)
    w(tm, "r3_panel3_topomatch.csv")
    sc = scale_variants()
    w(sc, "r3_panel3_scale.csv")
    topology_stress()
    case_macro()
    fs = fives_subsample()
    w(fs, "r3_panel3_fives_subsample.csv")
    if not args.skip_gap:
        g = gaps()
        w(g, "r3_panel3_gap.csv")
    t = pd.DataFrame(TR)
    t["status"] = ("skan3 = added post-hoc reporting hierarchy (sensitivity); "
                   "skan4 = locked")
    w(t, "r3_panel3_textranges.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30,
                           "display.max_rows", 300, "display.max_colwidth", 70):
        print(audit.drop(columns="source").round(3).to_string(index=False))
        print(fam.drop(columns="source").round(3).to_string(index=False))
        print(tm.drop(columns="source").round(4).to_string(index=False))
        print(t.drop(columns=["source", "status"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
