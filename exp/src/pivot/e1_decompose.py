"""E1 -- three-component measurement-error decomposition (+ two reference scales).

Everything is expressed on one axis: the pre-registered **training-split native
robust scale** sigma from ``results/gateA_biomarker_scales_train.csv`` (the same
single source of truth P1-P5 used).

Five quantities per (dataset, biomarker), all in sigma:

    bias_const    mean_i (B_pred,i - B_GT,i) / sigma           -- constant bias
    resid_sd      SD_i[(B_pred,i - B_GT,i)/sigma]              -- per-image scatter
                  (i.e. the residual after removing the constant bias)
                  reported alongside r_pearson / r_spearman
    topology      mean(H_net | repairable event) x mean(n_should per image)
                  -- the "expected topology harm per predicted image"
    observer      mean_i |B_obs1,i - B_obs2,i| / sigma         -- inter-observer
    view_spread   mean_i SD_views(B_i) / sigma                 -- acquisition

Provenance for every number is written into the CSV (``source`` column).

CLI
---
    python -m src.pivot.e1_decompose
"""
from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from src.pivot.common import (EXP_ROOT, PIVOT_DIR, PRIMARY_COLS, RESULTS_DIR,
                              RUNS_DIR, GATEA_DS, sigma_table)

SEED = 0
N_BOOT = 2000
DATASETS = ("drive", "chasedb1", "hrf", "fives")
FIGS_DIR = os.path.join(EXP_ROOT, "figs")


# ---------------------------------------------------------------- helpers
def boot_ci(x: np.ndarray, fn, n_boot: int = N_BOOT, seed: int = SEED):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    vals = [fn(x[rng.randint(0, x.size, x.size)]) for _ in range(n_boot)]
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def boot_ci_pair(x: np.ndarray, y: np.ndarray, fn, n_boot: int = N_BOOT,
                 seed: int = SEED):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 4:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, x.size, x.size)
        v = fn(x[idx], y[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def pearson(a, b):
    from scipy.stats import pearsonr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(pearsonr(a[ok], b[ok])[0])


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(a[ok], b[ok])[0])


def add(rows: List[dict], **kw):
    rows.append(kw)


# ------------------------------------------------- (i)+(ii) bias / reliability
def bias_and_reliability(master: pd.DataFrame, sig: Dict[str, Dict[str, float]],
                         rows: List[dict]) -> pd.DataFrame:
    """Seed-0 constant bias, reliability and residual SD from ``bio_master.csv``."""
    keep = []
    for ds in DATASETS:
        m = master[master["dataset"] == ds]
        gt = m[m["source"] == "gt"]
        gt = gt[gt["split"] == "test"][["image_id"] + list(PRIMARY_COLS)]
        pr = m[m["source"] == "pred_test"][["image_id"] + list(PRIMARY_COLS)]
        d = gt.merge(pr, on="image_id", suffixes=("__gt", "__pred"))
        n = len(d)
        for c in PRIMARY_COLS:
            s = sig[ds][c]
            g = d[c + "__gt"].to_numpy(float)
            p = d[c + "__pred"].to_numpy(float)
            e = (p - g) / s
            ok = np.isfinite(e)
            bias = float(np.mean(e[ok]))
            sd = float(np.std(e[ok], ddof=1))
            lo, hi = boot_ci(e[ok], np.mean)
            sdlo, sdhi = boot_ci(e[ok], lambda v: np.std(v, ddof=1))
            rp, rs = pearson(p, g), spearman(p, g)
            rplo, rphi = boot_ci_pair(p, g, pearson)
            add(rows, dataset=ds, biomarker=c, quantity="bias_const",
                value=bias, lo=lo, hi=hi, n=int(ok.sum()), seed=0,
                source="results/pivot/bio_master.csv (source=gt[test] vs pred_test, seed0); "
                       "sigma=results/gateA_biomarker_scales_train.csv")
            add(rows, dataset=ds, biomarker=c, quantity="resid_sd",
                value=sd, lo=sdlo, hi=sdhi, n=int(ok.sum()), seed=0,
                source="SD over test images of (B_pred-B_GT)/sigma, "
                       "results/pivot/bio_master.csv seed0")
            add(rows, dataset=ds, biomarker=c, quantity="r_pearson",
                value=rp, lo=rplo, hi=rphi, n=int(ok.sum()), seed=0,
                source="pearson r(B_pred, B_GT), results/pivot/bio_master.csv seed0")
            add(rows, dataset=ds, biomarker=c, quantity="r_spearman",
                value=rs, lo=float("nan"), hi=float("nan"), n=int(ok.sum()), seed=0,
                source="spearman rho(B_pred, B_GT), results/pivot/bio_master.csv seed0")
            keep.append(dict(dataset=ds, biomarker=c, n=n, bias=bias, resid_sd=sd,
                             r_pearson=rp, r_spearman=rs))
    return pd.DataFrame(keep)


def bias_seeds(sig: Dict[str, Dict[str, float]], rows: List[dict]) -> pd.DataFrame:
    """Seeds 0/1/2 constant bias from the repair per-image tables.

    ``runs/repair/geometric/<ds>/seed<S>/per_image.csv`` carries
    ``bio_gt_<col>`` and ``bio_before_<col>`` for the *unrepaired* seed-S
    prediction -- the probe report verified these reproduce ``bio_master.csv``
    to within 5e-3 relative.  This is the only cheap route to seeds 1-2
    (recomputing the pipeline would cost GPU inference + hours of CPU).
    """
    out = []
    for ds in DATASETS:
        for seed in (0, 1, 2):
            p = os.path.join(RUNS_DIR, "repair", "geometric", ds, f"seed{seed}",
                             "per_image.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            for c in PRIMARY_COLS:
                gcol, pcol = f"bio_gt_{c}", f"bio_before_{c}"
                if gcol not in d.columns:
                    continue
                s = sig[ds][c]
                e = (d[pcol].to_numpy(float) - d[gcol].to_numpy(float)) / s
                ok = np.isfinite(e)
                if ok.sum() < 2:
                    continue
                lo, hi = boot_ci(e[ok], np.mean)
                add(rows, dataset=ds, biomarker=c, quantity="bias_const_seed",
                    value=float(np.mean(e[ok])), lo=lo, hi=hi, n=int(ok.sum()),
                    seed=seed,
                    source=f"runs/repair/geometric/{ds}/seed{seed}/per_image.csv "
                           "(bio_before_* vs bio_gt_*, no_repair state)")
                out.append(dict(dataset=ds, seed=seed, biomarker=c,
                                n=int(ok.sum()), bias=float(np.mean(e[ok]))))
    return pd.DataFrame(out)


# ------------------------------------------------------- (iii) topology harm
def topology_component(rows: List[dict]) -> pd.DataFrame:
    """mean net excess harm per repairable event x mean repairable events/image."""
    cols = ["dataset", "split", "observer", "type", "control_ok"] + \
           [f"Hnet_{c}" for c in PRIMARY_COLS]
    ev = pd.read_parquet(os.path.join(RESULTS_DIR, "c1_events_harm.parquet"),
                         columns=cols)
    ev = ev[(ev["observer"] == "obs1") & (ev["split"] == "test") &
            (ev["control_ok"] == 1.0)]
    # n_should = number of candidate gaps that *should* be repaired, per image
    nsh = {}
    for ds in DATASETS:
        p = os.path.join(RUNS_DIR, "rigr", "uniform", ds, "seed0", "per_image.csv")
        d = pd.read_csv(p)
        nsh[ds] = (float(d["n_should"].mean()), len(d))

    out = []
    for ds in DATASETS:
        g = ev[ev["dataset"] == GATEA_DS[ds]]
        for scope, sub in (("sever", g[g["type"] == "sever"]), ("all_types", g)):
            for c in PRIMARY_COLS:
                h = sub[f"Hnet_{c}"].to_numpy(float)
                h = h[np.isfinite(h)]
                if h.size == 0:
                    continue
                mh = float(np.mean(h))
                lo, hi = boot_ci(h, np.mean)
                k, nimg = nsh[ds]
                q = "topology" if scope == "sever" else "topology_all_types"
                add(rows, dataset=ds, biomarker=c, quantity=q,
                    value=mh * k, lo=lo * k, hi=hi * k, n=int(h.size), seed=0,
                    source=f"mean H_net over {h.size} {scope} events "
                           f"(results/c1_events_harm.parquet, obs1/test/control_ok) "
                           f"x mean n_should={k:.3f} over {nimg} images "
                           f"(runs/rigr/uniform/{ds}/seed0/per_image.csv)")
                out.append(dict(dataset=ds, biomarker=c, scope=scope,
                                mean_Hnet=mh, n_events=int(h.size),
                                mean_n_should=k, n_images=nimg, per_image=mh * k))

    # empirical counterpart: what *actually repairing* the repairable events
    # does to the constant bias (RiGR-uniform, the precision-first repairer).
    bd = pd.read_csv(os.path.join(RESULTS_DIR, "bias_diagnostics.csv"))
    for ds in DATASETS:
        g = bd[bd["dataset"] == ds]
        b0 = g[g["method"] == "no_repair"]
        b1 = g[g["method"] == "rigr_uniform"]
        if len(b0) == 0 or len(b1) == 0:
            continue
        for c in PRIMARY_COLS:
            col = f"bias_after_{c}"
            if col not in g.columns:
                continue
            d0 = float(b0[col].iloc[0])
            d1 = float(b1[col].iloc[0])
            add(rows, dataset=ds, biomarker=c, quantity="topology_repair_measured",
                value=abs(d1 - d0), lo=float("nan"), hi=float("nan"),
                n=int(b1["n_images"].iloc[0]), seed=-1,
                source="|bias_after(rigr_uniform) - bias(no_repair)| in sigma, "
                       "results/bias_diagnostics.csv (3 seeds pooled) -- the "
                       "MEASURED effect of repairing the repairable events")
    return pd.DataFrame(out)


# ------------------------------------------------------- (iv) inter-observer
def observer_component(sig: Dict[str, Dict[str, float]], rows: List[dict]
                       ) -> pd.DataFrame:
    p = os.path.join(RESULTS_DIR, "fig4c_observer.csv")
    d = pd.read_csv(p)
    out = []
    for ds, sub in d.groupby("dataset"):
        ds = str(ds)
        if ds not in sig:
            continue
        for c in PRIMARY_COLS:
            col = f"absdiff_{c}"
            if col not in sub.columns:
                continue
            s = sig[ds][c]
            v = sub[col].to_numpy(float) / s
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            lo, hi = boot_ci(v, np.mean)
            add(rows, dataset=ds, biomarker=c, quantity="observer",
                value=float(np.mean(v)), lo=lo, hi=hi, n=int(v.size), seed=0,
                source="mean |obs1-obs2| / sigma_train, results/fig4c_observer.csv "
                       f"(absdiff_{c}; note the file's own z_* columns use the "
                       "ALL-MASK gateA sigma, these use the train-split sigma)")
            out.append(dict(dataset=ds, biomarker=c, n=int(v.size),
                            observer=float(np.mean(v))))
    return pd.DataFrame(out)


# ------------------------------------------------------ (v) acquisition views
def view_component(sig: Dict[str, Dict[str, float]], rows: List[dict]
                   ) -> pd.DataFrame:
    out = []
    for f in sorted(glob.glob(os.path.join(RESULTS_DIR,
                                           "fig4a_stability_*_none_per_view.csv"))):
        d = pd.read_csv(f)
        ds = str(d["dataset"].iloc[0])
        if ds not in sig:
            continue
        for c in PRIMARY_COLS:
            if c not in d.columns:
                continue
            s = sig[ds][c]
            sd = d.groupby("image_id")[c].std(ddof=1).to_numpy(float) / s
            sd = sd[np.isfinite(sd)]
            if sd.size == 0:
                continue
            lo, hi = boot_ci(sd, np.mean)
            add(rows, dataset=ds, biomarker=c, quantity="view_spread",
                value=float(np.mean(sd)), lo=lo, hi=hi, n=int(sd.size), seed=0,
                source=f"mean over images of SD across 8 acquisition views / "
                       f"sigma_train, {os.path.relpath(f, EXP_ROOT)}")
            out.append(dict(dataset=ds, biomarker=c, n=int(sd.size),
                            view_spread=float(np.mean(sd))))
    return pd.DataFrame(out)


# ----------------------------------------------------------------- figure
def make_figure(dec: pd.DataFrame, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.eval.savefig_util import save_fig

    quantities = [("bias_const", "|constant bias|", "#3b6ea5"),
                  ("resid_sd", "per-image residual SD", "#c0504d"),
                  ("topology", "topology harm / image", "#9bbb59"),
                  ("observer", "inter-observer", "#8064a2"),
                  ("view_spread", "acquisition spread", "#e8a33d")]
    dsets = list(DATASETS)
    biomarkers = ["FD", "tortuosity", "density", "total_length"]
    pipes = ["pvbm", "skan"]

    fig, axes = plt.subplots(4, 2, figsize=(11.5, 13.0), sharex=True)
    w = 0.16
    x = np.arange(len(dsets))
    for i, b in enumerate(biomarkers):
        for j, pp in enumerate(pipes):
            ax = axes[i, j]
            col = f"{b}_{pp}"
            for k, (q, lab, colr) in enumerate(quantities):
                vals, los, his = [], [], []
                for ds in dsets:
                    r = dec[(dec["dataset"] == ds) & (dec["biomarker"] == col) &
                            (dec["quantity"] == q)]
                    if len(r) == 0:
                        vals.append(np.nan); los.append(np.nan); his.append(np.nan)
                        continue
                    v = float(r["value"].iloc[0])
                    lo = float(r["lo"].iloc[0]); hi = float(r["hi"].iloc[0])
                    if q == "bias_const":       # magnitude on a positive axis
                        if v < 0:
                            v, lo, hi = -v, -hi, -lo
                        else:
                            v, lo, hi = v, lo, hi
                        lo, hi = abs(lo), abs(hi)
                        lo, hi = min(lo, hi), max(lo, hi)
                    vals.append(v); los.append(v - lo); his.append(hi - v)
                vals = np.array(vals, float)
                err = np.vstack([np.abs(np.nan_to_num(los)),
                                 np.abs(np.nan_to_num(his))])
                ax.bar(x + (k - 2) * w, vals, width=w, color=colr,
                       label=lab if (i == 0 and j == 0) else None,
                       yerr=err, capsize=1.6, error_kw=dict(lw=0.7, alpha=0.7))
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 20)
            ax.axhline(1.0, color="0.4", lw=0.8, ls=":")
            ax.set_title(f"{b}  ({pp})", fontsize=10)
            ax.grid(axis="y", alpha=0.25, lw=0.5)
            if j == 0:
                ax.set_ylabel("magnitude  [sigma]", fontsize=9)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in dsets], fontsize=9)
    fig.legend(loc="upper center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 0.995))
    fig.suptitle("E1  measurement-error decomposition on one sigma axis "
                 "(dotted line = 1 sigma)", y=0.965, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    save_fig(fig, out_png, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sig = sigma_table()
    master = pd.read_csv(os.path.join(PIVOT_DIR, "bio_master.csv"), low_memory=False)

    rows: List[dict] = []
    br = bias_and_reliability(master, sig, rows)
    sd = bias_seeds(sig, rows)
    tp = topology_component(rows)
    ob = observer_component(sig, rows)
    vw = view_component(sig, rows)

    dec = pd.DataFrame(rows)
    dec = dec[["dataset", "biomarker", "quantity", "value", "lo", "hi",
               "n", "seed", "source"]]
    out = os.path.join(args.out_dir, "e1_decomposition.csv")
    dec.to_csv(out, index=False)
    print("wrote", out, dec.shape)

    sd.to_csv(os.path.join(args.out_dir, "e1_bias_seeds.csv"), index=False)
    tp.to_csv(os.path.join(args.out_dir, "e1_topology_detail.csv"), index=False)
    print("wrote e1_bias_seeds.csv / e1_topology_detail.csv")

    make_figure(dec, os.path.join(FIGS_DIR, "pivot_fig_decomposition.png"))


if __name__ == "__main__":
    main()
