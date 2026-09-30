"""E1 -- the topology axis as an *image-level* estimand, on the frozen sigma axis.

Why this script exists
----------------------
The first version of the audit reported the topology axis as the change in the
**dataset mean offset** produced by repairing the breaks a repairer finds
(``topology_repair_measured`` in ``e1_decomposition.csv``).  A reviewer of the
manuscript pointed out, correctly, that this is not the same statistical
functional as the other two axes: a small change in a dataset *mean* is
compatible with large per-image movement in both directions cancelling out, so
"the topology axis is two orders of magnitude below the residual scatter'' was
not an apples-to-apples comparison.

This script recomputes the topology axis as the same kind of image-level,
sigma-scaled quantity the fidelity axis uses:

``e1_topology_image.csv`` -- per (dataset, biomarker, repairer arm, seed)

    mean_signed_dB   mean_i (B_after,i - B_before,i) / sigma_b   <- the OLD number
    mean_abs_dB      mean_i |B_after,i - B_before,i| / sigma_b   <- |Delta B| / sigma
    rms_dB           sqrt(mean_i ((B_after,i-B_before,i)/sigma_b)^2)
    d_abs_err        mean_i (|e_after,i| - |e_before,i|)         <- paired change in
                                                                    absolute error
    with e_i = (B_i - B_GT,i) / sigma_b and a 2000-resample paired image
    bootstrap on every one of them.

``e1_topology_matched.csv`` -- per (dataset, biomarker, scope) on the C1
intervention corpus, the *pixel-budget-matched non-topological control*:

    h_topo   |Delta B_topo| / sigma_b     (the verified structural edit)
    h_ctrl   |Delta B_ctrl| / sigma_b     (a control that changes the same number
                                           of foreground pixels without changing
                                           connectivity)
    paired difference h_topo - h_ctrl, with a paired bootstrap CI clustered on
    the image, at both the event level and the image level.

One sigma for everything
------------------------
Every quantity here is divided by the **frozen training-split native** robust
scale in ``results/gateA_biomarker_scales_train.csv``.  The corpus columns
``Hnet_*`` shipped with ``c1_events_harm.parquet`` use the pre-registered
all-mask scale instead (ratio 0.94-1.05 to the training scale), which was the
one remaining exception on the sigma axis; this script removes the exception by
recomputing from ``absdB_*``/``cabsdB_*`` with the stored native-unit
conversion factor ``nativef_*`` and the stored training scale ``sigmatr_*``.

CLI
---
    cd exp
    python -m src.pivot.e1_topology_image
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import (EXP_ROOT, GATEA_DS, PIVOT_DIR, PRIMARY_COLS,
                              RESULTS_DIR, RUNS_DIR, sigma_table)

SEED = 0
N_BOOT = 2000
DATASETS = ("drive", "chasedb1", "hrf", "fives")
SEEDS = (0, 1, 2)

#: the primary measurement panel frozen in DECISIONS.md 2026-09-17 13:30 --
#: the four biomarkers on the skan pipeline.  The PVBM columns are retained
#: everywhere as a *sensitivity* panel (the two pipelines duplicate density
#: exactly and disagree about tortuosity, so eight equally weighted columns
#: would double-count).
PANEL = {c: ("primary" if c.endswith("_skan") else "sensitivity")
         for c in PRIMARY_COLS}

#: repairer arm -> directory under runs/ holding its per-image tables.
ARMS: Dict[str, Tuple[str, ...]] = {
    "rigr_uniform": ("rigr", "uniform"),
    "rigr_risk": ("rigr", "risk"),
    "geometric": ("repair", "geometric"),
    "evapore_scorer": ("repair", "evapore_scorer"),
    "evapore_e2e": ("repair", "evapore_e2e"),
    "rnca": ("repair", "rnca"),
}
#: the arm whose measured effect is quoted as the topology axis of the audit.
PRIMARY_ARM = "rigr_uniform"


# --------------------------------------------------------------- bootstrap --
def _boot_idx(n: int, n_boot: int, rng: np.random.RandomState) -> np.ndarray:
    return rng.randint(0, n, size=(n_boot, n))


def boot_stats(values: Dict[str, np.ndarray], n_boot: int = N_BOOT,
               seed: int = SEED) -> Dict[str, Tuple[float, float]]:
    """Paired image bootstrap: every statistic is recomputed on the *same*
    resampled images, so the CIs of quantities that are differences are CIs on
    the difference rather than on two independently resampled quantities."""
    keys = list(values)
    n = len(values[keys[0]])
    out = {k: (float("nan"), float("nan")) for k in keys}
    if n < 3:
        return out
    rng = np.random.RandomState(seed)
    idx = _boot_idx(n, n_boot, rng)
    for k in keys:
        v = np.asarray(values[k], float)
        draws = v[idx]
        with np.errstate(invalid="ignore"):
            m = np.nanmean(draws, axis=1)
        m = m[np.isfinite(m)]
        if m.size:
            out[k] = (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))
    return out


def boot_cluster(df: pd.DataFrame, cols: Sequence[str], cluster: str,
                 n_boot: int = N_BOOT, seed: int = SEED
                 ) -> Dict[str, Tuple[float, float]]:
    """Bootstrap resampling whole clusters (images), not rows (events).

    Events inside one image share the mask and are emphatically not
    independent, so an event-level bootstrap would give intervals that are far
    too narrow.
    """
    keys = list(cols)
    out = {k: (float("nan"), float("nan")) for k in keys}
    groups = list(df.groupby(cluster, sort=True).indices.values())
    n = len(groups)
    if n < 3:
        return out
    arrs = {k: df[k].to_numpy(float) for k in keys}
    rng = np.random.RandomState(seed)
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.randint(0, n, n)
        sel = np.concatenate([groups[j] for j in pick])
        for k in keys:
            with np.errstate(invalid="ignore"):
                m = np.nanmean(arrs[k][sel])
            if np.isfinite(m):
                draws[k].append(m)
    for k in keys:
        v = np.asarray(draws[k], float)
        if v.size:
            out[k] = (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return out


# ------------------------------------------------------- part 1: image level
def _per_image_path(arm: str, ds: str, seed: int) -> str:
    return os.path.join(RUNS_DIR, *ARMS[arm], ds, f"seed{seed}", "per_image.csv")


def image_level(sig: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Image-level topology effect for every repairer arm, seed and biomarker.

    Two ``macro`` pseudo-biomarkers are emitted alongside the real ones: the
    equally weighted mean over the primary (skan) panel and over all eight
    columns, computed **per image** so that the repairer table can carry a
    paired bootstrap CI on the change in macro absolute error rather than a
    bare point estimate.
    """
    rows: List[dict] = []
    for ds in DATASETS:
        for arm in ARMS:
            frames = []
            for sd in SEEDS:
                p = _per_image_path(arm, ds, sd)
                if not os.path.exists(p):
                    continue
                d = pd.read_csv(p).assign(
                    _seed=sd,
                    _src=os.path.relpath(p, EXP_ROOT).replace("\\", "/"))
                frames.append(d)
            if not frames:
                continue
            allseeds = pd.concat(frames, ignore_index=True)
            srcs = sorted(set(allseeds["_src"]))
            imgs_all = (allseeds["image"].to_numpy(str) if "image" in allseeds
                        else np.arange(len(allseeds)).astype(str))
            seeds_all = allseeds["_seed"].to_numpy(int)

            # ---- per-column standardised arrays, all on the same row order
            specs: List[Tuple[str, str, float, np.ndarray, np.ndarray, np.ndarray]] = []
            for col in PRIMARY_COLS:
                gcol, bcol, acol = (f"bio_gt_{col}", f"bio_before_{col}",
                                    f"bio_after_{col}")
                if not {gcol, bcol, acol} <= set(allseeds.columns):
                    continue
                s = sig.get(ds, {}).get(col, float("nan"))
                if not np.isfinite(s) or s <= 0:
                    continue
                g = allseeds[gcol].to_numpy(float)
                b = allseeds[bcol].to_numpy(float)
                a = allseeds[acol].to_numpy(float)
                specs.append((col, PANEL[col], s, (a - b) / s, (b - g) / s,
                              (a - g) / s))

            # ---- macro pseudo-columns (equal weight over a panel, per image)
            base_specs = list(specs)

            def _macro(name: str, keep) -> None:
                sel = [sp for sp in base_specs if keep(sp[0])]
                if not sel:
                    return
                dB = np.nanmean(np.vstack([sp[3] for sp in sel]), axis=0)
                eb = np.nanmean(np.vstack([np.abs(sp[4]) for sp in sel]), axis=0)
                ea = np.nanmean(np.vstack([np.abs(sp[5]) for sp in sel]), axis=0)
                # the macro rows carry |e| directly, so feed signed arrays whose
                # absolute value is already the panel mean
                specs.append((name, "macro", float("nan"), dB, eb, ea))

            _macro("macro_primary4", lambda c: c.endswith("_skan"))
            _macro("macro_all8", lambda c: True)

            for col, panel, s, dB0, e_b0, e_a0 in specs:
                for seed_label in [str(k) for k in sorted(set(seeds_all))] + ["pooled"]:
                    m = (np.ones(len(dB0), bool) if seed_label == "pooled"
                         else seeds_all == int(seed_label))
                    dB, e_b, e_a = dB0[m], e_b0[m], e_a0[m]
                    imgs = imgs_all[m]
                    ok = np.isfinite(dB) & np.isfinite(e_b) & np.isfinite(e_a)
                    if ok.sum() < 2:
                        continue
                    dB, e_b, e_a, imgs = dB[ok], e_b[ok], e_a[ok], imgs[ok]
                    vals = {
                        "mean_signed_dB": dB,
                        "mean_abs_dB": np.abs(dB),
                        "sq_dB": dB ** 2,
                        "d_abs_err": np.abs(e_a) - np.abs(e_b),
                    }
                    if seed_label == "pooled":
                        cis = boot_cluster(pd.DataFrame(dict(image=imgs, **vals)),
                                           list(vals), "image")
                    else:
                        cis = boot_stats(vals)
                    rms = float(np.sqrt(np.nanmean(dB ** 2)))
                    rms_lo = (float(np.sqrt(max(cis["sq_dB"][0], 0.0)))
                              if np.isfinite(cis["sq_dB"][0]) else float("nan"))
                    rms_hi = (float(np.sqrt(max(cis["sq_dB"][1], 0.0)))
                              if np.isfinite(cis["sq_dB"][1]) else float("nan"))
                    rows.append(dict(
                        dataset=ds, biomarker=col,
                        pipeline=(col.split("_")[-1] if panel != "macro" else "--"),
                        panel=panel, arm=arm, seed=seed_label,
                        n_images=int(len(set(imgs))), n_rows=int(ok.sum()),
                        mean_signed_dB=float(np.nanmean(dB)),
                        mean_signed_dB_lo=cis["mean_signed_dB"][0],
                        mean_signed_dB_hi=cis["mean_signed_dB"][1],
                        mean_abs_dB=float(np.nanmean(np.abs(dB))),
                        mean_abs_dB_lo=cis["mean_abs_dB"][0],
                        mean_abs_dB_hi=cis["mean_abs_dB"][1],
                        rms_dB=rms, rms_dB_lo=rms_lo, rms_dB_hi=rms_hi,
                        d_abs_err=float(np.nanmean(np.abs(e_a) - np.abs(e_b))),
                        d_abs_err_lo=cis["d_abs_err"][0],
                        d_abs_err_hi=cis["d_abs_err"][1],
                        abs_err_before=float(np.nanmean(np.abs(e_b))),
                        abs_err_after=float(np.nanmean(np.abs(e_a))),
                        sigma=s,
                        sigma_source="results/gateA_biomarker_scales_train.csv "
                                     "(training-split, observer-1, native)",
                        source="; ".join(srcs),
                    ))
    return pd.DataFrame(rows)


# --------------------------------------------- part 2: matched-control corpus
def matched_control() -> pd.DataFrame:
    """Paired topological-vs-pixel-budget-matched-control harm, per biomarker.

    The C1 corpus pairs every verified structural edit with a control edit that
    changes the *same number of foreground pixels* without changing
    connectivity.  The difference of the two absolute biomarker changes is what
    isolates the topology-specific consequence; ``control_ok == 1`` selects the
    events for which a valid matched control was actually found.
    """
    need = ["dataset", "image_id", "split", "observer", "type", "control_ok"]
    for c in PRIMARY_COLS:
        need += [f"absdB_{c}", f"cabsdB_{c}", f"dB_{c}", f"cdB_{c}",
                 f"nativef_{c}", f"sigmatr_{c}"]
    path = os.path.join(RESULTS_DIR, "c1_events_harm.parquet")
    ev = pd.read_parquet(path, columns=need)
    ev = ev[(ev["observer"] == "obs1") & (ev["split"] == "test") &
            (ev["control_ok"] == 1.0)]
    rel = os.path.relpath(path, EXP_ROOT).replace("\\", "/")

    rows: List[dict] = []
    for ds in DATASETS:
        g = ev[ev["dataset"] == GATEA_DS[ds]]
        if len(g) == 0:
            continue
        for scope, sub0 in (("sever", g[g["type"] == "sever"]),
                            ("all_types", g)):
            if len(sub0) == 0:
                continue
            for col in PRIMARY_COLS:
                conv = sub0[f"nativef_{col}"].to_numpy(float)
                sig = sub0[f"sigmatr_{col}"].to_numpy(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    h_topo = (sub0[f"absdB_{col}"].to_numpy(float) * conv) / sig
                    h_ctrl = (sub0[f"cabsdB_{col}"].to_numpy(float) * conv) / sig
                    z_topo = (sub0[f"dB_{col}"].to_numpy(float) * conv) / sig
                ok = np.isfinite(h_topo) & np.isfinite(h_ctrl)
                if ok.sum() < 3:
                    continue
                tab = pd.DataFrame({
                    "image_id": sub0["image_id"].to_numpy(str)[ok],
                    "h_topo": h_topo[ok],
                    "h_ctrl": h_ctrl[ok],
                    "h_net": (h_topo - h_ctrl)[ok],
                    "sq_topo": (z_topo[ok] ** 2),
                })
                ev_ci = boot_cluster(tab, ["h_topo", "h_ctrl", "h_net"], "image_id")
                per_img = tab.groupby("image_id", as_index=False).mean(numeric_only=True)
                im_ci = boot_stats({k: per_img[k].to_numpy(float)
                                    for k in ("h_topo", "h_ctrl", "h_net")})
                rows.append(dict(
                    dataset=ds, biomarker=col, pipeline=col.split("_")[-1],
                    panel=PANEL[col], scope=scope,
                    n_events=int(ok.sum()), n_images=int(per_img.shape[0]),
                    events_per_image=float(ok.sum()) / max(per_img.shape[0], 1),
                    h_topo=float(np.nanmean(tab["h_topo"])),
                    h_topo_lo=ev_ci["h_topo"][0], h_topo_hi=ev_ci["h_topo"][1],
                    h_ctrl=float(np.nanmean(tab["h_ctrl"])),
                    h_ctrl_lo=ev_ci["h_ctrl"][0], h_ctrl_hi=ev_ci["h_ctrl"][1],
                    h_net=float(np.nanmean(tab["h_net"])),
                    h_net_lo=ev_ci["h_net"][0], h_net_hi=ev_ci["h_net"][1],
                    img_h_topo=float(np.nanmean(per_img["h_topo"])),
                    img_h_ctrl=float(np.nanmean(per_img["h_ctrl"])),
                    img_h_net=float(np.nanmean(per_img["h_net"])),
                    img_h_net_lo=im_ci["h_net"][0], img_h_net_hi=im_ci["h_net"][1],
                    rms_event_dB=float(np.sqrt(np.nanmean(tab["sq_topo"]))),
                    sigma_source="sigmatr_* column of c1_events_harm.parquet "
                                 "(= results/gateA_biomarker_scales_train.csv), "
                                 "with |dB| converted to native units by nativef_*",
                    source=f"{rel} (observer=obs1, split=test, control_ok=1)",
                ))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- main --
def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=PIVOT_DIR)
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    sig = sigma_table()

    img = image_level(sig)
    p1 = os.path.join(args.out_dir, "e1_topology_image.csv")
    img.to_csv(p1, index=False)
    print("wrote", p1, img.shape)

    mat = matched_control()
    p2 = os.path.join(args.out_dir, "e1_topology_matched.csv")
    mat.to_csv(p2, index=False)
    print("wrote", p2, mat.shape)

    # a short console summary so a run is self-checking
    prim = img[(img["arm"] == PRIMARY_ARM) & (img["seed"] == "pooled") &
               (img["panel"] == "primary")]
    if len(prim):
        print(f"\n[{PRIMARY_ARM}, pooled seeds, primary skan panel]")
        print(prim[["dataset", "biomarker", "mean_signed_dB", "mean_abs_dB",
                    "rms_dB", "d_abs_err"]].to_string(index=False))
    sev = mat[(mat["scope"] == "sever") & (mat["panel"] == "primary")]
    if len(sev):
        print("\n[matched control, sever events, primary skan panel]")
        print(sev[["dataset", "biomarker", "n_events", "h_topo", "h_ctrl",
                   "h_net", "h_net_lo", "h_net_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
