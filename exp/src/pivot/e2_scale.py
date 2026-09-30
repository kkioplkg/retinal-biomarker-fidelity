"""E2-scale -- does inference resolution explain HRF's poor measurement fidelity?

Pre-registered probe, DECISIONS.md 2026-09-17 18:58.

HRF is the one set where the baseline segmenter measures badly (r = 0.29-0.36
for FD and total_length against 0.90-0.98 on FIVES) *and* under-measures by
-1.6 to -2.2 sigma.  HRF is also the set whose native frame (3504 x 2336) is
downsampled hardest by the training convention (longest side 1536, i.e. 0.44x).
If that downsampling is what destroys the thin vessels the biomarkers count,
then re-inferring the *same frozen checkpoints* at a larger scale should
recover fidelity without retraining anything -- and the whole HRF fidelity
deficit would be an inference-resolution artefact rather than a property of the
segmenter.

Design
------
    dataset      HRF test split (30 images)
    checkpoints  baseline   runs/seg/hrf/seed<k>/best.pt
                 continued  runs/pivot/hrf/seed<k>/continued/last.pt
                 (both frozen; nothing is trained here)
    seeds        0, 1, 2
    scales       1536 (already on disk, the reference), 2048, 2560,
                 3504 (native -- no resampling at all)
    patch        768, stride 384 (the script default, patch // 2)

``continued`` is included because it is the same-budget control: if a scale
effect appears it must appear in both arms, otherwise it is confounded with
the arm.  ReliSeg is deliberately NOT run here -- this probe asks about
resolution, not about the loss.

Outputs
-------
    results/pivot/e2_scale.csv    per (config, seed, scale, biomarker,
                                  pipeline): n, sigma, r_pearson, r_spearman,
                                  bias_const_sigma, resid_sd_sigma, and
                                  delta vs the 1536 reference with a paired
                                  image bootstrap CI (per seed and pooled)
    results/pivot/e2_scale_pixel.csv   Dice / clDice and seconds-per-image at
                                  each scale
    a section appended to results/pivot/E2_REPORT.md

CLI
---
    python -m src.pivot.e2_scale run --min-free 4000   # infer + bio, gated
    python -m src.pivot.e2_scale analyze
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.pivot.common import EXP_ROOT, PIVOT_DIR, PRIMARY_COLS, sigma_table
from src.pivot.e2_analysis import (N_BOOT, PANEL, PRIMARY_PIPELINE,
                                   SENSITIVITY_PIPELINE, DELTA_R_MARGIN,
                                   DELTA_BIAS_MARGIN, PIXEL_MARGIN,
                                   boot_idx, boot_r, ci, gt_table, pearson,
                                   rel, sigma_for, spearman, split_col, srt,
                                   t_ci, _key, _tbl)

DATASET = "hrf"
SEEDS = (0, 1, 2)
#: 1536 is the training convention and the reference; it is already on disk
REFERENCE_SCALE = 1536
SCALES = (2048, 2560, 3504)          # 3504 = HRF native longest side
CONFIGS = ("baseline", "continued")
PY = sys.executable


def ckpt_of(cfg: str, seed: int) -> str:
    if cfg == "baseline":
        return os.path.join("runs", "seg", DATASET, f"seed{seed}", "best.pt")
    return os.path.join("runs", "pivot", DATASET, f"seed{seed}", cfg, "last.pt")


def root_of(cfg: str, seed: int, scale: int) -> str:
    """Prediction root.  The 1536 reference is the E2 in-domain run already
    on disk -- it is never recomputed, so the comparison is against exactly
    the numbers the main report used."""
    base = os.path.join("runs", "pivot", DATASET, f"seed{seed}", cfg)
    return os.path.join(base, "pred" if scale == REFERENCE_SCALE
                        else f"pred_L{scale}")


def tag_of(cfg: str, seed: int, scale: int) -> str:
    return f"e2scale_{DATASET}_s{seed}_{cfg}_L{scale}"


#: Order the work by how much it tells us, not by ascending scale.  3504 is
#: the decisive contrast -- native resolution, no resampling at all, the
#: largest departure from the 1536 training convention -- so if resolution
#: explains HRF's fidelity deficit it shows there first and largest.  Running
#: it first means the headline exists ~2 h earlier than it would if the probe
#: walked the scales in order.
RUN_ORDER = (3504, 2048, 2560)


def units() -> List[Tuple[str, int, int]]:
    return [(c, k, L) for L in RUN_ORDER for c in CONFIGS for k in SEEDS]


# --------------------------------------------------------------------------
# run: GPU inference (memory-gated) then CPU biomarkers
# --------------------------------------------------------------------------
def free_mem() -> List[int]:
    """Free MiB per GPU, index-ordered."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30).stdout
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:                                            # noqa: BLE001
        return []


def pick_gpu(min_free: int, poll: int = 60, quiet: bool = False) -> int:
    """Block until some GPU has ``min_free`` MiB, then return its index.

    E4 has priority on these cards (DECISIONS 2026-09-17 13:15): this probe
    waits for room rather than competing for it.
    """
    said = False
    while True:
        f = free_mem()
        if f:
            i = int(np.argmax(f))
            if f[i] >= min_free:
                return i
            if not said and not quiet:
                print("[scale] waiting for %d MiB free (now %s)" % (min_free, f),
                      flush=True)
                said = True
        time.sleep(poll)


def _sh(argv: List[str], log: str) -> int:
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n===== %s %s =====\n" % (time.strftime("%Y-%m-%d %H:%M:%S"),
                                            " ".join(argv)))
        fh.flush()
        return subprocess.call([PY] + argv, cwd=EXP_ROOT,
                               env=dict(os.environ, PYTHONUNBUFFERED="1"),
                               stdout=fh, stderr=subprocess.STDOUT)


def cmd_run(a) -> int:
    logdir = os.path.join("runs", "pivot", "_e2_logs")
    fails = []
    for cfg, k, L in units():
        root, tag = root_of(cfg, k, L), tag_of(cfg, k, L)
        meta = os.path.join(root, "infer_meta.json")
        bio = os.path.join(root, "bio.csv")
        log = os.path.join(logdir, f"scale_{DATASET}_s{k}_{cfg}_L{L}.log")

        if not os.path.exists(meta):
            gpu = pick_gpu(a.min_free)
            print("[scale] infer %s on cuda:%d" % (tag, gpu), flush=True)
            t0 = time.time()
            rc = _sh(["-m", "src.pivot.p5_eval", "infer", "--dataset", DATASET,
                      "--ckpt", ckpt_of(cfg, k), "--tag", tag, "--root", root,
                      "--infer-longest", str(L), "--infer-patch", "768",
                      "--gpu", str(gpu)], log)
            if rc != 0 or not os.path.exists(meta):
                print("[scale] FAILED infer %s rc=%s" % (tag, rc), flush=True)
                fails.append("infer:" + tag)
                continue
            print("[scale]   infer %s done in %.1f min"
                  % (tag, (time.time() - t0) / 60), flush=True)

        if not os.path.exists(bio):
            print("[scale] bio %s" % tag, flush=True)
            rc = _sh(["-m", "src.pivot.p5_eval", "bio", "--dataset", DATASET,
                      "--tag", tag, "--root", root,
                      "--procs", str(a.procs)], log)
            if rc != 0 or not os.path.exists(bio):
                print("[scale] FAILED bio %s rc=%s" % (tag, rc), flush=True)
                fails.append("bio:" + tag)
    print("[scale] done. failures=%s" % fails, flush=True)
    return 1 if fails else 0


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------
def _load(cfg: str, k: int, L: int) -> Optional[pd.DataFrame]:
    f = os.path.join(EXP_ROOT, root_of(cfg, k, L), "bio.csv")
    return pd.read_csv(f) if os.path.exists(f) else None


def _timing(cfg: str, k: int, L: int) -> Dict[str, float]:
    """Wall time per image at this scale, from the inference metadata."""
    f = os.path.join(EXP_ROOT, root_of(cfg, k, L), "infer_meta.json")
    if not os.path.exists(f):
        return {}
    m = json.load(open(f, encoding="utf-8"))
    n = float(m.get("n_images") or 0)
    tot = float(m.get("total_seconds") or 0)
    return {"n_images": n, "total_seconds": tot,
            "seconds_per_image": (tot / n) if n else np.nan,
            "mean_dice": m.get("mean_dice"), "mean_cldice": m.get("mean_cldice"),
            "resize_longest": m.get("resize_longest"), "patch": m.get("patch"),
            "stride": m.get("stride")}


def cmd_analyze(a) -> int:
    sig = sigma_table()
    gt = gt_table(DATASET)
    gt_src = str(gt["__src"].iloc[0])
    sigs, sig_src = sigma_for(DATASET, sig, gt)
    gt = gt.assign(__k=_key(gt["image_id"]))

    all_scales = (REFERENCE_SCALE,) + SCALES
    frames: Dict[Tuple[str, int, int], pd.DataFrame] = {}
    missing = []
    for cfg in CONFIGS:
        for k in SEEDS:
            for L in all_scales:
                d = _load(cfg, k, L)
                if d is None:
                    missing.append(rel(os.path.join(root_of(cfg, k, L), "bio.csv")))
                else:
                    frames[(cfg, k, L)] = d.assign(__k=_key(d["image_id"]))

    rows: List[dict] = []
    prows: List[dict] = []
    for cfg in CONFIGS:
        for k in SEEDS:
            have = [L for L in all_scales if (cfg, k, L) in frames]
            if REFERENCE_SCALE not in have or len(have) < 2:
                continue
            common = set(gt["__k"])
            for L in have:
                common &= set(frames[(cfg, k, L)]["__k"])
            common = sorted(common)
            if len(common) < 5:
                continue
            g = gt.set_index("__k").loc[common]
            A = {L: frames[(cfg, k, L)].set_index("__k").loc[common] for L in have}

            for L in have:
                t = _timing(cfg, k, L)
                prows.append(dict(dataset=DATASET, config=cfg, seed=k, scale=L,
                                  is_reference=(L == REFERENCE_SCALE), **t,
                                  px_path=rel(os.path.join(
                                      root_of(cfg, k, L), "pixel_metrics.csv"))))

            for c in PRIMARY_COLS:
                bm, pipe = split_col(c)
                s = float(sigs.get(c, np.nan))
                y = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
                xs = {L: pd.to_numeric(A[L][c], errors="coerce").to_numpy(float)
                      for L in have}
                ok = np.isfinite(y)
                for x in xs.values():
                    ok &= np.isfinite(x)
                if ok.sum() < 5:
                    continue
                yy, xx = y[ok], {L: x[ok] for L, x in xs.items()}
                bidx = boot_idx(int(ok.sum()))
                ref_boot = boot_r(xx[REFERENCE_SCALE], yy, bidx)
                r_ref = pearson(xx[REFERENCE_SCALE], yy)
                for L in have:
                    e = ((xx[L] - yy) / s if np.isfinite(s) and s > 0
                         else np.full_like(xx[L], np.nan))
                    row = dict(dataset=DATASET, config=cfg, seed=k, scale=L,
                               is_reference=(L == REFERENCE_SCALE),
                               biomarker=bm, pipeline=pipe,
                               n=int(ok.sum()), sigma=s,
                               r_pearson=pearson(xx[L], yy),
                               r_spearman=spearman(xx[L], yy),
                               bias_const_sigma=float(np.mean(e)),
                               resid_sd_sigma=float(np.std(e, ddof=1)),
                               d_r=np.nan, lo=np.nan, hi=np.nan,
                               d_bias_sigma=np.nan,
                               pred_path=rel(os.path.join(root_of(cfg, k, L),
                                                          "bio.csv")),
                               gt_path=gt_src, sigma_source=sig_src)
                    if L != REFERENCE_SCALE:
                        dd = boot_r(xx[L], yy, bidx) - ref_boot
                        lo, hi = ci(dd)
                        row.update(d_r=pearson(xx[L], yy) - r_ref, lo=lo, hi=hi,
                                   d_bias_sigma=float(
                                       np.mean(e) - np.mean(
                                           (xx[REFERENCE_SCALE] - yy) / s))
                                   if np.isfinite(s) and s > 0 else np.nan)
                    rows.append(row)

    df = pd.DataFrame(rows)
    # ---- pooled over seeds: mean of the per-seed delta-r, both CIs ----
    pool = []
    if len(df):
        for (cfg, L, bm, pipe), grp in df[~df["is_reference"]].groupby(
                ["config", "scale", "biomarker", "pipeline"]):
            pts = grp["d_r"].dropna().tolist()
            if len(pts) < 2:
                continue
            slo, shi = t_ci(pts)
            npos = int(sum(1 for v in pts if v > 0))
            pool.append(dict(
                dataset=DATASET, config=cfg, seed="pooled", scale=L,
                is_reference=False, biomarker=bm, pipeline=pipe,
                n=int(grp["n"].iloc[0]), sigma=grp["sigma"].iloc[0],
                r_pearson=float(grp["r_pearson"].mean()),
                r_spearman=float(grp["r_spearman"].mean()),
                bias_const_sigma=float(grp["bias_const_sigma"].mean()),
                resid_sd_sigma=float(grp["resid_sd_sigma"].mean()),
                d_r=float(np.mean(pts)), lo=slo, hi=shi,
                d_bias_sigma=float(grp["d_bias_sigma"].mean()),
                n_seeds=len(pts), n_seeds_positive=npos,
                direction_consistent=("yes" if npos in (0, len(pts)) else "no"),
                pred_path="runs/pivot/hrf/seed{0,1,2}/%s/pred_L%d/bio.csv"
                          % (cfg, L),
                gt_path=grp["gt_path"].iloc[0],
                sigma_source=grp["sigma_source"].iloc[0]))
    if pool:
        df = pd.concat([df, pd.DataFrame(pool)], ignore_index=True)

    # ---- safety flags against the pre-registered margins ----
    if len(df):
        d = df["d_r"]
        df["flag_delta_r"] = np.where(
            d.notna() & (d < DELTA_R_MARGIN), "BREACH",
            np.where(d.notna() & (df["lo"] < DELTA_R_MARGIN), "AT-RISK", "ok"))
        b = df["d_bias_sigma"].abs()
        df["flag_delta_bias"] = np.where(
            b.notna() & (b > DELTA_BIAS_MARGIN), "BREACH", "ok")
        df.loc[df["is_reference"], ["flag_delta_r", "flag_delta_bias"]] = ""

    px = pd.DataFrame(prows)
    if len(px):
        ref = px[px["is_reference"]][["config", "seed", "mean_dice",
                                      "mean_cldice", "seconds_per_image"]]
        ref = ref.rename(columns={"mean_dice": "ref_dice",
                                  "mean_cldice": "ref_cldice",
                                  "seconds_per_image": "ref_s_per_img"})
        px = px.merge(ref, on=["config", "seed"], how="left")
        px["d_dice"] = px["mean_dice"] - px["ref_dice"]
        px["d_cldice"] = px["mean_cldice"] - px["ref_cldice"]
        px["slowdown_x"] = px["seconds_per_image"] / px["ref_s_per_img"]
        px["flag_cldice"] = np.where(
            px["d_cldice"].notna() & (px["d_cldice"] < -PIXEL_MARGIN),
            "BREACH", "ok")
        px.loc[px["is_reference"], "flag_cldice"] = ""

    os.makedirs(a.out_dir, exist_ok=True)
    p1 = os.path.join(a.out_dir, "e2_scale.csv")
    p2 = os.path.join(a.out_dir, "e2_scale_pixel.csv")
    df.to_csv(p1, index=False)
    px.to_csv(p2, index=False)
    print("-> %s (%d rows)" % (rel(p1), len(df)))
    print("-> %s (%d rows)" % (rel(p2), len(px)))
    if missing:
        print("[scale] %d missing bio.csv (probe incomplete)" % len(missing))

    append_section(df, px, missing, a.report)
    return 0


def append_section(df: pd.DataFrame, px: pd.DataFrame,
                   missing: Sequence[str], report: str) -> None:
    SK = PRIMARY_PIPELINE
    L: List[str] = []
    A = L.append
    A("\n\n---\n\n## 9. Inference-scale probe (HRF, frozen checkpoints)\n")
    A("Pre-registered %s (DECISIONS.md 2026-09-17 18:58). Sources: "
      "`results/pivot/e2_scale.csv`, `results/pivot/e2_scale_pixel.csv`.\n"
      % time.strftime("%Y-%m-%d"))
    A("\nHRF is the set where fidelity is worst (baseline r = 0.29-0.36 for FD "
      "and total_length) **and** the one downsampled hardest by the training "
      "convention: its native frame is 3504 x 2336, so longest-side 1536 is a "
      "0.44x resampling. This probe re-runs the **same frozen checkpoints** at "
      "2048 / 2560 / 3504 (native) against the 1536 run already on disk. "
      "Nothing is trained. `continued` is included as the same-budget control: "
      "a genuine scale effect must appear in both arms.\n")
    A("\nDelta-r is L minus the 1536 reference, paired image bootstrap over the "
      "same 30 HRF test images, %d resamples. Pooled rows give the mean of the "
      "per-seed delta-r with a 95%% t interval over the three seeds.\n" % N_BOOT)

    if len(df):
        A("\n### Absolute fidelity by scale (skan, seed-averaged)\n\n")
        f = df[(df["pipeline"] == SK) & df["biomarker"].isin(PANEL)
               & (df["seed"].astype(str) != "pooled")]
        agg = (f.groupby(["config", "biomarker", "scale"])
               [["r_pearson", "bias_const_sigma", "resid_sd_sigma"]]
               .mean().reset_index())
        A(_tbl(srt(agg, ["config", "biomarker", "scale"]),
               ["config", "biomarker", "scale", "r_pearson",
                "bias_const_sigma", "resid_sd_sigma"]))

        A("\n### Delta-r vs 1536, pooled over seeds (skan)\n\n")
        p = df[(df["pipeline"] == SK) & df["biomarker"].isin(PANEL)
               & (df["seed"].astype(str) == "pooled")]
        A(_tbl(srt(p, ["config", "biomarker", "scale"]),
               ["config", "biomarker", "scale", "d_r", "lo", "hi",
                "n_seeds_positive", "direction_consistent", "d_bias_sigma",
                "flag_delta_r", "flag_delta_bias"]))

        A("\n### Delta-r vs 1536, seed by seed (skan)\n\n")
        s = df[(df["pipeline"] == SK) & df["biomarker"].isin(PANEL)
               & (df["seed"].astype(str) != "pooled") & ~df["is_reference"]]
        A(_tbl(srt(s, ["config", "biomarker", "scale", "seed"]),
               ["config", "seed", "biomarker", "scale", "d_r", "lo", "hi"]))

        A("\n### Sensitivity -- PVBM pipeline, pooled\n\n")
        q = df[(df["pipeline"] == SENSITIVITY_PIPELINE)
               & df["biomarker"].isin(PANEL)
               & (df["seed"].astype(str) == "pooled")]
        A(_tbl(srt(q, ["config", "biomarker", "scale"]),
               ["config", "biomarker", "scale", "d_r", "lo", "hi",
                "direction_consistent"]))

    if len(px):
        A("\n### Pixel metrics and cost by scale (seed-averaged)\n\n")
        agg = (px.groupby(["config", "scale"])
               [["mean_dice", "mean_cldice", "d_dice", "d_cldice",
                 "seconds_per_image", "slowdown_x"]].mean().reset_index())
        A(_tbl(srt(agg, ["config", "scale"]),
               ["config", "scale", "mean_dice", "mean_cldice", "d_dice",
                "d_cldice", "seconds_per_image", "slowdown_x"], nd=4))
        br = px[px["flag_cldice"] == "BREACH"]
        A("\nclDice non-inferiority (margin %.2f): %d breaching run(s).\n"
          % (PIXEL_MARGIN, len(br)))

    if missing:
        A("\n**Probe incomplete** -- %d prediction tables missing:\n\n" % len(missing))
        for m in sorted(set(missing))[:24]:
            A("- `%s`\n" % m)

    with open(report, "a", encoding="utf-8") as fh:
        fh.write("".join(L))
    print("-> appended section 9 to %s" % rel(report))


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--min-free", type=int, default=4000,
                   help="MiB that must be free on a GPU before a job starts; "
                        "E4 has priority on these cards, so this probe waits "
                        "rather than competing")
    r.add_argument("--procs", type=int, default=12)
    r.set_defaults(fn=cmd_run)
    an = sub.add_parser("analyze")
    an.add_argument("--out-dir", default=PIVOT_DIR)
    an.add_argument("--report", default=os.path.join(PIVOT_DIR, "E2_REPORT.md"))
    an.set_defaults(fn=cmd_analyze)
    a = ap.parse_args(argv)
    os.chdir(EXP_ROOT)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
