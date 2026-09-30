"""C1 driver: run the perturbation engine over the four main datasets.

Plan reference: exp/EXPERIMENT_PLAN.md S3.2 -- "per image ~40 loci per
perturbation family, both pipelines measured, store signed dB, |dB|, the
control dB and the Contract-F event features phi".

For every image the driver

1. loads ``(I, M_gt, FOV)``, optionally downscales to ``--max-side``,
2. locates the optic disc automatically (``src.bio.disc.locate_optic_disc``),
3. builds the GT context (skeleton, skan graph, quality-gated locus pool),
4. caches ``B(M)`` once,
5. samples ``--n-sever`` loci x 4 capsule lengths, ``--n-bridge`` Bezier
   bridges, ``--n-truncate`` terminal branches x 3 rho, ``--n-caliber``
   topology-neutral caliber controls,
6. builds the pixel- and context-matched control ``M_C`` for **every**
   perturbation, measures ``B(M')`` and ``B(M_C)``,
7. computes the conventional metrics of ``M'`` against the GT (Dice, clDice,
   Betti errors, BCS -- with the GT-side work cached, using the definitions of
   ``src.topo.metrics`` verbatim),
8. extracts ``phi`` (Contract F) and the GT-only strata separately.

Output
------
``exp/results/c1_events.parquet`` (+ ``.csv``)
    one row per perturbation event.
``exp/results/c1_summary.json``
    counts, verification-failure rates, control-failure rates, runtimes.

Usage
-----
    cd exp
    python -m src.c1.run_c1 --limit 2 --datasets DRIVE CHASE_DB1   # smoke
    python -m src.c1.run_c1 --workers 6                            # full run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# BLAS stays single-threaded: we parallelise over images, not inside them.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
RUNS_DIR = os.path.join(EXP_ROOT, "runs", "c1")

DATASETS = ("DRIVE", "CHASE_DB1", "HRF", "FIVES")
FIVES_SUBSET_N = 60

#: working resolution per dataset (longest side, 0 = native).  HRF and FIVES
#: are 8.2 MP / 4.2 MP; PVBM's box-counting fractal estimator costs ~25x more
#: there than on DRIVE, which would blow the 6 h budget on its own.  1536 is the
#: same working resolution the plan already fixes for HRF/FIVES segmentation
#: (S2: "HRF/FIVES scaled to longest side 1536"), so C1 stays commensurate with
#: the rest of the pipeline.  Vessel radii stay >= 2 px at that scale.
DEFAULT_MAX_SIDE = {"DRIVE": 0, "CHASE_DB1": 0, "HRF": 1536, "FIVES": 1536, "STARE": 0}

#: per-image event budget (EXPERIMENT_PLAN S3.2)
DEFAULT_N_SEVER = 40
DEFAULT_N_BRIDGE = 25
DEFAULT_N_TRUNCATE = 20
DEFAULT_N_CALIBER = 20

#: Per-dataset budget overrides, as ``(n_sever, n_bridge, n_truncate,
#: n_caliber, severity_mode)``.
#:
#: DRIVE and CHASE_DB1 run the full plan: ~40 severance loci, each taken through
#: the whole ladder ``L in {0.5, 1, 2, 4} D`` ("full" mode), so the dose-response
#: of Fig. 2 is measured *within* a locus.  HRF and FIVES cost ~4x more per
#: event (compute_B is 8.0 s at 1536 px vs 2.1 s on DRIVE) and carry most of the
#: image units, so there each locus receives a **single** severity, cycling
#: through the four L factors across loci ("cycle" mode).  That keeps the four
#: severity levels balanced and the anatomical coverage wide, at the cost of the
#: within-locus paired contrast, which the two smaller domains still provide.
DATASET_BUDGET = {
    "DRIVE":     (40, 25, 20, 20, "full"),
    "CHASE_DB1": (40, 25, 20, 20, "full"),
    "HRF":       (60, 15, 10, 12, "cycle"),
    "FIVES":     (60, 15, 10, 12, "cycle"),
    "STARE":     (40, 25, 20, 20, "full"),
}

#: PVBM ``MultifractalVBMs`` rotation count used throughout C1.
#:
#: PVBM rotates the mask over ``np.linspace(0, 360, n_rotations)`` and keeps the
#: rotation whose D0 > D1 > D2 ordering is best (``_optimize_dqs``).  The chosen
#: optimum is essentially always an **axis-aligned** rotation, because 90 / 180 /
#: 270 degrees are the only angles that need no interpolation.  A grid contains
#: those angles exactly when ``n_rotations - 1`` is a multiple of 4, i.e.
#: ``n in {5, 9, 13, ..., 25}``.
#:
#: Measured on 10 DRIVE masks (``--fd-check``): ``n = 5, 9, 13`` all reproduce
#: the ``n = 25`` FD **bit for bit** (max |dFD| = 0.000000, Spearman 1.0000) at
#: 2.19 s vs 3.88 s per ``compute_B``, while the off-grid counts
#: ``n = 4, 8, 12, 16, 20`` give a *different* estimator (Spearman 0.77 on DRIVE,
#: 0.89 on CHASE_DB1 against n = 25) and would break dimensional closure with the
#: Gate A sigma_B, which was estimated at n = 25.
#:
#: 5 is therefore the cheapest count that is exactly PVBM's default estimator.
FD_ROTATIONS = 5

#: biomarker dict keys that are not measurements
_SKIP_B_KEYS = ("runtime_", "version", "error", "disc_", "_ok", "import_")


# --------------------------------------------------------------------------- #
# biomarker wrapper
# --------------------------------------------------------------------------- #
def compute_B(mask, fov, disc, fd_rotations: int, zone_b: bool) -> Dict[str, object]:
    """``B(mask)`` from both pipelines.

    ``zone_b=True`` calls :func:`src.bio.biomarkers.compute_all` unchanged.
    ``zone_b=False`` calls the two pipelines directly with PVBM's zone-B
    annulus disabled and re-assembles exactly the same primary columns using
    ``biomarkers.PRIMARY_SOURCE``: the zone-B variants roughly double the PVBM
    cost and the C1 analysis stratifies by the *locus* zone, not by
    zone-restricted biomarkers.
    """
    from ..bio.biomarkers import compute_all, PRIMARY_SOURCE

    if zone_b:
        return compute_all(mask, fov, image=None, disc=disc, fd_rotations=fd_rotations)

    from ..bio.pvbm_pipe import compute_biomarkers_pvbm
    from ..bio.skan_pipe import compute_biomarkers_skan

    out: Dict[str, object] = {}
    t0 = time.perf_counter()
    try:
        pv = compute_biomarkers_pvbm(
            mask, fov, disc=disc, fd_rotations=fd_rotations, zone_b=False
        )
    except Exception as exc:  # noqa: BLE001
        pv = {"pipeline_error": f"{type(exc).__name__}: {exc}"}
    out["runtime_pvbm_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        sk = compute_biomarkers_skan(mask, fov, disc=disc)
    except Exception as exc:  # noqa: BLE001
        sk = {"pipeline_error": f"{type(exc).__name__}: {exc}"}
    out["runtime_skan_s"] = time.perf_counter() - t0

    out.update({f"pvbm_{k}": v for k, v in pv.items()})
    out.update({f"skan_{k}": v for k, v in sk.items()})
    for name, (pk, skk) in PRIMARY_SOURCE.items():
        out[f"{name}_pvbm"] = _num(pv.get(pk))
        out[f"{name}_skan"] = _num(sk.get(skk))
    return out


def _num(v) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def biomarker_keys(b: Dict[str, object]) -> List[str]:
    """Numeric measurement keys of a ``B(.)`` dict."""
    out = []
    for k, v in b.items():
        if any(s in k for s in _SKIP_B_KEYS):
            continue
        if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
            out.append(k)
    return sorted(out)


# --------------------------------------------------------------------------- #
# cached conventional metrics (identical definitions to src.topo.metrics)
# --------------------------------------------------------------------------- #
class CachedMetrics:
    """Dice / clDice / Betti errors / BCS of ``M'`` against the GT.

    The GT-side work (unpruned GT skeleton for clDice, GT Betti numbers, GT
    bifurcation probe points for BCS) is computed once per image; the
    definitions are those of :mod:`src.topo.metrics`, imported from there where
    possible so the two can never drift apart.
    """

    def __init__(self, gt: np.ndarray, fov: np.ndarray, probe_px: int = 15,
                 min_branch_px: int = 3, snap_px: float = 2.0):
        from ..topo import metrics as MT
        from ..topo import skeleton as SK

        self._MT = MT
        self._SK = SK
        self.gt = SK.as_bool(gt) & SK.fov_or_true(fov, SK.as_bool(gt).shape)
        self.fov = SK.fov_or_true(fov, self.gt.shape)
        self.snap_px = float(snap_px)
        self.gt_skel_raw = SK.skeletonize_mask(self.gt, prune=False)
        self.n_gt_skel = int(self.gt_skel_raw.sum())
        self.beta_gt = SK.betti_numbers(self.gt, self.fov)
        gskel = SK.skeletonize_mask(self.gt, min_branch_px=min_branch_px)
        self.probes = list(MT._probe_points(gskel, probe_px, 3, self.fov))
        self.n_gt_px = int(self.gt.sum())

    def evaluate(self, pred: np.ndarray) -> Dict[str, float]:
        SK = self._SK
        from scipy import ndimage as ndi

        p = SK.as_bool(pred) & self.fov
        inter = int(np.count_nonzero(p & self.gt))
        np_px = int(p.sum())
        dice = (2.0 * inter / (np_px + self.n_gt_px)) if (np_px + self.n_gt_px) else 1.0

        sp = SK.skeletonize_mask(p, prune=False)
        n_sp = int(sp.sum())
        if n_sp == 0 and self.n_gt_skel == 0:
            cldice = 1.0
        elif n_sp == 0 or self.n_gt_skel == 0:
            cldice = 0.0
        else:
            t_prec = float(np.count_nonzero(sp & self.gt)) / float(n_sp)
            t_sens = float(np.count_nonzero(self.gt_skel_raw & p)) / float(self.n_gt_skel)
            cldice = 0.0 if (t_prec + t_sens) == 0 else 2.0 * t_prec * t_sens / (t_prec + t_sens)

        b0p, b1p = SK.betti_numbers(p, self.fov)

        plab, _ = SK.connected_components(p, connectivity=8)
        if self.snap_px > 0 and p.any():
            dist, idx = ndi.distance_transform_edt(~p, return_indices=True)
        else:
            dist, idx = None, None
        n_total = n_conn = 0
        for _c, probes in self.probes:
            n_total += 1
            comps = set()
            ok = True
            for (r, c) in probes:
                lab_v = int(plab[r, c])
                if lab_v == 0 and dist is not None and dist[r, c] <= self.snap_px:
                    lab_v = int(plab[idx[0][r, c], idx[1][r, c]])
                if lab_v == 0:
                    ok = False
                    break
                comps.add(lab_v)
            if ok and len(comps) == 1:
                n_conn += 1
        bcs = float(n_conn / n_total) if n_total else float("nan")

        return {
            "m_dice": float(dice),
            "m_cldice": float(cldice),
            "m_beta0_err": float(abs(b0p - self.beta_gt[0])),
            "m_beta1_err": float(abs(b1p - self.beta_gt[1])),
            "m_beta0_pred": float(b0p),
            "m_beta1_pred": float(b1p),
            "m_bcs": bcs,
            "m_n_pred_px": float(np_px),
        }

    def gt_reference(self) -> Dict[str, float]:
        """Metrics of the GT against itself (the perfect-prediction reference)."""
        return {
            "m_dice": 1.0, "m_cldice": 1.0, "m_beta0_err": 0.0, "m_beta1_err": 0.0,
            "m_beta0_pred": float(self.beta_gt[0]), "m_beta1_pred": float(self.beta_gt[1]),
            "m_bcs": self.evaluate(self.gt)["m_bcs"], "m_n_pred_px": float(self.n_gt_px),
        }


# --------------------------------------------------------------------------- #
# per-image shards (resumability)
# --------------------------------------------------------------------------- #
def shard_dir(out_dir: str, tag: str = "") -> str:
    return os.path.join(out_dir, "c1_shards" + (f"_{tag}" if tag else ""))


def shard_path(out_dir: str, task: Dict[str, Any], tag: str = "") -> str:
    """One file per (dataset, image, observer) -- the unit of resumability."""
    stem = f"{task['dataset']}__{task['image_id']}__{task['observer']}"
    stem = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in stem)
    return os.path.join(shard_dir(out_dir, tag), stem + ".parquet")


def write_shard(rows: List[Dict[str, Any]], path: str) -> str:
    """Write one image's events atomically (temp file + rename)."""
    import pandas as pd

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows)
    tmp = path + ".tmp"
    try:
        df.to_parquet(tmp, index=False)
        out = path
    except Exception:  # noqa: BLE001 - no parquet engine available
        tmp = os.path.splitext(path)[0] + ".csv.tmp"
        out = os.path.splitext(path)[0] + ".csv"
        df.to_csv(tmp, index=False)
    if os.path.exists(out):
        os.remove(out)
    os.replace(tmp, out)
    return out


def shard_exists(out_dir: str, task: Dict[str, Any], tag: str = "") -> Optional[str]:
    pq = shard_path(out_dir, task, tag)
    for cand in (pq, os.path.splitext(pq)[0] + ".csv"):
        if os.path.exists(cand) and os.path.getsize(cand) > 0:
            return cand
    return None


def merge_shards(out_dir: str, tag: str = ""):
    """Concatenate every per-image shard into one table."""
    import glob as _glob

    import pandas as pd

    d = shard_dir(out_dir, tag)
    files = sorted(_glob.glob(os.path.join(d, "*.parquet"))
                   + _glob.glob(os.path.join(d, "*.csv")))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] unreadable shard {f}: {exc}", flush=True)
    if not frames:
        return pd.DataFrame(), files
    return pd.concat(frames, ignore_index=True), files


# --------------------------------------------------------------------------- #
# PVBM fractal-rotation sensitivity check
# --------------------------------------------------------------------------- #
def fd_rotation_check(datasets: Sequence[str], n_images: int, rot_low: int,
                      rot_ref: int, out_csv: str, max_side_map: Dict[str, int],
                      seed: int = 0):
    """Spearman of FD at ``rot_low`` vs ``rot_ref`` PVBM rotations, per dataset.

    Gate A showed ``MultifractalVBMs`` dominates the PVBM runtime (25 rotations:
    5.5 s DRIVE ... 113 s HRF per mask), which is unaffordable for ~2e4 events x
    2 masks.  C1 therefore runs PVBM with a reduced rotation count; this check
    certifies that the reduction does not change the FD *ranking* of images
    (pre-registered acceptance: Spearman >= 0.95).
    """
    import pandas as pd
    from scipy.stats import spearmanr

    from ..bio.disc import locate_optic_disc

    rows = []
    for ds in datasets:
        tasks = make_tasks(ds, limit=0, second_observer=False, seed=seed)
        tasks = sorted(tasks, key=lambda t: t["image_id"])[:int(n_images)]
        lo_v, hi_v, t_lo, t_hi = [], [], 0.0, 0.0
        for t in tasks:
            img, m, f, _ = load_task(t, max_side_map.get(ds.upper(), 0))
            disc = locate_optic_disc(img, f, vessel_mask=m)
            t0 = time.perf_counter()
            b_lo = compute_B(m, f, disc, rot_low, False)
            t_lo += time.perf_counter() - t0
            t0 = time.perf_counter()
            b_hi = compute_B(m, f, disc, rot_ref, False)
            t_hi += time.perf_counter() - t0
            lo_v.append(_num(b_lo.get("FD_pvbm")))
            hi_v.append(_num(b_hi.get("FD_pvbm")))
        lo_a, hi_a = np.array(lo_v), np.array(hi_v)
        ok = np.isfinite(lo_a) & np.isfinite(hi_a)
        rho = float(spearmanr(lo_a[ok], hi_a[ok])[0]) if ok.sum() >= 3 else float("nan")
        n = int(ok.sum())
        rows.append(dict(dataset=ds.upper(), n=n, rot_low=rot_low, rot_ref=rot_ref,
                         spearman=rho, passes=bool(np.isfinite(rho) and rho >= 0.95),
                         mean_abs_diff=float(np.mean(np.abs(lo_a[ok] - hi_a[ok]))) if n else np.nan,
                         mean_fd_low=float(np.mean(lo_a[ok])) if n else np.nan,
                         mean_fd_ref=float(np.mean(hi_a[ok])) if n else np.nan,
                         s_per_mask_low=t_lo / max(len(tasks), 1),
                         s_per_mask_ref=t_hi / max(len(tasks), 1),
                         max_side=max_side_map.get(ds.upper(), 0)))
        print(f"[fd-check] {ds:10s} n={n} rho={rho:.4f} "
              f"{t_lo / max(len(tasks), 1):.1f}s vs {t_hi / max(len(tasks), 1):.1f}s per mask",
              flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# --------------------------------------------------------------------------- #
# task construction
# --------------------------------------------------------------------------- #
_DS_ALIAS = {"DRIVE": "drive", "CHASE_DB1": "chasedb1", "HRF": "hrf",
             "FIVES": "fives", "STARE": "stare"}


def make_tasks(name: str, limit: int = 0, second_observer: bool = True,
               fives_n: int = FIVES_SUBSET_N, seed: int = 0,
               fives_split: str = "test") -> List[Dict[str, Any]]:
    """Per-image task dicts for one dataset.

    ``fives_split`` selects which FIVES split the seeded subset is drawn from:

    ``"test"``   the original C1 behaviour -- 60 images of the FIVES *test*
                 split, which is what the Fig.2 / dose-response tables were
                 built from;
    ``"train"``  60 images of the *training* split.  Needed because a deployed
                 head must be fitted on training-split images only, and with a
                 test-only subset FIVES contributes no fitting events at all;
    ``"both"``   the union of the two seeded subsets.

    The per-split subsets are drawn independently with the same seed, so adding
    ``train`` never changes which ``test`` images were chosen -- the existing
    shards stay valid and resumable.
    """
    from ..data.datasets import load_dataset

    recs = load_dataset(_DS_ALIAS[name.upper()])
    subset = "all"
    if name.upper() == "FIVES":
        if fives_split not in ("test", "train", "both"):
            raise ValueError("fives_split must be test/train/both, got %r" % fives_split)
        wanted = ("test", "train") if fives_split == "both" else (fives_split,)
        picked, tags = [], []
        for sp in wanted:
            pool = sorted((r for r in recs if r["split"] == sp),
                          key=lambda r: r["image_id"])
            if not pool:
                continue
            # deterministic seeded subset, drawn per split (fixed seed)
            rng = np.random.default_rng(int(seed))
            idx = np.sort(rng.choice(len(pool), size=min(fives_n, len(pool)),
                                     replace=False))
            picked += [pool[int(i)] for i in idx]
            tags.append("%s_seed%d[:%d]" % (sp, seed, len(idx)))
        recs = picked
        subset = "+".join(tags)

    tasks: List[Dict[str, Any]] = []
    for r in sorted(recs, key=lambda x: x["image_id"]):
        tasks.append(dict(
            dataset=name.upper(), image_id=r["image_id"], subject_id=r.get("subject_id", ""),
            split=r.get("split", ""), disease=r.get("disease", ""), subset=subset,
            image_path=r["image_path"], label_path=r["label_path"],
            fov_path=r.get("fov_path"), observer="obs1",
        ))
        # DRIVE ships a 2nd observer on its test split: extra, flagged rows
        if (second_observer and name.upper() == "DRIVE"
                and r.get("label2_path") and r.get("split") == "test"):
            tasks.append(dict(
                dataset=name.upper(), image_id=r["image_id"], subject_id=r.get("subject_id", ""),
                split=r.get("split", ""), disease=r.get("disease", ""), subset=subset,
                image_path=r["image_path"], label_path=r["label2_path"],
                fov_path=r.get("fov_path"), observer="obs2",
            ))
    if limit:
        keep = {t["image_id"] for t in tasks if t["observer"] == "obs1"}
        keep = set(sorted(keep)[:limit])
        tasks = [t for t in tasks if t["image_id"] in keep]
    return tasks


def _resize(img, shape, order: int):
    from skimage.transform import resize

    return resize(img, shape, order=order, preserve_range=True,
                  anti_aliasing=(order > 0)).astype(np.float32 if order > 0 else img.dtype)


def load_task(task: Dict[str, Any], max_side: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(image, mask, fov, scale)`` at the working resolution."""
    from ..data.datasets import read_image, read_binary

    img = read_image(task["image_path"])
    m = read_binary(task["label_path"])
    f = read_binary(task["fov_path"]) if task.get("fov_path") else np.ones(m.shape, bool)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3]

    scale = 1.0
    if max_side and max(m.shape) > max_side:
        scale = float(max_side) / float(max(m.shape))
        new = (int(round(m.shape[0] * scale)), int(round(m.shape[1] * scale)))
        img = _resize(img.astype(np.float32), new + (3,), order=1)
        m = _resize(m.astype(np.uint8), new, order=0).astype(bool)
        f = _resize(f.astype(np.uint8), new, order=0).astype(bool)
    return np.asarray(img), np.asarray(m, bool), np.asarray(f, bool), scale


# --------------------------------------------------------------------------- #
# the per-image worker
# --------------------------------------------------------------------------- #
def process_image(task: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run the full event budget for one (image, observer).  Picklable."""
    from ..bio.disc import locate_optic_disc
    from . import perturb as P

    t_start = time.perf_counter()
    key = f"{task['dataset']}:{task['image_id']}:{task['observer']}"
    rows: List[Dict[str, Any]] = []
    fails: Dict[str, int] = {}
    try:
        img, mask, fov, scale = load_task(task, cfg["max_side"])
        t_disc = time.perf_counter()
        disc = locate_optic_disc(img, fov, vessel_mask=mask)
        t_disc = time.perf_counter() - t_disc

        t_ctx = time.perf_counter()
        ctx = P.GTContext(mask, fov, img, disc, min_branch_px=cfg["min_branch_px"], key=key)
        inv_ref = P.graph_invariants(ctx.mask, ctx.fov, cfg["min_branch_px"])
        t_ctx = time.perf_counter() - t_ctx

        t_b = time.perf_counter()
        B0 = compute_B(ctx.mask, fov, disc, cfg["fd_rotations"], cfg["zone_b"])
        t_b0 = time.perf_counter() - t_b
        bkeys = biomarker_keys(B0)

        met = CachedMetrics(ctx.mask, fov, probe_px=cfg["probe_px"],
                            min_branch_px=cfg["min_branch_px"])

        rng = P.make_rng(cfg["seed"], key)
        scale_ev = float(cfg.get("event_scale", 1.0))
        sev_mode = str(cfg.get("severity_mode", "full"))
        n_sev = max(1, int(round(cfg["n_sever"] * scale_ev)))
        n_bri = max(1, int(round(cfg["n_bridge"] * scale_ev)))
        n_tru = max(1, int(round(cfg["n_truncate"] * scale_ev)))
        n_cal = max(1, int(round(cfg["n_caliber"] * scale_ev)))

        base = dict(
            dataset=task["dataset"], image_id=task["image_id"],
            subject_id=task["subject_id"], observer=task["observer"],
            split=task["split"], disease=task.get("disease", ""),
            subset=task.get("subset", ""), work_scale=scale,
            img_h=int(mask.shape[0]), img_w=int(mask.shape[1]),
            disc_cx=float(disc.cx), disc_cy=float(disc.cy), disc_r=float(disc.r),
            disc_confident=int(bool(disc.confident)),
        )
        # BCS of the GT against itself: the reference the "BCS drop" conventional
        # predictor of Exp1 is measured from.
        base["m_bcs_gt"] = met.gt_reference()["m_bcs"]

        def emit(kind, severity, sev_value, locus, res, pa, pb, extra=None):
            Mp = res.mask
            t0 = time.perf_counter()
            Bp = compute_B(Mp, fov, disc, cfg["fd_rotations"], cfg["zone_b"])
            Mc, cdiag = P.matched_control(ctx, Mp, locus, kind, rng,
                                          max_tries=cfg["control_tries"], inv_ref=inv_ref)
            Bc = (compute_B(Mc, fov, disc, cfg["fd_rotations"], cfg["zone_b"])
                  if Mc is not None else None)
            t_bio = time.perf_counter() - t0

            row = dict(base)
            row.update(
                event_id=f"{key}:{kind}:{len(rows):05d}",
                locus_id=locus.locus_id or f"{key}:{kind}:{locus.row}_{locus.col}",
                type=kind, severity=severity, severity_value=float(sev_value),
                locus_row=int(locus.row), locus_col=int(locus.col),
                n_changed=int(res.n_changed), t_biomarker_s=t_bio,
            )
            row.update({f"v_{k}": v for k, v in res.info.items()})
            row.update(cdiag)
            if extra:
                row.update(extra)

            for k in bkeys:
                b0 = _num(B0.get(k))
                bp = _num(Bp.get(k))
                row[f"dB_{k}"] = bp - b0
                row[f"absdB_{k}"] = abs(bp - b0)
                if Bc is not None:
                    bc = _num(Bc.get(k))
                    row[f"cdB_{k}"] = bc - b0
                    row[f"cabsdB_{k}"] = abs(bc - b0)
                else:
                    row[f"cdB_{k}"] = float("nan")
                    row[f"cabsdB_{k}"] = float("nan")

            row.update(met.evaluate(Mp))
            ev = P._EventPoints(kind, pa, pb)
            row.update(P.phi(ev, img, Mp, disc=disc, fov=fov,
                             min_branch_px=cfg["min_branch_px"]))
            row.update(P.gt_strata(ctx, locus))
            rows.append(row)

        # ---- (a) severing -------------------------------------------------
        # "full"  : every locus is taken through the whole L ladder (paired
        #           within-locus dose-response, used on DRIVE / CHASE_DB1).
        # "cycle" : one severity per locus, cycling the ladder across loci, so
        #           the four levels stay balanced at a quarter of the cost
        #           (used on HRF / FIVES, see DATASET_BUDGET).
        for li, locus in enumerate(ctx.sample_loci(n_sev, rng)):
            ladder = (P.L_FACTORS if sev_mode == "full"
                      else (P.L_FACTORS[li % len(P.L_FACTORS)],))
            for lf in ladder:
                res = P.sever(ctx, locus, lf)
                if not res.ok:
                    fails[f"sever:{res.reason}"] = fails.get(f"sever:{res.reason}", 0) + 1
                    continue
                pa, pb = P.new_endpoints(ctx, res.mask, locus)
                emit("sever", f"L{lf:g}D", lf, locus, res, pa, pb)

        # ---- (b) bridging -------------------------------------------------
        for pair in P.bridge_candidates(ctx, n_bri, rng, port_stride=cfg["port_stride"]):
            res = P.bridge(ctx, pair)
            if not res.ok:
                fails[f"bridge:{res.reason}"] = fails.get(f"bridge:{res.reason}", 0) + 1
                continue
            mid = pair.mid()
            locus = P.Locus(
                locus_id=pair.pair_id, row=int(round(mid[0])), col=int(round(mid[1])),
                branch=pair.branch_a, branch_type=ctx.branch_type(pair.branch_a),
                r_loc=0.5 * (pair.r_a + pair.r_b), d_loc=pair.r_a + pair.r_b,
                tangent=pair.t_a, arc_pos=0.0,
                dist_junction=float(ctx.dist_junc[pair.a[0], pair.a[1]]),
                dist_endpoint=float(ctx.dist_end[pair.a[0], pair.a[1]]),
                dist_fov=float(ctx.dist_fov[pair.a[0], pair.a[1]]), radius_cv=float("nan"),
                radius_bin=pair.radius_bin, zone=pair.zone, branch_order=pair.branch_order,
                density=pair.density, density_bin=pair.density_bin,
                contrast=pair.contrast, contrast_bin=pair.contrast_bin,
                gt_branch_length=ctx.branch_length(pair.branch_a),
            )
            emit("bridge", "bezier", pair.d_euclid, locus, res, pair.a, pair.b,
                 extra=dict(bridge_d=pair.d_euclid, bridge_geodesic=pair.geodesic,
                            bridge_branch_a=pair.branch_a, bridge_branch_b=pair.branch_b))

        # ---- (c) terminal loss --------------------------------------------
        tb = ctx.terminal_branches()
        if tb:
            pick = rng.permutation(len(tb))[:n_tru]
            for bi in (int(tb[int(i)]) for i in pick):
                locus = ctx.locus_for_branch(bi)
                if locus is None:
                    continue
                for rho in P.TRUNCATE_RHOS:
                    res = P.truncate(ctx, bi, rho)
                    if not res.ok:
                        fails[f"truncate:{res.reason}"] = fails.get(f"truncate:{res.reason}", 0) + 1
                        continue
                    co = ctx.path_coords[bi]
                    pa, _ = P.new_endpoints(ctx, res.mask, locus, search_pad=max(
                        20.0, 4.0 * locus.d_loc))
                    if pa is None:
                        e = P._free_end_index(co, ctx.end_px)
                        pa = (int(co[e, 0]), int(co[e, 1])) if e is not None else None
                    emit("truncate", f"rho{int(rho * 100)}", rho, locus, res, pa, None)

        # ---- (d) caliber (topology-neutral arm) ---------------------------
        for locus in ctx.sample_loci(n_cal, rng):
            fac = -1 if locus.r_loc >= P.MIN_ERODIBLE_RADIUS else 1
            res = P.caliber(ctx, locus, fac)
            if not res.ok:
                fails[f"caliber:{res.reason}"] = fails.get(f"caliber:{res.reason}", 0) + 1
                continue
            emit("caliber", f"px{fac:+d}", float(fac), locus, res,
                 (locus.row, locus.col), None)

        meta = dict(
            key=key, dataset=task["dataset"], image_id=task["image_id"],
            observer=task["observer"], n_events=len(rows), n_pool=len(ctx.locus_pool()),
            n_ports=len(ctx.port_pool()), n_terminal=len(tb),
            t_total_s=time.perf_counter() - t_start, t_disc_s=t_disc, t_ctx_s=t_ctx,
            t_b0_s=t_b0, work_scale=scale, fails=fails, error="",
            severity_mode=sev_mode,
        )
        # B(M) cache: the GT biomarker values of this image.  stats_c1 falls
        # back to these for the per-dataset robust scale sigma_B when the Gate A
        # csv is not available.
        meta.update({f"B0_{k}": _num(B0.get(k)) for k in bkeys})
        shard = cfg.get("shard_path")
        if shard and rows:
            meta["shard"] = write_shard(rows, shard)
            return dict(rows=[], meta=meta)          # keep the rows off the pipe
        return dict(rows=rows, meta=meta)
    except Exception as exc:  # noqa: BLE001
        return dict(rows=rows, meta=dict(
            key=key, dataset=task["dataset"], image_id=task["image_id"],
            observer=task["observer"], n_events=len(rows),
            t_total_s=time.perf_counter() - t_start, fails=fails,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="C1 perturbation engine driver")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="max images per dataset (smoke test)")
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default=RESULTS_DIR)
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--max-side", type=int, default=-1,
                    help="working longest side; -1 = per-dataset default, 0 = native")
    ap.add_argument("--fd-rotations", type=int, default=FD_ROTATIONS,
                    help="PVBM MultifractalVBMs n_rotations; the default of "
                         f"{FD_ROTATIONS} reproduces PVBM's 25-rotation FD exactly "
                         "at ~half the cost (see FD_ROTATIONS)")
    ap.add_argument("--zone-b", action="store_true",
                    help="also compute PVBM zone-B biomarkers (roughly doubles the cost)")
    ap.add_argument("--n-sever", type=int, default=DEFAULT_N_SEVER)
    ap.add_argument("--n-bridge", type=int, default=DEFAULT_N_BRIDGE)
    ap.add_argument("--n-truncate", type=int, default=DEFAULT_N_TRUNCATE)
    ap.add_argument("--n-caliber", type=int, default=DEFAULT_N_CALIBER)
    ap.add_argument("--event-scale", type=float, default=1.0,
                    help="global multiplier on every per-image event budget")
    ap.add_argument("--event-scale-ds", nargs="*", default=[],
                    help="per-dataset overrides, e.g. HRF=0.7 FIVES=0.7 -- HRF and "
                         "FIVES cost ~4x more per event than DRIVE (8 s vs 2.1 s per "
                         "compute_B at 1536 px) and together carry 145 of the 233 "
                         "image units, so they set the wall-clock budget")
    ap.add_argument("--control-tries", type=int, default=50)
    ap.add_argument("--port-stride", type=int, default=1)
    ap.add_argument("--probe-px", type=int, default=15)
    ap.add_argument("--min-branch-px", type=int, default=3)
    ap.add_argument("--no-second-observer", action="store_true")
    ap.add_argument("--fives-n", type=int, default=FIVES_SUBSET_N)
    ap.add_argument("--fives-split", default="test", choices=("test", "train", "both"),
                    help="which FIVES split the seeded subset comes from. 'test' is "
                         "the original C1 behaviour; 'train' adds training-split "
                         "images, which a deployed (train-split-only) head needs "
                         "because a test-only subset yields no fitting events; "
                         "'both' runs the union. Subsets are drawn per split with "
                         "the same seed, so existing shards stay valid.")
    ap.add_argument("--ds-budget", nargs="*", default=[],
                    help="per-dataset budget override "
                         "DS=n_sever,n_bridge,n_truncate,n_caliber[,mode]; "
                         "defaults come from DATASET_BUDGET")
    ap.add_argument("--no-resume", action="store_true",
                    help="recompute images whose per-image shard already exists")
    ap.add_argument("--merge-only", action="store_true",
                    help="skip computation, just merge the existing shards")
    ap.add_argument("--fd-check", type=int, default=0, metavar="N",
                    help="validate FD at --fd-rotations against --fd-check-ref on "
                         "N ground-truth masks per dataset, then exit")
    ap.add_argument("--fd-check-ref", type=int, default=25)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""

    max_side_map = {ds.upper(): (args.max_side if args.max_side >= 0
                                 else DEFAULT_MAX_SIDE.get(ds.upper(), 0))
                    for ds in args.datasets}

    if args.fd_check:
        fd_rotation_check(args.datasets, args.fd_check, args.fd_rotations,
                          args.fd_check_ref,
                          os.path.join(args.out_dir, f"c1_fd_rotation_check{tag}.csv"),
                          max_side_map, seed=args.seed)
        return 0

    if args.merge_only:
        return _finish(args, None, 0.0, [])

    ds_budget: Dict[str, tuple] = {}
    for spec in args.ds_budget:
        k, _, v = str(spec).partition("=")
        parts = v.split(",")
        nums = [int(float(x)) for x in parts[:4]]
        mode = parts[4] if len(parts) > 4 else "full"
        ds_budget[k.strip().upper()] = tuple(nums) + (mode,)

    ds_scale: Dict[str, float] = {}
    for spec in args.event_scale_ds:
        k, _, v = str(spec).partition("=")
        ds_scale[k.strip().upper()] = float(v)

    tasks: List[Dict[str, Any]] = []
    for ds in args.datasets:
        ds_tasks = make_tasks(ds, limit=args.limit,
                              second_observer=not args.no_second_observer,
                              fives_n=args.fives_n, seed=args.seed,
                              fives_split=args.fives_split)
        ms = max_side_map[ds.upper()]
        es = args.event_scale * float(ds_scale.get(ds.upper(), 1.0))
        bud = ds_budget.get(ds.upper(), DATASET_BUDGET.get(ds.upper()))
        if bud is None:
            bud = (args.n_sever, args.n_bridge, args.n_truncate, args.n_caliber, "full")
        for t in ds_tasks:
            t["_max_side"] = ms
            t["_event_scale"] = es
            t["_budget"] = bud
        n_skip = 0
        if not args.no_resume:
            keep = []
            for t in ds_tasks:
                if shard_exists(args.out_dir, t, args.tag):
                    n_skip += 1
                else:
                    keep.append(t)
            ds_tasks = keep
        tasks.extend(ds_tasks)
        print(f"[tasks] {ds:10s} todo={len(ds_tasks):4d} done={n_skip:4d} "
              f"max_side={ms} event_scale={es:g} "
              f"budget=sever{bud[0]}/bridge{bud[1]}/trunc{bud[2]}/cal{bud[3]}:{bud[4]}",
              flush=True)

    cfg_common = dict(
        seed=args.seed, fd_rotations=args.fd_rotations, zone_b=bool(args.zone_b),
        n_sever=args.n_sever, n_bridge=args.n_bridge, n_truncate=args.n_truncate,
        n_caliber=args.n_caliber, event_scale=args.event_scale,
        control_tries=args.control_tries, port_stride=args.port_stride,
        probe_px=args.probe_px, min_branch_px=args.min_branch_px,
    )
    jobs = []
    for t in tasks:
        bud = t.pop("_budget")
        jobs.append((t, dict(cfg_common,
                             max_side=t.pop("_max_side"),
                             event_scale=t.pop("_event_scale"),
                             n_sever=bud[0], n_bridge=bud[1],
                             n_truncate=bud[2], n_caliber=bud[3],
                             severity_mode=bud[4],
                             shard_path=shard_path(args.out_dir, t, args.tag))))

    t0 = time.perf_counter()
    all_rows: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    if args.workers <= 1:
        for i, (t, cfg) in enumerate(jobs):
            res = process_image(t, cfg)
            _report(res, i + 1, len(jobs))
            all_rows.extend(res["rows"])
            metas.append(res["meta"])
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx_mp = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx_mp) as ex:
            futs = {ex.submit(process_image, t, cfg): t for t, cfg in jobs}
            for i, fu in enumerate(as_completed(futs)):
                res = fu.result()
                _report(res, i + 1, len(jobs))
                all_rows.extend(res["rows"])
                metas.append(res["meta"])
    wall = time.perf_counter() - t0
    return _finish(args, all_rows, wall, metas, n_jobs=len(jobs))


def _finish(args, all_rows, wall: float, metas, n_jobs: int = 0) -> int:
    """Merge the per-image shards (plus anything held in memory) and report."""
    import pandas as pd

    tag = f"_{args.tag}" if args.tag else ""
    df, files = merge_shards(args.out_dir, args.tag)
    if all_rows:
        extra = pd.DataFrame(all_rows)
        df = pd.concat([df, extra], ignore_index=True) if len(df) else extra
    if len(df) and {"event_id"} <= set(df.columns):
        df = df.drop_duplicates(subset="event_id", keep="last").reset_index(drop=True)
    print(f"[merge] {len(files)} shards -> {len(df)} events", flush=True)

    pq = os.path.join(args.out_dir, f"c1_events{tag}.parquet")
    csv = os.path.join(args.out_dir, f"c1_events{tag}.csv")
    if len(df):
        try:
            df.to_parquet(pq, index=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] parquet write failed ({exc}); csv only", flush=True)
            pq = ""
        df.to_csv(csv, index=False)

    meta_csv = os.path.join(args.out_dir, f"c1_images{tag}.csv")
    meta_df = pd.DataFrame(metas or [])
    if os.path.exists(meta_csv) and len(meta_df):
        try:
            prev = pd.read_csv(meta_csv)
            meta_df = pd.concat([prev, meta_df], ignore_index=True)
            meta_df = meta_df.drop_duplicates(subset="key", keep="last")
        except Exception:  # noqa: BLE001
            pass
    elif os.path.exists(meta_csv) and not len(meta_df):
        meta_df = pd.read_csv(meta_csv)
    if len(meta_df):
        meta_df.to_csv(meta_csv, index=False)

    summary = build_summary(df, meta_df, wall, vars(args))
    summary["n_shards"] = len(files)
    js = os.path.join(args.out_dir, f"c1_summary{tag}.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)

    print("\n" + "=" * 72)
    print(json.dumps(summary, indent=2, default=str)[:6000])
    print("=" * 72)
    print(f"[out] {pq}\n[out] {csv}\n[out] {meta_csv}\n[out] {js}")
    if n_jobs:
        print(f"[time] wall {wall / 60:.1f} min for {n_jobs} images "
              f"({wall / max(n_jobs, 1):.1f} s/image, {args.workers} workers)")
    return 0


def _report(res, i, n):
    m = res["meta"]
    err = (" ERROR " + m["error"].splitlines()[0]) if m.get("error") else ""
    print(f"[{i:4d}/{n}] {m['key']:36s} events={m['n_events']:5d} "
          f"t={m['t_total_s']:7.1f}s{err}", flush=True)


def build_summary(df, meta_df, wall: float, args: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "wall_s": wall, "n_images": int(len(meta_df)), "n_events": int(len(df)),
        "args": {k: v for k, v in args.items() if not k.startswith("_")},
    }
    if len(meta_df):
        out["errors"] = int((meta_df["error"].astype(str).str.len() > 0).sum())
        out["per_image_s"] = {
            "mean": float(meta_df["t_total_s"].mean()),
            "median": float(meta_df["t_total_s"].median()),
            "max": float(meta_df["t_total_s"].max()),
        }
        by_ds = meta_df.groupby("dataset")["t_total_s"].agg(["count", "mean", "median", "max"])
        out["per_dataset_s"] = json.loads(by_ds.to_json(orient="index"))
        # verification failures, aggregated over images
        fails: Dict[str, int] = {}
        for d in meta_df.get("fails", []):
            if isinstance(d, dict):
                for k, v in d.items():
                    fails[k] = fails.get(k, 0) + int(v)
        out["verification_failures"] = dict(sorted(fails.items(), key=lambda kv: -kv[1]))
    if len(df):
        out["events_by_type"] = {str(k): int(v) for k, v in df["type"].value_counts().items()}
        out["events_by_dataset"] = {str(k): int(v) for k, v in df["dataset"].value_counts().items()}
        out["events_by_type_severity"] = {
            f"{a}|{b}": int(v) for (a, b), v in
            df.groupby(["type", "severity"]).size().items()
        }
        attempted = int(len(df)) + int(sum(out.get("verification_failures", {}).values()))
        out["verification_failure_rate"] = (
            float(sum(out.get("verification_failures", {}).values())) / max(attempted, 1))
        ok = df["control_ok"].astype(float)
        out["control_failure_rate_overall"] = float(1.0 - ok.mean())
        out["control_failure_rate_by_type"] = {
            str(k): float(1.0 - v) for k, v in df.groupby("type")["control_ok"].mean().items()
        }
        out["control_match_tier_counts"] = {
            str(k): int(v) for k, v in df["control_match_tier"].value_counts(dropna=False).items()
        }
        if "control_fail_reason" in df.columns:
            fr = df.loc[df["control_ok"] < 0.5, "control_fail_reason"].value_counts()
            out["control_fail_reasons"] = {str(k): int(v) for k, v in fr.items()}
        out["n_biomarker_delta_cols"] = int(sum(c.startswith("dB_") for c in df.columns))
        out["n_phi_cols"] = int(sum(c.startswith("phi_") for c in df.columns))
    return out


if __name__ == "__main__":
    sys.exit(main())
