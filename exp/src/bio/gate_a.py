"""Gate A: two-pipeline biomarker agreement on ground-truth masks (CPU).

Plan reference: exp/EXPERIMENT_PLAN.md S1.4 -- "run both pipelines on ALL GT
masks of the 4 datasets, report the Spearman correlation of each biomarker
between the two pipelines (>= 0.8 counts as dimensionally consistent), record
the per-image runtime.  Deliverable:
``results/gateA_pipeline_agreement.csv``."

Outputs
-------
``exp/results/gateA_biomarkers_gt.csv``
    one row per (dataset, image, observer) with every raw and primary
    biomarker, the disc estimate and the per-pipeline runtime.
``exp/results/gateA_pipeline_agreement.csv``
    one row per (dataset, primary biomarker) with Spearman rho, Pearson r,
    their p-values, n, and the mean per-image runtime of each pipeline.
``exp/runs/gateA/disc_check/<dataset>_<image>.png``
    4 optic-disc overlays per dataset for the visual check required by S1.3.

Usage
-----
    cd exp
    python -m src.bio.gate_a                 # all four datasets, 8 workers
    python -m src.bio.gate_a --probe         # print what load_dataset returns
    python -m src.bio.gate_a --datasets DRIVE --limit 4 --workers 2

Note on FIVES: the plan uses the **100-image test subset** of FIVES here for
speed (800 images x ~4 s would dominate Gate A); this is recorded in the
``subset`` column of both CSVs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Keep BLAS single-threaded: we parallelise over images, not inside them.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

DATASETS = ("DRIVE", "CHASE_DB1", "HRF", "FIVES")
FIVES_SUBSET_N = 100
N_DISC_OVERLAYS = 4

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
DISC_CHECK_DIR = os.path.join(EXP_ROOT, "runs", "gateA", "disc_check")


# --------------------------------------------------------------------------- #
# loader adaptation
# --------------------------------------------------------------------------- #
_IMAGE_KEYS = ("image", "image_path", "img", "img_path", "rgb", "image_file", "path", "file")
_FOV_KEYS = ("fov", "fov_path", "fov_mask", "fov_mask_path", "roi", "roi_mask", "mask_fov")
_GT_SINGLE_KEYS = (
    "gt", "gt_path", "label", "label_path", "vessel", "vessel_mask", "vessel_path",
    "manual1", "mask1", "gt1", "observer1", "ah", "seg", "segmentation",
)
_GT_MULTI_KEYS = ("gts", "gt_paths", "labels", "label_paths", "masks", "observers", "annotations")
_GT2_KEYS = (
    "label2_path", "gt2", "gt_path2", "label2", "manual2", "mask2", "observer2",
    "vk", "gt_2", "second_observer",
)
_SUBJECT_KEYS = ("subject_id", "subject", "patient_id", "eye_id")
_ID_KEYS = ("image_id", "id", "name", "stem", "key", "image_name", "basename")
_SPLIT_KEYS = ("split", "subset", "partition", "fold")
_SCALE_KEYS = ("px_per_mm", "pixels_per_mm", "scale_px_per_mm")


def _as_mapping(rec: Any) -> Dict[str, Any]:
    if isinstance(rec, dict):
        return rec
    if hasattr(rec, "_asdict"):
        return dict(rec._asdict())
    if hasattr(rec, "__dict__") and vars(rec):
        return dict(vars(rec))
    if hasattr(rec, "keys"):
        return {k: rec[k] for k in rec.keys()}
    raise TypeError(f"cannot interpret dataset record of type {type(rec)!r}")


def _records_of(ds: Any) -> List[Dict[str, Any]]:
    """Normalise whatever ``load_dataset`` returns into a list of dicts."""
    try:
        import pandas as pd

        if isinstance(ds, pd.DataFrame):
            return ds.to_dict("records")
    except ImportError:
        pass
    if isinstance(ds, dict):
        for k in ("records", "items", "images", "samples", "data", "rows", "all"):
            if k in ds:
                return _records_of(ds[k])
        # a dict of splits -> records
        vals = list(ds.values())
        if vals and all(isinstance(v, (list, tuple)) for v in vals):
            out: List[Dict[str, Any]] = []
            for split, v in ds.items():
                for r in _records_of(list(v)):
                    r.setdefault("split", split)
                    out.append(r)
            return out
        return [ds]
    if hasattr(ds, "records"):
        return _records_of(ds.records)
    if hasattr(ds, "items") and callable(getattr(ds, "items")):
        return _records_of(dict(ds))
    if isinstance(ds, (list, tuple)):
        return [_as_mapping(r) for r in ds]
    if hasattr(ds, "__iter__"):
        return [_as_mapping(r) for r in ds]
    raise TypeError(f"cannot interpret load_dataset() return of type {type(ds)!r}")


def _pick(rec: Dict[str, Any], keys: Sequence[str]):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    lower = {str(k).lower(): k for k in rec}
    for k in keys:
        if k in lower and rec[lower[k]] is not None:
            return rec[lower[k]]
    return None


def _observer_entries(rec: Dict[str, Any]) -> List[Any]:
    """Return the list of GT masks (one per observer) held by a record."""
    multi = _pick(rec, _GT_MULTI_KEYS)
    if multi is not None:
        if isinstance(multi, dict):
            return [multi[k] for k in sorted(multi)]
        if isinstance(multi, (list, tuple)):
            return list(multi)
        return [multi]
    obs = []
    first = _pick(rec, _GT_SINGLE_KEYS)
    if first is not None:
        obs.append(first)
    second = _pick(rec, _GT2_KEYS)
    if second is not None:
        obs.append(second)
    return obs


def _observer_names(rec: Dict[str, Any], n: int) -> List[str]:
    multi = _pick(rec, _GT_MULTI_KEYS)
    if isinstance(multi, dict):
        return [str(k) for k in sorted(multi)]
    return [f"obs{i + 1}" for i in range(n)]


def make_tasks(ds: Any, name: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Flatten a loaded dataset into picklable per-image task dicts."""
    recs = _records_of(ds)
    tasks: List[Dict[str, Any]] = []
    for i, rec in enumerate(recs):
        img = _pick(rec, _IMAGE_KEYS)
        obs = _observer_entries(rec)
        if img is None or not obs:
            continue
        iid = _pick(rec, _ID_KEYS)
        if iid is None:
            iid = os.path.splitext(os.path.basename(str(img)))[0] if isinstance(img, str) else f"img{i:04d}"
        tasks.append(
            dict(
                dataset=name,
                image_id=str(iid),
                image=img,
                fov=_pick(rec, _FOV_KEYS),
                observers=obs,
                observer_names=_observer_names(rec, len(obs)),
                split=str(_pick(rec, _SPLIT_KEYS) or ""),
                subject_id=str(_pick(rec, _SUBJECT_KEYS) or ""),
                disease=str(rec.get("disease", "") or ""),
                px_per_mm=_pick(rec, _SCALE_KEYS),
            )
        )

    subset = "all"
    if name.upper() == "FIVES" and (limit is None):
        test = [t for t in tasks if t["split"].lower() in ("test", "testing", "val", "validation")]
        pool = test if test else tasks
        pool = sorted(pool, key=lambda t: t["image_id"])[:FIVES_SUBSET_N]
        tasks = pool
        subset = f"test[:{FIVES_SUBSET_N}]" if test else f"first{FIVES_SUBSET_N}"
    if limit:
        tasks = sorted(tasks, key=lambda t: t["image_id"])[:limit]
        subset = f"limit{limit}"
    for t in tasks:
        t["subset"] = subset
    return tasks


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def _load_array(x) -> np.ndarray:
    """Read a path with the *dataset module's* reader so decoding matches S0.

    ``src.data.datasets.read_image`` uses imageio.v3, which handles every format
    in play here (DRIVE .gif labels, STARE .ppm, HRF .tif, FIVES .png);
    ``skimage.io.imread`` is only the fallback.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (str, os.PathLike)):
        try:
            from src.data.datasets import read_image

            return np.asarray(read_image(str(x)))
        except Exception:
            from skimage.io import imread

            return np.asarray(imread(str(x)))
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _squeeze_frames(a: np.ndarray) -> np.ndarray:
    """Drop a leading singleton frame axis (some readers return (1, H, W) for GIF)."""
    a = np.asarray(a)
    while a.ndim == 3 and a.shape[0] == 1 and a.shape[-1] not in (3, 4):
        a = a[0]
    while a.ndim == 4 and a.shape[0] == 1:
        a = a[0]
    return a


def _to_bool(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == bool:
        return a
    mx = float(a.max()) if a.size else 0.0
    return a > (0.5 * mx if mx > 1 else 0.5)


def _auto_fov(image: np.ndarray) -> np.ndarray:
    """Synthesise a FOV mask when the dataset does not ship one."""
    from scipy import ndimage as ndi

    img = np.asarray(image)
    if img.ndim == 3:
        g = img[..., :3].astype(np.float32).mean(axis=2)
    else:
        g = img.astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    fov = g > 0.05
    fov = ndi.binary_fill_holes(fov)
    lab, n = ndi.label(fov)
    if n > 1:
        sizes = ndi.sum(fov, lab, range(1, n + 1))
        fov = lab == (1 + int(np.argmax(sizes)))
    fov = ndi.binary_erosion(fov, structure=np.ones((3, 3), bool), iterations=2)
    if not fov.any():
        fov = np.ones(g.shape, bool)
    return fov


# --------------------------------------------------------------------------- #
# worker
# --------------------------------------------------------------------------- #
def process_image(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Measure all observers of one image with both pipelines."""
    from .biomarkers import compute_all
    from .disc import locate_optic_disc, save_disc_overlay

    rows: List[Dict[str, Any]] = []
    try:
        image = _squeeze_frames(_load_array(task["image"]))
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        image = image[..., :3]

        fov_path = task.get("fov")
        # A cached auto-FOV can legitimately be absent (datasets.py regenerates
        # exp/data/<name>/fov/ on demand, and a concurrent rebuild empties it),
        # so a missing file must degrade to our own FOV rather than kill the row.
        if isinstance(fov_path, (str, os.PathLike)) and not os.path.exists(str(fov_path)):
            fov_path = None
        if fov_path is not None:
            fov = _to_bool(_squeeze_frames(_load_array(fov_path)))
            fov_source = "dataset"
        else:
            fov = _auto_fov(image)
            fov_source = "auto" if task.get("fov") is None else "auto_missing_cache"
        if fov.shape != image.shape[:2]:
            raise ValueError(f"fov {fov.shape} != image {image.shape[:2]}")

        obs_masks = []
        for o in task["observers"]:
            m = _to_bool(_squeeze_frames(_load_array(o)))
            if m.shape != image.shape[:2]:
                raise ValueError(f"gt {m.shape} != image {image.shape[:2]}")
            obs_masks.append(m)

        t0 = time.perf_counter()
        disc = locate_optic_disc(image, fov, vessel_mask=obs_masks[0])
        t_disc = time.perf_counter() - t0

        if task.get("save_overlay"):
            os.makedirs(DISC_CHECK_DIR, exist_ok=True)
            save_disc_overlay(
                image,
                disc,
                os.path.join(
                    DISC_CHECK_DIR, f"{task['dataset']}_{task['image_id']}.png"
                ),
                fov=fov,
                vessel_mask=obs_masks[0],
            )

        for name, m in zip(task["observer_names"], obs_masks):
            row: Dict[str, Any] = dict(
                dataset=task["dataset"],
                image_id=task["image_id"],
                observer=name,
                subject_id=task.get("subject_id", ""),
                disease=task.get("disease", ""),
                split=task.get("split", ""),
                subset=task.get("subset", ""),
                height=int(image.shape[0]),
                width=int(image.shape[1]),
                fov_source=fov_source,
                runtime_disc_s=t_disc,
                error="",
            )
            res = compute_all(
                m,
                fov,
                image=None,
                disc=disc,
                px_per_mm=task.get("px_per_mm"),
                fd_rotations=task.get("fd_rotations", 25),
            )
            row.update(res)
            rows.append(row)
    except Exception as exc:  # noqa: BLE001
        rows.append(
            dict(
                dataset=task.get("dataset", "?"),
                image_id=task.get("image_id", "?"),
                observer="",
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=6),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# agreement table
# --------------------------------------------------------------------------- #
#: extra (label, pvbm column, skan column) pairs recorded with
#: ``zone == "diagnostic"``.  They are not gate criteria; they exist so that a
#: primary biomarker falling below Spearman 0.8 can be diagnosed immediately
#: (e.g. is it the FD *variant*, the tortuosity *summary statistic*, or the
#: length *definition* that disagrees?).
DIAGNOSTIC_PAIRS = (
    ("FD_D0_vs_maskbox", "pvbm_fractal_D0", "skan_fractal_dimension_mask"),
    ("FD_D1_vs_box", "pvbm_fractal_D1", "skan_fractal_dimension"),
    ("FD_D2_vs_box", "pvbm_fractal_D2", "skan_fractal_dimension"),
    ("tort_median_vs_mean", "pvbm_median_tortuosity", "skan_tortuosity_mean"),
    ("tort_index_vs_lw", "pvbm_tortuosity_index", "skan_tortuosity_length_weighted"),
    ("length_vs_skelpx", "pvbm_total_length", "skan_skeleton_px"),
    ("area_vs_area", "pvbm_vessel_area_px", "skan_vessel_area_px"),
    ("endpoints", "pvbm_n_endpoints", "skan_n_endpoints"),
    ("junctions", "pvbm_n_intersections", "skan_n_junctions"),
)


def _corr(g, ca, cb):
    """(n, spearman, p, pearson, p, mean_a, mean_b) for two columns of ``g``."""
    import pandas as pd
    from scipy import stats

    if ca not in g.columns or cb not in g.columns:
        return None
    x = pd.to_numeric(g[ca], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(g[cb], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    rho = p_rho = r = p_r = np.nan
    if n >= 3 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
        rho, p_rho = stats.spearmanr(x[ok], y[ok])
        r, p_r = stats.pearsonr(x[ok], y[ok])
    return (
        n,
        float(rho),
        float(p_rho),
        float(r),
        float(p_r),
        float(np.nanmean(x)) if n else np.nan,
        float(np.nanmean(y)) if n else np.nan,
    )


def agreement_table(df, runtimes: Dict[str, Dict[str, float]]):
    """Per dataset x primary biomarker Spearman / Pearson between pipelines."""
    import pandas as pd
    from scipy import stats

    from .biomarkers import PRIMARY_BIOMARKERS

    rows = []
    groups = [(ds, g) for ds, g in df.groupby("dataset", sort=True)]
    groups.append(("ALL", df))
    for ds, g in groups:
        rt = runtimes.get(ds, {})
        for zone in (False, True):
            suf = "_zoneB" if zone else ""
            for b in PRIMARY_BIOMARKERS:
                ca, cb = f"{b}_pvbm{suf}", f"{b}_skan{suf}"
                if ca not in g.columns or cb not in g.columns:
                    continue
                x = pd.to_numeric(g[ca], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(g[cb], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                n = int(ok.sum())
                rho = p_rho = r = p_r = np.nan
                if n >= 3 and np.ptp(x[ok]) > 0 and np.ptp(y[ok]) > 0:
                    rho, p_rho = stats.spearmanr(x[ok], y[ok])
                    r, p_r = stats.pearsonr(x[ok], y[ok])
                rows.append(
                    dict(
                        dataset=ds,
                        zone="zoneB" if zone else "global",
                        biomarker=b,
                        n=n,
                        n_missing=int(len(g) - n),
                        spearman=float(rho),
                        spearman_p=float(p_rho),
                        pearson=float(r),
                        pearson_p=float(p_r),
                        pvbm_mean=float(np.nanmean(x)) if n else np.nan,
                        skan_mean=float(np.nanmean(y)) if n else np.nan,
                        runtime_pvbm_s=rt.get("pvbm", np.nan),
                        runtime_skan_s=rt.get("skan", np.nan),
                        runtime_disc_s=rt.get("disc", np.nan),
                        subset=rt.get("subset", ""),
                        passes_gate=bool(np.isfinite(rho) and rho >= 0.8),
                    )
                )
        for label, ca, cb in DIAGNOSTIC_PAIRS:
            c = _corr(g, ca, cb)
            if c is None:
                continue
            n, rho, p_rho, r, p_r, ma, mb = c
            rows.append(
                dict(
                    dataset=ds,
                    zone="diagnostic",
                    biomarker=label,
                    n=n,
                    n_missing=int(len(g) - n),
                    spearman=rho,
                    spearman_p=p_rho,
                    pearson=r,
                    pearson_p=p_r,
                    pvbm_mean=ma,
                    skan_mean=mb,
                    runtime_pvbm_s=rt.get("pvbm", np.nan),
                    runtime_skan_s=rt.get("skan", np.nan),
                    runtime_disc_s=rt.get("disc", np.nan),
                    subset=rt.get("subset", ""),
                    passes_gate=False,
                )
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _probe(names: Sequence[str]) -> None:
    from src.data.datasets import load_dataset

    for name in names:
        print(f"\n=== {name} ===")
        try:
            ds = load_dataset(name)
        except Exception as exc:
            print(f"  load_dataset failed: {type(exc).__name__}: {exc}")
            continue
        print(f"  type: {type(ds)}")
        try:
            recs = _records_of(ds)
        except Exception as exc:
            print(f"  _records_of failed: {exc}")
            continue
        print(f"  n_records: {len(recs)}")
        if recs:
            r0 = recs[0]
            print(f"  record type: {type(r0)}")
            for k, v in r0.items():
                s = str(v)
                print(f"    {k!r:22s} {type(v).__name__:12s} {s[:90]}")
        tasks = make_tasks(ds, name)
        print(f"  -> {len(tasks)} tasks, subset={tasks[0]['subset'] if tasks else '-'}")
        if tasks:
            t = tasks[0]
            print(f"     image={str(t['image'])[:80]}")
            print(f"     fov={str(t['fov'])[:80]}")
            print(f"     observers={[str(o)[:60] for o in t['observers']]} names={t['observer_names']}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Gate A: biomarker pipeline agreement")
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="max images per dataset (debug)")
    ap.add_argument("--fd-rotations", type=int, default=25, help="PVBM MultifractalVBMs n_rotations")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--overlays", type=int, default=N_DISC_OVERLAYS)
    ap.add_argument("--probe", action="store_true", help="print the loader contract and exit")
    args = ap.parse_args(argv)

    if args.probe:
        _probe(args.datasets)
        return 0

    import pandas as pd

    from src.data.datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(DISC_CHECK_DIR, exist_ok=True)

    all_tasks: List[Dict[str, Any]] = []
    subsets: Dict[str, str] = {}
    for name in args.datasets:
        t0 = time.perf_counter()
        ds = load_dataset(name)
        tasks = make_tasks(ds, name, limit=args.limit or None)
        for i, t in enumerate(tasks):
            t["save_overlay"] = i < args.overlays
            t["fd_rotations"] = args.fd_rotations
        subsets[name] = tasks[0]["subset"] if tasks else "empty"
        all_tasks.extend(tasks)
        print(
            f"[load] {name:10s} {len(tasks):4d} images "
            f"({subsets[name]}) in {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    if not all_tasks:
        print("no tasks -- check load_dataset / --datasets", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    if args.workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {ex.submit(process_image, t): t for t in all_tasks}
            for i, fut in enumerate(as_completed(futs), 1):
                rows.extend(fut.result())
                if i % 10 == 0 or i == len(futs):
                    el = time.perf_counter() - t_start
                    print(
                        f"  {i}/{len(futs)} images  {el:.0f}s  "
                        f"({el / max(i, 1):.2f} s/img wall)",
                        flush=True,
                    )
    else:
        for i, t in enumerate(all_tasks, 1):
            rows.extend(process_image(t))
            print(f"  {i}/{len(all_tasks)}", flush=True)

    df = pd.DataFrame(rows)
    bio_csv = os.path.join(args.out_dir, "gateA_biomarkers_gt.csv")
    df.to_csv(bio_csv, index=False)
    print(f"[write] {bio_csv}  ({len(df)} rows)")

    n_err = int((df.get("error", pd.Series([""] * len(df))).astype(str) != "").sum())
    if n_err:
        print(f"[warn] {n_err} rows failed; first errors:")
        for e in df.loc[df["error"].astype(str) != "", "error"].unique()[:5]:
            print("   ", e)

    ok = df[df.get("error", pd.Series([""] * len(df))).astype(str) == ""] if "error" in df else df
    runtimes: Dict[str, Dict[str, float]] = {}
    for ds, g in ok.groupby("dataset", sort=True):
        runtimes[ds] = dict(
            pvbm=float(pd.to_numeric(g.get("runtime_pvbm_s"), errors="coerce").mean()),
            skan=float(pd.to_numeric(g.get("runtime_skan_s"), errors="coerce").mean()),
            disc=float(pd.to_numeric(g.get("runtime_disc_s"), errors="coerce").mean()),
            subset=subsets.get(ds, ""),
        )
    runtimes["ALL"] = dict(
        pvbm=float(pd.to_numeric(ok.get("runtime_pvbm_s"), errors="coerce").mean()),
        skan=float(pd.to_numeric(ok.get("runtime_skan_s"), errors="coerce").mean()),
        disc=float(pd.to_numeric(ok.get("runtime_disc_s"), errors="coerce").mean()),
        subset="",
    )

    agr = agreement_table(ok, runtimes)
    agr_csv = os.path.join(args.out_dir, "gateA_pipeline_agreement.csv")
    agr.to_csv(agr_csv, index=False)
    print(f"[write] {agr_csv}")

    # robust scales sigma_B (proposal 3.1.2 step 1)
    try:
        from .biomarkers import standardise

        scales, _ = standardise(ok, "dataset", add_z=False)
        sc_csv = os.path.join(args.out_dir, "gateA_biomarker_scales.csv")
        scales.to_csv(sc_csv, index=False)
        print(f"[write] {sc_csv}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] standardise failed: {exc}")

    # ---- console summary ---------------------------------------------------
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print("\n=== Gate A: pipeline agreement (global) ===")
        show = agr[agr["zone"] == "global"][
            ["dataset", "biomarker", "n", "spearman", "pearson",
             "runtime_pvbm_s", "runtime_skan_s", "passes_gate"]
        ]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    if "disc_confident" in ok.columns:
        per_img = ok.drop_duplicates(["dataset", "image_id"])
        bad = per_img[pd.to_numeric(per_img["disc_confident"], errors="coerce") < 1]
        print(f"\ndisc low-confidence images: {len(bad)} / {len(per_img)}")
        if len(bad):
            print(bad.groupby("dataset").size().to_string())

    failed = agr[(agr["zone"] == "global") & (~agr["passes_gate"])]
    if len(failed):
        print("\n[GATE A] biomarkers below Spearman 0.8:")
        print(failed[["dataset", "biomarker", "n", "spearman"]].to_string(index=False))
    else:
        print("\n[GATE A] all primary biomarkers pass Spearman >= 0.8")
    print(f"\ntotal wall time {time.perf_counter() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
