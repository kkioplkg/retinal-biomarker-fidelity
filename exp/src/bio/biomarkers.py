"""Unified biomarker interface: ``B(mask, fov, disc) -> dict``.

Plan reference: exp/EXPERIMENT_PLAN.md code layout -- ``bio/biomarkers.py``
(unified interface) -- and proposal section 3.1.2, which fixes the four PRIMARY
biomarkers of the main text

    b in {FD, tortuosity, density, total branch length}

and requires each of them to be available from **two independent pipelines**
(section 4.1: "the four main biomarkers can be computed by both main pipelines,
so a two-pipeline consistency check is possible").

``compute_all`` returns

* every raw output of pipeline #1 prefixed ``pvbm_``,
* every raw output of pipeline #2 prefixed ``skan_``,
* the eight primary columns ``{FD, tortuosity, density, total_length}
  x {_pvbm, _skan}`` (plus their ``_zoneB`` variants when a disc is given),
* the per-pipeline wall-clock runtimes.

``standardise`` implements the first step of section 3.1.2: the per-dataset
robust scale ``sigma_B`` used to non-dimensionalise the harm
``H_b = |Delta B_b| / sigma_b``.  It is estimated from the **ground-truth**
biomarker values of that dataset (that is what Gate A produces) with the
pre-registered MAD estimator, ``sigma = 1.4826 * median(|x - median(x)|)``.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from .disc import DiscEstimate, locate_optic_disc
from .pvbm_pipe import compute_biomarkers_pvbm
from .skan_pipe import compute_biomarkers_skan

__all__ = [
    "compute_all",
    "standardise",
    "PRIMARY_BIOMARKERS",
    "PRIMARY_SOURCE",
    "primary_columns",
]

#: the four biomarkers the paper standardises on (proposal section 3.1.2)
PRIMARY_BIOMARKERS: Tuple[str, ...] = ("FD", "tortuosity", "density", "total_length")

#: primary name -> (key inside the pvbm dict, key inside the skan dict)
#:
#: * FD           : PVBM capacity dimension D0  vs our box-counting slope.
#: * tortuosity   : PVBM median arc/chord tortuosity vs our length-weighted
#:                  arc/chord tortuosity (both are arc/chord ratios; PVBM's
#:                  "tortuosity index" TI is a different, non-comparable
#:                  quantity and is kept only as a raw column).
#: * density      : vessel pixels / FOV pixels in both.
#: * total_length : PVBM "overall length" vs our sum of skan branch arc lengths.
PRIMARY_SOURCE: Dict[str, Tuple[str, str]] = {
    "FD": ("fractal_D0", "fractal_dimension"),
    "tortuosity": ("median_tortuosity", "tortuosity_length_weighted"),
    "density": ("vessel_density", "vessel_density"),
    "total_length": ("total_length", "skeleton_length_total"),
}


def primary_columns(zone: bool = False) -> list:
    """Column names of the primary biomarkers, e.g. ``FD_pvbm``, ``FD_skan``."""
    suf = "_zoneB" if zone else ""
    return [f"{b}_{p}{suf}" for b in PRIMARY_BIOMARKERS for p in ("pvbm", "skan")]


def _prefix(d: Dict[str, object], p: str) -> Dict[str, object]:
    return {f"{p}{k}": v for k, v in d.items()}


def compute_all(
    mask,
    fov,
    image=None,
    disc: Optional[DiscEstimate] = None,
    px_per_mm: Optional[float] = None,
    fd_rotations: int = 25,
    run_pvbm: bool = True,
    run_skan: bool = True,
) -> Dict[str, object]:
    """Run both biomarker pipelines on one mask.

    Parameters
    ----------
    mask, fov : (H, W) arrays
        Binary vessel mask and field of view.
    image : (H, W, 3) array, optional
        RGB fundus image.  Used only to locate the optic disc when ``disc`` is
        not supplied; without either, the zone-B variants are skipped.
    disc : DiscEstimate, optional
        Pre-computed optic disc (avoids re-detecting it per observer).
    px_per_mm : float, optional
        Physical scale, forwarded to both pipelines.
    fd_rotations : int
        ``n_rotations`` of PVBM's multifractal estimator.
    run_pvbm, run_skan : bool
        Allow running a single pipeline (used by the unit smoke tests).

    Returns
    -------
    dict
        ``pvbm_*`` + ``skan_*`` raw outputs, the primary columns, the disc
        fields and ``runtime_{disc,pvbm,skan}_s``.
    """
    out: Dict[str, object] = {}

    if disc is None and image is not None:
        t0 = time.perf_counter()
        disc = locate_optic_disc(image, fov, vessel_mask=mask)
        out["runtime_disc_s"] = time.perf_counter() - t0
    if disc is not None:
        out["disc_cx"] = float(disc.cx)
        out["disc_cy"] = float(disc.cy)
        out["disc_r"] = float(disc.r)
        out["disc_confident"] = int(bool(disc.confident))
        out["disc_brightness_z"] = float(disc.brightness_z)
        out["disc_density_ratio"] = float(disc.density_ratio)
        out["disc_radius_estimated"] = int(bool(disc.radius_estimated))
        out["disc_fov_diameter"] = float(disc.fov_diameter)

    pv: Dict[str, object] = {}
    sk: Dict[str, object] = {}

    if run_pvbm:
        t0 = time.perf_counter()
        try:
            pv = compute_biomarkers_pvbm(
                mask, fov, disc=disc, px_per_mm=px_per_mm, fd_rotations=fd_rotations
            )
        except Exception as exc:  # noqa: BLE001
            pv = {"pipeline_error": f"{type(exc).__name__}: {exc}"}
        out["runtime_pvbm_s"] = time.perf_counter() - t0
        out.update(_prefix(pv, "pvbm_"))

    if run_skan:
        t0 = time.perf_counter()
        try:
            sk = compute_biomarkers_skan(mask, fov, disc=disc, px_per_mm=px_per_mm)
        except Exception as exc:  # noqa: BLE001
            sk = {"pipeline_error": f"{type(exc).__name__}: {exc}"}
        out["runtime_skan_s"] = time.perf_counter() - t0
        out.update(_prefix(sk, "skan_"))

    # ---- the four primary biomarkers, from both pipelines ------------------
    for name, (pk, skk) in PRIMARY_SOURCE.items():
        out[f"{name}_pvbm"] = _num(pv.get(pk))
        out[f"{name}_skan"] = _num(sk.get(skk))
        out[f"{name}_pvbm_zoneB"] = _num(pv.get(f"zoneB_{pk}"))
        out[f"{name}_skan_zoneB"] = _num(sk.get(f"zoneB_{skk}"))

    return out


def _num(v) -> float:
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# --------------------------------------------------------------------------- #
# per-dataset robust scale (proposal section 3.1.2, step 1)
# --------------------------------------------------------------------------- #
def standardise(
    df,
    dataset_col: str = "dataset",
    columns: Optional[Iterable[str]] = None,
    add_z: bool = True,
):
    """Per-dataset MAD scale ``sigma_B`` of GT-derived biomarker values.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per (image, observer); must contain ``dataset_col`` and the
        biomarker columns.  In Gate A this is
        ``exp/results/gateA_biomarkers_gt.csv``.
    dataset_col : str
        Column holding the dataset name.
    columns : iterable of str, optional
        Columns to standardise.  Defaults to the eight primary columns present
        in ``df`` (global and zone-B).
    add_z : bool
        Also return a copy of ``df`` with ``<col>_z = (x - median) / sigma``.

    Returns
    -------
    (scales, df_z)
        ``scales`` is a tidy DataFrame with columns
        ``[dataset, biomarker, n, median, mad, sigma]`` where
        ``sigma = 1.4826 * mad`` (the pre-registered robust scale).
        ``df_z`` is ``None`` when ``add_z`` is False.
    """
    import pandas as pd

    if columns is None:
        cand = primary_columns(False) + primary_columns(True)
        columns = [c for c in cand if c in df.columns]
    columns = list(columns)

    rows = []
    for ds, g in df.groupby(dataset_col, sort=True):
        for c in columns:
            x = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                rows.append(
                    dict(dataset=ds, biomarker=c, n=0, median=np.nan, mad=np.nan, sigma=np.nan)
                )
                continue
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            sigma = 1.4826 * mad
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = float(np.std(x)) if x.size > 1 else np.nan
            rows.append(
                dict(dataset=ds, biomarker=c, n=int(x.size), median=med, mad=mad, sigma=sigma)
            )
    scales = pd.DataFrame(rows)

    df_z = None
    if add_z:
        df_z = df.copy()
        lut = {(r.dataset, r.biomarker): (r.median, r.sigma) for r in scales.itertuples()}
        for c in columns:
            vals = np.full(len(df_z), np.nan)
            x = pd.to_numeric(df_z[c], errors="coerce").to_numpy(dtype=float)
            for i, (ds, v) in enumerate(zip(df_z[dataset_col].to_numpy(), x)):
                med, sig = lut.get((ds, c), (np.nan, np.nan))
                if np.isfinite(sig) and sig > 0:
                    vals[i] = (v - med) / sig
            df_z[f"{c}_z"] = vals
    return scales, df_z
