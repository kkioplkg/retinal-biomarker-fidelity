"""Pipeline #2 (ours): skeleton / skan graph biomarkers.

Plan reference: exp/EXPERIMENT_PLAN.md S1.2 -- "independent second pipeline:
skeleton -> skan graph -> vessel density (foreground/FOV), total skeleton
length, box-counting FD (log slope), arc/chord tortuosity (length weighted),
branch count".

This module is deliberately **self-contained**: it depends only on numpy,
scipy, scikit-image and skan, and does not import ``src.topo`` -- Gate A must
be runnable while the topology module is still being written, and the two
pipelines of Gate A must not share code (a shared bug would be invisible to the
agreement check).

Conventions (frozen for the whole project)
------------------------------------------
* Foreground 8-connectivity, background 4-connectivity (plan header).  This is
  what ``skimage.morphology.skeletonize`` (Zhang-Suen) produces and what the
  3x3 neighbour count below assumes.
* Every measurement is restricted to the FOV: ``mask &= fov`` before anything.
* Skeleton *length* is the sum of skan branch arc lengths (diagonal steps count
  ``sqrt(2)``), not the raw pixel count; the pixel count is also reported as
  ``skeleton_px`` because PVBM-style pipelines use it.
* Tortuosity = arc / chord per branch; branches shorter than
  ``MIN_BRANCH_PX`` (10 px) are excluded, as are closed loops (chord ~ 0).
* Fractal dimension = least-squares slope of ``log N(s)`` vs ``log(1/s)`` over
  geometric box sizes ``2 .. min(H, W) / 4`` on the skeleton, computed on the
  bounding box of the FOV (or of the zone).  ``fractal_dimension_mask`` applies
  the same estimator to the filled mask; it exists only to diagnose FD
  disagreement with PVBM, whose D0 is a segmentation (not skeleton) box-count.
* Zone B = annulus ``[1.0, 1.5]`` disc *diameters* from the disc centre
  (see ``disc.zone_b_mask``); zone metrics reuse the *global* skeleton
  intersected with the annulus, so the annulus boundary does not change the
  skeletonisation.

Deterministic: no randomness, no thresholds derived from the data other than
the fixed constants above.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi

from .disc import DiscEstimate, zone_b_mask

__all__ = [
    "compute_biomarkers_skan",
    "skeletonize_mask",
    "box_counting_fd",
    "MIN_BRANCH_PX",
]

MIN_BRANCH_PX = 10.0  # branches shorter than this are excluded from tortuosity
FD_N_SIZES = 12  # number of geometric box sizes
FD_MIN_BOX = 2

_NEIGH_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

# metric names produced by ``_region_metrics`` (used for the zone-B prefixing)
_METRIC_KEYS = (
    "vessel_density",
    "vessel_area_px",
    "region_area_px",
    "skeleton_length_total",
    "skeleton_length_total_norm",
    "skeleton_px",
    "n_branches",
    "n_junctions",
    "n_endpoints",
    "fractal_dimension",
    "fractal_dimension_r2",
    "fractal_dimension_mask",
    "tortuosity_mean",
    "tortuosity_length_weighted",
    "tortuosity_n_branches_used",
    "mean_radius",
)


# --------------------------------------------------------------------------- #
# low level helpers
# --------------------------------------------------------------------------- #
def _as_bool(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == bool:
        return a
    mx = float(a.max()) if a.size else 0.0
    return a > (0.5 * mx if mx > 1 else 0.5)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """8-connected 2D skeleton (Zhang-Suen, ``skimage.morphology.skeletonize``)."""
    from skimage.morphology import skeletonize

    m = _as_bool(mask)
    if not m.any():
        return np.zeros_like(m, dtype=bool)
    return skeletonize(m).astype(bool)


def _neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Number of 8-neighbours of every skeleton pixel (0 outside the skeleton)."""
    c = ndi.convolve(skel.astype(np.uint8), _NEIGH_KERNEL, mode="constant", cval=0)
    return np.where(skel, c, 0).astype(np.int16)


def _endpoints_junctions(skel: np.ndarray):
    """Endpoint pixel count and junction *node* count.

    Junction pixels (degree >= 3) are grouped into 8-connected components so a
    fat junction is counted once.
    """
    nb = _neighbour_count(skel)
    endpoint_mask = skel & (nb == 1)
    junction_pix = skel & (nb >= 3)
    if junction_pix.any():
        _, n_junc = ndi.label(junction_pix, structure=np.ones((3, 3), bool))
    else:
        n_junc = 0
    return int(endpoint_mask.sum()), int(n_junc), endpoint_mask, junction_pix


def box_counting_fd(binary: np.ndarray, n_sizes: int = FD_N_SIZES):
    """Box-counting fractal dimension of a binary image.

    Returns ``(D, r2)`` where ``D`` is the least-squares slope of
    ``log N(s)`` against ``log(1/s)`` and ``r2`` the coefficient of
    determination of that fit.  Box sizes are the unique integers of a
    geometric sequence from 2 to ``min(H, W) / 4``.
    """
    b = np.asarray(binary, dtype=bool)
    if b.sum() < 10:
        return float("nan"), float("nan")
    h, w = b.shape
    max_box = max(FD_MIN_BOX * 2, int(min(h, w) // 4))
    if max_box <= FD_MIN_BOX:
        return float("nan"), float("nan")
    sizes = np.unique(
        np.round(np.geomspace(FD_MIN_BOX, max_box, n_sizes)).astype(int)
    )
    sizes = sizes[sizes >= 1]
    if sizes.size < 3:
        return float("nan"), float("nan")

    counts = np.empty(sizes.size, dtype=np.float64)
    for i, s in enumerate(sizes):
        s = int(s)
        ph, pw = (-h) % s, (-w) % s
        padded = np.pad(b, ((0, ph), (0, pw)), mode="constant", constant_values=False)
        blocks = padded.reshape(padded.shape[0] // s, s, padded.shape[1] // s, s)
        counts[i] = float(blocks.any(axis=(1, 3)).sum())

    keep = counts > 0
    if int(keep.sum()) < 3:
        return float("nan"), float("nan")
    x = np.log(1.0 / sizes[keep].astype(np.float64))
    y = np.log(counts[keep])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(slope), float(r2)


def _branch_table(skel: np.ndarray):
    """skan branch summary -> (n_branches, arc lengths, chord lengths, error).

    Returns ``(n_branches, arc, chord, err)`` where ``arc``/``chord`` are 1-D
    float arrays.  ``err`` is ``None`` on success, otherwise the exception
    string (metrics that need the table then come back as NaN).
    """
    if not skel.any():
        return 0, np.zeros(0), np.zeros(0), None
    try:
        from skan import Skeleton, summarize

        sk = Skeleton(skel.astype(bool))
        try:
            df = summarize(sk, separator="_")
        except TypeError:  # skan < 0.12 has no separator kwarg
            df = summarize(sk)
        cols = {str(c).replace("-", "_"): c for c in df.columns}
        arc = np.asarray(df[cols["branch_distance"]], dtype=np.float64)
        chord = np.asarray(df[cols["euclidean_distance"]], dtype=np.float64)
        return int(len(df)), arc, chord, None
    except Exception as exc:  # pragma: no cover - version / degenerate input
        return -1, np.zeros(0), np.zeros(0), f"{type(exc).__name__}: {exc}"


def _tortuosity(arc: np.ndarray, chord: np.ndarray):
    """(mean, length-weighted mean, n_used) arc/chord tortuosity."""
    if arc.size == 0:
        return float("nan"), float("nan"), 0
    ok = (arc >= MIN_BRANCH_PX) & (chord > 1e-6)
    if not ok.any():
        return float("nan"), float("nan"), 0
    t = arc[ok] / chord[ok]
    wl = arc[ok]
    return float(np.mean(t)), float(np.sum(t * wl) / np.sum(wl)), int(ok.sum())


def _bbox_slices(region: np.ndarray):
    ys, xs = np.where(region)
    if ys.size == 0:
        return None
    return (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))


def _nan_metrics() -> Dict[str, float]:
    return {k: float("nan") for k in _METRIC_KEYS}


# --------------------------------------------------------------------------- #
# per-region measurement
# --------------------------------------------------------------------------- #
def _region_metrics(
    mask: np.ndarray,
    skel: np.ndarray,
    edt: np.ndarray,
    region: np.ndarray,
):
    """Returns ``(metrics, skan_error_or_None)``."""
    out = _nan_metrics()
    region_area = float(region.sum())
    out["region_area_px"] = region_area
    if region_area <= 0:
        return out, None

    m = mask & region
    s = skel & region
    out["vessel_area_px"] = float(m.sum())
    out["vessel_density"] = float(m.sum()) / region_area
    out["skeleton_px"] = float(s.sum())

    n_end, n_junc, _, _ = _endpoints_junctions(s)
    out["n_endpoints"] = float(n_end)
    out["n_junctions"] = float(n_junc)

    n_br, arc, chord, err = _branch_table(s)
    if err is None and n_br >= 0:
        out["n_branches"] = float(n_br)
        out["skeleton_length_total"] = float(arc.sum())
    else:
        out["n_branches"] = float("nan")
        out["skeleton_length_total"] = float(s.sum())  # pixel-count fallback
    out["skeleton_length_total_norm"] = out["skeleton_length_total"] / region_area

    tm, tw, n_used = _tortuosity(arc, chord)
    out["tortuosity_mean"] = tm
    out["tortuosity_length_weighted"] = tw
    out["tortuosity_n_branches_used"] = float(n_used)

    if s.any():
        out["mean_radius"] = float(edt[s].mean())

    bb = _bbox_slices(region)
    if bb is not None:
        fd, r2 = box_counting_fd(s[bb])
        out["fractal_dimension"] = fd
        out["fractal_dimension_r2"] = r2
        # Same estimator applied to the filled mask instead of the skeleton.
        # PVBM's D0 is a *segmentation* box-count, so this column isolates
        # "different object" from "different estimator" when the two pipelines
        # disagree on FD.  Diagnostic only -- the primary FD stays the skeleton.
        fdm, _ = box_counting_fd(m[bb])
        out["fractal_dimension_mask"] = fdm
    return out, err


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def compute_biomarkers_skan(
    mask,
    fov,
    disc: Optional[DiscEstimate] = None,
    px_per_mm: Optional[float] = None,
) -> Dict[str, float]:
    """Compute the skan/skeleton biomarkers of a binary vessel mask.

    Parameters
    ----------
    mask : (H, W) array
        Binary vessel mask (ground truth or prediction).
    fov : (H, W) array
        Field-of-view mask.  All measurements are restricted to it.
    disc : DiscEstimate, optional
        If given, the zone-B (annulus 1.0-1.5 disc diameters from the disc
        centre) versions of every metric are added with a ``zoneB_`` prefix.
    px_per_mm : float, optional
        If given, length-like metrics are additionally reported in mm.

    Returns
    -------
    dict
        Flat ``str -> float`` dictionary.  Missing/undefined values are NaN.
    """
    m = _as_bool(mask)
    f = _as_bool(fov)
    if m.shape != f.shape:
        raise ValueError(f"mask {m.shape} and fov {f.shape} shapes differ")
    m = m & f

    out: Dict[str, float] = {}
    out["fov_area_px"] = float(f.sum())

    skel = skeletonize_mask(m)
    skel &= f
    edt = ndi.distance_transform_edt(m) if m.any() else np.zeros(m.shape, np.float32)

    glob, err = _region_metrics(m, skel, edt, f)
    out.update(glob)
    out["skan_ok"] = float(err is None)
    if err is not None:
        out["skan_error"] = err  # type: ignore[assignment]

    if px_per_mm and px_per_mm > 0:
        out["skeleton_length_total_mm"] = out["skeleton_length_total"] / px_per_mm
        out["mean_radius_mm"] = out["mean_radius"] / px_per_mm
        out["skeleton_length_total_per_mm2"] = out["skeleton_length_total"] / (
            out["region_area_px"] / (px_per_mm ** 2)
        )

    if disc is not None:
        zone = zone_b_mask(m.shape, disc) & f
        zm, _zerr = _region_metrics(m, skel, edt, zone)
        for k, v in zm.items():
            out[f"zoneB_{k}"] = v
        out["zoneB_available"] = float(zone.sum() > 0)
        out["disc_cx"] = float(disc.cx)
        out["disc_cy"] = float(disc.cy)
        out["disc_r"] = float(disc.r)
        out["disc_confident"] = float(bool(disc.confident))
    else:
        out["zoneB_available"] = 0.0

    return out


if __name__ == "__main__":  # pragma: no cover - smoke test
    import time

    rng = np.random.default_rng(0)
    h, w = 584, 565
    yy, xx = np.mgrid[:h, :w]
    fov_t = ((yy - h / 2) ** 2 / (h / 2) ** 2 + (xx - w / 2) ** 2 / (w / 2) ** 2) < 1
    mask_t = np.zeros((h, w), bool)
    for k in range(12):
        a = rng.uniform(0, 2 * math.pi)
        t = np.linspace(0, 250, 600)
        ry = (h / 2 + t * math.sin(a) + 12 * np.sin(t / 25.0)).astype(int)
        rx = (w / 2 + t * math.cos(a) + 12 * np.cos(t / 25.0)).astype(int)
        ok = (ry >= 2) & (ry < h - 2) & (rx >= 2) & (rx < w - 2)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                mask_t[ry[ok] + dy, rx[ok] + dx] = True
    t0 = time.perf_counter()
    res = compute_biomarkers_skan(mask_t, fov_t)
    dt = time.perf_counter() - t0
    for key in sorted(res):
        print(f"{key:38s} {res[key]:.6g}")
    print(f"runtime {dt:.3f}s")
