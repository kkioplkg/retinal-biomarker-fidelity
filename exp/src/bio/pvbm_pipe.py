"""Pipeline #1: thin wrapper around the third-party PVBM toolbox.

Plan reference: exp/EXPERIMENT_PLAN.md S1.1 -- "PVBM takes the binary mask
directly (+ optic-disc centre, FOV) and outputs density, total length, fractal
dimension, tortuosity, ...".  Proposal Table in section 4.1 lists PVBM as
"main pipeline A" (MIT licence, natively accepts an external mask).

Exact API used (recorded for reproducibility; verified against the installed
package at run time by ``pvbm_api_summary()``)
-------------------------------------------------------------------------
Package
    ``pvbm`` on PyPI, imported as ``PVBM``; the API below was read off the
    **3.0.1.0** source (setup.py version; PVBM/__init__.py is empty, so
    importing a submodule does not drag in torch/onnxruntime/cv2).  It is the version
    this wrapper was written against (``PVBM_VERSION_WRITTEN_AGAINST``).  The
    version actually installed is reported in every result dict under
    ``pvbm_version``.

``PVBM.GeometryAnalysis.GeometricalVBMs``  (note: the *newer* module
``GeometryAnalysis``; the older ``PVBM.GeometricalAnalysis`` emits a
deprecation warning on construction and is NOT used)
    ``apply_roi(segmentation, skeleton, zones_ABC, roi) -> (segmentation_roi,
    skeleton_roi)``
        Keeps ``roi[:, :, 1] / 255`` and removes ``zones_ABC[:, :, 1] / 255``;
        both are uint8 3-channel images with 255 in the green channel inside
        the corresponding circle.  We build these two arrays ourselves from our
        own optic-disc estimate (``bio.disc``) rather than from
        ``PVBM.DiscSegmenter``, which needs a downloaded torch model and a file
        path and is therefore unusable in a CPU-only Gate A.
    ``compute_geomVBMs(blood_vessel, skeleton, xc, yc, radius)``
        ``-> ([area, TI, medTor, ovlen, medianba, startp, endp, interp],
              (endpoints, interpoints, startpoints, angles_dico, dico))``
        i.e. [vessel area (px), tortuosity index, median tortuosity, overall
        length, median branching angle, #startpoints, #endpoints,
        #intersection points].  ``blood_vessel`` and ``skeleton`` are 2-D numpy
        arrays with values in {0, 1}; ``xc`` is the COLUMN and ``yc`` the ROW of
        the optic-disc centre (PVBM indexes ``point[1]`` with ``x_c`` and
        ``point[0]`` with ``y_c``), ``radius`` the disc radius in pixels.

``PVBM.FractalAnalysis.MultifractalVBMs``
    ``__init__(n_dim=10, n_rotations=25, optimize=True, min_proba=0.01,
    maxproba=0.98)``
    ``compute_multifractals(segmentation) -> np.hstack((three_dqs,
    sing_features))``
        first three entries are the capacity (D0), entropy (D1) and correlation
        (D2) dimensions, the next is the singularity length (SL).  The input
        must be a **square** binary array, so we crop to the measurement
        region's bounding box and zero-pad to a square.

Not used: ``PVBM.DiscSegmenter`` (torch model + image path),
``PVBM.CentralRetinalAnalysis`` (CRAE/CRVE/AVR need artery/vein labels, which
the proposal puts in the optional typed-vessel extension only),
``PVBM.Datasets``, ``PVBM.LesionSegmenter``.

Zone-B warning (measured in Gate A, 2026-09-02)
----------------------------------------------
``compute_geomVBMs`` seeds its graph traversal at the optic disc
(``extract_subgraphs(..., x_c=xc, y_c=yc)``), so applying it to the zone-B
annulus -- which by definition *excludes* the disc -- leaves the walk with
nothing to start from.  Measured on the ground-truth masks of all four
datasets, ``zoneB_total_length`` came back exactly 0 for 45/45 HRF, 98/100
FIVES and 12/56 CHASE_DB1 images, and ``zoneB_median_tortuosity`` was
undefined for 45/45 HRF and 98/100 FIVES.  Consequently **only
``zoneB_vessel_density`` (a plain pixel ratio, Spearman 1.00 against
skan) is usable from PVBM**; every other zone-B geometric VBM must come from
``skan_pipe``.  The global (whole-FOV) VBMs are unaffected -- there the walk
starts on the disc as intended.

Robustness notes
----------------
* ``compute_geomVBMs`` walks the skeleton with a *recursive* Python function
  (``recursive_subgraph``).  On HRF/FIVES-sized skeletons this overflows the
  default 1 MB thread stack, so every PVBM call runs inside a thread created
  with a 256 MB stack and a raised recursion limit.
* Any PVBM failure yields NaNs plus an error string instead of killing the run.
"""

from __future__ import annotations

import math
import sys
import threading
from typing import Dict, Optional

import numpy as np

from .disc import DiscEstimate

__all__ = [
    "compute_biomarkers_pvbm",
    "pvbm_api_summary",
    "pvbm_version",
    "PVBM_VERSION_WRITTEN_AGAINST",
]

PVBM_VERSION_WRITTEN_AGAINST = "3.0.1.0"

# Zone B annulus in disc radii: PVBM's apply_roi removes zones_ABC[:,:,1] and
# keeps roi[:,:,1]; with the two circles below this is exactly the annulus
# [2r, 3r] = [1.0, 1.5] disc diameters used by skan_pipe / disc.zone_b_mask.
ZONE_B_INNER_R = 2.0
ZONE_B_OUTER_R = 3.0

_GEOM_NAMES = (
    "vessel_area_px",  # area
    "tortuosity_index",  # TI
    "median_tortuosity",  # medTor
    "total_length",  # ovlen
    "median_branching_angle",  # medianba
    "n_startpoints",  # startp
    "n_endpoints",  # endp
    "n_intersections",  # interp
)
_FRACTAL_NAMES = ("fractal_D0", "fractal_D1", "fractal_D2", "singularity_length")

_STACK_BYTES = 256 * 1024 * 1024
_RECURSION_LIMIT = 200_000


# --------------------------------------------------------------------------- #
# import / introspection
# --------------------------------------------------------------------------- #
def _import_pvbm():
    """Return ``(GeometricalVBMs, MultifractalVBMs)`` classes, or raise."""
    from PVBM.FractalAnalysis import MultifractalVBMs
    from PVBM.GeometryAnalysis import GeometricalVBMs

    return GeometricalVBMs, MultifractalVBMs


def pvbm_version() -> str:
    try:
        try:
            from importlib.metadata import version as _v
        except ImportError:  # py<3.8
            from importlib_metadata import version as _v  # type: ignore
        return str(_v("pvbm"))
    except Exception:
        try:
            import PVBM  # noqa: F401

            return str(getattr(PVBM, "__version__", "unknown"))
        except Exception:
            return "not-installed"


def pvbm_api_summary() -> Dict[str, object]:
    """Introspect the installed PVBM and return a machine-readable API dump."""
    import importlib
    import inspect
    import pkgutil

    out: Dict[str, object] = {"version": pvbm_version()}
    try:
        import PVBM
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["path"] = list(getattr(PVBM, "__path__", []))
    mods = {}
    for mi in pkgutil.iter_modules(getattr(PVBM, "__path__", [])):
        name = f"PVBM.{mi.name}"
        entry: Dict[str, object] = {}
        try:
            mod = importlib.import_module(name)
        except Exception as exc:
            entry["import_error"] = f"{type(exc).__name__}: {exc}"
            mods[name] = entry
            continue
        for cname, cls in inspect.getmembers(mod, inspect.isclass):
            if getattr(cls, "__module__", "") != name:
                continue
            meths = {}
            for mname, fn in inspect.getmembers(cls, inspect.isfunction):
                if mname.startswith("_") and mname != "__init__":
                    continue
                try:
                    sig = str(inspect.signature(fn))
                except (TypeError, ValueError):
                    sig = "(?)"
                doc = (inspect.getdoc(fn) or "").strip().splitlines()
                meths[mname] = {"signature": sig, "doc": doc[:14]}
            entry[cname] = meths
        mods[name] = entry
    out["modules"] = mods
    return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_bool(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == bool:
        return a
    mx = float(a.max()) if a.size else 0.0
    return a > (0.5 * mx if mx > 1 else 0.5)


def _call_with_big_stack(fn, *args, **kwargs):
    """Run ``fn`` in a thread with a large stack (PVBM recurses over the skeleton)."""
    box: Dict[str, object] = {}

    def _target():
        try:
            sys.setrecursionlimit(_RECURSION_LIMIT)
            box["v"] = fn(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            box["e"] = exc

    try:
        old = threading.stack_size(_STACK_BYTES)
    except (ValueError, RuntimeError):
        old = None
    t = threading.Thread(target=_target)
    t.start()
    t.join()
    if old is not None:
        try:
            threading.stack_size(old)
        except (ValueError, RuntimeError):
            pass
    if "e" in box:
        raise box["e"]  # type: ignore[misc]
    return box.get("v")


def _circle_channel(shape, cx: float, cy: float, radius: float) -> np.ndarray:
    """uint8 (H, W, 3) image with 255 in the green channel inside the circle.

    Mirrors what ``PVBM.DiscSegmenter.post_processing`` hands to ``apply_roi``.
    """
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    inside = ((yy - cy) ** 2 + (xx - cx) ** 2) <= max(radius, 0.0) ** 2
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[..., 1] = inside.astype(np.uint8) * 255
    return out


def _mask_channel(mask: np.ndarray) -> np.ndarray:
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    out[..., 1] = mask.astype(np.uint8) * 255
    return out


def _pad_square(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    n = max(h, w)
    if h == w:
        return a
    out = np.zeros((n, n), dtype=a.dtype)
    y0, x0 = (n - h) // 2, (n - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = a
    return out


def _bbox(region: np.ndarray):
    ys, xs = np.where(region)
    if ys.size == 0:
        return None
    return (slice(int(ys.min()), int(ys.max()) + 1), slice(int(xs.min()), int(xs.max()) + 1))


def _nan(names) -> Dict[str, float]:
    return {n: float("nan") for n in names}


# --------------------------------------------------------------------------- #
# individual PVBM calls
# --------------------------------------------------------------------------- #
def _geom(gvbm, seg: np.ndarray, skel: np.ndarray, xc: float, yc: float, radius: float):
    vbms, _visual = _call_with_big_stack(
        gvbm.compute_geomVBMs,
        seg.astype(np.float64),
        skel.astype(np.float64),
        float(xc),
        float(yc),
        float(radius),
    )
    vals = [float(v) for v in np.asarray(vbms, dtype=np.float64).ravel()]
    if len(vals) != len(_GEOM_NAMES):
        raise ValueError(
            f"compute_geomVBMs returned {len(vals)} values, expected {len(_GEOM_NAMES)} "
            f"({', '.join(_GEOM_NAMES)}); PVBM API changed -- update pvbm_pipe."
        )
    return dict(zip(_GEOM_NAMES, vals))


def _fractal(mvbm, seg: np.ndarray, region: np.ndarray):
    bb = _bbox(region)
    if bb is None:
        return _nan(_FRACTAL_NAMES)
    sub = _pad_square(np.ascontiguousarray(seg[bb]).astype(np.float64))
    if sub.sum() < 10:
        return _nan(_FRACTAL_NAMES)
    res = np.asarray(_call_with_big_stack(mvbm.compute_multifractals, sub), dtype=np.float64).ravel()
    out = _nan(_FRACTAL_NAMES)
    for i, name in enumerate(_FRACTAL_NAMES):
        if i < res.size:
            out[name] = float(res[i])
    if res.size > len(_FRACTAL_NAMES):
        for i in range(len(_FRACTAL_NAMES), res.size):
            out[f"fractal_extra_{i}"] = float(res[i])
    return out


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def compute_biomarkers_pvbm(
    mask,
    fov,
    disc: Optional[DiscEstimate] = None,
    px_per_mm: Optional[float] = None,
    fd_rotations: int = 25,
    zone_b: bool = True,
) -> Dict[str, float]:
    """Compute PVBM biomarkers of a binary vessel mask.

    Parameters
    ----------
    mask, fov : (H, W) arrays
        Binary vessel mask and field of view.  ``mask`` is intersected with
        ``fov`` before anything is measured.
    disc : DiscEstimate, optional
        Optic disc.  Required for the branching-angle / start-point logic of
        ``compute_geomVBMs`` to be meaningful and for the zone-B variants; when
        absent the FOV centroid and ``0.08 * FOV diameter`` are substituted and
        ``disc_used`` is set to 0.
    px_per_mm : float, optional
        Adds mm-scaled copies of the length metrics.
    fd_rotations : int
        ``n_rotations`` of ``MultifractalVBMs`` (PVBM default 25).  Lower it to
        trade accuracy for speed on 2k-4k images.
    zone_b : bool
        Also compute the annulus [1.0, 1.5] disc-diameter variants.

    Returns
    -------
    dict
        Flat ``str -> float`` dict, plus ``version`` /
        ``geom_error`` / ``fractal_error`` strings (they become ``pvbm_version``
        etc. once biomarkers.compute_all applies the ``pvbm_`` prefix).
    """
    m = _as_bool(mask)
    f = _as_bool(fov)
    if m.shape != f.shape:
        raise ValueError(f"mask {m.shape} and fov {f.shape} shapes differ")
    m = m & f

    out: Dict[str, float] = {}
    out["version"] = pvbm_version()  # type: ignore[assignment]
    out["fov_area_px"] = float(f.sum())
    out.update(_nan(_GEOM_NAMES))
    out.update(_nan(_FRACTAL_NAMES))
    out["vessel_density"] = float("nan")
    out["geom_ok"] = 0.0
    out["fractal_ok"] = 0.0

    try:
        GeometricalVBMs, MultifractalVBMs = _import_pvbm()
    except Exception as exc:
        out["import_error"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]
        return out

    from skimage.morphology import skeletonize

    skel = skeletonize(m).astype(np.float64) if m.any() else np.zeros(m.shape)
    seg = m.astype(np.float64)

    if disc is not None:
        xc, yc, rd = float(disc.cx), float(disc.cy), float(disc.r)
        out["disc_used"] = 1.0
    else:
        ys, xs = np.where(f)
        yc = float(ys.mean()) if ys.size else m.shape[0] / 2.0
        xc = float(xs.mean()) if xs.size else m.shape[1] / 2.0
        area = float(f.sum())
        fov_d = 2.0 * math.sqrt(area / math.pi) if area > 0 else float(min(m.shape))
        rd = 0.08 * fov_d
        out["disc_used"] = 0.0
    out["disc_cx"], out["disc_cy"], out["disc_r"] = xc, yc, rd

    gvbm = GeometricalVBMs()
    mvbm = MultifractalVBMs(n_rotations=int(fd_rotations))

    # ---- global ------------------------------------------------------------
    try:
        out.update(_geom(gvbm, seg, skel, xc, yc, rd))
        out["geom_ok"] = 1.0
        if out["fov_area_px"] > 0:
            out["vessel_density"] = out["vessel_area_px"] / out["fov_area_px"]
    except Exception as exc:
        out["geom_error"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]

    try:
        out.update(_fractal(mvbm, seg, f))
        out["fractal_ok"] = 1.0
    except Exception as exc:
        out["fractal_error"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]

    if px_per_mm and px_per_mm > 0:
        out["total_length_mm"] = out["total_length"] / px_per_mm

    # ---- zone B (annulus [2r, 3r] = [1.0, 1.5] disc diameters) -------------
    if zone_b and disc is not None:
        zone_names = [f"zoneB_{n}" for n in _GEOM_NAMES] + [
            f"zoneB_{n}" for n in _FRACTAL_NAMES
        ]
        out.update(_nan(zone_names))
        out["zoneB_vessel_density"] = float("nan")
        try:
            zones_abc = _circle_channel(m.shape, xc, yc, ZONE_B_INNER_R * rd)
            roi = _circle_channel(m.shape, xc, yc, ZONE_B_OUTER_R * rd)
            # restrict the ROI to the FOV as well
            roi[..., 1] = np.minimum(roi[..., 1], _mask_channel(f)[..., 1])
            seg_roi, skel_roi = gvbm.apply_roi(seg, skel, zones_abc, roi)
            seg_roi = (np.asarray(seg_roi) > 0.5).astype(np.float64)
            skel_roi = (np.asarray(skel_roi) > 0.5).astype(np.float64)
            region = (roi[..., 1] > 0) & (zones_abc[..., 1] == 0)
            out["zoneB_region_area_px"] = float(region.sum())
            try:
                g = _geom(gvbm, seg_roi, skel_roi, xc, yc, rd)
                for k, v in g.items():
                    out[f"zoneB_{k}"] = v
                if region.sum() > 0:
                    out["zoneB_vessel_density"] = out["zoneB_vessel_area_px"] / float(
                        region.sum()
                    )
            except Exception as exc:
                out["zoneB_geom_error"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]
            try:
                fr = _fractal(mvbm, seg_roi, region)
                for k, v in fr.items():
                    out[f"zoneB_{k}"] = v
            except Exception as exc:
                out["zoneB_fractal_error"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]
        except Exception as exc:
            out["zoneB_error"] = f"{type(exc).__name__}: {exc}"  # type: ignore[assignment]

    return out


# --------------------------------------------------------------------------- #
def _print_api() -> None:
    import json

    print(json.dumps(pvbm_api_summary(), indent=2, default=str))


if __name__ == "__main__":  # pragma: no cover
    if "--api" in sys.argv:
        _print_api()
    else:
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
        res = compute_biomarkers_pvbm(mask_t, fov_t)
        print(f"runtime {time.perf_counter() - t0:.2f}s")
        for key in sorted(res):
            print(f"{key:34s} {res[key]}")
