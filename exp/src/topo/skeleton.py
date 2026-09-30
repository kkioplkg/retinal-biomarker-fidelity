"""Skeletonisation, skeleton graphs and Betti numbers for binary vessel masks.

Project-wide conventions (proposal v3 §3.1.3 / §4.2, exp/EXPERIMENT_PLAN.md)
---------------------------------------------------------------------------
* **Foreground 8-connectivity / background 4-connectivity.**  Every connected
  component count on a *vessel* (foreground) mask uses the 3x3 all-ones
  structuring element; every connected component count on the *background*
  uses the 4-neighbour cross.  This pair is the only self-consistent choice on
  a square lattice (Jordan-curve property) and is declared explicitly because
  the two conventions give different Betti numbers.
* **Everything is computed inside a boolean FOV mask.**  Pixels outside the FOV
  are treated as if they were not part of the image: they are neither
  foreground nor background, and the FOV boundary counts as "outside".
* Coordinates are always ``(row, col)`` integer pairs, i.e. numpy index order.

Only numpy / scipy / skimage / skan are used; no GPU, no torch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

__all__ = [
    "EIGHT",
    "FOUR",
    "as_bool",
    "fov_or_true",
    "skeletonize_mask",
    "prune_spurs",
    "neighbour_count",
    "endpoints",
    "junctions",
    "junction_clusters",
    "local_radius",
    "local_diameter",
    "connected_components",
    "betti_numbers",
    "drop_isolated_pixels",
    "build_graph",
    "summarize",
    "SUMMARY_COLUMNS",
]

#: 3x3 all-ones structuring element -- **foreground / 8-connectivity**.
EIGHT = np.ones((3, 3), dtype=int)

#: 4-neighbour cross structuring element -- **background / 4-connectivity**.
FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)

#: 3x3 kernel counting the 8 neighbours of a pixel (centre excluded).
_NB_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def as_bool(arr: np.ndarray) -> np.ndarray:
    """Return ``arr`` as a contiguous boolean array (0 -> False, else True)."""
    a = np.asarray(arr)
    if a.dtype == bool:
        return a
    return a.astype(bool)


def fov_or_true(fov: Optional[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Return a boolean FOV mask; ``None`` means "the whole image"."""
    if fov is None:
        return np.ones(shape, dtype=bool)
    f = as_bool(fov)
    if f.shape != tuple(shape):
        raise ValueError(f"FOV shape {f.shape} != image shape {tuple(shape)}")
    return f


# --------------------------------------------------------------------------
# skeletonisation
# --------------------------------------------------------------------------
def neighbour_count(skel: np.ndarray) -> np.ndarray:
    """Number of 8-neighbours of each pixel that are skeleton pixels.

    Returns a ``uint8`` array of the same shape.  Values are only meaningful on
    skeleton pixels; the array is 0 outside.  Pixels outside the image count as
    background (``mode='constant'``).
    """
    s = as_bool(skel).astype(np.uint8)
    cnt = ndi.convolve(s, _NB_KERNEL, mode="constant", cval=0)
    return (cnt * s).astype(np.uint8)


def _thin(mask: np.ndarray) -> np.ndarray:
    """Zhang-Suen thinning; idempotent on a well-formed skeleton."""
    from skimage.morphology import skeletonize as _skeletonize

    m = as_bool(mask)
    if not m.any():
        return m
    try:
        return as_bool(_skeletonize(m, method="zhang"))
    except TypeError:  # older scikit-image
        return as_bool(_skeletonize(m))


def _prune_pass(skel: np.ndarray, min_branch_px: int) -> bool:
    """One in-place sweep removing short terminal segments.  Returns True if any."""
    deg = neighbour_count(skel)
    junc = skel & (deg >= 3)
    ends = skel & (deg == 1)
    if not junc.any() or not ends.any():
        return False
    seg = skel & ~junc
    lab, n = ndi.label(seg, structure=EIGHT)
    if n == 0:
        return False
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    has_end = np.zeros(n + 1, dtype=bool)
    end_labels = lab[ends]
    has_end[end_labels[end_labels > 0]] = True
    junc_dil = ndi.binary_dilation(junc, structure=EIGHT.astype(bool))
    touch = np.zeros(n + 1, dtype=bool)
    touch_labels = lab[junc_dil & seg]
    touch[touch_labels[touch_labels > 0]] = True
    remove = np.zeros(n + 1, dtype=bool)
    remove[1:] = has_end[1:] & touch[1:] & (sizes[1:] < int(min_branch_px))
    if not remove.any():
        return False
    skel[remove[lab]] = False
    return True


def prune_spurs(
    skel: np.ndarray, min_branch_px: int = 3, max_iter: int = 10, rethin: bool = True
) -> np.ndarray:
    """Remove short terminal branches ("spurs") from a 1-pixel-wide skeleton.

    A *spur* is a maximal run of non-junction skeleton pixels that (a) contains
    a degree-1 pixel (an endpoint), (b) is 8-adjacent to at least one junction
    pixel (degree >= 3), and (c) is shorter than ``min_branch_px`` pixels.
    Isolated segments with two endpoints and no junction contact are whole
    vessel fragments and are **never** removed.

    After each removal sweep the skeleton is re-thinned (``rethin=True``).  This
    is needed because the *first* pixel of a spur is 8-adjacent to three pixels
    of the parent branch and therefore has degree >= 3 itself: it is classified
    as a junction pixel and survives the segment removal.  Re-thinning deletes
    such now-redundant simple points, and Zhang-Suen thinning is idempotent on
    an already-thin skeleton (and never shortens a branch at an endpoint), so it
    is safe to apply repeatedly.  The sweep/re-thin cycle repeats until nothing
    changes or ``max_iter`` iterations have run.

    Parameters
    ----------
    skel : (H, W) bool
    min_branch_px : int
        Branches with strictly fewer than this many pixels are pruned.
        ``min_branch_px <= 1`` disables pruning.
    max_iter : int
    rethin : bool
        Disable to get the raw segment removal without the clean-up pass.
    """
    out = as_bool(skel).copy()
    if min_branch_px is None or min_branch_px <= 1 or not out.any():
        return out

    for _ in range(int(max_iter)):
        changed = _prune_pass(out, int(min_branch_px))
        if rethin:
            out = _thin(out)
        if not changed:
            break
    return out


def skeletonize_mask(
    mask: np.ndarray,
    min_branch_px: int = 3,
    prune: bool = True,
    fov: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Morphological skeleton of a binary vessel mask.

    Uses :func:`skimage.morphology.skeletonize` (Zhang-Suen thinning), which
    produces an **8-connected** skeleton of an 8-connected foreground -- the
    convention fixed for this project.

    Parameters
    ----------
    mask : (H, W) array-like
        Binary vessel mask.
    min_branch_px : int, default 3
        Spur-pruning length (see :func:`prune_spurs`).
    prune : bool, default True
        If False the raw thinning result is returned (this is what
        ``metrics.cldice`` uses, following Shit et al. 2021).
    fov : (H, W) bool, optional
        If given, the mask is intersected with the FOV *before* thinning so no
        skeleton is grown from pixels outside the FOV.

    Returns
    -------
    (H, W) bool
    """
    from skimage.morphology import skeletonize as _skeletonize

    m = as_bool(mask)
    if fov is not None:
        m = m & fov_or_true(fov, m.shape)
    if not m.any():
        return np.zeros(m.shape, dtype=bool)
    try:
        skel = _skeletonize(m, method="zhang")
    except TypeError:  # older scikit-image without the `method` kwarg
        skel = _skeletonize(m)
    skel = as_bool(skel)
    if prune:
        skel = prune_spurs(skel, min_branch_px=min_branch_px)
    return skel


# --------------------------------------------------------------------------
# skeleton topology primitives
# --------------------------------------------------------------------------
def endpoints(skel: np.ndarray) -> np.ndarray:
    """Degree-1 skeleton pixels as an ``(N, 2)`` int array of ``(row, col)``.

    Degree = number of 8-neighbours that are skeleton pixels.
    """
    deg = neighbour_count(skel)
    rr, cc = np.nonzero(as_bool(skel) & (deg == 1))
    return np.stack([rr, cc], axis=1).astype(np.int64)


def junctions(skel: np.ndarray) -> np.ndarray:
    """Degree->=3 skeleton pixels as an ``(N, 2)`` int array of ``(row, col)``.

    These are raw *pixels*: a single anatomical bifurcation usually yields two
    or three adjacent junction pixels.  Use :func:`junction_clusters` to get one
    point per bifurcation.
    """
    deg = neighbour_count(skel)
    rr, cc = np.nonzero(as_bool(skel) & (deg >= 3))
    return np.stack([rr, cc], axis=1).astype(np.int64)


def junction_clusters(skel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group adjacent junction pixels into one node per bifurcation.

    Returns
    -------
    centroids : (K, 2) float array
        Centroid ``(row, col)`` of each 8-connected cluster of junction pixels.
    labels : (H, W) int array
        Cluster label map (0 = not a junction pixel).
    degrees : (K,) int array
        Number of distinct incident branch stubs of each cluster.  A stub is an
        8-connected group of non-junction skeleton pixels adjacent to the
        cluster, so a bifurcation has degree 3 and a crossing degree 4.
    """
    s = as_bool(skel)
    deg_map = neighbour_count(s)
    junc = s & (deg_map >= 3)
    lab, n = ndi.label(junc, structure=EIGHT)
    if n == 0:
        return (
            np.zeros((0, 2), dtype=float),
            lab,
            np.zeros(0, dtype=np.int64),
        )
    centroids = np.array(ndi.center_of_mass(junc, lab, np.arange(1, n + 1)), dtype=float)
    centroids = centroids.reshape(n, 2)

    seg = s & ~junc
    degrees = np.zeros(n, dtype=np.int64)
    objs = ndi.find_objects(lab)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        pad = 2
        r0 = max(0, sl[0].start - pad)
        r1 = min(s.shape[0], sl[0].stop + pad)
        c0 = max(0, sl[1].start - pad)
        c1 = min(s.shape[1], sl[1].stop + pad)
        win = (slice(r0, r1), slice(c0, c1))
        cluster = lab[win] == (i + 1)
        stubs = ndi.binary_dilation(cluster, structure=EIGHT.astype(bool)) & seg[win]
        _, k = ndi.label(stubs, structure=EIGHT)
        degrees[i] = k
    return centroids, lab, degrees


def local_radius(mask: np.ndarray) -> np.ndarray:
    """Local vessel **radius** map, in pixels.

    Defined as the Euclidean distance transform of the foreground,
    ``EDT(mask)`` = distance from each vessel pixel to the nearest background
    pixel.  On the centreline of a vessel of width ``w`` this is ``~w/2``, i.e.
    a *radius*.  Use :func:`local_diameter` (= ``2 * EDT``) when a diameter is
    wanted (e.g. the "D" of the severance lengths ``L in {0.5D, 1D, 2D, 4D}``
    in proposal §3.1.3).

    Returns
    -------
    (H, W) float32, zero outside the mask.
    """
    m = as_bool(mask)
    return ndi.distance_transform_edt(m).astype(np.float32)


def local_diameter(mask: np.ndarray) -> np.ndarray:
    """Local vessel **diameter** map = ``2 * EDT(mask)``, float32."""
    return (2.0 * local_radius(mask)).astype(np.float32)


# --------------------------------------------------------------------------
# connected components and Betti numbers
# --------------------------------------------------------------------------
def connected_components(
    mask: np.ndarray, connectivity: int = 8, fov: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int]:
    """Label the connected components of ``mask``.

    Parameters
    ----------
    connectivity : {4, 8}, default 8
        8 = 3x3 all-ones structuring element (the foreground convention),
        4 = the 4-neighbour cross (the background convention).
    fov : bool array, optional
        Restrict to the FOV before labelling.

    Returns
    -------
    (labels, n) : ((H, W) int array, int)
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    m = as_bool(mask)
    if fov is not None:
        m = m & fov_or_true(fov, m.shape)
    structure = EIGHT if connectivity == 8 else FOUR
    lab, n = ndi.label(m, structure=structure)
    return lab, int(n)


def _fov_border_mask(fov: np.ndarray) -> np.ndarray:
    """Pixels inside the FOV that are 4-adjacent to the outside of the FOV.

    The image border counts as "outside", so for a full-frame FOV this is the
    one-pixel frame of the image.
    """
    outside = ~fov
    padded = np.pad(outside, 1, mode="constant", constant_values=True)
    grown = ndi.binary_dilation(padded, structure=FOUR.astype(bool))[1:-1, 1:-1]
    return grown & fov


def betti_numbers(
    mask: np.ndarray, fov: Optional[np.ndarray] = None
) -> Tuple[int, int]:
    """Betti numbers ``(beta0, beta1)`` of a binary mask inside the FOV.

    Convention (declared explicitly, cf. proposal §4.2 protocol rule 2):

    * ``beta0`` = number of **8-connected** foreground components of
      ``mask & fov``.
    * ``beta1`` = number of holes, obtained from the **4-connected** components
      of the background ``~mask & fov``: all background components that touch
      the FOV border are merged into the single unbounded component, and
      ``beta1`` = (number of merged background components) - 1.  Equivalently,
      ``beta1`` = number of background components that do **not** touch the FOV
      border.  If no background component touches the border (the FOV is fully
      covered by vessel) every background component is a hole and the formula
      degenerates gracefully to ``beta1 = n_bg``.

    The FOV boundary is treated as the image boundary: background touching it
    is "outside", not a hole.

    Returns
    -------
    (beta0, beta1) : (int, int)
    """
    m = as_bool(mask)
    f = fov_or_true(fov, m.shape)

    fg = m & f
    _, b0 = ndi.label(fg, structure=EIGHT)

    bg = (~m) & f
    lab_bg, n_bg = ndi.label(bg, structure=FOUR)
    if n_bg == 0:
        return int(b0), 0

    border = _fov_border_mask(f) & bg
    border_labels = np.unique(lab_bg[border])
    border_labels = border_labels[border_labels > 0]
    b1 = int(n_bg - border_labels.size)
    return int(b0), int(max(b1, 0))


# --------------------------------------------------------------------------
# skan graph + branch summary
# --------------------------------------------------------------------------
SUMMARY_COLUMNS = [
    "skeleton_id",
    "node_id_src",
    "node_id_dst",
    "branch_distance",
    "branch_type",
    "euclidean_distance",
    "src_row",
    "src_col",
    "dst_row",
    "dst_col",
    "n_pixels",
    "mean_radius",
    "min_radius",
    "max_radius",
]


def drop_isolated_pixels(skel: np.ndarray) -> np.ndarray:
    """Drop 8-connected skeleton components consisting of a single pixel.

    A one-pixel "vessel" carries no branch and makes skan's graph construction
    degenerate, so it is removed before building the graph.
    """
    s = as_bool(skel)
    lab, n = ndi.label(s, structure=EIGHT)
    if n == 0:
        return s
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes > 1
    keep[0] = False
    return keep[lab]


def build_graph(
    skel: np.ndarray,
    mask: Optional[np.ndarray] = None,
    spacing: float = 1.0,
    drop_isolated: bool = True,
):
    """Build a :class:`skan.Skeleton` graph from a 1-pixel-wide skeleton.

    ``mask`` is accepted for API symmetry with :func:`summarize` (the radius of
    a branch is read from ``EDT(mask)``) but is not needed to build the graph
    itself.  Returns ``None`` for an empty skeleton, because skan raises on one.
    With ``drop_isolated=True`` (default) single-pixel components are removed
    first (see :func:`drop_isolated_pixels`).

    Notes
    -----
    skan's own graph uses 8-connectivity for 2-D skeletons, matching our
    foreground convention, and merges adjacent junction pixels into one node.
    ``branch_type`` codes are skan's: ``0`` endpoint-to-endpoint (isolated
    branch), ``1`` junction-to-endpoint, ``2`` junction-to-junction,
    ``3`` isolated cycle.
    """
    import skan

    s = as_bool(skel)
    if drop_isolated:
        s = drop_isolated_pixels(s)
    if not s.any():
        return None
    return skan.Skeleton(s, spacing=spacing)


def _empty_summary():
    import pandas as pd

    return pd.DataFrame({c: pd.Series(dtype="float64") for c in SUMMARY_COLUMNS})


def summarize(
    skel: np.ndarray,
    mask: Optional[np.ndarray] = None,
    graph=None,
    radius_map: Optional[np.ndarray] = None,
):
    """Per-branch summary table of a skeleton.

    Parameters
    ----------
    skel : (H, W) bool
        1-pixel-wide skeleton (e.g. from :func:`skeletonize_mask`).
    mask : (H, W) bool, optional
        The vessel mask the skeleton came from.  Used to compute the local
        radius map ``EDT(mask)`` sampled along each branch.  If neither ``mask``
        nor ``radius_map`` is given the radius columns are NaN.
    graph : skan.Skeleton, optional
        Pre-built graph (avoids rebuilding it).
    radius_map : (H, W) float, optional
        Pre-computed radius map, overrides ``mask``.

    Returns
    -------
    pandas.DataFrame with (at least) the columns in :data:`SUMMARY_COLUMNS`:
    ``branch_type`` (skan code), ``branch_distance`` (path length in px),
    ``euclidean_distance``, endpoint coordinates ``src_row/src_col`` and
    ``dst_row/dst_col``, ``n_pixels``, and ``mean_radius`` / ``min_radius`` /
    ``max_radius`` -- the EDT of ``mask`` sampled at every pixel of the branch.

    Column names from skan are normalised to snake_case (skan changed the
    separator between releases), so both old and new skan versions give the
    same table here.
    """
    import pandas as pd
    import skan

    s = as_bool(skel)
    if not s.any():
        return _empty_summary()

    sk = graph if graph is not None else build_graph(s)
    if sk is None:
        return _empty_summary()

    try:
        df = skan.summarize(sk, separator="_")
    except TypeError:  # skan < 0.12 has no `separator` kwarg
        df = skan.summarize(sk)
    df = df.copy()
    df.columns = [str(c).replace("-", "_") for c in df.columns]

    # endpoint coordinates -> plain row/col columns
    for side in ("src", "dst"):
        rc = f"image_coord_{side}_0"
        cc = f"image_coord_{side}_1"
        if rc in df.columns:
            df[f"{side}_row"] = df[rc].to_numpy(dtype=float)
            df[f"{side}_col"] = df[cc].to_numpy(dtype=float)
        else:  # very old skan: coordinates live in `coord_src_0`
            df[f"{side}_row"] = df.get(f"coord_{side}_0", np.nan)
            df[f"{side}_col"] = df.get(f"coord_{side}_1", np.nan)

    if radius_map is None and mask is not None:
        radius_map = local_radius(mask)

    n_paths = int(sk.n_paths)
    n_px = np.zeros(n_paths, dtype=np.int64)
    r_mean = np.full(n_paths, np.nan, dtype=float)
    r_min = np.full(n_paths, np.nan, dtype=float)
    r_max = np.full(n_paths, np.nan, dtype=float)
    for i in range(n_paths):
        try:
            coords = np.asarray(sk.path_coordinates(i))
        except Exception:  # pragma: no cover - skan API fallback
            coords = np.asarray(sk.coordinates[sk.path(i)])
        n_px[i] = coords.shape[0]
        if radius_map is not None and coords.size:
            rr = np.clip(np.rint(coords[:, 0]).astype(int), 0, s.shape[0] - 1)
            cc2 = np.clip(np.rint(coords[:, 1]).astype(int), 0, s.shape[1] - 1)
            vals = np.asarray(radius_map)[rr, cc2].astype(float)
            r_mean[i] = float(vals.mean())
            r_min[i] = float(vals.min())
            r_max[i] = float(vals.max())

    df = df.reset_index(drop=True)
    df["n_pixels"] = n_px[: len(df)]
    df["mean_radius"] = r_mean[: len(df)]
    df["min_radius"] = r_min[: len(df)]
    df["max_radius"] = r_max[: len(df)]

    for col in SUMMARY_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    ordered = SUMMARY_COLUMNS + [c for c in df.columns if c not in SUMMARY_COLUMNS]
    return df[ordered]


# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - smoke demo
    from skimage.draw import disk, line

    img = np.zeros((80, 80), dtype=bool)
    rr, cc = line(70, 40, 40, 40)
    img[rr, cc] = True
    rr, cc = line(40, 40, 15, 20)
    img[rr, cc] = True
    rr, cc = line(40, 40, 15, 60)
    img[rr, cc] = True
    img = ndi.binary_dilation(img, structure=np.ones((5, 5), bool))

    sk = skeletonize_mask(img)
    print("Y mask   : betti =", betti_numbers(img), " endpoints =", len(endpoints(sk)))
    cen, _, deg = junction_clusters(sk)
    print("junction clusters:", cen.round(1).tolist(), "degrees:", deg.tolist())

    ring = np.zeros((80, 80), dtype=bool)
    rr, cc = disk((40, 40), 20, shape=ring.shape)
    ring[rr, cc] = True
    rr, cc = disk((40, 40), 14, shape=ring.shape)
    ring[rr, cc] = False
    print("ring mask: betti =", betti_numbers(ring))
    print("max radius of Y mask:", float(local_radius(img).max()))
