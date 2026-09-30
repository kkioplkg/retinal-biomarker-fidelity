"""Shared plumbing for the S6 repair baselines.

Every baseline in this package exposes the *same* entry point::

    repair(image, prob, mask, fov, **kw) -> M_repaired          # (H, W) bool

and, for anything that needs the individual connections (TRR / FCR), the
richer::

    repair_detailed(image, prob, mask, fov, **kw) -> RepairOutput

``RepairOutput.edges`` is a list of :class:`src.eval.trr_fcr.CandidateEdge`,
one per **accepted** connection, carrying the rasterised path pixels.  Methods
that natively emit edges (geometric, EVAPORE) fill this directly; methods that
emit a whole mask (rNCA) get it from :func:`extract_added_edges`, the post-hoc
edge extraction documented in the plan.

Nothing here imports ``src.rigr`` at module scope: the baselines must stay
runnable (and independent) while that package is being written.  The few places
that *can* profit from it try the import lazily and fall back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi

from src.topo import skeleton as sk
from src.eval.trr_fcr import CandidateEdge, rasterize_path, rasterize_tube

__all__ = [
    "RepairOutput",
    "as_bool",
    "endpoint_table",
    "bezier_path",
    "straight_path",
    "paint_edges",
    "extract_added_edges",
    "METHODS",
    "get_method",
]

_EPS = 1e-8

# --------------------------------------------------------------------------
# result container
# --------------------------------------------------------------------------


@dataclass
class RepairOutput:
    """What a baseline returns when the caller wants the connections too.

    Attributes
    ----------
    mask : (H, W) bool
        The repaired mask ``M_repaired``.
    edges : list of CandidateEdge
        One entry per accepted connection, with rasterised ``path`` pixels, the
        two endpoints and the tube ``radius`` that was painted.  May be empty.
    info : dict
        Free-form per-image diagnostics (counts, timings, method knobs).
    """

    mask: np.ndarray
    edges: List[CandidateEdge] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


def as_bool(a) -> np.ndarray:
    return sk.as_bool(a)


# --------------------------------------------------------------------------
# dangling endpoints, tangents, radii
# --------------------------------------------------------------------------


def _walk_branch(skel: np.ndarray, start: Tuple[int, int], n: int) -> np.ndarray:
    """Walk up to ``n`` skeleton pixels inward from ``start`` along its branch.

    Stops at a junction (degree >= 3) or when the branch runs out.  Returns an
    ``(k, 2)`` int array beginning at ``start``.
    """
    s = as_bool(skel)
    deg = sk.neighbour_count(s)
    h, w = s.shape
    pts = [(int(start[0]), int(start[1]))]
    seen = {pts[0]}
    cur = pts[0]
    for _ in range(int(n) - 1):
        nxt = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = cur[0] + dr, cur[1] + dc
                if not (0 <= r < h and 0 <= c < w):
                    continue
                if not s[r, c] or (r, c) in seen:
                    continue
                nxt = (r, c)
                break
            if nxt is not None:
                break
        if nxt is None:
            break
        pts.append(nxt)
        seen.add(nxt)
        cur = nxt
        if deg[cur] >= 3:  # do not trace through a bifurcation
            break
    return np.asarray(pts, dtype=np.int64)


def _tangent(skel: np.ndarray, ep: Tuple[int, int], n: int = 12) -> np.ndarray:
    """Outward unit tangent at an endpoint (PCA over the last ``n`` pixels).

    Directed *away* from the vessel body, i.e. into the gap -- the same
    convention as ``src.rigr.candidates.endpoint_tangent``.
    """
    pts = _walk_branch(skel, ep, n)
    if pts.shape[0] < 2:
        return np.zeros(2, dtype=np.float64)
    x = pts.astype(np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    cov = xc.T @ xc / max(1, len(xc) - 1)
    w, v = np.linalg.eigh(cov)
    d = v[:, int(np.argmax(w))]
    if float(d @ (x[0] - mu)) < 0:  # point outward
        d = -d
    nrm = float(np.linalg.norm(d))
    return d / nrm if nrm > _EPS else np.zeros(2, dtype=np.float64)


def endpoint_table(
    mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    skel: Optional[np.ndarray] = None,
    border_px: float = 5.0,
    trace_px: int = 12,
    min_branch_px: int = 3,
) -> Dict[str, np.ndarray]:
    """Dangling endpoints of ``mask`` with tangent, radius and component id.

    Endpoints within ``border_px`` of the FOV boundary are dropped: a vessel
    leaving the field of view is not a break.

    Returns a dict of parallel arrays: ``coord`` (N, 2) int, ``tangent`` (N, 2)
    float, ``radius`` (N,) float, ``comp`` (N,) int, plus the shared per-image
    fields ``skel``, ``radius_map``, ``labels``.
    """
    m = as_bool(mask)
    f = sk.fov_or_true(fov, m.shape)
    m = m & f
    if skel is None:
        skel = sk.skeletonize_mask(m, min_branch_px=min_branch_px, prune=True, fov=f)
    skel = as_bool(skel)
    radius_map = sk.local_radius(m)
    labels, _ = sk.connected_components(m, connectivity=8)

    ep = sk.endpoints(skel)
    if ep.shape[0] and border_px > 0:
        # distance from every pixel to the outside of the FOV
        d_out = ndi.distance_transform_edt(f)
        keep = d_out[ep[:, 0], ep[:, 1]] > float(border_px)
        ep = ep[keep]

    n = int(ep.shape[0])
    tan = np.zeros((n, 2), dtype=np.float64)
    rad = np.zeros(n, dtype=np.float64)
    comp = np.zeros(n, dtype=np.int64)
    for i in range(n):
        p = (int(ep[i, 0]), int(ep[i, 1]))
        tan[i] = _tangent(skel, p, trace_px)
        # radius from the EDT a few pixels *inside* the branch: the very tip of
        # a skeleton sits at the cap of the vessel where the EDT under-reads.
        walk = _walk_branch(skel, p, max(3, trace_px // 2))
        rad[i] = float(np.median(radius_map[walk[:, 0], walk[:, 1]])) if walk.size else 1.0
        rad[i] = max(rad[i], 0.5)
        comp[i] = int(labels[p])
    return dict(coord=ep.astype(np.int64), tangent=tan, radius=rad, comp=comp,
                skel=skel, radius_map=radius_map, labels=labels, fov=f, mask=m)


# --------------------------------------------------------------------------
# path geometry
# --------------------------------------------------------------------------


def straight_path(p0, p1) -> np.ndarray:
    """8-connected rasterised straight segment between two ``(row, col)``."""
    return rasterize_path([tuple(p0), tuple(p1)])


def bezier_path(p0, t0, p1, t1, n: int = 64) -> np.ndarray:
    """Cubic Bezier from ``p0`` to ``p1`` leaving along ``t0`` / arriving along ``t1``.

    Control points sit one third of the chord length along each outward
    tangent, the standard Hermite-to-Bezier conversion; ``t1`` is negated
    because both tangents point *into* the gap.  The curve is sampled at ``n``
    points and rasterised 8-connected.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    t0 = np.asarray(t0, dtype=np.float64)
    t1 = np.asarray(t1, dtype=np.float64)
    d = float(np.linalg.norm(p1 - p0))
    c0 = p0 + t0 * (d / 3.0)
    c1 = p1 - (-t1) * (d / 3.0)  # -t1 points from p1 back into the gap
    s = np.linspace(0.0, 1.0, int(n))[:, None]
    pts = ((1 - s) ** 3 * p0 + 3 * (1 - s) ** 2 * s * c0
           + 3 * (1 - s) * s ** 2 * c1 + s ** 3 * p1)
    way = np.rint(pts).astype(np.int64)
    # drop consecutive duplicates before rasterising
    keep = np.ones(len(way), dtype=bool)
    keep[1:] = np.any(way[1:] != way[:-1], axis=1)
    return rasterize_path([tuple(x) for x in way[keep]])


def paint_edges(mask: np.ndarray, edges: Sequence[CandidateEdge],
                fov: Optional[np.ndarray] = None) -> np.ndarray:
    """Union ``mask`` with the tube of every edge (clipped to the FOV)."""
    out = as_bool(mask).copy()
    f = sk.fov_or_true(fov, out.shape)
    for e in edges:
        out |= rasterize_tube(e.path, float(e.radius), out.shape)
    return out & f


# --------------------------------------------------------------------------
# post-hoc edge extraction for whole-mask methods (rNCA)
# --------------------------------------------------------------------------


def _order_blob_path(blob: np.ndarray, src: Tuple[int, int],
                     dst: Tuple[int, int]) -> np.ndarray:
    """Shortest 8-connected path from ``src`` to ``dst`` **inside** ``blob``.

    Plain BFS over the blob pixels.  Falls back to the straight segment when
    the two seeds are not connected within the blob (should not happen, the
    caller passes seeds of the same component).
    """
    b = as_bool(blob)
    h, w = b.shape
    src = (int(src[0]), int(src[1]))
    dst = (int(dst[0]), int(dst[1]))
    if not b[src] or not b[dst]:
        return straight_path(src, dst)
    prev = {src: None}
    queue = [src]
    head = 0
    while head < len(queue):
        cur = queue[head]
        head += 1
        if cur == dst:
            break
        r0, c0 = cur
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (r0 + dr, c0 + dc)
                if not (0 <= nb[0] < h and 0 <= nb[1] < w):
                    continue
                if not b[nb] or nb in prev:
                    continue
                prev[nb] = cur
                queue.append(nb)
    if dst not in prev:
        return straight_path(src, dst)
    path = []
    node = dst
    while node is not None:
        path.append(node)
        node = prev[node]
    return np.asarray(path[::-1], dtype=np.int64)


def extract_added_edges(
    base_mask: np.ndarray,
    repaired_mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    min_px: int = 2,
) -> List[CandidateEdge]:
    """Recover the *edges* implied by a whole-mask repair.

    The plan's post-hoc rule: take the connected components of
    ``M_repaired \\ M_hat``; a component that touches **two different**
    components of ``M_hat`` has joined them, so it counts as one accepted edge.
    Its path is the shortest 8-connected route through the added blob between
    the pixel nearest each of the two base components, and its radius is the
    median EDT of the blob (the tube it actually painted).

    Components touching zero or one base component are thickening, not
    reconnection, and are ignored.  A blob touching more than two base
    components is emitted as one edge per newly joined pair, using a spanning
    set (first component paired with each of the others), so the edge count
    stays equal to the number of merges it caused.
    """
    base = as_bool(base_mask)
    rep = as_bool(repaired_mask)
    f = sk.fov_or_true(fov, base.shape)
    base = base & f
    rep = rep & f

    added = rep & ~base
    if not added.any():
        return []

    base_lab, n_base = sk.connected_components(base, connectivity=8)
    add_lab, n_add = sk.connected_components(added, connectivity=8)
    if n_base == 0 or n_add == 0:
        return []

    edt_add = ndi.distance_transform_edt(added)
    h, w = base.shape
    edges: List[CandidateEdge] = []

    objs = ndi.find_objects(add_lab)
    for k in range(1, n_add + 1):
        sl = objs[k - 1]
        if sl is None:
            continue
        blob = add_lab[sl] == k
        if int(blob.sum()) < int(min_px):
            continue
        # which base components does this blob touch (8-neighbourhood)?
        grown = ndi.binary_dilation(blob, structure=np.ones((3, 3), bool))
        touch = base_lab[sl][grown & (base_lab[sl] > 0)]
        comps = [int(c) for c in np.unique(touch)]
        if len(comps) < 2:
            continue

        r0, c0 = sl[0].start, sl[1].start
        blob_full = np.zeros((h, w), dtype=bool)
        blob_full[sl] = blob
        radius = float(np.median(edt_add[blob_full])) if blob_full.any() else 1.0
        radius = max(radius, 0.5)

        # nearest blob pixel to each touched base component
        seeds: Dict[int, Tuple[int, int]] = {}
        bl_rr, bl_cc = np.nonzero(blob_full)
        blob_pts = np.stack([bl_rr, bl_cc], axis=1)
        for c in comps:
            comp_mask = base_lab == c
            d = ndi.distance_transform_edt(~comp_mask)
            i = int(np.argmin(d[bl_rr, bl_cc]))
            seeds[c] = (int(blob_pts[i, 0]), int(blob_pts[i, 1]))

        anchor = comps[0]
        for c in comps[1:]:
            path = _order_blob_path(blob_full, seeds[anchor], seeds[c])
            if path.shape[0] < 1:
                continue
            edges.append(CandidateEdge(
                path=path, radius=radius,
                edge_id=f"added{k}_{anchor}-{c}",
                meta=dict(source="post_hoc", blob=int(k),
                          comp_a=int(anchor), comp_b=int(c),
                          n_added_px=int(blob_full.sum())),
            ))
    return edges


# --------------------------------------------------------------------------
# method registry -- filled by the three baseline modules on import
# --------------------------------------------------------------------------

METHODS = ("geometric", "rnca", "evapore_e2e", "evapore_scorer")
"""The four scoreable repair baselines.

``evapore_e2e`` and ``evapore_scorer`` share one checkpoint and differ only in
*which candidates are scored and how acceptance is arbitrated* -- see
:mod:`src.baselines.evapore.evapore_adapter`.  The bare name ``evapore`` is
accepted as a deprecated alias for ``evapore_e2e``.
"""


def get_method(name: str):
    """Return the ``repair_detailed`` callable of one baseline (lazy import)."""
    key = name.lower().replace("-", "_")
    if key == "geometric":
        from src.baselines.geometric_repair import repair_detailed
        return repair_detailed
    if key == "rnca":
        from src.baselines.rnca.rnca_adapter import repair_detailed
        return repair_detailed
    if key in ("evapore", "evapore_e2e"):
        from src.baselines.evapore.evapore_adapter import repair_e2e
        return repair_e2e
    if key == "evapore_scorer":
        from src.baselines.evapore.evapore_adapter import repair_scorer
        return repair_scorer
    raise ValueError(f"unknown baseline {name!r}; available: {METHODS}")
