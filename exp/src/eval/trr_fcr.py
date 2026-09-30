"""Formal true-repair criterion, TRR and FCR (proposal v3 §4.2).

An accepted candidate edge ``e`` (a rasterised path that RiGR proposes to add
to the predicted mask) is a **TRUE repair** iff all three predefined criteria
hold:

  (i)  **graph match** -- both endpoints map onto the *same* continuous GT path:
       each endpoint has a GT-skeleton pixel within ``tol_px``, and a geodesic
       on the GT skeleton graph exists between the two mapped pixels with
       length ``<= kappa * euclid`` (``kappa = 3`` by default, ``euclid`` being
       the straight-line distance between the two mapped pixels, floored at
       1 px).  A pair of endpoints that map to different GT components has no
       geodesic and therefore fails.
  (ii) **spatial corridor** -- at least ``overlap_frac`` (default 80%) of the
       path pixels lie inside the GT lumen dilated by
       ``max(min_dilate, radius_frac * local_radius)`` px, where the local
       radius is ``EDT(gt)`` at the nearest GT pixel (defaults: 2 px, 0.5).
  (iii) **connectivity restoration** -- adding the tube of ``e`` to the
       predicted mask actually merges the two sides: the components hosting the
       two endpoints differ before and coincide after, or ``beta0`` decreases.

Event bookkeeping.  ``count_repairable_events`` enumerates the *repairable
events* of an image: places where a GT skeleton branch leaves one predicted
component, crosses a gap of predicted background, and re-enters a **different**
predicted component.  Each such gap is one event.  ``trr_fcr`` then matches
true-repair edges to events **one-to-one** (each event at most one edge, each
edge at most one event) so that

    TRR_recall    = m / N_should              (event level)
    TRR_precision = |A_true| / |A|            (edge level)
    FCR           = 1 - TRR_precision         (edge level)

Conventions for degenerate inputs:
  * ``|A| == 0`` (nothing accepted) -> ``TRR_precision = 1.0`` and ``FCR = 0.0``
    (making no connection cannot make a false connection); ``n_accepted`` is
    returned so such working points can be filtered out of a curve.
  * ``N_should == 0`` -> ``TRR_recall = nan``.

Pure numpy / scipy / skimage; no GPU.  Run
``cd exp && python -m src.eval.trr_fcr`` for a smoke demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from src.topo import skeleton as sk

__all__ = [
    "CandidateEdge",
    "matched_precision_fcr",
    "rasterize_path",
    "rasterize_tube",
    "gt_corridor_band",
    "SkeletonGeodesic",
    "classify_edges",
    "count_repairable_events",
    "trr_fcr",
]

_NAN = float("nan")

# 8-neighbour offsets used to build the GT skeleton graph, with edge weights.
_OFFSETS = ((0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0)))


# --------------------------------------------------------------------------
# candidate edge container
# --------------------------------------------------------------------------
@dataclass
class CandidateEdge:
    """One candidate connection proposed by RiGR.

    Attributes
    ----------
    path : (N, 2) int array
        Rasterised path pixels ``(row, col)``, ordered from one endpoint to the
        other (the A* solution of proposal §3.2.3, or a straight line for the
        geometric baseline).
    p0, p1 : (row, col), optional
        The two endpoints (or endpoint + attachment port).  Default to
        ``path[0]`` and ``path[-1]``.
    radius : float
        Radius of the tube painted around the path when the edge is applied.
    edge_id : any
        Caller-side identifier, echoed back in the classification records.
    score : float
        Calibrated acceptance probability / utility, carried through for sweeps.
    """

    path: np.ndarray
    p0: Optional[Tuple[int, int]] = None
    p1: Optional[Tuple[int, int]] = None
    radius: float = 1.0
    edge_id: Any = None
    score: float = _NAN
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.path = np.asarray(self.path, dtype=np.int64).reshape(-1, 2)
        if self.path.shape[0] == 0:
            raise ValueError("CandidateEdge.path is empty")
        if self.p0 is None:
            self.p0 = (int(self.path[0, 0]), int(self.path[0, 1]))
        if self.p1 is None:
            self.p1 = (int(self.path[-1, 0]), int(self.path[-1, 1]))
        self.p0 = (int(self.p0[0]), int(self.p0[1]))
        self.p1 = (int(self.p1[0]), int(self.p1[1]))


# --------------------------------------------------------------------------
# rasterisation helpers
# --------------------------------------------------------------------------
def rasterize_path(points: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Rasterise a polyline of ``(row, col)`` waypoints into 8-connected pixels."""
    from skimage.draw import line

    pts = np.asarray(points, dtype=np.int64).reshape(-1, 2)
    if pts.shape[0] == 1:
        return pts
    chunks = []
    for a, b in zip(pts[:-1], pts[1:]):
        rr, cc = line(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        chunks.append(np.stack([rr, cc], axis=1))
    out = np.concatenate(chunks, axis=0)
    # drop consecutive duplicates at the segment joins, keep the order
    keep = np.ones(len(out), dtype=bool)
    keep[1:] = np.any(out[1:] != out[:-1], axis=1)
    return out[keep]


def rasterize_tube(path: np.ndarray, radius: float, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean mask of the path dilated by a disk of ``radius`` pixels."""
    from skimage.morphology import disk

    m = np.zeros(shape, dtype=bool)
    p = np.asarray(path, dtype=np.int64).reshape(-1, 2)
    inside = (
        (p[:, 0] >= 0) & (p[:, 0] < shape[0]) & (p[:, 1] >= 0) & (p[:, 1] < shape[1])
    )
    p = p[inside]
    if p.size == 0:
        return m
    m[p[:, 0], p[:, 1]] = True
    r = int(max(0, round(float(radius))))
    if r > 0:
        m = ndi.binary_dilation(m, structure=disk(r).astype(bool))
    return m


def gt_corridor_band(
    gt_mask: np.ndarray, min_dilate: float = 2.0, radius_frac: float = 0.5
) -> np.ndarray:
    """The "GT lumen dilated band" of criterion (ii), as a boolean mask.

    A pixel belongs to the band iff its distance to the GT foreground is at most
    ``max(min_dilate, radius_frac * r)`` where ``r = EDT(gt)`` at the *nearest*
    GT pixel, i.e. the local vessel radius of the branch the pixel is closest
    to.  This makes the tolerance scale with calibre: thin peripheral vessels
    get the 2 px floor, thick arcade vessels get half their radius.
    """
    g = sk.as_bool(gt_mask)
    if not g.any():
        return np.zeros(g.shape, dtype=bool)
    radius = sk.local_radius(g)
    dist, idx = ndi.distance_transform_edt(~g, return_indices=True)
    nearest_r = radius[idx[0], idx[1]]
    tol = np.maximum(float(min_dilate), float(radius_frac) * nearest_r)
    return dist <= tol


# --------------------------------------------------------------------------
# geodesics on the GT skeleton
# --------------------------------------------------------------------------
class SkeletonGeodesic:
    """Shortest paths along a 1-pixel-wide skeleton (8-connected, metric edges).

    Edge weights are 1 for the 4-neighbours and sqrt(2) for the diagonals, so a
    geodesic length is in pixels and directly comparable to a Euclidean
    distance.  The adjacency matrix is built once; ``distance`` runs a single
    Dijkstra with a ``limit`` so per-edge queries stay cheap.
    """

    def __init__(self, skel: np.ndarray):
        s = sk.as_bool(skel)
        self.shape = s.shape
        self.mask = s
        lin = np.flatnonzero(s)
        self.n_nodes = lin.size
        self.node_of = np.full(s.size, -1, dtype=np.int64)
        self.node_of[lin] = np.arange(lin.size)
        node_map = self.node_of.reshape(s.shape)

        H, W = s.shape
        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        vals: List[np.ndarray] = []
        for dr, dc, w in _OFFSETS:
            src = (
                slice(max(0, -dr), H - max(0, dr)),
                slice(max(0, -dc), W - max(0, dc)),
            )
            dst = (
                slice(max(0, dr), H - max(0, -dr)),
                slice(max(0, dc), W - max(0, -dc)),
            )
            both = s[src] & s[dst]
            if not both.any():
                continue
            a = node_map[src][both]
            b = node_map[dst][both]
            rows.append(a)
            cols.append(b)
            vals.append(np.full(a.size, w, dtype=np.float64))
        if rows:
            r = np.concatenate(rows)
            c = np.concatenate(cols)
            v = np.concatenate(vals)
            g = coo_matrix((v, (r, c)), shape=(self.n_nodes, self.n_nodes)).tocsr()
            self.graph = g + g.T
        else:
            self.graph = coo_matrix((self.n_nodes, self.n_nodes)).tocsr()

        # nearest skeleton pixel for arbitrary query points
        if self.n_nodes:
            self._dist, self._idx = ndi.distance_transform_edt(~s, return_indices=True)
        else:
            self._dist, self._idx = None, None

    def snap(self, point, tol: float):
        """Nearest skeleton pixel to ``point`` within ``tol``; ``None`` if none."""
        if self._dist is None:
            return None
        r = int(np.clip(round(point[0]), 0, self.shape[0] - 1))
        c = int(np.clip(round(point[1]), 0, self.shape[1] - 1))
        if float(self._dist[r, c]) > float(tol):
            return None
        return (int(self._idx[0][r, c]), int(self._idx[1][r, c]))

    def distance(self, a, b, limit: float = np.inf) -> float:
        """Geodesic length between two skeleton pixels (``inf`` if unreachable)."""
        if self.n_nodes == 0:
            return float("inf")
        na = int(self.node_of[a[0] * self.shape[1] + a[1]])
        nb = int(self.node_of[b[0] * self.shape[1] + b[1]])
        if na < 0 or nb < 0:
            return float("inf")
        if na == nb:
            return 0.0
        d = dijkstra(self.graph, directed=False, indices=na, limit=float(limit))
        return float(d[nb])


# --------------------------------------------------------------------------
# criterion evaluation
# --------------------------------------------------------------------------
def _component_at(labels: np.ndarray, point, search_px: float = 2.0) -> int:
    """Label of the component hosting ``point``, searching ``search_px`` around."""
    H, W = labels.shape
    r = int(np.clip(round(point[0]), 0, H - 1))
    c = int(np.clip(round(point[1]), 0, W - 1))
    v = int(labels[r, c])
    if v > 0:
        return v
    rad = int(max(1, round(search_px)))
    r0, r1 = max(0, r - rad), min(H, r + rad + 1)
    c0, c1 = max(0, c - rad), min(W, c + rad + 1)
    win = labels[r0:r1, c0:c1]
    nz = np.nonzero(win)
    if nz[0].size == 0:
        return 0
    d = (nz[0] + r0 - r) ** 2 + (nz[1] + c0 - c) ** 2
    k = int(np.argmin(d))
    if d[k] > rad * rad:
        return 0
    return int(win[nz[0][k], nz[1][k]])


def classify_edges(
    edges: Iterable[CandidateEdge],
    base_mask: np.ndarray,
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    gt_skel: Optional[np.ndarray] = None,
    tol_px: float = 3.0,
    kappa: float = 3.0,
    overlap_frac: float = 0.80,
    min_dilate: float = 2.0,
    radius_frac: float = 0.5,
    endpoint_search_px: float = 2.0,
    geodesic: Optional[SkeletonGeodesic] = None,
    band: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """Classify candidate edges as true repairs by the three-part criterion.

    Parameters
    ----------
    edges : iterable of :class:`CandidateEdge`
        The *accepted* edges (set ``A``) whose paths are already rasterised.
    base_mask : (H, W) bool
        The base predicted mask ``M_hat`` **before** any repair.
    gt_mask : (H, W) bool
        Reference vessel mask.
    fov : (H, W) bool, optional
        Everything is restricted to the FOV first.
    gt_skel : (H, W) bool, optional
        Pre-computed GT skeleton (spur-pruned).  Computed if omitted.
    tol_px, kappa, overlap_frac, min_dilate, radius_frac
        Criterion parameters, see the module docstring.
    geodesic, band
        Optional pre-computed helpers, to amortise cost over many edges.

    Returns
    -------
    list of dict, one per edge, in input order, with keys
    ``edge_id, score, crit_graph, crit_corridor, crit_connect, is_true_repair,``
    ``geodesic_len, euclid, overlap, comp_src, comp_dst, comp_merged,``
    ``beta0_before, beta0_after``.
    """
    base = sk.as_bool(base_mask)
    gt = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, base.shape)
    base = base & f
    gt = gt & f

    edges = list(edges)
    if not edges:
        return []

    if gt_skel is None:
        gt_skel = sk.skeletonize_mask(gt)
    else:
        gt_skel = sk.as_bool(gt_skel) & f
    if geodesic is None:
        geodesic = SkeletonGeodesic(gt_skel)
    if band is None:
        band = gt_corridor_band(gt, min_dilate=min_dilate, radius_frac=radius_frac)

    labels, n0 = sk.connected_components(base, connectivity=8)
    b0_before, _ = sk.betti_numbers(base, f)

    out: List[Dict[str, Any]] = []
    for e in edges:
        rec: Dict[str, Any] = {
            "edge_id": e.edge_id,
            "score": float(e.score),
            "crit_graph": False,
            "crit_corridor": False,
            "crit_connect": False,
            "is_true_repair": False,
            "geodesic_len": _NAN,
            "euclid": _NAN,
            "overlap": _NAN,
            "comp_src": 0,
            "comp_dst": 0,
            "comp_merged": False,
            "beta0_before": float(b0_before),
            "beta0_after": _NAN,
        }

        # ---- (i) graph match -------------------------------------------------
        m0 = geodesic.snap(e.p0, tol_px)
        m1 = geodesic.snap(e.p1, tol_px)
        if m0 is not None and m1 is not None:
            eu = float(np.hypot(m0[0] - m1[0], m0[1] - m1[1]))
            eu = max(eu, 1.0)
            rec["euclid"] = eu
            limit = float(kappa) * eu
            gd = geodesic.distance(m0, m1, limit=limit)
            rec["geodesic_len"] = gd
            rec["crit_graph"] = bool(np.isfinite(gd) and gd <= limit)

        # ---- (ii) spatial corridor ------------------------------------------
        p = e.path
        inside_img = (
            (p[:, 0] >= 0)
            & (p[:, 0] < base.shape[0])
            & (p[:, 1] >= 0)
            & (p[:, 1] < base.shape[1])
        )
        if inside_img.any():
            pin = p[inside_img]
            ov = float(np.count_nonzero(band[pin[:, 0], pin[:, 1]])) / float(len(p))
            rec["overlap"] = ov
            rec["crit_corridor"] = bool(ov >= float(overlap_frac))

        # ---- (iii) connectivity restoration ---------------------------------
        cs = _component_at(labels, e.p0, endpoint_search_px)
        cd = _component_at(labels, e.p1, endpoint_search_px)
        rec["comp_src"] = int(cs)
        rec["comp_dst"] = int(cd)
        tube = rasterize_tube(e.path, e.radius, base.shape) & f
        new_mask = base | tube
        b0_after, _ = sk.betti_numbers(new_mask, f)
        rec["beta0_after"] = float(b0_after)
        new_labels, _ = sk.connected_components(new_mask, connectivity=8)
        ns = _component_at(new_labels, e.p0, endpoint_search_px)
        nd = _component_at(new_labels, e.p1, endpoint_search_px)
        merged = bool(cs != cd and ns != 0 and ns == nd)
        rec["comp_merged"] = merged
        rec["crit_connect"] = bool(merged or b0_after < b0_before)

        rec["is_true_repair"] = bool(
            rec["crit_graph"] and rec["crit_corridor"] and rec["crit_connect"]
        )
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# repairable events
# --------------------------------------------------------------------------
def count_repairable_events(
    base_mask: np.ndarray,
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    gt_skel: Optional[np.ndarray] = None,
    min_gap_px: int = 1,
    snap_px: float = 1.0,
) -> List[Dict[str, Any]]:
    """Enumerate the repairable events (``G``, ``N_should = len(...)``).

    A *repairable event* is a gap of predicted background crossed by a GT
    skeleton branch whose two flanks lie in **different** predicted components:
    the prediction split a vessel that the reference says is continuous, and a
    single added edge could put it back.

    Implementation.  Restrict the (spur-pruned) GT skeleton to the FOV, label
    the predicted mask (8-connected), and drop the GT skeleton pixels that fall
    inside predicted foreground.  What remains are the *gap runs*; label them
    8-connected.  For each run, collect the predicted-component labels of the
    GT-skeleton pixels 8-adjacent to it.  Runs flanked by ``>= 2`` distinct
    components are events; a run flanked by one component (a dent) or none (a
    completely missed vessel with no anchor) is not.

    A gap that sits on a bifurcation and touches three components counts as a
    **single** event, with all its component labels recorded, because one added
    edge is expected to be matched to it.

    ``snap_px`` (default 1 px) absorbs the sub-pixel offset between the GT
    centreline and the predicted lumen: a GT skeleton pixel is considered to be
    "on" a predicted component if that component is within ``snap_px``.  Without
    it a correctly detected vessel whose GT centreline runs one pixel beside the
    predicted one would be shredded into dozens of spurious 1-px events.

    Returns
    -------
    list of dict with keys ``gap_pixels`` ((k,2) int), ``centroid`` ((2,) float),
    ``components`` (sorted list of int), ``flanks`` (dict component -> (row,col)),
    ``length`` (number of gap skeleton pixels).
    """
    base = sk.as_bool(base_mask)
    gt = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, base.shape)
    base = base & f
    gt = gt & f

    if gt_skel is None:
        gt_skel = sk.skeletonize_mask(gt)
    else:
        gt_skel = sk.as_bool(gt_skel) & f
    if not gt_skel.any():
        return []

    labels, _ = sk.connected_components(base, connectivity=8)
    if snap_px and snap_px > 0 and base.any():
        dist, idx = ndi.distance_transform_edt(~base, return_indices=True)
        snapped = np.where(dist <= float(snap_px), labels[idx[0], idx[1]], 0)
    else:
        snapped = labels
    labels = snapped
    on_pred = gt_skel & (labels > 0)
    gap = gt_skel & ~on_pred
    if not gap.any():
        return []

    gap_lab, n_gap = ndi.label(gap, structure=sk.EIGHT)
    struct = sk.EIGHT.astype(bool)
    events: List[Dict[str, Any]] = []
    objs = ndi.find_objects(gap_lab)
    H, W = base.shape
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        r0 = max(0, sl[0].start - 2)
        r1 = min(H, sl[0].stop + 2)
        c0 = max(0, sl[1].start - 2)
        c1 = min(W, sl[1].stop + 2)
        win = (slice(r0, r1), slice(c0, c1))
        run = gap_lab[win] == (i + 1)
        n_px = int(run.sum())
        if n_px < int(min_gap_px):
            continue
        flank = ndi.binary_dilation(run, structure=struct) & on_pred[win]
        if not flank.any():
            continue
        fl_r, fl_c = np.nonzero(flank)
        fl_lab = labels[win][fl_r, fl_c]
        comps = np.unique(fl_lab)
        comps = comps[comps > 0]
        if comps.size < 2:
            continue
        flanks = {}
        for cmp_id in comps:
            k = int(np.argmax(fl_lab == cmp_id))
            flanks[int(cmp_id)] = (int(fl_r[k] + r0), int(fl_c[k] + c0))
        rr, cc = np.nonzero(run)
        pix = np.stack([rr + r0, cc + c0], axis=1)
        events.append(
            {
                "gap_pixels": pix,
                "centroid": pix.mean(axis=0),
                "components": sorted(int(x) for x in comps),
                "flanks": flanks,
                "length": n_px,
            }
        )
    return events


# --------------------------------------------------------------------------
# TRR / FCR
# --------------------------------------------------------------------------
def _match_edges_to_events(
    edges: Sequence[CandidateEdge],
    records: Sequence[Dict[str, Any]],
    events: Sequence[Dict[str, Any]],
    match_px: float = 12.0,
) -> List[Tuple[int, int]]:
    """One-to-one matching of true-repair edges to repairable events.

    An edge and an event are compatible when the edge's path passes within
    ``match_px`` of the event's gap pixels **and** the edge joins two of the
    components the event separates (when the edge's component labels are known;
    an edge whose endpoints could not be assigned a component falls back to the
    distance test alone).  Among the compatible pairs the assignment minimising
    the total edge-to-gap distance is taken (Hungarian).
    """
    true_idx = [i for i, r in enumerate(records) if r["is_true_repair"]]
    if not true_idx or not events:
        return []

    n_a, n_g = len(true_idx), len(events)
    big = float(match_px) * 100.0 + 1.0
    cost = np.full((n_a, n_g), big, dtype=np.float64)
    for ai, i in enumerate(true_idx):
        path = np.asarray(edges[i].path, dtype=float)
        comps = {records[i]["comp_src"], records[i]["comp_dst"]} - {0}
        for gi, ev in enumerate(events):
            gp = np.asarray(ev["gap_pixels"], dtype=float)
            d = np.min(
                np.hypot(
                    path[:, None, 0] - gp[None, :, 0], path[:, None, 1] - gp[None, :, 1]
                )
            )
            if d > match_px:
                continue
            if comps and len(comps & set(ev["components"])) < 2:
                continue
            cost[ai, gi] = float(d)

    from scipy.optimize import linear_sum_assignment

    ri, ci = linear_sum_assignment(cost)
    return [
        (true_idx[int(a)], int(g)) for a, g in zip(ri, ci) if cost[int(a), int(g)] < big
    ]


def matched_precision_fcr(n_matched, n_accepted):
    """Event-matched precision ``m / |A|`` and its FCR, with the |A| = 0 rule.

    The edge-level precision ``|A_true| / |A|`` counts an accepted edge as
    good if it satisfies the three-part true-repair criterion.  The *matched*
    precision instead counts only edges that were matched to a repairable
    event, so it is the exact complement of the recall numerator: with
    ``TRR_recall = m / N_should`` and ``TRR_precision_matched = m / |A|`` the
    two share ``m`` and form a genuine precision/recall pair.  That is why
    DECISIONS.md 2026-09-03 10:20 makes it the primary reported figure.

    Convention for an empty acceptance set (``|A| = 0``): **undefined, NaN**.
    Precision is ``m / |A|``, which is 0/0 when nothing was accepted -- there
    is no false-connection *rate* to report because there were no connections.
    This used to return precision 1.0 / FCR 0.0, i.e. "connects nothing =
    perfect precision", and because those zeros were then averaged in like any
    other image they dragged whole-dataset FCR means downward in proportion to
    how often a method declined to act, and planted a spurious "FCR 0, perfect
    precision" anchor at the left end of every operating curve (which is what
    made the pre-registered 1/2.5/5 % Fig.3 targets look reachable when they
    are not).  Review r3, DECISIONS.md 2026-09-16 15:00.

    Callers must therefore exclude NaN from FCR/precision means and report how
    many images were defined (``n_fcr_defined``).  ``TRR_recall`` is
    unaffected: its denominator is ``N_should``, which does not depend on what
    the method accepted.

    Returns ``(precision, fcr, is_empty)``; ``is_empty`` still flags the case.
    """
    na = int(n_accepted)
    nm = int(n_matched)
    if na <= 0:
        return float("nan"), float("nan"), True
    p = float(min(nm, na)) / float(na)
    return p, 1.0 - p, False


def trr_fcr(
    accepted_edges: Iterable[CandidateEdge],
    base_mask: np.ndarray,
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    gt_skel: Optional[np.ndarray] = None,
    events: Optional[Sequence[Dict[str, Any]]] = None,
    classifications: Optional[Sequence[Dict[str, Any]]] = None,
    match_px: float = 12.0,
    return_details: bool = False,
    **criteria,
) -> Dict[str, Any]:
    """True Repair Rate and False Connection Rate for one image.

    ``criteria`` is forwarded to :func:`classify_edges` (``tol_px``, ``kappa``,
    ``overlap_frac``, ``min_dilate``, ``radius_frac``).

    Returns
    -------
    dict with ``TRR_recall`` (= m / N_should, event level),
    ``TRR_precision_matched`` (= m / |A|) and ``FCR_matched`` (= 1 - that) --
    the **primary** pair, see :func:`matched_precision_fcr` --
    ``TRR_precision`` (= |A_true| / |A|, edge level) and ``FCR``
    (= 1 - TRR_precision) as the secondary pair, ``n_should`` (= N_should),
    ``n_accepted`` (= |A|), ``n_true`` (= |A_true|), ``n_matched`` (= m) and
    ``no_accepted_edges`` (the |A| = 0 flag).  With ``return_details=True`` the
    per-edge records, the events and the matching are added under
    ``records`` / ``events`` / ``matches``.
    """
    base = sk.as_bool(base_mask)
    gt = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, base.shape)
    if gt_skel is None:
        gt_skel = sk.skeletonize_mask(gt & f)

    edges = list(accepted_edges)
    if classifications is None:
        classifications = classify_edges(
            edges, base, gt, fov=f, gt_skel=gt_skel, **criteria
        )
    if events is None:
        events = count_repairable_events(base, gt, fov=f, gt_skel=gt_skel)

    n_accepted = len(edges)
    n_true = int(sum(1 for r in classifications if r["is_true_repair"]))
    n_should = len(events)

    matches = _match_edges_to_events(edges, classifications, events, match_px=match_px)
    m = len(matches)

    if n_accepted == 0:
        precision = 1.0
        fcr = 0.0
    else:
        precision = float(n_true) / float(n_accepted)
        fcr = 1.0 - precision
    recall = float(m) / float(n_should) if n_should > 0 else _NAN

    prec_m, fcr_m, empty = matched_precision_fcr(m, n_accepted)
    out: Dict[str, Any] = {
        "TRR_recall": float(recall),
        # primary (DECISIONS.md 2026-09-03 10:20): matched precision / FCR
        "TRR_precision_matched": float(prec_m),
        "FCR_matched": float(fcr_m),
        # secondary: the edge-level criterion, kept for continuity
        "TRR_precision": float(precision),
        "FCR": float(fcr),
        "n_should": int(n_should),
        "n_accepted": int(n_accepted),
        "n_true": int(n_true),
        "n_matched": int(m),
        "no_accepted_edges": int(empty),
    }
    if return_details:
        out["records"] = list(classifications)
        out["events"] = list(events)
        out["matches"] = matches
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - smoke demo
    H = W = 96
    gt = np.zeros((H, W), dtype=bool)
    gt[46:50, 8:88] = True
    base = gt.copy()
    base[:, 44:56] = False  # a 12 px gap

    edge = CandidateEdge(
        path=rasterize_path([(47, 43), (47, 56)]), radius=1.0, edge_id="good", score=0.9
    )
    res = trr_fcr([edge], base, gt, return_details=True)
    print("true repair :", {k: v for k, v in res.items() if not isinstance(v, list)})
    print("record      :", res["records"][0])
