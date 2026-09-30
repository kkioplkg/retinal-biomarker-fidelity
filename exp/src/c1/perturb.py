"""Algorithm 1: vascular structural perturbations + matched topology-neutral controls.

Reference
---------
proposal/03_full_proposal_v3.md section 3.1.3 (Algorithm 1), section 3.1.1
(Contract F), section 3.1.2 (signed / net / deployment harm) and
exp/EXPERIMENT_PLAN.md S3.1.

Everything in this module operates on a **ground-truth** mask ``M`` and is
deterministic given a seed.  Four perturbation families are implemented:

``sever``      connectivity loss -- a capsule (stadium) of length
               ``L = L_factor * D_loc`` and radius ``r_loc + 1`` is removed
               along the local tangent, cutting the whole lumen.
               ``L_factor in {0.5, 1, 2, 4}``.  Verified: ``beta0`` inside the
               FOV increases by exactly 1, **or** the two sides of the branch
               end up in different connected components.
``bridge``     shortcut / false connection -- a tangent-constrained cubic
               Bezier tube of radius ``(r_a + r_b)/2`` between two skeleton
               points on different branches.  Verified: components merged, or
               a new cycle (``beta1 += 1``), or a >= 3x graph-geodesic
               shortcut.
``truncate``   terminal loss -- the terminal ``rho in {0.10, 0.25, 0.50}`` of a
               terminal branch is removed tube-aware (Voronoi ownership of the
               lumen by the centreline).  Verified: ``beta0`` unchanged.
``caliber``    topology-neutral control arm -- a branch tube is eroded /
               dilated by 1 px.  Verified: every graph invariant unchanged.

``matched_control`` builds the pixel- **and** context-matched topology-neutral
control ``M_C`` by rejection sampling (up to 50 tries): it removes / adds
exactly ``N_T = |M' xor M|`` pixels (+-2) at a donor site matched on radius
bin, retinal zone (A/B/C w.r.t. the automatically located optic disc), GT
branch order, local vessel density bin and local contrast bin, and only accepts
a donor whose graph invariants (``beta0``, ``beta1``, #endpoints, #junctions,
skan branch count) are identical to those of ``M``.

``phi`` is the **Contract F** feature extractor: it sees only the image ``I``,
the *perturbed* mask ``M'`` and the automatically located disc -- never the
ground truth.  Ground-truth-only stratification variables (GT branch order, GT
branch length, true radius bin, ...) are returned by ``gt_strata`` in a
*separate* dict and must never be mixed into ``phi``.
"""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from ..topo import skeleton as SK

__all__ = [
    "L_FACTORS",
    "TRUNCATE_RHOS",
    "CALIBER_FACTORS",
    "Locus",
    "BridgePair",
    "PerturbResult",
    "Invariants",
    "GTContext",
    "make_rng",
    "graph_invariants",
    "window_invariants",
    "sever",
    "bridge",
    "bridge_candidates",
    "truncate",
    "caliber",
    "matched_control",
    "phi",
    "gt_strata",
    "new_endpoints",
    "PHI_COLUMNS",
    "STRATA_COLUMNS",
]

# --------------------------------------------------------------------------- #
# pre-registered severity ladders (proposal section 3.1.3)
# --------------------------------------------------------------------------- #
L_FACTORS: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
TRUNCATE_RHOS: Tuple[float, ...] = (0.10, 0.25, 0.50)
CALIBER_FACTORS: Tuple[int, ...] = (-1, 1)

#: structural-quality gate (Algorithm 1 lines 3-5)
MIN_JUNCTION_CLEARANCE = 2.5      # x D_loc
MIN_ENDPOINT_CLEARANCE = 1.5      # x D_loc
AMBIGUITY_RADIUS = 1.5            # x D_loc
MAX_RADIUS_CV = 0.35              # local radius stability along the capsule
MIN_FOV_MARGIN = 6.0              # px, on top of 2 x D_loc

#: bridge candidate gate (proposal section 3.1.3 lines 13-18 / section 3.2.1)
BRIDGE_MAX_D_OVER_R = 6.0         # d <= 6 * mean radius
BRIDGE_MAX_LOG_RATIO = math.log(2.0)
BRIDGE_MIN_GEODESIC = 8.0         # x D
BRIDGE_CROSSING_CLEARANCE = 2.0   # x D from a degree-4 GT junction
BRIDGE_SHORTCUT_FACTOR = 3.0

#: matched control (Algorithm 1 lines 23-29)
CONTROL_MAX_TRIES = 50
CONTROL_PIXEL_TOL = 2

#: a lumen thinner than this (radius, px) cannot be eroded topology-neutrally
MIN_ERODIBLE_RADIUS = 2.0

#: floor on a deletion-control donor's radius.  Kept low so that thin targets
#: can still find a *radius-bin matched* donor (a high floor forces every thin
#: target down to the unmatched tier); donors that cannot survive the erosion
#: are rejected by the graph-invariant test anyway.
MIN_CONTROL_DONOR_RADIUS = 1.2

#: zone boundaries in disc *radii* (zone A <= 2 r_d, B <= 3 r_d, C beyond);
#: identical to the annulus PVBM uses for its zone-B biomarkers.
ZONE_A_R = 2.0
ZONE_B_R = 3.0

#: radius bins in pixels (at the working resolution)
RADIUS_BIN_EDGES = (1.5, 2.5, 4.0)


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def _stable_hash(*parts: object) -> int:
    s = "|".join(str(p) for p in parts).encode("utf-8")
    return int(zlib.crc32(s) & 0xFFFFFFFF)


def make_rng(seed: int, *parts: object) -> np.random.Generator:
    """Deterministic generator keyed by ``seed`` and any number of string parts."""
    return np.random.default_rng([int(seed), _stable_hash(*parts)])


# --------------------------------------------------------------------------- #
# small geometry helpers
# --------------------------------------------------------------------------- #
def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.hypot(v[0], v[1]))
    if n < 1e-9:
        return np.array([1.0, 0.0])
    return v / n


def _win(shape, r: float, c: float, pad: float):
    """Bounding-box slices of a ``pad``-radius window around ``(r, c)``."""
    r0 = max(0, int(math.floor(r - pad)))
    r1 = min(shape[0], int(math.ceil(r + pad)) + 1)
    c0 = max(0, int(math.floor(c - pad)))
    c1 = min(shape[1], int(math.ceil(c + pad)) + 1)
    return slice(r0, max(r1, r0 + 1)), slice(c0, max(c1, c0 + 1))


def _win_of_points(shape, pts: np.ndarray, pad: float):
    r0 = max(0, int(math.floor(pts[:, 0].min() - pad)))
    r1 = min(shape[0], int(math.ceil(pts[:, 0].max() + pad)) + 1)
    c0 = max(0, int(math.floor(pts[:, 1].min() - pad)))
    c1 = min(shape[1], int(math.ceil(pts[:, 1].max() + pad)) + 1)
    return slice(r0, max(r1, r0 + 1)), slice(c0, max(c1, c0 + 1))


def capsule_region(shape, centre, tangent, half_len: float, radius: float):
    """``(slices, local_bool)`` of the stadium of half-length ``half_len``.

    The stadium is ``{x : dist(x, segment) <= radius}`` where the segment runs
    from ``centre - half_len * t`` to ``centre + half_len * t``.
    """
    cr, cc = float(centre[0]), float(centre[1])
    t = _unit(tangent)
    pad = half_len + radius + 2.0
    sl = _win(shape, cr, cc, pad)
    rr, cc_ = np.mgrid[sl[0], sl[1]]
    dy = rr - cr
    dx = cc_ - cc
    proj = np.clip(dy * t[0] + dx * t[1], -half_len, half_len)
    py = dy - proj * t[0]
    px = dx - proj * t[1]
    return sl, (py * py + px * px) <= radius * radius


def _bezier_points(p0, p1, p2, p3, n: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, int(n))[:, None]
    mt = 1.0 - t
    p0 = np.asarray(p0, float)[None, :]
    p1 = np.asarray(p1, float)[None, :]
    p2 = np.asarray(p2, float)[None, :]
    p3 = np.asarray(p3, float)[None, :]
    return (mt ** 3) * p0 + 3 * (mt ** 2) * t * p1 + 3 * mt * (t ** 2) * p2 + (t ** 3) * p3


def _polyline_length(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.0
    d = np.diff(pts, axis=0)
    return float(np.hypot(d[:, 0], d[:, 1]).sum())


def _tube_from_points(shape, pts: np.ndarray, radius: float):
    """``(slices, local_bool)`` of ``{x : dist(x, polyline) <= radius}``."""
    pad = radius + 3.0
    sl = _win_of_points(shape, pts, pad)
    h = sl[0].stop - sl[0].start
    w = sl[1].stop - sl[1].start
    seed = np.zeros((h, w), dtype=bool)
    rr = np.clip(np.rint(pts[:, 0]).astype(int) - sl[0].start, 0, h - 1)
    cc = np.clip(np.rint(pts[:, 1]).astype(int) - sl[1].start, 0, w - 1)
    seed[rr, cc] = True
    dist = ndi.distance_transform_edt(~seed)
    return sl, dist <= radius


# --------------------------------------------------------------------------- #
# graph invariants
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Invariants:
    """The five quantities Algorithm 1 line 29 requires a control to preserve."""

    beta0: int
    beta1: int
    n_endpoints: int
    n_junctions: int
    n_branches: int

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        return {
            f"{prefix}beta0": float(self.beta0),
            f"{prefix}beta1": float(self.beta1),
            f"{prefix}n_endpoints": float(self.n_endpoints),
            f"{prefix}n_junctions": float(self.n_junctions),
            f"{prefix}n_branches": float(self.n_branches),
        }


def graph_invariants(mask: np.ndarray, fov: np.ndarray, min_branch_px: int = 3) -> Invariants:
    """``(beta0, beta1, #endpoints, #junction clusters, #skan branches)``."""
    b0, b1 = SK.betti_numbers(mask, fov)
    skel = SK.skeletonize_mask(mask, min_branch_px=min_branch_px, fov=fov)
    n_end = int(SK.endpoints(skel).shape[0])
    cen, _, _ = SK.junction_clusters(skel)
    g = SK.build_graph(skel)
    n_br = int(g.n_paths) if g is not None else 0
    return Invariants(int(b0), int(b1), n_end, int(cen.shape[0]), n_br)


def betti_in_fov(mask: np.ndarray, fov: np.ndarray) -> Tuple[int, int]:
    return SK.betti_numbers(mask, fov)


#: margin (px) added around a locally-edited region before its invariants are
#: compared.  It must exceed the skeletonisation support of the thickest vessel
#: so that the skeleton just inside the window border is identical in both masks.
INVARIANT_MARGIN = 24


def expand_slices(sl, shape, pad: int):
    """Grow a pair of slices by ``pad`` px, clipped to ``shape``."""
    return (
        slice(max(0, sl[0].start - pad), min(shape[0], sl[0].stop + pad)),
        slice(max(0, sl[1].start - pad), min(shape[1], sl[1].stop + pad)),
    )


def window_invariants(mask: np.ndarray, fov: np.ndarray, sl, min_branch_px: int = 3) -> Invariants:
    """Graph invariants of ``mask`` restricted to the window ``sl``.

    Used to test topology neutrality of a **local** edit cheaply.  It is a valid
    substitute for the global test: two masks that differ only strictly inside
    ``sl`` have identical structure everywhere outside it, so any component,
    hole, endpoint, junction or branch that one has and the other lacks must
    intersect the changed pixels and is therefore visible inside the window.
    Comparing the two window-local counts thus detects exactly the same
    differences as comparing the global ones, as long as the edited pixels are
    kept ``INVARIANT_MARGIN`` px away from the window border (which
    :func:`expand_slices` guarantees).
    """
    return graph_invariants(mask[sl], fov[sl], min_branch_px)


# --------------------------------------------------------------------------- #
# data classes
# --------------------------------------------------------------------------- #
@dataclass
class Locus:
    """One intervention site on the GT centreline."""

    locus_id: str
    row: int
    col: int
    branch: int                 # skan path index
    branch_type: int            # skan code: 0 iso, 1 junc-end, 2 junc-junc, 3 cycle
    r_loc: float                # local radius (EDT), px
    d_loc: float                # local diameter = 2 * r_loc, px
    tangent: Tuple[float, float]
    arc_pos: float              # arc length from the path start, px
    dist_junction: float        # px to the nearest junction pixel
    dist_endpoint: float        # px to the nearest endpoint pixel
    dist_fov: float             # px to the FOV border
    radius_cv: float
    # context bins (used for control matching and for GT stratification)
    radius_bin: int = 0
    zone: int = 0
    branch_order: int = -1
    density: float = float("nan")
    density_bin: int = 0
    contrast: float = float("nan")
    contrast_bin: int = 0
    gt_branch_length: float = float("nan")

    def point(self) -> np.ndarray:
        return np.array([self.row, self.col], dtype=float)


@dataclass
class PerturbResult:
    """Outcome of one perturbation attempt."""

    ok: bool
    reason: str
    mask: Optional[np.ndarray] = None
    n_changed: int = 0
    info: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# helpers used by GTContext
# --------------------------------------------------------------------------- #
def _edt_to(points_mask: np.ndarray, shape) -> np.ndarray:
    """Distance to the nearest True pixel of ``points_mask`` (inf-free)."""
    if not points_mask.any():
        return np.full(shape, 1e6, dtype=np.float32)
    return ndi.distance_transform_edt(~points_mask).astype(np.float32)


def _tercile_edges(x: np.ndarray) -> np.ndarray:
    v = x[np.isfinite(x)]
    if v.size < 6:
        return np.array([np.inf, np.inf])
    return np.percentile(v, [33.333, 66.667])


def _bin_of(v: float, edges: np.ndarray) -> int:
    if not np.isfinite(v):
        return -1
    return int(np.searchsorted(edges, v))


def _free_end_index(co: np.ndarray, end_px: np.ndarray) -> Optional[int]:
    """Index of the degree-1 extremity of a path (0 or last), or None."""
    if end_px[int(co[0, 0]), int(co[0, 1])]:
        return 0
    if end_px[int(co[-1, 0]), int(co[-1, 1])]:
        return co.shape[0] - 1
    return None


def _green(image) -> Optional[np.ndarray]:
    if image is None:
        return None
    a = np.asarray(image)
    if a.ndim == 3:
        a = a[..., 1]
    return a.astype(np.float32)


def image_contrast(image, mask, fov, row: float, col: float, r_loc: float) -> float:
    """Robust green-channel contrast between vessel and background, locally.

    ``(median(bg) - median(vessel)) / (1.4826 * MAD(bg))`` inside a
    ``max(12, 8 * r_loc)`` window.  Vessels are darker than the retina on the
    green channel, so the value is normally positive.  NaN when the window has
    too little vessel or background.
    """
    g = _green(image)
    if g is None:
        return float("nan")
    pad = max(12.0, 8.0 * max(r_loc, 0.5))
    sl = _win(g.shape[:2], row, col, pad)
    gg = g[sl]
    mm = np.asarray(mask)[sl].astype(bool)
    ff = np.asarray(fov)[sl].astype(bool)
    ves = gg[mm & ff]
    bg = gg[(~mm) & ff]
    if ves.size < 5 or bg.size < 20:
        return float("nan")
    mb = float(np.median(bg))
    mad = float(np.median(np.abs(bg - mb)))
    scale = 1.4826 * mad
    if scale < 1e-6:
        scale = float(bg.std()) or 1.0
    return float((mb - float(np.median(ves))) / scale)


# --------------------------------------------------------------------------- #
# GT context
# --------------------------------------------------------------------------- #
class GTContext:
    """Everything derived once per image from the ground-truth mask.

    Attributes
    ----------
    mask, fov, image, disc
    skel : (H, W) bool                 pruned GT skeleton (inside the FOV)
    graph : skan.Skeleton or None
    summary : DataFrame                per-branch table
    radius : (H, W) float32            EDT of the mask ( = local radius )
    branch_label : (H, W) int32        path index + 1 on non-junction skeleton px
    junc_px, end_px, crossing_px : (H, W) bool
    dist_junc, dist_end, dist_fov : (H, W) float32
    branch_order : (n_paths,) int32    disc-rooted BFS depth, -1 if unreachable
    """

    def __init__(
        self,
        mask: np.ndarray,
        fov: np.ndarray,
        image: Optional[np.ndarray] = None,
        disc=None,
        min_branch_px: int = 3,
        seed: int = 0,
        key: str = "",
    ) -> None:
        self.fov = SK.fov_or_true(fov, SK.as_bool(mask).shape)
        self.mask = SK.as_bool(mask) & self.fov
        self.image = image
        self.disc = disc
        self.min_branch_px = int(min_branch_px)
        self.seed = int(seed)
        self.key = key
        self.shape = self.mask.shape

        self.radius = SK.local_radius(self.mask)
        skel = SK.skeletonize_mask(self.mask, min_branch_px=min_branch_px, fov=self.fov)
        self.skel = SK.drop_isolated_pixels(skel)
        self.graph = SK.build_graph(self.skel, self.mask, drop_isolated=False)
        self.summary = SK.summarize(self.skel, self.mask, graph=self.graph)

        deg = SK.neighbour_count(self.skel)
        self.junc_px = self.skel & (deg >= 3)
        self.end_px = self.skel & (deg == 1)

        self.dist_junc = _edt_to(self.junc_px, self.shape)
        self.dist_end = _edt_to(self.end_px, self.shape)
        self.dist_fov = ndi.distance_transform_edt(self.fov).astype(np.float32)

        self.branch_label = np.zeros(self.shape, dtype=np.int32)
        self.path_coords: List[np.ndarray] = []
        self.path_arc: List[np.ndarray] = []
        if self.graph is not None:
            for i in range(int(self.graph.n_paths)):
                co = np.rint(np.asarray(self.graph.path_coordinates(i))).astype(np.int64)
                self.path_coords.append(co)
                if co.shape[0] > 1:
                    d = np.diff(co.astype(float), axis=0)
                    step = np.hypot(d[:, 0], d[:, 1])
                else:
                    step = np.zeros(0)
                self.path_arc.append(np.concatenate([[0.0], np.cumsum(step)]))
                rr, cc = co[:, 0], co[:, 1]
                keep = ~self.junc_px[rr, cc]
                self.branch_label[rr[keep], cc[keep]] = i + 1

        self.junc_centroids, self.junc_labels, self.junc_degrees = SK.junction_clusters(self.skel)
        self.crossing_px = self._crossing_mask()
        self.branch_order = self._branch_orders()
        self._pool: Optional[List[Locus]] = None
        self._ports: Optional[List[Locus]] = None

    # ---------------------------------------------------------------- zones
    def zone_of(self, row: float, col: float) -> int:
        """0 = zone A (<= 1.0 disc diameter), 1 = zone B, 2 = zone C."""
        if self.disc is None:
            return 2
        d = math.hypot(float(col) - float(self.disc.cx), float(row) - float(self.disc.cy))
        rd = max(float(self.disc.r), 1e-6)
        if d <= ZONE_A_R * rd:
            return 0
        if d <= ZONE_B_R * rd:
            return 1
        return 2

    def eccentricity(self, row: float, col: float) -> float:
        """Distance to the disc centre in disc radii (NaN without a disc)."""
        if self.disc is None:
            return float("nan")
        d = math.hypot(float(col) - float(self.disc.cx), float(row) - float(self.disc.cy))
        return d / max(float(self.disc.r), 1e-6)

    # ------------------------------------------------------------- helpers
    def _crossing_mask(self) -> np.ndarray:
        out = np.zeros(self.shape, dtype=bool)
        if self.junc_degrees.size == 0:
            return out
        bad = np.nonzero(self.junc_degrees >= 4)[0] + 1
        if bad.size:
            out = np.isin(self.junc_labels, bad)
        return out

    def _branch_orders(self) -> np.ndarray:
        """Disc-rooted BFS depth of every skan branch (-1 when unreachable)."""
        n = len(self.path_coords)
        order = np.full(max(n, 1), -1, dtype=np.int32)[:n] if n else np.zeros(0, np.int32)
        if n == 0 or self.summary is None or len(self.summary) == 0:
            return order
        order = np.full(n, -1, dtype=np.int32)
        df = self.summary
        src = df["node_id_src"].to_numpy()
        dst = df["node_id_dst"].to_numpy()
        node_rc: Dict[int, Tuple[float, float]] = {}
        for i in range(len(df)):
            node_rc[int(src[i])] = (float(df["src_row"].iloc[i]), float(df["src_col"].iloc[i]))
            node_rc[int(dst[i])] = (float(df["dst_row"].iloc[i]), float(df["dst_col"].iloc[i]))
        if not node_rc:
            return order

        adj: Dict[int, List[Tuple[int, int]]] = {}
        for i in range(len(df)):
            a, b = int(src[i]), int(dst[i])
            adj.setdefault(a, []).append((b, i))
            adj.setdefault(b, []).append((a, i))

        if self.disc is not None:
            cy, cx = float(self.disc.cy), float(self.disc.cx)
        else:
            ys, xs = np.nonzero(self.fov)
            cy, cx = float(ys.mean()), float(xs.mean())

        # BFS from the node nearest the disc; every other component gets its own
        # root (the node nearest the disc inside that component).
        remaining = set(node_rc)
        while remaining:
            root = min(remaining, key=lambda k: (node_rc[k][0] - cy) ** 2 + (node_rc[k][1] - cx) ** 2)
            depth = {root: 0}
            queue = [root]
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                remaining.discard(u)
                for v, bi in adj.get(u, ()):
                    if 0 <= bi < n and order[bi] < 0:
                        order[bi] = depth[u] + 1
                    if v not in depth:
                        depth[v] = depth[u] + 1
                        queue.append(v)
            remaining.discard(root)
            remaining -= set(depth)
        return order

    def branch_length(self, bi: int) -> float:
        if self.summary is None or bi >= len(self.summary):
            return float("nan")
        return float(self.summary["branch_distance"].iloc[bi])

    def branch_type(self, bi: int) -> int:
        if self.summary is None or bi >= len(self.summary):
            return -1
        return int(self.summary["branch_type"].iloc[bi])

    # ------------------------------------------------------- local context
    def local_density(self, row: float, col: float, r_loc: float) -> float:
        """Vessel-pixel fraction inside a ``10 * r_loc`` window (FOV-restricted)."""
        pad = max(8.0, 10.0 * max(r_loc, 0.5))
        sl = _win(self.shape, row, col, pad)
        f = self.fov[sl]
        if not f.any():
            return float("nan")
        return float((self.mask[sl] & f).sum()) / float(f.sum())

    def local_contrast(self, row: float, col: float, r_loc: float) -> float:
        return image_contrast(self.image, self.mask, self.fov, row, col, r_loc)

    # ------------------------------------------------------------ the pool
    def locus_pool(self, stride: int = 3) -> List[Locus]:
        """All centreline positions passing the GT structural quality gate.

        Only branches of type 1 (junction-to-endpoint) or 2
        (junction-to-junction) are eligible.  A position must be

        * at least ``2.5 * D_loc`` from any junction pixel,
        * at least ``1.5 * D_loc`` from the branch's endpoint,
        * unambiguous: no skeleton pixel of a *different* branch (nor any
          junction pixel) within ``1.5 * D_loc``,
        * at least ``2 * D_loc + 6`` px inside the FOV,
        * on a locally stable lumen (radius CV <= 0.35 over +- 2 * D_loc).

        This is the pool severing / truncation / caliber loci are drawn from.
        """
        if self._pool is None:
            self._pool = self._build_pool(stride, ambiguity=True)
        return self._pool

    def port_pool(self, stride: int = 3) -> List[Locus]:
        """Pool for **bridge** endpoints: the same gate *without* the ambiguity test.

        The ambiguity gate exists so that a *cut* lands on an unambiguous piece
        of centreline; it rejects every position that has another branch within
        ``1.5 * D_loc`` -- which is exactly the "neighbouring non-connected
        branch / parallel vessel" configuration Algorithm 1 lines 17-18 asks
        bridging to study.  Bridge ports therefore keep the junction, endpoint,
        FOV and lumen-stability gates but drop the ambiguity one; the
        third-vessel and junction-disk tests are applied per *pair* instead
        (:func:`_chord_is_clean`).
        """
        if self._ports is None:
            self._ports = self._build_pool(stride, ambiguity=False)
        return self._ports

    def _build_pool(self, stride: int, ambiguity: bool) -> List[Locus]:
        pool: List[Locus] = []
        if self.graph is None:
            return pool

        rad = self.radius
        for bi, co in enumerate(self.path_coords):
            btype = self.branch_type(bi)
            if btype not in (1, 2):
                continue
            arc = self.path_arc[bi]
            n = co.shape[0]
            if n < 5:
                continue
            for k in range(1, n - 1, max(1, int(stride))):
                r, c = int(co[k, 0]), int(co[k, 1])
                if self.junc_px[r, c]:
                    continue
                r_loc = float(rad[r, c])
                if r_loc < 1.0:
                    continue
                d_loc = 2.0 * r_loc
                if self.dist_junc[r, c] < MIN_JUNCTION_CLEARANCE * d_loc:
                    continue
                if self.dist_end[r, c] < MIN_ENDPOINT_CLEARANCE * d_loc:
                    continue
                if self.dist_fov[r, c] < 2.0 * d_loc + MIN_FOV_MARGIN:
                    continue
                if ambiguity and not self._unambiguous(r, c, bi, AMBIGUITY_RADIUS * d_loc):
                    continue
                cv, tangent = self._local_profile(bi, k, d_loc)
                if not np.isfinite(cv) or cv > MAX_RADIUS_CV:
                    continue
                pool.append(
                    Locus(
                        locus_id="",
                        row=r,
                        col=c,
                        branch=bi,
                        branch_type=btype,
                        r_loc=r_loc,
                        d_loc=d_loc,
                        tangent=(float(tangent[0]), float(tangent[1])),
                        arc_pos=float(arc[k]),
                        dist_junction=float(self.dist_junc[r, c]),
                        dist_endpoint=float(self.dist_end[r, c]),
                        dist_fov=float(self.dist_fov[r, c]),
                        radius_cv=float(cv),
                        gt_branch_length=self.branch_length(bi),
                        branch_order=int(self.branch_order[bi]) if bi < self.branch_order.size else -1,
                    )
                )
        self._annotate_bins(pool)
        return pool

    def _unambiguous(self, r: int, c: int, bi: int, radius: float) -> bool:
        rad = max(2.0, float(radius))
        sl = _win(self.shape, r, c, rad)
        sub = self.skel[sl]
        if not sub.any():
            return True
        rr, cc = np.nonzero(sub)
        rr = rr + sl[0].start
        cc = cc + sl[1].start
        inside = (rr - r) ** 2 + (cc - c) ** 2 <= rad * rad
        if not inside.any():
            return True
        lab = self.branch_label[rr[inside], cc[inside]]
        return bool(np.all(lab == (bi + 1)))

    def _local_profile(self, bi: int, k: int, d_loc: float) -> Tuple[float, np.ndarray]:
        """Radius CV over +-2 D_loc and the local tangent (unit, row/col)."""
        co = self.path_coords[bi]
        arc = self.path_arc[bi]
        half = 2.0 * d_loc
        lo = max(0, int(np.searchsorted(arc, arc[k] - half)))
        hi = min(co.shape[0], max(int(np.searchsorted(arc, arc[k] + half)), lo + 2))
        seg = co[lo:hi]
        vals = self.radius[seg[:, 0], seg[:, 1]].astype(float)
        mu = float(vals.mean())
        cv = float(vals.std() / mu) if mu > 1e-6 else float("inf")
        span = max(2, int(round(max(3.0, d_loc))))
        a = max(0, k - span)
        b = min(co.shape[0] - 1, k + span)
        vec = co[b].astype(float) - co[a].astype(float)
        return cv, _unit(vec)

    def _annotate_bins(self, pool: Sequence[Locus]) -> None:
        if not pool:
            return
        for lo in pool:
            lo.radius_bin = int(np.searchsorted(RADIUS_BIN_EDGES, lo.r_loc))
            lo.zone = self.zone_of(lo.row, lo.col)
            lo.density = self.local_density(lo.row, lo.col, lo.r_loc)
            lo.contrast = self.local_contrast(lo.row, lo.col, lo.r_loc)
        d_edges = _tercile_edges(np.array([lo.density for lo in pool], dtype=float))
        c_edges = _tercile_edges(np.array([lo.contrast for lo in pool], dtype=float))
        for lo in pool:
            lo.density_bin = _bin_of(lo.density, d_edges)
            lo.contrast_bin = _bin_of(lo.contrast, c_edges)

    # ----------------------------------------------------------- selection
    def sample_loci(self, n: int, rng: np.random.Generator, min_sep_factor: float = 3.0) -> List[Locus]:
        """Sample ``n`` well-separated loci from the qualified pool."""
        pool = self.locus_pool()
        if not pool:
            return []
        idx = rng.permutation(len(pool))
        chosen: List[Locus] = []
        pts: List[Tuple[float, float, float]] = []
        for i in idx:
            lo = pool[int(i)]
            sep = max(6.0, min_sep_factor * lo.d_loc)
            ok = True
            for (pr, pc, ps) in pts:
                if (lo.row - pr) ** 2 + (lo.col - pc) ** 2 < max(sep, ps) ** 2:
                    ok = False
                    break
            if not ok:
                continue
            lo.locus_id = f"{self.key}:L{len(chosen):04d}"
            chosen.append(lo)
            pts.append((float(lo.row), float(lo.col), sep))
            if len(chosen) >= n:
                break
        return chosen

    def terminal_branches(self) -> List[int]:
        """Indices of terminal (junction-to-endpoint) branches inside the FOV."""
        out: List[int] = []
        if self.summary is None:
            return out
        for bi in range(len(self.path_coords)):
            if self.branch_type(bi) != 1:
                continue
            co = self.path_coords[bi]
            if co.shape[0] < 6:
                continue
            e = _free_end_index(co, self.end_px)
            if e is None:
                continue
            r, c = int(co[e, 0]), int(co[e, 1])
            if self.dist_fov[r, c] < MIN_FOV_MARGIN + 2.0 * float(self.radius[r, c]):
                continue
            out.append(bi)
        return out

    def locus_for_branch(self, bi: int, at_end: bool = True) -> Optional[Locus]:
        """A Locus-shaped context descriptor for a whole branch.

        Used so that ``truncate`` and ``caliber`` events carry the same context
        bins (radius / zone / order / density / contrast) as sever events, which
        is what ``matched_control`` matches on.
        """
        if bi >= len(self.path_coords):
            return None
        co = self.path_coords[bi]
        arc = self.path_arc[bi]
        k = co.shape[0] // 2
        r, c = int(co[k, 0]), int(co[k, 1])
        r_loc = float(self.radius[r, c])
        if r_loc <= 0:
            r_loc = 1.0
        cv, tangent = self._local_profile(bi, k, 2.0 * r_loc)
        lo = Locus(
            locus_id="",
            row=r,
            col=c,
            branch=bi,
            branch_type=self.branch_type(bi),
            r_loc=r_loc,
            d_loc=2.0 * r_loc,
            tangent=(float(tangent[0]), float(tangent[1])),
            arc_pos=float(arc[k]),
            dist_junction=float(self.dist_junc[r, c]),
            dist_endpoint=float(self.dist_end[r, c]),
            dist_fov=float(self.dist_fov[r, c]),
            radius_cv=float(cv),
            gt_branch_length=self.branch_length(bi),
            branch_order=int(self.branch_order[bi]) if bi < self.branch_order.size else -1,
        )
        lo.radius_bin = int(np.searchsorted(RADIUS_BIN_EDGES, lo.r_loc))
        lo.zone = self.zone_of(lo.row, lo.col)
        lo.density = self.local_density(lo.row, lo.col, lo.r_loc)
        lo.contrast = self.local_contrast(lo.row, lo.col, lo.r_loc)
        pool = self.locus_pool()
        d_edges = _tercile_edges(np.array([p.density for p in pool], dtype=float)) if pool else np.array([np.inf, np.inf])
        c_edges = _tercile_edges(np.array([p.contrast for p in pool], dtype=float)) if pool else np.array([np.inf, np.inf])
        lo.density_bin = _bin_of(lo.density, d_edges)
        lo.contrast_bin = _bin_of(lo.contrast, c_edges)
        return lo


# --------------------------------------------------------------------------- #
# (a) connectivity: capsule severing
# --------------------------------------------------------------------------- #
def sever(ctx: GTContext, locus: Locus, l_factor: float) -> PerturbResult:
    """Remove a capsule of length ``l_factor * D_loc`` across the whole lumen.

    Verification (Algorithm 1 line 10): ``beta0`` inside the FOV must increase
    by exactly 1, **or** the two sides of the branch must end up in different
    connected components of ``M'``.
    """
    M = ctx.mask
    L = float(l_factor) * locus.d_loc
    half = 0.5 * L
    r_cap = locus.r_loc + 1.0
    sl, reg = capsule_region(M.shape, (locus.row, locus.col), locus.tangent, half, r_cap)

    Mp = M.copy()
    sub = Mp[sl]
    n_changed = int(np.count_nonzero(sub & reg))
    if n_changed == 0:
        return PerturbResult(False, "empty_capsule")
    sub[reg] = False
    Mp[sl] = sub

    b0_before, b1_before = betti_in_fov(M, ctx.fov)
    b0_after, b1_after = betti_in_fov(Mp, ctx.fov)
    d_b0 = b0_after - b0_before

    ends = _branch_side_points(ctx, locus, half + r_cap + 2.0)
    sides_split = False
    if ends is not None:
        lab, _ = SK.connected_components(Mp, connectivity=8, fov=ctx.fov)
        (ar, ac), (br, bc) = ends
        la, lb = int(lab[ar, ac]), int(lab[br, bc])
        sides_split = bool(la > 0 and lb > 0 and la != lb)

    ok = (d_b0 == 1) or sides_split
    info = {
        "sever_L": L,
        "sever_half_len": half,
        "sever_r_cap": r_cap,
        "delta_beta0": float(d_b0),
        "delta_beta1": float(b1_after - b1_before),
        "sides_split": float(sides_split),
    }
    if not ok:
        return PerturbResult(False, "no_disconnection", info=info, n_changed=n_changed)
    return PerturbResult(True, "", mask=Mp, n_changed=n_changed, info=info)


def _branch_side_points(ctx: GTContext, locus: Locus, offset: float):
    """Centreline points at +- ``offset`` arc length from the locus."""
    co = ctx.path_coords[locus.branch]
    arc = ctx.path_arc[locus.branch]
    k = int(np.argmin(np.abs(arc - locus.arc_pos)))
    lo = int(np.searchsorted(arc, arc[k] - offset)) - 1
    hi = int(np.searchsorted(arc, arc[k] + offset))
    if lo < 0 or hi >= co.shape[0]:
        return None
    return (int(co[lo, 0]), int(co[lo, 1])), (int(co[hi, 0]), int(co[hi, 1]))


# --------------------------------------------------------------------------- #
# (b) shortcut / bridge
# --------------------------------------------------------------------------- #
@dataclass
class BridgePair:
    """A candidate false connection between two GT branches."""

    pair_id: str
    a: Tuple[int, int]
    b: Tuple[int, int]
    branch_a: int
    branch_b: int
    r_a: float
    r_b: float
    t_a: Tuple[float, float]
    t_b: Tuple[float, float]
    d_euclid: float
    geodesic: float           # inf means "farther than the cut-off"
    same_component: bool
    zone: int
    radius_bin: int
    branch_order: int
    density: float = float("nan")
    density_bin: int = 0
    contrast: float = float("nan")
    contrast_bin: int = 0

    def mid(self) -> Tuple[float, float]:
        return (0.5 * (self.a[0] + self.b[0]), 0.5 * (self.a[1] + self.b[1]))


def local_geodesic(skel: np.ndarray, a, b, limit: float) -> float:
    """Skeleton graph-geodesic between ``a`` and ``b``, cut off at ``limit``.

    Dijkstra runs on the 8-connected skeleton pixel graph restricted to a window
    of half-size ``limit`` centred on the pair.  Any path leaving that window is
    itself at least ``limit`` long (the pair is at the centre and
    ``||a - b|| <= limit``), so returning ``inf`` when ``b`` is unreached inside
    the window is a valid "at least ``limit``" statement -- which is all the
    acceptance test needs.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    W = max(8.0, float(limit))
    mid = (0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
    sl = _win(skel.shape, mid[0], mid[1], W)
    sub = skel[sl]
    h, w = sub.shape
    idx = -np.ones((h, w), dtype=np.int64)
    rr, cc = np.nonzero(sub)
    if rr.size == 0:
        return float("inf")
    idx[rr, cc] = np.arange(rr.size)
    ar, ac = int(a[0]) - sl[0].start, int(a[1]) - sl[1].start
    br, bc = int(b[0]) - sl[0].start, int(b[1]) - sl[1].start
    if not (0 <= ar < h and 0 <= ac < w and 0 <= br < h and 0 <= bc < w):
        return float("inf")
    ia, ib = int(idx[ar, ac]), int(idx[br, bc])
    if ia < 0 or ib < 0:
        return float("inf")

    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    vals: List[np.ndarray] = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r2, c2 = rr + dr, cc + dc
        ok = (r2 >= 0) & (r2 < h) & (c2 >= 0) & (c2 < w)
        if not ok.any():
            continue
        ok2 = ok.copy()
        ok2[ok] = sub[r2[ok], c2[ok]]
        if not ok2.any():
            continue
        rows.append(idx[rr[ok2], cc[ok2]])
        cols.append(idx[r2[ok2], c2[ok2]])
        vals.append(np.full(int(ok2.sum()), math.sqrt(2.0) if (dr and dc) else 1.0))
    if not rows:
        return float("inf")
    n = int(rr.size)
    g = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    ).tocsr()
    d = dijkstra(g, directed=False, indices=ia, limit=W)
    return float(d[ib])


def _nearest_branch_label(ctx: GTContext, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """Branch label of the skeleton pixel nearest to each ``(rr, cc)``."""
    pad = 6.0 + 2.0 * float(ctx.radius[rr, cc].max())
    pts = np.stack([rr, cc], axis=1).astype(float)
    sl = _win_of_points(ctx.shape, pts, pad)
    sub = ctx.skel[sl]
    if not sub.any():
        return np.zeros(rr.shape, dtype=np.int32)
    _, idx = ndi.distance_transform_edt(~sub, return_indices=True)
    lab = ctx.branch_label[sl]
    lr = np.clip(rr - sl[0].start, 0, sub.shape[0] - 1)
    lc = np.clip(cc - sl[1].start, 0, sub.shape[1] - 1)
    return lab[idx[0][lr, lc], idx[1][lr, lc]]


def _chord_is_clean(ctx: GTContext, pa: Locus, pb: Locus) -> bool:
    """The straight ``a -> b`` chord misses junction disks and third vessels."""
    n = max(4, int(round(float(np.hypot(pa.row - pb.row, pa.col - pb.col)))) + 1)
    rr = np.rint(np.linspace(pa.row, pb.row, n)).astype(int)
    cc = np.rint(np.linspace(pa.col, pb.col, n)).astype(int)
    D = pa.d_loc + pb.d_loc
    if float(ctx.dist_junc[rr, cc].min()) < max(2.0, 0.5 * D):
        return False
    inside = ctx.mask[rr, cc]
    if not inside.any():
        return True
    owner = _nearest_branch_label(ctx, rr[inside], cc[inside])
    allowed = np.array([0, pa.branch + 1, pb.branch + 1])
    return bool(np.all(np.isin(owner, allowed)))


def bridge_candidates(
    ctx: GTContext, n: int, rng: np.random.Generator, port_stride: int = 4
) -> List[BridgePair]:
    """Candidate pairs for shortcut bridging (Algorithm 1 lines 12-18).

    A pair ``(a, b)`` qualifies when

    (i)   the points are on different branches, not adjacent in the GT graph,
          and either graph-geodesically ``>= 8 * D`` apart or in different
          connected components;
    (ii)  ``||a - b|| <= 6 * mean(r_a, r_b)``;
    (iii) ``|log(r_a / r_b)| <= log 2``;
    (iv)  the straight chord neither passes through a junction disk nor crosses
          a third vessel;
    (v)   neither point lies within ``2 * D`` of a degree-4 GT junction
          (crossings are excluded -- binary masks carry no reliable crossing
          topology, proposal section 3.1.3 lines 17-18).
    """
    pool = ctx.port_pool()
    if len(pool) < 2:
        return []
    ports = pool[:: max(1, int(port_stride))]
    if len(ports) < 2:
        return []
    pts = np.array([[p.row, p.col] for p in ports], dtype=float)
    rads = np.array([p.r_loc for p in ports], dtype=float)

    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    max_d = BRIDGE_MAX_D_OVER_R * float(rads.max())
    raw = tree.query_pairs(r=max_d, output_type="ndarray")
    if raw.size == 0:
        return []
    raw = raw[rng.permutation(raw.shape[0])]

    dist_cross = _edt_to(ctx.crossing_px, ctx.shape)
    out: List[BridgePair] = []
    seen_sites: List[Tuple[float, float]] = []
    for i, j in raw:
        if len(out) >= n:
            break
        pa, pb = ports[int(i)], ports[int(j)]
        if pa.branch == pb.branch:
            continue
        r_a, r_b = pa.r_loc, pb.r_loc
        rbar = 0.5 * (r_a + r_b)
        d = float(np.hypot(pa.row - pb.row, pa.col - pb.col))
        if d < 2.0 or d > BRIDGE_MAX_D_OVER_R * rbar:
            continue
        if abs(math.log(max(r_a, 1e-6) / max(r_b, 1e-6))) > BRIDGE_MAX_LOG_RATIO:
            continue
        D = 2.0 * rbar
        if (dist_cross[pa.row, pa.col] < BRIDGE_CROSSING_CLEARANCE * D
                or dist_cross[pb.row, pb.col] < BRIDGE_CROSSING_CLEARANCE * D):
            continue
        if not _chord_is_clean(ctx, pa, pb):
            continue
        cut = max(BRIDGE_MIN_GEODESIC * D, 4.0 * d)
        geo = local_geodesic(ctx.skel, (pa.row, pa.col), (pb.row, pb.col), cut)
        if geo < BRIDGE_MIN_GEODESIC * D:
            continue
        mid = (0.5 * (pa.row + pb.row), 0.5 * (pa.col + pb.col))
        sep = max(8.0, 2.0 * D)
        if any((mid[0] - sr) ** 2 + (mid[1] - sc) ** 2 < sep * sep for sr, sc in seen_sites):
            continue
        seen_sites.append(mid)
        out.append(
            BridgePair(
                pair_id=f"{ctx.key}:B{len(out):04d}",
                a=(int(pa.row), int(pa.col)),
                b=(int(pb.row), int(pb.col)),
                branch_a=pa.branch,
                branch_b=pb.branch,
                r_a=r_a,
                r_b=r_b,
                t_a=pa.tangent,
                t_b=pb.tangent,
                d_euclid=d,
                geodesic=float(geo),
                same_component=bool(np.isfinite(geo)),
                zone=ctx.zone_of(*mid),
                radius_bin=int(np.searchsorted(RADIUS_BIN_EDGES, rbar)),
                branch_order=int(min(pa.branch_order, pb.branch_order)),
                density=pa.density,
                density_bin=pa.density_bin,
                contrast=pa.contrast,
                contrast_bin=pa.contrast_bin,
            )
        )
    return out


def bridge(ctx: GTContext, pair: BridgePair, n_samples: int = 0) -> PerturbResult:
    """Connect ``pair`` with a tangent-constrained cubic Bezier tube.

    Verification (Algorithm 1 line 16): components merged (``beta0`` drops), or
    a new cycle appears (``beta1`` rises), or the graph geodesic is shortened by
    at least a factor 3.
    """
    M = ctx.mask
    a = np.array(pair.a, dtype=float)
    b = np.array(pair.b, dtype=float)
    u = _unit(b - a)
    ta = _unit(pair.t_a)
    tb = _unit(pair.t_b)
    if float(ta @ u) < 0:
        ta = -ta
    if float(tb @ u) < 0:
        tb = -tb
    d = pair.d_euclid
    p1 = a + (d / 3.0) * ta
    p2 = b - (d / 3.0) * tb
    n = int(n_samples) if n_samples else max(16, int(round(4.0 * d)))
    curve = _bezier_points(a, p1, p2, b, n)
    curve_len = _polyline_length(curve)

    r_tube = 0.5 * (pair.r_a + pair.r_b)
    sl, tube = _tube_from_points(M.shape, curve, r_tube)

    Mp = M.copy()
    sub = Mp[sl]
    add = tube & (~sub) & ctx.fov[sl]
    n_changed = int(np.count_nonzero(add))
    if n_changed == 0:
        return PerturbResult(False, "empty_tube")
    sub |= add
    Mp[sl] = sub

    b0_before, b1_before = betti_in_fov(M, ctx.fov)
    b0_after, b1_after = betti_in_fov(Mp, ctx.fov)
    merged = b0_after < b0_before
    cycle = b1_after > b1_before
    shortcut = bool(pair.geodesic >= BRIDGE_SHORTCUT_FACTOR * max(curve_len, 1e-6))

    info = {
        "bridge_len": float(curve_len),
        "bridge_r_tube": float(r_tube),
        "delta_beta0": float(b0_after - b0_before),
        "delta_beta1": float(b1_after - b1_before),
        "merged": float(merged),
        "cycle": float(cycle),
        "shortcut": float(shortcut),
        "geodesic_before": float(pair.geodesic),
    }
    if not (merged or cycle or shortcut):
        return PerturbResult(False, "no_topology_change", info=info, n_changed=n_changed)
    return PerturbResult(True, "", mask=Mp, n_changed=n_changed, info=info)


# --------------------------------------------------------------------------- #
# (c) terminal loss: truncation
# --------------------------------------------------------------------------- #
def branch_tube(ctx: GTContext, centre_px: np.ndarray, mask: Optional[np.ndarray] = None):
    """Lumen pixels owned by ``centre_px`` under the skeleton Voronoi partition.

    ``centre_px`` is an ``(N, 2)`` int array of centreline pixels.  A mask pixel
    belongs to the tube when its **nearest skeleton pixel** is one of them; this
    is the "tube-aware" removal Algorithm 1 line 21 asks for, and it never eats
    into a neighbouring branch.
    """
    M = ctx.mask if mask is None else mask
    rmax = float(ctx.radius[centre_px[:, 0], centre_px[:, 1]].max())
    pad = 3.0 * rmax + 6.0
    sl = _win_of_points(M.shape, centre_px.astype(float), pad)
    sub_skel = ctx.skel[sl]
    empty = np.zeros((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start), dtype=bool)
    if not sub_skel.any():
        return sl, empty
    sel = np.zeros_like(sub_skel)
    sel[centre_px[:, 0] - sl[0].start, centre_px[:, 1] - sl[1].start] = True
    _, idx = ndi.distance_transform_edt(~sub_skel, return_indices=True)
    owned = sel[idx[0], idx[1]] & M[sl]
    return sl, owned


def truncate(ctx: GTContext, branch_idx: int, rho: float) -> PerturbResult:
    """Remove the terminal fraction ``rho`` of a terminal branch (tube-aware).

    Verification (proposal section 3.1.3): ``beta0`` must be unchanged -- a
    terminal loss shortens a branch and moves its endpoint, it does not split
    the tree.
    """
    co = ctx.path_coords[branch_idx]
    arc = ctx.path_arc[branch_idx]
    e = _free_end_index(co, ctx.end_px)
    if e is None:
        return PerturbResult(False, "no_free_end")
    total = float(arc[-1])
    if total < 6.0:
        return PerturbResult(False, "branch_too_short")
    cut = float(rho) * total
    keep = (arc <= cut) if e == 0 else (arc >= (total - cut))
    sel = co[keep]
    if sel.shape[0] < 2:
        return PerturbResult(False, "cut_too_short")

    M = ctx.mask
    sl, owned = branch_tube(ctx, sel)
    Mp = M.copy()
    sub = Mp[sl]
    n_changed = int(np.count_nonzero(sub & owned))
    if n_changed == 0:
        return PerturbResult(False, "empty_truncation")
    sub[owned] = False
    Mp[sl] = sub

    b0_before, b1_before = betti_in_fov(M, ctx.fov)
    b0_after, b1_after = betti_in_fov(Mp, ctx.fov)
    info = {
        "truncate_rho": float(rho),
        "truncate_len": cut,
        "gt_branch_length": total,
        "delta_beta0": float(b0_after - b0_before),
        "delta_beta1": float(b1_after - b1_before),
    }
    if b0_after != b0_before:
        return PerturbResult(False, "beta0_changed", info=info, n_changed=n_changed)
    return PerturbResult(True, "", mask=Mp, n_changed=n_changed, info=info)


# --------------------------------------------------------------------------- #
# (d) caliber: the topology-neutral control arm
# --------------------------------------------------------------------------- #
def caliber(ctx: GTContext, locus: Locus, factor: int, span_factor: float = 6.0) -> PerturbResult:
    """Erode (``factor = -1``) or dilate (``factor = +1``) a branch tube by 1 px.

    Verification: every graph invariant (``beta0``, ``beta1``, #endpoints,
    #junctions, skan branch count) must be unchanged -- this arm is
    topology-neutral by construction and is the "same pixel budget, no
    structural change" reference of the design.
    """
    if factor < 0 and locus.r_loc < MIN_ERODIBLE_RADIUS:
        # a lumen thinner than ~2 px cannot lose a boundary layer and stay
        # connected: peeling it is never topology-neutral, so it is not a valid
        # caliber control site.  Such loci get the dilation arm instead.
        return PerturbResult(False, "lumen_too_thin")
    co = ctx.path_coords[locus.branch]
    arc = ctx.path_arc[locus.branch]
    half = span_factor * locus.d_loc
    lo = max(0, int(np.searchsorted(arc, locus.arc_pos - half)))
    hi = int(np.searchsorted(arc, locus.arc_pos + half))
    sel = co[lo: max(hi, lo + 2)]
    if sel.shape[0] < 2:
        return PerturbResult(False, "span_too_short")

    M = ctx.mask
    sl, owned = branch_tube(ctx, sel)
    if not owned.any():
        return PerturbResult(False, "empty_tube")

    Mp = M.copy()
    sub = Mp[sl].copy()
    if factor < 0:
        eroded = ndi.binary_erosion(sub, structure=SK.EIGHT.astype(bool), border_value=1)
        change = owned & sub & (~eroded)
        sub[change] = False
    else:
        grown = ndi.binary_dilation(owned, structure=SK.EIGHT.astype(bool))
        change = grown & (~sub) & ctx.fov[sl]
        sub[change] = True
    n_changed = int(np.count_nonzero(change))
    if n_changed == 0:
        return PerturbResult(False, "no_change")
    Mp[sl] = sub

    wsl = expand_slices(sl, ctx.shape, INVARIANT_MARGIN)
    inv0 = window_invariants(M, ctx.fov, wsl, ctx.min_branch_px)
    inv1 = window_invariants(Mp, ctx.fov, wsl, ctx.min_branch_px)
    info = {"caliber_factor": float(factor), "caliber_span": float(2 * half)}
    info.update(inv0.as_dict("inv0_"))
    info.update(inv1.as_dict("inv1_"))
    if inv0 != inv1:
        return PerturbResult(False, "invariants_changed", info=info, n_changed=n_changed)
    return PerturbResult(True, "", mask=Mp, n_changed=n_changed, info=info)


# --------------------------------------------------------------------------- #
# pixel- AND context-matched topology-neutral control
# --------------------------------------------------------------------------- #
#: progressive relaxation of the context match (tier 0 = all five bins match)
_MATCH_TIERS = (
    ("radius_bin", "zone", "branch_order", "density_bin", "contrast_bin"),
    ("radius_bin", "zone", "branch_order", "density_bin"),
    ("radius_bin", "zone", "density_bin"),
    ("radius_bin", "zone"),
    ("radius_bin",),
    (),
)

ADDITION_KINDS = ("bridge",)


def matched_control(
    ctx: GTContext,
    perturbed: np.ndarray,
    locus_ctx: Locus,
    kind: str,
    rng: np.random.Generator,
    max_tries: int = CONTROL_MAX_TRIES,
    inv_ref: Optional[Invariants] = None,
    local: bool = True,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    """Build ``M_C`` (Algorithm 1 lines 23-29).

    ``N_T = |M' xor M|`` pixels are removed (deletion-type: ``sever``,
    ``truncate``, eroding ``caliber``) or added (addition-type: ``bridge``,
    dilating ``caliber``) at a **donor** site drawn from the qualified locus
    pool and matched on radius bin, zone, GT branch order, local-density bin and
    local-contrast bin.  The donor is accepted only when the graph invariants of
    ``M_C`` equal those of ``M``; otherwise another donor is drawn, up to
    ``max_tries`` times.

    Returns ``(M_C or None, diagnostics)``.
    """
    M = ctx.mask
    N_T = int(np.count_nonzero(perturbed ^ M))
    addition = _is_addition(kind, perturbed, M)
    diag: Dict[str, object] = {
        "control_N_T": float(N_T),
        "control_tries": 0.0,
        "control_ok": 0.0,
        "control_n_changed": float("nan"),
        "control_match_tier": float("nan"),
        "control_addition": float(addition),
        "control_fail_reason": "",
    }
    if N_T <= 0:
        diag["control_fail_reason"] = "no_pixel_budget"
        return None, diag

    inv0 = inv_ref
    if inv0 is None and not local:
        inv0 = graph_invariants(M, ctx.fov, ctx.min_branch_px)
    pool = ctx.locus_pool()
    if not pool:
        diag["control_fail_reason"] = "empty_pool"
        return None, diag

    exclude_r, exclude_c = locus_ctx.row, locus_ctx.col
    exclude_rad = max(12.0, 6.0 * locus_ctx.d_loc)

    tries = 0
    last_reason = "no_donor"
    for tier, keys in enumerate(_MATCH_TIERS):
        donors = [
            p for p in pool
            if all(getattr(p, k) == getattr(locus_ctx, k) for k in keys)
            and (p.row - exclude_r) ** 2 + (p.col - exclude_c) ** 2 > exclude_rad ** 2
            and p.r_loc >= (1.0 if addition else MIN_CONTROL_DONOR_RADIUS)
        ]
        if not donors:
            continue
        for oi in rng.permutation(len(donors)):
            if tries >= int(max_tries):
                break
            tries += 1
            donor = donors[int(oi)]
            Mc, n_changed, reason, sl = _apply_control(ctx, donor, N_T, addition, rng)
            if Mc is None:
                last_reason = reason
                continue
            if local:
                wsl = expand_slices(sl, ctx.shape, INVARIANT_MARGIN)
                ref = window_invariants(M, ctx.fov, wsl, ctx.min_branch_px)
                inv1 = window_invariants(Mc, ctx.fov, wsl, ctx.min_branch_px)
            else:
                ref = inv0
                inv1 = graph_invariants(Mc, ctx.fov, ctx.min_branch_px)
            if inv1 != ref:
                last_reason = "invariants_changed"
                continue
            diag.update(
                control_tries=float(tries),
                control_ok=1.0,
                control_n_changed=float(n_changed),
                control_match_tier=float(tier),
                control_donor_row=float(donor.row),
                control_donor_col=float(donor.col),
                control_donor_r=float(donor.r_loc),
                control_fail_reason="",
            )
            return Mc, diag
        if tries >= int(max_tries):
            break

    diag["control_tries"] = float(tries)
    diag["control_fail_reason"] = last_reason
    return None, diag


def _is_addition(kind: str, perturbed: np.ndarray, M: np.ndarray) -> bool:
    if kind in ADDITION_KINDS:
        return True
    if kind == "caliber":
        return int(np.count_nonzero(perturbed & ~M)) > int(np.count_nonzero(M & ~perturbed))
    return False


def _apply_control(
    ctx: GTContext, donor: Locus, N_T: int, addition: bool, rng: np.random.Generator
):
    """Remove / add exactly ``N_T`` (+-2) boundary pixels at the donor site.

    Returns ``(M_C, n_changed, reason, window)`` -- the window is the slice pair
    the edit is confined to, so the caller can test topology neutrality locally.
    """
    co = ctx.path_coords[donor.branch]
    arc = ctx.path_arc[donor.branch]
    M = ctx.mask

    span = max(4.0 * donor.d_loc, 8.0)
    cand = None
    sl = None
    n_cand = 0
    for _ in range(6):
        lo = max(0, int(np.searchsorted(arc, donor.arc_pos - span)))
        hi = int(np.searchsorted(arc, donor.arc_pos + span))
        sel = co[lo: max(hi, lo + 2)]
        if sel.shape[0] < 2:
            return None, 0, "span_too_short", None
        sl, owned = branch_tube(ctx, sel)
        sub = M[sl]
        if addition:
            grown = ndi.binary_dilation(owned, structure=SK.EIGHT.astype(bool))
            cand = grown & (~sub) & ctx.fov[sl]
        else:
            eroded = ndi.binary_erosion(sub, structure=SK.EIGHT.astype(bool), border_value=1)
            cand = owned & sub & (~eroded)
        n_cand = int(np.count_nonzero(cand))
        if n_cand >= N_T:
            break
        if lo <= 0 and hi >= co.shape[0] - 1:
            return None, 0, "donor_too_small", None
        span *= 2.0

    if cand is None or n_cand < N_T - CONTROL_PIXEL_TOL:
        return None, 0, "donor_too_small", None

    rr, cc = np.nonzero(cand)
    take = rng.permutation(rr.size)[: min(N_T, rr.size)]
    Mc = M.copy()
    sub = Mc[sl]
    sub[rr[take], cc[take]] = bool(addition)
    Mc[sl] = sub
    n_changed = int(np.count_nonzero(Mc[sl] ^ M[sl]))
    if abs(n_changed - N_T) > CONTROL_PIXEL_TOL:
        return None, n_changed, "pixel_budget_missed", sl
    return Mc, n_changed, "", sl


# --------------------------------------------------------------------------- #
# Contract F feature extractor
# --------------------------------------------------------------------------- #
PHI_COLUMNS = [
    "phi_r_a", "phi_r_b", "phi_r_mean", "phi_r_logratio",
    "phi_d", "phi_d_over_r", "phi_gap_len",
    "phi_cos_a", "phi_cos_b", "phi_cos_ab",
    "phi_zone", "phi_eccentricity", "phi_fov_dist", "phi_fov_dist_over_r",
    "phi_density", "phi_density_far", "phi_density_ratio",
    "phi_contrast", "phi_frangi_gap", "phi_frangi_ring", "phi_frangi_ratio",
    "phi_n_endpoints_local", "phi_n_junctions_local", "phi_skel_density_local",
    "phi_mask_frac_gap", "phi_curvature_a", "phi_curvature_b",
]

STRATA_COLUMNS = [
    "gt_branch_order", "gt_branch_length", "gt_radius_bin", "gt_branch_type",
    "gt_r_loc", "gt_d_loc", "gt_dist_junction", "gt_dist_endpoint",
    "gt_zone", "gt_density", "gt_contrast", "gt_density_bin", "gt_contrast_bin",
]


class _EventPoints:
    """Minimal event view accepted by :func:`phi` (Contract F)."""

    __slots__ = ("kind", "point_a", "point_b")

    def __init__(self, kind: str, point_a, point_b=None):
        self.kind = kind
        self.point_a = point_a
        self.point_b = point_b


def phi(
    event,
    image,
    mask_prime: np.ndarray,
    disc=None,
    fov: Optional[np.ndarray] = None,
    min_branch_px: int = 3,
) -> Dict[str, float]:
    """Contract-F features of one event, from ``(I, M')`` and the auto disc only.

    ``event`` must expose ``point_a`` / ``point_b`` (``(row, col)`` or ``None``)
    and ``kind``.  For *severing* events the two points are the **two new
    endpoints** created by the cut (located on ``M'``); for *bridging* events
    they are the two attachment points and every feature is evaluated on the
    **connected state**, exactly as section 3.1.5 prescribes ("both heads share
    one feature schema but observe different states").  No ground-truth quantity
    is read here.
    """
    M = SK.as_bool(mask_prime)
    f = SK.fov_or_true(fov, M.shape)
    pa = getattr(event, "point_a", None)
    pb = getattr(event, "point_b", None)
    out: Dict[str, float] = {k: float("nan") for k in PHI_COLUMNS}
    if pa is None:
        return out

    a = np.array(pa, dtype=float)
    mid = a.copy()
    pad_ctx = 64.0
    if pb is not None:
        b = np.array(pb, dtype=float)
        mid = 0.5 * (a + b)
        pad_ctx = max(64.0, 2.0 * float(np.hypot(*(a - b))))

    # local crop: all skeleton-derived features are computed on a window big
    # enough that the local neighbourhood is complete but small enough to be
    # cheap on 2k-4k images.
    sl = _win(M.shape, mid[0], mid[1], pad_ctx + 40.0)
    Mw = M[sl]
    fw = f[sl]
    rad = SK.local_radius(Mw)
    skel = SK.skeletonize_mask(Mw, min_branch_px=min_branch_px, fov=fw)
    deg = SK.neighbour_count(skel)
    end_px = skel & (deg == 1)
    junc_px = skel & (deg >= 3)
    aw = a - np.array([sl[0].start, sl[1].start], dtype=float)

    r_a = _radius_at(rad, aw)
    out["phi_r_a"] = r_a
    t_a = _tangent_on_skeleton(skel, aw, r_a)
    out["phi_curvature_a"] = _curvature_on_skeleton(skel, aw, r_a)

    if pb is not None:
        bw = np.array(pb, dtype=float) - np.array([sl[0].start, sl[1].start], dtype=float)
        r_b = _radius_at(rad, bw)
        d = float(np.hypot(*(aw - bw)))
        rbar = 0.5 * (r_a + r_b) if np.isfinite(r_a) and np.isfinite(r_b) else float("nan")
        u = _unit(bw - aw)
        t_b = _tangent_on_skeleton(skel, bw, r_b)
        out["phi_r_b"] = r_b
        out["phi_r_mean"] = rbar
        out["phi_r_logratio"] = float(abs(math.log(max(r_a, 1e-6) / max(r_b, 1e-6))))
        out["phi_d"] = d
        out["phi_d_over_r"] = d / max(rbar, 1e-6) if np.isfinite(rbar) else float("nan")
        out["phi_gap_len"] = _gap_length(Mw, aw, bw)
        out["phi_cos_a"] = float(abs(t_a @ u))
        out["phi_cos_b"] = float(abs(t_b @ u))
        out["phi_cos_ab"] = float(abs(t_a @ t_b))
        out["phi_curvature_b"] = _curvature_on_skeleton(skel, bw, r_b)
        out["phi_mask_frac_gap"] = _chord_mask_fraction(Mw, aw, bw)
        midw = 0.5 * (aw + bw)
    else:
        rbar = r_a
        out["phi_r_mean"] = r_a
        midw = aw
        bw = None

    rb_safe = rbar if np.isfinite(rbar) and rbar > 0 else 1.5

    # ---- disc-relative geometry (automatic disc only) ----------------------
    if disc is not None:
        dd = math.hypot(float(mid[1]) - float(disc.cx), float(mid[0]) - float(disc.cy))
        rd = max(float(disc.r), 1e-6)
        out["phi_eccentricity"] = dd / rd
        out["phi_zone"] = float(0 if dd <= ZONE_A_R * rd else (1 if dd <= ZONE_B_R * rd else 2))

    # ---- FOV geometry (full-image distance transform of the FOV is O(HW) but
    #      only the value at one point is needed, so use the crop + a margin) --
    out["phi_fov_dist"] = _fov_distance(f, mid)
    out["phi_fov_dist_over_r"] = out["phi_fov_dist"] / rb_safe

    # ---- local vessel density (window of 10 * r) ---------------------------
    w_near = max(8.0, 10.0 * rb_safe)
    w_far = max(24.0, 30.0 * rb_safe)
    out["phi_density"] = _window_fraction(Mw, fw, midw, w_near)
    out["phi_density_far"] = _window_fraction(Mw, fw, midw, w_far)
    if np.isfinite(out["phi_density_far"]) and out["phi_density_far"] > 1e-9:
        out["phi_density_ratio"] = out["phi_density"] / out["phi_density_far"]
    out["phi_skel_density_local"] = _window_fraction(skel, fw, midw, w_near)
    out["phi_n_endpoints_local"] = _window_count(end_px, midw, w_near)
    out["phi_n_junctions_local"] = _window_count(junc_px, midw, w_near)

    # ---- local contrast: green channel + Frangi ridge response -------------
    out["phi_contrast"] = image_contrast(image, M, f, mid[0], mid[1], rb_safe)
    out.update(_frangi_stats(image, M, f, a, pb, rb_safe))
    return out


def _fov_distance(f: np.ndarray, p) -> float:
    """Distance from ``p`` to the FOV border, computed on a local crop."""
    for pad in (48.0, 128.0, 384.0):
        sl = _win(f.shape, p[0], p[1], pad)
        sub = f[sl]
        r = int(round(p[0])) - sl[0].start
        c = int(round(p[1])) - sl[1].start
        r = int(np.clip(r, 0, sub.shape[0] - 1))
        c = int(np.clip(c, 0, sub.shape[1] - 1))
        if not sub.all():
            return float(ndi.distance_transform_edt(sub)[r, c])
        # the crop is entirely inside the FOV -> the distance is at least `pad`
        touches_border = (
            sl[0].start == 0 or sl[1].start == 0
            or sl[0].stop == f.shape[0] or sl[1].stop == f.shape[1]
        )
        if touches_border:
            return float(ndi.distance_transform_edt(sub)[r, c])
    return float(pad)


def _radius_at(rad: np.ndarray, p) -> float:
    r = int(round(p[0]))
    c = int(round(p[1]))
    if not (0 <= r < rad.shape[0] and 0 <= c < rad.shape[1]):
        return float("nan")
    v = float(rad[r, c])
    if v > 0:
        return v
    sl = _win(rad.shape, p[0], p[1], 4)
    sub = rad[sl]
    return float(sub.max()) if sub.size else float("nan")


def _skeleton_walk(skel: np.ndarray, p, steps: int) -> np.ndarray:
    """Skeleton pixels within ``steps`` 8-connected hops of ``p``."""
    sl = _win(skel.shape, p[0], p[1], steps + 2)
    sub = skel[sl]
    if not sub.any():
        return np.zeros((0, 2), dtype=float)
    pr = int(np.clip(int(round(p[0])) - sl[0].start, 0, sub.shape[0] - 1))
    pc = int(np.clip(int(round(p[1])) - sl[1].start, 0, sub.shape[1] - 1))
    if not sub[pr, pc]:
        _, idx = ndi.distance_transform_edt(~sub, return_indices=True)
        pr, pc = int(idx[0][pr, pc]), int(idx[1][pr, pc])
    reach = np.zeros_like(sub)
    reach[pr, pc] = True
    for _ in range(int(steps)):
        reach = ndi.binary_dilation(reach, structure=SK.EIGHT.astype(bool)) & sub
    rr, cc = np.nonzero(reach)
    return np.stack([rr + sl[0].start, cc + sl[1].start], axis=1).astype(float)


def _tangent_on_skeleton(skel: np.ndarray, p, r: float) -> np.ndarray:
    """Principal direction of the skeleton stretch around ``p`` (unit, row/col)."""
    steps = int(max(4, round(3.0 * max(r if np.isfinite(r) else 1.0, 1.0))))
    pts = _skeleton_walk(skel, p, steps)
    if pts.shape[0] < 3:
        return np.array([1.0, 0.0])
    q = pts - pts.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover
        return np.array([1.0, 0.0])
    return _unit(vt[0])


def _curvature_on_skeleton(skel: np.ndarray, p, r: float) -> float:
    """``1 - chord/arc`` of the local skeleton stretch: 0 straight, -> 1 bent."""
    steps = int(max(4, round(3.0 * max(r if np.isfinite(r) else 1.0, 1.0))))
    pts = _skeleton_walk(skel, p, steps)
    if pts.shape[0] < 3:
        return float("nan")
    d = np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1])
    far = pts[np.argsort(d)[-2:]]
    chord = float(np.hypot(*(far[0] - far[1])))
    arc = float(d.max() * 2.0)
    if arc < 1e-6:
        return float("nan")
    return float(max(0.0, 1.0 - chord / arc))


def _chord_pixels(M: np.ndarray, a, b):
    n = max(3, int(round(float(np.hypot(a[0] - b[0], a[1] - b[1])))) + 1)
    rr = np.clip(np.rint(np.linspace(a[0], b[0], n)).astype(int), 0, M.shape[0] - 1)
    cc = np.clip(np.rint(np.linspace(a[1], b[1], n)).astype(int), 0, M.shape[1] - 1)
    return rr, cc, n


def _gap_length(M: np.ndarray, a, b) -> float:
    """Length of the background run along the ``a -> b`` chord."""
    rr, cc, n = _chord_pixels(M, a, b)
    step = float(np.hypot(a[0] - b[0], a[1] - b[1])) / max(n - 1, 1)
    return float(np.count_nonzero(~M[rr, cc]) * step)


def _chord_mask_fraction(M: np.ndarray, a, b) -> float:
    rr, cc, _ = _chord_pixels(M, a, b)
    return float(M[rr, cc].mean())


def _window_fraction(M: np.ndarray, f: np.ndarray, p, pad: float) -> float:
    sl = _win(M.shape, p[0], p[1], pad)
    ff = f[sl]
    if not ff.any():
        return float("nan")
    return float((M[sl] & ff).sum()) / float(ff.sum())


def _window_count(M: np.ndarray, p, pad: float) -> float:
    sl = _win(M.shape, p[0], p[1], pad)
    return float(np.count_nonzero(M[sl]))


def _frangi_response(image, sl, sigmas):
    """Frangi ridge response of the *inverted* green channel on a crop."""
    g = _green(image)
    if g is None:
        return None
    from skimage.filters import frangi

    crop = g[sl]
    if crop.size < 64:
        return None
    v = crop.astype(np.float32)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        return None
    v = (v - lo) / (hi - lo)
    try:
        return frangi(1.0 - v, sigmas=sigmas, black_ridges=False).astype(np.float32)
    except Exception:  # noqa: BLE001 - skimage version differences
        return None


def _frangi_stats(image, M, f, a: np.ndarray, pb, rbar: float) -> Dict[str, float]:
    """Frangi ridge response inside the gap vs. in the surrounding ring."""
    out = {"phi_frangi_gap": float("nan"), "phi_frangi_ring": float("nan"),
           "phi_frangi_ratio": float("nan")}
    if _green(image) is None:
        return out
    b = np.array(pb, dtype=float) if pb is not None else a
    mid = 0.5 * (a + b)
    d = float(np.hypot(*(a - b)))
    pad = max(16.0, 1.5 * d, 12.0 * max(rbar, 1.0))
    sl = _win(np.asarray(M).shape, mid[0], mid[1], pad)
    sig = max(1.0, rbar)
    resp = _frangi_response(image, sl, sigmas=(0.7 * sig, sig, 1.6 * sig))
    if resp is None:
        return out
    h, w = resp.shape
    rr, cc_ = np.mgrid[0:h, 0:w]
    ar, ac = a[0] - sl[0].start, a[1] - sl[1].start
    br, bc = b[0] - sl[0].start, b[1] - sl[1].start
    vy, vx = br - ar, bc - ac
    L2 = vy * vy + vx * vx
    if L2 < 1e-6:
        dseg = np.hypot(rr - ar, cc_ - ac)
    else:
        t = np.clip(((rr - ar) * vy + (cc_ - ac) * vx) / L2, 0.0, 1.0)
        dseg = np.hypot(rr - ar - t * vy, cc_ - ac - t * vx)
    rw = max(1.5, rbar)
    gap = dseg <= rw
    ring = (dseg > 3.0 * rw) & (dseg <= 8.0 * rw) & (~np.asarray(M)[sl]) & np.asarray(f)[sl]
    if gap.any():
        out["phi_frangi_gap"] = float(np.mean(resp[gap]))
    if ring.any():
        out["phi_frangi_ring"] = float(np.mean(resp[ring]))
    if (np.isfinite(out["phi_frangi_gap"]) and np.isfinite(out["phi_frangi_ring"])
            and out["phi_frangi_ring"] > 1e-9):
        out["phi_frangi_ratio"] = out["phi_frangi_gap"] / out["phi_frangi_ring"]
    return out


# --------------------------------------------------------------------------- #
# ground-truth-only stratification variables (NEVER mixed into phi)
# --------------------------------------------------------------------------- #
def gt_strata(ctx: GTContext, locus: Locus) -> Dict[str, float]:
    """GT-only stratification variables (Contract F: scientific analysis only)."""
    return {
        "gt_branch_order": float(locus.branch_order),
        "gt_branch_length": float(locus.gt_branch_length),
        "gt_radius_bin": float(locus.radius_bin),
        "gt_branch_type": float(locus.branch_type),
        "gt_r_loc": float(locus.r_loc),
        "gt_d_loc": float(locus.d_loc),
        "gt_dist_junction": float(locus.dist_junction),
        "gt_dist_endpoint": float(locus.dist_endpoint),
        "gt_zone": float(locus.zone),
        "gt_density": float(locus.density),
        "gt_contrast": float(locus.contrast),
        "gt_density_bin": float(locus.density_bin),
        "gt_contrast_bin": float(locus.contrast_bin),
    }


# --------------------------------------------------------------------------- #
# new endpoints created by a cut (needed to build phi for severing events)
# --------------------------------------------------------------------------- #
def new_endpoints(ctx: GTContext, mask_prime: np.ndarray, locus: Locus, search_pad: float = 0.0):
    """The two endpoints of ``M'`` nearest the cut, one on each side.

    Computed from ``M'`` alone (plus the event's own coordinates, which are its
    identity, not ground truth).  Returns ``(point_a, point_b)`` or
    ``(None, None)`` when the cut did not leave two clean stubs.
    """
    pad = search_pad if search_pad > 0 else max(16.0, 6.0 * locus.d_loc)
    sl = _win(mask_prime.shape, locus.row, locus.col, pad)
    sub = SK.as_bool(mask_prime)[sl]
    if not sub.any():
        return None, None
    skel = SK.skeletonize_mask(sub, min_branch_px=ctx.min_branch_px)
    deg = SK.neighbour_count(skel)
    rr, cc = np.nonzero(skel & (deg == 1))
    if rr.size < 2:
        return None, None
    pr = locus.row - sl[0].start
    pc = locus.col - sl[1].start
    t = np.asarray(locus.tangent, dtype=float)
    proj = (rr - pr) * t[0] + (cc - pc) * t[1]
    dist = np.hypot(rr - pr, cc - pc)
    pos = np.nonzero(proj > 0)[0]
    neg = np.nonzero(proj < 0)[0]
    if pos.size == 0 or neg.size == 0:
        return None, None
    ia = pos[int(np.argmin(dist[pos]))]
    ib = neg[int(np.argmin(dist[neg]))]
    return (
        (int(rr[ia] + sl[0].start), int(cc[ia] + sl[1].start)),
        (int(rr[ib] + sl[0].start), int(cc[ib] + sl[1].start)),
    )
