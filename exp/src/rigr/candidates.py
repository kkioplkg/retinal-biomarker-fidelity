"""Candidate generation for RiGR (proposal v3 section 3.2.1).

Input is only ``(M_hat, P, fov)`` -- never the reference annotation.

Pipeline
--------
1. Skeletonise ``M_hat`` (``src.topo.skeleton``, foreground 8-connectivity,
   spur pruning) and label the predicted components.
2. **Dangling endpoints** = degree-1 skeleton pixels at least ``border_px``
   (default 10) inside the FOV.  For each one, walk back along its own branch
   for ``trace_px`` (default 7) skeleton pixels, run a PCA on those coordinates
   and orient the principal axis *outward* (pointing away from the vessel
   body).  The local radius ``r_i`` is the mean of ``EDT(M_hat)`` over the
   traced pixels.
3. **Endpoint-endpoint candidates** need, simultaneously,
   ``d_ij / rbar_ij <= alpha`` (alpha = 6), tangent compatibility
   ``cos(theta_i, u_ij) >= tau1`` and ``cos(theta_j, -u_ij) >= tau1``
   (tau1 = 0.5), radius compatibility ``|log(r_i/r_j)| <= log 2``, and no
   *unrelated* vessel crossed roughly perpendicularly by the straight segment
   (a component containing neither endpoint).
4. **Endpoint-port (T-type) candidates**: for every endpoint, mask pixels of
   *other* components lying inside the tangent cone (``cos >= 0.5``) within
   ``6 * r_i``.  Each such pixel is snapped to the nearest skeleton pixel;
   attachment points on the *same skeleton branch* within ``2 * rbar`` are
   single-linkage clustered into one **attachment-port pseudo-node** of
   capacity 1 (section 3.2.1).
5. **Corridor** ``Omega_ij``: the bounding box of the two anchors expanded by
   ``m = 2 * max(r_i, r_j, d_ij)`` and clipped to the image.  The A* search
   grid is capped at ``128`` px per side; a larger corridor is downsampled by
   an integer factor ``ds = ceil(max_side / 128)`` (which is 2 for essentially
   every candidate that exceeds the cap -- a strict generalisation of the
   "downsample by 2" rule).

Everything is numpy/scipy/skimage; no torch, no GPU.
Smoke demo: ``cd exp && python -m src.rigr.candidates``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from src.topo import skeleton as sk

__all__ = [
    "ALPHA_D_OVER_R",
    "TAU_TANGENT",
    "LOG_RADIUS_TOL",
    "BORDER_BAND_PX",
    "TRACE_PX",
    "PORT_RANGE_R",
    "PORT_CONE_COS",
    "PORT_CLUSTER_R",
    "CORRIDOR_MARGIN",
    "CORRIDOR_MAX_SIDE",
    "local_orientation_field",
    "propagate_to_mask",
    "axial_cos",
    "endpoint_tangent",
    "find_endpoints",
    "branch_labels",
    "attachment_port",
    "crosses_unrelated_vessel",
    "corridor_box",
    "generate_candidates",
    "CANDIDATE_COLUMNS",
]

# ---- pre-registered thresholds (section 3.2.1) ---------------------------
ALPHA_D_OVER_R = 6.0          #: max normalised gap length d_ij / rbar_ij
TAU_TANGENT = 0.5             #: min cos between a tangent and the chord
LOG_RADIUS_TOL = math.log(2)  #: max |log(r_i / r_j)|
BORDER_BAND_PX = 10           #: endpoints must be >= this far inside the FOV
TRACE_PX = 7                  #: skeleton pixels used for the endpoint PCA
PORT_RANGE_R = 6.0            #: T-type search radius, in units of r_i
PORT_CONE_COS = 0.5           #: half-angle cosine of the T-type search cone
PORT_CLUSTER_R = 2.0          #: port clustering distance, in units of rbar
CORRIDOR_MARGIN = 2.0         #: m = CORRIDOR_MARGIN * max(r_i, r_j, d_ij)
CORRIDOR_MAX_SIDE = 128       #: A* search grid cap, per side

_EPS = 1e-9


# --------------------------------------------------------------------------
# orientation fields
# --------------------------------------------------------------------------
def local_orientation_field(
    skel: np.ndarray, win: int = TRACE_PX
) -> Tuple[np.ndarray, np.ndarray]:
    """Axial orientation of a skeleton from a local PCA in a ``win x win`` box.

    The PCA is computed in closed form from box-filtered coordinate moments, so
    the whole field costs six separable convolutions rather than one
    eigen-decomposition per pixel.  With ``(dr, dc)`` the tangent vector in
    numpy index order, the returned angle is

        ``theta = atan2(dr, dc)  (mod pi)``

    i.e. **axial**: ``theta`` and ``theta + pi`` denote the same orientation,
    which is exactly what ``q(theta | x, vessel)`` is discretised over.

    Returns
    -------
    (theta, count) : ((H, W) float32 in [0, pi), (H, W) int32)
        ``theta`` is 0 outside the skeleton; ``count`` is the number of
        skeleton pixels inside the window (a coherence proxy -- 1 or 2 means
        the PCA is degenerate).
    """
    s_bool = sk.as_bool(skel)
    s = s_bool.astype(np.float64)
    h = int(win) // 2
    ax = np.arange(-h, h + 1, dtype=np.float64)
    one = np.ones_like(ax)

    def _sep(kr, kc):
        # correlation (not convolution) with the separable kernel kr (x) kc
        t = ndi.correlate1d(s, kr, axis=0, mode="constant", cval=0.0, origin=0)
        return ndi.correlate1d(t, kc, axis=1, mode="constant", cval=0.0, origin=0)

    n = _sep(one, one)
    m10 = _sep(ax, one)          # sum of di
    m01 = _sep(one, ax)          # sum of dj
    m20 = _sep(ax * ax, one)
    m02 = _sep(one, ax * ax)
    m11 = _sep(ax, ax)

    nz = np.maximum(n, 1.0)
    mr = m10 / nz
    mc = m01 / nz
    c_rr = m20 / nz - mr * mr
    c_cc = m02 / nz - mc * mc
    c_rc = m11 / nz - mr * mc

    theta = 0.5 * np.arctan2(2.0 * c_rc, c_cc - c_rr)
    theta = np.mod(theta, np.pi).astype(np.float32)
    theta[~s_bool] = 0.0
    return theta, n.astype(np.int32)


def propagate_to_mask(
    theta_skel: np.ndarray, skel: np.ndarray, where: Optional[np.ndarray] = None
) -> np.ndarray:
    """Give every pixel the orientation of its **nearest skeleton pixel**.

    This is the rule used both for the head's orientation targets on non-
    skeleton vessel pixels and for the "unrelated vessel" crossing test.
    """
    s = sk.as_bool(skel)
    if not s.any():
        return np.zeros(s.shape, dtype=np.float32)
    _, idx = ndi.distance_transform_edt(~s, return_indices=True)
    out = np.asarray(theta_skel, dtype=np.float32)[idx[0], idx[1]]
    if where is not None:
        out = np.where(sk.as_bool(where), out, 0.0)
    return np.ascontiguousarray(out, dtype=np.float32)


def axial_cos(a, b):
    """``|cos(a - b)|`` -- the axial (mod pi) alignment of two angles."""
    return np.abs(np.cos(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------
_NB8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _walk_branch(skel: np.ndarray, start: Tuple[int, int], n: int) -> np.ndarray:
    """Walk ``n`` skeleton pixels inward from an endpoint, stopping at a fork."""
    H, W = skel.shape
    pts = [(int(start[0]), int(start[1]))]
    seen = {pts[0]}
    cur = pts[0]
    for _ in range(int(n) - 1):
        nbrs = []
        for dr, dc in _NB8:
            r, c = cur[0] + dr, cur[1] + dc
            if 0 <= r < H and 0 <= c < W and skel[r, c] and (r, c) not in seen:
                nbrs.append((r, c))
        if len(nbrs) != 1:      # 0 = dead end, >1 = junction
            break
        cur = nbrs[0]
        seen.add(cur)
        pts.append(cur)
    return np.asarray(pts, dtype=np.int64)


def endpoint_tangent(
    skel: np.ndarray, ep: Tuple[int, int], n: int = TRACE_PX
) -> Tuple[np.ndarray, np.ndarray]:
    """Outward unit tangent at an endpoint from a PCA over the last ``n`` px.

    Returns ``(tangent, traced_points)``.  ``tangent`` is a **directed** unit
    vector ``(dr, dc)`` pointing away from the vessel body, i.e. into the gap;
    the corresponding axial angle is ``atan2(dr, dc) mod pi``.
    """
    pts = _walk_branch(sk.as_bool(skel), ep, n)
    if pts.shape[0] < 2:
        return np.array([0.0, 0.0]), pts
    x = pts.astype(np.float64)
    mu = x.mean(axis=0)
    xc = x - mu
    cov = xc.T @ xc / max(1, len(xc) - 1)
    w, v = np.linalg.eigh(cov)
    d = v[:, int(np.argmax(w))]
    outward = x[0] - mu                       # endpoint minus centroid
    if float(d @ outward) < 0:
        d = -d
    nrm = float(np.linalg.norm(d))
    return (d / nrm if nrm > _EPS else np.array([0.0, 0.0])), pts


def find_endpoints(
    mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    skel: Optional[np.ndarray] = None,
    radius: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    border_px: float = BORDER_BAND_PX,
    trace_px: int = TRACE_PX,
    min_branch_px: int = 3,
) -> Dict[str, np.ndarray]:
    """Dangling endpoints of ``mask`` with their tangent, radius and component.

    Endpoints whose distance to the outside of the FOV is below ``border_px``
    are dropped: a vessel that simply leaves the image is not a repairable
    dangling end.

    Returns a dict of parallel arrays: ``coord`` (N, 2) int, ``tangent``
    (N, 2) float outward unit vector, ``theta`` (N,) axial angle,
    ``radius`` (N,), ``comp`` (N,) component label, ``n_traced`` (N,).
    """
    m = sk.as_bool(mask)
    f = sk.fov_or_true(fov, m.shape)
    m = m & f
    if skel is None:
        skel = sk.skeletonize_mask(m, min_branch_px=min_branch_px, prune=True, fov=f)
    if radius is None:
        radius = sk.local_radius(m)
    if labels is None:
        labels, _ = sk.connected_components(m, connectivity=8)

    empty = dict(coord=np.zeros((0, 2), np.int64), tangent=np.zeros((0, 2)),
                 theta=np.zeros(0), radius=np.zeros(0),
                 comp=np.zeros(0, np.int64), n_traced=np.zeros(0, np.int64))
    eps = sk.endpoints(skel)
    if eps.size == 0:
        return empty

    fov_dist = ndi.distance_transform_edt(f)
    eps = eps[fov_dist[eps[:, 0], eps[:, 1]] >= float(border_px)]
    if eps.size == 0:
        return empty

    coords, tans, thetas, radii, comps, ntr = [], [], [], [], [], []
    for r, c in eps:
        t, pts = endpoint_tangent(skel, (r, c), trace_px)
        if float(np.linalg.norm(t)) < 0.5:
            continue                           # isolated pixel: no tangent
        rr = float(np.mean(radius[pts[:, 0], pts[:, 1]]))
        coords.append((int(r), int(c)))
        tans.append(t)
        thetas.append(math.atan2(float(t[0]), float(t[1])) % math.pi)
        radii.append(max(rr, 0.5))
        comps.append(int(labels[r, c]))
        ntr.append(int(pts.shape[0]))
    if not coords:
        return empty
    return dict(
        coord=np.asarray(coords, dtype=np.int64).reshape(-1, 2),
        tangent=np.asarray(tans, dtype=float).reshape(-1, 2),
        theta=np.asarray(thetas, dtype=float),
        radius=np.asarray(radii, dtype=float),
        comp=np.asarray(comps, dtype=np.int64),
        n_traced=np.asarray(ntr, dtype=np.int64),
    )


def branch_labels(skel: np.ndarray) -> np.ndarray:
    """Label the skeleton branches (8-connected runs of non-junction pixels)."""
    s = sk.as_bool(skel)
    deg = sk.neighbour_count(s)
    seg = s & (deg < 3)
    lab, _ = ndi.label(seg, structure=sk.EIGHT)
    return lab


# --------------------------------------------------------------------------
# attachment ports
# --------------------------------------------------------------------------
def attachment_port(pix: np.ndarray, radius: np.ndarray,
                    theta_skel: np.ndarray) -> Tuple[int, int, float, float]:
    """Collapse one cluster of attachment pixels into a port pseudo-node.

    **The exact port-clustering rule** (proposal v3 section 3.2.1), stated here
    so the paper can quote it verbatim:

    1. *Admissible attachment pixels.*  For a dangling endpoint ``i`` with
       outward tangent ``t_i`` and local radius ``r_i``, every predicted-mask
       pixel of a **different** connected component that lies inside the
       tangent cone (``cos(angle to t_i) >= 0.5``) and within ``6 r_i`` of the
       endpoint is snapped to its nearest skeleton pixel.  That skeleton pixel
       is an attachment pixel, tagged with the id of the **skeleton branch**
       (an 8-connected run of non-junction skeleton pixels) it lies on.
    2. *Clustering.*  Attachment pixels are grouped **per branch** -- two
       pixels on different branches are never merged, even if adjacent across a
       junction -- and single-linkage clustered with the distance threshold
       ``2 r_med``, where ``r_med`` is the median of ``EDT(M_hat)`` over that
       branch's attachment pixels.  So attachment points closer than ``2 r``
       along one branch become one port; points further apart become separate
       ports and can each accept a different incoming branch.
    3. *Representative pixel.*  The port's location is the cluster member
       **nearest the cluster centroid** -- an actual skeleton pixel, never an
       interpolated centroid -- and the port's **radius and axial tangent are
       read at that pixel** (``EDT(M_hat)`` and the local-PCA orientation
       field respectively).  No averaging is performed, which also avoids the
       wrap-around bias of a naive arithmetic mean of angles modulo pi.
    4. *Capacity.*  Each port pseudo-node carries capacity 1, exactly like an
       endpoint, so the selection stage remains a standard maximum-weight
       matching on ``V_c = V_end union V_port``.
    5. *Asymmetry.*  On the port side the reverse tangent-compatibility test is
       deliberately **not** applied: a T-junction arrives across the trunk, so
       its arrival orientation is unconstrained (and the A* goal accepts any
       orientation bin at a port).

    Returns ``(row, col, radius, theta)``.
    """
    pix = np.asarray(pix, dtype=np.int64).reshape(-1, 2)
    cen = pix.mean(axis=0)
    j = int(np.argmin(np.hypot(pix[:, 0] - cen[0], pix[:, 1] - cen[1])))
    r, c = int(pix[j, 0]), int(pix[j, 1])
    return r, c, max(float(radius[r, c]), 0.5), float(theta_skel[r, c])


# --------------------------------------------------------------------------
# obstruction test
# --------------------------------------------------------------------------
def crosses_unrelated_vessel(
    path: np.ndarray,
    labels: np.ndarray,
    allowed: Sequence[int],
    theta_mask: np.ndarray,
    radius: np.ndarray,
    min_run: int = 2,
    cos_thr: float = TAU_TANGENT,
    min_radius: float = 1.0,
) -> bool:
    """True when ``path`` cuts roughly perpendicularly through a third vessel.

    A run of >= ``min_run`` consecutive path pixels belonging to a connected
    component that is **not** in ``allowed`` and whose local radius is at least
    ``min_radius`` is a crossing.  It is *perpendicular* -- and therefore a
    veto -- when the mean axial alignment ``|cos(path_dir - vessel_dir)|`` over
    the run is below ``cos_thr`` (default 0.5, i.e. more than 60 degrees off).
    Running *along* a neighbouring vessel is not vetoed here; that is the job
    of the tangent / radius filters and of the scorer.
    """
    p = np.asarray(path, dtype=np.int64).reshape(-1, 2)
    H, W = labels.shape
    inside = (p[:, 0] >= 0) & (p[:, 0] < H) & (p[:, 1] >= 0) & (p[:, 1] < W)
    p = p[inside]
    if p.shape[0] < 2:
        return False
    lab = labels[p[:, 0], p[:, 1]]
    ok = set(int(a) for a in allowed) | {0}
    foreign = np.array([int(l) not in ok for l in lab], dtype=bool)
    foreign &= radius[p[:, 0], p[:, 1]] >= float(min_radius)
    if not foreign.any():
        return False

    d = np.zeros_like(p, dtype=float)
    d[1:-1] = p[2:] - p[:-2]
    d[0] = p[1] - p[0]
    d[-1] = p[-1] - p[-2]
    ang = np.arctan2(d[:, 0], d[:, 1])

    flags = foreign.astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flags, [0]))))
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < int(min_run):
            continue
        seg = p[a:b]
        al = axial_cos(ang[a:b], theta_mask[seg[:, 0], seg[:, 1]])
        if float(np.mean(al)) < float(cos_thr):
            return True
    return False


# --------------------------------------------------------------------------
# corridor
# --------------------------------------------------------------------------
def corridor_box(
    p0, p1, r0: float, r1: float, shape: Tuple[int, int],
    margin: float = CORRIDOR_MARGIN, max_side: int = CORRIDOR_MAX_SIDE,
) -> Tuple[int, int, int, int, int]:
    """Corridor bounding box and A* downsampling factor.

    ``m = margin * max(r0, r1, d)``; the box is the anchors' bbox grown by
    ``m`` and clipped to the image.  ``ds = ceil(box_side / max_side)`` keeps
    the lifted search grid at most ``max_side`` per side.

    Returns ``(r_lo, c_lo, r_hi, c_hi, ds)`` with ``r_hi`` / ``c_hi`` exclusive.
    """
    H, W = shape
    d = float(np.hypot(float(p0[0]) - float(p1[0]), float(p0[1]) - float(p1[1])))
    m = float(margin) * max(float(r0), float(r1), d)
    r_lo = int(max(0, math.floor(min(p0[0], p1[0]) - m)))
    r_hi = int(min(H, math.ceil(max(p0[0], p1[0]) + m) + 1))
    c_lo = int(max(0, math.floor(min(p0[1], p1[1]) - m)))
    c_hi = int(min(W, math.ceil(max(p0[1], p1[1]) + m) + 1))
    side = max(r_hi - r_lo, c_hi - c_lo)
    ds = int(max(1, math.ceil(side / float(max_side))))
    return r_lo, c_lo, r_hi, c_hi, ds


# --------------------------------------------------------------------------
# candidate table
# --------------------------------------------------------------------------
CANDIDATE_COLUMNS = [
    "cand_id", "kind", "node_i", "node_j",
    "r_i", "c_i", "r_j", "c_j",
    "rad_i", "rad_j", "rad_bar",
    "theta_i", "theta_j", "tan_i_r", "tan_i_c", "tan_j_r", "tan_j_c",
    "d", "d_over_rbar", "log_r_ratio", "tangent_mismatch",
    "comp_i", "comp_j", "is_port",
    "box_r0", "box_c0", "box_r1", "box_c1", "box_ds",
]


def _mismatch_angle(t_i, t_j, u) -> float:
    """Mean angular mismatch (radians) of the two tangents with the chord."""
    ci = float(np.clip(np.dot(t_i, u), -1.0, 1.0))
    cj = float(np.clip(np.dot(t_j, -u), -1.0, 1.0))
    return 0.5 * (math.acos(ci) + math.acos(cj))


def _empty_table():
    import pandas as pd

    return pd.DataFrame({c: pd.Series(dtype="float64") for c in CANDIDATE_COLUMNS})


def generate_candidates(
    mask: np.ndarray,
    prob: Optional[np.ndarray] = None,
    fov: Optional[np.ndarray] = None,
    alpha: float = ALPHA_D_OVER_R,
    tau1: float = TAU_TANGENT,
    log_r_tol: float = LOG_RADIUS_TOL,
    border_px: float = BORDER_BAND_PX,
    trace_px: int = TRACE_PX,
    port_range_r: float = PORT_RANGE_R,
    port_cone_cos: float = PORT_CONE_COS,
    port_cluster_r: float = PORT_CLUSTER_R,
    enable_ports: bool = True,
    margin: float = CORRIDOR_MARGIN,
    max_side: int = CORRIDOR_MAX_SIDE,
    min_branch_px: int = 3,
    max_candidates: Optional[int] = None,
) -> Dict[str, object]:
    """Full candidate generation from ``(M_hat, P, fov)``.

    ``prob`` is accepted for API symmetry (and is echoed back) but the
    candidate *geometry* deliberately depends on ``M_hat`` only; ``P`` enters
    later, through the scorer features and the A* appearance term.

    Returns a dict with

    ``table``       pandas DataFrame, one row per candidate (:data:`CANDIDATE_COLUMNS`)
    ``nodes``       DataFrame of graph nodes (endpoints + port pseudo-nodes)
    ``port_pixels`` dict ``port node_id -> (k, 2) int array`` of that port's
                    skeleton pixels (the A* goal set of a T-type candidate)
    ``skel``, ``radius``, ``labels``, ``theta_skel``, ``theta_mask``, ``mask``,
    ``fov``         shared per-image fields, so downstream stages do not
                    recompute them.
    """
    import pandas as pd
    from src.eval.trr_fcr import rasterize_path

    m = sk.as_bool(mask)
    f = sk.fov_or_true(fov, m.shape)
    m = m & f
    skel = sk.skeletonize_mask(m, min_branch_px=min_branch_px, prune=True, fov=f)
    radius = sk.local_radius(m)
    labels, _ = sk.connected_components(m, connectivity=8)
    theta_skel, _ = local_orientation_field(skel, win=trace_px)
    theta_mask = propagate_to_mask(theta_skel, skel)

    ep = find_endpoints(m, fov=f, skel=skel, radius=radius, labels=labels,
                        border_px=border_px, trace_px=trace_px,
                        min_branch_px=min_branch_px)
    n_ep = int(ep["coord"].shape[0])

    nodes: List[dict] = []
    for i in range(n_ep):
        nodes.append(dict(
            node_id=i, kind="end", row=int(ep["coord"][i, 0]),
            col=int(ep["coord"][i, 1]), radius=float(ep["radius"][i]),
            theta=float(ep["theta"][i]), tan_r=float(ep["tangent"][i, 0]),
            tan_c=float(ep["tangent"][i, 1]), comp=int(ep["comp"][i]), branch=-1,
        ))

    rows: List[dict] = []
    cid = 0

    # ---- endpoint-endpoint -------------------------------------------------
    if n_ep >= 2:
        C = ep["coord"].astype(float)
        R = ep["radius"]
        T = ep["tangent"]
        for i in range(n_ep):
            for j in range(i + 1, n_ep):
                rbar = 0.5 * (R[i] + R[j])
                v = C[j] - C[i]
                d = float(np.hypot(v[0], v[1]))
                if d < 1e-6 or d > alpha * rbar:
                    continue
                u = v / d
                if float(T[i] @ u) < tau1 or float(T[j] @ (-u)) < tau1:
                    continue
                lr = abs(math.log(max(R[i], _EPS) / max(R[j], _EPS)))
                if lr > log_r_tol:
                    continue
                line = rasterize_path([tuple(ep["coord"][i]), tuple(ep["coord"][j])])
                if crosses_unrelated_vessel(
                    line, labels, [int(ep["comp"][i]), int(ep["comp"][j])],
                    theta_mask, radius, cos_thr=tau1,
                ):
                    continue
                b = corridor_box(C[i], C[j], R[i], R[j], m.shape, margin, max_side)
                rows.append(dict(
                    cand_id=cid, kind="ee", node_i=i, node_j=j,
                    r_i=int(C[i, 0]), c_i=int(C[i, 1]),
                    r_j=int(C[j, 0]), c_j=int(C[j, 1]),
                    rad_i=float(R[i]), rad_j=float(R[j]), rad_bar=float(rbar),
                    theta_i=float(ep["theta"][i]), theta_j=float(ep["theta"][j]),
                    tan_i_r=float(T[i, 0]), tan_i_c=float(T[i, 1]),
                    tan_j_r=float(T[j, 0]), tan_j_c=float(T[j, 1]),
                    d=d, d_over_rbar=float(d / max(rbar, _EPS)),
                    log_r_ratio=float(lr),
                    tangent_mismatch=_mismatch_angle(T[i], T[j], u),
                    comp_i=int(ep["comp"][i]), comp_j=int(ep["comp"][j]),
                    is_port=0,
                    box_r0=b[0], box_c0=b[1], box_r1=b[2], box_c1=b[3], box_ds=b[4],
                ))
                cid += 1

    # ---- endpoint-port (T-type) -------------------------------------------
    port_pixels: Dict[int, np.ndarray] = {}
    if enable_ports and n_ep >= 1 and skel.any():
        bl = branch_labels(skel)
        _, near_idx = ndi.distance_transform_edt(~skel, return_indices=True)
        attach: List[Tuple[int, int, int, int]] = []   # (endpoint, branch, r, c)
        for i in range(n_ep):
            r0, c0 = ep["coord"][i]
            reach = float(port_range_r) * float(ep["radius"][i])
            rr0 = int(max(0, math.floor(r0 - reach)))
            rr1 = int(min(m.shape[0], math.ceil(r0 + reach) + 1))
            cc0 = int(max(0, math.floor(c0 - reach)))
            cc1 = int(min(m.shape[1], math.ceil(c0 + reach) + 1))
            sub = m[rr0:rr1, cc0:cc1]
            sub_lab = labels[rr0:rr1, cc0:cc1]
            yy, xx = np.nonzero(sub & (sub_lab != int(ep["comp"][i])))
            if yy.size == 0:
                continue
            dv = np.stack([yy + rr0 - r0, xx + cc0 - c0], axis=1).astype(float)
            dist = np.maximum(np.hypot(dv[:, 0], dv[:, 1]), _EPS)
            cosang = (dv @ ep["tangent"][i]) / dist
            good = (dist <= reach) & (cosang >= port_cone_cos) & (dist > 1.0)
            if not good.any():
                continue
            pr = (yy + rr0)[good]
            pc = (xx + cc0)[good]
            sr = near_idx[0][pr, pc]
            sc = near_idx[1][pr, pc]
            for a, b_ in zip(sr, sc):
                br = int(bl[a, b_])
                if br <= 0:
                    continue
                attach.append((i, br, int(a), int(b_)))

        by_branch: Dict[int, List[Tuple[int, int, int]]] = {}
        for i, br, a, b_ in attach:
            by_branch.setdefault(br, []).append((i, a, b_))
        next_id = n_ep
        for br in sorted(by_branch):
            items = by_branch[br]
            pts = np.array([[a, b_] for _, a, b_ in items], dtype=np.int64)
            uniq = np.unique(pts, axis=0)
            rad_here = radius[uniq[:, 0], uniq[:, 1]]
            thr = float(port_cluster_r) * float(np.median(rad_here) + _EPS)
            # single-linkage clustering of the attachment points on this branch
            cl = -np.ones(len(uniq), dtype=int)
            k = 0
            for oi in range(len(uniq)):
                if cl[oi] >= 0:
                    continue
                stack = [oi]
                cl[oi] = k
                while stack:
                    p = stack.pop()
                    dd = np.hypot(uniq[:, 0] - uniq[p, 0], uniq[:, 1] - uniq[p, 1])
                    for q in np.flatnonzero((dd <= thr) & (cl < 0)):
                        cl[q] = k
                        stack.append(int(q))
                k += 1
            for cluster in range(k):
                pix = uniq[cl == cluster]
                # representative pixel + its own radius/tangent: see
                # attachment_port() for the rule the paper quotes.
                pr_, pc_, port_rad, port_theta = attachment_port(
                    pix, radius, theta_skel)
                pid = next_id
                next_id += 1
                port_pixels[pid] = pix
                port_comp = int(labels[pix[0, 0], pix[0, 1]])
                nodes.append(dict(
                    node_id=pid, kind="port", row=pr_, col=pc_,
                    radius=port_rad, theta=port_theta,
                    tan_r=float(math.sin(port_theta)),
                    tan_c=float(math.cos(port_theta)),
                    comp=port_comp, branch=int(br),
                ))
                members = sorted({i for (i, a, b_) in items
                                  if np.any((pix[:, 0] == a) & (pix[:, 1] == b_))})
                for i in members:
                    ci_, cj_ = int(ep["coord"][i, 0]), int(ep["coord"][i, 1])
                    v = np.array([pr_ - ci_, pc_ - cj_], dtype=float)
                    d = float(np.hypot(v[0], v[1]))
                    if d < 1e-6:
                        continue
                    u = v / d
                    ti = ep["tangent"][i]
                    rbar = 0.5 * (float(ep["radius"][i]) + port_rad)
                    line = rasterize_path([(ci_, cj_), (pr_, pc_)])
                    if crosses_unrelated_vessel(
                        line, labels, [int(ep["comp"][i]), port_comp],
                        theta_mask, radius, cos_thr=tau1,
                    ):
                        continue
                    b = corridor_box((ci_, cj_), (pr_, pc_),
                                     float(ep["radius"][i]), port_rad,
                                     m.shape, margin, max_side)
                    # Port side: the trunk runs *across* the incoming end, so
                    # the reverse tangent test is deliberately NOT applied.
                    rows.append(dict(
                        cand_id=cid, kind="ep", node_i=i, node_j=pid,
                        r_i=ci_, c_i=cj_, r_j=pr_, c_j=pc_,
                        rad_i=float(ep["radius"][i]), rad_j=port_rad,
                        rad_bar=float(rbar),
                        theta_i=float(ep["theta"][i]), theta_j=port_theta,
                        tan_i_r=float(ti[0]), tan_i_c=float(ti[1]),
                        tan_j_r=float(math.sin(port_theta)),
                        tan_j_c=float(math.cos(port_theta)),
                        d=d, d_over_rbar=float(d / max(rbar, _EPS)),
                        log_r_ratio=float(abs(math.log(
                            max(float(ep["radius"][i]), _EPS) / max(port_rad, _EPS)))),
                        tangent_mismatch=float(math.acos(
                            float(np.clip(np.dot(ti, u), -1.0, 1.0)))),
                        comp_i=int(ep["comp"][i]), comp_j=port_comp, is_port=1,
                        box_r0=b[0], box_c0=b[1], box_r1=b[2], box_c1=b[3],
                        box_ds=b[4],
                    ))
                    cid += 1

    table = pd.DataFrame(rows, columns=CANDIDATE_COLUMNS) if rows else _empty_table()
    if max_candidates is not None and len(table) > int(max_candidates):
        table = table.nsmallest(int(max_candidates), "d_over_rbar")
    table = table.reset_index(drop=True)

    node_df = pd.DataFrame(nodes) if nodes else pd.DataFrame(
        columns=["node_id", "kind", "row", "col", "radius", "theta",
                 "tan_r", "tan_c", "comp", "branch"])
    return dict(
        table=table,
        nodes=node_df,
        port_pixels=port_pixels,
        skel=skel, radius=radius, labels=labels,
        theta_skel=theta_skel, theta_mask=theta_mask,
        fov=f, mask=m, prob=prob,
        n_endpoints=n_ep, n_ports=len(port_pixels),
    )


# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - smoke demo
    import time

    H = W = 160
    gt = np.zeros((H, W), dtype=bool)
    gt[76:84, 10:150] = True          # a horizontal trunk (radius 4)
    gt[20:78, 78:82] = True           # a branch coming down onto it (radius 2)
    gt[100:150, 40:44] = True         # an unrelated vertical vessel
    cut = gt.copy()
    cut[76:84, 55:64] = False         # sever the trunk        -> an ee candidate
    cut[70:78, 78:82] = False         # detach the branch      -> a T-type candidate

    t0 = time.time()
    out = generate_candidates(cut, fov=None)
    dt = time.time() - t0
    print("endpoints=%d ports=%d candidates=%d  (%.3f s)"
          % (out["n_endpoints"], out["n_ports"], len(out["table"]), dt))
    cols = ["cand_id", "kind", "node_i", "node_j", "d", "d_over_rbar",
            "tangent_mismatch", "box_r0", "box_r1", "box_ds"]
    if len(out["table"]):
        print(out["table"][cols].round(3).to_string(index=False))
    print(out["nodes"].round(3).to_string(index=False))
