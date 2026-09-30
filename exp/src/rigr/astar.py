"""Corridor-restricted lifted-state A* (proposal v3 section 3.2.3).

State space
-----------
``(x, y, theta_k)`` with ``K = 16`` **axial** orientation bins
(``theta_k = k pi / K``, ``theta == theta + pi``).  Moves are the 8 lattice
neighbours; a move may change the orientation bin by at most
``max_dtheta_bins`` (default 2), the circular distance being taken modulo
``K`` because the bins are axial.

Transition cost
---------------
``w = w_I + w_theta + w_kappa + w_dir`` with

* ``w_I     = -log(V_I(x') + eps) * ds``           appearance, from the head's
  vessel evidence -- *not* from the backbone's ``P``, because inside a real
  break ``P ~ 0`` and ``-log P`` would punish the true path hardest;
* ``w_theta = -lambda_theta log q(theta_k' | x') * ds``  (``lambda_theta = 1``);
* ``w_kappa = mu (Delta theta)^2 / ds``            (``mu = 2``), the natural
  discretisation of ``\\int kappa^2 ds`` -- "Euler-elastica-inspired lifted
  shortest path", *not* a claim of the continuous geodesic;
* ``w_dir   = nu (1 - |cos(phi_move - theta_k')|) * ds`` (``nu = 0.5``), a small
  penalty when the move direction disagrees with the orientation state the
  path claims to be in, so a path actually follows its own lifted state
  instead of using the orientation channel as free label noise.

Heuristic
---------
``h(x, theta) = w_min * dist_2(x, goal set)`` where

``w_min = min_x (-log(V_I + eps)) + lambda_theta * min_{x,k} (-log q)``  (>= 0
since both ``V_I`` and ``q`` are probabilities).  ``w_kappa`` and ``w_dir`` are
non-negative and are simply dropped from the bound, and orientation never
enters ``h``, so along any transition
``h(u) - h(u') <= w_min * ds <= w_I + w_theta <= w(u, u')``: ``h`` is
**consistent**, hence admissible, hence A* returns the minimum-cost path
**of the discrete lifted graph** (which is the only optimality this code and
the paper claim).

Implementation
--------------
Numba-jitted binary heap with lazy deletion; a pure-Python fallback with
``heapq`` is used when numba is unavailable.  Target: < 50 ms per candidate on
a ``128 x 128 x 16`` corridor.

Run ``cd exp && python -m src.rigr.astar`` for a correctness + timing demo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "K_BINS",
    "LAMBDA_THETA",
    "MU_KAPPA",
    "NU_DIR",
    "MAX_DTHETA_BINS",
    "EPS_PROB",
    "AStarResult",
    "astar_lifted",
    "solve_candidate",
    "warmup",
    "HAVE_NUMBA",
]

K_BINS = 16
LAMBDA_THETA = 1.0
MU_KAPPA = 2.0
NU_DIR = 0.5
MAX_DTHETA_BINS = 2
EPS_PROB = 1e-4

# 8-neighbourhood: (dy, dx, step length)
_DIRS = np.array(
    [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
    dtype=np.int64,
)
_DIR_LEN = np.array([math.sqrt(2), 1.0, math.sqrt(2), 1.0,
                     1.0, math.sqrt(2), 1.0, math.sqrt(2)], dtype=np.float64)
_DIR_ANG = np.arctan2(_DIRS[:, 0].astype(float), _DIRS[:, 1].astype(float))

try:  # pragma: no cover - depends on the environment
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def deco(f):
            return f
        return deco


# --------------------------------------------------------------------------
# numba kernel
# --------------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def _heap_push(hk, hv, hn, key, val):
    i = hn
    hk[i] = key
    hv[i] = val
    while i > 0:
        p = (i - 1) // 2
        if hk[p] <= hk[i]:
            break
        tk = hk[p]; hk[p] = hk[i]; hk[i] = tk
        tv = hv[p]; hv[p] = hv[i]; hv[i] = tv
        i = p
    return hn + 1


@njit(cache=True, fastmath=False)
def _heap_pop(hk, hv, hn):
    top_k = hk[0]
    top_v = hv[0]
    hn -= 1
    hk[0] = hk[hn]
    hv[0] = hv[hn]
    i = 0
    while True:
        l = 2 * i + 1
        r = l + 1
        s = i
        if l < hn and hk[l] < hk[s]:
            s = l
        if r < hn and hk[r] < hk[s]:
            s = r
        if s == i:
            break
        tk = hk[s]; hk[s] = hk[i]; hk[i] = tk
        tv = hv[s]; hv[s] = hv[i]; hv[i] = tv
        i = s
    return top_k, top_v, hn


@njit(cache=True, fastmath=False)
def _astar_kernel(
    cost_I,        # (h, w) float64  = -log(V_I + eps)
    cost_Q,        # (K, h, w) float64 = -log(q + eps)
    cost_dir,      # (8, K) float64 = 1 - |cos(phi_d - theta_k)|
    dtheta2,       # (K, K) float64 = (Delta theta)^2 in rad^2, -1 if forbidden
    hmap,          # (h, w) float64 = w_min * dist to the goal set
    start_states,  # (ns,) int64
    goal_ok,       # (h, w) uint8
    goal_bins,     # (K,) uint8
    dirs,          # (8, 2) int64
    dlen,          # (8,) float64  (already multiplied by the pixel scale)
    lam, mu, nu,
    blocked,       # (h, w) uint8, 1 = never enter
    cap,
):
    K = cost_Q.shape[0]
    h = cost_I.shape[0]
    w = cost_I.shape[1]
    n = h * w * K
    INF = 1e18
    g = np.full(n, INF, dtype=np.float64)
    parent = np.full(n, -1, dtype=np.int64)
    closed = np.zeros(n, dtype=np.uint8)
    hk = np.empty(cap, dtype=np.float64)
    hv = np.empty(cap, dtype=np.int64)
    hn = 0

    for i in range(start_states.shape[0]):
        s = start_states[i]
        if g[s] > 0.0:
            g[s] = 0.0
            yy = (s // K) // w
            xx = (s // K) % w
            hn = _heap_push(hk, hv, hn, hmap[yy, xx], s)

    best = -1
    n_pop = 0
    while hn > 0:
        f, s, hn = _heap_pop(hk, hv, hn)
        if closed[s] == 1:
            continue
        closed[s] = 1
        n_pop += 1
        k = s % K
        rest = s // K
        y = rest // w
        x = rest % w
        if goal_ok[y, x] == 1 and goal_bins[k] == 1:
            best = s
            break
        gs = g[s]
        for d in range(dirs.shape[0]):
            ny = y + dirs[d, 0]
            nx = x + dirs[d, 1]
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if blocked[ny, nx] == 1:
                continue
            ds = dlen[d]
            base = ds * cost_I[ny, nx]
            for k2 in range(K):
                dt2 = dtheta2[k, k2]
                if dt2 < 0.0:
                    continue
                ns_ = (ny * w + nx) * K + k2
                if closed[ns_] == 1:
                    continue
                cost = base + ds * (lam * cost_Q[k2, ny, nx] + nu * cost_dir[d, k2])
                cost += mu * dt2 / ds
                ng = gs + cost
                if ng < g[ns_]:
                    g[ns_] = ng
                    parent[ns_] = s
                    if hn < cap:
                        hn = _heap_push(hk, hv, hn, ng + hmap[ny, nx], ns_)
    return g, parent, best, n_pop


# --------------------------------------------------------------------------
# python fallback
# --------------------------------------------------------------------------
def _astar_kernel_py(cost_I, cost_Q, cost_dir, dtheta2, hmap, start_states,
                     goal_ok, goal_bins, dirs, dlen, lam, mu, nu, blocked, cap):
    import heapq

    K, h, w = cost_Q.shape
    n = h * w * K
    g = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    closed = np.zeros(n, dtype=bool)
    heap: List[Tuple[float, int]] = []
    for s in start_states:
        s = int(s)
        g[s] = 0.0
        yy, xx = (s // K) // w, (s // K) % w
        heapq.heappush(heap, (float(hmap[yy, xx]), s))
    best = -1
    n_pop = 0
    while heap:
        _, s = heapq.heappop(heap)
        if closed[s]:
            continue
        closed[s] = True
        n_pop += 1
        k = s % K
        y, x = (s // K) // w, (s // K) % w
        if goal_ok[y, x] and goal_bins[k]:
            best = s
            break
        gs = g[s]
        for d in range(len(dirs)):
            ny, nx = y + int(dirs[d, 0]), x + int(dirs[d, 1])
            if not (0 <= ny < h and 0 <= nx < w) or blocked[ny, nx]:
                continue
            ds = float(dlen[d])
            base = ds * cost_I[ny, nx]
            for k2 in range(K):
                dt2 = dtheta2[k, k2]
                if dt2 < 0:
                    continue
                ns_ = (ny * w + nx) * K + k2
                if closed[ns_]:
                    continue
                ng = gs + base + ds * (lam * cost_Q[k2, ny, nx] + nu * cost_dir[d, k2]) \
                    + mu * dt2 / ds
                if ng < g[ns_]:
                    g[ns_] = ng
                    parent[ns_] = s
                    heapq.heappush(heap, (ng + float(hmap[ny, nx]), ns_))
    return g, parent, best, n_pop


# --------------------------------------------------------------------------
# public wrapper
# --------------------------------------------------------------------------
@dataclass
class AStarResult:
    """Solution of one candidate corridor.

    ``path`` is in **full-image** ``(row, col, theta)`` triples (theta in
    radians, axial); ``path_px`` is the 8-connected rasterisation of it.
    ``energy`` is the discrete lifted-graph cost of the returned path, in
    full-resolution pixel units.
    """

    ok: bool
    path: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    path_px: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.int64))
    energy: float = float("inf")
    energy_per_len: float = float("inf")
    length: float = 0.0
    min_v: float = 0.0
    mean_v: float = 0.0
    min_q: float = 0.0
    mean_q: float = 0.0
    mean_wkappa: float = 0.0
    n_expanded: int = 0
    seconds: float = 0.0
    grid_shape: Tuple[int, int] = (0, 0)
    ds: int = 1


def _circ_bin_dist(k1, k2, K):
    d = abs(int(k1) - int(k2)) % K
    return min(d, K - d)


def _bin_of(theta: float, K: int) -> int:
    return int(round((float(theta) % math.pi) / (math.pi / K))) % K


def _tables(K: int, max_dtheta_bins: int):
    step = math.pi / K
    dtheta2 = np.full((K, K), -1.0, dtype=np.float64)
    for a in range(K):
        for b in range(K):
            d = _circ_bin_dist(a, b, K)
            if d <= max_dtheta_bins:
                dtheta2[a, b] = (d * step) ** 2
    tk = np.arange(K, dtype=np.float64) * step
    cost_dir = 1.0 - np.abs(np.cos(_DIR_ANG[:, None] - tk[None, :]))
    return dtheta2, np.ascontiguousarray(cost_dir)


def astar_lifted(
    v_evidence: np.ndarray,
    q_orient: np.ndarray,
    start_rc: Tuple[int, int],
    start_bins: Sequence[int],
    goal_mask: np.ndarray,
    goal_bins: Sequence[int],
    scale: float = 1.0,
    lam: float = LAMBDA_THETA,
    mu: float = MU_KAPPA,
    nu: float = NU_DIR,
    max_dtheta_bins: int = MAX_DTHETA_BINS,
    eps: float = EPS_PROB,
    blocked: Optional[np.ndarray] = None,
    use_numba: bool = True,
) -> AStarResult:
    """Solve one corridor.  All coordinates are **corridor-local**.

    ``v_evidence`` is ``(h, w)`` in ``[0, 1]``; ``q_orient`` is ``(K, h, w)``
    and must sum to 1 over the bin axis.  ``scale`` is the number of
    full-resolution pixels one corridor cell spans (the downsampling factor),
    so that energies of candidates solved at different resolutions remain
    comparable.
    """
    import time

    V = np.ascontiguousarray(np.asarray(v_evidence, dtype=np.float64))
    Q = np.ascontiguousarray(np.asarray(q_orient, dtype=np.float64))
    K, h, w = Q.shape
    cost_I = -np.log(np.clip(V, 0.0, 1.0) + eps)
    cost_Q = -np.log(np.clip(Q, 0.0, 1.0) + eps)

    dtheta2, cost_dir = _tables(K, max_dtheta_bins)
    dlen = _DIR_LEN * float(scale)

    gmask = np.ascontiguousarray(np.asarray(goal_mask, dtype=bool).astype(np.uint8))
    gbins = np.zeros(K, dtype=np.uint8)
    for b in goal_bins:
        gbins[int(b) % K] = 1
    blk = (np.zeros((h, w), np.uint8) if blocked is None
           else np.ascontiguousarray(np.asarray(blocked, bool).astype(np.uint8)))

    # ---- consistent heuristic --------------------------------------------
    from scipy import ndimage as ndi

    w_min = float(cost_I.min()) + float(lam) * float(cost_Q.min())
    w_min = max(w_min, 0.0)
    if gmask.any():
        dist = ndi.distance_transform_edt(gmask == 0) * float(scale)
    else:
        dist = np.zeros((h, w), dtype=np.float64)
    hmap = np.ascontiguousarray(w_min * dist.astype(np.float64))

    sy, sx = int(start_rc[0]), int(start_rc[1])
    starts = np.array(sorted({(sy * w + sx) * K + (int(b) % K) for b in start_bins}),
                      dtype=np.int64)

    n_states = h * w * K
    cap = int(min(max(4 * n_states, 1 << 16), 12_000_000))

    kern = _astar_kernel if (use_numba and HAVE_NUMBA) else _astar_kernel_py
    t0 = time.perf_counter()
    g, parent, best, n_pop = kern(
        cost_I, cost_Q, cost_dir, dtheta2, hmap, starts, gmask, gbins,
        _DIRS, dlen, float(lam), float(mu), float(nu), blk, cap,
    )
    dt = time.perf_counter() - t0

    res = AStarResult(ok=False, n_expanded=int(n_pop), seconds=float(dt),
                      grid_shape=(h, w), ds=int(round(scale)))
    if best < 0:
        return res

    chain = []
    s = int(best)
    while s >= 0:
        chain.append(s)
        s = int(parent[s])
    chain.reverse()
    step = math.pi / K
    ks = np.array([c % K for c in chain], dtype=np.int64)
    ys = np.array([(c // K) // w for c in chain], dtype=np.int64)
    xs = np.array([(c // K) % w for c in chain], dtype=np.int64)

    res.ok = True
    res.path = np.stack([ys, xs, ks * step], axis=1)
    res.energy = float(g[best])
    dd = np.hypot(np.diff(ys), np.diff(xs)) * float(scale)
    res.length = float(dd.sum())
    res.energy_per_len = float(res.energy / max(res.length, 1e-6))
    vv = V[ys, xs]
    qq = Q[ks, ys, xs]
    res.min_v = float(vv.min())
    res.mean_v = float(vv.mean())
    res.min_q = float(qq.min())
    res.mean_q = float(qq.mean())
    if len(chain) > 1:
        dk = np.array([_circ_bin_dist(ks[i], ks[i + 1], K) for i in range(len(ks) - 1)],
                      dtype=np.float64) * step
        res.mean_wkappa = float(np.mean(mu * dk ** 2 / np.maximum(dd, 1e-6)))
    return res


_WARMED = False


def warmup() -> None:
    """Trigger (or load) the numba JIT on a 6x6x16 toy problem.

    Without this the *first* candidate of a process is charged the one-off
    compilation / cache-load cost (~0.5 s), which would corrupt the per-
    candidate latency this stage has to report.  Idempotent and cheap.
    """
    global _WARMED
    if _WARMED:
        return
    _WARMED = True
    V = np.full((6, 6), 0.5, np.float32)
    Q = np.full((K_BINS, 6, 6), 1.0 / K_BINS, np.float32)
    g = np.zeros((6, 6), bool)
    g[5, 5] = True
    astar_lifted(V, Q, (0, 0), [0], g, list(range(K_BINS)))


def solve_candidate(
    row,
    v_evidence: np.ndarray,
    q_orient: np.ndarray,
    goal_pixels: Optional[np.ndarray] = None,
    lam: float = LAMBDA_THETA,
    mu: float = MU_KAPPA,
    nu: float = NU_DIR,
    max_dtheta_bins: int = MAX_DTHETA_BINS,
    start_tol_bins: int = 1,
    goal_tol_bins: int = 2,
    eps: float = EPS_PROB,
    use_numba: bool = True,
) -> AStarResult:
    """Solve one row of the :mod:`~src.rigr.candidates` table.

    ``row`` is a mapping with the keys of ``CANDIDATE_COLUMNS``.  The corridor
    box and its downsampling factor come from the table; ``goal_pixels`` is the
    port's skeleton pixel set for a T-type candidate (``kind == 'ep'``) and is
    ignored otherwise.

    Start states are the endpoint with its tangent bin +- ``start_tol_bins``;
    the goal is the other endpoint with any bin within ``goal_tol_bins`` of its
    tangent (endpoint-endpoint) or **any** port pixel with **any** bin
    (endpoint-port: a T junction arrives across the trunk, so its arrival
    orientation is unconstrained).
    """
    import cv2

    warmup()
    r0, c0 = int(row["box_r0"]), int(row["box_c0"])
    r1, c1 = int(row["box_r1"]), int(row["box_c1"])
    ds = max(1, int(row["box_ds"]))
    K = int(q_orient.shape[0])

    V = np.asarray(v_evidence, dtype=np.float32)[r0:r1, c0:c1]
    Q = np.asarray(q_orient, dtype=np.float32)[:, r0:r1, c0:c1]
    hh, ww = V.shape
    if ds > 1:
        nh, nw = max(1, hh // ds), max(1, ww // ds)
        V = cv2.resize(V, (nw, nh), interpolation=cv2.INTER_AREA)
        Q = np.stack([cv2.resize(Q[k], (nw, nh), interpolation=cv2.INTER_AREA)
                      for k in range(K)], axis=0)
        Q = Q / np.maximum(Q.sum(axis=0, keepdims=True), 1e-8)
    h, w = V.shape

    def to_local(r, c):
        return (int(np.clip((r - r0) // ds, 0, h - 1)),
                int(np.clip((c - c0) // ds, 0, w - 1)))

    sy, sx = to_local(int(row["r_i"]), int(row["c_i"]))
    k_i = _bin_of(float(row["theta_i"]), K)
    start_bins = [(k_i + d) % K for d in range(-start_tol_bins, start_tol_bins + 1)]

    gmask = np.zeros((h, w), dtype=bool)
    if int(row.get("is_port", 0)) == 1 and goal_pixels is not None and len(goal_pixels):
        for r, c in np.asarray(goal_pixels, dtype=np.int64):
            if r0 <= r < r1 and c0 <= c < c1:
                gy, gx = to_local(int(r), int(c))
                gmask[gy, gx] = True
        goal_bins = list(range(K))
    else:
        gy, gx = to_local(int(row["r_j"]), int(row["c_j"]))
        gmask[gy, gx] = True
        k_j = _bin_of(float(row["theta_j"]), K)
        goal_bins = [(k_j + d) % K for d in range(-goal_tol_bins, goal_tol_bins + 1)]
    if not gmask.any():
        gy, gx = to_local(int(row["r_j"]), int(row["c_j"]))
        gmask[gy, gx] = True
        goal_bins = list(range(K))

    res = astar_lifted(V, Q, (sy, sx), start_bins, gmask, goal_bins,
                       scale=float(ds), lam=lam, mu=mu, nu=nu,
                       max_dtheta_bins=max_dtheta_bins, eps=eps,
                       use_numba=use_numba)
    if not res.ok:
        return res

    # map back to full-image coordinates and rasterise
    from src.eval.trr_fcr import rasterize_path

    yy = np.rint((res.path[:, 0] + 0.5) * ds - 0.5).astype(np.int64) + r0
    xx = np.rint((res.path[:, 1] + 0.5) * ds - 0.5).astype(np.int64) + c0
    yy[0], xx[0] = int(row["r_i"]), int(row["c_i"])
    if int(row.get("is_port", 0)) != 1:
        yy[-1], xx[-1] = int(row["r_j"]), int(row["c_j"])
    res.path = np.stack([yy.astype(float), xx.astype(float), res.path[:, 2]], axis=1)
    res.path_px = rasterize_path(np.stack([yy, xx], axis=1))
    return res


# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - demo + timing
    import time

    rng = np.random.default_rng(0)
    K, h, w = 16, 128, 128

    # a curved vessel with a gap in the middle; V_I still sees it in the gap
    yy, xx = np.mgrid[0:h, 0:w]
    centre = 64 + 18.0 * np.sin(xx / 20.0)
    band = np.exp(-((yy - centre) ** 2) / (2 * 3.0 ** 2))
    V = np.clip(0.05 + 0.9 * band, 0, 1).astype(np.float32)
    tangent = np.arctan2(np.gradient(centre, axis=1), 1.0)     # d(row)/d(col)
    tk = np.arange(K) * (math.pi / K)
    logits = 4.0 * np.cos(2.0 * (tk[:, None, None] - tangent[None]))
    Q = np.exp(logits)
    Q = (Q / Q.sum(axis=0, keepdims=True)).astype(np.float32)
    Q = Q * band[None] + (1.0 / K) * (1 - band[None])
    Q = (Q / Q.sum(axis=0, keepdims=True)).astype(np.float32)

    sy = int(round(centre[0, 10]))
    gy = int(round(centre[0, 117]))
    goal = np.zeros((h, w), bool)
    goal[gy, 117] = True

    print("numba available:", HAVE_NUMBA)
    r = astar_lifted(V, Q, (sy, 10), [_bin_of(tangent[sy, 10], K)], goal, list(range(K)))
    print("warm-up jit     : ok=%s  %.1f ms" % (r.ok, 1e3 * r.seconds))

    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        r = astar_lifted(V, Q, (sy, 10), [_bin_of(tangent[sy, 10], K)],
                         goal, list(range(K)))
        ts.append(time.perf_counter() - t0)
    print("path found      : %s, %d states, E*=%.2f, E*/len=%.4f"
          % (r.ok, len(r.path), r.energy, r.energy_per_len))
    print("min/mean V_I    : %.3f / %.3f     min/mean q: %.3f / %.3f"
          % (r.min_v, r.mean_v, r.min_q, r.mean_q))
    print("expanded states : %d of %d" % (r.n_expanded, h * w * K))
    print("timing 128x128x16: mean %.1f ms, median %.1f ms, max %.1f ms"
          % (1e3 * np.mean(ts), 1e3 * np.median(ts), 1e3 * np.max(ts)))

    dev = np.abs(r.path[:, 0] - centre[0, np.clip(r.path[:, 1].astype(int), 0, w - 1)])
    print("max deviation from the true centreline: %.2f px" % float(dev.max()))
