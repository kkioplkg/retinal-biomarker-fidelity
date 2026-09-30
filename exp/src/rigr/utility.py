"""Asymmetric expected utility and global selection (proposal v3 section 3.2.6).

.. math::

    U(e) = p_e R_{miss}(e) - \\lambda (1 - p_e) R_{false}(e) - \\eta C_{geom}(e)

with all three terms on the same dimensionless scale, ``R_miss, R_false,
C_geom >= 0`` by construction.  The semantics are then unambiguous: the
positive term is the expected measurement benefit of repairing the candidate,
the two negative terms are the expected measurement cost of a wrong connection
and the geometric intervention cost, ``U(e) <= p_e R_miss(e)`` always, and *a
wrong connection can never raise the utility*.

Geometric cost
--------------
``C_geom_raw = 0.5 * len(gamma) / rbar + 0.5 * mean(w_kappa)`` -- a long thin
detour and a high-curvature detour are both penalised.  It is put on the same
scale as ``R`` by the *same* pre-registered robust scale used for the harm,
``sigma = 1.4826 * MAD``, estimated over the candidate pool (or supplied
per-dataset by the caller so a working point means the same thing in every
image).

Three modes (the three curves of Fig.3)
---------------------------------------
``prob``     ``U = p_e - tau``                       (probability only)
``uniform``  ``R_miss = R_false = 1``                (uniform event cost)
``risk``     ``R`` from the BTR heads of ``src.c1.btr``
             (``predict_R_miss`` on the severed state, ``predict_R_false`` on
             the connected state).  That module is written concurrently, so it
             is imported **lazily** and a missing / incompatible module falls
             back to ``R = 1`` with a warning -- the run then reports
             ``risk_backend = 'fallback-uniform'`` so no result can silently
             claim to be risk-guided when it is not.

Global selection (three ordered stages, section 3.2.6)
-----------------------------------------------------
1. **Hard vetoes**, applied as edge removals: ``U(e) <= 0``, a failed A*, or a
   path that crosses an unrelated vessel roughly perpendicularly.
2. **Corridor-conflict pruning**, also an edge removal and also *before* the
   matching: among any pair of surviving edges whose corridors overlap by more
   than 50% (Jaccard index of the corridor boxes) the lower-utility edge is
   dropped.  This is the additivity mitigation of section 3.2.6 -- two edges
   editing the same neighbourhood are not independent -- and it is deliberately
   *not* folded into the matching, because mutual exclusion of that kind is not
   a constraint a matching can express.
3. **Maximum-weight matching** (``networkx.max_weight_matching``, exact
   blossom) on the pruned graph, whose nodes are the endpoints **and the
   attachment-port pseudo-nodes**, each of capacity 1.  This stage is a
   textbook maximum-weight matching, solved to optimality, with nothing grafted
   onto it.

Run ``cd exp && python -m src.rigr.utility`` for a synthetic demo.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.topo import skeleton as sk

__all__ = [
    "MODES",
    "geom_cost",
    "mad_scale",
    "load_risk_backend",
    "risk_values",
    "_phi_frames",
    "compute_utility",
    "corridor_overlap",
    "prune_conflicting_edges",
    "select_edges",
    "apply_edges",
    "repair_mask",
    "self_test",
]

MODES = ("prob", "uniform", "risk")
_EPS = 1e-9
_WARNED_BTR = False


def _warn_btr(msg: str) -> None:
    """Warn once per process: the fallback is per-candidate but the fact is not."""
    global _WARNED_BTR
    if not _WARNED_BTR:
        _WARNED_BTR = True
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


# --------------------------------------------------------------------------
# scales and geometric cost
# --------------------------------------------------------------------------
def mad_scale(x: np.ndarray, floor: float = 1e-6) -> float:
    """Pre-registered robust scale ``sigma = 1.4826 * MAD`` (section 3.1.2)."""
    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 1.0
    med = float(np.median(v))
    s = 1.4826 * float(np.median(np.abs(v - med)))
    if not np.isfinite(s) or s <= floor:
        s = float(np.std(v)) if v.size > 1 else 1.0
    return float(max(s, floor))


def geom_cost(table, results: Sequence[Any], scale: Optional[float] = None
              ) -> Tuple[np.ndarray, float]:
    """``C_geom`` and the scale used, from the candidate table + A* results.

    Failed candidates get a large finite cost so they never win a matching.
    """
    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    raw = np.zeros(len(rows), dtype=float)
    for i, row in enumerate(rows):
        res = results[i] if i < len(results) else None
        if res is None or not getattr(res, "ok", False):
            raw[i] = np.nan
            continue
        rbar = max(float(row["rad_bar"]), _EPS)
        raw[i] = 0.5 * float(res.length) / rbar + 0.5 * float(res.mean_wkappa)
    finite = raw[np.isfinite(raw)]
    s = float(scale) if scale is not None else mad_scale(finite if finite.size else
                                                         np.array([1.0]))
    big = (np.nanmax(finite) if finite.size else 1.0) * 10.0 + 10.0
    raw = np.where(np.isfinite(raw), raw, big)
    return raw / s, s


# --------------------------------------------------------------------------
# risk backend (BTR)
# --------------------------------------------------------------------------
def load_risk_backend(module: Optional[Any] = None):
    """Lazily import ``src.c1.btr``; return ``None`` (with a warning) if absent.

    Expected contract, matching plan stage S3 item 4::

        btr.predict_R_miss(features)  -> (N,) non-negative array
        btr.predict_R_false(features) -> (N,) non-negative array

    where ``features`` is a DataFrame of contract-F candidate features.
    """
    if module is not None:
        return module
    try:
        from src.c1 import btr  # type: ignore

        if not (hasattr(btr, "predict_R_miss") and hasattr(btr, "predict_R_false")):
            _warn_btr("src.c1.btr lacks predict_R_miss / predict_R_false; "
                      "falling back to R = 1 (uniform event cost)")
            return None
        return btr
    except Exception as exc:  # noqa: BLE001 - the module is written concurrently
        _warn_btr("src.c1.btr is unavailable (%s: %s); falling back to R = 1 "
                  "(uniform event cost).  This run is NOT risk-guided."
                  % (type(exc).__name__, exc))
        return None


def _phi_frames(table, results, image, mask, fov=None, disc=None):
    """Contract-F feature frames for the two BTR heads (section 3.1.5).

    ``R_miss`` observes the **severed** state ``M^-`` (the current predicted
    mask, whose two new endpoints are the candidate anchors); ``R_false``
    observes the **connected** state ``M^+ = M^- U T_e`` -- the same feature
    schema evaluated on two different states, exactly as the proposal
    prescribes.  Feature extraction is delegated to ``src.c1.perturb.phi`` so
    the deployed risk heads see the identical schema they were trained on.
    """
    import pandas as pd

    from src.c1.perturb import PHI_COLUMNS, _EventPoints, phi
    from src.eval.trr_fcr import rasterize_tube

    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    m = sk.as_bool(mask)
    miss, false = [], []
    for i, row in enumerate(rows):
        a = (int(row["r_i"]), int(row["c_i"]))
        b = (int(row["r_j"]), int(row["c_j"]))
        miss.append(phi(_EventPoints("sever", a, b), image, m, disc=disc, fov=fov))
        res = results[i] if i < len(results) else None
        if res is not None and getattr(res, "ok", False) and len(res.path_px):
            m_plus = m | rasterize_tube(res.path_px, float(row["rad_bar"]), m.shape)
        else:
            m_plus = m
        false.append(phi(_EventPoints("bridge", a, b), image, m_plus,
                         disc=disc, fov=fov))
    return (pd.DataFrame(miss, columns=PHI_COLUMNS),
            pd.DataFrame(false, columns=PHI_COLUMNS))


def risk_values(
    table,
    results: Sequence[Any],
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    fov: Optional[np.ndarray] = None,
    disc: Optional[Any] = None,
    backend: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """``(R_miss, R_false, backend_name)``; ``R = 1`` when BTR is unavailable.

    ``image`` and ``mask`` are required: without them the Contract-F features
    the BTR heads were trained on cannot be reconstructed, and silently feeding
    the heads a different schema would produce numbers that look like risks but
    are not.  In that case the call degrades to ``R = 1`` and says so.
    """
    n = len(table)
    btr = load_risk_backend(backend)
    if btr is None:
        return np.ones(n), np.ones(n), "fallback-uniform"
    if image is None or mask is None:
        _warn_btr("risk mode needs `image` and `mask` to rebuild the Contract-F "
                  "features of src.c1.perturb.phi; falling back to R = 1")
        return np.ones(n), np.ones(n), "fallback-uniform"
    try:
        f_miss, f_false = _phi_frames(table, results, image, mask, fov, disc)
        rm = np.asarray(btr.predict_R_miss(f_miss), dtype=float).ravel()
        rf = np.asarray(btr.predict_R_false(f_false), dtype=float).ravel()
        if rm.shape[0] != n or rf.shape[0] != n:
            raise ValueError("BTR returned %s / %s values for %d candidates"
                             % (rm.shape, rf.shape, n))
        rm = np.nan_to_num(np.maximum(rm, 0.0), nan=1.0)
        rf = np.nan_to_num(np.maximum(rf, 0.0), nan=1.0)
        return rm, rf, getattr(btr, "__name__", "src.c1.btr")
    except Exception as exc:  # noqa: BLE001
        _warn_btr("BTR risk prediction failed (%s: %s); falling back to R = 1. "
                  "This run is NOT risk-guided -- every row is stamped "
                  "risk_backend=fallback-uniform." % (type(exc).__name__, exc))
        return np.ones(n), np.ones(n), "fallback-uniform"


# --------------------------------------------------------------------------
# utility
# --------------------------------------------------------------------------
def compute_utility(
    table,
    results: Sequence[Any],
    p: np.ndarray,
    mode: str = "uniform",
    lam: float = 1.0,
    eta: float = 0.1,
    tau: float = 0.5,
    X: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    fov: Optional[np.ndarray] = None,
    disc: Optional[Any] = None,
    risk_backend: Optional[Any] = None,
    c_scale: Optional[float] = None,
    geom_scale: Optional[float] = None,
):
    """Per-candidate utility table.

    Returns a DataFrame with ``p, R_miss, R_false, C_geom, U`` plus
    ``mode``/``lambda``/``eta`` echoed, and the attributes ``risk_backend`` and
    ``c_scale`` on ``df.attrs``.
    """
    import pandas as pd

    if mode not in MODES:
        raise ValueError("mode must be one of %s" % (MODES,))
    n = len(table)
    p = np.asarray(p, dtype=float).ravel()
    if p.shape[0] != n:
        raise ValueError("p has %d entries for %d candidates" % (p.shape[0], n))

    c_geom, used_scale = geom_cost(table, results, c_scale)
    # ``geom_scale`` (m_R) puts C_geom on the SAME dimensionless scale as R,
    # which proposal section 3.1.2 requires and the implementation violated.
    # ``geom_cost`` normalises C_geom by its own MAD, so it is O(1) (median
    # ~2.6), while the sigma-standardised single-event harms R_miss / R_false
    # are O(1e-3).  At the pre-registered eta = 0.1 that made
    # U ~ -eta * C_geom -- corr(U, -C_geom) = 0.9999 -- so risk mode accepted
    # almost nothing (LODO / STARE TRR = 0) and U anti-correlated with the
    # realised harm in Exp2.  m_R is the median R_miss over the dataset's
    # TRAINING candidates, fixed at fit time in geom_scale.json; test images
    # never enter it.  DECISIONS.md 2026-09-06 12:30.
    geom_scale_used = 1.0
    if mode == "prob":
        rm = np.ones(n)
        rf = np.zeros(n)
        backend = "none"
        u = p - float(tau)
        eta_eff = 0.0
    elif mode == "uniform":
        # R === 1 here, so C_geom is already on the scale of R and must NOT be
        # rescaled: doing so would move the uniform arm's operating point for
        # no reason and break its comparison with the risk arm.
        rm = np.ones(n)
        rf = np.ones(n)
        backend = "uniform"
        eta_eff = float(eta)
        u = p * rm - float(lam) * (1.0 - p) * rf - eta_eff * c_geom
    else:
        rm, rf, backend = risk_values(table, results, image=image,
                                      mask=mask, fov=fov, disc=disc,
                                      backend=risk_backend)
        if geom_scale is None:
            raise ValueError(
                "risk mode needs geom_scale (m_R): C_geom is O(1) while the "
                "BTR harms are O(1e-3), so an unscaled C_geom makes "
                "U ~ -eta*C_geom and the run is not risk-guided at all. "
                "Fit it with `python -m src.rigr.geom_scale --data "
                "runs/rigr_data/<ds> --out runs/rigr_models/<ds>/seed<k>` and "
                "pass it through (run_rigr reads geom_scale.json next to the "
                "scorer).  DECISIONS.md 2026-09-06 12:30.")
        geom_scale_used = float(geom_scale)
        eta_eff = float(eta)
        u = (p * rm - float(lam) * (1.0 - p) * rf
             - eta_eff * c_geom * geom_scale_used)

    df = pd.DataFrame(dict(
        cand_id=np.asarray(table["cand_id"], dtype=int) if n else np.zeros(0, int),
        p=p, R_miss=rm, R_false=rf, C_geom=c_geom, U=u,
        mode=mode, lam=float(lam), eta=float(eta_eff), tau=float(tau),
        geom_scale=float(geom_scale_used),
    ))
    df.attrs["risk_backend"] = backend
    df.attrs["c_scale"] = float(used_scale)
    return df


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------
def corridor_overlap(a, b, metric: str = "iou") -> float:
    """Overlap of the two candidate corridors ``Omega_i``, ``Omega_j``.

    ``metric='iou'`` (the default, and the definition the paper quotes) is the
    **Jaccard index of the two corridor boxes**,

        ``|Omega_i and Omega_j| / |Omega_i or Omega_j|``,

    a symmetric quantity in ``[0, 1]``.  Corridor boxes are axis-aligned by
    construction (:func:`src.rigr.candidates.corridor_box`), so both the
    intersection and the union are exact rather than approximations, and the
    measure costs four comparisons -- no tube rasterisation is involved.

    ``metric='min'`` is the containment variant ``|A and B| / min(|A|, |B|)``,
    kept as an option because it is the natural measure when one corridor is
    nested inside a much larger one.  It is never smaller than the IoU, so at
    the same threshold it prunes at least as many edges.
    """
    r0 = max(int(a["box_r0"]), int(b["box_r0"]))
    r1 = min(int(a["box_r1"]), int(b["box_r1"]))
    c0 = max(int(a["box_c0"]), int(b["box_c0"]))
    c1 = min(int(a["box_c1"]), int(b["box_c1"]))
    if r1 <= r0 or c1 <= c0:
        return 0.0
    inter = float((r1 - r0) * (c1 - c0))
    area_a = float((int(a["box_r1"]) - int(a["box_r0"])) *
                   (int(a["box_c1"]) - int(a["box_c0"])))
    area_b = float((int(b["box_r1"]) - int(b["box_r0"])) *
                   (int(b["box_c1"]) - int(b["box_c0"])))
    if metric == "iou":
        return inter / max(area_a + area_b - inter, 1.0)
    if metric == "min":
        return inter / max(min(area_a, area_b), 1.0)
    raise ValueError("metric must be 'iou' or 'min', got %r" % (metric,))


def prune_conflicting_edges(
    rows: Sequence[dict],
    idx: Sequence[int],
    u: np.ndarray,
    overlap_thresh: float = 0.5,
    metric: str = "iou",
) -> Tuple[List[int], List[int]]:
    """Deterministic **pre-matching** removal of corridor-conflicting edges.

    Rationale.  ``sum_e U(e) x_e`` is a first-order sparse-edit approximation
    (section 3.2.6); two edges whose corridors coincide are not independent
    edits, so at most one of them may enter the objective.  Mutual exclusion of
    that kind is *not* something maximum-weight matching can express -- a
    matching constrains node capacities and nothing else -- so it is applied
    here as an explicit **pruning of the edge set before the matching runs**,
    never as a post-hoc repair of the matching's answer.  The matching stage
    therefore remains standard and is solved exactly on the pruned edge set.

    Rule.  Edges are visited in order of decreasing ``U``, ties broken by
    ascending ``cand_id``, so the outcome depends on neither row order nor
    dict iteration order nor the platform.  An edge is kept when its corridor
    overlap with every already-kept edge is ``<= overlap_thresh``; otherwise it
    is pruned.  Equivalently: of any conflicting pair the lower-``U`` edge is
    the one dropped, and an edge whose only conflict was with an
    already-pruned edge survives.

    Returns ``(kept, pruned)``, both sorted lists of row indices.
    """
    order = sorted(idx, key=lambda i: (-float(u[i]), int(rows[i]["cand_id"])))
    kept: List[int] = []
    pruned: List[int] = []
    for i in order:
        clash = any(
            corridor_overlap(rows[i], rows[j], metric) > float(overlap_thresh)
            for j in kept
        )
        (pruned if clash else kept).append(i)
    return sorted(kept), sorted(pruned)


def select_edges(
    table,
    results: Sequence[Any],
    utility,
    labels: Optional[np.ndarray] = None,
    theta_mask: Optional[np.ndarray] = None,
    radius: Optional[np.ndarray] = None,
    overlap_thresh: float = 0.5,
    overlap_metric: str = "iou",
    veto_crossing: bool = True,
    u_floor: float = 0.0,
) -> Dict[str, Any]:
    """Global selection of section 3.2.6, as three strictly ordered stages.

    1. **Hard vetoes -- edge removals.**  An edge is deleted outright when its
       A* failed to find a path, when ``U(e) <= u_floor`` (the pre-registered
       constraint is ``x_e = 0 if U(e) <= 0``, i.e. the default
       ``u_floor = 0``), or when its path crosses an unrelated vessel
       roughly perpendicularly.

    ``u_floor`` exists only to give the risk and uniform arms an operating
    curve.  Their utility has no threshold knob -- ``tau`` is inert outside
    ``prob`` mode and acceptance is the hard ``U > 0`` of 3.2.6 -- so a
    lambda-only sweep produced just four operating points, too few to
    interpolate a matched-FCR comparison against ``prob`` (DECISIONS.md
    2026-09-16 03:00(f)).  Raising the floor to a quantile of the positive-U
    candidates walks the arm along its own precision/recall curve without
    touching the utility definition.  **Leave it at 0 for anything reported as
    the pre-registered operating point.**
    2. **Corridor-conflict pruning -- edge removals.**  Deterministic and
       applied *before* the matching; see :func:`prune_conflicting_edges`.
    3. **Maximum-weight matching** (``networkx.max_weight_matching``, exact
       blossom) on what remains.  Every node -- endpoint *and* attachment-port
       pseudo-node -- has capacity 1, which is exactly the constraint a
       matching enforces, so this stage is a textbook maximum-weight matching
       solved to optimality on the pruned edge set, with no extra step grafted
       onto it.

    Returns ``dict`` with ``selected`` (row indices), ``vetoed_crossing``,
    ``pruned_conflict``, ``n_positive`` (edges surviving stage 1),
    ``n_vetoed``, ``n_pruned_conflict`` and ``overlap_metric``.
    ``vetoed_overlap`` is kept as a deprecated alias of ``pruned_conflict``.
    """
    import networkx as nx

    from src.rigr.candidates import crosses_unrelated_vessel

    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    u = np.asarray(utility["U"] if hasattr(utility, "__getitem__")
                   and not isinstance(utility, np.ndarray) else utility, dtype=float)

    # ---- stage 1: hard vetoes -------------------------------------------
    survivors, vetoed_cross = [], []
    for i, row in enumerate(rows):
        res = results[i] if i < len(results) else None
        if res is None or not getattr(res, "ok", False):
            continue
        if not np.isfinite(u[i]) or u[i] <= u_floor:
            continue
        if veto_crossing and labels is not None and theta_mask is not None \
                and radius is not None:
            if crosses_unrelated_vessel(res.path_px, labels,
                                        [int(row["comp_i"]), int(row["comp_j"])],
                                        theta_mask, radius):
                vetoed_cross.append(i)
                continue
        survivors.append(i)

    n_positive = len(survivors)

    # ---- stage 2: deterministic corridor-conflict pruning ----------------
    kept, pruned = prune_conflicting_edges(rows, survivors, u, overlap_thresh,
                                           overlap_metric)

    def _out(selected):
        return dict(selected=sorted(selected),
                    vetoed_crossing=sorted(vetoed_cross),
                    pruned_conflict=sorted(pruned),
                    vetoed_overlap=sorted(pruned),      # deprecated alias
                    n_positive=int(n_positive),
                    n_vetoed=int(len(vetoed_cross)),
                    n_pruned_conflict=int(len(pruned)),
                    overlap_metric=str(overlap_metric))

    if not kept:
        return _out([])

    # ---- stage 3: exact maximum-weight matching --------------------------
    G = nx.Graph()
    for i in kept:
        row = rows[i]
        a, b = int(row["node_i"]), int(row["node_j"])
        w = float(u[i])
        if G.has_edge(a, b) and G[a][b]["weight"] >= w:
            continue
        G.add_edge(a, b, weight=w, idx=i)
    matched = nx.max_weight_matching(G, maxcardinality=False)
    return _out(G[a][b]["idx"] for a, b in matched)


# --------------------------------------------------------------------------
# application
# --------------------------------------------------------------------------
def apply_edges(
    mask: np.ndarray,
    table,
    results: Sequence[Any],
    selected: Sequence[int],
    fov: Optional[np.ndarray] = None,
    radius_scale: float = 1.0,
):
    """Rasterise a tube of radius ``rbar`` along each selected path onto ``M_hat``.

    Returns ``(repaired_mask, accepted_edges)`` where ``accepted_edges`` is the
    list of :class:`src.eval.trr_fcr.CandidateEdge` that TRR / FCR consumes.
    """
    from src.eval.trr_fcr import CandidateEdge, rasterize_tube

    m = sk.as_bool(mask).copy()
    f = sk.fov_or_true(fov, m.shape)
    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    edges: List[CandidateEdge] = []
    for i in selected:
        res = results[i]
        row = rows[i]
        r = float(row["rad_bar"]) * float(radius_scale)
        m |= rasterize_tube(res.path_px, r, m.shape) & f
        edges.append(CandidateEdge(path=res.path_px,
                                   p0=(int(row["r_i"]), int(row["c_i"])),
                                   p1=(int(row["r_j"]), int(row["c_j"])),
                                   radius=r, edge_id=int(row["cand_id"]),
                                   score=float(row.get("p", np.nan))))
    return m, edges


def repair_mask(
    mask: np.ndarray,
    table,
    results: Sequence[Any],
    p: np.ndarray,
    mode: str = "uniform",
    lam: float = 1.0,
    eta: float = 0.1,
    tau: float = 0.5,
    fov: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    theta_mask: Optional[np.ndarray] = None,
    radius: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    image: Optional[np.ndarray] = None,
    disc: Optional[Any] = None,
    risk_backend: Optional[Any] = None,
    c_scale: Optional[float] = None,
    geom_scale: Optional[float] = None,
    overlap_thresh: float = 0.5,
    overlap_metric: str = "iou",
    u_floor: float = 0.0,
) -> Dict[str, Any]:
    """Utility -> matching -> vetoes -> tube rasterisation, in one call.

    ``u_floor`` (default 0 = the pre-registered rule) raises the acceptance
    threshold on U; see :func:`select_edges`."""
    util = compute_utility(table, results, p, mode=mode, lam=lam, eta=eta,
                           tau=tau, X=X, image=image, mask=mask, fov=fov,
                           disc=disc, risk_backend=risk_backend,
                           c_scale=c_scale, geom_scale=geom_scale)
    selinfo = select_edges(table, results, util, labels=labels,
                           theta_mask=theta_mask, radius=radius,
                           overlap_thresh=overlap_thresh,
                           overlap_metric=overlap_metric, u_floor=u_floor)
    repaired, edges = apply_edges(mask, table, results, selinfo["selected"], fov)
    return dict(mask=repaired, edges=edges, utility=util, **selinfo)


# --------------------------------------------------------------------------
# unit test
# --------------------------------------------------------------------------
def _mk_row(cid, ni, nj, box):
    """A minimal candidate row carrying only the fields selection reads."""
    r0, c0, r1, c1 = box
    return dict(cand_id=cid, kind="ee", node_i=ni, node_j=nj,
                r_i=r0 + 2, c_i=c0 + 2, r_j=r1 - 2, c_j=c1 - 2,
                rad_i=2.0, rad_j=2.0, rad_bar=2.0, theta_i=0.0, theta_j=0.0,
                tan_i_r=0.0, tan_i_c=1.0, tan_j_r=0.0, tan_j_c=-1.0,
                d=10.0, d_over_rbar=5.0, log_r_ratio=0.0, tangent_mismatch=0.0,
                comp_i=ni + 1, comp_j=nj + 1, is_port=0,
                box_r0=r0, box_c0=c0, box_r1=r1, box_c1=c1, box_ds=1)


def self_test(verbose: bool = True) -> None:
    """Unit test of the pre-matching conflict pruning (3-edge chain conflict).

    Three **node-disjoint** candidate edges A, B, C with ``U(A) > U(B) > U(C)``
    and corridors arranged so that

        IoU(A, B) > 0.5,   IoU(B, C) > 0.5,   IoU(A, C) < 0.5.

    Because the three edges share no node, a bare maximum-weight matching would
    take all three.  The deterministic pruning must drop exactly **B** (it
    conflicts with the higher-utility A) and must **keep C**, whose only
    conflict was with the now-removed B -- the property that distinguishes the
    greedy-by-utility rule from naive pairwise deletion, which would drop both
    B and C.  Expected: ``selected = [A, C]``, ``n_pruned_conflict = 1``.
    """
    import numpy as np
    import pandas as pd

    from src.eval.trr_fcr import rasterize_path
    from src.rigr.astar import AStarResult

    def _res(box):
        r0, c0, r1, c1 = box
        px = rasterize_path([(r0 + 2, c0 + 2), (r1 - 2, c1 - 2)])
        return AStarResult(ok=True, path=np.c_[px, np.zeros(len(px))],
                           path_px=px, energy=1.0, energy_per_len=0.1,
                           length=float(len(px)), min_v=0.6, mean_v=0.8,
                           min_q=0.2, mean_q=0.3, mean_wkappa=0.01)

    # 100 x 100 boxes slid by 20 px: neighbours overlap 80/120 of their union
    # (IoU 0.667), A and C overlap 60/140 (IoU 0.429).
    boxes = [(0, 0, 100, 100), (0, 20, 100, 120), (0, 40, 100, 140)]
    rows = [_mk_row(0, 0, 1, boxes[0]), _mk_row(1, 2, 3, boxes[1]),
            _mk_row(2, 4, 5, boxes[2])]
    results = [_res(b) for b in boxes]
    u = np.array([0.90, 0.50, 0.20])

    iou_ab = corridor_overlap(rows[0], rows[1])
    iou_bc = corridor_overlap(rows[1], rows[2])
    iou_ac = corridor_overlap(rows[0], rows[2])
    assert iou_ab > 0.5 and iou_bc > 0.5 and iou_ac < 0.5, (iou_ab, iou_bc, iou_ac)
    assert corridor_overlap(rows[0], rows[1], "min") >= iou_ab
    assert corridor_overlap(rows[0], _mk_row(9, 8, 9, (500, 500, 560, 560))) == 0.0

    kept, pruned = prune_conflicting_edges(rows, [0, 1, 2], u)
    assert kept == [0, 2], kept
    assert pruned == [1], pruned

    table = pd.DataFrame(rows)
    out = select_edges(table, results, pd.DataFrame(dict(U=u)))
    assert out["selected"] == [0, 2], out["selected"]
    assert out["n_pruned_conflict"] == 1, out
    assert out["pruned_conflict"] == [1], out
    assert out["n_vetoed"] == 0, out
    assert out["n_positive"] == 3, out
    assert out["vetoed_overlap"] == out["pruned_conflict"], "alias broke"
    assert out["overlap_metric"] == "iou", out
    assert 1 not in out["selected"], "a pruned edge reappeared via the matching"

    # order independence: permuting the rows must not change the outcome
    perm = [2, 0, 1]
    rows_p = [rows[i] for i in perm]
    out_p = select_edges(pd.DataFrame(rows_p), [results[i] for i in perm],
                         pd.DataFrame(dict(U=u[perm])))
    assert sorted(rows_p[i]["cand_id"] for i in out_p["selected"]) == [0, 2], out_p
    assert out_p["n_pruned_conflict"] == 1, out_p

    # a tie in U is broken by cand_id, so the outcome is still deterministic
    u_tie = np.array([0.5, 0.5, 0.0])
    k1, p1 = prune_conflicting_edges(rows, [0, 1, 2], u_tie)
    k2, p2 = prune_conflicting_edges(rows, [2, 1, 0], u_tie)
    assert (k1, p1) == ([0, 2], [1]), (k1, p1)
    assert (k2, p2) == (k1, p1), (k2, p2)

    # U <= 0 is a stage-1 veto, counted separately from a conflict prune
    out0 = select_edges(table, results, pd.DataFrame(dict(U=[0.9, -0.1, 0.2])))
    assert out0["n_positive"] == 2, out0
    assert out0["n_pruned_conflict"] == 0, out0
    assert out0["selected"] == [0, 2], out0

    # the 'min' metric prunes at least as much as 'iou' at the same threshold
    _, pruned_min = prune_conflicting_edges(rows, [0, 1, 2], u, metric="min")
    assert len(pruned_min) >= len(pruned), (pruned_min, pruned)

    if verbose:
        print("conflict-pruning unit test (3-edge chain, node-disjoint)")
        print("  IoU(A,B)=%.3f  IoU(B,C)=%.3f  IoU(A,C)=%.3f   threshold 0.5"
              % (iou_ab, iou_bc, iou_ac))
        print("  U = %s" % u.tolist())
        print("  stage 1 survivors=%d  stage 2 pruned=%s  stage 3 selected=%s"
              % (out["n_positive"], out["pruned_conflict"], out["selected"]))
        print("  n_pruned_conflict=%d  n_vetoed=%d"
              % (out["n_pruned_conflict"], out["n_vetoed"]))
        print("  naive pairwise deletion would drop B and C; greedy-by-U keeps C")
        print("  order independence / U-tie determinism / U<=0 veto / "
              "'min' >= 'iou': OK")
        print("PASS")


# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - synthetic demo
    import pandas as pd

    from src.rigr.astar import AStarResult
    from src.eval.trr_fcr import rasterize_path

    H = W = 120
    base = np.zeros((H, W), bool)
    base[58:62, 10:50] = True
    base[58:62, 70:110] = True
    base[20:50, 58:62] = True

    rows = [
        dict(cand_id=0, kind="ee", node_i=0, node_j=1, r_i=59, c_i=49, r_j=59,
             c_j=70, rad_i=2.0, rad_j=2.0, rad_bar=2.0, theta_i=0.0, theta_j=0.0,
             tan_i_r=0.0, tan_i_c=1.0, tan_j_r=0.0, tan_j_c=-1.0, d=21.0,
             d_over_rbar=10.5, log_r_ratio=0.0, tangent_mismatch=0.0, comp_i=1,
             comp_j=2, is_port=0, box_r0=20, box_c0=20, box_r1=100, box_c1=100,
             box_ds=1),
        dict(cand_id=1, kind="ee", node_i=0, node_j=2, r_i=59, c_i=49, r_j=50,
             c_j=59, rad_i=2.0, rad_j=2.0, rad_bar=2.0, theta_i=0.0,
             theta_j=1.5708, tan_i_r=0.0, tan_i_c=1.0, tan_j_r=-1.0, tan_j_c=0.0,
             d=13.5, d_over_rbar=6.7, log_r_ratio=0.0, tangent_mismatch=0.9,
             comp_i=1, comp_j=3, is_port=0, box_r0=22, box_c0=22, box_r1=98,
             box_c1=98, box_ds=1),
    ]
    table = pd.DataFrame(rows)

    def mk(a, b, e, wk):
        px = rasterize_path([a, b])
        return AStarResult(ok=True, path=np.c_[px, np.zeros(len(px))], path_px=px,
                           energy=e, energy_per_len=e / len(px), length=float(len(px)),
                           min_v=0.6, mean_v=0.8, min_q=0.2, mean_q=0.3,
                           mean_wkappa=wk)

    results = [mk((59, 49), (59, 70), 20.0, 0.01), mk((59, 49), (50, 59), 40.0, 0.4)]
    p = np.array([0.9, 0.35])

    for mode in MODES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = repair_mask(base, table, results, p, mode=mode, lam=1.0, eta=0.1)
        print("%-8s U = %s  selected=%s  backend=%s  added=%d px"
              % (mode, np.round(out["utility"]["U"].to_numpy(), 3).tolist(),
                 out["selected"], out["utility"].attrs["risk_backend"],
                 int(out["mask"].sum() - base.sum())))
    b0_before = sk.betti_numbers(base)[0]
    b0_after = sk.betti_numbers(out["mask"])[0]
    print("beta0 %d -> %d" % (b0_before, b0_after))

    print()
    self_test()
