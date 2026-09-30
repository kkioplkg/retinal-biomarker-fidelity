"""Simple geometric repair -- the no-learning baseline of plan S6 / proposal Tab.2.

A 2D simplification of the classical morphological-plus-geodesic endpoint
reconnection family (OGMC [30] in the proposal's numbering).  It uses **no**
learning and **no** probability map: the geometry of ``M_hat`` alone decides.

Rule (all thresholds pre-registered in the task spec)
----------------------------------------------------
1. Skeletonise ``M_hat`` (Zhang-Suen, spurs < 3 px pruned) and take the
   degree-1 pixels as *dangling endpoints*, dropping any within ``border_px``
   of the FOV boundary (a vessel leaving the field of view is not a break).
2. For every endpoint pair ``(i, j)`` form a candidate if

   * ``d(i, j) <= 6 * r_bar``            with ``r_bar = (r_i + r_j) / 2``;
   * **tangent compatibility on both sides**: ``t_i . u >= 0.5`` and
     ``t_j . (-u) >= 0.5``, where ``u`` is the unit vector from ``i`` to ``j``
     and both tangents point outward, into the gap;
   * **radius compatibility**: ``max(r_i, r_j) / min(r_i, r_j) <= 2``;
   * (default) the two endpoints lie in **different** connected components, so
     the edge can actually restore connectivity -- criterion (iii) of the true
     repair definition.  Set ``require_diff_component=False`` to lift this.
3. Accept greedily in order of increasing distance, each endpoint used at most
   once.
4. Paint each accepted edge as a tube of radius ``r_bar`` around a straight
   segment, or a cubic Bezier honouring both tangents when either tangent
   deviates from the chord by more than ``bezier_deg`` degrees.

Every accepted edge is returned as a :class:`CandidateEdge` with its rasterised
path, so ``src.eval.trr_fcr`` can score TRR / FCR directly.

Steps 1-2 are exposed separately as :func:`propose_candidates`, so a learned
scorer can replace the distance ordering of step 3 while the candidate set
stays byte-identical -- that is what the ``evapore_scorer`` variant does.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from src.topo import skeleton as sk
from src.eval.trr_fcr import CandidateEdge
from src.baselines.common import (
    RepairOutput, as_bool, bezier_path, endpoint_table, paint_edges, straight_path,
)

__all__ = ["repair", "repair_detailed", "propose_candidates", "DEFAULTS"]

DEFAULTS = dict(
    alpha_d_over_r=6.0,      # d <= alpha * r_bar
    tau_tangent=0.5,         # cos threshold, both sides
    max_radius_ratio=2.0,    # max(r)/min(r)
    border_px=5.0,           # drop endpoints this close to the FOV edge
    trace_px=12,             # pixels used for the endpoint tangent PCA
    min_branch_px=3,         # spur pruning
    bezier_deg=15.0,         # straight below this tangent/chord deviation
    require_diff_component=True,
    max_candidates=None,
)


def propose_candidates(
    mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    **kw: Any,
):
    """Enumerate this harness's admissible endpoint-pair candidates.

    Steps 1-2 of the rule above -- endpoints, the distance / tangent / radius
    gates, and the rasterised straight-or-Bezier path -- **without** the greedy
    acceptance of step 3.

    Returns ``(candidates, endpoints, cfg)``.  Each candidate dict carries
    ``i``, ``j``, ``d``, ``r_bar``, ``path``, ``shape`` and the gate values;
    ``endpoints`` is the :func:`endpoint_table` dict, with the FOV-masked mask
    and the pair count added under ``mask_in_fov`` / ``fov_used`` /
    ``n_pairs_examined``.
    """
    cfg = dict(DEFAULTS)
    unknown = set(kw) - set(cfg)
    if unknown:
        raise TypeError(f"unknown geometric_repair option(s): {sorted(unknown)}")
    cfg.update(kw)

    m = as_bool(mask)
    f = sk.fov_or_true(fov, m.shape)
    m = m & f

    ep = endpoint_table(m, fov=f, border_px=cfg["border_px"],
                        trace_px=cfg["trace_px"],
                        min_branch_px=cfg["min_branch_px"])
    coord = ep["coord"].astype(np.float64)
    tan = ep["tangent"]
    rad = ep["radius"]
    comp = ep["comp"]
    n = int(coord.shape[0])

    cands: List[Dict[str, Any]] = []
    n_pairs = 0
    cos_min = math.cos(math.radians(float(cfg["bezier_deg"])))
    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            if cfg["require_diff_component"] and comp[i] == comp[j]:
                continue
            r_bar = 0.5 * (rad[i] + rad[j])
            v = coord[j] - coord[i]
            d = float(np.linalg.norm(v))
            if d < 1e-6 or d > cfg["alpha_d_over_r"] * r_bar:
                continue
            ratio = max(rad[i], rad[j]) / max(min(rad[i], rad[j]), 1e-6)
            if ratio > cfg["max_radius_ratio"]:
                continue
            u = v / d
            cos_i = float(tan[i] @ u)
            cos_j = float(tan[j] @ (-u))
            if cos_i < cfg["tau_tangent"] or cos_j < cfg["tau_tangent"]:
                continue
            # straight when both tangents already point along the chord
            if cos_i >= cos_min and cos_j >= cos_min:
                path, shape = straight_path(coord[i], coord[j]), "straight"
            else:
                path = bezier_path(coord[i], tan[i], coord[j], tan[j],
                                   n=max(16, int(2 * d)))
                shape = "bezier"
            cands.append(dict(i=i, j=j, d=d, r_bar=float(r_bar), path=path,
                              shape=shape, cos_i=cos_i, cos_j=cos_j,
                              ratio=float(ratio), comp_a=int(comp[i]),
                              comp_b=int(comp[j])))

    ep["n_pairs_examined"] = n_pairs
    ep["mask_in_fov"] = m
    ep["fov_used"] = f
    return cands, ep, cfg


def repair_detailed(
    image: Optional[np.ndarray],
    prob: Optional[np.ndarray],
    mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    **kw: Any,
) -> RepairOutput:
    """Geometric repair of ``mask``.  ``image`` and ``prob`` are unused.

    They stay in the signature so every baseline shares one interface; that
    this method ignores them is exactly the point of the comparison.
    """
    cands, ep, cfg = propose_candidates(mask, fov, **kw)
    m = ep["mask_in_fov"]
    f = ep["fov_used"]
    coord = ep["coord"].astype(np.float64)
    n = int(coord.shape[0])

    info: Dict[str, Any] = dict(method="geometric", n_endpoints=n,
                                n_pairs_examined=ep["n_pairs_examined"],
                                n_candidates=len(cands))

    # ---- greedy acceptance by increasing distance ----------------------
    cands = sorted(cands, key=lambda c: c["d"])
    if cfg["max_candidates"]:
        cands = cands[: int(cfg["max_candidates"])]

    used = np.zeros(n, dtype=bool)
    edges: List[CandidateEdge] = []
    for c in cands:
        i, j = c["i"], c["j"]
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        edges.append(CandidateEdge(
            path=c["path"],
            p0=(int(coord[i, 0]), int(coord[i, 1])),
            p1=(int(coord[j, 0]), int(coord[j, 1])),
            radius=float(c["r_bar"]),
            edge_id=f"geo_{i}-{j}", score=float(1.0 / (1.0 + c["d"])),
            meta=dict(source="geometric", shape=c["shape"], dist=c["d"],
                      r_bar=c["r_bar"], cos_i=c["cos_i"], cos_j=c["cos_j"],
                      radius_ratio=c["ratio"], comp_a=c["comp_a"],
                      comp_b=c["comp_b"]),
        ))

    info["n_accepted"] = len(edges)
    info["n_straight"] = sum(1 for e in edges if e.meta.get("shape") == "straight")
    info["n_bezier"] = len(edges) - info["n_straight"]

    out = paint_edges(m, edges, fov=f)
    info["n_added_px"] = int(np.count_nonzero(out & ~m))
    return RepairOutput(mask=out, edges=edges, info=info)


def repair(image, prob, mask, fov=None, **kw) -> np.ndarray:
    """Common interface: return only the repaired mask."""
    return repair_detailed(image, prob, mask, fov, **kw).mask
