"""Three-layer segmentation metrics (proposal v3 §4.2), all FOV-restricted.

Layer 1 -- pixel : ``dice`` / ``f1``, ``iou``, ``acc``, ``sen``, ``spe``,
                   ``auc`` (ROC-AUC from the probability map), ``pr_auc``.
Layer 2 -- topology : ``cldice``, ``betti_error``, ``junction_f1``, ``bcs``,
                   ``component_count_error``.

**Everything is computed inside a boolean FOV mask**: pixels with ``fov == 0``
are dropped from the pixel metrics and are removed from the masks before any
skeletonisation or connected-component labelling.  ``fov=None`` means "whole
image".

Connectivity convention: foreground 8-connected, background 4-connected
(see :mod:`src.topo.skeleton`).

Conventions for degenerate inputs (so a sweep over thresholds never crashes):

* ``dice``/``iou`` of two empty masks = 1.0; of one empty mask = 0.0.
* ``sen`` with no GT foreground, ``spe`` with no GT background, ``auc``/``pr_auc``
  with a single GT class, ``bcs`` with no GT bifurcation, and ``junction_f1``
  with no junction on either side all return ``nan``.
* ``junction_f1`` returns 0.0 when exactly one of the two masks has junctions.

Run ``cd exp && python -m src.topo.metrics`` for a smoke demo.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi

from src.topo import skeleton as sk

__all__ = [
    "dice",
    "f1",
    "iou",
    "acc",
    "sen",
    "spe",
    "auc",
    "pr_auc",
    "cldice",
    "betti_error",
    "junction_f1",
    "bcs",
    "component_count_error",
    "evaluate_all",
]

_NAN = float("nan")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _prep(pred, gt, fov):
    """Return (pred, gt, fov) as boolean arrays, masked to the FOV."""
    p = sk.as_bool(pred)
    g = sk.as_bool(gt)
    if p.shape != g.shape:
        raise ValueError(f"pred shape {p.shape} != gt shape {g.shape}")
    f = sk.fov_or_true(fov, p.shape)
    return p & f, g & f, f


def _counts(pred, gt, fov):
    """TP, FP, FN, TN inside the FOV."""
    p, g, f = _prep(pred, gt, fov)
    pv = p[f]
    gv = g[f]
    tp = float(np.count_nonzero(pv & gv))
    fp = float(np.count_nonzero(pv & ~gv))
    fn = float(np.count_nonzero(~pv & gv))
    tn = float(np.count_nonzero(~pv & ~gv))
    return tp, fp, fn, tn


# --------------------------------------------------------------------------
# layer 1: pixel metrics
# --------------------------------------------------------------------------
def dice(pred, gt, fov=None) -> float:
    """Dice / F1 of the foreground class inside the FOV.  Empty+empty = 1.0."""
    tp, fp, fn, _ = _counts(pred, gt, fov)
    denom = 2.0 * tp + fp + fn
    if denom == 0.0:
        return 1.0
    return float(2.0 * tp / denom)


def f1(pred, gt, fov=None) -> float:
    """Binary F1 -- identical to :func:`dice` for a binary mask."""
    return dice(pred, gt, fov)


def iou(pred, gt, fov=None) -> float:
    """Intersection over union (Jaccard) of the foreground inside the FOV."""
    tp, fp, fn, _ = _counts(pred, gt, fov)
    denom = tp + fp + fn
    if denom == 0.0:
        return 1.0
    return float(tp / denom)


def acc(pred, gt, fov=None) -> float:
    """Pixel accuracy inside the FOV."""
    tp, fp, fn, tn = _counts(pred, gt, fov)
    n = tp + fp + fn + tn
    if n == 0.0:
        return _NAN
    return float((tp + tn) / n)


def sen(pred, gt, fov=None) -> float:
    """Sensitivity / recall = TP / (TP + FN).  NaN if the GT is empty."""
    tp, _, fn, _ = _counts(pred, gt, fov)
    if tp + fn == 0.0:
        return _NAN
    return float(tp / (tp + fn))


def spe(pred, gt, fov=None) -> float:
    """Specificity = TN / (TN + FP).  NaN if the GT has no background."""
    _, fp, _, tn = _counts(pred, gt, fov)
    if tn + fp == 0.0:
        return _NAN
    return float(tn / (tn + fp))


def auc(prob, gt, fov=None) -> float:
    """ROC-AUC of the probability map inside the FOV (sklearn)."""
    if prob is None:
        return _NAN
    from sklearn.metrics import roc_auc_score

    g = sk.as_bool(gt)
    f = sk.fov_or_true(fov, g.shape)
    y = g[f].astype(np.uint8)
    if y.min() == y.max():
        return _NAN
    s = np.asarray(prob, dtype=np.float64)[f]
    return float(roc_auc_score(y, s))


def pr_auc(prob, gt, fov=None) -> float:
    """Area under the precision-recall curve inside the FOV.

    Implemented as sklearn's ``average_precision_score`` (the step-wise,
    non-interpolated estimator of AUPRC) -- the more informative summary at the
    ~8-12% foreground prevalence of fundus vessels (proposal §4.2 layer 1).
    """
    if prob is None:
        return _NAN
    from sklearn.metrics import average_precision_score

    g = sk.as_bool(gt)
    f = sk.fov_or_true(fov, g.shape)
    y = g[f].astype(np.uint8)
    if y.min() == y.max():
        return _NAN
    s = np.asarray(prob, dtype=np.float64)[f]
    return float(average_precision_score(y, s))


# --------------------------------------------------------------------------
# layer 2: topology metrics
# --------------------------------------------------------------------------
def cldice(pred, gt, fov=None) -> float:
    """centerlineDice (clDice) of Shit et al., CVPR 2021.

    ``Tprec = |S_pred & gt| / |S_pred|`` (topology precision),
    ``Tsens = |S_gt & pred| / |S_gt|`` (topology sensitivity),
    ``clDice = 2 * Tprec * Tsens / (Tprec + Tsens)``.

    Skeletons are the *unpruned* morphological skeletons of the FOV-restricted
    masks, following the original paper (pruning would change the metric).

    Degenerate cases: both skeletons empty -> 1.0; exactly one empty -> 0.0;
    ``Tprec + Tsens == 0`` -> 0.0.
    """
    p, g, f = _prep(pred, gt, fov)
    sp = sk.skeletonize_mask(p, prune=False)
    sg = sk.skeletonize_mask(g, prune=False)
    np_, ng = int(sp.sum()), int(sg.sum())
    if np_ == 0 and ng == 0:
        return 1.0
    if np_ == 0 or ng == 0:
        return 0.0
    t_prec = float(np.count_nonzero(sp & g)) / float(np_)
    t_sens = float(np.count_nonzero(sg & p)) / float(ng)
    if t_prec + t_sens == 0.0:
        return 0.0
    return float(2.0 * t_prec * t_sens / (t_prec + t_sens))


def betti_error(pred, gt, fov=None) -> Dict[str, float]:
    """Absolute Betti-number errors inside the FOV.

    Returns ``dict(beta0_err, beta1_err)`` plus the four raw counts.  Betti
    numbers use foreground 8-connectivity / background 4-connectivity with
    border-touching background merged (see :func:`src.topo.skeleton.betti_numbers`).

    Note (proposal §4.2): the Betti *number* error is not spatially faithful --
    it can be small while the errors are in the wrong places.  Betti *matching*
    error is reported separately in the supplement.
    """
    p, g, f = _prep(pred, gt, fov)
    b0p, b1p = sk.betti_numbers(p, f)
    b0g, b1g = sk.betti_numbers(g, f)
    return {
        "beta0_err": float(abs(b0p - b0g)),
        "beta1_err": float(abs(b1p - b1g)),
        "beta0_pred": float(b0p),
        "beta1_pred": float(b1p),
        "beta0_gt": float(b0g),
        "beta1_gt": float(b1g),
    }


def _match_points(a: np.ndarray, b: np.ndarray, tol: float):
    """One-to-one matching of two point sets within ``tol`` (min total distance).

    Uses the Hungarian algorithm on the dense distance matrix when the problem
    is small, and a greedy nearest-neighbour pass (KD-tree, shortest pairs
    first) when it is large.  Returns ``(n_matched, pairs)``.
    """
    if len(a) == 0 or len(b) == 0:
        return 0, []
    from scipy.spatial import cKDTree

    tree = cKDTree(b)
    pairs = tree.query_ball_point(a, r=tol)
    cand = [(i, j) for i, js in enumerate(pairs) for j in js]
    if not cand:
        return 0, []

    if len(a) * len(b) <= 4_000_000:
        from scipy.optimize import linear_sum_assignment

        big = float(tol) * 10.0 + 1.0
        cost = np.full((len(a), len(b)), big, dtype=np.float64)
        ia = np.array([c[0] for c in cand])
        ib = np.array([c[1] for c in cand])
        d = np.linalg.norm(np.asarray(a, float)[ia] - np.asarray(b, float)[ib], axis=1)
        cost[ia, ib] = d
        ri, ci = linear_sum_assignment(cost)
        keep = cost[ri, ci] <= tol
        matched = [(int(r), int(c)) for r, c, k in zip(ri, ci, keep) if k]
        return len(matched), matched

    # greedy fallback for very large point sets
    d = np.linalg.norm(
        np.asarray(a, float)[[c[0] for c in cand]] - np.asarray(b, float)[[c[1] for c in cand]],
        axis=1,
    )
    order = np.argsort(d)
    used_a, used_b, matched = set(), set(), []
    for k in order:
        i, j = cand[k]
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j))
    return len(matched), matched


def junction_f1(pred, gt, fov=None, tol_px: float = 3.0, min_branch_px: int = 3) -> float:
    """F1 of detected bifurcation points, matched one-to-one within ``tol_px``.

    Junction points are the centroids of 8-connected clusters of degree->=3
    pixels of the **spur-pruned** skeleton (pruning matters: a 1-2 px spur
    creates a spurious junction).  Predicted and GT junction points are matched
    one-to-one by minimum total distance (Hungarian), keeping only pairs within
    ``tol_px``; F1 = 2m / (n_pred + n_gt).

    Returns NaN when neither mask has any junction, 0.0 when exactly one does.
    """
    p, g, f = _prep(pred, gt, fov)
    jp, _, _ = sk.junction_clusters(sk.skeletonize_mask(p, min_branch_px=min_branch_px))
    jg, _, _ = sk.junction_clusters(sk.skeletonize_mask(g, min_branch_px=min_branch_px))
    n_p, n_g = len(jp), len(jg)
    if n_p == 0 and n_g == 0:
        return _NAN
    if n_p == 0 or n_g == 0:
        return 0.0
    m, _ = _match_points(jp, jg, float(tol_px))
    return float(2.0 * m / (n_p + n_g))


def _probe_points(
    skel: np.ndarray, probe_px: int, min_degree: int, fov: np.ndarray
):
    """For every GT bifurcation, the probe point of each incident branch.

    Yields ``(centroid, [probe_rc, ...])``.  A probe point is reached by walking
    from the junction along an incident branch for ``min(branch length,
    probe_px)`` 8-connected steps (hop count; diagonal steps count as one).
    """
    s = sk.as_bool(skel)
    deg_map = sk.neighbour_count(s)
    junc = s & (deg_map >= 3)
    seg = s & ~junc
    lab, n = ndi.label(junc, structure=sk.EIGHT)
    if n == 0:
        return
    objs = ndi.find_objects(lab)
    H, W = s.shape
    pad = int(probe_px) + 3
    struct = sk.EIGHT.astype(bool)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        r0 = max(0, sl[0].start - pad)
        r1 = min(H, sl[0].stop + pad)
        c0 = max(0, sl[1].start - pad)
        c1 = min(W, sl[1].stop + pad)
        win = (slice(r0, r1), slice(c0, c1))
        cluster = lab[win] == (i + 1)
        seg_w = seg[win]
        stubs = ndi.binary_dilation(cluster, structure=struct) & seg_w
        stub_lab, k = ndi.label(stubs, structure=sk.EIGHT)
        if k < min_degree:
            continue
        centroid = np.array(ndi.center_of_mass(cluster), dtype=float) + np.array([r0, c0])
        if not fov[int(round(centroid[0])), int(round(centroid[1]))]:
            continue
        probes = []
        for j in range(1, k + 1):
            start = stub_lab == j
            visited = start.copy()
            frontier = start.copy()
            for _ in range(int(probe_px)):
                nxt = ndi.binary_dilation(frontier, structure=struct) & seg_w & ~visited
                if not nxt.any():
                    break
                visited |= nxt
                frontier = nxt
            rr, cc = np.nonzero(frontier)
            if rr.size == 0:
                continue
            d = (rr + r0 - centroid[0]) ** 2 + (cc + c0 - centroid[1]) ** 2
            b = int(np.argmax(d))
            probes.append((int(rr[b] + r0), int(cc[b] + c0)))
        if len(probes) >= min_degree:
            yield centroid, probes


def bcs(
    pred,
    gt,
    fov=None,
    probe_px: int = 15,
    snap_px: float = 2.0,
    min_degree: int = 3,
    min_branch_px: int = 3,
) -> float:
    """Bifurcation Connectedness Score -- our re-implementation of the concept.

    Re-implementation of the *idea* of BCS (Owusu-Ansah, Lee, Venugopal, Jawaid,
    Duan, Brown, *Same Branches, Different Trees: A Bifurcation Connectedness
    Metric for Coronary Artery Segmentation and FFR-CT Decision Agreement*,
    arXiv:2607.28327, 2026): global centreline metrics such as clDice and
    Skeleton Recall average away *local* breaks at bifurcations, so score the
    bifurcations directly.  The exact operationalisation below is ours, not the
    authors', and the two numbers are not interchangeable with theirs.

    Definition used here
    --------------------
    1. Skeletonise the FOV-restricted GT and find every bifurcation (junction
       cluster with ``>= min_degree`` incident branch stubs, default 3).
    2. For each incident GT branch, walk ``min(branch length, probe_px)`` steps
       away from the junction along the GT skeleton -> one *probe point* per
       branch (default ``probe_px = 15``).
    3. Look each probe point up in the **8-connected component map of the
       prediction**.  A probe that falls on predicted background is snapped to
       the nearest predicted foreground pixel within ``snap_px`` (default 2 px,
       to absorb sub-pixel centreline offsets); if there is none, the probe has
       no component.
    4. The bifurcation is *connected* iff every probe resolves to the **same**
       predicted component.
    5. ``BCS`` = fraction of connected GT bifurcations.  NaN if the GT has no
       qualifying bifurcation inside the FOV.
    """
    p, g, f = _prep(pred, gt, fov)
    gskel = sk.skeletonize_mask(g, min_branch_px=min_branch_px)
    plab, _ = sk.connected_components(p, connectivity=8)

    # nearest predicted-foreground pixel, for the snap tolerance
    if snap_px and snap_px > 0 and p.any():
        dist, idx = ndi.distance_transform_edt(~p, return_indices=True)
    else:
        dist, idx = None, None

    n_total = 0
    n_connected = 0
    for _centroid, probes in _probe_points(gskel, probe_px, min_degree, f):
        n_total += 1
        comps = set()
        ok = True
        for (r, c) in probes:
            lab_v = int(plab[r, c])
            if lab_v == 0 and dist is not None and dist[r, c] <= snap_px:
                lab_v = int(plab[idx[0][r, c], idx[1][r, c]])
            if lab_v == 0:
                ok = False
                break
            comps.add(lab_v)
        if ok and len(comps) == 1:
            n_connected += 1
    if n_total == 0:
        return _NAN
    return float(n_connected / n_total)


def component_count_error(pred, gt, fov=None, connectivity: int = 8) -> float:
    """``|#components(pred) - #components(gt)|`` inside the FOV (8-connected)."""
    p, g, f = _prep(pred, gt, fov)
    _, np_ = sk.connected_components(p, connectivity=connectivity)
    _, ng = sk.connected_components(g, connectivity=connectivity)
    return float(abs(np_ - ng))


# --------------------------------------------------------------------------
# aggregate
# --------------------------------------------------------------------------
def evaluate_all(
    prob: Optional[np.ndarray],
    pred: np.ndarray,
    gt: np.ndarray,
    fov: Optional[np.ndarray] = None,
    tol_px: float = 3.0,
    probe_px: int = 15,
    min_branch_px: int = 3,
) -> Dict[str, float]:
    """Compute every metric of layers 1-2 for one image.  All values are floats.

    ``prob`` may be ``None`` (then ``auc``/``pr_auc`` are NaN).  Robust to an
    empty prediction: no metric raises, degenerate ones return the conventions
    documented at the top of this module.
    """
    p, g, f = _prep(pred, gt, fov)
    out: Dict[str, float] = {}
    out["dice"] = dice(p, g, f)
    out["f1"] = out["dice"]
    out["iou"] = iou(p, g, f)
    out["acc"] = acc(p, g, f)
    out["sen"] = sen(p, g, f)
    out["spe"] = spe(p, g, f)
    out["auc"] = auc(prob, g, f)
    out["pr_auc"] = pr_auc(prob, g, f)
    out["cldice"] = cldice(p, g, f)
    out.update(betti_error(p, g, f))
    out["junction_f1"] = junction_f1(p, g, f, tol_px=tol_px, min_branch_px=min_branch_px)
    out["bcs"] = bcs(p, g, f, probe_px=probe_px, min_branch_px=min_branch_px)
    out["component_count_error"] = component_count_error(p, g, f)
    _, n_cc_p = sk.connected_components(p, connectivity=8)
    _, n_cc_g = sk.connected_components(g, connectivity=8)
    out["n_cc_pred"] = float(n_cc_p)
    out["n_cc_gt"] = float(n_cc_g)
    out["n_pred_px"] = float(np.count_nonzero(p))
    out["n_gt_px"] = float(np.count_nonzero(g))
    out["n_fov_px"] = float(np.count_nonzero(f))
    return {k: float(v) for k, v in out.items()}


# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - smoke demo
    rng = np.random.default_rng(0)
    gt = np.zeros((96, 96), dtype=bool)
    gt[46:50, 10:86] = True
    gt[20:50, 46:50] = True
    pred = gt.copy()
    pred[:, 60:64] = False  # a break
    prob = np.where(pred, 0.9, 0.05) + rng.normal(0, 0.01, gt.shape)
    res = evaluate_all(prob, pred, gt, None)
    for k in sorted(res):
        print(f"{k:24s} {res[k]:.4f}")
