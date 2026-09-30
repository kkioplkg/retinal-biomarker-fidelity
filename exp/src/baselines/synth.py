"""Synthetic development inputs for the S6 baselines.

Real predictions land in ``exp/runs/seg/<ds>/seed<k>/pred/{prob,mask}`` once
the S2 training finishes.  Until then -- and for the unit tests afterwards --
a baseline needs a ``(M_hat, P)`` pair with a *known* set of breaks:

    M_hat = GT with random capsule cuts
    P     = Gaussian-blurred M_hat rescaled into [lo, hi]

That is exactly what ``src.rigr.synth_cuts`` provides, so this module simply
delegates when that package is importable, and falls back to a self-contained
re-implementation of the same two operations otherwise (the baselines must not
break while ``src/rigr`` is under construction).

The fallback follows ``synth_cuts.capsule_cut``: a locus on the GT skeleton,
the geodesic skeleton segment within ``L / 2`` of it (``L = length_factor * D``,
``D = 2 * EDT``), dilated by ``r_local + pad`` and subtracted from the mask.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.ndimage as ndi

from src.topo import skeleton as sk

__all__ = ["make_cut_mask", "synth_prob", "sample_loci", "synthesize",
           "load_pred_pair", "training_inputs"]

DEFAULT_LENGTH_FACTORS = (0.5, 1.0, 2.0, 4.0)


def _rigr():
    """``src.rigr.synth_cuts`` if it imports cleanly, else ``None``."""
    try:
        from src.rigr import synth_cuts  # type: ignore
        return synth_cuts
    except Exception:
        return None


# --------------------------------------------------------------------------


def _geodesic_segment(skel: np.ndarray, locus: Tuple[int, int],
                      half_len: float) -> np.ndarray:
    """Skeleton pixels within ``half_len`` geodesic distance of ``locus``.

    Dijkstra over the 8-neighbour skeleton graph with unit / sqrt(2) weights,
    stopped at the radius; it does **not** cross bifurcations differently from
    ordinary pixels, matching the reference implementation.
    """
    import heapq

    s = sk.as_bool(skel)
    h, w = s.shape
    start = (int(locus[0]), int(locus[1]))
    if not s[start]:
        return np.asarray([start], dtype=np.int64)
    dist = {start: 0.0}
    heap = [(0.0, start)]
    out = [start]
    while heap:
        d, cur = heapq.heappop(heap)
        if d > dist.get(cur, np.inf):
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nb = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nb[0] < h and 0 <= nb[1] < w) or not s[nb]:
                    continue
                nd = d + (1.0 if dr == 0 or dc == 0 else np.sqrt(2.0))
                if nd <= half_len and nd < dist.get(nb, np.inf):
                    dist[nb] = nd
                    heapq.heappush(heap, (nd, nb))
                    out.append(nb)
    return np.asarray(sorted(set(out)), dtype=np.int64)


def sample_loci(
    gt_mask: np.ndarray,
    n_cuts: int,
    fov: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    length_factors: Sequence[float] = DEFAULT_LENGTH_FACTORS,
    min_dist_to_junction: float = 6.0,
    min_radius: float = 1.0,
) -> List[Tuple[int, int, float]]:
    """Pick ``n_cuts`` ``(row, col, length_factor)`` severance loci on the GT.

    Loci are skeleton pixels of degree 2 (no endpoints, no bifurcations) that
    are at least ``min_dist_to_junction`` pixels away from any junction or
    endpoint -- the structural-quality gate of proposal Alg. 1 -- and thick
    enough that a capsule actually removes something.
    """
    rng = rng or np.random.default_rng(0)
    g = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, g.shape)
    g = g & f
    skel = sk.skeletonize_mask(g, min_branch_px=3, prune=True, fov=f)
    deg = sk.neighbour_count(skel)
    radius = sk.local_radius(g)

    special = skel & ((deg >= 3) | (deg <= 1))
    d_special = ndi.distance_transform_edt(~special) if special.any() else np.full(g.shape, np.inf)

    ok = skel & (deg == 2) & (d_special >= float(min_dist_to_junction)) & (radius >= float(min_radius))
    rr, cc = np.nonzero(ok)
    if rr.size == 0:
        return []
    idx = rng.choice(rr.size, size=min(int(n_cuts), rr.size), replace=False)
    lf = rng.choice(np.asarray(length_factors, dtype=float), size=len(idx))
    return [(int(rr[k]), int(cc[k]), float(lf[t])) for t, k in enumerate(idx)]


def make_cut_mask(
    gt_mask: np.ndarray,
    loci: Sequence[Tuple[int, int, float]],
    fov: Optional[np.ndarray] = None,
    pad: float = 1.0,
    verify_beta0: bool = True,
):
    """Apply capsule severances.  Returns ``(cut_mask, records_list)``.

    Delegates to ``src.rigr.synth_cuts.make_cut_mask`` when available (its
    ``records`` DataFrame is converted to a list of dicts so callers do not
    have to care which path was taken).
    """
    mod = _rigr()
    if mod is not None:
        try:
            cut, df = mod.make_cut_mask(gt_mask, loci, fov=fov, pad=pad,
                                        verify_beta0=verify_beta0)
            recs = df.to_dict("records") if hasattr(df, "to_dict") else list(df)
            for r in recs:
                r["backend"] = "rigr"
            return cut, recs
        except Exception:
            pass  # fall through to the local implementation

    from skimage.morphology import disk

    g = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, g.shape)
    g = g & f
    skel = sk.skeletonize_mask(g, min_branch_px=3, prune=True, fov=f)
    radius = sk.local_radius(g)

    cur = g.copy()
    recs: List[Dict[str, Any]] = []
    for (r, c, lf) in loci:
        b0_before = sk.betti_numbers(cur, f)[0] if verify_beta0 else 0
        r_loc = max(float(radius[int(r), int(c)]), 0.5)
        D = 2.0 * r_loc
        seg = _geodesic_segment(skel, (int(r), int(c)), 0.5 * float(lf) * D)
        stamp = np.zeros(g.shape, dtype=bool)
        stamp[seg[:, 0], seg[:, 1]] = True
        rad_px = int(max(1, round(r_loc + float(pad))))
        remove = ndi.binary_dilation(stamp, structure=disk(rad_px).astype(bool)) & cur
        nxt = cur & ~remove
        info: Dict[str, Any] = dict(
            locus=(int(r), int(c)), radius=r_loc, diameter=D,
            length=float(lf) * D, length_factor=float(lf),
            n_seg=int(seg.shape[0]), n_removed=int(remove.sum()),
            cut_radius=rad_px, backend="local",
        )
        if verify_beta0:
            b0_after = sk.betti_numbers(nxt, f)[0]
            info["delta_beta0"] = int(b0_after - b0_before)
            info["verified"] = bool(b0_after - b0_before == 1)
        else:
            info["delta_beta0"] = -1
            info["verified"] = True
        cur = nxt
        recs.append(info)
    return cur, recs


def synth_prob(mask: np.ndarray, sigma: float = 2.0, lo: float = 0.02,
               hi: float = 0.95, noise: float = 0.0,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Stand-in probability map: blurred mask rescaled to ``[lo, hi]``."""
    mod = _rigr()
    if mod is not None:
        try:
            return mod.synth_prob(mask, sigma=sigma, lo=lo, hi=hi,
                                  noise=noise, rng=rng)
        except Exception:
            pass
    m = sk.as_bool(mask).astype(np.float32)
    p = ndi.gaussian_filter(m, float(sigma))
    mx = float(p.max())
    p = p / mx if mx > 0 else p
    p = lo + (hi - lo) * p
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        p = np.clip(p + rng.normal(0, float(noise), p.shape), 0.0, 1.0)
    return p.astype(np.float32)


def synthesize(
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    n_cuts: int = 12,
    seed: int = 0,
    sigma: float = 2.0,
    noise: float = 0.0,
    length_factors: Sequence[float] = DEFAULT_LENGTH_FACTORS,
    **kw,
):
    """One-call ``GT -> (M_hat, P, cut_records)`` for development and tests."""
    rng = np.random.default_rng(int(seed))
    loci = sample_loci(gt_mask, n_cuts, fov=fov, rng=rng,
                       length_factors=length_factors, **kw)
    cut, recs = make_cut_mask(gt_mask, loci, fov=fov)
    prob = synth_prob(cut, sigma=sigma, noise=noise, rng=rng)
    return cut, prob, recs


# --------------------------------------------------------------------------
# training inputs for the learned baselines (stage S4 step E)
# --------------------------------------------------------------------------
def load_pred_pair(pred_dir, image_id, shape=None):
    """``(prob, mask)`` from a ``src.seg.infer`` output dir, or ``(None, None)``.

    Accepts the layout every prediction directory in this project uses --
    ``prob/<key>.npy`` (float, native resolution) + ``mask/<key>.png`` -- which
    is what ``runs/seg_oof/<ds>/pred`` and ``runs/seg/<ds>/seed<k>/pred`` both
    are.  ``<key>`` is the image path's basename without its extension, which
    is **not** always ``rec["image_id"]``: FIVES prefixes its ids with the split
    (``train_100_A``) because its ``train/``/``val/``/``test/`` folders reuse the
    same numbers, while the files are named ``100_A.png``.  ``image_id`` may
    therefore be an ordered list of candidate keys; the first that resolves
    wins.
    """
    import os

    if not pred_dir:
        return None, None
    keys = [image_id] if isinstance(image_id, str) else list(image_id)
    keys = [k for i, k in enumerate(keys) if k and k not in keys[:i]]

    prob = None
    mask = None
    for key in keys:
        for ext in (".npy", ".npz"):
            q = os.path.join(str(pred_dir), "prob", key + ext)
            if os.path.exists(q):
                a = np.load(q)
                prob = np.asarray(a["prob"] if ext == ".npz" else a,
                                  dtype=np.float32)
                break
        for ext in (".png", ".tif", ".npy"):
            q = os.path.join(str(pred_dir), "mask", key + ext)
            if os.path.exists(q):
                if ext == ".npy":
                    mask = np.load(q) > 0.5
                else:
                    from src.data.datasets import read_binary

                    mask = read_binary(q)
                break
        if mask is not None or prob is not None:
            break

    if mask is None and prob is not None:
        mask = prob >= 0.5
    if mask is None:
        return None, None
    if prob is None:
        prob = synth_prob(mask)
    return np.clip(prob, 0.0, 1.0), sk.as_bool(mask)


def training_inputs(
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray],
    image_id="",
    pred_dir=None,
    n_cuts: int = 12,
    seed: int = 0,
    extra_cuts: bool = True,
    sigma: float = 2.0,
    **kw,
):
    """The imperfect mask a learned repair baseline is trained on.

    Returns ``(M_hat, P, cut_records, source)``.

    ``pred_dir=None`` (the historical behaviour) gives ``M_hat`` = GT with
    ``n_cuts`` uniform capsule severances and ``P`` = its blurred copy.

    ``pred_dir`` set gives the **out-of-fold predicted mask** of that image as
    ``M_hat`` and the predicted probability map as ``P``, so the baseline is
    trained against the *segmenter's own* failures rather than synthetic ones;
    with ``extra_cuts`` (the default) the same uniform capsule severances are
    then applied on top, which is the training input stage S4 gives every
    method -- RiGR's pair scorer, rNCA and EVAPORE alike -- so that the
    comparison is not confounded by what each method saw.
    """
    prob_p, mask_p = (load_pred_pair(pred_dir, image_id) if pred_dir
                      else (None, None))
    if mask_p is None:
        cut, prob, recs = synthesize(gt_mask, fov, n_cuts=n_cuts, seed=seed,
                                     sigma=sigma, **kw)
        return cut, prob, recs, ("gt_cuts" if not pred_dir else
                                 "gt_cuts(NO PREDICTION for %r)" % (image_id,))
    base = sk.as_bool(mask_p)
    if fov is not None:
        base = base & sk.as_bool(fov)
    recs = []
    if extra_cuts and n_cuts > 0:
        rng = np.random.default_rng(int(seed))
        loci = sample_loci(base, n_cuts, fov=fov, rng=rng, **kw)
        base, recs = make_cut_mask(base, loci, fov=fov)
    prob = np.clip(np.asarray(prob_p, dtype=np.float32), 0.0, 1.0)
    if extra_cuts and n_cuts > 0:
        # the severances must also be visible in P, otherwise the model is
        # handed the answer through the probability channel
        prob = np.minimum(prob, np.where(base, 1.0, synth_prob(base, sigma=sigma)))
    return base, prob, recs, "oof_pred" + ("+cuts" if (extra_cuts and n_cuts) else "")
