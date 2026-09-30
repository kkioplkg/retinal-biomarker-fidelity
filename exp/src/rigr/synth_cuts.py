"""Two-layer synthetic severance supervision for the pair scorer (section 3.2.5).

Layer (a) -- uniform cuts
-------------------------
Capsule severances (the same rule as the C1 perturbation engine of section
3.1.3) are applied to the **reference** mask at uniformly sampled loci that
pass the structural quality gate (away from junctions and from branch ends),
with ``L in {1, 2, 4} * D`` where ``D = 2 r`` is the local diameter.  Each cut
is verified to raise ``beta0`` by exactly one.

The cut mask plays the role of ``M_hat``; :mod:`src.rigr.candidates` is then run
on it and **every** candidate it produces is labelled with the formal true-repair
criterion of :mod:`src.eval.trr_fcr` -- not just the pair that was severed.
Positives are therefore the candidates that genuinely restore a reference
connection, and the negatives come with the hard cases for free: parallel
vessels, neighbouring bifurcations, and the second-best partner of a severed
end.

Layer (b) -- failure-conditioned cuts
-------------------------------------
The uniform distribution of cut loci is not the distribution of real baseline
failures.  ``FailurePrior`` estimates
``pi(gap_len, radius, local_contrast, eccentricity_zone, fov_edge_dist)`` from
**out-of-fold** baseline predictions on **source-domain training** images
(anti-leakage rules 1 and 2 of section 3.2.5: the held-out domain never
contributes, and the error profile of an image never comes from a model
trained on it).  ``pi`` is a plain 5-D histogram, so it is inspectable and
serialisable; cut loci on reference masks are then drawn with weights equal to
the marginal of ``pi`` over the four locus features, and the cut length is
drawn from the conditional over the gap-length axis.

The two layers are mixed with weight ``omega`` (``omega in {1, 0.5, 0}`` enters
the Tab.3 ablation): ``omega`` is the fraction of failure-conditioned loci.

The conditioning variables are exactly the five stable, automatically
obtainable quantities the section allows; lesion status is deliberately absent
(no uniform annotation) and crossings only ever enter as a geometric proxy.

Run ``cd exp && python -m src.rigr.synth_cuts`` for a synthetic demo.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

from src.topo import skeleton as sk

__all__ = [
    "LENGTH_FACTORS",
    "eligible_loci",
    "capsule_cut",
    "make_cut_mask",
    "synth_prob",
    "label_candidates",
    "FailurePrior",
    "collect_failure_events",
    "estimate_failure_prior",
    "sample_cut_loci",
    "build_training_pairs",
]

#: uniform-cut severance lengths, in local diameters D (section 3.1.3 / 3.2.5)
LENGTH_FACTORS: Tuple[float, ...] = (1.0, 2.0, 4.0)

_EPS = 1e-9
_SQRT2 = math.sqrt(2.0)
_NB8 = [(-1, -1, _SQRT2), (-1, 0, 1.0), (-1, 1, _SQRT2), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, _SQRT2), (1, 0, 1.0), (1, 1, _SQRT2)]


# --------------------------------------------------------------------------
# loci and capsule severance
# --------------------------------------------------------------------------
def eligible_loci(
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    skel: Optional[np.ndarray] = None,
    min_junction_D: float = 2.0,
    min_end_D: float = 1.0,
    border_px: float = 12.0,
    min_radius: float = 0.75,
) -> Dict[str, np.ndarray]:
    """Skeleton pixels that may be severed (the structural quality gate).

    A locus must be a degree-2 skeleton pixel, at least ``min_junction_D``
    local diameters from any junction, ``min_end_D`` diameters from any
    endpoint, ``border_px`` inside the FOV, and on a branch of radius at least
    ``min_radius`` (a half-pixel "vessel" cannot be meaningfully severed).

    Returns ``coord`` (N, 2), ``radius`` (N,), ``diameter`` (N,).
    """
    g = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, g.shape)
    g = g & f
    if skel is None:
        skel = sk.skeletonize_mask(g, min_branch_px=3, prune=True, fov=f)
    radius = sk.local_radius(g)
    deg = sk.neighbour_count(skel)

    junc = skel & (deg >= 3)
    ends = skel & (deg == 1)
    d_j = ndi.distance_transform_edt(~junc) if junc.any() else np.full(g.shape, 1e6)
    d_e = ndi.distance_transform_edt(~ends) if ends.any() else np.full(g.shape, 1e6)
    d_fov = ndi.distance_transform_edt(f)

    cand = skel & (deg == 2)
    rr, cc = np.nonzero(cand)
    if rr.size == 0:
        return dict(coord=np.zeros((0, 2), np.int64), radius=np.zeros(0),
                    diameter=np.zeros(0))
    r_loc = radius[rr, cc]
    D = 2.0 * r_loc
    keep = (
        (r_loc >= float(min_radius))
        & (d_j[rr, cc] >= float(min_junction_D) * D)
        & (d_e[rr, cc] >= float(min_end_D) * D)
        & (d_fov[rr, cc] >= float(border_px))
    )
    return dict(coord=np.stack([rr[keep], cc[keep]], axis=1).astype(np.int64),
                radius=r_loc[keep].astype(float), diameter=D[keep].astype(float))


def _geodesic_segment(skel: np.ndarray, locus: Tuple[int, int],
                      half_len: float) -> np.ndarray:
    """Skeleton pixels within ``half_len`` arc length of ``locus`` (both ways)."""
    import heapq

    H, W = skel.shape
    r0, c0 = int(locus[0]), int(locus[1])
    dist = {(r0, c0): 0.0}
    heap = [(0.0, r0, c0)]
    out = [(r0, c0)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist.get((r, c), np.inf) + 1e-9:
            continue
        for dr, dc, wgt in _NB8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or not skel[nr, nc]:
                continue
            nd = d + wgt
            if nd > half_len:
                continue
            if nd < dist.get((nr, nc), np.inf) - 1e-9:
                dist[(nr, nc)] = nd
                heapq.heappush(heap, (nd, nr, nc))
                out.append((nr, nc))
    return np.asarray(sorted(set(out)), dtype=np.int64)


def capsule_cut(
    mask: np.ndarray,
    skel: np.ndarray,
    locus: Tuple[int, int],
    length_factor: float,
    radius_map: Optional[np.ndarray] = None,
    pad: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove one capsule of length ``length_factor * D`` centred on ``locus``.

    The capsule is the skeleton segment within ``L/2`` arc length of the locus,
    dilated by ``r_local + pad``; it is subtracted from ``mask``.  Returns the
    boolean **removal** mask and an info dict (the caller decides whether to
    apply it and whether ``delta_beta0 == 1``).
    """
    from skimage.morphology import disk

    m = sk.as_bool(mask)
    s = sk.as_bool(skel)
    if radius_map is None:
        radius_map = sk.local_radius(m)
    r_loc = float(radius_map[int(locus[0]), int(locus[1])])
    D = 2.0 * r_loc
    L = float(length_factor) * D
    seg = _geodesic_segment(s, locus, 0.5 * L)
    stamp = np.zeros(m.shape, dtype=bool)
    stamp[seg[:, 0], seg[:, 1]] = True
    rad = int(max(1, round(r_loc + float(pad))))
    remove = ndi.binary_dilation(stamp, structure=disk(rad).astype(bool)) & m
    return remove, dict(locus=(int(locus[0]), int(locus[1])), radius=r_loc,
                        diameter=D, length=L, length_factor=float(length_factor),
                        n_seg=int(seg.shape[0]), n_removed=int(remove.sum()),
                        cut_radius=rad)


def make_cut_mask(
    gt_mask: np.ndarray,
    loci: Sequence[Tuple[int, int, float]],
    fov: Optional[np.ndarray] = None,
    skel: Optional[np.ndarray] = None,
    pad: float = 1.0,
    verify_beta0: bool = True,
):
    """Apply a list of ``(row, col, length_factor)`` capsule severances.

    Returns ``(cut_mask, records)`` where ``records`` is a DataFrame with one
    row per attempted cut, including ``delta_beta0`` and ``verified`` (True
    when the cut raised ``beta0`` by exactly 1, i.e. it really disconnected the
    branch rather than nibbling an edge).
    """
    import pandas as pd

    g = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, g.shape)
    g = g & f
    if skel is None:
        skel = sk.skeletonize_mask(g, min_branch_px=3, prune=True, fov=f)
    radius = sk.local_radius(g)

    cur = g.copy()
    rows = []
    for (r, c, lf) in loci:
        b0_before, _ = sk.betti_numbers(cur, f) if verify_beta0 else (0, 0)
        rem, info = capsule_cut(cur, skel, (int(r), int(c)), float(lf), radius, pad)
        nxt = cur & ~rem
        if verify_beta0:
            b0_after, _ = sk.betti_numbers(nxt, f)
            info["delta_beta0"] = int(b0_after - b0_before)
            info["verified"] = bool(b0_after - b0_before == 1)
        else:
            info["delta_beta0"] = -1
            info["verified"] = True
        cur = nxt
        rows.append(info)
    return cur, pd.DataFrame(rows)


def synth_prob(mask: np.ndarray, sigma: float = 2.0, lo: float = 0.02,
               hi: float = 0.95, noise: float = 0.0,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """A stand-in probability map: Gaussian-blurred mask, rescaled to ``[lo, hi]``.

    Used **only** until the real ``runs/seg/<ds>/<seed>/pred/prob/*.npy`` exist:
    it reproduces the one property that matters for the pipeline's plumbing --
    ``P ~ 0`` inside a break -- without pretending to be a real segmenter.
    """
    m = sk.as_bool(mask).astype(np.float32)
    p = ndi.gaussian_filter(m, float(sigma))
    mx = float(p.max())
    p = p / mx if mx > 0 else p
    p = lo + (hi - lo) * p
    if noise > 0:
        rng = rng or np.random.default_rng(0)
        p = np.clip(p + rng.normal(0, float(noise), p.shape), 0.0, 1.0)
    return p.astype(np.float32)


# --------------------------------------------------------------------------
# labelling
# --------------------------------------------------------------------------
def label_candidates(
    table,
    results: Sequence[Any],
    cut_mask: np.ndarray,
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    gt_skel: Optional[np.ndarray] = None,
    **criteria,
):
    """Label **every** candidate with the formal true-repair criterion.

    Returns ``(y, records)``: ``y`` is a 0/1 array aligned with ``table`` and
    ``records`` is the per-edge diagnostic table of
    :func:`src.eval.trr_fcr.classify_edges` (which criterion failed, geodesic
    length, corridor overlap, ...).  Candidates whose A* failed are labelled 0.
    """
    from src.eval.trr_fcr import CandidateEdge, classify_edges, rasterize_path

    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    edges: List[CandidateEdge] = []
    idx: List[int] = []
    for i, row in enumerate(rows):
        res = results[i] if i < len(results) else None
        if res is not None and getattr(res, "ok", False) and len(res.path_px):
            path = res.path_px
        else:
            path = rasterize_path([(int(row["r_i"]), int(row["c_i"])),
                                   (int(row["r_j"]), int(row["c_j"]))])
        edges.append(CandidateEdge(path=path, p0=(int(row["r_i"]), int(row["c_i"])),
                                   p1=(int(row["r_j"]), int(row["c_j"])),
                                   radius=float(row["rad_bar"]),
                                   edge_id=int(row["cand_id"])))
        idx.append(i)
    if not edges:
        import pandas as pd

        return np.zeros(0, dtype=int), pd.DataFrame()
    recs = classify_edges(edges, cut_mask, gt_mask, fov=fov, gt_skel=gt_skel,
                          **criteria)
    y = np.zeros(len(rows), dtype=int)
    for i, rec in zip(idx, recs):
        y[i] = int(bool(rec["is_true_repair"]))
    import pandas as pd

    return y, pd.DataFrame(recs)


# --------------------------------------------------------------------------
# failure-conditioned prior
# --------------------------------------------------------------------------
#: bin edges of pi; the first axis is the gap length, the other four are the
#: locus features used to *weight* loci on a reference mask.
PRIOR_EDGES: Dict[str, List[float]] = {
    "gap_len": [0.0, 2.0, 4.0, 8.0, 16.0, 32.0, np.inf],
    "radius": [0.0, 1.0, 1.5, 2.0, 3.0, np.inf],
    "contrast": [-np.inf, 0.5, 1.0, 1.5, 2.0, np.inf],
    "ecc_zone": [0.0, 0.25, 0.5, 0.75, 1.0, np.inf],
    "fov_edge": [0.0, 0.10, 0.25, 0.50, np.inf],
}
PRIOR_AXES = ["gap_len", "radius", "contrast", "ecc_zone", "fov_edge"]


def _digitize(name: str, x) -> np.ndarray:
    e = np.asarray(PRIOR_EDGES[name], dtype=float)
    return np.clip(np.digitize(np.asarray(x, dtype=float), e[1:-1], right=False),
                   0, len(e) - 2)


@dataclass
class FailurePrior:
    """5-D histogram ``pi(gap_len, radius, contrast, ecc_zone, fov_edge)``.

    ``fit`` accepts the DataFrame produced by :func:`collect_failure_events`.
    ``locus_weights`` returns the marginal over the gap-length axis, evaluated
    at candidate loci; ``sample_gap_len`` draws a severance length from the
    conditional at those loci (falling back to the global gap-length marginal
    for an unpopulated cell).
    """

    counts: np.ndarray = field(default_factory=lambda: np.zeros(
        tuple(len(PRIOR_EDGES[a]) - 1 for a in PRIOR_AXES), dtype=np.float64))
    n_events: int = 0
    n_images: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- fitting -----------------------------------------------------------
    def fit(self, events, smoothing: float = 0.5) -> "FailurePrior":
        shape = tuple(len(PRIOR_EDGES[a]) - 1 for a in PRIOR_AXES)
        c = np.full(shape, float(smoothing), dtype=np.float64)
        if events is not None and len(events):
            idx = [_digitize(a, np.asarray(events[a], dtype=float)) for a in PRIOR_AXES]
            np.add.at(c, tuple(idx), 1.0)
            self.n_events = int(len(events))
            if "image" in getattr(events, "columns", []):
                self.n_images = int(events["image"].nunique())
        self.counts = c
        return self

    # -- sampling ----------------------------------------------------------
    def _locus_index(self, radius, contrast, ecc, fov_edge):
        return (_digitize("radius", radius), _digitize("contrast", contrast),
                _digitize("ecc_zone", ecc), _digitize("fov_edge", fov_edge))

    def locus_weights(self, radius, contrast, ecc, fov_edge) -> np.ndarray:
        marg = self.counts.sum(axis=0)
        i = self._locus_index(radius, contrast, ecc, fov_edge)
        w = marg[i]
        s = float(w.sum())
        return w / s if s > 0 else np.full(len(w), 1.0 / max(len(w), 1))

    def sample_gap_len(self, radius, contrast, ecc, fov_edge,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng or np.random.default_rng(0)
        i = self._locus_index(radius, contrast, ecc, fov_edge)
        cond = self.counts[:, i[0], i[1], i[2], i[3]]          # (n_len, N)
        tot = cond.sum(axis=0)
        glob = self.counts.sum(axis=(1, 2, 3, 4))
        glob = glob / max(glob.sum(), _EPS)
        edges = np.asarray(PRIOR_EDGES["gap_len"], dtype=float)
        hi = np.where(np.isfinite(edges[1:]), edges[1:], edges[-2] * 2.0)
        out = np.zeros(cond.shape[1], dtype=float)
        for n in range(cond.shape[1]):
            p = cond[:, n] / tot[n] if tot[n] > 0 else glob
            b = int(rng.choice(len(p), p=p / p.sum()))
            out[n] = float(rng.uniform(edges[b], hi[b]))
        return np.maximum(out, 1.0)

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(counts=self.counts.tolist(), n_events=self.n_events,
                           n_images=self.n_images, axes=PRIOR_AXES,
                           edges={k: [x if np.isfinite(x) else None
                                      for x in v] for k, v in PRIOR_EDGES.items()},
                           meta=self.meta), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "FailurePrior":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        p = cls()
        p.counts = np.asarray(d["counts"], dtype=np.float64)
        p.n_events = int(d.get("n_events", 0))
        p.n_images = int(d.get("n_images", 0))
        p.meta = d.get("meta", {})
        return p


def collect_failure_events(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    image: Optional[np.ndarray] = None,
    fov: Optional[np.ndarray] = None,
    disc_xy: Optional[Tuple[float, float]] = None,
    fov_radius: Optional[float] = None,
    image_id: str = "",
):
    """False-negative gaps of one out-of-fold prediction, with their five features.

    A "gap" here is exactly one *repairable event* of
    :func:`src.eval.trr_fcr.count_repairable_events`: a run of reference
    centreline that the prediction dropped and whose two flanks sit in
    different predicted components.
    """
    import pandas as pd

    from src.eval.trr_fcr import count_repairable_events
    from src.rigr.scorer import _fov_geometry, eccentricity_zone, local_contrast

    g = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, g.shape)
    ev = count_repairable_events(pred_mask, g, fov=f)
    if not ev:
        return pd.DataFrame(columns=["image"] + PRIOR_AXES)

    radius = sk.local_radius(g & f)
    d_fov = ndi.distance_transform_edt(f)
    c_xy, r_fov = _fov_geometry(f, g.shape)
    disc_xy = disc_xy if disc_xy is not None else c_xy
    fov_radius = float(fov_radius if fov_radius is not None else r_fov)

    rows = []
    for e in ev:
        pix = np.asarray(e["gap_pixels"], dtype=np.int64)
        cen = (float(e["centroid"][0]), float(e["centroid"][1]))
        r_loc = float(np.mean(radius[pix[:, 0], pix[:, 1]]))
        rows.append(dict(
            image=image_id,
            gap_len=float(e["length"]),
            radius=max(r_loc, 0.25),
            contrast=(local_contrast(image, g, cen, max(r_loc, 1.0), f)
                      if image is not None else 1.0),
            ecc_zone=eccentricity_zone(cen, disc_xy, fov_radius),
            fov_edge=float(np.mean(d_fov[pix[:, 0], pix[:, 1]])) / max(fov_radius, 1.0),
        ))
    return pd.DataFrame(rows)


def estimate_failure_prior(
    pairs: Sequence[Dict[str, Any]],
    smoothing: float = 0.5,
    verbose: bool = False,
) -> Tuple[FailurePrior, Any]:
    """Fit ``pi`` from a list of out-of-fold ``(pred, gt, image, fov, id)`` dicts.

    ``pairs`` entries need the keys ``pred``, ``gt`` and optionally ``image``,
    ``fov``, ``disc_xy``, ``fov_radius``, ``image_id``.  The caller is
    responsible for the anti-leakage rules: only **source-domain training**
    images, and only predictions from a model that did *not* see that image
    (cross-fitting).  ``load_oof_pairs`` builds this list from a prediction
    directory once ``runs/seg/<ds>/<seed>/pred/`` exists.
    """
    import pandas as pd

    frames = []
    for i, p in enumerate(pairs):
        df = collect_failure_events(
            p["pred"], p["gt"], p.get("image"), p.get("fov"),
            p.get("disc_xy"), p.get("fov_radius"),
            str(p.get("image_id", i)),
        )
        if verbose:
            print("  %-20s %d failure gaps" % (p.get("image_id", i), len(df)))
        frames.append(df)
    events = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["image"] + PRIOR_AXES)
    prior = FailurePrior().fit(events, smoothing=smoothing)
    prior.meta["n_source_images"] = len(pairs)
    return prior, events


def load_oof_pairs(
    dataset: str,
    pred_dir: str,
    split: str = "train",
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build the ``pairs`` list for :func:`estimate_failure_prior` from disk.

    ``pred_dir`` is a ``runs/seg/<ds>/<seed>/pred`` directory (it must contain
    either ``manifest.csv`` or a ``mask/`` subdirectory named by image key).
    Returns ``[]`` -- with a warning -- when the directory does not exist yet,
    which is the state of the repository until stage S2 has run.
    """
    import glob
    import warnings

    from src.seg import data as segdata

    if not pred_dir or not os.path.isdir(pred_dir):
        warnings.warn("prediction directory %r does not exist; the failure-"
                      "conditioned layer is disabled (omega is forced to 0)"
                      % pred_dir, RuntimeWarning)
        return []

    recs = segdata.get_records(dataset)
    tr, va, te, _ = segdata.make_splits(recs)
    subset = {"train": tr, "val": va, "test": te, "all": recs}[split]
    by_key = {os.path.splitext(os.path.basename(str(r["image_path"])))[0]: r
              for r in subset}

    out: List[Dict[str, Any]] = []
    for mp in sorted(glob.glob(os.path.join(pred_dir, "mask", "*.png"))):
        key = os.path.splitext(os.path.basename(mp))[0]
        rec = by_key.get(key)
        if rec is None:
            continue
        img = segdata._imread_color(rec["image_path"])
        gt = segdata._imread_gray(rec["label_path"]) > 127
        fov = (segdata._imread_gray(rec["fov_path"]) > 127
               if rec.get("fov_path") and os.path.exists(str(rec["fov_path"]))
               else segdata.derive_fov(img) > 0)
        pred = segdata._imread_gray(mp) > 127
        out.append(dict(pred=pred, gt=gt, image=img, fov=fov, image_id=key))
        if limit and len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# locus sampling (uniform / failure-conditioned mixture)
# --------------------------------------------------------------------------
def sample_cut_loci(
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    n_cuts: int = 15,
    omega: float = 0.0,
    prior: Optional[FailurePrior] = None,
    image: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    length_factors: Sequence[float] = LENGTH_FACTORS,
    min_separation: float = 20.0,
    skel: Optional[np.ndarray] = None,
    disc_xy: Optional[Tuple[float, float]] = None,
    fov_radius: Optional[float] = None,
) -> List[Tuple[int, int, float]]:
    """Draw ``n_cuts`` severance loci: a mixture of uniform and pi-weighted.

    ``omega`` is the fraction drawn from the failure-conditioned layer (0 =
    uniform only, 1 = failure-conditioned only).  Loci are kept at least
    ``min_separation`` px apart so the cuts do not interact.  Returns
    ``[(row, col, length_factor), ...]``.
    """
    from src.rigr.scorer import _fov_geometry, eccentricity_zone, local_contrast

    rng = rng or np.random.default_rng(0)
    g = sk.as_bool(gt_mask)
    f = sk.fov_or_true(fov, g.shape)
    el = eligible_loci(g, f, skel=skel)
    coords = el["coord"]
    if coords.shape[0] == 0:
        return []

    n_fc = int(round(float(omega) * n_cuts)) if prior is not None else 0
    n_uni = int(n_cuts) - n_fc

    weights_fc = None
    lens_fc = None
    if n_fc > 0:
        d_fov = ndi.distance_transform_edt(f)
        c_xy, r_fov = _fov_geometry(f, g.shape)
        disc_xy = disc_xy if disc_xy is not None else c_xy
        fov_radius = float(fov_radius if fov_radius is not None else r_fov)
        rad = el["radius"]
        ecc = np.array([eccentricity_zone((r, c), disc_xy, fov_radius)
                        for r, c in coords])
        fe = d_fov[coords[:, 0], coords[:, 1]] / max(fov_radius, 1.0)
        if image is not None:
            con = np.array([local_contrast(image, g, (r, c), max(rr, 1.0), f)
                            for (r, c), rr in zip(coords, rad)])
        else:
            con = np.full(len(coords), 1.0)
        weights_fc = prior.locus_weights(rad, con, ecc, fe)
        lens_fc = prior.sample_gap_len(rad, con, ecc, fe, rng)

    chosen: List[Tuple[int, int, float]] = []
    taken = np.zeros((0, 2), dtype=float)

    def _try_take(i: int, lf: float) -> bool:
        p = coords[i].astype(float)
        if len(taken) and np.min(np.hypot(taken[:, 0] - p[0],
                                          taken[:, 1] - p[1])) < min_separation:
            return False
        chosen.append((int(coords[i, 0]), int(coords[i, 1]), float(lf)))
        return True

    def _draw(k: int, w, lengths):
        tries = 0
        while sum(1 for _ in chosen) < k and tries < 60 * max(k, 1):
            tries += 1
            i = int(rng.choice(len(coords), p=w) if w is not None
                    else rng.integers(0, len(coords)))
            if lengths is None:
                lf = float(rng.choice(np.asarray(length_factors, dtype=float)))
            else:
                lf = float(lengths[i]) / max(2.0 * el["radius"][i], _EPS)
                lf = float(np.clip(lf, 0.5, 8.0))
            if _try_take(i, lf):
                nonlocal taken
                taken = np.concatenate(
                    [taken, coords[i][None].astype(float)], axis=0)

    target = n_fc
    if n_fc > 0:
        _draw(target, weights_fc, lens_fc)
    _draw(len(chosen) + n_uni, None, None)
    return chosen


# --------------------------------------------------------------------------
# end-to-end training-pair construction
# --------------------------------------------------------------------------
def build_training_pairs(
    image: np.ndarray,
    gt_mask: np.ndarray,
    fov: Optional[np.ndarray],
    evidence_fn: Callable[..., Tuple[np.ndarray, np.ndarray]],
    n_cuts: int = 15,
    omega: float = 0.0,
    prior: Optional[FailurePrior] = None,
    rng: Optional[np.random.Generator] = None,
    prob_sigma: float = 2.0,
    cand_kwargs: Optional[dict] = None,
    astar_kwargs: Optional[dict] = None,
    image_id: str = "",
    verify_beta0: bool = True,
) -> Dict[str, Any]:
    """One image -> ``(X, y)`` for the pair scorer.

    ``evidence_fn(image, cut_mask, prob, fov) -> (V_I, q)`` supplies the
    appearance / orientation fields (the learned head, or ``frangi_evidence``
    with a uniform ``q`` for the analytic ablation).

    Returns a dict with ``X``, ``y``, ``table``, ``results``, ``cut_mask``,
    ``prob``, ``cuts`` (the severance record table) and ``label_records``.
    """
    from src.rigr import astar as astar_mod
    from src.rigr import candidates as cand_mod
    from src.rigr.scorer import candidate_features

    rng = rng or np.random.default_rng(0)
    f = sk.fov_or_true(fov, sk.as_bool(gt_mask).shape)
    loci = sample_cut_loci(gt_mask, f, n_cuts=n_cuts, omega=omega, prior=prior,
                           image=image, rng=rng)
    cut_mask, cuts = make_cut_mask(gt_mask, loci, fov=f, verify_beta0=verify_beta0)
    prob = synth_prob(cut_mask, sigma=prob_sigma)

    V, Q = evidence_fn(image, cut_mask, prob, f)
    out = cand_mod.generate_candidates(cut_mask, prob, f, **(cand_kwargs or {}))
    table = out["table"]

    results = []
    for row in table.to_dict("records"):
        gp = out["port_pixels"].get(int(row["node_j"])) if int(row["is_port"]) else None
        results.append(astar_mod.solve_candidate(row, V, Q, goal_pixels=gp,
                                                 **(astar_kwargs or {})))

    X = candidate_features(table, results, image, cut_mask, prob, f)
    y, recs = label_candidates(table, results, cut_mask, gt_mask, fov=f)
    return dict(X=X, y=y, table=table, results=results, cut_mask=cut_mask,
                prob=prob, cuts=cuts, label_records=recs, cand=out,
                V=V, Q=Q, image_id=image_id, loci=loci)


# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - synthetic demo
    from src.rigr.head import K_BINS

    H = W = 220
    gt = np.zeros((H, W), dtype=bool)
    rng = np.random.default_rng(0)
    for y0 in (40, 90, 140, 190):
        gt[y0 - 2:y0 + 2, 15:205] = True
    gt[40:190, 108:112] = True
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 1] = 180
    img[..., 1][gt] = 70
    fov = np.zeros((H, W), bool)
    fov[6:-6, 6:-6] = True

    el = eligible_loci(gt, fov)
    print("eligible loci        : %d" % len(el["coord"]))

    loci = sample_cut_loci(gt, fov, n_cuts=6, rng=rng)
    cut, cuts = make_cut_mask(gt, loci, fov)
    print("cuts applied         : %d, verified delta_beta0==1: %d"
          % (len(cuts), int(cuts["verified"].sum())))
    print(cuts[["locus", "radius", "length_factor", "length", "n_removed",
                "delta_beta0"]].to_string(index=False))

    def ev_fn(image, mask, prob, fov_):
        from src.rigr.head import frangi_evidence

        V = frangi_evidence(image, fov_)
        Q = np.full((K_BINS,) + mask.shape, 1.0 / K_BINS, np.float32)
        return V, Q

    d = build_training_pairs(img, gt, fov, ev_fn, n_cuts=6, rng=np.random.default_rng(1))
    print("candidates           : %d   positives: %d"
          % (len(d["table"]), int(d["y"].sum())))
    print("feature matrix       : %s" % (d["X"].shape,))

    # failure prior from a synthetic "baseline prediction"
    pred = gt.copy()
    pred[88:92, 60:70] = False
    pred[138:142, 150:158] = False
    prior, events = estimate_failure_prior(
        [dict(pred=pred, gt=gt, image=img, fov=fov, image_id="syn0")], verbose=True)
    print(events.round(3).to_string(index=False))
    loci_fc = sample_cut_loci(gt, fov, n_cuts=6, omega=1.0, prior=prior,
                              image=img, rng=np.random.default_rng(2))
    print("failure-conditioned loci:", [(r, c, round(l, 2)) for r, c, l in loci_fc])
    p = os.path.join(os.environ.get("TEMP", "/tmp"), "_rigr_prior_demo.json")
    prior.save(p)
    print("prior round-trip ok  :", np.allclose(FailurePrior.load(p).counts,
                                                prior.counts))
