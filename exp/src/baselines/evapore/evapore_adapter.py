"""EVAPORE baseline adapter -- their candidate generator and their classifier.

Upstream: ``oscarmorand/EVAPORE`` branch **shapeMI**, commit
``05af7d028bd6f3cf9772100f8b344e20cfc8783d``, **GPL-3.0**, cloned read-only to
``exp/third_party/EVAPORE`` (see :mod:`.evapore_upstream` for the three
Windows / torch-2.6 import obstacles and how they are worked around without
patching their tree).

Everything load-bearing is **theirs, imported**, not reimplemented:

* candidates -- ``img_to_graph`` -> ``OversampleNodesTransform`` ->
  ``get_query_edges`` -> ``EuclideanPathReconstructionMethod`` ->
  ``cut_mask_from_negative_edges_for_all``;
* labels -- ``get_combined_graph`` + ``get_positive_samples`` /
  ``get_negative_samples`` from ``path_neural_networks.data.compute_samples``;
* model -- ``UnetFeaturesGenerator(UNet)`` -> ``MultiScaleSquarePathSampling``
  (max aggregation) -> ``ConvMaxPoolingPathEncoder`` -> ``FCNPathClassifier``,
  trained with their ``WeightedBCEWithLogitsLoss``.

What we supply is the training *driver* (they train through PyTorch Lightning,
which is not installed here) and the glue to our datasets.

Pretrained checkpoints
----------------------
The repo **does** ship two git-lfs checkpoints, both real and both
FIVES-specific:

* ``checkpoints/FIVES/unet_pretrained/best-checkpoint-epoch=65-val_loss=0.0685.ckpt``
* ``checkpoints/FIVES/main_model_pretrained/best-pr-auc-checkpoint-epoch=03-val_pr_auc=0.9134.ckpt``

They are **not** used here, for two reasons.  (1) Proposal 4.2 discipline 9
requires every baseline to be retrained under our preprocessing, FOV policy,
split and inference rule -- importing their FIVES numbers would break the
comparison.  (2) They are PyTorch-Lightning 2.5 checkpoints whose
``hyper_parameters`` blob holds Albumentations objects and a custom enum, so
they need ``weights_only=False`` under torch >= 2.6 and their own Lightning
version to reconstruct.  ``--init_from_upstream`` can still warm-start the
feature U-Net from their weights for a sensitivity check.

Training
--------
    python -m src.baselines.evapore.evapore_adapter --dataset drive --gpu 1 \
        --epochs 20 --out runs/repair/ckpt/evapore_drive.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.topo import skeleton as sk
from src.eval.trr_fcr import CandidateEdge
from src.baselines.common import RepairOutput, as_bool, paint_edges
from src.baselines.evapore.evapore_upstream import (
    EVAPORE_BRANCH, EVAPORE_COMMIT, EVAPORE_LICENSE, EVAPORE_ROOT,
    ensure_upstream, generate_candidates, load_components,
)

__all__ = ["repair", "repair_detailed", "train", "EvaporePipeline", "main"]

# their notebook-06 inference defaults; max_dist=100 matches the shipped
# checkpoint's ``centerline_dir="euclidean_lt_100_clean_centerlines"``
_EVAPORE_AMP = False        # set from --amp in main(); see train()

DEFAULTS = dict(max_dist=100.0, oversampling_max_dist=50.0, n_closest=5,
                closing_radius=1, scales=(1, 3, 5), feat_channels=32,
                accept_p=0.5)

_CACHE: Dict[str, Any] = {}


# --------------------------------------------------------------------------
# their pipeline, assembled without Lightning
# --------------------------------------------------------------------------


class EvaporePipeline(nn.Module):
    """``features -> per-path sampling -> path encoder -> path classifier``.

    A plain-``nn.Module`` transcription of their ``ReducedPipelineLitModule``
    forward: every component is their class, only the Lightning scaffolding is
    dropped.  Paths are processed one at a time in a Python loop, exactly as
    upstream does (path lengths vary, so they are not batched).
    """

    def __init__(self, feat_channels: int = 32, scales: Sequence[int] = (1, 3, 5),
                 unet_features_start: int = 32, unet_layers: int = 4,
                 unet_kernel_size: int = 5, unet_norm: str = "instance",
                 encoder_layers: Sequence[Optional[int]] = (None, None, 256),
                 classifier_hidden: int = 2, dropout: float = 0.0,
                 in_channels: int = 3):
        super().__init__()
        C = load_components()
        self.cfg = dict(feat_channels=feat_channels, scales=list(scales),
                        unet_features_start=unet_features_start,
                        unet_layers=unet_layers, unet_kernel_size=unet_kernel_size,
                        unet_norm=unet_norm, encoder_layers=list(encoder_layers),
                        classifier_hidden=classifier_hidden, dropout=dropout,
                        in_channels=in_channels)

        unet = C["UNet"](input_channels=in_channels, num_classes=feat_channels,
                         num_layers=unet_layers, features_start=unet_features_start,
                         norm_op=unet_norm, kernel_size=unet_kernel_size)
        self.net = unet                       # the feature generator's body
        self.path_sampler = C["MultiScaleSquarePathSampling"](
            in_channels=feat_channels, square_sizes=list(scales),
            aggregation=C["SamplingMaxAggregation"]())
        self.path_encoder = C["PathEncoder"](
            in_channels=self.path_sampler.out_channels,
            hidden_layers=list(encoder_layers),
            skip_connection=False, residual_blocks=False)
        self.path_classifier = C["FCNPathClassifier"](
            in_channels=self.path_encoder.out_channels,
            n_hidden_layers=classifier_hidden, num_classes=1, dropout=dropout)

    def features(self, img: torch.Tensor) -> torch.Tensor:
        return self.net(img)

    def forward(self, img: torch.Tensor, paths: List[torch.Tensor]) -> torch.Tensor:
        """``img`` is ``(1, 3, H, W)``, ``paths`` a list of ``(L, 2)`` tensors."""
        fmap = self.features(img)
        logits = []
        for p in paths:
            f = self.path_sampler(fmap, p)
            e = self.path_encoder(f)
            logits.append(self.path_classifier(e))
        if not logits:
            return torch.zeros(0, device=img.device)
        return torch.cat(logits, dim=1).reshape(-1)


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def _prep_image(image: Optional[np.ndarray], shape) -> np.ndarray:
    """Fundus image as ``(3, H, W)`` float32 in [0, 1] (their U-Net wants RGB)."""
    if image is None:
        return np.zeros((3,) + tuple(shape), dtype=np.float32)
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    a = a[..., :3]
    mx = float(a.max())
    if mx > 1.0:
        a = a / 255.0
    return np.transpose(a, (2, 0, 1)).copy()


def save_ckpt(path: str, model: EvaporePipeline, meta: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(), cfg=model.cfg, meta=meta), p)


def load_ckpt(path: str, device: str = "cpu") -> EvaporePipeline:
    key = f"{os.path.abspath(path)}::{device}"
    if key in _CACHE:
        return _CACHE[key]
    blob = torch.load(path, map_location=device, weights_only=False)
    model = EvaporePipeline(**blob["cfg"]).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    _CACHE[key] = model
    return model


# --------------------------------------------------------------------------
# inference
# --------------------------------------------------------------------------


MIN_PATH_LEN = 2
"""Shortest path their sampler can consume.

``MultiScaleSquarePathSampling.forward`` starts with ``path.squeeze(dim=0)``,
so a ``(1, 2)`` single-pixel path collapses to ``(2,)`` and the offset
broadcast raises ``size of tensor a (2) must match tensor b (9)``.  Their own
sample generators never emit one, so upstream never trips over it; our
inference-time candidates can, when a gap is a single pixel wide.  Such
candidates are dropped rather than padded -- a one-pixel gap needs no repair.
"""


def _filter_paths(paths):
    """Indices of the paths their sampler can actually consume."""
    keep = []
    for i, p in enumerate(paths):
        a = np.asarray(p, dtype=np.int64).reshape(-1, 2)
        if a.shape[0] >= MIN_PATH_LEN:
            keep.append(i)
    return keep


def _paths_to_tensor(paths, device):
    """``(L, 2)`` int64 tensors.  Caller must have applied :func:`_filter_paths`."""
    return [torch.from_numpy(np.asarray(p, dtype=np.int64).reshape(-1, 2)).to(device)
            for p in paths]


def _score_paths(model, image, shape, paths, device) -> np.ndarray:
    """Sigmoid probability for each path under their classifier."""
    img_t = torch.from_numpy(_prep_image(image, shape))[None].to(device)
    path_t = _paths_to_tensor(paths, device)
    with torch.no_grad():
        logits = model(img_t, path_t)
        return torch.sigmoid(logits).cpu().numpy().reshape(-1)


# ---- variant A: end-to-end, entirely their rules --------------------------


def repair_e2e(
    image: Optional[np.ndarray],
    prob: Optional[np.ndarray],
    mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    ckpt: Optional[str] = None,
    device: str = "cpu",
    accept_p: float = DEFAULTS["accept_p"],
    max_dist: float = DEFAULTS["max_dist"],
    oversampling_max_dist: float = DEFAULTS["oversampling_max_dist"],
    n_closest: int = DEFAULTS["n_closest"],
    closing_radius: int = DEFAULTS["closing_radius"],
    **kw: Any,
) -> RepairOutput:
    """**EVAPORE end-to-end**: their candidates, their classifier, their acceptance.

    Everything is upstream's, transcribed from ``process_case`` and the
    inference cells of ``06_test_path_classification_model.ipynb``:

    * candidates -- ``img_to_graph`` -> ``OversampleNodesTransform`` ->
      ``get_query_edges`` -> ``EuclideanPathReconstructionMethod`` ->
      ``cut_mask_from_negative_edges_for_all``;
    * acceptance -- ``sigmoid(logit) > tau``, applied **independently per
      path**.  Upstream has *no* one-edge-per-endpoint arbitration and no
      matching: every path over threshold is drawn.  ``tau`` is their
      ``max_f1_threshold``, chosen on the validation split (see
      :func:`tune_threshold`); ``accept_p`` overrides it;
    * tube radius -- their ``SmallestRadiusReconstructionMethod``,
      ``min(radius_u, radius_v)`` over the two skan graph nodes.

    Only the FOV mask and our metric protocol are ours.
    """
    if ckpt is None:
        raise ValueError(
            "evapore_e2e needs --ckpt: train one first with "
            "`python -m src.baselines.evapore.evapore_adapter --dataset <ds> "
            "--gpu 1 --epochs 20 --out <ckpt.pt>`")
    model = load_ckpt(ckpt, device)
    tau = float(accept_p)

    m = as_bool(mask)
    f = sk.fov_or_true(fov, m.shape)
    m = m & f

    t0 = time.time()
    cand = generate_candidates(m, max_dist=max_dist,
                               oversampling_max_dist=oversampling_max_dist,
                               n_closest=n_closest, closing_radius=closing_radius)
    t_cand = time.time() - t0

    paths = cand["path_centerlines"]
    edges_uv = cand["edges"]
    node_r = cand.get("nodes_radius", {}) or {}

    info: Dict[str, Any] = dict(
        method="evapore_e2e", variant="e2e", n_candidates=len(paths),
        device=str(device), accept_p=tau, seconds_candidates=round(t_cand, 3),
        candidate_source="upstream", acceptance="upstream_per_path_threshold",
        radius_rule="upstream_smallest_node_radius",
        upstream=f"{EVAPORE_BRANCH}@{EVAPORE_COMMIT[:10]}",
        license=EVAPORE_LICENSE, ckpt=str(ckpt))

    if not paths:
        info.update(n_accepted=0, n_added_px=0, seconds_score=0.0,
                    n_dropped_short=0)
        return RepairOutput(mask=m, edges=[], info=info)

    keep = _filter_paths(paths)
    info["n_dropped_short"] = len(paths) - len(keep)
    if not keep:
        info.update(n_accepted=0, n_added_px=0, seconds_score=0.0)
        return RepairOutput(mask=m, edges=[], info=info)
    paths = [paths[i] for i in keep]
    edges_uv = [edges_uv[i] for i in keep] if edges_uv else edges_uv

    t0 = time.time()
    probs = _score_paths(model, image, m.shape, paths, device)
    info["seconds_score"] = round(time.time() - t0, 3)

    # their acceptance: independent per-path threshold, no arbitration
    edges: List[CandidateEdge] = []
    for k, p in enumerate(probs):
        if float(p) <= tau:
            continue
        uv = edges_uv[k] if edges_uv and k < len(edges_uv) else (None, None)
        u, v = (uv[0], uv[1]) if uv is not None and len(uv) == 2 else (None, None)
        pts = np.asarray(paths[k], dtype=np.int64).reshape(-1, 2)
        ru, rv = node_r.get(u), node_r.get(v)
        radius = (min(ru, rv) if (ru is not None and rv is not None)
                  else _fallback_radius(m, pts))
        edges.append(CandidateEdge(
            path=pts, radius=max(float(radius), 0.5), edge_id=f"eva_{u}-{v}",
            score=float(p),
            meta=dict(source="evapore_e2e", p=float(p), node_u=u, node_v=v)))

    out = paint_edges(m, edges, fov=f)
    info.update(n_accepted=len(edges), n_added_px=int(np.count_nonzero(out & ~m)),
                p_max=float(probs.max()), p_mean=float(probs.mean()))
    return RepairOutput(mask=out, edges=edges, info=info)


# ---- variant B: their scorer inside our harness ---------------------------


def repair_scorer(
    image: Optional[np.ndarray],
    prob: Optional[np.ndarray],
    mask: np.ndarray,
    fov: Optional[np.ndarray] = None,
    ckpt: Optional[str] = None,
    device: str = "cpu",
    accept_p: float = DEFAULTS["accept_p"],
    candidates: str = "geometric",
    **kw: Any,
) -> RepairOutput:
    """**EVAPORE scorer**: our candidates, their classifier, our acceptance.

    The controlled comparison against RiGR: the candidate set is exactly
    :func:`src.baselines.geometric_repair.propose_candidates` (same endpoint
    detection, same distance / tangent / radius gates, same straight-or-Bezier
    paths, same ``r_bar`` tube), and acceptance is our greedy
    one-edge-per-endpoint rule.  The **only** thing that changes relative to
    the geometric baseline is that the ordering comes from their learned path
    probability instead of the inverse distance.

    ``candidates`` selects the generator:

    ``"geometric"`` (default)
        :func:`src.baselines.geometric_repair.propose_candidates` -- endpoint
        pairs with straight / Bezier paths.  Isolates the scorer exactly
        against the geometric row.
    ``"rigr"``
        ``src.rigr.candidates.generate_candidates`` + the corridor-restricted
        lifted-state A* paths, i.e. **RiGR's own candidate set**.  Isolates the
        scorer against RiGR with candidate generation and path geometry held
        fixed, so the only difference is their path classifier in place of our
        calibrated pair scorer.

    ``kw`` beyond ``accept_p`` / ``candidates`` is forwarded to the generator
    (e.g. ``alpha_d_over_r``, ``tau_tangent``).
    """
    if ckpt is None:
        raise ValueError(
            "evapore_scorer needs --ckpt: train one first with "
            "`python -m src.baselines.evapore.evapore_adapter --dataset <ds> "
            "--gpu 1 --epochs 20 --out <ckpt.pt>`")
    model = load_ckpt(ckpt, device)
    tau = float(accept_p)

    t0 = time.time()
    if str(candidates).lower() == "rigr":
        cands, ep = _rigr_candidates(mask, prob, fov, **kw)
        cand_src = "ours(rigr.candidates + A*)"
    else:
        from src.baselines.geometric_repair import propose_candidates
        cands, ep, _cfg = propose_candidates(mask, fov, **kw)
        cand_src = "ours(geometric_repair.propose_candidates)"
    t_cand = time.time() - t0
    m = ep["mask_in_fov"]
    f = ep["fov_used"]
    coord = ep["coord"].astype(np.float64)
    n = int(coord.shape[0])

    info: Dict[str, Any] = dict(
        method="evapore_scorer", variant="scorer", n_candidates=len(cands),
        n_endpoints=n, device=str(device), accept_p=tau,
        seconds_candidates=round(t_cand, 3),
        candidate_source=cand_src,
        acceptance="ours_greedy_one_edge_per_endpoint",
        radius_rule="ours_r_bar",
        upstream=f"{EVAPORE_BRANCH}@{EVAPORE_COMMIT[:10]}",
        license=EVAPORE_LICENSE, ckpt=str(ckpt))

    if not cands:
        info.update(n_accepted=0, n_added_px=0, seconds_score=0.0,
                    n_dropped_short=0)
        return RepairOutput(mask=m, edges=[], info=info)

    paths = [c["path"] for c in cands]
    keep = _filter_paths(paths)
    info["n_dropped_short"] = len(paths) - len(keep)
    if not keep:
        info.update(n_accepted=0, n_added_px=0, seconds_score=0.0)
        return RepairOutput(mask=m, edges=[], info=info)
    cands = [cands[i] for i in keep]
    paths = [paths[i] for i in keep]

    t0 = time.time()
    probs = _score_paths(model, image, m.shape, paths, device)
    info["seconds_score"] = round(time.time() - t0, 3)

    # our acceptance: greedy by descending probability, one edge per endpoint
    used = np.zeros(n, dtype=bool)
    edges: List[CandidateEdge] = []
    for k in np.argsort(-probs):
        p = float(probs[k])
        if p <= tau:
            break
        c = cands[k]
        i, j = c["i"], c["j"]
        if used[i] or used[j]:
            continue
        used[i] = used[j] = True
        edges.append(CandidateEdge(
            path=c["path"],
            p0=(int(coord[i, 0]), int(coord[i, 1])),
            p1=(int(coord[j, 0]), int(coord[j, 1])),
            radius=float(c["r_bar"]), edge_id=f"evasc_{i}-{j}", score=p,
            meta=dict(source="evapore_scorer", p=p, shape=c["shape"],
                      dist=c["d"], r_bar=c["r_bar"], comp_a=c["comp_a"],
                      comp_b=c["comp_b"])))

    out = paint_edges(m, edges, fov=f)
    info.update(n_accepted=len(edges), n_added_px=int(np.count_nonzero(out & ~m)),
                p_max=float(probs.max()), p_mean=float(probs.mean()))
    return RepairOutput(mask=out, edges=edges, info=info)


def _rigr_candidates(mask, prob, fov, **kw):
    """RiGR's own candidates (+ A* paths) in ``propose_candidates`` format.

    Imported lazily: ``src.rigr`` is developed in parallel, and this baseline
    must stay runnable when that package is mid-edit.  Returns
    ``(candidates, endpoints)`` with the same keys ``repair_scorer`` consumes.
    """
    from src.rigr.candidates import generate_candidates as _gc

    res = _gc(mask, prob=prob, fov=fov, **kw)
    table = res["table"]
    m = as_bool(res.get("mask", mask))
    f = sk.fov_or_true(res.get("fov", fov), m.shape)

    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    coords, cands = [], []
    for r in rows:
        path = r.get("path")
        if path is None:
            continue
        path = np.asarray(path, dtype=np.int64).reshape(-1, 2)
        if path.shape[0] < MIN_PATH_LEN:
            continue
        i, j = len(coords), len(coords) + 1
        coords.append((int(path[0, 0]), int(path[0, 1])))
        coords.append((int(path[-1, 0]), int(path[-1, 1])))
        cands.append(dict(
            i=i, j=j, d=float(r.get("dist", np.linalg.norm(path[0] - path[-1]))),
            r_bar=float(r.get("r_bar", r.get("radius", 1.0))), path=path,
            shape="astar", cos_i=float(r.get("cos_i", np.nan)),
            cos_j=float(r.get("cos_j", np.nan)),
            ratio=float(r.get("radius_ratio", np.nan)),
            comp_a=int(r.get("comp_i", -1)), comp_b=int(r.get("comp_j", -1))))

    ep = dict(coord=np.asarray(coords, dtype=np.int64).reshape(-1, 2),
              mask_in_fov=m, fov_used=f, n_pairs_examined=len(rows))
    return cands, ep


def _fallback_radius(mask: np.ndarray, pts: np.ndarray) -> float:
    """Mean local radius at the two path ends, when a graph node radius is absent."""
    rm = sk.local_radius(mask)
    h, w = mask.shape
    vals = [float(rm[min(max(int(a), 0), h - 1), min(max(int(b), 0), w - 1)])
            for a, b in (pts[0], pts[-1])]
    vals = [v for v in vals if v > 0]
    return float(np.mean(vals)) if vals else 1.0


# ---- dispatch -------------------------------------------------------------


def repair_detailed(image, prob, mask, fov=None, variant: str = "e2e", **kw):
    """Dispatch to :func:`repair_e2e` (default) or :func:`repair_scorer`."""
    v = str(variant).lower()
    if v in ("e2e", "end_to_end", "upstream"):
        return repair_e2e(image, prob, mask, fov, **kw)
    if v in ("scorer", "our_candidates", "harness"):
        return repair_scorer(image, prob, mask, fov, **kw)
    raise ValueError(f"unknown evapore variant {variant!r}; use 'e2e' or 'scorer'")


def repair(image, prob, mask, fov=None, **kw) -> np.ndarray:
    return repair_detailed(image, prob, mask, fov, **kw).mask


def tune_threshold(ckpt: str, dataset: str, device: str = "cpu",
                   split: str = "train", skip: int = 0,
                   limit: Optional[int] = None, n_cuts: int = 20,
                   seed: int = 0) -> float:
    """Their ``max_f1_threshold``: the tau maximising F1 on held-out paths.

    Reproduces notebook 06 cells 25-27 (``BinaryPrecisionRecallCurve`` ->
    ``argmax F1`` -> ``set_inference_threshold``) using their label definition,
    so ``evapore_e2e`` can be run at upstream's own operating point instead of
    a hard-coded 0.5.
    """
    from sklearn.metrics import precision_recall_curve

    model = load_ckpt(ckpt, device)
    data = build_samples(dataset, split, n_cuts=n_cuts, seed=seed,
                         limit=limit, skip=skip)
    ps, ys = [], []
    for d in data:
        keep = _filter_paths(d["paths"])
        if not keep:
            continue
        ps.append(_score_paths(model, d["image"].transpose(1, 2, 0),
                               d["image"].shape[1:],
                               [d["paths"][i] for i in keep], device))
        ys.append(d["labels"][keep])
    if not ps:
        return 0.5
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    if not (0 < y.sum() < y.size):
        return 0.5
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    k = int(np.argmax(f1[:-1])) if len(thr) else 0
    return float(thr[k]) if len(thr) else 0.5


# --------------------------------------------------------------------------
# training data, built with THEIR label definition
# --------------------------------------------------------------------------


def build_samples(
    dataset: str, split: str = "train", n_cuts: int = 20, seed: int = 0,
    max_dist: float = DEFAULTS["max_dist"],
    oversampling_max_dist: float = DEFAULTS["oversampling_max_dist"],
    n_closest: int = DEFAULTS["n_closest"], limit: Optional[int] = None,
    skip: int = 0, pred_dir: Optional[str] = None, extra_cuts: bool = True,
) -> List[Dict[str, Any]]:
    """One record per image: ``(image, paths, labels)``.

    Labels are produced by **their** ``get_positive_samples`` /
    ``get_negative_samples`` (via ``get_combined_graph``), so a positive is a
    straight line across a GT vessel segment the prediction missed, and a
    negative is a nearest-endpoint query edge that does not lie on the GT.

    Their pipeline gets ``pred`` from a trained U-Net (their notebooks 02-03).
    While S2 is still training we use the project's synthetic capsule
    severances instead -- the same imperfect masks the other two baselines are
    developed against.  Point ``--pred_dir`` at a real S2 prediction directory
    to train on genuine segmenter failures.
    """
    ensure_upstream()
    import importlib

    cs = importlib.import_module("path_neural_networks.data.compute_samples")
    gps = importlib.import_module("graph.graph_pred_state")
    u = ensure_upstream()

    from src.data.datasets import load_dataset, read_binary, read_image
    from src.baselines.run_baseline import pred_key as _pred_key
    from src.baselines.synth import training_inputs

    recs = load_dataset(dataset, split=split)
    if skip:
        recs = recs[int(skip):]
    if limit:
        recs = recs[: int(limit)]
    out: List[Dict[str, Any]] = []

    for k, r in enumerate(recs):
        gt = read_binary(r["label_path"])
        fov = read_binary(r["fov_path"]) if r.get("fov_path") else np.ones(gt.shape, bool)
        image = np.asarray(read_image(r["image_path"]))
        area_scale = float(gt.shape[0] * gt.shape[1]) / (584.0 * 565.0)
        n = max(4, int(round(n_cuts * max(1.0, area_scale) ** 0.5)))
        pred, _p, _c, src_kind = training_inputs(
            gt, fov, image_id=[_pred_key(r), r["image_id"]],
            pred_dir=pred_dir, n_cuts=n,
            seed=seed + k, extra_cuts=extra_cuts)

        gt_np = as_bool(gt) & fov
        pred_np = as_bool(pred) & fov
        if gt_np.sum() == 0 or pred_np.sum() == 0:
            continue
        try:
            combined = gps.get_combined_graph(gt_np, pred_np)
            gw = u["OversampleNodesTransform"](oversampling_max_dist,
                                               remove_original_edges=True)(
                u["GraphWrapper"](combined))
            G = gw.get_graph()
            in_pred = gw.in_pred_graph
            pos, pos_y = cs.get_positive_samples(G, pred_np, max_dist)
            neg, neg_y = cs.get_negative_samples(G, in_pred, pred_np, gt_np,
                                                 max_dist, n_closest)
        except Exception as exc:                       # noqa: BLE001
            print(f"  [{k+1}/{len(recs)}] {r['image_id']}: sample generation "
                  f"failed ({type(exc).__name__}: {exc}), skipped", flush=True)
            continue

        paths = list(neg) + list(pos)
        labels = list(neg_y) + list(pos_y)
        paths = [np.asarray(p, dtype=np.int64).reshape(-1, 2) for p in paths]
        keep = _filter_paths(paths)
        paths = [paths[i] for i in keep]
        labels = [float(labels[i]) for i in keep]
        if not paths:
            continue
        out.append(dict(image_id=r["image_id"], image=_prep_image(image, gt.shape),
                        paths=paths, labels=np.asarray(labels, dtype=np.float32)))
        print(f"  [{k+1}/{len(recs)}] {r['image_id']}: {src_kind}, "
              f"{int(sum(labels))} pos / {len(labels) - int(sum(labels))} neg",
              flush=True)
    return out


def train(
    dataset: str, out: str, device: str = "cpu", epochs: int = 20,
    lr: float = 3e-4, weight_decay: float = 1e-4, seed: int = 0,
    n_cuts: int = 20, limit: Optional[int] = None, skip: int = 0,
    max_paths_per_image: int = 256, feat_channels: int = 32,
    scales: Sequence[int] = DEFAULTS["scales"], log_every: int = 1,
    max_dist: float = DEFAULTS["max_dist"], val_frac: float = 0.2,
    pred_dir: Optional[str] = None, extra_cuts: bool = True,
    patience: Optional[int] = None, monitor: str = "val_auc",
) -> str:
    """Train their path classifier on our data with their recipe.

    AdamW(lr=3e-4, wd=1e-4) + CosineAnnealingLR, their
    ``WeightedBCEWithLogitsLoss`` with ``pos_weight = neg/pos``, one image per
    step, gradient-norm clipping at 1.0 -- upstream's ``configure_optimizers``
    and Trainer settings.  ``epochs`` counts passes over the training images.

    ``patience`` (optional): stop once ``monitor`` (``val_auc`` by default,
    falling back to ``val_ap`` if AUC is undefined for a given epoch, e.g. a
    val split with only one class present) has not improved for that many
    epochs; the **best** epoch's weights are the ones saved, not the
    final-at-stop weights.  Disabled (full ``epochs`` run) when ``patience``
    is ``None`` or there is no validation split.  Either way the checkpoint
    meta records ``early_stopping`` with ``enabled``/``patience``/``monitor``/
    ``best_epoch``/``stopped_epoch``/``stopped_early``, so a reader never has
    to guess which regime produced the weights.
    """
    ensure_upstream()
    import importlib

    # AMP is OFF by default so drive / chasedb1 / fives stay byte-for-byte
    # reproducible.  It exists only because HRF at its native 3504x2336
    # does not fit a 16 GB T4 in fp32: the UNet decoder concat asks for
    # 1.95 GiB on top of a 14.28 GiB resident model+activations, i.e. it
    # is short by ~700 MB, ALONE on the card.  See DECISIONS.md.
    amp = bool(globals().get("_EVAPORE_AMP", False))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    print(f"[evapore] building samples from {dataset}/train "
          f"(their label definition) ...", flush=True)
    data = build_samples(dataset, "train", n_cuts=n_cuts, seed=seed,
                         max_dist=max_dist, limit=limit, skip=skip,
                         pred_dir=pred_dir, extra_cuts=extra_cuts)
    if not data:
        raise SystemExit("no usable training samples")

    rng.shuffle(data)
    n_val = max(1, int(round(val_frac * len(data)))) if len(data) > 2 else 0
    val, tr = data[:n_val], data[n_val:]
    n_pos = float(sum(float(d["labels"].sum()) for d in tr))
    n_neg = float(sum(len(d["labels"]) for d in tr) - n_pos)
    print(f"[evapore] train {len(tr)} imgs / val {len(val)}  "
          f"paths: {int(n_pos)} pos, {int(n_neg)} neg", flush=True)

    model = EvaporePipeline(feat_channels=feat_channels, scales=scales).to(dev)

    wbce = importlib.import_module(
        "path_neural_networks.models.losses.weighted_bce_with_logits_loss")
    ratio = [n_neg / max(n_neg + n_pos, 1.0), n_pos / max(n_neg + n_pos, 1.0)]
    loss_fn = wbce.WeightedBCEWithLogitsLoss(classes_ratio=ratio).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = bool(amp) and dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print("[evapore] AMP (fp16 autocast + GradScaler) ENABLED -- forward/"
              "backward in fp16, master weights and the optimizer stay fp32",
              flush=True)

    t0 = time.time()
    hist: List[Dict[str, float]] = []
    es_enabled = bool(patience) and bool(val)
    best_metric = -np.inf
    best_epoch = 0
    best_state = None
    stopped_epoch = int(epochs)
    stopped_early = False
    for ep in range(1, int(epochs) + 1):
        model.train()
        losses = []
        for d in tr:
            paths, y = d["paths"], d["labels"]
            if len(paths) > max_paths_per_image:      # bound memory per step
                sel = rng.choice(len(paths), max_paths_per_image, replace=False)
                paths = [paths[i] for i in sel]
                y = y[sel]
            img = torch.from_numpy(d["image"])[None].to(dev)
            pt = _paths_to_tensor(paths, dev)
            target = torch.from_numpy(np.asarray(y, dtype=np.float32)).to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(img, pt)
                loss = loss_fn(logits.float(), target)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        sched.step()

        row = dict(epoch=ep, train_loss=float(np.mean(losses)) if losses else float("nan"))
        if val:
            model.eval()
            ps, ys = [], []
            with torch.no_grad():
                for d in val:
                    img = torch.from_numpy(d["image"])[None].to(dev)
                    pt = _paths_to_tensor(d["paths"], dev)
                    with torch.autocast("cuda", dtype=torch.float16,
                                        enabled=use_amp):
                        _lg = model(img, pt)
                    ps.append(torch.sigmoid(_lg.float()).cpu().numpy())
                    ys.append(d["labels"])
            p = np.concatenate(ps) if ps else np.zeros(0)
            yv = np.concatenate(ys) if ys else np.zeros(0)
            if p.size and 0 < yv.sum() < yv.size:
                acc = float(((p > 0.5) == (yv > 0.5)).mean())
                from sklearn.metrics import average_precision_score, roc_auc_score
                row.update(val_acc=acc,
                           val_ap=float(average_precision_score(yv, p)),
                           val_auc=float(roc_auc_score(yv, p)))
        hist.append(row)
        if ep % log_every == 0 or ep == 1:
            extra = "".join(f"  {k} {v:.4f}" for k, v in row.items()
                            if k.startswith("val_"))
            print(f"  ep {ep:3d}/{epochs}  loss {row['train_loss']:.4f}{extra}"
                  f"  {time.time()-t0:6.1f}s", flush=True)

        if es_enabled:
            m = row.get(monitor)
            if m is None:                       # AUC undefined this epoch
                m = row.get("val_ap")
            if m is not None and m > best_metric:
                best_metric, best_epoch = float(m), ep
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
            elif ep - best_epoch >= int(patience):
                stopped_epoch, stopped_early = ep, True
                print(f"[evapore] early stop at ep {ep}: no {monitor} "
                      f"improvement for {patience} epochs (best={best_metric:.4f} "
                      f"@ep{best_epoch})", flush=True)
                break

    if es_enabled and best_state is not None:
        model.load_state_dict(best_state)

    meta = dict(dataset=dataset, epochs=int(epochs), lr=float(lr), seed=int(seed),
                n_cuts=int(n_cuts), device=str(device), history=hist,
                n_train_images=len(tr), n_val_images=len(val), skip=int(skip),
                pred_dir=str(pred_dir or ""), extra_cuts=bool(extra_cuts),
                train_input=("oof_pred" + ("+uniform_cuts" if extra_cuts else "")
                             if pred_dir else "gt+uniform_cuts"),
                classes_ratio=ratio, max_dist=float(max_dist),
                seconds=round(time.time() - t0, 1),
                early_stopping=dict(enabled=bool(es_enabled),
                                    patience=int(patience) if patience else None,
                                    monitor=monitor,
                                    best_epoch=int(best_epoch) if es_enabled else None,
                                    best_metric=(float(best_metric)
                                                if es_enabled and best_epoch else None),
                                    stopped_epoch=int(stopped_epoch),
                                    stopped_early=bool(stopped_early)),
                upstream=f"oscarmorand/EVAPORE {EVAPORE_BRANCH}@{EVAPORE_COMMIT}",
                license=EVAPORE_LICENSE)
    save_ckpt(out, model, meta)
    print(f"[evapore] saved {out}  ({meta['seconds']}s"
          + (f", early-stopped ep{stopped_epoch} best ep{best_epoch} "
             f"{monitor}={best_metric:.4f}" if stopped_early else "") + ")",
          flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.baselines.evapore.evapore_adapter",
        description="Train the EVAPORE path classifier on our data.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpu", default="cpu", help="GPU index (e.g. 1), or 'cpu'")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_cuts", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip", type=int, default=0,
                    help="skip the first N training images (holds out a smoke-test subset)")
    ap.add_argument("--max_paths_per_image", type=int, default=256)
    ap.add_argument("--max_dist", type=float, default=DEFAULTS["max_dist"])
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--pred_dir", default=None,
                    help="out-of-fold prediction dir (runs/seg_oof/<ds>/pred): "
                         "their pipeline expects a real U-Net prediction, and "
                         "this is the S4 protocol for every learned baseline")
    ap.add_argument("--no_extra_cuts", action="store_true",
                    help="with --pred_dir, do NOT add the uniform capsule "
                         "severances on top of the predicted mask")
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stop after this many epochs without a "
                         "--monitor improvement on the validation split "
                         "(default: off, run the full --epochs)")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast + GradScaler.  OFF by default so the "
                         "existing checkpoints stay reproducible; needed only "
                         "for HRF, which does not fit a 16 GB card in fp32.")
    ap.add_argument("--monitor", default="val_auc", choices=["val_auc", "val_ap"],
                    help="validation metric early stopping watches")
    a = ap.parse_args(argv)

    device = "cpu" if str(a.gpu).lower() in ("cpu", "-1", "") else f"cuda:{a.gpu}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[evapore] CUDA unavailable, falling back to CPU", flush=True)
        device = "cpu"

    global _EVAPORE_AMP
    _EVAPORE_AMP = bool(a.amp)
    train(dataset=a.dataset, out=a.out, device=device, epochs=a.epochs, lr=a.lr,
          seed=a.seed, n_cuts=a.n_cuts, limit=a.limit, skip=a.skip,
          max_paths_per_image=a.max_paths_per_image, max_dist=a.max_dist,
          log_every=a.log_every, pred_dir=a.pred_dir,
          extra_cuts=not a.no_extra_cuts, patience=a.patience, monitor=a.monitor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
