"""End-to-end RiGR CLI (plan stage S4, proposal v3 sections 3.2 and 4.2).

::

    python -m src.rigr.run_rigr --dataset drive --seed 0 --mode risk \
        --head_ckpt runs/rigr_head/drive/seed0/best.pt \
        --pred_dir  runs/seg/drive/seed0/pred \
        --out       runs/rigr/drive/seed0 [--sweep]

Per test image it produces the repaired mask and one CSV row carrying

* pixel + topology metrics **before and after** repair
  (``src.topo.metrics.evaluate_all``: Dice/F1, IoU, AUC, PR-AUC, clDice,
  Betti errors, BCS, Junc-F1);
* ``TRR_recall`` / ``TRR_precision`` / ``FCR`` on the accepted edge set
  (``src.eval.trr_fcr``, the three-part true-repair criterion);
* biomarker errors against the reference (``src.bio.biomarkers.compute_all``),
  reduced to the **macro-MAE over the four primary biomarkers**, each
  non-dimensionalised by the per-dataset robust scale
  ``sigma = 1.4826 MAD`` of section 3.1.2.

``--sweep`` re-runs the *selection stage only* (candidates, A* and the
calibrated ``p_e`` are computed once) over a grid of ``lambda`` and of the
acceptance threshold, giving the FCR-TRR and FCR-biomarker-benefit operating
curves of Fig.3.

Segmentation predictions are read from ``--pred_dir``
(``prob/<key>.npy`` + ``mask/<key>.png``).  Until stage S2 has produced them,
``--synthetic_pred`` builds ``(P, M_hat)`` from the reference mask by random
capsule severances so the pipeline can be exercised; every output row is then
stamped ``pred_source=synthetic`` and must not be reported.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.eval.biomarker_eval import bio_columns as _bio_columns
from src.eval.biomarker_eval import biomarker_row  # noqa: F401  (re-export)
from src.eval.biomarker_eval import load_gt_scales, macro_mae, pipes_for
from src.topo import skeleton as sk

PRIMARY = ("FD", "tortuosity", "density", "total_length")


# --------------------------------------------------------------------------
# evidence providers
# --------------------------------------------------------------------------
def make_evidence_fn(
    head_ckpt: Optional[str] = None,
    gpu: int = 0,
    evidence: str = "auto",
    patch: Optional[int] = None,
    k_bins: int = 16,
    orientation: str = "learned",
) -> Tuple[Callable[..., Tuple[np.ndarray, np.ndarray]], str]:
    """Return ``(fn(image, mask, prob, fov) -> (V_I, q), name)``.

    ``evidence='head'`` uses the learned micro head, ``'frangi'`` the analytic
    Frangi vesselness with a **uniform** ``q`` (the Tab.3 ablation), and
    ``'auto'`` picks the head when a checkpoint is given.

    ``orientation='uniform'`` replaces ``q(theta|x)`` by the uniform
    distribution while keeping the chosen evidence term -- the Tab.3
    ``appearance_only`` / ``+orientation`` contrast, which must vary the
    orientation prior alone.  ``'learned'`` (the default) changes nothing.
    """
    from src.rigr.head import frangi_evidence

    if orientation not in ("learned", "uniform"):
        raise ValueError("orientation must be 'learned' or 'uniform'")

    use_head = (evidence == "head") or (evidence == "auto" and head_ckpt)
    if not use_head:
        def _fn(image, mask, prob, fov):
            V = frangi_evidence(image, fov)
            Q = np.full((k_bins,) + np.asarray(mask).shape, 1.0 / k_bins, np.float32)
            return V, Q
        return _fn, "frangi+uniform_q"

    import torch

    from src.rigr.head import load_head, predict_head

    device = torch.device("cuda:%d" % int(gpu) if torch.cuda.is_available() else "cpu")
    model, ck = load_head(head_ckpt, device)
    p = int(patch or ck.get("config", {}).get("patch", 512))

    def _fn(image, mask, prob, fov):
        return predict_head(image, prob, head_ckpt, gpu=gpu, fov=fov, patch=p,
                            model=model, device=device)

    src = ck.get("config", {}).get("prob_source", "?")
    name = "head(%s,P=%s)" % (os.path.basename(str(head_ckpt)), src)

    if orientation == "uniform":
        def _fn_uni(image, mask, prob, fov):
            V, Q = _fn(image, mask, prob, fov)
            K = int(np.asarray(Q).shape[0])
            U = np.full((K,) + tuple(np.asarray(V).shape[:2]), 1.0 / K, np.float32)
            return V, U
        return _fn_uni, name + "+uniform_q"
    return _fn, name


# --------------------------------------------------------------------------
# data access
# --------------------------------------------------------------------------
def load_case(rec: dict, pred_dir: Optional[str], synthetic: bool = False,
              rng: Optional[np.random.Generator] = None, n_cuts: int = 15
              ) -> Dict[str, Any]:
    """Load ``(image, gt, fov, P, M_hat)`` for one record."""
    from src.seg import data as segdata

    key = os.path.splitext(os.path.basename(str(rec["image_path"])))[0]
    img = segdata._imread_color(rec["image_path"])
    gt = segdata._imread_gray(rec["label_path"]) > 127
    if rec.get("fov_path") and os.path.exists(str(rec["fov_path"])):
        fov = segdata._imread_gray(rec["fov_path"]) > 127
    else:
        fov = segdata.derive_fov(img) > 0

    prob = None
    mask = None
    if pred_dir:
        pp = os.path.join(pred_dir, "prob", key + ".npy")
        mp = os.path.join(pred_dir, "mask", key + ".png")
        if os.path.exists(pp):
            prob = np.clip(np.load(pp).astype(np.float32), 0.0, 1.0)
        if os.path.exists(mp):
            mask = segdata._imread_gray(mp) > 127
    source = "pred_dir"
    if prob is None or mask is None:
        if not synthetic:
            raise FileNotFoundError(
                "no prediction for %r under %r; pass --synthetic_pred to run the "
                "pipeline on severed reference masks instead" % (key, pred_dir))
        from src.rigr.synth_cuts import make_cut_mask, sample_cut_loci, synth_prob

        rng = rng or np.random.default_rng(0)
        loci = sample_cut_loci(gt, fov, n_cuts=n_cuts, rng=rng)
        mask, _ = make_cut_mask(gt, loci, fov=fov, verify_beta0=False)
        prob = synth_prob(mask, sigma=2.0)
        source = "synthetic"
    return dict(key=key, image=img, gt=gt & fov, fov=fov,
                prob=prob, mask=np.asarray(mask, bool) & fov, source=source,
                subject_id=str(rec.get("subject_id", "")))


# --------------------------------------------------------------------------
# biomarkers
#
# All biomarker computation (the ``src.bio.biomarkers`` call, the primary-
# biomarker macro-MAE, and the single source of sigma) is factored into
# ``src.eval.biomarker_eval`` (imported at module top) so
# ``src.baselines.run_baseline`` computes the exact same numbers; see that
# module's docstring for the native-resolution / single-sigma contract.


# --------------------------------------------------------------------------
# per-image pipeline
# --------------------------------------------------------------------------
def _sweep_work_ctx(args, ds, case):
    """Down-scaled (image, fov, disc) + scale factor for work-scale sweeps.

    Returns ``None`` when the sweep is to be measured natively (the default,
    and always for DRIVE / CHASE_DB1, whose ``resize_longest`` is ``None``).

    Why this exists: at HRF's 3504x2336 one ``skan`` biomarker call dominates a
    sweep cell, and the sweep has 16 of them per image.  C1, ``build_data`` and
    the deployed ``R_false`` table already measure counterfactual harm at the
    working resolution (longest side 1536) and convert length-like biomarkers
    back to native units with ``1/s``; doing the same here keeps Fig.3's
    biomarker-benefit axis on the same scale as everything else it is compared
    against, at a fraction of the cost.  DECISIONS.md 2026-09-05 23:20.
    """
    if getattr(args, "sweep_scale", "native") != "work":
        return None
    from src.rigr.build_data import _to_work_scale
    from src.seg import data as segdata

    longest = segdata.dataset_cfg(ds).get("resize_longest")
    if not longest:
        return None
    h, w = case["mask"].shape[:2]
    if max(h, w) == int(longest):
        return None
    img_w = _to_work_scale(case["image"], longest, False)
    fov_w = _to_work_scale(case["fov"], longest, True)
    s_work = float(max(img_w.shape[:2])) / float(max(h, w))
    return dict(longest=int(longest), image=img_w, fov=fov_w,
                scale=s_work, disc=None, _disc_done=False)


def _bio_at_work_scale(ctx, mask, pipelines, fd_rotations):
    """``biomarker_row`` on the down-scaled mask, converted to NATIVE units.

    Only length-like biomarkers carry a unit (``x 1/s``); FD, density and
    tortuosity are dimensionless and are passed through unchanged -- which is
    exact for density and only *approximately* scale-invariant for FD and
    tortuosity on a finite grid (the same stated assumption as DECISIONS.md
    2026-09-03 10:45).
    """
    from src.c1.stats_c1 import native_factor
    from src.eval.biomarker_eval import auto_disc as _auto_disc
    from src.eval.biomarker_eval import biomarker_row
    from src.rigr.build_data import _to_work_scale

    m_w = _to_work_scale(mask, ctx["longest"], True)
    if not ctx["_disc_done"]:
        ctx["disc"] = _auto_disc(ctx["image"], ctx["fov"], vessel_mask=m_w)
        ctx["_disc_done"] = True
    raw = biomarker_row(m_w, ctx["fov"], ctx["image"], disc=ctx["disc"],
                        pipelines=pipelines, fd_rotations=fd_rotations)
    return {k: (v * native_factor(k, ctx["scale"])
                if isinstance(v, (int, float)) else v)
            for k, v in raw.items()}


def auto_disc(case: Dict[str, Any]):
    """Automatic optic-disc estimate (never an annotation).  None on failure.

    Thin wrapper of ``src.eval.biomarker_eval.auto_disc`` (shared with
    ``src.baselines.run_baseline``) around this module's ``case`` dict.
    """
    from src.eval.biomarker_eval import auto_disc as _auto_disc

    return _auto_disc(case["image"], case["fov"], vessel_mask=case["mask"])


def prepare_image(case: Dict[str, Any], evidence_fn, cand_kwargs=None,
                  astar_kwargs=None, disc=None) -> Dict[str, Any]:
    """Candidates + A* + features for one image (everything before the utility)."""
    from src.rigr import astar as astar_mod
    from src.rigr import candidates as cand_mod
    from src.rigr.scorer import candidate_features

    t0 = time.perf_counter()
    V, Q = evidence_fn(case["image"], case["mask"], case["prob"], case["fov"])
    t_ev = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = cand_mod.generate_candidates(case["mask"], case["prob"], case["fov"],
                                       **(cand_kwargs or {}))
    t_cand = time.perf_counter() - t0

    table = out["table"]
    t0 = time.perf_counter()
    results = []
    for row in table.to_dict("records"):
        gp = out["port_pixels"].get(int(row["node_j"])) if int(row["is_port"]) else None
        results.append(astar_mod.solve_candidate(row, V, Q, goal_pixels=gp,
                                                 **(astar_kwargs or {})))
    t_astar = time.perf_counter() - t0

    disc_xy = (float(disc.cx), float(disc.cy)) if disc is not None else None
    X = candidate_features(table, results, case["image"], case["mask"],
                           case["prob"], case["fov"], disc_xy=disc_xy)
    return dict(V=V, Q=Q, cand=out, table=table, results=results, X=X, disc=disc,
                t_evidence=t_ev, t_candidates=t_cand, t_astar=t_astar,
                t_astar_per_cand=(t_astar / max(len(table), 1)))


def evaluate_repair_light(case, repaired, edges) -> Dict[str, float]:
    """Cheap metric set for the sweep: clDice, Dice, beta0 and TRR / FCR.

    The full ``evaluate_all`` (BCS, Junc-F1, Betti matching, PR-AUC) costs
    seconds per call; the sweep evaluates every (lambda, tau) cell of every
    image, so it uses this subset and the full set is reserved for the single
    operating point written to ``per_image.csv`` (``--sweep_full`` overrides).
    """
    from src.eval.trr_fcr import trr_fcr
    from src.topo.metrics import cldice, dice

    gt, fov = case["gt"], case["fov"]
    row = dict(
        before_cldice=cldice(case["mask"], gt, fov),
        after_cldice=cldice(repaired, gt, fov),
        before_dice=dice(case["mask"], gt, fov),
        after_dice=dice(repaired, gt, fov),
        after_beta0=float(sk.betti_numbers(repaired, fov)[0]),
    )
    row["delta_cldice"] = row["after_cldice"] - row["before_cldice"]
    tf = trr_fcr(edges, case["mask"], gt, fov=fov)
    row.update({k: v for k, v in tf.items() if not isinstance(v, list)})
    return row


def evaluate_repair(case, prep, repaired, edges, return_edge_records: bool = False
                    ) -> Dict[str, float]:
    """Full before/after metric set + TRR/FCR.

    ``return_edge_records=True`` also asks ``trr_fcr`` for its per-accepted-
    edge classification records (``is_true_repair`` per edge, keyed by
    ``edge_id`` = ``cand_id``) instead of discarding them -- schema item 6:
    this label already existed inside ``evaluate_repair``/``trr_fcr`` and was
    simply never surfaced.  Returns ``(row, edge_records)`` in that case,
    ``row`` alone otherwise.
    """
    from src.eval.trr_fcr import trr_fcr
    from src.topo.metrics import evaluate_all

    gt, fov, prob = case["gt"], case["fov"], case["prob"]
    before = evaluate_all(prob, case["mask"], gt, fov)
    after = evaluate_all(prob, repaired, gt, fov)
    tf = trr_fcr(edges, case["mask"], gt, fov=fov, return_details=return_edge_records)
    row: Dict[str, float] = {}
    for k, v in before.items():
        row["before_" + k] = v
    for k, v in after.items():
        row["after_" + k] = v
    for k in ("dice", "cldice", "bcs", "junction_f1", "betti0_error",
              "betti1_error", "n_cc_pred"):
        if k in before and k in after:
            row["delta_" + k] = float(after[k] - before[k])
    row.update({k: v for k, v in tf.items()
               if k not in ("records", "events", "matches")})
    if return_edge_records:
        return row, list(tf.get("records", []))
    return row


def build_edge_table(prep: Dict[str, Any], out: Dict[str, Any],
                     edge_records: Optional[Sequence[Dict[str, Any]]] = None
                     ) -> "pd.DataFrame":
    """Schema item 6: one row per **candidate** (accepted and rejected),
    written to ``<out>/edges/<image>.csv``:

        cand_id         int    matches ``cand_<image>.csv``'s ``cand_id``
        accepted        bool   selected by the max-weight matching (and not
                               vetoed)
        is_true_repair  bool   ``src.eval.trr_fcr.classify_edges`` label,
                               ``False`` for every non-accepted candidate
                               (the criterion is only evaluated on ``A``)
        dangerous       bool   a *rejected* candidate whose false-connection
                               risk ``R_false`` is in the image's top decile,
                               or that was vetoed (crossing / corridor
                               overlap)
        p, U            float  calibrated acceptance probability / utility
        path_row, path_col     JSON-encoded list of the rasterised A* path's
                               pixel rows / columns (empty ``[]`` when A*
                               failed for that candidate) -- one row per path
                               pixel would be wasteful, so the path is packed
                               into these two cells instead of exploding rows
        radius          float  tube radius used to rasterise the path
                               (``rad_bar`` of the candidate table)

    ``edge_records`` is ``evaluate_repair(..., return_edge_records=True)``'s
    second return value: the accepted-only ``is_true_repair`` classification,
    keyed by ``edge_id`` (== ``cand_id``).
    """
    import json

    import pandas as pd

    table = prep["table"]
    rows_meta = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    results = prep["results"]
    util = out["utility"]
    n = len(rows_meta)

    selected = set(int(i) for i in out.get("selected", []))
    # "rejected" = a stage-1 hard veto OR stage-2 corridor-conflict pruning;
    # both are edge removals applied before the matching (see select_edges).
    vetoed = set(int(i) for i in out.get("vetoed_crossing", [])) | \
        set(int(i) for i in out.get("pruned_conflict",
                                    out.get("vetoed_overlap", [])))
    true_repair_by_cand = {int(r["edge_id"]): bool(r["is_true_repair"])
                           for r in (edge_records or [])}

    p_arr = util["p"].to_numpy(dtype=float) if "p" in util.columns else np.full(n, np.nan)
    u_arr = util["U"].to_numpy(dtype=float) if "U" in util.columns else np.full(n, np.nan)
    rf_arr = (util["R_false"].to_numpy(dtype=float) if "R_false" in util.columns
             else np.full(n, np.nan))
    finite_rf = rf_arr[np.isfinite(rf_arr)]
    top_decile = float(np.quantile(finite_rf, 0.9)) if finite_rf.size else float("inf")

    edge_rows: List[Dict[str, Any]] = []
    for i, row_meta in enumerate(rows_meta):
        cand_id = int(row_meta["cand_id"])
        accepted = i in selected
        res = results[i] if i < len(results) else None
        path_px = getattr(res, "path_px", None) if res is not None else None
        if getattr(res, "ok", False) and path_px is not None and len(path_px):
            path_px = np.asarray(path_px, dtype=np.int64).reshape(-1, 2)
            path_row, path_col = path_px[:, 0].tolist(), path_px[:, 1].tolist()
        else:
            path_row, path_col = [], []
        is_vetoed = i in vetoed
        rf = float(rf_arr[i]) if i < len(rf_arr) else float("nan")
        dangerous = (not accepted) and (
            is_vetoed or (np.isfinite(rf) and np.isfinite(top_decile) and rf >= top_decile))
        edge_rows.append(dict(
            cand_id=cand_id,
            accepted=bool(accepted),
            is_true_repair=bool(true_repair_by_cand.get(cand_id, False)) if accepted else False,
            dangerous=bool(dangerous),
            p=float(p_arr[i]) if i < len(p_arr) else float("nan"),
            U=float(u_arr[i]) if i < len(u_arr) else float("nan"),
            path_row=json.dumps(path_row),
            path_col=json.dumps(path_col),
            radius=float(row_meta.get("rad_bar", float("nan"))),
        ))
    return pd.DataFrame(edge_rows, columns=["cand_id", "accepted", "is_true_repair",
                                            "dangerous", "p", "U", "path_row",
                                            "path_col", "radius"])


# --------------------------------------------------------------------------
# scorer training on the training split
# --------------------------------------------------------------------------
def train_scorer_on_split(
    records: Sequence[dict],
    evidence_fn,
    model: str = "logreg",
    calibration: str = "temperature",
    n_cuts: int = 15,
    omega: float = 0.0,
    prior=None,
    seed: int = 0,
    val_fraction: float = 0.3,
    limit: Optional[int] = None,
    cand_kwargs=None,
    astar_kwargs=None,
    verbose: bool = True,
):
    """Uniform (+ failure-conditioned) severances on the training images -> scorer."""
    from src.rigr.scorer import PairScorer
    from src.rigr.synth_cuts import build_training_pairs
    from src.seg import data as segdata

    rng = np.random.default_rng(seed)
    recs = list(records)[: int(limit)] if limit else list(records)
    Xs, ys, groups = [], [], []
    for i, rec in enumerate(recs):
        key = os.path.splitext(os.path.basename(str(rec["image_path"])))[0]
        img = segdata._imread_color(rec["image_path"])
        gt = segdata._imread_gray(rec["label_path"]) > 127
        fov = (segdata._imread_gray(rec["fov_path"]) > 127
               if rec.get("fov_path") and os.path.exists(str(rec["fov_path"]))
               else segdata.derive_fov(img) > 0)
        d = build_training_pairs(img, gt & fov, fov, evidence_fn, n_cuts=n_cuts,
                                 omega=omega, prior=prior, rng=rng,
                                 cand_kwargs=cand_kwargs,
                                 astar_kwargs=astar_kwargs, image_id=key)
        if len(d["y"]):
            Xs.append(d["X"])
            ys.append(d["y"])
            groups += [i] * len(d["y"])
        if verbose:
            print("  [scorer] %-16s cand=%3d pos=%3d cuts=%d"
                  % (key, len(d["y"]), int(d["y"].sum()), len(d["cuts"])),
                  flush=True)
    if not Xs:
        raise RuntimeError("no training candidates were generated")
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    g = np.asarray(groups)

    uniq = np.unique(g)
    n_val = max(1, int(round(val_fraction * len(uniq)))) if len(uniq) > 1 else 0
    val_ids = set(uniq[-n_val:].tolist()) if n_val else set()
    m_val = np.isin(g, list(val_ids)) if val_ids else np.zeros(len(y), bool)
    Xtr, ytr = X[~m_val], y[~m_val]
    Xva, yva = X[m_val], y[m_val]
    if Xva.shape[0] == 0 or len(np.unique(yva)) < 2:
        Xva, yva = None, None
        warnings.warn("no usable validation group: the scorer is left UNCALIBRATED; "
                      "the strict probabilistic reading of U(e) does not apply",
                      RuntimeWarning)
    sc = PairScorer(model=model, calibration=calibration, seed=seed)
    sc.fit(Xtr, ytr, Xva, yva)
    return sc, dict(X=X, y=y, groups=g, n_val_images=len(val_ids))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    import pandas as pd

    from src.rigr import astar as astar_mod
    from src.rigr import utility as util_mod
    from src.seg import data as segdata

    ap = argparse.ArgumentParser(description="RiGR end-to-end")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="uniform", choices=list(util_mod.MODES))
    ap.add_argument("--head_ckpt", default=None)
    ap.add_argument("--evidence", default="auto", choices=["auto", "head", "frangi"],
                    help="'frangi' is the Tab.3 analytic-vesselness ablation")
    ap.add_argument("--orientation", default="learned",
                    choices=["learned", "uniform"],
                    help="q(theta|x): the head's prediction, or uniform "
                         "(Tab.3 appearance-only / +orientation contrast)")
    ap.add_argument("--pred_dir", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--scorer", default=None, help="a saved PairScorer to reuse")
    ap.add_argument("--geom_scale_file", default=None,
                    help="geom_scale.json holding m_R.  Default: the file "
                         "next to --scorer.  RISK mode requires it -- see "
                         "src/rigr/geom_scale.py for why an unscaled "
                         "C_geom makes U ~ -eta*C_geom.")
    ap.add_argument("--geom_scale", type=float, default=None,
                    help="override m_R directly (sensitivity runs)")
    ap.add_argument("--scorer_model", default="logreg", choices=["logreg", "mlp"])
    ap.add_argument("--omega", type=float, default=0.0,
                    help="fraction of failure-conditioned cuts (needs --pred_dir)")
    ap.add_argument("--n_cuts", type=int, default=15)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--mu", type=float, default=astar_mod.MU_KAPPA,
                    help="A* curvature weight mu (Tab.3 '+curvature': mu=0 "
                         "switches the (dtheta)^2/ds term off)")
    ap.add_argument("--lambda_theta", type=float, default=astar_mod.LAMBDA_THETA,
                    help="A* orientation weight lambda_theta")
    ap.add_argument("--btr_runs_dir", default=None,
                    help="directory holding btr_{miss,false}_<variant>.joblib "
                         "(default src.c1.btr.RUNS_DIR = runs/btr)")
    ap.add_argument("--btr_variant", default="all",
                    help="'all', or 'wo_<dataset>' for the LODO heads")
    ap.add_argument("--btr_miss", default=None,
                    help="explicit R_miss head .joblib (overrides --btr_variant)")
    ap.add_argument("--btr_false", default=None,
                    help="explicit *deployed* R_false head .joblib "
                         "(S4 item 6; overrides --btr_variant)")
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--bio", default="both", choices=["none", "skan", "pvbm", "both"])
    ap.add_argument("--fd_rotations", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train_limit", type=int, default=None)
    ap.add_argument("--synthetic_pred", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep_full", action="store_true",
                    help="use the full metric set in the sweep (slow)")
    ap.add_argument("--sweep_bio", default="none",
                    choices=["none", "skan", "pvbm", "both"],
                    help="biomarkers inside the sweep (skan is ~20x faster)")
    ap.add_argument("--sweep_lam", default="0.5,1,2,4")
    # Coordinator ruling 2026-09-05 23:20: 4 tau values, not 7 -- the grid
    # is 16 cells instead of 28 for the prob mode (the only mode that
    # sweeps tau at all).
    ap.add_argument("--sweep_tau", default="0.3,0.5,0.7,0.9")
    # --- sweep2: the operating-point sweep for the risk / uniform arms ------
    # Their utility has no threshold knob (tau is inert outside prob mode), so
    # --sweep gives them only len(sweep_lam) operating points -- 4, all in a
    # narrow FCR band, which is why the matched-FCR comparison in
    # results/fig3_matched_fcr_posthoc.csv could not place them against prob.
    # sweep2 adds a second axis: accept iff U > u0, where u0 is a QUANTILE OF
    # THE POSITIVE-U CANDIDATES OF THAT IMAGE (so it adapts to each image's
    # utility scale instead of imposing one absolute cut), then the usual
    # conflict pruning and matching run unchanged.  u0 = 0 reproduces the
    # pre-registered rule exactly, which is the grid's built-in control.
    ap.add_argument("--sweep2", action="store_true",
                    help="lambda x U-threshold operating-point sweep -> "
                         "sweep2_{curve,per_image}.csv (does not touch the "
                         "--sweep outputs)")
    ap.add_argument("--sweep2_lam", default="0.25,0.5,1,2,4,8,16")
    ap.add_argument("--sweep2_u0q", default="0,0.25,0.5,0.75",
                    help="quantiles of the positive-U candidates; 0 = the "
                         "pre-registered U > 0 rule")
    ap.add_argument("--sweep_scale", default="native",
                    choices=["native", "work"],
                    help="resolution at which the SWEEP CELLS' biomarkers "
                         "are measured.  'work' = the C1 working "
                         "resolution (longest side from dataset_cfg, 1536 "
                         "for HRF/FIVES) with the 1/s native-unit "
                         "conversion for length-like biomarkers -- the "
                         "same convention as C1 and build_data.  The main "
                         "operating point in per_image.csv is ALWAYS "
                         "measured at native resolution.")
    args = ap.parse_args(argv)

    ds = segdata.canon(args.dataset)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, "mask"), exist_ok=True)
    rng = np.random.default_rng(args.seed)

    evidence_fn, ev_name = make_evidence_fn(args.head_ckpt, args.gpu, args.evidence,
                                            orientation=args.orientation)
    # A* knobs are only forwarded when they differ from the module defaults, so
    # a plain run stays bit-identical to one made before these flags existed.
    astar_kwargs: Dict[str, Any] = {}
    if float(args.mu) != float(astar_mod.MU_KAPPA):
        astar_kwargs["mu"] = float(args.mu)
    if float(args.lambda_theta) != float(astar_mod.LAMBDA_THETA):
        astar_kwargs["lam"] = float(args.lambda_theta)

    # Risk backend: ``None`` == the plain ``src.c1.btr`` module (all-data
    # heads), which is what every previous run used.  Any of --btr_miss /
    # --btr_false / --btr_variant / --btr_runs_dir binds explicit files
    # instead; the choice is echoed into summary.json.
    risk_backend = None
    if args.mode == "risk" and (args.btr_miss or args.btr_false
                                or args.btr_variant != "all"
                                or args.btr_runs_dir):
        from src.rigr.btr_backend import make_backend

        def _try(miss, false, why):
            try:
                return make_backend(miss, false, args.btr_runs_dir,
                                    args.btr_variant), None
            except (FileNotFoundError, ValueError) as exc:
                return None, "%s (%s: %s)" % (why, type(exc).__name__, exc)

        risk_backend, err = _try(args.btr_miss, args.btr_false,
                                 "requested BTR pair")
        if risk_backend is None and args.btr_false and args.btr_runs_dir:
            # The deployed R_false is missing, or was fitted on a different
            # target family than the R_miss head it is paired with.  Retry with
            # BOTH heads taken from --btr_runs_dir: that keeps the run on the
            # deployed (H_abs, training-split-only) heads instead of silently
            # dropping to the flat runs/btr/btr_*_all.joblib layout, which was
            # fitted on perturbations of the very test images being repaired.
            print("[rigr][WARN] %s -> retrying with both heads from %s"
                  % (err, args.btr_runs_dir), flush=True)
            risk_backend, err2 = _try(args.btr_miss, None,
                                      "fallback pair from --btr_runs_dir")
            if risk_backend is None:
                err = "%s; then %s" % (err, err2)
        if risk_backend is None:
            warnings.warn(
                "no usable BTR pair: %s. Falling back to the all-data heads of "
                "src.c1.btr -- that layout is H_dep and was fitted on the test "
                "images too, so this run is NOT the deployed risk backend and "
                "must not be reported as risk-guided without saying so." % err,
                RuntimeWarning)
            print("[rigr][WARN] no usable BTR pair: %s -> falling back to "
                  "runs/btr/*_all.joblib" % err, flush=True)
    print("[rigr] dataset=%s seed=%d mode=%s evidence=%s astar=%s btr=%s"
          % (ds, args.seed, args.mode, ev_name, astar_kwargs or "defaults",
             getattr(risk_backend, "__name__", "src.c1.btr(all)")), flush=True)

    recs = segdata.get_records(ds)
    tr, va, te, info = segdata.make_splits(recs)
    subset = {"test": te, "val": va, "train": tr}[args.split]
    if args.limit:
        subset = subset[: int(args.limit)]

    # ---- failure prior (optional) ----------------------------------------
    prior = None
    if args.omega > 0:
        from src.rigr.synth_cuts import estimate_failure_prior, load_oof_pairs

        pairs = load_oof_pairs(ds, args.pred_dir or "", split="train")
        if pairs:
            prior, ev = estimate_failure_prior(pairs)
            prior.save(os.path.join(args.out, "failure_prior.json"))
            ev.to_csv(os.path.join(args.out, "failure_events.csv"), index=False)
            print("[rigr] failure prior from %d source images, %d gaps"
                  % (len(pairs), len(ev)), flush=True)
        else:
            args.omega = 0.0

    # ---- scorer ----------------------------------------------------------
    from src.rigr.scorer import PairScorer

    if args.scorer and os.path.exists(args.scorer):
        scorer = PairScorer.load(args.scorer)
        print("[rigr] scorer loaded from %s" % args.scorer, flush=True)
    else:
        print("[rigr] training the pair scorer on the %d training images"
              % len(tr), flush=True)
        scorer, diag = train_scorer_on_split(
            tr, evidence_fn, model=args.scorer_model, n_cuts=args.n_cuts,
            omega=args.omega, prior=prior, seed=args.seed,
            limit=args.train_limit, astar_kwargs=astar_kwargs or None)
        scorer.save(os.path.join(args.out, "scorer.joblib"))
        rep = scorer.evaluate(diag["X"], diag["y"])
        print("[rigr] scorer (in-sample) AUC=%.3f Brier=%.4f ECE=%.4f"
              % (rep["auc"], rep["brier"], rep["ece"]), flush=True)
        rep["reliability"].to_csv(os.path.join(args.out, "scorer_reliability.csv"),
                                  index=False)

    # ---- test images -----------------------------------------------------
    rows: List[Dict[str, Any]] = []
    gt_bio: List[Dict[str, float]] = []
    bio_before: List[Dict[str, float]] = []
    bio_after: List[Dict[str, float]] = []
    # ---- m_R: put C_geom on the scale of R (DECISIONS.md 2026-09-06 12:30) --
    geom_scale = args.geom_scale
    geom_scale_src = "--geom_scale" if geom_scale is not None else None
    if geom_scale is None:
        gsf = args.geom_scale_file
        if gsf is None and args.scorer:
            gsf = os.path.join(os.path.dirname(args.scorer), "geom_scale.json")
        if gsf and os.path.exists(gsf):
            with open(gsf, encoding="utf-8") as _f:
                geom_scale = float(json.load(_f)["m_R"])
            geom_scale_src = gsf
    if args.mode == "risk" and geom_scale is None:
        print("[rigr][fatal] risk mode needs m_R and no geom_scale.json was "
              "found next to the scorer (%s).  Fit it with: "
              "    python -m src.rigr.geom_scale --data runs/rigr_data/%s "
              "--out %s" % (args.scorer, args.dataset,
                            os.path.dirname(args.scorer or "")), flush=True)
        return 2
    if args.mode == "risk":
        print("[rigr] m_R = %.6g (from %s); C_geom is multiplied by it so the "
              "eta term is on the scale of R" % (geom_scale, geom_scale_src),
              flush=True)

    sweep_rows: List[Dict[str, Any]] = []
    sweep2_rows: List[Dict[str, Any]] = []
    # image key -> BEFORE biomarkers measured at the sweep resolution
    sweep_before_bio: Dict[str, Dict[str, float]] = {}
    keys: List[str] = []

    for rec in subset:
        case = load_case(rec, args.pred_dir, args.synthetic_pred, rng, args.n_cuts)
        t0 = time.perf_counter()
        disc = auto_disc(case)
        prep = prepare_image(case, evidence_fn, disc=disc,
                             astar_kwargs=astar_kwargs or None)
        p = (scorer.predict_proba(prep["X"]) if len(prep["table"])
             else np.zeros(0))
        out = util_mod.repair_mask(
            case["mask"], prep["table"], prep["results"], p, mode=args.mode,
            lam=args.lam, eta=args.eta, tau=args.tau, fov=case["fov"],
            labels=prep["cand"]["labels"], theta_mask=prep["cand"]["theta_mask"],
            radius=prep["cand"]["radius"], X=prep["X"], image=case["image"],
            disc=disc, risk_backend=risk_backend, geom_scale=geom_scale)
        t_total = time.perf_counter() - t0

        row = dict(dataset=ds, seed=args.seed, mode=args.mode, image=case["key"],
                   subject_id=case["subject_id"], pred_source=case["source"],
                   evidence=ev_name, n_endpoints=prep["cand"]["n_endpoints"],
                   n_ports=prep["cand"]["n_ports"],
                   n_candidates=len(prep["table"]),
                   n_positive_U=out["n_positive"],
                   n_selected=len(out["selected"]),
                   n_vetoed=out["n_vetoed"],
                   n_veto_crossing=len(out["vetoed_crossing"]),
                   n_pruned_conflict=out["n_pruned_conflict"],
                   overlap_metric=out["overlap_metric"],
                   risk_backend=out["utility"].attrs["risk_backend"],
                   t_evidence_s=prep["t_evidence"],
                   t_candidates_s=prep["t_candidates"],
                   t_astar_s=prep["t_astar"],
                   t_astar_per_cand_ms=1e3 * prep["t_astar_per_cand"],
                   t_total_s=t_total)
        row_metrics, edge_records = evaluate_repair(
            case, prep, out["mask"], out["edges"], return_edge_records=True)
        row.update(row_metrics)

        if args.bio != "none":
            g = biomarker_row(case["gt"], case["fov"], case["image"], disc=disc,
                              pipelines=args.bio, fd_rotations=args.fd_rotations)
            b = biomarker_row(case["mask"], case["fov"], case["image"], disc=disc,
                              pipelines=args.bio, fd_rotations=args.fd_rotations)
            a = biomarker_row(out["mask"], case["fov"], case["image"], disc=disc,
                              pipelines=args.bio, fd_rotations=args.fd_rotations)
            gt_bio.append(g)
            bio_before.append(b)
            bio_after.append(a)
        keys.append(case["key"])

        import cv2

        os.makedirs(os.path.join(args.out, "mask_before"), exist_ok=True)
        cv2.imwrite(os.path.join(args.out, "mask_before", case["key"] + ".png"),
                    case["mask"].astype(np.uint8) * 255)
        cv2.imwrite(os.path.join(args.out, "mask", case["key"] + ".png"),
                    out["mask"].astype(np.uint8) * 255)
        if len(prep["table"]):
            prep["table"].assign(
                p=p, U=out["utility"]["U"].to_numpy()).to_csv(
                os.path.join(args.out, "cand_%s.csv" % case["key"]), index=False)
            edge_df = build_edge_table(prep, out, edge_records)
            os.makedirs(os.path.join(args.out, "edges"), exist_ok=True)
            edge_df.to_csv(os.path.join(args.out, "edges", case["key"] + ".csv"),
                           index=False)
        rows.append(row)
        print("[%s] cand=%d sel=%d  clDice %.4f->%.4f  TRR_r=%.3f FCR=%.3f  "
              "A*=%.1f ms/cand  %.1fs"
              % (case["key"], row["n_candidates"], row["n_selected"],
                 row["before_cldice"], row["after_cldice"],
                 row["TRR_recall"], row["FCR"],
                 row["t_astar_per_cand_ms"], t_total), flush=True)

        # ---- sweep (selection stage only) --------------------------------
        if args.sweep:
            lams = [float(x) for x in args.sweep_lam.split(",")]
            taus = [float(x) for x in args.sweep_tau.split(",")]
            # Work-scale context, built once per image.  When it is active the
            # sweep's BEFORE baseline is re-measured at the same resolution as
            # its AFTER values, so ``macro_mae_benefit`` -- the axis Fig.3
            # actually plots -- is free of any resolution offset between the
            # two.  per_image.csv keeps its native-resolution numbers.
            wctx = _sweep_work_ctx(args, ds, case)
            if wctx is not None and args.sweep_bio != "none":
                sweep_before_bio[case["key"]] = _bio_at_work_scale(
                    wctx, case["mask"], args.sweep_bio, args.fd_rotations)
            # The two knobs are mode-exclusive, and sweeping the inert one
            # only duplicates cells (``compute_utility``):
            #   prob : U = p - tau            -- lambda and eta are unused
            #   else : U = p*R_miss - lambda*(1-p)*R_false - eta*C_geom
            #                                 -- tau is unused, acceptance is
            #                                    the hard U > 0 of 3.2.6
            # So prob iterates tau alone and the other two iterate lambda
            # alone.  Before this dedup a prob sweep computed every utility
            # vector len(lams) times over -- 4x the work for four identical
            # copies of each cell.  DECISIONS.md 2026-09-06.
            if args.mode != "prob":
                taus = taus[:1]
            else:
                lams = lams[:1]
            for lam in lams:
                for tau in taus:
                    o = util_mod.repair_mask(
                        case["mask"], prep["table"], prep["results"], p,
                        mode=args.mode, lam=lam, eta=args.eta, tau=tau,
                        fov=case["fov"], labels=prep["cand"]["labels"],
                        theta_mask=prep["cand"]["theta_mask"],
                        radius=prep["cand"]["radius"], X=prep["X"],
                        image=case["image"], disc=disc,
                        risk_backend=risk_backend, geom_scale=geom_scale)
                    sr = dict(dataset=ds, seed=args.seed, mode=args.mode,
                              image=case["key"], lam=lam, tau=tau,
                              n_selected=len(o["selected"]))
                    sr.update(evaluate_repair(case, prep, o["mask"], o["edges"])
                              if args.sweep_full
                              else evaluate_repair_light(case, o["mask"],
                                                         o["edges"]))
                    if args.sweep_bio != "none":
                        sr["_bio"] = (
                            _bio_at_work_scale(wctx, o["mask"],
                                               args.sweep_bio,
                                               args.fd_rotations)
                            if wctx is not None else
                            biomarker_row(
                                o["mask"], case["fov"], case["image"],
                                disc=disc, pipelines=args.sweep_bio,
                                fd_rotations=args.fd_rotations))
                        sr["sweep_scale"] = (wctx["scale"]
                                             if wctx is not None else 1.0)
                    sweep_rows.append(sr)

        # ---- sweep2: lambda x U-threshold operating points -----------------
        if args.sweep2:
            lam2 = [float(x) for x in args.sweep2_lam.split(",")]
            q2 = [float(x) for x in args.sweep2_u0q.split(",")]
            wctx2 = _sweep_work_ctx(args, ds, case)
            if wctx2 is not None and args.sweep_bio != "none" \
                    and case["key"] not in sweep_before_bio:
                sweep_before_bio[case["key"]] = _bio_at_work_scale(
                    wctx2, case["mask"], args.sweep_bio, args.fd_rotations)
            for lam in lam2:
                # The utility vector depends on lambda but not on u0, so it is
                # computed once per lambda and the four thresholds are read off
                # it -- otherwise every cell would recompute R_miss/R_false.
                util_l = util_mod.compute_utility(
                    prep["table"], prep["results"], p, mode=args.mode,
                    lam=lam, eta=args.eta, tau=args.tau, X=prep["X"],
                    image=case["image"], mask=case["mask"], fov=case["fov"],
                    disc=disc, risk_backend=risk_backend,
                    geom_scale=geom_scale)
                u_vec = np.asarray(util_l["U"], dtype=float)
                pos = u_vec[np.isfinite(u_vec) & (u_vec > 0)]
                for q in q2:
                    # quantile of the POSITIVE-U candidates of this image;
                    # q = 0 -> u_floor = 0 -> the pre-registered rule
                    u0 = 0.0 if (q <= 0 or pos.size == 0) \
                        else float(np.quantile(pos, q))
                    sel = util_mod.select_edges(
                        prep["table"], prep["results"], util_l,
                        labels=prep["cand"]["labels"],
                        theta_mask=prep["cand"]["theta_mask"],
                        radius=prep["cand"]["radius"], u_floor=u0)
                    rmask, redges = util_mod.apply_edges(
                        case["mask"], prep["table"], prep["results"],
                        sel["selected"], case["fov"])
                    sr = dict(dataset=ds, seed=args.seed, mode=args.mode,
                              image=case["key"], lam=lam, u0q=q, u0=u0,
                              n_positive=int(sel["n_positive"]),
                              n_selected=len(sel["selected"]))
                    sr.update(evaluate_repair(case, prep, rmask, redges)
                              if args.sweep_full
                              else evaluate_repair_light(case, rmask, redges))
                    if args.sweep_bio != "none":
                        sr["_bio"] = (
                            _bio_at_work_scale(wctx2, rmask, args.sweep_bio,
                                               args.fd_rotations)
                            if wctx2 is not None else
                            biomarker_row(rmask, case["fov"], case["image"],
                                          disc=disc, pipelines=args.sweep_bio,
                                          fd_rotations=args.fd_rotations))
                        sr["sweep_scale"] = (wctx2["scale"]
                                             if wctx2 is not None else 1.0)
                    sweep2_rows.append(sr)

    if not rows:
        print("[rigr] no images processed")
        return 2

    # ---- biomarker macro-MAE --------------------------------------------
    # schema item 3: sigma is read once from the single source of truth
    # (results/gateA_biomarker_scales.csv, GT masks at native resolution) --
    # never recomputed in-memory from this run's (possibly tiny) image set.
    pipes = pipes_for(args.bio)
    sigma = load_gt_scales(ds) if gt_bio else {}
    if gt_bio:
        for i, r in enumerate(rows):
            r.update(_bio_columns(gt_bio[i], bio_before[i], bio_after[i], sigma, pipes))

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "per_image.csv")
    df.to_csv(csv_path, index=False)

    if sweep_rows:
        for sr in sweep_rows:
            bio = sr.pop("_bio", None)
            if bio is not None and gt_bio:
                i = keys.index(sr["image"])
                spipes = pipes_for(args.sweep_bio)
                sr["macro_mae_after"] = macro_mae(bio, gt_bio[i], sigma, spipes)
                # same-resolution BEFORE when the sweep ran at work scale
                _bb = sweep_before_bio.get(sr["image"], bio_before[i])
                sr["macro_mae_before"] = macro_mae(_bb, gt_bio[i],
                                                   sigma, spipes)
                sr["macro_mae_benefit"] = (sr["macro_mae_before"]
                                           - sr["macro_mae_after"])
        sw = pd.DataFrame(sweep_rows)
        sw.to_csv(os.path.join(args.out, "sweep_per_image.csv"), index=False)
        # n_accepted / n_matched are carried through so a bare sweep_curve.csv
        # can still be re-expressed in the matched FCR of DECISIONS.md
        # 2026-09-03 10:20 (src.eval.tables.add_matched_fcr) without the
        # per-image file.
        agg_cols = [c for c in ("TRR_recall", "TRR_precision", "FCR",
                                "TRR_precision_matched", "FCR_matched",
                                "n_accepted", "n_matched", "n_should",
                                "after_cldice", "after_dice", "macro_mae_after",
                                "macro_mae_benefit", "n_selected") if c in sw]
        curve = sw.groupby(["lam", "tau"])[agg_cols].mean().reset_index()
        # schema item 4: sweep_curve.csv must carry dataset/seed/mode (not just
        # lam/tau) so a copy of it cannot be confused with another run's sweep
        # once pulled out of its run directory.
        curve.insert(0, "mode", args.mode)
        curve.insert(0, "seed", args.seed)
        curve.insert(0, "dataset", ds)
        curve.to_csv(os.path.join(args.out, "sweep_curve.csv"), index=False)

    if sweep2_rows:
        # written to sweep2_* so an existing --sweep result is never clobbered
        for sr in sweep2_rows:
            bio = sr.pop("_bio", None)
            if bio is not None and gt_bio:
                i = keys.index(sr["image"])
                spipes = pipes_for(args.sweep_bio)
                sr["macro_mae_after"] = macro_mae(bio, gt_bio[i], sigma, spipes)
                _bb = sweep_before_bio.get(sr["image"], bio_before[i])
                sr["macro_mae_before"] = macro_mae(_bb, gt_bio[i], sigma, spipes)
                sr["macro_mae_benefit"] = (sr["macro_mae_before"]
                                           - sr["macro_mae_after"])
        sw2 = pd.DataFrame(sweep2_rows)
        sw2.to_csv(os.path.join(args.out, "sweep2_per_image.csv"), index=False)
        agg2 = [c for c in ("TRR_recall", "TRR_precision", "FCR",
                            "TRR_precision_matched", "FCR_matched",
                            "n_accepted", "n_matched", "n_should",
                            "after_cldice", "after_dice", "macro_mae_after",
                            "macro_mae_benefit", "n_selected", "n_positive",
                            "u0") if c in sw2]
        curve2 = sw2.groupby(["lam", "u0q"])[agg2].mean().reset_index()
        curve2.insert(0, "mode", args.mode)
        curve2.insert(0, "seed", args.seed)
        curve2.insert(0, "dataset", ds)
        curve2.to_csv(os.path.join(args.out, "sweep2_curve.csv"), index=False)
        print("[rigr] sweep2: %d cells x %d images -> sweep2_curve.csv"
              % (len(curve2), len(keys)))

    num = df.select_dtypes("number")
    summary = dict(dataset=ds, seed=args.seed, mode=args.mode, evidence=ev_name,
                   split=args.split, n_images=len(df),
                   pred_source=str(df["pred_source"].iloc[0]),
                   risk_backend=str(df["risk_backend"].iloc[0]),
                   geom_scale=(float(geom_scale) if geom_scale is not None
                               else None),
                   geom_scale_source=geom_scale_src,
                   lam=args.lam, eta=args.eta, tau=args.tau, omega=args.omega,
                   mu=args.mu, lambda_theta=args.lambda_theta,
                   orientation=args.orientation,
                   btr=(risk_backend.describe() if risk_backend is not None
                        else dict(name="src.c1.btr(all)")),
                   mean={k: float(v) for k, v in num.mean().items()})
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    show = [c for c in ("before_dice", "after_dice", "before_cldice",
                        "after_cldice", "before_bcs", "after_bcs",
                        "TRR_recall", "TRR_precision", "FCR",
                        "macro_mae_before", "macro_mae_after",
                        "t_astar_per_cand_ms", "t_total_s") if c in df]
    print("\n[rigr] means over %d images:" % len(df))
    for c in show:
        print("  %-22s %.4f" % (c, float(df[c].mean())))
    print("wrote %s" % csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
