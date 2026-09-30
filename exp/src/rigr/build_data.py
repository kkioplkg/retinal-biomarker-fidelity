"""Stage S4 step B: build every *training* artefact RiGR fits models on.

    python -m src.rigr.build_data --dataset drive --seed 0 \
        --oof_dir runs/seg_oof/drive/pred \
        --head_ckpt runs/rigr_head/drive/seed0/best.pt --gpu 0 \
        --out runs/rigr_data/drive --omega 0.5 --rfalse

Three artefacts, all built from **training-split images only** (the anti-leakage
rule of proposal section 3.2.5; ``--datasets`` unions several *source* datasets
for the LODO block and the held-out dataset is simply never listed):

``<out>/failure_prior.json`` + ``failure_events.csv``   (per dataset, once)
    ``pi``, the failure prior of :func:`src.rigr.synth_cuts.estimate_failure_prior`,
    fitted on the **out-of-fold** predictions of the training images
    (``runs/seg_oof/<ds>/pred``).  Cross-fitting is what makes this legitimate:
    the model that produced the prediction for an image never saw that image.
    Passing a *test-split* prediction directory here is refused.

``<out>/seed<k>/pairs.npz``                             (per dataset x seed)
    the pair-scorer training set ``(X, y)`` of
    :func:`src.rigr.synth_cuts.build_training_pairs`: ``omega`` of the
    severances drawn from ``pi``, the rest uniform, candidates + A* solved with
    the **seed-matched** micro head, and ``y`` from
    :func:`src.rigr.synth_cuts.label_candidates`.  ``X`` depends on the head
    (the A* energy and the ``V``/``q`` statistics are three of its thirteen
    features), which is why this artefact is per seed.

``<out>/rfalse_train.csv``                              (per dataset, once)
    the **deployed** ``R_false`` training set of proposal section 3.1.5 /
    plan S4 item 6.  For every candidate the A* path is rasterised onto the
    severed mask, giving the connected state ``M+ = M- U T_e``; the
    Contract-F features ``phi_*`` are then evaluated on ``M+`` by
    ``src.c1.perturb.phi`` -- the identical schema and the identical call
    ``src.rigr.utility._phi_frames`` makes at inference time -- and the
    counterfactual harm is measured with the *same* biomarker pipeline and the
    *same* single source of sigma as every other number in the paper
    (``src.eval.biomarker_eval``, ``results/gateA_biomarker_scales.csv``).  Two
    target families are stored per candidate (see :func:`_harm_row`):

        Habs_<b>_<p> = |B(M+) - B(M-)| / sigma_b                    <- deployed
        Hdep_<b>_<p> = max(0, (|B(M+)-B(GT)| - |B(M-)-B(GT)|)/sigma_b)

    ``H_abs`` is the raw absolute standardized harm of the edit and is what the
    deployed head is fitted on (DECISIONS.md 2026-09-03 08:30): it is the full
    counterfactual consequence, it needs no reference mask, and it matches the
    target of the ``runs/btr/habs_trainonly`` ``R_miss`` head it is paired with.
    ``H_dep`` is the pre-registered net harm, kept for the dose-response
    analysis.
    ``is_true_repair`` is carried along so
    :func:`src.c1.btr.fit_deployed_false` can restrict the fit to the wrong
    connections -- the population the term ``lambda (1 - p_e) R_false(e)``
    actually prices.

    Because ``phi`` reads only ``(I, M+)``, this table is **head-independent
    given the paths**; only which paths exist depends on the seed.  It is
    therefore built once, from the seed-0 head, and reused by all three seeds
    (recorded as ``path_seed`` in ``meta.json``).

Cost control: the biomarker call is 1-4 s per mask, so ``--max_cand_bio``
(default 24) subsamples candidates per image (seeded, stratified by the A*
success flag) and ``--bio`` defaults to ``skan`` -- roughly 20x faster than
``both`` and the pipeline Gate A kept for every primary biomarker
(tortuosity has no second pipeline anyway; see ``exp/DECISIONS.md``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.topo import skeleton as sk

__all__ = ["build_prior", "build_pairs_and_rfalse", "main"]

PRIOR_FILE = "failure_prior.json"
EVENTS_FILE = "failure_events.csv"
RFALSE_FILE = "rfalse_train.csv"
PAIRS_FILE = "pairs.npz"


# --------------------------------------------------------------------------
def _training_records(datasets: Sequence[str], split_seed: Optional[int] = None,
                      per_dataset_limit: Optional[Dict[str, int]] = None):
    """``[(dataset, record), ...]`` over the *training* split of each dataset.

    ``per_dataset_limit`` caps each dataset independently and always takes the
    **first** n of the fixed split, so every method that honours the same cap
    sees exactly the same images (FIVES has 510 training images against
    DRIVE's 17 and would otherwise dominate both the runtime and any
    dataset-pooled fit)."""
    from src.seg import data as segdata

    caps = per_dataset_limit or {}
    out = []
    for ds in datasets:
        recs = segdata.get_records(ds)
        tr, _va, _te, _info = segdata.make_splits(
            recs, split_seed if split_seed is not None else segdata.DEFAULT_SPLIT_SEED)
        key = segdata.canon(ds)
        n = caps.get(key)
        if n:
            tr = tr[: int(n)]
        out += [(key, r) for r in tr]
    return out


def _read_case(rec: dict):
    from src.seg import data as segdata

    key = os.path.splitext(os.path.basename(str(rec["image_path"])))[0]
    img = segdata._imread_color(rec["image_path"])
    gt = segdata._imread_gray(rec["label_path"]) > 127
    if rec.get("fov_path") and os.path.exists(str(rec["fov_path"])):
        fov = segdata._imread_gray(rec["fov_path"]) > 127
    else:
        fov = segdata.derive_fov(img) > 0
    return key, img, (gt & fov), fov


# --------------------------------------------------------------------------
# pi
# --------------------------------------------------------------------------
def build_prior(datasets: Sequence[str], oof_dirs: Dict[str, str], out_dir: str,
                limit: Optional[int] = None, verbose: bool = True):
    """Fit ``pi`` from the OOF predictions of the training images and save it."""
    import pandas as pd

    from src.rigr.synth_cuts import estimate_failure_prior, load_oof_pairs

    pairs: List[Dict[str, Any]] = []
    per_ds = {}
    for ds in datasets:
        d = oof_dirs.get(ds)
        if not d or not os.path.isdir(d):
            print("[build_data][WARN] no OOF prediction dir for %s (%r); its "
                  "images contribute nothing to pi" % (ds, d), flush=True)
            per_ds[ds] = 0
            continue
        p = load_oof_pairs(ds, d, split="train", limit=limit)
        per_ds[ds] = len(p)
        pairs += p
        # How many training images SHOULD have an OOF prediction?  pi is the
        # one quantity here that silently degrades when a prediction file is
        # simply absent: load_oof_pairs skips what it cannot open, so a mirror
        # holding a subset of the OOF maps fits pi on that subset and says
        # nothing.  That happened on the remote box (178 of 510 FIVES maps
        # present), and the only symptom was a pi fitted on a different set
        # than the local run's.  Report found vs expected and warn on any gap.
        n_expect = -1
        try:
            from src.seg import data as _segdata
            _tr, _va, _te, _ = _segdata.make_splits(_segdata.get_records(ds))
            n_expect = len(_tr) if not limit else min(int(limit), len(_tr))
        except Exception as _exc:            # never let a diagnostic break the fit
            print("[build_data][WARN] could not determine the expected training "
                  "image count for %s (%s: %s)" % (ds, type(_exc).__name__, _exc),
                  flush=True)
        per_ds["%s_expected" % ds] = n_expect
        if verbose:
            print("[build_data] pi: %s -> %d out-of-fold training images "
                  "(expected %s)"
                  % (ds, len(p), n_expect if n_expect >= 0 else "?"), flush=True)
        if n_expect >= 0 and len(p) != n_expect:
            print("[build_data][WARN] pi: %s has %d/%d OOF training predictions "
                  "under %s -- %d missing.  pi is being fitted on a SUBSET, and "
                  "will not match a run where all %d are present."
                  % (ds, len(p), n_expect, d, n_expect - len(p), n_expect),
                  flush=True)
    if not pairs:
        raise SystemExit(
            "no out-of-fold predictions found under %s; run "
            "run_crossfit_all.ps1 first (the failure prior may only be fitted "
            "on cross-fitted predictions of *training* images)" % list(oof_dirs.values()))

    prior, events = estimate_failure_prior(pairs, verbose=False)
    prior.meta["datasets"] = list(datasets)
    prior.meta["n_images_per_dataset"] = per_ds
    os.makedirs(out_dir, exist_ok=True)
    prior.save(os.path.join(out_dir, PRIOR_FILE))
    events.to_csv(os.path.join(out_dir, EVENTS_FILE), index=False)
    print("[build_data] pi fitted on %d images / %d failure gaps -> %s"
          % (len(pairs), len(events), os.path.join(out_dir, PRIOR_FILE)), flush=True)
    return prior, events


# --------------------------------------------------------------------------
# pairs (+ the deployed R_false table)
# --------------------------------------------------------------------------
def _harm_row(bio_plus: Dict[str, float], bio_minus: Dict[str, float],
              bio_gt: Dict[str, float], sigma: Dict[str, float],
              pipelines: Sequence[str], work_scale: float = 1.0
              ) -> Dict[str, float]:
    """Both supervision targets of the counterfactual edit, per biomarker.

    ``Habs_<b>_<p> = |B(M+) - B(M-)| / sigma_b``
        the **raw absolute standardized harm**: the whole consequence of adding
        this edge, pixels and connectivity together.  This is the target the
        *deployed* ``R_false`` head is fitted on (DECISIONS.md 2026-09-03
        08:30) -- it is what ``U(e)`` has to price, it needs no reference mask
        at all, and it is the target of the ``habs_trainonly`` ``R_miss`` head
        this one is paired with.

    ``Hdep_<b>_<p> = max(0, (|B(M+) - B(GT)| - |B(M-) - B(GT)|) / sigma_b)``
        the pre-registered *net* harm, i.e. how much the edit increased the
        measurement error.  Gate B found it sits at pipeline-noise level, so it
        is kept for the scientific dose-response analysis and as a sensitivity
        fit, not for deployment.

    Both are non-negative and standardised by the single source of sigma.
    """
    from src.c1.stats_c1 import native_factor
    from src.eval.biomarker_eval import PRIMARY

    out: Dict[str, float] = {}
    out["work_scale"] = float(work_scale)
    for b in PRIMARY:
        for p in pipelines:
            col = "%s_%s" % (b, p)
            sg = float(sigma.get(col, np.nan))
            gv = float(bio_gt.get(col, np.nan))
            av = float(bio_plus.get(col, np.nan))
            bv = float(bio_minus.get(col, np.nan))
            # |dB| measured at the working resolution -> native units, exactly
            # the C1 convention: only length-like biomarkers rescale (1/s), the
            # dimensionless ones (FD, density, tortuosity) do not.
            conv = float(native_factor(col, work_scale))
            out["nativef_" + col] = conv
            ok_edit = (np.isfinite(sg) and sg > 0
                       and np.isfinite(av) and np.isfinite(bv))
            out["Habs_" + col] = (float(abs(av - bv) * conv / sg)
                                  if ok_edit else np.nan)
            out["Hdep_" + col] = (
                float(max(0.0, (abs(av - gv) - abs(bv - gv)) * conv / sg))
                if (ok_edit and np.isfinite(gv)) else np.nan)
    return out




# --------------------------------------------------------------------------
# per-image work (run in a multiprocessing pool)
# --------------------------------------------------------------------------
_EV_CACHE: Dict[Any, Any] = {}
_PRIOR_CACHE: Dict[str, Any] = {}


def _worker_setup() -> None:
    """One thread per worker, and no GPU.

    Six workers each grabbing every core through OpenMP/MKL is slower than one,
    and two of them touching the same card is worse.  ``CUDA_VISIBLE_DEVICES``
    is cleared before torch is first imported in the child (the head is a
    344k-parameter U-Net; on CPU it is a rounding error next to the biomarker
    pipelines this function exists to parallelise).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass


def _worker_evidence(job: Dict[str, Any]):
    key = (job.get("head_ckpt"), job.get("evidence"), job.get("orientation"))
    if key not in _EV_CACHE:
        from src.rigr.run_rigr import make_evidence_fn

        _EV_CACHE[key] = make_evidence_fn(job.get("head_ckpt"), 0,
                                          job.get("evidence", "auto"),
                                          orientation=job.get("orientation",
                                                              "learned"))[0]
    return _EV_CACHE[key]


def _worker_prior(path: Optional[str]):
    if not path:
        return None
    if path not in _PRIOR_CACHE:
        from src.rigr.synth_cuts import FailurePrior

        _PRIOR_CACHE[path] = FailurePrior.load(path)
    return _PRIOR_CACHE[path]


def _to_work_scale(arr, longest: Optional[int], is_mask: bool):
    """Resize one array to the C1 working resolution (longest side ``longest``)."""
    import cv2

    from src.seg import data as segdata

    if longest is None:
        return arr
    h, w = arr.shape[:2]
    if max(h, w) == int(longest):
        return arr
    if is_mask:
        return segdata._resize_longest(
            np.asarray(arr, bool).astype(np.uint8) * 255, int(longest),
            cv2.INTER_LINEAR) > 127
    return segdata._resize_longest(arr, int(longest), cv2.INTER_AREA)


def process_image(job: Dict[str, Any]) -> Dict[str, Any]:
    """Scorer pairs (+ optionally the R_false rows) for ONE training image."""
    from src.c1.perturb import PHI_COLUMNS, _EventPoints, phi
    from src.eval.biomarker_eval import auto_disc, biomarker_row, pipes_for
    from src.eval.trr_fcr import rasterize_tube
    from src.rigr.synth_cuts import build_training_pairs

    t0 = time.time()
    ds, rec, idx = job["dataset"], job["record"], int(job["index"])
    key, img, gt, fov = _read_case(rec)
    # Per-image RNG: the sequential build drew every image from one stream, which
    # a pool cannot reproduce.  Seeding per (seed, image index) keeps the run
    # deterministic and independent of worker scheduling.
    rng = np.random.default_rng([int(job["seed"]), idx, 991])

    evidence_fn = _worker_evidence(job)
    prior = _worker_prior(job.get("prior_path"))
    d = build_training_pairs(img, gt, fov, evidence_fn, n_cuts=int(job["n_cuts"]),
                             omega=float(job["omega"]), prior=prior, rng=rng,
                             cand_kwargs=job.get("cand_kwargs"),
                             astar_kwargs=job.get("astar_kwargs"), image_id=key)
    n = len(d["y"])
    out: Dict[str, Any] = dict(
        dataset=ds, image=key, index=idx,
        subject_id=str(rec.get("subject_id", key)),
        n_cand=int(n), n_pos=int(np.sum(d["y"])) if n else 0,
        n_cuts=int(len(d["cuts"])), rf_rows=[], t_pairs=time.time() - t0)
    if n:
        out["X"] = d["X"]
        out["y"] = np.asarray(d["y"], dtype=int)
        out["cand_id"] = [int(c) for c in d["table"]["cand_id"].to_numpy()]
    if not job.get("make_rfalse") or not n:
        out["seconds"] = time.time() - t0
        return out

    bio = job["bio"]
    pipes = pipes_for(bio)
    fd_rot = int(job["fd_rotations"])
    sigma = job["sigma"]
    longest = job.get("work_longest")

    m_minus = sk.as_bool(d["cut_mask"])
    f = sk.fov_or_true(fov, m_minus.shape)

    # ---- measure the harm at the C1 working resolution --------------------
    # PVBM's multifractal FD is the bottleneck and scales with pixel count
    # (113 s per HRF mask at native resolution, DECISIONS.md).  C1 already
    # perturbs HRF/FIVES at longest side 1536, so measuring here at the same
    # scale is the *same* convention, not a new approximation -- and
    # ``_harm_row`` converts the length-like |dB| back to native units with
    # 1/s before dividing by the native training-split sigma, so R_false stays
    # in the units R_miss was fitted in.
    h0, w0 = m_minus.shape[:2]
    s_work = (float(longest) / float(max(h0, w0))
              if longest and max(h0, w0) != int(longest) else 1.0)
    img_w = _to_work_scale(img, longest, False) if s_work != 1.0 else img
    f_w = _to_work_scale(f, longest, True) if s_work != 1.0 else f
    gt_w = _to_work_scale(gt, longest, True) if s_work != 1.0 else gt
    mm_w = _to_work_scale(m_minus, longest, True) if s_work != 1.0 else m_minus

    disc = auto_disc(img_w, f_w, vessel_mask=mm_w)
    bio_gt = biomarker_row(gt_w, f_w, img_w, disc=disc, pipelines=bio,
                           fd_rotations=fd_rot)
    bio_minus = biomarker_row(mm_w, f_w, img_w, disc=disc, pipelines=bio,
                              fd_rotations=fd_rot)

    rows_meta = d["table"].to_dict("records")
    ok_idx = [j for j, r in enumerate(d["results"])
              if getattr(r, "ok", False) and len(getattr(r, "path_px", []))]
    budget = int(job.get("max_cand_bio") or 0)
    if budget and len(ok_idx) > budget:
        # R_false is fitted on the WRONG connections, so the budget goes mostly
        # to negatives; a third is reserved for true repairs, which keeps
        # ``only_wrong=False`` usable as a sensitivity check.
        pos = [j for j in ok_idx if int(d["y"][j]) == 1]
        neg = [j for j in ok_idx if int(d["y"][j]) == 0]
        n_pos = min(len(pos), max(1, budget // 3))
        keep_pos = (list(rng.choice(pos, size=n_pos, replace=False))
                    if len(pos) > n_pos else pos)
        room = budget - len(keep_pos)
        keep_neg = (list(rng.choice(neg, size=room, replace=False))
                    if room > 0 and len(neg) > room else neg[: max(room, 0)])
        ok_idx = sorted(int(j) for j in list(keep_pos) + list(keep_neg))

    rf_rows: List[Dict[str, Any]] = []
    for j in ok_idx:
        row = rows_meta[j]
        res = d["results"][j]
        tube = rasterize_tube(np.asarray(res.path_px, dtype=np.int64),
                              float(row["rad_bar"]), m_minus.shape)
        m_plus = m_minus | (tube & f)
        mp_w = _to_work_scale(m_plus, longest, True) if s_work != 1.0 else m_plus
        bio_plus = biomarker_row(mp_w, f_w, img_w, disc=disc, pipelines=bio,
                                 fd_rotations=fd_rot)
        a = (int(row["r_i"]), int(row["c_i"]))
        b = (int(row["r_j"]), int(row["c_j"]))
        # Contract-F features stay at NATIVE resolution: that is the schema the
        # BTR heads were fitted on and the one inference reconstructs.
        feats = phi(_EventPoints("bridge", a, b), img, m_plus, disc=None, fov=f)
        rec_row: Dict[str, Any] = dict(
            dataset=ds, image=key, subject_id=str(rec.get("subject_id", key)),
            cand_id=int(row["cand_id"]), is_true_repair=int(d["y"][j]),
            y=int(d["y"][j]), ok=1,
            path_len=float(res.length), rad_bar=float(row["rad_bar"]),
            energy=float(res.energy), d_over_rbar=float(row["d_over_rbar"]),
            is_port=int(row.get("is_port", 0)),
        )
        rec_row.update({c: float(feats.get(c, np.nan)) for c in PHI_COLUMNS})
        rec_row.update(_harm_row(bio_plus, bio_minus, bio_gt, sigma, pipes,
                                 work_scale=s_work))
        rf_rows.append(rec_row)

    out["rf_rows"] = rf_rows
    out["n_measured"] = len(ok_idx)
    out["work_scale"] = s_work
    out["seconds"] = time.time() - t0
    return out


def _safe_process(job):
    try:
        return process_image(job)
    except Exception as exc:                     # noqa: BLE001
        import traceback

        print("[build_data][ERROR] %s/%s: %s: %s"
              % (job.get("dataset"), job.get("index"), type(exc).__name__, exc),
              flush=True)
        traceback.print_exc()
        return dict(dataset=job.get("dataset"), image="?", index=job.get("index"),
                    n_cand=0, rf_rows=[], error=str(exc), seconds=0.0)


def build_pairs_and_rfalse(
    datasets,
    evidence_cfg,
    out_dir: str,
    seed: int = 0,
    n_cuts: int = 15,
    omega: float = 0.5,
    prior_path: Optional[str] = None,
    make_rfalse: bool = False,
    bio: str = "skan",
    fd_rotations: int = 5,
    max_cand_bio: int = 24,
    limit: Optional[int] = None,
    split_seed: Optional[int] = None,
    astar_kwargs: Optional[dict] = None,
    cand_kwargs: Optional[dict] = None,
    per_dataset_limit: Optional[Dict[str, int]] = None,
    rfalse_max_images: Optional[Dict[str, int]] = None,
    workers: int = 6,
    work_scale_bio: bool = True,
    verbose: bool = True,
):
    """Scorer pairs for every training image; optionally the R_false table too.

    The per-image work runs in a multiprocessing pool (``workers``); each worker
    is pinned to one thread and to the CPU, because the cost here is the two
    biomarker pipelines, not the micro head.  ``workers <= 1`` runs it in this
    process, which is what the smoke tests use.

    ``rfalse_max_images`` caps *how many images per dataset* contribute R_false
    rows (the scorer pairs still come from every image): FIVES needs 120 images
    of pairs but 40 images x 24 candidates is already ~1000 measured edits, far
    more than the head can use.
    """
    import pandas as pd

    from src.eval.biomarker_eval import load_gt_scales, pipes_for
    from src.seg import data as segdata

    recs = _training_records(datasets, split_seed, per_dataset_limit)
    if limit:
        recs = recs[: int(limit)]
    pipes = pipes_for(bio)
    ds_set = {d for d, _ in recs}
    sigma_by_ds = {ds: load_gt_scales(ds) for ds in ds_set}
    longest_by_ds = {ds: (segdata.dataset_cfg(ds)["resize_longest"]
                          if work_scale_bio else None) for ds in ds_set}

    caps = dict(rfalse_max_images or {})
    seen: Dict[str, int] = {}
    jobs: List[Dict[str, Any]] = []
    for i, (ds, rec) in enumerate(recs):
        do_rf = bool(make_rfalse)
        if do_rf and caps.get(ds):
            seen[ds] = seen.get(ds, 0) + 1
            do_rf = seen[ds] <= int(caps[ds])
        jobs.append(dict(
            dataset=ds, record=rec, index=i, seed=int(seed),
            n_cuts=int(n_cuts), omega=float(omega), prior_path=prior_path,
            astar_kwargs=astar_kwargs, cand_kwargs=cand_kwargs,
            make_rfalse=do_rf, bio=bio, fd_rotations=int(fd_rotations),
            max_cand_bio=int(max_cand_bio), sigma=sigma_by_ds.get(ds, {}),
            work_longest=longest_by_ds.get(ds),
            head_ckpt=evidence_cfg.get("head_ckpt"),
            evidence=evidence_cfg.get("evidence", "auto"),
            orientation=evidence_cfg.get("orientation", "learned")))

    n_rf_imgs = sum(1 for j in jobs if j["make_rfalse"])
    print("[build_data] %d training images (%d with the R_false measurement), "
          "%d worker(s), bio=%s fd_rotations=%d, work-scale harm: %s"
          % (len(jobs), n_rf_imgs, workers, bio, fd_rotations,
             ", ".join("%s->%s" % (k, v) for k, v in sorted(longest_by_ds.items()))),
          flush=True)

    # Set the BLAS/OpenMP limits in the PARENT, before the pool exists: a
    # spawned child imports numpy while the interpreter starts, which is before
    # any initializer can run, so setting them in ``_worker_setup`` alone was
    # too late -- measured 86 threads per worker, i.e. ~1500 threads on 80
    # cores once two datasets were building at once.  The child inherits this
    # environment, so the limit is in force at its first numpy import.
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    if int(workers) <= 1:
        _worker_setup()
        for k, j in enumerate(jobs):
            r = _safe_process(j)
            results.append(r)
            if verbose:
                print("  [%d/%d] %-18s %-10s cand=%3d pos=%3d rf=%3d  %.1fs "
                      "(total %.1fs)"
                      % (k + 1, len(jobs), r.get("image"), r.get("dataset"),
                         r.get("n_cand", 0), r.get("n_pos", 0),
                         len(r.get("rf_rows", [])), r.get("seconds", 0.0),
                         time.time() - t0), flush=True)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(int(workers), initializer=_worker_setup) as pool:
            for k, r in enumerate(pool.imap(_safe_process, jobs)):
                results.append(r)
                if verbose:
                    print("  [%d/%d] %-18s %-10s cand=%3d pos=%3d rf=%3d  %.1fs "
                          "(total %.1fs)"
                          % (k + 1, len(jobs), r.get("image"), r.get("dataset"),
                             r.get("n_cand", 0), r.get("n_pos", 0),
                             len(r.get("rf_rows", [])), r.get("seconds", 0.0),
                             time.time() - t0), flush=True)

    results.sort(key=lambda r: int(r.get("index", 0)))
    Xs, ys, img_ids, ds_ids, subj_ids, cand_ids = [], [], [], [], [], []
    rf_rows: List[Dict[str, Any]] = []
    for r in results:
        rf_rows += list(r.get("rf_rows", []))
        if not r.get("n_cand"):
            continue
        Xs.append(r["X"])
        ys.append(r["y"])
        n = len(r["y"])
        img_ids += [r["image"]] * n
        ds_ids += [r["dataset"]] * n
        subj_ids += [r["subject_id"]] * n
        cand_ids += list(r["cand_id"])

    if not Xs:
        raise SystemExit("no training candidates were generated")
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)

    os.makedirs(out_dir, exist_ok=True)
    seed_dir = os.path.join(out_dir, "seed%d" % int(seed))
    os.makedirs(seed_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(seed_dir, PAIRS_FILE),
        X=X, y=y, image=np.asarray(img_ids), dataset=np.asarray(ds_ids),
        subject_id=np.asarray(subj_ids), cand_id=np.asarray(cand_ids, dtype=np.int64))
    print("[build_data] pairs: X%s  positives %d/%d -> %s  (%.1fs)"
          % (X.shape, int(y.sum()), len(y), os.path.join(seed_dir, PAIRS_FILE),
             time.time() - t0), flush=True)

    rf = None
    if make_rfalse:
        rf = pd.DataFrame(rf_rows)
        rf.to_csv(os.path.join(out_dir, RFALSE_FILE), index=False)
        print("[build_data] deployed R_false table: %d rows from %d images "
              "(%d wrong connections) -> %s"
              % (len(rf), n_rf_imgs,
                 int((rf["is_true_repair"] == 0).sum()) if len(rf) else 0,
                 os.path.join(out_dir, RFALSE_FILE)), flush=True)
    return X, y, rf


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    from src.rigr import astar as astar_mod
    from src.seg import data as segdata

    ap = argparse.ArgumentParser(
        prog="python -m src.rigr.build_data",
        description="S4 step B: failure prior + scorer pairs + deployed-R_false table")
    ap.add_argument("--dataset", default=None,
                    help="one dataset; use --datasets for the LODO union")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated source datasets (LODO: the three that "
                         "are NOT held out)")
    ap.add_argument("--oof_dir", default=None,
                    help="runs/seg_oof/<ds>/pred for a single dataset")
    ap.add_argument("--oof_root", default=os.path.join("runs", "seg_oof"),
                    help="root holding <ds>/pred for every dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--head_ckpt", default=None)
    ap.add_argument("--evidence", default="auto", choices=["auto", "head", "frangi"])
    ap.add_argument("--orientation", default="learned", choices=["learned", "uniform"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--omega", type=float, default=0.5,
                    help="fraction of failure-conditioned severances (S4: 0.5)")
    ap.add_argument("--n_cuts", type=int, default=15)
    ap.add_argument("--mu", type=float, default=astar_mod.MU_KAPPA)
    ap.add_argument("--lambda_theta", type=float, default=astar_mod.LAMBDA_THETA)
    ap.add_argument("--rfalse", action="store_true",
                    help="also build the deployed R_false training table")
    ap.add_argument("--bio", default="skan", choices=["skan", "pvbm", "both"])
    ap.add_argument("--fd_rotations", type=int, default=5,
                    help="PVBM multifractal rotations. 5 is the C1 setting "
                         "(DECISIONS.md: bit-identical to 25 rotations and "
                         "2-3x faster; 4 does not match)")
    ap.add_argument("--workers", type=int, default=6,
                    help="multiprocessing pool over training images; each "
                         "worker is single-threaded and CPU-only")
    ap.add_argument("--rfalse_max_images", default=None,
                    help="cap the images that contribute R_false rows, "
                         "'fives=40' or a bare int for every dataset; the "
                         "scorer pairs still come from every image")
    ap.add_argument("--no_work_scale_bio", action="store_true",
                    help="measure the R_false harm at native resolution "
                         "instead of the C1 working scale (slow; the default "
                         "converts length-like |dB| back to native units)")
    ap.add_argument("--max_cand_bio", type=int, default=24,
                    help="candidates per image measured for R_false (0 = all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the pooled training list (after --per_dataset_limit)")
    ap.add_argument("--per_dataset_limit", default=None,
                    help="'fives=120,hrf=13': cap each dataset's training split "
                         "independently, taking the first n of the fixed split")
    ap.add_argument("--split-seed", dest="split_seed", type=int, default=None)
    ap.add_argument("--skip_prior", action="store_true",
                    help="reuse <out>/failure_prior.json instead of refitting")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    names = ([s.strip() for s in args.datasets.split(",") if s.strip()]
             if args.datasets else [args.dataset])
    if not names or names == [None]:
        raise SystemExit("give --dataset or --datasets")
    names = [segdata.canon(n) for n in names]

    oof_dirs = {}
    for ds in names:
        if args.oof_dir and len(names) == 1:
            oof_dirs[ds] = args.oof_dir
        else:
            oof_dirs[ds] = os.path.join(args.oof_root, ds, "pred")

    os.makedirs(args.out, exist_ok=True)
    prior_path = os.path.join(args.out, PRIOR_FILE)

    # ---- pi ---------------------------------------------------------------
    prior = None
    if args.omega > 0:
        from src.rigr.synth_cuts import FailurePrior

        if os.path.exists(prior_path) and (args.skip_prior or not args.force):
            prior = FailurePrior.load(prior_path)
            print("[build_data] reusing %s (%d gaps from %d images)"
                  % (prior_path, prior.n_events, prior.n_images), flush=True)
        else:
            prior, _ev = build_prior(names, oof_dirs, args.out)

    # ---- evidence ---------------------------------------------------------
    from src.rigr.run_rigr import make_evidence_fn

    evidence_fn, ev_name = make_evidence_fn(args.head_ckpt, args.gpu, args.evidence,
                                            orientation=args.orientation)
    astar_kwargs: Dict[str, Any] = {}
    if float(args.mu) != float(astar_mod.MU_KAPPA):
        astar_kwargs["mu"] = float(args.mu)
    if float(args.lambda_theta) != float(astar_mod.LAMBDA_THETA):
        astar_kwargs["lam"] = float(args.lambda_theta)
    print("[build_data] datasets=%s seed=%d evidence=%s omega=%.2f astar=%s"
          % (",".join(names), args.seed, ev_name, args.omega,
             astar_kwargs or "defaults"), flush=True)

    make_rfalse = bool(args.rfalse) and (
        args.force or not os.path.exists(os.path.join(args.out, RFALSE_FILE)))
    if args.rfalse and not make_rfalse:
        print("[build_data] %s exists; not rebuilding (use --force)"
              % os.path.join(args.out, RFALSE_FILE), flush=True)

    pdl = {}
    if args.per_dataset_limit:
        for part in args.per_dataset_limit.split(","):
            if not part.strip():
                continue
            k, _, v = part.partition("=")
            pdl[segdata.canon(k.strip())] = int(v)

    t0 = time.time()
    rf_caps: Dict[str, int] = {}
    if args.rfalse_max_images:
        spec = str(args.rfalse_max_images).strip()
        if "=" in spec:
            for part in spec.split(","):
                if part.strip():
                    k, _, v = part.partition("=")
                    rf_caps[segdata.canon(k.strip())] = int(v)
        else:
            rf_caps = {n: int(spec) for n in names}

    X, y, rf = build_pairs_and_rfalse(
        names,
        dict(head_ckpt=args.head_ckpt, evidence=args.evidence,
             orientation=args.orientation),
        args.out, seed=args.seed, n_cuts=args.n_cuts,
        omega=args.omega if prior is not None else 0.0,
        prior_path=(prior_path if prior is not None else None),
        rfalse_max_images=rf_caps, workers=int(args.workers),
        work_scale_bio=not args.no_work_scale_bio,
        make_rfalse=make_rfalse, bio=args.bio, fd_rotations=args.fd_rotations,
        max_cand_bio=args.max_cand_bio, limit=args.limit,
        split_seed=args.split_seed, astar_kwargs=astar_kwargs or None,
        per_dataset_limit=pdl)

    meta = dict(datasets=names, seed=int(args.seed), evidence=ev_name,
                head_ckpt=args.head_ckpt or "", omega=float(args.omega),
                n_cuts=int(args.n_cuts), mu=float(args.mu),
                lambda_theta=float(args.lambda_theta),
                oof_dirs=oof_dirs, split="train",
                limit=args.limit, per_dataset_limit=pdl,
                n_pairs=int(len(y)), n_positive=int(y.sum()),
                rfalse_rows=(int(len(rf)) if rf is not None else 0),
                rfalse_bio=args.bio if make_rfalse else "",
                rfalse_max_images=rf_caps, workers=int(args.workers),
                fd_rotations=int(args.fd_rotations),
                work_scale_bio=not args.no_work_scale_bio,
                rfalse_path_seed=(int(args.seed) if make_rfalse else None),
                max_cand_bio=int(args.max_cand_bio),
                seconds=round(time.time() - t0, 1))
    mp = os.path.join(args.out, "meta_seed%d.json" % int(args.seed))
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[build_data] done in %.1fs -> %s" % (time.time() - t0, mp), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
