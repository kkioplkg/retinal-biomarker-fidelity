"""E4 -- the pre-registered independent replication (DECISIONS 2026-09-17 13:15).

Primary replication set: **Fundus-AVSeg** (n = 100, FOV diameter 1238 / 2080 px,
four disease classes, CC BY 4.0, never used for any selection in this project).

Design, exactly as locked
-------------------------
5-fold cross-validation.  Fold 0's held-out set is the dataset's *official*
80/20 test split; folds 1-4 partition the remaining 80 images, stratified by
disease label, with a fixed RNG seed (2026), so that **every one of the 100
images is held out exactly once**.  For each fold a baseline U-Net is trained
with the FIVES/HRF recipe (``src/seg/train.py`` defaults: Adam 1e-3, poly(0.9),
BCE+Dice, 150 epochs x 100 iters, batch 4, patch 768, longest side 1536,
patience 60), and that fold's checkpoint is then fine-tuned for 50 epochs with
the three E2 arms ``{continued, reliseg, cfloss}`` -- the *same* argv builder
and the *same* hyper-parameters as ``src/pivot/e2_run.py`` (the arm definitions
are imported from it, never re-typed).  Each fold's 20 held-out images are
predicted by the four arms, giving 100 out-of-fold predictions per arm.

Endpoints
---------
primary    out-of-fold delta-r (reliseg - continued) for density / total_length
           / FD on the **skan** pipeline, paired bootstrap 2000 resamples,
           pooled over the 100 images;  secondary reference: the untouched
           baseline;  also reported: cfloss - continued.
safety     clDice / Dice non-inferiority (0.01 absolute), tortuosity / density
           delta-r >= -0.05, |delta bias| <= 0.25 sigma -- flagged by the very
           same ``src.pivot.e2_analysis.safety_table``.
secondary  four-class macro-AUC (GT / baseline / reliseg, plus continued and
           cfloss) over the 100 out-of-fold biomarker vectors, using the P3
           protocol (``src/pivot/p3_downstream.py``: standardised multinomial
           logreg and a small GBDT, 5-fold stratified CV repeated 3x, pooled
           out-of-fold macro one-vs-rest AUC), paired image bootstrap.

Secondary replication set: **MAPLES-DR** (162 resolvable images) -- zero-shot
only.  Every E2 main-grid checkpoint (FIVES + HRF x {baseline, continued,
reliseg, cfloss} x seeds 0-2) is applied untouched at the *source* convention
(longest side 1536, patch 768), and fidelity is reported stratified by FOV
diameter: the >= 1380 px stratum is the analysis stratum, the 909 px stratum is
the pre-declared low-resolution negative control.

Sigma
-----
Fundus-AVSeg has no row in ``results/gateA_biomarker_scales_train.csv``.  The
scale used here is the **pooled 100-image GT sigma**, 1.4826 * MAD of the
reference-mask biomarkers of all 100 images, measured natively -- the same
robust estimator as the Gate A table (``src/c1/train_scales.py::_mad_scale``)
and the same fallback ``src.pivot.e2_analysis.sigma_for`` already applies to
STARE.  It is stated in every table via the ``sigma_source`` column, and the
per-fold train-portion sigmas are reported alongside as a sensitivity check.
MAPLES-DR likewise uses its own 162-image GT sigma.

CLI
---
    python -m src.pivot.e4_replication folds
    python -m src.pivot.e4_replication train-base --fold 0 --gpu 0
    python -m src.pivot.e4_replication finetune --fold 0 --arm reliseg --gpu 0
    python -m src.pivot.e4_replication infer --fold 0 --arm reliseg --gpu 0
    python -m src.pivot.e4_replication maples-zeroshot --dataset fives \\
        --seed 0 --config reliseg --gpu 0
    python -m src.pivot.e4_replication bio --procs 8
    python -m src.pivot.e4_replication analyse
    python -m src.pivot.e4_replication gpu --gpu 0      # driver lane
    python -m src.pivot.e4_replication plan
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

EXP_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
PY = sys.executable

DATASET = "fundusavseg"
MAPLES = "maplesdr"
N_FOLDS = 5
FOLD_SEED = 2026
#: the fine-tune arms replicated here, in report order.  ``baseline`` is the
#: fold's own untouched base checkpoint (no fine-tuning), exactly as in E2.
ARMS = ("baseline", "continued", "reliseg", "cfloss")
FT_ARMS = ("continued", "reliseg", "cfloss")
#: E2 main-grid arms carried to the MAPLES-DR zero-shot sweep
MAPLES_CONFIGS = ("baseline", "continued", "reliseg", "cfloss")
MAPLES_SOURCES = ("fives", "hrf")
#: FOV-diameter stratum boundary (DECISIONS 2026-09-17 13:15)
MAPLES_STRATUM_PX = 1380
#: nominal FOV diameter per MESSIDOR frame size, as tabulated in
#: ``research/08_replication_datasets.md`` (the numbers the stratification was
#: pre-registered against).  Stratifying on these instead of on the per-image
#: measurement keeps one acquisition setting in one stratum: the 2240x1488
#: group measures 1376-1384 px, so a per-image cut at 1380 would split it in
#: half on mask jitter.  Measured diameters stay in the CSV as verification.
MAPLES_NOMINAL_DIAMETER = {(960, 1440): 909.0,
                           (1488, 2240): 1380.0,
                           (1536, 2304): 1452.0}
#: stratum labels (DECISIONS 2026-09-17 13:30 (9): "higher-resolution", not
#: "high-resolution" -- 1380-1452 px of retina sits between CHASE_DB1 and FIVES)
STRAT_HI = "higher_resolution"
STRAT_LO = "lower_resolution"

#: geometry of the replication sets -- the locked 1536 / 768 convention
E4_GEOMETRY = {"resize_longest": 1536, "patch": 768}
E4_BATCH = 4
E4_BASE_EPOCHS = 150
#: per-epoch validation cap for the fine-tune stage; the fold's val split has
#: 12 images, so this is a no-op kept only for parity with E2's FIVES lane.
E4_VAL_LIMIT = 12
#: single training seed per fold (the fold *is* the replication unit here)
E4_SEED = 0

#: ---- CMIG review point 16 (2026-09-18): extra fine-tune seeds ------------
#: The reviewer asks for more seeds behind the key null (ReliSeg vs the
#: same-budget ``continued`` control).  These are **additional fine-tune
#: seeds started from the very same per-fold base checkpoint**
#: (``fold<k>/base/best.pt``) with every other hyper-parameter -- arm
#: definition, 50 epochs, LR schedule, batch, geometry, val limit -- byte
#: identical; only ``--seed`` (which drives torch/numpy/augmentation RNG and
#: the patch sampler) changes.  Seed ``E4_SEED`` keeps its original,
#: un-suffixed directories and tags, so nothing already computed moves.
#:
#: This adds *units*, not analysis definitions: the primary endpoint is still
#: the seed-0 out-of-fold delta-r.  The extra seeds are measured through the
#: same bio stage and reported per seed.
#: 2026-09-18, after the methods consultation: the null claim needs 5 seeds per
#: arm per fold, so seeds 1-4 join the original seed 0 (4 extra x 2 arms x
#: 5 folds = 40 fine-tunes).  The per-fold BASE checkpoints are unchanged.
E4_EXTRA_SEEDS: Tuple[int, ...] = (1, 2, 3, 4)
#: only the two arms of the key null get the extra seeds
EXTRA_SEED_ARMS = ("continued", "reliseg")

RUN_ROOT = os.path.join("runs", "pivot", "e4")
QUEUE_DIR = os.path.join("runs", "pivot", "_e4_queue")
LOG_DIR = os.path.join(RUN_ROOT, "logs")
RESULT_DIR = os.path.join("results", "pivot", "e4")
FOLDS_JSON = os.path.join(RESULT_DIR, "folds.json")
#: the driver will not start a job unless the GPU has this much free memory
MIN_FREE_MB = 6144


# ==========================================================================
# 0 -- folds
# ==========================================================================
def build_folds() -> dict:
    """The pre-registered 5-fold partition of Fundus-AVSeg.

    Definition (deterministic, reproducible from this function alone):

    1. Records are ``src.data.datasets.load_dataset('fundusavseg')``, 100
       images, each with a disease label in {Normal, DR, AMD, Glaucoma}
       (40 / 20 / 20 / 20) taken from the filename code, and an official
       ``split`` from the deposit's ``training.txt`` / ``testing.txt``.
    2. **Fold 0** = the 20 images of the official *test* split, sorted by
       ``image_id``.  This makes the replication's first fold the dataset's own
       published protocol.
    3. The remaining 80 (the official *train* split) are dealt to folds 1-4:
       for each disease class in sorted order (AMD, DR, Glaucoma, Normal) the
       class members are sorted by ``image_id`` and shuffled with
       ``random.Random(2026)``; the shuffled members are handed out one at a
       time to folds ``1 + (counter % 4)`` where ``counter`` is a single
       running position that **continues across classes**.  Dealing 80 images
       this way gives every fold exactly 20 images, and within a class the
       fold counts differ by at most one -- i.e. a stratified random
       partition.
    4. Assertion: the five held-out sets are pairwise disjoint and their union
       is all 100 images, so every image is held out exactly once.

    For fold ``k`` the *training pool* is the other 80 images; the 15 % by
    subject validation split is then drawn from that pool with the project's
    standard ``DEFAULT_SPLIT_SEED = 12345``, exactly as a normal
    ``src.seg.train`` run does.
    """
    from src.data.datasets import load_dataset

    recs = sorted(load_dataset(DATASET), key=lambda r: str(r["image_id"]))
    if len(recs) != 100:
        raise RuntimeError("expected 100 Fundus-AVSeg records, got %d" % len(recs))

    official_test = sorted(str(r["image_id"]) for r in recs
                           if str(r["split"]).lower() == "test")
    pool = sorted(str(r["image_id"]) for r in recs
                  if str(r["split"]).lower() != "test")
    label = {str(r["image_id"]): str(r["disease"]) for r in recs}

    rng = random.Random(FOLD_SEED)
    folds: Dict[int, List[str]] = {0: list(official_test)}
    for f in range(1, N_FOLDS):
        folds[f] = []
    counter = 0
    for cls in sorted({label[i] for i in pool}):
        members = sorted(i for i in pool if label[i] == cls)
        rng.shuffle(members)
        for i in members:
            folds[1 + (counter % (N_FOLDS - 1))].append(i)
            counter += 1
    for f in folds:
        folds[f] = sorted(folds[f])

    seen: Dict[str, int] = {}
    for f, ids in folds.items():
        for i in ids:
            if i in seen:
                raise RuntimeError("image %s held out in folds %d and %d"
                                   % (i, seen[i], f))
            seen[i] = f
    if len(seen) != 100:
        raise RuntimeError("folds cover %d images, not 100" % len(seen))

    import collections
    out = {
        "dataset": DATASET,
        "n_images": len(recs),
        "n_folds": N_FOLDS,
        "seed": FOLD_SEED,
        "split_seed_for_val": 12345,
        "cv_unit": "image",
        "cv_unit_note": (
            "Image-level cross-validation.  Fundus-AVSeg publishes no patient "
            "identifier -- only a left/right eye flag -- so two images of the "
            "same patient cannot be detected and both-eye leakage across folds "
            "cannot be excluded.  This is stated as a limitation rather than "
            "silently assumed away (DECISIONS 2026-09-17 13:30)."),
        "partition_note": (
            "The 80 non-official-test images are partitioned ONCE into four "
            "disjoint 20-image folds; nothing is resampled or re-drawn."),
        "geometry": dict(E4_GEOMETRY),
        "definition": build_folds.__doc__.strip(),
        "official_test_n": len(official_test),
        "folds": {str(f): folds[f] for f in sorted(folds)},
        "fold_sizes": {str(f): len(folds[f]) for f in sorted(folds)},
        "fold_class_counts": {
            str(f): dict(sorted(collections.Counter(label[i] for i in ids).items()))
            for f, ids in sorted(folds.items())},
        "label": label,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generator": "src/pivot/e4_replication.py::build_folds",
    }
    return out


def load_folds() -> dict:
    if not os.path.exists(FOLDS_JSON):
        raise SystemExit("missing %s -- run `python -m src.pivot.e4_replication "
                         "folds` first" % FOLDS_JSON)
    with open(FOLDS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def held_out(fold: int) -> List[str]:
    return list(load_folds()["folds"][str(int(fold))])


def cmd_folds(args) -> int:
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = build_folds()
    with open(FOLDS_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("wrote %s" % FOLDS_JSON)
    for f in sorted(out["folds"], key=int):
        print("  fold %s: n=%2d  %s" % (f, len(out["folds"][f]),
                                        out["fold_class_counts"][f]))
    return 0


# ==========================================================================
# 1 -- dataset plumbing (in-process, so the training code stays untouched)
# ==========================================================================
def install_geometry() -> None:
    """Teach ``src.seg.data`` the two replication sets, at runtime only.

    ``DATASET_CFG`` / ``DEFAULT_BATCH`` / ``DEFAULT_EPOCHS`` are mutated in the
    *imported module object* of this process.  Nothing is written to disk, so
    no other script's loops over those tables change (the replication sets are
    deliberately absent from ``src.data.datasets.DATASETS`` for the same
    reason, DECISIONS 2026-09-17 13:00).
    """
    from src.seg import data as segdata

    for name in (DATASET, MAPLES):
        segdata.DATASET_CFG.setdefault(name, dict(E4_GEOMETRY))
        segdata.DEFAULT_BATCH.setdefault(name, E4_BATCH)
        segdata.DEFAULT_EPOCHS.setdefault(name, E4_BASE_EPOCHS)


def install_fold(fold: int) -> List[str]:
    """Make ``src.seg.data.get_records('fundusavseg')`` return fold-``k`` splits.

    The 20 held-out images get ``split='test'``, the other 80 ``split='train'``.
    Every consumer downstream (``src.seg.train``, ``src.pivot.p5_finetune``,
    ``src.pivot.p5_eval infer``) then builds its train / val / test through the
    *unmodified* ``make_splits`` with the project's standard split seed, so the
    fold is the only thing that changes.
    """
    install_geometry()
    from src.seg import data as segdata

    ids = set(held_out(fold))
    original = segdata.get_records

    def patched(name: str):
        recs = original(name)
        if segdata.canon(name) != DATASET:
            return recs
        out = []
        for r in recs:
            r = dict(r)
            r["split"] = "test" if str(r["image_id"]) in ids else "train"
            out.append(r)
        return out

    segdata.get_records = patched
    return sorted(ids)


def install_maples_all() -> None:
    """Make every MAPLES-DR record a 'test' record (zero-shot over all 162)."""
    install_geometry()
    from src.seg import data as segdata

    original = segdata.get_records

    def patched(name: str):
        recs = original(name)
        if segdata.canon(name) != MAPLES:
            return recs
        out = []
        for r in recs:
            r = dict(r)
            r["split"] = "test"
            out.append(r)
        return out

    segdata.get_records = patched


# ==========================================================================
# 2 -- paths
# ==========================================================================
def fold_dir(fold: int) -> str:
    return os.path.join(RUN_ROOT, "fold%d" % int(fold))


def base_dir(fold: int) -> str:
    return os.path.join(fold_dir(fold), "base")


def _seed_suffix(seed: int) -> str:
    """``""`` for the original seed, ``"_s<k>"`` for a review-16 extra seed.

    Keeping the default empty is what guarantees every directory, tag and
    biomarker-cache key produced before 2026-09-18 stays exactly where it was.
    """
    return "" if int(seed) == E4_SEED else "_s%d" % int(seed)


def arm_dir(fold: int, arm: str, seed: int = E4_SEED) -> str:
    if arm == "baseline":
        return base_dir(fold)          # the base model has one seed by design
    return os.path.join(fold_dir(fold), arm + _seed_suffix(seed))


def arm_ckpt(fold: int, arm: str, seed: int = E4_SEED) -> str:
    """Checkpoint E4 evaluates for ``arm``.

    ``baseline`` -- the fold's own base model, ``best.pt``, which is also the
    starting point of every fine-tune (E2 starts from ``best.pt`` too);
    every fine-tuned arm -- ``last.pt`` after the full 50-epoch schedule, the
    pre-registered primary checkpoint.
    """
    if arm == "baseline":
        return os.path.join(base_dir(fold), "best.pt")
    return os.path.join(arm_dir(fold, arm, seed), "last.pt")


def infer_dep(fold: int, arm: str, seed: int = E4_SEED) -> str:
    """What an inference job must WAIT FOR -- not what it loads.

    BUG FIXED 2026-09-18: this used to be ``arm_ckpt`` (= ``<arm>/last.pt``).
    ``p5_finetune`` rewrites ``last.pt`` after **every** epoch, so that file
    exists from epoch 0 and the dependency was satisfied the moment training
    started.  With two GPU lanes sharing the queue, the second lane claimed the
    inference job while the first was still fine-tuning, and evaluated a
    partially trained checkpoint.  It really happened: fold 4 seed 0 was
    evaluated at epoch 24 (cfloss), 38 (continued) and 39 (reliseg) instead of
    49, ~10 min before training finished.

    ``summary.json`` is written once, after the last epoch, so it is the only
    safe gate.  ``baseline`` is not fine-tuned -- it evaluates the fold's base
    model, whose own ``summary.json`` is the right gate for the same reason.
    """
    if arm == "baseline":
        return os.path.join(base_dir(fold), "summary.json")
    return os.path.join(arm_dir(fold, arm, seed), "summary.json")


def pred_dir(fold: int, arm: str, seed: int = E4_SEED) -> str:
    return os.path.join(arm_dir(fold, arm, seed), "pred")


def tag_of(fold: int, arm: str, seed: int = E4_SEED) -> str:
    return "e4_f%d_%s%s" % (int(fold), arm, _seed_suffix(seed))


def extra_seed_units() -> List[Tuple[int, str, int]]:
    """(fold, arm, seed) of every review-16 extra fine-tune unit."""
    return [(k, arm, s) for s in E4_EXTRA_SEEDS
            for k in range(N_FOLDS) for arm in EXTRA_SEED_ARMS]


def maples_tag(ds: str, seed: int, cfg: str) -> str:
    return "e4mp_%s_s%d_%s" % (ds, int(seed), cfg)


def maples_dir(ds: str, seed: int, cfg: str) -> str:
    return os.path.join(RUN_ROOT, "maples", maples_tag(ds, seed, cfg))


def gt_tag() -> str:
    return "e4_gt"


def gt_csv(dataset: str = DATASET) -> str:
    return os.path.join("results", "pivot", "e4_gt_%s.csv" % dataset)


# ==========================================================================
# 3 -- stages
# ==========================================================================
def _replace_argv(argv: Sequence[str], repl: Dict[str, str]) -> List[str]:
    """Return ``argv`` with the value of each flag in ``repl`` replaced."""
    out = list(argv)
    for flag, value in repl.items():
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out += [flag, value]
    return out


def cmd_train_base(args) -> int:
    """Fold-``k`` baseline U-Net -- the FIVES/HRF recipe, unchanged.

    Every hyper-parameter is ``src.seg.train``'s own default (Adam 1e-3,
    poly(0.9), BCE 1.0 + Dice 1.0 + clDice 0.0, 100 iters/epoch, patience 60,
    oversample 0.7, AMP, augmentation on); only ``--epochs`` (150, the
    HRF/FIVES value, supplied explicitly so it does not depend on the runtime
    patch of ``DEFAULT_EPOCHS``), ``--batch-size`` (4) and the geometry
    (longest 1536 / patch 768, via :func:`install_geometry`) are named.
    """
    install_fold(args.fold)
    from src.seg import train as segtrain

    out = base_dir(args.fold)
    argv = ["--dataset", DATASET, "--seed", str(E4_SEED), "--gpu", str(args.gpu),
            "--epochs", str(E4_BASE_EPOCHS), "--batch-size", str(E4_BATCH),
            "--out", out]
    print("[e4] train-base fold=%d -> %s\n     argv %s"
          % (args.fold, out, " ".join(argv)), flush=True)
    return segtrain.main(argv)


def finetune_argv(fold: int, arm: str, gpu: int, seed: int = E4_SEED) -> List[str]:
    """The E2 fine-tune argv, re-pointed at this fold.

    ``src.pivot.e2_run._train_argv`` is the single source of truth for the arm
    hyper-parameters (loss kind, lambda, clDice weight, terms, FD ladder,
    cf-scale, 50 epochs).  It is called for the FIVES lane and only the
    dataset / seed / checkpoint / output / val-limit values are substituted,
    so an arm can never drift from its E2 definition.
    """
    from src.pivot import e2_run as E2

    if arm not in E2.CONFIGS or E2.CONFIGS[arm] is None:
        raise SystemExit("arm %r is not a fine-tuned E2 config" % arm)
    argv = list(E2._train_argv("fives", 0, arm, E2.CONFIGS[arm]))
    # ``_train_argv`` returns the full ``python -m src.pivot.p5_finetune ...``
    # vector; E4 calls ``p5_finetune.main`` in-process, so drop the module part
    # and keep everything after it byte-for-byte.
    if argv[:1] == ["-m"]:
        argv = argv[2:]
    return _replace_argv(argv, {
        "--dataset": DATASET,
        # the ONLY value that differs between the extra-seed units and the
        # original ones (review point 16): p5_finetune's --seed drives
        # set_determinism, the patch sampler and the augmentation RNG.
        "--seed": str(int(seed)),
        "--ckpt": os.path.join(base_dir(fold), "best.pt"),
        "--out": arm_dir(fold, arm, seed),
        "--val-limit": str(E4_VAL_LIMIT),
        "--gpu": str(gpu),
    })


def cmd_finetune(args) -> int:
    install_fold(args.fold)
    from src.pivot import p5_finetune

    seed = int(getattr(args, "seed", E4_SEED))
    argv = finetune_argv(args.fold, args.arm, args.gpu, seed)
    print("[e4] finetune fold=%d arm=%s seed=%d\n     argv %s"
          % (args.fold, args.arm, seed, " ".join(argv)), flush=True)
    return p5_finetune.main(argv)


def cmd_infer(args) -> int:
    install_fold(args.fold)
    from src.pivot import p5_eval

    seed = int(getattr(args, "seed", E4_SEED))
    ck = arm_ckpt(args.fold, args.arm, seed)
    root = pred_dir(args.fold, args.arm, seed)
    argv = ["infer", "--dataset", DATASET, "--ckpt", ck,
            "--tag", tag_of(args.fold, args.arm, seed), "--root", root,
            "--gpu", str(args.gpu)]
    print("[e4] infer fold=%d arm=%s seed=%d ckpt=%s -> %s"
          % (args.fold, args.arm, seed, ck, root), flush=True)
    return p5_eval.main(argv)


def cmd_maples_zeroshot(args) -> int:
    """One E2 checkpoint applied, untouched, to all 162 MAPLES-DR images."""
    install_maples_all()
    from src.pivot import e2_run as E2
    from src.pivot import p5_eval

    ck = E2.eval_ckpt(args.dataset, args.seed, args.config, "last")
    root = maples_dir(args.dataset, args.seed, args.config)
    argv = ["infer", "--dataset", MAPLES, "--ckpt", ck,
            "--tag", maples_tag(args.dataset, args.seed, args.config),
            "--root", root,
            "--infer-longest", str(E2.SRC_LONGEST),
            "--infer-patch", str(E2.SRC_PATCH),
            "--gpu", str(args.gpu)]
    print("[e4] maples-zeroshot %s s%d %s ckpt=%s -> %s"
          % (args.dataset, args.seed, args.config, ck, root), flush=True)
    return p5_eval.main(argv)


# ==========================================================================
# 4 -- biomarkers (CPU)
# ==========================================================================
def maples_preannot_path(rec, force: bool = False) -> Optional[str]:
    """The deposit's network PRE-annotation, mapped back to native geometry.

    ``AdditionalData/preannotations/Vessels/<id>.png`` is what the annotation
    network produced before the retinologists corrected it -- the anchor of the
    annotation-convention sensitivity analysis (DECISIONS 2026-09-17 13:30 (9)).

    **The deposit ships them on a fixed 1500x1500 canvas**, not on the native
    MESSIDOR frame (1440x960 / 2240x1488 / 2304x1536), so they cannot be used
    against a native FOV as-is.  The canvas is the square whose side is the FOV
    diameter, centred on the FOV: a grid search over scale (0.94-1.18 x FOV
    width) and centre offset (+-30 px horizontally, +-40 px vertically) put the
    Dice optimum of all eight probed images at exactly scale 1.00, offset (0, 0)
    -- so the placement is identified, not fitted.  The mask is resampled back
    with nearest neighbour and cached under ``data/maplesdr/preannot_native/``.

    Caveat carried into the report: nearest-neighbour downsampling of a
    1500-px canvas onto a 1380-1452-px square thins the finest vessels a
    little, which is the very structure the biomarkers measure.  The
    pre-annotation arm is therefore a **directional** sensitivity check on the
    annotation convention, not a precise second anchor.  (Mean Dice of the
    remapped pre-annotation against the final mask is 0.726 over the 162
    resolvable images; the 0.783 in `research/08` was measured on the
    1500x1500 canvas, where no resampling loss applies.)
    """
    import cv2
    import numpy as np

    from src.data.datasets import read_image
    from src.pivot.common import binarize, load_fov

    iid = str(rec["image_id"])
    src = os.path.join("data", "maplesdr", "raw", "AdditionalData",
                       "preannotations", "Vessels", iid + ".png")
    if not os.path.exists(src):
        return None
    out = os.path.join("data", "maplesdr", "preannot_native", iid + ".png")
    if os.path.exists(out) and not force:
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    gt = binarize(read_image(str(rec["label_path"])))
    h, w = gt.shape[:2]
    fov = load_fov(rec, (h, w)) > 0
    ys, xs = np.where(fov)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    side = int(x1 - x0)
    pa = binarize(read_image(src)).astype(np.uint8) * 255
    sq = cv2.resize(pa, (side, side), interpolation=cv2.INTER_NEAREST) > 127
    cy = (y0 + y1) // 2
    Y0 = max(0, cy - side // 2)
    m = np.zeros((h, w), bool)
    hh = min(side, h - Y0)
    m[Y0:Y0 + hh, x0:x0 + side] = sq[:hh, :min(side, w - x0)]
    cv2.imwrite(out, m.astype(np.uint8) * 255)
    return out


def _gt_bio(dataset: str, tag: str, procs: int, ids: Optional[Sequence[str]] = None,
            label_of=None, out_name: Optional[str] = None) -> str:
    """GT biomarkers of a replication set, through the p5_eval machinery.

    Identical estimator path to ``p5_eval.cmd_bio`` / ``cmd_gtbio``
    (``compute_all(fd_rotations=5)`` with the optic disc detected from the
    fundus image alone); only the record selection differs -- E4 needs *all*
    images of the set, not one official test split.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from src.pivot import p5_eval
    from src.pivot.common import records

    recs = sorted(records(dataset), key=lambda r: str(r["image_id"]))
    if ids is not None:
        keep = set(ids)
        recs = [r for r in recs if str(r["image_id"]) in keep]
    tasks = []
    for r in recs:
        lp = str(r["label_path"]) if label_of is None else label_of(r)
        if lp:
            tasks.append((dataset, r, tag, lp))
    print("[e4] gtbio %s: %d images tag=%s" % (dataset, len(tasks), tag), flush=True)
    with ProcessPoolExecutor(max_workers=procs) as ex:
        futs = [ex.submit(p5_eval._bio_one, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            iid, dt = fu.result()
            print("[%3d/%d] %s %ss" % (i + 1, len(tasks), iid, dt), flush=True)
    out = gt_csv(out_name or dataset)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p5_eval.load_tag(dataset, tag).to_csv(out, index=False)
    print("[e4] wrote %s" % out, flush=True)
    return out


def bio_units(include_maples: bool = True) -> List[Tuple[str, str, str]]:
    """(dataset, tag, root) for every prediction directory E4 must measure."""
    out: List[Tuple[str, str, str]] = []
    for k in range(N_FOLDS):
        for arm in ARMS:
            out.append((DATASET, tag_of(k, arm), pred_dir(k, arm)))
    # review point 16 (2026-09-18): the extra fine-tune seeds of the key null.
    # Appended AFTER the original units so an interrupted pass still measures
    # the pre-registered ones first; they are separate tags, so nothing merges.
    for k, arm, s in extra_seed_units():
        out.append((DATASET, tag_of(k, arm, s), pred_dir(k, arm, s)))
    if include_maples:
        for ds in MAPLES_SOURCES:
            for cfg in MAPLES_CONFIGS:
                for s in (0, 1, 2):
                    out.append((MAPLES, maples_tag(ds, s, cfg),
                                maples_dir(ds, s, cfg)))
    return out


def cmd_bio(args) -> int:
    from src.pivot import p5_eval

    done, skipped, gt_failed = 0, [], []

    def _try_gt(name, fn):
        """Run one GT stage; a failure must not block the rest of the pass.

        The reference-mask stages are independent of each other and of the
        per-arm measurements.  A crash in the *optional* pre-annotation anchor
        used to abort the whole pass, so nothing at all got measured (it did,
        on 2026-09-17 between 14:40 and 14:47).  Now it is recorded and the
        pass carries on.
        """
        try:
            fn()
        except Exception as exc:                                # noqa: BLE001
            import traceback
            gt_failed.append("%s: %r" % (name, exc))
            print("[e4] GT stage %s FAILED (continuing): %r" % (name, exc),
                  flush=True)
            traceback.print_exc()

    if not args.skip_gt:
        if not os.path.exists(gt_csv(DATASET)) or args.force:
            _try_gt("fundusavseg", lambda: _gt_bio(DATASET, gt_tag(), args.procs))
        if args.maples and (not os.path.exists(gt_csv(MAPLES)) or args.force):
            install_maples_all()
            _try_gt("maplesdr", lambda: _gt_bio(MAPLES, gt_tag() + "_maples",
                                                args.procs))
        if args.preannot and (not os.path.exists(gt_csv(MAPLES + "_preannot"))
                              or args.force):
            install_maples_all()
            _try_gt("maplesdr_preannot",
                    lambda: _gt_bio(MAPLES, gt_tag() + "_maples_preannot",
                                    args.procs, label_of=maples_preannot_path,
                                    out_name=MAPLES + "_preannot"))
    for dataset, tag, root in bio_units(include_maples=args.maples):
        if args.only and args.only not in tag:
            continue
        if dataset == MAPLES and not args.maples:
            continue
        out = os.path.join(root, "bio.csv")
        if os.path.exists(out) and not args.force:
            continue
        if not os.path.exists(os.path.join(root, "manifest.csv")):
            skipped.append(tag)
            continue
        print("\n[e4] bio %s tag=%s" % (dataset, tag), flush=True)
        rc = p5_eval.main(["bio", "--dataset", dataset, "--tag", tag,
                           "--root", root, "--procs", str(args.procs)])
        if rc != 0:
            print("[e4] FAILED bio %s rc=%s" % (tag, rc), flush=True)
            return rc
        done += 1
    print("[e4] bio: %d directories measured, %d not ready (%s)%s"
          % (done, len(skipped), ",".join(skipped[:8]),
             "; GT failures: " + "; ".join(gt_failed) if gt_failed else ""),
          flush=True)
    return 0


# ==========================================================================
# 5 -- MAPLES-DR FOV strata
# ==========================================================================
def maples_strata(force: bool = False) -> "object":
    """Per-image FOV diameter and stratum for the 162 resolvable MAPLES-DR images.

    The diameter is the longer side of the FOV mask's bounding box (the same
    quantity ``research/08_replication_datasets.md`` tabulates: 909 px for the
    1440x960 MESSIDOR frames, >= 1380 px for the larger ones).  The >= 1380 px
    stratum is the pre-registered analysis stratum; the 909 px stratum is the
    pre-declared low-resolution negative control (DECISIONS 2026-09-17 13:15).

    The stratum is assigned per **native frame size** (the acquisition
    setting), using that group's *median* FOV diameter, not per image: the
    2240x1488 group measures 1376-1384 px and a per-image cut at 1380 would
    split one acquisition setting down the middle on +-4 px of mask jitter.
    By frame size the strata are 1440x960 -> 909 px (91 images) and
    {2240x1488, 2304x1536} -> >= 1380 px (71 images), which is the 71-image
    analysis stratum the pre-registration names.
    """
    import numpy as np
    import pandas as pd

    out = os.path.join(RESULT_DIR, "maples_strata.csv")
    if os.path.exists(out) and not force:
        return pd.read_csv(out)
    from src.pivot.common import load_fov, records
    from src.data.datasets import read_image

    rows = []
    for r in sorted(records(MAPLES), key=lambda x: str(x["image_id"])):
        img = read_image(str(r["image_path"]))
        fov = load_fov(r, img.shape[:2]) > 0
        ys, xs = np.where(fov)
        dh = int(ys.max() - ys.min() + 1) if ys.size else 0
        dw = int(xs.max() - xs.min() + 1) if xs.size else 0
        d = max(dh, dw)
        rows.append(dict(image_id=str(r["image_id"]),
                         native_h=int(img.shape[0]), native_w=int(img.shape[1]),
                         fov_h=dh, fov_w=dw, fov_diameter=d))
    df = pd.DataFrame(rows)
    grp = df.groupby(["native_h", "native_w"])["fov_diameter"].median()
    df["group_median_diameter"] = [float(grp.loc[(h, w)]) for h, w in
                                   zip(df["native_h"], df["native_w"])]
    df["nominal_diameter"] = [float(MAPLES_NOMINAL_DIAMETER.get((h, w),
                                                               grp.loc[(h, w)]))
                              for h, w in zip(df["native_h"], df["native_w"])]
    # DECISIONS 2026-09-17 13:30 (9): the upper stratum is called
    # *higher-resolution*, never "high-resolution" -- 1380-1452 px of retina is
    # above CHASE_DB1 and below FIVES, not high in absolute terms.
    df["stratum"] = np.where(df["nominal_diameter"] >= MAPLES_STRATUM_PX,
                             STRAT_HI, STRAT_LO)
    df["stratum_px"] = np.where(df["nominal_diameter"] >= MAPLES_STRATUM_PX,
                                "ge%d" % MAPLES_STRATUM_PX,
                                "lt%d" % MAPLES_STRATUM_PX)
    os.makedirs(RESULT_DIR, exist_ok=True)
    df.to_csv(out, index=False)
    print("[e4] wrote %s (%d rows, %s)"
          % (out, len(df), df["stratum"].value_counts().to_dict()), flush=True)
    return df


# ==========================================================================
# 6 -- analysis
# ==========================================================================
def fold_boot_idx(fold_labels, n_boot: int, seed: int = 0):
    """Bootstrap resample matrix that **resamples within each outer fold**.

    DECISIONS 2026-09-17 13:30 (4): the 100 out-of-fold rows are not an i.i.d.
    sample -- each block of 20 was produced by a different fitted model.  The
    resample therefore preserves the fold structure (each fold contributes
    exactly its own 20 rows, drawn with replacement from within that fold), and
    the interval is reported as **conditional on the fixed folds and the fitted
    models**, not as an interval over re-training.
    """
    import numpy as np

    rng = np.random.RandomState(seed)
    lab = np.asarray(fold_labels)
    pos = {f: np.where(lab == f)[0] for f in np.unique(lab)}
    cols = []
    for f in sorted(pos):
        p = pos[f]
        cols.append(p[rng.randint(0, len(p), size=(n_boot, len(p)))])
    return np.concatenate(cols, axis=1)


def max_stat_replication(dboot, dhat, alpha: float = 0.05) -> dict:
    """Max-statistic bootstrap over the three primary delta-r endpoints.

    ``dboot`` maps endpoint -> (n_boot,) bootstrap delta-r, ``dhat`` endpoint ->
    point estimate.  The null distribution of ``max_j (delta_b^j - delta^j)``
    gives a single-step FWER-controlled adjusted p-value per endpoint
    (Westfall-Young style),

        p_adj(j) = P[ max_j' (delta_b^j' - delta^j') >= delta^j ].

    Pre-registered success rule (DECISIONS 2026-09-17 13:30 (6)): replication
    succeeds when at least one endpoint has ``p_adj < alpha`` **and** no
    endpoint runs in the opposite direction, i.e. no endpoint whose two-sided
    95 % percentile CI lies entirely below zero.  A CI containing zero is not
    evidence of no effect -- no equivalence margin was pre-specified.
    """
    import numpy as np

    keys = sorted(dboot)
    centred = np.vstack([dboot[k] - dhat[k] for k in keys])
    mx = centred.max(axis=0)
    p_adj = {k: float((mx >= dhat[k]).mean()) for k in keys}
    cis = {k: (float(np.percentile(dboot[k], 2.5)),
               float(np.percentile(dboot[k], 97.5))) for k in keys}
    opposite = [k for k in keys if cis[k][1] < 0]
    winners = [k for k in keys if p_adj[k] < alpha]
    return {"endpoints": keys, "p_adj": p_adj, "ci": cis,
            "alpha": alpha,
            "opposite_direction": opposite,
            "significant": winners,
            "replication_success": bool(winners and not opposite)}


def _sigma_from(frame, cols) -> Dict[str, float]:
    import numpy as np
    import pandas as pd

    out = {}
    for c in cols:
        v = pd.to_numeric(frame[c], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        s = float(1.4826 * np.median(np.abs(v - np.median(v)))) if v.size >= 3 else float("nan")
        out[c] = s if s > 0 else float("nan")
    return out


def _load_arm_oof(arm: str, seed: int = E4_SEED):
    """The 100 out-of-fold rows of one arm, concatenated over the five folds.

    ``seed`` selects which fine-tune seed of the arm to read (review point 16).
    ``baseline`` is not fine-tuned -- it is the fold's own base model -- so it
    is always read from the un-suffixed directory whatever ``seed`` says.
    """
    import pandas as pd

    from src.pivot.e2_analysis import rel

    parts, srcs = [], []
    for k in range(N_FOLDS):
        f = os.path.join(pred_dir(k, arm, E4_SEED if arm == "baseline" else seed),
                         "bio.csv")
        if not os.path.exists(f):
            return None, []
        d = pd.read_csv(f)
        d = d[d["image_id"].astype(str).isin(set(held_out(k)))].copy()
        d["fold"] = k
        parts.append(d)
        srcs.append(rel(f))
    out = pd.concat(parts, ignore_index=True)
    if out["image_id"].duplicated().any():
        raise RuntimeError("arm %s: duplicated out-of-fold image ids" % arm)
    return out, srcs


def per_image_sigma(gt, cols):
    """Per-outer-fold sigma, expanded to one value per image.

    DECISIONS 2026-09-17 13:30 (4): the scale of outer fold ``k`` is estimated
    from the reference-mask biomarkers of **that fold's 80 training images
    only** and applied to its 20 held-out images, so no image is standardised
    by a scale that saw it.  Returns ``(sigma_by_fold, sigma_per_image)`` where
    ``sigma_per_image[c]`` is a vector aligned with ``gt``'s row order.
    """
    import numpy as np

    by_fold = {int(k): _sigma_from(gt[gt["fold"] != k], cols)
               for k in sorted(gt["fold"].unique())}
    folds = gt["fold"].to_numpy()
    per_img = {c: np.array([by_fold[int(f)][c] for f in folds], dtype=float)
               for c in cols}
    return by_fold, per_img


def _class_centre(v, y):
    """Values centred on their disease-class mean (class-adjusted sensitivity)."""
    import numpy as np
    import pandas as pd

    s = pd.Series(np.asarray(v, dtype=float))
    return (s - s.groupby(pd.Series(np.asarray(y))).transform("mean")).to_numpy()


def cv_proba_aligned(X, y, kind, classes, fold_labels, n_repeat: int = 3,
                     seed: int = 0):
    """``p3_downstream.cv_proba`` with the outer folds pinned to the E4 folds.

    Same estimators and same standardisation as
    ``src/pivot/p3_downstream.py::cv_proba`` (standardised multinomial logistic
    regression, or the small HistGradientBoosting model), but the outer split
    is the segmentation fold partition rather than a fresh ``StratifiedKFold``
    -- so a classifier never sees a biomarker vector produced by the very
    segmenter it is being tested on (DECISIONS 2026-09-17 13:30 (7)).  The
    three repeats only vary the GBDT seed; logistic regression is
    deterministic and its repeats coincide.
    """
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lab = np.asarray(fold_labels)
    acc = np.zeros((len(y), len(classes)), dtype=float)
    for rep in range(n_repeat):
        for f in sorted(set(lab.tolist())):
            te = np.where(lab == f)[0]
            tr = np.where(lab != f)[0]
            if len(np.unique(y[tr])) < 2:
                continue
            if kind == "logreg":
                m = make_pipeline(StandardScaler(),
                                  LogisticRegression(max_iter=5000, C=1.0))
            else:
                m = HistGradientBoostingClassifier(
                    max_depth=3, max_iter=150, learning_rate=0.08,
                    min_samples_leaf=10, l2_regularization=1.0,
                    early_stopping=False, random_state=seed + rep)
            m.fit(X[tr], y[tr])
            p = m.predict_proba(X[te])
            order = [list(m.classes_).index(c) if c in list(m.classes_) else None
                     for c in classes]
            for j, o in enumerate(order):
                if o is not None:
                    acc[te, j] += p[:, o]
    return acc / n_repeat


#: the three pre-registered primary delta-r endpoints, on the skan pipeline
PRIMARY_ENDPOINTS = ("density", "total_length", "FD")


def fast_macro_auc_factory(y, classes, verify_proba=None):
    """Rank-based one-vs-rest macro AUC, identical to ``roc_auc_score``.

    The bootstrap evaluates the macro AUC 2000 x (number of arms) x (number of
    panels x classifiers) times; ``sklearn.metrics.roc_auc_score`` re-sorts and
    re-validates on every call, which dominates the analysis wall clock.  This
    computes the same quantity as the Mann-Whitney statistic with midranks for
    ties, which is exactly what ``roc_auc_score`` returns.

    The returned callable is **verified against sklearn** on ``verify_proba``
    before it is handed back; if the two ever disagree by more than 1e-9 the
    factory raises rather than silently substituting a different estimator.
    """
    import numpy as np
    from scipy.stats import rankdata

    y = np.asarray(y)
    masks = [(y == c) for c in classes]

    def auc(idx, proba):
        yy = [m[idx] for m in masks]
        out = []
        for j, m in enumerate(yy):
            n1 = int(m.sum())
            n0 = len(m) - n1
            if n1 == 0 or n0 == 0:
                continue
            r = rankdata(proba[idx, j])
            out.append((r[m].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))
        return float(np.mean(out)) if out else float("nan")

    if verify_proba is not None:
        from sklearn.metrics import roc_auc_score
        idx = np.arange(len(y))
        ref = []
        for j, c in enumerate(classes):
            yy = (y == c).astype(int)
            if 0 < yy.sum() < len(yy):
                ref.append(roc_auc_score(yy, verify_proba[:, j]))
        ref = float(np.mean(ref))
        got = auc(idx, verify_proba)
        if not np.isfinite(ref) or abs(ref - got) > 1e-9:
            raise RuntimeError("fast macro AUC disagrees with sklearn "
                               "(%r vs %r)" % (ref, got))
    return auc


def cmd_analyse_seeds(args) -> int:
    """Run the SAME per-seed analysis for every available fine-tune seed, then
    pool -- review point 16's "5 seeds per arm per fold".

    This deliberately does not re-implement anything: it calls
    :func:`cmd_analyse` once per seed (identical estimator, identical fold
    structure, identical replication rule) and then aggregates the resulting
    ``e4_delta[_s<k>].csv`` files with the SAME pooling rule ``e2_analysis``
    uses for E2 -- the mean of the per-seed delta-r, plus a 95 % t interval
    over the per-seed point estimates and the seed-level sign agreement.

    Seed 0's run rewrites the canonical artefacts, so the paper's E4 numbers
    are refreshed by the same command that produces the multi-seed table.
    """
    import numpy as np
    import pandas as pd
    from scipy import stats

    seeds = [s for s in (E4_SEED,) + tuple(E4_EXTRA_SEEDS)]
    have = []
    for sd in seeds:
        ok = all(os.path.exists(os.path.join(pred_dir(k, arm, sd), "bio.csv"))
                 for k in range(N_FOLDS) for arm in EXTRA_SEED_ARMS)
        if sd == E4_SEED:
            ok = ok and all(os.path.exists(os.path.join(pred_dir(k, a), "bio.csv"))
                            for k in range(N_FOLDS) for a in ARMS)
        if ok:
            have.append(sd)
        else:
            print("[e4] seed %d incomplete -- skipped" % sd, flush=True)
    if not have:
        raise SystemExit("no complete seed to analyse")

    for sd in have:
        print("[e4] ===== analyse seed %d =====" % sd, flush=True)
        a = argparse.Namespace(seed=sd, threads=getattr(args, "threads", 4),
                               skip_downstream=getattr(args, "skip_downstream",
                                                       False))
        rc = cmd_analyse(a)
        if rc != 0:
            return rc

    # ---- pool the per-seed delta-r ----
    parts = []
    for sd in have:
        f = os.path.join("results", "pivot",
                         "e4_delta%s.csv" % _seed_suffix(sd))
        d = pd.read_csv(f)
        d = d[d["scope"] == "oof_pooled"].copy()
        d["ft_seed"] = sd
        parts.append(d)
    alld = pd.concat(parts, ignore_index=True)

    rows = []
    keys = ["config", "reference", "biomarker", "pipeline"]
    for key, g in alld.groupby(keys):
        v = pd.to_numeric(g["d_r"], errors="coerce").to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        m = float(np.mean(v))
        if v.size > 1:
            se = float(stats.sem(v))
            t = float(stats.t.ppf(0.975, v.size - 1))
            lo, hi = m - t * se, m + t * se
        else:
            lo = hi = np.nan
        rows.append(dict(zip(keys, key), n_seeds=int(v.size),
                         seeds=",".join(str(x) for x in g["ft_seed"].tolist()),
                         d_r_mean=m, seed_lo=lo, seed_hi=hi,
                         n_seeds_positive=int((v > 0).sum()),
                         direction_consistent=("yes" if (v > 0).all() or
                                               (v < 0).all() else "no"),
                         per_seed=";".join("%d:%.6f" % (sd, x) for sd, x
                                           in zip(g["ft_seed"], g["d_r"]))))
    out = pd.DataFrame(rows).sort_values(keys)
    f = os.path.join("results", "pivot", "e4_delta_seeds.csv")
    out.to_csv(f, index=False)
    print("[e4] wrote %s (%d rows, seeds %s)"
          % (f, len(out), ",".join(str(s) for s in have)), flush=True)
    prim = out[(out["config"] == "reliseg") & (out["reference"] == "continued") &
               (out["pipeline"] == "skan")]
    with pd.option_context("display.width", 220):
        print("=== E4 primary contrast: reliseg - continued, skan, "
              "across %d fine-tune seeds ===" % len(have))
        print(prim[["biomarker", "n_seeds", "d_r_mean", "seed_lo", "seed_hi",
                    "n_seeds_positive", "direction_consistent",
                    "per_seed"]].to_string(index=False))
    return 0


def cmd_analyse(args) -> int:
    # The GBDT arm of the downstream protocol spreads over every core by
    # default; on this machine the E2/E4 biomarker workers and two GPU
    # trainings already saturate the CPU, and an unbounded OpenMP pool turns
    # a two-minute stage into half an hour of thread thrash (and starves the
    # biomarker stage).  Capped before sklearn is imported anywhere.
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = str(int(getattr(args, "threads", 4) or 4))

    import numpy as np
    import pandas as pd

    from src.pivot import e2_analysis as E2A
    from src.pivot.common import PRIMARY_COLS
    from src.pivot.e2_analysis import boot_r, ci, pearson, rel, spearman, split_col

    os.makedirs(RESULT_DIR, exist_ok=True)
    folds = load_folds()
    notes: List[str] = []
    # review point 16: the SAME analysis, run on one fine-tune seed at a time.
    # Seed 0 keeps every canonical filename, so the paper's artefacts are
    # rewritten in place; the extra seeds land beside them with a _s<k> suffix.
    a_seed = int(getattr(args, "seed", E4_SEED))
    SFX = _seed_suffix(a_seed)
    def _out(name: str) -> str:
        stem, ext = os.path.splitext(name)
        return stem + SFX + ext

    # ---------------- GT ----------------
    gtf = gt_csv(DATASET)
    if not os.path.exists(gtf):
        raise SystemExit("missing %s -- run the `bio` stage first" % gtf)
    gt = pd.read_csv(gtf).sort_values("image_id").reset_index(drop=True)
    fmap = {i: int(f) for f, ids in folds["folds"].items() for i in ids}
    gt["fold"] = gt["image_id"].astype(str).map(fmap)
    if gt["fold"].isna().any():
        raise RuntimeError("GT rows outside the fold definition")

    # ---------------- arms ----------------
    arms, arm_src = {}, {}
    for a in ARMS:
        d, s = _load_arm_oof(a, a_seed)
        if d is None:
            notes.append("arm `%s` incomplete -- skipped" % a)
            continue
        arms[a] = d.set_index(d["image_id"].astype(str))
        arm_src[a] = s
    if "baseline" not in arms:
        raise SystemExit("no complete arm -- nothing to analyse yet")

    common = sorted(set(gt["image_id"].astype(str)))
    for d in arms.values():
        common = [i for i in common if i in d.index]
    g = gt.set_index(gt["image_id"].astype(str)).loc[common]
    fold_lab = g["fold"].to_numpy(int)
    n = len(common)
    print("[e4] %d out-of-fold images across arms %s" % (n, sorted(arms)))

    # ---------------- sigma: per outer fold, from that fold's 80 train GT ----
    sig_by_fold, sig_img = per_image_sigma(g.reset_index(drop=True), PRIMARY_COLS)
    SIG_SRC = ("per outer fold: 1.4826*MAD of the reference-mask biomarkers of "
               "that fold's 80 TRAINING images, applied to its 20 held-out "
               "images (%s); no Gate A train scale exists for Fundus-AVSeg "
               "(DECISIONS 2026-09-17 13:30 (4))" % rel(gtf))

    # ---------------- bootstrap: resample within folds ----------------
    BIDX = fold_boot_idx(fold_lab, E2A.N_BOOT, seed=0)
    BOOT_NOTE = ("paired bootstrap, %d resamples drawn WITHIN each outer fold "
                 "(fold structure preserved); the interval is conditional on "
                 "the fixed folds and the fitted models" % E2A.N_BOOT)

    # ---------------- 1. fidelity ----------------
    frows = []
    disease = g["disease"].astype(str).to_numpy()
    for c in PRIMARY_COLS:
        bm, pipe = split_col(c)
        sv = sig_img[c]
        y = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
        for a, d in arms.items():
            x = pd.to_numeric(d.loc[common, c], errors="coerce").to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(sv)
            if ok.sum() < 5:
                continue
            e = (x[ok] - y[ok]) / sv[ok]
            base = dict(
                dataset=DATASET, config=a,
                checkpoint="last" if a != "baseline" else "base_best",
                biomarker=bm, pipeline=pipe,
                pred_path=";".join(arm_src[a]), gt_path=rel(gtf),
                sigma_source=SIG_SRC)
            frows.append(dict(
                scope="oof_pooled", fold="all", n=int(ok.sum()),
                sigma="per-fold", r_pearson=pearson(x, y),
                r_spearman=spearman(x, y),
                bias_const_sigma=float(np.mean(e)),
                resid_sd_sigma=float(np.std(e, ddof=1)),
                mae_sigma=float(np.mean(np.abs(e))), **base))
            # disease-class-adjusted sensitivity (DECISIONS 13:30 (10)):
            # within-class centred Pearson r -- how much of the agreement is
            # carried by between-disease differences alone.
            xc = _class_centre(x[ok], disease[ok])
            yc = _class_centre(y[ok], disease[ok])
            frows.append(dict(
                scope="class_adjusted", fold="all", n=int(ok.sum()),
                sigma="per-fold", r_pearson=pearson(xc, yc),
                r_spearman=spearman(xc, yc),
                bias_const_sigma=np.nan, resid_sd_sigma=np.nan,
                mae_sigma=np.nan, **base))
            for k in range(N_FOLDS):
                okk = ok & (fold_lab == k)
                if okk.sum() < 5:
                    continue
                ek = (x[okk] - y[okk]) / sv[okk]
                frows.append(dict(
                    scope="per_fold", fold=k, n=int(okk.sum()),
                    sigma=float(sig_by_fold[k][c]),
                    r_pearson=pearson(x[okk], y[okk]),
                    r_spearman=spearman(x[okk], y[okk]),
                    bias_const_sigma=float(np.mean(ek)),
                    resid_sd_sigma=float(np.std(ek, ddof=1)),
                    mae_sigma=float(np.mean(np.abs(ek))),
                    **dict(base, pred_path=rel(os.path.join(pred_dir(k, a),
                                                            "bio.csv")))))
    fid = pd.DataFrame(frows)
    fid.to_csv(os.path.join("results", "pivot", _out("e4_fidelity.csv")), index=False)

    # ---------------- 2. delta-r ----------------
    COMPARISONS = [("reliseg", "continued", "primary"),
                   ("reliseg", "baseline", "secondary"),
                   ("cfloss", "continued", "secondary"),
                   ("cfloss", "baseline", "secondary"),
                   ("continued", "baseline", "context")]
    drows = []
    prim_boot: Dict[str, "np.ndarray"] = {}
    prim_hat: Dict[str, float] = {}
    for c in PRIMARY_COLS:
        bm, pipe = split_col(c)
        sv = sig_img[c]
        y = pd.to_numeric(g[c], errors="coerce").to_numpy(float)
        xs = {a: pd.to_numeric(d.loc[common, c], errors="coerce").to_numpy(float)
              for a, d in arms.items()}
        ok = np.isfinite(y) & np.isfinite(sv)
        for x in xs.values():
            ok &= np.isfinite(x)
        if ok.sum() < 10:
            continue
        if int(ok.sum()) == n:
            bidx = BIDX
        else:
            bidx = fold_boot_idx(fold_lab[ok], E2A.N_BOOT, seed=0)
        yy, svv = y[ok], sv[ok]
        xx = {a: x[ok] for a, x in xs.items()}
        fl = fold_lab[ok]
        rb = {a: boot_r(x, yy, bidx) for a, x in xx.items()}
        for cfg, ref, role in COMPARISONS:
            if cfg not in xx or ref not in xx:
                continue
            dd = rb[cfg] - rb[ref]
            lo, hi = ci(dd)
            pt = pearson(xx[cfg], yy) - pearson(xx[ref], yy)
            db = float(np.mean((xx[cfg] - yy) / svv)
                       - np.mean((xx[ref] - yy) / svv))
            is_prim = bool(role == "primary" and pipe == "skan"
                           and bm in PRIMARY_ENDPOINTS)
            drows.append(dict(
                dataset=DATASET, target=DATASET, scope="oof_pooled",
                config=cfg, reference=ref, role=role, seed="oof(5 folds)",
                checkpoint="last", biomarker=bm, pipeline=pipe,
                n=int(ok.sum()), n_boot=E2A.N_BOOT,
                d_r=pt, lo=lo, hi=hi, d_bias_sigma=db, p_adj=np.nan,
                is_primary=is_prim, boot_note=BOOT_NOTE,
                pred_path=";".join(arm_src[cfg]), ref_path=";".join(arm_src[ref]),
                gt_path=rel(gtf), sigma_source=SIG_SRC))
            if is_prim:
                prim_boot[bm], prim_hat[bm] = dd, pt
            # leave-one-fold-out robustness (DECISIONS 13:30 (4))
            for k in range(N_FOLDS):
                m = fl != k
                if m.sum() < 20:
                    continue
                drows.append(dict(
                    dataset=DATASET, target=DATASET, scope="leave_out_fold%d" % k,
                    config=cfg, reference=ref, role=role,
                    seed="oof minus fold %d" % k, checkpoint="last",
                    biomarker=bm, pipeline=pipe, n=int(m.sum()),
                    n_boot=0,
                    d_r=pearson(xx[cfg][m], yy[m]) - pearson(xx[ref][m], yy[m]),
                    lo=np.nan, hi=np.nan,
                    d_bias_sigma=float(np.mean((xx[cfg][m] - yy[m]) / svv[m])
                                       - np.mean((xx[ref][m] - yy[m]) / svv[m])),
                    p_adj=np.nan, is_primary=False,
                    boot_note="point estimate on the other four folds",
                    pred_path=";".join(arm_src[cfg]),
                    ref_path=";".join(arm_src[ref]),
                    gt_path=rel(gtf), sigma_source=SIG_SRC))
    delta = pd.DataFrame(drows)

    # ---------------- 2b. multiplicity / replication-success rule ----------
    mstat = None
    if len(prim_boot) == len(PRIMARY_ENDPOINTS):
        mstat = max_stat_replication(prim_boot, prim_hat)
        for bm in mstat["endpoints"]:
            sel = ((delta["scope"] == "oof_pooled") & (delta["is_primary"]) &
                   (delta["biomarker"] == bm))
            delta.loc[sel, "p_adj"] = mstat["p_adj"][bm]
        with open(os.path.join(RESULT_DIR, _out("replication_rule.json")), "w",
                  encoding="utf-8") as fh:
            json.dump({"rule": max_stat_replication.__doc__.strip(),
                       "contrast": "reliseg - continued, skan pipeline",
                       **mstat}, fh, indent=2)
    else:
        notes.append("multiplicity rule not evaluated: only %d of the three "
                     "primary endpoints are available" % len(prim_boot))
    delta.to_csv(os.path.join("results", "pivot", _out("e4_delta.csv")), index=False)

    # ---------------- 3. pixel metrics ----------------
    px_all, prows = {}, []
    for a in arms:
        parts = []
        for k in range(N_FOLDS):
            f = os.path.join(pred_dir(k, a), "pixel_metrics.csv")
            if not os.path.exists(f):
                parts = None
                break
            p = pd.read_csv(f)
            p["fold"] = k
            parts.append(p)
        if parts is not None:
            px_all[a] = pd.concat(parts, ignore_index=True)
    for a, p in px_all.items():
        row = dict(dataset=DATASET, config=a, seed="oof", checkpoint="last",
                   n=len(p), mean_dice=float(p["dice"].mean()),
                   mean_cldice=float(p["cldice"].mean()),
                   units="absolute Dice / clDice -- never standardised by sigma "
                         "(DECISIONS 2026-09-17 13:30 (4))",
                   px_path="runs/pivot/e4/fold{0..4}/%s/pred/pixel_metrics.csv"
                           % ("base" if a == "baseline" else a))
        for ref in ("baseline", "continued"):
            for nm in ("dice", "cldice"):
                row["d_%s_vs_%s" % (nm, ref)] = np.nan
                row["d_%s_lo_vs_%s" % (nm, ref)] = np.nan
                row["d_%s_hi_vs_%s" % (nm, ref)] = np.nan
            if a == ref or ref not in px_all:
                continue
            m = p[["image", "fold", "dice", "cldice"]].merge(
                px_all[ref][["image", "dice", "cldice"]], on="image",
                suffixes=("_a", "_b"))
            if len(m) < 10:
                continue
            idx = fold_boot_idx(m["fold"].to_numpy(int), E2A.N_BOOT, seed=0)
            for nm in ("dice", "cldice"):
                dv = (m[nm + "_a"] - m[nm + "_b"]).to_numpy(float)
                lo, hi = ci(dv[idx].mean(1))
                row["d_%s_vs_%s" % (nm, ref)] = float(dv.mean())
                row["d_%s_lo_vs_%s" % (nm, ref)] = lo
                row["d_%s_hi_vs_%s" % (nm, ref)] = hi
        prows.append(row)
    pixel = pd.DataFrame(prows)
    pixel.to_csv(os.path.join("results", "pivot", _out("e4_pixel.csv")), index=False)

    # ---------------- 4. safety (E2's own flagger) ----------------
    safe_delta = delta[delta["scope"] == "oof_pooled"].copy()
    if len(safe_delta):
        safe_delta["seed"] = "oof"
    ren = {"d_dice_vs_baseline": "d_dice", "d_dice_lo_vs_baseline": "d_dice_lo",
           "d_dice_hi_vs_baseline": "d_dice_hi",
           "d_cldice_vs_baseline": "d_cldice",
           "d_cldice_lo_vs_baseline": "d_cldice_lo",
           "d_cldice_hi_vs_baseline": "d_cldice_hi"}
    safety = E2A.safety_table(safe_delta,
                              pixel.rename(columns=ren) if len(pixel) else pixel)
    if len(pixel) and "d_cldice_vs_continued" in pixel.columns:
        ren2 = {k.replace("baseline", "continued"): v for k, v in ren.items()}
        px2 = pixel.rename(columns=ren2)
        px2 = px2[px2["config"] != "continued"]
        s2 = E2A.safety_table(pd.DataFrame(), px2)
        if len(s2):
            s2["reference"] = "continued"
            safety = pd.concat([safety, s2], ignore_index=True)
    safety.to_csv(os.path.join("results", "pivot", _out("e4_safety.csv")), index=False)

    # ---------------- 5. downstream ----------------
    down = pd.DataFrame()
    if not args.skip_downstream:
        from src.pivot.p3_downstream import macro_auc

        y = g["disease"].astype(str).to_numpy()
        classes = sorted(pd.unique(y))
        panels = {"skan4": [c for c in PRIMARY_COLS if c.endswith("_skan")],
                  "primary8": list(PRIMARY_COLS)}
        arows = []
        for fs, cols in panels.items():
            feats = {"gt": np.nan_to_num(g[cols].to_numpy(float), nan=0.0,
                                         posinf=0.0, neginf=0.0)}
            for a, d in arms.items():
                feats[a] = np.nan_to_num(d.loc[common, cols].to_numpy(float),
                                         nan=0.0, posinf=0.0, neginf=0.0)
            for kind in ("logreg", "gbdt"):
                probas = {a: cv_proba_aligned(X, y, kind, classes, fold_lab)
                          for a, X in feats.items()}
                aucs = {a: macro_auc(y, p, classes) for a, p in probas.items()}
                fast = fast_macro_auc_factory(
                    y, classes, verify_proba=probas["gt"])
                boots = {a: [] for a in probas}
                for ix in BIDX:
                    if len(np.unique(y[ix])) < len(classes):
                        continue
                    for a, p in probas.items():
                        boots[a].append(fast(ix, p))
                for a in probas:
                    lo, hi = ci(boots[a])
                    gap = np.asarray(boots["gt"]) - np.asarray(boots[a])
                    glo, ghi = ci(gap)
                    for ref in ("continued", "baseline"):
                        d_auc = d_lo = d_hi = np.nan
                        if a not in (ref, "gt") and ref in boots:
                            dv = np.asarray(boots[a]) - np.asarray(boots[ref])
                            d_auc = aucs[a] - aucs[ref]
                            d_lo, d_hi = ci(dv)
                        elif ref != "continued":
                            continue
                        arows.append(dict(
                            dataset=DATASET, scope="oof_pooled", clf=kind,
                            featureset=fs,
                            is_primary=bool(fs == "skan4" and kind == "logreg"),
                            config=a, reference=ref, n=len(y),
                            n_classes=len(classes), n_boot=E2A.N_BOOT,
                            macro_auc=aucs[a], ci_lo=lo, ci_hi=hi,
                            d_auc_vs_ref=d_auc, d_auc_lo=d_lo, d_auc_hi=d_hi,
                            gt_minus_this=aucs["gt"] - aucs[a],
                            gt_gap_lo=glo, gt_gap_hi=ghi,
                            class_counts=";".join("%s=%d" % (c, int((y == c).sum()))
                                                  for c in classes),
                            features="+".join(cols), boot_note=BOOT_NOTE,
                            protocol="p3_downstream estimators (standardised "
                                     "multinomial logreg / small GBDT), outer "
                                     "folds pinned 1:1 to the five "
                                     "segmentation folds -- nested CV "
                                     "(DECISIONS 2026-09-17 13:30 (7))",
                            pred_path=(rel(gtf) if a == "gt"
                                       else ";".join(arm_src[a]))))
        down = pd.DataFrame(arows)
        down.to_csv(os.path.join("results", "pivot", _out("e4_downstream.csv")),
                    index=False)

    write_report(fid, delta, pixel, safety, down, folds, sig_by_fold,
                 SIG_SRC, BOOT_NOTE, mstat, gtf, notes, n, suffix=SFX)
    print("[e4] analysis written to results/pivot/e4_*%s.csv and "
          "results/pivot/E4_REPORT%s.md" % (SFX, SFX))
    return 0


def write_report(fid, delta, pixel, safety, down, folds, sig_by_fold,
                 sig_src, boot_note, mstat, gtf, notes, n_imgs,
                 suffix: str = "") -> None:
    """``suffix`` is the ``_s<k>`` tag of a non-default fine-tune seed, so the
    per-seed reports sit beside the canonical one instead of overwriting it."""
    import numpy as np
    import pandas as pd

    from src.pivot.e2_analysis import _tbl, rel, srt

    L: List[str] = []
    L.append("# E4 -- independent replication (Fundus-AVSeg)\n")
    L.append("\n_generated %s by `src/pivot/e4_replication.py analyse`_\n"
             % time.strftime("%Y-%m-%d %H:%M:%S"))
    L.append("\nPre-registered in `exp/DECISIONS.md` 2026-09-17 13:15 and "
             "clarified 13:30, both before any out-of-fold result was computed "
             "or looked at.  Timeline, stated plainly: E4 is a **prospectively "
             "locked replication run after E2**, not part of the project's "
             "original pre-registration; Fundus-AVSeg was chosen after the "
             "FIVES null result was seen.\n")

    L.append("\n## 0. Folds\n")
    L.append("Source: `%s` (generator `src/pivot/e4_replication.py::build_folds`).\n"
             % rel(FOLDS_JSON))
    L.append("\n```\n%s\n```\n" % folds["definition"])
    L.append("\n| fold | n | class counts |\n|---|---|---|\n")
    for f in sorted(folds["folds"], key=int):
        L.append("| %s | %d | %s |\n" % (f, len(folds["folds"][f]),
                                         folds["fold_class_counts"][f]))
    L.append("\n%s\n" % folds.get("partition_note", ""))
    L.append("\n**Cross-validation unit: image.** %s\n"
             % folds.get("cv_unit_note", ""))
    L.append("\nEvery image is held out exactly once; %d out-of-fold "
             "predictions per arm.\n" % n_imgs)

    L.append("\n## 1. Sigma\n")
    L.append("%s\n\n" % sig_src)
    L.append("| biomarker | %s |\n|%s|\n"
             % (" | ".join("fold %d" % k for k in sorted(sig_by_fold)),
                "|".join(["---"] * (1 + len(sig_by_fold)))))
    for c in sorted(next(iter(sig_by_fold.values()))):
        L.append("| %s | %s |\n" % (c, " | ".join(
            "%.6g" % sig_by_fold[k][c] for k in sorted(sig_by_fold))))
    L.append("\nDice, clDice, delta-r and delta-AUC are reported in their own "
             "units and are never divided by sigma.\n")

    L.append("\n## 2. Measurement fidelity (out-of-fold, pooled)\n")
    f = fid[fid["scope"] == "oof_pooled"] if len(fid) else fid
    L.append(_tbl(srt(f[f["pipeline"] == "skan"] if len(f) else f,
                      ["biomarker", "config"]),
                  ["biomarker", "config", "n", "r_pearson", "r_spearman",
                   "bias_const_sigma", "resid_sd_sigma", "mae_sigma"]))
    L.append("\nDisease-class-adjusted sensitivity (within-class centred r, "
             "`scope=class_adjusted` in the CSV):\n\n")
    ca = fid[(fid["scope"] == "class_adjusted") & (fid["pipeline"] == "skan")] \
        if len(fid) else fid
    L.append(_tbl(srt(ca, ["biomarker", "config"]),
                  ["biomarker", "config", "n", "r_pearson", "r_spearman"]))
    L.append("\nSource: `results/pivot/e4_fidelity.csv`, GT `%s`.\n" % rel(gtf))

    L.append("\n## 3. Delta-r (primary endpoint)\n")
    L.append("%s.  Primary contrast: **ReliSeg - continued** on the skan "
             "density / total_length / FD; ReliSeg - baseline is the secondary "
             "'total deployment difference'.\n\n" % boot_note)
    if len(delta):
        d = delta[(delta["pipeline"] == "skan") & (delta["scope"] == "oof_pooled")]
        L.append(_tbl(srt(d, ["role", "biomarker", "config", "reference"]),
                      ["role", "biomarker", "config", "reference", "n",
                       "d_r", "lo", "hi", "p_adj", "d_bias_sigma"]))
        L.append("\nLeave-one-fold-out robustness (point estimates on the "
                 "other four folds):\n\n")
        lo = delta[(delta["pipeline"] == "skan") &
                   (delta["role"] == "primary") &
                   (delta["scope"].astype(str).str.startswith("leave_out"))]
        L.append(_tbl(srt(lo, ["biomarker", "scope"]),
                      ["biomarker", "config", "reference", "scope", "n", "d_r"]))
    else:
        L.append("_(no rows yet)_\n")
    L.append("\nSource: `results/pivot/e4_delta.csv`.\n")

    L.append("\n## 3b. Multiplicity and the replication-success rule\n")
    if mstat:
        L.append("Max-statistic (Westfall-Young) bootstrap over the three "
                 "primary delta-r endpoints at alpha = %.2f.\n\n" % mstat["alpha"])
        L.append("| endpoint | delta-r 95%% CI | adjusted p |\n|---|---|---|\n")
        for k in mstat["endpoints"]:
            L.append("| %s | [%+.3f, %+.3f] | %.4f |\n"
                     % (k, mstat["ci"][k][0], mstat["ci"][k][1],
                        mstat["p_adj"][k]))
        L.append("\n**Replication success: %s** (significant after adjustment: "
                 "%s; endpoints running in the opposite direction: %s).  A CI "
                 "that contains zero is not evidence of no effect -- no "
                 "equivalence margin was pre-specified.\n"
                 % ("YES" if mstat["replication_success"] else "NO",
                    mstat["significant"] or "none",
                    mstat["opposite_direction"] or "none"))
        L.append("\nSource: `%s`.\n" % rel(os.path.join(RESULT_DIR,
                                                        "replication_rule.json")))
    else:
        L.append("_(not evaluated yet)_\n")

    L.append("\n## 4. Pixel metrics\n")
    L.append(_tbl(pixel, ["config", "n", "mean_dice", "mean_cldice",
                          "d_dice_vs_baseline", "d_cldice_vs_baseline",
                          "d_dice_vs_continued", "d_cldice_vs_continued"], 4))
    L.append("\nSource: `results/pivot/e4_pixel.csv`.\n")

    L.append("\n## 5. Safety endpoints\n")
    if len(safety):
        bad = safety[safety["flag"] != "ok"]
        L.append("%d of %d rows flagged.\n\n" % (len(bad), len(safety)))
        L.append(_tbl(srt(bad, ["endpoint", "config", "biomarker"]),
                      ["endpoint", "config", "reference", "biomarker",
                       "pipeline", "value", "lo", "hi", "margin_value", "flag"]))
    else:
        L.append("_(no rows yet)_\n")
    L.append("\nSource: `results/pivot/e4_safety.csv`; margins and flag logic "
             "are `src/pivot/e2_analysis.py::safety_table` unchanged.\n")

    L.append("\n## 6. Downstream four-class macro-AUC\n")
    if len(down):
        p = down[(down["is_primary"]) & (down["reference"] == "continued")]
        L.append(_tbl(srt(p, ["config"]),
                      ["config", "n", "macro_auc", "ci_lo", "ci_hi",
                       "gt_minus_this", "d_auc_vs_ref", "d_auc_lo",
                       "d_auc_hi"], 4))
        L.append("\nClassifier outer folds are pinned 1:1 to the segmentation "
                 "folds.  All feature sets / classifiers / references: "
                 "`results/pivot/e4_downstream.csv`.\n")
    else:
        L.append("_(not run)_\n")

    pw = os.path.join("results", "pivot", "e4_power.csv")
    L.append("\n## 7. Pre-unblinding power\n")
    if os.path.exists(pw):
        w = pd.read_csv(pw)
        L.append(_tbl(srt(w, ["biomarker", "true_delta_r"]),
                      ["biomarker", "r_reference", "true_delta_r", "n",
                       "n_sim", "power", "mean_d_r"], 3))
        L.append("\nSource: `%s` (Monte-Carlo on the E2 per-image correlation "
                 "structure, run before any E4 out-of-fold number existed).\n" % pw)
    else:
        L.append("_(not run)_\n")

    mz = os.path.join("results", "pivot", "e4_maples_zeroshot.csv")
    L.append("\n## 8. MAPLES-DR zero-shot (secondary set)\n")
    L.append("Role: annotation-convention / domain robustness and a "
             "resolution effect-modification check only.  Claims are phrased "
             "as agreement with the **manually corrected** annotation "
             "convention; the >= %d px stratum is called *higher-resolution*, "
             "not high-resolution.\n\n" % MAPLES_STRATUM_PX)
    if os.path.exists(mz):
        m = pd.read_csv(mz)
        sel = m[(m["pipeline"] == "skan") & (m["kind"] == "delta") &
                (m["reference"] == "continued")] if "kind" in m.columns else m
        L.append(_tbl(srt(sel, ["stratum", "source_dataset", "biomarker", "config"]),
                      ["stratum", "source_dataset", "seed", "config",
                       "biomarker", "n", "d_r", "lo", "hi"]))
        L.append("\nSource: `%s`; strata `%s`.\n"
                 % (mz, os.path.join(RESULT_DIR, "maples_strata.csv")))
    else:
        L.append("_(not run yet)_\n")
    pa = os.path.join("results", "pivot", "e4_maples_preannot.csv")
    L.append("\nPre-annotation vs manually corrected anchor sensitivity: %s\n"
             % ("`%s`" % pa if os.path.exists(pa) else "_(not run yet)_"))

    if notes:
        L.append("\n## 9. Coverage notes\n")
        for x in notes:
            L.append("- %s\n" % x)

    L.append("\n## Source paths\n")
    for p in ["results/pivot/e4/folds.json",
              "results/pivot/e4/replication_rule.json",
              "results/pivot/e4/maples_strata.csv",
              "results/pivot/e4_gt_fundusavseg.csv",
              "results/pivot/e4_fidelity.csv", "results/pivot/e4_delta.csv",
              "results/pivot/e4_pixel.csv", "results/pivot/e4_safety.csv",
              "results/pivot/e4_downstream.csv", "results/pivot/e4_power.csv",
              "results/pivot/e4_maples_zeroshot.csv",
              "results/pivot/e4_maples_preannot.csv",
              "runs/pivot/e4/fold<k>/base/{config.json,split.json,summary.json,best.pt}",
              "runs/pivot/e4/fold<k>/<arm>/{config.json,log.json,summary.json,last.pt}",
              "runs/pivot/e4/fold<k>/<arm>/pred/{manifest.csv,pixel_metrics.csv,bio.csv}",
              "runs/pivot/e4/maples/<tag>/{manifest.csv,pixel_metrics.csv,bio.csv}",
              "runs/pivot/e4/driver_gpu<g>.log"]:
        L.append("- `%s`\n" % p)

    stem, ext = os.path.splitext("E4_REPORT.md")
    out = os.path.join("results", "pivot", stem + suffix + ext)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("".join(L))
    print("[e4] wrote %s" % out)


# ==========================================================================
# 6b -- pre-unblinding power curve
# ==========================================================================
def cmd_power(args) -> int:
    """Monte-Carlo power for the E4 primary delta-r, from the E2 structure.

    DECISIONS 2026-09-17 13:30 (8).  Nothing here touches a replication-set
    prediction; the inputs are the E2 in-domain biomarker tables of FIVES and
    HRF (``runs/pivot/<ds>/seed0/<arm>/pred/bio.csv`` against
    ``results/pivot/bio_master.csv``), so the curve can be -- and is -- produced
    before any E4 out-of-fold number exists.

    Generative model, per simulated study of ``n`` images:
      * ``z`` standard normal stands for the (rank-transformed) GT biomarker;
      * the reference arm is ``r0*z + sqrt(1-r0^2) * e_ref`` and the candidate
        arm ``r1*z + sqrt(1-r1^2) * e_new`` with ``r1 = r0 + delta``;
      * the two error terms share a component with correlation ``rho_e``,
        ``e = sqrt(rho_e)*u + sqrt(1-rho_e)*v``, because the two arms are the
        same network fine-tuned differently and their errors are strongly
        paired.  ``r0`` and ``rho_e`` are both measured on E2: ``r0`` is the
        baseline arm's Pearson r against GT, ``rho_e`` the correlation of the
        baseline and ReliSeg residuals.
      * the test is the same paired percentile bootstrap the analysis uses
        (2.5 / 97.5 of ``r_new - r_ref``), resampling within five blocks of
        ``n/5`` to mimic the fold-stratified resample.
    """
    import numpy as np
    import pandas as pd

    from src.pivot.common import PIVOT_DIR, PRIMARY_COLS
    from src.pivot.e2_analysis import gt_table, pred_root, _key

    rng = np.random.RandomState(20260917)
    cols = [c for c in PRIMARY_COLS if c.endswith("_skan")]
    want = [c for c in cols if c.split("_skan")[0] in PRIMARY_ENDPOINTS or
            c.startswith("total_length") or c.startswith("density") or
            c.startswith("FD")]
    rows = []
    for ds in ("fives", "hrf"):
        try:
            gt = gt_table(ds)
        except Exception as exc:                                # noqa: BLE001
            print("[power] %s GT unavailable: %r" % (ds, exc))
            continue
        gt = gt.assign(__k=_key(gt["image_id"])).set_index("__k")
        frames = {}
        for arm in ("baseline", "reliseg"):
            f = os.path.join(pred_root(ds, 0, arm, "last"), "bio.csv")
            if not os.path.exists(f):
                continue
            d = pd.read_csv(f)
            frames[arm] = d.assign(__k=_key(d["image_id"])).set_index("__k")
        if "baseline" not in frames:
            print("[power] %s baseline bio.csv missing -- skipped" % ds)
            continue
        common = sorted(set(gt.index) & set.intersection(
            *[set(v.index) for v in frames.values()]))
        for c in want:
            y = pd.to_numeric(gt.loc[common, c], errors="coerce").to_numpy(float)
            xb = pd.to_numeric(frames["baseline"].loc[common, c],
                               errors="coerce").to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(xb)
            if ok.sum() < 20:
                continue
            r0 = float(np.corrcoef(xb[ok], y[ok])[0, 1])
            rho_e = 0.9
            if "reliseg" in frames:
                xr = pd.to_numeric(frames["reliseg"].loc[common, c],
                                   errors="coerce").to_numpy(float)
                m = ok & np.isfinite(xr)
                if m.sum() >= 20:
                    eb = xb[m] - np.polyval(np.polyfit(y[m], xb[m], 1), y[m])
                    er = xr[m] - np.polyval(np.polyfit(y[m], xr[m], 1), y[m])
                    if eb.std() > 0 and er.std() > 0:
                        rho_e = float(np.clip(np.corrcoef(eb, er)[0, 1], 0.0, 0.999))
            # The E2-measured residual pairing is very high (two arms of the
            # same fine-tuned network), which makes the paired test extremely
            # precise.  Two weaker pairings are simulated alongside it so the
            # curve is not read off a single optimistic assumption.
            rho_grid = [(rho_e, "measured on E2 (%s seed0 baseline vs reliseg "
                                "residuals)" % ds),
                        (0.5, "sensitivity: moderate residual pairing"),
                        (0.0, "sensitivity: independent residuals")]
            for rho_e, rho_src in rho_grid:
              for delta in (0.05, 0.10, 0.15):
                r1 = min(0.999, max(-0.999, r0 + delta))
                hits, dhat = 0, []
                n = args.n
                blocks = np.repeat(np.arange(5), n // 5)
                bidx = None
                for _ in range(args.n_sim):
                    z = rng.randn(n)
                    u = rng.randn(n)
                    e1 = np.sqrt(rho_e) * u + np.sqrt(1 - rho_e) * rng.randn(n)
                    e2 = np.sqrt(rho_e) * u + np.sqrt(1 - rho_e) * rng.randn(n)
                    xr_ = r0 * z + np.sqrt(max(0.0, 1 - r0 ** 2)) * e1
                    xn_ = r1 * z + np.sqrt(max(0.0, 1 - r1 ** 2)) * e2
                    if bidx is None or args.fresh_boot:
                        bidx = fold_boot_idx(blocks, args.n_boot,
                                             seed=int(rng.randint(1 << 30)))
                    a = xn_[bidx]; b = xr_[bidx]; g2 = z[bidx]
                    def _r(p, q):
                        p = p - p.mean(1, keepdims=True)
                        q = q - q.mean(1, keepdims=True)
                        den = np.sqrt((p * p).sum(1) * (q * q).sum(1))
                        return (p * q).sum(1) / np.where(den > 0, den, np.nan)
                    dd = _r(a, g2) - _r(b, g2)
                    lo = np.percentile(dd, 2.5)
                    hits += int(lo > 0)
                    dhat.append(float(np.corrcoef(xn_, z)[0, 1]
                                      - np.corrcoef(xr_, z)[0, 1]))
                rows.append(dict(
                    source_dataset=ds, biomarker=c, r_reference=round(r0, 4),
                    rho_residual=round(rho_e, 4), rho_source=rho_src,
                    true_delta_r=delta,
                    n=n, n_sim=args.n_sim, n_boot=args.n_boot,
                    power=hits / float(args.n_sim),
                    mean_d_r=float(np.mean(dhat)),
                    note="one-sided: fraction of simulated studies whose 95% "
                         "percentile CI of delta-r excludes zero from below",
                    source="runs/pivot/%s/seed0/{baseline,reliseg}/pred/bio.csv "
                           "+ results/pivot/bio_master.csv" % ds))
                print("[power] %s %s r0=%.3f rho_e=%.2f delta=%.2f -> power %.3f"
                      % (ds, c, r0, rho_e, delta, hits / float(args.n_sim)),
                      flush=True)
    out = os.path.join("results", "pivot", "e4_power.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("[e4] wrote %s (%d rows)" % (out, len(rows)))
    return 0


# ==========================================================================
# 7 -- MAPLES-DR analysis
# ==========================================================================
def cmd_maples_analyse(args) -> int:
    import numpy as np
    import pandas as pd

    from src.pivot import e2_analysis as E2A
    from src.pivot.common import PRIMARY_COLS
    from src.pivot.e2_analysis import (boot_idx, boot_r, ci, pearson, rel,
                                       spearman, split_col)

    anchor = getattr(args, "anchor", "final")
    gtf = gt_csv(MAPLES) if anchor == "final" else gt_csv(MAPLES + "_preannot")
    if not os.path.exists(gtf):
        raise SystemExit("missing %s (run `bio --maples%s`)"
                         % (gtf, "" if anchor == "final" else " --preannot"))
    gt = pd.read_csv(gtf)
    gt["__k"] = gt["image_id"].astype(str)
    strata = maples_strata().set_index("image_id")
    sig = _sigma_from(gt, PRIMARY_COLS)
    SIG_SRC = ("1.4826*MAD of MAPLES-DR's own %d-image %s-mask biomarkers (%s); "
               "no Gate A train scale exists for this set"
               % (len(gt), "final (manually corrected)" if anchor == "final"
                  else "network PRE-ANNOTATION", rel(gtf)))

    frames: Dict[Tuple[str, int, str], pd.DataFrame] = {}
    for ds in MAPLES_SOURCES:
        for cfg in MAPLES_CONFIGS:
            for s in (0, 1, 2):
                f = os.path.join(maples_dir(ds, s, cfg), "bio.csv")
                if os.path.exists(f):
                    d = pd.read_csv(f)
                    d["__k"] = d["image_id"].astype(str)
                    frames[(ds, s, cfg)] = d.set_index("__k")
    if not frames:
        raise SystemExit("no MAPLES-DR bio.csv yet")

    rows = []
    g_all = gt.set_index("__k")
    for stratum in ("all", STRAT_HI, STRAT_LO):
        if stratum == "all":
            ids = [i for i in g_all.index]
        else:
            ids = [i for i in g_all.index
                   if str(strata["stratum"].get(i, "")) == stratum]
        for (ds, s, cfg), d in sorted(frames.items()):
            sel = [i for i in ids if i in d.index]
            if len(sel) < 5:
                continue
            for c in PRIMARY_COLS:
                bm, pipe = split_col(c)
                y = pd.to_numeric(g_all.loc[sel, c], errors="coerce").to_numpy(float)
                x = pd.to_numeric(d.loc[sel, c], errors="coerce").to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 5:
                    continue
                sg = sig[c]
                e = (x[ok] - y[ok]) / sg
                rows.append(dict(
                    kind="fidelity", stratum=stratum, source_dataset=ds,
                    seed=s, config=cfg, reference="", biomarker=bm,
                    pipeline=pipe, n=int(ok.sum()), sigma=sg,
                    r_pearson=pearson(x, y), r_spearman=spearman(x, y),
                    bias_const_sigma=float(np.mean(e)),
                    resid_sd_sigma=float(np.std(e, ddof=1)),
                    mae_sigma=float(np.mean(np.abs(e))),
                    d_r=np.nan, lo=np.nan, hi=np.nan,
                    pred_path=rel(os.path.join(maples_dir(ds, s, cfg), "bio.csv")),
                    gt_path=rel(gtf), sigma_source=SIG_SRC))
        # paired delta-r within (source dataset, seed)
        for ds in MAPLES_SOURCES:
            for s in (0, 1, 2):
                have = {cfg: frames[(ds, s, cfg)] for cfg in MAPLES_CONFIGS
                        if (ds, s, cfg) in frames}
                if len(have) < 2:
                    continue
                sel = [i for i in ids if all(i in d.index for d in have.values())]
                if len(sel) < 10:
                    continue
                for c in PRIMARY_COLS:
                    bm, pipe = split_col(c)
                    y = pd.to_numeric(g_all.loc[sel, c], errors="coerce").to_numpy(float)
                    xs = {cfg: pd.to_numeric(d.loc[sel, c], errors="coerce").to_numpy(float)
                          for cfg, d in have.items()}
                    ok = np.isfinite(y)
                    for x in xs.values():
                        ok &= np.isfinite(x)
                    if ok.sum() < 10:
                        continue
                    yy = y[ok]
                    xx = {k: v[ok] for k, v in xs.items()}
                    bidx = boot_idx(int(ok.sum()), E2A.N_BOOT)
                    rb = {k: boot_r(v, yy, bidx) for k, v in xx.items()}
                    for cfg in ("reliseg", "cfloss"):
                        for ref in ("continued", "baseline"):
                            if cfg not in xx or ref not in xx:
                                continue
                            dd = rb[cfg] - rb[ref]
                            lo, hi = ci(dd)
                            rows.append(dict(
                                kind="delta", stratum=stratum,
                                source_dataset=ds, seed=s, config=cfg,
                                reference=ref, biomarker=bm, pipeline=pipe,
                                n=int(ok.sum()), sigma=sig[c],
                                r_pearson=np.nan, r_spearman=np.nan,
                                bias_const_sigma=np.nan, resid_sd_sigma=np.nan,
                                mae_sigma=np.nan,
                                d_r=pearson(xx[cfg], yy) - pearson(xx[ref], yy),
                                lo=lo, hi=hi,
                                pred_path=rel(os.path.join(
                                    maples_dir(ds, s, cfg), "bio.csv")),
                                gt_path=rel(gtf), sigma_source=SIG_SRC))
    out = os.path.join("results", "pivot",
                       "e4_maples_zeroshot.csv" if anchor == "final"
                       else "e4_maples_preannot.csv")
    res = pd.DataFrame(rows)
    if len(res):
        res.insert(0, "anchor", anchor)
    res.to_csv(out, index=False)
    print("[e4] wrote %s (%d rows)" % (out, len(rows)))
    return 0


# ==========================================================================
# 8 -- job queue / driver
# ==========================================================================
def build_jobs() -> List[dict]:
    """Ordered E4 job list.

    Order (the workers walk it top-down and claim the first free job, so with
    one lane per GPU fold 0 lands on the first lane, fold 1 on the second, ...):
      1. the five fold base trainings,
      2. the fifteen fine-tunes (3 arms x 5 folds), each waiting on its fold's
         ``base/best.pt``,
      3. the twenty in-domain inferences (4 arms x 5 folds),
      4. the MAPLES-DR zero-shot sweep (lowest priority; several of its
         checkpoints are still being produced by the E2 queue).
    """
    from src.pivot import e2_run as E2

    jobs: List[dict] = []
    for k in range(N_FOLDS):
        jobs.append(dict(id="e4:base:f%d" % k, kind="train",
                         argv=["-m", "src.pivot.e4_replication", "train-base",
                               "--fold", str(k)],
                         deps=[FOLDS_JSON],
                         done=os.path.join(base_dir(k), "summary.json")))
    for k in range(N_FOLDS):
        for arm in FT_ARMS:
            jobs.append(dict(id="e4:ft:f%d:%s" % (k, arm), kind="train",
                             argv=["-m", "src.pivot.e4_replication", "finetune",
                                   "--fold", str(k), "--arm", arm],
                             deps=[os.path.join(base_dir(k), "best.pt")],
                             done=os.path.join(arm_dir(k, arm), "summary.json")))
    for k in range(N_FOLDS):
        for arm in ARMS:
            jobs.append(dict(id="e4:infer:f%d:%s" % (k, arm), kind="infer",
                             argv=["-m", "src.pivot.e4_replication", "infer",
                                   "--fold", str(k), "--arm", arm],
                             deps=[infer_dep(k, arm)],
                             done=os.path.join(pred_dir(k, arm),
                                               "infer_meta.json")))
    # ---- review point 16: extra fine-tune seeds of the key null ----------
    # Same per-fold base checkpoint, same arm definition, only --seed differs.
    for k, arm, s in extra_seed_units():
        jobs.append(dict(id="e4:ft:f%d:%s:s%d" % (k, arm, s), kind="train",
                         argv=["-m", "src.pivot.e4_replication", "finetune",
                               "--fold", str(k), "--arm", arm,
                               "--seed", str(s)],
                         deps=[os.path.join(base_dir(k), "best.pt")],
                         done=os.path.join(arm_dir(k, arm, s), "summary.json")))
    for k, arm, s in extra_seed_units():
        jobs.append(dict(id="e4:infer:f%d:%s:s%d" % (k, arm, s), kind="infer",
                         argv=["-m", "src.pivot.e4_replication", "infer",
                               "--fold", str(k), "--arm", arm,
                               "--seed", str(s)],
                         deps=[infer_dep(k, arm, s)],
                         done=os.path.join(pred_dir(k, arm, s),
                                           "infer_meta.json")))
    for ds in MAPLES_SOURCES:
        for cfg in MAPLES_CONFIGS:
            for s in (0, 1, 2):
                ck = E2.eval_ckpt(ds, s, cfg, "last")
                jobs.append(dict(
                    id="e4:maples:%s:s%d:%s" % (ds, s, cfg), kind="infer",
                    argv=["-m", "src.pivot.e4_replication", "maples-zeroshot",
                          "--dataset", ds, "--seed", str(s), "--config", cfg],
                    deps=[ck],
                    done=os.path.join(maples_dir(ds, s, cfg), "infer_meta.json")))
    return jobs


def _safe(job_id: str) -> str:
    return job_id.replace(":", "__").replace(">", "_to_")


def claim(job_id: str) -> bool:
    os.makedirs(QUEUE_DIR, exist_ok=True)
    lock = os.path.join(QUEUE_DIR, _safe(job_id) + ".lock")
    try:
        os.mkdir(lock)
    except FileExistsError:
        return False
    with open(os.path.join(lock, "owner.json"), "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "t": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    return True


def release(job_id: str) -> None:
    lock = os.path.join(QUEUE_DIR, _safe(job_id) + ".lock")
    try:
        os.remove(os.path.join(lock, "owner.json"))
        os.rmdir(lock)
    except OSError:
        pass


def gpu_free_mb(gpu: int) -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits", "-i", str(gpu)],
            stderr=subprocess.STDOUT, timeout=60)
        return int(out.decode("utf-8", "replace").strip().splitlines()[0])
    except Exception as exc:                                    # noqa: BLE001
        print("[e4] nvidia-smi failed: %r" % (exc,), flush=True)
        return None


def wait_for_gpu(gpu: int, need_mb: int = MIN_FREE_MB, poll: int = 60) -> None:
    """Block until the GPU reports at least ``need_mb`` free.

    The E2 queue owns both cards; E4 only fills the gaps.  Nothing is ever
    killed -- the lane simply waits.
    """
    while True:
        free = gpu_free_mb(gpu)
        if free is None or free >= need_mb:
            if free is not None:
                print("[e4] gpu%d free=%d MiB >= %d -- go" % (gpu, free, need_mb),
                      flush=True)
            return
        print("[e4] gpu%d free=%d MiB < %d -- waiting %ds"
              % (gpu, free, need_mb, poll), flush=True)
        time.sleep(poll)


def run_job(job: dict, gpu: int) -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, _safe(job["id"]) + ".log")
    argv = list(job["argv"]) + ["--gpu", str(gpu)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    print("[e4] >>> %s\n     %s\n     log %s" % (job["id"], " ".join(argv), log),
          flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write("\n===== %s %s =====\n"
                 % (time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(argv)))
        fh.flush()
        rc = subprocess.call([PY] + argv, cwd=EXP_ROOT, env=env,
                             stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print("[e4] <<< %s rc=%d in %.1f min" % (job["id"], rc, dt / 60.0), flush=True)
    return rc


def worker(gpu: int, need_mb: int, only: str = "") -> int:
    jobs = [j for j in build_jobs() if (not only or only in j["id"])]
    fails: List[str] = []
    while True:
        did_work = False
        pending = False
        for job in jobs:
            if os.path.exists(job["done"]) or job["id"] in fails:
                continue
            if not all(os.path.exists(p) for p in job["deps"]):
                pending = True
                continue
            if not claim(job["id"]):
                pending = True
                continue
            wait_for_gpu(gpu, need_mb)
            rc = run_job(job, gpu)
            if rc != 0 or not os.path.exists(job["done"]):
                print("[e4] FAILED %s rc=%d" % (job["id"], rc), flush=True)
                fails.append(job["id"])
                release(job["id"])
            did_work = True
        if not did_work:
            if not pending:
                break
            print("[e4] nothing runnable; waiting 180 s", flush=True)
            time.sleep(180)
            if all(os.path.exists(j["done"]) or j["id"] in fails for j in jobs):
                break
    left = [j["id"] for j in jobs if not os.path.exists(j["done"])]
    print("[e4] lane gpu%d done. failures=%s not-done=%s" % (gpu, fails, left),
          flush=True)
    return 1 if fails else 0


def cmd_finish(args) -> int:
    """Wait for the queue and the biomarker stage, then run both analyses.

    The GPU lanes and the CPU biomarker watcher are detached; this is the
    last link, so the whole E4 chain completes without another hand.  It
    polls, never kills anything, and runs each analysis exactly once -- the
    Fundus-AVSeg analysis as soon as its 20 prediction directories have a
    ``bio.csv``, and the MAPLES-DR one (both anchors) when its 24 do.
    """
    import subprocess as sp

    def ready(units) -> bool:
        return all(os.path.exists(os.path.join(root, "bio.csv"))
                   for _, _, root in units)

    fold_units = [u for u in bio_units(include_maples=False)]
    mp_units = [u for u in bio_units(include_maples=True) if u[0] == MAPLES]
    done_main = done_mp = False
    t0 = time.time()
    while not (done_main and done_mp):
        if (time.time() - t0) > args.max_hours * 3600:
            print("[e4] finish: giving up after %.1f h" % args.max_hours,
                  flush=True)
            break
        if not done_main and ready(fold_units) and os.path.exists(gt_csv(DATASET)):
            print("[e4] finish: running analyse", flush=True)
            sp.call([PY, "-m", "src.pivot.e4_replication", "analyse",
                     "--threads", str(args.threads)], cwd=EXP_ROOT)
            done_main = True
        if not done_mp and ready(mp_units) and os.path.exists(gt_csv(MAPLES)):
            print("[e4] finish: running maples-analyse", flush=True)
            sp.call([PY, "-m", "src.pivot.e4_replication", "maples-analyse"],
                    cwd=EXP_ROOT)
            if os.path.exists(gt_csv(MAPLES + "_preannot")):
                sp.call([PY, "-m", "src.pivot.e4_replication", "maples-analyse",
                         "--anchor", "preannot"], cwd=EXP_ROOT)
            done_mp = True
            # the report's MAPLES section is filled in by the main analysis,
            # so refresh it once the zero-shot tables exist
            if done_main:
                sp.call([PY, "-m", "src.pivot.e4_replication", "analyse",
                         "--threads", str(args.threads)], cwd=EXP_ROOT)
        if done_main and done_mp:
            break
        time.sleep(args.poll)
    print("[e4] finish: analyse=%s maples=%s" % (done_main, done_mp), flush=True)
    return 0


def cmd_plan(args) -> int:
    jobs = build_jobs()
    for j in jobs:
        state = ("DONE " if os.path.exists(j["done"]) else
                 ("READY" if all(os.path.exists(p) for p in j["deps"]) else "WAIT "))
        print("%s %s" % (state, j["id"]))
    print("\n%d/%d complete" % (sum(os.path.exists(j["done"]) for j in jobs),
                                len(jobs)))
    return 0


# ==========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("folds").set_defaults(fn=cmd_folds)

    t = sub.add_parser("train-base")
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--gpu", type=int, default=0)
    t.set_defaults(fn=cmd_train_base)

    f = sub.add_parser("finetune")
    f.add_argument("--fold", type=int, required=True)
    f.add_argument("--arm", required=True, choices=list(FT_ARMS))
    f.add_argument("--gpu", type=int, default=0)
    f.add_argument("--seed", type=int, default=E4_SEED,
                   help="fine-tune seed (review point 16). %d keeps the "
                        "original directories/tags; any other value writes to "
                        "<arm>_s<seed> and leaves everything else identical."
                        % E4_SEED)
    f.set_defaults(fn=cmd_finetune)

    i = sub.add_parser("infer")
    i.add_argument("--fold", type=int, required=True)
    i.add_argument("--arm", required=True, choices=list(ARMS))
    i.add_argument("--gpu", type=int, default=0)
    i.add_argument("--seed", type=int, default=E4_SEED,
                   help="fine-tune seed of the checkpoint to evaluate "
                        "(review point 16); see `finetune --seed`")
    i.set_defaults(fn=cmd_infer)

    m = sub.add_parser("maples-zeroshot")
    m.add_argument("--dataset", required=True, choices=list(MAPLES_SOURCES))
    m.add_argument("--seed", type=int, required=True)
    m.add_argument("--config", required=True, choices=list(MAPLES_CONFIGS))
    m.add_argument("--gpu", type=int, default=0)
    m.set_defaults(fn=cmd_maples_zeroshot)

    b = sub.add_parser("bio")
    b.add_argument("--procs", type=int, default=8)
    b.add_argument("--only", default="")
    b.add_argument("--maples", action="store_true")
    b.add_argument("--preannot", action="store_true",
                   help="also measure MAPLES-DR's network pre-annotations "
                        "(the annotation-convention sensitivity anchor)")
    b.add_argument("--skip-gt", action="store_true")
    b.add_argument("--force", action="store_true")
    b.set_defaults(fn=cmd_bio)

    a = sub.add_parser("analyse")
    a.add_argument("--skip-downstream", action="store_true")
    a.add_argument("--seed", type=int, default=E4_SEED,
                   help="which fine-tune seed to analyse (review point 16). "
                        "%d writes the canonical e4_*.csv / replication_rule.json "
                        "/ E4_REPORT.md; any other seed writes the same files "
                        "with a _s<seed> suffix. The estimator, the fold "
                        "structure and the replication rule are identical for "
                        "every seed -- only which fine-tune is read changes."
                        % E4_SEED)
    a.add_argument("--threads", type=int, default=4,
                   help="cap on the BLAS/OpenMP thread pools (the downstream "
                        "GBDT otherwise takes every core from the biomarker "
                        "workers)")
    a.set_defaults(fn=cmd_analyse)

    asd = sub.add_parser("analyse-seeds",
                         help="run `analyse` once per available fine-tune seed "
                              "(seed 0 rewrites the canonical artefacts) and "
                              "pool the per-seed delta-r -- review point 16")
    asd.add_argument("--skip-downstream", action="store_true")
    asd.add_argument("--threads", type=int, default=4)
    asd.set_defaults(fn=cmd_analyse_seeds)

    ma = sub.add_parser("maples-analyse")
    ma.add_argument("--anchor", default="final", choices=["final", "preannot"],
                    help="which MAPLES-DR mask is the measurement anchor: the "
                         "published (manually corrected) one, or the network "
                         "pre-annotation (sensitivity analysis)")
    ma.set_defaults(fn=cmd_maples_analyse)

    pw = sub.add_parser("power")
    pw.add_argument("--n", type=int, default=100)
    pw.add_argument("--n-sim", type=int, default=400)
    pw.add_argument("--n-boot", type=int, default=1000)
    pw.add_argument("--fresh-boot", action="store_true",
                    help="draw a new resample matrix for every simulated study "
                         "(slower; the default reuses one matrix, which is what "
                         "the analysis itself does)")
    pw.set_defaults(fn=cmd_power)

    st = sub.add_parser("strata")
    st.add_argument("--force", action="store_true")
    st.set_defaults(fn=lambda args: (maples_strata(args.force), 0)[1])

    g = sub.add_parser("gpu")
    g.add_argument("--gpu", type=int, required=True)
    g.add_argument("--min-free-mb", type=int, default=MIN_FREE_MB)
    g.add_argument("--only", default="")
    g.set_defaults(fn=lambda args: worker(args.gpu, args.min_free_mb, args.only))

    fi = sub.add_parser("finish")
    fi.add_argument("--poll", type=int, default=300)
    fi.add_argument("--threads", type=int, default=4)
    fi.add_argument("--max-hours", type=float, default=24.0)
    fi.set_defaults(fn=cmd_finish)

    sub.add_parser("plan").set_defaults(fn=cmd_plan)

    args = ap.parse_args(argv)
    os.chdir(EXP_ROOT)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
