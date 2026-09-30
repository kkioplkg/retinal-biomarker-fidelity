"""E3 -- external downstream validity: do ReliSeg's biomarkers discriminate disease better?

The mask-bearing sets are too small to settle it (HRF n=45, FIVES n=800 with a
ceiling effect). E3 asks the same question on sets that have **disease labels
and no vessel masks**, where n is 1.7k-6.4k: freeze a segmenter, measure the
four primary biomarkers on every image, and score how well a fixed classifier
recovers the disease label from that biomarker vector. Two frozen segmenters
(baseline vs ReliSeg) give two biomarker tables and one paired difference.

Protocol (DECISIONS.md 2026-09-17 00:30) -- the parts that matter:

*   **Segmenters are frozen before any external label is touched.** This module
    never trains or tunes a segmenter; it loads a checkpoint, records its
    SHA-256 in meta.json, and refuses to append to a biomarker table that was
    produced by a different checkpoint.
*   **Nested CV on the pre-registered unit** (`COHORT_ROLE`, frozen
    2026-09-17 11:35). The outer loop is a grouped stratified 5-fold (x3
    repeats); hyperparameters are chosen by a grouped inner 3-fold *inside each
    outer training fold*, so no fold ever selects on its own test rows. ODIR's
    two eyes of one patient share a `subject_id` and never straddle a fold.
    APTOS / Messidor-2 / IDRiD publish no usable patient id, so those three are
    IMAGE-level throughout -- splits, grouping and bootstrap -- and no
    patient-level generalisation is claimed for them.
*   **Metrics.** Macro one-vs-rest AUROC and AUPRC for every target; for the
    ordinal DR grade also quadratic-weighted kappa (argmax and
    expected-grade-rounded) and the binary referable-DR (>=2) AUC.
*   **Paired bootstrap** on the difference between two checkpoint tags,
    resampling the pre-registered unit: patients for ODIR (so both eyes move
    together), images for APTOS / Messidor-2 / IDRiD.
*   **Confounder control.** Every target is fitted twice: `bio` (the four
    biomarkers only) and `bio+cov` (plus image-quality and acquisition
    covariates: Laplacian variance, brightness, contrast, CNR, FOV area,
    native resolution, disc radius, and age/sex where published). Both are
    reported -- a gain that survives the covariates is the interesting one.

Inference convention
--------------------
Crop to the FOV bounding box, resize the longest side to `--resize-longest`
(default 1536, the training convention), sliding-window predict with the checkpoint's training patch
size, resize the probability map back to the cropped native size, threshold at
0.5 inside the FOV. Cropping first is what makes the scale comparable across a
set like APTOS, whose images carry wildly different black borders; it does mean
the retina can land at a different pixels-per-degree than in HRF/FIVES
training, so both the native size and the
realised scale are written to every row and `--resolution-record` dumps them
separately for the fidelity-vs-resolution analysis.

CLI
---
    # stage 1: biomarkers under one frozen checkpoint (resumable)
    python -m src.pivot.e3_external bio --dataset aptos2019 \
        --ckpt runs/seg/hrf/seed0/best.pt --device cuda:0

    # smoke (CPU, 50 images)
    python -m src.pivot.e3_external bio --dataset aptos2019 \
        --ckpt runs/seg/hrf/seed0/best.pt --device cpu --limit 50

    # stage 2: classification + paired comparison between frozen checkpoints
    python -m src.pivot.e3_external clf --dataset aptos2019 \
        --tags hrf_seed0_best,fives_seed0_reliseg_last
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.pivot.common import PIVOT_DIR, image_features, prob_features

E3_DIR = os.path.join(PIVOT_DIR, "e3")

#: the four primary biomarkers, skan pipeline. PVBM is 10-30x slower per image
#: and adds a second estimate of the same four quantities; at 3.6k-6.4k images
#: it is opt-in (--pvbm), and the agreement between pipelines is already
#: established on the mask-bearing sets (Gate A).
SKAN_COLS = ["FD_skan", "tortuosity_skan", "density_skan", "total_length_skan"]
PVBM_COLS = ["FD_pvbm", "tortuosity_pvbm", "density_pvbm", "total_length_pvbm"]

#: acquisition / quality covariates for the adjusted model
COV_COLS = [
    "q_lap_var", "q_brightness", "feat_contrast", "feat_cnr",
    "feat_fov_area_px", "feat_longest", "native_longest", "scale",
    "disc_r",
]

#: inference scale for the external sets: the training convention
#: (DECISIONS 2026-09-17 02:00 withdrew the earlier 1024 -- resolution is a
#: mechanism variable and must not carry an extra downsampling confound).
DEFAULT_RESIZE_LONGEST = 1536
DEFAULT_PATCH = 768

SEED = 0
N_REPEAT = 3
N_SPLITS = 5
N_BOOT = 1000

# =========================================================================
# E3 pre-registration -- frozen 2026-09-17 11:35, before any external label
# was read by a model (DECISIONS 2026-09-17 13:30, paper2 review).
#
#   primary cohort      APTOS-2019 (3,662 images)
#   primary endpoint    referable DR (grade >= 2, src/data/external.py
#                       REFERABLE_THRESHOLD) AUROC
#   primary model       logistic regression on the 4-biomarker skan panel
#                       plus the acquisition/quality covariates ("bio+cov")
#   primary contrast    delta-AUC  ReliSeg - baseline, FIVES seeds 0-2,
#                       reported per seed AND as the seed mean
#   uncertainty         image-level paired bootstrap (N_BOOT = 1000 resamples)
#   secondary           GBDT; QWK on the 0-4 grade; the "bio"-only featureset
#   replication         IDRiD (516), Messidor-2 (1,582 = 1,744 minus the
#                       162-image MAPLES-DR overlap, dropped at clf time only)
#   sensitivity         ODIR-512 (low-resolution mirror; no positive claim)
#
# Resampling / splitting unit, frozen with the rest: APTOS, Messidor-2 and
# IDRiD mirrors carry no usable patient identifier (Messidor-2's original
# filenames are lost, so its 874 two-eye examinations cannot be recovered;
# APTOS has none; IDRiD is one image per subject).  Those three are therefore
# IMAGE-level throughout -- splits, grouping and bootstrap -- and the paper
# makes no patient-level generalisation claim for them.  ODIR-5K does carry
# real patient IDs (3,358 patients / 6,392 eyes) and keeps subject-level
# grouping so two eyes of one patient never straddle a fold.
# =========================================================================
#: dataset -> (role, unit of splitting/bootstrap)
COHORT_ROLE = {
    "aptos2019": ("primary", "image"),
    "idrid": ("replication", "image"),
    "messidor2": ("replication", "image"),
    "odir5k": ("sensitivity", "subject"),
}
#: Cohort exclusions applied at CLASSIFICATION time, never at bio time
#: (DECISIONS 2026-09-17 13:15).  Messidor-2 shares 162 images with MAPLES-DR;
#: those are dropped so the cohort cannot overlap a set used elsewhere in the
#: project, taking it from 1,744 to 1,582.  The exclusion is deliberately NOT
#: applied in ``run_bio``: bio.csv keeps every row it measured, so the
#: measurement stage stays reproducible and the exclusion stays auditable as a
#: single, reversible analysis decision.
CLF_EXCLUDE = {
    "messidor2": os.path.join("data", "external", "messidor2",
                              "overlap_maplesdr.csv"),
}
#: column in an exclusion CSV holding the image_id to drop
CLF_EXCLUDE_COL = {"messidor2": "messidor2_id"}


def load_exclude(dataset: str, path: Optional[str] = None) -> Tuple[set, str]:
    """(image_ids to drop, provenance string) for `dataset`.

    ``path`` overrides the pre-registered default; the literal string "none"
    disables the exclusion (for the sensitivity run that quantifies it).
    """
    if path is not None and str(path).lower() == "none":
        return set(), "disabled by --exclude-list none"
    f = path or CLF_EXCLUDE.get(dataset)
    if not f:
        return set(), "no exclusion pre-registered for %s" % dataset
    if not os.path.exists(f):
        raise SystemExit(
            "exclusion list %s not found. It is pre-registered for %s "
            "(DECISIONS 2026-09-17 13:15); pass --exclude-list none only for "
            "an explicitly labelled sensitivity run." % (f, dataset))
    col = CLF_EXCLUDE_COL.get(dataset, "image_id")
    with open(f, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and col not in rows[0]:
        col = "image_id" if "image_id" in rows[0] else list(rows[0])[0]
    ids = {r[col] for r in rows if r.get(col)}
    return ids, "%s (%d ids, column %r)" % (f.replace("\\", "/"), len(ids), col)


PRIMARY_TARGET = "y_referable"
PRIMARY_CLF = "logreg"
PRIMARY_FEATURESET = "bio+cov"
PRIMARY_COHORT = "aptos2019"


def grouping_for(dataset: str, df) -> Tuple[np.ndarray, str]:
    """Grouping vector for CV and for the paired bootstrap, per COHORT_ROLE."""
    unit = COHORT_ROLE.get(dataset, ("replication", "image"))[1]
    if unit == "subject" and "subject_id" in df.columns:
        return df["subject_id"].to_numpy(), "subject"
    return np.asarray(df.index, dtype=object), "image"


# =========================================================================
# checkpoint identity -- the frozen-segmenter guard
# =========================================================================
def sha256_of(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def ckpt_tag(path: str) -> str:
    """A stable, readable tag for a checkpoint path.

    runs/seg/hrf/seed0/best.pt              -> hrf_seed0_best
    runs/pivot/fives/seed0/reliseg/last.pt  -> fives_seed0_reliseg_last
    runs/pivot/ft_hrf_seed0_cfloss/best.pt  -> ft_hrf_seed0_cfloss_best
    """
    p = os.path.normpath(os.path.abspath(path)).replace("\\", "/").split("/")
    stem = os.path.splitext(p[-1])[0]
    parts = []
    for seg in p[:-1][::-1]:
        if seg in ("runs", "seg", "pivot", "exp") or seg.endswith(":"):
            break
        parts.append(seg)
    return "_".join(parts[::-1] + [stem]) or stem


# =========================================================================
# stage 1 -- biomarkers under a frozen segmenter
# =========================================================================
def _crop_to_fov(img: np.ndarray, fov: np.ndarray, pad: int = 0):
    """Bounding box of the FOV. Returns (img, fov, (y0, y1, x0, x1))."""
    ys, xs = np.where(fov > 0)
    if ys.size == 0:
        h, w = fov.shape[:2]
        return img, fov, (0, h, 0, w)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(fov.shape[0], int(ys.max()) + 1 + pad)
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(fov.shape[1], int(xs.max()) + 1 + pad)
    return img[y0:y1, x0:x1], fov[y0:y1, x0:x1], (y0, y1, x0, x1)


def _quality(img: np.ndarray, fov: np.ndarray) -> Dict[str, float]:
    """Image-quality covariates: focus (Laplacian variance) and brightness.

    Computed on the green channel inside the FOV, and normalised by the FOV
    linear size so a sharper *and larger* image is not scored as sharper only
    because it has more pixels.
    """
    import cv2

    g = img[..., 1] if img.ndim == 3 else img
    g = g.astype(np.float32)
    m = fov > 0
    lap = cv2.Laplacian(g, cv2.CV_32F)
    n = max(float(m.sum()), 1.0)
    scale = np.sqrt(n)
    return {
        "q_lap_var": float((lap[m] ** 2).mean() * scale / 1000.0),
        "q_brightness": float(g[m].mean()) if m.any() else float("nan"),
    }


def _predict_prob(model, img: np.ndarray, fov: np.ndarray, patch: int,
                  resize_longest: Optional[int], device, amp: bool,
                  sw_batch: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Sliding-window probability map at the *input* (cropped native) resolution."""
    import cv2
    import torch

    from src.seg.data import _resize_longest, normalize
    from src.seg.infer import sliding_window_predict, to_native

    native_hw = img.shape[:2]
    im = img
    fv = fov
    if resize_longest and max(native_hw) != resize_longest:
        im = _resize_longest(img, resize_longest, cv2.INTER_AREA)
        fv = (_resize_longest(fov * 255, resize_longest, cv2.INTER_LINEAR) > 127
              ).astype(np.uint8)

    m = fv > 0
    if m.sum() < 16:
        m = np.ones(fv.shape, bool)
    pix = im[m].astype(np.float32)
    mean, std = pix.mean(axis=0), np.maximum(pix.std(axis=0), 1e-3)
    x = torch.from_numpy(normalize(im, mean.astype(np.float32), std.astype(np.float32)))

    prob_r = sliding_window_predict(model, x, patch, device, stride=patch // 2,
                                    amp=amp, batch_size=sw_batch)
    return to_native(prob_r, native_hw), (im.shape[0], im.shape[1])


def _row_for(rec: dict, model, patch: int, resize_longest: Optional[int],
             device, amp: bool, sw_batch: int, run_pvbm: bool,
             crop_fov: bool, threshold: float) -> Dict[str, object]:
    import cv2

    from src.bio.biomarkers import compute_all
    from src.data.external import ensure_fov
    from src.seg.data import _imread_color, _imread_gray

    t0 = time.time()
    img = _imread_color(rec["image_path"])
    native_hw = img.shape[:2]
    ensure_fov(rec)
    fov = (_imread_gray(rec["fov_path"]) > 127).astype(np.uint8)
    if fov.shape[:2] != native_hw:
        fov = (cv2.resize(fov * 255, (native_hw[1], native_hw[0]),
                          interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

    if crop_fov:
        img, fov, box = _crop_to_fov(img, fov)
    else:
        box = (0, native_hw[0], 0, native_hw[1])

    prob, infer_hw = _predict_prob(model, img, fov, patch, resize_longest,
                                   device, amp, sw_batch)
    mask = ((prob >= threshold) & (fov > 0)).astype(np.uint8)

    bio = compute_all(mask, fov, image=img, run_pvbm=run_pvbm, run_skan=True)
    row: Dict[str, object] = {
        "image_id": rec["image_id"],
        "subject_id": rec["subject_id"],
        "split": rec["split"],
        "eye": rec.get("eye") or "",
    }
    for k, v in rec["labels"].items():
        row["y_" + k] = v
    for c in (SKAN_COLS + (PVBM_COLS if run_pvbm else [])):
        row[c] = bio.get(c)
    for k in ("disc_cx", "disc_cy", "disc_r", "disc_confident"):
        row[k] = bio.get(k)

    row.update(image_features(img, fov))
    row.update(_quality(img, fov))
    row.update(prob_features(prob, mask, fov, image=img))

    # --- resolution record ------------------------------------------------
    row.update({
        "native_h": int(native_hw[0]), "native_w": int(native_hw[1]),
        "native_longest": int(max(native_hw)),
        "crop_h": int(img.shape[0]), "crop_w": int(img.shape[1]),
        "crop_box": "%d:%d:%d:%d" % box,
        "infer_h": int(infer_hw[0]), "infer_w": int(infer_hw[1]),
        # realised scale: inference pixels per cropped-native pixel. 1.0 means
        # the network saw the retina at its native sampling.
        "scale": float(max(infer_hw)) / float(max(img.shape[:2])),
        "pred_fg_frac": float(mask.sum()) / max(1.0, float(fov.sum())),
        "seconds": round(time.time() - t0, 2),
    })
    return row


def run_bio(dataset: str, ckpt: str, device_str: str = "cuda:0",
            out_root: str = E3_DIR, tag: Optional[str] = None,
            limit: Optional[int] = None, resize_longest: Optional[int] = None,
            patch: Optional[int] = None, amp: bool = True, sw_batch: int = 4,
            run_pvbm: bool = False, crop_fov: bool = True,
            threshold: float = 0.5, force: bool = False,
            resolution_record: bool = False) -> str:
    """Measure biomarkers for every image of `dataset` under one frozen checkpoint.

    Resumable: rows already in bio.csv are skipped, so a long APTOS pass can be
    run in chunks (or restarted after an OOM) without recomputing.
    """
    import torch

    from src.data.external import load_external
    from src.seg.infer import load_model

    tag = tag or ckpt_tag(ckpt)
    out_dir = os.path.join(out_root, dataset, tag)
    os.makedirs(out_dir, exist_ok=True)
    bio_csv = os.path.join(out_dir, "bio.csv")
    meta_path = os.path.join(out_dir, "meta.json")

    digest = sha256_of(ckpt)
    if os.path.exists(meta_path) and not force:
        old = json.load(open(meta_path, encoding="utf-8"))
        if old.get("ckpt_sha256") != digest:
            raise SystemExit(
                "refusing to extend %s: it was produced by checkpoint %s..., "
                "this one is %s... . Segmenters are frozen per checkpoint tag; "
                "use a different --tag or pass --force."
                % (bio_csv, old.get("ckpt_sha256", "?")[:12], digest[:12]))

    device = torch.device(device_str if (device_str.startswith("cpu")
                                         or torch.cuda.is_available()) else "cpu")
    model, ck = load_model(ckpt, device)
    cfg = ck.get("config", {}) or {}
    patch = int(patch or cfg.get("patch") or DEFAULT_PATCH)
    train_longest = cfg.get("resize_longest")
    resize_longest = int(resize_longest or DEFAULT_RESIZE_LONGEST)
    if train_longest and int(train_longest) != resize_longest:
        print("[note] checkpoint trained at resize_longest=%s, inferring at %d "
              "(recorded in meta.json; the scale column carries the realised "
              "sampling for the fidelity analysis)" % (train_longest, resize_longest))
    amp = amp and device.type == "cuda"

    recs = load_external(dataset)
    if limit:
        recs = recs[:int(limit)]

    done = set()
    if os.path.exists(bio_csv) and not force:
        with open(bio_csv, newline="", encoding="utf-8") as fh:
            done = {r["image_id"] for r in csv.DictReader(fh)}
    todo = [r for r in recs if r["image_id"] not in done]
    print("%s / %s: %d images, %d already done, %d to do"
          % (dataset, tag, len(recs), len(done), len(todo)))

    rows: List[Dict[str, object]] = []
    t0 = time.time()
    for i, rec in enumerate(todo):
        rows.append(_row_for(rec, model, patch, resize_longest, device, amp,
                             sw_batch, run_pvbm, crop_fov, threshold))
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            el = time.time() - t0
            print("  [%d/%d] %s  %.1fs/img  eta %.0fmin"
                  % (i + 1, len(todo), rows[-1]["image_id"], el / (i + 1),
                     (len(todo) - i - 1) * el / (i + 1) / 60), flush=True)
            _append(bio_csv, rows)
            rows = []
    if rows:
        _append(bio_csv, rows)

    json.dump({
        "dataset": dataset, "tag": tag,
        "ckpt": os.path.abspath(ckpt), "ckpt_sha256": digest,
        "ckpt_train_config": {k: cfg.get(k) for k in
                              ("dataset", "seed", "patch", "resize_longest", "loss")},
        "patch": patch, "stride": patch // 2, "resize_longest": resize_longest,
        "crop_to_fov": bool(crop_fov), "threshold": threshold,
        "pipelines": ["skan"] + (["pvbm"] if run_pvbm else []),
        "device": str(device), "amp": bool(amp),
        "n_images": len(recs),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, open(meta_path, "w", encoding="utf-8"), indent=2)

    if resolution_record:
        _write_resolution_record(bio_csv, os.path.join(out_dir, "resolutions.csv"))
    print("-> %s" % bio_csv)
    return bio_csv


def _append(path: str, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    exists = os.path.exists(path)
    keys: List[str] = []
    if exists:
        with open(path, newline="", encoding="utf-8") as fh:
            keys = next(csv.reader(fh))
    else:
        keys = list(rows[0].keys())
    with open(path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_resolution_record(bio_csv: str, out: str) -> None:
    import pandas as pd

    cols = ["image_id", "native_w", "native_h", "native_longest",
            "crop_w", "crop_h", "infer_w", "infer_h", "scale", "feat_fov_frac"]
    df = pd.read_csv(bio_csv)
    df[[c for c in cols if c in df.columns]].to_csv(out, index=False)
    print("-> %s" % out)


# =========================================================================
# stage 2 -- classification
# =========================================================================
def targets_for(dataset: str, df) -> List[Tuple[str, str]]:
    """[(column, kind)] with kind in {'ordinal', 'multiclass', 'binary'}."""
    out = []
    if "y_dr_grade" in df.columns:
        out.append(("y_dr_grade", "ordinal"))
        out.append(("y_referable", "binary"))
    if "y_dme_risk" in df.columns:
        out.append(("y_dme_risk", "ordinal"))
    if "y_disease" in df.columns:
        out.append(("y_disease", "multiclass"))
    return out


def _grids(kind: str):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    logreg = Pipeline([("sc", StandardScaler()),
                       ("m", LogisticRegression(max_iter=5000))])
    gbdt = HistGradientBoostingClassifier(early_stopping=False,
                                          min_samples_leaf=20,
                                          l2_regularization=1.0)
    return {
        "logreg": (logreg, {"m__C": [0.03, 0.3, 3.0]}),
        "gbdt": (gbdt, {"max_depth": [2, 3], "learning_rate": [0.05, 0.1],
                        "max_iter": [150]}),
    }


def macro_auc(y, proba, classes) -> float:
    from sklearn.metrics import roc_auc_score

    a = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if 0 < yy.sum() < len(yy):
            a.append(roc_auc_score(yy, proba[:, i]))
    return float(np.mean(a)) if a else float("nan")


def macro_ap(y, proba, classes) -> float:
    from sklearn.metrics import average_precision_score

    a = []
    for i, c in enumerate(classes):
        yy = (y == c).astype(int)
        if 0 < yy.sum() < len(yy):
            a.append(average_precision_score(yy, proba[:, i]))
    return float(np.mean(a)) if a else float("nan")


def qwk(y, proba, classes) -> Tuple[float, float]:
    """Quadratic-weighted kappa from argmax, and from the rounded expected grade.

    The expected-grade form uses the ordinal structure the argmax throws away
    (a grade-4 image scored 0.4/0.6 across grades 3 and 4 should not count the
    same as one scored 0.4/0.6 across grades 0 and 4).
    """
    from sklearn.metrics import cohen_kappa_score

    cls = np.asarray(classes, dtype=float)
    pred_arg = np.asarray(classes)[proba.argmax(1)]
    exp = (proba * cls[None, :]).sum(1)
    pred_exp = np.clip(np.rint(exp), cls.min(), cls.max())
    k1 = cohen_kappa_score(y, pred_arg, weights="quadratic")
    k2 = cohen_kappa_score(y, pred_exp.astype(y.dtype), weights="quadratic")
    return float(k1), float(k2)


def nested_cv_proba(X, y, groups, kind: str, classes, n_splits: int = N_SPLITS,
                    n_repeat: int = N_REPEAT, seed: int = SEED, n_jobs: int = 1):
    """Grouped, stratified, *nested* out-of-fold probabilities.

    Outer: StratifiedGroupKFold on the pre-registered unit (``grouping_for``):
    subject for ODIR, so its two eyes of one patient never straddle the split;
    image for APTOS / Messidor-2 / IDRiD, whose mirrors carry no usable patient
    identifier. Inner: a grouped 3-fold GridSearchCV on the outer training rows
    only, so the hyperparameter is never chosen with sight of the rows it is
    scored on.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

    acc = np.zeros((len(y), len(classes)), dtype=float)
    used = np.zeros(len(y), dtype=float)
    for rep in range(n_repeat):
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                     random_state=seed + rep)
        for tr, te in outer.split(X, y, groups):
            base, grid = _grids(kind)[kind]
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                         random_state=seed + rep)
            gs = GridSearchCV(base, grid, scoring="neg_log_loss", cv=inner,
                              n_jobs=n_jobs, refit=True, error_score=np.nan)
            gs.fit(X[tr], y[tr], groups=groups[tr])
            est = gs.best_estimator_
            p = est.predict_proba(X[te])
            # A grouped fold can miss a rare class entirely (IDRiD grade 1 has
            # 25 images). Map by class value and leave the absent column at 0
            # rather than crashing on .index().
            seen = list(est.classes_)
            block = np.zeros((len(te), len(classes)), dtype=float)
            for j, c in enumerate(classes):
                if c in seen:
                    block[:, j] = p[:, seen.index(c)]
            acc[te] += block
            used[te] += 1
    used[used == 0] = 1.0
    return acc / used[:, None]


def paired_bootstrap(y, groups, probas: Dict[str, np.ndarray], classes,
                     ref: str, n_boot: int = N_BOOT, seed: int = SEED):
    """Paired bootstrap of macro-AUC and of (tag - ref) on the pre-registered
    resampling unit (``grouping_for``).

    Groups, not rows, are resampled. For ODIR the group is the patient -- with
    its paired eyes a row-level bootstrap would treat two correlated images as
    independent and shrink the interval. For APTOS / Messidor-2 / IDRiD the
    group IS the image, so this is the image-level bootstrap the E3
    pre-registration specifies.
    """
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    keys = list(probas)
    vals = {k: [] for k in keys}
    diffs = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by_group[uniq[i]] for i in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        cur = {k: macro_auc(y[idx], probas[k][idx], classes) for k in keys}
        for k in keys:
            vals[k].append(cur[k])
            diffs[k].append(cur[k] - cur[ref])
    def ci(d):
        return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                for k, v in d.items() if v}
    pval = {}
    for k, v in diffs.items():
        if k == ref:
            pval[k] = float("nan")       # the reference vs itself is identically 0
        elif v:
            v = np.asarray(v)
            pval[k] = float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))
    return ci(vals), ci(diffs), pval


def _feature_matrix(df, cols: Sequence[str]):
    X = df[[c for c in cols if c in df.columns]].to_numpy(dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def run_clf(dataset: str, tags: Sequence[str], out_root: str = E3_DIR,
            use_pvbm: bool = False, n_boot: int = N_BOOT, n_jobs: int = 1,
            min_rows: int = 100, exclude_list: Optional[str] = None):
    """Fit every target x featureset x classifier for each tag, and compare tags."""
    import pandas as pd

    base = os.path.join(out_root, dataset)
    frames = {}
    for t in tags:
        p = os.path.join(base, t, "bio.csv")
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run the `bio` stage for tag %r first" % (p, t))
        frames[t] = pd.read_csv(p).set_index("image_id")

    # Only images every tag measured: a paired comparison needs the same rows.
    common = None
    for t, d in frames.items():
        common = set(d.index) if common is None else (common & set(d.index))
    common = sorted(common)
    n_before = len(common)
    drop, drop_src = load_exclude(dataset, exclude_list)
    if drop:
        common = [i for i in common if i not in drop]
        print("[exclude] %s: %d -> %d images (%d dropped) via %s"
              % (dataset, n_before, len(common), n_before - len(common),
                 drop_src), flush=True)
    if len(common) < min_rows:
        print("[warn] only %d images common to all tags (%s); results will be noisy"
              % (len(common), ", ".join(tags)))
    frames = {t: d.loc[common] for t, d in frames.items()}
    ref_tag = tags[0]
    ref = frames[ref_tag]

    bio_cols = list(SKAN_COLS) + (list(PVBM_COLS) if use_pvbm else [])
    featuresets = {"bio": bio_cols, "bio+cov": bio_cols + COV_COLS}
    if "y_age" in ref.columns:
        featuresets["bio+cov"] = featuresets["bio+cov"] + ["y_age"]

    rows, drows = [], []
    for ycol, kind in targets_for(dataset, ref):
        # Drop rows whose label is missing, from the target, the grouping and
        # every tag's feature table at once, so all tags see identical rows.
        keep = pd.notna(ref[ycol]).to_numpy()
        y = ref.loc[keep, ycol].to_numpy()
        groups, unit = grouping_for(dataset, ref.loc[keep])
        role = COHORT_ROLE.get(dataset, ("replication", "image"))[0]
        classes = sorted(pd.unique(y))
        if len(classes) < 2:
            continue
        for fs, cols in featuresets.items():
            for clf in ("logreg", "gbdt"):
                probas = {}
                for t, d in frames.items():
                    X = _feature_matrix(d.loc[keep], cols)
                    probas[t] = nested_cv_proba(X, y, groups, clf, classes,
                                                n_jobs=n_jobs)
                for t, p in probas.items():
                    r = dict(dataset=dataset, cohort_role=role,
                             split_unit=unit,
                             n_excluded=n_before - len(common),
                             exclude_src=drop_src,
                             is_primary=bool(dataset == PRIMARY_COHORT
                                             and ycol == PRIMARY_TARGET
                                             and clf == PRIMARY_CLF
                                             and fs == PRIMARY_FEATURESET),
                             tag=t, target=ycol, kind=kind,
                             featureset=fs, clf=clf, n=len(y),
                             n_groups=len(np.unique(groups)),
                             n_classes=len(classes),
                             macro_auroc=macro_auc(y, p, classes),
                             macro_auprc=macro_ap(y, p, classes))
                    if kind == "ordinal":
                        k1, k2 = qwk(y, p, classes)
                        r["qwk_argmax"], r["qwk_expected"] = k1, k2
                    r["class_counts"] = ";".join(
                        "%s=%d" % (c, int((y == c).sum())) for c in classes)
                    rows.append(r)
                    np.save(os.path.join(base, "oof_%s_%s_%s_%s.npy"
                                         % (t, ycol, fs.replace("+", ""), clf)), p)
                ci, dci, pv = paired_bootstrap(y, groups, probas, classes,
                                               ref_tag, n_boot=n_boot)
                for t in probas:
                    drows.append(dict(
                        dataset=dataset, cohort_role=role, split_unit=unit,
                        n_excluded=n_before - len(common),
                        exclude_src=drop_src,
                        is_primary=bool(dataset == PRIMARY_COHORT
                                        and ycol == PRIMARY_TARGET
                                        and clf == PRIMARY_CLF
                                        and fs == PRIMARY_FEATURESET),
                        target=ycol, featureset=fs, clf=clf,
                        tag=t, ref=ref_tag,
                        auc_ci_lo=ci.get(t, (np.nan,) * 2)[0],
                        auc_ci_hi=ci.get(t, (np.nan,) * 2)[1],
                        delta_vs_ref=macro_auc(y, probas[t], classes)
                        - macro_auc(y, probas[ref_tag], classes),
                        delta_ci_lo=dci.get(t, (np.nan,) * 2)[0],
                        delta_ci_hi=dci.get(t, (np.nan,) * 2)[1],
                        delta_p=pv.get(t, np.nan)))
                print("  %-12s %-10s %-8s %-6s done" % (ycol, fs, clf, ""), flush=True)

    res = pd.DataFrame(rows)
    dif = pd.DataFrame(drows)
    os.makedirs(base, exist_ok=True)
    res.to_csv(os.path.join(base, "summary.csv"), index=False)
    dif.to_csv(os.path.join(base, "delta.csv"), index=False)
    with pd.option_context("display.width", 220, "display.max_rows", 300):
        print(res.round(4).to_string(index=False))
        if len(tags) > 1:
            print(dif.round(4).to_string(index=False))
    print("-> %s" % os.path.join(base, "summary.csv"))
    return res, dif


# =========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    b = sub.add_parser("bio", help="measure biomarkers under a frozen checkpoint")
    b.add_argument("--dataset", required=True)
    b.add_argument("--ckpt", required=True)
    b.add_argument("--tag", default=None)
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--resize-longest", type=int, default=DEFAULT_RESIZE_LONGEST)
    b.add_argument("--patch", type=int, default=None)
    b.add_argument("--sw-batch", type=int, default=4)
    b.add_argument("--no-amp", action="store_true")
    b.add_argument("--pvbm", action="store_true", help="also run the PVBM pipeline (slow)")
    b.add_argument("--no-crop-fov", action="store_true")
    b.add_argument("--threshold", type=float, default=0.5)
    b.add_argument("--force", action="store_true")
    b.add_argument("--resolution-record", action="store_true",
                   help="also write resolutions.csv (native size + realised scale)")
    b.add_argument("--out-root", default=E3_DIR)

    c = sub.add_parser("clf", help="classify disease from the biomarker vector")
    c.add_argument("--dataset", required=True)
    c.add_argument("--tags", required=True,
                   help="comma-separated checkpoint tags; the FIRST is the reference")
    c.add_argument("--pvbm", action="store_true")
    c.add_argument("--n-boot", type=int, default=N_BOOT)
    c.add_argument("--n-jobs", type=int, default=1)
    c.add_argument("--out-root", default=E3_DIR)
    c.add_argument("--exclude-list", default=None,
                   help="CSV of image_ids to drop at classification time. "
                        "Defaults to the pre-registered list for the dataset "
                        "(Messidor-2: the 162-image MAPLES-DR overlap, "
                        "1744 -> 1582). Pass 'none' to disable, for an "
                        "explicitly labelled sensitivity run only.")

    a = ap.parse_args(argv)
    if a.stage == "bio":
        run_bio(a.dataset, a.ckpt, device_str=a.device, out_root=a.out_root,
                tag=a.tag, limit=a.limit, resize_longest=a.resize_longest,
                patch=a.patch, amp=not a.no_amp, sw_batch=a.sw_batch,
                run_pvbm=a.pvbm, crop_fov=not a.no_crop_fov,
                threshold=a.threshold, force=a.force,
                resolution_record=a.resolution_record)
    else:
        run_clf(a.dataset, [t.strip() for t in a.tags.split(",") if t.strip()],
                out_root=a.out_root, use_pvbm=a.pvbm, n_boot=a.n_boot,
                n_jobs=a.n_jobs, exclude_list=a.exclude_list)
    return 0


if __name__ == "__main__":
    sys.exit(main())
