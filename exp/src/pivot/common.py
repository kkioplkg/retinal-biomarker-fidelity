"""Shared helpers for the three pivot probes.

Everything here is deliberately torch-free so the CPU biomarker workers stay
light (6 processes x a torch import is 6 x ~400 MB).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
PIVOT_DIR = os.path.join(RESULTS_DIR, "pivot")
CACHE_DIR = os.path.join(PIVOT_DIR, "cache")
RUNS_DIR = os.path.join(EXP_ROOT, "runs")

#: four primary biomarkers x two pipelines (src/eval/biomarker_eval.py)
PRIMARY = ("FD", "tortuosity", "density", "total_length")
PIPES = ("pvbm", "skan")
PRIMARY_COLS = [f"{b}_{p}" for b in PRIMARY for p in PIPES]
PRIMARY_COLS_ZONEB = [f"{b}_{p}_zoneB" for b in PRIMARY for p in PIPES]

DATASETS = ("drive", "chasedb1", "hrf", "fives")

#: canonical dataset name -> name used in results/gateA_biomarker_scales_train.csv
GATEA_DS = {"drive": "DRIVE", "chasedb1": "CHASE_DB1", "hrf": "HRF", "fives": "FIVES"}


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def read_gray(path: str) -> np.ndarray:
    from src.data.datasets import read_image

    a = read_image(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a


def binarize(a: np.ndarray) -> np.ndarray:
    """Binary mask from any label encoding.

    CHASE_DB1 ships its observer masks as boolean PNGs, DRIVE/HRF/FIVES as
    0/255 -- a plain ``> 127`` silently returns an all-zero mask for CHASE.
    """
    if a.dtype == bool:
        return a
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.floating):
        return a > 0.5
    return (a > 127) if a.max() > 1 else (a > 0)


def load_fov(rec: dict, native_hw: Tuple[int, int]) -> np.ndarray:
    """Native-resolution binary FOV, same convention as ``seg.infer.load_native_fov``."""
    import cv2

    p = rec.get("fov_path")
    f = binarize(read_gray(str(p))).astype(np.uint8)
    if f.shape[:2] != tuple(native_hw):
        f = (cv2.resize(f * 255, (native_hw[1], native_hw[0]),
                        interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    return f


def pred_paths(dataset: str, image_id: str, source: str, seed: int = 0
               ) -> Tuple[str, str]:
    """(prob_npy, mask_png) for ``source`` in {pred_test, pred_oof}.

    The file stem drops the FIVES split prefix, matching the manifests written
    by ``src.seg.infer`` (``runs/seg/fives/seed0/pred/prob/100_D.npy``).
    """
    stem = image_id
    if dataset == "fives":
        stem = image_id.split("_", 1)[1]
    if source == "pred_test":
        base = os.path.join(RUNS_DIR, "seg", dataset, f"seed{seed}", "pred")
    elif source == "pred_oof":
        base = os.path.join(RUNS_DIR, "seg_oof", dataset, "pred")
    else:
        raise ValueError(source)
    return (os.path.join(base, "prob", stem + ".npy"),
            os.path.join(base, "mask", stem + ".png"))


def records(dataset: str) -> List[dict]:
    from src.data.datasets import load_dataset

    return load_dataset(dataset)


# --------------------------------------------------------------------------
# cheap observable features (no GT, computable at inference time)
# --------------------------------------------------------------------------
def image_features(image: np.ndarray, fov: np.ndarray) -> Dict[str, float]:
    """Acquisition/quality proxies from the fundus image alone."""
    g = image[..., 1].astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
    m = fov > 0
    v = g[m]
    if v.size == 0:
        return {}
    h, w = g.shape[:2]
    p1, p99 = np.percentile(v, [1.0, 99.0])
    return {
        "feat_h": float(h),
        "feat_w": float(w),
        "feat_longest": float(max(h, w)),
        "feat_fov_area_px": float(m.sum()),
        "feat_fov_frac": float(m.mean()),
        "feat_green_mean": float(v.mean()),
        "feat_green_std": float(v.std()),
        "feat_green_p99_p1": float(p99 - p1),
        # global contrast proxy (Michelson-like on robust percentiles)
        "feat_contrast": float((p99 - p1) / max(p99 + p1, 1e-6)),
    }


def prob_features(prob: np.ndarray, mask: np.ndarray, fov: np.ndarray,
                  image: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Predicted-mask / probability-map statistics available at inference time."""
    m = fov > 0
    p = prob[m].astype(np.float32)
    mk = (mask[m] > 0)
    eps = 1e-6
    pc = np.clip(p, eps, 1.0 - eps)
    ent = -(pc * np.log2(pc) + (1 - pc) * np.log2(1 - pc))
    out = {
        "feat_pred_density": float(mk.mean()),
        "feat_prob_mean": float(p.mean()),
        "feat_prob_std": float(p.std()),
        "feat_prob_entropy": float(ent.mean()),
        "feat_frac_lowconf": float(((p > 0.2) & (p < 0.8)).mean()),
        "feat_frac_mid": float(((p > 0.4) & (p < 0.6)).mean()),
        "feat_prob_mass": float(p.mean()),              # soft density
        "feat_soft_hard_ratio": float(p.mean() / max(mk.mean(), 1e-8)),
        # mean probability inside the predicted foreground: how confident the
        # accepted vessel pixels are
        "feat_prob_in_fg": float(p[mk].mean()) if mk.any() else float("nan"),
    }
    if image is not None:
        g = image[..., 1].astype(np.float32) if image.ndim == 3 else image.astype(np.float32)
        gv = g[m]
        if mk.any() and (~mk).any():
            mu_v, mu_b = float(gv[mk].mean()), float(gv[~mk].mean())
            sd_b = float(gv[~mk].std())
            out["feat_cnr"] = (mu_b - mu_v) / max(sd_b, 1e-6)
            out["feat_vessel_bg_diff"] = mu_b - mu_v
    return out


# --------------------------------------------------------------------------
# sigma (single source of truth)
# --------------------------------------------------------------------------
def sigma_table(path: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """``{canonical dataset: {biomarker col: sigma}}`` from the Gate A train scales."""
    import pandas as pd

    if path is None:
        path = os.path.join(RESULTS_DIR, "gateA_biomarker_scales_train.csv")
    df = pd.read_csv(path)
    inv = {v: k for k, v in GATEA_DS.items()}
    out: Dict[str, Dict[str, float]] = {}
    for r in df.itertuples(index=False):
        ds = inv.get(str(r.dataset), str(r.dataset).lower())
        try:
            s = float(r.sigma)
        except (TypeError, ValueError):
            continue
        out.setdefault(ds, {})[str(r.biomarker)] = s
    return out


def json_dump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)
