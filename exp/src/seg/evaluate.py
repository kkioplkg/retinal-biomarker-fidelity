"""Evaluate segmentation predictions and write results/seg_per_image.csv.

Metrics come from ``src.topo.metrics.evaluate_all`` (Dice/F1, clDice, Betti-0/1
errors, BCS, Junction-F1 ...), which is written under S2 by another job.  If
that module is not importable yet, a clearly-labelled minimal fallback
(Dice/F1/precision/recall/accuracy/specificity/AUC/AP) is used instead and the
``metrics_source`` column of the CSV says so.

    cd exp
    python -m src.seg.evaluate --dataset drive --seed 0 \
        --pred-dir runs/seg/drive/seed0/pred
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.seg import data as segdata

cv2.setNumThreads(0)

DEFAULT_OUT = os.path.join("results", "seg_per_image.csv")


# --------------------------------------------------------------------------
# metric backend
# --------------------------------------------------------------------------
def _load_topo_metrics():
    try:
        from src.topo.metrics import evaluate_all  # type: ignore
        return evaluate_all
    except Exception:
        return None


def _fallback_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) ROC AUC; ties handled by average ranks."""
    pos = labels > 0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(s), dtype=np.float64)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty_like(ranks)
    r[order] = ranks
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fallback_ap(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = labels > 0
    if pos.sum() == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = pos[order].astype(np.float64)
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def fallback_evaluate(pred: np.ndarray, gt: np.ndarray, fov: np.ndarray,
                      prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    m = fov > 0
    p = (pred[m] > 0)
    g = (gt[m] > 0)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    tn = float(np.logical_and(~p, ~g).sum())
    dice = 2 * tp / max(1e-9, 2 * tp + fp + fn)
    prec = tp / max(1e-9, tp + fp)
    rec = tp / max(1e-9, tp + fn)
    out = {
        "dice": dice,
        "f1": dice,
        "iou": tp / max(1e-9, tp + fp + fn),
        "precision": prec,
        "sensitivity": rec,
        "specificity": tn / max(1e-9, tn + fp),
        "accuracy": (tp + tn) / max(1e-9, tp + tn + fp + fn),
    }
    if prob is not None:
        s = prob[m].astype(np.float64)
        y = g.astype(np.uint8)
        out["auc"] = _fallback_auc(s, y)
        out["ap"] = _fallback_ap(s, y)
    return out


def call_evaluate_all(fn, pred, gt, fov, prob) -> Dict[str, float]:
    """Call ``evaluate_all`` however it happens to be declared.

    Arguments are bound **by parameter name**, never by position: the current
    ``src.topo.metrics.evaluate_all`` is declared ``(prob, pred, gt, fov=...)``,
    so a positional (pred, gt, fov) call would silently pass the binary mask as
    the probability map.
    """
    try:
        params = list(inspect.signature(fn).parameters)
    except (TypeError, ValueError):
        params = []

    # Exact names only.  Substring matching is a trap here: "probe_px" contains
    # "prob" and would silently receive the probability array.
    NAMES = {
        "prob": ("prob", "probs", "probability", "prob_map", "probmap",
                 "y_score", "score", "scores", "soft", "softmax", "logits"),
        "pred": ("pred", "preds", "prediction", "y_pred", "pred_mask",
                 "mask_pred", "seg", "segmentation"),
        "gt": ("gt", "target", "targets", "label", "labels", "y_true",
               "gt_mask", "truth", "ref", "reference"),
        "fov": ("fov", "fov_mask", "mask", "roi", "valid", "valid_mask"),
    }
    VALUES = {"prob": prob, "pred": pred, "gt": gt, "fov": fov}

    def _bind(name: str):
        n = name.lower()
        for role, names in NAMES.items():
            if n in names:
                return ("ok", VALUES[role])
        return ("skip", None)

    kwargs = {}
    unknown = []
    for p in params:
        status, val = _bind(p)
        if status == "ok":
            kwargs[p] = val
        else:
            unknown.append(p)
    required = [p for p, sp in inspect.signature(fn).parameters.items()
                if sp.default is inspect.Parameter.empty] if params else []
    if not all(r in kwargs for r in required):
        raise TypeError(
            "cannot bind evaluate_all{} by name (unbound required: {})".format(
                tuple(params), [r for r in required if r not in kwargs]))
    res = fn(**kwargs)
    if not isinstance(res, dict):
        raise TypeError("evaluate_all returned {}, expected dict".format(type(res)))
    return {str(k): (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                     else v) for k, v in res.items()}


# --------------------------------------------------------------------------
def read_gt(rec_like: dict, native_hw) -> np.ndarray:
    lab = segdata._imread_gray(str(rec_like["label_path"]))
    gt = (lab > 127).astype(np.uint8)
    if gt.shape[:2] != tuple(native_hw):
        gt = (cv2.resize(gt * 255, (native_hw[1], native_hw[0]),
                         interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    return gt


def read_fov(rec_like: dict, native_hw) -> np.ndarray:
    from src.seg.infer import load_native_fov
    return load_native_fov(rec_like, native_hw)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="evaluate vessel segmentations")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-dir", required=True,
                    help="directory written by src.seg.infer (holds manifest.csv)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default="unet")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--append", type=int, default=1)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    man_path = os.path.join(args.pred_dir, "manifest.csv")
    if not os.path.exists(man_path):
        print("[error] no manifest at {}".format(man_path))
        return 2
    with open(man_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("[error] empty manifest")
        return 2

    evaluate_all = _load_topo_metrics()
    source = "src.topo.metrics.evaluate_all" if evaluate_all else "seg.evaluate:fallback"
    if evaluate_all is None:
        print("[warn] src.topo.metrics.evaluate_all not importable; "
              "using the minimal internal fallback (Dice/F1/IoU/precision/"
              "sensitivity/specificity/accuracy/AUC/AP)")

    results: List[dict] = []
    for i, r in enumerate(rows):
        native_hw = (int(r["native_h"]), int(r["native_w"]))
        mask = cv2.imread(r["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print("[warn] missing prediction {}".format(r["mask_path"]))
            continue
        pred = (mask > 127).astype(np.uint8)
        gt = read_gt(r, native_hw)
        fov = read_fov(r, native_hw)
        prob = None
        if r.get("prob_path") and os.path.exists(r["prob_path"]):
            prob = np.load(r["prob_path"]).astype(np.float32)

        row = {
            "dataset": r.get("dataset", segdata.canon(args.dataset)),
            "method": args.method,
            "seed": args.seed,
            "split": r.get("split", "test"),
            "image": r["image"],
            "subject_id": r.get("subject_id", ""),
            "metrics_source": source,
        }
        if evaluate_all is not None:
            try:
                m = call_evaluate_all(evaluate_all, pred, gt, fov, prob)
            except Exception as exc:
                print("[warn] evaluate_all failed on {}: {}; falling back".format(
                    r["image"], exc))
                m = fallback_evaluate(pred, gt, fov, prob)
                row["metrics_source"] = "seg.evaluate:fallback"
        else:
            m = fallback_evaluate(pred, gt, fov, prob)
        row.update(m)
        results.append(row)
        print("[{}/{}] {} ".format(i + 1, len(rows), r["image"]) + " ".join(
            "{}={:.4f}".format(k, v) for k, v in m.items()
            if isinstance(v, float)), flush=True)

    if not results:
        print("[error] nothing evaluated")
        return 2

    cols: List[str] = []
    for row in results:
        for k in row:
            if k not in cols:
                cols.append(k)

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    existing: List[dict] = []
    if args.append and os.path.exists(out_path):
        with open(out_path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))
        keep = {(r_["dataset"], r_.get("method", ""), str(r_["seed"]), r_["image"])
                for r_ in results}
        existing = [e for e in existing
                    if (e.get("dataset"), e.get("method", ""), str(e.get("seed")),
                        e.get("image")) not in keep]
        for e in existing:
            for k in e:
                if k not in cols:
                    cols.append(k)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for e in existing:
            w.writerow(e)
        for row in results:
            w.writerow(row)

    numeric = [k for k in cols if k not in
               ("dataset", "method", "seed", "split", "image", "subject_id",
                "metrics_source")]
    summary = {}
    for k in numeric:
        vals = [row[k] for row in results
                if isinstance(row.get(k), float) and not np.isnan(row[k])]
        if vals:
            summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    used = sorted({row["metrics_source"] for row in results})
    print("\n== {} / seed {} / {} images ({}) ==".format(
        results[0]["dataset"], args.seed, len(results), "; ".join(used)))
    for k, v in summary.items():
        print("  {:<16s} {:.4f} +- {:.4f}".format(k, v["mean"], v["std"]))
    print("wrote {} rows -> {}".format(len(results), os.path.abspath(out_path)))

    with open(os.path.join(args.pred_dir, "eval_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump({"dataset": results[0]["dataset"], "seed": args.seed,
                   "method": args.method, "n_images": len(results),
                   "metrics_source": used, "summary": summary}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
