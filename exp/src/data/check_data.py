"""
Sanity check for all five retinal-vessel datasets.

Prints per-dataset: file counts, split sizes, image/label shapes, vessel
foreground fraction (inside FOV), inter-observer agreement where a 2nd
observer exists, and FOV coverage. Saves one overlay PNG per dataset to
exp/runs/data_check/.

Usage:  python exp/src/data/check_data.py [dataset ...]
"""
from __future__ import annotations

import os
import sys
import time
import collections

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.datasets import (  # noqa: E402
    DATASETS, RUNS_ROOT, load_dataset, read_image, read_binary,
)

OUT_DIR = os.path.join(str(RUNS_ROOT), "data_check")

# how many images to sample for the shape / foreground statistics
SAMPLE = int(os.environ.get("CHECK_SAMPLE", "12"))


def _overlay(rec, out_png):
    """Save image with the vessel label in red and the FOV boundary in cyan."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = read_image(rec["image_path"])
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    img = img[..., :3].astype(np.float32)
    img = img / max(img.max(), 1.0)

    lab = read_binary(rec["label_path"])
    ov = img.copy()
    ov[lab] = [1.0, 0.15, 0.15]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(img)
    ax[0].set_title("%s  %s" % (rec["image_id"], img.shape[:2]))
    ax[1].imshow(ov)
    ax[1].set_title("label overlay (fg=%.2f%%)" % (100.0 * lab.mean()))
    if rec.get("fov_path") and os.path.exists(rec["fov_path"]):
        fov = read_binary(rec["fov_path"])
        ax[1].contour(fov, levels=[0.5], colors="cyan", linewidths=0.8)
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def check(name):
    t0 = time.time()
    recs = load_dataset(name)
    ntr = sum(r["split"] == "train" for r in recs)
    nte = len(recs) - ntr

    print("=" * 74)
    print("%s   total=%d  train=%d  test=%d  subjects=%d"
          % (name.upper(), len(recs), ntr, nte,
             len({r["subject_id"] for r in recs})))

    # --- existence -----------------------------------------------------
    miss = collections.Counter()
    for r in recs:
        for k in ("image_path", "label_path", "label2_path", "fov_path"):
            p = r.get(k)
            if p is None:
                continue
            if not os.path.exists(p):
                miss[k] += 1
    print("  missing files:", dict(miss) if miss else "none")
    n2 = sum(bool(r.get("label2_path")) for r in recs)
    nfov = sum(bool(r.get("fov_path")) for r in recs)
    print("  2nd-observer labels: %d/%d      FOV masks: %d/%d"
          % (n2, len(recs), nfov, len(recs)))
    if "disease" in recs[0]:
        print("  disease classes:", dict(collections.Counter(r["disease"] for r in recs)))

    # --- shapes + foreground on a sample -------------------------------
    idx = np.linspace(0, len(recs) - 1, min(SAMPLE, len(recs))).astype(int)
    shapes, fracs, fov_cov, agree = collections.Counter(), [], [], []
    gt_in_fov = []
    for i in idx:
        r = recs[int(i)]
        img = read_image(r["image_path"])
        lab = read_binary(r["label_path"])
        assert img.shape[:2] == lab.shape[:2], \
            "shape mismatch %s: img %s vs label %s" % (r["image_id"], img.shape, lab.shape)
        shapes[img.shape[:2] + (img.shape[2] if img.ndim == 3 else 1,)] += 1

        if r.get("fov_path") and os.path.exists(r["fov_path"]):
            fov = read_binary(r["fov_path"])
            fov_cov.append(fov.mean())
            fracs.append(lab[fov].mean() if fov.any() else lab.mean())
            # Annotated vessels are by definition inside the FOV, so the
            # fraction of label pixels the mask keeps must be ~1.0. This is the
            # criterion the FOV generator was tuned on (see DATASETS.md).
            if lab.any():
                gt_in_fov.append(float(np.logical_and(lab, fov).sum()) / float(lab.sum()))
        else:
            fracs.append(lab.mean())

        if r.get("label2_path") and os.path.exists(r["label2_path"]):
            l2 = read_binary(r["label2_path"])
            inter = np.logical_and(lab, l2).sum()
            union = np.logical_or(lab, l2).sum()
            if union:
                agree.append(inter / union)

    print("  shapes (H,W,C) over %d sampled: %s"
          % (len(idx), dict(shapes)))
    print("  vessel foreground fraction in FOV: mean=%.4f  min=%.4f  max=%.4f"
          % (float(np.mean(fracs)), float(np.min(fracs)), float(np.max(fracs))))
    if fov_cov:
        print("  FOV coverage of frame: mean=%.3f  min=%.3f  max=%.3f"
              % (float(np.mean(fov_cov)), float(np.min(fov_cov)), float(np.max(fov_cov))))
    if gt_in_fov:
        print("  GT vessel pixels inside FOV: mean=%.4f  worst=%.4f"
              % (float(np.mean(gt_in_fov)), float(np.min(gt_in_fov))))
    if agree:
        print("  inter-observer IoU (1st vs 2nd): mean=%.3f  min=%.3f"
              % (float(np.mean(agree)), float(np.min(agree))))

    # --- overlay -------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, name + "_overlay.png")
    _overlay(recs[0], out_png)
    print("  overlay -> %s     (%.1fs)" % (out_png, time.time() - t0))
    return len(recs)


if __name__ == "__main__":
    names = sys.argv[1:] or DATASETS
    total = 0
    for n in names:
        total += check(n)
    print("=" * 74)
    print("ALL OK - %d records across %d datasets" % (total, len(names)))
    print("overlays in", OUT_DIR)
