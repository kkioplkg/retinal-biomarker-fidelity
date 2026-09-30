"""m_R -- the scale that puts ``C_geom`` on the same footing as ``R``.

Proposal section 3.1.2 defines the deployment utility

    U(e) = p_e R_miss - lambda (1 - p_e) R_false - eta C_geom

with all three terms dimensionless and comparable.  The implementation did not
satisfy that: :func:`src.rigr.utility.geom_cost` normalises ``C_geom`` by its
own MAD, so it is O(1) (median ~2.6), while the sigma-standardised single-event
harms the BTR heads predict are O(1e-3).  At the pre-registered ``eta = 0.1``
the third term therefore dominated by three orders of magnitude:
``corr(U, -C_geom) = 0.9999``.  Risk mode accepted almost nothing (LODO and
STARE reported ``TRR = 0``) and ``U`` anti-correlated with the realised harm in
Exp2 -- it was not risk-guided at all, it was a geometry-cost ranking.

The fix is a per-(dataset, seed) constant

    m_R = median over the dataset's TRAINING candidates of R_miss

so that ``C_geom * m_R`` lives on the scale of ``R``.  ``lambda = 1`` and
``eta = 0.1`` keep their pre-registered values; only the units change.

Two properties matter for leakage:

* the median is taken over ``runs/rigr_data/<ds>/rfalse_train.csv``, which
  ``build_data`` writes from the **training split only** (it hard-codes
  ``split="train"``), so no test image influences the operating point;
* it is computed once, at fit time, and frozen in
  ``runs/rigr_models/<ds>/seed<k>/geom_scale.json`` -- ``run_rigr`` reads that
  file rather than recomputing anything per image.

For the LODO block the head is the held-out variant
``btr_miss_wo_<D>.joblib`` and the candidates are
``runs/rigr_data/lodo_<D>/rfalse_train.csv``, i.e. the three source domains --
the held-out domain enters neither.

Usage::

    python -m src.rigr.geom_scale --data runs/rigr_data/drive \\
        --out runs/rigr_models/drive/seed0
    python -m src.rigr.geom_scale --data runs/rigr_data/lodo_drive \\
        --out runs/rigr_models/lodo_drive/seed0 --btr_variant wo_DRIVE
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional, Sequence

import numpy as np

DEFAULT_RUNS_DIR = os.path.join("runs", "btr", "habs_trainonly")


def compute_geom_scale(data_dirs: Sequence[str],
                       btr_runs_dir: str = DEFAULT_RUNS_DIR,
                       btr_variant: str = "all",
                       target: Optional[str] = None) -> dict:
    """``m_R`` = median R_miss over the training candidates of ``data_dirs``."""
    import pandas as pd

    from src.c1.btr import BTRHead, head_path

    hp = head_path("miss", btr_runs_dir, btr_variant, target)
    head = BTRHead.load(hp)

    frames = []
    for d in data_dirs:
        f = os.path.join(d, "rfalse_train.csv")
        if os.path.exists(f):
            frames.append(pd.read_csv(f))
    if not frames:
        raise SystemExit("no rfalse_train.csv under %s -- run step B first"
                         % ", ".join(data_dirs))
    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in head.features if c not in df.columns]
    if missing:
        raise SystemExit(
            "rfalse_train.csv is missing %d feature(s) the R_miss head needs "
            "(%s ...); rebuild step B" % (len(missing), ", ".join(missing[:5])))

    r = np.asarray(head.predict(df[head.features]), dtype=float).ravel()
    r = r[np.isfinite(r)]
    if r.size == 0:
        raise SystemExit("R_miss returned no finite values on %d training "
                         "candidates" % len(df))
    m_r = float(np.median(r))
    if not (m_r > 0):
        raise SystemExit("median R_miss is %r -- refusing to write a scale "
                         "that would zero out C_geom" % m_r)

    return dict(
        m_R=m_r,
        n=int(r.size),
        n_rows_read=int(len(df)),
        source="median R_miss over training candidates",
        head=os.path.normpath(hp),
        head_target=str((head.meta or {}).get("target")),
        head_train_split_only=bool((head.meta or {}).get("train_split_only")),
        btr_runs_dir=btr_runs_dir,
        btr_variant=btr_variant,
        data_dirs=[os.path.normpath(d) for d in data_dirs],
        quantiles={q: float(np.quantile(r, q))
                   for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        note=("C_geom is multiplied by m_R in RISK mode only.  uniform mode "
              "has R === 1, so its C_geom is already on the scale of R and is "
              "left unscaled; prob mode does not use C_geom at all "
              "(eta_eff = 0).  DECISIONS.md 2026-09-06 12:30."),
    )


def write_geom_scale(out_dir: str, **kw) -> dict:
    info = compute_geom_scale(**kw)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "geom_scale.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print("[geom_scale] m_R = %.6g over %d training candidates -> %s"
          % (info["m_R"], info["n"], p), flush=True)
    return info


def load_geom_scale(model_dir: str) -> Optional[float]:
    """``m_R`` from ``<model_dir>/geom_scale.json``; ``None`` when absent."""
    p = os.path.join(model_dir, "geom_scale.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return float(json.load(f)["m_R"])
    except Exception:
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.rigr.geom_scale",
        description="freeze m_R (median training R_miss) for one model dir")
    ap.add_argument("--data", action="append", required=True,
                    help="a runs/rigr_data/<ds> directory; repeat for a union")
    ap.add_argument("--out", required=True, help="the runs/rigr_models/... dir")
    ap.add_argument("--btr_runs_dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--btr_variant", default="all")
    ap.add_argument("--btr_target", default=None)
    a = ap.parse_args(argv)
    write_geom_scale(a.out, data_dirs=a.data, btr_runs_dir=a.btr_runs_dir,
                     btr_variant=a.btr_variant, target=a.btr_target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
