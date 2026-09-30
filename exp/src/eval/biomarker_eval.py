"""Shared biomarker-error helper, used by ``src.rigr.run_rigr`` and
``src.baselines.run_baseline`` so the two producers compute the macro-MAE
biomarker error **identically** (schema mismatch #2/#3 of the aggregation
audit): same ``src.bio.biomarkers`` pipeline call, same primary biomarkers,
same MAD-standardisation scale, same output column names.

Single source of sigma (schema mismatch #3)
--------------------------------------------
The per-dataset, per-biomarker robust scale ``sigma = 1.4826 * MAD`` used to
non-dimensionalise ``|B(pred) - B(gt)|`` is read **once**, from
``results/gateA_biomarker_scales.csv`` -- the file ``src.bio.gate_a`` writes
from the 261 ground-truth masks at **native resolution**.  Nothing in this
module (or in its callers) re-estimates sigma in-memory from the current run:
a run over a handful of smoke images would otherwise get a wildly unstable
MAD, and different callers would silently disagree on what "one sigma" means.

HRF and FIVES predictions must be mapped back to native resolution (the
resolution Gate A itself ran on) **before** they reach ``biomarker_row`` /
``macro_mae`` here, so that this native-resolution scale applies unchanged --
see ``src.robust.run_resolution``'s docstring for the up/down-sampling
convention that keeps this true, and ``exp/DECISIONS.md`` for why tortuosity
uses only the skan pipeline (the PVBM/skan tortuosity agreement gate failed;
see Gate A decision log).

CLI
---
None -- this module is a library only.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PRIMARY",
    "GATE_A_SCALES_CSV",
    "GATE_A_SCALES_TRAIN_CSV",
    "GATE_A_SCALES_ALL_CSV",
    "pipes_for",
    "biomarker_row",
    "auto_disc",
    "load_gt_scales",
    "macro_mae",
    "bio_columns",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")

#: the four biomarkers the paper standardises on (proposal section 3.1.2);
#: kept as a local alias of ``src.bio.biomarkers.PRIMARY_BIOMARKERS`` so this
#: module has no import-order dependency on the caller.
PRIMARY: Tuple[str, ...] = ("FD", "tortuosity", "density", "total_length")

#: The single source of truth for sigma: per-dataset, per-primary-biomarker MAD
#: scale computed on reference masks at native resolution.
#:
#: DECISIONS.md 2026-09-03 10:20 restricts it to the **training split**.  The
#: previous file was estimated over every reference mask, test images included,
#: so the scale that non-dimensionalises a reported error had itself been fitted
#: on the evaluation set -- a small leak, but one that touches every
#: ``macro_mae_*`` number in the paper.  ``_train`` is now the default and the
#: full-set file is kept only so an old run can be reproduced deliberately.
GATE_A_SCALES_TRAIN_CSV = os.path.join(RESULTS_DIR,
                                       "gateA_biomarker_scales_train.csv")
GATE_A_SCALES_ALL_CSV = os.path.join(RESULTS_DIR, "gateA_biomarker_scales.csv")

#: what every consumer gets unless it deliberately asks for something else
GATE_A_SCALES_CSV = GATE_A_SCALES_TRAIN_CSV

_scale_cache: Dict[str, Dict[str, Dict[str, float]]] = {}


def pipes_for(bio_arg: str) -> Tuple[str, ...]:
    """``"none"/"skan"/"pvbm"/"both"`` -> the pipeline-suffix tuple to use."""
    if bio_arg == "both":
        return ("pvbm", "skan")
    if bio_arg == "none":
        return ()
    return (str(bio_arg),)


def biomarker_row(mask, fov, image=None, disc=None, pipelines: str = "both",
                  fd_rotations: int = 8) -> Dict[str, float]:
    """One call to ``src.bio.biomarkers.compute_all``, the exact call both
    ``run_rigr.py`` and ``run_baseline.py`` must make so their biomarker
    numbers are computed identically."""
    from src.bio.biomarkers import compute_all

    return compute_all(mask, fov, image=image, disc=disc,
                       fd_rotations=int(fd_rotations),
                       run_pvbm=pipelines in ("both", "pvbm"),
                       run_skan=pipelines in ("both", "skan"))


def auto_disc(image, fov, vessel_mask=None):
    """Automatic optic-disc estimate (never an annotation); ``None`` on
    failure, with a one-time-per-call warning (never silently pretends a disc
    was found)."""
    import warnings

    try:
        from src.bio.disc import locate_optic_disc

        return locate_optic_disc(image, fov, vessel_mask=vessel_mask)
    except Exception as exc:  # noqa: BLE001
        warnings.warn("optic-disc detection failed (%s); the eccentricity zone "
                      "falls back to the FOV centroid" % type(exc).__name__,
                      RuntimeWarning)
        return None


def load_gt_scales(dataset: str, path: str = GATE_A_SCALES_CSV,
                   required: bool = True) -> Dict[str, float]:
    """``{biomarker_column: sigma}`` for one dataset, from the training-split
    Gate A scale table -- e.g. ``{"FD_pvbm": 0.017, "tortuosity_skan": ...}``.

    Every consumer of sigma goes through here (``run_rigr``, ``run_baseline``,
    ``robust/*``, ``exp2_candidates``, ``build_data``), so this is the one place
    the protocol is enforced.

    A **missing file raises** (``required=True``, the default).  It used to warn
    and return ``{}``, which turned every ``macro_mae_*`` column into NaN --
    a whole phase could run to completion and only look wrong at aggregation
    time.  Pass ``required=False`` for callers that genuinely tolerate the
    absence.

    A file that exists but has no row for ``dataset`` is a loud warning rather
    than an error: some datasets (STARE) legitimately never appear.
    """
    from src.seg.data import canon

    if path not in _scale_cache:
        if not os.path.exists(path):
            msg = ("sigma table %r not found. DECISIONS.md 2026-09-03 10:20 "
                   "requires the TRAINING-SPLIT scales; produce it with "
                   "`python -m src.bio.gate_a --split train` (the C1 run owns "
                   "this file). Every macro_mae_* column depends on it."
                   % os.path.relpath(path, EXP_ROOT))
            if required:
                raise FileNotFoundError(msg)
            import warnings

            warnings.warn(msg + " Continuing with no scales: every biomarker "
                          "error will read as NaN.", RuntimeWarning)
            _scale_cache[path] = {}
        else:
            import pandas as pd

            table: Dict[str, Dict[str, float]] = {}
            df = pd.read_csv(path)
            missing_cols = {"dataset", "biomarker", "sigma"} - set(df.columns)
            if missing_cols:
                raise ValueError("%s is not a Gate A scale table: missing "
                                 "columns %s" % (path, sorted(missing_cols)))
            for r in df.itertuples(index=False):
                ds = canon(str(r.dataset))
                try:
                    sigma = float(r.sigma)
                except (TypeError, ValueError):
                    continue
                table.setdefault(ds, {})[str(r.biomarker)] = sigma
            _scale_cache[path] = table

    key = canon(dataset)
    out = _scale_cache[path].get(key, {})
    if not out:
        import warnings

        warnings.warn("no sigma rows for dataset %r in %s (it has: %s); every "
                      "macro_mae_* value for this dataset will be NaN"
                      % (key, os.path.relpath(path, EXP_ROOT),
                         ", ".join(sorted(_scale_cache[path])) or "nothing"),
                      RuntimeWarning)
    return dict(out)


def macro_mae(pred_row: Dict[str, float], gt_row: Dict[str, float],
              sigma: Dict[str, float], pipelines: Sequence[str] = ("pvbm", "skan")
              ) -> float:
    """Equal-weight mean of ``|B(pred) - B(gt)| / sigma_b`` over the four
    primary biomarkers (proposal section 3.1.2), using the single ``sigma``
    lookup (see :func:`load_gt_scales`)."""
    vals = []
    for b in PRIMARY:
        for p in pipelines:
            col = "%s_%s" % (b, p)
            s = float(sigma.get(col, np.nan))
            a, g = float(pred_row.get(col, np.nan)), float(gt_row.get(col, np.nan))
            if np.isfinite(s) and s > 0 and np.isfinite(a) and np.isfinite(g):
                vals.append(abs(a - g) / s)
    return float(np.mean(vals)) if vals else float("nan")


def bio_columns(gt_row: Dict[str, float], before_row: Dict[str, float],
                after_row: Dict[str, float], sigma: Dict[str, float],
                pipelines: Sequence[str] = ("pvbm", "skan")) -> Dict[str, Any]:
    """``macro_mae_before/after/delta`` + the raw ``bio_gt_*`` /
    ``bio_before_*`` / ``bio_after_*`` columns, named identically to what
    ``run_rigr.py`` has always written -- the contract this helper exists to
    enforce on ``run_baseline.py`` too."""
    out: Dict[str, Any] = {}
    out["macro_mae_before"] = macro_mae(before_row, gt_row, sigma, pipelines)
    out["macro_mae_after"] = macro_mae(after_row, gt_row, sigma, pipelines)
    out["macro_mae_delta"] = out["macro_mae_after"] - out["macro_mae_before"]
    for b in PRIMARY:
        for pl in pipelines:
            c = "%s_%s" % (b, pl)
            out["bio_gt_" + c] = gt_row.get(c, np.nan)
            out["bio_before_" + c] = before_row.get(c, np.nan)
            out["bio_after_" + c] = after_row.get(c, np.nan)
    return out
