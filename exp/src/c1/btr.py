"""BTR: multi-output counterfactual-harm risk heads + the Exp1 benchmark.

Plan reference: exp/EXPERIMENT_PLAN.md S3.4-S3.5, proposal sections 3.1.2,
3.1.5, 3.1.6 and 3.1.7.

Heads
-----
Two heads share one **Contract F** feature schema but observe different states
(proposal section 3.1.5):

``R_miss``   trained on **severing** events, whose features come from the
             *severed* state ``M^-`` -- "how much downstream measurement is lost
             if this break is left unrepaired".
``R_false``  trained on **bridging** events, whose features come from the
             *connected* state ``M^+`` -- "how much damage a wrong connection
             here would do".  (The interventional Bezier bridges train the
             *explanatory* head; section 3.1.5 requires the deployed
             ``R_false`` to be retrained on rasterised A* candidate paths in
             S4, which reuses :class:`BTRHead` unchanged.)

Each head is **multi-output**: one LightGBM regressor per
``(biomarker b in 4, pipeline p in 2)``, i.e. 8 regressors, each fitted on the
non-negative deployment target ``H_dep^(b,p)`` (proposal section 3.1.2 step 2).
Non-negativity is preserved at the output by clipping at 0.

Scalarisation (pre-registered, section 3.1.2 step 3)

    ``R(e) = sum_b w_b * median_p R^(b,p)(e)``   with equal ``w_b``.

Splitting (section 3.1.6)
-------------------------
* **inner**: ``GroupKFold`` with the **subject** as the group, so every
  perturbation type, severity, locus, control and pipeline of one subject stays
  inside one fold;
* **outer**: **leave one whole dataset out** -- fit on three domains, predict
  the fourth, which is what the "harm estimation transfers across datasets"
  claim needs.

Exp1 (section 3.1.7)
--------------------
(a) *Raw association*: Spearman rho of each conventional scalar
    (``dDice``, ``dclDice``, ``dBetti0``, ``dBetti1``, ``BCS drop``) with the
    ground-truth harm ``y`` -- no fitting.
(b) *Equal-capacity calibrated prediction*: the same GBDT (same depth, leaves,
    learning rate, CV protocol) is fitted on each conventional predictor, on
    the combined conventional vector, and on the BTR feature set; held-out MAE
    and R^2 are compared.
(c) Statistical unit = **image**: an image-level bootstrap (1000 resamples)
    gives the CI of ``MAE_BTR - MAE_best_conventional``.

Usage
-----
    cd exp
    python -m src.c1.btr --events results/c1_events_harm.parquet
    python -m src.c1.btr --limit-boot 200            # quicker smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")
RUNS_DIR = os.path.join(EXP_ROOT, "runs", "btr")

PRIMARY = ("FD", "tortuosity", "density", "total_length")
PIPELINES = ("pvbm", "skan")

#: equal pre-registered weights (proposal section 3.1.2 step 3)
BIOMARKER_WEIGHTS = {b: 0.25 for b in PRIMARY}

#: Which pipelines are primary for each biomarker -- imported from stats_c1 so
#: the Gate A outcome is stated in exactly one place.  Gate A passed FD, density
#: and total_length (Spearman >= 0.92, aggregated by median over pipelines) and
#: FAILED tortuosity (rho 0.49-0.75), which is therefore carried by its single
#: pre-registered pipeline (skan, length-weighted) alone.
from .stats_c1 import PRIMARY_PIPELINES, NON_PRIMARY_COLUMNS  # noqa: E402

#: one shared GBDT capacity for **every** model in Exp1 -- this is what makes
#: the BTR-vs-conventional comparison "equal capacity" (section 3.1.7 b).
GBDT_PARAMS = dict(
    objective="regression_l1",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=40,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    n_jobs=1,
    verbose=-1,
)

CONVENTIONAL = {
    "dDice": ("m_dice", -1.0, 1.0),          # column, sign, offset -> 1 - dice
    "dclDice": ("m_cldice", -1.0, 1.0),
    "dBetti0": ("m_beta0_err", 1.0, 0.0),
    "dBetti1": ("m_beta1_err", 1.0, 0.0),
    "BCSdrop": (None, 0.0, 0.0),             # special: m_bcs_gt - m_bcs
}

#: Inference-time **observable** baselines: single Contract-F quantities a
#: deployed repairer actually has (no reference annotation).  Every
#: ``CONVENTIONAL`` metric above is computed against the ground truth and is
#: therefore *unavailable* at deployment; comparing BTR only against those would
#: flatter the conventional side with information it cannot have in practice.
#: These four give the honest inference-time comparison (proposal 3.1.7 b).
OBSERVABLE = {
    "obs_d": "phi_d",                    # endpoint gap length d
    "obs_d_over_r": "phi_d_over_r",      # gap length in radii, d / r-bar
    "obs_radius": "phi_r_mean",          # local vessel radius r-bar
    "obs_density": "phi_density",        # local vessel density around the locus
}

#: The two supervision targets (see DECISIONS.md 2026-09-03 gate B).
#:
#: ``hdep`` is the pre-registered net harm ``H_dep = max(0, (|dB| - |dB_ctrl|)/sigma)``
#: -- the *excess* topological harm over a pixel-matched control.  Gate B found
#: it sits at pipeline-noise level (held-out R^2 <= 0 for every predictor), and
#: for ``density`` it is identically zero because the matched control removes
#: exactly the same pixel count by construction.
#:
#: ``habs`` is the raw absolute standardized harm ``H_abs = |dB_topo| / sigma``
#: -- the full counterfactual consequence of the edit, pixels *and* connectivity.
#: This is what a deployment utility U(e) actually needs to price, so the
#: **deployed** heads are trained on it, while ``hdep`` remains the scientific
#: dose-response quantity.  Both are non-negative.
TARGETS = ("hdep", "habs")
TARGET_PREFIX = {"hdep": "Hdep", "habs": "Habs"}

#: Suffix marking heads fitted on training-split images only.  A head that saw
#: perturbations of a *test* image must never price a repair on that same image
#: in stage S4 -- RiGR is applied to predictions of exactly those images, so
#: fitting on them leaks the answer.  ``"habs_trainonly"`` is therefore a
#: first-class target name that resolves to its own subdirectory.
TRAINONLY_SUFFIX = "_trainonly"

#: Path resolution default.  Deliberately ``hdep`` so that existing callers and
#: the already-written ``runs/btr/btr_*.joblib`` files keep resolving exactly as
#: before; a caller that wants the deployed heads asks for ``target="habs"``.
DEFAULT_TARGET = "hdep"


# --------------------------------------------------------------------------- #
# feature / target plumbing
# --------------------------------------------------------------------------- #
def feature_columns(df) -> List[str]:
    """Contract-F features only: everything named ``phi_*``.

    Nothing derived from the reference annotation (``gt_*``) may enter here --
    section 3.1.1 restricts the deployable model to ``(I, M-hat, P)``.
    """
    return sorted(c for c in df.columns if c.startswith("phi_"))


def parse_target(target: Optional[str]) -> Tuple[str, bool]:
    """``"habs_trainonly"`` -> ``("habs", True)``; ``None`` -> ``(DEFAULT_TARGET, False)``.

    The *base* decides which ``H`` columns are the supervision targets; the
    ``_trainonly`` flag only decides which **rows** are allowed into the fit (and
    hence which directory the head is stored in).
    """
    t = target or DEFAULT_TARGET
    train_only = t.endswith(TRAINONLY_SUFFIX)
    base = t[: -len(TRAINONLY_SUFFIX)] if train_only else t
    if base not in TARGETS:
        raise ValueError(f"unknown target {target!r}; expected one of "
                         f"{TARGETS} optionally suffixed {TRAINONLY_SUFFIX!r}")
    return base, train_only


def split_lookup(datasets: Optional[Sequence[str]] = None) -> Dict[Tuple[str, str], str]:
    """``(dataset, image_id) -> "train" | "test"`` from the dataset definitions.

    Read from :mod:`src.data.datasets` rather than re-derived here, so the C1
    event table and the segmentation grid can never disagree about which images
    are held out.
    """
    from src.data.datasets import load_dataset

    out: Dict[Tuple[str, str], str] = {}
    for ds in (datasets or ("DRIVE", "CHASE_DB1", "HRF", "FIVES", "STARE")):
        try:
            recs = load_dataset(ds)
        except Exception:  # noqa: BLE001 - a dataset that is not present locally
            continue
        for r in recs:
            out[(ds, str(r["image_id"]))] = str(r["split"])
    return out


def add_split(df, lut: Optional[Dict[Tuple[str, str], str]] = None):
    """Add a ``split`` column to an event table (``"train"``/``"test"``/NaN)."""
    lut = split_lookup(sorted(df["dataset"].astype(str).unique())) if lut is None else lut
    df["split"] = [lut.get((d, i), None) for d, i in
                   zip(df["dataset"].astype(str), df["image_id"].astype(str))]
    return df


def train_rows(df):
    """Boolean mask of events that may be used for **fitting** a deployed head.

    Training-split images only.  DRIVE's second-observer rows sit on test images
    (the 2nd_manual annotation only ships for the test half), so they are
    excluded here automatically rather than by a special case.
    """
    import pandas as pd

    if "split" not in df.columns:
        df = add_split(df.copy())
    return (df["split"].astype("object") == "train").to_numpy(dtype=bool)


def add_habs(df):
    """Add ``Habs_{biomarker}_{pipeline}`` = ``|dB_topo| / sigma_B`` in place.

    Derived from the columns :mod:`src.c1.stats_c1` already wrote into the harm
    table (``absdB_*`` and the per-dataset ``sigma_*``), so no re-run of the
    statistics stage is needed.  Existing ``Habs_*`` columns are left alone.
    """
    import pandas as pd

    for b in PRIMARY:
        for p in PIPELINES:
            col = f"{b}_{p}"
            out = f"Habs_{col}"
            if out in df.columns:
                continue
            # prefer the TRAINING-split scale: H_abs is the deployed target and
            # its denominator must not be informed by test images.  stats_c1
            # normally writes Habs_* directly (so this loop is a no-op); this is
            # the fallback path for a table that only carries the raw columns.
            s_col = (f"sigmatr_{col}" if f"sigmatr_{col}" in df.columns
                     else f"sigma_{col}")
            a_col = f"absdB_{col}"
            if a_col not in df.columns or s_col not in df.columns:
                continue
            a = pd.to_numeric(df[a_col], errors="coerce")
            sig = pd.to_numeric(df[s_col], errors="coerce").replace(0.0, np.nan)
            df[out] = (a / sig).clip(lower=0.0)
    return df


def ensure_target(df, target: str = DEFAULT_TARGET):
    """Make sure the columns for ``target`` exist, deriving them if possible."""
    base, _train_only = parse_target(target)
    if base == "habs":
        add_habs(df)
    prefix = TARGET_PREFIX[base]
    if not any(c.startswith(prefix + "_") for c in df.columns):
        raise ValueError(
            f"event table has no {prefix}_* columns; run `python -m src.c1.stats_c1` "
            "first (H_abs additionally needs the absdB_*/sigma_* columns)")
    return df


def target_columns(df, target: str = DEFAULT_TARGET) -> List[str]:
    """The multi-output targets: one per *primary* ``(biomarker, pipeline)``.

    ``tortuosity_pvbm`` is excluded -- it failed the Gate A two-pipeline
    agreement check, so it is neither a supervision target nor part of any
    scalarisation (its dB columns remain in the event table for reference).
    """
    prefix = TARGET_PREFIX[parse_target(target)[0]]
    return [f"{prefix}_{b}_{p}" for b in PRIMARY for p in PRIMARY_PIPELINES[b]
            if f"{prefix}_{b}_{p}" in df.columns]


def observable_frame(df):
    """The no-GT inference-time baselines as a DataFrame (larger = worse)."""
    import pandas as pd

    out = pd.DataFrame(index=df.index)
    for name, col in OBSERVABLE.items():
        out[name] = (pd.to_numeric(df[col], errors="coerce")
                     if col in df.columns else np.nan)
    return out


def conventional_frame(df):
    """The five conventional scalars as a DataFrame (larger = worse)."""
    import pandas as pd

    out = pd.DataFrame(index=df.index)
    for name, (col, sign, off) in CONVENTIONAL.items():
        if name == "BCSdrop":
            if "m_bcs_gt" in df.columns and "m_bcs" in df.columns:
                out[name] = (pd.to_numeric(df["m_bcs_gt"], errors="coerce")
                             - pd.to_numeric(df["m_bcs"], errors="coerce"))
            else:
                out[name] = np.nan
            continue
        if col in df.columns:
            out[name] = off + sign * pd.to_numeric(df[col], errors="coerce")
        else:
            out[name] = np.nan
    return out


def macro_target(df, target: str = DEFAULT_TARGET):
    """``y`` = equal-weight mean over biomarkers of the per-pipeline median harm."""
    import pandas as pd

    prefix = TARGET_PREFIX[parse_target(target)[0]]
    per_b = []
    for b in PRIMARY:
        cols = [f"{prefix}_{b}_{p}" for p in PRIMARY_PIPELINES[b]
                if f"{prefix}_{b}_{p}" in df.columns]
        if cols:
            per_b.append(df[cols].median(axis=1) * BIOMARKER_WEIGHTS[b] * len(PRIMARY))
    if not per_b:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(per_b, axis=1).mean(axis=1)


# --------------------------------------------------------------------------- #
# the head
# --------------------------------------------------------------------------- #
class BTRHead:
    """Multi-output non-negative harm regressor (one LightGBM per target)."""

    def __init__(self, name: str, params: Optional[Dict] = None):
        self.name = name
        self.params = dict(GBDT_PARAMS if params is None else params)
        self.features: List[str] = []
        self.targets: List[str] = []
        self.models: Dict[str, object] = {}
        self.meta: Dict[str, object] = {}

    def fit(self, X, y_multi, groups=None, n_splits: int = 4):
        """Fit one regressor per target.

        ``groups`` (the **subject** ids) drive an inner ``GroupKFold`` used for
        early stopping; with too few groups the fit falls back to the full
        training set and the fixed ``n_estimators``.
        """
        import lightgbm as lgb
        from sklearn.model_selection import GroupKFold

        self.features = list(X.columns)
        self.targets = list(y_multi.columns)
        Xv = X.to_numpy(dtype=float)
        g_all = np.asarray(groups) if groups is not None else None

        for t in self.targets:
            yv = y_multi[t].to_numpy(dtype=float)
            ok = np.isfinite(yv)
            if ok.sum() < 50:
                self.models[t] = None
                continue
            # the number of *usable* groups must be counted on the rows that
            # survive the finite-target mask, not on the whole frame
            n_groups = len(np.unique(g_all[ok])) if g_all is not None else 0
            use_cv = n_groups >= 2
            model = lgb.LGBMRegressor(**self.params)
            if use_cv:
                gkf = GroupKFold(n_splits=min(n_splits, n_groups))
                tr, va = next(iter(gkf.split(Xv[ok], yv[ok], g_all[ok])))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(Xv[ok][tr], yv[ok][tr],
                              eval_set=[(Xv[ok][va], yv[ok][va])],
                              callbacks=[lgb.early_stopping(40, verbose=False)])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(Xv[ok], yv[ok])
            self.models[t] = model
        return self

    def predict_multi(self, X):
        """Per-target non-negative predictions, as a DataFrame."""
        import pandas as pd

        Xv = X.reindex(columns=self.features).to_numpy(dtype=float)
        out = {}
        for t in self.targets:
            m = self.models.get(t)
            out[t] = np.clip(m.predict(Xv), 0.0, None) if m is not None else np.full(len(X), np.nan)
        return pd.DataFrame(out, index=X.index)

    def target_prefix(self) -> str:
        """``"Hdep"`` or ``"Habs"`` -- whichever this head was fitted on.

        Read from ``meta["target"]`` when present, else inferred from the
        stored target column names, else the historical default.  Hard-coding
        ``Hdep_`` here (as this method used to) made :meth:`predict` return
        **all-NaN** for every ``H_abs`` head -- which
        ``src.rigr.utility.risk_values`` then turns into ``R = 1`` for every
        candidate, i.e. a silently uniform-cost run still reporting a BTR
        backend.  The prefix must follow the head, never the caller.
        """
        t = (self.meta or {}).get("target")
        if t:
            try:
                return TARGET_PREFIX[parse_target(str(t))[0]]
            except Exception:
                pass
        for pref in TARGET_PREFIX.values():
            if any(str(c).startswith(pref + "_") for c in self.targets):
                return pref
        return TARGET_PREFIX[DEFAULT_TARGET]

    def predict(self, X):
        """The pre-registered scalarisation ``R = sum_b w_b median_p R^(b,p)``."""
        import pandas as pd

        pm = self.predict_multi(X)
        prefix = self.target_prefix()
        parts = []
        for b in PRIMARY:
            cols = [f"{prefix}_{b}_{p}" for p in PRIMARY_PIPELINES[b]
                    if f"{prefix}_{b}_{p}" in pm.columns]
            if cols:
                parts.append(pm[cols].median(axis=1) * BIOMARKER_WEIGHTS[b])
        if not parts:
            warnings.warn(
                "BTRHead %r: none of its %d target columns match the scalarisation "
                "prefix %r (targets: %s); predict() would return NaN, which a "
                "caller may silently read as R = 1"
                % (self.name, len(self.targets), prefix, self.targets[:4]),
                RuntimeWarning)
            return pd.Series(np.nan, index=X.index)
        return pd.concat(parts, axis=1).sum(axis=1).clip(lower=0.0)

    # ------------------------------------------------------------- persistence
    def save(self, path: str):
        import joblib

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(dict(name=self.name, params=self.params, features=self.features,
                         targets=self.targets, models=self.models,
                         meta=self.meta), path)
        return path

    @classmethod
    def load(cls, path: str) -> "BTRHead":
        import joblib

        d = joblib.load(path)
        h = cls(d["name"], d["params"])
        h.features, h.targets, h.models = d["features"], d["targets"], d["models"]
        h.meta = d.get("meta", {})
        return h


# --------------------------------------------------------------------------- #
# the *deployed* R_false head (stage S4 item 6)
# --------------------------------------------------------------------------- #
#: rows of the deployed-R_false training table that are not features/targets
DEPLOYED_FALSE_META = ("dataset", "image", "subject_id", "cand_id",
                       "is_true_repair", "y", "ok")


def fit_deployed_false(
    df,
    out_path: Optional[str] = None,
    name: str = "false_deployed",
    params: Optional[Dict] = None,
    only_wrong: bool = True,
    min_events: int = 60,
    group_col: str = "subject_id",
    target: str = "habs",
):
    """Fit the **deployed** ``R_false`` head on rasterised A* candidate paths.

    Section 3.1.5 trains the *explanatory* ``R_false`` on the interventional
    Bezier bridges of C1 and then requires the deployed head to be retrained on
    the connections RiGR actually proposes: the A* paths of
    :mod:`src.rigr.astar`, rasterised onto the severed mask, with the
    counterfactual harm measured by the *same* biomarker pipeline and the same
    single-source sigma (``results/gateA_biomarker_scales.csv``).

    ``df`` is the table :mod:`src.rigr.build_data` writes: Contract-F
    ``phi_*`` columns evaluated on the **connected state** ``M+ = M- U T_e``,
    both target families (``Hdep_*`` and ``Habs_*``), plus the metadata columns
    of :data:`DEPLOYED_FALSE_META`.

    ``target`` defaults to ``"habs"``: DECISIONS.md (2026-09-03 08:30) fixes the
    **deployed** heads as ``H_abs = |B(M+) - B(M-)| / sigma``, the full
    counterfactual consequence of the edit, because that is what ``U(e)`` has to
    price -- ``H_dep`` (the control-subtracted residual) sits at pipeline-noise
    level and is kept only for the scientific dose-response analysis.  It also
    has to match: this head is paired in one :class:`~src.rigr.btr_backend.BTRBackend`
    with ``runs/btr/habs_trainonly/btr_miss_all.joblib``, and that class refuses
    to mix two differently-targeted heads.  The chosen target is recorded in
    ``meta["target"]``, which is what both the pairing check and
    :meth:`BTRHead.target_prefix` read.

    ``only_wrong`` (default) restricts the fit to candidates that are **not**
    true repairs, i.e. exactly the population the term
    ``lambda (1 - p_e) R_false(e)`` prices: "what does a wrong connection here
    cost".  Set it to ``False`` to fit on every candidate.

    Returns the fitted :class:`BTRHead` (saved to ``out_path`` when given), or
    ``None`` when fewer than ``min_events`` usable rows survive -- the caller
    must then fall back to the C1 head and say so.
    """
    import numpy as _np

    sub = df
    if only_wrong and "is_true_repair" in sub.columns:
        sub = sub[_np.asarray(sub["is_true_repair"]).astype(float) <= 0.5]
    if "ok" in sub.columns:
        sub = sub[_np.asarray(sub["ok"]).astype(float) > 0.5]
    sub = sub.reset_index(drop=True)

    base, _train_only = parse_target(target)
    prefix = TARGET_PREFIX[base]
    feats = feature_columns(sub)
    tgts = target_columns(sub, target)
    if not feats or not tgts:
        warnings.warn("fit_deployed_false: no phi_* features (%d) or %s_* "
                      "targets (%d) in the table -- rebuild it with "
                      "`python -m src.rigr.build_data ... --force` (it now "
                      "stores both target families)"
                      % (len(feats), prefix, len(tgts)), RuntimeWarning)
        return None
    usable = int(_np.isfinite(sub[tgts].to_numpy(dtype=float)).any(axis=1).sum())
    if usable < int(min_events):
        warnings.warn("fit_deployed_false: only %d usable rows (< %d); the "
                      "deployed R_false head is NOT fitted -- the caller must "
                      "fall back to the C1 head and record that fact"
                      % (usable, min_events), RuntimeWarning)
        return None

    groups = (sub[group_col].astype(str).to_numpy()
              if group_col in sub.columns else None)
    head = BTRHead(name, params)
    head.fit(sub[feats], sub[tgts], groups=groups)
    head.meta = dict(target=base, n_rows=int(len(sub)), n_usable=int(usable),
                     only_wrong=bool(only_wrong),
                     datasets=sorted(set(sub["dataset"].astype(str)))
                     if "dataset" in sub.columns else [],
                     n_images=int(sub["image"].nunique())
                     if "image" in sub.columns else -1)
    if out_path:
        head.save(out_path)
    return head


# --------------------------------------------------------------------------- #
# the RiGR-facing API
# --------------------------------------------------------------------------- #
_CACHE: Dict[str, BTRHead] = {}


def head_path(kind: str, runs_dir: str = RUNS_DIR, variant: str = "all",
              target: Optional[str] = None) -> str:
    """Path of a saved head.

    ``variant`` is ``"all"`` for the head fitted on every dataset, or
    ``"wo_<dataset>"`` for the leave-one-dataset-out head :func:`main` writes
    (the LODO block of stage S4).

    ``target`` selects the supervision target the head was trained on.  ``None``
    and ``"hdep"`` both resolve to the historical flat layout
    ``<runs_dir>/btr_<kind>_<variant>.joblib`` so every existing caller and every
    already-written file keeps working unchanged; any other target lives in its
    own subdirectory, e.g. ``<runs_dir>/habs/btr_miss_all.joblib``.
    """
    if target is not None:
        parse_target(target)          # validates, including the _trainonly suffix
    root = runs_dir if target in (None, "hdep") else os.path.join(runs_dir, target)
    return os.path.join(root, f"btr_{kind}_{variant}.joblib")


def load_head(kind: str, runs_dir: str = RUNS_DIR, variant: str = "all",
              target: Optional[str] = None) -> BTRHead:
    """Load (and process-cache) one saved head.

    The cache key is the **resolved path**, not just ``kind`` -- otherwise a
    second call with a different ``runs_dir`` / ``variant`` / ``target`` would
    silently be served the first call's head (which is exactly the failure mode
    the LODO block must not have).
    """
    path = os.path.abspath(head_path(kind, runs_dir, variant, target))
    if path not in _CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"no trained head at {path}; run `python -m src.c1.btr` first")
        _CACHE[path] = BTRHead.load(path)
    return _CACHE[path]


def _load_head(kind: str, runs_dir: str = RUNS_DIR,
               target: Optional[str] = None) -> BTRHead:
    """Backwards-compatible alias of :func:`load_head` (variant ``all``)."""
    return load_head(kind, runs_dir, "all", target)


def predict_R_miss(phi_df, runs_dir: str = RUNS_DIR, target: Optional[str] = None):
    """``R_miss(e) >= 0`` for candidates observed in the **severed** state.

    ``phi_df`` is a DataFrame of Contract-F features (``phi_*`` columns) as
    produced by :func:`src.c1.perturb.phi`; missing columns are filled with NaN,
    which LightGBM handles natively.
    """
    return _load_head("miss", runs_dir, target).predict(phi_df)


def predict_R_false(phi_df, runs_dir: str = RUNS_DIR, target: Optional[str] = None):
    """``R_false(e) >= 0`` for candidates observed in the **connected** state."""
    return _load_head("false", runs_dir, target).predict(phi_df)


def predict_multi(kind: str, phi_df, runs_dir: str = RUNS_DIR,
                  target: Optional[str] = None):
    """Per-``(biomarker, pipeline)`` risks, before scalarisation."""
    return _load_head(kind, runs_dir, target).predict_multi(phi_df)


# --------------------------------------------------------------------------- #
# Exp1
# --------------------------------------------------------------------------- #
def _mae(y, p) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.mean(np.abs(y[m] - p[m]))) if m.any() else float("nan")


def _r2(y, p) -> float:
    m = np.isfinite(y) & np.isfinite(p)
    if m.sum() < 3:
        return float("nan")
    ss_res = float(np.sum((y[m] - p[m]) ** 2))
    ss_tot = float(np.sum((y[m] - y[m].mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _spearman(a, b) -> Tuple[float, float]:
    from scipy.stats import spearmanr

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan"), float("nan")
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p)


def _fit_predict_lodo(X, y, groups, datasets, params=None, fit_mask=None):
    """Leave-one-dataset-out out-of-fold predictions with an inner subject CV.

    ``fit_mask`` further restricts which rows may be *fitted* on (predictions are
    still produced for every held-out row).  Passing the training-split mask
    makes this table refer to the same heads stage S4 deploys, instead of heads
    that also saw test-split perturbations of the training datasets.
    """
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    params = dict(GBDT_PARAMS if params is None else params)
    pred = np.full(len(y), np.nan)
    Xv = X.to_numpy(dtype=float)
    yv = np.asarray(y, dtype=float)
    ds = np.asarray(datasets)
    gr = np.asarray(groups)
    fm = (np.ones(len(yv), dtype=bool) if fit_mask is None
          else np.asarray(fit_mask, dtype=bool))
    for held in np.unique(ds):
        te = ds == held
        tr = (~te) & fm
        if tr.sum() < 60 or te.sum() < 5:
            continue
        ok = tr & np.isfinite(yv)
        model = lgb.LGBMRegressor(**params)
        n_groups = len(np.unique(gr[ok]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if n_groups >= 4:
                gkf = GroupKFold(n_splits=min(4, n_groups))
                a, b = next(iter(gkf.split(Xv[ok], yv[ok], gr[ok])))
                model.fit(Xv[ok][a], yv[ok][a], eval_set=[(Xv[ok][b], yv[ok][b])],
                          callbacks=[lgb.early_stopping(40, verbose=False)])
            else:
                model.fit(Xv[ok], yv[ok])
        pred[te] = np.clip(model.predict(Xv[te]), 0.0, None)
    return pred


def bootstrap_ci(image_ids, y, p_a, p_b, n_boot: int = 1000, seed: int = 0):
    """Image-level bootstrap CI of ``MAE(p_a) - MAE(p_b)`` (section 3.1.7).

    The statistical unit is the **image**, not the perturbation: resampling
    events would report an absurdly tight CI on a sample of ~2e4 correlated
    events.  Whole images are resampled with replacement.
    """
    rng = np.random.default_rng(int(seed))
    ids = np.asarray(image_ids)
    uniq = np.unique(ids)
    idx_by_img = {u: np.nonzero(ids == u)[0] for u in uniq}
    diffs = np.empty(int(n_boot))
    for k in range(int(n_boot)):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by_img[u] for u in pick])
        diffs[k] = _mae(y[sel], p_a[sel]) - _mae(y[sel], p_b[sel])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return dict(mean=float(np.nanmean(diffs)), median=float(np.nanmedian(diffs)),
                ci_lo=float(lo), ci_hi=float(hi), n_boot=int(n_boot),
                excludes_zero=bool(lo > 0 or hi < 0))


def bootstrap_pair(image_ids, y, p_a, p_b, n_boot: int = 1000, seed: int = 0):
    """Image-level bootstrap of BOTH ``MAE(p_a)-MAE(p_b)`` and ``rho_a-rho_b``.

    Same resampling unit as :func:`bootstrap_ci` (whole images), reported for the
    two statistics the in-domain table quotes: a difference in error and a
    difference in rank agreement.  ``d_mae < 0`` and ``d_rho > 0`` both favour
    ``p_a`` (which is always BTR here).
    """
    rng = np.random.default_rng(int(seed))
    ids = np.asarray(image_ids)
    uniq = np.unique(ids)
    idx_by_img = {u: np.nonzero(ids == u)[0] for u in uniq}
    d_mae = np.full(int(n_boot), np.nan)
    d_rho = np.full(int(n_boot), np.nan)
    for k in range(int(n_boot)):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        sel = np.concatenate([idx_by_img[u] for u in pick])
        d_mae[k] = _mae(y[sel], p_a[sel]) - _mae(y[sel], p_b[sel])
        d_rho[k] = _spearman(p_a[sel], y[sel])[0] - _spearman(p_b[sel], y[sel])[0]

    def _summ(v, favour_negative):
        lo, hi = np.nanpercentile(v, [2.5, 97.5])
        return dict(mean=float(np.nanmean(v)), median=float(np.nanmedian(v)),
                    ci_lo=float(lo), ci_hi=float(hi),
                    excludes_zero=bool(lo > 0 or hi < 0),
                    favours_btr=bool((hi < 0) if favour_negative else (lo > 0)))

    return dict(d_mae=_summ(d_mae, True), d_rho=_summ(d_rho, False),
                n_boot=int(n_boot))


def _fit_predict_holdout(Xtr, ytr, gtr, Xte, params=None):
    """Fit once on the training rows, predict the held-out rows.

    Early stopping uses an inner subject-grouped split of the *training* data, so
    no held-out row influences model selection.
    """
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold

    params = dict(GBDT_PARAMS if params is None else params)
    Xv = Xtr.to_numpy(dtype=float)
    yv = np.asarray(ytr, dtype=float)
    gr = np.asarray(gtr)
    ok = np.isfinite(yv)
    if ok.sum() < 50:
        return np.full(len(Xte), np.nan)
    model = lgb.LGBMRegressor(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n_groups = len(np.unique(gr[ok]))
        if n_groups >= 4:
            gkf = GroupKFold(n_splits=min(4, n_groups))
            a, b = next(iter(gkf.split(Xv[ok], yv[ok], gr[ok])))
            model.fit(Xv[ok][a], yv[ok][a], eval_set=[(Xv[ok][b], yv[ok][b])],
                      callbacks=[lgb.early_stopping(40, verbose=False)])
        else:
            model.fit(Xv[ok], yv[ok])
    return np.clip(model.predict(Xte.to_numpy(dtype=float)), 0.0, None)


def run_exp1_indomain(df, kind: str, n_boot: int = 1000, seed: int = 0,
                      target: str = "habs"):
    """The **deployable** Exp1: fit on training-split images, score on test ones.

    This is the protocol the deployed heads actually run under -- the model has
    seen the dataset (and other images of it) but never the test image it prices.
    Contrast :func:`run_exp1`, which holds out a whole dataset and therefore
    measures cross-dataset *transfer*.

    A dataset with no training-split events in the table (FIVES: stage C1
    deliberately perturbed a subset of its **test** split) cannot be in-domain;
    its rows are still scored but flagged ``in_domain=False``.
    """
    import pandas as pd

    d = df if "split" in df.columns else add_split(df.copy())
    tr = d[d["split"] == "train"]
    te = d[d["split"] == "test"]
    if len(tr) < 50 or len(te) < 20:
        return None, {}, {}, None

    trained_ds = set(tr["dataset"].astype(str).unique())
    y_tr = macro_target(tr, target).to_numpy(dtype=float)
    y_te = macro_target(te, target).to_numpy(dtype=float)
    g_tr = tr["subject_id"].astype(str).to_numpy()
    feats = feature_columns(d)

    blocks = {
        "GT-informed conventional": (conventional_frame(tr), conventional_frame(te),
                                     "conventional_combined"),
        "no-GT observable": (observable_frame(tr), observable_frame(te),
                             "observable_combined"),
    }
    preds: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, object]] = []
    ds_te = te["dataset"].astype(str).to_numpy()
    images = (te["dataset"].astype(str) + ":" + te["image_id"].astype(str)
              + ":" + te["observer"].astype(str)).to_numpy()

    def _rows_for(name, block, p):
        ok = np.isfinite(p) & np.isfinite(y_te)
        rho, pv = _spearman(p[ok], y_te[ok])
        rows.append(dict(head=kind, target=target, block=block, predictor=name,
                         scope="ALL(test)", in_domain="", n=int(ok.sum()),
                         spearman=rho, spearman_p=pv,
                         mae=_mae(y_te[ok], p[ok]), r2=_r2(y_te[ok], p[ok])))
        for dsn in sorted(set(ds_te)):
            m = ok & (ds_te == dsn)
            if m.sum() < 20:
                continue
            r2_, rho2 = _r2(y_te[m], p[m]), _spearman(p[m], y_te[m])[0]
            rows.append(dict(head=kind, target=target, block=block, predictor=name,
                             scope=dsn, in_domain=bool(dsn in trained_ds),
                             n=int(m.sum()), spearman=rho2, spearman_p=float("nan"),
                             mae=_mae(y_te[m], p[m]), r2=r2_))

    for block, (Ftr, Fte, comb) in blocks.items():
        for c in Ftr.columns:
            p = _fit_predict_holdout(Ftr[[c]], y_tr, g_tr, Fte[[c]])
            preds[c] = p
            _rows_for(c, block, p)
        p = _fit_predict_holdout(Ftr, y_tr, g_tr, Fte)
        preds[comb] = p
        _rows_for(comb, block, p)

    p_btr = _fit_predict_holdout(tr[feats], y_tr, g_tr, te[feats])
    preds["BTR"] = p_btr
    _rows_for("BTR", "BTR", p_btr)

    tab = pd.DataFrame(rows)
    ok = np.isfinite(p_btr) & np.isfinite(y_te)
    maes = {c: _mae(y_te[ok], preds[c][ok]) for c in preds}
    conv_names = list(blocks["GT-informed conventional"][0].columns) +         ["conventional_combined"]
    obsv_names = list(blocks["no-GT observable"][0].columns) + ["observable_combined"]

    def _best(names):
        cand = [c for c in names if np.isfinite(maes.get(c, np.nan))]
        return min(cand, key=lambda c: maes[c]) if cand else None

    boot: Dict[str, object] = {}
    for label, rival in (("vs_conventional", _best(conv_names)),
                         ("vs_observable", _best(obsv_names))):
        if rival is None:
            continue
        b = bootstrap_pair(images[ok], y_te[ok], p_btr[ok], preds[rival][ok],
                           n_boot=n_boot, seed=seed)
        b["rival"] = rival
        b["mae_btr"] = maes.get("BTR")
        b["mae_rival"] = maes[rival]
        boot[label] = b
    boot["n_train_events"] = int(len(tr))
    boot["n_test_events"] = int(len(te))
    boot["trained_datasets"] = sorted(trained_ds)
    return tab, boot, preds, y_te


def run_exp1(df, kind: str, out_dir: str, n_boot: int = 1000, seed: int = 0,
             target: str = DEFAULT_TARGET, train_split_only: bool = False):
    """The Exp1 controlled-perturbation benchmark for one head.

    Three predictor blocks are scored under one protocol (same GBDT capacity,
    same leave-one-dataset-out folds):

    ``GT-informed conventional``
        the five reference-based scalars (``dDice`` ... ``BCSdrop``).  These need
        the ground truth and are *not* available at inference time.
    ``no-GT observable``
        single Contract-F quantities a deployed repairer actually has
        (:data:`OBSERVABLE`) -- the honest deployment-time baseline.
    ``BTR``
        the full Contract-F feature vector.
    """
    import pandas as pd

    y = macro_target(df, target).to_numpy(dtype=float)
    feats = feature_columns(df)
    X = df[feats]
    conv = conventional_frame(df)
    obsv = observable_frame(df)
    fit_mask = train_rows(df) if train_split_only else None
    groups = df["subject_id"].astype(str).to_numpy()
    datasets = df["dataset"].astype(str).to_numpy()
    images = (df["dataset"].astype(str) + ":" + df["image_id"].astype(str)
              + ":" + df["observer"].astype(str)).to_numpy()

    rows: List[Dict[str, object]] = []
    preds: Dict[str, np.ndarray] = {}

    def _block(frame, block_name, combined_name):
        """Score every column of ``frame`` singly, then the combined vector."""
        for name in frame.columns:
            rho, pval = _spearman(frame[name].to_numpy(dtype=float), y)
            pr = _fit_predict_lodo(frame[[name]], y, groups, datasets,
                                   fit_mask=fit_mask)
            preds[name] = pr
            rows.append(dict(head=kind, block=block_name, predictor=name,
                             n_features=1, spearman=rho, spearman_p=pval,
                             mae=_mae(y, pr), r2=_r2(y, pr)))
        pr = _fit_predict_lodo(frame, y, groups, datasets, fit_mask=fit_mask)
        preds[combined_name] = pr
        rows.append(dict(head=kind, block=block_name, predictor=combined_name,
                         n_features=int(frame.shape[1]),
                         spearman=float("nan"), spearman_p=float("nan"),
                         mae=_mae(y, pr), r2=_r2(y, pr)))

    # (a) GT-informed conventional metrics -- unavailable at inference time
    _block(conv, "GT-informed conventional", "conventional_combined")
    # (b) no-GT observables -- the honest deployment-time baseline
    _block(obsv, "no-GT observable", "observable_combined")

    # (c) BTR, same capacity, same CV protocol
    p_btr = _fit_predict_lodo(X, y, groups, datasets, fit_mask=fit_mask)
    preds["BTR"] = p_btr
    rho_btr, p_btr_p = _spearman(p_btr, y)
    rows.append(dict(head=kind, block="BTR", predictor="BTR",
                     n_features=len(feats), spearman=rho_btr, spearman_p=p_btr_p,
                     mae=_mae(y, p_btr), r2=_r2(y, p_btr)))

    tab = pd.DataFrame(rows)
    tab.insert(1, "target", target)
    tab["fit"] = "train-split-only" if train_split_only else "all-events"
    # (d) image-level bootstrap of BTR against the best of each rival block
    maes = {c: _mae(y, preds[c]) for c in preds}
    conv_names = list(conv.columns) + ["conventional_combined"]
    obsv_names = list(obsv.columns) + ["observable_combined"]

    def _best(names):
        cand = [c for c in names if np.isfinite(maes.get(c, np.nan))]
        return min(cand, key=lambda c: maes[c]) if cand else None

    best_conv, best_obsv = _best(conv_names), _best(obsv_names)
    boot: Dict[str, object] = {}
    for label, rival in (("vs_conventional", best_conv), ("vs_observable", best_obsv)):
        if rival is None:
            continue
        b = bootstrap_ci(images, y, p_btr, preds[rival], n_boot=n_boot, seed=seed)
        b["rival"] = rival
        b["mae_btr"] = maes.get("BTR", _mae(y, p_btr))
        b["mae_rival"] = maes[rival]
        boot[label] = b
    # kept flat for backwards compatibility with the earlier JSON consumers
    if "vs_conventional" in boot:
        boot.update({k: v for k, v in boot["vs_conventional"].items()
                     if k not in ("rival", "mae_rival")})
        boot["best_conventional"] = best_conv
        boot["mae_best_conventional"] = maes.get(best_conv, float("nan"))
    tab["best_conventional"] = best_conv
    tab["best_observable"] = best_obsv
    tab["n_events"] = int(len(df))
    tab["n_images"] = int(len(np.unique(images)))
    return tab, boot, preds, y


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="BTR risk heads + Exp1 benchmark")
    ap.add_argument("--events", default=os.path.join(RESULTS_DIR, "c1_events_harm.parquet"))
    ap.add_argument("--out-dir", "--out_dir", dest="out_dir", default=RESULTS_DIR)
    ap.add_argument("--runs-dir", default=RUNS_DIR)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--min-events", type=int, default=100)
    ap.add_argument("--target", default="hdep", choices=("hdep", "habs", "both"),
                    help="supervision target: hdep = pre-registered net harm "
                         "max(0,(|dB|-|dB_ctrl|)/sigma); habs = raw absolute "
                         "standardized harm |dB_topo|/sigma used by the DEPLOYED "
                         "heads (DECISIONS.md 2026-09-03); both = run each")
    ap.add_argument("--train-split-only", action="store_true",
                    help="fit heads only on events whose image is in the TRAINING "
                         "split of its dataset -- required for any head S4 will "
                         "apply to test-split predictions, otherwise the head has "
                         "seen perturbations of the very image it prices. Heads go "
                         "to <runs_dir>/<target>_trainonly/.")
    ap.add_argument("--indomain", action="store_true",
                    help="also run the in-domain Exp1 (fit on train split, score on "
                         "test split per dataset) -> tab1_exp1_indomain.csv")
    args = ap.parse_args(argv)

    import pandas as pd

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)

    path = args.events
    if not os.path.exists(path):
        alt = os.path.splitext(path)[0] + ".csv"
        if not os.path.exists(alt):
            raise FileNotFoundError(
                f"{path} not found -- run `python -m src.c1.stats_c1` first to add H_dep")
        path = alt
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    print(f"[in] {path} rows={len(df)}")

    targets = list(TARGETS) if args.target == "both" else [args.target]
    for t in targets:
        ensure_target(df, t)

    add_split(df)
    n_unmapped = int(df["split"].isna().sum())
    if n_unmapped:
        print("[split] WARNING %d events have no split mapping" % n_unmapped)
    for dsn, g in df.groupby(df["dataset"].astype(str)):
        ntr = g[g["split"] == "train"]["image_id"].nunique()
        nte = g[g["split"] == "test"]["image_id"].nunique()
        print("[split] %-10s %d train images / %d test images" % (dsn, ntr, nte))

    subsets = {"miss": df[df["type"] == "sever"].copy(),
               "false": df[df["type"] == "bridge"].copy()}

    all_tabs, boots = [], {}
    indomain_tabs, indomain_boots = [], {}
    for target in targets:
        # heads for the pre-registered target keep the historical flat layout;
        # any other target gets its own subdirectory (see head_path)
        head_target = target + (TRAINONLY_SUFFIX if args.train_split_only else "")
        rd = (args.runs_dir if head_target == "hdep"
              else os.path.join(args.runs_dir, head_target))
        os.makedirs(rd, exist_ok=True)
        tabs = []
        for kind, sub in subsets.items():
            sub = sub.reset_index(drop=True)
            print(f"[{target}/{kind}] n_events={len(sub)} n_images="
                  f"{sub['image_id'].nunique() if len(sub) else 0}")
            if len(sub) < args.min_events:
                print(f"[{target}/{kind}] too few events "
                      f"({len(sub)} < {args.min_events}); skipped")
                continue
            tcols = target_columns(sub, target)
            # rows allowed into a FIT (never test-split images when the flag is
            # on); evaluation below always uses the full subset
            fit = sub[train_rows(sub)] if args.train_split_only else sub
            if args.train_split_only:
                print("[%s/%s] fitting on %d/%d events from %s (%d train images)"
                      % (head_target, kind, len(fit), len(sub),
                         sorted(fit["dataset"].astype(str).unique()),
                         fit["image_id"].nunique()))
            if len(fit) < args.min_events:
                print("[%s/%s] too few fit events; skipped" % (head_target, kind))
                continue
            meta = dict(target=target, train_split_only=bool(args.train_split_only),
                        n_fit_events=int(len(fit)),
                        fit_datasets=sorted(fit["dataset"].astype(str).unique()))
            # ---- the deployable head -------------------------------------
            head = BTRHead(kind)
            head.fit(fit[feature_columns(fit)], fit[tcols],
                     groups=fit["subject_id"].astype(str).to_numpy())
            head.meta = dict(getattr(head, "meta", {}) or {}, **meta)
            p = head.save(head_path(kind, args.runs_dir, "all", head_target))
            print("[%s/%s] head saved -> %s" % (head_target, kind, p))
            # ---- leave-one-dataset-out heads (the LODO block of S4) ------
            for held in sorted(sub["dataset"].unique()):
                tr = fit[fit["dataset"] != held]
                if len(tr) < args.min_events:
                    print("[%s/%s] wo_%s: only %d fit events; skipped"
                          % (head_target, kind, held, len(tr)))
                    continue
                h = BTRHead("%s_wo_%s" % (kind, held))
                h.fit(tr[feature_columns(tr)], tr[target_columns(tr, target)],
                      groups=tr["subject_id"].astype(str).to_numpy())
                h.meta = dict(getattr(h, "meta", {}) or {},
                              **dict(meta, held_out=held, n_fit_events=int(len(tr))))
                h.save(head_path(kind, args.runs_dir, "wo_%s" % held, head_target))
            # ---- Exp1 ----------------------------------------------------
            tab, boot, _preds, _y = run_exp1(sub, kind, args.out_dir,
                                             n_boot=args.n_boot, seed=args.seed,
                                             target=target,
                                             train_split_only=args.train_split_only)
            tabs.append(tab)
            boots["%s/%s" % (target, kind)] = boot
            print(tab.to_string(index=False))
            # ---- in-domain Exp1 (the deployable protocol) ----------------
            if args.indomain:
                itab, iboot, _ip, _iy = run_exp1_indomain(
                    sub, kind, n_boot=args.n_boot, seed=args.seed, target=target)
                if itab is not None:
                    indomain_tabs.append(itab)
                    indomain_boots["%s/%s" % (target, kind)] = iboot
                    print("[in-domain %s/%s]" % (target, kind))
                    print(itab[itab.scope == "ALL(test)"].to_string(index=False))

        if tabs:
            out = pd.concat(tabs, ignore_index=True)
            all_tabs.append(out)
            fit_sfx = "_trainonly" if args.train_split_only else ""
            tab_csv = os.path.join(args.out_dir,
                                   f"tab1_exp1_{target}{fit_sfx}.csv")
            out.to_csv(tab_csv, index=False)
            print(f"[out] {tab_csv}")
            if target == "hdep" and not args.train_split_only:
                out.to_csv(os.path.join(args.out_dir, "tab1_exp1.csv"), index=False)

    if indomain_tabs:
        it = pd.concat(indomain_tabs, ignore_index=True)
        ip = os.path.join(args.out_dir, "tab1_exp1_indomain.csv")
        it.to_csv(ip, index=False)
        print("[out] %s" % ip)
        jp = os.path.join(args.out_dir, "tab1_exp1_indomain_bootstrap.json")
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(dict(events=path, protocol="fit train split / score test split",
                           bootstrap=indomain_boots), fh, indent=2, default=str)
        print("[out] %s" % jp)

    if all_tabs:
        merged = pd.concat(all_tabs, ignore_index=True)
        merged["protocol"] = "LODO"
        if indomain_tabs:
            it2 = pd.concat(indomain_tabs, ignore_index=True)
            it2 = it2[it2["scope"] == "ALL(test)"].drop(
                columns=["scope", "in_domain", "n"])
            it2["protocol"] = "in-domain"
            merged = pd.concat([merged, it2], ignore_index=True)
        merged.to_csv(os.path.join(args.out_dir, "tab1_exp1_all_targets.csv"),
                      index=False)
    suffix = ("" if args.target == "hdep" else f"_{args.target}") +         ("_trainonly" if args.train_split_only else "")
    js = os.path.join(args.out_dir, f"tab1_exp1_bootstrap{suffix}.json")
    with open(js, "w", encoding="utf-8") as fh:
        json.dump(dict(events=path, gbdt_params=GBDT_PARAMS, targets=targets,
                       bootstrap=boots),
                  fh, indent=2, default=str)
    print(f"[out] {js}")
    print(json.dumps(boots, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
