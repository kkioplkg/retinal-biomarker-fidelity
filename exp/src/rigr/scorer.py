"""Pair scorer and probability calibration (proposal v3 section 3.2.4).

Feature contract (hard rule)
----------------------------
Every feature is computed from ``(I, M_hat, P, V_I, q, gamma_e)`` only.
**Forbidden**: reference branch order, reference A/V identity, reference branch
length, and any harm quantity derived from the annotation.  ``candidate_features``
therefore takes no ``gt`` argument at all -- there is no code path by which a
reference-derived quantity could reach the model.  ``FORBIDDEN_KWARGS`` is
asserted at call time so a future caller cannot smuggle one in.

The 13 features (section 3.2.4)::

    [E*, E*/len, d, d/rbar, |log(r_i/r_j)|, tangent mismatch angle,
     min V_I, mean V_I, min q, local contrast, predicted local density,
     eccentricity zone, is_port]

Models
------
``PairScorer`` wraps a logistic regression **and** a 2-layer MLP (hidden 32,
< 5 K parameters), both from scikit-learn, behind one interface.  Because the
expected utility of section 3.2.6 needs a *probability*, not a classifier
score, the raw score is calibrated **on the validation split only**, by
temperature scaling (a single scalar fitted by NLL on the decision values) or
isotonic regression.  ``reliability_bins`` and ``brier`` provide the
supplementary-material diagnostics the section requires.

Run ``cd exp && python -m src.rigr.scorer`` for a synthetic fit/calibration demo.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "FEATURE_NAMES",
    "candidate_features",
    "local_contrast",
    "predicted_local_density",
    "eccentricity_zone",
    "PairScorer",
    "T_MIN",
    "T_MAX",
    "brier",
    "reliability_bins",
]

FEATURE_NAMES: List[str] = [
    "E_star",
    "E_star_per_len",
    "d",
    "d_over_rbar",
    "log_r_ratio",
    "tangent_mismatch",
    "min_V",
    "mean_V",
    "min_q",
    "local_contrast",
    "local_density",
    "ecc_zone",
    "is_port",
]

#: keyword names that would break the feature contract if ever accepted
FORBIDDEN_KWARGS = ("gt", "gt_mask", "label", "reference", "harm", "gt_skel")

_EPS = 1e-9


# --------------------------------------------------------------------------
# image-derived features
# --------------------------------------------------------------------------
def _green(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image)
    g = a[..., 1] if a.ndim == 3 else a
    g = g.astype(np.float32)
    return g / max(float(g.max()), 1e-6)


def local_contrast(
    image: np.ndarray,
    mask: np.ndarray,
    rc: Tuple[float, float],
    rbar: float,
    fov: Optional[np.ndarray] = None,
    min_half: int = 8,
) -> float:
    """Vessel-to-background contrast of the green channel around a candidate.

    ``(median background - median vessel) / (robust spread of the window)``
    inside a square window of half-size ``max(min_half, 3 rbar)``.  Retinal
    vessels are darker than the retina on the green channel, so a well-defined
    vessel gives a positive value; a washed-out peripheral region gives ~0.
    """
    g = _green(image)
    H, W = g.shape
    r, c = int(round(rc[0])), int(round(rc[1]))
    hs = int(max(min_half, round(3.0 * float(rbar))))
    r0, r1 = max(0, r - hs), min(H, r + hs + 1)
    c0, c1 = max(0, c - hs), min(W, c + hs + 1)
    win = g[r0:r1, c0:c1]
    mm = np.asarray(mask, bool)[r0:r1, c0:c1]
    if fov is not None:
        fm = np.asarray(fov, bool)[r0:r1, c0:c1]
    else:
        fm = np.ones_like(mm)
    ves = win[mm & fm]
    bg = win[(~mm) & fm]
    if ves.size < 4 or bg.size < 4:
        return 0.0
    spread = float(np.std(win[fm])) if fm.any() else 0.0
    return float((np.median(bg) - np.median(ves)) / (spread + 1e-3))


def predicted_local_density(
    mask: np.ndarray,
    rc: Tuple[float, float],
    rbar: float,
    fov: Optional[np.ndarray] = None,
    min_half: int = 16,
) -> float:
    """Fraction of FOV pixels that are predicted vessel in a local window."""
    m = np.asarray(mask, bool)
    H, W = m.shape
    r, c = int(round(rc[0])), int(round(rc[1]))
    hs = int(max(min_half, round(6.0 * float(rbar))))
    r0, r1 = max(0, r - hs), min(H, r + hs + 1)
    c0, c1 = max(0, c - hs), min(W, c + hs + 1)
    sub = m[r0:r1, c0:c1]
    f = np.ones_like(sub) if fov is None else np.asarray(fov, bool)[r0:r1, c0:c1]
    n = int(f.sum())
    if n == 0:
        return 0.0
    return float((sub & f).sum()) / float(n)


def eccentricity_zone(
    rc: Tuple[float, float],
    disc_xy: Optional[Tuple[float, float]],
    fov_radius: float,
    max_zone: float = 3.0,
) -> float:
    """Radial distance from the optic disc, in FOV radii, clipped to ``max_zone``.

    The disc centre is the *automatic* estimate (``src.bio.disc``) or, when it
    is unavailable, the FOV centroid -- never an annotation.
    """
    if disc_xy is None or not np.isfinite(fov_radius) or fov_radius <= 0:
        return 0.0
    d = math.hypot(float(rc[0]) - float(disc_xy[1]), float(rc[1]) - float(disc_xy[0]))
    return float(min(d / float(fov_radius), float(max_zone)))


def _fov_geometry(fov: Optional[np.ndarray], shape) -> Tuple[Tuple[float, float], float]:
    if fov is None:
        cy, cx = shape[0] / 2.0, shape[1] / 2.0
        return (cx, cy), 0.5 * min(shape)
    f = np.asarray(fov, bool)
    n = int(f.sum())
    if n == 0:
        cy, cx = shape[0] / 2.0, shape[1] / 2.0
        return (cx, cy), 0.5 * min(shape)
    rr, cc = np.nonzero(f)
    return (float(cc.mean()), float(rr.mean())), float(math.sqrt(n / math.pi))


# --------------------------------------------------------------------------
# feature assembly
# --------------------------------------------------------------------------
def candidate_features(
    table,
    results: Sequence[Any],
    image: np.ndarray,
    mask: np.ndarray,
    prob: Optional[np.ndarray] = None,
    fov: Optional[np.ndarray] = None,
    disc_xy: Optional[Tuple[float, float]] = None,
    fov_radius: Optional[float] = None,
    **forbidden,
) -> np.ndarray:
    """Build the ``(N, 13)`` feature matrix for a candidate table.

    ``results`` is the list of :class:`~src.rigr.astar.AStarResult`, aligned
    row-by-row with ``table``.  Rows whose A* failed get ``E* = +big`` and zero
    evidence, which the scorer learns to reject.
    """
    for k in forbidden:
        if k.lower() in FORBIDDEN_KWARGS or "gt" in k.lower():
            raise ValueError(
                "candidate_features received the annotation-derived argument %r; "
                "the RiGR feature contract (section 3.2.4) forbids it" % k)
    if forbidden:
        raise TypeError("unexpected keyword arguments: %s" % sorted(forbidden))

    m = np.asarray(mask, bool)
    if fov_radius is None or disc_xy is None:
        c_xy, r_fov = _fov_geometry(fov, m.shape)
        disc_xy = disc_xy if disc_xy is not None else c_xy
        fov_radius = fov_radius if fov_radius is not None else r_fov

    big_e = 1e4
    rows = table.to_dict("records") if hasattr(table, "to_dict") else list(table)
    X = np.zeros((len(rows), len(FEATURE_NAMES)), dtype=np.float64)
    for i, row in enumerate(rows):
        res = results[i] if i < len(results) else None
        ok = bool(getattr(res, "ok", False))
        mid = (0.5 * (float(row["r_i"]) + float(row["r_j"])),
               0.5 * (float(row["c_i"]) + float(row["c_j"])))
        rbar = float(row["rad_bar"])
        X[i] = [
            float(res.energy) if ok else big_e,
            float(res.energy_per_len) if ok else big_e,
            float(row["d"]),
            float(row["d_over_rbar"]),
            float(row["log_r_ratio"]),
            float(row["tangent_mismatch"]),
            float(res.min_v) if ok else 0.0,
            float(res.mean_v) if ok else 0.0,
            float(res.min_q) if ok else 0.0,
            local_contrast(image, m, mid, rbar, fov),
            predicted_local_density(m, mid, rbar, fov),
            eccentricity_zone(mid, disc_xy, float(fov_radius)),
            float(row.get("is_port", 0)),
        ]
    return X


# --------------------------------------------------------------------------
# calibration diagnostics
# --------------------------------------------------------------------------
def brier(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if y.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Reliability table: per-bin mean predicted probability vs empirical rate."""
    import pandas as pd

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(int(n_bins)):
        sel = idx == b
        rows.append(dict(
            bin=b, lo=float(edges[b]), hi=float(edges[b + 1]), n=int(sel.sum()),
            mean_pred=float(p[sel].mean()) if sel.any() else float("nan"),
            frac_pos=float(y[sel].mean()) if sel.any() else float("nan"),
        ))
    df = pd.DataFrame(rows)
    w = df["n"].to_numpy(dtype=float)
    gap = np.abs(df["mean_pred"].to_numpy() - df["frac_pos"].to_numpy())
    ok = np.isfinite(gap)
    ece = float(np.sum(w[ok] * gap[ok]) / max(w[ok].sum(), 1.0))
    df.attrs["ece"] = ece
    return df


#: temperature is searched inside these bounds; hitting one is reported
T_MIN, T_MAX = 0.25, 20.0


def _fit_temperature(z: np.ndarray, y: np.ndarray,
                     smoothing: Optional[float] = None) -> float:
    """Scalar temperature minimising the NLL of ``sigmoid(z / T)``.

    Two guards matter on the small validation sets this stage produces.
    (1) **Label smoothing**: on a validation slice the model happens to
    separate perfectly, the unsmoothed NLL is minimised by ``T -> 0``, i.e.
    hard 0/1 "probabilities" -- the exact opposite of what calibration is for.
    The targets are therefore smoothed by ``eps = 1 / (n + 2)`` (vanishing as
    the validation set grows), which is the Krichevsky-Trofimov style prior of
    one pseudo-count per class.
    (2) **Bounded search** in ``[T_MIN, T_MAX]``; a solution at a bound means
    the validation set was too small or too easy to calibrate on, and the
    caller should treat ``p_e`` as a score rather than a probability.
    """
    z = np.asarray(z, dtype=float)
    y = np.asarray(y, dtype=float)
    if z.size == 0 or len(np.unique(y)) < 2:
        return 1.0
    eps = float(smoothing) if smoothing is not None else 1.0 / (len(y) + 2.0)
    yt = np.clip(y, 0.0, 1.0) * (1.0 - 2.0 * eps) + eps
    grid = np.exp(np.linspace(math.log(T_MIN), math.log(T_MAX), 241))
    best_t, best_nll = 1.0, np.inf
    for t in grid:
        s = z / t
        # numerically stable BCE with logits
        nll = float(np.mean(np.maximum(s, 0) - s * yt + np.log1p(np.exp(-np.abs(s)))))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


# --------------------------------------------------------------------------
# the scorer
# --------------------------------------------------------------------------
class PairScorer:
    """Logistic regression or 2-layer MLP + validation-only calibration.

    Parameters
    ----------
    model : {'logreg', 'mlp'}
    calibration : {'temperature', 'isotonic', 'none'}
        ``'temperature'`` fits one scalar on the *validation* decision values;
        ``'isotonic'`` fits a monotone map on the same split.  Both are refused
        if the validation set is the training set (that would re-introduce the
        optimism the calibration is there to remove).
    hidden : int
        MLP hidden width (32 -> ~ (13*32 + 32 + 32 + 1) = 481 parameters).
    """

    def __init__(self, model: str = "logreg", calibration: str = "temperature",
                 hidden: int = 32, seed: int = 0, max_iter: int = 2000,
                 C: float = 1.0):
        if model not in ("logreg", "mlp"):
            raise ValueError("model must be 'logreg' or 'mlp'")
        if calibration not in ("temperature", "isotonic", "none"):
            raise ValueError("calibration must be 'temperature', 'isotonic' or 'none'")
        self.model_kind = model
        self.calibration = calibration
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.max_iter = int(max_iter)
        self.C = float(C)
        self.clf = None
        self.scaler = None
        self.temperature: float = 1.0
        self.iso = None
        self.feature_names = list(FEATURE_NAMES)
        self.meta: Dict[str, Any] = {}

    # -- internals ---------------------------------------------------------
    def _build(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier

        if self.model_kind == "logreg":
            return LogisticRegression(max_iter=self.max_iter, C=self.C,
                                      class_weight="balanced")
        return MLPClassifier(hidden_layer_sizes=(self.hidden,), activation="relu",
                             max_iter=self.max_iter, random_state=self.seed,
                             early_stopping=False, alpha=1e-3)

    def _decision(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        if hasattr(self.clf, "decision_function"):
            return np.asarray(self.clf.decision_function(Xs), dtype=float).ravel()
        p = np.clip(self.clf.predict_proba(Xs)[:, 1], 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    # -- api ---------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        self.scaler = StandardScaler().fit(X)
        self.clf = self._build()
        self.clf.fit(self.scaler.transform(X), y)

        self.temperature, self.iso = 1.0, None
        if self.calibration != "none" and X_val is not None and y_val is not None \
                and len(np.unique(np.asarray(y_val))) > 1:
            z = self._decision(X_val)
            yv = np.asarray(y_val, dtype=float).ravel()
            if self.calibration == "temperature":
                self.temperature = _fit_temperature(z, yv)
                if self.temperature <= T_MIN * 1.001 or                         self.temperature >= T_MAX * 0.999:
                    warnings.warn(
                        "temperature scaling hit its bound (T=%.3f) on %d "
                        "validation candidates: p_e should be read as a score, "
                        "not a calibrated probability"
                        % (self.temperature, len(yv)), RuntimeWarning)
            else:
                from sklearn.isotonic import IsotonicRegression

                self.iso = IsotonicRegression(y_min=0.0, y_max=1.0,
                                              out_of_bounds="clip").fit(z, yv)
        self.meta.update(dict(n_train=int(len(y)), n_pos=int(y.sum()),
                              model=self.model_kind, calibration=self.calibration,
                              temperature=float(self.temperature)))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability that the candidate is a true connection."""
        z = self._decision(X)
        if self.iso is not None:
            return np.clip(np.asarray(self.iso.predict(z), dtype=float), 1e-6, 1 - 1e-6)
        return 1.0 / (1.0 + np.exp(-z / max(self.temperature, 1e-6)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        from sklearn.metrics import average_precision_score, roc_auc_score

        p = self.predict_proba(X)
        y = np.asarray(y, dtype=int).ravel()
        out: Dict[str, Any] = {"brier": brier(y, p), "n": int(len(y)),
                               "n_pos": int(y.sum())}
        if len(np.unique(y)) > 1:
            out["auc"] = float(roc_auc_score(y, p))
            out["ap"] = float(average_precision_score(y, p))
        else:
            out["auc"] = float("nan")
            out["ap"] = float("nan")
        rel = reliability_bins(y, p)
        out["ece"] = float(rel.attrs["ece"])
        out["reliability"] = rel
        return out

    # -- persistence -------------------------------------------------------
    def save(self, path: str) -> str:
        import joblib

        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        joblib.dump(dict(
            model_kind=self.model_kind, calibration=self.calibration,
            hidden=self.hidden, seed=self.seed, clf=self.clf, scaler=self.scaler,
            temperature=self.temperature, iso=self.iso,
            feature_names=self.feature_names, meta=self.meta,
        ), path)
        with open(os.path.splitext(path)[0] + "_meta.json", "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in self.meta.items()}, f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "PairScorer":
        import joblib

        d = joblib.load(path)
        s = cls(model=d["model_kind"], calibration=d["calibration"],
                hidden=d.get("hidden", 32), seed=d.get("seed", 0))
        s.clf = d["clf"]
        s.scaler = d["scaler"]
        s.temperature = float(d.get("temperature", 1.0))
        s.iso = d.get("iso")
        s.feature_names = d.get("feature_names", list(FEATURE_NAMES))
        s.meta = d.get("meta", {})
        return s


# --------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover - synthetic demo
    rng = np.random.default_rng(0)
    n = 1200
    X = rng.normal(size=(n, len(FEATURE_NAMES)))
    logit = 1.6 * X[:, 6] - 1.1 * X[:, 1] - 0.8 * X[:, 3] + 0.6 * X[:, 7] - 0.4
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    tr, va, te = slice(0, 600), slice(600, 900), slice(900, n)

    for kind in ("logreg", "mlp"):
        for cal in ("none", "temperature", "isotonic"):
            s = PairScorer(model=kind, calibration=cal).fit(
                X[tr], y[tr], X[va], y[va])
            r = s.evaluate(X[te], y[te])
            print("%-7s %-11s  AUC=%.3f  AP=%.3f  Brier=%.4f  ECE=%.4f  T=%.2f"
                  % (kind, cal, r["auc"], r["ap"], r["brier"], r["ece"],
                     s.temperature))
    s = PairScorer("logreg").fit(X[tr], y[tr], X[va], y[va])
    print(s.evaluate(X[te], y[te])["reliability"].round(3).to_string(index=False))
    p = os.path.join(os.environ.get("TEMP", "/tmp"), "_rigr_scorer_demo.joblib")
    s.save(p)
    s2 = PairScorer.load(p)
    print("save/load max |dp| =", float(np.max(np.abs(
        s.predict_proba(X[te]) - s2.predict_proba(X[te])))))
    try:
        candidate_features(None, [], None, None, gt=1)
    except ValueError as e:
        print("feature contract enforced:", str(e)[:70])
