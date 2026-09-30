"""RiGR as a ``fn(mask, fov, image, prob=None) -> mask`` hook for stage S5.

``src.robust.run_stability`` evaluates biomarker stability under K
anatomy-preserving views, optionally repairing each view's predicted mask
through ``--repair rigr_cmd --repair-fn 'module:function'``.  That hook takes no
arguments of its own, so this module is configured through **environment
variables**, which is what ``src.pipeline.s4_all`` sets before launching the
robustness runs::

    RIGR_HOOK_HEAD      runs/rigr_head/<ds>/seed0/best.pt   (required for the
                        learned evidence; without it the analytic Frangi
                        evidence with a uniform q is used and the hook says so)
    RIGR_HOOK_SCORER    runs/rigr_models/<ds>/seed0/scorer.joblib
    RIGR_HOOK_MODE      prob | uniform | risk        (default: risk)
    RIGR_HOOK_GPU       torch device index, -1 = CPU  (default: -1)
    RIGR_HOOK_LAM / _ETA / _TAU / _MU / _LAMBDA_THETA
    RIGR_HOOK_BTR_MISS / _BTR_FALSE / _BTR_VARIANT / _BTR_RUNS_DIR

Everything is loaded once per process and cached.

One caveat, stated here because it changes what Fig.4A measures.  The head is
conditioned on ``[I, P]``.  ``run_stability`` now hands the hook the view's own
probability map (``prob=``), so the evidence is computed from the real ``P``
exactly as in the main runs.  If a *caller* uses the older three-argument
signature, ``P`` is unavailable and this module falls back to
``synth_cuts.synth_prob(mask)`` -- a blurred copy of the binary mask.  That
fallback is a **stand-in, not the segmenter's probability map**; every repaired
mask produced that way is flagged in ``LAST_CALL['prob_source']`` and the
first occurrence prints a warning, so a Fig.4A run can never quietly report
numbers produced from a surrogate ``P``.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional

import numpy as np

__all__ = ["repair", "config", "reset"]

_STATE: Dict[str, Any] = {}
#: diagnostics of the most recent call (n candidates, n selected, prob source)
LAST_CALL: Dict[str, Any] = {}


def _env(name: str, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _geom_scale():
    """``m_R`` from ``RIGR_HOOK_GEOM_SCALE`` or geom_scale.json by the scorer."""
    import json
    import os

    v = _env("RIGR_HOOK_GEOM_SCALE")
    if v:
        return float(v)
    sc = _env("RIGR_HOOK_SCORER")
    if not sc:
        return None
    p = os.path.join(os.path.dirname(sc), "geom_scale.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return float(json.load(f)["m_R"])


def config() -> Dict[str, Any]:
    """Resolved configuration (also what the orchestrator logs)."""
    return dict(
        head=_env("RIGR_HOOK_HEAD"),
        scorer=_env("RIGR_HOOK_SCORER"),
        mode=_env("RIGR_HOOK_MODE", "risk"),
        gpu=int(_env("RIGR_HOOK_GPU", "-1")),
        lam=float(_env("RIGR_HOOK_LAM", "1.0")),
        eta=float(_env("RIGR_HOOK_ETA", "0.1")),
        tau=float(_env("RIGR_HOOK_TAU", "0.5")),
        # m_R, the scale that puts C_geom on the same footing as R.  Default:
        # geom_scale.json next to RIGR_HOOK_SCORER, i.e. the same convention
        # run_rigr uses.  Risk mode without it raises -- see
        # src/rigr/geom_scale.py and DECISIONS.md 2026-09-06 12:30.  Fig.4's
        # repaired arms went through this hook, so they were affected by the
        # same defect and are being re-run.
        geom_scale=_geom_scale(),
        mu=_env("RIGR_HOOK_MU"),
        lambda_theta=_env("RIGR_HOOK_LAMBDA_THETA"),
        btr_miss=_env("RIGR_HOOK_BTR_MISS"),
        btr_false=_env("RIGR_HOOK_BTR_FALSE"),
        btr_variant=_env("RIGR_HOOK_BTR_VARIANT", "all"),
        btr_runs_dir=_env("RIGR_HOOK_BTR_RUNS_DIR"),
    )


def reset() -> None:
    """Drop the cached head / scorer / risk backend (used by the tests)."""
    _STATE.clear()


def _load():
    if _STATE:
        return _STATE
    cfg = config()
    from src.rigr import astar as astar_mod
    from src.rigr.run_rigr import make_evidence_fn

    head = cfg["head"] if cfg["head"] and os.path.exists(cfg["head"]) else None
    if cfg["head"] and head is None:
        warnings.warn("RIGR_HOOK_HEAD=%r does not exist; the hook falls back to "
                      "the analytic Frangi evidence with a uniform q -- this is "
                      "NOT the deployed RiGR configuration" % cfg["head"],
                      RuntimeWarning)
    evidence_fn, ev_name = make_evidence_fn(head, cfg["gpu"],
                                            "auto" if head else "frangi")

    scorer = None
    if cfg["scorer"] and os.path.exists(cfg["scorer"]):
        from src.rigr.scorer import PairScorer

        scorer = PairScorer.load(cfg["scorer"])
    elif cfg["scorer"]:
        warnings.warn("RIGR_HOOK_SCORER=%r does not exist; every candidate gets "
                      "p_e = 0.5 and the repair is effectively ungated"
                      % cfg["scorer"], RuntimeWarning)

    backend = None
    if cfg["mode"] == "risk" and (cfg["btr_miss"] or cfg["btr_false"]
                                  or cfg["btr_variant"] != "all"
                                  or cfg["btr_runs_dir"]):
        from src.rigr.btr_backend import make_backend

        try:
            backend = make_backend(cfg["btr_miss"], cfg["btr_false"],
                                   cfg["btr_runs_dir"], cfg["btr_variant"])
        except FileNotFoundError as exc:
            warnings.warn("repair hook: %s; falling back to the all-data BTR "
                          "heads" % exc, RuntimeWarning)

    astar_kwargs: Dict[str, Any] = {}
    if cfg["mu"] is not None and float(cfg["mu"]) != float(astar_mod.MU_KAPPA):
        astar_kwargs["mu"] = float(cfg["mu"])
    if (cfg["lambda_theta"] is not None
            and float(cfg["lambda_theta"]) != float(astar_mod.LAMBDA_THETA)):
        astar_kwargs["lam"] = float(cfg["lambda_theta"])

    _STATE.update(cfg=cfg, evidence_fn=evidence_fn, evidence_name=ev_name,
                  scorer=scorer, backend=backend, astar_kwargs=astar_kwargs,
                  warned_prob=False)
    print("[repair_hook] m_R=%s" % cfg.get("geom_scale"), flush=True)
    print("[repair_hook] mode=%s evidence=%s scorer=%s btr=%s astar=%s"
          % (cfg["mode"], ev_name, bool(scorer),
             getattr(backend, "__name__", "src.c1.btr(all)"),
             astar_kwargs or "defaults"), flush=True)
    return _STATE


def repair(mask, fov, image, prob: Optional[np.ndarray] = None):
    """Repair one binary mask with the configured RiGR pipeline."""
    from src.rigr import candidates as cand_mod
    from src.rigr import utility as util_mod
    from src.rigr.astar import solve_candidate
    from src.rigr.scorer import candidate_features
    from src.topo import skeleton as sk

    st = _load()
    cfg = st["cfg"]
    m = sk.as_bool(mask)
    f = sk.fov_or_true(fov, m.shape)
    m = m & f

    prob_source = "view_prob"
    if prob is None:
        from src.rigr.synth_cuts import synth_prob

        prob = synth_prob(m, sigma=1.5)
        prob_source = "synthetic(blurred mask)"
        if not st["warned_prob"]:
            st["warned_prob"] = True
            warnings.warn("repair hook called without the view's probability "
                          "map; P is a blurred copy of the binary mask, which is "
                          "a stand-in and not the segmenter's P", RuntimeWarning)
    prob = np.clip(np.asarray(prob, dtype=np.float32), 0.0, 1.0)

    V, Q = st["evidence_fn"](image, m, prob, f)
    out = cand_mod.generate_candidates(m, prob, f)
    table = out["table"]
    if len(table) == 0:
        LAST_CALL.update(n_candidates=0, n_selected=0, prob_source=prob_source)
        return m

    results = []
    for row in table.to_dict("records"):
        gp = out["port_pixels"].get(int(row["node_j"])) if int(row["is_port"]) else None
        results.append(solve_candidate(row, V, Q, goal_pixels=gp,
                                       **st["astar_kwargs"]))

    X = candidate_features(table, results, image, m, prob, f)
    p = (st["scorer"].predict_proba(X) if st["scorer"] is not None
         else np.full(len(table), 0.5))
    rep = util_mod.repair_mask(
        m, table, results, p, mode=cfg["mode"], lam=cfg["lam"], eta=cfg["eta"],
        tau=cfg["tau"], fov=f, labels=out["labels"], theta_mask=out["theta_mask"],
        radius=out["radius"], X=X, image=image, disc=None,
        risk_backend=st["backend"], geom_scale=cfg.get("geom_scale"))
    LAST_CALL.update(n_candidates=int(len(table)),
                     n_selected=int(len(rep["selected"])),
                     prob_source=prob_source,
                     risk_backend=str(rep["utility"].attrs.get("risk_backend", "?")))
    return np.asarray(rep["mask"], dtype=bool)
