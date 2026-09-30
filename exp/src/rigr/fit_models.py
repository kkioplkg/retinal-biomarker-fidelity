"""Stage S4 step C: fit the pair scorer and the deployed ``R_false`` head.

    python -m src.rigr.fit_models --data runs/rigr_data/drive --seed 0 \
        --out runs/rigr_models/drive/seed0

Inputs are exactly what :mod:`src.rigr.build_data` wrote:
``<data>/seed<k>/pairs.npz`` and (optionally) ``<data>/rfalse_train.csv``.
Several ``--data`` directories may be given; they are concatenated, which is
how the LODO block fits one scorer on the union of three source datasets.

Pair scorer
-----------
:class:`src.rigr.scorer.PairScorer` with validation-only calibration.  The
validation slice is **30% of the images, split by subject** (proposal section
3.2.5): whole subjects go to one side or the other, so no candidate of a
subject can calibrate a model that was fitted on another candidate of the same
subject.  Calibration on the training split would re-introduce exactly the
optimism temperature scaling exists to remove, so it is refused: when no usable
validation group exists the scorer is saved **uncalibrated** and says so in
``report.json``.

Deployed ``R_false``
--------------------
:func:`src.c1.btr.fit_deployed_false` -- LightGBM, one regressor per
``(biomarker, pipeline)``, fitted on the *wrong* connections only, with the
inner ``GroupKFold`` grouped by subject.  Saved as
``<out>/btr_false_deployed.joblib`` and consumed by ``run_rigr --btr_false``.
When there is not enough data the head is not written and ``report.json``
records ``rfalse_fitted: false``; the caller must then fall back to the C1
head, which ``run_rigr`` does with a printed warning rather than silently.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["load_pairs", "fit_scorer", "main"]


def load_pairs(data_dirs: Sequence[str], seed: int):
    """Concatenate ``<data>/seed<k>/pairs.npz`` over several data directories."""
    X, y, img, ds, subj = [], [], [], [], []
    for d in data_dirs:
        p = os.path.join(d, "seed%d" % int(seed), "pairs.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(
                "%s missing; run `python -m src.rigr.build_data --out %s "
                "--seed %d` first" % (p, d, seed))
        z = np.load(p, allow_pickle=False)
        X.append(z["X"])
        y.append(z["y"])
        img.append(z["image"].astype(str))
        ds.append(z["dataset"].astype(str) if "dataset" in z
                  else np.full(len(z["y"]), os.path.basename(d)))
        subj.append(z["subject_id"].astype(str) if "subject_id" in z
                    else z["image"].astype(str))
    return (np.concatenate(X, 0), np.concatenate(y, 0),
            np.concatenate(img), np.concatenate(ds), np.concatenate(subj))


def fit_scorer(X, y, subject, dataset, model: str = "logreg",
               calibration: str = "temperature", val_fraction: float = 0.3,
               seed: int = 0):
    """Fit + calibrate a :class:`PairScorer` on a subject-disjoint 70/30 split."""
    import warnings

    from src.rigr.scorer import PairScorer

    # group by (dataset, subject) so two datasets that happen to reuse a
    # subject id are not merged into one group
    groups = np.asarray(["%s/%s" % (d, s) for d, s in zip(dataset, subject)])
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_val = int(round(float(val_fraction) * len(uniq))) if len(uniq) > 1 else 0
    n_val = max(1, n_val) if len(uniq) > 1 else 0
    val_groups = set(uniq[order[:n_val]].tolist())
    m_val = np.isin(groups, list(val_groups)) if val_groups else np.zeros(len(y), bool)

    Xtr, ytr = X[~m_val], y[~m_val]
    Xva, yva = X[m_val], y[m_val]
    note = ""
    if Xva.shape[0] == 0 or len(np.unique(yva)) < 2 or len(np.unique(ytr)) < 2:
        note = ("no usable subject-disjoint validation slice (%d val candidates, "
                "%d classes); the scorer is saved UNCALIBRATED"
                % (int(Xva.shape[0]), int(len(np.unique(yva)))))
        warnings.warn(note, RuntimeWarning)
        Xva, yva = None, None

    sc = PairScorer(model=model, calibration=calibration, seed=seed)
    sc.fit(Xtr, ytr, Xva, yva)
    rep: Dict[str, object] = dict(
        n_total=int(len(y)), n_pos=int(y.sum()),
        n_train=int(len(ytr)), n_val=int(0 if Xva is None else len(yva)),
        n_groups=int(len(uniq)), n_val_groups=int(len(val_groups)),
        val_fraction=float(val_fraction), model=model,
        calibration=calibration if Xva is not None else "none(no val slice)",
        temperature=float(sc.temperature), note=note)
    if Xva is not None:
        ev = sc.evaluate(Xva, yva)
        rep["val"] = {k: v for k, v in ev.items() if k != "reliability"}
        rep["_reliability"] = ev["reliability"]
    ev_in = sc.evaluate(X, y)
    rep["in_sample"] = {k: v for k, v in ev_in.items() if k != "reliability"}
    return sc, rep


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(
        prog="python -m src.rigr.fit_models",
        description="S4 step C: fit the pair scorer + the deployed R_false head")
    ap.add_argument("--data", action="append", required=True,
                    help="a runs/rigr_data/<ds> directory; repeat for a union")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scorer_model", default="logreg", choices=["logreg", "mlp"])
    ap.add_argument("--calibration", default="temperature",
                    choices=["temperature", "isotonic", "none"])
    ap.add_argument("--val_fraction", type=float, default=0.3)
    ap.add_argument("--no_rfalse", action="store_true",
                    help="skip the deployed R_false head")
    ap.add_argument("--rfalse_all_candidates", action="store_true",
                    help="fit R_false on every candidate, not only the wrong ones")
    ap.add_argument("--min_rfalse_rows", type=int, default=60)
    ap.add_argument("--rfalse_target", default="habs", choices=["habs", "hdep"],
                    help="supervision target of the deployed R_false head. "
                         "'habs' (the default, DECISIONS.md 2026-09-03 08:30) "
                         "is |B(M+)-B(M-)|/sigma, which is also the target of "
                         "the runs/btr/habs_trainonly R_miss head it is paired "
                         "with -- BTRBackend refuses to mix the two families.")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    X, y, img, ds, subj = load_pairs(args.data, args.seed)
    print("[fit_models] pairs X%s  positives %d/%d  images %d  subjects %d"
          % (X.shape, int(y.sum()), len(y), len(np.unique(img)),
             len(np.unique(subj))), flush=True)

    scorer, rep = fit_scorer(X, y, subj, ds, model=args.scorer_model,
                             calibration=args.calibration,
                             val_fraction=args.val_fraction, seed=args.seed)
    rel = rep.pop("_reliability", None)
    scorer.save(os.path.join(args.out, "scorer.joblib"))
    if rel is not None:
        rel.to_csv(os.path.join(args.out, "scorer_reliability.csv"), index=False)
    print("[fit_models] scorer: %s" % json.dumps(
        {k: v for k, v in rep.items() if k in ("n_train", "n_val", "calibration",
                                               "temperature", "val", "in_sample")},
        default=float), flush=True)

    # ---- deployed R_false --------------------------------------------------
    report: Dict[str, object] = dict(seed=int(args.seed), data=list(args.data),
                                     scorer=rep, rfalse_fitted=False)
    if not args.no_rfalse:
        from src.c1.btr import fit_deployed_false

        frames = []
        for d in args.data:
            p = os.path.join(d, "rfalse_train.csv")
            if os.path.exists(p):
                frames.append(pd.read_csv(p))
            else:
                print("[fit_models][WARN] no %s -- that data directory "
                      "contributes nothing to the deployed R_false head" % p,
                      flush=True)
        if frames:
            rf = pd.concat(frames, ignore_index=True)
            # Diagnostic that matters: H_dep is floored at 0, and a single
            # candidate edge moves a *global* biomarker very little, so most
            # targets are exactly 0 by construction (C1's own bridge events
            # behave the same way: ~50% non-zero, median ~2e-3 sigma).  If a
            # target is ~100% zero the head for it can only predict 0, and the
            # lambda (1 - p_e) R_false term is inert for that biomarker.
            from src.c1.btr import TARGET_PREFIX, parse_target

            prefix = TARGET_PREFIX[parse_target(args.rfalse_target)[0]]
            tcols = [c for c in rf.columns if c.startswith(prefix + "_")]
            if not tcols:
                print("[fit_models][WARN] rfalse_train.csv has no %s_* columns; "
                      "rebuild step B with --force (build_data now writes both "
                      "target families)" % prefix, flush=True)
            wrong = (rf[rf["is_true_repair"] <= 0] if "is_true_repair" in rf
                     else rf)
            nz = {c: float((pd.to_numeric(wrong[c], errors="coerce") > 0).mean())
                  for c in tcols}
            report["rfalse_target"] = args.rfalse_target
            report["rfalse_target_nonzero_frac"] = nz
            print("[fit_models] R_false target=%s non-zero fraction (wrong "
                  "connections only): %s"
                  % (args.rfalse_target,
                     json.dumps({k: round(v, 3) for k, v in nz.items()})),
                  flush=True)
            head = fit_deployed_false(
                rf, os.path.join(args.out, "btr_false_deployed.joblib"),
                name="false_deployed_seed%d" % int(args.seed),
                only_wrong=not args.rfalse_all_candidates,
                min_events=args.min_rfalse_rows, target=args.rfalse_target)
            if head is not None:
                report["rfalse_fitted"] = True
                report["rfalse"] = dict(head.meta, targets=list(head.targets),
                                        n_features=len(head.features))
                print("[fit_models] deployed R_false head: %d targets, %d rows "
                      "-> %s" % (len(head.targets), head.meta.get("n_rows", -1),
                                 os.path.join(args.out, "btr_false_deployed.joblib")),
                      flush=True)
            else:
                report["rfalse_note"] = ("too few usable rows; run_rigr must fall "
                                         "back to runs/btr/btr_false_all.joblib")
        else:
            report["rfalse_note"] = "no rfalse_train.csv found under any --data dir"

    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print("[fit_models] wrote %s" % os.path.join(args.out, "report.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
