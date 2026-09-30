"""Exp2: the candidate-level *natural intervention* benchmark (plan S4 item 8).

    python -m src.rigr.exp2_candidates --datasets drive,chasedb1,hrf,fives \
        --seed 0 --pred_root runs/seg --head_root runs/rigr_head \
        --model_root runs/rigr_models --out results/tab1_exp2.csv --workers 8

Exp1 (``src.c1.btr.run_exp1``) asks whether the BTR heads predict the harm of a
*controlled* perturbation the engine itself applied.  Exp2 asks the deployment
question: on the **real** predictions of the trained segmenter, for every
candidate connection RiGR proposes, does any of ``p_e``, the conventional
topology deltas, or the BTR utility predict the harm that **actually applying
that single edge** does to the biomarkers?

So, per candidate, this module really applies the edge -- one edge, on its own,
never a matching -- and measures

    dH_actual = macroMAE(B(M u T_e), B(GT)) - macroMAE(B(M), B(GT))

with the same ``src.eval.biomarker_eval`` call and the same single source of
sigma (``results/gateA_biomarker_scales.csv``) every other number in the paper
uses.  **Positive means harm** (the biomarker error grew), which is the sign
convention ``src.eval.tables`` documents for this file.

Output columns are exactly the ``results/tab1_exp2.csv`` schema of
``src.eval.tables``' module docstring::

    dataset image cand_id p_e dDice dclDice dBCS dBetti0 U_btr dH_actual

plus these extra diagnostic columns, which ``build_tab1`` ignores:

    seed subject_id n_candidates is_port d_over_rbar rad_bar path_len
    accepted_by_matching R_miss R_false C_geom macro_mae_before macro_mae_after
    before_dice before_cldice before_bcs before_beta0_err (and the after_*)

Cost.  The biomarker pipeline is the bottleneck (1-4 s per mask), so the work is
spread over a process pool with **one image per task** (``--workers``); inside a
task the candidates are evaluated serially, sharing that image's ``B(M)`` and
``B(GT)``.  ``--max_cand`` caps how many candidates per image are actually
applied (default 40, sampled with a fixed per-image seed, always keeping the
ones the matching accepted), and ``--bio`` defaults to ``skan``.  Workers run on
CPU: the micro head is small, and a pool of GPU workers would contend with the
training grid.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["run_image", "main"]

SCHEMA = ["dataset", "image", "cand_id", "p_e", "dDice", "dclDice", "dBCS",
          "dBetti0", "U_btr", "dH_actual"]

def _subsample_seed(dataset: str, image_key: str, seed: int) -> int:
    """Stable seed for the ``--max_cand`` subsample of one image.

    This used to be ``abs(hash((ds, key, seed)))``.  Python salts ``hash()`` of
    str/bytes per interpreter process (PYTHONHASHSEED is not pinned anywhere in
    this tree outside ``src/seg/train.py``), and exp2 evaluates its images in
    Pool workers -- each its own process -- so the "fixed per-image seed" the
    module docstring promises was in fact different on every run AND between
    workers of the same run.  Two runs of step H therefore kept different
    subsets of candidates: the 2026-09-15 table and the 2026-09-06 one share
    only 2620 of 4184 candidate keys (every shared row agrees to <= 4e-6, so
    only the sampling moved, not the measurement).  A digest is stable across
    processes, machines and Python versions.  DECISIONS.md 2026-09-16 00:05.
    """
    import hashlib

    payload = "%s|%s|%d" % (dataset, image_key, int(seed))
    d = hashlib.blake2b(payload.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(d, "big")


# per-process caches (a Pool worker handles many images)
_EVIDENCE: Dict[str, Any] = {}
_SCORER: Dict[str, Any] = {}
_BACKEND: Dict[str, Any] = {}


def _get_evidence(head_ckpt: Optional[str], gpu: int, evidence: str):
    key = "%s|%d|%s" % (head_ckpt, gpu, evidence)
    if key not in _EVIDENCE:
        from src.rigr.run_rigr import make_evidence_fn

        _EVIDENCE[key] = make_evidence_fn(head_ckpt, gpu, evidence)
    return _EVIDENCE[key]


def _get_scorer(path: Optional[str]):
    if not path:
        return None
    if path not in _SCORER:
        from src.rigr.scorer import PairScorer

        _SCORER[path] = PairScorer.load(path)
    return _SCORER[path]


def _get_backend(miss: Optional[str], false: Optional[str], runs_dir: Optional[str],
                 variant: str):
    key = "%s|%s|%s|%s" % (miss, false, runs_dir, variant)
    if key not in _BACKEND:
        from src.rigr.btr_backend import make_backend

        try:
            _BACKEND[key] = make_backend(miss, false, runs_dir, variant)
        except FileNotFoundError:
            _BACKEND[key] = None
    return _BACKEND[key]


def _topo_light(pred, gt, fov) -> Dict[str, float]:
    """Dice / clDice / BCS / beta0 error -- the four Tab.1 conventional deltas.

    ``evaluate_all`` also computes AUC, PR-AUC, Betti-1 matching and Junc-F1;
    at one call per candidate that is minutes per image, and Tab.1 needs only
    these four.
    """
    from src.topo.metrics import bcs, betti_error, cldice, dice

    be = betti_error(pred, gt, fov)
    return dict(dice=float(dice(pred, gt, fov)),
                cldice=float(cldice(pred, gt, fov)),
                bcs=float(bcs(pred, gt, fov)),
                beta0_err=float(be.get("beta0_err", np.nan)))


def run_image(job: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One test image: candidates, A*, p_e, U_btr, then apply every edge alone."""
    from src.eval.biomarker_eval import (biomarker_row, load_gt_scales,
                                         macro_mae, pipes_for)
    from src.eval.trr_fcr import rasterize_tube
    from src.rigr import utility as util_mod
    from src.rigr.run_rigr import auto_disc, load_case, prepare_image
    from src.topo import skeleton as sk

    ds = job["dataset"]
    rec = job["record"]
    evidence_fn, _ev_name = _get_evidence(job["head_ckpt"], job["gpu"], job["evidence"])
    scorer = _get_scorer(job["scorer"])
    backend = _get_backend(job.get("btr_miss"), job.get("btr_false"),
                           job.get("btr_runs_dir"), job.get("btr_variant", "all"))

    case = load_case(rec, job["pred_dir"], job.get("synthetic", False),
                     np.random.default_rng(job["seed"]), job.get("n_cuts", 15))
    disc = auto_disc(case)
    prep = prepare_image(case, evidence_fn, disc=disc,
                         astar_kwargs=job.get("astar_kwargs") or None)
    table, results = prep["table"], prep["results"]
    n = len(table)
    if n == 0:
        return []

    p = (scorer.predict_proba(prep["X"]) if scorer is not None
         else np.full(n, np.nan))
    util = util_mod.compute_utility(table, results, np.nan_to_num(p, nan=0.0),
                                    mode="risk", lam=job.get("lam", 1.0),
                                    eta=job.get("eta", 0.1),
                                    X=prep["X"], image=case["image"],
                                    mask=case["mask"], fov=case["fov"], disc=disc,
                                    risk_backend=backend,
                                    geom_scale=job.get("geom_scale"))
    # which candidates the deployed matching would have taken (diagnostic only)
    sel = util_mod.select_edges(table, results, util, labels=prep["cand"]["labels"],
                                theta_mask=prep["cand"]["theta_mask"],
                                radius=prep["cand"]["radius"])
    accepted = set(int(i) for i in sel["selected"])

    m = sk.as_bool(case["mask"])
    f = sk.fov_or_true(case["fov"], m.shape)
    gt = case["gt"]
    pipes = pipes_for(job["bio"])
    sigma = load_gt_scales(ds)

    base_topo = _topo_light(m, gt, f)
    bio_gt = biomarker_row(gt, f, case["image"], disc=disc, pipelines=job["bio"],
                           fd_rotations=job["fd_rotations"])
    bio_before = biomarker_row(m, f, case["image"], disc=disc, pipelines=job["bio"],
                              fd_rotations=job["fd_rotations"])
    mae_before = macro_mae(bio_before, bio_gt, sigma, pipes)

    rows_meta = table.to_dict("records")
    ok = [i for i, r in enumerate(results)
          if getattr(r, "ok", False) and len(getattr(r, "path_px", []))]
    max_cand = int(job.get("max_cand", 40) or 0)
    if max_cand and len(ok) > max_cand:
        rng = np.random.default_rng(_subsample_seed(ds, case["key"], job["seed"]))
        keep = [i for i in ok if i in accepted][:max_cand]
        rest = [i for i in ok if i not in keep]
        room = max_cand - len(keep)
        if room > 0 and rest:
            keep += list(rng.choice(rest, size=min(room, len(rest)), replace=False))
        ok = sorted(int(i) for i in keep)

    u_arr = util["U"].to_numpy(dtype=float)
    rm_arr = util["R_miss"].to_numpy(dtype=float)
    rf_arr = util["R_false"].to_numpy(dtype=float)
    cg_arr = util["C_geom"].to_numpy(dtype=float)

    out: List[Dict[str, Any]] = []
    for i in ok:
        row = rows_meta[i]
        res = results[i]
        tube = rasterize_tube(np.asarray(res.path_px, dtype=np.int64),
                              float(row["rad_bar"]), m.shape)
        m_plus = m | (tube & f)
        topo = _topo_light(m_plus, gt, f)
        bio_after = biomarker_row(m_plus, f, case["image"], disc=disc,
                                  pipelines=job["bio"],
                                  fd_rotations=job["fd_rotations"])
        mae_after = macro_mae(bio_after, bio_gt, sigma, pipes)
        out.append(dict(
            dataset=ds, image=case["key"], cand_id=int(row["cand_id"]),
            p_e=float(p[i]),
            dDice=topo["dice"] - base_topo["dice"],
            dclDice=topo["cldice"] - base_topo["cldice"],
            dBCS=topo["bcs"] - base_topo["bcs"],
            dBetti0=topo["beta0_err"] - base_topo["beta0_err"],
            U_btr=float(u_arr[i]),
            dH_actual=float(mae_after - mae_before),
            # ---- diagnostics (ignored by build_tab1) ----------------------
            seed=int(job["seed"]), subject_id=case["subject_id"],
            n_candidates=int(n), is_port=int(row.get("is_port", 0)),
            d_over_rbar=float(row["d_over_rbar"]), rad_bar=float(row["rad_bar"]),
            path_len=float(res.length),
            accepted_by_matching=int(i in accepted),
            R_miss=float(rm_arr[i]), R_false=float(rf_arr[i]),
            C_geom=float(cg_arr[i]),
            risk_backend=str(util.attrs.get("risk_backend", "?")),
            macro_mae_before=float(mae_before), macro_mae_after=float(mae_after),
            before_dice=base_topo["dice"], after_dice=topo["dice"],
            before_cldice=base_topo["cldice"], after_cldice=topo["cldice"],
            before_bcs=base_topo["bcs"], after_bcs=topo["bcs"],
            before_beta0_err=base_topo["beta0_err"],
            after_beta0_err=topo["beta0_err"],
            pred_source=case["source"],
        ))
    return out


def _limit_for(spec: Optional[str], dataset: str) -> Optional[int]:
    """``--limit`` accepts ``"60"`` (all datasets) or ``"fives=60,hrf=30"``.

    FIVES has 200 test images against DRIVE's 20, so one global cap would
    either truncate the small datasets or leave FIVES dominating the runtime;
    the per-dataset form is what ``src.pipeline.s4_all`` passes.
    """
    if not spec:
        return None
    spec = str(spec).strip()
    if "=" not in spec:
        return int(spec)
    for part in spec.split(","):
        if not part.strip():
            continue
        k, _, v = part.partition("=")
        if k.strip() == dataset:
            return int(v)
    return None


def _resolve_geom_scale(args, scorer: Optional[str]) -> Optional[float]:
    """m_R for this dataset: same resolution order as ``run_rigr`` --
    ``--geom_scale`` override, else ``--geom_scale_file``, else
    ``geom_scale.json`` next to ``--scorer``.  DECISIONS.md 2026-09-06 12:30:
    risk mode's ``C_geom`` is O(1) while the BTR harms are O(1e-3); without
    this scale ``U ~ -eta*C_geom`` and the run is not risk-guided at all.
    """
    if args.geom_scale is not None:
        return float(args.geom_scale)
    gsf = args.geom_scale_file
    if gsf is None and scorer:
        gsf = os.path.join(os.path.dirname(scorer), "geom_scale.json")
    if gsf and os.path.exists(gsf):
        with open(gsf, encoding="utf-8") as f:
            return float(json.load(f)["m_R"])
    return None


def _jobs(args) -> List[Dict[str, Any]]:
    from src.seg import data as segdata

    jobs = []
    for name in args.datasets.split(","):
        ds = segdata.canon(name.strip())
        if not ds:
            continue
        recs = segdata.get_records(ds)
        _tr, _va, te, _info = segdata.make_splits(recs)
        cap = _limit_for(args.limit, ds)
        if cap:
            te = te[: int(cap)]
        head = args.head_ckpt or os.path.join(args.head_root, ds,
                                              "seed%d" % args.seed, "best.pt")
        scorer = args.scorer or os.path.join(args.model_root, ds,
                                             "seed%d" % args.seed, "scorer.joblib")
        btr_false = args.btr_false or os.path.join(
            args.model_root, ds, "seed%d" % args.seed, "btr_false_deployed.joblib")
        pred_dir = args.pred_dir or os.path.join(args.pred_root, ds,
                                                 "seed%d" % args.seed, "pred")
        geom_scale = _resolve_geom_scale(args, scorer if os.path.exists(scorer) else None)
        if geom_scale is None:
            # exp2 evaluates every candidate in RISK mode, so a missing m_R is
            # not a degraded run, it is the 2026-09-06 12:30 defect back again
            # (U ~ -eta*C_geom).  Refuse, the way run_rigr does (exit code 2),
            # instead of printing and continuing.  DECISIONS.md 2026-09-16 00:05.
            print("[exp2][fatal] %s: risk mode needs m_R and no geom_scale.json "
                  "was found next to the scorer (%s).  Fit it with: "
                  "python -m src.rigr.geom_scale --data runs/rigr_data/%s "
                  "--out %s" % (ds, scorer, ds, os.path.dirname(scorer)),
                  flush=True)
            raise SystemExit(2)
        for rec in te:
            jobs.append(dict(
                dataset=ds, record=rec, seed=int(args.seed), gpu=int(args.gpu),
                pred_dir=pred_dir, evidence=args.evidence,
                head_ckpt=(head if os.path.exists(head) else None),
                scorer=(scorer if os.path.exists(scorer) else None),
                btr_miss=args.btr_miss,
                btr_false=(btr_false if os.path.exists(btr_false) else None),
                btr_runs_dir=args.btr_runs_dir, btr_variant=args.btr_variant,
                bio=args.bio, fd_rotations=int(args.fd_rotations),
                max_cand=int(args.max_cand), lam=float(args.lam),
                eta=float(args.eta), synthetic=bool(args.synthetic_pred),
                n_cuts=int(args.n_cuts), astar_kwargs=None,
                geom_scale=geom_scale))
    return jobs


def _safe_run(job):
    try:
        return run_image(job)
    except Exception as exc:  # noqa: BLE001 - one bad image must not kill the run
        import traceback

        key = os.path.basename(str(job["record"].get("image_path", "?")))
        print("[exp2][ERROR] %s/%s: %s: %s" % (job["dataset"], key,
                                               type(exc).__name__, exc), flush=True)
        traceback.print_exc()
        return []


def main(argv=None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(
        prog="python -m src.rigr.exp2_candidates",
        description="Exp2: candidate-level natural-intervention benchmark")
    ap.add_argument("--datasets", default="drive,chasedb1,hrf,fives")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pred_root", default=os.path.join("runs", "seg"))
    ap.add_argument("--pred_dir", default=None, help="override for one dataset")
    ap.add_argument("--head_root", default=os.path.join("runs", "rigr_head"))
    ap.add_argument("--head_ckpt", default=None)
    ap.add_argument("--model_root", default=os.path.join("runs", "rigr_models"))
    ap.add_argument("--scorer", default=None)
    ap.add_argument("--geom_scale_file", default=None,
                    help="geom_scale.json holding m_R.  Default: the file "
                         "next to --scorer (per dataset).  Risk mode requires "
                         "it -- see src/rigr/geom_scale.py.")
    ap.add_argument("--geom_scale", type=float, default=None,
                    help="override m_R directly for every dataset (sensitivity runs)")
    ap.add_argument("--evidence", default="auto", choices=["auto", "head", "frangi"])
    ap.add_argument("--btr_runs_dir", default=None)
    ap.add_argument("--btr_variant", default="all")
    ap.add_argument("--btr_miss", default=None)
    ap.add_argument("--btr_false", default=None)
    ap.add_argument("--bio", default="skan", choices=["skan", "pvbm", "both"])
    ap.add_argument("--fd_rotations", type=int, default=8)
    ap.add_argument("--max_cand", type=int, default=40,
                    help="candidates actually applied per image (0 = all)")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--limit", default=None,
                    help="test images per dataset: '60' for all, or "
                         "'fives=60,hrf=30' per dataset")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--gpu", type=int, default=-1,
                    help="-1 (default) keeps the workers on CPU")
    ap.add_argument("--synthetic_pred", action="store_true")
    ap.add_argument("--n_cuts", type=int, default=15)
    ap.add_argument("--out", default=os.path.join("results", "tab1_exp2.csv"))
    args = ap.parse_args(argv)

    jobs = _jobs(args)
    if not jobs:
        print("[exp2] no test images selected")
        return 2
    print("[exp2] %d images, %d workers, bio=%s, max_cand=%d"
          % (len(jobs), args.workers, args.bio, args.max_cand), flush=True)

    t0 = time.time()
    rows: List[Dict[str, Any]] = []
    if int(args.workers) <= 1:
        for k, j in enumerate(jobs):
            rows += _safe_run(j)
            print("[exp2] %d/%d  %d rows  %.1fs"
                  % (k + 1, len(jobs), len(rows), time.time() - t0), flush=True)
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(int(args.workers)) as pool:
            for k, r in enumerate(pool.imap_unordered(_safe_run, jobs)):
                rows += r
                print("[exp2] %d/%d  %d rows  %.1fs"
                      % (k + 1, len(jobs), len(rows), time.time() - t0), flush=True)

    if not rows:
        print("[exp2] no candidates were applied; nothing written")
        return 2
    df = pd.DataFrame(rows)
    cols = SCHEMA + [c for c in df.columns if c not in SCHEMA]
    df = df[cols]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    meta = dict(n_rows=int(len(df)), n_images=int(df["image"].nunique()),
                datasets=sorted(df["dataset"].unique().tolist()),
                seed=int(args.seed), bio=args.bio, max_cand=int(args.max_cand),
                risk_backend=sorted(set(df.get("risk_backend", ["?"]))),
                harmful_frac=float((df["dH_actual"] > 0).mean()),
                seconds=round(time.time() - t0, 1))
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[exp2] wrote %d rows -> %s\n%s"
          % (len(df), args.out, json.dumps(meta, indent=2)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
