"""Per-stage latency of the RiGR repair pipeline, for the paper's budget table.

``run_rigr`` already records four of the numbers we want, per image, in
``per_image.csv``:

===========================  ============================================
``t_evidence_s``             head inference (on whatever device the run used)
``t_candidates_s``           candidate generation
``t_astar_s`` / ``..._ms``   A* over all candidates / per candidate
``t_total_s``               end-to-end repair, EXCLUDING biomarker measurement
                             (it is closed before ``evaluate_repair`` and the
                             ``biomarker_row`` calls -- see run_rigr.py:721-733)
===========================  ============================================

What it does *not* separate is the part after A*: candidate features, the pair
scorer, the utility and the matching all disappear into ``t_total_s``.  And it
only ever times the head on the device the run used.  This module fills both
gaps by driving the same internals directly:

* ``t_head_gpu`` / ``t_head_cpu`` -- the micro head on each device, same
  sliding-window settings, so the budget table can state the CPU fallback cost;
* ``t_features``   -- ``candidate_features``;
* ``t_scoring``    -- ``PairScorer.predict_proba`` (pair scoring);
* ``t_utility``    -- ``compute_utility`` (includes the BTR heads in risk mode);
* ``t_matching``   -- ``select_edges`` (greedy one-edge-per-endpoint arbitration
  plus the corridor-overlap pruning);
* ``t_paint``      -- ``apply_edges``;
* ``t_e2e_norepair_bio`` -- the whole thing exactly as ``run_rigr`` times it
  (``auto_disc`` + prepare + predict_proba + ``repair_mask``), i.e. the
  end-to-end repair latency with no biomarker work.

Every stage is run ``--repeats`` times per image and the MEDIAN is reported.
The first repeat of the first image also warms the CUDA context and the lazy
imports, so it is discarded when ``--warmup`` is set (default: on).

The load on the machine matters more than anything else here: run this on an
otherwise idle node, or the numbers measure contention, not the pipeline.

Usage::

    python -m src.eval.latency_bench --datasets drive,hrf --seed 0 \\
        --n_images 5 --repeats 3 --gpu 0 \\
        --out results/latency.csv --md results/latency.md
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from typing import Any, Dict, List, Optional

import numpy as np


def _median(xs: List[float]) -> float:
    return float(np.median(np.asarray(xs, dtype=float))) if xs else float("nan")


def _hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = dict(
        node=platform.node(), platform=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
    )
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_MiB"] = int(
                torch.cuda.get_device_properties(0).total_memory / 2 ** 20)
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:  # noqa: BLE001
        pass
    return info


def _sync(device_is_cuda: bool) -> None:
    """CUDA is asynchronous -- without this the GPU timings are launch times."""
    if device_is_cuda:
        import torch

        torch.cuda.synchronize()


def bench_image(case: Dict[str, Any], *, head_ckpt: str, scorer, gpu: int,
                mode: str, lam: float, eta: float, tau: float,
                geom_scale: Optional[float], risk_backend,
                repeats: int) -> List[Dict[str, Any]]:
    """One row per repeat for one image."""
    import torch

    from src.rigr import astar as astar_mod
    from src.rigr import candidates as cand_mod
    from src.rigr import utility as util_mod
    from src.rigr.head import load_head, predict_head
    from src.rigr.run_rigr import auto_disc, prepare_image
    from src.rigr.scorer import candidate_features

    has_cuda = bool(torch.cuda.is_available())
    dev_gpu = torch.device("cuda:%d" % int(gpu)) if has_cuda else None
    dev_cpu = torch.device("cpu")

    model_gpu, ck = (load_head(head_ckpt, dev_gpu) if dev_gpu is not None
                     else (None, {}))
    model_cpu, ck_cpu = load_head(head_ckpt, dev_cpu)
    patch = int((ck or ck_cpu).get("config", {}).get("patch", 512))

    rows: List[Dict[str, Any]] = []
    for r in range(repeats):
        rec: Dict[str, Any] = dict(image=case["key"], repeat=r)

        # ---- head inference, GPU then CPU -------------------------------
        if model_gpu is not None:
            _sync(True)
            t0 = time.perf_counter()
            V, Q = predict_head(case["image"], case["prob"], head_ckpt,
                                gpu=gpu, fov=case["fov"], patch=patch,
                                model=model_gpu, device=dev_gpu)
            _sync(True)
            rec["t_head_gpu_s"] = time.perf_counter() - t0
        else:
            rec["t_head_gpu_s"] = float("nan")

        t0 = time.perf_counter()
        V_cpu, Q_cpu = predict_head(case["image"], case["prob"], head_ckpt,
                                    gpu=-1, fov=case["fov"], patch=patch,
                                    model=model_cpu, device=dev_cpu)
        rec["t_head_cpu_s"] = time.perf_counter() - t0
        if model_gpu is None:
            V, Q = V_cpu, Q_cpu

        # ---- candidate generation --------------------------------------
        t0 = time.perf_counter()
        out = cand_mod.generate_candidates(case["mask"], case["prob"],
                                           case["fov"])
        rec["t_candidates_s"] = time.perf_counter() - t0
        table = out["table"]
        rec["n_candidates"] = int(len(table))
        rec["n_endpoints"] = int(out["n_endpoints"])
        rec["n_ports"] = int(out["n_ports"])

        # ---- A* ---------------------------------------------------------
        disc = auto_disc(case)
        t0 = time.perf_counter()
        results = []
        for row in table.to_dict("records"):
            gp = (out["port_pixels"].get(int(row["node_j"]))
                  if int(row["is_port"]) else None)
            results.append(astar_mod.solve_candidate(row, V, Q, goal_pixels=gp))
        rec["t_astar_s"] = time.perf_counter() - t0
        rec["t_astar_per_cand_ms"] = 1e3 * rec["t_astar_s"] / max(len(table), 1)

        # ---- candidate features ----------------------------------------
        disc_xy = (float(disc.cx), float(disc.cy)) if disc is not None else None
        t0 = time.perf_counter()
        X = candidate_features(table, results, case["image"], case["mask"],
                               case["prob"], case["fov"], disc_xy=disc_xy)
        rec["t_features_s"] = time.perf_counter() - t0

        # ---- pair scoring ----------------------------------------------
        t0 = time.perf_counter()
        p = (scorer.predict_proba(X) if (scorer is not None and len(table))
             else np.zeros(len(table)))
        rec["t_scoring_s"] = time.perf_counter() - t0

        # ---- utility ----------------------------------------------------
        t0 = time.perf_counter()
        util = util_mod.compute_utility(
            table, results, np.nan_to_num(p, nan=0.0), mode=mode, lam=lam,
            eta=eta, X=X, image=case["image"], mask=case["mask"],
            fov=case["fov"], disc=disc, risk_backend=risk_backend,
            geom_scale=geom_scale)
        rec["t_utility_s"] = time.perf_counter() - t0

        # ---- matching (greedy arbitration + corridor pruning) -----------
        t0 = time.perf_counter()
        sel = util_mod.select_edges(table, results, util,
                                    labels=out["labels"],
                                    theta_mask=out["theta_mask"],
                                    radius=out["radius"])
        rec["t_matching_s"] = time.perf_counter() - t0
        rec["n_selected"] = int(len(sel["selected"]))

        # ---- painting ---------------------------------------------------
        t0 = time.perf_counter()
        util_mod.apply_edges(case["mask"], table, results, sel["selected"],
                             fov=case["fov"])
        rec["t_paint_s"] = time.perf_counter() - t0

        # ---- end-to-end, exactly as run_rigr times t_total_s -------------
        _sync(model_gpu is not None)
        t0 = time.perf_counter()
        d2 = auto_disc(case)
        prep = prepare_image(case, lambda im, m, pr, fv: predict_head(
            im, pr, head_ckpt, gpu=gpu, fov=fv, patch=patch,
            model=(model_gpu if model_gpu is not None else model_cpu),
            device=(dev_gpu if model_gpu is not None else dev_cpu)), disc=d2)
        p2 = (scorer.predict_proba(prep["X"]) if (scorer is not None
                                                  and len(prep["table"]))
              else np.zeros(len(prep["table"])))
        util_mod.repair_mask(
            case["mask"], prep["table"], prep["results"], p2, mode=mode,
            lam=lam, eta=eta, tau=tau, fov=case["fov"],
            labels=prep["cand"]["labels"],
            theta_mask=prep["cand"]["theta_mask"],
            radius=prep["cand"]["radius"], X=prep["X"], image=case["image"],
            disc=d2, risk_backend=risk_backend, geom_scale=geom_scale)
        _sync(model_gpu is not None)
        rec["t_e2e_excl_biomarkers_s"] = time.perf_counter() - t0

        rows.append(rec)
    return rows


def main(argv=None) -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(
        prog="python -m src.eval.latency_bench",
        description="per-stage RiGR latency for the budget table")
    ap.add_argument("--datasets", default="drive,hrf")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_images", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mode", default="risk", choices=["prob", "uniform", "risk"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--pred_root", default=os.path.join("runs", "seg"))
    ap.add_argument("--head_root", default=os.path.join("runs", "rigr_head"))
    ap.add_argument("--model_root", default=os.path.join("runs", "rigr_models"))
    ap.add_argument("--btr_runs_dir", default=os.path.join("runs", "btr",
                                                           "habs_trainonly"))
    ap.add_argument("--btr_variant", default="all")
    ap.add_argument("--no_warmup", action="store_true",
                    help="keep the first (cold) repeat of the first image")
    ap.add_argument("--out", default=os.path.join("results", "latency.csv"))
    ap.add_argument("--md", default=os.path.join("results", "latency.md"))
    a = ap.parse_args(argv)

    from src.rigr import utility as util_mod
    from src.rigr.run_rigr import load_case
    from src.rigr.scorer import PairScorer
    from src.seg import data as segdata

    hw = _hardware()
    print("[latency] hardware: %s" % json.dumps(hw, ensure_ascii=False),
          flush=True)

    all_rows: List[Dict[str, Any]] = []
    for name in a.datasets.split(","):
        ds = segdata.canon(name.strip())
        if not ds:
            continue
        mdir = os.path.join(a.model_root, ds, "seed%d" % a.seed)
        head = os.path.join(a.head_root, ds, "seed%d" % a.seed, "best.pt")
        pred = os.path.join(a.pred_root, ds, "seed%d" % a.seed, "pred")
        scorer = (PairScorer.load(os.path.join(mdir, "scorer.joblib"))
                  if os.path.exists(os.path.join(mdir, "scorer.joblib"))
                  else None)

        geom_scale = None
        gsf = os.path.join(mdir, "geom_scale.json")
        if os.path.exists(gsf):
            with open(gsf, encoding="utf-8") as f:
                geom_scale = float(json.load(f)["m_R"])
        if a.mode == "risk" and geom_scale is None:
            raise SystemExit("[latency][fatal] %s: risk mode needs %s" % (ds, gsf))

        risk_backend = None
        if a.mode == "risk":
            from src.rigr.btr_backend import make_backend

            bf = os.path.join(mdir, "btr_false_deployed.joblib")
            risk_backend = make_backend(None, bf if os.path.exists(bf) else None,
                                        a.btr_runs_dir, a.btr_variant)

        recs = segdata.get_records(ds)
        _tr, _va, te, _info = segdata.make_splits(recs)
        te = te[: int(a.n_images)]
        print("[latency] %s: %d images, head=%s" % (ds, len(te), head), flush=True)

        rng = np.random.default_rng(a.seed)
        for rec in te:
            case = load_case(rec, pred, False, rng, 15)
            t0 = time.perf_counter()
            rows = bench_image(case, head_ckpt=head, scorer=scorer, gpu=a.gpu,
                               mode=a.mode, lam=a.lam, eta=a.eta, tau=a.tau,
                               geom_scale=geom_scale, risk_backend=risk_backend,
                               repeats=a.repeats)
            for r in rows:
                r["dataset"] = ds
                r["H"], r["W"] = case["mask"].shape[:2]
            all_rows += rows
            print("  %-10s %-12s %d repeats in %.1fs  (e2e median %.2fs)"
                  % (ds, case["key"], a.repeats, time.perf_counter() - t0,
                     _median([r["t_e2e_excl_biomarkers_s"] for r in rows])),
                  flush=True)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("[latency][fatal] nothing measured")
    front = ["dataset", "image", "H", "W", "repeat", "n_endpoints", "n_ports",
             "n_candidates", "n_selected"]
    cols = front + [c for c in df.columns if c not in front]
    df = df[cols]
    if not a.no_warmup and len(df) > 1:
        # drop the very first repeat of the very first image (cold CUDA context,
        # lazy imports, first numba/skan JIT) -- it is not a steady-state cost
        df = df.iloc[1:].reset_index(drop=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print("[latency] wrote %s (%d rows)" % (a.out, len(df)), flush=True)

    stages = [c for c in df.columns if c.startswith("t_")]
    lines = ["# RiGR per-stage latency", "",
             "Median over %d repeats x %d images per dataset, measured with "
             "`src/eval/latency_bench.py`.  `t_e2e_excl_biomarkers_s` is the "
             "same quantity `run_rigr` writes as `t_total_s`: it stops before "
             "`evaluate_repair` and any biomarker measurement." % (a.repeats,
                                                                  a.n_images),
             "", "## Hardware", "",
             "```", json.dumps(hw, indent=2, ensure_ascii=False), "```", ""]
    for ds, g in df.groupby("dataset"):
        lines += ["## %s (%d images, %dx%d)"
                  % (ds, g["image"].nunique(), int(g["H"].iloc[0]),
                     int(g["W"].iloc[0])), "",
                  "| stage | median s | median ms/candidate |",
                  "|---|---|---|"]
        ncand = float(np.median(g["n_candidates"]))
        for s in stages:
            if s.endswith("_ms"):
                continue
            med = _median(list(g[s]))
            lines.append("| `%s` | %.3f | %.3f |"
                         % (s, med, 1e3 * med / max(ncand, 1.0)))
        lines += ["",
                  "- candidates/image (median): **%.0f**" % ncand,
                  "- selected/image (median): **%.0f**"
                  % float(np.median(g["n_selected"])),
                  "- A* per candidate (median): **%.3f ms**"
                  % _median(list(g["t_astar_per_cand_ms"])), ""]
    with open(a.md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[latency] wrote %s" % a.md, flush=True)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
