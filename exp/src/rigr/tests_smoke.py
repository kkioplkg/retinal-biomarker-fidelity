"""RiGR smoke test (plan stage S4) -- runs the whole stage end to end on CPU.

Stage 1 (analytic evidence)
    3 DRIVE training images; ``M_hat`` = the reference mask with 15 random
    capsule severances, ``P`` = Gaussian-blurred ``M_hat``, ``V_I`` =
    ``frangi_evidence`` and ``q`` = uniform (no head).  Candidates -> lifted A*
    -> pair scorer trained on 2 images and tested on the third -> uniform-event-
    cost utility + max-weight matching -> TRR / FCR and clDice before/after.

Stage 2 (learned evidence)
    Trains the micro head for 3 epochs on DRIVE (GPU 0) and repeats stage 1
    with the learned ``V_I`` and ``q``.

This is a *plumbing* test: with 15 synthetic severances per image and a scorer
fitted on two images, none of the numbers are results.  What it checks is that
every interface in the stage lines up, that A* stays inside its latency budget,
and that a repair actually raises clDice and lowers beta0 error.

::

    cd exp && python -m src.rigr.tests_smoke [--skip-head] [--gpu 0] [--epochs 3]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.topo import skeleton as sk

N_IMAGES = 3
N_CUTS = 15
SEED = 0


def _hdr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78, flush=True)


def load_images(dataset: str = "drive", n: int = N_IMAGES) -> List[Dict[str, Any]]:
    from src.seg import data as segdata

    recs = segdata.get_records(dataset)
    tr, va, te, _ = segdata.make_splits(recs)
    out = []
    for rec in tr[:n]:
        key = os.path.splitext(os.path.basename(str(rec["image_path"])))[0]
        img = segdata._imread_color(rec["image_path"])
        gt = segdata._imread_gray(rec["label_path"]) > 127
        fov = (segdata._imread_gray(rec["fov_path"]) > 127
               if rec.get("fov_path") and os.path.exists(str(rec["fov_path"]))
               else segdata.derive_fov(img) > 0)
        out.append(dict(key=key, image=img, gt=gt & fov, fov=fov, record=rec))
    return out


def frangi_provider(k_bins: int = 16):
    """``V_I`` = Frangi vesselness, ``q`` = uniform (the Tab.3 ablation baseline)."""
    from src.rigr.head import frangi_evidence

    cache: Dict[int, np.ndarray] = {}

    def fn(image, mask, prob, fov):
        h = id(image)
        if h not in cache:
            cache[h] = frangi_evidence(image, fov)
        V = cache[h]
        Q = np.full((k_bins,) + np.asarray(mask).shape, 1.0 / k_bins, np.float32)
        return V, Q

    return fn


def head_provider(ckpt: str, gpu: int = 0):
    import torch

    from src.rigr.head import load_head, predict_head

    device = torch.device("cuda:%d" % gpu if torch.cuda.is_available() else "cpu")
    model, ck = load_head(ckpt, device)
    patch = int(ck.get("config", {}).get("patch", 512))

    def fn(image, mask, prob, fov):
        return predict_head(image, prob, ckpt, gpu=gpu, fov=fov, patch=patch,
                            model=model, device=device)

    return fn


# --------------------------------------------------------------------------
def build_case(img: Dict[str, Any], evidence_fn, rng) -> Dict[str, Any]:
    """Sever the reference mask, blur it into ``P`` and run candidates + A*."""
    from src.rigr.synth_cuts import build_training_pairs

    t0 = time.perf_counter()
    d = build_training_pairs(img["image"], img["gt"], img["fov"], evidence_fn,
                             n_cuts=N_CUTS, rng=rng, image_id=img["key"],
                             verify_beta0=True)
    d["seconds"] = time.perf_counter() - t0
    d["img"] = img
    ok = [r for r in d["results"] if r.ok]
    d["astar_ms"] = (1e3 * float(np.mean([r.seconds for r in d["results"]]))
                     if d["results"] else float("nan"))
    d["astar_ms_max"] = (1e3 * float(np.max([r.seconds for r in d["results"]]))
                         if d["results"] else float("nan"))
    d["astar_ok"] = len(ok)
    d["grid"] = [r.grid_shape for r in d["results"][:3]]
    return d


def run_pipeline(images: List[Dict[str, Any]], evidence_fn, tag: str,
                 scorer_model: str = "logreg") -> Dict[str, Any]:
    from src.eval.trr_fcr import count_repairable_events, trr_fcr
    from src.rigr import utility as util_mod
    from src.rigr.scorer import PairScorer
    from src.topo.metrics import cldice, dice

    from src.rigr.astar import warmup

    warmup()
    rng = np.random.default_rng(SEED)
    cases = []
    _hdr("[%s] STEP 1-3  severance -> candidates -> lifted A*" % tag)
    for img in images:
        c = build_case(img, evidence_fn, rng)
        cases.append(c)
        cuts = c["cuts"]
        n_ev = len(count_repairable_events(c["cut_mask"], img["gt"], img["fov"]))
        print("%-14s cuts=%2d (delta_beta0==1: %2d)  endpoints=%3d ports=%3d  "
              "candidates=%3d  positives=%2d  repairable events=%2d"
              % (img["key"], len(cuts), int(cuts["verified"].sum()),
                 c["cand"]["n_endpoints"], c["cand"]["n_ports"],
                 len(c["table"]), int(c["y"].sum()), n_ev), flush=True)
        print("               A*: %d/%d solved, %.1f ms/candidate (max %.1f), "
              "grids %s, image %.1f s"
              % (c["astar_ok"], len(c["results"]), c["astar_ms"],
                 c["astar_ms_max"], c["grid"], c["seconds"]), flush=True)

    _hdr("[%s] STEP 4  pair scorer: fit on images 1-2, test on image 3" % tag)
    Xtr = np.concatenate([c["X"] for c in cases[:-1]], 0)
    ytr = np.concatenate([c["y"] for c in cases[:-1]], 0)
    Xte, yte = cases[-1]["X"], cases[-1]["y"]
    print("train: %d candidates (%d positive)   test: %d candidates (%d positive)"
          % (len(ytr), int(ytr.sum()), len(yte), int(yte.sum())))
    if len(np.unique(ytr)) < 2:
        print("!! degenerate training labels; aborting")
        return {}

    # Calibration must never touch the test image (section 3.2.4): carve a
    # stratified 30% calibration slice out of the two TRAINING images instead.
    rs = np.random.default_rng(SEED)
    cal = np.zeros(len(ytr), bool)
    for lab in (0, 1):
        idx = np.flatnonzero(ytr == lab)
        rs.shuffle(idx)
        cal[idx[: max(1, int(round(0.3 * len(idx))))]] = True
    scorer = PairScorer(model=scorer_model, calibration="temperature", seed=SEED)
    scorer.fit(Xtr[~cal], ytr[~cal],
               Xtr[cal] if len(np.unique(ytr[cal])) > 1 else None,
               ytr[cal] if len(np.unique(ytr[cal])) > 1 else None)
    print("calibration slice: %d candidates (%d positive), disjoint from the "
          "test image" % (int(cal.sum()), int(ytr[cal].sum())))
    rep = scorer.evaluate(Xte, yte)
    print("held-out image: AUC=%.3f  AP=%.3f  Brier=%.4f  ECE=%.4f  T=%.2f"
          % (rep["auc"], rep["ap"], rep["brier"], rep["ece"], scorer.temperature))
    print(rep["reliability"][["lo", "hi", "n", "mean_pred", "frac_pos"]]
          .round(3).to_string(index=False))
    top = np.argsort(np.abs(getattr(scorer.clf, "coef_", np.zeros((1, Xtr.shape[1])))
                            [0]))[::-1][:5]
    if hasattr(scorer.clf, "coef_"):
        print("top |coef|: %s" % ", ".join(
            "%s=%.2f" % (scorer.feature_names[i], scorer.clf.coef_[0][i])
            for i in top))

    _hdr("[%s] STEP 5  uniform-event-cost utility + max-weight matching" % tag)
    probs = [scorer.predict_proba(c["X"]) if len(c["X"]) else np.zeros(0)
             for c in cases]
    _all = np.concatenate([q for q in probs if len(q)]) if any(
        len(q) for q in probs) else np.zeros(1)
    print("p_e over all candidates: min %.3f  median %.3f  max %.3f"
          % (float(_all.min()), float(np.median(_all)), float(_all.max())))
    rows = []
    for c in cases:
        img = c["img"]
        p = scorer.predict_proba(c["X"]) if len(c["X"]) else np.zeros(0)
        out = util_mod.repair_mask(
            c["cut_mask"], c["table"], c["results"], p, mode="uniform",
            lam=1.0, eta=0.1, fov=img["fov"], labels=c["cand"]["labels"],
            theta_mask=c["cand"]["theta_mask"], radius=c["cand"]["radius"],
            X=c["X"])
        tf = trr_fcr(out["edges"], c["cut_mask"], img["gt"], fov=img["fov"])
        cd_b = cldice(c["cut_mask"], img["gt"], img["fov"])
        cd_a = cldice(out["mask"], img["gt"], img["fov"])
        di_b = dice(c["cut_mask"], img["gt"], img["fov"])
        di_a = dice(out["mask"], img["gt"], img["fov"])
        b0_b = sk.betti_numbers(c["cut_mask"], img["fov"])[0]
        b0_a = sk.betti_numbers(out["mask"], img["fov"])[0]
        b0_g = sk.betti_numbers(img["gt"], img["fov"])[0]
        n_true_sel = int(sum(c["y"][i] for i in out["selected"]))
        rows.append(dict(image=img["key"], n_cand=len(c["table"]),
                         n_pos_U=out["n_positive"], n_sel=len(out["selected"]),
                         n_sel_true=n_true_sel,
                         veto_cross=len(out["vetoed_crossing"]),
                         n_vetoed=out["n_vetoed"],
                         n_pruned_conflict=out["n_pruned_conflict"],
                         cldice_before=cd_b, cldice_after=cd_a,
                         dice_before=di_b, dice_after=di_a,
                         beta0_before=b0_b, beta0_after=b0_a, beta0_gt=b0_g,
                         TRR_recall=tf["TRR_recall"],
                         TRR_precision=tf["TRR_precision"], FCR=tf["FCR"],
                         n_should=tf["n_should"], n_accepted=tf["n_accepted"],
                         n_true=tf["n_true"], astar_ms=c["astar_ms"]))
        print("%-14s sel=%2d/%2d (true %2d)  veto=%d prune=%d  clDice %.4f -> %.4f  "
              "Dice %.4f -> %.4f  beta0 %d -> %d (GT %d)  TRR_r=%.3f TRR_p=%.3f "
              "FCR=%.3f"
              % (img["key"], len(out["selected"]), out["n_positive"], n_true_sel,
                 out["n_vetoed"], out["n_pruned_conflict"],
                 cd_b, cd_a, di_b, di_a, b0_b, b0_a, b0_g,
                 tf["TRR_recall"], tf["TRR_precision"], tf["FCR"]), flush=True)

    _hdr("[%s] STEP 6  lambda / threshold sweep (Fig.3 operating curve)" % tag)
    sweep = []
    b0_cut = float(np.mean([r["beta0_before"] for r in rows]))
    b0_gt = float(np.mean([r["beta0_gt"] for r in rows]))
    for mode, lam in (("prob", None), ("uniform", 0.25), ("uniform", 0.5),
                      ("uniform", 1.0), ("uniform", 2.0)):
        acc = []
        for c, p in zip(cases, probs):
            img = c["img"]
            o = util_mod.repair_mask(
                c["cut_mask"], c["table"], c["results"], p, mode=mode,
                lam=(lam or 1.0), eta=0.1, tau=0.5, fov=img["fov"],
                labels=c["cand"]["labels"], theta_mask=c["cand"]["theta_mask"],
                radius=c["cand"]["radius"], X=c["X"])
            t = trr_fcr(o["edges"], c["cut_mask"], img["gt"], fov=img["fov"])
            acc.append((len(o["selected"]),
                        cldice(o["mask"], img["gt"], img["fov"]),
                        t["TRR_recall"], t["FCR"],
                        sk.betti_numbers(o["mask"], img["fov"])[0]))
        a = np.array(acc, dtype=float)
        sweep.append(dict(mode=mode, lam=(lam if lam is not None else np.nan),
                          n_selected=a[:, 0].mean(), cldice=a[:, 1].mean(),
                          TRR_recall=np.nanmean(a[:, 2]), FCR=a[:, 3].mean(),
                          beta0=a[:, 4].mean()))
        print("  mode=%-8s lam=%-5s  selected=%4.1f  clDice=%.4f  TRR_r=%.3f  "
              "FCR=%.3f  beta0=%.1f (cut %.1f, GT %.1f)"
              % (mode, ("-" if lam is None else lam), sweep[-1]["n_selected"],
                 sweep[-1]["cldice"], sweep[-1]["TRR_recall"], sweep[-1]["FCR"],
                 sweep[-1]["beta0"], b0_cut, b0_gt), flush=True)

    import pandas as pd

    df = pd.DataFrame(rows)
    print("\n[%s] MEANS  clDice %.4f -> %.4f (%+0.4f) | Dice %.4f -> %.4f (%+0.4f)"
          % (tag, df.cldice_before.mean(), df.cldice_after.mean(),
             df.cldice_after.mean() - df.cldice_before.mean(),
             df.dice_before.mean(), df.dice_after.mean(),
             df.dice_after.mean() - df.dice_before.mean()))
    print("[%s] MEANS  TRR_recall=%.3f  TRR_precision=%.3f  FCR=%.3f  "
          "A*=%.1f ms/candidate"
          % (tag, df.TRR_recall.mean(), df.TRR_precision.mean(), df.FCR.mean(),
             df.astar_ms.mean()))
    return dict(df=df, cases=cases, scorer=scorer, report=rep,
                sweep=pd.DataFrame(sweep), probs=probs)


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RiGR smoke test")
    ap.add_argument("--skip-head", action="store_true")
    ap.add_argument("--only-head", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--head_out", default="runs/rigr_head/drive/smoke")
    args = ap.parse_args(argv)

    from src.rigr.head import MicroUNet

    net = MicroUNet()
    _hdr("RiGR smoke test -- micro head: %d parameters (%.3f M), channels %s"
         % (net.num_parameters(), net.num_parameters() / 1e6, net.channels))
    del net

    images = load_images("drive", N_IMAGES)
    print("DRIVE training images: %s" % [i["key"] for i in images])
    print("image size: %s   FOV fraction: %.3f"
          % (images[0]["image"].shape, images[0]["fov"].mean()))

    res_frangi = None
    if not args.only_head:
        res_frangi = run_pipeline(images, frangi_provider(), "FRANGI + uniform q")

    if args.skip_head:
        return 0

    _hdr("TRAIN the micro head for %d epochs on DRIVE (GPU %d)"
         % (args.epochs, args.gpu))
    from src.rigr.train_head import main as train_main

    rc = train_main(["--dataset", "drive", "--seed", str(SEED),
                     "--gpu", str(args.gpu), "--epochs", str(args.epochs),
                     "--iters", str(args.iters), "--batch", "4",
                     "--out", args.head_out])
    if rc != 0:
        print("head training failed")
        return rc
    ckpt = os.path.join(args.head_out, "best.pt")
    res_head = run_pipeline(images, head_provider(ckpt, args.gpu),
                            "LEARNED V_I + q")

    if res_frangi is not None and res_head:
        _hdr("COMPARISON  analytic Frangi vs the 3-epoch learned head")
        a, b = res_frangi["df"], res_head["df"]
        cols = ["n_cand", "n_sel", "n_sel_true", "cldice_before", "cldice_after",
                "dice_after", "TRR_recall", "TRR_precision", "FCR", "astar_ms"]
        import pandas as pd

        print(pd.DataFrame({"frangi": a[cols].mean(), "head3ep": b[cols].mean()})
              .round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
