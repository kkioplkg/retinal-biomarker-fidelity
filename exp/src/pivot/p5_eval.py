"""Probe P5 evaluation -- fine-tuned vs original segmenter.

Stage 1 (GPU):   sliding-window inference of one checkpoint on the test split,
                 written with exactly the ``src/seg/infer.py`` conventions
                 (native-resolution ``prob/*.npy`` + ``mask/*.png``), plus the
                 pixel metrics (Dice/F1, clDice) of ``src/topo/metrics.py``.
Stage 2 (CPU):   ``src.bio.biomarkers.compute_all`` (``fd_rotations = 5``, optic
                 disc detected from the fundus image alone) on those masks,
                 cached per image under ``results/pivot/cache/fd5/<ds>/p5_<tag>/``.
Stage 3:         the comparison table -- signed bias, standardised MAE,
                 reliability ``r(B, B_GT)`` and the P3 downstream macro-AUC with
                 a paired image bootstrap.

CLI
---
    python -m src.pivot.p5_eval infer --dataset hrf --gpu 1 \
        --ckpt runs/pivot/ft_hrf_seed0/last.pt --tag ft_measure
    python -m src.pivot.p5_eval bio    --dataset hrf --tag ft_measure --procs 6
    python -m src.pivot.p5_eval report --dataset hrf --tags ft_measure,ft_cfloss
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from src.pivot.common import (CACHE_DIR, PIVOT_DIR, PRIMARY_COLS, binarize,
                              json_dump, load_fov, read_gray, records,
                              sigma_table)

FD_ROT = 5
SEED = 0


def out_root(dataset: str, tag: str, root: str = None) -> str:
    """Directory holding ``prob/``, ``mask/``, ``manifest.csv``, ``pixel_metrics.csv``.

    ``root`` (E2) overrides the flat P5 layout so a run can live under
    ``runs/pivot/<ds>/seed<k>/<config>/pred``; the per-image biomarker cache is
    still keyed by ``tag``, which E2 makes globally unique.
    """
    if root:
        return root
    return os.path.join("runs", "pivot", f"pred_{dataset}_{tag}")


def cache_path(dataset: str, tag: str, image_id: str) -> str:
    return os.path.join(CACHE_DIR, f"fd{FD_ROT}", dataset, f"p5_{tag}",
                        image_id + ".json")


# --------------------------------------------------------------------------
def cmd_infer(args) -> int:
    import torch

    from src.seg import data as segdata
    from src.seg.infer import load_model, predict_records

    ds = segdata.canon(args.dataset)
    cfg = segdata.dataset_cfg(ds)
    patch, longest = int(cfg["patch"]), cfg["resize_longest"]
    # zero-shot cross-dataset evaluation applies the *source* model's
    # resolution convention (the scale it was trained at); predictions are
    # still written back at the target's native resolution.
    if getattr(args, "infer_patch", 0):
        patch = int(args.infer_patch)
    if getattr(args, "infer_longest", None) is not None:
        longest = int(args.infer_longest) or None
    recs = segdata.get_records(ds)
    tr, va, te, _ = segdata.make_splits(recs, segdata.DEFAULT_SPLIT_SEED)
    subset = {"test": te, "train": tr, "val": va}[args.split]
    cache = os.path.join("runs", "_cache", f"{ds}_{longest}") if longest else None

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.ckpt, device)

    od = out_root(ds, args.tag, getattr(args, "root", None))
    os.makedirs(od, exist_ok=True)
    rows, meta = predict_records(model, subset, patch, longest, device, od,
                                 dataset_name=ds, seed_tag=args.tag,
                                 split_label=args.split, threshold=0.5,
                                 stride=max(1, patch // 2), amp=True,
                                 sw_batch=4, save_prob=True, cache_dir=cache,
                                 manifest_name="manifest.csv")
    meta = {"dataset": ds, "ckpt": os.path.abspath(args.ckpt), "tag": args.tag,
            "ckpt_epoch": ck.get("epoch"), "ckpt_val_dice": ck.get("val_dice"),
            "split": args.split, **meta}

    # ---- pixel metrics on the native-resolution masks ----
    from src.topo.metrics import cldice as cldice_fn
    from src.topo.metrics import dice as dice_fn
    from src.seg.evaluate import read_fov, read_gt

    px = []
    for r in rows:
        native_hw = (int(r["native_h"]), int(r["native_w"]))
        gt = read_gt(r, native_hw)
        fov = read_fov(r, native_hw)
        pred = (read_gray(r["mask_path"]) > 127).astype(np.uint8)
        px.append({"image": r["image"],
                   "dice": float(dice_fn(pred, gt, fov)),
                   "cldice": float(cldice_fn(pred, gt, fov)),
                   "pred_fg_frac": r["pred_fg_frac"]})
        print("  px {} dice={:.4f} cldice={:.4f}".format(
            px[-1]["image"], px[-1]["dice"], px[-1]["cldice"]), flush=True)
    pxdf = pd.DataFrame(px)
    pxdf.to_csv(os.path.join(od, "pixel_metrics.csv"), index=False)
    meta["mean_dice"] = float(pxdf["dice"].mean())
    meta["mean_cldice"] = float(pxdf["cldice"].mean())
    with open(os.path.join(od, "infer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return 0


# --------------------------------------------------------------------------
def _bio_one(a):
    dataset, rec, tag, mask_path = a
    cp = cache_path(dataset, tag, rec["image_id"])
    if os.path.exists(cp):
        return (rec["image_id"], 0)
    from src.bio.biomarkers import compute_all
    from src.bio.disc import locate_optic_disc
    from src.data.datasets import read_image

    t0 = time.perf_counter()
    image = read_image(rec["image_path"])
    native_hw = image.shape[:2]
    fov = load_fov(rec, native_hw)
    try:
        disc = locate_optic_disc(image, fov, vessel_mask=None)
    except Exception:                                            # noqa: BLE001
        disc = None
    mask = binarize(read_gray(mask_path)) & (fov > 0)
    row = dict(dataset=dataset, image_id=rec["image_id"], split=rec["split"],
               subject_id=rec.get("subject_id"), disease=rec.get("disease"),
               source="p5_" + tag, fd_rotations=FD_ROT, fg_px=int(mask.sum()))
    bio = compute_all(mask, fov, image=None, disc=disc, fd_rotations=FD_ROT)
    bio.pop("pvbm_version", None)
    row.update({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                for k, v in bio.items()})
    json_dump(row, cp)
    return (rec["image_id"], round(time.perf_counter() - t0, 1))


def cmd_bio(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    ds = args.dataset
    all_recs = records(ds)
    recs = {r["image_id"]: r for r in all_recs}
    by_path = {os.path.normcase(os.path.abspath(str(r["image_path"]))): r
               for r in all_recs}
    root = out_root(ds, args.tag, getattr(args, "root", None))
    man = pd.read_csv(os.path.join(root, "manifest.csv"))
    tasks = []
    for r in man.itertuples(index=False):
        # The manifest stem drops the FIVES split prefix, and FIVES numbers its
        # train and test images independently -- ``100_D`` exists in BOTH, so a
        # suffix match is ambiguous for ~half the test split and silently
        # attached the wrong image/FOV/optic disc to the predicted mask
        # (fixed 2026-09-17).  Resolve by the manifest's own absolute
        # ``image_path`` first; only accept a suffix match when it is unique.
        rec = None
        ip = str(getattr(r, "image_path", "") or "")
        if ip and ip.lower() != "nan":
            rec = by_path.get(os.path.normcase(os.path.abspath(ip)))
        if rec is None:
            rec = recs.get(str(r.image))
        if rec is None:
            rec = recs.get(str(getattr(r, "subject_id", "") or ""))
        if rec is None:
            cand = [v for k, v in recs.items()
                    if k == str(r.image) or k.endswith("_" + str(r.image))]
            if len(cand) > 1:
                raise KeyError(
                    "ambiguous manifest key %r in %s: matches %s. Refusing to "
                    "guess -- the manifest must carry image_path or a full "
                    "image_id." % (str(r.image), root,
                                   [c["image_id"] for c in cand]))
            rec = cand[0] if cand else None
        if rec is None:
            raise KeyError(str(r.image))
        tasks.append((ds, rec, args.tag, str(r.mask_path)))
    print(f"[plan] {len(tasks)} images for tag={args.tag}", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(_bio_one, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            iid, dt = fu.result()
            print(f"[{i+1:3d}/{len(tasks)}] {iid} {dt}s "
                  f"({time.perf_counter()-t0:.0f}s elapsed)", flush=True)
    df = load_tag(ds, args.tag)
    df.to_csv(os.path.join(root, "bio.csv"), index=False)
    print(f"[bio] wrote {os.path.join(root, 'bio.csv')} ({len(df)} rows)", flush=True)
    return 0


# --------------------------------------------------------------------------
def load_tag(dataset: str, tag: str) -> pd.DataFrame:
    root = os.path.join(CACHE_DIR, f"fd{FD_ROT}", dataset, f"p5_{tag}")
    rows = []
    for fn in sorted(os.listdir(root)):
        if fn.endswith(".json"):
            with open(os.path.join(root, fn), encoding="utf-8") as fh:
                rows.append(json.load(fh))
    return pd.DataFrame(rows)


def cmd_gtbio(args) -> int:
    """Biomarkers of the *reference* masks of a test split, tag ``gt_e2``.

    ``results/pivot/bio_master.csv`` covers DRIVE / CHASE_DB1 / HRF / FIVES but
    not STARE, and E2's zero-shot arm needs a GT column for both external sets.
    Identical estimator path to :func:`cmd_bio` (``compute_all(fd_rotations=5)``
    with the optic disc detected from the fundus image alone).
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from src.seg import data as segdata

    ds = segdata.canon(args.dataset)
    recs = segdata.get_records(ds)
    _, _, te, _ = segdata.make_splits(recs, segdata.DEFAULT_SPLIT_SEED)
    byid = {r["image_id"]: r for r in records(ds)}
    tasks = []
    for r in te:
        rec = byid[r["image_id"]]
        tasks.append((ds, rec, args.tag, str(rec["label_path"])))
    print(f"[plan] {len(tasks)} GT images for {ds} tag={args.tag}", flush=True)
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(_bio_one, t) for t in tasks]
        for i, fu in enumerate(as_completed(futs)):
            iid, dt = fu.result()
            print(f"[{i+1:3d}/{len(tasks)}] {iid} {dt}s", flush=True)
    out = os.path.join(PIVOT_DIR, f"e2_gt_{ds}.csv")
    load_tag(ds, args.tag).to_csv(out, index=False)
    print(f"[gtbio] wrote {out}", flush=True)
    return 0


def cmd_report(args) -> int:
    from scipy import stats

    from src.pivot.p3_downstream import boot_ci, cv_proba, macro_auc

    ds = args.dataset
    cols = list(PRIMARY_COLS)
    sig = sigma_table()[ds]
    master = pd.read_csv(os.path.join(PIVOT_DIR, "bio_master.csv"))
    m = master[master["dataset"] == ds]
    gt = m[m["source"] == "gt"][["image_id", "disease", "split"] + cols]
    gt = gt.rename(columns={c: c + "__gt" for c in cols})
    base = m[m["source"] == "pred_test"][["image_id"] + cols]
    base = base.rename(columns={c: c + "__orig" for c in cols})

    d = gt.merge(base, on="image_id", how="inner")
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    for t in tags:
        tt = load_tag(ds, t)[["image_id"] + cols].rename(
            columns={c: c + "__" + t for c in cols})
        d = d.merge(tt, on="image_id", how="inner")
    print(f"[{ds}] {len(d)} images compared across {['orig'] + tags}")

    sources = ["orig"] + tags
    rows = []
    for c in cols:
        y = d[c + "__gt"].to_numpy(dtype=float)
        s = float(sig.get(c, np.nan))
        for src in sources:
            x = d[c + "__" + src].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            rows.append(dict(
                dataset=ds, biomarker=c, source=src, n=int(ok.sum()), sigma=s,
                bias_sigma=float(np.mean(x[ok] - y[ok]) / s),
                mae_sigma=float(np.mean(np.abs(x[ok] - y[ok])) / s),
                r_pearson=float(stats.pearsonr(x[ok], y[ok])[0]) if ok.sum() > 3 else np.nan,
                r_spearman=float(stats.spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 3 else np.nan,
            ))
    bio = pd.DataFrame(rows)

    # ---- paired image bootstrap of delta-r vs the original model ----
    rng = np.random.RandomState(SEED)
    n = len(d)
    idxs = [rng.randint(0, n, n) for _ in range(1000)]
    drows = []
    for c in cols:
        y = d[c + "__gt"].to_numpy(dtype=float)
        x0 = d[c + "__orig"].to_numpy(dtype=float)
        for t in tags:
            x1 = d[c + "__" + t].to_numpy(dtype=float)
            dd = []
            for ix in idxs:
                a, b, g = x1[ix], x0[ix], y[ix]
                ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(g)
                if ok.sum() < 5:
                    continue
                dd.append(stats.pearsonr(a[ok], g[ok])[0] - stats.pearsonr(b[ok], g[ok])[0])
            drows.append(dict(dataset=ds, biomarker=c, source=t,
                              d_r_pearson=float(np.mean(dd)),
                              lo=float(np.percentile(dd, 2.5)),
                              hi=float(np.percentile(dd, 97.5))))
    dr = pd.DataFrame(drows)

    # ---- pixel metrics ----
    pxrows = []
    for src in ["orig"] + tags:
        f = os.path.join(out_root(ds, src, (args.roots or {}).get(src)), "pixel_metrics.csv")
        if not os.path.exists(f):
            pxrows.append(dict(source=src, mean_dice=np.nan, mean_cldice=np.nan))
            continue
        px = pd.read_csv(f)
        pxrows.append(dict(source=src, n=len(px), mean_dice=float(px["dice"].mean()),
                           mean_cldice=float(px["cldice"].mean())))
    pxdf = pd.DataFrame(pxrows)

    # ---- downstream ----
    dd = d[d["disease"].notna()].reset_index(drop=True)
    yv = dd["disease"].to_numpy()
    classes = sorted(pd.unique(yv))
    arows = []
    have_labels = len(dd) >= 24 and len(classes) >= 2 and \
        min(int((yv == c).sum()) for c in classes) >= 8
    if not have_labels:
        print(f"[{ds}] downstream skipped: no usable disease labels "
              f"({len(dd)} labelled images)")
    for kind in (("logreg", "gbdt") if have_labels else ()):
        probas = {}
        for src in ["gt"] + sources:
            X = np.nan_to_num(dd[[c + "__" + src for c in cols]].to_numpy(dtype=float),
                              nan=0.0, posinf=0.0, neginf=0.0)
            probas[src] = cv_proba(X, yv, kind, classes)
        ci, dci = boot_ci(yv, probas, classes)
        # paired bootstrap of (source - orig), the comparison P5 actually asks
        rng2 = np.random.RandomState(SEED)
        nn = len(yv)
        vs_orig: Dict[str, list] = {k: [] for k in probas}
        for _ in range(1000):
            ix = rng2.randint(0, nn, nn)
            if len(np.unique(yv[ix])) < len(classes):
                continue
            a0 = macro_auc(yv[ix], probas["orig"][ix], classes)
            for k in probas:
                vs_orig[k].append(macro_auc(yv[ix], probas[k][ix], classes) - a0)
        for src, p in probas.items():
            v = vs_orig[src]
            arows.append(dict(
                minus_orig=float(np.mean(v)) if v else np.nan,
                minus_orig_lo=float(np.percentile(v, 2.5)) if v else np.nan,
                minus_orig_hi=float(np.percentile(v, 97.5)) if v else np.nan,
                **dict(dataset=ds, clf=kind, source=src, n=len(dd),
                              macro_auc=macro_auc(yv, p, classes),
                              ci_lo=ci.get(src, (np.nan,) * 2)[0],
                              ci_hi=ci.get(src, (np.nan,) * 2)[1],
                              gt_minus_this=macro_auc(yv, probas["gt"], classes)
                              - macro_auc(yv, p, classes),
                              diff_ci_lo=dci.get(src, (np.nan,) * 2)[0],
                              diff_ci_hi=dci.get(src, (np.nan,) * 2)[1],
                              class_counts=";".join("%s=%d" % (c, int((yv == c).sum()))
                                                    for c in classes))))
    au = pd.DataFrame(arows)
    if len(au):
        au = au[["dataset", "clf", "source", "n", "macro_auc", "ci_lo", "ci_hi",
                 "gt_minus_this", "diff_ci_lo", "diff_ci_hi",
                 "minus_orig", "minus_orig_lo", "minus_orig_hi", "class_counts"]]

    os.makedirs(PIVOT_DIR, exist_ok=True)
    bio.to_csv(os.path.join(PIVOT_DIR, f"p5_biomarkers_{ds}.csv"), index=False)
    dr.to_csv(os.path.join(PIVOT_DIR, f"p5_delta_r_{ds}.csv"), index=False)
    pxdf.to_csv(os.path.join(PIVOT_DIR, f"p5_pixel_{ds}.csv"), index=False)
    au.to_csv(os.path.join(PIVOT_DIR, f"p5_downstream_{ds}.csv"), index=False)
    with pd.option_context("display.width", 240, "display.max_rows", 300):
        print("\n=== P5 biomarker bias / MAE / reliability ===")
        print(bio.round(4).to_string(index=False))
        print("\n=== P5 delta reliability vs original (paired image bootstrap) ===")
        print(dr.round(4).to_string(index=False))
        print("\n=== P5 pixel metrics ===")
        print(pxdf.round(4).to_string(index=False))
        print("\n=== P5 downstream macro-AUC ===")
        print(au.round(4).to_string(index=False))
    return 0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("infer")
    a.add_argument("--dataset", default="hrf"); a.add_argument("--ckpt", required=True)
    a.add_argument("--tag", required=True); a.add_argument("--gpu", type=int, default=1)
    a.add_argument("--split", default="test")
    a.add_argument("--infer-longest", type=int, default=None,
                   help="override resize_longest (0 = native) -- zero-shot "
                        "cross-dataset inference at the source model's scale")
    a.add_argument("--infer-patch", type=int, default=0,
                   help="override the sliding-window patch size")
    a.add_argument("--root", default=None,
                   help="write prob/, mask/, manifest.csv here instead of the "
                        "flat runs/pivot/pred_<ds>_<tag> layout (E2)")
    a.set_defaults(fn=cmd_infer)
    b = sub.add_parser("bio")
    b.add_argument("--dataset", default="hrf"); b.add_argument("--tag", required=True)
    b.add_argument("--procs", type=int, default=6)
    b.add_argument("--root", default=None, help="see `infer --root`")
    b.set_defaults(fn=cmd_bio)
    gb = sub.add_parser("gtbio")
    gb.add_argument("--dataset", required=True)
    gb.add_argument("--tag", default="gt_e2")
    gb.add_argument("--procs", type=int, default=8)
    gb.set_defaults(fn=cmd_gtbio)
    c = sub.add_parser("report")
    c.add_argument("--dataset", default="hrf"); c.add_argument("--tags", required=True)
    c.set_defaults(fn=cmd_report, roots=None)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
