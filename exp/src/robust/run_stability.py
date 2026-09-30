"""Fig.4A: within-image biomarker stability under acquisition perturbations.

Plan reference: exp/EXPERIMENT_PLAN.md S5 / proposal v3 section 4.3 -- "(A)
anatomy-preserving acquisition-perturbation biomarker CV (incl. worst-case
deviation)". For every test image this builds ``K=8`` deterministic
anatomy-preserving views (:mod:`src.robust.perturb_views`), segments each view
with the trained checkpoint (reusing :mod:`src.seg.infer`), optionally repairs
each predicted mask through a pluggable hook, computes the 4 primary
biomarkers (both pipelines) on every view's mask, and summarises the
within-image spread: coefficient of variation, MAD, worst-case deviation, and
a dataset-level ICC(2,1) across views per biomarker.

GPU note (authoring time): both project GPUs are busy training, so this
script is exercised here only via ``--help`` / a tiny CPU dry run; the CLI
below runs unchanged on CPU (``--gpu -1``) for a couple of images to smoke
test it, and on GPU at full scale once a training slot frees up.

Outputs (under ``--out-dir``, default ``exp/results``)
-------------------------------------------------------
``fig4a_stability_<dataset>_<repair>_per_view.csv``
    one row per (image, view): applied ops, the 8 primary biomarker columns,
    predicted foreground fraction, inference time.
``fig4a_stability_<dataset>_<repair>_per_image.csv``
    one row per (image, biomarker): mean, sd, CV, MAD, worst-case deviation
    (raw and MAD-standardised against the single source of sigma,
    ``exp/results/gateA_biomarker_scales.csv``, when present -- see
    ``src.eval.biomarker_eval``'s docstring; nothing here re-estimates sigma
    in-memory from this run's own images).
``fig4a_stability_<dataset>_<repair>_icc.csv``
    one row per biomarker: dataset-level ICC(2,1) across the K views
    (images = subjects, views = raters).

CLI
---
    python -m src.robust.run_stability --dataset drive \\
        --ckpt runs/seg/drive/seed0/best.pt --gpu 0 --repair none
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from typing import Callable, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")


# --------------------------------------------------------------------------- #
# repair hook
# --------------------------------------------------------------------------- #
def apply_repair(mask: np.ndarray, fov: np.ndarray, image: np.ndarray, *,
                  mode: str, cmd_template: Optional[str] = None,
                  fn_spec: Optional[str] = None,
                  prob: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
    """Optionally repair a binary mask. Returns ``(mask_out, applied)``.

    ``prob`` is this view's probability map.  A repair hook that wants it
    (``src.rigr.repair_hook.repair`` does: the RiGR micro head is conditioned on
    ``[I, P]``) may declare a ``prob`` keyword; hooks with the plain
    ``fn(mask, fov, image)`` signature keep working unchanged.

    ``mode == "none"``
        pass-through.
    ``mode == "rigr_cmd"``
        either
        * ``fn_spec = "module.submodule:function"`` -- imported and called as
          ``function(mask, fov, image) -> mask`` (preferred: no subprocess,
          no round trip through disk); or
        * ``cmd_template`` -- a shell command with ``{input}`` / ``{output}``
          placeholders; ``mask`` is written to ``{input}`` as an 8-bit PNG,
          the command is run, and ``{output}`` is read back as the repaired
          mask.

        This is a **hook**: RiGR (the repair method under evaluation, see
        ``exp/src/rigr/``) is developed by another workstream, so as of
        writing there is nothing to call yet. If neither ``fn_spec`` nor
        ``cmd_template`` is given, the mask passes through unchanged and a
        one-time warning is printed -- ``run_stability`` never silently
        pretends a repair happened.
    """
    if mode == "none":
        return mask, False
    if mode != "rigr_cmd":
        raise ValueError(f"unknown --repair mode {mode!r}")

    if fn_spec:
        mod_name, _, fn_name = fn_spec.partition(":")
        if not fn_name:
            raise ValueError(f"--repair-fn must be 'module:function', got {fn_spec!r}")
        import importlib

        fn: Callable = getattr(importlib.import_module(mod_name), fn_name)
        try:
            out = fn(mask, fov, image, prob=prob)
        except TypeError:
            out = fn(mask, fov, image)
        return np.asarray(out).astype(bool), True

    if cmd_template:
        import subprocess
        import tempfile

        import cv2

        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "in.png")
            outp = os.path.join(td, "out.png")
            cv2.imwrite(inp, (mask.astype(np.uint8) * 255))
            cmd = cmd_template.format(input=inp, output=outp)
            subprocess.run(cmd, shell=True, check=True)
            if not os.path.exists(outp):
                raise RuntimeError(f"repair command produced no output: {cmd}")
            rep = cv2.imread(outp, cv2.IMREAD_GRAYSCALE)
            return (rep > 127), True

    if not getattr(apply_repair, "_warned", False):
        print("[run_stability] --repair rigr_cmd given but neither --repair-fn nor "
              "--repair-cmd was set; masks pass through unrepaired (hook not wired yet)")
        apply_repair._warned = True  # type: ignore[attr-defined]
    return mask, False


# --------------------------------------------------------------------------- #
# per-view inference (mirrors src.seg.infer's normalisation/blend exactly)
# --------------------------------------------------------------------------- #
def infer_view(model, view_img: np.ndarray, view_fov: np.ndarray, *, patch: int,
                longest: Optional[int], device, threshold: float, stride: int,
                amp: bool, sw_batch: int):
    import cv2
    import torch

    from src.seg import data as segdata
    from src.seg.infer import sliding_window_predict, to_native

    native_hw = view_img.shape[:2]
    img, fov = view_img, view_fov
    if longest is not None and max(native_hw) != longest:
        img = segdata._resize_longest(img, longest, cv2.INTER_AREA)
        fov = (segdata._resize_longest(fov.astype(np.uint8) * 255, longest,
                                        cv2.INTER_LINEAR) > 127).astype(np.uint8)

    m = fov > 0
    if m.sum() < 16:
        m = np.ones_like(fov, bool)
    pix = img[m].astype(np.float32)
    mean = pix.mean(axis=0).astype(np.float32)
    std = np.maximum(pix.std(axis=0), 1e-3).astype(np.float32)

    x = torch.from_numpy(segdata.normalize(img, mean, std))
    prob_r = sliding_window_predict(model, x, patch, device, stride=stride, amp=amp,
                                     batch_size=sw_batch)
    prob = to_native(prob_r, native_hw)
    fov_native = view_fov if fov.shape == native_hw else (
        cv2.resize(fov.astype(np.uint8) * 255, (native_hw[1], native_hw[0]),
                   interpolation=cv2.INTER_NEAREST) > 127
    )
    mask = (prob >= threshold) & (fov_native > 0)
    return mask, fov_native, prob


# --------------------------------------------------------------------------- #
# stats: CV / MAD / worst-case / ICC(2,1)
# --------------------------------------------------------------------------- #
def icc21(matrix: np.ndarray) -> float:
    """Two-way random, single-measures, absolute-agreement ICC (Shrout & Fleiss 1979).

    ``matrix``: (n_subjects, k_raters), no missing values.
    """
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = x.mean()
    row_mean = x.mean(axis=1)
    col_mean = x.mean(axis=0)
    sst = float(((x - grand) ** 2).sum())
    ssr = float(k * ((row_mean - grand) ** 2).sum())
    ssc = float(n * ((col_mean - grand) ** 2).sum())
    sse = sst - ssr - ssc
    df_r, df_c, df_e = n - 1, k - 1, (n - 1) * (k - 1)
    if df_r <= 0 or df_c <= 0 or df_e <= 0:
        return float("nan")
    msr, msc, mse = ssr / df_r, ssc / df_c, sse / df_e
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if denom == 0:
        return float("nan")
    return float((msr - mse) / denom)


def _per_image_stats(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    if v.size == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, cv=np.nan, mad=np.nan, worst=np.nan)
    mean = float(v.mean())
    sd = float(v.std())
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    worst = float(np.max(np.abs(v - med)))
    cv = sd / abs(mean) if abs(mean) > 1e-9 else float("nan")
    return dict(n=int(v.size), mean=mean, sd=sd, cv=cv, mad=mad, worst=worst)


# schema item 3: sigma is read once from results/gateA_biomarker_scales.csv
# (see src.eval.biomarker_eval.load_gt_scales) -- this module no longer
# re-estimates it in-memory from exp/results/gateA_biomarkers_gt.csv.


# --------------------------------------------------------------------------- #
def _seed_for(seed: int, image_id: str) -> int:
    h = zlib.crc32(image_id.encode("utf-8")) & 0xFFFFFFFF
    return (int(seed) * 1_000_003 + h) & 0x7FFFFFFF


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fig.4A: within-image biomarker stability under "
                                              "anatomy-preserving acquisition perturbations")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0, help="-1 forces CPU")
    ap.add_argument("--repair", choices=["none", "rigr_cmd"], default="none")
    ap.add_argument("--repair-cmd", default=None,
                     help="shell template with {input}/{output} PNG paths, used when --repair rigr_cmd")
    ap.add_argument("--repair-fn", default=None,
                     help="'module:function(mask, fov, image) -> mask', used when --repair rigr_cmd "
                          "(preferred over --repair-cmd: no disk round trip)")
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap #images (smoke tests)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stride-div", type=int, default=2)
    ap.add_argument("--sw-batch", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    import torch

    from src.seg import data as segdata
    from src.seg.infer import load_model
    from src.bio.biomarkers import compute_all, primary_columns
    from src.robust.perturb_views import make_views

    device = torch.device("cpu" if args.gpu < 0 or not torch.cuda.is_available()
                           else f"cuda:{args.gpu}")
    cfg = segdata.dataset_cfg(args.dataset)
    patch, longest = int(cfg["patch"]), cfg["resize_longest"]
    stride = max(1, patch // max(1, args.stride_div))

    split_seed = args.split_seed if args.split_seed is not None else segdata.DEFAULT_SPLIT_SEED
    recs = segdata.get_records(args.dataset)
    tr, va, te, _info = segdata.make_splits(recs, split_seed)
    subset = {"test": te, "val": va, "train": tr, "all": recs}[args.split]
    if args.limit:
        subset = subset[: args.limit]
    if not subset:
        print(f"[run_stability] split '{args.split}' empty for {args.dataset}")
        return 2

    model, ck = load_model(args.ckpt, device)
    cols = primary_columns(zone=False)
    ds_key = segdata.canon(args.dataset)
    from src.eval.biomarker_eval import load_gt_scales

    sigma = load_gt_scales(ds_key)

    per_view_rows: List[dict] = []
    t_all = time.time()
    for ri, rec in enumerate(subset):
        image = segdata._imread_color(rec["image_path"])
        native_hw = image.shape[:2]
        from src.seg.infer import load_native_fov

        fov = load_native_fov(rec, native_hw)
        views, manifest = make_views(image, fov, K=args.K, seed=_seed_for(args.seed, rec["image_id"]))

        for view, man in zip(views, manifest):
            t0 = time.time()
            try:
                mask, vfov, vprob = infer_view(model, view["image"], view["fov"], patch=patch,
                                         longest=longest, device=device, threshold=args.threshold,
                                         stride=stride, amp=not args.no_amp, sw_batch=args.sw_batch)
                mask, repaired = apply_repair(mask, vfov, view["image"], mode=args.repair,
                                               cmd_template=args.repair_cmd, fn_spec=args.repair_fn,
                                               prob=vprob)
                bio = compute_all(mask, vfov, image=None, disc=None)
                row = dict(dataset=ds_key, image_id=rec["image_id"], subject_id=rec.get("subject_id", ""),
                           view_id=man["view_id"], ops=man["ops"], params_json=man["params_json"],
                           repair=args.repair, repaired=int(repaired), error="")
                for c in cols:
                    row[c] = bio.get(c, np.nan)
                row["pred_fg_frac"] = float(mask.sum()) / max(1.0, float(vfov.sum()))
                row["seconds"] = round(time.time() - t0, 3)
            except Exception as exc:  # noqa: BLE001
                row = dict(dataset=ds_key, image_id=rec["image_id"], subject_id=rec.get("subject_id", ""),
                           view_id=man["view_id"], ops=man["ops"], params_json=man["params_json"],
                           repair=args.repair, repaired=0, error=f"{type(exc).__name__}: {exc}")
            per_view_rows.append(row)
        print(f"[{ri + 1}/{len(subset)}] {rec['image_id']}: {args.K} views done "
              f"({time.time() - t_all:.1f}s elapsed)", flush=True)

    import pandas as pd

    per_view = pd.DataFrame(per_view_rows)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{ds_key}_{args.repair}"
    pv_path = os.path.join(args.out_dir, f"fig4a_stability_{tag}_per_view.csv")
    per_view.to_csv(pv_path, index=False)
    print(f"wrote {len(per_view)} rows -> {pv_path}")

    # ---- per-image summary -------------------------------------------------
    ok = per_view[per_view["error"] == ""] if "error" in per_view.columns else per_view
    pi_rows = []
    for (img_id,), g in ok.groupby(["image_id"]):
        for c in cols:
            if c not in g.columns:
                continue
            stats = _per_image_stats(pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float))
            sig = sigma.get(c, np.nan)
            pi_rows.append(dict(
                dataset=ds_key, repair=args.repair, image_id=img_id, biomarker=c,
                K=int(g.shape[0]), **stats,
                sd_z=stats["sd"] / sig if np.isfinite(sig) and sig > 0 else np.nan,
                mad_z=stats["mad"] / sig if np.isfinite(sig) and sig > 0 else np.nan,
                worst_z=stats["worst"] / sig if np.isfinite(sig) and sig > 0 else np.nan,
            ))
    pi_df = pd.DataFrame(pi_rows)
    pi_path = os.path.join(args.out_dir, f"fig4a_stability_{tag}_per_image.csv")
    pi_df.to_csv(pi_path, index=False)
    print(f"wrote per-image summary -> {pi_path}")
    # schema item 5: src.robust.fig4 looks for this per-image summary under
    # results/ specifically; when --out-dir points elsewhere (e.g. a
    # runs/robust/<name>/ detail directory), also drop a copy there so Fig.4
    # can find it regardless of where the per-view detail was written.
    if os.path.abspath(args.out_dir) != os.path.abspath(RESULTS_DIR):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fallback_path = os.path.join(RESULTS_DIR, f"fig4a_stability_{tag}_per_image.csv")
        pi_df.to_csv(fallback_path, index=False)
        print(f"[run_stability] also wrote the Fig.4 summary -> {fallback_path}")

    # ---- dataset-level ICC(2,1) per biomarker ------------------------------
    icc_rows = []
    for c in cols:
        if c not in ok.columns:
            continue
        piv = ok.pivot_table(index="image_id", columns="view_id", values=c, aggfunc="first")
        piv = piv.dropna(axis=0, how="any")
        n_subj, k_raters = piv.shape
        val = icc21(piv.to_numpy(dtype=float)) if n_subj >= 2 and k_raters >= 2 else float("nan")
        icc_rows.append(dict(dataset=ds_key, repair=args.repair, biomarker=c,
                              n_subjects=n_subj, K=k_raters, icc21=val))
    icc_df = pd.DataFrame(icc_rows)
    icc_path = os.path.join(args.out_dir, f"fig4a_stability_{tag}_icc.csv")
    icc_df.to_csv(icc_path, index=False)
    print(f"wrote ICC(2,1) summary -> {icc_path}")
    print(icc_df)

    meta = dict(dataset=ds_key, ckpt=os.path.abspath(args.ckpt), repair=args.repair,
                repair_cmd=args.repair_cmd, repair_fn=args.repair_fn, split=args.split,
                K=args.K, seed=args.seed, n_images=len(subset), device=str(device),
                gate_a_sigma_used=bool(sigma), total_seconds=round(time.time() - t_all, 2))
    with open(os.path.join(args.out_dir, f"fig4a_stability_{tag}_meta.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
