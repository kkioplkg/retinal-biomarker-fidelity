"""Deterministic sliding-window full-image inference.

Identical settings for every method (per exp/EXPERIMENT_PLAN.md S2):
    window = the training patch size, stride = window // 2,
    Gaussian-weighted blending, test-time flips OFF, threshold 0.5 inside FOV.

Outputs, for each image of the chosen split:
    <out>/prob/<key>.npy   float16 probability map, NATIVE resolution
    <out>/mask/<key>.png   uint8 {0,255} binary mask, NATIVE resolution
    <out>/manifest.csv     one row per image

For HRF / FIVES the network runs on the 1536-longest-side image and the
probability map is resized back to native resolution bilinearly *before*
thresholding.

CLI:
    python -m src.seg.infer --dataset drive --ckpt runs/seg/drive/seed0/best.pt \
        --gpu 0 --out runs/seg/drive/seed0/pred
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from src.seg import data as segdata
from src.seg.unet import UNet

cv2.setNumThreads(0)


# --------------------------------------------------------------------------
def gaussian_window(size: int, sigma_scale: float = 0.125,
                    dtype=np.float32) -> np.ndarray:
    """nnU-Net style Gaussian importance map, peak normalised to 1."""
    sigma = size * sigma_scale
    c = (size - 1) / 2.0
    ax = np.arange(size, dtype=np.float64) - c
    g1 = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    g = np.outer(g1, g1)
    g = g / g.max()
    g = np.maximum(g, g.max() * 1e-3)  # avoid zeros at the corners
    return g.astype(dtype)


def _tile_starts(length: int, window: int, stride: int):
    if length <= window:
        return [0]
    n = int(np.ceil((length - window) / float(stride))) + 1
    if n == 1:
        return [0]
    step = (length - window) / float(n - 1)
    return [int(round(i * step)) for i in range(n)]


@torch.no_grad()
def sliding_window_predict(
    model: torch.nn.Module,
    image: torch.Tensor,
    patch: int,
    device: torch.device,
    stride: Optional[int] = None,
    amp: bool = True,
    batch_size: int = 4,
) -> np.ndarray:
    """Return a float32 HxW probability map for a normalised CHW image tensor."""
    model.eval()
    if image.dim() == 4:
        image = image[0]
    c, h, w = image.shape
    stride = patch // 2 if stride is None else int(stride)

    ph, pw = max(patch, h), max(patch, w)
    if ph != h or pw != w:
        pad = torch.zeros((c, ph, pw), dtype=image.dtype)
        pad[:, :h, :w] = image
        image = pad

    ys = _tile_starts(ph, patch, stride)
    xs = _tile_starts(pw, patch, stride)
    gw = torch.from_numpy(gaussian_window(patch)).to(device)

    acc = torch.zeros((ph, pw), dtype=torch.float32, device=device)
    wsum = torch.zeros((ph, pw), dtype=torch.float32, device=device)
    image = image.to(device, non_blocking=True)

    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        tiles = torch.stack([image[:, y:y + patch, x:x + patch] for y, x in chunk], 0)
        with torch.autocast("cuda", enabled=(amp and device.type == "cuda")):
            logits = model(tiles)
        if not torch.is_tensor(logits):
            logits = logits[0]
        probs = torch.sigmoid(logits.float())[:, 0]
        for j, (y, x) in enumerate(chunk):
            acc[y:y + patch, x:x + patch] += probs[j] * gw
            wsum[y:y + patch, x:x + patch] += gw

    out = (acc / wsum.clamp_min(1e-8))[:h, :w]
    return out.detach().cpu().numpy().astype(np.float32)


def to_native(prob: np.ndarray, native_hw: Tuple[int, int]) -> np.ndarray:
    """Bilinear resize of the probability map back to native resolution."""
    if prob.shape[:2] == tuple(native_hw):
        return prob
    return cv2.resize(prob, (native_hw[1], native_hw[0]), interpolation=cv2.INTER_LINEAR)


def load_native_fov(rec: dict, native_hw: Tuple[int, int]) -> np.ndarray:
    p = rec.get("fov_path")
    if p and os.path.exists(str(p)):
        f = (segdata._imread_gray(str(p)) > 127).astype(np.uint8)
    else:
        f = segdata.derive_fov(segdata._imread_color(str(rec["image_path"])))
    if f.shape[:2] != tuple(native_hw):
        f = (cv2.resize(f * 255, (native_hw[1], native_hw[0]),
                        interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)
    return f


# --------------------------------------------------------------------------
def load_model(ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    model = UNet(
        in_channels=cfg.get("in_channels", 3),
        out_channels=cfg.get("out_channels", 1),
        base_channels=cfg.get("base_channels", 32),
        num_levels=cfg.get("num_levels", 5),
        max_channels=cfg.get("max_channels", 512),
        deep_supervision=False,  # inference never uses deep supervision heads
    )
    sd = ck.get("model", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print("[warn] missing keys: {}".format(missing[:6]))
    model.to(device).eval()
    return model, ck


def predict_records(
    model: torch.nn.Module,
    records,
    patch: int,
    longest: Optional[int],
    device: torch.device,
    out_dir: str,
    dataset_name: str,
    seed_tag="",
    split_label: str = "test",
    threshold: float = 0.5,
    stride: Optional[int] = None,
    amp: bool = True,
    sw_batch: int = 4,
    save_prob: bool = True,
    cache_dir: Optional[str] = None,
    manifest_name: str = "manifest.csv",
    verbose: bool = True,
):
    """Run sliding-window inference over ``records`` and write prob/mask/manifest.

    Factored out of :func:`main` so callers other than the CLI (e.g. the
    cross-fit OOF path in ``src.seg.train``) can reuse the exact same file
    conventions (``prob/<key>.npy`` float16, ``mask/<key>.png`` uint8,
    ``manifest.csv``) without shelling out to a second process.

    Returns ``(rows, meta)``; ``meta`` omits ``dataset``/``ckpt``/``split``
    so callers can assemble those exactly as the CLI ``main`` does.
    """
    ds = segdata.FullImageDataset(records, longest, cache_dir=cache_dir, mem_cache=2)
    prob_dir = os.path.join(out_dir, "prob")
    mask_dir = os.path.join(out_dir, "mask")
    os.makedirs(mask_dir, exist_ok=True)
    if save_prob:
        os.makedirs(prob_dir, exist_ok=True)
    stride_val = int(stride) if stride is not None else max(1, patch // 2)

    rows = []
    t_all = time.time()
    for i in range(len(ds)):
        item = ds[i]
        t0 = time.time()
        prob_r = sliding_window_predict(
            model, item["image"], patch, device,
            stride=stride_val, amp=amp, batch_size=sw_batch,
        )
        native_hw = item["native_hw"]
        prob = to_native(prob_r, native_hw)
        fov = load_native_fov(
            {"fov_path": item["fov_path"], "image_path": item["image_path"]}, native_hw
        )
        mask = ((prob >= threshold) & (fov > 0)).astype(np.uint8)
        dt = time.time() - t0

        key = item["key"]
        pp = os.path.join(prob_dir, key + ".npy")
        mp = os.path.join(mask_dir, key + ".png")
        if save_prob:
            np.save(pp, prob.astype(np.float16))
        cv2.imwrite(mp, mask * 255)

        rows.append({
            "dataset": dataset_name,
            "seed": seed_tag,
            "split": split_label,
            "image": key,
            "subject_id": item["subject_id"],
            "image_path": item["image_path"],
            "label_path": item["label_path"],
            "label2_path": item["label2_path"],
            "fov_path": item["fov_path"],
            "prob_path": os.path.abspath(pp) if save_prob else "",
            "mask_path": os.path.abspath(mp),
            "native_h": native_hw[0],
            "native_w": native_hw[1],
            "infer_hw": "{}x{}".format(prob_r.shape[0], prob_r.shape[1]),
            "patch": patch,
            "stride": stride_val,
            "threshold": threshold,
            "pred_fg_frac": float(mask.sum()) / max(1.0, float(fov.sum())),
            "seconds": round(dt, 3),
        })
        if verbose:
            print("[{}/{}] {} {}x{} {:.2f}s fg={:.4f}".format(
                i + 1, len(ds), key, native_hw[0], native_hw[1], dt, rows[-1]["pred_fg_frac"]),
                flush=True)

    if rows:
        man = os.path.join(out_dir, manifest_name)
        with open(man, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wcsv.writeheader()
            wcsv.writerows(rows)

    meta = {
        "n_images": len(rows),
        "patch": patch,
        "stride": stride_val,
        "resize_longest": longest,
        "threshold": threshold,
        "flips": False,
        "blend": "gaussian(sigma=patch/8)",
        "total_seconds": round(time.time() - t_all, 2),
    }
    return rows, meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sliding-window vessel inference")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"])
    ap.add_argument("--split-seed", type=int, default=segdata.DEFAULT_SPLIT_SEED)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--stride-div", type=int, default=2, help="stride = patch // this")
    ap.add_argument("--sw-batch", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--save-prob", type=int, default=1)
    args = ap.parse_args(argv)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    cfg = segdata.dataset_cfg(args.dataset)
    patch, longest = int(cfg["patch"]), cfg["resize_longest"]

    recs = segdata.get_records(args.dataset)
    tr, va, te, info = segdata.make_splits(recs, args.split_seed)
    subset = {"test": te, "val": va, "train": tr, "all": recs}[args.split]
    if not subset:
        print("[error] split '{}' is empty for {}".format(args.split, args.dataset))
        return 2

    cache = args.cache_dir
    if cache is None and longest is not None:
        cache = os.path.join("runs", "_cache", "{}_{}".format(segdata.canon(args.dataset), longest))

    model, ck = load_model(args.ckpt, device)

    out_dir = args.out
    rows, meta = predict_records(
        model, subset, patch, longest, device, out_dir,
        dataset_name=segdata.canon(args.dataset),
        seed_tag=ck.get("config", {}).get("seed", ""),
        split_label=args.split,
        threshold=args.threshold,
        stride=max(1, patch // max(1, args.stride_div)),
        amp=not args.no_amp,
        sw_batch=args.sw_batch,
        save_prob=bool(args.save_prob),
        cache_dir=cache,
        manifest_name="manifest.csv",
    )

    meta = {
        "dataset": segdata.canon(args.dataset),
        "ckpt": os.path.abspath(args.ckpt),
        "split": args.split,
        **meta,
    }
    with open(os.path.join(out_dir, "infer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("wrote {} predictions -> {}".format(len(rows), out_dir))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
