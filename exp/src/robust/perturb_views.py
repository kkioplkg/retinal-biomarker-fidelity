"""Anatomy-preserving acquisition perturbations of an RGB fundus image.

Plan reference: exp/EXPERIMENT_PLAN.md S5 / proposal v3 section 4.3 (Fig.4A) --
"K=8 anatomy-preserving perturbed views per test image -> baseline vs RiGR
biomarker CV".  Every operation here simulates a different *acquisition*
condition (resolution, focus, sensor noise, compression, illumination, colour
response, small camera shake) of the **same underlying retina**.  None of them
warps, rotates or rescales the anatomy itself -- the one exception, small
translation, is a rigid shift applied identically to the image and the FOV
mask, so the vessel tree's shape/branching/length is unaffected and only its
position in the frame moves. Consequently the four primary biomarkers
(``FD``/``tortuosity``/``density``/``total_length``, which are computed from
the mask + FOV without regard to absolute position) are expected to be
invariant to translation up to the usual pixelisation/skeletonisation noise,
which is why it is included as a *stability* stressor rather than excluded as
a "geometry-changing" op.

Public API
----------
``make_views(image, fov, K=8, seed=0)``
    Returns ``(views, manifest)``.  ``views`` is a list of ``K`` dicts
    ``{"view_id", "image", "fov", "ops"}``; ``manifest`` is a list of ``K``
    flat dicts (one row per view) suitable for ``pandas.DataFrame`` / csv.

Run ``python -m src.robust.perturb_views --demo`` for a smoke test that also
writes the 2x4 example grid used to sanity-check this module.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

__all__ = ["make_views", "OP_NAMES", "apply_op", "PerturbationError"]


class PerturbationError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# low-level ops.  All take/return uint8 HxWx3 RGB in [0,255] (and, for the two
# geometry-touching ops, the uint8 {0,1} FOV mask alongside).
# --------------------------------------------------------------------------- #
def _to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def op_downsample_upsample(img: np.ndarray, scale: float) -> np.ndarray:
    """Simulate a lower-resolution acquisition re-displayed at full size."""
    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def op_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Simulate mild defocus."""
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma, sigmaY=sigma,
                             borderType=cv2.BORDER_REFLECT101)


def op_gaussian_noise(img: np.ndarray, sigma_frac: float, rng: np.random.RandomState) -> np.ndarray:
    """Additive Gaussian sensor noise, sigma given as a fraction of [0,1]."""
    x = _to_float(img)
    noise = rng.normal(0.0, sigma_frac, size=x.shape).astype(np.float32)
    return _to_uint8(np.clip(x + noise, 0.0, 1.0))


def op_jpeg_recompress(img: np.ndarray, quality: int) -> np.ndarray:
    """Re-encode through JPEG at the given quality and decode back."""
    bgr = img[..., ::-1] if img.shape[-1] == 3 else img
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise PerturbationError("JPEG encode failed")
    dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if dec is None:
        raise PerturbationError("JPEG decode failed")
    return dec[..., ::-1]


def op_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Contrast/gamma response curve: out = in ** gamma (in [0,1])."""
    x = _to_float(img)
    return _to_uint8(np.clip(x, 0.0, 1.0) ** float(gamma))


def op_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Radial illumination darkening toward the frame corners (20-40%)."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = np.sqrt(cy ** 2 + cx ** 2) + 1e-6
    factor = 1.0 - float(strength) * (r / rmax) ** 2
    x = _to_float(img) * factor[..., None]
    return _to_uint8(np.clip(x, 0.0, 1.0))


def op_color_gain(img: np.ndarray, gains: Tuple[float, float, float]) -> np.ndarray:
    """Per-channel multiplicative colour-response change (each channel +-10%)."""
    x = _to_float(img) * np.asarray(gains, dtype=np.float32)[None, None, :]
    return _to_uint8(np.clip(x, 0.0, 1.0))


def op_translate(img: np.ndarray, fov: Optional[np.ndarray], dx: int, dy: int
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Rigid shift by (dx, dy) pixels, applied identically to image and FOV.

    Reflect-padded so no artificial black border is introduced within a few
    pixels of the frame edge.  Because the shift is rigid and applied to both
    arrays, the vessel tree itself is unchanged -- only its position in the
    array -- so the four primary biomarkers (computed from mask+FOV, position
    -invariant) should be unaffected within skeletonisation/discretisation
    tolerance.
    """
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    out_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT101)
    out_fov = None
    if fov is not None:
        f = (fov.astype(np.uint8) * 255)
        f2 = cv2.warpAffine(f, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out_fov = (f2 > 127).astype(np.uint8)
    return out_img, out_fov


def op_sharpen(img: np.ndarray, amount: float = 0.6, sigma: float = 1.5) -> np.ndarray:
    """Mild unsharp-mask sharpening."""
    x = _to_float(img)
    blur = cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma,
                             borderType=cv2.BORDER_REFLECT101)
    out = x + float(amount) * (x - blur)
    return _to_uint8(np.clip(out, 0.0, 1.0))


# --------------------------------------------------------------------------- #
# registry: name -> (sampler(rng) -> params dict, needs_fov)
# --------------------------------------------------------------------------- #
def _sample_downsample_upsample(rng: np.random.RandomState) -> dict:
    return {"scale": float(rng.choice([0.75, 0.5]))}


def _sample_gaussian_blur(rng: np.random.RandomState) -> dict:
    return {"sigma": float(rng.choice([1.0, 2.0]))}


def _sample_gaussian_noise(rng: np.random.RandomState) -> dict:
    return {"sigma_frac": float(rng.choice([5.0 / 255.0, 10.0 / 255.0]))}


def _sample_jpeg(rng: np.random.RandomState) -> dict:
    return {"quality": int(rng.choice([50, 30]))}


def _sample_gamma(rng: np.random.RandomState) -> dict:
    return {"gamma": float(rng.choice([0.7, 1.4]))}


def _sample_vignette(rng: np.random.RandomState) -> dict:
    return {"strength": float(rng.uniform(0.20, 0.40))}


def _sample_color_gain(rng: np.random.RandomState) -> dict:
    gains = tuple(float(1.0 + rng.uniform(-0.10, 0.10)) for _ in range(3))
    return {"gains": gains}


def _sample_translate(rng: np.random.RandomState) -> dict:
    # avoid a degenerate (0, 0) shift so "translation" always moves something
    def _nz():
        v = 0
        while v == 0:
            v = int(rng.randint(-8, 9))
        return v
    return {"dx": _nz(), "dy": _nz()}


def _sample_sharpen(rng: np.random.RandomState) -> dict:
    return {"amount": float(rng.uniform(0.4, 0.8))}


# (name, sampler, needs_fov, apply_priority) -- priority controls composition
# order when two ops are combined: resolution/geometry first, photometric
# next, re-encoding (JPEG) last, mimicking a realistic acquisition pipeline.
_REGISTRY: Dict[str, Tuple[Callable, int]] = {
    "downsample_upsample": (_sample_downsample_upsample, 0),
    "translate": (_sample_translate, 1),
    "gaussian_blur": (_sample_gaussian_blur, 2),
    "sharpen": (_sample_sharpen, 2),
    "gamma": (_sample_gamma, 3),
    "vignette": (_sample_vignette, 3),
    "color_gain": (_sample_color_gain, 3),
    "gaussian_noise": (_sample_gaussian_noise, 4),
    "jpeg": (_sample_jpeg, 5),
}

OP_NAMES: Tuple[str, ...] = tuple(_REGISTRY.keys())


def apply_op(name: str, img: np.ndarray, fov: Optional[np.ndarray], params: dict,
             rng: np.random.RandomState) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply one named, already-parameterised op to (img, fov)."""
    if name == "downsample_upsample":
        return op_downsample_upsample(img, **params), fov
    if name == "gaussian_blur":
        return op_gaussian_blur(img, **params), fov
    if name == "gaussian_noise":
        return op_gaussian_noise(img, rng=rng, **params), fov
    if name == "jpeg":
        return op_jpeg_recompress(img, **params), fov
    if name == "gamma":
        return op_gamma(img, **params), fov
    if name == "vignette":
        return op_vignette(img, **params), fov
    if name == "color_gain":
        return op_color_gain(img, **params), fov
    if name == "translate":
        return op_translate(img, fov, **params)
    if name == "sharpen":
        return op_sharpen(img, **params), fov
    raise PerturbationError(f"unknown op {name!r}")


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def make_views(
    image: np.ndarray,
    fov: Optional[np.ndarray] = None,
    K: int = 8,
    seed: int = 0,
    p_two_ops: float = 0.6,
    include_identity: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """Build ``K`` deterministic anatomy-preserving views of ``image``.

    Parameters
    ----------
    image : (H, W, 3) uint8 array
        RGB fundus image.
    fov : (H, W) array, optional
        Field-of-view mask; shifted identically to the image by ``translate``.
        If omitted, an all-ones FOV is used internally and returned per view.
    K : int
        Number of views (default 8, per Fig.4A).
    seed : int
        Seed; identical ``(image identity via seed, K)`` always yields the
        same K compositions, independent of image content, so the manifest
        alone documents exactly what happened.
    p_two_ops : float
        Probability of composing 2 ops instead of 1 for a given view.
    include_identity : bool
        If True, view 0 is the unperturbed image (``ops == []``); useful as
        an explicit baseline row. Default False -- all K views are perturbed
        (baseline is simply the model's normal test-set prediction).

    Returns
    -------
    (views, manifest)
        ``views``: list of ``{"view_id", "image", "fov", "ops"}``, ``ops`` a
        list of ``(name, params)`` tuples in application order.
        ``manifest``: list of flat dicts (one row per view): ``view_id``,
        ``n_ops``, ``ops`` (';'-joined names), ``params_json``, ``seed``.
    """
    if cv2 is None:
        raise ImportError("perturb_views requires opencv-python (cv2)")
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    image = np.ascontiguousarray(image[..., :3].astype(np.uint8))
    if fov is None:
        fov_in: Optional[np.ndarray] = np.ones(image.shape[:2], dtype=np.uint8)
    else:
        f = np.asarray(fov)
        if f.ndim == 3:
            f = f[..., 0]
        fov_in = (f > 0).astype(np.uint8)

    names = list(OP_NAMES)
    views: List[dict] = []
    manifest: List[dict] = []

    for i in range(K):
        rng = np.random.RandomState((int(seed) * 1_000_003 + i * 97 + 1) & 0x7FFFFFFF)

        if include_identity and i == 0:
            views.append({"view_id": 0, "image": image.copy(), "fov": fov_in.copy(), "ops": []})
            manifest.append({"view_id": 0, "n_ops": 0, "ops": "", "params_json": "{}", "seed": int(seed)})
            continue

        n_ops = 2 if rng.uniform() < p_two_ops else 1
        chosen = list(rng.choice(names, size=n_ops, replace=False))
        chosen.sort(key=lambda n: _REGISTRY[n][1])  # realistic pipeline order

        cur_img, cur_fov = image.copy(), fov_in.copy()
        ops_applied: List[Tuple[str, dict]] = []
        for name in chosen:
            sampler, _prio = _REGISTRY[name]
            params = sampler(rng)
            cur_img, cur_fov = apply_op(name, cur_img, cur_fov, params, rng)
            ops_applied.append((name, params))

        views.append({"view_id": i, "image": cur_img, "fov": cur_fov, "ops": ops_applied})
        manifest.append({
            "view_id": i,
            "n_ops": len(ops_applied),
            "ops": ";".join(n for n, _ in ops_applied),
            "params_json": json.dumps({n: p for n, p in ops_applied}),
            "seed": int(seed),
        })

    return views, manifest


# --------------------------------------------------------------------------- #
# demo / smoke test
# --------------------------------------------------------------------------- #
def _demo(image_path: Optional[str], fov_path: Optional[str], out_path: str,
          K: int, seed: int) -> None:
    import os

    if image_path is None:
        # fall back to the first DRIVE test image shipped with the repo
        here = os.path.dirname(os.path.abspath(__file__))
        exp_root = os.path.abspath(os.path.join(here, "..", ".."))
        cand = os.path.join(exp_root, "data", "drive", "raw", "test", "images", "01_test.tif")
        if os.path.exists(cand):
            image_path = cand
        else:
            raise FileNotFoundError(
                "no --image given and the default DRIVE demo image is missing: " + cand
            )

    from src.data.datasets import read_image

    img = read_image(image_path)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3].astype(np.uint8)

    fov = None
    if fov_path:
        fov = read_image(fov_path)
        if fov.ndim == 3:
            fov = fov[..., 0]
        fov = (fov > 0.5 * float(fov.max() if fov.max() else 1)).astype(np.uint8)

    views, manifest = make_views(img, fov, K=K, seed=seed)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 4
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, v, m in zip(axes, views, manifest):
        ax.imshow(v["image"])
        title = m["ops"] if m["ops"] else "(identity)"
        ax.set_title(f"view {v['view_id']}: {title}", fontsize=9)
        ax.axis("off")
    for ax in axes[len(views):]:
        ax.axis("off")
    fig.suptitle(f"perturb_views demo -- {os.path.basename(image_path)} (K={K}, seed={seed})")
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    print(f"wrote {len(views)} views -> {out_path}")
    for m in manifest:
        print(" ", m)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="anatomy-preserving perturbation demo")
    ap.add_argument("--image", default=None)
    ap.add_argument("--fov", default=None)
    ap.add_argument("--out", default=os.path.join("runs", "robust", "views_example.png"))
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo", action="store_true", help="no-op flag kept for CLI clarity")
    args = ap.parse_args(argv)
    _demo(args.image, args.fov, args.out, args.K, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
