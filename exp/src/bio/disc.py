"""Simple, robust optic-disc locator for fundus images.

Plan reference: exp/EXPERIMENT_PLAN.md S1.3 -- "automatic optic disc location:
brightest region + vessel-density peak; fall back to a centre-biased prior on
failure".  The disc is only needed to define the zone-B annulus for the
zone-restricted biomarkers, so a coarse (few-percent-of-FOV) centre estimate is
sufficient; we deliberately do NOT use a learned disc segmenter here to keep
Gate A CPU-only and dependency-light.

Algorithm
---------
1. Work at a reduced scale (longest side <= ``work_max_side``) so cost is
   independent of the source resolution (HRF is 3504x2336).
2. Brightness channel = mean of robustly normalised red and green channels.
   The disc is the brightest structure in both; red alone often saturates,
   green alone has the strongest vessel contrast.  A grey closing with a small
   disk suppresses the thin *dark* vessels crossing the disc -- without it the
   vessel-dense disc can be dimmer than an empty patch of retina.
3. Scale-selective blob response: a difference of two *normalised* Gaussians
   (``blur(x*fov)/blur(fov)``, sigmas ``0.35 r`` and ``3 r``).  Normalisation
   keeps the FOV rim from dragging the response down; the difference makes the
   response sensitive to a bright structure of disc size instead of to broad
   illumination plateaus.  A single blur at ``sigma = 0.6 r`` was tried first
   and mislocated 4 of 6 HRF images, whose 60 deg field makes the disc only
   ~5% of the FOV diameter.
4. Strongest response inside the FOV eroded by one disc radius -> coarse centre.
5. Refinement: inside a neighbourhood of one disc diameter around the coarse
   centre, maximise ``z(brightness) + z(vessel density)`` where vessel density
   is a sliding-window (disc-diameter box) mean of the vessel mask.  If no
   vessel mask is supplied, a cheap black-top-hat vessel proxy is used.
6. Radius = the strongest *edge* around the centre: the radius of the steepest
   negative gradient of the radial brightness profile, searched inside
   ``R_DISC_FRAC_RANGE`` of the FOV diameter.  (A fixed fraction-of-peak
   threshold and a local Otsu were both tried first; both leaked into the
   bright peripapillary retina and over-estimated CHASE_DB1 radii by ~45%.)
   If no edge is found the plan's default ``0.08 * FOV diameter`` is used and
   ``radius_estimated`` is set to ``False``.

Coordinate convention: ``cx`` is the *column* (x) index and ``cy`` the *row*
(y) index, both in FULL-RESOLUTION pixels of the input image.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi

__all__ = ["DiscEstimate", "locate_optic_disc", "save_disc_overlay", "zone_b_mask"]

# Default disc radius as a fraction of the FOV diameter (plan S1.3).
DEFAULT_R_DISC_FRAC = 0.08
# Plausible range for an *estimated* radius; outside it we fall back to the
# default above and set ``radius_estimated = False``.  A human optic disc is
# ~1.8 mm across on a ~40 deg (DRIVE/CHASE/FIVES) or ~60 deg (HRF) field, i.e.
# roughly 6-12% of the FOV diameter in radius.
R_DISC_FRAC_RANGE = (0.05, 0.13)
# Difference-of-Gaussians scales, in units of the expected disc radius.
DOG_SIGMA_IN = 0.35
DOG_SIGMA_OUT = 3.0
# Confidence thresholds (see ``locate_optic_disc``).
CONF_BRIGHTNESS_Z = 1.5
CONF_DENSITY_RATIO = 1.5
# Zone B annulus, in units of *disc diameter* measured from the disc centre.
ZONE_B_INNER_DD = 1.0
ZONE_B_OUTER_DD = 1.5


class DiscEstimate(NamedTuple):
    """Optic-disc estimate.

    Attributes
    ----------
    cx, cy : float
        Disc centre in full-resolution pixels (cx = column/x, cy = row/y).
    r : float
        Disc radius in full-resolution pixels.
    confident : bool
        ``True`` when both the brightness contrast and the vessel-density
        evidence support the location (see ``locate_optic_disc``).
    brightness_z : float
        (peak - median) / robust_std of the blurred brightness inside the FOV.
    density_ratio : float
        Vessel density at the centre divided by the median FOV vessel density.
    radius_estimated : bool
        ``False`` when ``r`` is the ``0.08 * FOV diameter`` fallback.
    fov_diameter : float
        Equivalent FOV diameter, ``2*sqrt(area/pi)``, in full-res pixels.
    refine_shift_dd : float
        Distance between the brightness-only peak and the final (vessel-density
        refined) centre, in disc diameters.  Diagnostic only.
    """

    cx: float
    cy: float
    r: float
    confident: bool
    brightness_z: float
    density_ratio: float
    radius_estimated: bool
    fov_diameter: float
    refine_shift_dd: float = 0.0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.cx, self.cy)

    def as_dict(self) -> dict:
        return dict(self._asdict())


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_bool(a) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == bool:
        return a
    mx = float(a.max()) if a.size else 0.0
    return a > (0.5 * mx if mx > 1 else 0.5)


def _robust_norm(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Percentile min-max normalise ``x`` using only pixels where ``m``."""
    v = x[m]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(v, [1.0, 99.0])
    if hi - lo < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _masked_gaussian(x: np.ndarray, m: np.ndarray, sigma: float) -> np.ndarray:
    """Normalised convolution: blur only with the in-mask evidence."""
    mf = m.astype(np.float32)
    num = ndi.gaussian_filter(x.astype(np.float32) * mf, sigma, mode="nearest")
    den = ndi.gaussian_filter(mf, sigma, mode="nearest")
    return num / np.maximum(den, 1e-6)


def _fov_diameter(fov: np.ndarray) -> float:
    area = float(fov.sum())
    if area <= 0:
        h, w = fov.shape
        return float(min(h, w))
    return 2.0 * math.sqrt(area / math.pi)


def _downscale(img: np.ndarray, factor: float, order: int) -> np.ndarray:
    """Downscale by ``factor`` (<=1) with skimage.resize."""
    if factor >= 1.0:
        return img
    from skimage.transform import resize

    h, w = img.shape[:2]
    nh, nw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    out = resize(
        img.astype(np.float32),
        (nh, nw) + img.shape[2:],
        order=order,
        mode="edge",
        anti_aliasing=(order > 0),
        preserve_range=True,
    )
    return out


def _vessel_proxy(green: np.ndarray, fov: np.ndarray, r_disc_w: float) -> np.ndarray:
    """Cheap vessel-likeness map when no vessel mask is available.

    Vessels are dark on the green channel: a black-top-hat with a small disc
    highlights them.  Only used to *refine* the disc centre, so a crude
    response is enough.
    """
    from skimage.morphology import black_tophat, disk

    rad = max(1, int(round(r_disc_w / 8.0)))
    fill = float(np.median(green[fov])) if fov.any() else 0.0
    g = np.where(fov, green, fill)
    th = black_tophat(g.astype(np.float32), disk(rad))
    th = th * fov
    if not fov.any():
        return np.zeros_like(th)
    t = np.percentile(th[fov], 92.0)
    return (th > t).astype(np.float32)


def _grey_close(x: np.ndarray, radius: int) -> np.ndarray:
    """Grey-scale morphological closing with a disk (suppresses dark vessels)."""
    from skimage.morphology import disk

    r = max(1, int(radius))
    return ndi.grey_closing(x.astype(np.float32), footprint=disk(r), mode="nearest")


def _box_mean(x: np.ndarray, size: int) -> np.ndarray:
    size = max(1, int(size))
    return ndi.uniform_filter(x.astype(np.float32), size=size, mode="constant", cval=0.0)


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def locate_optic_disc(
    image_rgb,
    fov,
    vessel_mask=None,
    r_disc_px: Optional[float] = None,
    work_max_side: int = 512,
) -> DiscEstimate:
    """Locate the optic disc.

    Parameters
    ----------
    image_rgb : (H, W, 3) or (H, W) array
        Fundus image (uint8 or float).
    fov : (H, W) bool-ish array
        Field-of-view mask.
    vessel_mask : (H, W) bool-ish array, optional
        Vessel (or skeleton) mask used for the vessel-density refinement.  In
        Gate A this is the ground-truth mask; at inference time pass the
        predicted mask.  If ``None`` a black-top-hat vessel proxy is used.
    r_disc_px : float, optional
        Force the disc radius (full-res pixels) instead of estimating it.
    work_max_side : int
        Longest side of the internal working image.

    Returns
    -------
    DiscEstimate
        ``(cx, cy, r, confident, ...)`` -- the first three fields are the
        ``(cx, cy, r_disc)`` of the plan; ``confident`` is the failure flag.
    """
    img = np.asarray(image_rgb)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] > 3:
        img = img[..., :3]
    fov_b = _as_bool(fov)
    h, w = fov_b.shape[:2]

    fov_diam = _fov_diameter(fov_b)
    r_default = DEFAULT_R_DISC_FRAC * fov_diam
    r_full = float(r_disc_px) if r_disc_px else r_default

    # ---- fallback (centre-biased prior) if the FOV is degenerate -----------
    if not fov_b.any():
        return DiscEstimate(
            w / 2.0, h / 2.0, r_default, False, 0.0, 0.0, False, fov_diam, 0.0
        )

    scale = min(1.0, float(work_max_side) / max(h, w))
    imgs = _downscale(img.astype(np.float32), scale, order=1)
    fovs = _downscale(fov_b.astype(np.float32), scale, order=0) > 0.5
    if not fovs.any():
        fovs = np.ones(imgs.shape[:2], bool)
    r_w = max(3.0, r_full * scale)  # disc radius at working scale
    dd_w = 2.0 * r_w

    red = _robust_norm(imgs[..., 0], fovs)
    green = _robust_norm(imgs[..., 1], fovs)
    bright = 0.5 * (red + green)
    # Grey closing removes the thin *dark* vessels that cross the disc; without
    # it the vessel-dense disc can be dimmer than an empty patch of retina.
    bright = _grey_close(bright, max(2, int(round(r_w / 8.0))))

    # ---- 1. brightness peak ------------------------------------------------
    # Scale-selective blob response (difference of masked Gaussians) rather
    # than a plain blur.  A plain blur at sigma ~ r ranks a broad bright
    # *plateau* above a small bright *blob*: on HRF (60 deg field, so the disc
    # is only ~5% of the FOV diameter rather than the ~8% of the 45 deg
    # DRIVE/CHASE images) that put the peak on bland mid-periphery retina in
    # 4 of 6 test images.  The DoG responds to a bright structure of roughly
    # disc size and is immune to broad illumination gradients and vignetting.
    blur = _masked_gaussian(bright, fovs, sigma=DOG_SIGMA_IN * r_w) - _masked_gaussian(
        bright, fovs, sigma=DOG_SIGMA_OUT * r_w
    )
    erode_iter = max(1, int(round(r_w)))
    fov_in = ndi.binary_erosion(fovs, structure=np.ones((3, 3), bool), iterations=erode_iter)
    if not fov_in.any():
        fov_in = fovs
    blur_in = np.where(fov_in, blur, -np.inf)
    y0, x0 = np.unravel_index(int(np.argmax(blur_in)), blur.shape)

    vals = blur[fovs]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    rstd = 1.4826 * mad if mad > 1e-9 else float(np.std(vals) + 1e-9)
    brightness_z = float((blur[y0, x0] - med) / rstd)

    # ---- 2. vessel-density refinement --------------------------------------
    if vessel_mask is not None:
        vm = _as_bool(vessel_mask)
        vmap = _downscale(vm.astype(np.float32), scale, order=1)
        vmap = np.clip(vmap, 0.0, 1.0)
    else:
        vmap = _vessel_proxy(imgs[..., 1], fovs, r_w)
    vmap = vmap[: fovs.shape[0], : fovs.shape[1]] * fovs
    dens = _box_mean(vmap, size=int(round(dd_w)))
    dmed = float(np.median(dens[fovs])) if fovs.any() else 0.0

    # neighbourhood of one disc diameter around the brightness peak
    yy, xx = np.ogrid[: blur.shape[0], : blur.shape[1]]
    near = ((yy - y0) ** 2 + (xx - x0) ** 2) <= (dd_w ** 2)
    search = near & fov_in
    if search.any():
        bz = (blur - med) / rstd
        dsig = dens[fovs]
        dm = float(np.median(dsig))
        dmad = float(np.median(np.abs(dsig - dm)))
        dstd = 1.4826 * dmad if dmad > 1e-9 else float(np.std(dsig) + 1e-9)
        dz = (dens - dm) / dstd
        comb = np.where(search, bz + dz, -np.inf)
        y1, x1 = np.unravel_index(int(np.argmax(comb)), comb.shape)
    else:
        y1, x1 = y0, x0

    density_ratio = float(dens[y1, x1] / dmed) if dmed > 1e-9 else 0.0

    # ---- 3. radius estimation ----------------------------------------------
    # Use a *lightly* smoothed brightness here: the heavy blur used for peak
    # finding (sigma = 0.6 r) shrinks the thresholded blob by ~15%.
    sharp = _masked_gaussian(bright, fovs, sigma=0.15 * r_w)
    r_est, estimated = _estimate_radius(sharp, fovs, (y1, x1), r_w)
    if estimated:
        r_full_out = float(r_est / max(scale, 1e-9))
        lo, hi = R_DISC_FRAC_RANGE[0] * fov_diam, R_DISC_FRAC_RANGE[1] * fov_diam
        if not (lo <= r_full_out <= hi):
            r_full_out, estimated = r_default, False
    else:
        r_full_out, estimated = r_default, False
    if r_disc_px:
        r_full_out, estimated = float(r_disc_px), True

    cx = float(x1) / max(scale, 1e-9)
    cy = float(y1) / max(scale, 1e-9)
    cx = float(np.clip(cx, 0, w - 1))
    cy = float(np.clip(cy, 0, h - 1))

    # Confidence: both cues must be present.  Thresholds are deliberately mild
    # -- brightness_z is 1.8-3.0 on CHASE_DB1 (broad FOV brightness spread) and
    # the vessel-density ratio at a correctly located disc is 3-5x the FOV
    # median.  Both raw numbers are returned so the flag can be re-thresholded
    # from the Gate A CSV without recomputing anything.
    confident = bool(brightness_z >= CONF_BRIGHTNESS_Z and density_ratio >= CONF_DENSITY_RATIO)
    shift = math.hypot(float(x1 - x0), float(y1 - y0)) / max(dd_w, 1e-6)
    return DiscEstimate(
        cx, cy, r_full_out, confident, brightness_z, density_ratio, estimated,
        fov_diam, float(shift),
    )


def _estimate_radius(sharp, fov, center, r_w) -> Tuple[float, bool]:
    """Disc margin = radius of the steepest drop of the radial brightness profile.

    The disc boundary is the strongest *edge* around the centre, not a
    particular grey level: a fixed fraction-of-peak threshold and Otsu inside a
    local window were both tried first and both leaked into the bright
    peripapillary retina (CHASE_DB1 over-estimated the radius by ~45%).  The
    radial mean profile is computed on the vessel-suppressed, lightly smoothed
    brightness, so the only strong negative gradient in the search band is the
    disc rim.
    """
    y, x = center
    win = int(round(2.6 * r_w))
    y_lo, y_hi = max(0, y - win), min(sharp.shape[0], y + win + 1)
    x_lo, x_hi = max(0, x - win), min(sharp.shape[1], x + win + 1)
    sub = sharp[y_lo:y_hi, x_lo:x_hi]
    subf = fov[y_lo:y_hi, x_lo:x_hi]
    if sub.size == 0 or subf.sum() < 64:
        return 0.0, False

    yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
    rad = np.rint(np.sqrt((yy - y) ** 2 + (xx - x) ** 2)).astype(np.int32)
    nb = int(min(rad.max(), win)) + 1
    if nb < 8:
        return 0.0, False
    w = subf.astype(np.float64)
    cnt = np.bincount(rad.ravel(), weights=w.ravel(), minlength=nb)[:nb]
    tot = np.bincount(rad.ravel(), weights=(sub * w).ravel(), minlength=nb)[:nb]
    good = cnt >= 8
    if good.sum() < 8:
        return 0.0, False
    prof = np.where(good, tot / np.maximum(cnt, 1.0), np.nan)
    # fill gaps then smooth
    idx = np.arange(nb)
    prof = np.interp(idx, idx[good], prof[good])
    prof = ndi.gaussian_filter1d(prof, sigma=max(1.0, 0.06 * r_w), mode="nearest")
    d = np.gradient(prof)

    lo = max(2, int(round(R_DISC_FRAC_RANGE[0] / DEFAULT_R_DISC_FRAC * r_w)))
    hi = min(nb - 2, int(round(R_DISC_FRAC_RANGE[1] / DEFAULT_R_DISC_FRAC * r_w)))
    if hi - lo < 3:
        return 0.0, False
    band = d[lo : hi + 1]
    if not np.isfinite(band).any() or float(band.min()) >= 0:
        return 0.0, False
    r = float(lo + int(np.argmin(band)))
    # require a real edge: the drop must be an appreciable fraction of the
    # centre-to-rim contrast, otherwise there is no detectable margin
    contrast = float(prof[: max(lo // 2, 1)].mean() - prof[lo : hi + 1].min())
    if contrast <= 1e-6 or -float(band.min()) < 0.01 * contrast:
        return 0.0, False
    return r, True


# --------------------------------------------------------------------------- #
# zone B
# --------------------------------------------------------------------------- #
def zone_b_mask(
    shape,
    disc: DiscEstimate,
    inner_dd: float = ZONE_B_INNER_DD,
    outer_dd: float = ZONE_B_OUTER_DD,
) -> np.ndarray:
    """Annulus ``[inner_dd, outer_dd]`` *disc diameters* from the disc centre.

    ``inner_dd=1.0, outer_dd=1.5`` (the Gate A default) means radii
    ``[2*r_disc, 3*r_disc]`` measured from the disc centre.
    """
    h, w = shape[:2]
    yy, xx = np.ogrid[:h, :w]
    d2 = (yy - disc.cy) ** 2 + (xx - disc.cx) ** 2
    dd = 2.0 * disc.r
    return (d2 >= (inner_dd * dd) ** 2) & (d2 <= (outer_dd * dd) ** 2)


# --------------------------------------------------------------------------- #
# visual check
# --------------------------------------------------------------------------- #
def save_disc_overlay(
    image_rgb,
    disc: DiscEstimate,
    path,
    fov=None,
    vessel_mask=None,
    max_side: int = 1024,
) -> str:
    """Write a PNG with the disc circle and the zone-B annulus drawn on top.

    Green = disc circle + centre crosshair, yellow = zone-B inner radius,
    red = zone-B outer radius, blue overlay = vessel mask (if given).
    """
    import os

    from skimage.draw import circle_perimeter
    from skimage.io import imsave

    img = np.asarray(image_rgb)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    img = img[..., :3].astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    h, w = img.shape[:2]
    s = min(1.0, float(max_side) / max(h, w))
    if s < 1.0:
        img = _downscale(img, s, order=1)
    out = np.clip(img, 0, 1)

    if vessel_mask is not None:
        vm = _as_bool(vessel_mask).astype(np.float32)
        vm = (_downscale(vm, s, order=0) > 0.5) if s < 1.0 else (vm > 0.5)
        vm = vm[: out.shape[0], : out.shape[1]]
        out[..., 2] = np.where(vm, 1.0, out[..., 2])

    cx, cy, r = disc.cx * s, disc.cy * s, disc.r * s
    dd = 2.0 * r
    colors = {
        "disc": (0.0, 1.0, 0.0),
        "inner": (1.0, 1.0, 0.0),
        "outer": (1.0, 0.2, 0.2),
    }
    for key, rad in (
        ("disc", r),
        ("inner", ZONE_B_INNER_DD * dd),
        ("outer", ZONE_B_OUTER_DD * dd),
    ):
        for dr in (-1, 0, 1):
            rad_i = max(1, int(round(rad + dr)))
            rr, cc = circle_perimeter(
                int(round(cy)), int(round(cx)), rad_i, shape=out.shape[:2]
            )
            out[rr, cc] = colors[key]
    # crosshair at the centre
    yi, xi = int(round(cy)), int(round(cx))
    yi = int(np.clip(yi, 0, out.shape[0] - 1))
    xi = int(np.clip(xi, 0, out.shape[1] - 1))
    k = max(3, int(round(0.3 * r)))
    ys = slice(max(0, yi - k), min(out.shape[0], yi + k + 1))
    xs = slice(max(0, xi - k), min(out.shape[1], xi + k + 1))
    out[ys, xi] = colors["disc"]
    out[yi, xs] = colors["disc"]

    d = os.path.dirname(str(path))
    if d:
        os.makedirs(d, exist_ok=True)
    imsave(str(path), (out * 255).astype(np.uint8), check_contrast=False)
    return str(path)
