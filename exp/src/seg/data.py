"""Datasets and augmentation for the S2 baseline segmenter.

Built on top of ``src.data.datasets.load_dataset(name)``, which returns a list of
dicts with keys ``image_path, label_path, label2_path, fov_path, split,
subject_id``.

Per-dataset geometry (exp/EXPERIMENT_PLAN.md S2):
    DRIVE / CHASE_DB1 / STARE : native resolution, 512 x 512 patches
    HRF / FIVES               : resized so the longest side is 1536, 768 patches

Normalisation: per-image z-score computed over the FOV pixels of the (resized)
image, applied to every channel independently.

Validation split: 15% of the *training* images, grouped by ``subject_id``, drawn
with a dedicated ``split_seed`` (default 12345) so that every training seed sees
the same split.  ``make_splits`` returns the chosen ids so they can be recorded.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

cv2.setNumThreads(0)

# --------------------------------------------------------------------------
# dataset geometry
# --------------------------------------------------------------------------
DEFAULT_SPLIT_SEED = 12345
VAL_FRACTION = 0.15

DATASET_CFG: Dict[str, Dict[str, Optional[int]]] = {
    "drive": {"resize_longest": None, "patch": 512},
    "chasedb1": {"resize_longest": None, "patch": 512},
    "stare": {"resize_longest": None, "patch": 512},
    "hrf": {"resize_longest": 1536, "patch": 768},
    "fives": {"resize_longest": 1536, "patch": 768},
}

# batch sizes that fit a 20 GB RTX 3080 with AMP
DEFAULT_BATCH: Dict[str, int] = {
    "drive": 8,
    "chasedb1": 8,
    "stare": 8,
    "hrf": 4,
    "fives": 4,
}

DEFAULT_EPOCHS: Dict[str, int] = {
    "drive": 300,
    "chasedb1": 300,
    "stare": 300,
    "hrf": 150,
    "fives": 150,
}


def canon(name: str) -> str:
    """Canonicalise a dataset name: 'CHASE_DB1' -> 'chasedb1'."""
    s = "".join(ch for ch in str(name).lower() if ch.isalnum())
    aliases = {
        "chase": "chasedb1",
        "chasedb": "chasedb1",
        "chasedb1": "chasedb1",
        "drive": "drive",
        "stare": "stare",
        "hrf": "hrf",
        "fives": "fives",
        "five": "fives",
    }
    return aliases.get(s, s)


def dataset_cfg(name: str) -> Dict[str, Optional[int]]:
    key = canon(name)
    if key not in DATASET_CFG:
        raise KeyError(
            "unknown dataset '{}' (known: {})".format(name, sorted(DATASET_CFG))
        )
    return dict(DATASET_CFG[key])


# --------------------------------------------------------------------------
# io helpers
# --------------------------------------------------------------------------
def _imread_any(path: str) -> np.ndarray:
    """cv2 first, imageio as fallback (DRIVE labels/FOV are .gif, which OpenCV
    cannot decode; STARE uses .ppm, HRF .tif)."""
    data = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if data is not None:
        if data.ndim == 3 and data.shape[2] >= 3:
            data = np.ascontiguousarray(data[..., ::-1] if data.shape[2] == 3
                                        else data[..., 2::-1])
        return data
    import imageio.v3 as iio  # only needed for the formats OpenCV rejects
    arr = np.asarray(iio.imread(str(path)))
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]
    return arr


def _imread_color(path: str) -> np.ndarray:
    img = _imread_any(path)
    if img is None:
        raise FileNotFoundError("cannot read image: {}".format(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        m = float(img.max()) if img.size else 1.0
        img = (img.astype(np.float32) / (m if m > 0 else 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(img[..., :3])


def _imread_gray(path: str) -> np.ndarray:
    data = _imread_any(path)
    if data is None:
        raise FileNotFoundError("cannot read mask: {}".format(path))
    if data.ndim == 3:
        data = data[..., 0] if data.shape[2] < 3 else cv2.cvtColor(
            np.ascontiguousarray(data[..., :3]), cv2.COLOR_RGB2GRAY
        )
    if data.dtype == np.bool_:
        data = data.astype(np.uint8) * 255
    if data.dtype != np.uint8:
        m = float(data.max()) if data.size else 1.0
        data = (data.astype(np.float32) / (m if m > 0 else 1.0) * 255.0).astype(np.uint8)
    return data


def derive_fov(img_rgb: np.ndarray, thresh: int = 20) -> np.ndarray:
    """Fallback FOV: threshold the intensity, keep the largest blob, fill holes."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, binm = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    binm = cv2.morphologyEx(binm, cv2.MORPH_OPEN, k)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binm, 8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binm = np.where(lab == biggest, 255, 0).astype(np.uint8)
    # fill interior holes
    ff = binm.copy()
    h, w = ff.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 255)
    binm = binm | cv2.bitwise_not(ff)
    return (binm > 0).astype(np.uint8)


def _resize_longest(arr: np.ndarray, longest: int, interp: int) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = float(longest) / float(max(h, w))
    nh, nw = int(round(h * scale)), int(round(w * scale))
    return cv2.resize(arr, (nw, nh), interpolation=interp)


@dataclass
class Sample:
    """One fully-prepared (possibly resized) image with its masks."""

    key: str
    image: np.ndarray      # uint8 HxWx3, RGB
    label: np.ndarray      # uint8 HxW, {0,1}
    fov: np.ndarray        # uint8 HxW, {0,1}
    mean: np.ndarray       # float32 (3,)
    std: np.ndarray        # float32 (3,)
    native_hw: Tuple[int, int]
    record: dict = field(default_factory=dict)


def load_sample(rec: dict, resize_longest: Optional[int], key: Optional[str] = None) -> Sample:
    img = _imread_color(rec["image_path"])
    native_hw = img.shape[:2]

    lab_path = rec.get("label_path")
    if lab_path:
        lab = _imread_gray(lab_path)
        lab = (lab > 127).astype(np.uint8)
    else:
        lab = np.zeros(native_hw, np.uint8)

    fov_path = rec.get("fov_path")
    if fov_path and os.path.exists(str(fov_path)):
        fov = (_imread_gray(fov_path) > 127).astype(np.uint8)
    else:
        fov = derive_fov(img)

    if resize_longest is not None and max(native_hw) != resize_longest:
        img = _resize_longest(img, resize_longest, cv2.INTER_AREA)
        lab = (_resize_longest(lab * 255, resize_longest, cv2.INTER_LINEAR) > 127).astype(np.uint8)
        fov = (_resize_longest(fov * 255, resize_longest, cv2.INTER_LINEAR) > 127).astype(np.uint8)

    if lab.shape != img.shape[:2]:
        lab = cv2.resize(lab * 255, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        lab = (lab > 127).astype(np.uint8)
    if fov.shape != img.shape[:2]:
        fov = cv2.resize(fov * 255, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_NEAREST)
        fov = (fov > 127).astype(np.uint8)

    m = fov > 0
    if m.sum() < 16:
        m = np.ones_like(fov, bool)
    pix = img[m].astype(np.float32)
    mean = pix.mean(axis=0)
    std = pix.std(axis=0)
    std = np.maximum(std, 1e-3)

    if key is None:
        key = os.path.splitext(os.path.basename(str(rec["image_path"])))[0]
    return Sample(key, img, lab, fov, mean.astype(np.float32), std.astype(np.float32),
                  (int(native_hw[0]), int(native_hw[1])), rec)


def normalize(img_u8: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-image z-score (FOV statistics), HWC uint8 -> CHW float32."""
    x = (img_u8.astype(np.float32) - mean[None, None, :]) / std[None, None, :]
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# --------------------------------------------------------------------------
# on-disk cache of the resized samples (HRF / FIVES)
# --------------------------------------------------------------------------
class SampleStore:
    """Loads samples on demand, with a small in-memory LRU and an optional
    on-disk cache of the resized uint8 arrays (big win for HRF / FIVES)."""

    def __init__(
        self,
        records: Sequence[dict],
        resize_longest: Optional[int],
        cache_dir: Optional[str] = None,
        mem_cache: int = 16,
    ):
        self.records = list(records)
        self.resize_longest = resize_longest
        self.cache_dir = cache_dir
        self.mem_cache = int(mem_cache)
        self._mem: "Dict[int, Sample]" = {}
        self._order: List[int] = []
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.records)

    def _cache_path(self, idx: int) -> Optional[str]:
        if not self.cache_dir:
            return None
        rec = self.records[idx]
        h = hashlib.md5(
            "{}|{}|{}".format(rec.get("image_path"), rec.get("label_path"),
                              self.resize_longest).encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(self.cache_dir, h + ".npz")

    def _load(self, idx: int) -> Sample:
        rec = self.records[idx]
        cp = self._cache_path(idx)
        if cp and os.path.exists(cp):
            try:
                z = np.load(cp, allow_pickle=False)
                return Sample(
                    key=str(z["key"]),
                    image=z["image"], label=z["label"], fov=z["fov"],
                    mean=z["mean"], std=z["std"],
                    native_hw=(int(z["native_hw"][0]), int(z["native_hw"][1])),
                    record=rec,
                )
            except Exception:
                pass
        s = load_sample(rec, self.resize_longest)
        if cp:
            try:
                tmp = cp[:-4] + ".{}.tmp.npz".format(os.getpid())
                np.savez(tmp, key=s.key, image=s.image, label=s.label, fov=s.fov,
                         mean=s.mean, std=s.std, native_hw=np.array(s.native_hw))
                os.replace(tmp, cp)
            except Exception:
                pass
        return s

    def get(self, idx: int) -> Sample:
        if idx in self._mem:
            return self._mem[idx]
        s = self._load(idx)
        if self.mem_cache > 0:
            self._mem[idx] = s
            self._order.append(idx)
            while len(self._order) > self.mem_cache:
                self._mem.pop(self._order.pop(0), None)
        return s


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------
def _try(ctor, *variants):
    """Instantiate an albumentations transform, tolerating API renames."""
    last = None
    for kw in variants:
        try:
            return ctor(**kw)
        except Exception as exc:  # TypeError on renamed kwargs
            last = exc
    raise last


def build_train_transform(patch: int, strength: float = 1.0):
    """Strong augmentation applied identically to image / label / fov."""
    import albumentations as A

    p = float(strength)
    tf = [
        _try(A.PadIfNeeded,
             dict(min_height=patch, min_width=patch, border_mode=cv2.BORDER_CONSTANT,
                  fill=0, fill_mask=0, p=1.0),
             dict(min_height=patch, min_width=patch, border_mode=cv2.BORDER_CONSTANT,
                  value=0, mask_value=0, p=1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        _try(A.Affine,
             dict(scale=(0.85, 1.20), translate_percent=(-0.05, 0.05), rotate=(-30, 30),
                  shear=(-5, 5), interpolation=cv2.INTER_LINEAR,
                  mask_interpolation=cv2.INTER_NEAREST,
                  border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.5 * p),
             dict(scale=(0.85, 1.20), translate_percent=(-0.05, 0.05), rotate=(-30, 30),
                  shear=(-5, 5), interpolation=cv2.INTER_LINEAR,
                  mode=cv2.BORDER_CONSTANT, cval=0, cval_mask=0, p=0.5 * p)),
        _try(A.ElasticTransform,
             dict(alpha=30.0, sigma=8.0, interpolation=cv2.INTER_LINEAR,
                  border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0, p=0.2 * p),
             dict(alpha=30.0, sigma=8.0, alpha_affine=0.0, interpolation=cv2.INTER_LINEAR,
                  border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.2 * p),
             dict(alpha=30.0, sigma=8.0, p=0.2 * p)),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5 * p),
        _try(A.RandomGamma, dict(gamma_limit=(70, 150), p=0.3 * p)),
        _try(A.GaussianBlur, dict(blur_limit=(3, 7), sigma_limit=(0.3, 1.5), p=0.2 * p),
             dict(blur_limit=(3, 7), p=0.2 * p)),
        _try(A.GaussNoise, dict(std_range=(0.02, 0.12), p=0.2 * p),
             dict(var_limit=(5.0, 40.0), p=0.2 * p)),
        _try(A.HueSaturationValue,
             dict(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3 * p)),
    ]
    return A.Compose(tf, additional_targets={"fov": "mask"})


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def make_splits(
    records: Sequence[dict],
    split_seed: int = DEFAULT_SPLIT_SEED,
    val_fraction: float = VAL_FRACTION,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    """Split the official train records into train / val by ``subject_id``.

    Returns (train, val, test, info) where ``info`` records the seed, the
    fraction and the exact validation subjects / image keys.
    """
    train_pool = [r for r in records if str(r.get("split", "train")).lower().startswith("train")]
    test = [r for r in records if not str(r.get("split", "train")).lower().startswith("train")]

    def _key(r):
        return os.path.splitext(os.path.basename(str(r["image_path"])))[0]

    subjects: Dict[str, List[dict]] = {}
    for r in train_pool:
        sid = str(r.get("subject_id") or _key(r))
        subjects.setdefault(sid, []).append(r)

    sids = sorted(subjects)
    rng = random.Random(int(split_seed))
    rng.shuffle(sids)
    n_val = max(1, int(round(len(sids) * float(val_fraction))))
    n_val = min(n_val, max(1, len(sids) - 1))
    val_sids = sorted(sids[:n_val])
    tr_sids = sorted(sids[n_val:])

    train = [r for s in tr_sids for r in subjects[s]]
    val = [r for s in val_sids for r in subjects[s]]
    info = {
        "split_seed": int(split_seed),
        "val_fraction": float(val_fraction),
        "n_train_pool": len(train_pool),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "val_subjects": val_sids,
        "val_images": sorted(_key(r) for r in val),
        "train_images": sorted(_key(r) for r in train),
        "test_images": sorted(_key(r) for r in test),
    }
    return train, val, test, info


def make_crossfit_splits(
    records: Sequence[dict],
    crossfit_k: int,
    fold: int,
    split_seed: int = DEFAULT_SPLIT_SEED,
    val_fraction: float = VAL_FRACTION,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], dict]:
    """K-fold cross-fit split of the official training pool, by ``subject_id``.

    Subjects of the training pool are shuffled once (deterministically, via
    ``split_seed``) and assigned round-robin to ``crossfit_k`` folds. Fold
    ``fold`` is held out entirely (used later for out-of-fold inference); the
    remaining subjects are split into train / val exactly like
    :func:`make_splits` (15% val, same shuffled order, same ``split_seed``),
    so a crossfit run's val split is drawn the same way as a normal run's.

    Returns ``(train, val, held_out, test, info)``. ``test`` is the dataset's
    official test split, untouched. ``info`` records the fold membership
    (``fold_subjects``, ``held_out_subjects`` / ``held_out_images``) alongside
    the usual train/val/test image lists, for ``split.json``.
    """
    train_pool = [r for r in records if str(r.get("split", "train")).lower().startswith("train")]
    test = [r for r in records if not str(r.get("split", "train")).lower().startswith("train")]

    def _key(r):
        return os.path.splitext(os.path.basename(str(r["image_path"])))[0]

    subjects: Dict[str, List[dict]] = {}
    for r in train_pool:
        sid = str(r.get("subject_id") or _key(r))
        subjects.setdefault(sid, []).append(r)

    sids = sorted(subjects)
    rng = random.Random(int(split_seed))
    rng.shuffle(sids)

    k = int(crossfit_k)
    if k < 2:
        raise ValueError("crossfit_k must be >= 2 (got {})".format(k))
    fold = int(fold)
    if not (0 <= fold < k):
        raise ValueError("fold must be in [0, {}) (got {})".format(k, fold))
    if k > len(sids):
        raise ValueError(
            "crossfit_k={} exceeds the number of subjects ({})".format(k, len(sids))
        )

    folds = [sids[i::k] for i in range(k)]
    held_sids = set(folds[fold])
    # subjects not in the held-out fold, in the same shuffled order as `sids`
    remain_sids = [s for s in sids if s not in held_sids]

    n_val = max(1, int(round(len(remain_sids) * float(val_fraction))))
    n_val = min(n_val, max(1, len(remain_sids) - 1))
    val_sids = sorted(remain_sids[:n_val])
    tr_sids = sorted(remain_sids[n_val:])

    train = [r for s in tr_sids for r in subjects[s]]
    val = [r for s in val_sids for r in subjects[s]]
    held = [r for s in sorted(held_sids) for r in subjects[s]]

    info = {
        "mode": "crossfit",
        "split_seed": int(split_seed),
        "val_fraction": float(val_fraction),
        "crossfit_k": k,
        "fold": fold,
        "n_train_pool": len(train_pool),
        "n_train": len(train),
        "n_val": len(val),
        "n_held_out": len(held),
        "n_test": len(test),
        "fold_subjects": {str(i): sorted(f) for i, f in enumerate(folds)},
        "held_out_subjects": sorted(held_sids),
        "val_subjects": val_sids,
        "train_subjects": tr_sids,
        "held_out_images": sorted(_key(r) for r in held),
        "val_images": sorted(_key(r) for r in val),
        "train_images": sorted(_key(r) for r in train),
        "test_images": sorted(_key(r) for r in test),
    }
    return train, val, held, test, info


def get_records(name: str):
    """Import ``src.data.datasets.load_dataset`` lazily and canonicalise names."""
    from src.data.datasets import load_dataset  # noqa: WPS433 (deliberate late import)

    last = None
    for cand in (name, canon(name), name.upper(), name.lower()):
        try:
            recs = load_dataset(cand)
            if recs:
                return list(recs)
        except Exception as exc:
            last = exc
    if last is not None:
        raise last
    raise RuntimeError("load_dataset('{}') returned nothing".format(name))


# --------------------------------------------------------------------------
# training dataset
# --------------------------------------------------------------------------
class VesselPatchDataset(Dataset):
    """FOV-aware random patch sampling with vessel oversampling.

    ``__len__`` is ``samples_per_epoch`` (an epoch is a fixed number of
    iterations, per the plan); indices are only used to drive the RNG.
    """

    def __init__(
        self,
        records: Sequence[dict],
        patch: int,
        resize_longest: Optional[int],
        samples_per_epoch: int,
        augment: bool = True,
        oversample_p: float = 0.7,
        seed: int = 0,
        cache_dir: Optional[str] = None,
        mem_cache: int = 16,
        min_vessel_px: int = 32,
        aug_strength: float = 1.0,
    ):
        self.store = SampleStore(records, resize_longest, cache_dir, mem_cache)
        self.patch = int(patch)
        self.samples_per_epoch = int(samples_per_epoch)
        self.oversample_p = float(oversample_p)
        self.seed = int(seed)
        self.epoch = 0
        self.min_vessel_px = int(min_vessel_px)
        self.transform = build_train_transform(self.patch, aug_strength) if augment else None
        self._rng: Optional[np.random.Generator] = None

    def set_epoch(self, epoch: int) -> None:
        """Bookkeeping only.  Sampling uses a per-worker RNG that advances
        continuously, so it stays reproducible under persistent workers (where
        a per-epoch reseed of the main-process copy would never reach them)."""
        self.epoch = int(epoch)

    def _worker_rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = int(info.id) if info is not None else 0
            self._rng = np.random.default_rng([self.seed, wid, 20240902])
        return self._rng

    def __len__(self) -> int:
        return self.samples_per_epoch

    # -- patch location ----------------------------------------------------
    def _sample_patch(self, s: Sample, rng: np.random.Generator):
        h, w = s.image.shape[:2]
        ps = self.patch
        ph, pw = min(ps, h), min(ps, w)
        want_vessel = rng.random() < self.oversample_p
        best = None
        tries = 12 if want_vessel else 4
        for _ in range(tries):
            y = int(rng.integers(0, max(1, h - ph + 1)))
            x = int(rng.integers(0, max(1, w - pw + 1)))
            lab = s.label[y:y + ph, x:x + pw]
            fov = s.fov[y:y + ph, x:x + pw]
            n_ves = int(lab.sum())
            n_fov = int(fov.sum())
            score = (n_ves, n_fov)
            if best is None or score > best[0]:
                best = (score, y, x)
            if want_vessel:
                if n_ves >= self.min_vessel_px:
                    return y, x, ph, pw
            elif n_fov > 0.2 * ph * pw:
                return y, x, ph, pw
        _, y, x = best
        return y, x, ph, pw

    def __getitem__(self, idx: int):
        rng = self._worker_rng()
        si = int(rng.integers(0, len(self.store)))
        s = self.store.get(si)
        y, x, ph, pw = self._sample_patch(s, rng)

        img = np.ascontiguousarray(s.image[y:y + ph, x:x + pw])
        lab = np.ascontiguousarray(s.label[y:y + ph, x:x + pw])
        fov = np.ascontiguousarray(s.fov[y:y + ph, x:x + pw])

        if self.transform is not None:
            seed_i = int(rng.integers(0, 2 ** 31 - 1))
            random.seed(seed_i)
            np.random.seed(seed_i)
            out = self.transform(image=img, mask=lab, fov=fov)
            img, lab, fov = out["image"], out["mask"], out["fov"]

        # guarantee the exact patch size
        if img.shape[0] != self.patch or img.shape[1] != self.patch:
            img, lab, fov = _pad_or_crop(img, lab, fov, self.patch)

        x_t = torch.from_numpy(normalize(img, s.mean, s.std))
        y_t = torch.from_numpy((lab > 0).astype(np.float32))[None]
        f_t = torch.from_numpy((fov > 0).astype(np.float32))[None]
        return {"image": x_t, "label": y_t, "fov": f_t, "key": s.key}


def _pad_or_crop(img, lab, fov, size):
    h, w = img.shape[:2]
    if h > size or w > size:
        img, lab, fov = img[:size, :size], lab[:size, :size], fov[:size, :size]
        h, w = img.shape[:2]
    ph, pw = size - h, size - w
    if ph > 0 or pw > 0:
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="constant")
        lab = np.pad(lab, ((0, ph), (0, pw)), mode="constant")
        fov = np.pad(fov, ((0, ph), (0, pw)), mode="constant")
    return img, lab, fov


# --------------------------------------------------------------------------
# full-image dataset (validation / inference)
# --------------------------------------------------------------------------
class FullImageDataset(Dataset):
    """Whole (resized) images, normalised, for sliding-window inference.

    Yields dicts with numpy/tensor fields; use ``batch_size=1`` and
    ``collate_fn=full_image_collate`` because image sizes differ.
    """

    def __init__(
        self,
        records: Sequence[dict],
        resize_longest: Optional[int],
        cache_dir: Optional[str] = None,
        mem_cache: int = 4,
    ):
        self.store = SampleStore(records, resize_longest, cache_dir, mem_cache)

    def __len__(self) -> int:
        return len(self.store)

    def __getitem__(self, idx: int):
        s = self.store.get(idx)
        x = torch.from_numpy(normalize(s.image, s.mean, s.std))
        y = torch.from_numpy((s.label > 0).astype(np.float32))[None]
        f = torch.from_numpy((s.fov > 0).astype(np.float32))[None]
        return {
            "image": x,
            "label": y,
            "fov": f,
            "key": s.key,
            "native_hw": s.native_hw,
            "image_path": str(s.record.get("image_path", "")),
            "label_path": str(s.record.get("label_path", "") or ""),
            "label2_path": str(s.record.get("label2_path", "") or ""),
            "fov_path": str(s.record.get("fov_path", "") or ""),
            "subject_id": str(s.record.get("subject_id", "") or ""),
            "split": str(s.record.get("split", "") or ""),
        }


def full_image_collate(batch):
    return batch  # keep the list of dicts; sizes differ


def seed_worker(worker_id: int) -> None:
    s = torch.initial_seed() % (2 ** 32)
    np.random.seed(s)
    random.seed(s)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="drive")
    ap.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    args = ap.parse_args()

    cfg = dataset_cfg(args.dataset)
    recs = get_records(args.dataset)
    tr, va, te, info = make_splits(recs, args.split_seed)
    print(json.dumps({k: v for k, v in info.items() if k != "train_images"}, indent=2))
    ds = VesselPatchDataset(tr, cfg["patch"], cfg["resize_longest"], 4, seed=0)
    for i in range(2):
        b = ds[i]
        print(b["key"], tuple(b["image"].shape), float(b["label"].mean()),
              float(b["fov"].mean()))
