"""Train the RiGR micro multi-task head (plan stage S4 item 1).

CLI::

    python -m src.rigr.train_head --dataset drive --seed 0 --gpu 0 --epochs 60 \
        --out runs/rigr_head/drive/seed0 --pred_dir runs/seg/drive/seed0/pred

Inputs are ``[I z-scored inside the FOV, P]`` (4 channels).  ``P`` comes from
``--pred_dir`` (a ``src.seg.infer`` output directory holding ``prob/<key>.npy``);
**until stage S2 has produced those files** the trainer falls back to a
synthetic ``P`` -- the reference mask with random capsule severances, Gaussian
blurred (``src.rigr.synth_cuts.synth_prob``) -- so the head can be smoke-tested
end to end.  The checkpoint records which source was used in
``config[prob_source]``; a head trained on synthetic ``P`` must never be used
for a reported result.

Geometry follows the segmenter exactly (``src.seg.data.dataset_cfg``):
DRIVE / CHASE_DB1 / STARE at native resolution with 512 px patches, HRF / FIVES
resized to a longest side of 1536 with 768 px patches.

Augmentation is restricted to **flips and rot90**, the lattice isometries under
which the axial orientation target transforms exactly and without resampling
(see :func:`src.rigr.head.flip_rot_theta`); image, label, FOV, ``P`` and the
orientation maps are all transformed together, and the angle map is then
shifted by the corresponding axial rotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.rigr.head import (BACKGROUND_WEIGHT, JUNCTION_WEIGHT, K_BINS, MicroUNet,
                           flip_rot_theta, head_loss, orientation_targets)
from src.seg import data as segdata

#: bump when the cached array set or its semantics change
HEAD_CACHE_VERSION = 1
HEAD_CACHE_ROOT = os.path.join("runs", "_cache", "rigr_head")

#: the only four values ``orientation_targets`` can put in ``weight``
#: (background / junction / vessel, times the boolean FOV) -- stored as a uint8
#: code so the map costs 1 byte instead of 4 per pixel and is decoded through
#: this exact table, so the tensor the loss sees is bit-identical to float32.
_W_LEVELS = np.array([0.0, BACKGROUND_WEIGHT, JUNCTION_WEIGHT, 1.0], dtype=np.float32)

_CACHE_ARRAYS = ("image", "label", "fov", "theta", "wcode", "valid", "prob")


# --------------------------------------------------------------------------
def load_prob(key: str, pred_dir: Optional[str], shape) -> Optional[np.ndarray]:
    """Read ``pred_dir/prob/<key>.npy`` and resize to ``shape``; None if absent."""
    if not pred_dir:
        return None
    p = os.path.join(pred_dir, "prob", key + ".npy")
    if not os.path.exists(p):
        return None
    arr = np.load(p).astype(np.float32)
    if arr.shape[:2] != tuple(shape):
        import cv2

        arr = cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(arr, 0.0, 1.0)


class HeadSample:
    """One prepared image, held as **memory-mapped** arrays.

    ``image``, ``label``, ``fov``, ``theta``, ``wcode``, ``valid`` and ``prob``
    are ``np.load(..., mmap_mode="r")`` views of the per-image cache written by
    :func:`build_image_cache`.  Constructing one costs a few file-header reads,
    so the store can keep every training image open at once and a patch fetch
    is pure array slicing -- the OS page cache does the rest.  ``weight`` is
    reconstructed per patch from ``wcode`` through :data:`_W_LEVELS`.
    """

    __slots__ = ("key", "image", "prob", "label", "fov", "mean", "std",
                 "theta", "wcode", "valid")

    def __init__(self, key, mean, std, arrays: Dict[str, np.ndarray]):
        self.key = key
        self.mean = mean
        self.std = std
        self.image = arrays["image"]
        self.prob = arrays["prob"]
        self.label = arrays["label"]
        self.fov = arrays["fov"]
        self.theta = arrays["theta"]
        self.wcode = arrays["wcode"]
        self.valid = arrays["valid"]


def _cache_id(rec: dict, resize_longest, pred_dir, prob_tag: str) -> str:
    raw = "|".join([str(rec.get("image_path")), str(rec.get("label_path")),
                    str(rec.get("fov_path")), str(resize_longest),
                    str(pred_dir or ""), prob_tag, str(HEAD_CACHE_VERSION)])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]


def encode_weight(w: np.ndarray):
    """``weight`` float32 -> uint8 code, or ``None`` when it is not one of the
    four expected levels (the caller then stores float32 and nothing is lost)."""
    code = np.zeros(w.shape, dtype=np.uint8)
    hit = np.zeros(w.shape, dtype=bool)
    for i, v in enumerate(_W_LEVELS):
        m = (w == v)
        code[m] = i
        hit |= m
    return code if bool(hit.all()) else None


def build_image_cache(rec: dict, resize_longest, pred_dir, out_dir: str,
                      idx: int = 0, seed: int = 0, synth_cuts: int = 12,
                      force: bool = False) -> str:
    """Materialise every per-image array the sampler needs, once.

    The expensive part is :func:`src.rigr.head.orientation_targets`
    (skeletonise + local PCA + two EDTs: ~1.1 s on DRIVE, several seconds on a
    1536 px HRF / FIVES image).  It used to run on every LRU miss *inside the
    training loop* -- 17 DRIVE training images against an 8-image cache is a
    ~50% miss rate over 800 samples per epoch, which is exactly what pinned the
    CPU at 100% and left the GPU at 0-2%.  Nothing about *what* is computed
    changes here: it is computed once and written to disk.
    """
    done = os.path.join(out_dir, "meta.json")
    if os.path.exists(done) and not force:
        return out_dir
    os.makedirs(out_dir, exist_ok=True)

    s = segdata.load_sample(rec, resize_longest)
    prob = load_prob(s.key, pred_dir, s.label.shape)
    if prob is None:
        from src.rigr.synth_cuts import (make_cut_mask, sample_cut_loci,
                                         synth_prob)

        rng = np.random.default_rng([int(seed), int(idx), 777])
        gt = s.label > 0
        fov = s.fov > 0
        loci = sample_cut_loci(gt, fov, n_cuts=int(synth_cuts), rng=rng)
        cut, _ = make_cut_mask(gt, loci, fov=fov, verify_beta0=False)
        prob = synth_prob(cut, sigma=2.0, noise=0.02, rng=rng)

    tgt = orientation_targets(s.label > 0, s.fov > 0)
    wcode = encode_weight(tgt["weight"])
    arrays = {
        "image": np.ascontiguousarray(s.image),
        "label": np.ascontiguousarray(s.label > 0),
        "fov": np.ascontiguousarray(s.fov > 0),
        "theta": np.ascontiguousarray(tgt["theta"].astype(np.float32)),
        "valid": np.ascontiguousarray(np.asarray(tgt["valid"], dtype=bool)),
        "prob": np.ascontiguousarray(np.asarray(prob, dtype=np.float32)),
        "wcode": (np.ascontiguousarray(wcode) if wcode is not None
                  else np.ascontiguousarray(tgt["weight"].astype(np.float32))),
    }
    pid = os.getpid()
    for name, arr in arrays.items():
        tmp = os.path.join(out_dir, "%s.%d.tmp.npy" % (name, pid))
        np.save(tmp, arr)
        os.replace(tmp, os.path.join(out_dir, name + ".npy"))
    meta = dict(key=s.key, mean=s.mean.tolist(), std=s.std.tolist(),
                shape=[int(x) for x in s.label.shape],
                weight_coded=bool(wcode is not None),
                version=HEAD_CACHE_VERSION)
    tmp = os.path.join(out_dir, "meta.%d.tmp.json" % pid)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(tmp, done)
    return out_dir


class HeadStore:
    """Per-image cache of everything the patch sampler needs.

    :meth:`prepare` builds the on-disk cache (once per image, ever) in the main
    process; :meth:`get` then only opens memory maps, so a ``DataLoader``
    worker does no image decoding, no resizing and no skeletonisation.
    """

    def __init__(self, records, resize_longest, pred_dir=None, cache_dir=None,
                 mem_cache=8, synth_cuts=12, seed=0, head_cache_dir=None,
                 rebuild=False):
        self.records = list(records)
        self.resize_longest = resize_longest
        self.pred_dir = pred_dir
        self.mem_cache = int(mem_cache)
        self.synth_cuts = int(synth_cuts)
        self.seed = int(seed)
        self.rebuild = bool(rebuild)
        self._mem: Dict[int, HeadSample] = {}
        self._order: List[int] = []
        self.prob_source = "pred_dir" if pred_dir else "synthetic"
        self.head_cache_dir = head_cache_dir or os.path.join(
            HEAD_CACHE_ROOT, "ds_%s" % resize_longest)
        self._dirs = [
            os.path.join(self.head_cache_dir,
                         _cache_id(r, resize_longest, pred_dir,
                                   ("synth:%d:%d" % (self.seed, i))
                                   if self.prob_source == "synthetic" else "p"))
            for i, r in enumerate(self.records)]

    def __len__(self):
        return len(self.records)

    # -- cache construction (main process, before any worker forks) --------
    def prepare(self, verbose: bool = True, tag: str = "") -> float:
        t0 = time.time()
        n_built = 0
        for i, rec in enumerate(self.records):
            d = self._dirs[i]
            if os.path.exists(os.path.join(d, "meta.json")) and not self.rebuild:
                continue
            build_image_cache(rec, self.resize_longest, self.pred_dir, d,
                              idx=i, seed=self.seed, synth_cuts=self.synth_cuts,
                              force=self.rebuild)
            n_built += 1
            if verbose:
                print("  [cache%s] %3d/%3d %s  (%.1fs)"
                      % (tag, i + 1, len(self.records),
                         os.path.splitext(os.path.basename(
                             str(rec["image_path"])))[0], time.time() - t0),
                      flush=True)
        dt = time.time() - t0
        if verbose:
            print("[cache%s] %d/%d images built in %.1fs -> %s"
                  % (tag, n_built, len(self.records), dt, self.head_cache_dir),
                  flush=True)
        return dt

    # -- access ------------------------------------------------------------
    def _open(self, idx: int) -> HeadSample:
        d = self._dirs[idx]
        if not os.path.exists(os.path.join(d, "meta.json")):
            build_image_cache(self.records[idx], self.resize_longest,
                              self.pred_dir, d, idx=idx, seed=self.seed,
                              synth_cuts=self.synth_cuts)
        with open(os.path.join(d, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        arrays = {n: np.load(os.path.join(d, n + ".npy"), mmap_mode="r")
                  for n in _CACHE_ARRAYS}
        return HeadSample(meta["key"],
                          np.asarray(meta["mean"], np.float32),
                          np.asarray(meta["std"], np.float32), arrays)

    def get(self, idx: int) -> HeadSample:
        if idx in self._mem:
            return self._mem[idx]
        hs = self._open(idx)
        if self.mem_cache > 0:
            self._mem[idx] = hs
            self._order.append(idx)
            while len(self._order) > self.mem_cache:
                self._mem.pop(self._order.pop(0), None)
        return hs


class MultiHeadStore:
    """Several :class:`HeadStore` behind one flat index (the LODO union).

    ``HeadPatchDataset`` only ever calls ``len(store)`` and ``store.get(i)``, so
    a wrapper is enough: each source dataset keeps **its own geometry** (native
    resolution for DRIVE / CHASE / STARE, longest side 1536 for HRF / FIVES) and
    its own out-of-fold prediction directory, which is what makes a union train
    on comparable vessel scales instead of resampling every domain onto one.
    The crop size is shared and comes from the caller (the smallest patch of the
    participating datasets).
    """

    def __init__(self, stores, names):
        self.stores = list(stores)
        self.names = list(names)
        self._offsets = []
        n = 0
        for st in self.stores:
            self._offsets.append(n)
            n += len(st)
        self._n = n
        srcs = sorted({st.prob_source for st in self.stores})
        self.prob_source = srcs[0] if len(srcs) == 1 else "mixed(%s)" % ",".join(srcs)

    def prepare(self, verbose: bool = True, tag: str = "") -> float:
        return sum(st.prepare(verbose, "%s:%s" % (tag, n))
                   for st, n in zip(self.stores, self.names))

    def __len__(self):
        return self._n

    def get(self, idx: int):
        for k in range(len(self.stores) - 1, -1, -1):
            if idx >= self._offsets[k]:
                return self.stores[k].get(idx - self._offsets[k])
        raise IndexError(idx)


class HeadPatchDataset(Dataset):
    """Random patches with vessel oversampling; flips + rot90 only."""

    def __init__(self, store: HeadStore, patch: int, samples_per_epoch: int,
                 augment: bool = True, oversample_p: float = 0.7, seed: int = 0,
                 min_vessel_px: int = 32, deterministic: bool = False):
        self.store = store
        self.patch = int(patch)
        self.n = int(samples_per_epoch)
        self.augment = bool(augment)
        self.oversample_p = float(oversample_p)
        self.seed = int(seed)
        self.min_vessel_px = int(min_vessel_px)
        self.deterministic = bool(deterministic)
        self._rng = None

    def __len__(self):
        return self.n

    def _rng_for(self, idx):
        if self.deterministic:
            return np.random.default_rng([self.seed, int(idx), 4242])
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = int(info.id) if info is not None else 0
            self._rng = np.random.default_rng([self.seed, wid, 20260902])
        return self._rng

    def _locate(self, s: HeadSample, rng):
        h, w = s.image.shape[:2]
        ps = self.patch
        ph, pw = min(ps, h), min(ps, w)
        want = rng.random() < self.oversample_p
        best = None
        for _ in range(12 if want else 4):
            y = int(rng.integers(0, max(1, h - ph + 1)))
            x = int(rng.integers(0, max(1, w - pw + 1)))
            lab = s.label[y:y + ph, x:x + pw]
            fov = s.fov[y:y + ph, x:x + pw]
            score = (int(lab.sum()), int(fov.sum()))
            if best is None or score > best[0]:
                best = (score, y, x)
            if want and score[0] >= self.min_vessel_px:
                return y, x, ph, pw
            if not want and score[1] > 0.2 * ph * pw:
                return y, x, ph, pw
        _, y, x = best
        return y, x, ph, pw

    def __getitem__(self, idx: int):
        rng = self._rng_for(idx)
        si = int(rng.integers(0, len(self.store)))
        s = self.store.get(si)
        y, x, ph, pw = self._locate(s, rng)
        sl = (slice(y, y + ph), slice(x, x + pw))

        # mmap slices -> real arrays (only the patch is paged in and copied)
        img = np.asarray(s.image[sl])
        prob = np.asarray(s.prob[sl])
        lab = np.asarray(s.label[sl])
        fov = np.asarray(s.fov[sl])
        th = np.asarray(s.theta[sl])
        wc = np.asarray(s.wcode[sl])
        wt = _W_LEVELS[wc] if wc.dtype == np.uint8 else wc
        va = np.asarray(s.valid[sl])

        if self.augment:
            k = int(rng.integers(0, 4))
            fr = bool(rng.random() < 0.5)
            fc = bool(rng.random() < 0.5)

            def _t(a):
                a = np.rot90(a, k)
                if fr:
                    a = a[::-1]
                if fc:
                    a = a[:, ::-1]
                return np.ascontiguousarray(a)

            img, prob, lab, fov, th, wt, va = (_t(img), _t(prob), _t(lab),
                                               _t(fov), _t(th), _t(wt), _t(va))
            th = flip_rot_theta(th, k, fr, fc)

        ps = self.patch
        if img.shape[0] != ps or img.shape[1] != ps:
            def _pad(a, ch=False):
                a = a[:ps, :ps]
                p0, p1 = ps - a.shape[0], ps - a.shape[1]
                pad = ((0, max(0, p0)), (0, max(0, p1)))
                if ch:
                    pad = pad + ((0, 0),)
                return np.pad(a, pad, mode="constant")

            img = _pad(img, True)
            prob, lab, fov = _pad(prob), _pad(lab), _pad(fov)
            th, wt, va = _pad(th), _pad(wt), _pad(va)

        xi = segdata.normalize(img, s.mean, s.std)
        xin = np.concatenate([xi, prob[None].astype(np.float32)], axis=0)
        return {
            "x": torch.from_numpy(np.ascontiguousarray(xin)),
            "gt": torch.from_numpy((lab > 0).astype(np.float32))[None],
            "fov": torch.from_numpy((fov > 0).astype(np.float32))[None],
            "theta": torch.from_numpy(th.astype(np.float32))[None],
            "ori_w": torch.from_numpy(wt.astype(np.float32))[None],
            "valid": torch.from_numpy((va > 0).astype(np.float32))[None],
            "key": s.key,
        }


# --------------------------------------------------------------------------
def _worker_init(worker_id: int) -> None:
    """Keep each loader worker to a couple of threads.

    The workers now only slice memory maps, so they need almost no CPU -- but
    torch and OpenCV each default to one thread per core, and N workers times
    all cores is what turns a data loader into a CPU storm (and, with the
    biomarker baselines running alongside, starves the training process
    itself).
    """
    try:
        torch.set_num_threads(2)
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass


def make_loader(dataset, batch: int, workers: int, shuffle: bool = False,
                drop_last: bool = False, pin: bool = True) -> DataLoader:
    """DataLoader with persistent workers + prefetch when ``workers > 0``."""
    kw = dict(batch_size=batch, num_workers=int(workers), pin_memory=bool(pin),
              drop_last=bool(drop_last), shuffle=bool(shuffle))
    if int(workers) > 0:
        kw.update(persistent_workers=True, prefetch_factor=4,
                  worker_init_fn=_worker_init)
    return DataLoader(dataset, **kw)


# --------------------------------------------------------------------------
def run_epoch(model, loader, device, optim=None, scaler=None, amp=True,
              lambda_ori=1.0, max_batches=None):
    train = optim is not None
    model.train(train)
    agg = {"loss": 0.0, "bce": 0.0, "dice": 0.0, "ori": 0.0}
    n = 0
    for bi, b in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = b["x"].to(device, non_blocking=True)
        gt = b["gt"].to(device, non_blocking=True)
        fov = b["fov"].to(device, non_blocking=True)
        th = b["theta"].to(device, non_blocking=True)
        ow = b["ori_w"].to(device, non_blocking=True)
        va = b["valid"].to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=(amp and device.type == "cuda")):
                v, q = model(x)
            out = head_loss(v.float(), q.float(), gt, fov, th, ow, va,
                            lambda_ori=lambda_ori)
            if train:
                optim.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(out["loss"]).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                    scaler.step(optim)
                    scaler.update()
                else:
                    out["loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                    optim.step()
        for k in agg:
            agg[k] += float(out[k])
        n += 1
    return {k: v / max(n, 1) for k, v in agg.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="train the RiGR micro multi-task head")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--datasets", default=None,
                    help="comma-separated union of source datasets (the LODO "
                         "block trains one head on the three that are not held "
                         "out); each keeps its own resolution/patch geometry")
    ap.add_argument("--pred_dirs", default=None,
                    help="comma-separated --pred_dir, one per --datasets entry; "
                         "omit to use <pred_root>/<ds>/pred")
    ap.add_argument("--pred_root", default=os.path.join("runs", "seg_oof"),
                    help="root holding <ds>/pred, used with --datasets when "
                         "--pred_dirs is not given")
    ap.add_argument("--tag", default=None,
                    help="output subdirectory name for a --datasets run "
                         "(default lodo_<joined names>)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=None)
    ap.add_argument("--pred_dir", default=None,
                    help="src.seg.infer output directory holding prob/<key>.npy")
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--patch", type=int, default=None)
    ap.add_argument("--iters", type=int, default=100, help="batches per epoch")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda_ori", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4,
                    help="DataLoader workers; 0 reproduces the pre-cache "
                         "sampling RNG exactly (used for the parity check)")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch intra-op threads in the training process")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap the number of training images (FIVES has 510; "
                         "the S4 orchestrator passes its image_caps value so "
                         "step A and step B see the same images)")
    ap.add_argument("--head-cache-root", dest="head_cache_root",
                    default=HEAD_CACHE_ROOT,
                    help="root of the per-image target cache")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--split-seed", type=int, default=segdata.DEFAULT_SPLIT_SEED)
    args = ap.parse_args(argv)

    names = ([segdata.canon(x.strip()) for x in args.datasets.split(",") if x.strip()]
             if args.datasets else ([segdata.canon(args.dataset)] if args.dataset
                                    else []))
    if not names:
        raise SystemExit("give --dataset or --datasets")
    ds = names[0] if len(names) == 1 else (args.tag or "union_" + "_".join(names))
    out_dir = args.out or os.path.join("runs", "rigr_head", ds, "seed%d" % args.seed)
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_num_threads(max(1, int(args.threads)))
        import cv2 as _cv2

        _cv2.setNumThreads(max(1, int(args.threads)))
    except Exception:
        pass
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")

    cfgs = [segdata.dataset_cfg(n) for n in names]
    # a union crops at the smallest participating patch so every domain can
    # actually supply a full crop
    patch = int(args.patch or min(int(c["patch"]) for c in cfgs))
    longest = cfgs[0]["resize_longest"]
    batch = int(args.batch or max(2, min(segdata.DEFAULT_BATCH.get(n, 4)
                                          for n in names)))

    pred_dirs = {}
    if args.pred_dirs:
        parts = [x.strip() for x in args.pred_dirs.split(",")]
        if len(parts) != len(names):
            raise SystemExit("--pred_dirs must have one entry per --datasets entry")
        pred_dirs = dict(zip(names, parts))
    for n in names:
        if n in pred_dirs and pred_dirs[n]:
            continue
        if len(names) == 1 and args.pred_dir:
            pred_dirs[n] = args.pred_dir
        else:
            pred_dirs[n] = (args.pred_dir if args.pred_dir and len(names) == 1
                            else os.path.join(args.pred_root, n, "pred"))

    tr_stores, va_stores, n_tr, n_va = [], [], 0, 0
    for n, c in zip(names, cfgs):
        lo = c["resize_longest"]
        recs_n = segdata.get_records(n)
        tr_n, va_n, _te_n, _info_n = segdata.make_splits(recs_n, args.split_seed)
        if args.limit:
            tr_n = tr_n[: int(args.limit)]
        cache_n = (os.path.join("runs", "_cache", "%s_%s" % (n, lo))
                   if lo is not None else None)
        pd_n = pred_dirs.get(n)
        if pd_n and not os.path.isdir(pd_n):
            print("[WARN] %s: prediction dir %r not found; P falls back to the "
                  "SYNTHETIC source for this dataset" % (n, pd_n), flush=True)
            pd_n = None
        hc = os.path.join(args.head_cache_root, "%s_%s" % (n, lo))
        tr_stores.append(HeadStore(tr_n, lo, pd_n, cache_n, mem_cache=1024,
                                   seed=args.seed, head_cache_dir=hc,
                                   rebuild=args.rebuild_cache))
        va_stores.append(HeadStore(va_n, lo, pd_n, cache_n, mem_cache=1024,
                                   seed=args.seed, head_cache_dir=hc,
                                   rebuild=args.rebuild_cache))
        n_tr += len(tr_n)
        n_va += len(va_n)
    if len(names) == 1:
        st_tr, st_va = tr_stores[0], va_stores[0]
    else:
        st_tr = MultiHeadStore(tr_stores, names)
        st_va = MultiHeadStore(va_stores, names)
    tr, va = [None] * n_tr, [None] * n_va   # only their lengths are used below

    # Build the per-image target cache ONCE, in the main process, before any
    # worker forks: from here on a patch fetch is an mmap slice.
    t_cache = st_tr.prepare(tag="/train") + st_va.prepare(tag="/val")

    dl_tr = make_loader(
        HeadPatchDataset(st_tr, patch, args.iters * batch, augment=True,
                         seed=args.seed),
        batch, args.workers, drop_last=True)
    dl_va = make_loader(
        HeadPatchDataset(st_va, patch, max(8, 2 * batch), augment=False,
                         seed=args.seed + 1, deterministic=True),
        batch, 0)

    model = MicroUNet(in_channels=4, k_bins=K_BINS).to(device)
    n_par = model.num_parameters()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    amp = not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and device.type == "cuda"))

    config = dict(dataset=ds, datasets=names, seed=args.seed, epochs=args.epochs,
                  patch=patch, resize_longest=longest, batch=batch, lr=args.lr,
                  in_channels=4, k_bins=K_BINS, base_channels=16, num_levels=4,
                  max_channels=96, lambda_ori=args.lambda_ori,
                  prob_source=st_tr.prob_source,
                  pred_dir=args.pred_dir or "", pred_dirs=pred_dirs,
                  n_parameters=n_par, augmentation="flip+rot90 (axial-exact)",
                  split_seed=args.split_seed, n_train=len(tr), n_val=len(va),
                  workers=int(args.workers), limit=args.limit,
                  head_cache_root=args.head_cache_root,
                  cache_build_seconds=round(t_cache, 1))
    print(json.dumps(config, indent=2), flush=True)
    if st_tr.prob_source == "synthetic":
        print("[WARN] no --pred_dir: P is SYNTHETIC (blurred, randomly severed "
              "GT). Fine for a smoke test, not for a reported result.", flush=True)

    hist, best = [], float("inf")
    t0 = time.time()
    for ep in range(int(args.epochs)):
        for g in optim.param_groups:
            g["lr"] = args.lr * (1 - ep / max(1, args.epochs)) ** 0.9
        tr_m = run_epoch(model, dl_tr, device, optim, scaler, amp, args.lambda_ori)
        va_m = run_epoch(model, dl_va, device, None, None, amp, args.lambda_ori)
        hist.append(dict(epoch=ep, lr=optim.param_groups[0]["lr"],
                         **{"train_" + k: v for k, v in tr_m.items()},
                         **{"val_" + k: v for k, v in va_m.items()}))
        print("ep %3d  lr %.2e | train loss %.4f (bce %.4f dice %.4f ori %.4f) | "
              "val loss %.4f (ori %.4f)  %.1fs"
              % (ep, optim.param_groups[0]["lr"], tr_m["loss"], tr_m["bce"],
                 tr_m["dice"], tr_m["ori"], va_m["loss"], va_m["ori"],
                 time.time() - t0), flush=True)
        ck = dict(model=model.state_dict(), config=config, epoch=ep,
                  metrics=hist[-1])
        torch.save(ck, os.path.join(out_dir, "last.pt"))
        if va_m["loss"] < best:
            best = va_m["loss"]
            torch.save(ck, os.path.join(out_dir, "best.pt"))
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(dict(config=config, history=hist, best_val_loss=best,
                       seconds=time.time() - t0), f, indent=2)
    print("done: %d params, best val loss %.4f, %.1fs -> %s"
          % (n_par, best, time.time() - t0, out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
