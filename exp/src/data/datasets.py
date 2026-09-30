"""
Unified retinal-vessel dataset loader.

load_dataset(name) -> List[dict] with keys:
    image_path  : str       absolute path to the fundus image
    label_path  : str       absolute path to the primary (1st observer) vessel label
    label2_path : str|None  2nd observer label, when the dataset provides one
    fov_path    : str|None  field-of-view mask (official, or auto-generated)
    split       : "train" | "test"
    subject_id  : str       subject identifier (L/R eyes of one person share it)
    image_id    : str       unique per-sample id

Datasets: drive, chasedb1, hrf, fives, stare, maplesdr, fundusavseg
Raw data lives in  exp/data/<name>/raw/  and is never modified.
Auto-generated FOV masks are written to  exp/data/<name>/fov/.

`maplesdr` and `fundusavseg` are the pre-registered independent replication
cohorts added on 2026-09-17; they are **not** in `DATASETS` (the five sets the
project trains and reports on) so that adding them cannot silently change any
existing loop. Ask for them by name, or use `ALL_DATASETS`.
"""
from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# ---------------------------------------------------------------- paths
_HERE = Path(__file__).resolve()
EXP_ROOT = _HERE.parents[2]              # .../medical1/exp
DATA_ROOT = EXP_ROOT / "data"
RUNS_ROOT = EXP_ROOT / "runs"

DATASETS = ["drive", "chasedb1", "hrf", "fives", "stare"]
# Independent replication cohorts (2026-09-17). Kept out of DATASETS on purpose.
REPLICATION_DATASETS = ["maplesdr", "fundusavseg"]
ALL_DATASETS = DATASETS + REPLICATION_DATASETS

FIVES_DIR = "FIVES A Fundus Image Dataset for AI-based Vessel Segmentation"

# FIVES disease-class code -> readable name
FIVES_CLASSES = {"A": "AMD", "D": "DR", "G": "Glaucoma", "N": "Normal"}

# Fundus-AVSeg uses the same single-letter disease codes as FIVES.
FUNDUSAVSEG_CLASSES = dict(FIVES_CLASSES)


def _p(*parts) -> str:
    return str(Path(*parts))


# ---------------------------------------------------------------- FOV masks
FOV_THRESHOLD_FRACTION = 0.10
# Longest side (px) at which the FOV morphological closing runs at full
# resolution; larger images are decimated for the closing step only.
MAX_MORPH_SIDE = 1024


def _generate_fov(image_path: str, out_path: str, close_radius: int = 9) -> str:
    """
    Build a field-of-view mask for datasets that ship none (CHASE_DB1, FIVES, STARE).

    Method: threshold the red channel at 10% of its maximum, then morphological
    closing to seal vessel shadows and lesions, fill holes, and keep the largest
    connected component. Saved as an 8-bit PNG with values {0, 255}.

    Why a low fixed fraction and not Otsu: the fundus surround is essentially
    black, so the task is separating "not black" from "black", not splitting two
    balanced populations. Otsu places the threshold between the dim retinal
    periphery and the bright disc centre, which eats the periphery on darker
    images. Measured over every CHASE_DB1 and STARE image and a FIVES sample,
    with "fraction of ground-truth vessel pixels falling inside the mask" as the
    correctness criterion:

        CHASE_DB1  Otsu: coverage sd 0.051, GT-in-FOV 0.967 (worst image 0.864)
        CHASE_DB1  10% : coverage sd 0.011, GT-in-FOV 0.998 (worst image 0.994)
        STARE      Otsu: coverage sd 0.053, GT-in-FOV 0.968 (worst image 0.734)
        STARE      10% : coverage sd 0.050, GT-in-FOV 1.000 (worst image 1.000)
        FIVES      10% : coverage sd 0.014, GT-in-FOV 0.998 (worst image 0.986)

    Under Otsu, up to 13.6% of annotated vessel pixels on the worst CHASE image
    fell outside the FOV and would have been silently excluded from every
    in-FOV metric. A 15% fraction regresses badly on FIVES (GT-in-FOV 0.788 on
    the worst image); using the 99th percentile instead of the max is identical
    on CHASE_DB1/FIVES and slightly worse on STARE.
    """
    from PIL import Image
    from skimage.morphology import disk, closing
    from scipy import ndimage as ndi

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = read_image(image_path)
    red = img if img.ndim == 2 else img[..., 0]
    red = red.astype(np.float32)

    m = red > FOV_THRESHOLD_FRACTION * float(red.max())

    # Morphological closing costs O(N * r^2) and dominates the runtime on large
    # images (4.4 s of the 5.5 s per 2048x2048 FIVES image). The FOV boundary is
    # a smooth circle, so the closing is run on a decimated copy and OR-ed back
    # at full resolution. OR-ing means the result is always a superset of the
    # un-closed mask, so this can only add pixels, never erode the retina away.
    step = max(1, int(max(m.shape) // MAX_MORPH_SIDE))
    if step > 1:
        small = closing(m[::step, ::step], disk(max(1, round(close_radius / step))))
        up = np.repeat(np.repeat(small, step, 0), step, 1)[:m.shape[0], :m.shape[1]]
        if up.shape != m.shape:                       # pad if decimation truncated
            pad = [(0, m.shape[i] - up.shape[i]) for i in range(2)]
            up = np.pad(up, pad, mode="edge")
        m = m | up
    else:
        m = closing(m, disk(close_radius))

    m = ndi.binary_fill_holes(m)
    # No small-object filter is needed: the largest-component step below already
    # discards every speck outside the fundus disc.

    lab, n = ndi.label(m)
    if n > 1:                                  # keep largest component only
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)

    Image.fromarray(m.astype(np.uint8) * 255, mode="L").save(out_path)
    return out_path


def _fov_for(name: str, image_path: str, image_id: str, regenerate: bool = False) -> str:
    out = _p(DATA_ROOT, name, "fov", image_id + "_fov.png")
    if regenerate or not os.path.exists(out):
        _generate_fov(image_path, out)
    return out


# ---------------------------------------------------------------- DRIVE
def _load_drive() -> List[Dict]:
    """Official 20 train / 20 test split. FOV masks and 2nd observer ship with it."""
    root = DATA_ROOT / "drive" / "raw"
    out = []
    for split, sub in (("train", "training"), ("test", "test")):
        for ip in sorted(glob.glob(_p(root, sub, "images", "*.tif"))):
            stem = Path(ip).stem                     # 21_training / 01_test
            num = stem.split("_")[0]                 # 21 / 01
            lab = _p(root, sub, "1st_manual", num + "_manual1.gif")
            lab2 = _p(root, "test", "2nd_manual", num + "_manual2.gif") if split == "test" else None
            fov = _p(root, sub, "mask", stem + "_mask.gif")
            out.append(dict(
                image_path=ip,
                label_path=lab,
                label2_path=lab2 if (lab2 and os.path.exists(lab2)) else None,
                fov_path=fov if os.path.exists(fov) else None,
                split=split, subject_id=num, image_id=stem,
            ))
    return out


# ---------------------------------------------------------------- CHASE_DB1
def _load_chasedb1(regenerate_fov: bool = False) -> List[Dict]:
    """
    28 images = 14 subjects x 2 eyes (L/R). Both eyes of a subject share subject_id.
    Convention: first 20 images train (subjects 01-10), last 8 test (subjects 11-14).
    This split is subject-disjoint. Two observers: _1stHO / _2ndHO. No official FOV.
    """
    root = DATA_ROOT / "chasedb1" / "raw"
    out = []
    for i, ip in enumerate(sorted(glob.glob(_p(root, "Image_*.jpg")))):
        stem = Path(ip).stem                        # Image_01L
        subject = stem.replace("Image_", "")[:2]    # 01
        out.append(dict(
            image_path=ip,
            label_path=_p(root, stem + "_1stHO.png"),
            label2_path=_p(root, stem + "_2ndHO.png"),
            fov_path=_fov_for("chasedb1", ip, stem, regenerate_fov),
            split="train" if i < 20 else "test",
            subject_id=subject, image_id=stem,
        ))
    return out


# ---------------------------------------------------------------- HRF
def _load_hrf() -> List[Dict]:
    """
    45 images: 15 healthy (h), 15 glaucoma (g), 15 diabetic retinopathy (dr).
    Convention: first 5 of each class -> train (15), remaining 10 of each -> test (30).
    Official FOV masks included; single observer.
    """
    root = DATA_ROOT / "hrf" / "raw"
    # HRF ships the `dr` images as .JPG and the rest as .jpg. On Windows glob is
    # case-insensitive, so globbing both patterns yields every file twice --
    # enumerate the directory once and filter by lowercased extension instead.
    imgs = sorted(
        _p(root, "images", fn)
        for fn in os.listdir(_p(root, "images"))
        if fn.lower().endswith(".jpg")
    )
    out = []
    for ip in imgs:
        stem = Path(ip).stem                        # 01_dr
        num, cls = stem.split("_")
        out.append(dict(
            image_path=ip,
            label_path=_p(root, "manual1", stem + ".tif"),
            label2_path=None,
            fov_path=_p(root, "mask", stem + "_mask.tif"),
            split="train" if int(num) <= 5 else "test",
            subject_id=stem, image_id=stem, disease=cls,
        ))
    return out


# ---------------------------------------------------------------- FIVES
def _load_fives(regenerate_fov: bool = False) -> List[Dict]:
    """
    Official 600 train / 200 test. Filename <n>_<C>.png where C in {A,D,G,N}
    = AMD / DR / Glaucoma / Normal. Single observer, no official FOV.
    """
    root = DATA_ROOT / "fives" / "raw" / FIVES_DIR
    out = []
    for split in ("train", "test"):
        for ip in sorted(glob.glob(_p(root, split, "Original", "*.png"))):
            stem = Path(ip).stem                    # 100_A
            if "_" not in stem:
                continue                            # skip stray files (Thumbs.db etc.)
            cls = stem.split("_")[-1]
            if cls not in FIVES_CLASSES:
                continue
            image_id = split + "_" + stem
            out.append(dict(
                image_path=ip,
                label_path=_p(root, split, "Ground truth", stem + ".png"),
                label2_path=None,
                fov_path=_fov_for("fives", ip, image_id, regenerate_fov),
                # subject_id must be the split-prefixed id, not the bare stem:
                # FIVES restarts its numbering in each split, so `train/10_A.png`
                # and `test/10_A.png` are different images that would otherwise
                # share a subject_id (50 such collisions) and be wrongly treated
                # as the same subject by subject-grouped statistics.
                split=split, subject_id=image_id, image_id=image_id,
                disease=FIVES_CLASSES[cls],
            ))
    return out


# ---------------------------------------------------------------- STARE
# Fixed 10/10 split (first 10 / last 10 by image number) so results are
# reproducible; STARE has no official split and is often run leave-one-out.
STARE_TRAIN = ["im0001", "im0002", "im0003", "im0004", "im0005",
               "im0044", "im0077", "im0081", "im0082", "im0139"]
STARE_TEST = ["im0162", "im0163", "im0235", "im0236", "im0239",
              "im0240", "im0255", "im0291", "im0319", "im0324"]


def _load_stare(regenerate_fov: bool = False) -> List[Dict]:
    """20 images, two observers (ah = primary, vk = secondary). No official FOV."""
    root = DATA_ROOT / "stare" / "raw"
    out = []
    for stem in STARE_TRAIN + STARE_TEST:
        ip = _p(root, "images", stem + ".ppm")
        out.append(dict(
            image_path=ip,
            label_path=_p(root, "labels-ah", stem + ".ah.ppm"),
            label2_path=_p(root, "labels-vk", stem + ".vk.ppm"),
            fov_path=_fov_for("stare", ip, stem, regenerate_fov),
            split="train" if stem in STARE_TRAIN else "test",
            subject_id=stem, image_id=stem,
        ))
    return out


# ---------------------------------------------------------------- MAPLES-DR
def _maplesdr_grades() -> Dict[str, Dict[str, str]]:
    """
    Consensus DR ('R0'..'R4A') and ME ('M0'..'M6') grade per image, read once
    from the MAPLES-DR AdditionalData workbook.
    """
    global _MAPLES_GRADES
    if _MAPLES_GRADES is None:
        import pandas as pd
        xls = _p(DATA_ROOT, "maplesdr", "raw", "AdditionalData", "diagnosis_infos.xls")
        out: Dict[str, Dict[str, str]] = {}
        for sheet, key in (("DR", "dr_grade"), ("ME", "me_grade")):
            df = pd.read_excel(xls, sheet_name=sheet)
            for name, val in zip(df["name"], df["Consensus"]):
                out.setdefault(str(name), {})[key] = str(val)
        _MAPLES_GRADES = out
    return _MAPLES_GRADES


_MAPLES_GRADES: Optional[Dict[str, Dict[str, str]]] = None


def _maplesdr_image_index() -> Dict[str, Dict[str, str]]:
    """
    image_id -> {"path": ..., "source": "messidor" | "messidor2-mirror"}.

    The MAPLES-DR figshare deposit contains labels only; the fundus images
    belong to the MESSIDOR consortium (registration form at
    adcis.net/en/third-party/messidor). Two sources are accepted, in order:

    1. ``exp/data/maplesdr/raw/images/<name>.tif`` - the official MESSIDOR
       download, once someone has completed the form. Always preferred.
    2. the project's existing Messidor-2 mirror, for the 162 of 198 MAPLES-DR
       images that are also in Messidor-2. The mapping is established by
       ``src/data/maples_overlap.py`` (vessel-mask/green-channel correlation,
       accepted matches lead the candidate pool by 14.9-87.0 sd and form a
       bijection) and cached in ``exp/data/maplesdr/messidor2_match.csv``.

    Images that neither source provides are reported with an empty path; the
    loader drops them unless ``include_unresolved=True``.
    """
    import csv as _csv
    idx: Dict[str, Dict[str, str]] = {}
    official = _p(DATA_ROOT, "maplesdr", "raw", "images")
    if os.path.isdir(official):
        for fn in os.listdir(official):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                idx[stem] = dict(path=_p(official, fn), source="messidor")
    match_csv = _p(DATA_ROOT, "maplesdr", "messidor2_match.csv")
    if os.path.exists(match_csv):
        with open(match_csv, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                name, mid = row["maples_name"], row.get("messidor2_id") or ""
                if name in idx or not mid or row.get("matched", "").lower() != "true":
                    continue
                p = _p(DATA_ROOT, "external", "messidor2", "raw", "images", mid + ".jpg")
                if os.path.exists(p):
                    idx[name] = dict(path=p, source="messidor2-mirror", messidor2_id=mid)
    return idx


def _load_maplesdr(regenerate_fov: bool = False,
                   include_unresolved: bool = False) -> List[Dict]:
    """
    MAPLES-DR: 198 MESSIDOR images with retinologist vessel segmentations.
    Official split 138 train / 60 test. Labels are CC BY 4.0 (figshare
    10.6084/m9.figshare.24328660); the images come from MESSIDOR (see
    `_maplesdr_image_index`), so a record whose image cannot be resolved is
    dropped unless `include_unresolved=True`.

    The record carries `disease` (the consensus DR grade R0..R4A), `me_grade`,
    and `image_source` so that "which pixels did this come from" is never
    guesswork downstream.
    """
    root = DATA_ROOT / "maplesdr" / "raw"
    grades = _maplesdr_grades()
    index = _maplesdr_image_index()
    out = []
    for split in ("train", "test"):
        for lp in sorted(glob.glob(_p(root, split, "Vessels", "*.png"))):
            stem = Path(lp).stem                      # 20051020_44923_0100_PP
            ent = index.get(stem)
            if ent is None and not include_unresolved:
                continue
            ip = ent["path"] if ent else None
            g = grades.get(stem, {})
            out.append(dict(
                image_path=ip,
                label_path=lp,
                label2_path=None,
                fov_path=_fov_for("maplesdr", ip, stem, regenerate_fov) if ip else None,
                # MESSIDOR publishes no patient id, so one image = one subject,
                # exactly as for the Messidor-2 external cohort.
                split=split, subject_id=stem, image_id=stem,
                disease=g.get("dr_grade"), me_grade=g.get("me_grade"),
                image_source=ent["source"] if ent else None,
                messidor2_id=(ent or {}).get("messidor2_id"),
            ))
    return out


# ---------------------------------------------------------------- Fundus-AVSeg
FUNDUSAVSEG_DIR = "Fundus-AVSeg"


def _fundusavseg_vessel_mask(ann_path: str, out_path: str) -> str:
    """
    Fundus-AVSeg ships a 5-colour artery/vein map, not a binary vessel mask.
    The binary vessel ground truth is the union of the four non-background
    classes - red (255,0,0) artery, blue (0,0,255) vein, green (0,255,0)
    artery/vein crossing, white (255,255,255) vessel of uncertain class.
    Cached as an 8-bit {0,255} PNG next to the generated FOV masks; the raw
    tree is left untouched.
    """
    from PIL import Image
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    a = read_image(ann_path)
    if a.ndim == 2:
        m = a > 0
    else:
        m = a[..., :3].any(axis=-1)
    Image.fromarray(m.astype(np.uint8) * 255, mode="L").save(out_path)
    return out_path


_AVSEG_META: Optional[Dict[str, Dict[str, str]]] = None


def _fundusavseg_meta() -> Dict[str, Dict[str, str]]:
    """image_id -> {'eye': 'left'|'right', 'quality': 'High-quality'|'Low-quality'}."""
    global _AVSEG_META
    if _AVSEG_META is None:
        import pandas as pd
        xlsx = _p(DATA_ROOT, "fundusavseg", "raw", FUNDUSAVSEG_DIR, "metadata.xlsx")
        df = pd.read_excel(xlsx, sheet_name="Sheet1")
        _AVSEG_META = {
            Path(str(n)).stem: dict(eye=str(e), quality=str(q))
            for n, e, q in zip(df["image name"], df["eye id"], df["image quality"])
        }
    return _AVSEG_META


def _load_fundusavseg(regenerate_fov: bool = False,
                      regenerate_labels: bool = False) -> List[Dict]:
    """
    Fundus-AVSeg: 100 colour fundus images from Shenzhen Eye Hospital with
    pixel-level artery/vein annotation, official 80/20 split, disease code in
    the filename (`001_G.png`; N/D/A/G as in FIVES). CC BY 4.0, figshare
    10.6084/m9.figshare.27938034.

    Two resolutions: 2656x1992 (21 images) and 1280x1280 (79). `native_size`
    is on every record so the high-resolution stratum can be selected without
    re-opening the files.
    """
    root = DATA_ROOT / "fundusavseg" / "raw" / FUNDUSAVSEG_DIR
    meta = _fundusavseg_meta()
    splits = {}
    for split, fn in (("train", "training.txt"), ("test", "testing.txt")):
        with open(_p(root, fn), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    splits[Path(line).stem] = split
    out = []
    for ip in sorted(glob.glob(_p(root, "images", "*.png"))):
        stem = Path(ip).stem                          # 001_G
        cls = stem.split("_")[-1]
        ann = _p(root, "annotation", stem + ".png")
        lab = _p(DATA_ROOT, "fundusavseg", "labels", stem + ".png")
        if regenerate_labels or not os.path.exists(lab):
            _fundusavseg_vessel_mask(ann, lab)
        out.append(dict(
            image_path=ip,
            label_path=lab,
            label2_path=None,
            av_path=ann,                              # the 5-colour A/V map
            fov_path=_fov_for("fundusavseg", ip, stem, regenerate_fov),
            split=splits.get(stem, "train"),
            subject_id=stem, image_id=stem,
            disease=FUNDUSAVSEG_CLASSES.get(cls, cls),
            eye=meta.get(stem, {}).get("eye"),
            # 'High-quality' / 'Low-quality', graded by the dataset authors.
            # A real per-image quality label, which none of the five training
            # sets provides, so it can serve as an acquisition covariate.
            quality=meta.get(stem, {}).get("quality"),
        ))
    return out


# ---------------------------------------------------------------- public API
_LOADERS = {
    "drive": _load_drive,
    "chasedb1": _load_chasedb1,
    "hrf": _load_hrf,
    "fives": _load_fives,
    "stare": _load_stare,
    "maplesdr": _load_maplesdr,
    "fundusavseg": _load_fundusavseg,
}

_ALIASES = {"chase": "chasedb1", "chase_db1": "chasedb1", "chasedb": "chasedb1",
            "maples": "maplesdr", "maples-dr": "maplesdr", "maples_dr": "maplesdr",
            "avseg": "fundusavseg", "fundus-avseg": "fundusavseg",
            "fundus_avseg": "fundusavseg"}


def load_dataset(name: str, split: Optional[str] = None, **kw) -> List[Dict]:
    """
    Return the record list for `name` (optionally filtered to split 'train'/'test').
    Datasets lacking an official FOV mask get one generated on first call and
    cached under exp/data/<name>/fov/ (pass regenerate_fov=True to rebuild).
    """
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in _LOADERS:
        raise ValueError("unknown dataset %r; available: %s" % (name, ALL_DATASETS))
    recs = _LOADERS[key](**kw)
    if split is not None:
        recs = [r for r in recs if r["split"] == split]
    return recs


def read_image(path: str) -> np.ndarray:
    """
    Read any format used here (tif/gif/ppm/png/jpg) as a numpy array.

    Pillow is the primary reader: it decodes DRIVE's LZW-compressed TIFFs and
    HRF's PackBits TIFFs out of the box, whereas imageio routes TIFFs through
    `tifffile`, which raises
    `ValueError: <COMPRESSION.LZW: 5> requires the 'imagecodecs' package`
    unless the optional `imagecodecs` dependency is installed. imageio is kept
    as a fallback for anything Pillow cannot open.
    """
    from PIL import Image
    try:
        with Image.open(path) as im:
            return np.asarray(im)
    except Exception:
        import imageio.v3 as iio
        return np.asarray(iio.imread(path))


def read_binary(path: str) -> np.ndarray:
    """Read a label / FOV mask and return a boolean array."""
    a = read_image(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a > (0.5 * float(a.max()) if a.max() > 1 else 0.5)


if __name__ == "__main__":
    for d in ALL_DATASETS:
        r = load_dataset(d)
        ntr = sum(x["split"] == "train" for x in r)
        missing = sum(
            not os.path.exists(x["image_path"]) or not os.path.exists(x["label_path"])
            for x in r
        )
        print("%-10s total=%4d train=%4d test=%4d subjects=%3d missing_files=%d"
              % (d, len(r), ntr, len(r) - ntr,
                 len({x["subject_id"] for x in r}), missing))
