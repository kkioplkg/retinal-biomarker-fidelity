"""
Loader for the *external* fundus datasets used by E3 (downstream validity).

These sets carry **disease labels but no vessel masks**. They exist so the
biomarker -> disease question can be asked at a sample size the mask-bearing
sets (HRF n=45, FIVES n=800) cannot reach. Nothing here is ever used for
segmentation training or for tuning a segmenter: the segmenters are frozen
before any external label is touched (DECISIONS.md, 2026-09-17 00:30).

    load_external(name) -> List[dict] with keys
        dataset     : str        dataset key
        image_path  : str        absolute path to the fundus image
        image_id    : str        unique per-sample id
        subject_id  : str        patient identifier (both eyes of one ODIR
                                 patient share it); the CV grouping key
        split       : str        'train' / 'test' where the set has an official
                                 split, otherwise 'all'
        label       : int|str    the primary target (see below)
        labels      : dict       every label the set provides
        fov_path    : str        auto-generated field-of-view mask
        eye         : str|None   'left' / 'right' when known
        native_size : (w, h)     only when the resolution cache has been built

Primary target per dataset
    aptos2019   label = DR grade 0-4                     (ordinal; QWK)
    idrid       label = DR grade 0-4, plus DME risk 0-2   (ordinal; QWK)
    messidor2   label = DR grade 0-4                     (ordinal; QWK)
    odir5k      label = one of 8 ocular-disease classes  (macro-AUC)

Raw data lives in  exp/data/external/<name>/raw/  and is never modified.
FOV masks are generated with the project's own method (datasets._generate_fov,
red channel at 10% of max -> closing -> fill -> largest component) and cached
under exp/data/external/<name>/fov/, exactly as for the segmentation sets.

Resolution matters for this project (the measurement-reliability mechanism was
not reproducible on low-resolution DRIVE), so every record can carry its native
size; `record_resolutions(name)` builds the cache once per dataset and
`load_external(..., with_size=True)` attaches it.
"""
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .datasets import _generate_fov, read_image  # noqa: F401  (read_image re-exported)

_HERE = Path(__file__).resolve()
EXP_ROOT = _HERE.parents[2]                       # .../medical1/exp
EXTERNAL_ROOT = EXP_ROOT / "data" / "external"

EXTERNAL_DATASETS = ["aptos2019", "idrid", "messidor2", "odir5k"]

# Datasets whose native resolution is high enough for the measurement mechanism
# to exist at all. ODIR is distributed here only as 512x512 thumbnails, which is
# below that scale, so it is a sensitivity/secondary set -- never a primary
# endpoint. See EXTERNAL_DATASETS.md.
PRIMARY_EXTERNAL = ["aptos2019", "idrid", "messidor2"]
SECONDARY_EXTERNAL = ["odir5k"]

# APTOS / IDRiD / Messidor-2 all use the same 5-point international DR scale.
DR_GRADES = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}
# "Referable DR" is the standard binary collapse of that scale.
REFERABLE_THRESHOLD = 2

ODIR_CLASSES = {
    "normal": "N",
    "diabetes": "D",
    "glaucoma": "G",
    "cataract": "C",
    "age_related_macular_degeneration": "A",
    "hypertension": "H",
    "pathological_myopia": "M",
    "other_diseases_abnormalities": "O",
}


def _p(*parts) -> str:
    return str(Path(*parts))


def raw_dir(name: str) -> Path:
    return EXTERNAL_ROOT / name / "raw"


# ---------------------------------------------------------------- FOV masks
def _fov_for(name: str, image_id: str) -> str:
    """Where this image's FOV mask lives (it may not exist yet)."""
    return _p(EXTERNAL_ROOT, name, "fov", image_id + "_fov.png")


def ensure_fov(rec: Dict, regenerate: bool = False) -> str:
    """Generate (once) and return the FOV mask for one record.

    Uses the project's existing generator (src/data/datasets._generate_fov) so
    the external sets and the segmentation sets get identical FOV semantics --
    the in-FOV restriction is part of every biomarker definition, and two
    different FOV rules would silently shift density and length.

    Generation is *not* done inside `load_external`: APTOS alone is 3,662
    images and ODIR 6,392, so building every mask on import would turn a
    metadata call into a ten-minute job. Callers that need pixels call this;
    `check_external.py --fov` pre-builds a whole set.
    """
    out = rec["fov_path"]
    if regenerate or not os.path.exists(out):
        # Write to a private temp file in the SAME directory and os.replace()
        # it into position, the way src/pivot/common.py::json_dump does.
        # Without this the mask is written directly to `out`, and two passes
        # over one dataset running concurrently both find it missing, both
        # generate it, and one can read the half-written file: imageio then
        # fails to recognise the PNG, falls back to another plugin, and the
        # job dies with something that looks nothing like an I/O error
        # (observed 2026-09-17, DECISIONS 17:23).  os.replace is atomic within
        # a filesystem, so a reader now sees either no file or a complete one.
        #
        # The temp name carries the pid -- unlike json_dump's fixed ".tmp",
        # which is safe only for a single writer.  A shared temp name would
        # just move the same race down one level.
        # The pid goes BEFORE the extension: _generate_fov saves through PIL,
        # which infers the format from the suffix, so a trailing ".tmp<pid>"
        # makes it raise "unknown file extension".
        os.makedirs(os.path.dirname(out), exist_ok=True)
        stem, ext = os.path.splitext(out)
        tmp = "%s.tmp%d%s" % (stem, os.getpid(), ext)
        try:
            _generate_fov(rec["image_path"], tmp)
            os.replace(tmp, out)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    return out


def ensure_fovs(name: str, regenerate: bool = False, limit: Optional[int] = None,
                verbose: bool = True) -> int:
    """Pre-build the FOV cache for a whole dataset; returns the number built."""
    recs = load_external(name)
    if limit:
        recs = recs[:limit]
    n = 0
    for i, r in enumerate(recs):
        if regenerate or not os.path.exists(r["fov_path"]):
            ensure_fov(r, regenerate=regenerate)
            n += 1
        if verbose and (i + 1) % 250 == 0:
            print("  fov %s %d/%d" % (name, i + 1, len(recs)), flush=True)
    return n


# ---------------------------------------------------------------- resolutions
def resolution_cache(name: str) -> Path:
    return EXTERNAL_ROOT / name / "raw" / "_resolutions.csv"


def record_resolutions(name: str, force: bool = False) -> Dict[str, Tuple[int, int]]:
    """Build (once) and return {image_id: (width, height)} for `name`.

    Only the image header is parsed -- PIL does not decode pixels for `.size` --
    so this is seconds even for APTOS' 8.6 GB.
    """
    from PIL import Image

    path = resolution_cache(name)
    if path.exists() and not force:
        with open(path, newline="") as fh:
            return {r["image_id"]: (int(r["width"]), int(r["height"]))
                    for r in csv.DictReader(fh)}

    sizes = {}
    for rec in load_external(name, with_size=False):
        with Image.open(rec["image_path"]) as im:
            sizes[rec["image_id"]] = im.size
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "width", "height"])
        for k, (ww, hh) in sorted(sizes.items()):
            w.writerow([k, ww, hh])
    return sizes


# ---------------------------------------------------------------- APTOS-2019
def _load_aptos2019() -> List[Dict]:
    """
    3,662 training images of the APTOS 2019 Blindness Detection competition,
    graded 0-4 on the international DR scale by Aravind Eye Hospital.

    Only the *train* split is here: the competition's test labels were never
    released, so the 1,928 test images carry no usable target.

    One image per row and no patient identifier is published, so image == subject.
    """
    root = raw_dir("aptos2019")
    out = []
    with open(root / "train.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            iid = row["id_code"]
            ip = _p(root, "train_images", iid + ".png")
            grade = int(row["diagnosis"])
            out.append(dict(
                dataset="aptos2019",
                image_path=ip, image_id=iid,
                subject_id=iid,                  # no patient id published
                split="train",
                label=grade,
                labels=dict(dr_grade=grade,
                            referable=int(grade >= REFERABLE_THRESHOLD)),
                fov_path=_fov_for("aptos2019", iid),
                eye=None,
            ))
    return out


# ---------------------------------------------------------------- IDRiD
_IDRID_SUB = "B. Disease Grading"
_IDRID_SPLIT_DIR = {"train": "a. Training Set", "test": "b. Testing Set"}
_IDRID_LABELS = {"train": "a. IDRiD_Disease Grading_Training Labels.csv",
                 "test": "b. IDRiD_Disease Grading_Testing Labels.csv"}


def _load_idrid() -> List[Dict]:
    """
    IDRiD part B (Disease Grading): 413 train + 103 test images at 4288x2848,
    each with a DR grade 0-4 and a risk-of-macular-edema grade 0-2.

    The official CSVs carry trailing empty columns ("Unnamed: 3" ...) and a
    trailing space in the "Risk of macular edema " header; both are normalised
    here rather than in every caller.

    Gotcha - IDRiD restarts its numbering in each split: the training set is
    IDRiD_001..IDRiD_413 and the testing set IDRiD_001..IDRiD_103, so the bare
    stem collides for 103 pairs of *different* images. `image_id` and
    `subject_id` are therefore split-prefixed ("train_IDRiD_001"), the same fix
    FIVES needed in datasets.py.
    """
    root = raw_dir("idrid") / _IDRID_SUB
    out = []
    for split in ("train", "test"):
        with open(root / "2. Groundtruths" / _IDRID_LABELS[split], newline="") as fh:
            for row in csv.DictReader(fh):
                row = {(k or "").strip(): (v or "").strip() for k, v in row.items()}
                stem = row.get("Image name", "")
                if not stem:
                    continue                      # trailing blank lines
                iid = split + "_" + stem          # numbering restarts per split
                ip = _p(root, "1. Original Images", _IDRID_SPLIT_DIR[split], stem + ".jpg")
                grade = int(row["Retinopathy grade"])
                dme = int(row["Risk of macular edema"])
                out.append(dict(
                    dataset="idrid",
                    image_path=ip, image_id=iid,
                    subject_id=iid,               # one image per examination
                    split=split,
                    label=grade,
                    labels=dict(dr_grade=grade, dme_risk=dme,
                                referable=int(grade >= REFERABLE_THRESHOLD)),
                    fov_path=_fov_for("idrid", iid),
                    eye=None,
                ))
    return out


# ---------------------------------------------------------------- Messidor-2
def _load_messidor2() -> List[Dict]:
    """
    1,744 Messidor-2 images with a 0-4 DR grade, materialised from the
    HuggingFace mirror into raw/images/ + raw/labels.csv by
    `python -m src.data.check_external --prepare messidor2`.

    Caveat recorded in EXTERNAL_DATASETS.md: the mirror ships images and grades
    only -- the original filenames (and therefore the 874 two-eye examinations)
    are lost, so image == subject here and no eye pairing is possible.
    """
    root = raw_dir("messidor2")
    out = []
    lab = root / "labels.csv"
    if not lab.exists():
        raise FileNotFoundError(
            "messidor2 not materialised yet; run "
            "`python -m src.data.check_external --prepare messidor2`")
    with open(lab, newline="") as fh:
        for row in csv.DictReader(fh):
            iid = Path(row["filename"]).stem
            ip = _p(root, "images", row["filename"])
            grade = int(row["dr_grade"])
            out.append(dict(
                dataset="messidor2",
                image_path=ip, image_id=iid,
                subject_id=iid,                   # examination id not published
                split="all",
                label=grade,
                labels=dict(dr_grade=grade,
                            referable=int(grade >= REFERABLE_THRESHOLD)),
                fov_path=_fov_for("messidor2", iid),
                eye=None,
            ))
    return out


# ---------------------------------------------------------------- ODIR-5K
def _load_odir5k() -> List[Dict]:
    """
    6,392 eye images from 3,358 ODIR-5K patients, one 8-way ocular-disease
    label per eye, plus age and sex.

    SECONDARY SET ONLY. This mirror is the Kaggle `preprocessed_images` folder:
    512x512 JPEG, not the native ~2976x1984. That is below the resolution at
    which the measurement-reliability mechanism was observable at all (it did
    not reproduce on DRIVE), so ODIR answers a sensitivity question, not a
    primary one.

    Both eyes of a patient share `subject_id`, so patient-level grouped CV must
    group on it -- eye-level splitting would leak.
    """
    root = raw_dir("odir5k")
    out = []
    with open(root / "labels.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            fn = row["filename"]
            iid = Path(fn).stem                   # e.g. 1234_left
            ip = _p(root, "images", fn)
            out.append(dict(
                dataset="odir5k",
                image_path=ip, image_id=iid,
                subject_id="odir_" + str(row["patient_id"]),
                split="all",
                label=row["label"],
                labels=dict(disease=row["label"],
                            disease_code=int(row["label_code"]),
                            abbrev=ODIR_CLASSES.get(row["label"], "?"),
                            age=int(row["age"]) if row["age"] not in ("", "None") else None,
                            sex=row["sex"]),
                fov_path=_fov_for("odir5k", iid),
                eye=row["eye"],
            ))
    return out


# ---------------------------------------------------------------- public API
_LOADERS = {
    "aptos2019": _load_aptos2019,
    "idrid": _load_idrid,
    "messidor2": _load_messidor2,
    "odir5k": _load_odir5k,
}

_ALIASES = {
    "aptos": "aptos2019", "aptos-2019": "aptos2019", "aptos_2019": "aptos2019",
    "odir": "odir5k", "odir-5k": "odir5k", "odir_5k": "odir5k",
    "messidor": "messidor2", "messidor-2": "messidor2", "messidor_2": "messidor2",
    "idrid": "idrid", "idrid_b": "idrid",
}


def load_external(name: str,
                  split: Optional[str] = None,
                  with_size: bool = False,
                  **kw) -> List[Dict]:
    """
    Records for external dataset `name` (see module docstring for the keys).

    split      : keep only 'train' / 'test' (APTOS is all-train; ODIR and
                 Messidor-2 have no official split and report 'all').
    with_size  : attach `native_size` from the resolution cache, building it on
                 first use.
    kw         : forwarded to the loader (currently none take arguments).

    `fov_path` is where the mask belongs, not a promise that it exists --
    call `ensure_fov(rec)` (or `ensure_fovs(name)`) before reading it.
    """
    key = _ALIASES.get(name.lower(), name.lower())
    if key not in _LOADERS:
        raise ValueError("unknown external dataset %r; available: %s"
                         % (name, EXTERNAL_DATASETS))
    recs = _LOADERS[key](**kw)
    if split is not None:
        recs = [r for r in recs if r["split"] == split]
    if with_size:
        sizes = record_resolutions(key)
        for r in recs:
            r["native_size"] = sizes.get(r["image_id"])
    return recs
