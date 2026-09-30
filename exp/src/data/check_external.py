"""
Sanity check (and one-time materialisation) for the E3 external datasets.

These sets have disease labels and NO vessel masks, so the checks differ from
check_data.py: instead of label/observer agreement it verifies counts against
the published label distributions, reports the native-resolution distribution
(APTOS varies by a factor of ~50 in pixel count), and confirms the generated
FOV masks are sane (a plausible fraction of the frame, one component, and the
bright disc actually inside it).

Usage
    python -m src.data.check_external                      # check every set
    python -m src.data.check_external aptos2019 idrid      # check some
    python -m src.data.check_external --prepare messidor2  # parquet -> files
    python -m src.data.check_external --fov aptos2019      # pre-build FOV cache
    python -m src.data.check_external --resolutions        # (re)build size cache

Run from exp/ with the project interpreter, e.g.
    D:/Anaconda/envs/medical1/python.exe -m src.data.check_external
"""
from __future__ import annotations

import collections
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from src.data.external import (  # noqa: E402
    EXTERNAL_DATASETS, EXTERNAL_ROOT, ODIR_CLASSES, ensure_fov, ensure_fovs,
    load_external, raw_dir, record_resolutions,
)
from src.data.datasets import RUNS_ROOT, read_binary, read_image  # noqa: E402

OUT_DIR = os.path.join(str(RUNS_ROOT), "data_check_external")

# Published label distributions. A mirror that silently dropped or duplicated
# images shows up here immediately; a mirror that matches is the real set.
EXPECTED = {
    # APTOS-2019 train (Kaggle competition, 3,662 images)
    "aptos2019": {"n": 3662, "dist": {0: 1805, 1: 370, 2: 999, 3: 193, 4: 295}},
    # IDRiD part B, official train+test grading labels
    "idrid": {"n": 516, "dist": {0: 168, 1: 25, 2: 168, 3: 93, 4: 62}},
    # Messidor-2, the 1,744 gradable images of the adjudicated release
    "messidor2": {"n": 1744, "dist": {0: 1017, 1: 270, 2: 347, 3: 75, 4: 35}},
    # ODIR-5K: 6,392 eye images from 3,358 patients, 8 classes
    "odir5k": {"n": 6392, "dist": {"normal": 2873, "diabetes": 1608,
                                   "other_diseases_abnormalities": 708,
                                   "cataract": 293, "glaucoma": 284,
                                   "age_related_macular_degeneration": 266,
                                   "pathological_myopia": 232,
                                   "hypertension": 128}},
}

SAMPLE = int(os.environ.get("CHECK_SAMPLE", "12"))


# ------------------------------------------------------------------ prepare
def prepare_messidor2(force: bool = False) -> int:
    """Materialise the Messidor-2 HF parquet shards into raw/images + labels.csv.

    The mirror (sngsfydy/Messidor2) stores only `image` and a 0-4 `label`; the
    original Messidor-2 filenames are not in it, so images are written as
    zero-padded indices in shard/row order and that index IS the image id.
    Consequence, recorded in EXTERNAL_DATASETS.md: the 874 two-eye examinations
    cannot be reconstructed, so Messidor-2 is analysed one image per subject.
    """
    import csv
    import glob
    import pyarrow.parquet as pq

    root = raw_dir("messidor2")
    img_dir = root / "images"
    lab = root / "labels.csv"
    if lab.exists() and not force:
        print("messidor2 already prepared ->", lab)
        return 0
    shards = sorted(glob.glob(str(root / "parquet" / "*.parquet")))
    if not shards:
        raise FileNotFoundError("no parquet shards under %s" % (root / "parquet"))
    img_dir.mkdir(parents=True, exist_ok=True)

    rows, n = [], 0
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=64):
            d = batch.to_pydict()
            for img, label in zip(d["image"], d["label"]):
                fn = "messidor2_%05d.jpg" % n
                with open(img_dir / fn, "wb") as fh:
                    fh.write(img["bytes"])
                rows.append((fn, int(label), os.path.basename(shard),
                             img.get("path") or ""))
                n += 1
    with open(lab, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "dr_grade", "source_shard", "mirror_path"])
        w.writerows(rows)
    print("messidor2: wrote %d images + %s" % (n, lab))
    return n


# ------------------------------------------------------------------ checks
def _fov_stats(recs, idx):
    """FOV coverage and a crude correctness proxy for masks with no vessel GT.

    With no vessel ground truth there is no "fraction of annotated vessels
    inside the FOV" to score (the criterion used for the segmentation sets), so
    instead: the mask must be a single component covering a believable slice of
    the frame, and the retina must be *inside* it -- checked as the ratio of
    mean green-channel intensity inside vs outside. A mask that grabbed the
    black surround, or lost the retina, breaks that ratio.
    """
    cov, ratio, ncomp = [], [], []
    from scipy import ndimage as ndi
    for i in idx:
        r = recs[int(i)]
        ensure_fov(r)
        fov = read_binary(r["fov_path"])
        img = read_image(r["image_path"])
        g = img[..., 1].astype(np.float32) if img.ndim == 3 else img.astype(np.float32)
        cov.append(float(fov.mean()))
        inside = g[fov].mean() if fov.any() else 0.0
        outside = g[~fov].mean() if (~fov).any() else 0.0
        ratio.append(float(inside / max(outside, 1e-3)))
        ncomp.append(int(ndi.label(fov)[1]))
    return cov, ratio, ncomp


def check(name: str) -> int:
    t0 = time.time()
    recs = load_external(name, with_size=True)
    exp = EXPECTED.get(name, {})

    print("=" * 78)
    splits = collections.Counter(r["split"] for r in recs)
    print("%s   n=%d  subjects=%d  splits=%s"
          % (name.upper(), len(recs), len({r["subject_id"] for r in recs}),
             dict(splits)))

    ok = True
    if exp.get("n") not in (None, len(recs)):
        ok = False
        print("  !! expected %d records, got %d" % (exp["n"], len(recs)))

    # --- files exist ---------------------------------------------------
    missing = [r["image_id"] for r in recs if not os.path.exists(r["image_path"])]
    print("  missing image files: %d%s"
          % (len(missing), (" e.g. " + ", ".join(missing[:3])) if missing else ""))
    ok = ok and not missing

    # --- label distribution --------------------------------------------
    dist = collections.Counter(r["label"] for r in recs)
    print("  label distribution:", dict(sorted(dist.items(), key=lambda kv: str(kv[0]))))
    if exp.get("dist"):
        if dict(dist) == exp["dist"]:
            print("  label distribution MATCHES the published one")
        else:
            ok = False
            print("  !! published distribution is", exp["dist"])
    extra = sorted(set().union(*[set(r["labels"]) for r in recs]))
    print("  label fields:", extra)
    if name == "idrid":
        print("  DME risk:", dict(collections.Counter(r["labels"]["dme_risk"] for r in recs)))
    if name == "odir5k":
        eyes = collections.Counter(r["eye"] for r in recs)
        per_pat = collections.Counter(len(v) for v in
                                      _group(recs, "subject_id").values())
        print("  eyes:", dict(eyes), " images per patient:", dict(per_pat))
        print("  abbrevs:", {ODIR_CLASSES[k]: v for k, v in dist.items()})
    ref = collections.Counter(r["labels"].get("referable") for r in recs)
    if set(ref) - {None}:
        print("  referable (grade>=2):", dict(ref))

    # --- resolutions ----------------------------------------------------
    sizes = [r["native_size"] for r in recs if r.get("native_size")]
    if sizes:
        uniq = collections.Counter(sizes)
        px = np.array([w * h for w, h in sizes], dtype=np.float64)
        longest = np.array([max(s) for s in sizes])
        print("  native resolutions: %d distinct; top5 %s"
              % (len(uniq), uniq.most_common(5)))
        print("  longest side: min=%d  p25=%d  median=%d  p75=%d  max=%d"
              % (longest.min(), np.percentile(longest, 25), np.median(longest),
                 np.percentile(longest, 75), longest.max()))
        print("  megapixels: min=%.2f  median=%.2f  max=%.2f  (ratio max/min=%.1fx)"
              % (px.min() / 1e6, np.median(px) / 1e6, px.max() / 1e6,
                 px.max() / max(px.min(), 1)))

    # --- FOV on a sample -------------------------------------------------
    idx = np.linspace(0, len(recs) - 1, min(SAMPLE, len(recs))).astype(int)
    cov, ratio, ncomp = _fov_stats(recs, idx)
    print("  FOV coverage of frame (n=%d): mean=%.3f min=%.3f max=%.3f"
          % (len(idx), np.mean(cov), np.min(cov), np.max(cov)))
    print("  in/out green ratio: min=%.1f (want >> 1)   components: %s"
          % (np.min(ratio), sorted(set(ncomp))))
    if np.min(ratio) < 2.0:
        ok = False
        print("  !! a sampled FOV mask does not separate retina from surround")

    print("  %s   (%.1fs)" % ("OK" if ok else "PROBLEMS FOUND", time.time() - t0))
    return len(recs)


def _group(recs, key):
    g = collections.defaultdict(list)
    for r in recs:
        g[r[key]].append(r)
    return g


# ------------------------------------------------------------------ main
def main(argv):
    args = list(argv)
    if "--prepare" in args:
        i = args.index("--prepare")
        which = args[i + 1] if len(args) > i + 1 else "messidor2"
        args = args[:i] + args[i + 2:]
        if which != "messidor2":
            raise SystemExit("only messidor2 needs preparing")
        prepare_messidor2(force="--force" in args)
        return

    force = "--force" in args
    args = [a for a in args if a != "--force"]

    if "--fov" in args:
        args.remove("--fov")
        names = args or EXTERNAL_DATASETS
        for n in names:
            t0 = time.time()
            built = ensure_fovs(n, regenerate=force)
            print("%s: built %d FOV masks (%.0fs) -> %s"
                  % (n, built, time.time() - t0, EXTERNAL_ROOT / n / "fov"))
        return

    if "--resolutions" in args:
        args.remove("--resolutions")
        for n in args or EXTERNAL_DATASETS:
            s = record_resolutions(n, force=force)
            print("%s: %d sizes -> %s"
                  % (n, len(s), EXTERNAL_ROOT / n / "raw" / "_resolutions.csv"))
        return

    names = args or EXTERNAL_DATASETS
    total = 0
    for n in names:
        total += check(n)
    print("=" * 78)
    print("%d records across %d external datasets" % (total, len(names)))


if __name__ == "__main__":
    main(sys.argv[1:])
