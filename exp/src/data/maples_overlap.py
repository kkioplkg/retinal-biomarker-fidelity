"""
Resolve MAPLES-DR fundus images against the project's Messidor-2 mirror, and
report the overlap between the two cohorts.

Why this exists
---------------
MAPLES-DR ships only *labels* (CC BY 4.0, figshare); the fundus images are the
property of the MESSIDOR consortium and are obtained separately from
<https://www.adcis.net/en/third-party/messidor/>, which requires a personal
registration form.  Meanwhile the project already holds 1,744 Messidor-2 images
(`exp/data/external/messidor2/raw/images`) — and Messidor-2 contains most of the
original MESSIDOR images.  The mirror, however, lost the original filenames
(`image_id` is a shard/row index), so we cannot tell by name which Messidor-2
row is which MAPLES-DR image.

Both questions — "which MAPLES-DR images do we already have?" and "which
Messidor-2 rows must be excluded from E3 so the replication cohort stays
disjoint?" — are the *same* question, and both are answered by matching each
MAPLES-DR vessel mask to the Messidor-2 image it was drawn on.

Method
------
A hand-drawn vessel mask overlays the dark, vessel-scale structure of the green
channel of exactly one image.  So:

  1. restrict candidates to the mirror rows with the *same* native size
     (MESSIDOR has only three: 1440x960, 2240x1488, 2304x1536);
  2. resize mask and green channel to 512x512;
  3. band-pass both (x - gaussian(x, sigma=3)) to delete the low-frequency term
     — the bright fundus disc on a black surround — which otherwise dominates
     and makes every mask correlate with every image (the first attempt scored
     174 of 198 masks onto the same handful of rows);
  4. L2-normalise and take the inner product, negating the image so that
     "vessel pixels are dark" scores positively;
  5. accept the top candidate when its lead over the rest of the pool is at
     least `MARGIN_SD` standard deviations.

Observed on this data: the accepted matches lead by 14.9-87.0 sd and the
rejected ones by 2.0-8.5 sd, i.e. an empty band between 8.5 and 14.9; the 162
accepted matches are a *bijection* (zero Messidor-2 row claimed twice) although
nothing in the scoring enforces that.  Both facts are printed by `--report`.

Usage
-----
    python -m src.data.maples_overlap            # match + write both CSVs
    python -m src.data.maples_overlap --report   # re-print stats from the CSV
"""
from __future__ import annotations

import argparse

import numpy as np

from .datasets import DATA_ROOT

S = 512                 # working resolution for the correlation
SIGMA = 3.0             # band-pass scale, in working-resolution pixels
MARGIN_SD = 10.0        # accept a match leading the pool by this many sd

MAPLES_ROOT = DATA_ROOT / "maplesdr" / "raw"
M2_ROOT = DATA_ROOT / "external" / "messidor2"
# Derived, so outside the pristine raw/ tree.
MATCH_CSV = DATA_ROOT / "maplesdr" / "messidor2_match.csv"
OVERLAP_CSV = M2_ROOT / "overlap_maplesdr.csv"


def _bandpass(a: np.ndarray) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    a = a.astype(np.float32)
    a = a - gaussian_filter(a, SIGMA)
    return (a / (np.linalg.norm(a) + 1e-8)).ravel()


def _vessel_masks():
    """(names, splits, sizes, band-passed 512x512 vectors) for all 198 masks."""
    from PIL import Image
    names, splits, sizes, vecs = [], [], [], []
    for split in ("train", "test"):
        d = MAPLES_ROOT / split / "Vessels"
        for p in sorted(d.glob("*.png")):
            with Image.open(p) as im:
                sizes.append(im.size)
                m = np.asarray(im.convert("L").resize((S, S), Image.BILINEAR),
                               dtype=np.float32) / 255.0
            names.append(p.stem)
            splits.append(split)
            vecs.append(_bandpass(m))
    return names, splits, sizes, np.stack(vecs)


def _messidor2_images():
    """
    (ids, sizes, negated band-passed 512x512 green channels) for the mirror.

    Deliberately *not* cached to disk: the array is 1,744 x 512 x 512 float32
    = 1.8 GB, and rebuilding it costs about three minutes.
    """
    import pandas as pd
    from PIL import Image
    res = pd.read_csv(M2_ROOT / "raw" / "_resolutions.csv")
    ids = list(res["image_id"])
    sizes = list(zip(res["width"], res["height"]))
    G = np.empty((len(ids), S * S), dtype=np.float32)
    for i, iid in enumerate(ids):
        with Image.open(M2_ROOT / "raw" / "images" / (iid + ".jpg")) as im:
            g = np.asarray(im.convert("RGB").split()[1].resize((S, S), Image.BILINEAR))
        G[i] = -_bandpass(g)                   # negate: vessels are dark
        if (i + 1) % 250 == 0:
            print("    bandpass %d/%d" % (i + 1, len(ids)), flush=True)
    return ids, sizes, G


def match(write: bool = True):
    import pandas as pd
    names, splits, msizes, M = _vessel_masks()
    print("MAPLES-DR vessel masks:", M.shape[0])
    ids, csizes, G = _messidor2_images()
    print("Messidor-2 candidates:", G.shape[0])

    rows = []
    for size in sorted(set(msizes)):
        mi = [i for i, s in enumerate(msizes) if s == size]
        ci = [j for j, s in enumerate(csizes) if s == size]
        print("  %dx%d: %d masks vs %d candidates" % (size[0], size[1], len(mi), len(ci)))
        if not ci:
            for i in mi:
                rows.append(dict(maples_name=names[i], maples_split=splits[i],
                                 width=size[0], height=size[1], n_cand=0,
                                 messidor2_id="", score=np.nan, second_score=np.nan,
                                 margin_sd=np.nan, matched=False))
            continue
        Sc = M[mi] @ G[ci].T
        for r, i in enumerate(mi):
            s = Sc[r]
            o = np.argsort(-s)
            rest = s[o[1:]]
            margin = float((s[o[0]] - rest.mean()) / (rest.std() + 1e-12))
            rows.append(dict(
                maples_name=names[i], maples_split=splits[i],
                width=size[0], height=size[1], n_cand=len(ci),
                messidor2_id=ids[ci[o[0]]], score=float(s[o[0]]),
                second_score=float(s[o[1]]), margin_sd=margin,
                matched=margin >= MARGIN_SD,
            ))
    df = pd.DataFrame(rows).sort_values(["maples_split", "maples_name"])
    df.loc[~df["matched"], "messidor2_id"] = ""
    if write:
        df.to_csv(MATCH_CSV, index=False)
        ov = df[df["matched"]][["messidor2_id", "maples_name", "maples_split",
                                "width", "height", "score", "margin_sd"]]
        ov = ov.sort_values("messidor2_id")
        ov.to_csv(OVERLAP_CSV, index=False)
        print("wrote", MATCH_CSV)
        print("wrote", OVERLAP_CSV)
    report(df)
    return df


def report(df=None):
    import pandas as pd
    if df is None:
        df = pd.read_csv(MATCH_CSV).fillna({"messidor2_id": ""})
    ok = df[df["matched"]]
    no = df[~df["matched"]]
    print("\nmatched   %3d / %3d   margin %.1f - %.1f sd   score %.3f - %.3f"
          % (len(ok), len(df), ok["margin_sd"].min(), ok["margin_sd"].max(),
             ok["score"].min(), ok["score"].max()))
    print("unmatched %3d / %3d   margin %.1f - %.1f sd   score %.3f - %.3f"
          % (len(no), len(df), no["margin_sd"].min(), no["margin_sd"].max(),
             no["score"].min(), no["score"].max()))
    print("bijective (no Messidor-2 row claimed twice):",
          not ok["messidor2_id"].duplicated().any())
    print("\nmatched by native size:")
    print(df.groupby(["width", "height"])["matched"].agg(["sum", "count"]).to_string())
    print("\nmatched by MAPLES-DR split:")
    print(df.groupby("maples_split")["matched"].agg(["sum", "count"]).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true",
                    help="print statistics from the existing CSV, do not re-match")
    a = ap.parse_args()
    if a.report:
        report()
    else:
        match()


if __name__ == "__main__":
    main()
