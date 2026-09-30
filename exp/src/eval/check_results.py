"""Consistency checks over the generated tables and the runs behind them.

Every check here exists because the corresponding mistake actually happened in
this project and was invisible in the output:

1. **no_repair provenance** -- the unrepaired row came from
   ``results/seg_per_image.csv`` (all 200 FIVES test images) while every repair
   row came from the 60-image cap, so FIVES no_repair clDice read 0.9066
   against ~0.8694 for methods that barely alter a mask.
2. **|A| = 0 FCR** -- an image that accepted nothing scored FCR 0.0 / precision
   1.0 and was averaged in, pulling low-acceptance arms' FCR down.
3. **Row counts** -- a tau sweep whose rows carried ``method=evapore_e2e`` was
   globbed into the headline row, making DRIVE 200 rows where 20 images x 3
   seeds = 60.
4. **Withdrawn runs** -- pre-C_geom-fix and partial runs sat as siblings of the
   live ones and were averaged in by a ``**`` glob.
5. **sigma scale** -- 10 ``geometric`` runs were scored with the leaked
   full-set Gate A sigma while everything else used the training-split table.
6. **Digest drift** -- numbers were quoted in RESULTS_DIGEST.md that no longer
   matched the regenerated tables.

Exit code is 0 when every check passes, 1 otherwise, so it can gate a pipeline
step.  ``--verbose`` prints each passing check too.

CLI
---
    cd exp
    python -m src.eval.check_results [--verbose]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import List, Tuple

import numpy as np

from .tables import (EXP_ROOT, RESULTS_DIR, RUNS_DIR, TAB2_DATASETS,
                     _live_glob, _load_baseline_method, _load_no_repair,
                     _load_rigr_mode, _read_csv)

EXPECTED_IMAGES = {"drive": 20, "chasedb1": 8, "hrf": 30, "fives": 60}
N_SEEDS = 3
TOL = 1e-6


class Report:
    def __init__(self, verbose: bool = False):
        self.fails: List[str] = []
        self.passes: List[str] = []
        self.verbose = verbose

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        if ok:
            self.passes.append(name)
            if self.verbose:
                print("  [ok]   %s" % name)
        else:
            self.fails.append("%s -- %s" % (name, detail))
            print("  [FAIL] %s -- %s" % (name, detail))
        return ok


# --------------------------------------------------------------------- 1
def check_no_repair(rep: Report):
    """no_repair must equal every method's before_* on the same (image, seed)."""
    import pandas as pd

    nr = _load_no_repair()
    if nr is None:
        rep.check(False, "no_repair loads", "returned None")
        return
    srcs = {"geometric": _load_baseline_method("geometric"),
            "rnca": _load_baseline_method("rnca"),
            "evapore_e2e": _load_baseline_method("evapore_e2e"),
            "evapore_scorer": _load_baseline_method("evapore_scorer")}
    # the repair frames carry after_* under 'cldice'; compare against the raw
    # before_* columns straight from disk instead
    for name, parts in (("geometric", (RUNS_DIR, "repair", "geometric", "*", "seed*")),
                        ("rnca", (RUNS_DIR, "repair", "rnca", "*", "seed*")),
                        ("rigr_risk", (RUNS_DIR, "rigr", "risk", "*", "seed*"))):
        frames = []
        for p in _live_glob(os.path.join(*parts, "per_image.csv")):
            d = _read_csv(p)
            if d is None or "before_cldice" not in d.columns:
                continue
            d = d.copy()
            d["dataset"] = d["dataset"].astype(str).str.lower()
            frames.append(d[["dataset", "image", "seed", "before_cldice",
                             "macro_mae_before"]])
        if not frames:
            continue
        f = pd.concat(frames, ignore_index=True)
        m = nr.merge(f, on=["dataset", "image", "seed"], how="inner")
        if m.empty:
            rep.check(False, "no_repair vs %s before_*" % name, "no shared rows")
            continue
        d1 = float((m["cldice"] - m["before_cldice"]).abs().max())
        d2 = float((m["macro_mae"] - m["macro_mae_before"]).abs().max())
        rep.check(d1 <= TOL and d2 <= TOL,
                  "no_repair == %s before_* (n=%d)" % (name, len(m)),
                  "max |dclDice|=%.3e max |dmacroMAE|=%.3e" % (d1, d2))

    # and it must be on the capped image set, not the full test split
    for ds, n in EXPECTED_IMAGES.items():
        got = int((nr["dataset"] == ds).sum())
        rep.check(got == n * N_SEEDS,
                  "no_repair row count %s" % ds,
                  "%d rows, expected %d images x %d seeds = %d"
                  % (got, n, N_SEEDS, n * N_SEEDS))

    # no_repair must not carry a repair's TRR/FCR
    for c in ("TRR_recall", "FCR", "n_accepted"):
        if c in nr.columns:
            rep.check(bool(nr[c].isna().all()),
                      "no_repair has no %s" % c,
                      "%d non-null values leaked from a repair run"
                      % int(nr[c].notna().sum()))


# --------------------------------------------------------------------- 2
def check_fcr_convention(rep: Report):
    """|A| = 0 must give FCR NaN, and it must be excluded from means."""
    import pandas as pd

    bad_files, bad_rows, checked = [], 0, 0
    for p in (_live_glob(os.path.join(RUNS_DIR, "rigr", "*", "*", "seed*", "per_image.csv"))
              + _live_glob(os.path.join(RUNS_DIR, "repair", "*", "*", "seed*", "per_image.csv"))):
        d = _read_csv(p)
        if d is None or "n_accepted" not in d.columns:
            continue
        from .tables import add_matched_fcr
        d = add_matched_fcr(d.copy())
        empty = d["n_accepted"].fillna(0) <= 0
        checked += int(empty.sum())
        leaked = int((empty & d["FCR"].notna()).sum())
        if leaked:
            bad_files.append(os.path.relpath(p, EXP_ROOT))
            bad_rows += leaked
    rep.check(not bad_files,
              "|A|=0 rows have undefined FCR (%d such rows checked)" % checked,
              "%d rows in %d file(s) still carry a finite FCR, e.g. %s"
              % (bad_rows, len(bad_files), bad_files[:3]))

    # Tab.2's n_fcr_defined must equal the real count of finite-FCR rows
    tab2 = os.path.join(RESULTS_DIR, "tab2.md")
    if os.path.exists(tab2):
        txt = open(tab2, encoding="utf-8").read()
        rep.check("n_fcr_defined" in txt, "tab2.md reports n_fcr_defined",
                  "column missing -- an FCR mean without its support")


# --------------------------------------------------------------------- 3
def check_row_counts(rep: Report):
    """One row per (dataset, image, seed); no sweep or duplicate leakage."""
    for m, loader in (("geometric", lambda: _load_baseline_method("geometric")),
                      ("rnca", lambda: _load_baseline_method("rnca")),
                      ("evapore_e2e", lambda: _load_baseline_method("evapore_e2e")),
                      ("evapore_scorer", lambda: _load_baseline_method("evapore_scorer")),
                      ("rigr_uniform", lambda: _load_rigr_mode("uniform")),
                      ("rigr_risk", lambda: _load_rigr_mode("risk"))):
        f = loader()
        if f is None:
            continue
        dup = int(f.duplicated(subset=["dataset", "image", "seed"]).sum())
        rep.check(dup == 0, "%s has no duplicate (dataset,image,seed)" % m,
                  "%d duplicates -- a sibling experiment is being globbed in" % dup)
        for ds, n in EXPECTED_IMAGES.items():
            sub = f[f["dataset"] == ds]
            if sub.empty:
                continue
            exp_n = n * sub["seed"].nunique()
            rep.check(len(sub) == exp_n, "%s/%s row count" % (m, ds),
                      "%d rows, expected %d images x %d seeds"
                      % (len(sub), n, sub["seed"].nunique()))


# --------------------------------------------------------------------- 4
def check_withdrawn_excluded(rep: Report):
    """No withdrawn / smoke directory may reach a table."""
    from .tables import _is_live_run

    leaked = []
    for root in ("rigr", "repair"):
        for p in _live_glob(os.path.join(RUNS_DIR, root, "**", "per_image.csv")):
            if not _is_live_run(p):
                leaked.append(os.path.relpath(p, EXP_ROOT))
    rep.check(not leaked, "withdrawn/smoke runs excluded from _live_glob",
              "%d leaked: %s" % (len(leaked), leaked[:3]))


# --------------------------------------------------------------------- 5
def check_sigma_scale(rep: Report):
    """Every macro_mae must be on the training-split Gate A sigma."""
    try:
        from .rescale_macro_mae import audit
    except Exception as e:                       # pragma: no cover
        rep.check(False, "sigma audit importable", str(e)); return
    t = audit(apply=False)
    if t is None or len(t) == 0:
        rep.check(False, "sigma audit ran", "no runs scored"); return
    bad = t[t["scale"] != "train"]
    rep.check(len(bad) == 0, "all %d runs on the training-split sigma" % len(t),
              "%d on another scale: %s" % (len(bad), list(bad["run"])[:3]))


# --------------------------------------------------------------------- 6
def _md_tables(text: str):
    """Yield (header, rows) for each markdown pipe-table in *text*."""
    rows, header = [], None
    for line in text.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = cells
            elif set("".join(cells)) <= set("-: "):
                continue
            else:
                rows.append(cells)
        else:
            if header and rows:
                yield header, rows
            header, rows = None, []
    if header and rows:
        yield header, rows


def check_digest(rep: Report):
    """Numbers quoted in RESULTS_DIGEST.md must match results/tab2.md."""
    dg = os.path.join(EXP_ROOT, "RESULTS_DIGEST.md")
    t2 = os.path.join(RESULTS_DIR, "tab2.md")
    if not (os.path.exists(dg) and os.path.exists(t2)):
        rep.check(False, "digest + tab2 present", "one is missing"); return

    truth = {}
    for header, rows in _md_tables(open(t2, encoding="utf-8").read()):
        if "method" not in header or "dataset" not in header:
            continue
        di, mi = header.index("dataset"), header.index("method")
        for r in rows:
            if len(r) <= max(di, mi):
                continue
            for col in ("cldice_mean", "macro_mae_mean"):
                if col in header:
                    v = r[header.index(col)]
                    try:
                        truth[(r[di], r[mi], col)] = float(v)
                    except ValueError:
                        pass

    mismatches, compared = [], 0
    for header, rows in _md_tables(open(dg, encoding="utf-8").read()):
        h = [c.lower() for c in header]
        if "method" not in h or "dataset" not in h:
            continue
        di, mi = h.index("dataset"), h.index("method")
        for r in rows:
            if len(r) <= max(di, mi):
                continue
            ds, me = r[di], r[mi]
            for label, col in (("cldice", "cldice_mean"),
                               ("macro-mae", "macro_mae_mean")):
                if label not in h:
                    continue
                cell = r[h.index(label)].replace("**", "").strip()
                try:
                    got = float(cell)
                except ValueError:
                    continue
                key = (ds, me, col)
                if key in truth:
                    compared += 1
                    if abs(truth[key] - got) > 5e-4:
                        mismatches.append("%s/%s %s digest=%.4f tab2=%.4f"
                                          % (ds, me, label, got, truth[key]))
    rep.check(not mismatches,
              "digest matches tab2.md (%d values compared)" % compared,
              "%d mismatch(es): %s" % (len(mismatches), mismatches[:4]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.eval.check_results")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    rep = Report(a.verbose)
    print("=" * 20 + " consistency checks " + "=" * 20)
    for fn in (check_no_repair, check_fcr_convention, check_row_counts,
               check_withdrawn_excluded, check_sigma_scale, check_digest):
        try:
            fn(rep)
        except Exception as e:                    # a broken check is a failure
            rep.check(False, fn.__name__, "raised %s: %s" % (type(e).__name__, e))
    print("\n%d passed, %d FAILED" % (len(rep.passes), len(rep.fails)))
    for f in rep.fails:
        print("  - %s" % f)
    return 1 if rep.fails else 0


if __name__ == "__main__":
    sys.exit(main())
