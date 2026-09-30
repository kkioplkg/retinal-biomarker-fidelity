"""Compact read-out of the E3 classification stage produced on the CPU node.

Reads only files the stage already wrote -- it computes no new statistic and
re-fits nothing.  For each cohort it prints:

*   the pre-registered PRIMARY cell (referable-DR, logistic regression, bio+cov)
    macro-AUROC per tag, from results/pivot/e3/<cohort>/summary.csv;
*   the per-seed paired contrasts with their bootstrap CIs, from
    results/pivot/e3_perseed/ref<ref>_s<k>/<cohort>/delta.csv -- these are the
    within-seed `ReliSeg - baseline` and `ReliSeg - continued` numbers;
*   the seed mean of the point estimates (the pre-registration asks for per-seed
    AND the seed mean; the mean has no CI of its own here, the per-seed CIs are
    the uncertainty statement).

    python exp/remote/cpu_e3_report.py idrid messidor2 aptos2019
    python exp/remote/cpu_e3_report.py --all-cells idrid      # every target/model
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

E3 = os.path.join("results", "pivot", "e3")
PERSEED = os.path.join("results", "pivot", "e3_perseed")
PRIMARY = dict(target="y_referable", clf="logreg", featureset="bio+cov")
SEEDS = (0, 1, 2)


def primary_filter(df: pd.DataFrame) -> pd.DataFrame:
    m = pd.Series(True, index=df.index)
    for k, v in PRIMARY.items():
        m &= df[k] == v
    return df[m]


def cohort(ds: str, all_cells: bool = False) -> None:
    print("=" * 78)
    can = os.path.join(E3, ds, "summary.csv")
    if not os.path.exists(can):
        print("%s: no summary.csv -- classification stage has not run" % ds)
        return
    s = pd.read_csv(can)
    role = s["cohort_role"].iloc[0]
    print("%s   role=%s   split_unit=%s   n=%d   n_excluded=%d"
          % (ds.upper(), role, s["split_unit"].iloc[0], s["n"].max(),
             s["n_excluded"].iloc[0]))
    print("   exclusion: %s" % s["exclude_src"].iloc[0])

    p = primary_filter(s).sort_values("tag")
    print("\n  PRIMARY cell (referable DR, logreg, bio+cov) -- macro AUROC per tag")
    for r in p.itertuples(index=False):
        print("    %-24s AUROC %.4f   AUPRC %.4f   n=%d%s"
              % (r.tag, r.macro_auroc, r.macro_auprc, r.n,
                 "   [is_primary]" if getattr(r, "is_primary", False) else ""))

    for ref in ("baseline", "continued"):
        rows = []
        for k in SEEDS:
            f = os.path.join(PERSEED, "ref%s_s%d" % (ref, k), ds, "delta.csv")
            if not os.path.exists(f):
                continue
            d = primary_filter(pd.read_csv(f))
            d = d[d["tag"] == "e3_fives_s%d_reliseg" % k]
            if len(d) == 1:
                r = d.iloc[0]
                rows.append((k, r["delta_vs_ref"], r["delta_ci_lo"],
                             r["delta_ci_hi"], r["delta_p"]))
        if not rows:
            print("\n  dAUC ReliSeg - %s : per-seed runs not present" % ref)
            continue
        print("\n  dAUC  ReliSeg - %s   (paired image-level bootstrap, 1000 resamples)"
              % ref)
        for k, d, lo, hi, pv in rows:
            print("    seed %d   %+.4f  [%+.4f, %+.4f]   p=%.3f" % (k, d, lo, hi, pv))
        mean = sum(r[1] for r in rows) / len(rows)
        print("    seed mean %+.4f  (n_seeds=%d)" % (mean, len(rows)))

    if all_cells:
        d = pd.read_csv(os.path.join(E3, ds, "delta.csv"))
        print("\n  all cells (canonical run, ref=%s):" % d["ref"].iloc[0])
        with pd.option_context("display.width", 200, "display.max_rows", 400):
            print(d[["target", "featureset", "clf", "tag", "delta_vs_ref",
                     "delta_ci_lo", "delta_ci_hi", "delta_p"]].round(4)
                  .to_string(index=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cohorts", nargs="*", default=["aptos2019", "idrid", "messidor2"])
    ap.add_argument("--all-cells", action="store_true")
    a = ap.parse_args(argv)
    for ds in (a.cohorts or ["aptos2019", "idrid", "messidor2"]):
        cohort(ds, a.all_cells)
    return 0


if __name__ == "__main__":
    sys.exit(main())
