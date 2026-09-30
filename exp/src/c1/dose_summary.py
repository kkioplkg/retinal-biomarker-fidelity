"""Ten-line reading of the C1 dose-response result.

Consumes exactly what :mod:`src.c1.stats_c1` wrote -- ``c1_mixedlm_<biomarker>.csv``
(the nested mixed models, part c) and ``fig2_dose_response.csv`` (the arm-wise
severity curves, part d) -- and prints the findings the paper's results section
has to state:

* which perturbation **type** moves each biomarker most,
* how the effect scales with GT **radius bin** and disc **zone**,
* whether the **net topological excess over the pixel-matched control** is
  significantly positive, i.e. whether a cut costs more than its pixel budget.

Reading only the saved tables (never the event parquet) keeps this reproducible
from the archived results alone.

Usage::

    python -m src.c1.dose_summary [--results-dir results]
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(os.path.dirname(_HERE))
RESULTS_DIR = os.path.join(EXP_ROOT, "results")

BIOMARKERS = ("FD", "tortuosity", "density", "total_length")


def _stars(p: float) -> str:
    if not (p == p):
        return "  "
    return "***" if p < 1e-3 else "** " if p < 1e-2 else "*  " if p < 0.05 else "   "


def _fmt(coef: float, p: float) -> str:
    return f"{coef:+.4f}{_stars(p)}"


def _pick(df, prefix: str):
    """Terms of one factor, as ``{level: (coef, p)}``."""
    out = {}
    for _, r in df.iterrows():
        t = str(r["term"])
        if t.startswith(prefix) and ":" not in t:
            out[t[len(prefix):].strip("[]").replace("T.", "")] = (
                float(r["coef"]), float(r["p"]))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="10-line C1 dose-response summary")
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args(argv)

    import numpy as np
    import pandas as pd

    rd = args.results_dir
    fig2_path = os.path.join(rd, "fig2_dose_response.csv")
    f2 = pd.read_csv(fig2_path) if os.path.exists(fig2_path) else None

    print("=" * 78)
    print("C1 dose-response summary  (y = H_net = (|dB| - |dB_ctrl|)/sigma_B, "
          "signed)")
    print("=" * 78)

    line = 0
    for b in BIOMARKERS:
        path = os.path.join(rd, f"c1_mixedlm_{b}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        icpt = d[d.term == "Intercept"]
        i_c = float(icpt["coef"].iloc[0]); i_p = float(icpt["p"].iloc[0])
        n_obs = int(d["n_obs"].iloc[0]); n_grp = int(d["n_groups"].iloc[0])
        conv = bool(d["converged"].iloc[0])

        types = _pick(d, "C(type)")
        rad = _pick(d, "C(gt_radius_bin)")
        zone = _pick(d, "C(gt_zone)")
        order = _pick(d, "C(order_bin)")
        pipe = _pick(d, "C(pipeline)")
        sev = {str(r["term"]): (float(r["coef"]), float(r["p"]))
               for _, r in d.iterrows() if str(r["term"]).startswith("sev_rcs")}

        # strongest type / radius / zone contrast (reference level = the
        # alphabetically first factor level, absorbed into the intercept)
        def _top(dd):
            if not dd:
                return "n/a"
            k = max(dd, key=lambda k: abs(dd[k][0]))
            return f"{k} {_fmt(*dd[k])}"

        line += 1
        degenerate = bool(float(d["coef"].abs().max()) == 0.0)
        print(f"\n[{line}] {b:12s} n={n_obs} obs / {n_grp} images  converged={conv}")
        if degenerate:
            print("     DEGENERATE: H_net is identically 0.  The pixel-matched control")
            print("     removes exactly the same pixel count as the perturbation, so a")
            print("     pure pixel-count biomarker cancels by construction.  This")
            print("     validates the control; it is not a null result about topology.")
            continue
        print(f"     intercept (bridge, r-bin0, zone0, pvbm) = {_fmt(i_c, i_p)}"
              f"   <- net excess at the reference cell")
        print(f"     type    : " + ", ".join(f"{k}={_fmt(*v)}" for k, v in types.items()))
        print(f"     radius  : " + ", ".join(f"bin{k}={_fmt(*v)}" for k, v in rad.items())
              + f"   | zone: " + ", ".join(f"z{k}={_fmt(*v)}" for k, v in zone.items()))
        print(f"     severity: " + ", ".join(f"{k}={_fmt(*v)}" for k, v in sev.items())
              + f"   | order: " + ", ".join(f"{k}={_fmt(*v)}" for k, v in order.items())
              + f"   | pipeline: " + ", ".join(f"{k}={_fmt(*v)}" for k, v in pipe.items()))
        print(f"     strongest: type[{_top(types)}]  radius[{_top(rad)}]  zone[{_top(zone)}]")    # ---- the control question, answered on H_net (not on the signed arms) --
    #
    # fig2_dose_response.csv stores the *signed* dB/sigma of each arm, so
    # differencing the two arms there is NOT H_net: a negative perturbation mean
    # against a positive control mean differences to something with no
    # interpretation.  The quantity the question asks about is
    # H_net = (|dB| - |dB_ctrl|)/sigma, which is exactly what the mixed models
    # above are fitted on, so the marginal H_net of a perturbation type is
    # intercept + that type's coefficient.
    print("\n" + "-" * 78)
    print("Is the net topological excess over the matched control > 0?")
    print("  marginal H_net = intercept + type coef, at radius-bin 0 / zone 0 / pvbm")
    print("-" * 78)
    order4 = ("bridge", "caliber", "sever", "truncate")
    print(f"  {'biomarker':13s}" + "".join(f"{t:>13s}" for t in order4))
    for b in BIOMARKERS:
        path = os.path.join(rd, f"c1_mixedlm_{b}.csv")
        if not os.path.exists(path):
            continue
        d = pd.read_csv(path)
        if float(d["coef"].abs().max()) == 0.0:
            print(f"  {b:13s}" + f"{'0 (degenerate by control design)':>52s}")
            continue
        i_c = float(d[d.term == "Intercept"]["coef"].iloc[0])
        cells = {"bridge": i_c}                      # reference level
        for k, (c, _p) in _pick(d, "C(type)").items():
            cells[k] = i_c + c
        print(f"  {b:13s}" + "".join(f"{cells.get(t, float('nan')):>13.4f}"
                                     for t in order4))

    if f2 is not None:
        print("\n  Fig.2 arms (signed dB/sigma, mean over severity cells) --")
        print("  descriptive only; magnitudes, not H_net:")
        piv = f2.pivot_table(index=["biomarker", "type"], columns="arm",
                             values="mean", aggfunc="mean")
        if {"perturbation", "matched_control"} <= set(piv.columns):
            piv = piv[["perturbation", "matched_control"]].copy()
            piv["|pert|-|ctrl|"] = (piv["perturbation"].abs()
                                    - piv["matched_control"].abs())
            print("  " + piv.round(5).to_string().replace("\n", "\n  "))
            pos = float((piv["|pert|-|ctrl|"] > 0).mean())
            print(f"\n  cells where |perturbation| > |matched control|: {pos:.0%} "
                  f"of {len(piv)} (biomarker x type) cells")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
