"""Driver: build every main-text display item (Tab.1-3, Fig.2-5).

Runs each ``build_*`` function from :mod:`src.eval.tables` and
:mod:`src.eval.figures`, catches any exception so one broken item never stops
the rest, and prints a final PRODUCED / SKIPPED / FAILED summary.

CLI
---
    cd exp
    python -m src.eval.make_all
"""

from __future__ import annotations

import sys
import traceback
from typing import Callable, List, Tuple

from . import figures, tables


def _run(name: str, fn: Callable, *args, **kwargs) -> Tuple[str, str]:
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    try:
        result = fn(*args, **kwargs)
        status = "PRODUCED" if result is not None else "SKIPPED"
        return name, status
    except Exception:
        print(f"[make_all] {name} raised an exception:")
        traceback.print_exc()
        return name, "FAILED"


def main(argv=None) -> int:
    # ``smoke_label`` stamps "(SMOKE DATA)" on Tab.2 and Fig.3.  It was
    # hard-coded True while the pipeline was still being exercised on the
    # 3-4 image smoke subsets, which means every full-grid rebuild since then
    # has also been labelled SMOKE DATA -- a label that is now wrong in the
    # more dangerous direction (real numbers marked as not-real).  It is a
    # flag now: pass --smoke to get it back, otherwise the tables are
    # labelled for what they are.
    import argparse

    ap = argparse.ArgumentParser(prog="python -m src.eval.make_all")
    ap.add_argument("--smoke", action="store_true",
                    help='stamp "(SMOKE DATA)" on Tab.2 and Fig.3')
    a = ap.parse_args(argv)
    smoke = bool(a.smoke)

    items: List[Tuple[str, str]] = []

    items.append(_run("Tab.1 (BTR predictive benchmark)", tables.build_tab1))
    items.append(_run("Tab.2 (RiGR main results)", tables.build_tab2,
                      smoke_label=smoke))
    items.append(_run("Tab.3 (ablation)", tables.build_tab3))
    items.append(_run("Fig.2 (BTR dose-response, check only)", figures.build_fig2))
    items.append(_run("Fig.3 (FCR-benefit tradeoff)", figures.build_fig3,
                      smoke_label=smoke))
    items.append(_run("Fig.4 (robustness)", figures.build_fig4))
    items.append(_run("Fig.5 (qualitative)", figures.build_fig5))

    # Consistency gate. Every check corresponds to a defect that actually
    # shipped silently here: no_repair computed on a different image set,
    # |A| = 0 scoring as perfect precision, a tau sweep globbed into a headline
    # row, withdrawn runs averaged in, a leaked sigma scale, digest drift.
    # Reported but never fatal -- make_all's contract is that one broken item
    # does not stop the rest.
    try:
        from . import check_results
        print(f"\n{'=' * 20} Consistency checks {'=' * 20}")
        rc = check_results.main([])
        items.append(("Consistency checks", "PRODUCED" if rc == 0 else "FAILED"))
    except Exception:
        print("[make_all] consistency checks raised:")
        traceback.print_exc()
        items.append(("Consistency checks", "FAILED"))

    print(f"\n{'=' * 20} SUMMARY {'=' * 20}")
    width = max(len(n) for n, _ in items)
    for name, status in items:
        print(f"  {name:<{width}}  {status}")
    n_ok = sum(1 for _, s in items if s == "PRODUCED")
    n_skip = sum(1 for _, s in items if s == "SKIPPED")
    n_fail = sum(1 for _, s in items if s == "FAILED")
    print(f"\n{n_ok} produced, {n_skip} skipped, {n_fail} failed "
          f"(out of {len(items)})")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
