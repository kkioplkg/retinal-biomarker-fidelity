"""One place that decides how a figure is written to disk.

Elsevier prefers vector artwork for line drawings, and every figure in this
project is matplotlib, so there is no reason to ship only raster.  ``save_fig``
writes the PNG the pipeline already expected — nothing downstream that globs
``figs/*.png`` breaks — and additionally writes a PDF of the same basename,
which is what the LaTeX ``\\includegraphics`` calls pick up (graphicx prefers
``.pdf`` over ``.png`` when the extension is omitted).

Kept deliberately tiny and dependency-free so that ``figures.py``,
``robust/fig4.py`` and ``c1/stats_c1.py`` can all route through it.
"""

from __future__ import annotations

import os
from typing import List, Optional


def save_fig(fig, out_path: str, dpi: int = 150, also_pdf: bool = True,
             quiet: bool = False) -> List[str]:
    """Write ``fig`` to ``out_path`` and, unless disabled, to a sibling PDF.

    Parameters
    ----------
    fig : matplotlib Figure
    out_path : str
        Destination, normally ending in ``.png``.  The PDF is written next to
        it with the extension swapped.
    dpi : int
        Raster resolution; ignored by the vector writer.
    also_pdf : bool
        Set False (or export ``FIGS_NO_PDF=1``) to suppress the vector copy.
    quiet : bool
        Suppress the "wrote" lines.

    Returns the list of paths actually written.
    """
    if os.environ.get("FIGS_NO_PDF", "").strip() not in ("", "0", "false"):
        also_pdf = False

    written: List[str] = []
    d = os.path.dirname(os.path.abspath(out_path))
    if d:
        os.makedirs(d, exist_ok=True)

    fig.savefig(out_path, dpi=dpi)
    written.append(out_path)

    if also_pdf:
        stem, ext = os.path.splitext(out_path)
        pdf_path = stem + ".pdf"
        if ext.lower() != ".pdf":
            try:
                # bbox_inches is left alone: the callers already size their
                # figures, and changing it would move the PNG/PDF apart.
                fig.savefig(pdf_path)
                written.append(pdf_path)
            except Exception as e:  # pragma: no cover - defensive
                print("[savefig] PDF export failed for %s: %s" % (pdf_path, e))

    if not quiet:
        for p in written:
            print("wrote %s" % p)
    return written
