"""Pluggable BTR risk backend for :func:`src.rigr.utility.risk_values`.

``src.rigr.utility.load_risk_backend`` accepts *any* object exposing
``predict_R_miss`` / ``predict_R_false``; the default is the ``src.c1.btr``
module itself, which always serves ``runs/btr/btr_{miss,false}_all.joblib``.
Stage S4 needs two departures from that default:

* the **deployed** ``R_false`` head (S4 item 6) -- retrained on rasterised A*
  candidate paths by :mod:`src.rigr.build_data` / :mod:`src.rigr.fit_models`
  and stored per dataset+seed under ``runs/rigr_models/<ds>/seed<k>/``;
* the **leave-one-dataset-out** heads (``btr_{kind}_wo_<ds>.joblib``) for the
  LODO block, so a held-out dataset is never priced by a head that saw it.

:class:`BTRBackend` binds both to explicit files and is passed straight through
``repair_mask(..., risk_backend=...)``.  ``__name__`` is what
``per_image.csv``'s ``risk_backend`` column records, so a run always states
which heads priced it.

Nothing here falls back silently: a missing file raises, and the caller
(``run_rigr``) decides whether to abort or to degrade to the all-data head with
an explicit warning.
"""

from __future__ import annotations

import os
from typing import Optional

__all__ = ["BTRBackend", "make_backend"]


class BTRBackend:
    """Duck-typed risk backend bound to two explicit :class:`BTRHead` files."""

    def __init__(self, miss_path: str, false_path: str, label: Optional[str] = None,
                 target: Optional[str] = None):
        from src.c1.btr import BTRHead

        for p in (miss_path, false_path):
            if not os.path.exists(p):
                raise FileNotFoundError("no BTR head at %r" % p)
        self.miss_path = os.path.abspath(miss_path)
        self.false_path = os.path.abspath(false_path)
        self._miss = BTRHead.load(miss_path)
        self._false = BTRHead.load(false_path)
        # what the heads were actually trained on, read back from the files
        # rather than trusted from the caller
        self.target = target or (self._miss.meta or {}).get("target", "hdep")
        stored = {(h.meta or {}).get("target", "hdep") for h in (self._miss, self._false)}
        if len(stored) > 1:
            raise ValueError(
                "miss/false heads were trained on different targets %r; refusing to "
                "mix them in one backend" % sorted(stored))
        self.__name__ = label or "btr[%s](%s|%s)" % (
            self.target, os.path.basename(miss_path), os.path.basename(false_path))

    # the two methods ``src.rigr.utility.load_risk_backend`` probes for
    def predict_R_miss(self, phi_df):
        return self._miss.predict(phi_df)

    def predict_R_false(self, phi_df):
        return self._false.predict(phi_df)

    def describe(self) -> dict:
        return dict(name=self.__name__, target=self.target,
                    miss=self.miss_path, false=self.false_path,
                    miss_meta=getattr(self._miss, "meta", {}),
                    false_meta=getattr(self._false, "meta", {}))


def make_backend(miss: Optional[str] = None, false: Optional[str] = None,
                 runs_dir: str = None, variant: str = "all",
                 label: Optional[str] = None,
                 target: Optional[str] = None) -> Optional[BTRBackend]:
    """Build a backend from explicit paths, falling back to ``runs/btr``.

    ``miss`` / ``false`` are explicit ``.joblib`` paths; whichever is omitted is
    resolved by :func:`src.c1.btr.head_path` from ``runs_dir``, ``variant`` and
    ``target``.

    ``target`` selects which supervision target the heads were trained on:

    * ``None`` / ``"hdep"`` -- the pre-registered net harm, resolved from the
      historical flat layout ``<runs_dir>/btr_<kind>_<variant>.joblib``;
    * ``"habs"`` -- the raw absolute standardized harm ``|dB_topo|/sigma`` that
      the **deployed** heads use (DECISIONS.md 2026-09-03), resolved from
      ``<runs_dir>/habs/btr_<kind>_<variant>.joblib``;
    * ``"habs_trainonly"`` -- the same target but fitted **only on training-split
      images**, from ``<runs_dir>/habs_trainonly/``.  This is the one stage S4
      must use: RiGR is applied to predicted masks of the *test* images, and a
      head that was fitted on perturbations of those same images has already
      seen the answer.  Any head applied to a test-split image should come from
      here.

    Returns ``None`` only for the fully default request (no paths, no runs_dir,
    ``variant == "all"``, no target) -- i.e. exactly the plain ``src.c1.btr``
    behaviour, which the caller should keep using so nothing changes.
    """
    from src.c1 import btr as btr_mod

    if (miss is None and false is None and variant == "all"
            and runs_dir is None and target in (None, "hdep")):
        return None
    rd = runs_dir or btr_mod.RUNS_DIR
    m = miss or btr_mod.head_path("miss", rd, variant, target)
    f = false or btr_mod.head_path("false", rd, variant, target)
    return BTRBackend(m, f, label=label, target=target)
