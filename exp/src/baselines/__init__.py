"""S6 repair baselines: geometric, rNCA, EVAPORE.

All three share one interface::

    repair(image, prob, mask, fov, **kw) -> M_repaired          (H, W) bool
    repair_detailed(image, prob, mask, fov, **kw) -> RepairOutput

Use :func:`src.baselines.common.get_method` to obtain one by name, or the CLI
``python -m src.baselines.run_baseline --method {geometric,rnca,evapore} ...``.
"""

from src.baselines.common import METHODS, RepairOutput, get_method

__all__ = ["METHODS", "RepairOutput", "get_method"]
