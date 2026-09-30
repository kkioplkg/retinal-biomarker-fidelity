"""Biomarker measurement (S1 / Gate A of exp/EXPERIMENT_PLAN.md).

Two *independent* biomarker pipelines are kept side by side so that the
dimensional-consistency check of Gate A (Spearman >= 0.8 between pipelines on
every primary biomarker) is meaningful:

  * ``pvbm_pipe``  -- pipeline #1, a thin wrapper around the third-party
                     ``PVBM`` package (Fhima et al., MIT licence).
  * ``skan_pipe``  -- pipeline #2, ours: skimage skeletonisation + skan graph
                     statistics + our own box-counting fractal dimension.

``biomarkers.compute_all`` runs both and exposes the four PRIMARY biomarkers of
the paper (FD, tortuosity, density, total_length) from each pipeline.

``disc`` provides a lightweight optic-disc locator used for the zone-B
(annulus around the optic disc) variants of the biomarkers.
"""

from .disc import DiscEstimate, locate_optic_disc, save_disc_overlay  # noqa: F401

__all__ = ["DiscEstimate", "locate_optic_disc", "save_disc_overlay"]
