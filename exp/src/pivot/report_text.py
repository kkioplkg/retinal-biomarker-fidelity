"""Narrative blocks of ``results/pivot/PROBE_REPORT.md``.

Kept apart from ``make_report.py`` so the prose can be edited without touching
the table code.  Every number quoted inside these strings is repeated in a
table generated from the CSVs.
"""

HEADER = """# Pivot feasibility probes P1-P3

Generated {date} by `python -m src.pivot.make_report` (Python {python},
`D:/Anaconda/envs/medical1`).  All code under `exp/src/pivot/`, all outputs
under `exp/results/pivot/`.

## What was run

| probe | question | entry point |
|---|---|---|
| P1 | do differentiable surrogates on the soft probability map track the real biomarker pipeline? (direction D1) | `python -m src.pivot.p1_run --device cuda:1` |
| P2 | can a post-hoc calibrator remove the systematic biomarker bias, with honest intervals? (direction D2) | `python -m src.pivot.p2_calibration` |
| P3 | does measurement error attenuate downstream disease-classification validity, and does calibration restore it? (direction D3) | `python -m src.pivot.p3_downstream` |

All three read one master table,
`results/pivot/bio_master.csv`, built by

    python -m src.pivot.build_table --procs 6 --fives_train_n 50

which recomputes `src.bio.biomarkers.compute_all` (`fd_rotations = 5`) on

* the observer-1 reference mask (`source = gt`),
* the held-out test prediction `runs/seg/<ds>/seed0/pred/mask` (`pred_test`),
* the out-of-fold training prediction `runs/seg_oof/<ds>/pred/mask` (`pred_oof`),

at **native resolution** (the probability maps in `runs/seg*/.../prob/*.npy`
are already native), with the optic disc detected once per image from the
fundus image alone so the GT row and the prediction row of an image share it.

Reference-mask rows reproduce `results/gateA_biomarkers_gt.csv` exactly for
DRIVE (relative difference 0.0000 on `FD_pvbm`, `FD_skan`, `density_pvbm`,
`total_length_skan`, `tortuosity_skan`; 0.0012 on `total_length_pvbm`, which is
the only stochastic column) and for CHASE_DB1 after a label-encoding fix
(CHASE ships boolean PNGs; a `> 127` test silently returns an empty mask --
`src.pivot.common.binarize` handles every encoding).  Prediction rows reproduce
`runs/repair/geometric/drive/seed0/per_image.csv`'s `bio_before_*` columns to
within 5e-3 relative.

Sigma is always the pre-registered training-split robust scale from
`results/gateA_biomarker_scales_train.csv`, the single source used everywhere
else in this project.

### Images covered

{counts}

FIVES contributes all 200 test images plus a deterministic class-balanced
subset of 200 training images (50 per disease class, lowest `image_id` first);
computing the biomarker pipeline for all 600 FIVES training images would have
cost ~2 h of the 4 h budget for no extra statistical power in these probes.

Seeds: segmentation seed 0 everywhere; `SEED = 0` for every split, bootstrap
and model in `p1_run` / `p2_calibration` / `p3_downstream`.

Environment: torch 2.6.0+cu124 (GPU 1, RTX 3080), scikit-learn 1.9.0,
numpy 2.2.6, scipy 1.17.1, pandas 3.0.5, scikit-image 0.26.0, OpenCV 5.0.0,
skan 0.13.1, PVBM 3.0.1.0.

---
"""

P1_INTRO = """## P1 -- differentiable biomarker surrogates (direction D1)

`src/pivot/surrogates.py` implements torch functions on `(B, 1, H, W)`
probability maps, all differentiable and all GPU-only:

* `soft_density` -- mean of `p` over the FOV;
* `soft_length` -- sum of the clDice soft skeleton (`soft_skeletonize` is
  **imported from `src/seg/losses.py`**, not re-implemented, so the surrogate
  and the existing clDice loss cannot drift apart);
* `soft_fd` -- box-counting fractal dimension, `-d log N(s) / d log s` with
  `N(s)` the summed occupancy of `s x s` max-pooled boxes for
  `s in {2, 4, 8, 16, 32, 64}`, plus two resolution-adaptive variants that use
  the same geometric ladder `2 .. min(H, W) / 4` as
  `src/bio/skan_pipe.box_counting_fd`: `s_fd_auto` on the mask (PVBM's D0 is a
  segmentation box-count) and `s_fd_skel` on the soft skeleton (skan's is a
  skeleton box-count);
* `soft_tortuosity` -- exploratory scale-ratio proxy
  `(4 * N_skel(4)) / soft_length`: a straight vessel fills one 4-px box per
  4 skeleton pixels so the ratio is ~1, a wandering vessel fills more;
  `s_tort_auto` is the same with the box scaled to `max(H, W) / 128`.

A unit test (`python -m src.pivot.surrogates`) confirms the gradient flows and
that the tortuosity proxy separates a straight line (1.04) from a zig-zag of
identical pixel count (2.04) while density and length stay identical, and that
the adaptive FD ladder separates them (0.906 vs 1.097) where the fixed ladder
does not (0.920 vs 0.918).

Evaluated on the seed-0 test predictions of all four datasets
(258 images), against the pipeline biomarkers of exactly the same masks.

"""

P1_VERDICT = """### P1 verdict

**PASS for density, total length and FD; FAIL for tortuosity.**

* `soft_density` is essentially the pipeline's own definition: Spearman 1.000
  in every dataset and pooled, against both pipelines.
* `soft_length` reaches 0.90-0.98 per dataset (pooled 0.97-0.98) against both
  `total_length_skan` and `total_length_pvbm`.  Comfortably over the 0.8 bar.
* `soft_fd` **depends entirely on the box ladder**.  The naive fixed ladder
  `s = 2..64` passes only on DRIVE (0.91-0.93) and collapses on FIVES
  (0.11-0.14), because 64 px is a thin slice of a 2048 px image while
  `skan_pipe.box_counting_fd` runs `2 .. min(H, W) / 4` = `2..512` there.  The
  resolution-adaptive ladder fixes it: `s_fd_skel` (adaptive ladder on the soft
  skeleton, matching skan's skeleton box-count) gives 0.85-1.00 per dataset and
  0.97 pooled, and `s_fd_auto` (adaptive ladder on the mask, matching PVBM's
  segmentation D0) gives 0.55-0.91 per dataset, 0.94-0.96 pooled.  **The
  usable FD surrogate is `s_fd_skel`, and the box ladder must scale with the
  image.**  This is a real design lesson for D1, not a tuning detail.
* `soft_tortuosity` fails: -0.48 to +0.33 per dataset.  Even on reference masks
  the scale-ratio proxy only reaches 0.41-0.55 on FIVES and ~0 elsewhere -- it
  is confounded by branch density, because coarse boxes merge neighbouring
  branches.  The `s_tort_auto` variant (box scaled to the image) does not help.
  **A biomarker-consistency loss should not include a tortuosity term**, which
  matches the memo's own hedge ("or not include it") and the Gate A finding
  that the two pipelines themselves only agree at rho ~ 0.65 on tortuosity.
* The soft/hard gap is negligible: `surrogate(prob)` and `surrogate(mask)`
  agree with the pipeline to within 0.01-0.05 Spearman everywhere, so nothing
  is lost by staying differentiable.
* Bias direction (P1.4): for every under-stated biomarker the unit-mapped
  surrogate of the prediction carries the **same negative sign** as the
  pipeline's own bias, usually larger in magnitude.  Descending a surrogate
  consistency loss therefore pushes the pipeline biomarker in the direction
  that removes the documented under-statement.

Cost: the whole P1 sweep is 72 s on one RTX 3080 for 258 images x 3 mask
variants -- the surrogates are cheap enough to sit inside a training loop.

"""

P2_INTRO = """## P2 -- post-hoc biomarker calibration (direction D2)

Fitting rows: the out-of-fold training predictions (`runs/seg_oof`) paired with
the reference biomarkers of the same image -- DRIVE 20, CHASE_DB1 20, HRF 15,
FIVES 200.  Scoring rows: the seed-0 held-out test predictions -- DRIVE 20,
CHASE_DB1 8, HRF 30, FIVES 200.  No test image enters any fit.

Nothing is regressed in raw units (HRF lengths are ~8x DRIVE lengths).  For
each dataset and biomarker the model predicts the relative correction
`y = (B_GT - B_pred) / s_ds`, where `s_ds = 1.4826 * MAD(B_pred)` over that
dataset's **fitting predictions** -- a quantity that needs no ground truth and
is therefore available in a new domain.  Reported errors divide by the
pre-registered GT sigma of `gateA_biomarker_scales_train.csv`.

Observables offered to the models: the eight predicted biomarkers (z-scored
within dataset), predicted vessel density, mean / std / binary entropy of the
probability map, fraction of low-confidence pixels (0.2 < p < 0.8) and of
ambiguous pixels (0.4 < p < 0.6), soft/hard mass ratio, mean probability inside
the predicted foreground, a vessel-vs-background CNR proxy on the green
channel, green mean / std / robust contrast, FOV fraction, relative optic-disc
radius, disc confidence flag and log10 of the longest side.

Five models, including two deliberately trivial ones:

* `offset_per_ds` -- subtract that dataset's mean signed bias.  **This is the
  bar**: a feature-based calibrator that does not beat it has bought nothing.
* `linear_per_ds` -- ridge on the observables, fitted per dataset.
* `gbdt_pooled` -- one gradient-boosted tree over all datasets, dataset
  identity as a feature (in-domain).
* `offset_lodo` -- mean signed bias of the *other* datasets (leave-one-
  dataset-out).
* `gbdt_lodo` -- GBDT on the other datasets, no dataset feature (LODO).

Sanity check against the existing diagnostic: the `bias_uncal` column here
reproduces `RESULTS_DIGEST.md` section 4.1 -- DRIVE FD_pvbm -0.985 (digest
-0.929), FD_skan -1.167 (-1.170), total_length_skan -1.050 (-1.078), HRF
total_length_pvbm -3.096 (-3.461), CHASE density +1.96 (+1.41).  The residual
differences are seed-0-test-only versus 3-seed pooling.

"""

P2_VERDICT = """### P2 verdict

**PASS in-domain, with a large caveat; FAIL under LODO.**

* **The systematic bias is almost entirely removable in-domain.**  A single
  constant per dataset takes HRF `total_length_pvbm` from -3.096 sigma to
  -0.377, HRF `FD_skan` from -1.627 to -0.044, DRIVE `FD_skan` from -1.167 to
  -0.043, FIVES `FD_skan` from -0.631 to -0.031.  That is the headline the
  pivot memo needs: the bias that repair could move by 0.03 sigma is worth
  1-3.5 sigma and a constant removes ~90 % of it.
* **MAE reduction is smaller than bias reduction, and only meets the >= 30 %
  bar where the bias is large.**  Macro over the eight primary columns, best
  in-domain model: HRF 42 %, DRIVE 33 %, CHASE_DB1 27 %, FIVES 18 %.  Per
  biomarker, the >= 30 % criterion for length/FD is met on DRIVE, CHASE_DB1 and
  HRF (`total_length_pvbm` 41-54 %, `FD_skan` 38-74 %) but not on FIVES, where
  the uncalibrated error is already small (`density` 0.17 sigma,
  `FD_pvbm` 0.28 sigma) and there is little left to remove.  Bias removal
  attacks the mean; the residual per-image scatter is what MAE is left with.
* **The learned models barely beat the constant.**  `linear_per_ds` and
  `gbdt_pooled` improve on `offset_per_ds` on CHASE_DB1 (0.27 / 0.25 vs 0.16)
  and FIVES (0.18 / 0.10 vs 0.13) and lose on HRF (0.29 / 0.29 vs 0.42) and
  DRIVE (ridge overfits 20 images badly: -0.12).  **The image-quality
  observables are not carrying the correction -- the dataset identity is.**
  For a paper this is the honest finding, and it means D2's novelty cannot rest
  on the feature set; it has to rest on the diagnosis plus the uncertainty.
* **LODO transfer fails, exactly where BTR failed.**  Macro reduction under
  LODO: HRF +21 %, DRIVE +20 %, CHASE_DB1 +6 %, FIVES **-167 % to -233 %**.
  FIVES is the outlier because its segmenter is nearly unbiased while the other
  three are strongly under-stating, so a calibrator trained on them applies a
  large spurious correction.  A per-dataset calibration is transportable only
  with a handful of labelled images from the new domain -- which is a
  deployable protocol, but it must be stated as such, not hidden.
* **Conformal intervals work where the calibration set is big enough and not
  otherwise.**  On FIVES (60 calibration images) the 90 % split-conformal
  interval covers 80-97 % with a width of 1.1-2.5 sigma.  On DRIVE / CHASE_DB1
  / HRF (5-6 calibration images) coverage is 0.75-1.00 with widths up to 7.8
  sigma -- the quantile is the maximum of five residuals, so the interval is
  unusable.  Marginal LODO coverage looks acceptable (86-90 % pooled) only
  because the intervals are 6-10 sigma wide.  **Any conformal claim in the
  paper needs a calibration split of at least a few dozen images per domain.**

"""

P3_INTRO = """## P3 -- downstream disease-classification validity (direction D3)

FIVES: 400 images (all 200 test + 50 per class of the training split),
4 classes (AMD / DR / Glaucoma / Normal).  HRF: 45 images, 3 classes
(dr / g / h).  Features: the eight primary biomarker columns, and a second
feature set that adds the eight zone-B columns.  Classifiers: multinomial
logistic regression (standardised) and a shallow `HistGradientBoosting`,
5-fold stratified CV repeated 3x; the score is the macro one-vs-rest AUC of
the pooled out-of-fold probabilities, with a 1000-sample image-level bootstrap.
The `gt - source` column is a **paired** bootstrap on the same resampled
images, so its CI is the right one for "did measurement error cost us AUC".

Calibrated features come from `p2_predictions.csv`; the rows of training images
there are cross-fitted (5-fold inside the fitting set, or LODO for the LODO
models), so no image is scored with a calibrator that saw its own reference
biomarkers.

"""

P3_VERDICT = """### P3 verdict

**Half PASS.  The attenuation half of the criterion is met on HRF and clearly
refuted on FIVES; the "calibration recovers part of it" half is REFUTED
everywhere -- and the reason is structural, not a tuning failure.**

* **HRF: measurement error is expensive.**  Logistic regression on the eight
  primary biomarkers reaches macro-AUC **0.968 [0.922, 0.999]** from the
  reference masks and only **0.753 [0.624, 0.861]** from the segmentation --
  a paired attenuation of **+0.215 [0.115, 0.321]**, i.e. the CI excludes zero.
  GBDT shows the same with a smaller gap (0.896 -> 0.776, +0.119 [0.000,
  0.241]).  This is the strongest single result of the three probes and it is
  exactly the "measurement fidelity limits clinical utility" claim the memo
  wants.
* **FIVES: no attenuation at all.**  GT 0.718 [0.688, 0.751] vs predicted
  0.713 [0.680, 0.750], difference +0.004 [-0.016, 0.025].  The FIVES
  segmenter is nearly unbiased (0.17-0.64 sigma) so there is nothing to lose;
  and the absolute AUC is only ~0.72, so these four biomarkers are weak
  four-class predictors to begin with.
* **Calibration recovers none of it, and that is a theorem, not bad luck.**
  On HRF every calibrated source lands at or below the uncalibrated one
  (`offset_per_ds` 0.727, `linear_per_ds` 0.752, `gbdt_pooled` 0.727 vs `pred`
  0.753).  A within-dataset classifier is invariant to a constant shift of a
  feature, and P2 showed that the constant shift is ~90 % of what calibration
  removes.  Table P3.3 makes it explicit: the reliability `r(B_pred, B_GT)`
  inside HRF is 0.49 (FD_skan), 0.49 (total_length_skan), 0.51 (density) --
  and after `offset_per_ds` calibration it is 0.46, 0.46, 0.43.  **Calibration
  moves the mean, attenuation is driven by the per-image scatter, and the
  scatter is untouched.**  FIVES, where there is no attenuation, has
  reliability 0.91-0.98.
* **Watch for a leak when calibrated features beat GT features.**  On FIVES
  `linear_per_ds` (0.728-0.731) and `gbdt_lodo` with zone B (0.757 [0.723,
  0.788], its CI for `gt - source` excludes zero) *exceed* the GT ceiling.
  That is not better measurement: the calibrator writes image-quality
  observables (contrast, entropy, CNR, disc radius) into the "biomarker", and
  those are themselves weakly disease-predictive.  Any future calibrated-
  biomarker classifier has to be reported against this confound.
* n = 45 for HRF (15 per class) is small; the CIs are wide and the result needs
  a second dataset with disease labels before it carries a paper.

"""

CLOSING = """## Overall assessment of the pivot

| direction | probe outcome | what it means for the paper |
|---|---|---|
| D1 biomarker-consistent segmentation | **feasible** for density / length / FD, **not** for tortuosity; the FD box ladder must be resolution-adaptive | the loss is implementable and cheap; VAFO-Loss overlap stays the novelty risk, so the contribution has to be the diagnosis + evaluation, not the loss |
| D2 post-hoc calibration | **works in-domain** (bias -3.1 sigma -> -0.4 sigma), but a **constant per dataset does most of it**, the observables add little, and **LODO transfer fails on FIVES** | honest framing: "per-domain recalibration with a few dozen labelled images", not "a universal calibrator"; conformal intervals need >= a few dozen calibration images |
| D3 downstream validity | **attenuation confirmed on HRF** (macro-AUC 0.968 GT vs 0.753 predicted, paired +0.215 [0.115, 0.321]), **absent on FIVES**, and **calibration recovers none of it** | the "measurement error costs clinical utility" claim holds; the "calibration buys it back" claim does not |

### The one result that should change the memo

The three probes together say something the memo does not currently say:

> The systematic bias (1-3.5 sigma) and the downstream validity loss are
> **two different problems**.  Post-hoc calibration fixes the first almost
> completely and the second not at all, because a constant per-dataset shift
> is invisible to any within-cohort analysis.  What attenuates disease
> classification is the per-image reliability `r(B_pred, B_GT)`, which is
> 0.43-0.51 on HRF and 0.91-0.98 on FIVES, and which calibration leaves
> unchanged (0.49 -> 0.46 on HRF).

Consequences for the plan:

1. **D2 cannot be the headline method.**  It is a cheap, honest component --
   it makes reported biomarker values comparable across cohorts and it gives
   intervals -- but it does not improve anything a clinician would act on
   within a cohort.  Sell it as *reporting-scale harmonisation plus
   uncertainty*, and do not claim downstream benefit.
2. **D1 is now the load-bearing direction**, because reducing per-image error
   (not mean error) is the only thing that can move P3's number, and P1 shows
   the surrogates needed for it are accurate (rho 0.97-1.00 for density,
   length and FD with the adaptive box ladder) and cheap.  The pre-registered
   endpoint for D1 should be **reliability `r(B_pred, B_GT)` and downstream
   macro-AUC**, not macro-MAE -- macro-MAE is the metric `RESULTS_DIGEST.md`
   section 4 already showed rewards bulk pixel addition.
3. **Drop tortuosity from the consistency loss** (P1: rho -0.48..+0.33) and
   keep it only as a reported biomarker with its known two-pipeline
   disagreement.
4. **The FD box ladder must scale with image size.**  A fixed 2..64 ladder
   passes on DRIVE and fails on FIVES (rho 0.11); `2..min(H,W)/4` on the soft
   skeleton gives 0.85-1.00.  Any VAFO-Loss-style comparison has to state
   which ladder it used.
5. **LODO remains the weak point** (P2 macro reduction -167 % on FIVES).
   Frame calibration as needing a few dozen labelled images per new domain, and
   pre-register that requirement rather than discovering it in review.

### Practical notes for whoever runs this next

1. `src/pivot/common.binarize` fixed a real bug: CHASE_DB1 observer masks are
   boolean PNGs, so `label > 127` returns an empty mask.  Any new code that
   reads `label_path` directly must use it.
2. The FIVES training split is sampled at 50 images per disease class (200 of
   600).  Extending to all 600 costs about 2 h of CPU at 6 processes and would
   mainly help P3's power, not P1 or P2.
3. `runs/seg*/**/prob/*.npy` are already at native resolution for every
   dataset, including HRF (2336x3504) and FIVES (2048x2048), so no
   resolution mapping is needed before the biomarker pipeline.
4. Everything is cached per (image, source) under `results/pivot/cache/fd5/`;
   re-running `build_table` only computes what is missing.
"""
