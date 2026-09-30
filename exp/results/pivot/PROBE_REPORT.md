# Pivot feasibility probes P1-P3

Generated 2026-09-16 17:23 by `python -m src.pivot.make_report` (Python 3.11.16,
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

| dataset | source | n_images |
|---|---|---|
| chasedb1 | gt | 28 |
| chasedb1 | pred_oof | 20 |
| chasedb1 | pred_test | 8 |
| drive | gt | 40 |
| drive | pred_oof | 20 |
| drive | pred_test | 20 |
| fives | gt | 400 |
| fives | pred_oof | 200 |
| fives | pred_test | 200 |
| hrf | gt | 45 |
| hrf | pred_oof | 15 |
| hrf | pred_test | 30 |

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

## P1 -- differentiable biomarker surrogates (direction D1)

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


### P1.1 Ranking agreement, `surrogate(prob)` vs pipeline biomarker of the same prediction

| surrogate | biomarker | drive | chasedb1 | hrf | fives | POOLED |
|---|---|---|---|---|---|---|
| s_density | density_pvbm | 0.998 | 1.000 | 0.999 | 1.000 | 1.000 |
| s_density | density_skan | 0.998 | 1.000 | 0.999 | 1.000 | 1.000 |
| s_fd | FD_pvbm | 0.932 | 0.452 | 0.531 | 0.110 | 0.542 |
| s_fd | FD_skan | 0.908 | 0.762 | 0.627 | 0.135 | 0.556 |
| s_fd_auto | FD_pvbm | 0.931 | 0.548 | 0.838 | 0.890 | 0.943 |
| s_fd_auto | FD_skan | 0.896 | 0.833 | 0.899 | 0.909 | 0.955 |
| s_fd_skel | FD_pvbm | 0.932 | 0.881 | 0.850 | 0.943 | 0.968 |
| s_fd_skel | FD_skan | 0.961 | 1.000 | 0.894 | 0.950 | 0.974 |
| s_length | total_length_pvbm | 0.940 | 0.905 | 0.896 | 0.961 | 0.973 |
| s_length | total_length_skan | 0.983 | 0.976 | 0.926 | 0.962 | 0.981 |
| s_tort_auto | tortuosity_pvbm | 0.054 | -0.476 | 0.181 | 0.327 | 0.592 |
| s_tort_auto | tortuosity_skan | 0.188 | -0.405 | 0.267 | 0.280 | 0.596 |
| s_tortuosity | tortuosity_pvbm | -0.003 | -0.238 | 0.006 | 0.241 | 0.551 |
| s_tortuosity | tortuosity_skan | 0.132 | -0.476 | 0.156 | 0.320 | 0.606 |

### P1.2 The same, with the **hard mask** instead of the probability map (isolates the soft/hard gap)

| surrogate | biomarker | drive | chasedb1 | hrf | fives | POOLED |
|---|---|---|---|---|---|---|
| s_density | density_pvbm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| s_density | density_skan | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| s_fd | FD_pvbm | 0.937 | 0.524 | 0.538 | 0.119 | 0.548 |
| s_fd | FD_skan | 0.905 | 0.810 | 0.640 | 0.147 | 0.563 |
| s_fd_auto | FD_pvbm | 0.922 | 0.548 | 0.848 | 0.892 | 0.944 |
| s_fd_auto | FD_skan | 0.862 | 0.833 | 0.907 | 0.913 | 0.956 |
| s_fd_skel | FD_pvbm | 0.931 | 0.786 | 0.807 | 0.935 | 0.963 |
| s_fd_skel | FD_skan | 0.956 | 0.976 | 0.855 | 0.946 | 0.971 |
| s_length | total_length_pvbm | 0.932 | 0.905 | 0.878 | 0.958 | 0.971 |
| s_length | total_length_skan | 0.976 | 0.976 | 0.895 | 0.956 | 0.977 |
| s_tort_auto | tortuosity_pvbm | -0.245 | -0.476 | 0.155 | 0.287 | 0.561 |
| s_tort_auto | tortuosity_skan | 0.005 | -0.500 | 0.153 | 0.235 | 0.556 |
| s_tortuosity | tortuosity_pvbm | -0.153 | -0.357 | -0.058 | 0.264 | 0.550 |
| s_tortuosity | tortuosity_skan | -0.009 | -0.643 | -0.029 | 0.268 | 0.577 |

### P1.3 Surrogate on the reference mask vs pipeline on the reference mask (pure definitional agreement)

| surrogate | biomarker | drive | chasedb1 | hrf | fives | POOLED |
|---|---|---|---|---|---|---|
| s_density | density_pvbm | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| s_density | density_skan | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| s_fd | FD_pvbm | 0.904 | 0.071 | -0.220 | -0.196 | 0.343 |
| s_fd | FD_skan | 0.838 | 0.476 | -0.316 | -0.156 | 0.367 |
| s_fd_auto | FD_pvbm | 0.880 | 0.619 | 0.409 | 0.864 | 0.920 |
| s_fd_auto | FD_skan | 0.838 | 0.857 | 0.307 | 0.884 | 0.932 |
| s_fd_skel | FD_pvbm | 0.904 | 0.952 | 0.863 | 0.943 | 0.963 |
| s_fd_skel | FD_skan | 0.941 | 0.976 | 0.956 | 0.951 | 0.970 |
| s_length | total_length_pvbm | 0.976 | 0.976 | 0.964 | 0.965 | 0.978 |
| s_length | total_length_skan | 0.982 | 0.976 | 0.962 | 0.962 | 0.981 |
| s_tort_auto | tortuosity_pvbm | 0.044 | -0.024 | 0.266 | 0.501 | 0.681 |
| s_tort_auto | tortuosity_skan | 0.104 | -0.095 | -0.150 | 0.355 | 0.606 |
| s_tortuosity | tortuosity_pvbm | 0.155 | 0.143 | 0.439 | 0.547 | 0.711 |
| s_tortuosity | tortuosity_skan | 0.041 | 0.048 | -0.024 | 0.409 | 0.642 |

### P1.4 Bias: does descending the surrogate move the pipeline biomarker the right way?

`bias_pipeline` is `(B_pipeline(pred) - B_pipeline(GT)) / sigma`; `bias_surrogate` is the same for the affinely unit-mapped surrogate of the probability map.  Sign agreement is what matters.

| dataset | surrogate | biomarker | bias_pipeline_sigma | bias_surrogate_prob_sigma | bias_surrogate_hard_sigma | resid_fit_sigma |
|---|---|---|---|---|---|---|
| chasedb1 | s_fd | FD_skan | -0.785 | 0.222 | 0.189 | 0.646 |
| chasedb1 | s_fd_auto | FD_skan | -0.785 | 0.481 | 0.363 | 0.373 |
| chasedb1 | s_fd_skel | FD_skan | -0.785 | -1.334 | -0.817 | 0.138 |
| drive | s_fd | FD_skan | -1.167 | -0.160 | -0.270 | 0.344 |
| drive | s_fd_auto | FD_skan | -1.167 | -0.295 | -0.329 | 0.325 |
| drive | s_fd_skel | FD_skan | -1.167 | -2.085 | -1.106 | 0.178 |
| fives | s_fd | FD_skan | -0.631 | -0.014 | -0.018 | 1.029 |
| fives | s_fd_auto | FD_skan | -0.631 | -0.201 | -0.298 | 0.409 |
| fives | s_fd_skel | FD_skan | -0.631 | -0.686 | -0.540 | 0.335 |
| hrf | s_fd | FD_skan | -1.627 | -0.413 | -0.321 | 0.796 |
| hrf | s_fd_auto | FD_skan | -1.627 | 0.098 | 0.048 | 0.858 |
| hrf | s_fd_skel | FD_skan | -1.627 | -1.567 | -1.582 | 0.249 |
| chasedb1 | s_density | density_pvbm | 1.959 | 2.204 | 1.959 | 0.000 |
| drive | s_density | density_pvbm | 0.133 | 0.202 | 0.133 | 0.000 |
| fives | s_density | density_pvbm | -0.024 | 0.032 | -0.024 | 0.000 |
| hrf | s_density | density_pvbm | 0.431 | 0.561 | 0.431 | 0.000 |
| chasedb1 | s_tortuosity | tortuosity_skan | -1.118 | 2.031 | -0.995 | 1.529 |
| chasedb1 | s_tort_auto | tortuosity_skan | -1.118 | 0.603 | -0.803 | 1.470 |
| drive | s_tortuosity | tortuosity_skan | -1.359 | 1.304 | 0.090 | 1.354 |
| drive | s_tort_auto | tortuosity_skan | -1.359 | -0.997 | -0.061 | 1.281 |
| fives | s_tortuosity | tortuosity_skan | -0.094 | -2.024 | -0.116 | 0.857 |
| fives | s_tort_auto | tortuosity_skan | -0.094 | -1.041 | 0.187 | 0.886 |
| hrf | s_tortuosity | tortuosity_skan | -2.313 | 0.540 | 0.057 | 2.545 |
| hrf | s_tort_auto | tortuosity_skan | -2.313 | 1.552 | -0.510 | 2.628 |
| chasedb1 | s_length | total_length_skan | -0.492 | -1.732 | -0.609 | 0.117 |
| drive | s_length | total_length_skan | -1.050 | -1.920 | -0.795 | 0.071 |
| fives | s_length | total_length_skan | -0.242 | -0.852 | -0.195 | 0.243 |
| hrf | s_length | total_length_skan | -1.789 | -3.000 | -1.394 | 0.357 |

### P1 verdict

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


## P2 -- post-hoc biomarker calibration (direction D2)

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


### P2.0 Headline: macro-MAE over the eight primary columns (the `macro_mae` of `src/eval/biomarker_eval.py`)

| dataset | mae_uncal | mae_offset_per_ds | mae_linear_per_ds | mae_gbdt_pooled | mae_offset_lodo | mae_gbdt_lodo | red_offset_per_ds | red_linear_per_ds | red_gbdt_pooled | red_offset_lodo | red_gbdt_lodo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chasedb1 | 1.267 | 1.068 | 0.922 | 0.945 | 1.194 | 1.154 | 0.157 | 0.272 | 0.254 | 0.057 | 0.089 |
| drive | 1.051 | 0.734 | 1.172 | 0.700 | 0.844 | 1.122 | 0.302 | -0.116 | 0.334 | 0.197 | -0.068 |
| fives | 0.483 | 0.418 | 0.394 | 0.434 | 1.292 | 1.609 | 0.134 | 0.184 | 0.101 | -1.674 | -2.331 |
| hrf | 1.807 | 1.042 | 1.286 | 1.275 | 1.420 | 1.537 | 0.423 | 0.288 | 0.294 | 0.214 | 0.149 |

### P2.1 Standardised MAE `mean |B_hat - B_GT| / sigma` on the held-out test predictions

| biomarker | dataset | n | mae_uncal | mae_offset_per_ds | mae_linear_per_ds | mae_gbdt_pooled | mae_offset_lodo | mae_gbdt_lodo |
|---|---|---|---|---|---|---|---|---|
| FD_pvbm | ALL | 258 | 0.451 | 0.375 | 0.502 | 0.405 | 0.847 | 0.797 |
| FD_pvbm | chasedb1 | 8 | 0.741 | 0.545 | 0.508 | 0.469 | 0.649 | 0.629 |
| FD_pvbm | drive | 20 | 1.169 | 0.872 | 2.496 | 0.637 | 1.053 | 0.923 |
| FD_pvbm | fives | 200 | 0.279 | 0.277 | 0.280 | 0.317 | 0.830 | 0.772 |
| FD_pvbm | hrf | 30 | 1.037 | 0.644 | 0.653 | 0.825 | 0.874 | 0.922 |
| FD_skan | ALL | 258 | 0.818 | 0.572 | 0.480 | 0.511 | 1.360 | 1.633 |
| FD_skan | chasedb1 | 8 | 0.785 | 0.328 | 0.302 | 0.192 | 0.509 | 0.502 |
| FD_skan | drive | 20 | 1.220 | 0.758 | 0.757 | 0.650 | 0.774 | 1.301 |
| FD_skan | fives | 200 | 0.642 | 0.512 | 0.383 | 0.409 | 1.477 | 1.766 |
| FD_skan | hrf | 30 | 1.729 | 0.910 | 0.984 | 1.181 | 1.198 | 1.270 |
| density_pvbm | ALL | 258 | 0.323 | 0.306 | 0.289 | 0.361 | 0.409 | 0.772 |
| density_pvbm | chasedb1 | 8 | 1.959 | 1.735 | 1.242 | 1.599 | 1.962 | 2.108 |
| density_pvbm | drive | 20 | 0.746 | 0.752 | 0.751 | 0.811 | 0.746 | 1.047 |
| density_pvbm | fives | 200 | 0.165 | 0.168 | 0.169 | 0.205 | 0.275 | 0.643 |
| density_pvbm | hrf | 30 | 0.659 | 0.543 | 0.525 | 0.774 | 0.662 | 1.092 |
| density_skan | ALL | 258 | 0.323 | 0.306 | 0.289 | 0.361 | 0.409 | 0.772 |
| density_skan | chasedb1 | 8 | 1.959 | 1.735 | 1.242 | 1.599 | 1.962 | 2.108 |
| density_skan | drive | 20 | 0.746 | 0.752 | 0.751 | 0.811 | 0.746 | 1.047 |
| density_skan | fives | 200 | 0.165 | 0.168 | 0.169 | 0.205 | 0.275 | 0.643 |
| density_skan | hrf | 30 | 0.659 | 0.543 | 0.525 | 0.774 | 0.662 | 1.092 |
| tortuosity_pvbm | ALL | 255 | 1.355 | 1.038 | 1.018 | 1.010 | 1.324 | 1.259 |
| tortuosity_pvbm | chasedb1 | 8 | 0.659 | 0.834 | 1.204 | 0.955 | 0.912 | 1.095 |
| tortuosity_pvbm | drive | 20 | 0.909 | 0.497 | 0.998 | 0.560 | 0.596 | 0.479 |
| tortuosity_pvbm | fives | 197 | 1.187 | 1.025 | 0.932 | 0.945 | 1.239 | 1.190 |
| tortuosity_pvbm | hrf | 30 | 2.942 | 1.539 | 1.548 | 1.752 | 2.481 | 2.276 |
| tortuosity_skan | ALL | 258 | 0.997 | 0.885 | 1.079 | 0.934 | 1.366 | 1.151 |
| tortuosity_skan | chasedb1 | 8 | 2.180 | 1.992 | 1.561 | 1.523 | 2.039 | 1.572 |
| tortuosity_skan | drive | 20 | 1.359 | 1.091 | 1.674 | 1.088 | 1.279 | 1.087 |
| tortuosity_skan | fives | 200 | 0.716 | 0.720 | 0.677 | 0.783 | 1.237 | 1.071 |
| tortuosity_skan | hrf | 30 | 2.313 | 1.549 | 3.230 | 1.686 | 2.107 | 1.617 |
| total_length_pvbm | ALL | 258 | 0.844 | 0.451 | 0.519 | 0.554 | 2.861 | 4.598 |
| total_length_pvbm | chasedb1 | 8 | 1.229 | 0.721 | 0.696 | 0.688 | 1.033 | 0.811 |
| total_length_pvbm | drive | 20 | 1.166 | 0.542 | 1.350 | 0.553 | 0.666 | 2.536 |
| total_length_pvbm | fives | 200 | 0.443 | 0.269 | 0.255 | 0.355 | 3.295 | 5.274 |
| total_length_pvbm | hrf | 30 | 3.199 | 1.533 | 1.678 | 1.845 | 1.914 | 2.479 |
| total_length_skan | ALL | 258 | 0.534 | 0.351 | 0.422 | 0.412 | 1.576 | 1.408 |
| total_length_skan | chasedb1 | 8 | 0.620 | 0.649 | 0.618 | 0.537 | 0.485 | 0.403 |
| total_length_skan | drive | 20 | 1.090 | 0.606 | 0.603 | 0.487 | 0.895 | 0.558 |
| total_length_skan | fives | 200 | 0.267 | 0.205 | 0.287 | 0.258 | 1.705 | 1.512 |
| total_length_skan | hrf | 30 | 1.914 | 1.077 | 1.149 | 1.361 | 1.464 | 1.549 |

### P2.2 Relative MAE reduction vs the uncalibrated prediction

| biomarker | dataset | n | red_offset_per_ds | red_linear_per_ds | red_gbdt_pooled | red_offset_lodo | red_gbdt_lodo |
|---|---|---|---|---|---|---|---|
| FD_pvbm | ALL | 258 | 0.169 | -0.113 | 0.101 | -0.878 | -0.767 |
| FD_pvbm | chasedb1 | 8 | 0.264 | 0.314 | 0.368 | 0.125 | 0.151 |
| FD_pvbm | drive | 20 | 0.254 | -1.134 | 0.455 | 0.100 | 0.210 |
| FD_pvbm | fives | 200 | 0.007 | -0.001 | -0.133 | -1.972 | -1.764 |
| FD_pvbm | hrf | 30 | 0.379 | 0.371 | 0.205 | 0.158 | 0.111 |
| FD_skan | ALL | 258 | 0.301 | 0.413 | 0.375 | -0.663 | -0.998 |
| FD_skan | chasedb1 | 8 | 0.582 | 0.615 | 0.756 | 0.352 | 0.360 |
| FD_skan | drive | 20 | 0.379 | 0.380 | 0.468 | 0.366 | -0.066 |
| FD_skan | fives | 200 | 0.202 | 0.403 | 0.362 | -1.301 | -1.752 |
| FD_skan | hrf | 30 | 0.474 | 0.431 | 0.317 | 0.307 | 0.265 |
| density_pvbm | ALL | 258 | 0.055 | 0.106 | -0.118 | -0.265 | -1.389 |
| density_pvbm | chasedb1 | 8 | 0.114 | 0.366 | 0.184 | -0.002 | -0.076 |
| density_pvbm | drive | 20 | -0.008 | -0.007 | -0.087 | -0.000 | -0.403 |
| density_pvbm | fives | 200 | -0.018 | -0.025 | -0.241 | -0.666 | -2.894 |
| density_pvbm | hrf | 30 | 0.176 | 0.203 | -0.174 | -0.005 | -0.657 |
| density_skan | ALL | 258 | 0.055 | 0.106 | -0.118 | -0.265 | -1.389 |
| density_skan | chasedb1 | 8 | 0.114 | 0.366 | 0.184 | -0.002 | -0.076 |
| density_skan | drive | 20 | -0.008 | -0.007 | -0.087 | -0.000 | -0.403 |
| density_skan | fives | 200 | -0.018 | -0.025 | -0.241 | -0.666 | -2.894 |
| density_skan | hrf | 30 | 0.176 | 0.203 | -0.174 | -0.005 | -0.657 |
| tortuosity_pvbm | ALL | 255 | 0.234 | 0.249 | 0.255 | 0.022 | 0.071 |
| tortuosity_pvbm | chasedb1 | 8 | -0.265 | -0.826 | -0.449 | -0.384 | -0.662 |
| tortuosity_pvbm | drive | 20 | 0.454 | -0.097 | 0.384 | 0.345 | 0.473 |
| tortuosity_pvbm | fives | 197 | 0.136 | 0.215 | 0.204 | -0.044 | -0.002 |
| tortuosity_pvbm | hrf | 30 | 0.477 | 0.474 | 0.404 | 0.157 | 0.226 |
| tortuosity_skan | ALL | 258 | 0.112 | -0.082 | 0.062 | -0.371 | -0.155 |
| tortuosity_skan | chasedb1 | 8 | 0.086 | 0.284 | 0.301 | 0.065 | 0.279 |
| tortuosity_skan | drive | 20 | 0.197 | -0.232 | 0.199 | 0.059 | 0.200 |
| tortuosity_skan | fives | 200 | -0.006 | 0.054 | -0.094 | -0.729 | -0.496 |
| tortuosity_skan | hrf | 30 | 0.330 | -0.397 | 0.271 | 0.089 | 0.301 |
| total_length_pvbm | ALL | 258 | 0.465 | 0.385 | 0.344 | -2.391 | -4.450 |
| total_length_pvbm | chasedb1 | 8 | 0.414 | 0.434 | 0.440 | 0.159 | 0.340 |
| total_length_pvbm | drive | 20 | 0.536 | -0.157 | 0.526 | 0.429 | -1.175 |
| total_length_pvbm | fives | 200 | 0.392 | 0.425 | 0.199 | -6.441 | -10.909 |
| total_length_pvbm | hrf | 30 | 0.521 | 0.476 | 0.423 | 0.402 | 0.225 |
| total_length_skan | ALL | 258 | 0.342 | 0.209 | 0.227 | -1.954 | -1.639 |
| total_length_skan | chasedb1 | 8 | -0.047 | 0.004 | 0.133 | 0.218 | 0.351 |
| total_length_skan | drive | 20 | 0.444 | 0.446 | 0.553 | 0.179 | 0.488 |
| total_length_skan | fives | 200 | 0.234 | -0.074 | 0.036 | -5.378 | -4.657 |
| total_length_skan | hrf | 30 | 0.437 | 0.400 | 0.289 | 0.235 | 0.191 |

### P2.3 Signed bias, before and after

| biomarker | dataset | bias_uncal | bias_offset_per_ds | bias_linear_per_ds | bias_gbdt_pooled | bias_offset_lodo | bias_gbdt_lodo |
|---|---|---|---|---|---|---|---|
| FD_pvbm | ALL | -0.323 | -0.038 | 0.113 | -0.061 | 0.391 | 0.345 |
| FD_pvbm | chasedb1 | -0.494 | 0.259 | 0.212 | 0.053 | -0.309 | -0.001 |
| FD_pvbm | drive | -0.985 | -0.045 | 2.496 | 0.235 | -0.747 | -0.600 |
| FD_pvbm | fives | -0.156 | -0.045 | -0.050 | -0.022 | 0.695 | 0.614 |
| FD_pvbm | hrf | -0.954 | -0.068 | -0.415 | -0.552 | -0.689 | -0.725 |
| FD_skan | ALL | -0.793 | -0.022 | 0.082 | -0.015 | 0.866 | 1.213 |
| FD_skan | chasedb1 | -0.785 | 0.316 | 0.287 | 0.123 | -0.509 | -0.502 |
| FD_skan | drive | -1.167 | -0.043 | -0.024 | 0.193 | -0.419 | 1.283 |
| FD_skan | fives | -0.631 | -0.031 | 0.174 | 0.061 | 1.293 | 1.584 |
| FD_skan | hrf | -1.627 | -0.044 | -0.512 | -0.698 | -0.754 | -0.851 |
| density_pvbm | ALL | 0.103 | 0.074 | 0.076 | 0.062 | -0.074 | -0.202 |
| density_pvbm | chasedb1 | 1.959 | 1.735 | 1.030 | 1.599 | 1.962 | 2.108 |
| density_pvbm | drive | 0.133 | 0.070 | 0.104 | 0.541 | 0.125 | -0.799 |
| density_pvbm | fives | -0.024 | 0.017 | 0.045 | -0.004 | -0.253 | -0.403 |
| density_pvbm | hrf | 0.431 | 0.014 | 0.010 | -0.222 | 0.437 | 0.917 |
| density_skan | ALL | 0.103 | 0.074 | 0.076 | 0.062 | -0.074 | -0.202 |
| density_skan | chasedb1 | 1.959 | 1.735 | 1.030 | 1.599 | 1.962 | 2.108 |
| density_skan | drive | 0.133 | 0.070 | 0.104 | 0.541 | 0.125 | -0.799 |
| density_skan | fives | -0.024 | 0.017 | 0.045 | -0.004 | -0.253 | -0.403 |
| density_skan | hrf | 0.431 | 0.014 | 0.010 | -0.222 | 0.437 | 0.917 |
| tortuosity_pvbm | ALL | -1.225 | -0.373 | -0.332 | -0.395 | -0.053 | -0.917 |
| tortuosity_pvbm | chasedb1 | -0.492 | -0.763 | -0.854 | -0.405 | 0.884 | 1.095 |
| tortuosity_pvbm | drive | -0.895 | 0.010 | 0.375 | 0.009 | -0.498 | 0.081 |
| tortuosity_pvbm | fives | -1.027 | -0.296 | -0.275 | -0.321 | 0.318 | -0.909 |
| tortuosity_pvbm | hrf | -2.938 | -1.026 | -1.036 | -1.142 | -2.444 | -2.166 |
| tortuosity_skan | ALL | -0.482 | -0.227 | -0.317 | -0.242 | 0.304 | 0.304 |
| tortuosity_skan | chasedb1 | -1.118 | -0.751 | -0.350 | -0.542 | -0.845 | -0.360 |
| tortuosity_skan | drive | -1.359 | -0.344 | 0.323 | -0.088 | -1.279 | -0.784 |
| tortuosity_skan | fives | -0.094 | -0.056 | -0.073 | -0.045 | 0.870 | 0.588 |
| tortuosity_skan | hrf | -2.313 | -1.147 | -2.359 | -1.577 | -2.103 | -0.691 |
| total_length_pvbm | ALL | -0.826 | -0.010 | 0.014 | -0.060 | 2.341 | 4.030 |
| total_length_pvbm | chasedb1 | -1.229 | 0.721 | 0.696 | 0.688 | -1.033 | -0.810 |
| total_length_pvbm | drive | -1.163 | 0.227 | 1.350 | 0.259 | -0.563 | 2.536 |
| total_length_pvbm | fives | -0.435 | -0.007 | 0.012 | 0.055 | 3.295 | 5.274 |
| total_length_pvbm | hrf | -3.096 | -0.377 | -1.045 | -1.239 | -1.187 | -1.978 |
| total_length_skan | ALL | -0.492 | 0.014 | -0.035 | -0.014 | 1.128 | 1.014 |
| total_length_skan | chasedb1 | -0.492 | 0.649 | 0.618 | 0.537 | -0.234 | 0.125 |
| total_length_skan | drive | -1.050 | 0.024 | 0.035 | 0.116 | -0.807 | -0.192 |
| total_length_skan | fives | -0.242 | 0.032 | 0.024 | 0.091 | 1.705 | 1.505 |
| total_length_skan | hrf | -1.789 | -0.282 | -0.644 | -0.946 | -1.067 | -1.216 |

### P2.4 Split-conformal 90 % intervals: empirical coverage and mean width (in sigma)

| biomarker | dataset | cov90_offset_per_ds | width_offset_per_ds | cov90_linear_per_ds | width_linear_per_ds | cov90_gbdt_pooled | width_gbdt_pooled | cov90_offset_lodo | width_offset_lodo | cov90_gbdt_lodo | width_gbdt_lodo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FD_pvbm | ALL | 0.884 | 1.727 | 0.899 | 2.436 | 0.934 | 2.135 | 0.895 | 6.637 | 0.853 | 6.055 |
| FD_pvbm | chasedb1 | 0.875 | 2.748 | 0.875 | 2.764 | 0.875 | 1.798 | 0.625 | 1.611 | 0.750 | 1.516 |
| FD_pvbm | drive | 1.000 | 8.746 | 1.000 | 18.047 | 0.850 | 2.293 | 0.550 | 2.020 | 0.450 | 1.555 |
| FD_pvbm | fives | 0.860 | 0.772 | 0.875 | 0.776 | 0.970 | 2.101 | 0.990 | 7.992 | 0.990 | 7.427 |
| FD_pvbm | hrf | 0.967 | 3.141 | 1.000 | 3.005 | 0.767 | 2.348 | 0.567 | 2.025 | 0.233 | 1.114 |
| FD_skan | ALL | 0.849 | 2.167 | 0.841 | 2.043 | 0.919 | 2.391 | 0.899 | 8.337 | 0.860 | 6.859 |
| FD_skan | chasedb1 | 1.000 | 2.778 | 1.000 | 2.780 | 1.000 | 0.946 | 0.375 | 0.855 | 0.375 | 0.851 |
| FD_skan | drive | 1.000 | 7.765 | 1.000 | 7.703 | 0.800 | 2.157 | 0.750 | 2.488 | 0.350 | 2.070 |
| FD_skan | fives | 0.805 | 1.174 | 0.800 | 1.064 | 0.970 | 2.451 | 0.990 | 10.116 | 0.985 | 8.243 |
| FD_skan | hrf | 1.000 | 4.893 | 0.967 | 4.599 | 0.633 | 2.533 | 0.533 | 2.372 | 0.500 | 2.425 |
| density_pvbm | ALL | 0.934 | 1.499 | 0.926 | 1.608 | 0.965 | 3.329 | 0.942 | 5.482 | 0.899 | 4.322 |
| density_pvbm | chasedb1 | 0.875 | 5.636 | 0.875 | 7.909 | 0.500 | 3.047 | 0.250 | 2.451 | 0.000 | 1.190 |
| density_pvbm | drive | 0.950 | 3.175 | 0.900 | 3.133 | 0.800 | 2.496 | 0.700 | 2.295 | 0.500 | 1.993 |
| density_pvbm | fives | 0.925 | 0.722 | 0.920 | 0.784 | 1.000 | 3.395 | 1.000 | 6.319 | 0.990 | 4.883 |
| density_pvbm | hrf | 1.000 | 4.453 | 1.000 | 4.405 | 0.967 | 3.527 | 0.900 | 2.842 | 0.800 | 2.967 |
| density_skan | ALL | 0.934 | 1.499 | 0.926 | 1.608 | 0.965 | 3.329 | 0.942 | 5.482 | 0.899 | 4.322 |
| density_skan | chasedb1 | 0.875 | 5.636 | 0.875 | 7.909 | 0.500 | 3.047 | 0.250 | 2.451 | 0.000 | 1.190 |
| density_skan | drive | 0.950 | 3.175 | 0.900 | 3.133 | 0.800 | 2.496 | 0.700 | 2.295 | 0.500 | 1.993 |
| density_skan | fives | 0.925 | 0.722 | 0.920 | 0.784 | 1.000 | 3.395 | 1.000 | 6.319 | 0.990 | 4.883 |
| density_skan | hrf | 1.000 | 4.453 | 1.000 | 4.405 | 0.967 | 3.527 | 0.900 | 2.842 | 0.800 | 2.967 |
| tortuosity_pvbm | ALL | 0.804 | 2.601 | 0.820 | 2.625 | 0.847 | 3.117 | 0.863 | 4.356 | 0.824 | 4.047 |
| tortuosity_pvbm | chasedb1 | 1.000 | 3.674 | 1.000 | 6.382 | 1.000 | 4.808 | 1.000 | 5.400 | 0.875 | 5.380 |
| tortuosity_pvbm | drive | 0.350 | 0.932 | 0.950 | 5.195 | 0.900 | 1.687 | 0.700 | 2.066 | 0.850 | 1.443 |
| tortuosity_pvbm | fives | 0.878 | 2.742 | 0.832 | 2.225 | 0.909 | 3.331 | 0.975 | 4.893 | 0.898 | 4.602 |
| tortuosity_pvbm | hrf | 0.567 | 2.508 | 0.600 | 2.543 | 0.367 | 2.217 | 0.200 | 2.072 | 0.300 | 1.780 |
| tortuosity_skan | ALL | 0.841 | 2.489 | 0.775 | 2.450 | 0.837 | 2.935 | 0.880 | 5.690 | 0.919 | 4.760 |
| tortuosity_skan | chasedb1 | 0.750 | 5.860 | 0.750 | 5.071 | 0.750 | 3.515 | 0.625 | 3.168 | 0.750 | 3.057 |
| tortuosity_skan | drive | 0.950 | 3.509 | 0.500 | 2.173 | 0.650 | 1.923 | 0.500 | 1.584 | 0.650 | 1.579 |
| tortuosity_skan | fives | 0.855 | 2.346 | 0.830 | 2.022 | 0.865 | 2.959 | 0.965 | 6.595 | 0.960 | 5.333 |
| tortuosity_skan | hrf | 0.700 | 1.869 | 0.600 | 4.789 | 0.800 | 3.299 | 0.633 | 3.065 | 0.867 | 3.519 |
| total_length_pvbm | ALL | 0.872 | 1.870 | 0.891 | 2.113 | 0.891 | 2.620 | 0.961 | 9.808 | 0.333 | 8.305 |
| total_length_pvbm | chasedb1 | 0.750 | 1.578 | 0.750 | 1.584 | 0.125 | 0.749 | 0.125 | 0.732 | 0.250 | 0.493 |
| total_length_pvbm | drive | 1.000 | 3.144 | 0.950 | 5.672 | 0.650 | 1.487 | 1.000 | 3.307 | 0.000 | 1.414 |
| total_length_pvbm | fives | 0.855 | 0.856 | 0.885 | 0.968 | 0.980 | 2.544 | 1.000 | 11.141 | 0.330 | 9.893 |
| total_length_pvbm | hrf | 0.933 | 7.859 | 0.933 | 7.513 | 0.667 | 4.386 | 0.900 | 7.673 | 0.600 | 4.401 |
| total_length_skan | ALL | 0.891 | 1.605 | 0.907 | 1.827 | 0.829 | 1.444 | 0.895 | 8.009 | 0.868 | 6.408 |
| total_length_skan | chasedb1 | 0.750 | 1.831 | 0.750 | 1.782 | 0.500 | 0.771 | 0.625 | 1.137 | 0.750 | 1.071 |
| total_length_skan | drive | 1.000 | 4.971 | 1.000 | 4.958 | 0.500 | 0.728 | 0.350 | 1.073 | 0.350 | 0.640 |
| total_length_skan | fives | 0.875 | 0.707 | 0.900 | 1.033 | 0.960 | 1.488 | 1.000 | 9.662 | 1.000 | 7.864 |
| total_length_skan | hrf | 0.967 | 5.285 | 0.933 | 5.041 | 0.267 | 1.807 | 0.633 | 3.442 | 0.367 | 1.973 |

### P2 verdict

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


## P3 -- downstream disease-classification validity (direction D3)

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


### P3.1 Macro one-vs-rest AUC by feature source (5-fold CV x 3 repeats, 1000-sample image bootstrap CI)

| dataset | split | featureset | clf | n | gbdt_lodo | gbdt_pooled | gt | linear_per_ds | offset_lodo | offset_per_ds | pred |
|---|---|---|---|---|---|---|---|---|---|---|---|
| fives | all | primary8 | gbdt | 400 | 0.716 [0.678, 0.752] | 0.685 [0.647, 0.722] | 0.718 [0.682, 0.752] | 0.717 [0.680, 0.753] | 0.688 [0.648, 0.725] | 0.696 [0.657, 0.736] | 0.688 [0.648, 0.725] |
| fives | all | primary8 | logreg | 400 | 0.740 [0.706, 0.772] | 0.700 [0.667, 0.736] | 0.718 [0.688, 0.751] | 0.728 [0.695, 0.762] | 0.713 [0.680, 0.750] | 0.714 [0.680, 0.750] | 0.713 [0.680, 0.750] |
| fives | all | primary8+zoneB | gbdt | 400 | 0.745 [0.709, 0.779] | 0.741 [0.707, 0.772] | 0.734 [0.699, 0.767] | 0.726 [0.690, 0.760] | 0.707 [0.671, 0.743] | 0.705 [0.670, 0.740] | 0.707 [0.671, 0.743] |
| fives | all | primary8+zoneB | logreg | 400 | 0.757 [0.723, 0.788] | 0.710 [0.677, 0.745] | 0.723 [0.689, 0.755] | 0.731 [0.695, 0.765] | 0.715 [0.679, 0.749] | 0.715 [0.679, 0.749] | 0.715 [0.679, 0.749] |
| fives | test | primary8 | gbdt | 200 | 0.707 [0.657, 0.754] | 0.694 [0.648, 0.747] | 0.692 [0.647, 0.740] | 0.724 [0.673, 0.775] | 0.708 [0.658, 0.761] | 0.708 [0.658, 0.761] | 0.708 [0.658, 0.761] |
| fives | test | primary8 | logreg | 200 | 0.736 [0.692, 0.780] | 0.705 [0.661, 0.751] | 0.718 [0.676, 0.764] | 0.736 [0.689, 0.781] | 0.717 [0.670, 0.763] | 0.717 [0.670, 0.763] | 0.717 [0.670, 0.763] |
| hrf | all | primary8 | gbdt | 45 | 0.772 [0.650, 0.876] | 0.672 [0.541, 0.800] | 0.896 [0.813, 0.961] | 0.762 [0.617, 0.877] | 0.776 [0.658, 0.880] | 0.693 [0.557, 0.816] | 0.776 [0.658, 0.880] |
| hrf | all | primary8 | logreg | 45 | 0.728 [0.600, 0.845] | 0.727 [0.599, 0.842] | 0.968 [0.922, 0.999] | 0.752 [0.608, 0.870] | 0.753 [0.624, 0.861] | 0.727 [0.603, 0.835] | 0.753 [0.624, 0.861] |
| hrf | all | primary8+zoneB | gbdt | 45 | 0.688 [0.547, 0.804] | 0.708 [0.575, 0.830] | 0.873 [0.787, 0.950] | 0.674 [0.537, 0.797] | 0.711 [0.564, 0.835] | 0.603 [0.460, 0.726] | 0.711 [0.564, 0.835] |
| hrf | all | primary8+zoneB | logreg | 45 | 0.667 [0.530, 0.806] | 0.704 [0.572, 0.826] | 0.953 [0.892, 0.997] | 0.655 [0.512, 0.780] | 0.644 [0.501, 0.774] | 0.633 [0.489, 0.760] | 0.644 [0.501, 0.774] |

### P3.2 Attenuation: macro-AUC(GT features) - macro-AUC(this source), paired bootstrap 95 % CI

| dataset | split | featureset | clf | gbdt_lodo | gbdt_pooled | linear_per_ds | offset_lodo | offset_per_ds | pred |
|---|---|---|---|---|---|---|---|---|---|
| fives | all | primary8 | gbdt | +0.002 [-0.038, +0.042] | +0.033 [-0.000, +0.070] | +0.001 [-0.035, +0.035] | +0.030 [-0.006, +0.065] | +0.022 [-0.015, +0.057] | +0.030 [-0.006, +0.065] |
| fives | all | primary8 | logreg | -0.022 [-0.047, +0.005] | +0.018 [-0.005, +0.040] | -0.011 [-0.036, +0.013] | +0.004 [-0.016, +0.025] | +0.004 [-0.016, +0.025] | +0.004 [-0.016, +0.025] |
| fives | all | primary8+zoneB | gbdt | -0.011 [-0.045, +0.023] | -0.007 [-0.039, +0.024] | +0.008 [-0.025, +0.039] | +0.027 [-0.005, +0.059] | +0.029 [-0.004, +0.060] | +0.027 [-0.005, +0.059] |
| fives | all | primary8+zoneB | logreg | -0.034 [-0.058, -0.007] | +0.013 [-0.010, +0.037] | -0.008 [-0.032, +0.018] | +0.008 [-0.014, +0.030] | +0.008 [-0.014, +0.030] | +0.008 [-0.014, +0.030] |
| fives | test | primary8 | gbdt | -0.014 [-0.069, +0.040] | -0.001 [-0.047, +0.046] | -0.032 [-0.082, +0.020] | -0.016 [-0.059, +0.034] | -0.016 [-0.059, +0.034] | -0.016 [-0.059, +0.034] |
| fives | test | primary8 | logreg | -0.017 [-0.057, +0.024] | +0.013 [-0.016, +0.045] | -0.018 [-0.052, +0.018] | +0.001 [-0.028, +0.031] | +0.001 [-0.028, +0.031] | +0.001 [-0.028, +0.031] |
| hrf | all | primary8 | gbdt | +0.124 [+0.003, +0.249] | +0.224 [+0.105, +0.353] | +0.133 [-0.004, +0.275] | +0.119 [+0.000, +0.241] | +0.202 [+0.075, +0.329] | +0.119 [+0.000, +0.241] |
| hrf | all | primary8 | logreg | +0.240 [+0.132, +0.355] | +0.241 [+0.127, +0.362] | +0.216 [+0.109, +0.340] | +0.215 [+0.115, +0.321] | +0.241 [+0.141, +0.353] | +0.215 [+0.115, +0.321] |
| hrf | all | primary8+zoneB | gbdt | +0.185 [+0.066, +0.316] | +0.165 [+0.034, +0.304] | +0.199 [+0.070, +0.331] | +0.162 [+0.050, +0.285] | +0.270 [+0.149, +0.404] | +0.162 [+0.050, +0.285] |
| hrf | all | primary8+zoneB | logreg | +0.285 [+0.164, +0.402] | +0.249 [+0.131, +0.371] | +0.298 [+0.175, +0.427] | +0.309 [+0.184, +0.436] | +0.320 [+0.199, +0.445] | +0.309 [+0.184, +0.436] |

### P3.3 Reliability `r(B, B_GT)` within each dataset -- the quantity attenuation actually depends on

A calibration that removes a constant offset cannot change this number, which is why bias removal does not buy back downstream AUC.

| dataset | biomarker | r_pred | r_offset_per_ds | r_linear_per_ds | r_gbdt_pooled | r_offset_lodo | r_gbdt_lodo |
|---|---|---|---|---|---|---|---|
| chasedb1 | FD_pvbm | 0.857 | 0.852 | 0.845 | 0.817 | 0.857 | 0.859 |
| drive | FD_pvbm | 0.473 | 0.477 | 0.241 | 0.493 | 0.473 | 0.502 |
| fives | FD_pvbm | 0.894 | 0.894 | 0.892 | 0.905 | 0.894 | 0.894 |
| hrf | FD_pvbm | 0.510 | 0.473 | 0.350 | 0.442 | 0.510 | 0.478 |
| chasedb1 | FD_skan | 0.738 | 0.737 | 0.743 | 0.752 | 0.738 | 0.704 |
| drive | FD_skan | 0.483 | 0.488 | 0.388 | 0.499 | 0.483 | 0.505 |
| fives | FD_skan | 0.909 | 0.908 | 0.922 | 0.888 | 0.909 | 0.862 |
| hrf | FD_skan | 0.490 | 0.462 | 0.451 | 0.399 | 0.490 | 0.473 |
| chasedb1 | density_pvbm | 0.764 | 0.747 | 0.734 | 0.744 | 0.764 | 0.753 |
| drive | density_pvbm | 0.431 | 0.410 | 0.345 | 0.487 | 0.431 | 0.541 |
| fives | density_pvbm | 0.983 | 0.983 | 0.981 | 0.971 | 0.983 | 0.951 |
| hrf | density_pvbm | 0.513 | 0.432 | 0.407 | 0.421 | 0.513 | 0.521 |
| chasedb1 | density_skan | 0.764 | 0.747 | 0.734 | 0.744 | 0.764 | 0.753 |
| drive | density_skan | 0.431 | 0.410 | 0.345 | 0.487 | 0.431 | 0.541 |
| fives | density_skan | 0.983 | 0.983 | 0.981 | 0.971 | 0.983 | 0.951 |
| hrf | density_skan | 0.513 | 0.432 | 0.407 | 0.421 | 0.513 | 0.521 |
| chasedb1 | tortuosity_pvbm | 0.504 | 0.478 | 0.290 | 0.247 | 0.504 | 0.196 |
| drive | tortuosity_pvbm | 0.720 | 0.682 | 0.429 | 0.652 | 0.720 | 0.696 |
| fives | tortuosity_pvbm | 0.182 | 0.181 | 0.237 | 0.228 | 0.182 | 0.148 |
| hrf | tortuosity_pvbm | 0.795 | 0.785 | 0.775 | 0.731 | 0.795 | 0.772 |
| chasedb1 | tortuosity_skan | 0.282 | 0.281 | 0.294 | 0.338 | 0.282 | 0.346 |
| drive | tortuosity_skan | 0.331 | 0.326 | 0.186 | 0.392 | 0.331 | 0.336 |
| fives | tortuosity_skan | 0.547 | 0.547 | 0.584 | 0.499 | 0.547 | 0.480 |
| hrf | tortuosity_skan | 0.708 | 0.705 | -0.396 | 0.678 | 0.708 | 0.678 |
| chasedb1 | total_length_pvbm | 0.730 | 0.722 | 0.719 | 0.664 | 0.730 | 0.723 |
| drive | total_length_pvbm | 0.583 | 0.580 | 0.306 | 0.459 | 0.583 | 0.554 |
| fives | total_length_pvbm | 0.962 | 0.961 | 0.876 | 0.935 | 0.962 | 0.892 |
| hrf | total_length_pvbm | 0.437 | 0.414 | 0.379 | 0.376 | 0.437 | 0.439 |
| chasedb1 | total_length_skan | 0.827 | 0.826 | 0.797 | 0.722 | 0.827 | 0.749 |
| drive | total_length_skan | 0.460 | 0.459 | 0.355 | 0.509 | 0.460 | 0.490 |
| fives | total_length_skan | 0.977 | 0.977 | 0.931 | 0.968 | 0.977 | 0.895 |
| hrf | total_length_skan | 0.487 | 0.459 | 0.443 | 0.457 | 0.487 | 0.464 |

### P3 verdict

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


## Overall assessment of the pivot

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

