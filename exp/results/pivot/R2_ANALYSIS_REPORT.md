# R2 -- revision analyses for the CMIG positioning review (2026-09-18)

Covers reviewer points **3, 4, 5, 8, 10, 11, 15, 20** of
`review/user_cmig_review_20260918.md`.

Everything below is produced by the `r2_*` scripts under `exp/src/pivot/`, writes
into `exp/results/pivot/r2/`, and touches neither `paper2/` nor any GPU job.
Environment: Python 3.11.16 (`D:/Anaconda/envs/medical1`), single-threaded BLAS
(`OMP/MKL/OPENBLAS_NUM_THREADS=1`). Every CSV carries a `source` column naming
the file(s) each number came from. Unless stated otherwise the primary panel is
the **four skan biomarkers** (FD, tortuosity, density, total length) with the
PVBM duplicates retained as a sensitivity panel, and the sigma axis is the frozen
`results/gateA_biomarker_scales_train.csv`.

The design was additionally adjusted on 2026-09-18 after an external methods
consultation (`review/cmig_plan_reply.md`): the common scale now has the one
definition that consultation accepts, raw units are the primary result, the
proportional-distortion arm has a locked-deployment variant reporting decision
drift rather than AUC, the injection panel's label-dependent arm is demoted to
an adversarial stress test, and topology re-matching follows the consultation's
covariate-priority order with balance diagnostics at each level.

| script | reviewer point | outputs |
|---|---|---|
| `r2_scale.py` | 3, 4 | **`r2_headline_raw_primary.csv`** (raw units primary), `r2_common_scale_formula.txt`, `r2_sigma_estimators.csv`, `r2_audit_raw_units.csv`, `r2_scale_sensitivity.csv`, `r2_scale_claims.csv` |
| `r2_calibration.py` | 5 | **`r2_main_table.csv`** (mu / alpha / beta / residual SD / CCC / r / AUC gap), `r2_calibration.csv`, `r2_dose_proportional.csv`, `r2_dose_proportional_fixedrule.csv`, `figs/pivot/r2_calibration_<ds>.png\|pdf` |
| `r2_injection_panel.py` | 8 | `r2_injection_panel.csv`, `r2_injection_panel_summary.csv`, `figs/pivot/r2_injection_panel.png\|pdf` |
| `r2_pixel_baselines.py` | 10 | `r2_pixel_baselines.csv`, `r2_pixel_vs_literature.csv` (+ `pixel_literature.csv`, 97 rows with source URLs) |
| `r2_fives_full.py` | 11 | `r2_fives_audit_800.csv`, `r2_fives_downstream_800.csv`, `r2_fives_subsample.csv`, `fives_oof_coverage.csv`, `bio_master_full.csv` |
| `r2_delta_auc.py` | 15 | `r2_delta_auc.csv` |
| `r2_topology_matching.py` | 20 | `r2_topology_covariates.csv`, `r2_topology_balance.csv`, `r2_topology_matched.csv`, `r2_topology_strata.csv`, `r2_topology_claims.csv`, `reconnector_description.md` |

---

## Points 3 and 4 -- sigma is a within-dataset scale, written out and stress-tested

### 4a. The estimator, exactly

    sigma(dataset, biomarker) = 1.4826 * median_i | x_i - median_j x_j |

estimated on the **training-split observer-1 reference masks only**, in native
units, one value per (dataset, biomarker). Reproducing it from
`bio_master.csv` returns the frozen Gate A value **bit-identically on DRIVE,
CHASE_DB1 and HRF** (ratio 1.000). On FIVES the ratio is 0.836--1.031, because
the frozen scale was estimated on 120 training images and `bio_master.csv` now
holds 200. Recomputed on **all 600** FIVES training reference masks (now that
point 11 has them) the ratios to the frozen scale are FD 0.975, tortuosity
**0.835**, density 0.896, total length 1.005 -- so the drift is not a
small-sample accident that goes away with more data; the frozen 120-image
tortuosity scale is genuinely ~20 % larger than the 600-image one. That is
itself a finding: **sigma moves when the reference sample moves**, and the
manuscript must say which sample defines it. The audit tables keep the frozen
scale (it is the pre-registered one and changing it now would be
outcome-driven); this is reported as a caveat, not applied as a correction.

The two alternatives, also written out:

    ordinary SD   sigma = sqrt( (1/(n-1)) * sum_i (x_i - mean_j x_j)^2 )
    IQR           sigma = ( Q75(x) - Q25(x) ) / 1.349

### 4b. sigma itself is imprecise, and that uncertainty is not in any CI we report

2000-draw image bootstrap of sigma, mean relative width `(hi-lo)/sigma` over the
primary panel:

| dataset | n (train) | 1.4826*MAD | SD | IQR/1.349 |
|---|---:|---:|---:|---:|
| HRF | 15 | **1.79** | 0.58 | 1.36 |
| CHASE_DB1 | 20 | 1.37 | 0.61 | 1.07 |
| DRIVE | 20 | 1.30 | 0.76 | 1.42 |
| FIVES | 200 | 0.41 | 0.26 | 0.43 |

Worst single cell: HRF `total_length_skan`, sigma = 6457 px with CI
[2074, 14992] -- a factor of seven. **Every HRF number quoted in sigma units
carries roughly a factor-of-two scale uncertainty on top of its own CI.** This
is the strongest available argument for reporting raw units alongside, which is
what point 3 asks for.

### 3a. The audit in raw units (`r2_audit_raw_units.csv`, primary skan panel, test split, seg seed 0)

| dataset | n | FD offset | tortuosity offset | density offset (area fraction) | total length offset (px) | total length offset (FOV diameters) |
|---|---:|---:|---:|---:|---:|---:|
| DRIVE | 20 | -0.0355 | -0.0240 | +0.0019 | -1417 | -2.635 |
| CHASE_DB1 | 8 | -0.0184 | -0.0124 | +0.0124 | -644 | -0.700 |
| HRF | 30 | -0.0310 | -0.0134 | +0.0067 | -11550 | -3.893 |
| FIVES | 200 | -0.0222 | -0.0008 | -0.0005 | -1428 | -0.713 |

Each cell also carries its 95 % image-bootstrap CI, the residual SD and the
reference median in the same units in the CSV. Pearson/Spearman r are scale-free
and therefore identical in raw, FOV-normalised and sigma representations --
which is the cleanest way to make the reviewer's point in the paper: the
*fidelity* axis is a common unit across datasets, the *offset* and *residual SD*
axes are not.

FOV normalisation uses the per-image FOV diameter already measured in
`bio_master.csv::disc_fov_diameter` (detected from the fundus image alone, so it
never sees a mask). It applies only to `total_length`; FD, tortuosity and density
are dimensionless already.

### 3c. The COMMON reference scale, in the one form the methods consultation accepts

`review/cmig_plan_reply.md` section 2A rejects "pool all the raw values and take
a MAD" (it confounds cohort location shift, sample size and within-cohort
biology) and specifies one definition. It is implemented verbatim in
`r2_scale.py::common_scale` and written to
`results/pivot/r2/r2_common_scale_formula.txt`:

    For biomarker k, using REFERENCE-MASK measurements x_ick^ref only:
      (1) per-cohort reference centre   m_ck  = median_i x_ick^ref
      (2) cohort-centred deviations     d_ick = x_ick^ref - m_ck
      (3) common scale                  s_k   = 1.4826 * weighted_median_ick |d_ick|,
          with weights w_ick = 1/n_c so every cohort carries equal total weight
      (4) every cohort and every segmenter then uses the SAME denominator:
          e_ick = ( x_ick^pred - x_ick^ref ) / s_k

The denominator never depends on model performance; FIVES' n = 800 cannot
dominate it; between-cohort location shifts cannot inflate it. Length-like
biomarkers are FOV-diameter normalised before step (1), because a pixel count is
not comparable across cohorts of different resolution. **The same `s_k` applies
unchanged to the second segmenter family the GPU job is training** -- that is
the point of fixing it now.

Values (FOV-normalised units, all four training splits, equal cohort weight):
FD_skan **0.02709**, tortuosity_skan **0.008819**, density_skan **0.014138**,
total_length_skan **2.06425 FOV diameters** (PVBM: 0.021387, 0.006052,
0.014138, 1.80960).

Under this scale the per-dataset primary-panel means are

| dataset | mean \|offset\| | mean residual SD |
|---|---:|---:|
| DRIVE | 1.361 | 1.893 |
| CHASE_DB1 | 0.823 | 1.146 |
| HRF | 1.258 | 1.191 |
| FIVES | 0.322 | 0.828 |

**Raw units stay primary.** `r2_headline_raw_primary.csv` is laid out that way:
the raw (or FOV-normalised) offset, residual SD and r with their CIs are the
primary columns, and `offset_sigma_own` / `offset_sigma_common` are explicitly
labelled sensitivity columns. The common scale must not be renamed a "natural
unit" or a "population SD" anywhere in the manuscript.

### 3b/4c. Sensitivity: estimators x reference cohorts

Thirteen variants in total. `own` = each dataset's own training split (status
quo, x 3 estimators x raw/FOV-normalised); `common` = the accepted definition
above (one estimator by construction); `pooled_naive` = the naive pooled MAD/SD/
IQR the consultation rejects, retained only so the paper can show *why* it is
rejected; `fives` = FIVES' training split transported to every dataset.
Pooling or transporting a scale is only defined on a dimensionless
representation, so all but `own` are computed on the FOV-normalised columns.

Per-dataset mean over the primary panel:

| cohort | estimator | DRIVE off / SD | CHASE off / SD | HRF off / SD | FIVES off / SD |
|---|---|---|---|---|---|
| own | MAD | 0.927 / 1.233 | 1.088 / 1.125 | **1.540 / 1.480** | **0.249 / 0.779** |
| own | SD | 0.741 / 0.957 | 0.806 / 0.944 | 1.381 / 1.337 | 0.224 / 0.674 |
| own | IQR | 0.989 / 1.316 | 0.861 / 0.959 | 1.502 / 1.426 | 0.245 / 0.770 |
| **common (accepted)** | **1.4826 x wtd-median MAD** | **1.361 / 1.893** | **0.823 / 1.146** | **1.258 / 1.191** | **0.322 / 0.828** |
| pooled_naive (rejected) | MAD | 1.086 / 1.608 | 0.659 / 1.029 | 0.933 / 0.925 | 0.212 / 0.635 |
| pooled_naive (rejected) | SD | 0.750 / 1.117 | 0.471 / 0.704 | 0.664 / 0.659 | 0.149 / 0.443 |
| pooled_naive (rejected) | IQR | 1.090 / 1.601 | 0.647 / 1.029 | 0.929 / 0.916 | 0.213 / 0.634 |
| FIVES-as-calibrator | MAD | 1.356 / 2.018 | 0.789 / 1.326 | 1.107 / 1.107 | 0.246 / 0.777 |
| FIVES-as-calibrator | SD | 1.170 / 1.711 | 0.676 / 1.111 | 0.975 / 0.961 | 0.225 / 0.675 |
| FIVES-as-calibrator | IQR | 1.345 / 1.997 | 0.781 / 1.311 | 1.100 / 1.097 | 0.247 / 0.772 |

### Which headline claims survive (`r2_scale_claims.csv`)

**All of them, in 13 of 13 variants.**

* **HRF vs FIVES ordering.** HRF mean |offset| / FIVES mean |offset| =
  **3.90--6.24** (**3.90** under the accepted common scale); HRF mean residual
  SD / FIVES = **1.42--1.99** (**1.44** under the common scale). The ordering
  never inverts and never comes close to inverting.
* **"Orders of magnitude between the axes."** The largest primary-panel mean
  |offset| or residual SD divided by the largest primary-panel paired topology
  harm is **56x to 102x** depending on the variant (**95x** under the common
  scale) -- i.e. between one and two orders of magnitude, as claimed.
* One ordering that *does* move under the common scale: DRIVE overtakes HRF as
  the cohort with the largest mean |offset| (1.361 vs 1.258), because DRIVE's
  own reference spread is small and its own-sigma numbers were being deflated.
  The manuscript never claims an ordering among DRIVE / CHASE_DB1 / HRF, so
  nothing breaks -- but it must not start claiming one.
* What *does* move: the absolute sigma-scaled numbers. The ordinary-SD estimator
  shrinks everything by 10--25 % (because sigma_SD > sigma_MAD for these
  right-skewed reference distributions); a pooled common scale shrinks FIVES to
  0.15--0.21 sigma and HRF to 0.66--0.93 sigma. **Any sentence quoting a
  particular sigma value must name the cohort and the estimator.**

### What the paper must change

1. Define sigma in the text with the formula and say it is a *within-dataset
   effect-size scale*, not a cross-dataset physical unit.
2. Add the raw-unit table (main text or supplement) and state the FOV-diameter
   normalisation for total length.
3. Report the bootstrap CI of sigma, at least for HRF, and note that
   sigma-scaled CIs do not include it.
4. Keep the HRF-vs-FIVES ordering and the "orders of magnitude" statement; cite
   `r2_scale_claims.csv` as the sensitivity that supports them.
5. State that FIVES' frozen sigma came from 120 training images.

---

## Point 5 -- calibration, CCC, Bland-Altman, and the proportional-distortion arm

### 5a. Method-comparison battery (`r2_calibration.csv`, figures `figs/pivot/r2_calibration_<ds>.png|pdf`)

Deming with lambda = 1 (orthogonal regression) is primary; OLS and Deming
lambda = 4 are carried as sensitivities. Everything is on the sigma axis
**centred on the cohort's reference median** (`ref_median_raw` in the CSV), so
that `alpha` is the interpretable quantity -- the predicted-minus-reference
offset a median-valued image carries -- instead of a meaningless intercept at
B = 0 (tortuosity sits at ~130 sigma). Centring subtracts the same constant from
both variables, so beta, CCC, r, the residual SD, the Bland-Altman limits and
the proportional-bias slope are all unaffected. 2000-draw paired image
bootstrap. Primary skan panel, Deming lambda = 1:

| dataset | biomarker | n | alpha [CI] | beta [CI] | CCC | BA mean diff [LoA] | prop-bias slope (p) |
|---|---|---:|---|---|---:|---|---|
| DRIVE | FD | 20 | -1.208 [-1.683, -0.818] | 0.767 [0.203, 2.865] | 0.201 | -1.17 [-2.95, 0.62] | -0.155 (0.62) |
| DRIVE | tortuosity | 20 | -0.719 [-1.007, -0.441] | 0.034 [-0.036, 0.793] | 0.043 | -1.36 [-5.97, 3.26] | **-1.621 (<1e-4)** |
| DRIVE | density | 20 | +0.126 [-0.508, 0.593] | 0.717 [-1.275, 3.225] | 0.323 | +0.13 [-1.72, 1.98] | -0.167 (0.62) |
| DRIVE | total length | 20 | -1.050 [-1.305, -0.815] | 0.454 [-0.034, 1.450] | 0.134 | -1.05 [-2.47, 0.37] | -0.436 (0.19) |
| CHASE_DB1 | FD | 8 | -0.758 [-0.959, -0.568] | 0.820 [0.650, 1.245] | 0.603 | -0.79 [-1.38, -0.19] | -0.192 (0.21) |
| CHASE_DB1 | tortuosity | 8 | -1.250 [-10.88, 10.96] | 1.253 [-12.7, 21.1] | 0.103 | -1.12 [-6.75, 4.51] | +0.049 (0.95) |
| CHASE_DB1 | density | 8 | +1.948 [1.194, 2.448] | 1.117 [0.575, 1.563] | 0.571 | +1.96 [0.38, 3.54] | +0.106 (0.56) |
| CHASE_DB1 | total length | 8 | -0.433 [-0.735, -0.238] | 0.666 [0.380, 0.908] | 0.761 | -0.49 [-1.50, 0.52] | **-0.388 (0.041)** |
| HRF | FD | 30 | -1.695 [-2.046, -1.373] | 0.479 [-0.079, 1.241] | 0.116 | -1.63 [-3.88, 0.63] | -0.359 (0.20) |
| HRF | tortuosity | 30 | -1.474 [-1.729, -1.143] | 0.367 [0.119, 0.693] | 0.402 | -2.31 [-7.80, 3.17] | **-0.821 (<1e-4)** |
| HRF | density | 30 | +0.461 [0.222, 0.701] | 0.814 [0.522, 1.275] | 0.570 | +0.43 [-0.90, 1.76] | -0.164 (0.35) |
| HRF | total length | 30 | -1.875 [-2.213, -1.531] | 0.415 [-0.082, 0.952] | 0.121 | -1.79 [-4.32, 0.74] | -0.451 (0.10) |
| FIVES | FD | 200 | -0.476 [-0.572, -0.375] | **1.582 [1.248, 1.932]** | 0.787 | -0.63 [-2.64, 1.38] | **+0.433 (<1e-4)** |
| FIVES | tortuosity | 200 | +0.001 [-0.096, 0.110] | 0.576 [0.198, 0.996] | 0.407 | -0.09 [-2.73, 2.54] | **-0.336 (2e-4)** |
| FIVES | density | 200 | -0.014 [-0.042, 0.013] | 1.027 [0.992, 1.061] | **0.984** | -0.02 [-0.47, 0.42] | **+0.026 (0.040)** |
| FIVES | total length | 200 | -0.245 [-0.282, -0.209] | 0.988 [0.956, 1.019] | **0.962** | -0.24 [-0.77, 0.28] | -0.012 (0.41) |

(The centred `alpha` reproduces the audit's constant offset to within the
slope correction, e.g. HRF FD -1.695 against a `bias_const` of -1.627; the
residual difference is exactly `(beta - 1) x (mean - median)` and is the part a
constant-offset model cannot represent.)

**Three findings the manuscript does not currently contain.**

1. **FIVES is not uniformly high-fidelity.** Density (CCC 0.984, beta 1.027) and
   total length (CCC 0.962, beta 0.988) are essentially calibrated. But FIVES
   **FD has a Deming slope of 1.582 with a CI that excludes 1** (OLS 1.387) and a
   Bland-Altman proportional bias of +0.433 (p < 1e-4): the segmentation
   *stretches* the between-image spread of fractal dimension. "Where fidelity is
   high (FIVES)" should be narrowed to the density/length columns.
2. **HRF is attenuated, not merely shifted.** All four slopes are below 1
   (0.37--0.81) and the CCCs are 0.12--0.57. A constant-offset description
   understates what is happening there.
3. **Proportional bias is significant in 6 of 16 primary cells**
   (DRIVE tortuosity, CHASE_DB1 total length, HRF tortuosity, FIVES FD /
   tortuosity / density). The manuscript's "constant offset + residual scatter"
   decomposition is incomplete in exactly the way the reviewer suspected, and
   the slope belongs in the main table.

A scatter + Bland-Altman figure per dataset is written for the writing pass:
`figs/pivot/r2_calibration_{drive,chasedb1,hrf,fives}.{png,pdf}` (4 rows x 2
columns; left = identity-line scatter with the OLS and Deming fits, right =
Bland-Altman with mean difference, 95 % limits and the proportional-bias line).

### 5a-bis. The main-text table (`r2_main_table.csv`)

The methods consultation (point 24) fixes the main-text column set as
**mu, alpha, beta, residual SD, CCC, r, AUC gap**, and says CCC must not sit
below r in prominence. `r2_main_table.csv` is that table, one row per
(dataset, biomarker), with `mu` = the Bland-Altman mean difference (identical to
the audit's constant offset), `alpha` = the Deming intercept in the
median-centred frame, every quantity carrying its bootstrap CI, plus the
proportional-bias slope with CI and p, and the dataset-level AUC gap joined from
`r2_delta_auc.csv` (skan4 / logreg). The writing pass should take the main table
from this file rather than assembling it from `r2_calibration.csv`.

### 5b. Proportional-distortion arm (`r2_dose_proportional.csv`, `r2_dose_proportional_fixedrule.csv`)

`B'_i = mean + k (B_i - mean)` per biomarker, `k in {0.5, 0.75, 1.25, 1.5, 2}`,
applied to the **reference** biomarkers of HRF and FIVES and traced through the
identical downstream protocol as `e1_dose_response.py`.

**With a retrained classifier the arm changes the AUC by exactly zero --
0.000000 in all 20 injected cells (2 datasets x 2 classifiers x 5 doses).**
The reason is mechanical and should be stated rather than presented as an
empirical surprise: a per-feature affine map is undone by the `StandardScaler`
in front of the logistic regression, and a tree ensemble is invariant to any
monotone per-feature transform, so the CV-fitted models are literally identical.

**With a fixed rule it does not.** Fitting on the undistorted reference features
and applying the frozen model to the distorted ones gives

| dataset | clf | k=0.5 | k=0.75 | k=1.25 | k=1.5 | k=2 |
|---|---|---:|---:|---:|---:|---:|
| HRF | logreg | -0.0089 | -0.0044 | -0.0007 | +0.0007 | +0.0015 |
| HRF | gbdt | -0.0126 | +0.0326 | +0.0119 | +0.0185 | +0.0052 |
| FIVES | logreg | -0.0042 | -0.0015 | +0.0009 | +0.0019 | +0.0022 |
| FIVES | gbdt | -0.0321 | -0.0149 | +0.0110 | -0.0141 | -0.0266 |

So the correct sentence for the paper is: *a proportional distortion of the
biomarker scale is invisible to a classifier that is refitted on the distorted
features, and visible to any fixed threshold, published cut-off or frozen model
-- the invariance is a property of the retrained-classifier protocol, not of the
biomarker.* This is the same narrowing the reviewer already asked for on the
constant-offset claim (point 6).

---

## Point 8 -- robustness panel for the residual-injection dose-response

`r2_injection_panel.py` re-runs the dose-response under six error models on HRF
and FIVES, calibrating each to the same target image-level Pearson r and
plotting AUC against the *achieved* r. Priority per the methods consultation:
**Gaussian independent, empirical residual bootstrap and heteroscedastic are
the required arms; the label-dependent arm is an adversarial worst-case stress
test** and is reported as such, not as part of the main narrative.

| mechanism | what it changes | status |
|---|---|---|
| `gaussian_indep` | independent Gaussian per biomarker | required |
| `gaussian_corr` | Gaussian with the observed cross-biomarker residual correlation | **the shipped arm**, repeated for reference |
| `resid_bootstrap` | the observed de-biased residual *vectors* resampled whole (keeps the real marginal shape, skew, ties and dependence) | required |
| `hetero_reference` | noise multiplier proportional to the image's reference level | required |
| `hetero_quality` | noise multiplier driven by `feat_contrast`, a label-free image-quality proxy | required |
| `label_dependent` | per-class residual mean and SD taken from the observed residuals | **adversarial stress test only** |
| `proportional` | `B' = mean + gamma (B - mean)`; achieved r = 1 by construction | plotted as a separate marker |

10 draws per (mechanism, target r) for logistic regression, 5 for the GBDT;
HRF n = 45, FIVES n = 400 (quality proxy available on 45/45 and 400/400).

### What the panel shows: the injection was conservative, not friendly

AUC at the injected fidelity nearest the real measurement's own achieved r:

**HRF (real measurement: r = 0.549, AUC 0.7533 logreg / 0.7281 GBDT)**

| mechanism | logreg AUC @ r~0.55 | GBDT AUC @ r~0.56 |
|---|---:|---:|
| `hetero_quality` | 0.7414 | 0.6695 |
| `hetero_reference` | 0.7353 | 0.6852 |
| `gaussian_corr` (shipped) | 0.7328 | 0.6458 |
| `resid_bootstrap` | 0.7176 | 0.6812 |
| `gaussian_indep` | 0.6790 | 0.5785 |
| `label_dependent` (adversarial) | 0.5770 | 0.6333 |
| **real measurement** | **0.7533** | **0.7281** |

**The real measurement sits at or above every injection arm at its own
fidelity.** That is the opposite of the reviewer's concern: the shipped
correlated-Gaussian arm *overstates* the harm at the observed r
(Delta AUC -0.238 simulated against -0.217 real on logreg), and the naive
independent-Gaussian arm overstates it much more (-0.291). Only the adversarial
label-dependent arm is more damaging than reality (-0.393), which is what an
adversarial arm is for.

On **FIVES** the gap is larger in the same direction: at r ~ 0.90 every
non-adversarial arm already costs -0.042 to -0.051 AUC, while the real
measurement at a *lower* fidelity (r = 0.854) costs nothing (+0.003 logreg,
+0.022 GBDT, both in favour of the predicted features). Real segmentation error
on FIVES is structured so that it destroys far less ranking information than
independent noise of the same Pearson r.

### What this means for the manuscript

1. The claim "the real measurement lands on its own injection curve" should be
   sharpened to **"lands on or above its own injection curve"**, with the panel
   cited: the simulation is a conservative upper bound on the harm, under five
   non-adversarial error models.
2. Mechanism does matter, and in a direction worth one sentence: **preserving
   the cross-biomarker correlation structure matters more than Gaussianity**.
   The independent-Gaussian arm is the most damaging non-adversarial model
   (-0.291 vs -0.238 on HRF logreg), while the empirical residual bootstrap --
   which keeps the real marginal shape as well as the correlation -- lands
   between the two (-0.253). Heteroscedasticity in either the reference value or
   an image-quality proxy changes the curve very little.
3. The **label-dependent arm must be labelled adversarial** and kept out of the
   main comparison. It is far more damaging than reality, which is itself the
   finding: the real residuals are not label-informative in the way that arm
   assumes.

Outputs: `r2_injection_panel.csv` (per draw), `r2_injection_panel_summary.csv`
(mean and 2.5/97.5 percentile over draws), `figs/pivot/r2_injection_panel.png|pdf`
(AUC vs achieved r, one panel per dataset x classifier, six curves plus the real
measurement as a star and the proportional arm as separate markers at r = 1).

### The proportional-distortion arm under a LOCKED deployment
(`r2_dose_proportional_fixedrule.csv`)

The consultation is explicit that AUC is the wrong endpoint for a frozen rule --
it does not depend on the threshold at all -- so the locked arm now reports what
a locked rule actually moves: the argmax decision, macro sensitivity and
specificity, the **predicted positive rate per class** (decision drift), the mean
probability assigned to the true class, and the multi-class Brier score. Model,
preprocessing, calibration and decision rule are all fitted on the undistorted
reference features and applied frozen to the distorted ones.

| dataset / clf | gamma | Delta AUC | Delta accuracy | Delta Brier | predicted positive rate drift |
|---|---:|---:|---:|---:|---|
| HRF / logreg | 0.50 | -0.0089 | **-0.067** | **+0.120** | dr 0.289 -> **0.400** |
| HRF / logreg | 2.00 | +0.0015 | 0.000 | -0.038 | g 0.356 -> 0.400 |
| HRF / gbdt | 0.50 | -0.0126 | -0.022 | +0.026 | dr 0.333 -> 0.400 |
| FIVES / logreg | 0.50 | -0.0042 | -0.040 | +0.027 | Normal 0.355 -> **0.208**, DR 0.180 -> **0.318** |
| FIVES / logreg | 2.00 | +0.0022 | 0.000 | +0.039 | Normal 0.355 -> **0.445**, DR 0.180 -> **0.115** |
| FIVES / gbdt | 2.00 | -0.0266 | **-0.060** | +0.074 | AMD 0.238 -> 0.308 |

This is the sentence the paper needs: **a proportional distortion of the
biomarker scale is invisible to AUC and to any classifier refitted on the
distorted features (Delta AUC = 0.000000 in all 20 retrained cells), and yet it
moves a frozen decision rule's class-assignment rate by up to 14 percentage
points and its Brier score by up to +0.12.** Calibration drift and discrimination
loss are different failure modes, and only the second one is what the
retrained-classifier protocol can see.

---

## Point 10 -- absolute pixel quality of the baselines, against the published range

### 10a. Our baselines (`r2_pixel_baselines.csv`; seeds 0-2, test split, 2000-draw image-clustered bootstrap)

| dataset | n | Dice [CI] | IoU | clDice | sensitivity | specificity | accuracy | AUC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| DRIVE | 20 | **0.8262** [0.8213, 0.8319] | 0.7041 | 0.8331 | 0.8349 | 0.9736 | 0.9555 | 0.9806 |
| CHASE_DB1 | 8 | **0.8165** [0.8075, 0.8257] | 0.6901 | 0.8404 | 0.8572 | 0.9758 | 0.9650 | 0.9858 |
| HRF | 30 | **0.8136** [0.7995, 0.8283] | 0.6878 | 0.8224 | 0.8356 | 0.9784 | 0.9651 | 0.9856 |
| FIVES | 200 | **0.9021** [0.8874, 0.9139] | 0.8319 | 0.9066 | 0.8986 | 0.9921 | 0.9849 | 0.9913 |

Between-seed spread is at most 0.027 (CHASE_DB1 sensitivity) and at most 0.006
for every Dice. Architecture: nnU-Net-style U-Net, 7.77 M parameters.
**Working resolution matters and is on every row:** DRIVE and CHASE_DB1 are
native; **HRF and FIVES are inferred at `resize_longest = 1536`**, not at their
native 3504 and 2048.

### 10b. Against the literature (`pixel_literature.csv`, 97 rows, every row with its source URL; `r2_pixel_vs_literature.csv`)

| dataset | our Dice | n published | published min / median / max | fraction of published entries we exceed | plain-U-Net published range |
|---|---:|---:|---|---:|---|
| DRIVE | 0.8262 | 34 | 0.771 / 0.817 / 0.869 | 0.62 | 0.772--0.814 |
| CHASE_DB1 | 0.8165 | 32 | 0.732 / 0.804 / 0.849 | 0.78 | 0.790--0.806 |
| HRF | 0.8136 | 14 | 0.573 / 0.777 / 0.816 | 0.79 | 0.724 (single report) |
| FIVES | 0.9021 | 15 | 0.756 / 0.873 / 0.918 | 0.87 | 0.808--0.902 |

* Our Dice sits **above the published median on all four datasets** and above
  every published plain-U-Net baseline.
* clDice: we exceed every published value found on DRIVE, HRF and FIVES; clDice
  is reported in only 14 of 97 rows (four papers), and **no CHASE_DB1 clDice
  exists in the literature we could verify**, so reporting it is ahead of the
  field's norm.
* IoU: 0.71--1.00 quantile of the published entries.
* **The one metric where we are below the published range is specificity**
  (DRIVE 0.9736 vs a published 0.9803--0.9869; HRF 0.9784 vs 0.9837--0.9874).
  This is an operating-point choice -- a 0.5 threshold that buys sensitivity
  0.835--0.899 -- and the paper should say so rather than omit specificity.
* Two caveats the paper must state, both documented in the literature CSV's
  `notes`: (i) **HRF Dice is bimodal purely by evaluation resolution** --
  downsampled reports cluster at 0.707--0.800, the one native-resolution report
  at 0.814--0.816, so our 0.8136 at `resize_longest = 1536` must be quoted with
  its resolution; (ii) **AUC is not poolable across these papers** (two
  incompatible definitions are in circulation: 0.82--0.89 in some tables against
  0.97--0.99 in others), so we should not claim a rank on AUC.

---

## Point 11 -- the FIVES audit on all 800 images

### 11a. Were out-of-fold predictions available? Yes, all 600.

`runs/seg_oof/fives/pred/{prob,mask}` holds **600 of 600** cross-fitted
out-of-fold predictions for the FIVES training split (verified file by file;
`fives_oof_coverage.csv`). **Nothing is missing, so no request to the GPU job
was needed and `fives_oof_missing.csv` was not written.** The shipped
`bio_master.csv` capped the training split at 50 images per class (200 of 600)
purely to bound CPU time. `src.pivot.build_table --datasets fives --sources
gt,pred_oof --fives_train_n 150 --procs 12` filled in the remaining 400 images
(~27 min, CPU) into `results/pivot/r2/bio_master_full.csv`, which now carries
**600 gt + 600 pred_oof (train) and 200 gt + 200 pred_test (test) = all 800
FIVES images.**

### 11b. The audit on all 800 (`r2_fives_audit_800.csv`, primary skan panel, sigma units)

| block | n | mean \|offset\| | mean residual SD | mean r |
|---|---:|---:|---:|---:|
| test 200 (seed-0 held-out prediction) | 200 | 0.248 | 0.717 | 0.824 |
| **train 600 (cross-fitted OOF)** | 600 | **0.214** | **0.542** | **0.861** |
| train 200 (the shipped cap) | 200 | 0.230 | 0.503 | 0.898 |
| **all 800 pooled** | 800 | **0.211** | **0.592** | **0.850** |

Per biomarker on all 800: FD offset -0.581 [-0.637, -0.529] with r = 0.913;
tortuosity +0.017 [-0.061, +0.096] with r = 0.527; density -0.021 with
r = 0.984; total length -0.224 with r = 0.975. **The headline does not move**:
FIVES still has a small offset (0.21 sigma mean) and high image-level fidelity
(mean r 0.85).

One difference worth reporting: the shipped 200-image cap gave a *higher*
tortuosity r (0.729) than the full 600 (0.573) and a lower tortuosity residual
SD (0.744 vs 1.011), so **the cap was mildly optimistic about tortuosity
specifically** while being pessimistic about the downstream gap (below).

The two prediction sources are different objects -- one seed-0 model on held-out
images versus a cross-fitted ensemble on training images -- so the pooled block
is reported as a pooled description with the source recorded per row, never as a
silent merge.

### 11c. The reference-gap downstream AUC on all 800 (`r2_fives_downstream_800.csv`)

5-fold x 3-repeat stratified CV, macro one-vs-rest AUC, 1000-draw paired image
bootstrap -- identical protocol to `p3_downstream.py`.

| block | panel | clf | n | AUC_ref | AUC_pred | Delta AUC [CI] |
|---|---|---|---:|---:|---:|---|
| test 200 | skan4 | logreg | 200 | 0.7135 | 0.7135 | 0.0000 [-0.0280, +0.0278] |
| test 200 | skan4 | gbdt | 200 | 0.6869 | 0.7187 | -0.0319 [-0.0819, +0.0230] |
| train 600 | skan4 | logreg | 600 | 0.7271 | 0.7111 | +0.0160 [-0.0024, +0.0358] |
| train 600 | skan4 | gbdt | 600 | 0.7268 | 0.7096 | +0.0172 [-0.0118, +0.0439] |
| train 600 | primary8 | logreg | 600 | 0.7377 | 0.7272 | +0.0105 [-0.0072, +0.0302] |
| train 600 | primary8 | gbdt | 600 | 0.7518 | 0.7289 | +0.0229 [-0.0044, +0.0514] |
| train 200 (shipped cap) | primary8 | gbdt | 200 | 0.7012 | 0.6453 | **+0.0559 [+0.0039, +0.1132]** |
| **all 800** | skan4 | logreg | 800 | 0.7235 | 0.7103 | +0.0132 [-0.0016, +0.0283] |
| **all 800** | skan4 | gbdt | 800 | 0.7274 | 0.7265 | +0.0009 [-0.0210, +0.0235] |
| **all 800** | primary8 | logreg | 800 | 0.7322 | 0.7211 | +0.0111 [-0.0047, +0.0259] |
| **all 800** | primary8 | gbdt | 800 | 0.7511 | 0.7236 | **+0.0275 [+0.0056, +0.0495]** |

**One claim needs narrowing.** At n = 200 every FIVES interval covered zero and
the paper says the predicted-feature classifier "matches" the reference-feature
one. At n = 800 the point estimates are consistently small and positive
(+0.001 to +0.028) and **one of four pooled cells now excludes zero**
(primary8 + GBDT, +0.0275 [+0.0056, +0.0495]). The honest reading is that the
n = 200 result was *underpowered*, not that there is no gap: on FIVES the gap is
resolvable at n = 800 and is **about a tenth of the HRF gap** (+0.028 against
+0.180 to +0.217). The sentence should become *"on FIVES the gap is at most a
few AUC points and an order of magnitude smaller than on HRF"* rather than *"a
classifier on predicted biomarkers matches one on reference biomarkers"*.

Note the direction of the sampling artefact: the **shipped 200-image cap gave the
largest gap of any block** (+0.0559), so lifting the cap moved the number
*towards* the paper's claim, not away from it. Nothing was flattered by the cap.

### 11d. Stratified sub-sampling representativeness (`r2_fives_subsample.csv`)

Five class-stratified random subsamples of the 600 training images at n = 200
(50 per disease class), each re-running the audit and the downstream gap, against
the full-600 reference row.

| subsample | mean \|offset\| | mean residual SD | mean r | Delta AUC (logreg) | Delta AUC (gbdt) |
|---|---:|---:|---:|---:|---:|
| seed 0 | 0.223 | 0.534 | 0.878 | +0.0260 | +0.0118 |
| seed 1 | 0.186 | 0.457 | 0.878 | -0.0189 | -0.0055 |
| seed 2 | 0.201 | 0.565 | 0.850 | +0.0314 | +0.0212 |
| seed 3 | 0.217 | 0.569 | 0.855 | +0.0004 | -0.0086 |
| seed 4 | 0.208 | 0.549 | 0.848 | +0.0281 | -0.0032 |
| **all 600** | **0.214** | **0.542** | **0.861** | **+0.0160** | **+0.0172** |

This says something sharper than "the sample was representative". **The audit
quantities are insensitive to the sampling scheme** -- mean |offset| spans
0.186--0.223 against a full-600 value of 0.214, mean r spans 0.848--0.878
against 0.861 -- **while the downstream Delta AUC at n = 200 is
sampling-noise-dominated**, swinging from -0.019 to +0.031 across five draws of
the *same* cohort. That is the cleanest available demonstration that the
original n = 200 downstream null was an underpowered measurement and not a
finding, and it is why 11c above asks for that sentence to be narrowed.

---

## Point 15 -- the exact Delta AUC, with both denominators named

`r2_delta_auc.csv`. Same protocol and same paired image bootstrap as
`p3_downstream.py`; the absolute numbers reproduce the manuscript's to four
decimals.

| dataset | panel | clf | AUC_ref | AUC_pred | **Delta AUC (absolute)** | / AUC_ref | / (AUC_ref - 0.5) |
|---|---|---|---:|---:|---|---:|---:|
| HRF | skan4 | logreg | 0.9704 | 0.7533 | **+0.2170 [+0.1188, +0.3323]** | 0.2237 [0.121, 0.349] | 0.4614 [0.247, 0.725] |
| HRF | skan4 | gbdt | 0.9081 | 0.7281 | **+0.1800 [+0.0615, +0.3052]** | 0.1982 [0.069, 0.329] | 0.4410 [0.154, 0.739] |
| HRF | primary8 | logreg | 0.9681 | 0.7533 | **+0.2148 [+0.1151, +0.3211]** | 0.2219 | 0.4589 |
| HRF | primary8 | gbdt | 0.8956 | 0.7763 | **+0.1193 [0.0000, +0.2413]** | 0.1332 | 0.3015 |
| FIVES | skan4 | logreg | 0.7027 | 0.7055 | **-0.0027 [-0.0223, +0.0162]** | -0.0039 | -0.0135 |
| FIVES | skan4 | gbdt | 0.6807 | 0.7025 | **-0.0218 [-0.0594, +0.0157]** | -0.0320 | -0.1206 |
| FIVES | primary8 | logreg | 0.7177 | 0.7134 | **+0.0043 [-0.0158, +0.0248]** | 0.0060 | 0.0198 |
| FIVES | primary8 | gbdt | 0.7180 | 0.6876 | **+0.0305 [-0.0060, +0.0650]** | 0.0424 | 0.1397 |

**Why "a fifth" has to go.** The value 0.198 (HRF, GBDT) is Delta AUC divided by
**AUC_ref**, but the manuscript attaches it to the phrase "of the available
discrimination", which is the other denominator. Divided by the discrimination
*above chance*, the same result is **0.44, nearly a half**. The two denominators
differ here by a factor AUC_ref/(AUC_ref - 0.5) = 2.06 (logreg) / 2.22 (GBDT).
Quote the absolute difference -- **+0.2170 and +0.1800 on HRF, -0.0027 to
+0.0305 on FIVES** -- and, if a ratio is wanted anywhere, name its denominator.

---

## Point 20 -- covariate-matched controls for the topology counterfactual

Corpus: `results/c1_events_harm.parquet`, observer 1, test split,
`control_ok = 1`, `type = sever` -> **5,584 events over 118 images and 4
datasets** (the full corpus is 31,975 events / 25,649 with a valid control; the
audit's estimand uses the sever subset).

### 20a. What the corpus actually records (`r2_topology_covariates.csv`, 46 entries)

* **Event side (30 covariates, coverage 94--100 %)**: local vessel radius
  `gt_r_loc` and diameter `gt_d_loc`, calibre bin, **branch order**
  `gt_branch_order`, branch length, branch type, distance to nearest junction and
  endpoint, disc-relative zone, **local vessel density** `gt_density` (+ bin),
  local contrast (+ bin), and 17 `phi_*` features including endpoint radii, gap
  length, curvature, local endpoint/junction counts, local skeleton density and a
  far-field density ratio.
* **Control side (6 covariates, coverage 100 %)**: donor position
  `control_donor_row/col`, donor radius `control_donor_r`, changed pixel count,
  matching tier, and an addition flag.
* **Derived here for both sides**: radial distance from the optic-disc centre in
  disc radii.

**Limitation stated rather than hidden:** the corpus does **not** record the
donor's local density or local contrast, so density can enter as a
*stratification* and as an event-side exact-matching coordinate, but not as a
two-sided matching covariate. Calibre, edit length and radial distance are
available on both sides and are the ones actually matched.

### 20b. Balance before matching (`r2_topology_balance.csv`)

* **Edit burden is already exactly matched**: `n_changed == control_n_changed`
  on 5,584 of 5,584 rows, SMD = 0. The pixel-budget matching in the original
  design is exact, not approximate -- which matters, because edit burden is the
  top of the consultation's covariate-priority list.
* **Vessel calibre is not**: standardised mean difference (control minus event)
  is **+0.763 on DRIVE** and **+0.326 on HRF**, +0.030 on CHASE_DB1 and -0.008 on
  FIVES -- controls sit on systematically *wider* vessels on the two small,
  low-resolution cohorts.
* **Radial distance is essentially balanced**: |SMD| <= 0.054 everywhere.

### 20c. Nested matching levels, in the consultation's priority order

`review/cmig_plan_reply.md` section 2D fixes the priority as
**edit burden > local calibre > branch order / topological context > radial
position**, with local density *stratified* rather than hard-matched, and asks
for balance diagnostics at each level rather than a bare "we matched". Each
level adds one covariate to the previous one (`r2_topology_matched.csv`,
column `level`).

Covariates present on **both** sides (edit burden, calibre, radial position) are
pair-matched; branch order is recorded **only on the event side**, so it enters
as exact blocking, and the cells are reweighted back to the full event
distribution afterwards. That limitation is stated, not hidden.

A binning detail that mattered: local radius is near-discrete on the small
cohorts (DRIVE's 25th, 50th and 75th percentiles of `gt_r_loc` are all 1 px), so
quantile bins collapsed to a single break and "matching on calibre" did nothing
-- the first run left DRIVE's calibre SMD at +0.76 *after* matching. Radius is
therefore binned on a fixed 0.5 px grid, which makes it an effectively exact
match.

**Balance after matching (standardised mean differences, control minus event):**

| level | pair-matched | blocked | DRIVE radius SMD | worst \|SMD\| any covariate/cohort |
|---|---|---|---:|---:|
| unmatched (shipped) | edit burden only | -- | **+0.763** | 0.763 |
| `L0_edit_burden` | edit burden | -- | +0.763 | 0.763 |
| `L1_plus_calibre` | + calibre | -- | **-0.016** | 0.063 |
| `L2_plus_branch_order` | + calibre | branch order | -0.016 | 0.063 |
| `L3_plus_radial` | + calibre + radial | branch order | +0.000 | 0.027 |
| `L4_density_stratified` | + calibre + radial | branch order, density bin | +0.000 | 0.027 |
| `nn_within_image` | weighted NN, all three | -- | 0.000 | 0.013 |

**Harm at each level (primary skan panel):**

| level | mean n events/cell | match rate | mean h_net | max h_net |
|---|---:|---:|---:|---:|
| `L0_edit_burden` | 1396 | 1.00 | 0.00455 | 0.0198 |
| `L1_plus_calibre` | 585 | 0.43 | 0.00542 | 0.0297 |
| `L2_plus_branch_order` | 585 | 0.43 | 0.00511 | 0.0254 |
| `L3_plus_radial` | 362 | 0.26 | 0.00560 | 0.0264 |
| `L4_density_stratified` | 362 | 0.26 | 0.00556 | 0.0286 |
| `nn_within_image` (caliper 0.25 SD, priority weights 3/2/1) | 90 | 0.08 | 0.00677 | 0.0478 |

**The estimate is flat across every level.** The mean topology-specific harm
moves from 0.0046 to 0.0068 sigma as matching is tightened from the shipped
design to the strictest one; the maximum single cell moves from 0.0198 to
0.0478 sigma and is **CHASE_DB1 `tortuosity_skan` at every level**. Per-cell
examples: HRF `total_length_skan` 0.0030 -> 0.0028 across L0 -> NN; FIVES
`FD_skan` 0.0024 -> 0.0028; DRIVE `total_length_skan` 0.0063 -> 0.0056.

**Density stays exactly 0 in every design**, which is the construction check --
density is a function of the foreground pixel count and the control matches that
count identically.

The NN design's low match rate on DRIVE (6.7 %) and HRF (7.3 %) is the price of
the 0.25 SD caliper on small images; it is the *strictest* design, with the CEM
levels (26--43 % retained and reweighted to the full event distribution) as the
ones whose estimand stays closest to "all the events we generated".

### 20d. Stratified sensitivity (`r2_topology_strata.csv`)

Tertiles of calibre, radial distance, local density and edit length, 360 rows.
The largest topology-specific harm anywhere is **CHASE_DB1 `tortuosity_skan` in
the top local-density tertile: h_net = 0.0544 sigma [0.0177, 0.1079]**
(n = 274 events). The next four are all CHASE_DB1 tortuosity as well
(0.0464, 0.0407, 0.0352, 0.0224); the largest non-CHASE stratum value is FIVES
tortuosity at 0.0170.

### 20e. Does the claim survive? (`r2_topology_claims.csv`)

Against the primary-panel mean |offset| (0.951 sigma) and mean residual SD
(1.139 sigma):

| design | max \|h_net\| | ratio to mean \|offset\| | ratio to mean residual SD | orders of magnitude |
|---|---:|---:|---:|---:|
| `L0_edit_burden` (shipped) | 0.0198 | 47.9x | 57.4x | 1.68 / 1.76 |
| `L1_plus_calibre` | 0.0297 | 32.0x | 38.3x | 1.51 / 1.58 |
| `L2_plus_branch_order` | 0.0254 | 37.5x | 44.9x | 1.57 / 1.65 |
| `L3_plus_radial` | 0.0264 | 36.1x | 43.2x | 1.56 / 1.64 |
| `L4_density_stratified` | 0.0286 | 33.3x | 39.9x | 1.52 / 1.60 |
| `nn_within_image` | 0.0478 | 19.9x | 23.8x | 1.30 / 1.38 |
| **worst single stratum** | **0.0544** | **17.5x** | **20.9x** | **1.24 / 1.32** |

**Yes, the claim survives every design, all 28 comparisons.** But the honest
wording tightens: the margin is 1.5--1.7 orders of magnitude under full
covariate matching and drops to **1.2--1.3 orders under the strictest
nearest-neighbour design and in the worst stratum**. The paper should write
*"at least one order of magnitude below the offset and residual axes, and still
a factor of 17 to 20 below them under the strictest matching and in the least
favourable stratum"* rather than "one to two orders of magnitude", and should
point at the level table and the stratified table.

### 20f. The conservative reconnector, self-contained

`results/pivot/r2/reconnector_description.md` -- a self-contained description of
the arm the audit uses (`runs/rigr/uniform/...`, uniform-event-cost RiGR at
lambda = 1.0, eta = 0.1, mu = 2.0, learned orientation), covering candidate
generation and its four admissibility gates, the backbone-decoupled micro head,
the corridor-restricted lifted A*, the calibrated 13-feature pair scorer, and the
prune-then-exact-matching selection, with a provenance table. **No TODO and no
companion-paper dependency remains.**

---

## Second segmenter family -- does the audit replicate outside the U-Net lineage?

`r2_family2.py` repeats the whole C1 audit for the second families on exactly
the frozen conventions of family 1: same primary panel, same frozen per-cohort
sigma, **the same common scale `s_k` loaded from `r2_scale_sensitivity.csv`
rather than recomputed**, the same median-centred frame for alpha, the same
Deming lambda = 1 fit, the same 2000-draw paired image bootstrap and the same
downstream protocol as `p3_downstream.py`.

Loading `s_k` instead of recomputing it matters: family 1 was run against
`bio_master.csv` (FIVES training split = 200 images) while `bio_master_full.csv`
now has 600. Recomputing would have shifted `density_skan` 0.014138 -> 0.013819
and `total_length_skan` 2.06425 -> 2.13181, putting the two families on
different denominators. The script warns loudly if the frozen table is missing.

### Coverage: 12 prediction sets

| family | what it is | datasets x seeds | in-domain? |
|---|---|---|---|
| `lwnet` | public 68 k-parameter W-Net checkpoint | 4 x 1 | **DRIVE is this checkpoint's own training set**; CHASE_DB1 / HRF / FIVES are genuinely zero-shot (`zero_shot` flag per row) |
| `segformer_b0` | 3.71 M-parameter transformer trained here on the same splits, resolution and early-stopping rule as the U-Net | **4 x 2** | yes, trained per dataset -- never zero-shot |

SegFormer-B0 best Dice (seed 0 / seed 1) against the U-Net: DRIVE 0.796/0.796
vs 0.826, CHASE_DB1 0.809/0.807 vs 0.817, HRF 0.809/0.808 vs 0.814, FIVES
0.903/0.904 vs 0.902. A competent, slightly weaker second architecture -- which
is what makes it a fair control rather than a straw man. The two seeds are
genuinely distinct runs (different checkpoints and different predictions; the
identical DRIVE `best_dice` to four decimals is a coincidence).

### Mask-failure accounting under the pre-declared rules (`family2_mask_quality.csv`)

Rules locked in DECISIONS.md 2026-09-18 11:03 **before any family-2 audit number
was looked at**: `fg_px == 0` is a segmentation failure (excluded from
calibration, counted); `pred_fg_frac < 0.02` is degenerate (kept in the main
analysis, excluded in a sensitivity analysis); both reported.

| family | seed | dataset | n | empty | degenerate | failed | failures per class |
|---|---:|---|---:|---:|---:|---:|---|
| lwnet | 0 | DRIVE (in-domain) / CHASE_DB1 / HRF | 20 / 8 / 30 | 0 | 0 | 0 | -- |
| lwnet | 0 | **FIVES** | 200 | **2** | **15** | **17 (8.5 %)** | **Glaucoma 13/50, AMD 2/50, DR 2/50, Normal 0/50** |
| segformer_b0 | 0, 1 | DRIVE / CHASE_DB1 / HRF | 20 / 8 / 30 | 0 | 0 | 0 | -- |
| segformer_b0 | 0 | **FIVES** | 200 | 0 | **2** | 2 (1.0 %) | **Glaucoma 2/50**, all others 0/50 |
| segformer_b0 | 1 | **FIVES** | 200 | 0 | **3** | 3 (1.5 %) | **Glaucoma 3/50**, all others 0/50 |

The LWNet counts reproduce the GPU job's shipped `lwnet_mask_quality.csv`
**exactly** (`crosscheck` column).

**This is the substantive new finding, and it is stronger than expected.**
Selective segmentation failure on glaucoma is **not** peculiar to the
out-of-domain public checkpoint. An in-domain transformer, trained on FIVES'
own training split, also fails only on glaucoma images -- 2 and 3 of 50 across
its two seeds, and zero of 150 on the other three classes. The rate differs by
roughly an order of magnitude (8.5 % against 1.0-1.5 %) but the *class
selectivity is identical and total*. Whatever makes a glaucomatous fundus hard
to segment survives in-domain training and survives a change of architecture.

### The audit, side by side (`family2_audit.csv`, primary skan panel, `analysis_set = main`)

Mean over the four primary biomarkers, sigma units on the frozen per-cohort scale:

| dataset | family (seed) | mean abs(mu) | mean residual SD | mean CCC | mean r |
|---|---|---:|---:|---:|---:|
| **HRF** | U-Net (family 1) | 1.540 | 0.911 | 0.302 | 0.487 |
| | LWNet | 3.015 | 1.486 | 0.131 | 0.343 |
| | **SegFormer-B0 (0)** | **1.926** | 1.618 | 0.285 | 0.414 |
| | **SegFormer-B0 (1)** | **1.870** | 1.555 | 0.254 | 0.386 |
| **FIVES** | U-Net (family 1) | 0.248 | 0.612 | 0.785 | 0.824 |
| | LWNet | 0.694 | 1.757 | 0.585 | 0.733 |
| | **SegFormer-B0 (0)** | **0.255** | 0.461 | 0.869 | 0.900 |
| | **SegFormer-B0 (1)** | **0.278** | 0.537 | 0.844 | 0.877 |
| DRIVE | U-Net (family 1) | 0.927 | 0.692 | 0.175 | 0.299 |
| | LWNet (in-domain) | 1.074 | 0.755 | 0.185 | 0.346 |
| | SegFormer-B0 (0 / 1) | 1.629 / 1.614 | 0.721 / 0.702 | 0.083 / 0.080 | 0.245 / 0.230 |
| CHASE_DB1 | U-Net (family 1) | 1.088 | 1.153 | 0.510 | 0.728 |
| | LWNet | 1.732 | 1.116 | 0.457 | 0.607 |
| | SegFormer-B0 (0 / 1) | 1.413 / 1.453 | 0.445 / 0.451 | 0.398 / 0.390 | 0.784 / 0.792 |

**Seed stability is excellent**: across all 32 SegFormer (dataset x biomarker)
cells the median absolute seed-0-minus-seed-1 difference is 0.057 sigma for mu,
0.028 for beta, 0.058 for the residual SD, 0.013 for CCC and 0.018 for r. None
of the conclusions below turns on which seed is read.

### The reference-gap Delta AUC (`family2_delta_auc.csv`)

Same protocol as family 1. **FIVES is test-only 200 images for every family** --
no family has out-of-fold predictions on the FIVES training split -- so the
apples-to-apples family-1 comparator is its `test200` block, not the full-800
figure. Every row records this in `split_coverage`.

skan4 panel, logistic regression:

| dataset | family (seed) | AUC_ref | AUC_pred | Delta AUC [CI] |
|---|---|---:|---:|---|
| **HRF** | U-Net (family 1) | 0.9704 | 0.7533 | **+0.2170** [+0.1188, +0.3323] |
| | LWNet | 0.9650 | 0.7067 | **+0.2583** [+0.1251, +0.3921] |
| | **SegFormer-B0 (0)** | 0.9650 | 0.7667 | **+0.1983** [+0.0614, +0.3480] |
| | **SegFormer-B0 (1)** | 0.9650 | 0.7550 | **+0.2100** [+0.0744, +0.3572] |
| **FIVES** | U-Net (family 1, test 200) | 0.7135 | 0.7135 | 0.0000 [-0.0280, +0.0278] |
| | LWNet | 0.7087 | 0.7161 | -0.0074 [-0.0473, +0.0295] |
| | **SegFormer-B0 (0)** | 0.7135 | 0.7109 | +0.0026 [-0.0229, +0.0270] |
| | **SegFormer-B0 (1)** | 0.7135 | 0.7062 | +0.0073 [-0.0184, +0.0318] |

**Every family and every seed reproduces the contrast: a large, zero-excluding
gap on HRF (+0.198 to +0.258) and nothing on FIVES (-0.007 to +0.007).** Three
architectures -- a U-Net, a 68 k-parameter public W-Net applied zero-shot, and a
3.7 M-parameter transformer trained in-domain over two seeds -- agree on both
halves of the contrast.

### How much of a FIVES gap is segmentation failure? (`family2_failure_attribution.csv`)

Negative Delta AUC means the **predicted** features out-perform the reference
features -- the signature to watch for, because if mask collapse tracks a
disease class then the collapse itself becomes a diagnostic feature.

| family (seed) | panel / clf | keeping all 200 | drop empty (`main`) | drop all degenerate | share removed |
|---|---|---:|---:|---:|---|
| LWNet | primary8 / gbdt | **-0.0648** [-0.1152, -0.0153] | -0.0377 | -0.0331 | **42 % from 2 images** |
| LWNet | skan4 / gbdt | **-0.0604** [-0.1140, -0.0058] | -0.0446 | -0.0596 | 26 % from 2 images |
| **SegFormer (1)** | primary8 / gbdt | **-0.0462** [-0.0917, -0.0003] | -0.0462 (no empties) | **-0.0335** [-0.0786, +0.0140] | **28 % from 3 images** |
| SegFormer (0) | primary8 / gbdt | -0.0165 | -0.0165 | -0.0006 | ratio undefined (base < 0.02) |
| any family | HRF, all cells | -- | **0.0000 exactly** | 0.0000 | zero failures |

Three things the discussion should say:

1. **Keeping the failed masks makes the predicted features significantly
   out-perform the reference features on FIVES** -- LWNet -0.0648 and -0.0604,
   SegFormer seed 1 -0.0462, all three CIs excluding zero. That is not
   measurement fidelity; it is selective failure on glaucoma acting as a free
   disease label.
2. **A handful of images does it.** Two of 200 account for 42 % of LWNet's
   effect; three of 200 account for 28 % of SegFormer's, and removing them moves
   its interval from [-0.0917, -0.0003] to [-0.0786, +0.0140] -- the
   significance disappears. **One per cent of a cohort can manufacture a
   statistically significant downstream result.**
3. **HRF, with zero mask failures, gives exactly 0.0000 attribution in every
   cell** -- a clean construction check that the machinery measures what it
   claims to.

This is the direct evidence for the narrowing requested at review points 14/15:
a downstream AUC computed on automatically segmented biomarkers can move for
reasons that have nothing to do with the vasculature, so "downstream
discriminative performance" must not be read as clinical or biological validity.

### Does each family-1 conclusion replicate? (`family2_replication.csv`)

`main` analysis set. **The four ordering conclusions -- the ones the paper's
narrative actually rests on -- replicate in every family and every seed:**

| conclusion | U-Net | LWNet | SegFormer (0) | SegFormer (1) |
|---|---|---|---|---|
| offset ordering HRF > FIVES | 6.2x | **4.35x** yes | **7.55x** yes | **6.74x** yes |
| fidelity ordering FIVES > HRF | r 0.824/0.487 | **0.733/0.343** yes | **0.900/0.414** yes | **0.877/0.386** yes |
| residual-scatter ordering HRF > FIVES | 1.5x | 0.85x **no** (see below) | **3.51x** yes | **2.90x** yes |
| reference gap large HRF, ~0 FIVES | +0.217 / -0.003 | **+0.258 / -0.007** yes | **+0.198 / +0.003** yes | **+0.210 / +0.007** yes |

The single ordering failure is LWNet's residual scatter, and it is **entirely
the 15 degenerate FIVES masks**: in the sensitivity set the two cohorts are tied
(1.486 against 1.483). SegFormer, which has almost no degenerate masks,
replicates it cleanly at 2.9-3.5x.

**A criterion that should NOT be used: the raw "how many slopes are below 1"
count.** On a low-fidelity cohort the Deming slope is *unidentified* --
SegFormer HRF FD is beta = 2.40 with a bootstrap CI of [-16.1, 21.0] because
r = 0.20, and HRF total length is 1.06 with CI [-2.3, 3.2]. The replication
table now reports `n_identified` (bootstrap CI width <= 2) alongside the raw
count and returns an **undecidable** verdict where fewer than half the columns
are identified. Among *identified* slopes the picture is consistent: SegFormer
HRF tortuosity is 0.28 [0.01, 0.65] and 0.29 [0.05, 0.63], firmly below 1 and
matching the U-Net's 0.37; DRIVE is 4/4 below 1 in both trained families
(SegFormer 0.70 / 0.06 / 0.73 / 0.26 against the U-Net's 0.77 / 0.03 / 0.72 /
0.45 -- a near column-by-column match). **Read attenuation off the identified
slopes, never off the count.**

Overall SegFormer-B0 replicates 8 of 9 checkable conclusions on each seed, and
the one difference (the FIVES slope count) is SegFormer being *better* calibrated
than the U-Net on FIVES (beta 1.27 / 0.69 / 1.05 / 0.98 against 1.58 / 0.58 /
1.03 / 0.99), not a contradiction.

### What this licenses the paper to say

The measurement-error structure the audit describes is **not a U-Net artefact**.
It reproduces in a public 68 k-parameter zero-shot checkpoint and in a 3.7 M
transformer trained in-domain over two seeds, on the frozen scale, with the
offset ordering, the fidelity ordering, the residual-scatter ordering and the
downstream reference gap all intact. The one genuinely family-specific behaviour
is the *direction* of the FIVES slope under severe domain shift (LWNet stretches,
beta 1.3-3.1; both in-domain families sit near 1), and that should be described
as a domain-shift property rather than averaged into a single claim.

## Caveats the writing pass must carry

1. **The literature table's DOI column is nearly empty.** Only 5 of 97 rows in
   `pixel_literature.csv` carry a DOI, because the run that built it recorded
   a DOI only where it actually saw the string on a fetched page. Every row does
   carry `source_url` (mostly arXiv HTML). **The DOIs must be resolved before
   these are cited**, and the `notes` column must be read: several rows are
   third-party transcriptions of a method's numbers rather than the method's own
   paper, and the `f1_reported_as` column records whether a paper wrote "Dice",
   "DSC" or "F1".
2. **`mIoU` trap in the literature table**: FSG-Net's "mIoU" is class-averaged
   over vessel + background, not vessel-only Jaccard, and is about 0.15 higher.
   Those cells were deliberately left empty; do not fill them in.
3. **The FIVES dataset paper (Jin et al., Sci Data 9:475, 2022) reports no model
   benchmark** -- only annotation agreement (intra 0.9679, inter same-level
   0.9241, inter different-level 0.9608). It is in the table with empty metric
   cells. Those agreement figures are the right *human ceiling* context for a
   FIVES model Dice of 0.90, and should be used that way rather than as a
   baseline.
4. **HRF and FIVES pixel metrics are at `resize_longest = 1536`, not native.**
   Quote the working resolution wherever the HRF Dice appears.
5. **The `pooled_naive` rows in `r2_scale_sensitivity.csv` are the rejected
   definition**, retained only to show why it is rejected. Do not quote them as
   "the common scale".
6. **The common scale is a sensitivity, not a unit.** It must not be called a
   "natural unit" or a "population SD" anywhere.
7. **`r2_topology_matched.csv` contains six designs.** The manuscript should
   quote the level table, not a single design, because the point of the exercise
   is that the estimate is flat across levels.
8. Everything here is segmentation seed 0 for the audit tables and seeds 0-2 for
   the pixel metrics; the second segmenter family (GPU job) and the extra
   ReliSeg seeds are not in scope for this report. The common scale `s_k` is
   fixed and the same `s_k` applies to the second family when it lands.
