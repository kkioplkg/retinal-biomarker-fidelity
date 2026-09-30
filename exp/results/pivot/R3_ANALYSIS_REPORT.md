# R3 -- analyses for the third CMIG review (points 3, 4, 5)

Review: `review/user_cmig_review3_20260929.md`. Plan adjustments: DECISIONS 2026-09-29 22:06
(`review/cmig_r3_plan_reply.md`). CPU only. `D:/Anaconda/envs/medical1/python.exe` with
OMP/MKL/OPENBLAS_NUM_THREADS=1. Every output is new, under `results/pivot/r3/` and `figs/pivot/r3_*`.
**No locked output was modified.** Wherever a locked number is re-derived, the script checks that it
reproduces the locked value:

- Deming at λ=1 and λ=4 matches `r2_calibration.csv` to 3.6e-15.
- σ_b recomputed from its source images matches the frozen table exactly (ratio 1.0).
- The common scale s_k matches `r2_common_scale_formula.txt`.
- skan4 reference gaps reproduce 0.2170 / 0.1800 / 0.0132 / 0.0009 and every family-2 gap.
- The skan4 macro |e| repair change reproduces `macro_primary4`.

```
python -m src.pivot.r3_deming_lambda      # ~30 s
python -m src.pivot.r3_sigma_joint_boot   # ~20 s
python -m src.pivot.r3_panel3             # ~25 min (downstream refits)
```

---

## 3. Deming error-variance ratio λ (`src/pivot/r3_deming_lambda.py`)

**Model.** B_pred = α + β·B_ref. The error-variance ratio is λ = var(err_pred)/var(err_ref), and

β = (S_yy − λS_xx + √((S_yy − λS_xx)² + 4λS_xy²)) / (2S_xy).

Everything is on the σ_b axis, centred on the cohort reference median, with y = pred and x = ref. Two limits are added as brackets: λ→∞ is OLS of pred on ref (`ols_y_on_x`), and λ→0 is the inverse regression (`ols_x_on_y_inv`). The CI comes from a 2000-draw paired image bootstrap. The indices are the `r2_calibration.py` stream, and the same images are used at every λ.

**Inter-observer λ.** Only DRIVE (20 test images) and CHASE_DB1 (28 images) have a second observer (`results/fig4c_observer.csv`). Define d = (obs2 − obs1)/σ_b and e = (pred − ref)/σ_b. Two estimates of λ are computed:

- `λ_mom` (preferred): s²_ref = var(d)/2 and s²_pred = var(e) − s²_ref, so λ = s²_pred / s²_ref. It is floored at 0.05 when var(e) ≤ s²_ref.
- `λ_raw` (the reviewer's literal wording): λ = var(e)/var(d).

Each estimate comes with a 2000-draw bootstrap range. β is then refitted at the point estimate and at both ends of the range (`r3_deming_lambda_obs.csv`). HRF and FIVES have one observer, so their own-data λ is **NA**. For them an **extrapolated** range transports s²_ref, in σ_b units, from DRIVE or from CHASE_DB1. This extrapolation carries no confirmatory weight. On FIVES density and length the transported reference error exceeds FIVES' entire residual variance, so the transport is not credible there.

Own-data λ_mom:

| Cohort | Density | Length | FD | Tortuosity |
|---|---|---|---|---|
| DRIVE | 0.61 | 0.27 | 0.54 | 1.13 |
| CHASE_DB1 | ≤0.05 | 0.26 | ≤0.05 | 0.99 |

The bootstrap ranges are very wide (up to 0.05–62). The observers therefore place λ at or below 1 on the small cohorts.

**λ-stability** was defined in advance (DECISIONS 22:06): the λ=1 verdict (the 95 % interval of β excludes 1, or not) must be unchanged for every λ ∈ {0.25, 0.5, 1, 2, 4}. Main-text table (`r3_deming_stability.csv`, `main_text=True`):

| Cell | β(λ=1) [95 % CI] | β range over λ∈[0.25,4] | λ where the verdict flips | λ-stable | Inter-observer / extrapolated λ settings |
|---|---|---|---|---|---|
| HRF total length | 0.42 [−0.08, 0.95] | 0.26–1.49 | 0.25, 0.5 (CI covers 1) | **no** | β 0.23–0.46; the verdict flips at one end of a transported range (λ=0.885 → [−0.09, 1.07]) |
| HRF tortuosity | 0.37 [0.12, 0.69] | 0.34–0.47 | none | **yes** | β 0.33–0.62, always excludes 1 |
| FIVES FD | 1.58 [1.25, 1.93] | 1.48–1.64 | none | **yes** | β 1.42–1.67, always excludes 1; OLS limit 1.39 [1.18, 1.63]; all 800 images 1.48 [1.31, 1.66], also stable |

Other cells (supplement):

- **HRF, "all four slopes < 1".** This holds for λ ≥ 1 only. At λ=0.5, density has β = 0.97 but FD has 1.02. At λ=0.25 the range is 0.47–1.75.
- **HRF density and FD.** Density is λ-unstable only at λ=4, where its CI excludes 1. FD becomes "excludes 1" at λ ≥ 2.
- **FIVES density and length.** Length is stable. Density's CI just excludes 1 at λ=0.25 only.
- **FIVES tortuosity.** Its bold "excludes 1" verdict (0.58 [0.20, 1.00]) is **not** λ-stable: it flips at λ ≤ 0.5.
- **DRIVE.** No cell is stable, as expected at n=20.
- **CHASE_DB1.** All cells are stable.

**Headline.** Two conclusions are λ-robust: HRF tortuosity attenuation and the FIVES FD stretch. FIVES FD even survives the OLS/inverse-OLS brackets and the 800-image block. The HRF total-length exclusion of 1 holds only for λ ≥ 1, i.e. only if the prediction error is at least as large as the reference-mask error. The HRF residual variance is 1.66 σ² against a transported s²_ref of 0.21–0.41, which makes λ ≥ 1 plausible (extrapolated λ_mom 3.0–6.8), but it is not established from HRF's own data. "All four HRF slope estimates below 1" is likewise conditional on λ ≥ 1.

Figure: `figs/pivot/r3_deming_lambda.{png,pdf}` shows β against λ for each cohort × biomarker. Bands are 95 % CIs. Green marks the own inter-observer λ range, orange the transported range. FIVES also shows the 800-image curve.

---

## 4. Joint bootstrap of test images and σ_b (`src/pivot/r3_sigma_joint_boot.py`)

**Resampling units and scales.**

- **σ_b** is re-estimated (1.4826·MAD) in every draw from resampled training reference masks. These are the exact images behind the frozen table: DRIVE 20 and CHASE_DB1 20 training images, HRF 15 (`gateA_biomarkers_gt.csv`, obs1), and FIVES 120 (`fives_native_train_biomarkers.csv`).
- **The common scale s_k** is re-estimated from all four cohorts' training masks.
- **Independence unit.** This is the image, except in **CHASE_DB1, whose images are left/right eye pairs of 14 children**. There, both the test resample (8 images = 4 children) and the σ_b resample (20 = 10 children) draw whole children. The published r2 intervals resampled CHASE images, so the plan note that "only Fundus-AVSeg has eye pairs" is incorrect for the audit set. Clustering changes the CHASE conditional widths by a factor of 0.93–1.55. DRIVE, HRF and FIVES have one image per subject_id.
- **Draws and intervals.** 2000 draws, percentile and BCa. BCa acceleration comes from a joint leave-one-unit-out jackknife over test and training units.
- **Zero MAD.** 0.05 % of HRF draws have MAD = 0 (more than half the draws are one image) and are dropped.

**Relative half-width of σ_b itself** (`r3_sigma_boot_scale.csv`):

| Cohort | Training units | Relative half-width |
|---|---|---|
| DRIVE | 20 images | 0.63–0.68 |
| CHASE_DB1 | 10 children | 0.57–1.41 |
| HRF | 15 images | 0.76–0.97 |
| FIVES | 120 images | 0.20–0.22 |
| common s_k | all four cohorts | 0.29–0.43 |

On the small cohorts the scale, not the test sample, dominates the uncertainty.

**Paper-facing file: `r3_main_table_joint.csv`.** It holds every row and column of `r2_main_table.csv` (all 8 columns, the same keys and point values), plus:

- `{mu,alpha,resid_sd}_{joint,joint_bca,cond_unit}_{lo,hi}`
- `sigma`, `sigma_lo`, `sigma_hi`, `sigma_n_units`, `resample_unit`

For DRIVE, HRF and FIVES the test indices are the r2 stream, so `cond_unit` equals the published interval exactly. β, CCC and r are scale-free and do not change. The supplement comparison of conditional against joint widths is in `r3_sigma_joint_width_compare.csv`.

**Joint/published width ratios (skan cells):**

| Cohort | μ | α | Residual SD | Notes |
|---|---|---|---|---|
| FIVES | 1.03–1.73 | 1.08–1.71 | 1.13–1.98 | |
| DRIVE | 1.6–11 | 1.5–14 | 1.9–7.3 | BCa 1.3–3.8 |
| CHASE_DB1 | 1.5–4.6 | | | |
| HRF | 4–13 | | | |

On the small cohorts the joint interval is strongly right-skewed: the lower bound moves little and the upper tail balloons. Example: HRF length μ −1.79 is [−2.22, −1.33] published, [−6.06, −0.71] joint, and [−7.43, −0.80] BCa.

**No μ, α or residual-SD interval changes its exclude-0 verdict** under the joint percentile or BCa interval. All 48 skan cells in the compare file were checked.

**Qualitative claims** (`r3_sigma_joint_claims.csv`; skan4 unless stated):

| Claim | Image-only (published type) | Joint (images + σ_b) | Verdict |
|---|---|---|---|
| HRF > FIVES mean \|offset\| | ratio [4.04, 8.93], P = 1.00 | ratio [3.51, 19.1], P = 1.00; common scale [2.55, 5.65], P = 1.00; skan3 [1.90, 15.0], P = 1.00 | **unchanged** |
| HRF > FIVES mean residual SD | P = 0.994 | P = 0.985 (ratio [1.10, 7.03]); common scale P = 0.92 (ratio [0.86, 2.20]); skan3 P = 0.96 | holds on own σ_b; its interval already covered 1 under the common scale before σ uncertainty (P = 0.94) |
| "Of order one σ_b" (four-cohort grand means within [1/3, 3]) | \|μ\| [0.84, 1.11], SD [0.84, 1.28] | \|μ\| [0.81, 2.02], SD [0.88, 2.14]; P(in band) = 0.996 | **unchanged**; per cohort, FIVES \|μ\| 0.25 [0.18, 0.36] was already below 1/3 |
| Topology margin "17–20×" (worst stratum; within-image NN) | worst stratum 17.5 [15.4, 20.4]; NN 19.9 [17.5, 23.2] | worst stratum [8.7, 40.5], P(≥10×) = 0.94; NN [9.9, 46.7], P(≥10×) = 0.97; edit-burden design [23, 94] | point unchanged; the lower end drops to about 9×, so it should read "about an order of magnitude or more" rather than a tight "17–20×". Event-level sampling of h_net is not resampled (its own CIs are in `r2_topology_matched.csv`) |

Figure: `figs/pivot/r3_sigma_joint_boot.{png,pdf}` is a per-cell forest plot of μ and residual SD, with the image-only and joint intervals shown side by side, plus panel means.

---

## 5. Three-biomarker primary panel (`src/pivot/r3_panel3.py`)

**Status.** The three-biomarker summaries are an *added post-hoc reporting hierarchy* (DECISIONS 22:06).

- skan4 stays locked for every downstream number, and skan3 downstream is a sensitivity analysis.
- Tortuosity numbers are all still reported per biomarker. Tortuosity is only excluded from the three-biomarker **means**.
- The external label-only cohorts (APTOS-2019, IDRiD, Messidor-2) were **not run**. They have no reference masks, so there is no reference gap, and `e3_external.py` hard-codes the skan4 featureset (no panel argument). E2/E4 were not changed: they report per-biomarker Δr with tortuosity already a safety endpoint, and use no fidelity panel mean. Their ΔAUC column (range over seeds, skan4 classifier) was not recomputed.

### Audit panel means (`r3_panel3_audit.csv`; Table `tab:audit` and §R1 text)

| Cohort | Mean \|μ\| 4→3 | Residual SD 4→3 | r 4→3 | ρ 4→3 | Topology mean \|ΔB\| 4→3 | Topology RMS 4→3 | Observer 4→3 | View 4→3 |
|---|---|---|---|---|---|---|---|---|
| DRIVE | 0.927→0.783 | 1.233→0.860 | 0.299→0.362 | 0.394→0.346 | 0.019→0.020 | 0.029→0.027 | 0.880→0.785 | 0.522→0.549 |
| CHASE_DB1 | 1.088→1.079 | 1.125→0.542 | 0.728→0.930 | 0.839→0.929 | 0.036→0.024 | 0.051→0.031 | 1.471→1.411 | 0.487→0.396 |
| HRF | 1.540→1.282 | 1.479→1.039 | 0.487→0.420 | 0.521→0.414 | 0.027→0.017 | 0.036→0.018 | -- | 0.518→0.577 |
| FIVES | 0.248→0.299 | 0.717→0.508 | 0.824→0.958 | 0.898→0.969 | 0.041→0.017 | 0.109→0.028 | -- | 0.268→0.223 |
| Four-cohort grand mean | 0.951→0.861 | 1.139→0.737 | | | | | | |

### Reference gap, skan4 (locked) against skan3 (sensitivity) (`r3_panel3_gap.csv`)

Each row gives the gap on both panels and the paired bootstrap of gap(skan3) − gap(skan4) on the same resampled images (1000 draws, locked protocol). The classifier columns are kept in the locked order (FD, [tortuosity], density, length). Logistic regression is order-invariant, but HistGradientBoosting breaks split ties by feature index: a first run with the columns reordered gave HRF GBDT +0.140 instead of the locked +0.180. That is a reproducibility caveat for every GBDT number.

| Cohort / block | Classifier | skan4 (locked) | skan3 | Δ (3−4) [95 %] |
|---|---|---|---|---|
| HRF (n=45) | logistic | +0.2170 [0.119, 0.332] | +0.2022 [0.075, 0.337] | −0.015 [−0.093, +0.070] |
| HRF | GBDT | +0.1800 [0.062, 0.305] | +0.1126 [−0.060, 0.283] | −0.067 [−0.189, +0.049] |
| FIVES all 800 | logistic | +0.0132 [−0.002, 0.028] | +0.0100 [−0.003, 0.022] | −0.003 [−0.013, +0.007] |
| FIVES all 800 | GBDT | +0.0009 [−0.021, 0.024] | +0.0138 [−0.009, 0.035] | +0.013 [−0.004, +0.030] |
| FIVES test 200 | logistic / GBDT | 0.0000 / −0.0319 | +0.0061 / −0.0530 [−0.099, −0.004] | +0.006 / −0.021 |
| FIVES train 600 | logistic / GBDT | +0.0160 / +0.0172 | +0.0016 / +0.0325 [0.004, 0.059] | **−0.015 [−0.026, −0.002]** / +0.015 |
| FIVES p2-400 | logistic / GBDT | −0.0027 / −0.0218 | +0.0077 / −0.0168 | +0.010 / +0.005 |

Second-segmenter families (logistic regression, "main" analysis set):

| Family | Cohort | skan4 | skan3 |
|---|---|---|---|
| W-Net | HRF | +0.258 | +0.185 [0.038, 0.342] |
| SegFormer seed 0 | HRF | +0.198 | +0.087 [−0.083, 0.270] |
| SegFormer seed 1 | HRF | +0.210 | +0.088 [−0.079, 0.278] |
| W-Net, all images incl. failed masks | FIVES | −0.060 [−0.114, −0.006] (GBDT) | −0.078 [−0.139, −0.018] |

For the two SegFormer seeds on HRF, the Δ of −0.11 and −0.12 has an interval reaching +0.010 and +0.003.

**Headline.**

- The primary HRF gap (logistic) is essentially unchanged: +0.202 against +0.217, still excluding 0.
- The FIVES 800 gap stays a few AUC points.
- 1 of 7 family-1 paired differences excludes 0 (FIVES train600 logistic, −0.015).
- However, on skan3 the **HRF GBDT gap and both SegFormer HRF gaps no longer exclude 0**. Tortuosity carries part of the HRF gap for these segmenters. "The reference-gap contrast replicates in every family and seed" therefore holds on skan4 only; on skan3 it holds for U-Net and W-Net (logistic).

### Scale, topology and other panel-derived numbers (`r3_panel3_textranges.csv`)

See the next section.

---

## Paper numbers that change if the primary panel becomes skan3 (old → new)

Downstream numbers stay on the locked skan4 panel; the skan3 values are only for the added reporting hierarchy.

**§R1 text, Fig. 2 caption, `tab:audit`**

- Mean offset 0.248 / 0.927 / 1.088 / 1.540 → **0.299 / 0.783 / 1.079 / 1.282** (FIVES, DRIVE, CHASE_DB1, HRF)
- Residual SD 0.717 / 1.233 / 1.125 / 1.479 → **0.508 / 0.860 / 0.542 / 1.039**
- Mean Pearson r 0.299 / 0.487 / 0.728 / 0.824 (DRIVE, HRF, CHASE_DB1, FIVES) → **0.362 / 0.420 / 0.930 / 0.958**. The ordering changes: HRF becomes the lowest-fidelity cohort, and "almost threefold" becomes about 2.6×.
- Mean ρ 0.394 / 0.839 / 0.521 / 0.898 → 0.346 / 0.929 / 0.414 / 0.969 (DRIVE, CHASE_DB1, HRF, FIVES)
- Topology panel-mean \|ΔB\| 0.019 / 0.036 / 0.027 / 0.041 → 0.020 / 0.024 / 0.017 / 0.017; RMS 0.029 / 0.051 / 0.036 / 0.109 → 0.027 / 0.031 / 0.018 / 0.028
- Topology repair range "0.006–0.115 σ (0.006–0.351 RMS)" → **0.006–0.032 (0.006–0.059)**
- Inter-observer 0.880 / 1.471 → **0.785 / 1.411**
- View spread "0.268–0.523" (the table actually says 0.522) → **0.223–0.577**
- "Worst offsets: tortuosity HRF −2.313σ" should be restated as "worst primary offset: HRF length −1.789σ", with the tortuosity value kept as the secondary endpoint.

**§R1 / Supplement, topology matching (`tab:topomatch`)**

- Denominators 0.951 / 1.139 → **0.861 / 0.737**
- Mean h_net 0.0046–0.0068 → 0.0027–0.0032; max h_net 0.020–0.048 → 0.0063–0.0076. The largest cell changes from CHASE tortuosity to DRIVE/CHASE length.
- Ratios vs offset 47.9 / 32.0 / 37.5 / 36.1 / 33.3 / 19.9 → **135.7 / 117.7 / 117.6 / 112.8 / 114.4 / 114.4**; worst stratum 17.5 → **109.6**
- Ratios vs residual SD 57.4 … 23.8 → 116.2 … 98.0; worst stratum 20.9 → 93.9
- "Factor of 17–20" → **"factor of about 110"**. Under the joint σ bootstrap the lower 2.5 % end is 28–36 (skan3) against 8.7–9.9 (skan4).

**Supplement / stress table, per event and per image**

- \|ΔB_topo\| 0.002–0.039 → 0.002–0.018
- H_net 0.000–0.020 → 0.000–0.006; CI upper end 0.037 → 0.007
- "Largest estimate CHASE_DB1 tortuosity, factor 55" → DRIVE total length
- Macro \|e\| change −0.021 to −0.008 → **−0.014 to −0.006**; baseline error 0.64–1.72 → **0.51–1.56**
- The additive-bracket sentence (0.025–0.692) uses n_should from runs/rigr and was **not** recomputed.

**§R2 (`tab:main` is per biomarker and does not change)**

- "All four HRF slope estimates lie below 1 (0.37–0.81)" → "all three primary slopes … (0.42–0.81)"
- Intervals exclude 1 for "length and tortuosity" → "length" (tortuosity is reported as secondary: 0.37 [0.12, 0.69])
- The HRF CCC range 0.12–0.57 is unchanged.
- Proportional bias "6 of 16 primary cells" → **3 of 12**
- Slope intervals excluding 1: 6/16 → 3/12
- The abstract and highlight "all four HRF slope estimates …" need the same edit.

**§R1 families (`tab:family` means)**

| Segmenter | Cohort | Mean \|μ\| 4→3 | Residual SD 4→3 | Mean r 4→3 |
|---|---|---|---|---|
| U-Net | HRF | 1.540→1.282 | 1.479→1.039 | 0.487→0.420 |
| W-Net | HRF | 3.015→3.609 | 1.486→1.159 | 0.343→0.323 |
| SegFormer s0 | HRF | 1.926→2.164 | 1.618→1.665 | 0.414→0.380 |
| SegFormer s1 | HRF | 1.870→2.068 | 1.555→1.608 | 0.386→0.328 |
| U-Net | FIVES | 0.248→0.299 | 0.717→0.508 | 0.824→0.958 |
| W-Net | FIVES | 0.694→0.834 | 1.757→1.111 | 0.733→0.868 |
| SegFormer s0 | FIVES | 0.255→0.285 | 0.461→0.348 | 0.900→0.971 |
| SegFormer s1 | FIVES | 0.278→0.301 | 0.537→0.377 | 0.877→0.969 |

- Replication of the residual-SD ordering goes from 3/4 to **4/4**. The W-Net exception and its "tie 1.486 vs 1.483" sentence disappear (the tie sentence is sens_no_degen on skan4 and was not recomputed).
- The ΔAUC column stays skan4 (locked). The skan3 sensitivity values are given above.

**Supplement, 13 scale variants**

- HRF/FIVES \|offset\| ratio 3.9–6.2 → **2.8–4.4**; residual-SD ratio 1.4–2.0 → 1.4–2.1
- Axis separation 56–102 → **88–202**
- "Under the cross-cohort scale DRIVE overtakes HRF (1.36 vs 1.26)" → **no longer true (0.91 vs 1.17)**. The caveat sentence can be dropped, though the no-ordering statement can stay.

**Supplement, FIVES sample flow**

- Subsample mean \|offset\| 0.186–0.223 (0.214) → **0.225–0.273 (0.267)**
- Mean r 0.848–0.878 (0.861) → **0.953–0.973 (0.958)**
- The skan3 subsample gap range is −0.020 to +0.025; the skan4 range −0.019 to +0.031 stays locked.

**Fig. 8 cases (§R1 "pixel metrics" paragraph)**

- The selection rule picks different images: FIVES test_132_G → test_143_G and HRF 07_dr → 06_g.
- Dice 0.927 vs 0.803 → 0.928 vs 0.814 (difference 0.12 → 0.11)
- Macro error 0.30 vs 1.65 → **0.27 vs 1.53**, ratio 5.5 → 5.7

**Unchanged by construction**

- Oracle-table r̄ and min r (already non-tortuosity)
- Every per-biomarker table row
- All skan4 downstream AUCs and gaps (locked)
- E2/E4 per-biomarker Δr and safety endpoints
- The injection arm (skan4 classifier)
- The E5 gating "macro r 0.785 → 0.869", which is an 8-column mean, not the primary panel

## Files

| Script | Outputs |
|---|---|
| `src/pivot/r3_deming_lambda.py` | `r3_deming_lambda.csv`, `r3_deming_lambda_obs.csv`, `r3_deming_stability.csv`, `r3_deming_conclusions.csv`, `r3_deming_repro_check.csv`, `figs/pivot/r3_deming_lambda.{png,pdf}` |
| `src/pivot/r3_sigma_joint_boot.py` | `r3_main_table_joint.csv` (Table 4 feed), `r3_sigma_joint_width_compare.csv`, `r3_sigma_joint_boot.csv`, `r3_sigma_joint_panel.csv`, `r3_sigma_joint_claims.csv`, `r3_sigma_boot_scale.csv`, `figs/pivot/r3_sigma_joint_boot.{png,pdf}` |
| `src/pivot/r3_panel3.py` | `r3_panel3_audit.csv`, `r3_panel3_gap.csv`, `r3_panel3_family.csv`, `r3_panel3_topomatch.csv`, `r3_panel3_scale.csv`, `r3_panel3_fives_subsample.csv`, `r3_panel3_textranges.csv` |
