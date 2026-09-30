# E2 -- measurement-fidelity fine-tuning grid (ReliSeg vs faithful CF-Loss)
Generated 2026-09-18 20:42:04 by `src/pivot/e2_analysis.py`.

## Pre-registration this report is read against
- Design: `DECISIONS.md` 2026-09-17 00:30 / 02:00 / 13:30 and `proposal/07_pivot_proposal_v4.md`.
- **Primary endpoint**: FIVES (confirmatory set), `last.pt`, density / total_length / FD, **skan pipeline**. PVBM is a sensitivity analysis (section 8), never the headline.
- **Two reference arms**: the untouched `baseline` (`runs/seg/fives/seed<k>/best.pt`, re-inferred through the identical E2 path) and `continued` -- the same-budget control: same starting checkpoint, same 50 epochs, same LR schedule and batch size, base loss only (CE-Dice + 0.5*clDice, lambda = 0). A gain that survives the second reference is not simply more training.
- HRF is the development/mechanism set (its test split was touched by probes P1-P5); FIVES is confirmatory.
- `fid` = the checkpoint with the best *validation* measurement fidelity. Reported alongside, always labelled. Nothing is selected on a test metric.
- Safety margins: Dice/clDice drop <= 0.01 absolute; tortuosity and density delta-r >= -0.05; |delta bias| <= 0.25 sigma.
- sigma axis: `results/gateA_biomarker_scales_train.csv` (robust train-split scale, 1.4826 * MAD); bias_const / resid_sd / r defined exactly as in `src/pivot/e1_decompose.py`.

## Artefacts
- `results/pivot/e2_fidelity.csv` -- 696 rows
- `results/pivot/e2_delta.csv` -- 1328 rows
- `results/pivot/e2_pixel.csv` -- 87 rows
- `results/pivot/e2_downstream.csv` -- 224 rows
- `results/pivot/e2_zeroshot.csv` -- 480 rows
- `results/pivot/e2_safety.csv` -- 1484 rows

## 0. Coverage -- which cells this version of the report covers
All 36 (dataset x config x checkpoint) cells are complete.

_(no rows yet)_

Complete cells:

| dataset | config | checkpoint | seeds_done | zeroshot_done |
|---|---|---|---|---|
| fives | baseline | last | 6/6 | 6/6 |
| fives | cfloss | fid | 3/3 | n/a |
| fives | cfloss | last | 3/3 | 6/6 |
| fives | continued | fid | 6/6 | n/a |
| fives | continued | last | 6/6 | 6/6 |
| fives | reliseg | fid | 6/6 | n/a |
| fives | reliseg | last | 6/6 | 6/6 |
| fives | reliseg_nocl | fid | 3/3 | n/a |
| fives | reliseg_nocl | last | 3/3 | 6/6 |
| hrf | abl_cf_matched | fid | 1/1 | n/a |
| hrf | abl_cf_matched | last | 1/1 | n/a |
| hrf | abl_cf_on_base | fid | 1/1 | n/a |
| hrf | abl_cf_on_base | last | 1/1 | n/a |
| hrf | abl_density_only | fid | 1/1 | n/a |
| hrf | abl_density_only | last | 1/1 | n/a |
| hrf | abl_fd_only | fid | 1/1 | n/a |
| hrf | abl_fd_only | last | 1/1 | n/a |
| hrf | abl_fixed_ladder | fid | 1/1 | n/a |
| hrf | abl_fixed_ladder | last | 1/1 | n/a |
| hrf | abl_length_only | fid | 1/1 | n/a |
| hrf | abl_length_only | last | 1/1 | n/a |
| hrf | abl_no_density | fid | 1/1 | n/a |
| hrf | abl_no_density | last | 1/1 | n/a |
| hrf | abl_no_fd | fid | 1/1 | n/a |
| hrf | abl_no_fd | last | 1/1 | n/a |
| hrf | abl_no_length | fid | 1/1 | n/a |
| hrf | abl_no_length | last | 1/1 | n/a |
| hrf | baseline | last | 3/3 | 6/6 |
| hrf | cfloss | fid | 3/3 | n/a |
| hrf | cfloss | last | 3/3 | 6/6 |
| hrf | continued | fid | 3/3 | n/a |
| hrf | continued | last | 3/3 | 6/6 |
| hrf | reliseg | fid | 3/3 | n/a |
| hrf | reliseg | last | 3/3 | 6/6 |
| hrf | reliseg_nocl | fid | 3/3 | n/a |
| hrf | reliseg_nocl | last | 3/3 | 6/6 |

## 1. Primary endpoint -- FIVES confirmatory set, `last.pt`, skan
Paired image bootstrap over the FIVES test images, 2000 resamples, 95% percentile CI, against the **same-seed** reference. Source: `results/pivot/e2_delta.csv`.

### vs `baseline`, pooled over seeds (mean of the per-seed delta-r; `n_seeds` gives how many seeds each row pools)

| config | biomarker | d_r | lo | hi | seed_lo | seed_hi | n_seeds_positive | direction_consistent | d_bias_sigma |
|---|---|---|---|---|---|---|---|---|---|
| cfloss | FD | -0.007 | -0.011 | 0.001 | -0.043 | 0.029 | 1 | no | -0.015 |
| cfloss | density | -0.001 | -0.002 | 0.000 | -0.002 | 0.001 | 0 | yes | -0.045 |
| cfloss | tortuosity | -0.007 | -0.045 | 0.031 | -0.223 | 0.210 | 2 | no | -0.026 |
| cfloss | total_length | -0.001 | -0.001 | -0.000 | -0.002 | 0.001 | 0 | yes | -0.022 |
| continued | FD | 0.013 | 0.000 | 0.023 | 0.000 | 0.026 | 5 | no | 0.004 |
| continued | density | 0.001 | 0.000 | 0.002 | 0.000 | 0.001 | 6 | yes | 0.051 |
| continued | tortuosity | -0.017 | -0.090 | 0.060 | -0.102 | 0.069 | 3 | no | -0.005 |
| continued | total_length | 0.001 | 0.000 | 0.001 | -0.000 | 0.001 | 5 | no | -0.007 |
| reliseg | FD | 0.015 | 0.000 | 0.029 | -0.003 | 0.032 | 5 | no | 0.144 |
| reliseg | density | 0.001 | 0.000 | 0.002 | 0.000 | 0.002 | 6 | yes | 0.081 |
| reliseg | tortuosity | 0.001 | -0.063 | 0.064 | -0.080 | 0.083 | 4 | no | -0.030 |
| reliseg | total_length | 0.001 | 0.000 | 0.002 | 0.000 | 0.002 | 6 | yes | 0.102 |
| reliseg_nocl | FD | 0.002 | -0.004 | 0.008 | -0.016 | 0.021 | 2 | no | 0.149 |
| reliseg_nocl | density | 0.000 | -0.000 | 0.001 | -0.001 | 0.002 | 2 | no | 0.052 |
| reliseg_nocl | tortuosity | 0.007 | -0.072 | 0.066 | -0.154 | 0.168 | 2 | no | -0.059 |
| reliseg_nocl | total_length | 0.001 | -0.000 | 0.002 | -0.001 | 0.003 | 3 | yes | 0.103 |

### vs `continued`, pooled over seeds (mean of the per-seed delta-r; `n_seeds` gives how many seeds each row pools)

| config | biomarker | d_r | lo | hi | seed_lo | seed_hi | n_seeds_positive | direction_consistent | d_bias_sigma |
|---|---|---|---|---|---|---|---|---|---|
| cfloss | FD | -0.015 | -0.025 | 0.001 | -0.062 | 0.031 | 1 | no | -0.024 |
| cfloss | density | -0.001 | -0.004 | 0.000 | -0.002 | -0.000 | 0 | yes | -0.109 |
| cfloss | tortuosity | 0.057 | -0.023 | 0.190 | -0.080 | 0.194 | 3 | yes | 0.003 |
| cfloss | total_length | -0.001 | -0.002 | -0.000 | -0.002 | -0.000 | 0 | yes | -0.026 |
| reliseg | FD | 0.001 | -0.004 | 0.007 | -0.007 | 0.010 | 4 | no | 0.140 |
| reliseg | density | -0.000 | -0.000 | 0.000 | -0.000 | 0.000 | 3 | no | 0.030 |
| reliseg | tortuosity | 0.018 | -0.033 | 0.074 | -0.040 | 0.075 | 3 | no | -0.024 |
| reliseg | total_length | 0.001 | -0.000 | 0.001 | 0.000 | 0.001 | 5 | no | 0.109 |
| reliseg_nocl | FD | -0.006 | -0.009 | 0.001 | -0.011 | -0.001 | 0 | yes | 0.139 |
| reliseg_nocl | density | -0.000 | -0.001 | 0.000 | -0.001 | 0.000 | 1 | no | -0.012 |
| reliseg_nocl | tortuosity | 0.071 | -0.008 | 0.145 | -0.105 | 0.247 | 3 | yes | -0.030 |
| reliseg_nocl | total_length | 0.001 | -0.000 | 0.002 | -0.000 | 0.001 | 3 | yes | 0.099 |

`lo`/`hi` = paired image-bootstrap CI of the seed-averaged delta-r; `seed_lo`/`seed_hi` = 95% t interval over the `n_seeds` per-seed point estimates; `n_seeds_positive` out of `n_seeds` with `direction_consistent` gives the seed-level sign agreement.

### Seed by seed (vs `baseline`)

| config | seed | biomarker | d_r | lo | hi |
|---|---|---|---|---|---|
| cfloss | 0 | FD | 0.007 | -0.002 | 0.013 |
| cfloss | 1 | FD | -0.022 | -0.029 | 0.001 |
| cfloss | 2 | FD | -0.006 | -0.010 | 0.002 |
| cfloss | 0 | density | -0.001 | -0.004 | 0.001 |
| cfloss | 1 | density | -0.000 | -0.002 | 0.001 |
| cfloss | 2 | density | -0.000 | -0.001 | 0.000 |
| cfloss | 0 | tortuosity | 0.070 | -0.050 | 0.247 |
| cfloss | 1 | tortuosity | 0.012 | -0.034 | 0.059 |
| cfloss | 2 | tortuosity | -0.101 | -0.270 | 0.063 |
| cfloss | 0 | total_length | -0.001 | -0.003 | -0.000 |
| cfloss | 1 | total_length | -0.001 | -0.001 | -0.000 |
| cfloss | 2 | total_length | -0.000 | -0.001 | 0.000 |
| continued | 0 | FD | 0.002 | -0.004 | 0.008 |
| continued | 1 | FD | 0.010 | -0.004 | 0.018 |
| continued | 2 | FD | 0.013 | 0.001 | 0.022 |
| continued | 3 | FD | 0.025 | 0.001 | 0.035 |
| continued | 4 | FD | 0.030 | 0.001 | 0.050 |
| continued | 5 | FD | -0.000 | -0.005 | 0.006 |
| continued | 0 | density | 0.000 | -0.000 | 0.001 |
| continued | 1 | density | 0.001 | -0.000 | 0.002 |
| continued | 2 | density | 0.001 | -0.000 | 0.003 |
| continued | 3 | density | 0.002 | 0.000 | 0.003 |
| continued | 4 | density | 0.001 | 0.000 | 0.003 |
| continued | 5 | density | 0.001 | 0.000 | 0.002 |
| continued | 0 | tortuosity | 0.043 | -0.006 | 0.092 |
| continued | 1 | tortuosity | -0.109 | -0.317 | 0.043 |
| continued | 2 | tortuosity | -0.126 | -0.290 | 0.028 |
| continued | 3 | tortuosity | 0.071 | -0.137 | 0.321 |
| continued | 4 | tortuosity | 0.020 | -0.146 | 0.217 |
| continued | 5 | tortuosity | -0.000 | -0.092 | 0.070 |
| continued | 0 | total_length | -0.000 | -0.001 | 0.000 |
| continued | 1 | total_length | 0.000 | -0.000 | 0.001 |
| continued | 2 | total_length | 0.001 | -0.000 | 0.002 |
| continued | 3 | total_length | 0.002 | 0.001 | 0.003 |
| continued | 4 | total_length | 0.001 | -0.000 | 0.002 |
| continued | 5 | total_length | 0.001 | -0.000 | 0.001 |
| reliseg | 0 | FD | 0.005 | -0.004 | 0.014 |
| reliseg | 1 | FD | -0.001 | -0.010 | 0.007 |
| reliseg | 2 | FD | 0.011 | -0.006 | 0.028 |
| reliseg | 3 | FD | 0.039 | 0.002 | 0.055 |
| reliseg | 4 | FD | 0.033 | 0.002 | 0.060 |
| reliseg | 5 | FD | 0.002 | -0.008 | 0.016 |
| reliseg | 0 | density | 0.000 | -0.000 | 0.001 |
| reliseg | 1 | density | 0.001 | -0.000 | 0.002 |
| reliseg | 2 | density | 0.001 | -0.001 | 0.003 |
| reliseg | 3 | density | 0.002 | 0.001 | 0.004 |
| reliseg | 4 | density | 0.002 | 0.000 | 0.003 |
| reliseg | 5 | density | 0.000 | -0.000 | 0.001 |
| reliseg | 0 | tortuosity | 0.033 | -0.135 | 0.199 |
| reliseg | 1 | tortuosity | 0.012 | -0.090 | 0.105 |
| reliseg | 2 | tortuosity | -0.133 | -0.293 | 0.002 |
| reliseg | 3 | tortuosity | 0.105 | -0.103 | 0.337 |
| reliseg | 4 | tortuosity | -0.013 | -0.076 | 0.043 |
| reliseg | 5 | tortuosity | 0.003 | -0.043 | 0.056 |
| reliseg | 0 | total_length | 0.000 | -0.001 | 0.001 |
| reliseg | 1 | total_length | 0.000 | -0.001 | 0.002 |
| reliseg | 2 | total_length | 0.001 | -0.000 | 0.003 |
| reliseg | 3 | total_length | 0.003 | 0.001 | 0.005 |
| reliseg | 4 | total_length | 0.002 | 0.000 | 0.003 |
| reliseg | 5 | total_length | 0.001 | -0.000 | 0.002 |
| reliseg_nocl | 0 | FD | -0.006 | -0.014 | 0.002 |
| reliseg_nocl | 1 | FD | 0.006 | -0.002 | 0.010 |
| reliseg_nocl | 2 | FD | 0.008 | -0.001 | 0.016 |
| reliseg_nocl | 0 | density | -0.000 | -0.001 | 0.001 |
| reliseg_nocl | 1 | density | 0.000 | -0.001 | 0.002 |
| reliseg_nocl | 2 | density | 0.001 | -0.000 | 0.003 |
| reliseg_nocl | 0 | tortuosity | 0.051 | -0.048 | 0.148 |
| reliseg_nocl | 1 | tortuosity | 0.039 | -0.008 | 0.085 |
| reliseg_nocl | 2 | tortuosity | -0.067 | -0.242 | 0.087 |
| reliseg_nocl | 0 | total_length | 0.000 | -0.001 | 0.001 |
| reliseg_nocl | 1 | total_length | 0.001 | -0.000 | 0.002 |
| reliseg_nocl | 2 | total_length | 0.002 | 0.000 | 0.003 |

### Seed by seed (vs `continued`)

| config | seed | biomarker | d_r | lo | hi |
|---|---|---|---|---|---|
| cfloss | 0 | FD | 0.005 | -0.005 | 0.016 |
| cfloss | 1 | FD | -0.032 | -0.044 | 0.003 |
| cfloss | 2 | FD | -0.019 | -0.030 | -0.001 |
| cfloss | 0 | density | -0.002 | -0.004 | 0.000 |
| cfloss | 1 | density | -0.001 | -0.003 | 0.001 |
| cfloss | 2 | density | -0.001 | -0.004 | 0.000 |
| cfloss | 0 | tortuosity | 0.026 | -0.112 | 0.226 |
| cfloss | 1 | tortuosity | 0.121 | -0.036 | 0.325 |
| cfloss | 2 | tortuosity | 0.024 | -0.028 | 0.085 |
| cfloss | 0 | total_length | -0.001 | -0.002 | -0.000 |
| cfloss | 1 | total_length | -0.001 | -0.002 | -0.000 |
| cfloss | 2 | total_length | -0.001 | -0.002 | -0.000 |
| reliseg | 0 | FD | 0.003 | -0.001 | 0.007 |
| reliseg | 1 | FD | -0.011 | -0.016 | 0.003 |
| reliseg | 2 | FD | -0.002 | -0.010 | 0.006 |
| reliseg | 3 | FD | 0.014 | 0.001 | 0.021 |
| reliseg | 4 | FD | 0.002 | -0.005 | 0.010 |
| reliseg | 5 | FD | 0.002 | -0.005 | 0.012 |
| reliseg | 0 | density | -0.000 | -0.001 | 0.001 |
| reliseg | 1 | density | 0.000 | -0.000 | 0.001 |
| reliseg | 2 | density | -0.000 | -0.001 | 0.000 |
| reliseg | 3 | density | 0.000 | -0.000 | 0.001 |
| reliseg | 4 | density | 0.000 | -0.000 | 0.001 |
| reliseg | 5 | density | -0.001 | -0.001 | 0.000 |
| reliseg | 0 | tortuosity | -0.010 | -0.168 | 0.155 |
| reliseg | 1 | tortuosity | 0.120 | -0.010 | 0.254 |
| reliseg | 2 | tortuosity | -0.008 | -0.103 | 0.064 |
| reliseg | 3 | tortuosity | 0.034 | -0.053 | 0.132 |
| reliseg | 4 | tortuosity | -0.033 | -0.206 | 0.103 |
| reliseg | 5 | tortuosity | 0.003 | -0.065 | 0.093 |
| reliseg | 0 | total_length | 0.000 | -0.000 | 0.002 |
| reliseg | 1 | total_length | 0.000 | -0.001 | 0.002 |
| reliseg | 2 | total_length | 0.000 | -0.000 | 0.001 |
| reliseg | 3 | total_length | 0.001 | 0.000 | 0.002 |
| reliseg | 4 | total_length | 0.001 | -0.000 | 0.002 |
| reliseg | 5 | total_length | -0.000 | -0.001 | 0.001 |
| reliseg_nocl | 0 | FD | -0.008 | -0.012 | 0.000 |
| reliseg_nocl | 1 | FD | -0.004 | -0.011 | 0.004 |
| reliseg_nocl | 2 | FD | -0.005 | -0.008 | 0.001 |
| reliseg_nocl | 0 | density | -0.000 | -0.001 | 0.000 |
| reliseg_nocl | 1 | density | -0.000 | -0.001 | 0.001 |
| reliseg_nocl | 2 | density | 0.000 | -0.001 | 0.001 |
| reliseg_nocl | 0 | tortuosity | 0.007 | -0.078 | 0.097 |
| reliseg_nocl | 1 | tortuosity | 0.148 | -0.018 | 0.346 |
| reliseg_nocl | 2 | tortuosity | 0.058 | -0.011 | 0.125 |
| reliseg_nocl | 0 | total_length | 0.000 | -0.001 | 0.002 |
| reliseg_nocl | 1 | total_length | 0.000 | -0.001 | 0.001 |
| reliseg_nocl | 2 | total_length | 0.001 | -0.000 | 0.002 |

### `fid` checkpoint (validation-fidelity selection; secondary)

| config | biomarker | d_r | lo | hi | direction_consistent |
|---|---|---|---|---|---|
| cfloss | FD | -0.027 | -0.042 | 0.000 | no |
| cfloss | density | -0.001 | -0.003 | 0.000 | yes |
| cfloss | tortuosity | 0.017 | -0.065 | 0.087 | no |
| cfloss | total_length | -0.001 | -0.002 | -0.000 | yes |
| continued | FD | 0.007 | -0.001 | 0.014 | no |
| continued | density | 0.000 | -0.000 | 0.001 | no |
| continued | tortuosity | -0.061 | -0.128 | 0.049 | no |
| continued | total_length | -0.000 | -0.000 | 0.000 | no |
| reliseg | FD | 0.017 | 0.001 | 0.031 | yes |
| reliseg | density | 0.000 | -0.000 | 0.001 | no |
| reliseg | tortuosity | 0.023 | -0.043 | 0.084 | no |
| reliseg | total_length | 0.001 | -0.000 | 0.002 | no |
| reliseg_nocl | FD | -0.004 | -0.007 | 0.001 | yes |
| reliseg_nocl | density | -0.001 | -0.002 | 0.000 | yes |
| reliseg_nocl | tortuosity | -0.034 | -0.152 | 0.052 | no |
| reliseg_nocl | total_length | -0.000 | -0.001 | 0.001 | no |

## 2. Absolute fidelity (FIVES, `last.pt`, skan, seed-averaged)
Source: `results/pivot/e2_fidelity.csv`.

| config | biomarker | r_pearson | r_spearman | bias_const_sigma | resid_sd_sigma |
|---|---|---|---|---|---|
| baseline | FD | 0.900 | 0.956 | -0.674 | 1.119 |
| cfloss | FD | 0.902 | 0.956 | -0.682 | 1.104 |
| continued | FD | 0.913 | 0.958 | -0.670 | 1.040 |
| reliseg | FD | 0.915 | 0.961 | -0.530 | 1.008 |
| reliseg_nocl | FD | 0.911 | 0.962 | -0.518 | 1.061 |
| baseline | density | 0.984 | 0.975 | -0.030 | 0.238 |
| cfloss | density | 0.983 | 0.974 | -0.084 | 0.249 |
| continued | density | 0.984 | 0.976 | 0.021 | 0.230 |
| reliseg | density | 0.984 | 0.976 | 0.051 | 0.231 |
| reliseg_nocl | density | 0.984 | 0.976 | 0.013 | 0.235 |
| baseline | tortuosity | 0.483 | 0.681 | -0.052 | 1.331 |
| cfloss | tortuosity | 0.477 | 0.693 | -0.093 | 1.315 |
| continued | tortuosity | 0.466 | 0.671 | -0.057 | 1.316 |
| reliseg | tortuosity | 0.484 | 0.684 | -0.081 | 1.288 |
| reliseg_nocl | tortuosity | 0.491 | 0.680 | -0.127 | 1.255 |
| baseline | total_length | 0.978 | 0.972 | -0.254 | 0.277 |
| cfloss | total_length | 0.977 | 0.971 | -0.283 | 0.281 |
| continued | total_length | 0.978 | 0.972 | -0.261 | 0.272 |
| reliseg | total_length | 0.979 | 0.975 | -0.152 | 0.270 |
| reliseg_nocl | total_length | 0.979 | 0.975 | -0.158 | 0.271 |

## 3. Pixel metrics (non-inferiority margin 0.01 absolute)
Source: `results/pivot/e2_pixel.csv` (per-image `runs/pivot/<ds>/seed<k>/<cfg>/pred*/pixel_metrics.csv`, cross-checked against `infer_meta.json`). Deltas are vs the same-seed baseline.

| dataset | config | mean_dice | mean_cldice | d_dice | d_cldice |
|---|---|---|---|---|---|
| fives | cfloss | 0.9029 | 0.9060 | 0.0008 | -0.0007 |
| fives | continued | 0.9027 | 0.9100 | 0.0012 | 0.0037 |
| fives | reliseg | 0.9010 | 0.9075 | -0.0005 | 0.0012 |
| fives | reliseg_nocl | 0.9011 | 0.9039 | -0.0010 | -0.0028 |
| hrf | abl_cf_matched | 0.8139 | 0.8257 | -0.0000 | 0.0006 |
| hrf | abl_cf_on_base | 0.8139 | 0.8312 | -0.0001 | 0.0062 |
| hrf | abl_density_only | 0.8137 | 0.8264 | -0.0003 | 0.0014 |
| hrf | abl_fd_only | 0.8132 | 0.8364 | -0.0007 | 0.0114 |
| hrf | abl_fixed_ladder | 0.8085 | 0.8224 | -0.0054 | -0.0026 |
| hrf | abl_length_only | 0.8078 | 0.8352 | -0.0061 | 0.0102 |
| hrf | abl_no_density | 0.8078 | 0.8331 | -0.0061 | 0.0081 |
| hrf | abl_no_fd | 0.8081 | 0.8207 | -0.0059 | -0.0043 |
| hrf | abl_no_length | 0.8135 | 0.8266 | -0.0005 | 0.0016 |
| hrf | cfloss | 0.8077 | 0.7991 | -0.0059 | -0.0233 |
| hrf | continued | 0.8125 | 0.8330 | -0.0011 | 0.0106 |
| hrf | reliseg | 0.8075 | 0.8177 | -0.0061 | -0.0048 |
| hrf | reliseg_nocl | 0.8076 | 0.8046 | -0.0061 | -0.0178 |

## 4. Downstream utility (P3 protocol)
Source: `results/pivot/e2_downstream.csv`; protocol `src/pivot/p3_downstream.py` (standardised multinomial logistic regression and a small GBDT, 5-fold stratified CV repeated 3x, pooled out-of-fold macro one-vs-rest AUC, paired image bootstrap of delta-AUC vs the same-seed baseline).

Primary panel (skan4), FIVES (confirmatory):

| clf | config | reference | seed | macro_auc | ci_lo | ci_hi | d_auc_vs_ref | d_auc_lo | d_auc_hi |
|---|---|---|---|---|---|---|---|---|---|
| gbdt | baseline | baseline | 0 | 0.7187 | 0.6672 | 0.7694 |  |  |  |
| gbdt | baseline | baseline | 1 | 0.7167 | 0.6661 | 0.7656 |  |  |  |
| gbdt | baseline | baseline | 2 | 0.6758 | 0.6268 | 0.7272 |  |  |  |
| gbdt | baseline | continued | 0 | 0.7187 | 0.6672 | 0.7694 | 0.0140 | -0.0154 | 0.0423 |
| gbdt | baseline | continued | 1 | 0.7167 | 0.6661 | 0.7656 | 0.0117 | -0.0157 | 0.0386 |
| gbdt | baseline | continued | 2 | 0.6758 | 0.6268 | 0.7272 | -0.0480 | -0.0813 | -0.0138 |
| gbdt | cfloss | baseline | 0 | 0.6955 | 0.6443 | 0.7457 | -0.0232 | -0.0562 | 0.0102 |
| gbdt | cfloss | baseline | 1 | 0.6899 | 0.6381 | 0.7395 | -0.0268 | -0.0556 | 0.0017 |
| gbdt | cfloss | baseline | 2 | 0.7024 | 0.6511 | 0.7520 | 0.0266 | -0.0027 | 0.0561 |
| gbdt | cfloss | continued | 0 | 0.6955 | 0.6443 | 0.7457 | -0.0092 | -0.0367 | 0.0198 |
| gbdt | cfloss | continued | 1 | 0.6899 | 0.6381 | 0.7395 | -0.0151 | -0.0424 | 0.0133 |
| gbdt | cfloss | continued | 2 | 0.7024 | 0.6511 | 0.7520 | -0.0214 | -0.0531 | 0.0094 |
| gbdt | continued | baseline | 0 | 0.7047 | 0.6515 | 0.7550 | -0.0140 | -0.0423 | 0.0154 |
| gbdt | continued | baseline | 1 | 0.7050 | 0.6547 | 0.7554 | -0.0117 | -0.0386 | 0.0157 |
| gbdt | continued | baseline | 2 | 0.7238 | 0.6745 | 0.7726 | 0.0480 | 0.0138 | 0.0813 |
| gbdt | gt | baseline | - | 0.6869 | 0.6382 | 0.7353 |  |  |  |
| gbdt | reliseg | baseline | 0 | 0.7562 | 0.7080 | 0.8021 | 0.0374 | 0.0042 | 0.0748 |
| gbdt | reliseg | baseline | 1 | 0.6839 | 0.6335 | 0.7349 | -0.0328 | -0.0650 | -0.0010 |
| gbdt | reliseg | baseline | 2 | 0.7234 | 0.6737 | 0.7704 | 0.0476 | 0.0122 | 0.0825 |
| gbdt | reliseg | continued | 0 | 0.7562 | 0.7080 | 0.8021 | 0.0515 | 0.0216 | 0.0832 |
| gbdt | reliseg | continued | 1 | 0.6839 | 0.6335 | 0.7349 | -0.0211 | -0.0495 | 0.0082 |
| gbdt | reliseg | continued | 2 | 0.7234 | 0.6737 | 0.7704 | -0.0005 | -0.0305 | 0.0295 |
| gbdt | reliseg_nocl | baseline | 0 | 0.7192 | 0.6668 | 0.7681 | 0.0005 | -0.0339 | 0.0354 |
| gbdt | reliseg_nocl | baseline | 1 | 0.7075 | 0.6564 | 0.7586 | -0.0092 | -0.0426 | 0.0267 |
| gbdt | reliseg_nocl | baseline | 2 | 0.7109 | 0.6618 | 0.7606 | 0.0351 | 0.0032 | 0.0671 |
| gbdt | reliseg_nocl | continued | 0 | 0.7192 | 0.6668 | 0.7681 | 0.0145 | -0.0153 | 0.0451 |
| gbdt | reliseg_nocl | continued | 1 | 0.7075 | 0.6564 | 0.7586 | 0.0025 | -0.0280 | 0.0339 |
| gbdt | reliseg_nocl | continued | 2 | 0.7109 | 0.6618 | 0.7606 | -0.0129 | -0.0467 | 0.0198 |
| logreg | baseline | baseline | 0 | 0.7135 | 0.6632 | 0.7596 |  |  |  |
| logreg | baseline | baseline | 1 | 0.7042 | 0.6530 | 0.7496 |  |  |  |
| logreg | baseline | baseline | 2 | 0.7009 | 0.6506 | 0.7476 |  |  |  |
| logreg | baseline | continued | 0 | 0.7135 | 0.6632 | 0.7596 | 0.0071 | -0.0023 | 0.0168 |
| logreg | baseline | continued | 1 | 0.7042 | 0.6530 | 0.7496 | -0.0094 | -0.0306 | 0.0055 |
| logreg | baseline | continued | 2 | 0.7009 | 0.6506 | 0.7476 | -0.0078 | -0.0291 | 0.0106 |
| logreg | cfloss | baseline | 0 | 0.6955 | 0.6459 | 0.7403 | -0.0179 | -0.0397 | -0.0002 |
| logreg | cfloss | baseline | 1 | 0.7059 | 0.6546 | 0.7523 | 0.0017 | -0.0068 | 0.0113 |
| logreg | cfloss | baseline | 2 | 0.7087 | 0.6609 | 0.7540 | 0.0078 | -0.0076 | 0.0264 |
| logreg | cfloss | continued | 0 | 0.6955 | 0.6459 | 0.7403 | -0.0108 | -0.0314 | 0.0041 |
| logreg | cfloss | continued | 1 | 0.7059 | 0.6546 | 0.7523 | -0.0077 | -0.0274 | 0.0061 |
| logreg | cfloss | continued | 2 | 0.7087 | 0.6609 | 0.7540 | -0.0001 | -0.0139 | 0.0129 |
| logreg | continued | baseline | 0 | 0.7064 | 0.6573 | 0.7518 | -0.0071 | -0.0168 | 0.0023 |
| logreg | continued | baseline | 1 | 0.7136 | 0.6625 | 0.7594 | 0.0094 | -0.0055 | 0.0306 |
| logreg | continued | baseline | 2 | 0.7088 | 0.6601 | 0.7534 | 0.0078 | -0.0106 | 0.0291 |
| logreg | gt | baseline | - | 0.7135 | 0.6673 | 0.7583 |  |  |  |
| logreg | reliseg | baseline | 0 | 0.7144 | 0.6656 | 0.7603 | 0.0010 | -0.0146 | 0.0149 |
| logreg | reliseg | baseline | 1 | 0.7184 | 0.6701 | 0.7628 | 0.0142 | -0.0004 | 0.0321 |
| logreg | reliseg | baseline | 2 | 0.7105 | 0.6627 | 0.7540 | 0.0096 | -0.0108 | 0.0312 |
| logreg | reliseg | continued | 0 | 0.7144 | 0.6656 | 0.7603 | 0.0081 | -0.0076 | 0.0231 |
| logreg | reliseg | continued | 1 | 0.7184 | 0.6701 | 0.7628 | 0.0049 | -0.0075 | 0.0175 |
| logreg | reliseg | continued | 2 | 0.7105 | 0.6627 | 0.7540 | 0.0017 | -0.0141 | 0.0159 |
| logreg | reliseg_nocl | baseline | 0 | 0.7130 | 0.6633 | 0.7598 | -0.0004 | -0.0103 | 0.0088 |
| logreg | reliseg_nocl | baseline | 1 | 0.7034 | 0.6534 | 0.7494 | -0.0008 | -0.0087 | 0.0081 |
| logreg | reliseg_nocl | baseline | 2 | 0.7184 | 0.6720 | 0.7630 | 0.0175 | 0.0009 | 0.0386 |
| logreg | reliseg_nocl | continued | 0 | 0.7130 | 0.6633 | 0.7598 | 0.0067 | -0.0052 | 0.0189 |
| logreg | reliseg_nocl | continued | 1 | 0.7034 | 0.6534 | 0.7494 | -0.0102 | -0.0273 | 0.0031 |
| logreg | reliseg_nocl | continued | 2 | 0.7184 | 0.6720 | 0.7630 | 0.0097 | -0.0015 | 0.0218 |

Primary panel (skan4), HRF (development set):

| clf | config | reference | seed | macro_auc | ci_lo | ci_hi | d_auc_vs_ref | d_auc_lo | d_auc_hi |
|---|---|---|---|---|---|---|---|---|---|
| gbdt | baseline | baseline | 0 | 0.6683 | 0.4859 | 0.8328 |  |  |  |
| gbdt | baseline | baseline | 1 | 0.7167 | 0.5586 | 0.8590 |  |  |  |
| gbdt | baseline | baseline | 2 | 0.6500 | 0.4846 | 0.8076 |  |  |  |
| gbdt | baseline | continued | 0 | 0.6683 | 0.4859 | 0.8328 | -0.1133 | -0.2468 | 0.0021 |
| gbdt | baseline | continued | 1 | 0.7167 | 0.5586 | 0.8590 | -0.0300 | -0.1485 | 0.0805 |
| gbdt | baseline | continued | 2 | 0.6500 | 0.4846 | 0.8076 | -0.0983 | -0.2002 | -0.0156 |
| gbdt | cfloss | baseline | 0 | 0.7033 | 0.5288 | 0.8587 | 0.0350 | -0.1163 | 0.1870 |
| gbdt | cfloss | baseline | 1 | 0.6400 | 0.4572 | 0.8139 | -0.0767 | -0.1938 | 0.0323 |
| gbdt | cfloss | baseline | 2 | 0.7233 | 0.5512 | 0.8734 | 0.0733 | -0.0666 | 0.2115 |
| gbdt | cfloss | continued | 0 | 0.7033 | 0.5288 | 0.8587 | -0.0783 | -0.1796 | 0.0036 |
| gbdt | cfloss | continued | 1 | 0.6400 | 0.4572 | 0.8139 | -0.1067 | -0.2464 | 0.0277 |
| gbdt | cfloss | continued | 2 | 0.7233 | 0.5512 | 0.8734 | -0.0250 | -0.1420 | 0.0831 |
| gbdt | continued | baseline | 0 | 0.7817 | 0.6402 | 0.9058 | 0.1133 | -0.0021 | 0.2468 |
| gbdt | continued | baseline | 1 | 0.7467 | 0.5990 | 0.8777 | 0.0300 | -0.0805 | 0.1485 |
| gbdt | continued | baseline | 2 | 0.7483 | 0.5835 | 0.8932 | 0.0983 | 0.0156 | 0.2002 |
| gbdt | gt | baseline | - | 0.8367 | 0.7107 | 0.9508 |  |  |  |
| gbdt | reliseg | baseline | 0 | 0.7383 | 0.5891 | 0.8802 | 0.0700 | -0.1452 | 0.2857 |
| gbdt | reliseg | baseline | 1 | 0.7500 | 0.5944 | 0.8856 | 0.0333 | -0.0540 | 0.1455 |
| gbdt | reliseg | baseline | 2 | 0.7167 | 0.5817 | 0.8440 | 0.0667 | -0.0154 | 0.1620 |
| gbdt | reliseg | continued | 0 | 0.7383 | 0.5891 | 0.8802 | -0.0433 | -0.1995 | 0.0951 |
| gbdt | reliseg | continued | 1 | 0.7500 | 0.5944 | 0.8856 | 0.0033 | -0.0609 | 0.0756 |
| gbdt | reliseg | continued | 2 | 0.7167 | 0.5817 | 0.8440 | -0.0317 | -0.1426 | 0.0573 |
| gbdt | reliseg_nocl | baseline | 0 | 0.7433 | 0.5967 | 0.8684 | 0.0750 | -0.0254 | 0.1875 |
| gbdt | reliseg_nocl | baseline | 1 | 0.7467 | 0.6028 | 0.8778 | 0.0300 | -0.0858 | 0.1655 |
| gbdt | reliseg_nocl | baseline | 2 | 0.7200 | 0.5620 | 0.8612 | 0.0700 | -0.0163 | 0.1641 |
| gbdt | reliseg_nocl | continued | 0 | 0.7433 | 0.5967 | 0.8684 | -0.0383 | -0.1330 | 0.0427 |
| gbdt | reliseg_nocl | continued | 1 | 0.7467 | 0.6028 | 0.8778 | 0.0000 | -0.0918 | 0.1004 |
| gbdt | reliseg_nocl | continued | 2 | 0.7200 | 0.5620 | 0.8612 | -0.0283 | -0.1317 | 0.0584 |
| logreg | baseline | baseline | 0 | 0.7883 | 0.6424 | 0.9084 |  |  |  |
| logreg | baseline | baseline | 1 | 0.7333 | 0.5867 | 0.8649 |  |  |  |
| logreg | baseline | baseline | 2 | 0.7467 | 0.5867 | 0.8859 |  |  |  |
| logreg | baseline | continued | 0 | 0.7883 | 0.6424 | 0.9084 | 0.0583 | -0.0136 | 0.1391 |
| logreg | baseline | continued | 1 | 0.7333 | 0.5867 | 0.8649 | -0.0317 | -0.0936 | 0.0154 |
| logreg | baseline | continued | 2 | 0.7467 | 0.5867 | 0.8859 | -0.0283 | -0.0772 | 0.0180 |
| logreg | cfloss | baseline | 0 | 0.7633 | 0.6033 | 0.8996 | -0.0250 | -0.0953 | 0.0426 |
| logreg | cfloss | baseline | 1 | 0.7350 | 0.5592 | 0.8810 | 0.0017 | -0.0881 | 0.0921 |
| logreg | cfloss | baseline | 2 | 0.7733 | 0.6110 | 0.9136 | 0.0267 | -0.0176 | 0.0770 |
| logreg | cfloss | continued | 0 | 0.7633 | 0.6033 | 0.8996 | 0.0333 | -0.0439 | 0.1183 |
| logreg | cfloss | continued | 1 | 0.7350 | 0.5592 | 0.8810 | -0.0300 | -0.0970 | 0.0229 |
| logreg | cfloss | continued | 2 | 0.7733 | 0.6110 | 0.9136 | -0.0017 | -0.0314 | 0.0309 |
| logreg | continued | baseline | 0 | 0.7300 | 0.5831 | 0.8675 | -0.0583 | -0.1391 | 0.0136 |
| logreg | continued | baseline | 1 | 0.7650 | 0.6118 | 0.8984 | 0.0317 | -0.0154 | 0.0936 |
| logreg | continued | baseline | 2 | 0.7750 | 0.6125 | 0.9103 | 0.0283 | -0.0180 | 0.0772 |
| logreg | gt | baseline | - | 0.9650 | 0.9036 | 1.0000 |  |  |  |
| logreg | reliseg | baseline | 0 | 0.7817 | 0.6287 | 0.9094 | -0.0067 | -0.1257 | 0.1071 |
| logreg | reliseg | baseline | 1 | 0.7650 | 0.6209 | 0.8964 | 0.0317 | -0.0386 | 0.1043 |
| logreg | reliseg | baseline | 2 | 0.7800 | 0.6312 | 0.9092 | 0.0333 | -0.0168 | 0.0911 |
| logreg | reliseg | continued | 0 | 0.7817 | 0.6287 | 0.9094 | 0.0517 | -0.0770 | 0.1868 |
| logreg | reliseg | continued | 1 | 0.7650 | 0.6209 | 0.8964 | 0.0000 | -0.1043 | 0.0773 |
| logreg | reliseg | continued | 2 | 0.7800 | 0.6312 | 0.9092 | 0.0050 | -0.0438 | 0.0498 |
| logreg | reliseg_nocl | baseline | 0 | 0.7317 | 0.5748 | 0.8708 | -0.0567 | -0.1295 | 0.0051 |
| logreg | reliseg_nocl | baseline | 1 | 0.7617 | 0.6226 | 0.8875 | 0.0283 | -0.0459 | 0.0991 |
| logreg | reliseg_nocl | baseline | 2 | 0.7700 | 0.6098 | 0.9094 | 0.0233 | -0.0383 | 0.0925 |
| logreg | reliseg_nocl | continued | 0 | 0.7317 | 0.5748 | 0.8708 | 0.0017 | -0.0624 | 0.0737 |
| logreg | reliseg_nocl | continued | 1 | 0.7617 | 0.6226 | 0.8875 | -0.0033 | -0.0762 | 0.0553 |
| logreg | reliseg_nocl | continued | 2 | 0.7700 | 0.6098 | 0.9094 | -0.0050 | -0.0486 | 0.0370 |

Sensitivity -- the eight-column panel (both pipelines), FIVES:

| clf | config | reference | seed | macro_auc | ci_lo | ci_hi | d_auc_vs_ref | d_auc_lo | d_auc_hi |
|---|---|---|---|---|---|---|---|---|---|
| gbdt | baseline | baseline | 0 | 0.7082 | 0.6559 | 0.7599 |  |  |  |
| gbdt | baseline | baseline | 1 | 0.7219 | 0.6711 | 0.7706 |  |  |  |
| gbdt | baseline | baseline | 2 | 0.6789 | 0.6262 | 0.7314 |  |  |  |
| gbdt | baseline | continued | 0 | 0.7082 | 0.6559 | 0.7599 | -0.0105 | -0.0401 | 0.0191 |
| gbdt | baseline | continued | 1 | 0.7219 | 0.6711 | 0.7706 | 0.0153 | -0.0125 | 0.0457 |
| gbdt | baseline | continued | 2 | 0.6789 | 0.6262 | 0.7314 | -0.0385 | -0.0784 | 0.0003 |
| gbdt | cfloss | baseline | 0 | 0.7145 | 0.6628 | 0.7682 | 0.0063 | -0.0285 | 0.0399 |
| gbdt | cfloss | baseline | 1 | 0.6938 | 0.6443 | 0.7440 | -0.0282 | -0.0577 | 0.0003 |
| gbdt | cfloss | baseline | 2 | 0.6973 | 0.6466 | 0.7494 | 0.0183 | -0.0148 | 0.0518 |
| gbdt | cfloss | continued | 0 | 0.7145 | 0.6628 | 0.7682 | -0.0042 | -0.0385 | 0.0300 |
| gbdt | cfloss | continued | 1 | 0.6938 | 0.6443 | 0.7440 | -0.0129 | -0.0432 | 0.0173 |
| gbdt | cfloss | continued | 2 | 0.6973 | 0.6466 | 0.7494 | -0.0202 | -0.0574 | 0.0166 |
| gbdt | continued | baseline | 0 | 0.7187 | 0.6665 | 0.7687 | 0.0105 | -0.0191 | 0.0401 |
| gbdt | continued | baseline | 1 | 0.7067 | 0.6562 | 0.7565 | -0.0153 | -0.0457 | 0.0125 |
| gbdt | continued | baseline | 2 | 0.7174 | 0.6677 | 0.7684 | 0.0385 | -0.0003 | 0.0784 |
| gbdt | gt | baseline | - | 0.6924 | 0.6444 | 0.7381 |  |  |  |
| gbdt | reliseg | baseline | 0 | 0.7727 | 0.7259 | 0.8172 | 0.0645 | 0.0322 | 0.1001 |
| gbdt | reliseg | baseline | 1 | 0.7238 | 0.6727 | 0.7734 | 0.0019 | -0.0317 | 0.0342 |
| gbdt | reliseg | baseline | 2 | 0.7415 | 0.6907 | 0.7914 | 0.0626 | 0.0228 | 0.1030 |
| gbdt | reliseg | continued | 0 | 0.7727 | 0.7259 | 0.8172 | 0.0541 | 0.0201 | 0.0894 |
| gbdt | reliseg | continued | 1 | 0.7238 | 0.6727 | 0.7734 | 0.0172 | -0.0156 | 0.0493 |
| gbdt | reliseg | continued | 2 | 0.7415 | 0.6907 | 0.7914 | 0.0241 | -0.0140 | 0.0601 |
| gbdt | reliseg_nocl | baseline | 0 | 0.7260 | 0.6735 | 0.7743 | 0.0178 | -0.0179 | 0.0536 |
| gbdt | reliseg_nocl | baseline | 1 | 0.6877 | 0.6355 | 0.7380 | -0.0343 | -0.0658 | -0.0040 |
| gbdt | reliseg_nocl | baseline | 2 | 0.7323 | 0.6811 | 0.7824 | 0.0534 | 0.0151 | 0.0900 |
| gbdt | reliseg_nocl | continued | 0 | 0.7260 | 0.6735 | 0.7743 | 0.0073 | -0.0295 | 0.0434 |
| gbdt | reliseg_nocl | continued | 1 | 0.6877 | 0.6355 | 0.7380 | -0.0190 | -0.0497 | 0.0126 |
| gbdt | reliseg_nocl | continued | 2 | 0.7323 | 0.6811 | 0.7824 | 0.0149 | -0.0244 | 0.0546 |
| logreg | baseline | baseline | 0 | 0.7169 | 0.6679 | 0.7615 |  |  |  |
| logreg | baseline | baseline | 1 | 0.7134 | 0.6645 | 0.7585 |  |  |  |
| logreg | baseline | baseline | 2 | 0.7098 | 0.6586 | 0.7555 |  |  |  |
| logreg | baseline | continued | 0 | 0.7169 | 0.6679 | 0.7615 | 0.0081 | -0.0035 | 0.0206 |
| logreg | baseline | continued | 1 | 0.7134 | 0.6645 | 0.7585 | -0.0025 | -0.0173 | 0.0100 |
| logreg | baseline | continued | 2 | 0.7098 | 0.6586 | 0.7555 | 0.0004 | -0.0187 | 0.0179 |
| logreg | cfloss | baseline | 0 | 0.6994 | 0.6508 | 0.7441 | -0.0175 | -0.0374 | 0.0006 |
| logreg | cfloss | baseline | 1 | 0.7177 | 0.6681 | 0.7630 | 0.0044 | -0.0056 | 0.0139 |
| logreg | cfloss | baseline | 2 | 0.7186 | 0.6700 | 0.7641 | 0.0088 | -0.0056 | 0.0257 |
| logreg | cfloss | continued | 0 | 0.6994 | 0.6508 | 0.7441 | -0.0094 | -0.0259 | 0.0050 |
| logreg | cfloss | continued | 1 | 0.7177 | 0.6681 | 0.7630 | 0.0018 | -0.0161 | 0.0151 |
| logreg | cfloss | continued | 2 | 0.7186 | 0.6700 | 0.7641 | 0.0092 | -0.0062 | 0.0247 |
| logreg | continued | baseline | 0 | 0.7089 | 0.6597 | 0.7534 | -0.0081 | -0.0206 | 0.0035 |
| logreg | continued | baseline | 1 | 0.7159 | 0.6663 | 0.7612 | 0.0025 | -0.0100 | 0.0173 |
| logreg | continued | baseline | 2 | 0.7094 | 0.6615 | 0.7532 | -0.0004 | -0.0179 | 0.0187 |
| logreg | gt | baseline | - | 0.7183 | 0.6736 | 0.7630 |  |  |  |
| logreg | reliseg | baseline | 0 | 0.7203 | 0.6719 | 0.7657 | 0.0033 | -0.0139 | 0.0185 |
| logreg | reliseg | baseline | 1 | 0.7308 | 0.6842 | 0.7751 | 0.0174 | 0.0042 | 0.0323 |
| logreg | reliseg | baseline | 2 | 0.7174 | 0.6716 | 0.7608 | 0.0075 | -0.0118 | 0.0279 |
| logreg | reliseg | continued | 0 | 0.7203 | 0.6719 | 0.7657 | 0.0114 | -0.0050 | 0.0268 |
| logreg | reliseg | continued | 1 | 0.7308 | 0.6842 | 0.7751 | 0.0149 | 0.0035 | 0.0267 |
| logreg | reliseg | continued | 2 | 0.7174 | 0.6716 | 0.7608 | 0.0080 | -0.0085 | 0.0239 |
| logreg | reliseg_nocl | baseline | 0 | 0.7184 | 0.6685 | 0.7644 | 0.0015 | -0.0094 | 0.0119 |
| logreg | reliseg_nocl | baseline | 1 | 0.7111 | 0.6630 | 0.7562 | -0.0023 | -0.0115 | 0.0073 |
| logreg | reliseg_nocl | baseline | 2 | 0.7239 | 0.6778 | 0.7671 | 0.0141 | -0.0025 | 0.0331 |
| logreg | reliseg_nocl | continued | 0 | 0.7184 | 0.6685 | 0.7644 | 0.0095 | -0.0046 | 0.0243 |
| logreg | reliseg_nocl | continued | 1 | 0.7111 | 0.6630 | 0.7562 | -0.0048 | -0.0166 | 0.0061 |
| logreg | reliseg_nocl | continued | 2 | 0.7239 | 0.6778 | 0.7671 | 0.0145 | 0.0018 | 0.0283 |

## 5. Zero-shot cross-dataset fidelity (CHASE_DB1 / STARE)
**These two sets are the only masked evidence outside the development set**: neither model ever saw them, and no probe touched them. They are therefore reported seed by seed rather than pooled -- with n this small, a seed mean hides more than it summarises.

Sources: `results/pivot/e2_zeroshot.csv` (fidelity + delta-r against both references), `results/pivot/e2_zeroshot_delta.csv` (every delta row incl. the seed-pooled ones), `results/pivot/e2_zeroshot_delta_pooled.csv`. GT biomarkers: `results/pivot/e2_gt_chasedb1.csv` (n=8) and `results/pivot/e2_gt_stare.csv` (n=10), written by `p5_eval gtbio` on the same estimator path. Inference runs at the *source* model's scale (longest side 1536, patch 768).

**Caveat that governs every number here: n = 8 (CHASE_DB1) and n = 10 (STARE) test images.** A Pearson r on 8 points has a 95 %% interval roughly +-0.5 wide, and STARE has no Gate A train scale, so its sigma is the robust scale of its own GT biomarkers (stated in the `sigma_source` column). Read these as directional evidence, not as estimates.

### chasedb1 -- absolute fidelity, seed by seed

| dataset | config | seed | biomarker | n | r_pearson | r_spearman | bias_const_sigma | resid_sd_sigma |
|---|---|---|---|---|---|---|---|---|
| fives | baseline | 0 | FD | 8 | 0.889 | 0.833 | -0.379 | 0.423 |
| fives | baseline | 1 | FD | 8 | 0.948 | 0.976 | -0.268 | 0.300 |
| fives | baseline | 2 | FD | 8 | 0.867 | 0.952 | -0.537 | 0.448 |
| fives | cfloss | 0 | FD | 8 | 0.906 | 0.952 | -0.379 | 0.388 |
| fives | cfloss | 1 | FD | 8 | 0.958 | 0.976 | -0.321 | 0.252 |
| fives | cfloss | 2 | FD | 8 | 0.882 | 0.952 | -0.353 | 0.448 |
| fives | continued | 0 | FD | 8 | 0.871 | 0.857 | -0.421 | 0.453 |
| fives | continued | 1 | FD | 8 | 0.946 | 0.976 | -0.406 | 0.295 |
| fives | continued | 2 | FD | 8 | 0.886 | 0.952 | -0.458 | 0.413 |
| fives | reliseg | 0 | FD | 8 | 0.876 | 0.833 | -0.316 | 0.460 |
| fives | reliseg | 1 | FD | 8 | 0.949 | 0.976 | -0.289 | 0.296 |
| fives | reliseg | 2 | FD | 8 | 0.886 | 0.952 | -0.395 | 0.415 |
| fives | reliseg_nocl | 0 | FD | 8 | 0.898 | 0.833 | -0.279 | 0.386 |
| fives | reliseg_nocl | 1 | FD | 8 | 0.903 | 0.952 | -0.207 | 0.380 |
| fives | reliseg_nocl | 2 | FD | 8 | 0.859 | 0.833 | -0.287 | 0.517 |
| fives | baseline | 0 | density | 8 | 0.904 | 0.952 | 5.084 | 0.904 |
| fives | baseline | 1 | density | 8 | 0.926 | 0.976 | 5.278 | 0.896 |
| fives | baseline | 2 | density | 8 | 0.914 | 0.976 | 4.794 | 0.888 |
| fives | cfloss | 0 | density | 8 | 0.914 | 0.976 | 4.825 | 0.907 |
| fives | cfloss | 1 | density | 8 | 0.930 | 1.000 | 5.082 | 0.874 |
| fives | cfloss | 2 | density | 8 | 0.920 | 0.929 | 4.888 | 0.961 |
| fives | continued | 0 | density | 8 | 0.911 | 0.952 | 5.259 | 0.895 |
| fives | continued | 1 | density | 8 | 0.927 | 0.976 | 5.212 | 0.864 |
| fives | continued | 2 | density | 8 | 0.924 | 0.929 | 5.284 | 0.932 |
| fives | reliseg | 0 | density | 8 | 0.907 | 0.952 | 5.325 | 0.900 |
| fives | reliseg | 1 | density | 8 | 0.928 | 1.000 | 5.413 | 0.872 |
| fives | reliseg | 2 | density | 8 | 0.922 | 0.929 | 5.198 | 0.900 |
| fives | reliseg_nocl | 0 | density | 8 | 0.908 | 0.881 | 5.197 | 0.896 |
| fives | reliseg_nocl | 1 | density | 8 | 0.912 | 0.976 | 5.401 | 0.915 |
| fives | reliseg_nocl | 2 | density | 8 | 0.917 | 0.976 | 5.257 | 0.936 |
| fives | baseline | 0 | tortuosity | 8 | 0.186 | 0.524 | -1.318 | 2.184 |
| fives | baseline | 1 | tortuosity | 8 | 0.213 | 0.571 | -1.332 | 2.170 |
| fives | baseline | 2 | tortuosity | 8 | 0.215 | 0.524 | -1.344 | 2.125 |
| fives | cfloss | 0 | tortuosity | 8 | 0.804 | 0.810 | -1.314 | 1.463 |
| fives | cfloss | 1 | tortuosity | 8 | 0.169 | 0.548 | -1.492 | 2.186 |
| fives | cfloss | 2 | tortuosity | 8 | 0.090 | 0.595 | -1.255 | 2.256 |
| fives | continued | 0 | tortuosity | 8 | 0.153 | 0.524 | -1.249 | 2.217 |
| fives | continued | 1 | tortuosity | 8 | 0.174 | 0.429 | -1.356 | 2.156 |
| fives | continued | 2 | tortuosity | 8 | 0.365 | 0.643 | -1.418 | 1.991 |
| fives | reliseg | 0 | tortuosity | 8 | 0.271 | 0.619 | -1.358 | 2.125 |
| fives | reliseg | 1 | tortuosity | 8 | 0.119 | 0.452 | -1.099 | 2.249 |
| fives | reliseg | 2 | tortuosity | 8 | 0.205 | 0.381 | -1.302 | 2.161 |
| fives | reliseg_nocl | 0 | tortuosity | 8 | 0.852 | 0.833 | -1.480 | 1.375 |
| fives | reliseg_nocl | 1 | tortuosity | 8 | 0.275 | 0.571 | -1.561 | 2.054 |
| fives | reliseg_nocl | 2 | tortuosity | 8 | 0.226 | 0.405 | -1.435 | 2.119 |
| fives | baseline | 0 | total_length | 8 | 0.907 | 0.881 | 0.046 | 0.498 |
| fives | baseline | 1 | total_length | 8 | 0.931 | 0.929 | 0.101 | 0.429 |
| fives | baseline | 2 | total_length | 8 | 0.906 | 0.881 | -0.033 | 0.508 |
| fives | cfloss | 0 | total_length | 8 | 0.918 | 0.905 | -0.016 | 0.466 |
| fives | cfloss | 1 | total_length | 8 | 0.939 | 0.929 | 0.039 | 0.405 |
| fives | cfloss | 2 | total_length | 8 | 0.902 | 0.881 | 0.013 | 0.505 |
| fives | continued | 0 | total_length | 8 | 0.891 | 0.881 | -0.038 | 0.534 |
| fives | continued | 1 | total_length | 8 | 0.935 | 0.976 | -0.062 | 0.423 |
| fives | continued | 2 | total_length | 8 | 0.900 | 0.881 | -0.012 | 0.515 |
| fives | reliseg | 0 | total_length | 8 | 0.905 | 0.810 | 0.141 | 0.499 |
| fives | reliseg | 1 | total_length | 8 | 0.935 | 0.976 | 0.100 | 0.419 |
| fives | reliseg | 2 | total_length | 8 | 0.927 | 0.976 | 0.061 | 0.459 |
| fives | reliseg_nocl | 0 | total_length | 8 | 0.916 | 0.905 | 0.122 | 0.477 |
| fives | reliseg_nocl | 1 | total_length | 8 | 0.918 | 0.905 | 0.257 | 0.466 |
| fives | reliseg_nocl | 2 | total_length | 8 | 0.891 | 0.810 | 0.241 | 0.533 |
| hrf | baseline | 0 | FD | 8 | -0.012 | 0.262 | 2.098 | 2.146 |
| hrf | baseline | 1 | FD | 8 | 0.365 | 0.357 | 0.093 | 1.490 |
| hrf | baseline | 2 | FD | 8 | -0.131 | -0.071 | 1.308 | 2.149 |
| hrf | cfloss | 0 | FD | 8 | 0.104 | 0.333 | 0.426 | 1.989 |
| hrf | cfloss | 1 | FD | 8 | 0.236 | 0.214 | -0.395 | 2.052 |
| hrf | cfloss | 2 | FD | 8 | -0.081 | -0.095 | 0.840 | 2.009 |
| hrf | continued | 0 | FD | 8 | 0.122 | 0.333 | 0.833 | 1.753 |
| hrf | continued | 1 | FD | 8 | 0.383 | 0.381 | 0.575 | 1.497 |
| hrf | continued | 2 | FD | 8 | -0.053 | 0.000 | 1.117 | 1.916 |
| hrf | reliseg | 0 | FD | 8 | 0.060 | 0.095 | 3.185 | 1.769 |
| hrf | reliseg | 1 | FD | 8 | 0.183 | 0.429 | 3.032 | 1.763 |
| hrf | reliseg | 2 | FD | 8 | 0.009 | 0.048 | 4.311 | 1.539 |
| hrf | reliseg_nocl | 0 | FD | 8 | 0.054 | 0.167 | 3.766 | 1.875 |
| hrf | reliseg_nocl | 1 | FD | 8 | 0.036 | 0.214 | 2.008 | 2.112 |
| hrf | reliseg_nocl | 2 | FD | 8 | 0.052 | 0.000 | 3.523 | 1.572 |
| hrf | baseline | 0 | density | 8 | 0.319 | 0.524 | -0.305 | 2.760 |
| hrf | baseline | 1 | density | 8 | 0.583 | 0.619 | -3.568 | 1.744 |
| hrf | baseline | 2 | density | 8 | 0.112 | 0.167 | -3.005 | 2.706 |
| hrf | cfloss | 0 | density | 8 | 0.416 | 0.452 | -2.081 | 2.345 |
| hrf | cfloss | 1 | density | 8 | 0.460 | 0.405 | -4.404 | 2.320 |
| hrf | cfloss | 2 | density | 8 | 0.040 | 0.048 | -2.971 | 2.748 |
| hrf | continued | 0 | density | 8 | 0.411 | 0.405 | -1.369 | 2.397 |
| hrf | continued | 1 | density | 8 | 0.553 | 0.524 | -2.442 | 2.006 |
| hrf | continued | 2 | density | 8 | 0.157 | 0.190 | -2.029 | 2.520 |
| hrf | reliseg | 0 | density | 8 | 0.466 | 0.595 | -0.948 | 2.220 |
| hrf | reliseg | 1 | density | 8 | 0.549 | 0.643 | -1.634 | 2.084 |
| hrf | reliseg | 2 | density | 8 | 0.075 | 0.167 | -1.091 | 2.696 |
| hrf | reliseg_nocl | 0 | density | 8 | 0.426 | 0.452 | -0.692 | 2.337 |
| hrf | reliseg_nocl | 1 | density | 8 | 0.406 | 0.405 | -2.881 | 2.384 |
| hrf | reliseg_nocl | 2 | density | 8 | 0.123 | 0.190 | -2.231 | 2.612 |
| hrf | baseline | 0 | tortuosity | 8 | -0.017 | 0.214 | -2.388 | 2.152 |
| hrf | baseline | 1 | tortuosity | 8 | 0.004 | -0.095 | -2.472 | 2.175 |
| hrf | baseline | 2 | tortuosity | 8 | -0.247 | -0.238 | -2.432 | 2.363 |
| hrf | cfloss | 0 | tortuosity | 8 | -0.415 | -0.357 | -2.579 | 2.320 |
| hrf | cfloss | 1 | tortuosity | 8 | 0.077 | 0.048 | -2.388 | 2.158 |
| hrf | cfloss | 2 | tortuosity | 8 | -0.072 | -0.119 | -2.415 | 2.229 |
| hrf | continued | 0 | tortuosity | 8 | -0.152 | 0.143 | -2.418 | 2.223 |
| hrf | continued | 1 | tortuosity | 8 | 0.098 | -0.048 | -2.253 | 2.150 |
| hrf | continued | 2 | tortuosity | 8 | 0.086 | 0.333 | -2.205 | 2.152 |
| hrf | reliseg | 0 | tortuosity | 8 | -0.133 | 0.262 | -2.389 | 2.308 |
| hrf | reliseg | 1 | tortuosity | 8 | -0.034 | 0.333 | -2.525 | 2.202 |
| hrf | reliseg | 2 | tortuosity | 8 | -0.069 | 0.143 | -2.527 | 2.267 |
| hrf | reliseg_nocl | 0 | tortuosity | 8 | -0.194 | 0.119 | -2.604 | 2.235 |
| hrf | reliseg_nocl | 1 | tortuosity | 8 | -0.037 | 0.238 | -2.517 | 2.196 |
| hrf | reliseg_nocl | 2 | tortuosity | 8 | -0.187 | -0.190 | -2.663 | 2.282 |
| hrf | baseline | 0 | total_length | 8 | 0.197 | 0.357 | 3.172 | 2.096 |
| hrf | baseline | 1 | total_length | 8 | 0.528 | 0.429 | 0.909 | 1.432 |
| hrf | baseline | 2 | total_length | 8 | 0.179 | 0.214 | 2.303 | 1.997 |
| hrf | cfloss | 0 | total_length | 8 | 0.318 | 0.381 | 1.571 | 1.883 |
| hrf | cfloss | 1 | total_length | 8 | 0.412 | 0.381 | 0.503 | 1.963 |
| hrf | cfloss | 2 | total_length | 8 | 0.272 | 0.238 | 1.747 | 1.772 |
| hrf | continued | 0 | total_length | 8 | 0.344 | 0.381 | 1.969 | 1.736 |
| hrf | continued | 1 | total_length | 8 | 0.514 | 0.357 | 1.481 | 1.555 |
| hrf | continued | 2 | total_length | 8 | 0.335 | 0.310 | 2.040 | 1.582 |
| hrf | reliseg | 0 | total_length | 8 | 0.234 | 0.214 | 3.958 | 2.014 |
| hrf | reliseg | 1 | total_length | 8 | 0.340 | 0.357 | 3.786 | 2.010 |
| hrf | reliseg | 2 | total_length | 8 | 0.333 | 0.262 | 5.052 | 1.648 |
| hrf | reliseg_nocl | 0 | total_length | 8 | 0.250 | 0.357 | 4.072 | 2.022 |
| hrf | reliseg_nocl | 1 | total_length | 8 | 0.309 | 0.262 | 2.511 | 2.037 |
| hrf | reliseg_nocl | 2 | total_length | 8 | 0.339 | 0.190 | 3.735 | 1.751 |

#### chasedb1 -- delta-r vs `baseline` (paired bootstrap over the same images, 2000 resamples)

| dataset | config | seed | biomarker | d_r_vs_baseline | d_r_vs_baseline_lo | d_r_vs_baseline_hi |
|---|---|---|---|---|---|---|
| fives | cfloss | 0 | FD | 0.017 | -0.085 | 0.169 |
| fives | cfloss | 1 | FD | 0.010 | -0.009 | 0.026 |
| fives | cfloss | 2 | FD | 0.015 | -0.139 | 0.113 |
| fives | continued | 0 | FD | -0.018 | -0.111 | 0.046 |
| fives | continued | 1 | FD | -0.002 | -0.037 | 0.047 |
| fives | continued | 2 | FD | 0.020 | -0.104 | 0.127 |
| fives | reliseg | 0 | FD | -0.013 | -0.182 | 0.098 |
| fives | reliseg | 1 | FD | 0.001 | -0.011 | 0.031 |
| fives | reliseg | 2 | FD | 0.019 | -0.021 | 0.075 |
| fives | reliseg_nocl | 0 | FD | 0.009 | -0.076 | 0.123 |
| fives | reliseg_nocl | 1 | FD | -0.045 | -0.163 | -0.004 |
| fives | reliseg_nocl | 2 | FD | -0.008 | -0.260 | 0.124 |
| fives | cfloss | 0 | density | 0.010 | -0.013 | 0.047 |
| fives | cfloss | 1 | density | 0.004 | -0.008 | 0.027 |
| fives | cfloss | 2 | density | 0.006 | -0.046 | 0.036 |
| fives | continued | 0 | density | 0.007 | -0.000 | 0.023 |
| fives | continued | 1 | density | 0.001 | -0.004 | 0.015 |
| fives | continued | 2 | density | 0.011 | -0.033 | 0.045 |
| fives | reliseg | 0 | density | 0.003 | -0.010 | 0.018 |
| fives | reliseg | 1 | density | 0.001 | -0.004 | 0.022 |
| fives | reliseg | 2 | density | 0.009 | -0.029 | 0.023 |
| fives | reliseg_nocl | 0 | density | 0.004 | -0.027 | 0.031 |
| fives | reliseg_nocl | 1 | density | -0.014 | -0.044 | 0.004 |
| fives | reliseg_nocl | 2 | density | 0.004 | -0.013 | 0.010 |
| fives | cfloss | 0 | tortuosity | 0.618 | -0.243 | 1.188 |
| fives | cfloss | 1 | tortuosity | -0.044 | -0.365 | 0.358 |
| fives | cfloss | 2 | tortuosity | -0.125 | -0.338 | 0.221 |
| fives | continued | 0 | tortuosity | -0.033 | -0.271 | 0.120 |
| fives | continued | 1 | tortuosity | -0.039 | -0.207 | 0.105 |
| fives | continued | 2 | tortuosity | 0.149 | -0.037 | 0.569 |
| fives | reliseg | 0 | tortuosity | 0.085 | -0.027 | 0.220 |
| fives | reliseg | 1 | tortuosity | -0.094 | -0.330 | 0.234 |
| fives | reliseg | 2 | tortuosity | -0.010 | -0.127 | 0.097 |
| fives | reliseg_nocl | 0 | tortuosity | 0.666 | -0.226 | 1.223 |
| fives | reliseg_nocl | 1 | tortuosity | 0.062 | -0.301 | 0.343 |
| fives | reliseg_nocl | 2 | tortuosity | 0.010 | -0.117 | 0.115 |
| fives | cfloss | 0 | total_length | 0.011 | -0.014 | 0.097 |
| fives | cfloss | 1 | total_length | 0.008 | -0.010 | 0.092 |
| fives | cfloss | 2 | total_length | -0.004 | -0.043 | 0.025 |
| fives | continued | 0 | total_length | -0.015 | -0.097 | 0.005 |
| fives | continued | 1 | total_length | 0.004 | -0.007 | 0.058 |
| fives | continued | 2 | total_length | -0.006 | -0.015 | 0.023 |
| fives | reliseg | 0 | total_length | -0.001 | -0.056 | 0.014 |
| fives | reliseg | 1 | total_length | 0.004 | -0.010 | 0.049 |
| fives | reliseg | 2 | total_length | 0.020 | 0.002 | 0.115 |
| fives | reliseg_nocl | 0 | total_length | 0.009 | -0.030 | 0.073 |
| fives | reliseg_nocl | 1 | total_length | -0.013 | -0.117 | 0.021 |
| fives | reliseg_nocl | 2 | total_length | -0.015 | -0.160 | 0.009 |
| hrf | cfloss | 0 | FD | 0.115 | -0.010 | 0.288 |
| hrf | cfloss | 1 | FD | -0.129 | -0.338 | 0.035 |
| hrf | cfloss | 2 | FD | 0.051 | -0.060 | 0.200 |
| hrf | continued | 0 | FD | 0.134 | -0.009 | 0.338 |
| hrf | continued | 1 | FD | 0.018 | -0.076 | 0.118 |
| hrf | continued | 2 | FD | 0.078 | -0.003 | 0.233 |
| hrf | reliseg | 0 | FD | 0.072 | -0.225 | 0.251 |
| hrf | reliseg | 1 | FD | -0.183 | -0.405 | 0.155 |
| hrf | reliseg | 2 | FD | 0.140 | -0.184 | 0.584 |
| hrf | reliseg_nocl | 0 | FD | 0.065 | -0.181 | 0.249 |
| hrf | reliseg_nocl | 1 | FD | -0.330 | -0.585 | -0.008 |
| hrf | reliseg_nocl | 2 | FD | 0.183 | -0.103 | 0.466 |
| hrf | cfloss | 0 | density | 0.098 | -0.043 | 0.198 |
| hrf | cfloss | 1 | density | -0.123 | -0.332 | 0.065 |
| hrf | cfloss | 2 | density | -0.071 | -0.296 | 0.202 |
| hrf | continued | 0 | density | 0.092 | -0.063 | 0.173 |
| hrf | continued | 1 | density | -0.031 | -0.172 | 0.090 |
| hrf | continued | 2 | density | 0.045 | -0.182 | 0.172 |
| hrf | reliseg | 0 | density | 0.147 | -0.037 | 0.272 |
| hrf | reliseg | 1 | density | -0.034 | -0.226 | 0.169 |
| hrf | reliseg | 2 | density | -0.037 | -0.317 | 0.278 |
| hrf | reliseg_nocl | 0 | density | 0.107 | -0.061 | 0.242 |
| hrf | reliseg_nocl | 1 | density | -0.177 | -0.405 | 0.056 |
| hrf | reliseg_nocl | 2 | density | 0.012 | -0.228 | 0.223 |
| hrf | cfloss | 0 | tortuosity | -0.398 | -1.166 | 0.233 |
| hrf | cfloss | 1 | tortuosity | 0.072 | -0.122 | 0.586 |
| hrf | cfloss | 2 | tortuosity | 0.175 | -0.046 | 0.472 |
| hrf | continued | 0 | tortuosity | -0.135 | -0.469 | 0.290 |
| hrf | continued | 1 | tortuosity | 0.093 | -0.247 | 0.575 |
| hrf | continued | 2 | tortuosity | 0.333 | -0.171 | 1.185 |
| hrf | reliseg | 0 | tortuosity | -0.116 | -0.555 | 0.791 |
| hrf | reliseg | 1 | tortuosity | -0.039 | -0.700 | 0.975 |
| hrf | reliseg | 2 | tortuosity | 0.178 | -0.105 | 0.653 |
| hrf | reliseg_nocl | 0 | tortuosity | -0.178 | -0.881 | 0.439 |
| hrf | reliseg_nocl | 1 | tortuosity | -0.041 | -0.614 | 0.992 |
| hrf | reliseg_nocl | 2 | tortuosity | 0.060 | -0.060 | 0.283 |
| hrf | cfloss | 0 | total_length | 0.121 | -0.021 | 0.196 |
| hrf | cfloss | 1 | total_length | -0.116 | -0.344 | 0.026 |
| hrf | cfloss | 2 | total_length | 0.093 | -0.069 | 0.148 |
| hrf | continued | 0 | total_length | 0.147 | 0.011 | 0.212 |
| hrf | continued | 1 | total_length | -0.014 | -0.173 | 0.099 |
| hrf | continued | 2 | total_length | 0.156 | 0.006 | 0.287 |
| hrf | reliseg | 0 | total_length | 0.038 | -0.268 | 0.208 |
| hrf | reliseg | 1 | total_length | -0.188 | -0.490 | 0.114 |
| hrf | reliseg | 2 | total_length | 0.154 | -0.314 | 0.460 |
| hrf | reliseg_nocl | 0 | total_length | 0.054 | -0.186 | 0.201 |
| hrf | reliseg_nocl | 1 | total_length | -0.220 | -0.471 | 0.015 |
| hrf | reliseg_nocl | 2 | total_length | 0.160 | -0.051 | 0.254 |

#### chasedb1 -- delta-r vs `continued` (paired bootstrap over the same images, 2000 resamples)

| dataset | config | seed | biomarker | d_r_vs_continued | d_r_vs_continued_lo | d_r_vs_continued_hi |
|---|---|---|---|---|---|---|
| fives | cfloss | 0 | FD | 0.035 | -0.001 | 0.176 |
| fives | cfloss | 1 | FD | 0.012 | -0.024 | 0.042 |
| fives | cfloss | 2 | FD | -0.005 | -0.067 | 0.033 |
| fives | reliseg | 0 | FD | 0.005 | -0.080 | 0.064 |
| fives | reliseg | 1 | FD | 0.003 | -0.019 | 0.032 |
| fives | reliseg | 2 | FD | -0.000 | -0.089 | 0.159 |
| fives | reliseg_nocl | 0 | FD | 0.028 | -0.028 | 0.111 |
| fives | reliseg_nocl | 1 | FD | -0.042 | -0.210 | 0.018 |
| fives | reliseg_nocl | 2 | FD | -0.028 | -0.219 | 0.108 |
| fives | cfloss | 0 | density | 0.003 | -0.023 | 0.029 |
| fives | cfloss | 1 | density | 0.003 | -0.009 | 0.020 |
| fives | cfloss | 2 | density | -0.005 | -0.018 | -0.000 |
| fives | reliseg | 0 | density | -0.004 | -0.026 | 0.002 |
| fives | reliseg | 1 | density | 0.000 | -0.004 | 0.011 |
| fives | reliseg | 2 | density | -0.002 | -0.032 | 0.014 |
| fives | reliseg_nocl | 0 | density | -0.003 | -0.037 | 0.014 |
| fives | reliseg_nocl | 1 | density | -0.015 | -0.053 | 0.002 |
| fives | reliseg_nocl | 2 | density | -0.007 | -0.049 | 0.031 |
| fives | cfloss | 0 | tortuosity | 0.651 | -0.211 | 1.165 |
| fives | cfloss | 1 | tortuosity | -0.005 | -0.354 | 0.413 |
| fives | cfloss | 2 | tortuosity | -0.274 | -0.645 | 0.078 |
| fives | reliseg | 0 | tortuosity | 0.118 | -0.009 | 0.342 |
| fives | reliseg | 1 | tortuosity | -0.055 | -0.261 | 0.209 |
| fives | reliseg | 2 | tortuosity | -0.159 | -0.611 | 0.045 |
| fives | reliseg_nocl | 0 | tortuosity | 0.699 | -0.205 | 1.198 |
| fives | reliseg_nocl | 1 | tortuosity | 0.101 | -0.298 | 0.401 |
| fives | reliseg_nocl | 2 | tortuosity | -0.139 | -0.518 | 0.001 |
| fives | cfloss | 0 | total_length | 0.027 | 0.001 | 0.174 |
| fives | cfloss | 1 | total_length | 0.005 | -0.024 | 0.053 |
| fives | cfloss | 2 | total_length | 0.002 | -0.050 | 0.019 |
| fives | reliseg | 0 | total_length | 0.014 | -0.013 | 0.069 |
| fives | reliseg | 1 | total_length | -0.000 | -0.030 | 0.011 |
| fives | reliseg | 2 | total_length | 0.027 | 0.002 | 0.107 |
| fives | reliseg_nocl | 0 | total_length | 0.025 | -0.002 | 0.140 |
| fives | reliseg_nocl | 1 | total_length | -0.017 | -0.173 | 0.020 |
| fives | reliseg_nocl | 2 | total_length | -0.009 | -0.183 | 0.020 |
| hrf | cfloss | 0 | FD | -0.018 | -0.103 | 0.046 |
| hrf | cfloss | 1 | FD | -0.147 | -0.324 | 0.026 |
| hrf | cfloss | 2 | FD | -0.028 | -0.163 | 0.107 |
| hrf | reliseg | 0 | FD | -0.062 | -0.420 | 0.147 |
| hrf | reliseg | 1 | FD | -0.201 | -0.357 | 0.081 |
| hrf | reliseg | 2 | FD | 0.061 | -0.270 | 0.458 |
| hrf | reliseg_nocl | 0 | FD | -0.069 | -0.402 | 0.181 |
| hrf | reliseg_nocl | 1 | FD | -0.348 | -0.575 | -0.022 |
| hrf | reliseg_nocl | 2 | FD | 0.105 | -0.156 | 0.340 |
| hrf | cfloss | 0 | density | 0.005 | -0.033 | 0.078 |
| hrf | cfloss | 1 | density | -0.093 | -0.186 | -0.006 |
| hrf | cfloss | 2 | density | -0.117 | -0.266 | 0.152 |
| hrf | reliseg | 0 | density | 0.055 | -0.094 | 0.189 |
| hrf | reliseg | 1 | density | -0.004 | -0.103 | 0.127 |
| hrf | reliseg | 2 | density | -0.082 | -0.263 | 0.280 |
| hrf | reliseg_nocl | 0 | density | 0.015 | -0.094 | 0.137 |
| hrf | reliseg_nocl | 1 | density | -0.146 | -0.255 | -0.014 |
| hrf | reliseg_nocl | 2 | density | -0.034 | -0.160 | 0.225 |
| hrf | cfloss | 0 | tortuosity | -0.263 | -0.766 | 0.115 |
| hrf | cfloss | 1 | tortuosity | -0.021 | -0.383 | 0.233 |
| hrf | cfloss | 2 | tortuosity | -0.158 | -0.932 | 0.224 |
| hrf | reliseg | 0 | tortuosity | 0.019 | -0.280 | 0.866 |
| hrf | reliseg | 1 | tortuosity | -0.132 | -0.803 | 0.700 |
| hrf | reliseg | 2 | tortuosity | -0.155 | -0.683 | 0.220 |
| hrf | reliseg_nocl | 0 | tortuosity | -0.043 | -0.490 | 0.262 |
| hrf | reliseg_nocl | 1 | tortuosity | -0.135 | -0.732 | 0.692 |
| hrf | reliseg_nocl | 2 | tortuosity | -0.273 | -1.124 | 0.316 |
| hrf | cfloss | 0 | total_length | -0.026 | -0.077 | 0.023 |
| hrf | cfloss | 1 | total_length | -0.102 | -0.309 | 0.064 |
| hrf | cfloss | 2 | total_length | -0.063 | -0.252 | 0.045 |
| hrf | reliseg | 0 | total_length | -0.109 | -0.434 | 0.134 |
| hrf | reliseg | 1 | total_length | -0.174 | -0.358 | 0.032 |
| hrf | reliseg | 2 | total_length | -0.001 | -0.387 | 0.300 |
| hrf | reliseg_nocl | 0 | total_length | -0.093 | -0.325 | 0.105 |
| hrf | reliseg_nocl | 1 | total_length | -0.206 | -0.431 | 0.020 |
| hrf | reliseg_nocl | 2 | total_length | 0.004 | -0.158 | 0.120 |

### stare -- absolute fidelity, seed by seed

| dataset | config | seed | biomarker | n | r_pearson | r_spearman | bias_const_sigma | resid_sd_sigma |
|---|---|---|---|---|---|---|---|---|
| fives | baseline | 0 | FD | 10 | 0.941 | 0.770 | 0.091 | 0.528 |
| fives | baseline | 1 | FD | 10 | 0.954 | 0.842 | 0.144 | 0.513 |
| fives | baseline | 2 | FD | 10 | 0.954 | 0.770 | -0.031 | 0.505 |
| fives | cfloss | 0 | FD | 10 | 0.940 | 0.842 | 0.122 | 0.542 |
| fives | cfloss | 1 | FD | 10 | 0.957 | 0.867 | 0.127 | 0.496 |
| fives | cfloss | 2 | FD | 10 | 0.957 | 0.891 | -0.001 | 0.495 |
| fives | continued | 0 | FD | 10 | 0.939 | 0.806 | 0.130 | 0.540 |
| fives | continued | 1 | FD | 10 | 0.962 | 0.855 | 0.109 | 0.471 |
| fives | continued | 2 | FD | 10 | 0.959 | 0.879 | 0.008 | 0.470 |
| fives | reliseg | 0 | FD | 10 | 0.952 | 0.830 | 0.143 | 0.480 |
| fives | reliseg | 1 | FD | 10 | 0.962 | 0.867 | 0.222 | 0.480 |
| fives | reliseg | 2 | FD | 10 | 0.951 | 0.842 | 0.082 | 0.516 |
| fives | reliseg_nocl | 0 | FD | 10 | 0.936 | 0.830 | 0.166 | 0.543 |
| fives | reliseg_nocl | 1 | FD | 10 | 0.959 | 0.818 | 0.201 | 0.477 |
| fives | reliseg_nocl | 2 | FD | 10 | 0.947 | 0.830 | 0.102 | 0.518 |
| fives | baseline | 0 | density | 10 | 0.899 | 0.867 | 0.774 | 1.038 |
| fives | baseline | 1 | density | 10 | 0.911 | 0.855 | 0.556 | 0.974 |
| fives | baseline | 2 | density | 10 | 0.915 | 0.867 | 0.501 | 0.955 |
| fives | cfloss | 0 | density | 10 | 0.886 | 0.770 | 0.764 | 1.108 |
| fives | cfloss | 1 | density | 10 | 0.911 | 0.867 | 0.483 | 0.977 |
| fives | cfloss | 2 | density | 10 | 0.916 | 0.842 | 0.508 | 0.947 |
| fives | continued | 0 | density | 10 | 0.901 | 0.867 | 0.951 | 1.027 |
| fives | continued | 1 | density | 10 | 0.910 | 0.806 | 0.619 | 0.984 |
| fives | continued | 2 | density | 10 | 0.917 | 0.830 | 0.691 | 0.946 |
| fives | reliseg | 0 | density | 10 | 0.908 | 0.867 | 0.904 | 0.991 |
| fives | reliseg | 1 | density | 10 | 0.914 | 0.867 | 0.745 | 0.958 |
| fives | reliseg | 2 | density | 10 | 0.913 | 0.830 | 0.666 | 0.966 |
| fives | reliseg_nocl | 0 | density | 10 | 0.909 | 0.867 | 0.803 | 0.986 |
| fives | reliseg_nocl | 1 | density | 10 | 0.914 | 0.855 | 0.588 | 0.961 |
| fives | reliseg_nocl | 2 | density | 10 | 0.914 | 0.879 | 0.670 | 0.959 |
| fives | baseline | 0 | tortuosity | 10 | 0.940 | 0.842 | -0.049 | 0.324 |
| fives | baseline | 1 | tortuosity | 10 | 0.964 | 0.891 | -0.086 | 0.258 |
| fives | baseline | 2 | tortuosity | 10 | 0.807 | 0.867 | 0.069 | 0.581 |
| fives | cfloss | 0 | tortuosity | 10 | 0.684 | 0.818 | 0.192 | 0.757 |
| fives | cfloss | 1 | tortuosity | 10 | 0.953 | 0.842 | -0.093 | 0.288 |
| fives | cfloss | 2 | tortuosity | 10 | 0.792 | 0.855 | 0.052 | 0.615 |
| fives | continued | 0 | tortuosity | 10 | 0.967 | 0.915 | -0.128 | 0.276 |
| fives | continued | 1 | tortuosity | 10 | 0.963 | 0.891 | -0.075 | 0.266 |
| fives | continued | 2 | tortuosity | 10 | 0.693 | 0.600 | 0.204 | 0.763 |
| fives | reliseg | 0 | tortuosity | 10 | 0.902 | 0.855 | -0.084 | 0.413 |
| fives | reliseg | 1 | tortuosity | 10 | 0.969 | 0.891 | -0.059 | 0.250 |
| fives | reliseg | 2 | tortuosity | 10 | 0.801 | 0.733 | 0.076 | 0.586 |
| fives | reliseg_nocl | 0 | tortuosity | 10 | 0.943 | 0.879 | -0.069 | 0.315 |
| fives | reliseg_nocl | 1 | tortuosity | 10 | 0.958 | 0.879 | -0.101 | 0.273 |
| fives | reliseg_nocl | 2 | tortuosity | 10 | 0.786 | 0.867 | 0.047 | 0.615 |
| fives | baseline | 0 | total_length | 10 | 0.960 | 0.903 | 0.067 | 0.467 |
| fives | baseline | 1 | total_length | 10 | 0.967 | 0.879 | 0.068 | 0.429 |
| fives | baseline | 2 | total_length | 10 | 0.967 | 0.903 | -0.084 | 0.452 |
| fives | cfloss | 0 | total_length | 10 | 0.959 | 0.903 | 0.066 | 0.456 |
| fives | cfloss | 1 | total_length | 10 | 0.966 | 0.879 | 0.059 | 0.432 |
| fives | cfloss | 2 | total_length | 10 | 0.970 | 0.879 | -0.056 | 0.426 |
| fives | continued | 0 | total_length | 10 | 0.961 | 0.903 | 0.104 | 0.453 |
| fives | continued | 1 | total_length | 10 | 0.969 | 0.879 | 0.056 | 0.416 |
| fives | continued | 2 | total_length | 10 | 0.973 | 0.879 | -0.025 | 0.409 |
| fives | reliseg | 0 | total_length | 10 | 0.968 | 0.903 | 0.130 | 0.414 |
| fives | reliseg | 1 | total_length | 10 | 0.973 | 0.891 | 0.165 | 0.401 |
| fives | reliseg | 2 | total_length | 10 | 0.968 | 0.903 | 0.034 | 0.436 |
| fives | reliseg_nocl | 0 | total_length | 10 | 0.961 | 0.903 | 0.149 | 0.448 |
| fives | reliseg_nocl | 1 | total_length | 10 | 0.968 | 0.891 | 0.129 | 0.415 |
| fives | reliseg_nocl | 2 | total_length | 10 | 0.965 | 0.879 | 0.058 | 0.444 |
| hrf | baseline | 0 | FD | 10 | 0.930 | 0.830 | 0.626 | 0.620 |
| hrf | baseline | 1 | FD | 10 | 0.929 | 0.842 | 0.483 | 0.670 |
| hrf | baseline | 2 | FD | 10 | 0.927 | 0.842 | 0.407 | 0.674 |
| hrf | cfloss | 0 | FD | 10 | 0.936 | 0.842 | 0.291 | 0.616 |
| hrf | cfloss | 1 | FD | 10 | 0.920 | 0.842 | 0.293 | 0.741 |
| hrf | cfloss | 2 | FD | 10 | 0.921 | 0.842 | 0.186 | 0.664 |
| hrf | continued | 0 | FD | 10 | 0.946 | 0.855 | 0.582 | 0.619 |
| hrf | continued | 1 | FD | 10 | 0.905 | 0.842 | 0.874 | 0.697 |
| hrf | continued | 2 | FD | 10 | 0.926 | 0.842 | 0.632 | 0.648 |
| hrf | reliseg | 0 | FD | 10 | 0.909 | 0.842 | 1.393 | 0.659 |
| hrf | reliseg | 1 | FD | 10 | 0.860 | 0.867 | 1.587 | 0.816 |
| hrf | reliseg | 2 | FD | 10 | 0.866 | 0.818 | 1.511 | 0.768 |
| hrf | reliseg_nocl | 0 | FD | 10 | 0.909 | 0.855 | 1.400 | 0.676 |
| hrf | reliseg_nocl | 1 | FD | 10 | 0.921 | 0.867 | 1.334 | 0.678 |
| hrf | reliseg_nocl | 2 | FD | 10 | 0.915 | 0.842 | 1.141 | 0.645 |
| hrf | baseline | 0 | density | 10 | 0.708 | 0.794 | -1.576 | 1.862 |
| hrf | baseline | 1 | density | 10 | 0.725 | 0.673 | -1.705 | 1.881 |
| hrf | baseline | 2 | density | 10 | 0.671 | 0.673 | -1.776 | 1.919 |
| hrf | cfloss | 0 | density | 10 | 0.746 | 0.745 | -2.342 | 1.828 |
| hrf | cfloss | 1 | density | 10 | 0.686 | 0.673 | -2.379 | 2.157 |
| hrf | cfloss | 2 | density | 10 | 0.658 | 0.673 | -2.439 | 2.020 |
| hrf | continued | 0 | density | 10 | 0.781 | 0.758 | -1.667 | 1.712 |
| hrf | continued | 1 | density | 10 | 0.680 | 0.673 | -1.236 | 2.014 |
| hrf | continued | 2 | density | 10 | 0.685 | 0.624 | -1.469 | 1.927 |
| hrf | reliseg | 0 | density | 10 | 0.751 | 0.745 | -1.546 | 1.762 |
| hrf | reliseg | 1 | density | 10 | 0.677 | 0.673 | -1.246 | 2.053 |
| hrf | reliseg | 2 | density | 10 | 0.639 | 0.624 | -1.573 | 2.020 |
| hrf | reliseg_nocl | 0 | density | 10 | 0.749 | 0.745 | -1.603 | 1.821 |
| hrf | reliseg_nocl | 1 | density | 10 | 0.695 | 0.673 | -1.558 | 2.070 |
| hrf | reliseg_nocl | 2 | density | 10 | 0.626 | 0.624 | -1.787 | 2.076 |
| hrf | baseline | 0 | tortuosity | 10 | 0.862 | 0.818 | -0.418 | 0.554 |
| hrf | baseline | 1 | tortuosity | 10 | 0.781 | 0.794 | -0.483 | 0.621 |
| hrf | baseline | 2 | tortuosity | 10 | 0.893 | 0.867 | -0.436 | 0.515 |
| hrf | cfloss | 0 | tortuosity | 10 | 0.871 | 0.952 | -0.417 | 0.514 |
| hrf | cfloss | 1 | tortuosity | 10 | 0.571 | 0.685 | -0.504 | 0.775 |
| hrf | cfloss | 2 | tortuosity | 10 | 0.887 | 0.915 | -0.458 | 0.506 |
| hrf | continued | 0 | tortuosity | 10 | 0.826 | 0.927 | -0.444 | 0.562 |
| hrf | continued | 1 | tortuosity | 10 | 0.582 | 0.648 | -0.367 | 0.767 |
| hrf | continued | 2 | tortuosity | 10 | 0.945 | 0.964 | -0.382 | 0.429 |
| hrf | reliseg | 0 | tortuosity | 10 | 0.672 | 0.867 | -0.536 | 0.699 |
| hrf | reliseg | 1 | tortuosity | 10 | 0.659 | 0.588 | -0.412 | 0.719 |
| hrf | reliseg | 2 | tortuosity | 10 | 0.934 | 0.988 | -0.567 | 0.472 |
| hrf | reliseg_nocl | 0 | tortuosity | 10 | 0.850 | 0.867 | -0.608 | 0.538 |
| hrf | reliseg_nocl | 1 | tortuosity | 10 | 0.651 | 0.648 | -0.535 | 0.724 |
| hrf | reliseg_nocl | 2 | tortuosity | 10 | 0.832 | 0.818 | -0.471 | 0.607 |
| hrf | baseline | 0 | total_length | 10 | 0.932 | 0.879 | 0.657 | 0.567 |
| hrf | baseline | 1 | total_length | 10 | 0.935 | 0.879 | 0.572 | 0.596 |
| hrf | baseline | 2 | total_length | 10 | 0.922 | 0.830 | 0.507 | 0.627 |
| hrf | cfloss | 0 | total_length | 10 | 0.911 | 0.770 | 0.249 | 0.664 |
| hrf | cfloss | 1 | total_length | 10 | 0.907 | 0.830 | 0.269 | 0.708 |
| hrf | cfloss | 2 | total_length | 10 | 0.903 | 0.830 | 0.195 | 0.682 |
| hrf | continued | 0 | total_length | 10 | 0.954 | 0.879 | 0.677 | 0.520 |
| hrf | continued | 1 | total_length | 10 | 0.933 | 0.879 | 0.926 | 0.575 |
| hrf | continued | 2 | total_length | 10 | 0.937 | 0.879 | 0.689 | 0.561 |
| hrf | reliseg | 0 | total_length | 10 | 0.919 | 0.830 | 1.251 | 0.634 |
| hrf | reliseg | 1 | total_length | 10 | 0.907 | 0.879 | 1.450 | 0.672 |
| hrf | reliseg | 2 | total_length | 10 | 0.894 | 0.879 | 1.286 | 0.695 |
| hrf | reliseg_nocl | 0 | total_length | 10 | 0.911 | 0.770 | 1.063 | 0.668 |
| hrf | reliseg_nocl | 1 | total_length | 10 | 0.934 | 0.830 | 1.071 | 0.590 |
| hrf | reliseg_nocl | 2 | total_length | 10 | 0.920 | 0.830 | 0.874 | 0.615 |

#### stare -- delta-r vs `baseline` (paired bootstrap over the same images, 2000 resamples)

| dataset | config | seed | biomarker | d_r_vs_baseline | d_r_vs_baseline_lo | d_r_vs_baseline_hi |
|---|---|---|---|---|---|---|
| fives | cfloss | 0 | FD | -0.001 | -0.013 | 0.015 |
| fives | cfloss | 1 | FD | 0.003 | -0.007 | 0.011 |
| fives | cfloss | 2 | FD | 0.003 | 0.000 | 0.013 |
| fives | continued | 0 | FD | -0.002 | -0.013 | 0.012 |
| fives | continued | 1 | FD | 0.009 | -0.002 | 0.018 |
| fives | continued | 2 | FD | 0.006 | -0.003 | 0.015 |
| fives | reliseg | 0 | FD | 0.011 | 0.000 | 0.026 |
| fives | reliseg | 1 | FD | 0.008 | -0.002 | 0.015 |
| fives | reliseg | 2 | FD | -0.003 | -0.024 | 0.014 |
| fives | reliseg_nocl | 0 | FD | -0.005 | -0.011 | 0.004 |
| fives | reliseg_nocl | 1 | FD | 0.006 | -0.007 | 0.014 |
| fives | reliseg_nocl | 2 | FD | -0.007 | -0.032 | 0.018 |
| fives | cfloss | 0 | density | -0.013 | -0.110 | 0.001 |
| fives | cfloss | 1 | density | -0.000 | -0.013 | 0.005 |
| fives | cfloss | 2 | density | 0.002 | -0.010 | 0.011 |
| fives | continued | 0 | density | 0.003 | -0.000 | 0.019 |
| fives | continued | 1 | density | -0.002 | -0.010 | 0.004 |
| fives | continued | 2 | density | 0.002 | -0.004 | 0.020 |
| fives | reliseg | 0 | density | 0.010 | 0.002 | 0.064 |
| fives | reliseg | 1 | density | 0.003 | -0.001 | 0.029 |
| fives | reliseg | 2 | density | -0.002 | -0.023 | 0.018 |
| fives | reliseg_nocl | 0 | density | 0.011 | 0.002 | 0.069 |
| fives | reliseg_nocl | 1 | density | 0.002 | 0.000 | 0.017 |
| fives | reliseg_nocl | 2 | density | -0.000 | -0.035 | 0.040 |
| fives | cfloss | 0 | tortuosity | -0.256 | -0.799 | 0.059 |
| fives | cfloss | 1 | tortuosity | -0.011 | -0.042 | -0.001 |
| fives | cfloss | 2 | tortuosity | -0.015 | -0.064 | 0.009 |
| fives | continued | 0 | tortuosity | 0.027 | -0.006 | 0.184 |
| fives | continued | 1 | tortuosity | -0.001 | -0.010 | 0.007 |
| fives | continued | 2 | tortuosity | -0.114 | -0.552 | 0.058 |
| fives | reliseg | 0 | tortuosity | -0.038 | -0.185 | -0.001 |
| fives | reliseg | 1 | tortuosity | 0.004 | -0.007 | 0.028 |
| fives | reliseg | 2 | tortuosity | -0.006 | -0.123 | 0.052 |
| fives | reliseg_nocl | 0 | tortuosity | 0.003 | -0.014 | 0.048 |
| fives | reliseg_nocl | 1 | tortuosity | -0.006 | -0.045 | 0.003 |
| fives | reliseg_nocl | 2 | tortuosity | -0.022 | -0.064 | 0.006 |
| fives | cfloss | 0 | total_length | -0.000 | -0.012 | 0.005 |
| fives | cfloss | 1 | total_length | -0.001 | -0.003 | 0.002 |
| fives | cfloss | 2 | total_length | 0.003 | -0.001 | 0.013 |
| fives | continued | 0 | total_length | 0.001 | -0.001 | 0.007 |
| fives | continued | 1 | total_length | 0.003 | -0.001 | 0.009 |
| fives | continued | 2 | total_length | 0.005 | 0.000 | 0.016 |
| fives | reliseg | 0 | total_length | 0.008 | -0.001 | 0.019 |
| fives | reliseg | 1 | total_length | 0.006 | 0.001 | 0.016 |
| fives | reliseg | 2 | total_length | 0.001 | -0.009 | 0.018 |
| fives | reliseg_nocl | 0 | total_length | 0.001 | -0.006 | 0.005 |
| fives | reliseg_nocl | 1 | total_length | 0.001 | -0.002 | 0.004 |
| fives | reliseg_nocl | 2 | total_length | -0.002 | -0.011 | 0.015 |
| hrf | cfloss | 0 | FD | 0.006 | -0.059 | 0.041 |
| hrf | cfloss | 1 | FD | -0.010 | -0.088 | 0.025 |
| hrf | cfloss | 2 | FD | -0.006 | -0.042 | 0.021 |
| hrf | continued | 0 | FD | 0.017 | -0.007 | 0.056 |
| hrf | continued | 1 | FD | -0.024 | -0.054 | 0.016 |
| hrf | continued | 2 | FD | -0.002 | -0.049 | 0.064 |
| hrf | reliseg | 0 | FD | -0.021 | -0.080 | 0.041 |
| hrf | reliseg | 1 | FD | -0.069 | -0.129 | 0.026 |
| hrf | reliseg | 2 | FD | -0.061 | -0.229 | 0.069 |
| hrf | reliseg_nocl | 0 | FD | -0.021 | -0.080 | 0.035 |
| hrf | reliseg_nocl | 1 | FD | -0.008 | -0.028 | 0.048 |
| hrf | reliseg_nocl | 2 | FD | -0.013 | -0.091 | 0.073 |
| hrf | cfloss | 0 | density | 0.038 | -0.036 | 0.145 |
| hrf | cfloss | 1 | density | -0.039 | -0.084 | 0.003 |
| hrf | cfloss | 2 | density | -0.013 | -0.072 | 0.069 |
| hrf | continued | 0 | density | 0.073 | -0.003 | 0.165 |
| hrf | continued | 1 | density | -0.045 | -0.119 | -0.004 |
| hrf | continued | 2 | density | 0.014 | -0.054 | 0.064 |
| hrf | reliseg | 0 | density | 0.043 | -0.070 | 0.160 |
| hrf | reliseg | 1 | density | -0.049 | -0.112 | 0.023 |
| hrf | reliseg | 2 | density | -0.031 | -0.125 | 0.021 |
| hrf | reliseg_nocl | 0 | density | 0.041 | -0.071 | 0.172 |
| hrf | reliseg_nocl | 1 | density | -0.030 | -0.084 | 0.011 |
| hrf | reliseg_nocl | 2 | density | -0.044 | -0.144 | 0.012 |
| hrf | cfloss | 0 | tortuosity | 0.009 | -0.131 | 0.434 |
| hrf | cfloss | 1 | tortuosity | -0.210 | -0.455 | -0.001 |
| hrf | cfloss | 2 | tortuosity | -0.005 | -0.075 | 0.234 |
| hrf | continued | 0 | tortuosity | -0.036 | -0.202 | 0.400 |
| hrf | continued | 1 | tortuosity | -0.199 | -0.846 | 0.161 |
| hrf | continued | 2 | tortuosity | 0.053 | 0.000 | 0.218 |
| hrf | reliseg | 0 | tortuosity | -0.190 | -0.526 | 0.373 |
| hrf | reliseg | 1 | tortuosity | -0.122 | -0.762 | 0.184 |
| hrf | reliseg | 2 | tortuosity | 0.041 | -0.032 | 0.223 |
| hrf | reliseg_nocl | 0 | tortuosity | -0.012 | -0.127 | 0.334 |
| hrf | reliseg_nocl | 1 | tortuosity | -0.131 | -0.385 | 0.032 |
| hrf | reliseg_nocl | 2 | tortuosity | -0.060 | -0.154 | 0.062 |
| hrf | cfloss | 0 | total_length | -0.021 | -0.117 | 0.029 |
| hrf | cfloss | 1 | total_length | -0.027 | -0.112 | 0.005 |
| hrf | cfloss | 2 | total_length | -0.018 | -0.066 | 0.015 |
| hrf | continued | 0 | total_length | 0.022 | 0.000 | 0.065 |
| hrf | continued | 1 | total_length | -0.002 | -0.030 | 0.050 |
| hrf | continued | 2 | total_length | 0.015 | -0.025 | 0.082 |
| hrf | reliseg | 0 | total_length | -0.013 | -0.064 | 0.042 |
| hrf | reliseg | 1 | total_length | -0.027 | -0.062 | 0.048 |
| hrf | reliseg | 2 | total_length | -0.027 | -0.133 | 0.055 |
| hrf | reliseg_nocl | 0 | total_length | -0.021 | -0.107 | 0.038 |
| hrf | reliseg_nocl | 1 | total_length | -0.001 | -0.014 | 0.038 |
| hrf | reliseg_nocl | 2 | total_length | -0.001 | -0.043 | 0.047 |

#### stare -- delta-r vs `continued` (paired bootstrap over the same images, 2000 resamples)

| dataset | config | seed | biomarker | d_r_vs_continued | d_r_vs_continued_lo | d_r_vs_continued_hi |
|---|---|---|---|---|---|---|
| fives | cfloss | 0 | FD | 0.001 | -0.019 | 0.013 |
| fives | cfloss | 1 | FD | -0.006 | -0.015 | 0.001 |
| fives | cfloss | 2 | FD | -0.003 | -0.009 | 0.007 |
| fives | reliseg | 0 | FD | 0.014 | -0.001 | 0.035 |
| fives | reliseg | 1 | FD | -0.000 | -0.006 | 0.006 |
| fives | reliseg | 2 | FD | -0.009 | -0.028 | 0.002 |
| fives | reliseg_nocl | 0 | FD | -0.002 | -0.015 | 0.008 |
| fives | reliseg_nocl | 1 | FD | -0.003 | -0.013 | 0.003 |
| fives | reliseg_nocl | 2 | FD | -0.012 | -0.034 | 0.005 |
| fives | cfloss | 0 | density | -0.016 | -0.126 | -0.000 |
| fives | cfloss | 1 | density | 0.001 | -0.013 | 0.011 |
| fives | cfloss | 2 | density | -0.000 | -0.017 | 0.004 |
| fives | reliseg | 0 | density | 0.007 | 0.001 | 0.047 |
| fives | reliseg | 1 | density | 0.004 | 0.000 | 0.032 |
| fives | reliseg | 2 | density | -0.004 | -0.033 | 0.008 |
| fives | reliseg_nocl | 0 | density | 0.008 | 0.002 | 0.051 |
| fives | reliseg_nocl | 1 | density | 0.004 | -0.000 | 0.022 |
| fives | reliseg_nocl | 2 | density | -0.002 | -0.042 | 0.030 |
| fives | cfloss | 0 | tortuosity | -0.283 | -0.863 | -0.001 |
| fives | cfloss | 1 | tortuosity | -0.010 | -0.039 | -0.001 |
| fives | cfloss | 2 | tortuosity | 0.099 | -0.079 | 0.548 |
| fives | reliseg | 0 | tortuosity | -0.065 | -0.347 | -0.001 |
| fives | reliseg | 1 | tortuosity | 0.006 | -0.006 | 0.028 |
| fives | reliseg | 2 | tortuosity | 0.109 | -0.036 | 0.451 |
| fives | reliseg_nocl | 0 | tortuosity | -0.024 | -0.141 | 0.004 |
| fives | reliseg_nocl | 1 | tortuosity | -0.005 | -0.049 | 0.009 |
| fives | reliseg_nocl | 2 | tortuosity | 0.093 | -0.092 | 0.553 |
| fives | cfloss | 0 | total_length | -0.002 | -0.018 | 0.005 |
| fives | cfloss | 1 | total_length | -0.003 | -0.009 | 0.001 |
| fives | cfloss | 2 | total_length | -0.002 | -0.005 | 0.001 |
| fives | reliseg | 0 | total_length | 0.007 | -0.000 | 0.016 |
| fives | reliseg | 1 | total_length | 0.003 | -0.001 | 0.010 |
| fives | reliseg | 2 | total_length | -0.005 | -0.014 | 0.003 |
| fives | reliseg_nocl | 0 | total_length | 0.000 | -0.009 | 0.004 |
| fives | reliseg_nocl | 1 | total_length | -0.001 | -0.009 | 0.003 |
| fives | reliseg_nocl | 2 | total_length | -0.007 | -0.016 | 0.003 |
| hrf | cfloss | 0 | FD | -0.010 | -0.075 | 0.021 |
| hrf | cfloss | 1 | FD | 0.014 | -0.100 | 0.077 |
| hrf | cfloss | 2 | FD | -0.004 | -0.071 | 0.033 |
| hrf | reliseg | 0 | FD | -0.037 | -0.083 | -0.001 |
| hrf | reliseg | 1 | FD | -0.045 | -0.077 | 0.012 |
| hrf | reliseg | 2 | FD | -0.059 | -0.189 | 0.014 |
| hrf | reliseg_nocl | 0 | FD | -0.038 | -0.082 | -0.002 |
| hrf | reliseg_nocl | 1 | FD | 0.016 | -0.001 | 0.039 |
| hrf | reliseg_nocl | 2 | FD | -0.011 | -0.046 | 0.016 |
| hrf | cfloss | 0 | density | -0.035 | -0.060 | 0.015 |
| hrf | cfloss | 1 | density | 0.006 | -0.050 | 0.074 |
| hrf | cfloss | 2 | density | -0.027 | -0.054 | 0.040 |
| hrf | reliseg | 0 | density | -0.030 | -0.103 | 0.021 |
| hrf | reliseg | 1 | density | -0.004 | -0.027 | 0.063 |
| hrf | reliseg | 2 | density | -0.045 | -0.115 | 0.000 |
| hrf | reliseg_nocl | 0 | density | -0.032 | -0.100 | 0.036 |
| hrf | reliseg_nocl | 1 | density | 0.015 | -0.027 | 0.069 |
| hrf | reliseg_nocl | 2 | density | -0.058 | -0.119 | -0.007 |
| hrf | cfloss | 0 | tortuosity | 0.044 | 0.004 | 0.089 |
| hrf | cfloss | 1 | tortuosity | -0.011 | -0.455 | 0.576 |
| hrf | cfloss | 2 | tortuosity | -0.058 | -0.104 | 0.039 |
| hrf | reliseg | 0 | tortuosity | -0.154 | -0.410 | 0.046 |
| hrf | reliseg | 1 | tortuosity | 0.077 | -0.205 | 0.228 |
| hrf | reliseg | 2 | tortuosity | -0.012 | -0.070 | 0.045 |
| hrf | reliseg_nocl | 0 | tortuosity | 0.023 | -0.089 | 0.096 |
| hrf | reliseg_nocl | 1 | tortuosity | 0.069 | -0.253 | 0.535 |
| hrf | reliseg_nocl | 2 | tortuosity | -0.113 | -0.265 | -0.013 |
| hrf | cfloss | 0 | total_length | -0.043 | -0.148 | 0.001 |
| hrf | cfloss | 1 | total_length | -0.026 | -0.156 | 0.030 |
| hrf | cfloss | 2 | total_length | -0.033 | -0.115 | 0.007 |
| hrf | reliseg | 0 | total_length | -0.035 | -0.087 | -0.004 |
| hrf | reliseg | 1 | total_length | -0.025 | -0.040 | 0.009 |
| hrf | reliseg | 2 | total_length | -0.042 | -0.123 | 0.001 |
| hrf | reliseg_nocl | 0 | total_length | -0.043 | -0.129 | -0.004 |
| hrf | reliseg_nocl | 1 | total_length | 0.001 | -0.046 | 0.030 |
| hrf | reliseg_nocl | 2 | total_length | -0.016 | -0.063 | 0.005 |

## 6. Safety endpoints
Source: `results/pivot/e2_safety.csv`. Margins are the pre-registered ones listed at the top. `BREACH` = the point estimate is past the margin; `AT-RISK` = the point estimate holds but the 95% CI reaches past it.

Flagged rows: 316 of 1484.

| dataset | config | reference | seed | endpoint | biomarker | pipeline | value | lo | hi | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| fives | cfloss | baseline | 0 | delta_r | tortuosity | pvbm | 0.108 | -0.070 | 0.212 | AT-RISK |
| fives | cfloss | baseline | 1 | delta_r | tortuosity | pvbm | 0.041 | -0.102 | 0.094 | AT-RISK |
| fives | cfloss | baseline | 2 | delta_r | tortuosity | pvbm | 0.024 | -0.083 | 0.088 | AT-RISK |
| fives | cfloss | baseline | pooled(0,1,2) | delta_r | tortuosity | pvbm | 0.057 | -0.053 | 0.115 | AT-RISK |
| fives | cfloss | continued | 1 | delta_r | tortuosity | pvbm | 0.018 | -0.143 | 0.074 | AT-RISK |
| fives | cfloss | continued | 2 | delta_r | tortuosity | pvbm | 0.109 | -0.160 | 0.234 | AT-RISK |
| fives | cfloss | continued | pooled(0,1,2) | delta_r | tortuosity | pvbm | 0.091 | -0.086 | 0.175 | AT-RISK |
| fives | cfloss | baseline | 0 | delta_r | tortuosity | skan | 0.070 | -0.050 | 0.247 | AT-RISK |
| fives | cfloss | baseline | 2 | delta_r | tortuosity | skan | -0.101 | -0.270 | 0.063 | BREACH |
| fives | cfloss | continued | 0 | delta_r | tortuosity | skan | 0.026 | -0.112 | 0.226 | AT-RISK |
| fives | continued | baseline | 0 | delta_r | tortuosity | pvbm | -0.037 | -0.109 | -0.011 | AT-RISK |
| fives | continued | baseline | 2 | delta_r | tortuosity | pvbm | -0.085 | -0.175 | 0.141 | BREACH |
| fives | continued | baseline | 3 | delta_r | tortuosity | pvbm | -0.136 | -0.278 | 0.145 | BREACH |
| fives | continued | baseline | 5 | delta_r | tortuosity | pvbm | -0.047 | -0.142 | 0.190 | AT-RISK |
| fives | continued | baseline | pooled(0,1,2,3,4,5) | delta_r | tortuosity | pvbm | -0.023 | -0.051 | 0.055 | AT-RISK |
| fives | continued | baseline | 1 | delta_r | tortuosity | skan | -0.109 | -0.317 | 0.043 | BREACH |
| fives | continued | baseline | 2 | delta_r | tortuosity | skan | -0.126 | -0.290 | 0.028 | BREACH |
| fives | continued | baseline | 3 | delta_r | tortuosity | skan | 0.071 | -0.137 | 0.321 | AT-RISK |
| fives | continued | baseline | 4 | delta_r | tortuosity | skan | 0.020 | -0.146 | 0.217 | AT-RISK |
| fives | continued | baseline | 5 | delta_r | tortuosity | skan | -0.000 | -0.092 | 0.070 | AT-RISK |
| fives | continued | baseline | pooled(0,1,2,3,4,5) | delta_r | tortuosity | skan | -0.017 | -0.090 | 0.060 | AT-RISK |
| fives | reliseg | baseline | 0 | delta_bias_sigma | tortuosity | pvbm | 0.897 |  |  | BREACH |
| fives | reliseg | baseline | 1 | delta_bias_sigma | tortuosity | pvbm | 1.972 |  |  | BREACH |
| fives | reliseg | baseline | 2 | delta_bias_sigma | tortuosity | pvbm | 0.444 |  |  | BREACH |
| fives | reliseg | baseline | 3 | delta_bias_sigma | tortuosity | pvbm | 0.614 |  |  | BREACH |
| fives | reliseg | baseline | 4 | delta_bias_sigma | tortuosity | pvbm | 0.492 |  |  | BREACH |
| fives | reliseg | baseline | 5 | delta_bias_sigma | tortuosity | pvbm | 0.676 |  |  | BREACH |
| fives | reliseg | baseline | pooled(0,1,2,3,4,5) | delta_bias_sigma | tortuosity | pvbm | 0.849 |  |  | BREACH |
| fives | reliseg | continued | 0 | delta_bias_sigma | tortuosity | pvbm | 0.979 |  |  | BREACH |
| fives | reliseg | continued | 1 | delta_bias_sigma | tortuosity | pvbm | 2.038 |  |  | BREACH |
| fives | reliseg | continued | 2 | delta_bias_sigma | tortuosity | pvbm | 0.463 |  |  | BREACH |
| fives | reliseg | continued | 3 | delta_bias_sigma | tortuosity | pvbm | 0.730 |  |  | BREACH |
| fives | reliseg | continued | 4 | delta_bias_sigma | tortuosity | pvbm | 0.453 |  |  | BREACH |
| fives | reliseg | continued | 5 | delta_bias_sigma | tortuosity | pvbm | 0.759 |  |  | BREACH |
| fives | reliseg | continued | pooled(0,1,2,3,4,5) | delta_bias_sigma | tortuosity | pvbm | 0.904 |  |  | BREACH |
| fives | reliseg | baseline | 0 | delta_r | tortuosity | pvbm | 0.024 | -0.163 | 0.132 | AT-RISK |
| fives | reliseg | baseline | 1 | delta_r | tortuosity | pvbm | -0.062 | -0.239 | 0.137 | BREACH |
| fives | reliseg | baseline | 2 | delta_r | tortuosity | pvbm | -0.024 | -0.143 | 0.068 | AT-RISK |
| fives | reliseg | baseline | 3 | delta_r | tortuosity | pvbm | -0.100 | -0.271 | 0.074 | BREACH |
| fives | reliseg | baseline | 4 | delta_r | tortuosity | pvbm | 0.104 | -0.055 | 0.213 | AT-RISK |
| fives | reliseg | baseline | 5 | delta_r | tortuosity | pvbm | -0.023 | -0.105 | 0.179 | AT-RISK |
| fives | reliseg | baseline | pooled(0,1,2,3,4,5) | delta_r | tortuosity | pvbm | -0.013 | -0.085 | 0.055 | AT-RISK |
| fives | reliseg | continued | 0 | delta_r | tortuosity | pvbm | 0.061 | -0.104 | 0.187 | AT-RISK |
| fives | reliseg | continued | 1 | delta_r | tortuosity | pvbm | -0.084 | -0.285 | 0.149 | BREACH |
| fives | reliseg | continued | 2 | delta_r | tortuosity | pvbm | 0.061 | -0.199 | 0.174 | AT-RISK |
| fives | reliseg | continued | 3 | delta_r | tortuosity | pvbm | 0.036 | -0.161 | 0.127 | AT-RISK |
| fives | reliseg | continued | 4 | delta_r | tortuosity | pvbm | -0.042 | -0.091 | 0.077 | AT-RISK |
| fives | reliseg | continued | 5 | delta_r | tortuosity | pvbm | 0.024 | -0.156 | 0.093 | AT-RISK |
| fives | reliseg | continued | pooled(0,1,2,3,4,5) | delta_r | tortuosity | pvbm | 0.009 | -0.111 | 0.061 | AT-RISK |
| fives | reliseg | baseline | 0 | delta_r | tortuosity | skan | 0.033 | -0.135 | 0.199 | AT-RISK |
| fives | reliseg | baseline | 1 | delta_r | tortuosity | skan | 0.012 | -0.090 | 0.105 | AT-RISK |
| fives | reliseg | baseline | 2 | delta_r | tortuosity | skan | -0.133 | -0.293 | 0.002 | BREACH |
| fives | reliseg | baseline | 3 | delta_r | tortuosity | skan | 0.105 | -0.103 | 0.337 | AT-RISK |
| fives | reliseg | baseline | 4 | delta_r | tortuosity | skan | -0.013 | -0.076 | 0.043 | AT-RISK |
| fives | reliseg | baseline | pooled(0,1,2,3,4,5) | delta_r | tortuosity | skan | 0.001 | -0.063 | 0.064 | AT-RISK |
| fives | reliseg | continued | 0 | delta_r | tortuosity | skan | -0.010 | -0.168 | 0.155 | AT-RISK |
| fives | reliseg | continued | 2 | delta_r | tortuosity | skan | -0.008 | -0.103 | 0.064 | AT-RISK |
| fives | reliseg | continued | 3 | delta_r | tortuosity | skan | 0.034 | -0.053 | 0.132 | AT-RISK |
| fives | reliseg | continued | 4 | delta_r | tortuosity | skan | -0.033 | -0.206 | 0.103 | AT-RISK |
| fives | reliseg | continued | 5 | delta_r | tortuosity | skan | 0.003 | -0.065 | 0.093 | AT-RISK |
| fives | reliseg_nocl | continued | 0 | delta_bias_sigma | tortuosity | pvbm | 0.319 |  |  | BREACH |
| fives | reliseg_nocl | continued | pooled(0,1,2) | delta_bias_sigma | tortuosity | pvbm | 0.269 |  |  | BREACH |
| fives | reliseg_nocl | baseline | 0 | delta_r | tortuosity | pvbm | 0.004 | -0.222 | 0.067 | AT-RISK |
| fives | reliseg_nocl | baseline | 2 | delta_r | tortuosity | pvbm | -0.043 | -0.122 | 0.170 | AT-RISK |
| fives | reliseg_nocl | continued | 0 | delta_r | tortuosity | pvbm | 0.041 | -0.160 | 0.106 | AT-RISK |
| fives | reliseg_nocl | continued | 1 | delta_r | tortuosity | pvbm | 0.008 | -0.057 | 0.128 | AT-RISK |
| fives | reliseg_nocl | continued | 2 | delta_r | tortuosity | pvbm | 0.042 | -0.075 | 0.151 | AT-RISK |
| fives | reliseg_nocl | baseline | 2 | delta_r | tortuosity | skan | -0.067 | -0.242 | 0.087 | BREACH |
| fives | reliseg_nocl | baseline | pooled(0,1,2) | delta_r | tortuosity | skan | 0.007 | -0.072 | 0.066 | AT-RISK |
| fives | reliseg_nocl | continued | 0 | delta_r | tortuosity | skan | 0.007 | -0.078 | 0.097 | AT-RISK |
| hrf | abl_cf_matched | baseline | 0 | delta_bias_sigma | density | pvbm | -0.332 |  |  | BREACH |
| hrf | abl_cf_matched | continued | 0 | delta_bias_sigma | density | pvbm | -0.285 |  |  | BREACH |
| hrf | abl_cf_matched | baseline | 0 | delta_bias_sigma | density | skan | -0.332 |  |  | BREACH |
| hrf | abl_cf_matched | continued | 0 | delta_bias_sigma | density | skan | -0.285 |  |  | BREACH |
| hrf | abl_cf_matched | baseline | 0 | delta_r | density | pvbm | -0.044 | -0.084 | 0.003 | AT-RISK |
| hrf | abl_cf_matched | continued | 0 | delta_r | density | pvbm | -0.040 | -0.082 | -0.004 | AT-RISK |
| hrf | abl_cf_matched | baseline | 0 | delta_r | density | skan | -0.044 | -0.084 | 0.003 | AT-RISK |
| hrf | abl_cf_matched | continued | 0 | delta_r | density | skan | -0.040 | -0.082 | -0.004 | AT-RISK |
| hrf | abl_cf_matched | baseline | 0 | delta_r | tortuosity | pvbm | -0.036 | -0.165 | 0.049 | AT-RISK |
| hrf | abl_cf_matched | continued | 0 | delta_r | tortuosity | pvbm | 0.013 | -0.056 | 0.129 | AT-RISK |

## 6b. HRF -- development set, all seeds, `last.pt`, skan
HRF's test split was touched by probes P1-P5, so nothing here is confirmatory. It is reported in full because it is where the headroom is: the HRF baseline sits at r = 0.29-0.36 for FD and total_length against 0.90-0.98 on FIVES, so an effect that exists at all should be visible here first.


### Pooled over seeds 0-2, vs `baseline`

| config | biomarker | d_r | lo | hi | seed_lo | seed_hi | n_seeds_positive | direction_consistent | d_bias_sigma |
|---|---|---|---|---|---|---|---|---|---|
| cfloss | FD | -0.151 | -0.285 | -0.050 | -0.290 | -0.012 | 0 | yes | -0.690 |
| cfloss | density | -0.020 | -0.061 | 0.018 | -0.083 | 0.043 | 1 | no | -0.660 |
| cfloss | tortuosity | -0.052 | -0.165 | 0.017 | -0.333 | 0.230 | 1 | no | 0.021 |
| cfloss | total_length | -0.144 | -0.258 | -0.058 | -0.279 | -0.009 | 0 | yes | -0.715 |
| continued | FD | 0.005 | -0.097 | 0.076 | -0.151 | 0.160 | 1 | no | -0.021 |
| continued | density | 0.016 | -0.018 | 0.053 | -0.031 | 0.063 | 2 | no | 0.077 |
| continued | tortuosity | -0.057 | -0.179 | 0.033 | -0.314 | 0.199 | 1 | no | -0.043 |
| continued | total_length | 0.004 | -0.092 | 0.071 | -0.140 | 0.148 | 1 | no | -0.004 |
| reliseg | FD | 0.039 | -0.064 | 0.142 | -0.185 | 0.263 | 1 | no | 0.920 |
| reliseg | density | 0.007 | -0.031 | 0.049 | -0.071 | 0.084 | 2 | no | -0.201 |
| reliseg | tortuosity | -0.031 | -0.130 | 0.025 | -0.288 | 0.225 | 1 | no | 1.576 |
| reliseg | total_length | 0.019 | -0.079 | 0.112 | -0.217 | 0.255 | 1 | no | 0.893 |
| reliseg_nocl | FD | 0.020 | -0.067 | 0.109 | -0.104 | 0.143 | 2 | no | 0.800 |
| reliseg_nocl | density | 0.003 | -0.042 | 0.050 | -0.064 | 0.070 | 2 | no | -0.347 |
| reliseg_nocl | tortuosity | -0.040 | -0.164 | 0.030 | -0.395 | 0.315 | 1 | no | 0.418 |
| reliseg_nocl | total_length | -0.017 | -0.098 | 0.056 | -0.127 | 0.093 | 1 | no | 0.572 |

### Seed by seed, vs `baseline`

| config | seed | biomarker | d_r | lo | hi |
|---|---|---|---|---|---|
| cfloss | 0 | FD | -0.096 | -0.188 | -0.001 |
| cfloss | 1 | FD | -0.208 | -0.373 | -0.073 |
| cfloss | 2 | FD | -0.148 | -0.321 | -0.023 |
| cfloss | 0 | density | -0.047 | -0.092 | -0.001 |
| cfloss | 1 | density | 0.004 | -0.081 | 0.078 |
| cfloss | 2 | density | -0.018 | -0.079 | 0.038 |
| cfloss | 0 | tortuosity | 0.034 | -0.154 | 0.180 |
| cfloss | 1 | tortuosity | -0.009 | -0.197 | 0.138 |
| cfloss | 2 | tortuosity | -0.180 | -0.377 | -0.036 |
| cfloss | 0 | total_length | -0.089 | -0.163 | -0.013 |
| cfloss | 1 | total_length | -0.197 | -0.333 | -0.082 |
| cfloss | 2 | total_length | -0.147 | -0.298 | -0.031 |
| continued | 0 | FD | 0.073 | -0.007 | 0.155 |
| continued | 1 | FD | -0.050 | -0.154 | 0.042 |
| continued | 2 | FD | -0.009 | -0.151 | 0.090 |
| continued | 0 | density | -0.004 | -0.060 | 0.061 |
| continued | 1 | density | 0.033 | -0.026 | 0.091 |
| continued | 2 | density | 0.020 | -0.038 | 0.078 |
| continued | 0 | tortuosity | -0.052 | -0.198 | 0.040 |
| continued | 1 | tortuosity | 0.044 | -0.055 | 0.169 |
| continued | 2 | tortuosity | -0.163 | -0.368 | -0.001 |
| continued | 0 | total_length | 0.068 | -0.023 | 0.154 |
| continued | 1 | total_length | -0.044 | -0.132 | 0.034 |
| continued | 2 | total_length | -0.012 | -0.150 | 0.078 |
| reliseg | 0 | FD | 0.143 | 0.010 | 0.280 |
| reliseg | 1 | FD | -0.006 | -0.116 | 0.101 |
| reliseg | 2 | FD | -0.020 | -0.153 | 0.104 |
| reliseg | 0 | density | -0.028 | -0.073 | 0.030 |
| reliseg | 1 | density | 0.033 | -0.033 | 0.098 |
| reliseg | 2 | density | 0.015 | -0.060 | 0.092 |
| reliseg | 0 | tortuosity | -0.036 | -0.267 | 0.130 |
| reliseg | 1 | tortuosity | 0.074 | 0.003 | 0.223 |
| reliseg | 2 | tortuosity | -0.132 | -0.309 | -0.027 |
| reliseg | 0 | total_length | 0.129 | -0.015 | 0.281 |
| reliseg | 1 | total_length | -0.038 | -0.127 | 0.049 |
| reliseg | 2 | total_length | -0.034 | -0.158 | 0.077 |
| reliseg_nocl | 0 | FD | 0.074 | 0.003 | 0.149 |
| reliseg_nocl | 1 | FD | -0.024 | -0.156 | 0.100 |
| reliseg_nocl | 2 | FD | 0.010 | -0.143 | 0.149 |
| reliseg_nocl | 0 | density | -0.024 | -0.076 | 0.036 |
| reliseg_nocl | 1 | density | 0.029 | -0.032 | 0.088 |
| reliseg_nocl | 2 | density | 0.003 | -0.077 | 0.085 |
| reliseg_nocl | 0 | tortuosity | -0.084 | -0.267 | -0.012 |
| reliseg_nocl | 1 | tortuosity | 0.120 | -0.026 | 0.299 |
| reliseg_nocl | 2 | tortuosity | -0.156 | -0.417 | -0.021 |
| reliseg_nocl | 0 | total_length | 0.032 | -0.032 | 0.100 |
| reliseg_nocl | 1 | total_length | -0.053 | -0.170 | 0.053 |
| reliseg_nocl | 2 | total_length | -0.030 | -0.173 | 0.090 |

### Pooled over seeds 0-2, vs `continued`

| config | biomarker | d_r | lo | hi | seed_lo | seed_hi | n_seeds_positive | direction_consistent | d_bias_sigma |
|---|---|---|---|---|---|---|---|---|---|
| cfloss | FD | -0.156 | -0.222 | -0.092 | -0.193 | -0.118 | 0 | yes | -0.669 |
| cfloss | density | -0.037 | -0.074 | -0.002 | -0.053 | -0.020 | 0 | yes | -0.737 |
| cfloss | tortuosity | 0.006 | -0.094 | 0.087 | -0.174 | 0.185 | 1 | no | 0.064 |
| cfloss | total_length | -0.148 | -0.208 | -0.090 | -0.178 | -0.118 | 0 | yes | -0.712 |
| reliseg | FD | 0.034 | -0.037 | 0.107 | -0.068 | 0.137 | 2 | no | 0.941 |
| reliseg | density | -0.010 | -0.023 | 0.006 | -0.040 | 0.021 | 1 | no | -0.278 |
| reliseg | tortuosity | 0.026 | -0.099 | 0.149 | 0.006 | 0.046 | 3 | yes | 1.619 |
| reliseg | total_length | 0.015 | -0.040 | 0.074 | -0.089 | 0.119 | 2 | no | 0.897 |
| reliseg_nocl | FD | 0.015 | -0.037 | 0.077 | -0.017 | 0.047 | 3 | yes | 0.821 |
| reliseg_nocl | density | -0.013 | -0.032 | 0.005 | -0.035 | 0.009 | 0 | yes | -0.424 |
| reliseg_nocl | tortuosity | 0.017 | -0.060 | 0.086 | -0.119 | 0.153 | 2 | no | 0.461 |
| reliseg_nocl | total_length | -0.021 | -0.060 | 0.028 | -0.055 | 0.013 | 0 | yes | 0.576 |

### Seed by seed, vs `continued`

| config | seed | biomarker | d_r | lo | hi |
|---|---|---|---|---|---|
| cfloss | 0 | FD | -0.169 | -0.223 | -0.104 |
| cfloss | 1 | FD | -0.158 | -0.256 | -0.068 |
| cfloss | 2 | FD | -0.139 | -0.213 | -0.075 |
| cfloss | 0 | density | -0.042 | -0.079 | -0.009 |
| cfloss | 1 | density | -0.029 | -0.082 | 0.019 |
| cfloss | 2 | density | -0.038 | -0.075 | -0.007 |
| cfloss | 0 | tortuosity | 0.087 | -0.168 | 0.328 |
| cfloss | 1 | tortuosity | -0.053 | -0.208 | 0.075 |
| cfloss | 2 | tortuosity | -0.017 | -0.080 | 0.032 |
| cfloss | 0 | total_length | -0.157 | -0.215 | -0.094 |
| cfloss | 1 | total_length | -0.153 | -0.244 | -0.068 |
| cfloss | 2 | total_length | -0.135 | -0.206 | -0.069 |
| reliseg | 0 | FD | 0.070 | -0.020 | 0.171 |
| reliseg | 1 | FD | 0.044 | -0.052 | 0.141 |
| reliseg | 2 | FD | -0.011 | -0.084 | 0.065 |
| reliseg | 0 | density | -0.023 | -0.044 | -0.000 |
| reliseg | 1 | density | 0.000 | -0.016 | 0.014 |
| reliseg | 2 | density | -0.006 | -0.031 | 0.020 |
| reliseg | 0 | tortuosity | 0.017 | -0.268 | 0.288 |
| reliseg | 1 | tortuosity | 0.031 | -0.131 | 0.240 |
| reliseg | 2 | tortuosity | 0.030 | -0.074 | 0.126 |
| reliseg | 0 | total_length | 0.061 | -0.037 | 0.176 |
| reliseg | 1 | total_length | 0.006 | -0.059 | 0.071 |
| reliseg | 2 | total_length | -0.022 | -0.075 | 0.037 |
| reliseg_nocl | 0 | FD | 0.001 | -0.081 | 0.099 |
| reliseg_nocl | 1 | FD | 0.026 | -0.038 | 0.095 |
| reliseg_nocl | 2 | FD | 0.020 | -0.034 | 0.086 |
| reliseg_nocl | 0 | density | -0.020 | -0.046 | 0.009 |
| reliseg_nocl | 1 | density | -0.003 | -0.025 | 0.012 |
| reliseg_nocl | 2 | density | -0.017 | -0.049 | 0.015 |
| reliseg_nocl | 0 | tortuosity | -0.031 | -0.174 | 0.036 |
| reliseg_nocl | 1 | tortuosity | 0.076 | -0.030 | 0.247 |
| reliseg_nocl | 2 | tortuosity | 0.007 | -0.125 | 0.148 |
| reliseg_nocl | 0 | total_length | -0.036 | -0.118 | 0.064 |
| reliseg_nocl | 1 | total_length | -0.009 | -0.061 | 0.044 |
| reliseg_nocl | 2 | total_length | -0.018 | -0.060 | 0.032 |

### HRF safety flags (clDice non-inferiority and the unconstrained markers)

Flagged HRF rows: 246.

| config | reference | seed | endpoint | biomarker | pipeline | value | lo | hi | margin_value | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| abl_cf_matched | baseline | 0 | delta_bias_sigma | density | pvbm | -0.3323 |  |  | 0.2500 | BREACH |
| abl_cf_matched | continued | 0 | delta_bias_sigma | density | pvbm | -0.2846 |  |  | 0.2500 | BREACH |
| abl_cf_matched | baseline | 0 | delta_bias_sigma | density | skan | -0.3323 |  |  | 0.2500 | BREACH |
| abl_cf_matched | continued | 0 | delta_bias_sigma | density | skan | -0.2846 |  |  | 0.2500 | BREACH |
| abl_cf_on_base | baseline | 0 | delta_bias_sigma | tortuosity | skan | 0.2662 |  |  | 0.2500 | BREACH |
| abl_density_only | baseline | 0 | delta_bias_sigma | density | pvbm | -0.3715 |  |  | 0.2500 | BREACH |
| abl_density_only | continued | 0 | delta_bias_sigma | density | pvbm | -0.3238 |  |  | 0.2500 | BREACH |
| abl_density_only | baseline | 0 | delta_bias_sigma | density | skan | -0.3715 |  |  | 0.2500 | BREACH |
| abl_density_only | continued | 0 | delta_bias_sigma | density | skan | -0.3238 |  |  | 0.2500 | BREACH |
| abl_density_only | baseline | 0 | delta_bias_sigma | tortuosity | skan | 0.2850 |  |  | 0.2500 | BREACH |
| abl_fixed_ladder | baseline | 0 | delta_bias_sigma | density | pvbm | -0.3398 |  |  | 0.2500 | BREACH |
| abl_fixed_ladder | continued | 0 | delta_bias_sigma | density | pvbm | -0.2921 |  |  | 0.2500 | BREACH |
| abl_fixed_ladder | baseline | 0 | delta_bias_sigma | density | skan | -0.3398 |  |  | 0.2500 | BREACH |
| abl_fixed_ladder | continued | 0 | delta_bias_sigma | density | skan | -0.2921 |  |  | 0.2500 | BREACH |
| abl_fixed_ladder | baseline | 0 | delta_bias_sigma | tortuosity | pvbm | 10.8443 |  |  | 0.2500 | BREACH |
| abl_fixed_ladder | continued | 0 | delta_bias_sigma | tortuosity | pvbm | 10.7088 |  |  | 0.2500 | BREACH |
| abl_fixed_ladder | baseline | 0 | delta_bias_sigma | tortuosity | skan | 3.1789 |  |  | 0.2500 | BREACH |
| abl_fixed_ladder | continued | 0 | delta_bias_sigma | tortuosity | skan | 3.0006 |  |  | 0.2500 | BREACH |
| abl_length_only | baseline | 0 | delta_bias_sigma | tortuosity | pvbm | 1.6238 |  |  | 0.2500 | BREACH |
| abl_length_only | continued | 0 | delta_bias_sigma | tortuosity | pvbm | 1.4883 |  |  | 0.2500 | BREACH |
| abl_length_only | baseline | 0 | delta_bias_sigma | tortuosity | skan | 0.7465 |  |  | 0.2500 | BREACH |
| abl_length_only | continued | 0 | delta_bias_sigma | tortuosity | skan | 0.5682 |  |  | 0.2500 | BREACH |
| abl_no_density | baseline | 0 | delta_bias_sigma | tortuosity | pvbm | 11.2845 |  |  | 0.2500 | BREACH |
| abl_no_density | continued | 0 | delta_bias_sigma | tortuosity | pvbm | 11.1490 |  |  | 0.2500 | BREACH |
| abl_no_density | baseline | 0 | delta_bias_sigma | tortuosity | skan | 2.5472 |  |  | 0.2500 | BREACH |
| abl_no_density | continued | 0 | delta_bias_sigma | tortuosity | skan | 2.3689 |  |  | 0.2500 | BREACH |
| abl_no_fd | baseline | 0 | delta_bias_sigma | density | pvbm | -0.3737 |  |  | 0.2500 | BREACH |
| abl_no_fd | continued | 0 | delta_bias_sigma | density | pvbm | -0.3260 |  |  | 0.2500 | BREACH |
| abl_no_fd | baseline | 0 | delta_bias_sigma | density | skan | -0.3737 |  |  | 0.2500 | BREACH |
| abl_no_fd | continued | 0 | delta_bias_sigma | density | skan | -0.3260 |  |  | 0.2500 | BREACH |
| abl_no_fd | baseline | 0 | delta_bias_sigma | tortuosity | pvbm | 2.8848 |  |  | 0.2500 | BREACH |
| abl_no_fd | continued | 0 | delta_bias_sigma | tortuosity | pvbm | 2.7493 |  |  | 0.2500 | BREACH |
| abl_no_fd | baseline | 0 | delta_bias_sigma | tortuosity | skan | 1.6910 |  |  | 0.2500 | BREACH |
| abl_no_fd | continued | 0 | delta_bias_sigma | tortuosity | skan | 1.5127 |  |  | 0.2500 | BREACH |
| abl_no_length | baseline | 0 | delta_bias_sigma | density | pvbm | -0.3920 |  |  | 0.2500 | BREACH |
| abl_no_length | continued | 0 | delta_bias_sigma | density | pvbm | -0.3443 |  |  | 0.2500 | BREACH |
| abl_no_length | baseline | 0 | delta_bias_sigma | density | skan | -0.3920 |  |  | 0.2500 | BREACH |
| abl_no_length | continued | 0 | delta_bias_sigma | density | skan | -0.3443 |  |  | 0.2500 | BREACH |
| abl_no_length | baseline | 0 | delta_bias_sigma | tortuosity | skan | 0.2850 |  |  | 0.2500 | BREACH |
| cfloss | baseline | 0 | delta_bias_sigma | density | pvbm | -0.8286 |  |  | 0.2500 | BREACH |
| cfloss | continued | 0 | delta_bias_sigma | density | pvbm | -0.7809 |  |  | 0.2500 | BREACH |
| cfloss | baseline | 0 | delta_bias_sigma | density | skan | -0.8286 |  |  | 0.2500 | BREACH |
| cfloss | continued | 0 | delta_bias_sigma | density | skan | -0.7809 |  |  | 0.2500 | BREACH |
| cfloss | baseline | 1 | delta_bias_sigma | density | pvbm | -0.5528 |  |  | 0.2500 | BREACH |
| cfloss | continued | 1 | delta_bias_sigma | density | pvbm | -0.7417 |  |  | 0.2500 | BREACH |
| cfloss | baseline | 1 | delta_bias_sigma | density | skan | -0.5528 |  |  | 0.2500 | BREACH |
| cfloss | continued | 1 | delta_bias_sigma | density | skan | -0.7417 |  |  | 0.2500 | BREACH |
| cfloss | baseline | 2 | delta_bias_sigma | density | pvbm | -0.5991 |  |  | 0.2500 | BREACH |
| cfloss | continued | 2 | delta_bias_sigma | density | pvbm | -0.6888 |  |  | 0.2500 | BREACH |
| cfloss | baseline | 2 | delta_bias_sigma | density | skan | -0.5991 |  |  | 0.2500 | BREACH |
| cfloss | continued | 2 | delta_bias_sigma | density | skan | -0.6888 |  |  | 0.2500 | BREACH |
| cfloss | baseline | pooled(0,1,2) | delta_bias_sigma | density | pvbm | -0.6602 |  |  | 0.2500 | BREACH |
| cfloss | continued | pooled(0,1,2) | delta_bias_sigma | density | pvbm | -0.7371 |  |  | 0.2500 | BREACH |
| cfloss | baseline | pooled(0,1,2) | delta_bias_sigma | density | skan | -0.6602 |  |  | 0.2500 | BREACH |
| cfloss | continued | pooled(0,1,2) | delta_bias_sigma | density | skan | -0.7371 |  |  | 0.2500 | BREACH |
| cfloss | baseline | 0 | delta_bias_sigma | tortuosity | skan | 0.2636 |  |  | 0.2500 | BREACH |
| reliseg | baseline | 0 | delta_bias_sigma | density | pvbm | -0.3569 |  |  | 0.2500 | BREACH |
| reliseg | continued | 0 | delta_bias_sigma | density | pvbm | -0.3092 |  |  | 0.2500 | BREACH |
| reliseg | baseline | 0 | delta_bias_sigma | density | skan | -0.3569 |  |  | 0.2500 | BREACH |
| reliseg | continued | 0 | delta_bias_sigma | density | skan | -0.3092 |  |  | 0.2500 | BREACH |
| reliseg | continued | 1 | delta_bias_sigma | density | pvbm | -0.2774 |  |  | 0.2500 | BREACH |
| reliseg | continued | 1 | delta_bias_sigma | density | skan | -0.2774 |  |  | 0.2500 | BREACH |
| reliseg | continued | pooled(0,1,2) | delta_bias_sigma | density | pvbm | -0.2779 |  |  | 0.2500 | BREACH |
| reliseg | continued | pooled(0,1,2) | delta_bias_sigma | density | skan | -0.2779 |  |  | 0.2500 | BREACH |
| reliseg | baseline | 0 | delta_bias_sigma | tortuosity | pvbm | 14.4601 |  |  | 0.2500 | BREACH |
| reliseg | continued | 0 | delta_bias_sigma | tortuosity | pvbm | 14.3246 |  |  | 0.2500 | BREACH |
| reliseg | baseline | 0 | delta_bias_sigma | tortuosity | skan | 2.9172 |  |  | 0.2500 | BREACH |
| reliseg | continued | 0 | delta_bias_sigma | tortuosity | skan | 2.7389 |  |  | 0.2500 | BREACH |
| reliseg | baseline | 1 | delta_bias_sigma | tortuosity | pvbm | 2.3338 |  |  | 0.2500 | BREACH |
| reliseg | continued | 1 | delta_bias_sigma | tortuosity | pvbm | 2.4164 |  |  | 0.2500 | BREACH |
| reliseg | baseline | 1 | delta_bias_sigma | tortuosity | skan | 0.7711 |  |  | 0.2500 | BREACH |
| reliseg | continued | 1 | delta_bias_sigma | tortuosity | skan | 0.9251 |  |  | 0.2500 | BREACH |
| reliseg | baseline | 2 | delta_bias_sigma | tortuosity | pvbm | 2.9405 |  |  | 0.2500 | BREACH |
| reliseg | continued | 2 | delta_bias_sigma | tortuosity | pvbm | 3.0956 |  |  | 0.2500 | BREACH |
| reliseg | baseline | 2 | delta_bias_sigma | tortuosity | skan | 1.0403 |  |  | 0.2500 | BREACH |
| reliseg | continued | 2 | delta_bias_sigma | tortuosity | skan | 1.1939 |  |  | 0.2500 | BREACH |
| reliseg | baseline | pooled(0,1,2) | delta_bias_sigma | tortuosity | pvbm | 6.5782 |  |  | 0.2500 | BREACH |
| reliseg | continued | pooled(0,1,2) | delta_bias_sigma | tortuosity | pvbm | 6.6122 |  |  | 0.2500 | BREACH |
| reliseg | baseline | pooled(0,1,2) | delta_bias_sigma | tortuosity | skan | 1.5762 |  |  | 0.2500 | BREACH |
| reliseg | continued | pooled(0,1,2) | delta_bias_sigma | tortuosity | skan | 1.6193 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | 0 | delta_bias_sigma | density | pvbm | -0.4531 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 0 | delta_bias_sigma | density | pvbm | -0.4054 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | 0 | delta_bias_sigma | density | skan | -0.4531 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 0 | delta_bias_sigma | density | skan | -0.4054 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 1 | delta_bias_sigma | density | pvbm | -0.4366 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 1 | delta_bias_sigma | density | skan | -0.4366 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | 2 | delta_bias_sigma | density | pvbm | -0.3390 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 2 | delta_bias_sigma | density | pvbm | -0.4287 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | 2 | delta_bias_sigma | density | skan | -0.3390 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 2 | delta_bias_sigma | density | skan | -0.4287 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | pooled(0,1,2) | delta_bias_sigma | density | pvbm | -0.3466 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | pooled(0,1,2) | delta_bias_sigma | density | pvbm | -0.4235 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | pooled(0,1,2) | delta_bias_sigma | density | skan | -0.3466 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | pooled(0,1,2) | delta_bias_sigma | density | skan | -0.4235 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | 0 | delta_bias_sigma | tortuosity | pvbm | 4.3089 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 0 | delta_bias_sigma | tortuosity | pvbm | 4.1734 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | 0 | delta_bias_sigma | tortuosity | skan | 0.7960 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 0 | delta_bias_sigma | tortuosity | skan | 0.6177 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | 1 | delta_bias_sigma | tortuosity | pvbm | 1.1328 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 1 | delta_bias_sigma | tortuosity | pvbm | 1.2153 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | 2 | delta_bias_sigma | tortuosity | pvbm | 1.6402 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 2 | delta_bias_sigma | tortuosity | pvbm | 1.7952 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | 2 | delta_bias_sigma | tortuosity | skan | 0.4157 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | 2 | delta_bias_sigma | tortuosity | skan | 0.5693 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | pooled(0,1,2) | delta_bias_sigma | tortuosity | pvbm | 2.3606 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | pooled(0,1,2) | delta_bias_sigma | tortuosity | pvbm | 2.3946 |  |  | 0.2500 | BREACH |
| reliseg_nocl | baseline | pooled(0,1,2) | delta_bias_sigma | tortuosity | skan | 0.4180 |  |  | 0.2500 | BREACH |
| reliseg_nocl | continued | pooled(0,1,2) | delta_bias_sigma | tortuosity | skan | 0.4611 |  |  | 0.2500 | BREACH |
| cfloss | baseline | 0 | delta_cldice | pixel | - | -0.0254 | -0.0318 | -0.0186 | -0.0100 | BREACH |
| cfloss | baseline | 1 | delta_cldice | pixel | - | -0.0231 | -0.0299 | -0.0163 | -0.0100 | BREACH |
| cfloss | baseline | 2 | delta_cldice | pixel | - | -0.0214 | -0.0269 | -0.0157 | -0.0100 | BREACH |
| reliseg_nocl | baseline | 0 | delta_cldice | pixel | - | -0.0183 | -0.0204 | -0.0162 | -0.0100 | BREACH |
| reliseg_nocl | baseline | 1 | delta_cldice | pixel | - | -0.0182 | -0.0229 | -0.0135 | -0.0100 | BREACH |
| reliseg_nocl | baseline | 2 | delta_cldice | pixel | - | -0.0169 | -0.0206 | -0.0132 | -0.0100 | BREACH |
| abl_cf_matched | baseline | 0 | delta_r | density | pvbm | -0.0441 | -0.0845 | 0.0034 | -0.0500 | AT-RISK |
| abl_cf_matched | continued | 0 | delta_r | density | pvbm | -0.0396 | -0.0824 | -0.0042 | -0.0500 | AT-RISK |
| abl_cf_matched | baseline | 0 | delta_r | density | skan | -0.0441 | -0.0845 | 0.0034 | -0.0500 | AT-RISK |
| abl_cf_matched | continued | 0 | delta_r | density | skan | -0.0396 | -0.0824 | -0.0042 | -0.0500 | AT-RISK |
| abl_cf_matched | baseline | 0 | delta_r | tortuosity | pvbm | -0.0362 | -0.1647 | 0.0485 | -0.0500 | AT-RISK |
| abl_cf_matched | continued | 0 | delta_r | tortuosity | pvbm | 0.0130 | -0.0556 | 0.1288 | -0.0500 | AT-RISK |
| abl_cf_matched | baseline | 0 | delta_r | tortuosity | skan | -0.0343 | -0.1647 | 0.0248 | -0.0500 | AT-RISK |
| abl_cf_matched | continued | 0 | delta_r | tortuosity | skan | 0.0180 | -0.1171 | 0.1374 | -0.0500 | AT-RISK |
| abl_cf_on_base | baseline | 0 | delta_r | density | pvbm | -0.0126 | -0.0606 | 0.0426 | -0.0500 | AT-RISK |
| abl_cf_on_base | baseline | 0 | delta_r | density | skan | -0.0126 | -0.0606 | 0.0426 | -0.0500 | AT-RISK |
| abl_cf_on_base | baseline | 0 | delta_r | tortuosity | pvbm | -0.0375 | -0.1538 | 0.0158 | -0.0500 | AT-RISK |
| abl_cf_on_base | continued | 0 | delta_r | tortuosity | pvbm | 0.0118 | -0.0599 | 0.1034 | -0.0500 | AT-RISK |
| abl_cf_on_base | baseline | 0 | delta_r | tortuosity | skan | 0.0114 | -0.1156 | 0.1531 | -0.0500 | AT-RISK |
| abl_density_only | baseline | 0 | delta_r | density | pvbm | -0.0229 | -0.0767 | 0.0416 | -0.0500 | AT-RISK |
| abl_density_only | baseline | 0 | delta_r | density | skan | -0.0229 | -0.0767 | 0.0416 | -0.0500 | AT-RISK |
| abl_density_only | baseline | 0 | delta_r | tortuosity | pvbm | -0.0012 | -0.0926 | 0.0972 | -0.0500 | AT-RISK |
| abl_density_only | baseline | 0 | delta_r | tortuosity | skan | 0.0102 | -0.1308 | 0.1428 | -0.0500 | AT-RISK |
| abl_density_only | continued | 0 | delta_r | tortuosity | skan | 0.0626 | -0.1202 | 0.2656 | -0.0500 | AT-RISK |
| abl_fd_only | baseline | 0 | delta_r | density | pvbm | -0.0119 | -0.0660 | 0.0520 | -0.0500 | AT-RISK |
| abl_fd_only | baseline | 0 | delta_r | density | skan | -0.0119 | -0.0660 | 0.0520 | -0.0500 | AT-RISK |
| abl_fd_only | baseline | 0 | delta_r | tortuosity | pvbm | 0.0053 | -0.0658 | 0.1028 | -0.0500 | AT-RISK |
| abl_fd_only | baseline | 0 | delta_r | tortuosity | skan | 0.0410 | -0.0594 | 0.2191 | -0.0500 | AT-RISK |
| abl_fd_only | continued | 0 | delta_r | tortuosity | skan | 0.0933 | -0.0733 | 0.3747 | -0.0500 | AT-RISK |
| abl_fixed_ladder | baseline | 0 | delta_r | density | pvbm | -0.0417 | -0.0878 | 0.0169 | -0.0500 | AT-RISK |
| abl_fixed_ladder | continued | 0 | delta_r | density | pvbm | -0.0372 | -0.0677 | -0.0066 | -0.0500 | AT-RISK |
| abl_fixed_ladder | baseline | 0 | delta_r | density | skan | -0.0417 | -0.0878 | 0.0169 | -0.0500 | AT-RISK |
| abl_fixed_ladder | continued | 0 | delta_r | density | skan | -0.0372 | -0.0677 | -0.0066 | -0.0500 | AT-RISK |
| abl_fixed_ladder | baseline | 0 | delta_r | tortuosity | pvbm | -0.6497 | -1.0140 | -0.3311 | -0.0500 | BREACH |
| abl_fixed_ladder | continued | 0 | delta_r | tortuosity | pvbm | -0.6004 | -0.9133 | -0.2774 | -0.0500 | BREACH |
| abl_fixed_ladder | baseline | 0 | delta_r | tortuosity | skan | -0.0821 | -0.2524 | -0.0030 | -0.0500 | BREACH |
| abl_fixed_ladder | continued | 0 | delta_r | tortuosity | skan | -0.0298 | -0.1862 | 0.0730 | -0.0500 | AT-RISK |
| abl_length_only | baseline | 0 | delta_r | density | pvbm | 0.0001 | -0.0584 | 0.0725 | -0.0500 | AT-RISK |
| abl_length_only | baseline | 0 | delta_r | density | skan | 0.0001 | -0.0584 | 0.0725 | -0.0500 | AT-RISK |
| abl_length_only | baseline | 0 | delta_r | tortuosity | pvbm | -0.0702 | -0.3250 | 0.0649 | -0.0500 | BREACH |
| abl_length_only | continued | 0 | delta_r | tortuosity | pvbm | -0.0209 | -0.1936 | 0.1169 | -0.0500 | AT-RISK |
| abl_length_only | baseline | 0 | delta_r | tortuosity | skan | -0.0145 | -0.1256 | 0.0627 | -0.0500 | AT-RISK |
| abl_length_only | continued | 0 | delta_r | tortuosity | skan | 0.0379 | -0.0664 | 0.1789 | -0.0500 | AT-RISK |
| abl_no_density | baseline | 0 | delta_r | density | pvbm | -0.0185 | -0.0725 | 0.0461 | -0.0500 | AT-RISK |
| abl_no_density | baseline | 0 | delta_r | density | skan | -0.0185 | -0.0725 | 0.0461 | -0.0500 | AT-RISK |
| abl_no_density | baseline | 0 | delta_r | tortuosity | pvbm | -0.6606 | -0.9569 | -0.3380 | -0.0500 | BREACH |
| abl_no_density | continued | 0 | delta_r | tortuosity | pvbm | -0.6113 | -0.8729 | -0.2639 | -0.0500 | BREACH |
| abl_no_density | baseline | 0 | delta_r | tortuosity | skan | -0.0928 | -0.3015 | 0.0240 | -0.0500 | BREACH |
| abl_no_density | continued | 0 | delta_r | tortuosity | skan | -0.0404 | -0.2502 | 0.1380 | -0.0500 | AT-RISK |
| abl_no_fd | baseline | 0 | delta_r | density | pvbm | -0.0225 | -0.0737 | 0.0358 | -0.0500 | AT-RISK |
| abl_no_fd | baseline | 0 | delta_r | density | skan | -0.0225 | -0.0737 | 0.0358 | -0.0500 | AT-RISK |
| abl_no_fd | baseline | 0 | delta_r | tortuosity | pvbm | -0.0446 | -0.2417 | 0.0763 | -0.0500 | AT-RISK |
| abl_no_fd | continued | 0 | delta_r | tortuosity | pvbm | 0.0047 | -0.1298 | 0.1821 | -0.0500 | AT-RISK |
| abl_no_fd | baseline | 0 | delta_r | tortuosity | skan | -0.0237 | -0.2538 | 0.1897 | -0.0500 | AT-RISK |
| abl_no_fd | continued | 0 | delta_r | tortuosity | skan | 0.0286 | -0.2531 | 0.3265 | -0.0500 | AT-RISK |
| abl_no_length | baseline | 0 | delta_r | density | pvbm | -0.0291 | -0.0888 | 0.0451 | -0.0500 | AT-RISK |
| abl_no_length | baseline | 0 | delta_r | density | skan | -0.0291 | -0.0888 | 0.0451 | -0.0500 | AT-RISK |
| abl_no_length | baseline | 0 | delta_r | tortuosity | skan | 0.0050 | -0.1450 | 0.1558 | -0.0500 | AT-RISK |
| abl_no_length | continued | 0 | delta_r | tortuosity | skan | 0.0573 | -0.1318 | 0.2830 | -0.0500 | AT-RISK |
| cfloss | baseline | 0 | delta_r | density | pvbm | -0.0468 | -0.0918 | -0.0006 | -0.0500 | AT-RISK |
| cfloss | continued | 0 | delta_r | density | pvbm | -0.0424 | -0.0788 | -0.0091 | -0.0500 | AT-RISK |
| cfloss | baseline | 0 | delta_r | density | skan | -0.0468 | -0.0918 | -0.0006 | -0.0500 | AT-RISK |
| cfloss | continued | 0 | delta_r | density | skan | -0.0424 | -0.0788 | -0.0091 | -0.0500 | AT-RISK |
| cfloss | baseline | 1 | delta_r | density | pvbm | 0.0036 | -0.0810 | 0.0782 | -0.0500 | AT-RISK |
| cfloss | continued | 1 | delta_r | density | pvbm | -0.0292 | -0.0824 | 0.0187 | -0.0500 | AT-RISK |
| cfloss | baseline | 1 | delta_r | density | skan | 0.0036 | -0.0810 | 0.0782 | -0.0500 | AT-RISK |
| cfloss | continued | 1 | delta_r | density | skan | -0.0292 | -0.0824 | 0.0187 | -0.0500 | AT-RISK |
| cfloss | baseline | 2 | delta_r | density | pvbm | -0.0176 | -0.0794 | 0.0375 | -0.0500 | AT-RISK |
| cfloss | continued | 2 | delta_r | density | pvbm | -0.0381 | -0.0751 | -0.0069 | -0.0500 | AT-RISK |
| cfloss | baseline | 2 | delta_r | density | skan | -0.0176 | -0.0794 | 0.0375 | -0.0500 | AT-RISK |
| cfloss | continued | 2 | delta_r | density | skan | -0.0381 | -0.0751 | -0.0069 | -0.0500 | AT-RISK |
| cfloss | baseline | pooled(0,1,2) | delta_r | density | pvbm | -0.0203 | -0.0611 | 0.0177 | -0.0500 | AT-RISK |
| cfloss | continued | pooled(0,1,2) | delta_r | density | pvbm | -0.0366 | -0.0738 | -0.0020 | -0.0500 | AT-RISK |
| cfloss | baseline | pooled(0,1,2) | delta_r | density | skan | -0.0203 | -0.0611 | 0.0177 | -0.0500 | AT-RISK |
| cfloss | continued | pooled(0,1,2) | delta_r | density | skan | -0.0366 | -0.0738 | -0.0020 | -0.0500 | AT-RISK |
| cfloss | baseline | 0 | delta_r | tortuosity | pvbm | -0.0770 | -0.2167 | 0.0067 | -0.0500 | BREACH |
| cfloss | continued | 0 | delta_r | tortuosity | pvbm | -0.0277 | -0.1267 | 0.0985 | -0.0500 | AT-RISK |
| cfloss | baseline | 0 | delta_r | tortuosity | skan | 0.0343 | -0.1536 | 0.1802 | -0.0500 | AT-RISK |
| cfloss | continued | 0 | delta_r | tortuosity | skan | 0.0866 | -0.1680 | 0.3279 | -0.0500 | AT-RISK |
| cfloss | baseline | 1 | delta_r | tortuosity | pvbm | -0.0895 | -0.3097 | 0.0008 | -0.0500 | BREACH |
| cfloss | continued | 1 | delta_r | tortuosity | pvbm | -0.1040 | -0.3365 | -0.0203 | -0.0500 | BREACH |
| cfloss | baseline | 1 | delta_r | tortuosity | skan | -0.0092 | -0.1969 | 0.1379 | -0.0500 | AT-RISK |
| cfloss | continued | 1 | delta_r | tortuosity | skan | -0.0527 | -0.2076 | 0.0748 | -0.0500 | BREACH |
| cfloss | baseline | 2 | delta_r | tortuosity | pvbm | -0.0439 | -0.1685 | 0.0211 | -0.0500 | AT-RISK |
| cfloss | continued | 2 | delta_r | tortuosity | pvbm | -0.0624 | -0.2579 | 0.0566 | -0.0500 | BREACH |
| cfloss | baseline | 2 | delta_r | tortuosity | skan | -0.1800 | -0.3765 | -0.0360 | -0.0500 | BREACH |
| cfloss | continued | 2 | delta_r | tortuosity | skan | -0.0173 | -0.0800 | 0.0321 | -0.0500 | AT-RISK |
| cfloss | baseline | pooled(0,1,2) | delta_r | tortuosity | pvbm | -0.0701 | -0.1914 | -0.0164 | -0.0500 | BREACH |
| cfloss | continued | pooled(0,1,2) | delta_r | tortuosity | pvbm | -0.0647 | -0.1817 | -0.0105 | -0.0500 | BREACH |
| cfloss | baseline | pooled(0,1,2) | delta_r | tortuosity | skan | -0.0516 | -0.1646 | 0.0165 | -0.0500 | BREACH |
| cfloss | continued | pooled(0,1,2) | delta_r | tortuosity | skan | 0.0055 | -0.0939 | 0.0866 | -0.0500 | AT-RISK |
| continued | baseline | 0 | delta_r | density | pvbm | -0.0045 | -0.0600 | 0.0606 | -0.0500 | AT-RISK |
| continued | baseline | 0 | delta_r | density | skan | -0.0045 | -0.0600 | 0.0606 | -0.0500 | AT-RISK |
| continued | baseline | 0 | delta_r | tortuosity | pvbm | -0.0493 | -0.2026 | 0.0169 | -0.0500 | AT-RISK |
| continued | baseline | 0 | delta_r | tortuosity | skan | -0.0523 | -0.1980 | 0.0404 | -0.0500 | BREACH |
| continued | baseline | 1 | delta_r | tortuosity | pvbm | 0.0145 | -0.0582 | 0.1345 | -0.0500 | AT-RISK |
| continued | baseline | 1 | delta_r | tortuosity | skan | 0.0435 | -0.0550 | 0.1690 | -0.0500 | AT-RISK |
| continued | baseline | 2 | delta_r | tortuosity | pvbm | 0.0185 | -0.1066 | 0.1709 | -0.0500 | AT-RISK |
| continued | baseline | 2 | delta_r | tortuosity | skan | -0.1627 | -0.3678 | -0.0012 | -0.0500 | BREACH |
| continued | baseline | pooled(0,1,2) | delta_r | tortuosity | pvbm | -0.0054 | -0.0677 | 0.0474 | -0.0500 | AT-RISK |
| continued | baseline | pooled(0,1,2) | delta_r | tortuosity | skan | -0.0572 | -0.1787 | 0.0329 | -0.0500 | BREACH |
| reliseg | baseline | 0 | delta_r | density | pvbm | -0.0277 | -0.0727 | 0.0295 | -0.0500 | AT-RISK |
| reliseg | baseline | 0 | delta_r | density | skan | -0.0277 | -0.0727 | 0.0295 | -0.0500 | AT-RISK |
| reliseg | baseline | 2 | delta_r | density | pvbm | 0.0147 | -0.0602 | 0.0922 | -0.0500 | AT-RISK |
| reliseg | baseline | 2 | delta_r | density | skan | 0.0147 | -0.0602 | 0.0922 | -0.0500 | AT-RISK |
| reliseg | baseline | 0 | delta_r | tortuosity | pvbm | -0.7403 | -1.0315 | -0.4083 | -0.0500 | BREACH |
| reliseg | continued | 0 | delta_r | tortuosity | pvbm | -0.6910 | -0.9539 | -0.3483 | -0.0500 | BREACH |
| reliseg | baseline | 0 | delta_r | tortuosity | skan | -0.0358 | -0.2666 | 0.1303 | -0.0500 | AT-RISK |
| reliseg | continued | 0 | delta_r | tortuosity | skan | 0.0165 | -0.2681 | 0.2882 | -0.0500 | AT-RISK |
| reliseg | baseline | 1 | delta_r | tortuosity | pvbm | -0.0121 | -0.1666 | 0.0946 | -0.0500 | AT-RISK |
| reliseg | continued | 1 | delta_r | tortuosity | pvbm | -0.0266 | -0.1459 | 0.0176 | -0.0500 | AT-RISK |
| reliseg | continued | 1 | delta_r | tortuosity | skan | 0.0305 | -0.1307 | 0.2397 | -0.0500 | AT-RISK |
| reliseg | baseline | 2 | delta_r | tortuosity | pvbm | 0.0047 | -0.0937 | 0.1341 | -0.0500 | AT-RISK |
| reliseg | continued | 2 | delta_r | tortuosity | pvbm | -0.0139 | -0.1576 | 0.1441 | -0.0500 | AT-RISK |
| reliseg | baseline | 2 | delta_r | tortuosity | skan | -0.1324 | -0.3091 | -0.0269 | -0.0500 | BREACH |
| reliseg | continued | 2 | delta_r | tortuosity | skan | 0.0303 | -0.0744 | 0.1264 | -0.0500 | AT-RISK |
| reliseg | baseline | pooled(0,1,2) | delta_r | tortuosity | pvbm | -0.2492 | -0.3957 | -0.1026 | -0.0500 | BREACH |
| reliseg | continued | pooled(0,1,2) | delta_r | tortuosity | pvbm | -0.2438 | -0.3745 | -0.1031 | -0.0500 | BREACH |
| reliseg | baseline | pooled(0,1,2) | delta_r | tortuosity | skan | -0.0314 | -0.1301 | 0.0254 | -0.0500 | AT-RISK |
| reliseg | continued | pooled(0,1,2) | delta_r | tortuosity | skan | 0.0258 | -0.0994 | 0.1488 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | 0 | delta_r | density | pvbm | -0.0244 | -0.0755 | 0.0365 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | 0 | delta_r | density | skan | -0.0244 | -0.0755 | 0.0365 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | 2 | delta_r | density | pvbm | 0.0033 | -0.0773 | 0.0848 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | 2 | delta_r | density | skan | 0.0033 | -0.0773 | 0.0848 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | 0 | delta_r | tortuosity | pvbm | -0.1191 | -0.3399 | 0.0015 | -0.0500 | BREACH |
| reliseg_nocl | continued | 0 | delta_r | tortuosity | pvbm | -0.0699 | -0.2259 | 0.0603 | -0.0500 | BREACH |
| reliseg_nocl | baseline | 0 | delta_r | tortuosity | skan | -0.0837 | -0.2671 | -0.0121 | -0.0500 | BREACH |
| reliseg_nocl | continued | 0 | delta_r | tortuosity | skan | -0.0314 | -0.1743 | 0.0360 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | 1 | delta_r | tortuosity | pvbm | 0.0166 | -0.0647 | 0.1388 | -0.0500 | AT-RISK |
| reliseg_nocl | continued | 1 | delta_r | tortuosity | pvbm | 0.0021 | -0.0801 | 0.0731 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | 2 | delta_r | tortuosity | pvbm | -0.0141 | -0.1490 | 0.0958 | -0.0500 | AT-RISK |
| reliseg_nocl | continued | 2 | delta_r | tortuosity | pvbm | -0.0327 | -0.1899 | 0.0834 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | 2 | delta_r | tortuosity | skan | -0.1560 | -0.4166 | -0.0207 | -0.0500 | BREACH |
| reliseg_nocl | continued | 2 | delta_r | tortuosity | skan | 0.0067 | -0.1248 | 0.1483 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | pooled(0,1,2) | delta_r | tortuosity | pvbm | -0.0389 | -0.1376 | 0.0273 | -0.0500 | AT-RISK |
| reliseg_nocl | continued | pooled(0,1,2) | delta_r | tortuosity | pvbm | -0.0335 | -0.1185 | 0.0250 | -0.0500 | AT-RISK |
| reliseg_nocl | baseline | pooled(0,1,2) | delta_r | tortuosity | skan | -0.0400 | -0.1643 | 0.0298 | -0.0500 | AT-RISK |
| reliseg_nocl | continued | pooled(0,1,2) | delta_r | tortuosity | skan | 0.0172 | -0.0600 | 0.0857 | -0.0500 | AT-RISK |

The clDice rows are complete for the whole grid already: they come from `pred*/pixel_metrics.csv`, written by the inference stage, and do not wait on the CPU biomarker stage.

## 7. Ablations (HRF seed 0 only -- development set, skan)
Leave-one-out (`abl_no_*`) says which term is dispensable; single-term (`abl_*_only`) says which term carries the effect. Reference: the same-seed baseline.

| config | biomarker | d_r | lo | hi |
|---|---|---|---|---|
| abl_cf_matched | FD | 0.006 | -0.055 | 0.074 |
| abl_cf_matched | density | -0.044 | -0.084 | 0.003 |
| abl_cf_matched | tortuosity | -0.034 | -0.165 | 0.025 |
| abl_cf_matched | total_length | 0.012 | -0.049 | 0.076 |
| abl_cf_on_base | FD | 0.044 | -0.028 | 0.118 |
| abl_cf_on_base | density | -0.013 | -0.061 | 0.043 |
| abl_cf_on_base | tortuosity | 0.011 | -0.116 | 0.153 |
| abl_cf_on_base | total_length | 0.049 | -0.033 | 0.128 |
| abl_density_only | FD | 0.032 | -0.051 | 0.118 |
| abl_density_only | density | -0.023 | -0.077 | 0.042 |
| abl_density_only | tortuosity | 0.010 | -0.131 | 0.143 |
| abl_density_only | total_length | 0.032 | -0.051 | 0.113 |
| abl_fd_only | FD | 0.095 | 0.029 | 0.171 |
| abl_fd_only | density | -0.012 | -0.066 | 0.052 |
| abl_fd_only | tortuosity | 0.041 | -0.059 | 0.219 |
| abl_fd_only | total_length | 0.089 | 0.021 | 0.163 |
| abl_fixed_ladder | FD | 0.134 | 0.027 | 0.252 |
| abl_fixed_ladder | density | -0.042 | -0.088 | 0.017 |
| abl_fixed_ladder | tortuosity | -0.082 | -0.252 | -0.003 |
| abl_fixed_ladder | total_length | 0.117 | -0.002 | 0.242 |
| abl_length_only | FD | 0.163 | 0.089 | 0.238 |
| abl_length_only | density | 0.000 | -0.058 | 0.073 |
| abl_length_only | tortuosity | -0.014 | -0.126 | 0.063 |
| abl_length_only | total_length | 0.151 | 0.080 | 0.222 |
| abl_no_density | FD | 0.198 | 0.056 | 0.354 |
| abl_no_density | density | -0.018 | -0.072 | 0.046 |
| abl_no_density | tortuosity | -0.093 | -0.302 | 0.024 |
| abl_no_density | total_length | 0.199 | 0.048 | 0.364 |
| abl_no_fd | FD | 0.075 | 0.013 | 0.140 |
| abl_no_fd | density | -0.023 | -0.074 | 0.036 |
| abl_no_fd | tortuosity | -0.024 | -0.254 | 0.190 |
| abl_no_fd | total_length | 0.055 | -0.004 | 0.116 |
| abl_no_length | FD | 0.048 | -0.029 | 0.125 |
| abl_no_length | density | -0.029 | -0.089 | 0.045 |
| abl_no_length | tortuosity | 0.005 | -0.145 | 0.156 |
| abl_no_length | total_length | 0.047 | -0.031 | 0.122 |

## 8. Sensitivity -- the PVBM pipeline (FIVES, `last.pt`)
The same primary contrast measured with the second biomarker pipeline. Agreement with section 1 is the check that the effect is a property of the segmentation, not of one estimator.

| config | biomarker | d_r | lo | hi | direction_consistent |
|---|---|---|---|---|---|
| cfloss | FD | -0.023 | -0.034 | 0.001 | yes |
| cfloss | density | -0.001 | -0.002 | 0.000 | yes |
| cfloss | tortuosity | 0.057 | -0.053 | 0.115 | yes |
| cfloss | total_length | -0.001 | -0.003 | 0.000 | no |
| continued | FD | 0.007 | -0.000 | 0.011 | no |
| continued | density | 0.001 | 0.000 | 0.002 | yes |
| continued | tortuosity | -0.023 | -0.051 | 0.055 | no |
| continued | total_length | 0.002 | 0.000 | 0.003 | no |
| reliseg | FD | 0.007 | -0.000 | 0.010 | no |
| reliseg | density | 0.001 | 0.000 | 0.002 | yes |
| reliseg | tortuosity | -0.013 | -0.085 | 0.055 | no |
| reliseg | total_length | 0.004 | 0.002 | 0.006 | no |
| reliseg_nocl | FD | 0.005 | -0.002 | 0.009 | yes |
| reliseg_nocl | density | 0.000 | -0.000 | 0.001 | no |
| reliseg_nocl | tortuosity | -0.003 | -0.048 | 0.075 | no |
| reliseg_nocl | total_length | 0.002 | 0.001 | 0.005 | no |
