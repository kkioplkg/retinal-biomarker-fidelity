# E4 -- independent replication (Fundus-AVSeg)

_generated 2026-09-18 22:29:08 by `src/pivot/e4_replication.py analyse`_

Pre-registered in `exp/DECISIONS.md` 2026-09-17 13:15 and clarified 13:30, both before any out-of-fold result was computed or looked at.  Timeline, stated plainly: E4 is a **prospectively locked replication run after E2**, not part of the project's original pre-registration; Fundus-AVSeg was chosen after the FIVES null result was seen.

## 0. Folds
Source: `results/pivot/e4/folds.json` (generator `src/pivot/e4_replication.py::build_folds`).

```
The pre-registered 5-fold partition of Fundus-AVSeg.

    Definition (deterministic, reproducible from this function alone):

    1. Records are ``src.data.datasets.load_dataset('fundusavseg')``, 100
       images, each with a disease label in {Normal, DR, AMD, Glaucoma}
       (40 / 20 / 20 / 20) taken from the filename code, and an official
       ``split`` from the deposit's ``training.txt`` / ``testing.txt``.
    2. **Fold 0** = the 20 images of the official *test* split, sorted by
       ``image_id``.  This makes the replication's first fold the dataset's own
       published protocol.
    3. The remaining 80 (the official *train* split) are dealt to folds 1-4:
       for each disease class in sorted order (AMD, DR, Glaucoma, Normal) the
       class members are sorted by ``image_id`` and shuffled with
       ``random.Random(2026)``; the shuffled members are handed out one at a
       time to folds ``1 + (counter % 4)`` where ``counter`` is a single
       running position that **continues across classes**.  Dealing 80 images
       this way gives every fold exactly 20 images, and within a class the
       fold counts differ by at most one -- i.e. a stratified random
       partition.
    4. Assertion: the five held-out sets are pairwise disjoint and their union
       is all 100 images, so every image is held out exactly once.

    For fold ``k`` the *training pool* is the other 80 images; the 15 % by
    subject validation split is then drawn from that pool with the project's
    standard ``DEFAULT_SPLIT_SEED = 12345``, exactly as a normal
    ``src.seg.train`` run does.
```

| fold | n | class counts |
|---|---|---|
| 0 | 20 | {'AMD': 2, 'DR': 4, 'Glaucoma': 5, 'Normal': 9} |
| 1 | 20 | {'AMD': 5, 'DR': 4, 'Glaucoma': 4, 'Normal': 7} |
| 2 | 20 | {'AMD': 5, 'DR': 4, 'Glaucoma': 3, 'Normal': 8} |
| 3 | 20 | {'AMD': 4, 'DR': 4, 'Glaucoma': 4, 'Normal': 8} |
| 4 | 20 | {'AMD': 4, 'DR': 4, 'Glaucoma': 4, 'Normal': 8} |

The 80 non-official-test images are partitioned ONCE into four disjoint 20-image folds; nothing is resampled or re-drawn.

**Cross-validation unit: image.** Image-level cross-validation.  Fundus-AVSeg publishes no patient identifier -- only a left/right eye flag -- so two images of the same patient cannot be detected and both-eye leakage across folds cannot be excluded.  This is stated as a limitation rather than silently assumed away (DECISIONS 2026-09-17 13:30).

Every image is held out exactly once; 100 out-of-fold predictions per arm.

## 1. Sigma
per outer fold: 1.4826*MAD of the reference-mask biomarkers of that fold's 80 TRAINING images, applied to its 20 held-out images (results/pivot/e4_gt_fundusavseg.csv); no Gate A train scale exists for Fundus-AVSeg (DECISIONS 2026-09-17 13:30 (4))

| biomarker | fold 0 | fold 1 | fold 2 | fold 3 | fold 4 |
|---|---|---|---|---|---|
| FD_pvbm | 0.0184067 | 0.0185159 | 0.0174548 | 0.0187547 | 0.0166902 |
| FD_skan | 0.0229666 | 0.0202906 | 0.0213762 | 0.0229193 | 0.0224017 |
| density_pvbm | 0.00899398 | 0.00819082 | 0.00868458 | 0.00867571 | 0.00834674 |
| density_skan | 0.00899398 | 0.00819082 | 0.00868458 | 0.00867571 | 0.00834674 |
| tortuosity_pvbm | 0.00504991 | 0.00476991 | 0.00499739 | 0.00524877 | 0.00570867 |
| tortuosity_skan | 0.00816866 | 0.00807566 | 0.00791223 | 0.00896685 | 0.00787061 |
| total_length_pvbm | 2832.1 | 2470.55 | 2510.21 | 2615.75 | 2171.48 |
| total_length_skan | 2970.58 | 2460.01 | 2496.68 | 2819.63 | 2244.87 |

Dice, clDice, delta-r and delta-AUC are reported in their own units and are never divided by sigma.

## 2. Measurement fidelity (out-of-fold, pooled)
| biomarker | config | n | r_pearson | r_spearman | bias_const_sigma | resid_sd_sigma | mae_sigma |
|---|---|---|---|---|---|---|---|
| FD | baseline | 100 | 0.899 | 0.896 | -0.309 | 0.425 | 0.376 |
| FD | continued | 100 | 0.902 | 0.895 | -0.287 | 0.421 | 0.367 |
| FD | reliseg | 100 | 0.898 | 0.892 | 0.007 | 0.391 | 0.302 |
| density | baseline | 100 | 0.940 | 0.936 | -0.141 | 0.375 | 0.301 |
| density | continued | 100 | 0.940 | 0.935 | -0.023 | 0.374 | 0.295 |
| density | reliseg | 100 | 0.937 | 0.931 | -0.004 | 0.381 | 0.303 |
| tortuosity | baseline | 100 | 0.967 | 0.923 | -0.129 | 0.433 | 0.311 |
| tortuosity | continued | 100 | 0.965 | 0.910 | -0.098 | 0.446 | 0.326 |
| tortuosity | reliseg | 100 | 0.891 | 0.714 | 0.027 | 0.773 | 0.548 |
| total_length | baseline | 100 | 0.987 | 0.967 | -0.363 | 0.643 | 0.436 |
| total_length | continued | 100 | 0.987 | 0.964 | -0.331 | 0.627 | 0.421 |
| total_length | reliseg | 100 | 0.986 | 0.964 | 0.121 | 0.579 | 0.386 |

Disease-class-adjusted sensitivity (within-class centred r, `scope=class_adjusted` in the CSV):

| biomarker | config | n | r_pearson | r_spearman |
|---|---|---|---|---|
| FD | baseline | 100 | 0.902 | 0.887 |
| FD | continued | 100 | 0.905 | 0.893 |
| FD | reliseg | 100 | 0.899 | 0.892 |
| density | baseline | 100 | 0.939 | 0.932 |
| density | continued | 100 | 0.939 | 0.927 |
| density | reliseg | 100 | 0.936 | 0.924 |
| tortuosity | baseline | 100 | 0.966 | 0.927 |
| tortuosity | continued | 100 | 0.964 | 0.918 |
| tortuosity | reliseg | 100 | 0.889 | 0.740 |
| total_length | baseline | 100 | 0.987 | 0.962 |
| total_length | continued | 100 | 0.987 | 0.960 |
| total_length | reliseg | 100 | 0.986 | 0.968 |

Source: `results/pivot/e4_fidelity.csv`, GT `results/pivot/e4_gt_fundusavseg.csv`.

## 3. Delta-r (primary endpoint)
paired bootstrap, 2000 resamples drawn WITHIN each outer fold (fold structure preserved); the interval is conditional on the fixed folds and the fitted models.  Primary contrast: **ReliSeg - continued** on the skan density / total_length / FD; ReliSeg - baseline is the secondary 'total deployment difference'.

| role | biomarker | config | reference | n | d_r | lo | hi | p_adj | d_bias_sigma |
|---|---|---|---|---|---|---|---|---|---|
| context | FD | continued | baseline | 100 | 0.003 | -0.003 | 0.009 |  | 0.022 |
| context | density | continued | baseline | 100 | -0.000 | -0.006 | 0.006 |  | 0.118 |
| context | tortuosity | continued | baseline | 100 | -0.002 | -0.012 | 0.004 |  | 0.031 |
| context | total_length | continued | baseline | 100 | -0.000 | -0.001 | 0.001 |  | 0.032 |
| primary | FD | reliseg | continued | 100 | -0.004 | -0.031 | 0.021 | 1.000 | 0.294 |
| primary | density | reliseg | continued | 100 | -0.003 | -0.006 | -0.001 | 1.000 | 0.020 |
| primary | tortuosity | reliseg | continued | 100 | -0.074 | -0.142 | -0.039 |  | 0.125 |
| primary | total_length | reliseg | continued | 100 | -0.001 | -0.003 | 0.002 | 0.916 | 0.452 |
| secondary | FD | reliseg | baseline | 100 | -0.001 | -0.028 | 0.024 |  | 0.316 |
| secondary | density | reliseg | baseline | 100 | -0.003 | -0.010 | 0.003 |  | 0.137 |
| secondary | tortuosity | reliseg | baseline | 100 | -0.076 | -0.144 | -0.040 |  | 0.155 |
| secondary | total_length | reliseg | baseline | 100 | -0.001 | -0.003 | 0.002 |  | 0.484 |

Leave-one-fold-out robustness (point estimates on the other four folds):

| biomarker | config | reference | scope | n | d_r |
|---|---|---|---|---|---|
| FD | reliseg | continued | leave_out_fold0 | 80 | -0.000 |
| FD | reliseg | continued | leave_out_fold1 | 80 | -0.013 |
| FD | reliseg | continued | leave_out_fold2 | 80 | 0.001 |
| FD | reliseg | continued | leave_out_fold3 | 80 | -0.005 |
| FD | reliseg | continued | leave_out_fold4 | 80 | -0.002 |
| density | reliseg | continued | leave_out_fold0 | 80 | -0.003 |
| density | reliseg | continued | leave_out_fold1 | 80 | -0.003 |
| density | reliseg | continued | leave_out_fold2 | 80 | -0.004 |
| density | reliseg | continued | leave_out_fold3 | 80 | -0.002 |
| density | reliseg | continued | leave_out_fold4 | 80 | -0.003 |
| tortuosity | reliseg | continued | leave_out_fold0 | 80 | -0.072 |
| tortuosity | reliseg | continued | leave_out_fold1 | 80 | -0.063 |
| tortuosity | reliseg | continued | leave_out_fold2 | 80 | -0.092 |
| tortuosity | reliseg | continued | leave_out_fold3 | 80 | -0.089 |
| tortuosity | reliseg | continued | leave_out_fold4 | 80 | -0.060 |
| total_length | reliseg | continued | leave_out_fold0 | 80 | -0.000 |
| total_length | reliseg | continued | leave_out_fold1 | 80 | 0.000 |
| total_length | reliseg | continued | leave_out_fold2 | 80 | -0.002 |
| total_length | reliseg | continued | leave_out_fold3 | 80 | -0.000 |
| total_length | reliseg | continued | leave_out_fold4 | 80 | 0.000 |

Source: `results/pivot/e4_delta.csv`.

## 3b. Multiplicity and the replication-success rule
Max-statistic (Westfall-Young) bootstrap over the three primary delta-r endpoints at alpha = 0.05.

| endpoint | delta-r 95%% CI | adjusted p |
|---|---|---|
| FD | [-0.031, +0.021] | 1.0000 |
| density | [-0.006, -0.001] | 1.0000 |
| total_length | [-0.003, +0.002] | 0.9160 |

**Replication success: NO** (significant after adjustment: none; endpoints running in the opposite direction: ['density']).  A CI that contains zero is not evidence of no effect -- no equivalence margin was pre-specified.

Source: `results/pivot/e4/replication_rule.json`.

## 4. Pixel metrics
| config | n | mean_dice | mean_cldice | d_dice_vs_baseline | d_cldice_vs_baseline | d_dice_vs_continued | d_cldice_vs_continued |
|---|---|---|---|---|---|---|---|
| baseline | 100 | 0.9375 | 0.9468 |  |  | 0.0002 | -0.0014 |
| continued | 100 | 0.9373 | 0.9482 | -0.0002 | 0.0014 |  |  |
| reliseg | 100 | 0.9363 | 0.9469 | -0.0012 | 0.0000 | -0.0010 | -0.0014 |

Source: `results/pivot/e4_pixel.csv`.

## 5. Safety endpoints
6 of 30 rows flagged.

| endpoint | config | reference | biomarker | pipeline | value | lo | hi | margin_value | flag |
|---|---|---|---|---|---|---|---|---|---|
| delta_bias_sigma | reliseg | continued | tortuosity | pvbm | 1.756 |  |  | 0.250 | BREACH |
| delta_bias_sigma | reliseg | baseline | tortuosity | pvbm | 1.742 |  |  | 0.250 | BREACH |
| delta_r | reliseg | continued | tortuosity | pvbm | -0.787 | -1.012 | -0.541 | -0.050 | BREACH |
| delta_r | reliseg | baseline | tortuosity | pvbm | -0.772 | -1.007 | -0.524 | -0.050 | BREACH |
| delta_r | reliseg | continued | tortuosity | skan | -0.074 | -0.142 | -0.039 | -0.050 | BREACH |
| delta_r | reliseg | baseline | tortuosity | skan | -0.076 | -0.144 | -0.040 | -0.050 | BREACH |

Source: `results/pivot/e4_safety.csv`; margins and flag logic are `src/pivot/e2_analysis.py::safety_table` unchanged.

## 6. Downstream four-class macro-AUC
| config | n | macro_auc | ci_lo | ci_hi | gt_minus_this | d_auc_vs_ref | d_auc_lo | d_auc_hi |
|---|---|---|---|---|---|---|---|---|
| baseline | 100 | 0.5583 | 0.4719 | 0.6378 | -0.0184 | 0.0101 | -0.0012 | 0.0223 |
| continued | 100 | 0.5482 | 0.4605 | 0.6255 | -0.0083 |  |  |  |
| gt | 100 | 0.5399 | 0.4598 | 0.6137 | 0.0000 |  |  |  |
| reliseg | 100 | 0.5573 | 0.4677 | 0.6318 | -0.0174 | 0.0091 | -0.0076 | 0.0273 |

Classifier outer folds are pinned 1:1 to the segmentation folds.  All feature sets / classifiers / references: `results/pivot/e4_downstream.csv`.

## 7. Pre-unblinding power
| biomarker | r_reference | true_delta_r | n | n_sim | power | mean_d_r |
|---|---|---|---|---|---|---|
| FD_skan | 0.911 | 0.050 | 100 | 400 | 1.000 | 0.050 |
| FD_skan | 0.911 | 0.050 | 100 | 400 | 0.995 | 0.049 |
| FD_skan | 0.911 | 0.050 | 100 | 400 | 0.983 | 0.050 |
| FD_skan | 0.295 | 0.050 | 100 | 400 | 0.223 | 0.047 |
| FD_skan | 0.295 | 0.050 | 100 | 400 | 0.083 | 0.044 |
| FD_skan | 0.295 | 0.050 | 100 | 400 | 0.058 | 0.050 |
| FD_skan | 0.911 | 0.100 | 100 | 400 | 1.000 | 0.089 |
| FD_skan | 0.911 | 0.100 | 100 | 400 | 1.000 | 0.089 |
| FD_skan | 0.911 | 0.100 | 100 | 400 | 1.000 | 0.089 |
| FD_skan | 0.295 | 0.100 | 100 | 400 | 0.632 | 0.097 |
| FD_skan | 0.295 | 0.100 | 100 | 400 | 0.195 | 0.095 |
| FD_skan | 0.295 | 0.100 | 100 | 400 | 0.158 | 0.104 |
| FD_skan | 0.911 | 0.150 | 100 | 400 | 1.000 | 0.089 |
| FD_skan | 0.911 | 0.150 | 100 | 400 | 1.000 | 0.088 |
| FD_skan | 0.911 | 0.150 | 100 | 400 | 1.000 | 0.087 |
| FD_skan | 0.295 | 0.150 | 100 | 400 | 0.932 | 0.146 |
| FD_skan | 0.295 | 0.150 | 100 | 400 | 0.468 | 0.152 |
| FD_skan | 0.295 | 0.150 | 100 | 400 | 0.258 | 0.146 |
| density_skan | 0.984 | 0.050 | 100 | 400 | 1.000 | 0.015 |
| density_skan | 0.984 | 0.050 | 100 | 400 | 1.000 | 0.015 |
| density_skan | 0.984 | 0.050 | 100 | 400 | 1.000 | 0.015 |
| density_skan | 0.658 | 0.050 | 100 | 400 | 0.968 | 0.052 |
| density_skan | 0.658 | 0.050 | 100 | 400 | 0.195 | 0.052 |
| density_skan | 0.658 | 0.050 | 100 | 400 | 0.125 | 0.050 |
| density_skan | 0.984 | 0.100 | 100 | 400 | 1.000 | 0.015 |
| density_skan | 0.984 | 0.100 | 100 | 400 | 1.000 | 0.015 |
| density_skan | 0.984 | 0.100 | 100 | 400 | 1.000 | 0.015 |
| density_skan | 0.658 | 0.100 | 100 | 400 | 1.000 | 0.100 |
| density_skan | 0.658 | 0.100 | 100 | 400 | 0.595 | 0.103 |
| density_skan | 0.658 | 0.100 | 100 | 400 | 0.405 | 0.100 |
| density_skan | 0.984 | 0.150 | 100 | 400 | 1.000 | 0.015 |
| density_skan | 0.984 | 0.150 | 100 | 400 | 1.000 | 0.015 |
| density_skan | 0.984 | 0.150 | 100 | 400 | 1.000 | 0.015 |
| density_skan | 0.658 | 0.150 | 100 | 400 | 1.000 | 0.149 |
| density_skan | 0.658 | 0.150 | 100 | 400 | 0.960 | 0.147 |
| density_skan | 0.658 | 0.150 | 100 | 400 | 0.765 | 0.151 |
| total_length_skan | 0.979 | 0.050 | 100 | 400 | 1.000 | 0.021 |
| total_length_skan | 0.979 | 0.050 | 100 | 400 | 1.000 | 0.020 |
| total_length_skan | 0.979 | 0.050 | 100 | 400 | 1.000 | 0.021 |
| total_length_skan | 0.305 | 0.050 | 100 | 400 | 0.265 | 0.053 |
| total_length_skan | 0.305 | 0.050 | 100 | 400 | 0.102 | 0.052 |
| total_length_skan | 0.305 | 0.050 | 100 | 400 | 0.072 | 0.058 |
| total_length_skan | 0.979 | 0.100 | 100 | 400 | 1.000 | 0.021 |
| total_length_skan | 0.979 | 0.100 | 100 | 400 | 1.000 | 0.020 |
| total_length_skan | 0.979 | 0.100 | 100 | 400 | 1.000 | 0.020 |
| total_length_skan | 0.305 | 0.100 | 100 | 400 | 0.593 | 0.098 |
| total_length_skan | 0.305 | 0.100 | 100 | 400 | 0.245 | 0.106 |
| total_length_skan | 0.305 | 0.100 | 100 | 400 | 0.155 | 0.109 |
| total_length_skan | 0.979 | 0.150 | 100 | 400 | 1.000 | 0.020 |
| total_length_skan | 0.979 | 0.150 | 100 | 400 | 1.000 | 0.020 |
| total_length_skan | 0.979 | 0.150 | 100 | 400 | 1.000 | 0.020 |
| total_length_skan | 0.305 | 0.150 | 100 | 400 | 0.922 | 0.146 |
| total_length_skan | 0.305 | 0.150 | 100 | 400 | 0.435 | 0.148 |
| total_length_skan | 0.305 | 0.150 | 100 | 400 | 0.310 | 0.155 |

Source: `results\pivot\e4_power.csv` (Monte-Carlo on the E2 per-image correlation structure, run before any E4 out-of-fold number existed).

## 8. MAPLES-DR zero-shot (secondary set)
Role: annotation-convention / domain robustness and a resolution effect-modification check only.  Claims are phrased as agreement with the **manually corrected** annotation convention; the >= 1380 px stratum is called *higher-resolution*, not high-resolution.

| stratum | source_dataset | seed | config | biomarker | n | d_r | lo | hi |
|---|---|---|---|---|---|---|---|---|
| all | fives | 0 | cfloss | FD | 162 | 0.001 | -0.005 | 0.009 |
| all | fives | 1 | cfloss | FD | 162 | 0.002 | -0.004 | 0.008 |
| all | fives | 2 | cfloss | FD | 162 | -0.007 | -0.014 | -0.001 |
| all | fives | 0 | reliseg | FD | 162 | -0.003 | -0.013 | 0.006 |
| all | fives | 1 | reliseg | FD | 162 | -0.008 | -0.020 | 0.002 |
| all | fives | 2 | reliseg | FD | 162 | -0.009 | -0.016 | -0.003 |
| all | fives | 0 | cfloss | density | 162 | -0.005 | -0.010 | -0.001 |
| all | fives | 1 | cfloss | density | 162 | -0.004 | -0.006 | -0.001 |
| all | fives | 2 | cfloss | density | 162 | 0.001 | -0.003 | 0.006 |
| all | fives | 0 | reliseg | density | 162 | -0.006 | -0.012 | -0.000 |
| all | fives | 1 | reliseg | density | 162 | -0.010 | -0.016 | -0.004 |
| all | fives | 2 | reliseg | density | 162 | 0.004 | -0.000 | 0.008 |
| all | fives | 0 | cfloss | tortuosity | 162 | -0.019 | -0.080 | 0.022 |
| all | fives | 1 | cfloss | tortuosity | 162 | -0.014 | -0.119 | 0.022 |
| all | fives | 2 | cfloss | tortuosity | 162 | 0.004 | -0.018 | 0.027 |
| all | fives | 0 | reliseg | tortuosity | 162 | -0.034 | -0.079 | 0.024 |
| all | fives | 1 | reliseg | tortuosity | 162 | -0.003 | -0.019 | 0.027 |
| all | fives | 2 | reliseg | tortuosity | 162 | -0.006 | -0.042 | 0.004 |
| all | fives | 0 | cfloss | total_length | 162 | -0.000 | -0.001 | 0.001 |
| all | fives | 1 | cfloss | total_length | 162 | -0.000 | -0.001 | 0.001 |
| all | fives | 2 | cfloss | total_length | 162 | -0.001 | -0.002 | -0.000 |
| all | fives | 0 | reliseg | total_length | 162 | -0.004 | -0.006 | -0.002 |
| all | fives | 1 | reliseg | total_length | 162 | -0.006 | -0.009 | -0.003 |
| all | fives | 2 | reliseg | total_length | 162 | -0.003 | -0.005 | -0.001 |
| all | hrf | 0 | cfloss | FD | 162 | -0.054 | -0.076 | -0.036 |
| all | hrf | 1 | cfloss | FD | 162 | -0.057 | -0.088 | -0.034 |
| all | hrf | 2 | cfloss | FD | 162 | -0.044 | -0.064 | -0.029 |
| all | hrf | 0 | reliseg | FD | 162 | -0.014 | -0.032 | 0.005 |
| all | hrf | 1 | reliseg | FD | 162 | 0.003 | -0.013 | 0.019 |
| all | hrf | 2 | reliseg | FD | 162 | -0.019 | -0.037 | -0.002 |
| all | hrf | 0 | cfloss | density | 162 | -0.051 | -0.072 | -0.034 |
| all | hrf | 1 | cfloss | density | 162 | -0.061 | -0.086 | -0.038 |
| all | hrf | 2 | cfloss | density | 162 | -0.065 | -0.086 | -0.045 |
| all | hrf | 0 | reliseg | density | 162 | -0.040 | -0.060 | -0.022 |
| all | hrf | 1 | reliseg | density | 162 | -0.013 | -0.027 | 0.001 |
| all | hrf | 2 | reliseg | density | 162 | -0.030 | -0.044 | -0.017 |
| all | hrf | 0 | cfloss | tortuosity | 162 | -0.035 | -0.122 | 0.055 |
| all | hrf | 1 | cfloss | tortuosity | 162 | -0.417 | -0.682 | 0.168 |
| all | hrf | 2 | cfloss | tortuosity | 162 | 0.070 | -0.075 | 0.190 |
| all | hrf | 0 | reliseg | tortuosity | 162 | 0.023 | -0.104 | 0.113 |
| all | hrf | 1 | reliseg | tortuosity | 162 | -0.525 | -0.832 | -0.006 |
| all | hrf | 2 | reliseg | tortuosity | 162 | -0.017 | -0.072 | 0.020 |
| all | hrf | 0 | cfloss | total_length | 162 | -0.018 | -0.026 | -0.012 |
| all | hrf | 1 | cfloss | total_length | 162 | -0.018 | -0.026 | -0.011 |
| all | hrf | 2 | cfloss | total_length | 162 | -0.015 | -0.020 | -0.010 |
| all | hrf | 0 | reliseg | total_length | 162 | -0.004 | -0.012 | 0.002 |
| all | hrf | 1 | reliseg | total_length | 162 | -0.001 | -0.007 | 0.005 |
| all | hrf | 2 | reliseg | total_length | 162 | -0.001 | -0.008 | 0.005 |
| higher_resolution | fives | 0 | cfloss | FD | 71 | 0.001 | -0.009 | 0.011 |
| higher_resolution | fives | 1 | cfloss | FD | 71 | -0.008 | -0.019 | 0.001 |
| higher_resolution | fives | 2 | cfloss | FD | 71 | -0.007 | -0.020 | 0.001 |
| higher_resolution | fives | 0 | reliseg | FD | 71 | -0.013 | -0.028 | -0.001 |
| higher_resolution | fives | 1 | reliseg | FD | 71 | -0.015 | -0.035 | 0.000 |
| higher_resolution | fives | 2 | reliseg | FD | 71 | -0.001 | -0.012 | 0.008 |
| higher_resolution | fives | 0 | cfloss | density | 71 | -0.011 | -0.018 | -0.004 |
| higher_resolution | fives | 1 | cfloss | density | 71 | -0.006 | -0.011 | -0.002 |
| higher_resolution | fives | 2 | cfloss | density | 71 | -0.003 | -0.009 | 0.004 |
| higher_resolution | fives | 0 | reliseg | density | 71 | -0.003 | -0.014 | 0.007 |
| higher_resolution | fives | 1 | reliseg | density | 71 | -0.013 | -0.026 | -0.003 |
| higher_resolution | fives | 2 | reliseg | density | 71 | -0.002 | -0.011 | 0.005 |
| higher_resolution | fives | 0 | cfloss | tortuosity | 71 | 0.007 | -0.044 | 0.044 |
| higher_resolution | fives | 1 | cfloss | tortuosity | 71 | 0.006 | 0.001 | 0.074 |
| higher_resolution | fives | 2 | cfloss | tortuosity | 71 | -0.003 | -0.063 | 0.003 |
| higher_resolution | fives | 0 | reliseg | tortuosity | 71 | -0.027 | -0.077 | 0.052 |
| higher_resolution | fives | 1 | reliseg | tortuosity | 71 | 0.001 | -0.042 | 0.044 |
| higher_resolution | fives | 2 | reliseg | tortuosity | 71 | -0.008 | -0.130 | 0.000 |
| higher_resolution | fives | 0 | cfloss | total_length | 71 | -0.003 | -0.010 | 0.003 |
| higher_resolution | fives | 1 | cfloss | total_length | 71 | -0.003 | -0.008 | 0.000 |
| higher_resolution | fives | 2 | cfloss | total_length | 71 | -0.010 | -0.019 | -0.003 |
| higher_resolution | fives | 0 | reliseg | total_length | 71 | -0.018 | -0.032 | -0.006 |
| higher_resolution | fives | 1 | reliseg | total_length | 71 | -0.027 | -0.054 | -0.008 |
| higher_resolution | fives | 2 | reliseg | total_length | 71 | -0.010 | -0.025 | 0.002 |
| higher_resolution | hrf | 0 | cfloss | FD | 71 | -0.080 | -0.119 | -0.045 |
| higher_resolution | hrf | 1 | cfloss | FD | 71 | -0.059 | -0.100 | -0.027 |
| higher_resolution | hrf | 2 | cfloss | FD | 71 | -0.052 | -0.089 | -0.025 |
| higher_resolution | hrf | 0 | reliseg | FD | 71 | -0.025 | -0.074 | 0.016 |
| higher_resolution | hrf | 1 | reliseg | FD | 71 | -0.003 | -0.051 | 0.038 |
| higher_resolution | hrf | 2 | reliseg | FD | 71 | -0.040 | -0.103 | 0.004 |
| higher_resolution | hrf | 0 | cfloss | density | 71 | -0.063 | -0.087 | -0.039 |
| higher_resolution | hrf | 1 | cfloss | density | 71 | -0.053 | -0.088 | -0.016 |
| higher_resolution | hrf | 2 | cfloss | density | 71 | -0.057 | -0.084 | -0.033 |
| higher_resolution | hrf | 0 | reliseg | density | 71 | -0.041 | -0.074 | -0.012 |
| higher_resolution | hrf | 1 | reliseg | density | 71 | -0.017 | -0.043 | 0.004 |
| higher_resolution | hrf | 2 | reliseg | density | 71 | -0.031 | -0.054 | -0.008 |
| higher_resolution | hrf | 0 | cfloss | tortuosity | 71 | -0.047 | -0.132 | -0.018 |
| higher_resolution | hrf | 1 | cfloss | tortuosity | 71 | -0.630 | -0.827 | 0.026 |
| higher_resolution | hrf | 2 | cfloss | tortuosity | 71 | 0.040 | -0.099 | 0.143 |
| higher_resolution | hrf | 0 | reliseg | tortuosity | 71 | -0.001 | -0.167 | 0.074 |
| higher_resolution | hrf | 1 | reliseg | tortuosity | 71 | -0.720 | -1.027 | 0.043 |
| higher_resolution | hrf | 2 | reliseg | tortuosity | 71 | -0.029 | -0.075 | -0.007 |
| higher_resolution | hrf | 0 | cfloss | total_length | 71 | -0.055 | -0.080 | -0.033 |
| higher_resolution | hrf | 1 | cfloss | total_length | 71 | -0.045 | -0.075 | -0.020 |
| higher_resolution | hrf | 2 | cfloss | total_length | 71 | -0.050 | -0.084 | -0.025 |
| higher_resolution | hrf | 0 | reliseg | total_length | 71 | -0.006 | -0.049 | 0.029 |
| higher_resolution | hrf | 1 | reliseg | total_length | 71 | -0.002 | -0.042 | 0.030 |
| higher_resolution | hrf | 2 | reliseg | total_length | 71 | -0.002 | -0.045 | 0.031 |
| lower_resolution | fives | 0 | cfloss | FD | 91 | 0.002 | -0.009 | 0.013 |
| lower_resolution | fives | 1 | cfloss | FD | 91 | 0.008 | -0.001 | 0.020 |
| lower_resolution | fives | 2 | cfloss | FD | 91 | -0.008 | -0.020 | 0.002 |
| lower_resolution | fives | 0 | reliseg | FD | 91 | 0.002 | -0.013 | 0.023 |
| lower_resolution | fives | 1 | reliseg | FD | 91 | -0.004 | -0.021 | 0.011 |
| lower_resolution | fives | 2 | reliseg | FD | 91 | -0.014 | -0.026 | -0.003 |
| lower_resolution | fives | 0 | cfloss | density | 91 | -0.002 | -0.008 | 0.004 |
| lower_resolution | fives | 1 | cfloss | density | 91 | -0.001 | -0.005 | 0.002 |
| lower_resolution | fives | 2 | cfloss | density | 91 | 0.004 | -0.002 | 0.010 |
| lower_resolution | fives | 0 | reliseg | density | 91 | -0.006 | -0.011 | -0.001 |
| lower_resolution | fives | 1 | reliseg | density | 91 | -0.005 | -0.010 | -0.001 |
| lower_resolution | fives | 2 | reliseg | density | 91 | 0.007 | 0.003 | 0.012 |
| lower_resolution | fives | 0 | cfloss | tortuosity | 91 | -0.049 | -0.136 | 0.032 |
| lower_resolution | fives | 1 | cfloss | tortuosity | 91 | -0.099 | -0.246 | 0.030 |
| lower_resolution | fives | 2 | cfloss | tortuosity | 91 | 0.035 | -0.010 | 0.090 |
| lower_resolution | fives | 0 | reliseg | tortuosity | 91 | -0.042 | -0.125 | 0.032 |
| lower_resolution | fives | 1 | reliseg | tortuosity | 91 | 0.001 | -0.034 | 0.041 |
| lower_resolution | fives | 2 | reliseg | tortuosity | 91 | 0.004 | -0.016 | 0.022 |
| lower_resolution | fives | 0 | cfloss | total_length | 91 | -0.001 | -0.006 | 0.005 |
| lower_resolution | fives | 1 | cfloss | total_length | 91 | 0.003 | -0.002 | 0.009 |
| lower_resolution | fives | 2 | cfloss | total_length | 91 | -0.002 | -0.007 | 0.003 |
| lower_resolution | fives | 0 | reliseg | total_length | 91 | -0.005 | -0.017 | 0.004 |
| lower_resolution | fives | 1 | reliseg | total_length | 91 | -0.003 | -0.012 | 0.004 |
| lower_resolution | fives | 2 | reliseg | total_length | 91 | -0.002 | -0.008 | 0.004 |
| lower_resolution | hrf | 0 | cfloss | FD | 91 | -0.039 | -0.073 | -0.017 |
| lower_resolution | hrf | 1 | cfloss | FD | 91 | -0.061 | -0.114 | -0.031 |
| lower_resolution | hrf | 2 | cfloss | FD | 91 | -0.042 | -0.069 | -0.024 |
| lower_resolution | hrf | 0 | reliseg | FD | 91 | -0.006 | -0.025 | 0.016 |
| lower_resolution | hrf | 1 | reliseg | FD | 91 | 0.004 | -0.012 | 0.024 |
| lower_resolution | hrf | 2 | reliseg | FD | 91 | -0.016 | -0.035 | 0.003 |
| lower_resolution | hrf | 0 | cfloss | density | 91 | -0.041 | -0.069 | -0.021 |
| lower_resolution | hrf | 1 | cfloss | density | 91 | -0.069 | -0.107 | -0.042 |
| lower_resolution | hrf | 2 | cfloss | density | 91 | -0.062 | -0.095 | -0.040 |
| lower_resolution | hrf | 0 | reliseg | density | 91 | -0.035 | -0.066 | -0.015 |
| lower_resolution | hrf | 1 | reliseg | density | 91 | -0.008 | -0.023 | 0.006 |
| lower_resolution | hrf | 2 | reliseg | density | 91 | -0.027 | -0.046 | -0.015 |
| lower_resolution | hrf | 0 | cfloss | tortuosity | 91 | -0.011 | -0.175 | 0.168 |
| lower_resolution | hrf | 1 | cfloss | tortuosity | 91 | 0.093 | -0.094 | 0.300 |
| lower_resolution | hrf | 2 | cfloss | tortuosity | 91 | 0.108 | -0.085 | 0.319 |
| lower_resolution | hrf | 0 | reliseg | tortuosity | 91 | 0.048 | -0.067 | 0.211 |
| lower_resolution | hrf | 1 | reliseg | tortuosity | 91 | -0.072 | -0.127 | -0.015 |
| lower_resolution | hrf | 2 | reliseg | tortuosity | 91 | -0.023 | -0.108 | 0.063 |
| lower_resolution | hrf | 0 | cfloss | total_length | 91 | -0.058 | -0.096 | -0.033 |
| lower_resolution | hrf | 1 | cfloss | total_length | 91 | -0.079 | -0.135 | -0.045 |
| lower_resolution | hrf | 2 | cfloss | total_length | 91 | -0.045 | -0.074 | -0.026 |
| lower_resolution | hrf | 0 | reliseg | total_length | 91 | -0.004 | -0.021 | 0.011 |
| lower_resolution | hrf | 1 | reliseg | total_length | 91 | 0.010 | 0.000 | 0.023 |
| lower_resolution | hrf | 2 | reliseg | total_length | 91 | -0.002 | -0.013 | 0.009 |

Source: `results\pivot\e4_maples_zeroshot.csv`; strata `results\pivot\e4\maples_strata.csv`.

Pre-annotation vs manually corrected anchor sensitivity: `results\pivot\e4_maples_preannot.csv`

## 9. Coverage notes
- arm `cfloss` incomplete -- skipped

## Source paths
- `results/pivot/e4/folds.json`
- `results/pivot/e4/replication_rule.json`
- `results/pivot/e4/maples_strata.csv`
- `results/pivot/e4_gt_fundusavseg.csv`
- `results/pivot/e4_fidelity.csv`
- `results/pivot/e4_delta.csv`
- `results/pivot/e4_pixel.csv`
- `results/pivot/e4_safety.csv`
- `results/pivot/e4_downstream.csv`
- `results/pivot/e4_power.csv`
- `results/pivot/e4_maples_zeroshot.csv`
- `results/pivot/e4_maples_preannot.csv`
- `runs/pivot/e4/fold<k>/base/{config.json,split.json,summary.json,best.pt}`
- `runs/pivot/e4/fold<k>/<arm>/{config.json,log.json,summary.json,last.pt}`
- `runs/pivot/e4/fold<k>/<arm>/pred/{manifest.csv,pixel_metrics.csv,bio.csv}`
- `runs/pivot/e4/maples/<tag>/{manifest.csv,pixel_metrics.csv,bio.csv}`
- `runs/pivot/e4/driver_gpu<g>.log`
