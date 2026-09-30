# Bias diagnostics -- is macro-MAE rewarding over-connection?

Source: `results/bias_diagnostics.csv` (built by `python -m src.eval.bias_diagnostics`). Signed bias is `mean_images (B_pred - B_GT) / sigma` with the Gate A sigma; **negative = the mask under-states the biomarker**. `px_added` is `after_n_pred_px - before_n_pred_px`.


## 1. Signed bias before repair (the `no_repair` row)

If these are negative, the unrepaired segmentation under-states the biomarker and anything that adds pixels will move it toward the truth.

| dataset | FD_pvbm | FD_skan | tortuosity_pvbm | tortuosity_skan | density_pvbm | density_skan | total_length_pvbm | total_length_skan |
|---|---|---|---|---|---|---|---|---|
| drive | -0.9291 | -1.1695 | -0.9048 | -1.3094 | 0.1194 | 0.1194 | -1.2136 | -1.0777 |
| chasedb1 | -0.5182 | -0.7896 | -0.4793 | -1.1878 | 1.4079 | 1.4079 | -1.3504 | -0.5642 |
| hrf | -1.1035 | -1.8971 | -2.7806 | -2.0916 | 0.2783 | 0.2783 | -3.4610 | -2.0365 |
| fives | -0.2377 | -1.0524 | -1.2313 | -0.4574 | -0.0430 | -0.0430 | -0.4010 | -0.2586 |


## 2. Pixels added, edges accepted, and macro-MAE change

| dataset | method | n_accepted_mean | px_added_mean | px_added_frac_of_pred | macro_mae_before | macro_mae_after | macro_mae_delta |
|---|---|---|---|---|---|---|---|
| drive | no_repair | 0.0000 | 0.0000 | 0.0000 | 1.0653 | 1.0653 | 0.0000 |
| drive | geometric | 14.5500 | 158.0333 | 0.0054 | 1.0653 | 1.0445 | -0.0207 |
| drive | evapore_scorer | 13.6333 | 148.9000 | 0.0051 | 1.0653 | 1.0463 | -0.0189 |
| drive | evapore_e2e | 113.3833 | 1857.8667 | 0.0643 | 1.0653 | 0.9037 | -0.1616 |
| drive | rnca | 50.6333 | 12030.6833 | 0.4112 | 1.0653 | 1.6972 | 0.6319 |
| drive | rigr_uniform | 7.0167 | 86.6500 | 0.0030 | 1.0653 | 1.0543 | -0.0110 |
| drive | rigr_risk | 10.1833 | 130.1000 | 0.0044 | 1.0653 | 1.0487 | -0.0166 |
| chasedb1 | no_repair | 0.0000 | 0.0000 | 0.0000 | 1.1876 | 1.1876 | 0.0000 |
| chasedb1 | geometric | 16.4167 | 199.5417 | 0.0031 | 1.1876 | 1.1632 | -0.0245 |
| chasedb1 | evapore_scorer | 16.0833 | 195.9167 | 0.0030 | 1.1876 | 1.1642 | -0.0234 |
| chasedb1 | evapore_e2e | 289.9167 | 4972.3333 | 0.0767 | 1.1876 | 1.5612 | 0.3735 |
| chasedb1 | rnca | 61.9167 | 29736.9583 | 0.4539 | 1.1876 | 3.4035 | 2.2159 |
| chasedb1 | rigr_uniform | 10.5000 | 118.8333 | 0.0018 | 1.1876 | 1.1697 | -0.0179 |
| chasedb1 | rigr_risk | 9.4583 | 103.5000 | 0.0016 | 1.1876 | 1.1708 | -0.0168 |
| hrf | no_repair | 0.0000 | 0.0000 | 0.0000 | 1.8735 | 1.8735 | 0.0000 |
| hrf | geometric | 41.0778 | 571.5000 | 0.0009 | 1.8735 | 1.8497 | -0.0238 |
| hrf | evapore_scorer | 40.8111 | 562.1556 | 0.0009 | 1.8735 | 1.8490 | -0.0245 |
| hrf | evapore_e2e | 734.1556 | 12514.9667 | 0.0190 | 1.8735 | 1.4905 | -0.3830 |
| hrf | rnca | 165.4444 | 324891.7889 | 0.4863 | 1.8735 | 3.9956 | 2.1221 |
| hrf | rigr_uniform | 56.0222 | 606.1667 | 0.0009 | 1.8735 | 1.8510 | -0.0225 |
| hrf | rigr_risk | 59.9222 | 700.3000 | 0.0011 | 1.8735 | 1.8445 | -0.0290 |
| fives | no_repair | 0.0000 | 0.0000 | 0.0000 | 0.6468 | 0.6468 | 0.0000 |
| fives | geometric | 7.2167 | 237.3222 | 0.0021 | 0.6468 | 0.6389 | -0.0079 |
| fives | evapore_scorer | 6.8333 | 210.2167 | 0.0019 | 0.6468 | 0.6391 | -0.0077 |
| fives | evapore_e2e | 143.2222 | 2173.6278 | 0.0185 | 0.6468 | 0.6203 | -0.0264 |
| fives | rnca | 31.5944 | 54802.6889 | 0.4383 | 0.6468 | 1.7098 | 1.0630 |
| fives | rigr_uniform | 7.6167 | 430.4722 | 0.0035 | 0.6468 | 0.6294 | -0.0174 |
| fives | rigr_risk | 9.6722 | 509.3611 | 0.0041 | 0.6468 | 0.6293 | -0.0175 |


## 3. Spearman(pixels added, delta macro-MAE) across images

Negative = adding pixels *lowers* macro-MAE, i.e. the metric rewards over-connection.

| dataset | method | n_images | spearman_pxadded_vs_dmacromae | spearman_p |
|---|---|---|---|---|
| drive | geometric | 60 | -0.2476 | 0.0564 |
| drive | evapore_scorer | 60 | -0.2641 | 0.0414 |
| drive | evapore_e2e | 60 | -0.2902 | 0.0245 |
| drive | rnca | 60 | 0.3526 | 0.0057 |
| drive | rigr_uniform | 60 | -0.5233 | 0.0000 |
| drive | rigr_risk | 60 | -0.4218 | 0.0008 |
| chasedb1 | geometric | 24 | -0.5510 | 0.0053 |
| chasedb1 | evapore_scorer | 24 | -0.5210 | 0.0090 |
| chasedb1 | evapore_e2e | 24 | 0.2583 | 0.2229 |
| chasedb1 | rnca | 24 | 0.4409 | 0.0311 |
| chasedb1 | rigr_uniform | 24 | -0.5820 | 0.0028 |
| chasedb1 | rigr_risk | 24 | -0.5341 | 0.0072 |
| hrf | geometric | 90 | -0.5474 | 0.0000 |
| hrf | evapore_scorer | 90 | -0.5873 | 0.0000 |
| hrf | evapore_e2e | 90 | -0.3164 | 0.0024 |
| hrf | rnca | 90 | 0.3112 | 0.0028 |
| hrf | rigr_uniform | 90 | -0.6555 | 0.0000 |
| hrf | rigr_risk | 90 | -0.6330 | 0.0000 |
| fives | geometric | 180 | -0.1348 | 0.0712 |
| fives | evapore_scorer | 180 | -0.1323 | 0.0767 |
| fives | evapore_e2e | 180 | -0.1439 | 0.0539 |
| fives | rnca | 180 | 0.5780 | 0.0000 |
| fives | rigr_uniform | 180 | -0.2938 | 0.0001 |
| fives | rigr_risk | 180 | -0.2290 | 0.0020 |
