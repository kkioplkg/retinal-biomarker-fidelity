# E4 -- what changed after the fold-4 repair
Generated 2026-09-18 22:35.  Source of the defect and its diagnosis: DECISIONS.md 2026-09-18 19:45.

## The defect
Inference jobs in `e4_replication.build_jobs()` depended on `<arm>/last.pt`, which
`p5_finetune` rewrites after **every** epoch.  The dependency was therefore satisfied
the moment training started, and with two GPU lanes sharing the queue the second lane
evaluated a checkpoint that was still training.  On **fold 4, seed 0** the three
fine-tuned arms were inferred at epoch **24 (cfloss), 38 (continued), 39 (reliseg)**
instead of 49 -- `pred/infer_meta.json` was written 18:56-18:57 on 2026-09-17 while
`summary.json` was written 19:05-19:11.  `baseline` reads `base/best.pt` and is
unaffected.  The dependency is now `summary.json`, which is written once, at the end.

## Verification that the repair is localised to fold 4
`leave_out_fold4` is the same contrast computed on the **other four folds**.  It is
bit-identical before and after, which proves nothing outside fold 4 moved:

| biomarker | before | after |
|---|---:|---:|
| density | -0.001604 | -0.001604 |
| total_length | -0.000293 | -0.000293 |
| FD | -0.007814 | -0.007814 |
| tortuosity | -0.052796 | -0.052796 |

## Fold-4 fidelity r (skan), by arm

| arm | biomarker | before | after | change |
|---|---|---:|---:|---:|
| continued | density | 0.98561 | 0.98563 | +0.00002 |
| continued | total_length | 0.98703 | 0.98733 | +0.00030 |
| continued | FD | 0.90275 | 0.88353 | -0.01922 |
| continued | tortuosity | 0.92477 | 0.91334 | -0.01143 |
| reliseg | density | 0.98518 | 0.98366 | -0.00152 |
| reliseg | total_length | 0.98494 | 0.98439 | -0.00055 |
| reliseg | FD | 0.88088 | 0.87216 | -0.00872 |
| reliseg | tortuosity | 0.75400 | 0.73762 | -0.01638 |
| cfloss | density | 0.98614 | 0.98719 | +0.00106 |
| cfloss | total_length | 0.98693 | 0.98705 | +0.00012 |
| cfloss | FD | 0.88134 | 0.88089 | -0.00045 |
| cfloss | tortuosity | 0.91544 | 0.92434 | +0.00890 |

## Seed-0 pooled out-of-fold delta-r, reliseg - continued (skan) -- THE NUMBERS THE PAPER MUST REPLACE

| biomarker | before d_r [95% CI] | after d_r [95% CI] | p_adj before | p_adj after |
|---|---|---|---:|---:|
| density | -0.00034 [-0.00483, 0.00380] | -0.00127 [-0.00582, 0.00279] | 0.8585 | 0.9725 |
| total_length | -0.00057 [-0.00244, 0.00152] | -0.00084 [-0.00258, 0.00120] | 0.8895 | 0.9435 |
| FD | -0.01036 [-0.03758, 0.01414] | -0.00882 [-0.03746, 0.01673] | 1.0000 | 1.0000 |
| tortuosity | -0.06310 [-0.13658, -0.02961] | -0.06208 [-0.13518, -0.02915] | n/a | n/a |

## Conclusions that did NOT change

- `replication_success`: **False** before, **False** after; `significant`: [] -> [].
- Safety flags: {'ok': 42, 'BREACH': 8} before, {'ok': 42, 'BREACH': 8} after (identical).  The tortuosity `delta_r` BREACH of the
  pre-registered -0.05 margin is pre-existing and survives the repair.
- MAPLES-DR zero-shot artefacts are untouched (no MAPLES unit was re-run).

The repair moves the seed-0 pooled estimates by at most 0.0015 in delta-r and changes
no qualitative conclusion -- but the affected predictions were nonetheless produced by
models that had not finished training, so the old numbers must not be published.

## And the reason this matters more than the repair: 5 fine-tune seeds

With one seed these contrasts were indistinguishable from noise.  With the five seeds
review point 16 asked for, `reliseg` is **consistently worse** than the same-budget
`continued` control -- every seed has the same sign on FD, density and tortuosity, and
the seed-level interval excludes zero:

| biomarker | mean d_r over 5 seeds | seed-level 95% t interval | seeds positive | same sign? |
|---|---:|---|---:|---|
| density | -0.00220 | [-0.00353, -0.00087] | 0/5 | yes |
| total_length | -0.00047 | [-0.00092, -0.00001] | 1/5 | no |
| FD | -0.00694 | [-0.01199, -0.00189] | 0/5 | yes |
| tortuosity | -0.05900 | [-0.07058, -0.04742] | 0/5 | yes |

Per-seed values are in `results/pivot/e4_delta_seeds.csv`; each seed's full analysis is
`E4_REPORT_s<k>.md` / `e4_*_s<k>.csv`, produced by the identical estimator
(`e4_replication analyse --seed <k>`).
