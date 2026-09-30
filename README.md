# Retinal biomarker fidelity

**A validation protocol for biomarkers measured on automatic segmentations, applied to retinal vessels.**

Retinal vascular biomarkers such as vessel density, total length, fractal dimension and tortuosity
are usually computed from automatic vessel segmentations, while the segmenters themselves are judged
by pixel overlap. Good overlap does not guarantee good biomarkers. This protocol checks the biomarkers
directly: for every cohort it measures the offset, the image-level scatter and the agreement between
biomarkers from predicted masks and from reference masks, all on one frozen training-split scale, and
it runs the same downstream classifier on both so the effect of measurement error on disease
prediction can be read off as a paired difference.

On top of that audit it injects controlled errors (topology breaks against a pixel-matched control,
constant and correlated offsets), compares fine-tuning remedies against a same-budget training
control, and repeats the key findings on held-out mask-bearing cohorts and on large label-only cohorts.
The decision rules were fixed before the results were looked at; every such decision and every later
change is timestamped in [`exp/DECISIONS.md`](exp/DECISIONS.md) (written in Chinese, code names in English).

> **Paper:** *An integrated validation protocol for segmentation-derived imaging biomarkers: a retinal
> vessel case study* (under review). The citation will be finalised on publication.

## What's here

```
exp/src/            all code; run from exp/ as `python -m src.<package>.<module>`
exp/src/seg/        U-Net segmenter (training, sliding-window inference)
exp/src/bio/        biomarker pipelines (skan primary panel, PVBM sensitivity)
exp/src/pivot/      the analyses reported in the paper
exp/configs/        pipeline stage configurations
exp/data/           dataset notes (DATASETS.md, EXTERNAL_DATASETS.md); no images
exp/results/        summary CSVs and per-analysis reports (results/pivot/*_REPORT.md)
exp/remote/         helpers used to spread stages over several machines (hosts replaced
                    by placeholders; not needed on a single machine)
exp/DECISIONS.md    decision log
```

No images, masks, labels or model weights are included. Weights will be released on acceptance and
are available from the authors on request until then.

## Setup

Python 3.11. Install PyTorch for your CUDA version first, then the rest:

```bash
pip install -r requirements.txt        # or requirements-lock.txt for the exact versions used
                                       # (torch 2.6.0 + CUDA 12.4)
```

Datasets are public but not redistributed. Download them from the official sources and unpack each
under `exp/data/<name>/raw/`, using the names `drive`, `chasedb1`, `hrf`, `fives`, `stare`,
`fundusavseg` and `maplesdr` for the vessel-mask cohorts, and `exp/data/external/<name>/raw/` with
`aptos2019`, `idrid`, `messidor2` and `odir5k` for the label-only cohorts.
[`exp/data/DATASETS.md`](exp/data/DATASETS.md) and
[`exp/data/EXTERNAL_DATASETS.md`](exp/data/EXTERNAL_DATASETS.md) give the download link, the exact
folder layout, file counts, splits and licence for each one.

## Reproduce

Everything runs from `exp/`. The first two commands check that the data are in place:

```bash
cd exp
python -m src.data.check_data          # vessel-mask cohorts
python -m src.data.check_external      # label-only cohorts
```

The pipeline stages, in order:

```bash
python -m src.seg.train --dataset fives --seed 0 --gpu 0     # segmenters (per dataset and seed)
python -m src.bio.gate_a                                     # biomarkers on reference masks
python -m src.c1.train_scales                                # frozen training-split scales
python -m src.pivot.build_table --procs 6                    # per-image biomarker table
python -m src.pivot.e1_decompose                             # calibration audit
python -m src.pivot.e1_oracle                                # reference-mask downstream arm
python -m src.pivot.e2_run plan                              # fine-tuning grid, then `gpu` and `bio`
python -m src.pivot.e2_analysis
python -m src.pivot.e3_external --help                       # label-only cohorts (`bio`, then `clf`)
python -m src.pivot.e4_replication plan                      # held-out replication
```

The shipped `results/` tree already holds the per-image biomarker table, so the later calibration
analyses run on a laptop without any data or GPU, for example:

```bash
OMP_NUM_THREADS=1 python -m src.pivot.r2_calibration
OMP_NUM_THREADS=1 python -m src.pivot.r3_deming_lambda
```

Each module's docstring gives its full command line, inputs and outputs, and the
`results/pivot/*_REPORT.md` files describe what each analysis produced.

## Citation

See [`CITATION.cff`](CITATION.cff). A full reference will be added on publication.

## License

Code: MIT (see [`LICENSE`](LICENSE)). The datasets keep their own licences and terms of use. The
EVAPORE baseline adapter (`exp/src/baselines/evapore/`) loads the upstream GPL-3.0 code at run time;
that code is not included.
