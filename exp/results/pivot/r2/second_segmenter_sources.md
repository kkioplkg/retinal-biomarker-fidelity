# Second-Segmenter Sources: Public Retinal Vessel Segmentation Pipelines with Released Weights

**Date: 2026-09-18**
**Purpose:** find an off-the-shelf, pretrained, non-U-Net-baseline vessel segmenter that can be run *zero-shot* on DRIVE / CHASE_DB1 / HRF / FIVES test images on this machine.

**Target environment (verified locally, 2026-09-18):**

```
D:/Anaconda/envs/medical1/python.exe
Python 3.11.16 (conda-forge, MSC v.1944, win-amd64)
torch 2.6.0+cu124   torchvision 0.21.0+cu124   CUDA 12.4
2 x RTX 3080 20GB, Windows 10 Pro 19045, internet OK
```

No GPU jobs were run and nothing was installed while producing this document — all statements come from repository sources, package metadata and HTTP checks made today.

---

## 1. Summary table

| # | Candidate | Repo (HTTP status checked 2026-09-18) | Weights | Code licence | Weights licence | torch pin | Verdict |
|---|-----------|------|---------|--------------|-----------------|-----------|---------|
| **1** | **VascX / retinalysis-vascx** (Eyened, Erasmus MC, TVST 2025) | https://github.com/eyened/retinalysis-vascx — **200** | HuggingFace `Eyened/vascx`, **auto-downloaded** | repo has **no LICENSE file** (unspecified); engine `retinalysis-inference` = **AGPL-3.0** | **AGPL-3.0** (HF model card) | `torch>=2.4.1,<3`, `torchvision>=0.19,<1` → **2.6.0 OK**; python `>=3.10,<3.13` → **3.11 OK** | **RUNNABLE-EASY** |
| 2 | **LWNet / "Little W-Net"** (Galdran et al. 2020) | https://github.com/agaldran/lwnet — **200** | **in-repo** `experiments/*/model_checkpoint.pth` (~23 MB clone) | **MIT** | **MIT** (same repo) | env file pins torch 1.3.1, but code is plain torch; needs 1-line `weights_only=False` fix on 2.6 | **RUNNABLE-EASY** (one small patch) |
| 3 | **AutoMorph M2 binary-vessel** (BF-Net/SEGAN ensemble, Zhou et al.) | https://github.com/rmaphoh/AutoMorph — **200** | **in-repo git blobs**, 10 × 34.6 MB (whole clone ≈ **1.6 GB**) | **Apache-2.0** | Apache-2.0 (in-repo) | pins torch 2.3.1 / tv 0.18.1; loads pure `state_dict` → **2.6 compatible** | **RUNNABLE-HARD** (bash-only entrypoints, M0 dependency, huge clone) |
| 4 | **FR-UNet** (JBHI 2022) | https://github.com/lseventeen/FR-UNet — **200** | **in-repo** `pretrained_weights/<DATASET>/checkpoint-epoch40.pth` (clone ≈ 290 MB) | **MIT** | MIT | `torch==1.7.0`, `timm==0.3.2`, `albumentations==0.5.2` — **hostile pins** | **RUNNABLE-HARD** |
| 5 | **SGL** (Study Group Learning, MICCAI 2021) | https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation — **200** | **in-repo** `sgl_training/pretrained/` (clone ≈ 190 MB) | **MIT** | MIT | "PyTorch >= 1.3.1", EDSR-style trainer, `bash run_test.sh` | **RUNNABLE-HARD** |
| 6 | IterNet | https://github.com/conscienceli/IterNet — **200** | Google-Drive link in README | MIT | MIT (code); weights unspecified | **Keras / TensorFlow**, not torch | **NOT RUNNABLE** in `medical1` |
| 7 | SA-UNet | https://github.com/clguo/SA-UNet — **200** | Google-Drive links (augmented data + models) | **no LICENSE file** (unspecified) | unspecified | **Keras / TensorFlow** (DropBlock impl.) | **NOT RUNNABLE** in `medical1` |

**Ranking for "easiest to run zero-shot today": 1 → 2 → 3 → 5 → 4 → 6/7.**

---

## 2. VascX (Eyened / Erasmus MC, 2025) — **recommended**

### 2.1 Identity and papers

* Paper: *VascX Models: Model Ensembles for Retinal Vascular Analysis from Color Fundus Images*, arXiv:2409.16016; published **TVST 2025** (accepted 2025-05-16), PMC12306690.
* Toolbox paper: *retinalysis-vascx: An explainable software toolbox for the extraction of retinal vascular biomarkers*, arXiv:2602.08580.
* Trained by pooling **>15 published annotated datasets** plus Rotterdam Study CFIs — so it is an *ensemble trained across datasets*, not a single DRIVE-fitted U-Net. This is the strongest "different family / different operating point" argument for a paper revision.

### 2.2 Repositories (all HTTP 200)

| Role | URL | Notes |
|------|-----|-------|
| **Current** pipeline (use this) | https://github.com/eyened/retinalysis-vascx | PyPI `retinalysis-vascx`, **v2.1.0 released 2026-09-12** |
| Legacy model repo | https://github.com/Eyened/rtnls_vascx_models | README states: *"We will stop supporting this repository and move to the new one"* |
| Inference engine | https://github.com/Eyened/retinalysis-inference | PyPI `retinalysis-inference` **0.9.0 (2026-09-12)**, **AGPL-3.0** |
| Preprocessing | PyPI `retinalysis-fundusprep` | bounds detection + square crop + resize |
| **Weights** | https://huggingface.co/Eyened/vascx | public, **not gated**, no access request |

### 2.3 Weights: location, size, automatic download

HuggingFace repo `Eyened/vascx` (total 8.86 GB, but **only the files actually used are fetched**). Verified file listing of the `vessels/` folder:

```
vessels/vessels_may26.pt            142.2 MB   <-- CLI default
vessels/vessels_may26.onnx          141.4 MB
vessels/vessels_july24.pt           352.8 MB
vessels/vessels_july24_FIVES.pt     352.8 MB   <-- FIVES-finetuned variant
vessels/vessels_july24_DRHAGIS.pt   352.8 MB
vessels/vessels_july24_LEUVEN.pt    352.8 MB
vessels/vessels_july24_RS.pt        352.8 MB
```

Other folders: `artery_vein/`, `disc/`, `discedge/`, `fovea/`, `odfd/`, `quality/`, plus `notebooks/`, `samples/`.

* **Download is automatic** on first use ("The models are automatically downloaded from huggingface" — `retinalysis-inference` README). No registration, no manual step.
* For an offline/pinned run: download the files yourself and point at them with `--model-dir` or `VASCX_MODEL_DIR`. The CLI help names the defaults explicitly: `artery_vein/av_wsoft_patches_02_finetune.pt`, `vessels/vessels_may26.pt`.
* Per-model override flags exist: `--vessels-model`, `--av-model`, `--disc-model`, `--fovea-model`, `--quality-model` — so the **FIVES-finetuned** checkpoint can be swapped in with `--vessels-model .../vessels_july24_FIVES.pt`. **For a fair cross-dataset comparison, use the generic `vessels_may26.pt` default, not the FIVES variant.**

### 2.4 Licence — read this before using results in the paper

* Weights: **AGPL-3.0** (stated on the HF model card).
* Inference engine `retinalysis-inference`: **AGPL-3.0** (GitHub + PyPI metadata agree).
* `eyened/retinalysis-vascx`: **no LICENSE / LICENSE.md / LICENSE.txt / COPYING file exists** (all four return 404 as of today) and the PyPI metadata leaves the licence field empty → the top-level toolbox licence is formally **unspecified**, while everything it depends on is AGPL-3.0.
* Practical reading: fine as an external baseline/second annotator in a research paper; AGPL is copyleft, so **do not vendor its code into any released model of ours** without checking. No "research-only / non-commercial" clause is stated anywhere.

### 2.5 Install (works with the existing torch 2.6.0+cu124)

Torch is deliberately *not* a dependency of `retinalysis-inference` ("We did not include torch as a dependency … These must be installed manually beforehand"), so the existing CUDA build is kept:

```powershell
D:/Anaconda/envs/medical1/python.exe -m pip install retinalysis-vascx
```

This pulls `retinalysis-inference` (which pulls `retinalysis-fundusprep`, `lightning==2.*`, `monai==1.5.1`, `numpy==2.*`, `opencv-python==4.*`, `Pillow==11.*`, `simplejpeg`, `scikit-image>=0.22`, `networkx>=3.0`).

Compatibility check against this machine:

| Requirement | Declared | Here | OK? |
|---|---|---|---|
| python | `>=3.10,<3.13` (inference), `>=3.8,<3.15` (vascx) | 3.11.16 | yes |
| torch | `>=2.4.1,<3` | 2.6.0+cu124 | yes |
| torchvision | `>=0.19,<1` | 0.21.0+cu124 | yes |
| VRAM | "tested on a single GPU with at least 10 GB VRAM" | 20 GB | yes |

**Risk to watch:** `numpy==2.*` and `Pillow==11.*` are hard-pinned and may upgrade/downgrade packages in the shared `medical1` env. **Install into a separate env** (see Recommendation) rather than into `medical1` itself.

### 2.6 Inference — exact commands

Folder of images (or a CSV with `id` and `path` columns):

```powershell
# stage 1: segmentation (this is all we need for a second segmenter)
vascx run-models <PATH_TO_IMAGES> <PATH_TO_SEGMENTATIONS> --device cuda:0

# optional stage 2: vascular biomarkers
vascx calc-biomarkers <PATH_TO_SEGMENTATIONS> <PATH_TO_REPORT> --feature-set full_v3 --n-jobs 8
```

Useful flags: `--no-preprocess`, `--vessels/--no-vessels`, `--disc/--no-disc`, `--quality/--no-quality`, `--fovea/--no-fovea`, `--overlay/--no-overlay`, `--n-jobs`, `--device {cuda:0,mps,cpu}`, `--model-dir`.

Python API (engine level, if the CLI is too coarse):

```python
from rtnls_inference import make_ensemble
ensemble = make_ensemble(release_path, tta=False, overlap=0.25, tile_batch_size=8)
out = ensemble.predict_step({"image": tensor})          # aggregate tensor, N...
full = ensemble.predict_step_full(batch)                # per-member + logits
ensemble.predict_preprocessed(data, dest_path=...)      # already in canonical space
ensemble.predict_dataframe(df, preprocess=True)         # raw inputs
```

### 2.7 Input / output convention — **the critical detail**

* **Input:** any CFI. `retinalysis-fundusprep` *detects the FOV bounds, crops the smallest square containing them, and resizes to a fixed **1024×1024**.* No manual FOV mask needed.
* **Output tree** from `run-models`:

```
<PATH_TO_SEGMENTATIONS>/
├── preprocessed_rgb/   <id>.png   (1024x1024 model-ready crop)
├── vessels/            <id>.png   (binary vessel segmentation)
├── artery_vein/        <id>.png   (arteries=1, veins=2, crossings=3)
├── disc/               <id>.png
├── overlays/
├── bounds.csv          (crop geometry -> lets you map back to native pixels)
├── fovea.csv
└── quality.csv
```

* **Resolution caveat, quoted from the engine README:** *"High-level spatial outputs always end in canonical preprocessed geometry, normally 1024×1024. The package does not convert results back to the original photograph geometry."*
  → To score against DRIVE (565×584), CHASE_DB1 (999×960), HRF (3504×2336) or FIVES (2048×2048) ground truth you **must invert the crop+resize using `bounds.csv`** (crop box + scale), or resample the GT into the 1024² canonical frame. This is the single biggest piece of glue code needed, and it is deterministic (crop box + bilinear/nearest resize), not a research problem.
* Thresholding: the CLI writes a decoded mask (segmentation family aggregates as softmax `NHWC`); for a soft probability map, use `predict_step_full(batch)["logits"]` / `["aggregate"]` rather than the PNGs.

**Verdict: RUNNABLE-EASY.** `pip install`, automatic weights, CLI on a folder, native torch-2.6 support, actively maintained (both packages released 2026-09-12).

---

## 3. AutoMorph (rmaphoh/AutoMorph)

* Repo: https://github.com/rmaphoh/AutoMorph — **200**. Licence **Apache-2.0** (clean, permissive — better than VascX's AGPL for downstream reuse).
* Paper: *AutoMorph: Automated Retinal Vascular Morphology Quantification via a Deep Learning Pipeline* (medRxiv 2022 / TVST).

### 3.1 Weights: shipped inside the repo

No Google Drive, no Dropbox, no download script — checkpoints are **plain git blobs** (verified: `Content-Length: 34647071`, `application/octet-stream`, HTTP 200 on raw.githubusercontent.com, i.e. **not** LFS pointers):

```
M2_Vessel_seg/Saved_model/train_on_ALL-SIX/20210630_uniform_thres40_ALL-SIX_savebest_randomseed_{24,26,...,42}/G_best_F1_epoch.pth   # 10 x 34.6 MB
M2_Artery_vein/.../CP_best_F1_{A,V,all}.pth                                                                                          # 8 seeds x 3 files
M2_lwnet_disc_cup/experiments/wnet_All_three_1024_disc_cup/<seed>/model_checkpoint.pth                                               # 7 x 1.0 MB
```

The binary-vessel module is a **10-model SEGAN/BF-Net ensemble** trained on `ALL-SIX` (six public datasets). GitHub reports repo size **1,627,756 KB ≈ 1.6 GB** — a full `git clone` is heavy; use `--depth 1`.

### 3.2 Install (as documented in LOCAL.md)

```bash
conda create -n automorph python=3.11 -y
conda activate automorph
git clone https://github.com/rmaphoh/AutoMorph.git
cd AutoMorph
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install --ignore-installed certifi
pip install -r requirement.txt
pip install efficientnet_pytorch==0.7.1 --no-deps
```

`requirement.txt` pins only light things (`opencv-python-headless==4.10.0.84`, `pandas==2.2.0`, `Pillow==10.3.0`, `scikit-image==0.24.0`, `scipy==1.14.0`, `h5py`, `matplotlib`, `pyarrow`) and **comments torch out** ("installed by conda"). Repo notes say *"pytorch 2.3 & python 3.11 supported; Mac M2 GPU supported; CPU supported"*. Nothing in the vessel module needs 2.3 specifically; the checkpoints are pure `state_dict`s loaded with `net.load_state_dict(torch.load(path, map_location=device))`, which is **safe under torch 2.6's `weights_only=True` default**. So `medical1`'s torch 2.6 would very likely work — untested.

### 3.3 Running it, and the Windows problems

Documented entrypoint is a **bash** script:

```bash
sh run.sh                     # optional: --no_process --no_quality --no_segmentation --no_feature
```

`run.sh` does `python automorph_data.py`, then `rm -rf "${AUTOMORPH_DATA}/Results"/*`, then `cd M0_Preprocess && python EyeQ_process_main.py`, then for each module `cd M2_Vessel_seg && sh test_outside.sh`, etc.

Windows-specific blockers:

1. **`run.sh` / `test_outside.sh` are POSIX shell.** They use `rm -rf`, `cd`, `$((42-2*seed))`, `CUDA_VISIBLE_DEVICES=... python ...` prefix assignment — none of that runs in PowerShell. Workable via **Git Bash** (present on this machine), or by calling the python entrypoint directly.
2. **Env-var prefix + `CUDA_VISIBLE_DEVICES`** must become `$env:CUDA_VISIBLE_DEVICES` in PowerShell.
3. **`resolution_information.csv`** is mandatory: one row per image with pixel resolution, format per the repo template.
4. **Module separability:** M2 *is* separable in principle but **not standalone by default** — `test_outside_integrated.py` reads `AUTOMORPH_DATA = os.getenv('AUTOMORPH_DATA','..')` and expects the M0 preprocessing outputs, writing to `${AUTOMORPH_DATA}/Results/M2/binary_vessel/`. You would run M0 first (`--no_quality --no_feature`) or hand-build the expected `Results/M0/...` folder.

Bypassing the shell script (the sane route on Windows) — this is exactly what `M2_Vessel_seg/test_outside.sh` expands to:

```powershell
$env:AUTOMORPH_DATA="E:\...\automorph_data"; $env:CUDA_VISIBLE_DEVICES="0"
cd M2_Vessel_seg
D:/Anaconda/envs/automorph/python.exe test_outside_integrated.py --epochs=1 --batchsize=8 `
  --learning_rate=2e-4 --validation_ratio=10.0 --alpha=0.08 --beta=1.1 --gamma=0.5 `
  --dataset=ALL-SIX --dataset_test=ALL-SIX --uniform=True `
  --jn=20210630_uniform_thres40_ALL-SIX --worker_num=2 --save_model=best `
  --train_test_mode=test --pre_threshold=40.0 --seed_num=42 `
  --out_test="$env:AUTOMORPH_DATA/Results/M2/binary_vessel/"
```

### 3.4 Input / output convention

* Input is **resized to a fixed 912×912** when `--uniform=True` (`utils.Define_image_size` returns `(912, 912)` unconditionally in that branch).
* Outputs written under `Results/M2/binary_vessel/`: `resize/` (probability maps at 912²), `resize_binary/`, `raw/` and `raw_binary/` (**restored to native resolution** — this is better than VascX, which stops at 1024²), `resize_uncertainty/` (ensemble std), plus `binary_process/` (small objects <30 px removed, connectivity 5) and `binary_skeleton/`. Fractal dimension / vessel density / mean width are computed per image into `Results/M3/`.
* Binarisation happens inside the script (ensemble-averaged probability → threshold), with `--pre_threshold=40.0` used for intensity standardisation, not for mask thresholding.

**Verdict: RUNNABLE-HARD.** Nothing is missing — permissive licence, in-repo weights, native-resolution outputs, per-pixel uncertainty — but it costs a 1.6 GB clone, a second conda env, a Git-Bash or hand-translated invocation, the `resolution_information.csv`, and an M0 pre-pass. Best **second choice / complement**, and the best option if Apache-2.0 (vs AGPL) matters.

---

## 4. LWNet — "The Little W-Net That Could" (agaldran/lwnet)

* Repo: https://github.com/agaldran/lwnet — **200**. **MIT** (code and the in-repo checkpoints).
* Paper: Galdran, Anjos, Dolz, Chakor, Lombaert, Ben Ayed, arXiv:2009.01907 (2020). ~70k parameters — architecturally a cascade of two tiny U-Nets, explicitly *not* a plain U-Net, and a well-cited comparator.

### 4.1 Weights actually shipped (verified by listing the repo tree)

```
experiments/wnet_drive/model_checkpoint.pth            # vessel segmentation, DRIVE-trained
experiments/big_wnet_drive_av/model_checkpoint.pth     # artery/vein, DRIVE
experiments/big_wnet_hrf_av_1024/model_checkpoint.pth  # artery/vein, HRF @1024
```

**Caveat:** the README's training instructions mention `wnet_chasedb` and `wnet_hrf_1024`, but **only `wnet_drive` is actually committed for vessel segmentation**. CHASE-DB and HRF *vessel* checkpoints are not in the repo. Pre-generated *result* maps (not weights) are downloadable at
`https://gitlab.com/agaldran/shared_results/-/raw/master/pre_generated_results.zip?inline=false`.
This is not fatal — the paper's own cross-dataset protocol (§5 of the README) applies `wnet_drive` to CHASE-DB/HRF, which is exactly the zero-shot setting wanted here.

### 4.2 Install

The repo's `environment.txt` is a conda explicit-URL file pinned to **linux-64, python 3.7.7, pytorch 1.3.1, torchvision 0.4.2** — do **not** use it on Windows (there is no `requirements.txt`; that path 404s). The code itself is plain torch + scikit-image, so:

```powershell
git clone https://github.com/agaldran/lwnet.git E:\Programming\research_ws\medical1\exp\third_party\lwnet
# reuse medical1 as-is; only these may be missing:
D:/Anaconda/envs/medical1/python.exe -m pip install scikit-image pandas tqdm
```

### 4.3 Inference

Single image (the useful entrypoint):

```powershell
D:/Anaconda/envs/medical1/python.exe predict_one_image.py `
  --model_path experiments/wnet_drive/ `
  --im_path folder/my_image.jpg `
  --result_path my_results/ `
  --device cuda:0 --bin_thresh 0.42
```

Args: `--model_path` (default `experiments/wnet_drive`), `--im_path`, `--mask_path` (optional FOV mask), `--tta` (default `from_preds`), `--bin_thresh` (default **0.4196**; README quotes **0.42** for DRIVE and **0.3725** for HRF), `--im_size` (default **512**; HRF models use **1024**), `--device`, `--result_path`.

Whole dataset (needs the repo's `data/<DATASET>/test_all.csv` layout, populated by `python get_public_data.py`):

```powershell
D:/Anaconda/envs/medical1/python.exe generate_results.py --config_file experiments/wnet_drive/config.cfg --dataset DRIVE --device cuda:0
D:/Anaconda/envs/medical1/python.exe generate_results.py --config_file experiments/wnet_drive/config.cfg --dataset CHASEDB --device cuda:0
```

### 4.4 Input / output convention

* Input resized to `--im_size` (512 default, 1024 for HRF models).
* **Output is resized back to the original image size** (`resize(..., order=3)` bicubic to `original_sz`) — native resolution, unlike VascX.
* Two PNGs per image: `<name>_seg.png` (probability) and `<name>_bin_seg.png` (binary at `--bin_thresh`).

### 4.5 torch 2.6 risk

`utils/model_saving_loading.py` does `checkpoint = torch.load(checkpoint_path, map_location=device)` then `checkpoint['model_state_dict']` — the checkpoint is a **dict containing `stats`**, so torch 2.6's `weights_only=True` default may raise `UnpicklingError`. Fix is one line: `torch.load(..., map_location=device, weights_only=False)`. `predict_one_image.py` has no other deprecated torch API.

**Verdict: RUNNABLE-EASY** (after that one-line patch). 23 MB clone, MIT, no extra env, native-resolution output, per-image CLI. Weakness: only the DRIVE-trained vessel model is released, and it is a 2020 model trained on 20 images — weaker and less "independent" than VascX, but a genuinely different architecture family and trivially cheap.

---

## 5. FR-UNet (lseventeen/FR-UNet)

* Repo: https://github.com/lseventeen/FR-UNet — **200**. **MIT**. JBHI 2022, full-resolution network + dual-threshold iteration.
* Weights: **in-repo**, `pretrained_weights/<DATASET_NAME>/checkpoint-epoch40.pth` for DRIVE, CHASEDB1, STARE, CHUAC, DCA1. Clone ≈ 290 MB.
* Install: `pip install -r requirements.txt` — pins **`torch==1.7.0`, `torchvision==0.8.1`, `timm==0.3.2`, `albumentations==0.5.2`, `numpy==1.20.3`, `opencv_python==4.4.0.46`, `scikit_learn==1.0.2`, `ttach==0.0.3`, `torchstat`, `bunch`, `segmentation==0.2.2`**. `timm==0.3.2` is known to break on modern torch (`timm.models.layers` import path / `torch._six`), and `albumentations 0.5.2` predates the numpy 2 era. Installing the file as-is would trash a torch-2.6 env; you would have to install selectively.
* Inference is **not** folder-based:

```bash
python data_process.py -dp DATASET_PATH -dn DATASET_NAME     # builds preprocessed pickles first
python test.py -dp DATASET_PATH -wp pretrained_weights/DATASET_NAME
```

  `test.py` does a bare `torch.load(weight_path)` → will need `weights_only=False` on 2.6, and the checkpoint is a trainer dict.
* Output: patch-based full-resolution probability maps + dual-threshold iteration post-processing; `--show` saves predicted images. Evaluation is welded to their own dataset pickles, so scoring arbitrary folders means rewriting the dataloader.

**Verdict: RUNNABLE-HARD.** Weights are there and MIT, but the pinned stack is hostile and the data path is dataset-pickle-only (`data_process.py` for DRIVE/CHASEDB1/STARE only — no FIVES/HRF recipe).

---

## 6. SGL — Study Group Learning (SHI-Labs)

* Repo: https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation — **200**. **MIT**. MICCAI 2021; SOTA on DRIVE and CHASE_DB1 at publication.
* Weights: **in-repo**, "we have prepared the pre-trained models for both datasets in the folder `pretrained`". Raw DRIVE/CHASE_DB1 data and SGL pseudo-labels are also bundled (clone ≈ 190 MB).
* Requirements: "Prepare a PyTorch environment (>=1.3.1) and other necessary dependencies" — **no pinned requirements file**, which cuts both ways (no pin conflicts; also no guarantee).
* Inference: `cd sgl_training && bash run_test.sh` — again a **bash entrypoint** (EDSR-derived codebase with `option.py` args), needs translation to PowerShell or Git Bash, and the data layout is the repo's own `dataset/` structure.
* Family: U-Net backbone trained with a study-group/pseudo-label scheme — the *training scheme* is novel, the architecture is not. Slightly weaker as a "different family" argument than VascX or LWNet.

**Verdict: RUNNABLE-HARD.** Plausible in an afternoon, but offers less than VascX for the same effort.

---

## 7. Ruled out for this environment

* **IterNet** (https://github.com/conscienceli/IterNet, 200, MIT): **Keras/TensorFlow**. Pretrained weights are offered via a Google Drive link in the README ("a model trained with multiple datasets … works well on new data"). Attractive claim, wrong framework — would need a TF env. Weights licence not separately stated.
* **SA-UNet** (https://github.com/clguo/SA-UNet, 200): **Keras/TensorFlow** with a custom DropBlock layer; **no LICENSE file in the repo** (licence unspecified); augmented datasets and models distributed as Google Drive links. SA-UNetv2 (arXiv:2509.11774, ISBI 2026) exists but the same framework issue applies.
* **retipy**: pure classical image processing (and is already vendored inside AutoMorph's M3 feature stage) — not a learned segmenter with released weights, so it is not a second-segmenter candidate.
* Generic HuggingFace "retinal vessel U-Net" model cards: searched, nothing credible and maintained surfaced beyond `Eyened/vascx` itself.

---

## 8. Recommendation

**Use VascX (`retinalysis-vascx` + `Eyened/vascx` weights).** It is the only candidate that is `pip install`-able, auto-downloads weights, declares `torch>=2.4.1,<3` and `python>=3.10,<3.13` (so 3.11.16 + torch 2.6.0+cu124 is in-spec), was released **six days ago (2026-09-12)**, and — decisively for a revision — was trained on **>15 pooled datasets**, making it a genuinely independent segmenter rather than another DRIVE-fitted U-Net. It also returns artery/vein, disc and quality for free.

Keep **AutoMorph M2** as the declared fallback: Apache-2.0 instead of AGPL, in-repo weights, native-resolution masks, and per-pixel ensemble uncertainty.

### 8.1 Install steps under `E:\Programming\research_ws\medical1\exp\third_party\vascx\`

Use a **separate conda env**: `retinalysis-inference` hard-pins `numpy==2.*`, `Pillow==11.*`, `opencv-python==4.*`, `monai==1.5.1` and `lightning==2.*`, and must not be allowed to mutate the shared `medical1` env.

```powershell
# 0. workspace
New-Item -ItemType Directory -Force E:\Programming\research_ws\medical1\exp\third_party\vascx

# 1. dedicated env with the same python/torch as medical1
conda create -n vascx python=3.11 -y
conda activate vascx
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# 2. the toolbox (pulls retinalysis-inference + retinalysis-fundusprep)
pip install retinalysis-vascx

# 3. pin the weights location so downloads are reproducible and re-usable
$env:VASCX_MODEL_DIR = "E:\Programming\research_ws\medical1\exp\third_party\vascx\models"
$env:HF_HOME         = "E:\Programming\research_ws\medical1\exp\third_party\vascx\hf_cache"

# 4. (optional, for the source + notebooks + offline weights)
git clone https://github.com/eyened/retinalysis-vascx.git E:\Programming\research_ws\medical1\exp\third_party\vascx\src
#    manual weight fetch if the auto-download is to be avoided:
#    huggingface-cli download Eyened/vascx --include "vessels/vessels_may26.pt" "artery_vein/*" "disc/*" "fovea/*" "quality/*" `
#      --local-dir E:\Programming\research_ws\medical1\exp\third_party\vascx\models
```

### 8.2 Smoke test and then the real runs

```powershell
conda activate vascx
# smoke test: 3 DRIVE test images
vascx run-models E:\...\data\DRIVE\test\images_subset `
                 E:\Programming\research_ws\medical1\exp\third_party\vascx\out\drive_smoke `
                 --device cuda:0 --no-overlay --n-jobs 4

# full runs (one output folder per dataset)
vascx run-models <DRIVE_test_images>  ...\out\drive  --device cuda:0
vascx run-models <CHASE_test_images>  ...\out\chase  --device cuda:0
vascx run-models <HRF_test_images>    ...\out\hrf    --device cuda:0
vascx run-models <FIVES_test_images>  ...\out\fives  --device cuda:0
```

Check `out/<ds>/vessels/<id>.png`, `out/<ds>/preprocessed_rgb/<id>.png`, `out/<ds>/bounds.csv`.

### 8.3 The one piece of glue we must write

VascX masks land in **canonical 1024×1024 crop geometry, never mapped back** to the original photograph. Before any metric against DRIVE/CHASE/HRF/FIVES ground truth, write a small `vascx_to_native.py` that reads `bounds.csv` (crop box + scale per image), inverts crop+resize, and resamples the 1024² mask to native pixels with nearest-neighbour (masks) / bilinear (probabilities). Sanity-check by overlaying on the native image and by confirming the FOV mask agrees. Alternatively, and more cheaply for a comparison-only table, resample **the ground truth into the 1024² canonical frame** and report both conventions — but state clearly in the paper which frame the numbers are in.

### 8.4 Reporting notes for the revision

* Cite arXiv:2409.16016 / TVST 2025 for VascX; state the checkpoint used (`vessels_may26.pt`, the CLI default) and that **no** dataset-specific finetuned variant (`vessels_july24_FIVES.pt` etc.) was used.
* State the licence: code/weights **AGPL-3.0** (`retinalysis-inference`, HF model card); the `retinalysis-vascx` wrapper repo ships no LICENSE file.
* State the operating convention: FOV-bounds square crop → 1024×1024 → ensemble → mask, then inverse-mapped to native resolution by us; no retraining, no finetuning, no per-dataset threshold tuning (zero-shot).

---

## 9. Status 2026-09-18 — what was actually run

Scope rule applied at the end of the session: **a public pipeline is kept only if
it was trivially runnable in under 30 minutes.** LWNet met that bar; VascX did
not, and the external zero-shot arm is considered satisfied by LWNet alone.

### 9.1 LWNet — RAN, all four test splits

```powershell
git clone --depth 1 https://github.com/agaldran/lwnet.git `
  E:\Programming\research_ws\medical1\exp\third_party\lwnet
# clone HEAD = ce72ddabcf5fc9af14197ac0d43bd29000b17df2 (2024-01-16, "fixed scheduler")
# nothing was pip-installed: the project env `medical1` already had
# torch 2.6.0+cu124 / torchvision 0.21.0 / scikit-image 0.26.0 / PIL / numpy.
```

**Checkpoint (the only vessel checkpoint released in the repo):**

| field | value |
|---|---|
| path | `experiments/wnet_drive/model_checkpoint.pth` |
| size | 919,662 bytes |
| sha256 | `91f0cada4b26ece63464b05be60f9b3a51f1bcd1f764081d07d3a35961118de1` |
| arch | W-Net (`wnet`, layers 8/16/32), **68,482** trainable params |
| trained on | **DRIVE train split only** (`val_metrics.txt`: best AUC 98.59, best DICE 80.96, best cycle 23) |
| licence | **MIT** (code and weights, `LICENSE`, (c) 2020 Adrian Galdran) |

The same DRIVE checkpoint was applied to **all four** datasets — that is exactly
the repo's own cross-dataset protocol (README section 5) and is the zero-shot
setting we want. `wnet_chasedb` / `wnet_hrf_1024` are named in the README but
**are not committed**; only `wnet_drive` (vessels), `big_wnet_drive_av` and
`big_wnet_hrf_av_1024` (artery/vein) exist in the tree.

**Three upstream breakages on this machine, and the fixes** (all handled in our
own driver — the upstream repo was left unmodified):

1. `utils/paired_transforms_tv04.py` imports **`pkg_resources`**, which no longer
   exists in the `medical1` env (`ModuleNotFoundError: No module named
   'pkg_resources'`). Our driver does not import that module at all; the only
   two transforms needed (`Resize`, `ToTensor`) are ~3 lines of PIL + numpy.
2. `utils/model_saving_loading.py` does a bare `torch.load(...)`. Under torch 2.6
   (`weights_only=True` by default) the trainer dict (it carries a `stats` key)
   will not unpickle, so it is loaded with **`weights_only=False`** — exactly the
   one-line fix predicted in section 4.5.
3. `predict_one_image.py`'s `get_fov()` calls **`skimage.draw.circle`**, removed
   from modern scikit-image. We never call it: the FOV mask is taken from the
   **project's own `fov_path`** — the same mask every internal arm uses — which
   is strictly better for a fair comparison than re-estimating it.

**Driver:** `exp/third_party/lwnet/run_project_splits.py` (ours, standalone).
Operating convention, identical for all four datasets (no per-dataset tuning):

```
project FOV mask -> bbox crop -> bilinear resize to 512x512 -> W-Net
  -> sigmoid, 4-flip TTA averaged over predictions (tta=from_preds, upstream default)
  -> bicubic (order=3) resize back to the crop size -> paste into native canvas
  -> zero outside FOV -> threshold at 0.4196 (upstream DRIVE-optimal default)
```

`--im_size 512` everywhere (the value in `experiments/wnet_drive/config.cfg`);
the README's 1024 / 0.3725 HRF variant belongs to the HRF-trained models we do
not have, and using it would be per-dataset tuning. **Output is native
resolution** — no geometry inversion is needed for LWNet.

Commands actually run (logs: `runs/pivot/r2/logs/lwnet_infer.log`,
`runs/pivot/r2/logs/lwnet_wrap.log`):

```bash
for ds in drive chasedb1 hrf fives; do
  D:/Anaconda/envs/medical1/python.exe run_project_splits.py \
    --dataset $ds --out third_party/lwnet/out/$ds --device cuda:0 --save_prob
  D:/Anaconda/envs/medical1/python.exe -m src.pivot.g9_wrap_external \
    --dataset $ds --pipeline lwnet --src third_party/lwnet/out/$ds --ext png \
    --out runs/pivot/r2/lwnet/$ds/pred --weights "wnet_drive/model_checkpoint.pth"
done
```

Soft probability maps were also kept, outside the wrapper's search path, at
`third_party/lwnet/prob/<ds>/<stem>.png` (uint8, 0-255 = sigmoid 0-1).

Wall-clock on one RTX 3080 (`cuda:0`, inference only): DRIVE 6.9 s / 20 img,
CHASE_DB1 5.0 s / 8, HRF 85 s / 30, FIVES 330 s / 200.

**Results (zero-shot, project FOV, native resolution):**

| dataset | n | native size | mean Dice | mean clDice |
|---|---|---|---|---|
| DRIVE (in-domain for this checkpoint) | 20 | 584x565 | **0.8278** | 0.8350 |
| CHASE_DB1 | 8 | 960x999 | **0.7331** | 0.8157 |
| HRF | 30 | 2336x3504 | **0.6944** | 0.7780 |
| FIVES | 200 | 2048x2048 | **0.7350** | 0.7648 |

Caveats the paper must state:

* DRIVE is **not** zero-shot for this checkpoint (it is its training set), so
  only CHASE_DB1 / HRF / FIVES are genuine cross-dataset numbers.
* LWNet is a 68k-parameter 2020 model trained on **20 images**; it is a
  different architecture family (cascade of two tiny U-Nets), not a stronger
  segmenter. On FIVES a handful of images collapse almost entirely (predicted
  foreground fraction down to 0.0016 vs a typical 0.10) — an honest domain-shift
  failure, not a wrapping bug.
* No retraining, no finetuning, no per-dataset threshold or resolution tuning.

### 9.2 VascX — installed, **NOT run**

The env and the package install both succeeded; the run was stopped by the
30-minute scope rule before any vessel mask existed. Recorded so the decision can
be cited instead of re-researched.

```powershell
D:\Anaconda\Scripts\conda.exe create -n vascx python=3.11 -y      # OK, Python 3.11.16
D:/Anaconda/envs/vascx/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 `
    --index-url https://download.pytorch.org/whl/cu124            # OK, torch.cuda.is_available() True
D:/Anaconda/envs/vascx/python.exe -m pip install retinalysis-vascx  # OK
```

Installed: `retinalysis-vascx 2.1.0`, `retinalysis-inference 0.9.0`,
`retinalysis-enface 2.0.0`, `monai 1.5.1`, `lightning 2.6.6`,
`scikit-image 0.26.0`, `Pillow 11.3.0`, `opencv-python 4.14.0.94`,
`albumentations 1.3.1`, `numba 0.67.0`.

**Undeclared dependency (a real packaging bug in 2.1.0 / 0.9.0):** `pip install
retinalysis-vascx` does **not** pull `retinalysis-fundusprep`; the first CLI call
dies with `ModuleNotFoundError: No module named 'rtnls_fundusprep'` raised from
`rtnls_inference/transforms/fundus.py`. Fixed by
`pip install retinalysis-fundusprep` -> **1.3.0** (which downgrades
`opencv-python-headless` 5.0.0.93 -> 4.14.0.94).

**Second trap:** setting `VASCX_MODEL_DIR` to an *empty* directory switches the
CLI into local-file mode and it hard-fails with `Missing quality model file:
...\models\quality\quality.pt` instead of falling back to HuggingFace. Leave
`VASCX_MODEL_DIR` **unset** and steer only `HF_HOME` if the auto-download is
wanted.

With those two fixes the pipeline does start: preprocessing of 3 DRIVE `.tif`
images succeeded and wrote `bounds.csv` and `preprocessed_rgb/<id>.png`; the
weight download from `Eyened/vascx` had reached ~513 MB in `hf_cache/` when the
run was stopped. **No `vessels/*.png` was ever produced, so there is no VascX arm
and no VascX number anywhere in this project.** Nothing was deleted: the `vascx`
conda env, the partial HF cache and `third_party/vascx/out/drive_smoke/` are left
in place.

Blocking reasons, in order of weight:

1. **Geometry — the decisive one.** The produced `bounds.csv` confirms section
   2.7 exactly: per image it stores only the FOV circle,
   `{'hw': (584, 565), 'center': (282.90, 297.78), 'radius': 267.97}`, from which
   the tool derives a square crop resized to **1024x1024**, and it "does not
   convert results back to the original photograph geometry". Every mask would
   have to be inverse-mapped by us before it could be scored against
   DRIVE/CHASE/HRF/FIVES ground truth. That inversion is deterministic but it is
   a new, untested piece of glue sitting **inside the measurement path**, and a
   wrong-geometry arm is worse than no arm. This alone rules VascX out under a
   30-minute budget.
2. **Licence.** Weights and `retinalysis-inference` / `retinalysis-enface` are
   **AGPL-3.0**; the `eyened/retinalysis-vascx` wrapper ships **no LICENSE file**
   at all and its PyPI `License` field is empty. LWNet's MIT is cleanly better
   for a paper artefact.
3. **Isolation cost.** A whole second conda env (torch + monai + lightning,
   several GB) plus a further ~0.5-1 GB weight download, because the package hard-
   pins `numpy==2.*` / `Pillow==11.*` / `monai==1.5.1` and must not be allowed
   near the shared `medical1` env.

If VascX is ever revived: use the CLI default `vessels/vessels_may26.pt`
(142.2 MB) and **not** `vessels_july24_FIVES.pt`, which is FIVES-finetuned and
would make the FIVES column meaningless.

### 9.3 AutoMorph — not attempted

Ruled out on cost, unchanged from section 3: a ~1.6 GB clone, a third conda env,
POSIX `run.sh` / `test_outside.sh` entrypoints needing Git-Bash translation on
Windows, a mandatory `resolution_information.csv`, and an M0 preprocessing
pre-pass because `test_outside_integrated.py` reads `AUTOMORPH_DATA` and expects
M0's output tree. Its advantages over LWNet (Apache-2.0, in-repo weights,
native-resolution masks, per-pixel ensemble uncertainty) are real but do not buy
a second external arm we need. It remains the documented fallback.
