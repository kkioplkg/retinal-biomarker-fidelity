# External fundus datasets (E3 — downstream validity)

These four sets carry **disease labels and no vessel masks**. They exist for
E3 of `proposal/06_pivot_proposal_v3.md`: the mask-bearing sets cannot settle
whether ReliSeg's biomarkers are clinically more discriminative (HRF n=45,
FIVES n=800 with a ceiling), so the question is asked instead at n = 1.7k–6.4k
on sets where no mask is needed — predict the disease label from the four
primary biomarkers and compare frozen segmenters under one fixed protocol.

Raw files live under `exp/data/external/<name>/raw/` and are never modified.
FOV masks are generated with the project's own method and cached in
`exp/data/external/<name>/fov/`. Access everything through
`exp/src/data/external.py::load_external(name)`; verify with
`python -m src.data.check_external`.

| Dataset | Images | Subjects | Label | Native resolution | Role | Licence |
|---------|-------:|---------:|-------|-------------------|------|---------|
| APTOS-2019 | 3,662 | 3,662 | DR grade 0–4 | 17 sizes, 474→4288 longest | **primary** | competition terms; mirror card CC BY 4.0 |
| IDRiD (part B) | 516 | 516 | DR grade 0–4 + DME risk 0–2 | 4288×2848 (uniform) | **primary** | CC BY 4.0 (ships with the data) |
| Messidor-2 | 1,744 | 1,744 | DR grade 0–4 | 1440×960 / 2240×1488 / 2304×1536 | **primary** | Messidor-2 research licence (ADCIS) |
| ODIR-5K | 6,392 | 3,358 | 8 ocular-disease classes | **512×512 only** (see caveat) | secondary / sensitivity | MIT (mirror card) |

Total on disk ≈ 14 GB. **DR-HAGIS was not obtained** — see "Failed" below.

## Roles — which set carries a claim

**Primary: APTOS-2019, IDRiD, Messidor-2.** All three are native-resolution DR
grading sets on the same 5-point international scale, so the same ordinal
endpoints (QWK, macro-AUROC, referable-DR AUC) apply to all three and the
effect can be checked for consistency across three independent acquisitions.

**Secondary: ODIR-5K.** The only login-free mirror is the 512×512 preprocessed
folder, not the native ~2976×1984 images. 512 px is **below the resolution at
which the measurement mechanism exists at all** — the pivot probes showed the
fine-tuning effect does not reproduce on low-resolution DRIVE — so ODIR cannot
support a positive claim. It is a sensitivity set: if ReliSeg helps there too,
that is a bonus; if it does not, that is expected and says nothing. Its value
is the 8-way disease breadth (glaucoma, AMD, myopia, hypertension, cataract),
which no DR set has.

---

## 1. APTOS-2019 Blindness Detection

- **Source used:** HuggingFace `Tejaswini628/aptos-fundus-images`,
  `data/train_images/*.png` + `data/train.csv`. A verbatim copy of the Kaggle
  competition files (APTOS images are natively PNG — nothing was re-encoded).
  Every one of the 3,662 files was downloaded with its byte length checked
  against the HuggingFace API size; 0 mismatches, 0 failures, 8.60 GB.
- **Why a mirror:** no `~/.kaggle/kaggle.json` on this machine and no
  `KAGGLE_*` environment variables, so the competition API is unusable.
- **Origin:** Asia Pacific Tele-Ophthalmology Society / Aravind Eye Hospital,
  Madurai; Kaggle competition `aptos2019-blindness-detection`, 2019.
- **Canonical home:** <https://www.kaggle.com/c/aptos2019-blindness-detection/data>
- **Licence / terms:** the Kaggle competition rules (research use, attribution
  to APTOS); the mirror's dataset card states CC BY 4.0. Cite the competition.
- **Counts (verified):** 3,662 images, grade histogram
  **0:1805, 1:370, 2:999, 3:193, 4:295** — exactly the published APTOS train
  distribution. `train.csv` has 3,662 rows + header.
- **Split:** only the *train* split exists here. The competition's 1,928 test
  images were never released with labels, so they were deliberately **not**
  downloaded (they would add ~3 GB of unusable pixels). `load_external` reports
  `split="train"` for every row; E3 makes its own CV folds.
- **Resolution — the reason APTOS is the interesting set:** 17 distinct sizes,
  a **72× spread in pixel count** (0.17 → 12.21 MP).

  | size | n | | size | n |
  |------|--:|-|------|--:|
  | 1050×1050 | 974 | | 3388×2588 | 141 |
  | 2416×1736 | 638 | | 1504×1000 | 92 |
  | 2588×1958 | 533 | | 1844×1226 | 61 |
  | 3216×2136 | 410 | | 4288×2848 | 52 |
  | 2048×1536 | 351 | | 640×480 | 42 |
  | 819×614 | 287 | | 2896×1944 | 34 |
  | | | | 2144×1424 | 28 |
  | | | | 1476×1117 | 14 |
  | | | | 474×358, 1467×1110, 2146×1764 | 2, 2, 1 |

  Longest side: min 474, p25 1050, median 2144, p75 2588, max 4288. This makes
  APTOS a built-in resolution-fidelity experiment: the same protocol is applied
  to sub-megapixel and 12-megapixel images, and `e3_external.py` records every
  image's native size and realised inference scale so the biomarker effect can
  be stratified by resolution rather than confounded by it.
- **Subject grouping:** APTOS publishes no patient id, so one image = one
  subject. Stated explicitly because grouped CV silently degenerates to
  row-level CV here — which is correct for APTOS and wrong for ODIR.

## 2. IDRiD — part B (Disease Grading)

- **Source used:** HuggingFace `MahsaTorki/IDRiD_Dataset`,
  `B.Disease_Grading.zip` (212,405,123 bytes), extracted in place. The other
  two parts of the archive (segmentation, localization) were not downloaded.
- **Why a mirror:** the canonical home is IEEE DataPort, which requires a
  login. The mirror carries the official directory layout, the official label
  CSVs and the original `LICENSE.txt` + `CC-BY-4.0.txt`.
- **Origin:** Porwal et al., *Data* 3(3):25, 2018 — "Indian Diabetic
  Retinopathy Image Dataset (IDRiD)". Kowa VX-10α, 50° FOV, Nanded, India.
- **Canonical home:** <https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid>
- **Licence:** **CC BY 4.0**, shipped with the data.
- **Counts (verified):** 413 train + 103 test = 516 images.
  DR grade **0:168, 1:25, 2:168, 3:93, 4:62**; DME risk **0:222, 1:51, 2:243**.
  Per split: train 134/20/136/74/49, test 34/5/32/19/13 — the official numbers.
- **Resolution:** 4288×2848 for all 516 (12.21 MP), the highest and most
  uniform of the four sets. IDRiD is therefore the clean high-resolution
  control against APTOS' heterogeneity.
- **Split:** the official 413/103.
- **Gotcha — numbering restarts per split.** The training set is
  `IDRiD_001 … IDRiD_413` and the testing set `IDRiD_001 … IDRiD_103`, so 103
  pairs of *different* images share a bare stem. `image_id` and `subject_id`
  are split-prefixed (`train_IDRiD_001`) — the same collision FIVES had in
  `datasets.py`. A loader that keys on the stem silently merges 103 image pairs.
- **Second label:** the DME risk grade is kept as a second target
  (`y_dme_risk`), giving a non-DR endpoint on a native-resolution set.

## 3. Messidor-2

- **Source used:** HuggingFace `sngsfydy/Messidor2`, 5 parquet shards
  (2.54 GB), materialised to `raw/images/*.jpg` + `raw/labels.csv` by
  `python -m src.data.check_external --prepare messidor2`.
- **Origin:** the Messidor-2 database (Messidor programme partners / ADCIS),
  with the 0–4 grades of the adjudicated release used in
  Abràmoff et al. 2016 / Krause et al. 2018.
- **Canonical home:** <https://www.adcis.net/en/third-party/messidor2/>
  (registration form; the grades are distributed separately).
- **Licence / terms:** Messidor-2 is free for research with attribution and
  no redistribution for commercial use; the Messidor programme partners must be
  acknowledged. The mirror's own card carries no licence field.
- **Counts (verified):** 1,744 images, grade histogram
  **0:1017, 1:270, 2:347, 3:75, 4:35** — exactly the published distribution of
  the 1,744 gradable images of the adjudicated Messidor-2 release (1,748 total
  minus 4 ungradable).
- **Resolution (verified):** three sizes and only three — 1440×960 (527),
  2240×1488 (614), 2304×1536 (603) — which are precisely Messidor-2's three
  acquisition resolutions. Together with the exact grade histogram this is
  strong evidence the mirror is the real set and not a re-graded subset.
- **Which label file:** the grades are **embedded in the mirror's parquet**
  (`label`, a 0–4 `ClassLabel`); there is no separate CSV and the mirror cites
  no grade file. Provenance is therefore established by distribution matching
  (histogram + resolution triple), not by a checksum against an official file.
  Recorded here because it is weaker evidence than APTOS/IDRiD have.
- **Caveat — no filenames, so no examination pairing.** The mirror stores only
  `image` and `label`; the original Messidor-2 filenames are gone. Messidor-2's
  1,748 images are 874 two-eye examinations, and without the filenames those
  pairs cannot be reconstructed. Consequence: `image_id` is the shard/row index
  (`messidor2_00000`) and **one image = one subject**, so a patient-level fold
  can put both eyes of an examination on opposite sides of the split. This is a
  real (small) optimism in the Messidor-2 numbers and must be stated in the
  paper. `raw/labels.csv` keeps `source_shard` so the order is reproducible.
- **Split:** none official; E3 makes its own CV folds.
- **OVERLAP with the MAPLES-DR replication cohort (added 2026-09-17).**
  MAPLES-DR annotates 198 *original* MESSIDOR images, and Messidor-2 contains
  most of original MESSIDOR. Because this mirror lost the filenames, the overlap
  had to be measured on pixels: `python -m src.data.maples_overlap` matches each
  MAPLES-DR vessel mask to the image it was drawn on (band-passed green-channel
  correlation within the same native size). **162 of the 198 MAPLES-DR images
  are in this mirror**; accepted matches lead their 527–614-image candidate pool
  by 14.9–87.0 sd against 2.0–8.5 sd for the rejected ones, and form a
  bijection. The 162 `image_id`s are listed in
  `exp/data/external/messidor2/overlap_maplesdr.csv` and **must be excluded from
  the E3 Messidor-2 cohort (1,744 → 1,582) whenever MAPLES-DR is used as a
  replication cohort**, or the two analyses are not independent. `load_external`
  does not apply the exclusion automatically — it is an analysis-time decision,
  and dropping rows silently would change every Messidor-2 number already
  recorded. See `exp/data/DATASETS.md` §7 and
  `research/08_replication_datasets.md` §4.

## 4. ODIR-5K — secondary, low-resolution

- **Source used:** HuggingFace `bumbledeep/odir`, one parquet (397 MB),
  extracted to `raw/images/*.jpg` + `raw/labels.csv`.
- **Origin:** Ocular Disease Intelligent Recognition, Peking University /
  Shanggong Medical Technology, ODIR-2019 challenge. Real-world data from
  Chinese hospitals, mixed Canon / Zeiss / Kowa cameras.
- **Canonical home:** <https://odir2019.grand-challenge.org/dataset/>
  (login; the download page 404s) and the Kaggle mirror
  `andrewmvd/ocular-disease-recognition-odir5k`.
- **Licence:** the mirror's card states MIT; the original challenge terms are
  research use with attribution to the ODIR-2019 organisers.
- **Counts (verified):** 6,392 eye images from 3,358 patients
  (3,034 patients with both eyes, 324 with one; left 3,198 / right 3,194).
  Class histogram **normal 2873, diabetes 1608, other 708, cataract 293,
  glaucoma 284, AMD 266, pathological myopia 232, hypertension 128** — the
  published per-eye distribution of the ODIR-5K training set.
- **Extra fields:** `age`, `sex`, and eye laterality from the filename. Age and
  sex are used as covariates in the adjusted model; no camera/site field
  survives in this mirror.
- **CAVEAT — resolution.** Every image is **512×512 JPEG**. This is the Kaggle
  `preprocessed_images` folder, not ODIR's native ~2976×1984. Downsampling by
  ~6× destroys exactly the fine-calibre structure the biomarkers measure, and
  the project has already shown the ReliSeg mechanism does not reproduce at
  DRIVE-scale resolution. ODIR therefore **cannot support a positive claim**
  and is reported as a low-resolution sensitivity analysis only.
- **CAVEAT — single label per eye.** The mirror gives one of the 8 classes per
  eye, not the ODIR per-patient 8-way multi-label. Multi-label metrics are not
  available; the target is an 8-class macro-AUROC/AUPRC.
- **Subject grouping matters here.** Both eyes of a patient share
  `subject_id` and are highly correlated; eye-level CV would leak. E3 groups on
  `subject_id` in both the folds and the bootstrap.
- **Route to the native-resolution set, if it is ever wanted:** an Academic
  Torrents mirror exists (`cf3b8d5ecdd4284eb9b3a80fcfe9b1d621548f72`), but no
  torrent client is installed on this machine, so it was not pursued.

---

## FOV generation

The external sets ship no field-of-view mask, so one is generated per image
with **the project's existing generator**, `src/data/datasets._generate_fov`,
unchanged: red channel thresholded at 10% of its maximum → morphological
closing (radius 9, decimated above 1024 px) → fill holes → largest connected
component, saved 8-bit {0,255}. Using the same generator matters because every
biomarker is defined *inside the FOV*; two different FOV rules would shift
density and total length between the internal and external sets and look like a
segmenter effect.

Masks are **not** built when `load_external` is called (APTOS 3,662 + ODIR
6,392 images would turn a metadata call into a ten-minute job). Call
`ensure_fov(rec)` for one, `ensure_fovs(name)` for a set, or
`python -m src.data.check_external --fov <name>`. E3's biomarker stage calls
`ensure_fov` per image, so the cache fills as it runs.

> ### FOV prebuild — the race is fixed, the prebuild is still recommended
>
> **Fixed 2026-09-17 21:55** (DECISIONS 21:55): `ensure_fov` now generates the
> mask into a pid-unique temp file in the same directory and `os.replace()`s it
> into position, so a reader sees either no file or a complete one and the
> race below can no longer occur. Verified by regenerating masks and checking
> them byte-identical to the originals — the fix changes no pixel.
>
> **Still recommended, for reasons the fix does not address**: prebuilding
> single-process keeps FOV generation out of the timed inference loop (so
> per-image timings mean what they say), surfaces a corrupt or unreadable
> source image immediately rather than hours into a sharded wave, and lets you
> check the mask count against the record count before committing a long run.
>
> **Prebuild the FOV cache single-process, and verify the mask count against
> the record count, before starting more than one pass over the same dataset.**
>
> ```bash
> python -m src.data.check_external --fov messidor2   # then aptos2019, idrid, odir5k
> ls data/external/messidor2/fov/*.png | wc -l        # must equal len(load_external('messidor2'))
> ```
>
> *What the race was, kept because the failure signature is worth recognising:*
> `ensure_fov` **used to write** the mask directly to its target path, not
> atomically. Two passes over the *same* dataset running at the same time both
> found the same mask missing and both generated it; one could read the file
> while the other was still writing it and get a truncated PNG. imageio then
> failed to recognise it as a PNG, fell back to another plugin, and the job died
> with something that looked nothing like an I/O error — observed 2026-09-17 as
> `ValueError: cannot reshape array of size 1701 into shape (36396,60828)`
> raised from the SPE plugin, killing a Messidor-2 pass at image 181/1744
> (DECISIONS 2026-09-17 17:23). The file is transient: the surviving writer
> completes it, so by the time you investigate, every mask on disk validates
> and the crash looks unreproducible.
>
> Different datasets in parallel were always safe — the caches are separate
> directories. It was **same dataset, concurrent passes** that was unsafe.
>
> Worked example (LAN node, 2026-09-17): FOV caches prebuilt per set before
> each wave fanned out to 9–11 workers — messidor2 finished 12:42:38 with the
> wave starting 13:20:18, aptos2019 finished 13:39:22 with the wave starting
> 14:28 — so no worker ever reached the write path. Verified afterwards by
> re-reading all 5,922 masks: 0 unreadable, 0 degenerate, every job `rc=0`,
> and 0 NaN across 56,416 rows of the four skan biomarker columns.
>
> **Applied 2026-09-17 21:55**, after every E3 bio job on all three nodes was
> complete and pulled. All 43 delivered E3 biomarker tables were produced under
> the previous `external.py` (SHA-256 `6e96340d…`); the current file is
> `1693aaa7…`. The change cannot alter any of them: it only moves where the
> mask bytes are written before being renamed into place, and regenerated masks
> are byte-identical to the originals.

**Validation without vessel ground truth.** The criterion used for the
segmentation sets ("fraction of annotated vessel pixels inside the mask") does
not exist here. `check_external.py` instead requires, on a 12-image sample per
set: exactly one connected component, a plausible frame coverage, and a mean
green intensity **inside** the mask far above the mean outside it. Observed:

| dataset | FOV coverage (mean / min / max) | in/out green ratio (min) | components |
|---------|--------------------------------|-------------------------:|------------|
| APTOS-2019 | 0.791 / 0.475 / 0.902 | 3.9 | 1 |
| IDRiD | 0.691 / 0.691 / 0.693 | 24.1 | 1 |
| Messidor-2 | 0.463 / 0.448 / 0.469 | 20.7 | 1 |
| ODIR-5K | 0.793 / 0.777 / 0.845 | 289.7 | 1 |

APTOS' low in/out ratio and wide coverage spread are expected and are the
dataset itself, not a mask failure: many APTOS images are already cropped tight
to the retina, so "outside the FOV" is a thin sliver of dark rim rather than a
large black surround.

## Inference convention used by E3

`src/pivot/e3_external.py` crops each image to the FOV bounding box, resizes the
longest side to **1024**, runs sliding-window inference at the checkpoint's
training patch size (stride = patch/2, Gaussian blending, no flips), resizes the
probability map back to the cropped native size and thresholds at 0.5 inside the
FOV. Cropping first is what makes scale comparable across APTOS' wildly varying
black borders. The HRF/FIVES checkpoints were trained at longest side 1536, so
1024 is a deliberate departure; every row records `native_*`, `crop_*`,
`infer_*` and the realised `scale`, and `--resolution-record` dumps them
separately, so the fidelity-vs-resolution question can be answered from the data
rather than assumed away.

## Failed / not obtained

- **DR-HAGIS (40 images, DR / hypertension / AMD / glaucoma, with vessel
  masks)** — **unavailable.** The sole distribution point was
  `personalpages.manchester.ac.uk/staff/niall.p.mcloughlin/DRHAGIS.zip`. The
  whole personal page is gone: the server returns Apache's "Object not found!"
  (404) for the directory and for every filename variant, with a browser
  User-Agent too. The Wayback Machine holds the 2017 description page
  (`DRHAGIS_database.htm`, which confirms `DRHAGIS.zip` was the only link) but
  only two 2025-10-26 captures of the zip URL itself, both 301/404 — the
  archive never captured the file's bytes. No copy on HuggingFace, Zenodo or
  GitHub (searched by name and by paper title). Dropped; it is 40 images and
  the three primary sets cover the endpoint.
- **Kaggle direct downloads** (APTOS competition files, ODIR-5K, EyePACS) —
  `~/.kaggle/kaggle.json` does not exist on this machine and no `KAGGLE_*`
  variables are set, so the Kaggle API cannot authenticate. Mirrors were used
  instead; nothing was fabricated.
- **ODIR-5K at native resolution** — no login-free mirror found on HuggingFace
  (name search, full-text search across dataset cards), Zenodo or GitHub. The
  Academic Torrents mirror needs a torrent client that is not installed.
- **`huggingface_hub`** is not installed in the `medical1` env, so downloads
  went through a small threaded `requests` fetcher against the HF tree/resolve
  API, with per-file byte-length verification.
- **HF tree API gotcha:** `?expand=true` 400s for `limit > 100`. The plain
  listing already carries `size` / `lfs.size`, so pagination uses
  `?recursive=true&limit=1000` without `expand`.

## Smoke run (CPU, 50 APTOS images, two frozen checkpoints)

Recorded so the pipeline's shape is verifiable before the real ReliSeg
checkpoints land. **These are not results** — n=50, a single grade-3 pair, and
neither checkpoint is ReliSeg. Files are prefixed `SMOKE_`.

    python -m src.pivot.e3_external bio --dataset aptos2019 \
        --ckpt runs/seg/hrf/seed0/best.pt   --device cpu --limit 50 \
        --tag smoke_hrf_seed0_best   --sw-batch 1 --resolution-record
    python -m src.pivot.e3_external bio --dataset aptos2019 \
        --ckpt runs/seg/fives/seed0/best.pt --device cpu --limit 50 \
        --tag smoke_fives_seed0_best --sw-batch 1
    python -m src.pivot.e3_external clf --dataset aptos2019 \
        --tags smoke_hrf_seed0_best,smoke_fives_seed0_best --n-boot 200

- per-image biomarkers: `results/pivot/e3/aptos2019/<tag>/bio.csv` (50 rows,
  47 columns), `meta.json` with the checkpoint SHA-256, `resolutions.csv`
- summary: `results/pivot/e3/aptos2019/SMOKE_summary.csv`,
  paired differences `SMOKE_delta.csv`, out-of-fold probabilities `oof_*.npy`
- throughput: ~8 s/image on CPU (40 threads, `--sw-batch 1`), so a full APTOS
  pass is ~8 h on CPU and should be run on a GPU.

The two frozen segmenters already produce visibly different biomarkers on the
same 50 images (mean vessel density 0.079 vs 0.101, mean FD 1.166 vs 1.148),
and the pipeline turns that into a paired ΔAUC with a subject-level bootstrap
interval — which is exactly the quantity E3 exists to measure.
