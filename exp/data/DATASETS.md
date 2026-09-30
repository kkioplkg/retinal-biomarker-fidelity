# Retinal vessel segmentation datasets

All five datasets are public. Raw files are stored verbatim under
`exp/data/<name>/raw/` — nothing has been resampled, re-encoded, renamed or
otherwise altered. Auto-generated field-of-view (FOV) masks live in a separate
`exp/data/<name>/fov/` tree so the raw trees stay pristine.

Access everything through `exp/src/data/datasets.py::load_dataset(name)`.

Datasets with **disease labels but no vessel masks** (APTOS-2019, IDRiD,
Messidor-2, ODIR-5K), used for the E3 downstream-validity experiment, live
under `exp/data/external/` and are documented separately in
[EXTERNAL_DATASETS.md](EXTERNAL_DATASETS.md); load them with
`exp/src/data/external.py::load_external(name)`.

| Dataset   | Images | Resolution  | Observers | Official FOV | Split used (train/test) |
|-----------|-------:|-------------|-----------|--------------|-------------------------|
| DRIVE     |     40 | 565 x 584   | 2 (test only) | yes      | 20 / 20 (official)      |
| CHASE_DB1 |     28 | 999 x 960   | 2         | no (generated) | 20 / 8 (convention)  |
| HRF       |     45 | 3504 x 2336 | 1         | yes          | 15 / 30 (convention)    |
| FIVES     |    800 | 2048 x 2048 | 1         | no (generated) | 600 / 200 (official) |
| STARE     |     20 | 700 x 605   | 2         | no (generated) | 10 / 10 (fixed)      |

Total: **933 images**, all with pixel-level vessel ground truth.

Two further mask-bearing sets were added on 2026-09-17 as **independent
replication cohorts** (DECISIONS.md 2026-09-17 16:40; rationale and the full
candidate ranking in `research/08_replication_datasets.md`). They are
deliberately **not** in `datasets.DATASETS` — that list still names exactly the
five sets the project trains and reports on, so adding these cannot silently
change an existing loop. Ask for them by name, or use `ALL_DATASETS`.

| Dataset     | Images | Resolution | FOV diameter | Observers | Official FOV | Split (train/test) |
|-------------|-------:|------------|-------------:|-----------|--------------|--------------------|
| Fundus-AVSeg |   100 | 1280 x 1280 (79), 2656 x 1992 (21) | 1238 / 2080 | 1 (+ senior review) | no (generated) | 80 / 20 (official) |
| MAPLES-DR   | 198 (**162 usable**) | 1440 x 960 (104), 2240 x 1488 (53), 2304 x 1536 (41) | 909 / 1380 / 1452 | 1 of 7 retinologists per image | no (generated) | 138 / 60 (official) |

---

## 1. DRIVE

*Digital Retinal Images for Vessel Extraction.*

- **Source used:** `https://github.com/rajsahu2004/Retina-Blood-Vessel-Segmentation-using-UNET`,
  path `data/{training,test}/…`. The image bytes there are stored with Git LFS.
  The repo's `.gitattributes` is corrupt, so `git lfs pull` does **not** work;
  the files were instead recovered by reading each pointer's `oid`/`size` and
  POSTing them to GitHub's LFS batch endpoint
  `https://github.com/rajsahu2004/Retina-Blood-Vessel-Segmentation-using-UNET.git/info/lfs/objects/batch`,
  then downloading the returned signed URLs. All 140 files verified: byte length
  matches the declared LFS size and the content SHA-256 matches the pointer oid,
  so the data is bit-identical to what was committed.
- **Origin:** Image Sciences Institute, University Medical Center Utrecht.
  Canon CR5 non-mydriatic 3CCD, 45 degree FOV. Staal et al., *IEEE TMI* 23(4):501-509, 2004.
- **Canonical home:** <https://drive.grand-challenge.org/> (login required — hence the mirror).
- **License / terms:** free for research and educational use; redistribution for
  commercial purposes is not permitted. Cite Staal et al. 2004.
- **Counts (verified):**
  - `raw/training/images` 20 (`21_training.tif` … `40_training.tif`)
  - `raw/training/1st_manual` 20, `raw/training/mask` 20
  - `raw/test/images` 20 (`01_test.tif` … `20_test.tif`)
  - `raw/test/1st_manual` 20, `raw/test/2nd_manual` 20, `raw/test/mask` 20
- **Resolution:** 565 x 584 RGB TIFF; labels and FOV masks are 565 x 584 GIF.
- **Split convention:** the **official** 20/20. Training set is images 21-40,
  test set is images 01-20. `subject_id` is the two-digit image number.
- **2nd observer:** test set only (`*_manual2.gif`), used as the human-performance
  reference, never for training.
- **Gotcha - label value ranges differ:** `1st_manual` GIFs are `{0, 255}` but
  `2nd_manual` GIFs are `{0, 1}`. A loader that hard-codes `> 127` silently
  returns an all-empty second-observer mask. `datasets.read_binary` thresholds
  relative to each file's own maximum, so it handles both, along with
  CHASE_DB1's 1-bit boolean PNGs and FIVES' 3-channel RGB masks.

## 2. CHASE_DB1

*Child Heart and Health Study in England, database 1.*

- **Source used:** `https://raw.githubusercontent.com/playyalp1/CHASE_DB1/main/CHASEDB1.zip`
  — a verbatim copy of the official `CHASEDB1.zip` (84 files, timestamps 2014-04-24).
- **Origin:** Kingston University / St George's, University of London.
  Nikon NM-200D, 30 degree FOV, both eyes of 14 school children.
  Fraz et al., *IEEE TBME* 59(9):2538-2548, 2012.
- **Canonical home:** <https://researchdata.kingston.ac.uk/96/>
  (the older `blogs.kingston.ac.uk/retinal/chasedb1` URL is dead).
- **License / terms:** free for non-commercial research use. Cite Fraz et al. 2012.
- **Counts (verified):** 28 `Image_XXY.jpg` + 28 `*_1stHO.png` + 28 `*_2ndHO.png` = 84 files.
- **Resolution:** 999 x 960 RGB JPEG; labels are 999 x 960 1-bit PNG.
- **Split convention:** first 20 images train / last 8 test, the split used by
  most published work. Sorted order is `Image_01L, Image_01R, Image_02L, …`, so
  the first 20 are **subjects 01-10** and the last 8 are **subjects 11-14**.
  The split is therefore **subject-disjoint** — an important property, since the
  left and right eye of one child are highly correlated. `subject_id` is the
  two-digit child number, shared by the L and R image of that child.
- **2nd observer:** all 28 images (`_2ndHO.png`).
- **FOV:** none shipped. Generated (see "FOV generation" below).

## 3. HRF

*High-Resolution Fundus image database.*

- **Source used:** <https://www5.cs.fau.de/fileadmin/research/datasets/fundus-images/all.zip>
  (the "all" archive, 76.3 MB) — the official download, still live.
- **Origin:** Pattern Recognition Lab, FAU Erlangen-Nuremberg, with Brno
  University of Technology and Masaryk University.
  Budai et al., *Int. J. Biomed. Imaging*, 2013.
- **Canonical home:** <https://www5.cs.fau.de/research/data/fundus-images/>
- **License / terms:** free for research; cite Budai et al. 2013.
- **Counts (verified):** `raw/images` 45, `raw/manual1` 45, `raw/mask` 45.
  Three diagnostic groups of 15 each: `h` healthy, `g` glaucomatous,
  `dr` diabetic retinopathy (`01_h.jpg` … `15_h.jpg`, etc.;
  the `dr` images use an uppercase `.JPG` extension).
- **Resolution:** 3504 x 2336 RGB JPEG; labels and FOV masks are 3504 x 2336 TIFF.
- **Split convention:** the common 15/30 — **subjects 01-05 of each of the three
  classes** form the 15-image training set (5 healthy + 5 glaucoma + 5 DR), and
  subjects 06-15 of each class form the 30-image test set. `subject_id` is
  `<num>_<class>`; the record also carries a `disease` field.
- **2nd observer:** none.

## 4. FIVES

*A Fundus Image Dataset for AI-based Vessel Segmentation.*

- **Source used:** <https://ndownloader.figshare.com/files/34969398>
  (1,764,974,308 bytes, MD5 `789c80dd5376a82063e27fa49192bac9` — verified against
  the figshare API record). Note the `figshare.com/ndownloader/...` form returns
  HTTP 202 and an empty body; the `ndownloader.figshare.com` host must be used.
  The archive is RAR; extracted with `bsdtar` (libarchive, ships with Anaconda),
  since no `unrar`/7-Zip is installed on this machine.
- **Origin:** Jin et al., *Scientific Data* 9:475, 2022.
- **Canonical home:** <https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169>
- **License:** CC BY 4.0.
- **Counts (verified):**
  - `train/Original` 600 images, `train/Ground truth` 600
  - `test/Original` 200 images, `test/Ground truth` 200
  - (`train/Original` also contains one stray `Thumbs.db`, which the loader skips.)
- **Resolution:** 2048 x 2048 RGB PNG; labels 2048 x 2048 PNG.
- **Disease classes** are encoded in the filename suffix `<n>_<C>.png`:
  `A` = AMD, `D` = diabetic retinopathy, `G` = glaucoma, `N` = normal.
  Balanced: 150 of each class in train, 50 of each in test.
- **Split convention:** the **official** train/test directories (600/200).
  `image_id` **and** `subject_id` are both the split-prefixed stem
  (`train_100_A`), not the bare stem. FIVES restarts its numbering in each
  split, so `train/10_A.png` and `test/10_A.png` are different images; using the
  bare stem produced **50 subject_id collisions** across the train/test
  boundary, which would make subject-grouped statistics treat a training image
  and a test image as the same subject. FIVES has one image per subject, so
  after the fix image-level and subject-level grouping coincide (800 subjects).
  The record also carries a `disease` field (AMD / DR / Glaucoma / Normal).
- **2nd observer:** none. A per-image `Quality Assessment.xlsx` ships with the archive.
- **FOV:** none shipped. Generated.

## 5. STARE

*STructured Analysis of the Retina — vessel probing set.*

- **Source used:** <https://cecas.clemson.edu/~ahoover/stare/probing/> —
  `stare-images.tar`, `labels-ah.tar`, `labels-vk.tar` (official, still live).
  Files inside are gzipped PPM; decompressed in place (the original `.tar`
  archives are kept alongside).
- **Origin:** Adam Hoover, Clemson University. TopCon TRV-50, 35 degree FOV.
  Hoover et al., *IEEE TMI* 19(3):203-210, 2000.
- **Canonical home:** <https://cecas.clemson.edu/~ahoover/stare/>
- **License / terms:** free for research use; cite Hoover et al. 2000.
- **Counts (verified):** `raw/images` 20, `raw/labels-ah` 20, `raw/labels-vk` 20.
  Image ids are **not** contiguous: `im0001-im0005, im0044, im0077, im0081,
  im0082, im0139, im0162, im0163, im0235, im0236, im0239, im0240, im0255,
  im0291, im0319, im0324`.
- **Resolution:** 700 x 605 RGB PPM (P6); labels are 700 x 605 PGM (P5).
- **Split convention:** STARE has **no official split** and is usually reported
  leave-one-out. For reproducibility we fix a **10/10 split** by image number:
  - **train:** im0001, im0002, im0003, im0004, im0005, im0044, im0077, im0081, im0082, im0139
  - **test:**  im0162, im0163, im0235, im0236, im0239, im0240, im0255, im0291, im0319, im0324

  This is recorded in `STARE_TRAIN` / `STARE_TEST` in `datasets.py` and must be
  cited as a fixed split when comparing against leave-one-out numbers.
  `subject_id` is the image id (e.g. `im0001`); STARE has one image per subject.
- **2nd observer:** yes — `ah` (Adam Hoover) is the primary label used everywhere,
  `vk` (Valentina Kouznetsova) is the second observer. `vk` labels vessels more
  liberally, so `ah` vs `vk` agreement is markedly lower than DRIVE's two observers.

## 6. Fundus-AVSeg — primary replication cohort

*A Fundus Image Dataset for AI-based Artery-Vein Vessel Segmentation.*

- **Source used:** <https://ndownloader.figshare.com/files/54093641>
  (212,768,224 bytes, MD5 `20c85c9343ff95435f131b684afadd50` — verified against
  the figshare API `computed_md5`). Plain zip; the archive is kept in
  `raw/Fundus-AVSeg.zip` next to its extracted tree.
- **Origin:** Deng et al., *Scientific Data* 12, 2025-07-25,
  doi:10.1038/s41597-025-05381-2. Shenzhen Eye Hospital image database,
  Zeiss VISUCAM200 and Canon fundus cameras, collected Jan–Mar 2022.
- **Canonical home:** <https://doi.org/10.6084/m9.figshare.27938034> (v2)
- **Licence:** **CC BY 4.0** on the figshare deposit (the *article* is
  CC BY-NC-ND 4.0 per Crossref — cite the paper, use the data under CC BY).
- **Counts (verified):** `raw/Fundus-AVSeg/images` 100 PNG,
  `raw/Fundus-AVSeg/annotation` 100 PNG (identical filenames),
  `metadata.xlsx` 100 rows, `training.txt` 80, `testing.txt` 20.
- **Resolution (verified):** 1280×1280 (79 images) and 2656×1992 (21).
  Both are tight crops — measured FOV diameter 1238 px and 2080 px, i.e. the
  1280 px images have a *larger* retina than any MAPLES-DR image below the top
  stratum, and the 2656 px ones sit exactly in the FIVES regime (2013 px).
- **Disease classes** are the same four as FIVES, encoded the same way in the
  filename `<nnn>_<C>.png`: `A` AMD, `D` DR, `G` Glaucoma, `N` Normal.
  Balanced **40 N / 20 D / 20 A / 20 G**.
- **Extra fields** from `metadata.xlsx`, both attached to the record: `eye`
  (55 left / 45 right) and `quality` (83 High-quality / 17 Low-quality) — a real
  per-image quality grade, which none of the five training sets provides.
- **Split convention:** the **official** `training.txt` / `testing.txt` (80/20).
  `image_id` = `subject_id` = the file stem (`001_G`); there are no patient
  identifiers, so one image = one subject and two eyes of one patient could in
  principle fall on opposite sides of the split.
- **2nd observer:** none.
- **Gotcha — the shipped annotation is not a binary mask.** It is a 5-colour
  artery/vein map: red `(255,0,0)` artery (3.7 % of frame), blue `(0,0,255)`
  vein (4.3 %), green `(0,255,0)` A/V crossing (0.12 %), white `(255,255,255)`
  vessel of uncertain class (0.11 %), black background. `read_binary` would keep
  only the red channel and silently **drop every vein**. The loader therefore
  derives the binary vessel mask as the union of all four non-background
  classes and caches it at `exp/data/fundusavseg/labels/<image_id>.png`
  (8-bit {0,255}); the raw tree is untouched and the original A/V map stays on
  the record as `av_path`. Rebuild with `load_dataset(..., regenerate_labels=True)`.
- **FOV:** none shipped. Generated (see "FOV generation"). Checked: GT-in-FOV
  mean 1.0000, worst image 0.9995; FOV coverage of frame 0.642–0.785.

## 7. MAPLES-DR — secondary replication cohort (labels only)

*MESSIDOR Anatomical and Pathological Labels for Explainable Screening of DR.*

- **Source used (labels):** <https://ndownloader.figshare.com/files/45795381>
  (`MAPLES-DR.zip`, 14,765,980 bytes, MD5 `ee8bf3591abcf6c261f85eb6f61416bc`)
  and <https://ndownloader.figshare.com/files/45795384> (`AdditionalData.zip`,
  40,782,288 bytes, MD5 `d77c5891d881ca5cf13ca4c70f52fd49`) — both MD5s
  verified against the figshare API. Archives kept in `raw/` beside the
  extracted trees.
- **Origin:** Lepetit-Aimon, Playout, Boucher, Duval, Brent, Cheriet,
  *Scientific Data* 11:914, 2024-08-23, doi:10.1038/s41597-024-03739-6.
  Seven senior retinologists (Montréal / Toronto), Feb 2019 – Feb 2020.
- **Canonical home:** <https://doi.org/10.6084/m9.figshare.24328660> (v3),
  code + docs at <https://github.com/LIV4D/MAPLES-DR>.
- **Licence:** **CC BY 4.0** for the labels. The *images* are MESSIDOR and carry
  the MESSIDOR research licence (research and education only, no redistribution,
  acknowledge the Messidor programme partners).
- **THE IMAGES ARE NOT IN THE DEPOSIT.** MAPLES-DR ships labels only. The
  fundus images must be fetched from the MESSIDOR consortium at
  <https://www.adcis.net/en/third-party/messidor/>, which requires a
  personal-information form plus email verification (self-serve, no
  institutional sign-off — but not automatable, and the site's own
  `/en/download/messidor-bib/` endpoint currently returns HTTP 500).
  **162 of the 198 images were instead resolved from this project's existing
  Messidor-2 mirror** — see "Image resolution" below.
- **Counts (verified):** `raw/train/Vessels` 138, `raw/test/Vessels` 60 = 198;
  11 further biomarkers per image (OpticDisc, OpticCup, Macula, Exudates,
  CottonWoolSpots, Drusens, Microaneurysms, Hemorrhages, Neovascularization,
  BrightUncertains, RedUncertains — OpticCup is missing on 6 and Macula on 1).
  `raw/AdditionalData/` holds the same 200 annotations (198 unique + 2
  duplicates) plus the network `preannotations/` and three metadata files.
- **Resolution (verified):** labels are at native MESSIDOR size —
  1440×960 (104), 2240×1488 (53), 2304×1536 (41). MESSIDOR frames keep a wide
  black surround, so the **retina** is only 909 / 1380 / 1452 px across: the
  1440×960 stratum is *below CHASE_DB1* in effective resolution even though it
  passes a "longest side ≥ 1400" test. Stratify, do not average.
- **Disease labels:** consensus DR grade `R0`–`R4A` on `disease` and consensus
  ME grade `M0`–`M6` on `me_grade`, read from
  `raw/AdditionalData/diagnosis_infos.xls`, which also keeps the three
  individual retinologists' grades. On the 162 resolvable images the DR
  distribution is **R0 12, R1 129, R2 10, R3 7, R4A 4** — far too imbalanced for
  a disease-stratified endpoint.
- **Split convention:** the **official** 138/60 (`raw/train` vs `raw/test`,
  also listed in `raw/AdditionalData/dataset_record.yaml`). On the resolvable
  subset this becomes 109/53. `image_id` = `subject_id` = the MESSIDOR filename
  stem (`20051020_44923_0100_PP`); MESSIDOR publishes no patient id, so one
  image = one subject, as already assumed for the Messidor-2 external cohort.
- **2nd observer:** none for the vessel mask (one of seven retinologists per
  image; `raw/AdditionalData/biomarkers_annotation_infos.xls` says which, and
  how long it took). The *diagnosis* does have three independent graders.
- **Gotcha — the vessel masks are corrected network output.** The deposit ships
  the pre-annotations the retinologists started from; mean Dice between
  pre-annotation and final label is **0.783** (median 0.805, min 0.358, max
  0.894, none above 0.90), so substantial human editing did happen — but any
  systematic bias of the pre-annotating network can partially survive into what
  this project treats as ground truth. Say so in the paper.
- **Gotcha — thick strokes.** MAPLES-DR has the coarsest calibre of any set
  here: mean vessel width 1.54 % of FOV diameter, against HRF's 0.60 % and
  FIVES' 1.40 %; vessel foreground inside the FOV is 0.150 vs FIVES' 0.109.
  Calibre-derived biomarkers inherit that.
- **FOV:** none shipped. Generated. Checked: GT-in-FOV 1.0000 mean and worst,
  FOV coverage of frame 0.448–0.472 (the black surround is ~55 % of the frame).

### Image resolution for MAPLES-DR, and the Messidor-2 overlap

`datasets._maplesdr_image_index()` looks for images in two places, in order:

1. `exp/data/maplesdr/raw/images/<name>.tif` — the official MESSIDOR download,
   once someone completes the ADCIS form. **Always preferred.** Dropping the
   files there is the only step needed; no code change.
2. the project's existing Messidor-2 mirror, for the images that are in both.

Every record carries `image_source` (`"messidor"` or `"messidor2-mirror"`) and,
for the mirror route, `messidor2_id`, so provenance is never guesswork. Records
whose image is in neither place are dropped; pass `include_unresolved=True` to
see all 198.

The mirror stores only pixels — the original MESSIDOR filenames were lost — so
the mapping had to be recovered by matching each MAPLES-DR vessel mask to the
image it was drawn on (`python -m src.data.maples_overlap`, see that module's
docstring for the method and for the failure mode of the first attempt).
**Result: 162 of 198.** Accepted matches lead their candidate pool (527/614/603
images of the same native size) by 14.9–87.0 sd, rejected ones by 2.0–8.5 sd —
an empty band in between, so the 10 sd threshold is not tuned — and the 162
matches form a bijection although nothing enforces one. Spot-checked visually at
the two weakest accepted margins; the masks overlay the vessels exactly.

The same 162 rows are this project's **Messidor-2 / MAPLES-DR overlap**, written
to `exp/data/external/messidor2/overlap_maplesdr.csv`. They must be excluded
from the E3 Messidor-2 cohort (1,744 → 1,582) if MAPLES-DR is used as a
replication cohort, or the two analyses are not independent. The full 198-row
table, including the scores of the 36 rejected ones, is
`exp/data/maplesdr/messidor2_match.csv`.

---

## FOV generation (CHASE_DB1, FIVES, STARE, Fundus-AVSeg, MAPLES-DR)

These three datasets ship no field-of-view mask, so one is generated once per
image and cached at `exp/data/<name>/fov/<image_id>_fov.png` (8-bit, {0,255}).

Algorithm (`datasets._generate_fov`):

1. Take the **red channel** - brightest inside the fundus disc and near-zero on
   the black surround, giving the cleanest disc/background separation.
2. **Threshold at 10% of the channel maximum** (`FOV_THRESHOLD_FRACTION`).
3. **Morphological closing** with a radius-9 disc, to seal vessel shadows,
   lesions and the optic-disc rim. On images whose longest side exceeds
   `MAX_MORPH_SIDE` (1024 px) the closing runs on a decimated copy and is
   OR-ed back at full resolution - a strict superset of the un-closed mask, so
   it can only add pixels, never erode the retina.
4. **Fill holes** and keep only the **largest connected component**, discarding
   stray bright specks outside the disc.

Regenerate with `load_dataset(name, regenerate_fov=True)`, or by deleting
`exp/data/<name>/fov/`.

### Why 10% of max and not Otsu

The fundus surround is essentially black, so the problem is separating "not
black" from "black", not splitting two balanced populations. Otsu puts the
threshold between the dim retinal periphery and the bright disc centre, which
eats the periphery on darker images. Validated on every CHASE_DB1 and STARE
image plus a FIVES sample, scoring **the fraction of ground-truth vessel pixels
that fall inside the generated mask** (must be ~1.0, since annotated vessels are
by definition inside the FOV):

| dataset | method | FOV coverage sd | GT-in-FOV mean | GT-in-FOV worst image |
|---------|--------|----------------:|---------------:|----------------------:|
| CHASE_DB1 | Otsu | 0.051 | 0.967 | **0.864** |
| CHASE_DB1 | 10% of max | 0.011 | 0.998 | 0.994 |
| STARE | Otsu | 0.053 | 0.968 | **0.734** |
| STARE | 10% of max | 0.050 | 1.000 | 1.000 |
| FIVES | 10% of max | 0.014 | 0.998 | 0.986 |

Under Otsu, **13.6% of annotated vessel pixels on the worst CHASE_DB1 image and
26.6% on the worst STARE image fell outside the FOV** and would have been
silently excluded from every in-FOV metric. A 15% fraction regresses badly on
FIVES (worst image 0.788); using the 99th percentile instead of the max is
identical on CHASE_DB1/FIVES and slightly worse on STARE.

The decimated closing was checked against the full-resolution closing on the
same images: CHASE_DB1 and STARE results are identical, FIVES differs by
0.0002 in mean coverage and 0.00001 in GT-in-FOV, at 4.9x the speed
(4.4 s -> 0.9 s per 2048x2048 image).

## Split summary

| Dataset   | train | test | subject-disjoint? | source of convention |
|-----------|------:|-----:|-------------------|----------------------|
| DRIVE     |    20 |   20 | yes (1 image/subject) | official |
| CHASE_DB1 |    20 |    8 | yes (subjects 01-10 / 11-14) | community convention |
| HRF       |    15 |   30 | yes (subjects 01-05 / 06-15 per class) | community convention |
| FIVES     |   600 |  200 | yes (official dirs) | official |
| STARE     |    10 |   10 | yes (1 image/subject) | fixed here, recorded in code |
| Fundus-AVSeg | 80 |  20 | assumed (no patient id; 1 image = 1 subject) | official |
| MAPLES-DR | 109* |  53* | assumed (no patient id; 1 image = 1 subject) | official |

\* MAPLES-DR's official split is 138/60; these are the counts on the 162 images
whose pixels are currently resolvable (see §7).

## Sources that failed

- **DRIVE via <https://github.com/orobix/retina-unet>** — the README references
  `./DRIVE` but the directory is gitignored and not in the repository. No DRIVE
  data present after `git clone --depth 1`.
- **DRIVE via <https://github.com/hamdan92/retinal-vessel-segmentation>** — the
  paths `data/DRIVE/{training,test}/…` exist with correct filenames and correct
  counts (20 each), but every file is a **134-byte macOS iCloud placeholder stub**
  (content begins `/Users/hamdan/Library/Mobile Documents/…`), not image data.
  Unusable. This is easy to miss: the counts look right.
- **DRIVE via <https://github.com/zhengyuan-liu/Retinal-Vessel-Segmentation-DRIVE>** —
  repository returns 404 from the GitHub API; it no longer exists.
- **DRIVE via Kaggle** (`andrewmvd/drive-digital-retinal-images-for-vessel-extraction`)
  — not attempted: no `~/.kaggle/kaggle.json` on this machine.
- **FIVES via `https://figshare.com/ndownloader/files/34969398`** — returns
  HTTP 202 with `Content-Length: 0`, producing a 0-byte file. Use the
  `ndownloader.figshare.com` host instead (works, 302 to a signed S3 URL).
- **Replication-cohort candidates rejected on 2026-09-17** (full ranking and
  evidence in `research/08_replication_datasets.md`): **RETA** (81 IDRiD images,
  CC BY 4.0, anonymous — but the masks are released at 1024×1024);
  **Leuven-Haifa/UZLF** (240 images at 1444², excellent — but KU Leuven requires
  a signed Data Transfer Agreement); **UoA-DR** (200 at 2124×2056 — signed EULA
  emailed to two named staff); **ORVS** (49 at 1444² — the paper's own GitHub
  link `AbdullahSarhan/ICPRVessels` returns 404, confirmed twice, and no mirror
  exists); **ARIA** (143 at 768×576 — all three known hosts dead: domain parked,
  404, NXDOMAIN); **IOSTAR** (30 SLO at 1024² — registration form self-reported
  broken, author email only); **LES-AV** (n=22); **RVD** (handheld fundus video,
  semi-automated masks, CC BY-NC-ND); **VAMPIRE** (software, not a released mask
  set; `vampire.computing.dundee.ac.uk` NXDOMAIN); **HRF-Seg+** (new annotations
  on HRF's *own* images, so not independent); **FOVEA** (n=40 exactly, mixed
  with intraoperative video); plus PRIME-FP20 (n=15), REVIEW (n=16), AV-WIDE
  (n=30, SLO), RAVIR (infrared), RECOVERY-FA19 (n=8, FA), INSPIRE-AVR (no
  masks), HEI-MED (lesion masks), GRAPE (disc masks), CHUAC/DCA1 (X-ray).
- **MESSIDOR images for MAPLES-DR** — not obtained autonomously. The only
  distribution point is the ADCIS form (name, email, organisation, country, GDPR
  consent, then email verification); no mirror of the *original* MESSIDOR with
  its filenames exists on HuggingFace (only Messidor-2), Zenodo or GitHub. The
  Zenodo record "Messidor-A" (22246151) is lesion annotations only and itself
  points back at ADCIS behind a Google Forms agreement. 162 of the 198 images
  were recovered from this project's own Messidor-2 mirror instead (§7).
- **`git lfs pull` on the rajsahu2004 DRIVE mirror** — the repo's own
  `.gitattributes` is itself an LFS pointer, so git never registers the LFS
  filter and every checkout yields 133-byte pointer stubs. Worked around via the
  LFS batch API as described above.
