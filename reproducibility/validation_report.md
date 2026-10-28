# Validation report: local regeneration workflow

Run on 2026-10-28 on the lab workstation (RTX 4090, driver 595.99, Ubuntu, Python 3.10.12, torch 2.10.0+cu128,
TensorFlow 2.20.0, cuDNN 9.10.2, imageio 2.37.2 / ffmpeg 7.0.2, numpy 2.2.6, opencv 4.13.0; full list in
[`environment-validation-freeze.txt`](environment-validation-freeze.txt)). Code: commit `81bebde` plus the
uncommitted changes that add this folder. Outputs went to a new folder outside the repository; the
original data folders were only read.

**Scope.** These are smoke tests on the smallest real-data subset: one training and one validation clip
per task. They show that the pipeline reproduces the paper's inputs and settings. They do **not**
reproduce the published training results, and no model was trained.

## 1. Manifests (all clips, original records)

| Check | Result |
|---|---|
| DHF1K: clips selected / train / val | 541 / 431 / 110. Same as `split.json` and every recorded run. **Not** the 441/100 in the manuscript |
| DHF1K: train ∩ val | empty |
| DHF1K: exclusions accounted for | 700 − 541 = 159 rows in `dhf1k_saliency_exclusions.csv`: 158 with ≥ 750 frames (rule re-run on the data selects exactly 542 clips) + clip 058 (manual, upsampling failure) |
| DHF1K: `video_NNNNN` → source clip | all 541 clips: first, middle and last frame of `raw_images` equal to the decoded, resized source AVI (max abs diff 0); frame counts equal; 535/535 parsed upsampling-log lines agree |
| DHF1K: targets and timestamps | all 541 clips: `outputs.npz` and `raw_images.npz` have N = source frames at 180×320; 200 FPS timestamps = k/200 s on [0, N/30); optical-flow timestamps strictly increasing on [0, (N−1)/30), with N−1 pairs |
| DHF1K: source hashes | 700 distinct SHA-256 values |
| ECOCO: sequences | 948 train + 51 validation, disjoint IDs, 999 distinct content hashes |

## 2. Commands

```bash
OUT=<lab disk>/regen_smoke   # new folder, next to (not inside) the data/ folder
FILM=--film-model=<checkout>/models/film_net/Style/saved_model

python -m reproducibility.regenerate --task saliency --subset smoke $FILM --out $OUT \
       --source-root src/neural_network/data/saliency_dataset_processed --dry-run
python -m reproducibility.regenerate --task saliency --subset smoke $FILM --out $OUT \
       --source-root src/neural_network/data/saliency_dataset_processed \
       --repeat --reference-root src/neural_network/data/saliency_dataset_ready
python -m reproducibility.regenerate --task intensity --subset smoke $FILM --out $OUT \
       --source-root src/neural_network/data/ecoco_depthmaps_test \
       --repeat --reference-root src/neural_network/data
# changed setting must not reuse the cache
python -m reproducibility.regenerate --task intensity --subset smoke $FILM --out $OUT \
       --source-root src/neural_network/data/ecoco_depthmaps_test \
       --variants retina_ff200 --set retina_ff200.silicon_retina_config.p2=1.25
```

The original DHF1K download (`video/`, `annotation/`) is not kept on the workstation. The saliency runs
therefore used the restructured folders (`NNNN/input.AVI`, byte-identical to the DHF1K AVIs, and
`maps.mp4`). The DHF1K-layout code path (`reorder_clip`) was run separately, on PNGs extracted from
`maps.mp4` (§5).

## 3. Results

Naming: these runs were made before `FIREnetwork.json` was corrected. Their variants `dvs_of`
(num_bins 5, linear input) and `dvs_of_log` (num_bins 5, log input) are now called `dvs_of_linear`
and `dvs_of`.

**Selected inputs** (rule: the shortest clip of each split; saliency frames 0–30, ECOCO whole sequences):

| Task | Split | Source | Processed ID | Frames used / total |
|---|---|---|---|---|
| saliency | train | DHF1K 343 | `video_00261` | 31 / 186 |
| saliency | val | DHF1K 585 | `video_00454` | 31 / 210 |
| intensity | train | `sequence_0000000464` | `video_00462` | 37 / 37 |
| intensity | validation | `sequence_0000000999` | `video_00049` | 52 / 52 |

**Generated tensors.** All grids are finite. Event timestamps are sorted and lie inside the clip.

| Task / variant | Clip | Upsampled frames (200 FPS / OF) | Grids (T×B×H×W) | Events (+/−) | Non-zero voxels |
|---|---|---|---|---|---|
| saliency retina_ff200 | 00261 / 00454 | 207 / 414, 207 / 410 | 30×5×180×320 | 312,896 (+168,371/−144,525) / 155,062 | 6.1 % / 2.9 % |
| saliency dvs_of | 00261 / 00454 | | 30×5×180×320 | 3,310,175 / 1,021,360 | 16.7 % / 6.4 % |
| saliency retina_of | 00261 / 00454 | | 30×5×180×320 | 378,872 / 144,084 | 6.5 % / 2.0 % |
| intensity dvs_of (linear) | 00462 / 00049 | 275 / 72, 386 / 276 | 32×5×180×240 / 50×5×180×240 | 13,249 / 133,324 | 0.4 % / 2.3 % |
| intensity dvs_of_log | 00462 / 00049 | | 35×5×180×240 / 50×5×180×240 | 151,833 / 2,196,081 | 3.3 % / 13.8 % |
| intensity retina_ff200 | 00462 / 00049 | | 36×2×180×240 / 51×2×180×240 | 972,688 / 5,236,862 | 10.5 % / 36.0 % |

The linear DVS grids for `video_00462` number 32, not 36, because the ECOCO DVS loader drops intervals without events (`audit.md`).

**Fixed-rate vs optical-flow input.** The same retina configuration produces different event streams from the two upsampled videos (312,896 vs 378,872 events on `video_00261`), and they are kept in separate cache directories.

**Repeatability (same settings, run twice).**

| Stage | Result |
|---|---|
| Source frames, saliency targets | identical to the original files (bitwise) |
| FILM upsampling | **not bitwise reproducible.** Two runs in the same environment differ by mean 1.9 / 1.5 grey levels (PSNR 38.0 / 39.6 dB; max 73). The difference from the original upsampled videos is the same size (mean 1.90 / 1.50, PSNR 38.1 / 39.6 dB). Fixed-rate timestamps are identical. Optical-flow frames per pair agree on 30/30 and 29/30 pairs (saliency) and 36/36 and 51/51 (ECOCO) |
| Retina events (from one upsampled video) | bitwise identical (all cache files, both tasks) |
| DVS events | event lists bitwise identical; voxel grids identical except ≤ 0.001 % of voxels (max 0.016 = 1–2 fp16 steps), from the order of GPU atomic adds |
| Whole pipeline twice from the source (second run with the final code into a fresh folder) | event counts per clip within 0.05–1.5 % of the first run (e.g. retina `video_00261`: 312,896 vs 313,033; DVS log `video_00049`: 2,196,081 vs 2,165,830). All grids finite and of identical shape; same repeat behaviour as above |
| Events: regenerated video vs original video | total voxel mass within 1 % (retina) and 0.1–20 % (DVS on the near-static `video_00462`); non-zero fraction within 0.1 percentage point. Per-voxel correlation 0.78–0.94 (retina) and 0.19–0.73 (DVS). Because of the FILM variation, regenerated events are statistically, not bitwise, equivalent |
| Cache isolation | the p2 = 1.25 run wrote a new directory `retina_ff200-911d926ea570`, left the p2 = 1.0 cache untouched (0 files modified) and reused the upsampled videos |
| Corrupted input | appending one byte to one ECOCO frame makes `regenerate` stop before writing anything ("does not match the manifest") |

**Original DVS caches (ECOCO).** Events generated from the lab's original upsampled videos with **log**
input reproduce the caches used by the DVS baseline run: same event timestamps, and grids equal to
within fp16 rounding (≤ 0.016) for `video_00462` and `video_00049`. Linear input, as in the committed
config, gives 13,306 and 133,136 events instead of 151,665 and 2,175,915.

**Runtime and disk use (smoke).**

| Task | Upsampling | Events (3 variants) | Total wall time | Upsampled | Events per variant |
|---|---|---|---|---|---|
| saliency (2 × 31 frames) | 88 s | 3.3 s | 117 s incl. checks | 5.3 MiB | 90–101 MiB |
| intensity (37 + 52 frames) | 41 s | 2.5 s | 53 s incl. checks | 31.9 MiB | 36–95 MiB |

The `--repeat` and `--reference-root` checks add their own directories: 1.6 GiB for the saliency
reference events, which cover the two whole original clips, and 3.0 GiB in total.

**Full-size estimates** (extrapolated, not run here): one retina event cache takes about 1.4 MB per
source frame (grids 0.56 MB, 3-channel float targets 0.69 MB, frames 0.17 MB). That is about 350 GiB per
retina variant for the 263,989 DHF1K frames. On the lab machine the original folders take 73 GB
(DHF1K upsampled videos + frame arrays) and 18 GB (ECOCO train videos + frames + flow), plus 80 GB of
DVS and 193 GB of retina caches for ECOCO train. FILM upsampling (fixed-rate and optical-flow passes together)
took 0.5–1.4 s per source frame in the smoke runs, including TensorFlow start-up for each pass. At that
rate the full DHF1K upsampling would take several GPU-days (rough estimate).

## 4. Figure notebooks

`analysis_saliency.ipynb` and `analysis_intensity.ipynb` were run on copies in a separate environment
(pandas, matplotlib, seaborn, SciencePlots, nbconvert). Both run to completion from the published
`paper-figures/analysis/*.csv`. The `*_stats_*.csv` files they write are byte-identical to the
committed ones. The SciencePlots `science` style needs a LaTeX installation, which this machine lacks,
so the copies were run with the `no-latex` style added and the PNGs differ only in typesetting. No
preprocessing or manifest mapping that the figures depend on was changed.

## 5. Other checks

* **DHF1K layout** (`video/NNN.AVI`, `annotation/NNNN/maps/*.png`) through `reorder_clip`: sources found and hashes match. Raw frames are identical to the originals. Targets differ slightly, as expected, because the stand-in PNGs went through one more lossy encoding. This tests the code, not the original annotation files.
* **Torch/TF in one process.** With torch 2.10 and TF 2.20, FILM fails (`INTERNAL: an unsupported value or parameter`) if torch initialises CUDA before TF creates its cuDNN handle. `regenerate.py` runs a dummy TF convolution first. The original DHF1K upsampling never hit this, because its worker initialised TF first.

## 6. Not checked

* The full manifests (all 541 / 999 clips) were not regenerated, and no model was trained or evaluated.
* No original DHF1K or ECOCO download was available. The source checks rely on the byte-identical AVI copies and the extracted ECOCO frames.
* The pre-correction retina kernel (`396ebf1^`) used by the main saliency retina run and the ECOCO retina runs was not rebuilt.
* Cross-machine reproducibility (other GPUs, drivers, TF/cuDNN versions) was not tested. Expect FILM-level variation as in §3.
