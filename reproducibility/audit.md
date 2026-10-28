# Pipeline audit: what the paper runs actually used

Audit of the path from the source datasets to the training inputs, carried out on 2026-10-28. It used the
original lab data folders, run logs, W&B run directories, shell history and git history. Evidence is
cited by file, log or commit. **Confirmed** means checked against data or records. **Inferred**
means it is consistent with the records but was not checked directly. **Unknown** means the records do
not settle it.

## Saliency prediction (DHF1K)

### Confirmed

| Item | Finding | Evidence |
|---|---|---|
| Source | DHF1K, annotated clips 001–700 (official train 001–600, val 601–700). The test set 701–1000 has no public annotations and is not used. Videos are 640×360 at 30 fps | `help_scripts/reorder_saliency_map_ds.py`; `params.json` of all 700 clips |
| Source identity | `saliency_dataset_processed/NNNN/input.AVI` is a byte copy (`shutil.copy`) of DHF1K `video/NNN.AVI`. Its SHA-256 is recorded per clip in the manifests | reorder script; `dhf1k-restructured.zip` |
| Target | DHF1K continuous saliency maps (`annotation/NNNN/maps/*.png`) → `maps.mp4` (imageio, 30 fps) → ×0.5 INTER_AREA resize, giving `outputs.npz` (N×180×320×3 uint8). Fixation maps are not used as targets | `VideoDatasetFromPairs.py`; `VideoDatasetUpsampling.py`; all 541 `outputs.npz` have N = source frames |
| Selection | 541 of 700 clips. 158 clips with ≥ 750 frames were removed by `copy_short_videos.py --max-frames 750`. Clip **058** (739 frames) was then removed by hand after optical-flow upsampling kept failing on it | Shell history: runs with `--max-frames 850`, then `750`. Re-running the script selects exactly 542 clips = 541 + 058. `records/upsample_of_pass1_starts.txt`: 542 clips, 77 restarts at index 40 (= 058) |
| Exclusion cause (inferred) | Before commit `14be3ce` (2026-02-04), a frame pair with maximum flow < 0.5 px gave a negative bisection count, so the adaptive upsampler never terminated on it. This is consistent with the 058 failures, but 058 itself was not re-tested | `git show 14be3ce` |
| Mapping | `video_{i:05d}` = i-th folder of `sorted(saliency_dataset_truncated)`. Checked for all 541 clips: first, middle and last frames of `raw_images.npz` equal the decoded and resized source AVI (max abs diff 0), and frame counts are equal. The 200 FPS upsampling log names the clip of each index (535 of 541 lines parse; all agree) | `records/upsample_200fps_index_map.txt`; `tools/build_dhf1k_manifest.py` |
| Split | 431 train / 110 val, no overlap. It is a stratified re-split of the 541 clips (5 length quintiles, 20 % val, `np.random.default_rng(49210)`), so it is **not** the official DHF1K split: 92 official-train clips are in val and 56 official-val clips are in train | `saliency_dataset_ready/split.json` (identical in the `(copy)` folder); `data_help/partition_saliency.py`; every run log of the paper set records 431/110 (216/55 batches) |
| Frame rates | Source 30 fps. Fixed-rate video: t = k/200 s on [0, N/30); the last interval interpolates toward a repeat of the last frame. Optical-flow video: 2^⌈log₂ maxflow⌉ frames per pair (cap 16) on [0, (N−1)/30); the last source frame is not included. Timestamps are stored in seconds (`params.json`) and converted to int64 ns in the loaders. Voxel bins span [i/30, (i+1)/30) | all 541 `params.json` checked |
| Upsampling | FILM "Style" saved model (vid2e archive), `mixed_float16` Keras policy, TF 2.19.1 / CUDA 12.5 at the time | logs `upsample_200fps.txt`, `upsampling_adaptive.txt` |
| Main comparison inputs | **Retina reads the 200 FPS video; DVS reads the optical-flow video**, by design (paper §4.3): the retina model is stepped once per frame and needs a fixed time step, while the DVS simulation follows vid2e's optical-flow upsampling. The DVS loader hard-codes `video_upsampled_using_optical_flow.mp4`, and every retina run records `use_upsampled_from_OF: false` | `EventDatasetFromUpsampledSaliency.py`; W&B configs |
| DVS settings | ESIM, log input (`log(I/255 + 1e-4)`), C+ = C− = 0.2 (sweep: `thr_sweep/`), refractory 0, 5 bins, bilinear in time, polarity ±1, no normalisation | config, loader |
| Retina settings | `silicon_retina_config` as in each config. Linear input `I/255 + 1e-4`. One `reset()` per clip, then one step per 200 FPS frame; no warm-up other than the network's `burn_in_steps` (10). Events are binned into 5 voxel bins with representation `EVENTS` (class 1) | configs, loader |
| Main retina run | `FIRE_saliency_seq=50_gk5_20251201_014614` was launched with `FIREnetwork_retina_saliency_p2_eq_1.0_seq=30_gk7.json` (sequence 30, gain-control kernel Gaussian 7), **not** with the `…seq=50_gk5.json` named in the earlier `scripts/paper/saliency_main_comparison.sh` | `retina_old/wandb/run-20251201_014615-h1l2i6zo/files/wandb-metadata.json` args |
| Target conversion | Retina loader: mean of RGB, as float. DVS loader: luminance-weighted grey, truncated to uint8. These differ by ≤ 1/255 | loaders, `SaliencyDatasetWrapper.py` |
| Randomness | Data generation uses no random numbers. Training is unseeded (random sequence start, flips, brightness, grid noise; `cudnn.benchmark=True`) | code search |

### Discrepancies and risks

0. **Main retina curve used an earlier retina kernel.** The run plotted in the main comparison (2025-12-01) predates commit `396ebf1` (2025-12-14, "kernel correction"). Before that commit, `silicon_retina_cuda_kernel.cu` clamped convolution and blur outputs to [0, 1] instead of [−1, 1], and did not clamp the bipolar and LIF states. Events generated with the released code therefore follow the corrected model, not that run. To regenerate that run's events, build `esim_torch` from `396ebf1^`; this was not done here. The feedback, HPF, kernel and sparsity ablations (from 2025-12-17) postdate the correction. The kernel sources have not changed since.
1. **441/100 vs 431/110.** No record supports 441 train / 100 val. Every recorded paper run used 431/110. The only "100" in the records is an early DVS run (`FIREnetwork_20251106_124018`, `logs/saliency_firenetwork_events.txt`): it used `val_limit: 100` over the same 541-clip directory, so its training and validation sets overlapped. 541 − 100 = 441 is the likely origin of the manuscript numbers (inferred). **The manuscript should say 431/110**, unless the author has records of another split.
2. **Two generations of optical-flow videos.**
   - **OF-v1:** Oct 2025, kept in `saliency_dataset_upsampled_backup/` and `saliency_dataset_ready (copy)/`.
   - **OF-v2:** regenerated on 2026-02-05 after the bisection fix, and copied into `saliency_dataset_ready/` on 2026-02-06 by `help_scripts/refresh_dataset.py`.
   - **Differences:** 43 of 541 clips have a different frame schedule; all files differ at byte level.
   - **Which runs:** DVS runs before 2026-02-06 read OF-v1. Later runs (upsampling comparison Feb 6–12, DVS threshold sweep May 2026) read OF-v2, *if* their DVS caches were rebuilt. The DVS cache name (`cache_B=5_ctn=…_ctp=…_v2`) does not encode the video, so stale v1 caches could have been reused. The shell history shows `rm saliency_dataset_ready/*/*/cache*` but not when it ran. **Unknown.**
3. **"DVS (Fixed 200 FPS)" runs used uncommitted code.**
   - `git stash@{0}` (on `6999aaa`, local checkout) changes the DVS loader to read `video_upsampled_fps_eq_200.mp4`, but it still uses `timestamps_OF` and `new_images_per_pair_OF`. That is 3000 frames against 3356 timestamps for `video_00000`. ESIM does not check lengths, so the k-th 200 FPS frame is stamped with the k-th optical-flow timestamp.
   - Over all 541 clips (from the stored timestamps): the median timing error per clip is 2.2 s (90th percentile 4.9 s); the error at the last frame is typically 4.6 s (≈ 140 target frames); and typically 13 % (up to 48 %) of each clip gets no events. In 207 clips the 200 FPS video has more frames than there are timestamps, so ESIM reads past the end of the timestamp array.
   - The "DVS (Fixed 200 FPS)" curve in Fig. 12 (runs `FIREnetwork_20260211_131459`, `…_200608`, `FIREnetwork_20260212_025427`) is therefore invalid. Its higher loss is explained by the timing error and is not evidence about fixed-rate upsampling.
   - The committed code cannot produce this variant, and `reproducibility/regenerate.py` does not offer it.
   - The optical-flow group of that comparison, described in the paper as "7 different seeds", is 7 runs with thresholds 0.2 ×3, 0.225, 0.25, 0.275 and 0.3. No training run sets a seed.
4. **Feedback ablation labels.**
   - The kernel enum is 0 = ABS, 1 = QUADRATIC (`retina_params.h`). The run labelled "quad_fb" recorded mode 0 and "abs_fb" recorded mode 1.
   - Both record the same grid occupancy to every digit (0.036492055…). It is unknown whether `q_feedback_mode` reached the compiled kernel in that build.
   - The notebook hard-codes occupancies 0.068282 (the Nov-30 `seq=30_quad_fb` run) and 0.03694 (record: 0.03649).
5. **Missing run records.** There is no local W&B record for `FIREnetwork_DVS_seq=50` (main DVS curve), the Jan 6–10 `best_b*` sparsity runs, or the early-Jan kernel-sweep reruns. The `bandwidth_*.csv` export shows 12 steps/epoch for the sparsity runs (about 108 batches, against 216 for the full split), so their batch size or data differ from the committed configs.
6. **Hard-coded values.** The feedback and HPF figures use final losses and occupancies typed into the notebook cells. How they were aggregated is not documented. t2/t3/t4 have a NaN final validation loss in the records.
7. **Configs absent from the repository.** DVS sweep configs `thr_500 … thr_1400` exist only in the local checkout (commit `eda2a37` and untracked files).

## Intensity reconstruction (ECOCO)

### Confirmed

| Item | Finding | Evidence |
|---|---|---|
| Source | `ecoco_depthmaps_test.zip` (48,011,579,794 bytes, served by UZH-RPG, Last-Modified 2021-07-08). It is the depth-map variant of Event-MS-COCO from the E2VID work. Contents: 948 `train` (IDs 0–949 without 107 and 382) + 51 `validation` (950–1000) sequences, each 37–59 grey PNG frames at 240×180 with `frames/timestamps.txt` in seconds (≈ 27.6 fps) | `download.sh`; `frames/params.json` |
| Used content | Only the PNG frames. The archive's own `VoxelGrid-betweenframes-5` events and `flow/` files are read only by older E2VID scripts, not by the paper's entry points | `ECOCO_dataset.py` vs `reconstruct_intensity_using_FIREnetwork_{events,silicon_retina_v2}.py` |
| Mapping | `upsampled_{train,validation}/video_{i:05d}` = i-th folder of `sorted(<split>)`. Pixel-identical for the sequences checked | `VideoDataset.py`; manifest |
| Split | The archive's own train/validation folders. `train_limit`/`val_limit` null | scripts, run log |
| Upsampling | `help_scripts/upsampling_dataset_prototyping.py`: global 27 fps (the real ≈ 27.6 fps timestamps are ignored), FILM, 200 FPS and optical-flow videos, no mixed precision | script, `metadata.json` |
| DVS baseline run | `FIREnetwork_20250901_173249` (W&B `yfoy3nt3`, `logs/firenetwork_events.log`). It logged `num_bins: 5` and `log_input: False`. `configs/FIREnetwork.json` had been changed to `num_bins: 2` afterwards (`51297d0`). It now says `num_bins: 5, log_input: true, use_wandb: true`, the events the run actually trained on; it otherwise matches the logged configuration | log, git |
| **DVS caches were log-input** (the physically standard DVS model) | The run reused caches built before it started, and those caches are log-input events. `reproducibility/regenerate.py` was run on the lab's own upsampled video for two sequences: log input reproduces the lab cache exactly (same event timestamps; voxel grids equal to fp16 precision), while linear input gives about 11× fewer events. The cache name does not encode the log flag | `validation_report.md` §3 |
| Training-time exclusion | The DVS run kept 947 of 948 training sequences. Sequence 745 (`video_00743`) fails the occupancy check because empty intervals are skipped, leaving only 20 grids | log; `EventDatasetFromUpsampled.py` |

### Discrepancies, risks, unknowns

1. **Retina curves of the intensity figure are not identified.** The curves are "burnin=20, seq=40", "burn in=10, seq=30" and "seq=20". No local W&B run matches their losses. They probably come from the Aug 2025 history of `FIREnetwork_retina.json` (inferred), with unknown retina parameters. `FIREnetwork_event_bins_eq2.json`, named in `scripts/paper/intensity_reconstruction.sh`, ran as `FIREnetwork_20250902_065333` and logged no completed epoch. Whichever runs they are, they predate the retina-kernel correction `396ebf1` (2025-12-14).
2. **On-disk retina caches do not match the paper's runs.** In `upsampled_{train,validation}`, clips 0–199 / 0–49 were overwritten by Bayesian-optimisation trials (2025-09-11), and the rest hold the `bins_eq2` config.
3. **Empty DVS intervals are skipped.** The linear-input DVS loader drops intervals without events, so later grids shift relative to the frames in that clip (seen in the smoke test: 32 grids for 37 frames).
4. **Training is unseeded.** Validation also uses random starts and augmentation, so validation losses vary between runs.

## Shared

* **Cache identity.** Each loader decides cache validity from file names, and the retina loaders also use a copy of the config in the clip's `params.json`, which they rewrite. Neither covers the input video version, `log_input`, the frame rate, the source identity or the code version. `reproducibility/regenerate.py` therefore keys every output directory on a hash of all of these.
* **Numerics.** FILM upsampling on the GPU is not deterministic. Two regenerations in the same environment differ by a mean of 1.5–1.9 grey levels (PSNR 38–40 dB), which is the same size as the difference from the original files. Retina event caches are bit-identical across repeats. DVS voxel grids differ in ≈ 0.001 % of voxels (≤ 0.008, from atomic adds on the GPU); the DVS event lists are identical.
* **Environment drift.**
  - Original upsampling: TF 2.19.1.
  - Validation: TF 2.20.0, torch 2.10.0+cu128, cuDNN 9.10.2.
  - `scripts/setup/requirements-full-freeze.txt` is not the environment of the saliency upsampling.
  - With torch 2.10 and TF 2.20 in one process, TF must initialise cuDNN first or FILM fails (handled in `regenerate.py`).
