# `src/neural_network`: event generation, datasets and training

Everything here is run **from the repository root** as a module, e.g.

```bash
python -m src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina \
       -c src/neural_network/configs/saliency/retina/christmas_run/best.json
```

Imports are absolute (`from src.neural_network...`), and dataset, checkpoint and config paths are relative to the root.

## Layout

| Folder | Contents |
|---|---|
| [`experiment/`](experiment/) | Entry points (see table below). |
| [`configs/`](configs/) | One JSON per run (see [Configs](#configs)). |
| [`dataloaders/`](dataloaders/) | Video datasets, FILM upsampling (`VideoDatasetUpsampling.py`, `interpolator.py`), DVS event datasets (`EventDatasetFromUpsampled*.py`, ESIM), silicon-retina event datasets (`SiliconRetinaEventDatasetFromUpsampled*.py`), voxel-grid encodings (`Encodings.py`), sequence windowing and augmentation (`*DatasetWrapper.py`). |
| [`networks/`](networks/) | `FIRE_network.py` (FireNet-style ConvGRU network, used for all results), `SaliencyLoss.py` (KL + CC), `PerceptualLoss.py` (LPIPS + L1), `TemporalConsistencyLoss.py`. Also the E2VID baselines (`E2VIDModel.py`, `E2VID_convLSTM.py`), which were used only in early exploration. |
| [`utils/`](utils/) | Config handling, occupancy statistics, event rendering. |
| [`help_scripts/`](help_scripts/), [`data_help/`](data_help/), [`dataset_helpers/`](dataset_helpers/) | Dataset preparation and maintenance. |
| [`reusing_videos/`](reusing_videos/) | Adapted copy of [vid2e](https://github.com/uzh-rpg/rpg_vid2e). **`esim_torch/`** is the CUDA extension providing both `ESIM` (DVS) and the silicon-retina kernels (`silicon_retina_cuda_kernel.cu`, `silicon_retina_torch.py`). Install it with `pip install -e src/neural_network/reusing_videos/esim_torch --no-build-isolation`. |

## Entry points (`experiment/`)

| Module | Task | Event source |
|---|---|---|
| `saliency_using_FIREnetwork_silicon_retina` | Video saliency (DHF1K) | Silicon retina |
| `saliency_using_FIREnetwork_events_v2` | Video saliency (DHF1K) | DVS (ESIM) |
| `reconstruct_intensity_using_FIREnetwork_silicon_retina_v2` | Intensity reconstruction (ECOCO) | Silicon retina |
| `reconstruct_intensity_using_FIREnetwork_events` | Intensity reconstruction (ECOCO) | DVS (ESIM) |
| `compute_occupancy_retina` | Event-occupancy statistics + rendered videos, no training (`-c cfg -o outdir`) | Silicon retina |
| `sweep_retina_params` | Runs `compute_occupancy_retina` over one-parameter sweeps | Silicon retina |
| `saliency_BO_sweep` | Optuna search of retina parameters, saliency objective | Silicon retina |
| `retina_parameter_search_using_BO` | Optuna search of retina parameters, reconstruction objective (`--help`) | Silicon retina |

The remaining `reconstruct_intensity_using_E2VID*` and `reconstruct_intensity_using_FIREnetwork{,_silicon_retina}` scripts are earlier versions, kept for completeness.

Every training script logs to Weights & Biases. Set `WANDB_MODE=offline` to keep logs local. Checkpoints go to `checkpoint_dir`, which is set in the config.

## Configs

The fields in each config are:

* the network: `hid1`, `ks1`, `add_bn`, `out_mode`;
* training: `num_epochs`, `sequence_length`, `burn_in_steps`, `batch_size`, `learning_rate`, `augment`, `train_limit`, `val_limit`;
* the event representation: `event_representation_class`, `event_representation_params`, `num_bins`;
* the retina, under `silicon_retina_config`, whose fields map 1:1 to `SiliconRetinaConfig` in `esim_torch/silicon_retina_torch.py`;
* the DVS: `contrast_threshold_pos` / `contrast_threshold_neg` inside `event_representation_params`.

`wandb_prefix` / `wandb_name` is the run name used in the W&B exports in `paper-figures/analysis/`.

| Folder | Experiment |
|---|---|
| `saliency/retina/christmas_run/` | Tuned retina (`best.json`) and its ablations: feedback (`best_no_feedback`, `best_quad_abs_fb`, `best_quad_fb`), HPF modes (`best_t1..t4[a]`), p₂ sparsity sweep (`best_b1..b5`), spatial kernels (`best_k_eq_*`, `best_no_spatial`) |
| `saliency/retina/kernel_sweep/`, `kernel_sweep_ks1/` | Spatial kernel size/type of the gain-control (`kernel2`) and OPL (`kernel1`) stages |
| `saliency/retina/fb_run/` | Quadratic-feedback reruns |
| `saliency/retina/*.json` | Earlier p₂ sweeps and sequence-length / kernel variants (incl. `…p2_eq_1.0_seq=30_gk7.json`, the config of the main-comparison retina run) |
| `saliency/events/` | DVS baseline (`FIREnetwork_event_saliency_seq=50.json`) and contrast-threshold variants; `*_create.json` force event re-generation |
| `saliency/events/thr_sweep/` | DVS contrast-threshold sweep (sparsity figure) |
| `saliency_sweeps/`, top-level `FIREnetwork_retina_saliency*.json` | Early saliency runs |
| `saliency_retina_BO_base.json` | Base config for `saliency_BO_sweep` |
| `occupancy_plots/`, `occupancy_sweeps/` | Occupancy analysis inputs, generated configs and results (`occupancy_sweeps/results/sweep_summary.csv`) |
| top-level `FIREnetwork*.json`, `retina_FIREnetwork_*.json`, `E2VID*.json` | Intensity-reconstruction runs on ECOCO |

[`scripts/paper/`](../../scripts/paper/) maps each paper figure to the configs it was trained with.

## Dataset preparation

All commands are run from the repository root. Upsampling needs TensorFlow and the FILM weights (`bash scripts/setup/download_model.sh`).

### DHF1K (video saliency)

The exact clips and split of the paper are in [`reproducibility/manifests/`](../../reproducibility/manifests/), and
`python -m reproducibility.regenerate` runs steps 1 and 3 (and the event generation) for any subset of them,
writing to a directory of your choice. The commands below are the original whole-dataset procedure.

```bash
D=src/neural_network/data
# 1. reorganise the raw DHF1K download (video/*.AVI, annotation/NNNN/maps/*.png) into NNNN/{input.AVI, maps.mp4}
python -m src.neural_network.help_scripts.reorder_saliency_map_ds --input_path /path/to/DHF1K --output_path $D/saliency_dataset_processed \
       --video_subdir video --annotation_subdir annotation
# 2. keep clips shorter than 750 frames (542 clips). The paper then removed clip 0058 by hand, because
#    optical-flow upsampling failed on it with the code of the time -> 541 clips
python -m src.neural_network.help_scripts.copy_short_videos --root $D/saliency_dataset_processed --dest $D/saliency_dataset_truncated --max-frames 750
rm -r $D/saliency_dataset_truncated/0058
# 3. FILM-upsample every clip -> saliency_dataset_upsampled/video_NNNNN (i-th clip in sorted order; resumable)
python -m src.neural_network.help_scripts.upsample_until_success                    # optical-flow-adaptive video (DVS loaders)
UPSAMPLE_FPS=200 python -m src.neural_network.help_scripts.upsample_until_success   # fixed 200 FPS video (retina loaders)
# 4. 80/20 train/val split per clip-length bin (seed 49210) -> saliency_dataset_ready/{train,val} + split.json (431/110)
python -m src.neural_network.data_help.partition_saliency
```

Step 3 writes two videos per clip. `video_upsampled_using_optical_flow.mp4` is produced by adaptive bisection driven by FILM optical flow and is read by the DVS loaders. `video_upsampled_fps_eq_200.mp4` is the fixed-rate video, read by the silicon-retina loaders and by `partition_saliency.py`. The split depends only on the clip lengths and the seed, but `reproducibility/manifests/dhf1k_saliency_split.csv` is the authoritative record.

TensorFlow runs FILM on the GPU if it finds a CUDA 12 runtime, and falls back to the CPU otherwise. The CPU is fine for testing but slow for the full dataset. FILM on the GPU is not bit-exact between runs (see `reproducibility/README.md`).

### ECOCO (intensity reconstruction)

```bash
mkdir -p src/neural_network/data && (cd src/neural_network/data && bash ../download.sh && unzip ecoco_depthmaps_test.zip)
python -m src.neural_network.help_scripts.upsampling_dataset_prototyping   # -> data/upsampled_{train,validation}/
```

### Events

DVS and retina events are not stored separately. Each dataset class simulates them from the upsampled video the first time a clip is accessed, and caches voxel grids and event lists as `cache_*.npy` inside the clip folder. Changing the retina / DVS parameters in a config, or setting `force_recomputing: true`, regenerates the cache.
