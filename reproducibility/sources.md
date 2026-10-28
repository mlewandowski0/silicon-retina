# Source data and third-party models

None of the items below are redistributed in this repository. Download them from their original
providers and accept those providers' terms. Links were checked on 2026-10-28 (HTTP 200).

## DHF1K: video saliency prediction

| | |
|---|---|
| Dataset | DHF1K (Dynamic Human Fixation 1K) |
| Reference | W. Wang, J. Shen, F. Guo, M.-M. Cheng, A. Borji, "Revisiting Video Saliency: A Large-scale Benchmark and a New Model", CVPR 2018; extended in IEEE TPAMI 2021 |
| Access | <https://github.com/wenguanwang/DHF1K>. The README links to a Google Drive folder (`drive.google.com/open?id=1sW0tf9RQMO4RR7SyKhU8Kmbm4jwkFGpQ`) and a Baidu Pan mirror. There is no registration step. The README states no licence for the data; check the terms with the authors before redistributing anything derived from it |
| Parts needed | `video/` (`001.AVI`–`700.AVI`) and `annotation/NNNN/maps/*.png` (continuous saliency maps) for clips 001–700. `annotation/NNNN/fixation/` is converted by the preparation script but is not used as a target. The test clips 701–1000 have no public annotations and are not used |
| Layout expected by `reproducibility/regenerate.py` | `<root>/video/NNN.AVI` and `<root>/annotation/NNNN/maps/`; pass `--video-dir` / `--annotation-dir` if the download unpacked elsewhere (the original download unpacked to `video-002/video` and `annotation-003/annotation`) |
| Identity check | `dhf1k_saliency_split.csv` and `dhf1k_saliency_exclusions.csv` record the **SHA-256 of each `.AVI` file's bytes** for all 700 clips. These are the files used in the paper, which were copied byte for byte. The annotation PNGs are not hashed, because the original PNGs were not kept: only the MP4 re-encodings made from them survive |
| Retrieved | Before 2025-10-15 (date of the restructured archive). The exact web-disk upload version was not recorded |

## Event-MS-COCO ("ECOCO"): intensity reconstruction

| | |
|---|---|
| Variant used | **`ecoco_depthmaps_test.zip`**: the depth-map variant of the Event-MS-COCO sequences, with 948 `train` + 51 `validation` sequences of grey 240×180 frames (37–59 frames, ≈ 2 s each), frame timestamps, optical flow and pre-computed voxel grids. The paper pipeline uses **only `frames/*.png` and `frames/timestamps.txt`**; it simulates its own DVS and retina events |
| Reference | H. Rebecq, R. Ranftl, V. Koltun, D. Scaramuzza, "High Speed and High Dynamic Range Video with an Event Camera", IEEE TPAMI 2019 (E2VID), which introduced the synthetic MS-COCO event sequences. The images come from MS-COCO (T.-Y. Lin et al., ECCV 2014) |
| Access | Direct download (no registration): <https://download.ifi.uzh.ch/rpg/web/data/E2VID/datasets/ecoco_depthmaps_test.zip>, also used by `src/neural_network/download.sh`. On 2026-10-28 the server reported: size **48,011,579,794 bytes**, Last-Modified 2021-07-08, ETag `"b2db69192-5c69b48acb339"`. No project page describing this archive was found; the citation and terms above should be confirmed with UZH-RPG (see the author checklist) |
| Identity check | The archive itself was not kept, so it has no checksum. `ecoco_intensity_split.csv` records, per sequence, a canonical hash of the frame files: SHA-256 over the lines `"<name> <sha256 of bytes>\n"` for `frames/*.png` and `frames/timestamps.txt`, sorted by name (`reproducibility.regenerate.ecoco_frames_sha256`) |

## FILM frame-interpolation weights

| | |
|---|---|
| Model | FILM (F. Reda et al., "FILM: Frame Interpolation for Large Motion", ECCV 2022), "Style" saved model. Code: <https://github.com/google-research/frame-interpolation> (Apache-2.0) |
| Access | Redistributed by the vid2e authors: <https://rpg.ifi.uzh.ch/data/VID2E/pretrained_models.zip> (918,954,135 bytes, Last-Modified 2022-04-01, ETag `"36c62097-5db96463b5a5e"`). Install it with `bash scripts/setup/download_model.sh`, which puts it at `models/film_net/Style/saved_model` |
| SHA-256 of the weights used | `saved_model.pb` `4df311e80e9a7282b362a7e93bef22a1ce4f84e7cdeda01f246894545eaaf985`<br>`keras_metadata.pb` `0291f451e35e62a042fa49a1341af1dc8a94632188a24a16b71a9516e9fc6853`<br>`variables/variables.data-00000-of-00001` `8c47323923bc4826b730dd882c8c7700761aa3ac03b2c8180d3ffc82d18111f9`<br>`variables/variables.index` `d19bb117eb9abe6121b5711649bb7d5d1c4fe1912b9deabbdafa2be3f5a273e5` |

## Code reused from other projects

| Component | Origin | Licence |
|---|---|---|
| `src/neural_network/reusing_videos/` (incl. ESIM in `esim_torch`) | vid2e, <https://github.com/uzh-rpg/rpg_vid2e> (D. Gehrig et al., CVPR 2020) | GPL-3.0 |
| `dataloaders/interpolator.py` | FILM | Apache-2.0 |

## Software versions

| Stage | Original runs | Validation of this workflow (2026-10-28) |
|---|---|---|
| FILM upsampling | TensorFlow 2.19.1 (CUDA 12.5 build, cuDNN 9), RTX 4090; `mixed_float16` policy for DHF1K (from the upsampling logs) | TensorFlow 2.20.0, cuDNN 9.10.2, RTX 4090, driver 595.99 |
| Event simulation | `esim_torch` built from this repository at the commit of each run. The retina kernel sources last changed on 2025-12-14 (`396ebf1`, clamping correction), so runs before that date, including the main saliency comparison's retina run, used the earlier kernel (see `audit.md`) | torch 2.10.0+cu128, `esim_torch` built with nvcc 12.9 |
| Video I/O | imageio / imageio-ffmpeg (versions not logged) | imageio 2.37.2, imageio-ffmpeg 0.6.0 (ffmpeg 7.0.2), opencv-python 4.13.0, numpy 2.2.6, Pillow 12.0.0 |

The full environment used for the validation is in [`environment-validation-freeze.txt`](environment-validation-freeze.txt). `scripts/setup/requirements-full-freeze.txt` is a freeze of a later experiment environment, not the one used for the original DHF1K upsampling (that one had TF 2.19.1).
