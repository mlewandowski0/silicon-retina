# Corrections for the manuscript

Checked against the preprint dated 2026-10-28. The evidence for each item is in [`audit.md`](audit.md)
("Manuscript cross-check").

## 1. Figure 12 and Section 4.6: rerun the fixed-rate DVS curve

The three runs behind the gray "DVS (Fixed 200 FPS)" curve gave each 200 FPS frame the timestamp of
a different frame (from the optical-flow video). Events ended up seconds away from the saliency maps
they were trained against, so that curve's higher loss is not evidence about fixed-rate upsampling.
The blue and orange curves are not affected.

* Rerun the curve with `scripts/runpod/dvs_fixed_200fps.sh`. Instructions for a cloud GPU are in
  [`scripts/runpod/AGENT_BRIEF.md`](../scripts/runpod/AGENT_BRIEF.md). Then insert the new runs with
  `scripts/runpod/export_fig12_runs.py` and re-run `paper-figures/analysis_saliency.ipynb`.
* Rewrite the conclusion of Section 4.6 according to the new result.
* Wording: no training run sets a seed. "DVS (Optical Flow) over 7 different seeds" is 7 runs with
  contrast thresholds 0.2 (three runs), 0.225, 0.25, 0.275 and 0.3. The fixed-rate runs are 3
  repeats of the same configuration.

## 2. Short text fixes

| Where | Now | Should be |
|---|---|---|
| §4.4.2 | 541 videos, 441 training / 100 validation | 431 training / 110 validation (every recorded run used this split) |
| §4.4.2 | Long videos (900–5000 frames after upsampling) removed; videos with unreliable optical flow removed | Clips with 750 or more source frames removed (542 left), and one clip (DHF1K 058) removed because optical-flow upsampling failed on it |
| §4.3.2 | FireNet with ≈ 38k parameters | ≈ 131k parameters (the network used, 32 hidden channels) |
| §4.4.3 | ≈ 100k parameters | ≈ 131k parameters |
| §4.3.2 and Data availability | Event-MS-COCO cited as [2] (FireNet paper) | Cite Rebecq et al. [48] (E2VID), where the dataset comes from |
| §1, list of contributions | Code, models, ... "are currently being prepared for release" | The code is available at the repository; trained models are not released (or release them) |
| §1 and §4.4.4 | 0.695 vs 0.799, "13 % lower loss, 47 % sparser", occupancy ≈ 5.7 % vs ≈ 10.8 % | 0.695 and 5.7 % come from two different retina runs. Use one run: either 0.695 with 6.8 % (13 % lower, 37 % sparser), or 0.704 with 5.7 % (12 % lower, 47 % sparser) |
| Data availability | Current text | Proposed replacement: [`data_availability.tex`](data_availability.tex) |

## 3. Worth checking

* **Feedback ablation (§4.5.1, Fig. 8).** The run labelled "quadratic" stopped after 2 epochs. The
  plotted quadratic values (0.714, 6.83 %) were typed into the notebook from an older run with a
  different configuration. The run labelled "absolute" recorded the quadratic setting
  (`q_feedback_mode` 1 = quadratic). The labels may be swapped.
* **High-pass ablation (§4.5.2, Fig. 9).** The plotted values are typed into the notebook, and three
  of the eight runs end with a NaN validation loss. State how the values were taken (e.g. best epoch).
* **Main saliency curve (Fig. 7).** That retina run was trained on 2025-12-01. On 2025-12-14 the
  retina CUDA code was changed: intermediate filter outputs used to be clipped to [0, 1], cutting off
  negative values, and are now clipped to [−1, 1]. So that curve came from slightly different code
  than the released one. The later retina runs (feedback, kernel size), made with the released code,
  reach the same or lower validation loss (0.690–0.724 vs 0.704), so the conclusion is unchanged.
  Optionally mention this, or rerun the curve with the released code.
