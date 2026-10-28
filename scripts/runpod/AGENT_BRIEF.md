# Brief: rerun the "DVS (Fixed 200 FPS)" experiment (Fig. 12) on a RunPod GPU

You are an agent on the author's laptop. You reach the GPU pod over SSH. Your job: run three DVS
saliency trainings on the pod, monitor them to completion (about 21 h in total), bring back the small
result files, and update the Figure 12 data. Report progress at each milestone below.

## Background (one paragraph)

The paper "Programmable Silicon Retina on Pixel Processor Array" compares DVS and silicon-retina events
for video saliency prediction (DHF1K). In Figure 12, the curve "DVS (Fixed 200 FPS)" came from runs
whose events were mistimed: 200 FPS frames were paired with timestamps from a different video. The
repository branch `reproducibility` contains a corrected data loader and the config
`src/neural_network/configs/saliency/events/FIREnetwork_event_saliency_fixed_200fps.json`
(`"dvs_input_video": "fixed_200fps"`). It also contains `scripts/runpod/dvs_fixed_200fps.sh`, which
sets up the pod, downloads the data and runs the trainings. The data (31.3 GB) is the **private**
Hugging Face dataset `mlewand/silicon-retina-dhf1k-200fps`.

## What you need from the author before starting

1. The SSH command for the pod, e.g. `ssh root@<ip> -p <port> -i ~/.ssh/<key>`.
2. A Hugging Face token with read access to `mlewand/silicon-retina-dhf1k-200fps`.
3. A Weights & Biases API key and the W&B entity (user or team name). The runs log to project `e2vid-saliency2`. If there is no key, use offline mode, described below.

**Pod requirements.** If the pod doesn't meet them, stop and tell the author.

* 1× RTX 4090.
* A RunPod **PyTorch "devel"** template (CUDA 12.x with `nvcc`), because the `esim_torch` CUDA extension is compiled on the pod.
* A volume at `/workspace` of **at least 300 GB**: 31 GB of data plus about 212 GB of event caches built during the first run.
* At least 8 vCPUs and 32 GB RAM. The training uses 10 data-loader workers.

## Rules

* Do not change code or configs in the repository. Settings may only be changed in the per-run copies under `/workspace/configs`, and only for failures listed below.
* Never print the tokens, and never write them into the repository or a log. Keep them in `/workspace/.env` (mode 600) on the pod.
* Never make the Hugging Face dataset public, and never upload the data anywhere else.
* Do not terminate the pod or delete its volume. When done, tell the author it can be stopped.
* Do not push to GitHub. The author reviews the final changes.

## Steps

`POD` below stands for the SSH command from the author.

### 1. Check the pod

```bash
POD 'nvidia-smi; nvcc --version | tail -1; python --version; nproc; free -g | head -2; df -h /workspace'
```

Check each value:

* The GPU is an RTX 4090.
* `nvcc` exists.
* `/workspace` has at least 300 GB free.
* Note the "CUDA Version" in the `nvidia-smi` header, which is the highest CUDA version the driver supports.

Choose the PyTorch wheel from that value:

| Driver CUDA version | `TORCH_INDEX` |
|---|---|
| ≥ 12.8 | `https://download.pytorch.org/whl/cu128` (default) |
| 12.6–12.7 | `https://download.pytorch.org/whl/cu126` |

The major version of `nvcc` must also be 12.

### 2. Secrets and code

```bash
POD 'umask 077 && cat > /workspace/.env' <<'EOF'
export HF_TOKEN=<token from the author>
export WANDB_API_KEY=<key from the author>
export TORCH_INDEX=https://download.pytorch.org/whl/cu128
EOF
POD 'cd /workspace && git clone -b reproducibility https://github.com/mlewandowski0/sillicon-retina.git && cd sillicon-retina && git log --oneline -1'
```

Without a W&B key, put `export WANDB_MODE=offline` in place of the `WANDB_API_KEY` line.

### 3. Start the runs in tmux, so they survive SSH disconnects

```bash
POD 'command -v tmux || (apt-get update && apt-get install -y tmux)'
POD 'mkdir -p /workspace/logs && cd /workspace/sillicon-retina && tmux new-session -d -s fig12 "source /workspace/.env && bash scripts/runpod/dvs_fixed_200fps.sh 3 2>&1 | tee -a /workspace/logs/runner.log"'
```

The script does the following:

1. Installs PyTorch and the training requirements, and builds `esim_torch` (10–20 min).
2. Downloads the dataset into `/workspace/data/dhf1k_200fps` and checks all 2,167 files against `SHA256SUMS`.
3. Links the data to `src/neural_network/data/saliency_dataset_ready`.
4. Trains `DVS_fixed_200fps_run1`, `run2` and `run3` one after another.

### 4. Monitor

```bash
POD 'tail -n 5 /workspace/logs/runner.log; ls /workspace/logs; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv; df -h /workspace | tail -1'
POD 'grep -a -E "Train Loss|Best validation" /workspace/logs/DVS_fixed_200fps_run1.log | tail -3'
```

Poll every 30–60 min; there is no need to watch continuously.

**Milestones to report:**

* **A. Setup done.** The log shows the download finished and `sha256sum -c` printed nothing (no failures).
* **B. Run 1 started.** Check three things:
  * `DVS_fixed_200fps_run1.log` contains `dvs_input_video: fixed_200fps` in the printed configuration.
  * Cache files named `cache_B=5_ctn=0p2000_ctp=0p2000_v2_ff200_*.npy` appear in the clip folders: `ls /workspace/data/dhf1k_200fps/saliency_dataset_ready/train/video_00000/`.
  * Report the "avg grids occupancy" lines the script prints for train and val. For reference, the optical-flow DVS runs had ≈ 10.9 %; the old, mistimed fixed-rate runs had ≈ 7.8 %.
* **C. Epoch 1 of run 1 finished** ("Train Loss: …, Val Loss: …"). Building the event caches for 541 clips makes the first epoch slower, up to about an hour. Later epochs take about 2 min.
* **D. Each run finished.** The log ends with `Best validation loss: …`, and `/workspace/logs/DVS_fixed_200fps_runN.done` exists. Each run takes about 6.7 h. Report the best validation loss.

### 5. When something fails

| Symptom | Action |
|---|---|
| `esim_torch` build fails (nvcc / architecture errors) | Make sure `nvcc` is CUDA 12.x and the PyTorch wheel is cu12x. Retry with `export TORCH_CUDA_ARCH_LIST=8.9` added to `/workspace/.env`. If it still fails, report the error and stop |
| `CUDA error: no kernel image` or a driver-too-old error | Use the cu126 wheel: `pip uninstall -y torch torchvision`, change `TORCH_INDEX`, rerun step 3 |
| Hugging Face 401 / 403 / repository not found | The token lacks access. Ask the author |
| `sha256sum` reports FAILED files | Rerun step 3; the download resumes. If files still fail, report them |
| Disk full | Stop (`POD 'tmux kill-session -t fig12'`) and report. Do not delete data |
| Workers killed or out of memory (RAM) | Add `export NUM_WORKERS=4` and `export PREFETCH_FACTOR=2` to `/workspace/.env` and rerun step 3. This changes only data loading, not the results. Report the change |
| Training crashes mid-run | Report the end of the log. Rerunning step 3 restarts that run from epoch 1 (W&B then has two runs with that name; the export takes the finished one) and skips runs already marked `.done` |

### 6. Collect the results (small files only)

```bash
mkdir -p ~/fig12_rerun
rsync -av -e "<ssh options from POD>" root@<ip>:/workspace/logs root@<ip>:/workspace/configs ~/fig12_rerun/
```

Copy only the logs and configs. Do not copy the data or the event caches. Checkpoints are under
`/workspace/checkpoints`; copy them only if the author asks.

### 7. Update Figure 12 (on the laptop, in a checkout of the `reproducibility` branch)

```bash
pip install wandb pandas matplotlib seaborn SciencePlots jupyter
wandb login                       # offline mode on the pod: first run `wandb sync /workspace/sillicon-retina/wandb/offline-run-*` there
python scripts/runpod/export_fig12_runs.py --entity <W&B entity> --dry-run   # lists the 3 runs and their best losses
python scripts/runpod/export_fig12_runs.py --entity <W&B entity>
cd paper-figures && jupyter nbconvert --to notebook --execute analysis_saliency.ipynb --output analysis_saliency.ipynb
```

The export script does three things:

* replaces the three old fixed-rate runs in `paper-figures/analysis/comparison_{train,val}_loss.csv` with the new ones;
* renames them in the notebook;
* leaves re-rendering to the notebook, which regenerates `upsampling_convergence_curves_fixed.png`, `upsampling_comparison_boxplot_{train,val}.png` and `{train,val}_stats_*.csv`.

If LaTeX is missing, the notebook's `science` style fails. Run it on a copy with `'no-latex'` added to the style lists, and say so. Do not commit; show the author `git diff --stat`.

## Final report to the author

* The pod used: GPU, driver CUDA version, `TORCH_INDEX`, and the image.
* The grid occupancy the runs printed (train and val).
* For each run: best validation loss, epochs completed, wall time, and W&B run URL or id.
* For comparison, from the exported CSVs:
  * the top-5 mean validation loss of the new fixed-rate DVS runs;
  * the same for the optical-flow DVS runs (≈ 0.80–0.83 best);
  * the same for the retina runs (≈ 0.69–0.72 best).
* Any deviations from this brief (changed settings, restarts).
* Paths of the updated files, and a reminder that the pod can now be stopped.
