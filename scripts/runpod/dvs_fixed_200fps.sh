#!/usr/bin/env bash
# Corrected rerun of the "DVS (Fixed 200 FPS)" curve of Fig. 12 (Sec. 4.6) on a cloud GPU (e.g. RunPod RTX 4090).
#
# The original runs fed the 200 FPS video to ESIM with the optical-flow timestamps (uncommitted code),
# so their events were mistimed (reproducibility/audit.md). This script trains the same DVS
# saliency model on the same 200 FPS videos the retina runs used, with their own timestamps
# (config saliency/events/FIREnetwork_event_saliency_fixed_200fps.json).
#
# On the pod, from the repository root (use a persistent volume, e.g. /workspace):
#   export HF_TOKEN=hf_...                   # read access to the private dataset
#   export HF_DATASET=...                    # default: the private repo mlewand/silicon-retina-dhf1k-200fps
#   export WANDB_API_KEY=...                 # or WANDB_MODE=offline
#   bash scripts/runpod/dvs_fixed_200fps.sh [number of runs, default 3]
#
# About 6.7 h per run on an RTX 4090 (200 epochs); the first run also builds the DVS event caches.
# Runs are sequential because they share those caches. Re-running the script skips setup steps
# already done and the runs that already finished (marker files in $WORK/logs).
set -euo pipefail

RUNS=${1:-3}
WORK=${WORK:-/workspace}
DATA=${DATA_DIR:-$WORK/data/dhf1k_200fps}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}   # match the pod's CUDA driver
CONFIG=src/neural_network/configs/saliency/events/FIREnetwork_event_saliency_fixed_200fps.json
HF_DATASET=${HF_DATASET:-mlewand/silicon-retina-dhf1k-200fps}   # staged with reproducibility/tools/stage_saliency_upload.py
mkdir -p "$WORK/logs" "$WORK/configs"

# 1. environment (training only: no TensorFlow / GUI packages)
if ! python -c "import esim_torch" 2>/dev/null; then
    pip install torch torchvision --index-url "$TORCH_INDEX"
    grep -v -i -E '^(tensorflow|dearpygui|jupyter)' requirements.txt > "$WORK/requirements-train.txt"
    pip install -r "$WORK/requirements-train.txt" "huggingface_hub>=0.34"
    # needs nvcc (RunPod "pytorch ... devel" images have it); 8.9 = RTX 4090
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-8.9} \
        pip install -e src/neural_network/reusing_videos/esim_torch --no-build-isolation
fi

# 2. data: download, verify, link to the path the training script reads
if [ ! -f "$DATA/.verified" ]; then
    hf download "$HF_DATASET" --repo-type dataset --local-dir "$DATA"
    (cd "$DATA" && sha256sum -c --quiet SHA256SUMS)
    touch "$DATA/.verified"
fi
mkdir -p src/neural_network/data
ln -sfn "$DATA/saliency_dataset_ready" src/neural_network/data/saliency_dataset_ready

# 3. training runs (training is unseeded: runs differ by initialisation, sampling and augmentation)
for i in $(seq 1 "$RUNS"); do
    name=DVS_fixed_200fps_run$i
    [ -f "$WORK/logs/$name.done" ] && { echo "skip $name (done)"; continue; }
    # NUM_WORKERS / PREFETCH_FACTOR (optional) only change data loading, e.g. on a pod with little RAM
    python - "$CONFIG" "$WORK/configs/$name.json" "$name" "$WORK" <<'EOF'
import json, os, sys
src, dst, name, work = sys.argv[1:]
c = json.load(open(src))
c.update(wandb_name=name, checkpoint_dir=f"{work}/checkpoints/{name}", debug_folder=f"{work}/debug/{name}")
for key, env in (("num_workers", "NUM_WORKERS"), ("prefetch_factor", "PREFETCH_FACTOR")):
    if os.environ.get(env):
        c[key] = int(os.environ[env])
json.dump(c, open(dst, "w"), indent=2)
EOF
    python -m src.neural_network.experiment.saliency_using_FIREnetwork_events_v2 \
        -c "$WORK/configs/$name.json" 2>&1 | tee "$WORK/logs/$name.log"
    touch "$WORK/logs/$name.done"
done
echo "finished; export the runs from W&B (project e2vid-saliency2) as in paper-figures/analysis/comparison_*_loss.csv"
