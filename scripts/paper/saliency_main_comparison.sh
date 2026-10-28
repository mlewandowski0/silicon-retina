#!/usr/bin/env bash
# Run from the repository root:  bash scripts/paper/<this script>
# Requires the prepared dataset (see README "Data and models") and the
# environment from scripts/setup/install.sh. Metrics are logged to Weights & Biases
# (set WANDB_MODE=offline to keep them local).
set -euo pipefail
C=src/neural_network/configs
RETINA="python -m src.neural_network.experiment.saliency_using_FIREnetwork_silicon_retina"
DVS="python -m src.neural_network.experiment.saliency_using_FIREnetwork_events_v2"

# Fig. "saliency_comparison" (train/val loss, Retina vs DVS events, DHF1K).
# W&B runs in paper-figures/analysis/retina_{train,val}_saliency.csv:
#   FIRE_saliency_seq=50_gk5_*  and  FIREnetwork_DVS_seq=50
# Despite its W&B name, run FIRE_saliency_seq=50_gk5_20251201_014614 was launched with the seq=30_gk7
# config (W&B metadata), and with the retina kernel before commit 396ebf1. The DVS run has no local
# record; seq=50 below matches its name. See reproducibility/experiments.yaml.
$RETINA -c $C/saliency/retina/FIREnetwork_retina_saliency_p2_eq_1.0_seq=30_gk7.json
$DVS    -c $C/saliency/events/FIREnetwork_event_saliency_seq=50.json
