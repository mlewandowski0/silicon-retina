#!/usr/bin/env bash
# Run from the repository root:  bash scripts/paper/intensity_reconstruction.sh
# Fig. "comparison_intensity": FireNet intensity reconstruction from DVS vs Retina
# events on FILM-upsampled ECOCO (src/neural_network/data/upsampled_{train,validation}).
# These configs use the legacy `out_sigmoid` key (accepted by the current code) and
# add_bn=false, so the network is the same as in the original runs. The run-to-config mapping
# for this figure was not recorded; the configs below are the ones used by the
# original job scripts (scripts/history/jobs/job3.sh). The retina runs in the
# figure vary sequence length / burn-in (see legend), set in the config.
# FIREnetwork.json matches the DVS baseline run (FIREnetwork_20250901_173249): 5 bins and log-intensity
# DVS events. That run logged log_input=false but read caches built with log input; the config now says
# so explicitly. See reproducibility/experiments.yaml.
set -euo pipefail
C=src/neural_network/configs
python -m src.neural_network.experiment.reconstruct_intensity_using_FIREnetwork_events            -c $C/FIREnetwork.json
python -m src.neural_network.experiment.reconstruct_intensity_using_FIREnetwork_silicon_retina_v2 -c $C/FIREnetwork_event_bins_eq2.json
