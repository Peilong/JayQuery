#!/usr/bin/env bash
# Reproduce all JayQuery paper results from scratch.
# Assumes: conda activate jayquery, OULAD downloaded, PSI run.

set -euo pipefail

echo "=== JayQuery: Reproducing All Experiments ==="
echo "Started: $(date)"

# Exp 1: Centralized baselines
echo "[Exp 1] Centralized baselines ..."
python -m baselines.centralized_mlp --config experiments/configs/exp01_centralized_baselines.yaml
python -m baselines.centralized_xgb

# Exp 2: VFL no-DP
echo "[Exp 2] VFL no-DP ..."
python -m fl_server.server --config experiments/configs/exp02_vfl_no_dp.yaml

# Exp 3: DP epsilon grid
echo "[Exp 3] DP epsilon grid ..."
for eps in 0.1 0.5 1.0 2.0 5.0 10.0; do
  echo "  epsilon = $eps"
  python -m fl_server.server --config experiments/configs/exp03_dp_epsilon_grid.yaml --epsilon "$eps"
done

echo "=== Done: $(date) ==="
echo "View all runs at: https://wandb.ai/etown-cs/jayquery"
