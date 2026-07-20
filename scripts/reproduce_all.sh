#!/usr/bin/env bash
# Reproduce all JayQuery paper results from scratch.
#
# Rewritten 2026-07-18 around the real run path (run_deployment.py → flwr run).
# The old version called `python -m fl_server.server`, which never existed.
#
# Prerequisites (run once, see README):
#   1. env installed:      pip install -r requirements.txt && pip install -e .
#   2. OULAD downloaded:   bash scripts/download_oulad.sh
#   3. PSI alignment:      python scripts/extract_ids_for_psi.py
#                          python -m psi.psi_server & python -m psi.psi_client
#   4. silos prepared:     python scripts/prepare_silos.py

set -euo pipefail

echo "=== JayQuery: Reproducing All Experiments ==="
echo "Started: $(date)"

# ── Exp 1: Centralized + isolated-silo baselines ────────────────────────────
echo "[Exp 1] Baselines ..."
python -m baselines.run_baselines \
  --config experiments/configs/exp01_centralized_baselines.yaml --run all

# ── Exp 2: VFL no-DP (upper bound) ──────────────────────────────────────────
# STUB — blocked on a run-config passthrough in run_deployment.py.
# run_deployment.py only forwards --rounds today; exp 2 needs DP disabled:
#   python scripts/run_deployment.py --rounds 50 \
#     --run-config "server-dp-enabled=false cut-noise-enabled=false"
# TODO(Alex): add a --run-config passthrough to run_deployment.py, then enable.
echo "[Exp 2] VFL no-DP ... SKIPPED (stub — see TODO above)"

# ── Exp 3: DP epsilon grid (core privacy-utility experiment) ────────────────
# STUB — same passthrough dependency as Exp 2. Per-epsilon runs look like:
#   for eps in 0.1 0.5 1.0 2.0 5.0 10.0; do
#     python scripts/run_deployment.py --rounds 50 \
#       --run-config "server-dp-epsilon=$eps cut-noise-target-epsilon=$eps"
#   done
# Each run must export a checkpoint bundle to
# experiments/checkpoints/exp03_eps{eps}/model.pt (both bottom models + top
# model + epsilon/delta + cut-layer noise params) so the attack code can load it.
# Hyperparameter sweeps are separate: python scripts/run_sweep.py (wandb sweep).
echo "[Exp 3] DP epsilon grid ... SKIPPED (stub — see TODO above)"

# ── Exp 4: Red Team attacks across the epsilon grid ─────────────────────────
# STUB — run model inversion (attack/inversion.py:run_inversion_attack) and
# label inference against each exp03 checkpoint; report attack success vs
# epsilon. TODO(Wyatt): add a CLI entry point over the checkpoint bundles.
echo "[Exp 4] Inversion / label-inference attacks ... SKIPPED (stub)"

# ── Exp 5: Temporal sufficiency ─────────────────────────────────────────────
# STUB — re-prepare silos truncated to the first N weeks of VLE data and
# re-run VFL to find the earliest actionable prediction week.
# TODO(Alex): add a --max-week option to scripts/prepare_silos.py.
echo "[Exp 5] Temporal sufficiency ... SKIPPED (stub)"

echo "=== Done: $(date) ==="
echo "View runs at: https://wandb.ai/jayquery/jayquery-vfl (VFL) and https://wandb.ai/jayquery/baselines"
