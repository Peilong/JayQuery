# Setup Instructions for JayQuery
> These instructions are meant to set up a local environment up to the most recent iterations of the project. This does not necessarily mean that the project with be completely functional after completing these setup instructions.

## Basic setup for the repository and environment
```bash
# 1. Clone and set up environment
git clone https://github.com/Peilong/JayQuery.git
cd JayQuery
conda create -n jayquery python=3.11 && conda activate jayquery
pip install -r requirements.txt
pip install -e .

# 2. Copy and configure environment variables
cp .env.example .env # fill in WANDB_API_KEY

# 3. Download OULAD dataset
bash scripts/download_oulad.sh
```

## Running PSI & Silo Setup
```bash

python scripts/extract_ids_for_psi.py

python -m psi.psi_server &
python -m psi.psi_client

python scripts/prepare_silos.py

# Multi-module variants (--modules replaces --module; see CLAUDE.md step 5):
#   python scripts/prepare_silos.py --modules all --dedup-rule first \
#       --out_dir data/processed/dedup_first
```

## Testing the flwr server/client

Two run modes — identical app code, ~10x speed difference. Full details: CLAUDE.md step 6.

```bash
# SIMULATION (~0.75 s/round) — fast path for iteration and sweeps.
# `local` is the stock ~/.flwr/config.toml entry whose address ":local:" makes the CLI
# auto-start `flower-superlink --insecure --simulation`.
# num-supernodes=2 is REQUIRED: the default is 10, and the ServerApp needs exactly 2.
flwr run . local --federation-config "num-supernodes=2" \
    --run-config "processed-dir='data/processed/dedup_first' num-server-rounds=10"

# DEPLOYMENT (~8 s/round) — real SuperLink + 2 SuperNode processes.
# Use this for any result reported in the paper.
python scripts/run_deployment.py --rounds 10 \
    --run-config "processed-dir='data/processed/dedup_first'"
```

> NOTE (flwr 1.28): the old `flwr federation simulation-config --num-supernodes=2` +
> bare `flwr run .` flow **no longer exists** — that subcommand was removed and
> `[tool.flwr.federations]` in `pyproject.toml` is not parsed. Use the commands above.

## Run Model Baselines
```bash
#Wandb login, enter the API key you saved to the env variables if asked.
wandb login

#Can replace the --run all with --run x, where x is the model you want to run. The model names are defined in the baselines/run_baselines file.
#Also can replace any model parameters in exp01 with any that you want the baselines to use.
python -m baselines.run_baselines --config experiments/configs/exp01_centralized_baselines.yaml --run all

```