# ----- MARKED FOR DELETION 22-05-26 : AMXP
# -- Using server_app.py instead.

"""Flower server entry point for JayQuery VFL training.

Owner: Alex Pellet

Usage:
    python -m fl_server.server --config experiments/configs/exp02_vfl_no_dp.yaml
    python -m fl_server.server --config experiments/configs/exp03_dp_epsilon_grid.yaml --epsilon 1.0
"""

import argparse
from pathlib import Path

import flwr as fl

import wandb
from fl_server.vfl_strategy import VFLStrategy
from utils.config import config_to_dict, load_config
from utils.reproducibility import seed_everything


def main() -> None:
    """Parse args, initialize wandb, and start Flower server."""
    parser = argparse.ArgumentParser(description="JayQuery VFL Server")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    parser.add_argument("--epsilon", type=float, default=None, help="Override DP epsilon")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epsilon is not None:
        cfg.privacy.epsilon = args.epsilon

    seed_everything(args.seed)

    # Initialize wandb
    wandb.init(
        project="jayquery",
        name=f"exp{cfg.experiment.id:02d}_{cfg.experiment.name}_eps{cfg.privacy.epsilon}",
        tags=[f"week{cfg.experiment.week}", "alex", f"eps_{cfg.privacy.epsilon}"],
        config=config_to_dict(cfg),
    )

    # Start Flower server
    strategy = VFLStrategy()
    fl.server.start_server(
        server_address=f"{cfg.server.host}:{cfg.server.port}",
        config=fl.server.ServerConfig(num_rounds=cfg.training.num_rounds),
        strategy=strategy,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
