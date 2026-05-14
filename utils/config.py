"""Configuration loader — reads YAML experiment configs via OmegaConf."""

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    """Load a YAML experiment config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        An OmegaConf DictConfig object with dot-access to all fields.

    Example:
        cfg = load_config("experiments/configs/exp02_vfl_no_dp.yaml")
        print(cfg.training.num_rounds)  # 50
    """
    return OmegaConf.load(config_path)


def config_to_dict(cfg: DictConfig) -> dict[str, Any]:
    """Convert OmegaConf config to a plain dict (for wandb.config logging)."""
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
