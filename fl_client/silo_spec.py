"""SiloSpec: per-silo configuration consumed by common.py's generic handlers.

Owner: Alex Pellet
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class SiloSpec:
    """Everything common.py needs to know to treat a silo generically.

    Attributes:
        partition_id: Must match the server's _SILO_*_POS for this silo.
        model_cls: Bottom model class, e.g. BottomModelA.
        model_kwargs_fn: context -> kwargs dict for model_cls(**kwargs).
            A function, not a fixed dict, since each model's constructor
            signature may differ (e.g. dropout_rate may not apply to all).
        load_fn: Loader returning (train_loader, val_loader, test_loader).
        sends_labels: True only for the silo that forwards y to the server.
        wandb_tag: Short tag for metric keys / wandb tags, e.g. "silo_a".
    """

    partition_id: int
    model_cls: type[torch.nn.Module]
    model_kwargs_fn: Callable[[object], dict]
    load_fn: Callable[..., tuple[DataLoader, DataLoader, DataLoader]]
    sends_labels: bool
    wandb_tag: str
    dp_config_fn: Callable[[object], dict] = field(default=lambda context: {})
    extra_wandb_config_fn: Callable[[object], dict] = field(default=lambda context: {})
    cut_noise_config_fn: Callable[[object], dict] = field(default=lambda context: {})

    def __post_init__(self) -> None:
        #helps verify silo is formatted correctly.
        if not (isinstance(self.model_cls, type) and issubclass(self.model_cls, torch.nn.Module)):
            raise TypeError(f"model_cls must be a torch.nn.Module subclass, got {self.model_cls!r}")
        if not callable(self.model_kwargs_fn):
            raise TypeError("model_kwargs_fn must be callable(context) -> dict")
        if not callable(self.load_fn):
            raise TypeError("load_fn must be callable(...) -> (train, val, test) loaders")
        if not isinstance(self.partition_id, int):
            raise TypeError(f"partition_id must be int, got {self.partition_id!r}")
        if not callable(self.dp_config_fn):
            raise TypeError("dp_config_fn must be callable(context) -> dict")
        if not callable(self.cut_noise_config_fn):
            raise TypeError("cut_noise_config_fn must be callable(context) -> dict")