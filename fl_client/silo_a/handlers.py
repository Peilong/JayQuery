"""Silo A (Financial Aid / Demographics) — SiloSpec + handler wrappers.

Owner: Alex Pellet
"""

from flwr.app import Context, Message

from fl_client import common
from fl_client.silo_spec import SiloSpec
from silo_a.bottom_model_a import BottomModelA
from silo_a.data_loader_a import load_silo_a

SPEC = SiloSpec(
    partition_id=0,  # MUST match _SILO_A_POS in server_app.py
    model_cls=BottomModelA,
    model_kwargs_fn=lambda rc: {
        "input_dim": rc.run_config.get("input-dim-a", 18),
        "embedding_dim": rc.run_config.get("embedding-dim", 32),
        "dropout_rate": rc.run_config.get("dropout-a", 0.3),
        "hidden_dim": rc.run_config.get("hidden-dim-a", 64),
    },
    load_fn=load_silo_a,
    sends_labels=True,  # Silo A is the active (label-holding) party
    wandb_tag="silo_a",
    extra_wandb_config_fn=lambda rc: {
        "input-dim-a": rc.run_config.get("input-dim-a", 18),
        "dropout-a": rc.run_config.get("dropout-a", 0.3),
        "hidden-dim-a": rc.run_config.get("hidden-dim-a", 64),
    },
)


def get_emb(msg: Message, context: Context) -> Message:
    return common.generate_embeddings(SPEC, msg, context)

def get_eval_emb(msg: Message, context: Context) -> Message:
    return common.generate_eval_embeddings(SPEC, msg, context)

def apply_grad(msg: Message, context: Context) -> Message:
    return common.apply_gradients(SPEC, msg, context)