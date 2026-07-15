"""Silo B (Academic Affairs / VLE + Assessments) — SiloSpec + handler wrappers.

Owner: Alex Pellet
"""

from flwr.app import Context, Message

from fl_client import common
from fl_client.silo_spec import SiloSpec
from silo_b.bottom_model_b import BottomModelB
from silo_b.data_loader_b import load_silo_b

SPEC = SiloSpec(
    partition_id=1,  # MUST match _SILO_B_POS in server_app.py
    model_cls=BottomModelB,
    model_kwargs_fn=lambda rc: {
        "input_dim": rc.run_config.get("input-dim-b", 27),
        "embedding_dim": rc.run_config.get("embedding-dim", 32),
        "dropout_rate": rc.run_config.get("dropout-b", 0.3),
        "hidden_dim": rc.run_config.get("hidden-dim-b", 128),
    },
    load_fn=load_silo_b,
    sends_labels=False,  # Silo A holds the labels
    wandb_tag="silo_b",
    extra_wandb_config_fn=lambda rc: {
        "input-dim-b": rc.run_config.get("input-dim-b", 27),
        "dropout-b": rc.run_config.get("dropout-b", 0.3),
        "hidden-dim-b": rc.run_config.get("hidden-dim-b", 128),
    },
)


def get_emb(msg: Message, context: Context) -> Message:
    return common.generate_embeddings(SPEC, msg, context)

def get_eval_emb(msg: Message, context: Context) -> Message:
    return common.generate_eval_embeddings(SPEC, msg, context)

def apply_grad(msg: Message, context: Context) -> Message:
    return common.apply_gradients(SPEC, msg, context)