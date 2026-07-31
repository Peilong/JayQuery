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
        "input_dim": rc.run_config.get("input-dim-b", 26),
        "embedding_dim": rc.run_config.get("embedding-dim", 32),
        "dropout_rate": rc.run_config.get("dropout-b", 0.3),
        "hidden_dim": rc.run_config.get("hidden-dim-b", 128),
    },
    load_fn=load_silo_b,
    sends_labels=False,  # Silo A holds the labels
    wandb_tag="silo_b",
    extra_wandb_config_fn=lambda rc: {
        "input-dim-b": rc.run_config.get("input-dim-b", 26),
        "dropout-b": rc.run_config.get("dropout-b", 0.3),
        "hidden-dim-b": rc.run_config.get("hidden-dim-b", 128),
    },
    # Per-silo DP sensitivity. The shared `cut-noise-max-norm` was a documented
    # PLACEHOLDER of 1.0, while measured embedding row norms are ~5.6 (Silo A) and
    # ~4.0 (Silo B) — so 100% of rows were clipped to a fifth of their length,
    # destroying the relative-magnitude signal. The two silos have different bottom
    # models and different norm distributions, so they get separate values.
    cut_noise_config_fn=lambda rc: (
        {"max_norm": float(rc.run_config["cut-noise-max-norm-b"])}
        if "cut-noise-max-norm-b" in rc.run_config
        else {}
    ),
)


def get_emb(msg: Message, context: Context) -> Message:
    return common.generate_embeddings(SPEC, msg, context)

def get_eval_emb(msg: Message, context: Context) -> Message:
    return common.generate_eval_embeddings(SPEC, msg, context)

def apply_grad(msg: Message, context: Context) -> Message:
    return common.apply_gradients(SPEC, msg, context)