"""Model architecture unit tests — owned by Alex Pellet.

Run: pytest tests/test_models.py -v
"""

import pytest
import torch

SILO_A_FEATURES = 18
SILO_B_FEATURES = 26
EMBEDDING_DIM = 32

def _server_backward(top, emb_a, emb_b, labels):
    """Mirror of the server's inline forward/backward (server_app.py).

    The server detaches incoming embeddings, owns the backward (required for
    Opacus), and returns dL/d_emb per silo.
    """
    da = emb_a.detach().requires_grad_(True)
    db = emb_b.detach().requires_grad_(True)
    logits = top(da, db).squeeze(1)
    loss = top.loss_fn(logits, labels.float())
    loss.backward()
    return loss.detach(), da.grad.clone(), db.grad.clone()

@pytest.mark.model
def test_bottom_model_a_output_shape() -> None:
    """BottomModelA outputs (batch, embedding_dim) with correct shape."""
    from silo_a.bottom_model_a import BottomModelA

    model = BottomModelA(input_dim=SILO_A_FEATURES, embedding_dim=EMBEDDING_DIM)
    model.eval()
    x = torch.randn(16, SILO_A_FEATURES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, EMBEDDING_DIM), f"Expected (16, 32), got {out.shape}"


@pytest.mark.model
def test_bottom_model_b_output_shape() -> None:
    """BottomModelB outputs (batch, embedding_dim) with correct shape."""
    from silo_b.bottom_model_b import BottomModelB

    model = BottomModelB(input_dim=SILO_B_FEATURES, embedding_dim=EMBEDDING_DIM)
    model.eval()
    x = torch.randn(16, SILO_B_FEATURES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, EMBEDDING_DIM), f"Expected (16, 32), got {out.shape}"


@pytest.mark.model
def test_top_model_output_shape() -> None:
    """TopModel outputs (batch, 1) logit tensor."""
    from fl_server.top_model import TopModel

    model = TopModel(embedding_dim=EMBEDDING_DIM)
    model.eval()
    emb_a = torch.randn(16, EMBEDDING_DIM)
    emb_b = torch.randn(16, EMBEDDING_DIM)
    with torch.no_grad():
        out = model(emb_a, emb_b)
    assert out.shape == (16, 1), f"Expected (16, 1), got {out.shape}"


@pytest.mark.model
def test_top_model_gradient_flow() -> None:
    """TopModel.compute_loss_and_gradients() returns valid gradients for both silos."""
    from fl_server.top_model import TopModel

    model = TopModel(embedding_dim=EMBEDDING_DIM)
    emb_a = torch.randn(16, EMBEDDING_DIM)
    emb_b = torch.randn(16, EMBEDDING_DIM)
    labels = torch.randint(0, 2, (16,)).float()

    loss, grad_a, grad_b = _server_backward(model,emb_a, emb_b, labels)

    assert loss.item() > 0, "Loss should be positive"
    assert grad_a.shape == emb_a.shape, f"grad_a shape {grad_a.shape} != emb_a shape {emb_a.shape}"
    assert grad_b.shape == emb_b.shape, f"grad_b shape {grad_b.shape} != emb_b shape {emb_b.shape}"
    assert not torch.all(grad_a == 0), "grad_a is all zeros — gradient not flowing"
    assert not torch.all(grad_b == 0), "grad_b is all zeros — gradient not flowing"

@pytest.mark.model
def test_pos_weight_affects_loss() -> None:
    """set_pos_weight() actually changes the loss the server optimizes."""
    from fl_server.top_model import TopModel

    top = TopModel(embedding_dim=EMBEDDING_DIM)
    top.eval()
    torch.manual_seed(0)
    emb_a = torch.randn(16, EMBEDDING_DIM)
    emb_b = torch.randn(16, EMBEDDING_DIM)
    labels = torch.cat([torch.ones(4), torch.zeros(12)])

    with torch.no_grad():
        logits = top(emb_a, emb_b).squeeze(1)   # computed once, reused

    baseline = top.loss_fn(logits, labels).item()
    top.set_pos_weight(n_neg=12, n_pos=4)

    assert float(top.pos_weight) == pytest.approx(3.0)
    assert top.loss_fn(logits, labels).item() > baseline

@pytest.mark.model
def test_bottom_model_a_get_embedding_matches_forward() -> None:
    """get_embedding() and forward() return identical results."""
    from silo_a.bottom_model_a import BottomModelA

    model = BottomModelA(input_dim=SILO_A_FEATURES, embedding_dim=EMBEDDING_DIM)
    model.eval()
    x = torch.randn(8, SILO_A_FEATURES)
    with torch.no_grad():
        emb1 = model.forward(x)
        emb2 = model.get_embedding(x)
    assert torch.allclose(emb1, emb2), "forward() and get_embedding() must return identical tensors"


@pytest.mark.model
def test_eval_utils_metrics_keys() -> None:
    """compute_metrics() returns all expected metric keys."""
    import numpy as np

    from baselines.eval_utils import compute_metrics

    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3])
    metrics = compute_metrics(y_true, y_prob)
    required = {
        "auc_roc", "accuracy", "f1_macro", "precision", "recall", "brier_score",
        "pred_std", "pred_mean", "pred_pos_rate", "base_rate", "pred_collapsed",
    }
    assert required.issubset(metrics.keys()), f"Missing keys: {required - metrics.keys()}"


@pytest.mark.model
def test_eval_utils_health_flags_collapse() -> None:
    """pred_collapsed fires on one-class predictions, stays 0 on healthy ones."""
    import numpy as np

    from baselines.eval_utils import compute_metrics

    y_true = np.array([0, 1, 0, 1, 1, 0, 0, 1])

    # Healthy: spread-out probabilities on both sides of the threshold.
    healthy = compute_metrics(y_true, np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.15, 0.85]))
    assert healthy["pred_collapsed"] == 0.0
    assert healthy["pred_std"] > 0.05
    assert healthy["base_rate"] == 0.5

    # Collapsed: near-constant output → predicts negative for every row.
    constant = compute_metrics(y_true, np.full(8, 0.42))
    assert constant["pred_collapsed"] == 1.0
    assert constant["pred_std"] == 0.0
    assert constant["pred_pos_rate"] == 0.0
    assert constant["recall"] == 0.0  # classic all-negative signature

    # Collapsed the other way: everything predicted positive.
    all_pos = compute_metrics(y_true, np.linspace(0.6, 0.99, 8))
    assert all_pos["pred_collapsed"] == 1.0
    assert all_pos["pred_pos_rate"] == 1.0
    assert all_pos["recall"] == 1.0  # recall=1 + precision≈base_rate signature
    assert all_pos["precision"] == all_pos["base_rate"]


# ---------------------------------------------------------------------------
# Integration tests — the full SplitNN cut-layer wiring across all three models.
# These verify the pieces fit together, not just that each piece works alone.
# ---------------------------------------------------------------------------

@pytest.mark.model
def test_integration_bottom_to_top_forward() -> None:
    """Random data > both Bottom Models > Top Model"""
    from fl_server.top_model import TopModel
    from silo_a.bottom_model_a import BottomModelA
    from silo_b.bottom_model_b import BottomModelB

    batch = 16
    model_a = BottomModelA(input_dim=SILO_A_FEATURES, embedding_dim=EMBEDDING_DIM)
    model_b = BottomModelB(input_dim=SILO_B_FEATURES, embedding_dim=EMBEDDING_DIM)
    top = TopModel(embedding_dim=EMBEDDING_DIM)
    for m in (model_a, model_b, top):
        m.eval()

    x_a = torch.randn(batch, SILO_A_FEATURES)
    x_b = torch.randn(batch, SILO_B_FEATURES)

    with torch.no_grad():
        emb_a = model_a(x_a)
        emb_b = model_b(x_b)
        # TopModel.forward concatenates [emb_a | emb_b] -> (batch, 2*EMBEDDING_DIM).
        logits = top(emb_a, emb_b)

    assert emb_a.shape == (batch, EMBEDDING_DIM), f"emb_a shape {emb_a.shape}"
    assert emb_b.shape == (batch, EMBEDDING_DIM), f"emb_b shape {emb_b.shape}"
    assert logits.shape == (batch, 1), f"Expected ({batch}, 1), got {logits.shape}"
    assert torch.isfinite(logits).all(), "Top model produced non-finite logits"


@pytest.mark.model
def test_integration_bottom_models_receive_gradients() -> None:
    """Gradients returned by the Top Model backprop into both Bottom Models' params.

    The server detaches the incoming embeddings and returns dL/d_emb for each
    silo. Each silo then uses that as the upstream gradient for its own local
    backward pass — this test confirms that wiring populates real parameter
    gradients in both Bottom Models.
    """
    from fl_server.top_model import TopModel
    from silo_a.bottom_model_a import BottomModelA
    from silo_b.bottom_model_b import BottomModelB

    batch = 16
    model_a = BottomModelA(input_dim=SILO_A_FEATURES, embedding_dim=EMBEDDING_DIM)
    model_b = BottomModelB(input_dim=SILO_B_FEATURES, embedding_dim=EMBEDDING_DIM)
    top = TopModel(embedding_dim=EMBEDDING_DIM)
    for m in (model_a, model_b, top):
        m.train()

    x_a = torch.randn(batch, SILO_A_FEATURES)
    x_b = torch.randn(batch, SILO_B_FEATURES)
    labels = torch.randint(0, 2, (batch,)).float()

    # Silo-side forward — embeddings stay attached to each Bottom Model's graph.
    emb_a = model_a(x_a)
    emb_b = model_b(x_b)

    # Server returns the per-silo embedding gradients (it detaches internally).
    _loss, grad_a, grad_b = _server_backward(top, emb_a, emb_b, labels)
    assert grad_a.shape == emb_a.shape, "grad_a shape must match emb_a"
    assert grad_b.shape == emb_b.shape, "grad_b shape must match emb_b"

    # Each silo applies the server gradient as the upstream gradient.
    emb_a.backward(grad_a)
    emb_b.backward(grad_b)

    a_grads = [p.grad for p in model_a.parameters() if p.requires_grad]
    b_grads = [p.grad for p in model_b.parameters() if p.requires_grad]

    assert a_grads and all(g is not None for g in a_grads), "BottomModelA params got no gradient"
    assert b_grads and all(g is not None for g in b_grads), "BottomModelB params got no gradient"
    assert any(torch.any(g != 0) for g in a_grads), "BottomModelA gradients are all zero"
    assert any(torch.any(g != 0) for g in b_grads), "BottomModelB gradients are all zero"


@pytest.mark.model
def test_integration_full_round_trip() -> None:
    """Full VFL round: data → Bottom → Top → gradients → Bottom, with a learning step.

    Verifies every stage of one cut-layer round:
      1. Both Bottom Models emit correctly shaped embeddings.
      2. The Top Model yields a positive loss and per-silo gradients, and its own
         parameters receive gradients from the loss.
      3. Those gradients backprop into both Bottom Models.
      4. An optimizer step actually updates the weights of all three models.
    """
    from fl_server.top_model import TopModel
    from silo_a.bottom_model_a import BottomModelA
    from silo_b.bottom_model_b import BottomModelB

    batch = 16
    model_a = BottomModelA(input_dim=SILO_A_FEATURES, embedding_dim=EMBEDDING_DIM)
    model_b = BottomModelB(input_dim=SILO_B_FEATURES, embedding_dim=EMBEDDING_DIM)
    top = TopModel(embedding_dim=EMBEDDING_DIM)
    top.set_pos_weight(n_neg=12, n_pos=4)  # mimic class imbalance
    for m in (model_a, model_b, top):
        m.train()

    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-2)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-2)
    opt_top = torch.optim.Adam(top.parameters(), lr=1e-2)

    # Snapshot weights so we can confirm they change after the step.
    before_a = [p.detach().clone() for p in model_a.parameters()]
    before_b = [p.detach().clone() for p in model_b.parameters()]
    before_top = [p.detach().clone() for p in top.parameters()]

    opt_a.zero_grad()
    opt_b.zero_grad()
    opt_top.zero_grad()

    # --- Step 1: silo-local data -> embeddings ---
    x_a = torch.randn(batch, SILO_A_FEATURES)
    x_b = torch.randn(batch, SILO_B_FEATURES)
    labels = torch.randint(0, 2, (batch,)).float()

    emb_a = model_a(x_a)
    emb_b = model_b(x_b)
    assert emb_a.shape == (batch, EMBEDDING_DIM), f"emb_a shape {emb_a.shape}"
    assert emb_b.shape == (batch, EMBEDDING_DIM), f"emb_b shape {emb_b.shape}"

    # --- Step 2: server forward + loss + per-silo gradients ---
    loss, grad_a, grad_b = _server_backward(top, emb_a, emb_b, labels)
    assert loss.item() > 0, "Top model loss should be positive"
    assert grad_a.shape == emb_a.shape, "grad_a shape mismatch"
    assert grad_b.shape == emb_b.shape, "grad_b shape mismatch"
    assert torch.isfinite(grad_a).all() and torch.isfinite(grad_b).all(), "non-finite gradients"

    # The server's own parameters should have received gradients from the loss.
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in top.parameters()
    ), "Top model parameters received no gradient"

    # --- Step 3: gradients flow back into both Bottom Models ---
    emb_a.backward(grad_a)
    emb_b.backward(grad_b)
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in model_a.parameters()
    ), "BottomModelA received no usable gradient"
    assert any(
        p.grad is not None and torch.any(p.grad != 0) for p in model_b.parameters()
    ), "BottomModelB received no usable gradient"

    # --- Step 4: every model takes a learning step and its weights change ---
    opt_top.step()
    opt_a.step()
    opt_b.step()

    assert any(
        not torch.equal(b, p) for b, p in zip(before_a, model_a.parameters())
    ), "BottomModelA weights did not update after optimizer step"
    assert any(
        not torch.equal(b, p) for b, p in zip(before_b, model_b.parameters())
    ), "BottomModelB weights did not update after optimizer step"
    assert any(
        not torch.equal(b, p) for b, p in zip(before_top, top.parameters())
    ), "TopModel weights did not update after optimizer step"