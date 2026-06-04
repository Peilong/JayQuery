"""Model architecture unit tests — owned by Alex Pellet.

Run: pytest tests/test_models.py -v
"""

import pytest
import torch


@pytest.mark.model
def test_bottom_model_a_output_shape() -> None:
    """BottomModelA outputs (batch, embedding_dim) with correct shape."""
    from silo_a.bottom_model_a import BottomModelA

    model = BottomModelA(input_dim=12, embedding_dim=32)
    model.eval()
    x = torch.randn(16, 12)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, 32), f"Expected (16, 32), got {out.shape}"


@pytest.mark.model
def test_bottom_model_b_output_shape() -> None:
    """BottomModelB outputs (batch, embedding_dim) with correct shape."""
    from silo_b.bottom_model_b import BottomModelB

    model = BottomModelB(input_dim=22, embedding_dim=32)
    model.eval()
    x = torch.randn(16, 22)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (16, 32), f"Expected (16, 32), got {out.shape}"


@pytest.mark.model
def test_top_model_output_shape() -> None:
    """TopModel outputs (batch, 1) logit tensor."""
    from fl_server.top_model import TopModel

    model = TopModel(embedding_dim=32)
    model.eval()
    emb_a = torch.randn(16, 32)
    emb_b = torch.randn(16, 32)
    with torch.no_grad():
        out = model(emb_a, emb_b)
    assert out.shape == (16, 1), f"Expected (16, 1), got {out.shape}"


@pytest.mark.model
def test_top_model_gradient_flow() -> None:
    """TopModel.compute_loss_and_gradients() returns valid gradients for both silos."""
    from fl_server.top_model import TopModel

    model = TopModel(embedding_dim=32)
    emb_a = torch.randn(16, 32)
    emb_b = torch.randn(16, 32)
    labels = torch.randint(0, 2, (16,)).float()

    loss, grad_a, grad_b = model.compute_loss_and_gradients(emb_a, emb_b, labels)

    assert loss.item() > 0, "Loss should be positive"
    assert grad_a.shape == emb_a.shape, f"grad_a shape {grad_a.shape} != emb_a shape {emb_a.shape}"
    assert grad_b.shape == emb_b.shape, f"grad_b shape {grad_b.shape} != emb_b shape {emb_b.shape}"
    assert not torch.all(grad_a == 0), "grad_a is all zeros — gradient not flowing"
    assert not torch.all(grad_b == 0), "grad_b is all zeros — gradient not flowing"


@pytest.mark.model
def test_bottom_model_a_get_embedding_matches_forward() -> None:
    """get_embedding() and forward() return identical results."""
    from silo_a.bottom_model_a import BottomModelA

    model = BottomModelA(input_dim=12, embedding_dim=32)
    model.eval()
    x = torch.randn(8, 12)
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
    required = {"auc_roc", "accuracy", "f1_macro", "precision", "recall", "brier_score"}
    assert required.issubset(metrics.keys()), f"Missing keys: {required - metrics.keys()}"
