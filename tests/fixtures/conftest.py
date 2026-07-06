"""Shared fixtures for testing fl_client.

Stubs ONLY flwr.app / flwr.clientapp (external Flower internals) so the
tests don't require a running Flower installation just to exercise plain
function logic. Implements only the surface fl_client/common.py and
fl_client/client_app.py actually use: Array, ArrayRecord, ConfigRecord,
RecordDict, Message, Context, and ClientApp's @query/@train decorators.

Everything else — utils.wandb_utils, silo_a.*, silo_b.* — is your REAL
project code and is imported as-is. If those modules are missing, broken,
or have a changed signature, these tests will fail with an
ImportError/AttributeError, which is the intended behavior: this suite is
meant to catch exactly that.

NOTE: the flwr stub below installs unconditionally on collection (it only
skips if flwr.app/flwr.clientapp are ALREADY imported into sys.modules by
something else first). That means this suite tests your logic's behavior
against this stand-in's contract, not against real installed Flower — it
will not catch a real flwr.app API change. Delete _install_flwr_stubs()'s
call below if you want these tests to run against your real flwr install
instead (requires flwr to actually be installed in the test environment).
"""

import sys
import types

import pytest
import torch
import torch.nn as nn


# ── flwr.app / flwr.clientapp stand-ins ─────────────────────────────────────


def _install_flwr_stubs() -> None:
    if "flwr.app" in sys.modules and "flwr.clientapp" in sys.modules:
        return

    flwr_pkg = types.ModuleType("flwr")
    flwr_pkg.__path__ = []
    flwr_app = types.ModuleType("flwr.app")

    class Array:
        """Wraps a numpy array; mirrors flwr.app.Array's .numpy() accessor."""

        def __init__(self, arr):
            self._arr = arr

        def numpy(self):
            return self._arr

    class ArrayRecord(dict):
        """dict[str, Array] with the to_torch_state_dict() helper common.py uses."""

        def to_torch_state_dict(self):
            return {k: torch.from_numpy(v.numpy()) for k, v in self.items()}

    class ConfigRecord(dict):
        pass

    class RecordDict(dict):
        pass

    class Message:
        def __init__(self, content=None, reply_to=None, message_type=None, dst_node_id=None):
            self.content = content if content is not None else RecordDict()
            self.reply_to = reply_to
            self.message_type = message_type
            self.dst_node_id = dst_node_id

    class _StateStore(dict):
        """context.state — get(key, default) plus dict-style assignment."""

        pass

    class Context:
        def __init__(self, run_config=None, node_config=None):
            self.run_config = run_config or {}
            self.node_config = node_config or {}
            self.state = _StateStore()

    flwr_app.Array = Array
    flwr_app.ArrayRecord = ArrayRecord
    flwr_app.ConfigRecord = ConfigRecord
    flwr_app.RecordDict = RecordDict
    flwr_app.Message = Message
    flwr_app.Context = Context

    flwr_clientapp = types.ModuleType("flwr.clientapp")

    class ClientApp:
        """@app.query(name)/@app.train(name) — registers nothing, just
        returns the function unchanged, since client_app.py only needs
        get_emb/apply_grad to be plain callables for these tests."""

        def query(self, name):
            def deco(fn):
                return fn

            return deco

        def train(self, name):
            def deco(fn):
                return fn

            return deco

    flwr_clientapp.ClientApp = ClientApp

    sys.modules["flwr"] = flwr_pkg
    sys.modules["flwr.app"] = flwr_app
    sys.modules["flwr.clientapp"] = flwr_clientapp


_install_flwr_stubs()


# ── fake model + loader (stand in for BottomModelA / BottomModelB) ─────────


class FakeBottomModel(nn.Module):
    """Minimal model with dropout + a linear layer — enough to exercise the
    RNG-replay path (dropout mask must match between the two forwards)."""

    def __init__(self, input_dim=5, embedding_dim=4, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(8, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class _FakeLoader:
    """Yields one fixed (X, y) batch — deterministic, no shuffling."""

    def __init__(self, X, y):
        self.X, self.y = X, y

    def __iter__(self):
        yield self.X, self.y


def make_fake_load_fn(n_rows=20, input_dim=5, seed=0):
    """Returns a load_fn(batch_size=...) -> (train_loader, None, None)
    closure with fixed, seeded synthetic data."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_rows, input_dim, generator=g)
    y = (torch.rand(n_rows, generator=g) > 0.5).float()

    def load_fn(batch_size=64):
        return _FakeLoader(X, y), None, None

    return load_fn


# ── pytest fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def wandb_calls(monkeypatch):
    """Spies on the REAL utils.wandb_utils.log_client_round (does not stub the
    module — if utils.wandb_utils doesn't exist or its signature has changed,
    this fixture will fail loudly via ImportError/AttributeError, as it should).

    Returns a list that fills with one dict of kwargs per log_client_round call.
    """
    import utils.wandb_utils as wandb_utils_module

    calls: list[dict] = []

    def _spy(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(wandb_utils_module, "log_client_round", _spy)
    # common.py does `from utils.wandb_utils import log_client_round` — patch
    # the bound name inside fl_client.common too, since rebinding the
    # attribute on the source module doesn't affect an already-imported name.
    import fl_client.common as common_module

    monkeypatch.setattr(common_module, "log_client_round", _spy)
    return calls


@pytest.fixture
def silo_spec_module():
    import fl_client.silo_spec as silo_spec_module

    return silo_spec_module


@pytest.fixture
def common_module():
    import fl_client.common as common_module

    return common_module


@pytest.fixture
def fresh_context():
    from flwr.app import Context

    def _make(run_config=None, node_config=None):
        return Context(
            run_config={
                "learning-rate": 1e-3,
                "batch-size": 64,
                "embedding-dim": 4,
                "track-wandb": True,
                "wandb-mode": "offline",
                **(run_config or {}),
            },
            node_config=node_config or {},
        )

    return _make


@pytest.fixture
def fake_spec_factory(silo_spec_module):
    """Returns a function building a fresh SiloSpec (independent fake data
    per call) with overridable fields."""

    def _make(**overrides):
        defaults = dict(
            partition_id=0,
            model_cls=FakeBottomModel,
            model_kwargs_fn=lambda ctx: {
                "input_dim": 5,
                "embedding_dim": 4,
                "dropout_rate": 0.5,
            },
            load_fn=make_fake_load_fn(),
            sends_labels=True,
            wandb_tag="silo_fake",
        )
        defaults.update(overrides)
        return silo_spec_module.SiloSpec(**defaults)

    return _make