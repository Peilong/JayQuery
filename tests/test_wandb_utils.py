"""Tests for the sweep orchestration the ICC grid runs on.

Run: pytest tests/test_wandb_utils.py -v

utils/wandb_utils.py drives every trial of the ~100-run grid: it renders the
sampled parameters into a `flwr run --run-config` string, shells out, then
reads the run's meta.json back and logs the summary. A bug anywhere in that
path costs a full grid rerun, and the module sat at 27% coverage.
"""

import json

import pytest

# ── the --run-config renderer ───────────────────────────────────────────────


class TestRunConfigRendering:
    """Sampled params become a TOML scalar string for `flwr run --run-config`."""

    def test_booleans_render_lowercase(self) -> None:
        """TOML booleans are `true`/`false`; Python's `True` is not valid TOML
        and would make flwr reject the whole run-config."""
        from utils.wandb_utils import _fmt_run_config_value

        assert _fmt_run_config_value(True) == "true"
        assert _fmt_run_config_value(False) == "false"

    def test_bool_is_not_rendered_as_an_int(self) -> None:
        """bool subclasses int in Python, so an isinstance(v, int) check that
        ran first would turn server-dp-enabled=True into `1`."""
        from utils.wandb_utils import _fmt_run_config_value

        assert _fmt_run_config_value(True) not in ("1", "True")

    def test_numbers_round_trip(self) -> None:
        from utils.wandb_utils import _fmt_run_config_value

        assert _fmt_run_config_value(150) == "150"
        assert float(_fmt_run_config_value(0.0005)) == pytest.approx(0.0005)
        assert float(_fmt_run_config_value(1e-5)) == pytest.approx(1e-5)

    def test_strings_are_quoted(self) -> None:
        from utils.wandb_utils import _fmt_run_config_value

        assert _fmt_run_config_value("data/processed") == "'data/processed'"

    def test_quotes_inside_strings_are_escaped(self) -> None:
        """An unescaped quote would terminate the TOML string early and
        silently truncate every later key in the run-config."""
        from utils.wandb_utils import _fmt_run_config_value

        out = _fmt_run_config_value("it's")
        assert out.startswith("'") and out.endswith("'")
        assert "\\'" in out

    def test_epsilon_grid_values_all_render(self) -> None:
        """Every value in the ICC grid must survive rendering."""
        from utils.wandb_utils import _fmt_run_config_value

        for eps in (0.5, 1, 2, 4, 8):
            assert float(_fmt_run_config_value(eps)) == pytest.approx(float(eps))
        for width in (8, 16, 32, 64):
            assert _fmt_run_config_value(width) == str(width)


# ── the sweep trial ─────────────────────────────────────────────────────────


class _FakeRun:
    def __init__(self, config, run_id="trial123"):
        self.config = config
        self.id = run_id
        self.tags = ()
        self.sweep_id = "sweep9"
        self.logged = {}

    def log(self, data, **_kw):
        self.logged.update(data)


@pytest.fixture
def sweep_env(monkeypatch, tmp_path):
    """Run a sweep trial with wandb and `flwr run` both faked out."""
    import utils.wandb_utils as wu

    state = {"cmd": None, "run": None}

    def _make(meta: dict, params: dict | None = None):
        run = _FakeRun(params if params is not None else {"learning-rate": 0.001})
        state["run"] = run

        fake_wandb = type("W", (), {"run": run, "init": lambda *a, **k: run})()
        monkeypatch.setattr(wu, "_require_wandb", lambda: fake_wandb)
        monkeypatch.setattr(wu, "finish_run", lambda *a, **k: None)
        monkeypatch.setattr(wu, "log_params", lambda *a, **k: None)

        meta_dir = tmp_path / "ckpt"
        (meta_dir / run.id).mkdir(parents=True)
        (meta_dir / run.id / "meta.json").write_text(json.dumps(meta))

        import subprocess

        def fake_run(cmd, **kwargs):
            state["cmd"] = cmd
            return type("P", (), {"returncode": 0})()

        monkeypatch.setattr(subprocess, "run", fake_run)

        trial = wu.build_flwr_sweep_fn(meta_dir=str(meta_dir))
        trial()
        return run, state["cmd"]

    return _make


_META = {
    "metrics": {"auc_roc": 0.83, "recall": 0.79, "tuned_threshold": 0.31},
    "best_val_auc": 0.85,
    "epsilon_spent": {
        "privacy/combined_epsilon": 1.79,
        "privacy/label_epsilon": 2.0,
    },
    "communication": {
        "bytes_up": 4096,
        "bytes_down": 2048,
        "bytes_total": 6144,
        "bytes_per_round": 3072.0,
        "rounds_counted": 2,
    },
}


class TestSweepTrial:
    """One trial: submit `flwr run`, read meta.json back, log the summary."""

    def test_tags_the_run_so_meta_lands_where_it_is_read(self, sweep_env) -> None:
        """run-tag must equal the wandb run id: the ServerApp writes
        meta.json under that name and the trial reads it back by the same
        name. A mismatch makes every trial score as a missing metric."""
        run, cmd = sweep_env(_META)
        rendered = cmd[cmd.index("--run-config") + 1]
        assert f"run-tag='{run.id}'" in rendered

    def test_sampled_params_reach_the_command(self, sweep_env) -> None:
        run, cmd = sweep_env(_META, params={"cut-noise-target-epsilon": 4,
                                            "embedding-dim": 16})
        rendered = cmd[cmd.index("--run-config") + 1]
        assert "cut-noise-target-epsilon=4" in rendered
        assert "embedding-dim=16" in rendered

    def test_logs_test_metrics_and_best_val_auc(self, sweep_env) -> None:
        run, _ = sweep_env(_META)
        assert run.logged["val/auc_best"] == pytest.approx(0.85)
        assert run.logged["test/auc_roc"] == pytest.approx(0.83)

    def test_logs_every_privacy_channel(self, sweep_env) -> None:
        """Each channel's epsilon must reach the sweep, including the label
        channel, since Table I reports them side by side."""
        run, _ = sweep_env(_META)
        assert run.logged["privacy/combined_epsilon"] == pytest.approx(1.79)
        assert run.logged["privacy/label_epsilon"] == pytest.approx(2.0)

    def test_logs_communication_cost(self, sweep_env) -> None:
        """C3 lives in meta.json's communication block. If the trial does not
        surface it, the bytes never reach wandb.ai and the entire
        communication-cost contribution is missing from the grid results."""
        run, _ = sweep_env(_META)
        assert run.logged["comm/bytes_total"] == pytest.approx(6144)
        assert run.logged["comm/bytes_per_round"] == pytest.approx(3072.0)

    def test_survives_a_run_without_dp_or_comm(self, sweep_env) -> None:
        """A no-DP baseline trial has no epsilon_spent; it must still log."""
        run, _ = sweep_env({"metrics": {"auc_roc": 0.9}, "best_val_auc": 0.91})
        assert run.logged["test/auc_roc"] == pytest.approx(0.9)


# ── reproducibility ─────────────────────────────────────────────────────────


class TestSeedEverything:
    """The 5-seed claim rests on this actually seeding every source."""

    def test_makes_torch_deterministic(self) -> None:
        import torch

        from utils.reproducibility import seed_everything

        seed_everything(42)
        a = torch.randn(32)
        seed_everything(42)
        b = torch.randn(32)
        assert torch.equal(a, b)

    def test_makes_numpy_deterministic(self) -> None:
        import numpy as np

        from utils.reproducibility import seed_everything

        seed_everything(7)
        a = np.random.rand(16)
        seed_everything(7)
        assert np.allclose(a, np.random.rand(16))

    def test_makes_python_random_deterministic(self) -> None:
        import random

        from utils.reproducibility import seed_everything

        seed_everything(3)
        a = [random.random() for _ in range(8)]
        seed_everything(3)
        assert a == [random.random() for _ in range(8)]

    def test_different_seeds_give_different_draws(self) -> None:
        """Otherwise a 5-seed grid would report one run five times with
        artificially tight error bars."""
        import torch

        from utils.reproducibility import seed_everything

        seed_everything(1)
        a = torch.randn(32)
        seed_everything(2)
        assert not torch.equal(a, torch.randn(32))


# ── placement expansion for the ICC grid ────────────────────────────────────


class TestPlacementExpansion:
    """`placement` + `epsilon` become the three channel switches.

    The sweep samples a readable placement name; the run-config needs three
    booleans and the right per-channel epsilon. Getting this wrong mislabels
    every cell of the grid, and the mistake would be invisible in the results.
    """

    def _expand(self, **params):
        from utils.wandb_utils import expand_placement

        return expand_placement(params)

    def test_none_disables_every_channel(self) -> None:
        cfg = self._expand(placement="none")
        assert cfg["server-dp-enabled"] is False
        assert cfg["cut-noise-enabled"] is False
        assert cfg["label-dp-enabled"] is False

    def test_server_placement_enables_only_dp_sgd(self) -> None:
        cfg = self._expand(placement="server", epsilon=2)
        assert cfg["server-dp-enabled"] is True
        assert cfg["server-dp-epsilon"] == 2
        assert cfg["cut-noise-enabled"] is False
        assert cfg["label-dp-enabled"] is False

    def test_cut_layer_placement_enables_only_embedding_noise(self) -> None:
        cfg = self._expand(placement="cut-layer", epsilon=4)
        assert cfg["cut-noise-enabled"] is True
        assert cfg["cut-noise-target-epsilon"] == 4
        assert cfg["server-dp-enabled"] is False
        assert cfg["label-dp-enabled"] is False

    def test_label_placement_enables_only_randomized_response(self) -> None:
        cfg = self._expand(placement="label", epsilon=1)
        assert cfg["label-dp-enabled"] is True
        assert cfg["label-dp-epsilon"] == 1
        assert cfg["server-dp-enabled"] is False
        assert cfg["cut-noise-enabled"] is False

    def test_all_enables_every_channel_at_the_same_epsilon(self) -> None:
        """Each channel gets the per-channel budget; the composed total is
        reported, not configured."""
        cfg = self._expand(placement="all", epsilon=8)
        assert cfg["server-dp-enabled"] is True
        assert cfg["cut-noise-enabled"] is True
        assert cfg["label-dp-enabled"] is True
        assert cfg["server-dp-epsilon"] == 8
        assert cfg["cut-noise-target-epsilon"] == 8
        assert cfg["label-dp-epsilon"] == 8

    def test_consumes_its_own_keys(self) -> None:
        """`placement` and `epsilon` are sweep axes, not run-config keys.
        Leaving them in would make flwr reject the run-config outright."""
        from utils.wandb_utils import expand_placement

        # passed by reference, not via **kwargs: the point is in-place removal
        params = {"placement": "server", "epsilon": 2, "seed": 42}
        expand_placement(params)
        assert "placement" not in params
        assert "epsilon" not in params
        assert params["seed"] == 42, "unrelated params must survive"

    def test_rejects_an_unknown_placement(self) -> None:
        """A typo in the sweep yaml must fail loudly, not silently run a
        no-DP configuration labelled as something else."""
        with pytest.raises(ValueError, match="placement"):
            self._expand(placement="cutlayer", epsilon=1)

    def test_requires_epsilon_when_a_channel_is_on(self) -> None:
        with pytest.raises(ValueError, match="epsilon"):
            self._expand(placement="all")

    def test_absent_placement_is_a_no_op(self) -> None:
        """Other sweeps (sweep_vfl.yaml) have no placement axis."""
        assert self._expand(**{"learning-rate": 0.001}) == {}
