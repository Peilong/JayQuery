"""ServerApp unit tests — the VFL round loop and its helpers.

Run: pytest tests/test_fl_server.py -v

Covers `fl_server/server_app.py`, which had no tests. The emphasis is on the
invariants where a silent bug corrupts a training run without raising:

  * gradient routing — grad_a must reach the Silo A node, grad_b Silo B
  * batch alignment — both silos must receive the SAME batch seed, or the
    PSI row alignment breaks and the model trains on mismatched pairs
  * privacy gating — budget exhaustion must stop the run, not keep spending
  * best-state checkpointing — the reported metric must come from the peak
    state, not the collapsed last one

Like tests/test_fl_client.py, these run against the `flwr` stand-in installed
by tests/fixtures/conftest.py, not a real Flower federation. Everything else
(TopModel, privacy/, baselines.eval_utils) is the real project code.
"""

import pytest
import torch

# The flwr stubs must be installed (via tests/fixtures/conftest.py, loaded as a
# pytest plugin) before fl_server.server_app is imported.
from tests.fixtures.server_fakes import (
    SILO_A_POS,
    SILO_B_POS,
    FakeGrid,
    balanced_labels,
    emb_reply,
)


@pytest.fixture
def server_module():
    import fl_server.server_app as server_app_module

    return server_app_module


class TestWarmupFactor:
    """_warmup_factor: linear ramp start_factor -> 1.0 over warmup_rounds."""

    def test_disabled_warmup_returns_one(self, server_module) -> None:
        assert server_module._warmup_factor(1, 0, 0.1) == 1.0
        assert server_module._warmup_factor(1, -5, 0.1) == 1.0

    def test_past_warmup_returns_one(self, server_module) -> None:
        """At and after the cutoff round the multiplier is exactly 1.0."""
        assert server_module._warmup_factor(5, 5, 0.1) == 1.0
        assert server_module._warmup_factor(50, 5, 0.1) == 1.0

    def test_ramps_monotonically_within_bounds(self, server_module) -> None:
        factors = [server_module._warmup_factor(r, 5, 0.1) for r in range(1, 6)]
        assert factors == sorted(factors), f"not monotonic: {factors}"
        assert all(0.1 <= f <= 1.0 for f in factors), factors
        # round 1 of 5: 0.1 + 0.9 * (1/5)
        assert factors[0] == pytest.approx(0.28)

    def test_start_factor_one_is_flat(self, server_module) -> None:
        """A start factor of 1.0 means no warmup — every round is unscaled."""
        assert all(
            server_module._warmup_factor(r, 5, 1.0) == 1.0 for r in range(1, 7)
        )


class TestPosName:
    """_pos_name maps partition-id -> silo label used in metric keys."""

    def test_known_positions(self, server_module) -> None:
        assert server_module._pos_name(server_module._SILO_A_POS) == "silo_a"
        assert server_module._pos_name(server_module._SILO_B_POS) == "silo_b"


class TestServerDpConfig:
    """_server_dp_config reads the server-dp-* run-config keys."""

    def test_defaults_when_keys_absent(self, server_module) -> None:
        from flwr.app import Context

        cfg = server_module._server_dp_config(Context(run_config={}))
        assert cfg == {
            "enabled": False,
            "epsilon": 1.0,
            "delta": 1e-5,
            "max_grad_norm": 1.0,
        }

    def test_reads_configured_values(self, server_module) -> None:
        from flwr.app import Context

        cfg = server_module._server_dp_config(
            Context(
                run_config={
                    "server-dp-enabled": True,
                    "server-dp-epsilon": 2.5,
                    "server-dp-delta": 1e-6,
                    "server-dp-max-grad-norm": 0.5,
                }
            )
        )
        assert cfg["enabled"] is True
        assert cfg["epsilon"] == 2.5
        assert cfg["delta"] == 1e-6
        assert cfg["max_grad_norm"] == 0.5

    def test_coerces_string_numerics(self, server_module) -> None:
        """Run-config values arriving as strings must not poison the arithmetic."""
        from flwr.app import Context

        cfg = server_module._server_dp_config(
            Context(run_config={"server-dp-epsilon": "3.0"})
        )
        assert isinstance(cfg["epsilon"], float)
        assert cfg["epsilon"] == 3.0


class TestSendStop:
    """_send_stop: tell every node the run is over without expecting a reply."""

    def test_pushes_stop_to_every_node(self, server_module) -> None:
        grid = FakeGrid(node_ids=[11, 22])
        server_module._send_stop(grid, [11, 22], eps_tag="eps_1.0")

        assert len(grid.pushed) == 2, "expected one stop message per node"
        assert {m.dst_node_id for m in grid.pushed} == {11, 22}
        for msg in grid.pushed:
            assert msg.message_type == "train.apply_gradients"
            cfg = msg.content["config"]
            assert cfg["stop-run"] is True
            assert cfg["eps-tag"] == "eps_1.0"

    def test_does_not_wait_for_replies(self, server_module) -> None:
        """Stop is fire-and-forget — a send_and_receive here would hang a
        client that has already shut down."""
        grid = FakeGrid(node_ids=[11, 22])
        server_module._send_stop(grid, [11, 22], eps_tag="t")
        assert grid.send_and_receive_calls == 0


class TestGetEmbeddings:
    """_get_embeddings: collect one round's batch from both silos."""

    def test_assembles_both_silos_and_labels(self, server_module) -> None:
        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)

        rd = server_module._get_embeddings(
            grid=grid, node_ids=[11, 22], round_num=1,
            batch_seed=12345, sample_rate=0.5,
        )

        assert rd.emb_a is not None and rd.emb_b is not None
        assert rd.emb_a.shape == (8, 4)
        assert rd.emb_b.shape == (8, 4)
        assert rd.labels is not None and rd.labels.shape == (8,)
        assert not torch.equal(rd.emb_a, rd.emb_b), "silos returned identical embeddings"

    def test_sends_identical_batch_seed_to_both_silos(self, server_module) -> None:
        """THE PSI alignment invariant: both silos must Poisson-sample the same
        rows, which only works if they get the same seed and sample rate. A
        per-node seed here would silently train on mismatched student pairs."""
        grid = FakeGrid(node_ids=[11, 22])
        server_module._get_embeddings(
            grid=grid, node_ids=[11, 22], round_num=7,
            batch_seed=999, sample_rate=0.25,
        )

        queries = grid.queries_of_type("query.generate_embeddings")
        assert len(queries) == 2
        seeds = {q.content["config"]["batch-seed"] for q in queries}
        rates = {q.content["config"]["batch-sample-rate"] for q in queries}
        rounds = {q.content["config"]["server-round"] for q in queries}
        assert seeds == {999}, f"silos got different batch seeds: {seeds}"
        assert rates == {0.25}, f"silos got different sample rates: {rates}"
        assert rounds == {7}

    def test_records_node_position_map(self, server_module) -> None:
        """node_pos_map drives gradient routing, so it must come from the
        reply's own declared pos, not from node ordering."""
        def responder(messages, call_index):
            # node 11 answers as Silo B, node 22 as Silo A — deliberately
            # the reverse of the node_ids order.
            return [
                emb_reply(pos=SILO_B_POS, src_node_id=11),
                emb_reply(pos=SILO_A_POS, src_node_id=22),
            ]

        grid = FakeGrid(node_ids=[11, 22], responder=responder)
        rd = server_module._get_embeddings(
            grid=grid, node_ids=[11, 22], round_num=1,
            batch_seed=1, sample_rate=1.0,
        )
        assert rd.node_pos_map == {11: SILO_B_POS, 22: SILO_A_POS}

    def test_class_counts_taken_from_silo_a(self, server_module) -> None:
        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=10, n_pos=4)
        grid.n_rows = 10
        grid.n_total = 55

        rd = server_module._get_embeddings(
            grid=grid, node_ids=[11, 22], round_num=1,
            batch_seed=1, sample_rate=1.0,
        )
        assert (rd.n_pos, rd.n_neg) == (4, 6)
        assert rd.n_total == 55

    def test_collects_client_privacy_primitives(self, server_module) -> None:
        prims = {
            "target_epsilon": 1.0, "sigma": 2.0,
            "sample_rate": 0.1, "delta": 1e-5, "steps": 3,
        }

        def responder(messages, call_index):
            return [
                emb_reply(pos=SILO_A_POS, src_node_id=11, privacy=prims),
                emb_reply(pos=SILO_B_POS, src_node_id=22, privacy=prims),
            ]

        grid = FakeGrid(node_ids=[11, 22], responder=responder)
        rd = server_module._get_embeddings(
            grid=grid, node_ids=[11, 22], round_num=1,
            batch_seed=1, sample_rate=0.1,
        )
        assert set(rd.client_privacy) == {SILO_A_POS, SILO_B_POS}
        assert rd.client_privacy[SILO_A_POS]["steps"] == 3

    def test_budget_exhausted_flag_is_reported_not_raised(self, server_module) -> None:
        """An exhausted silo sends no embedding. That is a stop condition the
        loop handles, not an error — _get_embeddings must return cleanly."""
        def responder(messages, call_index):
            return [
                emb_reply(pos=SILO_A_POS, src_node_id=11, budget_exhausted=True),
                emb_reply(pos=SILO_B_POS, src_node_id=22),
            ]

        grid = FakeGrid(node_ids=[11, 22], responder=responder)
        rd = server_module._get_embeddings(
            grid=grid, node_ids=[11, 22], round_num=1,
            batch_seed=1, sample_rate=1.0,
        )
        assert rd.exhausted == [SILO_A_POS]
        assert rd.emb_a is None, "exhausted silo must not contribute an embedding"

    def test_missing_silo_raises(self, server_module) -> None:
        """Only one silo replied — training on a half batch would be wrong."""
        def responder(messages, call_index):
            return [emb_reply(pos=SILO_A_POS, src_node_id=11)]

        grid = FakeGrid(node_ids=[11, 22], responder=responder)
        with pytest.raises(RuntimeError, match="both silos"):
            server_module._get_embeddings(
                grid=grid, node_ids=[11, 22], round_num=1,
                batch_seed=1, sample_rate=1.0,
            )

    def test_contentless_reply_is_skipped(self, server_module) -> None:
        """A dropped/errored reply must not be parsed as a payload."""
        def responder(messages, call_index):
            return [
                emb_reply(pos=SILO_A_POS, src_node_id=11),
                emb_reply(pos=SILO_B_POS, src_node_id=22, error="node crashed"),
            ]

        grid = FakeGrid(node_ids=[11, 22], responder=responder)
        with pytest.raises(RuntimeError, match="both silos"):
            server_module._get_embeddings(
                grid=grid, node_ids=[11, 22], round_num=1,
                batch_seed=1, sample_rate=1.0,
            )


class TestSendGradients:
    """_send_gradients: route each silo its own slice of the loss gradient."""

    def test_routes_gradients_by_position_not_node_order(self, server_module) -> None:
        """THE routing invariant. node_pos_map is deliberately reversed
        (node 11 -> Silo B), so a helper that paired gradients with nodes by
        iteration order instead of by declared position would hand each silo
        the other silo's gradient — corrupting both bottom models with no
        error anywhere."""
        grad_a = torch.full((4, 3), 1.0)
        grad_b = torch.full((4, 3), 2.0)
        grid = FakeGrid(node_ids=[11, 22])

        server_module._send_gradients(
            grid=grid,
            node_pos_map={11: SILO_B_POS, 22: SILO_A_POS},
            grad_a=grad_a, grad_b=grad_b,
            round_num=3, num_rounds=10, eps_tag="eps_1.0",
        )

        by_node = {m.dst_node_id: m for m in grid.gradient_pushes()}
        assert set(by_node) == {11, 22}
        got_11 = by_node[11].content["gradients"]["local-gradients"].numpy()
        got_22 = by_node[22].content["gradients"]["local-gradients"].numpy()
        assert (got_11 == 2.0).all(), "node 11 is Silo B — it must receive grad_b"
        assert (got_22 == 1.0).all(), "node 22 is Silo A — it must receive grad_a"

    def test_gradient_values_round_trip(self, server_module) -> None:
        torch.manual_seed(0)
        grad_a = torch.randn(5, 4)
        grad_b = torch.randn(5, 4)
        grid = FakeGrid(node_ids=[11, 22])

        server_module._send_gradients(
            grid=grid, node_pos_map={11: SILO_A_POS, 22: SILO_B_POS},
            grad_a=grad_a, grad_b=grad_b,
            round_num=1, num_rounds=1, eps_tag="t",
        )

        by_node = {m.dst_node_id: m for m in grid.gradient_pushes()}
        sent_a = torch.from_numpy(by_node[11].content["gradients"]["local-gradients"].numpy())
        assert torch.allclose(sent_a, grad_a)

    def test_config_carries_round_and_control_flags(self, server_module) -> None:
        grid = FakeGrid(node_ids=[11, 22])
        server_module._send_gradients(
            grid=grid, node_pos_map={11: SILO_A_POS, 22: SILO_B_POS},
            grad_a=torch.zeros(2, 2), grad_b=torch.zeros(2, 2),
            round_num=4, num_rounds=20, eps_tag="eps_0.5",
            save_best=True, lr=0.003,
        )

        for msg in grid.gradient_pushes():
            cfg = msg.content["config"]
            assert cfg["server-round"] == 4
            assert cfg["num-server-rounds"] == 20
            assert cfg["eps-tag"] == "eps_0.5"
            assert cfg["save-best"] is True
            assert cfg["lr"] == pytest.approx(0.003)

    def test_absent_lr_uses_sentinel(self, server_module) -> None:
        """lr=None must serialize as a number (ConfigRecord takes no None)."""
        grid = FakeGrid(node_ids=[11, 22])
        server_module._send_gradients(
            grid=grid, node_pos_map={11: SILO_A_POS, 22: SILO_B_POS},
            grad_a=torch.zeros(2, 2), grad_b=torch.zeros(2, 2),
            round_num=1, num_rounds=1, eps_tag="t", lr=None,
        )
        for msg in grid.gradient_pushes():
            assert msg.content["config"]["lr"] == -1.0

    def test_push_not_send_and_receive(self, server_module) -> None:
        """Gradients are fire-and-forget; waiting on a reply would serialize
        the round and deadlock against the clients' next query."""
        grid = FakeGrid(node_ids=[11, 22])
        server_module._send_gradients(
            grid=grid, node_pos_map={11: SILO_A_POS, 22: SILO_B_POS},
            grad_a=torch.zeros(2, 2), grad_b=torch.zeros(2, 2),
            round_num=1, num_rounds=1, eps_tag="t",
        )
        assert grid.send_and_receive_calls == 0
        assert grid.push_calls == 1, "both gradients should go in one push batch"


class TestCollectEvalPredictions:
    """_collect_eval_predictions: one distributed eval round -> (labels, probs)."""

    def test_returns_probabilities_and_node_map(self, server_module) -> None:
        """Post-refactor contract: the coordinator gets predictions and the
        node map, NOT labels. It holds no ground truth."""
        from fl_server.top_model import TopModel

        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)
        top = TopModel(embedding_dim=4, hidden_dim=8)

        probs, node_pos_map = server_module._collect_eval_predictions(
            grid, [11, 22], top, split="val"
        )

        assert probs.shape == (8,), "probs must be 1-D (squeezed), not (N, 1)"
        assert torch.all((probs >= 0.0) & (probs <= 1.0)), "sigmoid output out of range"
        assert node_pos_map == {11: SILO_A_POS, 22: SILO_B_POS}

    def test_coordinator_never_receives_eval_labels(self, server_module) -> None:
        """THE neutral-coordinator invariant. Eval labels must not cross the
        wire at all: if they did, the label-DP guarantee would hold for
        training and silently leak the entire test split at evaluation."""
        from fl_server.top_model import TopModel

        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)

        server_module._collect_eval_predictions(
            grid, [11, 22], TopModel(embedding_dim=4, hidden_dim=8), split="test"
        )

        for batch in grid.received:
            for reply in batch:
                arrays = reply.content.get("arrays", {})
                assert "labels" not in arrays, (
                    "a silo returned labels in its eval reply — the "
                    "coordinator must never see ground truth"
                )

    def test_puts_model_in_eval_mode(self, server_module) -> None:
        """Dropout must be off during eval, or the reported metric is noise."""
        from fl_server.top_model import TopModel

        grid = FakeGrid(node_ids=[11, 22])
        top = TopModel(embedding_dim=4, hidden_dim=8, dropout_rate=0.5)
        top.train()

        server_module._collect_eval_predictions(grid, [11, 22], top, split="val")
        assert not top.training, "model left in train mode during eval"

    def test_eval_is_deterministic(self, server_module) -> None:
        """Same weights + same embeddings -> identical probs (dropout off)."""
        from fl_server.top_model import TopModel

        grid = FakeGrid(node_ids=[11, 22])
        grid.emb_seed = 3  # same embeddings both calls, so only dropout could differ
        top = TopModel(embedding_dim=4, hidden_dim=8, dropout_rate=0.5)

        probs1, _ = server_module._collect_eval_predictions(grid, [11, 22], top, "val")
        probs2, _ = server_module._collect_eval_predictions(grid, [11, 22], top, "val")
        assert torch.allclose(probs1, probs2)

    def test_split_and_extra_config_reach_the_clients(self, server_module) -> None:
        from fl_server.top_model import TopModel

        grid = FakeGrid(node_ids=[11, 22])
        top = TopModel(embedding_dim=4, hidden_dim=8)

        server_module._collect_eval_predictions(
            grid, [11, 22], top, split="test",
            extra_cfg={"restore-best": True, "eps-tag": "eps_1.0"},
        )

        queries = grid.queries_of_type("query.generate_eval_embeddings")
        assert len(queries) == 2
        for q in queries:
            cfg = q.content["config"]
            assert cfg["split"] == "test"
            assert cfg["restore-best"] is True
            assert cfg["eps-tag"] == "eps_1.0"

    def test_missing_silo_raises(self, server_module) -> None:
        from fl_server.top_model import TopModel

        def responder(messages, call_index):
            return [emb_reply(pos=SILO_B_POS, src_node_id=22)]

        grid = FakeGrid(node_ids=[11, 22], responder=responder)
        with pytest.raises(RuntimeError, match="both silos"):
            server_module._collect_eval_predictions(
                grid, [11, 22], TopModel(embedding_dim=4, hidden_dim=8), "val"
            )


class TestRunEval:
    """_run_eval: eval round -> metrics dict at a given threshold."""

    def test_returns_metric_suite(self, server_module) -> None:
        from fl_server.top_model import TopModel

        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)

        metrics = server_module._run_eval(
            grid, [11, 22], TopModel(embedding_dim=4, hidden_dim=8), split="val"
        )

        for key in ("auc_roc", "recall", "precision", "base_rate", "pred_collapsed"):
            assert key in metrics, f"missing {key}"
        assert metrics["base_rate"] == pytest.approx(3 / 8)

    def test_threshold_changes_the_binary_metrics(self, server_module) -> None:
        """The threshold must actually reach compute_metrics — the DP path
        depends on scoring at a tuned, non-0.5 operating point."""
        from fl_server.top_model import TopModel

        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)
        top = TopModel(embedding_dim=4, hidden_dim=8)

        strict = server_module._run_eval(grid, [11, 22], top, "val", threshold=0.99)
        loose = server_module._run_eval(grid, [11, 22], top, "val", threshold=0.01)

        assert loose["pred_pos_rate"] > strict["pred_pos_rate"], (
            "a lower threshold must flag more students"
        )


def _tracker(steps: int = 0, target_epsilon: float = 10.0, sigma: float = 2.0,
             sample_rate: float = 0.1, delta: float = 1e-5):
    """A PrivacyBudgetTracker that has already spent `steps` rounds."""
    from privacy.privacy_accountant import PrivacyBudgetTracker

    t = PrivacyBudgetTracker(
        target_epsilon=target_epsilon, sigma=sigma,
        sample_rate=sample_rate, delta=delta,
    )
    t.steps = steps
    return t


@pytest.mark.dp
class TestEpsilonMetrics:
    """_epsilon_metrics: per-source epsilons plus the composed system total."""

    def test_no_sources_gives_no_metrics(self, server_module) -> None:
        assert server_module._epsilon_metrics(None, {}, delta=1e-5) == {}

    def test_server_only(self, server_module) -> None:
        metrics = server_module._epsilon_metrics(_tracker(steps=5), {}, delta=1e-5)
        # NB: "silo" is a substring of "epsilon" — match whole keys, not parts.
        assert set(metrics) == {
            "privacy/server_epsilon",
            "privacy/combined_epsilon",
        }

    def test_clients_only_are_named_per_silo(self, server_module) -> None:
        metrics = server_module._epsilon_metrics(
            None, {SILO_A_POS: _tracker(steps=3), SILO_B_POS: _tracker(steps=3)},
            delta=1e-5,
        )
        assert "privacy/silo_a_epsilon" in metrics
        assert "privacy/silo_b_epsilon" in metrics
        assert "privacy/server_epsilon" not in metrics
        assert "privacy/combined_epsilon" in metrics

    def test_all_three_sources_reported(self, server_module) -> None:
        metrics = server_module._epsilon_metrics(
            _tracker(steps=4),
            {SILO_A_POS: _tracker(steps=4), SILO_B_POS: _tracker(steps=4)},
            delta=1e-5,
        )
        assert set(metrics) == {
            "privacy/server_epsilon",
            "privacy/silo_a_epsilon",
            "privacy/silo_b_epsilon",
            "privacy/combined_epsilon",
        }

    def test_combined_is_at_least_any_single_source(self, server_module) -> None:
        """Composition across three mechanisms can only cost more than one of
        them alone. A combined epsilon below a per-source value would mean the
        paper under-reports what the system actually spends."""
        metrics = server_module._epsilon_metrics(
            _tracker(steps=6),
            {SILO_A_POS: _tracker(steps=6), SILO_B_POS: _tracker(steps=6)},
            delta=1e-5,
        )
        singles = [v for k, v in metrics.items() if k != "privacy/combined_epsilon"]
        assert metrics["privacy/combined_epsilon"] >= max(singles) - 1e-9

    def test_unspent_sources_report_zero_epsilon(self, server_module) -> None:
        """A tracker at steps=0 has spent nothing."""
        metrics = server_module._epsilon_metrics(_tracker(steps=0), {}, delta=1e-5)
        assert metrics["privacy/server_epsilon"] == pytest.approx(0.0)


@pytest.mark.dp
class TestNextRoundCombined:
    """_next_round_combined: prospective epsilon if every source spends again."""

    def test_none_when_no_active_sources(self, server_module) -> None:
        assert server_module._next_round_combined(None, {}, delta=1e-5) is None

    def test_does_not_mutate_the_live_trackers(self, server_module) -> None:
        """THE gating invariant. This is a read-only lookahead run every round.
        If it advanced the real trackers, each gate check would itself burn a
        step of privacy budget — inflating the reported epsilon and halting the
        run early, with nothing in the logs to show why."""
        server = _tracker(steps=5)
        silo_a, silo_b = _tracker(steps=5), _tracker(steps=5)
        before = (server.steps, silo_a.steps, silo_b.steps)
        eps_before = server.get_epsilon()

        server_module._next_round_combined(
            server, {SILO_A_POS: silo_a, SILO_B_POS: silo_b}, delta=1e-5
        )

        assert (server.steps, silo_a.steps, silo_b.steps) == before
        assert server.get_epsilon() == pytest.approx(eps_before)

    def test_prospective_exceeds_current(self, server_module) -> None:
        """One more step must cost strictly more budget, or the gate can never
        trip and combined-target-epsilon is unenforced."""
        trackers = {SILO_A_POS: _tracker(steps=5), SILO_B_POS: _tracker(steps=5)}
        server = _tracker(steps=5)

        current = server_module._epsilon_metrics(server, trackers, delta=1e-5)[
            "privacy/combined_epsilon"
        ]
        prospective = server_module._next_round_combined(server, trackers, delta=1e-5)
        assert prospective > current

    def test_server_only_still_probes(self, server_module) -> None:
        assert server_module._next_round_combined(
            _tracker(steps=2), {}, delta=1e-5
        ) is not None

    def test_clients_only_still_probes(self, server_module) -> None:
        assert server_module._next_round_combined(
            None, {SILO_A_POS: _tracker(steps=2)}, delta=1e-5
        ) is not None


# ── the round loop ──────────────────────────────────────────────────────────


EMB_DIM = 4


@pytest.fixture
def run_server(server_module, tmp_path, monkeypatch):
    """Run server_app.main() against a FakeGrid in an isolated cwd.

    Checkpoints are written to a relative path, so the cwd must be temporary
    or a test run would litter the real experiments/ tree.
    """
    from flwr.app import Context

    def _run(grid=None, **run_config):
        monkeypatch.chdir(tmp_path)
        if grid is None:
            grid = FakeGrid(node_ids=[11, 22])
            grid.labels = balanced_labels(n_rows=8, n_pos=3)
        cfg = {
            "num-server-rounds": 2,
            "embedding-dim": EMB_DIM,
            "learning-rate": 1e-2,
            "hidden-dim-top": 8,
            "track-wandb": False,
            "server-dp-enabled": False,
            "batch-sample-rate": 0.1,
            "seed": 42,
            "eval-every": 100,  # off unless a test asks for it
            **run_config,
        }
        server_module.main(grid, Context(run_config=cfg))
        return grid, tmp_path

    return _run


def _fake_eval(auc_by_round, states=None):
    """A _run_eval stand-in returning a scripted val/auc per call.

    Args:
        states: If given, each call appends the top-model state it was handed,
            so a test can prove which round's weights got checkpointed.
    """
    calls = {"n": 0}

    def _run_eval(grid, node_ids, top_model, split, extra_cfg=None, threshold=0.5):
        calls["n"] += 1
        auc = auc_by_round[min(calls["n"], len(auc_by_round)) - 1]
        if states is not None:
            states.append(
                {k: v.detach().clone() for k, v in top_model.state_dict().items()}
            )
        return {
            "auc_roc": auc,
            "pred_collapsed": 0.0,
            "pred_std": 0.2,
            "pred_pos_rate": 0.3,
            "base_rate": 0.375,
            "recall": 0.5,
            "precision": 0.5,
        }

    _run_eval.calls = calls
    return _run_eval


class TestNodeGuard:
    """The loop requires exactly two SuperNodes — one per silo."""

    @pytest.mark.parametrize("node_ids", [[], [11], [11, 22, 33]])
    def test_wrong_node_count_raises(self, run_server, node_ids) -> None:
        grid = FakeGrid(node_ids=node_ids)
        with pytest.raises(ValueError, match="exactly 2 SuperNodes"):
            run_server(grid=grid)


class TestTrainingLoop:
    """main(): the per-round query -> backward -> push cycle."""

    def test_runs_every_round_and_pushes_gradients(self, run_server) -> None:
        grid, _ = run_server(**{"num-server-rounds": 3})

        assert len(grid.gradient_pushes()) == 6, (
            "expected 2 gradient messages (one per silo) for each of 3 rounds"
        )
        rounds_seen = sorted(
            m.content["config"]["server-round"] for m in grid.gradient_pushes()
        )
        assert rounds_seen == [1, 1, 2, 2, 3, 3]

    def test_batch_seed_changes_every_round(self, run_server) -> None:
        """A fixed seed across rounds would train on the same batch forever."""
        grid, _ = run_server(**{"num-server-rounds": 3, "seed": 7})

        seeds = [
            q.content["config"]["batch-seed"]
            for q in grid.queries_of_type("query.generate_embeddings")
        ]
        # two silos per round, same seed within a round, different across rounds
        per_round = [seeds[i] for i in range(0, len(seeds), 2)]
        assert len(set(per_round)) == 3, f"batch seeds repeated: {per_round}"
        # documented formula: seed * 100_000 + round_num
        assert per_round == [7 * 100_000 + r for r in (1, 2, 3)]

    def test_gradient_shape_matches_embedding(self, run_server) -> None:
        grid, _ = run_server()
        for msg in grid.gradient_pushes():
            arr = msg.content["gradients"]["local-gradients"].numpy()
            assert arr.shape == (8, EMB_DIM)

    def test_gradients_are_nonzero(self, run_server) -> None:
        """An all-zero gradient means the bottom models receive no learning
        signal — the silos would never train."""
        grid, _ = run_server()
        for msg in grid.gradient_pushes():
            arr = msg.content["gradients"]["local-gradients"].numpy()
            assert abs(arr).sum() > 0, "gradient is all zeros"

    def test_top_model_weights_actually_update(self, server_module, run_server,
                                               monkeypatch) -> None:
        """The optimizer must step — otherwise the whole run is a no-op."""
        snapshots = []
        real_send = server_module._send_gradients

        def spy(*args, **kwargs):
            # capture the live top model's first-layer weights each round
            snapshots.append(kwargs["grad_a"].clone())
            return real_send(*args, **kwargs)

        monkeypatch.setattr(server_module, "_send_gradients", spy)
        run_server(**{"num-server-rounds": 3, "learning-rate": 0.5})

        assert len(snapshots) == 3
        assert not torch.allclose(snapshots[0], snapshots[1]), (
            "gradient identical across rounds — weights are not updating"
        )

    def test_lr_warmup_is_applied_to_sent_lr(self, run_server) -> None:
        """Clients mirror the server LR; warmup must reach them."""
        grid, _ = run_server(
            **{
                "num-server-rounds": 4,
                "learning-rate": 1.0,
                "lr-warmup-rounds": 4,
                "lr-warmup-start-factor": 0.25,
            }
        )
        lrs = [
            m.content["config"]["lr"]
            for m in grid.gradient_pushes()
            if m.dst_node_id == 11
        ]
        assert lrs == sorted(lrs), f"LR not ramping up during warmup: {lrs}"
        assert lrs[0] < lrs[-1]


class TestPosWeight:
    """Round-1 class-imbalance weighting (the pos-weight run-config modes)."""

    def test_auto_mode_uses_reported_class_counts(self, server_module, run_server,
                                                  monkeypatch) -> None:
        captured = {}
        real = server_module.TopModel.set_pos_weight

        def spy(self, n_neg, n_pos):
            captured["n_neg"], captured["n_pos"] = n_neg, n_pos
            return real(self, n_neg, n_pos)

        monkeypatch.setattr(server_module.TopModel, "set_pos_weight", spy)
        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=2)  # 2 pos, 6 neg
        run_server(grid=grid, **{"pos-weight": -1.0})

        assert (captured["n_neg"], captured["n_pos"]) == (6, 2)

    def test_zero_disables_weighting(self, server_module, run_server,
                                     monkeypatch) -> None:
        captured = {}
        real = server_module.TopModel.set_pos_weight

        def spy(self, n_neg, n_pos):
            real(self, n_neg, n_pos)
            captured["weight"] = float(self.pos_weight)

        monkeypatch.setattr(server_module.TopModel, "set_pos_weight", spy)
        run_server(**{"pos-weight": 0.0})
        assert captured["weight"] == pytest.approx(1.0)

    def test_explicit_value_is_honored(self, server_module, run_server,
                                       monkeypatch) -> None:
        captured = {}
        real = server_module.TopModel.set_pos_weight

        def spy(self, n_neg, n_pos):
            real(self, n_neg, n_pos)
            captured["weight"] = float(self.pos_weight)

        monkeypatch.setattr(server_module.TopModel, "set_pos_weight", spy)
        run_server(**{"pos-weight": 2.5})
        assert captured["weight"] == pytest.approx(2.5)

    def test_set_once_not_every_round(self, server_module, run_server,
                                      monkeypatch) -> None:
        """Re-deriving pos_weight from each mini-batch would make the loss
        objective wobble round to round."""
        calls = {"n": 0}
        real = server_module.TopModel.set_pos_weight

        def spy(self, n_neg, n_pos):
            calls["n"] += 1
            return real(self, n_neg, n_pos)

        monkeypatch.setattr(server_module.TopModel, "set_pos_weight", spy)
        run_server(**{"num-server-rounds": 4})
        assert calls["n"] == 1


class TestCheckpointing:
    """Best-state selection and the artifacts written at the end of a run."""

    def test_writes_checkpoint_and_meta(self, run_server) -> None:
        _, tmp_path = run_server(**{"run-tag": "unit"})

        ckpt_dir = tmp_path / "experiments" / "checkpoints" / "unit"
        assert (ckpt_dir / "top.pt").exists()
        assert (ckpt_dir / "meta.json").exists()

    def test_meta_records_run_parameters(self, run_server) -> None:
        import json

        _, tmp_path = run_server(**{"num-server-rounds": 2, "run-tag": "unit"})
        meta = json.loads(
            (tmp_path / "experiments" / "checkpoints" / "unit" / "meta.json").read_text()
        )

        assert meta["num_rounds"] == 2
        assert meta["server_dp_enabled"] is False
        assert meta["epsilon"] is None, "DP off must report no epsilon"
        assert meta["stopped_early"] is False
        assert "tuned_threshold" in meta
        assert "metrics" in meta and "auc_roc" in meta["metrics"]
        assert "metrics_at_0.5" in meta

    def test_eps_tag_defaults_to_infinity_when_dp_off(self, run_server) -> None:
        _, tmp_path = run_server()
        assert (tmp_path / "experiments" / "checkpoints" / "eps_inf").is_dir()

    def test_no_eval_checkpoints_last_state(self, run_server) -> None:
        import json

        _, tmp_path = run_server(**{"num-server-rounds": 2, "eval-every": 100,
                                    "run-tag": "unit"})
        meta = json.loads(
            (tmp_path / "experiments" / "checkpoints" / "unit" / "meta.json").read_text()
        )
        assert meta["best_val_auc"] is None, "no eval ran, so there is no best"
        assert meta["best_round"] == 0

    def test_checkpoints_best_not_last_state(self, server_module, run_server,
                                             monkeypatch) -> None:
        """THE checkpoint invariant. val/auc peaks at round 2 then degrades.
        The saved model and the reported metric must come from round 2 — a run
        that collapses late must not be reported at its collapsed state."""
        import json

        states: list[dict] = []
        monkeypatch.setattr(
            server_module, "_run_eval",
            _fake_eval([0.60, 0.90, 0.55, 0.50], states=states),
        )
        _, tmp_path = run_server(
            **{"num-server-rounds": 4, "eval-every": 1,
               "early-stop-patience": 99, "run-tag": "unit"}
        )

        ckpt_dir = tmp_path / "experiments" / "checkpoints" / "unit"
        meta = json.loads((ckpt_dir / "meta.json").read_text())
        assert meta["best_val_auc"] == pytest.approx(0.90)
        assert meta["best_round"] == 2

        # The bookkeeping above is not enough — check the WEIGHTS on disk are
        # the round-2 ones, and that they are genuinely not the final state.
        saved = torch.load(ckpt_dir / "top.pt", weights_only=True)
        best_state, last_state = states[1], states[-1]
        assert set(saved) == set(best_state)
        for key in saved:
            assert torch.allclose(saved[key], best_state[key]), (
                f"{key} does not match the round-2 (best) weights"
            )
        assert any(
            not torch.allclose(best_state[k], last_state[k]) for k in saved
        ), "weights never changed after the best round — test cannot discriminate"

    def test_save_best_flag_rides_the_improving_round(self, server_module,
                                                      run_server, monkeypatch) -> None:
        """Clients snapshot their bottom models when they see save-best, so the
        flag must be set on exactly the rounds whose val/auc improved."""
        monkeypatch.setattr(
            server_module, "_run_eval", _fake_eval([0.60, 0.90, 0.55, 0.50]),
        )
        grid, _ = run_server(
            **{"num-server-rounds": 4, "eval-every": 1, "early-stop-patience": 99}
        )

        flags = {
            m.content["config"]["server-round"]: m.content["config"]["save-best"]
            for m in grid.gradient_pushes()
            if m.dst_node_id == 11
        }
        assert flags == {1: True, 2: True, 3: False, 4: False}

    def test_restore_best_rides_the_final_test_query(self, run_server) -> None:
        """The final test eval must carry restore-best so clients restore and
        checkpoint their peak bottoms in the same round-trip."""
        grid, _ = run_server()

        eval_queries = grid.queries_of_type("query.generate_eval_embeddings")
        test_queries = [
            q for q in eval_queries if q.content["config"]["split"] == "test"
        ]
        assert test_queries, "no test-split eval was run"
        for q in test_queries:
            assert q.content["config"]["restore-best"] is True


class TestEarlyStopping:
    """Patience-based early stop on stalled val/auc."""

    def test_stops_after_patience_without_improvement(self, server_module,
                                                      run_server, monkeypatch) -> None:
        monkeypatch.setattr(
            server_module, "_run_eval",
            _fake_eval([0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20]),
        )
        grid, _ = run_server(
            **{"num-server-rounds": 7, "eval-every": 1, "early-stop-patience": 2}
        )

        trained_rounds = {
            m.content["config"]["server-round"] for m in grid.gradient_pushes()
        }
        assert max(trained_rounds) == 3, (
            f"should stop at round 3 (first eval sets best, then 2 stalls), "
            f"trained through {max(trained_rounds)}"
        )

    def test_improving_run_is_not_stopped(self, server_module, run_server,
                                          monkeypatch) -> None:
        monkeypatch.setattr(
            server_module, "_run_eval",
            _fake_eval([0.50, 0.60, 0.70, 0.80]),
        )
        grid, _ = run_server(
            **{"num-server-rounds": 4, "eval-every": 1, "early-stop-patience": 2}
        )
        trained = {m.content["config"]["server-round"] for m in grid.gradient_pushes()}
        assert trained == {1, 2, 3, 4}

    def test_meta_flags_early_stop(self, server_module, run_server,
                                   monkeypatch) -> None:
        import json

        monkeypatch.setattr(
            server_module, "_run_eval", _fake_eval([0.80, 0.70, 0.60, 0.50]),
        )
        _, tmp_path = run_server(
            **{"num-server-rounds": 9, "eval-every": 1,
               "early-stop-patience": 2, "run-tag": "unit"}
        )
        meta = json.loads(
            (tmp_path / "experiments" / "checkpoints" / "unit" / "meta.json").read_text()
        )
        assert meta["stopped_early"] is True


@pytest.mark.dp
class TestPrivacyGating:
    """Budget gates must halt the run rather than keep spending."""

    def test_client_exhaustion_stops_the_run(self, run_server) -> None:
        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)
        grid.exhaust_after = 2  # rounds 1-2 fine, round 3 reports exhausted

        run_server(grid=grid, **{"num-server-rounds": 10})

        assert grid.stop_pushes(), "no stop-run message was sent"
        trained = {m.content["config"]["server-round"] for m in grid.gradient_pushes()}
        assert max(trained) == 2, f"kept training after exhaustion: {trained}"

    def test_exhaustion_stop_carries_eps_tag(self, run_server) -> None:
        """Clients need the tag to write their checkpoint to the right dir."""
        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)
        grid.exhaust_after = 1

        run_server(grid=grid, **{"num-server-rounds": 5, "run-tag": "unit"})
        for msg in grid.stop_pushes():
            assert msg.content["config"]["eps-tag"] == "unit"

    def test_combined_epsilon_gate_stops_the_run(self, run_server) -> None:
        """The composed (all-sources) epsilon is the real guarantee. Once the
        NEXT round would breach combined-target-epsilon, the run must stop."""
        prims = {
            "target_epsilon": 100.0, "sigma": 0.5,
            "sample_rate": 1.0, "delta": 1e-5, "steps": 20,
        }
        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)
        grid.privacy_by_pos = {SILO_A_POS: prims, SILO_B_POS: prims}

        run_server(grid=grid, **{"num-server-rounds": 6,
                                 "combined-target-epsilon": 0.001})

        assert grid.stop_pushes(), "combined-epsilon gate never fired"
        trained = {m.content["config"]["server-round"] for m in grid.gradient_pushes()}
        assert max(trained) == 1, f"gate fired too late: {trained}"

    def test_generous_combined_budget_does_not_stop(self, run_server) -> None:
        prims = {
            "target_epsilon": 100.0, "sigma": 8.0,
            "sample_rate": 0.1, "delta": 1e-5, "steps": 1,
        }
        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)
        grid.privacy_by_pos = {SILO_A_POS: prims, SILO_B_POS: prims}

        run_server(grid=grid, **{"num-server-rounds": 3,
                                 "combined-target-epsilon": 1000.0})
        assert not grid.stop_pushes()
        trained = {m.content["config"]["server-round"] for m in grid.gradient_pushes()}
        assert trained == {1, 2, 3}


@pytest.mark.dp
class TestServerDpSgd:
    """The Opacus DP-SGD path on the top model."""

    def test_dp_run_completes_and_reports_epsilon(self, run_server) -> None:
        import json

        _, tmp_path = run_server(
            **{
                "num-server-rounds": 2,
                "server-dp-enabled": True,
                "server-dp-epsilon": 5.0,
                "combined-target-epsilon": 1000.0,
                "run-tag": "dp",
            }
        )
        meta = json.loads(
            (tmp_path / "experiments" / "checkpoints" / "dp" / "meta.json").read_text()
        )
        assert meta["server_dp_enabled"] is True
        assert meta["epsilon"] == 5.0
        assert meta["epsilon_spent"]["privacy/server_epsilon"] > 0, (
            "DP was enabled but no budget was recorded as spent"
        )

    def test_dp_still_returns_input_gradients(self, run_server) -> None:
        """Opacus wraps the top model; if the wrapper stopped exposing input
        grads the silos would get nothing back. server_app raises on this, so
        reaching the push at all is the assertion."""
        grid, _ = run_server(
            **{"num-server-rounds": 2, "server-dp-enabled": True,
               "server-dp-epsilon": 5.0, "combined-target-epsilon": 1000.0}
        )
        pushes = grid.gradient_pushes()
        assert len(pushes) == 4
        for msg in pushes:
            arr = msg.content["gradients"]["local-gradients"].numpy()
            assert arr.shape == (8, EMB_DIM)
            assert abs(arr).sum() > 0

    def test_dp_checkpoint_is_unwrapped(self, run_server) -> None:
        """Saved weights must be clean TopModel keys, not Opacus's
        `_module.`-prefixed ones, or the checkpoint cannot be reloaded."""
        _, tmp_path = run_server(
            **{"num-server-rounds": 2, "server-dp-enabled": True,
               "server-dp-epsilon": 5.0, "combined-target-epsilon": 1000.0,
               "run-tag": "dp"}
        )
        state = torch.load(
            tmp_path / "experiments" / "checkpoints" / "dp" / "top.pt",
            weights_only=True,
        )
        assert state, "empty checkpoint"
        assert not any(k.startswith("_module.") for k in state), list(state)

        from fl_server.top_model import TopModel

        TopModel(embedding_dim=EMB_DIM, hidden_dim=8).load_state_dict(state)


class TestWandbLogging:
    """What the run reports to wandb (the paper's figures come from these)."""

    def test_logs_loss_and_privacy_per_round(self, server_module, run_server,
                                             monkeypatch) -> None:
        logged = []

        monkeypatch.setattr(server_module.wandb, "init", lambda **kw: None)
        monkeypatch.setattr(
            server_module.wandb, "log",
            lambda d, step=None, **kw: logged.append((step, d)),
        )
        monkeypatch.setattr(server_module.wandb, "finish", lambda **kw: None)

        run_server(**{"num-server-rounds": 2, "track-wandb": True})

        per_round = [d for step, d in logged if step is not None]
        assert len(per_round) >= 2
        assert all("train/loss" in d for d in per_round)
        assert all("train/lr" in d for d in per_round)
        assert all("train/pred_std" in d for d in per_round), (
            "prediction-collapse health metric missing"
        )

    def test_final_summary_includes_test_metrics(self, server_module, run_server,
                                                 monkeypatch) -> None:
        logged = []

        monkeypatch.setattr(server_module.wandb, "init", lambda **kw: None)
        monkeypatch.setattr(
            server_module.wandb, "log",
            lambda d, step=None, **kw: logged.append((step, d)),
        )
        monkeypatch.setattr(server_module.wandb, "finish", lambda **kw: None)

        run_server(**{"num-server-rounds": 2, "track-wandb": True})

        summary = [d for step, d in logged if step is None]
        assert summary, "no final summary was logged"
        final = summary[-1]
        assert any(k.startswith("test/") for k in final)
        assert any(k.startswith("test_at_0.5/") for k in final), (
            "the naive-threshold comparison must stay visible"
        )


@pytest.fixture
def wandb_spy(server_module, monkeypatch):
    """Capture wandb.init kwargs and every wandb.log payload."""
    captured = {"init": None, "logs": []}

    def _init(**kwargs):
        captured["init"] = kwargs

    monkeypatch.setattr(server_module.wandb, "init", _init)
    monkeypatch.setattr(
        server_module.wandb, "log",
        lambda d, step=None, **kw: captured["logs"].append((step, d)),
    )
    monkeypatch.setattr(server_module.wandb, "finish", lambda **kw: None)
    return captured


@pytest.fixture
def log_spy(server_module, monkeypatch):
    """Capture fl_server.server_app log() calls as rendered strings."""
    records: list[str] = []

    def _log(level, msg, *args):
        try:
            records.append(str(msg) % args if args else str(msg))
        except TypeError:
            records.append(str(msg))

    monkeypatch.setattr(server_module, "log", _log)
    return records


@pytest.mark.dp
class TestServerBudgetGate:
    """The server's own would_exceed() gate, checked before each round's step."""

    def test_stops_before_spending_over_budget(self, server_module, run_server,
                                               monkeypatch) -> None:
        """With an undersized sigma the very first step already breaches the
        target, so the run must stop WITHOUT training — a gate that fired after
        the step would have already leaked the budget it was protecting."""
        monkeypatch.setattr(server_module, "solve_noise_multiplier",
                            lambda **kwargs: 0.05)

        grid, _ = run_server(
            **{"num-server-rounds": 5, "server-dp-enabled": True,
               "server-dp-epsilon": 0.01, "batch-sample-rate": 1.0}
        )

        assert grid.stop_pushes(), "server budget gate never fired"
        assert not grid.gradient_pushes(), (
            "took a training step despite being over budget"
        )

    def test_gate_does_not_fire_with_adequate_budget(self, run_server) -> None:
        grid, _ = run_server(
            **{"num-server-rounds": 2, "server-dp-enabled": True,
               "server-dp-epsilon": 5.0, "combined-target-epsilon": 1000.0}
        )
        assert not grid.stop_pushes()
        assert len(grid.gradient_pushes()) == 4


class TestGradientClipping:
    """grad-clip-norm applies plain clipping — but only when DP is off."""

    def test_clip_applied_when_configured(self, run_server, monkeypatch) -> None:
        import torch.nn.utils as nn_utils

        calls = []
        real = nn_utils.clip_grad_norm_
        monkeypatch.setattr(
            nn_utils, "clip_grad_norm_",
            lambda params, max_norm, *a, **k: calls.append(max_norm)
            or real(params, max_norm, *a, **k),
        )

        run_server(**{"num-server-rounds": 2, "grad-clip-norm": 0.7})
        assert calls == [0.7, 0.7], f"expected one clip per round, got {calls}"

    def test_no_clip_when_disabled(self, run_server, monkeypatch) -> None:
        import torch.nn.utils as nn_utils

        calls = []
        monkeypatch.setattr(
            nn_utils, "clip_grad_norm_",
            lambda params, max_norm, *a, **k: calls.append(max_norm),
        )
        run_server(**{"num-server-rounds": 2, "grad-clip-norm": 0.0})
        assert calls == []

    @pytest.mark.dp
    def test_dp_path_does_not_double_clip(self, run_server, monkeypatch) -> None:
        """Opacus already clips per-sample gradients. An extra global clip on
        top would shrink the gradients the DP noise was calibrated against,
        silently weakening the privacy/utility trade-off being reported."""
        import torch.nn.utils as nn_utils

        calls = []
        real = nn_utils.clip_grad_norm_
        monkeypatch.setattr(
            nn_utils, "clip_grad_norm_",
            lambda params, max_norm, *a, **k: calls.append(max_norm)
            or real(params, max_norm, *a, **k),
        )

        run_server(
            **{"num-server-rounds": 2, "grad-clip-norm": 0.7,
               "server-dp-enabled": True, "server-dp-epsilon": 5.0,
               "combined-target-epsilon": 1000.0}
        )
        assert calls == [], "clipped on top of Opacus's own per-sample clipping"


class TestCollapseWarning:
    """A one-class val prediction must be shouted about, not buried."""

    def test_warns_when_predictions_collapse(self, server_module, run_server,
                                             log_spy, monkeypatch) -> None:
        def collapsed_eval(grid, node_ids, top_model, split,
                           extra_cfg=None, threshold=0.5):
            return {
                "auc_roc": 0.5, "pred_collapsed": 1.0, "pred_std": 0.0,
                "pred_pos_rate": 0.0, "base_rate": 0.375,
            }

        monkeypatch.setattr(server_module, "_run_eval", collapsed_eval)
        run_server(**{"num-server-rounds": 2, "eval-every": 1,
                      "early-stop-patience": 99})

        assert any("COLLAPSED" in r for r in log_spy), (
            "prediction collapse was not reported"
        )

    def test_silent_when_predictions_are_healthy(self, server_module, run_server,
                                                 log_spy, monkeypatch) -> None:
        monkeypatch.setattr(server_module, "_run_eval", _fake_eval([0.7, 0.8]))
        run_server(**{"num-server-rounds": 2, "eval-every": 1,
                      "early-stop-patience": 99})
        assert not any("COLLAPSED" in r for r in log_spy)


class TestLrPlateau:
    """Plateau-triggered LR reduction after stalled eval rounds."""

    def test_lr_drops_after_stalled_evals(self, server_module, run_server,
                                          monkeypatch) -> None:
        """The reduced LR must reach the clients, not just the server's own
        optimizer — the silos mirror it to update their bottom models."""
        monkeypatch.setattr(
            server_module, "_run_eval",
            _fake_eval([0.90, 0.50, 0.40, 0.30]),
        )
        grid, _ = run_server(
            **{"num-server-rounds": 4, "eval-every": 1, "learning-rate": 1.0,
               "early-stop-patience": 99, "lr-plateau-patience": 1,
               "lr-plateau-factor": 0.5}
        )

        lrs = [
            m.content["config"]["lr"]
            for m in grid.gradient_pushes() if m.dst_node_id == 11
        ]
        assert lrs[0] == pytest.approx(1.0)
        assert lrs[-1] < lrs[0], f"LR never reduced on plateau: {lrs}"

    def test_lr_floor_is_respected(self, server_module, run_server,
                                   monkeypatch) -> None:
        monkeypatch.setattr(
            server_module, "_run_eval",
            _fake_eval([0.9] + [0.1] * 10),
        )
        grid, _ = run_server(
            **{"num-server-rounds": 8, "eval-every": 1, "learning-rate": 1.0,
               "early-stop-patience": 99, "lr-plateau-patience": 1,
               "lr-plateau-factor": 0.01, "lr-min": 0.05}
        )
        lrs = [
            m.content["config"]["lr"]
            for m in grid.gradient_pushes() if m.dst_node_id == 11
        ]
        assert min(lrs) >= 0.05, f"LR fell below lr-min: {lrs}"


class TestWandbEvalLogging:
    """Eval-round and summary logging (the paper's per-round curves)."""

    def test_val_metrics_logged_on_eval_rounds(self, server_module, run_server,
                                               wandb_spy, monkeypatch) -> None:
        monkeypatch.setattr(server_module, "_run_eval", _fake_eval([0.7, 0.8]))
        run_server(**{"num-server-rounds": 2, "eval-every": 1,
                      "track-wandb": True, "early-stop-patience": 99})

        val_logs = [d for _, d in wandb_spy["logs"] if "val/auc" in d]
        assert len(val_logs) == 2
        assert val_logs[0]["val/auc"] == pytest.approx(0.7)

    def test_summary_carries_best_val_auc(self, server_module, run_server,
                                          wandb_spy, monkeypatch) -> None:
        monkeypatch.setattr(server_module, "_run_eval", _fake_eval([0.7, 0.95]))
        run_server(**{"num-server-rounds": 2, "eval-every": 1,
                      "track-wandb": True, "early-stop-patience": 99})

        summary = [d for step, d in wandb_spy["logs"] if step is None][-1]
        assert summary["val/auc_best"] == pytest.approx(0.95)

    def test_sweep_tag_is_attached(self, run_server, wandb_spy) -> None:
        run_server(**{"num-server-rounds": 1, "track-wandb": True,
                      "wandb-sweep-tag": "sweep-abc"})
        assert "sweep-abc" in wandb_spy["init"]["tags"]

    def test_default_entity_and_project(self, run_server, wandb_spy) -> None:
        """The wandb convention in CLAUDE.md: entity jayquery, project
        jayquery-vfl for VFL runs."""
        run_server(**{"num-server-rounds": 1, "track-wandb": True})
        assert wandb_spy["init"]["entity"] == "jayquery"
        assert wandb_spy["init"]["project"] == "jayquery-vfl"

    def test_nothing_logged_when_tracking_off(self, run_server, wandb_spy) -> None:
        run_server(**{"num-server-rounds": 2, "track-wandb": False})
        assert wandb_spy["logs"] == []
        assert wandb_spy["init"] is None


class TestInputGradientGuard:
    """The loop refuses to continue if the backward yields no input gradients."""

    def test_raises_when_input_grads_vanish(self, run_server, monkeypatch) -> None:
        """The silos learn ONLY from dL/d_embedding. If a wrapper (Opacus) ever
        stops exposing input grads, the run must die loudly rather than push
        empty gradients and silently train nothing."""
        from fl_server.top_model import TopModel

        def forward_ignoring_inputs(self, emb_a, emb_b):
            # A graph that never touches emb_a/emb_b, so their .grad stays None.
            return self.net[0].weight.sum().reshape(1, 1).expand(emb_a.shape[0], 1)

        monkeypatch.setattr(TopModel, "forward", forward_ignoring_inputs)

        with pytest.raises(RuntimeError, match="input grads are None"):
            run_server(**{"num-server-rounds": 1})


class TestEvalContentlessReply:
    """A dropped reply during eval must not be parsed as a payload."""

    def test_errored_eval_reply_raises(self, server_module) -> None:
        from fl_server.top_model import TopModel

        def responder(messages, call_index):
            return [
                emb_reply(pos=SILO_A_POS, src_node_id=11),
                emb_reply(pos=SILO_B_POS, src_node_id=22, error="dropped"),
            ]

        grid = FakeGrid(node_ids=[11, 22], responder=responder)
        with pytest.raises(RuntimeError, match="both silos"):
            server_module._collect_eval_predictions(
                grid, [11, 22], TopModel(embedding_dim=EMB_DIM, hidden_dim=8), "val"
            )


_LABEL_PRIMS = {
    "mechanism": "randomized response",
    "neighboring_relation": "label-adjacent (one student's label differs)",
    "epsilon": 2.0,
    "delta": 0.0,
    "keep_prob": 0.8807970779778823,  # sigmoid(2.0)
    "seed": 5,
    "steps": 0,
}


@pytest.mark.dp
class TestLabelChannelAtServer:
    """The server consumes the label channel without composing it away."""

    def test_get_embeddings_captures_label_primitives(self, server_module) -> None:
        def responder(messages, call_index):
            return [
                emb_reply(pos=SILO_A_POS, src_node_id=11, label_privacy=_LABEL_PRIMS),
                emb_reply(pos=SILO_B_POS, src_node_id=22),
            ]

        grid = FakeGrid(node_ids=[11, 22], responder=responder)
        rd = server_module._get_embeddings(
            grid=grid, node_ids=[11, 22], round_num=1, batch_seed=1, sample_rate=1.0
        )
        assert rd.label_privacy, "label primitives not collected"
        assert float(rd.label_privacy["epsilon"]) == pytest.approx(2.0)

    def test_server_applies_the_reported_keep_prob(self, server_module,
                                                   run_server, monkeypatch) -> None:
        """THE correction seam. The holder randomized its labels; if the server
        does not apply the matching keep_prob, it trains on noisy labels as if
        they were clean and calibrates to the corrupted base rate."""
        seen = {}
        real = server_module.TopModel.set_label_noise

        def spy(self, keep_prob):
            seen["keep_prob"] = keep_prob
            return real(self, keep_prob)

        monkeypatch.setattr(server_module.TopModel, "set_label_noise", spy)

        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)
        grid.label_privacy_prims = _LABEL_PRIMS
        run_server(grid=grid, **{"num-server-rounds": 2})

        assert seen.get("keep_prob") == pytest.approx(_LABEL_PRIMS["keep_prob"])

    def test_no_correction_when_label_dp_is_off(self, server_module, run_server,
                                                monkeypatch) -> None:
        calls = []
        monkeypatch.setattr(
            server_module.TopModel, "set_label_noise",
            lambda self, keep_prob: calls.append(keep_prob),
        )
        run_server(**{"num-server-rounds": 2})
        assert calls == [], "applied a label correction with label DP off"

    def test_label_epsilon_reported_separately(self, server_module) -> None:
        """Label epsilon appears as its own metric and is NOT folded into the
        combined Gaussian number — different mechanism, different relation."""
        metrics = server_module._epsilon_metrics(
            _tracker(steps=5),
            {SILO_A_POS: _tracker(steps=5)},
            delta=1e-5,
            label_primitives=_LABEL_PRIMS,
        )
        assert metrics["privacy/label_epsilon"] == pytest.approx(2.0)

        without = server_module._epsilon_metrics(
            _tracker(steps=5), {SILO_A_POS: _tracker(steps=5)}, delta=1e-5
        )
        assert metrics["privacy/combined_epsilon"] == pytest.approx(
            without["privacy/combined_epsilon"]
        ), "label epsilon leaked into the composed Gaussian epsilon"

    def test_meta_records_the_label_channel(self, run_server) -> None:
        import json

        grid = FakeGrid(node_ids=[11, 22])
        grid.labels = balanced_labels(n_rows=8, n_pos=3)
        grid.label_privacy_prims = _LABEL_PRIMS
        _, tmp_path = run_server(grid=grid, **{"num-server-rounds": 2,
                                               "run-tag": "lbl"})

        meta = json.loads(
            (tmp_path / "experiments" / "checkpoints" / "lbl" / "meta.json").read_text()
        )
        assert "label_privacy" in meta, "label channel missing from meta.json"
        assert float(meta["label_privacy"]["epsilon"]) == pytest.approx(2.0)
        assert float(meta["label_privacy"]["delta"]) == 0.0
        assert "label" in str(meta["label_privacy"]["neighboring_relation"]).lower()
