"""Tests for fl_client: SiloSpec, common.py's generic VFL logic, and the
client_app.py dispatcher.

Place this file alongside conftest.py (same directory), with fl_client/ and
your real silo_a/, silo_b/, utils/ packages importable from that same
directory (or installed/on PYTHONPATH).

Run with:  pytest test_fl_client.py -v

Layout:
    TestSiloSpecValidation   — unit: __post_init__ rejects bad specs
    TestCommonHelpers        — unit: individual common.py helper functions
    TestRoundTrip            — integration: generate_embeddings -> apply_gradients
    TestMultiRound           — integration: state persists correctly across rounds
    TestDispatcher           — client_app.py routes to the REAL silo_a/silo_b
                                handler modules (SPEC monkeypatched to a fake
                                spec so no real OULAD data is required; the
                                real BottomModelA/B classes are still imported
                                and type-checked by SiloSpec.__post_init__)
"""

import io

import pytest
import torch

from tests.fixtures.conftest import FakeBottomModel, make_fake_load_fn

# ── unit: SiloSpec validation ───────────────────────────────────────────────


class TestSiloSpecValidation:
    def test_valid_spec_constructs(self, fake_spec_factory):
        spec = fake_spec_factory()
        assert spec.partition_id == 0
        assert spec.wandb_tag == "silo_fake"

    def test_rejects_non_module_model_cls(self, fake_spec_factory):
        with pytest.raises(TypeError, match="model_cls"):
            fake_spec_factory(model_cls=int)

    def test_rejects_noncallable_model_kwargs_fn(self, fake_spec_factory):
        with pytest.raises(TypeError, match="model_kwargs_fn"):
            fake_spec_factory(model_kwargs_fn={"not": "callable"})

    def test_rejects_noncallable_load_fn(self, fake_spec_factory):
        with pytest.raises(TypeError, match="load_fn"):
            fake_spec_factory(load_fn="not callable")

    def test_rejects_non_int_partition_id(self, fake_spec_factory):
        with pytest.raises(TypeError, match="partition_id"):
            fake_spec_factory(partition_id="zero")

    def test_extra_wandb_config_fn_defaults_to_empty(self, fake_spec_factory):
        spec = fake_spec_factory()
        assert spec.extra_wandb_config_fn(None) == {}


# ── unit: common.py helper functions ────────────────────────────────────────


class TestCommonHelpers:
    def test_build_model_fresh_has_no_state_restore(self, common_module, fake_spec_factory, fresh_context):
        spec = fake_spec_factory()
        ctx = fresh_context()
        model = common_module._build_model(spec, ctx)
        assert isinstance(model, FakeBottomModel)

    def test_build_model_restores_persisted_weights(self, common_module, fake_spec_factory, fresh_context):
        from flwr.app import ArrayRecord

        spec = fake_spec_factory()
        ctx = fresh_context()
        model_a = common_module._build_model(spec, ctx)
        ctx.state["model"] = ArrayRecord(model_a.state_dict())

        model_b = common_module._build_model(spec, ctx)
        for k in model_a.state_dict():
            assert torch.allclose(model_a.state_dict()[k], model_b.state_dict()[k]), (
                f"restored weight {k} does not match persisted weight"
            )

    def test_full_trainset_concatenates_all_batches(self, common_module, fake_spec_factory, fresh_context):
        spec = fake_spec_factory(load_fn=make_fake_load_fn(n_rows=20, input_dim=5))
        ctx = fresh_context()
        X, y = common_module._full_trainset(spec, ctx)
        assert X.shape == (20, 5)
        assert y.shape == (20,)
        assert X.dtype == torch.float32

    def test_run_id_persists_across_calls(self, common_module, fake_spec_factory, fresh_context):
        ctx = fresh_context()
        rid1 = common_module._run_id(ctx)
        rid2 = common_module._run_id(ctx)
        assert rid1 == rid2, "run id must be stable across calls on the same context"

    def test_round_number_advances_when_requested(self, common_module, fresh_context):
        from flwr.app import Message, RecordDict

        ctx = fresh_context()
        msg = Message(content=RecordDict())  # no server-round -> use local counter
        r0 = common_module._round_number(ctx, msg, advance=True)
        r1 = common_module._round_number(ctx, msg, advance=True)
        r2 = common_module._round_number(ctx, msg, advance=False)
        assert (r0, r1, r2) == (0, 1, 1), f"expected (0, 1, 1), got {(r0, r1, r2)}"

    def test_round_number_prefers_server_provided_round(self, common_module, fresh_context):
        from flwr.app import ConfigRecord, Message, RecordDict

        ctx = fresh_context()
        msg = Message(content=RecordDict({"config": ConfigRecord({"server-round": 7})}))
        r = common_module._round_number(ctx, msg, advance=True)
        assert r == 7

    def test_wandb_config_merges_extra_fn(self, common_module, fake_spec_factory, fresh_context):
        spec = fake_spec_factory(extra_wandb_config_fn=lambda ctx: {"custom-key": 123})
        ctx = fresh_context()
        cfg = common_module._wandb_config(spec, ctx)
        assert cfg["silo"] == "silo_fake"
        assert cfg["custom-key"] == 123

    def test_log_noop_when_tracking_disabled(self, common_module, fake_spec_factory, fresh_context, wandb_calls):
        spec = fake_spec_factory()
        ctx = fresh_context(run_config={"track-wandb": False})
        common_module._log(spec, ctx, {"x": 1}, step=0)
        assert len(wandb_calls) == 0

    def test_log_calls_through_when_enabled(self, common_module, fake_spec_factory, fresh_context, wandb_calls):
        spec = fake_spec_factory()
        ctx = fresh_context(run_config={"track-wandb": True})
        common_module._log(spec, ctx, {"x": 1}, step=0)
        assert len(wandb_calls) == 1
        assert wandb_calls[0]["entries"] == [[0, {"x": 1}]]

    def test_log_buffers_until_interval_then_flushes(
        self, common_module, fake_spec_factory, fresh_context, wandb_calls
    ):
        """With interval=3, entries accumulate (2/round) and upload as one batch."""
        spec = fake_spec_factory()
        ctx = fresh_context(run_config={"wandb-log-interval": 3})

        for step in range(5):
            common_module._log(spec, ctx, {"x": step}, step=step)
        assert len(wandb_calls) == 0, "must not upload before 2*interval entries"

        common_module._log(spec, ctx, {"x": 5}, step=5)  # 6th entry = 2*3 -> flush
        assert len(wandb_calls) == 1
        assert [e[0] for e in wandb_calls[0]["entries"]] == [0, 1, 2, 3, 4, 5]

        common_module._log(spec, ctx, {"x": 6}, step=6)
        assert len(wandb_calls) == 1, "buffer must restart empty after a flush"

    def test_log_force_flush_uploads_partial_buffer(
        self, common_module, fake_spec_factory, fresh_context, wandb_calls
    ):
        spec = fake_spec_factory()
        ctx = fresh_context(run_config={"wandb-log-interval": 50})
        common_module._log(spec, ctx, {"x": 0}, step=0)
        common_module._log(spec, ctx, {"x": 1}, step=1, flush=True)
        assert len(wandb_calls) == 1
        assert len(wandb_calls[0]["entries"]) == 2


# ── integration: full generate_embeddings -> apply_gradients round trip ────


class TestRoundTrip:
    def test_silo_a_shaped_round_trip(self, common_module, fake_spec_factory, fresh_context):
        """sends_labels=True silo: embedding round-trips, labels are included,
        optimizer actually updates the weights."""
        from flwr.app import Array, ArrayRecord, Message, RecordDict

        spec = fake_spec_factory(sends_labels=True, partition_id=0)
        ctx = fresh_context()

        reply = common_module.generate_embeddings(spec, Message(), ctx)
        arrays = reply.content["arrays"]
        assert "embedding" in arrays
        assert "labels" in arrays
        assert reply.content["config"]["pos"] == 0

        emb_sent = arrays["embedding"].numpy()
        weights_before = {k: v.clone() for k, v in ctx.state["model"].to_torch_state_dict().items()}

        grad = Array(emb_sent.copy() * 0.1)  # arbitrary nonzero upstream gradient
        grad_msg = Message(content=RecordDict({"gradients": ArrayRecord({"local-gradients": grad})}))
        common_module.apply_gradients(spec, grad_msg, ctx)  # must not raise

        weights_after = ctx.state["model"].to_torch_state_dict()
        changed = any(
            not torch.allclose(weights_before[k], weights_after[k]) for k in weights_before
        )
        assert changed, "optimizer.step() should have changed at least one weight"

    def test_silo_b_shaped_round_trip_excludes_labels(self, common_module, fake_spec_factory, fresh_context):
        """sends_labels=False silo: embedding round-trips, NO labels in the reply."""
        from flwr.app import Array, ArrayRecord, Message, RecordDict

        spec = fake_spec_factory(sends_labels=False, partition_id=1)
        ctx = fresh_context()

        reply = common_module.generate_embeddings(spec, Message(), ctx)
        arrays = reply.content["arrays"]
        assert "embedding" in arrays
        assert "labels" not in arrays
        assert reply.content["config"]["pos"] == 1

        emb_sent = arrays["embedding"].numpy()
        grad = Array(emb_sent.copy() * 0.1)
        grad_msg = Message(content=RecordDict({"gradients": ArrayRecord({"local-gradients": grad})}))
        common_module.apply_gradients(spec, grad_msg, ctx)  # must not raise

    def test_detects_embedding_desync(self, common_module, fake_spec_factory, fresh_context):
        """If the persisted model state is tampered with between
        generate_embeddings and apply_gradients (simulating an RNG/data-order
        regression), apply_gradients must raise rather than silently apply a
        misattributed gradient."""
        from flwr.app import Array, ArrayRecord, Message, RecordDict

        spec = fake_spec_factory()
        ctx = fresh_context()

        reply = common_module.generate_embeddings(spec, Message(), ctx)
        emb_sent = reply.content["arrays"]["embedding"].numpy()

        tampered = {k: v.numpy() + 5.0 for k, v in ctx.state["model"].items()}
        ctx.state["model"] = ArrayRecord({k: Array(v) for k, v in tampered.items()})

        grad = Array(emb_sent.copy() * 0.1)
        grad_msg = Message(content=RecordDict({"gradients": ArrayRecord({"local-gradients": grad})}))
        with pytest.raises(RuntimeError, match="Recomputed embedding"):
            common_module.apply_gradients(spec, grad_msg, ctx)

    def test_round_trip_logs_expected_metric_keys(self, common_module, fake_spec_factory, fresh_context, wandb_calls):
        from flwr.app import Array, ArrayRecord, Message, RecordDict

        spec = fake_spec_factory(wandb_tag="silo_check")
        ctx = fresh_context()
        reply = common_module.generate_embeddings(spec, Message(), ctx)
        emb_sent = reply.content["arrays"]["embedding"].numpy()
        grad_msg = Message(
            content=RecordDict(
                {"gradients": ArrayRecord({"local-gradients": Array(emb_sent.copy() * 0.1)})}
            )
        )
        common_module.apply_gradients(spec, grad_msg, ctx)

        assert len(wandb_calls) == 2  # one from generate_embeddings, one from apply_gradients
        gen_metrics, apply_metrics = (c["entries"][0][1] for c in wandb_calls)
        assert "silo_check/embedding_l2" in gen_metrics
        assert "silo_check/grad_l2" in apply_metrics


# ── integration: state persists correctly across multiple rounds ──────────


class TestMultiRound:
    def test_two_rounds_advance_and_mutate_weights(self, common_module, fake_spec_factory, fresh_context):
        from flwr.app import Array, ArrayRecord, Message, RecordDict

        spec = fake_spec_factory()
        ctx = fresh_context()

        def run_one_round():
            reply = common_module.generate_embeddings(spec, Message(), ctx)
            emb_sent = reply.content["arrays"]["embedding"].numpy()
            grad_msg = Message(
                content=RecordDict(
                    {"gradients": ArrayRecord({"local-gradients": Array(emb_sent.copy() * 0.1)})}
                )
            )
            common_module.apply_gradients(spec, grad_msg, ctx)
            return {k: v.clone() for k, v in ctx.state["model"].to_torch_state_dict().items()}

        weights_r1 = run_one_round()
        weights_r2 = run_one_round()

        changed = any(not torch.allclose(weights_r1[k], weights_r2[k]) for k in weights_r1)
        assert changed, "weights should keep evolving across rounds"

        # round counter should have advanced by 1 across the two full rounds
        assert ctx.state["round"]["v"] == 1

    def test_optimizer_state_survives_round_boundary(self, common_module, fake_spec_factory, fresh_context):
        """Confirms optimizer_state is actually persisted+restored (not
        silently re-initialized each round) by checking Adam's step count."""
        from flwr.app import Array, ArrayRecord, Message, RecordDict

        spec = fake_spec_factory()
        ctx = fresh_context()

        for _ in range(2):
            reply = common_module.generate_embeddings(spec, Message(), ctx)
            emb_sent = reply.content["arrays"]["embedding"].numpy()
            grad_msg = Message(
                content=RecordDict(
                    {"gradients": ArrayRecord({"local-gradients": Array(emb_sent.copy() * 0.1)})}
                )
            )
            common_module.apply_gradients(spec, grad_msg, ctx)

        assert "optimizer_state" in ctx.state
        restored = torch.load(
            io.BytesIO(ctx.state["optimizer_state"]["serialized"]), weights_only=False
        )
        step_counts = [s["step"] for s in restored["state"].values() if "step" in s]
        assert step_counts and all(s >= 1 for s in step_counts)


# ── mini-batching: shared-seed Poisson sampling + budget gating ────────────


def _batch_msg(seed=7, q=0.5, rnd=1):
    from flwr.app import ConfigRecord, Message, RecordDict

    return Message(
        content=RecordDict({"config": ConfigRecord({
            "server-round": rnd, "batch-seed": seed, "batch-sample-rate": q,
        })})
    )


class TestMiniBatching:
    def test_batch_indices_identical_across_silos(self, common_module):
        """Same seed + rate -> identical row selection (the cross-silo alignment invariant)."""
        cfg = {"batch-seed": 3, "batch-sample-rate": 0.3}
        idx1, q1 = common_module._batch_indices(200, cfg)
        idx2, _ = common_module._batch_indices(200, cfg)
        assert torch.equal(idx1, idx2)
        assert q1 == 0.3
        assert 0 < idx1.numel() < 200

        idx3, _ = common_module._batch_indices(200, {"batch-seed": 4, "batch-sample-rate": 0.3})
        assert not torch.equal(idx1, idx3), "different seeds must give different batches"

    def test_batch_indices_full_batch_without_config(self, common_module):
        idx, q = common_module._batch_indices(10, None)
        assert q == 1.0
        assert torch.equal(idx, torch.arange(10))

    def test_batch_indices_requires_seed_when_sampling(self, common_module):
        import pytest as _pytest

        with _pytest.raises(ValueError, match="batch-seed"):
            common_module._batch_indices(10, {"batch-sample-rate": 0.5})

    def test_minibatch_round_trip(self, common_module, fake_spec_factory, fresh_context):
        """A sampled round returns batch-sized embedding+labels and replays cleanly."""
        from flwr.app import Array, ArrayRecord, Message, RecordDict

        from tests.fixtures.conftest import make_fake_load_fn

        spec = fake_spec_factory(load_fn=make_fake_load_fn(n_rows=40), sends_labels=True)
        ctx = fresh_context()

        reply = common_module.generate_embeddings(spec, _batch_msg(seed=11, q=0.5), ctx)
        emb = reply.content["arrays"]["embedding"].numpy()
        labels = reply.content["arrays"]["labels"].numpy()
        assert 0 < emb.shape[0] < 40, "Poisson batch should be a strict subset"
        assert labels.shape[0] == emb.shape[0]
        assert reply.content["config"]["n-total"] == 40

        grad_msg = Message(
            content=RecordDict({"gradients": ArrayRecord({"local-gradients": Array(emb * 0.1)})})
        )
        common_module.apply_gradients(spec, grad_msg, ctx)  # replay on the same subset

    def test_budget_exhausted_reply_carries_flag_not_embedding(
        self, common_module, fake_spec_factory, fresh_context
    ):
        from privacy.privacy_accountant import PrivacyBudgetTracker

        spec = fake_spec_factory()
        ctx = fresh_context(run_config={"cut-noise-enabled": True, "num-server-rounds": 1})
        spent = PrivacyBudgetTracker(target_epsilon=0.5, sigma=1.0, sample_rate=1.0)
        spent.steps = 100
        common_module._save_tracker(ctx, spent)

        reply = common_module.generate_embeddings(spec, _batch_msg(q=1.0), ctx)
        assert reply.content["config"]["budget-exhausted"] is True
        assert "arrays" not in reply.content
        assert reply.content["privacy"]["steps"] == 100

    def test_cut_noise_reports_privacy_primitives_and_perturbs(
        self, common_module, fake_spec_factory, fresh_context
    ):
        spec = fake_spec_factory()
        ctx = fresh_context(
            run_config={"cut-noise-enabled": True, "cut-noise-target-epsilon": 1.0,
                        "num-server-rounds": 5}
        )
        reply = common_module.generate_embeddings(spec, _batch_msg(q=1.0), ctx)

        privacy = reply.content["privacy"]
        assert privacy["steps"] == 1
        assert privacy["sigma"] > 0

        clean = ctx.state["sent_embedding"]["v"].numpy()
        sent = reply.content["arrays"]["embedding"].numpy()
        assert sent.shape == clean.shape
        assert not torch.allclose(torch.from_numpy(sent), torch.from_numpy(clean)), (
            "sent embedding must be clipped+noised, stored embedding must stay clean"
        )

    def test_gradient_backprops_through_clip(
        self, common_module, fake_spec_factory, fresh_context
    ):
        """With cut noise on, apply_gradients trains through the clip without desync."""
        from flwr.app import Array, ArrayRecord, Message, RecordDict

        spec = fake_spec_factory()
        ctx = fresh_context(run_config={"cut-noise-enabled": True, "num-server-rounds": 5})

        reply = common_module.generate_embeddings(spec, _batch_msg(q=1.0), ctx)
        emb = reply.content["arrays"]["embedding"].numpy()
        before = {k: v.clone() for k, v in ctx.state["model"].to_torch_state_dict().items()}

        grad_msg = Message(
            content=RecordDict({"gradients": ArrayRecord({"local-gradients": Array(emb * 0.1)})})
        )
        common_module.apply_gradients(spec, grad_msg, ctx)

        after = ctx.state["model"].to_torch_state_dict()
        assert any(not torch.allclose(before[k], after[k]) for k in before), (
            "gradient must reach the bottom model through the clip"
        )

    def test_eval_embeddings_clipped_when_cut_noise_enabled(
        self, common_module, fake_spec_factory, fresh_context
    ):
        """Eval embeddings match the training distribution: clipped, not noised."""
        from flwr.app import ConfigRecord, Message, RecordDict

        from tests.fixtures.conftest import make_fake_load_fn

        base = make_fake_load_fn(n_rows=20)

        def load3(batch_size=64):
            loader, _, _ = base(batch_size)
            return loader, loader, loader

        spec = fake_spec_factory(load_fn=load3)
        ctx = fresh_context(
            run_config={"cut-noise-enabled": True, "cut-noise-max-norm": 0.5,
                        "num-server-rounds": 5}
        )
        msg = Message(content=RecordDict({"config": ConfigRecord({"split": "val"})}))
        reply = common_module.generate_eval_embeddings(spec, msg, ctx)
        emb = torch.from_numpy(reply.content["arrays"]["embedding"].numpy())
        assert (emb.norm(p=2, dim=1) <= 0.5 + 1e-5).all(), "eval rows must respect the clip bound"

    def test_stop_run_checkpoints_without_training(
        self, common_module, fake_spec_factory, fresh_context, tmp_path, monkeypatch
    ):
        from flwr.app import ConfigRecord, Message, RecordDict

        monkeypatch.chdir(tmp_path)
        spec = fake_spec_factory(wandb_tag="silo_stop")
        ctx = fresh_context()

        stop_msg = Message(
            content=RecordDict({"config": ConfigRecord({"stop-run": True, "eps-tag": "eps_t"})})
        )
        common_module.apply_gradients(spec, stop_msg, ctx)
        assert (tmp_path / "experiments/checkpoints/eps_t/silo_stop.pt").exists()
        assert "optimizer_state" not in ctx.state, "stop-run must not train"


# ── dispatcher: client_app.py routes to the REAL silo_a/silo_b handlers ────


class TestDispatcher:
    @pytest.fixture(autouse=True)
    def _patch_real_silo_modules(self, monkeypatch, fake_spec_factory):
        """fl_client.silo_a.handlers / silo_b.handlers import real
        BottomModelA/B + real data loaders from your project. We monkeypatch
        each handler module's SPEC to a fake-but-shaped spec so these tests
        run on small synthetic data instead of real OULAD pickles. The REAL
        handler wrapper functions (get_emb/apply_grad) and the REAL dispatcher
        logic in client_app.py run unmodified — only SPEC's data is swapped."""
        import fl_client.silo_a.handlers as silo_a_handlers
        import fl_client.silo_b.handlers as silo_b_handlers

        monkeypatch.setattr(
            silo_a_handlers, "SPEC", fake_spec_factory(partition_id=0, sends_labels=True, wandb_tag="silo_a")
        )
        monkeypatch.setattr(
            silo_b_handlers, "SPEC", fake_spec_factory(partition_id=1, sends_labels=False, wandb_tag="silo_b")
        )

    def test_partition_0_routes_to_silo_a(self, fresh_context):
        from flwr.app import Message

        import fl_client.client_app as client_app

        ctx = fresh_context(node_config={"partition-id": 0})
        reply = client_app.get_emb(Message(), ctx)
        assert "labels" in reply.content["arrays"], "partition 0 should hit silo_a (sends_labels=True)"

    def test_partition_1_routes_to_silo_b(self, fresh_context):
        from flwr.app import Message

        import fl_client.client_app as client_app

        ctx = fresh_context(node_config={"partition-id": 1})
        reply = client_app.get_emb(Message(), ctx)
        assert "labels" not in reply.content["arrays"], "partition 1 should hit silo_b (sends_labels=False)"

    def test_unmapped_partition_raises_keyerror(self, fresh_context):
        from flwr.app import Message

        import fl_client.client_app as client_app

        ctx = fresh_context(node_config={"partition-id": 7})
        with pytest.raises(KeyError, match="No handler registered"):
            client_app.get_emb(Message(), ctx)

    def test_dispatcher_full_round_trip_both_silos(self, fresh_context):
        """End-to-end: route both silos' generate_embeddings + apply_gradients
        through the real dispatcher, exactly as the ServerApp would."""
        from flwr.app import Array, ArrayRecord, Message, RecordDict

        import fl_client.client_app as client_app

        for pid in (0, 1):
            ctx = fresh_context(node_config={"partition-id": pid})
            reply = client_app.get_emb(Message(), ctx)
            emb_sent = reply.content["arrays"]["embedding"].numpy()
            grad_msg = Message(
                content=RecordDict(
                    {"gradients": ArrayRecord({"local-gradients": Array(emb_sent.copy() * 0.1)})}
                )
            )
            client_app.apply_grad(grad_msg, ctx)  # must not raise for either silo