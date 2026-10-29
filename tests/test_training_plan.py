"""Tests for baselines/training_plan.py.

The baseline MLPs are the paper's architecture-matched controls, so their
optimization budget must match the VFL run's. These pin the mapping from
config keys to the budget actually used, so a key that stops being honored
fails a test rather than silently changing what the control measures.
"""

from __future__ import annotations

import pytest

N_TRAIN = 20411        # the dedup_first cohort's training split
VFL_STEPS = 200        # the ICC grid's horizon
VFL_FRACTION = 0.05    # batch-sample-rate


class TestStepBudget:
    """gradient_steps must drive the optimizer, not epochs."""

    def test_gradient_steps_is_honored(self) -> None:
        from baselines.training_plan import resolve_plan

        plan = resolve_plan({"gradient_steps": VFL_STEPS}, n_train=N_TRAIN)
        assert plan.steps == VFL_STEPS

    def test_matches_the_vfl_budget_when_given_the_grid_config(self) -> None:
        """The whole point: the control gets the same optimization as VFL."""
        from baselines.training_plan import resolve_plan

        plan = resolve_plan(
            {"gradient_steps": VFL_STEPS, "batch_fraction": VFL_FRACTION},
            n_train=N_TRAIN,
        )
        assert plan.steps == VFL_STEPS
        assert plan.batch_size == pytest.approx(round(VFL_FRACTION * N_TRAIN), abs=1)

    def test_epochs_still_work_when_no_step_budget_is_given(self) -> None:
        """Backward compatibility: callers that never set gradient_steps keep
        their old behavior rather than silently training for one step."""
        from baselines.training_plan import resolve_plan

        plan = resolve_plan({"epochs": 3, "batch_size": 64}, n_train=640)
        assert plan.steps == 3 * (640 // 64)
        assert plan.batch_size == 64

    def test_a_step_budget_wins_over_epochs(self) -> None:
        from baselines.training_plan import resolve_plan

        plan = resolve_plan(
            {"gradient_steps": 200, "epochs": 50, "batch_size": 64}, n_train=N_TRAIN
        )
        assert plan.steps == 200, "epochs must not reinstate the 80x gap"

    def test_rejects_a_nonsensical_budget(self) -> None:
        from baselines.training_plan import resolve_plan

        for bad in ({"gradient_steps": 0}, {"gradient_steps": -5}):
            with pytest.raises(ValueError, match="gradient_steps"):
                resolve_plan(bad, n_train=N_TRAIN)


class TestBatchSize:
    """batch_fraction mirrors the VFL Poisson sampling rate."""

    def test_fraction_converts_to_a_row_count(self) -> None:
        from baselines.training_plan import resolve_plan

        assert resolve_plan({"batch_fraction": 0.05}, n_train=1000).batch_size == 50

    def test_never_degenerates_to_an_empty_batch(self) -> None:
        from baselines.training_plan import resolve_plan

        assert resolve_plan({"batch_fraction": 1e-6}, n_train=10).batch_size == 1

    def test_is_capped_at_the_training_set(self) -> None:
        from baselines.training_plan import resolve_plan

        assert resolve_plan({"batch_fraction": 2.0}, n_train=100).batch_size == 100


class TestClassWeighting:
    """The VFL grid runs pos-weight=0, i.e. OFF. The control must match.

    Both MLPs hardcoded pos_weight = n_neg/n_pos, handing the baselines class
    balancing the federated model never had.
    """

    def test_none_means_no_weighting(self) -> None:
        from baselines.training_plan import resolve_plan

        assert resolve_plan({"class_weighting": "none"}, n_train=100).pos_weight is None

    def test_default_is_no_weighting(self) -> None:
        """Silence must mean 'match VFL', not 'reinstate the advantage'."""
        from baselines.training_plan import resolve_plan

        assert resolve_plan({}, n_train=100).pos_weight is None

    def test_balanced_computes_the_ratio(self) -> None:
        from baselines.training_plan import resolve_plan

        plan = resolve_plan(
            {"class_weighting": "balanced"}, n_train=100, n_pos=25, n_neg=75
        )
        assert plan.pos_weight == pytest.approx(3.0)

    def test_balanced_needs_the_counts(self) -> None:
        from baselines.training_plan import resolve_plan

        with pytest.raises(ValueError, match="class_weighting"):
            resolve_plan({"class_weighting": "balanced"}, n_train=100)

    def test_rejects_an_unknown_policy(self) -> None:
        """A typo must not silently fall through to 'none' and quietly change
        what the control measures."""
        from baselines.training_plan import resolve_plan

        with pytest.raises(ValueError, match="class_weighting"):
            resolve_plan({"class_weighting": "balancd"}, n_train=100, n_pos=1, n_neg=1)
