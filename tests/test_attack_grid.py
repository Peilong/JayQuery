"""Tests for scripts/run_attack_grid.py, the per-adversary attack driver.

The driver replays two attacks against every checkpoint and labels each result
with the arm and epsilon it belongs to. These cover the failures that would be
silent rather than loud:

  * mislabelling an arm, which misplaces a number in the attack figure;
  * calibrating to the configured noise instead of the realized noise, which
    mis-sizes the attack for any run that stopped early;
  * clipping or noising the undefended arms, which the client does not do, so
    the attack would understate what those placements expose;
  * scoring the inverse on auxiliary rows that also appear in the target set,
    which reports memorization as leakage.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch


def _meta(**over) -> dict:
    base = {
        "embedding_dim": 32,
        "num_rounds": 200,
        "stopped_early": False,
        "privacy_primitives": {},
        "label_privacy": None,
        "metrics": {"auc_roc": 0.9},
        "best_val_auc": 0.9,
    }
    base.update(over)
    return base


def _prims(eps: float, sigma: float, steps: int = 200) -> dict:
    return {
        "target_epsilon": eps, "sigma": sigma,
        "sample_rate": 0.05, "delta": 1e-5, "steps": steps,
    }


class TestArmClassification:
    """Which channels a run had on, read back from its own metadata."""

    @pytest.mark.parametrize(
        "prims,label,expected_arm,expected_eps",
        [
            ({}, None, "none", None),
            ({"server": _prims(2, 3.0)}, None, "server", 2),
            ({"silo_a": _prims(1, 3.0), "silo_b": _prims(1, 3.0)}, None, "cut-layer", 1),
            ({}, {"epsilon": 4.0}, "label", 4.0),
            (
                {"server": _prims(8, 1.1), "silo_a": _prims(8, 1.1), "silo_b": _prims(8, 1.1)},
                {"epsilon": 8.0},
                "all",
                8,
            ),
        ],
    )
    def test_arm_and_epsilon(self, prims, label, expected_arm, expected_eps) -> None:
        from scripts.run_attack_grid import classify

        arm, eps = classify(_meta(privacy_primitives=prims, label_privacy=label))
        assert arm == expected_arm
        assert eps == expected_eps

    def test_label_only_epsilon_comes_from_the_label_block(self) -> None:
        """The label channel records its budget nowhere else, so a run with
        only label DP must not come back with eps=None and land in the
        no-DP row."""
        from scripts.run_attack_grid import classify

        arm, eps = classify(_meta(label_privacy={"epsilon": 0.5}))
        assert (arm, eps) == ("label", 0.5)


class TestNoiseCalibration:
    """The attack must use what the run SPENT, not what it was configured for."""

    def test_reads_realized_sigma_and_steps(self) -> None:
        from scripts.run_attack_grid import cut_noise_for

        sigma, steps = cut_noise_for(
            _meta(privacy_primitives={"silo_a": _prims(1, 3.0746, steps=137)})
        )
        assert sigma == pytest.approx(3.0746)
        assert steps == 137, "an early-stopped run spends fewer steps than configured"

    def test_undefended_run_reports_no_noise(self) -> None:
        """A run with the cut-layer channel off sends a raw embedding: no clip,
        no noise. Reporting a sigma here would make the driver noise an
        embedding the client never noised."""
        from scripts.run_attack_grid import cut_noise_for

        assert cut_noise_for(_meta(privacy_primitives={"server": _prims(2, 3.0)})) == (None, 0)


class TestAuxiliarySplit:
    """Disjointness is enforced in FEATURE space, not by row index.

    Silo A has 18 mostly-categorical columns, so students with identical
    demographics produce byte-identical rows: 44% of train rows also appear in
    the test set. Splitting by index alone would leave the inverse scored on
    rows it had trained on.
    """

    def test_removes_target_rows_from_the_auxiliary_set(self) -> None:
        from scripts.run_attack_grid import disjoint_auxiliary

        train = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [1.0, 0.0]])
        test = np.array([[1.0, 0.0], [9.0, 0.0]])
        keep = disjoint_auxiliary(train, test)
        assert keep.tolist() == [False, True, True, False]

    def test_result_satisfies_the_attack_modules_own_guard(self) -> None:
        """run_learned_inversion raises on overlap; the mask must prevent it."""
        from attack.embedding_inversion import _overlaps
        from scripts.run_attack_grid import disjoint_auxiliary

        rng = np.random.default_rng(0)
        train = rng.integers(0, 3, size=(300, 4)).astype(np.float32)
        test = rng.integers(0, 3, size=(60, 4)).astype(np.float32)
        keep = disjoint_auxiliary(train, test)
        assert keep.sum() > 0, "nothing left to train the inverse on"
        assert not _overlaps(
            torch.tensor(test), torch.tensor(train[keep])
        ), "the attack module's own guard still sees an overlap"

    def test_raises_when_nothing_survives(self) -> None:
        """Better to fail than to silently attack with an empty auxiliary set."""
        from scripts.run_attack_grid import disjoint_auxiliary

        same = np.array([[1.0], [1.0]])
        with pytest.raises(ValueError, match="auxiliary"):
            disjoint_auxiliary(same, same)


class TestModelLoading:
    """Dimensions come from the saved tensors, not from assumptions."""

    def test_infers_dims_from_a_state_dict(self, tmp_path) -> None:
        from fl_server.top_model import TopModel
        from scripts.run_attack_grid import load_models
        from silo_a.bottom_model_a import BottomModelA
        from silo_b.bottom_model_b import BottomModelB

        a = BottomModelA(input_dim=18, embedding_dim=16, hidden_dim=32)
        b = BottomModelB(input_dim=26, embedding_dim=16, hidden_dim=48)
        t = TopModel(embedding_dim=16, hidden_dim=8)
        torch.save(a.state_dict(), tmp_path / "silo_a.pt")
        torch.save(b.state_dict(), tmp_path / "silo_b.pt")
        torch.save(t.state_dict(), tmp_path / "top.pt")
        (tmp_path / "meta.json").write_text(json.dumps(_meta(embedding_dim=16)))

        A, B, T = load_models(tmp_path)
        with torch.no_grad():
            ea, eb = A(torch.zeros(2, 18)), B(torch.zeros(2, 26))
            assert ea.shape == (2, 16) and eb.shape == (2, 16)
            assert T(ea, eb).shape == (2, 1)


class TestProtectedAttributes:
    """Inversion must be runnable against more than one attribute.

    Reporting a single attribute invites the reading that it was chosen for
    the result. The registry makes the choice explicit and lets the same
    attack run against every sensitive column Silo A holds.
    """

    def _X(self):
        # cols: 0 imd(10), 1 education(5), 2 age(3), 3 gender(2),
        # 4 disability(2), 5 continuous, 6..17 region one-hots (drop-first)
        import numpy as np
        X = np.zeros((4, 18), dtype=np.float32)
        X[:, 0] = [0, 9, 3, 3]
        X[:, 3] = [0, 1, 1, 0]
        X[0, 6] = 1.0          # first region
        X[1, 17] = 1.0         # last region
        # rows 2 and 3 have no hot bit: the dropped reference region
        return X

    def test_scalar_attribute_reads_its_column(self) -> None:
        from scripts.run_attack_grid import ATTRIBUTES

        spec = ATTRIBUTES["imd_band"]
        assert spec.n_classes == 10
        assert spec.extract(self._X()).tolist() == [0, 9, 3, 3]

    def test_region_recovers_the_dropped_reference_level(self) -> None:
        """Region is one-hot with drop-first, so an all-zero row is not
        missing data, it is the reference region and must get its own class.
        Treating it as class 0 would merge two regions into one."""
        from scripts.run_attack_grid import ATTRIBUTES

        spec = ATTRIBUTES["region"]
        assert spec.n_classes == 13
        assert spec.extract(self._X()).tolist() == [0, 11, 12, 12]

    def test_every_registered_attribute_is_usable(self) -> None:
        from scripts.run_attack_grid import ATTRIBUTES

        X = self._X()
        for name, spec in ATTRIBUTES.items():
            y = spec.extract(X)
            assert y.min() >= 0, name
            assert y.max() < spec.n_classes, f"{name} emits an out-of-range class"
            assert len(y) == len(X), name
