"""Data loading unit tests — owned by Alex Pellet.

Run: pytest tests/test_loaders.py -v
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from pathlib import Path

# ── Fixtures ──────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
MODULE = "CCC_2014J"

@pytest.fixture(scope="module")
def raw_student_info():
    return pd.read_csv(RAW_DIR / "studentInfo.csv")

@pytest.fixture(scope="module")
def raw_student_vle():
    return pd.read_csv(RAW_DIR / "studentVle.csv")

@pytest.fixture(scope="module")
def raw_vle():
    return pd.read_csv(RAW_DIR / "vle.csv")

@pytest.fixture(scope="module")
def raw_assessments():
    return pd.read_csv(RAW_DIR / "assessments.csv")

@pytest.fixture(scope="module")
def raw_student_assessment():
    return pd.read_csv(RAW_DIR / "studentAssessment.csv")

@pytest.fixture(scope="module")
def prepared_silo_a(raw_student_info):
    from scripts.prepare_silos import prepare_silo_a_data
    return prepare_silo_a_data(MODULE, raw_student_info)

@pytest.fixture(scope="module")
def prepared_silo_b(raw_student_vle, raw_vle, raw_assessments, raw_student_assessment):
    from scripts.prepare_silos import prepare_silo_b_data
    return prepare_silo_b_data(MODULE, raw_student_vle, raw_vle, raw_assessments, raw_student_assessment)

@pytest.fixture(scope="module")
def intersection_ids(prepared_silo_a, prepared_silo_b):
    """Simulate PSI intersection: students present in both silos."""
    ids_a = set(prepared_silo_a.index.tolist())
    ids_b = set(prepared_silo_b["id_student"].dropna().astype(int).tolist())
    return sorted(ids_a & ids_b)

@pytest.fixture(scope="module")
def built_silo_a(prepared_silo_a, intersection_ids):
    from scripts.prepare_silos import build_silo_a
    return build_silo_a(prepared_silo_a, intersection_ids)

@pytest.fixture(scope="module")
def built_silo_b(prepared_silo_b, intersection_ids):
    from scripts.prepare_silos import build_silo_b
    return build_silo_b(prepared_silo_b, intersection_ids)


# ── Silo A: prepare_silo_a_data() ────────────────────────────────────────────

class TestPreparedSiloA:

    def test_has_rows(self, prepared_silo_a):
        assert len(prepared_silo_a) > 0, "Silo A DataFrame should not be empty"

    def test_expected_encoded_columns_present(self, prepared_silo_a):
        expected = {"imd_band_enc", "highest_education_enc", "age_band_enc",
                    "gender_enc", "disability_enc", "studied_credits_norm", "label"}
        assert expected.issubset(prepared_silo_a.columns), (
            f"Missing columns: {expected - set(prepared_silo_a.columns)}"
        )

    def test_raw_columns_dropped(self, prepared_silo_a):
        dropped = {"imd_band", "highest_education", "age_band", "region",
                   "gender", "disability", "studied_credits",
                   "code_module", "code_presentation", "final_result"}
        assert dropped.isdisjoint(prepared_silo_a.columns), (
            f"Raw columns still present: {dropped & set(prepared_silo_a.columns)}"
        )

    def test_region_dummies_present(self, prepared_silo_a):
        region_cols = [c for c in prepared_silo_a.columns if c.startswith("region_")]
        assert len(region_cols) > 0, "No OHE region columns found"

    def test_no_nulls_in_encoded_features(self, prepared_silo_a):
        feature_cols = ["imd_band_enc", "highest_education_enc", "age_band_enc",
                        "gender_enc", "disability_enc", "studied_credits_norm"]
        null_counts = prepared_silo_a[feature_cols].isnull().sum()
        assert null_counts.sum() == 0, f"Unexpected nulls:\n{null_counts[null_counts > 0]}"

    def test_label_is_binary(self, prepared_silo_a):
        unique_labels = set(prepared_silo_a["label"].unique())
        assert unique_labels <= {0, 1}, f"Label column contains unexpected values: {unique_labels}"

    def test_label_distribution_not_degenerate(self, prepared_silo_a):
        counts = prepared_silo_a["label"].value_counts(normalize=True)
        assert counts.min() >= 0.05, (
            f"Label distribution is heavily skewed: {counts.to_dict()}"
        )

    def test_index_is_id_student(self, prepared_silo_a):
        assert prepared_silo_a.index.name == "id_student", (
            "Silo A index should be id_student after prepare step"
        )

    def test_index_dtype_is_int(self, prepared_silo_a):
        assert np.issubdtype(prepared_silo_a.index.dtype, np.integer), (
            f"id_student index dtype should be int, got {prepared_silo_a.index.dtype}"
        )


# ── Silo B: prepare_silo_b_data() ────────────────────────────────────────────

class TestPreparedSiloB:

    def test_has_rows(self, prepared_silo_b):
        assert len(prepared_silo_b) > 0, "Silo B DataFrame should not be empty"

    def test_no_label_column(self, prepared_silo_b):
        assert "label" not in prepared_silo_b.columns, (
            "Silo B must not contain a label column"
        )

    def test_no_final_result_column(self, prepared_silo_b):
        assert "final_result" not in prepared_silo_b.columns, (
            "Silo B must not contain final_result"
        )

    def test_expected_columns_present(self, prepared_silo_b):
        expected = {"id_student", "weighted_score_pct", "participation_rate",
                    "num_assessments", "start_date", "end_date", "active_days", "sum_click"}
        assert expected.issubset(prepared_silo_b.columns), (
            f"Missing columns: {expected - set(prepared_silo_b.columns)}"
        )

    def test_vle_click_columns_present(self, prepared_silo_b):
        vle_cols = [c for c in prepared_silo_b.columns if c.startswith("vle_clicks_")]
        assert len(vle_cols) > 0, "No vle_clicks_ columns found in Silo B"

    def test_id_student_dtype_is_int(self, prepared_silo_b):
        dtype = prepared_silo_b["id_student"].dropna().dtype
        assert np.issubdtype(dtype, np.integer), (
            f"id_student should be int dtype, got {dtype}"
        )

    def test_participation_rate_in_range(self, prepared_silo_b):
        col = prepared_silo_b["participation_rate"].dropna()
        assert col.between(0, 1).all(), "participation_rate values should be in [0, 1]"

    def test_weighted_score_pct_in_range(self, prepared_silo_b):
        col = prepared_silo_b["weighted_score_pct"].dropna()
        assert col.between(0, 1).all(), "weighted_score_pct values should be in [0, 1]"

    def test_vle_clicks_non_negative(self, prepared_silo_b):
        vle_cols = [c for c in prepared_silo_b.columns if c.startswith("vle_clicks_")]
        result = (prepared_silo_b[vle_cols].fillna(0) >= 0).all(axis=None)
        assert bool(result), "All vle_clicks_ values should be non-negative"


# ── build_silo_a() ────────────────────────────────────────────────────────────

class TestBuiltSiloA:

    def test_returns_two_arrays(self, built_silo_a):
        assert len(built_silo_a) == 2, "build_silo_a should return a 2-tuple (X, y)"

    def test_X_shape_rows_match_intersection(self, built_silo_a, intersection_ids):
        X, _ = built_silo_a
        assert X.shape[0] == len(intersection_ids), (
            f"X row count {X.shape[0]} != intersection size {len(intersection_ids)}"
        )

    def test_X_dtype_float32(self, built_silo_a):
        X, _ = built_silo_a
        assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"

    def test_y_shape_matches_intersection(self, built_silo_a, intersection_ids):
        _, y = built_silo_a
        assert y.shape == (len(intersection_ids),), (
            f"y shape {y.shape} != ({len(intersection_ids)},)"
        )

    def test_y_dtype_float32(self, built_silo_a):
        _, y = built_silo_a
        assert y.dtype == np.float32, f"y should be float32, got {y.dtype}"

    def test_X_no_nans(self, built_silo_a):
        X, _ = built_silo_a
        assert not np.isnan(X).any(), "X from build_silo_a contains NaN values"

    def test_y_is_binary(self, built_silo_a):
        _, y = built_silo_a
        assert set(np.unique(y)) <= {0.0, 1.0}, (
            f"y should only contain 0.0 and 1.0, got {np.unique(y)}"
        )

    def test_y_label_distribution_not_degenerate(self, built_silo_a):
        _, y = built_silo_a
        positive_rate = y.mean()
        assert 0.05 <= positive_rate <= 0.95, (
            f"Label distribution is degenerate: positive_rate={positive_rate:.2f}"
        )


# ── build_silo_b() ────────────────────────────────────────────────────────────

class TestBuiltSiloB:

    def test_returns_array(self, built_silo_b):
        assert isinstance(built_silo_b, np.ndarray), (
            "build_silo_b should return an ndarray"
        )

    def test_X_shape_rows_match_intersection(self, built_silo_b, intersection_ids):
        assert built_silo_b.shape[0] == len(intersection_ids), (
            f"X row count {built_silo_b.shape[0]} != intersection size {len(intersection_ids)}"
        )

    def test_X_has_columns(self, built_silo_b):
        assert built_silo_b.shape[1] > 0, "X from build_silo_b should have at least one column"

    def test_X_dtype_float32(self, built_silo_b):
        assert built_silo_b.dtype == np.float32, (
            f"X should be float32, got {built_silo_b.dtype}"
        )

    def test_X_no_nans(self, built_silo_b):
        assert not np.isnan(built_silo_b).any(), "X from build_silo_b contains NaN values"

    def test_X_non_negative_click_cols(self, built_silo_b, prepared_silo_b):
        """VLE click columns (which make up most of X) should be non-negative."""
        vle_cols = [c for c in prepared_silo_b.columns if c.startswith("vle_clicks_")]
        # VLE cols are the last len(vle_cols) columns in X
        n_vle = len(vle_cols)
        X_vle = built_silo_b[:, -n_vle:]
        assert bool((X_vle >= 0).all(axis=None)), (
            "VLE click columns in build_silo_b X contain unexpected negative values"
        )


# ── Cross-silo alignment ──────────────────────────────────────────────────────

class TestCrossSimoAlignment:

    def test_silo_a_and_b_row_counts_match(self, built_silo_a, built_silo_b):
        X_a, _ = built_silo_a
        assert X_a.shape[0] == built_silo_b.shape[0], (
            f"Silo A rows ({X_a.shape[0]}) != Silo B rows ({built_silo_b.shape[0]})"
        )

    def test_intersection_ids_are_unique(self, intersection_ids):
        assert len(intersection_ids) == len(set(intersection_ids)), (
            "intersection_ids contains duplicate student IDs"
        )

    def test_intersection_ids_are_ints(self, intersection_ids):
        assert all(isinstance(i, int) for i in intersection_ids), (
            "All intersection_ids should be Python ints"
        )