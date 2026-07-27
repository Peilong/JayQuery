"""Data loading unit tests — owned by Alex Pellet.

Run: pytest tests/test_loaders.py -v
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
FIXTURE_DIR = Path("tests/fixtures")  # synthetic OULAD data committed for CI
MODULE = "CCC_2014J"
A_FEATURE_COUNT = 18
B_FEATURE_COUNT = 26  # rate_v1 layout (docs/DATA_DICTIONARY.md)
PSI_AVAILABLE = False


def _csv_dir() -> Path:
    """Return data/raw if the real OULAD data exists, else fall back to tests/fixtures/."""
    return RAW_DIR if (RAW_DIR / "studentInfo.csv").exists() else FIXTURE_DIR


@pytest.fixture(scope="module")
def raw_student_info():
    return pd.read_csv(_csv_dir() / "studentInfo.csv")


@pytest.fixture(scope="module")
def raw_student_vle():
    return pd.read_csv(_csv_dir() / "studentVle.csv")


@pytest.fixture(scope="module")
def raw_vle():
    return pd.read_csv(_csv_dir() / "vle.csv")


@pytest.fixture(scope="module")
def raw_assessments():
    return pd.read_csv(_csv_dir() / "assessments.csv")


@pytest.fixture(scope="module")
def raw_student_assessment():
    return pd.read_csv(_csv_dir() / "studentAssessment.csv")


@pytest.fixture(scope="module")
def raw_courses():
    return pd.read_csv(_csv_dir() / "courses.csv")


@pytest.fixture(scope="module")
def raw_student_registration():
    return pd.read_csv(_csv_dir() / "studentRegistration.csv")


@pytest.fixture(scope="module")
def cohort(raw_student_info, raw_student_registration):
    """Deduplicated, label-corrected cohort for MODULE (docs/action_plan.md §E-a, §F)."""
    from scripts.prepare_silos import build_cohort

    cohort_df, _ = build_cohort(
        raw_student_info, raw_student_registration, modules=[MODULE], dedup_rule="first"
    )
    return cohort_df


@pytest.fixture(scope="module")
def prepared_silo_a(cohort):
    from scripts.prepare_silos import prepare_silo_a_data

    return prepare_silo_a_data(cohort)


@pytest.fixture(scope="module")
def prepared_silo_b(raw_student_vle, raw_vle, raw_assessments, raw_student_assessment):
    """[DEPRECATED] legacy v0 features — retained only to keep the v0 path tested."""
    from scripts.prepare_silos import prepare_silo_b_data

    return prepare_silo_b_data(
        MODULE, raw_student_vle, raw_vle, raw_assessments, raw_student_assessment
    )


@pytest.fixture(scope="module")
def prepared_silo_b_rate(
    cohort, raw_student_vle, raw_vle, raw_assessments, raw_student_assessment, raw_courses
):
    from scripts.prepare_silos import prepare_silo_b_rate_v1

    return prepare_silo_b_rate_v1(
        cohort, raw_student_vle, raw_vle, raw_assessments, raw_student_assessment, raw_courses
    )


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


# ── DataLoader fixtures ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_loaders_a():
    from silo_a.data_loader_a import make_synthetic_loader_a

    return make_synthetic_loader_a()


@pytest.fixture(scope="module")
def synthetic_loaders_b():
    from silo_b.data_loader_b import make_synthetic_loader_b

    return make_synthetic_loader_b()


@pytest.fixture(scope="module")
def loaded_silo_a(tmp_path_factory):
    """Load Silo A from data/processed if available, else skip."""
    from silo_a.data_loader_a import load_silo_a

    silo_path = Path("data/processed/silo_a.pkl")
    if not silo_path.exists():
        pytest.skip("data/processed/silo_a.pkl not found — skipping real-data loader tests")
    return load_silo_a(bypass_psi=(not PSI_AVAILABLE))


@pytest.fixture(scope="module")
def loaded_silo_b():
    """Load Silo B from data/processed if available, else skip."""
    from silo_b.data_loader_b import load_silo_b

    silo_path = Path("data/processed/silo_b.pkl")
    if not silo_path.exists():
        pytest.skip("data/processed/silo_b.pkl not found — skipping real-data loader tests")
    return load_silo_b()


# ── Silo A: prepare_silo_a_data() ────────────────────────────────────────────


class TestPreparedSiloA:
    def test_has_rows(self, prepared_silo_a):
        assert len(prepared_silo_a) > 0, "Silo A DataFrame should not be empty"

    def test_expected_encoded_columns_present(self, prepared_silo_a):
        expected = {
            "imd_band_enc",
            "highest_education_enc",
            "age_band_enc",
            "gender_enc",
            "disability_enc",
            "studied_credits",
            "label",
        }
        assert expected.issubset(
            prepared_silo_a.columns
        ), f"Missing columns: {expected - set(prepared_silo_a.columns)}"

    def test_raw_columns_dropped(self, prepared_silo_a):
        dropped = {
            "imd_band",
            "highest_education",
            "age_band",
            "region",
            "gender",
            "disability",
            "code_module",
            "code_presentation",
            "final_result",
        }
        assert dropped.isdisjoint(
            prepared_silo_a.columns
        ), f"Raw columns still present: {dropped & set(prepared_silo_a.columns)}"

    def test_region_dummies_present(self, prepared_silo_a):
        region_cols = [c for c in prepared_silo_a.columns if c.startswith("region_")]
        assert len(region_cols) > 0, "No OHE region columns found"

    def test_no_nulls_in_encoded_features(self, prepared_silo_a):
        feature_cols = [
            "imd_band_enc",
            "highest_education_enc",
            "age_band_enc",
            "gender_enc",
            "disability_enc",
            "studied_credits",
        ]
        null_counts = prepared_silo_a[feature_cols].isnull().sum()
        assert null_counts.sum() == 0, f"Unexpected nulls:\n{null_counts[null_counts > 0]}"

    def test_label_is_binary(self, prepared_silo_a):
        unique_labels = set(prepared_silo_a["label"].unique())
        assert unique_labels <= {0, 1}, f"Label column contains unexpected values: {unique_labels}"

    def test_label_distribution_not_degenerate(self, prepared_silo_a):
        counts = prepared_silo_a["label"].value_counts(normalize=True)
        assert counts.min() >= 0.05, f"Label distribution is heavily skewed: {counts.to_dict()}"

    def test_index_is_id_student(self, prepared_silo_a):
        assert (
            prepared_silo_a.index.name == "id_student"
        ), "Silo A index should be id_student after prepare step"

    def test_index_dtype_is_int(self, prepared_silo_a):
        assert np.issubdtype(
            prepared_silo_a.index.dtype, np.integer
        ), f"id_student index dtype should be int, got {prepared_silo_a.index.dtype}"


# ── Silo B: prepare_silo_b_data() ────────────────────────────────────────────


class TestPreparedSiloB:
    def test_has_rows(self, prepared_silo_b):
        assert len(prepared_silo_b) > 0, "Silo B DataFrame should not be empty"

    def test_no_label_column(self, prepared_silo_b):
        assert "label" not in prepared_silo_b.columns, "Silo B must not contain a label column"

    def test_no_final_result_column(self, prepared_silo_b):
        assert "final_result" not in prepared_silo_b.columns, "Silo B must not contain final_result"

    def test_expected_columns_present(self, prepared_silo_b):
        expected = {
            "id_student",
            "weighted_score_pct",
            "participation_rate",
            "num_assessments",
            "start_date",
            "end_date",
            "active_days",
            "sum_click",
        }
        assert expected.issubset(
            prepared_silo_b.columns
        ), f"Missing columns: {expected - set(prepared_silo_b.columns)}"

    def test_vle_click_columns_present(self, prepared_silo_b):
        vle_cols = [c for c in prepared_silo_b.columns if c.startswith("vle_clicks_")]
        assert len(vle_cols) > 0, "No vle_clicks_ columns found in Silo B"

    def test_id_student_dtype_is_int(self, prepared_silo_b):
        dtype = prepared_silo_b["id_student"].dropna().dtype
        assert np.issubdtype(dtype, np.integer), f"id_student should be int dtype, got {dtype}"

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


# ── Silo B: prepare_silo_b_rate_v1() ─────────────────────────────────────────


class TestPreparedSiloBRateV1:
    def test_has_rows(self, prepared_silo_b_rate):
        assert len(prepared_silo_b_rate) > 0, "rate_v1 Silo B DataFrame should not be empty"

    def test_expected_columns_present(self, prepared_silo_b_rate):
        expected = {
            "id_student",
            "weighted_score_pct",
            "participation_rate",
            "submission_ratio",
            "regularity",
            "start_date",
            "clicks_per_active_week",
        }
        assert expected.issubset(
            prepared_silo_b_rate.columns
        ), f"Missing columns: {expected - set(prepared_silo_b_rate.columns)}"

    def test_leaky_v0_columns_absent(self, prepared_silo_b_rate):
        leaky = {"end_date", "sum_click", "active_days", "num_assessments"}
        assert leaky.isdisjoint(
            prepared_silo_b_rate.columns
        ), f"Leaky v0 columns present: {leaky & set(prepared_silo_b_rate.columns)}"

    def test_vle_rate_columns_present(self, prepared_silo_b_rate):
        vle_cols = [c for c in prepared_silo_b_rate.columns if c.startswith("vle_clicks_")]
        assert len(vle_cols) > 0, "No vle_clicks_*_per_week columns found"
        assert all(c.endswith("_per_week") for c in vle_cols), "VLE columns must be rates"

    def test_ratios_in_range(self, prepared_silo_b_rate):
        for col in ["weighted_score_pct", "participation_rate", "submission_ratio"]:
            vals = prepared_silo_b_rate[col].dropna()
            assert vals.between(0, 1).all(), f"{col} values should be in [0, 1]"

    def test_regularity_bounded(self, prepared_silo_b_rate):
        vals = prepared_silo_b_rate["regularity"].dropna()
        # Max is 8/7 from span-boundary effects (docs/DATA_DICTIONARY.md)
        assert vals.between(0, 8 / 7 + 1e-9).all(), "regularity out of expected bounds"

    def test_rates_non_negative(self, prepared_silo_b_rate):
        rate_cols = ["clicks_per_active_week"] + [
            c for c in prepared_silo_b_rate.columns if c.startswith("vle_clicks_")
        ]
        result = (prepared_silo_b_rate[rate_cols].fillna(0) >= 0).all(axis=None)
        assert bool(result), "Rate columns should be non-negative"

    def test_no_label_column(self, prepared_silo_b_rate):
        assert "label" not in prepared_silo_b_rate.columns
        assert "final_result" not in prepared_silo_b_rate.columns


class TestRateV1Computation:
    """Exact-value checks on a tiny hand-computed synthetic module (docs/DATA_DICTIONARY.md)."""

    @pytest.fixture(scope="class")
    def rate_output(self):
        from scripts.prepare_silos import prepare_silo_b_rate_v1

        vle = pd.DataFrame(
            {
                "id_site": [1],
                "code_module": ["CCC"],
                "code_presentation": ["2014J"],
                "activity_type": ["resource"],
                "week_from": [np.nan],
                "week_to": [np.nan],
            }
        )
        # Student 1: clicks on days 0/14, submits on day 21. Student 3: one day-0 click.
        student_vle = pd.DataFrame(
            {
                "code_module": ["CCC"] * 3,
                "code_presentation": ["2014J"] * 3,
                "id_student": [1, 1, 3],
                "id_site": [1, 1, 1],
                "date": [0, 14, 0],
                "sum_click": [30, 40, 10],
            }
        )
        assessments = pd.DataFrame(
            {
                "id_assessment": [11, 12, 13],
                "code_module": ["CCC"] * 3,
                "code_presentation": ["2014J"] * 3,
                "assessment_type": ["TMA", "TMA", "Exam"],
                "date": [10.0, 50.0, np.nan],
                "weight": [40.0, 60.0, 100.0],
            }
        )
        # Student 2 is banked-only (sentinel date_submitted = -1, no VLE activity)
        student_assessment = pd.DataFrame(
            {
                "id_student": [1, 2],
                "id_assessment": [11, 11],
                "date_submitted": [21, -1],
                "is_banked": [0, 1],
                "score": [80.0, 50.0],
            }
        )
        courses = pd.DataFrame(
            {
                "code_module": ["CCC"],
                "code_presentation": ["2014J"],
                "module_presentation_length": [100],
            }
        )
        # Minimal cohort: prepare_silo_b_rate_v1 needs only id_student +
        # module_presentation to route each student to their enrolment.
        cohort = pd.DataFrame(
            {"id_student": [1, 2, 3], "module_presentation": ["CCC_2014J"] * 3}
        )
        return prepare_silo_b_rate_v1(
            cohort, student_vle, vle, assessments, student_assessment, courses
        ).set_index("id_student")

    def test_assessment_ratios_student1(self, rate_output):
        # Window [0, 21] -> weeks 3; D = {A11 (w=40)}; submitted A11 with score 80
        row = rate_output.loc[1]
        assert row["weighted_score_pct"] == pytest.approx(0.8)
        assert row["participation_rate"] == pytest.approx(1.0)
        assert row["submission_ratio"] == pytest.approx(1.0)

    def test_click_rates_student1(self, rate_output):
        row = rate_output.loc[1]
        assert row["clicks_per_active_week"] == pytest.approx(70 / 3)
        assert row["vle_clicks_resource_per_week"] == pytest.approx(70 / 3)
        assert row["regularity"] == pytest.approx(2 / 21)
        assert row["start_date"] == pytest.approx(0.0)

    def test_banked_only_student_excluded(self, rate_output):
        assert 2 not in rate_output.index, "Banked-only student must have no active window"

    def test_empty_due_set_ratios_zero(self, rate_output):
        # Student 3's window [0, 0] closes before the first due date (day 10)
        row = rate_output.loc[3]
        assert row["weighted_score_pct"] == 0.0
        assert row["participation_rate"] == 0.0
        assert row["submission_ratio"] == 0.0

    def test_week_clamp_single_day_student(self, rate_output):
        # Span 0 clamps weeks_active to 1
        row = rate_output.loc[3]
        assert row["clicks_per_active_week"] == pytest.approx(10.0)
        assert row["regularity"] == pytest.approx(1 / 7)


# ── build_silo_a() ────────────────────────────────────────────────────────────


class TestBuiltSiloA:
    def test_returns_two_arrays(self, built_silo_a):
        assert len(built_silo_a) == 2, "build_silo_a should return a 2-tuple (X, y)"

    def test_X_shape_rows_match_intersection(self, built_silo_a, intersection_ids):
        X, _ = built_silo_a
        assert X.shape[0] == len(
            intersection_ids
        ), f"X row count {X.shape[0]} != intersection size {len(intersection_ids)}"

    def test_X_dtype_float32(self, built_silo_a):
        X, _ = built_silo_a
        assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"

    def test_y_shape_matches_intersection(self, built_silo_a, intersection_ids):
        _, y = built_silo_a
        assert y.shape == (
            len(intersection_ids),
        ), f"y shape {y.shape} != ({len(intersection_ids)},)"

    def test_y_dtype_float32(self, built_silo_a):
        _, y = built_silo_a
        assert y.dtype == np.float32, f"y should be float32, got {y.dtype}"

    def test_X_no_nans(self, built_silo_a):
        X, _ = built_silo_a
        assert not np.isnan(X).any(), "X from build_silo_a contains NaN values"

    def test_y_is_binary(self, built_silo_a):
        _, y = built_silo_a
        assert set(np.unique(y)) <= {
            0.0,
            1.0,
        }, f"y should only contain 0.0 and 1.0, got {np.unique(y)}"

    def test_y_label_distribution_not_degenerate(self, built_silo_a):
        _, y = built_silo_a
        positive_rate = y.mean()
        assert (
            0.05 <= positive_rate <= 0.95
        ), f"Label distribution is degenerate: positive_rate={positive_rate:.2f}"


# ── build_silo_b() ────────────────────────────────────────────────────────────


class TestBuiltSiloB:
    def test_returns_array(self, built_silo_b):
        assert isinstance(built_silo_b, np.ndarray), "build_silo_b should return an ndarray"

    def test_X_shape_rows_match_intersection(self, built_silo_b, intersection_ids):
        assert built_silo_b.shape[0] == len(
            intersection_ids
        ), f"X row count {built_silo_b.shape[0]} != intersection size {len(intersection_ids)}"

    def test_X_has_columns(self, built_silo_b):
        assert built_silo_b.shape[1] > 0, "X from build_silo_b should have at least one column"

    def test_X_dtype_float32(self, built_silo_b):
        assert built_silo_b.dtype == np.float32, f"X should be float32, got {built_silo_b.dtype}"

    def test_X_no_nans(self, built_silo_b):
        assert not np.isnan(built_silo_b).any(), "X from build_silo_b contains NaN values"

    def test_X_non_negative_click_cols(self, built_silo_b, prepared_silo_b):
        """VLE click columns (which make up most of X) should be non-negative."""
        vle_cols = [c for c in prepared_silo_b.columns if c.startswith("vle_clicks_")]
        # VLE cols are the last len(vle_cols) columns in X
        n_vle = len(vle_cols)
        X_vle = built_silo_b[:, -n_vle:]
        assert bool(
            (X_vle >= 0).all(axis=None)
        ), "VLE click columns in build_silo_b X contain unexpected negative values"


# ── Cross-silo alignment ──────────────────────────────────────────────────────


class TestCrossSimoAlignment:
    def test_silo_a_and_b_row_counts_match(self, built_silo_a, built_silo_b):
        X_a, _ = built_silo_a
        assert (
            X_a.shape[0] == built_silo_b.shape[0]
        ), f"Silo A rows ({X_a.shape[0]}) != Silo B rows ({built_silo_b.shape[0]})"

    def test_intersection_ids_are_unique(self, intersection_ids):
        assert len(intersection_ids) == len(
            set(intersection_ids)
        ), "intersection_ids contains duplicate student IDs"

    def test_intersection_ids_are_ints(self, intersection_ids):
        assert all(
            isinstance(i, int) for i in intersection_ids
        ), "All intersection_ids should be Python ints"


# ── SiloADataset ─────────────────────────────────────────────────────────────


class TestSiloADataset:
    """Unit tests for the SiloADataset PyTorch Dataset class."""

    @pytest.fixture
    def sample_dataset(self):
        from silo_a.data_loader_a import SiloADataset

        X = np.random.default_rng(0).standard_normal((50, A_FEATURE_COUNT)).astype(np.float32)
        y = np.random.default_rng(0).integers(0, 2, size=50).astype(np.float32)
        return SiloADataset(X, y)

    def test_len(self, sample_dataset):
        assert len(sample_dataset) == 50, "Dataset __len__ should equal the number of samples"

    def test_getitem_returns_tuple(self, sample_dataset):
        item = sample_dataset[0]
        assert (
            isinstance(item, tuple) and len(item) == 2
        ), "__getitem__ should return a 2-tuple of (features, label)"

    def test_getitem_feature_dtype(self, sample_dataset):
        import torch

        X, _ = sample_dataset[0]
        assert X.dtype == torch.float32, f"Feature tensor dtype should be float32, got {X.dtype}"

    def test_getitem_label_dtype(self, sample_dataset):
        import torch

        _, y = sample_dataset[0]
        assert y.dtype == torch.float32, f"Label tensor dtype should be float32, got {y.dtype}"

    def test_getitem_feature_shape(self, sample_dataset):
        X, _ = sample_dataset[0]
        assert X.shape == (
            A_FEATURE_COUNT,
        ), f"Expected feature shape ({A_FEATURE_COUNT},), got {X.shape}"

    def test_getitem_label_is_scalar(self, sample_dataset):
        _, y = sample_dataset[0]
        assert y.ndim == 0, f"Label tensor should be a scalar (0-D), got ndim={y.ndim}"

    def test_all_indices_accessible(self, sample_dataset):
        """Ensure no IndexError is raised for any valid index."""
        for i in range(len(sample_dataset)):
            _ = sample_dataset[i]


# ── SiloBDataset ─────────────────────────────────────────────────────────────


class TestSiloBDataset:
    """Unit tests for the SiloBDataset PyTorch Dataset class."""

    @pytest.fixture
    def sample_dataset(self):
        from silo_b.data_loader_b import SiloBDataset

        X = np.random.default_rng(1).standard_normal((50, B_FEATURE_COUNT)).astype(np.float32)
        y = np.random.default_rng(1).integers(0, 2, size=50).astype(np.float32)
        return SiloBDataset(X, y)

    def test_len(self, sample_dataset):
        assert len(sample_dataset) == 50, "Dataset __len__ should equal the number of samples"

    def test_getitem_returns_tuple(self, sample_dataset):
        item = sample_dataset[0]
        assert (
            isinstance(item, tuple) and len(item) == 2
        ), "__getitem__ should return a 2-tuple of (features, label)"

    def test_getitem_feature_dtype(self, sample_dataset):
        import torch

        X, _ = sample_dataset[0]
        assert X.dtype == torch.float32, f"Feature tensor dtype should be float32, got {X.dtype}"

    def test_getitem_label_dtype(self, sample_dataset):
        import torch

        _, y = sample_dataset[0]
        assert y.dtype == torch.float32, f"Label tensor dtype should be float32, got {y.dtype}"

    def test_getitem_feature_shape(self, sample_dataset):
        X, _ = sample_dataset[0]
        assert X.shape == (
            B_FEATURE_COUNT,
        ), f"Expected feature shape ({B_FEATURE_COUNT},), got {X.shape}"

    def test_getitem_label_is_scalar(self, sample_dataset):
        _, y = sample_dataset[0]
        assert y.ndim == 0, f"Label tensor should be a scalar (0-D), got ndim={y.ndim}"

    def test_all_indices_accessible(self, sample_dataset):
        for i in range(len(sample_dataset)):
            _ = sample_dataset[i]


# ── make_synthetic_loader_a() ─────────────────────────────────────────────────


class TestMakeSyntheticLoaderA:
    """Tests for the synthetic Silo A DataLoader factory."""

    def test_returns_three_loaders(self, synthetic_loaders_a):
        assert (
            len(synthetic_loaders_a) == 3
        ), "make_synthetic_loader_a should return a 3-tuple (train, val, test)"

    def test_train_batch_has_features_and_labels(self, synthetic_loaders_a):
        train_loader, _, _ = synthetic_loaders_a
        batch = next(iter(train_loader))
        assert len(batch) == 2, "Each batch should be a (X, y) tuple"

    def test_train_feature_shape(self, synthetic_loaders_a):
        train_loader, _, _ = synthetic_loaders_a
        X, _ = next(iter(train_loader))
        assert (
            X.ndim == 2 and X.shape[1] == A_FEATURE_COUNT
        ), f"Synthetic Silo A features should have {A_FEATURE_COUNT} columns, got shape {X.shape}"

    def test_train_label_shape(self, synthetic_loaders_a):
        train_loader, _, _ = synthetic_loaders_a
        _, y = next(iter(train_loader))
        assert y.ndim == 1, f"Labels should be 1-D, got ndim={y.ndim}"

    def test_split_sizes(self, synthetic_loaders_a):
        """Default split: 800 train / 100 val / 100 test."""
        train_loader, val_loader, test_loader = synthetic_loaders_a
        n_train = sum(X.shape[0] for X, _ in train_loader)
        n_val = sum(X.shape[0] for X, _ in val_loader)
        n_test = sum(X.shape[0] for X, _ in test_loader)
        assert n_train == 800, f"Expected 800 training samples, got {n_train}"
        assert n_val == 100, f"Expected 100 validation samples, got {n_val}"
        assert n_test == 100, f"Expected 100 test samples, got {n_test}"

    def test_val_loader_not_shuffled(self, synthetic_loaders_a):
        """Two passes over val loader should yield identical label sequences."""
        _, val_loader, _ = synthetic_loaders_a
        labels_pass1 = [y for _, y in val_loader]
        labels_pass2 = [y for _, y in val_loader]
        import torch

        for a, b in zip(labels_pass1, labels_pass2):
            assert torch.equal(a, b), "Val loader should not shuffle between epochs"

    def test_reproducibility_across_calls(self):
        """Two calls with the same seed should produce identical first batches."""
        import torch

        from silo_a.data_loader_a import make_synthetic_loader_a

        loader1, _, _ = make_synthetic_loader_a(seed=7)
        loader2, _, _ = make_synthetic_loader_a(seed=7)
        X1, y1 = next(iter(loader1))
        X2, y2 = next(iter(loader2))
        assert torch.allclose(X1, X2), "Features should be identical for the same seed"
        assert torch.equal(y1, y2), "Labels should be identical for the same seed"


# ── make_synthetic_loader_b() ─────────────────────────────────────────────────


class TestMakeSyntheticLoaderB:
    """Tests for the synthetic Silo B DataLoader factory."""

    def test_returns_three_loaders(self, synthetic_loaders_b):
        assert (
            len(synthetic_loaders_b) == 3
        ), "make_synthetic_loader_b should return a 3-tuple (train, val, test)"

    def test_train_batch_has_features_and_labels(self, synthetic_loaders_b):
        train_loader, _, _ = synthetic_loaders_b
        batch = next(iter(train_loader))
        assert len(batch) == 2, "Each batch should be a (X, y) tuple"

    def test_train_feature_shape(self, synthetic_loaders_b):
        train_loader, _, _ = synthetic_loaders_b
        X, _ = next(iter(train_loader))
        assert (
            X.ndim == 2 and X.shape[1] == B_FEATURE_COUNT
        ), f"Synthetic Silo B features should have {B_FEATURE_COUNT} columns, got shape {X.shape}"

    def test_train_label_shape(self, synthetic_loaders_b):
        train_loader, _, _ = synthetic_loaders_b
        _, y = next(iter(train_loader))
        assert y.ndim == 1, f"Labels should be 1-D, got ndim={y.ndim}"

    def test_split_sizes(self, synthetic_loaders_b):
        train_loader, val_loader, test_loader = synthetic_loaders_b
        n_train = sum(X.shape[0] for X, _ in train_loader)
        n_val = sum(X.shape[0] for X, _ in val_loader)
        n_test = sum(X.shape[0] for X, _ in test_loader)
        assert n_train == 800, f"Expected 800 training samples, got {n_train}"
        assert n_val == 100, f"Expected 100 validation samples, got {n_val}"
        assert n_test == 100, f"Expected 100 test samples, got {n_test}"

    def test_val_loader_not_shuffled(self, synthetic_loaders_b):
        _, val_loader, _ = synthetic_loaders_b
        labels_pass1 = [y for _, y in val_loader]
        labels_pass2 = [y for _, y in val_loader]
        import torch

        for a, b in zip(labels_pass1, labels_pass2):
            assert torch.equal(a, b), "Val loader should not shuffle between epochs"

    def test_feature_differs_from_silo_a(self, synthetic_loaders_a, synthetic_loaders_b):
        """Silo A and B synthetic features should not be identical (different seeds)."""
        import torch

        # Both have different numbers of cols, so comparing common rows via labels
        _, val_a, _ = synthetic_loaders_a
        _, val_b, _ = synthetic_loaders_b
        X_a = next(iter(val_a))[0]
        X_b = next(iter(val_b))[0]
        # Can only compare labels, which must match
        y_a = next(iter(val_a))[1]
        y_b = next(iter(val_b))[1]
        assert torch.equal(
            y_a, y_b
        ), "Val labels must match between Silo A and Silo B (shared seed+100)"
        assert not torch.allclose(
            X_a[:, :A_FEATURE_COUNT], X_b[:, :A_FEATURE_COUNT]
        ), "Silo A and B features should differ (distinct RNG seeds)"


# ── Cross-synthetic-silo label alignment ──────────────────────────────────────


class TestSyntheticLabelAlignment:
    """Verify that synthetic loaders for A and B share identical labels (seed+100)."""

    def _collect_labels(self, loader):
        import torch

        return torch.cat([y for _, y in loader])

    def test_train_labels_match(self, synthetic_loaders_a, synthetic_loaders_b):
        import torch

        y_a = self._collect_labels(synthetic_loaders_a[0])
        y_b = self._collect_labels(synthetic_loaders_b[0])
        # Train loaders shuffle, so compare sorted label distributions
        assert torch.equal(
            y_a.sort().values, y_b.sort().values
        ), "Sorted train labels should be identical across synthetic silos"

    def test_val_labels_match_exactly(self, synthetic_loaders_a, synthetic_loaders_b):
        """Val loaders do not shuffle, so order must match exactly."""
        import torch

        y_a = self._collect_labels(synthetic_loaders_a[1])
        y_b = self._collect_labels(synthetic_loaders_b[1])
        assert torch.equal(
            y_a, y_b
        ), "Val labels must be element-wise equal across synthetic silos (shared seed+100)"

    def test_test_labels_match_exactly(self, synthetic_loaders_a, synthetic_loaders_b):
        import torch

        y_a = self._collect_labels(synthetic_loaders_a[2])
        y_b = self._collect_labels(synthetic_loaders_b[2])
        assert torch.equal(
            y_a, y_b
        ), "Test labels must be element-wise equal across synthetic silos (shared seed+100)"

    def test_label_positive_rate_approx_0_3(self, synthetic_loaders_a):
        """Labels are drawn from Binomial(1, 0.3); check the rate is plausible."""

        y = self._collect_labels(synthetic_loaders_a[0])
        rate = y.mean().item()
        assert 0.15 <= rate <= 0.45, f"Expected positive rate near 0.3, got {rate:.3f}"


# ── validate_alignment() ──────────────────────────────────────────────────────


class TestValidateAlignment:
    """Tests for the cross-silo alignment check utility."""

    def test_returns_true_for_aligned_loaders(self, synthetic_loaders_a, synthetic_loaders_b):
        from silo_a.data_loader_a import validate_alignment

        train_a, _, _ = synthetic_loaders_a
        train_b, _, _ = synthetic_loaders_b
        # Alignment check uses first-batch labels; train loaders shuffle so reload
        # with shuffle=False to get a stable first batch for this assertion
        from silo_a.data_loader_a import make_synthetic_loader_a
        from silo_b.data_loader_b import make_synthetic_loader_b

        stable_a, _, _ = make_synthetic_loader_a(n_train=100, batch_size=100)
        stable_b, _, _ = make_synthetic_loader_b(n_train=100, batch_size=100)
        # Both cover the full train split without shuffling between batches
        result = validate_alignment(stable_a, stable_b, n_check=10)
        assert result is True, "validate_alignment should return True for matching loaders"

    def test_returns_false_for_misaligned_loaders(self):
        """Loaders built from different seeds should fail alignment check."""
        from silo_a.data_loader_a import make_synthetic_loader_a, validate_alignment

        loader_seed42, _, _ = make_synthetic_loader_a(seed=42, n_train=100, batch_size=100)
        loader_seed99, _, _ = make_synthetic_loader_a(seed=99, n_train=100, batch_size=100)
        result = validate_alignment(loader_seed42, loader_seed99, n_check=10)
        assert result is False, "validate_alignment should return False when label sequences differ"

    def test_returns_bool(self, synthetic_loaders_a, synthetic_loaders_b):
        from silo_a.data_loader_a import validate_alignment

        train_a, _, _ = synthetic_loaders_a
        train_b, _, _ = synthetic_loaders_b
        result = validate_alignment(train_a, train_b)
        assert isinstance(
            result, bool
        ), f"validate_alignment should return a bool, got {type(result)}"


# ── load_silo_a() / load_silo_b() — real data ────────────────────────────────


class TestLoadSiloA:
    """Tests for load_silo_a() against real processed data (skipped if unavailable)."""

    def test_returns_three_loaders(self, loaded_silo_a):
        assert len(loaded_silo_a) == 3, "load_silo_a should return a 3-tuple"

    def test_train_batch_shape(self, loaded_silo_a):
        train_loader, _, _ = loaded_silo_a
        X, y = next(iter(train_loader))
        assert X.ndim == 2, "Feature batch should be 2-D"
        assert y.ndim == 1, "Label batch should be 1-D"

    def test_feature_dtype_float32(self, loaded_silo_a):
        import torch

        train_loader, _, _ = loaded_silo_a
        X, _ = next(iter(train_loader))
        assert X.dtype == torch.float32, f"Features should be float32, got {X.dtype}"

    def test_label_dtype_float32(self, loaded_silo_a):
        import torch

        train_loader, _, _ = loaded_silo_a
        _, y = next(iter(train_loader))
        assert y.dtype == torch.float32, f"Labels should be float32, got {y.dtype}"

    def test_labels_are_binary(self, loaded_silo_a):

        train_loader, _, _ = loaded_silo_a
        _, y = next(iter(train_loader))
        assert set(y.unique().tolist()) <= {
            0.0,
            1.0,
        }, f"Labels should only be 0.0 or 1.0, got {y.unique().tolist()}"

    def test_missing_pkl_raises_file_not_found(self, tmp_path):
        from silo_a.data_loader_a import load_silo_a

        with pytest.raises(FileNotFoundError, match="silo_a.pkl"):
            load_silo_a(processed_dir=tmp_path, bypass_psi=True)

    def test_missing_id_list_raises_file_not_found(self, tmp_path):
        """When bypass_psi=False, absence of ordered_id_list.json should raise."""
        import pickle

        from silo_a.data_loader_a import load_silo_a

        # Create a dummy pkl so the first check passes
        dummy = {
            "X_train": np.zeros((2, 12), dtype=np.float32),
            "y_train": np.zeros(2, dtype=np.float32),
            "X_val": np.zeros((2, 12), dtype=np.float32),
            "y_val": np.zeros(2, dtype=np.float32),
            "X_test": np.zeros((2, 12), dtype=np.float32),
            "y_test": np.zeros(2, dtype=np.float32),
        }
        with open(tmp_path / "silo_a.pkl", "wb") as f:
            pickle.dump(dummy, f)
        with pytest.raises(FileNotFoundError, match="ordered_id_list.json"):
            load_silo_a(processed_dir=tmp_path, bypass_psi=False)


class TestLoadSiloB:
    """Tests for load_silo_b() against real processed data (skipped if unavailable)."""

    def test_returns_three_loaders(self, loaded_silo_b):
        assert len(loaded_silo_b) == 3, "load_silo_b should return a 3-tuple"

    def test_train_batch_shape(self, loaded_silo_b):
        train_loader, _, _ = loaded_silo_b
        X, y = next(iter(train_loader))
        assert X.ndim == 2, "Feature batch should be 2-D"
        assert y.ndim == 1, "Label batch should be 1-D"

    def test_feature_dtype_float32(self, loaded_silo_b):
        import torch

        train_loader, _, _ = loaded_silo_b
        X, _ = next(iter(train_loader))
        assert X.dtype == torch.float32, f"Features should be float32, got {X.dtype}"

    def test_label_dtype_float32(self, loaded_silo_b):
        import torch

        train_loader, _, _ = loaded_silo_b
        _, y = next(iter(train_loader))
        assert y.dtype == torch.float32, f"Labels should be float32, got {y.dtype}"

    def test_missing_pkl_raises_file_not_found(self, tmp_path):
        from silo_b.data_loader_b import load_silo_b

        with pytest.raises(FileNotFoundError, match="silo_b.pkl"):
            load_silo_b(processed_dir=tmp_path)


# ── build_cohort(): dedup + pre-start-withdrawal exclusion ────────────────────
# Guards the invariants the DP guarantee depends on (docs/action_plan.md §E-a, §F).


def _multi_module_info() -> pd.DataFrame:
    """studentInfo-shaped frame where student 1 has MIXED outcomes across presentations.

    Student 1: 2013B Withdrawn, 2013J Pass, 2014J Fail  -> dedup rule decides the label.
    Student 2: single enrolment, Pass. Student 3: single enrolment, Withdrawn (pre-start).
    """
    return pd.DataFrame(
        {
            "code_module": ["AAA", "AAA", "BBB", "AAA", "BBB"],
            "code_presentation": ["2013B", "2013J", "2014J", "2013J", "2013B"],
            "id_student": [1, 1, 1, 2, 3],
            "region": ["East Anglian Region"] * 5,
            "final_result": ["Withdrawn", "Pass", "Fail", "Pass", "Withdrawn"],
        }
    )


def _multi_module_registration() -> pd.DataFrame:
    """Student 1's 2013B withdrawal and student 3's are pre-start (day <= 0)."""
    return pd.DataFrame(
        {
            "code_module": ["AAA", "AAA", "BBB", "AAA", "BBB"],
            "code_presentation": ["2013B", "2013J", "2014J", "2013J", "2013B"],
            "id_student": [1, 1, 1, 2, 3],
            "date_registration": [-30, -30, -30, -30, -30],
            "date_unregistration": [-5.0, np.nan, np.nan, np.nan, 0.0],
        }
    )


class TestBuildCohort:
    def test_dedup_yields_one_row_per_student(self):
        """The one-record-per-person invariant the DP sensitivity bound relies on."""
        from scripts.prepare_silos import DEDUP_RULES, build_cohort

        for rule in DEDUP_RULES:
            cohort, stats = build_cohort(
                _multi_module_info(),
                _multi_module_registration(),
                dedup_rule=rule,
                withdrawal_cutoff_day=None,
            )
            assert not cohort["id_student"].duplicated().any(), f"{rule} left duplicates"
            assert stats["n_students"] == len(cohort) == 3

    def test_dedup_rules_select_expected_enrolment(self):
        """first = chronological; best/worst select on outcome. Student 1 is mixed."""
        from scripts.prepare_silos import build_cohort

        def label_of(rule, student=1):
            cohort, _ = build_cohort(
                _multi_module_info(),
                _multi_module_registration(),
                dedup_rule=rule,
                withdrawal_cutoff_day=None,
            )
            return int(cohort.set_index("id_student").loc[student, "label"])

        # earliest is AAA_2013B (Withdrawn -> 1); best picks a Pass (0); worst picks a 1
        assert label_of("first") == 1
        assert label_of("best") == 0
        assert label_of("worst") == 1

    def test_best_worst_bracket_first(self):
        """best <= first <= worst on label rate — the sensitivity-bound property."""
        from scripts.prepare_silos import build_cohort

        rates = {
            rule: build_cohort(
                _multi_module_info(), _multi_module_registration(), dedup_rule=rule
            )[1]["label_rate"]
            for rule in ("best", "first", "worst")
        }
        assert rates["best"] <= rates["first"] <= rates["worst"]

    def test_prestart_withdrawals_excluded(self):
        """§F: withdrawals unregistering on/before the cutoff are dropped."""
        from scripts.prepare_silos import build_cohort

        info, reg = _multi_module_info(), _multi_module_registration()
        cohort, stats = build_cohort(info, reg, dedup_rule="first")
        # student 3's only enrolment is a pre-start withdrawal -> student disappears
        assert 3 not in set(cohort["id_student"])
        assert stats["n_excluded_prestart_withdrawals"] == 2  # student 1 (2013B) + student 3
        # with the exclusion off, student 3 is retained
        kept, _ = build_cohort(info, reg, dedup_rule="first", withdrawal_cutoff_day=None)
        assert 3 in set(kept["id_student"])

    def test_cutoff_is_parameterized(self):
        """§F4's day<=14 sensitivity cut must need no code change."""
        from scripts.prepare_silos import build_cohort

        info, reg = _multi_module_info(), _multi_module_registration()
        n_none = build_cohort(info, reg, withdrawal_cutoff_day=None)[1]["n_students"]
        n_zero = build_cohort(info, reg, withdrawal_cutoff_day=0)[1]["n_students"]
        assert n_none > n_zero

    def test_module_filter_and_unknown_module_raises(self):
        from scripts.prepare_silos import build_cohort

        info, reg = _multi_module_info(), _multi_module_registration()
        cohort, stats = build_cohort(info, reg, modules=["AAA_2013J"], withdrawal_cutoff_day=None)
        assert set(cohort["code_presentation"]) == {"2013J"}
        assert stats["modules"] == ["AAA_2013J"]
        with pytest.raises(ValueError, match="Unknown module-presentation"):
            build_cohort(info, reg, modules=["ZZZ_1999J"])

    def test_invalid_dedup_rule_raises(self):
        from scripts.prepare_silos import build_cohort

        with pytest.raises(ValueError, match="dedup_rule must be one of"):
            build_cohort(_multi_module_info(), _multi_module_registration(), dedup_rule="newest")

    def test_stats_are_manifest_ready(self):
        """Every field a variant manifest needs is present and JSON-serializable."""
        import json

        from scripts.prepare_silos import build_cohort

        _, stats = build_cohort(_multi_module_info(), _multi_module_registration())
        required = {
            "dedup_rule", "withdrawal_cutoff_day", "modules", "n_module_presentations",
            "n_raw_rows", "n_after_module_filter", "n_excluded_prestart_withdrawals",
            "n_before_dedup", "n_dropped_by_dedup", "n_students", "label_rate",
        }
        assert required.issubset(stats), f"missing: {required - set(stats)}"
        json.dumps(stats)  # must not raise


class TestCohortFeatureStability:
    """Feature dimensions must not drift when the cohort spans many presentations."""

    def test_silo_a_dims_match_run_config(self, prepared_silo_a):
        region_cols = [c for c in prepared_silo_a.columns if c.startswith("region_")]
        encoded = [
            "imd_band_enc", "highest_education_enc", "age_band_enc",
            "gender_enc", "disability_enc", "studied_credits",
        ]
        assert len(encoded) + len(region_cols) == A_FEATURE_COUNT

    def test_module_identity_is_not_a_feature(self, prepared_silo_a):
        """Stratify-only decision: module must not leak in as a column (§E)."""
        leaked = [
            c for c in prepared_silo_a.columns
            if "module" in c.lower() or "presentation" in c.lower()
        ]
        assert leaked == [], f"module identity leaked into Silo A features: {leaked}"

    def test_silo_b_unique_per_student(self, prepared_silo_b_rate):
        """A deduped student must be scored on exactly one enrolment."""
        assert not prepared_silo_b_rate["id_student"].duplicated().any()
