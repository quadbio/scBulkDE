"""Tests for _generate_samples function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scbulkde.ut.ut_basic import _generate_samples


class TestGenerateSamples:
    def test_empty_stratify_by_returns_false_and_empty_df(self, make_obs):
        obs = make_obs(
            n_query=2,
            n_reference=1,
            condition_col="condition",
            strata={"batch": {"query": ["A", "A"], "reference": ["B"]}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=[],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 0
        assert isinstance(sample_stats, pd.DataFrame)
        assert len(sample_stats) == 0

    def test_no_query_cells_returns_false(self, make_obs):
        obs = make_obs(
            n_query=0,
            n_reference=3,
            condition_col="condition",
            strata={"batch": {"query": [], "reference": ["A", "B", "C"]}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0
        assert len(sample_stats) == 0

    def test_no_reference_cells_returns_false(self, make_obs):
        obs = make_obs(
            n_query=3,
            n_reference=0,
            condition_col="condition",
            strata={"batch": {"query": ["A", "B", "C"], "reference": []}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0
        assert len(sample_stats) == 0

    def test_min_cells_only_filters_small_groups(self, make_obs):
        obs = make_obs(
            n_query=10,
            n_reference=10,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 5 + ["B"] * 5, "reference": ["A"] * 8 + ["B"] * 2}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 18

        reference_filtered = filtered[filtered["condition"] == "reference"]
        assert set(reference_filtered["batch"].unique()) == {"A"}

    def test_min_cells_no_qualifying_groups_in_one_condition(self, make_obs):
        obs = make_obs(
            n_query=10,
            n_reference=10,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10, "reference": ["B"] * 3 + ["C"] * 3 + ["D"] * 4}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0

    def test_min_fraction_only_filters_by_relative_size(self, make_obs):
        obs = make_obs(
            n_query=100,
            n_reference=100,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 30 + ["B"] * 70, "reference": ["A"] * 15 + ["B"] * 85}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=None,
            min_fraction=0.2,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 185

        reference_filtered = filtered[filtered["condition"] == "reference"]
        assert set(reference_filtered["batch"].unique()) == {"B"}

    def test_min_fraction_calculated_per_condition_not_global(self, make_obs):
        obs = make_obs(
            n_query=20,
            n_reference=80,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10 + ["B"] * 10, "reference": ["A"] * 40 + ["B"] * 40}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=None,
            min_fraction=0.4,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 100

    def test_qualify_strategy_and_requires_both_conditions(self, make_obs):
        obs = make_obs(
            n_query=100,
            n_reference=100,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 40 + ["B"] * 60, "reference": ["A"] * 5 + ["B"] * 95}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=10,
            min_fraction=0.5,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 155

        query_filtered = filtered[filtered["condition"] == "query"]
        assert set(query_filtered["batch"].unique()) == {"B"}
        reference_filtered = filtered[filtered["condition"] == "reference"]
        assert set(reference_filtered["batch"].unique()) == {"B"}

    def test_qualify_strategy_or_requires_either_condition(self, make_obs):
        obs = make_obs(
            n_query=100,
            n_reference=100,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 40 + ["B"] * 60, "reference": ["A"] * 5 + ["B"] * 95}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=10,
            min_fraction=0.5,
            min_coverage=None,
            qualify_strategy="or",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 195

        query_filtered = filtered[filtered["condition"] == "query"]
        assert set(query_filtered["batch"].unique()) == {"A", "B"}
        reference_filtered = filtered[filtered["condition"] == "reference"]
        assert set(reference_filtered["batch"].unique()) == {"B"}

    def test_invalid_qualify_strategy_raises_error(self, make_obs):
        obs = make_obs(
            n_query=2,
            n_reference=2,
            condition_col="condition",
            strata={"batch": {"query": ["A", "B"], "reference": ["A", "B"]}},
        )

        # Both min_cells and min_fraction must be not None in order to enter the
        # if statement that checks qualify_strategy
        with pytest.raises(ValueError, match="qualify_strategy must be 'and' or 'or'"):
            _generate_samples(
                obs=obs,
                stratify_by=["batch"],
                min_cells=1,
                min_fraction=1,
                min_coverage=None,
                qualify_strategy="invalid",
                group_key_internal="condition",
            )

    def test_min_coverage_enforces_minimum_cell_retention(self, make_obs):
        obs = make_obs(
            n_query=100,
            n_reference=100,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10 + ["B"] * 90, "reference": ["A"] * 10 + ["B"] * 90}},
        )

        can_generate_pass, _, _ = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=None,
            min_coverage=0.8,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        can_generate_fail, _, _ = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=None,
            min_coverage=0.95,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate_pass is True
        assert can_generate_fail is False

    def test_min_coverage_calculated_per_condition(self, make_obs):
        obs = make_obs(
            n_query=100,
            n_reference=100,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 95 + ["B"] * 5, "reference": ["A"] * 10 + ["B"] * 90}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=None,
            min_coverage=0.9,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True

    def test_min_coverage_fails_if_one_condition_below_threshold(self, make_obs):
        obs = make_obs(
            n_query=100,
            n_reference=100,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 95 + ["B"] * 5, "reference": ["A"] * 10 + ["B"] * 90}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=None,
            min_coverage=0.91,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0

    def test_single_column_stratification(self, make_obs):
        obs = make_obs(
            n_query=20,
            n_reference=20,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10 + ["B"] * 10, "reference": ["A"] * 10 + ["B"] * 10}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 40

    def test_two_column_stratification(self, make_obs):
        obs = make_obs(
            n_query=40,
            n_reference=40,
            condition_col="condition",
            strata={
                "batch": {"query": ["A"] * 20 + ["B"] * 20, "reference": ["A"] * 20 + ["B"] * 20},
                "donor": {
                    "query": ["D1"] * 10 + ["D2"] * 10 + ["D1"] * 10 + ["D2"] * 10,
                    "reference": ["D1"] * 10 + ["D2"] * 10 + ["D1"] * 10 + ["D2"] * 10,
                },
            },
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch", "donor"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 80

    def test_multi_column_stratification_filters_correctly(self, make_obs):
        obs = make_obs(
            n_query=50,
            n_reference=50,
            condition_col="condition",
            strata={
                "batch": {"query": ["A"] * 25 + ["B"] * 25, "reference": ["A"] * 25 + ["B"] * 25},
                "donor": {
                    "query": ["D1"] * 20 + ["D2"] * 5 + ["D1"] * 20 + ["D2"] * 5,
                    "reference": ["D1"] * 20 + ["D2"] * 5 + ["D1"] * 20 + ["D2"] * 5,
                },
            },
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch", "donor"],
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 80
        assert "D2" not in filtered["donor"].values

    def test_three_column_stratification(self, make_obs):
        obs = make_obs(
            n_query=80,
            n_reference=80,
            condition_col="condition",
            strata={
                "batch": {"query": ["A", "B"] * 40, "reference": ["A", "B"] * 40},
                "donor": {"query": ["D1", "D1", "D2", "D2"] * 20, "reference": ["D1", "D1", "D2", "D2"] * 20},
                "tissue": {"query": ["T1"] * 40 + ["T2"] * 40, "reference": ["T1"] * 40 + ["T2"] * 40},
            },
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch", "donor", "tissue"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 160

    def test_filtered_obs_preserves_original_index(self, make_obs):
        obs = make_obs(
            n_query=20,
            n_reference=20,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10 + ["B"] * 10, "reference": ["A"] * 15 + ["B"] * 5}},
        )
        obs.index = range(100, 140)

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert all(idx >= 100 and idx < 140 for idx in filtered.index)
        assert filtered.index.is_unique

    def test_filtered_obs_contains_only_qualifying_cells(self, make_obs):
        obs = make_obs(
            n_query=30,
            n_reference=30,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 20 + ["B"] * 10, "reference": ["A"] * 5 + ["B"] * 25}},
        )
        obs["cell_id"] = list(range(30)) + list(range(100, 130))

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True

        query_filtered = filtered[filtered["condition"] == "query"]
        ref_filtered = filtered[filtered["condition"] == "reference"]

        assert set(query_filtered["cell_id"]) == set(range(30))
        assert set(ref_filtered["cell_id"]) == set(range(105, 130))

    def test_empty_obs_returns_false(self, make_obs):
        obs = make_obs(
            n_query=0, n_reference=0, condition_col="condition", strata={"batch": {"query": [], "reference": []}}
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0

    def test_categorical_dtype_in_stratification_columns(self, make_obs):
        obs = make_obs(
            n_query=20,
            n_reference=20,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10 + ["B"] * 10, "reference": ["A"] * 10 + ["B"] * 10}},
        )
        obs["condition"] = pd.Categorical(obs["condition"])
        obs["batch"] = pd.Categorical(obs["batch"])

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 40

    def test_missing_values_in_stratification_columns(self, make_obs):
        obs = make_obs(
            n_query=20,
            n_reference=20,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 5 + [None] * 5 + ["B"] * 10, "reference": ["A"] * 10 + ["B"] * 10}},
        )

        try:
            can_generate, filtered, sample_stats = _generate_samples(
                obs=obs,
                stratify_by=["batch"],
                min_cells=5,
                min_fraction=None,
                min_coverage=None,
                qualify_strategy="and",
                group_key_internal="condition",
            )
            assert isinstance(can_generate, bool)
            assert isinstance(filtered, pd.DataFrame)
        except (ValueError, KeyError):
            pass

    def test_no_requirements_specified_returns_false(self, make_obs):
        obs = make_obs(
            n_query=10,
            n_reference=10,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10, "reference": ["A"] * 10}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=None,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0

    def test_filtered_obs_is_subset_of_input_obs(self, make_obs):
        np.random.seed(42)
        obs = make_obs(
            n_query=50,
            n_reference=50,
            condition_col="condition",
            strata={
                "batch": {
                    "query": list(np.random.choice(["A", "B", "C"], 50)),
                    "reference": list(np.random.choice(["A", "B", "C"], 50)),
                },
                "donor": {
                    "query": list(np.random.choice(["D1", "D2", "D3", "D4"], 50)),
                    "reference": list(np.random.choice(["D1", "D2", "D3", "D4"], 50)),
                },
            },
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch", "donor"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert set(filtered.index).issubset(set(obs.index))

    def test_both_conditions_present_in_filtered_obs_when_successful(self, make_obs):
        obs = make_obs(
            n_query=30,
            n_reference=30,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 15 + ["B"] * 15, "reference": ["A"] * 15 + ["B"] * 15}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        if can_generate:
            assert "query" in filtered["condition"].values
            assert "reference" in filtered["condition"].values

    def test_sample_stats_contains_expected_columns(self, make_obs):
        """Test that sample_stats DataFrame contains the new statistics columns."""
        obs = make_obs(
            n_query=20,
            n_reference=20,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10 + ["B"] * 10, "reference": ["A"] * 10 + ["B"] * 10}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert "n_cells" in sample_stats.columns
        assert "n_cells_condition" in sample_stats.columns
        assert "fraction" in sample_stats.columns
        assert "coverage" in sample_stats.columns
        assert "condition" in sample_stats.columns
        assert "batch" in sample_stats.columns

    def test_sample_stats_values_are_correct(self, make_obs):
        """Test that sample_stats values are computed correctly."""
        obs = make_obs(
            n_query=20,
            n_reference=20,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10 + ["B"] * 10, "reference": ["A"] * 10 + ["B"] * 10}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        # Each batch should have 10 cells per condition
        assert all(sample_stats["n_cells"] == 10)
        # Total cells per condition is 20
        assert all(sample_stats["n_cells_condition"] == 20)
        # Fraction is 10/20 = 0.5
        assert all(sample_stats["fraction"] == 0.5)
        # Coverage is 1.0 (all cells qualify)
        assert all(sample_stats["coverage"] == 1.0)
