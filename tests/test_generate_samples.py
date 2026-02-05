"""Tests for _generate_samples function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scbulkde.ut.ut_basic import _generate_samples


class TestCanGenerateSamples:
    def test_empty_stratify_by_returns_false_and_empty_df(self):
        obs = pd.DataFrame({"condition": ["query", "query", "reference"], "batch": ["A", "A", "B"]})

        can_generate, filtered = _generate_samples(
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

    def test_no_query_cells_returns_false(self):
        obs = pd.DataFrame({"condition": ["reference", "reference", "reference"], "batch": ["A", "B", "C"]})

        can_generate, filtered = _generate_samples(
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

    def test_no_reference_cells_returns_false(self):
        obs = pd.DataFrame({"condition": ["query", "query", "query"], "batch": ["A", "B", "C"]})

        can_generate, filtered = _generate_samples(
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

    def test_min_cells_only_filters_small_groups(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 10 + ["reference"] * 10,
                "batch": ["A"] * 5 + ["B"] * 5 + ["A"] * 8 + ["B"] * 2,
                "cell_id": range(20),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_min_cells_no_qualifying_groups_in_one_condition(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 10 + ["reference"] * 10,
                "batch": ["A"] * 10 + ["B"] * 3 + ["C"] * 3 + ["D"] * 4,
                "cell_id": range(20),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_min_fraction_only_filters_by_relative_size(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A"] * 30 + ["B"] * 70 + ["A"] * 15 + ["B"] * 85,
                "cell_id": range(200),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_min_fraction_calculated_per_condition_not_global(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 80,
                "batch": ["A"] * 10 + ["B"] * 10 + ["A"] * 40 + ["B"] * 40,
                "cell_id": range(100),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_qualify_strategy_and_requires_both_conditions(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A"] * 40 + ["B"] * 60 + ["A"] * 5 + ["B"] * 95,
                "cell_id": range(200),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_qualify_strategy_or_requires_either_condition(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A"] * 40 + ["B"] * 60 + ["A"] * 5 + ["B"] * 95,
                "cell_id": range(200),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_invalid_qualify_strategy_raises_error(self):
        obs = pd.DataFrame({"condition": ["query", "query", "reference", "reference"], "batch": ["A", "B", "A", "B"]})

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

    def test_min_coverage_enforces_minimum_cell_retention(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A"] * 10 + ["B"] * 90 + ["A"] * 10 + ["B"] * 90,
                "cell_id": range(200),
            }
        )

        can_generate_pass, _ = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=None,
            min_coverage=0.8,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        can_generate_fail, _ = _generate_samples(
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

    def test_min_coverage_calculated_per_condition(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A"] * 95 + ["B"] * 5 + ["A"] * 10 + ["B"] * 90,
                "cell_id": range(200),
            }
        )

        can_generate, filtered = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=None,
            min_coverage=0.9,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True

    def test_min_coverage_fails_if_one_condition_below_threshold(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A"] * 95 + ["B"] * 5 + ["A"] * 10 + ["B"] * 90,
                "cell_id": range(200),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_single_column_stratification(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 20,
                "batch": ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
                "cell_id": range(40),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_two_column_stratification(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 40 + ["reference"] * 40,
                "batch": ["A"] * 20 + ["B"] * 20 + ["A"] * 20 + ["B"] * 20,
                "donor": ["D1"] * 10
                + ["D2"] * 10
                + ["D1"] * 10
                + ["D2"] * 10
                + ["D1"] * 10
                + ["D2"] * 10
                + ["D1"] * 10
                + ["D2"] * 10,
                "cell_id": range(80),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_multi_column_stratification_filters_correctly(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 50 + ["reference"] * 50,
                "batch": ["A"] * 25 + ["B"] * 25 + ["A"] * 25 + ["B"] * 25,
                "donor": ["D1"] * 20
                + ["D2"] * 5
                + ["D1"] * 20
                + ["D2"] * 5
                + ["D1"] * 20
                + ["D2"] * 5
                + ["D1"] * 20
                + ["D2"] * 5,
                "cell_id": range(100),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_three_column_stratification(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 80 + ["reference"] * 80,
                "batch": ["A", "B"] * 40 + ["A", "B"] * 40,
                "donor": ["D1", "D1", "D2", "D2"] * 20 + ["D1", "D1", "D2", "D2"] * 20,
                "tissue": ["T1"] * 40 + ["T2"] * 40 + ["T1"] * 40 + ["T2"] * 40,
                "cell_id": range(160),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_filtered_obs_preserves_original_index(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 20,
                "batch": ["A"] * 10 + ["B"] * 10 + ["A"] * 15 + ["B"] * 5,
                "cell_id": range(40),
            },
            index=range(100, 140),
        )

        can_generate, filtered = _generate_samples(
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

    def test_filtered_obs_contains_only_qualifying_cells(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 30 + ["reference"] * 30,
                "batch": ["A"] * 20 + ["B"] * 10 + ["A"] * 5 + ["B"] * 25,
                "cell_id": list(range(30)) + list(range(100, 130)),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_filtered_obs_has_no_duplicates(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 50 + ["reference"] * 50,
                "batch": ["A"] * 25 + ["B"] * 25 + ["A"] * 25 + ["B"] * 25,
                "cell_id": range(100),
            }
        )

        can_generate, filtered = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert not filtered.index.duplicated().any()
        assert len(filtered) == len(filtered.drop_duplicates())

    def test_empty_obs_returns_false(self):
        obs = pd.DataFrame({"condition": pd.Series([], dtype=str), "batch": pd.Series([], dtype=str)})

        can_generate, filtered = _generate_samples(
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

    def test_categorical_dtype_in_stratification_columns(self):
        obs = pd.DataFrame(
            {
                "condition": pd.Categorical(["query"] * 20 + ["reference"] * 20),
                "batch": pd.Categorical(["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10),
                "cell_id": range(40),
            }
        )

        can_generate, filtered = _generate_samples(
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

    def test_missing_values_in_stratification_columns(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 20,
                "batch": ["A"] * 5 + [None] * 5 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
                "cell_id": range(40),
            }
        )

        try:
            can_generate, filtered = _generate_samples(
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

    def test_no_requirements_specified_returns_false(self):
        obs = pd.DataFrame(
            {"condition": ["query"] * 10 + ["reference"] * 10, "batch": ["A"] * 10 + ["A"] * 10, "cell_id": range(20)}
        )

        can_generate, filtered = _generate_samples(
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

    def test_filtered_obs_is_subset_of_input_obs(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 50 + ["reference"] * 50,
                "batch": np.random.choice(["A", "B", "C"], 100),
                "donor": np.random.choice(["D1", "D2", "D3", "D4"], 100),
                "cell_id": range(100),
            }
        )

        can_generate, filtered = _generate_samples(
            obs=obs,
            stratify_by=["batch", "donor"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert set(filtered.index).issubset(set(obs.index))

    def test_both_conditions_present_in_filtered_obs_when_successful(self):
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 30 + ["reference"] * 30,
                "batch": ["A"] * 15 + ["B"] * 15 + ["A"] * 15 + ["B"] * 15,
                "cell_id": range(60),
            }
        )

        can_generate, filtered = _generate_samples(
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
