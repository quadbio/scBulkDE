"""Tests for _validate_strata function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scbulkde.ut.ut_basic import _generate_samples, _validate_strata


class TestValidateStrata:
    def test_none_strata_returns_empty(self):
        """When strata is None, should return empty results."""
        obs = pd.DataFrame({"condition": ["query", "reference"], "batch": ["A", "A"]})

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=None,
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert valid_strata == []
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 0
        assert isinstance(sample_stats, pd.DataFrame)
        assert len(sample_stats) == 0

    def test_empty_strata_list_returns_empty(self):
        """When strata is empty list, should return empty results."""
        obs = pd.DataFrame({"condition": ["query", "reference"], "batch": ["A", "A"]})

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=[],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert valid_strata == []
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 0
        assert len(sample_stats) == 0

    def test_returns_strata_when_all_requirements_met(self):
        """When initial strata meet all requirements, should return them without dropping."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 20,
                "batch": ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
                "donor": ["D1"] * 10 + ["D2"] * 10 + ["D1"] * 10 + ["D2"] * 10,
                "cell_id": range(40),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch", "donor"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert set(valid_strata) == {"batch", "donor"}
        assert len(filtered) == 40

    def test_sequence_order_drops_last_covariate_first(self):
        """With sequence_order strategy, should drop from the end of the list first."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 40 + ["reference"] * 40,
                "batch": ["A"] * 20 + ["B"] * 20 + ["A"] * 20 + ["B"] * 20,
                "donor": ["D1"] * 5
                + ["D2"] * 5
                + ["D3"] * 5
                + ["D4"] * 5
                + ["D1"] * 5
                + ["D2"] * 5
                + ["D3"] * 5
                + ["D4"] * 5
                + ["D1"] * 5
                + ["D2"] * 5
                + ["D3"] * 5
                + ["D4"] * 5
                + ["D1"] * 5
                + ["D2"] * 5
                + ["D3"] * 5
                + ["D4"] * 5,
                "tissue": ["T1", "T2"] * 20 + ["T1", "T2"] * 20,
                "cell_id": range(80),
            }
        )

        valid_strata, _, _ = _validate_strata(
            obs=obs,
            strata=["batch", "donor", "tissue"],
            min_cells=15,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        # With all three strata, combinations are too small
        # batch alone has 20 cells per group, which meets min_cells=15
        # Should drop tissue first, then donor, leaving just batch
        assert valid_strata == ["batch"]

    def test_most_levels_drops_highest_cardinality_first(self):
        """With most_levels strategy, should drop covariate with most unique values first."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A", "B"] * 50 + ["A", "B"] * 50,  # 2 levels
                "donor": ["D1", "D2", "D3", "D4", "D5"] * 20 + ["D1", "D2", "D3", "D4", "D5"] * 20,  # 5 levels
                "tissue": ["T1"] * 100 + ["T1"] * 100,  # 1 level
                "cell_id": range(200),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch", "donor", "tissue"],
            min_cells=40,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="most_levels",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        # donor has 5 levels (most), should be dropped first
        # Then batch (2 levels) with 50 cells per group in each condition meets min_cells=40
        assert set(valid_strata) == {"batch", "tissue"}

    def test_does_not_modify_input_strata_list(self):
        """Should not mutate the input strata list."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 20,
                "batch": ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
                "donor": ["D1"] * 10 + ["D2"] * 10 + ["D1"] * 10 + ["D2"] * 10,
                "cell_id": range(40),
            }
        )

        input_strata = ["batch", "donor"]
        original_length = len(input_strata)

        _validate_strata(
            obs=obs,
            strata=input_strata,
            min_cells=100,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert len(input_strata) == original_length

    def test_resolve_conflicts_true_returns_empty_when_impossible(self):
        """With resolve_conflicts=True and impossible requirements, should return empty."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 10 + ["reference"] * 10,
                "batch": ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5,
                "cell_id": range(20),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=1000,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert valid_strata == []
        assert len(filtered) == 0

    def test_resolve_conflicts_false_raises_when_impossible(self):
        """With resolve_conflicts=False and impossible requirements, should raise."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 10 + ["reference"] * 10,
                "batch": ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5,
                "cell_id": range(20),
            }
        )

        with pytest.raises(ValueError):
            _validate_strata(
                obs=obs,
                strata=["batch"],
                min_cells=1000,
                min_fraction=None,
                min_coverage=None,
                qualify_strategy="and",
                covariate_strategy="sequence_order",
                group_key_internal="condition",
                resolve_conflicts=False,
            )

    def test_returns_subset_or_equal_of_input_strata(self):
        """Returned strata should be subset (or equal to) input strata."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 50 + ["reference"] * 50,
                "batch": np.random.choice(["A", "B", "C"], 100),
                "donor": np.random.choice(["D1", "D2"], 100),
                "tissue": np.random.choice(["T1", "T2"], 100),
                "cell_id": range(100),
            }
        )

        input_strata = ["batch", "donor", "tissue"]

        valid_strata, _, _ = _validate_strata(
            obs=obs,
            strata=input_strata,
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert set(valid_strata).issubset(set(input_strata))

    def test_filtered_obs_matches_generate_samples_output(self):
        """Filtered obs should match what _generate_samples returns for valid_strata."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 30 + ["reference"] * 30,
                "batch": ["A"] * 15 + ["B"] * 15 + ["A"] * 15 + ["B"] * 15,
                "donor": ["D1"] * 15 + ["D2"] * 15 + ["D1"] * 15 + ["D2"] * 15,
                "cell_id": range(60),
            }
        )

        valid_strata, filtered_validate, _ = _validate_strata(
            obs=obs,
            strata=["batch", "donor"],
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        can_gen, filtered_gen, _ = _generate_samples(
            obs=obs,
            stratify_by=valid_strata,
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        if len(valid_strata) > 0:
            assert can_gen is True
            assert len(filtered_validate) == len(filtered_gen)
            assert set(filtered_validate.index) == set(filtered_gen.index)

    def test_filtered_obs_only_contains_qualifying_groups(self):
        """All cells in filtered obs should be in groups that meet requirements."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 50 + ["reference"] * 50,
                "batch": ["A"] * 25 + ["B"] * 25 + ["A"] * 25 + ["B"] * 25,
                "donor": ["D1"] * 5
                + ["D2"] * 20
                + ["D3"] * 5
                + ["D4"] * 20
                + ["D1"] * 5
                + ["D2"] * 20
                + ["D3"] * 5
                + ["D4"] * 20,
                "cell_id": range(100),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch", "donor"],
            min_cells=15,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        # Only donor D2 and D4 in each batch have >=15 cells per condition
        assert len(filtered) == 80

        # Check each group in filtered meets min_cells requirement
        for condition in ["query", "reference"]:
            cond_filtered = filtered[filtered["condition"] == condition]
            if len(cond_filtered) > 0:
                for _, group_df in cond_filtered.groupby(valid_strata):
                    assert len(group_df) >= 15

    def test_coverage_requirement_enforced_in_final_result(self):
        """If min_coverage specified, filtered obs should meet it for each condition."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A"] * 80 + ["B"] * 20 + ["A"] * 80 + ["B"] * 20,
                "cell_id": range(200),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=50,
            min_fraction=None,
            min_coverage=0.7,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        if len(valid_strata) > 0:
            query_total = (obs["condition"] == "query").sum()
            ref_total = (obs["condition"] == "reference").sum()

            query_kept = len(filtered[filtered["condition"] == "query"])
            ref_kept = len(filtered[filtered["condition"] == "reference"])

            assert query_kept / query_total >= 0.7
            assert ref_kept / ref_total >= 0.7

    def test_both_conditions_represented_in_successful_result(self):
        """When returning valid strata, both query and reference must be in filtered obs."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 30 + ["reference"] * 30,
                "batch": ["A"] * 15 + ["B"] * 15 + ["A"] * 15 + ["B"] * 15,
                "cell_id": range(60),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        if len(valid_strata) > 0:
            assert "query" in filtered["condition"].values
            assert "reference" in filtered["condition"].values

    def test_impossible_requirements_with_multiple_strata(self):
        """Should try all combinations by dropping strata until finding valid or exhausting."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A", "B"] * 50 + ["A", "B"] * 50,  # 50 cells per level and condition
                "donor": ["D1", "D2", "D3", "D4"] * 25
                + ["D1", "D2", "D3", "D4"] * 25,  # 25 cells per level and condition
                "tissue": ["T1", "T2"] * 50 + ["T1", "T2"] * 50,  # 50 cells per level and condition
                "cell_id": range(200),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch", "donor", "tissue"],
            min_cells=80,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert valid_strata == []

    def test_with_only_min_fraction_requirement(self):
        """Should work correctly with only min_fraction specified."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A"] * 60 + ["B"] * 40 + ["A"] * 60 + ["B"] * 40,
                "cell_id": range(200),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=None,
            min_fraction=0.5,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        # batch A has 60/100 = 0.6 >= 0.5, qualifies
        # batch B has 40/100 = 0.4 < 0.5, doesn't qualify
        assert valid_strata == ["batch"]
        assert len(filtered) == 120  # Only batch A cells

    def test_with_both_min_cells_and_min_fraction_and_strategy(self):
        """Should correctly apply qualify_strategy when both constraints given."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 100 + ["reference"] * 100,
                "batch": ["A"] * 10 + ["B"] * 90 + ["A"] * 10 + ["B"] * 90,
                "cell_id": range(200),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=50,
            min_fraction=0.05,
            min_coverage=None,
            qualify_strategy="or",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        # With 'or': A has 10 cells (<50) but 0.1 fraction (>=0.05), qualifies
        #            B has 90 cells (>=50) and 0.9 fraction (>=0.05), qualifies
        assert len(valid_strata) > 0
        assert len(filtered) == 200

    def test_empty_result_when_one_condition_has_no_cells(self):
        """Should return empty when one condition has no cells."""
        obs = pd.DataFrame({"condition": ["query"] * 20, "batch": ["A"] * 10 + ["B"] * 10, "cell_id": range(20)})

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert valid_strata == []
        assert len(filtered) == 0

    def test_preserves_original_index_in_filtered_obs(self):
        """Filtered obs should preserve original DataFrame indices."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 20,
                "batch": ["A"] * 15 + ["B"] * 5 + ["A"] * 15 + ["B"] * 5,
                "cell_id": range(40),
            },
            index=range(1000, 1040),
        )

        _, filtered, _ = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert all(1000 <= idx < 1040 for idx in filtered.index)
        assert filtered.index.is_unique

    def test_no_duplicates_in_filtered_obs(self):
        """Filtered obs should not contain duplicate rows."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 30 + ["reference"] * 30,
                "batch": ["A"] * 15 + ["B"] * 15 + ["A"] * 15 + ["B"] * 15,
                "cell_id": range(60),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert not filtered.index.duplicated().any()

    def test_categorical_dtype_handled_correctly(self):
        """Should work with categorical dtypes in stratification columns."""
        obs = pd.DataFrame(
            {
                "condition": pd.Categorical(["query"] * 20 + ["reference"] * 20),
                "batch": pd.Categorical(["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10),
                "cell_id": range(40),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert len(valid_strata) > 0
        assert len(filtered) > 0

    def test_invalid_covariate_strategy_raises_error(self):
        """Should raise error for invalid covariate_strategy."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 20,
                "batch": ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
                "cell_id": range(40),
            }
        )

        with pytest.raises(ValueError):
            _validate_strata(
                obs=obs,
                strata=["batch"],
                min_cells=100,
                min_fraction=None,
                min_coverage=None,
                qualify_strategy="and",
                covariate_strategy="invalid_strategy",
                group_key_internal="condition",
                resolve_conflicts=False,
            )

    def test_sample_stats_returned_correctly(self):
        """Test that sample_stats is returned and contains expected columns."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 20,
                "batch": ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
                "cell_id": range(40),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=5,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert len(valid_strata) > 0
        assert isinstance(sample_stats, pd.DataFrame)
        assert "n_cells" in sample_stats.columns
        assert "n_cells_condition" in sample_stats.columns
        assert "fraction" in sample_stats.columns
        assert "coverage" in sample_stats.columns
