"""Tests for functions in the ut module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scbulkde.ut.ut_basic import (
    _aggregate_counts,
    _build_design_formula,
    _build_full_rank_design,
    _drop_covariate,
    _generate_samples,
    _prepare_internal_groups,
    _validate_strata,
)

# ==================== _prepare_internal_groups Tests ====================


class TestPrepareInternalGroups:
    """Tests for _prepare_internal_groups."""

    def test_subset_excludes_other_groups(self, make_adata):
        """Groups not in query or reference should be excluded."""
        adata = make_adata(n_cells=120, groups=["A", "B", "C", "D"], group_counts=[30, 30, 30, 30])

        obs = _prepare_internal_groups(
            adata=adata,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="A",
            reference="B",
        )

        # Only A and B cells should be in the output
        assert len(obs) == 60
        assert set(obs["cell_type"].unique()) == {"A", "B"}

    def test_reference_rest_excludes_query(self, make_adata):
        """When reference='rest', ensure query groups are excluded from reference."""
        adata = make_adata(
            n_cells=90,
            groups=["A", "B", "C"],
            group_counts=[30, 30, 30],
        )

        obs = _prepare_internal_groups(
            adata=adata,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="A",
            reference="rest",
        )

        # Check that 'A' cells are labeled 'query'
        query_cells = obs[obs["psbulk_condition"] == "query"]
        assert all(query_cells["cell_type"] == "A")

        # Check that 'B' and 'C' cells are labeled 'reference'
        ref_cells = obs[obs["psbulk_condition"] == "reference"]
        assert set(ref_cells["cell_type"].unique()) == {"B", "C"}

        # Ensure no 'A' cells in reference
        assert "A" not in ref_cells["cell_type"].values

    def test_reference_rest_with_multiple_query(self, make_adata):
        """reference='rest' with multiple query groups."""
        adata = make_adata(
            n_cells=120,
            groups=["A", "B", "C", "D"],
            group_counts=[30, 30, 30, 30],
        )

        obs = _prepare_internal_groups(
            adata=adata,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query=["A", "B"],
            reference="rest",
        )

        query_cells = obs[obs["psbulk_condition"] == "query"]
        ref_cells = obs[obs["psbulk_condition"] == "reference"]

        assert set(query_cells["cell_type"].unique()) == {"A", "B"}
        assert set(ref_cells["cell_type"].unique()) == {"C", "D"}

    def test_overlapping_query_and_reference_query_wins(self, make_adata):
        """When a group is in both query and reference, it should be labeled as query."""
        adata = make_adata(
            n_cells=90,
            groups=["A", "B", "C"],
            group_counts=[30, 30, 30],
        )

        # 'B' appears in both query and reference
        obs = _prepare_internal_groups(
            adata=adata,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query=["A", "B"],
            reference=["B", "C"],
        )

        # 'B' cells should be 'query' due to np.where logic
        b_cells = obs[obs["cell_type"] == "B"]
        assert all(b_cells["psbulk_condition"] == "query")

    def test_non_categorical_group_key_converted(self, make_adata):
        """Ensure non-categorical group_key is converted properly."""
        adata = make_adata(
            n_cells=60,
            groups=["A", "B"],
            group_counts=[30, 30],
            categorical_group_key=False,  # String dtype, not categorical
        )

        # Verify it's not categorical before
        assert not isinstance(adata.obs["cell_type"].dtype, pd.CategoricalDtype)

        obs = _prepare_internal_groups(
            adata=adata,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="A",
            reference="rest",
        )

        # Check that it has been converted to categorical
        assert isinstance(adata.obs["cell_type"].dtype, pd.CategoricalDtype)

        # Check proper output
        assert "psbulk_condition" in obs.columns

    def test_query_not_in_obs_raises(self, make_adata):
        """Query values not present in data should raise ValueError."""
        adata = make_adata(
            n_cells=60,
            groups=["A", "B"],
            group_counts=[30, 30],
        )

        with pytest.raises(ValueError, match="No cells found for query groups"):
            _prepare_internal_groups(
                adata=adata,
                group_key="cell_type",
                group_key_internal="psbulk_condition",
                query="NonExistent",
                reference="rest",
            )

    def test_reference_not_in_obs_raises(self, make_adata):
        """Reference values not present in data should raise ValueError."""
        adata = make_adata(
            n_cells=60,
            groups=["A", "B"],
            group_counts=[30, 30],
        )

        with pytest.raises(ValueError, match="No cells found for reference groups"):
            _prepare_internal_groups(
                adata=adata,
                group_key="cell_type",
                group_key_internal="psbulk_condition",
                query="A",
                reference="NonExistent",
            )

    def test_query_and_reference_as_string(self, make_adata):
        """Test query and reference provided as strings."""
        adata = make_adata(
            n_cells=80,
            groups=["A", "B", "C"],
            group_counts=[30, 30, 20],
        )

        obs = _prepare_internal_groups(
            adata=adata,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="A",
            reference="C",
        )

        query_cells = obs[obs["psbulk_condition"] == "query"]
        ref_cells = obs[obs["psbulk_condition"] == "reference"]

        assert all(query_cells["cell_type"] == "A")
        assert all(ref_cells["cell_type"] == "C")

        assert len(query_cells) == 30
        assert len(ref_cells) == 20

    def test_query_and_reference_as_list(self, make_adata):
        """Test query and reference provided as lists."""
        adata = make_adata(
            n_cells=100,
            groups=["A", "B", "C", "D", "E"],
            group_counts=[10, 10, 20, 30, 30],
        )

        obs = _prepare_internal_groups(
            adata=adata,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query=["A", "B"],
            reference=["D", "E"],
        )

        query_cells = obs[obs["psbulk_condition"] == "query"]
        ref_cells = obs[obs["psbulk_condition"] == "reference"]

        assert set(query_cells["cell_type"].unique()) == {"A", "B"}
        assert set(ref_cells["cell_type"].unique()) == {"D", "E"}

        assert len(query_cells) == 20
        assert len(ref_cells) == 60

    def test_preserves_other_columns(self, make_adata):
        """Other obs columns should be preserved in output."""
        adata = make_adata(
            n_cells=60,
            groups=["A", "B"],
            group_counts=[30, 30],
            replicate_key="batch",
            continuous_covariates={"age": list(range(60))},
        )

        obs = _prepare_internal_groups(
            adata=adata,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="A",
            reference="B",
        )

        assert "batch" in obs.columns
        assert "age" in obs.columns


# ==================== _validate_strata Tests ====================


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


# ==================== _generate_samples Tests ====================


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


# ==================== _build_design_formula Tests ====================


class TestBuildDesign:
    """Tests for _build_design_formula formula construction."""

    def test_minimal_design_only_condition(self):
        """With no covariates, should only include condition with reference base."""
        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=[],
            factors_continuous=[],
        )

        assert formula == "C(psbulk_condition, contr.treatment(base='reference'))"

    def test_with_categorical_covariates(self):
        """Should add categorical covariates with C() wrapper."""
        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=["batch", "donor"],
            factors_continuous=[],
        )

        assert "C(psbulk_condition, contr.treatment(base='reference'))" in formula
        assert "C(batch)" in formula
        assert "C(donor)" in formula
        assert " + " in formula

    def test_with_continuous_covariates(self):
        """Should add continuous covariates without C() wrapper."""
        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=[],
            factors_continuous=["age", "weight"],
        )

        assert "C(psbulk_condition, contr.treatment(base='reference'))" in formula
        assert "age" in formula
        assert "weight" in formula
        assert "C(age)" not in formula  # Should NOT wrap continuous
        assert " + " in formula

    def test_with_both_categorical_and_continuous(self):
        """Should handle both types of covariates."""
        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=["batch"],
            factors_continuous=["age"],
        )

        assert "C(psbulk_condition, contr.treatment(base='reference'))" in formula
        assert "C(batch)" in formula
        assert "age" in formula
        # Ensure age is not wrapped
        assert formula.count("age") == 1
        assert "C(age)" not in formula

    def test_term_order(self):
        """Condition should come first, then categorical, then continuous."""
        formula = _build_design_formula(
            group_key_internal="condition",
            factors_categorical=["cat1", "cat2"],
            factors_continuous=["cont1", "cont2"],
        )

        terms = formula.split(" + ")
        assert terms[0] == "C(condition, contr.treatment(base='reference'))"
        assert "C(cat1)" in terms[1:3]
        assert "C(cat2)" in terms[1:3]
        assert "cont1" in terms[3:5]
        assert "cont2" in terms[3:5]


class TestBuildDesignIntegration:
    """Integration tests with formulaic model_matrix."""

    def test_formula_parseable_by_formulaic(self):
        """Generated formula should be parseable by formulaic."""
        import pandas as pd
        from formulaic import model_matrix

        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=["batch"],
            factors_continuous=["age"],
        )

        # Create sample data
        data = pd.DataFrame(
            {
                "psbulk_condition": ["query", "reference", "query", "reference"],
                "batch": ["b0", "b0", "b1", "b1"],
                "age": [25, 30, 35, 40],
            }
        )

        # Should not raise
        mm = model_matrix(formula, data=data)
        assert mm.shape[0] == 4
        # Should have: Intercept, condition[query], batch[T.b1], age
        assert mm.shape[1] >= 3

    def test_formula_reference_level_correct(self):
        """Reference level should be 'reference' for condition."""
        import pandas as pd
        from formulaic import model_matrix

        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=[],
            factors_continuous=[],
        )

        data = pd.DataFrame(
            {
                "psbulk_condition": ["query", "reference"],
            }
        )

        mm = model_matrix(formula, data=data)

        # When psbulk_condition is 'reference', the coefficient should be 0
        # When 'query', coefficient should be 1
        # Check column names
        assert any("query" in str(col) for col in mm.columns)


# ==================== _drop_covariate Tests ====================


class TestDropCovariate:
    """Tests for _drop_covariate."""

    def test_sequence_order_pops_last(self):
        """sequence_order should drop the last covariate."""
        covariates = ["batch", "donor", "time"]
        obs = pd.DataFrame(
            {
                "batch": ["b0", "b1", "b0", "b1"],
                "donor": ["d0", "d0", "d1", "d1"],
                "time": ["t0", "t1", "t0", "t1"],
            }
        )

        result_covs, dropped = _drop_covariate(
            covariates=covariates,
            obs=obs,
            covariate_strategy="sequence_order",
        )

        assert dropped == "time"
        assert result_covs == ["batch", "donor"]

    def test_sequence_order_successive_drops(self):
        """Multiple sequence_order drops should remove from end."""
        covariates = ["a", "b", "c", "d"]
        obs = pd.DataFrame({"a": [1], "b": [1], "c": [1], "d": [1]})

        _, d1 = _drop_covariate(covariates.copy(), obs, "sequence_order")
        assert d1 == "d"

        covariates = ["a", "b", "c"]
        _, d2 = _drop_covariate(covariates.copy(), obs, "sequence_order")
        assert d2 == "c"

    def test_most_levels_drops_highest(self):
        """most_levels should drop covariate with most unique values."""
        covariates = ["batch", "donor", "time"]
        obs = pd.DataFrame(
            {
                "batch": ["b0", "b1", "b2"] * 10,  # 3 levels
                "donor": [f"d{i}" for i in range(10)] * 3,  # 10 levels
                "time": ["t0", "t1"] * 15,  # 2 levels
            }
        )

        result_covs, dropped = _drop_covariate(
            covariates=covariates,
            obs=obs,
            covariate_strategy="most_levels",
        )

        assert dropped == "donor"  # Has 10 levels
        assert result_covs == ["batch", "time"]

    def test_most_levels_tiebreaker_first_occurrence(self):
        """When multiple covariates have same number of levels, np.argmax returns first."""
        covariates = ["a", "b", "c"]
        obs = pd.DataFrame(
            {
                "a": ["x", "y", "z"] * 10,  # 3 levels
                "b": ["p", "q", "r"] * 10,  # 3 levels (tie)
                "c": ["i", "j"] * 15,  # 2 levels
            }
        )

        result_covs, dropped = _drop_covariate(
            covariates=covariates,
            obs=obs,
            covariate_strategy="most_levels",
        )

        # np.argmax returns first occurrence of max, so 'a' should be dropped
        assert dropped == "a"
        assert result_covs == ["b", "c"]

    def test_invalid_strategy_raises(self):
        """Invalid covariate_strategy should raise ValueError."""
        covariates = ["a", "b"]
        obs = pd.DataFrame({"a": [1], "b": [1]})

        with pytest.raises(ValueError, match="Unknown covariate_strategy"):
            _drop_covariate(covariates, obs, "invalid_strategy")

    def test_single_covariate(self):
        """Dropping from single-element list."""
        covariates = ["only_one"]
        obs = pd.DataFrame({"only_one": [1, 2, 3]})

        result_covs, dropped = _drop_covariate(covariates, obs, "sequence_order")

        assert dropped == "only_one"
        assert result_covs == []

    def test_empty_covariates_raises(self):
        """Dropping from empty list should raise IndexError."""
        covariates = []
        obs = pd.DataFrame({"a": [1]})

        with pytest.raises(IndexError):
            _drop_covariate(covariates, obs, "sequence_order")


# ==================== _aggregate_counts Tests ====================


class TestAggregateCounts:
    """Tests for _aggregate_counts."""

    def test_aggregate_counts_dense_sum(self, make_adata):
        """Verify sum aggregation on dense matrix."""
        adata = make_adata(
            n_cells=6,
            n_genes=3,
            groups=["A", "B"],
            group_counts=[3, 3],
            sparse=False,
        )

        # Set known values
        adata.X = np.array(
            [
                [1, 2, 3],  # A
                [4, 5, 6],  # A
                [7, 8, 9],  # A
                [10, 11, 12],  # B
                [13, 14, 15],  # B
                [16, 17, 18],  # B
            ],
            dtype=np.float32,
        )

        grouped = adata.obs.groupby("cell_type", observed=True, sort=False)
        result = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="sum")

        # A: sum = [1+4+7, 2+5+8, 3+6+9] = [12, 15, 18]
        # B: sum = [10+13+16, 11+14+17, 12+15+18] = [39, 42, 45]
        expected_A = [12, 15, 18]
        expected_B = [39, 42, 45]

        # Check shape
        assert result.shape == (2, 3)

        # Check values (order might vary based on groupby)
        assert result.values.sum() == sum(expected_A) + sum(expected_B)

    def test_aggregate_counts_dense_mean(self, make_adata):
        """Verify mean aggregation on dense matrix."""
        adata = make_adata(
            n_cells=6,
            n_genes=3,
            groups=["A", "B"],
            group_counts=[3, 3],
            sparse=False,
        )

        adata.X = np.array(
            [
                [3, 6, 9],  # A
                [3, 6, 9],  # A
                [3, 6, 9],  # A
                [12, 15, 18],  # B
                [12, 15, 18],  # B
                [12, 15, 18],  # B
            ],
            dtype=np.float32,
        )

        grouped = adata.obs.groupby("cell_type", observed=True, sort=False)
        result = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="mean")

        # Expected sums:
        expected = np.array(
            [
                [3, 6, 9],  # A mean
                [12, 15, 18],  # B mean
            ],
            dtype=np.float32,
        )

        # Check that mean was computed correctly
        assert np.array_equal(result.values, expected)

    def test_aggregate_counts_uses_layer(self, make_adata):
        """Verify layer parameter correctly selects the right matrix."""
        adata = make_adata(
            n_cells=4,
            n_genes=2,
            groups=["A", "B"],
            group_counts=[2, 2],
            sparse=False,
            layer_name="raw",
        )

        # X and layer have different values
        adata.X = np.array([[1, 1], [1, 1], [1, 1], [1, 1]], dtype=np.float32)
        adata.layers["raw"] = np.array([[10, 10], [10, 10], [10, 10], [10, 10]], dtype=np.float32)

        grouped = adata.obs.groupby("cell_type", observed=True, sort=False)

        # Using X
        result_x = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="sum")
        # Using layer
        result_layer = _aggregate_counts(adata, grouped, layer="raw", layer_aggregation="sum")

        # Results should be different
        assert result_x.values.sum() != result_layer.values.sum()
        assert result_x.values.sum() == 8  # 4 cells * 1 * 2 genes
        assert result_layer.values.sum() == 80  # 4 cells * 10 * 2 genes

    def test_aggregate_counts_shape(self, make_adata):
        """Output should have n_groups rows and n_genes columns."""
        n_genes = 50
        adata = make_adata(
            n_cells=100,
            n_genes=n_genes,
            groups=["A", "B", "C", "D", "E"],
            group_counts=[20, 20, 20, 20, 20],
            sparse=False,
        )

        grouped = adata.obs.groupby("cell_type", observed=True, sort=False)
        result = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="sum")

        assert result.shape[0] == 5  # 5 groups
        assert result.shape[1] == n_genes

    def test_aggregate_counts_invalid_aggregation(self, make_adata):
        """Invalid layer_aggregation should raise ValueError."""
        adata = make_adata(n_cells=10, n_genes=5, sparse=False)
        grouped = adata.obs.groupby("cell_type", observed=True, sort=False)

        with pytest.raises(ValueError, match="Invalid layer_aggregation"):
            _aggregate_counts(adata, grouped, layer=None, layer_aggregation="median")


# ==================== _build_full_rank_design Tests ====================


class TestBuildFullRankDesign:
    """Tests for _build_full_rank_design logic."""

    @pytest.fixture
    def sample_table_full_rank(self):
        """Sample table that should produce full rank design."""
        return pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference"],
                "batch": ["b0", "b1", "b0", "b1"],
            }
        )

    @pytest.fixture
    def sample_table_rank_deficient(self):
        """Sample table that creates rank deficiency with all covariates."""
        # When condition and batch are perfectly confounded
        return pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference"],
                "batch": ["b0", "b0", "b1", "b1"],  # Confounded with condition
            }
        )

    def test_full_rank_design_returns_immediately(self, sample_table_full_rank):
        """When design is full rank, should return without dropping."""

        formula, mm, _, _ = _build_full_rank_design(
            sample_table=sample_table_full_rank,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["batch"],
            design_factors_continuous=[],
            covariate_strategy="sequence_order",
        )

        # Should have full rank
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]
        # Should include batch
        assert "batch" in formula

    def test_rank_deficient_drops_categorical_first(self, sample_table_rank_deficient):
        """When rank deficient, should drop categorical covariates first."""

        formula, mm, _, _ = _build_full_rank_design(
            sample_table=sample_table_rank_deficient,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["batch"],
            design_factors_continuous=[],
            covariate_strategy="sequence_order",
        )

        # Should drop batch to achieve full rank
        assert "C(batch)" not in formula
        # Should still have condition
        assert "psbulk_condition" in formula
        # Should have full rank
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]

    def test_drops_continuous_after_categorical(self):
        """Should drop categorical first, then continuous."""

        # Create data where we have rank deficiency
        sample_table = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 4 + ["reference"] * 4,
                "cat1": ["c0", "c1"] * 4,  # Not correlated with condition
                "cat2": ["c0", "c1"] * 4,  # Correlated with cat1
                "cont1": [1.0, 2.0] * 4,  # Not correlated with condition
            }
        )

        # The initial model matrix looks like:

        #   Intercept  [T.query]  C(cat1)[T.c1]  C(cat2)[T.c1]  cont1
        # 0        1.0          1              0              0    1.0
        # 1        1.0          1              1              1    2.0
        # 2        1.0          1              0              0    1.0
        # 3        1.0          1              1              1    2.0
        # 4        1.0          0              0              0    1.0
        # 5        1.0          0              1              1    2.0
        # 6        1.0          0              0              0    1.0

        # Then cat1 and cat2 are perfectly correlated, so one must be dropped.
        # The covariate strategy is sequence_order, so cat2 should be dropped first.
        # The model matrix after dropping cat2:

        #   Intercept  [T.query]  C(cat1)[T.c1]  cont1
        # 0        1.0          1              0    1.0
        # 1        1.0          1              1    2.0
        # 2        1.0          1              0    1.0
        # 3        1.0          1              1    2.0
        # 4        1.0          0              0    1.0
        # 5        1.0          0              1    2.0
        # 6        1.0          0              0    1.0

        # The model matrix is still not full rank, because cont1 = Intercept + C(cat1)[T.c1]
        # So cat1 should be dropped next, resulting in a full rank model matrix

        formula, mm, _, _ = _build_full_rank_design(
            sample_table=sample_table,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["cat1", "cat2"],
            design_factors_continuous=["cont1"],
            covariate_strategy="sequence_order",
        )

        # Should achieve full rank
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]

        # Check the correct covariates were dropped
        assert "psbulk_condition" in formula
        assert "C(cat1)" not in formula
        assert "C(cat2)" not in formula
        assert "cont1" in formula

    def test_covariate_strategy_sequence_order(self):
        """sequence_order should drop from end of list first."""

        sample_table = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 2 + ["reference"] * 2,
                "first": ["a", "b", "a", "b"],
                "second": ["x", "x", "y", "y"],  # Confounded
            }
        )

        formula, mm, _, _ = _build_full_rank_design(
            sample_table=sample_table,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["first", "second"],
            design_factors_continuous=[],
            covariate_strategy="sequence_order",
        )

        # Should drop 'second' first (last in sequence)
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]
        assert "psbulk_condition" in formula
        assert "C(first)" in formula
        assert "C(second)" not in formula

    def test_fallback_to_minimal_design(self):
        """If all covariates dropped, should return minimal design with just condition."""

        # Create highly confounded data
        sample_table = pd.DataFrame(
            {
                "psbulk_condition": ["query", "reference"],
                "cat": ["a", "b"],  # Perfectly confounded with condition
            }
        )

        formula, mm, _, _ = _build_full_rank_design(
            sample_table=sample_table,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["cat"],
            design_factors_continuous=[],
            covariate_strategy="sequence_order",
        )

        # Should only have condition
        assert "psbulk_condition" in formula
        assert "C(cat)" not in formula
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]
