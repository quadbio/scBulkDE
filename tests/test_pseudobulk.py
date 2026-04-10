"""Test suite for pseudobulk, _build_empty_pseudobulk_result, and _build_pseudobulk_result.

These tests are designed based on expected logical behavior, not just to pass based on
current implementation. Tests may fail if there are bugs in the implementation.
"""

from __future__ import annotations

# Assume these imports work in the test environment
import numpy as np
import pandas as pd
import pytest

from scbulkde.pp.pp_basic import (
    _build_pseudobulk_result,
    pseudobulk,
)
from scbulkde.ut._containers import PseudobulkResult


class TestPseudobulk:
    """Tests for the main pseudobulk function."""

    def test_basic_functionality(self, adata_balanced):
        """Test that pseudobulk returns a PseudobulkResult with expected structure."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=5,
            min_fraction=0.01,
        )

        assert isinstance(result, PseudobulkResult)
        assert result.group_key == "cell_type"
        assert result.group_key_internal == "psbulk_condition"
        assert result.query == ["TypeA"] or result.query == "TypeA"

    def test_query_reference_cell_selection(self, adata_balanced):
        """Test that only query and reference cells are retained."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=1,
        )

        # adata_sub should only contain TypeA and TypeB cells
        cell_types_in_result = result.adata_sub.obs["cell_type"].unique()
        assert set(cell_types_in_result) == {"TypeA", "TypeB"}
        assert "TypeC" not in cell_types_in_result

    def test_reference_rest(self, adata_balanced):
        """Test reference='rest' includes all non-query groups."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="rest",
            replicate_key="batch",
            min_cells=1,
        )

        # Reference should include TypeB and TypeC
        cell_types_in_result = result.adata_sub.obs["cell_type"].unique()
        assert "TypeA" in cell_types_in_result
        assert "TypeB" in cell_types_in_result or "TypeC" in cell_types_in_result

    def test_multiple_query_groups(self, adata_balanced):
        """Test with multiple query groups."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query=["TypeA", "TypeB"],
            reference="TypeC",
            replicate_key="batch",
            min_cells=1,
        )

        # All should be included
        cell_types_in_result = set(result.adata_sub.obs["cell_type"].unique())
        assert {"TypeA", "TypeB", "TypeC"}.issubset(cell_types_in_result)

    def test_no_strata_returns_empty_counts(self, adata_balanced):
        """Test that without strata, pb_counts is empty."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key=None,
            categorical_covariates=None,
            min_cells=5,
        )

        # Should return empty pseudobulk result
        assert result.strata == []
        assert len(result.pb_counts) == 0
        assert list(result.pb_counts.columns) == list(adata_balanced.var_names)

        # Check collapsed column exists and is True for all rows
        assert "collapsed" in result.sample_table.columns
        assert result.sample_table["collapsed"].all()
        assert result.collapsed

    def test_strata_creates_samples(self, adata_balanced):
        """Test that valid strata create proper samples."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=1,
            min_coverage=0.1,
        )

        # Should have strata and samples
        assert len(result.strata) > 0
        assert len(result.pb_counts) > 0

        # Number of samples should match sample_table rows
        assert len(result.pb_counts) == len(result.sample_table)

        # Check collapsed column exists and is False for all rows
        assert "collapsed" in result.sample_table.columns
        assert not result.sample_table["collapsed"].any()
        assert not result.collapsed

    def test_single_stratum_produces_valid_samples_not_collapsed(self, adata_single_stratum):
        """Test that single stratum per condition produces valid samples (not collapsed).

        Critical distinction: valid strata that produce only one sample per condition
        should have collapsed=False because these are proper independent samples.
        """
        result = pseudobulk(
            adata_single_stratum,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=1,
        )

        assert len(result.strata) > 0
        assert len(result.sample_table) == 2
        assert len(result.pb_counts) == 2

        # Critical: collapsed should be False
        assert "collapsed" in result.sample_table.columns
        assert not result.sample_table["collapsed"].any()
        assert not result.collapsed

    def test_sample_table_has_statistics_columns(self, adata_balanced):
        """Test that sample_table contains the new statistics columns."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=1,
        )

        if result.strata:
            assert "n_cells" in result.sample_table.columns
            assert "n_cells_condition" in result.sample_table.columns
            assert "fraction" in result.sample_table.columns
            assert "coverage" in result.sample_table.columns

    def test_sample_table_rows_match_groups(self, adata_balanced):
        """Test that sample_table has one row per unique sample group."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=1,
        )

        if result.strata:
            # Each row should be a unique combination of condition + strata
            _groupby_cols = ["psbulk_condition"] + list(result.strata)
            expected_groups = result.grouped.ngroups
            assert len(result.sample_table) == expected_groups

    def test_design_formula_excludes_replicate_key(self, adata_balanced):
        """Test that replicate_key is not in the design formula."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            categorical_covariates=["donor"],
            min_cells=1,
        )

        # batch (replicate_key) should be in strata but NOT in design formula
        assert "batch" in result.strata
        assert "batch" not in result.design_formula

    def test_continuous_covariates_aggregated(self, adata_balanced):
        """Test that continuous covariates are aggregated in sample table."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            continuous_covariates=["age"],
            continuous_aggregation="mean",
            min_cells=1,
        )

        if result.strata:
            assert "age" in result.sample_table.columns
            # Age should be aggregated (mean of cells in each sample)
            assert result.sample_table["age"].dtype in [np.float64, np.float32, float]

    def test_qualifying_cells_filtered(self, adata_unbalanced):
        """Test that only cells in qualifying groups are used."""
        # With min_cells=20, small groups should be excluded
        result = pseudobulk(
            adata_unbalanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=20,
            min_coverage=0.5,
        )

        # adata_sub should have fewer cells than original if some groups are filtered
        # The exact count depends on which groups qualify
        if result.strata:
            original_cells = 200  # TypeA + TypeB
            # Some cells should be filtered out
            assert len(result.adata_sub) <= original_cells

    def test_raises_on_no_query_cells(self, adata_balanced):
        """Test that ValueError is raised if no query cells exist."""
        with pytest.raises(ValueError, match="No cells found for query"):
            pseudobulk(
                adata_balanced,
                group_key="cell_type",
                query="NonexistentType",
                reference="TypeB",
            )

    def test_raises_on_no_reference_cells(self, adata_balanced):
        """Test that ValueError is raised if no reference cells exist."""
        with pytest.raises(ValueError, match="No cells found for reference"):
            pseudobulk(
                adata_balanced,
                group_key="cell_type",
                query="TypeA",
                reference="NonexistentType",
            )

    def test_sparse_matrix_handling(self, adata_sparse):
        """Test that sparse matrices are handled correctly."""
        result = pseudobulk(
            adata_sparse,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=1,
        )

        # pb_counts should have numeric values
        if len(result.pb_counts) > 0:
            assert not result.pb_counts.isna().all().all()


class TestBuildPseudobulkResult:
    """Tests for _build_pseudobulk_result."""

    @pytest.fixture
    def prepared_data(self, adata_balanced):
        """Create prepared obs and adata_sub with internal groups."""
        obs = adata_balanced.obs.copy()
        mask = obs["cell_type"].isin(["TypeA", "TypeB"])
        obs = obs[mask].copy()
        obs["psbulk_condition"] = np.where(obs["cell_type"] == "TypeA", "query", "reference")
        adata_sub = adata_balanced[obs.index, :]
        return adata_sub, obs

    @pytest.fixture
    def sample_stats_for_batch(self, prepared_data):
        """Create sample_stats DataFrame for batch stratification."""
        _, obs = prepared_data
        # Create sample stats with batch stratification
        sample_stats = pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference"],
                "batch": ["batch1", "batch2", "batch1", "batch2"],
                "n_cells": [20, 20, 15, 15],
                "n_cells_condition": [40, 40, 30, 30],
                "fraction": [0.5, 0.5, 0.5, 0.5],
                "coverage": [1.0, 1.0, 1.0, 1.0],
            }
        )
        return sample_stats

    @pytest.fixture
    def sample_stats_for_batch_donor(self, prepared_data):
        """Create sample_stats DataFrame for batch+donor stratification."""
        sample_stats = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 8 + ["reference"] * 6,
                "batch": ["batch1"] * 4 + ["batch2"] * 4 + ["batch1"] * 3 + ["batch2"] * 3,
                "donor": ["donor1", "donor2", "donor3", "donor4"] * 2 + ["donor1", "donor2", "donor3"] * 2,
                "n_cells": [5] * 14,
                "n_cells_condition": [40] * 8 + [30] * 6,
                "fraction": [0.125] * 8 + [0.167] * 6,
                "coverage": [1.0] * 14,
            }
        )
        return sample_stats

    def test_pb_counts_has_samples(self, prepared_data, sample_stats_for_batch):
        """Test that pb_counts has rows (samples)."""
        adata_sub, obs = prepared_data

        result = _build_pseudobulk_result(
            adata_sub=adata_sub,
            obs=obs,
            strata=["batch"],
            sample_stats=sample_stats_for_batch,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            layer=None,
            layer_aggregation="sum",
            categorical_covariates=None,
            continuous_covariates=None,
            continuous_aggregation=None,
            covariate_strategy="sequence_order",
            min_cells=1,
            min_fraction=0.01,
            min_coverage=0.1,
            qualify_strategy="or",
            n_cells=pd.Series({"query": 40, "reference": 30}),
        )

        assert len(result.pb_counts) > 0

    def test_sample_table_has_statistics_columns(self, prepared_data, sample_stats_for_batch):
        """Test that sample_table contains the statistics columns from sample_stats."""
        adata_sub, obs = prepared_data

        result = _build_pseudobulk_result(
            adata_sub=adata_sub,
            obs=obs,
            strata=["batch"],
            sample_stats=sample_stats_for_batch,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            layer=None,
            layer_aggregation="sum",
            categorical_covariates=None,
            continuous_covariates=None,
            continuous_aggregation=None,
            covariate_strategy="sequence_order",
            min_cells=1,
            min_fraction=0.01,
            min_coverage=0.1,
            qualify_strategy="or",
            n_cells=pd.Series({"query": 40, "reference": 30}),
        )

        assert "n_cells" in result.sample_table.columns
        assert "n_cells_condition" in result.sample_table.columns
        assert "fraction" in result.sample_table.columns
        assert "coverage" in result.sample_table.columns

    def test_design_matrix_full_rank(self, prepared_data, sample_stats_for_batch):
        """Test that design matrix is full rank."""
        adata_sub, obs = prepared_data

        result = _build_pseudobulk_result(
            adata_sub=adata_sub,
            obs=obs,
            strata=["batch"],
            sample_stats=sample_stats_for_batch,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            layer=None,
            layer_aggregation="sum",
            categorical_covariates=None,
            continuous_covariates=None,
            continuous_aggregation=None,
            covariate_strategy="sequence_order",
            min_cells=1,
            min_fraction=0.01,
            min_coverage=0.1,
            qualify_strategy="or",
            n_cells=pd.Series({"query": 40, "reference": 30}),
        )

        rank = np.linalg.matrix_rank(result.design_matrix.values)
        assert rank == result.design_matrix.shape[1]

    def test_design_drops_covariates_for_rank(self, adata_confounded):
        """Test that covariates are dropped if they cause rank deficiency."""
        obs = adata_confounded.obs.copy()
        mask = obs["cell_type"].isin(["TypeA", "TypeB"])
        obs = obs[mask].copy()
        obs["psbulk_condition"] = np.where(obs["cell_type"] == "TypeA", "query", "reference")
        adata_sub = adata_confounded[obs.index, :]

        # Create sample_stats for confounded data
        sample_stats = pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference"],
                "batch": ["batch1", "batch1", "batch2", "batch2"],
                "donor": ["donor1", "donor2", "donor3", "donor4"],
                "n_cells": [25, 25, 25, 25],
                "n_cells_condition": [50, 50, 50, 50],
                "fraction": [0.5, 0.5, 0.5, 0.5],
                "coverage": [1.0, 1.0, 1.0, 1.0],
            }
        )

        # batch is perfectly confounded with condition
        result = _build_pseudobulk_result(
            adata_sub=adata_sub,
            obs=obs,
            strata=["batch", "donor"],
            sample_stats=sample_stats,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="TypeA",
            reference="TypeB",
            replicate_key=None,
            layer=None,
            layer_aggregation="sum",
            categorical_covariates=["batch", "donor"],
            continuous_covariates=None,
            continuous_aggregation=None,
            covariate_strategy="sequence_order",
            min_cells=1,
            min_fraction=0.01,
            min_coverage=0.1,
            qualify_strategy="or",
            n_cells=pd.Series({"query": 50, "reference": 50}),
        )

        # Design matrix should be full rank (covariates dropped if needed)
        rank = np.linalg.matrix_rank(result.design_matrix.values)
        assert rank == result.design_matrix.shape[1]

        # batch should NOT be in design (confounded with condition)
        # The function should drop it
        assert "batch" not in result.design_formula

    def test_aggregation_sum(self, prepared_data, sample_stats_for_batch):
        """Test sum aggregation produces correct totals."""
        adata_sub, obs = prepared_data

        result = _build_pseudobulk_result(
            adata_sub=adata_sub,
            obs=obs,
            strata=["batch"],
            sample_stats=sample_stats_for_batch,
            group_key="cell_type",
            group_key_internal="psbulk_condition",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            layer=None,
            layer_aggregation="sum",
            categorical_covariates=None,
            continuous_covariates=None,
            continuous_aggregation=None,
            covariate_strategy="sequence_order",
            min_cells=1,
            min_fraction=0.01,
            min_coverage=0.1,
            qualify_strategy="or",
            n_cells=pd.Series({"query": 40, "reference": 30}),
        )

        # Verify that sums are non-negative (counts can't be negative)
        assert (result.pb_counts.values >= 0).all()

        # Total pseudobulk counts should equal total cell counts
        total_pb = result.pb_counts.sum().sum()
        total_cells = adata_sub.X.sum()
        np.testing.assert_almost_equal(total_pb, total_cells, decimal=5)


# ==================== Edge Case Tests ====================


class TestEdgeCases:
    """Tests for edge cases and potential bugs."""

    def test_query_reference_overlap_assigns_to_query(self, adata_balanced):
        """Test that overlapping groups are assigned to query."""
        # Modify adata to have overlapping groups
        adata_balanced.obs["cell_type"] = pd.Categorical(list(adata_balanced.obs["cell_type"]))

        # This should warn and assign overlap to query
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query=["TypeA", "TypeB"],
            reference=["TypeB", "TypeC"],  # TypeB overlaps
            replicate_key="batch",
            min_cells=1,
        )

        # TypeB cells should be in query, not reference
        typeB_mask = result.adata_sub.obs["cell_type"] == "TypeB"
        typeB_conditions = result.grouped.obj.loc[typeB_mask, "psbulk_condition"].unique()
        # Current implementation: overlap goes to query
        assert "query" in typeB_conditions

    def test_empty_strata_after_filtering(self, adata_unbalanced):
        """Test behavior when all strata are filtered out."""
        # Very strict criteria that no group can meet
        result = pseudobulk(
            adata_unbalanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=500,  # No group has this many cells
            min_fraction=0.9,
            qualify_strategy="and",
        )

        # Should fall back to empty result
        assert result.strata == []
        assert len(result.pb_counts) == 0

    def test_many_strata_columns(self, adata_balanced):
        """Test with many stratification columns."""
        # Add more columns
        adata_balanced.obs["tissue"] = pd.Categorical(np.random.choice(["brain", "heart"], len(adata_balanced)))
        adata_balanced.obs["sex"] = pd.Categorical(np.random.choice(["M", "F"], len(adata_balanced)))

        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            categorical_covariates=["donor", "tissue", "sex"],
            min_cells=1,
            min_coverage=0.01,
        )

        # All strata should be preserved (if they pass validation)
        assert "batch" in result.strata

    def test_sample_table_statistics_values_correct(self, adata_balanced):
        """Test that sample_table statistics columns have reasonable values."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=1,
        )

        if result.strata:
            # n_cells should be positive
            assert (result.sample_table["n_cells"] > 0).all()
            # n_cells_condition should be >= n_cells
            assert (result.sample_table["n_cells_condition"] >= result.sample_table["n_cells"]).all()
            # fraction should be between 0 and 1
            assert (result.sample_table["fraction"] > 0).all()
            assert (result.sample_table["fraction"] <= 1).all()
            # coverage should be between 0 and 1
            assert (result.sample_table["coverage"] >= 0).all()
            assert (result.sample_table["coverage"] <= 1).all()


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests ensuring the full pipeline works correctly."""

    def test_full_pipeline_with_all_options(self, adata_balanced):
        """Test full pipeline with all options specified."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference=["TypeB", "TypeC"],
            replicate_key="batch",
            categorical_covariates=["donor"],
            continuous_covariates=["age"],
            continuous_aggregation="mean",
            min_cells=5,
            min_fraction=0.05,
            min_coverage=0.5,
            layer=None,
            layer_aggregation="sum",
            qualify_strategy="or",
            covariate_strategy="sequence_order",
            resolve_conflicts=True,
        )

        # Basic sanity checks
        assert isinstance(result, PseudobulkResult)
        assert result.adata_sub.n_obs > 0

        # If strata exist, samples should exist
        if result.strata:
            assert len(result.pb_counts) > 0
            assert len(result.sample_table) == len(result.pb_counts)

    def test_result_can_be_used_for_downstream(self, adata_balanced):
        """Test that result contains all necessary info for DE analysis."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=1,
        )

        # Check all required fields for DE are present
        assert result.pb_counts is not None
        assert result.sample_table is not None
        assert result.design_matrix is not None
        assert result.design_formula is not None
        assert result.group_key_internal is not None

        # Design matrix should match sample table
        if result.strata:
            assert len(result.design_matrix) == len(result.sample_table)

    def test_sample_table_contains_all_required_columns(self, adata_balanced):
        """Test that sample_table has all the new statistics columns."""
        result = pseudobulk(
            adata_balanced,
            group_key="cell_type",
            query="TypeA",
            reference="TypeB",
            replicate_key="batch",
            min_cells=1,
        )

        if result.strata:
            required_columns = ["psbulk_condition", "n_cells", "n_cells_condition", "fraction", "coverage"]
            for col in required_columns:
                assert col in result.sample_table.columns, f"Missing column: {col}"
