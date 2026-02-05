"""Tests for scbulkde.pp.pseudobulk function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestPseudobulkBasicLogic:
    """Test basic pseudobulking logic and cell selection."""

    def test_query_and_reference_cells_selected(self, make_adata):
        """Should include only cells from query and reference groups."""
        adata = make_adata(
            n_cells=120,
            groups=["A", "B", "C", "D"],
            group_counts=[30, 30, 30, 30],
            replicate_key="batch",
            replicate_values=(["b0", "b1", "b2"] * 40),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
        )

        # Should only include A and B cells (60 total)
        assert result.adata_sub.n_obs == 60

        grouped_df = result.grouped.obj
        # Should have cells labeled as query or reference
        assert "psbulk_condition" in grouped_df.columns
        # Query cells should be from group A
        query_mask = grouped_df["psbulk_condition"] == "query"
        assert all(grouped_df.loc[query_mask, "cell_type"] == "A")
        # Reference cells should be from group B
        ref_mask = grouped_df["psbulk_condition"] == "reference"
        assert all(grouped_df.loc[ref_mask, "cell_type"] == "B")

    def test_reference_rest_includes_all_non_query(self, make_adata):
        """reference='rest' should include all groups not in query."""
        adata = make_adata(
            n_cells=100,
            groups=["A", "B", "C", "D"],
            group_counts=[25, 25, 25, 25],
            replicate_key="batch",
            replicate_values=(["b0", "b1"] * 50),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="rest",
            replicate_key="batch",
        )

        # Should include all 100 cells (A vs B+C+D)
        assert result.adata_sub.n_obs == 100

        # Access to grouped
        grouped_df = result.grouped.obj
        # Query should only be A cells
        query_mask = grouped_df["psbulk_condition"] == "query"
        assert (grouped_df.loc[query_mask, "cell_type"] == "A").all()
        # Reference should be B, C, D cells
        ref_mask = grouped_df["psbulk_condition"] == "reference"
        ref_groups = set(grouped_df.loc[ref_mask, "cell_type"].unique())
        assert ref_groups == {"B", "C", "D"}

    def test_multiple_query_groups(self, make_adata):
        """Should handle multiple groups in query."""
        adata = make_adata(
            n_cells=120,
            groups=["A", "B", "C", "D"],
            group_counts=[30, 30, 30, 30],
            replicate_key="batch",
            replicate_values=(["b0", "b1", "b2"] * 40),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query=["A", "B"],
            reference=["C", "D"],
            replicate_key="batch",
        )

        # Should include all 120 cells
        assert result.adata_sub.n_obs == 120

        # Access to grouped
        grouped_df = result.grouped.obj

        # Query should be A and B
        query_mask = grouped_df["psbulk_condition"] == "query"
        query_groups = set(grouped_df.loc[query_mask, "cell_type"].unique())
        assert query_groups == {"A", "B"}


class TestPseudobulkStratification:
    """Test stratification logic with replicate_key and covariates."""

    def test_with_replicate_key_creates_samples_per_replicate(self, make_adata):
        """With replicate_key, should create separate samples per replicate."""
        adata = make_adata(
            n_cells=120,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="batch",
            replicate_values=(["b0", "b1", "b2"] * 40),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
        )

        # Should have 6 samples: 3 batches × 2 conditions
        assert result.pb_counts.shape[0] == 6
        assert result.sample_table.shape[0] == 6
        # Sample table should have batch column
        assert "batch" in result.sample_table.columns
        # Each condition should have 3 replicates
        assert (result.sample_table["psbulk_condition"].value_counts() == 3).all()

    def test_with_categorical_covariates_stratifies_by_them(self, make_adata):
        """Categorical covariates should create stratified samples."""
        adata = make_adata(
            n_cells=8,
            groups=["A", "B"],
            group_counts=[4, 4],
            categorical_covariates={
                "sex": (["M", "F"] * 4),
                "experiment": (["e1", "e1", "e2", "e2"] * 2),
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            min_cells=1,
            min_fraction=None,
            categorical_covariates=["sex", "experiment"],
        )

        # Should stratify by sex and treatment: 2×2 = 4 combinations per condition = 8 samples
        assert result.pb_counts.shape[0] == 8
        assert "sex" in result.sample_table.columns
        assert "experiment" in result.sample_table.columns

    def test_replicate_key_and_covariates_combined(self, make_adata):
        """replicate_key and categorical_covariates should both stratify."""
        adata = make_adata(
            n_cells=8,
            groups=["A", "B"],
            group_counts=[4, 4],
            replicate_key="donor",
            replicate_values=(["d0", "d1", "d2", "d3"] * 2),
            categorical_covariates={
                "experiment": (["e1", "e2"] * 4),
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            min_cells=1,
            min_fraction=None,
            replicate_key="donor",
            categorical_covariates=["experiment"],
        )

        # Should stratify by donor and treatment: d0 and d2 in e1, d1 and d3 in e2
        assert result.pb_counts.shape[0] == 8
        assert "donor" in result.sample_table.columns
        assert "experiment" in result.sample_table.columns

    def test_no_stratification_returns_empty_counts(self, make_adata):
        """Without replicate_key or covariates, should return empty counts."""
        adata = make_adata(
            n_cells=100,
            groups=["A", "B"],
            group_counts=[50, 50],
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key=None,
            categorical_covariates=None,
        )

        # Should return empty pseudobulk counts
        assert result.pb_counts.shape[0] == 0
        # But should still have sample_table with condition information
        assert result.sample_table.shape[0] == 2  # query and reference
        assert set(result.sample_table["psbulk_condition"]) == {"query", "reference"}
        # grouped should not be None
        assert result.grouped is not None


class TestPseudobulkSampleQualification:
    """Test min_cells, min_fraction, min_coverage filtering logic."""

    def test_min_cells_filters_small_samples(self, make_adata):
        """Samples with fewer than min_cells should be filtered out or cause covariate drop."""
        adata = make_adata(
            n_cells=120,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="batch",
            # Uneven distribution: b0 has 50 cells, b1 has 8, b2 has 2 per group
            replicate_values=(["b0"] * 50 + ["b1"] * 8 + ["b2"] * 2) * 2,
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            min_cells=10,  # Only b0 qualifies
            min_fraction=None,
            min_coverage=None,
        )
        assert result.pb_counts.shape[0] == 2  # b0 for query and reference

        # With min_cells=10, only b0 (50 cells) qualifies
        # But min_coverage is None, so as long as ANY samples qualify, it might pass
        # However, the logic should check that BOTH conditions have qualifying samples
        # This specific case depends on implementation details

        # Alternative: strict requirements that can't be met should drop the covariate
        result_strict = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            min_cells=10,
            min_fraction=None,
            min_coverage=0.9,  # Requires 90% of cells to be in qualifying samples
        )

        # Only b0 qualifies (50/60 = 83% < 90%), so should drop batch and return empty
        assert result_strict.pb_counts.shape[0] == 0

    def test_min_fraction_relative_to_condition(self, make_adata):
        """min_fraction should be relative to total cells in each condition, not global."""
        adata = make_adata(
            n_cells=200,
            groups=["A", "B"],
            group_counts=[100, 100],
            replicate_key="batch",
            # For A: b0=60, b1=40
            # For B: b0=20, b1=80
            replicate_values=(["b0"] * 60 + ["b1"] * 40 + ["b0"] * 20 + ["b1"] * 80),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            min_cells=None,
            min_fraction=0.3,  # Each batch must have ≥30% of cells within its condition
            min_coverage=None,
        )

        # For query (A): b0=60/100=60% ✓, b1=40/100=40% ✓
        # For reference (B): b0=20/100=20% ✗, b1=80/100=80% ✓
        # Since b0 fails in reference, stratification should fail
        # Should drop batch and return empty counts
        assert result.pb_counts.shape[0] == 0

    def test_min_coverage_requires_sufficient_cell_coverage(self, make_adata):
        """min_coverage requires that qualifying samples cover sufficient fraction of cells."""
        adata = make_adata(
            n_cells=200,
            groups=["A", "B"],
            group_counts=[100, 100],
            replicate_key="batch",
            replicate_values=(["b0"] * 70 + ["b1"] * 30 + ["b0"] * 70 + ["b1"] * 30),
        )

        from scbulkde.pp import pseudobulk

        # b0 has 70 cells (70%), b1 has 30 cells (30%)
        result_pass = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            min_cells=25,  # Both qualify
            min_fraction=None,
            min_coverage=0.6,  # Need 60% coverage - both batches together give 100%
        )

        result_fail = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            min_cells=60,  # Only b0 qualifies (70 cells)
            min_fraction=None,
            min_coverage=0.8,  # Need 80% coverage - only have 70%
        )

        # First should pass
        assert result_pass.pb_counts.shape[0] > 0
        # Second should fail and return empty
        assert result_fail.pb_counts.shape[0] == 0

    # def test_qualify_strategy_and_combines_requirements(self, make_adata):
    #     """qualify_strategy='and' requires BOTH min_cells AND min_fraction."""
    #     adata = make_adata(
    #         n_cells=200,
    #         groups=["A", "B"],
    #         group_counts=[100, 100],
    #         replicate_key="batch",
    #         # b0: 40 cells (40%), b1: 60 cells (60%)
    #         replicate_values=(["b0"] * 40 + ["b1"] * 60 + ["b0"] * 40 + ["b1"] * 60),
    #     )

    #     from scbulkde.pp import pseudobulk

    #     result_and = pseudobulk(
    #         adata=adata,
    #         group_key="cell_type",
    #         query="A",
    #         reference="B",
    #         replicate_key="batch",
    #         min_cells=50,  # b0=40 fails, b1=60 passes
    #         min_fraction=0.35,  # b0=40% passes, b1=60% passes
    #         qualify_strategy="and",  # BOTH requirements must be met
    #     )

    #     # With 'and': b0 fails min_cells (40 < 50), so only b1 qualifies
    #     # This is just one batch, so may not meet other requirements
    #     # Exact behavior depends on min_coverage (default 0.75)
    #     # Let's check more explicitly:

    #     result_and_explicit = pseudobulk(
    #         adata=adata,
    #         group_key="cell_type",
    #         query="A",
    #         reference="B",
    #         replicate_key="batch",
    #         min_cells=50,
    #         min_fraction=0.35,
    #         min_coverage=0.5,  # Need 50% of cells covered
    #         qualify_strategy="and",
    #     )

    #     # Only b1 (60 cells = 60%) qualifies, which gives 60% coverage ≥ 50%
    #     # Should create samples
    #     assert result_and_explicit.pb_counts.shape[0] == 2  # query and ref, b1 only

    def test_qualify_strategy_or_allows_either_requirement(self, make_adata):
        """qualify_strategy='or' requires EITHER min_cells OR min_fraction."""
        adata = make_adata(
            n_cells=200,
            groups=["A", "B"],
            group_counts=[100, 100],
            replicate_key="batch",
            replicate_values=(["b0"] * 40 + ["b1"] * 60 + ["b0"] * 40 + ["b1"] * 60),
        )

        from scbulkde.pp import pseudobulk

        result_or = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            min_cells=50,  # b0=40 fails, b1=60 passes
            min_fraction=0.35,  # b0=40% passes, b1=60% passes
            min_coverage=0.9,  # Need 90% of cells
            qualify_strategy="or",  # EITHER requirement is sufficient
        )

        # With 'or': b0 passes fraction (40% ≥ 35%), b1 passes both
        # Both batches qualify, giving 100% coverage
        assert result_or.pb_counts.shape[0] == 4  # 2 batches × 2 conditions


class TestPseudobulkCovariateDropping:
    """Test automatic covariate dropping when requirements can't be met."""

    def test_drops_covariates_when_cant_meet_requirements(self, make_adata):
        """Should drop covariates iteratively when sample requirements can't be met."""
        adata = make_adata(
            n_cells=200,
            groups=["A", "B"],
            group_counts=[100, 100],
            categorical_covariates={
                "batch": (["b0", "b1", "b2", "b3", "b4"] * 40),  # 5 batches, 20 cells each
                "treatment": (["ctrl", "drug"] * 100),  # 2 treatments
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            categorical_covariates=["batch", "treatment"],
            min_cells=30,  # Each stratum needs 30 cells
            min_fraction=None,
            min_coverage=0.8,
        )

        # With both batch and treatment: 5×2=10 strata, each with ~10 cells
        # Can't meet min_cells=30, so should drop covariates
        # After dropping treatment: 5 batches with 20 cells each - still fails
        # After dropping batch: no stratification - returns empty
        assert result.pb_counts.shape[0] == 0
        # strata should be empty
        assert result.strata == []

    def test_covariate_strategy_sequence_order_drops_from_end(self, make_adata):
        """covariate_strategy='sequence_order' should drop last covariate first."""
        adata = make_adata(
            n_cells=200,
            groups=["A", "B"],
            group_counts=[100, 100],
            categorical_covariates={
                "first": (["a", "b"] * 100),
                "second": (["x", "y", "z", "w"] * 50),  # This creates too many strata
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            categorical_covariates=["first", "second"],
            min_cells=40,
            covariate_strategy="sequence_order",
        )

        # "second" should be dropped first (it's last in sequence)
        # After dropping, should have stratification by "first" only
        if result.pb_counts.shape[0] > 0:
            assert "first" in result.strata
            assert "second" not in result.strata

    def test_covariate_strategy_most_levels_drops_most_complex(self, make_adata):
        """covariate_strategy='most_levels' should drop covariate with most levels first."""
        adata = make_adata(
            n_cells=200,
            groups=["A", "B"],
            group_counts=[100, 100],
            categorical_covariates={
                "few": (["a", "b"] * 100),  # 2 levels
                "many": ([f"x{i % 8}" for i in range(200)]),  # 8 levels
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            categorical_covariates=["few", "many"],
            min_cells=40,
            covariate_strategy="most_levels",
        )

        # "many" with 8 levels should be dropped first
        if result.pb_counts.shape[0] > 0:
            assert "few" in result.strata
            assert "many" not in result.strata

    def test_resolve_conflicts_false_raises_when_dropping_all(self, make_adata):
        """resolve_conflicts=False should raise error instead of returning empty."""
        adata = make_adata(
            n_cells=100,
            groups=["A", "B"],
            group_counts=[50, 50],
            categorical_covariates={
                "batch": (["b0", "b1", "b2", "b3", "b4"] * 20),  # 5 batches, 10 cells each
            },
        )

        from scbulkde.pp import pseudobulk

        with pytest.raises(ValueError, match="Cannot generate samples"):
            pseudobulk(
                adata=adata,
                group_key="cell_type",
                query="A",
                reference="B",
                categorical_covariates=["batch"],
                min_cells=30,  # Can't be met
                resolve_conflicts=False,  # Should raise instead of dropping
            )


class TestPseudobulkAggregation:
    """Test count aggregation logic."""

    def test_layer_aggregation_sum_adds_counts(self, make_adata):
        """layer_aggregation='sum' should sum counts across cells."""
        adata = make_adata(
            n_cells=20,
            n_genes=5,
            groups=["A", "B"],
            group_counts=[10, 10],
            replicate_key="batch",
            replicate_values=(["b0"] * 10 + ["b1"] * 10),
            sparse=False,
        )
        # Set known values
        adata.X = np.ones((20, 5), dtype=np.float32)

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            layer_aggregation="sum",
        )

        # Each sample should have 5 cells (half of each batch)
        # With all values = 1, sum should be 5.0 per gene per sample
        assert result.pb_counts.shape == (4, 5)  # 2 batches × 2 conditions
        # Each pseudobulk sample has 5 cells, so sum should be 5.0
        assert np.allclose(result.pb_counts.values, 5.0)

    def test_layer_aggregation_mean_averages_counts(self, make_adata):
        """layer_aggregation='mean' should average counts across cells."""
        adata = make_adata(
            n_cells=20,
            n_genes=5,
            groups=["A", "B"],
            group_counts=[10, 10],
            replicate_key="batch",
            replicate_values=(["b0"] * 10 + ["b1"] * 10),
            sparse=False,
        )
        adata.X = np.ones((20, 5), dtype=np.float32) * 10

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            layer_aggregation="mean",
        )

        # Mean of 10.0 values should be 10.0
        assert np.allclose(result.pb_counts.values, 10.0)

    def test_layer_parameter_uses_specified_layer(self, make_adata):
        """layer parameter should use specified layer instead of .X."""
        adata = make_adata(
            n_cells=20,
            n_genes=5,
            groups=["A", "B"],
            group_counts=[10, 10],
            replicate_key="batch",
            replicate_values=(["b0"] * 10 + ["b1"] * 10),
            sparse=False,
            layer_name="raw_counts",
        )
        # X has values of ~5, layer has values of ~50
        adata.X = np.ones((20, 5), dtype=np.float32) * 5
        adata.layers["raw_counts"] = np.ones((20, 5), dtype=np.float32) * 50

        from scbulkde.pp import pseudobulk

        result_x = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            layer=None,  # Use .X
        )

        result_layer = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            layer="raw_counts",
        )

        # Results should differ by factor of 10
        assert np.allclose(result_x.pb_counts.values, 25.0)  # 5 cells × 5
        assert np.allclose(result_layer.pb_counts.values, 250.0)  # 5 cells × 50

    def test_sparse_matrices_handled_correctly(self, make_adata):
        """Should handle sparse matrices without converting to dense."""
        adata = make_adata(
            n_cells=20,
            n_genes=5,
            groups=["A", "B"],
            group_counts=[10, 10],
            replicate_key="batch",
            replicate_values=(["b0"] * 10 + ["b1"] * 10),
            sparse=True,
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
        )

        # Result should work correctly
        assert result.pb_counts.shape[0] == 4
        # pb_counts might be sparse DataFrame
        assert isinstance(result.pb_counts, pd.DataFrame)


class TestPseudobulkContinuousCovariates:
    """Test continuous covariate aggregation."""

    def test_continuous_covariates_aggregated_in_sample_table(self, make_adata):
        """Continuous covariates should be aggregated per sample."""
        adata = make_adata(
            n_cells=40,
            groups=["A", "B"],
            group_counts=[20, 20],
            replicate_key="batch",
            replicate_values=(["b0", "b1"] * 20),
            continuous_covariates={
                "age": [25.0, 30.0, 35.0, 40.0] * 10,
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            continuous_covariates=["age"],
            continuous_aggregation="mean",
        )

        # Sample table should have age column
        assert "age" in result.sample_table.columns
        # Age values should be aggregated means
        assert result.sample_table.shape[0] == 4  # 2 batches × 2 conditions

    def test_continuous_aggregation_mean(self, make_adata):
        """continuous_aggregation='mean' should average continuous values."""
        adata = make_adata(
            n_cells=20,
            groups=["A", "B"],
            group_counts=[10, 10],
            replicate_key="batch",
            replicate_values=(["b0"] * 10 + ["b1"] * 10),
            continuous_covariates={
                "age": [20.0, 40.0] * 10,  # Alternating 20 and 40
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            continuous_covariates=["age"],
            continuous_aggregation="mean",
        )

        # Mean of 20 and 40 should be 30
        assert all(result.sample_table["age"] == 30.0)

    def test_continuous_aggregation_median(self, make_adata):
        """continuous_aggregation='median' should use median."""
        adata = make_adata(
            n_cells=30,
            groups=["A", "B"],
            group_counts=[15, 15],
            replicate_key="batch",
            replicate_values=(["b0"] * 15 + ["b1"] * 15),
            continuous_covariates={
                "age": [10.0, 20.0, 30.0, 40.0, 50.0] * 6,
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            continuous_covariates=["age"],
            continuous_aggregation="median",
        )

        # Median should be 30.0
        assert all(result.sample_table["age"] == 30.0)


class TestPseudobulkDesignMatrix:
    """Test design matrix construction."""

    def test_design_matrix_has_condition_column(self, make_adata):
        """Design matrix should include condition (query vs reference)."""
        adata = make_adata(
            n_cells=40,
            groups=["A", "B"],
            group_counts=[20, 20],
            replicate_key="batch",
            replicate_values=(["b0", "b1"] * 20),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
        )

        # Design formula should include condition with reference base
        assert "psbulk_condition" in result.design_formula
        assert "reference" in result.design_formula
        # Design matrix should have query coefficient
        assert any("query" in str(col) for col in result.design_matrix.columns)

    def test_design_matrix_full_rank(self, make_adata):
        """Design matrix should have full column rank."""
        adata = make_adata(
            n_cells=60,
            groups=["A", "B"],
            group_counts=[30, 30],
            replicate_key="batch",
            replicate_values=(["b0", "b1", "b2"] * 20),
            categorical_covariates={
                "treatment": (["ctrl", "drug"] * 30),
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            categorical_covariates=["treatment"],
        )

        # Design matrix should have full rank
        rank = np.linalg.matrix_rank(result.design_matrix.values)
        assert rank == result.design_matrix.shape[1]

    def test_replicate_key_excluded_from_design(self, make_adata):
        """replicate_key should be used for stratification but not in design formula."""
        adata = make_adata(
            n_cells=60,
            groups=["A", "B"],
            group_counts=[30, 30],
            replicate_key="donor",
            replicate_values=(["d0", "d1", "d2"] * 20),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
        )

        # donor should be in sample_table for stratification
        assert "donor" in result.sample_table.columns
        # But donor should NOT be in design_formula (used only for sample creation)
        assert "C(donor)" not in result.design_formula

    def test_rank_deficient_covariates_dropped_from_design(self, make_adata):
        """Covariates causing rank deficiency should be dropped from design."""
        adata = make_adata(
            n_cells=40,
            groups=["A", "B"],
            group_counts=[20, 20],
            replicate_key="batch",
            replicate_values=(["b0", "b1"] * 20),
            categorical_covariates={
                # This covariate is perfectly confounded with condition
                "confounded": (["x"] * 20 + ["y"] * 20),
            },
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            categorical_covariates=["confounded"],
        )

        # confounded should be dropped from design to maintain full rank
        # (It's in strata but creates rank deficiency in design matrix)
        rank = np.linalg.matrix_rank(result.design_matrix.values)
        assert rank == result.design_matrix.shape[1]


class TestPseudobulkResultContainer:
    """Test PseudobulkResult container properties."""

    def test_result_contains_adata_sub(self, make_adata):
        """Result should contain subset AnnData with only relevant cells."""
        adata = make_adata(
            n_cells=100,
            groups=["A", "B", "C"],
            group_counts=[30, 40, 30],
            replicate_key="batch",
            replicate_values=(["b0", "b1"] * 50),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
        )

        # adata_sub should only have A and B cells (70 total)
        assert result.adata_sub.n_obs == 70
        assert set(result.adata_sub.obs["cell_type"].unique()) == {"A", "B"}

    def test_result_contains_all_parameters(self, make_adata):
        """Result should store all input parameters for reproducibility."""
        adata = make_adata(
            n_cells=80,
            groups=["A", "B"],
            group_counts=[40, 40],
            replicate_key="batch",
            replicate_values=(["b0", "b1"] * 40),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            min_cells=10,
            min_fraction=0.2,
            min_coverage=0.75,
            layer="raw",
            layer_aggregation="mean",
        )

        # Should store parameters
        assert result.group_key == "cell_type"
        assert result.min_cells == 10
        assert result.min_fraction == 0.2
        assert result.min_coverage == 0.75
        assert result.layer == "raw"
        assert result.layer_aggregation == "mean"
        assert result.qualify_strategy in ["and", "or"]

    def test_n_cells_diagnostic_stored(self, make_adata):
        """Result should store n_cells diagnostic info."""
        adata = make_adata(
            n_cells=100,
            groups=["A", "B"],
            group_counts=[40, 60],
            replicate_key="batch",
            replicate_values=(["b0", "b1"] * 50),
        )

        from scbulkde.pp import pseudobulk

        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
        )

        # Should have cell count info
        assert result.n_cells is not None
        assert "query" in result.n_cells
        assert "reference" in result.n_cells
        # Counts should match
        assert result.n_cells["query"] == 40
        assert result.n_cells["reference"] == 60


class TestPseudobulkEdgeCases:
    """Test edge cases and error handling."""

    def test_no_query_cells_raises(self, make_adata):
        """Should raise error when no cells match query."""
        adata = make_adata(
            n_cells=100,
            groups=["A", "B"],
            group_counts=[50, 50],
        )

        from scbulkde.pp import pseudobulk

        with pytest.raises(ValueError, match="No cells found for query"):
            pseudobulk(
                adata=adata,
                group_key="cell_type",
                query="NonExistent",
                reference="B",
            )

    def test_no_reference_cells_raises(self, make_adata):
        """Should raise error when no cells match reference."""
        adata = make_adata(
            n_cells=100,
            groups=["A", "B"],
            group_counts=[50, 50],
        )

        from scbulkde.pp import pseudobulk

        with pytest.raises(ValueError, match="No cells found for reference"):
            pseudobulk(
                adata=adata,
                group_key="cell_type",
                query="A",
                reference="NonExistent",
            )

    def test_single_cell_groups_handled(self, make_adata):
        """Should handle case where some strata have very few cells."""
        adata = make_adata(
            n_cells=50,
            groups=["A", "B"],
            group_counts=[25, 25],
            replicate_key="batch",
            # Very uneven: b0 has 1 cell, b1 has 24 cells
            replicate_values=(["b0"] + ["b1"] * 24 + ["b0"] + ["b1"] * 24),
        )

        from scbulkde.pp import pseudobulk

        # With default min_cells=50, this should fail and drop batch
        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="batch",
            min_cells=50,  # Can't be met
        )

        # Should return empty counts
        assert result.pb_counts.shape[0] == 0

    def test_overlapping_query_and_reference_warns(self, make_adata):
        """Should warn but prioritize query when groups overlap."""
        adata = make_adata(
            n_cells=90,
            groups=["A", "B", "C"],
            group_counts=[30, 30, 30],
            replicate_key="batch",
            replicate_values=(["b0", "b1", "b2"] * 30),
        )

        from scbulkde.pp import pseudobulk

        # B appears in both query and reference
        result = pseudobulk(
            adata=adata,
            group_key="cell_type",
            query=["A", "B"],
            reference=["B", "C"],
            replicate_key="batch",
        )

        # B cells should be assigned to query (prioritized)
        query_mask = result.adata_sub.obs["psbulk_condition"] == "query"
        query_groups = set(result.adata_sub.obs.loc[query_mask, "cell_type"].unique())
        assert "B" in query_groups

        ref_mask = result.adata_sub.obs["psbulk_condition"] == "reference"
        ref_groups = set(result.adata_sub.obs.loc[ref_mask, "cell_type"].unique())
        assert "B" not in ref_groups
        assert "C" in ref_groups
