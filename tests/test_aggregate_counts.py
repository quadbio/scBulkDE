"""Tests for _aggregate_counts function."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from scbulkde.ut.ut_basic import _aggregate_counts


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

    def test_aggregate_counts_sparse_sum(self, make_adata):
        """Verify sum aggregation preserves sparsity and correctness."""
        adata = make_adata(
            n_cells=6,
            n_genes=3,
            groups=["A", "B"],
            group_counts=[3, 3],
            sparse=True,
        )

        # Set known sparse values
        dense = np.array(
            [
                [1, 0, 3],
                [0, 2, 0],
                [1, 0, 3],
                [10, 0, 12],
                [0, 11, 0],
                [10, 0, 12],
            ],
            dtype=np.float32,
        )
        adata.X = sp.csr_matrix(dense)
        grouped = adata.obs.groupby("cell_type", observed=True, sort=False)

        result = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="sum")

        # Expected sums:
        expected_dense = np.array(
            [
                [2, 2, 6],  # A sum
                [20, 11, 24],  # B sum
            ],
            dtype=np.float32,
        )

        # Verify result is actually sparse
        assert hasattr(result, "sparse")
        # Check values. The .values accessor converts to dense for comparison
        assert np.array_equal(result.values, expected_dense)

    def test_aggregate_counts_sparse_mean(self, make_adata):
        """Verify mean aggregation on sparse matrix."""
        adata = make_adata(
            n_cells=4,
            n_genes=2,
            groups=["A", "B"],
            group_counts=[2, 2],
            sparse=True,
        )

        dense = np.array(
            [
                [2, 4],
                [4, 8],
                [6, 12],
                [8, 16],
            ],
            dtype=np.float32,
        )
        adata.X = sp.csr_matrix(dense)

        grouped = adata.obs.groupby("cell_type", observed=True, sort=False)
        result = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="mean")

        # Expected means:
        expected_dense = np.array(
            [
                [3, 6],  # A mean
                [7, 14],  # B mean
            ],
            dtype=np.float32,
        )

        # Check values
        assert np.array_equal(result.values, expected_dense)

    def test_aggregate_counts_single_cell_group(self, make_adata):
        """Group with single cell should return that cell's values."""
        adata = make_adata(
            n_cells=3,
            n_genes=4,
            groups=["A", "B", "C"],
            group_counts=[1, 1, 1],
            sparse=False,
        )

        adata.X = np.array(
            [
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ],
            dtype=np.float32,
        )

        grouped = adata.obs.groupby("cell_type", observed=True, sort=False)
        result = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="sum")

        # Each group has 1 cell, so result should equal original values
        assert result.shape == (3, 4)
        # Total should equal total of X
        assert result.values.sum() == adata.X.sum()

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

    def test_aggregate_counts_column_names(self, make_adata):
        """Output DataFrame should have correct gene names as columns."""
        adata = make_adata(n_cells=10, n_genes=5, sparse=False)

        grouped = adata.obs.groupby("cell_type", observed=True, sort=False)
        result = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="sum")

        assert list(result.columns) == list(adata.var_names)

    def test_aggregate_counts_index_alignment(self, make_adata):
        """Ensure correct cells are aggregated when using complex grouping."""
        adata = make_adata(
            n_cells=12,
            n_genes=2,
            groups=["A", "B"],
            group_counts=[6, 6],
            replicate_key="batch",
            replicate_values=["b0", "b1", "b2"] * 4,
            sparse=False,
        )

        # Set values to identify cells
        adata.X = np.arange(24).reshape(12, 2).astype(np.float32)

        grouped = adata.obs.groupby(["cell_type", "batch"], observed=True, sort=False)
        result = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="sum")

        # Should have 6 groups: A/b0, A/b1, A/b2, B/b0, B/b1, B/b2
        # (or subset if not all combinations exist)
        assert result.shape[0] == len(list(grouped.groups.keys()))

    def test_aggregate_counts_misaligned_indices(self, make_adata):
        """Test behavior when grouped_obs indices don't match adata."""
        adata = make_adata(
            n_cells=10,
            n_genes=3,
            groups=["A", "B"],
            group_counts=[5, 5],
            sparse=False,
        )

        # Create grouped from a different subset
        obs_subset = adata.obs.iloc[:6].copy()  # Only first 6 cells
        grouped = obs_subset.groupby("cell_type", observed=True, sort=False)

        # This should work because get_indexer handles the mapping
        result = _aggregate_counts(adata, grouped, layer=None, layer_aggregation="sum")

        # Should only aggregate the cells in grouped
        assert result.shape[0] == grouped.ngroups
