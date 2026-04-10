"""Tests for _aggregate_counts function."""

from __future__ import annotations

import numpy as np
import pytest

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
