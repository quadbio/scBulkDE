"""Tests for rank_genes_groups function.

These tests validate the Scanpy-compatible API and output structure.
Focus is on correct group handling, output formatting, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestRankGenesGroups:
    """Tests for rank_genes_groups Scanpy-compatible wrapper."""

    @pytest.fixture
    def mock_de_for_rank_genes(self, monkeypatch):
        """Mock the de() function to return deterministic results for rank_genes_groups."""

        def mock_de(adata, group_key, query, reference, layer=None, **kwargs):
            """Return mock DE results with predictable patterns."""
            from scbulkde.ut import DEResult

            n_genes = adata.shape[1]
            gene_names = adata.var_names

            # Create results where genes are ranked by their index
            # This makes it easy to verify sorting behavior
            results = pd.DataFrame(
                {
                    "pvalue": np.linspace(0.001, 0.1, n_genes),
                    "padj": np.linspace(0.01, 0.2, n_genes),
                    "stat": np.linspace(10, 1, n_genes),  # Decreasing
                    "stat_sign": np.linspace(10, 1, n_genes),
                    "log2FoldChange": np.linspace(3, -3, n_genes),  # Decreasing
                },
                index=gene_names,
            )

            return DEResult(
                results=results,
                query=query,
                reference=reference,
                design=f"~{group_key}",
                engine="mock",
            )

        # Patch at the module where rank_genes_groups imports it from
        monkeypatch.setattr("scbulkde.tl.tl_basic.de", mock_de)

    def test_basic_functionality_all_groups(self, make_adata, mock_de_for_rank_genes):
        """Test basic usage with groups='all'."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B", "C"],
            group_counts=[40, 40, 40],
            replicate_key="donor",
            replicate_values=(["d1"] * 20 + ["d2"] * 20) * 3,
        )

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        # Check results structure
        assert "rank_genes_groups" in adata.uns
        assert "params" in adata.uns["rank_genes_groups"]

        # Check required fields exist
        result = adata.uns["rank_genes_groups"]
        assert "names" in result
        assert "scores" in result
        assert "pvals" in result
        assert "pvals_adj" in result
        assert "logfoldchanges" in result

        # Check that all groups were tested
        names_dtype = result["names"].dtype
        assert len(names_dtype.names) == 3  # A, B, C
        assert set(names_dtype.names) == {"A", "B", "C"}

    def test_specific_groups_subset(self, make_adata, mock_de_for_rank_genes):
        """Test with specific subset of groups."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B", "C"],
            group_counts=[40, 40, 40],
            replicate_key="donor",
            replicate_values=(["d1"] * 20 + ["d2"] * 20) * 3,
        )

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups=["A", "B"],  # Only test A and B
            reference="rest",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        result = adata.uns["rank_genes_groups"]
        names_dtype = result["names"].dtype

        # Should only have results for A and B
        assert len(names_dtype.names) == 2
        assert set(names_dtype.names) == {"A", "B"}

    def test_reference_rest_behavior(self, make_adata, mock_de_for_rank_genes):
        """Test that reference='rest' compares each group to all others."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B", "C"],
            group_counts=[40, 40, 40],
            replicate_key="donor",
            replicate_values=(["d1"] * 20 + ["d2"] * 20) * 3,
        )

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        # Params should reflect reference='rest'
        assert adata.uns["rank_genes_groups"]["params"]["reference"] == "rest"

    def test_specific_reference_group(self, make_adata, mock_de_for_rank_genes):
        """Test with a specific reference group - should not test reference against itself."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B", "C"],
            group_counts=[40, 40, 40],
            replicate_key="donor",
            replicate_values=(["d1"] * 20 + ["d2"] * 20) * 3,
        )

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups=["A", "B"],
            reference="C",  # Use C as reference
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        result = adata.uns["rank_genes_groups"]

        # Params should reflect specific reference
        assert result["params"]["reference"] == "C"

        # Should only have results for A and B (not C, since it's the reference)
        names_dtype = result["names"].dtype
        assert len(names_dtype.names) == 2
        assert set(names_dtype.names) == {"A", "B"}
        assert "C" not in names_dtype.names  # C should not be tested against itself

    def test_n_genes_limits_output(self, make_adata, mock_de_for_rank_genes):
        """Test that n_genes parameter limits returned genes."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        n_genes_requested = 10

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            n_genes=n_genes_requested,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        # Each group should have exactly n_genes results
        result = adata.uns["rank_genes_groups"]
        for group in ["A", "B"]:
            assert len(result["names"][group]) == n_genes_requested
            assert len(result["scores"][group]) == n_genes_requested

    def test_rankby_abs_uses_absolute_values(self, make_adata, mock_de_for_rank_genes):
        """Test that rankby_abs=True ranks by absolute log fold changes."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            rankby_abs=True,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        # Results should exist
        assert "rank_genes_groups" in adata.uns
        result = adata.uns["rank_genes_groups"]
        assert "logfoldchanges" in result

    def test_pts_calculates_fraction_expressing(self, make_adata, mock_de_for_rank_genes):
        """Test that pts=True adds fraction expressing information."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B", "C"],
            group_counts=[40, 40, 40],
            replicate_key="donor",
            replicate_values=(["d1"] * 20 + ["d2"] * 20) * 3,
        )

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            pts=True,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        result = adata.uns["rank_genes_groups"]

        # pts should be present
        assert "pts" in result
        assert isinstance(result["pts"], pd.DataFrame)

        # Should have columns for each group
        assert set(result["pts"].columns) == {"A", "B", "C"}

        # Should have rows for all genes
        assert len(result["pts"]) == 50

        # With reference='rest', pts_rest should also be present
        assert "pts_rest" in result
        assert isinstance(result["pts_rest"], pd.DataFrame)

    def test_use_raw_true_uses_raw_layer(self, make_adata, mock_de_for_rank_genes):
        """Test that use_raw=True uses adata.raw if available."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        # Create a raw layer
        adata.raw = adata.copy()

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            use_raw=True,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        # Params should reflect use_raw
        assert adata.uns["rank_genes_groups"]["params"]["use_raw"] is True

    def test_raises_error_use_raw_true_no_raw(self, make_adata):
        """Test that use_raw=True raises error when adata.raw is None.

        This test validates early - before DE is called.
        """
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
        )

        # Ensure no raw layer
        adata.raw = None

        with pytest.raises(ValueError):
            rank_genes_groups(
                adata,
                groupby="cell_type",
                groups="all",
                reference="rest",
                use_raw=True,
                de_kwargs={"replicate_key": "donor"},
            )

    def test_raises_error_invalid_reference(self, make_adata):
        """Test that invalid reference raises error.

        This test validates early - before DE is called.
        """
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B", "C"],
            group_counts=[40, 40, 40],
        )

        with pytest.raises(ValueError, match="reference.*needs to be one of"):
            rank_genes_groups(
                adata,
                groupby="cell_type",
                groups="all",
                reference="INVALID",  # Not a valid group
                de_kwargs={"replicate_key": "donor"},
            )

    def test_raises_error_groups_not_sequence(self, make_adata, mock_de_for_rank_genes):
        """Test that passing a single string (not sequence) for groups raises error."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B", "C"],
            group_counts=[40, 40, 40],
            replicate_key="donor",
            replicate_values=(["d1"] * 20 + ["d2"] * 20) * 3,
        )

        # This should work - groups="all" is valid
        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        # But a single group name string should fail (validates early, before DE)
        with pytest.raises(ValueError, match="Specify a sequence of groups"):
            rank_genes_groups(
                adata,
                groupby="cell_type",
                groups="A",  # Single string, not a list
                de_kwargs={"replicate_key": "donor"},
            )

    def test_copy_returns_new_adata(self, make_adata, mock_de_for_rank_genes):
        """Test that copy=True returns a new AnnData object."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        result = rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            copy=True,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        # Should return a new AnnData object
        assert result is not None
        assert result is not adata
        assert "rank_genes_groups" in result.uns

        # Original should not be modified
        assert "rank_genes_groups" not in adata.uns

    def test_copy_false_modifies_inplace(self, make_adata, mock_de_for_rank_genes):
        """Test that copy=False modifies adata in-place and returns None."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        result = rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            copy=False,  # Default
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        # Should return None
        assert result is None

        # Original should be modified
        assert "rank_genes_groups" in adata.uns

    def test_key_added_custom_key(self, make_adata, mock_de_for_rank_genes):
        """Test that custom key_added stores results under different key."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        custom_key = "my_custom_results"

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            key_added=custom_key,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        # Results should be under custom key
        assert custom_key in adata.uns
        assert "names" in adata.uns[custom_key]

        # Default key should not exist
        assert "rank_genes_groups" not in adata.uns

    def test_stores_correct_params(self, make_adata, mock_de_for_rank_genes):
        """Test that params are stored correctly in adata.uns."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            use_raw=False,
            layer=None,
            de_kwargs={"replicate_key": "donor", "min_cells": 5, "engine": "anova"},
        )

        params = adata.uns["rank_genes_groups"]["params"]

        # Check all params are stored
        assert params["groupby"] == "cell_type"
        assert params["reference"] == "rest"
        assert params["use_raw"] is False
        assert params["layer"] is None
        assert params["method"] == "pseudobulk_anova"

    def test_output_dtypes_match_scanpy(self, make_adata, mock_de_for_rank_genes):
        """Test that output dtypes match Scanpy's expected types."""
        from scbulkde.tl.rank_genes_groups import rank_genes_groups

        adata = make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        rank_genes_groups(
            adata,
            groupby="cell_type",
            groups="all",
            reference="rest",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        result = adata.uns["rank_genes_groups"]

        # Check dtypes match Scanpy expectations
        # These are structured arrays (numpy record arrays)
        assert result["names"].dtype.names is not None
        assert result["scores"].dtype.names is not None
        assert result["pvals"].dtype.names is not None
        assert result["pvals_adj"].dtype.names is not None
        assert result["logfoldchanges"].dtype.names is not None

        # Access one group's data to check field dtypes
        group = "A"
        assert result["names"][group].dtype == np.dtype("O")  # object
        assert result["scores"][group].dtype == np.float32
        assert result["pvals"][group].dtype == np.float64
        assert result["pvals_adj"][group].dtype == np.float64
        assert result["logfoldchanges"][group].dtype == np.float32


class TestRankGenesGroupsHelpers:
    """Tests for helper functions used by rank_genes_groups."""

    def test_select_top_n(self):
        """Test _select_top_n selects correct indices."""
        from scbulkde.ut.ut_basic import _select_top_n

        scores = np.array([0.1, 0.5, 0.3, 0.9, 0.2, 0.7])
        top_3 = _select_top_n(scores, n_top=3)

        # Should return indices [3, 5, 1] (scores 0.9, 0.7, 0.5)
        assert len(top_3) == 3
        assert top_3[0] == 3  # Highest score (0.9)
        assert top_3[1] == 5  # Second (0.7)
        assert top_3[2] == 1  # Third (0.5)

    def test_fraction_expressing_dense(self):
        """Test _fraction_expressing with dense matrix."""
        from scbulkde.ut.ut_basic import _fraction_expressing

        # Create a simple expression matrix
        X = np.array(
            [
                [0, 1, 0, 3],
                [1, 0, 2, 0],
                [0, 0, 1, 1],
                [2, 1, 0, 0],
            ]
        )

        # Mask for first two cells
        mask = np.array([True, True, False, False])

        fractions = _fraction_expressing(X, mask)

        # Gene 0: 1/2 cells express (cells 0,1 -> values 0,1)
        # Gene 1: 1/2 cells express (cells 0,1 -> values 1,0)
        # Gene 2: 1/2 cells express (cells 0,1 -> values 0,2)
        # Gene 3: 1/2 cells express (cells 0,1 -> values 3,0)
        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(fractions, expected)

    def test_fraction_expressing_sparse(self):
        """Test _fraction_expressing with sparse matrix."""
        import scipy.sparse as sp

        from scbulkde.ut.ut_basic import _fraction_expressing

        # Create a sparse expression matrix
        X = sp.csr_matrix(
            np.array(
                [
                    [0, 1, 0, 3],
                    [1, 0, 2, 0],
                    [0, 0, 1, 1],
                    [2, 1, 0, 0],
                ]
            )
        )

        mask = np.array([True, True, False, False])

        fractions = _fraction_expressing(X, mask)

        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(fractions, expected)
