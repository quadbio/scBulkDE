"""Tests for rank_genes_groups function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from scbulkde.tl.rank_genes_groups import rank_genes_groups
from scbulkde.ut import DEResult


class TestRankGenesGroups:
    """Tests for rank_genes_groups Scanpy-compatible wrapper."""

    @pytest.fixture
    def adata_3groups(self, make_adata):
        """AnnData with three groups A, B, C and two donors each."""
        return make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B", "C"],
            group_counts=[40, 40, 40],
            replicate_key="donor",
            replicate_values=(["d1"] * 20 + ["d2"] * 20) * 3,
        )

    @pytest.fixture
    def adata_2groups(self, make_adata):
        """AnnData with two groups A, B and two donors each."""
        return make_adata(
            n_cells=120,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

    @pytest.fixture
    def mock_de_for_rank_genes(self, monkeypatch):
        """Mock the de() function to return deterministic results."""

        def mock_de(adata, group_key, query, reference, layer=None, **kwargs):
            n_genes = adata.shape[1]
            gene_names = adata.var_names

            results = pd.DataFrame(
                {
                    "pvalue": np.linspace(0.001, 0.1, n_genes),
                    "padj": np.linspace(0.01, 0.2, n_genes),
                    "stat": np.linspace(10, 1, n_genes),
                    "stat_sign": np.linspace(10, 1, n_genes),
                    "log2FoldChange": np.linspace(3, -3, n_genes),
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

        monkeypatch.setattr("scbulkde.tl.tl_basic.de", mock_de)

    def test_basic_functionality_all_groups(self, adata_3groups, mock_de_for_rank_genes):
        """Test basic usage with groups='all'."""
        rank_genes_groups(
            adata_3groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        assert "rank_genes_groups" in adata_3groups.uns
        assert "params" in adata_3groups.uns["rank_genes_groups"]

        result = adata_3groups.uns["rank_genes_groups"]
        assert "names" in result
        assert "scores" in result
        assert "pvals" in result
        assert "pvals_adj" in result
        assert "logfoldchanges" in result

        names_dtype = result["names"].dtype
        assert len(names_dtype.names) == 3
        assert set(names_dtype.names) == {"A", "B", "C"}

    def test_specific_groups_subset(self, adata_3groups, mock_de_for_rank_genes):
        """Test with specific subset of groups."""
        rank_genes_groups(
            adata_3groups,
            groupby="cell_type",
            groups=["A", "B"],
            reference="rest",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        result = adata_3groups.uns["rank_genes_groups"]
        names_dtype = result["names"].dtype

        assert len(names_dtype.names) == 2
        assert set(names_dtype.names) == {"A", "B"}

    def test_reference_rest_behavior(self, adata_3groups, mock_de_for_rank_genes):
        """Test that reference='rest' compares each group to all others."""
        rank_genes_groups(
            adata_3groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        assert adata_3groups.uns["rank_genes_groups"]["params"]["reference"] == "rest"

    def test_specific_reference_group(self, adata_3groups, mock_de_for_rank_genes):
        """Test with a specific reference group."""
        rank_genes_groups(
            adata_3groups,
            groupby="cell_type",
            groups=["A", "B"],
            reference="C",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        result = adata_3groups.uns["rank_genes_groups"]

        assert result["params"]["reference"] == "C"

        names_dtype = result["names"].dtype
        assert len(names_dtype.names) == 2
        assert set(names_dtype.names) == {"A", "B"}
        assert "C" not in names_dtype.names

    def test_n_genes_limits_output(self, adata_2groups, mock_de_for_rank_genes):
        """Test that n_genes parameter limits returned genes."""
        n_genes_requested = 10

        rank_genes_groups(
            adata_2groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            n_genes=n_genes_requested,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        result = adata_2groups.uns["rank_genes_groups"]
        for group in ["A", "B"]:
            assert len(result["names"][group]) == n_genes_requested
            assert len(result["scores"][group]) == n_genes_requested

    def test_rankby_abs_uses_absolute_values(self, adata_2groups, mock_de_for_rank_genes):
        """Test that rankby_abs=True ranks by absolute log fold changes."""
        rank_genes_groups(
            adata_2groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            rankby_abs=True,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        assert "rank_genes_groups" in adata_2groups.uns
        result = adata_2groups.uns["rank_genes_groups"]
        assert "logfoldchanges" in result

    def test_pts_calculates_fraction_expressing(self, adata_3groups, mock_de_for_rank_genes):
        """Test that pts=True adds fraction expressing information."""
        rank_genes_groups(
            adata_3groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            pts=True,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        result = adata_3groups.uns["rank_genes_groups"]

        assert "pts" in result
        assert isinstance(result["pts"], pd.DataFrame)
        assert set(result["pts"].columns) == {"A", "B", "C"}
        assert len(result["pts"]) == 50

        assert "pts_rest" in result
        assert isinstance(result["pts_rest"], pd.DataFrame)

    def test_use_raw_true_uses_raw_layer(self, adata_2groups, mock_de_for_rank_genes):
        """Test that use_raw=True uses adata.raw if available."""
        adata_2groups.raw = adata_2groups.copy()

        rank_genes_groups(
            adata_2groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            use_raw=True,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        assert adata_2groups.uns["rank_genes_groups"]["params"]["use_raw"] is True

    def test_raises_error_use_raw_true_no_raw(self, make_adata):
        """Test that use_raw=True raises error when adata.raw is None."""
        adata = make_adata(n_cells=120, n_genes=50, groups=["A", "B"], group_counts=[60, 60])
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
        """Test that invalid reference raises error."""
        adata = make_adata(n_cells=120, n_genes=50, groups=["A", "B", "C"], group_counts=[40, 40, 40])

        with pytest.raises(ValueError, match="reference.*needs to be one of"):
            rank_genes_groups(
                adata,
                groupby="cell_type",
                groups="all",
                reference="INVALID",
                de_kwargs={"replicate_key": "donor"},
            )

    def test_raises_error_groups_not_sequence(self, adata_3groups, mock_de_for_rank_genes):
        """Test that passing a single string for groups raises error."""
        rank_genes_groups(
            adata_3groups,
            groupby="cell_type",
            groups="all",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        with pytest.raises(ValueError, match="Specify a sequence of groups"):
            rank_genes_groups(
                adata_3groups,
                groupby="cell_type",
                groups="A",
                de_kwargs={"replicate_key": "donor"},
            )

    def test_copy_returns_new_adata(self, adata_2groups, mock_de_for_rank_genes):
        """Test that copy=True returns a new AnnData object."""
        result = rank_genes_groups(
            adata_2groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            copy=True,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        assert result is not None
        assert result is not adata_2groups
        assert "rank_genes_groups" in result.uns
        assert "rank_genes_groups" not in adata_2groups.uns

    def test_copy_false_modifies_inplace(self, adata_2groups, mock_de_for_rank_genes):
        """Test that copy=False modifies adata in-place and returns None."""
        result = rank_genes_groups(
            adata_2groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            copy=False,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        assert result is None
        assert "rank_genes_groups" in adata_2groups.uns

    def test_key_added_custom_key(self, adata_2groups, mock_de_for_rank_genes):
        """Test that custom key_added stores results under different key."""
        custom_key = "my_custom_results"

        rank_genes_groups(
            adata_2groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            key_added=custom_key,
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        assert custom_key in adata_2groups.uns
        assert "names" in adata_2groups.uns[custom_key]
        assert "rank_genes_groups" not in adata_2groups.uns

    def test_stores_correct_params(self, adata_2groups, mock_de_for_rank_genes):
        """Test that params are stored correctly in adata.uns."""
        rank_genes_groups(
            adata_2groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            use_raw=False,
            layer=None,
            de_kwargs={"replicate_key": "donor", "min_cells": 5, "engine": "anova"},
        )

        params = adata_2groups.uns["rank_genes_groups"]["params"]

        assert params["groupby"] == "cell_type"
        assert params["reference"] == "rest"
        assert params["use_raw"] is False
        assert params["layer"] is None
        assert params["method"] == "pseudobulk_anova"

    def test_output_dtypes_match_scanpy(self, adata_2groups, mock_de_for_rank_genes):
        """Test that output dtypes match Scanpy's expected types."""
        rank_genes_groups(
            adata_2groups,
            groupby="cell_type",
            groups="all",
            reference="rest",
            de_kwargs={"replicate_key": "donor", "min_cells": 5},
        )

        result = adata_2groups.uns["rank_genes_groups"]

        assert result["names"].dtype.names is not None
        assert result["scores"].dtype.names is not None
        assert result["pvals"].dtype.names is not None
        assert result["pvals_adj"].dtype.names is not None
        assert result["logfoldchanges"].dtype.names is not None

        group = "A"
        assert result["names"][group].dtype == np.dtype("O")
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

        assert len(top_3) == 3
        assert top_3[0] == 3
        assert top_3[1] == 5
        assert top_3[2] == 1

    def test_fraction_expressing_dense(self):
        """Test _fraction_expressing with dense matrix."""
        from scbulkde.ut.ut_basic import _fraction_expressing

        X = np.array([[0, 1, 0, 3], [1, 0, 2, 0], [0, 0, 1, 1], [2, 1, 0, 0]])
        mask = np.array([True, True, False, False])

        fractions = _fraction_expressing(X, mask)

        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(fractions, expected)

    def test_fraction_expressing_sparse(self):
        """Test _fraction_expressing with sparse matrix."""
        from scbulkde.ut.ut_basic import _fraction_expressing

        X = sp.csr_matrix(np.array([[0, 1, 0, 3], [1, 0, 2, 0], [0, 0, 1, 1], [2, 1, 0, 0]]))
        mask = np.array([True, True, False, False])

        fractions = _fraction_expressing(X, mask)

        expected = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_array_equal(fractions, expected)
