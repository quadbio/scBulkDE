"""Tests for _prepare_internal_groups function."""

from __future__ import annotations

import pandas as pd
import pytest

from scbulkde.ut.ut_basic import _prepare_internal_groups


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
