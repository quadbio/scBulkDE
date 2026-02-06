"""Tests for _count_existing_samples function."""

from __future__ import annotations

import pandas as pd

from scbulkde.tl.tl_basic import _count_existing_samples


class TestCountExistingSamples:
    """Tests for _count_existing_samples."""

    def test_single_column_groupby_counts_correctly(self):
        """Should count samples when grouped by single column."""
        obs = pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference", "reference"],
                "cell_id": range(5),
            }
        )
        grouped = obs.groupby("psbulk_condition", observed=True, sort=False)

        result = _count_existing_samples(grouped)

        # Should count number of groups, not number of cells
        # With single column groupby on psbulk_condition, there are 2 groups
        assert result == {"query": 1, "reference": 1}

    def test_multi_column_groupby_counts_correctly(self):
        """Should count samples when grouped by multiple columns."""
        obs = pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference", "query"],
                "batch": ["b1", "b2", "b1", "b2", "b1"],
                "cell_id": range(5),
            }
        )
        grouped = obs.groupby(["psbulk_condition", "batch"], observed=True, sort=False)

        result = _count_existing_samples(grouped)

        # query+b1: 2 cells, query+b2: 1 cell, ref+b1: 1 cell, ref+b2: 1 cell
        # But counting groups where condition matches
        # Potential bug: function iterates over tuple and checks if "query" or "reference" in tuple
        # ("query", "b1") has "query" so query += 1
        # ("query", "b2") has "query" so query += 1
        # ("reference", "b1") has "reference" so reference += 1
        # ("reference", "b2") has "reference" so reference += 1
        assert result["query"] == 2
        assert result["reference"] == 2

    def test_empty_condition_returns_zero(self):
        """Should return 0 for conditions not present."""
        obs = pd.DataFrame(
            {
                "psbulk_condition": ["query", "query"],
                "batch": ["b1", "b2"],
            }
        )
        grouped = obs.groupby(["psbulk_condition", "batch"], observed=True, sort=False)

        result = _count_existing_samples(grouped)

        assert result["query"] == 2
        assert result["reference"] == 0

    def test_handles_additional_strata_columns(self):
        """Should correctly count when there are multiple strata."""
        obs = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 4 + ["reference"] * 4,
                "batch": ["b1", "b1", "b2", "b2"] * 2,
                "donor": ["d1", "d2", "d1", "d2"] * 2,
            }
        )
        grouped = obs.groupby(["psbulk_condition", "batch", "donor"], observed=True, sort=False)

        result = _count_existing_samples(grouped)

        # 4 unique (condition, batch, donor) for each condition
        assert result["query"] == 4
        assert result["reference"] == 4
