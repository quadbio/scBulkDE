"""Tests for _count_existing_samples function."""

from __future__ import annotations

import pandas as pd

from scbulkde.tl.tl_basic import _count_existing_samples


class TestCountExistingSamples:
    """Tests for _count_existing_samples."""

    def test_multiindex_counts_unique_strata_per_condition(self):
        """With MultiIndex groupby, should count unique strata combinations per condition."""
        obs = pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference", "query"],
                "batch": ["b1", "b2", "b1", "b2", "b1"],
            }
        )
        grouped = obs.groupby(["psbulk_condition", "batch"], observed=True, sort=False)

        result = _count_existing_samples(grouped)

        # Groups: (query, b1), (query, b2), (reference, b1), (reference, b2)
        # query appears in 2 groups, reference appears in 2 groups
        assert result["query"] == 2
        assert result["reference"] == 2

    def test_multiindex_with_three_strata_levels(self):
        """Should correctly count with multiple strata columns."""
        obs = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 4 + ["reference"] * 4,
                "batch": ["b1", "b1", "b2", "b2"] * 2,
                "donor": ["d1", "d2", "d1", "d2"] * 2,
            }
        )
        grouped = obs.groupby(["psbulk_condition", "batch", "donor"], observed=True, sort=False)

        result = _count_existing_samples(grouped)

        # 4 unique (batch, donor) combinations for each condition
        assert result["query"] == 4
        assert result["reference"] == 4

    def test_single_index_returns_zero_for_both(self):
        """Single-column groupby (no strata) should return 0 for both conditions.

        This is the 'collapsed' case where no valid strata exist.
        """
        obs = pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference"],
            }
        )
        grouped = obs.groupby("psbulk_condition", observed=True, sort=False)

        result = _count_existing_samples(grouped)

        # No valid samples because there's no stratification
        assert result == {"query": 0, "reference": 0}

    def test_missing_condition_returns_zero(self):
        """If a condition has no groups in the strata, should return 0 for it."""
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

    def test_unbalanced_strata(self):
        """Should handle unbalanced strata correctly."""
        obs = pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "query", "reference"],
                "batch": ["b1", "b2", "b3", "b1"],
            }
        )
        grouped = obs.groupby(["psbulk_condition", "batch"], observed=True, sort=False)

        result = _count_existing_samples(grouped)

        # query has 3 unique batches, reference has 1
        assert result["query"] == 3
        assert result["reference"] == 1

    def test_empty_grouped_raises_or_returns_zero(self):
        """Edge case: empty DataFrame should handle gracefully."""
        obs = pd.DataFrame({"psbulk_condition": [], "batch": []})
        grouped = obs.groupby(["psbulk_condition", "batch"], observed=True, sort=False)

        result = _count_existing_samples(grouped)

        # No groups exist
        assert result["query"] == 0
        assert result["reference"] == 0
