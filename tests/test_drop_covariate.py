"""Tests for _drop_covariate function."""

from __future__ import annotations

import pandas as pd
import pytest

from scbulkde.ut.ut_basic import _drop_covariate


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
