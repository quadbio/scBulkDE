"""Tests for _can_generate_samples function."""

from __future__ import annotations

import pandas as pd
import pytest

from scbulkde.ut.ut_basic import _can_generate_samples


class TestCanGenerateSamples:
    """Tests for _can_generate_samples."""

    def test_empty_stratify_by_returns_false(self, make_obs):
        """Empty stratify_by should return False."""
        obs = make_obs()

        result = _can_generate_samples(
            obs=obs,
            stratify_by=[],
            min_cells=10,
            min_fraction=0.1,
            min_coverage=0.5,
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )

        assert result is False

    def test_all_thresholds_none_returns_false(self, make_obs):
        """When min_cells=None, min_fraction=None, no group should qualify."""
        obs = make_obs(
            n_query=10,
            n_reference=10,
            query_strata={"batch": ["b0"] * 5 + ["b1"] * 5},
            reference_strata={"batch": ["b0"] * 5 + ["b1"] * 5},
        )

        result = _can_generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=None,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )

        assert result is False

    def test_qualify_strategy_and_requires_both(self, make_obs):
        """With 'and', both min_cells AND min_fraction must be met."""
        # Group with 60 cells out of 100 total = 0.6 fraction
        # If min_cells=50 (met) but min_fraction=0.7 (not met), should fail with 'and'
        obs = make_obs(
            n_query=100,
            n_reference=100,
            query_strata={"batch": ["b0"] * 60 + ["b1"] * 40},
            reference_strata={"batch": ["b0"] * 60 + ["b1"] * 40},
        )

        # Strategy 'and': both conditions needed
        result_and = _can_generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=0.7,  # 60/100 = 0.6 < 0.7, so b0 fails fraction
            min_coverage=0.5,
            qualify_strategy="and",
            group_key_internal="psbulk_condition",
        )

        # b0: 60 cells >= 50 (ok), 0.6 < 0.7 (fail) -> doesn't qualify
        # b1: 40 cells < 50 (fail), 0.4 < 0.7 (fail) -> doesn't qualify
        # No qualifying groups -> should fail
        assert result_and is False

    def test_qualify_strategy_or_needs_one(self, make_obs):
        """With 'or', either min_cells OR min_fraction is sufficient."""
        obs = make_obs(
            n_query=100,
            n_reference=100,
            query_strata={"batch": ["b0"] * 60 + ["b1"] * 40},
            reference_strata={"batch": ["b0"] * 60 + ["b1"] * 40},
        )

        # Strategy 'or': either condition sufficient
        result_or = _can_generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=0.7,
            min_coverage=0.5,
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )

        # b0: 60 >= 50 (ok) OR 0.6 < 0.7 (fail) -> qualifies (60 cells)
        # b1: 40 < 50 (fail) OR 0.4 < 0.7 (fail) -> doesn't qualify
        # Coverage: 60/100 = 0.6 >= 0.5 -> passes
        assert result_or is True

    def test_coverage_with_mixed_qualifying_groups(self, make_obs):
        """Ensure coverage is computed as sum(qualifying cells) / total cells."""
        obs = make_obs(
            n_query=125,
            n_reference=125,
            query_strata={"batch": ["b0"] * 100 + ["b1"] * 20 + ["b2"] * 5},
            reference_strata={"batch": ["b0"] * 100 + ["b1"] * 20 + ["b2"] * 5},
        )

        # Only b0 qualifies (100 >= 30)
        # Coverage = 100/125 = 0.8

        # Should pass with min_coverage=0.75
        result_pass = _can_generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=30,
            min_fraction=None,
            min_coverage=0.75,
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )
        assert result_pass is True

        # Should fail with min_coverage=0.85
        result_fail = _can_generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=30,
            min_fraction=None,
            min_coverage=0.85,
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )
        assert result_fail is False

    def test_one_condition_failing_returns_false(self, make_obs):
        """If query passes but reference fails, should return False."""
        # Query has good distribution, reference has poor distribution

        obs = make_obs(
            n_query=100,
            n_reference=20,
            query_strata={"batch": ["b0"] * 50 + ["b1"] * 50},
            reference_strata={"batch": ["b0"] * 10 + ["b1"] * 10},
        )

        result = _can_generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=40,
            min_fraction=None,
            min_coverage=0.5,
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )

        # Query: b0=50 (ok), b1=50 (ok) -> passes
        # Reference: b0=10 (<40), b1=10 (<40) -> no qualifying groups -> False
        assert result is False

    def test_one_condition_failing_with_one_batch(self, make_obs):
        """If query passes but reference fails, should return False."""
        # Query has good distribution, reference has poor distribution

        obs = make_obs(
            n_query=100,
            n_reference=100,
            query_strata={"batch": ["b0"] * 50 + ["b1"] * 50},
            reference_strata={"batch": ["b0"] * 65 + ["b1"] * 35},
        )

        result = _can_generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=40,
            min_fraction=None,
            min_coverage=0.7,
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )

        # query: b0=50 (ok), b1=50 (ok), coverage = 1.0 -> passes
        # reference: b0=65 (ok), b1=35 (<40), coverage = 65/100=0.65 <0.7 -> fails
        assert result is False

    def test_invalid_qualify_strategy_raises(self, make_obs):
        """Invalid qualify_strategy should raise ValueError."""
        obs = make_obs()

        with pytest.raises(ValueError, match="qualify_strategy must be 'and' or 'or'"):
            _can_generate_samples(
                obs=obs,
                stratify_by=["batch"],
                min_cells=10,
                min_fraction=0.1,
                min_coverage=0.5,
                qualify_strategy="invalid",
                group_key_internal="psbulk_condition",
            )

    def test_multiple_strata_columns(self, make_obs):
        """Test with multiple stratification columns."""
        obs = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 40 + ["reference"] * 40,
                "batch": (["b0"] * 20 + ["b1"] * 20) * 2,
                "donor": (["d0"] * 10 + ["d1"] * 10) * 4,
            }
        )

        result = _can_generate_samples(
            obs=obs,
            stratify_by=["batch", "donor"],
            min_cells=5,
            min_fraction=None,
            min_coverage=0.5,
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )

        # Should have 4 groups per condition: (b0,d0), (b0,d1), (b1,d0), (b1,d1)
        # Each with 10 cells
        assert result is True

    def test_min_fraction_boundary(self, make_obs):
        """Test boundary condition for min_fraction."""
        obs = make_obs(
            n_query=100,
            n_reference=100,
            query_strata={"batch": ["b0"] * 20 + ["b1"] * 80},
            reference_strata={"batch": ["b0"] * 20 + ["b1"] * 80},
        )

        # b0 has exactly 0.2 fraction
        result_at_boundary = _can_generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=None,
            min_fraction=0.2,  # Exactly met by b0
            min_coverage=0.1,
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )
        assert result_at_boundary is True

        # Just above boundary should fail for b0
        result_above_boundary = _can_generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=None,
            min_fraction=0.21,  # b0 at 0.2 doesn't meet this
            min_coverage=0.9,  # Need high coverage
            qualify_strategy="or",
            group_key_internal="psbulk_condition",
        )
        # b1 qualifies (0.8 >= 0.21), coverage = 80/100 = 0.8 < 0.9
        assert result_above_boundary is False
