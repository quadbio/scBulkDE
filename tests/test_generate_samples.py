"""Tests for _generate_samples function."""

from __future__ import annotations

import pandas as pd
import pytest

from scbulkde.ut.ut_basic import _generate_samples


class TestGenerateSamples:
    def test_empty_stratify_by_returns_false_and_empty_df(self, make_obs):
        obs = make_obs(
            n_query=2,
            n_reference=1,
            condition_col="condition",
            strata={"batch": {"query": ["A", "A"], "reference": ["B"]}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=[],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 0
        assert isinstance(sample_stats, pd.DataFrame)
        assert len(sample_stats) == 0

    def test_no_query_cells_returns_false(self, make_obs):
        obs = make_obs(
            n_query=0,
            n_reference=3,
            condition_col="condition",
            strata={"batch": {"query": [], "reference": ["A", "B", "C"]}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0
        assert len(sample_stats) == 0

    def test_no_reference_cells_returns_false(self, make_obs):
        obs = make_obs(
            n_query=3,
            n_reference=0,
            condition_col="condition",
            strata={"batch": {"query": ["A", "B", "C"], "reference": []}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0
        assert len(sample_stats) == 0

    def test_min_fraction_only_filters_by_relative_size(self, make_obs):
        obs = make_obs(
            n_query=100,
            n_reference=100,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 30 + ["B"] * 70, "reference": ["A"] * 15 + ["B"] * 85}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=None,
            min_fraction=0.2,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is True
        assert len(filtered) == 185

        reference_filtered = filtered[filtered["condition"] == "reference"]
        assert set(reference_filtered["batch"].unique()) == {"B"}

    def test_invalid_qualify_strategy_raises_error(self, make_obs):
        obs = make_obs(
            n_query=2,
            n_reference=2,
            condition_col="condition",
            strata={"batch": {"query": ["A", "B"], "reference": ["A", "B"]}},
        )

        # Both min_cells and min_fraction must be not None in order to enter the
        # if statement that checks qualify_strategy
        with pytest.raises(ValueError, match="qualify_strategy must be 'and' or 'or'"):
            _generate_samples(
                obs=obs,
                stratify_by=["batch"],
                min_cells=1,
                min_fraction=1,
                min_coverage=None,
                qualify_strategy="invalid",
                group_key_internal="condition",
            )

    def test_min_coverage_enforces_minimum_cell_retention(self, make_obs):
        obs = make_obs(
            n_query=100,
            n_reference=100,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10 + ["B"] * 90, "reference": ["A"] * 10 + ["B"] * 90}},
        )

        can_generate_pass, _, _ = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=None,
            min_coverage=0.8,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        can_generate_fail, _, _ = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=50,
            min_fraction=None,
            min_coverage=0.95,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate_pass is True
        assert can_generate_fail is False

    def test_empty_obs_returns_false(self, make_obs):
        obs = make_obs(
            n_query=0, n_reference=0, condition_col="condition", strata={"batch": {"query": [], "reference": []}}
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0

    def test_no_requirements_specified_returns_false(self, make_obs):
        obs = make_obs(
            n_query=10,
            n_reference=10,
            condition_col="condition",
            strata={"batch": {"query": ["A"] * 10, "reference": ["A"] * 10}},
        )

        can_generate, filtered, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=["batch"],
            min_cells=None,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            group_key_internal="condition",
        )

        assert can_generate is False
        assert len(filtered) == 0
