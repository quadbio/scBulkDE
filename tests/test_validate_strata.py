"""Tests for _validate_strata function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scbulkde.ut.ut_basic import _validate_strata


class TestValidateStrata:
    def test_none_strata_returns_empty(self):
        """When strata is None, should return empty results."""
        obs = pd.DataFrame({"condition": ["query", "reference"], "batch": ["A", "A"]})

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=None,
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert valid_strata == []
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 0
        assert isinstance(sample_stats, pd.DataFrame)
        assert len(sample_stats) == 0

    def test_empty_strata_list_returns_empty(self):
        """When strata is empty list, should return empty results."""
        obs = pd.DataFrame({"condition": ["query", "reference"], "batch": ["A", "A"]})

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=[],
            min_cells=1,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert valid_strata == []
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) == 0
        assert len(sample_stats) == 0

    def test_resolve_conflicts_true_returns_empty_when_impossible(self):
        """With resolve_conflicts=True and impossible requirements, should return empty."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 10 + ["reference"] * 10,
                "batch": ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5,
                "cell_id": range(20),
            }
        )

        valid_strata, filtered, sample_stats = _validate_strata(
            obs=obs,
            strata=["batch"],
            min_cells=1000,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert valid_strata == []
        assert len(filtered) == 0

    def test_resolve_conflicts_false_raises_when_impossible(self):
        """With resolve_conflicts=False and impossible requirements, should raise."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 10 + ["reference"] * 10,
                "batch": ["A"] * 5 + ["B"] * 5 + ["A"] * 5 + ["B"] * 5,
                "cell_id": range(20),
            }
        )

        with pytest.raises(ValueError):
            _validate_strata(
                obs=obs,
                strata=["batch"],
                min_cells=1000,
                min_fraction=None,
                min_coverage=None,
                qualify_strategy="and",
                covariate_strategy="sequence_order",
                group_key_internal="condition",
                resolve_conflicts=False,
            )

    def test_returns_subset_or_equal_of_input_strata(self):
        """Returned strata should be subset (or equal to) input strata."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 50 + ["reference"] * 50,
                "batch": np.random.choice(["A", "B", "C"], 100),
                "donor": np.random.choice(["D1", "D2"], 100),
                "tissue": np.random.choice(["T1", "T2"], 100),
                "cell_id": range(100),
            }
        )

        input_strata = ["batch", "donor", "tissue"]

        valid_strata, _, _ = _validate_strata(
            obs=obs,
            strata=input_strata,
            min_cells=10,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="and",
            covariate_strategy="sequence_order",
            group_key_internal="condition",
            resolve_conflicts=True,
        )

        assert set(valid_strata).issubset(set(input_strata))

    def test_invalid_covariate_strategy_raises_error(self):
        """Should raise error for invalid covariate_strategy."""
        obs = pd.DataFrame(
            {
                "condition": ["query"] * 20 + ["reference"] * 20,
                "batch": ["A"] * 10 + ["B"] * 10 + ["A"] * 10 + ["B"] * 10,
                "cell_id": range(40),
            }
        )

        with pytest.raises(ValueError):
            _validate_strata(
                obs=obs,
                strata=["batch"],
                min_cells=100,
                min_fraction=None,
                min_coverage=None,
                qualify_strategy="and",
                covariate_strategy="invalid_strategy",
                group_key_internal="condition",
                resolve_conflicts=False,
            )
