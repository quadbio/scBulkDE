"""Tests for _build_full_rank_design function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestBuildFullRankDesign:
    """Tests for _build_full_rank_design logic."""

    @pytest.fixture
    def sample_table_full_rank(self):
        """Sample table that should produce full rank design."""
        return pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference"],
                "batch": ["b0", "b1", "b0", "b1"],
            }
        )

    @pytest.fixture
    def sample_table_rank_deficient(self):
        """Sample table that creates rank deficiency with all covariates."""
        # When condition and batch are perfectly confounded
        return pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference"],
                "batch": ["b0", "b0", "b1", "b1"],  # Confounded with condition
            }
        )

    def test_full_rank_design_returns_immediately(self, sample_table_full_rank):
        """When design is full rank, should return without dropping."""
        from scbulkde.pp.pp_basic import _build_full_rank_design

        formula, mm = _build_full_rank_design(
            sample_table=sample_table_full_rank,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["batch"],
            design_factors_continuous=[],
            covariate_strategy="sequence_order",
        )

        # Should have full rank
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]
        # Should include batch
        assert "batch" in formula

    def test_rank_deficient_drops_categorical_first(self, sample_table_rank_deficient):
        """When rank deficient, should drop categorical covariates first."""
        from scbulkde.pp.pp_basic import _build_full_rank_design

        formula, mm = _build_full_rank_design(
            sample_table=sample_table_rank_deficient,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["batch"],
            design_factors_continuous=[],
            covariate_strategy="sequence_order",
        )

        # Should drop batch to achieve full rank
        assert "C(batch)" not in formula
        # Should still have condition
        assert "psbulk_condition" in formula
        # Should have full rank
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]

    def test_drops_continuous_after_categorical(self):
        """Should drop categorical first, then continuous."""
        from scbulkde.pp.pp_basic import _build_full_rank_design

        # Create data where we have rank deficiency
        sample_table = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 4 + ["reference"] * 4,
                "cat1": ["c0", "c1"] * 4,  # Not correlated with condition
                "cat2": ["c0", "c1"] * 4,  # Correlated with cat1
                "cont1": [1.0, 2.0] * 4,  # Not correlated with condition
            }
        )

        # The initial model matrix looks like:

        #   Intercept  [T.query]  C(cat1)[T.c1]  C(cat2)[T.c1]  cont1
        # 0        1.0          1              0              0    1.0
        # 1        1.0          1              1              1    2.0
        # 2        1.0          1              0              0    1.0
        # 3        1.0          1              1              1    2.0
        # 4        1.0          0              0              0    1.0
        # 5        1.0          0              1              1    2.0
        # 6        1.0          0              0              0    1.0

        # Then cat1 and cat2 are perfectly correlated, so one must be dropped.
        # The covariate strategy is sequence_order, so cat2 should be dropped first.
        # The model matrix after dropping cat2:

        #   Intercept  [T.query]  C(cat1)[T.c1]  cont1
        # 0        1.0          1              0    1.0
        # 1        1.0          1              1    2.0
        # 2        1.0          1              0    1.0
        # 3        1.0          1              1    2.0
        # 4        1.0          0              0    1.0
        # 5        1.0          0              1    2.0
        # 6        1.0          0              0    1.0

        # The model matrix is still not full rank, because cont1 = Intercept + C(cat1)[T.c1]
        # So cat1 should be dropped next, resulting in a full rank model matrix

        formula, mm = _build_full_rank_design(
            sample_table=sample_table,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["cat1", "cat2"],
            design_factors_continuous=["cont1"],
            covariate_strategy="sequence_order",
        )

        # Should achieve full rank
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]

        # Check the correct covariates were dropped
        assert "psbulk_condition" in formula
        assert "C(cat1)" not in formula
        assert "C(cat2)" not in formula
        assert "cont1" in formula

    def test_covariate_strategy_sequence_order(self):
        """sequence_order should drop from end of list first."""
        from scbulkde.pp.pp_basic import _build_full_rank_design

        sample_table = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 2 + ["reference"] * 2,
                "first": ["a", "b", "a", "b"],
                "second": ["x", "x", "y", "y"],  # Confounded
            }
        )

        formula, mm = _build_full_rank_design(
            sample_table=sample_table,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["first", "second"],
            design_factors_continuous=[],
            covariate_strategy="sequence_order",
        )

        # Should drop 'second' first (last in sequence)
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]
        assert "psbulk_condition" in formula
        assert "C(first)" in formula
        assert "C(second)" not in formula

    def test_covariate_strategy_most_levels(self):
        """most_levels should drop covariate with most levels first."""
        from scbulkde.pp.pp_basic import _build_full_rank_design

        # The "few" category cause the rank deficiency, but "many" has more levels.
        # When using most_levels strategy, "many" should be dropped first, and then
        # because the matrix is still rank deficient, "few" should be dropped next.
        sample_table = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 6 + ["reference"] * 6,
                "few": ["a"] * 6 + ["b"] * 6,  # 2 levels
                "many": ["x", "y", "z", "x", "y", "z"] * 2,  # 3 levels
            }
        )

        formula, mm = _build_full_rank_design(
            sample_table=sample_table,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["few", "many"],
            design_factors_continuous=[],
            covariate_strategy="most_levels",
        )

        # Should drop 'many' first (has more levels)
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]
        assert "psbulk_condition" in formula
        assert "C(few)" not in formula
        assert "C(many)" not in formula

    def test_continuous_dropped_with_sequence_order_strategy(self):
        """When dropping continuous, always use sequence_order."""
        from scbulkde.pp.pp_basic import _build_full_rank_design

        sample_table = pd.DataFrame(
            {
                "psbulk_condition": ["query"] * 4 + ["reference"] * 4,
                "cont1": [1.0, 2.0, 3.0, 4.0] * 2,
                "cont2": [1.0, 2.0, 3.0, 4.0] * 2,  # Perfectly correlated
            }
        )

        formula, mm = _build_full_rank_design(
            sample_table=sample_table,
            group_key_internal="psbulk_condition",
            design_factors_categorical=[],
            design_factors_continuous=["cont1", "cont2"],
            covariate_strategy="most_levels",  # Shouldn't matter for continuous
        )

        # Should drop cont2 (last in sequence)
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]
        assert "cont2" not in formula

    def test_fallback_to_minimal_design(self):
        """If all covariates dropped, should return minimal design with just condition."""
        from scbulkde.pp.pp_basic import _build_full_rank_design

        # Create highly confounded data
        sample_table = pd.DataFrame(
            {
                "psbulk_condition": ["query", "reference"],
                "cat": ["a", "b"],  # Perfectly confounded with condition
            }
        )

        formula, mm = _build_full_rank_design(
            sample_table=sample_table,
            group_key_internal="psbulk_condition",
            design_factors_categorical=["cat"],
            design_factors_continuous=[],
            covariate_strategy="sequence_order",
        )

        # Should only have condition
        assert "psbulk_condition" in formula
        assert "C(cat)" not in formula
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]

    def test_empty_covariates_returns_minimal_design(self):
        """With no covariates, should return just condition."""
        from scbulkde.pp.pp_basic import _build_full_rank_design

        sample_table = pd.DataFrame(
            {
                "psbulk_condition": ["query", "query", "reference", "reference"],
            }
        )

        formula, mm = _build_full_rank_design(
            sample_table=sample_table,
            group_key_internal="psbulk_condition",
            design_factors_categorical=[],
            design_factors_continuous=[],
            covariate_strategy="sequence_order",
        )

        assert "psbulk_condition" in formula
        assert np.linalg.matrix_rank(mm.values) == mm.shape[1]
        # Should have 2 columns: Intercept and condition
        assert mm.shape[1] == 2
