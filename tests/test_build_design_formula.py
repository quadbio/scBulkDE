"""Tests for _build_design_formula function."""

from __future__ import annotations

from scbulkde.ut.ut_basic import _build_design_formula


class TestBuildDesign:
    """Tests for _build_design_formula formula construction."""

    def test_minimal_design_only_condition(self):
        """With no covariates, should only include condition with reference base."""
        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=[],
            factors_continuous=[],
        )

        assert formula == "C(psbulk_condition, contr.treatment(base='reference'))"

    def test_with_categorical_covariates(self):
        """Should add categorical covariates with C() wrapper."""
        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=["batch", "donor"],
            factors_continuous=[],
        )

        assert "C(psbulk_condition, contr.treatment(base='reference'))" in formula
        assert "C(batch)" in formula
        assert "C(donor)" in formula
        assert " + " in formula

    def test_with_continuous_covariates(self):
        """Should add continuous covariates without C() wrapper."""
        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=[],
            factors_continuous=["age", "weight"],
        )

        assert "C(psbulk_condition, contr.treatment(base='reference'))" in formula
        assert "age" in formula
        assert "weight" in formula
        assert "C(age)" not in formula  # Should NOT wrap continuous
        assert " + " in formula

    def test_with_both_categorical_and_continuous(self):
        """Should handle both types of covariates."""
        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=["batch"],
            factors_continuous=["age"],
        )

        assert "C(psbulk_condition, contr.treatment(base='reference'))" in formula
        assert "C(batch)" in formula
        assert "age" in formula
        # Ensure age is not wrapped
        assert formula.count("age") == 1
        assert "C(age)" not in formula

    def test_term_order(self):
        """Condition should come first, then categorical, then continuous."""
        formula = _build_design_formula(
            group_key_internal="condition",
            factors_categorical=["cat1", "cat2"],
            factors_continuous=["cont1", "cont2"],
        )

        terms = formula.split(" + ")
        assert terms[0] == "C(condition, contr.treatment(base='reference'))"
        assert "C(cat1)" in terms[1:3]
        assert "C(cat2)" in terms[1:3]
        assert "cont1" in terms[3:5]
        assert "cont2" in terms[3:5]


class TestBuildDesignIntegration:
    """Integration tests with formulaic model_matrix."""

    def test_formula_parseable_by_formulaic(self):
        """Generated formula should be parseable by formulaic."""
        import pandas as pd
        from formulaic import model_matrix

        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=["batch"],
            factors_continuous=["age"],
        )

        # Create sample data
        data = pd.DataFrame(
            {
                "psbulk_condition": ["query", "reference", "query", "reference"],
                "batch": ["b0", "b0", "b1", "b1"],
                "age": [25, 30, 35, 40],
            }
        )

        # Should not raise
        mm = model_matrix(formula, data=data)
        assert mm.shape[0] == 4
        # Should have: Intercept, condition[query], batch[T.b1], age
        assert mm.shape[1] >= 3

    def test_formula_reference_level_correct(self):
        """Reference level should be 'reference' for condition."""
        import pandas as pd
        from formulaic import model_matrix

        formula = _build_design_formula(
            group_key_internal="psbulk_condition",
            factors_categorical=[],
            factors_continuous=[],
        )

        data = pd.DataFrame(
            {
                "psbulk_condition": ["query", "reference"],
            }
        )

        mm = model_matrix(formula, data=data)

        # When psbulk_condition is 'reference', the coefficient should be 0
        # When 'query', coefficient should be 1
        # Check column names
        assert any("query" in str(col) for col in mm.columns)
