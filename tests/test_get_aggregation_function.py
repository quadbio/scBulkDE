"""Tests for _get_aggregation_function."""

from __future__ import annotations

import numpy as np
import pytest

from scbulkde.ut.ut_basic import _get_aggregation_function


class TestGetAggregationFunction:
    """Tests for _get_aggregation_function logic."""

    def test_mean_string_returns_np_mean(self):
        """String 'mean' should return np.mean."""
        func = _get_aggregation_function("mean")
        assert func is np.mean

    def test_sum_string_returns_np_sum(self):
        """String 'sum' should return np.sum."""
        func = _get_aggregation_function("sum")
        assert func is np.sum

    def test_median_string_returns_np_median(self):
        """String 'median' should return np.median."""
        func = _get_aggregation_function("median")
        assert func is np.median

    def test_case_insensitive(self):
        """Aggregation strings should be case-insensitive."""
        assert _get_aggregation_function("MEAN") is np.mean
        assert _get_aggregation_function("Sum") is np.sum
        assert _get_aggregation_function("MeDiAn") is np.median

    def test_callable_returned_as_is(self):
        """User-supplied callable should be returned unchanged."""

        def custom_agg(x):
            return x.max()

        func = _get_aggregation_function(custom_agg)
        assert func is custom_agg

    def test_empty_list_returns_none(self):
        """Empty list should return None."""
        func = _get_aggregation_function([])
        assert func is None

    def test_empty_tuple_returns_none(self):
        """Empty tuple should return None."""
        func = _get_aggregation_function(())
        assert func is None

    def test_invalid_string_raises(self):
        """Invalid aggregation string should raise ValueError."""
        with pytest.raises(ValueError, match="not recognized"):
            _get_aggregation_function("invalid")

    def test_non_string_non_callable_raises(self):
        """Non-string, non-callable input should raise ValueError."""
        with pytest.raises(ValueError, match="Aggregation must be"):
            _get_aggregation_function(123)  # type: ignore

    def test_non_empty_list_raises(self):
        """Non-empty list should raise (only empty list/tuple special-cased)."""
        with pytest.raises((ValueError, AttributeError)):
            _get_aggregation_function([1, 2, 3])  # type: ignore

    def test_lambda_function_accepted(self):
        """Lambda functions should be accepted as callable."""
        func = _get_aggregation_function(lambda x: x.std())
        assert callable(func)
        # Test it works
        arr = np.array([1, 2, 3, 4, 5])
        result = func(arr)
        assert result == arr.std()

    def test_custom_allow_set(self):
        """Custom allow set should restrict valid strings."""
        func = _get_aggregation_function("mean", allow={"mean", "custom"})
        assert func is np.mean

        with pytest.raises(ValueError, match="not recognized"):
            _get_aggregation_function("median", allow={"mean", "custom"})

    def test_none_input_raises(self):
        """None input should raise ValueError."""
        with pytest.raises((ValueError, AttributeError)):
            _get_aggregation_function(None)  # type: ignore


class TestAggregationFunctionBehavior:
    """Test that returned functions behave correctly."""

    def test_mean_function_computes_correctly(self):
        """np.mean should compute correct mean."""
        func = _get_aggregation_function("mean")
        arr = np.array([1, 2, 3, 4, 5])
        assert func(arr) == 3.0

    def test_sum_function_computes_correctly(self):
        """np.sum should compute correct sum."""
        func = _get_aggregation_function("sum")
        arr = np.array([1, 2, 3, 4, 5])
        assert func(arr) == 15

    def test_median_function_computes_correctly(self):
        """np.median should compute correct median."""
        func = _get_aggregation_function("median")
        arr = np.array([1, 2, 3, 4, 5])
        assert func(arr) == 3.0

    def test_custom_function_computes_correctly(self):
        """Custom function should work on data."""

        def geometric_mean(x):
            return np.exp(np.log(x).mean())

        func = _get_aggregation_function(geometric_mean)
        arr = np.array([1, 2, 4, 8])
        result = func(arr)
        expected = geometric_mean(arr)
        assert np.isclose(result, expected)
