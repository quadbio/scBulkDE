"""Utility functions."""

from ._containers import DEResult, PseudobulkResult
from ._logging import logger, set_log_level
from ._performance import performance
from ._validation import validate_adata, validate_groups
from .ut_basic import _fraction_expressing, _get_X_and_var_names, _select_groups, _select_top_n, aggregate_counts

__all__ = [
    "aggregate_counts",
    "PseudobulkResult",
    "DEResult",
    "logger",
    "set_log_level",
    "validate_adata",
    "validate_groups",
    "performance",
    "_get_X_and_var_names",
    "_select_top_n",
    "_select_groups",
    "_fraction_expressing",
]
