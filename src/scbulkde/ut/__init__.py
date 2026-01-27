"""Utility functions."""

from ._containers import DEResult, PseudobulkResult
from ._logging import logger, set_log_level
from ._performance import performance
from .ut_basic import (
    _aggregate_counts,
    _build_design,
    _can_generate_samples,
    _drop_covariate,
    _get_aggregation_function,
    _prepare_internal_groups,
)

__all__ = [
    "DEResult",
    "PseudobulkResult",
    "logger",
    "set_log_level",
    "performance",
    "_prepare_internal_groups",
    "_can_generate_samples",
    "_build_design",
    "_drop_covariate",
    "_get_aggregation_function",
    "_aggregate_counts",
]
