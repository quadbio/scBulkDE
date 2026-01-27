from ._containers import DEResult, PseudobulkResult
from ._logging import logger, set_log_level
from ._performance import performance, set_performance_enabled
from .ut_basic import (
    _aggregate_counts,
    _aggregate_results,
    _build_design,
    _can_generate_samples,
    _compute_required_samples,
    _drop_covariate,
    _fraction_expressing,
    _generate_pseudoreplicate,
    _get_aggregation_function,
    _get_X_and_var_names,
    _prepare_internal_groups,
    _select_groups,
    _select_top_n,
)

__all__ = [
    "DEResult",
    "PseudobulkResult",
    "logger",
    "set_log_level",
    "performance",
    "set_performance_enabled",
    "_prepare_internal_groups",
    "_can_generate_samples",
    "_build_design",
    "_drop_covariate",
    "_get_aggregation_function",
    "_compute_required_samples",
    "_generate_pseudoreplicate",
    "_aggregate_results",
    "_aggregate_counts",
    "_get_X_and_var_names",
    "_select_top_n",
    "_select_groups",
    "_fraction_expressing",
]
