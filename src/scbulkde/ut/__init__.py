from ._containers import DEResult, PseudobulkResult
from ._logging import _in_notebook, logger, set_log_level
from ._performance import performance, set_performance_enabled
from .ut_basic import (
    _aggregate_counts,
    _aggregate_results,
    _build_design_formula,
    _compute_required_samples,
    _drop_covariate,
    _fraction_expressing,
    _generate_pseudoreplicate,
    _generate_samples,
    _get_aggregation_function,
    _get_X_and_var_names,
    _prepare_internal_groups,
    _select_groups,
    _select_top_n,
    _validate_strata,
)

__all__ = [
    "DEResult",
    "PseudobulkResult",
    "_in_notebook",
    "logger",
    "set_log_level",
    "performance",
    "set_performance_enabled",
    "_prepare_internal_groups",
    "_validate_strata",
    "_generate_samples",
    "_build_design_formula",
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
