from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from formulaic import model_matrix

from scbulkde.ut._containers import PseudobulkResult
from scbulkde.ut._logging import logger
from scbulkde.ut.ut_basic import (
    _aggregate_counts,
    _build_design_formula,
    _drop_covariate,
    _get_aggregation_function,
    _prepare_internal_groups,
    _validate_strata,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Literal

    import anndata as ad


def pseudobulk(
    adata: ad.AnnData,
    group_key: str,
    query: str | Sequence[str],
    reference: str | Sequence[str] = "rest",
    *,
    replicate_key: str | None = None,
    min_cells: int | None = 50,
    min_fraction: float | None = 0.2,
    min_coverage: float | None = 0.75,
    categorical_covariates: Sequence[str] | None = None,
    continuous_covariates: Sequence[str] | None = None,
    continuous_aggregation: Literal["mean", "sum", "median"] | Callable | None = "mean",
    layer: str | None = None,
    layer_aggregation: Literal["sum", "mean"] = "sum",
    qualify_strategy: Literal["and", "or"] = "or",
    covariate_strategy: Literal["sequence_order", "most_levels"] = "sequence_order",
    resolve_conflicts: bool = True,
):
    """Main function to perform pseudobulking on an AnnData object."""
    group_key_internal = "psbulk_condition"

    # Label cells as 'query' or 'reference'
    # This also subsets the obs to only contain query and reference cells
    obs = _prepare_internal_groups(
        adata=adata, group_key=group_key, group_key_internal=group_key_internal, query=query, reference=reference
    )
    cell_counts = obs[group_key_internal].value_counts()
    logger.info(f"Using {cell_counts['query']} query and {cell_counts['reference']} reference cells for pseudobulking.")

    # Combine replicate_key and categorical_covariates
    strata_list = []
    if replicate_key is not None:
        strata_list.append(replicate_key)
    if categorical_covariates is not None:
        strata_list.extend(categorical_covariates)

    # Validate strata and get filtered obs with only qualifying cells
    strata, obs_filtered = _validate_strata(
        obs=obs,
        strata=strata_list,
        min_cells=min_cells,
        min_fraction=min_fraction,
        min_coverage=min_coverage,
        qualify_strategy=qualify_strategy,
        covariate_strategy=covariate_strategy,
        group_key_internal=group_key_internal,
        resolve_conflicts=resolve_conflicts,
    )

    # Use filtered obs (contains only cells in qualifying samples)
    obs = obs_filtered if not obs_filtered.empty else obs

    # Subset adata to relevant cells
    adata_sub = adata[obs.index, :]

    # Handle empty strata case
    if not strata:
        return _build_empty_pseudobulk_result(
            adata_sub=adata_sub,
            obs=obs,
            group_key=group_key,
            group_key_internal=group_key_internal,
            query=query,
            reference=reference,
            layer=layer,
            layer_aggregation=layer_aggregation,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
            continuous_aggregation=continuous_aggregation,
            min_cells=min_cells,
            min_fraction=min_fraction,
            min_coverage=min_coverage,
            qualify_strategy=qualify_strategy,
            n_cells=cell_counts,
        )

    # Build result with filtered data
    return _build_pseudobulk_result(
        adata_sub=adata_sub,
        obs=obs,  # Already filtered to qualifying cells only
        strata=strata,
        group_key=group_key,
        group_key_internal=group_key_internal,
        query=query,
        reference=reference,
        replicate_key=replicate_key,
        layer=layer,
        layer_aggregation=layer_aggregation,
        categorical_covariates=categorical_covariates,
        continuous_covariates=continuous_covariates,
        continuous_aggregation=continuous_aggregation,
        covariate_strategy=covariate_strategy,
        min_cells=min_cells,
        min_fraction=min_fraction,
        min_coverage=min_coverage,
        qualify_strategy=qualify_strategy,
        n_cells=cell_counts,
    )


def _build_empty_pseudobulk_result(
    adata_sub: ad.AnnData,
    obs: pd.DataFrame,
    group_key: str,
    group_key_internal: str,
    query: str | Sequence[str],
    reference: str | Sequence[str],
    layer: str | None,
    layer_aggregation: str,
    categorical_covariates: Sequence[str] | None,
    continuous_covariates: Sequence[str] | None,
    continuous_aggregation: str | None,
    min_cells: int | None,
    min_fraction: float | None,
    min_coverage: float | None,
    qualify_strategy: str,
    n_cells: pd.Series,
) -> PseudobulkResult:
    """
    Build a PseudobulkResult with empty pb_counts when no valid strata exist.

    The sample_table and design_matrix are still properly constructed with
    the two conditions (query/reference), but pb_counts has 0 rows.
    """
    obs_grouped = obs.groupby(group_key_internal, observed=True, sort=False)

    # Create empty pseudobulk counts DataFrame (0 rows, genes as columns)
    pb_counts = pd.DataFrame(columns=adata_sub.var_names)

    # Create the design formula (just the condition column)
    design_formula = _build_design_formula(
        group_key_internal=group_key_internal,
        factors_categorical=[],
        factors_continuous=[],
    )

    # Create the sample table with two rows for the two conditions
    sample_table = pd.DataFrame({group_key_internal: obs[group_key_internal].unique()})

    # Create the design matrix
    mm = model_matrix(design_formula, data=sample_table)

    return PseudobulkResult(
        adata_sub=adata_sub,
        pb_counts=pb_counts,
        grouped=obs_grouped,
        sample_table=sample_table,
        design_matrix=mm,
        design_formula=design_formula,
        group_key=group_key,
        group_key_internal=group_key_internal,
        query=query,
        reference=reference,
        strata=[],
        layer=layer,
        layer_aggregation=layer_aggregation,
        categorical_covariates=categorical_covariates,
        continuous_covariates=continuous_covariates,
        continuous_aggregation=continuous_aggregation,
        min_cells=min_cells,
        min_fraction=min_fraction,
        min_coverage=min_coverage,
        qualify_strategy=qualify_strategy,
        n_cells=n_cells,
    )


def _build_pseudobulk_result(
    adata_sub: ad.AnnData,
    obs: pd.DataFrame,
    strata: list[str],
    group_key: str,
    group_key_internal: str,
    query: str | Sequence[str],
    reference: str | Sequence[str],
    replicate_key: str | None,
    layer: str | None,
    layer_aggregation: str,
    categorical_covariates: Sequence[str] | None,
    continuous_covariates: Sequence[str] | None,
    continuous_aggregation: str | None,
    covariate_strategy: str,
    min_cells: int | None,
    min_fraction: float | None,
    min_coverage: float | None,
    qualify_strategy: str,
    n_cells: pd.Series,
) -> PseudobulkResult:
    """Build a PseudobulkResult with aggregated counts when valid strata exist."""
    sample_factors_categorical = [group_key_internal] + strata
    sample_factors_continuous = list(continuous_covariates) if continuous_covariates else []

    obs_grouped = obs.groupby(sample_factors_categorical, observed=True, sort=False)

    # Build sample table
    if sample_factors_continuous:
        agg_func = _get_aggregation_function(continuous_aggregation)
        sample_table = obs_grouped[sample_factors_continuous].agg(agg_func).reset_index()
    else:
        sample_table = obs_grouped.first().reset_index()[sample_factors_categorical]

    # Build design formula, excluding replicate_key and group_key_internal
    design_factors_categorical = [f for f in strata if f != replicate_key]
    design_factors_continuous = sample_factors_continuous.copy()

    # Iteratively build design matrix, dropping covariates if needed for full rank
    design_formula, mm = _build_full_rank_design(
        sample_table=sample_table,
        group_key_internal=group_key_internal,
        design_factors_categorical=design_factors_categorical,
        design_factors_continuous=design_factors_continuous,
        covariate_strategy=covariate_strategy,
    )

    # Aggregate counts into pseudobulk samples
    pb_counts = _aggregate_counts(
        adata=adata_sub, grouped_obs=obs_grouped, layer=layer, layer_aggregation=layer_aggregation
    )

    return PseudobulkResult(
        adata_sub=adata_sub,
        pb_counts=pb_counts,
        grouped=obs_grouped,
        sample_table=sample_table,
        design_matrix=mm,
        design_formula=design_formula,
        group_key=group_key,
        group_key_internal=group_key_internal,
        query=query,
        reference=reference,
        strata=strata,
        layer=layer,
        layer_aggregation=layer_aggregation,
        categorical_covariates=categorical_covariates,
        continuous_covariates=continuous_covariates,
        continuous_aggregation=continuous_aggregation,
        min_cells=min_cells,
        min_fraction=min_fraction,
        min_coverage=min_coverage,
        qualify_strategy=qualify_strategy,
        n_cells=n_cells,
    )


def _build_full_rank_design(
    sample_table: pd.DataFrame,
    group_key_internal: str,
    design_factors_categorical: list[str],
    design_factors_continuous: list[str],
    covariate_strategy: str,
) -> tuple[str, pd.DataFrame]:
    """
    Build a full-rank design matrix, dropping covariates if necessary.

    Returns the design formula and design matrix.
    """
    max_iterations = len(design_factors_categorical) + len(design_factors_continuous) + 1

    for _ in range(max_iterations):
        design_formula = _build_design_formula(
            group_key_internal=group_key_internal,
            factors_categorical=design_factors_categorical,
            factors_continuous=design_factors_continuous,
        )
        mm = model_matrix(design_formula, data=sample_table)

        if np.linalg.matrix_rank(mm.values) == mm.shape[1]:
            logger.info(f"Design matrix with shape {mm.shape} has full rank using design formula:\n{design_formula}")
            return design_formula, mm

        # Drop categorical covariates first (they generate more columns)
        if design_factors_categorical:
            design_factors_categorical, dropped = _drop_covariate(
                covariates=design_factors_categorical,
                obs=sample_table,
                covariate_strategy=covariate_strategy,
            )
            logger.warning(f"Dropped categorical covariate '{dropped}' to achieve full column rank.")
            continue

        # Then drop continuous covariates
        if design_factors_continuous:
            design_factors_continuous, dropped = _drop_covariate(
                covariates=design_factors_continuous,
                obs=sample_table,
                covariate_strategy="sequence_order",
            )
            logger.warning(f"Dropped continuous covariate '{dropped}' to achieve full rank.")
            continue

        # No more covariates to drop - this shouldn't happen with just the intercept
        break

    # Final attempt with no additional covariates
    design_formula = _build_design_formula(
        group_key_internal=group_key_internal,
        factors_categorical=[],
        factors_continuous=[],
    )
    mm = model_matrix(design_formula, data=sample_table)
    logger.info(f"Design matrix with shape {mm.shape} using minimal design formula:\n{design_formula}")

    return design_formula, mm
