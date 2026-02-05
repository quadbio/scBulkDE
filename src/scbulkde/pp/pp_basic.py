from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from formulaic import model_matrix

from scbulkde.ut._containers import PseudobulkResult
from scbulkde.ut._logging import logger
from scbulkde.ut.ut_basic import (
    _aggregate_counts,
    _build_design,
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

    # Label cells as 'query' or 'reference' in a new internal column
    # Also raises ValueError if there are no cells in either query or reference
    obs = _prepare_internal_groups(
        adata=adata, group_key=group_key, group_key_internal=group_key_internal, query=query, reference=reference
    )
    cell_counts = obs[group_key_internal].value_counts()
    logger.info(f"Using {cell_counts['query']} query and {cell_counts['reference']} reference cells for pseudobulking.")

    # Validate strata and potentially drop covariates to meet sample requirements. This leads to two cases
    # 1) There are no strata provided or all strata have been dropped. In this case no independent samples can be created
    #    and pseudoreplicates have to be created. Using all cells as a first sample and then creating additional pseudoreplicates
    #    is not acceptable as it would increase the dependency between the samples.
    # 2) There are strata provided that allow to create independent samples

    strata = _validate_strata(
        obs=obs,
        strata=categorical_covariates,
        min_cells=min_cells,
        min_fraction=min_fraction,
        min_coverage=min_coverage,
        qualify_strategy=qualify_strategy,
        covariate_strategy=covariate_strategy,
        group_key_internal=group_key_internal,
        resolve_conflicts=resolve_conflicts,
    )

    # Subset adata to relevant cells
    adata_sub = adata[obs.index, :]

    # Handle the case where no valid strata exist
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

    # Build sample table, design matrix, and aggregate counts
    return _build_pseudobulk_result(
        adata_sub=adata_sub,
        obs=obs,
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
    # Create empty pseudobulk counts DataFrame (0 rows, genes as columns)
    pb_counts = pd.DataFrame(columns=adata_sub.var_names)

    # Create the design formula (just the condition column)
    design_formula = _build_design(
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
        grouped=None,
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
        design_formula = _build_design(
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
    design_formula = _build_design(
        group_key_internal=group_key_internal,
        factors_categorical=[],
        factors_continuous=[],
    )
    mm = model_matrix(design_formula, data=sample_table)
    logger.info(f"Design matrix with shape {mm.shape} using minimal design formula:\n{design_formula}")

    return design_formula, mm

    # if not strata:

    #     # Warn the user that no replicate_key or covariates are provided
    #     logger.warning("No replicate_key or categorical_covariates provided. Cannot create independent samples - returning empty pseudobulk counts.")

    #     # Create empty pseudobulk counts DataFrame
    #     pb_counts = pd.DataFrame(columns=adata.var_names)

    #     # Create the design. This will just the condition column, using _build_design for consistency
    #     design_formula = _build_design(
    #         group_key_internal=group_key_internal,
    #         factors_categorical=[],
    #         factors_continuous=[],
    #     )

    #     # Create the sample table. This will just be two rows for the two conditions
    #     sample_table = pd.DataFrame({
    #         group_key_internal: obs[group_key_internal].unique()
    #     })

    #     # Create the design matrix
    #     mm = model_matrix(design_formula, data=sample_table)

    # else:

    #     while True:

    #         # Check if samples can be generated given the current strata
    #         if _can_generate_samples(
    #             obs,
    #             stratify_by=strata,
    #             min_cells=min_cells,
    #             min_fraction=min_fraction,
    #             min_coverage=min_coverage,
    #             qualify_strategy=qualify_strategy,
    #             group_key_internal=group_key_internal,
    #         ):
    #             break

    #         # If strata empty: Either use all cells for query and reference or raise error
    #         if not strata and resolve_conflicts:
    #             logger.warning(f"Cannot generate samples stratifying by {strata}. Falling back to using all cells.")
    #             break
    #         elif not strata and not resolve_conflicts:
    #             raise ValueError(f"Cannot generate samples stratifying by {strata}. and no covariates left to drop.")

    #         # If strata not empty and not able to generate samples: Drop a covariate
    #         strata, dropped = _drop_covariate(covariates=strata, obs=obs, covariate_strategy=covariate_strategy)
    #         logger.warning(f"Dropped covariate: {dropped} to meet sample requirements.")

    #     # Now we know which strata can be used to generate samples. Let's summarize it into a design table.
    #     sample_factors_categorical = [group_key_internal] + strata
    #     sample_factors_continuous = continuous_covariates if continuous_covariates is not None else []

    #     obs_grouped = obs.groupby(sample_factors_categorical, observed=True, sort=False)

    #     if sample_factors_continuous:
    #         agg_func = _get_aggregation_function(continuous_aggregation)
    #         sample_table = obs_grouped[sample_factors_continuous].agg(agg_func).reset_index()
    #     else:
    #         sample_table = obs_grouped.first().reset_index()[sample_factors_categorical]

    #     # Based on that, we can now decide on the design formula. Here, we will use the sample table with one important
    #     # detail: although the replicate key does not necessarily introduce collinearity in the columns of the design matrix
    #     # it is exclusively used as a means to generate a sample stratification. In the majority of setups, it does not make
    #     # sense to include the replicate_key in the design
    #     # Let's also remove the group_key_internal from the design_factors here, as we will add it explicitly later with the correct reference level.
    #     design_factors_categorical_formula = [
    #         f for f in sample_factors_categorical if f != replicate_key and f != group_key_internal
    #     ]
    #     design_factors_continuous_formula = sample_factors_continuous.copy()

    #     while True:
    #         # Although the while loop should be exited at some point, this is a bit risky. Maybe let's put a safeguard in the future
    #         design_formula = _build_design(
    #             group_key_internal=group_key_internal,
    #             factors_categorical=design_factors_categorical_formula,
    #             factors_continuous=design_factors_continuous_formula,
    #         )
    #         mm = model_matrix(design_formula, data=sample_table)
    #         if np.linalg.matrix_rank(mm.values) == mm.shape[1]:
    #             logger.info(f"Design matrix with shape {mm.shape} has full rank using design formula:\n{design_formula}")
    #             break

    #         # If there are categorical factors, consume them first, as each generates n - 1 level columns
    #         if design_factors_categorical_formula:
    #             design_factors_categorical_formula, dropped = _drop_covariate(
    #                 covariates=design_factors_categorical_formula, obs=sample_table, covariate_strategy=covariate_strategy
    #             )
    #             logger.warning(f"Dropped categorical covariate '{dropped}' to achieve full column rank.")
    #             continue

    #         # If this didn't help, consume continuous covariates. Each one only generates one column
    #         # And it is unlikely that they cause collinearity unless they are constant. In that case the algorithm
    #         # would converge if all constant and continuous covariates are removed
    #         if design_factors_continuous_formula:
    #             design_factors_continuous_formula, dropped = _drop_covariate(
    #                 covariates=design_factors_continuous_formula,
    #                 obs=sample_table,
    #                 covariate_strategy="sequence_order",  # enforce sequence order for continuous
    #             )
    #             logger.warning(f"Dropped continuous covariate '{dropped}' to achieve full rank.")
    #             continue

    #     # Now we can finally aggregate the counts into pseudobulk samples
    #     pb_counts = _aggregate_counts(
    #         adata=adata, grouped_obs=obs_grouped, layer=layer, layer_aggregation=layer_aggregation
    #     )

    # # We want to also store the original per-cell count matrix because it might be needed for generating pseudoreplicates
    # adata_sub = adata[obs.index, :]

    # return PseudobulkResult(
    #     adata_sub=adata_sub,
    #     pb_counts=pb_counts,
    #     grouped=obs_grouped,
    #     sample_table=sample_table,
    #     design_matrix=mm,
    #     design_formula=design_formula,
    #     group_key=group_key,
    #     group_key_internal=group_key_internal,
    #     query=query,
    #     reference=reference,
    #     strata=strata,
    #     layer=layer,
    #     layer_aggregation=layer_aggregation,
    #     categorical_covariates=categorical_covariates,
    #     continuous_covariates=continuous_covariates,
    #     continuous_aggregation=continuous_aggregation,
    #     min_cells=min_cells,
    #     min_fraction=min_fraction,
    #     min_coverage=min_coverage,
    #     qualify_strategy=qualify_strategy,
    #     n_cells=cell_counts,
    # )
