from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from formulaic import model_matrix

from scbulkde.ut._containers import PseudobulkResult
from scbulkde.ut._logging import logger
from scbulkde.ut.ut_basic import (
    _aggregate_counts,
    _build_design,
    _can_generate_samples,
    _drop_covariate,
    _get_aggregation_function,
    _prepare_internal_groups,
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
    obs = _prepare_internal_groups(
        adata=adata, group_key=group_key, group_key_internal=group_key_internal, query=query, reference=reference
    )
    vc = obs[group_key_internal].value_counts()
    logger.info(f"Using {vc['query']} query and {vc['reference']} reference cells for pseudobulking.")

    # Check if samples can be generated through grouping by replicate_key and the categorical covariates
    strata = []
    if replicate_key:
        strata.append(replicate_key)
    if categorical_covariates:
        strata.extend(categorical_covariates)

    while True:
        # The user might intentionally not provide a repliate key or covariates
        if not strata:
            logger.info("No replicate_key or categorical_covariates provided. Using all cells for pseudobulking.")
            break

        # Check if samples can be generated given the current strata
        if _can_generate_samples(
            obs,
            stratify_by=strata,
            min_cells=min_cells,
            min_fraction=min_fraction,
            min_coverage=min_coverage,
            qualify_strategy=qualify_strategy,
            group_key_internal=group_key_internal,
        ):
            break

        # If strata empty: Either use all cells for query and reference or raise error
        if not strata and resolve_conflicts:
            logger.warning(f"Cannot generate samples stratifying by {strata}. Falling back to using all cells.")
            break
        elif not strata and not resolve_conflicts:
            raise ValueError(f"Cannot generate samples stratifying by {strata}. and no covariates left to drop.")

        # If strata not empty and not able to generate samples: Drop a covariate
        strata, dropped = _drop_covariate(covariates=strata, obs=obs, covariate_strategy=covariate_strategy)
        logger.warning(f"Dropped covariate: {dropped} to meet sample requirements.")

    # Now we know which strata can be used to generate samples. Let's summarize it into a design table.
    sample_factors_categorical = [group_key_internal] + strata
    sample_factors_continuous = continuous_covariates if continuous_covariates is not None else []

    obs_grouped = obs.groupby(sample_factors_categorical, observed=True, sort=False)

    if sample_factors_continuous:
        agg_func = _get_aggregation_function(continuous_aggregation)
        sample_table = obs_grouped[sample_factors_continuous].agg(agg_func).reset_index()
    else:
        sample_table = obs_grouped.first().reset_index()[sample_factors_categorical]

    # Based on that, we can now decide on the design formula. Here, we will use the sample table with one important
    # detail: although the replicate key does not necessarily introduce collinearity in the columns of the design matrix
    # it is exclusively used as a means to generate a sample stratification. In the majority of setups, it does not make
    # sense to include the replicate_key in the design
    # Let's also remove the group_key_internal from the design_factors here, as we will add it explicitly later with the correct reference level.
    design_factors_categorical_formula = [
        f for f in sample_factors_categorical if f != replicate_key and f != group_key_internal
    ]
    design_factors_continuous_formula = sample_factors_continuous.copy()

    while True:
        # Although the while loop should be exited at some point, this is a bit risky. Maybe let's put a safeguard in the future
        design_formula = _build_design(
            group_key_internal=group_key_internal,
            factors_categorical=design_factors_categorical_formula,
            factors_continuous=design_factors_continuous_formula,
        )
        mm = model_matrix(design_formula, data=sample_table)
        if np.linalg.matrix_rank(mm.values) == mm.shape[1]:
            logger.info(f"Design matrix with shape {mm.shape} has full rank using design formula:\n{design_formula}")
            break

        # If there are categorical factors, consume them first, as each generates n - 1 level columns
        if design_factors_categorical_formula:
            design_factors_categorical_formula, dropped = _drop_covariate(
                covariates=design_factors_categorical_formula, obs=sample_table, covariate_strategy=covariate_strategy
            )
            logger.warning(f"Dropped categorical covariate '{dropped}' to achieve full column rank.")
            continue

        # If this didn't help, consume continuous covariates. Each one only generates one column
        # And it is unlikely that they cause collinearity unless they are constant. In that case the algorithm
        # would converge if all constant and continuous covariates are removed
        if design_factors_continuous_formula:
            design_factors_continuous_formula, dropped = _drop_covariate(
                covariates=design_factors_continuous_formula,
                obs=sample_table,
                covariate_strategy="sequence_order",  # enforce sequence order for continuous
            )
            logger.warning(f"Dropped continuous covariate '{dropped}' to achieve full rank.")
            continue

    # Now we can finally aggregate the counts into pseudobulk samples
    pb_counts = _aggregate_counts(
        adata=adata, grouped_obs=obs_grouped, layer=layer, layer_aggregation=layer_aggregation
    )

    # We want to also store the original per-cell count matrix because it might be needed for generating pseudoreplicates
    adata_sub = adata[obs.index, :]

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
        n_cells=vc,
    )
