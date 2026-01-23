from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from formulaic import model_matrix

from scbulkde.ut._containers import PseudobulkResult
from scbulkde.ut._logging import logger

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
    agg_func = _get_aggregation_function(continuous_aggregation)

    obs_grouped = obs.groupby(sample_factors_categorical, observed=True)

    if sample_factors_continuous:
        sample_table = obs_grouped[sample_factors_continuous].agg(agg_func).reset_index()
    else:
        sample_table = obs[sample_factors_categorical].drop_duplicates().reset_index(drop=True)

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
            logger.info(
                f"Design matrix with shape {mm.shape} has full column rank using design formula:\n{design_formula}"
            )
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
            logger.warning(f"Dropped continuous covariate '{dropped}' to achieve full column rank.")
            continue

    # Now we can finally aggregate the counts into pseudobulk samples
    df = _aggregate_counts(adata=adata, grouped_obs=obs_grouped, layer=layer, layer_aggregation=layer_aggregation)

    return PseudobulkResult(
        counts=df,
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
        continuous_covariates=continuous_covariates,
        categorical_covariates=categorical_covariates,
        continuous_aggregation=continuous_aggregation,
        min_cells=min_cells,
        min_fraction=min_fraction,
        min_coverage=min_coverage,
        qualify_strategy=qualify_strategy,
        n_cells=vc.to_dict(),
    )


def _prepare_internal_groups(
    adata: ad.AnnData,
    group_key: str,
    group_key_internal: str,
    query: str | Sequence[str],
    reference: str | Sequence[str],
):
    # View
    obs = adata.obs

    # Ensure correct datatypes. Move later to validation
    if not isinstance(obs[group_key].dtype, pd.CategoricalDtype):
        obs[group_key] = obs[group_key].astype("category")

    # Normalize inputs to lists
    if isinstance(query, str):
        query = [query]
    else:
        query = list(query)

    if reference == "rest":
        reference = [cat for cat in obs[group_key].cat.categories if cat not in query]
    elif isinstance(reference, str):
        reference = [reference]
    else:
        reference = list(reference)

    # Subset obs to relevant cells, make a copy
    mask_query = obs[group_key].isin(query)
    mask_reference = obs[group_key].isin(reference)

    if not mask_query.any():
        raise ValueError(f"No cells found for query groups: {query}")
    if not mask_reference.any():
        raise ValueError(f"No cells found for reference groups: {reference}")

    mask = mask_query | mask_reference
    obs = obs[mask].copy()

    obs[group_key_internal] = np.where(obs[group_key].isin(query), "query", "reference")

    return obs


def _can_generate_samples(
    obs: pd.DataFrame,
    stratify_by: Sequence[str],
    min_cells: int,
    min_fraction: float,
    min_coverage: float,
    qualify_strategy: str,
    group_key_internal: str,
) -> bool:
    """Given stratify_by keys, determine if samples can be generated for 'query' and 'reference'"""
    if not stratify_by:
        return False

    for label in ("query", "reference"):
        # Subset to the relevant group. Has to be at least one cell due to earlier checks
        obs_sub = obs[obs[group_key_internal] == label]
        total_cells = len(obs_sub)

        grouped = obs_sub.groupby(list(stratify_by)).size()
        counts = grouped.values

        qualifying = []
        for n_cells in counts:
            # Consider min_cells/min_fraction as fulfilled if None
            min_cells_ok = True if min_cells is None else n_cells >= min_cells
            min_fraction_ok = True if min_fraction is None else (n_cells / total_cells) >= min_fraction
            if qualify_strategy == "and":
                qualifies = min_cells_ok and min_fraction_ok
            elif qualify_strategy == "or":
                qualifies = min_cells_ok or min_fraction_ok
            else:
                raise ValueError("qualify_strategy must be 'and' or 'or'")
            if qualifies:
                qualifying.append(n_cells)

        # If there are no qualifying cells, return false
        if not qualifying:
            return False

        # If there are some, check the total coverage
        coverage = sum(qualifying) / total_cells
        if min_coverage is not None and coverage < min_coverage:
            return False

    # If total coverage is met for both query and reference, return True
    return True


def _build_design(group_key_internal: str, factors_categorical: list, factors_continuous: list):
    terms = [f"C({group_key_internal}, contr.treatment(base='reference'))"]
    if factors_categorical:
        terms += [f"C({f})" for f in factors_categorical]
    if factors_continuous:
        terms += factors_continuous
    return " + ".join(terms)


def _drop_covariate(covariates: list, obs: pd.DataFrame, covariate_strategy: str) -> tuple:
    if covariate_strategy == "sequence_order":
        dropped = covariates.pop()
        return covariates, dropped
    elif covariate_strategy == "most_levels":
        levels = [obs[cov].nunique() for cov in covariates]
        idx = int(np.argmax(levels))
        dropped = covariates.pop(idx)
        return covariates, dropped
    else:
        raise ValueError(f"Unknown covariate_strategy: {covariate_strategy}")


def _get_aggregation_function(
    agg: str | Callable | list,
    allow: set[str] | None = None,
) -> Callable | None:
    """Return a numpy aggregation function, a user-supplied callable, or None."""
    if allow is None:
        allow = {"mean", "sum", "median"}

    if isinstance(agg, (list, tuple)) and len(agg) == 0:
        return None

    if callable(agg):
        return agg

    if not isinstance(agg, str):
        raise ValueError(f"Aggregation must be a string, callable, or empty list, got {type(agg)}.")

    agg = agg.lower()
    if agg not in allow:
        raise ValueError(f"Aggregation '{agg}' not recognized. Allowed: {allow}")

    if agg == "mean":
        return np.mean
    if agg == "sum":
        return np.sum
    if agg == "median":
        return np.median

    raise ValueError(f"Aggregation '{agg}' not supported.")


def _aggregate_counts(adata, grouped_obs, layer=None, layer_aggregation="sum"):
    """
    Aggregates counts for each group, resulting in a pseudobulk matrix (n_samples x n_genes).

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    grouped_obs : pd.core.groupby.generic.DataFrameGroupBy
        Grouped adata.obs (e.g., adata.obs.groupby(["batch", "cell_type"])).
    layer : str or None
        .layers key to use for counts, or None for .X.
    layer_aggregation : {'sum', 'mean'}
        How to aggregate the counts.

    Returns
    -------
    pd.DataFrame
        DataFrame of shape (n_samples, n_genes), index are pseudobulked sample names.
    """
    if layer is None:
        X = adata.X
    else:
        X = adata.layers[layer]
    var_names = adata.var_names

    # Try to keep everything in sparse unless it's dense to begin with
    is_sparse = sp.issparse(X)
    results = []

    for _, group in grouped_obs:
        idx = adata.obs.index.get_indexer(group.index)

        # Get the matrix for this group (sparse indexing)
        if is_sparse:
            X_group = X[idx]
        else:
            X_group = X[idx, :]
        # Aggregate
        if layer_aggregation == "sum":
            agg = X_group.sum(axis=0)
        elif layer_aggregation == "mean":
            agg = X_group.mean(axis=0)
        else:
            raise ValueError("Invalid layer_aggregation; must be 'sum' or 'mean'.")
        # keep as 1D sparse matrix if possible
        if is_sparse:
            agg = sp.csr_matrix(agg)
        else:
            agg = np.asarray(agg)
        results.append(agg)

    # Stack vertically
    if is_sparse:
        stacked = sp.vstack(results)
        df = pd.DataFrame.sparse.from_spmatrix(stacked, columns=var_names)
    else:
        stacked = np.vstack(results)
        df = pd.DataFrame(stacked, columns=var_names)
    return df
