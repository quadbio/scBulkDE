from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from formulaic import model_matrix

from scbulkde.ut._logging import logger
from scbulkde.ut._performance import performance

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence
    from typing import Literal

    import anndata as ad
    from numpy.typing import NDArray


def _prepare_internal_groups(
    adata: ad.AnnData,
    group_key: str,
    group_key_internal: str,
    query: str | Sequence[str],
    reference: str | Sequence[str],
) -> pd.DataFrame:
    """Prepare adata.obs with internal 'query' and 'reference' labels."""
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

    # Logger warning if categories appear in both query and reference
    overlap = set(query).intersection(set(reference))
    if overlap:
        logger.warning(
            f"Detected overlap between query and reference: {overlap}. Cells in these groups will be assigned to 'query'."
        )

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


@performance(logger=logger)
def _validate_strata(
    obs: pd.DataFrame,
    strata: list[str] | None,
    min_cells: int | None,
    min_fraction: float | None,
    min_coverage: float | None,
    qualify_strategy: str,
    covariate_strategy: str,
    group_key_internal: str,
    resolve_conflicts: bool,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """Validate strata and return filtered obs with only qualifying cells.

    Returns
    -------
    valid_strata : list[str]
        List of strata that can generate valid samples
    filtered_obs : pd.DataFrame
        obs filtered to only cells in qualifying samples (empty if no valid strata)
    sample_stats : pd.DataFrame
        Sample-level statistics (empty if no valid strata)
    """
    # Convert None to empty list
    if strata is None:
        strata = []
    else:
        strata = list(strata)  # Make a copy

    # No strata provided
    if not strata:
        logger.warning(
            "No replicate_key or categorical_covariates provided. "
            "Cannot create independent samples - returning empty pseudobulk counts."
        )
        return [], pd.DataFrame(), pd.DataFrame()

    while True:
        # Check if samples can be generated and get filtered obs
        can_generate, filtered_obs, sample_stats = _generate_samples(
            obs=obs,
            stratify_by=strata,
            min_cells=min_cells,
            min_fraction=min_fraction,
            min_coverage=min_coverage,
            qualify_strategy=qualify_strategy,
            group_key_internal=group_key_internal,
        )

        if can_generate:
            logger.info(f"Valid strata found: {strata}. Kept {len(filtered_obs)} cells in qualifying samples.")
            return strata, filtered_obs, sample_stats

        # Drop a covariate
        strata, dropped = _drop_covariate(covariates=strata, obs=obs, covariate_strategy=covariate_strategy)
        logger.warning(f"Dropped covariate: {dropped} to meet sample requirements.")

        # If no covariates left
        if not strata:
            if resolve_conflicts:
                logger.warning("Cannot generate samples with any stratification. Returning empty pseudobulk counts.")
                return [], pd.DataFrame(), pd.DataFrame()
            else:
                raise ValueError("Cannot generate samples with any stratification and no covariates left to drop.")


@performance(logger=logger)
def _generate_samples(
    obs: pd.DataFrame,
    stratify_by: Sequence[str],
    min_cells: int | None,
    min_fraction: float | None,
    min_coverage: float | None,
    qualify_strategy: str,
    group_key_internal: str,
) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
    """Check if samples can be generated and return filtered obs with only qualifying cells.

    Parameters
    ----------
    obs : pd.DataFrame
        Observation metadata
    stratify_by : Sequence[str]
        Columns to stratify by (e.g., ['batch', 'donor'])
    min_cells : int | None
        Minimum cells per sample
    min_fraction : float | None
        Minimum fraction of cells per sample (relative to condition)
    min_coverage : float | None
        Minimum fraction of cells that must be in qualifying samples
    qualify_strategy : {'and', 'or'}
        How to combine min_cells and min_fraction requirements
    group_key_internal : str
        Column name for condition (query/reference)

    Returns
    -------
    can_generate : bool
        Whether valid samples can be generated
    filtered_obs : pd.DataFrame
        obs DataFrame filtered to only cells in qualifying samples
        (empty if can't generate)
    sample_stats : pd.DataFrame
        DataFrame with sample-level statistics including:
        - stratify_by columns (as index or columns)
        - group_key_internal: condition label
        - n_cells: number of cells in this sample
        - n_cells_condition: total cells in the condition
        - fraction: fraction of cells (n_cells / n_cells_condition)
        - coverage: coverage of the condition (for all qualifying samples)
    """
    if not stratify_by:
        return False, pd.DataFrame(), pd.DataFrame()

    # Pre-create masks for both conditions
    query_mask = obs[group_key_internal] == "query"
    reference_mask = obs[group_key_internal] == "reference"

    qualifying_indices = []
    sample_stats_list = []

    for label, condition_mask in [("query", query_mask), ("reference", reference_mask)]:
        total_cells = condition_mask.sum()

        if total_cells == 0:
            return False, pd.DataFrame(), pd.DataFrame()

        # Get cell counts per stratum - use observed=True to avoid empty groups
        # Work directly with the full obs DataFrame using the mask
        grouped = obs.loc[condition_mask].groupby(list(stratify_by), observed=True, sort=False)
        group_sizes = grouped.size()

        # Vectorized qualification check
        if min_cells is not None and min_fraction is not None:
            cell_check = group_sizes >= min_cells
            fraction_check = (group_sizes / total_cells) >= min_fraction

            if qualify_strategy == "and":
                qualifying_mask = cell_check & fraction_check
            elif qualify_strategy == "or":
                qualifying_mask = cell_check | fraction_check
            else:
                raise ValueError("qualify_strategy must be 'and' or 'or'")
        elif min_cells is not None:
            qualifying_mask = group_sizes >= min_cells
        elif min_fraction is not None:
            qualifying_mask = (group_sizes / total_cells) >= min_fraction
        else:
            # No requirements - shouldn't happen, treat as none qualifying
            return False, pd.DataFrame(), pd.DataFrame()

        # Filter to qualifying groups
        qualifying_groups = group_sizes[qualifying_mask]

        # Check if any groups qualify
        if len(qualifying_groups) == 0:
            return False, pd.DataFrame(), pd.DataFrame()

        # Check coverage requirement
        qualifying_cell_count = qualifying_groups.sum()
        coverage = qualifying_cell_count / total_cells
        if min_coverage is not None and coverage < min_coverage:
            return False, pd.DataFrame(), pd.DataFrame()

        # Build sample statistics for qualifying groups
        stats_df = qualifying_groups.reset_index(name="n_cells")
        stats_df[group_key_internal] = label
        stats_df["n_cells_condition"] = total_cells
        stats_df["fraction"] = stats_df["n_cells"] / total_cells
        stats_df["coverage"] = coverage
        sample_stats_list.append(stats_df)

        # Build mask for cells in qualifying groups
        if len(stratify_by) == 1:
            # Simple case: single stratification column
            col = stratify_by[0]
            qualifying_values = qualifying_groups.index.tolist()
            group_mask = obs.loc[condition_mask, col].isin(qualifying_values)
            # Get the actual indices where both condition and group masks are True
            qualifying_idx = obs.index[condition_mask][group_mask]
        else:
            # Multiple stratification columns - use MultiIndex for efficiency
            # Create MultiIndex from the condition subset
            obs_subset = obs.loc[condition_mask, list(stratify_by)]
            multi_idx = pd.MultiIndex.from_frame(obs_subset)

            # Check membership efficiently
            group_mask = multi_idx.isin(qualifying_groups.index)

            # Get the actual indices
            qualifying_idx = obs.index[condition_mask][group_mask]

        qualifying_indices.append(qualifying_idx)

    # Combine indices from both conditions efficiently
    if qualifying_indices:
        # Use np.concatenate on the underlying arrays for speed
        all_qualifying_indices = np.concatenate([idx.values for idx in qualifying_indices])
        # Create filtered dataframe using iloc with sorted unique indices for efficiency
        filtered_obs = obs.loc[all_qualifying_indices]

        # Combine sample statistics from both conditions
        sample_stats = pd.concat(sample_stats_list, ignore_index=True)

        return True, filtered_obs, sample_stats
    else:
        return False, pd.DataFrame(), pd.DataFrame()


def _build_design_formula(group_key_internal: str, factors_categorical: list, factors_continuous: list) -> str:
    """Build a design formula string."""
    terms = [f"C({group_key_internal}, contr.treatment(base='reference'))"]
    if factors_categorical:
        terms += [f"C({f})" for f in factors_categorical]
    if factors_continuous:
        terms += factors_continuous
    return " + ".join(terms)


def _drop_covariate(covariates: list, obs: pd.DataFrame, covariate_strategy: str) -> tuple[list, str]:
    """Drop a covariate from the list based on the specified strategy."""
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


def _get_aggregation_function(agg: str | Callable | list, allow: set[str] | None = None) -> Callable | None:
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


def _build_full_rank_design(
    sample_table: pd.DataFrame,
    group_key_internal: str,
    design_factors_categorical: list[str],
    design_factors_continuous: list[str],
    covariate_strategy: str,
) -> tuple[str, pd.DataFrame, list[str], list[str]]:
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
            return design_formula, mm, design_factors_categorical, design_factors_continuous

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

    return design_formula, mm, design_factors_categorical, design_factors_continuous


def _compute_required_samples(
    grouped: pd.api.typing.DataFrameGroupBy,
    min_samples: int,
) -> dict[str, int]:
    """Compute how many pseudoreplicates are needed per condition."""
    counts = Counter()
    for meta, _ in grouped:
        for m in meta:
            if m in {"query", "reference"}:
                counts[m] += 1

    return {c: max(0, min_samples - counts.get(c, 0)) for c in ["query", "reference"]}


@performance(logger=logger)
def _aggregate_counts(
    adata: ad.AnnData, grouped_obs: pd.api.typing.DataFrameGroupBy, layer: str | None, layer_aggregation: str
) -> pd.DataFrame:
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


def _get_X_and_var_names(
    adata: ad.AnnData,
    *,
    use_raw: bool,
    layer: str | None,
    mask_var: np.ndarray | None,
):
    """
    Get data matrix X and variable names from AnnData

    This function reimplements logic from:
    https://github.com/scverse/scanpy/blob/cf8b46dea735c35a629abfaa2e1bab9047281e34/src/scanpy/tools/_rank_genes_groups.py#L159-L181
    """
    adata_comp = adata

    if layer is not None:
        if use_raw:
            raise ValueError("Cannot specify `layer` and `use_raw=True`.")
        X = adata.layers[layer]
    elif use_raw and adata.raw is not None:
        adata_comp = adata.raw
        X = adata.raw.X
    else:
        X = adata.X

    if mask_var is not None:
        X = X[:, mask_var]
        var_names = adata_comp.var_names[mask_var]
    else:
        var_names = adata_comp.var_names

    return X, var_names


def _select_top_n(scores: NDArray, n_top: int):
    """Select indices of top n_top scores."""
    n_from = scores.shape[0]
    reference_indices = np.arange(n_from, dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]

    return global_indices


def _select_groups(
    adata: ad.AnnData,
    groups_order_subset: Iterable[str] | Literal["all"] = "all",
    key: str = "groups",
) -> tuple[list[str], NDArray[np.bool_]]:
    """
    Get subset of groups in adata.obs[key].

    This is an exact copy of the select_groups function from scanpy:
    https://github.com/scverse/scanpy/blob/cf8b46dea735c35a629abfaa2e1bab9047281e34/src/scanpy/_utils/__init__.py#L839-L886
    In line 875 the logger was replaced with a simple print statement.
    """
    groups_order = adata.obs[key].cat.categories
    if f"{key}_masks" in adata.uns:
        groups_masks_obs = adata.uns[f"{key}_masks"]
    else:
        groups_masks_obs = np.zeros((len(adata.obs[key].cat.categories), adata.obs[key].values.size), dtype=bool)
        for iname, name in enumerate(adata.obs[key].cat.categories):
            # if the name is not found, fallback to index retrieval
            if name in adata.obs[key].values:
                mask_obs = name == adata.obs[key].values
            else:
                mask_obs = str(iname) == adata.obs[key].values
            groups_masks_obs[iname] = mask_obs
    groups_ids = list(range(len(groups_order)))
    if groups_order_subset != "all":
        groups_ids = []
        for name in groups_order_subset:
            groups_ids.append(np.where(adata.obs[key].cat.categories.values == name)[0][0])
        if len(groups_ids) == 0:
            # fallback to index retrieval
            groups_ids = np.where(
                np.isin(
                    np.arange(len(adata.obs[key].cat.categories)).astype(str),
                    np.array(groups_order_subset),
                )
            )[0]
        if len(groups_ids) == 0:
            print(
                f"{np.array(groups_order_subset)} invalid! specify valid "
                f"groups_order (or indices) from {adata.obs[key].cat.categories}",
            )
            from sys import exit

            exit(0)
        groups_masks_obs = groups_masks_obs[groups_ids]
        groups_order_subset = adata.obs[key].cat.categories[groups_ids].values
    else:
        groups_order_subset = groups_order.values
    return groups_order_subset, groups_masks_obs


def _fraction_expressing(X, mask_obs):
    """Compute fraction of cells expressing each gene in X for given mask."""
    if sp.issparse(X):
        sub = X[mask_obs]
        return sub.getnnz(axis=0) / max(1, sub.shape[0])
    else:
        sub = X[mask_obs]
        return np.count_nonzero(sub, axis=0) / max(1, sub.shape[0])
