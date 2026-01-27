from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp

from scbulkde.ut._logging import logger
from scbulkde.ut._performance import performance

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Literal

    import anndata as ad

# =================== Helper functions for pp ================= #


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


def _build_design(group_key_internal: str, factors_categorical: list, factors_continuous: list) -> str:
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


# ================= Helper functions for tl ================= #


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


def _generate_pseudoreplicate(
    adata: ad.AnnData,
    condition: str,
    grouped: pd.api.typing.DataFrameGroupBy,
    layer: str,
    layer_aggregation: Literal["sum", "mean"],
    continuous_covariates: Sequence[str],
    continuous_aggregation: Literal["mean", "sum", "median"] | Callable,
    resampling_fraction: float,
    rng: np.random.Generator,
):
    """Generate a single pseudoreplicate for a given condition."""
    # Get a random sample from the available samples for the condition
    # rng.choice() doesn't work because of heterogeneous data types
    all_condition_samples = [(m, o) for m, o in grouped if condition in m]
    rng.shuffle(all_condition_samples)
    source_meta, source_obs = all_condition_samples[0]

    # Resample cells without replacement
    n_sample = max(1, int(len(source_obs) * resampling_fraction))
    sampled_cells = rng.choice(source_obs.index, size=n_sample, replace=False)
    sampled_obs = source_obs.loc[sampled_cells, :]

    # Get the grouping variables for aggregation
    groupby = grouped.grouper.names

    # In order to use _aggregate_counts, we need to re-group the sampled_obs
    # now, there is only one group of course
    sampled_grouped = sampled_obs.groupby(groupby, observed=True, sort=False)

    # Aggregate counts
    pr_counts = _aggregate_counts(adata, sampled_grouped, layer=layer, layer_aggregation=layer_aggregation)

    # Generate a new row for the sample table
    if continuous_covariates:
        agg_func = _get_aggregation_function(continuous_aggregation)
        pr_meta = sampled_grouped[continuous_covariates].agg(agg_func).reset_index()
    else:
        pr_meta = sampled_grouped.first().reset_index()[groupby]

    return pr_counts, pr_meta


def _aggregate_results(
    results: dict[int, pd.DataFrame],
    min_list_overlap: float,
    alpha: float,
) -> tuple[pd.DataFrame, int, int]:
    # Store how many genes were tested in each iteration
    n_genes_tested = results[0].shape[0]

    # Collect all significant genes for each iteration
    sig_gene_lists = []
    for _, res in results.items():
        sig_genes = res.index[res["padj"] < alpha].tolist()
        sig_gene_lists.append(set(sig_genes))

    # Find genes that are in at least min_list_overlap fraction of lists
    gene_counter = Counter()
    for gene_list in sig_gene_lists:
        for gene in gene_list:
            gene_counter[gene] += 1
    n_required = int(len(sig_gene_lists) * min_list_overlap)
    selected_genes = {gene for gene, count in gene_counter.items() if count >= n_required}

    # Handle the case where there are no selected genes
    if not selected_genes:
        logger.warning("No genes pass the significance threshold and overlap criteria. Aggregating over all genes.")
        selected_genes = set().union(*sig_gene_lists)

    # Use mean as an aggregation function
    aggregated_results = []
    for _, res in results.items():
        aggregated_results.append(res.loc[list(selected_genes), :])

    results_df = pd.concat(aggregated_results).groupby(level=0).mean()

    # Store how many genes were significant in at least min_list_overlap fraction of lists
    n_genes_significant = len(results_df)

    return results_df, n_genes_tested, n_genes_significant


# ==================== Helper functions for pp and tl ================= #


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


###################
# def _get_X_and_var_names(
#     adata: AnnData,
#     *,
#     use_raw: bool,
#     layer: str | None,
#     mask_var: np.ndarray | None,
# ):
#     """
#     Get data matrix X and variable names from AnnData

#     This function reimplements logic from:
#     https://github.com/scverse/scanpy/blob/cf8b46dea735c35a629abfaa2e1bab9047281e34/src/scanpy/tools/_rank_genes_groups.py#L159-L181
#     """
#     adata_comp = adata

#     if layer is not None:
#         if use_raw:
#             raise ValueError("Cannot specify `layer` and `use_raw=True`.")
#         X = adata.layers[layer]
#     elif use_raw and adata.raw is not None:
#         adata_comp = adata.raw
#         X = adata.raw.X
#     else:
#         X = adata.X

#     if mask_var is not None:
#         X = X[:, mask_var]
#         var_names = adata_comp.var_names[mask_var]
#     else:
#         var_names = adata_comp.var_names

#     return X, var_names


# def _select_top_n(scores: NDArray, n_top: int):
#     """Select indices of top n_top scores."""
#     n_from = scores.shape[0]
#     reference_indices = np.arange(n_from, dtype=int)
#     partition = np.argpartition(scores, -n_top)[-n_top:]
#     partial_indices = np.argsort(scores[partition])[::-1]
#     global_indices = reference_indices[partition][partial_indices]

#     return global_indices


# def _select_groups(
#     adata: AnnData,
#     groups_order_subset: Iterable[str] | Literal["all"] = "all",
#     key: str = "groups",
# ) -> tuple[list[str], NDArray[np.bool_]]:
#     """
#     Get subset of groups in adata.obs[key].

#     This is an exact copy of the select_groups function from scanpy:
#     https://github.com/scverse/scanpy/blob/cf8b46dea735c35a629abfaa2e1bab9047281e34/src/scanpy/_utils/__init__.py#L839-L886
#     In line 875 the logger was replaced with a simple print statement.
#     """
#     import numpy as np

#     groups_order = adata.obs[key].cat.categories
#     if f"{key}_masks" in adata.uns:
#         groups_masks_obs = adata.uns[f"{key}_masks"]
#     else:
#         groups_masks_obs = np.zeros((len(adata.obs[key].cat.categories), adata.obs[key].values.size), dtype=bool)
#         for iname, name in enumerate(adata.obs[key].cat.categories):
#             # if the name is not found, fallback to index retrieval
#             if name in adata.obs[key].values:
#                 mask_obs = name == adata.obs[key].values
#             else:
#                 mask_obs = str(iname) == adata.obs[key].values
#             groups_masks_obs[iname] = mask_obs
#     groups_ids = list(range(len(groups_order)))
#     if groups_order_subset != "all":
#         groups_ids = []
#         for name in groups_order_subset:
#             groups_ids.append(np.where(adata.obs[key].cat.categories.values == name)[0][0])
#         if len(groups_ids) == 0:
#             # fallback to index retrieval
#             groups_ids = np.where(
#                 np.isin(
#                     np.arange(len(adata.obs[key].cat.categories)).astype(str),
#                     np.array(groups_order_subset),
#                 )
#             )[0]
#         if len(groups_ids) == 0:
#             print(
#                 f"{np.array(groups_order_subset)} invalid! specify valid "
#                 f"groups_order (or indices) from {adata.obs[key].cat.categories}",
#             )
#             from sys import exit

#             exit(0)
#         groups_masks_obs = groups_masks_obs[groups_ids]
#         groups_order_subset = adata.obs[key].cat.categories[groups_ids].values
#     else:
#         groups_order_subset = groups_order.values
#     return groups_order_subset, groups_masks_obs


# def _fraction_expressing(X, mask_obs):
#     """Compute fraction of cells expressing each gene in X for given mask."""
#     from scipy import sparse

#     if sparse.issparse(X):
#         sub = X[mask_obs]
#         return sub.getnnz(axis=0) / max(1, sub.shape[0])
#     else:
#         sub = X[mask_obs]
#         return np.count_nonzero(sub, axis=0) / max(1, sub.shape[0])
