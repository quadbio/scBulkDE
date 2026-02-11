from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from formulaic import model_matrix

from scbulkde.ut._containers import PseudobulkResult
from scbulkde.ut._logging import logger
from scbulkde.ut.ut_basic import (
    _aggregate_counts,
    _build_design_formula,
    _build_full_rank_design,
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
    """
    Perform pseudobulking on single-cell data to aggregate expression across cells.

    This function aggregates single-cell expression data into pseudobulk samples by
    combining cells from specified groups (query vs. reference) across biological
    replicates, if present, and optional covariates. It creates a design matrix
    suitable for  downstream differential expression analysis while filtering
    samples based on quality control metrics.

    Parameters
    ----------
    adata : ad.AnnData
        Annotated data matrix containing single-cell expression data.
    group_key : str
        Column name in `adata.obs` that defines the cell groups for comparison
        (e.g., 'cell_type', 'condition', 'cluster').
    query : str or Sequence[str]
        Cell group(s) to be used as the query/test condition. Must be present
        in `adata.obs[group_key]`.
    reference : str or Sequence[str], default="rest"
        Cell group(s) to be used as the reference/control condition. If "rest",
        all groups not in `query` are used as reference. Must be present in
        `adata.obs[group_key]`.
    replicate_key : str, optional
        Column name in `adata.obs` defining biological replicates (e.g., 'sample_id',
        'donor', 'batch'). Required for creating multiple pseudobulk samples per
        condition, but never included in the design. If None, cells are not stratified
        by replicate.
    min_cells : int, optional, default=50
        Minimum number of cells required per pseudobulk sample. Samples with fewer
        cells are excluded from analysis.
    min_fraction : float, optional, default=0.2
        Minimum fraction of cells of the condition in that pseudobulk sample for it
        to be considered valid. Samples with a lower fraction are excluded from analysis.
    min_coverage : float, optional, default=0.75
        Minimum coverage provided by all valid samples per condition. Conditions with
        lower coverage are collapsed. Range: [0.0, 1.0].
    categorical_covariates : Sequence[str], optional
        Column names in `adata.obs` representing categorical covariates to include
        in the design (e.g., ['experiment', 'chemistry', 'batch']). These are added as
        stratification factors along with `replicate_key`.
    continuous_covariates : Sequence[str], optional
        Column names in `adata.obs` representing continuous covariates to include
        in the design (e.g., ['cellcycle', 'pct_mito']). These are aggregated
        per pseudobulk sample.
    continuous_aggregation : {"mean", "sum", "median"} or callable, default="mean"
        Method to aggregate continuous covariates across cells within each
        pseudobulk sample. Can be a string specifying a standard aggregation
        or a custom callable.
    layer : str, optional
        Layer in `adata.layers` to use for aggregation. If None, uses `adata.X`.
    layer_aggregation : {"sum", "mean"}, default="sum"
        Method to aggregate expression values across cells.
    qualify_strategy : {"and", "or"}, default="or"
        Strategy for sample qualification when multiple criteria are specified:
        - "and": Sample candidate must pass both `min_cells` AND `min_fraction` thresholds
        - "or": Samples candidate must pass either `min_cells` OR `min_fraction` threshold
    covariate_strategy : {"sequence_order", "most_levels"}, default="sequence_order"
        Strategy for ordering covariates in the design formula when conflicts arise:
        - "sequence_order": Drop covariates from back to front in the provided list
        - "most_levels": Prioritize covariates with more unique levels
    resolve_conflicts : bool, default=True
        If True, automatically resolve confounded covariates by iteratively
        removing them to ensure a full-rank design matrix. If False, raise
        an error when confounding is detected.

    Returns
    -------
    PseudobulkResult
        Container object with the following attributes:

        - **adata_sub** : ad.AnnData
            Subset of input AnnData containing only query and reference cells
        - **pb_counts** : pd.DataFrame
            Aggregated pseudobulk expression matrix (samples × genes). Empty if
            no valid strata exist (collapsed case)
        - **grouped** : pd.api.typing.DataFrameGroupBy
            Grouped observation data for internal use
        - **sample_table** : pd.DataFrame
            Metadata for each pseudobulk sample, including covariates, cell counts,
            and quality metrics
        - **design_matrix** : pd.DataFrame
            Design matrix for statistical testing, created from `design_formula`
        - **design_formula** : str
            Patsy-style formula describing the statistical model
        - **group_key** : str
            Original group key parameter
        - **group_key_internal** : str
            Internal column name for query/reference labels ('psbulk_condition')
        - **query** : str or list
            Query group(s) used
        - **reference** : str or list
            Reference group(s) used
        - **strata** : list of str
            Final stratification factors used (may be subset of requested due to
            conflict resolution). Empty list indicates collapsed pseudobulk
        - **collapsed** : bool
            True if insufficient replicates exist and data was collapsed across
            all cells per condition
        - **n_samples** : int
            Number of pseudobulk samples created

    Warnings
    --------
    - If `min_cells`, `min_fraction`, or `min_coverage` thresholds are not met,
      samples or entire conditions may be excluded or collapsed
    - Confounded covariates are automatically removed when `resolve_conflicts=True`
    - Empty `pb_counts` (collapsed case) indicates no valid independent samples
      exist and differential expression testing may require special handling

    See Also
    --------
    tl.de : Perform differential expression testing on pseudobulk data
    PseudobulkResult : Container class for pseudobulk results

    Examples
    --------
    n.a.

    Notes
    -----
    The pseudobulking approach aggregates cells from the same biological replicate
    and condition, reducing the computational burden and addressing the issue of
    pseudoreplication in single-cell data. This enables the use of standard bulk
    RNA-seq differential expression methods while accounting for biological variability.

    When `collapsed=True`, the result contains only aggregated condition-level
    information without independent replicates. In that case one needs to use the
    `tl.de` function with fallback strategies (`'pseudoreplicates'` or `'single_cell'`).

    The function automatically:

    - Filters cells to only query and reference groups
    - Validates stratification factors (replicates and covariates)
    - Removes samples not meeting quality thresholds
    - Resolves confounded covariates to ensure full-rank design
    - Creates both count matrix and metadata for downstream analysis

    References
    ----------
    .. [1] Squair, J.W., et al. "Confronting false discoveries in single-cell differential expression." Nature Communications 12, 5692 (2021).
    """
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
    strata, obs_filtered, sample_stats = _validate_strata(
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
        obs=obs,
        strata=strata,
        sample_stats=sample_stats,
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

    The sample_table is constructed with harmonized structure matching the case
    with valid strata. It contains two rows (query/reference) with:
    - n_cells: number of cells per condition (same as n_cells_condition)
    - n_cells_condition: total cells in the condition
    - fraction: 1.0 (all cells used)
    - coverage: 1.0 (all cells covered)
    - collapsed: True (indicating these are not valid independent samples)

    pb_counts is empty (0 rows) since no valid samples exist.
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
    # with harmonized metadata matching the case with valid strata
    conditions = obs[group_key_internal].unique()
    sample_table_rows = []
    for condition in conditions:
        n_cells_cond = n_cells.get(condition, 0)
        sample_table_rows.append(
            {
                group_key_internal: condition,
                "n_cells": n_cells_cond,
                "n_cells_condition": n_cells_cond,
                "fraction": 1.0,
                "coverage": 1.0,
                "collapsed": True,  # Mark as collapsed (not valid independent samples)
            }
        )
    sample_table = pd.DataFrame(sample_table_rows)

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
    sample_stats: pd.DataFrame,
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

    # Merge sample statistics into sample_table
    merge_keys = [group_key_internal] + strata
    sample_table = sample_table.merge(
        sample_stats[merge_keys + ["n_cells", "n_cells_condition", "fraction", "coverage"]],
        on=merge_keys,
        how="left",
    )

    # Add collapsed column - these are valid samples, so collapsed = False
    sample_table["collapsed"] = False

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
