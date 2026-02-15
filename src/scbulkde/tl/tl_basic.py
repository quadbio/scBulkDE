"""Differential expression testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
from formulaic import model_matrix

from scbulkde.engines import get_engine_instance
from scbulkde.pp import pseudobulk
from scbulkde.ut import DEResult, PseudobulkResult, logger
from scbulkde.ut._performance import performance
from scbulkde.ut.ut_basic import (
    _aggregate_counts,
    _get_aggregation_function,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Literal

    import anndata as ad


def de(
    data: ad.AnnData | PseudobulkResult,
    *,
    # pseudobulking parameters
    group_key: str | None = None,
    query: str | Sequence[str] | None = None,
    reference: str | Sequence[str] = "rest",
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
    # pseudoreplicate parameters
    n_repetitions: int = 3,
    resampling_fraction: float = 0.33,
    # DE parameters
    min_samples: int = 3,
    alpha: float = 0.05,
    alpha_fallback: float | None = 0.05,
    correction_method: str = "fdr_bh",
    engine: str = "anova",
    engine_kwargs: dict | None = None,
    # fallback parameters
    fallback_strategy: Literal["pseudoreplicates", "single_cell"] | None = "pseudoreplicates",
    # general parameters
    seed: int = 42,
) -> DEResult:
    """
    Perform differential expression analysis on pseudobulked single-cell data.

    This function integrates pseudobulking and differential expression testing
    with fallback strategies when insufficient biological replicates exist.

    Parameters
    ----------
    data : ad.AnnData or PseudobulkResult
        Input data. Either an AnnData object (will be pseudobulked automatically)
        or a pre-computed PseudobulkResult from `pp.pseudobulk()`.
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
    n_repetitions : int, default=3
        Number of pseudoreplicate iterations to generate.
    resampling_fraction : float, default=0.6
        Fraction of cells to sample (with replacement) from a valid pseudobulk
        to generate a pseudoreplicate.
    min_samples : int, default=3
        Minimum number of pseudobulk samples required per condition for direct
        DE testing. If fewer exist, falls back according to `fallback_strategy`.
    alpha : float, default=0.05
        Significance threshold for direct pseudobulk DE testing.
    alpha_fallback : float, optional, default=0.05
        Separate significance threshold for fallback methods (pseudoreplicates
        or single-cell). If None, uses `alpha`.
    correction_method : str, default="fdr_bh"
        Multiple testing correction method. Options include:
        - 'fdr_bh': Benjamini-Hochberg FDR (recommended)
        - 'bonferroni': Bonferroni correction
        - Others supported by `statsmodels.stats.multitest.multipletests`
    engine : str, default="anova"
        Statistical engine for DE testing. Available engines are 'pydeseq2' and 'anova'
    engine_kwargs : dict, optional
        Additional keyword arguments passed to the DE engine.
    fallback_strategy : {"pseudoreplicates", "single_cell", None}, default="pseudoreplicates"
        Strategy when fewer than `min_samples` exist per condition:
        - 'pseudoreplicates': Generate synthetic replicates by resampling cells
          and run multiple DE tests, aggregating results
        - 'single_cell': Perform DE at single-cell resolution using all cells
        - None: Raise an error if insufficient samples
    seed : int, default=42
        Random seed for reproducibility of pseudoreplicate generation.

    Returns
    -------
    DEResult
        Container object with differential expression results and metadata:

        - **results** : pd.DataFrame
            Main results table with columns:

            * gene: Gene identifier
            * baseMean: Mean expression across samples
            * log2FoldChange: Log2 fold change (query vs reference)
            * lfcSE: Standard error of log2 fold change
            * stat: Test statistic
            * stat_sign: Signed statistic for ranking
            * pvalue: Raw p-value
            * padj: Adjusted p-value (FDR)

        - **query** : str or list
            Query condition(s) tested
        - **reference** : str or list
            Reference condition(s) tested
        - **design** : str
            Design formula used for testing
        - **engine** : str
            Statistical engine used
        - **used_pseudoreplicates** : bool
            True if pseudoreplicates were generated
        - **used_single_cell** : bool
            True if single-cell level testing was performed
        - **n_repetitions** : int
            Number of repetitions (1 for direct testing, >1 for pseudoreplicates)
        - **repetition_results** : dict, optional
            Individual results from each repetition (only for pseudoreplicates)

    Raises
    ------
    ValueError
        - If `fallback_strategy=None` and insufficient samples exist
        - If `data` is AnnData but `group_key` or `query` is not provided
        - If specified groups/keys don't exist in the data

    Warnings
    --------
    - Single-cell fallback testing treats each cell as an independent sample,
      which inflates test statistics
    - Pseudoreplicate fallback is more conservative but if a large fraction of
      cells are sampled, the independence assumption may still be violated.
    - Results from fallback strategies should be interpreted with caution and
      ideally validated with independent biological replicates

    See Also
    --------
    pp.pseudobulk : Perform pseudobulking without DE testing
    rank_genes_groups : Perform multi-group DE analysis
    DEResult : Container class for DE results

    Examples
    --------
    n.a.

    References
    ----------
    .. [1] Squair, J.W., et al. "Confronting false discoveries in single-cell differential expression." Nature Communications 12, 5692 (2021).
    .. [2] Love, M.I., Huber, W. & Anders, S. "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2." Genome Biology 15, 550 (2014).
    """
    if engine_kwargs is None:
        engine_kwargs = {}

    # Determine effective alpha for fallback
    effective_alpha_fallback = alpha_fallback if alpha_fallback is not None else alpha

    if isinstance(data, PseudobulkResult):
        pb_result = data
        logger.info("Using provided PseudobulkResult")
    else:
        logger.info("Running pseudobulking...")
        pb_result = pseudobulk(
            adata=data,
            group_key=group_key,
            query=query,
            reference=reference,
            replicate_key=replicate_key,
            min_cells=min_cells,
            min_fraction=min_fraction,
            min_coverage=min_coverage,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
            continuous_aggregation=continuous_aggregation,
            layer=layer,
            layer_aggregation=layer_aggregation,
            qualify_strategy=qualify_strategy,
            covariate_strategy=covariate_strategy,
            resolve_conflicts=resolve_conflicts,
        )

    rng = np.random.default_rng(seed)
    de_engine = get_engine_instance(engine)

    # Count existing samples per condition
    existing_samples = _count_existing_samples(pb_result.grouped)

    # Compute how many samples are needed
    required_samples = {c: max(0, min_samples - existing_samples.get(c, 0)) for c in ["query", "reference"]}

    # Check if I need to fall back to pseudoreplicates or single-cell DE.
    # This is the case when either condition has fewer than min_samples samples
    needs_fallback = any(v > 0 for v in required_samples.values())

    # Case 1: Sufficient samples exist
    if not needs_fallback:
        logger.debug("Using direct DE")
        logger.info(f"Running DE with {engine} engine (sufficient samples)...")
        return _run_de_direct(
            pb_result=pb_result,
            alpha=alpha,
            correction_method=correction_method,
            de_engine=de_engine,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
            used_fallback=False,
        )

    # Case 2: Insufficient samples - need fallback
    if fallback_strategy is None:
        raise ValueError(
            f"Insufficient samples for DE analysis. "
            f"Existing: query={existing_samples.get('query', 0)}, "
            f"reference={existing_samples.get('reference', 0)}, "
            f"min_samples={min_samples}. "
            f"Set fallback_strategy='pseudoreplicates' or 'single_cell' to proceed."
        )

    # Case 2a: Single-cell fallback
    if fallback_strategy == "single_cell":
        logger.debug("Using single-cell DE.")
        logger.info(f"Running single-cell DE with {engine} engine...")
        return _run_de_single_cell(
            pb_result=pb_result,
            alpha=effective_alpha_fallback,
            correction_method=correction_method,
            de_engine=de_engine,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
        )

    # Case 2b: Pseudoreplicates fallback
    if fallback_strategy == "pseudoreplicates":
        logger.debug("Using pseudoreplicate DE.")
        logger.info(
            f"Insufficient samples - generating pseudoreplicates: "
            f"Existing: {existing_samples}, Required additional: {required_samples}"
        )
        return _run_de_pseudoreplicates(
            pb_result=pb_result,
            alpha=effective_alpha_fallback,
            correction_method=correction_method,
            de_engine=de_engine,
            required_samples=required_samples,
            n_repetitions=n_repetitions,
            resampling_fraction=resampling_fraction,
            rng=rng,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
        )

    raise ValueError(f"Unknown fallback_strategy: {fallback_strategy}")


@performance(logger=logger)
def _count_existing_samples(
    grouped: pd.api.typing.DataFrameGroupBy,
) -> dict[str, int]:
    """Count existing pseudobulk samples per condition.

    For a valid MultiIndex groupby (with strata), counts unique strata combinations
    for each condition. For a single-column groupby (collapsed case), returns 0 for
    both conditions since there are no independent samples.
    """
    idx = grouped.grouper.result_index

    # The multiindex stores the group keys as tuples, the .to_frame(index=False)
    # converts to a DataFrame with one column per group key, without consuming a column as the index,
    # .values.ravel() then flattens this to a 1D array of all group key values across all levels.
    if isinstance(idx, pd.MultiIndex):
        values = idx.to_frame(index=False).values.ravel()
    else:
        # Critical: If the only group key is the condition, no valid samples exist and the index
        # is a SingleIndex
        return {
            "query": 0,
            "reference": 0,
        }

    # Simply count how many query and reference groups there are
    return pd.Series(values).value_counts().reindex(["query", "reference"], fill_value=0).to_dict()


def _run_de_direct(
    pb_result: PseudobulkResult,
    alpha: float,
    correction_method: str,
    de_engine,
    engine_name: str,
    engine_kwargs: dict,
    used_fallback: bool = False,
) -> DEResult:
    """Run DE directly on pseudobulk samples."""
    results = de_engine.run(
        counts=pb_result.pb_counts,
        metadata=pb_result.sample_table,
        design_matrix=pb_result.design_matrix,
        design_formula=pb_result.design_formula,
        alpha=alpha,
        correction_method=correction_method,
        **engine_kwargs,
    )

    n_sig = (results["padj"] < alpha).sum()
    logger.info(f"DE complete: {len(results)} genes tested, {n_sig} significant (padj < {alpha})")

    return DEResult(
        results=results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=pb_result.design_formula,
        engine=engine_name,
        used_pseudoreplicates=False,
        used_single_cell=False,
        n_repetitions=1,
    )


def _run_de_single_cell(
    pb_result: PseudobulkResult,
    alpha: float,
    correction_method: str,
    de_engine,
    engine_name: str,
    engine_kwargs: dict,
) -> DEResult:
    """Run DE at single-cell level using all cells - optimized for large datasets.

    This function avoids creating large DataFrames by passing numpy arrays
    directly to the engine.
    """
    adata_sub = pb_result.adata_sub
    layer = pb_result.layer
    group_key_internal = pb_result.group_key_internal

    # Get expression matrix - keep as numpy array
    if layer is None:
        X = adata_sub.X
    else:
        X = adata_sub.layers[layer]

    # For engines that need dense arrays, convert efficiently
    if sp.issparse(X):
        # Use tocsr() first if not already for faster row operations
        if not sp.isspmatrix_csr(X):
            X = X.tocsr()
        X = X.toarray()
    else:
        # Ensure we have a numpy array (not a matrix or other type)
        X = np.asarray(X)

    # Get gene names as numpy array
    gene_names = adata_sub.var_names.to_numpy()

    # Create minimal metadata DataFrame
    condition_values = pb_result.grouped.obj[group_key_internal].values
    metadata = pd.DataFrame({group_key_internal: condition_values}, index=adata_sub.obs_names)

    # Simple design
    design_formula = f"C({group_key_internal}, contr.treatment(base='reference'))"
    design_matrix = model_matrix(design_formula, data=metadata)

    n_query = (condition_values == "query").sum()
    n_ref = (condition_values == "reference").sum()
    logger.info(f"Single-cell DE: {n_query} query cells, {n_ref} reference cells, {X.shape[1]} genes")

    # Pass numpy array directly to engine - no DataFrame construction
    results = de_engine.run(
        counts=X,
        metadata=metadata,
        design_matrix=design_matrix,
        design_formula=design_formula,
        alpha=alpha,
        correction_method=correction_method,
        gene_names=gene_names,
        **engine_kwargs,
    )

    n_sig = (results["padj"] < alpha).sum()
    logger.info(f"Single-cell DE complete: {len(results)} genes tested, {n_sig} significant (padj < {alpha})")

    return DEResult(
        results=results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=design_formula,
        engine=engine_name,
        used_pseudoreplicates=False,
        used_single_cell=True,
        n_repetitions=1,
    )


def _run_de_pseudoreplicates(
    pb_result: PseudobulkResult,
    alpha: float,
    correction_method: str,
    de_engine,
    required_samples: dict[str, int],
    n_repetitions: int,
    resampling_fraction: float,
    rng: np.random.Generator,
    engine_name: str,
    engine_kwargs: dict,
) -> DEResult:
    """Run DE analysis with pseudoreplicate generation."""
    # Get the grouped obj
    obs_grouped = pb_result.grouped.obj
    groupby_cols = list(pb_result.grouped.grouper.names)

    # Initialize the cell usage tracker and cache. The cache is used to quickly access cell indices for each condition
    # The initialization is done before starting any repetition, so the usage is tracked globally, so for each repetition
    # and for each pseudoreplicate that needs to be generated in that repetition.
    all_cell_indices = obs_grouped.index.to_numpy()
    cell_usage_tracker = dict.fromkeys(all_cell_indices, 0)
    cell_pool_cache = _build_cell_pool_cache(pb_result, rng, shuffle=False)

    repetition_results = {}

    for it in range(n_repetitions):
        # Now this here is funny. We are going to use the list to collect all cell ids that are used for pseudoreplicates
        # in this iteration but don't keep track of the actual metadata. Further down we can then just subset the obs dataframe
        # to these cells and by grouping them again, we will retrieve the metadata information because each set of cells of a
        # pseudoreplicate is generated from exactly on sample (or all cells if collapsed)
        pr_indices_collected = []

        # There is one catch though: If pseudoreplicates are generated from the same sample, they would be added together and
        # appear as one when using this approach. So we need an additional ID
        pr_id_map = {}
        pr_id_counter = 0

        # Generate required pseudoreplicates for each condition
        for condition, n_needed in required_samples.items():
            for _ in range(n_needed):
                logger.debug(
                    f"Generating pseudoreplicate for condition '{condition}' (iteration {it + 1}/{n_repetitions})"
                )
                pr_indices = _generate_pseudoreplicate(
                    condition=condition,
                    cell_pool_cache=cell_pool_cache,
                    cell_usage_tracker=cell_usage_tracker,
                    resampling_fraction=resampling_fraction,
                    rng=rng,
                )

                for idx in pr_indices:
                    pr_id_map[idx] = pr_id_counter

                pr_indices_collected.extend(pr_indices)
                pr_id_counter += 1

        # Now here comes the magic, even though we don't know which cell in pr_indices_collected belongs to what sample
        # we can get this information back easily by subsetting and re-grouping, while respecting the pseudoreplicate ID
        pr_obs = obs_grouped.loc[pr_indices_collected]
        pr_obs["__prID__"] = pr_obs.index.map(pr_id_map)

        pr_obs_grouped = pr_obs.groupby(groupby_cols + ["__prID__"], observed=True, sort=False)

        # Now we can call _aggregate counts on ALL pseudoreplicates at once! That saves enourmous amounts of
        # run time as we only need to do one pd.concat operation
        pr_counts = _aggregate_counts(
            pb_result.adata_sub, pr_obs_grouped, layer=pb_result.layer, layer_aggregation=pb_result.layer_aggregation
        )

        # Aggregate metadata
        continuous_covariates = pb_result.continuous_covariates
        if continuous_covariates:
            agg_func = _get_aggregation_function(pb_result.continuous_aggregation)
            pr_meta = pr_obs_grouped[continuous_covariates].agg(agg_func).reset_index()
        else:
            pr_meta = pr_obs_grouped.first().reset_index()[groupby_cols + ["__prID__"]]

        # Remove the pseudoreplicate ID column, we don't need it anymore
        pr_meta = pr_meta.drop(columns=["__prID__"])

        # Combine with original samples. If samples exist, append the pseudoreplicates
        if len(pb_result.pb_counts) > 0:
            counts = pd.concat([pb_result.pb_counts, pr_counts], axis=0, ignore_index=True)
            sample_table = pd.concat([pb_result.sample_table, pr_meta], axis=0, ignore_index=True)
        # If they don't, just take them as is
        else:
            counts = pr_counts
            sample_table = pr_meta

        counts.index = counts.index.astype(str)
        sample_table.index = sample_table.index.astype(str)

        # Recompute design matrix
        design_matrix = model_matrix(pb_result.design_formula, data=sample_table)

        # Run DE
        results = de_engine.run(
            counts=counts,
            metadata=sample_table,
            design_matrix=design_matrix,
            design_formula=pb_result.design_formula,
            alpha=alpha,
            correction_method=correction_method,
            **engine_kwargs,
        )
        repetition_results[it] = results

    # Aggregate results. Well so this is statistically wrong, will implemement a
    # recalculation of the p-value using the mean test statistic later
    results = pd.concat(repetition_results.values()).groupby(level=0).mean()
    n_sig = (results["padj"] < alpha).sum()

    # Inform the user about the cell usage
    usage_counts = np.array(list(cell_usage_tracker.values()))
    logger.debug(
        f"Cell reusage per repetition: mean={usage_counts.mean() / n_repetitions:.2f}, max={usage_counts.max() / n_repetitions:.2f}\n"
        f"Unused cells across all repetitions: {np.sum(usage_counts == 0)}/{len(cell_usage_tracker)}"
    )

    # Inform the user about the de results
    logger.info(f"DE complete with pseudoreplicates: {len(results)} genes tested, {n_sig} significant (padj < {alpha})")

    return DEResult(
        results=results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=pb_result.design_formula,
        engine=engine_name,
        used_pseudoreplicates=True,
        used_single_cell=False,
        n_repetitions=n_repetitions,
        repetition_results=repetition_results,
    )


def _build_cell_pool_cache(
    pb_result: PseudobulkResult,
    rng: np.random.Generator,
    shuffle: bool = False,
) -> dict[str, list[np.ndarray]]:
    """Build cell pools for each condition."""
    # Initialize the cache
    cache = {"query": [], "reference": []}

    # Get the relevant attributes from the PseudobulkResult class
    # list(grouped.grouper.names) gets the names of the groupby keys in order
    # e.g. ['psbulk_condition', 'animal_id', 'experiment']
    grouped = pb_result.grouped
    group_key_internal = pb_result.group_key_internal
    groupby_cols = list(grouped.grouper.names)

    # The grouped df allows to iterate over each sample and get the corresponding
    # strata the group is associated with, e.g. ('query', 'animal_1', 'experiment_3')
    # and the associated dataframe with the cells that belong to that sample (the group_df)
    for name, group_df in grouped:
        # Extract condition from group name
        # If there are covariates, this is a tuple
        if isinstance(name, tuple):
            # Find the index of the condition in the groupby columns and extract it from the name tuple
            # It is in all cases the first entry, if not something would be very wrong, but coding this
            # but it's good to not hardcode it
            condition_index = groupby_cols.index(group_key_internal)
            condition = name[condition_index]
        # If the condition is collapsed (no sample could be found, all cells are used)
        # this is just a string, take it as is
        else:
            condition = name

        # Get cell labels (name of a cell, not position)
        cell_indices = group_df.index.to_numpy()

        # Shuffle cells within this sample if requested
        if shuffle:
            rng.shuffle(cell_indices)

        # I store both the size of the sample as well as the actual cell indices here
        cache[condition].append((len(cell_indices), cell_indices))

    # Shuffle sample order if requested for diversity across iterations
    # Not needed for the greedy sampling strategy, but maybe in the future there could be
    # a more efficient round-robin approach that just keeps track of a global sample pointer
    # and a local cell pointer. In that case one would need shuffling of samples and cells
    # in between repetitions.
    if shuffle:
        for condition in cache:
            rng.shuffle(cache[condition])

    return cache


def _generate_pseudoreplicate(
    condition: str,
    cell_pool_cache: dict,
    cell_usage_tracker: dict,
    resampling_fraction: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate pseudoreplicate using greedy cell usage minimization.

    Strategy:
    1. Select sample with minimum total cell usage
    2. Within that sample, randomly select from least-used cells to ensure diversity

    The cell_usage_tracker persists across all repetitions and pseudoreplicates,
    ensuring maximum independence across the entire DE analysis.
    """
    # This will generate a list of arrays containing cell indices. One array for each sample
    condition_samples = cell_pool_cache[condition]

    # Compute usage of each sample. The sample size is already pre-defined. Move it over to
    # The sample_usage_scores
    sample_usage_scores = []
    for idx, (sample_size, cell_indices) in enumerate(condition_samples):
        total_usage = sum(cell_usage_tracker.get(cell_idx, 0) for cell_idx in cell_indices)
        usage_rate = round(total_usage / sample_size, 2)
        sample_usage_scores.append((usage_rate, sample_size, idx))
    logger.debug(sample_usage_scores)

    # Select least-used sample. Sort by usage_rate first (x[0]) and if there is a tie,
    # sort by size in descending order (-x[1]). The reason for that is that we don't want
    # to generate a pseudoreplicate from an already small sample
    sample_usage_scores.sort(key=lambda x: (x[0], -x[1]))
    selected_sample_idx = sample_usage_scores[0][2]

    # Compute how many cells need to be sampled using resampling_fraction
    sample_size, cell_indices = condition_samples[selected_sample_idx]
    n_sample = max(1, int(sample_size * resampling_fraction))

    # Now within a sample, we also only want to select the least used cells.
    # So, get an array of how many times a cell has been used, and get the min of that
    cell_usage_counts = np.array([cell_usage_tracker.get(idx, 0) for idx in cell_indices])
    min_usage = cell_usage_counts.min()

    # Find all cells with minimum usage
    min_usage_mask = cell_usage_counts == min_usage
    min_usage_positions = np.where(min_usage_mask)[0]

    # Randomly sample from the least used cells. Two things can happen. Either there are enough
    # cells that are least used to sample from, then just sample randomly from those. Or, there
    # are not enough least used cells, then take all of the available ones, and sample the rest from
    # the next tier of usage. It should always be possible to fill up with enough cells from the next
    # usage tier, due to the fact that we work with fractions and the size of each pseudoreplicate from
    # a sample always stays the same, so it can't be that we would need to move up another tier of usage
    if len(min_usage_positions) >= n_sample:
        # Plenty of least used cells available, sample randomly
        sampled_positions = rng.choice(min_usage_positions, size=n_sample, replace=False)
    else:
        # Not enough least-used cells so take all and fill from next tier
        sampled_positions = list(min_usage_positions)
        remaining = n_sample - len(sampled_positions)

        # Get cells with next-lowest usage
        next_tier_mask = cell_usage_counts == (min_usage + 1)
        next_tier_positions = np.where(next_tier_mask)[0]

        additional = rng.choice(next_tier_positions, size=remaining, replace=False)
        sampled_positions.extend(additional)

        sampled_positions = np.array(sampled_positions)

    sampled_cell_indices = cell_indices[sampled_positions]

    # Update usage tracker
    for cell_idx in sampled_cell_indices:
        cell_usage_tracker[cell_idx] = cell_usage_tracker.get(cell_idx, 0) + 1

    # Aggregate expression and metadata
    return sampled_cell_indices
