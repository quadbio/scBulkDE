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
from scbulkde.ut.ut_basic import (
    _aggregate_results,
    _generate_pseudoreplicate,
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
    resampling_fraction: float = 0.6,
    min_list_overlap: float = 1.0,
    # DE parameters
    min_samples: int = 3,
    alpha: float = 0.05,
    alpha_fallback: float | None = None,
    correction_method: str = "fdr_bh",
    engine: str = "anova",
    engine_kwargs: dict | None = None,
    # fallback parameters
    fallback_strategy: Literal["pseudoreplicates", "single_cell"] | None = None,
    # general parameters
    seed: int = 42,
) -> DEResult:
    """Perform pseudobulked differential expression analysis.

    Parameters
    ----------
    data
        AnnData object or PseudobulkResult.
    fallback_strategy
        Strategy when insufficient samples exist:
        - 'pseudoreplicates': Generate pseudoreplicates to fill up to min_samples
        - 'single_cell': Perform DE at single-cell level using all cells
        - None: Raise error if insufficient samples (default)
    alpha_fallback
        Separate significance threshold for fallback methods. If None, uses alpha.
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
    # n_existing_total = existing_samples.get("query", 0) + existing_samples.get("reference", 0)
    pb_counts_empty = pb_result.pb_counts.empty or len(pb_result.pb_counts) == 0

    # Compute how many samples are needed
    required_samples = {c: max(0, min_samples - existing_samples.get(c, 0)) for c in ["query", "reference"]}
    needs_fallback = any(v > 0 for v in required_samples.values()) or pb_counts_empty

    # Case 1: Sufficient samples exist
    if not needs_fallback:
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
        # Check if we can generate pseudoreplicates (need at least 1 sample per condition to resample from)
        can_generate_query = _can_generate_pseudoreplicates(pb_result.grouped, "query")
        can_generate_ref = _can_generate_pseudoreplicates(pb_result.grouped, "reference")

        if not can_generate_query or not can_generate_ref:
            raise ValueError(
                f"Cannot generate pseudoreplicates: need at least some cells per condition. "
                f"Query groups available: {can_generate_query}, Reference groups available: {can_generate_ref}. "
                f"Consider using fallback_strategy='single_cell' instead."
            )

        logger.info(
            f"Insufficient samples - generating pseudoreplicates. "
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
            min_list_overlap=min_list_overlap,
            rng=rng,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
        )

    raise ValueError(f"Unknown fallback_strategy: {fallback_strategy}")


def _count_existing_samples(
    grouped: pd.api.typing.DataFrameGroupBy,
) -> dict[str, int]:
    """Count existing samples per condition from grouped object."""
    counts = {"query": 0, "reference": 0}
    # grouper_names = grouped.grouper.names

    for meta, _ in grouped:
        # meta can be a tuple or a single value depending on groupby columns
        if isinstance(meta, tuple):
            for m in meta:
                if m == "query":
                    counts["query"] += 1
                elif m == "reference":
                    counts["reference"] += 1
        else:
            # Single groupby column case
            if meta == "query":
                counts["query"] += 1
            elif meta == "reference":
                counts["reference"] += 1

    return counts


def _can_generate_pseudoreplicates(
    grouped: pd.api.typing.DataFrameGroupBy,
    condition: str,
) -> bool:
    """Check if there are any groups for a given condition to sample from."""
    for meta, obs in grouped:
        if isinstance(meta, tuple):
            if condition in meta and len(obs) > 0:
                return True
        else:
            if meta == condition and len(obs) > 0:
                return True
    return False


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
    """Run DE at single-cell level using all cells."""
    adata_sub = pb_result.adata_sub
    layer = pb_result.layer
    group_key_internal = pb_result.group_key_internal

    # Get expression matrix
    if layer is None:
        X = adata_sub.X
    else:
        X = adata_sub.layers[layer]

    import time

    start = time.time()
    # Convert sparse to dense if needed (engines expect DataFrame)
    if sp.issparse(X):
        X = X.toarray()

    # Create counts DataFrame: cells x genes
    counts = pd.DataFrame(X, index=adata_sub.obs_names, columns=adata_sub.var_names)

    # Create metadata with condition column
    metadata = pd.DataFrame(
        {group_key_internal: pb_result.grouped.obj[group_key_internal].values}, index=adata_sub.obs_names
    )
    end = time.time()
    print(f"Data preparation time: {end - start:.2f} seconds")

    # Simple design: intercept + condition
    design_formula = f"C({group_key_internal}, contr.treatment(base='reference'))"
    design_matrix = model_matrix(design_formula, data=metadata)

    n_query = (metadata[group_key_internal] == "query").sum()
    n_ref = (metadata[group_key_internal] == "reference").sum()
    logger.info(f"Single-cell DE: {n_query} query cells, {n_ref} reference cells, {counts.shape[1]} genes")

    # Run DE
    start = time.time()
    results = de_engine.run(
        counts=counts,
        metadata=metadata,
        design_matrix=design_matrix,
        design_formula=design_formula,
        alpha=alpha,
        correction_method=correction_method,
        **engine_kwargs,
    )
    end = time.time()
    print(f"DE engine time: {end - start:.2f} seconds")

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
    min_list_overlap: float,
    rng: np.random.Generator,
    engine_name: str,
    engine_kwargs: dict,
) -> DEResult:
    """Run DE analysis with pseudoreplicate generation."""
    repetition_results = {}

    # Get columns that should be in sample_table for pseudoreplicates
    sample_table_cols = list(pb_result.sample_table.columns)
    groupby_cols = list(pb_result.grouped.grouper.names)

    for it in range(n_repetitions):
        pr_counts_collected = []
        pr_meta_collected = []

        # For each condition, generate the required number of pseudoreplicates
        for condition, n_needed in required_samples.items():
            for _ in range(n_needed):
                pr_counts, pr_meta = _generate_pseudoreplicate(
                    adata=pb_result.adata_sub,
                    condition=condition,
                    grouped=pb_result.grouped,
                    layer=pb_result.layer,
                    layer_aggregation=pb_result.layer_aggregation,
                    continuous_covariates=pb_result.continuous_covariates,
                    continuous_aggregation=pb_result.continuous_aggregation,
                    resampling_fraction=resampling_fraction,
                    rng=rng,
                )

                # Ensure pr_meta has all columns that sample_table has
                # Fill missing columns with appropriate values
                for col in sample_table_cols:
                    if col not in pr_meta.columns:
                        if col in groupby_cols:
                            # Already should be there from groupby
                            pass
                        else:
                            # Set to NaN or a default - continuous covariates should be handled
                            pr_meta[col] = np.nan

                # Reorder columns to match sample_table
                pr_meta = pr_meta.reindex(columns=sample_table_cols)

                pr_counts_collected.append(pr_counts)
                pr_meta_collected.append(pr_meta)

        # Combine original samples with pseudoreplicates
        if len(pb_result.pb_counts) > 0:
            counts = pd.concat([pb_result.pb_counts] + pr_counts_collected, axis=0, ignore_index=True)
            sample_table = pd.concat([pb_result.sample_table] + pr_meta_collected, axis=0, ignore_index=True)
        else:
            # No original samples, only pseudoreplicates
            counts = pd.concat(pr_counts_collected, axis=0, ignore_index=True)
            sample_table = pd.concat(pr_meta_collected, axis=0, ignore_index=True)

        # String indices for compatibility
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

    # Aggregate results
    aggregated_results, n_genes_tested, n_genes_significant = _aggregate_results(
        results=repetition_results,
        min_list_overlap=min_list_overlap,
        alpha=alpha,
    )

    logger.info(
        f"DE complete with pseudoreplicates: {n_genes_tested} genes tested, "
        f"{n_genes_significant} significant in >= {min_list_overlap * 100:.0f}% of {n_repetitions} repetitions"
    )

    return DEResult(
        results=aggregated_results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=pb_result.design_formula,
        engine=engine_name,
        used_pseudoreplicates=True,
        used_single_cell=False,
        n_repetitions=n_repetitions,
        repetition_results=repetition_results,
    )
