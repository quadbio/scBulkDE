"""Differential expression testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scbulkde.engines import get_engine
from scbulkde.pp import pseudobulk
from scbulkde.ut import DEResult, PseudobulkResult, logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    import anndata as ad


def de(
    data: ad.AnnData | PseudobulkResult,
    *,
    group_key: str | None = None,
    query: str | None = None,
    reference: str | Sequence[str] = "rest",
    layer: str | None = None,
    replicate_key: str | None = None,
    batch_key: str | None = None,
    min_cells: int = 10,
    min_fraction: float = 0.0,
    min_coverage: float = 0.0,
    min_bridging_batches: int = 1,
    mode: str = "sum",
    min_samples: int = 3,
    n_repetitions: int = 10,
    resampling_fraction: float = 0.6,
    min_list_overlap: float = 0.9,
    seed: int = 42,
    engine: str = "pydeseq2",
    **engine_kwargs,
) -> DEResult:
    """Perform differential expression analysis using pseudobulk data."""
    if isinstance(data, PseudobulkResult):
        pb_result = data
        logger.info("Using provided PseudobulkResult")
    else:
        if group_key is None or query is None:
            raise ValueError(
                "When passing AnnData, 'group_key' and 'query' are required. "
                "Alternatively, run scb.pp.pseudobulk() first and pass the result."
            )

        logger.info("Running pseudobulking...")
        pb_result = pseudobulk(
            adata=data,
            group_key=group_key,
            query=query,
            reference=reference,
            layer=layer,
            replicate_key=replicate_key,
            batch_key=batch_key,
            min_cells=min_cells,
            min_fraction=min_fraction,
            min_coverage=min_coverage,
            min_bridging_batches=min_bridging_batches,
            mode=mode,
        )

    rng = np.random.default_rng(seed)

    required_samples = _compute_required_samples(
        pb_result=pb_result,
        min_samples=min_samples,
    )

    de_engine = get_engine(engine)

    if all(v == 0 for v in required_samples.values()):
        logger.info(f"Running DE with {engine} engine...")
        return _run_de_direct(
            pb_result=pb_result,
            de_engine=de_engine,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
        )
    else:
        logger.info(
            f"Insufficient samples - generating pseudoreplicates "
            f"({n_repetitions} repetitions, {resampling_fraction:.0%} resampling)"
        )
        logger.info(f"Additional samples needed: {required_samples}")
        return _run_de_with_pseudoreplicates(
            pb_result=pb_result,
            de_engine=de_engine,
            required_samples=required_samples,
            n_repetitions=n_repetitions,
            resampling_fraction=resampling_fraction,
            min_list_overlap=min_list_overlap,
            rng=rng,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
        )


def _compute_required_samples(
    pb_result: PseudobulkResult,
    min_samples: int,
) -> dict[str, int]:
    required = {}
    for cond, samples in pb_result.valid_samples_by_condition.items():
        if pb_result.include_batch:
            current = sum(
                len(batches)
                for sample_id in samples
                if sample_id in pb_result.sample_hierarchy.get(cond, {})
                for batches in [pb_result.sample_hierarchy[cond][sample_id]]
            )
        else:
            current = len(samples)
        required[cond] = max(0, min_samples - current)
    return required


def _run_de_direct(
    pb_result: PseudobulkResult,
    de_engine,
    engine_name: str,
    engine_kwargs: dict,
) -> DEResult:
    results = de_engine.run(
        counts=pb_result.counts,
        metadata=pb_result.metadata,
        design=pb_result.design,
        contrast=pb_result.contrast,
        **engine_kwargs,
    )

    logger.info(f"DE complete: {len(results)} genes tested, {(results['padj'] < 0.05).sum()} significant (padj < 0.05)")

    return DEResult(
        results=results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=pb_result.design,
        engine=engine_name,
        used_pseudoreplicates=False,
        n_repetitions=1,
    )


def _run_de_with_pseudoreplicates(
    pb_result: PseudobulkResult,
    de_engine,
    required_samples: dict[str, int],
    n_repetitions: int,
    resampling_fraction: float,
    min_list_overlap: float,
    rng: np.random.Generator,
    engine_name: str,
    engine_kwargs: dict,
) -> DEResult:
    repetition_results = {}
    repetition_stats = {}

    # === PRE-COMPUTE EXPENSIVE OPERATIONS ONCE ===
    adata_sub = pb_result.adata_subset
    layer = pb_result.layer

    # Get count matrix ONCE
    X_full = adata_sub.layers[layer] if layer is not None else adata_sub.X

    # Convert to dense array ONCE (if sparse)
    if hasattr(X_full, "toarray"):
        X_full = X_full.toarray()

    # Build index mapping as numpy array for vectorized lookup
    obs_names_array = np.array(adata_sub.obs_names)
    obs_idx_to_pos = {idx: pos for pos, idx in enumerate(obs_names_array)}
    var_names = list(adata_sub.var_names)
    n_genes = len(var_names)

    # Pre-extract base counts as numpy array
    base_counts = pb_result.counts.values
    base_sample_names = list(pb_result.counts.index)
    n_base_samples = len(base_sample_names)

    # Pre-extract base metadata
    base_metadata = pb_result.metadata
    group_key = pb_result.contrast[0]
    include_batch = pb_result.include_batch
    batch_key = pb_result.batch_key if include_batch else None

    # Calculate total pseudoreplicates needed
    total_pr = sum(max(0, n) for n in required_samples.values())
    n_total_samples = n_base_samples + total_pr

    # Pre-allocate combined counts array (reused each repetition)
    combined_counts_array = np.empty((n_total_samples, n_genes), dtype=base_counts.dtype)
    combined_counts_array[:n_base_samples, :] = base_counts

    # Pre-compute sample hierarchy info for fast access
    sample_hierarchy = pb_result.sample_hierarchy
    pr_info = []  # List of (condition, sample_ids, batches_dict) for conditions needing PRs
    for condition, n_needed in required_samples.items():
        if n_needed > 0:
            sample_ids = list(sample_hierarchy.get(condition, {}).keys())
            if sample_ids:
                pr_info.append((condition, n_needed, sample_ids, sample_hierarchy[condition]))

    for rep_idx in range(n_repetitions):
        logger.debug(f"Repetition {rep_idx + 1}/{n_repetitions}")

        # Generate pseudoreplicates directly into pre-allocated array
        pr_names = []
        pr_meta_conditions = []
        pr_meta_batches = []
        pr_idx = n_base_samples

        for condition, n_needed, sample_ids, cond_hierarchy in pr_info:
            for i in range(n_needed):
                source_sample = rng.choice(sample_ids)
                batches = cond_hierarchy[source_sample]

                if include_batch and len(batches) > 1:
                    source_batch = rng.choice(list(batches.keys()))
                    obs_names = batches[source_batch]
                else:
                    obs_names = [cell for cells in batches.values() for cell in cells]

                # Sample cells
                n_sample = max(1, int(len(obs_names) * resampling_fraction))
                sampled_cells = rng.choice(obs_names, size=n_sample, replace=False)

                # Vectorized position lookup and sum
                sampled_positions = np.array([obs_idx_to_pos[c] for c in sampled_cells])
                combined_counts_array[pr_idx, :] = X_full[sampled_positions, :].sum(axis=0)

                pseudo_sample_name = f"{source_sample}_pr_{i + 1}"
                pr_names.append(pseudo_sample_name)
                pr_meta_conditions.append(condition)

                if include_batch:
                    assigned_batch = source_batch if len(batches) > 1 else list(batches.keys())[0]
                else:
                    assigned_batch = "_psbulk_no_batch"
                pr_meta_batches.append(assigned_batch)

                pr_idx += 1

        # Build DataFrames efficiently
        all_sample_names = base_sample_names + pr_names
        rep_counts = pd.DataFrame(
            combined_counts_array,
            index=all_sample_names,
            columns=var_names,
        )

        # Build metadata by extending base
        if pr_names:
            pr_meta_dict = {group_key: pr_meta_conditions}
            if include_batch:
                pr_meta_dict[batch_key] = pr_meta_batches
            pr_metadata_df = pd.DataFrame(pr_meta_dict, index=pr_names)
            rep_metadata = pd.concat([base_metadata, pr_metadata_df], axis=0)
        else:
            rep_metadata = base_metadata

        try:
            rep_result = de_engine.run(
                counts=rep_counts,
                metadata=rep_metadata,
                design=pb_result.design,
                contrast=pb_result.contrast,
                **engine_kwargs,
            )
            repetition_results[rep_idx] = rep_result
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Repetition {rep_idx + 1} failed: {e}")
            continue

    if not repetition_results:
        raise RuntimeError("All repetitions failed. Check your data and parameters.")

    aggregated_results = _aggregate_de_results(
        results=repetition_results,
        min_list_overlap=min_list_overlap,
    )

    n_successful = len(repetition_results)
    logger.info(
        f"DE complete: {n_successful}/{n_repetitions} repetitions successful, "
        f"{len(aggregated_results)} genes tested, "
        f"{(aggregated_results['padj'] < 0.05).sum()} significant (padj < 0.05)"
    )

    return DEResult(
        results=aggregated_results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=pb_result.design,
        engine=engine_name,
        used_pseudoreplicates=True,
        n_repetitions=n_successful,
        repetition_results={str(k): v for k, v in repetition_results.items()},
        repetition_stats=repetition_stats,
    )


def _aggregate_de_results(
    results: dict[int, pd.DataFrame],
    min_list_overlap: float,
) -> pd.DataFrame:
    """Aggregate DE results efficiently using numpy operations."""
    n_runs = len(results)
    min_occurrences = int(np.ceil(min_list_overlap * n_runs))

    # Get all unique genes and count occurrences
    all_genes = {}
    for df in results.values():
        for gene in df.index:
            all_genes[gene] = all_genes.get(gene, 0) + 1

    # Filter genes by occurrence threshold
    keep_genes = [g for g, count in all_genes.items() if count >= min_occurrences]

    if not keep_genes:
        logger.warning(
            f"No genes passed min_list_overlap threshold ({min_list_overlap:.0%}). "
            "Returning all genes with aggregated stats."
        )
        keep_genes = list(all_genes.keys())

    keep_genes_set = set(keep_genes)

    # Pre-allocate arrays for aggregation
    numeric_cols = None
    gene_sums = {}
    gene_counts = {}

    for df in results.values():
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for gene in df.index:
            if gene in keep_genes_set:
                row_values = df.loc[gene, numeric_cols].values
                if gene not in gene_sums:
                    gene_sums[gene] = row_values.astype(float)
                    gene_counts[gene] = 1
                else:
                    gene_sums[gene] += row_values
                    gene_counts[gene] += 1

    # Compute means
    result_data = {gene: gene_sums[gene] / gene_counts[gene] for gene in keep_genes}
    aggregated = pd.DataFrame.from_dict(result_data, orient="index", columns=numeric_cols)

    return aggregated
