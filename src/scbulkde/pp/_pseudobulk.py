"""Pseudobulking functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from scbulkde.utils import logger, validate_adata, validate_groups

from ._result import PseudobulkResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import anndata as ad


def pseudobulk(
    adata: ad.AnnData,
    group_key: str,
    query: str,
    reference: str | Sequence[str] = "rest",
    *,
    layer: str | None = None,
    replicate_key: str | None = None,
    batch_key: str | None = None,
    min_cells: int = 10,
    min_fraction: float = 0.0,
    min_coverage: float = 0.0,
    min_bridging_batches: int = 1,
    mode: str = "sum",
) -> PseudobulkResult:
    """Pseudobulk single-cell data for DE analysis.

    Parameters
    ----------
    adata
        Annotated data matrix with raw counts.
    group_key
        Column in adata.obs containing condition groups.
    query
        Query condition (e.g., "treated").
    reference
        Reference condition(s), or "rest" for all other groups.
        If multiple conditions are provided, they are combined into
        a single "reference" group.
    layer
        Layer containing counts. If None, uses adata.X.
    replicate_key
        Column defining biological replicates.
        If None and batch_key is None, each condition is collapsed into one sample.
    batch_key
        Column defining batches.
        If replicate_key is None, batches are used as technical replicates.
        If replicate_key is provided, samples are (replicate, batch) combinations.
    min_cells
        Minimum cells required per sample.
    min_fraction
        Minimum fraction of condition total required per sample.
        Samples passing either min_cells OR min_fraction are kept.
    min_coverage
        Minimum fraction of cells that must be in valid samples.
        If not met, condition is collapsed.
    min_bridging_batches
        Minimum number of batches that must bridge both conditions
        to include batch in design formula.
    mode
        Aggregation mode for pseudobulk. Options: "sum", "mean", "median".

    Returns
    -------
    PseudobulkResult
        Container with pseudobulked counts, metadata, and QC stats.
    """
    # Validate inputs
    validate_adata(adata, layer, group_key, replicate_key, batch_key)
    validate_groups(adata, group_key, query, reference)

    # Resolve reference
    if reference == "rest":
        all_groups = adata.obs[group_key].unique().tolist()
        reference_list = [g for g in all_groups if g != query]
    else:
        reference_list = [reference] if isinstance(reference, str) else list(reference)

    # Subset to relevant conditions
    all_conditions = [query] + reference_list
    mask = adata.obs[group_key].isin(all_conditions)
    adata_subset = adata[mask].copy()

    # Create internal comparison column
    # Combine multiple reference groups into single "reference" label
    adata_subset.obs["_psbulk_condition"] = adata_subset.obs[group_key].apply(
        lambda x: "query" if x == query else "reference"
    )
    internal_group_key = "_psbulk_condition"
    conditions = ["query", "reference"]

    # Identify samples and build design
    (
        adata_subset,
        sample_hierarchy,
        info,
        design,
        include_batch,
    ) = _identify_samples_and_design(
        adata_sub=adata_subset,
        group_key=internal_group_key,
        conditions=conditions,
        replicate_key=replicate_key,
        batch_key=batch_key,
        min_cells=min_cells,
        min_fraction=min_fraction,
        min_coverage=min_coverage,
        min_bridging_batches=min_bridging_batches,
    )

    # Calculate condition totals (from original subset, before filtering)
    condition_totals = (
        adata[mask].obs[group_key].apply(lambda x: "query" if x == query else "reference").value_counts().to_dict()
    )

    # Build sample stats table
    sample_stats = _compute_sample_stats(
        sample_hierarchy=sample_hierarchy,
        valid_samples_by_condition=info["valid_samples_by_condition"],
        condition_totals=condition_totals,
    )

    # Aggregate counts using decoupler
    counts, metadata = _aggregate_counts(
        adata_sub=adata_subset,
        sample_key=info["used_replicate_key"],
        batch_key=info["used_batch_key"] if include_batch else None,
        layer=layer,
        mode=mode,
    )

    # Build contrast (using internal labels)
    contrast = [internal_group_key, "query", "reference"]

    # Mark original adata
    adata.uns["is_pseudobulked"] = True

    return PseudobulkResult(
        counts=counts,
        metadata=metadata,
        design=design,
        contrast=contrast,
        query=query,
        reference=reference_list[0] if len(reference_list) == 1 else reference_list,
        sample_stats=sample_stats,
        valid_samples_by_condition=info["valid_samples_by_condition"],
        collapsed_conditions=info["collapsed_conditions"],
        condition_totals=condition_totals,
        used_replicate_key=info["used_replicate_key"],
        used_batch_key=info["used_batch_key"],
        replicate_min_cells=min_cells,
        replicate_min_fraction=min_fraction,
        sample_hierarchy=sample_hierarchy,
        adata_subset=adata_subset,
        include_batch=include_batch,
        layer=layer,
        mode=mode,
    )


def _identify_samples_and_design(
    adata_sub: ad.AnnData,
    group_key: str,
    conditions: list[str],
    replicate_key: str | None,
    batch_key: str | None,
    min_cells: int,
    min_fraction: float,
    min_coverage: float,
    min_bridging_batches: int,
) -> tuple[ad.AnnData, dict, dict, str, bool]:
    """Identify samples and determine design formula.

    Logic:
    - If replicate_key provided with batch_key: samples are (replicate, batch) combinations
    - If only replicate_key: biological replicates, no batch
    - If only batch_key: batches are used as technical replicates
    - If neither: collapse each condition into single sample

    Returns
    -------
    tuple
        (adata_sub, sample_hierarchy, info, design, include_batch)
    """
    obs = adata_sub.obs.copy()

    # Determine mode based on provided keys
    if replicate_key is not None and batch_key is not None:
        # Both provided: samples are (replicate, batch) combinations
        mode = "replicate_and_batch"
        sample_key = replicate_key
        stratify_by_batch = True
    elif replicate_key is not None:
        # Only replicate: biological replicates, no batch
        mode = "replicate_only"
        sample_key = replicate_key
        stratify_by_batch = False
    elif batch_key is not None:
        # Only batch: use as technical replicates
        mode = "batch_only"
        sample_key = batch_key
        stratify_by_batch = False
    else:
        # Neither: collapse everything
        mode = "no_replicates"
        sample_key = None
        stratify_by_batch = False

    # Create internal columns
    internal_sample_key = "_psbulk_sample"
    internal_batch_key = "_psbulk_batch"

    # Set up batch column
    if batch_key is not None:
        adata_sub.obs[internal_batch_key] = obs[batch_key].astype(str)
        used_batch_key = batch_key
    else:
        adata_sub.obs[internal_batch_key] = "_psbulk_no_batch"
        used_batch_key = internal_batch_key

    # Create sample identifiers (prepend condition to make unique)
    if sample_key is not None:
        if stratify_by_batch:
            # Sample = condition_replicate_batch
            adata_sub.obs[internal_sample_key] = (
                obs[group_key].astype(str)
                + "_"
                + obs[sample_key].astype(str)
                + "_"
                + adata_sub.obs[internal_batch_key].astype(str)
            )
        else:
            # Sample = condition_replicate (or condition_batch if batch_only)
            adata_sub.obs[internal_sample_key] = obs[group_key].astype(str) + "_" + obs[sample_key].astype(str)
        used_replicate_key = internal_sample_key
    else:
        used_replicate_key = internal_sample_key

    sample_hierarchy = {}
    valid_samples_by_condition = {}
    collapsed_conditions = []

    for cond in conditions:
        cond_mask = adata_sub.obs[group_key] == cond
        cond_cells = adata_sub.obs[cond_mask]
        cond_total = len(cond_cells)

        if sample_key is None:
            # No sample info - collapse entire condition
            collapsed_id = f"{cond}_collapsed"
            sample_hierarchy[cond] = {collapsed_id: {"_psbulk_no_batch": cond_cells.index.tolist()}}
            valid_samples_by_condition[cond] = [collapsed_id]
            collapsed_conditions.append(cond)
            adata_sub.obs.loc[cond_mask, internal_sample_key] = collapsed_id
            continue

        # Count cells per sample
        sample_counts = cond_cells.groupby(internal_sample_key, observed=True).size()
        sample_fractions = sample_counts / cond_total

        # Identify valid samples (pass min_cells OR min_fraction)
        valid_mask = (sample_counts >= min_cells) | (sample_fractions >= min_fraction)
        valid_samples = sample_counts[valid_mask].index.tolist()

        # Check coverage
        if valid_samples:
            coverage = sample_counts[valid_samples].sum() / cond_total
            if coverage < min_coverage:
                valid_samples = []

        # Collapse if no valid samples
        if not valid_samples:
            collapsed_id = f"{cond}_collapsed"
            valid_samples = [collapsed_id]
            collapsed_conditions.append(cond)
            adata_sub.obs.loc[cond_mask, internal_sample_key] = collapsed_id
            cond_cells = adata_sub.obs[cond_mask]

        valid_samples_by_condition[cond] = valid_samples

        # Build hierarchy for this condition
        sample_hierarchy[cond] = {}
        for sample_id in valid_samples:
            sample_cells = cond_cells[cond_cells[internal_sample_key] == sample_id]

            if cond in collapsed_conditions:
                # Collapsed - no batch info
                sample_hierarchy[cond][sample_id] = {"_psbulk_no_batch": sample_cells.index.tolist()}
            elif mode == "replicate_and_batch":
                # Already stratified by batch in sample_id
                # Extract batch from obs
                batch_val = sample_cells[internal_batch_key].iloc[0]
                sample_hierarchy[cond][sample_id] = {batch_val: sample_cells.index.tolist()}
            elif mode == "batch_only":
                # Batch is the sample, store original batch name for bridging
                original_batch = sample_id.replace(f"{cond}_", "", 1)
                sample_hierarchy[cond][sample_id] = {original_batch: sample_cells.index.tolist()}
            else:
                # replicate_only - no real batch
                sample_hierarchy[cond][sample_id] = {"_psbulk_no_batch": sample_cells.index.tolist()}

    # Filter adata_sub to only include cells in valid samples
    all_valid_samples = [s for samples in valid_samples_by_condition.values() for s in samples]
    mask = adata_sub.obs[internal_sample_key].isin(all_valid_samples)
    adata_sub = adata_sub[mask]

    # Determine if batch should be included in design
    include_batch = False
    bridging_batches = []

    # Only consider batch if we have real batch info and no collapsed conditions
    if batch_key is not None and not any(c in collapsed_conditions for c in conditions):
        batches_per_cond = {}
        for cond in conditions:
            batches_per_cond[cond] = {
                batch_name
                for sample_dict in sample_hierarchy[cond].values()
                for batch_name in sample_dict.keys()
                if batch_name != "_psbulk_no_batch"
            }

        # Find bridging batches (present in all conditions)
        all_batch_sets = [batches_per_cond[c] for c in conditions if batches_per_cond[c]]
        if len(all_batch_sets) == len(conditions):
            bridging_batches = list(set.intersection(*all_batch_sets))
            include_batch = len(bridging_batches) >= min_bridging_batches

    # Determine design formula
    if include_batch:
        design = f"~{group_key}+{batch_key}"
    else:
        design = f"~{group_key}"

    info = {
        "valid_samples_by_condition": valid_samples_by_condition,
        "collapsed_conditions": collapsed_conditions,
        "used_replicate_key": used_replicate_key,
        "used_batch_key": used_batch_key,
        "bridging_batches": bridging_batches,
        "mode": mode,
    }

    logger.debug(f"Mode: {mode}")
    logger.debug(f"Design formula: {design}")
    logger.debug(f"Valid samples: {valid_samples_by_condition}")
    if collapsed_conditions:
        logger.warning(f"Collapsed conditions (insufficient replicates): {collapsed_conditions}")
    if bridging_batches:
        logger.debug(f"Bridging batches: {bridging_batches}")

    return adata_sub, sample_hierarchy, info, design, include_batch


def _compute_sample_stats(
    sample_hierarchy: dict,
    valid_samples_by_condition: dict[str, list[str]],
    condition_totals: dict[str, int],
) -> pd.DataFrame:
    """Compute per-sample statistics."""
    rows = []

    for cond, samples in sample_hierarchy.items():
        valid_samples = valid_samples_by_condition.get(cond, [])

        for sample_id, batches in samples.items():
            for batch_id, cell_indices in batches.items():
                n_cells = len(cell_indices)
                rows.append(
                    {
                        "condition": cond,
                        "batch": batch_id,
                        "psbulk_sample": sample_id,
                        "n_cells": n_cells,
                        "fraction": n_cells / condition_totals[cond],
                        "is_valid": sample_id in valid_samples,
                    }
                )

    return pd.DataFrame(rows)


def _aggregate_counts(
    adata_sub: ad.AnnData,
    sample_key: str,
    batch_key: str | None,
    layer: str | None,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate counts by sample using decoupler.

    Parameters
    ----------
    adata_sub
        Subsetted AnnData with valid samples only.
    sample_key
        Column in obs containing sample identifiers.
    batch_key
        Column in obs containing batch identifiers (for grouping).
        If None, no grouping is applied.
    layer
        Layer to use for counts. If None, uses adata.X.
    mode
        Aggregation mode: "sum", "mean", or "median".

    Returns
    -------
    tuple
        (counts DataFrame, metadata DataFrame)
    """
    import decoupler as dc

    # Run decoupler pseudobulk
    adata_pb = dc.pp.pseudobulk(
        adata_sub,
        sample_col=sample_key,
        groups_col=batch_key,
        layer=layer,
        mode=mode,
        empty=False,
        verbose=False,
    )

    # Filter out empty samples
    adata_pb = adata_pb[(adata_pb.obs["psbulk_cells"] > 0) & (adata_pb.obs["psbulk_counts"] > 0)].copy()

    # Extract counts
    X = adata_pb.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    counts = pd.DataFrame(
        X,
        index=adata_pb.obs_names,
        columns=adata_pb.var_names,
    )

    # Extract metadata (drop decoupler's QC columns for cleaner output)
    metadata_cols = [col for col in adata_pb.obs.columns if not col.startswith("psbulk_")]
    metadata = adata_pb.obs[metadata_cols].copy()

    return counts, metadata
