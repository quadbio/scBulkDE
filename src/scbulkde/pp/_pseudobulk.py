"""Pseudobulking functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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
    layer: str = "X",
    replicate_key: str | None = None,
    batch_key: str | None = None,
    min_cells: int = 50,
    min_fraction: float = 0.2,
    min_coverage: float = 0.8,
    min_bridging_batches: int = 2,
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
        Layer containing counts. Use "X" for adata.X.
    replicate_key
        Column defining biological replicates.
        If None and batch_key is None, each condition is collapsed into one sample.
    batch_key
        Column defining batches.
        If replicate_key is None, batches are used as technical replicates.
        If replicate_key is provided, samples are stratified by batch.
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
    adata_subset, sample_hierarchy, info, design, include_batch = _identify_samples_and_design(
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

    # Aggregate counts for valid samples
    counts, metadata = _aggregate_counts(
        adata_sub=adata_subset,
        sample_hierarchy=sample_hierarchy,
        group_key=internal_group_key,
        batch_key=info["used_batch_key"],
        layer=layer,
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
    - If replicate_key provided: use as biological replicates, optionally stratify by batch
    - If only batch_key provided: use batches as technical replicates, check for bridging
    - If neither: collapse each condition into single sample

    Returns
    -------
    tuple
        (adata_sub, sample_hierarchy, info, design, include_batch)
    """
    adata_sub = adata_sub.copy()
    obs = adata_sub.obs

    # Determine sample key and whether to stratify by batch
    if replicate_key is not None:
        sample_key = replicate_key
        stratify_by_batch = batch_key is not None
        batch_is_sample = False
    elif batch_key is not None:
        sample_key = batch_key
        stratify_by_batch = False
        batch_is_sample = True
    else:
        sample_key = None
        stratify_by_batch = False
        batch_is_sample = False

    # Create internal sample column (prepend condition to make unique)
    internal_sample_key = "_psbulk_sample"
    if sample_key is not None:
        adata_sub.obs[internal_sample_key] = obs[group_key].astype(str) + "_" + obs[sample_key].astype(str)

    # Determine used keys
    used_replicate_key = internal_sample_key if sample_key is not None else "_psbulk_sample"
    used_batch_key = batch_key if batch_key is not None else "_psbulk_batch"

    if batch_key is None:
        adata_sub.obs["_psbulk_batch"] = "_psbulk_no_batch"

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

            if stratify_by_batch and cond not in collapsed_conditions:
                sample_hierarchy[cond][sample_id] = {}
                for batch_val, batch_group in sample_cells.groupby(batch_key, observed=True):
                    if len(batch_group) > 0:
                        sample_hierarchy[cond][sample_id][batch_val] = batch_group.index.tolist()
            else:
                # When batch_is_sample, store original batch name for bridging check
                if batch_is_sample and cond not in collapsed_conditions:
                    original_batch = sample_id.replace(f"{cond}_", "", 1)
                    sample_hierarchy[cond][sample_id] = {original_batch: sample_cells.index.tolist()}
                else:
                    sample_hierarchy[cond][sample_id] = {"_psbulk_no_batch": sample_cells.index.tolist()}

    # Filter adata_sub to only include cells in valid samples
    all_valid_samples = [s for samples in valid_samples_by_condition.values() for s in samples]
    adata_sub = adata_sub[adata_sub.obs[internal_sample_key].isin(all_valid_samples)].copy()

    # Determine if batch should be included in design
    include_batch = False
    bridging_batches = []

    # Case 1: replicate_key provided with batch_key (stratify by batch)
    if stratify_by_batch and batch_key is not None:
        batches_per_cond = {}
        for cond in conditions:
            if cond not in collapsed_conditions:
                batches_per_cond[cond] = {b for samples in sample_hierarchy[cond].values() for b in samples.keys()}
            else:
                batches_per_cond[cond] = set()

        # Find batches present in all conditions
        all_batch_sets = [batches_per_cond[c] for c in conditions if batches_per_cond[c]]
        if all_batch_sets:
            bridging_batches = list(set.intersection(*all_batch_sets))
        include_batch = len(bridging_batches) >= min_bridging_batches

    # Case 2: only batch_key provided (batch IS the sample)
    elif batch_is_sample and batch_key is not None:
        batches_per_cond = {}
        for cond in conditions:
            if cond not in collapsed_conditions:
                batches_per_cond[cond] = {
                    batch_name for sample_dict in sample_hierarchy[cond].values() for batch_name in sample_dict.keys()
                }
            else:
                batches_per_cond[cond] = set()

        all_batch_sets = [batches_per_cond[c] for c in conditions if batches_per_cond[c]]
        if all_batch_sets:
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
        "is_technical_replicates": batch_is_sample,
    }

    logger.debug(f"Design formula: {design}")
    logger.debug(f"Valid samples: {valid_samples_by_condition}")
    if collapsed_conditions:
        logger.warning(f"Collapsed conditions (insufficient replicates): {collapsed_conditions}")
    if info["is_technical_replicates"]:
        logger.info("Using batches as technical replicates")
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
    sample_hierarchy: dict,
    group_key: str,
    batch_key: str,
    layer: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate counts by sample."""
    # Get counts matrix
    if layer == "X":
        X = adata_sub.X
    else:
        X = adata_sub.layers[layer]

    # Convert to dense if sparse
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Create mapping from obs index to row number
    obs_to_row = {obs_name: i for i, obs_name in enumerate(adata_sub.obs_names)}

    counts_list = []
    metadata_rows = []

    for cond, samples in sample_hierarchy.items():
        for sample_id, batches in samples.items():
            for batch_id, cell_indices in batches.items():
                # Get row indices for these cells
                row_indices = [obs_to_row[idx] for idx in cell_indices if idx in obs_to_row]

                if not row_indices:
                    continue

                # Sum counts
                sample_counts = X[row_indices].sum(axis=0)
                counts_list.append(np.asarray(sample_counts).flatten())

                # Sample name includes batch if stratified
                if len(batches) > 1 or batch_id != "_psbulk_no_batch":
                    full_sample_id = f"{sample_id}_{batch_id}"
                else:
                    full_sample_id = sample_id

                metadata_rows.append(
                    {
                        "sample": full_sample_id,
                        group_key: cond,
                        batch_key: batch_id,
                    }
                )

    sample_names = [row["sample"] for row in metadata_rows]

    counts = pd.DataFrame(
        counts_list,
        index=sample_names,
        columns=adata_sub.var_names,
    )

    metadata = pd.DataFrame(metadata_rows).set_index("sample")

    return counts, metadata
