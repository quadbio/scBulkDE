"""Pseudobulking functions."""

from __future__ import annotations

import time
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
    layer: str | None = None,
    replicate_key: str | None = None,
    batch_key: str | None = None,
    min_cells: int = 10,
    min_fraction: float = 0.0,
    min_coverage: float = 0.0,
    min_bridging_batches: int = 1,
    mode: str = "sum",
) -> PseudobulkResult:
    """Pseudobulk single-cell data for DE analysis."""
    # Validate inputs
    validate_adata(adata, layer, group_key, replicate_key, batch_key)
    validate_groups(adata, group_key, query, reference)

    # Resolve reference. If reference is rest, use all other groups and
    # if not, use provided reference(s) and make sure it is a list.
    if reference == "rest":
        all_groups = adata.obs[group_key].unique()
        reference_list = [g for g in all_groups if g != query]
    else:
        reference_list = [reference] if isinstance(reference, str) else list(reference)

    # Mask stores which cells to keep (query + reference). This yields all cells when
    # reference is "rest" or reference contains all other groups.
    group_values = adata.obs[group_key].values
    all_conditions = np.asarray([query, *reference_list])
    mask = np.isin(group_values, all_conditions)

    # Create unified condition labels "query" and "reference" for internal use
    condition_labels = np.empty_like(group_values)
    condition_labels[:] = "reference"
    condition_labels[group_values == query] = "query"

    # Subset AnnData and condition labels
    adata_subset = adata[mask]
    condition_labels_subset = condition_labels[mask]

    internal_group_key = "_psbulk_condition"
    conditions = ["query", "reference"]

    start = time.time()
    adata_subset, sample_hierarchy, info, design, include_batch, condition_labels_subset = _identify_samples_and_design(
        adata_sub=adata_subset,
        condition_labels=condition_labels_subset,
        group_key=internal_group_key,
        conditions=conditions,
        replicate_key=replicate_key,
        batch_key=batch_key,
        min_cells=min_cells,
        min_fraction=min_fraction,
        min_coverage=min_coverage,
        min_bridging_batches=min_bridging_batches,
    )
    end = time.time()
    logger.info(f"Identified samples and design in {end - start:.4f} seconds.")

    condition_totals = {
        "query": np.count_nonzero(condition_labels_subset == "query"),
        "reference": np.count_nonzero(condition_labels_subset == "reference"),
    }

    start = time.time()
    sample_stats = _compute_sample_stats(
        sample_hierarchy=sample_hierarchy,
        valid_samples_by_condition=info["valid_samples_by_condition"],
        condition_totals=condition_totals,
    )
    end = time.time()
    logger.info(f"Computed sample statistics in {end - start:.4f} seconds.")

    start = time.time()
    counts, metadata = _aggregate_counts_direct(
        adata_sub=adata_subset,
        sample_hierarchy=sample_hierarchy,
        valid_samples_by_condition=info["valid_samples_by_condition"],
        condition_labels=condition_labels_subset,
        group_key=internal_group_key,
        batch_key=info["used_batch_key"] if include_batch else None,
        layer=layer,
        mode=mode,
    )
    end = time.time()
    logger.info(f"Aggregated counts in {end - start:.4f} seconds.")

    contrast = [internal_group_key, "query", "reference"]

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
    adata_sub,
    condition_labels: np.ndarray,
    group_key: str,
    conditions: list[str],
    replicate_key: str | None,
    batch_key: str | None,
    min_cells: int,
    min_fraction: float,
    min_coverage: float,
    min_bridging_batches: int,
):
    """Identify valid samples, build design formula, and detect collapsing."""
    obs = adata_sub.obs
    obs_index = np.asarray(obs.index)
    n_cells = adata_sub.n_obs

    # Batch and replicate value arrays, or placeholders if not provided
    batch_values = (
        np.asarray(obs[batch_key]).astype(str, copy=False)
        if batch_key
        else np.full(n_cells, "_psbulk_no_batch", dtype=object)
    )
    sample_values = np.asarray(obs[replicate_key]).astype(str, copy=False) if replicate_key else None

    # Determine mode and keys
    if replicate_key and batch_key:
        mode, stratify_by_batch, sample_key = "replicate_and_batch", True, replicate_key
    elif replicate_key:
        mode, stratify_by_batch, sample_key = "replicate_only", False, replicate_key
    elif batch_key:
        mode, stratify_by_batch, sample_key = "batch_only", False, batch_key
    else:
        mode, stratify_by_batch, sample_key = "no_replicates", False, None

    internal_sample_key = "_psbulk_sample"
    internal_batch_key = "_psbulk_batch"
    used_batch_key = batch_key if batch_key else internal_batch_key

    # Build sample_ids per cell
    condition_labels_str = condition_labels.astype(str)
    if sample_key:
        if stratify_by_batch:
            sample_ids = condition_labels_str + "_" + sample_values + "_" + batch_values
        else:
            sample_ids = condition_labels_str + "_" + sample_values
        used_replicate_key = internal_sample_key
    else:
        sample_ids = np.empty(n_cells, dtype=object)
        used_replicate_key = internal_sample_key

    sample_hierarchy = {}
    valid_samples_by_condition = {}
    collapsed_conditions = []
    valid_mask = np.ones(n_cells, dtype=bool)

    for cond in conditions:
        cond_mask = condition_labels == cond
        cond_indices = np.where(cond_mask)[0]
        cond_total = cond_indices.size
        cond_obs_names = obs_index[cond_indices]

        if not sample_key:
            collapsed_id = f"{cond}_collapsed"
            sample_hierarchy[cond] = {collapsed_id: {"_psbulk_no_batch": cond_obs_names}}
            valid_samples_by_condition[cond] = [collapsed_id]
            collapsed_conditions.append(cond)
            sample_ids[cond_indices] = collapsed_id
            continue

        cond_sample_ids = sample_ids[cond_indices]

        # Group and count sample IDs for this condition
        unique_samples, counts = np.unique(cond_sample_ids, return_counts=True)
        sample_fractions = counts / cond_total

        # Find valid samples by min_cells or fraction
        valid_sample_mask = (counts >= min_cells) | (sample_fractions >= min_fraction)
        valid_sample_names = unique_samples[valid_sample_mask]

        # Filter for global min_coverage per condition
        if valid_sample_names.size:
            valid_counts = counts[valid_sample_mask]
            coverage = valid_counts.sum() / cond_total
            if coverage < min_coverage:
                valid_sample_names = np.array([], dtype=unique_samples.dtype)

        # Collapse if no valid samples
        if valid_sample_names.size == 0:
            collapsed_id = f"{cond}_collapsed"
            valid_sample_names = np.array([collapsed_id], dtype=object)
            collapsed_conditions.append(cond)
            sample_ids[cond_indices] = collapsed_id
            cond_sample_ids = sample_ids[cond_indices]

        valid_samples_by_condition[cond] = valid_sample_names.tolist()

        # Mark invalid cells in valid_mask
        is_valid = np.isin(cond_sample_ids, valid_sample_names)
        valid_mask[cond_indices] = is_valid

        # Construct per-sample hierarchy for this condition
        sample_hierarchy[cond] = {}
        cond_batch_values = batch_values[cond_indices]
        for sample_name in valid_sample_names:
            sample_cell_mask = cond_sample_ids == sample_name
            sample_obs_names = cond_obs_names[sample_cell_mask]
            sample_batches = cond_batch_values[sample_cell_mask]

            if cond in collapsed_conditions:
                sample_hierarchy[cond][sample_name] = {"_psbulk_no_batch": sample_obs_names}
            elif mode == "replicate_and_batch":
                batch_val = sample_batches[0]
                sample_hierarchy[cond][sample_name] = {batch_val: sample_obs_names}
            elif mode == "batch_only":
                original_batch = sample_name.replace(f"{cond}_", "", 1)
                sample_hierarchy[cond][sample_name] = {original_batch: sample_obs_names}
            else:
                sample_hierarchy[cond][sample_name] = {"_psbulk_no_batch": sample_obs_names}

    # Filter all arrays by valid_mask
    adata_sub = adata_sub[valid_mask].copy()
    condition_labels = condition_labels[valid_mask]
    sample_ids = sample_ids[valid_mask]
    batch_values = batch_values[valid_mask]

    adata_sub.obs[group_key] = condition_labels
    adata_sub.obs[internal_sample_key] = sample_ids
    adata_sub.obs[internal_batch_key] = batch_values

    # Design: Should batch be included?
    include_batch, bridging_batches = False, []
    if batch_key and not any(c in collapsed_conditions for c in conditions):
        # For every condition, collect batch names for its samples (except collapsed)
        batches_per_cond = {
            cond: {
                batch
                for samples in sample_hierarchy[cond].values()
                for batch in samples.keys()
                if batch != "_psbulk_no_batch"
            }
            for cond in conditions
        }
        # Only consider conditions with some non-empty batch set
        all_batch_sets = [batches for batches in batches_per_cond.values() if batches]
        if len(all_batch_sets) == len(conditions):
            # Bridging batches: present in all conditions
            bridging_batches = list(set.intersection(*all_batch_sets))
            include_batch = len(bridging_batches) >= min_bridging_batches

    design = f"~{group_key}+{batch_key}" if include_batch else f"~{group_key}"

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
    logger.debug(f"Valid samples:  {valid_samples_by_condition}")
    if collapsed_conditions:
        logger.warning(f"Collapsed conditions (insufficient replicates): {collapsed_conditions}")

    return adata_sub, sample_hierarchy, info, design, include_batch, condition_labels


def _compute_sample_stats(
    sample_hierarchy: dict,
    valid_samples_by_condition: dict[str, list[str]],
    condition_totals: dict[str, int],
) -> pd.DataFrame:
    """Compute per-sample statistics."""
    rows = []
    for cond, samples in sample_hierarchy.items():
        valid_set = set(valid_samples_by_condition.get(cond, []))
        cond_total = condition_totals[cond]
        for sample_id, batches in samples.items():
            is_valid = sample_id in valid_set
            for batch_id, cell_indices in batches.items():
                n_cells = len(cell_indices)
                rows.append(
                    {
                        "condition": cond,
                        "batch": batch_id,
                        "psbulk_sample": sample_id,
                        "n_cells": n_cells,
                        "fraction": n_cells / cond_total,
                        "is_valid": is_valid,
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_counts_direct(
    adata_sub,
    sample_hierarchy,
    valid_samples_by_condition,
    condition_labels,
    group_key,
    batch_key,
    layer,
    mode,
):
    """Optimized: Aggregate counts directly without decoupler."""
    import numpy as np
    import pandas as pd

    X = adata_sub.layers[layer] if layer is not None else adata_sub.X
    is_sparse = hasattr(X, "tocsc") or hasattr(X, "tocoo")

    var_names = np.array(adata_sub.var_names)
    obs_names = np.array(adata_sub.obs_names)
    obs_to_pos = {name: i for i, name in enumerate(obs_names)}

    # Precompute all sample and cell mappings first
    all_samples = []
    all_positions = []

    for cond in ["query", "reference"]:
        for sample_id in valid_samples_by_condition.get(cond, []):
            for batch_id, cell_names in sample_hierarchy[cond][sample_id].items():
                all_samples.append((sample_id, cond, batch_id))
                # Instead of generator, use np.fromiter
                positions = np.fromiter(
                    (obs_to_pos[name] for name in cell_names if name in obs_to_pos), dtype=int, count=len(cell_names)
                )
                all_positions.append(positions)

    n_samples = len(all_samples)
    n_genes = len(var_names)
    counts_array = np.zeros((n_samples, n_genes), dtype=np.float64)

    sample_names = []
    meta_conditions = []
    meta_batches = []

    for i, ((sample_id, cond, batch_id), positions) in enumerate(zip(all_samples, all_positions, strict=True)):
        if positions.size > 0:
            subset_X = X[positions, :]  # This can be sparse or dense
            if is_sparse:
                subset_X = subset_X.toarray() if mode != "sum" else subset_X  # Only densify for mean/median
            if mode == "sum":
                # For sparse, .sum(axis=0) returns a dense np.array
                counts_array[i, :] = np.asarray(subset_X.sum(axis=0)).ravel()
            elif mode == "mean":
                counts_array[i, :] = np.mean(subset_X, axis=0)
            elif mode == "median":
                counts_array[i, :] = np.median(subset_X, axis=0)

        sample_names.append(sample_id)
        meta_conditions.append(cond)
        meta_batches.append(batch_id)

    counts = pd.DataFrame(counts_array, index=sample_names, columns=var_names)
    meta_dict = {group_key: meta_conditions}
    if batch_key is not None:
        meta_dict[batch_key] = meta_batches
    metadata = pd.DataFrame(meta_dict, index=sample_names)

    return counts, metadata
