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
    adata_subset, sample_hierarchy, info, design, include_batch = _identify_samples_and_design(
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

    start = time.time()
    condition_totals = {
        "query": np.count_nonzero(condition_labels_subset == "query"),
        "reference": np.count_nonzero(condition_labels_subset == "reference"),
    }
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
    """
    Identify valid samples, build design formula, and detect collapsing.

    A "sample" is the minimal observational unit over which cell counts are aggregated prior to differential analysis.
    Its definition depends on which keys are provided:
      - replicate_key and batch_key: Sample = same condition + replicate + batch (ex: 'disease_mouse1_runA')
      - replicate_key only: Sample = same condition + replicate (ex: 'disease_mouse1')
      - batch_key only: Sample = same condition + batch (ex: 'disease_runA')
      - neither: All cells in a condition are collapsed to one sample (ex: 'disease_collapsed')
    """
    obs = adata_sub.obs
    obs_index = np.asarray(obs.index)
    n_cells = adata_sub.n_obs

    # Prepare keys and arrays
    batch_vals = (
        np.asarray(obs[batch_key]).astype(str, copy=False)
        if batch_key
        else np.full(n_cells, "psbulk-no-batch", dtype=object)
    )
    replicate_vals = (
        np.asarray(obs[replicate_key]).astype(str, copy=False)
        if replicate_key
        else np.full(n_cells, "psbulk-no-replicate", dtype=object)
    )
    condition_str = condition_labels.astype(str)

    # sample_ids must be dtype=object.
    # NumPy fixed-width string arrays (<U*) silently truncate on assignment;
    # collapsing conditions assigns longer IDs later, which would otherwise be cut off.
    sample_ids = np.array(
        condition_str + "_" + replicate_vals + "_" + batch_vals,
        dtype=object,
    )

    sample_hierarchy = {}
    valid_samples_by_condition = {}
    collapsed_conditions = []
    valid_mask = np.ones(n_cells, dtype=bool)

    for cond in conditions:
        indices = np.where(condition_labels == cond)[0]
        cond_sample_ids = sample_ids[indices]
        cond_obs_names = obs_index[indices]

        # Count samples and determine validity
        unique_samples, sample_counts = np.unique(cond_sample_ids, return_counts=True)
        fractions = sample_counts / len(indices)
        valid_mask_samples = (sample_counts >= min_cells) | (fractions >= min_fraction)

        if valid_mask_samples.any():
            coverage = sample_counts[valid_mask_samples].sum() / len(indices)
            if coverage < min_coverage:
                valid_mask_samples[:] = False

        if not valid_mask_samples.any():
            collapsed_id = f"{cond}_psbulk-no-replicate_psbulk-no-batch"
            sample_ids[indices] = collapsed_id
            cond_sample_ids = sample_ids[indices]
            unique_samples = np.array([collapsed_id], dtype=object)
            valid_mask_samples = np.array([True])
            collapsed_conditions.append(cond)

        valid_samples = unique_samples[valid_mask_samples]
        valid_samples_by_condition[cond] = valid_samples

        # Mark valid cells
        is_valid = np.isin(cond_sample_ids, valid_samples)
        valid_mask[indices] = is_valid

        # Build hierarchy: condition -> replicate -> batch -> cell names
        sample_hierarchy[cond] = {}
        for sample_name in valid_samples:
            mask = cond_sample_ids == sample_name
            sample_parts = sample_name.split("_")
            sample_replicate, sample_batch = sample_parts[1], sample_parts[2]
            sample_obs_names = cond_obs_names[mask]
            rep_dict = sample_hierarchy[cond].setdefault(sample_replicate, {})
            rep_dict.setdefault(sample_batch, []).extend(sample_obs_names)

    # Filter arrays by valid_mask
    adata_sub = adata_sub[valid_mask]

    # Decide design formula and batch inclusion
    include_batch = False
    bridging_batches = []
    if batch_key and not any(c in collapsed_conditions for c in conditions):
        batches_per_cond = {
            cond: {
                batch
                for samples in sample_hierarchy[cond].values()
                for batch in samples.keys()
                if batch != "psbulk-no-batch"
            }
            for cond in conditions
        }
        all_batch_sets = [batches for batches in batches_per_cond.values() if batches]
        if len(all_batch_sets) == len(conditions):
            bridging_batches = list(set.intersection(*all_batch_sets))
            include_batch = len(bridging_batches) >= min_bridging_batches

    design = f"~{group_key}+{batch_key}" if include_batch else f"~{group_key}"

    info = {
        "valid_samples_by_condition": valid_samples_by_condition,
        "collapsed_conditions": collapsed_conditions,
        "used_replicate_key": replicate_key,
        "used_batch_key": batch_key,
        "bridging_batches": bridging_batches,
    }

    logger.debug(f"Design formula: {design}")
    logger.debug(f"Valid samples: {valid_samples_by_condition}")

    if collapsed_conditions:
        logger.warning(f"Collapsed conditions (insufficient samples): {collapsed_conditions}")

    return adata_sub, sample_hierarchy, info, design, include_batch


def _compute_sample_stats(
    sample_hierarchy: dict,
    valid_samples_by_condition: dict[str, list[str]],
    condition_totals: dict[str, int],
) -> pd.DataFrame:
    """Compute per-sample statistics for hierarchical sample structure."""
    rows = []

    for cond, replicates in sample_hierarchy.items():
        valid_sample_set = set(valid_samples_by_condition.get(cond, []))
        cond_total = condition_totals[cond]
        for rep, batches in replicates.items():
            for batch, cell_names in batches.items():
                sample_id = f"{cond}_{rep}_{batch}"
                is_valid = sample_id in valid_sample_set
                n_cells = len(cell_names)
                rows.append(
                    {
                        "condition": cond,
                        "replicate": rep,
                        "batch": batch,
                        "_psbulk_sample": sample_id,
                        "n_cells": n_cells,
                        "fraction": n_cells / cond_total if cond_total > 0 else 0.0,
                        "is_valid": is_valid,
                    }
                )
    return pd.DataFrame(rows)


def _aggregate_counts_direct(
    adata_sub,
    sample_hierarchy,
    group_key,
    batch_key,
    layer,
    mode,
):
    """Optimized: Aggregate counts directly with updated sample_hierarchy structure."""
    import numpy as np
    import pandas as pd

    X = adata_sub.layers[layer] if layer is not None else adata_sub.X
    is_sparse = hasattr(X, "tocsc") or hasattr(X, "tocoo")

    var_names = np.array(adata_sub.var_names)
    obs_names = np.array(adata_sub.obs_names)
    obs_to_pos = {name: i for i, name in enumerate(obs_names)}

    all_samples = []
    all_positions = []
    meta_conditions = []
    meta_replicates = []
    meta_batches = []

    # Traverse: condition -> replicate -> batch -> cell_names
    for cond in ["query", "reference"]:
        valid_replicates = sample_hierarchy.get(cond, {})
        for rep, batches in valid_replicates.items():
            for batch, cell_names in batches.items():
                all_samples.append(f"{cond}_{rep}_{batch}")
                meta_conditions.append(cond)
                meta_replicates.append(rep)
                meta_batches.append(batch)
                # Convert cell names to positions, skip if name not found
                positions = np.fromiter(
                    (obs_to_pos[name] for name in cell_names if name in obs_to_pos), dtype=int, count=len(cell_names)
                )
                all_positions.append(positions)

    n_samples = len(all_samples)
    n_genes = len(var_names)
    counts_array = np.zeros((n_samples, n_genes), dtype=np.float64)

    for i, positions in enumerate(all_positions):
        if positions.size > 0:
            subset_X = X[positions, :]
            if is_sparse:
                subset_X = subset_X.toarray() if mode != "sum" else subset_X  # Densify for mean/median
            if mode == "sum":
                counts_array[i, :] = np.asarray(subset_X.sum(axis=0)).ravel()
            elif mode == "mean":
                counts_array[i, :] = np.mean(subset_X, axis=0)
            elif mode == "median":
                counts_array[i, :] = np.median(subset_X, axis=0)
            else:
                raise ValueError(f"Unsupported mode '{mode}'.")

    # For sample_names, use constructed IDs: f"{cond}_{rep}_{batch}"
    counts = pd.DataFrame(counts_array, index=all_samples, columns=var_names)

    # Metadata DataFrame construction
    meta_dict = {
        group_key: meta_conditions,
        "psbulk_replicate": meta_replicates,
        "psbulk_batch": meta_batches,
    }
    if batch_key is not None:
        # set user batch_key to meta_batches as well, as in the old code
        meta_dict[batch_key] = meta_batches
    metadata = pd.DataFrame(meta_dict, index=all_samples)

    return counts, metadata
