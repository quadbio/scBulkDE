"""Pseudobulking functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scbulkde.ut import (
    PseudobulkResult,
    aggregate_counts,
    logger,
    performance,
    validate_adata,
    validate_groups,
)

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
    min_cells: int = 50,
    min_fraction: float = 0.2,
    min_coverage: float = 0.75,
    min_bridging_batches: int = 2,
    mode: str = "sum",
    compute_sample_stats: bool = True,
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

    # Inform about replicate and batch key
    if replicate_key is not None and batch_key is not None:
        logger.info(f"Using replicate key '{replicate_key}' and batch key '{batch_key}' to define samples.")
    elif replicate_key is not None:
        logger.info(f"Using replicate key '{replicate_key}' to define samples.")
    elif batch_key is not None:
        logger.warning(
            f"Using batch key '{batch_key}' to define samples. They do not represent biological replicates, interpret results with care."
        )

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

    # Identify samples and design
    internal_group_key = "_psbulk_condition"
    conditions = ["query", "reference"]

    adata_subset, sample_hierarchy, info, design, include_batch = _identify_samples_and_design(
        adata_sub=adata_subset,
        condition_labels=condition_labels_subset,
        conditions=conditions,
        replicate_key=replicate_key,
        batch_key=batch_key,
        min_cells=min_cells,
        min_fraction=min_fraction,
        min_coverage=min_coverage,
        min_bridging_batches=min_bridging_batches,
    )

    condition_totals = {
        "query": np.count_nonzero(condition_labels_subset == "query"),
        "reference": np.count_nonzero(condition_labels_subset == "reference"),
    }

    if compute_sample_stats:
        sample_stats = _compute_sample_stats(
            sample_hierarchy=sample_hierarchy,
            valid_samples_by_condition=info["valid_samples_by_condition"],
            condition_totals=condition_totals,
            original_samples_by_condition=info["original_samples_by_condition"],
            collapsed_conditions=info["collapsed_conditions"],
        )
    else:
        sample_stats = None

    counts, metadata = aggregate_counts(
        adata_sub=adata_subset,
        sample_hierarchy=sample_hierarchy,
        layer=layer,
        mode=mode,
        return_metadata=True,
        group_key=internal_group_key,
    )

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
        original_samples_by_condition=info["original_samples_by_condition"],
        collapsed_conditions=info["collapsed_conditions"],
        condition_totals=condition_totals,
        replicate_key=info["replicate_key"],
        batch_key=info["batch_key"],
        replicate_min_cells=min_cells,
        replicate_min_fraction=min_fraction,
        sample_hierarchy=sample_hierarchy,
        adata_subset=adata_subset,
        include_batch=include_batch,
        layer=layer,
        mode=mode,
    )


@performance(logger=logger)
def _identify_samples_and_design(
    adata_sub,
    condition_labels: np.ndarray,
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
    original_samples_by_condition = {}
    collapsed_conditions = []
    valid_mask = np.ones(n_cells, dtype=bool)

    # Info about deduplication (only relevant when replicate_key and batch_key are provided)
    dedup_info = {}

    for cond in conditions:
        indices = np.where(condition_labels == cond)[0]
        cond_sample_ids = sample_ids[indices]
        cond_obs_names = obs_index[indices]
        # store original unique sample ids for this condition
        original_samples_by_condition[cond] = list(np.unique(cond_sample_ids))

        # Count samples and determine initial validity
        unique_samples, sample_counts = np.unique(cond_sample_ids, return_counts=True)
        fractions = sample_counts / len(indices)
        valid_mask_samples = (sample_counts >= min_cells) | (fractions >= min_fraction)

        # If both replicate and batch keys provided, ensure at most one sample per replicate:
        # for replicates that appear in multiple batches, keep the sample (batch) with largest cell count.
        if replicate_key and batch_key:
            # dedup mapping for this condition
            dedup_info_cond = {}
            # extract replicate part from each unique sample id
            # Assumes sample format condition_replicate_batch (same as construction above)
            sample_replicates = []
            for s in unique_samples:
                parts = s.split("_")
                # fallback if unexpected format
                rep = parts[1] if len(parts) > 1 else ""
                sample_replicates.append(rep)
            sample_replicates = np.array(sample_replicates, dtype=object)

            for rep in np.unique(sample_replicates):
                idxs = np.where(sample_replicates == rep)[0]
                if len(idxs) > 1:
                    # choose the sample index (batch) with the largest cell count to keep
                    counts_for_idxs = sample_counts[idxs]
                    keep_local_idx = idxs[np.argmax(counts_for_idxs)]
                    dropped_idxs = [int(i) for i in idxs if int(i) != int(keep_local_idx)]
                    # mark the dropped samples as invalid
                    for di in dropped_idxs:
                        valid_mask_samples[di] = False
                    dedup_info_cond[rep] = {
                        "kept": unique_samples[int(keep_local_idx)],
                        "dropped": [unique_samples[int(i)] for i in dropped_idxs],
                    }
            if dedup_info_cond:
                dedup_info[cond] = dedup_info_cond

        # After deduplication, check coverage requirement
        if valid_mask_samples.any():
            coverage = sample_counts[valid_mask_samples].sum() / len(indices)
            if coverage < min_coverage:
                valid_mask_samples[:] = False

        # If no valid samples remain for this condition, collapse all cells in this condition into one sample
        if not valid_mask_samples.any():
            collapsed_id = f"{cond}_psbulk-no-replicate_psbulk-no-batch"
            sample_ids[indices] = collapsed_id
            cond_sample_ids = sample_ids[indices]
            unique_samples = np.array([collapsed_id], dtype=object)
            sample_counts = np.array([len(indices)], dtype=int)
            valid_mask_samples = np.array([True])
            collapsed_conditions.append(cond)
            # update dedup info to reflect collapse (if present)
            if cond in dedup_info:
                dedup_info[cond]["collapsed_due_to_no_valid_samples"] = True

        valid_samples = unique_samples[valid_mask_samples]
        valid_samples_by_condition[cond] = list(valid_samples)

        # Mark valid cells
        is_valid = np.isin(cond_sample_ids, valid_samples)
        valid_mask[indices] = is_valid

        # Build hierarchy: condition -> replicate -> batch -> cell names
        sample_hierarchy[cond] = {}
        for sample_name in valid_samples:
            mask = cond_sample_ids == sample_name
            sample_parts = sample_name.split("_")
            sample_replicate = sample_parts[1] if len(sample_parts) > 1 else "psbulk-no-replicate"
            sample_batch = sample_parts[2] if len(sample_parts) > 2 else "psbulk-no-batch"
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

    design = "~_psbulk_condition+psbulk_batch" if include_batch else "~_psbulk_condition"

    info = {
        "valid_samples_by_condition": valid_samples_by_condition,
        "original_samples_by_condition": original_samples_by_condition,
        "collapsed_conditions": collapsed_conditions,
        "replicate_key": replicate_key,
        "batch_key": batch_key,
        "bridging_batches": bridging_batches,
        "deduplicated_replicates": dedup_info,
    }

    logger.debug(f"Design formula: {design}")
    logger.debug(f"Valid samples: {valid_samples_by_condition}")
    if dedup_info:
        logger.info(f"Deduplicated replicates (kept/dropped): {dedup_info}")

    if collapsed_conditions:
        logger.warning(f"Collapsed conditions (insufficient samples): {collapsed_conditions}")

    return adata_sub, sample_hierarchy, info, design, include_batch


# @performance(logger=logger)
# def _identify_samples_and_design(
#     adata_sub,
#     condition_labels: np.ndarray,
#     conditions: list[str],
#     replicate_key: str | None,
#     batch_key: str | None,
#     min_cells: int,
#     min_fraction: float,
#     min_coverage: float,
#     min_bridging_batches: int,
# ):
#     """
#     Identify valid samples, build design formula, and detect collapsing.

#     A "sample" is the minimal observational unit over which cell counts are aggregated prior to differential analysis.
#     Its definition depends on which keys are provided:
#       - replicate_key and batch_key: Sample = same condition + replicate + batch (ex: 'disease_mouse1_runA')
#       - replicate_key only: Sample = same condition + replicate (ex: 'disease_mouse1')
#       - batch_key only: Sample = same condition + batch (ex: 'disease_runA')
#       - neither: All cells in a condition are collapsed to one sample (ex: 'disease_collapsed')
#     """
#     obs = adata_sub.obs
#     obs_index = np.asarray(obs.index)
#     n_cells = adata_sub.n_obs

#     # Prepare keys and arrays
#     batch_vals = (
#         np.asarray(obs[batch_key]).astype(str, copy=False)
#         if batch_key
#         else np.full(n_cells, "psbulk-no-batch", dtype=object)
#     )
#     replicate_vals = (
#         np.asarray(obs[replicate_key]).astype(str, copy=False)
#         if replicate_key
#         else np.full(n_cells, "psbulk-no-replicate", dtype=object)
#     )
#     condition_str = condition_labels.astype(str)

#     # sample_ids must be dtype=object.
#     # NumPy fixed-width string arrays (<U*) silently truncate on assignment;
#     # collapsing conditions assigns longer IDs later, which would otherwise be cut off.
#     sample_ids = np.array(
#         condition_str + "_" + replicate_vals + "_" + batch_vals,
#         dtype=object,
#     )

#     sample_hierarchy = {}
#     valid_samples_by_condition = {}
#     original_samples_by_condition = {}
#     collapsed_conditions = []
#     valid_mask = np.ones(n_cells, dtype=bool)

#     original_samples_by_condition = {}

#     for cond in conditions:
#         indices = np.where(condition_labels == cond)[0]
#         cond_sample_ids = sample_ids[indices]
#         cond_obs_names = obs_index[indices]
#         original_samples_by_condition[cond] = cond_sample_ids.copy()
#         original_samples_by_condition[cond] = list(np.unique(cond_sample_ids))

#         # Count samples and determine validity
#         unique_samples, sample_counts = np.unique(cond_sample_ids, return_counts=True)
#         fractions = sample_counts / len(indices)
#         valid_mask_samples = (sample_counts >= min_cells) | (fractions >= min_fraction)

#         if valid_mask_samples.any():
#             coverage = sample_counts[valid_mask_samples].sum() / len(indices)
#             if coverage < min_coverage:
#                 valid_mask_samples[:] = False

#         if not valid_mask_samples.any():
#             collapsed_id = f"{cond}_psbulk-no-replicate_psbulk-no-batch"
#             sample_ids[indices] = collapsed_id
#             cond_sample_ids = sample_ids[indices]
#             unique_samples = np.array([collapsed_id], dtype=object)
#             valid_mask_samples = np.array([True])
#             collapsed_conditions.append(cond)

#         valid_samples = unique_samples[valid_mask_samples]
#         valid_samples_by_condition[cond] = list(valid_samples)

#         # Mark valid cells
#         is_valid = np.isin(cond_sample_ids, valid_samples)
#         valid_mask[indices] = is_valid

#         # Build hierarchy: condition -> replicate -> batch -> cell names
#         sample_hierarchy[cond] = {}
#         for sample_name in valid_samples:
#             mask = cond_sample_ids == sample_name
#             sample_parts = sample_name.split("_")
#             sample_replicate, sample_batch = sample_parts[1], sample_parts[2]
#             sample_obs_names = cond_obs_names[mask]
#             rep_dict = sample_hierarchy[cond].setdefault(sample_replicate, {})
#             rep_dict.setdefault(sample_batch, []).extend(sample_obs_names)

#     # Filter arrays by valid_mask
#     adata_sub = adata_sub[valid_mask]

#     # Decide design formula and batch inclusion
#     include_batch = False
#     bridging_batches = []
#     if batch_key and not any(c in collapsed_conditions for c in conditions):
#         batches_per_cond = {
#             cond: {
#                 batch
#                 for samples in sample_hierarchy[cond].values()
#                 for batch in samples.keys()
#                 if batch != "psbulk-no-batch"
#             }
#             for cond in conditions
#         }
#         all_batch_sets = [batches for batches in batches_per_cond.values() if batches]
#         if len(all_batch_sets) == len(conditions):
#             bridging_batches = list(set.intersection(*all_batch_sets))
#             include_batch = len(bridging_batches) >= min_bridging_batches

#     design = "~_psbulk_condition+psbulk_batch" if include_batch else "~_psbulk_condition"

#     info = {
#         "valid_samples_by_condition": valid_samples_by_condition,
#         "original_samples_by_condition": original_samples_by_condition,
#         "collapsed_conditions": collapsed_conditions,
#         "replicate_key": replicate_key,
#         "batch_key": batch_key,
#         "bridging_batches": bridging_batches,
#     }

#     logger.debug(f"Design formula: {design}")
#     logger.debug(f"Valid samples: {valid_samples_by_condition}")

#     if collapsed_conditions:
#         logger.warning(f"Collapsed conditions (insufficient samples): {collapsed_conditions}")

#     return adata_sub, sample_hierarchy, info, design, include_batch


@performance(logger=logger)
def _compute_sample_stats(
    sample_hierarchy: dict,
    valid_samples_by_condition: dict[str, list[str]],
    condition_totals: dict[str, int],
    *,
    original_samples_by_condition: dict[str, np.ndarray] = None,
    collapsed_conditions: list[str] = None,
) -> pd.DataFrame:
    """Compute per-sample statistics for all sample candidates."""
    import pandas as pd

    rows = []
    collapsed_conditions = collapsed_conditions or []

    for cond in sample_hierarchy.keys():
        candidates = {}

        sample_ids = np.asarray(original_samples_by_condition[cond])
        unique_samples, sample_counts = np.unique(sample_ids, return_counts=True)
        for sample_id, n_cells in zip(unique_samples, sample_counts, strict=True):
            candidates[sample_id] = n_cells

        valid_set = set(valid_samples_by_condition.get(cond, []))
        cond_total = condition_totals.get(cond, 0)
        for sample_id, n_cells in candidates.items():
            is_valid = sample_id in valid_set
            rep, batch = sample_id.split("_", 2)[1:]
            rows.append(
                {
                    "condition": cond,
                    "replicate": rep,
                    "batch": batch,
                    "_psbulk_sample": sample_id,
                    "n_cells": n_cells,
                    "fraction": n_cells / cond_total if cond_total else 0.0,
                    "is_valid": is_valid,
                }
            )

        # If condition is collapsed, add the collapsed sample explicitly
        if cond in (collapsed_conditions or []):
            collapsed_id = f"{cond}_psbulk-no-replicate_psbulk-no-batch"
            rows.append(
                {
                    "condition": cond,
                    "replicate": "psbulk-no-replicate",
                    "batch": "psbulk-no-batch",
                    "_psbulk_sample": collapsed_id,
                    "n_cells": cond_total,
                    "fraction": 1.0 if cond_total else 0.0,
                    "is_valid": True,
                }
            )
    return pd.DataFrame(rows)
