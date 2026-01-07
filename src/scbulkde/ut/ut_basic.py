from __future__ import annotations

from scbulkde.ut._logging import logger
from scbulkde.ut._performance import performance


@performance(logger=logger)
def aggregate_counts(
    adata_sub,
    sample_hierarchy,
    layer,
    mode,
    *,
    return_metadata: bool = False,
    group_key: str | None = None,
    batch_key: str | None = None,
):
    """Aggregate counts directly (sum or mean); metadata optional."""
    import numpy as np
    import pandas as pd

    X = adata_sub.layers[layer] if layer is not None else adata_sub.X
    is_sparse = hasattr(X, "tocsr")

    # Ensure fast row slicing
    if is_sparse:
        X = X.tocsr()

    var_names = np.asarray(adata_sub.var_names)
    obs_names = np.asarray(adata_sub.obs_names)
    obs_to_pos = {name: i for i, name in enumerate(obs_names)}

    sample_ids = []
    all_positions = []

    if return_metadata:
        if group_key is None:
            raise ValueError("group_key must be provided when return_metadata=True")
        meta_conditions = []
        meta_replicates = []
        meta_batches = []

    # Traverse: condition -> replicate -> batch -> cell_names
    for cond in ("query", "reference"):
        for rep, batches in sample_hierarchy.get(cond, {}).items():
            for batch, cell_names in batches.items():
                positions = np.array(
                    [obs_to_pos[name] for name in cell_names if name in obs_to_pos],
                    dtype=np.int64,
                )

                if positions.size == 0:
                    continue

                sample_ids.append(f"{cond}_{rep}_{batch}")
                all_positions.append(positions)

                if return_metadata:
                    meta_conditions.append(cond)
                    meta_replicates.append(rep)
                    meta_batches.append(batch)

    n_samples = len(sample_ids)
    n_genes = len(var_names)
    counts_array = np.zeros((n_samples, n_genes), dtype=np.float64)

    if mode == "sum":
        for i, positions in enumerate(all_positions):
            counts_array[i, :] = np.asarray(X[positions, :].sum(axis=0)).ravel()

    elif mode == "mean":
        for i, positions in enumerate(all_positions):
            counts_array[i, :] = np.asarray(X[positions, :].sum(axis=0)).ravel() / positions.size

    else:
        raise ValueError("mode must be 'sum' or 'mean'")

    counts = pd.DataFrame(counts_array, index=sample_ids, columns=var_names)

    if not return_metadata:
        return counts

    meta_dict = {
        group_key: meta_conditions,
        "psbulk_replicate": meta_replicates,
        "psbulk_batch": meta_batches,
    }
    if batch_key is not None:
        meta_dict[batch_key] = meta_batches

    metadata = pd.DataFrame(meta_dict, index=sample_ids)

    return counts, metadata
