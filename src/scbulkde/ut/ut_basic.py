from __future__ import annotations

from typing import TYPE_CHECKING

from scbulkde.ut._logging import logger
from scbulkde.ut._performance import performance

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal

    import numpy as np
    from anndata import AnnData
    from numpy.typing import NDArray


@performance(logger=logger)
def aggregate_counts(
    adata_sub, sample_hierarchy, layer, mode, *, return_metadata: bool = False, group_key: str | None = None
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

    metadata = pd.DataFrame(meta_dict, index=sample_ids)

    return counts, metadata


def _get_X_and_var_names(
    adata: AnnData,
    *,
    use_raw: bool,
    layer: str | None,
    mask_var: np.ndarray | None,
):
    """
    Get data matrix X and variable names from AnnData

    This function reimplements logic from:
    https://github.com/scverse/scanpy/blob/cf8b46dea735c35a629abfaa2e1bab9047281e34/src/scanpy/tools/_rank_genes_groups.py#L159-L181
    """
    adata_comp = adata

    if layer is not None:
        if use_raw:
            raise ValueError("Cannot specify `layer` and `use_raw=True`.")
        X = adata.layers[layer]
    elif use_raw and adata.raw is not None:
        adata_comp = adata.raw
        X = adata.raw.X
    else:
        X = adata.X

    if mask_var is not None:
        X = X[:, mask_var]
        var_names = adata_comp.var_names[mask_var]
    else:
        var_names = adata_comp.var_names

    return X, var_names


def _select_top_n(scores: NDArray, n_top: int):
    """Select indices of top n_top scores."""
    import numpy as np

    n_from = scores.shape[0]
    reference_indices = np.arange(n_from, dtype=int)
    partition = np.argpartition(scores, -n_top)[-n_top:]
    partial_indices = np.argsort(scores[partition])[::-1]
    global_indices = reference_indices[partition][partial_indices]

    return global_indices


def _select_groups(
    adata: AnnData,
    groups_order_subset: Iterable[str] | Literal["all"] = "all",
    key: str = "groups",
) -> tuple[list[str], NDArray[np.bool_]]:
    """
    Get subset of groups in adata.obs[key].

    This is an exact copy of the select_groups function from scanpy:
    https://github.com/scverse/scanpy/blob/cf8b46dea735c35a629abfaa2e1bab9047281e34/src/scanpy/_utils/__init__.py#L839-L886
    In line 875 the logger was replaced with a simple print statement.
    """
    import numpy as np

    groups_order = adata.obs[key].cat.categories
    if f"{key}_masks" in adata.uns:
        groups_masks_obs = adata.uns[f"{key}_masks"]
    else:
        groups_masks_obs = np.zeros((len(adata.obs[key].cat.categories), adata.obs[key].values.size), dtype=bool)
        for iname, name in enumerate(adata.obs[key].cat.categories):
            # if the name is not found, fallback to index retrieval
            if name in adata.obs[key].values:
                mask_obs = name == adata.obs[key].values
            else:
                mask_obs = str(iname) == adata.obs[key].values
            groups_masks_obs[iname] = mask_obs
    groups_ids = list(range(len(groups_order)))
    if groups_order_subset != "all":
        groups_ids = []
        for name in groups_order_subset:
            groups_ids.append(np.where(adata.obs[key].cat.categories.values == name)[0][0])
        if len(groups_ids) == 0:
            # fallback to index retrieval
            groups_ids = np.where(
                np.isin(
                    np.arange(len(adata.obs[key].cat.categories)).astype(str),
                    np.array(groups_order_subset),
                )
            )[0]
        if len(groups_ids) == 0:
            print(
                f"{np.array(groups_order_subset)} invalid! specify valid "
                f"groups_order (or indices) from {adata.obs[key].cat.categories}",
            )
            from sys import exit

            exit(0)
        groups_masks_obs = groups_masks_obs[groups_ids]
        groups_order_subset = adata.obs[key].cat.categories[groups_ids].values
    else:
        groups_order_subset = groups_order.values
    return groups_order_subset, groups_masks_obs


def _fraction_expressing(X, mask_obs):
    """Compute fraction of cells expressing each gene in X for given mask."""
    from scipy import sparse

    if sparse.issparse(X):
        sub = X[mask_obs]
        return sub.getnnz(axis=0) / max(1, sub.shape[0])
    else:
        sub = X[mask_obs]
        return np.count_nonzero(sub, axis=0) / max(1, sub.shape[0])
