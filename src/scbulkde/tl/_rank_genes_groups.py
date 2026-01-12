"""Drop-in replacement for `scanpy.tl.rank_genes_groups` using pseudobulk DE."""

from __future__ import annotations

import inspect
from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from scbulkde.ut import (
    _fraction_expressing,
    _get_X_and_var_names,
    _select_groups,
    _select_top_n,
)

from .tl_basic import de


def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    *,
    mask_var: np.ndarray | str | None = None,
    use_raw: bool | None = None,
    groups: Literal["all"] | Iterable[str] = "all",
    reference: str = "rest",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: str | None = None,
    layer: str | None = None,
    copy: bool = False,
    **de_kwargs,
) -> AnnData | None:
    """
    Pseudobulk-backed replacement for scanpy.tl.rank_genes_groups.

    Implements the same logic and output structure as scanpy.tl.rank_genes_groups, but reduces rigorousity in order
    to avoid importing scanpy itself.
    """
    # Adhere to:
    # https://github.com/scverse/scanpy/blob/cf8b46dea735c35a629abfaa2e1bab9047281e34/src/scanpy/tools/_rank_genes_groups.py#L626C1-L630C30
    if use_raw is None:
        use_raw = adata.raw is not None
    elif use_raw is True and adata.raw is None:
        msg = "Received `use_raw=True`, but `adata.raw` is empty."
        raise ValueError(msg)

    # Make a working copy if requested. Scanpy uses _utils.sanitize_anndata here
    adata = adata.copy() if copy else adata

    # Resolve groups, rename variable for clarity and adhere to:
    # https://github.com/scverse/scanpy/blob/cf8b46dea735c35a629abfaa2e1bab9047281e34/src/scanpy/tools/_rank_genes_groups.py#L651-L665
    # If groups is "all", we keep it as is for later processing
    if groups == "all":
        groups_order = "all"
    # Check that groups is a sequence and not just a single string or int
    elif isinstance(groups, str | int):
        msg = "Specify a sequence of groups"
        raise ValueError(msg)
    else:
        groups_order = list(groups)
        # If groups are integers, convert to strings
        if isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]
        # Ensure reference is included in groups_order
        if reference != "rest" and reference not in set(groups_order):
            groups_order += [reference]
    # Check that reference is valid
    if reference != "rest" and reference not in adata.obs[groupby].cat.categories:
        cats = adata.obs[groupby].cat.categories.tolist()
        msg = f"reference = {reference} needs to be one of groupby = {cats}."
        raise ValueError(msg)

    # Key to store results in adata.uns
    if key_added is None:
        key_added = "rank_genes_groups"

    # Store parameters
    if "engine" in de_kwargs and de_kwargs["engine"] is not None:
        en = de_kwargs["engine"]
    else:
        en = inspect.signature(de).parameters["engine"].default

        # This is currently not ideal and based on that I set the default correction
        # method to benjamini hochberg in the engines
    corr_method = de_kwargs.get("corr_method", "bh")

    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "groupby": groupby,
        "reference": reference,
        "method": f"pseudobulk_{en}",
        "use_raw": use_raw,
        "layer": layer,
        "corr_method": corr_method,
    }

    # Get the data matrix and the variable names, which can be
    # optionally filtered to only include variables that are to be included in the test
    X, var_names = _get_X_and_var_names(
        adata,
        use_raw=use_raw,
        layer=layer,
        mask_var=mask_var,
    )

    # Select the groups to be tested
    adata.obs[groupby] = adata.obs[groupby].astype("category")
    groups_order, groups_masks_obs = _select_groups(adata, groups_order, groupby)

    # Select number of genes to be returned
    n_vars = X.shape[1]
    if n_genes is None or n_genes > n_vars:
        n_genes_user = n_vars
    else:
        n_genes_user = n_genes

    # Do the actual DE
    stats = None

    for group in groups_order:
        de_res = de(
            adata,
            group_key=groupby,
            query=group,
            reference=reference,
            layer=layer,
            **de_kwargs,
        )

        res = de_res.results

        scores = res["stat"].to_numpy()
        pvals = res["pvalue"].to_numpy()
        padj = res["padj"].to_numpy()
        lfc = res["log2FoldChange"].to_numpy()

        scores_sort = np.abs(scores) if rankby_abs else scores
        top_idx = _select_top_n(scores_sort, n_genes_user)

        if stats is None:
            idx = pd.MultiIndex.from_tuples([(str(group), "names")])
            stats = pd.DataFrame(columns=idx)

        stats[(str(group), "names")] = var_names[top_idx]
        stats[(str(group), "scores")] = scores[top_idx]
        stats[(str(group), "pvals")] = pvals[top_idx]
        stats[(str(group), "pvals_adj")] = padj[top_idx]
        stats[(str(group), "logfoldchanges")] = lfc[top_idx]

    stats.columns = stats.columns.swaplevel()

    # Compute the fraction of cells expressing each gene in each group
    if pts:
        pts_arr = []
        pts_rest_arr = []

        for mask_obs in groups_masks_obs:
            pts_arr.append(_fraction_expressing(X, mask_obs))
            if reference == "rest":
                pts_rest_arr.append(_fraction_expressing(X, ~mask_obs))

        groups_names = [str(g) for g in groups_order]
        adata.uns[key_added]["pts"] = pd.DataFrame(np.array(pts_arr).T, index=var_names, columns=groups_names)

        if reference == "rest":
            adata.uns[key_added]["pts_rest"] = pd.DataFrame(
                np.array(pts_rest_arr).T, index=var_names, columns=groups_names
            )

    # Format to match Scanpy style
    # https://github.com/scverse/scanpy/blob/cf8b46dea735c35a629abfaa2e1bab9047281e34/src/scanpy/tools/_rank_genes_groups.py#L727-L738
    dtypes = {
        "names": "O",
        "scores": "float32",
        "logfoldchanges": "float32",
        "pvals": "float64",
        "pvals_adj": "float64",
    }

    for col in stats.columns.levels[0]:
        adata.uns[key_added][col] = stats[col].to_records(index=False, column_dtypes=dtypes[col])

    return adata if copy else None
