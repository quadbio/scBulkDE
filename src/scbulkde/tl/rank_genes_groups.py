"""Drop-in replacement for `scanpy.tl.rank_genes_groups` using pseudobulk DE."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Literal

    from anndata import AnnData

from scbulkde.ut import (
    _fraction_expressing,
    _get_X_and_var_names,
    _in_notebook,
    _select_groups,
    _select_top_n,
)

from .tl_basic import de

if _in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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
    de_kwargs,
) -> AnnData | None:
    """
    Rank genes for characterizing groups using pseudobulk differential expression.

    This is a drop-in replacement for scanpy.tl.rank_genes_groups that uses
    pseudobulk aggregation followed by differential expression testing instead
    of single-cell statistical tests. This approach is more statistically rigorous
    for single-cell RNA-seq data with biological replicates.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider (e.g., 'cell_type', 'cluster').
    mask_var
        Select subset of genes to use in statistical tests. Can be a boolean array
        or a string key from `adata.var`.
    use_raw
        Use raw attribute of adata if present. The default behavior (`None`) is to
        use raw if present. Set to `False` to force use of normalized data.
    groups
        Subset of groups, e.g. `['g1', 'g2', 'g3']`, to which comparison shall be
        restricted, or `'all'` (default), for all groups. Note that if `reference='rest'`
        all groups will still be used as the reference, not just those specified in groups.
    reference
        If `'rest'`, compare each group to the union of the rest of the groups.
        If a group identifier, compare with respect to this specific group.
        When a specific reference is provided, it will not be tested against itself.
    n_genes
        The number of genes that appear in the returned tables. Defaults to all genes.
    rankby_abs
        Rank genes by the absolute value of the log fold change, not by the log fold
        change itself. The returned scores are never the absolute values.
    pts
        Compute the fraction of cells expressing the genes in each group.
    key_added
        The key in `adata.uns` where information is saved to. Defaults to 'rank_genes_groups'.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
        Cannot be used together with `use_raw=True`.
    copy
        Whether to copy adata or modify it inplace.
    de_kwargs
        Keyword arguments passed to the underlying `de()` function. Must include
        pseudobulk-specific parameters such as `replicate_key`. Common parameters include:

        - `replicate_key` : str
            Column in `adata.obs` defining biological replicates (required for pseudobulk).
        - `min_cells` : int, default=50
            Minimum number of cells required per pseudobulk sample.
        - `min_fraction` : float, default=0.2
            Minimum fraction of cells per pseudobulk sample.
        - `min_coverage` : float, default=0.75
            Minimum coverage required per condition.
        - `categorical_covariates` : Sequence[str], optional
            Categorical covariates to include in the design.
        - `continuous_covariates` : Sequence[str], optional
            Continuous covariates to include in the design.
        - `engine` : str, default='anova'
            Statistical engine for DE testing ('anova' or 'pydeseq2').
        - `fallback_strategy` : {'pseudoreplicates', 'single_cell', None}, default='pseudoreplicates'
            Strategy when insufficient biological replicates exist.
        - `min_samples` : int, default=3
            Minimum number of pseudobulk samples required per condition for direct testing.

    Returns
    -------
    AnnData | None
        Returns `adata` if `copy=True`, otherwise returns `None` and modifies `adata` inplace.
        Results are stored in `adata.uns[key_added]` with the following structure:

        - `names` : numpy.recarray
            Structured array with top gene names for each group.
        - `scores` : numpy.recarray
            Structured array with test statistics for each group.
        - `logfoldchanges` : numpy.recarray
            Structured array with log2 fold changes for each group.
        - `pvals` : numpy.recarray
            Structured array with p-values for each group.
        - `pvals_adj` : numpy.recarray
            Structured array with adjusted p-values (FDR) for each group.
        - `pts` : pd.DataFrame (if `pts=True`)
            Fraction of cells expressing each gene in each group.
        - `pts_rest` : pd.DataFrame (if `pts=True` and `reference='rest'`)
            Fraction of cells expressing each gene in the rest of the cells.
        - `params` : dict
            Dictionary containing parameters used for the analysis.

    Notes
    -----
    Unlike scanpy.tl.rank_genes_groups which uses single-cell statistical tests
    (t-test, Wilcoxon, etc.), this implementation:

    1. Aggregates cells into pseudobulk samples based on biological replicates
    2. Performs differential expression testing on pseudobulk data
    3. Properly accounts for sample-level variation and biological replicates

    This approach is more statistically appropriate for single-cell RNA-seq data
    and reduces false discovery rates [Squair2021]_.

    When insufficient biological replicates are available, the function can fall back
    to pseudoreplicate generation or single-cell testing (controlled by `de_kwargs`).

    References
    ----------
    .. [Squair2021] Squair, J.W., et al. (2021)
       "Confronting false discoveries in single-cell differential expression."
       Nature Communications 12, 5692.

    See Also
    --------
    de : Core differential expression function
    pp.pseudobulk : Pseudobulk aggregation without DE testing

    Examples
    --------
    n.a.
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
        groups_to_test = "all"  # Will be resolved after _select_groups
    # Check that groups is a sequence and not just a single string or int
    elif isinstance(groups, str | int):
        msg = "Specify a sequence of groups"
        raise ValueError(msg)
    else:
        groups_order = list(groups)
        # If groups are integers, convert to strings
        if len(groups_order) > 0 and isinstance(groups_order[0], int):
            groups_order = [str(n) for n in groups_order]

        # Store which groups to actually test (don't test reference against itself)
        groups_to_test = groups_order.copy()

        # Ensure reference is included in groups_order for _select_groups
        # (needed for creating masks) but don't test it against itself
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
    # optionally filtered to only include variables that should be tested
    X, var_names = _get_X_and_var_names(
        adata,
        use_raw=use_raw,
        layer=layer,
        mask_var=mask_var,
    )

    # Select the groups to be tested
    adata.obs[groupby] = adata.obs[groupby].astype("category")
    groups_order, groups_masks_obs = _select_groups(adata, groups_order, groupby)

    # Now resolve groups_to_test if it was "all"
    if groups_to_test == "all":
        if reference == "rest":
            # Test all groups against rest
            groups_to_test = groups_order
        else:
            # Test all groups except the reference (don't test reference against itself)
            groups_to_test = [g for g in groups_order if g != reference]
    else:
        # groups_to_test already set above, but ensure reference is excluded if specific
        if reference != "rest":
            groups_to_test = [g for g in groups_to_test if g != reference]

    # Select number of genes to be returned
    n_vars = X.shape[1]
    if n_genes is None or n_genes > n_vars:
        n_genes_user = n_vars
    else:
        n_genes_user = n_genes

    # Do the actual DE - only test groups_to_test, not the reference
    stats = None

    for group in tqdm(groups_to_test):
        de_res = de(
            adata,
            group_key=groupby,
            query=group,
            reference=reference,
            layer=layer,
            **de_kwargs,
        )

        res = de_res.results
        res = res.loc[var_names, :]  # ensure same order of genes as in X

        scores = res["stat_sign"].to_numpy()
        pvals = res["pvalue"].to_numpy()
        padj = res["padj"].to_numpy()
        lfc = res["log2FoldChange"].to_numpy()

        # FIX LATER: using lfc here because it looks better than the test statistic
        scores_sort = np.abs(lfc) if rankby_abs else lfc
        top_idx = _select_top_n(scores_sort, n_genes_user)

        if stats is None:
            idx = pd.MultiIndex.from_tuples([(str(group), "names")])
            stats = pd.DataFrame(columns=idx)

        var_names = res.index

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
