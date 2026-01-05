"""Input validation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import anndata as ad


def validate_adata(
    adata: ad.AnnData,
    layer: str,
    group_key: str,
    replicate_key: str | None = None,
    batch_key: str | None = None,
) -> None:
    """Validate AnnData object for DE analysis.

    Parameters
    ----------
    adata
        Annotated data matrix.
    layer
        Layer name that should contain counts.
    group_key
        Column name for condition groups.
    replicate_key
        Column name for biological replicates.
    batch_key
        Column name for batch information.

    Raises
    ------
    ValueError
        If any validation check fails.
    """
    if layer != "X" and layer not in adata.layers:
        available = list(adata.layers.keys())
        raise ValueError(
            f"Layer '{layer}' not found in adata.layers.\n"
            f"Available layers: {available if available else 'None (only .X)'}\n"
            f"Hint: If counts are in .X, use layer='X' or run: "
            f"adata.layers['counts'] = adata.X.copy()"
        )

    _validate_obs_key(adata, group_key, "group_key")

    if replicate_key is not None:
        _validate_obs_key(adata, replicate_key, "replicate_key")

    if batch_key is not None:
        _validate_obs_key(adata, batch_key, "batch_key")


def _validate_obs_key(adata: ad.AnnData, key: str, param_name: str) -> None:
    """Validate that a key exists in adata.obs."""
    if key not in adata.obs.columns:
        available = list(adata.obs.columns)
        raise ValueError(f"{param_name}='{key}' not found in adata.obs.\nAvailable columns: {available}")


def validate_groups(
    adata: ad.AnnData,
    group_key: str,
    query: str,
    reference: str | Sequence[str],
) -> None:
    """Validate that query and reference groups exist.

    Parameters
    ----------
    adata
        Annotated data matrix.
    group_key
        Column name containing group labels.
    query
        Query group value.
    reference
        Reference group value(s), or "rest".

    Raises
    ------
    ValueError
        If query or reference groups not found.
    """
    available_groups = adata.obs[group_key].unique().tolist()

    if query not in available_groups:
        raise ValueError(
            f"Query group '{query}' not found in adata.obs['{group_key}'].\nAvailable groups: {available_groups}"
        )

    if reference != "rest":
        ref_list = [reference] if isinstance(reference, str) else list(reference)
        missing = [r for r in ref_list if r not in available_groups]
        if missing:
            raise ValueError(
                f"Reference group(s) {missing} not found in adata.obs['{group_key}'].\n"
                f"Available groups: {available_groups}"
            )

        if query in ref_list:
            raise ValueError(f"Query '{query}' cannot also be in reference groups.")
