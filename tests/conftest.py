"""Shared fixtures for scbulkde tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp


@pytest.fixture
def make_adata():
    """Factory fixture to create AnnData objects with specified parameters."""

    def _make_adata(
        n_cells: int = 100,
        n_genes: int = 50,
        group_key: str = "cell_type",
        groups: list[str] | None = None,
        group_counts: list[int] | None = None,
        replicate_key: str | None = None,
        replicate_values: list | None = None,
        categorical_covariates: dict[str, list] | None = None,
        continuous_covariates: dict[str, list] | None = None,
        sparse: bool = False,
        layer_name: str | None = None,
        categorical_group_key: bool = True,
    ):
        """Create a test AnnData object.

        Parameters
        ----------
        n_cells : int
            Number of cells.
        n_genes : int
            Number of genes.
        group_key : str
            Column name for group labels.
        groups : list[str]
            Group names. If None, defaults to ['A', 'B', 'C'].
        group_counts : list[int]
            Number of cells per group. If None, distributes evenly.
        replicate_key : str
            Column name for replicate labels.
        replicate_values : list
            Replicate values per cell. If None and replicate_key given, auto-generates.
        categorical_covariates : dict
            Additional categorical covariates {name: values_per_cell}.
        continuous_covariates : dict
            Additional continuous covariates {name: values_per_cell}.
        sparse : bool
            Whether to use sparse matrix for X.
        layer_name : str
            If provided, also create a layer with different values.
        categorical_group_key : bool
            Whether to make group_key categorical dtype.
        """
        import anndata as ad

        if groups is None:
            groups = ["A", "B", "C"]

        if group_counts is None:
            # Distribute cells evenly across groups
            base_count = n_cells // len(groups)
            remainder = n_cells % len(groups)
            group_counts = [base_count + (1 if i < remainder else 0) for i in range(len(groups))]

        assert sum(group_counts) == n_cells, "group_counts must sum to n_cells"

        # Create group labels
        group_labels = []
        for g, count in zip(groups, group_counts, strict=True):
            group_labels.extend([g] * count)

        # Create expression matrix
        rng = np.random.default_rng(42)
        X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

        if sparse:
            X = sp.csr_matrix(X)

        # Create obs DataFrame
        obs_data = {group_key: group_labels}

        if categorical_group_key:
            obs_data[group_key] = pd.Categorical(group_labels)

        if replicate_key is not None:
            if replicate_values is None:
                # Auto-generate replicate values
                replicate_values = [f"rep_{i % 3}" for i in range(n_cells)]
            obs_data[replicate_key] = replicate_values

        if categorical_covariates:
            for name, values in categorical_covariates.items():
                obs_data[name] = values

        if continuous_covariates:
            for name, values in continuous_covariates.items():
                obs_data[name] = values

        obs = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])

        # Create var DataFrame
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        adata = ad.AnnData(X=X, obs=obs, var=var)

        if layer_name is not None:
            layer_X = rng.poisson(10, size=(n_cells, n_genes)).astype(np.float32)
            if sparse:
                layer_X = sp.csr_matrix(layer_X)
            adata.layers[layer_name] = layer_X

        return adata

    return _make_adata


@pytest.fixture
def simple_adata(make_adata):
    """Simple AnnData with 3 groups and replicates."""
    return make_adata(
        n_cells=120,
        n_genes=20,
        groups=["A", "B", "C"],
        group_counts=[40, 40, 40],
        replicate_key="batch",
    )
