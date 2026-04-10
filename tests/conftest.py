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
        seed: int = 42,
    ):
        import anndata as ad

        if groups is None:
            groups = ["A", "B", "C"]

        if group_counts is None:
            base_count = n_cells // len(groups)
            remainder = n_cells % len(groups)
            group_counts = [base_count + (1 if i < remainder else 0) for i in range(len(groups))]

        assert sum(group_counts) == n_cells, "group_counts must sum to n_cells"

        group_labels = []
        for g, count in zip(groups, group_counts, strict=True):
            group_labels.extend([g] * count)

        rng = np.random.default_rng(seed)
        X = rng.poisson(5, size=(n_cells, n_genes)).astype(np.float32)

        if sparse:
            X = sp.csr_matrix(X)

        obs_data = {group_key: group_labels}

        if categorical_group_key:
            obs_data[group_key] = pd.Categorical(group_labels)

        if replicate_key is not None:
            if replicate_values is None:
                replicate_values = [f"rep_{i % 3}" for i in range(n_cells)]
            obs_data[replicate_key] = replicate_values

        if categorical_covariates:
            for name, values in categorical_covariates.items():
                obs_data[name] = values

        if continuous_covariates:
            for name, values in continuous_covariates.items():
                obs_data[name] = values

        obs = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
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
def make_obs():
    """Factory to create obs DataFrames for testing."""

    def _make_obs(
        n_query: int = 100,
        n_reference: int = 100,
        condition_col: str = "psbulk_condition",
        strata: dict[str, dict[str, list]] | None = None,
    ):
        n_total = n_query + n_reference
        data = {condition_col: ["query"] * n_query + ["reference"] * n_reference}

        if strata:
            for col, values in strata.items():
                data[col] = values["query"] + values["reference"]

        return pd.DataFrame(data, index=[f"cell_{i}" for i in range(n_total)])

    return _make_obs


@pytest.fixture
def make_cell_pool():
    """Factory fixture to create cell pool cache and usage tracker for pseudoreplicate tests."""

    def _make_cell_pool(n_cells_per_sample: int = 20):
        cell_pool = {
            "query": [
                (n_cells_per_sample, np.arange(0, n_cells_per_sample)),
                (n_cells_per_sample, np.arange(n_cells_per_sample, 2 * n_cells_per_sample)),
                (n_cells_per_sample, np.arange(2 * n_cells_per_sample, 3 * n_cells_per_sample)),
            ],
            "reference": [
                (n_cells_per_sample, np.arange(3 * n_cells_per_sample, 4 * n_cells_per_sample)),
                (n_cells_per_sample, np.arange(4 * n_cells_per_sample, 5 * n_cells_per_sample)),
            ],
        }
        cell_usage = dict.fromkeys(range(5 * n_cells_per_sample), 0)
        return cell_pool, cell_usage

    return _make_cell_pool


@pytest.fixture
def make_mock_engine():
    """Factory fixture to create a configurable MockEngine for DE tests.

    The returned factory accepts an optional ``capture`` list. When provided,
    each call to ``run()`` appends the full kwargs dict to that list, enabling
    per-test assertions on what the engine received.
    """

    def _make_mock_engine(capture: list | None = None):
        class MockEngine:
            name = "mock"

            def run(
                self,
                counts,
                metadata,
                design_matrix,
                design_formula,
                alpha,
                correction_method,
                gene_names=None,
                **kwargs,
            ):
                if gene_names is not None:
                    genes = gene_names
                elif hasattr(counts, "columns"):
                    genes = counts.columns
                else:
                    raise ValueError("Cannot determine gene names")

                if capture is not None:
                    capture.append(
                        {
                            "counts": counts,
                            "metadata": metadata,
                            "design_matrix": design_matrix,
                            "design_formula": design_formula,
                            "gene_names": gene_names,
                            **kwargs,
                        }
                    )

                n_genes = len(genes)
                rng = np.random.RandomState(42)
                return pd.DataFrame(
                    {
                        "pvalue": rng.uniform(0, 0.1, n_genes),
                        "padj": rng.uniform(0, 0.1, n_genes),
                        "stat": rng.uniform(1, 5, n_genes),
                        "log2FoldChange": rng.uniform(-2, 2, n_genes),
                        "stat_sign": rng.uniform(1, 5, n_genes),
                    },
                    index=genes,
                )

        return MockEngine()

    return _make_mock_engine


@pytest.fixture
def adata_balanced(make_adata):
    """AnnData with balanced groups across conditions."""
    return make_adata(
        n_cells=100,
        n_genes=50,
        groups=["TypeA", "TypeB", "TypeC"],
        group_counts=[40, 30, 30],
        categorical_covariates={
            "batch": pd.Categorical(["batch1"] * 50 + ["batch2"] * 50),
            "donor": pd.Categorical(["donor1"] * 25 + ["donor2"] * 25 + ["donor3"] * 25 + ["donor4"] * 25),
        },
        continuous_covariates={
            "age": np.random.uniform(20, 80, 100),
        },
        seed=42,
    )


@pytest.fixture
def adata_unbalanced(make_adata):
    """AnnData with unbalanced groups."""
    return make_adata(
        n_cells=200,
        n_genes=20,
        groups=["TypeA", "TypeB"],
        group_counts=[100, 100],
        categorical_covariates={
            "batch": pd.Categorical(["batch1"] * 90 + ["batch2"] * 10 + ["batch1"] * 10 + ["batch2"] * 90),
        },
        seed=42,
    )


@pytest.fixture
def adata_sparse(make_adata):
    """AnnData with sparse matrix."""
    return make_adata(
        n_cells=80,
        n_genes=30,
        groups=["TypeA", "TypeB"],
        group_counts=[40, 40],
        categorical_covariates={
            "batch": pd.Categorical(["batch1"] * 40 + ["batch2"] * 40),
        },
        sparse=True,
        seed=42,
    )


@pytest.fixture
def adata_confounded(make_adata):
    """AnnData where covariates are confounded."""
    return make_adata(
        n_cells=100,
        n_genes=20,
        groups=["TypeA", "TypeB"],
        group_counts=[50, 50],
        categorical_covariates={
            "batch": ["batch1"] * 50 + ["batch2"] * 50,
            "donor": ["donor1"] * 25 + ["donor2"] * 25 + ["donor3"] * 25 + ["donor4"] * 25,
        },
        seed=42,
    )


@pytest.fixture
def adata_single_stratum(make_adata):
    """AnnData where one stratum has only one sample."""
    return make_adata(
        n_cells=100,
        n_genes=20,
        groups=["TypeA", "TypeB"],
        group_counts=[90, 10],
        categorical_covariates={
            "batch": ["batch1"] * 90 + ["batch2"] * 10,
        },
        seed=42,
    )
