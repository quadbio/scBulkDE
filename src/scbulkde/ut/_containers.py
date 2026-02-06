from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import anndata as ad
    import pandas as pd
    from pandas.core.groupby.generic import DataFrameGroupBy


@dataclass
class PseudobulkResult:
    """Container for the results of a pseudobulking procedure."""

    # Core outputs
    adata_sub: ad.AnnData
    pb_counts: pd.DataFrame
    grouped: DataFrameGroupBy
    sample_table: pd.DataFrame
    design_matrix: pd.DataFrame
    design_formula: str

    # Keys and grouping
    group_key: str
    group_key_internal: str
    query: Sequence[str]
    reference: Sequence[str]
    strata: Sequence[str]

    # Aggregation settings
    layer: str | None
    layer_aggregation: str
    categorical_covariates: Sequence[str] | None
    continuous_covariates: Sequence[str] | None
    continuous_aggregation: str | None

    # Filtering / qualification parameters
    min_cells: int | None
    min_fraction: float | None
    min_coverage: float | None
    qualify_strategy: str

    # Diagnostics
    n_cells: dict[str, int] | None = None

    @property
    def n_samples(self) -> int:
        """Number of pseudobulk samples."""
        return len(self.pb_counts)

    def __repr__(self) -> str:
        return (
            f"PseudobulkResult(\n"
            f"  n_samples={self.n_samples},\n"
            f"  n_genes={self.pb_counts.shape[1] if not self.pb_counts.empty else 0},\n"
            f"  strata={list(self.strata)},\n"
            f"  design_formula='{self.design_formula}'\n"
            f")"
        )


@dataclass
class DEResult:
    """Container for differential expression results.

    Attributes
    ----------
    results
        DE results with columns: log2FoldChange, pvalue, padj, baseMean.
    query
        Query condition name.
    reference
        Reference condition name(s).
    design
        Design formula used.
    engine
        DE engine used.
    used_pseudoreplicates
        Whether pseudoreplicates were generated.
    used_single_cell
        Whether single-cell level testing was performed.
    n_repetitions
        Number of repetitions run (1 if no pseudoreplicates).
    repetition_results
        Per-repetition DE results (only if pseudoreplicates used).
    repetition_stats
        Per-repetition sample statistics (only if pseudoreplicates used).
    """

    results: pd.DataFrame
    query: str
    reference: str | list[str]
    design: str
    engine: str
    used_pseudoreplicates: bool = False
    used_single_cell: bool = False
    n_repetitions: int = 1
    repetition_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    repetition_stats: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def n_significant(self) -> int:
        """Number of significant genes (padj < 0.05)."""
        return (self.results["padj"] < 0.05).sum()

    @property
    def n_genes(self) -> int:
        """Number of genes tested."""
        return len(self.results)

    @property
    def fallback_used(self) -> str | None:
        """Return which fallback was used, if any."""
        if self.used_pseudoreplicates:
            return "pseudoreplicates"
        if self.used_single_cell:
            return "single_cell"
        return None

    def __repr__(self) -> str:
        rep_info = ""
        if self.used_pseudoreplicates:
            rep_info = f"\n  n_repetitions={self.n_repetitions},"

        fallback_info = ""
        if self.used_single_cell:
            fallback_info = "\n  fallback='single_cell',"
        elif self.used_pseudoreplicates:
            fallback_info = "\n  fallback='pseudoreplicates',"

        return (
            f"DEResult(\n"
            f"  n_genes={self.n_genes},\n"
            f"  n_significant={self.n_significant} (padj < 0.05),\n"
            f"  design='{self.design}',\n"
            f"  query='{self.query}',\n"
            f"  reference='{self.reference}',\n"
            f"  engine='{self.engine}',{fallback_info}{rep_info}\n"
            f")"
        )

    def summary(self) -> pd.DataFrame:
        """Return the main results table."""
        return self.results.copy()

    def get_repetition_stats(self, repetition: int | str) -> pd.DataFrame:
        """Get sample statistics for a specific repetition."""
        if not self.used_pseudoreplicates:
            raise ValueError("No pseudoreplicates were used.")
        key = str(repetition)
        if key not in self.repetition_stats:
            raise KeyError(f"Repetition '{key}' not found.")
        return self.repetition_stats[key].copy()

    def get_repetition_results(self, repetition: int | str) -> pd.DataFrame:
        """Get DE results for a specific repetition."""
        if not self.used_pseudoreplicates:
            raise ValueError("No pseudoreplicates were used.")
        key = str(repetition)
        if key not in self.repetition_results:
            raise KeyError(f"Repetition '{key}' not found.")
        return self.repetition_results[key].copy()
