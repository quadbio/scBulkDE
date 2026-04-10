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
    """Container for the results of a pseudobulking procedure.

    Attributes
    ----------
    adata_sub : ad.AnnData
        Subset of input AnnData containing only query and reference cells.
    pb_counts : pd.DataFrame
        Aggregated pseudobulk expression matrix (samples x genes). Empty if
        no valid strata exist (collapsed case).
    grouped : DataFrameGroupBy
        Grouped observation data used for aggregation.
    sample_table : pd.DataFrame
        Metadata for each pseudobulk sample, including covariates, cell
        counts, and quality metrics.
    design_matrix : pd.DataFrame
        Design matrix for statistical testing, created from
        ``design_formula``.
    design_formula : str
        Patsy-style formula describing the statistical model.
    group_key : str
        Column name in ``adata.obs`` used to define cell groups.
    group_key_internal : str
        Internal column name for query/reference labels
        (``'psbulk_condition'``).
    query : Sequence[str]
        Query group(s) used for the comparison.
    reference : Sequence[str]
        Reference group(s) used for the comparison.
    strata : Sequence[str]
        Final stratification factors used (may be a subset of requested
        due to conflict resolution). Empty list indicates collapsed
        pseudobulk.
    layer : str or None
        Layer in ``adata.layers`` used for aggregation, or ``None`` for
        ``adata.X``.
    layer_aggregation : str
        Method used to aggregate expression values across cells
        (``'sum'`` or ``'mean'``).
    categorical_covariates : Sequence[str] or None
        Categorical covariates included in the design.
    continuous_covariates : Sequence[str] or None
        Continuous covariates included in the design.
    continuous_aggregation : str or None
        Method used to aggregate continuous covariates per pseudobulk
        sample.
    min_cells : int or None
        Minimum number of cells required per pseudobulk sample.
    min_fraction : float or None
        Minimum fraction of condition cells required per pseudobulk
        sample.
    min_coverage : float or None
        Minimum coverage required per condition.
    qualify_strategy : str
        Strategy used for sample qualification (``'and'`` or ``'or'``).
    n_cells : dict[str, int] or None
        Number of cells per condition (``'query'`` and ``'reference'``).
    """

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

    @property
    def collapsed(self) -> bool:
        """Whether samples are collapsed (all cells used without valid strata).

        Returns True if no valid strata were found and all cells per condition
        are used as a single sample. In this case, pb_counts is empty.
        """
        if "collapsed" not in self.sample_table.columns:
            return False
        return self.sample_table["collapsed"].all()

    def __repr__(self) -> str:
        collapsed_info = ""
        if self.collapsed:
            collapsed_info = "\n  collapsed=True,"
        return (
            f"PseudobulkResult(\n"
            f"  n_samples={self.n_samples},\n"
            f"  n_genes={self.pb_counts.shape[1] if not self.pb_counts.empty else 0},\n"
            f"  strata={list(self.strata)},{collapsed_info}\n"
            f"  design_formula='{self.design_formula}'\n"
            f")"
        )


@dataclass
class DEResult:
    """Container for differential expression results.

    Attributes
    ----------
    results : pd.DataFrame
        DE results table with columns: ``log2FoldChange``, ``pvalue``,
        ``padj``, ``stat``, ``stat_sign``.
    query : str
        Query condition name.
    reference : str or list[str]
        Reference condition name(s).
    design : str
        Design formula used for testing.
    engine : str
        Name of the DE engine used (e.g. ``'anova'``, ``'pydeseq2'``).
    used_pseudoreplicates : bool
        Whether pseudoreplicates were generated to meet the minimum
        sample requirement.
    used_single_cell : bool
        Whether single-cell level testing was performed as a fallback.
    n_repetitions : int
        Number of pseudoreplicate iterations run. 1 for direct testing,
        >1 when pseudoreplicates are used.
    repetition_results : dict[str, pd.DataFrame]
        Per-repetition DE results. Only populated when pseudoreplicates
        are used.
    repetition_stats : dict[str, pd.DataFrame]
        Per-repetition sample statistics. Only populated when
        pseudoreplicates are used.
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
        """Get sample statistics for a specific repetition.

        Parameters
        ----------
        repetition : int or str
            The repetition index or key to retrieve.

        Returns
        -------
        pd.DataFrame
            Sample statistics for the requested repetition.

        Raises
        ------
        ValueError
            If no pseudoreplicates were used.
        KeyError
            If the specified repetition is not found.
        """
        if not self.used_pseudoreplicates:
            raise ValueError("No pseudoreplicates were used.")
        key = str(repetition)
        if key not in self.repetition_stats:
            raise KeyError(f"Repetition '{key}' not found.")
        return self.repetition_stats[key].copy()

    def get_repetition_results(self, repetition: int | str) -> pd.DataFrame:
        """Get DE results for a specific repetition.

        Parameters
        ----------
        repetition : int or str
            The repetition index or key to retrieve.

        Returns
        -------
        pd.DataFrame
            DE results for the requested repetition.

        Raises
        ------
        ValueError
            If no pseudoreplicates were used.
        KeyError
            If the specified repetition is not found.
        """
        if not self.used_pseudoreplicates:
            raise ValueError("No pseudoreplicates were used.")
        key = str(repetition)
        if key not in self.repetition_results:
            raise KeyError(f"Repetition '{key}' not found.")
        return self.repetition_results[key].copy()
