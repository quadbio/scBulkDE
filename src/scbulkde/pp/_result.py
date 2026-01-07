"""Pseudobulk result container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class PseudobulkResult:
    """Container for pseudobulked data.

    Attributes
    ----------
    counts
        Pseudobulked counts (samples x genes).
    metadata
        Sample metadata with condition, replicate, batch info.
    design
        Design formula for DE (e.g., "~condition" or "~condition+batch").
    contrast
        Contrast as [factor, query, reference].
    query
        Query condition name.
    reference
        Reference condition name(s).
    sample_stats
        Per-sample statistics with columns:
        - condition: condition group
        - batch: batch group
        - psbulk_sample: pseudobulk sample name
        - n_cells: number of cells in sample
        - fraction: n_cells / condition_total
        - is_valid: whether sample passed filtering
    valid_samples_by_condition
        Dict mapping condition to list of valid sample names.
    collapsed_conditions
        List of conditions that were collapsed due to insufficient replicates.
    condition_totals
        Dict mapping condition to total cell count.
    used_replicate_key
        Replicate key used (original or auto-generated).
    used_batch_key
        Batch key used (original or auto-generated).
    replicate_min_cells
        Minimum cells threshold used for filtering.
    replicate_min_fraction
        Minimum fraction threshold used for filtering.
    sample_hierarchy
        Nested dict: {condition -> sample_id -> batch_id -> [cell_indices]}.
        Used for pseudoreplicate generation.
    adata_subset
        Subsetted AnnData containing only valid cells.
        Used for pseudoreplicate generation.
    include_batch
        Whether batch is included in design formula.
    layer
        Layer used for counts.
    mode
        Aggregation mode used ("sum", "mean", "median").
    """

    counts: pd.DataFrame
    metadata: pd.DataFrame
    design: str
    contrast: list[str]
    query: str
    reference: str | list[str]
    sample_stats: pd.DataFrame
    valid_samples_by_condition: dict[str, list[str]]
    collapsed_conditions: list[str]
    condition_totals: dict[str, int]
    replicate_key: str
    batch_key: str
    replicate_min_cells: int
    replicate_min_fraction: float
    sample_hierarchy: dict[str, dict[str, dict[str, list]]]
    adata_subset: object  # ad.AnnData, but avoid import for dataclass
    include_batch: bool
    layer: str | None
    mode: str

    @property
    def n_obs(self) -> int:
        """Number of valid samples."""
        return self.counts.shape[0]

    @property
    def n_vars(self) -> int:
        """Number of genes."""
        return self.counts.shape[1]

    def __repr__(self) -> str:
        n_valid = sum(len(v) for v in self.valid_samples_by_condition.values())
        n_total = len(self.sample_stats)
        return (
            f"PseudobulkResult(\n"
            f"  n_samples={n_valid}/{n_total} (valid/total),\n"
            f"  n_genes={self.n_obs},\n"
            f"  design='{self.design}',\n"
            f"  query='{self.query}',\n"
            f"  reference='{self.reference}',\n"
            f"  collapsed_conditions={self.collapsed_conditions}\n"
            f")"
        )

    def summary(self) -> pd.DataFrame:
        """Return sample statistics for QC inspection.

        Returns
        -------
        pd.DataFrame
            Sample statistics with n_cells, fraction, and validity per sample.
        """
        return self.sample_stats.copy()
