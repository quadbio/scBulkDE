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
        Batch key used (original or auto-generated "_batch_1").
    replicate_min_cells
        Minimum cells threshold used for filtering.
    replicate_min_fraction
        Minimum fraction threshold used for filtering.
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
    used_replicate_key: str
    used_batch_key: str
    replicate_min_cells: int
    replicate_min_fraction: float

    @property
    def n_samples(self) -> int:
        """Number of valid samples."""
        return self.counts.shape[0]

    @property
    def n_genes(self) -> int:
        """Number of genes."""
        return self.counts.shape[1]

    def __repr__(self) -> str:
        n_valid = sum(len(v) for v in self.valid_samples_by_condition.values())
        n_total = len(self.sample_stats)
        return (
            f"PseudobulkResult(\n"
            f" n_samples={n_valid}/{n_total} (valid/total),\n"
            f" n_genes={self.n_genes},\n"
            f" design='{self.design}',\n"
            f" query='{self.query}',\n"
            f" reference='{self.reference}',\n"
            f" collapsed_conditions={self.collapsed_conditions}\n"
            f")"
        )

    def summary(self) -> pd.DataFrame:
        """Return sample statistics for QC inspection.

        Returns
        -------
        pd. DataFrame
            Sample statistics with n_cells, fraction, and validity per sample.
        """
        return self.sample_stats.copy()
