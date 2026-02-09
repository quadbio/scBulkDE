from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class DEEngineBase(ABC):
    """Abstract base class for differential expression engines.

    Subclass this to implement new DE backends.
    """

    name: str = "base"

    @abstractmethod
    def run(
        self,
        counts: pd.DataFrame | np.ndarray,
        metadata: pd.DataFrame,
        design_matrix: pd.DataFrame,
        design_formula: str,
        alpha: float,
        correction_method: str,
        *,
        gene_names: np.ndarray | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Run differential expression analysis.

        Parameters
        ----------
        counts
            Gene expression counts (samples x genes). Can be DataFrame or numpy array.
            When numpy array is provided, gene_names must also be provided.
        metadata
            Sample metadata with design variables.
        design_matrix
            Design matrix for the regression model.
        design_formula
            Design formula (e.g., "~condition" or "~condition+batch").
        alpha
            Significance threshold for adjusted p-values.
        correction_method
            Method for multiple testing correction.
        gene_names
            Gene names corresponding to columns of counts array.
            Required when counts is a numpy array.
        **kwargs
            Additional engine-specific parameters.
        """
        pass

    @staticmethod
    def _get_counts_and_gene_names(
        counts: pd.DataFrame | np.ndarray,
        gene_names: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract numpy array and gene names from counts input.

        Parameters
        ----------
        counts
            Either a DataFrame or numpy array of counts.
        gene_names
            Gene names if counts is a numpy array.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Counts as numpy array and gene names as numpy array.
        """
        if isinstance(counts, np.ndarray):
            if gene_names is None:
                raise ValueError("gene_names must be provided when counts is a numpy array")
            return counts, np.asarray(gene_names)
        else:
            # DataFrame - extract values and column names
            return counts.values, counts.columns.to_numpy()
