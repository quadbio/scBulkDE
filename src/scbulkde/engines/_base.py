"""Base class for DE engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class BaseEngine(ABC):
    """Abstract base class for differential expression engines.

    Subclass this to implement new DE backends.
    """

    name: str = "base"

    @abstractmethod
    def run(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        design: str,
        contrast: list[str],
        **kwargs,
    ) -> pd.DataFrame:
        """Run differential expression analysis.

        Parameters
        ----------
        counts
            Gene expression counts (samples x genes).
        metadata
            Sample metadata with design variables.
        design
            Design formula (e. g., "~condition" or "~condition+batch").
        contrast
            Contrast as [factor, query, reference].
            E.g., ["condition", "treated", "control"] compares treated vs control.
        **kwargs
            Engine-specific parameters.

        Returns
        -------
        pd.DataFrame
            Results with columns: log2FoldChange, pvalue, padj, baseMean.
        """
        pass
