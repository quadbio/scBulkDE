from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

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
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        design_matrix: pd.DataFrame,
        design_formula: str,
        alpha: float,
        correction_method: str,
        **kwargs,
    ) -> pd.DataFrame:
        """Run differential expression analysis.

        Parameters
        ----------
        counts
            Gene expression counts (samples x genes).
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
        **kwargs
            Additional engine-specific parameters.
        """
        pass
