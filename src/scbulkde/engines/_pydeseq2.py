"""PyDESeq2 engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from ._base import DEEngineBase

if TYPE_CHECKING:
    from typing import Literal


class PyDESeq2Engine(DEEngineBase):
    """DESeq2 engine using PyDESeq2 (pure Python)."""

    name = "pydeseq2"

    _inference_cache: dict[int, DefaultInference] = {}

    @classmethod
    def _get_inference(cls, n_cpus: int) -> DefaultInference:
        """Get or create a cached DefaultInference instance."""
        if n_cpus not in cls._inference_cache:
            cls._inference_cache[n_cpus] = DefaultInference(n_cpus=n_cpus)
        return cls._inference_cache[n_cpus]

    def run(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        design: str,
        contrast: list[str],
        *,
        alpha: float = 0.05,
        fit_type: Literal["mean", "parametric"] = "mean",
        correction_method: str = "fdr_bh",
        n_cpus: int = 16,
        quiet: bool = True,
    ) -> pd.DataFrame:
        """Run PyDESeq2 differential expression.

        Parameters
        ----------
        counts
            Gene expression counts (samples x genes).
        metadata
            Sample metadata with design variables.
        design
            Design formula (e.g., "~condition" or "~condition+batch").
        contrast
            Contrast as [factor, query, reference].
        alpha
            Significance threshold for independent filtering.
        fit_type
            Dispersion fit type: "mean" or "parametric".
        correction_method
            Method for multiple testing correction.
        n_cpus
            Number of CPUs for parallel processing.
        quiet
            Whether to suppress PyDESeq2 output.

        Returns
        -------
        pd.DataFrame
            DE results with log2FoldChange, pvalue, padj, baseMean.
        """
        try:
            inference = self._get_inference(n_cpus)

            dds = DeseqDataSet(
                counts=counts,
                metadata=metadata,
                design=design,
                inference=inference,
                fit_type=fit_type,
                quiet=quiet,
            )

            dds.deseq2()

        except Exception as e:
            raise RuntimeError(
                f"DESeq2 model fitting failed: {e}\n"
                f"Design: {design}\n"
                f"Samples: {counts.shape[0]}, Genes: {counts.shape[1]}"
            ) from e

        try:
            ds = DeseqStats(
                dds,
                contrast=contrast,
                alpha=alpha,
                independent_filter=False,
                quiet=quiet,
                n_cpus=n_cpus,
            )

            ds.summary()
            results = ds.results_df.copy()

            padj = sm.stats.multipletests(
                results["pvalue"][results["pvalue"].notna()].values,
                alpha=alpha,
                method=correction_method,
            )[1]
            results["padj"] = np.nan
            results.loc[results["pvalue"].notna(), "padj"] = padj

            return results

        except Exception as e:
            raise RuntimeError(f"DESeq2 statistics computation failed: {e}") from e
