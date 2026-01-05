"""PyDESeq2 engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from ._base import DEEngine

if TYPE_CHECKING:
    from typing import Literal


class PyDESeq2Engine(DEEngine):
    """DESeq2 engine using PyDESeq2 (pure Python)."""

    name = "pydeseq2"

    # Cache inference objects by n_cpus to avoid repeated initialization
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
        cooks: bool = True,
        fit_type: Literal["mean", "parametric"] = "mean",
        independent_filter: bool = True,
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
        cooks
            Whether to apply Cook's distance filtering.
        fit_type
            Dispersion fit type: "mean" or "parametric".
        independent_filter
            Whether to apply independent filtering.
        n_cpus
            Number of CPUs for parallel processing.
        quiet
            Whether to suppress PyDESeq2 output.
        **kwargs
            Additional arguments (unused).

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
                refit_cooks=cooks,
                fit_type=fit_type,
                quiet=quiet,
            )

            dds.fit_size_factors()
            dds.fit_genewise_dispersions()
            dds.fit_dispersion_trend()
            dds.fit_dispersion_prior()
            dds.fit_MAP_dispersions()
            dds.fit_LFC()
            dds.calculate_cooks()
            dds.refit()

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
                cooks_filter=cooks,
                independent_filter=independent_filter,
                quiet=quiet,
                n_cpus=n_cpus,
            )

            ds.run_wald_test()
            ds._cooks_filtering()
            ds._p_value_adjustment()
            ds.summary()

        except Exception as e:
            raise RuntimeError(f"DESeq2 statistical testing failed: {e}\nContrast: {contrast}") from e

        return ds.results_df.copy()
