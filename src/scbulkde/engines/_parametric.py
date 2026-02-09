from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ._base import DEEngineBase


class AnovaEngine(DEEngineBase):
    """Test nested regression models using ANOVA F-test."""

    name = "anova"

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
    ) -> pd.DataFrame:
        """Run ANOVA F-test differential expression."""
        try:
            # Extract numpy arrays efficiently
            counts_arr, gene_names_arr = self._get_counts_and_gene_names(counts, gene_names)
            n = metadata.shape[0]

            # --- Design matrices ---
            X_full = design_matrix.astype(np.float32).values
            X_reduced = (
                design_matrix.loc[:, [c for c in design_matrix.columns if "psbulk_condition" not in c]]
                .astype(np.float32)
                .values
            )

            p_full = X_full.shape[1]
            p_reduced = X_reduced.shape[1]
            q = p_full - p_reduced

            # --- Counts → log-normalized expression ---
            # Work directly with numpy array, avoid copy if possible
            Y = counts_arr.astype(np.float32, copy=True)
            row_sums = Y.sum(axis=1, keepdims=True)
            np.divide(Y, row_sums, out=Y)
            Y *= 1e6
            np.log2(Y + 1, out=Y)

            # --- QR-based projections (shared across genes) ---
            Qf, _ = np.linalg.qr(X_full, mode="reduced")
            Qr, _ = np.linalg.qr(X_reduced, mode="reduced")

            # Residuals
            res_full = Y - Qf @ (Qf.T @ Y)
            res_reduced = Y - Qr @ (Qr.T @ Y)

            rss_full = np.sum(res_full**2, axis=0)
            rss_reduced = np.sum(res_reduced**2, axis=0)

            # --- F-statistic ---
            eps = 1e-20
            F = ((rss_reduced - rss_full) / (rss_full + eps)) * ((n - p_full) / q)
            pvals = stats.f.sf(F, q, n - p_full)

            results = pd.DataFrame(
                {"pvalue": pvals, "stat": F},
                index=gene_names_arr,
            )

            # --- Multiple testing correction ---
            results["padj"] = sm.stats.multipletests(
                results["pvalue"],
                alpha=alpha,
                method=correction_method,
            )[1]

            # --- Manual log2 fold change ---
            query_mask = (metadata["psbulk_condition"] == "query").values
            reference_mask = (metadata["psbulk_condition"] == "reference").values

            mean_query = np.mean(Y[query_mask, :], axis=0)
            mean_reference = np.mean(Y[reference_mask, :], axis=0)
            lfc = mean_query - mean_reference

            results["log2FoldChange"] = lfc
            results["stat_sign"] = results["stat"] * np.sign(lfc)

            return results.loc[:, ["pvalue", "stat", "padj", "log2FoldChange", "stat_sign"]]

        except Exception as e:
            raise RuntimeError("ANOVA failed") from e
