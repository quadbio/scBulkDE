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
            # Extract counts matrix and gene names
            counts_arr, gene_names_arr = self._get_counts_and_gene_names(counts, gene_names)

            # Number of samples
            n = metadata.shape[0]

            # Set up design matrices for full and reduced models
            X_full = design_matrix.astype(np.float32).values
            X_reduced = (
                design_matrix.loc[:, [c for c in design_matrix.columns if "psbulk_condition" not in c]]
                .astype(np.float32)
                .values
            )

            p_full = X_full.shape[1]  # parameters in the full model
            p_reduced = X_reduced.shape[1]  # parameters in the reduced model. Is p_full - 1
            q = p_full - p_reduced

            # Lognormalize counts using pseudocounts
            Y = counts_arr.astype(np.float32)
            row_sums = Y.sum(axis=1, keepdims=True)
            np.divide(Y, row_sums, out=Y)
            Y *= 1e6
            np.log2(Y + 1, out=Y)

            # Project Y onto the column spaces using QR decomposition
            # The mode='reduced' returns a non-square matrix Q with orthonormal columns,
            # which satisfies Q^tQ = Id (but QQ^t is not Id)
            Qf, _ = np.linalg.qr(X_full, mode="reduced")
            Qr, _ = np.linalg.qr(X_reduced, mode="reduced")

            # Residuals
            res_full = Y - Qf @ (Qf.T @ Y)
            res_reduced = Y - Qr @ (Qr.T @ Y)

            rss_full = np.sum(res_full**2, axis=0)
            rss_reduced = np.sum(res_reduced**2, axis=0)

            # F-statistic
            eps = 1e-20
            F = ((rss_reduced - rss_full) / (rss_full + eps)) * ((n - p_full) / q)
            pvals = stats.f.sf(F, q, n - p_full)

            results = pd.DataFrame(
                {"pvalue": pvals, "stat": F},
                index=gene_names_arr,
            )

            # Multiple testing correction
            results["padj"] = sm.stats.multipletests(
                results["pvalue"],
                alpha=alpha,
                method=correction_method,
            )[1]

            # Get the lfc from the full model by accessing the second element in
            # beta_hat_full ("C(psbulk_condition, contr.treatment(base='reference'))[T.query]").
            # NOTE: that this is not the same as the lfc between means of the two groups,
            # as the coefficient is corrected for other covariates in the model.
            # NOTE: actually I noticed the lfcs are inflated when generating pseudoreplicates,
            # and so is the statistic, maybe switch back to manual computation in the future?
            # results["log2FoldChange"] = beta_hat_full[1, :]

            # NOTE: Manual computation of lfc between means of the two groups
            query_mask = (metadata["psbulk_condition"] == "query").values
            reference_mask = (metadata["psbulk_condition"] == "reference").values

            mean_query = np.mean(Y[query_mask, :], axis=0)
            mean_reference = np.mean(Y[reference_mask, :], axis=0)
            lfc = mean_query - mean_reference

            results["log2FoldChange"] = lfc

            # Now for plotting purposes, genes are often sorted by the statistic.
            # This statistic is always positive, which means also genes that are downregulated
            # in query would be listed. To fix this, we multiply the stat with the sign of the lfc.
            results["stat_sign"] = results["stat"] * np.sign(lfc)

            return results.loc[:, ["pvalue", "stat", "padj", "log2FoldChange", "stat_sign"]]

        except Exception as e:
            raise RuntimeError("ANOVA failed") from e
