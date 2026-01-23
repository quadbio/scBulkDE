"""T-test DE engine implementing a vectorized Welch t-test.

Subclasses DEEngineBase and exposes name = "t-test".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._base import DEEngineBase


class AnovaEngine(DEEngineBase):
    """Test nested regression models using ANOVA F-test."""

    name = "anova"

    def run(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        design: str,
        contrast: list[str],
        *,
        alpha: float = 0.05,
        correction_method: str = "fdr_bh",
    ) -> pd.DataFrame:
        """Run ANOVA F-test differential expression.

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
            Significance threshold for adjusted p-values.
        correction_method
            Method for multiple testing correction.

        Returns
        -------
        pd.DataFrame
            Results with columns: F_statistic, pvalue, padj, mean_expression.
        """
        import statsmodels.api as sm
        from scipy import stats

        try:
            n = metadata.shape[0]
            gene_names = counts.columns

            if design == "~psbulk_condition+psbulk_batch":
                X_full = pd.get_dummies(metadata[["psbulk_condition", "psbulk_batch"]], drop_first=True)
                X_full = sm.add_constant(X_full)
            else:
                X_full = pd.get_dummies(metadata[["psbulk_condition"]], drop_first=True)
                X_full = sm.add_constant(X_full)

            X_reduced = X_full.loc[:, [c for c in X_full.columns if "psbulk_condition" not in c]]

            # Convert to numpy arrays
            X_full = X_full.astype(float).values
            X_reduced = X_reduced.astype(float).values

            # Scaling factors
            p_full = X_full.shape[1]
            p_reduced = X_reduced.shape[1]
            q = p_full - p_reduced

            # Lognormalize counts, get the response variable
            counts = counts.values
            row_sums = counts.sum(axis=1, keepdims=True)
            counts = counts / row_sums * 1e6
            Y = np.log2(counts + 1)

            # OLS
            beta_hat_full, _, _, _ = np.linalg.lstsq(X_full, Y, rcond=None)
            Y_hat_full = X_full @ beta_hat_full
            rss_full = np.sum((Y - Y_hat_full) ** 2, axis=0)

            beta_hat_reduced, _, _, _ = np.linalg.lstsq(X_reduced, Y, rcond=None)
            Y_hat_reduced = X_reduced @ beta_hat_reduced
            rss_reduced = np.sum((Y - Y_hat_reduced) ** 2, axis=0)

            # F-statistic. Note that the original formula is n - k - 1 in the
            # numerator of the second term. Here, k does not include the intercept
            # term, so an additional degree of freedom is subtracted. This is not
            # necessary here because p_full is used directly

            eps = 1e-20
            F = ((rss_reduced - rss_full) / (rss_full + eps)) * ((n - p_full) / q)

            # Convert to p-values
            pvals = stats.f.sf(F, q, n - p_full)

            results = pd.DataFrame({"pvalue": pvals}, index=gene_names)

            # Multiple testing correction
            results["padj"] = sm.stats.multipletests(results["pvalue"], alpha=alpha, method=correction_method)[1]

            # Compute the base mean
            results["baseMean"] = np.mean(counts, axis=0)

            # Compute log fold change between query and reference
            query_mask = metadata["psbulk_condition"] == contrast[1]
            reference_mask = metadata["psbulk_condition"] == contrast[2]
            mean_query = np.mean(Y[query_mask, :], axis=0)
            mean_reference = np.mean(Y[reference_mask, :], axis=0)
            results["log2FoldChange"] = mean_query - mean_reference

            return results

        except Exception as e:
            raise RuntimeError("ANOVA failed") from e
