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
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        design_matrix: pd.DataFrame,
        design_formula: str,
        alpha: float,
        correction_method: str,
    ) -> pd.DataFrame:
        """Run ANOVA F-test differential expression.

        Parameters
        ----------
        counts
            Gene expression counts (samples x genes).
        metadata
            Sample metadata with design variables.
        design_matrix
            Design matrix for the regression model.
        design_formula
            For compatibility, not used.
        alpha
            Significance threshold for adjusted p-values.
        correction_method
            Method for multiple testing correction.

        Returns
        -------
        pd.DataFrame
            Results with columns: stat, pvalue, padj, mean_expression.
        """
        try:
            n = metadata.shape[0]
            gene_names = counts.columns

            X_full = design_matrix
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

            # # DEBUGGING: check how many columns are zero
            # colsums = counts.sum(axis=0)
            # zero_cols = np.sum(colsums == 0)
            # print(f"Number of zero-sum columns: {zero_cols} out of {counts.shape[1]}")

            row_sums = counts.sum(axis=1, keepdims=True)
            counts = counts / row_sums * 1e6
            Y = np.log2(counts + 1)

            # DEBUGGING: inspect specific gene and covariate
            # gene="PAX5"
            # gene_idx = gene_names.get_loc(gene)
            # gene_vec = Y[:, gene_idx]
            # x2_vec = X_full[:, 1]
            # df = pd.DataFrame({
            #     gene: gene_vec,
            #     "X_full_col2": x2_vec
            # })
            # print(df)

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

            results = pd.DataFrame({"pvalue": pvals, "stat": F}, index=gene_names)

            # Multiple testing correction
            results["padj"] = sm.stats.multipletests(results["pvalue"], alpha=alpha, method=correction_method)[1]

            # Get the lfc from the full model by accessing the second element in beta_hat_full
            # ("C(psbulk_condition, contr.treatment(base='reference'))[T.query]"). Note, that this is not the same
            # as the lfc between means of the two groups, as the coefficient is corrected for other covariates in the model.
            # NOTE: actually I noticed the lfcs are highly inflated when generating pseudoreplicates, and so is the statistic
            # maybe switch back to manual computation in the future?
            # results["log2FoldChange"] = beta_hat_full[1, :]

            # NOTE: Manual computation of lfc between means of the two groups
            query_mask = metadata["psbulk_condition"] == "query"
            reference_mask = metadata["psbulk_condition"] == "reference"
            mean_query = np.mean(Y[query_mask, :], axis=0)
            mean_reference = np.mean(Y[reference_mask, :], axis=0)
            results["log2FoldChange"] = mean_query - mean_reference

            # Now for plotting purposes, genes are often sorted by the statistic. This statistic is always positive,
            # which means also genes that are downregulated in query would be listed. To fix this, we multiply the stat with the sign of the lfc.
            results["stat_sign"] = results["stat"] * np.sign(results["log2FoldChange"])

            # For consistency, subset to relevant columns
            results = results.loc[:, ["pvalue", "stat", "padj", "log2FoldChange", "stat_sign"]]

            return results

        except Exception as e:
            raise RuntimeError("ANOVA failed") from e
