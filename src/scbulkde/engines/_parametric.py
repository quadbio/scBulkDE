from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from scipy.special import digamma, polygamma

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


def _trigamma_inverse(x: np.ndarray, tol: float = 1e-08, iter_limit: int = 50) -> np.ndarray:
    """Newton's method to solve trigamma inverse.

    Parameters
    ----------
    x
        Input values.
    tol
        Convergence tolerance.
    iter_limit
        Maximum number of iterations.

    Returns
    -------
    np.ndarray
        Trigamma inverse values.
    """
    y = 0.5 + 1 / x
    for _ in range(iter_limit):
        tri = polygamma(1, y)
        diff = tri * (1 - tri / x) / polygamma(2, y)
        y += diff
        if np.max(-diff / y) < tol:
            break
    else:
        warnings.warn(f"trigamma_inverse iteration limit ({iter_limit}) exceeded")  # noqa: B028
    return y


def _fit_f_dist(x: np.ndarray, df1: int) -> tuple[float, float]:
    """Fit f-distribution using method of moments.

    Parameters
    ----------
    x
        Sample variances.
    df1
        Degrees of freedom.

    Returns
    -------
    tuple[float, float]
        Prior degrees of freedom (df2) and scale parameter.
    """
    z = np.log(x)
    e = z - digamma(df1 / 2) + np.log(df1 / 2)

    e_mean = np.mean(e)
    e_var = np.var(e, ddof=1)

    e_var -= polygamma(1, df1 / 2)

    if e_var > 0:
        df2 = 2 * _trigamma_inverse(np.array([e_var]))[0]
        scale = np.exp(e_mean + digamma(df2 / 2) - np.log(df2 / 2))
    else:
        df2 = np.inf
        scale = np.exp(e_mean)

    return df2, scale


def _moderate_variances(
    variances: np.ndarray,
    df: int,
) -> tuple[np.ndarray, float, float]:
    """Compute moderated (posterior) variances using empirical Bayes.

    Assumes each gene's variance is sampled from a scaled inverse chi-square
    prior distribution with degrees of freedom d0 and location s_0^2.

    Parameters
    ----------
    variances
        Sample variances for each gene.
    df
        Degrees of freedom.

    Returns
    -------
    tuple[np.ndarray, float, float]
        Posterior variances, prior variance, and prior degrees of freedom.
    """
    var = variances.copy()

    # Handle zero variances
    idxs_zero = np.where(var == 0)[0]
    if idxs_zero.size > 0:
        var[idxs_zero] += np.finfo(var.dtype).eps

    df_prior, var_prior = _fit_f_dist(var, df)

    # Posterior variance calculation
    var_post = (df_prior * var_prior + df * var) / (df + df_prior)

    return var_post, var_prior, df_prior


class EbayesEngine(DEEngineBase):
    """Empirical Bayes moderated t-test for differential expression.

    Implements moderated t-statistics as defined in Smyth (2004) and Phipson et al. (2016).
    This approach:
    - Fits a linear model for each gene
    - Moderates gene expression residual variances using empirical Bayes
    - Performs moderated t-test comparing query vs reference condition
    """

    name = "ebayes"

    def run(
        self,
        counts: pd.DataFrame,
        metadata: pd.DataFrame,
        design_matrix: pd.DataFrame,
        design_formula: str,
        alpha: float,
        correction_method: str,
    ) -> pd.DataFrame:
        """Run empirical Bayes moderated t-test differential expression.

        Parameters
        ----------
        counts
            Gene expression counts (samples x genes).
        metadata
            Sample metadata with design variables. Must contain 'psbulk_condition'
            column with 'query' and 'reference' values.
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
            Results with columns: pvalue, stat, padj, log2FoldChange, stat_sign.
        """
        try:
            n = metadata.shape[0]
            gene_names = counts.columns

            # Convert design matrix to numpy
            X = design_matrix.astype(float).values
            p = X.shape[1]

            # Lognormalize counts
            counts_vals = counts.values
            row_sums = counts_vals.sum(axis=1, keepdims=True)
            counts_norm = counts_vals / row_sums * 1e6
            Y = np.log2(counts_norm + 1)

            # Fit OLS for each gene
            beta_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            Y_hat = X @ beta_hat
            residuals = Y - Y_hat

            # Compute residual sum of squares and sample variances
            df_residual = n - p
            rss = np.sum(residuals**2, axis=0)
            sigma_sq = rss / df_residual

            # Moderate variances using empirical Bayes
            sigma_sq_post, var_prior, df_prior = _moderate_variances(sigma_sq, df_residual)

            # Find the coefficient index for the query condition
            # Look for column containing "psbulk_condition" and "query"
            coef_idx = None
            for i, col in enumerate(design_matrix.columns):
                if "psbulk_condition" in col and "query" in col:
                    coef_idx = i
                    break

            if coef_idx is None:
                # Fall back to second column (after intercept) if pattern not found
                coef_idx = 1 if p > 1 else 0

            # Get the contrast coefficient (query effect)
            contrast_coef = beta_hat[coef_idx, :]

            # Compute unscaled standard error for the contrast
            # Standard error = sqrt(sigma^2 * (X'X)^-1[coef_idx, coef_idx])
            XtX_inv = np.linalg.inv(X.T @ X)
            stdev_unscaled = np.sqrt(XtX_inv[coef_idx, coef_idx])

            # Compute total degrees of freedom (capped at pooled df)
            df_total = df_residual + df_prior
            df_pooled = df_residual * len(gene_names)  # Total pooled df
            df_total = min(df_total, df_pooled)

            # Moderated t-statistic
            t_vals = contrast_coef / (np.sqrt(sigma_sq_post) * stdev_unscaled)

            # Two-tailed p-values
            pvals = 2 * stats.t.sf(np.abs(t_vals), df_total)

            # Build results dataframe
            results = pd.DataFrame(
                {"pvalue": pvals, "stat": t_vals},
                index=gene_names,
            )

            # Multiple testing correction
            results["padj"] = sm.stats.multipletests(results["pvalue"], alpha=alpha, method=correction_method)[1]

            # Log2 fold change: difference of means between query and reference
            query_mask = metadata["psbulk_condition"] == "query"
            reference_mask = metadata["psbulk_condition"] == "reference"
            mean_query = np.mean(Y[query_mask, :], axis=0)
            mean_reference = np.mean(Y[reference_mask, :], axis=0)
            results["log2FoldChange"] = mean_query - mean_reference

            # Signed statistic for plotting (already signed for t-test)
            results["stat_sign"] = results["stat"]

            # Subset to relevant columns for consistency
            results = results.loc[:, ["pvalue", "stat", "padj", "log2FoldChange", "stat_sign"]]

            return results

        except Exception as e:
            raise RuntimeError("Empirical Bayes moderated t-test failed") from e
