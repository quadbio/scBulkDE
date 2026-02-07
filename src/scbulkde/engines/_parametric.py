from __future__ import annotations

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
        """Run ANOVA F-test differential expression."""
        try:
            n = metadata.shape[0]
            gene_names = counts.columns

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
            counts = counts.values.astype(np.float32, copy=False)

            row_sums = counts.sum(axis=1, keepdims=True)
            np.divide(counts, row_sums, out=counts)
            counts *= 1e6
            np.log2(counts + 1, out=counts)
            Y = counts  # (cells × genes)

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
                index=gene_names,
            )

            # --- Multiple testing correction ---
            results["padj"] = sm.stats.multipletests(
                results["pvalue"],
                alpha=alpha,
                method=correction_method,
            )[1]

            # --- Manual log2 fold change ---
            query_mask = metadata["psbulk_condition"] == "query"
            reference_mask = metadata["psbulk_condition"] == "reference"

            mean_query = np.mean(Y[query_mask, :], axis=0)
            mean_reference = np.mean(Y[reference_mask, :], axis=0)
            lfc = mean_query - mean_reference

            results["log2FoldChange"] = lfc
            results["stat_sign"] = results["stat"] * np.sign(lfc)

            return results.loc[:, ["pvalue", "stat", "padj", "log2FoldChange", "stat_sign"]]

        except Exception as e:
            raise RuntimeError("ANOVA failed") from e


def _trigamma_inverse_scalar(x: float, tol: float = 1e-8, iter_limit: int = 25) -> float:
    """Optimized Newton's method for scalar trigamma inverse."""
    y = 0.5 + 1.0 / x
    for _ in range(iter_limit):
        tri = float(polygamma(1, y))
        denom = float(polygamma(2, y))
        if abs(denom) < 1e-15:
            break
        diff = tri * (1.0 - tri / x) / denom
        y += diff
        if abs(diff) < tol * abs(y):
            break
    return y


def _fit_f_dist_fast(var: np.ndarray, df1: int) -> tuple[float, float]:
    """Optimized f-distribution fitting."""
    half_df1 = df1 * 0.5
    log_half_df1 = np.log(half_df1)
    digamma_half_df1 = float(digamma(half_df1))
    trigamma_half_df1 = float(polygamma(1, half_df1))

    # Vectorized log
    z = np.log(var)
    e = z - digamma_half_df1 + log_half_df1

    e_mean = float(np.mean(e))
    e_var = float(np.var(e, ddof=1)) - trigamma_half_df1

    if e_var > 0:
        df2 = 2.0 * _trigamma_inverse_scalar(e_var)
        half_df2 = df2 * 0.5
        scale = np.exp(e_mean + float(digamma(half_df2)) - np.log(half_df2))
    else:
        df2 = np.inf
        scale = np.exp(e_mean)

    return df2, scale


def _moderate_variances_fast(
    sigma_sq: np.ndarray,
    df: int,
) -> tuple[np.ndarray, float, float]:
    """Optimized variance moderation."""
    # Handle zeros in-place
    var = np.maximum(sigma_sq, np.finfo(sigma_sq.dtype).eps)

    df_prior, var_prior = _fit_f_dist_fast(var, df)

    # Vectorized posterior
    if np.isinf(df_prior):
        var_post = np.full_like(var, var_prior)
    else:
        var_post = (df_prior * var_prior + df * var) / (df + df_prior)

    return var_post, var_prior, df_prior


class EbayesEngine(DEEngineBase):
    """Empirical Bayes moderated t-test for differential expression.

    Implements moderated t-statistics as defined in Smyth (2004) and Phipson et al. (2016).
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
        """Run empirical Bayes moderated t-test differential expression."""
        try:
            n = metadata.shape[0]
            gene_names = counts.columns
            n_genes = len(gene_names)

            # Get numpy arrays directly - avoid copies
            X = design_matrix.values.astype(np.float64)
            p = X.shape[1]

            # Get counts - handle both DataFrame and ndarray
            if isinstance(counts, pd.DataFrame):
                counts_arr = counts.values
            else:
                counts_arr = counts

            # Lognormalize - single pass
            row_sums = counts_arr.sum(axis=1, keepdims=True)
            Y = np.log2(counts_arr / row_sums * 1e6 + 1)

            # Precompute X'X inverse using Cholesky (faster & more stable)
            XtX = X.T @ X
            try:
                L = np.linalg.cholesky(XtX)
                # Solve for inverse diagonal element we need
                e_coef = np.zeros(p)
                # Find coefficient index
                coef_idx = None
                for i, col in enumerate(design_matrix.columns):
                    if "psbulk_condition" in col and "query" in col:
                        coef_idx = i
                        break
                if coef_idx is None:
                    coef_idx = 1 if p > 1 else 0

                e_coef[coef_idx] = 1.0
                # Solve L @ z = e, then L.T @ inv_col = z
                z = np.linalg.solve(L, e_coef)
                stdev_unscaled_sq = np.dot(z, z)
                stdev_unscaled = np.sqrt(stdev_unscaled_sq)
            except np.linalg.LinAlgError:
                # Fallback to full inverse
                XtX_inv = np.linalg.pinv(XtX)
                coef_idx = 1 if p > 1 else 0
                for i, col in enumerate(design_matrix.columns):
                    if "psbulk_condition" in col and "query" in col:
                        coef_idx = i
                        break
                stdev_unscaled = np.sqrt(XtX_inv[coef_idx, coef_idx])

            # OLS fit - vectorized across all genes
            beta_hat, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            Y_hat = X @ beta_hat

            # Residual variance - use einsum for speed
            df_residual = n - p
            residuals = Y - Y_hat
            rss = np.einsum("ij,ij->j", residuals, residuals)
            sigma_sq = rss / df_residual

            # Moderate variances
            sigma_sq_post, var_prior, df_prior = _moderate_variances_fast(sigma_sq, df_residual)

            # Get contrast coefficient
            contrast_coef = beta_hat[coef_idx, :]

            # Degrees of freedom
            df_total = min(df_residual + df_prior, df_residual * n_genes)

            # Moderated t-statistic
            t_vals = contrast_coef / (np.sqrt(sigma_sq_post) * stdev_unscaled)

            # P-values
            pvals = 2.0 * stats.t.sf(np.abs(t_vals), df_total)

            # Build results
            results = pd.DataFrame(
                {"pvalue": pvals, "stat": t_vals},
                index=gene_names,
            )

            # Multiple testing
            results["padj"] = sm.stats.multipletests(pvals, alpha=alpha, method=correction_method)[1]

            # Log2 fold change
            query_mask = (metadata["psbulk_condition"] == "query").values
            reference_mask = (metadata["psbulk_condition"] == "reference").values
            results["log2FoldChange"] = Y[query_mask].mean(axis=0) - Y[reference_mask].mean(axis=0)

            results["stat_sign"] = t_vals
            results = results.loc[:, ["pvalue", "stat", "padj", "log2FoldChange", "stat_sign"]]

            return results

        except Exception as e:
            raise RuntimeError("Empirical Bayes moderated t-test failed") from e
