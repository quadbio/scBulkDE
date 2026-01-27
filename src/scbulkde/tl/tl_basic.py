"""Differential expression testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from formulaic import model_matrix

from scbulkde.engines import get_engine_instance
from scbulkde.pp import pseudobulk
from scbulkde.ut import DEResult, PseudobulkResult, logger
from scbulkde.ut.ut_basic import (
    _aggregate_results,
    _compute_required_samples,
    _generate_pseudoreplicate,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import Literal

    import anndata as ad


def de(
    data: ad.AnnData | PseudobulkResult,
    *,
    # pseudobulking parameters
    group_key: str | None = None,
    query: str | Sequence[str] | None = None,
    reference: str | Sequence[str] = "rest",
    replicate_key: str | None = None,
    min_cells: int | None = 50,
    min_fraction: float | None = 0.2,
    min_coverage: float | None = 0.75,
    categorical_covariates: Sequence[str] | None = None,
    continuous_covariates: Sequence[str] | None = None,
    continuous_aggregation: Literal["mean", "sum", "median"] | Callable | None = "mean",
    layer: str | None = None,
    layer_aggregation: Literal["sum", "mean"] = "sum",
    qualify_strategy: Literal["and", "or"] = "or",
    covariate_strategy: Literal["sequence_order", "most_levels"] = "sequence_order",
    resolve_conflicts: bool = True,
    # pseudoreplicate parameters
    n_repetitions: int = 10,
    resampling_fraction: float = 0.6,
    min_list_overlap: float = 1.0,
    # DE paramters
    min_samples: int = 3,
    alpha: float = 0.05,
    correction_method: str = "fdr_bh",
    engine: str = "anova",
    engine_kwargs: dict | None = None,
    # general parameters
    seed: int = 42,
) -> DEResult:
    """Perform pseudobulked differential expression analysis."""
    if engine_kwargs is None:
        engine_kwargs = {}

    if isinstance(data, PseudobulkResult):
        pb_result = data
        logger.info("Using provided PseudobulkResult")
    else:
        logger.info("Running pseudobulking...")
        pb_result = pseudobulk(
            adata=data,
            group_key=group_key,
            query=query,
            reference=reference,
            replicate_key=replicate_key,
            min_cells=min_cells,
            min_fraction=min_fraction,
            min_coverage=min_coverage,
            categorical_covariates=categorical_covariates,
            continuous_covariates=continuous_covariates,
            continuous_aggregation=continuous_aggregation,
            layer=layer,
            layer_aggregation=layer_aggregation,
            qualify_strategy=qualify_strategy,
            covariate_strategy=covariate_strategy,
            resolve_conflicts=resolve_conflicts,
        )

    rng = np.random.default_rng(seed)

    required_samples = _compute_required_samples(
        grouped=pb_result.grouped,
        min_samples=min_samples,
    )
    de_engine = get_engine_instance(engine)

    if all(v == 0 for v in required_samples.values()):
        logger.info(f"Running DE with {engine} engine...")
        return _run_de_direct(
            pb_result=pb_result,
            alpha=alpha,
            correction_method=correction_method,
            de_engine=de_engine,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
        )
    else:
        logger.info(f"Insufficient samples - generating pseudoreplicates ({required_samples})")
        return run_de_pseudoreplicates(
            pb_result=pb_result,
            alpha=alpha,
            correction_method=correction_method,
            de_engine=de_engine,
            required_samples=required_samples,
            n_repetitions=n_repetitions,
            resampling_fraction=resampling_fraction,
            min_list_overlap=min_list_overlap,
            rng=rng,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
        )


def _run_de_direct(
    pb_result: PseudobulkResult,
    alpha: float,
    correction_method: str,
    de_engine,
    engine_name: str,
    engine_kwargs: dict,
) -> DEResult:
    results = de_engine.run(
        counts=pb_result.pb_counts,
        metadata=pb_result.sample_table,
        design_matrix=pb_result.design_matrix,
        design_formula=pb_result.design_formula,
        alpha=alpha,
        correction_method=correction_method,
        **engine_kwargs,
    )

    # Get pval threshold
    logger.info(
        f"DE complete: {len(results)} genes tested, {(results['padj'] < alpha).sum()} significant (padj < {alpha})"
    )

    return DEResult(
        results=results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=pb_result.design_formula,
        engine=engine_name,
        used_pseudoreplicates=False,
        n_repetitions=1,
    )


def run_de_pseudoreplicates(
    pb_result: PseudobulkResult,
    alpha: float,
    correction_method: str,
    de_engine,
    required_samples: dict[str, int],
    n_repetitions: int,
    resampling_fraction: float,
    min_list_overlap: float,
    rng: np.random.Generator,
    engine_name: str,
    engine_kwargs: dict,
) -> DEResult:
    """Run DE analysis with pseudoreplicate generation."""
    # Iterate over n_repetitions to generate pseudoreplicates

    repetition_results = {}
    for it in range(n_repetitions):
        # Generate pseudoreplicates
        pr_counts_collected = []
        pr_meta_collected = []

        # For each condition, generate the required number of pseudoreplicates
        for condition, n_needed in required_samples.items():
            # Now loop over the number of needed pseudoreplicates
            for _ in range(n_needed):
                pr_counts, pr_meta = _generate_pseudoreplicate(
                    adata=pb_result.adata_sub,
                    condition=condition,
                    grouped=pb_result.grouped,
                    layer=pb_result.layer,
                    layer_aggregation=pb_result.layer_aggregation,
                    continuous_covariates=pb_result.continuous_covariates,
                    continuous_aggregation=pb_result.continuous_aggregation,
                    resampling_fraction=resampling_fraction,
                    rng=rng,
                )

                # Append
                pr_counts_collected.append(pr_counts)
                pr_meta_collected.append(pr_meta)

        # Add pseudoreplicates to the original counts and sample_table
        counts = pd.concat([pb_result.pb_counts] + pr_counts_collected, axis=0)
        sample_table = pd.concat([pb_result.sample_table] + pr_meta_collected, axis=0)

        # Now we actually need to re-compute the design matrix based on the new sample_table
        design_matrix = model_matrix(
            pb_result.design_formula,
            data=sample_table,
        )

        # Do the DE
        results = de_engine.run(
            counts=counts,
            metadata=sample_table,
            design_matrix=design_matrix,
            design_formula=pb_result.design_formula,
            alpha=alpha,
            correction_method=correction_method,
            **engine_kwargs,
        )
        repetition_results[it] = results

    # Aggregate results across repetitions. Careful with this list now. It can be, that the not all genes have a padj < alpha
    # because they were significant in most iterations to meet min_list_overlap, but in some other iterations they were not and
    # had a larger padj, which made the mean of the padj larger than alpha
    aggregated_results, n_genes_tested, n_genes_significant = _aggregate_results(
        results=repetition_results,
        min_list_overlap=min_list_overlap,
        alpha=alpha,
    )

    logger.info(
        f"DE complete with pseudoreplicates: {n_genes_tested} genes tested, {n_genes_significant} were significant with padj < {alpha} in at least {min_list_overlap * 100}% of repetitions"
    )

    return DEResult(
        results=aggregated_results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=pb_result.design_formula,
        engine=engine_name,
        used_pseudoreplicates=True,
        n_repetitions=n_repetitions,
        repetition_results=repetition_results,
    )
