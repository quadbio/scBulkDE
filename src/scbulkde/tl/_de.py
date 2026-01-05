"""Differential expression testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from scbulkde.engines import get_engine
from scbulkde.pp import PseudobulkResult, pseudobulk
from scbulkde.utils import logger

from ._result import DEResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    import anndata as ad


def de(
    data: ad.AnnData | PseudobulkResult,
    *,
    group_key: str | None = None,
    query: str | None = None,
    reference: str | Sequence[str] = "rest",
    layer: str | None = None,
    replicate_key: str | None = None,
    batch_key: str | None = None,
    min_cells: int = 10,
    min_fraction: float = 0.0,
    min_coverage: float = 0.0,
    min_bridging_batches: int = 1,
    mode: str = "sum",
    min_samples: int = 3,
    n_repetitions: int = 10,
    resampling_fraction: float = 0.7,
    min_list_overlap: float = 0.5,
    seed: int = 42,
    engine: str = "pydeseq2",
    **engine_kwargs,
) -> DEResult:
    if isinstance(data, PseudobulkResult):
        pb_result = data
        logger.info("Using provided PseudobulkResult")
    else:
        if group_key is None or query is None:
            raise ValueError(
                "When passing AnnData, 'group_key' and 'query' are required. "
                "Alternatively, run scb.pp.pseudobulk() first and pass the result."
            )

        logger.info("Running pseudobulking...")
        pb_result = pseudobulk(
            adata=data,
            group_key=group_key,
            query=query,
            reference=reference,
            layer=layer,
            replicate_key=replicate_key,
            batch_key=batch_key,
            min_cells=min_cells,
            min_fraction=min_fraction,
            min_coverage=min_coverage,
            min_bridging_batches=min_bridging_batches,
            mode=mode,
        )

    rng = np.random.default_rng(seed)

    required_samples = _compute_required_samples(
        pb_result=pb_result,
        min_samples=min_samples,
    )

    de_engine = get_engine(engine)

    if all(v == 0 for v in required_samples.values()):
        logger.info(f"Running DE with {engine} engine...")
        return _run_de_direct(
            pb_result=pb_result,
            de_engine=de_engine,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
        )
    else:
        logger.info(
            f"Insufficient samples - generating pseudoreplicates "
            f"({n_repetitions} repetitions, {resampling_fraction:.0%} resampling)"
        )
        logger.info(f"Additional samples needed: {required_samples}")
        return _run_de_with_pseudoreplicates(
            pb_result=pb_result,
            de_engine=de_engine,
            required_samples=required_samples,
            n_repetitions=n_repetitions,
            resampling_fraction=resampling_fraction,
            min_list_overlap=min_list_overlap,
            rng=rng,
            engine_name=engine,
            engine_kwargs=engine_kwargs,
        )


def _compute_required_samples(
    pb_result: PseudobulkResult,
    min_samples: int,
) -> dict[str, int]:
    required = {}
    for cond, samples in pb_result.valid_samples_by_condition.items():
        if pb_result.include_batch:
            current = sum(
                len(batches)
                for sample_id in samples
                if sample_id in pb_result.sample_hierarchy.get(cond, {})
                for batches in [pb_result.sample_hierarchy[cond][sample_id]]
            )
        else:
            current = len(samples)
        required[cond] = max(0, min_samples - current)
    return required


def _run_de_direct(
    pb_result: PseudobulkResult,
    de_engine,
    engine_name: str,
    engine_kwargs: dict,
) -> DEResult:
    results = de_engine.run(
        counts=pb_result.counts,
        metadata=pb_result.metadata,
        design=pb_result.design,
        contrast=pb_result.contrast,
        **engine_kwargs,
    )

    logger.info(f"DE complete: {len(results)} genes tested, {(results['padj'] < 0.05).sum()} significant (padj < 0.05)")

    return DEResult(
        results=results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=pb_result.design,
        engine=engine_name,
        used_pseudoreplicates=False,
        n_repetitions=1,
    )


def _run_de_with_pseudoreplicates(
    pb_result: PseudobulkResult,
    de_engine,
    required_samples: dict[str, int],
    n_repetitions: int,
    resampling_fraction: float,
    min_list_overlap: float,
    rng: np.random.Generator,
    engine_name: str,
    engine_kwargs: dict,
) -> DEResult:
    repetition_results = {}
    repetition_stats = {}

    for rep_idx in range(n_repetitions):
        logger.debug(f"Repetition {rep_idx + 1}/{n_repetitions}")

        rep_counts, rep_metadata, rep_stats = _generate_pseudoreplicates_for_repetition(
            pb_result=pb_result,
            required_samples=required_samples,
            resampling_fraction=resampling_fraction,
            rng=rng,
        )

        try:
            rep_result = de_engine.run(
                counts=rep_counts,
                metadata=rep_metadata,
                design=pb_result.design,
                contrast=pb_result.contrast,
                **engine_kwargs,
            )
            repetition_results[str(rep_idx)] = rep_result
            repetition_stats[str(rep_idx)] = rep_stats
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Repetition {rep_idx + 1} failed: {e}")
            continue

    if not repetition_results:
        raise RuntimeError("All repetitions failed. Check your data and parameters.")

    aggregated_results = _aggregate_de_results(
        results=repetition_results,
        min_list_overlap=min_list_overlap,
    )

    n_successful = len(repetition_results)
    logger.info(
        f"DE complete: {n_successful}/{n_repetitions} repetitions successful, "
        f"{len(aggregated_results)} genes tested, "
        f"{(aggregated_results['padj'] < 0.05).sum()} significant (padj < 0.05)"
    )

    return DEResult(
        results=aggregated_results,
        query=pb_result.query,
        reference=pb_result.reference,
        design=pb_result.design,
        engine=engine_name,
        used_pseudoreplicates=True,
        n_repetitions=n_successful,
        repetition_results=repetition_results,
        repetition_stats=repetition_stats,
    )


def _generate_pseudoreplicate(
    adata_sub: ad.AnnData,
    obs_names: list[str],
    sample_key: str,
    batch_key: str | None,
    resampling_fraction: float,
    rng: np.random.Generator,
    layer: str | None,
    mode: str,
) -> ad.AnnData:
    import decoupler as dc

    n_sample = max(1, int(len(obs_names) * resampling_fraction))
    sampled_cells = rng.choice(obs_names, size=n_sample, replace=False)
    adata_sampled = adata_sub[sampled_cells].copy()

    return dc.pp.pseudobulk(
        adata_sampled,
        sample_col=sample_key,
        groups_col=batch_key,
        layer=layer,
        mode=mode,
        verbose=False,
    )


def _generate_pseudoreplicates_for_repetition(
    pb_result: PseudobulkResult,
    required_samples: dict[str, int],
    resampling_fraction: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    adata_sub = pb_result.adata_subset
    sample_hierarchy = pb_result.sample_hierarchy
    include_batch = pb_result.include_batch
    layer = pb_result.layer
    mode = pb_result.mode
    group_key = pb_result.contrast[0]
    batch_key = pb_result.used_batch_key if include_batch else None

    all_counts = [pb_result.counts]
    all_metadata = [pb_result.metadata]
    stats_rows = []

    for _, row in pb_result.sample_stats[pb_result.sample_stats["is_valid"]].iterrows():
        stats_rows.append(
            {
                "condition": row["condition"],
                "batch": row["batch"],
                "psbulk_sample": row["psbulk_sample"],
                "n_cells": row["n_cells"],
                "fraction": row["fraction"],
                "is_pseudoreplicate": False,
                "source_sample": None,
            }
        )

    for condition, n_needed in required_samples.items():
        if n_needed <= 0:
            continue

        sample_ids = list(sample_hierarchy.get(condition, {}).keys())
        if not sample_ids:
            continue

        for i in range(n_needed):
            source_sample = rng.choice(sample_ids)
            batches = sample_hierarchy[condition][source_sample]

            if include_batch and len(batches) > 1:
                source_batch = rng.choice(list(batches.keys()))
                obs_names = batches[source_batch]
            else:
                obs_names = [cell for cells in batches.values() for cell in cells]

            n_sample = max(1, int(len(obs_names) * resampling_fraction))
            sampled_cells = list(rng.choice(obs_names, size=n_sample, replace=False))

            adata_sampled = adata_sub[sampled_cells].copy()

            pseudo_sample_name = f"{source_sample}_pr_{i + 1}"
            adata_sampled.obs[pb_result.used_replicate_key] = pseudo_sample_name

            adata_pr = _generate_pseudoreplicate(
                adata_sub=adata_sub,
                obs_names=obs_names,
                sample_key=pb_result.used_replicate_key,
                batch_key=batch_key,
                resampling_fraction=resampling_fraction,
                rng=rng,
                layer=layer,
                mode=mode,
            )

            adata_pr.obs_names = [pseudo_sample_name]

            X = adata_pr.X
            if hasattr(X, "toarray"):
                X = X.toarray()

            pseudo_counts = pd.DataFrame(
                X,
                index=[pseudo_sample_name],
                columns=adata_pr.var_names,
            )
            all_counts.append(pseudo_counts)

            if include_batch:
                if len(batches) > 1:
                    assigned_batch = source_batch
                else:
                    assigned_batch = list(batches.keys())[0]
            else:
                assigned_batch = "_psbulk_no_batch"

            meta_row = {group_key: condition}
            if include_batch:
                meta_row[batch_key] = assigned_batch

            pseudo_metadata = pd.DataFrame(
                [meta_row],
                index=[pseudo_sample_name],
            )
            all_metadata.append(pseudo_metadata)

            stats_rows.append(
                {
                    "condition": condition,
                    "batch": assigned_batch,
                    "psbulk_sample": pseudo_sample_name,
                    "n_cells": n_sample,
                    "fraction": n_sample / pb_result.condition_totals.get(condition, 1),
                    "is_pseudoreplicate": True,
                    "source_sample": source_sample,
                }
            )

    combined_counts = pd.concat(all_counts, axis=0)
    combined_metadata = pd.concat(all_metadata, axis=0)
    stats_df = pd.DataFrame(stats_rows)

    return combined_counts, combined_metadata, stats_df


def _aggregate_de_results(
    results: dict[str, pd.DataFrame],
    min_list_overlap: float,
) -> pd.DataFrame:
    n_runs = len(results)
    min_occurrences = int(np.ceil(min_list_overlap * n_runs))

    gene_counts = pd.concat([df.reset_index() for df in results.values()]).groupby("index").size()
    keep_genes = gene_counts[gene_counts >= min_occurrences].index

    if len(keep_genes) == 0:
        logger.warning(
            f"No genes passed min_list_overlap threshold ({min_list_overlap:.0%}). "
            "Returning all genes with aggregated stats."
        )
        keep_genes = gene_counts.index

    all_results = pd.concat(results.values())
    all_results = all_results.loc[all_results.index.isin(keep_genes)]

    aggregated = all_results.groupby(all_results.index).mean(numeric_only=True)

    return aggregated
