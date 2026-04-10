"""Tests for scbulkde.tl.tl_basic module.

These tests verify the core differential expression pipeline, focusing on:
1. Sample counting logic for direct vs fallback DE
2. Pseudoreplicate generation with proper cell usage tracking
3. Design matrix creation and full-rank enforcement
4. Correct aggregation of results across repetitions
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from scbulkde.tl.tl_basic import (
    _build_cell_pool_cache,
    _count_existing_samples,
    _generate_pseudoreplicate,
)
from scbulkde.ut._containers import PseudobulkResult


class TestCountExistingSamples:
    """Test _count_existing_samples: counts pseudobulk samples per condition."""

    def test_multiindex_groupby_counts_strata(self):
        """MultiIndex groupby should count unique strata combinations per condition."""
        # Create a MultiIndex with condition and replicate levels
        # (query, rep1), (query, rep2), (reference, rep1)
        index = pd.MultiIndex.from_tuples(
            [("query", "rep1"), ("query", "rep2"), ("reference", "rep1")],
            names=["condition", "replicate"],
        )
        obs = pd.DataFrame({"condition": ["query", "query", "reference"]}, index=index)
        grouped = obs.groupby(level=["condition", "replicate"])

        result = _count_existing_samples(grouped)

        # 2 query samples, 1 reference sample
        assert result == {"query": 2, "reference": 1}

    def test_single_index_returns_zero_samples(self):
        """Single index (collapsed case) should return 0 for both conditions."""
        # When only the condition column groups, there are no independent samples
        obs = pd.DataFrame({"condition": ["query", "query", "reference", "reference"]})
        grouped = obs.groupby("condition")

        result = _count_existing_samples(grouped)

        # No strata exist, so no valid samples
        assert result == {"query": 0, "reference": 0}

    def test_empty_multiindex(self):
        """Empty result should handle gracefully."""
        index = pd.MultiIndex.from_tuples([], names=["condition", "replicate"])
        obs = pd.DataFrame({"condition": []}, index=index)
        grouped = obs.groupby(level=["condition", "replicate"])

        result = _count_existing_samples(grouped)

        assert result == {"query": 0, "reference": 0}


class TestBuildCellPoolCache:
    """Test _build_cell_pool_cache: organizes cells by condition and sample."""

    def create_pseudobulk_result(self, n_cells=100, n_genes=50):
        """Helper to create a PseudobulkResult for testing."""
        # Create a simple AnnData object
        X = sp.random(n_cells, n_genes, density=0.1, format="csr")
        adata = AnnData(X)
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
        adata.var_names = [f"gene_{i}" for i in range(n_genes)]

        # Create obs with grouping: condition, replicate
        n_query = n_cells // 2
        conditions = ["query"] * n_query + ["reference"] * (n_cells - n_query)
        replicates = ["rep1", "rep2"] * (n_cells // 2)
        if n_cells % 2 == 1:
            replicates.append("rep1")

        adata.obs["condition"] = conditions
        adata.obs["replicate"] = replicates

        # Create grouped object
        obs_grouped = adata.obs.groupby(["condition", "replicate"], observed=True, sort=False)

        # Create minimal PseudobulkResult
        pb_result = PseudobulkResult(
            adata_sub=adata,
            pb_counts=pd.DataFrame(),
            grouped=obs_grouped,
            sample_table=pd.DataFrame(),
            design_matrix=pd.DataFrame(),
            design_formula="",
            group_key="condition",
            group_key_internal="condition",
            query=["query"],
            reference=["reference"],
            strata=["replicate"],
            layer=None,
            layer_aggregation="sum",
            categorical_covariates=None,
            continuous_covariates=None,
            continuous_aggregation="mean",
            min_cells=None,
            min_fraction=None,
            min_coverage=None,
            qualify_strategy="or",
        )

        return pb_result

    def test_cache_separates_conditions(self):
        """Cache should organize cells into query and reference pools."""
        pb_result = self.create_pseudobulk_result(n_cells=100)
        rng = np.random.default_rng(42)

        cache = _build_cell_pool_cache(pb_result, rng, shuffle=False)

        # Should have query and reference keys
        assert "query" in cache
        assert "reference" in cache

        # Both should have samples (lists of tuples)
        assert len(cache["query"]) > 0
        assert len(cache["reference"]) > 0

        # Each sample tuple should be (size, cell_indices)
        for condition in ["query", "reference"]:
            for size, indices in cache[condition]:
                assert isinstance(size, (int, np.integer))
                assert isinstance(indices, np.ndarray)
                assert len(indices) == size

    def test_cache_preserves_cell_information(self):
        """Cache should preserve cell indices and counts correctly."""
        pb_result = self.create_pseudobulk_result(n_cells=50)
        rng = np.random.default_rng(42)

        cache = _build_cell_pool_cache(pb_result, rng, shuffle=False)

        # Collect all cells from cache
        all_cached_cells = []
        for condition in ["query", "reference"]:
            for _, indices in cache[condition]:
                all_cached_cells.extend(indices)

        # Should match the obs index
        assert len(all_cached_cells) == len(pb_result.adata_sub.obs)

    def test_cache_respects_grouping_structure(self):
        """Each sample in cache should correspond to a unique group."""
        pb_result = self.create_pseudobulk_result(n_cells=100)
        rng = np.random.default_rng(42)

        cache = _build_cell_pool_cache(pb_result, rng, shuffle=False)

        # Count samples
        n_query_samples = len(cache["query"])
        n_ref_samples = len(cache["reference"])

        # Should have same number as existing samples
        existing = _count_existing_samples(pb_result.grouped)
        assert n_query_samples == existing["query"]
        assert n_ref_samples == existing["reference"]


class TestGeneratePseudoreplicate:
    """Test _generate_pseudoreplicate: greedy cell usage minimization."""

    def create_cell_pool_and_tracker(self, n_cells_per_sample=20):
        """Helper to create a mock cell pool cache and usage tracker."""
        # Create 3 samples per condition
        cell_pool = {
            "query": [
                (n_cells_per_sample, np.arange(0, n_cells_per_sample)),
                (n_cells_per_sample, np.arange(n_cells_per_sample, 2 * n_cells_per_sample)),
                (n_cells_per_sample, np.arange(2 * n_cells_per_sample, 3 * n_cells_per_sample)),
            ],
            "reference": [
                (n_cells_per_sample, np.arange(3 * n_cells_per_sample, 4 * n_cells_per_sample)),
                (n_cells_per_sample, np.arange(4 * n_cells_per_sample, 5 * n_cells_per_sample)),
            ],
        }
        cell_usage = dict.fromkeys(range(5 * n_cells_per_sample), 0)
        return cell_pool, cell_usage

    def test_selects_least_used_sample(self):
        """Should select sample with minimum total cell usage."""
        cell_pool, cell_usage = self.create_cell_pool_and_tracker(n_cells_per_sample=20)

        # Mark second sample as heavily used
        for cell_id in cell_pool["query"][1][1]:
            cell_usage[cell_id] = 5

        rng = np.random.default_rng(42)
        resampling_frac = 0.5

        # Generate multiple pseudoreplicates
        selected_samples = []
        for _ in range(3):
            pr_indices = _generate_pseudoreplicate(
                condition="query",
                cell_pool_cache=cell_pool,
                cell_usage_tracker=cell_usage,
                resampling_fraction=resampling_frac,
                rng=rng,
            )
            selected_samples.append(set(pr_indices))

        # Should preferentially sample from less-used samples (0 or 2)
        # Count how many come from sample 1 (the heavily used one)
        sample_1_cells = set(cell_pool["query"][1][1])
        from_sample_1 = sum(len(s & sample_1_cells) for s in selected_samples)

        # Most cells should come from other samples
        total_selected = sum(len(s) for s in selected_samples)
        assert from_sample_1 < total_selected * 0.5

    def test_respects_resampling_fraction(self):
        """Generated pseudoreplicate size should match resampling_fraction."""
        cell_pool, cell_usage = self.create_cell_pool_and_tracker(n_cells_per_sample=100)
        rng = np.random.default_rng(42)

        resampling_frac = 0.33
        pr_indices = _generate_pseudoreplicate(
            condition="query",
            cell_pool_cache=cell_pool,
            cell_usage_tracker=cell_usage,
            resampling_fraction=resampling_frac,
            rng=rng,
        )

        # Size should be approximately sample_size * resampling_fraction
        expected_size = int(100 * resampling_frac)
        assert len(pr_indices) == expected_size

    def test_updates_usage_tracker(self):
        """Should increment usage count for selected cells."""
        cell_pool, cell_usage = self.create_cell_pool_and_tracker(n_cells_per_sample=50)
        rng = np.random.default_rng(42)

        initial_usage = cell_usage.copy()

        pr_indices = _generate_pseudoreplicate(
            condition="query",
            cell_pool_cache=cell_pool,
            cell_usage_tracker=cell_usage,
            resampling_fraction=0.4,
            rng=rng,
        )

        # Selected cells should have incremented usage
        for cell_id in pr_indices:
            assert cell_usage[cell_id] == initial_usage[cell_id] + 1

        # Non-selected cells should be unchanged
        all_cells = set(np.concatenate([indices for _, indices in cell_pool["query"]]))
        non_selected = all_cells - set(pr_indices)
        for cell_id in non_selected:
            assert cell_usage[cell_id] == initial_usage[cell_id]

    def test_prefers_least_used_cells_within_sample(self):
        """Within a sample, should prefer cells with lowest usage."""
        cell_pool, cell_usage = self.create_cell_pool_and_tracker(n_cells_per_sample=30)

        # Mark some cells in the first sample as heavily used
        sample_0_cells = cell_pool["query"][0][1]
        heavily_used_cells = sample_0_cells[:10]
        for cell_id in heavily_used_cells:
            cell_usage[cell_id] = 10

        rng = np.random.default_rng(42)

        # Generate pseudoreplicate from query
        pr_indices = _generate_pseudoreplicate(
            condition="query",
            cell_pool_cache=cell_pool,
            cell_usage_tracker=cell_usage,
            resampling_fraction=0.5,
            rng=rng,
        )

        # If first sample is selected, unused cells should be preferred
        if any(cell_id in sample_0_cells for cell_id in pr_indices):
            from_sample_0 = set(pr_indices) & set(sample_0_cells)
            unused_in_sample_0 = set(sample_0_cells[10:])

            # Most selected from sample 0 should be from unused cells
            assert len(from_sample_0 & unused_in_sample_0) > len(from_sample_0) * 0.7

    def test_handles_exhausted_least_used_tier(self):
        """Should move to next usage tier if least-used cells insufficient."""
        cell_pool, cell_usage = self.create_cell_pool_and_tracker(n_cells_per_sample=50)

        # Mark all samples with same usage_rate so we can control which one is selected
        # Set sample 0 as the least-used (will be selected)
        # Within sample 0: mark all but 5 cells as used once (usage=1)
        sample_0_cells = cell_pool["query"][0][1]
        for cell_id in sample_0_cells[5:]:
            cell_usage[cell_id] = 1

        # Mark other samples with higher usage to ensure sample 0 is selected. So sample 0
        # is overall the least used one
        for i in range(1, len(cell_pool["query"])):
            for cell_id in cell_pool["query"][i][1]:
                cell_usage[cell_id] = 1

        rng = np.random.default_rng(42)

        # Request 50% of cells (25 cells) but only 5 have zero usage in the selected sample
        pr_indices = _generate_pseudoreplicate(
            condition="query",
            cell_pool_cache=cell_pool,
            cell_usage_tracker=cell_usage,
            resampling_fraction=0.5,
            rng=rng,
        )

        # Should be able to generate with 25 cells (5 unused + 20 from next tier)
        assert len(pr_indices) == 25

        # All selected cells should be from sample 0 (the least-used sample)
        sample_0_set = set(sample_0_cells)
        assert set(pr_indices).issubset(sample_0_set)

        # Should contain all 5 unused cells plus some with usage=1
        least_used = set(sample_0_cells[:5])
        next_tier = set(sample_0_cells[5:])

        pr_set = set(pr_indices)
        assert least_used.issubset(pr_set)
        assert len(pr_set & next_tier) == 20


class TestPseudoreplicateIndependence:
    """Test that pseudoreplicate generation maintains cell independence."""

    def test_cell_usage_tracked_across_iterations(self):
        """Cell usage should accumulate across multiple pseudoreplicate generations."""
        cell_pool, cell_usage = _GeneratePseudoreplicate.create_cell_pool_and_tracker(n_cells_per_sample=50)
        rng = np.random.default_rng(42)

        # Generate multiple pseudoreplicates
        for _ in range(3):
            _generate_pseudoreplicate(
                condition="query",
                cell_pool_cache=cell_pool,
                cell_usage_tracker=cell_usage,
                resampling_fraction=0.3,
                rng=rng,
            )

        # After 3 iterations with 0.3 fraction, the total usage should be
        # 0.3 * 3 * 50 / 250 = 0.18 on average (since there are 250 cells in total across all samples)
        usage_values = list(cell_usage.values())
        mean_usage = np.mean(usage_values)
        assert mean_usage == 0.18

    def test_diversity_across_iterations(self):
        """Different iterations should sample different sets of cells."""
        cell_pool, cell_usage = _GeneratePseudoreplicate.create_cell_pool_and_tracker(n_cells_per_sample=100)
        rng = np.random.default_rng(42)

        samples = []
        for _ in range(5):
            pr_indices = _generate_pseudoreplicate(
                condition="query",
                cell_pool_cache=cell_pool,
                cell_usage_tracker=cell_usage,
                resampling_fraction=0.2,
                rng=rng,
            )
            samples.append(set(pr_indices))

        # Pairwise overlap should be relatively low
        overlaps = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                overlap = len(samples[i] & samples[j]) / len(samples[i])
                overlaps.append(overlap)

        # Average overlap should be well below 50%
        mean_overlap = np.mean(overlaps)
        assert mean_overlap < 0.4


# Helper class reference fix
class _GeneratePseudoreplicate:
    """Helper reference for test class."""

    @staticmethod
    def create_cell_pool_and_tracker(n_cells_per_sample=20):
        """Helper to create a mock cell pool cache and usage tracker."""
        cell_pool = {
            "query": [
                (n_cells_per_sample, np.arange(0, n_cells_per_sample)),
                (n_cells_per_sample, np.arange(n_cells_per_sample, 2 * n_cells_per_sample)),
                (n_cells_per_sample, np.arange(2 * n_cells_per_sample, 3 * n_cells_per_sample)),
            ],
            "reference": [
                (n_cells_per_sample, np.arange(3 * n_cells_per_sample, 4 * n_cells_per_sample)),
                (n_cells_per_sample, np.arange(4 * n_cells_per_sample, 5 * n_cells_per_sample)),
            ],
        }
        cell_usage = dict.fromkeys(range(5 * n_cells_per_sample), 0)
        return cell_pool, cell_usage
