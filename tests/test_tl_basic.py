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
import pytest
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


class TestDEFunction:
    """Integration tests for the main de() entry point.

    These tests validate the routing logic and pipeline behavior by mocking
    the DE engine to isolate the tl_basic orchestration from statistical computation.
    """

    @pytest.fixture
    def mock_engine(self, monkeypatch):
        """Mock the DE engine to return deterministic results."""

        class MockEngine:
            name = "mock"

            def run(
                self,
                counts,
                metadata,
                design_matrix,
                design_formula,
                alpha,
                correction_method,
                gene_names=None,
                **kwargs,
            ):
                # Determine gene names from input
                if gene_names is not None:
                    genes = gene_names
                elif hasattr(counts, "columns"):
                    genes = counts.columns
                else:
                    raise ValueError("Cannot determine gene names")

                n_genes = len(genes)
                rng = np.random.RandomState(42)

                return pd.DataFrame(
                    {
                        "pvalue": rng.uniform(0, 0.1, n_genes),
                        "padj": rng.uniform(0, 0.1, n_genes),
                        "stat": rng.uniform(1, 5, n_genes),
                        "log2FoldChange": rng.uniform(-2, 2, n_genes),
                        "stat_sign": rng.uniform(1, 5, n_genes),
                    },
                    index=genes,
                )

        def mock_get_engine(name):
            return MockEngine()

        monkeypatch.setattr("scbulkde.tl.tl_basic.get_engine_instance", mock_get_engine)
        return MockEngine()

    def test_direct_de_with_sufficient_samples(self, make_adata, mock_engine):
        """With ≥3 samples per condition, should use direct DE without fallback."""
        from scbulkde.tl.tl_basic import de

        # Create data with 3+ samples per condition (90 cells, 3 donors each condition)
        adata = make_adata(
            n_cells=180,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[90, 90],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30 + ["d3"] * 30) * 2,
        )

        result = de(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_samples=3,
            min_cells=10,
            engine="mock",
        )

        # Validate direct DE was used
        assert result.used_pseudoreplicates is False
        assert result.used_single_cell is False
        assert result.n_repetitions == 1
        assert result.engine == "mock"
        assert result.fallback_used is None

        # Validate result structure
        assert "pvalue" in result.results.columns
        assert "padj" in result.results.columns
        assert "log2FoldChange" in result.results.columns
        assert len(result.results) == 50  # n_genes

    def test_pseudoreplicate_fallback_insufficient_samples(self, make_adata, mock_engine):
        """With <3 samples per condition, should fall back to pseudoreplicates."""
        from scbulkde.tl.tl_basic import de

        # Create data with only 1 sample per condition
        adata = make_adata(
            n_cells=100,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[50, 50],
            replicate_key="donor",
            replicate_values=["d1"] * 50 + ["d2"] * 50,
        )

        result = de(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_samples=3,
            fallback_strategy="pseudoreplicates",
            n_repetitions=2,
            resampling_fraction=0.3,
            min_cells=10,
        )

        # Validate pseudoreplicate fallback was used
        assert result.used_pseudoreplicates is True
        assert result.used_single_cell is False
        assert result.n_repetitions == 2
        assert result.fallback_used == "pseudoreplicates"

        # Validate repetition results exist
        assert result.repetition_results is not None
        assert len(result.repetition_results) == 2

        # Each repetition should have results for all genes
        for _, rep_results in result.repetition_results.items():
            assert len(rep_results) == 50

    def test_single_cell_fallback(self, make_adata, mock_engine):
        """With fallback_strategy='single_cell', should run at single-cell level."""
        from scbulkde.tl.tl_basic import de

        adata = make_adata(
            n_cells=100,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[50, 50],
            replicate_key="donor",
            replicate_values=["d1"] * 50 + ["d2"] * 50,
        )

        result = de(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_samples=3,
            fallback_strategy="single_cell",
        )

        # Validate single-cell fallback was used
        assert result.used_single_cell is True
        assert result.used_pseudoreplicates is False
        assert result.n_repetitions == 1
        assert result.fallback_used == "single_cell"

        # Validate results
        assert len(result.results) == 50

    def test_raises_error_no_fallback_strategy(self, make_adata):
        """Should raise ValueError when insufficient samples and fallback_strategy=None."""
        from scbulkde.tl.tl_basic import de

        adata = make_adata(
            n_cells=100,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[50, 50],
            replicate_key="donor",
            replicate_values=["d1"] * 50 + ["d2"] * 50,
        )

        with pytest.raises(ValueError, match="Insufficient samples"):
            de(
                adata,
                group_key="cell_type",
                query="A",
                reference="B",
                replicate_key="donor",
                min_samples=3,
                fallback_strategy=None,
            )

    def test_accepts_pseudobulk_result_input(self, make_adata, mock_engine):
        """Should accept pre-computed PseudobulkResult as input."""
        from scbulkde.pp import pseudobulk
        from scbulkde.tl.tl_basic import de

        adata = make_adata(
            n_cells=180,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[90, 90],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30 + ["d3"] * 30) * 2,
        )

        # Pre-compute pseudobulk
        pb_result = pseudobulk(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_cells=10,
        )

        # Pass PseudobulkResult to de()
        result = de(pb_result, min_samples=3)

        # Validate result matches the input
        assert result.query == pb_result.query
        assert result.reference == pb_result.reference
        assert result.used_pseudoreplicates is False

    def test_uses_alpha_fallback_when_provided(self, make_adata, mock_engine):
        """Should use alpha_fallback for fallback strategies when specified."""
        from scbulkde.tl.tl_basic import de

        adata = make_adata(
            n_cells=100,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[50, 50],
            replicate_key="donor",
            replicate_values=["d1"] * 50 + ["d2"] * 50,
        )

        # Test with alpha_fallback different from alpha
        result = de(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_samples=3,
            alpha=0.05,
            alpha_fallback=0.01,  # More stringent for fallback
            fallback_strategy="single_cell",
        )

        # Should complete without error
        assert result.used_single_cell is True

    def test_no_strata_triggers_fallback(self, make_adata, mock_engine):
        """When no replicate_key provided, should use fallback strategies."""
        from scbulkde.tl.tl_basic import de

        adata = make_adata(
            n_cells=100,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[50, 50],
        )

        # No replicate_key means no strata, which means 0 samples
        result = de(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key=None,  # No strata
            min_samples=3,
            fallback_strategy="single_cell",
        )

        # Should use single-cell fallback
        assert result.used_single_cell is True


class TestRunDESingleCell:
    """Tests for single-cell DE execution logic."""

    def create_mock_pb_result(self, make_adata, sparse=False, layer_name=None):
        """Helper to create a minimal PseudobulkResult for testing."""
        from scbulkde.pp import pseudobulk

        adata = make_adata(
            n_cells=80,
            n_genes=30,
            groups=["A", "B"],
            group_counts=[40, 40],
            sparse=sparse,
            layer_name=layer_name,
        )

        return pseudobulk(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key=None,  # No replicates = collapsed case
            layer=layer_name,
        )

    def test_handles_sparse_csr_matrix(self, make_adata):
        """Should convert sparse CSR matrix to dense array before engine."""
        from scbulkde.tl.tl_basic import _run_de_single_cell

        counts_received = []

        class MockEngine:
            def run(
                self,
                counts,
                metadata,
                design_matrix,
                design_formula,
                alpha,
                correction_method,
                gene_names=None,
                **kwargs,
            ):
                # Capture what was passed to engine
                counts_received.append(counts)

                # Verify counts is dense numpy array, not sparse
                assert isinstance(counts, np.ndarray)
                assert not sp.issparse(counts)

                return pd.DataFrame(
                    {
                        "pvalue": [0.01] * len(gene_names),
                        "padj": [0.05] * len(gene_names),
                        "stat": [2.0] * len(gene_names),
                        "log2FoldChange": [1.0] * len(gene_names),
                        "stat_sign": [2.0] * len(gene_names),
                    },
                    index=gene_names,
                )

        pb_result = self.create_mock_pb_result(make_adata, sparse=True)

        # Verify input is actually sparse
        assert sp.issparse(pb_result.adata_sub.X)

        result = _run_de_single_cell(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            engine_name="mock",
            engine_kwargs={},
        )

        # Validate engine was called
        assert len(counts_received) == 1

        # Validate result structure
        assert result.used_single_cell is True
        assert result.used_pseudoreplicates is False
        assert len(result.results) == 30  # n_genes

    def test_handles_dense_matrix(self, make_adata):
        """Should handle dense matrices correctly."""
        from scbulkde.tl.tl_basic import _run_de_single_cell

        class MockEngine:
            def run(
                self,
                counts,
                metadata,
                design_matrix,
                design_formula,
                alpha,
                correction_method,
                gene_names=None,
                **kwargs,
            ):
                assert isinstance(counts, np.ndarray)
                assert not sp.issparse(counts)

                return pd.DataFrame(
                    {
                        "pvalue": [0.01] * len(gene_names),
                        "padj": [0.05] * len(gene_names),
                        "stat": [2.0] * len(gene_names),
                        "log2FoldChange": [1.0] * len(gene_names),
                        "stat_sign": [2.0] * len(gene_names),
                    },
                    index=gene_names,
                )

        pb_result = self.create_mock_pb_result(make_adata, sparse=False)

        result = _run_de_single_cell(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            engine_name="mock",
            engine_kwargs={},
        )

        assert result.used_single_cell is True

    def test_uses_layer_when_specified(self, make_adata):
        """Should extract expression from specified layer."""
        from scbulkde.tl.tl_basic import _run_de_single_cell

        layer_used = []

        class MockEngine:
            def run(
                self,
                counts,
                metadata,
                design_matrix,
                design_formula,
                alpha,
                correction_method,
                gene_names=None,
                **kwargs,
            ):
                # Store counts to verify layer was used
                layer_used.append(counts)

                return pd.DataFrame(
                    {
                        "pvalue": [0.01] * len(gene_names),
                        "padj": [0.05] * len(gene_names),
                        "stat": [2.0] * len(gene_names),
                        "log2FoldChange": [1.0] * len(gene_names),
                        "stat_sign": [2.0] * len(gene_names),
                    },
                    index=gene_names,
                )

        pb_result = self.create_mock_pb_result(make_adata, layer_name="test_layer")

        result = _run_de_single_cell(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            engine_name="mock",
            engine_kwargs={},
        )

        # Validate layer was accessed
        assert len(layer_used) == 1
        assert result.used_single_cell is True

    def test_creates_simple_design_formula(self, make_adata):
        """Should create a simple design formula for single-cell DE."""
        from scbulkde.tl.tl_basic import _run_de_single_cell

        design_formula_received = []

        class MockEngine:
            def run(
                self,
                counts,
                metadata,
                design_matrix,
                design_formula,
                alpha,
                correction_method,
                gene_names=None,
                **kwargs,
            ):
                design_formula_received.append(design_formula)

                return pd.DataFrame(
                    {
                        "pvalue": [0.01] * len(gene_names),
                        "padj": [0.05] * len(gene_names),
                        "stat": [2.0] * len(gene_names),
                        "log2FoldChange": [1.0] * len(gene_names),
                        "stat_sign": [2.0] * len(gene_names),
                    },
                    index=gene_names,
                )

        pb_result = self.create_mock_pb_result(make_adata)

        _ = _run_de_single_cell(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            engine_name="mock",
            engine_kwargs={},
        )

        # Verify simple design formula was created
        assert len(design_formula_received) == 1
        assert "psbulk_condition" in design_formula_received[0]
        assert "reference" in design_formula_received[0]


class TestRunDEPseudoreplicates:
    """Tests for pseudoreplicate DE execution and aggregation."""

    def test_runs_multiple_repetitions(self, make_adata):
        """Should execute DE n_repetitions times."""
        from scbulkde.pp import pseudobulk
        from scbulkde.tl.tl_basic import _run_de_pseudoreplicates

        call_count = []

        class MockEngine:
            def run(self, counts, metadata, design_matrix, design_formula, alpha, correction_method, **kwargs):
                call_count.append(1)
                genes = counts.columns
                return pd.DataFrame(
                    {
                        "pvalue": np.full(len(genes), 0.01),
                        "padj": np.full(len(genes), 0.05),
                        "stat": np.full(len(genes), 2.0),
                        "log2FoldChange": np.full(len(genes), 1.0),
                        "stat_sign": np.full(len(genes), 2.0),
                    },
                    index=genes,
                )

        adata = make_adata(
            n_cells=120,
            n_genes=20,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        pb_result = pseudobulk(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_cells=10,
        )

        n_reps = 3
        result = _run_de_pseudoreplicates(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            required_samples={"query": 1, "reference": 1},
            n_repetitions=n_reps,
            resampling_fraction=0.3,
            rng=np.random.default_rng(42),
            engine_name="mock",
            engine_kwargs={},
        )

        # Should call engine exactly n_repetitions times
        assert len(call_count) == n_reps

        # Should store per-repetition results
        assert len(result.repetition_results) == n_reps

        # Result metadata should reflect pseudoreplicate usage
        assert result.used_pseudoreplicates is True
        assert result.n_repetitions == n_reps

    def test_aggregates_results_across_repetitions(self, make_adata):
        """Should average results across repetitions."""
        from scbulkde.pp import pseudobulk
        from scbulkde.tl.tl_basic import _run_de_pseudoreplicates

        class MockEngine:
            def run(self, counts, metadata, design_matrix, design_formula, alpha, correction_method, **kwargs):
                genes = counts.columns
                # Return different values each call
                offset = len(genes) * 0.1
                return pd.DataFrame(
                    {
                        "pvalue": np.arange(len(genes)) * offset,
                        "padj": np.arange(len(genes)) * offset + 0.01,
                        "stat": np.arange(len(genes)) * offset + 1.0,
                        "log2FoldChange": np.arange(len(genes)) * offset - 1.0,
                        "stat_sign": np.arange(len(genes)) * offset + 0.5,
                    },
                    index=genes,
                )

        adata = make_adata(
            n_cells=120,
            n_genes=20,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        pb_result = pseudobulk(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_cells=10,
        )

        result = _run_de_pseudoreplicates(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            required_samples={"query": 1, "reference": 1},
            n_repetitions=2,
            resampling_fraction=0.3,
            rng=np.random.default_rng(42),
            engine_name="mock",
            engine_kwargs={},
        )

        # Final results should exist and be averaged
        assert len(result.results) == 20
        assert all(col in result.results.columns for col in ["pvalue", "padj", "stat"])

    def test_combines_pseudoreplicates_with_existing_samples(self, make_adata):
        """Should combine generated pseudoreplicates with existing samples."""
        from scbulkde.pp import pseudobulk
        from scbulkde.tl.tl_basic import _run_de_pseudoreplicates

        metadata_sizes = []

        class MockEngine:
            def run(self, counts, metadata, design_matrix, design_formula, alpha, correction_method, **kwargs):
                # Track how many samples are in each call
                metadata_sizes.append(len(metadata))
                genes = counts.columns
                return pd.DataFrame(
                    {
                        "pvalue": np.full(len(genes), 0.01),
                        "padj": np.full(len(genes), 0.05),
                        "stat": np.full(len(genes), 2.0),
                        "log2FoldChange": np.full(len(genes), 1.0),
                        "stat_sign": np.full(len(genes), 2.0),
                    },
                    index=genes,
                )

        adata = make_adata(
            n_cells=120,
            n_genes=20,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )

        pb_result = pseudobulk(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_cells=10,
        )

        # We have 2 existing samples, need 1 more per condition
        _ = _run_de_pseudoreplicates(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            required_samples={"query": 1, "reference": 1},
            n_repetitions=1,
            resampling_fraction=0.3,
            rng=np.random.default_rng(42),
            engine_name="mock",
            engine_kwargs={},
        )

        # Should have combined existing + pseudoreplicates
        # Original: 2 samples per condition = 4 total
        # Added: 1 pseudoreplicate per condition = 2 total
        # Expected: 6 samples
        assert metadata_sizes[0] > len(pb_result.pb_counts)

    def test_handles_collapsed_case(self, make_adata):
        """Should handle collapsed case (no existing samples)."""
        from scbulkde.pp import pseudobulk
        from scbulkde.tl.tl_basic import _run_de_pseudoreplicates

        class MockEngine:
            def run(self, counts, metadata, design_matrix, design_formula, alpha, correction_method, **kwargs):
                genes = counts.columns
                return pd.DataFrame(
                    {
                        "pvalue": np.full(len(genes), 0.01),
                        "padj": np.full(len(genes), 0.05),
                        "stat": np.full(len(genes), 2.0),
                        "log2FoldChange": np.full(len(genes), 1.0),
                        "stat_sign": np.full(len(genes), 2.0),
                    },
                    index=genes,
                )

        adata = make_adata(
            n_cells=100,
            n_genes=20,
            groups=["A", "B"],
            group_counts=[50, 50],
        )

        # No replicate_key = collapsed case
        pb_result = pseudobulk(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key=None,
            min_cells=10,
        )

        # Should have 0 existing samples
        assert len(pb_result.pb_counts) == 0

        result = _run_de_pseudoreplicates(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            required_samples={"query": 3, "reference": 3},
            n_repetitions=2,
            resampling_fraction=0.3,
            rng=np.random.default_rng(42),
            engine_name="mock",
            engine_kwargs={},
        )

        # Should successfully generate pseudoreplicates
        assert result.used_pseudoreplicates is True
        assert len(result.results) == 20


class TestRunDEDirect:
    """Tests for direct DE execution on pseudobulk samples."""

    def test_returns_correct_result_structure(self, make_adata):
        """Should return DEResult with correct metadata."""
        from scbulkde.pp import pseudobulk
        from scbulkde.tl.tl_basic import _run_de_direct

        class MockEngine:
            def run(self, counts, metadata, design_matrix, design_formula, alpha, correction_method, **kwargs):
                genes = counts.columns
                return pd.DataFrame(
                    {
                        "pvalue": np.full(len(genes), 0.01),
                        "padj": np.full(len(genes), 0.05),
                        "stat": np.full(len(genes), 2.0),
                        "log2FoldChange": np.full(len(genes), 1.0),
                        "stat_sign": np.full(len(genes), 2.0),
                    },
                    index=genes,
                )

        adata = make_adata(
            n_cells=180,
            n_genes=30,
            groups=["A", "B"],
            group_counts=[90, 90],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30 + ["d3"] * 30) * 2,
        )

        pb_result = pseudobulk(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_cells=10,
        )

        result = _run_de_direct(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            engine_name="mock",
            engine_kwargs={},
        )

        # Validate DEResult structure
        assert result.used_pseudoreplicates is False
        assert result.used_single_cell is False
        assert result.n_repetitions == 1
        assert result.engine == "mock"
        assert result.query == pb_result.query
        assert result.reference == pb_result.reference
        assert result.design == pb_result.design_formula

        # Validate results DataFrame
        assert len(result.results) == 30
        assert all(col in result.results.columns for col in ["pvalue", "padj", "stat"])

    def test_passes_engine_kwargs(self, make_adata):
        """Should pass engine_kwargs to the DE engine."""
        from scbulkde.pp import pseudobulk
        from scbulkde.tl.tl_basic import _run_de_direct

        kwargs_received = []

        class MockEngine:
            def run(self, counts, metadata, design_matrix, design_formula, alpha, correction_method, **kwargs):
                kwargs_received.append(kwargs)
                genes = counts.columns
                return pd.DataFrame(
                    {
                        "pvalue": np.full(len(genes), 0.01),
                        "padj": np.full(len(genes), 0.05),
                        "stat": np.full(len(genes), 2.0),
                        "log2FoldChange": np.full(len(genes), 1.0),
                        "stat_sign": np.full(len(genes), 2.0),
                    },
                    index=genes,
                )

        adata = make_adata(
            n_cells=180,
            n_genes=30,
            groups=["A", "B"],
            group_counts=[90, 90],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30 + ["d3"] * 30) * 2,
        )

        pb_result = pseudobulk(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key="donor",
            min_cells=10,
        )

        custom_kwargs = {"custom_param": 42}

        _ = _run_de_direct(
            pb_result=pb_result,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=MockEngine(),
            engine_name="mock",
            engine_kwargs=custom_kwargs,
        )

        # Verify custom kwargs were passed
        assert len(kwargs_received) == 1
        assert "custom_param" in kwargs_received[0]
        assert kwargs_received[0]["custom_param"] == 42
