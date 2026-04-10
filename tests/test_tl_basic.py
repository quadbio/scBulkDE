"""Tests for scbulkde.tl.tl_basic module."""

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
    _run_de_direct,
    _run_de_pseudoreplicates,
    _run_de_single_cell,
    de,
)
from scbulkde.ut._containers import PseudobulkResult


class TestCountExistingSamples:
    """Test _count_existing_samples: counts pseudobulk samples per condition."""

    def test_multiindex_groupby_counts_strata(self):
        """MultiIndex groupby should count unique strata combinations per condition."""
        index = pd.MultiIndex.from_tuples(
            [("query", "rep1"), ("query", "rep2"), ("reference", "rep1")],
            names=["condition", "replicate"],
        )
        obs = pd.DataFrame({"condition": ["query", "query", "reference"]}, index=index)
        grouped = obs.groupby(level=["condition", "replicate"])

        result = _count_existing_samples(grouped)

        assert result == {"query": 2, "reference": 1}

    def test_single_index_returns_zero_samples(self):
        """Single index (collapsed case) should return 0 for both conditions."""
        obs = pd.DataFrame({"condition": ["query", "query", "reference", "reference"]})
        grouped = obs.groupby("condition")

        result = _count_existing_samples(grouped)

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

    @pytest.fixture
    def pb_result_100(self, make_cell_pool):
        """PseudobulkResult with 100 cells for cache tests."""
        return _make_minimal_pb_result(n_cells=100)

    @pytest.fixture
    def pb_result_50(self):
        """PseudobulkResult with 50 cells for cache tests."""
        return _make_minimal_pb_result(n_cells=50)

    def test_cache_separates_conditions(self, pb_result_100):
        """Cache should organize cells into query and reference pools."""
        rng = np.random.default_rng(42)

        cache = _build_cell_pool_cache(pb_result_100, rng, shuffle=False)

        assert "query" in cache
        assert "reference" in cache
        assert len(cache["query"]) > 0
        assert len(cache["reference"]) > 0

        for condition in ["query", "reference"]:
            for size, indices in cache[condition]:
                assert isinstance(size, (int, np.integer))
                assert isinstance(indices, np.ndarray)
                assert len(indices) == size

    def test_cache_preserves_cell_information(self, pb_result_50):
        """Cache should preserve cell indices and counts correctly."""
        rng = np.random.default_rng(42)

        cache = _build_cell_pool_cache(pb_result_50, rng, shuffle=False)

        all_cached_cells = []
        for condition in ["query", "reference"]:
            for _, indices in cache[condition]:
                all_cached_cells.extend(indices)

        assert len(all_cached_cells) == len(pb_result_50.adata_sub.obs)

    def test_cache_respects_grouping_structure(self, pb_result_100):
        """Each sample in cache should correspond to a unique group."""
        rng = np.random.default_rng(42)

        cache = _build_cell_pool_cache(pb_result_100, rng, shuffle=False)

        n_query_samples = len(cache["query"])
        n_ref_samples = len(cache["reference"])

        existing = _count_existing_samples(pb_result_100.grouped)
        assert n_query_samples == existing["query"]
        assert n_ref_samples == existing["reference"]


class TestGeneratePseudoreplicate:
    """Test _generate_pseudoreplicate: greedy cell usage minimization."""

    def test_selects_least_used_sample(self, make_cell_pool):
        """Should select sample with minimum total cell usage."""
        cell_pool, cell_usage = make_cell_pool(n_cells_per_sample=20)

        # Mark second sample as heavily used
        for cell_id in cell_pool["query"][1][1]:
            cell_usage[cell_id] = 5

        rng = np.random.default_rng(42)

        selected_samples = []
        for _ in range(3):
            pr_indices = _generate_pseudoreplicate(
                condition="query",
                cell_pool_cache=cell_pool,
                cell_usage_tracker=cell_usage,
                resampling_fraction=0.5,
                rng=rng,
            )
            selected_samples.append(set(pr_indices))

        sample_1_cells = set(cell_pool["query"][1][1])
        from_sample_1 = sum(len(s & sample_1_cells) for s in selected_samples)

        total_selected = sum(len(s) for s in selected_samples)
        assert from_sample_1 < total_selected * 0.5

    def test_respects_resampling_fraction(self, make_cell_pool):
        """Generated pseudoreplicate size should match resampling_fraction."""
        cell_pool, cell_usage = make_cell_pool(n_cells_per_sample=100)
        rng = np.random.default_rng(42)

        resampling_frac = 0.33
        pr_indices = _generate_pseudoreplicate(
            condition="query",
            cell_pool_cache=cell_pool,
            cell_usage_tracker=cell_usage,
            resampling_fraction=resampling_frac,
            rng=rng,
        )

        expected_size = int(100 * resampling_frac)
        assert len(pr_indices) == expected_size

    def test_updates_usage_tracker(self, make_cell_pool):
        """Should increment usage count for selected cells."""
        cell_pool, cell_usage = make_cell_pool(n_cells_per_sample=50)
        rng = np.random.default_rng(42)

        initial_usage = cell_usage.copy()

        pr_indices = _generate_pseudoreplicate(
            condition="query",
            cell_pool_cache=cell_pool,
            cell_usage_tracker=cell_usage,
            resampling_fraction=0.4,
            rng=rng,
        )

        for cell_id in pr_indices:
            assert cell_usage[cell_id] == initial_usage[cell_id] + 1

        all_cells = set(np.concatenate([indices for _, indices in cell_pool["query"]]))
        non_selected = all_cells - set(pr_indices)
        for cell_id in non_selected:
            assert cell_usage[cell_id] == initial_usage[cell_id]

    def test_prefers_least_used_cells_within_sample(self, make_cell_pool):
        """Within a sample, should prefer cells with lowest usage."""
        cell_pool, cell_usage = make_cell_pool(n_cells_per_sample=30)

        sample_0_cells = cell_pool["query"][0][1]
        heavily_used_cells = sample_0_cells[:10]
        for cell_id in heavily_used_cells:
            cell_usage[cell_id] = 10

        rng = np.random.default_rng(42)

        pr_indices = _generate_pseudoreplicate(
            condition="query",
            cell_pool_cache=cell_pool,
            cell_usage_tracker=cell_usage,
            resampling_fraction=0.5,
            rng=rng,
        )

        if any(cell_id in sample_0_cells for cell_id in pr_indices):
            from_sample_0 = set(pr_indices) & set(sample_0_cells)
            unused_in_sample_0 = set(sample_0_cells[10:])
            assert len(from_sample_0 & unused_in_sample_0) > len(from_sample_0) * 0.7

    def test_handles_exhausted_least_used_tier(self, make_cell_pool):
        """Should move to next usage tier if least-used cells insufficient."""
        cell_pool, cell_usage = make_cell_pool(n_cells_per_sample=50)

        sample_0_cells = cell_pool["query"][0][1]
        for cell_id in sample_0_cells[5:]:
            cell_usage[cell_id] = 1

        for i in range(1, len(cell_pool["query"])):
            for cell_id in cell_pool["query"][i][1]:
                cell_usage[cell_id] = 1

        rng = np.random.default_rng(42)

        pr_indices = _generate_pseudoreplicate(
            condition="query",
            cell_pool_cache=cell_pool,
            cell_usage_tracker=cell_usage,
            resampling_fraction=0.5,
            rng=rng,
        )

        assert len(pr_indices) == 25

        sample_0_set = set(sample_0_cells)
        assert set(pr_indices).issubset(sample_0_set)

        least_used = set(sample_0_cells[:5])
        next_tier = set(sample_0_cells[5:])

        pr_set = set(pr_indices)
        assert least_used.issubset(pr_set)
        assert len(pr_set & next_tier) == 20


class TestPseudoreplicateIndependence:
    """Test that pseudoreplicate generation maintains cell independence."""

    def test_cell_usage_tracked_across_iterations(self, make_cell_pool):
        """Cell usage should accumulate across multiple pseudoreplicate generations."""
        cell_pool, cell_usage = make_cell_pool(n_cells_per_sample=50)
        rng = np.random.default_rng(42)

        for _ in range(3):
            _generate_pseudoreplicate(
                condition="query",
                cell_pool_cache=cell_pool,
                cell_usage_tracker=cell_usage,
                resampling_fraction=0.3,
                rng=rng,
            )

        usage_values = list(cell_usage.values())
        mean_usage = np.mean(usage_values)
        assert mean_usage == 0.18

    def test_diversity_across_iterations(self, make_cell_pool):
        """Different iterations should sample different sets of cells."""
        cell_pool, cell_usage = make_cell_pool(n_cells_per_sample=100)
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

        overlaps = []
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                overlap = len(samples[i] & samples[j]) / len(samples[i])
                overlaps.append(overlap)

        mean_overlap = np.mean(overlaps)
        assert mean_overlap < 0.4


class TestDEFunction:
    """Integration tests for the main de() entry point."""

    @pytest.fixture
    def mock_engine(self, make_mock_engine, monkeypatch):
        """Patch get_engine_instance to return a MockEngine."""
        engine = make_mock_engine()

        def mock_get_engine(name):
            return engine

        monkeypatch.setattr("scbulkde.tl.tl_basic.get_engine_instance", mock_get_engine)
        return engine

    def test_direct_de_with_sufficient_samples(self, make_adata, mock_engine):
        """With ≥3 samples per condition, should use direct DE without fallback."""
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

        assert result.used_pseudoreplicates is False
        assert result.used_single_cell is False
        assert result.n_repetitions == 1
        assert result.engine == "mock"
        assert result.fallback_used is None
        assert "pvalue" in result.results.columns
        assert "padj" in result.results.columns
        assert "log2FoldChange" in result.results.columns
        assert len(result.results) == 50

    def test_pseudoreplicate_fallback_insufficient_samples(self, make_adata, mock_engine):
        """With <3 samples per condition, should fall back to pseudoreplicates."""
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

        assert result.used_pseudoreplicates is True
        assert result.used_single_cell is False
        assert result.n_repetitions == 2
        assert result.fallback_used == "pseudoreplicates"
        assert result.repetition_results is not None
        assert len(result.repetition_results) == 2
        for _, rep_results in result.repetition_results.items():
            assert len(rep_results) == 50

    def test_single_cell_fallback(self, make_adata, mock_engine):
        """With fallback_strategy='single_cell', should run at single-cell level."""
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

        assert result.used_single_cell is True
        assert result.used_pseudoreplicates is False
        assert result.n_repetitions == 1
        assert result.fallback_used == "single_cell"
        assert len(result.results) == 50

    def test_raises_error_no_fallback_strategy(self, make_adata):
        """Should raise ValueError when insufficient samples and fallback_strategy=None."""
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

        adata = make_adata(
            n_cells=180,
            n_genes=50,
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

        result = de(pb_result, min_samples=3)

        assert result.query == pb_result.query
        assert result.reference == pb_result.reference
        assert result.used_pseudoreplicates is False

    def test_uses_alpha_fallback_when_provided(self, make_adata, mock_engine):
        """Should use alpha_fallback for fallback strategies when specified."""
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
            alpha=0.05,
            alpha_fallback=0.01,
            fallback_strategy="single_cell",
        )

        assert result.used_single_cell is True

    def test_no_strata_triggers_fallback(self, make_adata, mock_engine):
        """When no replicate_key provided, should use fallback strategies."""
        adata = make_adata(
            n_cells=100,
            n_genes=50,
            groups=["A", "B"],
            group_counts=[50, 50],
        )

        result = de(
            adata,
            group_key="cell_type",
            query="A",
            reference="B",
            replicate_key=None,
            min_samples=3,
            fallback_strategy="single_cell",
        )

        assert result.used_single_cell is True


class TestRunDESingleCell:
    """Tests for single-cell DE execution logic."""

    @pytest.fixture
    def pb_result_dense(self, make_adata):
        """PseudobulkResult from dense matrix (no replicates = collapsed)."""
        from scbulkde.pp import pseudobulk

        adata = make_adata(
            n_cells=80,
            n_genes=30,
            groups=["A", "B"],
            group_counts=[40, 40],
            sparse=False,
        )
        return pseudobulk(adata, group_key="cell_type", query="A", reference="B", replicate_key=None)

    @pytest.fixture
    def pb_result_sparse(self, make_adata):
        """PseudobulkResult from sparse matrix."""
        from scbulkde.pp import pseudobulk

        adata = make_adata(
            n_cells=80,
            n_genes=30,
            groups=["A", "B"],
            group_counts=[40, 40],
            sparse=True,
        )
        return pseudobulk(adata, group_key="cell_type", query="A", reference="B", replicate_key=None)

    @pytest.fixture
    def pb_result_with_layer(self, make_adata):
        """PseudobulkResult from data with a named layer."""
        from scbulkde.pp import pseudobulk

        adata = make_adata(
            n_cells=80,
            n_genes=30,
            groups=["A", "B"],
            group_counts=[40, 40],
            layer_name="test_layer",
        )
        return pseudobulk(
            adata, group_key="cell_type", query="A", reference="B", replicate_key=None, layer="test_layer"
        )

    def test_handles_sparse_csr_matrix(self, pb_result_sparse, make_mock_engine):
        """Should convert sparse CSR matrix to dense array before engine."""
        counts_received = []
        engine = make_mock_engine(capture=counts_received)

        assert sp.issparse(pb_result_sparse.adata_sub.X)

        result = _run_de_single_cell(
            pb_result=pb_result_sparse,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            engine_name="mock",
            engine_kwargs={},
        )

        assert len(counts_received) == 1
        assert isinstance(counts_received[0]["counts"], np.ndarray)
        assert not sp.issparse(counts_received[0]["counts"])
        assert result.used_single_cell is True
        assert result.used_pseudoreplicates is False
        assert len(result.results) == 30

    def test_handles_dense_matrix(self, pb_result_dense, make_mock_engine):
        """Should handle dense matrices correctly."""
        engine = make_mock_engine()

        result = _run_de_single_cell(
            pb_result=pb_result_dense,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            engine_name="mock",
            engine_kwargs={},
        )

        assert result.used_single_cell is True

    def test_uses_layer_when_specified(self, pb_result_with_layer, make_mock_engine):
        """Should extract expression from specified layer."""
        captured = []
        engine = make_mock_engine(capture=captured)

        result = _run_de_single_cell(
            pb_result=pb_result_with_layer,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            engine_name="mock",
            engine_kwargs={},
        )

        assert len(captured) == 1
        assert result.used_single_cell is True

    def test_creates_simple_design_formula(self, pb_result_dense, make_mock_engine):
        """Should create a simple design formula for single-cell DE."""
        captured = []
        engine = make_mock_engine(capture=captured)

        _run_de_single_cell(
            pb_result=pb_result_dense,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            engine_name="mock",
            engine_kwargs={},
        )

        assert len(captured) == 1
        assert "psbulk_condition" in captured[0]["design_formula"]
        assert "reference" in captured[0]["design_formula"]


class TestRunDEPseudoreplicates:
    """Tests for pseudoreplicate DE execution and aggregation."""

    @pytest.fixture
    def pb_result_with_replicates(self, make_adata):
        """PseudobulkResult with two donors per condition."""
        from scbulkde.pp import pseudobulk

        adata = make_adata(
            n_cells=120,
            n_genes=20,
            groups=["A", "B"],
            group_counts=[60, 60],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30) * 2,
        )
        return pseudobulk(adata, group_key="cell_type", query="A", reference="B", replicate_key="donor", min_cells=10)

    @pytest.fixture
    def pb_result_collapsed(self, make_adata):
        """PseudobulkResult with no replicates (collapsed case)."""
        from scbulkde.pp import pseudobulk

        adata = make_adata(
            n_cells=100,
            n_genes=20,
            groups=["A", "B"],
            group_counts=[50, 50],
        )
        return pseudobulk(adata, group_key="cell_type", query="A", reference="B", replicate_key=None, min_cells=10)

    def test_runs_multiple_repetitions(self, pb_result_with_replicates, make_mock_engine):
        """Should execute DE n_repetitions times."""
        call_count = []
        engine = make_mock_engine(capture=call_count)

        n_reps = 3
        result = _run_de_pseudoreplicates(
            pb_result=pb_result_with_replicates,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            required_samples={"query": 1, "reference": 1},
            n_repetitions=n_reps,
            resampling_fraction=0.3,
            rng=np.random.default_rng(42),
            engine_name="mock",
            engine_kwargs={},
        )

        assert len(call_count) == n_reps
        assert len(result.repetition_results) == n_reps
        assert result.used_pseudoreplicates is True
        assert result.n_repetitions == n_reps

    def test_aggregates_results_across_repetitions(self, pb_result_with_replicates, make_mock_engine):
        """Should average results across repetitions."""
        engine = make_mock_engine()

        result = _run_de_pseudoreplicates(
            pb_result=pb_result_with_replicates,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            required_samples={"query": 1, "reference": 1},
            n_repetitions=2,
            resampling_fraction=0.3,
            rng=np.random.default_rng(42),
            engine_name="mock",
            engine_kwargs={},
        )

        assert len(result.results) == 20
        assert all(col in result.results.columns for col in ["pvalue", "padj", "stat"])

    def test_combines_pseudoreplicates_with_existing_samples(self, pb_result_with_replicates, make_mock_engine):
        """Should combine generated pseudoreplicates with existing samples."""
        metadata_sizes = []
        engine = make_mock_engine(capture=metadata_sizes)

        _ = _run_de_pseudoreplicates(
            pb_result=pb_result_with_replicates,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            required_samples={"query": 1, "reference": 1},
            n_repetitions=1,
            resampling_fraction=0.3,
            rng=np.random.default_rng(42),
            engine_name="mock",
            engine_kwargs={},
        )

        assert metadata_sizes[0]["metadata"] is not None
        # Combined samples > original samples
        assert len(metadata_sizes[0]["metadata"]) > len(pb_result_with_replicates.pb_counts)

    def test_handles_collapsed_case(self, pb_result_collapsed, make_mock_engine):
        """Should handle collapsed case (no existing samples)."""
        engine = make_mock_engine()

        assert len(pb_result_collapsed.pb_counts) == 0

        result = _run_de_pseudoreplicates(
            pb_result=pb_result_collapsed,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            required_samples={"query": 3, "reference": 3},
            n_repetitions=2,
            resampling_fraction=0.3,
            rng=np.random.default_rng(42),
            engine_name="mock",
            engine_kwargs={},
        )

        assert result.used_pseudoreplicates is True
        assert len(result.results) == 20


class TestRunDEDirect:
    """Tests for direct DE execution on pseudobulk samples."""

    @pytest.fixture
    def pb_result_3donors(self, make_adata):
        """PseudobulkResult with 3 donors per condition."""
        from scbulkde.pp import pseudobulk

        adata = make_adata(
            n_cells=180,
            n_genes=30,
            groups=["A", "B"],
            group_counts=[90, 90],
            replicate_key="donor",
            replicate_values=(["d1"] * 30 + ["d2"] * 30 + ["d3"] * 30) * 2,
        )
        return pseudobulk(adata, group_key="cell_type", query="A", reference="B", replicate_key="donor", min_cells=10)

    def test_returns_correct_result_structure(self, pb_result_3donors, make_mock_engine):
        """Should return DEResult with correct metadata."""
        engine = make_mock_engine()

        result = _run_de_direct(
            pb_result=pb_result_3donors,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            engine_name="mock",
            engine_kwargs={},
        )

        assert result.used_pseudoreplicates is False
        assert result.used_single_cell is False
        assert result.n_repetitions == 1
        assert result.engine == "mock"
        assert result.query == pb_result_3donors.query
        assert result.reference == pb_result_3donors.reference
        assert result.design == pb_result_3donors.design_formula
        assert len(result.results) == 30
        assert all(col in result.results.columns for col in ["pvalue", "padj", "stat"])

    def test_passes_engine_kwargs(self, pb_result_3donors, make_mock_engine):
        """Should pass engine_kwargs to the DE engine."""
        captured = []
        engine = make_mock_engine(capture=captured)

        custom_kwargs = {"custom_param": 42}

        _ = _run_de_direct(
            pb_result=pb_result_3donors,
            alpha=0.05,
            correction_method="fdr_bh",
            de_engine=engine,
            engine_name="mock",
            engine_kwargs=custom_kwargs,
        )

        assert len(captured) == 1
        assert "custom_param" in captured[0]
        assert captured[0]["custom_param"] == 42


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _make_minimal_pb_result(n_cells: int = 100, n_genes: int = 50) -> PseudobulkResult:
    """Create a minimal PseudobulkResult with condition + replicate grouping."""
    X = sp.random(n_cells, n_genes, density=0.1, format="csr")
    adata = AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]

    n_query = n_cells // 2
    conditions = ["query"] * n_query + ["reference"] * (n_cells - n_query)
    replicates = ["rep1", "rep2"] * (n_cells // 2)
    if n_cells % 2 == 1:
        replicates.append("rep1")

    adata.obs["condition"] = conditions
    adata.obs["replicate"] = replicates

    obs_grouped = adata.obs.groupby(["condition", "replicate"], observed=True, sort=False)

    return PseudobulkResult(
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
