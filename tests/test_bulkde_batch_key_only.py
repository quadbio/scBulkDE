import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scbulkde as scb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def make_test_adata_batch_only(
    cells_per_group: dict[str, dict[str, int]],
    n_genes: int = 100,
    seed: int = 42,
) -> ad.AnnData:
    import anndata as ad

    rng = np.random.default_rng(seed)
    obs_records = []
    cell_idx = 0
    for condition, batches in cells_per_group.items():
        for batch_id, n_cells in batches.items():
            for _ in range(n_cells):
                obs_records.append(
                    {
                        "cell_id": f"cell_{cell_idx}",
                        "condition": condition,
                        "batch": batch_id,
                    }
                )
                cell_idx += 1
    obs = pd.DataFrame(obs_records).set_index("cell_id")
    n_cells_total = len(obs)
    counts = rng.poisson(lam=5, size=(n_cells_total, n_genes))
    adata = ad.AnnData(
        X=sp.csr_matrix(counts),
        obs=obs,
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    adata.layers["counts"] = adata.X.copy()
    return adata


# ===========================
# BATCH-KEY-ONLY SCENARIOS
# ===========================


def _run_pseudobulk_and_de(adata, scenario_name, verbose=True):
    # Parameters tuned to batch_key_only scenario
    group_key = "condition"
    query = "disease"
    reference = "healthy"
    layer = "counts"
    replicate_key = None  # No replicate key
    batch_key = "batch"
    pb_kwargs = {
        "group_key": group_key,
        "query": query,
        "reference": reference,
        "layer": layer,
        "replicate_key": replicate_key,
        "batch_key": batch_key,
        "min_cells": 50,
        "min_fraction": 0.3,
        "min_coverage": 0.8,
        "min_bridging_batches": 2,
        "mode": "sum",
    }
    de_kwargs = {"min_samples": 3, "n_repetitions": 5, "resampling_fraction": 0.7, "min_list_overlap": 0.7, "n_cpus": 1}
    try:
        pb_result = scb.pp.pseudobulk(adata, **pb_kwargs)
        de_result = scb.tl.de(pb_result, **de_kwargs)
        if verbose:
            print(f"[{scenario_name}] DESIGN: {pb_result.design}")
            print(f"[{scenario_name}] Include batch: {getattr(pb_result, 'include_batch', None)}")
            if hasattr(pb_result, "collapsed_conditions"):
                print(f"[{scenario_name}] Collapsed conditions: {pb_result.collapsed_conditions}")
            if hasattr(pb_result, "bridging_batches"):
                print(f"[{scenario_name}] Bridging batches: {pb_result.bridging_batches}")
            print(
                f"[{scenario_name}] DE result shape: {getattr(de_result, 'shape', de_result.shape if hasattr(de_result, 'shape') else '<unknown>')}"
            )
            print("SUCCESS")
        return pb_result, de_result
    except Exception as e:  # noqa: BLE001
        print(f"[{scenario_name}] FAILED: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def scenario_1_ideal_bridging():
    "Multiple batches, all bridging both conditions"
    cells_per_group = {
        "disease": {"batch_1": 200, "batch_2": 180, "batch_3": 220},
        "healthy": {"batch_1": 190, "batch_2": 210, "batch_3": 175},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_2_partial_bridging():
    "Some batches bridge, others are condition-specific"
    cells_per_group = {
        "disease": {"batch_1": 200, "batch_2": 180, "batch_3": 150},
        "healthy": {"batch_1": 190, "batch_2": 210, "batch_4": 175},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_3_insufficient_bridging():
    "Only one bridging batch (below min_bridging_batches=2)"
    cells_per_group = {
        "disease": {"batch_1": 200, "batch_2": 180, "batch_3": 150},
        "healthy": {"batch_1": 190, "batch_4": 210, "batch_5": 175},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_4_no_bridging():
    "No bridging batches at all; all batches are condition-specific"
    cells_per_group = {
        "disease": {"batch_1": 200, "batch_2": 180, "batch_3": 150},
        "healthy": {"batch_4": 190, "batch_5": 210, "batch_6": 175},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_5_some_small_batches():
    "Some batches are too small, but coverage is sufficient"
    cells_per_group = {
        "disease": {"batch_1": 200, "batch_2": 180, "batch_3": 10},
        "healthy": {"batch_1": 190, "batch_2": 15, "batch_4": 175},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_6_small_batches_break_bridging():
    "Bridging batches exist but are too small, so after filtering, no bridging."
    cells_per_group = {
        "disease": {"batch_1": 15, "batch_2": 20, "batch_3": 200},
        "healthy": {"batch_1": 10, "batch_2": 12, "batch_4": 180},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_7_collapse_one_condition():
    "One condition's batches all too small; gets collapsed"
    cells_per_group = {
        "disease": {"batch_1": 20, "batch_2": 25, "batch_3": 30},
        "healthy": {"batch_1": 200, "batch_2": 180, "batch_3": 190},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_8_collapse_both_conditions():
    "Both conditions' batches all too small, both get collapsed"
    cells_per_group = {
        "disease": {"batch_1": 20, "batch_2": 25, "batch_3": 30},
        "healthy": {"batch_4": 15, "batch_5": 20, "batch_6": 25},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_9_single_batch_per_condition():
    "Case A: Same batch bridges; Case B: Different batches (no bridge)"
    cells_per_group_bridging = {
        "disease": {"batch_1": 500},
        "healthy": {"batch_1": 480},
    }
    cells_per_group_no_bridge = {
        "disease": {"batch_1": 500},
        "healthy": {"batch_2": 480},
    }
    return (
        make_test_adata_batch_only(cells_per_group_bridging),
        make_test_adata_batch_only(cells_per_group_no_bridge),
    )


def scenario_10_insufficient_coverage():
    "Valid batches exist, but not enough coverage for one condition, so collapse"
    cells_per_group = {
        "disease": {"batch_1": 100, "batch_2": 30, "batch_3": 25, "batch_4": 20, "batch_5": 25},
        "healthy": {"batch_1": 200, "batch_6": 180, "batch_7": 170},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_11_asymmetric_batch_counts():
    "One condition has many batches, the other has few."
    cells_per_group = {
        "disease": {"batch_1": 100, "batch_2": 120, "batch_3": 110, "batch_4": 130, "batch_5": 105, "batch_6": 115},
        "healthy": {"batch_1": 300, "batch_2": 280},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_12_fraction_based_validity():
    "Small batches are valid due to meeting fraction threshold"
    cells_per_group = {
        "disease": {"batch_1": 40, "batch_2": 35, "batch_3": 25},
        "healthy": {"batch_1": 45, "batch_2": 45},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_13_many_bridging_batches():
    "Stress test: many bridging batches"
    n_batches = 10
    cells_per_group = {
        "disease": {f"batch_{i}": 100 + i * 10 for i in range(1, n_batches + 1)},
        "healthy": {f"batch_{i}": 95 + i * 10 for i in range(1, n_batches + 1)},
    }
    return make_test_adata_batch_only(cells_per_group)


def scenario_14_edge_min_bridging():
    "Exactly at min_bridging_batches threshold"
    cells_per_group = {
        "disease": {"batch_1": 200, "batch_2": 180, "batch_3": 150, "batch_4": 160},
        "healthy": {"batch_1": 190, "batch_2": 210, "batch_5": 175, "batch_6": 165},
    }
    return make_test_adata_batch_only(cells_per_group)


@pytest.mark.parametrize(
    "scenario_func,desc",
    [
        (scenario_1_ideal_bridging, "Ideal bridging"),
        (scenario_2_partial_bridging, "Partial bridging"),
        (scenario_3_insufficient_bridging, "Insufficient bridging"),
        (scenario_4_no_bridging, "No bridging"),
        (scenario_5_some_small_batches, "Some small batches"),
        (scenario_6_small_batches_break_bridging, "Small batches break bridging"),
        (scenario_7_collapse_one_condition, "Collapse one condition"),
        (scenario_8_collapse_both_conditions, "Collapse both conditions"),
        (lambda: scenario_9_single_batch_per_condition()[0], "Single batch/condition, bridging"),
        (lambda: scenario_9_single_batch_per_condition()[1], "Single batch/condition, no bridging"),
        (scenario_10_insufficient_coverage, "Insufficient coverage"),
        (scenario_11_asymmetric_batch_counts, "Asymmetric batch counts"),
        (scenario_12_fraction_based_validity, "Fraction threshold validity"),
        (scenario_13_many_bridging_batches, "Many bridging batches"),
        (scenario_14_edge_min_bridging, "Edge min bridging"),
    ],
)
def test_batch_key_only_scenarios(scenario_func, desc):
    adata = scenario_func()
    pb_result, de_result = _run_pseudobulk_and_de(adata, desc, verbose=True)
    assert pb_result is not None, f"{desc} pseudobulk failed"
    assert de_result is not None, f"{desc} de failed"


def run_all_batch_only_scenarios():
    print("Running all batch_key_only DE scenarios")
    for func, desc in [
        (scenario_1_ideal_bridging, "Ideal bridging"),
        (scenario_2_partial_bridging, "Partial bridging"),
        (scenario_3_insufficient_bridging, "Insufficient bridging"),
        (scenario_4_no_bridging, "No bridging"),
        (scenario_5_some_small_batches, "Some small batches"),
        (scenario_6_small_batches_break_bridging, "Small batches break bridging"),
        (scenario_7_collapse_one_condition, "Collapse one condition"),
        (scenario_8_collapse_both_conditions, "Collapse both conditions"),
        (lambda: scenario_9_single_batch_per_condition()[0], "Single batch/condition (bridging)"),
        (lambda: scenario_9_single_batch_per_condition()[1], "Single batch/condition (no bridging)"),
        (scenario_10_insufficient_coverage, "Insufficient coverage"),
        (scenario_11_asymmetric_batch_counts, "Asymmetric batch counts"),
        (scenario_12_fraction_based_validity, "Fraction threshold validity"),
        (scenario_13_many_bridging_batches, "Many bridging batches"),
        (scenario_14_edge_min_bridging, "Edge min bridging"),
    ]:
        adata = func()
        pb_result, de_result = _run_pseudobulk_and_de(adata, desc, verbose=True)
        print(f"{desc}: {'PASS' if (pb_result is not None and de_result is not None) else 'FAIL'}")
    print("All scenarios done.")


if __name__ == "__main__":
    run_all_batch_only_scenarios()
