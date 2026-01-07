import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scbulkde as scb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def make_test_adata_replicate_only(
    cells_per_group: dict[str, dict[str, int]],
    n_genes: int = 100,
    seed: int = 42,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    obs_records = []
    cell_idx = 0
    for condition, replicates in cells_per_group.items():
        for rep_id, n_cells in replicates.items():
            for _ in range(n_cells):
                obs_records.append(
                    {
                        "cell_id": f"cell_{cell_idx}",
                        "condition": condition,
                        "replicate": rep_id,
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


def _run_pseudobulk_and_de(adata, scenario_name, verbose=True):
    group_key = "condition"
    query = "disease"
    reference = "healthy"
    layer = "counts"
    replicate_key = "replicate"
    batch_key = None
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
        "mode": "sum",
    }
    de_kwargs = {
        "min_samples": 3,
        "n_repetitions": 5,
        "resampling_fraction": 0.7,
        "min_list_overlap": 0.7,
        "n_cpus": 1,
    }
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


# ===========================
# REPLICATE-KEY-ONLY SCENARIOS
# ===========================


def scenario_1_ideal_case():
    cells_per_group = {
        "disease": {"patient_1": 200, "patient_2": 180, "patient_3": 220},
        "healthy": {"patient_4": 190, "patient_5": 210, "patient_6": 175},
    }
    return make_test_adata_replicate_only(cells_per_group)


def scenario_2_some_small_replicates():
    cells_per_group = {
        "disease": {"patient_1": 200, "patient_2": 180, "patient_3": 10},
        "healthy": {"patient_4": 190, "patient_5": 15, "patient_6": 175},
    }
    return make_test_adata_replicate_only(cells_per_group)


def scenario_3_small_replicates_insufficient_coverage():
    cells_per_group = {
        "disease": {
            "patient_1": 100,
            "patient_2": 30,
            "patient_3": 25,
            "patient_4": 20,
            "patient_5": 25,
        },
        "healthy": {"patient_6": 200, "patient_7": 180, "patient_8": 170},
    }
    return make_test_adata_replicate_only(cells_per_group)


def scenario_4_no_valid_replicates_one_condition():
    cells_per_group = {
        "disease": {"patient_1": 20, "patient_2": 25, "patient_3": 30, "patient_4": 15},
        "healthy": {"patient_5": 200, "patient_6": 180, "patient_7": 190},
    }
    return make_test_adata_replicate_only(cells_per_group)


def scenario_5_no_valid_replicates_both_conditions():
    cells_per_group = {
        "disease": {"patient_1": 20, "patient_2": 25, "patient_3": 30},
        "healthy": {"patient_4": 15, "patient_5": 20, "patient_6": 25},
    }
    return make_test_adata_replicate_only(cells_per_group)


def scenario_6_single_replicate_per_condition():
    cells_per_group = {
        "disease": {"patient_1": 500},
        "healthy": {"patient_2": 480},
    }
    return make_test_adata_replicate_only(cells_per_group)


def scenario_7_asymmetric_replicates():
    cells_per_group = {
        "disease": {
            "patient_1": 100,
            "patient_2": 120,
            "patient_3": 110,
            "patient_4": 130,
            "patient_5": 105,
            "patient_6": 115,
        },
        "healthy": {"patient_7": 300, "patient_8": 280},
    }
    return make_test_adata_replicate_only(cells_per_group)


def scenario_8_fraction_based_validity():
    cells_per_group = {
        "disease": {"patient_1": 40, "patient_2": 35, "patient_3": 25},
        "healthy": {"patient_4": 45, "patient_5": 45},
    }
    return make_test_adata_replicate_only(cells_per_group)


def scenario_9_edge_case_exact_thresholds():
    cells_per_group = {
        "disease": {"patient_1": 50, "patient_2": 50, "patient_3": 49},
        "healthy": {"patient_4": 50, "patient_5": 50, "patient_6": 50},
    }
    return make_test_adata_replicate_only(cells_per_group)


def scenario_10_very_unequal_replicate_sizes():
    cells_per_group = {
        "disease": {"patient_1": 1000, "patient_2": 5, "patient_3": 3, "patient_4": 2},
        "healthy": {"patient_5": 800, "patient_6": 10, "patient_7": 200},
    }
    return make_test_adata_replicate_only(cells_per_group)


SCENARIOS = [
    (scenario_1_ideal_case, "Ideal case"),
    (scenario_2_some_small_replicates, "Some small replicates"),
    (scenario_3_small_replicates_insufficient_coverage, "Small replicates, insufficient coverage"),
    (scenario_4_no_valid_replicates_one_condition, "No valid replicates one condition"),
    (scenario_5_no_valid_replicates_both_conditions, "No valid replicates both conditions"),
    (scenario_6_single_replicate_per_condition, "Single replicate per condition"),
    (scenario_7_asymmetric_replicates, "Asymmetric replicates"),
    (scenario_8_fraction_based_validity, "Fraction-based validity"),
    (scenario_9_edge_case_exact_thresholds, "Edge case: exact thresholds"),
    (scenario_10_very_unequal_replicate_sizes, "Very unequal replicate sizes"),
]


@pytest.mark.parametrize("scenario_func,desc", SCENARIOS)
def test_replicate_only_scenarios(scenario_func, desc):
    adata = scenario_func()
    pb_result, de_result = _run_pseudobulk_and_de(adata, desc, verbose=True)
    assert pb_result is not None, f"{desc} pseudobulk failed"
    assert de_result is not None, f"{desc} de failed"


def run_all_replicate_only_scenarios():
    print("Running all replicate_key_only DE scenarios")
    for func, desc in SCENARIOS:
        adata = func()
        pb_result, de_result = _run_pseudobulk_and_de(adata, desc, verbose=True)
        print(f"{desc}: {'PASS' if (pb_result is not None and de_result is not None) else 'FAIL'}")
    print("All scenarios done.")


if __name__ == "__main__":
    run_all_replicate_only_scenarios()
