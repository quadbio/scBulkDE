import warnings

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import scbulkde as scb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def make_test_adata_both_keys(
    cells_per_group: dict[str, dict[str, dict[str, int]]],
    n_genes: int = 100,
    seed: int = 42,
):
    import anndata as ad

    rng = np.random.default_rng(seed)
    obs_records = []
    cell_idx = 0
    for condition, replicates in cells_per_group.items():
        for rep_id, batches in replicates.items():
            for batch_id, n_cells in batches.items():
                for _ in range(n_cells):
                    obs_records.append(
                        {
                            "cell_id": f"cell_{cell_idx}",
                            "condition": condition,
                            "replicate": rep_id,
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


def _run_pseudobulk_and_de(adata, scenario_name, verbose=True):
    group_key = "condition"
    query = "disease"
    reference = "healthy"
    layer = "counts"
    replicate_key = "replicate"
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


# -------------------------------------
# Scenario definitions for both keys
# -------------------------------------


def scenario_1_ideal_all_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 100, "batch_B": 100},
            "patient_2": {"batch_A": 90, "batch_B": 110},
            "patient_3": {"batch_A": 95, "batch_B": 105},
        },
        "healthy": {
            "patient_4": {"batch_A": 105, "batch_B": 95},
            "patient_5": {"batch_A": 100, "batch_B": 100},
            "patient_6": {"batch_A": 110, "batch_B": 90},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_2_partial_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 100, "batch_B": 100},
            "patient_2": {"batch_A": 90, "batch_C": 110},
            "patient_3": {"batch_B": 95, "batch_C": 105},
        },
        "healthy": {
            "patient_4": {"batch_A": 105, "batch_B": 95},
            "patient_5": {"batch_A": 100, "batch_D": 100},
            "patient_6": {"batch_B": 110, "batch_D": 90},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_3_insufficient_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 100, "batch_B": 100},
            "patient_2": {"batch_B": 90, "batch_C": 110},
            "patient_3": {"batch_C": 95, "batch_B": 105},
        },
        "healthy": {
            "patient_4": {"batch_A": 105, "batch_D": 95},
            "patient_5": {"batch_D": 100, "batch_E": 100},
            "patient_6": {"batch_E": 110, "batch_D": 90},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_4_no_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 100, "batch_B": 100},
            "patient_2": {"batch_A": 90, "batch_B": 110},
            "patient_3": {"batch_B": 95, "batch_A": 105},
        },
        "healthy": {
            "patient_4": {"batch_C": 105, "batch_D": 95},
            "patient_5": {"batch_C": 100, "batch_D": 100},
            "patient_6": {"batch_D": 110, "batch_C": 90},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_5_some_small_reps_bridging_ok():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 100, "batch_B": 100},
            "patient_2": {"batch_A": 90, "batch_B": 110},
            "patient_3": {"batch_A": 5, "batch_B": 5},
        },
        "healthy": {
            "patient_4": {"batch_A": 105, "batch_B": 95},
            "patient_5": {"batch_A": 100, "batch_B": 100},
            "patient_6": {"batch_C": 3, "batch_D": 7},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_6_small_reps_break_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 100, "batch_B": 100},
            "patient_2": {"batch_C": 90, "batch_D": 110},
            "patient_3": {"batch_A": 5, "batch_C": 5},
        },
        "healthy": {
            "patient_4": {"batch_C": 105, "batch_D": 95},
            "patient_5": {"batch_E": 100, "batch_F": 100},
            "patient_6": {"batch_A": 3, "batch_B": 7},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_7_collapse_one_condition():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 10, "batch_B": 15},
            "patient_2": {"batch_A": 12, "batch_B": 13},
            "patient_3": {"batch_A": 8, "batch_B": 17},
        },
        "healthy": {
            "patient_4": {"batch_A": 105, "batch_B": 95},
            "patient_5": {"batch_A": 100, "batch_B": 100},
            "patient_6": {"batch_A": 110, "batch_B": 90},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_8_collapse_both_conditions():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 10, "batch_B": 15},
            "patient_2": {"batch_A": 12, "batch_B": 13},
        },
        "healthy": {
            "patient_3": {"batch_A": 8, "batch_B": 17},
            "patient_4": {"batch_C": 11, "batch_D": 14},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_9_insufficient_coverage_one_condition():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 60, "batch_B": 60},
            "patient_2": {"batch_A": 15, "batch_B": 15},
            "patient_3": {"batch_A": 20, "batch_B": 20},
            "patient_4": {"batch_A": 25, "batch_B": 25},
        },
        "healthy": {
            "patient_5": {"batch_A": 100, "batch_B": 100},
            "patient_6": {"batch_A": 90, "batch_B": 110},
            "patient_7": {"batch_A": 95, "batch_B": 105},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_10_single_rep_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 250, "batch_B": 250},
        },
        "healthy": {
            "patient_2": {"batch_A": 240, "batch_B": 260},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_11_single_rep_no_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 250, "batch_B": 250},
        },
        "healthy": {
            "patient_2": {"batch_C": 240, "batch_D": 260},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_12_asymmetric_reps_batches():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 80, "batch_B": 80, "batch_C": 80},
            "patient_2": {"batch_A": 70, "batch_B": 90},
            "patient_3": {"batch_B": 85, "batch_C": 75},
            "patient_4": {"batch_A": 90, "batch_C": 70},
            "patient_5": {"batch_A": 75, "batch_B": 85},
        },
        "healthy": {
            "patient_6": {"batch_A": 150, "batch_B": 150},
            "patient_7": {"batch_B": 140, "batch_D": 160},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_13_replicate_in_single_batch():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 200},
            "patient_2": {"batch_B": 180},
            "patient_3": {"batch_A": 100, "batch_B": 100},
        },
        "healthy": {
            "patient_4": {"batch_A": 190},
            "patient_5": {"batch_B": 210},
            "patient_6": {"batch_A": 95, "batch_B": 105},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_14_many_batches_few_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 80, "batch_B": 80, "batch_C": 80},
            "patient_2": {"batch_D": 70, "batch_E": 90, "batch_F": 80},
            "patient_3": {"batch_G": 85, "batch_H": 75, "batch_I": 80},
        },
        "healthy": {
            "patient_4": {"batch_A": 90, "batch_J": 85, "batch_K": 85},
            "patient_5": {"batch_B": 80, "batch_L": 90, "batch_M": 90},
            "patient_6": {"batch_N": 95, "batch_O": 85, "batch_P": 80},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_15_fraction_based_validity():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 25, "batch_B": 25},
            "patient_2": {"batch_A": 20, "batch_B": 20},
            "patient_3": {"batch_A": 5, "batch_B": 5},
        },
        "healthy": {
            "patient_4": {"batch_A": 30, "batch_B": 30},
            "patient_5": {"batch_A": 20, "batch_B": 20},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_16_edge_exactly_min_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 100, "batch_B": 100, "batch_C": 100},
            "patient_2": {"batch_A": 90, "batch_C": 110, "batch_D": 100},
            "patient_3": {"batch_B": 95, "batch_D": 105, "batch_E": 100},
        },
        "healthy": {
            "patient_4": {"batch_A": 105, "batch_B": 95, "batch_F": 100},
            "patient_5": {"batch_F": 100, "batch_G": 100, "batch_H": 100},
            "patient_6": {"batch_G": 110, "batch_H": 90, "batch_I": 100},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_17_unbalanced_cells_per_batch():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 180, "batch_B": 20},
            "patient_2": {"batch_A": 10, "batch_B": 190},
            "patient_3": {"batch_A": 100, "batch_B": 100},
        },
        "healthy": {
            "patient_4": {"batch_A": 195, "batch_B": 5},
            "patient_5": {"batch_A": 50, "batch_B": 150},
            "patient_6": {"batch_A": 90, "batch_B": 110},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_18_three_plus_batches_all_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 70, "batch_B": 70, "batch_C": 70, "batch_D": 70},
            "patient_2": {"batch_A": 65, "batch_B": 75, "batch_C": 80, "batch_D": 60},
            "patient_3": {"batch_A": 80, "batch_B": 60, "batch_C": 65, "batch_D": 75},
        },
        "healthy": {
            "patient_4": {"batch_A": 75, "batch_B": 65, "batch_C": 70, "batch_D": 70},
            "patient_5": {"batch_A": 60, "batch_B": 80, "batch_C": 75, "batch_D": 65},
            "patient_6": {"batch_A": 70, "batch_B": 70, "batch_C": 60, "batch_D": 80},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_19_mixed_validity_and_bridging():
    cells_per_group = {
        "disease": {
            "patient_1": {"batch_A": 100, "batch_B": 100},
            "patient_2": {"batch_A": 80, "batch_C": 120},
            "patient_3": {"batch_D": 10, "batch_E": 15},
            "patient_4": {"batch_B": 90, "batch_F": 110},
        },
        "healthy": {
            "patient_5": {"batch_A": 95, "batch_B": 105},
            "patient_6": {"batch_G": 100, "batch_H": 100},
            "patient_7": {"batch_A": 5, "batch_I": 10},
            "patient_8": {"batch_B": 110, "batch_J": 90},
        },
    }
    return make_test_adata_both_keys(cells_per_group)


def scenario_20_stress_many_reps_many_batches():
    n_reps_per_cond = 8
    n_batches = 6
    cells_per_group = {
        "disease": {
            f"patient_{i}": {f"batch_{chr(65 + j)}": 50 + (i * j) % 30 for j in range(n_batches)}
            for i in range(1, n_reps_per_cond + 1)
        },
        "healthy": {
            f"patient_{i + n_reps_per_cond}": {f"batch_{chr(65 + j)}": 55 + (i * j) % 25 for j in range(n_batches)}
            for i in range(1, n_reps_per_cond + 1)
        },
    }
    return make_test_adata_both_keys(cells_per_group)


SCENARIOS = [
    (scenario_1_ideal_all_bridging, "Ideal bridging"),
    (scenario_2_partial_bridging, "Partial bridging"),
    (scenario_3_insufficient_bridging, "Insufficient bridging"),
    (scenario_4_no_bridging, "No bridging"),
    (scenario_5_some_small_reps_bridging_ok, "Some small reps, bridging OK"),
    (scenario_6_small_reps_break_bridging, "Small reps break bridging"),
    (scenario_7_collapse_one_condition, "Collapse one condition"),
    (scenario_8_collapse_both_conditions, "Collapse both conditions"),
    (scenario_9_insufficient_coverage_one_condition, "Insufficient coverage of one cond"),
    (scenario_10_single_rep_bridging, "Single rep per condition, bridging"),
    (scenario_11_single_rep_no_bridging, "Single rep per condition, no bridging"),
    (scenario_12_asymmetric_reps_batches, "Asymmetric reps and batches"),
    (scenario_13_replicate_in_single_batch, "Replicates in single batch"),
    (scenario_14_many_batches_few_bridging, "Many batches, few bridging"),
    (scenario_15_fraction_based_validity, "Fraction-based validity"),
    (scenario_16_edge_exactly_min_bridging, "Edge: exactly min bridging"),
    (scenario_17_unbalanced_cells_per_batch, "Unbalanced cells per batch/rep"),
    (scenario_18_three_plus_batches_all_bridging, "3+ bridging batches"),
    (scenario_19_mixed_validity_and_bridging, "Mixed validity/bridging"),
    (scenario_20_stress_many_reps_many_batches, "Stress: many reps/batches"),
]


@pytest.mark.parametrize("scenario_func,desc", SCENARIOS)
def test_replicate_and_batch_scenarios(scenario_func, desc):
    adata = scenario_func()
    pb_result, de_result = _run_pseudobulk_and_de(adata, desc, verbose=True)
    assert pb_result is not None, f"{desc} pseudobulk failed"
    assert de_result is not None, f"{desc} de failed"


def run_all_replicate_and_batch_scenarios():
    for func, desc in SCENARIOS:
        adata = func()
        pb_result, de_result = _run_pseudobulk_and_de(adata, desc, verbose=True)
        print(f"{desc}: {'PASS' if (pb_result is not None and de_result is not None) else 'FAIL'}")
    print("All scenarios done.")


if __name__ == "__main__":
    run_all_replicate_and_batch_scenarios()
