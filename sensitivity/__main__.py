# sensitivity/__main__.py

from pathlib import Path
import argparse
import gc

import pandas as pd

from src.main import run_pipeline
from src.metrics import (
    aggregated_matrix_difference,
    gravity_summary,
    matrix_difference,
    positive_spatial_composition,
)
from src.mrio_io import load_mrio


BASELINE_DELTA = 0.30
BASELINE_DISTANCE = "travel_time"
BASELINE_FINAL_DEMAND_PROXY = "component"


def case_list(case_group):

    flq_cases = [
        {
            "case_group": "flq_delta",
            "case_id": "delta_020",
            "delta": 0.20,
            "distance_name": BASELINE_DISTANCE,
        },
        {
            "case_group": "flq_delta",
            "case_id": "delta_040",
            "delta": 0.40,
            "distance_name": BASELINE_DISTANCE,
        },
    ]

    distance_cases = [
        {
            "case_group": "distance",
            "case_id": "road_distance",
            "delta": BASELINE_DELTA,
            "distance_name": "road_distance",
        },
        {
            "case_group": "distance",
            "case_id": "euclidean_distance",
            "delta": BASELINE_DELTA,
            "distance_name": "euclidean_distance",
        },
    ]

    final_demand_cases = [
        {
            "case_group": "final_demand_proxy",
            "case_id": "grdp_only_final_demand",
            "delta": BASELINE_DELTA,
            "distance_name": BASELINE_DISTANCE,
            "final_demand_proxy": "grdp",
        },
    ]

    if case_group == "flq":
        return flq_cases
    if case_group == "distance":
        return distance_cases
    if case_group == "final_demand":
        return final_demand_cases

    return flq_cases + distance_cases + final_demand_cases


def baseline_summary(ROOT, baseline_Z):

    gravity = pd.read_csv(ROOT / "release" / "diagnostics" / "gravity_coefficients.csv")
    convergence = pd.read_csv(ROOT / "release" / "diagnostics" / "mrgras_convergence.csv")
    allocation = pd.read_csv(ROOT / "release" / "diagnostics" / "positive_allocation_summary.csv")

    out = {
        "case_group": "baseline",
        "case_id": "baseline",
        "delta": BASELINE_DELTA,
        "distance_name": BASELINE_DISTANCE,
        "final_demand_proxy": BASELINE_FINAL_DEMAND_PROXY,
        "province_sector_abs_diff_sum": 0.0,
        "province_sector_relative_abs_diff": 0.0,
        "province_sector_max_abs_diff": 0.0,
        "province_sector_corr_with_baseline": 1.0,
        "province_sector_nonzero_count": 0,
        "Y_d_relative_abs_diff": 0.0,
        "Y_m_relative_abs_diff": 0.0,
        "Y_total_relative_abs_diff": 0.0,
    }
    out.update(positive_spatial_composition(baseline_Z))
    out.update(gravity_summary(gravity))
    out.update(allocation.iloc[0].to_dict())
    out.update(convergence.iloc[0].to_dict())

    return out


def run_case(ROOT, OUT, case, baseline, save_matrices=False):

    case_dir = OUT / "cases" / case["case_id"]
    diagnostics_dir = case_dir / "diagnostics"
    final_demand_proxy = case.get(
        "final_demand_proxy",
        BASELINE_FINAL_DEMAND_PROXY,
    )

    print(f"[RUN] {case['case_id']}")

    outputs = run_pipeline(
        delta=case["delta"],
        distance_name=case["distance_name"],
        final_demand_proxy=final_demand_proxy,
        out_dir=case_dir / "matrices",
        diagnostics_dir=diagnostics_dir,
        save_outputs=save_matrices,
        save_diagnostics=True,
        save_allocation_report=False,
        return_allocation_report=False,
        verbose=True,
    )

    out = {
        "case_group": case["case_group"],
        "case_id": case["case_id"],
        "delta": case["delta"],
        "distance_name": case["distance_name"],
        "final_demand_proxy": final_demand_proxy,
    }
    out.update(aggregated_matrix_difference(outputs["Z"], baseline.Z))

    y_base = pd.concat([baseline.Y_d, baseline.Y_m], axis=0)
    y_case = pd.concat([outputs["Y_d"], outputs["Y_m"]], axis=0)
    yd_diff = matrix_difference(outputs["Y_d"], baseline.Y_d)
    ym_diff = matrix_difference(outputs["Y_m"], baseline.Y_m)
    y_diff = matrix_difference(y_case, y_base)

    out["Y_d_relative_abs_diff"] = yd_diff["relative_abs_diff"]
    out["Y_m_relative_abs_diff"] = ym_diff["relative_abs_diff"]
    out["Y_total_relative_abs_diff"] = y_diff["relative_abs_diff"]

    out.update(positive_spatial_composition(outputs["Z"]))
    out.update(gravity_summary(outputs["gravity"]))
    out.update(outputs["allocation_summary"])
    out.update(outputs["mrgras_info"])

    del outputs
    gc.collect()

    print(f"[DONE] {case['case_id']}")

    return out


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        choices=["all", "flq", "distance", "final_demand"],
        default="all",
    )
    parser.add_argument(
        "--save-matrices",
        action="store_true",
    )
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    OUT = ROOT / "release" / "sensitivity"
    OUT.mkdir(parents=True, exist_ok=True)

    print("[RUN] Sensitivity analysis")

    baseline = load_mrio(ROOT / "release" / "baseline", as_object=True)

    rows = [baseline_summary(ROOT, baseline.Z)]

    for case in case_list(args.cases):
        rows.append(
            run_case(
                ROOT=ROOT,
                OUT=OUT,
                case=case,
                baseline=baseline,
                save_matrices=args.save_matrices,
            )
        )

    pd.DataFrame(rows).to_csv(
        OUT / "sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[DONE] Sensitivity analysis completed")


if __name__ == "__main__":
    main()
