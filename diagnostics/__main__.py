# diagnostics/__main__.py

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from metadata.sector import SECTOR_76, SECTOR_83, SECTOR_MAPPING_83_TO_76

from src.metrics import (
    gravity_summary,
    positive_spatial_composition,
    zero_employment_support_check,
)
from src.mrio_io import load_mrio
from src.mrio_preprocessing import mrio_preprocessing


warnings.filterwarnings("ignore")


def aggregate_municipal_Z(Z):

    idx = Z.index.to_frame(index=False)
    col = Z.columns.to_frame(index=False)

    name_to_id = {name: sector_id for sector_id, name in SECTOR_76.items()}

    if idx["Sector"].dtype == object:
        idx["Sector"] = idx["Sector"].map(name_to_id).fillna(idx["Sector"])
    if col["Sector"].dtype == object:
        col["Sector"] = col["Sector"].map(name_to_id).fillna(col["Sector"])

    Z = Z.copy()
    Z.index = pd.MultiIndex.from_frame(
        idx[["Province", "Sector"]],
        names=["Province", "Sector"],
    )
    Z.columns = pd.MultiIndex.from_frame(
        col[["Province", "Sector"]],
        names=["Province", "Sector"],
    )

    return (
        Z
        .groupby(level=["Province", "Sector"], axis=0)
        .sum()
        .groupby(level=["Province", "Sector"], axis=1)
        .sum()
    )


def province_sector_difference(ROOT, Z):

    Z_bench, *_ = mrio_preprocessing(
        benchmark_path=ROOT / "data" / "benchmark_io.csv",
        SECTOR_83=SECTOR_83,
        SECTOR_MAPPING_83_TO_76=SECTOR_MAPPING_83_TO_76,
    )

    Z_agg = aggregate_municipal_Z(Z)
    Z_bench, Z_agg = Z_bench.align(Z_agg, join="inner", axis=None)

    diff = Z_agg.to_numpy() - Z_bench.to_numpy()
    bench = Z_bench.to_numpy()

    return {
        "province_sector_abs_diff_sum": float(np.abs(diff).sum()),
        "province_sector_relative_abs_diff": (
            float(np.abs(diff).sum() / np.abs(bench).sum())
            if np.abs(bench).sum() != 0
            else 0.0
        ),
        "province_sector_max_abs_diff": float(np.abs(diff).max()),
    }


def gravity_sector_table(ROOT):

    gravity = pd.read_csv(ROOT / "release" / "diagnostics" / "gravity_coefficients.csv")
    gravity = gravity.copy()
    gravity["sector_name"] = gravity["input_sector"].map(SECTOR_76)

    keep = [
        "input_sector",
        "sector_name",
        "method",
        "n_obs",
        "gamma",
        "log_distance_coef",
        "log_distance_p",
        "r2",
        "adj_r2",
        "positive_flow_count",
        "denominator_zero_count",
    ]

    return gravity[keep]


def main():

    ROOT = Path(__file__).resolve().parents[1]
    OUT = ROOT / "release" / "diagnostics"
    OUT.mkdir(parents=True, exist_ok=True)

    print("[RUN] Diagnostics")

    mrio = load_mrio(ROOT / "release" / "baseline", as_object=True)
    gravity = pd.read_csv(OUT / "gravity_coefficients.csv")
    allocation = pd.read_csv(OUT / "positive_allocation_summary.csv")
    convergence = pd.read_csv(OUT / "mrgras_convergence.csv")
    zero = pd.read_csv(ROOT / "data" / "zero_employee.csv")

    exception = {("Sejong", "Sejong-si", "Ships")}

    support = zero_employment_support_check(
        matrices={
            "Z": mrio.Z,
            "M": mrio.M,
            "Y_d": mrio.Y_d,
            "VA": mrio.VA,
            "x_i": mrio.x_i,
            "x_j": mrio.x_j,
        },
        zero=zero,
        exception=exception,
    )
    support.to_csv(
        OUT / "zero_employment_support_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    sector_table = gravity_sector_table(ROOT)
    sector_table.to_csv(
        OUT / "gravity_sector_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    spatial = positive_spatial_composition(mrio.Z)
    province_sector = province_sector_difference(ROOT, mrio.Z)
    gravity_stats = gravity_summary(gravity)

    summary = {}
    summary.update(spatial)
    summary.update(province_sector)
    summary.update(gravity_stats)
    summary.update(allocation.iloc[0].to_dict())
    summary.update(convergence.iloc[0].to_dict())
    summary["zero_employment_violation_abs_sum"] = float(support["abs_sum"].sum())
    summary["zero_employment_violation_count"] = int(support["nonzero_count"].sum())

    pd.DataFrame([summary]).to_csv(
        OUT / "diagnostic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("[DONE] Diagnostics completed")


if __name__ == "__main__":
    main()
