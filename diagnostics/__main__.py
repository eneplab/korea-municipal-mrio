from pathlib import Path

import numpy as np
import pandas as pd

from metadata.sector import SECTOR_83, SECTOR_MAPPING_83_TO_76
from src.mrio_io import load_mrio
from src.mrio_preprocessing import mrio_preprocessing


def aggregate_province_blocks(Z):
    Z = Z.groupby(level="Province").sum()
    return Z.T.groupby(level="Province").sum().T


def constraint_residuals(mrio, W, eps=1e-12):
    Z = mrio.Z
    x_i = mrio.X.iloc[:, 0]
    x_j = mrio.input_total.iloc[0, :]
    u = x_i - mrio.Y_d.sum(axis=1)
    v = x_j - mrio.M.sum(axis=0) - mrio.VA.sum(axis=0)

    x_i_safe = x_i.abs().copy()
    x_i_safe[x_i_safe < 1e-6] = 1.0
    u[(u.abs() < 1e-3) & ((u.abs() / x_i_safe) < 1e-7)] = 0.0

    x_j_safe = x_j.abs().copy()
    x_j_safe[x_j_safe < 1e-6] = 1.0
    v[(v.abs() < 1e-3) & ((v.abs() / x_j_safe) < 1e-7)] = 0.0

    row_error = (Z.sum(axis=1) - u).abs()
    col_error = (Z.sum(axis=0) - v).abs()

    Z_block = aggregate_province_blocks(Z)
    Z_block, W = Z_block.align(W, join="inner", axis=None)
    block_error = (Z_block - W).abs()

    def scaled(error, target):
        return float(
            np.max(
                error.to_numpy() / np.maximum(np.abs(target.to_numpy()), 1.0),
                initial=0.0,
            )
        )

    zero_error = max(
        float(row_error[u.abs() <= eps].max()) if (u.abs() <= eps).any() else 0.0,
        float(col_error[v.abs() <= eps].max()) if (v.abs() <= eps).any() else 0.0,
        float(block_error[W.abs() <= eps].max().max())
        if (W.abs() <= eps).any().any()
        else 0.0,
    )

    return {
        "row_abs": float(row_error.max()),
        "column_abs": float(col_error.max()),
        "block_abs": float(block_error.max().max()),
        "scaled": max(
            scaled(row_error, u),
            scaled(col_error, v),
            scaled(block_error, W),
        ),
        "zero_target_abs": zero_error,
        "block_relative": float(
            block_error.to_numpy().sum() / np.abs(W.to_numpy()).sum()
        ),
    }


def production_activity_violations(mrio, zero, eps=1e-12):
    zero = zero.copy()
    for column in ("Province", "Municipality", "Sector"):
        zero[column] = zero[column].astype(str).str.strip()

    inactive = set(
        map(
            tuple,
            zero[["Province", "Municipality", "Sector"]].astype(str).to_numpy(),
        )
    )
    inactive.discard(("Sejong", "Sejong-si", "Ships"))

    idx = mrio.Z.index.to_frame(index=False)
    keys = list(
        map(
            tuple,
            idx[["Province", "Municipality", "Sector"]].astype(str).to_numpy(),
        )
    )
    mask = pd.Series(keys).isin(inactive).to_numpy()

    amounts = (
        mrio.Z.iloc[mask, :].abs().sum(axis=1).to_numpy()
        + mrio.Z.iloc[:, mask].abs().sum(axis=0).to_numpy()
        + mrio.M.iloc[:, mask].abs().sum(axis=0).to_numpy()
        + mrio.Y_d.iloc[mask, :].abs().sum(axis=1).to_numpy()
        + mrio.VA.iloc[:, mask].abs().sum(axis=0).to_numpy()
        + mrio.X.iloc[mask, 0].abs().to_numpy()
    )

    return {
        "count": int((amounts > eps).sum()),
        "absolute_sum": float(amounts.sum()),
    }


def add(rows, section, metric, value, tolerance="", status="INFO"):
    rows.append(
        {
            "Section": section,
            "Metric": metric,
            "Value": value,
            "Tolerance": tolerance,
            "Status": status,
        }
    )


def main():
    root = Path(__file__).resolve().parents[1]
    release = root / "release"
    print("[RUN] Diagnostics")

    mrio = load_mrio(release)
    if mrio.manifest is None:
        raise FileNotFoundError(release / "manifest.json")

    Z_bench, *_ = mrio_preprocessing(
        benchmark_path=root / "data" / "benchmark_io.csv",
        SECTOR_83=SECTOR_83,
        SECTOR_MAPPING_83_TO_76=SECTOR_MAPPING_83_TO_76,
    )
    W = aggregate_province_blocks(Z_bench)
    residuals = constraint_residuals(mrio, W)
    activity = production_activity_violations(
        mrio,
        pd.read_csv(root / "data" / "zero_employee.csv"),
    )

    balancing = mrio.manifest["balancing"]
    mult_tol = float(balancing["multiplier_tolerance"])
    constraint_tol = float(balancing["constraint_tolerance"])

    rows = []
    add(
        rows,
        "Accounting consistency",
        "MR-GRAS converged",
        bool(balancing["converged"]),
        "",
        "PASS" if balancing["converged"] else "FAIL",
    )
    add(
        rows,
        "Accounting consistency",
        "MR-GRAS iterations",
        int(balancing["iterations"]),
    )
    add(
        rows,
        "Accounting consistency",
        "Maximum multiplier change",
        float(balancing["max_multiplier_change"]),
        mult_tol,
        "PASS" if balancing["max_multiplier_change"] < mult_tol else "FAIL",
    )
    add(
        rows,
        "Accounting consistency",
        "Maximum scaled constraint residual",
        residuals["scaled"],
        constraint_tol,
        "PASS" if residuals["scaled"] < constraint_tol else "FAIL",
    )
    add(
        rows,
        "Accounting consistency",
        "Maximum zero-target absolute residual",
        residuals["zero_target_abs"],
        constraint_tol,
        "PASS" if residuals["zero_target_abs"] < constraint_tol else "FAIL",
    )
    add(
        rows,
        "Accounting consistency",
        "Maximum row absolute residual",
        residuals["row_abs"],
    )
    add(
        rows,
        "Accounting consistency",
        "Maximum column absolute residual",
        residuals["column_abs"],
    )
    add(
        rows,
        "Accounting consistency",
        "Maximum province-pair block absolute residual",
        residuals["block_abs"],
    )
    add(
        rows,
        "Production activity condition",
        "Violation count",
        activity["count"],
        0,
        "PASS" if activity["count"] == 0 else "FAIL",
    )
    add(
        rows,
        "Production activity condition",
        "Absolute violation sum",
        activity["absolute_sum"],
        1e-12,
        "PASS" if activity["absolute_sum"] <= 1e-12 else "FAIL",
    )
    add(
        rows,
        "Benchmark consistency",
        "Maximum province-pair block difference",
        residuals["block_abs"],
    )
    add(
        rows,
        "Benchmark consistency",
        "Relative absolute province-pair block difference",
        residuals["block_relative"],
    )

    pd.DataFrame(rows).to_csv(
        release / "diagnostic_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("[DONE] Diagnostics completed")


if __name__ == "__main__":
    main()
