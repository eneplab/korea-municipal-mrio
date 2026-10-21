# src/benchmark_accounting.py

import math

import numpy as np
import pandas as pd


from src.utils import stable_sum


def reconcile_benchmark(frames, max_adjustment=0.01):

    result = {name: frame.copy() for name, frame in frames.items()}
    Z, M = result["Z"], result["M"]

    ## Cost-side output
    x = result["x_i"].iloc[:, 0].to_numpy(float)
    x_j = result["x_j"].iloc[0].to_numpy(float)
    if not np.array_equal(x, x_j):
        raise ValueError("Reported output axes differ")
    z_row = stable_sum(Z.to_numpy(), 1)
    z_col = stable_sum(Z.to_numpy(), 0)
    m_col = stable_sum(M.to_numpy(), 0)
    va_col = stable_sum(result["VA"].to_numpy(), 0)
    output = np.array([
        math.fsum(parts)
        for parts in zip(z_col, m_col, va_col)
    ])

    if (output < 0).any() or np.max(np.abs(output - x)) > max_adjustment:
        raise ValueError("Cost-side output requires a substantive or negative adjustment")
    if not np.array_equal(output == 0, x == 0):
        raise ValueError("Benchmark output reconciliation would change zero-output positions")
    result["x_i"].iloc[:, 0] = output
    result["x_j"].iloc[0] = output

    ## Domestic final demand
    frame = result["Y_d"]
    values = frame.to_numpy(float, copy=True)
    totals = output - z_row

    for k, vector in enumerate(values):
        difference = float(totals[k] - math.fsum(vector))
        if abs(difference) > max_adjustment:
            raise ValueError(f"Benchmark row discrepancy exceeds the bound: {Z.index[k]}")
        if difference == 0:
            continue

        nonzero = np.flatnonzero(vector != 0)
        if len(nonzero) == 0:
            raise ValueError(f"Cannot reconcile final demand without a new nonzero: {Z.index[k]}")

        anchor = nonzero[np.argmax(np.abs(vector[nonzero]))]
        old = float(vector[anchor])
        vector[anchor] += difference
        if np.sign(vector[anchor]) != np.sign(old):
            raise ValueError("Benchmark reconciliation would change an account sign")

        # Compensate summation error at the same existing cell.
        vector[anchor] += totals[k] - math.fsum(vector)
        if np.sign(vector[anchor]) != np.sign(old):
            raise ValueError("Benchmark compensation would change an account sign")

    result["Y_d"] = pd.DataFrame(values, index=frame.index, columns=frame.columns)

    ## Sign and zero preservation
    for name in ["Y_d", "VA"]:
        if not np.array_equal(np.sign(result[name]), np.sign(frames[name])):
            raise ValueError("Benchmark reconciliation changed a sign or zero position")

    return result


def municipal_transaction_targets(Z_ref, allocation, index):

    weights = allocation.set_index(
        ["province_en", "municipality_id", "io"]
    )["output_weight"]
    row_total = pd.Series(stable_sum(Z_ref.to_numpy(), 1), index=Z_ref.index)
    col_total = pd.Series(stable_sum(Z_ref.to_numpy(), 0), index=Z_ref.columns)

    row = np.array([
        row_total.loc[(p, i)] * weights.loc[(p, m, i)]
        for p, m, i in index
    ])
    col = np.array([
        col_total.loc[(p, i)] * weights.loc[(p, m, i)]
        for p, m, i in index
    ])

    return pd.Series(row, index=index), pd.Series(col, index=index)
