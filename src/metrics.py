# src/metrics.py

import numpy as np
import pandas as pd


def positive_spatial_composition(Z):

    Z_pos = Z.clip(lower=0)
    row = Z_pos.index.to_frame(index=False)
    col = Z_pos.columns.to_frame(index=False)

    row_province = row["Province"].to_numpy()
    row_municipality = row["Municipality"].to_numpy()
    col_province = col["Province"].to_numpy()
    col_municipality = col["Municipality"].to_numpy()

    values = Z_pos.to_numpy()

    intra = 0.0
    same_province = 0.0
    cross_province = 0.0

    for r in range(values.shape[0]):

        if values[r, :].sum() == 0:
            continue

        is_intra = (
            (row_province[r] == col_province)
            & (row_municipality[r] == col_municipality)
        )
        is_same_province = (
            (row_province[r] == col_province)
            & (row_municipality[r] != col_municipality)
        )
        is_cross_province = row_province[r] != col_province

        intra += values[r, is_intra].sum()
        same_province += values[r, is_same_province].sum()
        cross_province += values[r, is_cross_province].sum()

    total = intra + same_province + cross_province

    return {
        "positive_Z_total": float(total),
        "intra_municipal": float(intra),
        "same_province_intermunicipal": float(same_province),
        "cross_province_intermunicipal": float(cross_province),
        "intra_share": float(intra / total) if total != 0 else 0.0,
        "same_province_inter_share": (
            float(same_province / total) if total != 0 else 0.0
        ),
        "cross_province_share": float(cross_province / total) if total != 0 else 0.0,
    }


def matrix_difference(Z, baseline_Z):

    Z, baseline_Z = Z.align(baseline_Z, join="inner", axis=None)

    z = Z.to_numpy()
    b = baseline_Z.to_numpy()

    diff_abs = np.abs(z - b)
    z_flat = z.ravel()
    b_flat = b.ravel()

    z_mean = z_flat.mean()
    b_mean = b_flat.mean()
    z_centered = z_flat - z_mean
    b_centered = b_flat - b_mean
    denom = np.sqrt(np.sum(z_centered ** 2) * np.sum(b_centered ** 2))

    return {
        "abs_diff_sum": float(diff_abs.sum()),
        "relative_abs_diff": (
            float(diff_abs.sum() / np.abs(b).sum())
            if np.abs(b).sum() != 0
            else 0.0
        ),
        "max_abs_diff": float(diff_abs.max()),
        "corr_with_baseline": (
            float(np.sum(z_centered * b_centered) / denom) if denom != 0 else np.nan
        ),
        "nonzero_count": int((np.abs(z) > 0).sum()),
    }


def aggregate_to_province_sector(Z):

    idx = Z.index.to_frame(index=False)
    col = Z.columns.to_frame(index=False)

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


def aggregated_matrix_difference(Z, baseline_Z):

    Z_agg = aggregate_to_province_sector(Z)
    baseline_agg = aggregate_to_province_sector(baseline_Z)

    out = matrix_difference(Z_agg, baseline_agg)

    return {
        f"province_sector_{key}": value
        for key, value in out.items()
    }


def gravity_summary(gravity):

    estimated = gravity.loc[gravity["method"].eq("yamada_like_gravity")].copy()
    simple = gravity.loc[gravity["method"].eq("simple_trade")].copy()

    out = {
        "n_sectors": int(len(gravity)),
        "n_gravity_sectors": int(len(estimated)),
        "n_simple_trade_sectors": int(len(simple)),
        "gamma_mean": float(estimated["gamma"].mean()) if len(estimated) else np.nan,
        "gamma_median": float(estimated["gamma"].median()) if len(estimated) else np.nan,
        "gamma_min": float(estimated["gamma"].min()) if len(estimated) else np.nan,
        "gamma_max": float(estimated["gamma"].max()) if len(estimated) else np.nan,
        "negative_gamma_sectors": int((estimated["gamma"] < 0).sum()),
        "significant_distance_sectors": int((estimated["log_distance_p"] < 0.05).sum()),
    }

    return out


def zero_employment_support_check(matrices, zero, exception=None):

    exception = exception or set()
    zero_set = set(
        map(
            tuple,
            zero[["Province", "Municipality", "Sector"]].astype(str).values,
        )
    )
    zero_set = zero_set - exception

    rows = []

    for name, obj, axes in [
        ("Z", matrices["Z"], ("index", "columns")),
        ("M", matrices["M"], ("columns",)),
        ("Y_d", matrices["Y_d"], ("index",)),
        ("VA", matrices["VA"], ("columns",)),
        ("X", matrices["x_i"], ("index",)),
        ("x_i", matrices["x_i"], ("index",)),
        ("x_j", matrices["x_j"], ("columns",)),
    ]:

        if "index" in axes:
            mask = index_mask(obj.index, zero_set)
            values = obj.iloc[mask, :].to_numpy()
            rows.append(
                {
                    "matrix": name,
                    "axis": "index",
                    "zero_support_count": int(mask.sum()),
                    "abs_sum": float(np.abs(values).sum()),
                    "nonzero_count": int((np.abs(values) > 0).sum()),
                }
            )

        if "columns" in axes:
            mask = index_mask(obj.columns, zero_set)
            values = obj.iloc[:, mask].to_numpy()
            rows.append(
                {
                    "matrix": name,
                    "axis": "columns",
                    "zero_support_count": int(mask.sum()),
                    "abs_sum": float(np.abs(values).sum()),
                    "nonzero_count": int((np.abs(values) > 0).sum()),
                }
            )

    return pd.DataFrame(rows)


def index_mask(index, zero_set):

    idx = index.to_frame(index=False)
    keys = list(map(tuple, idx[["Province", "Municipality", "Sector"]].astype(str).values))

    return pd.Series(keys).isin(zero_set).to_numpy()
