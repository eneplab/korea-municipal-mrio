# src/intermunicipal_allocation.py

import numpy as np
import pandas as pd

from src.intramunicipal_allocation import calculate_flq_intra
from src.utils import is_zero, normalize_kernel


EPS = 1e-12

try:
    from scipy import stats
except Exception:
    stats = None


def fit_log_ols(obs, variable_names):

    if len(obs) == 0:
        return None

    obs = pd.DataFrame(
        obs,
        columns=["log_T"] + variable_names,
    )

    y = obs["log_T"].values.astype(float)
    X_raw = obs[variable_names].values.astype(float)
    X = np.column_stack([np.ones(len(obs)), X_raw])

    n_obs = len(obs)
    n_params = X.shape[1]

    if n_obs <= n_params:
        return None

    try:
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return None

    y_hat = X @ beta
    resid = y - y_hat

    rss = float(np.sum(resid ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))

    r2 = 1.0 - rss / tss if tss > EPS else np.nan
    adj_r2 = (
        1.0 - (1.0 - r2) * (n_obs - 1) / (n_obs - n_params)
        if n_obs > n_params
        else np.nan
    )

    sigma2 = rss / (n_obs - n_params)

    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(xtx_inv) * sigma2)
    except Exception:
        se = np.full(n_params, np.nan)

    t_stat = np.divide(
        beta,
        se,
        out=np.full_like(beta, np.nan),
        where=se > EPS,
    )

    if stats is not None:
        p_value = 2.0 * stats.t.sf(np.abs(t_stat), df=n_obs - n_params)
    else:
        p_value = np.full(n_params, np.nan)

    result = {
        "n_obs": n_obs,
        "r2": r2,
        "adj_r2": adj_r2,
        "rss": rss,
    }

    names = ["const"] + variable_names
    for idx, name in enumerate(names):
        result[f"{name}_coef"] = beta[idx]
        result[f"{name}_se"] = se[idx]
        result[f"{name}_t"] = t_stat[idx]
        result[f"{name}_p"] = p_value[idx]

    return result


def estimate_gravity(Z_ref, distance_17):

    ## Province-level positive flows
    Z_pos = Z_ref.clip(lower=0)
    provinces = Z_pos.index.get_level_values("Province").unique().tolist()
    sectors = Z_pos.index.get_level_values("Sector").unique().tolist()

    results = []
    zero_cases = []

    for i in sectors:

        trade_flows = (
            Z_pos
            .xs(i, level="Sector")
            .groupby(level="Province", axis=1)
            .sum()
        )

        origin = trade_flows.sum(axis=1)
        destination = trade_flows.sum(axis=0)

        obs = []
        positive_flow_count = 0
        denominator_zero_count = 0

        for R in provinces:
            for S in provinces:
                if R == S:
                    continue

                try:
                    T_RS = float(trade_flows.loc[R, S])
                    O_R = float(origin.loc[R])
                    D_S = float(destination.loc[S])
                    dist = float(distance_17.loc[R, S])
                except KeyError:
                    continue

                if T_RS <= 0:
                    continue

                positive_flow_count += 1

                if O_R <= 0 or D_S <= 0 or dist <= 0:
                    denominator_zero_count += 1
                    zero_cases.append(
                        {
                            "input_sector": i,
                            "R": R,
                            "S": S,
                            "T_RS": T_RS,
                            "O_R": O_R,
                            "D_S": D_S,
                            "distance": dist,
                        }
                    )
                    continue

                obs.append(
                    [
                        np.log(T_RS),
                        np.log(O_R),
                        np.log(D_S),
                        np.log(dist),
                    ]
                )

        result = fit_log_ols(
            obs,
            ["log_origin", "log_destination", "log_distance"],
        )

        row = {
            "input_sector": i,
            "positive_flow_count": positive_flow_count,
            "denominator_zero_count": denominator_zero_count,
        }

        if result is None:
            row.update(
                {
                    "method": "simple_trade",
                    "n_obs": np.nan,
                    "log_origin_coef": 1.0,
                    "log_destination_coef": 1.0,
                    "log_distance_coef": 0.0,
                    "gamma": 0.0,
                }
            )
        else:
            row.update(result)
            row["method"] = "yamada_like_gravity"
            row["gamma"] = -row["log_distance_coef"]

        results.append(row)

    gravity = pd.DataFrame(results)
    zero_cases = pd.DataFrame(zero_cases)

    return gravity, zero_cases


def build_positive_seed(
    Z_seed,
    x_i_seed,
    distance_229,
    gravity,
    zero_set,
    exception_set,
    delta=0.30,
):

    ## Index preparation
    idx = Z_seed.index.to_frame(index=False)

    provinces = idx["Province"].drop_duplicates().tolist()
    sectors = idx["Sector"].drop_duplicates().tolist()

    municipalities_by_province = {
        R: idx.loc[idx["Province"] == R, "Municipality"].drop_duplicates().tolist()
        for R in provinces
    }

    all_municipalities = idx["Municipality"].drop_duplicates().tolist()

    position = {
        key: pos
        for pos, key in enumerate(Z_seed.index)
    }

    positions = {
        (R, i): np.array(
            [position[(R, r, i)] for r in municipalities_by_province[R]],
            dtype=int,
        )
        for R in provinces
        for i in sectors
    }

    ## Positive output vector
    x = x_i_seed.iloc[:, 0].rename("x").astype(float)
    x.index.names = ["Province", "Municipality", "Sector"]
    x_pos = x.clip(lower=0)

    ## Positive origin mass
    origin_mass = Z_seed.clip(lower=0).sum(axis=1).values

    ## Commodity-level destination mass
    destination_mass = pd.DataFrame(
        0.0,
        index=sectors,
        columns=all_municipalities,
    )

    for i in sectors:
        row_pos_i = np.concatenate([positions[(R, i)] for R in provinces])
        destination_mass.loc[i] = (
            Z_seed
            .iloc[row_pos_i, :]
            .clip(lower=0)
            .groupby(level="Municipality", axis=1)
            .sum()
            .sum(axis=0)
            .reindex(all_municipalities)
            .fillna(0.0)
            .values
        )

    ## FLQ intramunicipal estimate
    flq_ijr, flq_sum, flq_summary = calculate_flq_intra(
        Z_seed=Z_seed,
        x_i_seed=x_i_seed,
        delta=delta,
    )

    Z_pos_new = np.zeros(Z_seed.shape, dtype=float)

    for (R, r), grp in flq_ijr.groupby(["Province", "Municipality"], sort=False):

        row_pos = np.array([position[(R, r, i)] for i in sectors], dtype=int)
        col_pos = row_pos.copy()

        mat = (
            grp
            .pivot(
                index="input_sector",
                columns="purchasing_sector",
                values="Z_rr_FLQ_pos",
            )
            .reindex(index=sectors, columns=sectors)
            .fillna(0.0)
            .values
        )

        Z_pos_new[np.ix_(row_pos, col_pos)] = mat

    ## Intermunicipal allocation
    allocation_records = []

    distance_values = distance_229.loc[
        all_municipalities,
        all_municipalities,
    ]

    gravity_map = gravity.set_index("input_sector").to_dict("index")

    for R in provinces:
        R_municipalities = municipalities_by_province[R]

        for S in provinces:
            S_municipalities = municipalities_by_province[S]

            row_pos_all = np.concatenate([positions[(R, i)] for i in sectors])
            col_pos_all = np.concatenate([positions[(S, j)] for j in sectors])

            block = Z_seed.iloc[row_pos_all, col_pos_all].clip(lower=0)
            block_values = block.values

            target_values = (
                block_values
                .reshape(
                    len(sectors),
                    len(R_municipalities),
                    len(sectors),
                    len(S_municipalities),
                )
                .sum(axis=(1, 3))
            )

            target = pd.DataFrame(
                target_values,
                index=sectors,
                columns=sectors,
            )

            if R == S:
                flq_block = (
                    flq_sum
                    .xs(R, level="Province")
                    .unstack("purchasing_sector")
                    .reindex(index=sectors, columns=sectors)
                    .fillna(0.0)
                )
                target = (target - flq_block).clip(lower=0.0)

            time_block = distance_values.loc[
                R_municipalities,
                S_municipalities,
            ].values.astype(float)
            time_block = np.where(time_block <= 0, 1e-9, time_block)

            for i in sectors:

                params = gravity_map.get(i, {})
                method = params.get("method", "simple_trade")
                alpha = params.get("log_origin_coef", 1.0)
                beta = params.get("log_destination_coef", 1.0)
                gamma = params.get("gamma", 0.0)

                row_pos_i = positions[(R, i)]

                O_vec = origin_mass[row_pos_i]
                D_vec = (
                    destination_mass
                    .loc[i]
                    .reindex(S_municipalities)
                    .fillna(0.0)
                    .values
                )

                active_origin = np.array(
                    [
                        not is_zero(R, r, i, zero_set, exception_set)
                        for r in R_municipalities
                    ],
                    dtype=bool,
                )

                if method == "yamada_like_gravity":
                    base_kernel = (
                        (O_vec[:, None] ** alpha)
                        * (D_vec[None, :] ** beta)
                        / (time_block ** gamma)
                    )
                else:
                    base_kernel = O_vec[:, None] * D_vec[None, :]

                if R == S:
                    np.fill_diagonal(base_kernel, 0.0)

                for j in sectors:

                    value = float(target.loc[i, j])
                    if value <= EPS:
                        continue

                    col_pos_j = positions[(S, j)]

                    active_destination = np.array(
                        [
                            not is_zero(S, s, j, zero_set, exception_set)
                            for s in S_municipalities
                        ],
                        dtype=bool,
                    )

                    allowed = active_origin[:, None] & active_destination[None, :]

                    if R == S:
                        np.fill_diagonal(allowed, False)

                    weights, allocation_method = normalize_kernel(
                        kernel=base_kernel,
                        allowed=allowed,
                    )

                    allocated = 0.0
                    if allocation_method == "kernel":
                        allocated_value = value * weights
                        Z_pos_new[np.ix_(row_pos_i, col_pos_j)] += allocated_value
                        allocated = float(allocated_value.sum())

                    allocation_records.append(
                        {
                            "R": R,
                            "S": S,
                            "input_sector": i,
                            "purchasing_sector": j,
                            "target": value,
                            "allocated": allocated,
                            "method": method,
                            "allocation_method": allocation_method,
                        }
                    )

    Z_pos = pd.DataFrame(
        Z_pos_new,
        index=Z_seed.index,
        columns=Z_seed.columns,
    )

    allocation_report = pd.DataFrame(allocation_records)

    target_pos = (
        Z_seed.clip(lower=0)
        .groupby(level=["Province", "Sector"], axis=0)
        .sum()
        .groupby(level=["Province", "Sector"], axis=1)
        .sum()
    )

    result_pos = (
        Z_pos
        .groupby(level=["Province", "Sector"], axis=0)
        .sum()
        .groupby(level=["Province", "Sector"], axis=1)
        .sum()
    )

    diff = result_pos - target_pos

    summary = {
        "total_target": float(target_pos.values.sum()),
        "total_result": float(result_pos.values.sum()),
        "total_unallocated": float(
            allocation_report["target"].sum() - allocation_report["allocated"].sum()
        ),
        "max_abs_block_diff": float(np.abs(diff.values).max()),
        "allocation_rows": len(allocation_report),
        "gravity_rows": int(
            (allocation_report["method"] == "yamada_like_gravity").sum()
        ),
        "simple_trade_rows": int(
            (allocation_report["method"] == "simple_trade").sum()
        ),
        "kernel_rows": int(
            (allocation_report["allocation_method"] == "kernel").sum()
        ),
        "zero_denominator_rows": int(
            (allocation_report["allocation_method"] == "zero_denominator").sum()
        ),
        "zero_support_rows": int(
            (allocation_report["allocation_method"] == "zero_support").sum()
        ),
    }
    summary.update(flq_summary)

    return Z_pos, gravity, allocation_report, summary
