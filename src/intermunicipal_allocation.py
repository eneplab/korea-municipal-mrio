# src/intermunicipal_allocation.py

import numpy as np
import pandas as pd

from src.intramunicipal_allocation import calculate_flq_intra
from src.utils import allocate_array, group_sum, is_inactive


EPS = 1e-12


def fit_log_ols(obs, variable_names):

    if len(obs) == 0:
        return None

    obs = pd.DataFrame(obs, columns=["log_T"] + variable_names)
    y = obs["log_T"].values.astype(float)
    X_raw = obs[variable_names].values.astype(float)
    X = np.column_stack([np.ones(len(obs)), X_raw])

    n_obs = len(obs)
    n_params = X.shape[1]
    if n_obs <= n_params:
        return None

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    result = {"n_obs": n_obs}
    names = ["const"] + variable_names

    for idx, name in enumerate(names):
        result[f"{name}_coef"] = beta[idx]

    return result


def estimate_gravity(Z_ref, distance_17):

    ## Province-level positive flows
    Z_pos = Z_ref.clip(lower=0)
    provinces = Z_pos.index.get_level_values("Province").unique().tolist()
    sectors = Z_pos.index.get_level_values("Sector").unique().tolist()

    results = []
    for i in sectors:

        trade_flows = group_sum(
            Z_pos.xs(i, level="Sector"),
            level="Province",
            axis=1,
        )

        origin = trade_flows.sum(axis=1)
        destination = trade_flows.sum(axis=0)

        obs = []

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

                if O_R <= 0 or D_S <= 0 or dist <= 0:
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

        row = {"input_sector": i}

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
            row["method"] = "gravity"
            row["gamma"] = -row["log_distance_coef"]

        results.append(row)

    gravity = pd.DataFrame(results)

    return gravity


def build_positive_seed(
    Z_seed,
    x_i_seed,
    distance_229,
    gravity,
    inactive,
    exceptions,
    delta=0.30,
    work_dir=None,
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

    ## Positive origin mass
    origin_mass = np.zeros(len(Z_seed))
    for start in range(0, len(Z_seed), 256):
        values = Z_seed.iloc[start:start + 256, :].to_numpy()
        origin_mass[start:start + len(values)] = np.maximum(values, 0).sum(axis=1)

    ## Commodity-level destination mass
    destination_mass = pd.DataFrame(
        0.0,
        index=sectors,
        columns=all_municipalities,
    )

    for i in sectors:
        row_pos_i = np.concatenate([positions[(R, i)] for R in provinces])
        destination_mass.loc[i] = (
            group_sum(
                Z_seed.iloc[row_pos_i, :].clip(lower=0),
                level="Municipality",
                axis=1,
            )
            .sum(axis=0)
            .reindex(all_municipalities)
            .fillna(0.0)
            .values
        )

    ## FLQ intramunicipal estimate
    flq_ijr, flq_sum, _ = calculate_flq_intra(
        Z_seed=Z_seed,
        x_i_seed=x_i_seed,
        delta=delta,
    )
    local_activity = np.array([
        not is_inactive(p, r, i, inactive, exceptions)
        and not is_inactive(p, r, j, inactive, exceptions)
        for p, r, i, j in flq_ijr[
            ["Province", "Municipality", "input_sector", "purchasing_sector"]
        ].itertuples(index=False, name=None)
    ])
    flq_ijr.loc[~local_activity, "Z_rr_FLQ_pos"] = 0.0
    flq_sum = flq_ijr.groupby(
        ["Province", "input_sector", "purchasing_sector"], sort=False
    )["Z_rr_FLQ_pos"].sum()

    Z_pos_new = allocate_array(Z_seed.shape, work_dir, "positive_allocation")
    unallocated = []

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
            benchmark_values = target_values.copy()

            if R == S:
                flq_block = (
                    flq_sum
                    .xs(R, level="Province")
                    .unstack("purchasing_sector")
                    .reindex(index=sectors, columns=sectors)
                    .fillna(0.0)
                )
                target_values = np.maximum(target_values - flq_block.to_numpy(), 0.0)

            time_block = distance_values.loc[
                R_municipalities,
                S_municipalities,
            ].values.astype(float)
            if R == S:
                offdiagonal = ~np.eye(len(R_municipalities), dtype=bool)
            else:
                offdiagonal = np.ones(time_block.shape, dtype=bool)
            if not np.isfinite(time_block).all() or (time_block[offdiagonal] <= 0).any():
                raise ValueError("Distance matrix has invalid off-diagonal values")
            time_block = np.where(time_block <= 0, 1.0, time_block)

            for idx_i, i in enumerate(sectors):

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
                        not is_inactive(R, r, i, inactive, exceptions)
                        for r in R_municipalities
                    ],
                    dtype=bool,
                )

                positive_mass = (O_vec[:, None] > 0) & (D_vec[None, :] > 0)
                log_origin = np.log(np.maximum(O_vec, np.finfo(float).tiny))
                log_destination = np.log(np.maximum(D_vec, np.finfo(float).tiny))
                if method != "gravity":
                    alpha, beta, gamma = 1.0, 1.0, 0.0
                log_kernel = (
                    alpha * log_origin[:, None]
                    + beta * log_destination[None, :]
                    - gamma * np.log(time_block)
                )

                for idx_j, j in enumerate(sectors):

                    value = float(target_values[idx_i, idx_j])
                    benchmark_value = float(benchmark_values[idx_i, idx_j])
                    if benchmark_value <= EPS:
                        continue

                    col_pos_j = positions[(S, j)]

                    active_destination = np.array(
                        [
                            not is_inactive(S, s, j, inactive, exceptions)
                            for s in S_municipalities
                        ],
                        dtype=bool,
                    )

                    allowed = active_origin[:, None] & active_destination[None, :] & positive_mass

                    if R == S:
                        local_allowed = np.diag(allowed).copy()
                        np.fill_diagonal(allowed, False)
                        if not allowed.any() and not local_allowed.any():
                            Z_pos_new[np.ix_(row_pos_i, col_pos_j)] = 0.0
                            unallocated.append((R, S, i, j))
                            continue
                        if not allowed.any() and local_allowed.any():
                            # No cross-municipal support: conserve over admissible local pairs.
                            old_local = Z_pos_new[row_pos_i, col_pos_j].copy()
                            prior = np.where(local_allowed, old_local, 0.0)
                            if prior.sum() <= 0:
                                signed_seed_block = Z_seed.iloc[row_pos_i, col_pos_j].to_numpy()
                                prior = np.where(
                                    local_allowed,
                                    np.maximum(np.diag(signed_seed_block), 0),
                                    0.0,
                                )
                            if prior.sum() <= 0:
                                raise ValueError("Admissible local pairs lack a positive allocation prior")
                            local_values = benchmark_value * prior / prior.sum()
                            Z_pos_new[np.ix_(row_pos_i, col_pos_j)] = np.diag(local_values)
                            continue

                    if value <= EPS:
                        continue

                    if allowed.any():
                        weights = np.zeros(log_kernel.shape)
                        shifted = log_kernel[allowed] - log_kernel[allowed].max()
                        weights[allowed] = np.exp(shifted)
                        weights /= weights.sum()
                        allocated_value = value * weights
                        Z_pos_new[np.ix_(row_pos_i, col_pos_j)] += allocated_value
                    else:
                        unallocated.append((R, S, i, j))

    Z_pos = pd.DataFrame(
        Z_pos_new,
        index=Z_seed.index,
        columns=Z_seed.columns,
        copy=False,
    )
    if unallocated:
        raise ValueError(
            "Positive benchmark transactions have no municipal allocation pair: "
            + str(unallocated[:3])
        )

    ## Provincial transaction conservation
    block_checks = []
    for R in provinces:
        rows = np.concatenate([positions[(R, i)] for i in sectors])
        for S in provinces:
            cols = np.concatenate([positions[(S, j)] for j in sectors])
            shape = (
                len(sectors), len(municipalities_by_province[R]),
                len(sectors), len(municipalities_by_province[S]),
            )
            expected = (
                np.maximum(Z_seed.iloc[rows, cols].to_numpy(), 0)
                .reshape(shape).sum(axis=(1, 3))
            )
            estimated = Z_pos_new[np.ix_(rows, cols)].reshape(shape).sum(axis=(1, 3))
            error = np.abs(estimated - expected)
            scaled = error / np.maximum(np.abs(expected), 1.0)
            block_checks.append(float(scaled.max()))
    if max(block_checks) >= 1e-10:
        raise ValueError("Positive allocation failed provincial sector-pair conservation")

    return Z_pos
