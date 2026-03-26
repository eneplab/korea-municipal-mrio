# src/gravity_parameters.py

import numpy as np
import pandas as pd
import statsmodels.api as sm

def gravity_parameters(
    Z_ref,
    distance_17,
    min_obs: int = 50,
    excluded_sectors: set = None,
    verbose: bool = False,
):
    if excluded_sectors is None:
        excluded_sectors = set(range(59, 84))

    sectors = Z_ref.index.get_level_values("Sector").unique()
    sectors = [sector for sector in sectors if sector not in excluded_sectors]

    provinces = Z_ref.index.get_level_values("Province").unique()

    results = []
    gravity_sectors = []

    ## Destination demand
    destination = (
        Z_ref
        .sum(axis=0)
        .groupby(level="Province")
        .sum()
    )

    ## OLS regression
    for sector in sectors:
        
        # Origin supply
        try:
            origin = (
                Z_ref
                .xs(sector, level="Sector")
                .sum(axis=1)
            )
        except KeyError:
            continue

        # Interregional trade flows
        trade_flows = (
            Z_ref
            .xs(sector, level="Sector")
            .groupby(level="Province", axis=1)
            .sum()
        )

        # Observations
        obs = []

        for R in provinces:
            for S in provinces:
                if R == S:
                    continue

                try:
                    T_RS = float(trade_flows.loc[R, S])
                    O_R  = float(origin.loc[R])
                    D_S = float(destination.loc[S])
                    dist = float(distance_17.loc[R, S])
                except KeyError:
                    continue

                # Only positive flows
                if T_RS <= 0 or O_R <= 0 or D_S <= 0 or dist <= 0:
                    continue

                obs.append([
                    np.log(T_RS),
                    np.log(O_R),
                    np.log(D_S),
                    np.log(dist),
                ])

        # if len(obs) < min_obs:
        #     if verbose:
        #         print(f"[SKIP] {sector}: insufficient obs ({len(obs)} < {min_obs})")
        #     continue

        ## OLS estimation
        obs_list = pd.DataFrame(
            obs,
            columns=["log_T", "log_origin", "log_destination", "log_distance"],
        )

        X = sm.add_constant(obs_list[["log_origin", "log_destination", "log_distance"]])
        y = obs_list["log_T"]

        try:
            ols_result = sm.OLS(y, X).fit()
        except Exception as e:
            if verbose:
                print(f"[ERROR] {sector}: regression failed - {e}")
            continue

        ## Significance test (F-test)
        is_significant = ols_result.f_pvalue < 0.05

        # if verbose:
        #     print(
        #         f"[GRAVITY] sector={sector} | "
        #         f"α={ols_result.params['log_origin']:.3f}, "
        #         f"β={ols_result.params['log_destination']:.3f}, "
        #         f"γ={-ols_result.params['log_distance']:.3f}, "
        #         f"R²={ols_result.rsquared:.3f}, "
        #         f"n={len(obs_list)}"
        #     )

        results.append({
            "sector"            : sector,
            "alpha"             : ols_result.params["log_origin"],
            "beta"              : ols_result.params["log_destination"],
            "gamma"             : -ols_result.params["log_distance"],
            "n_obs"             : len(obs_list),
            "r2"                : ols_result.rsquared,
        })

        gravity_sectors.append(sector)

    ## Summary
    coef_gravity = pd.DataFrame(results)

    # if verbose:
    #     print(f"\n[SUMMARY] {len(gravity_sectors)}/{len(sectors)} sectors estimated")
    #     print(f"  - Min observations : {min_obs}")
    #     print(f"  - Excluded sectors : 59–83")

    return coef_gravity