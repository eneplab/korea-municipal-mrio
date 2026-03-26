# src/interregional_calibration.py

import numpy as np
import pandas as pd

def gravity_calibration(
    Z: pd.DataFrame,
    distance_229: pd.DataFrame,
    coef_gravity: pd.DataFrame,
    zero_set: set | None = None,
    exception_set: set | None = None,
):

    zero_set = zero_set or set()
    exception_set = exception_set or set()

    def is_zero(p, m, sector):
        return (p, m, sector) in zero_set and (p, m, sector) not in exception_set

    ## Index preparation
    idx = Z.index.to_frame(index=False)

    municipalities  = idx["Municipality"].unique()
    sectors = idx["Sector"].unique()

    n_municipalities = len(municipalities)
    n_sectors = len(sectors)

    # Municipality to province mapping
    region_hierarchy = (
        idx.drop_duplicates("Municipality")
        .set_index("Municipality")["Province"]
        .to_dict()
    )

    ## Pre-compute aggregates
    # Origin supply
    origin = (
        Z
        .sum(axis=1)
        .groupby(["Municipality", "Sector"])
        .sum()
    )
  
    # Destination demand
    destination = (
        Z
        .sum(axis=0)
        .groupby(level="Municipality")
        .sum()
    )

    # Sector shares at destination
    sector_col_sum = (
        Z
        .sum(axis=0)
        .groupby(["Municipality", "Sector"])
        .sum()
    )

    sector_shares = (
        sector_col_sum
        / sector_col_sum.groupby("Municipality").sum()
    ).fillna(0.0)

    destination_shares = (
        sector_shares
        .unstack("Sector")
        .reindex(index=municipalities, columns=sectors)
        .fillna(0.0)
        .values
    )

    # Distance matrix
    distance_229 = distance_229.loc[municipalities, municipalities].values.astype(float)
    distance_229 = np.where(distance_229 <= 0, 1e-9, distance_229)

    ## Interregional trade support mask
    # Origin-destination pairs
    Z_municipality = Z.groupby(level="Municipality", axis=1).sum()

    seed_support = (
        Z_municipality
        .groupby(level=["Municipality", "Sector"], axis=0)
        .sum()
        .groupby(level="Municipality", axis=0)
        .sum()
        .reindex(index=municipalities, columns=municipalities)
        .fillna(0.0)
        .values
        > 0.0
    )

    ## Intra-municipal self-trade
    Z_municipality_sector = (
        Z_municipality
        .groupby(level=["Municipality", "Sector"], axis=0)
        .sum()
    )

    intra_flows = {
        (m, sector): Z_municipality_sector.at[(m, sector), m]
        for (m, sector) in Z_municipality_sector.index
    }

    # Intra-province preservation
    province_map = np.array([region_hierarchy[m] for m in municipalities])
    intra_province_mask = province_map[:, None] == province_map[None, :]

    ## Gravity calibration
    Z_temp = Z.copy()

    coef_map = (
        coef_gravity
        .set_index("sector")[["alpha", "beta", "gamma"]]
        .to_dict("index")
    )

    # Column index for slicing
    Z_column_index = [
        (region_hierarchy[m], m, sector)
        for m in municipalities
        for sector in sectors
    ]

    for sector_idx, (sector, params) in enumerate(coef_map.items(), start=1):

        if sector not in origin.index.get_level_values("Sector"):
            continue

        alpha, beta, gamma = params["alpha"], params["beta"], params["gamma"]

        # Origin supply vector
        origin_vec = (
            origin
            .xs(sector, level="Sector")
            .reindex(municipalities)
            .fillna(0.0)
            .values
        )

        # Destination demand vector
        destination_vec = (
            destination
            .reindex(municipalities)
            .fillna(0.0)
            .values
            .reshape(1, -1)
        )

        # Exclude municipalities with zero employment
        active_origin = np.array(
            [not is_zero(region_hierarchy[m], m, sector) for m in municipalities],
            dtype=bool
        )

        if active_origin.sum() <= 1:
            continue

        ## Gravity kernel
        gravity_kernel = (
            (origin_vec[:, None] ** alpha)
            * (destination_vec ** beta)
            / (distance_229 ** gamma)
        )

        gravity_kernel[~active_origin, :] = 0.0
        gravity_kernel[~seed_support] = 0.0
        np.fill_diagonal(gravity_kernel, 0.0)

        kernel_row_sum = gravity_kernel.sum(axis=1)
        kernel_row_sum[kernel_row_sum == 0.0] = 1.0

        intra_flow_vec = np.array(
            [intra_flows.get((m, sector), 0.0) for m in municipalities]
        )

        inter_flow_target = np.clip(origin_vec - intra_flow_vec, 0.0, None)

        gravity_weights = (gravity_kernel / kernel_row_sum[:, None]) * inter_flow_target[:, None]

        ## Disaggregate gravity weights into sector allocation
        allocation_block = (
            gravity_weights[:, :, None] * destination_shares[None, :, :]
        ).reshape(n_municipalities, n_municipalities * n_sectors)

        ## Row labels for current sector
        sector_row_labels = [
            (region_hierarchy[m], m, sector)
            for m in municipalities
        ]

        original_block = Z_temp.loc[sector_row_labels, Z_column_index].values

        ## Restore self-trade (diagonal)
        for r in range(n_municipalities):
            start = r * n_sectors
            end = (r + 1) * n_sectors
            allocation_block[r, start:end] = original_block[r, start:end]

        ## Restore intra-province flows
        for r in range(n_municipalities):
            for s in range(n_municipalities):
                if intra_province_mask[r, s]:
                    start = s * n_sectors
                    end = (s + 1) * n_sectors
                    allocation_block[r, start:end] = original_block[r, start:end]

        calibrated_block = np.nan_to_num(allocation_block, nan=0.0)

        Z_temp.loc[sector_row_labels, Z_column_index] = calibrated_block

    Z = Z_temp
    
    return Z