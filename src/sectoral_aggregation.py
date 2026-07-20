# src/sectoral_aggregation.py

import numpy as np
import pandas as pd


def sectoral_aggregation(
    Z, M,
    VA,
    Y_d, Y_m,
    x_j, x_i, x_m,
    sector_ids_83,
    sector_mapping,
):

    ## Province list
    provinces = Z.index.get_level_values("Province").unique().tolist()
    n_provinces = len(provinces)

    ## Sector ID mapping
    sector_ids_76 = []
    seen = set()
    for id_83 in sector_ids_83:
        id_76 = sector_mapping[id_83]
        if id_76 not in seen:
            sector_ids_76.append(id_76)
            seen.add(id_76)

    n_sectors_83 = len(sector_ids_83)
    n_sectors_76 = len(sector_ids_76)

    sector_agg = np.zeros((n_sectors_83, n_sectors_76))
    sector_pos = {id_83: j for j, id_83 in enumerate(sector_ids_76)}
    for i, id_83 in enumerate(sector_ids_83):
        sector_agg[i, sector_pos[sector_mapping[id_83]]] = 1.0

    full_agg = np.kron(np.eye(n_provinces), sector_agg)

    ## Index and matrix aggregation
    idx_agg = pd.MultiIndex.from_product(
        [provinces, sector_ids_76],
        names=["Province", "Sector"],
    )

    # Domestic intermediate
    Z = pd.DataFrame(
        full_agg.T @ Z.values @ full_agg,
        index=idx_agg,
        columns=idx_agg,
    )

    # Import intermediate
    M = pd.DataFrame(
        sector_agg.T @ M.values @ full_agg,
        index=pd.MultiIndex.from_product(
            [["Imports"], sector_ids_76],
            names=["Province", "Sector"],
        ),
        columns=idx_agg,
    )

    # Domestic final demand
    Y_d = pd.DataFrame(
        full_agg.T @ Y_d.values,
        index=idx_agg,
        columns=Y_d.columns,
    )
    Y_d.columns.names = ["Province", "Final Demand"]

    # Import final demand
    Y_m = pd.DataFrame(
        sector_agg.T @ Y_m.values,
        index=pd.MultiIndex.from_product(
            [["Imports"], sector_ids_76],
            names=["Province", "Sector"],
        ),
        columns=Y_m.columns,
    )
    Y_m.columns.names = ["Province", "Final Demand"]

    # Value added
    VA = pd.DataFrame(
        VA.values @ full_agg,
        index=VA.index,
        columns=idx_agg,
    )
    VA.index.names = ["Account", "Value Added"]

    # Total input
    x_j = pd.DataFrame(
        x_j.values @ full_agg,
        index=x_j.index,
        columns=idx_agg,
    )
    x_j.index.names = ["Account", "Total Input"]

    # Total output
    x_i = pd.DataFrame(
        full_agg.T @ x_i.values,
        index=idx_agg,
        columns=x_i.columns,
    )
    x_i.columns.names = ["Account", "Total Output"]

    # Import total output
    x_m = pd.DataFrame(
        sector_agg.T @ x_m.values,
        index=pd.MultiIndex.from_product(
            [["Imports"], sector_ids_76],
            names=["Province", "Sector"],
        ),
        columns=x_m.columns,
    )
    x_m.columns.names = ["Account", "Total Output"]

    ## Block constraint reference
    Z_block = Z.copy()

    return (
        Z,
        Z_block,
        M,
        VA,
        Y_d,
        Y_m,
        x_j,
        x_i,
        x_m,
    )
