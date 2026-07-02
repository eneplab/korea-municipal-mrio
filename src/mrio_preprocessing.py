# src/mrio_preprocessing.py

import pandas as pd

from src.sectoral_aggregation import sectoral_aggregation


## Load benchmark IRIO and aggregate sectors
def mrio_preprocessing(
    benchmark_path,
    SECTOR_83,
    SECTOR_MAPPING_83_TO_76,
):

    SECTOR_NAME_TO_ID_83 = {name: sector_id for sector_id, name in SECTOR_83.items()}

    ## Load IRIO benchmark table
    IRIO = pd.read_csv(
        benchmark_path,
        header=[0, 1],
        index_col=[0, 1],
    )

    ## Block split
    Z = IRIO.iloc[:1411, :1411]
    M = IRIO.iloc[1411:-2, :1411]
    VA = IRIO.iloc[[-2], :1411]
    Y_d = IRIO.iloc[:1411, 1411:-1]
    Y_m = IRIO.iloc[1411:-2, 1411:-1]
    x_j = IRIO.iloc[[-1], :1411]
    x_i = IRIO.iloc[:1411, [-1]]
    x_m = IRIO.iloc[1411:-2, [-1]]

    ## Sector name to ID mapping
    def sector_allocation(idx):
        return pd.MultiIndex.from_arrays(
            [
                idx.get_level_values(0),
                idx.get_level_values(1).map(SECTOR_NAME_TO_ID_83),
            ],
            names=["Province", "Sector"],
        )

    Z.index = sector_allocation(Z.index)
    Z.columns = Z.index
    M.index = sector_allocation(M.index)
    Y_d.index = sector_allocation(Y_d.index)
    Y_m.index = sector_allocation(Y_m.index)
    x_i.index = sector_allocation(x_i.index)
    x_m.index = sector_allocation(x_m.index)

    sector_ids_83 = Z.index.get_level_values("Sector").dropna().unique().tolist()

    ## Sectoral aggregation
    (
        Z,
        Z_block,
        M,
        VA,
        Y_d,
        Y_m,
        x_j,
        x_i,
        x_m,
    ) = sectoral_aggregation(
        Z,
        M,
        VA,
        Y_d,
        Y_m,
        x_j,
        x_i,
        x_m,
        sector_ids_83,
        SECTOR_MAPPING_83_TO_76,
    )

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
