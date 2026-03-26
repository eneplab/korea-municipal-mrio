# src/utils.py

import pandas as pd

## Convert distance matrix index from names to municipality IDs
def municipality_name_to_id(distance_229, MUNICIPALITIES_229):

    name_to_id = {
        (province, municipality): region_id
        for region_id, (province, municipality) in MUNICIPALITIES_229.items()
    }

    new_index = [
        name_to_id[(p, m)]
        for p, m in distance_229.index
        if (p, m) in name_to_id
    ]

    new_columns = [
        name_to_id[(p, m)]
        for p, m in distance_229.columns
        if (p, m) in name_to_id
    ]

    distance_id = pd.DataFrame(index=new_index, columns=new_columns)

    # Map distance values
    for (province_row, municipality_row), region_id_row in name_to_id.items():
        for (province_col, municipality_col), region_id_col in name_to_id.items():

            if (
                (province_row, municipality_row) in distance_229.index
                and (province_col, municipality_col) in distance_229.columns
            ):
                distance_id.loc[region_id_row, region_id_col] = distance_229.loc[
                    (province_row, municipality_row), (province_col, municipality_col)
                ]

    return distance_id.astype(float)

## Restore multi-index labels
def restore_io_labels(
    obj,
    region_ids,
    sector_ids,
    MUNICIPALITIES_229,
    SECTOR_76,
    axis="both",
    is_import=False,
    fd_items=None,
):
    df = obj.copy()

    # Reset index
    df.index = range(len(df.index))
    df.columns = range(len(df.columns))

    def safe(x):
        if pd.isna(x):
            return ""
        return str(x)

    # Index builders
    def build_io_index():

        province = []
        municipality = []
        sector = []

        for region_id in region_ids:

            p, m = MUNICIPALITIES_229[region_id]

            for sector_id in sector_ids:
                province.append(safe(p))
                municipality.append(safe(m))
                sector.append(safe(SECTOR_76[sector_id]))

        return pd.MultiIndex.from_arrays(
            [province, municipality, sector],
            names=["Province", "Municipality", "Sector"],
        )

    def build_import_index():

        province = ["Imports"] * len(sector_ids)
        municipality = [""] * len(sector_ids)
        sector = [safe(SECTOR_76[sector_id]) for sector_id in sector_ids]

        return pd.MultiIndex.from_arrays(
            [province, municipality, sector],
            names=["Province", "Municipality", "Sector"],
        )

    def build_account_index(label):

        if isinstance(label, tuple):
            label = label[-1]
        return pd.MultiIndex.from_arrays(
            [["Account"], [""], [safe(label)]],
            names=["Province", "Municipality", "Sector"],
        )

    def build_final_demand_index():

        province = []
        municipality = []
        fd_col = []

        for region_id in region_ids:

            p, m = MUNICIPALITIES_229[region_id]

            for fd in fd_items:
                province.append(safe(p))
                municipality.append(safe(m))
                fd_col.append(safe(fd))

        return pd.MultiIndex.from_arrays(
            [province, municipality, fd_col],
            names=["Province", "Municipality", "Final Demand"],
        )

    ## Apply index
    if axis in ("index", "both"):

        if is_import:
            df.index = build_import_index()

        elif len(df.index) == 1:
            df.index = build_account_index(obj.index[0])

        else:
            df.index = build_io_index()

    # Apply columns
    if axis in ("columns", "both"):

        if fd_items is not None:
            df.columns = build_final_demand_index()

        elif len(df.columns) == 1:
            df.columns = build_account_index(obj.columns[0])

        else:
            df.columns = build_io_index()

    return df


## Merge final MRIO table
def merge_io(
    Z,
    M,
    VA,
    x_j,
    Y_d,
    Y_m,
    x_i,
    x_m,
):

    # Row blocks
    trade_va_input = pd.concat(
        [Z, M, VA, x_j],
        axis=0,
    )

    final_demand = pd.concat(
        [Y_d, Y_m],
        axis=0,
    )

    total_output = pd.concat(
        [x_i, x_m],
        axis=0,
    )

    # Column merge
    MRIO = pd.concat(
        [trade_va_input, final_demand, total_output],
        axis=1,
    )

    return MRIO

## Convert all values to float for parquet format
def sanitize_for_parquet(MRIO):

    MRIO = MRIO.copy()
    MRIO = MRIO.apply(pd.to_numeric, errors="coerce")
    MRIO = MRIO.astype(float)

    return MRIO