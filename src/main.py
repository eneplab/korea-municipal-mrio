# src/main.py

import pandas as pd
from pathlib import Path
import warnings

from metadata.sector import SECTOR_83, SECTOR_76, SECTOR_MAPPING_83_TO_76
from metadata.region import REGION_HIERARCHY, MUNICIPALITIES_229
from metadata.account import FINAL_DEMAND_ITEMS

from src.sectoral_aggregation import sectoral_aggregation
from src.regional_disaggregation import regional_disaggregation
from src.gravity_parameters import gravity_parameters
from src.interregional_calibration import gravity_calibration
from src.multi_regional_balancing import multi_regional_gras
from src.utils import municipality_name_to_id
from src.utils import restore_io_labels
from src.utils import merge_io
from src.utils import sanitize_for_parquet

warnings.filterwarnings("ignore")

SECTOR_NAME_TO_ID_83 = {name: sector_id for sector_id, name in SECTOR_83.items()}
SECTOR_NAME_TO_ID_76 = {name: sector_id for sector_id, name in SECTOR_76.items()}

def main():

    ROOT = Path(__file__).resolve().parents[1]
    DATA = ROOT / "data"
    OUT  = ROOT / "release"
    OUT.mkdir(exist_ok=True)

    ## Load IRIO benchmark table
    IRIO = pd.read_csv(
        DATA / "benchmark_io.csv",
        header=[0, 1],
        index_col=[0, 1],
    )
    
    print("[DONE] IRIO benchmark table loaded")

    ## Block split
    Z   = IRIO.iloc[:1411, :1411]
    M   = IRIO.iloc[1411:-2, :1411]
    VA  = IRIO.iloc[[-2], :1411]
    Y_d = IRIO.iloc[:1411, 1411:-1]
    Y_m = IRIO.iloc[1411:-2, 1411:-1]
    x_j = IRIO.iloc[[-1], :1411]
    x_i = IRIO.iloc[:1411, [-1]]
    x_m = IRIO.iloc[1411:-2, [-1]]

    print("[DONE] Raw IO blocks extracted")

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

    print("[DONE] Sector name to ID mapping completed")

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

    print("[DONE] Sectoral aggregation completed")

    ## Load economic and employment data
    grdp = pd.read_csv(DATA / "regional_grdp.csv")
    zero = pd.read_csv(DATA / "zero_employee.csv")
    
    zero["sector"] = zero["Sector"].map(SECTOR_NAME_TO_ID_76)

    zero_set = set(zip(zero["Province"], zero["ID"], zero["sector"]))
    exception_set = {("Sejong", 72, SECTOR_NAME_TO_ID_76["Ships"])}

    print("[DONE] GRDP and employment data loaded")

    ## Regional disaggregation
    (
        Z,
        M,
        Y_d,
        Y_m,
        VA,
        x_j,
        x_i,
        x_m,
        Z_ref,
    ) = regional_disaggregation(
        dict(
            Z=Z,
            M=M,
            Y_d=Y_d,
            Y_m=Y_m,
            VA=VA,
            x_j=x_j,
            x_i=x_i,
            x_m=x_m,
            region_hierarchy=REGION_HIERARCHY,
            grdp=grdp,
            zero_set=zero_set,
            exception_set=exception_set,
        )
    )

    print("[DONE] Regional disaggregation completed")

    ## Load distance matrices
    distance_229 = pd.read_csv(
        DATA / "distance_229.csv",
        header=[0, 1],
        index_col=[0, 1],
    )

    distance_229 = municipality_name_to_id(
        distance_229=distance_229,
        MUNICIPALITIES_229=MUNICIPALITIES_229,
    )

    distance_17 = pd.read_csv(
        DATA / "distance_17.csv",
        index_col=0,
    )

    print("[DONE] Distance matrices loaded")

    ## Gravity parameter estimation
    coef_gravity = gravity_parameters(
        Z_ref=Z_ref,
        distance_17=distance_17,
        verbose=True,
    )

    print("[DONE] Gravity parameter estimation completed")

    ## Interregional calibration
    Z = gravity_calibration(
        Z=Z,
        distance_229=distance_229,
        coef_gravity=coef_gravity,
        zero_set=zero_set,
        exception_set=exception_set,
    )
    
    print("[DONE] Interregional calibration completed")

    ## Multi-regional balancing (MR-GRAS)
    (
        Z,
        M,
        VA,
        Y_d,
        Y_m,
        x_i,
        x_j,
        x_m,
        info,
    ) = multi_regional_gras(
        Z=Z,
        Z_block=Z_block,
        M=M,
        VA=VA,
        Y_d=Y_d,
        Y_m=Y_m,
        x_i=x_i,
        x_j=x_j,
        x_m=x_m,
    )
 
    print("[DONE] Multi-regional balancing completed")
    print("[INFO]", info)

    ## Label restoration
    municipality_ids = list(MUNICIPALITIES_229.keys())
    sector_ids_76 = list(SECTOR_76.keys())

    Z = restore_io_labels(
        Z, municipality_ids, sector_ids_76,
        MUNICIPALITIES_229, SECTOR_76, axis="both"
    )

    M = restore_io_labels(
        M, municipality_ids, sector_ids_76,
        MUNICIPALITIES_229, SECTOR_76,
        axis="both", is_import=True
    )

    VA = restore_io_labels(
        VA, municipality_ids, sector_ids_76,
        MUNICIPALITIES_229, SECTOR_76,
        axis="both"
    )

    x_j = restore_io_labels(
        x_j, municipality_ids, sector_ids_76,
        MUNICIPALITIES_229, SECTOR_76,
        axis="both"
    )

    Y_d = restore_io_labels(
        Y_d, municipality_ids, sector_ids_76,
        MUNICIPALITIES_229, SECTOR_76,
        axis="both",
        fd_items=FINAL_DEMAND_ITEMS
    )

    Y_m = restore_io_labels(
        Y_m, municipality_ids, sector_ids_76,
        MUNICIPALITIES_229, SECTOR_76,
        axis="both",
        is_import=True,
        fd_items=FINAL_DEMAND_ITEMS
    )

    x_i = restore_io_labels(
        x_i, municipality_ids, sector_ids_76,
        MUNICIPALITIES_229, SECTOR_76,
        axis="both"
    )

    x_m = restore_io_labels(
        x_m, municipality_ids, sector_ids_76,
        MUNICIPALITIES_229, SECTOR_76,
        axis="both",
        is_import=True
    ) 

    print("[DONE] Label restoration completed")

    ## Final MRIO merge
    MRIO = merge_io(
        Z,
        M,
        VA,
        x_j,
        Y_d,
        Y_m,
        x_i,
        x_m,
    )

    MRIO = sanitize_for_parquet(MRIO)
    MRIO.to_parquet(OUT / "MRIO.parquet")

    print("[DONE] MRIO dataset merged and saved")

if __name__ == "__main__":
    main()