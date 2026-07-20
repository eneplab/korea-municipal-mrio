# src/main.py

from pathlib import Path

import pandas as pd

from metadata.account import FINAL_DEMAND_ITEMS
from metadata.region import MUNICIPALITIES_229, REGION_HIERARCHY
from metadata.sector import SECTOR_76, SECTOR_83, SECTOR_MAPPING_83_TO_76

from src.intermunicipal_allocation import build_positive_seed, estimate_gravity
from src.mrio_io import save_mrio
from src.mrio_preprocessing import mrio_preprocessing
from src.mrgras_balancing import multi_regional_gras
from src.municipal_disaggregation import municipal_disaggregation
from src.utils import (
    build_activity_sets,
    municipality_name_to_id,
    restore_io_labels,
    split_signed,
)

SECTOR_NAME_TO_ID_76 = {name: sector_id for sector_id, name in SECTOR_76.items()}


def load_distance(data_dir, distance_name):

    distance_17 = pd.read_csv(
        data_dir / f"{distance_name}_17.csv",
        index_col=0,
    )

    distance_229 = pd.read_csv(
        data_dir / f"{distance_name}_229.csv",
        header=[0, 1],
        index_col=[0, 1],
    )
    distance_229 = municipality_name_to_id(
        distance_229=distance_229,
        MUNICIPALITIES_229=MUNICIPALITIES_229,
    )

    return distance_17, distance_229


def restore_outputs(Z, M, VA, Y_d, Y_m, x_i, x_j, x_m):

    municipality_ids = list(MUNICIPALITIES_229.keys())
    sector_ids_76 = list(SECTOR_76.keys())

    Z = restore_io_labels(
        Z,
        municipality_ids,
        sector_ids_76,
        MUNICIPALITIES_229,
        SECTOR_76,
        axis="both",
    )

    M = restore_io_labels(
        M,
        municipality_ids,
        sector_ids_76,
        MUNICIPALITIES_229,
        SECTOR_76,
        axis="both",
        is_import=True,
    )

    VA = restore_io_labels(
        VA,
        municipality_ids,
        sector_ids_76,
        MUNICIPALITIES_229,
        SECTOR_76,
        axis="both",
    )

    x_j = restore_io_labels(
        x_j,
        municipality_ids,
        sector_ids_76,
        MUNICIPALITIES_229,
        SECTOR_76,
        axis="both",
    )

    Y_d = restore_io_labels(
        Y_d,
        municipality_ids,
        sector_ids_76,
        MUNICIPALITIES_229,
        SECTOR_76,
        axis="both",
        fd_items=FINAL_DEMAND_ITEMS,
    )

    Y_m = restore_io_labels(
        Y_m,
        municipality_ids,
        sector_ids_76,
        MUNICIPALITIES_229,
        SECTOR_76,
        axis="both",
        is_import=True,
        fd_items=FINAL_DEMAND_ITEMS,
    )

    x_i = restore_io_labels(
        x_i,
        municipality_ids,
        sector_ids_76,
        MUNICIPALITIES_229,
        SECTOR_76,
        axis="both",
    )

    x_m = restore_io_labels(
        x_m,
        municipality_ids,
        sector_ids_76,
        MUNICIPALITIES_229,
        SECTOR_76,
        axis="both",
        is_import=True,
    )

    return Z, M, VA, Y_d, Y_m, x_i, x_j, x_m


def run_pipeline(
    delta=0.30,
    distance_name="travel_time",
    final_demand_proxy="component",
    out_dir=None,
    save_outputs=True,
    verbose=True,
):

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"

    if out_dir is None:
        out_dir = root / "release"

    out_dir = Path(out_dir)

    if save_outputs:
        out_dir.mkdir(parents=True, exist_ok=True)

    ## MRIO preprocessing
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
    ) = mrio_preprocessing(
        benchmark_path=data_dir / "benchmark_io.csv",
        SECTOR_83=SECTOR_83,
        SECTOR_MAPPING_83_TO_76=SECTOR_MAPPING_83_TO_76,
    )

    if verbose:
        print("[DONE] MRIO preprocessing completed")

    ## Load municipal data
    grdp = pd.read_csv(data_dir / "municipal_grdp.csv")
    population = pd.read_csv(data_dir / "population_229.csv")
    expenditure = pd.read_csv(data_dir / "municipal_expenditure.csv")
    zero = pd.read_csv(data_dir / "zero_employee.csv")

    inactive, exceptions = build_activity_sets(
        zero=zero,
        SECTOR_NAME_TO_ID_76=SECTOR_NAME_TO_ID_76,
        MUNICIPALITIES_229=MUNICIPALITIES_229,
    )

    if verbose:
        print("[DONE] Municipal proxy data loaded")

    ## Municipal disaggregation
    (
        Z_seed,
        M,
        Y_d,
        Y_m,
        VA,
        x_j,
        x_i,
        x_m,
        Z_ref,
    ) = municipal_disaggregation(
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
            municipalities_229=MUNICIPALITIES_229,
            grdp=grdp,
            population=population,
            expenditure=expenditure,
            inactive=inactive,
            exceptions=exceptions,
            final_demand_proxy=final_demand_proxy,
        )
    )

    if verbose:
        print("[DONE] Municipal disaggregation completed")

    ## Distance matrices
    distance_17, distance_229 = load_distance(
        data_dir=data_dir,
        distance_name=distance_name,
    )

    if verbose:
        print(f"[DONE] {distance_name} matrices loaded")

    ## Gravity parameters
    gravity = estimate_gravity(
        Z_ref=Z_ref,
        distance_17=distance_17,
    )

    if verbose:
        print("[DONE] Gravity parameter estimation completed")

    ## Positive and negative component handling
    _, Z_neg = split_signed(Z_seed)

    Z_pos = build_positive_seed(
        Z_seed=Z_seed,
        x_i_seed=x_i,
        distance_229=distance_229,
        gravity=gravity,
        inactive=inactive,
        exceptions=exceptions,
        delta=delta,
    )

    Z0 = Z_pos - Z_neg

    if verbose:
        print("[DONE] Positive and signed baseline seeds completed")

    ## Multi-regional balancing
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
        Z=Z0,
        Z_block=Z_block,
        M=M,
        VA=VA,
        Y_d=Y_d,
        Y_m=Y_m,
        x_i=x_i,
        x_j=x_j,
        x_m=x_m,
    )

    if verbose:
        print("[DONE] Multi-regional balancing completed")

    ## Label restoration
    Z, M, VA, Y_d, Y_m, x_i, x_j, x_m = restore_outputs(
        Z=Z,
        M=M,
        VA=VA,
        Y_d=Y_d,
        Y_m=Y_m,
        x_i=x_i,
        x_j=x_j,
        x_m=x_m,
    )

    if verbose:
        print("[DONE] Label restoration completed")

    outputs = {
        "Z": Z,
        "M": M,
        "Y_d": Y_d,
        "Y_m": Y_m,
        "VA": VA,
        "x_i": x_i,
        "x_j": x_j,
        "x_m": x_m,
        "gravity": gravity,
        "mrgras_info": info,
        "delta": delta,
        "distance_name": distance_name,
    }

    ## Save outputs
    if save_outputs:
        MRIO, manifest = save_mrio(
            out_dir=out_dir,
            Z=Z,
            M=M,
            Y_d=Y_d,
            Y_m=Y_m,
            VA=VA,
            x_i=x_i,
            x_j=x_j,
            x_m=x_m,
            info=info,
            municipality_map=MUNICIPALITIES_229,
            sector_map=SECTOR_76,
            config={
                "delta": delta,
                "distance_measure": distance_name,
                "final_demand_proxy": final_demand_proxy,
            },
        )

        outputs["MRIO"] = MRIO
        outputs["manifest"] = manifest

        if verbose:
            print("[DONE] Municipal MRIO output saved")

    return outputs


def main():

    run_pipeline(
        delta=0.30,
        distance_name="travel_time",
        save_outputs=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
