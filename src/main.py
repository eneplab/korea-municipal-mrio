# src/main.py

from pathlib import Path
import argparse
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from metadata.account import FINAL_DEMAND_ITEMS
from metadata.region import MUNICIPALITIES_229, REGION_HIERARCHY
from metadata.sector import SECTOR_76, SECTOR_83, SECTOR_MAPPING_83_TO_76

from src.intermunicipal_allocation import build_positive_seed, estimate_gravity
from src.benchmark_accounting import municipal_transaction_targets, reconcile_benchmark
from src.mrio_io import save_mrio
from src.mrio_preprocessing import mrio_preprocessing, prepare_production_weights
from src.mrgras_balancing import multi_regional_gras
from src.municipal_disaggregation import municipal_disaggregation
from src.utils import (
    allocate_array,
    municipality_name_to_id,
    restore_io_labels,
)

def load_distance(data_dir):

    distance_name = "travel_time"

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


def run_pipeline(out_dir=None, verbose=True):

    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    if not (data_dir / "municipal_workers.csv").is_file():
        raise FileNotFoundError(
            "data/municipal_workers.csv is required. See README.md for the input format."
        )

    out_dir = root / "release" if out_dir is None else Path(out_dir)
    delta = 0.30
    tolerance = 1e-6

    ## MRIO preprocessing
    names = ["Z", "Z_block", "M", "VA", "Y_d", "Y_m", "x_j", "x_i", "x_m"]
    frames = dict(zip(
        names,
        mrio_preprocessing(
            benchmark_path=data_dir / "benchmark_io.csv",
            SECTOR_83=SECTOR_83,
            SECTOR_MAPPING_83_TO_76=SECTOR_MAPPING_83_TO_76,
        ),
    ))
    reported_output = frames["x_i"].copy()
    frames = reconcile_benchmark(frames)
    Z, Z_block, M, VA, Y_d, Y_m, x_j, x_i, x_m = (
        frames[name] for name in names
    )
    del frames

    if verbose:
        print("[DONE] MRIO preprocessing completed", flush=True)

    ## Municipal inputs and production weights
    grdp = pd.read_csv(data_dir / "municipal_grdp.csv")
    population = pd.read_csv(data_dir / "population_229.csv")
    expenditure = pd.read_csv(data_dir / "municipal_expenditure.csv")
    allocation = prepare_production_weights(data_dir, Z, reported_output)

    if allocation.duplicated(["municipality_id", "io"]).any() or len(allocation) != 17404:
        raise ValueError("Production weights must contain 17,404 unique accounts")

    inactive = set(zip(
        allocation.loc[allocation.output_weight.eq(0), "province_en"],
        allocation.loc[allocation.output_weight.eq(0), "municipality_id"],
        allocation.loc[allocation.output_weight.eq(0), "io"],
    ))
    exceptions = set()
    provincial_Z = Z.copy()

    if verbose:
        print("[DONE] Municipal inputs and production weights prepared", flush=True)

    with TemporaryDirectory(prefix="municipal_mrio_") as temporary:
        work_dir = Path(temporary)

        ## Municipal disaggregation
        Z_seed, M, Y_d, Y_m, VA, x_j, x_i, x_m, Z_ref = municipal_disaggregation(
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
                allocation=allocation,
                work_dir=work_dir,
            )
        )

        if verbose:
            print("[DONE] Municipal disaggregation completed", flush=True)

        ## Gravity estimation
        distance_17, distance_229 = load_distance(data_dir)
        gravity = estimate_gravity(Z_ref=Z_ref, distance_17=distance_17)
        row_targets, column_targets = municipal_transaction_targets(
            provincial_Z, allocation, Z_seed.index
        )

        if verbose:
            print("[DONE] Gravity parameter estimation completed", flush=True)

        ## FLQ and gravity allocation
        Z_pos = build_positive_seed(
            Z_seed=Z_seed,
            x_i_seed=x_i,
            distance_229=distance_229,
            gravity=gravity,
            inactive=inactive,
            exceptions=exceptions,
            delta=delta,
            work_dir=work_dir,
        )

        ## Signed matrix for balancing
        Z0_values = allocate_array(Z_seed.shape, work_dir, "allocation_adjusted")
        for start in range(0, len(Z_seed), 256):
            seed = Z_seed.iloc[start:start + 256, :].to_numpy()
            Z0_values[start:start + len(seed), :] = (
                Z_pos.iloc[start:start + len(seed), :].to_numpy()
                + np.minimum(seed, 0)
            )
        Z0 = pd.DataFrame(
            Z0_values, index=Z_seed.index, columns=Z_seed.columns, copy=False
        )
        del Z_pos, Z_seed, seed
        for name in ["initial_seed", "positive_allocation"]:
            (work_dir / (name + ".npy")).unlink()

        if verbose:
            print("[DONE] FLQ and gravity allocation completed", flush=True)

        ## Multi-regional balancing
        Z, M, VA, Y_d, Y_m, x_i, x_j, x_m, info = multi_regional_gras(
            Z=Z0,
            Z_block=Z_block,
            M=M,
            VA=VA,
            Y_d=Y_d,
            Y_m=Y_m,
            x_i=x_i,
            x_j=x_j,
            x_m=x_m,
            work_dir=work_dir,
            verbose=verbose,
            mult_tol=tolerance,
            constraint_tol=tolerance,
            row_targets=row_targets,
            column_targets=column_targets,
        )
        del Z0, Z0_values
        (work_dir / "allocation_adjusted.npy").unlink()

        if verbose:
            print("[DONE] Multi-regional balancing completed", flush=True)

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

    ## Save the baseline dataset
    MRIO = save_mrio(
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
    )

    if verbose:
        print("[DONE] Municipal MRIO output saved", flush=True)

    return dict(
        Z=Z,
        M=M,
        Y_d=Y_d,
        Y_m=Y_m,
        VA=VA,
        x_i=x_i,
        x_j=x_j,
        x_m=x_m,
        gravity=gravity,
        mrgras_info=info,
        MRIO=MRIO,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()
