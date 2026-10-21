# src/mrio_preprocessing.py

import pandas as pd
import numpy as np
from pathlib import Path

from src.sectoral_aggregation import sectoral_aggregation
from metadata.region import MUNICIPALITIES_229
from metadata.sector import SECTOR_76
from metadata.account import VA_COMPONENTS


## Load benchmark IRIO and aggregate sectors
def mrio_preprocessing(
    benchmark_path,
    SECTOR_83,
    SECTOR_MAPPING_83_TO_76,
):

    benchmark_path = Path(benchmark_path)

    SECTOR_NAME_TO_ID_83 = {name: sector_id for sector_id, name in SECTOR_83.items()}

    ## Load IRIO benchmark table
    IRIO = pd.read_csv(
        benchmark_path,
        header=[0, 1],
        index_col=[0, 1],
    )

    ## Block split
    Z = IRIO.iloc[:1411, :1411]
    if IRIO.shape != (1499, 1531):
        raise ValueError("benchmark_io.csv must contain 83 sectors, four VA rows, and seven final-demand categories")
    M = IRIO.iloc[1411:1494, :1411]
    VA = IRIO.iloc[1494:1498, :1411]
    if VA.index.get_level_values(-1).tolist() != VA_COMPONENTS:
        raise ValueError("The four value-added rows are missing or out of order")
    Y_d = IRIO.iloc[:1411, 1411:-1]
    Y_m = IRIO.iloc[1411:1494, 1411:-1]
    x_j = IRIO.iloc[[-1], :1411]
    x_i = IRIO.iloc[:1411, [-1]]
    x_m = IRIO.iloc[1411:1494, [-1]]

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

    return Z, Z_block, M, VA, Y_d, Y_m, x_j, x_i, x_m


def prepare_production_weights(data_dir, Z_ref, reported_output, workers=None):

    data_dir = Path(data_dir)
    bridge = pd.read_csv(
        data_dir.parent / "metadata/IO_KSIC_activity.csv",
        dtype={"ksic5": str},
    )
    if bridge.ksic5.duplicated().any():
        raise ValueError("KSIC categories must have a unique IO/activity assignment")

    ## The number of workers
    if workers is None:
        path = data_dir / "municipal_workers.csv"
        if not path.is_file():
            raise FileNotFoundError(
                "data/municipal_workers.csv is required. See README.md for the input format."
            )
        workers = pd.read_csv(path, dtype={"ksic5": str})

    workers = workers.copy()
    required = {"municipality_id", "number_of_workers"}
    if not required.issubset(workers):
        raise ValueError("Worker data require municipality_id and number_of_workers")

    values = pd.to_numeric(workers.number_of_workers, errors="raise").to_numpy(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            "Worker data must contain nonnegative finite counts; "
            "unavailable entries cannot be zero-filled"
        )
    workers["number_of_workers"] = values
    if not workers.municipality_id.isin(MUNICIPALITIES_229).all():
        raise ValueError("Worker data contain an unknown municipality ID")

    ## KSIC and economic-activity mapping
    if "ksic5" in workers:
        workers["ksic5"] = workers.ksic5.str.zfill(5)
        if workers.duplicated(["municipality_id", "ksic5"]).any():
            raise ValueError("Duplicate municipality/KSIC count")

        full = pd.MultiIndex.from_product(
            [MUNICIPALITIES_229, bridge.ksic5],
            names=["municipality_id", "ksic5"],
        )
        actual = pd.MultiIndex.from_frame(workers[["municipality_id", "ksic5"]])
        if len(actual) != len(full) or len(full.difference(actual)):
            raise ValueError(
                "KSIC input must cover all mapped categories, "
                "including documented coverage zeros"
            )

        workers = workers.merge(
            bridge[["ksic5", "io", "activity_id"]],
            on="ksic5",
            validate="many_to_one",
        )
        workers = workers.groupby(
            ["municipality_id", "io", "activity_id"], as_index=False
        ).number_of_workers.sum()

    keys = ["municipality_id", "io", "activity_id"]
    if not set(keys).issubset(workers) or workers.duplicated(keys).any():
        raise ValueError("Worker components require unique municipality/io/activity keys")

    pairs = bridge[["io", "activity_id"]].drop_duplicates()
    expected = pd.MultiIndex.from_tuples(
        [
            (m, int(i), int(a))
            for m in MUNICIPALITIES_229
            for i, a in pairs.itertuples(index=False, name=None)
        ],
        names=keys,
    )
    actual = pd.MultiIndex.from_frame(workers[keys])
    if len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError("Worker components do not cover the complete disjoint concordance")

    ## Economic-activity output
    activity = pd.read_csv(data_dir / "municipal_activity.csv")
    activity_keys = ["municipality_id", "activity_id"]
    expected_activity = pd.MultiIndex.from_product(
        [MUNICIPALITIES_229, range(1, 17)], names=activity_keys
    )
    actual_activity = pd.MultiIndex.from_frame(activity[activity_keys])
    if (
        activity.duplicated(activity_keys).any()
        or len(activity) != len(expected_activity)
        or len(expected_activity.difference(actual_activity))
    ):
        raise ValueError("Activity output must cover 229 municipalities and 16 activities")

    activity["output_used"] = pd.to_numeric(activity.output, errors="raise").fillna(0.)
    if not np.isfinite(activity.output_used).all() or (activity.output_used < 0).any():
        raise ValueError("Activity output must be finite and nonnegative after missing-value treatment")

    ## Activity-specific production scores
    totals = workers.groupby(activity_keys).number_of_workers.sum().rename("activity_workers")
    activity = activity.merge(totals, on=activity_keys, validate="one_to_one")
    comp = workers.merge(
        activity[activity_keys + ["output_used", "activity_workers"]],
        on=activity_keys,
        validate="many_to_one",
    )
    comp["output_score"] = np.divide(
        comp.output_used * comp.number_of_workers,
        comp.activity_workers,
        out=np.zeros(len(comp)),
        where=comp.activity_workers.gt(0),
    )
    index = pd.MultiIndex.from_product(
        [MUNICIPALITIES_229, SECTOR_76], names=["municipality_id", "io"]
    )
    score = (
        comp.groupby(["municipality_id", "io"]).output_score.sum()
        .reindex(index, fill_value=0.)
    )

    ## Agriculture without mapped workers
    positive_unmapped = activity.output_used.gt(0) & activity.activity_workers.eq(0)
    for row in activity.loc[positive_unmapped].itertuples():
        if row.activity_id != 1:
            raise ValueError("Positive output without mapped workers requires a reviewed treatment")

        province = MUNICIPALITIES_229[row.municipality_id][0]
        composition = reported_output.loc[
            [(province, 1), (province, 4)]
        ].iloc[:, 0].to_numpy(float)
        if (composition < 0).any() or composition.sum() <= 0:
            raise ValueError("Agricultural output cannot define nonnegative sector proportions")

        for io, share in zip([1, 4], composition / composition.sum()):
            score.loc[(row.municipality_id, io)] = row.output_used * share

    ## IO83 and Sejong Ships
    output_total = activity.groupby("municipality_id").output_used.sum()
    for m in MUNICIPALITIES_229:
        score.loc[(m, 83)] = output_total.loc[m]
    score.loc[(72, 41)] = 1.0

    ## Common production weights
    allocation = score.rename("output_score").reset_index()
    allocation["province_en"] = allocation.municipality_id.map(
        lambda m: MUNICIPALITIES_229[m][0]
    )
    denominators = allocation.groupby(
        ["province_en", "io"]
    ).output_score.transform("sum")
    allocation["output_weight"] = np.divide(
        allocation.output_score,
        denominators,
        out=np.zeros(len(allocation)),
        where=denominators.gt(0),
    )
    allocation["value_added_weight"] = allocation.output_weight

    demand = Z_ref.abs().sum(axis=1) + Z_ref.abs().sum(axis=0)
    required_accounts = demand.gt(0) | reported_output.iloc[:, 0].abs().gt(0)
    sums = allocation.groupby(["province_en", "io"]).output_weight.sum()
    sums = sums.reindex(required_accounts.index)
    if not np.allclose(sums.loc[required_accounts], 1., rtol=0, atol=1e-12):
        raise ValueError("A nonzero provincial account has no municipal production weights")

    return allocation
