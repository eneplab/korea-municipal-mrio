# src/intramunicipal_allocation.py

import numpy as np
import pandas as pd


def calculate_flq_intra(Z_seed, x_i_seed, delta=0.30, eps=1e-12):

    ## Positive output vector
    x = x_i_seed.iloc[:, 0].rename("x").astype(float)
    x.index.names = ["Province", "Municipality", "Sector"]
    x_pos = x.clip(lower=0)

    ## Regional output aggregates
    x_ip = (
        x_pos
        .groupby(level=["Province", "Sector"], sort=False)
        .sum()
        .rename("x_ip")
    )

    x_r = (
        x_pos
        .groupby(level=["Province", "Municipality"], sort=False)
        .sum()
        .rename("x_r")
    )

    x_p = (
        x_pos
        .groupby(level="Province", sort=False)
        .sum()
        .rename("x_p")
    )

    ## SLQ and lambda
    flq_base = (
        x_pos.rename("x_ir")
        .reset_index()
        .merge(x_r.reset_index(), on=["Province", "Municipality"], how="left")
        .merge(x_ip.reset_index(), on=["Province", "Sector"], how="left")
        .merge(x_p.reset_index(), on=["Province"], how="left")
    )

    flq_base["share_ir"] = np.where(
        flq_base["x_r"] > eps,
        flq_base["x_ir"] / flq_base["x_r"],
        0.0,
    )

    flq_base["share_ip"] = np.where(
        flq_base["x_p"] > eps,
        flq_base["x_ip"] / flq_base["x_p"],
        0.0,
    )

    flq_base["SLQ"] = np.where(
        flq_base["share_ip"] > eps,
        flq_base["share_ir"] / flq_base["share_ip"],
        0.0,
    )

    flq_base["lambda_r"] = np.where(
        flq_base["x_p"] > eps,
        np.log2(1 + flq_base["x_r"] / flq_base["x_p"]) ** delta,
        0.0,
    )

    slq = flq_base.set_index(["Province", "Municipality", "Sector"])["SLQ"]

    lambda_r = flq_base.set_index(["Province", "Municipality"])["lambda_r"]
    lambda_r = lambda_r[~lambda_r.index.duplicated()]

    ## Same-province positive benchmark coefficient
    provinces = Z_seed.index.get_level_values("Province").unique().tolist()

    a_pp_list = []

    for R in provinces:

        Z_pp = (
            Z_seed
            .xs(R, level="Province", axis=0)
            .xs(R, level="Province", axis=1)
            .clip(lower=0)
        )

        Z_ij_pp = (
            Z_pp
            .groupby(level="Sector", axis=0, sort=False)
            .sum()
            .groupby(level="Sector", axis=1, sort=False)
            .sum()
        )

        x_j_R = x_ip.xs(R, level="Province").reindex(Z_ij_pp.columns).fillna(0)

        a_ij_pp = Z_ij_pp.div(x_j_R.replace(0, np.nan), axis=1).fillna(0)
        a_ij_pp["Province"] = R

        a_pp_list.append(
            a_ij_pp
            .reset_index()
            .rename(columns={"Sector": "input_sector"})
        )

    a_pp_long = (
        pd.concat(a_pp_list, ignore_index=True)
        .melt(
            id_vars=["Province", "input_sector"],
            var_name="purchasing_sector",
            value_name="a_pp",
        )
    )

    ## FLQ coefficient
    slq_df = (
        slq
        .rename("SLQ")
        .reset_index()
    )

    slq_i = slq_df.rename(
        columns={
            "Sector": "input_sector",
            "SLQ": "SLQ_i",
        }
    )

    slq_j = slq_df.rename(
        columns={
            "Sector": "purchasing_sector",
            "SLQ": "SLQ_j",
        }
    )

    coords = (
        x_pos
        .reset_index()[["Province", "Municipality"]]
        .drop_duplicates()
    )

    flq_ijr = (
        a_pp_long[["Province", "input_sector", "purchasing_sector", "a_pp"]]
        .merge(coords, on="Province", how="left")
        .merge(slq_i, on=["Province", "Municipality", "input_sector"], how="left")
        .merge(slq_j, on=["Province", "Municipality", "purchasing_sector"], how="left")
        .merge(
            lambda_r.rename("lambda_r").reset_index(),
            on=["Province", "Municipality"],
            how="left",
        )
    )

    flq_ijr["SLQ_i"] = flq_ijr["SLQ_i"].fillna(0)
    flq_ijr["SLQ_j"] = flq_ijr["SLQ_j"].fillna(0)
    flq_ijr["lambda_r"] = flq_ijr["lambda_r"].fillna(0)

    flq_ijr["CILQ"] = np.where(
        flq_ijr["SLQ_j"] > eps,
        flq_ijr["SLQ_i"] / flq_ijr["SLQ_j"],
        0.0,
    )

    flq_ijr["FLQ"] = flq_ijr["CILQ"] * flq_ijr["lambda_r"]
    flq_ijr["phi"] = np.minimum(flq_ijr["FLQ"], 1.0)

    ## Intramunicipal positive estimate
    x_jr = (
        x_pos
        .rename("x_jr")
        .reset_index()
        .rename(columns={"Sector": "purchasing_sector"})
    )

    flq_ijr = flq_ijr.merge(
        x_jr,
        on=["Province", "Municipality", "purchasing_sector"],
        how="left",
    )

    flq_ijr["x_jr"] = flq_ijr["x_jr"].fillna(0)
    flq_ijr["Z_rr_FLQ_pos"] = flq_ijr["a_pp"] * flq_ijr["phi"] * flq_ijr["x_jr"]

    flq_sum = (
        flq_ijr
        .groupby(["Province", "input_sector", "purchasing_sector"], sort=False)["Z_rr_FLQ_pos"]
        .sum()
    )

    summary = {
        "delta": delta,
        "total_FLQ_intra_pos": float(flq_ijr["Z_rr_FLQ_pos"].sum()),
    }

    return flq_ijr, flq_sum, summary
