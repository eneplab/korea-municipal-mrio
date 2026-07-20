# src/municipal_disaggregation.py

import numpy as np
import pandas as pd

from src.utils import add_municipality_id


def municipal_disaggregation(data):
    eps = 1e-12

    ## Inputs
    Z = data["Z"]
    M = data["M"]
    Y_d = data["Y_d"]
    Y_m = data["Y_m"]
    VA = data["VA"]
    x_j = data["x_j"]
    x_i = data["x_i"]
    x_m = data["x_m"]
    Z_ref = Z.copy()

    region_hierarchy = data["region_hierarchy"]
    municipalities_229 = data["municipalities_229"]
    grdp = data["grdp"]
    population = data["population"]
    expenditure = data["expenditure"]
    inactive = data.get("inactive", set())
    exceptions = data.get("exceptions", set())
    final_demand_proxy = data.get("final_demand_proxy", "component")

    ## Dimensions
    provinces = Z.index.get_level_values("Province").unique().tolist()
    sectors = Z.index.get_level_values("Sector").unique().tolist()
    fd_provinces = Y_d.columns.get_level_values("Province").unique().tolist()
    fd_items = Y_d.columns.get_level_values(1).unique().tolist()

    ## Index
    region_sector_pairs = [
        (p, m, sector)
        for p in provinces
        for m in region_hierarchy[p]
        for sector in sectors
    ]

    region_sector_index = pd.MultiIndex.from_tuples(
        region_sector_pairs, names=["Province", "Municipality", "Sector"]
    )

    index_map = {k: idx for idx, k in enumerate(region_sector_pairs)}
    n = len(region_sector_pairs)

    ## Regional data
    grdp = add_municipality_id(grdp, municipalities_229)
    population = add_municipality_id(population, municipalities_229)
    expenditure = add_municipality_id(expenditure, municipalities_229)

    grdp["GRDP"] = pd.to_numeric(grdp["GRDP"], errors="coerce").fillna(0.0)
    population["Population"] = pd.to_numeric(
        population["Population"], errors="coerce"
    ).fillna(0.0)
    expenditure["general_account_expenditure"] = pd.to_numeric(
        expenditure["general_account_expenditure"], errors="coerce"
    ).fillna(0.0)
    expenditure["special_account_expenditure"] = pd.to_numeric(
        expenditure["special_account_expenditure"], errors="coerce"
    ).fillna(0.0)

    grdp_map = {
        p: grdp.loc[grdp["Province"] == p].set_index("ID")["GRDP"]
        for p in provinces
    }
    population_map = {
        p: population.loc[population["Province"] == p].set_index("ID")["Population"]
        for p in provinces
    }
    general_expenditure_map = {
        p: expenditure.loc[expenditure["Province"] == p]
        .set_index("ID")["general_account_expenditure"]
        for p in provinces
    }
    special_expenditure_map = {
        p: expenditure.loc[expenditure["Province"] == p]
        .set_index("ID")["special_account_expenditure"]
        for p in provinces
    }

    def is_inactive(p, m, sector):
        return (p, m, sector) in inactive and (p, m, sector) not in exceptions

    def normalized_weight(series, municipalities):
        values = series.reindex(municipalities).fillna(0.0).astype(float)
        total = values.sum()
        if total > eps:
            return values.values / total
        return np.zeros(len(municipalities))

    def fd_weight(p, fd):
        municipalities = region_hierarchy[p]

        if final_demand_proxy == "grdp":
            return normalized_weight(grdp_map[p], municipalities)

        if "Private Final Consumption" in fd:
            return normalized_weight(population_map[p], municipalities)

        if "Government Final Consumption" in fd:
            return normalized_weight(general_expenditure_map[p], municipalities)

        if "Government Gross Fixed Capital" in fd:
            return normalized_weight(special_expenditure_map[p], municipalities)

        if "Private Gross Fixed Capital" in fd:
            return normalized_weight(grdp_map[p], municipalities)

        return normalized_weight(grdp_map[p], municipalities)

    ## Weights
    # Production accounts and intermediate transactions
    W_inter = {}
    active_municipalities = {}

    for p in provinces:
        g = grdp_map[p]
        municipalities = region_hierarchy[p]

        for sector in sectors:
            active = [m for m in municipalities if not is_inactive(p, m, sector)]
            active_municipalities[(p, sector)] = active

            w = np.zeros(len(active))
            if active:
                grdp_total = g.reindex(active).sum()
                if grdp_total > eps:
                    w = g.reindex(active).values / grdp_total
            W_inter[(p, sector)] = w

    # Final demand weights are not subject to the production activity condition.
    W_fd = {
        (p, fd): fd_weight(p, fd)
        for p in fd_provinces
        for fd in fd_items
    }

    ## Domestic intermediate transactions
    Z_temp = np.zeros((n, n))

    for R in provinces:
        for S in provinces:
            blk = Z.xs(R, level="Province").xs(S, axis=1, level="Province")

            for idx_i, i in enumerate(sectors):
                active_R = active_municipalities[(R, i)]
                w_R = W_inter[(R, i)]
                if len(active_R) == 0:
                    continue
                idx_R = [index_map[(R, r, i)] for r in active_R]

                for idx_j, j in enumerate(sectors):
                    active_S = active_municipalities[(S, j)]
                    w_S = W_inter[(S, j)]
                    if len(active_S) == 0:
                        continue
                    idx_S = [index_map[(S, s, j)] for s in active_S]

                    Z_temp[np.ix_(idx_R, idx_S)] += (
                        blk.iloc[idx_i, idx_j] * np.outer(w_R, w_S)
                    )

    ## Import intermediate transactions
    M_temp = np.zeros((len(sectors), n))
    blk_imp = M.xs("Imports", level="Province")

    for S in blk_imp.columns.get_level_values("Province").unique():
        for idx_j, j in enumerate(sectors):
            active_S = active_municipalities[(S, j)]
            w_S = W_inter[(S, j)]
            if len(active_S) == 0:
                continue
            idx_S = [index_map[(S, s, j)] for s in active_S]

            for idx_i, i in enumerate(sectors):
                val = blk_imp.loc[i, (S, j)]
                if abs(val) > eps:
                    M_temp[idx_i, idx_S] += val * w_S

    ## Final demand index
    fd_pairs = [
        (p, m, fd)
        for p in fd_provinces
        for m in region_hierarchy[p]
        for fd in fd_items
    ]
    fd_pair_index = pd.MultiIndex.from_tuples(
        fd_pairs, names=["Province", "Municipality", "Final Demand"]
    )
    fd_index_map = {k: idx for idx, k in enumerate(fd_pairs)}

    ## Domestic final demand
    Y_d_temp = np.zeros((n, len(fd_pairs)))

    for R in provinces:
        blk = Y_d.xs(R, level="Province")

        for idx_i, i in enumerate(sectors):
            active_R = active_municipalities[(R, i)]
            w_R = W_inter[(R, i)]
            if len(active_R) == 0:
                continue
            idx_R = [index_map[(R, r, i)] for r in active_R]

            for S in fd_provinces:
                for fd in fd_items:
                    w_S = W_fd[(S, fd)]
                    if w_S.sum() <= eps:
                        continue
                    val = blk.loc[i, (S, fd)]
                    if abs(val) > eps:
                        idx_S = [fd_index_map[(S, s, fd)] for s in region_hierarchy[S]]
                        Y_d_temp[np.ix_(idx_R, idx_S)] += val * np.outer(w_R, w_S)

    ## Import final demand
    Y_m_temp = np.zeros((len(sectors), len(fd_pairs)))
    blk_imp = Y_m.xs("Imports", level="Province")

    for S in fd_provinces:
        for idx_i, i in enumerate(sectors):
            for fd in fd_items:
                w_S = W_fd[(S, fd)]
                if w_S.sum() <= eps:
                    continue
                val = blk_imp.loc[i, (S, fd)]
                if abs(val) > eps:
                    idx_S = [fd_index_map[(S, s, fd)] for s in region_hierarchy[S]]
                    Y_m_temp[idx_i, idx_S] += val * w_S

    ## Value added, total output, total input
    VA_temp = np.zeros((len(VA), n))
    x_j_temp = np.zeros((len(x_j), n))
    x_i_temp = np.zeros(n)

    for p in provinces:
        for sector in sectors:
            active_P = active_municipalities[(p, sector)]
            w = W_inter[(p, sector)]
            if len(active_P) == 0:
                continue
            idx_P = [index_map[(p, m, sector)] for m in active_P]

            VA_temp[:, idx_P] += VA.loc[:, (p, sector)].values[:, None] * w
            x_j_temp[:, idx_P] += x_j.loc[:, (p, sector)].values[:, None] * w
            x_i_temp[idx_P] += float(x_i.loc[(p, sector)].iloc[0]) * w

    Z = pd.DataFrame(Z_temp, index=region_sector_index, columns=region_sector_index)
    M = pd.DataFrame(M_temp, index=M.index, columns=region_sector_index)
    Y_d = pd.DataFrame(Y_d_temp, index=region_sector_index, columns=fd_pair_index)
    Y_m = pd.DataFrame(Y_m_temp, index=Y_m.index, columns=fd_pair_index)
    VA = pd.DataFrame(VA_temp, index=VA.index, columns=region_sector_index)
    x_j = pd.DataFrame(x_j_temp, index=x_j.index, columns=region_sector_index)
    x_i = pd.DataFrame(
        x_i_temp.reshape(-1, 1),
        index=region_sector_index,
        columns=x_i.columns,
    )

    return Z, M, Y_d, Y_m, VA, x_j, x_i, x_m, Z_ref
