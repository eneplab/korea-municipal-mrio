# src/mrgras_balancing.py

import numpy as np
import pandas as pd
import time
import warnings

warnings.filterwarnings("ignore")

def multi_regional_gras(
    Z,
    Z_block,
    M,
    VA,
    Y_d,
    Y_m,
    x_i,             
    x_j,
    x_m,
    mult_tol=1e-6,
    constraint_tol=1e-6,
    eps=1e-12,
    max_iter=100_000,
    print_iter=10,
):
    t0 = time.time()

    ## Row and Column targets
    row_target_init = (
        x_i.reindex(Z.index).fillna(0).iloc[:, 0]
        - Y_d.reindex(Z.index).fillna(0).sum(axis=1)
    )

    col_target_init = (
        x_j.reindex(Z.columns, axis=1).fillna(0).iloc[0, :]
        - M.reindex(Z.columns, axis=1).fillna(0).sum(axis=0)
        - VA.reindex(Z.columns, axis=1).fillna(0).sum(axis=0)
    )

    ## Threshold small targets to zero
    x_i_safe = x_i.reindex(Z.index).fillna(0).iloc[:, 0].abs()
    x_i_safe[x_i_safe < 1e-6] = 1.0

    row_negligible_mask = (np.abs(row_target_init) < 1e-3) & \
                   ((np.abs(row_target_init) / x_i_safe) < 1e-7)

    row_target = row_target_init.copy()
    row_target[row_negligible_mask] = 0.0

    x_j_safe = x_j.reindex(Z.columns, axis=1).fillna(0).iloc[0, :].abs()
    x_j_safe[x_j_safe < 1e-6] = 1.0

    col_negligible_mask = (np.abs(col_target_init) < 1e-3) & \
                   ((np.abs(col_target_init) / x_j_safe) < 1e-7)

    col_target = col_target_init.copy()
    col_target[col_negligible_mask] = 0.0

    u = row_target.values
    v = col_target.values

    ## Seed matrix decomposition
    Z = Z.copy()
    Z_values = Z.to_numpy(dtype=float)
    n_rows, n_cols = Z_values.shape

    mask_nonzero = (~np.isclose(Z_values, 0.0, atol=eps))
    Z_pos = np.where(Z_values > 0, Z_values, 0.0)
    Z_neg = np.where(Z_values < 0, -Z_values, 0.0)

    valid_rows = np.any(mask_nonzero, axis=1)
    valid_cols = np.any(mask_nonzero, axis=0)

    ## Block targets
    if isinstance(Z_block.index, pd.MultiIndex):
        Z_block_province = (
            Z_block
            .groupby(level=0, axis=0)
            .sum()
            .groupby(level=0, axis=1)
            .sum()
        )
    else:
        Z_block_province = Z_block.copy()

    ## Block mapping between row/column targets and province index
    row_province = Z.index.get_level_values("Province")
    col_province = Z.columns.get_level_values("Province")

    province_labels = Z_block_province.index.to_list()
    province_idx = {lbl: k for k, lbl in enumerate(province_labels)}
    n_provinces = len(province_labels)

    block_R = np.array([province_idx.get(lbl, -1) for lbl in row_province])
    block_S = np.array([province_idx.get(lbl, -1) for lbl in col_province])

    if (block_R == -1).any() or (block_S == -1).any():
        raise RuntimeError("BLOCK MAPPING FAILED")

    block_target = Z_block_province.to_numpy(dtype=float)
    block_constraint_mask = ~np.isnan(block_target)

    ## Multiplier initialization
    rho = np.ones(n_rows)
    sigma = np.ones(n_cols)
    tau = np.ones((n_provinces, n_provinces))

    ## Multiplier update
    def update_multiplier(target, p_term, n_term):
        discriminant = np.sqrt(target**2 + 4 * p_term * n_term)
        mask_p = (p_term > eps)
        mask_pos_target = (target >= 0)

        new_mult = np.ones_like(p_term)

        idx_std = mask_p & mask_pos_target
        if np.any(idx_std):
            new_mult[idx_std] = (target[idx_std] + discriminant[idx_std]) / (2 * p_term[idx_std])

        idx_rat = mask_p & (~mask_pos_target)
        if np.any(idx_rat):
            scaling_factor = discriminant[idx_rat] - target[idx_rat]
            new_mult[idx_rat] = (2 * n_term[idx_rat]) / scaling_factor

        idx_neg = (~mask_p) & (np.abs(target) > eps)
        if np.any(idx_neg):
            new_mult[idx_neg] = np.abs(n_term[idx_neg] / target[idx_neg])

        return np.nan_to_num(new_mult, nan=1.0)

    ## Iterative balancing
    for n_iter in range(1, max_iter + 1):
        rho_prev, sigma_prev, tau_prev = rho.copy(), sigma.copy(), tau.copy()

        tau_block = tau[block_R[:, None], block_S[None, :]]

        # Column update (sigma)
        rho_tau = rho[:, None] * tau_block
        pos_col_sum = (Z_pos * rho_tau).sum(axis=0)
        neg_col_sum = (Z_neg / (rho_tau + 1e-100)).sum(axis=0)
        sigma[valid_cols] = update_multiplier(v[valid_cols], pos_col_sum[valid_cols], neg_col_sum[valid_cols])

        col_estimated = sigma * pos_col_sum - (neg_col_sum / (sigma + 1e-100))
        col_residual = np.max(np.abs(col_estimated[valid_cols] - v[valid_cols]))

        # Row update (rho)
        sigma_tau = sigma[None, :] * tau_block
        pos_row_sum = (Z_pos * sigma_tau).sum(axis=1)
        neg_row_sum = (Z_neg / (sigma_tau + 1e-100)).sum(axis=1)
        rho[valid_rows] = update_multiplier(u[valid_rows], pos_row_sum[valid_rows], neg_row_sum[valid_rows])

        row_estimated = rho * pos_row_sum - (neg_row_sum / (rho + 1e-100))
        row_residual = np.max(np.abs(row_estimated[valid_rows] - u[valid_rows]))

        # Block update (tau)
        rho_sigma = rho[:, None] * sigma[None, :]
        pos_scaled = Z_pos * rho_sigma
        neg_scaled = Z_neg / (rho_sigma + 1e-100)

        block_pos = np.zeros((n_provinces, n_provinces))
        block_neg = np.zeros((n_provinces, n_provinces))

        for i in range(n_rows):
            province_row = block_R[i]
            block_pos[province_row, :] += np.bincount(block_S, weights=pos_scaled[i, :], minlength=n_provinces)
            block_neg[province_row, :] += np.bincount(block_S, weights=neg_scaled[i, :], minlength=n_provinces)

        block_update_mask = block_constraint_mask & ((block_pos > eps) | (block_neg > eps))
        if np.any(block_update_mask):
            tau[block_update_mask] = update_multiplier(
                block_target[block_update_mask],
                block_pos[block_update_mask],
                block_neg[block_update_mask]
            )

        block_estimated = tau * block_pos - (block_neg / (tau + 1e-100))
        block_residual = np.max(np.abs(block_estimated[block_constraint_mask] - block_target[block_constraint_mask]))

        mult_diff = max(
            np.abs(rho - rho_prev).max(),
            np.abs(sigma - sigma_prev).max(),
            np.abs(tau - tau_prev).max()
        )
        max_residual = max(row_residual, col_residual, block_residual)

        if n_iter % print_iter == 0 or n_iter == 1:
            print(f"[Iter {n_iter:6d}] Diff: {mult_diff:.1e} | "
                  f"Resid: {max_residual:.1e} "
                  f"(R:{row_residual:.1e}, C:{col_residual:.1e}, B:{block_residual:.1e})")

        if mult_diff < mult_tol and max_residual < constraint_tol:
            print(f"[DONE] Converged at iteration {n_iter}")
            break

    ## Final balanced matrix
    tau_block = tau[block_R[:, None], block_S[None, :]]
    scaling_factor = rho[:, None] * sigma[None, :] * tau_block

    Z_pos_final = Z_pos * scaling_factor
    Z_neg_final = np.zeros_like(Z_neg)
    np.divide(Z_neg, scaling_factor, out=Z_neg_final, where=(scaling_factor > 1e-100))

    Z_balanced = Z_pos_final - Z_neg_final
    Z = pd.DataFrame(Z_balanced, index=Z.index, columns=Z.columns)

    score_r = np.mean(np.abs(np.log(rho + 1e-20)))
    score_s = np.mean(np.abs(np.log(sigma + 1e-20)))
    score_t = np.mean(np.abs(np.log(tau + 1e-20)))
    
    total_distortion = score_r + score_s + score_t

    info = {
        "iterations": n_iter,
        "final_residual": float(max_residual),
        "score_total": float(total_distortion), 
        "score_r": float(score_r),
        "score_s": float(score_s),
        "score_t": float(score_t),
        
        "time_sec": time.time() - t0,
    }

    return Z, M, VA, Y_d, Y_m, x_i, x_j, x_m, info
