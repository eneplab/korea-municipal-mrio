# src/mrgras_balancing.py

import time

import numpy as np
import pandas as pd

from src.utils import allocate_array, group_sum, stable_sum


class MRGRASFeasibilityError(RuntimeError):
    pass


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
    max_iter=100_000,
    print_iter=10,
    work_dir=None,
    verbose=True,
    row_targets=None,
    column_targets=None,
):

    t0 = time.time()

    ## Inputs and municipal targets
    if max_iter < 1 or print_iter < 1 or mult_tol <= 0 or constraint_tol <= 0:
        raise ValueError("Iteration limits and tolerances must be positive")
    if not Z.index.equals(Z.columns):
        raise ValueError("The transaction matrix must have matching account axes")
    if not x_i.index.equals(Z.index) or not x_j.columns.equals(Z.columns):
        raise ValueError("Output labels do not match the transaction matrix")
    if (
        not M.columns.equals(Z.columns)
        or not VA.columns.equals(Z.columns)
        or not Y_d.index.equals(Z.index)
    ):
        raise ValueError("Import, value-added or final-demand account labels do not match")

    for name, frame in [
        ("M", M), ("VA", VA), ("Y_d", Y_d), ("x_i", x_i), ("x_j", x_j)
    ]:
        for start in range(0, len(frame), 256):
            if not np.isfinite(frame.iloc[start:start + 256].to_numpy(float)).all():
                raise ValueError(name + " contains unavailable or non-finite values")

    row_target_init = x_i.iloc[:, 0] - stable_sum(Y_d.to_numpy(), 1)
    col_target_init = (
        x_j.iloc[0, :]
        - stable_sum(M.to_numpy(), 0)
        - stable_sum(VA.to_numpy(), 0)
    )
    if not np.isfinite(row_target_init).all() or not np.isfinite(col_target_init).all():
        raise ValueError("MR-GRAS targets contain non-finite values")

    row_target = row_target_init if row_targets is None else row_targets.reindex(Z.index)
    col_target = col_target_init if column_targets is None else column_targets.reindex(Z.columns)

    for side, target, calculated in [
        ("row", row_target, row_target_init),
        ("column", col_target, col_target_init),
    ]:
        if not np.isfinite(target).all():
            raise ValueError("Transaction targets must cover every municipal account")
        scale = np.maximum(np.abs(target), 1.0)
        if np.max(np.abs(target - calculated) / scale) >= constraint_tol:
            raise ValueError(f"Accounts do not agree with the {side} transaction targets")

    u, v = row_target.to_numpy(), col_target.to_numpy()

    ## Provincial blocks
    if isinstance(Z_block.index, pd.MultiIndex):
        Z_block_province = group_sum(
            group_sum(Z_block, level=0, axis=0), level=0, axis=1
        )
    else:
        Z_block_province = Z_block.copy()

    province_labels = Z_block_province.index.to_list()
    if province_labels != Z_block_province.columns.to_list():
        raise ValueError("Province-pair block labels do not match")

    block_target = Z_block_province.to_numpy(float)
    if not np.isfinite(block_target).all():
        raise ValueError("Every province-pair block must have a finite target")

    row_province = Z.index.get_level_values("Province")
    col_province = Z.columns.get_level_values("Province")
    if (
        not set(row_province).issubset(province_labels)
        or not set(col_province).issubset(province_labels)
    ):
        raise ValueError("Province block mapping failed")

    row_positions = [np.flatnonzero(row_province == p) for p in province_labels]
    col_positions = [np.flatnonzero(col_province == p) for p in province_labels]
    n_rows, n_cols = Z.shape
    n_provinces = len(province_labels)
    Z_values = Z.to_numpy(dtype=float, copy=False)

    # Read province blocks without copying the full positive and negative matrices.
    def blocks():
        for R, rows in enumerate(row_positions):
            for S, cols in enumerate(col_positions):
                values = Z_values[np.ix_(rows, cols)]
                yield R, S, rows, cols, np.maximum(values, 0), np.maximum(-values, 0)

    ## Sign and zero checks
    positive_rows, negative_rows = np.zeros(n_rows, bool), np.zeros(n_rows, bool)
    positive_cols, negative_cols = np.zeros(n_cols, bool), np.zeros(n_cols, bool)
    positive_blocks = np.zeros_like(block_target, bool)
    negative_blocks = np.zeros_like(block_target, bool)

    for R, S, rows, cols, pos, neg in blocks():
        if not np.isfinite(pos).all() or not np.isfinite(neg).all():
            raise ValueError("Seed matrix contains non-finite values")
        positive_rows[rows] |= (pos > 0).any(axis=1)
        negative_rows[rows] |= (neg > 0).any(axis=1)
        positive_cols[cols] |= (pos > 0).any(axis=0)
        negative_cols[cols] |= (neg > 0).any(axis=0)
        positive_blocks[R, S] = (pos > 0).any()
        negative_blocks[R, S] = (neg > 0).any()

    valid_rows = positive_rows | negative_rows
    valid_cols = positive_cols | negative_cols
    conflicts = []

    for side, target, positive, negative, labels in [
        ("row", u, positive_rows, negative_rows, Z.index),
        ("column", v, positive_cols, negative_cols, Z.columns),
    ]:
        for k in range(len(target)):
            reason = None
            if not positive[k] and not negative[k] and abs(target[k]) >= constraint_tol:
                reason = "zero seed with nonzero target"
            elif target[k] < -constraint_tol and not negative[k]:
                reason = "negative target without negative seed"
            elif target[k] > constraint_tol and not positive[k]:
                reason = "positive target without positive seed"
            if reason:
                conflicts.append((side, labels[k], reason))

    for R in range(n_provinces):
        for S in range(n_provinces):
            target = block_target[R, S]
            if (
                (target < -constraint_tol and not negative_blocks[R, S])
                or (target > constraint_tol and not positive_blocks[R, S])
            ):
                conflicts.append(("block", (province_labels[R], province_labels[S]), "target sign"))

    if conflicts:
        raise MRGRASFeasibilityError(
            "MR-GRAS targets conflict with the seed sign or zero pattern: "
            + str(conflicts[:3])
        )

    ## Multiplier initialization and updates
    rho = np.ones(n_rows)
    sigma = np.ones(n_cols)
    tau = np.ones((n_provinces, n_provinces))

    def update_multiplier(target, p_term, n_term):
        discriminant = np.hypot(target, 2 * np.sqrt(p_term) * np.sqrt(n_term))
        result = np.ones_like(target)
        p = p_term > 0
        nonnegative = target >= 0

        first = p & nonnegative
        result[first] = (target[first] + discriminant[first]) / (2 * p_term[first])

        second = p & ~nonnegative
        result[second] = (2 * n_term[second]) / (discriminant[second] - target[second])

        third = ~p & (n_term > 0) & (target < 0)
        result[third] = -n_term[third] / target[third]

        if not np.isfinite(result).all():
            raise RuntimeError("MR-GRAS multiplier update produced a non-finite value")

        return np.maximum(result, 1e-100)

    ## Constraint residuals
    def residual_summary(row_estimated, col_estimated, block_estimated):
        def residual(estimated, target):
            error = np.abs(estimated - target)
            scale = np.maximum(np.abs(target), 1.0)
            zero = target == 0
            return (
                float(error.max(initial=0)),
                float((error / scale).max(initial=0)),
                float(error[zero].max(initial=0)),
            )

        row_abs, row_scaled, row_zero = residual(row_estimated, u)
        col_abs, col_scaled, col_zero = residual(col_estimated, v)
        block_abs, block_scaled, block_zero = residual(block_estimated, block_target)

        return dict(
            row_residual=row_abs,
            column_residual=col_abs,
            block_residual=block_abs,
            max_absolute_residual=max(row_abs, col_abs, block_abs),
            max_scaled_residual=max(row_scaled, col_scaled, block_scaled),
            zero_target_residual=max(row_zero, col_zero, block_zero),
        )

    def constraint_residuals():
        row_estimated = np.zeros(n_rows)
        col_estimated = np.zeros(n_cols)
        block_estimated = np.zeros_like(block_target)

        for R, S, rows, cols, pos, neg in blocks():
            scale = rho[rows, None] * sigma[None, cols] * tau[R, S]
            values = pos * scale - neg / scale
            if not np.isfinite(values).all():
                raise RuntimeError("MR-GRAS produced non-finite transaction values")
            row_estimated[rows] += values.sum(axis=1)
            col_estimated[cols] += values.sum(axis=0)
            block_estimated[R, S] = values.sum()

        return residual_summary(row_estimated, col_estimated, block_estimated)

    ## Iterative balancing
    converged = False

    for n_iter in range(1, max_iter + 1):
        rho_prev, sigma_prev, tau_prev = rho.copy(), sigma.copy(), tau.copy()

        # Column update
        pos_col_sum, neg_col_sum = np.zeros(n_cols), np.zeros(n_cols)
        for R, S, rows, cols, pos, neg in blocks():
            scale = rho[rows, None] * tau[R, S]
            pos_col_sum[cols] += (pos * scale).sum(axis=0)
            neg_col_sum[cols] += (neg / scale).sum(axis=0)
        sigma[valid_cols] = update_multiplier(
            v[valid_cols], pos_col_sum[valid_cols], neg_col_sum[valid_cols]
        )

        # Row update
        pos_row_sum, neg_row_sum = np.zeros(n_rows), np.zeros(n_rows)
        for R, S, rows, cols, pos, neg in blocks():
            scale = sigma[None, cols] * tau[R, S]
            pos_row_sum[rows] += (pos * scale).sum(axis=1)
            neg_row_sum[rows] += (neg / scale).sum(axis=1)
        rho[valid_rows] = update_multiplier(
            u[valid_rows], pos_row_sum[valid_rows], neg_row_sum[valid_rows]
        )

        # Provincial block update
        block_pos = np.zeros_like(block_target)
        block_neg = np.zeros_like(block_target)
        for R, S, rows, cols, pos, neg in blocks():
            scale = rho[rows, None] * sigma[None, cols]
            block_pos[R, S] = (pos * scale).sum()
            block_neg[R, S] = (neg / scale).sum()
        support = positive_blocks | negative_blocks
        tau[support] = update_multiplier(
            block_target[support], block_pos[support], block_neg[support]
        )

        ## Convergence checks
        mult_diff = max(
            np.abs(rho - rho_prev).max(),
            np.abs(sigma - sigma_prev).max(),
            np.abs(tau - tau_prev).max(),
        )
        if not np.isfinite(mult_diff):
            raise RuntimeError("MR-GRAS multiplier change is non-finite")

        if mult_diff < mult_tol or n_iter == 1 or n_iter % print_iter == 0 or n_iter == max_iter:
            residuals = constraint_residuals()
            if verbose:
                print(
                    f"[Iter {n_iter:6d}] Diff: {mult_diff:.1e} | "
                    f"Scaled resid: {residuals['max_scaled_residual']:.1e}",
                    flush=True,
                )
            if (
                mult_diff < mult_tol
                and residuals["max_scaled_residual"] < constraint_tol
                and residuals["zero_target_residual"] < constraint_tol
            ):
                converged = True
                break

    if not converged:
        raise RuntimeError(
            f"MR-GRAS did not converge after {max_iter} iterations: {residuals}"
        )

    ## Final matrix and all-target checks
    Z_balanced = allocate_array(Z.shape, work_dir, "balanced_transactions")
    row_estimated = np.zeros(n_rows)
    col_estimated = np.zeros(n_cols)
    block_estimated = np.zeros_like(block_target)

    for R, S, rows, cols, pos, neg in blocks():
        scale = rho[rows, None] * sigma[None, cols] * tau[R, S]
        values = pos * scale - neg / scale
        if (
            not np.isfinite(values).all()
            or not np.array_equal(np.sign(values), np.sign(pos - neg))
        ):
            raise RuntimeError("Final MR-GRAS values changed a sign or zero position")
        Z_balanced[np.ix_(rows, cols)] = values
        row_estimated[rows] += values.sum(axis=1)
        col_estimated[cols] += values.sum(axis=0)
        block_estimated[R, S] = values.sum()

    final_residuals = residual_summary(row_estimated, col_estimated, block_estimated)
    if (
        final_residuals["max_scaled_residual"] >= constraint_tol
        or final_residuals["zero_target_residual"] >= constraint_tol
    ):
        raise RuntimeError("Final MR-GRAS matrix failed the all-target constraint check")

    info = dict(
        iterations=n_iter,
        converged=True,
        multiplier_tolerance=mult_tol,
        constraint_tolerance=constraint_tol,
        final_residual=final_residuals["max_absolute_residual"],
        max_multiplier_change=float(mult_diff),
        time_sec=time.time() - t0,
        **final_residuals,
    )

    return (
        pd.DataFrame(Z_balanced, index=Z.index, columns=Z.columns, copy=False),
        M, VA, Y_d, Y_m, x_i, x_j, x_m, info,
    )
