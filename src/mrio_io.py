from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import merge_io, sanitize_for_parquet
from metadata.account import VA_COMPONENTS


PUBLIC_MATRICES = ("Z", "A", "M", "Y", "Y_d", "Y_m", "VA", "X")


def make_A(Z, X, eps=1e-12):
    x = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
    x = x.reindex(Z.columns).astype(float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Technical coefficients require available nonnegative output")
    zero = x.abs() <= eps
    if zero.any() and Z.loc[:, zero].abs().to_numpy().max(initial=0) > 1e-6:
        raise ValueError("A zero-output account has nonzero domestic input transactions")
    A = Z.div(x.replace(0.0, np.nan), axis=1).fillna(0.0)
    A.loc[:, x.abs() <= eps] = 0.0
    return A


class MRIO:
    def __init__(self, table):
        if not isinstance(table.index, pd.MultiIndex):
            raise ValueError("The MRIO table must have a MultiIndex on both axes")
        if not isinstance(table.columns, pd.MultiIndex):
            raise ValueError("The MRIO table must have a MultiIndex on both axes")

        self.table = table
        self._cache = {}

        row_group = table.index.get_level_values(0).astype(str)
        col_group = table.columns.get_level_values(0).astype(str)
        self._domestic_rows = np.flatnonzero(
            (row_group != "Imports") & (row_group != "Account")
        )
        self._import_rows = np.flatnonzero(row_group == "Imports")
        self._account_rows = np.flatnonzero(row_group == "Account")
        self._domestic_cols = np.arange(len(self._domestic_rows))
        self._output_cols = np.flatnonzero(col_group == "Account")
        self._fd_cols = np.arange(
            len(self._domestic_cols),
            self._output_cols.min(initial=table.shape[1]),
        )

        if not np.array_equal(
            self._domestic_rows,
            np.arange(len(self._domestic_rows)),
        ):
            raise ValueError("Municipality-sector accounts are not contiguous")
        if len(self._output_cols) != 1:
            raise ValueError("The MRIO table must contain one total-output column")

    def _get(self, name, builder):
        if name not in self._cache:
            columns = "Final Demand" if name in ("Y", "Y_d", "Y_m") else "Sector"
            self._cache[name] = builder().rename_axis(
                index=["Province", "Municipality", "Sector"],
                columns=["Province", "Municipality", columns],
                copy=False,
            )
        return self._cache[name]

    @property
    def Z(self):
        return self._get(
            "Z",
            lambda: self.table.iloc[self._domestic_rows, self._domestic_cols],
        )

    @property
    def M(self):
        return self._get(
            "M",
            lambda: self.table.iloc[self._import_rows, self._domestic_cols],
        )

    @property
    def Y_d(self):
        return self._get(
            "Y_d",
            lambda: self.table.iloc[self._domestic_rows, self._fd_cols],
        )

    @property
    def Y_m(self):
        return self._get(
            "Y_m",
            lambda: self.table.iloc[self._import_rows, self._fd_cols],
        )

    @property
    def Y(self):
        return self._get("Y", lambda: pd.concat([self.Y_d, self.Y_m], axis=0))

    @property
    def VA(self):
        def build():
            labels = self.table.index[self._account_rows].get_level_values(-1)
            rows = self._account_rows[pd.Index(labels).isin(VA_COMPONENTS)]
            if len(rows) != 4:
                raise ValueError("The MRIO table must contain four value-added rows")
            return self.table.iloc[rows, self._domestic_cols]

        return self._get("VA", build)

    @property
    def X(self):
        return self._get(
            "X",
            lambda: self.table.iloc[self._domestic_rows, self._output_cols],
        )

    @property
    def A(self):
        return self._get("A", lambda: make_A(self.Z, self.X))

    @property
    def input_total(self):
        def build():
            labels = self.table.index[self._account_rows].get_level_values(-1)
            rows = self._account_rows[
                pd.Index(labels).astype(str).str.contains("Total Input", case=False)
            ]
            if len(rows) != 1:
                raise ValueError("The MRIO table must contain one total-input row")
            return self.table.iloc[rows, self._domestic_cols]

        return self._get("input_total", build)

    def keys(self):
        return list(PUBLIC_MATRICES)


def save_frame(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitize_for_parquet(obj).to_parquet(path)
    return path


def read_frame(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def save_mrio(
    out_dir,
    Z,
    M,
    Y_d,
    Y_m,
    VA,
    x_i,
    x_j,
    x_m,
    info,
):

    if not info.get("converged"):
        raise ValueError("Only a converged MRIO table can be saved")

    table = merge_io(
        Z=Z,
        M=M,
        VA=VA,
        x_j=x_j,
        Y_d=Y_d,
        Y_m=Y_m,
        x_i=x_i,
        x_m=x_m,
    )
    save_frame(table, Path(out_dir) / "municipal_mrio_korea_2020.parquet")

    return table


def load_mrio(path):

    path = Path(path)
    if path.is_dir():
        path = path / "municipal_mrio_korea_2020.parquet"

    return MRIO(read_frame(path))


def load_matrix(path, name):
    if name not in PUBLIC_MATRICES:
        raise KeyError(f"Unknown matrix: {name}")

    path = Path(path)
    if path.is_dir():
        matrix_path = path / "matrices" / f"{name}.parquet"
        if matrix_path.exists():
            return read_frame(matrix_path)

    return getattr(load_mrio(path), name)
