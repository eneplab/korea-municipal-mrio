import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow
import scipy

from src.utils import merge_io, sanitize_for_parquet


PUBLIC_MATRICES = ("Z", "A", "M", "Y", "Y_d", "Y_m", "VA", "X")


def make_A(Z, X, eps=1e-12):
    x = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else X
    x = x.reindex(Z.columns).fillna(0.0).astype(float)
    A = Z.div(x.replace(0.0, np.nan), axis=1).fillna(0.0)
    A.loc[:, x.abs() <= eps] = 0.0
    return A


class MRIO:
    def __init__(self, table, manifest=None):
        if not isinstance(table.index, pd.MultiIndex):
            raise ValueError("The MRIO table must have a MultiIndex on both axes")
        if not isinstance(table.columns, pd.MultiIndex):
            raise ValueError("The MRIO table must have a MultiIndex on both axes")

        self.table = table
        self.manifest = manifest
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
            self._cache[name] = builder()
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
            rows = self._account_rows[
                pd.Index(labels).astype(str).str.contains("Value Added", case=False)
            ]
            if len(rows) != 1:
                raise ValueError("The MRIO table must contain one value-added row")
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


def build_index(Z, municipality_map, sector_map):
    idx = Z.index.to_frame(index=False)
    municipality_ids = {
        (province, municipality): municipality_id
        for municipality_id, (province, municipality) in municipality_map.items()
    }
    sector_ids = {name: sector_id for sector_id, name in sector_map.items()}

    return pd.DataFrame(
        {
            "account_order": np.arange(1, len(idx) + 1),
            "municipality_id": [
                municipality_ids[(province, municipality)]
                for province, municipality in zip(
                    idx["Province"], idx["Municipality"]
                )
            ],
            "province": idx["Province"].to_numpy(),
            "municipality": idx["Municipality"].to_numpy(),
            "sector_id": idx["Sector"].map(sector_ids).astype(int).to_numpy(),
            "sector": idx["Sector"].to_numpy(),
        }
    )


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
    municipality_map,
    sector_map,
    config,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    mrio_path = save_frame(table, out_dir / "municipal_mrio_korea_2020.parquet")

    index = build_index(Z, municipality_map, sector_map)
    index.to_csv(out_dir / "index.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "dataset": "Municipal MRIO for Korea",
        "year": 2020,
        "unit": "million KRW",
        "coverage": {
            "municipalities": int(index["municipality_id"].nunique()),
            "sectors": int(index["sector_id"].nunique()),
            "municipality_sector_accounts": int(len(index)),
        },
        "configuration": {
            "flq_delta": float(config["delta"]),
            "distance_measure": config["distance_measure"],
            "final_demand_proxy": config["final_demand_proxy"],
            "final_demand_proxies": {
                "private_consumption": "population",
                "government_consumption": "general_account_expenditure",
                "private_gfcf": "municipal_grdp",
                "government_gfcf": "special_account_expenditure",
                "inventories": "municipal_grdp",
                "valuables": "municipal_grdp",
                "exports": "municipal_grdp",
            },
        },
        "shape": {
            "full_mrio": list(table.shape),
            "Z": list(Z.shape),
            "M": list(M.shape),
            "Y_d": list(Y_d.shape),
            "Y_m": list(Y_m.shape),
            "VA": list(VA.shape),
            "X": list(x_i.shape),
        },
        "layout": {
            "rows": ["municipality_sector", "imports", "value_added", "total_input"],
            "columns": ["municipality_sector", "final_demand", "total_output"],
        },
        "balancing": info,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
        },
        "files": {
            "mrio": mrio_path.name,
            "mrio_bytes": mrio_path.stat().st_size,
            "index": "index.csv",
        },
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return table, manifest


def load_mrio(path):
    path = Path(path)
    if path.is_dir():
        path = path / "municipal_mrio_korea_2020.parquet"

    table = read_frame(path)
    manifest_path = path.with_name("manifest.json")
    manifest = None
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    return MRIO(table, manifest)


def load_matrix(path, name):
    if name not in PUBLIC_MATRICES:
        raise KeyError(f"Unknown matrix: {name}")

    path = Path(path)
    if path.is_dir():
        matrix_path = path / "matrices" / f"{name}.parquet"
        if matrix_path.exists():
            return read_frame(matrix_path)

    return getattr(load_mrio(path), name)
