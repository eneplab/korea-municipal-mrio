# src/mrio_io.py

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import merge_io, sanitize_for_parquet


class MRIO:

    def __init__(self, blocks):

        self.Z = blocks.get("Z")
        self.A = blocks.get("A")
        self.M = blocks.get("M")
        self.Y = blocks.get("Y")
        self.Y_d = blocks.get("Y_d")
        self.Y_m = blocks.get("Y_m")
        self.VA = blocks.get("VA")
        self.X = blocks.get("X")
        self.x_i = blocks.get("x_i")
        self.x_j = blocks.get("x_j")
        self.x_m = blocks.get("x_m")
        self.manifest = blocks.get("manifest")

    def keys(self):

        return [
            key
            for key in [
                "Z",
                "A",
                "M",
                "Y",
                "Y_d",
                "Y_m",
                "VA",
                "X",
                "x_i",
                "x_j",
                "x_m",
                "manifest",
            ]
            if getattr(self, key) is not None
        ]


def save_frame(obj, path):

    path = Path(path)
    sanitize_for_parquet(obj).to_parquet(path)

    return path


def read_frame(path):

    path = Path(path)

    if path.exists():
        return pd.read_parquet(path)

    parquet_path = path.with_suffix(".parquet")

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)

    raise FileNotFoundError(path)


def make_A(Z, X, eps=1e-12):

    if isinstance(X, pd.DataFrame):
        if len(X.columns) == 1:
            x = X.iloc[:, 0]
        elif len(X.index) == 1:
            x = X.iloc[0, :]
        else:
            raise ValueError("X must be a vector-like DataFrame")
    else:
        x = X

    x = x.reindex(Z.columns).fillna(0.0).astype(float)
    denom = x.replace(0.0, np.nan)

    A = Z.div(denom, axis=1).fillna(0.0)
    A.loc[:, x.abs() <= eps] = 0.0

    return A


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
    info=None,
):

    out_dir = Path(out_dir)
    matrix_dir = out_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    A = make_A(Z, x_i)
    Y = pd.concat([Y_d, Y_m], axis=0)

    MRIO = merge_io(
        Z=Z,
        M=M,
        VA=VA,
        x_j=x_j,
        Y_d=Y_d,
        Y_m=Y_m,
        x_i=x_i,
        x_m=x_m,
    )

    matrices = {
        "Z": Z,
        "A": A,
        "M": M,
        "Y": Y,
        "Y_d": Y_d,
        "Y_m": Y_m,
        "VA": VA,
        "X": x_i,
        "x_i": x_i,
        "x_j": x_j,
        "x_m": x_m,
    }

    matrix_files = {}
    for name, obj in matrices.items():
        saved_path = save_frame(obj, matrix_dir / f"{name}.parquet")
        matrix_files[name] = str(saved_path.relative_to(out_dir))

    mrio_path = save_frame(MRIO, out_dir / "municipal_mrio_korea_2020.parquet")

    idx = Z.index.to_frame(index=False)
    idx.insert(0, "order", range(1, len(idx) + 1))
    idx.to_csv(out_dir / "index.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n_municipalities": int(
            idx[["Province", "Municipality"]].drop_duplicates().shape[0]
        ),
        "n_sectors": int(idx["Sector"].nunique()),
        "matrix_shape": {
            "Z": list(Z.shape),
            "A": list(A.shape),
            "M": list(M.shape),
            "Y": list(Y.shape),
            "VA": list(VA.shape),
            "X": list(x_i.shape),
            "MRIO": list(MRIO.shape),
        },
        "unit": "same as benchmark input-output table",
        "files": {
            "municipal_mrio": str(mrio_path.relative_to(out_dir)),
            "index": "index.csv",
            "matrices": matrix_files,
        },
        "balancing": info or {},
    }

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return matrices, MRIO, manifest


def load_mrio(path, as_object=False):

    path = Path(path)
    matrix_dir = path / "matrices"

    blocks = {}
    for name in ["Z", "A", "M", "Y", "Y_d", "Y_m", "VA", "X", "x_i", "x_j", "x_m"]:
        parquet_path = matrix_dir / f"{name}.parquet"
        if parquet_path.exists():
            blocks[name] = read_frame(parquet_path)

    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            blocks["manifest"] = json.load(f)

    if as_object:
        return MRIO(blocks)

    return blocks
