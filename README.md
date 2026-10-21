# Municipal MRIO Korea 2020

This repository contains the baseline workflow for constructing a municipal multi-regional input-output (MRIO) dataset for Korea in 2020, covering 229 municipalities and 76 commodity sectors.

## Structure

```text
data/          Construction inputs
metadata/      Region, sector, and account classifications
src/           MRIO construction workflow
release/       Generated dataset
```

## Inputs

- Provincial IRIO tables
- Municipal gross output by economic activity
- The number of workers
- Municipal GRDP
- Population
- Municipal expenditure
- Road-based travel time

## Construction

Tested with Python 3.12.

```bash
pip install numpy==2.1.3 pandas==2.2.3 pyarrow==19.0.0
python -m src
```

The baseline uses gross output by economic activity and the number of workers for production-side weights, category-specific final demand proxies, FLQ with delta = 0.30, road-based travel time, and MR-GRAS with a common tolerance of 1e-6.

The combined dataset is written to `release/municipal_mrio_korea_2020.parquet`. Temporary calculation files are removed after execution. The implementation checks accounting targets, transaction signs, and structural zeros before saving.

## Loading generated output

```python
from src.mrio_io import load_mrio

mrio = load_mrio("release")
Z = mrio.Z
VA = mrio.VA
X = mrio.X
```

The loader also provides `A`, `M`, `Y`, `Y_d`, and `Y_m`. The finalized dataset and separate matrix files are distributed through Zenodo.
