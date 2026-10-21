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

Public inputs are included in `data/`. The 83-sector provincial IRIO table is aggregated to 76 sectors during construction. `metadata/IO_KSIC_activity.csv` links IO sectors, KSIC Revision 10 categories, and economic activities.

The number of workers is not distributed because of confidentiality restrictions. Supply `data/municipal_workers.csv` with either of the following column sets:

```text
municipality_id,ksic5,number_of_workers
municipality_id,io,activity_id,number_of_workers
```

The file must cover all 229 municipalities and every mapped KSIC category or distinct IO-activity pair. Counts must be nonnegative and must not overlap across activities. Include confirmed zeros explicitly; do not replace suppressed or unavailable counts with zero. Municipality IDs follow `metadata/region.py`, and sector and activity IDs follow the concordance. Reproducing the published estimates requires the same prepared counts used in the study.

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
