# Municipal MRIO Korea 2020

This repository contains the Python workflow used to construct a municipality-level multi-regional input-output (MRIO) dataset for Korea in 2020.

The workflow reconstructs the dataset from the input files in `data/` and metadata in `metadata/`. The finalized data release is distributed separately through Zenodo.

## Structure

```text
data/          Input data used in the construction workflow
metadata/      Region, sector, and account metadata
src/           Baseline MRIO construction code
diagnostics/   Diagnostic checks for the baseline result
sensitivity/   FLQ, distance, and final-demand proxy sensitivity checks
release/       Generated outputs, ignored by Git
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

Run the baseline construction first:

```bash
python -m src
```

Then run diagnostics:

```bash
python -m diagnostics
```

Run sensitivity checks:

```bash
python -m sensitivity
```

Sensitivity checks can also be run by group:

```bash
python -m sensitivity --cases flq
python -m sensitivity --cases distance
python -m sensitivity --cases final_demand
```

## Outputs

All generated files are written to `release/`.

```text
release/baseline/
release/diagnostics/
release/sensitivity/
```

The merged MRIO table is saved as:

```text
release/baseline/municipal_mrio_korea_2020.parquet
```

Separate matrices are saved under:

```text
release/baseline/matrices/
```

The `release/` directory is ignored by Git because it contains generated and large files.

## Zenodo Data Release

Users who only need the released dataset should download the Zenodo archive and use the standalone loader included in that archive. This GitHub repository is intended for reproducing the construction workflow and diagnostic/sensitivity checks.
