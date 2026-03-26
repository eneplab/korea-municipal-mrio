# Korea Municipal MRIO Construction

This repository provides the source code and metadata required to construct a high-resolution municipal multi-regional input-output (MRIO) dataset for Korea for the year 2020.

The workflow begins with the benchmark interregional input-output (IRIO) table and proceeds through a sequence of structured processing steps, including sectoral aggregation, regional disaggregation, gravity-based interregional calibration, and multi-regional balancing. The final output is a municipal-level MRIO dataset stored in parquet format.

This repository is intended to support a transparent and reproducible construction workflow for the final dataset release.

## Repository structure

The repository is organized as follows:

- `data/`  
  Input data required for the construction workflow.

- `metadata/`  
  Metadata used to define the accounting structure of the model, including sector classifications, regional hierarchies, and final demand items.

- `src/`  
  Source code implementing the MRIO construction workflow.

- `release/`  
  Output directory where the final released dataset is saved.

## Input data

Before running the workflow, the required input files must be placed in the `data/` directory.

The current pipeline requires the following files:

- `benchmark_io.csv`  
  Benchmark interregional input-output table used as the starting point of the construction.

- `regional_grdp.csv`  
  Regional GRDP data used in the regional disaggregation step.

- `zero_employee.csv`  
  Zero-employment constraint data used to identify structurally absent sector-region combinations.

- `distance_17.csv`  
  Distance matrix for the 17 provinces, used for gravity parameter estimation.

- `distance_229.csv`  
  Distance matrix for the 229 municipalities, used for interregional gravity calibration.

## Metadata

The `metadata/` directory defines the structural classifications used in the model.

- `sector.py`  
  Sector definitions and concordance information, including the mapping from 83 sectors to 76 sectors.

- `region.py`  
  Regional hierarchy and municipality definitions used for the municipal disaggregation workflow.

- `account.py`  
  Definitions of final demand items and related accounting metadata.

## Source files

The construction workflow is primarily implemented through the following modules in `src/`:
- `sectoral_aggregation.py`
- `regional_disaggregation.py`
- `gravity_parameters.py`
- `interregional_calibration.py`
- `multi_regional_balancing.py`
- `utils.py`

These modules are run through `src/main`.py.

## Construction workflow

The full construction workflow is implemented in `src/main.py`.

The main steps are:

1. **Load benchmark IRIO table**  
   Read the benchmark interregional input-output table and extract the relevant transaction, import, value added, final demand, and output blocks.

2. **Sectoral aggregation**  
   Aggregate the benchmark table from 83 sectors to 76 sectors.

3. **Regional disaggregation**  
   Disaggregate the benchmark regional structure to the municipal level using regional GRDP data and zero-employment constraints.

4. **Gravity parameter estimation**  
   Estimate gravity parameters from benchmark interregional transaction patterns.

5. **Interregional calibration**  
   Calibrate municipal interregional transactions using the estimated gravity parameters and distance information.

6. **Multi-regional balancing**  
   Apply multi-regional GRAS balancing to restore accounting consistency across the municipal MRIO system.

7. **Label restoration and final merge**  
   Restore region-sector labels and merge all components into the final MRIO table.

## Run the code

To execute the full workflow, run the following command from the project root directory:

```bash
python -m src.main
```

This command runs the entire construction pipeline and writes the final output dataset to the `release/` directory.

## Output

If the workflow completes successfully, the final dataset is saved as:
- `release/MRIO.parquet`

The final dataset contains the core components of the municipal MRIO system, including:
- domestic intermediate transactions
- imports
- value added
- final demand
- total input and total output accounts

## File format

The final output is stored in **Parquet** format.

Because the dataset is large, it is recommended to handle the output in a computing environment such as **Python**, **R**, or **MATLAB**, rather than in standard spreadsheet software.

