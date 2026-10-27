# Aggregation of Soil and Environmental Data to Predict Drought in Eastern and Central Europe

This research project investigates whether combining soil characteristics with environmental observations improves drought prediction in Eastern and Central Europe. It compares soil-only, environmental-only, and combined feature sets using Random Forest, support vector machine (SVM), and long short-term memory (LSTM) models.

## Approach

- **Soil:** SOTWIS/SOVEUR soil attributes, including synthetically enhanced attributes.
- **Environment:** NASA POWER observations, including soil wetness variables.
- **Target:** the Standardized Precipitation Evapotranspiration Index (SPEI). The main comparison defines drought as SPEI-01 below -1 and predicts the next month's class.
- **Evaluation:** chronological train/test splits and drought-class precision, recall, F1, and precision-recall AUC. The project also examines spatial and temporal extrapolation, missingness, feature importance, and other sensitivities.

The NASA-only and combined experiments use common support. The SOTWIS-only experiment is a separate static susceptibility analysis and should not be interpreted as a directly matched comparison. See the [full experimental report](results/full_experimental_report.md) for the detailed results and [extrapolation validation summary](results/extrapolation_validation_summary_1990_2024.md) for spatial and temporal validation.

## Repository structure

```text
src/             Data preparation, model comparison, validation, and analysis scripts
results/         Experimental reports and result tables
data/            Existing data and intermediate inputs, where available
requirements.txt Python dependencies
```

CSV and NetCDF files tracked by the repository use Git Large File Storage (LFS). Install [Git LFS](https://git-lfs.com/), then run `git lfs install` before cloning and `git lfs pull` after cloning if the files have not been downloaded.

## Setup

From the repository root, create and activate a Python virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or `source .venv/bin/activate` on macOS/Linux, then run:

```bash
python -m pip install -r requirements.txt
```

TensorFlow installation and model training can require substantial resources. Dependency versions are not pinned, so results and installation behavior may vary between environments.

## Run the comparison

The main comparison defaults to 1990-2024 and expects these inputs:

- `data/intermediary_steps/synthetically_enhanced_sotwis.csv`
- `data/intermediary_steps/complete_nasa_power_data_1990_to_2024.csv`
- `data/complete_integrated_dataset_1990_to_2024.csv`
- `data/intermediary_steps/complete_spei_benchmark.csv`

**The extended 1990-2024 NASA and integrated datasets are not included in this repository.** Obtain or generate the required data before running the experiment. Other local datasets may also be excluded, and the availability of source data is subject to the original providers' terms. The retrieval and preparation scripts in `src/` include NASA POWER and SPEI downloads, SOTWIS preparation, and dataset merging.

With the required files in place, run from the repository root:

```bash
python src/run_comparison_experiments.py
```

The script writes results to `results/`, including `full_experimental_report.md`, `comparison_metrics.csv`, `common_support_summary.csv`, and, when computed, `rf_permutation_importance.csv`. **A run can overwrite existing results.** Output paths can be changed through `COMPARISON_REPORT_OUTPUT_PATH`, `COMPARISON_METRICS_OUTPUT_PATH`, `COMMON_SUPPORT_OUTPUT_PATH`, and `PERMUTATION_IMPORTANCE_OUTPUT_PATH` environment variables.

Configuration is provided through environment variables. Examples include `DATA_START_YEAR`, `DATA_END_YEAR`, `NASA_DATA_PATH`, `FUSED_DATA_PATH`, `SPEI_SCALE`, `SPEI_THRESHOLD`, `EXPERIMENT_MODELS`, and `EXPERIMENT_SEED`. Changing the period or SPEI scale requires the corresponding input files. See the configuration section of [the comparison script](src/run_comparison_experiments.py) for defaults and additional options.

## Results and limitations

The checked-in [full report](results/full_experimental_report.md) contains precision, recall, F1, confusion matrices, and comparisons across feature sets. For example, in that report the common-support SVM drought-class F1 is 0.4107 for NASA-only features and 0.4101 for combined features. These values belong to that specific experiment and should not be generalized to all regions, periods, or validation schemes.

The [results directory](results/) also contains extrapolation, uncertainty, missingness, and sensitivity analyses. Drought classes are imbalanced, and validation performance varies by model and evaluation scheme; consult the reports rather than relying on accuracy or a single summary metric.