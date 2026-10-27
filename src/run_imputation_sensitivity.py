"""Compare the corrected primary experiment under the two NASA imputation strategies."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
OUTPUT_PATH = RESULTS_DIR / "imputation_sensitivity.csv"
BASELINE_PATH = RESULTS_DIR / "comparison_metrics_median_indicators.csv"


def run_strategy(strategy):
    with tempfile.TemporaryDirectory(prefix="imputation_sensitivity_") as temp_dir:
        temp_path = Path(temp_dir)
        metrics_path = temp_path / "comparison_metrics.csv"
        environment = os.environ.copy()
        environment["NASA_IMPUTATION_STRATEGY"] = strategy
        environment["SVM_EVALUATION_SIZE"] = "5000"
        environment["EXPERIMENT_MODELS"] = "Random Forest,SVM (RBF Kernel)"
        environment["RUN_SOTWIS_ONLY"] = "false"
        environment["COMPARISON_METRICS_OUTPUT_PATH"] = str(metrics_path)
        environment["COMPARISON_REPORT_OUTPUT_PATH"] = str(temp_path / "report.md")
        environment["COMMON_SUPPORT_OUTPUT_PATH"] = str(temp_path / "support.csv")
        environment["PERMUTATION_IMPORTANCE_OUTPUT_PATH"] = str(temp_path / "importance.csv")
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "run_comparison_experiments.py")],
            check=True,
            env=environment,
        )
        metrics = pd.read_csv(metrics_path)
        metrics.insert(0, "data_start_year", int(environment.get("DATA_START_YEAR", "1990")))
        metrics.insert(1, "data_end_year", int(environment.get("DATA_END_YEAR", "2024")))
    metrics["imputation_strategy"] = strategy
    return metrics


def main():
    baseline = run_strategy("median_indicators")
    baseline.to_csv(BASELINE_PATH, index=False)
    zero_metrics = run_strategy("zero")
    combined = pd.concat([baseline, zero_metrics], ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved imputation sensitivity results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
