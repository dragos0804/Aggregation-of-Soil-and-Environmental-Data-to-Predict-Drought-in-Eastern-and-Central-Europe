"""Compare full NASA summaries with a reduced means-only feature set."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"
OUTPUT_PATH = RESULTS_DIR / "vif_feature_reduction_sensitivity.csv"


def run_feature_set(feature_set):
    with tempfile.TemporaryDirectory(prefix="vif_sensitivity_") as temp_dir:
        temp_path = Path(temp_dir)
        metrics_path = temp_path / "comparison_metrics.csv"
        environment = os.environ.copy()
        environment["NASA_FEATURE_SET"] = feature_set
        environment["NASA_IMPUTATION_STRATEGY"] = "median_indicators"
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
        return metrics


def main():
    full_metrics = run_feature_set("full")
    reduced_metrics = run_feature_set("means_only")
    pd.concat([full_metrics, reduced_metrics], ignore_index=True).to_csv(
        OUTPUT_PATH, index=False
    )
    print(f"Saved VIF feature-reduction sensitivity results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()