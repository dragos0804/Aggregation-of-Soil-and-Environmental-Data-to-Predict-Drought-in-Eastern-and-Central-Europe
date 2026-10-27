"""Run the configured comparison pipeline across drought-label definitions."""

import os
import subprocess
import sys
import tempfile

import pandas as pd


SCRIPT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
OUTPUT_PATH = os.getenv(
    "SPEI_SENSITIVITY_OUTPUT_PATH",
    os.path.join(RESULTS_DIR, "spei_threshold_sensitivity.csv"),
)


def main():
    thresholds = os.getenv("SPEI_SENSITIVITY_THRESHOLDS", "-0.5,-1.0,-1.5").split(",")
    scales = os.getenv("SPEI_SENSITIVITY_SCALES", "01").split(",")
    sensitivity_rows = []
    for scale in scales:
        for threshold in thresholds:
            with tempfile.TemporaryDirectory(prefix="spei_sensitivity_") as temp_dir:
                metrics_path = os.path.join(temp_dir, "comparison_metrics.csv")
                environment = os.environ.copy()
                environment["SPEI_SCALE"] = scale.zfill(2)
                environment["SPEI_THRESHOLD"] = threshold
                environment["COMPARISON_METRICS_OUTPUT_PATH"] = metrics_path
                environment["COMPARISON_REPORT_OUTPUT_PATH"] = os.path.join(temp_dir, "report.md")
                environment["COMMON_SUPPORT_OUTPUT_PATH"] = os.path.join(temp_dir, "support.csv")
                environment["PERMUTATION_IMPORTANCE_OUTPUT_PATH"] = os.path.join(temp_dir, "importance.csv")
                subprocess.run(
                    [sys.executable, os.path.join(SCRIPT_DIR, "run_comparison_experiments.py")],
                    check=True,
                    env=environment,
                )
                metrics = pd.read_csv(metrics_path)
                metrics.insert(0, "data_start_year", int(environment.get("DATA_START_YEAR", "1990")))
                metrics.insert(1, "data_end_year", int(environment.get("DATA_END_YEAR", "2024")))
                metrics.insert(2, "svm_subsample_size", int(environment.get("SVM_SUBSAMPLE_SIZE", "50000")))
                metrics.insert(3, "svm_evaluation_size", int(environment.get("SVM_EVALUATION_SIZE", "100000")))
            sensitivity_rows.append(metrics)
            pd.concat(sensitivity_rows, ignore_index=True).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved SPEI sensitivity results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()