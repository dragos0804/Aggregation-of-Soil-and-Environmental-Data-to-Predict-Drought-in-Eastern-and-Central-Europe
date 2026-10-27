"""Measure LSTM sequence-length sensitivity on the common-support pipelines.

This is a descriptive sensitivity analysis on the chronological evaluation
split, not a model-selection procedure. Do not select a primary window from its
held-out test scores.
"""

import os
import subprocess
import sys
import tempfile

import pandas as pd


SCRIPT_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
DATA_START_YEAR = int(os.getenv("DATA_START_YEAR", "1990"))
DATA_END_YEAR = int(os.getenv("DATA_END_YEAR", "2024"))
OUTPUT_PATH = os.getenv(
    "SEQUENCE_LENGTH_SENSITIVITY_OUTPUT_PATH",
    os.path.join(
        RESULTS_DIR,
        f"lstm_sequence_length_sensitivity_{DATA_START_YEAR}_to_{DATA_END_YEAR}.csv",
    ),
)


def main():
    lengths = [int(v) for v in os.getenv("LSTM_SEQUENCE_LENGTHS", "3,6,12,14").split(",")]
    sensitivity_rows = []
    for length in lengths:
        with tempfile.TemporaryDirectory(prefix="sequence_sensitivity_") as temp_dir:
            metrics_path = os.path.join(temp_dir, "comparison_metrics.csv")
            environment = os.environ.copy()
            environment["LSTM_SEQUENCE_LENGTH"] = str(length)
            environment["EXPERIMENT_MODELS"] = "LSTM"
            environment["RUN_SOTWIS_ONLY"] = "false"
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
            metrics.insert(0, "data_start_year", DATA_START_YEAR)
            metrics.insert(1, "data_end_year", DATA_END_YEAR)
        sensitivity_rows.append(metrics)
        pd.concat(sensitivity_rows, ignore_index=True).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved LSTM sequence-length sensitivity results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
