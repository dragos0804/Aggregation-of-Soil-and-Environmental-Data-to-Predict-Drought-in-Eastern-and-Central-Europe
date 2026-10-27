"""Repeat the primary LSTM configuration over several seeds on the common-support pipelines."""

import os
import subprocess
import sys
import tempfile

import pandas as pd

SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "..", "results", "lstm_seed_repeats_1990_to_2024.csv")


def main():
    seeds = [int(v) for v in os.getenv("LSTM_REPEAT_SEEDS", "42,43,44,45,46").split(",")]
    rows = []
    for seed in seeds:
        with tempfile.TemporaryDirectory(prefix="lstm_seed_") as temp_dir:
            metrics_path = os.path.join(temp_dir, "comparison_metrics.csv")
            environment = os.environ.copy()
            environment.update({
                "EXPERIMENT_SEED": str(seed),
                "EXPERIMENT_MODELS": "LSTM",
                "RUN_SOTWIS_ONLY": "false",
                "COMPARISON_METRICS_OUTPUT_PATH": metrics_path,
                "COMPARISON_REPORT_OUTPUT_PATH": os.path.join(temp_dir, "report.md"),
                "COMMON_SUPPORT_OUTPUT_PATH": os.path.join(temp_dir, "support.csv"),
                "PERMUTATION_IMPORTANCE_OUTPUT_PATH": os.path.join(temp_dir, "importance.csv"),
            })
            subprocess.run(
                [sys.executable, os.path.join(SCRIPT_DIR, "run_comparison_experiments.py")],
                check=True,
                env=environment,
            )
            metrics = pd.read_csv(metrics_path)
            metrics.insert(0, "seed", seed)
        rows.append(metrics)
        pd.concat(rows, ignore_index=True).to_csv(OUTPUT_PATH, index=False)
    print(f"Saved LSTM seed repeats to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
