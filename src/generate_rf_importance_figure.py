"""Generate manuscript RF held-out permutation-importance figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT.parent / "SoilEnvironmentUpdates" / "figures"
IMPORTANCE_PATH = RESULTS_DIR / "rf_permutation_importance.csv"
PIPELINES = (
    (
        "NASA-Only (common support)",
        "rf_nasa_feature_importance",
        "#2878b5",
        "#174a70",
        "NASA POWER feature",
        "NASA-only RF held-out permutation importance",
    ),
    (
        "Fused (SOTWIS + NASA, common support)",
        "rf_fused_feature_importance",
        "#3b8c5a",
        "#215c38",
        "Fused model feature",
        "Fused RF held-out permutation importance",
    ),
)


def save_importance_figure(
    importance: pd.DataFrame,
    pipeline_name: str,
    output_stem: str,
    color: str,
    edgecolor: str,
    feature_label: str,
    title: str,
):
    selected = importance.loc[importance["pipeline"].eq(pipeline_name)].copy()
    selected = selected.sort_values("importance_mean", ascending=False).head(12)
    selected = selected.sort_values("importance_mean")
    selected.to_csv(FIGURES_DIR / f"{output_stem}.csv", index=False)
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    axis.barh(
        selected["feature"],
        selected["importance_mean"],
        xerr=selected["importance_std"],
        color=color,
        edgecolor=edgecolor,
        linewidth=0.5,
        capsize=3,
    )
    axis.set_xlabel("Mean decrease in held-out drought F1 after permutation")
    axis.set_ylabel(feature_label)
    axis.set_title(title)
    axis.grid(axis="x", linestyle=":", alpha=0.45)
    axis.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"{output_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    importance = pd.read_csv(IMPORTANCE_PATH)
    for pipeline in PIPELINES:
        save_importance_figure(importance, *pipeline)


if __name__ == "__main__":
    main()
