import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HEADING_PATTERN = re.compile(r"^### (.+?) on (.+?)$", re.MULTILINE)
ROW_PATTERN = re.compile(
    r"\| Actual (No Drought|Drought)\s*\|\s*([\d,]+)\s*\|\s*([\d,]+)\s*\|"
)


def parse_confusion_matrices(report_text):
    matches = list(HEADING_PATTERN.finditer(report_text))
    rows = []

    for index, match in enumerate(matches):
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(report_text)
        section = report_text[match.end():section_end]
        matrix_rows = {
            label: (int(no_drought.replace(",", "")), int(drought.replace(",", "")))
            for label, no_drought, drought in ROW_PATTERN.findall(section)
        }
        if set(matrix_rows) != {"No Drought", "Drought"}:
            raise ValueError(f"Could not parse confusion matrix for {match.group(0)}")

        true_negative, false_positive = matrix_rows["No Drought"]
        false_negative, true_positive = matrix_rows["Drought"]
        sample_count = true_negative + false_positive + false_negative + true_positive
        rows.append(
            {
                "model": match.group(1),
                "pipeline": match.group(2),
                "sample_count": sample_count,
                "true_negative": true_negative,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_positive": true_positive,
                "observed_drought_rate": (false_negative + true_positive) / sample_count,
                "predicted_drought_rate": (false_positive + true_positive) / sample_count,
            }
        )

    if len(rows) != 9:
        raise ValueError(f"Expected 9 model-pipeline matrices, found {len(rows)}")
    return pd.DataFrame(rows)


def plot_summary(frame, output_path):
    pipeline_order = [
        "SOTWIS-Only (separate static susceptibility experiment)",
        "NASA-Only (common support)",
        "Fused (SOTWIS + NASA, common support)",
    ]
    pipeline_titles = ["SOTWIS-Only", "NASA-Only", "Fused"]
    model_order = ["Random Forest", "SVM (RBF Kernel)", "LSTM"]
    colors = {"Random Forest": "#287271", "SVM (RBF Kernel)": "#d97706", "LSTM": "#7b2cbf"}

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharey=True, constrained_layout=True)
    x_positions = np.array([0, 1])

    for axis, pipeline, title in zip(axes, pipeline_order, pipeline_titles):
        subset = frame.loc[frame["pipeline"] == pipeline].set_index("model").loc[model_order]
        values = 100 * subset[["observed_drought_rate", "predicted_drought_rate"]].to_numpy()

        for model_index, model in enumerate(model_order):
            offset = (model_index - 1) * 0.06
            axis.plot(
                x_positions + offset,
                values[model_index],
                marker="o",
                linewidth=1.4,
                markersize=5,
                color=colors[model],
                alpha=0.85,
                label=model if axis is axes[0] else None,
            )

        medians = np.median(values, axis=0)
        standard_deviations = np.std(values, axis=0, ddof=1)
        axis.errorbar(
            x_positions,
            medians,
            yerr=standard_deviations,
            fmt="s",
            color="black",
            markerfacecolor="white",
            capsize=4,
            linewidth=2,
            markersize=6,
            label="Median +/- SD" if axis is axes[0] else None,
            zorder=5,
        )
        axis.set_title(title)
        axis.set_xticks(x_positions, ["Observed", "Predicted"])
        axis.grid(axis="y", alpha=0.25)

    fig.supylabel("Drought share of evaluated samples (%)")
    fig.legend(loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("Observed and predicted drought prevalence by model and pipeline")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(frame, output_path):
    pipeline_order = [
        "SOTWIS-Only (separate static susceptibility experiment)",
        "NASA-Only (common support)",
        "Fused (SOTWIS + NASA, common support)",
    ]
    pipeline_titles = ["SOTWIS-Only", "NASA-Only", "Fused"]
    model_order = ["Random Forest", "SVM (RBF Kernel)", "LSTM"]
    model_titles = ["Random Forest", "SVM (RBF)", "LSTM"]

    indexed = frame.set_index(["model", "pipeline"])
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 9), constrained_layout=True)

    for row_index, (model, model_title) in enumerate(zip(model_order, model_titles)):
        for column_index, (pipeline, pipeline_title) in enumerate(zip(pipeline_order, pipeline_titles)):
            axis = axes[row_index, column_index]
            record = indexed.loc[(model, pipeline)]
            matrix = np.array(
                [
                    [record["true_negative"], record["false_positive"]],
                    [record["false_negative"], record["true_positive"]],
                ],
                dtype=int,
            )
            normalized = matrix / matrix.sum(axis=1, keepdims=True)
            axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            axis.set_xticks([0, 1], ["Pred ND", "Pred D"])
            axis.set_yticks([0, 1], ["True ND", "True D"])
            if row_index == 0:
                axis.set_title(pipeline_title)
            if column_index == 0:
                axis.set_ylabel(model_title)
            for matrix_row in range(2):
                for matrix_column in range(2):
                    axis.text(
                        matrix_column,
                        matrix_row,
                        f"{matrix[matrix_row, matrix_column]:,}\n{normalized[matrix_row, matrix_column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    fig.suptitle("Canonical held-out drought confusion matrices\n(count and row-normalized rate)")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_confusion_matrices(frame, output_path):
    scored = frame.copy()
    scored["precision_drought"] = scored["true_positive"] / (
        scored["true_positive"] + scored["false_positive"]
    )
    scored["recall_drought"] = scored["true_positive"] / (
        scored["true_positive"] + scored["false_negative"]
    )
    scored["f1_drought"] = (
        2
        * scored["precision_drought"]
        * scored["recall_drought"]
        / (scored["precision_drought"] + scored["recall_drought"])
    )
    best_rows = scored.loc[scored.groupby("pipeline")["f1_drought"].idxmax()]
    best_rows = best_rows.sort_values("pipeline")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    for axis, (_, record) in zip(axes, best_rows.iterrows()):
        matrix = np.array(
            [
                [record["true_negative"], record["false_positive"]],
                [record["false_negative"], record["true_positive"]],
            ],
            dtype=int,
        )
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        axis.set_xticks([0, 1], ["Pred ND", "Pred D"])
        axis.set_yticks([0, 1], ["True ND", "True D"])
        short_pipeline = record["pipeline"].split(" (")[0].replace("Fused", "Fused")
        axis.set_title(f"{short_pipeline}\n{record['model']} (F1={record['f1_drought']:.3f})")
        for matrix_row in range(2):
            for matrix_column in range(2):
                axis.text(
                    matrix_column,
                    matrix_row,
                    f"{matrix[matrix_row, matrix_column]:,}\n{normalized[matrix_row, matrix_column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.suptitle("Highest drought-F1 model within each pipeline")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_confusion_metrics(frame, csv_path, markdown_path):
    metrics = frame.copy()
    metrics["Precision_D"] = metrics["true_positive"] / (
        metrics["true_positive"] + metrics["false_positive"]
    )
    metrics["Recall_D"] = metrics["true_positive"] / (
        metrics["true_positive"] + metrics["false_negative"]
    )
    metrics["F1_D"] = (
        2 * metrics["Precision_D"] * metrics["Recall_D"]
        / (metrics["Precision_D"] + metrics["Recall_D"])
    )
    metrics = metrics.rename(
        columns={
            "model": "Model",
            "pipeline": "Dataset",
            "true_negative": "TN",
            "false_positive": "FP",
            "false_negative": "FN",
            "true_positive": "TP",
        }
    )[
        ["Model", "Dataset", "TN", "FP", "FN", "TP", "Precision_D", "Recall_D", "F1_D"]
    ]
    metrics.to_csv(csv_path, index=False)
    markdown_path.write_text(
        "# Canonical Model/Dataset Confusion-Matrix Metrics\n\n"
        + metrics.round(4).to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def main():
    repo_root = Path(__file__).resolve().parent.parent
    report_path = repo_root / "results" / "full_experimental_report.md"
    output_csv = repo_root / "results" / "observed_predicted_drought_rates.csv"
    confusion_csv = repo_root / "results" / "all_models_confusion_metrics.csv"
    confusion_markdown = repo_root / "results" / "all_models_confusion_metrics.md"
    figure_dir = repo_root.parent / "SoilEnvironmentUpdates" / "figures"
    output_figure = figure_dir / "observed_predicted_drought_rates.png"
    confusion_figure = figure_dir / "confusion_matrices_all_models.png"
    best_confusion_figure = figure_dir / "confusion_matrices_best_models.png"

    frame = parse_confusion_matrices(report_path.read_text(encoding="utf-8"))
    frame.to_csv(output_csv, index=False)
    write_confusion_metrics(frame, confusion_csv, confusion_markdown)
    plot_summary(frame, output_figure)
    plot_confusion_matrices(frame, confusion_figure)
    plot_best_confusion_matrices(frame, best_confusion_figure)
    print(frame.to_string(index=False))
    print(f"Saved {output_csv}")
    print(f"Saved {output_figure}")
    print(f"Saved {confusion_figure}")
    print(f"Saved {best_confusion_figure}")


if __name__ == "__main__":
    main()