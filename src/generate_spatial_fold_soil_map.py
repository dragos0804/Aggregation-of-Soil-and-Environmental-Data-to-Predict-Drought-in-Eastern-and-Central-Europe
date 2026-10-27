"""Map spatial-fold transfer performance and test its relationship to soil texture.

Addresses the reviewer request to show the spatial distribution of soil parameters
alongside spatial-fold transfer performance (Section: Supplementary extrapolation
results). Re-uses the already-computed fold boundaries/scores from
``results/extrapolation_validation_detailed.csv`` (all evaluated leave-block-out
folds, not just a handful) and aggregates SOTWIS soil attributes within each fold's
bounding box, then tests the soil-texture/transfer-performance relationship with a
Pearson correlation rather than an anecdotal reading of a few folds.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from generate_spatial_matching_diagnostics import add_map_furniture, load_country_borders

MISSING_SENTINEL = -1.0
START_YEAR = int(os.getenv("DATA_START_YEAR", "1990"))
END_YEAR = int(os.getenv("DATA_END_YEAR", "2024"))
SOIL_PROPERTIES = {
    "CLPC": "Clay content (mass %)",
    "BULK": "Bulk density (g/cm3)",
}


def compute_fold_soil_means(sotwis_df, fold_summary):
    rows = []
    for _, row in fold_summary.iterrows():
        block_mask = (
            (sotwis_df["LAT"] >= row["holdout_lat_min"])
            & (sotwis_df["LAT"] <= row["holdout_lat_max"])
            & (sotwis_df["LON"] >= row["holdout_lon_min"])
            & (sotwis_df["LON"] <= row["holdout_lon_max"])
        )
        record = {
            "fold": int(row["fold"]),
            "holdout_block": row["holdout_block"],
            "holdout_lat_min": row["holdout_lat_min"],
            "holdout_lat_max": row["holdout_lat_max"],
            "holdout_lon_min": row["holdout_lon_min"],
            "holdout_lon_max": row["holdout_lon_max"],
            "aggregated_rf_spatial_drought_f1": row["drought_f1"],
            "n_polygons": int(block_mask.sum()),
        }
        for property_name in SOIL_PROPERTIES:
            values = sotwis_df.loc[block_mask & (sotwis_df[property_name] != MISSING_SENTINEL), property_name]
            record[f"mean_{property_name}"] = float(values.mean()) if len(values) else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def plot_fold_choropleth(ax, fold_df, country_borders):
    """Geographic view of every evaluated fold, colored by its drought-F1."""
    lat_min = fold_df["holdout_lat_min"].min() - 1
    lat_max = fold_df["holdout_lat_max"].max() + 1
    lon_min = fold_df["holdout_lon_min"].min() - 1
    lon_max = fold_df["holdout_lon_max"].max() + 1

    visible_borders = None
    if country_borders is not None and len(country_borders):
        visible = country_borders.cx[lon_min:lon_max, lat_min:lat_max]
        if len(visible):
            visible_borders = visible

    cmap = plt.get_cmap("RdYlGn")
    f1_min, f1_max = fold_df["aggregated_rf_spatial_drought_f1"].min(), fold_df["aggregated_rf_spatial_drought_f1"].max()
    norm = plt.Normalize(vmin=f1_min, vmax=f1_max)
    for _, row in fold_df.iterrows():
        width = row["holdout_lon_max"] - row["holdout_lon_min"]
        height = row["holdout_lat_max"] - row["holdout_lat_min"]
        color = cmap(norm(row["aggregated_rf_spatial_drought_f1"]))
        ax.add_patch(
            plt.Rectangle(
                (row["holdout_lon_min"], row["holdout_lat_min"]),
                width, height,
                facecolor=color, edgecolor="#34434b", linewidth=0.6, alpha=0.56, zorder=2,
            )
        )
    if visible_borders is not None:
        visible_borders.boundary.plot(ax=ax, color="white", linewidth=1.6, zorder=5)
        visible_borders.boundary.plot(ax=ax, color="#34434b", linewidth=0.55, zorder=6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    colorbar = plt.colorbar(sm, ax=ax, label="Drought-F1", fraction=0.046, pad=0.04)
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    add_map_furniture(ax, lon_min, lon_max, lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"All {len(fold_df)} evaluated leave-block-out folds")
    return colorbar


def plot_soil_vs_f1(ax, fold_df, property_name, property_label):
    x = fold_df[f"mean_{property_name}"]
    y = fold_df["aggregated_rf_spatial_drought_f1"]
    r, p = pearsonr(x, y)
    ax.scatter(x, y, color="#2878b5", edgecolor="#174a70", zorder=2)
    coeffs = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 50)
    ax.plot(x_line, np.polyval(coeffs, x_line), color="#d62828", linestyle="--", linewidth=1.2, zorder=1)
    ax.set_xlabel(property_label)
    ax.set_ylabel("Aggregated RF spatial drought-F1")
    ax.set_title(f"r={r:.2f}, p={p:.2f} (n={len(fold_df)})")
    ax.grid(linestyle=":", alpha=0.4)
    return r, p


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    paper_fig_dir = os.path.abspath(os.path.join(repo_root, "..", "SoilEnvironmentUpdates", "figures"))
    results_dir = os.path.join(repo_root, "results")

    sotwis_path = os.path.join(repo_root, "data", "intermediary_steps", "synthetically_enhanced_sotwis.csv")
    period_results_path = os.path.join(
        results_dir, f"extrapolation_validation_detailed_{START_YEAR}_{END_YEAR}.csv"
    )
    if os.path.exists(period_results_path):
        extrapolation_path = period_results_path
        result_start_year, result_end_year = START_YEAR, END_YEAR
    else:
        extrapolation_path = os.path.join(results_dir, "extrapolation_validation_detailed.csv")
        result_start_year, result_end_year = 1990, 2013

    sotwis_df = pd.read_csv(sotwis_path)
    sotwis_df["LAT"] = (sotwis_df["BOTTOM_LEFT_LAT"] + sotwis_df["UPPER_RIGHT_LAT"]) / 2.0
    sotwis_df["LON"] = (sotwis_df["BOTTOM_LEFT_LON"] + sotwis_df["UPPER_RIGHT_LON"]) / 2.0

    extrapolation_df = pd.read_csv(extrapolation_path)
    if {"data_start_year", "data_end_year"}.issubset(extrapolation_df.columns):
        extrapolation_df = extrapolation_df[
            (extrapolation_df["data_start_year"] == START_YEAR)
            & (extrapolation_df["data_end_year"] == END_YEAR)
        ]
    fold_summary = extrapolation_df[
        (extrapolation_df["dataset"] == "Aggregated")
        & (extrapolation_df["model"] == "Random Forest")
        & (extrapolation_df["scheme"] == "Spatial Block (leave-block-out)")
    ].dropna(subset=["holdout_lat_min"])

    fold_df = compute_fold_soil_means(sotwis_df, fold_summary)
    country_borders = load_country_borders()

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 14), dpi=180, gridspec_kw={"height_ratios": [3.3, 1, 1]}
    )
    map_colorbar = plot_fold_choropleth(axes[0], fold_df, country_borders)
    correlations = {}
    for axis, (property_name, property_label) in zip(axes[1:], SOIL_PROPERTIES.items()):
        r, p = plot_soil_vs_f1(axis, fold_df, property_name, property_label)
        correlations[property_name] = (r, p)
    axes[2].set_ylabel("")

    fig.suptitle(
        f"Spatial-fold transfer across all {len(fold_df)} evaluated leave-block-out folds\n"
        f"Block-level SOTWIS soil properties ({result_start_year}--{result_end_year})",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.8)
    fig.canvas.draw()
    map_group = (axes[0], map_colorbar.ax)
    group_left = min(axis.get_position().x0 for axis in map_group)
    group_right = max(axis.get_position().x1 for axis in map_group)
    horizontal_shift = 0.5 - (group_left + group_right) / 2
    for axis in map_group:
        position = axis.get_position()
        axis.set_position((position.x0 + horizontal_shift, position.y0, position.width, position.height))
    output_path = os.path.join(paper_fig_dir, "spatial_fold_soil_property_map.png")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    summary_csv = os.path.join(
        paper_fig_dir, f"spatial_fold_soil_property_summary_{result_start_year}_{result_end_year}.csv"
    )
    fold_df.round(4).to_csv(summary_csv, index=False)

    print(f"Saved {output_path}")
    print(f"Saved {summary_csv}")
    for property_name, (r, p) in correlations.items():
        print(f"Pearson r ({property_name}, drought-F1) = {r:.4f}, p = {p:.4f}")


if __name__ == "__main__":
    main()

