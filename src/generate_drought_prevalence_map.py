"""Map the spatial distribution of drought prevalence across the study area.

Addresses the reviewer request for spatial drought maps/plots of the study area
(R3 Major 6), which was previously missing entirely -- prior figures covered
matching diagnostics and soil/fold-performance, but not drought conditions
themselves. Uses the primary SPEI-01 benchmark directly (no model training).
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_spatial_matching_diagnostics import add_map_furniture, load_country_borders

SPEI_THRESHOLD = float(os.getenv("SPEI_THRESHOLD", "-1.0"))
START_YEAR = int(os.getenv("DATA_START_YEAR", "1990"))
END_YEAR = int(os.getenv("DATA_END_YEAR", "2024"))
GRID_DEG = 1.0


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    paper_fig_dir = os.path.abspath(os.path.join(repo_root, "..", "SoilEnvironmentUpdates", "figures"))
    spei_path = os.path.join(repo_root, "data", "intermediary_steps", "complete_spei_benchmark.csv")

    spei_df = pd.read_csv(spei_path, usecols=["LAT", "LON", "DROUGHT_INDEX", "YEAR"])
    spei_df = spei_df[spei_df["YEAR"].between(START_YEAR, END_YEAR)].copy()
    spei_df["is_drought"] = (spei_df["DROUGHT_INDEX"] < SPEI_THRESHOLD).astype(int)

    lat_origin = np.floor(spei_df["LAT"].min() / GRID_DEG) * GRID_DEG
    lon_origin = np.floor(spei_df["LON"].min() / GRID_DEG) * GRID_DEG
    spei_df["lat_cell"] = np.floor((spei_df["LAT"] - lat_origin) / GRID_DEG) * GRID_DEG + lat_origin
    spei_df["lon_cell"] = np.floor((spei_df["LON"] - lon_origin) / GRID_DEG) * GRID_DEG + lon_origin

    grid = spei_df.groupby(["lat_cell", "lon_cell"])["is_drought"].mean().reset_index()

    lat_min, lat_max = spei_df["LAT"].min(), spei_df["LAT"].max()
    lon_min, lon_max = spei_df["LON"].min(), spei_df["LON"].max()
    lat_edges = np.arange(lat_origin, lat_max + GRID_DEG, GRID_DEG)
    lon_edges = np.arange(lon_origin, lon_max + GRID_DEG, GRID_DEG)
    value_grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)
    for _, row in grid.iterrows():
        lat_idx = int(round((row["lat_cell"] - lat_origin) / GRID_DEG))
        lon_idx = int(round((row["lon_cell"] - lon_origin) / GRID_DEG))
        if 0 <= lat_idx < value_grid.shape[0] and 0 <= lon_idx < value_grid.shape[1]:
            value_grid[lat_idx, lon_idx] = row["is_drought"]

    country_borders = load_country_borders()
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    visible_borders = None
    if country_borders is not None and len(country_borders):
        visible = country_borders.cx[lon_min:lon_max, lat_min:lat_max]
        if len(visible):
            visible_borders = visible

    mesh = ax.pcolormesh(lon_edges, lat_edges, value_grid, cmap="YlOrRd", shading="flat", zorder=1, alpha=0.86)
    if visible_borders is not None:
        visible_borders.boundary.plot(ax=ax, color="white", linewidth=1.6, zorder=3)
        visible_borders.boundary.plot(ax=ax, color="#34434b", linewidth=0.55, zorder=4)
    fig.colorbar(
        mesh,
        ax=ax,
        label=f"Fraction of months with SPEI-01 $<$ {SPEI_THRESHOLD:g} ({START_YEAR}--{END_YEAR})",
    )
    ax.set_xlim(lon_min - 1, lon_max + 1)
    ax.set_ylim(lat_min - 1, lat_max + 1)
    add_map_furniture(ax, lon_min, lon_max, lat_min, lat_max)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"Drought prevalence, {START_YEAR}--{END_YEAR} ({GRID_DEG:g}\u00b0 grid, "
        f"SPEI-01 $<$ {SPEI_THRESHOLD:g})"
    )
    fig.tight_layout()

    output_path = os.path.join(paper_fig_dir, "drought_prevalence_map.png")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)

    summary_csv = os.path.join(paper_fig_dir, "drought_prevalence_grid.csv")
    grid.rename(columns={"is_drought": "drought_prevalence"}).round(4).to_csv(summary_csv, index=False)

    print(f"Saved {output_path}")
    print(f"Saved {summary_csv}")
    print(f"Period: {START_YEAR}--{END_YEAR}; months: {spei_df['YEAR'].nunique() * 12}")
    print(f"Overall mean drought prevalence: {spei_df['is_drought'].mean():.4f}")
    print(f"Grid-cell prevalence range: {grid['is_drought'].min():.4f} to {grid['is_drought'].max():.4f}")


if __name__ == "__main__":
    main()
