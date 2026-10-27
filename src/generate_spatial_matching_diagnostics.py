import glob
import math
import os
import site
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

START_YEAR = int(os.getenv("DATA_START_YEAR", "1990"))
END_YEAR = int(os.getenv("DATA_END_YEAR", "2024"))


def load_country_borders():
    """Load a bundled offline Natural Earth country-borders dataset, if available.

    No network access is used; this looks for the low-resolution Natural Earth
    admin-0 countries shapefile that ships as a test fixture inside the
    installed ``pyogrio`` package. Returns ``None`` if it cannot be found or
    ``geopandas`` is unavailable, so map generation degrades gracefully.
    """
    try:
        import geopandas as gpd
    except ImportError:
        return None

    search_roots = list(site.getsitepackages()) if hasattr(site, "getsitepackages") else []
    user_site = site.getusersitepackages() if hasattr(site, "getusersitepackages") else None
    if user_site:
        search_roots.append(user_site)

    for root in search_roots:
        candidates = glob.glob(
            os.path.join(root, "pyogrio", "tests", "fixtures", "naturalearth_lowres", "naturalearth_lowres.shp")
        )
        if candidates:
            try:
                return gpd.read_file(candidates[0])
            except Exception:
                return None
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two points in degrees."""
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def add_map_furniture(ax, lon_min, lon_max, lat_min, lat_max, bar_km=200):
    """Add a scale bar, north arrow, and CRS label to a lat/lon-projected map axis."""
    mean_lat = (lat_min + lat_max) / 2.0
    lon_span_km = haversine_km(mean_lat, lon_min, mean_lat, lon_max)
    bar_lon_span = (lon_max - lon_min) * (bar_km / lon_span_km) if lon_span_km else 0.0

    bar_x0 = lon_min + (lon_max - lon_min) * 0.03
    bar_y0 = lat_min + (lat_max - lat_min) * 0.05
    ax.plot([bar_x0, bar_x0 + bar_lon_span], [bar_y0, bar_y0], color="black", linewidth=2.5, solid_capstyle="butt", zorder=10)
    ax.text(
        bar_x0 + bar_lon_span / 2.0, bar_y0 + (lat_max - lat_min) * 0.015,
        f"{bar_km} km", ha="center", va="bottom", fontsize=8, zorder=10,
    )

    arrow_x = lon_min + (lon_max - lon_min) * 0.03
    arrow_y0 = lat_min + (lat_max - lat_min) * 0.14
    arrow_y1 = arrow_y0 + (lat_max - lat_min) * 0.07
    ax.annotate(
        "N", xy=(arrow_x, arrow_y1), xytext=(arrow_x, arrow_y0),
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.5),
        ha="center", va="bottom", fontsize=9, fontweight="bold", zorder=10,
    )

    ax.text(
        0.99, 0.01, "CRS: EPSG:4326 (WGS84 geographic)",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#333333",
        bbox=dict(boxstyle="round", fc="white", ec="none", alpha=0.7), zorder=10,
    )


def infer_grid_step(values, rounding=6):
    """Infer native grid step from sorted unique coordinate values."""
    uniq = np.unique(np.round(values, rounding))
    if len(uniq) < 2:
        return np.nan
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return np.nan
    return float(np.median(diffs))


def load_unique_nasa_coords_with_counts(nasa_path, chunk_size=500_000):
    """Read NASA file in chunks to collect unique coordinates and per-coordinate record counts."""
    coord_counts = defaultdict(int)
    total_records = 0

    for chunk in pd.read_csv(nasa_path, usecols=["LAT", "LON", "YEAR"], chunksize=chunk_size):
        chunk = chunk[chunk["YEAR"].between(START_YEAR, END_YEAR)]
        grouped = chunk.groupby(["LAT", "LON"]).size()
        for (lat, lon), count in grouped.items():
            coord_counts[(float(lat), float(lon))] += int(count)
        total_records += int(len(chunk))

    coords_df = pd.DataFrame(
        [(lat, lon, cnt) for (lat, lon), cnt in coord_counts.items()],
        columns=["LAT", "LON", "record_count"],
    )
    return coords_df, total_records


def classify_matches(coords_df, sotwis_df):
    """Classify each NASA coordinate as unmatched, uniquely matched, or overlapping match."""
    bounds = sotwis_df[["BOTTOM_LEFT_LAT", "UPPER_RIGHT_LAT", "BOTTOM_LEFT_LON", "UPPER_RIGHT_LON"]].values

    overlap_counts = []
    chosen_index = []
    centroids = []

    for lat, lon in coords_df[["LAT", "LON"]].itertuples(index=False, name=None):
        mask = (
            (lat >= bounds[:, 0])
            & (lat <= bounds[:, 1])
            & (lon >= bounds[:, 2])
            & (lon <= bounds[:, 3])
        )
        idx = np.where(mask)[0]
        overlap_counts.append(int(len(idx)))

        if len(idx) > 0:
            first_idx = int(idx[0])
            chosen_index.append(first_idx)
            row = sotwis_df.iloc[first_idx]
            c_lat = (row["BOTTOM_LEFT_LAT"] + row["UPPER_RIGHT_LAT"]) / 2.0
            c_lon = (row["BOTTOM_LEFT_LON"] + row["UPPER_RIGHT_LON"]) / 2.0
            centroids.append((c_lat, c_lon))
        else:
            chosen_index.append(np.nan)
            centroids.append((np.nan, np.nan))

    out = coords_df.copy()
    out["overlap_count"] = overlap_counts
    out["sotwis_index"] = chosen_index
    out["matched"] = out["overlap_count"] > 0
    out["low_confidence_overlap"] = out["overlap_count"] > 1
    out[["matched_centroid_lat", "matched_centroid_lon"]] = pd.DataFrame(centroids, index=out.index)

    matched_mask = out["matched"]
    if matched_mask.any():
        out.loc[matched_mask, "distance_to_centroid_km"] = [
            haversine_km(lat, lon, c_lat, c_lon)
            for lat, lon, c_lat, c_lon in out.loc[
                matched_mask, ["LAT", "LON", "matched_centroid_lat", "matched_centroid_lon"]
            ].itertuples(index=False, name=None)
        ]
    else:
        out["distance_to_centroid_km"] = np.nan

    return out


def build_resolution_summary(coords_df, sotwis_df):
    """Build a concise native-resolution comparison for NASA points and SOTWIS polygons."""
    lat_step_deg = infer_grid_step(coords_df["LAT"].values)
    lon_step_deg = infer_grid_step(coords_df["LON"].values)
    mean_lat = float(coords_df["LAT"].mean())

    n_lat_km = haversine_km(mean_lat - lat_step_deg / 2.0, 0.0, mean_lat + lat_step_deg / 2.0, 0.0)
    n_lon_km = haversine_km(mean_lat, -lon_step_deg / 2.0, mean_lat, lon_step_deg / 2.0)

    h_deg = (sotwis_df["UPPER_RIGHT_LAT"] - sotwis_df["BOTTOM_LEFT_LAT"]).astype(float)
    w_deg = (sotwis_df["UPPER_RIGHT_LON"] - sotwis_df["BOTTOM_LEFT_LON"]).astype(float)

    s_lat_km = np.array([haversine_km(mean_lat - d / 2.0, 0.0, mean_lat + d / 2.0, 0.0) for d in h_deg])
    s_lon_km = np.array([haversine_km(mean_lat, -d / 2.0, mean_lat, d / 2.0) for d in w_deg])

    nasa_area_km2 = float(n_lat_km * n_lon_km)
    sotwis_area_km2 = float(np.median(s_lat_km * s_lon_km))

    rows = [
        {
            "Source": "NASA POWER (inferred from coordinate lattice)",
            "Lat step (deg)": lat_step_deg,
            "Lon step (deg)": lon_step_deg,
            "Lat step (km, approx)": n_lat_km,
            "Lon step (km, approx)": n_lon_km,
            "Representative area (km2, approx)": nasa_area_km2,
        },
        {
            "Source": "SOTWIS polygons (median bounding-box support)",
            "Lat step (deg)": float(np.median(h_deg)),
            "Lon step (deg)": float(np.median(w_deg)),
            "Lat step (km, approx)": float(np.median(s_lat_km)),
            "Lon step (km, approx)": float(np.median(s_lon_km)),
            "Representative area (km2, approx)": sotwis_area_km2,
        },
    ]

    summary = pd.DataFrame(rows)
    summary["Area ratio vs NASA"] = summary["Representative area (km2, approx)"] / nasa_area_km2
    return summary, mean_lat


def build_match_quality_summary(match_df, total_records):
    """Build record-level and coordinate-level matching quality summary."""
    total_unique = int(len(match_df))
    matched_unique = int(match_df["matched"].sum())
    unmatched_unique = total_unique - matched_unique
    overlap_unique = int(match_df["low_confidence_overlap"].sum())

    matched_records = int(match_df.loc[match_df["matched"], "record_count"].sum())
    unmatched_records = int(total_records - matched_records)
    overlap_records = int(match_df.loc[match_df["low_confidence_overlap"], "record_count"].sum())

    dist = match_df.loc[match_df["matched"], "distance_to_centroid_km"].dropna()

    rows = [
        {
            "Metric": "Total NASA records",
            "Value": total_records,
            "Share": 1.0,
        },
        {
            "Metric": "Matched NASA records",
            "Value": matched_records,
            "Share": matched_records / total_records if total_records else np.nan,
        },
        {
            "Metric": "Unmatched NASA records",
            "Value": unmatched_records,
            "Share": unmatched_records / total_records if total_records else np.nan,
        },
        {
            "Metric": "Total unique NASA coordinate pairs",
            "Value": total_unique,
            "Share": 1.0,
        },
        {
            "Metric": "Matched unique coordinate pairs",
            "Value": matched_unique,
            "Share": matched_unique / total_unique if total_unique else np.nan,
        },
        {
            "Metric": "Unmatched unique coordinate pairs",
            "Value": unmatched_unique,
            "Share": unmatched_unique / total_unique if total_unique else np.nan,
        },
        {
            "Metric": "Low-confidence unique pairs (overlap_count > 1)",
            "Value": overlap_unique,
            "Share": overlap_unique / total_unique if total_unique else np.nan,
        },
        {
            "Metric": "Low-confidence records (overlap_count > 1)",
            "Value": overlap_records,
            "Share": overlap_records / total_records if total_records else np.nan,
        },
        {
            "Metric": "Median coord-to-matched-centroid distance (km)",
            "Value": float(dist.median()) if len(dist) else np.nan,
            "Share": np.nan,
        },
        {
            "Metric": "95th percentile coord-to-matched-centroid distance (km)",
            "Value": float(dist.quantile(0.95)) if len(dist) else np.nan,
            "Share": np.nan,
        },
    ]

    return pd.DataFrame(rows)


def build_retained_excluded_summary(match_df):
    """Summarize the geographic support retained and excluded by box containment."""
    grouped = match_df.assign(
        support=np.where(match_df["matched"], "Retained (matched)", "Excluded (unmatched)")
    ).groupby("support", sort=False)

    rows = []
    for support, group in grouped:
        rows.append(
            {
                "Support": support,
                "Unique coordinate pairs": int(len(group)),
                "NASA records": int(group["record_count"].sum()),
                "Latitude mean": float(group["LAT"].mean()),
                "Latitude 5th percentile": float(group["LAT"].quantile(0.05)),
                "Latitude 95th percentile": float(group["LAT"].quantile(0.95)),
                "Longitude mean": float(group["LON"].mean()),
                "Longitude 5th percentile": float(group["LON"].quantile(0.05)),
                "Longitude 95th percentile": float(group["LON"].quantile(0.95)),
            }
        )
    return pd.DataFrame(rows)


def plot_unmatched_map(match_df, output_path, sotwis_bounds=None, country_borders=None):
    """Generate a spatial distribution map for matched vs unmatched NASA coordinates.

    Adds country borders/names and the SOTWIS coverage bounding box when the
    corresponding data is available, so the figure includes geographic
    context rather than bare latitude-longitude scatter.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)

    lon_min, lon_max = match_df["LON"].min(), match_df["LON"].max()
    lat_min, lat_max = match_df["LAT"].min(), match_df["LAT"].max()
    lon_pad = (lon_max - lon_min) * 0.05
    lat_pad = (lat_max - lat_min) * 0.05
    extent = (lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad)

    visible_borders = None
    if country_borders is not None and len(country_borders):
        visible = country_borders.cx[extent[0]:extent[1], extent[2]:extent[3]]
        if len(visible):
            visible_borders = visible

    bbox_patch = None
    if sotwis_bounds is not None:
        s_lon_min, s_lon_max, s_lat_min, s_lat_max = sotwis_bounds
        bbox_patch = plt.Rectangle(
                (s_lon_min, s_lat_min),
                s_lon_max - s_lon_min,
                s_lat_max - s_lat_min,
                fill=False,
                edgecolor="#264653",
                linewidth=1.4,
                linestyle="--",
                label="SOTWIS coverage extent (bounding box)",
                zorder=3,
        )
        ax.add_patch(bbox_patch)

    grid_deg = 1.0
    binned = match_df.assign(
        lat_cell=np.floor(match_df["LAT"] / grid_deg) * grid_deg,
        lon_cell=np.floor(match_df["LON"] / grid_deg) * grid_deg,
    ).groupby(["lat_cell", "lon_cell"])["matched"].mean().reset_index()
    lat_edges = np.arange(np.floor(extent[2] / grid_deg) * grid_deg,
                          np.ceil(extent[3] / grid_deg) * grid_deg + grid_deg, grid_deg)
    lon_edges = np.arange(np.floor(extent[0] / grid_deg) * grid_deg,
                          np.ceil(extent[1] / grid_deg) * grid_deg + grid_deg, grid_deg)
    share_grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)
    for _, row in binned.iterrows():
        lat_idx = int(round((row["lat_cell"] - lat_edges[0]) / grid_deg))
        lon_idx = int(round((row["lon_cell"] - lon_edges[0]) / grid_deg))
        share_grid[lat_idx, lon_idx] = row["matched"]

    mesh = ax.pcolormesh(
        lon_edges, lat_edges, share_grid, cmap="RdYlGn", vmin=0, vmax=1,
        shading="flat", rasterized=True, zorder=2,
    )
    fig.colorbar(mesh, ax=ax, label="Share of coordinates matched", fraction=0.046, pad=0.04)

    if visible_borders is not None:
        visible_borders.boundary.plot(ax=ax, color="white", linewidth=1.5, zorder=5)
        visible_borders.boundary.plot(ax=ax, color="#34434b", linewidth=0.55, zorder=6)
        for _, row in visible_borders.iterrows():
            label_point = row.geometry.representative_point()
            if extent[0] <= label_point.x <= extent[1] and extent[2] <= label_point.y <= extent[3]:
                ax.annotate(
                    row["name"],
                    xy=(label_point.x, label_point.y),
                    fontsize=6,
                    color="#263238",
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.62),
                    zorder=7,
                )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    add_map_furniture(ax, extent[0], extent[1], extent[2], extent[3])
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        f"NASA POWER--SOTWIS matching share, {START_YEAR}--{END_YEAR} ({grid_deg:g}\u00b0 grid)"
    )
    if bbox_patch is not None:
        ax.legend(handles=[bbox_patch], frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    paper_fig_dir = os.path.abspath(os.path.join(repo_root, "..", "SoilEnvironmentUpdates", "figures"))

    os.makedirs(paper_fig_dir, exist_ok=True)

    nasa_path = os.path.join(repo_root, "data", "intermediary_steps", "complete_nasa_power_data_1990_to_2024.csv")
    sotwis_path = os.path.join(repo_root, "data", "intermediary_steps", "synthetically_enhanced_sotwis.csv")

    print("Loading unique NASA coordinates and record counts...")
    nasa_coords_df, total_records = load_unique_nasa_coords_with_counts(nasa_path)

    print("Loading SOTWIS polygons...")
    sotwis_df = pd.read_csv(sotwis_path)

    print("Classifying matches and overlap confidence...")
    match_df = classify_matches(nasa_coords_df, sotwis_df)

    print("Building summaries...")
    resolution_df, mean_lat = build_resolution_summary(nasa_coords_df, sotwis_df)
    match_quality_df = build_match_quality_summary(match_df, total_records)
    retained_excluded_df = build_retained_excluded_summary(match_df)

    # Persist detailed point-level diagnostics for auditing and reproducibility.
    match_df_out = os.path.join(paper_fig_dir, "spatial_match_diagnostics_points.csv")
    resolution_out = os.path.join(paper_fig_dir, "spatial_resolution_comparison.csv")
    quality_out = os.path.join(paper_fig_dir, "spatial_matching_quality_summary.csv")
    retained_excluded_out = os.path.join(paper_fig_dir, "spatial_matching_retained_excluded_summary.csv")
    map_out = os.path.join(paper_fig_dir, "spatial_matching_unmatched_map.png")

    match_df.to_csv(match_df_out, index=False)
    resolution_df.round(6).to_csv(resolution_out, index=False)
    match_quality_df.round(6).to_csv(quality_out, index=False)
    retained_excluded_df.round(6).to_csv(retained_excluded_out, index=False)

    sotwis_bounds = (
        float(sotwis_df["BOTTOM_LEFT_LON"].min()),
        float(sotwis_df["UPPER_RIGHT_LON"].max()),
        float(sotwis_df["BOTTOM_LEFT_LAT"].min()),
        float(sotwis_df["UPPER_RIGHT_LAT"].max()),
    )
    country_borders = load_country_borders()
    if country_borders is None:
        print("Offline country-borders dataset not found; map will omit borders/names.")
    plot_unmatched_map(match_df, map_out, sotwis_bounds=sotwis_bounds, country_borders=country_borders)

    print("Saved diagnostics files:")
    print(f"  - {match_df_out}")
    print(f"  - {resolution_out}")
    print(f"  - {quality_out}")
    print(f"  - {retained_excluded_out}")
    print(f"  - {map_out}")
    print(f"Inferred mean-latitude reference used for km conversion: {mean_lat:.3f}")


if __name__ == "__main__":
    main()
