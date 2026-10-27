"""Build a quality-screened monthly ISMN panel near matched study locations."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = None
DEFAULT_LOCATIONS = (
    PROJECT_ROOT / "data" / "external_validation" / "smap_spl3smp_v9"
    / "matched_sampling_locations.csv"
)
DEFAULT_SPEI = PROJECT_ROOT / "data" / "intermediary_steps" / "complete_spei_benchmark.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "external_validation" / "ismn"
SHALLOW_DEPTH_M = 0.10


def station_header(archive: zipfile.ZipFile, name: str) -> dict[str, object]:
    fields = shlex.split(archive.open(name).readline().decode("utf-8", "replace").strip())
    if len(fields) < 8:
        raise ValueError(f"Unexpected ISMN station header in {name}")
    return {
        "network": fields[0],
        "station": fields[2],
        "latitude": float(fields[3]),
        "longitude": float(fields[4]),
        "depth_from_m": float(fields[6]),
        "depth_to_m": float(fields[7]),
    }


def nearest_locations(
    latitude: float, longitude: float, locations: pd.DataFrame
) -> tuple[int, float]:
    latitudes = np.radians(locations["LAT"].to_numpy(dtype=float))
    longitudes = np.radians(locations["LON"].to_numpy(dtype=float))
    station_latitude = math.radians(latitude)
    station_longitude = math.radians(longitude)
    delta_latitude = latitudes - station_latitude
    delta_longitude = longitudes - station_longitude
    haversine = (
        np.sin(delta_latitude / 2) ** 2
        + math.cos(station_latitude)
        * np.cos(latitudes)
        * np.sin(delta_longitude / 2) ** 2
    )
    distances = 6371.0 * 2 * np.arcsin(np.sqrt(haversine))
    index = int(np.argmin(distances))
    return index, float(distances[index])


def spei_consistency(panel: pd.DataFrame, spei_path: Path, output_dir: Path) -> dict[str, object]:
    station = panel.groupby("station_id", as_index=False).agg(
        network=("network", "first"),
        station=("station", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
    )
    spei_coordinates = (
        pd.read_csv(spei_path, usecols=["LAT", "LON"])
        .dropna()
        .drop_duplicates()
        .to_numpy(dtype=float)
    )
    station_latitudes = np.radians(station["latitude"].to_numpy()[:, None])
    station_longitudes = np.radians(station["longitude"].to_numpy()[:, None])
    grid_latitudes = np.radians(spei_coordinates[:, 0][None, :])
    grid_longitudes = np.radians(spei_coordinates[:, 1][None, :])
    haversine = (
        np.sin((grid_latitudes - station_latitudes) / 2) ** 2
        + np.cos(station_latitudes)
        * np.cos(grid_latitudes)
        * np.sin((grid_longitudes - station_longitudes) / 2) ** 2
    )
    distances = 6371.0 * 2 * np.arcsin(np.sqrt(haversine))
    nearest_indices = distances.argmin(axis=1)
    station["spei_lat"] = spei_coordinates[nearest_indices, 0]
    station["spei_lon"] = spei_coordinates[nearest_indices, 1]
    station["spei_distance_km"] = distances[np.arange(len(station)), nearest_indices]
    selected_grid_cells = set(
        zip(station["spei_lat"].round(5), station["spei_lon"].round(5))
    )

    spei_parts = []
    columns = ["LAT", "LON", "YEAR", "MO", "DROUGHT_INDEX"]
    for chunk in pd.read_csv(spei_path, usecols=columns, chunksize=250_000):
        chunk["grid_key"] = list(zip(chunk["LAT"].round(5), chunk["LON"].round(5)))
        matched = chunk.loc[chunk["grid_key"].isin(selected_grid_cells)].copy()
        if matched.empty:
            continue
        matched["month"] = (
            matched["YEAR"].astype(int).astype(str)
            + "-"
            + matched["MO"].astype(int).astype(str).str.zfill(2)
        )
        spei_parts.append(matched[["LAT", "LON", "month", "DROUGHT_INDEX"]])

    spei = pd.concat(spei_parts, ignore_index=True).drop_duplicates(["LAT", "LON", "month"])
    station_spei = station.merge(
        spei,
        left_on=["spei_lat", "spei_lon"],
        right_on=["LAT", "LON"],
        how="inner",
    )
    pairs = panel.merge(
        station_spei[["station_id", "month", "DROUGHT_INDEX", "spei_distance_km"]],
        on=["station_id", "month"],
        how="inner",
    )
    pairs["calendar_month"] = pairs["month"].str[-2:]
    seasonal = pairs.groupby(["station_id", "calendar_month"])[
        "mean_soil_moisture_m3_m3"
    ]
    pairs["soil_moisture_anomaly"] = (
        pairs["mean_soil_moisture_m3_m3"] - seasonal.transform("mean")
    ) / seasonal.transform("std")

    correlation_rows = []
    for station_id, group in pairs.dropna(
        subset=["soil_moisture_anomaly", "DROUGHT_INDEX"]
    ).groupby("station_id"):
        station_row = station.loc[station["station_id"] == station_id].iloc[0]
        record = {
            "station_id": station_id,
            "network": station_row["network"],
            "station": station_row["station"],
            "spei_distance_km": station_row["spei_distance_km"],
        }
        for label, subset in (
            ("all", group),
            ("2015_onward", group.loc[group["month"] >= "2015-01"]),
        ):
            valid = subset.dropna(subset=["soil_moisture_anomaly", "DROUGHT_INDEX"])
            rho = np.nan
            if len(valid) >= 24 and valid["soil_moisture_anomaly"].nunique() > 1:
                rho = float(valid["soil_moisture_anomaly"].corr(valid["DROUGHT_INDEX"], method="spearman"))
            record[f"paired_months_{label}"] = int(len(valid))
            record[f"spearman_rho_{label}"] = rho
        correlation_rows.append(record)

    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(output_dir / "ismn_spei_station_correlations.csv", index=False, float_format="%.6f")
    all_rho = correlations["spearman_rho_all"].dropna()
    recent_rho = correlations["spearman_rho_2015_onward"].dropna()
    result = {
        "paired_station_months": int(len(pairs)),
        "stations_with_spei_pairs": int(pairs["station_id"].nunique()),
        "spei_grid_distance_km_median": float(station["spei_distance_km"].median()),
        "spei_grid_distance_km_p90": float(station["spei_distance_km"].quantile(0.9)),
        "stations_with_24_or_more_paired_months": int(correlations["spearman_rho_all"].notna().sum()),
        "median_station_spearman_rho": float(correlations["spearman_rho_all"].median()),
        "share_positive_station_correlations": float((all_rho > 0).mean()),
        "stations_with_24_or_more_paired_months_since_2015": int(
            correlations["spearman_rho_2015_onward"].notna().sum()
        ),
        "median_station_spearman_rho_since_2015": float(
            recent_rho.median()
        ),
        "share_positive_station_correlations_since_2015": float((recent_rho > 0).mean()),
        "correlation_file": "ismn_spei_station_correlations.csv",
        "interpretation": (
            "Concurrent observational consistency only; this does not evaluate out-of-sample model predictions. "
            "Soil-moisture anomalies are standardized within station and calendar month."
        ),
    }
    return result


def analyze(
    archive_path: Path,
    locations_path: Path,
    spei_path: Path,
    output_dir: Path,
    max_distance_km: float,
) -> None:
    locations = pd.read_csv(locations_path).drop_duplicates(subset=["LAT", "LON"]).reset_index(drop=True)
    if "point_id" not in locations:
        locations.insert(0, "point_id", [f"P{index:05d}" for index in range(1, len(locations) + 1)])

    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        series = []
        for name in archive.namelist():
            if not name.endswith(".stm"):
                continue
            metadata = station_header(archive, name)
            point_index, distance_km = nearest_locations(
                metadata["latitude"], metadata["longitude"], locations
            )
            metadata.update(
                {
                    "path": name,
                    "point_id": str(locations.iloc[point_index]["point_id"]),
                    "distance_km": distance_km,
                }
            )
            series.append(metadata)

        shallow = [item for item in series if item["depth_to_m"] <= SHALLOW_DEPTH_M]
        selected = [item for item in shallow if item["distance_km"] <= max_distance_km]
        station_months: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
            lambda: {"sensor_mean_sum": 0.0, "sensor_series_count": 0, "good_observation_count": 0}
        )
        good_observations = 0

        for item in selected:
            sensor_months: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
            with archive.open(str(item["path"])) as stream:
                stream.readline()
                for raw_line in stream:
                    fields = raw_line.decode("utf-8", "replace").split()
                    if len(fields) < 4 or fields[3] != "G":
                        continue
                    try:
                        value = float(fields[2])
                        month = fields[0][:7].replace("/", "-")
                    except ValueError:
                        continue
                    if not math.isfinite(value):
                        continue
                    sensor_months[month][0] += value
                    sensor_months[month][1] += 1
                    good_observations += 1

            station_id = f"{item['network']}::{item['station']}"
            for month, (total, count) in sensor_months.items():
                key = (station_id, str(item["network"]), month)
                record = station_months[key]
                record["sensor_mean_sum"] += total / count
                record["sensor_series_count"] += 1
                record["good_observation_count"] += count

    rows = []
    station_metadata = {}
    for item in selected:
        station_id = f"{item['network']}::{item['station']}"
        station_metadata[station_id] = item
    for (station_id, network, month), values in station_months.items():
        item = station_metadata[station_id]
        rows.append(
            {
                "station_id": station_id,
                "network": network,
                "station": item["station"],
                "latitude": item["latitude"],
                "longitude": item["longitude"],
                "nearest_point_id": item["point_id"],
                "distance_km": item["distance_km"],
                "month": month,
                "mean_soil_moisture_m3_m3": values["sensor_mean_sum"] / values["sensor_series_count"],
                "sensor_series_count": int(values["sensor_series_count"]),
                "good_observation_count": int(values["good_observation_count"]),
            }
        )
    panel = pd.DataFrame(rows).sort_values(["station_id", "month"])
    panel_path = output_dir / "ismn_shallow_station_monthly.csv"
    panel.to_csv(panel_path, index=False, float_format="%.6f")
    consistency = spei_consistency(panel, spei_path, output_dir)

    monthly_2015 = panel.loc[panel["month"] >= "2015-01"]
    months_by_station = monthly_2015.groupby("station_id")["month"].nunique()
    threshold_summary = {}
    for threshold in (10, 25, 50, 100):
        nearby = [item for item in shallow if item["distance_km"] <= threshold]
        threshold_summary[str(threshold)] = {
            "series": len(nearby),
            "stations": len({(item["network"], item["station"]) for item in nearby}),
        }
    summary = {
        "archive": archive_path.name,
        "matched_location_count": int(len(locations)),
        "shallow_definition": f"depth_to_m <= {SHALLOW_DEPTH_M:.2f}",
        "selected_max_distance_km": max_distance_km,
        "selected_series_count": len(selected),
        "selected_station_count": len({(item["network"], item["station"]) for item in selected}),
        "selected_good_quality_observations": good_observations,
        "station_month_rows": int(len(panel)),
        "station_month_rows_2015_onward": int(len(monthly_2015)),
        "stations_with_24_or_more_months_since_2015": int((months_by_station >= 24).sum()),
        "spatial_support_sensitivity": threshold_summary,
        "concurrent_spei_consistency": consistency,
        "panel_file": panel_path.name,
        "quality_flag": "Only standardized ISMN soil-moisture quality flag G retained; network flags preserved in source and not filtered.",
    }
    summary_path = output_dir / "ismn_coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    archives = sorted((PROJECT_ROOT / "data").glob("Data_separate_files_header_*.zip"))
    parser.add_argument("--archive", type=Path, default=archives[-1] if archives else DEFAULT_ARCHIVE)
    parser.add_argument("--locations", type=Path, default=DEFAULT_LOCATIONS)
    parser.add_argument("--spei", type=Path, default=DEFAULT_SPEI)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-distance-km", type=float, default=25.0)
    args = parser.parse_args()
    if args.archive is None or not args.archive.exists():
        parser.error("ISMN archive not found; pass --archive")
    if not args.spei.exists():
        parser.error(f"SPEI benchmark not found: {args.spei}")
    analyze(args.archive, args.locations, args.spei, args.output, args.max_distance_km)


if __name__ == "__main__":
    main()