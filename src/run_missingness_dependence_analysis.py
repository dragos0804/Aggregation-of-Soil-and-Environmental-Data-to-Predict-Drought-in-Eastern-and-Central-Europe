"""Audit NASA monthly missingness and predictor-target dependence.

The script deliberately keeps monthly NASA aggregates unfilled until the
chronological split. It reports the effect of the former zero-fill rule,
training-only median imputation with missingness indicators, same-month and
one-month-lag correlations with SPEI, and VIF among NASA predictors.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from statsmodels.stats.outliers_influence import variance_inflation_factor


CHUNK_SIZE = 500_000
DYNAMIC_FEATURES = ["GWETPROF", "GWETROOT", "GWETTOP"]
KEYS = ["YEAR", "MO", "LAT", "LON"]
SPEI_COLUMN = "DROUGHT_INDEX"
START_YEAR = int(os.getenv("DATA_START_YEAR", "1990"))
END_YEAR = int(os.getenv("DATA_END_YEAR", "2024"))


def paths():
    script_dir = Path(__file__).resolve().parent
    root = script_dir.parent
    return {
        "nasa": root / "data" / "intermediary_steps" / f"complete_nasa_power_data_{START_YEAR}_to_{END_YEAR}.csv",
        "fused": root / "data" / f"complete_integrated_dataset_{START_YEAR}_to_{END_YEAR}.csv",
        "spei": root / "data" / "intermediary_steps" / "complete_spei_benchmark.csv",
        "results": root / "results",
    }


def _partial_monthly(chunk, static_features):
    grouped = chunk.groupby(KEYS, sort=False, dropna=False)
    pieces = []

    for feature in DYNAMIC_FEATURES:
        stats = grouped[feature].agg(["sum", "count", "min", "max"])
        stats["sumsq"] = (chunk[feature] * chunk[feature]).groupby(
            [chunk[key] for key in KEYS], sort=False, dropna=False
        ).sum()
        stats = stats.rename(
            columns={
                "sum": f"{feature}__sum",
                "count": f"{feature}__count",
                "min": f"{feature}__min",
                "max": f"{feature}__max",
                "sumsq": f"{feature}__sumsq",
            }
        )
        pieces.append(stats)

    result = pd.concat(pieces, axis=1).reset_index()
    if static_features:
        static = grouped[static_features].first().reset_index()
        result = result.merge(static, on=KEYS, how="left", sort=False)
    return result


def aggregate_monthly(path, include_static=False):
    """Aggregate daily records using mergeable sums/counts, not zero-fill."""
    usecols = KEYS + DYNAMIC_FEATURES
    static_features = []
    if include_static:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        static_features = [column for column in header if column not in usecols + ["DY"]]
        usecols += static_features

    partials = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=CHUNK_SIZE):
        chunk = chunk[chunk["YEAR"].between(START_YEAR, END_YEAR)]
        if chunk.empty:
            continue
        partials.append(_partial_monthly(chunk, static_features))

    combined = pd.concat(partials, ignore_index=True)
    dynamic_columns = [
        f"{feature}__{stat}"
        for feature in DYNAMIC_FEATURES
        for stat in ["sum", "count", "min", "max", "sumsq"]
    ]
    aggregation = {
        column: ("sum" if column.endswith(("__sum", "__count", "__sumsq")) else "min"
                 if column.endswith("__min") else "max")
        for column in dynamic_columns
    }
    monthly = combined.groupby(KEYS, sort=False, dropna=False)[dynamic_columns].agg(aggregation).reset_index()

    if static_features:
        static = combined.groupby(KEYS, sort=False, dropna=False)[static_features].first().reset_index()
        monthly = monthly.merge(static, on=KEYS, how="left", sort=False)

    for feature in DYNAMIC_FEATURES:
        count = monthly[f"{feature}__count"].to_numpy(dtype=float)
        total = monthly[f"{feature}__sum"].to_numpy(dtype=float)
        sumsq = monthly[f"{feature}__sumsq"].to_numpy(dtype=float)
        mean = np.divide(total, count, out=np.full(len(monthly), np.nan), where=count > 0)
        variance = np.divide(
            sumsq - np.divide(total * total, count, out=np.zeros(len(monthly)), where=count > 0),
            count - 1,
            out=np.full(len(monthly), np.nan),
            where=count > 1,
        )
        monthly[f"{feature}_mean"] = mean
        monthly[f"{feature}_min"] = monthly.pop(f"{feature}__min")
        monthly[f"{feature}_max"] = monthly.pop(f"{feature}__max")
        monthly[f"{feature}_std"] = np.sqrt(np.maximum(variance, 0))
        monthly.drop(columns=[f"{feature}__sum", f"{feature}__count", f"{feature}__sumsq"], inplace=True)

    return monthly


def add_date(frame):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(
        frame["YEAR"].astype(str) + "-" + frame["MO"].astype(str) + "-01"
    )
    return frame.sort_values(["date", "LAT", "LON"]).reset_index(drop=True)


def missingness_report(monthly, dataset_name):
    rows = []
    feature_columns = [
        f"{feature}_{stat}" for feature in DYNAMIC_FEATURES for stat in ["mean", "min", "max", "std"]
    ]
    for column in feature_columns:
        missing = int(monthly[column].isna().sum())
        rows.append({
            "dataset": dataset_name,
            "variable_statistic": column,
            "monthly_cells": int(len(monthly)),
            "missing_cells": missing,
            "missing_percent": 100 * missing / len(monthly) if len(monthly) else np.nan,
        })
    return pd.DataFrame(rows)


def imputation_report(monthly, dataset_name):
    frame = add_date(monthly)
    cutoff = frame["date"].sort_values().iloc[max(0, int(len(frame) * 0.8) - 1)]
    train = frame[frame["date"] <= cutoff]
    test = frame[frame["date"] > cutoff]
    rows = []
    feature_columns = [
        f"{feature}_{stat}" for feature in DYNAMIC_FEATURES for stat in ["mean", "min", "max", "std"]
    ]
    for column in feature_columns:
        median = train[column].median()
        rows.append({
            "dataset": dataset_name,
            "variable_statistic": column,
            "training_end": cutoff.date().isoformat(),
            "training_cells": len(train),
            "test_cells": len(test),
            "training_missing": int(train[column].isna().sum()),
            "test_missing": int(test[column].isna().sum()),
            "training_median": median,
            "zero_fill_value": 0.0,
            "median_minus_zero": median,
            "missingness_indicator_added": True,
        })
    return pd.DataFrame(rows)


def merge_spei(monthly, spei):
    monthly_groups = monthly.groupby(["YEAR", "MO"], sort=False)
    spei_groups = spei.groupby(["YEAR", "MO"], sort=False)
    merged = []
    for key, group in monthly_groups:
        try:
            target = spei_groups.get_group(key)
        except KeyError:
            continue
        tree = cKDTree(target[["LAT", "LON"]].to_numpy())
        _, indices = tree.query(group[["LAT", "LON"]].to_numpy(), k=1)
        result = group.reset_index(drop=True).copy()
        result[SPEI_COLUMN] = target.iloc[indices][SPEI_COLUMN].to_numpy()
        merged.append(result)
    return add_date(pd.concat(merged, ignore_index=True))


def pearson(x, y):
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan, int(valid.sum())
    return float(x[valid].corr(y[valid])), int(valid.sum())


def dependence_report(monthly, spei, dataset_name):
    frame = merge_spei(monthly, spei)
    frame = frame.sort_values(["LAT", "LON", "date"]).copy()
    rows = []
    feature_columns = [
        f"{feature}_{stat}" for feature in DYNAMIC_FEATURES for stat in ["mean", "min", "max", "std"]
    ]
    for column in feature_columns:
        same_r, same_n = pearson(frame[column], frame[SPEI_COLUMN])
        prior_month_predictor = frame.groupby(["LAT", "LON"], sort=False)[column].shift(1)
        next_month_r, next_month_n = pearson(prior_month_predictor, frame[SPEI_COLUMN])
        rows.extend([
            {"dataset": dataset_name, "feature": column, "relationship": "same_month", "pearson_r": same_r, "n": same_n},
            {"dataset": dataset_name, "feature": column, "relationship": "predictor_t_to_spei_t_plus_1", "pearson_r": next_month_r, "n": next_month_n},
        ])
    return pd.DataFrame(rows), frame


def vif_report(frame, dataset_name, max_rows=200_000):
    feature_columns = [
        f"{feature}_{stat}" for feature in DYNAMIC_FEATURES for stat in ["mean", "min", "max", "std"]
    ]
    complete = frame[feature_columns].dropna()
    if len(complete) > max_rows:
        complete = complete.sample(max_rows, random_state=42)
    design = np.column_stack([np.ones(len(complete)), complete.to_numpy(dtype=float)])
    rows = []
    for index, column in enumerate(feature_columns, start=1):
        rows.append({
            "dataset": dataset_name,
            "feature": column,
            "vif": variance_inflation_factor(design, index),
            "complete_rows_used": len(complete),
        })
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def write_markdown(frame, path, title):
    columns = [str(column) for column in frame.columns]

    def cell(value):
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"# {title}\n\n")
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in frame.itertuples(index=False, name=None):
            handle.write("| " + " | ".join(cell(value) for value in row) + " |\n")
        handle.write("\n")


def main():
    file_paths = paths()
    file_paths["results"].mkdir(parents=True, exist_ok=True)
    spei = pd.read_csv(file_paths["spei"], usecols=["YEAR", "MO", "LAT", "LON", SPEI_COLUMN])
    spei = spei[spei["YEAR"].between(START_YEAR, END_YEAR)].copy()

    missing_frames = []
    imputation_frames = []
    dependence_frames = []
    vif_frames = []
    for dataset_name, path, include_static in [
        ("NASA-Only", file_paths["nasa"], False),
        ("Aggregated", file_paths["fused"], True),
    ]:
        print(f"Aggregating {dataset_name} from {path.name}...")
        monthly = aggregate_monthly(path, include_static=include_static)
        missing_frames.append(missingness_report(monthly, dataset_name))
        imputation_frames.append(imputation_report(monthly, dataset_name))
        dependence, merged = dependence_report(monthly, spei, dataset_name)
        dependence_frames.append(dependence)
        vif_frames.append(vif_report(merged, dataset_name))

    outputs = {
        "monthly_missingness_report.csv": pd.concat(missing_frames, ignore_index=True),
        "monthly_imputation_sensitivity.csv": pd.concat(imputation_frames, ignore_index=True),
        "predictor_target_dependence.csv": pd.concat(dependence_frames, ignore_index=True),
        "predictor_vif.csv": pd.concat(vif_frames, ignore_index=True),
    }
    for name, frame in outputs.items():
        frame.insert(0, "data_start_year", START_YEAR)
        frame.insert(1, "data_end_year", END_YEAR)
        frame.to_csv(file_paths["results"] / name, index=False)
    write_markdown(outputs["monthly_missingness_report.csv"], file_paths["results"] / "monthly_missingness_report.md", "Monthly NASA Missingness Audit")
    write_markdown(outputs["monthly_imputation_sensitivity.csv"], file_paths["results"] / "monthly_imputation_sensitivity.md", "Chronological Imputation Sensitivity")
    write_markdown(outputs["predictor_target_dependence.csv"], file_paths["results"] / "predictor_target_dependence.md", "Predictor-Target Dependence")
    write_markdown(outputs["predictor_vif.csv"], file_paths["results"] / "predictor_vif.md", "NASA Predictor VIF")
    print(f"Wrote diagnostics to {file_paths['results']}")


if __name__ == "__main__":
    main()