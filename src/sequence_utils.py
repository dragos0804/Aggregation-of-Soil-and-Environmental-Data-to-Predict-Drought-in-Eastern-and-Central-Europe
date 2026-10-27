"""Utilities for constructing location-specific monthly sequences."""

import numpy as np


def create_location_sequences(
    features,
    targets,
    dates,
    latitudes,
    longitudes,
    sequence_length,
):
    """Create next-month sequences without crossing locations or month gaps."""
    features = np.asarray(features)
    targets = np.asarray(targets)
    dates = np.asarray(dates, dtype="datetime64[M]")
    latitudes = np.asarray(latitudes)
    longitudes = np.asarray(longitudes)

    row_count = len(features)
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if not all(len(values) == row_count for values in (targets, dates, latitudes, longitudes)):
        raise ValueError("features, targets, dates, latitudes, and longitudes must have equal length")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")

    locations = {}
    for row_index, location in enumerate(zip(latitudes, longitudes)):
        locations.setdefault(location, []).append(row_index)

    sequence_features = []
    sequence_targets = []
    input_dates = []
    target_dates = []
    target_latitudes = []
    target_longitudes = []

    for (latitude, longitude), location_indices in locations.items():
        ordered_indices = sorted(location_indices, key=lambda index: dates[index])
        for start in range(len(ordered_indices) - sequence_length):
            window_indices = ordered_indices[start:start + sequence_length]
            target_index = ordered_indices[start + sequence_length]
            window_dates = dates[window_indices]
            target_date = dates[target_index]
            if not np.all(np.diff(window_dates) == np.timedelta64(1, "M")):
                continue
            if target_date - window_dates[-1] != np.timedelta64(1, "M"):
                continue
            sequence_features.append(features[window_indices])
            sequence_targets.append(targets[target_index])
            input_dates.append(window_dates)
            target_dates.append(target_date)
            target_latitudes.append(latitude)
            target_longitudes.append(longitude)

    if not sequence_features:
        return {
            "features": np.empty((0, sequence_length, features.shape[1]), dtype=features.dtype),
            "targets": np.empty((0,), dtype=targets.dtype),
            "input_dates": np.empty((0, sequence_length), dtype="datetime64[M]"),
            "target_dates": np.empty((0,), dtype="datetime64[M]"),
            "target_latitudes": np.empty((0,), dtype=latitudes.dtype),
            "target_longitudes": np.empty((0,), dtype=longitudes.dtype),
        }

    return {
        "features": np.asarray(sequence_features),
        "targets": np.asarray(sequence_targets),
        "input_dates": np.asarray(input_dates, dtype="datetime64[M]"),
        "target_dates": np.asarray(target_dates, dtype="datetime64[M]"),
        "target_latitudes": np.asarray(target_latitudes),
        "target_longitudes": np.asarray(target_longitudes),
    }