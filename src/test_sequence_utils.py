import unittest

import numpy as np

from sequence_utils import create_location_sequences


class LocationSequenceTests(unittest.TestCase):
    def test_sequences_are_location_safe_and_next_month(self):
        dates = np.array(
            [
                "2020-01", "2020-01", "2020-02", "2020-02",
                "2020-03", "2020-03", "2020-05", "2020-06",
            ],
            dtype="datetime64[M]",
        )
        latitudes = np.array([1, 2, 1, 2, 1, 2, 1, 1], dtype=float)
        longitudes = np.ones(len(dates))
        features = np.arange(len(dates), dtype=float).reshape(-1, 1)
        targets = np.arange(len(dates))

        sequences = create_location_sequences(
            features,
            targets,
            dates,
            latitudes,
            longitudes,
            sequence_length=2,
        )

        self.assertEqual(sequences["features"].shape, (2, 2, 1))
        np.testing.assert_array_equal(sequences["target_latitudes"], [1, 2])
        np.testing.assert_array_equal(
            sequences["input_dates"],
            np.array([["2020-01", "2020-02"], ["2020-01", "2020-02"]], dtype="datetime64[M]"),
        )
        np.testing.assert_array_equal(
            sequences["target_dates"],
            np.array(["2020-03", "2020-03"], dtype="datetime64[M]"),
        )


if __name__ == "__main__":
    unittest.main()