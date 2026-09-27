"""Boundary, extremal-distribution, and coverage checks for variance bounds."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import binom

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.calibration import calibrate_counts, calibrate_merged_counts, variance_upper_bound


class VarianceBoundTests(unittest.TestCase):
    def test_boundaries_and_intervals_containing_one(self):
        lower = [0, 2, 1, 0, 0.3, 0.3, 1]
        upper = [0, 2, 1, 2, 1.7, 1, 1.7]
        assert_array_equal(variance_upper_bound(lower, upper), [0, 0, 1, 1, 1, 1, 1])

    def test_intervals_on_either_side_use_endpoint_nearest_one(self):
        # At mean .4, a variable taking 0 or 2 has variance .64.
        # Its reflection has mean 1.6 and the same variance.
        assert_allclose(variance_upper_bound([0.1, 1.6], [0.4, 1.9]), [0.64, 0.64])
        self.assertAlmostEqual(variance_upper_bound(0.2, 0.2), 0.36)

    def test_broadcasting_and_inputs_remain_unchanged(self):
        lower = np.array([[0.0], [0.2]])
        upper = np.array([0.4, 0.8, 1.5])
        before_lower, before_upper = lower.copy(), upper.copy()
        assert_allclose(variance_upper_bound(lower, upper), [[0.64, 0.96, 1], [0.64, 0.96, 1]])
        assert_array_equal(lower, before_lower)
        assert_array_equal(upper, before_upper)
        assert_allclose(variance_upper_bound(0, [0.4, 0.8]), [0.64, 0.96])

    def test_invalid_intervals_are_rejected(self):
        invalid = [
            (-0.01, 1), (1, 2.01), (1.1, 0.9),
            (np.nan, 1), (0, np.nan), (-np.inf, 1), (0, np.inf),
            ([0, 0], [1, np.nan]), ([0, 1], [1, 1, 2]),
        ]
        for lower, upper in invalid:
            with self.subTest(lower=lower, upper=upper), self.assertRaises(ValueError):
                variance_upper_bound(lower, upper)

    def test_mean_range_bound_is_sharp_for_endpoint_distributions(self):
        # Probability theta/2 at 2 and the remainder at 0 fixes mean theta.
        # These are realizable true-RDR laws for disjoint P/Q support in a cell.
        means = np.array([0, 0.02, 0.4, 1, 1.6, 1.98, 2])
        probability_two = means / 2
        variance = probability_two * (2 - means) ** 2 + (1 - probability_two) * means**2
        assert_allclose(variance_upper_bound(means, means), variance, atol=2e-16)

    def test_zero_mean_uncertainty_does_not_remove_true_variance(self):
        # Disjoint P/Q support gives true RDR 0 or 2 with equal mixture mass.
        # The whole-space mean is nevertheless known exactly to equal 1.
        result = calibrate_merged_counts(
            [4, 0], [0, 4], 4, 4, min_count=5, bootstrap_repetitions=7
        )
        selected = result["selected_indices"]
        self.assertEqual(len(selected), 1)
        for method in ("c1", "c2"):
            lower = result["candidates"][method + "_lower"][selected]
            upper = result["candidates"][method + "_upper"][selected]
            assert_array_equal(lower, [1])
            assert_array_equal(upper, [1])
            assert_array_equal(variance_upper_bound(lower, upper), [1])

    def test_exact_enumerated_c2_coverage_is_inherited(self):
        # Enumerate all possible calibration outcomes, without Monte Carlo.
        # Each cell has disjoint P/Q supports, attaining the largest possible
        # conditional variance. On every mean-coverage event, both variance
        # bounds must cover jointly; their probability must be at least 1-alpha.
        n, m, alpha = 8, 13, 0.2
        x, y = np.meshgrid(np.arange(n + 1), np.arange(m + 1), indexing="ij")
        result = calibrate_counts(
            np.stack((x, n - x), axis=-1),
            np.stack((y, m - y), axis=-1), n, m, alpha,
        )
        bound = variance_upper_bound(result["c2_lower"], result["c2_upper"])
        for p in [0.01, 0.15, 0.5, 0.9, 0.99]:
            for q in [0.01, 0.15, 0.5, 0.9, 0.99]:
                means = 2 * np.array([p, 1 - p]) / np.array([p + q, 2 - p - q])
                probability_two = means / 2
                variance = (
                    probability_two * (2 - means) ** 2
                    + (1 - probability_two) * means**2
                )
                mean_event = np.all(
                    (result["c2_lower"] <= means) & (means <= result["c2_upper"]), axis=-1
                )
                variance_event = np.all(variance <= bound + 2e-15, axis=-1)
                self.assertTrue(np.all(~mean_event | variance_event))
                probability = binom.pmf(x, n, p) * binom.pmf(y, m, q)
                self.assertGreaterEqual(
                    np.sum(probability * variance_event), 1 - alpha - 1e-12
                )


if __name__ == "__main__":
    unittest.main()
