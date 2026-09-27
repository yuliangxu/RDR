"""Numerical and exact finite-sample checks for the calibration algorithms."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import binom, norm

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.calibration import bin_scores, calibrate_counts


class CalibrationTests(unittest.TestCase):
    def test_unequal_sample_sizes_and_delta_method(self):
        result = calibrate_counts([20, 80], [60, 140], n=100, m=200)
        # Hand calculation for p=.2, q=.3: r=.8 and SE^2=.011904.
        self.assertAlmostEqual(result["estimate"][0], 0.8)
        self.assertAlmostEqual(result["se"][0] ** 2, 0.011904)
        radius = norm.ppf(0.975) * np.sqrt(0.011904)
        assert_allclose(result["c1_lower"][0], 0.8 - radius)
        assert_allclose(result["c1_upper"][0], 0.8 + radius)
        self.assertNotAlmostEqual(result["estimate"][0], 2 * 20 / (20 + 60))

    def test_empty_boundary_and_full_cells(self):
        result = calibrate_counts([0, 0, 10, 0], [0, 7, 0, 3], 10, 10)
        assert_array_equal(result["estimate"], [1, 0, 2, 0])
        assert_array_equal(result["se"], 0)
        assert_array_equal(result["c1_empty"], [True, False, False, False])
        assert_array_equal(result["c1_unavailable"], True)
        assert_array_equal(result["c1_lower"], 0)
        assert_array_equal(result["c1_upper"], 2)
        self.assertEqual(result["c2_lower"][0], 0)
        self.assertEqual(result["c2_upper"][0], 2)
        full = calibrate_counts([10], [20], 10, 20)
        self.assertEqual(full["estimate"][0], 1)
        self.assertTrue(full["c1_unavailable"][0])
        self.assertEqual(full["c1_lower"][0], 0)
        self.assertEqual(full["c1_upper"][0], 2)

    def test_cp_limits_solve_binomial_tail_equations(self):
        result = calibrate_counts([0, 1, 4, 5], [2, 3, 5, 10], 10, 20, alpha=0.1)
        tail = 0.1 / (4 * 4)
        for prefix, counts, size in [("p", np.array([0, 1, 4, 5]), 10),
                                     ("q", np.array([2, 3, 5, 10]), 20)]:
            positive = counts > 0
            nonfull = counts < size
            # Independent verification via binomial CDF/SF inversion.
            assert_allclose(
                binom.sf(counts[positive] - 1, size, result[prefix + "_lower"][positive]),
                tail, rtol=1e-10, atol=1e-13,
            )
            assert_allclose(
                binom.cdf(counts[nonfull], size, result[prefix + "_upper"][nonfull]),
                tail, rtol=1e-10, atol=1e-13,
            )
        self.assertEqual(result["p_lower"][0], 0)
        self.assertAlmostEqual(result["p_upper"][0], 1 - tail ** (1 / 10))
        full = calibrate_counts([10, 0], [20, 0], 10, 20, alpha=0.1)
        self.assertEqual(full["p_upper"][0], 1)
        self.assertAlmostEqual(full["p_lower"][0], (0.1 / 8) ** (1 / 10))

    def test_prespecified_empty_bins_increase_c2_band_width(self):
        two = calibrate_counts([4, 6], [7, 3], 10, 10)
        three = calibrate_counts([4, 6, 0], [7, 3, 0], 10, 10)
        assert_allclose(two["c1_lower"], three["c1_lower"][:2])
        assert_allclose(two["c1_upper"], three["c1_upper"][:2])
        self.assertTrue(np.all(three["c2_lower"][:2] < two["c2_lower"]))
        self.assertTrue(np.all(three["c2_upper"][:2] > two["c2_upper"]))

    def test_vectorized_replicates_match_separate_calls(self):
        k_p = np.array([[3, 7, 0], [5, 2, 3]])
        k_q = np.array([[7, 10, 3], [3, 8, 9]])
        vectorized = calibrate_counts(k_p, k_q, 10, 20)
        for i in range(2):
            separate = calibrate_counts(k_p[i], k_q[i], 10, 20)
            for key in separate:
                assert_allclose(vectorized[key][i], separate[key])

    def test_exact_enumerated_simultaneous_coverage(self):
        # Enumerate every pair of independent two-bin multinomial outcomes.
        # This checks the finite-sample guarantee without Monte Carlo error.
        n, m, alpha = 8, 13, 0.2
        x, y = np.meshgrid(np.arange(n + 1), np.arange(m + 1), indexing="ij")
        counts_p = np.stack((x, n - x), axis=-1)
        counts_q = np.stack((y, m - y), axis=-1)
        result = calibrate_counts(counts_p, counts_q, n, m, alpha)
        for p in [0.01, 0.15, 0.5, 0.9, 0.99]:
            for q in [0.01, 0.15, 0.5, 0.9, 0.99]:
                target = 2 * np.array([p, 1 - p]) / np.array([p + q, 2 - p - q])
                covered = np.all(
                    (result["c2_lower"] <= target) & (target <= result["c2_upper"]),
                    axis=-1,
                )
                probability = binom.pmf(x, n, p) * binom.pmf(y, m, q)
                self.assertGreaterEqual(np.sum(probability * covered), 1 - alpha - 1e-12)

    def test_bin_boundaries_are_a_partition(self):
        edges = [0, 0.5, 1, 1.5, 2]
        scores = [[0, 0.5, 1], [1.5, 2, np.nextafter(0.5, 0)]]
        assert_array_equal(bin_scores(scores, edges), [[0, 1, 2], [3, 3, 0]])
        self.assertEqual(bin_scores(2, edges), 3)
        self.assertEqual(bin_scores([], edges).size, 0)
        for bad_scores in [[-0.01], [2.01], [np.nan], [np.inf]]:
            with self.assertRaises(ValueError):
                bin_scores(bad_scores, edges)
        for bad_edges in [[0, 1], [1, 2], [0, 1, 1, 2], [0, 2, 1], [0, np.nan, 2]]:
            with self.assertRaises(ValueError):
                bin_scores([1], bad_edges)

    def test_invalid_counts_and_inputs_are_rejected(self):
        invalid = [
            ([1, 8], [2, 8], 10, 10, 0.05),  # incomplete partition
            ([1.5, 8.5], [2, 8], 10, 10, 0.05),
            ([-1, 11], [2, 8], 10, 10, 0.05),
            ([np.nan, 10], [2, 8], 10, 10, 0.05),
            ([np.inf, 0], [2, 8], 10, 10, 0.05),
            ([1, 9], [[2, 8]], 10, 10, 0.05),
            ([], [], 10, 10, 0.05),
            (10, 10, 10, 10, 0.05),
            ([1, 9], [2, 8], 0, 10, 0.05),
            ([1, 9], [2, 8], 10, -1, 0.05),
            ([1, 9], [2, 8], 10.0, 10, 0.05),
            ([1, 9], [2, 8], True, 10, 0.05),
            ([1, 9], [2, 8], 10, 10, 0),
            ([1, 9], [2, 8], 10, 10, 1),
            ([1, 9], [2, 8], 10, 10, np.nan),
            ([1, 9], [2, 8], 10, 10, [0.05]),
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                calibrate_counts(*arguments)


if __name__ == "__main__":
    unittest.main()
