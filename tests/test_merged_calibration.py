"""Independent mathematical checks for data-selected merged calibration cells."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import binom, multinomial

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.calibration import (
    _joint_multiplier_fluctuations,
    bin_scores,
    calibrate_merged_counts,
    contiguous_candidates,
    merge_adjacent_counts,
)


def calibrate(k_p, k_q, **kwargs):
    """Small bootstrap budget for checks unrelated to Monte Carlo precision."""
    kwargs.setdefault("bootstrap_repetitions", 17)
    return calibrate_merged_counts(k_p, k_q, sum(k_p), sum(k_q), **kwargs)


class MergedCalibrationTests(unittest.TestCase):
    def test_candidate_family_contains_every_contiguous_union(self):
        start, stop = contiguous_candidates(4)
        expected = [(i, j) for i in range(4) for j in range(i + 1, 5)]
        self.assertEqual(list(zip(start, stop)), expected)
        result = calibrate([1, 2, 3, 4], [4, 2, 1, 3])
        candidates = result["candidates"]
        assert_array_equal(candidates["start"], start)
        assert_array_equal(candidates["stop"], stop)
        for index, (left, right) in enumerate(expected):
            self.assertEqual(candidates["k_p"][index], sum([1, 2, 3, 4][left:right]))
            self.assertEqual(candidates["k_q"][index], sum([4, 2, 1, 3][left:right]))
        self.assertEqual(result["metadata"]["n_candidates"], 10)

    def test_greedy_rule_requires_each_count_and_absorbs_the_tail(self):
        fixtures = [
            # A pooled count of two in the first cell is insufficient.
            ([2, 0, 2, 1], [0, 2, 2, 0], 2, [[0, 2], [2, 4]]),
            # Neither distribution can meet the threshold anywhere.
            ([1, 0, 0], [0, 0, 1], 2, [[0, 3]]),
            # Every elementary cell already qualifies.
            ([2, 3, 4], [2, 2, 2], 2, [[0, 1], [1, 2], [2, 3]]),
            # Even a completely empty right tail must stay in the partition.
            ([2, 2, 0], [2, 2, 0], 2, [[0, 1], [1, 3]]),
            ([2, 0, 0], [2, 0, 0], 2, [[0, 3]]),
            ([2], [3], 2, [[0, 1]]),
        ]
        for k_p, k_q, threshold, expected in fixtures:
            with self.subTest(k_p=k_p, k_q=k_q, threshold=threshold):
                assert_array_equal(merge_adjacent_counts(k_p, k_q, threshold), expected)
                result = calibrate(k_p, k_q, min_count=threshold)
                selected = result["selected_indices"]
                candidates = result["candidates"]
                assert_array_equal(
                    np.column_stack((candidates["start"][selected], candidates["stop"][selected])),
                    expected,
                )
                assert_array_equal(np.flatnonzero(candidates["selected"]), selected)

    def test_unequal_sample_sizes_and_analytic_delta_variance(self):
        result = calibrate([20, 80], [60, 140], min_count=20)
        candidates = result["candidates"]
        # For [0,1), p=.2, q=.3, r=.8; unequal n,m must stay normalized.
        self.assertAlmostEqual(candidates["estimate"][0], 0.8)
        self.assertAlmostEqual(candidates["a"][0], 2.4)
        self.assertAlmostEqual(candidates["b"][0], -1.6)
        self.assertAlmostEqual(candidates["se"][0] ** 2, 0.011904)
        radius = result["metadata"]["critical_value"] * np.sqrt(0.011904)
        self.assertAlmostEqual(candidates["c1_lower"][0], max(0, 0.8 - radius))
        self.assertAlmostEqual(candidates["c1_upper"][0], min(2, 0.8 + radius))
        self.assertNotAlmostEqual(candidates["estimate"][0], 2 * 20 / (20 + 60))

    def test_cp_endpoints_invert_binomial_tails_for_original_family(self):
        alpha = 0.1
        result = calibrate([0, 1, 4, 5], [2, 3, 5, 10], alpha=alpha, min_count=100)
        candidates = result["candidates"]
        self.assertEqual(len(result["selected_indices"]), 1)
        # The error allocation still protects all ten candidates, not one.
        tail = alpha / (4 * 10)
        self.assertEqual(result["metadata"]["c2_tail_probability"], tail)
        for prefix, size in [("p", 10), ("q", 20)]:
            counts = candidates["k_" + prefix]
            positive, nonfull = counts > 0, counts < size
            assert_allclose(
                binom.sf(counts[positive] - 1, size, candidates[prefix + "_lower"][positive]),
                tail, rtol=2e-10, atol=1e-13,
            )
            assert_allclose(
                binom.cdf(counts[nonfull], size, candidates[prefix + "_upper"][nonfull]),
                tail, rtol=2e-10, atol=1e-13,
            )
            assert_array_equal(candidates[prefix + "_lower"][counts == 0], 0)
            assert_array_equal(candidates[prefix + "_upper"][counts == size], 1)
        nonwhole = ~candidates["whole_space"]
        assert_allclose(
            candidates["c2_lower"][nonwhole],
            (2 * candidates["p_lower"] / (candidates["p_lower"] + candidates["q_upper"]))[nonwhole],
        )
        assert_allclose(
            candidates["c2_upper"][nonwhole],
            (2 * candidates["p_upper"] / (candidates["p_upper"] + candidates["q_lower"]))[nonwhole],
        )

    def test_exact_enumerated_coverage_after_data_dependent_selection(self):
        # Enumerate all 150 pairs of three-cell multinomial count outcomes.
        # No Monte Carlo coverage estimate or bootstrap approximation enters C.2.
        n, m, alpha = 3, 4, 0.2
        outcomes_p = [(a, b, n - a - b) for a in range(n + 1) for b in range(n - a + 1)]
        outcomes_q = [(a, b, m - a - b) for a in range(m + 1) for b in range(m - a + 1)]
        records = []
        selected_partitions = set()
        for k_p in outcomes_p:
            for k_q in outcomes_q:
                result = calibrate(k_p, k_q, alpha=alpha, min_count=1, bootstrap_repetitions=2)
                candidates = result["candidates"]
                selected = result["selected_indices"]
                selected_partitions.add(tuple(selected))
                records.append((k_p, k_q, candidates, selected))
        self.assertGreaterEqual(len(selected_partitions), 4)
        probabilities = [[0.02, 0.08, 0.9], [0.4, 0.2, 0.4], [0.8, 0.15, 0.05], [0.2, 0.6, 0.2]]
        for p in probabilities:
            for q in probabilities:
                covered_probability = all_covered_probability = total_probability = 0.0
                for k_p, k_q, candidates, selected in records:
                    mass = multinomial.pmf(k_p, n, p) * multinomial.pmf(k_q, m, q)
                    ph = np.array([sum(p[a:b]) for a, b in zip(candidates["start"], candidates["stop"])])
                    qh = np.array([sum(q[a:b]) for a, b in zip(candidates["start"], candidates["stop"])])
                    target = 2 * ph / (ph + qh)
                    # Whole-space target is analytically 1, avoiding summation roundoff.
                    target[candidates["whole_space"]] = 1
                    covered = (candidates["c2_lower"] <= target) & (target <= candidates["c2_upper"])
                    total_probability += mass
                    covered_probability += mass * np.all(covered[selected])
                    all_covered_probability += mass * np.all(covered)
                with self.subTest(p=p, q=q):
                    self.assertAlmostEqual(total_probability, 1)
                    self.assertGreaterEqual(all_covered_probability, 1 - alpha - 1e-12)
                    self.assertGreaterEqual(covered_probability, all_covered_probability - 1e-12)

    def test_grouped_multipliers_equal_observation_formula_and_joint_covariance(self):
        k_p, k_q = np.array([1, 3, 1]), np.array([3, 0, 4])
        n, m = int(sum(k_p)), int(sum(k_q))
        result = calibrate(k_p, k_q)
        candidates = result["candidates"]
        start, stop = candidates["start"], candidates["stop"]
        p_labels, q_labels = np.repeat(np.arange(3), k_p), np.repeat(np.arange(3), k_q)
        p_indicator = (p_labels[:, None] >= start) & (p_labels[:, None] < stop)
        q_indicator = (q_labels[:, None] >= start) & (q_labels[:, None] < stop)
        p, q = p_indicator.mean(axis=0), q_indicator.mean(axis=0)
        a, b = 2 * q / (p + q) ** 2, -2 * p / (p + q) ** 2
        # Each basis vector activates one original-observation multiplier.
        # Their outer products sum to the exact Gaussian covariance matrix.
        multipliers = np.eye(n + m)
        p_group_sums = multipliers[:, :n] @ np.eye(3)[p_labels]
        q_group_sums = multipliers[:, n:] @ np.eye(3)[q_labels]
        actual = _joint_multiplier_fluctuations(
            p_group_sums, q_group_sums, start, stop, p, q, a, b, n, m,
        )
        expected = (
            multipliers[:, :n] @ (p_indicator - p) * (a / n)
            + multipliers[:, n:] @ (q_indicator - q) * (b / m)
        )
        assert_allclose(actual, expected, atol=1e-15)
        p_joint = p_indicator.astype(float).T @ p_indicator / n
        q_joint = q_indicator.astype(float).T @ q_indicator / m
        covariance = (
            np.outer(a, a) * (p_joint - np.outer(p, p)) / n
            + np.outer(b, b) * (q_joint - np.outer(q, q)) / m
        )
        assert_allclose(actual.T @ actual, covariance, atol=1e-15)
        assert_allclose(np.diag(covariance), candidates["se"] ** 2, atol=1e-15)
        # Nonzero off-diagonal terms are lost with independent candidate draws.
        self.assertGreater(np.max(np.abs(covariance - np.diag(np.diag(covariance)))), 0.01)

    def test_bootstrap_maximum_protects_unselected_candidates_and_is_reproducible(self):
        k_p, k_q = np.array([2, 3, 1, 4]), np.array([3, 1, 5, 2])
        n, m, repetitions, seed, alpha = 10, 11, 41, 123, 0.1
        result = calibrate(k_p, k_q, min_count=20, seed=seed, alpha=alpha, bootstrap_repetitions=repetitions)
        candidates = result["candidates"]
        self.assertEqual(len(result["selected_indices"]), 1)
        self.assertTrue(candidates["whole_space"][result["selected_indices"][0]])
        rng = np.random.default_rng(seed)
        p_sums = rng.standard_normal((repetitions, 4)) * np.sqrt(k_p)
        q_sums = rng.standard_normal((repetitions, 4)) * np.sqrt(k_q)
        standardized = []
        for left in range(4):
            for right in range(left + 1, 5):
                p, q = sum(k_p[left:right]) / n, sum(k_q[left:right]) / m
                a, b = 2 * q / (p + q) ** 2, -2 * p / (p + q) ** 2
                se = np.sqrt(a * a * p * (1 - p) / n + b * b * q * (1 - q) / m)
                if se == 0:
                    continue
                fluctuation = (
                    a / n * (p_sums[:, left:right].sum(axis=1) - p * p_sums.sum(axis=1))
                    + b / m * (q_sums[:, left:right].sum(axis=1) - q * q_sums.sum(axis=1))
                )
                standardized.append(np.abs(fluctuation) / se)
        expected_maxima = np.max(standardized, axis=0)
        assert_allclose(result["bootstrap_maxima"], expected_maxima, atol=1e-14)
        self.assertAlmostEqual(result["metadata"]["critical_value"], np.quantile(expected_maxima, 1 - alpha, method="higher"))
        self.assertGreater(result["metadata"]["critical_value"], 0)
        other_threshold = calibrate(k_p, k_q, min_count=1, seed=seed, alpha=alpha, bootstrap_repetitions=repetitions)
        assert_array_equal(result["bootstrap_maxima"], other_threshold["bootstrap_maxima"])
        for name in ["c1_lower", "c1_upper", "c2_lower", "c2_upper"]:
            assert_array_equal(candidates[name], other_threshold["candidates"][name])
        different_seed = calibrate(k_p, k_q, min_count=20, seed=124, alpha=alpha, bootstrap_repetitions=repetitions)
        self.assertFalse(np.array_equal(result["bootstrap_maxima"], different_seed["bootstrap_maxima"]))

    def test_empty_and_zero_variance_fallback_is_distinct_from_whole_space(self):
        result = calibrate([3, 0, 0], [5, 0, 0])
        candidates = result["candidates"]
        whole = candidates["whole_space"]
        self.assertEqual(np.count_nonzero(whole), 1)
        assert_array_equal(candidates["se"], 0)
        assert_array_equal(candidates["c1_unavailable"], ~whole)
        assert_array_equal(candidates["c1_lower"][~whole], 0)
        assert_array_equal(candidates["c1_upper"][~whole], 2)
        for name in ["estimate", "c1_lower", "c1_upper", "c2_lower", "c2_upper"]:
            assert_array_equal(candidates[name][whole], 1)
        empty = candidates["k_p"] + candidates["k_q"] == 0
        assert_array_equal(candidates["c1_empty"], empty)
        assert_array_equal(candidates["c2_lower"][empty], 0)
        assert_array_equal(candidates["c2_upper"][empty], 2)
        assert_array_equal(result["bootstrap_maxima"], 0)
        self.assertEqual(result["metadata"]["critical_value"], 0)
        one_sided = calibrate([3, 0], [0, 5])["candidates"]
        assert_array_equal(one_sided["estimate"], [2, 1, 0])
        assert_array_equal(one_sided["c1_unavailable"], [True, False, True])

    def test_query_mapping_includes_boundaries_and_right_endpoint(self):
        result = calibrate([2, 0, 2, 1], [0, 2, 2, 0], min_count=2)
        assert_array_equal(result["elementary_to_merged"], [0, 0, 1, 1])
        edges = [0, 0.5, 1, 1.5, 2]
        scores = [0, 0.5, np.nextafter(1, 0), 1, 1.5, 2]
        elementary = bin_scores(scores, edges)
        merged = result["elementary_to_merged"][elementary]
        assert_array_equal(merged, [0, 0, 0, 1, 1, 1])
        candidate_indices = result["selected_indices"][merged]
        candidates = result["candidates"]
        self.assertTrue(np.all(candidates["start"][candidate_indices] <= elementary))
        self.assertTrue(np.all(elementary < candidates["stop"][candidate_indices]))

    def test_invalid_inputs_are_rejected(self):
        base = dict(k_p=[1, 2], k_q=[2, 1], n=3, m=3, bootstrap_repetitions=2)
        invalid = [
            {"k_p": []}, {"k_p": [1, 1]}, {"k_p": [1.5, 1.5]},
            {"k_p": [-1, 4]}, {"k_p": [np.nan, 3]}, {"k_p": [np.inf, 0]},
            {"k_p": [[1, 2]]}, {"k_q": [1, 1, 1]}, {"k_p": [True, True]},
            {"n": 0}, {"m": -1}, {"n": 3.0}, {"n": True},
            {"alpha": 0}, {"alpha": 1}, {"alpha": np.nan}, {"alpha": [0.05]},
            {"min_count": 0}, {"min_count": -1}, {"min_count": 1.5}, {"min_count": True},
            {"bootstrap_repetitions": 0}, {"bootstrap_repetitions": 1.5},
            {"bootstrap_repetitions": True},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                calibrate_merged_counts(**(base | changes))


if __name__ == "__main__":
    unittest.main()
