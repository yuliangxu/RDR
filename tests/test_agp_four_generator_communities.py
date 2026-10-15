"""Mathematical checks for comparing source-normalized shared communities."""
import numpy as np
import pytest

from experiments.AGP.four_generator_communities import region_summary


def test_unequal_source_sizes_require_conditional_mixture_weights():
    p = np.tile([[1.8], [.4]], (1, 5))
    q = np.tile([[.6], [.2], [.2], [.2], [.2], [.2]], (1, 5))
    row = region_summary(p, q, np.array([0, 1]), np.array([0, 1, 1, 1, 1, 1]), 0, "method", "C01", "rule")
    assert row["p_fraction"] == .5
    assert row["q_fraction"] == pytest.approx(1 / 6)
    assert row["empirical_region_rdr"] == pytest.approx(1.5)
    assert row["mean_fitted_rdr"] == pytest.approx(1.5)
    assert row["mean_fitted_rdr"] != pytest.approx((1.8 + .6) / 2)


def test_regions_partition_sources_and_recover_global_mixture_mean():
    rng = np.random.default_rng(1)
    p, q = rng.uniform(0, 2, (3, 5)), rng.uniform(0, 2, (7, 5))
    pl, ql = np.array([0, 0, 1]), np.array([0, 1, 1, 1, 2, 2, 2])
    rows = [region_summary(p, q, pl, ql, leaf, "method", str(leaf), "rule") for leaf in range(3)]
    assert sum(row["p_count"] for row in rows) == 3
    assert sum(row["q_count"] for row in rows) == 7
    assert sum(row["balanced_mass"] for row in rows) == pytest.approx(1)
    assert sum(row["balanced_mass"] * row["empirical_region_rdr"] for row in rows) == pytest.approx(1)
    assert sum(row["balanced_mass"] * row["mean_fitted_rdr"] for row in rows) == pytest.approx(.5 * (p.mean() + q.mean()))
    assert rows[2]["empirical_region_rdr"] == 0
    assert rows[2]["mean_real_rdr"] is None


def test_real_only_region_and_empty_region_are_distinguished():
    p, q = np.full((3, 5), 1.8), np.full((7, 5), .2)
    row = region_summary(p, q, np.zeros(3), np.ones(7), 0, "method", "C01", "rule")
    assert row["empirical_region_rdr"] == 2
    assert row["mean_fitted_rdr"] == pytest.approx(1.8)
    assert row["mean_generated_rdr"] is None
    with pytest.raises(ValueError, match="Empty assessment region"):
        region_summary(p, q, np.zeros(3), np.ones(7), 2, "method", "C03", "rule")
