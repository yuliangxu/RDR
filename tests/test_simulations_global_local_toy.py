"""Standard Bernoulli coordinates rescale FID but preserve mass-based metrics."""
import numpy as np
import pandas as pd

from experiments.AGP import rdr_metric_toy as historical
from experiments.simulations import global_local_toy as toy


def test_bernoulli_moments_and_fid_use_zero_one_support():
    table, regional, _ = toy.example_tables()
    np.testing.assert_array_equal(toy.SUPPORT, [0., 1.])
    np.testing.assert_array_equal(regional.feature_coordinate, [0., 1., 0., 1.])
    expected = np.array([[.5, .2, .25, .16], [.5, .8, .25, .16]])
    np.testing.assert_allclose(
        table[["p_feature_mean", "q_feature_mean", "p_feature_variance", "q_feature_variance"]],
        expected, rtol=0, atol=2e-16,
    )
    # (.5 - theta)^2 + (sqrt(.25) - sqrt(.16))^2 = .09 + .01.
    np.testing.assert_allclose(table.population_gaussian_frechet, .1, rtol=0, atol=2e-16)
    old_table, _, _ = historical.example_tables()
    np.testing.assert_allclose(old_table.population_gaussian_frechet, .4, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(historical.STATES, [-1., 1.])


def test_coordinate_change_preserves_probability_masses_rdr_is_and_prd():
    table, regional, curves = toy.example_tables()
    old_table, old_regional, old_curves = historical.example_tables()
    np.testing.assert_array_equal(regional.p_probability, [.5, .5, .5, .5])
    np.testing.assert_array_equal(regional.q_probability, [.8, .2, .2, .8])
    np.testing.assert_allclose(
        regional.population_region_rdr, [10/13, 10/7, 10/7, 10/13], rtol=0, atol=2e-16,
    )
    pd.testing.assert_frame_equal(
        regional.drop(columns="feature_coordinate"),
        old_regional.drop(columns=["community", "feature_coordinate"]),
    )
    pd.testing.assert_frame_equal(
        table[["generator", "ideal_two_label_is", "population_total_variation"]],
        old_table[["generator", "ideal_two_label_is", "population_total_variation"]],
    )
    pd.testing.assert_frame_equal(curves, old_curves)
