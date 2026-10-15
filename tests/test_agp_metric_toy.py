"""Analytic counterexamples audit location information in global summaries."""
import numpy as np
import pytest

from experiments.AGP import rdr_metric_toy as toy


def test_exact_gaussian_population_distance_and_ideal_classifier_score():
    global_table, regions, _ = toy.example_tables()
    np.testing.assert_allclose(global_table.population_gaussian_frechet, 2/5, atol=2e-16, rtol=0)
    np.testing.assert_allclose(global_table.ideal_two_label_is, 5/(4**.8), atol=3e-16, rtol=0)
    np.testing.assert_allclose(regions.population_region_rdr, [10/13, 10/7, 10/7, 10/13], atol=2e-16, rtol=0)
    assert toy.ideal_label_is([1., 0.]) == 1.
    assert toy.ideal_label_is([.5, .5]) == 2.


def test_prd_curve_matches_independent_piecewise_formula_at_breakpoints_and_between():
    slopes = np.r_[np.logspace(-8, 8, 81), .4, 1.6]
    # For lambda<=.4, both Q masses exceed lambda P; above1.6 neither does.
    expected_alpha = np.where(slopes <= .4, slopes,
                              np.where(slopes <= 1.6, .2+.5*slopes, 1.))
    for q in [toy.Q["Q1"], toy.Q["Q2"]]:
        alpha, beta = toy.analytic_prd(toy.P, q, slopes)
        np.testing.assert_allclose(alpha, expected_alpha, rtol=0, atol=2e-16)
        np.testing.assert_allclose(beta, expected_alpha/slopes, rtol=0, atol=2e-16)


def test_regional_rdr_is_density_ratio_and_identifies_opposite_underrepresentation():
    for q in toy.Q.values():
        mixture = (toy.P+q)/2
        rdr = toy.P/mixture
        np.testing.assert_allclose(mixture @ rdr, 1., rtol=0, atol=2e-16)
        np.testing.assert_array_equal(rdr > 1, q < toy.P)
    np.testing.assert_array_equal(toy.P/((toy.P+toy.Q["Q1"])/2),
                                  (toy.P/((toy.P+toy.Q["Q2"])/2))[::-1])


@pytest.mark.parametrize("invalid", [[.2, .2], [-.1, 1.1], [np.nan, 1.], [[.5, .5]], []])
def test_invalid_probabilities_fail_explicitly(invalid):
    with pytest.raises(ValueError):
        toy.probabilities(invalid)


def test_null_and_disjoint_support_prd_have_expected_limits():
    slopes = np.array([.1, 1., 10.])
    alpha, beta = toy.analytic_prd([.5, .5], [.5, .5], slopes)
    np.testing.assert_allclose(alpha, [.1, 1., 1.])
    np.testing.assert_allclose(beta, [1., 1., .1])
    alpha, beta = toy.analytic_prd([1., 0.], [0., 1.], slopes)
    np.testing.assert_array_equal(alpha, 0.)
    np.testing.assert_array_equal(beta, 0.)
    with pytest.raises(ValueError):
        toy.analytic_prd([.5, .5], [.5, .5], [0.])


def test_scalar_gaussian_distance_translation_invariance_and_scale_dependence():
    original = toy.gaussian_frechet_1d(toy.P, toy.Q["Q1"])
    np.testing.assert_allclose(toy.gaussian_frechet_1d(toy.P, toy.Q["Q1"], toy.STATES+6), original)
    np.testing.assert_allclose(toy.gaussian_frechet_1d(toy.P, toy.Q["Q1"], 3*toy.STATES), 9*original)


def test_existing_artifacts_cannot_be_overwritten(tmp_path):
    sentinel = tmp_path/"earlier.txt"
    sentinel.write_text("preserve")
    with pytest.raises(FileExistsError):
        toy.run(tmp_path)
    assert sentinel.read_text() == "preserve"
