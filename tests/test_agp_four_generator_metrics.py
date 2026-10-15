"""Analytic checks for the AGP four-generator Frechet and PRD metrics."""
import numpy as np
import pytest
from experiments.AGP import rdr_metric_utils as m


def test_closure_and_embedding_zero_semantics():
    raw = np.array([[2., 0., 3.], [4., 0., 6.]])
    np.testing.assert_array_equal(m.close(raw)[0], m.close(raw)[1])
    np.testing.assert_allclose(np.square(m.embed(raw)).sum(axis=1), 1)
    assert np.all(m.embed(raw)[:, 1] == 0)
    np.testing.assert_allclose(m.embed(raw, "clr").sum(axis=1), 0, atol=2e-15)
    for bad in (np.zeros((1, 2)), [[1., -1.]], [[np.inf, 1.]]):
        with pytest.raises(ValueError):
            m.close(bad)

def test_frechet_translation_scale_singular_and_symmetry():
    x = np.array([[-1., -2., 0.], [0., 0., 0.], [1., 2., 0.]])
    assert m.frechet(x, x.copy()) == 0
    np.testing.assert_allclose(m.frechet(x, x+[1., 2., 3.]), 14, atol=1e-12)
    np.testing.assert_allclose(m.frechet(x, 2*x), 5, atol=1e-12)
    rng = np.random.default_rng(193)
    a, b = rng.normal(size=(9, 15)), rng.normal(size=(13, 15))
    np.testing.assert_allclose(m.frechet(a, b), m.frechet(b, a), rtol=1e-12, atol=1e-12)
    # Alternate sample-factor nuclear-norm formula, no covariance eigensolver.
    ac, bc = (a-a.mean(0))/np.sqrt(len(a)-1), (b-b.mean(0))/np.sqrt(len(b)-1)
    expected = np.square(a.mean(0)-b.mean(0)).sum() + (ac*ac).sum() + (bc*bc).sum()
    expected -= 2*np.linalg.svd(ac @ bc.T, compute_uv=False).sum()
    np.testing.assert_allclose(m.frechet(a, b), expected, atol=1e-7)

def test_prd_probability_normalization_symmetry_disjoint_and_identity():
    p, q = np.array([.5, .5]), np.array([.8, .2])
    precision, recall = m.prd(p, q)
    precision2, recall2 = m.prd(p[::-1], q[::-1])
    np.testing.assert_array_equal(precision, precision2)
    np.testing.assert_array_equal(recall, recall2)
    swapped_p, swapped_r = m.prd(q, p)
    np.testing.assert_allclose(precision, swapped_r[::-1], atol=1e-15)
    np.testing.assert_allclose(recall, swapped_p[::-1], atol=1e-15)
    np.testing.assert_allclose(m.prd(p*22, q*30), m.prd(p, q), atol=1e-15)
    assert m.prd_max_f1(*m.prd(p, p)) == pytest.approx(1)
    assert m.prd_max_f1(*m.prd([1, 0], [0, 1])) == 0
