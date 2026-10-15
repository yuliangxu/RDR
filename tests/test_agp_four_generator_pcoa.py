"""Independent dense-distance checks for the iterative Lingoes ordination."""
import numpy as np
import pytest
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform

from experiments.AGP.four_generator_pcoa import pcoa_from_distances


@pytest.mark.parametrize('n,d', [(39, 8), (83, 61), (21, 4)])
def test_lingoes_matches_explicit_corrected_distances(n, d):
    rng = np.random.default_rng(42+d)
    x = rng.exponential(size=(n, d))
    x[rng.random(x.shape) < .65] = 0
    x[:, 0] += .01
    x /= x.sum(axis=1, keepdims=True)
    distances = squareform(pdist(x, metric='braycurtis'))
    coords, diagnostics, _ = pcoa_from_distances(distances, threads=1)
    center = np.eye(n)-np.ones((n, n))/n
    raw = -.5*center@(distances**2)@center
    original = np.linalg.eigvalsh(raw)
    correction = max(0., -original[0])
    corrected_d2 = distances**2+2*correction*(1-np.eye(n))
    corrected = -.5*center@corrected_d2@center
    values, vectors = eigh(corrected)
    expected = vectors[:, -2:]*np.sqrt(values[-2:])
    np.testing.assert_allclose(coords@coords.T, expected@expected.T, atol=1e-10, rtol=1e-9)
    np.testing.assert_allclose(diagnostics['lingoes_constant'], correction, atol=1e-10)
    np.testing.assert_allclose(diagnostics['corrected_total_inertia'], np.trace(corrected), atol=1e-10)
    np.testing.assert_allclose(diagnostics['displayed_corrected_inertia_fractions'],
                               values[-2:][::-1]/np.trace(corrected), atol=1e-10)
    np.testing.assert_allclose(coords.mean(axis=0), 0, atol=1e-12)


def test_euclidean_case_needs_no_material_correction():
    rng = np.random.default_rng(12)
    distances = squareform(pdist(rng.normal(size=(25, 3))))
    _, diagnostics, _ = pcoa_from_distances(distances, threads=1)
    assert diagnostics['lingoes_constant'] < 1e-10


def test_rejects_asymmetric_distances():
    distances = np.ones((4, 4))-np.eye(4)
    distances[0, 1] = .5
    with pytest.raises(ValueError, match='symmetric'):
        pcoa_from_distances(distances, threads=1)
