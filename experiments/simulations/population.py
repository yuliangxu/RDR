"""Prespecified Gaussian-mixture populations for selection and convergence."""

import numpy as np
from scipy.special import expit, logsumexp

SETTINGS = {
    "D2_noise0.00": (2, 0.), "D20_noise0.10": (20, .1),
    "D40_noise0.10": (40, .1), "D40_noise0.30": (40, .3),
    "D100_noise0.30": (100, .3),
}


class Mixture:
    """Three-component Gaussian mixtures lifted into a fixed noisy subspace.

    The shared orthogonal noise cancels from p/q; truth integrates the
    projected noise into each component covariance before taking the ratio.
    """
    def __init__(self, case):
        self.dimension, self.noise = SETTINGS[case]
        self.k = np.eye(2) if self.dimension == 2 else np.linalg.qr(
            np.random.default_rng(42).normal(size=(self.dimension, 2)), mode='reduced')[0]
        self.w = np.array([.3, .3, .4])
        self.means = {'P': np.array([[-2., -2.], [-1., 5.], [5., 5.]]),
                      'Q': np.array([[0., 0.], [-1., 5.], [5., 5.]])}
        self.covs = {'P': np.array([[[1., .5], [.5, 1.]], np.eye(2), [[2., -1.8], [-1.8, 2.]]]),
                     'Q': np.array([[[1., .5], [.5, 1.]], np.eye(2), 2 * np.eye(2)])}
        self.chol = {key: np.linalg.cholesky(value) for key, value in self.covs.items()}
        convolved = {key: value + self.noise**2 * np.eye(2) for key, value in self.covs.items()}
        self.inv = {key: np.linalg.inv(value) for key, value in convolved.items()}
        self.logdet = {key: np.linalg.slogdet(value)[1] for key, value in convolved.items()}

    def draw(self, rng, n, source):
        if source == 'M':
            p = rng.integers(0, 2, size=n).astype(bool)
            component = rng.choice(3, n, p=self.w)
            means = np.where(p[:, None], self.means['P'][component], self.means['Q'][component])
            chol = np.where(p[:, None, None], self.chol['P'][component], self.chol['Q'][component])
        else:
            component = rng.choice(3, n, p=self.w)
            means, chol = self.means[source][component], self.chol[source][component]
        y = means + np.einsum('nij,nj->ni', chol, rng.normal(size=(n, 2)))
        return y if self.dimension == 2 else y @ self.k.T + rng.normal(0, self.noise, (n, self.dimension))

    def truth(self, x):
        z = x @ self.k
        logs = {}
        for source in ('P', 'Q'):
            delta = z[:, None, :] - self.means[source]
            quad = np.einsum('nki,kij,nkj->nk', delta, self.inv[source], delta)
            logs[source] = logsumexp(np.log(self.w) - .5 * (quad + self.logdet[source]), axis=1)
        return 2 * expit(logs['P'] - logs['Q'])

