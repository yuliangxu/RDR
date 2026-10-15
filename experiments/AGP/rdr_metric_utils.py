"""Numerical primitives for declared-feature global and local AGP metrics.

Fréchet distance is the Gaussian mean/covariance analogue, not image FID.
PRD uses P as the reference distribution and Q as the evaluated distribution.
All Euclidean neighborhood calculations use explicit SciPy distances so exact
duplicate vectors have exactly zero distance (no Gram-matrix cancellation).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import cdist


def close(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Expected finite nonempty nonnegative composition matrix")
    sums = x.sum(axis=1, keepdims=True)
    if not np.isfinite(sums).all() or (sums <= 0).any():
        raise ValueError("Cannot close empty composition")
    return x / sums


def embed(x, representation="sqrt", epsilon=1e-6):
    x = close(x)
    if representation == "sqrt":
        return np.sqrt(x)
    if representation == "clr":
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("CLR requires positive finite epsilon")
        logs = np.log(x + epsilon)
        return logs - logs.mean(axis=1, keepdims=True)
    raise ValueError("Unknown representation: " + str(representation))


def _features(x, minimum=1):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < minimum or x.shape[1] < 1 or not np.isfinite(x).all():
        raise ValueError("Expected finite feature matrix with sufficient observations")
    return x


def _psd_eigh(a):
    a = (a + a.T) / 2
    values, vectors = eigh(a, check_finite=False)
    scale = max(1., float(np.max(np.abs(values))))
    if values.min() < -1e-9 * scale:
        raise FloatingPointError("Covariance-derived matrix is materially non-PSD")
    return np.maximum(values, 0), vectors


def frechet(x, y):
    """Squared Gaussian 2-Wasserstein distance, unbiased sample covariances."""
    x, y = _features(x, 2), _features(y, 2)
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions differ")
    if x.shape == y.shape and np.array_equal(x, y):
        return 0.
    a = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    b = np.atleast_2d(np.cov(y, rowvar=False, ddof=1))
    eigenvalues, eigenvectors = _psd_eigh(a)
    # Use the SVD nuclear norm of sqrt(A)sqrt(B), which remains accurate for
    # rank-deficient covariances; eig(sqrt(A) B sqrt(A)) amplifies small noise.
    other_values, other_vectors = _psd_eigh(b)
    factor = (np.sqrt(eigenvalues)[:, None] * (eigenvectors.T @ other_vectors)
              * np.sqrt(other_values)[None, :])
    cross = np.linalg.svd(factor, compute_uv=False).sum()
    mean_term = np.square(x.mean(axis=0) - y.mean(axis=0)).sum()
    distance = float(mean_term + np.trace(a) + np.trace(b) - 2 * cross)
    tolerance = 1e-9 * max(1., float(mean_term + np.trace(a) + np.trace(b)))
    if distance < -tolerance:
        raise FloatingPointError("Negative Fréchet distance beyond rounding")
    return max(0., distance)


def prd(p_hist, q_hist, num_angles=201):
    """Finite histogram precision/recall distribution curve (Sajjadi et al.).

    Endpoints use angles [1e-10, pi/2-1e-10], matching the reference code;
    precision=sum min(lambda*p,q), recall=precision/lambda.
    """
    p, q = np.asarray(p_hist, float), np.asarray(q_hist, float)
    if (p.ndim != 1 or p.shape != q.shape or p.size == 0 or
            not np.isfinite(np.r_[p, q]).all() or (p < 0).any() or (q < 0).any()
            or p.sum() <= 0 or q.sum() <= 0 or num_angles < 3):
        raise ValueError("Invalid PRD histogram or angle count")
    p, q = p / p.sum(), q / q.sum()
    slopes = np.tan(np.linspace(1e-10, np.pi/2 - 1e-10, int(num_angles)))
    precision = np.minimum(slopes[:, None] * p, q).sum(axis=1)
    recall = precision / slopes
    return np.clip(precision, 0, 1), np.clip(recall, 0, 1)


def prd_max_f1(precision, recall):
    precision, recall = np.asarray(precision), np.asarray(recall)
    return float(np.max(np.divide(2 * precision * recall, precision + recall,
                                 out=np.zeros_like(precision, dtype=float),
                                 where=(precision + recall) > 0)))


def manifold_radii_many(reference, ks=(3, 5, 10), chunk_size=256):
    reference = _features(reference, 2)
    ks = tuple(sorted(set(int(k) for k in ks)))
    if not ks or min(ks) < 1 or max(ks) >= len(reference) or chunk_size < 1:
        raise ValueError("Invalid neighborhood size or chunk size")
    out = {k: np.empty(len(reference)) for k in ks}
    for start in range(0, len(reference), chunk_size):
        stop = min(start + chunk_size, len(reference))
        distances = cdist(reference[start:stop], reference, metric="euclidean")
        distances[np.arange(stop-start), np.arange(start, stop)] = np.inf
        partitioned = np.partition(distances, [k-1 for k in ks], axis=1)
        for k in ks:
            out[k][start:stop] = partitioned[:, k-1]
    return out


def manifold_radii(reference, k, chunk_size=256):
    return manifold_radii_many(reference, (k,), chunk_size)[int(k)]


def realism_many(query, reference, radii_dict, chunk_size=256, reference_masks=None):
    query, reference = _features(query), _features(reference)
    if query.shape[1] != reference.shape[1] or chunk_size < 1 or not radii_dict:
        raise ValueError("Invalid realism dimensions or chunk size")
    checked = {}
    for key, radius in radii_dict.items():
        radius = np.asarray(radius, float)
        if radius.shape != (len(reference),) or not np.isfinite(radius).all() or (radius < 0).any():
            raise ValueError("Invalid manifold radius vector")
        checked[key] = radius
    out = {k: np.empty(len(query)) for k in checked}
    masks = {}
    for key in checked:
        mask = (np.ones(len(reference), dtype=bool) if reference_masks is None
                else np.asarray(reference_masks[key], dtype=bool))
        if mask.shape != (len(reference),) or not mask.any():
            raise ValueError("Reference masks must select nonempty centers")
        masks[key] = mask
    for start in range(0, len(query), chunk_size):
        stop = min(start + chunk_size, len(query))
        distances = cdist(query[start:stop], reference, metric="euclidean")
        for key, radius in checked.items():
            ratios = np.divide(radius[None, :], distances,
                               out=np.zeros_like(distances), where=distances > 0)
            ratios[(distances == 0) & (radius[None, :] > 0)] = np.inf
            # A zero-radius closed ball includes its center, including duplicates.
            ratios[(distances == 0) & (radius[None, :] == 0)] = 1.
            ratios[:, ~masks[key]] = -np.inf
            out[key][start:stop] = ratios.max(axis=1)
    return out


def realism(query, reference, radii, chunk_size=256):
    return realism_many(query, reference, {0: radii}, chunk_size)[0]


def nearest_distance(query, reference, chunk_size=256):
    query, reference = _features(query), _features(reference)
    if query.shape[1] != reference.shape[1] or chunk_size < 1:
        raise ValueError("Invalid distance dimensions or chunk size")
    result = np.empty(len(query))
    for start in range(0, len(query), chunk_size):
        result[start:start+chunk_size] = cdist(query[start:start+chunk_size], reference).min(axis=1)
    return result


def support_fraction(query, reference, radii, chunk_size=256):
    return float(np.mean(realism(query, reference, radii, chunk_size) >= 1))
