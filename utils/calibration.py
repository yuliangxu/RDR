"""Dataset-independent fixed-partition and merged score-cell RDR calibration.

The intervals concern population cell averages 2 P(A)/(P(A) + Q(A)).
They do not provide confidence intervals for each individual true RDR value.
Training, partition choice, and independent P/Q calibration draws are the
caller's responsibility. The optional variance_upper_bound transform bounds
true within-cell RDR variance using these mean intervals and RDR's [0, 2] range.
"""

from __future__ import annotations

import operator

import numpy as np
from scipy.stats import beta, norm


def variance_upper_bound(lower, upper):
    """Bound true within-cell RDR variance from an interval for its mean.

    For RDR R in [0, 2] with mean theta, Var(R) <= theta * (2 - theta).
    Maximizing this expression over [lower, upper] gives c * (2 - c),
    where c is the point in that interval closest to 1. The returned bound
    inherits the input interval's coverage event, including any simultaneous
    or adaptive-selection guarantee, without an additional error allocation.

    Endpoints must be finite with 0 <= lower <= upper <= 2. Scalar and array
    inputs may broadcast. A [1, 1] mean interval gives variance bound 1, not
    zero; [0, 2] also gives 1. This bounds population heterogeneity, not the
    sampling variance of an estimated mean or individual true RDR values.
    """
    lower, upper = np.broadcast_arrays(
        np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    )
    if (
        not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower < 0)
        or np.any(upper > 2)
        or np.any(lower > upper)
    ):
        raise ValueError("endpoints must be finite with 0 <= lower <= upper <= 2")
    center = np.clip(1.0, lower, upper)
    return center * (2.0 - center)


def bin_scores(scores, edges):
    """Return bin indices, using [left, right) and including 2 in the last bin.

    ``edges`` must be a finite, strictly increasing partition of [0, 2].
    Scores must be finite and in [0, 2]; invalid predictions are not clipped.
    The returned array has the same shape as ``scores``.
    """
    edges = np.asarray(edges, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if (
        edges.ndim != 1
        or edges.size < 2
        or not np.all(np.isfinite(edges))
        or edges[0] != 0.0
        or edges[-1] != 2.0
        or not np.all(np.diff(edges) > 0)
    ):
        raise ValueError("edges must strictly partition [0, 2]")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 2)):
        raise ValueError("scores must be finite and in [0, 2]")
    indices = np.searchsorted(edges, scores, side="right") - 1
    return np.minimum(indices, edges.size - 2)


def _sample_size(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1 or value > np.iinfo(np.int64).max:
        raise ValueError(f"{name} must be a positive int64 integer")
    return value


def _counts(value, sample_size, name):
    value = np.asarray(value)
    if value.ndim < 1 or value.shape[-1] < 1:
        raise ValueError(f"{name} must have a nonempty final bin axis")
    if value.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain integer counts")
    if (
        not np.all(np.isfinite(value))
        or np.any(value < 0)
        or np.any(value > sample_size)
        or np.any(value != np.floor(value))
    ):
        raise ValueError(f"{name} must contain counts between 0 and its sample size")
    # The final axis is the full prespecified partition, including empty bins.
    if np.any(np.sum(value, axis=-1, dtype=np.float64) != sample_size):
        raise ValueError(f"{name} must sum to its sample size along the bin axis")
    return value.astype(np.float64)


def _cp_limits(counts, sample_size, tail_probability):
    lower = np.zeros_like(counts)
    upper = np.ones_like(counts)
    positive = counts > 0
    nonfull = counts < sample_size
    lower[positive] = beta.ppf(
        tail_probability, counts[positive], sample_size - counts[positive] + 1
    )
    # isf avoids cancellation when computing a very small upper-tail error.
    upper[nonfull] = beta.isf(
        tail_probability, counts[nonfull] + 1, sample_size - counts[nonfull]
    )
    return lower, upper


def calibrate_counts(k_p, k_q, n, m, alpha=0.05):
    """Compute C.1 marginal intervals and C.2 simultaneous bands for all bins.

    ``k_p`` and ``k_q`` have identical shape ``(..., K)``. Leading dimensions
    may index independent calibration replicates, and each final axis must
    sum to ``n`` or ``m`` respectively. K includes all prespecified bins,
    including empty bins. n and m may differ, but are scalar positive integers.

    Every returned value has the count-array shape. The required outputs are
    ``estimate``, ``se``, ``c1_lower``, ``c1_upper``, ``c1_unavailable``,
    ``c1_empty``, ``c2_lower``, and ``c2_upper``. Probability limits
    ``p_lower``, ``p_upper``, ``q_lower``, ``q_upper`` support auditing C.2.

    Empty cells get estimate 1, stored SE 0, and interval [0, 2]. Nonempty
    cells with zero estimated SE also get C.1 interval [0, 2]. Both cases
    set ``c1_unavailable``; no pseudocounts are introduced.
    """
    n = _sample_size(n, "n")
    m = _sample_size(m, "m")
    alpha_array = np.asarray(alpha)
    if alpha_array.ndim != 0 or alpha_array.dtype.kind not in "iuf":
        raise ValueError("alpha must be a scalar strictly between 0 and 1")
    alpha = float(alpha_array)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a scalar strictly between 0 and 1")
    k_p = _counts(k_p, n, "k_p")
    k_q = _counts(k_q, m, "k_q")
    if k_p.shape != k_q.shape:
        raise ValueError("k_p and k_q must have identical shapes")

    p = k_p / n
    q = k_q / m
    total = p + q
    empty = total == 0
    estimate = np.divide(2 * p, total, out=np.ones_like(total), where=~empty)
    numerator = 4 * q**2 * p * (1 - p) / n + 4 * p**2 * q * (1 - q) / m
    variance = np.divide(numerator, total**4, out=np.zeros_like(total), where=~empty)
    se = np.sqrt(variance)
    unavailable = empty | (se == 0)
    radius = norm.isf(alpha / 2) * se
    c1_lower = np.where(unavailable, 0.0, np.maximum(0.0, estimate - radius))
    c1_upper = np.where(unavailable, 2.0, np.minimum(2.0, estimate + radius))

    tail_probability = alpha / (4 * k_p.shape[-1])
    p_lower, p_upper = _cp_limits(k_p, n, tail_probability)
    q_lower, q_upper = _cp_limits(k_q, m, tail_probability)
    c2_lower = 2 * p_lower / (p_lower + q_upper)
    c2_upper = 2 * p_upper / (p_upper + q_lower)
    return {
        "estimate": estimate,
        "se": se,
        "c1_lower": c1_lower,
        "c1_upper": c1_upper,
        "c1_unavailable": unavailable,
        "c1_empty": empty,
        "c2_lower": c2_lower,
        "c2_upper": c2_upper,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "q_lower": q_lower,
        "q_upper": q_upper,
    }


def _elementary_counts(k_p, k_q):
    """Validate one complete, ordered elementary histogram per distribution."""
    result = []
    maximum = np.iinfo(np.int64).max
    for value, name in ((k_p, "k_p"), (k_q, "k_q")):
        value = np.asarray(value)
        if value.ndim != 1 or value.size == 0:
            raise ValueError(f"{name} must be a nonempty one-dimensional histogram")
        if value.dtype.kind not in "iuf" or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must contain finite integer counts")
        if np.any(value < 0) or np.any(value != np.floor(value)):
            raise ValueError(f"{name} must contain nonnegative integer counts")
        # Convert through Python integers so large unsigned/float inputs cannot
        # silently wrap on conversion or during cumulative sums.
        counts = [int(item) for item in value]
        if any(item > maximum for item in counts) or sum(counts) > maximum:
            raise ValueError(f"{name} total must fit in a nonnegative int64")
        result.append(np.asarray(counts, dtype=np.int64))
    if result[0].shape != result[1].shape:
        raise ValueError("k_p and k_q must have identical shapes")
    return tuple(result)


def contiguous_candidates(n_bins):
    """Enumerate every contiguous union as [start, stop), ordered by start/stop.

    There are K(K+1)/2 candidates for K elementary bins, including the whole
    score range. This family is fixed before inspecting calibration counts.
    """
    n_bins = _sample_size(n_bins, "n_bins")
    start, last = np.triu_indices(n_bins)
    return start, last + 1


def merge_adjacent_counts(k_p, k_q, min_count=20):
    """Return selected [start, stop) pairs using the common C.1/C.2 rule.

    Scan left to right and close a region once both accumulated counts reach
    ``min_count``. Attach any leftover right tail to the preceding region. If
    no region reaches the threshold, return the entire elementary range.
    The threshold may exceed either sample size; no pseudocounts are used.
    """
    min_count = _sample_size(min_count, "min_count")
    k_p, k_q = _elementary_counts(k_p, k_q)
    regions = []
    start = 0
    accumulated_p = accumulated_q = 0
    for stop, (count_p, count_q) in enumerate(zip(k_p, k_q), start=1):
        accumulated_p += int(count_p)
        accumulated_q += int(count_q)
        if accumulated_p >= min_count and accumulated_q >= min_count:
            regions.append([start, stop])
            start = stop
            accumulated_p = accumulated_q = 0
    if start < len(k_p):
        if regions:
            regions[-1][1] = len(k_p)
        else:
            regions.append([0, len(k_p)])
    return np.asarray(regions, dtype=np.int64)


def _joint_multiplier_fluctuations(
    p_group_sums, q_group_sums, start, stop, p, q, a, b, n, m
):
    """Evaluate joint candidate fluctuations from elementary-bin normal sums.

    Each input row is one bootstrap repetition. A bin containing k observations
    has a multiplier sum distributed exactly as sqrt(k) times a standard normal.
    Reusing those bin sums for every candidate preserves the complete covariance
    of the observation-level multiplier procedure, including overlapping regions.
    This is an exact conditional Gaussian representation, not an independence
    approximation between candidates. P and Q group sums must be independent.
    """
    p_prefix = np.column_stack(
        (np.zeros(len(p_group_sums)), np.cumsum(p_group_sums, axis=1))
    )
    q_prefix = np.column_stack(
        (np.zeros(len(q_group_sums)), np.cumsum(q_group_sums, axis=1))
    )
    centered_p = p_prefix[:, stop] - p_prefix[:, start] - p * p_prefix[:, -1, None]
    centered_q = q_prefix[:, stop] - q_prefix[:, start] - q * q_prefix[:, -1, None]
    return (a / n) * centered_p + (b / m) * centered_q


def calibrate_merged_counts(
    k_p, k_q, n, m, alpha=0.05, min_count=20,
    bootstrap_repetitions=10000, seed=20260920,
):
    """Calibrate all contiguous candidates, then select common merged regions.

    The frozen network and K elementary bins must be fixed independently of the
    calibration sample. Inputs are complete one-dimensional P/Q histograms and
    may have unequal positive sample sizes. All J=K(K+1)/2 contiguous candidates
    are protected before selecting regions with ``merge_adjacent_counts``.

    C.1 uses one joint Gaussian-multiplier critical value: the empirical
    (1-alpha) quantile (NumPy's ``higher`` convention) of the maximum absolute
    studentized fluctuation across *all* positive-SE candidates. It is an
    asymptotic simultaneous procedure, subject to its regularity conditions.
    Grouped elementary-bin normal sums exactly reproduce the observation-level
    bootstrap conditional on the observed counts; the network is never refitted.

    C.2 uses Clopper-Pearson limits at tail probability alpha/(4J), retaining J
    even when selection yields fewer regions. Under independent iid P/Q draws,
    its finite-sample simultaneous guarantee survives this adaptive selection.
    Both methods target 2 P(A)/(P(A)+Q(A)) for the selected region, not individual
    RDR values inside it. An undefined population target at zero mixture mass
    is outside either coverage claim.

    Empty candidates have placeholder estimate 1 and SE 0. C.1 uses [0, 2] for
    empty or zero-SE candidates, except that the whole-space candidate has the
    known target 1 and receives [1, 1] under both methods. ``c1_unavailable``
    marks only candidates receiving the fallback, so it is false for whole space.

    Returns a dictionary with per-candidate arrays in ``candidates``, indices of
    the selected candidates in ``selected_indices``, the elementary-bin to
    selected-region map in ``elementary_to_merged``, the bootstrap maxima, and
    reproducibility metadata. Start/stop indices use [start, stop) throughout.
    ``calibrate_counts`` retains its original fixed-partition behavior.
    """
    n = _sample_size(n, "n")
    m = _sample_size(m, "m")
    min_count = _sample_size(min_count, "min_count")
    bootstrap_repetitions = _sample_size(bootstrap_repetitions, "bootstrap_repetitions")
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a nonnegative integer")
    try:
        seed = operator.index(seed)
    except TypeError as exc:
        raise ValueError("seed must be a nonnegative integer") from exc
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    alpha_array = np.asarray(alpha)
    if alpha_array.ndim != 0 or alpha_array.dtype.kind not in "iuf":
        raise ValueError("alpha must be a scalar strictly between 0 and 1")
    alpha = float(alpha_array)
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be a scalar strictly between 0 and 1")
    k_p, k_q = _elementary_counts(k_p, k_q)
    if sum(map(int, k_p)) != n or sum(map(int, k_q)) != m:
        raise ValueError("each complete histogram must sum to its sample size")

    n_bins = len(k_p)
    start, stop = contiguous_candidates(n_bins)
    n_candidates = len(start)
    p_prefix = np.concatenate(([0], np.cumsum(k_p, dtype=np.int64)))
    q_prefix = np.concatenate(([0], np.cumsum(k_q, dtype=np.int64)))
    count_p = p_prefix[stop] - p_prefix[start]
    count_q = q_prefix[stop] - q_prefix[start]
    p, q = count_p / n, count_q / m
    total = p + q
    empty = total == 0
    whole_space = (start == 0) & (stop == n_bins)
    estimate = np.divide(2 * p, total, out=np.ones_like(total), where=~empty)
    a = np.divide(2 * q, total**2, out=np.zeros_like(total), where=~empty)
    b = np.divide(-2 * p, total**2, out=np.zeros_like(total), where=~empty)
    se = np.sqrt(a**2 * p * (1 - p) / n + b**2 * q * (1 - q) / m)
    studentized = (se > 0) & ~whole_space

    maxima = np.zeros(bootstrap_repetitions)
    rng = np.random.default_rng(seed)
    # Bound the large repetition-by-candidate temporaries while retaining the
    # maxima for a reproducible empirical quantile and downstream auditing.
    batch_size = min(512, max(1, 2_000_000 // n_candidates))
    if np.any(studentized):
        for offset in range(0, bootstrap_repetitions, batch_size):
            size = min(batch_size, bootstrap_repetitions - offset)
            p_sums = rng.standard_normal((size, n_bins)) * np.sqrt(k_p)
            q_sums = rng.standard_normal((size, n_bins)) * np.sqrt(k_q)
            fluctuations = _joint_multiplier_fluctuations(
                p_sums, q_sums, start, stop, p, q, a, b, n, m
            )
            maxima[offset:offset + size] = np.max(
                np.abs(fluctuations[:, studentized]) / se[studentized], axis=1
            )
    critical_value = float(np.quantile(maxima, 1 - alpha, method="higher"))
    unavailable = (empty | (se == 0)) & ~whole_space
    radius = critical_value * se
    c1_lower = np.where(unavailable, 0.0, np.maximum(0.0, estimate - radius))
    c1_upper = np.where(unavailable, 2.0, np.minimum(2.0, estimate + radius))

    tail_probability = alpha / (4 * n_candidates)
    p_lower, p_upper = _cp_limits(count_p.astype(float), n, tail_probability)
    q_lower, q_upper = _cp_limits(count_q.astype(float), m, tail_probability)
    c2_lower = 2 * p_lower / (p_lower + q_upper)
    c2_upper = 2 * p_upper / (p_upper + q_lower)
    for bounds in (c1_lower, c1_upper, c2_lower, c2_upper):
        bounds[whole_space] = 1.0

    selected_regions = merge_adjacent_counts(k_p, k_q, min_count)
    lookup = {(int(left), int(right)): i for i, (left, right) in enumerate(zip(start, stop))}
    selected_indices = np.asarray(
        [lookup[tuple(region)] for region in selected_regions], dtype=np.int64
    )
    selected = np.zeros(n_candidates, dtype=bool)
    selected[selected_indices] = True
    elementary_to_merged = np.empty(n_bins, dtype=np.int64)
    for index, (left, right) in enumerate(selected_regions):
        elementary_to_merged[left:right] = index

    return {
        "candidates": {
            "start": start, "stop": stop, "k_p": count_p, "k_q": count_q,
            "p": p, "q": q, "estimate": estimate, "se": se, "a": a, "b": b,
            "c1_lower": c1_lower, "c1_upper": c1_upper,
            "c1_unavailable": unavailable, "c1_empty": empty,
            "c2_lower": c2_lower, "c2_upper": c2_upper,
            "p_lower": p_lower, "p_upper": p_upper,
            "q_lower": q_lower, "q_upper": q_upper,
            "whole_space": whole_space, "selected": selected,
        },
        "selected_indices": selected_indices,
        "elementary_to_merged": elementary_to_merged,
        "bootstrap_maxima": maxima,
        "metadata": {
            "alpha": alpha, "n": n, "m": m,
            "n_elementary_bins": n_bins, "n_candidates": n_candidates,
            "n_selected_regions": len(selected_indices), "min_count": min_count,
            "bootstrap_repetitions": bootstrap_repetitions, "seed": seed,
            "critical_value": critical_value, "quantile_method": "higher",
            "c2_tail_probability": tail_probability,
            "bootstrap_batch_size": batch_size,
            "c1_studentized_candidates": int(np.count_nonzero(studentized)),
            "candidate_family": "all_contiguous_elementary_unions",
        },
    }
