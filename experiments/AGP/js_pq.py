"""Held-out, equal-source-weight variational JS(P,Q), in natural-log units."""
from __future__ import annotations

import math


def _source(values, name):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f'{name} must be a nonempty sequence of finite scores')
    return values


def _softplus(value):
    """log(1+exp(value)) without overflowing exp for large positive logits."""
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def evaluate_js_pq(p, q, *, representation='rdr'):
    """Evaluate every P/Q score with separate means and equal source weights.

    ``rdr`` inputs estimate 2p/(p+q) and lie in [0,2]. P=0 or Q=2 cannot
    recover a finite log from rounded scores and require ``logits`` instead.
    P=2 and Q=0 have valid observed-source logs. ``logits`` inputs are scaled
    logits t for r=2*sigmoid(t), including any fitted output slope.

    No clipping, endpoint replacement, or truncation of negative estimates is
    performed. This fitted variational score need not be nonnegative and is
    not an exact population divergence. No observations are pooled or dropped.
    """
    p, q = _source(p, 'P'), _source(q, 'Q')
    log_two = math.log(2)
    if representation == 'logits':
        log_p = [log_two - _softplus(-value) for value in p]
        log_q = [log_two - _softplus(value) for value in q]
    elif representation == 'rdr':
        if any(not 0 <= value <= 2 for value in p + q):
            raise ValueError('RDR scores must lie in [0,2]')
        if 0 in p or 2 in q:
            raise ValueError('P=0 or Q=2 requires finite logits from the frozen checkpoint; no clipping is used')
        log_p = [math.log(value) for value in p]
        log_q = [log_two + math.log1p(-value/2) for value in q]
    else:
        raise ValueError("representation must be 'rdr' or 'logits'")
    mean_p = math.fsum(log_p)/len(p)
    mean_q = math.fsum(log_q)/len(q)
    value = .5*mean_p + .5*mean_q
    return dict(js_p_q_variational=value, mean_log_r_p=mean_p,
                mean_log_two_minus_r_q=mean_q, balanced_bce=log_two-value,
                n_p=len(p), n_q=len(q))
