"""Reusable RDR objectives, preserving existing workflows' conventions.

R denotes the distribution actually supplied as the denominator. Functions
whose names include midpoint explicitly expand the (P+Q)/2 expectation.
Output parameterization,
independent sample means, mixed versus separate forwards, and constant terms
are deliberately preserved; similarly named objectives need not be equivalent.
"""

import torch
import torch.nn as nn

__all__ = [
    'Hellinger_loss',
    'KL_loss',
    'Chisq_loss',
    'JS_loss',
    'hellinger_stable',
    'kl_from_outputs',
    'chisq_from_outputs',
    'original_hellinger_midpoint_loss',
    'midpoint_hellinger_loss',
    'concatenated_midpoint_hellinger_loss',
]


# Model-based objectives

def Hellinger_loss(model, x_p, x_q, xi=0.5, log_scale = False):
    """Weighted Hellinger objective for model outputs on P and a denominator R.

    ``x_p`` and ``x_q`` are independently averaged, even with unequal counts.
    The objective is xi*mean_P(w**-0.5) + (1-xi)*mean_R(w**0.5) - 1.
    For 0 < xi < 1 the population minimizer is (xi/(1-xi))*p/r; xi=0.5
    targets p/r. For relative ratios, callers must supply R=(P+Q)/2.

    Inputs are moved to the model's device/dtype and forwarded together so
    BatchNorm sees one mixed batch. ``log_scale=True`` interprets outputs
    as log(w); otherwise strictly positive outputs are required. No clipping
    is applied, and the model's output range may constrain the minimizer.
    """
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    x_p = x_p.to(device=device, dtype=dtype)
    x_q = x_q.to(device=device, dtype=dtype)

    X = torch.cat([x_p, x_q], dim=0)
    w = model(X)  # BN sees the same mixed distribution once
    w_p, w_q = w[:x_p.size(0)], w[x_p.size(0):]


    if log_scale:
        loss_p = torch.mean(torch.exp( -0.5 * w_p))
        loss_q = torch.mean(torch.exp(0.5*w_q))
    else:
        loss_p = torch.mean(w_p.pow(-0.5))
        loss_q = torch.mean(w_q.pow(0.5))
    loss = xi*loss_p + (1-xi)*loss_q -1
    return loss


def KL_loss(model, x_p, x_q):
    """Return mean_R(w) - mean_P(log(w)) - 1 for a positive-output model.

    Each sample mean uses its own count. The unrestricted target is p/r;
    pass midpoint samples as ``x_q`` to target the relative ratio 2p/(p+q).
    P and R are forwarded separately, preserving historical BatchNorm behavior.
    Inputs must already match the model's device/dtype. No clipping is applied.
    """
    w_p = model(x_p)  # shape: (N_p, 1)
    w_q = model(x_q)  # shape: (N_q, 1)

    loss_p = torch.mean(torch.log(w_p))

    loss_q = - torch.mean(w_q)

    loss = - (1+ loss_p + loss_q)
    return loss


def Chisq_loss(model, x_p, x_q):
    """Return mean_R(w**2) - 2*mean_P(w) - 1 (historical convention).

    Each sample mean uses its own count and the unrestricted target is p/r.
    ``x_q`` denotes the supplied denominator distribution R, not necessarily Q.
    Separate forwards preserve BatchNorm behavior. This objective differs from
    ``chisq_from_outputs``; their similarly named losses are not interchangeable.
    """
    w_p = model(x_p)  # shape: (N_p, 1)
    w_q = model(x_q)  # shape: (N_q, 1)

    loss_p = 2 * torch.mean(w_p)


    loss_q = - torch.mean(w_q.pow(2))

    loss = - (loss_p + loss_q + 1)
    return loss


def JS_loss(model, x_p, x_q):
    """Bounded Jensen-Shannon objective for a model emitting w in (0, 2).

    Returns -0.5*mean_P(log(w/(2-w))) - mean_R(log(2-w)); the
    unrestricted stationary target is p/r within (0,2). To estimate the
    relative ratio 2p/(p+q), supply midpoint samples R=(P+Q)/2 as ``x_q``.
    Each sample mean uses its own count. Outputs are clipped to dtype epsilon
    and 2-epsilon, and P and R use separate model forwards.

    The generating function has phi''(w)=1/(w*(2-w)) and
    phi'(w)=0.5*log(w/(2-w)). This is not binary cross-entropy on
    classifier probabilities.
    """
    w_p = model(x_p)
    w_q = model(x_q)
    eps = torch.finfo(w_p.dtype).eps
    w_p = torch.clamp(w_p, min=eps, max=2 - eps)
    w_q = torch.clamp(w_q, min=eps, max=2 - eps)

    loss_p = 0.5 * torch.mean(torch.log(w_p / (2 - w_p)))
    loss_q = torch.mean(torch.log(2 - w_q))
    return - (loss_p + loss_q)


# Output-based CNN objectives

def hellinger_stable(w_p, w_q, *, log_space: bool = False,
                    eps: float = 1e-6, log_clip: float = 8.0):
    """Return 0.5*mean_P(w**-0.5) + 0.5*mean_R(w**0.5) - 1.

    Historical output-based objective requiring positive ratio outputs.
    Despite the function name, ``log_space``, ``eps``, and ``log_clip`` are
    accepted but unused: there is no log conversion or clipping. Preserve
    this behavior when reproducing old CNN fits; log outputs are not valid
    inputs. Each tensor is averaged independently and the target is p/r.
    """
    return 0.5*torch.mean(w_p.pow(-0.5)) + 0.5*torch.mean(w_q.pow(0.5))-1


def kl_from_outputs(logw_p, logw_q):
    """Return -mean_P(log(w)) + mean_R(exp(log(w))).

    Inputs must be log ratios. Each mean uses its own count, the target is
    p/r, and this differs from ``KL_loss`` only by the constant +1 when
    evaluated on equivalent outputs.
    """
    return -torch.mean(logw_p) + torch.mean(torch.exp(logw_q))


def chisq_from_outputs(w_p, w_q):
    """Return mean_P((w-1)**2) + mean_R(w**2).

    This historical CNN objective has unrestricted target p/(p+r), not p/r.
    It is intentionally separate from ``Chisq_loss``. Inputs are direct
    model outputs, with each tensor averaged using its own sample count.
    """
    return torch.mean((w_p - 1.0) ** 2) + torch.mean(w_q ** 2)


def original_hellinger_midpoint_loss(
    model: nn.Module,
    p_batch: torch.Tensor,
    q_batch: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Clipped Hellinger midpoint risk evaluated from separate P/Q samples.

    Uses one concatenated forward, then clips outputs to [epsilon,2-epsilon].
    Each mean uses its own sample count, retaining the 1/2,1/4,1/4 weights
    when counts differ. The population target before clipping is 2p/(p+q).
    Distinct from midpoint_hellinger_loss, which uses two unclipped forwards.
    """
    outputs = model(torch.cat([p_batch, q_batch], dim=0)).reshape(-1)
    p_ratio = outputs[: len(p_batch)].clamp(epsilon, 2.0 - epsilon)
    q_ratio = outputs[len(p_batch) :].clamp(epsilon, 2.0 - epsilon)
    return (
        0.5 * p_ratio.rsqrt().mean()
        + 0.25 * p_ratio.sqrt().mean()
        + 0.25 * q_ratio.sqrt().mean()
        - 1.0
    )


def midpoint_hellinger_loss(model, x_p, x_q):
    """Empirical Hellinger risk targeting r=2p/(p+q).

    The original loss compares P against M=(P+Q)/2.  Expanding the M mean
    yields .5 E_P[r^-1/2] + .25 E_P[sqrt(r)] + .25 E_Q[sqrt(r)] - 1.
    Separate means retain equal mixture weights even when sample sizes differ.
    No clipping is applied to the model output or objective.
    """
    r_p, r_q = model(x_p), model(x_q)
    return .5 * r_p.pow(-.5).mean() + .25 * r_p.sqrt().mean() + .25 * r_q.sqrt().mean() - 1


def concatenated_midpoint_hellinger_loss(model, p, q):
    """Exact empirical 50:50 midpoint objective even when n_P != n_Q.

    L(r) = .5 E_P[r^-1/2] + .25 E_P[sqrt(r)] + .25 E_Q[sqrt(r)] - 1.
    Its population minimizer is r = p / ((p+q)/2), on [0,2].
    A single concatenated forward preserves mixed-batch BatchNorm statistics.
    The model must return (N,1) outputs; no clipping or log conversion is used.
    This differs from the other midpoint variants in forwarding and clipping.
    """
    predictions = model(torch.cat((p, q)))[:, 0]
    rp, rq = predictions[:len(p)], predictions[len(p):]
    return .5 * rp.rsqrt().mean() + .25 * rp.sqrt().mean() + .25 * rq.sqrt().mean() - 1


def midpoint_rdr_loss(model, p, q, loss="hellinger"):
    """Expanded midpoint risks from separate P/Q draws, targeting 2p/(p+q).

    Each distribution is averaged separately, including unequal sample sizes.
    Hellinger, KL and chi-square expand the historical denominator M=(P+Q)/2.
    Expanding the bounded JS objective cancels its P log(2-r) terms, giving
    -.5 E_P log(r) -.5 E_Q log(2-r). This is balanced P/Q binary cross-entropy
    for probability r/2, minus log(2). It is distinct from the historical
    classifier trained against midpoint samples and converted using odds.
    Only JS clips predictions at dtype epsilon before its logarithms; the
    other risks retain the historical unmodified formulas.
    """
    if loss == "hellinger":
        return midpoint_hellinger_loss(model, p, q)
    rp, rq = model(p), model(q)
    if loss == "kl":
        return .5 * rp.mean() + .5 * rq.mean() - rp.log().mean() - 1
    if loss == "chisq":
        return .5 * rp.square().mean() + .5 * rq.square().mean() - 2 * rp.mean() - 1
    if loss == "js":
        epsilon = torch.finfo(rp.dtype).eps
        return -.5 * rp.clamp(epsilon, 2-epsilon).log().mean() - .5 * (2-rq.clamp(epsilon, 2-epsilon)).log().mean()
    raise ValueError(f"Unknown RDR objective: {loss}")


__all__ += ["midpoint_rdr_loss"]
