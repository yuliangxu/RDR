"""Reusable evaluation, numeric summaries, and plots for ratio experiments.

Scores are used as supplied: these helpers do not convert ordinary density
ratios p/q into relative density ratios 2p/(p+q). In particular,
``density_ratio_classifier`` estimates p/q, while ``select_extremes`` assumes
relative scores on [0, 2] with equality at 1. Analytic simulation populations
live in ``experiments.simulations.population`` and the toy notebook.

Plotting, pandas, and scikit-learn dependencies are imported by the functions
that use them, so prediction and tensor summaries do not require those tools.
The evaluation functions leave the model in evaluation mode, matching the
historical research workflow.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

__all__ = [
    'evaluate_model',
    'evaluate_model_embed',
    'density_ratio_classifier',
    'summarize_vector',
    'summarize_by_group',
    'compare_l2',
    'minmax_text_from_idx',
    'select_extremes',
    'plot_losses',
    'add_loss_line',
    'plot_1d_density',
    'plot_1d_density_compare',
    'plot_theoretical_ratio',
    'add_density_ratio_line',
    'plot_ratio_hist',
    'add_ratio_hist',
    'scatter_compare_ratios',
    'twoD_generate_grid_numpy',
    'twoD_plot_density_ratio',
    'plot_calibration_cells',
]


# Model evaluation
# ----------------

def evaluate_model(
    model: torch.nn.Module,
    x_p: torch.Tensor,
    x_q: torch.Tensor,
    x_mixed: Optional[torch.Tensor] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """
    Evaluate a model on two or three sets of inputs.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model (e.g., DCGAN discriminator).
    x_p, x_q : torch.Tensor
        Input batches.
    x_mixed : torch.Tensor, optional
        Input batch for mixture. If None, it will be skipped.

    Returns
    -------
    (g_p, g_q, g_mixed) if x_mixed is not None,
    otherwise (g_p, g_q).
    """
    model.eval()
    with torch.no_grad():
        p = next(model.parameters())
        device, dtype = p.device, p.dtype

        x_p_ = x_p.to(device=device, dtype=dtype)
        x_q_ = x_q.to(device=device, dtype=dtype)

        g_p = model(x_p_).squeeze(-1)
        g_q = model(x_q_).squeeze(-1)

        if x_mixed is not None:
            x_mixed_ = x_mixed.to(device=device, dtype=dtype)
            g_mixed = model(x_mixed_).squeeze(-1)
            return g_p, g_q, g_mixed
        else:
            return g_p, g_q


def _unpack_loader_batch(batch):
    # returns (z, imgs_or_None)
    if isinstance(batch, dict):
        z = batch.get('z', batch.get('emb', None))
        if z is None:
            for v in batch.values():
                if torch.is_tensor(v):
                    z = v; break
        imgs = batch.get('imgs', None)
        return z, imgs
    if isinstance(batch, (list, tuple)):
        # common case: (z, imgs)
        if len(batch) >= 1 and torch.is_tensor(batch[0]):
            z = batch[0]
            imgs = batch[1] if (len(batch) >= 2 and torch.is_tensor(batch[1])) else None
            return z, imgs
        # fallback: first tensor found
        for v in batch:
            if torch.is_tensor(v): return v, None
    if torch.is_tensor(batch):
        return batch, None
    raise TypeError(f"Unsupported batch type: {type(batch)}")


def evaluate_model_embed(
    model: torch.nn.Module,
    p_embed_loader,
    q_embed_loader,
    mixed_embed_loader,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Evaluate the first batch from each of three embedding loaders.

    This is a batch diagnostic, not full-loader evaluation.
    If loaders yield images alongside embeddings, also return those images.

    Returns
    -------
    g_p, g_q, g_mixed : (B,) tensors of model outputs
    x_p, x_q, x_mixed : (B, C, H, W) tensors of original images, or None if unavailable
    """
    model.eval()
    with torch.no_grad():
        # model device/dtype
        p = next(model.parameters())
        device, dtype = p.device, p.dtype

        # pull one batch from each loader
        batch_p     = next(iter(p_embed_loader))
        batch_q     = next(iter(q_embed_loader))
        batch_mixed = next(iter(mixed_embed_loader))

        z_p, x_p     = _unpack_loader_batch(batch_p)
        z_q, x_q     = _unpack_loader_batch(batch_q)
        z_m, x_mixed = _unpack_loader_batch(batch_mixed)

        # flatten embeddings if needed
        if z_p.ndim > 2: z_p = z_p.view(z_p.size(0), -1)
        if z_q.ndim > 2: z_q = z_q.view(z_q.size(0), -1)
        if z_m.ndim > 2: z_m = z_m.view(z_m.size(0), -1)

        # move to model device/dtype
        z_p = z_p.to(device=device, dtype=dtype)
        z_q = z_q.to(device=device, dtype=dtype)
        z_m = z_m.to(device=device, dtype=dtype)

        # forward
        g_p     = model(z_p).squeeze(-1)
        g_q     = model(z_q).squeeze(-1)
        g_mixed = model(z_m).squeeze(-1)

    return g_p, g_q, g_mixed, x_p, x_q, x_mixed




# Classifier density-ratio baseline
# ---------------------------------

def density_ratio_classifier(Xp, Xq, X_eval=None, balance_priors=True, C=1.0, random_state=0):
    """
    Estimate r(x)=p(x)/q(x) via binary classification and Bayes' rule.
    Xp: (n_p, d) samples from p
    Xq: (n_q, d) samples from q
    X_eval: points to evaluate r(x) on (defaults to stacked [Xp; Xq])
    balance_priors: if True, use effective priors π0=π1=0.5 (recommended)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV

    Xp, Xq = np.asarray(Xp), np.asarray(Xq)
    X = np.vstack([Xp, Xq])
    y = np.hstack([np.ones(len(Xp)), np.zeros(len(Xq))])

    # Effective priors
    if balance_priors:
        pi1, pi0 = 0.5, 0.5
        # balance by weighting the loss (keeps all samples)
        w = np.where(y == 1, len(X) / (2*len(Xp)), len(X) / (2*len(Xq)))
    else:
        pi1, pi0 = len(Xp) / len(X), len(Xq) / len(X)
        w = np.ones_like(y)

    base = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, random_state=random_state)
    # Calibrate to improve probability quality (important for ratios)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X, y, sample_weight=w)

    if X_eval is None:
        X_eval = X

    # Posterior η(x) = P(Y=1|x)
    eta = clf.predict_proba(X_eval)[:, 1]
    # clip to avoid division issues
    eps = 1e-12
    eta = np.clip(eta, eps, 1 - eps)

    # r(x) = (pi0/pi1) * eta/(1-eta)
    r = (pi0 / pi1) * (eta / (1.0 - eta))
    return r, eta, {"pi1": pi1, "pi0": pi0, "clf": clf}




# Numeric summaries and score selection
# -------------------------------------

def summarize_vector(vec):
    """
    Compute summary statistics of a numeric vector.

    Parameters
    ----------
    vec : array-like
        Input 1D vector.

    Returns
    -------
    stats : dict
        Dictionary of summary statistics: mean, std, min, max, median, q1, q3, length.
    """
    arr = np.asarray(vec).reshape(-1)

    stats = {
        "length": arr.size,
        "mean": np.mean(arr),
        "std": np.std(arr, ddof=1),   # sample standard deviation
        "min": np.min(arr),
        "q1": np.percentile(arr, 25),
        "median": np.median(arr),
        "q3": np.percentile(arr, 75),
        "max": np.max(arr),
    }
    return stats


def summarize_by_group(a, b):
    """
    Summarize a continuous vector 'a' by levels of a grouping vector 'b'.

    Parameters
    ----------
    a : array-like
        Continuous numeric vector.
    b : array-like
        Group labels of the same length as a.

    Returns
    -------
    summary : pandas.DataFrame
        Summary statistics of 'a' for each level of 'b'.
    """
    import pandas as pd

    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    if a.shape[0] != b.shape[0]:
        raise ValueError("a and b must have the same length.")

    df = pd.DataFrame({"a": a, "b": b})
    summary = df.groupby("b")["a"].agg(
        length="count",
        mean="mean",
        std=lambda x: np.std(x, ddof=1),
        min="min",
        q1=lambda x: np.percentile(x, 25),
        median="median",
        q3=lambda x: np.percentile(x, 75),
        max="max"
    )
    return summary


def compare_l2(true_vec, candidates):
    """
    Compute L2 distances between a true vector and a list of candidate vectors.

    Parameters
    ----------
    true_vec : array-like, shape (d,)
        The ground truth vector.
    candidates : list of array-like
        List of candidate vectors, each of shape (d,).

    Returns
    -------
    distances : list of floats
        L2 distances between true_vec and each candidate.

    Raises
    ------
    ValueError
        If any element in candidates is not a 1D vector of the same length as true_vec.
    """
    true_vec = np.asarray(true_vec).reshape(-1)
    d = true_vec.shape[0]

    distances = []
    for i, cand in enumerate(candidates):
        cand = np.asarray(cand).reshape(-1)

        if cand.ndim != 1:
            raise ValueError(f"Candidate at index {i} is not a 1D vector.")
        if cand.shape[0] != d:
            raise ValueError(f"Candidate at index {i} has length {cand.shape[0]}, expected {d}.")

        dist = np.linalg.norm(true_vec - cand, ord=2)
        distances.append(dist)

    return distances


def minmax_text_from_idx(t: torch.Tensor, idx, default="—") -> str:
    """
    Safely compute min/max over t[idx]. If empty, return placeholders.
    Works with int, slice, list/tuple, or Tensor indices.
    """
    try:
        vals = t[idx]
    except Exception:
        vals = t.new_empty(0)

    if not isinstance(vals, torch.Tensor) or vals.numel() == 0:
        return f"min={default}, max={default}"
    return f"min={vals.min().item():.3g}, max={vals.max().item():.3g}"


def select_extremes(
    g: torch.Tensor,
    thresh: float = 0.1,
    k_top: int = 25,      # how many of the largest > 2 - thresh
    k_near: int = 25,     # how many closest to 1 within [1-thresh, 1+thresh]
    k_small: int = 25,    # how many smallest < thresh
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Returns three groups (each with 'values' and 'indices' in the ORIGINAL flat indexing):
      1) 'largest':   top-k values where g > 2 - thresh (sorted desc)
      2) 'near_one':  k values closest to 1 within [1 - thresh, 1 + thresh] (sorted by |x-1|)
      3) 'smallest':  k smallest values where g < thresh (sorted asc)
    """
    x = g.reshape(-1)
    dev = x.device

    # --- 1) Largest ones > 2 - thresh ---
    cutoff_hi = 2.0 - thresh
    mask_hi = x > cutoff_hi
    if mask_hi.any():
        vals_hi = x[mask_hi]
        idx_hi_all = mask_hi.nonzero(as_tuple=False).squeeze(1)
        k = min(k_top, vals_hi.numel())
        top_vals, order_in_vals = torch.topk(vals_hi, k, largest=True)
        top_idx = idx_hi_all[order_in_vals]
    else:
        top_vals = x.new_empty((0,))
        top_idx  = torch.empty(0, dtype=torch.long, device=dev)

    # --- 2) Closest to 1 within [1 - thresh, 1 + thresh] ---
    lo, hi = 1.0 - thresh, 1.0 + thresh
    mask_mid = (x >= lo) & (x <= hi)
    if mask_mid.any():
        vals_mid = x[mask_mid]
        idx_mid_all = mask_mid.nonzero(as_tuple=False).squeeze(1)
        d = (vals_mid - 1.0).abs()
        k = min(k_near, vals_mid.numel())
        # Get k smallest distances -> use topk on negative distances
        order = torch.topk(-d, k, largest=True).indices
        near_vals = vals_mid[order]
        near_idx  = idx_mid_all[order]
    else:
        near_vals = x.new_empty((0,))
        near_idx  = torch.empty(0, dtype=torch.long, device=dev)

    # --- 3) Smallest ones < thresh ---
    mask_lo = x < thresh
    if mask_lo.any():
        vals_lo = x[mask_lo]
        idx_lo_all = mask_lo.nonzero(as_tuple=False).squeeze(1)
        k = min(k_small, vals_lo.numel())
        # k smallest values -> topk on negative values
        order = torch.topk(-vals_lo, k, largest=True).indices
        small_vals = vals_lo[order]
        small_idx  = idx_lo_all[order]
    else:
        small_vals = x.new_empty((0,))
        small_idx  = torch.empty(0, dtype=torch.long, device=dev)

    return {
        "largest":  {"values": top_vals,   "indices": top_idx},
        "near_one": {"values": near_vals,  "indices": near_idx},
        "smallest": {"values": small_vals, "indices": small_idx},
    }




# Loss and density plots
# ----------------------

def plot_losses(losses, xlabel="Iteration", ylabel="Loss", title="Training Loss", 
                color="tab:blue", figsize=(6,4), ax=None, label=None, logy=False):
    """
    Plot a sequence of losses.

    Parameters
    ----------
    losses : list, np.ndarray, or torch.Tensor
        Sequence of scalar loss values.
    xlabel, ylabel, title : str
        Labels for the axes and the plot title.
    color : str
        Line color.
    figsize : tuple
        Figure size if ax is None.
    ax : matplotlib Axes or None
        If provided, draw into this Axes; otherwise create a new figure.
    label : str or None
        Legend label for the loss curve.
    logy : bool
        If True, use log-scale on the y-axis.
    """
    import matplotlib.pyplot as plt

    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy()
    losses = np.asarray(losses).reshape(-1)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.arange(1, len(losses)+1), losses, color=color, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if label is not None:
        ax.legend()
    return ax


def add_loss_line(losses, ax, color="tab:orange", label=None, logy=False):
    """
    Add another loss curve to an existing loss plot.

    Parameters
    ----------
    losses : list, np.ndarray, or torch.Tensor
        Sequence of scalar loss values.
    ax : matplotlib Axes
        Existing Axes object returned from plot_losses.
    color : str
        Line color for the new curve.
    label : str or None
        Legend label for the new curve.
    logy : bool
        If True, use log-scale on the y-axis.
    """
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy()
    losses = np.asarray(losses).reshape(-1)

    ax.plot(np.arange(1, len(losses)+1), losses, color=color, label=label)
    if logy:
        ax.set_yscale("log")
    if label is not None:
        ax.legend()
    return ax


def plot_1d_density(X, bins=50, bandwidth=None):
    """
    Plot the empirical density of 1D samples.

    Parameters
    ----------
    X : array-like, shape (n,) or (n,1)
        Input samples.
    bins : int, optional
        Number of bins for histogram.
    bandwidth : float or None
        Bandwidth for KDE. If None, scipy picks automatically.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    X = np.asarray(X).reshape(-1)  # flatten to 1D

    # Histogram (empirical density)
    counts, bin_edges = np.histogram(X, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Kernel density estimate
    kde = gaussian_kde(X, bw_method=bandwidth)
    x_grid = np.linspace(X.min() - 1, X.max() + 1, 500)
    kde_vals = kde(x_grid)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(x_grid, kde_vals, label="KDE", color="navy")
    plt.bar(bin_centers, counts, width=(bin_edges[1]-bin_edges[0]),
            alpha=0.3, label="Histogram (density)", color="gray")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Estimated Density of 1D Sample")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_1d_density_compare(
    X, Y, bins=50, bandwidth=None, 
    x_lim = None,
    title="Comparison of 1D Sample Densities",
    labels=("X", "Y"),
    ax=None
):
    """
    Plot and compare the empirical density of two 1D samples.

    Parameters
    ----------
    X, Y : array-like
        Input samples.
    bins : int, optional
        Number of bins for histograms.
    bandwidth : float or None
        Bandwidth for KDE. If None, scipy picks automatically.
    title : str
        Title of the plot.
    labels : tuple of str
        Labels for the two sample sets (default: ("X","Y")).
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, a new figure and axis are created.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    X = np.asarray(X).reshape(-1)
    Y = np.asarray(Y).reshape(-1)

    # Shared x-grid spanning both samples
    if x_lim is None:
        x_min = min(X.min(), Y.min()) - 1
        x_max = max(X.max(), Y.max()) + 1
    else:
        x_min = x_lim[0]
        x_max = x_lim[1]
    x_grid = np.linspace(x_min, x_max, 500)


    # KDE estimates
    kde_X = gaussian_kde(X, bw_method=bandwidth)
    kde_Y = gaussian_kde(Y, bw_method=bandwidth)

    kde_vals_X = kde_X(x_grid)
    kde_vals_Y = kde_Y(x_grid)

    # Histograms (density normalized)
    counts_X, bin_edges_X = np.histogram(X, bins=bins, density=True)
    counts_Y, bin_edges_Y = np.histogram(Y, bins=bins, density=True)

    bin_centers_X = 0.5 * (bin_edges_X[:-1] + bin_edges_X[1:])
    bin_centers_Y = 0.5 * (bin_edges_Y[:-1] + bin_edges_Y[1:])

    # Create axis if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        created_fig = True

    # KDEs
    ax.plot(x_grid, kde_vals_X,  color="navy")
    ax.plot(x_grid, kde_vals_Y,  color="darkred")

    # Histograms
    ax.bar(bin_centers_X, counts_X, width=(bin_edges_X[1]-bin_edges_X[0]),
           alpha=0.3, label=f" {labels[0]}", color="gray", edgecolor="black")
    ax.bar(bin_centers_Y, counts_Y, width=(bin_edges_Y[1]-bin_edges_Y[0]),
           alpha=0.3, label=f" {labels[1]}", color="orange", edgecolor="black")

    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()

    if created_fig:
        plt.tight_layout()
        plt.show()

    return ax




# Ratio plots
# -----------

def plot_theoretical_ratio(x, ratios, label="theoretical ratio", color="purple",
                                    ylim = None, title = None, ax=None):
    """
    Create a standalone plot of a theoretical density ratio r(x) vs. x.
    Sorts (x, ratios) pairs before plotting and opens a new figure.

    Parameters
    ----------
    x : array-like, shape (n,) or (n,1)
        1D sample points.
    ratios : array-like, shape (n,)
        Theoretical density ratio values at each x.
    label : str, optional
        Label for the curve.
    color : str, optional
        Line color.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x).reshape(-1)
    ratios = np.asarray(ratios).reshape(-1)

    if x.shape[0] != ratios.shape[0]:
        raise ValueError("x and ratios must have the same length.")

    if title is None:
        title = "Density Ratio Comparison (1D)"

    # sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    ratios_sorted = ratios[order]

    # prepare axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(x_sorted, ratios_sorted, lw=2, label=label, color=color)
    ax.axhline(1.0, linestyle="--", color="gray", alpha=0.6)
    ax.set_xlabel("x")
    ax.set_ylabel("r(x)")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title)
    ax.legend()

    if ax is None:  # only tighten if standalone
        plt.tight_layout()
        plt.show()

    return ax


def add_density_ratio_line(x, ratios, label="estimated ratio", color=None, ax=None):
    """
    Add another density ratio curve to a plot.

    Parameters
    ----------
    x : array-like, shape (n,) or (n,1)
        1D sample points.
    ratios : array-like, shape (n,)
        Density ratio values at each x.
    label : str, optional
        Label for the curve.
    color : str, optional
        Line color.
    ax : matplotlib.axes.Axes or None
        Axis to plot on. If None, use current axis.
    """
    import matplotlib.pyplot as plt

    x = np.asarray(x).reshape(-1)
    ratios = np.asarray(ratios).reshape(-1)

    if x.shape[0] != ratios.shape[0]:
        raise ValueError("x and ratios must have the same length.")

    order = np.argsort(x)
    x_sorted = x[order]
    ratios_sorted = ratios[order]

    if ax is None:
        ax = plt.gca()

    ax.plot(x_sorted, ratios_sorted, lw=2, label=label, color=color)
    ax.legend()

    return ax


def plot_ratio_hist(ratios, bins=40, range=None, density=True, label="theoretical",
                    color=None, edgecolor="black", alpha=0.35, figsize=(7,4), ax=None):
    """Plot histogram of ratios. If ax is None, make a new figure."""
    import matplotlib.pyplot as plt

    r = np.asarray(ratios).reshape(-1)
    if ax is None: fig, ax = plt.subplots(figsize=figsize)
    counts, bin_edges, _ = ax.hist(r, bins=bins, range=range, density=density,
                                   alpha=alpha, color=color, edgecolor=edgecolor, label=label)
    ax.set_xlabel("ratio r(x)"); ax.set_ylabel("Density" if density else "Count")
    ax.set_title("Histogram of Density Ratios"); ax.legend()
    if 'fig' in locals(): fig.tight_layout()
    return ax, bin_edges


def _infer_bin_edges_from_ax(ax):
    """
    Try to infer bin edges from existing histogram bars on ax.
    Works for standard bar histograms produced by plt.hist.
    """
    import matplotlib.pyplot as plt

    patches = ax.patches
    if not patches:
        return None

    # Collect left/right edges from rectangle patches
    edges = []
    for p in patches:
        x = p.get_x()
        w = p.get_width()
        edges.extend([x, x + w])

    if not edges:
        return None

    # Unique + sorted -> candidate edges
    edges = np.unique(np.round(edges, decimals=12))
    # Ensure strictly increasing and length >= 2
    if edges.size >= 2:
        return edges
    return None


def add_ratio_hist(
    ratios,
    ax=None,
    bin_edges=None,
    density=True,
    label="estimated",
    color=None,
    edgecolor="black",
    alpha=0.35,
):
    """
    Overlay another histogram of ratios on the *current* plot.

    Notes
    -----
    - For perfect bar alignment, pass the `bin_edges` returned by plot_ratio_hist.
    - If `bin_edges` is None, this tries to infer them from the existing axes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()

    r = np.asarray(ratios).reshape(-1)

    # Use provided bin_edges if available; else try to infer from ax
    if bin_edges is None:
        bin_edges = _infer_bin_edges_from_ax(ax)

    if bin_edges is None:
        # Fall back: use numpy's automatic bins (may not align perfectly)
        counts, bin_edges, patches = ax.hist(
            r,
            bins=40,
            density=density,
            alpha=alpha,
            color=color,
            edgecolor=edgecolor,
            label=label,
        )
    else:
        counts, bin_edges, patches = ax.hist(
            r,
            bins=bin_edges,
            density=density,
            alpha=alpha,
            color=color,
            edgecolor=edgecolor,
            label=label,
        )

    ax.set_xlabel("ratio r(x)")
    ax.set_ylabel("Density" if density else "Count")
    ax.legend()
    return ax, bin_edges


def scatter_compare_ratios(ratios, ratios_list, lims = [0,2],labels=None, figsize=(12,4), ax=None):
    """
    Scatter plot(s) comparing theoretical ratios with estimated ratios.

    Parameters
    ----------
    ratios : array-like, shape (n,)
        Theoretical density ratio values.
    ratios_list : list of array-like
        Each element is an estimated density ratio vector (length n).
    labels : list of str or None
        Labels for each estimator. If None, auto-generated as "Estimator i".
    figsize : tuple
        Figure size (only used if ax is None).
    ax : matplotlib.axes.Axes or list of Axes or None
        If None, a new figure and axes are created.
        If a single Axes is provided, ratios_list must have length 1.
        If a list/array of Axes is provided, must match the length of ratios_list.
    """
    import matplotlib.pyplot as plt

    ratios = np.asarray(ratios).reshape(-1)
    n = len(ratios)

    if labels is None:
        labels = [f"Estimator {i+1}" for i in range(len(ratios_list))]

    # Handle axes
    if ax is None:
        fig, axes = plt.subplots(1, len(ratios_list), figsize=figsize, squeeze=False)
        axes = axes[0]
    else:
        if isinstance(ax, plt.Axes):
            if len(ratios_list) != 1:
                raise ValueError("If a single Axes is provided, ratios_list must have length 1.")
            axes = [ax]
            fig = ax.figure
        else:
            if len(ax) != len(ratios_list):
                raise ValueError("Length of ax list must match length of ratios_list.")
            axes = ax
            fig = axes[0].figure

    # Determine common axis limits
   
    if lims is None:
        lims = [ratios.min(), ratios.max()]
        for est in ratios_list:
            lims[0] = min(lims[0], np.min(est))
            lims[1] = max(lims[1], np.max(est))


    # Plot
    for ax_i, est, lab in zip(axes, ratios_list, labels):
        est = np.asarray(est).reshape(-1)
        if est.shape[0] != n:
            raise ValueError("All estimated ratio vectors must have same length as ratios.")
        ax_i.scatter(ratios, est, alpha=0.5, s=10)
        ax_i.plot(lims, lims, "r--", lw=2)  # 45° line
        ax_i.set_xlabel("Theoretical")
        ax_i.set_ylabel("Estimated")
        ax_i.set_title(lab)
        ax_i.set_xlim(lims)
        ax_i.set_ylim(lims)

    if ax is None:  # only tighten if we created the figure
        plt.tight_layout()

    return fig, axes


def twoD_generate_grid_numpy(xmin, xmax, ymin, ymax, xstep=1, ystep=1):
    # create 1D arrays of x and y
    x = np.arange(xmin, xmax + xstep, xstep)
    y = np.arange(ymin, ymax + ystep, ystep)
    # meshgrid gives 2D grids XX, YY
    XX, YY = np.meshgrid(x, y, indexing='xy')
    # stack and reshape into N×2 array of points
    points = np.vstack([XX.ravel(), YY.ravel()]).T
    return points


def twoD_plot_density_ratio(
    Y1, Y2, w_vis_np1, ratios,
    lims=(-20, 20),
    color_range=(-20, 20),
    cmap="coolwarm",
    point_size=10,
    alpha=0.7,
    equal_aspect=True
):
    """
    Plot raw samples, estimated density ratio heatmap, theoretical heatmap,
    and a scatter comparison between estimated and true *log* density ratios.

    Layout: 2 × 3 grid. Inputs must have finite, positive scores for
    every sample; the historical per-group histograms use original indices.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm


    # stack and sanity checks
    stacked_Y = np.vstack((Y1, Y2))
    n1_samples = Y1.shape[0]
    n2_samples = Y2.shape[0]

    w_vis_np1 = np.asarray(w_vis_np1).ravel()
    ratios    = np.asarray(ratios).ravel()
    if stacked_Y.shape[0] != w_vis_np1.size or w_vis_np1.size != ratios.size:
        raise ValueError("Lengths of w_vis_np1/ratios must match stacked_Y rows (len(Y1)+len(Y2)).")

    # mask invalid values
    valid_mask = np.isfinite(w_vis_np1) & np.isfinite(ratios) & (w_vis_np1 > 0) & (ratios > 0)
    stacked_Y = stacked_Y[valid_mask]
    log_w = np.log(w_vis_np1[valid_mask])
    log_r = np.log(ratios[valid_mask])

    # color normalization centered at 0
    vmin, vmax = color_range
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # make 2×2 layout
    fig, axs = plt.subplots(2, 3, figsize=(15,8))

    # --- (0,0) Raw data ---
    ax = axs[0, 0]
    ax.scatter(Y1[:, 0], Y1[:, 1], alpha=0.5, label='p')
    ax.scatter(Y2[:, 0], Y2[:, 1], alpha=0.5, label='q')
    ax.set_title(f'Bivariate Samples: n_p={n1_samples}, n_q={n2_samples}')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    ax.legend()
    if equal_aspect: ax.set_aspect('equal', adjustable='box')

    # --- (0,1) Estimated vs True (log ratios) ---
    ax = axs[0, 1]
    ax.scatter(log_w, log_r, alpha=alpha)
    ax.plot(lims, lims, 'r--', linewidth=2)
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_title('Estimated vs True (log ratios)')
    ax.set_xlabel('Estimated log ratio')
    ax.set_ylabel('True log ratio')

    # --- (0,2) true histogram ---
    ax = axs[0, 2]
    sc2 = ax.hist(log_r[np.arange(n1_samples)], bins=30, 
            alpha=0.6, density=True, label="p")
    ax.hist(log_r[np.arange(n2_samples)+n1_samples], bins=30, 
            alpha=0.6, density=True, label="q")
    ax.set_title('Truth')

    # --- (1,0) Estimated log ratio heatmap ---
    ax = axs[1, 0]
    sc1 = ax.scatter(stacked_Y[:, 0], stacked_Y[:, 1], c=log_w,
                     cmap=cmap, s=point_size, marker='o', norm=norm)
    ax.set_title('Estimated: log(p) - log(q)')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    if equal_aspect: ax.set_aspect('equal', adjustable='box')
    cbar1 = fig.colorbar(sc1, ax=ax); cbar1.set_label('log ratio')

    # --- (1,1) True log ratio heatmap ---
    ax = axs[1, 1]
    sc2 = ax.scatter(stacked_Y[:, 0], stacked_Y[:, 1], c=log_r,
                     cmap=cmap, s=point_size, marker='o', norm=norm)
    ax.set_title('Theoretical: log(p/q)')
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    if equal_aspect: ax.set_aspect('equal', adjustable='box')
    cbar2 = fig.colorbar(sc2, ax=ax); cbar2.set_label('log ratio')


    # --- (1,2) hellinger histogram ---
    ax = axs[1, 2]
    sc2 = ax.hist(log_w[np.arange(n1_samples)], bins=30, 
            alpha=0.6, density=True, label="p")
    ax.hist(log_w[np.arange(n2_samples)+n1_samples], bins=30, 
            alpha=0.6, density=True, label="q")
    ax.set_title('Hellinger')
    

    plt.tight_layout()
    plt.show()
    return fig, axs


def plot_calibration_cells(
    neural_mean, estimate, lower, upper, *, counts_p=None, counts_q=None,
    masses=None, ax=None, title=None,
):
    """Plot cell-average calibration intervals against mean neural RDR scores.

    All inputs are one-dimensional arrays of equal length. The intervals concern
    population cell-average RDR, not individual ratios or uncertainty in the
    neural means on the horizontal axis. Callers supply the desired confidence
    procedure and its multiplicity adjustment; no true densities are required.

    Nonfinite/out-of-range means and estimates are omitted. If both count arrays
    are supplied, cells with no calibration observations have no estimate marker,
    but retain a supplied interval when their neural mean is available. Marker
    areas increase with ``masses`` when supplied, otherwise with total counts;
    without either they have equal area. Return the Matplotlib axes.
    """
    import matplotlib.pyplot as plt

    def vector(values, name, length=None):
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or (length is not None and len(values) != length):
            raise ValueError(f"{name} must be a one-dimensional array of matching length")
        return values

    means = vector(neural_mean, "neural_mean")
    n_cells = len(means)
    centers = vector(estimate, "estimate", n_cells)
    lows = vector(lower, "lower", n_cells)
    highs = vector(upper, "upper", n_cells)
    populated = np.ones(n_cells, dtype=bool)
    weights = np.ones(n_cells)
    if (counts_p is None) != (counts_q is None):
        raise ValueError("Provide both counts_p and counts_q, or neither")
    if counts_p is not None:
        p = vector(counts_p, "counts_p", n_cells)
        q = vector(counts_q, "counts_q", n_cells)
        if any(np.any(~np.isfinite(v) | (v < 0) | (v != np.floor(v))) for v in (p, q)):
            raise ValueError("Calibration counts must be finite nonnegative integers")
        weights = p + q
        populated = weights > 0
    if masses is not None:
        weights = vector(masses, "masses", n_cells)
        if np.any(~np.isfinite(weights) | (weights < 0)):
            raise ValueError("Cell masses must be finite and nonnegative")

    valid_means = np.isfinite(means) & (means >= 0) & (means <= 2)
    valid_intervals = valid_means & np.isfinite(lows) & np.isfinite(highs) & (lows <= highs)
    valid_points = valid_means & populated & np.isfinite(centers) & (centers >= 0) & (centers <= 2)
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.plot([0, 2], [0, 2], color="0.55", linestyle="--", linewidth=1, label="Agreement")
    if np.any(valid_intervals):
        midpoint = (lows[valid_intervals] + highs[valid_intervals]) / 2
        radius = (highs[valid_intervals] - lows[valid_intervals]) / 2
        ax.errorbar(means[valid_intervals], midpoint, yerr=radius, fmt="none",
                    capsize=3, color="tab:blue", alpha=0.65, label="Cell-average CI")
    if np.any(valid_points):
        point_weights = weights[valid_points]
        scale = float(point_weights.max())
        sizes = 30 + 80 * point_weights / scale if scale > 0 else np.full(len(point_weights), 30.)
        ax.scatter(means[valid_points], centers[valid_points], s=sizes,
                   color="tab:blue", edgecolors="white", linewidths=0.6,
                   zorder=3, label="Calibration estimate")
    if not np.any(valid_means):
        ax.text(0.5, 0.08, "No cells with valid neural means", ha="center", transform=ax.transAxes)
    ax.set(xlim=(0, 2), ylim=(0, 2), xlabel="Mean neural RDR in cell",
           ylabel="Calibrated cell-average RDR")
    ax.set_aspect("equal", adjustable="box")
    if title is not None:
        ax.set_title(title)
    ax.legend(loc="best", fontsize="small")
    return ax
