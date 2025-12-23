import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
import pandas as pd
import torch
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


# -------- use binary classification and Bayes rule to learn  RDR ---------- #
def density_ratio_classifier(Xp, Xq, X_eval=None, balance_priors=True, C=1.0, random_state=0):
    """
    Estimate r(x)=p(x)/q(x) via binary classification and Bayes' rule.
    Xp: (n_p, d) samples from p
    Xq: (n_q, d) samples from q
    X_eval: points to evaluate r(x) on (defaults to stacked [Xp; Xq])
    balance_priors: if True, use effective priors π0=π1=0.5 (recommended)
    """
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

# ----------20D sim helpers ----------

def make_orthonormal_K(D=20, d=2, seed=0):
    rng = np.random.default_rng(seed)
    G = rng.normal(size=(D, d))          # 20 x 2
    Q, _ = np.linalg.qr(G, mode='reduced')
    # Q has orthonormal columns; shape (20, 2)
    return Q

# --- 2) project 2D points into 20D with isotropic Gaussian noise
def lift_to_20D(Y2, K, noise_sd=0.1, seed=1):
    """
    Y2: (n, 2) points
    K:  (20, 2) with orthonormal columns
    returns Y20: (n, 20)
    """
    rng = np.random.default_rng(seed)
    n = Y2.shape[0]
    eps = rng.normal(scale=noise_sd, size=(n, K.shape[0]))  # (n, 20)
    # Y20 = Y2 @ K^T + noise
    Y20 = Y2 @ K.T + eps
    return Y20
def mvn_logpdf(x, mean, cov):
    # x: (n,2), mean: (2,), cov: (2,2)
    L = np.linalg.cholesky(cov)
    diff = x - mean
    # solve (L L^T)^{-1} diff via triangular solves
    sol = np.linalg.solve(L, diff.T)       # (2,n)
    quad = np.sum(sol**2, axis=0)          # (n,)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (quad + logdet + 2*np.log(2*np.pi))

def logsumexp(arr, axis=0):
    m = np.max(arr, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(arr - m), axis=axis, keepdims=True))).squeeze(axis)

def mixture_logpdf(x, weights, means, covs):
    # weights: (K,), means: list/array of K (2,), covs: list/array of K (2,2)
    logs = []
    for w, mu, S in zip(weights, means, covs):
        logs.append(np.log(w) + mvn_logpdf(x, mu, S))
    logs = np.vstack(logs)                 # (K, n)
    return logsumexp(logs, axis=0)         # (n,)

# ---------- theoretical 20D ratio ----------
def RDR_20D_theoretical(X20, K, weights,
                          means_p, covs_p, means_q, covs_q,
                          sigma):
    """
    X20: (n,20)
    K: (20,2) with K.T @ K = I2
    weights: (Kmix,)
    means_*(list): length Kmix of 2D means for p and q
    covs_*(list): length Kmix of 2x2 covs for p and q
    sigma: noise sd in 20D lift (epsilon ~ N(0, sigma^2 I_20))
    """
    # project to 2D; since K has orthonormal columns, K^T x == x @ K
    z = X20 @ K                              # (n,2)
    I2 = np.eye(2) * (sigma**2)

    # convolved 2D mixtures: Σ -> Σ + σ^2 I_2
    covs_p_conv = [S + I2 for S in covs_p]
    covs_q_conv = [S + I2 for S in covs_q]

    logp = mixture_logpdf(z, weights, means_p, covs_p_conv)
    logq = mixture_logpdf(z, weights, means_q, covs_q_conv)

    p1 = np.exp(logp)
    p2 = np.exp(logq)

    ratios = 2*p1/(p1+p2)
    # return np.exp(logp - logq)              # (n,)
    return ratios

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
    
def sample_mixture_normal_2d(n_samples, weights, means, covariances):
        # Generate samples from the mixture
    n_components = len(weights)
    samples = []
    for _ in range(n_samples):
        # Choose a component based on the weights
        component = np.random.choice(n_components, p=weights)
        # Generate a sample from the chosen component
        sample = np.random.multivariate_normal(means[component], covariances[component])
        samples.append(sample)

    samples = np.array(samples)

    return samples

def sample_mixture_normals(
    n,
    weights,
    means,
    covs,
    rng=None,
):
    """
    Draw n samples from a Gaussian Mixture in 1D or 2D.

    Parameters
    ----------
    n : int
        Number of samples.
    weights : array-like, shape (K,)
        Mixing weights, nonnegative and summing to 1 (will be normalized if not).
    means : array-like
        If 1D: shape (K,) or (K,1)
        If 2D: shape (K,2)
    covs : array-like
        If 1D:
            - shape (K,) for scalar variances, OR
            - shape (K,1,1) for 1x1 cov matrices.
        If 2D:
            - shape (K,2) for diagonal variances, OR
            - shape (K,2,2) for full cov matrices.
    rng : np.random.Generator or int or None
        Random seed or Generator. If None, uses default Generator.

    Returns
    -------
    X : ndarray, shape (n, d)
        Samples. Always 2D (n,1) if d=1, (n,2) if d=2.
    z : ndarray, shape (n,)
        Component indices in {0, …, K-1}.
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0)
    if weights.sum() <= 0:
        raise ValueError("All weights are zero or negative.")
    weights = weights / weights.sum()

    means = np.asarray(means, dtype=float)
    # Infer dimension d and normalize mean shape
    if means.ndim == 1:           # (K,) -> 1D
        K = means.shape[0]
        d = 1
        means_2d = means.reshape(K, 1)
    elif means.ndim == 2:
        K, d = means.shape
        if d not in (1, 2):
            raise ValueError("Only d=1 or d=2 supported.")
        means_2d = means.reshape(K, d)
    else:
        raise ValueError("means must be shape (K,) or (K,d).")

    covs = np.asarray(covs, dtype=float)

    # Normalize covariance shapes
    if d == 1:
        if covs.ndim == 1:
            covs_ = covs.reshape(K, 1, 1)
        elif covs.shape == (K, 1, 1):
            covs_ = covs
        else:
            raise ValueError("For d=1, covs must be (K,) or (K,1,1).")
        if np.any(covs_[:, 0, 0] <= 0):
            raise ValueError("All variances must be positive.")
    else:  # d == 2
        if covs.ndim == 2 and covs.shape == (K, 2):
            if np.any(covs <= 0):
                raise ValueError("All diagonal variances must be positive.")
            covs_ = np.zeros((K, 2, 2), dtype=float)
            covs_[:, 0, 0] = covs[:, 0]
            covs_[:, 1, 1] = covs[:, 1]
        elif covs.shape == (K, 2, 2):
            covs_ = covs
        else:
            raise ValueError("For d=2, covs must be (K,2) or (K,2,2).")
        for k in range(K):
            if not np.allclose(covs_[k], covs_[k].T, atol=1e-10):
                raise ValueError(f"covs[{k}] must be symmetric.")
            try:
                np.linalg.cholesky(covs_[k])
            except np.linalg.LinAlgError:
                raise ValueError(f"covs[{k}] is not positive definite.")

    # Draw component indices
    z = rng.choice(K, size=n, p=weights)

    # Sample per-component
    X = np.empty((n, d), dtype=float)
    for k in range(K):
        idx = np.where(z == k)[0]
        if idx.size == 0:
            continue
        if d == 1:
            mu = means_2d[k, 0]
            var = covs_[k, 0, 0]
            X[idx, 0] = rng.normal(loc=mu, scale=np.sqrt(var), size=idx.size)
        else:
            mu = means_2d[k]
            L = np.linalg.cholesky(covs_[k])
            eps = rng.normal(size=(idx.size, d))
            X[idx] = mu + eps @ L.T

    return X, z



def twoD_mixture_density(x, weights, means, covariances):
    """
    Compute the density of a weighted 2D Gaussian mixture at x.
    Works for x shape (2,) or (N, 2). Returns scalar or (N,) accordingly.
    """
    x = np.asarray(x, dtype=float)
    one_point = (x.ndim == 1)
    X = x.reshape(1, 2) if one_point else x  # (N,2)
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)            # (K,2)
    covariances = np.asarray(covariances, dtype=float) # (K,2,2)

    N = X.shape[0]
    d = 2
    total = np.zeros(N, dtype=float)

    for w, mu, Sigma in zip(weights, means, covariances):
        # inv and log|Sigma| for numerical stability
        inv = np.linalg.inv(Sigma)
        sign, logabsdet = np.linalg.slogdet(Sigma)
        if sign <= 0:
            raise ValueError("Covariance matrix must be positive definite.")
        diff = X - mu  # (N,2)
        # Mahalanobis distance for each row
        mahal = np.einsum('ni,ij,nj->n', diff, inv, diff)  # (N,)
        norm_const = np.sqrt((2 * np.pi) ** d * np.exp(logabsdet))
        comp = np.exp(-0.5 * mahal) / norm_const
        total += w * comp

    return total[0] if one_point else total

def twoD_density_ratio(x,weights,means1,means2,covariances1,covariances2):
    """
    Compute r(x) = p1(x) / p2(x) with non-uniform weights.
    """
    p1 = twoD_mixture_density(x, weights, means1, covariances1)
    p2 = twoD_mixture_density(x, weights, means2, covariances2)
    return p1 / p2

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

    Layout: 2 × 2 grid.
    """

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


def _prep_gmm_params(weights, means, covariances):
    """Normalize shapes and basic checks. Returns (K, d, means2d, covs3d)."""
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be shape (K,)")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    if w.sum() == 0:
        raise ValueError("at least one weight must be positive")
    w = w / w.sum()

    means = np.asarray(means, dtype=float)
    if means.ndim == 1:  # (K,) → 1D
        K = means.shape[0]
        d = 1
        means2d = means.reshape(K, 1)
    elif means.ndim == 2:  # (K,d)
        K, d = means.shape
        if d not in (1, 2):
            raise ValueError("Only 1D or 2D supported for means.")
        means2d = means.reshape(K, d)
    else:
        raise ValueError("means must be (K,) or (K,d).")

    covs = np.asarray(covariances, dtype=float)
    if d == 1:
        if covs.ndim == 1 and covs.shape == (K,):
            covs3d = covs.reshape(K, 1, 1)
        elif covs.shape == (K, 1, 1):
            covs3d = covs
        else:
            raise ValueError("For 1D, covariances must be (K,) or (K,1,1).")
        if np.any(covs3d[:, 0, 0] <= 0):
            raise ValueError("All 1D variances must be positive.")
    else:  # d == 2
        if covs.ndim == 2 and covs.shape == (K, 2):
            # diagonal variances
            if np.any(covs <= 0):
                raise ValueError("All diagonal variances must be positive.")
            covs3d = np.zeros((K, 2, 2), dtype=float)
            covs3d[:, 0, 0] = covs[:, 0]
            covs3d[:, 1, 1] = covs[:, 1]
        elif covs.shape == (K, 2, 2):
            covs3d = covs
        else:
            raise ValueError("For 2D, covariances must be (K,2) or (K,2,2).")
        # symmetry + PD check
        for k in range(K):
            if not np.allclose(covs3d[k], covs3d[k].T, atol=1e-10):
                raise ValueError(f"covariances[{k}] must be symmetric.")
            try:
                np.linalg.cholesky(covs3d[k])
            except np.linalg.LinAlgError:
                raise ValueError(f"covariances[{k}] is not positive definite.")

    return w, K, d, means2d, covs3d

def mixture_density(x, weights, means, covariances):
    """
    Density p(x) of a Gaussian mixture at x. Supports 1D and 2D.
    - x: shape (d,) or (n,d) with d in {1,2}
    - returns: scalar if x is a single point, else array (n,)
    """
    w, K, d, M, S = _prep_gmm_params(weights, means, covariances)

    X = np.asarray(x, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != d:
        # allow passing 1D as (n,) or scalar
        if d == 1 and X.shape[1] != 1 and X.ndim == 2:
            X = X.reshape(-1, 1)
        if X.shape[1] != d:
            raise ValueError(f"x must have dimension d={d} (got {X.shape[1]}).")

    n = X.shape[0]
    dens = np.zeros(n, dtype=float)

    if d == 1:
        # vectorized 1D formula
        # For each component k: N(x | mu_k, var_k)
        mus = M[:, 0]                    # (K,)
        vars_ = S[:, 0, 0]               # (K,)
        inv_vars = 1.0 / vars_
        norm_consts = np.sqrt(2 * np.pi * vars_)  # (K,)

        # Broadcast: (n,1) - (K,) -> (n,K)
        diff = X[:, [0]] - mus[None, :]
        exponents = -0.5 * (diff**2) * inv_vars[None, :]
        comp = w[None, :] * np.exp(exponents) / norm_consts[None, :]
        dens = comp.sum(axis=1)
    else:
        # 2D: pre-compute inverses and normalizing constants per component
        invS = np.zeros_like(S)
        detS = np.zeros(K, dtype=float)
        for k in range(K):
            invS[k] = np.linalg.inv(S[k])
            detS[k] = np.linalg.det(S[k])
        norm_consts = np.sqrt(((2 * np.pi) ** d) * detS)  # (K,)

        # For each component, compute Mahalanobis distances for all X
        for k in range(K):
            diff = X - M[k]                           # (n,2)
            quad = np.einsum('ni,ij,nj->n', diff, invS[k], diff)  # (n,)
            comp = w[k] * np.exp(-0.5 * quad) / norm_consts[k]
            dens += comp

    return dens[0] if dens.size == 1 else dens

def density_ratio(x, weights, means1, means2, covariances1, covariances2, 
                    eps=0.0, mixed = False):
    """
    r(x) = p1(x) / p2(x), sharing the same weights by default (like your original).
    - If you need different weights for p1 and p2, call mixture_density twice with
      different weight vectors instead.
    """
    p1 = mixture_density(x, weights, means1, covariances1)
    p2 = mixture_density(x, weights, means2, covariances2)
    if np.isscalar(p1):
        

        if mixed:
            denom = p1+p2
            return 2*p1/denom
        else:
            denom = p2 if p2 > eps else eps
            return p1 / denom
    else:

        if mixed:
            denom = np.maximum(p1+p2, eps)
            return 2*p1/denom
        else:
            denom = np.maximum(p2, eps)
            return p1 / denom
        

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

def summarize_by_group(values, groups):
    values = np.asarray(values).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if values.shape[0] != groups.shape[0]:
        raise ValueError("values and groups must be the same length")

    summary = {}
    for g in np.unique(groups):
        summary[g] = summarize_vector(values[groups == g])
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

def summarize_by_group(a, b):
    """
    Summarize a continuous vector 'a' by levels of a binary vector 'b'.

    Parameters
    ----------
    a : array-like
        Continuous numeric vector.
    b : array-like
        Binary vector (0/1 or True/False) of the same length as a.

    Returns
    -------
    summary : pandas.DataFrame
        Summary statistics of 'a' for each level of 'b'.
    """
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

# ---------------- 1D Beta example ------------------------ #

# ---- small utilities ----
def _as_rng(rng):
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

def _betaln(a, b):
    """Vectorized log Beta(a,b) without SciPy; robust to scalar or array inputs."""
    import math
    lgamma = np.frompyfunc(math.lgamma, 1, 1)  # returns object arrays
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    # Ensure array outputs, then cast to float dtype
    A = np.asarray(lgamma(a)).astype(float)
    B = np.asarray(lgamma(b)).astype(float)
    C = np.asarray(lgamma(a + b)).astype(float)
    return A + B - C
def _beta_pdf(x, a, b, clip=1e-12):
    """Vectorized Beta(a,b) PDF on (0,1); returns 0 outside [0,1]."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)

    # valid support mask
    m = (x >= 0.0) & (x <= 1.0)
    if not np.any(m):
        return out

    x_safe = np.clip(x[m], clip, 1.0 - clip)
    log_pdf = (a - 1.0) * np.log(x_safe) + (b - 1.0) * np.log1p(-x_safe) - _betaln(a, b)
    out[m] = np.exp(log_pdf)
    return out

def _prep_beta_mix_params(weights, alphas, betas=None):
    """
    Normalize/validate Beta-mixture parameters.
    Supports:
      - alphas: shape (K,), betas: shape (K,)  OR
      - alphas: shape (K,2) with (alpha_k, beta_k) pairs and betas=None
    Returns normalized (w, K, a, b)
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1D of shape (K,).")
    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        raise ValueError("All weights are zero or negative.")
    w = w / w.sum()
    K = w.size

    a = np.asarray(alphas, dtype=float)
    if betas is None:
        if a.ndim != 2 or a.shape != (K, 2):
            raise ValueError("If betas is None, alphas must be shape (K,2) with (alpha, beta) pairs.")
        a, b = a[:, 0], a[:, 1]
    else:
        b = np.asarray(betas, dtype=float)
        if a.shape != (K,) or b.shape != (K,):
            raise ValueError("alphas and betas must both be shape (K,).")

    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("All alpha and beta parameters must be strictly positive.")

    return w, K, a, b

# ---- densities ----
def mixture_beta_density(x, weights, alphas, betas=None):
    """
    Density p(x) of a Beta mixture on [0,1].
    - x: scalar, (n,), or (n,1)
    - weights: (K,)
    - alphas: (K,) with betas=(K,), OR (K,2) with (alpha,beta) pairs and betas=None
    Returns: scalar if x is scalar/shape (1,), else array (n,)
    """
    w, K, a, b = _prep_beta_mix_params(weights, alphas, betas)

    X = np.asarray(x, dtype=float)
    scalar_input = False
    if X.ndim == 0:
        X = X.reshape(1)
        scalar_input = True
    elif X.ndim == 2 and X.shape[1] == 1:
        X = X.reshape(-1)
    elif X.ndim != 1:
        raise ValueError("x must be scalar, (n,), or (n,1).")

    # sum_k w_k * BetaPDF(x | a_k, b_k)
    dens = np.zeros_like(X, dtype=float)
    for k in range(K):
        dens += w[k] * _beta_pdf(X, a[k], b[k])

    if scalar_input or dens.size == 1:
        return float(dens[0])
    return dens

def beta_density_ratio(
    x,
    weights,
    alphas1, betas1=None,
    alphas2=None, betas2=None,
    *,
    mixed=False,
    eps=1e-300,
    weights2=None
):
    """
    r(x) = p1(x) / p2(x) by default, or r_mixed(x) = 2*p1(x)/(p1(x)+p2(x)) if mixed=True.

    Parameters
    ----------
    x : scalar, (n,), or (n,1)
    weights : (K,)    weights for p1 (and p2 if weights2 is None)
    alphas1, betas1 : params for p1 (see mixture_beta_density)
    alphas2, betas2 : params for p2 (if None, defaults to p1)
    mixed : bool
        If True, return 2*p1/(p1+p2). Otherwise p1/p2 with eps floor.
    eps : float
        Numerical floor for denominators.
    weights2 : (K,) or None
        Optional different weights for p2.

    Returns
    -------
    r : scalar or (n,)
    """
    p1 = mixture_beta_density(x, weights, alphas1, betas1)

    if alphas2 is None and betas2 is None and weights2 is None:
        # default: compare p1 with itself -> ratio 1 or 1 in mixed form -> 1
        p2 = p1
    else:
        w2 = weights if weights2 is None else weights2
        p2 = mixture_beta_density(x, w2, alphas2, betas2)

    p1_arr = np.asarray(p1, dtype=float)
    p2_arr = np.asarray(p2, dtype=float)

    if mixed:
        denom = np.maximum(p1_arr + p2_arr, eps)
        r = 2.0 * p1_arr / denom
    else:
        denom = np.maximum(p2_arr, eps)
        r = p1_arr / denom

    if np.ndim(p1) == 0 and np.ndim(p2) == 0:
        return float(r)
    return r


def sample_mixture_betas(
    n,
    weights,
    alphas,
    betas=None,
    rng=None,
):
    """
    Draw n samples from a mixture of K independent Beta(a_k, b_k) distributions on [0,1].

    Parameters
    ----------
    n : int
        Number of samples.
    weights : array-like, shape (K,)
        Mixing weights, nonnegative and summing to 1 (will be normalized if not).
    alphas : array-like
        Either shape (K,) giving alpha_k, or shape (K,2) giving (alpha_k, beta_k) pairs.
    betas : array-like or None
        If provided, shape (K,) giving beta_k. If None, `alphas` must be shape (K,2).
    rng : np.random.Generator or int or None
        Random seed or Generator. If None, uses default Generator.

    Returns
    -------
    X : ndarray, shape (n, 1)
        Samples in [0, 1].
    z : ndarray, shape (n,)
        Component indices in {0, …, K-1}.
    """
    # RNG
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    # Weights
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0)
    if w.sum() <= 0:
        raise ValueError("All weights are zero or negative.")
    w = w / w.sum()
    K = w.size

    # Parameters
    a = np.asarray(alphas, dtype=float)
    if betas is None:
        # Expect alphas as (K,2) -> columns are (alpha, beta)
        if a.ndim != 2 or a.shape != (K, 2):
            raise ValueError("If betas is None, `alphas` must be shape (K,2) with (alpha, beta) pairs.")
        a, b = a[:, 0], a[:, 1]
    else:
        b = np.asarray(betas, dtype=float)
        if a.shape != (K,) or b.shape != (K,):
            raise ValueError("alphas and betas must both be shape (K,).")

    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("All alpha and beta parameters must be strictly positive.")

    # Draw component indices
    z = rng.choice(K, size=n, p=w)

    # Sample per-component
    X = np.empty((n, 1), dtype=float)
    for k in range(K):
        idx = np.where(z == k)[0]
        if idx.size == 0:
            continue
        X[idx, 0] = rng.beta(a[k], b[k], size=idx.size)

    return X, z
