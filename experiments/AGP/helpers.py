import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import torch
from collections import defaultdict, deque
import re

# ------------------------ LT transform ------------------------ #
def parse_node_int(label):
    # Extract trailing integer from strings like "Node_615"
    m = re.search(r'(\d+)$', str(label))
    return int(m.group(1)) if m else None

def build_topology_from_child_df(child_df, taxa_names=None):
    """
    child_df: pandas.DataFrame with index = parent labels (e.g., "Node_615")
              and columns ["Child.1", "Child.2"] = child labels
    taxa_names: optional list/array of leaf (tip) labels in the order you want for the transform
                (must match labels used in child_df). If None, use natural order Node_1..Node_K.

    Returns:
      child_l, child_r : np.int64 arrays of length M (internal nodes)
      leaf_order       : np.array of leaf labels (length K)
      internal_order   : np.array of internal labels (length M) in topological order
      label_to_index   : dict mapping every node label -> global index [0..K+M-1]
    """
    # Parents and children as labels
    parents = list(child_df.index.astype(str))
    ch1 = child_df.iloc[:, 0].astype(str).tolist()
    ch2 = child_df.iloc[:, 1].astype(str).tolist()

    # Universe of nodes
    all_nodes = set(parents) | set(ch1) | set(ch2)

    # Leaves = nodes that never appear as parent
    leaves = sorted(all_nodes - set(parents),
                    key=lambda s: parse_node_int(s) if parse_node_int(s) is not None else s)

    # Optionally force leaf order to provided taxa_names (recommended!)
    if taxa_names is not None:
        taxa_names = [str(t) for t in taxa_names]
        leaf_set = set(leaves)
        if set(taxa_names) != leaf_set:
            missing = leaf_set - set(taxa_names)
            extra   = set(taxa_names) - leaf_set
            raise ValueError(
                "taxa_names must be exactly the leaf labels.\n"
                f"Missing in taxa_names: {sorted(missing)}\n"
                f"Extra in taxa_names:   {sorted(extra)}"
            )
        leaf_order = np.array(taxa_names, dtype=str)
    else:
        # default: Node_1, Node_2, ... ordering
        leaf_order = np.array(leaves, dtype=str)

    # Build adjacency (parent -> [children])
    children_of = {p: [c1, c2] for p, c1, c2 in zip(parents, ch1, ch2)}

    # Topological sort for internal nodes so children come first
    indeg = defaultdict(int)
    for p, (c1, c2) in children_of.items():
        # Only count edges into nodes that are internal (i.e., appear as parent somewhere)
        if c1 in children_of: indeg[c1] += 1
        if c2 in children_of: indeg[c2] += 1
        if p not in indeg: indeg[p] += 0  # ensure presence

    # Kahn's algorithm over the internal subgraph
    q = deque([u for u in children_of if indeg[u] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in children_of[u]:
            if v in children_of:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

    if len(topo) != len(children_of):
        # fallback: still try a deterministic order (shouldn't happen if the graph is a proper tree)
        raise RuntimeError("Internal-node graph is not a DAG or is disconnected. Check the input.")

    internal_order = np.array(topo, dtype=str)

    # Assign global indices: leaves 0..K-1 in leaf_order; internals K..K+M-1 in internal_order
    K = len(leaf_order)
    M = len(internal_order)
    label_to_index = {lab: i for i, lab in enumerate(leaf_order)}
    label_to_index.update({lab: K + i for i, lab in enumerate(internal_order)})

    # Build child index arrays aligned with internal_order
    child_l = np.empty(M, dtype=np.int64)
    child_r = np.empty(M, dtype=np.int64)
    for j, parent in enumerate(internal_order):
        c1, c2 = children_of[parent]
        try:
            child_l[j] = label_to_index[c1]
            child_r[j] = label_to_index[c2]
        except KeyError as e:
            raise KeyError(f"Child label {e} not found in label_to_index. "
                           "Are you mixing label spaces?") from e

    # Sanity: root should be the last internal after topological sort
    # (parent appears after its children)
    # Not strictly required, but nice to have.
    return child_l, child_r, leaf_order, internal_order, label_to_index

def build_indexer(out_names, taxa_names):
    # map original table order -> tip order
    name_to_out = {nm: i for i, nm in enumerate(out_names)}
    indexer = np.array([name_to_out[nm] for nm in taxa_names], dtype=int)
    return indexer

# 4) Forward transform using that indexer (adapted from earlier)
def lt_forward_with_indexer_safe(X, child_l, child_r, indexer, eps=0.0):
    # reorder leaves to tree order
    Xo = X[:, indexer]                      # shape: [N, K]
    N, K = Xo.shape
    J = len(child_l)

    # node sums in "absolute index" space: 0..K-1 are leaves, K..K+J-1 are internals
    S = np.zeros((N, K + J))
    S[:, :K] = Xo

    Z = np.zeros((N, J))
    for j in range(J):
        L = child_l[j]; R = child_r[j]
        left  = S[:, L]
        right = S[:, R]

        # if both sides are zero, ratio is undefined but irrelevant (parent mass is zero)
        mask = (left + right) == 0
        z = np.empty(N); z.fill(0.0)                # any number works; 0 is convenient
        # otherwise compute log-ratio with optional epsilon
        z[~mask] = np.log((left[~mask] + eps) / (right[~mask] + eps))
        Z[:, j] = z

        S[:, K + j] = left + right                 # parent sum for next levels

    return Z

# 5) Back transform (same as before), then map back to original column order
def lt_back_with_indexer(z, child_l, child_r, indexer):
    z = np.asarray(z, float)
    n, M = z.shape
    K = len(indexer)
    N = K + M
    # inverse permutation to go back to original column order
    inv_indexer = np.empty_like(indexer)
    inv_indexer[indexer] = np.arange(K)
    x_rec = np.empty((n, K), float)
    for i in range(n):
        s = np.zeros(N, float)
        s[N-1] = 1.0
        for j in reversed(range(M)):
            p = 1.0 / (1.0 + np.exp(-z[i, j]))
            L, R = child_l[j], child_r[j]
            s[L] += s[K+j] * p
            s[R] += s[K+j] * (1.0 - p)
        tip_vec = s[:K]
        x_rec[i] = tip_vec[inv_indexer]
    # renormalize to be safe
    x_rec /= x_rec.sum(axis=1, keepdims=True)
    return x_rec

def lt_back_anyorder(z, child_l, child_r, indexer):
    """
    Reconstruct X from z (log-odds) for ANY internal-node ordering.
    child_l, child_r are arrays of child indices (0..K+M-1); leaves are 0..K-1, internals are K..K+M-1
    indexer maps original column order -> leaf order used by the tree.
    """
    z = np.asarray(z, float)
    n, M = z.shape
    K = len(indexer)
    N = K + M

    # Map absolute internal index -> j (0..M-1)
    abs_to_j = {K + j: j for j in range(M)}

    # Find the root internal node (the only internal that never appears as a child)
    child_abs = set()
    for j in range(M):
        L, R = child_l[j], child_r[j]
        if L >= K: child_abs.add(L)
        if R >= K: child_abs.add(R)
    all_internals = set(range(K, N))
    roots = list(all_internals - child_abs)
    if len(roots) != 1:
        raise RuntimeError(f"Expected exactly one root, found {len(roots)}: {roots}")
    root = roots[0]

    # Precompute sigmoid once
    def sigmoid(v): return 1.0 / (1.0 + np.exp(-v))

    # Inverse permutation to map back to original column order
    inv_indexer = np.empty_like(indexer)
    inv_indexer[indexer] = np.arange(K)

    X_rec = np.empty((n, K), float)
    for i in range(n):
        s = np.zeros(N, float)
        s[root] = 1.0  # put all mass at the root

        # Propagate mass top-down from the root
        stack = [root]
        while stack:
            node = stack.pop()
            if node < K:
                continue  # leaf
            j = abs_to_j[node]
            p = sigmoid(z[i, j])
            L, R = child_l[j], child_r[j]
            m = s[node]
            s[L] += m * p
            s[R] += m * (1.0 - p)
            if L >= K: stack.append(L)
            if R >= K: stack.append(R)

        tips = s[:K]
        X_rec[i] = tips[inv_indexer]

    # tidy up numerical drift
    X_rec = np.maximum(X_rec, 0.0)
    X_rec /= X_rec.sum(axis=1, keepdims=True)
    return X_rec


# ------------ CLR transform ------------- #

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
def _is_tensor(x):
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def clr(X, eps=1e-12, renorm=True):
    """
    Centered log-ratio transform (row-wise).

    Parameters
    ----------
    X : (n, p) array-like (NumPy or Torch)
        Compositions on the simplex: rows nonnegative and (about) sum to 1.
    eps : float
        Small floor to avoid log(0). Values are clipped below by eps.
    renorm : bool
        If True, renormalize each row to sum to 1 before transforming.

    Returns
    -------
    Z : (n, p) same type as X
        CLR-transformed rows (each row sums to 0).
    """
    if _is_tensor(X):
        if renorm:
            X = X / (X.sum(dim=1, keepdim=True) + 1e-30)
        X = X.clamp_min(eps)
        logX = X.log()
        gm_log = logX.mean(dim=1, keepdim=True)
        return logX - gm_log
    else:
        X = np.asarray(X, dtype=float)
        if renorm:
            X = X / (X.sum(axis=1, keepdims=True) + 1e-30)
        X = np.clip(X, eps, None)
        logX = np.log(X)
        gm_log = logX.mean(axis=1, keepdims=True)
        return logX - gm_log

def lt_back_anyorder_fixed(z, child_l, child_r, indexer, abs_to_j):
    z = np.asarray(z, float)
    n, M = z.shape
    K = len(indexer)
    N = K + M

    # find root (internal that never appears as a child)
    child_abs = set()
    for j in range(M):
        if child_l[j] >= K: child_abs.add(child_l[j])
        if child_r[j] >= K: child_abs.add(child_r[j])
    roots = list(set(range(K, N)) - child_abs)
    if len(roots) != 1:
        raise RuntimeError(f"Expected 1 root, found {len(roots)}: {roots}")
    root = roots[0]

    sigm = lambda v: 1.0 / (1.0 + np.exp(-v))
    inv_indexer = np.empty_like(indexer); inv_indexer[indexer] = np.arange(K)

    X_rec = np.empty((n, K), float)
    for i in range(n):
        mass = np.zeros(N, float)
        mass[root] = 1.0
        stack = [root]
        while stack:
            node = stack.pop()
            if node < K:                       # leaf
                continue
            j = abs_to_j[node]                 # <-- correct z column for this internal
            p = sigm(z[i, j])
            L, R = child_l[j], child_r[j]
            m = mass[node]
            mass[L] += m * p
            mass[R] += m * (1.0 - p)
            if L >= K: stack.append(L)
            if R >= K: stack.append(R)

        tips = mass[:K]
        X_rec[i] = tips[inv_indexer]

    X_rec = np.maximum(X_rec, 0.0)
    X_rec /= X_rec.sum(axis=1, keepdims=True)
    return X_rec

def inv_clr(Z):
    """
    Inverse CLR: map back to the simplex (row-wise).

    Parameters
    ----------
    Z : (n, p) array-like (NumPy or Torch)
        CLR vectors (rows typically sum to 0, but that is not required).

    Returns
    -------
    X : (n, p) same type as Z
        Rows are strictly positive and sum to 1.
    """
    if _is_tensor(Z):
        # Softmax along features is exactly the inverse, and numerically stable
        return torch.softmax(Z, dim=1)
    else:
        Z = np.asarray(Z, dtype=float)
        # Stable softmax
        Zs = Z - Z.max(axis=1, keepdims=True)
        expZ = np.exp(Zs)
        return expZ / expZ.sum(axis=1, keepdims=True)
# ---------- utilities ----------
def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, pd.DataFrame):
        return x.values
    return np.asarray(x)

def _coords_pcoa(X, metric='braycurtis'):
    n = X.shape[0]
    D = squareform(pdist(X, metric=metric))
    D2 = D**2
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = np.clip(eigvals, 0, None)
    coords = eigvecs * np.sqrt(eigvals + 0.0)
    total = eigvals.sum()
    pct = (eigvals / total * 100.0) if total > 0 else np.zeros_like(eigvals)
    return coords[:, :2], pct[:2]

def _coords_pca(X):
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    pct = pca.explained_variance_ratio_ * 100.0
    return coords, pct

def _legend_for_groups(ax, sc1, sc2, labels, w1, w2, marker1, marker2):
    if (w1 is None) and (w2 is None):
        ax.legend(handles=[sc1, sc2], labels=labels, title="Group", loc='upper right')
    else:
        elems = [
            Line2D([0], [0], marker=marker1, linestyle='None', color='k',
                   markerfacecolor='none', markersize=8, label=labels[0]),
            Line2D([0], [0], marker=marker2, linestyle='None', color='k',
                   markerfacecolor='none', markersize=8, label=labels[1]),
        ]
        ax.legend(handles=elems, title="Group", loc='upper right')

def _scatter_two_groups(ax, XY, n1, w1, w2, marker1, marker2, labels, norm, cmap):
    sc1 = ax.scatter(
        XY[:n1, 0], XY[:n1, 1],
        c=None if w1 is None else _to_numpy(w1),
        cmap=cmap if w1 is not None else None,
        norm=norm if w1 is not None else None,
        marker=marker1, edgecolors='none', s=30, alpha=0.9, label=labels[0]
    )
    sc2 = ax.scatter(
        XY[n1:, 0], XY[n1:, 1],
        c=None if w2 is None else _to_numpy(w2),
        cmap=cmap if w2 is not None else None,
        norm=norm if w2 is not None else None,
        marker=marker2, edgecolors='none', s=30, alpha=0.9, label=labels[1]
    )
    return sc1, sc2

# ---------- main plotter ----------
def embed_two_group(
    data1, data2, w1=None, w2=None,
    *, method='pcoa', metric='braycurtis',
    marker1='o', marker2='^', cmap_name='viridis',
    labels=('Group 1', 'Group 2'),
    title=None, vmin=None, vmax=None,
    colorbar_label='w',
    ax=None, show=True
):
    X1, X2 = _to_numpy(data1), _to_numpy(data2)
    if X1.ndim != 2 or X2.ndim != 2 or X1.shape[1] != X2.shape[1]:
        raise ValueError("data1 and data2 must be 2D with the same number of columns.")
    n1 = X1.shape[0]
    X = np.vstack([X1, X2])

    # 1) coordinates + % variance
    if method.lower() == 'pca':
        coords, pct = _coords_pca(X)
        axis_prefix = "PC"
        default_title = "PCA: two groups"
    elif method.lower() == 'pcoa':
        coords, pct = _coords_pcoa(X, metric=metric)
        axis_prefix = "PCo"
        default_title = f"PCoA ({metric}): two groups"
    else:
        raise ValueError("method must be 'pca' or 'pcoa'.")

    # 2) normalization for optional weights
    cmap = plt.get_cmap(cmap_name)
    norm = None
    w1a = None if w1 is None else _to_numpy(w1)
    w2a = None if w2 is None else _to_numpy(w2)
    if (w1a is not None) or (w2a is not None):
        vals = np.concatenate([v for v in [w1a, w2a] if v is not None])
        vmin = np.min(vals) if vmin is None else vmin
        vmax = np.max(vals) if vmax is None else vmax
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if (vmin < 0 < vmax) else Normalize(vmin=vmin, vmax=vmax)

    # 3) fig/ax
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        created_fig = True
    else:
        fig = ax.figure

    # 4) scatter + legend + colorbar
    sc1, sc2 = _scatter_two_groups(ax, coords, n1, w1a, w2a, marker1, marker2, labels, norm, cmap)
    ax.set_xlabel(f"{axis_prefix}1 ({pct[0]:.2f}%)")
    ax.set_ylabel(f"{axis_prefix}2 ({pct[1]:.2f}%)")
    ax.set_title(title or default_title)
    _legend_for_groups(ax, sc1, sc2, labels, w1a, w2a, marker1, marker2)

    if (w1a is not None) or (w2a is not None):
        mappable = sc1 if w1a is not None else sc2
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)

    if show and created_fig:
        plt.tight_layout()
        plt.show()
    return fig, ax

# ---------- wrappers ----------
def pcoa_two_group(
    data1, data2, w1=None, w2=None,
    metric='braycurtis',
    marker1='o', marker2='^',
    cmap_name='viridis',
    labels=('Group 1', 'Group 2'),
    title="PCoA (bray-curtis): two groups",
    vmin=None, vmax=None,
    colorbar_label='w',
    ax=None, show=True, **kwargs
):
    return embed_two_group(
        data1, data2, w1=w1, w2=w2,
        method='pcoa', metric=metric,
        marker1=marker1, marker2=marker2,
        cmap_name=cmap_name, labels=labels,
        title=title, vmin=vmin, vmax=vmax,
        colorbar_label=colorbar_label,
        ax=ax, show=show, **kwargs
    )

def pca_two_group(
    data1, data2, w1=None, w2=None,
    marker1='o', marker2='^',
    cmap_name='viridis',
    labels=('Group 1', 'Group 2'),
    title="PCA: two groups",
    vmin=None, vmax=None,
    colorbar_label='w',
    ax=None, show=True, **kwargs
):
    return embed_two_group(
        data1, data2, w1=w1, w2=w2,
        method='pca',
        marker1=marker1, marker2=marker2,
        cmap_name=cmap_name, labels=labels,
        title=title, vmin=vmin, vmax=vmax,
        colorbar_label=colorbar_label,
        ax=ax, show=show, **kwargs
    )

# ------------ mixed sampler ----------------- #
def make_q_mixed_sampler(p_source, q_source, *, device=None, p_weight=0.5):
    """
    Build a sampler that draws from the 50-50 (or p_weight) mixture of
    the real-data source p_source and generator q_source.

    p_source, q_source: either callables `fn(n)->Tensor/ndarray`
                        or objects with `.sample(n)->Tensor/ndarray`.

    device: torch.device or str or None. If provided, move outputs to this device.
    p_weight: proportion from the real source (default 0.5).
    """
    def _as_callable(src):
        # Prefer explicit .sample(...) first
        if hasattr(src, "sample") and callable(getattr(src, "sample")):
            return src.sample
        # Fall back to plain callables (functions, closures)
        if callable(src):
            return src
        raise TypeError("Source must be callable or have a .sample(n) method.")


    def _to_tensor(x, *, device=None):
        if isinstance(x, torch.Tensor):
            return x.to(device=device) if device is not None else x
        x = torch.as_tensor(x)  # handles numpy/array-like
        return x.to(device=device) if device is not None else x

    p_fn = _as_callable(p_source)
    q_fn = _as_callable(q_source)

    def sample(n, *, return_labels=False, shuffle=True):
        # split counts (handle odd n without bias)
        n_p = int(round(n * p_weight))
        n_q = n - n_p

        x_p = _to_tensor(p_fn(n_p), device=device)
        x_q = _to_tensor(q_fn(n_q), device=device)

        # sanity check on feature dims
        if x_p.ndim != x_q.ndim or x_p.shape[1:] != x_q.shape[1:]:
            raise ValueError(f"Shape mismatch between real {tuple(x_p.shape)} and gen {tuple(x_q.shape)} samples.")

        X = torch.cat([x_p, x_q], dim=0)

        y = None
        if return_labels:
            # 0 = real, 1 = generated
            y = torch.cat([
                torch.zeros(n_p, dtype=torch.long, device=X.device),
                torch.ones(n_q,  dtype=torch.long, device=X.device)
            ], dim=0)

        if shuffle:
            perm = torch.randperm(n, device=X.device)
            X = X[perm]
            if y is not None:
                y = y[perm]

        return (X, y) if return_labels else X

    return sample

def plot_cca(
    X_train, X_self, logw_train, logw_self,
    *,
    key=None,
    method_names=None,
    cmap="viridis",
    norm=None,
    ax=None,
    n_components=1,
    component=0
):
    """
    Fit CCA between X and log(w), then scatter U vs V for the chosen component.
    """

    # --- helpers: accept torch/numpy and ensure 2D shapes ---
    def to_numpy_2d(a):
        try:
            import torch
            if isinstance(a, torch.Tensor):
                a = a.detach().cpu().numpy()
        except Exception:
            pass
        a = np.asarray(a)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return a

    X_train = to_numpy_2d(X_train)   # (n_tr, p)
    X_self  = to_numpy_2d(X_self)    # (n_sf, p)

    # weights can be 1D; make them (n, 1)
    logw_train = to_numpy_2d(logw_train)  # (n_tr, 1) desired
    logw_self  = to_numpy_2d(logw_self)   # (n_sf, 1) desired

    # --- stack X and Y properly ---
    stacked_X = np.vstack([X_train, X_self])            # (n_tr+n_sf, p)
    stacked_y = np.vstack([logw_train, logw_self])      # (n_tr+n_sf, 1)

    # --- fit CCA ---
    cca = CCA(n_components=n_components)
    cca.fit(stacked_X, stacked_y)
    U_all, V_all = cca.transform(stacked_X, stacked_y)  # (n_total, k)

    # --- split back ---
    n_tr = X_train.shape[0]
    U_tr, V_tr = U_all[:n_tr, component], V_all[:n_tr, component]
    U_sf, V_sf = U_all[n_tr:, component], V_all[n_tr:, component]

    # --- axis and coloring ---
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # If no norm provided, use global min/max of both groups' weights
    if norm is None:
        w_all = np.concatenate([logw_train.ravel(), logw_self.ravel()])
        wmin, wmax = float(np.min(w_all)), float(np.max(w_all))
        from matplotlib import colors
        norm = colors.Normalize(vmin=wmin, vmax=wmax)

    ax.scatter(
        U_tr, V_tr,
        c=logw_train.ravel(), cmap=cmap, norm=norm,
        marker='1', s=30, alpha=0.8, label='train'
    )
    ax.scatter(
        U_sf, V_sf,
        c=logw_self.ravel(), cmap=cmap, norm=norm,
        marker='+', s=30, alpha=0.8, label=key or 'self'
    )

    # canonical correlation of the selected component
    ev = np.corrcoef(U_all[:, component], V_all[:, component])[0, 1]

    ax.set_xlabel(f"CCA U = aᵀX ({100*ev:.1f}% corr(U,V))")
    ax.set_ylabel("CCA V = bᵀlog(w)")
    if method_names and key in method_names:
        ax.set_title(method_names[key])
    elif key:
        ax.set_title(key)
    ax.legend(fontsize="small", title="Group")

    return cca, ax