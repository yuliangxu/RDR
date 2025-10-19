import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.lines import Line2D
import statsmodels.api as sm
import torch

# https://www.dropbox.com/developers/apps 
DropBox_token = "sl.u.AFwZl3blNG5JiKXrxXUcj4fb2wtQYJMpLcRJ6a_flh7NLIT2D2kIuj8Vp1jZ621nKcLx2EarxZ7niXOJoPxGyN3qXCCxpkQ0bPvLILZsSsI786ofUMHiyZ5aKfh8OWKoj5ERc1EvnUTltAlhWmh6mLN_PMf34fZXlTUD8eo68W1KOlJxT2Pwy04hbgjwW9ipOWx8a6HTPQvY5PMby2pYgcb6Pr3oN7qOmkGIIEuOisTnmN2rnk6pgOlK8WHMXFAOpfpxF3hf5TbSlT2w8xwGbV4VTa-ieilM7auYiKUeytHhfe64hgQG2SI_1dT9lwGyrXSwhuApz0XGBAKY3yqczaraCjQCuJqjEEz-SiG95V8ZSh9KLLfEw-rW72zhm0jz-Ty57HLxpPvFGSidnAgM0vwjU2s_chmxQfUt90rvLlt0lKv5PKqurDPIadcZSnv4FUqQeMU8oYKUNG-ntK9GXwTVb_uGArTJKbdrlg-Cam8cck0e1eU9pbIO6NPtPmgb8kY-OmD-fKyRoxUW90a-32QNyVYsYLItklV25lkPxmyRePk7SVWkyl7k73elpwrTCIRKF3CXSPYHE5z_V1wUXL1TbOTXLt33mTtD5fT8kFXf7-GnJHZvLwcIds-Z_uPwRS9EFVkZMObxWmUBJXrodAJnLMFBAy5S8KIRPI_x1wcQWKxPInHfb-XfjXEQccxaA3XmXTh5WsIqtt00kf3IPpPuE-sQQfNNX-mb1OfmsyCyLGtJIPqa4xTE6Cu54KuSTuaRR9_SvmVvtN-VHVcYYgFTQuAxfUPjaj6ssUT1Ck9XzOMh0DnY6zphvQwrc08b-7UovyJogBNFnqStCmq8xPymab6Ahob9ii_lg31DbcQ_ZuPpWPzQ7u_P7w1CP3Kxt8I2Tj2sCeaXq8rrOHDhFFnCBGfcWY9KSxJQUHbZNu_Qr26lw-9CvaYhDtTs8b8IDuU2wY_1hOfr17x3HP3bnl3tV9LEssuiarLDbtJTsLXPuigBu5PVEhB4VuMWfPh05rI0dU8i9pruh7lK1nP4uLcyAgr3Cyq4DbfNyoPigvPcl_riS9PF9mleuR3Qo9sS0QQWn7xLuQSV0H--kmhIA6iWpqFy4G0SUeY3uEYCniQs1CHOPwjvgOIpHd3Evw3Zaeo4LgKhsfRjZZqlFjJG3Yx-BRyAtguxqkDMEcbDwoP5jvFaS4XB1bMIkPOBZyF_F2cBe2Q25I1ZLpwkLo4dV5EfaCgJnycHmO6bgoKOBtmX4IAhXc2hRfzvlE8fscIu3JSkqnSJsoLcg5-03Z7Api90"

def first_canonical_weight(X, Y):
    """
    Given X (n×k) and Y (n×1), return
      - a:    weight vector of length k so that X a is most correlated with Y
      - rho:  the sample correlation corr(X a, Y)
      - U:    the canonical variate X a (centered, unit‐variance)
      - V:    the canonical variate Y (centered, unit‐variance)
    """
    X = np.asarray(X)
    Y = np.asarray(Y).reshape(-1, 1)
    n, k = X.shape
    
    # 1. Center:
    X_mean = X.mean(axis=0, keepdims=True)   # shape (1, k)
    Y_mean = Y.mean(axis=0, keepdims=True)   # shape (1, 1)
    Xc = X - X_mean                          # (n, k)
    Yc = Y - Y_mean                          # (n, 1)

    # 2. Raw weight = (Xc'Xc)^{-1} Xc'Yc
    XtX = Xc.T @ Xc                         # (k, k)
    XtY = Xc.T @ Yc                         # (k, 1)
    a_raw = np.linalg.solve(XtX, XtY)       # (k, 1)

    # 3. Normalize so Var(Xc·a) = 1
    U_raw = Xc @ a_raw                      # (n, 1)
    var_U = (1/(n-1)) * np.sum(U_raw**2)    # scalar
    a = a_raw / np.sqrt(var_U)              # (k, 1)

    # 4. Compute the unit‐variance canonical variates
    U = Xc @ a                              # (n, 1), Var=1 by construction
    V = Yc / np.sqrt((1/(n-1)) * np.sum(Yc**2))  # (n, 1), Var=1

    # 5. Sample corr:
    rho = np.dot(U.T, Yc) / (n-1)           # scalar
    
    return a.flatten(), float(rho), U.flatten(), V.flatten()


def stratified_stacked_barplot(
    log_w_g,
    log_w_g0,
    sample_method,
    sample_train,
    taxa_name,
    method='Method',
    taxa_level = "V3",
    title=None,
    figsize=(14, 8),
    *,
    stratify: bool = True,         # False => raw two-panel plot
    include_left: bool = True,     # stratified toggles
    include_overlap: bool = True,
    include_right: bool = True,
    # ---- NEW performance knobs (used when stratify=False) ----
    top_k: int = 30,               # keep top-K taxa; rest -> "Other"
    max_samples: int = 250,        # max bars to draw per panel
    random_state: int = 0,
    show: bool = True, return_fig: bool = False, close: bool = False
):
    """
    Plot stacked-bar comparison between a method and training samples.

    When stratify=False:
      - Reduces to top_k taxa (others collapsed into 'Other').
      - Subsamples to max_samples samples per panel for speed.

    Returns
    -------
    matplotlib.figure.Figure
    """

    # ---- helpers ----
    def _to_np(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _count_or_eps(idx):
        cnt = idx.size
        return float(cnt) if cnt > 0 else 0.1

    def _extract_taxa_labels(taxa_like):
        if isinstance(taxa_like, pd.DataFrame):
            for col in (taxa_level, 'taxon', 'taxa', 'name'):
                if col in taxa_like.columns:
                    return taxa_like[col].astype(str).tolist()
            return taxa_like.iloc[:, 0].astype(str).tolist()
        if isinstance(taxa_like, (pd.Series, pd.Index)):
            return taxa_like.astype(str).tolist()
        return list(map(str, np.asarray(taxa_like)))

    def _reduce_for_plot_by_label(samples_np, other_np, taxa_like,
                              top_k=30, max_samples=200, random_state=0):
        """
        Collapse by unique V3 labels:
        - rank labels by pooled mean abundance,
        - keep top_k labels (sum all columns within each kept label),
        - optionally add an 'Other' column if leftover labels remain,
        - subsample rows for speed.
        """
        import numpy as np, pandas as pd

        # extract per-column labels (length K)
        def _extract_taxa_labels(taxa_like):
            if isinstance(taxa_like, pd.DataFrame):
                for col in ('V3','taxon','taxa','name'):
                    if col in taxa_like.columns: return taxa_like[col].astype(str).tolist()
                return taxa_like.iloc[:,0].astype(str).tolist()
            if isinstance(taxa_like, (pd.Series, pd.Index)):
                return taxa_like.astype(str).tolist()
            return list(map(str, np.asarray(taxa_like)))

        labels = np.asarray(_extract_taxa_labels(taxa_like))         # (K,)
        uniq_labels, inv = np.unique(labels, return_inverse=True)    # uniq labels (L,), map K->L

        # pooled mean per LABEL (sum cols in label, then mean over samples)
        pooled = np.concatenate([samples_np, other_np], axis=0)      # (n_tot, K)
        pooled_label_mean = np.zeros(len(uniq_labels))
        for j in range(len(uniq_labels)):
            cols = (inv == j)
            pooled_label_mean[j] = pooled[:, cols].sum(axis=1).mean()

        order = np.argsort(-pooled_label_mean)
        keep_labels = uniq_labels[order[:min(top_k, len(uniq_labels))]]
        keep_set = set(keep_labels)
        has_other = len(keep_set) < len(uniq_labels)

        def collapse_by_label(X):
            parts = []
            for lab in keep_labels:
                cols = (labels == lab)
                parts.append(X[:, cols].sum(axis=1, keepdims=True))
            if has_other:
                parts.append(X[:, ~np.isin(labels, list(keep_set))].sum(axis=1, keepdims=True))
            return np.hstack(parts)

        sm_red = collapse_by_label(samples_np)
        st_red = collapse_by_label(other_np)

        labels_red = list(keep_labels) + (["Other"] if has_other else [])
        taxa_name_red = pd.DataFrame({"V3": labels_red})

        # subsample rows for speed
        rng = np.random.default_rng(random_state)
        def _subsample(X):
            if X.shape[0] > max_samples:
                keep = rng.choice(X.shape[0], size=max_samples, replace=False)
                return X[keep]
            return X

        return _subsample(sm_red), _subsample(st_red), taxa_name_red

    # You must have this available in your environment:
    # def stacked_bar_one_new(ax, reconstructed_ra_all, taxa_name, plot_name, n_sub=10):
    #     ... (returns taxa->color dict the first time it's called) ...

    # ---- prep inputs ----
    arr_g = _to_np(log_w_g)
    arr_0 = _to_np(log_w_g0)
    sample_method_np = _to_np(sample_method)  # (n_method, K)
    sample_train_np  = _to_np(sample_train)   # (n_train,  K)

    legend_space = 0.15
    plot_space = 1.0 - legend_space
    gap = 0.05
    bot_row_y = 0.05
    row_h     = 0.40
    top_row_y = 0.57  

    fig = plt.figure(figsize=figsize)
    taxa2color_all = {}

    # -----------------------
    # RAW (non-stratified) mode
    # -----------------------
    if not stratify:
        # Reduce complexity for speed
        sm_red, st_red, taxa_name_red = _reduce_for_plot_by_label(
            sample_method_np, sample_train_np, taxa_name,
            top_k=top_k, max_samples=max_samples, random_state=random_state
        )

        width_block = plot_space - gap

        # Method (reduced)
        ax = fig.add_axes([0.0, top_row_y, width_block, row_h])
        result = stacked_bar_one_new(ax, sm_red.T, taxa_name_red, f'All: {method}')
        if not taxa2color_all:
            taxa2color_all = result

        # Train (reduced)
        ax = fig.add_axes([0.0, bot_row_y, width_block, row_h])
        stacked_bar_one_new(ax, st_red.T, taxa_name_red, 'All: Observed')

        # Legend from reduced taxa
        taxa_labels = list(pd.unique(taxa_name_red[taxa_level]))

    else:
        # -----------------------
        # STRATIFIED mode
        # -----------------------
        # Determine overlapping log-weight bounds
        left_cut = max(arr_g.min(), arr_0.min())
        right_cut = min(arr_g.max(), arr_0.max())

        # Indices for each region
        idx_g_left     = np.where(arr_g < left_cut)[0]
        idx_g_overlap  = np.where((arr_g >= left_cut) & (arr_g <= right_cut))[0]
        idx_g_right    = np.where(arr_g > right_cut)[0]
        idx_0_left     = np.where(arr_0 < left_cut)[0]
        idx_0_overlap  = np.where((arr_0 >= left_cut) & (arr_0 <= right_cut))[0]
        idx_0_right    = np.where(arr_0 > right_cut)[0]

        # Proportional widths
        w_g_left    = _count_or_eps(idx_g_left)    if include_left    else 0.0
        w_g_overlap = _count_or_eps(idx_g_overlap) if include_overlap else 0.0
        w_g_right   = _count_or_eps(idx_g_right)   if include_right   else 0.0

        w_0_left    = _count_or_eps(idx_0_left)    if include_left    else 0.0
        w_0_overlap = _count_or_eps(idx_0_overlap) if include_overlap else 0.0
        w_0_right   = _count_or_eps(idx_0_right)   if include_right   else 0.0

        total_w = (w_g_left + w_g_overlap + w_g_right +
                   w_0_left + w_0_overlap + w_0_right)

        # Fallback to raw if nothing to plot
        if total_w == 0.0:
            width_block = plot_space - gap
            ax = fig.add_axes([0.0, top_row_y, width_block, row_h])
            result = stacked_bar_one_new(ax, sample_method_np.T, taxa_name, f'All: {method}')
            if not taxa2color_all:
                taxa2color_all = result
            ax = fig.add_axes([0.0, bot_row_y, width_block, row_h])
            stacked_bar_one_new(ax, sample_train_np.T, taxa_name, 'All: Observed')
            taxa_labels = sorted(pd.unique(_extract_taxa_labels(taxa_name)))
        else:
            available_width = plot_space - 2 * gap

            # Row widths
            width_g_left    = (w_g_left    / total_w) * available_width
            width_g_overlap = (w_g_overlap / total_w) * available_width
            width_g_right   = (w_g_right   / total_w) * available_width

            width_0_left    = (w_0_left    / total_w) * available_width
            width_0_overlap = (w_0_overlap / total_w) * available_width
            width_0_right   = (w_0_right   / total_w) * available_width

            # Columns (align rows by max width per column)
            blocks = []
            if include_left   and (width_g_left > 0 or width_0_left > 0):
                blocks.append(("Left",    max(width_g_left,    width_0_left)))
            if include_overlap and (width_g_overlap > 0 or width_0_overlap > 0):
                blocks.append(("Overlap", max(width_g_overlap, width_0_overlap)))
            if include_right  and (width_g_right > 0 or width_0_right > 0):
                blocks.append(("Right",   max(width_g_right,   width_0_right)))

            x_cursor = 0.0
            for name, block_w in blocks:
                # Method row
                if name == "Left" and width_g_left > 0:
                    ax = fig.add_axes([x_cursor, top_row_y, width_g_left, row_h])
                    result = stacked_bar_one_new(
                        ax, sample_method_np[idx_g_left, :].T, taxa_name, f'Left: {method}'
                    )
                    if not taxa2color_all:
                        taxa2color_all = result
                if name == "Overlap" and width_g_overlap > 0:
                    ax = fig.add_axes([x_cursor, top_row_y, width_g_overlap, row_h])
                    result = stacked_bar_one_new(
                        ax, sample_method_np[idx_g_overlap, :].T, taxa_name, f'Overlap: {method}'
                    )
                    if not taxa2color_all:
                        taxa2color_all = result
                if name == "Right" and width_g_right > 0:
                    ax = fig.add_axes([x_cursor, top_row_y, width_g_right, row_h])
                    result = stacked_bar_one_new(
                        ax, sample_method_np[idx_g_right, :].T, taxa_name, f'Right: {method}'
                    )
                    if not taxa2color_all:
                        taxa2color_all = result

                # Train row
                if name == "Left" and width_0_left > 0:
                    ax = fig.add_axes([x_cursor, bot_row_y, width_0_left, row_h])
                    stacked_bar_one_new(ax, sample_train_np[idx_0_left, :].T, taxa_name, 'Left: Observed')
                if name == "Overlap" and width_0_overlap > 0:
                    ax = fig.add_axes([x_cursor, bot_row_y, width_0_overlap, row_h])
                    stacked_bar_one_new(ax, sample_train_np[idx_0_overlap, :].T, taxa_name, 'Overlap: Observed')
                if name == "Right" and width_0_right > 0:
                    ax = fig.add_axes([x_cursor, bot_row_y, width_0_right, row_h])
                    stacked_bar_one_new(ax, sample_train_np[idx_0_right, :].T, taxa_name, 'Right: Observed')

                x_cursor += block_w + gap

            taxa_labels = sorted(pd.unique(_extract_taxa_labels(taxa_name)))

    # ---- legend ----
    if 'taxa_labels' not in locals():
        taxa_labels = sorted(pd.unique(_extract_taxa_labels(taxa_name)))
    if taxa2color_all:
        patches = [mpatches.Patch(color=taxa2color_all.get(t, '#999999'), label=t)
                   for t in taxa_labels]
        fig.legend(
            handles=patches,
            title='Taxa',
            bbox_to_anchor=(1 - legend_space/2, 0.5),
            loc='center',
            borderaxespad=0.5,
            fontsize='small',
            title_fontsize='small'
        )

    if title:
        fig.suptitle(title, fontsize='large', fontweight='bold')

    if show:
        plt.show()
    if close:
        plt.close(fig)  # prevents auto-display of returned fig in notebooks

    return fig if return_fig else None


def compute_pcoa_coords(X,metric='braycurtis'):
        n = X.shape[0]
        D = squareform(pdist(X, metric=metric))
        D2 = D**2
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J.dot(D2).dot(J)

        eigvals, eigvecs = np.linalg.eigh(B)
        idx_desc = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_desc]
        eigvecs = eigvecs[:, idx_desc]

        # Clip negatives and compute coords
        eigvals_clipped = np.where(eigvals > 0, eigvals, 0.0)
        coords = eigvecs * np.sqrt(eigvals_clipped)
        pc1 = coords[:, 0]
        pc2 = coords[:, 1]
        total_var = eigvals_clipped.sum()
        pct1 = 100 * eigvals_clipped[0] / total_var if total_var > 0 else 0.0
        pct2 = 100 * eigvals_clipped[1] / total_var if total_var > 0 else 0.0
        return pc1, pc2, pct1, pct2

def ols_with_pvalues(data1, w1, data2, w2, p, alpha=0.05, by_absolute=True):
    """
    Stack two datasets (data1, data2) and two weight vectors (w1, w2),
    regress log(w_combined) on the combined feature matrix, and return 
    a DataFrame of the top‐p significant coefficients (with their p‐values).

    Parameters
    ----------
    data1 : array‐like or pandas.DataFrame, shape (n1, k)
        First feature matrix. If it's a torch.Tensor, it will be moved to 
        CPU and converted to NumPy.
    w1 : array‐like or torch.Tensor, length n1
        Positive weights for data1.
    data2 : array‐like or pandas.DataFrame, shape (n2, k)
        Second feature matrix (same number of columns k as data1).
    w2 : array‐like or torch.Tensor, length n2
        Positive weights for data2.
    p : int
        Number of “top” significant coefficients to return.
    alpha : float, default=0.05
        Significance threshold: keep only coefficients with p‐value < alpha.
    by_absolute : bool, default=True
        If True, rank by |coef| descending. If False, rank by coef descending.

    Returns
    -------
    top_df : pandas.DataFrame
        A DataFrame with up to p rows, indexed by parameter name, and two columns:
          - 'coef' : coefficient estimate
          - 'pval' : two‐tailed p‐value
        Sorted in descending order according to the chosen criterion. If no 
        coefficient is significant, returns an empty DataFrame.
    """
    # 1) Convert torch.Tensor → NumPy if needed
    try:
        import torch
        if isinstance(data1, torch.Tensor):
            data1 = data1.cpu().detach().numpy()
        if isinstance(data2, torch.Tensor):
            data2 = data2.cpu().detach().numpy()
        if isinstance(w1, torch.Tensor):
            w1 = w1.cpu().detach().numpy()
        if isinstance(w2, torch.Tensor):
            w2 = w2.cpu().detach().numpy()
    except ImportError:
        pass

    # 2) Extract arrays from DataFrames or array-likes
    if isinstance(data1, pd.DataFrame):
        X1 = data1.values
    else:
        X1 = np.asarray(data1)
    if isinstance(data2, pd.DataFrame):
        X2 = data2.values
    else:
        X2 = np.asarray(data2)

    w1 = np.asarray(w1)
    w2 = np.asarray(w2)

    n1, k1 = X1.shape
    n2, k2 = X2.shape
    if k1 != k2:
        raise ValueError("Both datasets must have the same number of columns (features).")
    if w1.shape[0] != n1 or w2.shape[0] != n2:
        raise ValueError("Length of w1/w2 must match number of rows in data1/data2.")

    # 3) Build combined X and w
    if n1 == 0:
        X_comb = X2.copy()
        w_comb = w2.copy()
    elif n2 == 0:
        X_comb = X1.copy()
        w_comb = w1.copy()
    else:
        X_comb = np.vstack([X1, X2])       # shape (n1+n2, k)
        w_comb = np.concatenate([w1, w2])

    # 4) Check positivity, compute log(w)
    if np.any(w_comb <= 0):
        raise ValueError("All entries of w must be > 0 to take log.")
    y = np.log(w_comb)

    # 5) Add intercept
    X_design = sm.add_constant(X_comb)  # adds a column of ones automatically

    # 6) Fit OLS
    model = sm.OLS(y, X_design)
    results = model.fit()

    # 7) Extract params & pvals
    coefs = results.params        # NumPy array of length (k+1,)
    pvals = results.pvalues       # NumPy array of length (k+1,)
    # Construct parameter names manually:
    #   first is "const", then "X1", "X2", ..., "Xk"
    k = X_comb.shape[1]
    names = ['const'] + [f"X{i+1}" for i in range(k)]

    # 8) Filter to significant (pval < alpha)
    coefs = np.asarray(coefs)
    pvals = np.asarray(pvals)
    sig_mask = pvals < alpha
    if not sig_mask.any():
        # return empty DataFrame with correct columns
        return pd.DataFrame(columns=['coef', 'pval'])

    sig_names = [names[i] for i in np.where(sig_mask)[0]]
    sig_coefs = coefs[sig_mask]
    sig_pvals = pvals[sig_mask]

    # 9) Build a small DataFrame
    df = pd.DataFrame({
        'coef': sig_coefs,
        'pval': sig_pvals
    }, index=sig_names)

    # 10) Sort & take top p
    if by_absolute:
        df = df.reindex(df['coef'].abs().sort_values(ascending=False).index)
    else:
        df = df.sort_values('coef', ascending=False)

    top_df = df.iloc[:p].copy()
    return top_df


def stacked_bar_one_old(reconstructed_ra_all, plot_name,taxa_name, w, w_order="decreasing", n_sub=None, width=12, height=6):
    """
    Plots a stacked bar chart for the reconstructed data.
    
    Parameters:
      reconstructed_ra_all : 2D array
          The raw reconstructed data.
      plot_name : str
          Title for the plot.
      w : array-like
          A vector of values (one per sample) used for ordering the samples.
      w_order : str, optional
          Order for sorting the samples according to w. Accepts "increasing" or "decreasing".
          Default is "decreasing".
      n_sub : int, optional
          The number of samples to display. Defaults to the total number of samples.
      width : int, optional
          Width of the plot.
      height : int, optional
          Height of the plot.
    """
    # Set number of samples to plot if not provided
    if n_sub is None:
        n_sub = reconstructed_ra_all.shape[0]
        
    if hasattr(reconstructed_ra_all, 'cpu'):
        reconstructed_ra_all = reconstructed_ra_all.cpu().detach().numpy()

    # Create a DataFrame with transposed data and set column names from taxa_name['V3']
    gz0_comp_df = pd.DataFrame(reconstructed_ra_all)
    gz0_comp_df.columns = taxa_name['V3']
    # Group by taxon (or component) and sum columns that belong to the same taxon
    gz0_comp_grouped = gz0_comp_df.groupby(axis=1, level=0).sum()

    # Check that the length of w matches the number of samples available for plotting
    if len(w) != gz0_comp_grouped.shape[0]:
        raise ValueError("Length of w must equal the number of samples in the grouped data.")


    # Extract component names and number of components
    c_names = gz0_comp_grouped.columns
    n_comp_c = gz0_comp_grouped.shape[1]

    # Subselect w for the samples that are about to be plotted
    # Convert w to a NumPy array if it's a PyTorch tensor or similar.
    if hasattr(w, 'cpu'):
        w_np = w.cpu().numpy()
    else:
        w_np = np.array(w)
    # Subselect w for the samples that are about to be plotted
    


    # **Step 1: Sort samples based on w**
    if w_order == "decreasing":
        sorted_sample_indices = np.argsort(-w_np)
    else:
        sorted_sample_indices = np.argsort(w_np)

    sorted_sample_indices = sorted_sample_indices[:n_sub]
    gz0_comp_np_sub = gz0_comp_grouped.values
    gz0_comp_np_sub = gz0_comp_np_sub[sorted_sample_indices]  # Apply sample sorting

    

    # **Step 2: Sort components by their mean abundance (descending)**
    component_means = np.mean(gz0_comp_np_sub, axis=0)  # Compute mean per component
    sorted_component_indices = np.argsort(-component_means)  # Sort components in descending order
    gz0_comp_np_sub = gz0_comp_np_sub[:, sorted_component_indices]  # Apply component sorting
    c_names = c_names[sorted_component_indices]  # Update category labels

    # Create figure and initialize the bottom array for stacking
    fig, ax = plt.subplots(figsize=(width, height))
    bottom = np.zeros(n_sub)

    # **Step 3: Select colormap dynamically**
    base_cmap = plt.colormaps.get_cmap('tab20') if n_comp_c <= 20 else plt.colormaps.get_cmap('tab20b')  
    colors = [base_cmap(i / max(1, n_comp_c - 1)) for i in range(n_comp_c)]  # Generate distinct colors
    listed_cmap = mcolors.ListedColormap(colors)  # Create a discrete colormap

    # **Step 4: Plot the stacked bar chart**
    for i in range(n_comp_c):
        ax.bar(range(n_sub), gz0_comp_np_sub[:, i], bottom=bottom, width=1.0, color=colors[i])
        bottom += gz0_comp_np_sub[:, i]

    # **Step 5: Create a discrete colorbar**
    bounds = np.arange(n_comp_c + 1) - 0.5  # For proper alignment
    norm = mcolors.BoundaryNorm(bounds, listed_cmap.N)
    sm = cm.ScalarMappable(cmap=listed_cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ticks=np.arange(n_comp_c), ax=ax)
    cbar.ax.set_yticklabels(c_names)
    cbar.set_label("Bacterial Class (Sorted by Mean Abundance)")

    # Labels and title
    plt.xlabel("Sample Index (Sorted based on provided w)")
    plt.ylabel("Proportion")
    plt.title(plot_name)

def stacked_bar_one2_old(ax, reconstructed_ra_all, taxa_name, plot_name, n_sub = None):
    
    if n_sub is None:
        n_sub = reconstructed_ra_all.shape[1]
    
    gz0_comp_df = pd.DataFrame(reconstructed_ra_all.T)
    gz0_comp_df.columns = taxa_name['V3']
    gz0_comp_grouped = gz0_comp_df.groupby(axis=1, level=0).sum()
    
    c_names = gz0_comp_grouped.columns
    n_comp_c = gz0_comp_grouped.shape[1]
    gz0_comp_np_sub = gz0_comp_grouped.values[:n_sub, :]

    # **Step 1: Sort samples by the largest component (descending)**
    sorted_sample_indices = np.argsort(-gz0_comp_np_sub.max(axis=1))  # Sort by highest component
    gz0_comp_np_sub = gz0_comp_np_sub[sorted_sample_indices]  # Apply sample sorting

    # **Step 2: Sort components by their mean abundance (descending)**
    component_means = np.mean(gz0_comp_np_sub, axis=0)  # Compute mean per component
    sorted_component_indices = np.argsort(-component_means)  # Sort components in descending order
    gz0_comp_np_sub = gz0_comp_np_sub[:, sorted_component_indices]  # Apply component sorting
    c_names = c_names[sorted_component_indices]  # Update category labels
    
    bottom = np.zeros(n_sub)
    
    # **Step 3: Select colormap dynamically**
    base_cmap = plt.colormaps.get_cmap('tab20') if n_comp_c <= 20 else plt.colormaps.get_cmap('tab20b')  
    colors = [base_cmap(i / max(1, n_comp_c - 1)) for i in range(n_comp_c)]  # Generate distinct colors
    listed_cmap = mcolors.ListedColormap(colors)  # Create a discrete colormap

    # **Step 4: Plot the stacked bar chart**
    for i in range(n_comp_c):  # For each component
        ax.bar(range(n_sub), gz0_comp_np_sub[:, i], bottom=bottom, width=1.0, color=colors[i])
        bottom += gz0_comp_np_sub[:, i]  # Update bottom to stack bars

    # Labels and title
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Proportion")
    ax.set_title(plot_name)


def stacked_bar_one(reconstructed_ra_all, taxa_name, plot_name, n_sub=None, width=12, height=6):
    # ——— Build global taxa → color map ———
    all_taxa = sorted(taxa_name['V3'].unique())
    n_taxa = len(all_taxa)
    base_cmap = plt.get_cmap('tab20') if n_taxa <= 20 else plt.get_cmap('tab20b')
    taxa2color = {
        t: base_cmap(i / (n_taxa - 1))
        for i, t in enumerate(all_taxa)
    }
    # ——— Prepare data ———
    if n_sub is None:
        n_sub = reconstructed_ra_all.shape[1]
    df = pd.DataFrame(reconstructed_ra_all.T, columns=taxa_name['V3'])
    grouped = df.groupby(axis=1, level=0).sum()
    c_names = grouped.columns
    X = grouped.values[:n_sub, :]
    # sort samples by largest component
    idx_samp = np.argsort(-X.max(axis=1))
    X = X[idx_samp]
    # sort taxa by mean abundance
    means = X.mean(axis=0)
    idx_taxa = np.argsort(-means)
    X = X[:, idx_taxa]
    c_names = c_names[idx_taxa]
    # pull colors for the bars in abundance‐order
    bar_colors = [taxa2color[t] for t in c_names]
    # ——— Plot ———
    fig, ax = plt.subplots(figsize=(width, height))
    bottom = np.zeros(n_sub)
    for i, t in enumerate(c_names):
        ax.bar(np.arange(n_sub), X[:, i], bottom=bottom, width=1.0, color=bar_colors[i])
        bottom += X[:, i]
    # ——— Static alphabetical colorbar ———
    full_colors = [taxa2color[t] for t in all_taxa]
    cmap_full = mcolors.ListedColormap(full_colors)
    bounds = np.arange(n_taxa+1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap_full.N)
    sm = cm.ScalarMappable(cmap=cmap_full, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=np.arange(n_taxa), ax=ax)
    cbar.ax.set_yticklabels(all_taxa)
    cbar.set_label("Taxa (alphabetical)")
    ax.set_xlabel("Sample Index (sorted)")
    ax.set_ylabel("Proportion")
    ax.set_title(plot_name)
    plt.tight_layout()
#     plt.show()


def stacked_bar_one2(ax, reconstructed_ra_all, taxa_name, plot_name, n_sub=None):

    # If tensor on GPU, move to CPU and convert to numpy
    if hasattr(reconstructed_ra_all, 'cpu'):
        reconstructed_ra_all = reconstructed_ra_all.cpu().detach().numpy()

    # ——— Build global taxa → color map ———
    all_taxa = sorted(taxa_name['V3'].unique())
    n_taxa = len(all_taxa)
    base_cmap = plt.get_cmap('tab20') if n_taxa <= 20 else plt.get_cmap('tab20b')
    taxa2color = {
        t: base_cmap(i / (n_taxa - 1))
        for i, t in enumerate(all_taxa)
    }

    # ——— Prepare data ———
    if n_sub is None:
        n_sub = reconstructed_ra_all.shape[1]
    df = pd.DataFrame(reconstructed_ra_all.T, columns=taxa_name['V3'])
    grouped = df.groupby(axis=1, level=0).sum()
    c_names = grouped.columns
    X = grouped.values[:n_sub, :]

    # sort samples by their maximum value
    idx_samp = np.argsort(-X.max(axis=1))
    X = X[idx_samp]

    # sort taxa by mean abundance across the selected samples
    means = X.mean(axis=0)
    idx_taxa = np.argsort(-means)
    X = X[:, idx_taxa]
    c_names = c_names[idx_taxa]

    # pull colors for bars in the new taxa order
    bar_colors = [taxa2color[t] for t in c_names]

    # ——— Plot ———
    bottom = np.zeros(n_sub)
    for i, t in enumerate(c_names):
        ax.bar(np.arange(n_sub), X[:, i], bottom=bottom, width=1.0, color=bar_colors[i])
        bottom += X[:, i]

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Proportion")
    ax.set_title(plot_name)

    # ——— Add taxa legend on the side ———
    patches = [mpatches.Patch(color=taxa2color[t], label=t) for t in c_names]
    ax.legend(
        handles=patches,
        title="Taxa",
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0.5,
        fontsize='small',
        title_fontsize='small'
    )

    # Adjust layout so the legend fits
    plt.tight_layout(rect=[0, 0, 0.85, 1])

def stacked_bar_one_new(ax, reconstructed_ra_all, taxa_name, plot_name, n_sub=None):
    # If tensor on GPU, move to CPU and convert to numpy
    if hasattr(reconstructed_ra_all, 'cpu'):
        reconstructed_ra_all = reconstructed_ra_all.cpu().detach().numpy()

    # ——— Build global taxa → color map ———
    all_taxa = sorted(taxa_name['V3'].unique())
    n_taxa = len(all_taxa)
    base_cmap = plt.get_cmap('tab20') if n_taxa <= 20 else plt.get_cmap('tab20b')
    taxa2color = {
        t: base_cmap(i / (n_taxa - 1))
        for i, t in enumerate(all_taxa)
    }

    # ——— Prepare data ———
    if n_sub is None:
        n_sub = reconstructed_ra_all.shape[1]
    df = pd.DataFrame(reconstructed_ra_all.T, columns=taxa_name['V3'])
    grouped = df.groupby(axis=1, level=0).sum()
    c_names = grouped.columns
    X = grouped.values[:n_sub, :]

    # sort samples by their maximum value
    idx_samp = np.argsort(-X.max(axis=1))
    X = X[idx_samp]

    # sort taxa by mean abundance across the selected samples
    means = X.mean(axis=0)
    idx_taxa = np.argsort(-means)
    X = X[:, idx_taxa]
    c_names = c_names[idx_taxa]

    # pull colors for bars in the new taxa order
    bar_colors = [taxa2color[t] for t in c_names]

    # ——— Plot bars ———
    bottom = np.zeros(n_sub)
    for i, t in enumerate(c_names):
        ax.bar(np.arange(n_sub), X[:, i], bottom=bottom, width=1.0, color=bar_colors[i])
        bottom += X[:, i]

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Proportion")
    ax.set_title(plot_name)

    return taxa2color  # return the mapping so we can build a shared legend later

def pcoa_side_by_side(
    data1, w1,
    data2, w2,
    metric='braycurtis',
    cmap_name='coolwarm',
    titles=('Dataset 1', 'Dataset 2'),
    vmin=None,
    vmax=None
):
    """
    Compute PCoA for two n×k matrices and plot them side by side,
    sharing a single continuous-color legend for w1 and w2.

    Parameters
    ----------
    data1, data2 : array-like, pandas.DataFrame, or torch.Tensor of shape (n_samples_i, n_features)
        If torch.Tensor, will be moved to CPU and converted to NumPy.
    w1, w2 : array-like or torch.Tensor of length n_samples_i (continuous numeric)
        If torch.Tensor, will be moved to CPU and converted to NumPy.
    metric : str, optional (default='braycurtis')
        Distance metric for scipy.spatial.distance.pdist.
    cmap_name : str, optional (default='viridis')
        Name of the matplotlib colormap to use.
    titles : tuple of length 2, optional (default=('Dataset 1', 'Dataset 2'))
        Titles for the left and right subplots, respectively.

    Returns
    -------
    fig, (ax1, ax2) : matplotlib.Figure, tuple of matplotlib.Axes
    """
    # 1) Convert torch.Tensors → NumPy if needed
    try:
        import torch
        if isinstance(data1, torch.Tensor):
            data1 = data1.cpu().detach().numpy()
        if isinstance(data2, torch.Tensor):
            data2 = data2.cpu().detach().numpy()
        if isinstance(w1, torch.Tensor):
            w1 = w1.cpu().detach().numpy()
        if isinstance(w2, torch.Tensor):
            w2 = w2.cpu().detach().numpy()
    except ImportError:
        pass

    # 2) Prepare X1, X2 matrices and optional sample IDs (not used in scatter)
    if isinstance(data1, pd.DataFrame):
        X1 = data1.values
    else:
        X1 = np.asarray(data1)

    if isinstance(data2, pd.DataFrame):
        X2 = data2.values
    else:
        X2 = np.asarray(data2)

    n1 = X1.shape[0]
    n2 = X2.shape[0]

    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    if w1.shape[0] != n1:
        raise ValueError("Length of w1 must match number of samples in data1.")
    if w2.shape[0] != n2:
        raise ValueError("Length of w2 must match number of samples in data2.")

    # 4) Compute PCoA coordinates for each dataset
    pc1_1, pc2_1, pct1_1, pct2_1 = compute_pcoa_coords(X1)
    pc1_2, pc2_2, pct1_2, pct2_2 = compute_pcoa_coords(X2)

    # 5) Determine shared color normalization across w1 and w2
    w_combined = np.concatenate([w1, w2])
    if vmin is None:
        vmin = min(w1.min(), w2.min())
    if vmax is None:
        vmax = max(w1.max(), w2.max())

    norm = TwoSlopeNorm(vmin=vmin, vcenter = 0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # 6) Create side-by-side subplots
    # Combine to find default vmin/vmax, but override if user passed them:
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=False, sharey=False)

    # Left subplot (data1)
    sc1 = ax1.scatter(
        pc1_1, pc2_1,
        c=w1,
        cmap=cmap,
        norm=norm,
        s=50,
        edgecolor='k',
        linewidth=0.5
    )
    ax1.set_xlabel(f"PC1 ({pct1_1:.2f}%)")
    ax1.set_ylabel(f"PC2 ({pct2_1:.2f}%)")
    ax1.set_title(titles[0])

    # Right subplot (data2)
    sc2 = ax2.scatter(
        pc1_2, pc2_2,
        c=w2,
        cmap=cmap,
        norm=norm,
        s=50,
        edgecolor='k',
        linewidth=0.5
    )
    ax2.set_xlabel(f"PC1 ({pct1_2:.2f}%)")
    ax2.set_ylabel(f"PC2 ({pct2_2:.2f}%)")
    ax2.set_title(titles[1])

    # 7) Add a single colorbar on the right, shared by both subplots
    cbar = fig.colorbar(
        sc1,
        ax=[ax1, ax2],
        fraction=0.046,
        pad=0.04
    )
    cbar.set_label("log(w)")

    plt.tight_layout(rect=[0, 0, 0.85, 1.00])
    plt.show()

    return fig, (ax1, ax2)


def pcoa_two_group(
    data1, w1,
    data2, w2,
    metric='braycurtis',
    marker1='o',
    marker2='^',
    cmap_name='viridis',
    labels=('Group 1', 'Group 2'),
    title="PCoA: two groups",
    vmin=None,
    vmax=None
):
    """
    Stack two datasets (n1×k and n2×k), run PCoA on the combined matrix,
    and plot PC1 vs. PC2 with different markers for each group and
    continuous coloring according to w1, w2.

    Parameters
    ----------
    data1 : array-like, pandas.DataFrame, or torch.Tensor of shape (n1, k)
        First group’s feature matrix.
    w1 : array-like or torch.Tensor of length n1 (continuous)
        Continuous values for coloring the first group.
    data2 : array-like, pandas.DataFrame, or torch.Tensor of shape (n2, k)
        Second group’s feature matrix.
    w2 : array-like or torch.Tensor of length n2 (continuous)
        Continuous values for coloring the second group.
    metric : str, optional (default='braycurtis')
        Distance metric passed to scipy.spatial.distance.pdist.
    marker1 : str, optional (default='o')
        Marker style for the first group.
    marker2 : str, optional (default='^')
        Marker style for the second group.
    cmap_name : str, optional (default='viridis')
        Name of matplotlib colormap for continuous coloring.
    labels : tuple of length 2, optional
        Legend labels for (data1, data2).
    title : str, optional
        Title for the plot.

    Returns
    -------
    fig, ax : matplotlib.Figure, matplotlib.Axes
    """
    # 1) Convert torch.Tensor to NumPy if needed
    try:
        import torch
        if isinstance(data1, torch.Tensor):
            data1 = data1.cpu().detach().numpy()
        if isinstance(data2, torch.Tensor):
            data2 = data2.cpu().detach().numpy()
        if isinstance(w1, torch.Tensor):
            w1 = w1.cpu().detach().numpy()
        if isinstance(w2, torch.Tensor):
            w2 = w2.cpu().detach().numpy()
    except ImportError:
        pass

    # 2) Extract NumPy arrays from DataFrames or array-likes
    if isinstance(data1, pd.DataFrame):
        X1 = data1.values
    else:
        X1 = np.asarray(data1)
    if isinstance(data2, pd.DataFrame):
        X2 = data2.values
    else:
        X2 = np.asarray(data2)

    w1 = np.asarray(w1)
    w2 = np.asarray(w2)
    n1, k1 = X1.shape
    n2, k2 = X2.shape
    if k1 != k2:
        raise ValueError("Both datasets must have the same number of columns (features).")

    if w1.shape[0] != n1 or w2.shape[0] != n2:
        raise ValueError("Length of w1/w2 must match number of samples in data1/data2 respectively.")

    # 3) Stack data and w
    X = np.vstack([X1, X2])       # shape (n1+n2, k)
    w_combined = np.concatenate([w1, w2])
    n = n1 + n2

    # 4) Compute pairwise distance matrix D (n×n)
    dist_vec = pdist(X, metric=metric)
    D = squareform(dist_vec)

    # 5) Double-center to get B = -0.5 * J * D^2 * J
    D2 = D**2
    J = np.eye(n) - np.ones((n, n))/n
    B = -0.5 * J.dot(D2).dot(J)

    # 6) Eigen-decompose B
    eigvals, eigvecs = np.linalg.eigh(B)
    idx_desc = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_desc]
    eigvecs = eigvecs[:, idx_desc]

    # 7) Compute coordinates (clip negatives)
    eigvals_clipped = np.where(eigvals > 0, eigvals, 0.0)
    coords = eigvecs * np.sqrt(eigvals_clipped)   # shape (n, n)
    pc1 = coords[:, 0]
    pc2 = coords[:, 1]

    total_var = eigvals_clipped.sum()
    pct1 = 100 * eigvals_clipped[0] / total_var if total_var > 0 else 0.0
    pct2 = 100 * eigvals_clipped[1] / total_var if total_var > 0 else 0.0

    # 8) Set up continuous colormap normalization over combined w
    if vmin is None:
        vmin = min(w1.min(), w2.min())
    if vmax is None:
        vmax = max(w1.max(), w2.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter = 0, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    # 9) Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # First group points (indices 0 to n1-1), small and borderless
    sc1 = ax.scatter(
        pc1[:n1], pc2[:n1],
        c=w1,
        cmap=cmap,
        norm=norm,
        marker=marker1,
        edgecolors='none',
        s=30,
        alpha=0.9,
        label=labels[0]
    )
    # Second group points (indices n1 to n1+n2-1), small and borderless
    sc2 = ax.scatter(
        pc1[n1:], pc2[n1:],
        c=w2,
        cmap=cmap,
        norm=norm,
        marker=marker2,
        edgecolors='none',
        s=30,
        alpha=0.9,
        label=labels[1]
    )

    ax.set_xlabel(f"PC1 ({pct1:.2f}%)")
    ax.set_ylabel(f"PC2 ({pct2:.2f}%)")
    ax.set_title(title)

    # 10) Create legend for group markers
    legend_elements = [
        Line2D([0], [0], marker=marker1, color='w',
               markerfacecolor='gray', label=labels[0], markersize=8, markeredgecolor='k'),
        Line2D([0], [0], marker=marker2, color='w',
               markerfacecolor='gray', label=labels[1], markersize=8, markeredgecolor='k')
    ]
    ax.legend(handles=legend_elements, title="Group", loc='upper right')

    # 11) Add a single colorbar for w
    cbar = fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log(w)")

    # 12) Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.show()
    return fig, ax


def _reduce_for_plot(samples_np, other_np, taxa_name, top_k=30, max_samples=200, random_state=0):
    """
    Reduce (n_samples, K) to (n_plot_samples, K') by:
      - picking top_k taxa by global mean (using samples+other)
      - aggregating the rest into an 'Other' column
      - subsampling samples to max_samples
    Returns: reduced_samples (n_plot_samples, top_k+1), reduced_taxa_name (DataFrame with V3)
    """
    import numpy as np, pandas as pd
    # samples_np, other_np: (n, K)
    K = samples_np.shape[1]
    # rank taxa by global mean across BOTH sets so colors/legend align
    pooled_mean = np.concatenate([samples_np, other_np], axis=0).mean(axis=0)  # (K,)
    top_idx = np.argsort(-pooled_mean)[:top_k]
    rest_idx = np.setdiff1d(np.arange(K), top_idx, assume_unique=False)

    def _collapse(X):
        top = X[:, top_idx]
        other = X[:, rest_idx].sum(axis=1, keepdims=True) if rest_idx.size else np.zeros((X.shape[0],1))
        return np.concatenate([top, other], axis=1)

    samples_red = _collapse(samples_np)
    other_red   = _collapse(other_np)

    # consistent taxa labels
    def _extract_labels(taxa_like):
        import pandas as pd, numpy as np
        if isinstance(taxa_like, pd.DataFrame):
            for col in ('V3','taxon','taxa','name'):
                if col in taxa_like.columns: 
                    return taxa_like[col].astype(str).tolist()
            return taxa_like.iloc[:,0].astype(str).tolist()
        if isinstance(taxa_like, (pd.Series, pd.Index)): return taxa_like.astype(str).tolist()
        return list(map(str, np.asarray(taxa_like)))
    labels_full = _extract_labels(taxa_name)
    labels_red = [labels_full[i] for i in top_idx] + ["Other"]
    taxa_name_red = pd.DataFrame({"V3": labels_red})

    # subsample samples for display
    rng = np.random.default_rng(random_state)
    def _subsample(X):
        if X.shape[0] > max_samples:
            keep = rng.choice(X.shape[0], size=max_samples, replace=False)
            return X[keep]
        return X

    return _subsample(samples_red), _subsample(other_red), taxa_name_red