# %%
# ----- 0. Load library -----
import torch
import os
import sys
os.chdir("/hpc/home/yx306/RDR")


agp_data_path = "/hpc/group/mastatlab/yx306/AGP/data/"
ganchao_path = "/hpc/group/mastatlab/microbiome/clean/"
sys.path.append(ganchao_path)
import microbiome_unet as ganchao
import utils.AGP_help as agp_help



from importlib import reload
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from scipy.stats import describe
# from utils import *
import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import describe
from keras.models import load_model
from sklearn.model_selection import train_test_split
import torch
import utils.help_func as help
import utils.microbiome_help as mb_help

# DRE functions
import utils.DRE_func as dre
# %% 
# load data
x_lt = np.loadtxt(agp_data_path + "yx1_abundance_lt_train.csv", delimiter=",")
x_train_raw = np.loadtxt(agp_data_path + "yx1_xtrain_raw.csv", delimiter=",")
x_test_raw = np.loadtxt(agp_data_path + "yx1_xtest_raw.csv", delimiter=",")

path = f"{agp_data_path}/yx2_metadata.csv"
metadata = pd.read_csv(path, index_col=0)



# to reconstruct lt
abundance = pd.read_csv(agp_data_path+"yx1_abundance.csv", index_col=0)

taxa_list = abundance.index

saveFolder = agp_data_path
# 1) child node matrix (keep labels as strings)
child_node_pd = pd.read_csv(
    f"{saveFolder}/yx3_chid_node_df.csv",
    index_col=0,
    dtype=str
)

# 2) leaf/taxa names from the tree (single-column DF of strings)
tree_taxa_pd = pd.read_csv(
    f"{saveFolder}/yx4_taxa_name_df.csv",
    index_col=0,
    dtype=str
)
# If you want a 1D array:
tree_taxa = tree_taxa_pd.squeeze("columns").to_numpy()


# 4) taxonomy table (keep strings; don't auto-convert 'NA' to NaN)
taxa_df = pd.read_csv(
    f"{saveFolder}/yx6_taxa_df.csv",
    index_col=0,
    dtype=str,
    keep_default_na=False
)

out_names = abundance.index.values
outTaxa_pd = pd.DataFrame(out_names)
x_lt_torch = torch.tensor(x_lt).to(torch.float32).to(device)
# %%
# # train ICFM
# 
batch_size = round(x_lt_torch.shape[0]/10)
dim = x_lt_torch.shape[1]

# warmup = 1000
# n_epochs = 8000
# # warmup = 100 # for test
# # n_epochs = 200 # for test
# lr = 1e-5
# sig_use = 1e-4 # Gaussian noise
# w = 128 # channel



# icfm_lt, ema_icfm_lt = ganchao.train_model(x_lt_torch, nn_structure = 'unet', cfm_method = 'icfm',
#                                    xcov = None, n_epochs = n_epochs, batch_size = batch_size,
#                                    ImpSamp = False, beta_a = 0., beta_b = 0.,
#                                    sigma = sig_use, lr = lr, grad_clip = 1.0, ema_decay = 0.9999,
#                                     w = w,
#                                   weight_decay= 0.)
# torch.save({
#     "icfm_lt_state": icfm_lt.state_dict(),
#     "ema_icfm_lt_state": ema_icfm_lt.state_dict(),
#     "epoch": n_epochs,
#     "lr": lr,
#     "config": {
#         "nn_structure": "unet",
#         "cfm_method": "icfm",
#         "batch_size": batch_size,
#         "sigma": sig_use,
#         "w": w,
#     },
# }, agp_data_path+"icfm_checkpoint.pth")

# %%
# reload
# ----- 1. Load checkpoint -----
checkpoint = torch.load(agp_data_path + "icfm_checkpoint.pth", map_location="cuda")

# ----- 2. Recreate models using saved config -----
cfg = checkpoint["config"]
lr = 1e-5
sig_use = 1e-4 # Gaussian noise
w = 128 # channel
# Recreate model manually with same hyperparameters
icfm_lt, ema_icfm_lt = ganchao.train_model(
    x_lt_torch, 
    nn_structure="unet",
    cfm_method="icfm",
    xcov=None,
    n_epochs=0,          # 0 so it doesn’t train again
    batch_size=cfg["batch_size"],
    sigma=sig_use,
    lr=lr,
    w=w,
)
# Then load weights
icfm_lt.load_state_dict(checkpoint["icfm_lt_state"])
ema_icfm_lt.load_state_dict(checkpoint["ema_icfm_lt_state"])


# %%
# generation
n_samp = x_test_raw.shape[0]
nt_gen = 2
seed = 2

icfm_lt_samp = ganchao.gen_samp(ema_icfm_lt.to(device), 
    dim, n_samp, nt_gen, seed, x_start = None, z = None)

# %%
# PCoA plot
agp_help.pcoa_two_group(x_lt_torch[0:1000,:], icfm_lt_samp[0:1000,:], 
                labels=('Observed','ICFM'),
               metric='braycurtis', show=False)

# %%

# recon test

icfm_recon = x_recon = ganchao.reconstruct_lt_all(
    icfm_lt_samp, 
    child_node_pd,
    tree_taxa_pd,
    outTaxa_pd
) # takes 8m
# PCoA plot
agp_help.pcoa_two_group(x_train_raw[0:1000,:], icfm_recon, 
                labels=('Observed','ICFM'),
               metric='braycurtis', show=False)

# %%
# on test
icfm_recon_test = ganchao.reconstruct_lt_all(
    icfm_lt_samp, 
    child_node_pd,
    tree_taxa_pd,
    outTaxa_pd
) # takes 8m
# PCoA plot
agp_help.pcoa_two_group(x_test_raw, icfm_recon_test, 
                labels=('Observed','ICFM'),
               metric='braycurtis', show=False)




# %%
# generate for DRE training
n_samp = x_train_raw.shape[0]
nt_gen = 2
seed = 2

icfm_lt_samp = ganchao.gen_samp(ema_icfm_lt.to(device), 
    dim, n_samp, nt_gen, seed, x_start = None, z = None)
torch.save(icfm_lt_samp, agp_data_path + '/yx1_icfm_lt_train.pth')

# %%
icfm_recon_train = ganchao.reconstruct_lt_all(
    icfm_lt_samp, 
    child_node_pd,
    tree_taxa_pd,
    outTaxa_pd
) # takes 8m
# PCoA plot
# agp_help.pcoa_two_group(x_train_raw, icfm_recon_train, 
#                 labels=('Observed','ICFM'),
#                metric='braycurtis', show=False)



np.savetxt(agp_data_path + '/yx1_icfm_recon_train.csv', icfm_recon_train, delimiter=",")
# %%
# ------------------ analysis from here --------------------------

# Analysis 1. load test data
# evaluate x_test_raw vs icfm_test

icfm_test = np.loadtxt(agp_data_path + '/yx1_icfm_recon_test.csv', delimiter=",")

# transform to lt space
# LT transform
chid_node_pd = pd.read_csv(saveFolder + '/yx3_chid_node_df.csv', index_col=0)
child_node_matrix = chid_node_pd
tree_taxa_pd = pd.read_csv(saveFolder + '/yx4_taxa_name_df.csv', index_col=0)
taxa_names = tree_taxa_pd.values
out_names = abundance.index.values
outTaxa_pd = pd.DataFrame(out_names)
out_names = outTaxa_pd.values

icfm_test_lt = ganchao.smooth_transformation(icfm_test,
                             child_node_matrix = child_node_matrix,
                             out_names = out_names.flatten(),
                             taxa_names = taxa_names.flatten(),
                             epsilon=1e-7,
                             noise_scale=0.) # takes 6s

x_test_lt = ganchao.smooth_transformation(x_test_raw,
                             child_node_matrix = child_node_matrix,
                             out_names = out_names.flatten(),
                             taxa_names = taxa_names.flatten(),
                             epsilon=1e-7,
                             noise_scale=0.) # takes 4s

num_epochs = 3000

icfm_test_torch = torch.tensor(icfm_test, dtype=torch.float32).to(device)
x_test_torch = torch.tensor(x_test_raw, dtype=torch.float32).to(device)

icfm_test_lt_torch = torch.tensor(icfm_test_lt, dtype=torch.float32).to(device)
x_test_lt_torch = torch.tensor(x_test_lt, dtype=torch.float32).to(device)

x_mixed = torch.cat([x_test_torch, icfm_test_torch], dim=0) 
x_mixed_lt = torch.cat([x_test_lt_torch, icfm_test_lt_torch], dim=0) 

model_r, losses_r = dre.run_DRE_fdiv(x_test_torch,x_mixed,xi=0.5,
                                num_epochs = num_epochs,
                                log_scale = False,
                                loss_method='Hellinger')

# model_lt, losses_lt = dre.run_DRE_fdiv(x_test_lt_torch,x_mixed_lt,xi=0.5,
#                                 num_epochs = num_epochs,
#                                 log_scale = False,
#                                 loss_method='Hellinger')

# %%
# check visualz

# # 1. PCoA
plt.figure(figsize=(6,4))
plt.rcParams.update({
    "font.size": 14,           # base font size
    "axes.titlesize": 16,      # title
    "axes.labelsize": 14,      # x/y labels
    "xtick.labelsize": 12,     # x ticks
    "ytick.labelsize": 12,     # y ticks
    "legend.fontsize": 12,     # legend
})
agp_help.pcoa_two_group(x_test_torch, icfm_test_torch, 
                labels=('Observed','ICFM'),
               metric='braycurtis', show=False)

# 2. hist
g_p, g_q = dre.evaluate_model(model_r, x_test_torch, icfm_test_torch)
bool_density = False
ax, bin_edges = help.plot_ratio_hist(g_p.detach().cpu().numpy(), 
                                     bins=50, range=(0,2), density=bool_density,
                                     figsize = (5,4),
                                     label="p-observed sample", color="tab:purple")
help.add_ratio_hist(g_q.detach().cpu().numpy(), ax=ax, bin_edges=bin_edges, 
                    density=bool_density,
                    label="q (AGP ICFM)", color="tab:green")

last_epoch = losses_r[(num_epochs-100):num_epochs]
mean_last_epoch = float(np.mean(last_epoch))
ax.set_title(f"Histogram ddim, h^2(p,(p+q)/2) = {-mean_last_epoch:.3f}")
plt.tight_layout(); plt.show()# %%

# 3. check summary stats
stats = {
    "g_p_ICFM": help.summarize_vector(g_p.detach().cpu().numpy()),
    "g_q_ICFM": help.summarize_vector(g_q.detach().cpu().numpy()),
}

df = pd.DataFrame(stats)
print(df.round(2))  # show 4 digits

# %%
# 4. stacked bar plot


path = os.path.join(agp_data_path, 'yx6_taxa_df.csv')
taxa_df = pd.read_csv(path, index_col=0) 
taxa_df.columns = [f"V{i}" for i in range(1, 8)]

reload(mb_help)
mb_help.stratified_stacked_barplot(g_q, g_p, 
                                    icfm_test_torch, 
                                    x_test_torch,taxa_df,
    method="ICFM", 
    taxa_level = "V3",
    stratify=False, top_k=50, max_samples=1000, random_state=1,
                                    figsize=(18, 10),)



# %%
# 5. test for association
from scipy.stats import spearmanr

# ---- helper: works for numpy or torch (CPU/GPU) ----
# ---------- helpers ----------
# ---------- helpers ----------
def to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def clr_transform(X, eps=1e-8):
    Xp = X + eps
    logX = np.log(Xp)
    return logX - logX.mean(axis=1, keepdims=True)

def aggregate_to_V3(X, taxa_df, level ="V4"):
    """
    X: (n, K_species) at V7; taxa_df has column 'V3' (length K_species), aligned to X columns
    returns: X_V3 (n, L), V3_labels (list of length L)
    """
    v3 = taxa_df[level].astype(str).values
    df = pd.DataFrame(X.T, index=v3)
    X_V3_df = df.groupby(level=0).sum().T     # (n, L)
    return X_V3_df.values, X_V3_df.columns.tolist()

def bh_fdr(p):
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    ranks = np.empty(m, int); ranks[order] = np.arange(1, m+1)
    q = p * m / ranks
    # monotone
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    q = np.empty_like(q); q[order] = q_sorted
    return np.clip(q, 0, 1)

def spearman_block_permutation(y, X, blocks=None, B=200, rng=None):
    """
    Return permutation p-values for Spearman |rho|, optionally permuting y within blocks.
    y: (n,), X: (n, K), blocks: (n,) with group ids, B permutations
    """
    rng = np.random.default_rng(rng)
    n, K = X.shape
    # observed
    r_obs = np.empty(K)
    for j in range(K):
        r, _ = spearmanr(y, X[:, j])
        r_obs[j] = r
    abs_obs = np.abs(r_obs)

    # permutation distribution of |rho| per feature
    ge_count = np.zeros(K, int)
    y_perm = y.copy()
    if blocks is None:
        idx = np.arange(n)
        for _ in range(B):
            rng.shuffle(idx)
            y_perm[:] = y[idx]
            for j in range(K):
                r, _ = spearmanr(y_perm, X[:, j])
                if np.abs(r) >= abs_obs[j]:
                    ge_count[j] += 1
    else:
        # permute labels within each block
        blocks = np.asarray(blocks)
        uniq = np.unique(blocks)
        block_indices = [np.where(blocks==b)[0] for b in uniq]
        for _ in range(B):
            y_perm[:] = y
            for bi in block_indices:
                shuf = bi.copy()
                rng.shuffle(shuf)
                y_perm[bi] = y[shuf]
            for j in range(K):
                r, _ = spearmanr(y_perm, X[:, j])
                if np.abs(r) >= abs_obs[j]:
                    ge_count[j] += 1

    p_perm = (ge_count + 1) / (B + 1)  # add-one smoothing
    return r_obs, p_perm

# ---------- build stacked data ----------
# outcomes in [0,2]
g = np.concatenate([to_np(g_p).ravel(), to_np(g_q).ravel()], axis=0).astype(float)
y01 = g / 2.0  # scale to [0,1]

# features at V7 (species), stacked
X_species = np.vstack([to_np(x_test_raw), to_np(icfm_test)])   # shape (n, K_species)
# indicator to guard against group confounding (0=train, 1=generated)
# block = np.concatenate([np.zeros(len(x_test_raw), int), np.ones(len(icfm_test), int)], axis=0)

# ---------- aggregate to V3 ----------
X_V3, V3_labels = aggregate_to_V3(X_species, taxa_df)          # (n, L)
X_V3_clr = clr_transform(X_V3)

X_V3, V3_labels = aggregate_to_V3(X_species, taxa_df,"V5")          # (n, L)
X_V3_clr = clr_transform(X_V3)

# spieces level
# X_V3_clr = clr_transform(X_species)
# V3_labels = taxa_df["V7"]

# ---------- Spearman (marginal) ----------
rhos = []; pvals = []
for j in range(X_V3_clr.shape[1]):
    r, p = spearmanr(y01, X_V3_clr[:, j])
    rhos.append(r); pvals.append(p)

res = pd.DataFrame({
    "V3": V3_labels,
    "rho": rhos,
    "abs_rho": np.abs(rhos),
    "pval_asym": pvals
})
res["qval_bh"] = bh_fdr(res["pval_asym"].values)

# OPTIONAL: block-permutation p-values (recommended if train vs generated differ a lot)
# (keeps y permutation within each block to avoid spurious significance)
# This is ~ (L * B) Spearman calls; with L≈50 and B=200 it's fine.


r_obs, p_perm = spearman_block_permutation(y01, X_V3_clr, blocks=None, B=200, rng=1)
res["pval_perm"] = p_perm
res["qval_perm_bh"] = bh_fdr(p_perm)

# # sort by effect size
# res = res.sort_values("qval_perm_bh", ascending=True).reset_index(drop=True)
# print(res.head(10))


# taxa with q < 0.05, ranked by |rho| desc (and q asc as a tiebreaker)
sig = (
    res.loc[(res["qval_perm_bh"] < 0.05 )& (res["abs_rho"] > 0.3 )]
       .sort_values(["abs_rho", "qval_perm_bh"], ascending=[False, True], kind="mergesort")
       .reset_index(drop=True)
)

# display the key columns
cols = ["V3", "rho", "abs_rho", "pval_perm", "qval_perm_bh"]
print(sig[cols].to_string(index=False))
sig[cols].round(3)

# take top 10 taxa by absolute correlation
top10 = sig.head(10).copy()

# display rounded results
cols = ["V3", "rho", "abs_rho", "pval_perm", "qval_perm_bh"]
top10[cols].round(3)
# print(top10[cols].round(3).to_string(index=False))


# # Lollipop chart
# df = sig.nlargest(30, "abs_rho").sort_values("rho")  # e.g., top 30
# y = np.arange(len(df))

# fig, ax = plt.subplots(figsize=(7, max(3, 0.35*len(df))))
# ax.hlines(y, 0, df["rho"].values, linewidth=2)
# ax.plot(df["rho"].values, y, "o")
# ax.axvline(0, linestyle="--", linewidth=1)
# ax.set_yticks(y)
# ax.set_yticklabels(df["V3"].values)
# ax.set_xlabel("Spearman rho")
# ax.set_title("Top taxa by |rho|")
# plt.tight_layout()
# plt.show()

# %%
# show taxa lineage
taxa_df.index.name = "index"
taxa_df.columns = [f"V{i}" for i in range(1, 8)]

import numpy as np, pandas as pd
from scipy.stats import spearmanr

def assoc_at_level(level, X_species, taxa_df, y01, B=0, rng=1):
    """
    Compute per-taxon Spearman rho at a given taxonomy level.
    Returns: DataFrame with columns [level, label, rho, abs_rho, p_perm(optional), q_perm(optional)]
    """
    # 1) aggregate to `level` (pass-through for V7)
    if level == "V7":
        X = X_species
        labels = taxa_df["V7"].to_numpy()
    else:
        X, labels = aggregate_to_V3(X_species, taxa_df, level)  # your aggregator supports rank arg

    # 2) CLR (ensure zeros handled upstream)
    X_clr = clr_transform(X)

    # 3) Spearman per feature
    rhos = np.empty(X_clr.shape[1])
    for j in range(X_clr.shape[1]):
        rhos[j], _ = spearmanr(y01, X_clr[:, j])

    out = pd.DataFrame({
        "level": level,
        "label": labels,
        "rho": rhos,
        "abs_rho": np.abs(rhos),
    })

    return out

# compute at multiple levels and concat
levels = ["V1","V2","V3","V4","V5","V6","V7"]
assoc_all = pd.concat([assoc_at_level(L, X_species, taxa_df, y01, B=0) for L in levels],
                      ignore_index=True)
# %%
# sunburst plot
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import TwoSlopeNorm
import matplotlib.patheffects as pe

def sunburst_matplotlib_labeled(
    taxa_df, assoc_all, outfile="taxonomy_sunburst_labeled.png",
    levels=("V1","V2","V3","V4","V5","V6","V7"),
    figsize=(12, 12), dpi=300,
    max_labels_per_level=18,
    min_deg=5.0,            # don’t label wedges narrower than this angle
    fontbase=8
):
    # ---------- clean ----------
    X = taxa_df.loc[:, levels].replace({"-": np.nan, "": np.nan})
    for c in X.columns:
        X[c] = X[c].astype(str).str.strip().replace({"nan": np.nan})

    A = assoc_all.copy()
    A["level"] = A["level"].astype(str).str.strip()
    A["label"] = A["label"].astype(str).str.strip()
    A["rho"]   = pd.to_numeric(A["rho"], errors="coerce")
    rho_map = {(r.level, r.label): r.rho for r in A.itertuples(index=False)}

    def short_name(s):
        return s.split("__", 1)[-1] if "__" in s else s

    # ---------- tree ----------
    class Node:
        __slots__=("level","label","children","size","rho")
        def __init__(self, level, label):
            self.level = level; self.label = label
            self.children = {}; self.size = 0
            self.rho = rho_map.get((level, label), np.nan) if level in levels else np.nan

    root = Node("ROOT","All taxa")
    for _, row in X.iterrows():
        cur = root; cur.size += 1
        for lvl in levels:
            val = row[lvl]
            if pd.isna(val): break
            if val not in cur.children:
                cur.children[val] = Node(lvl, val)
            cur = cur.children[val]; cur.size += 1

    # ---------- color scale ----------
    def gather_rhos(n):
        out = [n.rho] if np.isfinite(n.rho) else []
        for ch in n.children.values(): out += gather_rhos(ch)
        return out
    rhos = np.array(gather_rhos(root))
    vmax = np.percentile(np.abs(rhos[np.isfinite(rhos)]), 95) if np.any(np.isfinite(rhos)) else 1.0
    vmax = 1.0 if vmax == 0 else float(vmax)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdBu_r

    # ---------- draw ----------
    n_levels = len(levels)
    ring_w   = 1.0 / (n_levels + 1)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_aspect("equal", "box"); ax.axis("off")

    labels_left = {lvl: max_labels_per_level for lvl in levels}

    def draw_node(node, start_deg, span_deg, depth):
        if depth >= 0:
            r_inner = (depth + 1) * ring_w
            r_outer = r_inner + ring_w
            color = cmap(norm(node.rho if np.isfinite(node.rho) else 0.0))
            ax.add_patch(Wedge((0,0), r_outer, start_deg, start_deg+span_deg,
                               width=ring_w, facecolor=color,
                               edgecolor="white", lw=0.6))

            # label (budgeted, big slices only)
            lvl = levels[depth]
            if labels_left[lvl] > 0 and span_deg >= min_deg:
                mid  = start_deg + span_deg/2.0
                theta= np.deg2rad(mid)
                r_text = r_inner + 0.55*ring_w
                txt = short_name(node.label)
                rot = mid - 90
                if 90 < mid < 270:  # keep upright
                    rot += 180
                t = ax.text(r_text*np.cos(theta), r_text*np.sin(theta),
                            txt, rotation=rot, ha="center", va="center",
                            fontsize=fontbase + max(0, 4 - depth),
                            color="black", clip_on=True)
                # thin white halo for contrast
                t.set_path_effects([pe.Stroke(linewidth=2.2, foreground="white"), pe.Normal()])
                labels_left[lvl] -= 1

        if depth+1 >= n_levels or not node.children:
            return
        total = sum(ch.size for ch in node.children.values())
        cur_s = start_deg
        for _, ch in sorted(node.children.items(), key=lambda kv: -kv[1].size):
            frac = ch.size / total
            ang  = span_deg * frac
            draw_node(ch, cur_s, ang, depth+1)
            cur_s += ang

    draw_node(root, 0.0, 360.0, depth=-1)

    # LOCK THE VIEWPORT so text doesn’t rescale the axes
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.autoscale(False)

    # colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Association ρ", rotation=90)

    ax.set_title("Taxonomy Sunburst — size = lineage count, color = association (ρ)", pad=16)
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved labeled sunburst PNG to: {outfile}")

# usage
sunburst_matplotlib_labeled(
    taxa_df, assoc_all,
    outfile="taxonomy_sunburst_labeled.png",
    max_labels_per_level=5,   # tweak per your preference
    min_deg=6.0,               # increase to suppress tiny labels
    fontbase=5
)



# %%
# check covariate data association


metadata_test = pd.read_csv(f"{agp_data_path}/yx2_metadata_test.csv", index_col=0)
import numpy as np, pandas as pd, statsmodels.api as sm
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import pandas as pd

# prescreeening

y = np.asarray(g_p.detach().cpu().numpy()).reshape(-1)

from statsmodels.stats.multitest import multipletests
import pandas as pd

pvals = {}

for col in metadata_test.columns:
    s = metadata_test[col]
    # skip columns with ≤1 unique level
    if s.dropna().nunique() <= 1:
        continue
    
    # group y by each level (drop NaN)
    groups = [y[s == lvl] for lvl in s.dropna().unique()]
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) >= 2:
        stat, p = kruskal(*groups)
        pvals[col] = p

res_df = pd.DataFrame({
    "variable": list(pvals.keys()),
    "pval": list(pvals.values())
})
res_df["qval"] = multipletests(res_df["pval"], method="fdr_bh")[1]
res_df = res_df.sort_values("pval")
print(res_df.head(20))

selected_vars = res_df.query("pval < 0.05")["variable"].tolist()

# selected_vars = res_df.head(50)["variable"].tolist()
# len(selected_vars)

# # try regression
# X_sel = metadata_test[selected_vars]
# X_sel = pd.get_dummies(X_sel, drop_first=True)
# X_sel = X_sel.replace([np.inf,-np.inf], np.nan)
# X_sel = X_sel.fillna(X_sel.median(numeric_only=True)).astype(float)


# # Fractional logit (Binomial GLM)
# glm = sm.GLM(y/2.0, X_sel, family=sm.families.Binomial())
# res = glm.fit(cov_type="HC3")   # robust SEs
# print(res.summary())

# %%
# heatmap for taxa corrrelation comparison
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt

# Suppose X_real and X_gen are CLR-transformed (n_real x K) and (n_gen x K)
n_real = x_test_raw.shape[0]
n_gen  = icfm_test.shape[0]

X_real = X_V3_clr[:n_real, :]                   # first block: observed samples
X_gen  = X_V3_clr[n_real : n_real + n_gen, :]   
K = X_real.shape[1]
taxa_labels = V3_labels

# Compute pairwise Spearman correlations
R_real = pd.DataFrame(spearmanr(X_real).correlation, index=taxa_labels, columns=taxa_labels)
R_gen  = pd.DataFrame(spearmanr(X_gen).correlation,  index=taxa_labels, columns=taxa_labels)

# Option A: side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(R_real, ax=axes[0], cmap="coolwarm", center=0, vmin=-1, vmax=1)
axes[0].set_title("Observed taxa correlations")
sns.heatmap(R_gen, ax=axes[1], cmap="coolwarm", center=0, vmin=-1, vmax=1)
axes[1].set_title("Generated taxa correlations")
plt.tight_layout()

# Option C: correlation of correlations
corr_of_corrs, _ = spearmanr(R_real.values.ravel(), R_gen.values.ravel())
print(f"Correlation of correlations: {corr_of_corrs:.3f}")



# %%
# alpha-diversity
from scipy.stats import mannwhitneyu

# Example: split from your stacked matrix
n_real = x_test_raw.shape[0]
n_gen  = icfm_test.shape[0]

# Use species-level or V3-level on the simplex (NOT CLR):
X = np.vstack([to_np(x_test_raw), to_np(icfm_test)])  # (n, K)
group = np.array(["Observed"]*n_real + ["Generated"]*n_gen)

# (optional) ensure tiny eps to avoid log(0)
X = np.clip(X, 1e-12, None)
X = X / X.sum(axis=1, keepdims=True)

def shannon(p):
    return -(p * np.log(p)).sum()

def gini_simpson(p):
    return 1.0 - (p**2).sum()

def richness(p, thresh=0):
    return (p > thresh).sum()

H   = np.apply_along_axis(shannon, 1, X)
GS  = np.apply_along_axis(gini_simpson, 1, X)
S0  = np.apply_along_axis(richness, 1, X)  # observed taxa

df_alpha = pd.DataFrame({
    "Group": group,
    "Shannon": H,
    "GiniSimpson": GS,
    "Richness": S0
})

def box_jitter(df, metric, ax):
    cats = df["Group"].unique()
    data = [df.loc[df["Group"]==c, metric].values for c in cats]
    # ax.boxplot(data, showfliers=False)
    # jitter
    for i, d in enumerate(data, start=1):
        x = np.random.normal(loc=i, scale=0.06, size=len(d))
        ax.plot(x, d, 'o', alpha=0.5, markersize=3)
    ax.set_xticks(range(1, len(cats)+1))
    ax.set_xticklabels(cats)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by group")

fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
for ax, m in zip(axes, ["Shannon", "GiniSimpson"]):
    box_jitter(df_alpha, m, ax)

# Stats (Observed vs Generated)
gA = df_alpha.loc[df_alpha["Group"]=="Observed"]
gB = df_alpha.loc[df_alpha["Group"]=="Generated"]
for m in ["Shannon", "GiniSimpson"]:
    stat, p = mannwhitneyu(gA[m], gB[m], alternative="two-sided")
    print(f"{m}: U={stat:.1f}, p={p:.3g}")
plt.show()



# %%
# Volcano / heatmap of significant taxa
X_V7_clr = clr_transform(X_species)

import dendropy
with zipfile.ZipFile(zip_content_otu) as z:
    tree_path = "03-otus/100nt/gg-13_8-97-percent/97_otus.tree"
    with z.open(tree_path) as f:
        newick_str = f.read().decode('utf-8')
tree = dendropy.Tree.get(data=newick_str, schema='newick')
# %%