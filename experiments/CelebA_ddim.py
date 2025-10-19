# %%
# 0. download pretrained DDIM
# pretrained model: https://github.com/ermongroup/ddim/tree/main
# import gdown

# # Replace with actual file ID from the Google Drive link in the DDIM README.
# # Example: if the link is
# # https://drive.google.com/file/d/1AbCdEfGhIjKlMnOp/view?usp=sharing
# # https://drive.google.com/file/d/1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR/view?usp=sharing
# # then the file_id = "1AbCdEfGhIjKlMnOp"
# file_id = "1R_H-fJYXSH79wfSKs9D-fuKQVan5L-GR"
# url = f"https://drive.google.com/uc?id={file_id}"

# # output_path = data_path
# gdown.download(url, output_path, quiet=False)

# %%
# # 0. set path
import os
os.chdir("/hpc/home/yx306/RDR")
os.getcwd() 
import pandas as pd
from importlib import reload
import utils.sampler_ddim_celeba64 as ddim
import utils.CelebA_help as celeb
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import utils.help_func as help
import matplotlib.pyplot as plt
import utils.DRE_batch as dre_batch
import utils.DRE_func as dre
import torch

# %%
# real data loader

data_path = "/hpc/group/mastatlab/yx306/"+"CelebA/"
# transform: crop → resize → tensor → normalize
transform = transforms.Compose([
    transforms.CenterCrop(178),   # crop the aligned faces
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # to [-1,1]
])

# download=True will fetch CelebA (~1.3 GB) into ./data
trainset = datasets.CelebA(
    root=data_path,
    split="train",
    transform=transform,
    download=False,
)

batch_size = 512
celeba_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


# # %%
# # not required: load DDIM pretraiend model
# # Paths you set:
# data_path = "/hpc/group/mastatlab/yx306/"+"CelebA/"
# REPO = "/hpc/group/mastatlab/yx306/CelebA/DDIM"                 # git clone https://github.com/ermongroup/ddim
# CKPT = data_path + "ckpt.pth"               # from the README’s CelebA-64 link

# # A) pure-Python internal sampler


# reload(ddim)

# x = ddim.ddim_sampler_celeba64(REPO, CKPT, n=512, steps=50, eta=0.0, device="cuda")
# print(x.shape, x.min().item(), x.max().item())  # (512, 3, 64, 64), approx in [-1,1]

# celeb.show_batch(
#     x, nrow=5, idx=list(range(25)), plot=True, show=True,
#     title="DDIM"
# )
# %%
# load mixture sampler from disk
reload(celeb)
ddim_data_dir = "/hpc/group/mastatlab/yx306/CelebA/DDIM/data"

q_mixed_sampler = celeb.DDIMDiskSampler(
    real_loader=celeba_loader,
    ddim_data_dir=ddim_data_dir,
    gen_frac=0.5,
    split="train",   
)


# Sample a batch
x_mixed = q_mixed_sampler(batch_size=batch_size)
print(x_mixed.shape)   # torch.Size([64, 3, 64, 64])




# %%
# # ---------- (takes a long time to run)try to run ratio estimator -------------#
# reload(dre)
# reload(dre_batch)
# model_ddim, losses_ddim = dre_batch.run_DRE_fdiv_cnn_minibatch_celeba64(
#         celeba_loader,                  # DataLoader yielding real CelebA-64 (N,3,64,64) in [-1,1]
#         q_mixed_sampler,                 # callable(bs, device, dtype) -> (N,3,64,64) in [-1,1]
#         num_epochs= 2
# )
# save_path = data_path + "ratio_ddim_celeba64_nep2.pt"
# torch.save(
#     {
#         "model_state": model_ddim.state_dict(),
#         "losses": losses_ddim,
#     },
#     save_path,
# )


# %%
# reload trained model
# this pretrained check point is saved at ./checkpoints/ratio_ddim_celeba64_nep2.pt in this GitHub Repo
save_path = data_path + "ratio_ddim_celeba64_nep2.pt"
# reload
checkpoint = torch.load(save_path)
model_ddim = dre.RatioNetCelebA64(in_ch=3, ndf=64, log_scale=False) 
model_ddim.load_state_dict(checkpoint["model_state"])
losses_ddim = checkpoint["losses"]

# check loss
reload(help)
fig, ax = plt.subplots(figsize=(6,4))
help.plot_losses(losses_ddim, label="ddim", ax=ax, color="tab:blue")
ax.legend()

# %% 
# evaluation
# split = "train"
split = "test"
# 1) Load test split WITH attributes
testset = datasets.CelebA(
    root=data_path,
    split=split,
    target_type="attr",      # <- get the 40 attributes
    transform=transform,     # your [-1,1] image transform
    download=False,
)

# 2) DataLoader (use testset, not trainset)
test_size = 1000
test_celeba_loader = DataLoader(
    testset, batch_size=test_size, 
    shuffle=False, num_workers=2, pin_memory=True
)

# 3) Fetch a batch
x_test, attrs = next(iter(test_celeba_loader))   # x_test: (B,3,64,64), attrs: (B,40) in {-1, +1}

# 4) Convert attributes to {0,1} if you prefer
attrs01 = (attrs == 1).to(torch.uint8)           # (B,40) in {0,1}

# 5) Attribute (covariate) names
attr_names = testset.attr_names    

ddim_data_dir = "/hpc/group/mastatlab/yx306/CelebA/DDIM/data"
q_sampler = celeb.DDIMFakeOnlySampler(ddim_data_dir=ddim_data_dir,split=split)
x_q =  q_sampler(batch_size=test_size)

x_p, x_p_labels = next(iter(test_celeba_loader))
x_p = x_p.to(device=x_q.device, dtype=x_q.dtype, non_blocking=True)


x_mixed = torch.cat([x_p, x_q], dim=0) 
g_p, g_q, g_mixed = dre.evaluate_model(model_ddim, x_p, x_q, x_mixed)


# %%
# check summary stats
stats = {
    "g_p_ddim": help.summarize_vector(g_p.detach().cpu().numpy()),
    "g_mixed_ddim": help.summarize_vector(g_mixed.detach().cpu().numpy()),
    "g_q_ddim": help.summarize_vector(g_q.detach().cpu().numpy()),
}

df = pd.DataFrame(stats)
print(df.round(2))  # show 4 digits



# plot hist
bool_density = False
ax, bin_edges = help.plot_ratio_hist(g_p.detach().cpu().numpy(), 
                                     bins=50, range=(0,2), density=bool_density,
                                     figsize = (5,4),
                                     label="p-observed sample", color="tab:purple")
help.add_ratio_hist(g_q.detach().cpu().numpy(), ax=ax, bin_edges=bin_edges, 
                    density=bool_density,
                    label="q (CelebA64 ddim)", color="tab:green")
ax.legend(); 
steps_per_epoch = len(test_celeba_loader)
last_epoch = losses_ddim[-steps_per_epoch:]
mean_last_epoch = float(np.mean(last_epoch))
ax.set_title(f"Histogram ddim, h^2(p,(p+q)/2) = {-mean_last_epoch:.3f}")
plt.tight_layout(); plt.show()# %%
# %%
# Eval (1): check samples around 0,1,2
reload(help)
reload(celeb)
g_p_flat = g_p.view(-1)
# Count how many elements are below 0.5
nrow = 5
ncol = 8
n_total = nrow*ncol
res = help.select_extremes(g_p_flat, thresh=0.5, k_top=n_total, k_near=n_total, k_small=n_total)


largest_vals, largest_idx   = res["largest"]["values"],  res["largest"]["indices"]
closest_vals, closest_idx         = res["near_one"]["values"], res["near_one"]["indices"]
smallest_vals, smallest_idx = res["smallest"]["values"], res["smallest"]["indices"]

title_fontsize = 20
# plt.rcParams.update({'font.size': title_fontsize})
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120, constrained_layout=True)


title_mid = f"Real:Close-to-1\n ({help.minmax_text_from_idx(g_p_flat, closest_idx)})"
celeb.show_batch(
    x_p, nrows=nrow, ncols=ncol, idx=closest_idx, plot=True, ax=axes[1], show=False,
    title=title_mid,
    title_fontsize = title_fontsize
)
title_left = f"Real:Smallest\n ({help.minmax_text_from_idx(g_p_flat, smallest_idx)})"
celeb.show_batch(
    x_p, nrows=nrow, ncols=ncol,idx=smallest_idx, plot=True, ax=axes[0], show=False,
    title=title_left,
    title_fontsize = title_fontsize
)

title_right = f"Real:Largest\n ({help.minmax_text_from_idx(g_p_flat, largest_idx)})"
celeb.show_batch(
    x_p, nrows=nrow, ncols=ncol,idx=largest_idx, plot=True, ax=axes[2], show=False,
    title=title_right,
    title_fontsize = title_fontsize
)

plt.show()
# %%
# check covariate association
reload(celeb)
gp, A01 = celeb.prepare_gp_attrs(g_p, attrs)

# gp, A01 as prepared; attr_names from testset.attr_names
df_ranked, res = celeb.run_linear(
    gp, A01,
    attr_names=list(testset.attr_names)[:A01.shape[1]],  # ensure same length (40)
    print_summary=False
)
print(df_ranked.head(10))       # top 10 most significant attributes
# If you want the intercept stats:
intercept_row = {
    "coef": res.params[0], "se": res.bse[0], "t": res.tvalues[0], "pvalue": res.pvalues[0],
    "ci_low": res.conf_int()[0,0], "ci_high": res.conf_int()[0,1],
}

# %%
# beta regression when the outcome is in [0,2]
# check covariate association
reload(celeb)
gp, A01 = celeb.prepare_gp_attrs(g_p, attrs)

# gp, A01 as prepared; attr_names from testset.attr_names
df_ranked, res = celeb.run_beta_on_02(
    gp, A01,
    attr_names=list(testset.attr_names)[:A01.shape[1]],  # ensure same length (40)
    print_summary=False
)
print(df_ranked.head(10))       # top 10 most significant attributes
# If you want the intercept stats:
intercept_row = {
    "coef": res.params[0], "se": res.bse[0], "t": res.tvalues[0], "pvalue": res.pvalues[0],
    "ci_low": res.conf_int()[0,0], "ci_high": res.conf_int()[0,1],
}



# %%
# Eval (2): check generated samples around 0,1,2
reload(help)
reload(celeb)
g_q_flat = g_q.view(-1)
# Count how many elements are below 0.5
nrow = 5
ncol = 8
n_total = nrow*ncol
res = help.select_extremes(g_q_flat, thresh=0.5, k_top=n_total, k_near=n_total, k_small=n_total)
largest_vals, largest_idx   = res["largest"]["values"],  res["largest"]["indices"]
closest_vals, closest_idx         = res["near_one"]["values"], res["near_one"]["indices"]
smallest_vals, smallest_idx = res["smallest"]["values"], res["smallest"]["indices"]

title_fontsize = 20
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120, constrained_layout=True)


title_mid = f"Generated:Close-to-1\n ({help.minmax_text_from_idx(g_q_flat, closest_idx)})"
celeb.show_batch(
    x_q, nrows=nrow, ncols=ncol, idx=closest_idx, plot=True, ax=axes[1], show=False,
    title=title_mid,
    title_fontsize = title_fontsize
)
title_left = f"Generated:Smallest\n ({help.minmax_text_from_idx(g_q_flat, smallest_idx)})"
celeb.show_batch(
    x_q, nrows=nrow, ncols=ncol, idx=smallest_idx, plot=True, ax=axes[0], show=False,
    title=title_left,
    title_fontsize = title_fontsize
)

title_right = f"Generated:Largest\n ({help.minmax_text_from_idx(g_q_flat, largest_idx)})"
celeb.show_batch(
    x_q, nrows=nrow, ncols=ncol,idx=largest_idx, plot=True, ax=axes[2], show=False,
    title=title_right,
    title_fontsize = title_fontsize
)

plt.show()
# %%
# (l2-distance) find images closest to the generated image
# reload(celeb)
# x_q_smallest = x_q.index_select(0, smallest_idx.to(x_q.device).long())
# celeba_loader_fixed = DataLoader(trainset, batch_size=batch_size, 
#                         shuffle=False, num_workers=2) # fixed loader

# d_train, i_train = celeb.nearest_pixel_l2_stream(x_q_smallest[:1],
#                          celeba_loader_fixed, device="cuda", k=25)
# print(i_train)
# matched_train = celeb.fetch_by_indices(celeba_loader.dataset, i_train[0], device="cuda")  # (25,C,H,W)


# model_ddim.eval()
# with torch.no_grad():
#     p = next(model_ddim.parameters())
#     device, dtype = p.device, p.dtype
#     x_p_     = x_q_smallest.to(device=device, dtype=dtype)
#     g_p_matched_small     = model_ddim(x_p_).squeeze(-1)

# stats = f"min={g_p_matched_small.min().item():.4g}, max={g_p_matched_small.max().item():.4g}"
# title_mid = f"Matched real sample r(x) range ({stats})"

# celeb.show_batch(
#     matched_train, nrow=5, plot=True, show=False,
#     title=title_mid
# )



# %%
# check barycenter
# bc = celeb.wasserstein_barycenter_2d(matched_train, epsilon=0.02, max_iter=300, tol=1e-6, device="cuda")
# bc_disp = (bc / bc.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]).clamp(0,1)
# bc_minus1_1 = (bc_disp * 2.0) - 1.0
# celeb.show_batch(
#     bc_minus1_1, nrow=1, plot=True, show=False,
#     title="Barycenter of top 25 closest in l2 distance"
# )

# %%
# evaluate r(x) over the entire training set
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
reload(celeb)
reload(dre)

# Real dataset (absolute indices) – unchanged
trainset_idx = celeb.IndexedCelebA(trainset)
celeba_loader = DataLoader(trainset_idx, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

# DDIM sampler restricted to train shards (00..15)
q_sampler = celeb.DDIMFakeOnlySampler(ddim_data_dir=ddim_data_dir, split="train")

device = torch.device("cuda")
model_ddim = model_ddim.to(device).eval()

# Allocate per-dataset outputs
N_p = len(trainset_idx)      # 162,770 (real)
N_q = 160_000                # 16 shards × 10k (fakes)

g_p_all     = torch.empty(N_p, dtype=torch.float32)
attrs01_all = torch.empty((N_p, 40), dtype=torch.uint8)

g_q_all     = torch.empty(N_q, dtype=torch.float32)
q_shard     = torch.empty(N_q, dtype=torch.int64)   # shard id per fake
q_idx_local = torch.empty(N_q, dtype=torch.int64)   # index inside shard

# %%
# evaluate r(x) over the entire training set
# part 2: this takes long!
# ---------------- Pass A: compute g_p on all real images ----------------
with torch.no_grad():
    for (x_p, attrs, idxs_abs) in tqdm(celeba_loader, desc="Real pass (g_p)"):
        b = x_p.size(0)
        x_p = x_p.to(device=device, dtype=torch.float32, non_blocking=True)
        x_q_dummy = q_sampler(batch_size=b, device=device, dtype=torch.float32)  # discard meta

        g_p, _ = dre.evaluate_model(model_ddim, x_p, x_q_dummy)
        idxs_abs = idxs_abs.view(-1).long()

        g_p_all[idxs_abs] = g_p.view(-1).cpu()
        attrs01_all[idxs_abs] = attrs.to(torch.uint8).cpu()

# ---------------- Pass B: compute g_q on 160k DDIM fakes ----------------
produced = 0
bs = 512
n_batches = (N_q + bs - 1) // bs  # ceiling division

with torch.no_grad(), tqdm(total=N_q, desc="DDIM pass (g_q)") as pbar:
    real_iter = iter(celeba_loader)
    p_cache = None

    while produced < N_q:
        need = min(bs, N_q - produced)

        # fakes + metadata
        x_q, meta = q_sampler(batch_size=need, device=device, dtype=torch.float32, return_meta=True)

        # dummy real batch of same size
        try:
            x_p, _, _ = next(real_iter)
            if x_p.shape[0] < need:
                # recycle if last partial batch
                p_cache = x_p.clone()
                real_iter = iter(celeba_loader)
                x_p2, _, _ = next(real_iter)
                x_p = torch.cat([p_cache, x_p2], dim=0)
        except StopIteration:
            real_iter = iter(celeba_loader)
            x_p, _, _ = next(real_iter)

        x_p = x_p[:need].to(device=device, dtype=torch.float32, non_blocking=True)

        # evaluate
        _, g_q = dre.evaluate_model(model_ddim, x_p, x_q)

        # write out
        s = slice(produced, produced + need)
        g_q_all[s]     = g_q.view(-1).cpu()
        q_shard[s]     = meta["shard"].cpu()
        q_idx_local[s] = meta["idx_in_shard"].cpu()

        produced += need
        pbar.update(need)   # progress bar update


# Save: you can always reconstruct the exact image by (shard, idx_in_shard)
torch.save({
    "g_p": g_p_all,               # len 162,770, indexed by absolute CelebA index
    "attrs01": attrs01_all,
}, "/hpc/group/mastatlab/yx306/CelebA/DDIM/g_p_real_train.pt")

torch.save({
    "g_q": g_q_all,               # len 160,000
    "shard": q_shard,             # len 160,000
    "idx_in_shard": q_idx_local,  # len 160,000
}, "/hpc/group/mastatlab/yx306/CelebA/DDIM/g_q_ddim_train_meta.pt")


# %%
# evaluate on the full training data
g_q_ddim_train_meta = torch.load("/hpc/group/mastatlab/yx306/CelebA/DDIM/g_q_ddim_train_meta.pt")
g_q_all = g_q_ddim_train_meta["g_q"]
q_shard = g_q_ddim_train_meta["shard"]

g_p_real_train = torch.load("/hpc/group/mastatlab/yx306/CelebA/DDIM/g_p_real_train.pt")
g_p_all = g_p_real_train["g_p"]
attrs01_all = g_p_real_train["attrs01"]


# check summary stats
stats = {
    "g_p_ddim": help.summarize_vector(g_p_all.detach().cpu().numpy()),
    "g_q_ddim": help.summarize_vector(g_q_all.detach().cpu().numpy()),
}

df = pd.DataFrame(stats)
print(df.round(2))  # show 4 digits







# # Create 2×2 grid
# fig, axs = plt.subplots(2, 2, figsize=(10, 6), gridspec_kw={'height_ratios':[1,1]})
# # Adjust layout
# plt.subplots_adjust(hspace=0.5)

# # --- Row 1: left plot ---
# # plot hist
# bool_density = False
# ax, bin_edges = help.plot_ratio_hist(g_p_all.detach().cpu().numpy(), 
#                                      bins=50, range=(0,2), density=bool_density,
#                                      figsize = (5,4),
#                                      label="p-observed sample", color="tab:purple")
# help.add_ratio_hist(g_q_all.detach().cpu().numpy(), ax=axs[0,0], bin_edges=bin_edges, 
#                     density=bool_density,
#                     label="q (CelebA64 ddim)", color="tab:green")
# ax.legend(); 
# steps_per_epoch = len(test_celeba_loader)
# last_epoch = losses_ddim[-steps_per_epoch:]
# mean_last_epoch = float(np.mean(last_epoch))
# ax.set_title(f"Histogram ddim, h^2(p,(p+q)/2) = {-mean_last_epoch:.3f}")
# plt.tight_layout(); plt.show()# %%

# # --- Row 1: right table ---
# axs[0,1].axis("off")  # hide axis
# table = axs[0,1].table(cellText=df.round(2),
#                        colLabels=["Label", "Value"],
#                        loc="center")
# table.auto_set_font_size(False)
# table.set_fontsize(10)

# # --- Row 2: single figure spanning two columns ---
# from matplotlib.gridspec import GridSpec
# fig.clf()  # clear old subplots
# gs = fig.add_gridspec(2, 2, height_ratios=[1,1])

# ax1 = fig.add_subplot(gs[0,0])
# ax2 = fig.add_subplot(gs[0,1])
# ax3 = fig.add_subplot(gs[1,:])  # span both columns

# # Row 1 left: figure
# ax1.plot(x, y)
# ax1.set_title("Figure 1")

# # Row 1 right: table
# ax2.axis("off")
# table = ax2.table(cellText=table_data,
#                   colLabels=["Label", "Value"],
#                   loc="center")
# table.auto_set_font_size(False)
# table.set_fontsize(10)

# # Row 2: big figure spanning both columns
# ax3.plot(x, np.cos(x))
# ax3.set_title("Figure 2 (spans both columns)")

# plt.tight_layout()
# plt.show()

# %%
# try regression on entire training
# batch_size = 512
# celeba_loader = DataLoader(trainset_idx, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# N = len(trainset_idx)
# attrs01_all = torch.empty((N, 40), dtype=torch.uint8)   # 40 CelebA attributes

# with torch.no_grad():
#     for (_, attrs, idxs_abs) in celeba_loader:
#         # If your dataset gives -1/1 attributes, convert to 0/1:
#         if attrs.min().item() < 0:
#             attrs = (attrs > 0).to(torch.uint8)
#         else:
#             attrs = attrs.to(torch.uint8)
#         attrs01_all[idxs_abs.long()] = attrs
# mask = torch.isfinite(g_p_all)
# num_true  = mask.sum().item()
# num_false = (~mask).sum().item()
# print({"True": num_true, "False": num_false})
# gp_vals   = g_p_all[mask]
# attrs_gp  = attrs01_all[mask]       # same order as gp_vals
# idxs_gp   = torch.nonzero(mask).view(-1)   # absolute CelebA indices used

# torch.save({"g_p": gp_vals, "attrs01": attrs_gp, "idxs": idxs_gp},
#            "/hpc/group/mastatlab/yx306/CelebA/DDIM/gp_with_attrs.pt")


# data = torch.load("/hpc/group/mastatlab/yx306/CelebA/DDIM/gp_with_attrs.pt")
# # Unpack
# g_p_all   = data["g_p"]       # 1D tensor of g_p values
# attrs  = data["attrs01"]   # 2D tensor (N, 40) of attributes



# gp, A01 = celeb.prepare_gp_attrs(g_p_all, attrs)

gp, A01 = celeb.prepare_gp_attrs(g_p_all, attrs01_all)

# Linear regression
df_ranked, res = celeb.run_linear(
    gp, A01,
    attr_names=list(testset.attr_names)[:A01.shape[1]],  # ensure same length (40)
    print_summary=False
)
print(df_ranked.head(10))      
intercept_row = {
    "coef": res.params[0], "se": res.bse[0], "t": res.tvalues[0], "pvalue": res.pvalues[0],
    "ci_low": res.conf_int()[0,0], "ci_high": res.conf_int()[0,1],
}

# Beta regression
df_ranked, res = celeb.run_beta_on_02(
    gp, A01,
    attr_names=list(testset.attr_names)[:A01.shape[1]],  # ensure same length (40)
    print_summary=False
)
print(df_ranked.head(10))       # top 10 most significant attributes
# If you want the intercept stats:
intercept_row = {
    "coef": res.params[0], "se": res.bse[0], "t": res.tvalues[0], "pvalue": res.pvalues[0],
    "ci_low": res.conf_int()[0,0], "ci_high": res.conf_int()[0,1],
}

# Logistic regression P(g>1)
df_ranked, res = celeb.run_logistic(
    gp, A01,
    attr_names=list(testset.attr_names)[:A01.shape[1]],  # ensure same length (40)
    print_summary=False
)
print(df_ranked.head(10))      
intercept_row = {
    "coef": res.params[0], "se": res.bse[0], "t": res.tvalues[0], "pvalue": res.pvalues[0],
    "ci_low": res.conf_int()[0,0], "ci_high": res.conf_int()[0,1],
}



# %%
# check extreme images in both the real and generated entire training set

# For real
reload(celeb)
def fetch_real(indices: torch.Tensor):
    imgs = [trainset_idx[i][0].cpu() for i in indices.tolist()]
    return torch.stack(imgs, dim=0)

nrow = 8
ncol = 5
k_each = nrow*ncol
plt.rcParams.update({'font.size': 14})
celeb.plot_generated_extremes_by_targets(
    scores=g_p_all,
    fetch_fn=fetch_real,
    targets=(0.0, 1.0, 2.0),
    k_each=k_each,
    nrow=nrow,
    thresh = 0.5,
    title_prefix="Real"
)


# Assume you reloaded these from disk alongside g_q_all
# q_shard: LongTensor [160000], q_idx_local: LongTensor [160000]
fetch_fake = celeb.make_fake_fetcher(
    ddim_data_dir=ddim_data_dir,
    q_shard=q_shard,
    q_idx_local=q_idx_local,
    fake_key=q_sampler.fake_key,         # or None if not used
    assume_chw=q_sampler.assume_chw,     # mirror your sampler
)

celeb.plot_generated_extremes_by_targets(
    scores=g_q_all,
    fetch_fn=fetch_fake,
    targets=(0.0, 1.0, 2.0),
    k_each=k_each,
    nrow=nrow,
    thresh = 0.5,
    title_prefix="DDIM"
)

celeb.plot_generated_extremes_by_targets(
    scores=g_q_all,
    fetch_fn=fetch_fake,
    targets=(0.00012,0.486),
    k_each=k_each,
    nrow=nrow,
    thresh = 0.5,
    title_prefix="DDIM"
)

# %%
# check the closest pairs around 0,1,2
reload(celeb)
# res = celeb.find_pairs_for_groups(g_p_all, g_q_all, targets=(0,1,2), k_per_target=1, proximity_weight=1.0)

# # Example: indices for the pair nearest to 0 in g_p
# pairs_near0_gp = res["g_p"][0]["pairs"]      # list of (i,j)
# vals_near0_gp  = res["g_p"][0]["values"]     # list of (x_i, x_j)
# scores_near0_gp= res["g_p"][0]["scores"]     # combined scores

sel = celeb.visualize_cross_pairs_oneplot(
    g_p_all, g_q_all,
    fetch_real, fetch_fake,
    targets=(0,1,2),
    k_per_target=5,
    proximity_weight=1.0,
    figsize=(14, 10)
)

# Example: inspect chosen indices for target 1.0
# [(i_p, i_q), ...]
pairs_near1 = [d["pair"] for d in sel[1.0]]
# %%
# for a single fake image, try to find the closest training image (in l2), and check Barycenter
# --- 1) pick the fake with the smallest g ---
idx_q_min = int(torch.argmin(g_q_all).item())
print(f"Fake with smallest g at index {idx_q_min}, g={float(g_q_all[idx_q_min]):.6f}")

# --- 2) obtain that fake image (works with or without x_q in memory) ---
reload(celeb)
# find the closest training samples to generated images
def get_fake_by_index(i, *, prefer_ram=True):
    use_ram = (
        prefer_ram
        and ('x_q' in globals())
        and isinstance(x_q, torch.Tensor)
        and x_q.ndim == 4
        and x_q.size(0) > i            # <- ensure slice won’t be empty
    )
    if use_ram:
        out = x_q[i:i+1]
    else:
        assert 'fetch_fake' in globals(), "fetch_fake is not defined"
        out = fetch_fake(torch.tensor([i], dtype=torch.long))

    # Normalize to (1,C,H,W)
    if out.ndim == 3:  # (C,H,W) or (H,W,C)
        out = out.unsqueeze(0)
    if out.size(0) == 0:
        raise RuntimeError(f"Fetched empty fake for i={i}; check x_q length and fetcher metadata.")
    return out

x_q_min = get_fake_by_index(idx_q_min)  # shape (1,C,H,W)

# --- 3) stream NN search over the ENTIRE training set ---
reload(celeb)


celeba_loader_fixed = DataLoader(trainset, batch_size=batch_size, 
                        shuffle=False, num_workers=2) # fixed loader

d_train, i_train = celeb.nearest_pixel_l2_stream(x_q_min[:1],
                         celeba_loader_fixed, device="cuda", k=100)
print(i_train)
print("i_train.shape",i_train.shape)

# %%
print(i_train)
matched_train = celeb.fetch_by_indices(celeba_loader_fixed.dataset, i_train[0], device="cuda")  # (25,C,H,W)


model_ddim.eval()
with torch.no_grad():
    p = next(model_ddim.parameters())
    device, dtype = p.device, p.dtype
    x_p_     = matched_train.to(device=device, dtype=dtype)
    g_p_matched_small     = model_ddim(x_p_).squeeze(-1)

stats = f"min={g_p_matched_small.min().item():.4g}, max={g_p_matched_small.max().item():.4g}"
title_mid = f"Matched real sample r(x) range ({stats})"

celeb.show_batch(
    matched_train, nrow=10, plot=True, show=False,
    title=title_mid
)

celeb.show_batch(
    x_q_min, nrow=1, plot=True, show=False
)



# %%
# check barycenter
bc = celeb.wasserstein_barycenter_2d(matched_train, epsilon=1e-6, 
                                    max_iter=300, tol=1e-6, device="cuda")
bc_disp = (bc / bc.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]).clamp(0,1)
bc_minus1_1 = (bc_disp * 2.0) - 1.0
celeb.show_batch(
    bc_minus1_1, nrow=1, plot=True, show=False,
    title="Barycenter of top 25 closest in l2 distance"
)
# %%
# try to find the nearest wasserstein neighbors
reload(celeb)
# 1) pick your query fake (e.g., the smallest g fake)
idx_q_min = int(torch.argmin(g_q_all).item())
x_q_min = fetch_fake(torch.tensor([idx_q_min], dtype=torch.long))   # (1,C,H,W)

# 2) run streaming nearest under Wasserstein proxy
# best_d_w1, best_i_w1 = celeb.nearest_wasserstein_stream(
#     gen_imgs=x_q_min,
#     real_loader=celeba_loader_fixed,   # your sequential, shuffle=False loader
#     device="cuda",
#     k=100,
#     use_amp=False,
#     treat_second_as_labels=True,       # synthesize absolute indices if loader doesn't give them
#     progress=True,
#     pbar_desc="Nearest W1",
# )

# # 3) fetch the matched reals
# nn_idx = torch.as_tensor(best_i_w1[0]).long()
# x_p_w1 = celeb.fetch_by_indices(celeba_loader_fixed.dataset, nn_idx, device="cuda")


# 2) run streaming nearest under sliced W₂²
best_d_w2, best_i_w2 = celeb.nearest_w2_stream(
    gen_imgs=x_q_min,
    real_loader=celeba_loader_fixed,   # your fixed, shuffle=False loader over trainset
    device="cuda",
    k=100,
    n_projs=64,                        # try 64..256 for better accuracy
    map_mode="auto",                   # auto-handle [-1,1] vs [0,1]
    use_amp=False,
    treat_second_as_labels=True,
    progress=True,
    pbar_desc="Nearest W2"
)

nn_idx = torch.as_tensor(best_i_w2[0]).long()
x_p_w = celeb.fetch_by_indices(celeba_loader_fixed.dataset, nn_idx, device="cuda")  # (25,C,H,W)


# %%
model_ddim.eval()
with torch.no_grad():
    p = next(model_ddim.parameters())
    device, dtype = p.device, p.dtype
    x_p_     = x_p_w.to(device=device, dtype=dtype)
    g_p_matched_small     = model_ddim(x_p_).squeeze(-1)

stats = f"min={g_p_matched_small.min().item():.4g}, max={g_p_matched_small.max().item():.4g}"
title_mid = f"Matched real sample r(x) range ({stats})"

celeb.show_batch(
    x_p_w, nrow=10, plot=True, show=False,
    title=title_mid
)

# check barycenter
bc = celeb.wasserstein_barycenter_2d(x_p_w, epsilon=1e-6, 
                                    max_iter=300, tol=1e-6, device="cuda")
bc_disp = (bc / bc.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]).clamp(0,1)
bc_minus1_1 = (bc_disp * 2.0) - 1.0
celeb.show_batch(
    bc_minus1_1, nrow=1, plot=True, show=False,
    title="Barycenter of closest x_p in W1 distance"
)

# %%
# how many images are there that are completely outside the support of p?
reload(celeb)
res = celeb.all_q_indices_below_min_p(g_p_all, g_q_all, q_shard=q_shard, q_idx_local=q_idx_local)

idx_q_below = res["idx"]          
idx_q_below = torch.as_tensor(res["idx"]).long().view(-1)
x_q_below = fetch_fake(idx_q_below)        # (M, C, H, W) or (M, H, W, C) per your fetcher

# %%
idx_q_below.shape

n_row_display = 30

celeb.show_batch(
    x_q_below[:n_row_display*n_row_display], nrow=n_row_display, plot=True, show=False,
    title="Images below the smallest r_p(x)"
)
# %%
