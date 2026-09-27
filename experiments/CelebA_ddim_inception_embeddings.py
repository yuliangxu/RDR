# %%
# CelebA DDIM RDR trained on Inception embedding space.
#
# Open this file in VS Code and use "Run Cell" on each # %% block.
# This mirrors CelebA_ddim.py, except the RDR input is an Inception-v3
# embedding vector instead of the original 64x64 image.

import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rdr_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import Inception_V3_Weights, inception_v3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

import experiments.CelebA.helpers as celeb
import utils.DRE_func as dre
import utils.diagnostics as help
# %%
# Configuration.
#
# For the real FID-style run, keep INCEPTION_WEIGHTS="imagenet" and
# FEATURE_LAYER="pool3". Use INCEPTION_WEIGHTS="none" only for quick plumbing
# checks, because those features are random and scientifically meaningless.

SEED = 42
DATA_PATH = Path("/hpc/group/mastatlab/yx306/CelebA/")
DDIM_DATA_DIR = DATA_PATH / "DDIM/data"
CACHE_DIR = DATA_PATH / "DDIM/inception_embeddings"

FEATURE_LAYER = "pool3"          # "pool3" for FID features, "logits" for IS logits
INCEPTION_WEIGHTS = "imagenet"   # "imagenet" or "none"
RECOMPUTE_EMBEDDINGS = False

EMBED_BATCH_SIZE = 256
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096
NUM_WORKERS = 0

NUM_EPOCHS = 20
STEPS_PER_EPOCH = 400
VAL_STEPS = None                  # None means use the full validation loaders
MLP_HIDDEN = 512
LOSS_METHOD = "Hellinger"        # "Hellinger", "KL", or "Chisq"

# Optional small debug run. Leave as None for the full experiment.
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None
MAX_FAKE_TRAIN_SAMPLES = None     # None uses min(len(trainset), 160000)
MAX_FAKE_VAL_SAMPLES = None       # None uses len(valid set)


args = SimpleNamespace(
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    val_steps=VAL_STEPS,
    mlp_hidden=MLP_HIDDEN,
    loss_method=LOSS_METHOD,
)


# %%
# Helpers.

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed set to {seed}]")


class InceptionEmbeddingNet(nn.Module):
    """Inception-v3 pool3/logit feature extractor for CelebA tensors in [-1, 1]."""

    def __init__(self, feature_layer="pool3", weights_name="imagenet"):
        super().__init__()
        if feature_layer not in {"pool3", "logits"}:
            raise ValueError("feature_layer must be 'pool3' or 'logits'")
        if weights_name not in {"imagenet", "none"}:
            raise ValueError("weights_name must be 'imagenet' or 'none'")

        weights = Inception_V3_Weights.IMAGENET1K_V1 if weights_name == "imagenet" else None
        model = inception_v3(weights=weights, aux_logits=True, init_weights=False)
        model.eval()

        self.feature_layer = feature_layer
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if feature_layer == "pool3":
            model.fc = nn.Identity()
        self.model = model

    def _preprocess(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.size(1) != 3:
            raise ValueError(f"Expected RGB tensors, got shape {tuple(x.shape)}")

        # CelebA_ddim.py uses [-1,1]. Convert to [0,1] before ImageNet normalization.
        if x.min() < -1e-4:
            x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    @torch.no_grad()
    def forward(self, x):
        x = self._preprocess(x)
        y = self.model(x)
        if isinstance(y, tuple):
            y = y[0]
        return y.flatten(1)


def load_or_compute(path, compute_fn, recompute=False):
    path = Path(path)
    if path.exists() and not recompute:
        print(f"Loading {path}")
        return torch.load(path, map_location="cpu")

    print(f"Computing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = compute_fn()
    torch.save(obj, path)
    return obj


def embed_loader(feature_net, loader, device):
    embeddings, attrs = [], []
    feature_net.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            embeddings.append(feature_net(x).cpu())
            if len(batch) > 1 and torch.is_tensor(batch[1]):
                attrs.append(batch[1].cpu())
    z_all = torch.cat(embeddings, dim=0)
    attrs_all = torch.cat(attrs, dim=0) if attrs else None
    return {"z": z_all, "attrs": attrs_all}


def embed_fake_sampler(feature_net, sampler, n, batch_size, device):
    embeddings, shard_ids, idx_in_shard = [], [], []
    feature_net.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            size = min(batch_size, n - start)
            x, meta = sampler(batch_size=size, device=device, dtype=torch.float32, return_meta=True)
            embeddings.append(feature_net(x).cpu())
            shard_ids.append(meta["shard"].cpu())
            idx_in_shard.append(meta["idx_in_shard"].cpu())
    return {
        "z": torch.cat(embeddings, dim=0),
        "shard": torch.cat(shard_ids, dim=0),
        "idx_in_shard": torch.cat(idx_in_shard, dim=0),
    }


def make_loader(z, batch_size, shuffle, seed, drop_last=False):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(z.float()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=drop_last,
        num_workers=0,
    )


def make_mixed_embeddings(real_z, fake_z, seed=SEED, shuffle=True):
    n = min(real_z.size(0), fake_z.size(0))
    mixed = torch.cat([real_z[:n], fake_z[:n]], dim=0)
    if not shuffle:
        return mixed
    generator = torch.Generator()
    generator.manual_seed(seed)
    return mixed[torch.randperm(mixed.size(0), generator=generator)]


def predict_embeddings(model, loader):
    model.eval()
    scores = []
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for batch in loader:
            z = batch[0] if isinstance(batch, (tuple, list)) else batch
            if z.ndim > 2:
                z = z.view(z.size(0), -1)
            scores.append(model(z.to(device=device, dtype=dtype)).squeeze(-1).cpu())
    return torch.cat(scores, dim=0)


def train_embedding_rdr(name, p_train_z, q_train_z, p_val_z, q_val_z, fake_val_z, args, device):
    p_loader = make_loader(p_train_z, args.batch_size, shuffle=True, seed=SEED, drop_last=True)
    q_loader = make_loader(q_train_z, args.batch_size, shuffle=True, seed=SEED + 1, drop_last=True)
    p_val_loader = make_loader(p_val_z, args.batch_size, shuffle=False, seed=SEED + 2)
    q_val_loader = make_loader(q_val_z, args.batch_size, shuffle=False, seed=SEED + 3)

    model, losses, val_losses = dre.run_DRE_fdiv_embed(
        p_loader,
        q_loader,
        num_epochs=args.num_epochs,
        steps_per_epoch=min(len(p_loader), args.steps_per_epoch),
        p_val_loader=p_val_loader,
        q_val_loader=q_val_loader,
        loss_method=args.loss_method,
        device=device,
        mlp_hidden=args.mlp_hidden,
        val_steps=args.val_steps,
        return_val_losses=True,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, label="training", color="tab:blue")
    best_val_loss = None
    best_epoch = None
    if val_losses:
        ax.plot(np.arange(1, len(val_losses) + 1), val_losses, "o-", label="validation", color="tab:orange")
        best_idx = int(np.argmin(val_losses))
        best_epoch = best_idx + 1
        best_val_loss = float(val_losses[best_idx])
        ax.axvline(best_epoch, linestyle="--", alpha=0.35)
        ax.scatter([best_epoch], [best_val_loss], s=60, zorder=5, label=f"best @ epoch {best_epoch}")
        print(f"{name}: best validation loss = {best_val_loss:.6f} at epoch {best_epoch}")
    ax.set(title=f"CelebA {name} RDR on Inception embeddings", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    n_eval = min(p_val_z.size(0), fake_val_z.size(0))
    mixed_eval_z = torch.cat([p_val_z[:n_eval], fake_val_z[:n_eval]], dim=0)
    p_eval_loader = make_loader(p_val_z, args.eval_batch_size, False, SEED)
    q_eval_loader = make_loader(fake_val_z, args.eval_batch_size, False, SEED)
    mixed_eval_loader = make_loader(mixed_eval_z, args.eval_batch_size, False, SEED)

    g_p = predict_embeddings(model, p_eval_loader)
    g_q = predict_embeddings(model, q_eval_loader)
    g_mixed = predict_embeddings(model, mixed_eval_loader)

    stats = {
        f"g_p_{name}": help.summarize_vector(g_p.detach().cpu().numpy()),
        f"g_mixed_{name}": help.summarize_vector(g_mixed.detach().cpu().numpy()),
        f"g_q_{name}": help.summarize_vector(g_q.detach().cpu().numpy()),
    }
    print(pd.DataFrame(stats).round(2))

    fig, ax = plt.subplots(figsize=(5, 4))
    ax, bin_edges = help.plot_ratio_hist(
        g_p.detach().cpu().numpy(),
        bins=50,
        range=(0, 2),
        density=True,
        label="p-observed validation",
        color="tab:purple",
        ax=ax,
    )
    help.add_ratio_hist(
        g_q.detach().cpu().numpy(),
        ax=ax,
        bin_edges=bin_edges,
        density=True,
        label=f"q ({name})",
        color="tab:green",
    )
    ax.legend()
    ax.set_title(f"Histogram {name.upper()} in Inception embedding space")
    plt.tight_layout()

    return model, losses, val_losses, best_val_loss, best_epoch, g_p, g_q, g_mixed


def collect_loader_images(loader):
    images = []
    for batch in loader:
        x = batch[0].detach().cpu()
        images.append(x)
    return torch.cat(images, dim=0)


def show_rdr_extremes(scores, images, title_prefix, nrow=5, ncol=8):
    scores = scores.view(-1).detach().cpu()
    n_total = nrow * ncol
    res = help.select_extremes(scores, thresh=0.5, k_top=n_total, k_near=n_total, k_small=n_total)

    largest_idx = res["largest"]["indices"]
    closest_idx = res["near_one"]["indices"]
    smallest_idx = res["smallest"]["indices"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120, constrained_layout=True)
    panels = [
        ("Smallest", smallest_idx, axes[0]),
        ("Close-to-1", closest_idx, axes[1]),
        ("Largest", largest_idx, axes[2]),
    ]
    for name, idx, ax in panels:
        title = f"{title_prefix}:{name}\n({help.minmax_text_from_idx(scores, idx)})"
        celeb.show_batch(
            images,
            nrows=nrow,
            ncols=ncol,
            idx=idx,
            plot=True,
            ax=ax,
            show=False,
            title=title,
            title_fontsize=18,
        )
    plt.show()
    return res


def fetch_fake_images(ddim_data_dir, shard, idx_in_shard, indices):
    fetch = celeb.make_fake_fetcher(
        str(ddim_data_dir),
        shard,
        idx_in_shard,
        fake_key=None,
        assume_chw=True,
    )
    return fetch(indices)


def show_fake_rdr_extremes(scores, fake_meta, title_prefix, nrow=5, ncol=8):
    scores = scores.view(-1).detach().cpu()
    n_total = nrow * ncol
    res = help.select_extremes(scores, thresh=0.5, k_top=n_total, k_near=n_total, k_small=n_total)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=120, constrained_layout=True)
    panels = [
        ("Smallest", res["smallest"]["indices"], axes[0]),
        ("Close-to-1", res["near_one"]["indices"], axes[1]),
        ("Largest", res["largest"]["indices"], axes[2]),
    ]
    for name, idx, ax in panels:
        imgs = fetch_fake_images(DDIM_DATA_DIR, fake_meta["shard"], fake_meta["idx_in_shard"], idx)
        title = f"{title_prefix}:{name}\n({help.minmax_text_from_idx(scores, idx)})"
        celeb.show_batch(
            imgs,
            nrows=nrow,
            ncols=ncol,
            plot=True,
            ax=ax,
            show=False,
            title=title,
            title_fontsize=18,
        )
    plt.show()
    return res


# %%
# Initialize device and reproducibility.

set_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
tag = f"{FEATURE_LAYER}_{INCEPTION_WEIGHTS}"
cache_tag = tag
debug_parts = []
if MAX_TRAIN_SAMPLES is not None:
    debug_parts.append(f"train{MAX_TRAIN_SAMPLES}")
if MAX_VAL_SAMPLES is not None:
    debug_parts.append(f"valid{MAX_VAL_SAMPLES}")
if MAX_FAKE_TRAIN_SAMPLES is not None:
    debug_parts.append(f"faketrain{MAX_FAKE_TRAIN_SAMPLES}")
if MAX_FAKE_VAL_SAMPLES is not None:
    debug_parts.append(f"fakevalid{MAX_FAKE_VAL_SAMPLES}")
if debug_parts:
    cache_tag = f"{tag}_debug_{'_'.join(debug_parts)}"
print(f"Using device: {device}")
print(f"Embedding cache tag: {cache_tag}")


# %%
# Load CelebA train and validation splits.

transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize((64, 64), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

trainset = datasets.CelebA(
    root=DATA_PATH,
    split="train",
    target_type="attr",
    transform=transform,
    download=False,
)
valset = datasets.CelebA(
    root=DATA_PATH,
    split="valid",
    target_type="attr",
    transform=transform,
    download=False,
)

if MAX_TRAIN_SAMPLES is not None:
    trainset = torch.utils.data.Subset(trainset, range(min(MAX_TRAIN_SAMPLES, len(trainset))))
if MAX_VAL_SAMPLES is not None:
    valset = torch.utils.data.Subset(valset, range(min(MAX_VAL_SAMPLES, len(valset))))

train_loader = DataLoader(
    trainset,
    batch_size=EMBED_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
val_loader = DataLoader(
    valset,
    batch_size=EMBED_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

print(f"CelebA train: {len(trainset):,}")
print(f"CelebA validation: {len(valset):,}")


# %%
# Build the Inception feature extractor.

feature_net = InceptionEmbeddingNet(
    feature_layer=FEATURE_LAYER,
    weights_name=INCEPTION_WEIGHTS,
).to(device)
feature_net.eval()


# %%
# Compute or load real CelebA embeddings.

real_train = load_or_compute(
    CACHE_DIR / f"celeba_train_real_{cache_tag}.pt",
    lambda: embed_loader(feature_net, train_loader, device),
    recompute=RECOMPUTE_EMBEDDINGS,
)
real_val = load_or_compute(
    CACHE_DIR / f"celeba_valid_real_{cache_tag}.pt",
    lambda: embed_loader(feature_net, val_loader, device),
    recompute=RECOMPUTE_EMBEDDINGS,
)

p_train_z = real_train["z"]
p_val_z = real_val["z"]
val_attrs = real_val["attrs"]
attr_names = list(getattr(valset, "attr_names", getattr(getattr(valset, "dataset", None), "attr_names", [])))

print("p_train_z:", tuple(p_train_z.shape))
print("p_val_z:", tuple(p_val_z.shape))


# %%
# Compute or load DDIM fake embeddings.
#
# Train split uses shards 00000..00015. Test split uses shards 00016..00019,
# following CelebA_ddim.py.

n_fake_train = 160_000 if MAX_FAKE_TRAIN_SAMPLES is None else MAX_FAKE_TRAIN_SAMPLES
n_fake_train = min(n_fake_train, len(trainset))
n_fake_val = len(valset) if MAX_FAKE_VAL_SAMPLES is None else min(MAX_FAKE_VAL_SAMPLES, len(valset))

train_fake_rng = torch.Generator()
train_fake_rng.manual_seed(SEED + 1000)
val_fake_rng = torch.Generator()
val_fake_rng.manual_seed(SEED + 2000)

train_fake_sampler = celeb.DDIMFakeOnlySampler(
    ddim_data_dir=str(DDIM_DATA_DIR),
    split="train",
    device=device,
    rng=train_fake_rng,
)
val_fake_sampler = celeb.DDIMFakeOnlySampler(
    ddim_data_dir=str(DDIM_DATA_DIR),
    split="test",
    device=device,
    rng=val_fake_rng,
)

fake_train = load_or_compute(
    CACHE_DIR / f"celeba_train_ddim_fake_{cache_tag}.pt",
    lambda: embed_fake_sampler(feature_net, train_fake_sampler, n_fake_train, EMBED_BATCH_SIZE, device),
    recompute=RECOMPUTE_EMBEDDINGS,
)
fake_val = load_or_compute(
    CACHE_DIR / f"celeba_valid_ddim_fake_{cache_tag}.pt",
    lambda: embed_fake_sampler(feature_net, val_fake_sampler, n_fake_val, EMBED_BATCH_SIZE, device),
    recompute=RECOMPUTE_EMBEDDINGS,
)

fake_train_z = fake_train["z"]
fake_val_z = fake_val["z"]

print("fake_train_z:", tuple(fake_train_z.shape))
print("fake_val_z:", tuple(fake_val_z.shape))


# %%
# Train RDR on Inception embeddings: p=real train, q=50/50 real+DDIM.

q_train_z = make_mixed_embeddings(p_train_z, fake_train_z, seed=SEED + 10)
q_val_z = make_mixed_embeddings(p_val_z, fake_val_z, seed=SEED + 11)

model_ddim_embed, losses_ddim_embed, val_losses_ddim_embed, best_val_loss_ddim_embed, best_epoch_ddim_embed, g_p_ddim_embed, g_q_ddim_embed, g_mixed_ddim_embed = (
    train_embedding_rdr(
        "ddim",
        p_train_z,
        q_train_z,
        p_val_z,
        q_val_z,
        fake_val_z,
        args,
        device,
    )
)


# %%
# Save the embedding-space RDR checkpoint.

save_path = DATA_PATH / f"ratio_ddim_celeba_inception_{cache_tag}.pt"
torch.save(
    {
        "model_state": model_ddim_embed.state_dict(),
        "losses": losses_ddim_embed,
        "val_losses": val_losses_ddim_embed,
        "best_val_loss": best_val_loss_ddim_embed,
        "best_epoch": best_epoch_ddim_embed,
        "config": {
            "feature_layer": FEATURE_LAYER,
            "inception_weights": INCEPTION_WEIGHTS,
            "mlp_hidden": MLP_HIDDEN,
            "loss_method": LOSS_METHOD,
        },
    },
    save_path,
)
print(f"Saved {save_path}")
print(f"Best validation loss: {best_val_loss_ddim_embed:.6f} at epoch {best_epoch_ddim_embed}")


# %%
# Attribute association for real validation scores.

gp, A01 = celeb.prepare_gp_attrs(g_p_ddim_embed, val_attrs)
df_ranked, linear_res = celeb.run_linear(
    gp,
    A01,
    attr_names=attr_names[:A01.shape[1]],
    print_summary=False,
)
df_ranked.head(10)


# %%
# Beta-style regression for scores in [0,2].

gp, A01 = celeb.prepare_gp_attrs(g_p_ddim_embed, val_attrs)
df_beta_ranked, beta_res = celeb.run_beta_on_02(
    gp,
    A01,
    attr_names=attr_names[:A01.shape[1]],
    print_summary=False,
)
df_beta_ranked.head(10)


# %%
# Visualize real validation images with RDR near 0, 1, and 2.

x_p_val = collect_loader_images(val_loader)
real_extremes = show_rdr_extremes(
    g_p_ddim_embed,
    x_p_val,
    title_prefix="Real validation",
    nrow=5,
    ncol=8,
)


# %%
# Visualize DDIM fake validation images with RDR near 0, 1, and 2.

fake_extremes = show_fake_rdr_extremes(
    g_q_ddim_embed,
    fake_val,
    title_prefix="DDIM fake validation",
    nrow=5,
    ncol=8,
)


# %%
# Side-by-side summary.

summary = pd.DataFrame(
    {
        "real_valid": help.summarize_vector(g_p_ddim_embed.detach().cpu().numpy()),
        "mixed_valid": help.summarize_vector(g_mixed_ddim_embed.detach().cpu().numpy()),
        "ddim_valid": help.summarize_vector(g_q_ddim_embed.detach().cpu().numpy()),
    }
)
print(f"Best validation loss: {best_val_loss_ddim_embed:.6f} at epoch {best_epoch_ddim_embed}")
summary.round(3)
