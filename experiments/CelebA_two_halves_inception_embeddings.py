# %% [markdown]
# # RDR between two random CelebA halves in Inception embedding space
#
# This null experiment mirrors `CelebA_two_halves.py`, but the density-ratio
# model is trained on Inception-v3 embedding vectors instead of real image
# tensors.
#
# The official CelebA training observations are split into two disjoint random
# groups, p and q. The RDR is
#
#     RDR(z) = 2 p(z) / (p(z) + q(z)) = p(z) / m(z),
#
# where z is the embedding vector and m=(p+q)/2. Therefore, the density-ratio
# trainer receives samples from p as its numerator and samples from the 50/50
# midpoint mixture m as its denominator.
#
# Open this file in VS Code and run cells individually with **Run Cell**.

# %% Imports and configuration
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
from IPython.display import display
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.models import Inception_V3_Weights, inception_v3


REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import experiments.CelebA.helpers as celeb
import utils.DRE_func as dre


SEED = 42
DATA_PATH = Path(os.environ.get("CELEBA_DATA_ROOT", "/hpc/group/mastatlab/yx306/CelebA/"))
CACHE_DIR = DATA_PATH / "DDIM/inception_embeddings"
VALIDATION_SPLIT = "valid"        # "valid" reuses the DDIM embedding script cache; "test" is also allowed

FEATURE_LAYER = "pool3"           # "pool3" for FID features, "logits" for IS logits
INCEPTION_WEIGHTS = "imagenet"    # "imagenet" or "none"
RECOMPUTE_EMBEDDINGS = False

EMBED_BATCH_SIZE = 256
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 4096
NUM_WORKERS = 0

NUM_EPOCHS = 20
STEPS_PER_EPOCH = 400
VAL_STEPS = None                   # None means use the full validation loaders
MLP_HIDDEN = 512
LOSS_METHOD = "Hellinger"         # "Hellinger", "KL", or "Chisq"

# Optional small debug run. Leave as None for the full experiment.
MAX_TRAIN_SAMPLES = None
MAX_VAL_SAMPLES = None


args = SimpleNamespace(
    batch_size=BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    val_steps=VAL_STEPS,
    mlp_hidden=MLP_HIDDEN,
    loss_method=LOSS_METHOD,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# %% Reproducibility and helpers
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed set to {seed}]")


def random_halves(n, seed):
    """Return a reproducible random partition; p gets floor(n/2)."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    cut = n // 2
    return order[:cut], order[cut:]


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

        # CelebA loaders here normalize to [-1, 1]. Convert to [0, 1] first.
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


def embed_loader(feature_net, loader, device):
    embeddings, attrs = [], []
    feature_net.eval()
    with torch.no_grad():
        for images, batch_attrs in loader:
            embeddings.append(feature_net(images.to(device)).cpu())
            attrs.append(batch_attrs.cpu())
    return {"z": torch.cat(embeddings, dim=0), "attrs": torch.cat(attrs, dim=0)}


def load_or_compute(path, compute_fn, expected_n, recompute=False):
    path = Path(path)
    if path.exists() and not recompute:
        print(f"Loading {path}")
        obj = torch.load(path, map_location="cpu")
        if obj["z"].size(0) == expected_n:
            return obj
        print(f"Cached rows ({obj['z'].size(0):,}) do not match expected rows ({expected_n:,}); recomputing.")

    print(f"Computing {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = compute_fn()
    torch.save(obj, path)
    return obj


def make_embedding_loader(z, batch_size, shuffle, seed, drop_last=False):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(z.float()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        drop_last=drop_last,
        num_workers=0,
    )


def make_midpoint_embeddings(p_z, q_z, seed, shuffle=True):
    n = min(p_z.size(0), q_z.size(0))
    midpoint_z = torch.cat([p_z[:n], q_z[:n]], dim=0)
    if not shuffle:
        return midpoint_z
    generator = torch.Generator().manual_seed(seed)
    return midpoint_z[torch.randperm(midpoint_z.size(0), generator=generator)]


@torch.no_grad()
def predict_embeddings(model, z, batch_size):
    loader = make_embedding_loader(z, batch_size, shuffle=False, seed=SEED)
    model.eval()
    scores = []
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    for batch in loader:
        batch_z = batch[0]
        values = model(batch_z.to(device=model_device, dtype=model_dtype)).reshape(-1)
        scores.append(values.cpu())
    return torch.cat(scores, dim=0)


def summarize(values):
    values = values.float()
    return {
        "n": values.numel(),
        "mean": values.mean().item(),
        "std": values.std().item(),
        "min": values.min().item(),
        "q05": torch.quantile(values, 0.05).item(),
        "median": values.median().item(),
        "q95": torch.quantile(values, 0.95).item(),
        "max": values.max().item(),
    }


set_seed(SEED)
base_cache_tag = f"{FEATURE_LAYER}_{INCEPTION_WEIGHTS}"
debug_parts = []
if MAX_TRAIN_SAMPLES is not None:
    debug_parts.append(f"train{MAX_TRAIN_SAMPLES}")
if MAX_VAL_SAMPLES is not None:
    debug_parts.append(f"{VALIDATION_SPLIT}{MAX_VAL_SAMPLES}")
cache_tag = base_cache_tag if not debug_parts else f"{base_cache_tag}_debug_{'_'.join(debug_parts)}"
print(f"Embedding cache tag: {cache_tag}")


# %% Load CelebA training and validation data
if VALIDATION_SPLIT not in {"valid", "test"}:
    raise ValueError("VALIDATION_SPLIT must be 'valid' or 'test'")

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
    split=VALIDATION_SPLIT,
    target_type="attr",
    transform=transform,
    download=False,
)

if MAX_TRAIN_SAMPLES is not None:
    trainset = torch.utils.data.Subset(trainset, range(min(MAX_TRAIN_SAMPLES, len(trainset))))
if MAX_VAL_SAMPLES is not None:
    valset = torch.utils.data.Subset(valset, range(min(MAX_VAL_SAMPLES, len(valset))))

train_image_loader = DataLoader(
    trainset,
    batch_size=EMBED_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)
val_image_loader = DataLoader(
    valset,
    batch_size=EMBED_BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
)

attr_names = list(getattr(valset, "attr_names", getattr(getattr(valset, "dataset", None), "attr_names", [])))
if not attr_names:
    attr_names = [f"attr_{j}" for j in range(40)]

print(f"CelebA train: {len(trainset):,}")
print(f"CelebA {VALIDATION_SPLIT}: {len(valset):,}")


# %% Build the Inception feature extractor
#
# The first run with INCEPTION_WEIGHTS="imagenet" may download torchvision's
# Inception-v3 checkpoint if it is not already in the Torch cache.

feature_net = InceptionEmbeddingNet(
    feature_layer=FEATURE_LAYER,
    weights_name=INCEPTION_WEIGHTS,
).to(device)
feature_net.eval()


# %% Compute or load real CelebA embeddings
train_cache_name = f"celeba_train_real_{cache_tag}.pt"
val_cache_name = f"celeba_{VALIDATION_SPLIT}_real_{cache_tag}.pt"

real_train = load_or_compute(
    CACHE_DIR / train_cache_name,
    lambda: embed_loader(feature_net, train_image_loader, device),
    expected_n=len(trainset),
    recompute=RECOMPUTE_EMBEDDINGS,
)
real_val = load_or_compute(
    CACHE_DIR / val_cache_name,
    lambda: embed_loader(feature_net, val_image_loader, device),
    expected_n=len(valset),
    recompute=RECOMPUTE_EMBEDDINGS,
)

train_z = real_train["z"]
train_attrs = real_train["attrs"]
val_z = real_val["z"]
val_attrs = real_val["attrs"]

print("train_z:", tuple(train_z.shape))
print("val_z:  ", tuple(val_z.shape))


# %% Create independent half-and-half splits in embedding space
p_indices, q_indices = random_halves(train_z.size(0), SEED)
val_p_indices, val_q_indices = random_halves(val_z.size(0), SEED + 10)

p_train_z = train_z[p_indices]
q_train_z = train_z[q_indices]
p_train_attrs = train_attrs[p_indices]
q_train_attrs = train_attrs[q_indices]

p_val_z = val_z[val_p_indices]
q_val_z = val_z[val_q_indices]
p_val_attrs = val_attrs[val_p_indices]
q_val_attrs = val_attrs[val_q_indices]

midpoint_train_z = make_midpoint_embeddings(p_train_z, q_train_z, seed=SEED + 20)
midpoint_val_z = make_midpoint_embeddings(p_val_z, q_val_z, seed=SEED + 21)

print(f"Training p:     {p_train_z.size(0):,} embeddings")
print(f"Training q:     {q_train_z.size(0):,} embeddings")
print(f"Training m:     {midpoint_train_z.size(0):,} embeddings")
print(f"Validation p:   {p_val_z.size(0):,} embeddings")
print(f"Validation q:   {q_val_z.size(0):,} embeddings")
print(f"Validation m:   {midpoint_val_z.size(0):,} embeddings")


# %% Train the embedding-space RDR estimator
set_seed(SEED)

p_loader = make_embedding_loader(
    p_train_z,
    args.batch_size,
    shuffle=True,
    seed=SEED + 1,
    drop_last=True,
)
midpoint_loader = make_embedding_loader(
    midpoint_train_z,
    args.batch_size,
    shuffle=True,
    seed=SEED + 2,
    drop_last=True,
)
p_val_loader = make_embedding_loader(
    p_val_z,
    args.batch_size,
    shuffle=False,
    seed=SEED + 3,
)
midpoint_val_loader = make_embedding_loader(
    midpoint_val_z,
    args.batch_size,
    shuffle=False,
    seed=SEED + 4,
)

model, train_losses, val_losses = dre.run_DRE_fdiv_embed(
    p_loader,
    midpoint_loader,
    num_epochs=args.num_epochs,
    steps_per_epoch=min(len(p_loader), args.steps_per_epoch),
    p_val_loader=p_val_loader,
    q_val_loader=midpoint_val_loader,
    loss_method=args.loss_method,
    device=device,
    mlp_hidden=args.mlp_hidden,
    val_steps=args.val_steps,
    return_val_losses=True,
)

best_epoch = int(np.argmin(val_losses)) + 1
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {min(val_losses):.6f}")


# %% Plot training and validation loss
fig, ax = plt.subplots(figsize=(7, 4))
epochs = np.arange(1, len(train_losses) + 1)
ax.plot(epochs, train_losses, label="training", color="tab:blue")
ax.plot(epochs, val_losses, "o-", label=f"{VALIDATION_SPLIT} validation", color="tab:orange")
ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.6)
ax.scatter(best_epoch, val_losses[best_epoch - 1], color="black", zorder=3)
ax.set(
    title=f"CelebA half-vs-half RDR on Inception embeddings (best epoch: {best_epoch})",
    xlabel="Epoch",
    ylabel=f"{LOSS_METHOD} objective",
)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()


# %% Evaluate RDR on every training observation in both halves
p_scores = predict_embeddings(model, p_train_z, args.eval_batch_size)
q_scores = predict_embeddings(model, q_train_z, args.eval_batch_size)
midpoint_scores = predict_embeddings(model, midpoint_train_z, args.eval_batch_size)

p_original_indices = p_indices
q_original_indices = q_indices

# These variables remain available for further interactive analysis:
# p_scores, q_scores, midpoint_scores, p_train_attrs, q_train_attrs,
# p_original_indices, q_original_indices, attr_names


# %% Numerical summaries
summary = pd.DataFrame({
    "p half": summarize(p_scores),
    "q half": summarize(q_scores),
    "midpoint": summarize(midpoint_scores),
})
display(summary.round(4))

# Under a successful null comparison, both randomly formed groups should have
# RDR values concentrated near 1.


# %% Compare the training score distributions
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 2, 51)
ax.hist(p_scores.numpy(), bins=bins, alpha=0.55, label="p half")
ax.hist(q_scores.numpy(), bins=bins, alpha=0.55, label="q half")
ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="RDR = 1")
ax.set(
    title="Estimated RDR for two random CelebA training halves",
    xlabel=r"Estimated $2p(z)/(p(z)+q(z))$",
    ylabel="Count",
    xlim=(0, 2),
)
ax.legend()
plt.tight_layout()
plt.show()


# %% Evaluate the independently split validation embeddings
p_val_scores = predict_embeddings(model, p_val_z, args.eval_batch_size)
q_val_scores = predict_embeddings(model, q_val_z, args.eval_batch_size)

val_summary = pd.DataFrame({
    "validation p half": summarize(p_val_scores),
    "validation q half": summarize(q_val_scores),
})
display(val_summary.round(4))

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(p_val_scores.numpy(), bins=bins, alpha=0.55, label="validation p half")
ax.hist(q_val_scores.numpy(), bins=bins, alpha=0.55, label="validation q half")
ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="RDR = 1")
ax.set(
    title=f"Estimated RDR for two random CelebA {VALIDATION_SPLIT} halves",
    xlabel=r"Estimated $2p(z)/(p(z)+q(z))$",
    ylabel="Count",
    xlim=(0, 2),
)
ax.legend()
plt.tight_layout()
plt.show()


# %% Attribute association checks for p-half validation scores
#
# In a null experiment, strong attribute associations can point to random split
# imbalance or overfitting. These tables are only diagnostic.

gp, A01 = celeb.prepare_gp_attrs(p_val_scores, p_val_attrs)
df_ranked, linear_res = celeb.run_linear(
    gp,
    A01,
    attr_names=attr_names[:A01.shape[1]],
    print_summary=False,
)
display(df_ranked.head(10))

df_beta_ranked, beta_res = celeb.run_beta_on_02(
    gp,
    A01,
    attr_names=attr_names[:A01.shape[1]],
    print_summary=False,
)
display(df_beta_ranked.head(10))

# %%
