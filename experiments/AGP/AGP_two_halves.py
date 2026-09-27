# %% [markdown]
# # RDR between two random halves of AGP
#
# This null experiment splits the AGP training observations into two disjoint
# random groups, p and q. It estimates
#
#     RDR(x) = 2 p(x) / (p(x) + q(x)) = p(x) / m(x),
#
# where m=(p+q)/2. The independent AGP test data are also split in half and
# used only to calculate validation loss and select the stopping epoch.

# %% Imports and configuration
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from torch.utils.data import DataLoader, Subset, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.DRE_func as dre

SEED = 42
AGP_DATA_ROOT = Path(
    os.environ.get("AGP_DATA_ROOT", "/hpc/group/mastatlab/yx306/AGP/data")
)
BATCH_SIZE = int(os.environ.get("AGP_BATCH_SIZE", "512"))
NUM_WORKERS = 0
NUM_EPOCHS = int(os.environ.get("AGP_NUM_EPOCHS", "200"))
EARLY_STOP_PATIENCE = int(os.environ.get("AGP_EARLY_STOP_PATIENCE", "10"))
EARLY_STOP_MIN_DELTA = float(os.environ.get("AGP_EARLY_STOP_MIN_DELTA", "1e-5"))
OUTPUT_DIR = Path(
    os.environ.get(
        "AGP_TWO_HALVES_OUTPUT_DIR",
        REPO_ROOT / "experiments" / "results" / "AGP_two_halves",
    )
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %% Reproducibility and experiment helpers
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def random_halves(n, seed):
    """Return a reproducible random partition; p gets floor(n/2)."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    cut = n // 2
    return order[:cut].tolist(), order[cut:].tolist()


def make_loader_mixture_sampler(p_loader, q_loader, p_frac=0.5):
    """Return batches from p_frac * p + (1 - p_frac) * q."""
    if not 0.0 <= p_frac <= 1.0:
        raise ValueError("p_frac must be in [0, 1]")

    p_iter = iter(p_loader)
    q_iter = iter(q_loader)

    def take(loader, iterator, n):
        chunks = []
        needed = n
        while needed > 0:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            values = batch[0] if isinstance(batch, (list, tuple)) else batch
            chunk = values[:needed]
            chunks.append(chunk)
            needed -= chunk.size(0)
        return torch.cat(chunks, dim=0), iterator

    @torch.no_grad()
    def sampler(batch_size):
        nonlocal p_iter, q_iter
        n_p = int(round(batch_size * p_frac))
        n_q = batch_size - n_p
        parts = []
        if n_p:
            values, p_iter = take(p_loader, p_iter, n_p)
            parts.append(values)
        if n_q:
            values, q_iter = take(q_loader, q_iter, n_q)
            parts.append(values)
        mixed = torch.cat(parts, dim=0)
        return mixed[torch.randperm(mixed.size(0))]

    return sampler


@torch.no_grad()
def predict(model, data, indices):
    model.eval()
    model_device = next(model.parameters()).device
    dataset = TensorDataset(
        data[indices],
        torch.as_tensor(indices, dtype=torch.long),
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    scores, original_indices = [], []
    for values, batch_indices in loader:
        scores.append(model(values.to(model_device)).reshape(-1).cpu())
        original_indices.append(batch_indices)
    return torch.cat(scores), torch.cat(original_indices)


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


def score_frame(split, group, scores, indices):
    return pd.DataFrame({
        "split": split,
        "group": group,
        "original_index": indices.numpy(),
        "rdr": scores.numpy(),
    })


set_seed(SEED)


# %% Load the existing AGP training and test compositions
train_path = AGP_DATA_ROOT / "yx1_xtrain_raw.csv"
test_path = AGP_DATA_ROOT / "yx1_xtest_raw.csv"
if not train_path.exists() or not test_path.exists():
    raise FileNotFoundError(
        "AGP inputs were not found. Set AGP_DATA_ROOT to the directory "
        "containing yx1_xtrain_raw.csv and yx1_xtest_raw.csv."
    )

x_train = torch.from_numpy(np.loadtxt(train_path, delimiter=",", dtype=np.float32))
x_test = torch.from_numpy(np.loadtxt(test_path, delimiter=",", dtype=np.float32))

if x_train.ndim != 2 or x_test.ndim != 2:
    raise ValueError("AGP train and test inputs must both be two-dimensional.")
if x_train.shape[1] != x_test.shape[1]:
    raise ValueError("AGP train and test inputs must have the same number of taxa.")

print(f"Training data: {tuple(x_train.shape)}")
print(f"Test data:     {tuple(x_test.shape)}")


# %% Create independent half-and-half training and validation splits
p_indices, q_indices = random_halves(len(x_train), SEED)
val_p_indices, val_q_indices = random_halves(len(x_test), SEED + 10)

train_dataset = TensorDataset(x_train)
test_dataset = TensorDataset(x_test)

p_data = Subset(train_dataset, p_indices)
q_data = Subset(train_dataset, q_indices)
val_p_data = Subset(test_dataset, val_p_indices)
val_q_data = Subset(test_dataset, val_q_indices)

print(f"Training p:   {len(p_data):,} observations")
print(f"Training q:   {len(q_data):,} observations")
print(f"Validation p: {len(val_p_data):,} observations")
print(f"Validation q: {len(val_q_data):,} observations")


# %% Construct numerator loaders and independent midpoint samplers
def make_loader(dataset, *, shuffle, seed, drop_last):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
        num_workers=NUM_WORKERS,
        drop_last=drop_last,
    )


p_loader = make_loader(p_data, shuffle=True, seed=SEED + 1, drop_last=True)
p_mixture_loader = make_loader(
    p_data, shuffle=True, seed=SEED + 2, drop_last=False
)
q_mixture_loader = make_loader(
    q_data, shuffle=True, seed=SEED + 3, drop_last=False
)
midpoint_sampler = make_loader_mixture_sampler(
    p_mixture_loader,
    q_mixture_loader,
    p_frac=0.5,
)

val_p_loader = make_loader(
    val_p_data, shuffle=False, seed=SEED + 11, drop_last=False
)
val_p_mixture_loader = make_loader(
    val_p_data, shuffle=True, seed=SEED + 12, drop_last=False
)
val_q_mixture_loader = make_loader(
    val_q_data, shuffle=True, seed=SEED + 13, drop_last=False
)
validation_midpoint_sampler = make_loader_mixture_sampler(
    val_p_mixture_loader,
    val_q_mixture_loader,
    p_frac=0.5,
)


# %% Train the RDR estimator
set_seed(SEED)

model, history = dre.run_DRE_fdiv_SGD(
    p_loader=p_loader,
    q_source=midpoint_sampler,
    xi=0.5,
    num_epochs=NUM_EPOCHS,
    loss_method="Hellinger",
    NN="MLP",
    log_scale=False,
    val_p_loader=val_p_loader,
    val_q_source=validation_midpoint_sampler,
    early_stop_patience=EARLY_STOP_PATIENCE,
    early_stop_min_delta=EARLY_STOP_MIN_DELTA,
    restore_best=True,
    device=device,
)

train_losses = history["train_loss"]
val_losses = history["val_loss"]
best_epoch = int(np.argmin(val_losses)) + 1

print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {min(val_losses):.6f}")
print("The in-memory model has been restored to this epoch.")


# %% Plot training and validation loss
fig, ax = plt.subplots(figsize=(7, 4))
epochs = np.arange(1, len(train_losses) + 1)
ax.plot(epochs, train_losses, color="tab:blue", label="training")
ax.plot(epochs, val_losses, color="tab:orange", marker="o", label="test validation")
ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.7)
ax.scatter(best_epoch, val_losses[best_epoch - 1], color="black", zorder=3)
ax.set(
    title=f"AGP half-vs-half RDR loss (best epoch: {best_epoch})",
    xlabel="Epoch",
    ylabel="Hellinger objective",
)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()


# %% Evaluate RDR on every training and validation observation
train_p_scores, train_p_original_indices = predict(model, x_train, p_indices)
train_q_scores, train_q_original_indices = predict(model, x_train, q_indices)
val_p_scores, val_p_original_indices = predict(model, x_test, val_p_indices)
val_q_scores, val_q_original_indices = predict(model, x_test, val_q_indices)

# Backwards-compatible aliases for interactive cells that used the old names.
p_scores, p_original_indices = train_p_scores, train_p_original_indices
q_scores, q_original_indices = train_q_scores, train_q_original_indices

# These variables remain available for further interactive analysis:
# train_p_scores, train_q_scores, val_p_scores, val_q_scores,
# train_p_original_indices, train_q_original_indices,
# val_p_original_indices, val_q_original_indices


# %% Report and save numerical RDR summaries
rdr_summary = pd.DataFrame({
    "train p half": summarize(train_p_scores),
    "train q half": summarize(train_q_scores),
    "validation p half": summarize(val_p_scores),
    "validation q half": summarize(val_q_scores),
})
display(rdr_summary.round(4))

# Under a successful null comparison, both randomly formed groups should have
# RDR values concentrated near 1.

rdr_scores = pd.concat(
    [
        score_frame("train", "p", train_p_scores, train_p_original_indices),
        score_frame("train", "q", train_q_scores, train_q_original_indices),
        score_frame("validation", "p", val_p_scores, val_p_original_indices),
        score_frame("validation", "q", val_q_scores, val_q_original_indices),
    ],
    ignore_index=True,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rdr_scores_path = OUTPUT_DIR / "rdr_scores.csv"
rdr_summary_path = OUTPUT_DIR / "rdr_summary.csv"
rdr_bundle_path = OUTPUT_DIR / "rdr_scores.pt"
rdr_scores.to_csv(rdr_scores_path, index=False)
rdr_summary.to_csv(rdr_summary_path)
torch.save(
    {
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_loss": float(min(val_losses)),
        "train": {
            "p": {
                "scores": train_p_scores,
                "original_indices": train_p_original_indices,
            },
            "q": {
                "scores": train_q_scores,
                "original_indices": train_q_original_indices,
            },
        },
        "validation": {
            "p": {
                "scores": val_p_scores,
                "original_indices": val_p_original_indices,
            },
            "q": {
                "scores": val_q_scores,
                "original_indices": val_q_original_indices,
            },
        },
    },
    rdr_bundle_path,
)
print(f"Saved RDR scores to {rdr_scores_path}")
print(f"Saved RDR summary to {rdr_summary_path}")
print(f"Saved tensor bundle to {rdr_bundle_path}")


# %% Compare the score distributions
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 2, 51)
ax.hist(p_scores.numpy(), bins=bins, alpha=0.55, label="p half")
ax.hist(q_scores.numpy(), bins=bins, alpha=0.55, label="q half")
ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="RDR = 1")
ax.set(
    title="Estimated RDR for two random AGP training halves",
    xlabel=r"Estimated $2p(x)/(p(x)+q(x))$",
    ylabel="Count",
    xlim=(0, 2),
)
ax.legend()
plt.tight_layout()
plt.show()

# %%
