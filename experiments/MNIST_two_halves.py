# %% [markdown]
# # RDR between two random halves of MNIST
#
# This interactive experiment splits the 60,000 MNIST training observations
# into two disjoint random groups of 30,000. The first group is p and the
# second group is q.
#
# The RDR is
#
#     RDR(x) = 2 p(x) / (p(x) + q(x)) = p(x) / m(x),
#
# where m=(p+q)/2. Therefore, the density-ratio trainer receives samples from
# p as its numerator and samples from the 50/50 midpoint mixture m as its
# denominator.
#
# Run cells individually in VS Code using **Run Cell**. Score summaries and
# tensor/CSV score artifacts are saved under `OUTPUT_DIR`.

# %% Imports and configuration
import random
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from importlib import reload


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.DRE_batch as dre_batch

SEED = 123
DATA_ROOT = REPO_ROOT / "data"
DOWNLOAD = True
BATCH_SIZE = 512
# Interactive Python cells do not provide a reliably importable __main__
# module to spawned DataLoader workers on Python 3.14. Keep loading in the
# kernel process to avoid multiprocessing pickling errors.
NUM_WORKERS = 0
NUM_EPOCHS = 20
EARLY_STOP_PATIENCE = 5
VALIDATION_BATCHES = 10
OUTPUT_DIR = Path(
    os.environ.get(
        "MNIST_TWO_HALVES_OUTPUT_DIR",
        REPO_ROOT / "experiments" / "results" / "MNIST_two_halves",
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def scale_to_minus_one_one(image):
    """Map an image tensor from [0, 1] to [-1, 1]."""
    return image * 2.0 - 1.0


def random_halves(n, seed):
    """Return a reproducible random partition; group p gets floor(n/2)."""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    cut = n // 2
    return order[:cut].tolist(), order[cut:].tolist()


class IndexedSubset(Subset):
    """Return the original MNIST index along with each image and label."""

    def __getitem__(self, position):
        image, label = self.dataset[self.indices[position]]
        return image, label, self.indices[position]

    def __getitems__(self, positions):
        """Support the batched fetching API required by recent PyTorch."""
        return [self[position] for position in positions]


@torch.no_grad()
def predict(model, loader):
    model.eval()
    model_device = next(model.parameters()).device
    scores, labels, indices = [], [], []
    for images, batch_labels, batch_indices in loader:
        values = model(images.to(model_device)).reshape(-1)
        scores.append(values.cpu())
        labels.append(batch_labels)
        indices.append(batch_indices)
    return torch.cat(scores), torch.cat(labels), torch.cat(indices)


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


def score_frame(split, group, scores, labels, indices):
    return pd.DataFrame({
        "split": split,
        "group": group,
        "original_index": indices.numpy(),
        "label": labels.numpy(),
        "rdr": scores.numpy(),
    })


set_seed(SEED)


# %% Load MNIST training and test data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(scale_to_minus_one_one),
])

mnist_train = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    download=DOWNLOAD,
    transform=transform,
)

mnist_test = datasets.MNIST(
    root=DATA_ROOT,
    train=False,
    download=DOWNLOAD,
    transform=transform,
)

# %% Create independent half-and-half training and validation splits

p_indices, q_indices = random_halves(len(mnist_train), SEED)
p_data = Subset(mnist_train, p_indices)
q_data = Subset(mnist_train, q_indices)

# The official MNIST test data are used only to calculate validation loss.
val_p_indices, val_q_indices = random_halves(len(mnist_test), SEED + 10)
val_p_data = Subset(mnist_test, val_p_indices)
val_q_data = Subset(mnist_test, val_q_indices)

print(f"Training p:   {len(p_data):,} observations")
print(f"Training q:   {len(q_data):,} observations")
print(f"Validation p: {len(val_p_data):,} observations")
print(f"Validation q: {len(val_q_data):,} observations")

# %% Construct p, q, and midpoint-mixture loaders
loader_args = {
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "drop_last": True,
}

p_loader = DataLoader(
    p_data,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED + 1),
    **loader_args,
)
q_loader = DataLoader(
    q_data,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED + 2),
    **loader_args,
)

# Samples m=(p+q)/2. Training p against m estimates p/m, which is the RDR.
midpoint_sampler = dre_batch.make_loader_mixture_sampler(
    p_loader,
    q_loader,
    p_frac=0.5,
)

# Validation loss is computed independently on the official MNIST test set.
val_p_loader = DataLoader(
    val_p_data,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    drop_last=False,
)
val_p_mixture_loader = DataLoader(
    val_p_data,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED + 11),
    **loader_args,
)
val_q_mixture_loader = DataLoader(
    val_q_data,
    shuffle=True,
    generator=torch.Generator().manual_seed(SEED + 12),
    **loader_args,
)
validation_midpoint_sampler = dre_batch.make_loader_mixture_sampler(
    val_p_mixture_loader,
    val_q_mixture_loader,
    p_frac=0.5,
)

# %% Train the RDR estimator
set_seed(SEED)

model, losses, val_losses = dre_batch.run_DRE_fdiv_cnn_minibatch(
    p_loader=p_loader,
    q_sampler=midpoint_sampler,
    num_epochs=NUM_EPOCHS,
    print_every=100,
    bn_freeze_epoch=5,
    val_loader=val_p_loader,
    val_q_sampler=validation_midpoint_sampler,
    val_q_batches=VALIDATION_BATCHES,
    early_stop_patience=EARLY_STOP_PATIENCE,
    restore_best=True,
    return_val_losses=True,
)

# %% Plot the training loss
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(losses, color="tab:blue", alpha=0.8)
ax.set(
    title="MNIST half-vs-half RDR training loss",
    xlabel="Training step",
    ylabel="Hellinger objective",
)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.show()

# %% Plot validation loss and identify the restored epoch
best_epoch = int(np.argmin(val_losses)) + 1

fig, ax = plt.subplots(figsize=(7, 4))
epochs = np.arange(1, len(val_losses) + 1)
ax.plot(epochs, val_losses, "o-", color="tab:orange", label="test-set validation")
ax.axvline(best_epoch, color="black", linestyle="--", alpha=0.7)
ax.scatter(best_epoch, val_losses[best_epoch - 1], color="black", zorder=3)
ax.set(
    title=f"Validation loss (best epoch: {best_epoch})",
    xlabel="Epoch",
    ylabel="Hellinger objective",
    xticks=epochs,
)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()

print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {min(val_losses):.6f}")
print("The in-memory model has been restored to this epoch.")

# %% Evaluate RDR on every training and validation observation
eval_loader_args = {
    "batch_size": BATCH_SIZE,
    "shuffle": False,
    "num_workers": NUM_WORKERS,
    "drop_last": False,
}

p_eval_loader = DataLoader(
    IndexedSubset(mnist_train, p_indices),
    **eval_loader_args,
)
q_eval_loader = DataLoader(
    IndexedSubset(mnist_train, q_indices),
    **eval_loader_args,
)
val_p_eval_loader = DataLoader(
    IndexedSubset(mnist_test, val_p_indices),
    **eval_loader_args,
)
val_q_eval_loader = DataLoader(
    IndexedSubset(mnist_test, val_q_indices),
    **eval_loader_args,
)

train_p_scores, train_p_labels, train_p_original_indices = predict(model, p_eval_loader)
train_q_scores, train_q_labels, train_q_original_indices = predict(model, q_eval_loader)
val_p_scores, val_p_labels, val_p_original_indices = predict(model, val_p_eval_loader)
val_q_scores, val_q_labels, val_q_original_indices = predict(model, val_q_eval_loader)

# Backwards-compatible aliases for interactive cells that used the old names.
p_scores, p_labels, p_original_indices = train_p_scores, train_p_labels, train_p_original_indices
q_scores, q_labels, q_original_indices = train_q_scores, train_q_labels, train_q_original_indices

# These variables remain available for further interactive analysis:
# train_p_scores, train_q_scores, val_p_scores, val_q_scores,
# train_p_labels, train_q_labels, val_p_labels, val_q_labels,
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
        score_frame("train", "p", train_p_scores, train_p_labels, train_p_original_indices),
        score_frame("train", "q", train_q_scores, train_q_labels, train_q_original_indices),
        score_frame("validation", "p", val_p_scores, val_p_labels, val_p_original_indices),
        score_frame("validation", "q", val_q_scores, val_q_labels, val_q_original_indices),
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
                "labels": train_p_labels,
                "original_indices": train_p_original_indices,
            },
            "q": {
                "scores": train_q_scores,
                "labels": train_q_labels,
                "original_indices": train_q_original_indices,
            },
        },
        "validation": {
            "p": {
                "scores": val_p_scores,
                "labels": val_p_labels,
                "original_indices": val_p_original_indices,
            },
            "q": {
                "scores": val_q_scores,
                "labels": val_q_labels,
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
    title="Estimated RDR for two random MNIST halves",
    xlabel=r"Estimated $2p(x)/(p(x)+q(x))$",
    ylabel="Count",
    xlim=(0, 2),
)
ax.legend()
plt.tight_layout()
plt.show()

# %% RDR summaries by digit
scores_by_digit = []
for group, scores, labels in (
    ("train p half", train_p_scores, train_p_labels),
    ("train q half", train_q_scores, train_q_labels),
    ("validation p half", val_p_scores, val_p_labels),
    ("validation q half", val_q_scores, val_q_labels),
):
    for digit in range(10):
        digit_scores = scores[labels == digit]
        scores_by_digit.append({
            "group": group,
            "digit": digit,
            "n": digit_scores.numel(),
            "mean_rdr": digit_scores.mean().item(),
            "std_rdr": digit_scores.std().item(),
        })

digit_summary = pd.DataFrame(scores_by_digit)
display(digit_summary.round(4))

# %%
