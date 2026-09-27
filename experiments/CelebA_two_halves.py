# %% [markdown]
# # RDR between two random halves of CelebA
#
# This null experiment splits the official CelebA training observations into
# two disjoint random groups, p and q. It estimates
#
#     RDR(x) = 2 p(x) / (p(x) + q(x)) = p(x) / m(x),
#
# where m=(p+q)/2. The official CelebA test split is independently divided in
# half and used only to calculate validation loss and select the stopping epoch.

# %% Imports and configuration
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.DRE_batch as dre_batch

SEED = 42
CELEBA_DATA_ROOT = Path(
    os.environ.get("CELEBA_DATA_ROOT", "/hpc/group/mastatlab/yx306/CelebA")
)
BATCH_SIZE = int(os.environ.get("CELEBA_BATCH_SIZE", "512"))
NUM_WORKERS = int(os.environ.get("CELEBA_NUM_WORKERS", "2"))
NUM_EPOCHS = int(os.environ.get("CELEBA_NUM_EPOCHS", "20"))
EARLY_STOP_PATIENCE = int(os.environ.get("CELEBA_EARLY_STOP_PATIENCE", "5"))
EARLY_STOP_MIN_DELTA = float(
    os.environ.get("CELEBA_EARLY_STOP_MIN_DELTA", "1e-5")
)
PRINT_EVERY = int(os.environ.get("CELEBA_PRINT_EVERY", "50"))
OUTPUT_DIR = Path(
    os.environ.get(
        "CELEBA_TWO_HALVES_OUTPUT_DIR",
        REPO_ROOT / "experiments" / "outputs" / "celeba_two_halves",
    )
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Output directory: {OUTPUT_DIR}")
SCRIPT_START_TIME = time.perf_counter()


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


class IndexedSubset(Subset):
    """Return each CelebA observation with its original split index."""

    def __getitem__(self, position):
        image, attributes = self.dataset[self.indices[position]]
        return image, attributes, self.indices[position]

    def __getitems__(self, positions):
        return [self[position] for position in positions]


@torch.no_grad()
def predict(model, loader):
    model.eval()
    model_device = next(model.parameters()).device
    scores, attributes, indices = [], [], []
    for images, batch_attributes, batch_indices in loader:
        values = model(images.to(model_device, non_blocking=True)).reshape(-1)
        scores.append(values.cpu())
        attributes.append(batch_attributes.cpu())
        indices.append(batch_indices.cpu())
    return torch.cat(scores), torch.cat(attributes), torch.cat(indices)


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


def make_score_frame(split, group, scores, attributes, original_indices, attr_names):
    frame = pd.DataFrame({
        "split": split,
        "group": group,
        "original_index": original_indices.numpy(),
        "rdr": scores.numpy(),
    })
    attr_frame = pd.DataFrame(
        attributes.numpy(),
        columns=[f"attr_{name}" for name in attr_names],
    )
    return pd.concat([frame, attr_frame], axis=1)


timing_records = []


def record_timing(stage, elapsed_seconds, **details):
    record = {
        "stage": stage,
        "elapsed_seconds": float(elapsed_seconds),
        "elapsed_minutes": float(elapsed_seconds) / 60.0,
    }
    record.update(details)
    timing_records.append(record)
    print(
        f"{stage} time: {elapsed_seconds:.2f} seconds "
        f"({elapsed_seconds / 60.0:.2f} minutes)"
    )


set_seed(SEED)


# %% Load the official CelebA training and test splits
transform = transforms.Compose([
    transforms.CenterCrop(178),
    transforms.Resize(
        (64, 64),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

trainset = datasets.CelebA(
    root=CELEBA_DATA_ROOT,
    split="train",
    target_type="attr",
    transform=transform,
    download=False,
)
testset = datasets.CelebA(
    root=CELEBA_DATA_ROOT,
    split="test",
    target_type="attr",
    transform=transform,
    download=False,
)

print(f"Training data: {len(trainset):,} observations")
print(f"Test data:     {len(testset):,} observations")


# %% Create independent half-and-half training and validation splits
p_indices, q_indices = random_halves(len(trainset), SEED)
val_p_indices, val_q_indices = random_halves(len(testset), SEED + 10)

p_data = Subset(trainset, p_indices)
q_data = Subset(trainset, q_indices)
val_p_data = Subset(testset, val_p_indices)
val_q_data = Subset(testset, val_q_indices)

print(f"Training p:   {len(p_data):,} observations")
print(f"Training q:   {len(q_data):,} observations")
print(f"Validation p: {len(val_p_data):,} observations")
print(f"Validation q: {len(val_q_data):,} observations")


# %% Construct numerator loaders and independent midpoint samplers
def make_loader(dataset, *, shuffle, seed, drop_last):
    kwargs = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "shuffle": shuffle,
        "num_workers": NUM_WORKERS,
        "drop_last": drop_last,
        "pin_memory": torch.cuda.is_available(),
    }
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(**kwargs)


p_loader = make_loader(p_data, shuffle=True, seed=SEED + 1, drop_last=True)
p_mixture_loader = make_loader(
    p_data, shuffle=True, seed=SEED + 2, drop_last=False
)
q_mixture_loader = make_loader(
    q_data, shuffle=True, seed=SEED + 3, drop_last=False
)
midpoint_sampler = dre_batch.make_loader_mixture_sampler(
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
validation_midpoint_sampler = dre_batch.make_loader_mixture_sampler(
    val_p_mixture_loader,
    val_q_mixture_loader,
    p_frac=0.5,
)


# %% Train the RDR estimator
set_seed(SEED)

training_start_time = time.perf_counter()
model, train_losses, val_losses = (
    dre_batch.run_DRE_fdiv_cnn_minibatch_celeba64(
        p_loader=p_loader,
        q_sampler=midpoint_sampler,
        num_epochs=NUM_EPOCHS,
        val_loader=val_p_loader,
        val_q_sampler=validation_midpoint_sampler,
        val_q_batches=len(val_p_loader),
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_min_delta=EARLY_STOP_MIN_DELTA,
        restore_best=True,
        return_val_losses=True,
        print_every=PRINT_EVERY,
        bn_freeze_epoch=5,
    )
)
training_time = time.perf_counter() - training_start_time
record_timing(
    "training",
    training_time,
    epochs_run=len(val_losses),
    training_steps=len(train_losses),
    steps_per_epoch=len(p_loader),
    validation_batches_per_epoch=len(val_p_loader),
)

best_epoch = int(np.argmin(val_losses)) + 1
print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {min(val_losses):.6f}")
print("The in-memory model has been restored to this epoch.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
loss_history = pd.concat(
    [
        pd.DataFrame({
            "kind": "training_step",
            "step": np.arange(1, len(train_losses) + 1),
            "loss": train_losses,
        }),
        pd.DataFrame({
            "kind": "validation_epoch",
            "step": np.arange(1, len(val_losses) + 1),
            "loss": val_losses,
        }),
    ],
    ignore_index=True,
)
loss_history.to_csv(OUTPUT_DIR / "loss_history.csv", index=False)


# %% Plot training and validation loss
steps_per_epoch = len(p_loader)
validation_steps = steps_per_epoch * np.arange(1, len(val_losses) + 1)
validation_steps = np.minimum(validation_steps, len(train_losses))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(
    np.arange(1, len(train_losses) + 1),
    train_losses,
    color="tab:blue",
    alpha=0.8,
    label="training",
)
ax.plot(
    validation_steps,
    val_losses,
    "o-",
    color="tab:orange",
    label="test validation",
)
best_step = validation_steps[best_epoch - 1]
ax.axvline(best_step, color="black", linestyle="--", alpha=0.7)
ax.scatter(best_step, val_losses[best_epoch - 1], color="black", zorder=3)
ax.set(
    title=f"CelebA half-vs-half RDR loss (best epoch: {best_epoch})",
    xlabel="Training step",
    ylabel="Hellinger objective",
)
ax.grid(alpha=0.25)
ax.legend()
plt.tight_layout()
plt.show()


# %% Evaluate RDR on every observation in both training and validation groups
eval_loader_args = {
    "batch_size": BATCH_SIZE,
    "shuffle": False,
    "num_workers": NUM_WORKERS,
    "drop_last": False,
    "pin_memory": torch.cuda.is_available(),
}

p_eval_loader = DataLoader(
    IndexedSubset(trainset, p_indices),
    **eval_loader_args,
)
q_eval_loader = DataLoader(
    IndexedSubset(trainset, q_indices),
    **eval_loader_args,
)
val_p_eval_loader = DataLoader(
    IndexedSubset(testset, val_p_indices),
    **eval_loader_args,
)
val_q_eval_loader = DataLoader(
    IndexedSubset(testset, val_q_indices),
    **eval_loader_args,
)

train_scoring_start_time = time.perf_counter()
p_scores, p_attributes, p_original_indices = predict(model, p_eval_loader)
q_scores, q_attributes, q_original_indices = predict(model, q_eval_loader)
train_scoring_time = time.perf_counter() - train_scoring_start_time
record_timing(
    "train_rdr_scoring",
    train_scoring_time,
    observations=len(p_scores) + len(q_scores),
    batches=len(p_eval_loader) + len(q_eval_loader),
)
validation_scoring_start_time = time.perf_counter()
val_p_scores, val_p_attributes, val_p_original_indices = predict(
    model, val_p_eval_loader
)
val_q_scores, val_q_attributes, val_q_original_indices = predict(
    model, val_q_eval_loader
)
validation_scoring_time = time.perf_counter() - validation_scoring_start_time
record_timing(
    "validation_rdr_scoring",
    validation_scoring_time,
    observations=len(val_p_scores) + len(val_q_scores),
    batches=len(val_p_eval_loader) + len(val_q_eval_loader),
)

# These variables remain available for further interactive analysis:
# p_scores, q_scores, p_attributes, q_attributes,
# val_p_scores, val_q_scores, val_p_attributes, val_q_attributes,
# p_original_indices, q_original_indices,
# val_p_original_indices, val_q_original_indices, trainset.attr_names


# %% Numerical summaries and saved per-observation RDR values
summary = pd.DataFrame({
    "train p half": summarize(p_scores),
    "train q half": summarize(q_scores),
    "validation p half": summarize(val_p_scores),
    "validation q half": summarize(val_q_scores),
})
display(summary.round(4))

train_rdr = pd.concat(
    [
        make_score_frame(
            "train", "p", p_scores, p_attributes, p_original_indices, trainset.attr_names
        ),
        make_score_frame(
            "train", "q", q_scores, q_attributes, q_original_indices, trainset.attr_names
        ),
    ],
    ignore_index=True,
)
validation_rdr = pd.concat(
    [
        make_score_frame(
            "test_validation",
            "p",
            val_p_scores,
            val_p_attributes,
            val_p_original_indices,
            testset.attr_names,
        ),
        make_score_frame(
            "test_validation",
            "q",
            val_q_scores,
            val_q_attributes,
            val_q_original_indices,
            testset.attr_names,
        ),
    ],
    ignore_index=True,
)
summary.to_csv(OUTPUT_DIR / "rdr_summary.csv")
train_rdr.to_csv(OUTPUT_DIR / "train_rdr_values.csv", index=False)
validation_rdr.to_csv(OUTPUT_DIR / "validation_rdr_values.csv", index=False)
total_time = time.perf_counter() - SCRIPT_START_TIME
record_timing(
    "total_through_rdr_save",
    total_time,
    observations=len(train_rdr) + len(validation_rdr),
)
timing_summary = pd.DataFrame(timing_records)
display(timing_summary.round(4))
timing_summary.to_csv(OUTPUT_DIR / "timing_summary.csv", index=False)
print(f"Saved RDR summary to {OUTPUT_DIR / 'rdr_summary.csv'}")
print(f"Saved training RDR values to {OUTPUT_DIR / 'train_rdr_values.csv'}")
print(
    f"Saved validation RDR values to {OUTPUT_DIR / 'validation_rdr_values.csv'}"
)
print(f"Saved timing summary to {OUTPUT_DIR / 'timing_summary.csv'}")

# Under a successful null comparison, both randomly formed groups should have
# RDR values concentrated near 1.


# %% Compare the score distributions
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 2, 51)
ax.hist(p_scores.numpy(), bins=bins, alpha=0.55, label="p half")
ax.hist(q_scores.numpy(), bins=bins, alpha=0.55, label="q half")
ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="RDR = 1")
ax.set(
    title="Estimated RDR for two random CelebA training halves",
    xlabel=r"Estimated $2p(x)/(p(x)+q(x))$",
    ylabel="Count",
    xlim=(0, 2),
)
ax.legend()
plt.tight_layout()
plt.show()

# %%
