# %% [markdown]
# # Controlled non-null MNIST RDR experiment
#
# This interactive experiment keeps P as the real MNIST data distribution and
# defines Q by sampling real MNIST images with a controlled digit-label
# distribution:
#
# - digits 0, 1: removed modes, Q(Y=k)=0
# - digits 2, 3: under-generated, Q(Y=k)=0.025
# - digits 4, 5, 6, 7: correct frequency, Q(Y=k)=0.10
# - digits 8, 9: over-generated, Q(Y=k)=0.275
#
# The RDR is
#
#     RDR(x) = 2 p(x) / (p(x) + q(x)) = p(x) / m(x),
#
# where m=(p+q)/2. Therefore, the density-ratio trainer receives real MNIST
# samples from p as its numerator and samples from the 50/50 midpoint mixture m
# as its denominator.
#
# Run cells individually in VS Code using **Run Cell**. Score summaries,
# intended/realized label frequencies, and tensor/CSV score artifacts are saved
# under `OUTPUT_DIR`.

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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


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
TRAIN_Q_SIZE = 60_000
VALIDATION_Q_SIZE = 5_000
TEST_Q_SIZE = 5_000
OUTPUT_DIR = Path(
    os.environ.get(
        "MNIST_LABEL_PERTURB_OUTPUT_DIR",
        REPO_ROOT / "experiments" / "results" / "MNIST_label_perturbation",
    )
)

P_LABEL_PROBS = torch.full((10,), 0.10, dtype=torch.float64)
Q_LABEL_PROBS = torch.tensor(
    [0.0, 0.0, 0.025, 0.025, 0.10, 0.10, 0.10, 0.10, 0.275, 0.275],
    dtype=torch.float64,
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


def labels_tensor(dataset):
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long)
    return torch.tensor([label for _, label in dataset], dtype=torch.long)


def build_label_index(labels):
    return {digit: (labels == digit).nonzero(as_tuple=True)[0] for digit in range(10)}


def sample_indices_by_label(labels, label_probs, n, seed):
    if not torch.isclose(label_probs.sum(), torch.tensor(1.0, dtype=label_probs.dtype)):
        raise ValueError(f"label probabilities must sum to 1, got {label_probs.sum().item()}")

    label_index = build_label_index(labels)
    for digit, prob in enumerate(label_probs):
        if prob > 0 and label_index[digit].numel() == 0:
            raise ValueError(f"cannot sample digit {digit}; no examples are available")

    generator = torch.Generator().manual_seed(seed)
    sampled_labels = torch.multinomial(label_probs, n, replacement=True, generator=generator)
    sampled_indices = torch.empty(n, dtype=torch.long)
    for digit in range(10):
        positions = (sampled_labels == digit).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        source_indices = label_index[digit]
        draws = torch.randint(
            source_indices.numel(),
            (positions.numel(),),
            generator=generator,
        )
        sampled_indices[positions] = source_indices[draws]
    return sampled_indices.tolist()


def split_indices(n, fractions, seed):
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    cutpoints = [int(round(n * sum(fractions[:k]))) for k in range(1, len(fractions))]
    starts = [0] + cutpoints
    ends = cutpoints + [n]
    return [perm[start:end] for start, end in zip(starts, ends)]


def sample_original_indices_by_label(all_labels, source_indices, label_probs, n, seed):
    source_indices = list(source_indices)
    source_labels = all_labels[source_indices]
    sampled_positions = sample_indices_by_label(source_labels, label_probs, n=n, seed=seed)
    return [source_indices[position] for position in sampled_positions]


class IndexedSubset(Subset):
    """Return the original MNIST index along with each image and label."""

    def __getitem__(self, position):
        image, label = self.dataset[self.indices[position]]
        return image, label, self.indices[position]

    def __getitems__(self, positions):
        """Support the batched fetching API required by recent PyTorch."""
        return [self[position] for position in positions]


class IndexedResampledDataset(Dataset):
    """Dataset view whose positions are deterministic resampled source indices."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        original_index = self.indices[position]
        image, label = self.dataset[original_index]
        return image, label, original_index

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
    if values.numel() == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q05": np.nan,
            "median": np.nan,
            "q95": np.nan,
            "max": np.nan,
        }
    return {
        "n": values.numel(),
        "mean": values.mean().item(),
        "std": values.std().item() if values.numel() > 1 else np.nan,
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


def label_frequency_frame(name, labels, target_probs=None):
    counts = torch.bincount(labels.long(), minlength=10).double()
    observed = counts / counts.sum()
    frame = pd.DataFrame({
        "distribution": name,
        "digit": np.arange(10),
        "count": counts.numpy().astype(int),
        "observed_prob": observed.numpy(),
    })
    if target_probs is not None:
        frame["target_prob"] = target_probs.numpy()
    return frame


def theoretical_rdr_by_digit(p_probs, q_probs):
    denominator = p_probs + q_probs
    rdr = torch.where(denominator > 0, 2.0 * p_probs / denominator, torch.nan)
    return pd.DataFrame({
        "digit": np.arange(10),
        "p_prob": p_probs.numpy(),
        "q_prob": q_probs.numpy(),
        "theoretical_rdr": rdr.numpy(),
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

train_labels_all = labels_tensor(mnist_train)
test_labels_all = labels_tensor(mnist_test)


# %% Create independent P and controlled-Q training/validation/test datasets
p_indices = list(range(len(mnist_train)))
val_p_indices, test_p_indices = split_indices(len(mnist_test), [0.5, 0.5], seed=SEED + 10)

q_indices = sample_indices_by_label(
    train_labels_all,
    Q_LABEL_PROBS,
    n=TRAIN_Q_SIZE,
    seed=SEED + 20,
)
val_q_indices = sample_original_indices_by_label(
    test_labels_all,
    val_p_indices,
    Q_LABEL_PROBS,
    n=VALIDATION_Q_SIZE,
    seed=SEED + 30,
)
test_q_indices = sample_original_indices_by_label(
    test_labels_all,
    test_p_indices,
    Q_LABEL_PROBS,
    n=TEST_Q_SIZE,
    seed=SEED + 40,
)

p_data = Subset(mnist_train, p_indices)
q_data = IndexedResampledDataset(mnist_train, q_indices)
val_p_data = Subset(mnist_test, val_p_indices)
val_q_data = IndexedResampledDataset(mnist_test, val_q_indices)
test_p_data = Subset(mnist_test, test_p_indices)
test_q_data = IndexedResampledDataset(mnist_test, test_q_indices)

train_p_label_freq = label_frequency_frame("train_p_real", train_labels_all, P_LABEL_PROBS)
train_q_label_freq = label_frequency_frame(
    "train_q_controlled",
    torch.tensor([train_labels_all[i].item() for i in q_indices]),
    Q_LABEL_PROBS,
)
val_p_label_freq = label_frequency_frame(
    "validation_loss_p_real",
    torch.tensor([test_labels_all[i].item() for i in val_p_indices]),
    P_LABEL_PROBS,
)
val_q_label_freq = label_frequency_frame(
    "validation_loss_q_controlled",
    torch.tensor([test_labels_all[i].item() for i in val_q_indices]),
    Q_LABEL_PROBS,
)
test_p_label_freq = label_frequency_frame(
    "test_p_real",
    torch.tensor([test_labels_all[i].item() for i in test_p_indices]),
    P_LABEL_PROBS,
)
test_q_label_freq = label_frequency_frame(
    "test_q_controlled",
    torch.tensor([test_labels_all[i].item() for i in test_q_indices]),
    Q_LABEL_PROBS,
)
label_frequencies = pd.concat(
    [
        train_p_label_freq,
        train_q_label_freq,
        val_p_label_freq,
        val_q_label_freq,
        test_p_label_freq,
        test_q_label_freq,
    ],
    ignore_index=True,
)

theoretical_digit_rdr = theoretical_rdr_by_digit(P_LABEL_PROBS, Q_LABEL_PROBS)

print(f"Training p:   {len(p_data):,} real observations")
print(f"Training q:   {len(q_data):,} controlled resampled observations")
print(f"Validation-loss p: {len(val_p_data):,} real observations")
print(f"Validation-loss q: {len(val_q_data):,} controlled resampled observations")
print(f"Evaluation-test p: {len(test_p_data):,} real observations")
print(f"Evaluation-test q: {len(test_q_data):,} controlled resampled observations")
display(label_frequencies.round(4))
display(theoretical_digit_rdr.round(4))


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

# Validation loss is computed on a held-out subset reserved for early stopping.
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

print(f"Full training epochs use {len(p_loader)} p-batches with drop_last=True.")
print(f"Dropped training p samples per epoch: {len(p_data) - len(p_loader) * BATCH_SIZE}")
print(f"Validation loss uses up to {VALIDATION_BATCHES} p-batches per epoch.")
print("Independent evaluation uses the disjoint test split and is not used for early stopping.")


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
    title="MNIST controlled-label RDR training loss",
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
ax.plot(epochs, val_losses, "o-", color="tab:orange", label="validation-loss split")
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


# %% Evaluate RDR on every training and independent test observation
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
    q_data,
    **eval_loader_args,
)
test_p_eval_loader = DataLoader(
    IndexedSubset(mnist_test, test_p_indices),
    **eval_loader_args,
)
test_q_eval_loader = DataLoader(
    test_q_data,
    **eval_loader_args,
)

train_p_scores, train_p_labels, train_p_original_indices = predict(model, p_eval_loader)
train_q_scores, train_q_labels, train_q_original_indices = predict(model, q_eval_loader)
test_p_scores, test_p_labels, test_p_original_indices = predict(model, test_p_eval_loader)
test_q_scores, test_q_labels, test_q_original_indices = predict(model, test_q_eval_loader)

# Backwards-compatible aliases for interactive cells.
p_scores, p_labels, p_original_indices = train_p_scores, train_p_labels, train_p_original_indices
q_scores, q_labels, q_original_indices = train_q_scores, train_q_labels, train_q_original_indices

# These variables remain available for further interactive analysis:
# train_p_scores, train_q_scores, test_p_scores, test_q_scores,
# train_p_labels, train_q_labels, test_p_labels, test_q_labels,
# train_p_original_indices, train_q_original_indices,
# test_p_original_indices, test_q_original_indices


# %% Report and save numerical RDR summaries
rdr_summary = pd.DataFrame({
    "train p real": summarize(train_p_scores),
    "train q controlled": summarize(train_q_scores),
    "test p real": summarize(test_p_scores),
    "test q controlled": summarize(test_q_scores),
})
display(rdr_summary.round(4))

rdr_scores = pd.concat(
    [
        score_frame("train", "p", train_p_scores, train_p_labels, train_p_original_indices),
        score_frame("train", "q", train_q_scores, train_q_labels, train_q_original_indices),
        score_frame("test", "p", test_p_scores, test_p_labels, test_p_original_indices),
        score_frame("test", "q", test_q_scores, test_q_labels, test_q_original_indices),
    ],
    ignore_index=True,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rdr_scores_path = OUTPUT_DIR / "rdr_scores.csv"
rdr_summary_path = OUTPUT_DIR / "rdr_summary.csv"
rdr_bundle_path = OUTPUT_DIR / "rdr_scores.pt"
label_frequencies_path = OUTPUT_DIR / "label_frequencies.csv"
theoretical_digit_rdr_path = OUTPUT_DIR / "theoretical_digit_rdr.csv"

rdr_scores.to_csv(rdr_scores_path, index=False)
rdr_summary.to_csv(rdr_summary_path)
label_frequencies.to_csv(label_frequencies_path, index=False)
theoretical_digit_rdr.to_csv(theoretical_digit_rdr_path, index=False)
torch.save(
    {
        "seed": SEED,
        "q_label_probs": Q_LABEL_PROBS,
        "p_label_probs_assumed": P_LABEL_PROBS,
        "best_epoch": best_epoch,
        "best_validation_loss": float(min(val_losses)),
        "split_indices": {
            "train_p_indices": torch.tensor(p_indices, dtype=torch.long),
            "validation_loss_p_indices": torch.tensor(val_p_indices, dtype=torch.long),
            "test_p_indices": torch.tensor(test_p_indices, dtype=torch.long),
        },
        "label_frequencies": label_frequencies,
        "theoretical_digit_rdr": theoretical_digit_rdr,
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
                "sampled_indices": torch.tensor(q_indices, dtype=torch.long),
            },
        },
        "validation_loss": {
            "p_indices": torch.tensor(val_p_indices, dtype=torch.long),
            "q_sampled_indices": torch.tensor(val_q_indices, dtype=torch.long),
        },
        "test": {
            "p": {
                "scores": test_p_scores,
                "labels": test_p_labels,
                "original_indices": test_p_original_indices,
            },
            "q": {
                "scores": test_q_scores,
                "labels": test_q_labels,
                "original_indices": test_q_original_indices,
                "sampled_indices": torch.tensor(test_q_indices, dtype=torch.long),
            },
        },
    },
    rdr_bundle_path,
)
print(f"Saved RDR scores to {rdr_scores_path}")
print(f"Saved RDR summary to {rdr_summary_path}")
print(f"Saved label frequencies to {label_frequencies_path}")
print(f"Saved theoretical digit RDR values to {theoretical_digit_rdr_path}")
print(f"Saved tensor bundle to {rdr_bundle_path}")


# %% Compare the score distributions
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, 2, 51)
ax.hist(p_scores.numpy(), bins=bins, alpha=0.55, label="p real")
ax.hist(q_scores.numpy(), bins=bins, alpha=0.55, label="q controlled")
ax.axvline(1.0, color="black", linestyle="--", linewidth=1.5, label="RDR = 1")
ax.set(
    title="Estimated RDR for controlled-label MNIST",
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
    ("train p real", train_p_scores, train_p_labels),
    ("train q controlled", train_q_scores, train_q_labels),
    ("test p real", test_p_scores, test_p_labels),
    ("test q controlled", test_q_scores, test_q_labels),
):
    for digit in range(10):
        digit_scores = scores[labels == digit]
        row = {
            "group": group,
            "digit": digit,
        }
        row.update({f"rdr_{key}": value for key, value in summarize(digit_scores).items()})
        scores_by_digit.append(row)

digit_summary = pd.DataFrame(scores_by_digit)
digit_summary = digit_summary.merge(
    theoretical_digit_rdr[["digit", "theoretical_rdr"]],
    on="digit",
    how="left",
)
display(digit_summary.round(4))

test_digit_summary = digit_summary[
    digit_summary["group"].isin(["test p real", "test q controlled"])
].reset_index(drop=True)
display(test_digit_summary.round(4))

digit_summary_path = OUTPUT_DIR / "digit_rdr_summary.csv"
test_digit_summary_path = OUTPUT_DIR / "test_digit_rdr_summary.csv"
digit_summary.to_csv(digit_summary_path, index=False)
test_digit_summary.to_csv(test_digit_summary_path, index=False)
print(f"Saved digit RDR summary to {digit_summary_path}")
print(f"Saved test digit RDR summary to {test_digit_summary_path}")


# %% Plot mean RDR by digit against the controlled target
fig, ax = plt.subplots(figsize=(8, 4))
for group, marker in (
    ("train p real", "o"),
    ("train q controlled", "s"),
    ("test p real", "^"),
    ("test q controlled", "D"),
):
    group_frame = digit_summary[digit_summary["group"] == group]
    ax.plot(group_frame["digit"], group_frame["rdr_mean"], marker=marker, label=group)

ax.plot(
    theoretical_digit_rdr["digit"],
    theoretical_digit_rdr["theoretical_rdr"],
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="label-only target",
)
ax.set(
    title="Mean estimated RDR by digit",
    xlabel="Digit",
    ylabel=r"Estimated $2p(x)/(p(x)+q(x))$",
    xticks=np.arange(10),
    ylim=(0, 2.1),
)
ax.grid(alpha=0.25)
ax.legend(ncol=2)
plt.tight_layout()
plt.show()

# %%
