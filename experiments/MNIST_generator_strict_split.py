# %%
"""Strict train/validation/test MNIST generator RDR experiment.

This recreates the first-version VAE/DCGAN RDR figure with disjoint roles:
training uses 55,000 observations from the official MNIST train split,
validation loss uses the remaining 5,000 train observations, and final plotted
evaluation uses all 10,000 official MNIST test observations.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils.DRE_batch as dre_batch
from experiments.MNIST import sampling as mnist_sampling
import experiments.MNIST.helpers as mnist


DEFAULT_DATA_ROOT = Path("/hpc/group/mastatlab/yx306/MNIST")
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "results" / "MNIST_generator_trainval_testall"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("MNIST_DATA_ROOT", DEFAULT_DATA_ROOT)))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("MNIST_GENERATOR_STRICT_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)))
    parser.add_argument("--generators", nargs="+", default=os.environ.get("MNIST_GENERATORS", "vae dcgan").split(), choices=["vae", "dcgan"])
    parser.add_argument("--seed", type=int, default=int(os.environ.get("MNIST_STRICT_SEED", 42)))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("MNIST_STRICT_BATCH_SIZE", 512)))
    parser.add_argument("--num-workers", type=int, default=int(os.environ.get("MNIST_STRICT_NUM_WORKERS", 0)))
    parser.add_argument("--num-epochs", type=int, default=int(os.environ.get("MNIST_STRICT_NUM_EPOCHS", 20)))
    parser.add_argument("--vae-epochs", type=int, default=int(os.environ.get("MNIST_STRICT_VAE_EPOCHS", 3)))
    parser.add_argument("--patience", type=int, default=int(os.environ.get("MNIST_STRICT_PATIENCE", 5)))
    parser.add_argument("--validation-batches", type=int, default=int(os.environ.get("MNIST_STRICT_VALIDATION_BATCHES", 10)))
    parser.add_argument("--validation-size", type=int, default=int(os.environ.get("MNIST_STRICT_VALIDATION_SIZE", 5000)))
    parser.add_argument("--eval-n", type=int, default=int(os.environ.get("MNIST_STRICT_EVAL_N", 0)), help="Number of test observations to score; 0 means all test data.")
    parser.add_argument("--top-k", type=int, default=int(os.environ.get("MNIST_STRICT_TOP_K", 40)))
    parser.add_argument("--output-alpha", type=float, default=float(os.environ.get("MNIST_STRICT_OUTPUT_ALPHA", 0.5)))
    parser.add_argument(
        "--eval-input-scale",
        choices=["model", "zero_one"],
        default=os.environ.get("MNIST_STRICT_EVAL_INPUT_SCALE", "model"),
        help="model scores final test tensors in the training scale; zero_one scores final test tensors in [0,1].",
    )
    parser.add_argument(
        "--protocol",
        choices=["trainval_testall", "legacy_test_validation"],
        default=os.environ.get("MNIST_STRICT_PROTOCOL", "trainval_testall"),
        help="trainval_testall holds out 5,000 train images for validation; legacy_test_validation uses full test for validation and evaluation.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny plumbing check.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_indices(n, fractions, seed):
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    cutpoints = [int(round(n * sum(fractions[:k]))) for k in range(1, len(fractions))]
    starts = [0] + cutpoints
    ends = cutpoints + [n]
    return [perm[start:end] for start, end in zip(starts, ends)]


def split_train_validation_indices(n, validation_size, seed):
    if not 0 < validation_size < n:
        raise ValueError(f"validation_size must be in 1..{n - 1}, got {validation_size}")
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=generator).tolist()
    return perm[validation_size:], perm[:validation_size]


def labels_tensor(dataset):
    if hasattr(dataset, "targets"):
        return torch.as_tensor(dataset.targets, dtype=torch.long)
    return torch.tensor([label for _, label in dataset], dtype=torch.long)


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    model_device = next(model.parameters()).device
    scores, labels, images = [], [], []
    for batch in loader:
        x = batch[0]
        y = batch[1] if isinstance(batch, (tuple, list)) and len(batch) > 1 else None
        values = model(x.to(model_device)).reshape(-1).cpu()
        scores.append(values)
        images.append(x.cpu())
        if y is not None:
            labels.append(y.cpu())
    out_labels = torch.cat(labels) if labels else None
    return torch.cat(scores), out_labels, torch.cat(images)


@torch.no_grad()
def predict_tensor(model, x, batch_size):
    model.eval()
    model_device = next(model.parameters()).device
    chunks = []
    for start in range(0, x.size(0), batch_size):
        values = model(x[start:start + batch_size].to(model_device)).reshape(-1).cpu()
        chunks.append(values)
    return torch.cat(chunks)


def to_display(x):
    x = x.detach().cpu()
    if x.ndim == 2:
        x = x.view(-1, 1, 28, 28)
    if float(x.min()) < -0.01:
        x = (x + 1.0) / 2.0
    return x.clamp(0, 1)


def to_zero_one(x):
    x = x.detach().cpu()
    if float(x.min()) < -0.01:
        x = (x + 1.0) / 2.0
    return x.clamp(0, 1)


def summarize(values):
    values = values.float().cpu()
    return {
        "n": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std()) if values.numel() > 1 else float("nan"),
        "min": float(values.min()),
        "q05": float(torch.quantile(values, 0.05)),
        "median": float(values.median()),
        "q95": float(torch.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def hellinger_p_vs_midpoint_from_scores(p_scores, q_scores):
    p_scores = p_scores.float().clamp_min(1e-8)
    q_scores = q_scores.float().clamp_min(1e-8)
    affinity = 0.5 * (torch.sqrt(p_scores).mean() + torch.sqrt(q_scores).mean())
    raw = float(1.0 - affinity)
    return raw, max(0.0, raw)


def load_vae(data_root, device):
    if str(data_root) not in sys.path:
        sys.path.insert(0, str(data_root))
    return mnist.VAEWrapper.from_repo(
        weights=data_root / "mnist_vae" / "vae_epoch_25.pth",
        module_path="mnist_vae",
        class_name="VAE",
        latent_dim=20,
        device=str(device),
    )


def make_generator_bundle(name, data_root, device):
    if name == "dcgan":
        generator, z_dim = mnist.build_dcgan28(
            data_root / "mnist_dcgan" / "netG_epoch_99.pth",
            device=device,
        )

        def midpoint_sampler(real_loader):
            return mnist_sampling.make_q_mixed_sampler(
                generator,
                z_dim,
                real_loader,
                gen_frac=0.5,
                post=None,
            )

        def sample_generated(n, seed):
            sampler = mnist.get_generator_sampler(
                generator=generator,
                z_dim=z_dim,
                device=device,
                z_type="noise_4d",
                generator_output_range="[-1,1]",
                return_range="[-1,1]",
                out_size=(28, 28),
                flatten=False,
                channels=1,
            )
            return sampler(n=n, seed=seed).cpu()

        return midpoint_sampler, sample_generated

    if name == "vae":
        vae = load_vae(data_root, device)

        def midpoint_sampler(real_loader):
            return mnist_sampling.make_mnist_vae_50_50_sampler(
                vae,
                real_loader=real_loader,
                post=lambda t: t * 2.0 - 1.0,
                return_source=False,
            )

        def sample_generated(n, seed):
            set_seed(seed)
            return (vae.generate(n).cpu() * 2.0 - 1.0).clamp(-1, 1)

        return midpoint_sampler, sample_generated

    raise ValueError(f"unknown generator {name}")


def train_one(
    name,
    args,
    mnist_train,
    mnist_test,
    train_indices,
    val_indices,
    test_indices,
    val_dataset,
):
    generator_seed_offset = {"dcgan": 1000, "vae": 2000}[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed + generator_seed_offset)

    epochs = args.vae_epochs if name == "vae" else args.num_epochs
    if args.smoke:
        epochs = 1

    train_data = Subset(mnist_train, train_indices)
    val_data = Subset(val_dataset, val_indices)
    test_data = Subset(mnist_test, test_indices)

    loader_train_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "drop_last": True,
    }
    p_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + generator_seed_offset + 1),
        **loader_train_args,
    )
    midpoint_real_loader = DataLoader(
        train_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + generator_seed_offset + 2),
        **loader_train_args,
    )
    val_p_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_midpoint_real_loader = DataLoader(
        val_data,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + generator_seed_offset + 3),
        **loader_train_args,
    )

    make_midpoint_sampler, sample_generated = make_generator_bundle(name, args.data_root, device)
    q_sampler = make_midpoint_sampler(midpoint_real_loader)
    val_q_sampler = make_midpoint_sampler(val_midpoint_real_loader)

    model, losses, val_losses = dre_batch.run_DRE_fdiv_cnn_minibatch(
        p_loader=p_loader,
        q_sampler=q_sampler,
        num_epochs=epochs,
        print_every=100,
        bn_freeze_epoch=0 if name == "vae" else 5,
        val_loader=val_p_loader,
        val_q_sampler=val_q_sampler,
        val_q_batches=args.validation_batches,
        early_stop_patience=args.patience,
        restore_best=True,
        return_val_losses=True,
        output_alpha=args.output_alpha,
    )

    eval_n = len(test_data) if args.eval_n <= 0 else min(args.eval_n, len(test_data))
    test_eval_data = Subset(test_data, list(range(eval_n)))
    test_eval_loader = DataLoader(
        test_eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    test_p_scores, test_p_labels, test_p_images_model = predict_loader(model, test_eval_loader)
    test_q_images_model = sample_generated(eval_n, seed=args.seed + generator_seed_offset + 4)
    if args.eval_input_scale == "zero_one":
        test_p_images_for_score = to_zero_one(test_p_images_model)
        test_q_images_for_score = to_zero_one(test_q_images_model)
        test_p_scores = predict_tensor(model, test_p_images_for_score, args.batch_size)
        test_q_scores = predict_tensor(model, test_q_images_for_score, args.batch_size)
    else:
        test_p_images_for_score = test_p_images_model
        test_q_images_for_score = test_q_images_model
        test_q_scores = predict_tensor(model, test_q_images_for_score, args.batch_size)

    h2_raw, h2_clipped = hellinger_p_vs_midpoint_from_scores(test_p_scores, test_q_scores)
    result = {
        "name": name,
        "model": model,
        "losses": losses,
        "val_losses": val_losses,
        "best_epoch": int(np.argmin(val_losses)) + 1 if val_losses else None,
        "best_validation_loss": float(min(val_losses)) if val_losses else None,
        "test_p_scores": test_p_scores,
        "test_q_scores": test_q_scores,
        "test_p_labels": test_p_labels,
        "test_p_images": to_display(test_p_images_model),
        "test_q_images": to_display(test_q_images_model),
        "test_eval_input_scale": args.eval_input_scale,
        "test_h2_p_midpoint_raw": h2_raw,
        "test_h2_p_midpoint": h2_clipped,
    }
    return result


def image_grid(ax, images, scores, kind, n=40):
    extremes = mnist.find_extreme_indices(scores, top_k=n)
    rows, cols = 5, 8
    panel_names = [("smallest", "Smallest"), ("closest", "Close-to-1"), ("largest", "Largest")]
    ax.axis("off")
    subfig = ax.get_subplotspec().subgridspec(1, 3, wspace=0.16)
    fig = ax.figure
    for panel_idx, (key, label) in enumerate(panel_names):
        vals, idx = extremes[key]
        vals = vals.detach().cpu()
        idx = idx.detach().cpu()
        panel_ax = fig.add_subplot(subfig[panel_idx])
        panel_ax.axis("off")
        panel_ax.set_title(
            f"{kind}: {label} scores\n(min={float(vals.min()):.4g}, max={float(vals.max()):.4g})",
            fontsize=9,
        )
        selected = images[idx[: rows * cols], 0].numpy()
        mosaic = selected.reshape(rows, cols, 28, 28).transpose(0, 2, 1, 3).reshape(rows * 28, cols * 28)
        panel_ax.imshow(mosaic, cmap="gray", vmin=0, vmax=1, aspect="equal")


def add_panel_label(fig, ax, label, *, dx=-0.035, dy=0.015):
    bbox = ax.get_position()
    fig.text(
        bbox.x0 + dx,
        bbox.y1 + dy,
        label,
        fontsize=18,
        fontweight="normal",
        ha="left",
        va="bottom",
    )


def make_figure(results, output_dir):
    fig = plt.figure(figsize=(20, 10))
    outer = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.12, 1.12, 1.12, 1.0],
        height_ratios=[1.0, 1.55, 1.55],
        wspace=0.42,
        hspace=0.55,
    )

    if "vae" in results:
        result = results["vae"]
        title = "VAE"
        fig.text(0.26, 0.97, title, ha="center", va="top", fontsize=20)

        hist_ax = fig.add_subplot(outer[0, 0])
        bins = np.linspace(0, 2, 51)
        hist_ax.hist(result["test_p_scores"].numpy(), bins=bins, density=True, alpha=0.35, color="tab:purple", edgecolor="black", label="p-observed sample")
        hist_ax.hist(result["test_q_scores"].numpy(), bins=bins, density=True, alpha=0.35, color="tab:blue", edgecolor="black", label=f"q ({title})")
        hist_ax.set_xlabel("ratio r(x)")
        hist_ax.set_ylabel("Density")
        hist_ax.set_xlim(0, 2)
        hist_ax.set_title(f"Histogram {title}, h^2(p,(p+q)/2) = {result['test_h2_p_midpoint']:.3f}", fontsize=11)
        hist_ax.legend(fontsize=9)
        add_panel_label(fig, hist_ax, "A")

        fig.add_subplot(outer[0, 1]).axis("off")

        p_grid_ax = fig.add_subplot(outer[1, 0:2])
        image_grid(p_grid_ax, result["test_p_images"], result["test_p_scores"], f"{title} Real")
        add_panel_label(fig, p_grid_ax, "B")
        q_grid_ax = fig.add_subplot(outer[2, 0:2])
        image_grid(q_grid_ax, result["test_q_images"], result["test_q_scores"], f"{title} Generated")

    if "dcgan" in results:
        result = results["dcgan"]
        title = "DCGAN"
        fig.text(0.75, 0.97, title, ha="center", va="top", fontsize=20)

        hist_ax = fig.add_subplot(outer[0, 2])
        bins = np.linspace(0, 2, 51)
        hist_ax.hist(result["test_p_scores"].numpy(), bins=bins, density=True, alpha=0.35, color="tab:purple", edgecolor="black", label="p-observed sample")
        hist_ax.hist(result["test_q_scores"].numpy(), bins=bins, density=True, alpha=0.35, color="tab:green", edgecolor="black", label=f"q ({title})")
        hist_ax.set_xlabel("ratio r(x)")
        hist_ax.set_ylabel("Density")
        hist_ax.set_xlim(0, 2)
        hist_ax.set_title(f"Histogram {title}, h^2(p,(p+q)/2) = {result['test_h2_p_midpoint']:.3f}", fontsize=11)
        hist_ax.legend(fontsize=9)
        add_panel_label(fig, hist_ax, "C")

        violin_ax = fig.add_subplot(outer[0, 3])
        sns.violinplot(
            x=result["test_p_labels"].numpy(),
            y=result["test_p_scores"].numpy(),
            inner="quartile",
            cut=0,
            bw_adjust=0.8,
            ax=violin_ax,
        )
        violin_ax.set_ylim(0, 2)
        violin_ax.set_xlabel("Digits")
        violin_ax.set_ylabel("r(x)")
        violin_ax.set_title(f"{title}: Distribution of r(X_p) by digits", fontsize=10)
        add_panel_label(fig, violin_ax, "D")

        p_grid_ax = fig.add_subplot(outer[1, 2:4])
        image_grid(p_grid_ax, result["test_p_images"], result["test_p_scores"], f"{title} Real")
        add_panel_label(fig, p_grid_ax, "E")
        q_grid_ax = fig.add_subplot(outer[2, 2:4])
        image_grid(q_grid_ax, result["test_q_images"], result["test_q_scores"], f"{title} Generated")

    if {"vae", "dcgan"}.issubset(results):
        fig.add_artist(plt.Line2D([0.5, 0.5], [0.04, 0.94], color="0.82", linewidth=1))

    output_png = output_dir / "mnist_generator_strict_split_figure.png"
    output_pdf = output_dir / "mnist_generator_strict_split_figure.pdf"
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)
    return output_png, output_pdf


def save_outputs(results, args, train_indices, val_indices, test_indices, split_sources):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary_rows = []
    for name, result in results.items():
        for group, scores, labels in (
            ("test_p_real", result["test_p_scores"], result["test_p_labels"]),
            ("test_q_generated", result["test_q_scores"], None),
        ):
            for i, score in enumerate(scores.tolist()):
                rows.append({
                    "generator": name,
                    "split": "test",
                    "group": group,
                    "position": i,
                    "label": int(labels[i]) if labels is not None else np.nan,
                    "rdr": float(score),
                })
            summary = summarize(scores)
            summary.update({"generator": name, "split": "test", "group": group})
            summary_rows.append(summary)
        summary_rows.append({
            "generator": name,
            "split": "test",
            "group": "p_vs_midpoint",
            "n": int(result["test_p_scores"].numel() + result["test_q_scores"].numel()),
            "mean": result["test_h2_p_midpoint"],
            "std": np.nan,
            "min": np.nan,
            "q05": np.nan,
            "median": np.nan,
            "q95": np.nan,
            "max": np.nan,
        })

        torch.save(
            {
                "config": vars(args),
                "generator": name,
                "model_state": result["model"].state_dict(),
                "losses": result["losses"],
                "val_losses": result["val_losses"],
                "best_epoch": result["best_epoch"],
                "best_validation_loss": result["best_validation_loss"],
                "test_h2_p_midpoint_raw": result["test_h2_p_midpoint_raw"],
                "test_h2_p_midpoint": result["test_h2_p_midpoint"],
                "test_p_scores": result["test_p_scores"],
                "test_q_scores": result["test_q_scores"],
                "test_p_labels": result["test_p_labels"],
                "test_eval_input_scale": result["test_eval_input_scale"],
                "split_indices": {
                    "train_indices": torch.tensor(train_indices, dtype=torch.long),
                    "validation_loss_indices": torch.tensor(val_indices, dtype=torch.long),
                    "test_indices": torch.tensor(test_indices, dtype=torch.long),
                },
            },
            args.output_dir / f"{name}_rdr_checkpoint.pt",
        )

    scores_path = args.output_dir / "rdr_scores.csv"
    summary_path = args.output_dir / "rdr_summary.csv"
    split_path = args.output_dir / "split_manifest.json"
    pd.DataFrame(rows).to_csv(scores_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    split_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "protocol": args.protocol,
                "eval_input_scale": args.eval_input_scale,
                "train": {"source": split_sources["train"], "n": len(train_indices)},
                "validation_loss": {"source": split_sources["validation_loss"], "n": len(val_indices)},
                "test": {"source": split_sources["test"], "n": len(test_indices)},
                "train_indices": train_indices,
                "validation_loss_indices": val_indices,
                "test_indices": test_indices,
            },
            indent=2,
        )
    )
    return scores_path, summary_path, split_path


def main():
    args = parse_args()
    if args.smoke:
        args.batch_size = min(args.batch_size, 64)
        args.validation_batches = 1
        args.eval_n = min(args.eval_n, 128)

    set_seed(args.seed)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t * 2.0 - 1.0),
    ])
    mnist_train = datasets.MNIST(root=args.data_root, train=True, download=False, transform=transform)
    mnist_test = datasets.MNIST(root=args.data_root, train=False, download=False, transform=transform)

    if args.protocol == "trainval_testall":
        train_indices, val_indices = split_train_validation_indices(
            len(mnist_train),
            validation_size=args.validation_size,
            seed=args.seed + 10,
        )
        val_dataset = mnist_train
        test_indices = list(range(len(mnist_test)))
        split_sources = {
            "train": "MNIST train minus validation-loss subset",
            "validation_loss": "deterministic validation subset of MNIST train",
            "test": "all official MNIST test observations",
        }
        if set(train_indices).intersection(val_indices):
            raise RuntimeError("training and validation-loss indices overlap")
    else:
        train_indices = list(range(len(mnist_train)))
        val_indices = list(range(len(mnist_test)))
        val_dataset = mnist_test
        test_indices = list(range(len(mnist_test)))
        split_sources = {
            "train": "all official MNIST train observations",
            "validation_loss": "all official MNIST test observations, reused for final evaluation",
            "test": "all official MNIST test observations, reused for validation loss",
        }
    if args.smoke:
        train_indices = train_indices[:512]
        val_indices = val_indices[:128]
        test_indices = test_indices[:128]

    print(f"Training P: {len(train_indices):,} official MNIST train observations")
    print(f"Validation-loss P: {len(val_indices):,} observations from {split_sources['validation_loss']}")
    print(f"Final test P: {len(test_indices):,} official MNIST test observations")
    print("Generated Q samples use independent latent draws for train, validation, and final test.")
    print(f"Ratio-network bounded sigmoid alpha: {args.output_alpha:g}")
    print(f"Protocol: {args.protocol}")
    print(f"Final evaluation input scale: {args.eval_input_scale}")

    results = {}
    for generator_name in args.generators:
        print(f"\n=== Training {generator_name.upper()} RDR ===")
        results[generator_name] = train_one(
            generator_name,
            args,
            mnist_train,
            mnist_test,
            train_indices,
            val_indices,
            test_indices,
            val_dataset,
        )
        print(
            f"{generator_name.upper()} best epoch: {results[generator_name]['best_epoch']} | "
            f"test h^2(p,(p+q)/2): {results[generator_name]['test_h2_p_midpoint']:.6f}"
        )

    scores_path, summary_path, split_path = save_outputs(
        results,
        args,
        train_indices,
        val_indices,
        test_indices,
        split_sources,
    )
    figure_png, figure_pdf = make_figure(results, args.output_dir)
    print(f"Saved scores: {scores_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved split manifest: {split_path}")
    print(f"Saved figure PNG: {figure_png}")
    print(f"Saved figure PDF: {figure_pdf}")


if __name__ == "__main__":
    main()
