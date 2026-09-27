#!/usr/bin/env python3
"""Train and summarize the CelebA-vs-DDIM RDR model for Slurm.

This is the batch-job version of ``CelebA_ddim.py`` through the cell titled:
``reload the saved trained RDR and summarize on full train/test samples``.
"""

import argparse
import math
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    f"/tmp/matplotlib-celeba-ddim-{os.environ.get('SLURM_JOB_ID', os.getpid())}",
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import experiments.CelebA.helpers as celeb
import utils.DRE_batch as dre_batch
import utils.DRE_func as dre
import utils.diagnostics as help_func
DDIM_SAMPLES_PER_SHARD = 10_000


def env_int(name, default):
    return int(os.environ.get(name, default))


def env_float(name, default):
    return float(os.environ.get(name, default))


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_shards(value):
    shards = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            shards.extend(range(int(start), int(end) + 1))
        else:
            shards.append(int(part))
    if not shards:
        raise ValueError("Shard list is empty.")
    return tuple(shards)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed set to {seed}]")


def ddim_shard_sample_count(ddim_data_dir, shard_ids, verify_counts=False):
    shard_ids = list(shard_ids)
    if not verify_counts:
        return len(shard_ids) * DDIM_SAMPLES_PER_SHARD

    total = 0
    for shard_id in shard_ids:
        shard_path = celeb._find_shard_path(ddim_data_dir, int(shard_id))
        shard_obj = torch.load(shard_path, map_location="cpu")
        total += int(celeb._extract_tensor_from_obj(shard_obj).shape[0])
        del shard_obj
    return total


def make_markdown_table(df, floatfmt=".6g"):
    table = df.reset_index() if df.index.names != [None] else df.copy()
    headers = [str(col) for col in table.columns]
    rows = []
    for row in table.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append("" if not np.isfinite(value) else format(float(value), floatfmt))
            else:
                formatted.append(str(value))
        rows.append(formatted)

    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    lines = [
        "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return "\n".join(lines)


def summarize_losses(losses, val_losses, steps_per_epoch):
    rows = []
    num_loss_epochs = max(
        int(math.ceil(len(losses) / steps_per_epoch)) if losses else 0,
        len(val_losses),
    )
    for epoch_idx in range(num_loss_epochs):
        start = epoch_idx * steps_per_epoch
        end = min((epoch_idx + 1) * steps_per_epoch, len(losses))
        train_epoch_losses = np.asarray(losses[start:end], dtype=float)
        train_finite = np.isfinite(train_epoch_losses)
        val_loss = val_losses[epoch_idx] if epoch_idx < len(val_losses) else np.nan
        rows.append(
            {
                "epoch": epoch_idx + 1,
                "train_steps": int(end - start),
                "train_loss_mean": (
                    float(np.mean(train_epoch_losses[train_finite]))
                    if train_finite.any()
                    else np.nan
                ),
                "train_loss_last": (
                    float(train_epoch_losses[train_finite][-1])
                    if train_finite.any()
                    else np.nan
                ),
                "validation_loss": float(val_loss) if np.isfinite(val_loss) else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["train_steps", "train_loss_mean", "train_loss_last", "validation_loss"])
    return pd.DataFrame(rows).set_index("epoch")


def save_loss_plot(losses, val_losses, steps_per_epoch, output_path):
    epochs_run = len(val_losses)
    val_steps = steps_per_epoch * np.arange(1, epochs_run + 1)
    val_steps = np.minimum(val_steps, len(losses))

    fig, ax = plt.subplots(figsize=(6, 4))
    help_func.plot_losses(losses, label="train (CelebA64)", ax=ax, color="tab:blue")
    if val_losses:
        ax.plot(val_steps, val_losses, "o-", label="validation", color="tab:orange", linewidth=2)
        best_idx = int(np.argmin(val_losses))
        best_epoch = best_idx + 1
        best_step = int(val_steps[best_idx])
        best_val = float(val_losses[best_idx])
        ax.axvline(best_step, linestyle="--", alpha=0.35)
        ax.scatter([best_step], [best_val], s=60, zorder=5, label=f"best @ epoch {best_epoch}")
    else:
        print("No finite validation losses were recorded.")

    ax.set_title("Training vs Validation Loss (CelebA64)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def batch_images(batch):
    return batch[0] if isinstance(batch, (tuple, list)) else batch


@torch.no_grad()
def score_rdr_tensor(model, x):
    model.eval()
    p_model = next(model.parameters())
    x = x.to(device=p_model.device, dtype=p_model.dtype, non_blocking=True)
    r = model(x).view(-1)
    if getattr(model, "log_scale", False):
        r = torch.exp(r)
    return r.detach().cpu()


@torch.no_grad()
def score_rdr_loader(model, loader, max_batches=None):
    scores = []
    for k, batch in enumerate(loader):
        if max_batches is not None and k >= max_batches:
            break
        scores.append(score_rdr_tensor(model, batch_images(batch)))
    return torch.cat(scores, dim=0)


@torch.no_grad()
def score_rdr_sampler_n(model, sampler, batch_size, n_samples):
    p_model = next(model.parameters())
    if hasattr(sampler, "reset"):
        sampler.reset(epoch=0)
    scores = []
    remaining = int(n_samples)
    while remaining > 0:
        this_batch = min(int(batch_size), remaining)
        x = sampler(batch_size=this_batch, device=p_model.device, dtype=p_model.dtype)
        scores.append(score_rdr_tensor(model, x))
        remaining -= this_batch
    return torch.cat(scores, dim=0)


def summarize_rdr_scores(split, group, scores):
    arr = scores.numpy()
    finite = np.isfinite(arr)
    if finite.any():
        stats = help_func.summarize_vector(arr[finite])
    else:
        stats = {
            "length": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q1": np.nan,
            "median": np.nan,
            "q3": np.nan,
            "max": np.nan,
        }
    stats.update(
        {
            "split": split,
            "group": group,
            "n_nonfinite": int((~finite).sum()),
        }
    )
    return stats


def summarize_full_train_test_rdr(
    checkpoint_path,
    celeba_loader,
    celeba_testloader,
    trainset,
    testset,
    ddim_data_dir,
    train_q_shards,
    test_q_shards,
    train_q_n,
    test_q_n,
    batch_size,
):
    ratio_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=ratio_device)
    model_ddim = dre.RatioNetCelebA64(in_ch=3, ndf=64, log_scale=False).to(ratio_device)
    model_ddim.load_state_dict(checkpoint["model_state"])

    losses_ddim = checkpoint["losses"]
    val_losses = checkpoint.get("val_losses", [])

    rdr_train_fake_sampler = celeb.DDIMFakeOnlySampler(
        ddim_data_dir=ddim_data_dir,
        shard_ids=train_q_shards,
    )
    rdr_test_fake_sampler = celeb.DDIMFakeOnlySampler(
        ddim_data_dir=ddim_data_dir,
        shard_ids=test_q_shards,
    )

    rdr_full_summary_rows = [
        summarize_rdr_scores("train", "real", score_rdr_loader(model_ddim, celeba_loader)),
        summarize_rdr_scores(
            "train",
            "generated",
            score_rdr_sampler_n(model_ddim, rdr_train_fake_sampler, batch_size, train_q_n),
        ),
        summarize_rdr_scores("test", "real", score_rdr_loader(model_ddim, celeba_testloader)),
        summarize_rdr_scores(
            "test",
            "generated",
            score_rdr_sampler_n(model_ddim, rdr_test_fake_sampler, batch_size, test_q_n),
        ),
    ]
    rdr_full_summary_df = (
        pd.DataFrame(rdr_full_summary_rows)
        .set_index(["split", "group"])
        [["length", "n_nonfinite", "mean", "std", "min", "q1", "median", "q3", "max"]]
    )
    print(
        "Full-split RDR summary; "
        f"train P/Q use {len(trainset):,}/{train_q_n:,} images and "
        f"test P/Q use {len(testset):,}/{test_q_n:,} images. "
        "Validation P/Q is reserved for early stopping."
    )
    print(rdr_full_summary_df.round(4))
    return rdr_full_summary_df, losses_ddim, val_losses


def write_markdown_report(
    report_path,
    args,
    split_sample_sizes,
    loss_epoch_df,
    rdr_full_summary_df,
    checkpoint_path,
    loss_plot_path,
    sample_batch_shape,
):
    best_validation = np.nan
    best_epoch = ""
    if not loss_epoch_df.empty and loss_epoch_df["validation_loss"].notna().any():
        best_epoch = int(loss_epoch_df["validation_loss"].idxmin())
        best_validation = float(loss_epoch_df.loc[best_epoch, "validation_loss"])

    config_df = pd.DataFrame(
        [
            {"setting": "seed", "value": args.seed},
            {"setting": "batch_size", "value": args.batch_size},
            {"setting": "num_epochs_requested", "value": args.num_epochs},
            {"setting": "steps_per_epoch", "value": args.steps_per_epoch},
            {"setting": "train_steps_recorded", "value": int(loss_epoch_df["train_steps"].sum()) if not loss_epoch_df.empty else 0},
            {"setting": "validation_q_batches", "value": args.val_q_batches},
            {"setting": "early_stop_patience", "value": args.early_stop_patience},
            {"setting": "early_stop_min_delta", "value": args.early_stop_min_delta},
            {"setting": "restore_mode", "value": args.restore_mode},
            {"setting": "best_validation_epoch", "value": best_epoch},
            {"setting": "best_validation_loss", "value": best_validation},
            {"setting": "sample_mixture_batch_shape", "value": sample_batch_shape},
        ]
    )

    text = [
        "# CelebA DDIM RDR Training",
        "",
        "This report covers training, checkpoint reload, and full train/test RDR summaries.",
        "The validation split is reserved for early stopping and is not reused for the final RDR table.",
        "",
        "## Run Configuration",
        "",
        make_markdown_table(config_df),
        "",
        "## Independent Split Sample Sizes",
        "",
        make_markdown_table(split_sample_sizes),
        "",
        "## Training And Validation Loss By Epoch",
        "",
        make_markdown_table(loss_epoch_df, floatfmt=".6f"),
        "",
        "## Full Train/Test RDR Summary",
        "",
        make_markdown_table(rdr_full_summary_df.round(6), floatfmt=".6f"),
        "",
        "## Artifacts",
        "",
        make_markdown_table(
            pd.DataFrame(
                [
                    {"artifact": "checkpoint", "path": str(checkpoint_path)},
                    {"artifact": "loss_plot", "path": str(loss_plot_path)},
                    {"artifact": "loss_table_csv", "path": str(report_path.with_name("loss_epoch_summary.csv"))},
                    {"artifact": "split_table_csv", "path": str(report_path.with_name("split_sample_sizes.csv"))},
                    {"artifact": "rdr_summary_csv", "path": str(report_path.with_name("rdr_full_summary.csv"))},
                ]
            )
        ),
        "",
    ]
    report_path.write_text("\n".join(text), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", default=os.environ.get("RDR_REPO_DIR", str(REPO_DIR)))
    parser.add_argument("--data-root", default=os.environ.get("CELEBA_DATA_ROOT", "/hpc/group/mastatlab/yx306/CelebA/"))
    parser.add_argument("--ddim-data-dir", default=os.environ.get("CELEBA_DDIM_DATA_DIR", "/hpc/group/mastatlab/yx306/CelebA/DDIM/data"))
    parser.add_argument("--output-dir", default=os.environ.get("CELEBA_DDIM_TRAIN_OUTPUT_DIR", "experiments/outputs/celeba_ddim_train"))
    parser.add_argument("--checkpoint-name", default=os.environ.get("CELEBA_DDIM_CHECKPOINT_NAME", "ratio_ddim_celeba64_valshards16-17_testshards18-19_alpha01_nep10.pt"))
    parser.add_argument("--seed", type=int, default=env_int("CELEBA_SEED", 42))
    parser.add_argument("--batch-size", type=int, default=env_int("CELEBA_BATCH_SIZE", 512))
    parser.add_argument("--num-workers", type=int, default=env_int("CELEBA_NUM_WORKERS", 2))
    parser.add_argument("--num-epochs", type=int, default=env_int("CELEBA_NUM_EPOCHS", 20))
    parser.add_argument("--val-q-batches", type=int, default=env_int("CELEBA_VAL_Q_BATCHES", 2))
    parser.add_argument("--early-stop-patience", type=int, default=env_int("CELEBA_EARLY_STOP_PATIENCE", 5))
    parser.add_argument("--early-stop-min-delta", type=float, default=env_float("CELEBA_EARLY_STOP_MIN_DELTA", 0.0))
    parser.add_argument("--print-every", type=int, default=env_int("CELEBA_PRINT_EVERY", 1))
    parser.add_argument("--restore-mode", choices=["best_val", "last_finite_train"], default=os.environ.get("CELEBA_RESTORE_MODE", "best_val"))
    parser.add_argument("--train-q-shards", default=os.environ.get("CELEBA_TRAIN_Q_SHARDS", "0-15"))
    parser.add_argument("--validation-q-shards", default=os.environ.get("CELEBA_VALIDATION_Q_SHARDS", "16-17"))
    parser.add_argument("--test-q-shards", default=os.environ.get("CELEBA_TEST_Q_SHARDS", "18-19"))
    parser.add_argument("--verify-ddim-shard-counts", action="store_true", default=env_bool("CELEBA_VERIFY_DDIM_SHARD_COUNTS", False))
    parser.add_argument("--require-cuda", action="store_true", default=env_bool("CELEBA_REQUIRE_CUDA", False))
    return parser


def main():
    args = build_parser().parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    os.chdir(repo_dir)
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("--require-cuda was set, but CUDA is not available.")

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / args.checkpoint_name
    report_path = output_dir / "training_report.md"
    loss_plot_path = output_dir / "training_validation_loss.png"

    transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize((64, 64), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )

    trainset = datasets.CelebA(root=args.data_root, split="train", transform=transform, download=False)
    valset = datasets.CelebA(root=args.data_root, split="valid", transform=transform, download=False)
    testset = datasets.CelebA(root=args.data_root, split="test", transform=transform, download=False)

    celeba_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    celeba_valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )
    celeba_testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    train_q_shards = parse_shards(args.train_q_shards)
    validation_q_shards = parse_shards(args.validation_q_shards)
    test_q_shards = parse_shards(args.test_q_shards)

    train_q_pool_n = ddim_shard_sample_count(args.ddim_data_dir, train_q_shards, args.verify_ddim_shard_counts)
    validation_q_pool_n = ddim_shard_sample_count(args.ddim_data_dir, validation_q_shards, args.verify_ddim_shard_counts)
    test_q_pool_n = ddim_shard_sample_count(args.ddim_data_dir, test_q_shards, args.verify_ddim_shard_counts)

    q_mixed_sampler = celeb.DDIMDiskSampler(
        real_loader=celeba_loader,
        ddim_data_dir=args.ddim_data_dir,
        gen_frac=0.5,
        shard_ids=train_q_shards,
    )
    q_val_sampler = celeb.DDIMDiskSampler(
        real_loader=celeba_valloader,
        ddim_data_dir=args.ddim_data_dir,
        gen_frac=0.5,
        shard_ids=validation_q_shards,
    )

    split_sample_sizes = pd.DataFrame(
        [
            {
                "phase": "training",
                "p_source": "CelebA train",
                "q_source": f"DDIM shards {args.train_q_shards}",
                "p_n": len(trainset),
                "q_n": train_q_pool_n,
                "q_pool_n": train_q_pool_n,
            },
            {
                "phase": "validation_loss",
                "p_source": "CelebA valid",
                "q_source": f"DDIM shards {args.validation_q_shards}",
                "p_n": len(valset),
                "q_n": validation_q_pool_n,
                "q_pool_n": validation_q_pool_n,
            },
            {
                "phase": "evaluation_test",
                "p_source": "CelebA test",
                "q_source": f"DDIM shards {args.test_q_shards}",
                "p_n": len(testset),
                "q_n": len(testset),
                "q_pool_n": test_q_pool_n,
            },
        ]
    )
    print("Independent split sample sizes")
    print(split_sample_sizes.to_string(index=False))

    x_mixed = q_mixed_sampler(batch_size=args.batch_size)
    sample_batch_shape = tuple(int(dim) for dim in x_mixed.shape)
    print(f"Sample mixture batch shape: {sample_batch_shape}")
    del x_mixed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    args.steps_per_epoch = len(celeba_loader)
    model_ddim, losses_ddim, val_losses = dre_batch.run_DRE_fdiv_cnn_minibatch_celeba64(
        p_loader=celeba_loader,
        q_sampler=q_mixed_sampler,
        num_epochs=args.num_epochs,
        val_loader=celeba_valloader,
        val_q_sampler=q_val_sampler,
        val_q_batches=args.val_q_batches,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        restore_best=True,
        restore_mode=args.restore_mode,
        return_val_losses=True,
        print_every=args.print_every,
    )

    torch.save(
        {
            "model_state": model_ddim.state_dict(),
            "losses": losses_ddim,
            "val_losses": val_losses,
            "config": vars(args),
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")

    loss_epoch_df = summarize_losses(losses_ddim, val_losses, args.steps_per_epoch)
    save_loss_plot(losses_ddim, val_losses, args.steps_per_epoch, loss_plot_path)
    rdr_full_summary_df, losses_ddim, val_losses = summarize_full_train_test_rdr(
        checkpoint_path=checkpoint_path,
        celeba_loader=celeba_loader,
        celeba_testloader=celeba_testloader,
        trainset=trainset,
        testset=testset,
        ddim_data_dir=args.ddim_data_dir,
        train_q_shards=train_q_shards,
        test_q_shards=test_q_shards,
        train_q_n=train_q_pool_n,
        test_q_n=len(testset),
        batch_size=args.batch_size,
    )
    loss_epoch_df = summarize_losses(losses_ddim, val_losses, args.steps_per_epoch)
    split_sample_sizes.to_csv(output_dir / "split_sample_sizes.csv", index=False)
    loss_epoch_df.to_csv(output_dir / "loss_epoch_summary.csv")
    rdr_full_summary_df.to_csv(output_dir / "rdr_full_summary.csv")
    write_markdown_report(
        report_path=report_path,
        args=args,
        split_sample_sizes=split_sample_sizes,
        loss_epoch_df=loss_epoch_df,
        rdr_full_summary_df=rdr_full_summary_df,
        checkpoint_path=checkpoint_path,
        loss_plot_path=loss_plot_path,
        sample_batch_shape=sample_batch_shape,
    )
    print(f"Saved Markdown report to {report_path}")


if __name__ == "__main__":
    main()
