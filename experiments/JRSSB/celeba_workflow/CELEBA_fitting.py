"""Primary/null model recipes, training-only scaling, fitting, and test inference.

``train`` loads only training/validation images or features. ``score`` restores
that checkpoint and scaler, then evaluates the held-out test rows separately.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from CELEBA_agent3 import atomic_csv, atomic_json, atomic_torch_save
from CELEBA_data import load_yaml, set_seed, sha256_file
from CELEBA_rdr import (
    RatioMLP,
    train_ratio_model,
    tensor_dataset,
    predict_ratio,
    estimate_midpoint_hellinger,
)
from .CELEBA_data import ManifestImages, feature_values
from .CELEBA_paths import HERE


def model_and_optimizer(level, config):
    """Build the original feature MLP or pixel CNN and its optimizer recipe."""
    if level == "feature":
        return (
            RatioMLP(
                2048, int(config["hidden_dimension"]), float(config["output_alpha"])
            ),
            None,
        )
    from utils.networks import RatioNetCelebA64

    model = RatioNetCelebA64(in_ch=3, ndf=int(config["ndf"]), log_scale=False)
    model.out_act.alpha = float(config["output_alpha"])

    def optimizer(network):
        return torch.optim.Adam(
            network.parameters(),
            lr=float(config["learning_rate"]),
            betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
            weight_decay=float(config["weight_decay"]),
        )

    return model, optimizer


def experiment(args, spec):
    """Resolve manifests, training settings, seed, and output path for one fit."""
    level, branch = args.level, args.branch
    if args.null:
        if branch == "real":
            if level != "pixel":
                raise ValueError("No real-feature null is reported")
            config = load_yaml(HERE / "configs/train.yaml")
            config = {**config, **config["pixel_rdr"]}
            seed = 21060824 + args.repeat
            locations = spec["null"]["real"]
        else:
            settings = load_yaml(HERE / f"configs/agent3_{level}_null.yaml")
            config = dict(settings[f"{level}_rdr"])
            seed = (
                int(settings["model_seed_base"])
                + (100 if branch == "upper" else 0)
                + args.repeat
            )
            locations = spec["null"][branch][str(args.repeat)]
        frames = {
            role: {f: pd.read_csv(locations[role][f]) for f in ("A", "B")}
            for role in ("train", "validation", "test")
        }
        target = args.work_root / f"models/null/{branch}/{level}/{args.repeat}"
    else:
        settings = load_yaml(HERE / "configs/agent3_real_generator_rdr.yaml")
        config = dict(settings[f"{level}_rdr"])
        seed = (
            int(settings["seed"])
            + (100 if level == "feature" else 200)
            + (1 if branch == "upper" else 0)
        )
        frames = {
            role: {
                "A": pd.read_csv(spec["primary"]["real"][role]),
                "B": pd.read_csv(spec["primary"][branch][role]),
            }
            for role in ("train", "validation", "test")
        }
        target = args.work_root / f"models/primary/{branch}/{level}"
    config.update(
        constant_ratio_validation_baseline=False, ratio_epsilon=1e-6, num_workers=0
    )
    if args.limit:
        frames = {
            role: {f: df.iloc[: args.limit].copy() for f, df in data.items()}
            for role, data in frames.items()
        }
        config.update(
            num_epochs=1, early_stop_min_epochs=1, batch_size=min(8, args.limit)
        )
    return frames, config, seed, target


def datasets(frames, args, spec, scaler=None):
    """Build role-specific inputs and fit feature scaling on training rows only."""
    if args.level == "pixel":
        return {
            role: {
                f: ManifestImages(df, args.work_root, args.fresh).preload()
                for f, df in data.items()
            }
            for role, data in frames.items()
        }, None
    values = {
        role: {f: feature_values(df, args, spec) for f, df in data.items()}
        for role, data in frames.items()
    }
    if scaler is None:
        scaler = StandardScaler()
        if args.null:
            # The reported generated nulls use the lower+upper PRIMARY TRAINING scaler.
            for branch in ("lower", "upper"):
                df = pd.read_csv(spec["primary"][branch]["train"])
                if args.limit:
                    df = df.iloc[: args.limit]
                scaler.partial_fit(feature_values(df, args, spec))
        else:
            for fold in ("A", "B"):
                scaler.partial_fit(values["train"][fold])
    result = {
        role: {
            f: tensor_dataset(scaler.transform(v).astype(np.float32))
            for f, v in data.items()
        }
        for role, data in values.items()
    }
    return result, scaler


def train(args, spec):
    """Fit on training rows, select by validation loss, and save checkpoint plus scaler."""
    frames, config, seed, target = experiment(args, spec)
    # Do not read test images/features during training or early stopping.
    set_seed(seed)
    data, scaler = datasets({r: frames[r] for r in ("train", "validation")}, args, spec)
    model, optimizer = model_and_optimizer(args.level, config)
    result = train_ratio_model(
        model,
        data["train"]["A"],
        data["train"]["B"],
        data["validation"]["A"],
        data["validation"]["B"],
        config,
        seed,
        args.device,
        optimizer_factory=optimizer,
    )
    atomic_torch_save(
        dict(
            model_state=result.model.state_dict(),
            config=config,
            model_seed=seed,
            best_epoch=result.best_epoch,
            best_validation_loss=result.best_validation_loss,
            scaler_mean=None if scaler is None else scaler.mean_,
            scaler_scale=None if scaler is None else scaler.scale_,
        ),
        target / "checkpoint.pt",
    )
    atomic_csv(result.history, target / "loss_history.csv")
    print(target / "checkpoint.pt", flush=True)


def score(args, spec):
    """Restore the selected model and write predictions for the complete test folds."""
    frames, config, seed, target = experiment(args, spec)
    path = args.checkpoint or target / "checkpoint.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        config = checkpoint["config"]
    model, _ = model_and_optimizer(args.level, config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(args.device).eval()
    scaler = None
    if args.level == "feature":
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(checkpoint["scaler_mean"])
        scaler.scale_ = np.asarray(checkpoint["scaler_scale"])
    data, _ = datasets({"test": frames["test"]}, args, spec, scaler)
    rows = []
    for fold in ("A", "B"):
        values = predict_ratio(
            model,
            data["test"][fold],
            int(config["evaluation_batch_size"]),
            0,
            args.device,
        )
        rows.append(frames["test"][fold].assign(fold=fold, rdr=values))
    atomic_csv(pd.concat(rows), target / "test_predictions.csv")
    p, q = [r.rdr.to_numpy() for r in rows]
    atomic_json(
        dict(
            checkpoint=str(path),
            checkpoint_sha256=sha256_file(path),
            **estimate_midpoint_hellinger(p, q, 1e-6),
        ),
        target / "test_metrics.json",
    )
    print(target / "test_metrics.json", flush=True)
