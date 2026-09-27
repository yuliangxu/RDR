"""Shared numerical and data code for the final Agent 3 workflow."""

from __future__ import annotations

import copy

import math

from dataclasses import dataclass

from typing import Callable, Mapping, Optional

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, TensorDataset

from utils.networks import RatioMLP
from utils.losses import original_hellinger_midpoint_loss



def estimate_midpoint_hellinger(
    p_scores: np.ndarray,
    q_scores: np.ndarray,
    epsilon: float,
) -> dict[str, float]:
    """Evaluate the held-out Hellinger objective for r=dP/d((P+Q)/2).

    The negative objective is the primary variational estimate. The plug-in
    identity is also returned as a ratio-calibration diagnostic; the two agree
    at the population-optimal density ratio but need not agree for a fitted
    network.
    """
    p_values = np.asarray(p_scores, dtype=np.float64).reshape(-1)
    q_values = np.asarray(q_scores, dtype=np.float64).reshape(-1)
    if len(p_values) == 0 or len(q_values) == 0:
        raise ValueError("Midpoint Hellinger evaluation requires non-empty P and Q scores.")
    if not np.isfinite(p_values).all() or not np.isfinite(q_values).all():
        raise ValueError("Midpoint Hellinger scores must be finite.")
    p_clipped = np.clip(p_values, float(epsilon), 2.0 - float(epsilon))
    q_clipped = np.clip(q_values, float(epsilon), 2.0 - float(epsilon))
    p_inverse_component = 0.5 * float(np.mean(np.power(p_clipped, -0.5)))
    p_sqrt_component = 0.25 * float(np.mean(np.sqrt(p_clipped)))
    q_sqrt_component = 0.25 * float(np.mean(np.sqrt(q_clipped)))
    objective = p_inverse_component + p_sqrt_component + q_sqrt_component - 1.0
    variational = -objective
    plugin = 1.0 - 0.5 * (
        float(np.mean(np.sqrt(p_clipped)))
        + float(np.mean(np.sqrt(q_clipped)))
    )
    return {
        "heldout_objective": objective,
        "variational_lower_bound": variational,
        "plugin_midpoint_hellinger": plugin,
        "plugin_minus_variational": plugin - variational,
        "p_inverse_component": p_inverse_component,
        "p_sqrt_component": p_sqrt_component,
        "q_sqrt_component": q_sqrt_component,
        "p_clip_rate": float(np.mean(p_values != p_clipped)),
        "q_clip_rate": float(np.mean(q_values != q_clipped)),
        "n_p": int(len(p_values)),
        "n_q": int(len(q_values)),
    }

def _batch_tensor(batch) -> torch.Tensor:
    return batch[0] if isinstance(batch, (tuple, list)) else batch

def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=False,
    )

@torch.inference_mode()
def validation_loss(
    model: nn.Module,
    p_loader: DataLoader,
    q_loader: DataLoader,
    device: torch.device,
    epsilon: float,
) -> float:
    model.eval()
    p_inverse_sum = 0.0
    p_sqrt_sum = 0.0
    q_sqrt_sum = 0.0
    p_count = 0
    q_count = 0
    for loader, source in ((p_loader, "p"), (q_loader, "q")):
        for batch in loader:
            values = _batch_tensor(batch).to(device=device, dtype=torch.float32, non_blocking=True)
            ratio = model(values).reshape(-1).clamp(epsilon, 2.0 - epsilon)
            if source == "p":
                p_inverse_sum += float(ratio.rsqrt().sum())
                p_sqrt_sum += float(ratio.sqrt().sum())
                p_count += len(ratio)
            else:
                q_sqrt_sum += float(ratio.sqrt().sum())
                q_count += len(ratio)
    if p_count == 0 or q_count == 0:
        raise ValueError("Validation loaders must be non-empty.")
    return 0.5 * p_inverse_sum / p_count + 0.25 * p_sqrt_sum / p_count + 0.25 * q_sqrt_sum / q_count - 1.0

@dataclass
class TrainResult:
    model: nn.Module
    history: pd.DataFrame
    best_epoch: int
    best_validation_loss: float
    restore_mode: str
    stop_reason: str

def _constant_ratio_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a compatible state whose bounded ratio output is identically one."""
    state = copy.deepcopy(model.state_dict())
    for name, _ in model.named_parameters():
        state[name].zero_()
    return state

def train_ratio_model(
    model: nn.Module,
    p_train: Dataset,
    q_train: Dataset,
    p_validation: Dataset,
    q_validation: Dataset,
    config: Mapping,
    seed: int,
    device: torch.device,
    optimizer_factory: Optional[Callable[[nn.Module], torch.optim.Optimizer]] = None,
) -> TrainResult:
    batch_size = int(config["batch_size"])
    num_workers = int(config.get("num_workers", 0))
    p_train_loader = make_loader(p_train, batch_size, True, seed + 1, num_workers)
    q_train_loader = make_loader(q_train, batch_size, True, seed + 2, num_workers)
    p_val_loader = make_loader(p_validation, int(config.get("evaluation_batch_size", batch_size)), False, seed + 3, num_workers)
    q_val_loader = make_loader(q_validation, int(config.get("evaluation_batch_size", batch_size)), False, seed + 4, num_workers)
    model = model.to(device=device, dtype=torch.float32)
    optimizer = (
        optimizer_factory(model)
        if optimizer_factory is not None
        else torch.optim.AdamW(
            model.parameters(),
            lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    )
    scheduler_name = str(config.get("scheduler", "plateau"))
    if scheduler_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(config["onecycle_max_lr"]),
            total_steps=int(config["num_epochs"]) * len(p_train_loader),
            pct_start=float(config.get("onecycle_pct_start", 0.1)),
            div_factor=float(config.get("onecycle_div_factor", 10.0)),
            final_div_factor=float(config.get("onecycle_final_div_factor", 1000.0)),
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("scheduler_factor", 0.5)),
            patience=int(config.get("scheduler_patience", 2)),
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    epsilon = float(config.get("ratio_epsilon", 1e-6))
    use_constant_baseline = bool(
        config.get("constant_ratio_validation_baseline", False)
    )
    best_state = _constant_ratio_state_dict(model) if use_constant_baseline else None
    last_finite_state = copy.deepcopy(model.state_dict())
    best_loss = 0.0 if use_constant_baseline else math.inf
    best_epoch = 0
    without_improvement = 0
    rows = []
    stop_reason = "completed_all_epochs"
    if use_constant_baseline:
        print(
            "validation baseline: epoch=000 constant_ratio=1 loss=0.000000",
            flush=True,
        )
    for epoch in range(1, int(config["num_epochs"]) + 1):
        model.train()
        freeze_epoch = int(config.get("bn_freeze_epoch", -1))
        if freeze_epoch >= 0 and epoch > freeze_epoch:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
                    for parameter in module.parameters():
                        parameter.requires_grad_(False)
        q_iterator = iter(q_train_loader)
        losses = []
        nonfinite_training = False
        for p_batch in p_train_loader:
            try:
                q_batch = next(q_iterator)
            except StopIteration:
                q_iterator = iter(q_train_loader)
                q_batch = next(q_iterator)
            p_values = _batch_tensor(p_batch).to(device=device, dtype=torch.float32, non_blocking=True)
            q_values = _batch_tensor(q_batch).to(device=device, dtype=torch.float32, non_blocking=True)
            size = min(len(p_values), len(q_values))
            p_values = p_values[:size]
            q_values = q_values[:size]
            optimizer.zero_grad(set_to_none=True)
            loss = original_hellinger_midpoint_loss(model, p_values, q_values, epsilon=epsilon)
            if not torch.isfinite(loss):
                stop_reason = f"nonfinite_training_loss_epoch_{epoch}"
                nonfinite_training = True
                print(f"Stopping: {stop_reason}", flush=True)
                break
            # This is the latest network known to have produced a finite loss.
            last_finite_state = copy.deepcopy(model.state_dict())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("gradient_clip_norm", 1.0)))
            optimizer.step()
            if scheduler_name == "onecycle":
                scheduler.step()
            losses.append(float(loss.detach()))
        if nonfinite_training:
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)) if losses else np.nan,
                    "validation_loss": np.nan,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "improved": False,
                }
            )
            break
        val_loss = validation_loss(model, p_val_loader, q_val_loader, device, epsilon)
        if not np.isfinite(val_loss):
            stop_reason = f"nonfinite_validation_loss_epoch_{epoch}"
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "validation_loss": np.nan,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "improved": False,
                }
            )
            print(f"Stopping: {stop_reason}", flush=True)
            break
        if scheduler_name == "plateau":
            scheduler.step(val_loss)
        improved = val_loss < best_loss - float(config.get("early_stop_min_delta", 0.0))
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            without_improvement = 0
        else:
            without_improvement += 1
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_loss": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "improved": improved,
            }
        )
        print(
            f"epoch={epoch:03d} train={rows[-1]['train_loss']:.6f} "
            f"validation={val_loss:.6f} best={best_loss:.6f}",
            flush=True,
        )
        if (
            epoch >= int(config.get("early_stop_min_epochs", 1))
            and without_improvement >= int(config["early_stop_patience"])
        ):
            stop_reason = f"early_stopping_patience_{int(config['early_stop_patience'])}"
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        restore_mode = (
            "constant_ratio_validation_baseline"
            if use_constant_baseline and best_epoch == 0
            else "best_validation"
        )
    else:
        model.load_state_dict(last_finite_state)
        restore_mode = "last_finite_train"
    return TrainResult(
        model=model.eval(),
        history=pd.DataFrame(rows),
        best_epoch=best_epoch,
        best_validation_loss=float(best_loss),
        restore_mode=restore_mode,
        stop_reason=stop_reason,
    )

@torch.inference_mode()
def predict_ratio(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> np.ndarray:
    loader = make_loader(dataset, batch_size, False, 0, num_workers)
    values = []
    model.eval()
    for batch in loader:
        tensor = _batch_tensor(batch).to(device=device, dtype=torch.float32, non_blocking=True)
        values.append(model(tensor).reshape(-1).detach().cpu())
    return torch.cat(values).numpy()

def phi_value(name: str, ratio: np.ndarray) -> np.ndarray:
    if name == "hellinger":
        return 1.0 - np.sqrt(ratio)
    if name == "kl":
        return ratio * np.log(ratio)
    if name == "reverse_kl":
        return -np.log(ratio)
    if name == "chi_square":
        return (ratio - 1.0) ** 2
    raise ValueError(f"Unknown phi divergence: {name}")

def estimate_phi_divergences(
    p_scores: np.ndarray,
    q_scores: np.ndarray,
    epsilon: float,
    names=("hellinger", "kl", "reverse_kl", "chi_square"),
) -> pd.DataFrame:
    p_scores = np.asarray(p_scores, dtype=np.float64)
    q_scores = np.asarray(q_scores, dtype=np.float64)
    all_scores = np.concatenate([p_scores, q_scores])
    clipped = np.clip(all_scores, epsilon, 2.0 - epsilon)
    density_ratio = clipped / (2.0 - clipped)
    midpoint_ratio_mean = float(0.5 * (clipped[: len(p_scores)].mean() + clipped[len(p_scores) :].mean()))
    rows = []
    for name in names:
        general_integrand = (2.0 - clipped) * phi_value(name, density_ratio)
        raw_general = float(
            0.5
            * (
                general_integrand[: len(p_scores)].mean()
                + general_integrand[len(p_scores) :].mean()
            )
        )
        if name == "hellinger":
            integrand = 1.0 - np.sqrt(clipped * (2.0 - clipped))
        else:
            integrand = general_integrand
        p_term = float(integrand[: len(p_scores)].mean())
        q_term = float(integrand[len(p_scores) :].mean())
        rows.append(
            {
                "divergence": name,
                "estimate": float(0.5 * (p_term + q_term)),
                "p_midpoint_component": float(0.5 * p_term),
                "q_midpoint_component": float(0.5 * q_term),
                "raw_general_phi_estimate": raw_general,
                "midpoint_ratio_mean": midpoint_ratio_mean,
                "clip_rate": float(np.mean(all_scores != clipped)),
                "n_p": len(p_scores),
                "n_q": len(q_scores),
            }
        )
    return pd.DataFrame(rows)

def _probability_label(value: float) -> str:
    return f"q{int(round(100 * float(value))):02d}"

def summarize_ratio_scores(
    p_scores: np.ndarray,
    q_scores: np.ndarray,
    tolerances=(0.05, 0.10, 0.20),
    quantiles=(0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99),
) -> pd.DataFrame:
    """Summarize held-out midpoint-ratio scores around their null target one."""
    groups = (
        ("fold_A", np.asarray(p_scores, dtype=np.float64)),
        ("fold_B", np.asarray(q_scores, dtype=np.float64)),
        (
            "combined",
            np.concatenate(
                [np.asarray(p_scores, dtype=np.float64), np.asarray(q_scores, dtype=np.float64)]
            ),
        ),
    )
    rows = []
    for group, values in groups:
        if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"Invalid ratio scores for {group}.")
        deviations = values - 1.0
        row = {
            "group": group,
            "n": len(values),
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean_absolute_deviation_from_one": float(np.mean(np.abs(deviations))),
            "rmse_from_one": float(np.sqrt(np.mean(deviations**2))),
        }
        row.update(
            {
                _probability_label(probability): float(np.quantile(values, probability))
                for probability in quantiles
            }
        )
        for tolerance in tolerances:
            suffix = f"{float(tolerance):.2f}".replace(".", "_")
            row[f"fraction_within_{suffix}"] = float(
                np.mean(np.abs(deviations) <= float(tolerance))
            )
        rows.append(row)
    return pd.DataFrame(rows)

def tensor_dataset(features: np.ndarray) -> TensorDataset:
    return TensorDataset(torch.as_tensor(features, dtype=torch.float32))
