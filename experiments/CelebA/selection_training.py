"""Paired CelebA feature/pixel fits with independent Brier stopping data.

Only callers allocate observations to roles or authorize test access. Arrays
may be read-only memory maps; training copies only the current minibatch.
Feature standardization uses pooled training P and Q exclusively, and its
buffers travel with the checkpoint. Models emit raw logits throughout.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from utils.losses import midpoint_loss_from_logits
from utils.model_selection import balanced_brier, calibration_diagnostics
from utils.networks import MLP, RatioNetCelebA64


class FeatureLogitMLP(MLP):
    """Historical three-layer ReLU MLP with frozen training standardization."""

    def __init__(self, input_dimension=2048, hidden_dimension=512):
        super().__init__(input_dim=input_dimension, hidden_dim=hidden_dimension)
        self.model[-1] = nn.Identity()
        self.register_buffer("input_mean", torch.zeros(input_dimension))
        self.register_buffer("input_scale", torch.ones(input_dimension))
        self.representation = "feature"

    def forward(self, x):
        return super().forward((x - self.input_mean) / self.input_scale).flatten()


def seed_for(cfg, repeat, purpose):
    """Seeds deliberately exclude pair, architecture, loss, and output slope."""
    offsets = {"initialization": 1, "training_order_p": 2,
               "training_order_q": 3, "diagnostic_p": 4, "diagnostic_q": 5}
    if purpose not in offsets:
        raise ValueError(f"Unknown seed purpose: {purpose}")
    return (int(cfg.get("seed", 2026092901)) + 1009 * int(repeat) + offsets[purpose]) % (2**63 - 1)


def make_model(cfg, repeat, representation, architecture="baseline", device="cpu"):
    if representation not in ("feature", "pixel"):
        raise ValueError(f"Unknown representation: {representation}")
    if architecture not in ("baseline", "small"):
        raise ValueError(f"Unknown architecture: {architecture}")
    recipe = cfg.get(representation, {})
    # Construct on CPU to preserve paired initialization and caller RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed_for(cfg, repeat, "initialization"))
        if representation == "feature":
            width = recipe.get("hidden_dimension", 512) if architecture == "baseline" else recipe.get("small_hidden_dimension", 256)
            model = FeatureLogitMLP(recipe.get("input_dimension", 2048), width)
        else:
            width = recipe.get("ndf", 64) if architecture == "baseline" else recipe.get("small_ndf", 32)
            # log_scale=True installs Identity after the head: forward is the
            # raw z. It is NOT log(2*sigmoid(z)); slope is applied by our caller.
            model = RatioNetCelebA64(in_ch=3, ndf=width, log_scale=True)
            model.representation = "pixel"
    return model.to(device)


def paired_epoch_indices(n_p, n_q, batch_size, p_rng, q_rng):
    """Use every observation once, with nonempty P and Q in every step.

    Partition sizes differ by at most one within a source. For severely
    unequal source counts, the larger source can exceed batch_size so that
    the smaller source never needs recycling or dropping observations.
    """
    if min(n_p, n_q, batch_size) <= 0:
        raise ValueError("Source sizes and batch_size must be positive")
    steps = min(max(math.ceil(n_p / batch_size), math.ceil(n_q / batch_size)), n_p, n_q)
    p = torch.randperm(n_p, generator=p_rng).tensor_split(steps)
    q = torch.randperm(n_q, generator=q_rng).tensor_split(steps)
    return zip(p, q)


def _batch(values, indices, device, representation):
    if isinstance(values, torch.Tensor):
        tensor = values[indices].detach().to(device=device, dtype=torch.float32)
    else:
        # Copying avoids writable-buffer warnings for read-only cache arrays.
        tensor = torch.from_numpy(np.array(values[indices], copy=True)).to(device=device, dtype=torch.float32)
    if representation == "pixel":
        if values.dtype in (np.uint8, torch.uint8):
            tensor = tensor.div(127.5).sub(1)
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (3, 64, 64):
            raise ValueError(f"Pixel arrays must have shape (N,3,64,64), got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all() or tensor.min() < -1.00001 or tensor.max() > 1.00001:
            raise ValueError("Pixel input must be finite uint8 or model-space [-1,1]")
    elif tensor.ndim != 2 or not torch.isfinite(tensor).all():
        raise ValueError("Feature arrays must be finite matrices")
    return tensor


@torch.no_grad()
def predict(model, x, alpha, batch_size, device, representation=None):
    if len(x) == 0 or batch_size <= 0 or not math.isfinite(float(alpha)) or alpha <= 0:
        raise ValueError("Prediction requires nonempty data, positive batch size and finite positive slope")
    representation = representation or model.representation
    model.eval()
    predictions = []
    for start in range(0, len(x), batch_size):
        logits = model(_batch(x, slice(start, start + batch_size), device, representation))
        predictions.append((2 * torch.sigmoid(alpha * logits)).flatten().double().cpu().numpy())
    result = np.concatenate(predictions)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite predictions")
    return result


def fit_feature_scaler(model, p, q, batch_size=4096):
    """Streaming pooled P+Q population mean/std, fitted to training rows only."""
    count, mean, m2 = 0, None, None
    for values in (p, q):
        for start in range(0, len(values), batch_size):
            chunk = np.asarray(values[start:start + batch_size], dtype=np.float64)
            if chunk.ndim != 2 or not np.isfinite(chunk).all():
                raise ValueError("Scaler requires finite feature matrices")
            batch_mean = chunk.mean(axis=0)
            batch_m2 = np.square(chunk - batch_mean).sum(axis=0)
            if mean is None:
                count, mean, m2 = len(chunk), batch_mean, batch_m2
            else:
                delta = batch_mean - mean
                updated = count + len(chunk)
                m2 += batch_m2 + delta**2 * count * len(chunk) / updated
                mean += delta * len(chunk) / updated
                count = updated
    if not count:
        raise ValueError("Scaler requires nonempty training data")
    scale = np.sqrt(np.maximum(m2 / count, 0))
    scale[scale == 0] = 1
    model.input_mean.copy_(torch.as_tensor(mean, dtype=model.input_mean.dtype, device=model.input_mean.device))
    model.input_scale.copy_(torch.as_tensor(scale, dtype=model.input_scale.dtype, device=model.input_scale.device))
    return {"scope": "pooled_train_p_and_train_q", "n_p": len(p), "n_q": len(q),
            "variance_ddof": 0, "constant_feature_scale": 1.0}


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _require_arrays(arrays, roles):
    for role in roles:
        for side in ("p", "q"):
            name = f"{role}_{side}"
            if name not in arrays or len(arrays[name]) == 0:
                raise ValueError(f"Missing or empty array: {name}")


def _diagnostic_indices(cfg, recipe, repeat, side, size):
    limit = int(recipe.get("train_diagnostic_n", cfg.get("train_diagnostic_n", 2048)))
    if limit <= 0:
        raise ValueError("train_diagnostic_n must be positive")
    rng = np.random.default_rng(seed_for(cfg, repeat, "diagnostic_" + side))
    return np.sort(rng.choice(size, min(limit, size), replace=False))


def _evaluate_predictions(model, arrays, roles, alpha, batch_size, device):
    return {f"{role}_{side}": predict(model, arrays[f"{role}_{side}"], alpha, batch_size, device)
            for role in roles for side in ("p", "q")}


def _diagnostics(predictions, roles, cfg):
    return calibration_diagnostics(
        *[predictions[f"{role}_{side}"] for role in roles for side in ("p", "q")],
        alpha=cfg.get("ci_alpha", .05), bins=cfg.get("bins", 20))


def scaled_training_risk(logits, n_p, loss):
    """Return the exact risk and a scaled backward risk, without ratio clipping.

    For Hellinger, factor out its largest exponential before backward so
    the float32 network never receives enormous unscaled derivatives.
    Gradient clipping below accounts for this factor exactly.
    """
    if loss != 'hellinger':
        risk = midpoint_loss_from_logits(logits[:n_p], logits[n_p:], loss)
        return risk, risk, 0.0
    log_ratio = math.log(2) + torch.nn.functional.logsigmoid(logits.double())
    lp, lq = log_ratio[:n_p], log_ratio[n_p:]
    shift = max(0., float((-.5 * lp.detach()).max()))
    scaled = (.5 * torch.exp(-.5 * lp - shift).mean()
              + .25 * torch.exp(.5 * lp - shift).mean()
              + .25 * torch.exp(.5 * lq - shift).mean())
    risk = scaled.detach() * torch.exp(log_ratio.new_tensor(shift)) - 1
    return risk, scaled, shift


@torch.no_grad()
def clip_scaled_gradients(parameters, log_scale, max_norm=1.):
    """Clip the true gradient using a float64 norm of scaled float32 grads."""
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients or max_norm <= 0 or not math.isfinite(log_scale):
        raise ValueError('Invalid scaled gradient clipping inputs')
    norms = torch.stack([torch.linalg.vector_norm(g.double()) for g in gradients])
    norm = float(torch.linalg.vector_norm(norms))
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient entries (float64 norm)')
    if norm == 0:
        return
    log_factor = min(log_scale, math.log(max_norm) - math.log(norm))
    factor = math.exp(log_factor)
    for gradient in gradients:
        if factor > 1e30:
            gradient.copy_((gradient.double() * factor).to(gradient.dtype))
        else:
            gradient.mul_(factor)


def fit_model(task, cfg, arrays, output_dir, device):
    """Fit on train, restore absolute-minimum earlystop Brier, then select.

    Returns the metrics dictionary and writes model.pt, predictions.npz,
    history.json, cells.json, metrics.json. Caller owns integrity/completion
    markers. No test role is read here.
    """
    _require_arrays(arrays, ("train", "earlystop", "selection_calibration", "selection_evaluation"))
    output_dir, device = Path(output_dir), torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    representation, repeat = task["representation"], int(task["repeat"])
    recipe = cfg.get(representation, {})
    alpha = float(task["output_alpha"])
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Output slope must be finite and positive")
    model = make_model(cfg, repeat, representation, task.get("architecture", "baseline"), device)
    n_p, n_q = len(arrays["train_p"]), len(arrays["train_q"])
    batch_size = int(recipe.get("batch_size", 128))
    eval_batch_size = int(recipe.get("evaluation_batch_size", 4096 if representation == "feature" else 512))
    max_epochs, min_epochs, patience = [int(recipe.get(key, default)) for key, default in
                                      (("max_epochs", 50), ("min_epochs", 1), ("patience", 5))]
    if batch_size <= 0 or eval_batch_size <= 0 or max_epochs <= 0 or not 1 <= min_epochs <= max_epochs or patience <= 0:
        raise ValueError("Invalid minibatch, epoch, or patience configuration")
    steps_per_epoch = min(max(math.ceil(n_p / batch_size), math.ceil(n_q / batch_size)), n_p, n_q)
    scaler = None
    effective_onecycle_pct_start = None
    if representation == "feature":
        scaler = fit_feature_scaler(model, arrays["train_p"], arrays["train_q"], eval_batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.get("learning_rate", 5e-4),
                                      weight_decay=recipe.get("weight_decay", .01))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=recipe.get("plateau_factor", .5),
            patience=recipe.get("plateau_patience", 5), cooldown=recipe.get("plateau_cooldown", 2))
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=recipe.get("learning_rate", 2e-4),
                                     betas=(.9, .999), weight_decay=recipe.get("weight_decay", 0))
        total_steps = max_epochs * steps_per_epoch
        effective_onecycle_pct_start = float(recipe.get("onecycle_pct_start", .1))
        if effective_onecycle_pct_start * total_steps == 1:
            # PyTorch's inclusive warmup boundary divides by zero for this
            # tiny schedule (e.g. ten-step smoke and pct_start=.1). Use two
            # warmup steps only in this degenerate case; record the change.
            effective_onecycle_pct_start = 2 / total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=recipe.get("max_lr", 6e-4), total_steps=total_steps,
            pct_start=effective_onecycle_pct_start, anneal_strategy="cos",
            div_factor=recipe.get("onecycle_div_factor", 10),
            final_div_factor=recipe.get("onecycle_final_div_factor", 1000))
    scheduled_initial_lr = optimizer.param_groups[0]["lr"]
    p_rng = torch.Generator().manual_seed(seed_for(cfg, repeat, "training_order_p"))
    q_rng = torch.Generator().manual_seed(seed_for(cfg, repeat, "training_order_q"))
    train_diag = {side: arrays["train_" + side][_diagnostic_indices(cfg, recipe, repeat, side, n)]
                  for side, n in (("p", n_p), ("q", n_q))}
    history, best_state, best_epoch, best, stale = [], None, None, float("inf"), 0
    started = time.monotonic()
    for epoch in range(max_epochs):
        model.train()
        freeze_epoch = int(recipe.get("bn_freeze_epoch", 5))
        if representation == "pixel" and freeze_epoch >= 0 and epoch >= freeze_epoch:
            for module in model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
                    module.requires_grad_(False)
        objective_sum, observed_p, observed_q = 0., 0, 0
        for pi, qi in paired_epoch_indices(n_p, n_q, batch_size, p_rng, q_rng):
            p = _batch(arrays["train_p"], pi.numpy(), device, representation)
            q = _batch(arrays["train_q"], qi.numpy(), device, representation)
            logits = alpha * model(torch.cat((p, q), dim=0))
            scaled_gradients = cfg.get('numerical_gradient_mode') == 'scaled_hellinger_float64_norm'
            if scaled_gradients:
                risk, backward_risk, log_scale = scaled_training_risk(logits, len(p), task['loss'])
            else:
                risk = midpoint_loss_from_logits(logits[:len(p)], logits[len(p):], task["loss"])
                backward_risk = risk
            if not torch.isfinite(risk):
                raise FloatingPointError(f"Nonfinite objective at epoch {epoch + 1}")
            optimizer.zero_grad(set_to_none=True)
            backward_risk.backward()
            if scaled_gradients:
                clip_scaled_gradients(model.parameters(), log_scale, recipe.get('clip_max_norm', 1.))
            else:
                nn.utils.clip_grad_norm_(model.parameters(), recipe.get("clip_max_norm", 1.), error_if_nonfinite=True)
            optimizer.step()
            if representation == "pixel":
                scheduler.step()
            objective_sum += float(risk.detach())
            observed_p += len(p)
            observed_q += len(q)
        if (observed_p, observed_q) != (n_p, n_q):
            raise AssertionError("Training epoch dropped or repeated observations")
        early = _evaluate_predictions(model, arrays, ("earlystop",), alpha, eval_batch_size, device)
        earlystop_brier = balanced_brier(early["earlystop_p"], early["earlystop_q"])
        train_brier = balanced_brier(*[predict(model, train_diag[side], alpha, eval_batch_size, device) for side in ("p", "q")])
        if earlystop_brier < best:
            best, best_epoch, stale = earlystop_brier, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if representation == "feature":
            scheduler.step(earlystop_brier)
        history.append({"epoch": epoch + 1, "training_objective": objective_sum / steps_per_epoch,
                        "training_objective_aggregation": "mean_of_minibatch_objectives",
                        "train_brier": train_brier, "earlystop_brier": earlystop_brier,
                        "train_p_used": observed_p, "train_q_used": observed_q,
                        "lr": optimizer.param_groups[0]["lr"], "epochs_without_improvement": stale})
        print(f"{task.get('candidate', task['loss'])} {task.get('branch', '')} repeat={repeat} "
              f"epoch={epoch + 1} train_BS={train_brier:.7f} earlystop_BS={earlystop_brier:.7f}", flush=True)
        if epoch + 1 >= min_epochs and stale >= patience:
            break
    fit_seconds = time.monotonic() - started
    model.load_state_dict(best_state)
    restored = balanced_brier(*[predict(model, arrays["earlystop_" + side], alpha, eval_batch_size, device) for side in ("p", "q")])
    if not math.isclose(restored, best, rel_tol=0, abs_tol=1e-10):
        raise AssertionError("Absolute-minimum earlystop-Brier checkpoint was not restored")
    roles = ("selection_calibration", "selection_evaluation")
    predictions = _evaluate_predictions(model, arrays, roles, alpha, eval_batch_size, device)
    summary, cells = _diagnostics(predictions, roles, cfg)
    metrics = {**task, **summary,
               "numerical_gradient_mode": cfg.get('numerical_gradient_mode', 'legacy_float32_norm'),
               "brier": balanced_brier(predictions["selection_evaluation_p"], predictions["selection_evaluation_q"]),
               "best_epoch": best_epoch, "epochs": len(history), "earlystop_brier": best,
               "restored_earlystop_brier": restored, "fit_seconds": fit_seconds,
               "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "train_p_per_epoch": n_p, "train_q_per_epoch": n_q,
               "train_diagnostic_p": len(train_diag["p"]), "train_diagnostic_q": len(train_diag["q"]),
               "training_diagnostic_scope": "fixed_training_subset_eval_mode",
               "stopping_reason": "patience" if stale >= patience and len(history) >= min_epochs else "max_epochs",
               "reached_max_epochs": len(history) == max_epochs,
               "best_epoch_at_boundary": best_epoch == max_epochs,
               "parameter_count": sum(value.numel() for value in model.parameters()),
               "scaler": scaler, "optimizer": "AdamW" if representation == "feature" else "Adam",
               "scheduler": "ReduceLROnPlateau_earlystop_Brier" if representation == "feature" else "OneCycleLR",
               "scheduled_initial_lr": scheduled_initial_lr,
               "effective_onecycle_pct_start": effective_onecycle_pct_start,
               "bn_freeze_epoch": int(recipe.get("bn_freeze_epoch", 5)) if representation == "pixel" else None,
               "bn_freeze_affine": representation == "pixel",
               "seed_initialization": seed_for(cfg, repeat, "initialization"),
               "seed_order_p": seed_for(cfg, repeat, "training_order_p"),
               "seed_order_q": seed_for(cfg, repeat, "training_order_q"),
               "role_counts": {name: len(arrays[name]) for name in predictions}}
    torch.save({"model": best_state, "task": task, "config": cfg, "best_epoch": best_epoch,
                "earlystop_brier": best, "scaler": scaler, "schema": 1}, output_dir / "model.pt")
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "cells.json", cells)
    _write(output_dir / "history.json", history)
    return metrics


def evaluate_model(task, cfg, arrays, checkpoint_dir, output_dir, device):
    """Evaluate an authorized frozen fit on retrospective test sample roles."""
    roles = ("test_calibration", "test_evaluation")
    _require_arrays(arrays, roles)
    device, output_dir = torch.device(device), Path(output_dir)
    checkpoint = torch.load(Path(checkpoint_dir) / "model.pt", map_location="cpu", weights_only=True)
    identity_fields = ("representation", "branch", "architecture", "candidate", "loss", "output_alpha", "repeat")
    if any(checkpoint["task"].get(key) != task.get(key) for key in identity_fields) or checkpoint["config"] != cfg:
        raise ValueError("Checkpoint task or configuration differs from frozen evaluation request")
    model = make_model(cfg, task["repeat"], task["representation"], task.get("architecture", "baseline"), device)
    model.load_state_dict(checkpoint["model"], strict=True)
    recipe = cfg.get(task["representation"], {})
    batch_size = int(recipe.get("evaluation_batch_size", 4096 if task["representation"] == "feature" else 512))
    predictions = _evaluate_predictions(model, arrays, roles, task["output_alpha"], batch_size, device)
    summary, cells = _diagnostics(predictions, roles, cfg)
    evaluation_brier = balanced_brier(predictions["test_evaluation_p"], predictions["test_evaluation_q"])
    whole_brier = balanced_brier(*[np.concatenate([predictions[f"{role}_{side}"] for role in roles]) for side in ("p", "q")])
    metrics = {**task, **summary, "brier": evaluation_brier, "evaluation_brier": evaluation_brier,
               "whole_test_brier": whole_brier, "best_epoch": checkpoint["best_epoch"],
               "earlystop_brier": checkpoint["earlystop_brier"], "scaler": checkpoint["scaler"],
               "assessment_scope": "retrospective_previously_inspected_test_pool",
               "role_counts": {name: len(arrays[name]) for name in predictions}}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "predictions.npz", **predictions)
    _write(output_dir / "metrics.json", metrics)
    _write(output_dir / "cells.json", cells)
    return metrics
