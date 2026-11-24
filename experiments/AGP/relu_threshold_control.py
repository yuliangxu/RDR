"""Published AGP sigmoid MLP fitted to square-root abundances and binary presences."""

import math
import time

import numpy as np
import torch
from torch import nn

from experiments.AGP import mask_metrics, selected_workflow as published
from utils.model_selection import balanced_brier


def control_features(raw):
    """Return square-root relative abundance and a binary presence indicator for each taxon."""
    raw = np.asarray(raw)
    if raw.ndim == 2 and len(raw) == 1:
        # The archived helper validates a bank of two or more rows. Its
        # transform is strictly rowwise, including for a singleton query.
        x = mask_metrics.close(np.repeat(raw, 2, axis=0))[:1]
    else:
        x = mask_metrics.close(raw)
    return np.column_stack((np.sqrt(x), x > 0)).astype(np.float32)


class CompositionControl(nn.Module):
    def __init__(self, n_taxa, width, blocks, output_alpha):
        super().__init__()
        if not math.isfinite(output_alpha) or output_alpha <= 0:
            raise ValueError("The published sigmoid slope must be finite and positive")
        self.mlp = published.DeepLogitMLP(2 * n_taxa, width, blocks)
        self.register_buffer("output_alpha", torch.tensor(output_alpha, dtype=torch.float64))

    def logits(self, features):
        return self.output_alpha * self.mlp(features)

    def forward(self, features):
        return 2 * torch.sigmoid(self.output_alpha * self.mlp(features).double())


def make_control(n_taxa, width=128, blocks=3, output_alpha=1.0):
    return CompositionControl(n_taxa, width, blocks, output_alpha)


@torch.no_grad()
def predict_control(model, raw, device):
    """Predict RDR for raw, unclosed rows in the supplied order."""
    model.eval()
    scores = []
    for start in range(0, len(raw), 4096):
        features = control_features(raw[start:start + 4096])
        scores.append(model(torch.as_tensor(features, device=device)).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float64)


@torch.no_grad()
def _stopping_brier(model, p, q):
    model.eval()
    return balanced_brier(model(p).cpu().numpy(), model(q).cpu().numpy())


def fit_control(raw, cfg, task, device):
    """Use the archived final-fit optimizer, selected risk/slope, and stopping rule.

    Each optimizer step uses all training observations with equal source weight.
    Only train and stopping arrays are accessed. Features have no fitted state.
    Returns the restored model, per-epoch history, and fit metadata.
    """
    train = torch.as_tensor(np.concatenate([
        control_features(raw["train_p"]), control_features(raw["train_q"])
    ]), device=device)
    n_p = len(raw["train_p"])
    stop_p = torch.as_tensor(control_features(raw["stopping_p"]), device=device)
    stop_q = torch.as_tensor(control_features(raw["stopping_q"]), device=device)
    seed = published.seed_for(cfg, task["repeat"], "final")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = make_control(raw["train_p"].shape[1], cfg["hidden_dimension"],
                             cfg["blocks"], task["output_alpha"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=cfg["plateau_patience"], cooldown=cfg["plateau_cooldown"])
    best, best_epoch, best_weights, stale = float("inf"), None, None, 0
    history = []
    started = time.monotonic()
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk, backward, shift = published.stable_risk(model.logits(train), n_p, task["loss"])
        if not torch.isfinite(risk).item():
            raise FloatingPointError(f"Nonfinite control training risk at epoch {epoch}")
        backward.backward()
        norm = published.clip_scaled_gradients(model.parameters(), shift, cfg["clip_max_norm"])
        optimizer.step()
        stopping = _stopping_brier(model, stop_p, stop_q)
        if not math.isfinite(stopping):
            raise FloatingPointError(f"Nonfinite control stopping Brier at epoch {epoch}")
        if stopping < best:
            best, best_epoch, stale = stopping, epoch, 0
            best_weights = {name: w.detach().cpu().clone() for name, w in model.state_dict().items()}
        else:
            stale += 1
        history.append(dict(epoch=epoch, training_objective=float(risk.detach()),
                            stopping_brier=stopping, learning_rate=optimizer.param_groups[0]["lr"],
                            scaled_gradient_norm=norm, gradient_log_scale=shift,
                            epochs_without_improvement=stale))
        scheduler.step(stopping)
        if epoch >= cfg["min_epochs"] and stale >= cfg["patience"]:
            break
    if best_weights is None:
        raise ValueError("Control requires at least one training epoch")
    model.load_state_dict(best_weights)
    restored = _stopping_brier(model, stop_p, stop_q)
    if not math.isclose(restored, best, rel_tol=0, abs_tol=1e-12):
        raise AssertionError("Control checkpoint did not restore the best stopping Brier")
    training = dict(best_epoch=best_epoch, epochs_run=len(history), stopping_brier=best,
                    restored_stopping_brier=restored, seed_initialization=seed,
                    stopping_reason="patience" if stale >= cfg["patience"] else "max_epochs",
                    train_p=n_p, train_q=len(train) - n_p, fit_seconds=time.monotonic() - started,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    loss=task["loss"], output_alpha=float(task["output_alpha"]),
                    features="sqrt_abundance_and_presence", feature_dimension=train.shape[1])
    return model, history, training
