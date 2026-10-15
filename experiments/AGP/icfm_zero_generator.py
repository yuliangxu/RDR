#!/usr/bin/env python3
"""Mask-conditional AGP ICFM on the positive face of each composition.

The caller supplies dependent masks from the frozen full-template MIDASim fit.
This is a newly trained conditional abundance flow, not post-hoc masking of an
unconditional generator. No pseudocounts or abundance cutoffs are introduced.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch
from torch import nn

if __package__:
    from . import icfm_corrected_generator as shared
else:
    _spec = importlib.util.spec_from_file_location(
        "_agp_zero_shared_generator", Path(__file__).with_name("icfm_corrected_generator.py"))
    shared = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(shared)


DEFAULT_MODEL = dict(kind="masked_clr_residual_mlp", dim=614, width=1024, blocks=4,
                     time_features=128, time_width=512, dropout=0.1)
DEFAULT_TRAINING = dict(shared.DEFAULT_TRAINING, seed=2026100701)


def _masks(masks, shape=None):
    masks = np.asarray(masks)
    if masks.ndim != 2 or not len(masks) or masks.dtype.kind not in "biuf":
        raise ValueError("Presence masks must be a nonempty binary matrix")
    if not np.isfinite(masks).all() or not np.all((masks == 0) | (masks == 1)):
        raise ValueError("Presence masks must contain only zero and one")
    if shape is not None and masks.shape != shape:
        raise ValueError("Mask and abundance/latent dimensions differ")
    result = masks.astype(bool, copy=False)
    if np.any(result.sum(1) == 0):
        raise ValueError("All-zero presence masks are outside the conditional composition model")
    return result


def encode_masked_clr(raw):
    """Return present-taxon centered log abundances and their exact observed mask."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not len(raw) or not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("Observed abundances must be a finite nonnegative nonempty matrix")
    masks = _masks(raw > 0, raw.shape)
    logs = np.zeros_like(raw)
    logs[masks] = np.log(raw[masks])
    centers = logs.sum(1, keepdims=True) / masks.sum(1, keepdims=True)
    latent = np.where(masks, logs - centers, 0.0)
    if not np.isfinite(latent).all():
        raise FloatingPointError("Invalid positive-abundance log coordinates")
    return latent, masks


def decode_masked_clr(latent, masks):
    """Float64 masked softmax, with strict positive support and exact absent zeros.

    Underflow is an error rather than a reason to clip or silently lose a taxon.
    """
    latent = np.asarray(latent, dtype=np.float64)
    masks = _masks(masks, latent.shape)
    if latent.ndim != 2 or not np.isfinite(latent).all():
        raise ValueError("Generated masked log coordinates must be finite")
    if np.any(latent[~masks] != 0):
        raise ValueError("Absent taxa must have exactly zero masked log coordinates")
    row_max = np.where(masks, latent, -np.inf).max(1, keepdims=True)
    weights = np.zeros_like(latent)
    shifted = latent - row_max
    with np.errstate(under="ignore"):
        weights[masks] = np.exp(shifted[masks])
    if np.any(weights[masks] <= 0) or not np.isfinite(weights).all():
        raise FloatingPointError("Masked softmax underflowed on a present taxon; no clipping was applied")
    result = weights / weights.sum(1, keepdims=True)
    if np.any(result[masks] <= 0) or np.any(result[~masks] != 0):
        raise FloatingPointError("Masked composition lost its declared support")
    if not np.allclose(result.sum(1), 1.0, atol=2e-15, rtol=0):
        raise FloatingPointError("Masked composition failed closure")
    return result


def project_masked(values, masks):
    """Project rows onto the zero-sum subspace on their declared positive taxa."""
    if values.ndim != 2 or masks.shape != values.shape:
        raise ValueError("Projection requires equally shaped latent and mask matrices")
    masks = masks.to(values)
    counts = masks.sum(1, keepdim=True)
    if bool(torch.any(counts <= 0)):
        raise ValueError("Cannot project onto an empty support")
    means = (values * masks).sum(1, keepdim=True) / counts
    return masks * (values - means)


class MaskConditionalVelocity(nn.Module):
    """A full-width velocity conditioned on the same mask used throughout its ODE."""
    def __init__(self, dim=614, width=1024, blocks=4, time_features=128,
                 time_width=512, dropout=0.1):
        super().__init__()
        if width < dim or blocks < 1 or time_features < 2:
            raise ValueError("Conditional velocity needs full data width and positive block count")
        self.dim, self.width, self.time_features = dim, width, time_features
        self.input_projection = nn.Linear(2 * dim, width)
        self.time_embed = nn.Sequential(nn.Linear(time_features, time_width), nn.SiLU(),
                                        nn.Linear(time_width, time_width))
        self.blocks = nn.ModuleList([
            shared.FullWidthBlock(width, time_width, dropout) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, dim))
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, state_and_time, masks):
        if state_and_time.ndim != 2 or state_and_time.shape[1] != self.dim + 1:
            raise ValueError("Expected masked log coordinates followed by scalar time")
        if masks.shape != (len(state_and_time), self.dim):
            raise ValueError("Each conditional velocity row requires its own full presence mask")
        masks = masks.to(state_and_time)
        h = self.input_projection(torch.cat([state_and_time[:, :-1], masks], dim=1))
        emb = self.time_embed(shared.timestep_embedding(state_and_time[:, -1], self.time_features))
        for block in self.blocks:
            h = block(h, emb)
        return project_masked(self.out(h), masks)


def build_model(model_config):
    cfg = dict(DEFAULT_MODEL, **model_config)
    if cfg.pop("kind") != "masked_clr_residual_mlp":
        raise ValueError("Unexpected conditional generator architecture")
    return MaskConditionalVelocity(**cfg)


def normalized_config(config):
    return dict(model=dict(DEFAULT_MODEL, **config.get("model", {})),
                training=dict(DEFAULT_TRAINING, **config.get("training", {})))


def load_trained_model(checkpoint_path, device="cpu", ema=True):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(saved["model_config"])
    model.load_state_dict(saved["ema_state" if ema else "model_state"], strict=True)
    return model.to(device).eval()


def conditional_flow_loss(predicted, target, masks):
    """Average per-observation squared error per nonredundant active coordinate."""
    if predicted.shape != target.shape or masks.shape != target.shape:
        raise ValueError("Conditional flow loss dimensions differ")
    denominator = (masks.sum(1) - 1).clamp_min(1)
    return (((predicted - target).square() * masks).sum(1) / denominator).mean()


def _step(model, ema_model, optimizer, scheduler, x1, masks, cfg):
    x0 = project_masked(torch.randn_like(x1), masks)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    epsilon = project_masked(torch.randn_like(x1), masks)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1 + cfg["sigma"] * epsilon
    predicted = model(torch.cat([xt, t[:, None]], dim=1), masks)
    loss = conditional_flow_loss(predicted, x1 - x0, masks)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite conditional ICFM training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    shared._ema_update(model, ema_model, cfg["ema_decay"])
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach()), float(norm)


def _training_objects(config, raw):
    cfg = config["training"]
    latent, masks = encode_masked_clr(raw)
    shared._seed(int(cfg["seed"]))
    model = build_model(config["model"]).to(cfg["device"])
    ema_model = copy.deepcopy(model).requires_grad_(False).eval()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warmup = int(cfg["warmup_steps"])
    if warmup < 1:
        raise ValueError("The fixed historical warmup must be positive")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step, warmup) / warmup)
    m = torch.as_tensor(masks, dtype=torch.float32, device=cfg["device"])
    x = project_masked(torch.as_tensor(latent, dtype=torch.float32, device=cfg["device"]), m)
    return model, ema_model, optimizer, scheduler, x, m


def train(config, raw_data_path, output):
    """Train a fixed-budget conditional abundance model, with exact epoch resume."""
    config = normalized_config(config)
    cfg = config["training"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = shared.read_matrix(raw_data_path)
    if raw.shape[1] != config["model"]["dim"] or len(raw) < cfg["batch_size"]:
        raise ValueError("Raw training table does not match the fixed conditional generator")
    if cfg["epochs"] < 1 or cfg["checkpoint_epochs"] < 1:
        raise ValueError("Training and checkpoint epoch counts must be positive")
    identity = dict(config_sha256=shared.canonical_hash(config), data_sha256=shared.sha256(raw_data_path),
                    representation="positive-taxon-centered-log-abundance")
    if (output / "metrics.json").exists():
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics.get("identity") != identity or metrics.get("status") != "complete":
            raise ValueError("Existing conditional fit uses different inputs")
        if shared.sha256(output / "final.pt") != metrics["final_checkpoint_sha256"]:
            raise ValueError("Completed conditional checkpoint changed")
        return metrics
    model, ema_model, optimizer, scheduler, x, masks = _training_objects(config, raw)
    epoch = updates = 0
    elapsed_before = 0.0
    history = []
    if (output / "latest.pt").exists():
        saved = torch.load(output / "latest.pt", map_location=cfg["device"], weights_only=False)
        if saved["identity"] != identity or saved["model_config"] != config["model"]:
            raise ValueError("Resume checkpoint differs from frozen conditional inputs")
        model.load_state_dict(saved["model_state"])
        ema_model.load_state_dict(saved["ema_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch, updates, history = saved["epochs_completed"], saved["updates"], saved["history"]
        elapsed_before = saved["elapsed_seconds"]
        shared._restore_rng(saved["rng"])
    start = time.monotonic()
    interrupted = []
    old_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGUSR1):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda received, frame: interrupted.append(received))

    def save_checkpoint(name):
        saved = dict(version=1, identity=identity, model_config=config["model"], training_config=cfg,
                     model_state=model.state_dict(), ema_state=ema_model.state_dict(),
                     optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     rng=shared._rng_state(), epochs_completed=epoch, updates=updates, history=history,
                     elapsed_seconds=elapsed_before + time.monotonic() - start)
        shared.atomic_save(saved, output / name)
        shared.write_json(output / "training.json", dict(config=config, identity=identity,
                                                        epochs_completed=epoch, updates=updates,
                                                        history=history))

    try:
        nbatch = len(raw) // int(cfg["batch_size"])
        while epoch < cfg["epochs"]:
            epoch_start = time.monotonic()
            order = np.random.permutation(len(raw))[:nbatch * cfg["batch_size"]]
            losses, norms = [], []
            for indices in order.reshape(nbatch, cfg["batch_size"]):
                rows = torch.as_tensor(indices, device=x.device)
                loss, norm = _step(model, ema_model, optimizer, scheduler, x[rows], masks[rows], cfg)
                losses.append(loss)
                norms.append(norm)
                updates += 1
            epoch += 1
            history.append(dict(epoch=epoch, updates=updates,
                                mean_flow_mse_per_active_dimension=float(np.mean(losses)),
                                max_preclip_gradient_norm=float(np.max(norms)),
                                lr=optimizer.param_groups[0]["lr"],
                                elapsed_seconds=time.monotonic() - epoch_start))
            if epoch == 1 or epoch % cfg["checkpoint_epochs"] == 0 or interrupted or epoch == cfg["epochs"]:
                save_checkpoint("latest.pt")
                print(json.dumps(history[-1]), flush=True)
            if interrupted:
                raise InterruptedError("Conditional epoch checkpoint saved; resume the same fixed training job")
        save_checkpoint("final.pt")
        richness = (raw > 0).sum(1)
        metrics = dict(status="complete", identity=identity, config=config, rows=len(raw),
                       coordinates=raw.shape[1], epochs_completed=epoch, updates=updates,
                       updates_per_epoch=nbatch, omitted_rows_per_epoch=len(raw) - nbatch * cfg["batch_size"],
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       elapsed_seconds=elapsed_before + time.monotonic() - start,
                       final_flow_mse_per_active_dimension=history[-1]["mean_flow_mse_per_active_dimension"],
                       loss_definition="mean_rows(sum_active((v-(z1-z0))^2)/max(k-1,1))",
                       mask_definition="raw_training_abundance>0", pseudocount=None,
                       training_richness_range=[int(richness.min()), int(richness.max())],
                       final_checkpoint_sha256=shared.sha256(output / "final.pt"),
                       training_log_sha256=shared.sha256(output / "training.json"),
                       torch_version=str(torch.__version__), numpy_version=np.__version__,
                       cuda_version=torch.version.cuda,
                       device_name=torch.cuda.get_device_name(x.device) if x.is_cuda else "cpu")
        shared.write_json(output / "metrics.json", metrics)
        return metrics
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def benchmark(config, raw_data_path, output, steps=30):
    """Disposable throughput benchmark, not a generator checkpoint or model choice."""
    config = normalized_config(config)
    raw = shared.read_matrix(raw_data_path)
    model, ema_model, optimizer, scheduler, x, masks = _training_objects(config, raw)
    cfg = config["training"]
    batch, batch_masks = x[:cfg["batch_size"]], masks[:cfg["batch_size"]]
    if len(batch) != cfg["batch_size"] or steps < 2:
        raise ValueError("Benchmark needs a full batch and at least two timed updates")
    for _ in range(3):
        _step(model, ema_model, optimizer, scheduler, batch, batch_masks, cfg)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
    start = time.monotonic()
    losses = [_step(model, ema_model, optimizer, scheduler, batch, batch_masks, cfg)[0] for _ in range(steps)]
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    elapsed = time.monotonic() - start
    result = dict(status="benchmark_only", config=config, steps=steps, seconds=elapsed,
                  seconds_per_update=elapsed / steps,
                  estimated_training_hours=elapsed / steps * cfg["epochs"] * (len(raw) // cfg["batch_size"]) / 3600,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  loss_first=losses[0], loss_last=losses[-1],
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(x.device) if x.is_cuda else None,
                  device_name=torch.cuda.get_device_name(x.device) if x.is_cuda else "cpu")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shared.write_json(output, result)
    return result


def sample_model(model, noise, masks, *, atol=1e-4, rtol=1e-4, batch_size=128,
                 device="cpu", return_metrics=False):
    """Integrate the conditional projected field with immutable caller-supplied masks."""
    from torchdyn.core import NeuralODE

    noise = np.asarray(noise)
    masks = _masks(masks, noise.shape)
    if noise.ndim != 2 or not np.isfinite(noise).all() or batch_size < 1:
        raise ValueError("Conditional sampling requires a finite Gaussian-noise matrix")
    if atol <= 0 or rtol <= 0:
        raise ValueError("ODE tolerances must be positive")
    model = model.to(device).eval()
    if any(part.training for part in model.modules()):
        raise AssertionError("A conditional generator submodule remained in training mode")

    class ConditionalField(nn.Module):
        def __init__(self, velocity, batch_masks):
            super().__init__()
            self.velocity = velocity
            self.register_buffer("masks", batch_masks)
            self.nfe = 0

        def forward(self, t, x, *args, **kwargs):
            self.nfe += 1
            times = t.to(x).expand(len(x), 1)
            result = self.velocity(torch.cat([x, times], dim=1), self.masks)
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError("Non-finite conditional ICFM sampling velocity")
            return result

    start = time.monotonic()
    pieces = []
    nfe = 0
    with torch.no_grad():
        for begin in range(0, len(noise), batch_size):
            batch_masks = torch.as_tensor(masks[begin:begin + batch_size], dtype=torch.float32, device=device)
            initial = project_masked(torch.as_tensor(
                noise[begin:begin + batch_size], dtype=torch.float32, device=device), batch_masks)
            vector_field = ConditionalField(model, batch_masks).eval()
            node = NeuralODE(vector_field, solver="dopri5", sensitivity="adjoint",
                             atol=atol, rtol=rtol).to(device).eval()
            if any(part.training for part in node.modules()):
                raise AssertionError("A conditional ODE submodule remained in training mode")
            result = node.trajectory(initial, t_span=torch.tensor([0., 1.], device=device))[-1]
            pieces.append(result.cpu().numpy())
            nfe += vector_field.nfe
    latent = np.concatenate(pieces)
    if not np.isfinite(latent).all() or np.any(latent[~masks] != 0):
        raise FloatingPointError("Conditional trajectories violated the finite masked-coordinate contract")
    # Validate that a strict support-preserving composition can be decoded, without
    # changing trajectories or clipping values. The caller stores the actual decode.
    decode_masked_clr(latent, masks)
    metrics = dict(rows=len(noise), dim=noise.shape[1], batch_size=batch_size,
                   solver="torchdyn.dopri5", saved_times=[0, 1], atol=atol, rtol=rtol,
                   function_evaluations=nfe, elapsed_seconds=time.monotonic() - start,
                   evaluation_mode=True, prior="mask-centered isotropic Gaussian",
                   mask_conditioned_velocity=True, pseudocount=None,
                   max_active_coordinate_sum_error=float(np.max(np.abs(latent.sum(1, dtype=np.float64)))),
                   input_noise_array_sha256=hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest(),
                   input_mask_array_sha256=hashlib.sha256(np.ascontiguousarray(masks).tobytes()).hexdigest())
    return (latent, metrics) if return_metrics else latent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "benchmark"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = train(config, args.data, args.output) if args.action == "train" else benchmark(
        config, args.data, args.output, args.steps)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
