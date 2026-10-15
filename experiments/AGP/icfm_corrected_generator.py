#!/usr/bin/env python3
"""Audited AGP ICFM inference correction and full-width velocity model.

This module does not choose data splits, tune a generator, or restore zeros.
The orchestration layer supplies frozen inputs and paired Gaussian draws.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_MODEL = dict(kind="residual_mlp", dim=613, width=1024, blocks=4,
                     time_features=128, time_width=512, dropout=0.1)
DEFAULT_TRAINING = dict(epochs=8000, batch_size=1044, lr=1e-5, sigma=1e-4,
                        warmup_steps=2000, grad_clip=1.0, ema_decay=0.9999,
                        weight_decay=0.0, seed=2026100601, device="cuda",
                        checkpoint_epochs=100)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def atomic_save(value, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def timestep_embedding(timesteps, dim, max_period=10000):
    """The historical sinusoidal time encoding, evaluated on the input device."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(
        half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    result = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        result = torch.cat([result, torch.zeros_like(result[:, :1])], dim=-1)
    return result


class FullWidthBlock(nn.Module):
    def __init__(self, width, time_width, dropout):
        super().__init__()
        self.in_layers = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, width))
        self.time_layers = nn.Sequential(nn.SiLU(), nn.Linear(time_width, 2 * width))
        self.out_norm = nn.LayerNorm(width)
        self.out_layers = nn.Sequential(nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width))

    def forward(self, x, time_emb):
        h = self.in_layers(x)
        scale, shift = self.time_layers(time_emb).chunk(2, dim=-1)
        h = self.out_norm(h) * (1 + scale) + shift
        return x + self.out_layers(h)


class FullWidthVelocity(nn.Module):
    """A time-conditioned residual MLP without a dimension-reducing hidden layer."""
    def __init__(self, dim=613, width=1024, blocks=4, time_features=128,
                 time_width=512, dropout=0.1):
        super().__init__()
        if width < dim or blocks < 1 or time_features < 2:
            raise ValueError("Velocity width must cover every data coordinate and blocks must be positive")
        self.dim, self.width, self.time_features = dim, width, time_features
        self.input_projection = nn.Linear(dim, width)
        self.time_embed = nn.Sequential(nn.Linear(time_features, time_width), nn.SiLU(),
                                        nn.Linear(time_width, time_width))
        self.blocks = nn.ModuleList([FullWidthBlock(width, time_width, dropout) for _ in range(blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, dim))
        # Match the legacy initialization convention while removing its narrow head.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.dim + 1:
            raise ValueError("Expected a batch of data coordinates followed by scalar time")
        h = self.input_projection(x[:, :-1])
        emb = self.time_embed(timestep_embedding(x[:, -1], self.time_features))
        for block in self.blocks:
            h = block(h, emb)
        return self.out(h)


def build_model(model_config):
    cfg = dict(DEFAULT_MODEL, **model_config)
    if cfg.pop("kind") != "residual_mlp":
        raise ValueError("Only the prespecified full-width residual MLP can be trained here")
    return FullWidthVelocity(**cfg)


def load_legacy_model(checkpoint_path, legacy_source_path, device="cpu"):
    """Load exactly the six architecture definitions without external import side effects.

    The caller must hash-pin both source and checkpoint before calling this function.
    No training helpers, plotting imports, or module-level device settings execute.
    """
    wanted = {"timestep_embedding", "TimestepBlock", "ResBlock1D", "AttentionBlock1D", "UNet1D", "UNet"}
    source = Path(legacy_source_path)
    parsed = ast.parse(source.read_text())
    definitions = [node for node in parsed.body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted]
    if {node.name for node in definitions} != wanted or len(definitions) != len(wanted):
        raise ValueError("Legacy architecture definitions changed or are duplicated")
    namespace = dict(torch=torch, nn=nn, math=math)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    cfg = saved["config"]
    state = saved["ema_icfm_lt_state"]
    dim = int(state["unet.out.2.weight"].shape[0])
    if cfg["nn_structure"] != "unet" or cfg["cfm_method"] != "icfm":
        raise ValueError("Unexpected legacy generator recipe")
    model = namespace["UNet"](dim=dim, out_dim=dim, w=cfg["w"], time_varying=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def load_trained_model(checkpoint_path, device="cpu", ema=True):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(saved["model_config"])
    model.load_state_dict(saved["ema_state" if ema else "model_state"], strict=True)
    return model.to(device).eval()


def read_matrix(path):
    path = Path(path)
    values = np.load(path, allow_pickle=False) if path.suffix == ".npy" else np.loadtxt(path, delimiter=",")
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Generator inputs must be a finite matrix")
    return values


def normalized_config(config):
    return dict(model=dict(DEFAULT_MODEL, **config.get("model", {})),
                training=dict(DEFAULT_TRAINING, **config.get("training", {})))


def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


@torch.no_grad()
def _ema_update(model, ema_model, decay):
    for source, target in zip(model.parameters(), ema_model.parameters()):
        target.mul_(decay).add_(source, alpha=1 - decay)


def _step(model, ema_model, optimizer, scheduler, x1, cfg):
    # Exactly the constant-sigma independent conditional flow-matching objective:
    # x_t=(1-t)x0+t*x1+sigma*epsilon, u_t=x1-x0. No feature rescaling.
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    epsilon = torch.randn_like(x1)
    xt = (1 - t[:, None]) * x0 + t[:, None] * x1 + cfg["sigma"] * epsilon
    velocity = model(torch.cat([xt, t[:, None]], dim=1))
    loss = (velocity - (x1 - x0)).square().mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite ICFM training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    _ema_update(model, ema_model, cfg["ema_decay"])
    optimizer.zero_grad(set_to_none=True)
    return float(loss.detach()), float(norm)


def _training_objects(config, values):
    cfg = config["training"]
    _seed(int(cfg["seed"]))
    model = build_model(config["model"]).to(cfg["device"])
    ema_model = copy.deepcopy(model).requires_grad_(False).eval()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    warmup = int(cfg["warmup_steps"])
    if warmup < 1:
        raise ValueError("The historical linear warmup must be positive")
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step, warmup) / warmup)
    x = torch.as_tensor(values, dtype=torch.float32, device=cfg["device"])
    return model, ema_model, optimizer, scheduler, x


def train(config, data_path, output):
    """Fixed-budget training; atomically checkpoint complete epochs for exact resume."""
    config = normalized_config(config)
    cfg = config["training"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    values = read_matrix(data_path)
    if values.shape[1] != config["model"]["dim"] or len(values) < cfg["batch_size"]:
        raise ValueError("Training matrix does not match the fixed generator architecture/batch size")
    if cfg["epochs"] < 1 or cfg["checkpoint_epochs"] < 1:
        raise ValueError("Positive training and checkpoint epoch counts are required")
    identity = dict(config_sha256=canonical_hash(config), data_sha256=sha256(data_path))
    if (output / "metrics.json").exists():
        metrics = json.loads((output / "metrics.json").read_text())
        if metrics.get("identity") != identity or metrics.get("status") != "complete":
            raise ValueError("Existing completed training uses different inputs")
        if sha256(output / "final.pt") != metrics["final_checkpoint_sha256"]:
            raise ValueError("Completed checkpoint changed")
        return metrics
    model, ema_model, optimizer, scheduler, x = _training_objects(config, values)
    epoch = updates = 0
    elapsed_before = 0.0
    history = []
    if (output / "latest.pt").exists():
        saved = torch.load(output / "latest.pt", map_location=cfg["device"], weights_only=False)
        if saved["identity"] != identity or saved["model_config"] != config["model"]:
            raise ValueError("Resume checkpoint does not match frozen training inputs")
        model.load_state_dict(saved["model_state"])
        ema_model.load_state_dict(saved["ema_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        epoch, updates, history = saved["epochs_completed"], saved["updates"], saved["history"]
        elapsed_before = saved["elapsed_seconds"]
        _restore_rng(saved["rng"])
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
                     rng=_rng_state(), epochs_completed=epoch, updates=updates, history=history,
                     elapsed_seconds=elapsed_before + time.monotonic() - start)
        atomic_save(saved, output / name)
        write_json(output / "training.json", dict(config=config, identity=identity,
                                                   epochs_completed=epoch, updates=updates,
                                                   history=history))

    try:
        nbatch = len(values) // int(cfg["batch_size"])
        while epoch < cfg["epochs"]:
            epoch_start = time.monotonic()
            order = np.random.permutation(len(values))[:nbatch * cfg["batch_size"]]
            losses, norms = [], []
            for indices in order.reshape(nbatch, cfg["batch_size"]):
                rows = torch.as_tensor(indices, device=x.device)
                loss, norm = _step(model, ema_model, optimizer, scheduler, x[rows], cfg)
                losses.append(loss)
                norms.append(norm)
                updates += 1
            epoch += 1
            history.append(dict(epoch=epoch, updates=updates, mean_flow_mse=float(np.mean(losses)),
                                max_preclip_gradient_norm=float(np.max(norms)),
                                lr=optimizer.param_groups[0]["lr"],
                                elapsed_seconds=time.monotonic() - epoch_start))
            if epoch == 1 or epoch % cfg["checkpoint_epochs"] == 0 or interrupted or epoch == cfg["epochs"]:
                save_checkpoint("latest.pt")
                print(json.dumps(history[-1]), flush=True)
            if interrupted:
                raise InterruptedError("Checkpoint saved after signal; resume the same fixed training job")
        save_checkpoint("final.pt")
        metrics = dict(status="complete", identity=identity, config=config, rows=len(values),
                       coordinates=values.shape[1], epochs_completed=epoch, updates=updates,
                       updates_per_epoch=nbatch, omitted_rows_per_epoch=len(values) - nbatch * cfg["batch_size"],
                       parameter_count=sum(p.numel() for p in model.parameters()),
                       elapsed_seconds=elapsed_before + time.monotonic() - start,
                       final_flow_mse=history[-1]["mean_flow_mse"],
                       final_checkpoint_sha256=sha256(output / "final.pt"),
                       training_log_sha256=sha256(output / "training.json"),
                       torch_version=str(torch.__version__), numpy_version=np.__version__,
                       cuda_version=torch.version.cuda,
                       device_name=torch.cuda.get_device_name(x.device) if x.is_cuda else "cpu")
        write_json(output / "metrics.json", metrics)
        return metrics
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def benchmark(config, data_path, output, steps=30):
    """Short disposable throughput benchmark; never produces a study checkpoint."""
    config = normalized_config(config)
    values = read_matrix(data_path)
    model, ema_model, optimizer, scheduler, x = _training_objects(config, values)
    cfg = config["training"]
    batch = x[:cfg["batch_size"]]
    if len(batch) != cfg["batch_size"] or steps < 2:
        raise ValueError("Benchmark needs a complete batch and at least two timed steps")
    for _ in range(3):
        _step(model, ema_model, optimizer, scheduler, batch, cfg)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
        torch.cuda.reset_peak_memory_stats(x.device)
    start = time.monotonic()
    losses = [_step(model, ema_model, optimizer, scheduler, batch, cfg)[0] for _ in range(steps)]
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    elapsed = time.monotonic() - start
    result = dict(status="benchmark_only", config=config, steps=steps, seconds=elapsed,
                  seconds_per_update=elapsed / steps,
                  estimated_training_hours=elapsed / steps * cfg["epochs"] * (len(values) // cfg["batch_size"]) / 3600,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  loss_first=losses[0], loss_last=losses[-1],
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(x.device) if x.is_cuda else None,
                  device_name=torch.cuda.get_device_name(x.device) if x.is_cuda else "cpu")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


def sample_model(model, noise, *, atol=1e-4, rtol=1e-4, batch_size=128,
                 device="cpu", return_metrics=False):
    """Dopri5 integration with dropout disabled and caller-provided paired noise."""
    from torchdyn.core import NeuralODE

    noise = np.asarray(noise)
    if noise.ndim != 2 or not np.isfinite(noise).all() or not len(noise) or batch_size < 1:
        raise ValueError("Sampling requires a nonempty finite Gaussian-noise matrix")
    if atol <= 0 or rtol <= 0:
        raise ValueError("ODE tolerances must be positive")
    model = model.to(device).eval()
    if any(part.training for part in model.modules()):
        raise AssertionError("A generator submodule remained in training mode")
    class VectorField(nn.Module):
        def __init__(self, velocity):
            super().__init__()
            self.velocity = velocity
            self.nfe = 0

        def forward(self, t, x, *args, **kwargs):
            self.nfe += 1
            times = t.to(x).expand(len(x), 1)
            result = self.velocity(torch.cat([x, times], dim=1))
            if not bool(torch.isfinite(result).all()):
                raise FloatingPointError("Non-finite ICFM sampling velocity")
            return result

    vector_field = VectorField(model).eval()
    node = NeuralODE(vector_field, solver="dopri5", sensitivity="adjoint",
                     atol=atol, rtol=rtol).to(device).eval()
    if any(part.training for part in node.modules()):
        raise AssertionError("An ODE submodule remained in training mode")

    start = time.monotonic()
    pieces = []
    with torch.no_grad():
        for begin in range(0, len(noise), batch_size):
            initial = torch.as_tensor(noise[begin:begin + batch_size], dtype=torch.float32, device=device)
            result = node.trajectory(initial, t_span=torch.tensor([0., 1.], device=device))[-1]
            pieces.append(result.cpu().numpy())
    values = np.concatenate(pieces)
    if not np.isfinite(values).all():
        raise FloatingPointError("Non-finite generated tree coordinates")
    metrics = dict(rows=len(noise), dim=noise.shape[1], batch_size=batch_size,
                   solver="torchdyn.dopri5", saved_times=[0, 1], atol=atol, rtol=rtol,
                   function_evaluations=vector_field.nfe, elapsed_seconds=time.monotonic() - start,
                   evaluation_mode=True, input_noise_array_sha256=hashlib.sha256(
                       np.ascontiguousarray(noise).tobytes()).hexdigest())
    return (values, metrics) if return_metrics else values


def invert_tree(values, child_node_path, tree_taxa_path, abundance_path):
    """Vectorize the historical ordered tree inversion, without thresholding zeros.

    Sigmoid arithmetic retains the input dtype before allocation into the historical
    float64 branch-mass array. Evaluating each branch only on its safe domain avoids
    overflow warnings without changing the mathematical reconstruction.
    """
    values = np.asarray(values)
    if values.ndim != 2 or not np.isfinite(values).all() or values.dtype.kind != "f":
        raise ValueError("Tree coordinates must be a finite floating-point matrix")
    children = pd.read_csv(child_node_path, index_col=0, dtype=str)
    taxa_names = pd.read_csv(tree_taxa_path, index_col=0, dtype=str).values.flatten()
    out_names = pd.read_csv(abundance_path, index_col=0, usecols=[0]).index.to_numpy()
    n, internal = values.shape
    tips = internal + 1
    if children.shape != (internal, 2) or len(taxa_names) != tips or len(out_names) != tips:
        raise ValueError("Tree and abundance table dimensions disagree")
    if len(set(taxa_names)) != tips or set(taxa_names) != set(out_names):
        raise ValueError("Tree and abundance taxa differ or contain duplicates")
    sigmoid = np.empty_like(values)
    positive = values >= 0
    sigmoid[positive] = 1 / (1 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    sigmoid[~positive] = exponent / (1 + exponent)
    branches = np.empty((n, internal, 2), dtype=np.float64)
    branches[:, :, 0] = sigmoid
    branches[:, :, 1] = 1 - branches[:, :, 0]
    node_rows = {name: row for row, name in enumerate(children.index)}
    leaves = np.zeros((n, tips), dtype=np.float64)
    seen_leaves = set()
    for row, pair in enumerate(children.itertuples(index=False, name=None)):
        for branch, child in enumerate(pair):
            if child in node_rows:
                child_row = node_rows[child]
                if child_row <= row:
                    raise ValueError("Historical reconstruction requires parent-before-child tree order")
                branches[:, child_row, :] *= branches[:, row, branch, None]
            else:
                leaf = int(child.replace("Node_", "")) - 1
                if leaf < 0 or leaf >= tips or leaf in seen_leaves:
                    raise ValueError("Invalid or repeated tree tip")
                seen_leaves.add(leaf)
                leaves[:, leaf] = branches[:, row, branch]
    if len(seen_leaves) != tips:
        raise ValueError("Tree failed to reconstruct every tip")
    order = np.argsort(taxa_names)
    final_indices = order[np.searchsorted(taxa_names[order], out_names)]
    result = leaves[:, final_indices]
    if not np.isfinite(result).all() or np.any(result < 0) or not np.allclose(result.sum(1), 1, atol=1e-10):
        raise FloatingPointError("Reconstructed tree compositions are invalid")
    return result


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
