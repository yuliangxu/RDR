"""Fitting primitives with explicit denominator and checkpoint conventions.

Experiments own data loading, random seeds, split construction, and the chosen
training recipe. These entry points preserve their distinct optimizer,
BatchNorm, scheduler, early stopping, and checkpoint-restoration behavior;
consolidating their loops would change historical reproduction semantics.

No trainer silently converts Q samples into midpoint samples, except the
explicitly named run_DRE_fdiv_SGD_mix_in_loss. Callers are responsible for
independent training, checkpoint-selection, and final-evaluation data.
"""

import copy
import itertools
from itertools import cycle
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import networks as dre
from .diagnostics import _unpack_loader_batch
from .losses import (
    Hellinger_loss, KL_loss, Chisq_loss, JS_loss,
    hellinger_stable, kl_from_outputs, chisq_from_outputs,
)
from .networks import MLP, ViewToMNIST, dcgan_init

__all__ = [
    'run_DRE_fdiv',
    'run_DRE_fdiv_cnn',
    'run_DRE_fdiv_precond',
    'run_DRE_fdiv_embed',
    'run_DRE_fdiv_SGD',
    'run_DRE_fdiv_SGD_mix_in_loss',
    'run_DRE_fdiv_cnn_minibatch',
    'run_DRE_fdiv_cnn_minibatch_celeba64',
    'make_loader_mixture_sampler',
]


# Tensor and loader helpers

def _model_device_dtype(model: torch.nn.Module):
    p = next(model.parameters())
    return p.device, p.dtype


def _to_model(x, model: torch.nn.Module):
    if x is None:
        return None
    device, dtype = _model_device_dtype(model)
    return x.to(device=device, dtype=dtype, non_blocking=True)


def _extract_x(batch):
    if isinstance(batch, torch.Tensor): return batch
    if isinstance(batch, (list, tuple)): return batch[0]
    if isinstance(batch, dict): return batch.get("x", next(iter(batch.values())))
    raise TypeError("Unsupported batch type; expected Tensor/tuple/dict.")


def _match_batch_sizes(x_p: torch.Tensor, x_q: torch.Tensor):
    b = min(x_p.shape[0], x_q.shape[0])
    return x_p[:b], x_q[:b]


def _flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1) if x.ndim > 2 else x


def _ensure_each_side(n_total: int, p_weight: float):
    n_p = int(n_total * p_weight)
    n_q = n_total - n_p
    if n_total > 0 and n_p == 0: n_p, n_q = 1, n_total - 1
    if n_total > 0 and n_q == 0: n_q, n_p = 1, n_total - 1
    return n_p, n_q


def _first_batch_dim(loader):
    it = iter(loader)
    batch = next(it)
    z, _ = _unpack_loader_batch(batch)
    if z.ndim > 2:
        z = z.view(z.size(0), -1)
    return z.shape[1]


def _freeze_bn(module: nn.Module):
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.eval()           # use running stats
        for p in module.parameters():
            p.requires_grad_(False)   # optional: also freeze affine params


# Explicit denominator mixture sampler

def make_loader_mixture_sampler(
    p_loader,
    q_loader,
    p_frac: float = 0.5,
):
    """Sample from ``p_frac * p + (1 - p_frac) * q`` using two loaders.

    This is useful for relative density-ratio estimation.  In particular, with
    ``p_frac=0.5`` the returned distribution is m=(p+q)/2, so a density-ratio
    model trained with p as its numerator estimates p/m = 2p/(p+q).
    """
    if not 0.0 <= p_frac <= 1.0:
        raise ValueError("p_frac must be in [0, 1]")

    p_iter = iter(p_loader)
    q_iter = iter(q_loader)

    def take(loader, iterator, n):
        chunks = []
        needed = n
        while needed > 0:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            chunks.append(images[:needed])
            needed -= min(needed, images.size(0))
        return torch.cat(chunks, dim=0), iterator

    @torch.no_grad()
    def sampler(bs, device, dtype=torch.float32):
        nonlocal p_iter, q_iter
        n_p = int(round(bs * p_frac))
        n_q = bs - n_p
        parts = []
        if n_p:
            images, p_iter = take(p_loader, p_iter, n_p)
            parts.append(images)
        if n_q:
            images, q_iter = take(q_loader, q_iter, n_q)
            parts.append(images)
        images = torch.cat(parts, dim=0).to(device=device, dtype=dtype)
        return images[torch.randperm(images.size(0), device=device)]

    return sampler


# Full-batch and feature-loader training

def run_DRE_fdiv(
    x_p, x_q, xi=0.5, num_epochs: int = 2000,
    x_p_val=None, x_q_val=None,
    loss_method: str = 'Hellinger',
    NN: str = "MLP",
    output_alpha: float = 2.0,
    log_scale: bool = False,
    optimizer=None,
    scheduler=None,
    # early-stop controls
    early_start: int = 300,                 # don't check early stop before this epoch
    early_patience: Optional[int] = None,   # e.g., 5 (None disables early stop)
    min_delta: float = 1e-5,                # improvement threshold on val loss
    restore_best: bool = True,              # reload best weights at stop
    clip_max_norm: float = 1.0,
):


    """Fit the historical full-batch MLP to numerator P and denominator R.

    Inputs are (N,D) tensors; pass R=(P+Q)/2 as ``x_q`` for relative ratios.
    Returns (model, train_losses), adding val_losses when x_p_val is supplied.
    Validation controls scheduling. A best checkpoint is tracked only after
    ``early_start`` when patience is enabled, and restored only at early stop.
    The NN='UNet' and 'CNN_DCGAN' branches reference unavailable legacy classes;
    retained executable workflows use NN='MLP'. The MLP output stays bounded
    even if log_scale=True; the flag only changes loss interpretation.
    """
    device = x_p.device if isinstance(x_p, torch.Tensor) else 'cpu'

    input_dim = x_p.shape[1]

    if NN == "UNet":
        model = DynamicMLP(input_dim=input_dim, hidden_dims=[32, 32])
    elif NN == "CNN_DCGAN":
        model = DREConvNet_DCGAN(in_ch=1, base=64, log_scale=log_scale).to(device)
    else:
        # Default to MLP
        model = MLP(input_dim=input_dim, output_alpha=output_alpha)

    def _model_device_dtype(model):
        p = next(model.parameters())
        return p.device, p.dtype

    def _to_model(x, model):
        if x is None:
            return None
        device, dtype = _model_device_dtype(model)
        return x.to(device=device, dtype=dtype)
    x_p    = _to_model(x_p,    model)
    x_q    = _to_model(x_q,    model)
    x_p_val = _to_model(x_p_val, model) if x_p_val is not None else None
    x_q_val = _to_model(x_q_val, model) if x_q_val is not None else None


    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    if scheduler is None:
        # Works with/without val data (ReduceLROnPlateau expects a metric each epoch)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, cooldown=2
        )

    # ensure tensors are on the same device
    def _prep(X):
        if isinstance(X, torch.Tensor):
            return X.to(device)
        return X

    x_p, x_q = _prep(x_p), _prep(x_q)


    x_p_val = _prep(x_p_val) if x_p_val is not None else None
    x_q_val = _prep(x_q_val) if x_q_val is not None else None


    losses, val_losses = [], []
    # early-stop state
    best_val = float('inf')
    best_state = None
    no_improve = 0

    show_iter = max(1, num_epochs // 10)

    for epoch in range(num_epochs):
        optimizer.zero_grad(set_to_none=True)

        # ---- training loss ----
        if loss_method == 'Hellinger':
            loss = Hellinger_loss(model, x_p, x_q, xi, log_scale=log_scale)
        elif loss_method == 'KL':
            loss = KL_loss(model, x_p, x_q)
        elif loss_method == 'Chisq':
            loss = Chisq_loss(model, x_p, x_q)
        elif loss_method == 'JS':
            loss = JS_loss(model, x_p, x_q)
        else:
            raise ValueError(f"Unknown loss_method: {loss_method}")

        loss.backward()
        # clip & (optionally) log grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        optimizer.step()

        losses.append(float(loss.detach().cpu()))

        # ---- validation loss (if provided) ----
        val_loss = None
        metric = loss.detach()  # default: train loss if no val
        if (x_p_val is not None) and (x_q_val is not None):
            model.eval()
            with torch.no_grad():
                if loss_method == 'Hellinger':
                    val_loss = Hellinger_loss(model, x_p_val, x_q_val, xi, log_scale=log_scale)
                elif loss_method == 'KL':
                    val_loss = KL_loss(model, x_p_val, x_q_val)
                elif loss_method == 'Chisq':
                    val_loss = Chisq_loss(model, x_p_val, x_q_val)
                elif loss_method == 'JS':
                    val_loss = JS_loss(model, x_p_val, x_q_val)
                else:
                    raise ValueError(f"Unknown loss_method: {loss_method}")
            model.train()

            v = float(val_loss.detach().cpu())
            val_losses.append(v)
            metric = val_loss.detach()  # scheduler steps on val

            # ---- patience-based early stopping (after early_start) ----
            if early_patience is not None and epoch >= early_start:
                if v < best_val - min_delta:
                    best_val = v
                    no_improve = 0
                    if restore_best:
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    no_improve += 1
                if no_improve >= early_patience:
                    print(f"[EarlyStop] epoch {epoch} | best_val={best_val:.6f} (patience {early_patience})")
                    if restore_best and best_state is not None:
                        model.load_state_dict(best_state)
                    break

        # ---- scheduler step (val if present, else train) ----
        scheduler.step(metric)

        # ---- logging ----
        if (epoch % show_iter) == 0:
            if val_loss is not None:
                print(f"Epoch {epoch:4d}: loss={loss.item():.6f} | val_loss={val_loss.item():.6f}")
            else:
                print(f"Epoch {epoch:4d}: loss={loss.item():.6f}")

        if torch.isnan(loss):
            print("NaN loss encountered; stopping.")
            break
    if x_p_val is None:
        return model, losses
    else:
        return model, losses, val_losses


def run_DRE_fdiv_embed(
    p_embed_loader,              # Iterable/DataLoader yielding (B,D)
    q_embed_loader,              # Iterable/DataLoader yielding (B,D)
    xi=0.5,
    num_epochs=100,
    steps_per_epoch=400,
    p_val_loader=None,           # optional val loaders (embedding)
    q_val_loader=None,
    loss_method='Hellinger',     # 'Hellinger' | 'KL' | 'Chisq'
    NN="MLP",                    # kept for API parity; embeddings -> MLP
    log_scale=False,
    optimizer=None,
    scheduler=None,
    clip_max_norm: float = 1.0,
    amp: bool = False,
    device: str = "cuda",
    mlp_hidden=512,        # tweak capacity here
    val_steps: Optional[int] = None,
    return_val_losses: bool = False,
):
    """Fit an MLP from numerator and denominator embedding loaders.

    Loaders may yield tensors, (embeddings, images), or dictionaries understood
    by ``_unpack_loader_batch``. Features are flattened, and each loader is
    cached/cycled with itertools.cycle exactly as in the historical workflow.
    Validation drives the scheduler; no best checkpoint is restored. Returns
    (model.eval(), epoch_losses), plus validation losses when requested.
    ``log_scale`` affects the loss, not the bounded MLP architecture.
    """

    # --- infer input_dim from the p loader ---
    input_dim = _first_batch_dim(p_embed_loader)

    # --- build model ---
    if NN == "MLP":
        model = MLP(input_dim=input_dim, hidden_dim=mlp_hidden).to(device)
    else:
        # For embeddings, default to MLP regardless
        model = MLP(input_dim=input_dim, hidden_dim=mlp_hidden).to(device)

    # --- opt / sched ---
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, cooldown=2
        )

    # --- loss selector ---
    def compute_loss(xp, xq):
        if loss_method == 'Hellinger':
            return Hellinger_loss(model, xp, xq, xi, log_scale=log_scale)
        elif loss_method == 'KL':
            return KL_loss(model, xp, xq)
        elif loss_method == 'Chisq':
            return Chisq_loss(model, xp, xq)
        else:
            raise ValueError(f"Unknown loss_method: {loss_method}")

    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    losses = []
    val_losses = []
    show_iter = max(1, num_epochs // 10)

    p_it = cycle(p_embed_loader)   # infinite
    q_it = cycle(q_embed_loader)

    for epoch in range(num_epochs):
        model.train()
        running = 0.0

        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)

            # --- fetch minibatches ---
            batch_p = next(p_it); xp, _ = _unpack_loader_batch(batch_p)
            batch_q = next(q_it); xq, _ = _unpack_loader_batch(batch_q)
            if xp.ndim > 2: xp = xp.view(xp.size(0), -1)
            if xq.ndim > 2: xq = xq.view(xq.size(0), -1)
            xp = _to_model(xp, model); xq = _to_model(xq, model)

            # ensure 2D
            if xp.ndim > 2: xp = xp.view(xp.size(0), -1)
            if xq.ndim > 2: xq = xq.view(xq.size(0), -1)
            xp = _to_model(xp, model)
            xq = _to_model(xq, model)

            # --- forward/backward ---
            if amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = compute_loss(xp, xq)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = compute_loss(xp, xq)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
                optimizer.step()

            running += float(loss.item())

            if torch.isnan(loss):
                print("NaN loss encountered; stopping.")
                return model.eval(), losses

        epoch_loss = running / steps_per_epoch

        # --- validation metric (optional) ---
        metric = torch.as_tensor(epoch_loss, device=device)
        if (p_val_loader is not None) and (q_val_loader is not None):
            model.eval()
            with torch.no_grad():
                pv_it = cycle(p_val_loader)
                qv_it = cycle(q_val_loader)
                # one epoch-level val estimate; default to a full validation pass
                n_val_steps = val_steps if val_steps is not None else min(len(p_val_loader), len(q_val_loader))
                val_sum = 0.0
                for _ in range(n_val_steps):
                    batch_pv = next(pv_it); xp, _ = _unpack_loader_batch(batch_pv)
                    batch_qv = next(qv_it); xq, _ = _unpack_loader_batch(batch_qv)
                    if xp.ndim > 2: xp = xp.view(xp.size(0), -1)
                    if xq.ndim > 2: xq = xq.view(xq.size(0), -1)
                    xp = _to_model(xp, model); xq = _to_model(xq, model)
                    if loss_method == 'Hellinger':
                        v = Hellinger_loss(model, xp, xq, xi, log_scale=log_scale)
                    elif loss_method == 'KL':
                        v = KL_loss(model, xp, xq)
                    else:
                        v = Chisq_loss(model, xp, xq)
                    val_sum += float(v.item())
                val_loss = val_sum / n_val_steps
                val_losses.append(val_loss)
                metric = torch.as_tensor(val_loss, device=device)

        scheduler.step(metric)

        losses.append(epoch_loss)
        if (epoch % show_iter) == 0:
            print(f"[embed] Epoch {epoch:4d}: train_loss={epoch_loss:.6f}"
                  + (f", val_metric={metric.item():.6f}" if (p_val_loader is not None and q_val_loader is not None) else ""))

    if return_val_losses:
        return model.eval(), losses, val_losses
    return model.eval(), losses


def run_DRE_fdiv_SGD(
    p_loader: DataLoader,
    q_source,                        # EITHER DataLoader-like iterable OR callable(n)->Tensor
    *,
    xi: float = 0.5,
    num_epochs: int = 50,
    loss_method: str = "Hellinger",  # {"Hellinger","KL","Chisq"}
    NN: str = "MLP",
    log_scale: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    clip_max_norm: float = 1.0,
    steps_per_epoch: Optional[int] = None,
    val_p_loader: Optional[DataLoader] = None,
    val_q_source: Optional[object] = None,  # same flexibility as q_source
    early_stop_patience: Optional[int] = None,
    early_stop_min_delta: float = 0.0,
    restore_best: bool = True,
    device: Optional[torch.device] = None,
):
    # ---- infer device and input dim ----
    """Fit a flattened-feature MLP from P and a supplied denominator source.

    ``q_source`` is a loader or callable(batch_size) returning tensors. It
    must already represent the intended denominator R. Batch pairs are
    truncated to equal sizes before independently normalized loss evaluation.
    Returns (model, {'train_loss': [...], 'val_loss': [...]}). Validation tracks
    the strict best checkpoint independently of the patience/min_delta test,
    and restores that checkpoint on any exit when restore_best=True.
    NN='MLP' is the available model; other legacy branches are unresolved.
    """
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    first_p = _extract_x(next(iter(p_loader)))
    input_dim = int(first_p.view(first_p.size(0), -1).size(1))

    # ---- build model ----
    if NN == "UNet":
        model = DynamicMLP(input_dim=input_dim, hidden_dims=[32, 32])
    elif NN == "CNN_DCGAN":
        model = DREConvNet_DCGAN(in_ch=1, base=64, log_scale=log_scale)
    else:
        model = MLP(input_dim=input_dim)
    model = model.to(device)

    # ---- opt & sched ----
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, cooldown=2
    )

    # ---- helpers for q-source ----
    def _make_iter(src, fallback_len=None):
        """Return an iterator over q-batches for one epoch.
        """
        if callable(src):
            # callable: we'll call with current p-batch size inside the loop
            return None  # signal "call per step"
        else:
            return iter(src)

    def _next_q_batch(src, q_iter, want_bsz: int):
        if callable(src):
            qb = src(want_bsz)  # e.g., q_mixed(batch_size)
        else:
            qb = _extract_x(next(q_iter))
        return qb

    # ---- steps/epoch ----
    if steps_per_epoch is None:
        if not callable(q_source) and hasattr(q_source, "__len__"):
            steps_per_epoch = min(len(p_loader), len(q_source))
        else:
            steps_per_epoch = len(p_loader)

    history = {"train_loss": [], "val_loss": []}
    best_metric = float("inf")
    early_stop_metric = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        running = 0.0

        p_iter = iter(p_loader)
        q_iter = _make_iter(q_source)

        # if both are real loaders with lengths, cycle the shorter
        if (not callable(q_source)) and hasattr(p_loader, "__len__") and hasattr(q_source, "__len__"):
            if len(p_loader) < len(q_source):
                p_iter = itertools.cycle(p_loader)
            elif len(q_source) < len(p_loader):
                q_iter = itertools.cycle(q_source)

        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)

            batch_p = _extract_x(next(p_iter))
            # flatten to (B, D) if needed (remove if your model expects images)
            if batch_p.ndim > 2: batch_p = batch_p.view(batch_p.size(0), -1)

            batch_q = _next_q_batch(q_source, q_iter, batch_p.size(0))
            if batch_q.ndim > 2: batch_q = batch_q.view(batch_q.size(0), -1)

            batch_p, batch_q = _match_batch_sizes(_to_model(batch_p, model), _to_model(batch_q, model))

            if loss_method == "Hellinger":
                loss = Hellinger_loss(model, batch_p, batch_q, xi, log_scale=log_scale)
            elif loss_method == "KL":
                loss = KL_loss(model, batch_p, batch_q)
            elif loss_method == "Chisq":
                loss = Chisq_loss(model, batch_p, batch_q)
            else:
                raise ValueError(f"Unknown loss_method: {loss_method}")

            loss.backward()
            if clip_max_norm and clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
            running += float(loss.detach())

        train_loss = running / steps_per_epoch
        history["train_loss"].append(train_loss)

        # ---- validation (optional) ----
        if (val_p_loader is not None) and (val_q_source is not None):
            model.eval()
            val_running, val_steps = 0.0, min(len(val_p_loader), len(val_q_source)) if (hasattr(val_q_source, "__len__") and not callable(val_q_source)) else len(val_p_loader)
            vp_iter = iter(val_p_loader)
            vq_iter = _make_iter(val_q_source)

            with torch.no_grad():
                for _ in range(val_steps):
                    vp = _extract_x(next(vp_iter));
                    if vp.ndim > 2: vp = vp.view(vp.size(0), -1)
                    vq = _next_q_batch(val_q_source, vq_iter, vp.size(0))
                    if vq.ndim > 2: vq = vq.view(vq.size(0), -1)
                    vp, vq = _match_batch_sizes(_to_model(vp, model), _to_model(vq, model))
                    if loss_method == "Hellinger":
                        vloss = Hellinger_loss(model, vp, vq, xi, log_scale=log_scale)
                    elif loss_method == "KL":
                        vloss = KL_loss(model, vp, vq)
                    else:
                        vloss = Chisq_loss(model, vp, vq)
                    val_running += float(vloss)
            val_loss = val_running / max(1, val_steps)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_metric:
                best_metric = val_loss
                if restore_best:
                    best_state = copy.deepcopy(model.state_dict())

            if val_loss < early_stop_metric - early_stop_min_delta:
                early_stop_metric = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            scheduler.step(train_loss)

        if (epoch % max(1, num_epochs // 10)) == 0:
            msg = f"[{epoch:04d}] train={train_loss:.6f}"
            if history["val_loss"]:
                msg += f"  val={history['val_loss'][-1]:.6f}"
            print(msg)

        if (
            val_p_loader is not None
            and val_q_source is not None
            and early_stop_patience is not None
            and epochs_without_improvement >= early_stop_patience
        ):
            print(
                f"[EarlyStop] epoch {epoch:04d} | "
                f"best_val={best_metric:.6f} (patience {early_stop_patience})"
            )
            break

    if best_state is not None and restore_best:
        model.load_state_dict(best_state)

    return model, history


def run_DRE_fdiv_SGD_mix_in_loss(
    p_loader: DataLoader,
    q_source,                        # callable(n)->Tensor (generator); DO NOT pass a mixed sampler here
    *,
    xi: float = 0.5,
    p_weight: float = 0.5,           # fraction of real samples to include in x_q mixture
    num_epochs: int = 50,
    loss_method: str = "Hellinger",  # {"Hellinger","KL","Chisq"}
    NN: str = "MLP",
    log_scale: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
    clip_max_norm: float = 1.0,
    steps_per_epoch: Optional[int] = None,
    val_p_loader: Optional[DataLoader] = None,
    val_q_source: Optional[object] = None,  # callable(n)->Tensor for generated validation
    device: Optional[torch.device] = None,
):
    # ---- infer device and input dim from real loader ----
    """Fit an MLP while constructing the denominator mixture inside each step.

    ``q_source`` must be an unmixed generator callable(batch_size). The routine
    combines subsets of P and generated Q using ``p_weight``; effective integer
    counts follow ``_ensure_each_side``. A step reuses its P batch in both the
    numerator and mixture. Validation follows the same construction. Returns
    (model, {'train_loss': [...], 'val_loss': [...]}), with no early stopping or
    checkpoint restoration. NN='MLP' is the available model; other branches
    reference unresolved legacy classes.
    """
    device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    first_p = _extract_x(next(iter(p_loader)))
    input_dim = int(_flatten_if_needed(first_p).size(1))

    # ---- build model ----
    if NN == "UNet":
        model = DynamicMLP(input_dim=input_dim, hidden_dims=[32, 32])
    elif NN == "CNN_DCGAN":
        model = DREConvNet_DCGAN(in_ch=1, base=64, log_scale=log_scale)
    else:
        model = MLP(input_dim=input_dim)
    model = model.to(device)

    # ---- opt & sched ----
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    scheduler = scheduler or torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, cooldown=2, verbose=True
    )

    # ---- steps/epoch ----
    if steps_per_epoch is None:
        steps_per_epoch = len(p_loader)  # q is callable; drive by p_loader

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        p_iter = iter(p_loader)

        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)

            # real batch x_p
            batch_p = _flatten_if_needed(_extract_x(next(p_iter)))
            bsz = batch_p.size(0)

            # generated batch (same target size as x_p)
            gen_q = q_source(bsz)
            gen_q = _flatten_if_needed(gen_q)

            # move to model device/dtype
            batch_p = _to_model(batch_p, model)
            gen_q   = _to_model(gen_q, model)

            # trim to common size (defensive)
            b = min(batch_p.size(0), gen_q.size(0))
            batch_p = batch_p[:b]
            gen_q   = gen_q[:b]

            # ---- build x_q as 50-50 (or p_weight) mixture of *current* x_p and generated ----
            n_p, n_g = _ensure_each_side(b, p_weight)
            idx_p = torch.randperm(b, device=batch_p.device)[:n_p]
            idx_g = torch.randperm(b, device=gen_q.device)[:n_g]
            x_q = torch.cat([batch_p[idx_p], gen_q[idx_g]], dim=0)
            perm = torch.randperm(x_q.size(0), device=x_q.device)
            x_q = x_q[perm]

            # ---- compute loss ----
            if loss_method == "Hellinger":
                loss = Hellinger_loss(model, batch_p, x_q, xi, log_scale=log_scale)
            elif loss_method == "KL":
                loss = KL_loss(model, batch_p, x_q)
            elif loss_method == "Chisq":
                loss = Chisq_loss(model, batch_p, x_q)
            else:
                raise ValueError(f"Unknown loss_method: {loss_method}")

            loss.backward()
            if clip_max_norm and clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
            optimizer.step()
            running += float(loss.detach())

        train_loss = running / steps_per_epoch
        history["train_loss"].append(train_loss)

        # ---- validation (optional) ----
        if (val_p_loader is not None) and (val_q_source is not None):
            model.eval()
            with torch.no_grad():
                vp_iter = iter(val_p_loader)
                val_steps = len(val_p_loader)
                val_running = 0.0
                for _ in range(val_steps):
                    vp = _flatten_if_needed(_extract_x(next(vp_iter)))
                    bsz = vp.size(0)
                    vq_gen = val_q_source(bsz)
                    vq_gen = _flatten_if_needed(vq_gen)

                    vp = _to_model(vp, model)
                    vq_gen = _to_model(vq_gen, model)
                    b = min(vp.size(0), vq_gen.size(0))
                    vp, vq_gen = vp[:b], vq_gen[:b]

                    n_p, n_g = _ensure_each_side(b, p_weight)
                    idx_p = torch.randperm(b, device=vp.device)[:n_p]
                    idx_g = torch.randperm(b, device=vq_gen.device)[:n_g]
                    vq = torch.cat([vp[idx_p], vq_gen[idx_g]], dim=0)
                    perm = torch.randperm(vq.size(0), device=vq.device)
                    vq = vq[perm]

                    if loss_method == "Hellinger":
                        vloss = Hellinger_loss(model, vp, vq, xi, log_scale=log_scale)
                    elif loss_method == "KL":
                        vloss = KL_loss(model, vp, vq)
                    else:
                        vloss = Chisq_loss(model, vp, vq)
                    val_running += float(vloss)
            val_loss = val_running / max(1, val_steps)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        if (epoch % max(1, num_epochs // 10)) == 0:
            print(f"[{epoch:04d}] train={train_loss:.6f}" + (f"  val={history['val_loss'][-1]:.6f}" if history["val_loss"] else ""))

    return model, history


# Image-loader training

def run_DRE_fdiv_cnn_minibatch(
    p_loader,                  # DataLoader yielding real MNIST batches (N,1,28,28)
    q_sampler,                 # callable(bs, device, dtype) -> fake/mixed batch (N,1,28,28)
    num_epochs=20,
    loss_method='Hellinger',   # 'Hellinger' | 'KL' | 'Chisq'
    log_scale=False,           # if True, model outputs log w; else w>0 via BoundedSoftplus
    optimizer=None, scheduler=None,
    clip_max_norm: float = 1.0,
    in_ch: int = 1, img_hw=(28,28), base: int = 64,
    val_loader=None,           # optional DataLoader for validation p
    val_q_sampler=None,        # optional independent validation denominator sampler
    val_q_batches: int = 4,    # how many q batches to use for val
    early_stop_patience: Optional[int] = None,
    early_stop_min_delta: float = 0.0,
    restore_best: bool = True,
    return_val_losses: bool = False,
    print_every: int = 100,    # steps
    bn_freeze_epoch: int = 5,  # freeze BN after this epoch (set <0 to disable)
    onecycle_max_lr: float = 6e-4,   # used if scheduler is None
    onecycle_pct_start: float = 0.1,
    weight_decay: float = 0.0,
    output_alpha: float = 0.5,
):
    """Fit the historical single-channel 28x28 CNN using a denominator sampler.

    ``p_loader`` yields (images, ...); ``q_sampler(bs, device, dtype)`` must
    sample the intended denominator R. P/R share one forward for BatchNorm.
    Pass independent validation loaders/samplers for checkpoint selection;
    otherwise validation falls back to the training denominator sampler.
    Tracks/restores the best validation state with the min_delta threshold.
    Returns (model, per_step_losses), plus validation losses when requested.
    See losses.py: the historical Hellinger output loss ignores log_scale,
    and the Chisq objective differs from the model-based Chisq_loss.
    """
    H, W = img_hw
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    core = dre.DREConvNet_DCGAN_MNIST(
        in_ch=in_ch,
        base=base,
        log_scale=log_scale,
        output_alpha=output_alpha,
    )
    model = nn.Sequential(dre.ViewToMNIST(), core).to(device)
    model.apply(dcgan_init)

    # optimizer / scheduler
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=weight_decay)

    total_steps = num_epochs * len(p_loader)
    use_onecycle = False
    if scheduler is None:
        # Per-batch warmup + cosine decay
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=onecycle_max_lr,
            total_steps=total_steps,
            pct_start=onecycle_pct_start,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1e3,
        )
        use_onecycle = True

    step = 0
    train_losses = []
    val_losses = []
    ema = None
    best_metric = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()

        # Optionally freeze BN after a few epochs
        if bn_freeze_epoch >= 0 and epoch >= bn_freeze_epoch:
            model.apply(_freeze_bn)

        for x_p, *_ in p_loader:         # MNIST DataLoader usually yields (images, labels)
            x_p = x_p.to(device=device, dtype=next(model.parameters()).dtype)
            bs = x_p.size(0)

            # sample q to match batch size
            x_q = q_sampler(bs, device, dtype=x_p.dtype)

            # one forward on concatenated batch -> stable BatchNorm
            X = torch.cat([x_p, x_q], dim=0)
            w = model(X)                   # (2B,1); if log_scale=True this is log w
            w_p, w_q = w[:bs], w[bs:]

            # ----- choose loss -----
            if loss_method == 'Hellinger':
                loss = hellinger_stable(w_p, w_q, log_space=log_scale, eps=1e-6, log_clip=8.0)
            elif loss_method == 'KL':
                if log_scale:
                    loss = kl_from_outputs(w_p, w_q)
                else:
                    loss = kl_from_outputs(torch.log(w_p.clamp_min(1e-8)),
                                           torch.log(w_q.clamp_min(1e-8)))
            elif loss_method == 'Chisq':
                if log_scale:
                    loss = chisq_from_outputs(torch.exp(w_p), torch.exp(w_q))
                else:
                    loss = chisq_from_outputs(w_p, w_q)
            else:
                raise ValueError(f"Unknown loss_method: {loss_method}")

            # ----- update -----
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            optimizer.step()
            if use_onecycle:
                scheduler.step()          # per-batch step

            # ----- logging -----
            l = loss.item()
            train_losses.append(l)
            ema = l if ema is None else (0.95 * ema + 0.05 * l)
            if (step % print_every) == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch:03d} step {step:06d}  loss {l:.4f}  ema {ema:.4f}  lr {lr:.2e}")
            step += 1

        # ---- validation (optional) ----
        metric = torch.tensor(train_losses[-1], device=device)
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vals = []
                it = iter(val_loader)
                validation_sampler = val_q_sampler or q_sampler
                for _ in range(val_q_batches):
                    try:
                        x_pv, *_ = next(it)
                    except StopIteration:
                        break
                    bs_v = x_pv.size(0)
                    x_pv = x_pv.to(device=device, dtype=next(model.parameters()).dtype)
                    x_qv = validation_sampler(bs_v, device, dtype=x_pv.dtype)
                    Xv = torch.cat([x_pv, x_qv], dim=0)
                    wv = model(Xv)
                    w_pv, w_qv = wv[:bs_v], wv[bs_v:]
                    if loss_method == 'Hellinger':
                        vals.append(hellinger_stable(w_pv, w_qv, log_space=log_scale).item())
                    elif loss_method == 'KL':
                        if log_scale:
                            vals.append(kl_from_outputs(w_pv, w_qv).item())
                        else:
                            vals.append(kl_from_outputs(torch.log(w_pv.clamp_min(1e-8)),
                                                        torch.log(w_qv.clamp_min(1e-8))).item())
                    else:  # Chisq
                        if log_scale:
                            vals.append(chisq_from_outputs(torch.exp(w_pv), torch.exp(w_qv)).item())
                        else:
                            vals.append(chisq_from_outputs(w_pv, w_qv).item())
                if vals:
                    metric = torch.tensor(sum(vals) / len(vals), device=device)
                    val_losses.append(metric.item())

                    if metric.item() < best_metric - early_stop_min_delta:
                        best_metric = metric.item()
                        epochs_without_improvement = 0
                        if restore_best:
                            best_state = copy.deepcopy(model.state_dict())
                    else:
                        epochs_without_improvement += 1

                    print(
                        f"epoch {epoch:03d} validation loss {metric.item():.4f} "
                        f"(best {best_metric:.4f})"
                    )

        # If caller passed ReduceLROnPlateau, step it per epoch with metric
        if (not use_onecycle) and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)

        if (
            val_loader is not None
            and early_stop_patience is not None
            and epochs_without_improvement >= early_stop_patience
        ):
            print(f"Early stopping after epoch {epoch:03d}")
            break

    if best_state is not None and restore_best:
        model.load_state_dict(best_state)

    if return_val_losses:
        return model, train_losses, val_losses
    return model, train_losses


def run_DRE_fdiv_cnn_minibatch_celeba64(
    p_loader,                  # DataLoader yielding real CelebA-64 (N,3,64,64) in [-1,1]
    q_sampler,                 # callable(bs, device, dtype) -> (N,3,64,64) in [-1,1]
    num_epochs: int = 20,
    loss_method: str = 'Hellinger',  # 'Hellinger' | 'KL' | 'Chisq'
    log_scale: bool = False,   # model outputs log w if True, else w>0
    optimizer=None, scheduler=None,
    clip_max_norm: float = 1.0,
    in_ch: int = 3, img_hw=(64,64), ndf: int = 64,
    val_loader=None,           # optional DataLoader for validation p
    val_q_sampler=None,        # optional independent validation denominator sampler
    val_q_batches: int = 4,    # how many q batches to use for val
    early_stop_patience: Optional[int] = None,
    early_stop_min_delta: float = 0.0,
    restore_best: bool = True,
    restore_mode: str = "best_val",
    return_val_losses: bool = False,
    print_every: int = 100,    # steps
    bn_freeze_epoch: int = 5,  # freeze BN after this epoch (set <0 to disable)
    onecycle_max_lr: float = 6e-4,
    onecycle_pct_start: float = 0.1,
    weight_decay: float = 0.0,
):
    """Fit the historical RGB 64x64 CNN using a denominator sampler.

    ``p_loader`` and ``q_sampler(bs, device, dtype)`` provide (N,3,64,64)
    tensors in [-1,1]; the denominator must already be R=(P+Q)/2 when wanted.
    P/R share one BatchNorm forward. Independent validation sources are caller
    responsibilities. Validation checkpoint selection uses strict improvement,
    separately from early stopping's min_delta threshold. ``restore_mode`` is
    'best_val' or 'last_finite_train'; the latter records the state before a
    finite-loss optimizer update. Nonfinite training losses stop before update.
    Returns (model, per_step_losses), plus validation losses when requested.
    See losses.py for the unchanged output-based objective conventions.
    """
    H, W = img_hw
    assert (H, W) == (64, 64), "Set img_hw=(64,64) for CelebA-64."
    assert in_ch == 3, "CelebA-64 uses RGB (in_ch=3)."

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    # Model: DCGAN-style CelebA64 backbone + ratio head
    core = dre.RatioNetCelebA64(in_ch=in_ch, ndf=ndf, log_scale=log_scale).to(device=device, dtype=dtype)
    model = core  # no view wrapper needed

    # optimizer / scheduler
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=weight_decay)

    total_steps = num_epochs * len(p_loader)
    use_onecycle = False
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=onecycle_max_lr,
            total_steps=total_steps,
            pct_start=onecycle_pct_start,
            anneal_strategy='cos',
            div_factor=10.0,
            final_div_factor=1e3,
        )
        use_onecycle = True

    step = 0
    train_losses = []
    val_losses = []
    ema = None
    best_metric = float("inf")
    early_stop_metric = float("inf")
    best_state = None
    last_finite_state = copy.deepcopy(model.state_dict()) if restore_best else None
    last_finite_step = None
    epochs_without_improvement = 0
    stop_training = False

    valid_restore_modes = {"best_val", "last_finite_train"}
    if restore_mode not in valid_restore_modes:
        raise ValueError(f"restore_mode must be one of {sorted(valid_restore_modes)}")

    for epoch in range(num_epochs):
        model.train()

        # Optionally freeze BatchNorm after a few epochs
        if bn_freeze_epoch >= 0 and epoch >= bn_freeze_epoch:
            model.apply(_freeze_bn)

        for x_p, *_ in p_loader:
            x_p = x_p.to(device=device, dtype=dtype)          # (B,3,64,64) in [-1,1]
            bs = x_p.size(0)

            # sample q to match batch size
            x_q = q_sampler(bs, device, dtype=dtype)          # (B,3,64,64) in [-1,1]

            # one forward on concatenated batch -> better BN stats
            X = torch.cat([x_p, x_q], dim=0)
            w = model(X)                   # (2B,)
            w_p, w_q = w[:bs], w[bs:]

            # ----- choose loss (stable forms you already use) -----
            if loss_method == 'Hellinger':
                loss = hellinger_stable(w_p, w_q, log_space=log_scale, eps=1e-6, log_clip=8.0)
            elif loss_method == 'KL':
                if log_scale:
                    loss = kl_from_outputs(w_p, w_q)
                else:
                    loss = kl_from_outputs(torch.log(w_p.clamp_min(1e-8)),
                                           torch.log(w_q.clamp_min(1e-8)))
            elif loss_method == 'Chisq':
                if log_scale:
                    loss = chisq_from_outputs(torch.exp(w_p), torch.exp(w_q))
                else:
                    loss = chisq_from_outputs(w_p, w_q)
            else:
                raise ValueError(f"Unknown loss_method: {loss_method}")

            l = loss.item()
            train_losses.append(l)
            if not torch.isfinite(loss):
                print(
                    f"[CelebA64] non-finite training loss {l} at "
                    f"epoch {epoch:03d} step {step:06d}; stopping before optimizer update"
                )
                stop_training = True
                break
            if restore_best:
                last_finite_state = copy.deepcopy(model.state_dict())
                last_finite_step = step

            # ----- update -----
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            optimizer.step()
            if use_onecycle:
                scheduler.step()  # per-batch

            # ----- logging -----
            ema = l if ema is None else (0.95 * ema + 0.05 * l)
            if (step % print_every) == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[CelebA64] epoch {epoch:03d} step {step:06d}  loss {l:.4f}  ema {ema:.4f}  lr {lr:.2e}")
            step += 1

        if stop_training:
            break

        # ---- optional validation on p vs q ----
        metric = torch.tensor(train_losses[-1], device=device)
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vals = []
                it = iter(val_loader)
                validation_sampler = val_q_sampler or q_sampler
                for _ in range(val_q_batches):
                    try:
                        x_pv, *_ = next(it)
                    except StopIteration:
                        break
                    bs_v = x_pv.size(0)
                    x_pv = x_pv.to(device=device, dtype=dtype)
                    x_qv = validation_sampler(bs_v, device, dtype=dtype)
                    Xv = torch.cat([x_pv, x_qv], dim=0)
                    wv = model(Xv)
                    w_pv, w_qv = wv[:bs_v], wv[bs_v:]
                    if loss_method == 'Hellinger':
                        v_loss = hellinger_stable(w_pv, w_qv, log_space=log_scale)
                    elif loss_method == 'KL':
                        if log_scale:
                            v_loss = kl_from_outputs(w_pv, w_qv)
                        else:
                            v_loss = kl_from_outputs(
                                torch.log(w_pv.clamp_min(1e-8)),
                                torch.log(w_qv.clamp_min(1e-8)),
                            )
                    else:  # Chisq
                        if log_scale:
                            v_loss = chisq_from_outputs(torch.exp(w_pv), torch.exp(w_qv))
                        else:
                            v_loss = chisq_from_outputs(w_pv, w_qv)
                    if torch.isfinite(v_loss):
                        vals.append(v_loss.item())
                if vals:
                    metric = torch.tensor(sum(vals) / len(vals), device=device)
                    val_losses.append(metric.item())

                    if metric.item() < best_metric:
                        best_metric = metric.item()
                        if restore_best:
                            best_state = copy.deepcopy(model.state_dict())

                    if metric.item() < early_stop_metric - early_stop_min_delta:
                        early_stop_metric = metric.item()
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    print(
                        f"[CelebA64] epoch {epoch:03d} validation loss "
                        f"{metric.item():.4f} (best {best_metric:.4f})"
                    )
                else:
                    epochs_without_improvement += 1
                    print(f"[CelebA64] epoch {epoch:03d} validation loss is non-finite")

        # If caller passed ReduceLROnPlateau, step it per epoch with metric
        if (not use_onecycle) and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)

        if (
            val_loader is not None
            and early_stop_patience is not None
            and epochs_without_improvement >= early_stop_patience
        ):
            print(f"[CelebA64] early stopping after epoch {epoch:03d}")
            break

    if restore_best:
        state_to_restore = None
        restore_label = None
        if restore_mode == "best_val" and best_state is not None:
            state_to_restore = best_state
            restore_label = f"best validation loss {best_metric:.6f}"
        elif last_finite_state is not None:
            state_to_restore = last_finite_state
            restore_label = f"last finite training step {last_finite_step}"
        if state_to_restore is not None:
            model.load_state_dict(state_to_restore)
            print(f"[CelebA64] restored {restore_label}")

    if return_val_losses:
        return model, train_losses, val_losses
    return model, train_losses


# Retained historical full-batch CNN and preconditioning recipes

def run_DRE_fdiv_cnn(
    x_p, x_q, xi=0.5, num_epochs=2000, x_p_val=None, x_q_val=None,
    loss_method='Hellinger', log_scale=False, optimizer=None, scheduler=None,
    clip_max_norm: float = 1.0, *, in_ch: int = 1, img_hw: tuple = (28,28), base: int = 64
):
    """Retained full-batch CNN entry point with an unresolved legacy backbone.

    This historical function references undefined ``DREConvNet_DCGAN`` and is
    not a working reproduction entry point. Use the existing minibatch recipe
    selected by an experiment; no replacement model is silently substituted.
    """
    H, W = img_hw
    # Build model with a guaranteed-shape front end\
    device = x_p.device if isinstance(x_p, torch.Tensor) else 'cpu'
    core = DREConvNet_DCGAN(in_ch=in_ch, base=base, log_scale=log_scale)
    model = nn.Sequential(ViewToMNIST(), core).to(x_p.device)
    import inspect
    print(model)  # architecture
    print("Model class:", model.__class__.__name__)
    print("Forward defined at:", inspect.getsourcefile(model.forward))
    print("Forward starts line:", inspect.getsourcelines(model.forward)[1])

    # Move data to model device/dtype (adapter handles reshape/validation at forward)
    def _to_model(x):
        if x is None: return None
        p = next(model.parameters())
        return (x if isinstance(x, torch.Tensor) else torch.as_tensor(x)).to(device=p.device, dtype=p.dtype)
    x_p = _to_model(x_p)
    x_q = _to_model(x_q)
    x_p_val      = _to_model(x_p_val) if x_p_val is not None else None
    x_q_val      = _to_model(x_q_val) if x_q_val is not None else None

    # Optimizer / scheduler
    if optimizer is None: optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, cooldown=2, verbose=True)

    losses, show_iter = [], max(1, num_epochs // 10)
    for epoch in range(num_epochs):
        model.train(); optimizer.zero_grad(set_to_none=True)
        if loss_method == 'Hellinger':
            loss = Hellinger_loss(model, x_p, x_q, xi, log_scale=log_scale)
        elif loss_method == 'KL':
            loss = KL_loss(model, x_p, x_q)
        elif loss_method == 'Chisq':
            loss = Chisq_loss(model, x_p, x_q)
        else:
            raise ValueError(f"Unknown loss_method: {loss_method}")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        optimizer.step(); losses.append(loss.item())

        metric = loss.detach()
        if (x_p_val is not None) and (x_q_val is not None):
            model.eval()
            with torch.no_grad():
                if loss_method == 'Hellinger':
                    val = Hellinger_loss(model, x_p_val, x_q_val, xi, log_scale=log_scale)
                elif loss_method == 'KL':
                    val = KL_loss(model, x_p_val, x_q_val)
                else:
                    val = Chisq_loss(model, x_p_val, x_q_val)
            metric = val.detach()

        scheduler.step(metric)
        if (epoch % show_iter) == 0: print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}")
        if torch.isnan(loss): print("NaN loss encountered; stopping."); break

    return model, losses


def _u_from_model_out(y, log_scale: bool):
    """Return u = log g. If model already outputs log g (log_scale=True), use it.
    If model outputs g>0 (e.g., via Softplus), take log(y + eps) safely.
    """
    if log_scale:
        return y
    else:
        eps = 1e-8
        return torch.log(y + eps)


@torch.no_grad()
def _renorm_on_q(model, x_q, log_scale: bool, strength: float = 1.0):
    """Optional tiny re-centering so that E_q[g] ≈ 1 on the current batch.
    This is a conservative correction: it shifts u by a scalar offset.
    Works only if the model's final mapping supports bias-like shifting.
    If unsure, set strength=0 to disable.
    """
    if strength <= 0:
        return
    yq = model(x_q)               # shape (n_q, 1) or (n_q,)
    if log_scale:
        uq = yq
        gq = torch.exp(uq)
    else:
        gq = yq
        uq = torch.log(gq + 1e-8)

    c = gq.mean().clamp_min(1e-8)
    # shift u -> u - log(c)  i.e., scale g -> g / c
    shift = -torch.log(c)
    # Try to apply shift to the last bias if present
    last_bias = None
    for p in reversed(list(model.parameters())):
        if p.ndim == 1:  # heuristic: a bias parameter (vector)
            last_bias = p
            break
    if last_bias is not None:
        last_bias.add_(strength * shift.item())


def run_DRE_fdiv_precond(
    x_p, x_q, xi=0.5, num_epochs=2000,
    x_p_val=None, x_q_val=None,
    loss_method='Hellinger',
    NN="MLP",
    log_scale=False,
    optimizer=None,
    scheduler=None,
    clip_max_norm: float = 1.0,
    ngd_beta: float = 0.95,        # EMA for Fisher diag
    ngd_damping: float = 1e-3,     # λ for (F+λ)^{-1/2}
    ngd_eps: float = 1e-12,        # numerical epsilon
    renorm_strength: float = 0.0   # set to e.g. 0.2 to softly enforce E_q[g]=1
):
    """Adds a balanced diagonal 'natural gradient' preconditioner:
      v <- beta*v + (1-beta)* E_{ν}[J_x(θ)^2]  with ν ≈ sqrt(p q)
    and uses grad / sqrt(v + damping).

    The ν-weights are approximated on each epoch by
      w_p = 1/sqrt(g(x)),  x~p
      w_q = sqrt(g(x)),    x~q
    with stop-gradient on the weights.
    """

    input_dim = x_p.shape[1]

    if NN == "UNet":
        model = DynamicMLP(input_dim=input_dim, hidden_dims=[32, 32])
    elif NN == "RBFNet":
        model = RBFNetAdaptive(
            input_dim=x_p.shape[1],
            n_centers=50,
            output_dim=1,
            beta_init=1.0,
            beta_min=1e-6,
            beta_max=1e2
        )
    else:
        model = MLP(input_dim=input_dim)

    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, cooldown=2
        )

    # --- state for the diagonal Fisher (per-parameter EMA of squared "Jacobian") ---
    fisher_ema = {p: torch.zeros_like(p, memory_format=torch.preserve_format)
                  for p in model.parameters() if p.requires_grad}

    losses = []
    show_iter = max(1, num_epochs // 10)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        # ---------------- forward for loss ----------------
        if loss_method == 'Hellinger':
            loss = Hellinger_loss(model, x_p, x_q, xi, log_scale=log_scale)
        elif loss_method == 'KL':
            loss = KL_loss(model, x_p, x_q)
        elif loss_method == 'Chisq':
            loss = Chisq_loss(model, x_p, x_q)
        else:
            raise ValueError(f"Unknown loss_method: {loss_method}")

        # ---------------- forward for u and ν-weights (stop-grad) ----------------
        # We only need model outputs again; keep this detached branch light.
        with torch.no_grad():
            yp = model(x_p)
            yq = model(x_q)
            up = _u_from_model_out(yp, log_scale=log_scale)
            uq = _u_from_model_out(yq, log_scale=log_scale)
            gp = torch.exp(up)
            gq = torch.exp(uq)
            # balanced reference weights (no grad):
            w_p = 1.0 / torch.sqrt(gp + 1e-8)
            w_q = torch.sqrt(gq + 1e-8)

        # ---------------- backprop true loss to get raw grads ----------------
        loss.backward()

        # (optional) clip raw grads before preconditioning
        if clip_max_norm is not None and clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)

        # ---------------- estimate diag Fisher under ν ≈ sqrt(pq) ---------------
        # We use a single extra backward pass on a *proxy* scalar that produces
        # gradients proportional to sum_x w(x) * J_x(θ), then square that to keep
        # a cheap diagonal preconditioner. This is a practical approximation of
        # E_ν[J^2] that works well in practice.
        optimizer.zero_grad(set_to_none=True)  # clear grads to accumulate proxy

        # Re-forward u with grad enabled (small overhead).
        yp = model(x_p)
        yq = model(x_q)
        up = _u_from_model_out(yp, log_scale=log_scale)
        uq = _u_from_model_out(yq, log_scale=log_scale)

        # Build proxy objective: mean_x w(x) * u(x)
        # (weights are detached constants from above)
        # Normalization keeps magnitudes stable across epochs.
        Z = (w_p.numel() + w_q.numel())
        proxy = (w_p.view(-1) * up.view(-1)).sum()
        proxy += (w_q.view(-1) * uq.view(-1)).sum()
        proxy = proxy / max(1.0, float(Z))

        proxy.backward()

        # Update EMA of squared proxy-grads (diagonal Fisher approx)
        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                g = p.grad
                if g is None:
                    continue
                v = fisher_ema[p]
                v.mul_(ngd_beta).addcmul_(g, g, value=(1.0 - ngd_beta))

        # ---------------- apply NGD-style preconditioning to the true grads ------
        # Recompute true loss grads (we cleared them for proxy).
        optimizer.zero_grad(set_to_none=True)
        if loss_method == 'Hellinger':
            loss = Hellinger_loss(model, x_p, x_q, xi, log_scale=log_scale)
        elif loss_method == 'KL':
            loss = KL_loss(model, x_p, x_q)
        else:
            loss = Chisq_loss(model, x_p, x_q)

        loss.backward()

        with torch.no_grad():
            for p in model.parameters():
                if not p.requires_grad:
                    continue
                if p.grad is None:
                    continue
                denom = torch.sqrt(fisher_ema[p] + ngd_damping) + ngd_eps
                p.grad.div_(denom)  # preconditioned gradient (F+λ)^(-1/2) * grad

        # final (optional) clip after preconditioning
        if clip_max_norm is not None and clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)

        # step with your chosen optimizer (AdamW by default)
        optimizer.step()

        # gentle batch renormalization to keep E_q[g] ~ 1 (optional)
        _renorm_on_q(model, x_q, log_scale=log_scale, strength=renorm_strength)

        # ---- bookkeeping / scheduler metric ----
        losses.append(float(loss.detach().cpu()))
        metric = loss.detach()

        if (x_p_val is not None) and (x_q_val is not None):
            model.eval()
            with torch.no_grad():
                if loss_method == 'Hellinger':
                    val_loss = Hellinger_loss(model, x_p_val, x_q_val, xi, log_scale=log_scale)
                elif loss_method == 'KL':
                    val_loss = KL_loss(model, x_p_val, x_q_val)
                else:
                    val_loss = Chisq_loss(model, x_p_val, x_q_val)
            model.train()
            metric = val_loss.detach()

        scheduler.step(metric)

        if epoch % show_iter == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}")

        if torch.isnan(loss):
            print("NaN loss detected; stopping.")
            break

    return model, losses


def fit_ratio_mlp(model, train_p, train_q, val_p, val_q, loss="hellinger",
                  max_epochs=1000, min_epochs=300, patience=30):
    """Fit an RDR network; always restore its absolute best validation risk.

    Full-batch AdamW (lr=5e-4, decay=.01), gradient norm cap 1, and the
    architecture-pilot learning-rate schedule are common to every objective.
    Checkpoint risk is the chosen training loss; cross-model selection should
    use a common proper score such as balanced validation Brier. Patience's
    1e-5 threshold does not suppress smaller checkpoint improvements.
    Returns (model, metadata, history), where history is a list of dictionaries.
    """
    import math
    from numbers import Integral
    from .losses import midpoint_rdr_loss

    if loss not in ("hellinger", "kl", "chisq", "js"):
        raise ValueError(f"Unknown RDR objective: {loss}")
    if any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0
           for x in (max_epochs, min_epochs, patience)) or min_epochs > max_epochs:
        raise ValueError("Require positive epoch limits, patience, and min_epochs <= max_epochs")
    parameter = next(model.parameters())
    arrays = []
    for value in (train_p, train_q, val_p, val_q):
        tensor = torch.as_tensor(value, dtype=parameter.dtype, device=parameter.device).detach()
        if tensor.ndim != 2 or not len(tensor) or tensor.shape[1] != model.dimension:
            raise ValueError(f"Samples must be nonempty matrices with {model.dimension} columns")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("Samples must be finite")
        arrays.append(tensor)
    train_p, train_q, val_p, val_q = arrays
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=5, cooldown=2)
    best, stopping_best, stale = float("inf"), float("inf"), 0
    history, best_state, best_epoch = [], None, None
    stop_reason = "max_epochs"
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = midpoint_rdr_loss(model, train_p, train_q, loss)
        if not bool(torch.isfinite(risk)):
            raise FloatingPointError(f"Nonfinite training objective at epoch {epoch}")
        risk.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(midpoint_rdr_loss(model, val_p, val_q, loss))
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite validation objective at epoch {epoch}")
        history.append({"epoch": epoch, "train_loss": float(risk.detach()), "validation_loss": value,
                        "learning_rate": optimizer.param_groups[0]["lr"], "gradient_norm": float(norm)})
        if value < best:
            best, best_epoch, best_state = value, epoch, copy.deepcopy(model.state_dict())
        if epoch >= min_epochs:
            if value < stopping_best - 1e-5:
                stopping_best, stale = value, 0
            else:
                stale += 1
        scheduler.step(value)
        if epoch >= min_epochs and stale >= patience:
            stop_reason = "validation_patience"
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored = float(midpoint_rdr_loss(model, val_p, val_q, loss))
    if abs(restored-best) > 1e-7:
        raise RuntimeError("Best-validation checkpoint restoration failed")
    return model, {"loss": loss, "epochs": len(history), "best_epoch": best_epoch,
                   "best_val_loss": best, "restored_val_loss": restored,
                   "n_parameters": sum(p.numel() for p in model.parameters()),
                   "stop_reason": stop_reason, "checkpoint_rule": "absolute minimum validation training loss",
                   "min_epochs": min_epochs, "max_epochs": max_epochs, "patience": patience,
                   "patience_min_delta": 1e-5}, history


__all__ += ["fit_ratio_mlp"]
