import torch
import torch.nn as nn
import math
from typing import Callable, Optional, Tuple, Iterable
import utils.DRE_func as dre

# ---------- generator sampler ----------

# For CNN

def make_q_mixed_sampler(G, z_dim, real_loader, gen_frac=0.5, post=None):
    """
    Create a sampler that returns a batch sampled from a mixture:
        q = gen_frac * G  +  (1 - gen_frac) * real(MNIST)

    Parameters
    ----------
    G : nn.Module
        Pretrained generator mapping z -> images of shape (N,1,28,28) in [-1,1] or [0,1].
    z_dim : int
        Latent dimension for the generator.
    real_loader : DataLoader
        DataLoader yielding real MNIST batches (images[, labels]).
        This can be a separate loader from your p_loader so samples don't overlap.
    gen_frac : float, default 0.5
        Fraction of the batch drawn from the generator (0.0–1.0).
    post : callable or None
        Optional transform to apply to generator outputs to match the real data scaling.

    Returns
    -------
    sampler : callable
        sampler(bs, device, dtype=torch.float32) -> (bs,1,28,28) tensor
    """
    real_iter = iter(real_loader)

    @torch.no_grad()
    def sampler(bs, device, dtype=torch.float32):
        nonlocal real_iter

        # how many from G vs real
        n_gen  = int(round(bs * gen_frac))
        n_real = bs - n_gen

        # --- get real images (may need to pull from multiple mini-batches) ---
        needed = n_real
        real_imgs_chunks = []
        while needed > 0:
            try:
                batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                batch = next(real_iter)
            # DataLoader could return (imgs, labels) or just imgs
            if isinstance(batch, (list, tuple)):
                imgs = batch[0]
            else:
                imgs = batch
            imgs = imgs.to(device=device, dtype=dtype)
            if imgs.dim() == 3:  # (N,H,W) -> (N,1,H,W)
                imgs = imgs.unsqueeze(1)
            real_imgs_chunks.append(imgs)
            needed -= imgs.size(0)

        real_imgs = torch.cat(real_imgs_chunks, dim=0)[:n_real]

        # --- sample generator images ---
        if n_gen > 0:
            z = torch.randn(n_gen, z_dim, device=device, dtype=dtype)
            gen_imgs = G(z)
            if post is not None:
                gen_imgs = post(gen_imgs)
            if gen_imgs.dim() == 3:  # (N,H,W) -> (N,1,H,W)
                gen_imgs = gen_imgs.unsqueeze(1)
        else:
            gen_imgs = real_imgs.new_empty((0, *real_imgs.shape[1:]))

        # --- mix and shuffle within the batch ---
        X = torch.cat([gen_imgs, real_imgs], dim=0)
        perm = torch.randperm(X.size(0), device=device)
        return X[perm]

    return sampler
# For VAE
def make_mnist_vae_50_50_sampler(
    vae: "VAEWrapper",       # <-- quoted
    real_loader,
    *,
    post=None,                 # optional transform for generated imgs (e.g., scale)
    return_source: bool = False  # if True, also returns a boolean mask: is_gen
):
    """
    Create a sampler that returns a batch from a 50/50 mixture of:
      - VAE samples (via VAEWrapper.generate)
      - Real MNIST images from `real_loader`

    Returns
    -------
    sampler : callable
        sampler(bs, device, dtype=torch.float32) -> X
        If return_source=True, returns (X, is_gen_mask) where mask is (bs,) bool
    """
    real_iter = iter(real_loader)

    @torch.no_grad()
    def sampler(bs: int, device: str, dtype: torch.dtype = torch.float32):
        nonlocal real_iter

        # --- split 50/50 ---
        n_gen  = bs // 2
        n_real = bs - n_gen

        # --- grab real images (may span multiple loader batches) ---
        need = n_real
        real_chunks = []
        while need > 0:
            try:
                batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                batch = next(real_iter)

            imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            if imgs.dim() == 3:  # (N,H,W) -> (N,1,H,W)
                imgs = imgs.unsqueeze(1)
            real_chunks.append(imgs)
            need -= imgs.size(0)

        real_imgs = torch.cat(real_chunks, dim=0)[:n_real].to(device=device, dtype=dtype)

        # --- generate with VAE ---
        if n_gen > 0:
            gen_imgs = vae.generate(n_gen)            # on VAE device, in [0,1]
            if gen_imgs.dim() == 3:
                gen_imgs = gen_imgs.unsqueeze(1)      # (N,H,W) -> (N,1,H,W)
            if post is not None:
                gen_imgs = post(gen_imgs)
            gen_imgs = gen_imgs.to(device=device, dtype=dtype)
        else:
            gen_imgs = real_imgs.new_empty((0, *real_imgs.shape[1:]))

        # --- mix & shuffle ---
        X = torch.cat([gen_imgs, real_imgs], dim=0)
        is_gen = torch.cat([
            torch.ones(n_gen,  dtype=torch.bool, device=device),
            torch.zeros(n_real, dtype=torch.bool, device=device)
        ], dim=0)

        perm = torch.randperm(bs, device=device)
        X = X[perm]
        is_gen = is_gen[perm]

        return (X, is_gen) if return_source else X

    return sampler

# ---------- helpers ----------
def dcgan_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

def _freeze_bn(module: nn.Module):
    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
        module.eval()           # use running stats
        for p in module.parameters():
            p.requires_grad_(False)   # optional: also freeze affine params

def hellinger_stable(w_p, w_q, *, log_space: bool = False, 
                    eps: float = 1e-6, log_clip: float = 8.0):
    """
    Stable Hellinger term:
        E_p[w^{-1/2}] + E_q[w^{1/2}]
    Compute in log-space with clipping. If log_space=True, inputs are log w.
    """
    # if log_space:
    #     lp, lq = w_p, w_q                       # already log w
    # else:
    #     lp = (w_p.clamp_min(eps)).log()
    #     lq = (w_q.clamp_min(eps)).log()
    # lp = lp.clamp(-log_clip, log_clip)
    # lq = lq.clamp(-log_clip, log_clip)
    # return 0.5*torch.mean(torch.exp(-0.5 * lp)) + 0.5*torch.mean(torch.exp(0.5 * lq))-1
    return 0.5*torch.mean(w_p.pow(-0.5)) + 0.5*torch.mean(w_q.pow(0.5))-1

def kl_from_outputs(logw_p, logw_q):
    # E_p[-log w] + E_q[w]  with logw_* provided
    return -torch.mean(logw_p) + torch.mean(torch.exp(logw_q))

def chisq_from_outputs(w_p, w_q):
    # simple symmetric variant
    return torch.mean((w_p - 1.0) ** 2) + torch.mean(w_q ** 2)

# ---- training with minibatches ----
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
    val_q_batches: int = 4,    # how many q batches to use for val
    print_every: int = 100,    # steps
    bn_freeze_epoch: int = 5,  # freeze BN after this epoch (set <0 to disable)
    onecycle_max_lr: float = 6e-4,   # used if scheduler is None
    onecycle_pct_start: float = 0.1,
    weight_decay: float = 0.0,
):
    H, W = img_hw
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    core = dre.DREConvNet_DCGAN_MNIST(in_ch=in_ch, base=base, log_scale=log_scale)
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
    ema = None

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
                for _ in range(val_q_batches):
                    try:
                        x_pv, *_ = next(it)
                    except StopIteration:
                        break
                    bs_v = x_pv.size(0)
                    x_pv = x_pv.to(device=device, dtype=next(model.parameters()).dtype)
                    x_qv = q_sampler(bs_v, device, dtype=x_pv.dtype)
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

        # If caller passed ReduceLROnPlateau, step it per epoch with metric
        if (not use_onecycle) and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)

    return model, train_losses

# ##### for CelebA64 #######


# -------- helpers --------
def _cycle(dl: Iterable):
    """Endless iterator over a DataLoader (or any iterable)."""
    it = iter(dl)
    while True:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        yield batch

def _maybe_to_minus1_1(x: torch.Tensor) -> torch.Tensor:
    """
    If x looks like [0,1], map to [-1,1]. Heuristic check to avoid double scaling.
    """
    if x.min() >= -1.001 and x.max() <= 1.001:
        return x  # already in [-1,1] (approx)
    if x.min() >= -0.001 and x.max() <= 1.001:
        return x * 2.0 - 1.0
    return x  # leave as-is; caller can pass an explicit `post`

# -------- main factory --------
def make_q_mixed_sampler_celeba64(
    real_loader,                         # DataLoader yielding (images, *_) with images (B,3,64,64)
    gen_fn: Optional[Callable[[int, torch.device, torch.dtype], torch.Tensor]] = None,
    G: Optional[nn.Module] = None,       # If gen_fn is None, provide a module G + z_dim
    z_dim: Optional[int] = None,         # DCGAN-like latent dim (e.g., 100)
    gen_frac: float = 0.5,               # fraction from generator (0..1)
    post: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,  # final transform (e.g., lambda t: t*2-1)
    use_autocast: bool = False,          # set True if your G benefits from autocast on GPU
    channels: int = 3,                   # RGB
    size_hw: Tuple[int,int] = (64, 64),  # (H,W)
):
    """
    Returns: q_sampler(bs, device, dtype) -> torch.Tensor of shape (bs,3,64,64) in [-1,1].
    """

    assert 0.0 <= gen_frac <= 1.0, "gen_frac must be in [0,1]"
    H, W = size_hw
    assert (H, W) == (64, 64), "This sampler is tailored for 64x64."
    assert channels == 3, "CelebA-64 is RGB (3 channels)."

    real_iter = _cycle(real_loader)

    # Build a default generator function if needed (DCGAN-style z -> x)
    if gen_fn is None:
        assert (G is not None) and (z_dim is not None), "Provide either gen_fn or (G and z_dim)."
        G.eval()

        @torch.no_grad()
        def _gen_with_G(bs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
            z = torch.randn(bs, z_dim, 1, 1, device=device, dtype=dtype)
            if use_autocast and device.type == "cuda":
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    xg = G(z)
            else:
                xg = G(z)
            # Ensure dtype, shape, and range
            xg = xg.to(device=device, dtype=dtype)
            if xg.shape[1] != channels or xg.shape[-2:] != (H, W):
                # Try a safe resize only if needed; better if your G already outputs 3x64x64
                xg = torch.nn.functional.interpolate(xg, size=(H, W), mode="bilinear", align_corners=False)
            return xg
        gen_fn = _gen_with_G

    @torch.no_grad()
    def q_sampler(bs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        k_gen = int(round(gen_frac * bs))
        k_real = bs - k_gen

        # ---- generator part ----
        if k_gen > 0:
            x_g = gen_fn(k_gen, device, dtype)
        else:
            x_g = None

        # ---- real part ----
        if k_real > 0:
            real_batch = next(real_iter)
            # Support datasets that yield (images, labels, ...) or just images
            if isinstance(real_batch, (list, tuple)):
                x_p = real_batch[0]
            else:
                x_p = real_batch
            # Ensure right size/channels and dtype/device
            x_p = x_p.to(device=device, dtype=dtype)
            if x_p.shape[1] != channels or x_p.shape[-2:] != (H, W):
                x_p = torch.nn.functional.interpolate(x_p, size=(H, W), mode="bilinear", align_corners=False)
            if x_p.size(0) < k_real:
                # If we got fewer than requested (e.g., last partial batch), top up from next iter
                extra = []
                need = k_real - x_p.size(0)
                extra.append(x_p)
                while need > 0:
                    rb = next(real_iter)
                    rb = rb[0] if isinstance(rb, (list, tuple)) else rb
                    rb = rb.to(device=device, dtype=dtype)
                    if rb.shape[1] != channels or rb.shape[-2:] != (H, W):
                        rb = torch.nn.functional.interpolate(rb, size=(H, W), mode="bilinear", align_corners=False)
                    take = min(need, rb.size(0))
                    extra.append(rb[:take])
                    need -= take
                x_p = torch.cat(extra, dim=0)
            else:
                x_p = x_p[:k_real]
        else:
            x_p = None

        # ---- combine ----
        parts = [t for t in (x_g, x_p) if t is not None]
        X = torch.cat(parts, dim=0) if len(parts) == 2 else parts[0]

        # ---- range & optional post ----
        X = _maybe_to_minus1_1(X)
        if post is not None:
            X = post(X)

        # ---- shuffle to avoid block structure ----
        perm = torch.randperm(X.size(0), device=device)
        return X[perm]

    return q_sampler

# ---------- TRAINING (CelebA64) ----------
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
    val_q_batches: int = 4,    # how many q batches to use for val
    print_every: int = 100,    # steps
    bn_freeze_epoch: int = 5,  # freeze BN after this epoch (set <0 to disable)
    onecycle_max_lr: float = 6e-4,
    onecycle_pct_start: float = 0.1,
    weight_decay: float = 0.0,
):
    """
    Trains a ratio model w(x) ~ p(x)/q(x) (or its log) on CelebA-64.

    Notes:
      - p_loader MUST yield images scaled to [-1,1].
      - q_sampler(bs, device, dtype) MUST return (bs,3,64,64) in [-1,1].
      - loss_method matches your MNIST loop (Hellinger stable by default).
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
    ema = None

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

            # ----- update -----
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
            optimizer.step()
            if use_onecycle:
                scheduler.step()  # per-batch

            # ----- logging -----
            l = loss.item()
            train_losses.append(l)
            ema = l if ema is None else (0.95 * ema + 0.05 * l)
            if (step % print_every) == 0:
                lr = optimizer.param_groups[0]["lr"]
                print(f"[CelebA64] epoch {epoch:03d} step {step:06d}  loss {l:.4f}  ema {ema:.4f}  lr {lr:.2e}")
            step += 1

        # ---- optional validation on p vs q ----
        metric = torch.tensor(train_losses[-1], device=device)
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                vals = []
                it = iter(val_loader)
                for _ in range(val_q_batches):
                    try:
                        x_pv, *_ = next(it)
                    except StopIteration:
                        break
                    bs_v = x_pv.size(0)
                    x_pv = x_pv.to(device=device, dtype=dtype)
                    x_qv = q_sampler(bs_v, device, dtype=dtype)
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

        # If caller passed ReduceLROnPlateau, step it per epoch with metric
        if (not use_onecycle) and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)

    return model, train_losses