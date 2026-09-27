"""Historical CelebA image mixtures, with unchanged range handling.

The legacy range heuristic is preserved; callers can supply explicit postprocessing.
"""

import torch
from torch import nn
from typing import Callable, Optional, Tuple, Iterable

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
