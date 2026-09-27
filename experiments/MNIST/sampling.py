"""Historical MNIST generator/real mixtures used as RDR denominators.

Sampling proportions and preprocessing are preserved during code relocation.
"""

import torch
from torch import nn
from typing import Callable, Optional, Tuple, Iterable

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
