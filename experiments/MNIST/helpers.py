
# use pretrained models in https://github.com/csinva/gan-vae-pretrained-pytorch
import importlib
import torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
import math


from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Union

Tensor = torch.Tensor
DeviceLike = Union[str, torch.device]

# ---- Core loaders ----
def load_mnist(
    root,
    train: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None,
):
    """
    Load MNIST from a given root folder.
    root: str | Path  -> path where MNIST will be stored/read.
    """
    root = Path(root)
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])  # -> [0,1], (1,28,28)

    try:
        ds = datasets.MNIST(str(root), train=train, download=download, transform=transform)
    except Exception as e:
        # Fallback for environments that block downloads (HPC, etc.)
        print(f"Download failed with: {e}\nRetrying with download=False...")
        ds = datasets.MNIST(str(root), train=train, download=False, transform=transform)
    return ds

# ---- Sampling helpers ----
def sample_random(ds, n: int = 36, seed: Optional[int] = 123):
    """Sample n random images (and labels) from a dataset."""
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n]
    imgs = torch.stack([ds[i][0] for i in idx])  # (n,1,28,28)
    labs = torch.tensor([ds[i][1] for i in idx])
    return imgs, labs

def build_label_index(ds) -> Dict[int, torch.Tensor]:
    """Precompute indices for each digit to speed up per-class sampling."""
    labels = torch.tensor([y for _, y in ds])
    return {d: (labels == d).nonzero(as_tuple=True)[0] for d in range(10)}

def sample_by_digit(
    ds,
    digit: int,
    n: int = 6,
    seed: Optional[int] = 123,
    idx_map: Optional[Dict[int, torch.Tensor]] = None,
):
    """Sample n images of a given digit."""
    if idx_map is None:
        idx_map = build_label_index(ds)
    idx_all = idx_map[digit]
    if idx_all.numel() < n:
        raise ValueError(f"Requested {n} samples for digit {digit}, but only {idx_all.numel()} available.")
    g = torch.Generator()
    if seed is not None:
        g.manual_seed((seed + digit) if seed is not None else None)
    pick = idx_all[torch.randperm(idx_all.numel(), generator=g)[:n]]
    imgs = torch.stack([ds[i][0] for i in pick])
    labs = torch.tensor([ds[i][1] for i in pick])
    return imgs, labs

def to_vectors(imgs: torch.Tensor) -> torch.Tensor:
    """Flatten images to (n, 784) for DRE."""
    return imgs.view(imgs.size(0), -1).contiguous()

# ---------------------------
# ---- Visualization ----
# ---------------------------

def plot_mnist_samples(x, idx, nrow=5, ncol=5, figsize=(6,6)):
    """
    Visualize selected MNIST-like samples from a flattened (N,784) dataset.
    Works with torch.Tensor on GPU or CPU.
    """
    import matplotlib.pyplot as plt

    # Ensure idx is a plain list of indices
    if torch.is_tensor(idx):
        idx = idx.cpu().numpy()

    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    axes = axes.flatten()

    for ax, i in zip(axes, idx[:nrow*ncol]):
        img = x[i].detach().cpu().numpy().reshape(28,28)  # <-- move to CPU, numpy
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    # hide unused axes
    for ax in axes[len(idx):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_extremes(x, extremes, nrow=1, ncol=5, figsize=(15, 5)):
    """
    Plot three panels of MNIST samples: smallest, closest to 1, largest.
    
    Parameters
    ----------
    x : torch.Tensor
        Dataset of shape (N, 784) or (N, 1, 28, 28).
    extremes : dict
        Output of find_extreme_indices (dict with keys "smallest", "closest", "largest").
    nrow, ncol : int
        Grid size for each panel.
    figsize : tuple
        Figure size.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, nrow * ncol, figsize=figsize)
    
    # If axes is 1D array, ensure it’s iterable in the right shape
    if axes.ndim == 1:
        axes = axes.reshape(3, -1)
    
    categories = ["smallest", "closest", "largest"]
    
    for row, cat in enumerate(categories):
        _, idx = extremes[cat]  # values, indices
        if torch.is_tensor(idx):
            idx = idx.cpu().numpy()
            
        for col, i in enumerate(idx[:nrow * ncol]):
            img = x[i].detach().cpu().numpy().reshape(28, 28)
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        
        # Label each row on the left
        axes[row, 0].set_ylabel(cat, fontsize=12, rotation=90, labelpad=10)

    plt.tight_layout()
    plt.show()


def _extract_img(xi, img_hw=28):
    """Return a (H,W) numpy image from xi shaped (784), (1,H,W), or (H,W)."""
    if torch.is_tensor(xi):
        xi = xi.detach().cpu()
    if xi.ndim == 1:          # (784,)
        return xi.view(img_hw, img_hw).numpy()
    if xi.ndim == 2:          # (H,W)
        return xi.numpy()
    if xi.ndim == 3:          # (1,H,W) or (C,H,W) but assume C==1
        return xi.squeeze(0).numpy()
    raise ValueError(f"Unsupported image shape {tuple(xi.shape)}")

def plot_extremes_three_panels(
    x,
    extremes,
    *,
    img_hw=28,
    ncols=(5, 5, 5),          # columns in each panel: (smallest, close, largest)
    figsize=(18, 6),
    cmap="gray",
    vmin=0.0,
    vmax=1.0,
    panel_titles=("Smallest", "Close-to-1", "Largest"),
    super_title_prefix="Generated"
):
    """
    x : Tensor of images with shape (N, 784) or (N, 1, H, W) or (N, H, W)
    extremes : dict from find_extreme_indices with keys "smallest","closest","largest"
    ncols : 3-tuple of ints, number of columns in each panel (left→right)
    """
    cats = ["smallest", "closest", "largest"]
    assert len(ncols) == 3, "ncols must be a 3-tuple"
    assert all(c in extremes for c in cats), "extremes needs smallest/closest/largest"

    # Prepare values and indices per category
    vals_idx = []
    for c in cats:
        vals, idx = extremes[c]
        if torch.is_tensor(idx):
            idx = idx.detach().cpu()
        vals_idx.append((vals.detach().cpu(), idx))

    # Figure + outer grid (width proportional to desired columns)
    # if ax is None:
    #     fig, ax = plt.subplots(1, 3, figsize=figsize, gridspec_kw={"width_ratios": ncols})
    # else:
    #     fig = ax.get_figure()
    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(1, 3, width_ratios=list(ncols), wspace=0.25)

    for p, (cat, (vals, idx)) in enumerate(zip(cats, vals_idx)):
        # compute rows from requested columns
        num = idx.numel() if torch.is_tensor(idx) else len(idx)
        cols = ncols[p]
        rows = max(1, math.ceil(num / cols))

        # inner grid for this panel
        inner = outer[p].subgridspec(rows, cols, wspace=0.02, hspace=0.02)

        # panel title with value range
        vmin_g = float(vals.min().item()) if num > 0 else float("nan")
        vmax_g = float(vals.max().item()) if num > 0 else float("nan")
        panel_title = (
            f"{super_title_prefix}: {panel_titles[p]} scores\n"
            f"(min={vmin_g:.3g}, max={vmax_g:.3g})"
        )
        # Add an invisible axis for the panel title (spans the panel)
        ax_title = fig.add_subplot(outer[p])
        ax_title.set_title(panel_title, fontsize=12, y=1.02)
        ax_title.axis("off")

        # plot each image cell
        for k in range(rows * cols):
            r, c = divmod(k, cols)
            ax = fig.add_subplot(inner[r, c])
            if k < num:
                i = int(idx[k])
                img = _extract_img(x[i], img_hw=img_hw)
                ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")

    plt.tight_layout()
    # return fig, ax
    plt.show()


def show_grid(
    imgs: torch.Tensor,
    labs: Optional[torch.Tensor] = None,
    title: str = "MNIST samples",
    nrow: int = 6,
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Display a grid of images. Optionally save to file.
    imgs: (N,1,28,28) in [0,1]
    """
    grid = utils.make_grid(imgs, nrow=nrow, padding=2)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    plt.axis("off")
    if labs is not None:
        title = f"{title}\nlabels: {labs.tolist()}"
    plt.title(title)
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
def _unflatten_if_needed(x: torch.Tensor) -> torch.Tensor:
    """Return (N,1,28,28) in [0,1]."""
    if x.ndim == 2 and x.size(1) == 784:
        x = x.view(-1, 1, 28, 28)
    elif x.ndim == 3:
        x = x.unsqueeze(1)  # (N,28,28) -> (N,1,28,28)
    return x

def _to_01(x: torch.Tensor) -> torch.Tensor:
    """Best-effort clamp/scale to [0,1]. If already in [0,1], this is a no-op."""
    x = x.detach()
    if x.min() < 0.0 or x.max() > 1.0:
        # assume possibly tanh output
        x = (x + 1.0) / 2.0
    return x.clamp(0, 1)

def visualize_pq(
    x_p: torch.Tensor,
    x_q: torch.Tensor,
    y_p: torch.Tensor = None,
    n_show: int = 36,
    nrow: int = 6,
    seed: int = 123,
    save_prefix: str = "pq"
):
    g = torch.Generator()
    g.manual_seed(seed)

    # convert to image tensors in [0,1]
    xp = _to_01(_unflatten_if_needed(x_p))
    xq = _to_01(_unflatten_if_needed(x_q))

    # choose a random subset (or the first n if you prefer)
    idx_p = torch.randperm(xp.size(0), generator=g)[:min(n_show, xp.size(0))]
    idx_q = torch.randperm(xq.size(0), generator=g)[:min(n_show, xq.size(0))]

    imgs_p = xp[idx_p].cpu()
    imgs_q = xq[idx_q].cpu()
    labs_p = y_p[idx_p].cpu() if y_p is not None else None

    # visualize
    show_grid(imgs_p, labs_p,
              title=f"Real MNIST (p): {imgs_p.size(0)}",
              nrow=nrow, save_path=f"{save_prefix}_p.png")

    show_grid(imgs_q, None,
              title=f"Generator (q): {imgs_q.size(0)}",
              nrow=nrow, save_path=f"{save_prefix}_q.png")

# ---------------------------
# REAL MNIST SAMPLER (p)
# ---------------------------

def _build_label_index(ds) -> Dict[int, Tensor]:
    labels = torch.tensor([y for _, y in ds])
    return {d: (labels == d).nonzero(as_tuple=True)[0] for d in range(10)}

def get_mnist_real_sampler(
    root: Union[str, Path],
    split: str = "train",
    device: DeviceLike = "cpu",
    flatten: bool = True,
    download: bool = True,
    transform: Optional[transforms.Compose] = None,
) -> Callable[..., Union[Tensor, Tuple[Tensor, Tensor]]]:
    """
    Returns a closure `sample_real(n, seed=None, digits=None, per_digit=None, return_labels=False)`
      - root: where MNIST lives
      - split: "train" or "test"
      - device: target device for returned tensors
      - flatten: if True, returns (n, 784); else (n,1,28,28)
      - transform: defaults to ToTensor() -> [0,1]
    """
    root = Path(root)
    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    train_flag = (split.lower() == "train")
    try:
        ds = datasets.MNIST(str(root), train=train_flag, download=download, transform=transform)
    except Exception as e:
        print(f"[MNIST] Download failed with: {e}\nRetrying with download=False...")
        ds = datasets.MNIST(str(root), train=train_flag, download=False, transform=transform)

    idx_map = _build_label_index(ds)

    def sample_real(
        n: int,
        seed: Optional[int] = None,
        digits: Optional[List[int]] = None,
        per_digit: Optional[int] = None,
        return_labels: bool = False,
        dtype: Optional[torch.dtype] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        - digits=None: sample uniformly at random from the whole split
        - digits=[...]: sample only from those digits
          * if per_digit is set, takes exactly per_digit per listed digit (n is ignored)
          * else distributes ~n/len(digits) per digit (last chunk gets the remainder)
        """
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)

        if digits is None:
            # random selection over the entire dataset
            idx = torch.randperm(len(ds), generator=g)[:n]
            imgs = torch.stack([ds[i][0] for i in idx])
            labs = torch.tensor([ds[i][1] for i in idx])
        else:
            digits = list(digits)
            if per_digit is None:
                base = n // len(digits)
                remainder = n - base * len(digits)
                counts = [base + (1 if i < remainder else 0) for i in range(len(digits))]
            else:
                counts = [per_digit] * len(digits)

            imgs_list, labs_list = [], []
            for i, d in enumerate(digits):
                all_idx = idx_map[d]
                if all_idx.numel() < counts[i]:
                    raise ValueError(f"Need {counts[i]} of digit {d}, but only {all_idx.numel()} available.")
                pick = all_idx[torch.randperm(all_idx.numel(), generator=g)[:counts[i]]]
                imgs_list.append(torch.stack([ds[j][0] for j in pick]))
                labs_list.append(torch.tensor([ds[j][1] for j in pick]))
            imgs = torch.cat(imgs_list, 0)
            labs = torch.cat(labs_list, 0)

        if dtype is not None:
            imgs = imgs.to(dtype)
        imgs = imgs.to(device)
        labs = labs.to(device)

        if flatten:
            imgs = imgs.view(imgs.size(0), -1).contiguous()  # (n, 784)

        return (imgs, labs) if return_labels else imgs

    return sample_real


# ---------------------------
# PRETRAINED GENERATOR SAMPLER (q)
# ---------------------------

def get_generator_sampler(
    generator: nn.Module,
    z_dim: int,
    device: DeviceLike = "cpu",
    z_type: str = "noise_4d",
    out_size: Tuple[int, int] = (28, 28),
    generator_output_range: str = "[-1,1]",  # "[-1,1]" | "[0,1]" | "logits"
    return_range: str = "[0,1]",             # "[0,1]" or "[-1,1]"
    flatten: bool = True,
    channels: int = 1,
) -> Callable[[int, Optional[int], Optional[torch.dtype]], Tensor]:
    """
    Wrap ANY pretrained generator as a callable:
      sample_q = get_generator_sampler(G, z_dim=100, z_type="noise_4d", ...)
      x_q = sample_q(n=5000, seed=0)  # -> shape (n,784) if flatten=True else (n,1,28,28)

    Args
    ----
    generator: your nn.Module with a forward(z) -> image tensor
    z_dim: latent dimensionality
    z_type: "noise_4d" for ConvTranspose GANs (z: (n,z_dim,1,1)), "noise_2d" for MLP/VAEs (z: (n,z_dim))
    out_size: force output spatial size (resize if needed)
    generator_output_range: range that your GEN produces
    return_range: desired output range
    flatten: return flattened features if True
    channels: expected channels (1 for MNIST)

    Notes
    -----
    - If your generator already outputs 28x28 MNIST in [0,1], set generator_output_range='[0,1]'.
    - For VAEs that output logits, set generator_output_range='logits' (we'll apply sigmoid).
    """
    device = torch.device(device)
    generator = generator.to(device).eval()

    def _make_z(n: int, g: torch.Generator) -> Tensor:
        if z_type == "noise_4d":
            return torch.randn(n, z_dim, 1, 1, device=device, generator=g)
        elif z_type == "noise_2d":
            return torch.randn(n, z_dim, device=device, generator=g)
        else:
            raise ValueError("z_type must be 'noise_4d' or 'noise_2d'.")

    @torch.no_grad()
    def sample_q(n: int, seed: Optional[int] = None, dtype: Optional[torch.dtype] = None) -> Tensor:
        g = torch.Generator(device=device)
        if seed is not None:
            g.manual_seed(seed)
        z = _make_z(n, g)

        x = generator(z)
        # Normalize to [0,1] first
        if generator_output_range == "[-1,1]":
            x = (x + 1.0) / 2.0
        elif generator_output_range == "logits":
            x = x.sigmoid()
        elif generator_output_range == "[0,1]":
            pass
        else:
            raise ValueError("Unsupported generator_output_range.")

        # Ensure channel dimension is present
        if x.ndim == 3:  # (n,H,W)
            x = x.unsqueeze(1)  # -> (n,1,H,W)

        # Force channels if needed (rarely needed for MNIST; asserts here)
        if x.size(1) != channels:
            raise ValueError(f"Generator returned {x.size(1)} channels, expected {channels}.")

        # Resize to desired size if needed
        if out_size is not None and (x.size(-2) != out_size[0] or x.size(-1) != out_size[1]):
            x = F.interpolate(x, size=out_size, mode="bilinear", align_corners=False)

        # Clamp and convert to requested output range
        x = x.clamp(0, 1)
        if return_range == "[-1,1]":
            x = x * 2.0 - 1.0
        elif return_range == "[0,1]":
            pass
        else:
            raise ValueError("Unsupported return_range.")

        if dtype is not None:
            x = x.to(dtype)

        if flatten:
            x = x.view(x.size(0), -1).contiguous()  # (n, out_dim)

        return x

    return sample_q


# --------------------------------------------------------
# OPTIONAL: Convenience builder for a common MNIST DCGAN
# --------------------------------------------------------

class DCGAN_G_MNIST(nn.Module):
    """
    Matches csinva's MNIST DCGAN netG_epoch_99.pth
    Blocks: 100→512→256→128→64→1, final convT (k=1, pad=2) to get 28x28.
    """
    def __init__(self, nz=100, n_planes=64, nc=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, n_planes * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_planes * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_planes * 8, n_planes * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_planes * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_planes * 4, n_planes * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_planes * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(n_planes * 2, n_planes, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_planes),
            nn.ReLU(True),

            # 32x32 -> 28x28
            nn.ConvTranspose2d(n_planes, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh(),  # outputs in [-1,1]
        )

    def forward(self, z):
        if z.ndim == 2:  # allow (N, nz) too
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.main(z)

def build_dcgan28(weights_path, nz=100, n_planes=64, nc=1, device="cpu"):
    G = DCGAN_G_MNIST(nz=nz, n_planes=n_planes, nc=nc).to(device).eval()
    state = torch.load(str(weights_path), map_location=device)
    G.load_state_dict(state, strict=True)
    return G, nz


# --------------------------------------------------------
# OPTIONAL: VAE
# --------------------------------------------------------

class VAEWrapper:
    """
    Lightweight wrapper around the repo's MNIST VAE.

    Usage
    -----
    vae = VAEWrapper.from_repo(weights="mnist_vae/weights/vae.pt")  # adjust path/filename
    imgs = vae.generate(n=64)                  # (64, 1, 28, 28) in [0,1]
    vae.save_grid(imgs, "mnist_vae_samples.png", nrow=8)
    """
    def __init__(self, model: torch.nn.Module, latent_dim: int, device: str):
        self.model = model.eval()
        self.latent_dim = int(latent_dim)
        self.device = device

    # ---------- constructors ----------

    @staticmethod
    def _infer_latent_dim(model: torch.nn.Module, fallback: Optional[int] = None) -> int:
        # Try common attribute names seen across VAE codebases
        candidates = ["latent_dim", "z_dim", "nz", "dim_z", "n_z", "zd"]
        for name in candidates:
            if hasattr(model, name):
                val = getattr(model, name)
                try:
                    return int(val)
                except Exception:
                    pass
        if fallback is not None:
            return int(fallback)
        # Last resort: try a tiny forward through encode to infer size
        # We assume MNIST (1×28×28); many repos accept (N,1,28,28)
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28, device=next(model.parameters()).device)
            if hasattr(model, "encode"):
                enc_out = model.encode(x)
                # enc_out might be (mu, logvar) or a single tensor
                if isinstance(enc_out, (tuple, list)) and len(enc_out) >= 1:
                    mu = enc_out[0]
                else:
                    mu = enc_out
                return int(mu.shape[-1])
        raise RuntimeError(
            "Could not infer latent dimension; pass latent_dim=... explicitly."
        )

    @classmethod
    def from_repo(
        cls,
        weights: Union[str, Path],
        *,
        module_path: str = "mnist_vae",   # repo subfolder to import from
        class_name: str = "VAE",          # default class name used in many simple VAEs
        model_ctor: Optional[Callable[[], torch.nn.Module]] = None,
        latent_dim: Optional[int] = None,
        device: Optional[str] = None,
    ) -> "VAEWrapper":
        """
        Load the pre-trained MNIST VAE from the repo.

        Parameters
        ----------
        weights : path to the checkpoint (state_dict or full model)
        module_path : python module path for the VAE code in the repo (default: 'mnist_vae')
        class_name : class to instantiate (default: 'VAE')
        model_ctor : optional zero-arg constructor if the class/module names differ
        latent_dim : optional override if it can't be inferred from the model
        device : 'cuda' or 'cpu' (auto-detect if None)
        """
        weights = Path(weights)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Instantiate model
        if model_ctor is not None:
            model = model_ctor()
        else:
            # Dynamically import e.g. mnist_vae.vae and fetch VAE class
            # If the repo uses a different filename, change 'vae' below to match.
            try:
                mod = importlib.import_module(f"{module_path}.vae")
            except ModuleNotFoundError:
                # fallbacks: common file names
                for fname in ("model", "models", "net", "network"):
                    try:
                        mod = importlib.import_module(f"{module_path}.{fname}")
                        break
                    except ModuleNotFoundError:
                        mod = None
                if mod is None:
                    raise
            VAEClass = getattr(mod, class_name)
            model = VAEClass()

        model = model.to(device)

        # Load weights (support state_dict or full-model checkpoints)
        ckpt = torch.load(str(weights), map_location=device)
        try:
            model.load_state_dict(ckpt)
        except Exception:
            # try common nesting patterns
            if isinstance(ckpt, dict):
                for key in ("state_dict", "model_state", "model", "net", "vae"):
                    if key in ckpt and isinstance(ckpt[key], dict):
                        model.load_state_dict(ckpt[key])
                        break
                else:
                    # maybe it's a full serialized model:
                    if hasattr(ckpt, "state_dict"):
                        model.load_state_dict(ckpt.state_dict())
                    else:
                        raise

        zdim = cls._infer_latent_dim(model, fallback=latent_dim)
        return cls(model, zdim, device)

    # ---------- generation ----------

    @torch.no_grad()
    def sample_latent(self, n: int) -> Tensor:
        return torch.randn(n, self.latent_dim, device=self.device)

    @torch.no_grad()
    def decode(self, z, clamp: bool = True):
        # decode from the model
        x = self.model.decode(z) if hasattr(self.model, "decode") else self.model(z)

        # reshape if the decoder returns flat 784 vectors
        if x.dim() == 2 and x.shape[-1] == 784:
            x = x.view(-1, 1, 28, 28)

        # only apply sigmoid if it looks like logits (outside [0,1])
        try:
            x_min = float(x.min())
            x_max = float(x.max())
            if x_min < -1e-6 or x_max > 1 + 1e-6:
                x = x.sigmoid()
        except Exception:
            pass

        return x.clamp(0, 1) if clamp else x

    @torch.no_grad()
    def generate(self, n: int = 64) -> Tensor:
        """
        Sample n random images from the prior N(0, I) in latent space.
        Output shape: (n, 1, 28, 28)
        """
        z = self.sample_latent(n)
        return self.decode(z)

    @staticmethod
    def save_grid(imgs: Tensor, path: Union[str, Path], nrow: int = 8) -> None:
        path = Path(path)
        utils.save_image(imgs, path, nrow=nrow)

def find_extreme_indices(tensor: torch.Tensor, top_k: int = 5, target: float = 1.0):
    """
    Find top-k largest, smallest, and closest-to-target values in a flattened tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor (any shape).
    top_k : int
        Number of values to return for each category.
    target : float
        Value to measure closeness against (default = 1.0).

    Returns
    -------
    dict with keys {"largest", "smallest", "closest"}.
    Each value is a tuple (values, indices).
    """
    flat = tensor.view(-1)

    # Largest
    largest_vals, largest_idx = torch.topk(flat, top_k)

    # Smallest
    smallest_vals, smallest_idx = torch.topk(-flat, top_k)
    smallest_vals = -smallest_vals  # restore original sign

    # Closest to target
    dist = torch.abs(flat - target)
    closest_vals, closest_idx = torch.topk(-dist, top_k)
    closest_vals = flat[closest_idx]  # recover actual values

    return {
        "largest": (largest_vals, largest_idx),
        "smallest": (smallest_vals, smallest_idx),
        "closest": (closest_vals, closest_idx),
    }