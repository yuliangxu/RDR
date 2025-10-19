
import io, re
import os, random, torch, math
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Optional, Literal, Union, Sequence, Tuple, Dict
from huggingface_hub import hf_hub_download
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
from PIL import Image
from torchvision.utils import make_grid
import bisect
from tqdm import tqdm
import statsmodels.api as sm


# ---------------------------
# ---- load real sample  ----
# ---------------------------
def denormalize(images: torch.Tensor) -> torch.Tensor:
    """
    Undo CelebA normalization ([-1,1] → [0,1]).
    """
    return images * 0.5 + 0.5

def save_batch_grid(
    images: torch.Tensor,
    outpath: str = "celeba64_grid.png",
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
):
    """
    Save a grid of images to file.
    
    Parameters
    ----------
    images : torch.Tensor
        Batch of images [B, C, H, W] in [-1,1] if normalize=True.
    outpath : str
        Output filename.
    nrow : int
        Number of images per row.
    padding : int
        Padding between images in the grid.
    normalize : bool
        If True, denormalize from [-1,1] → [0,1] before saving.
    """
    if normalize:
        images = denormalize(images).clamp(0,1)

    grid = vutils.make_grid(images, nrow=nrow, padding=padding)
    vutils.save_image(grid, outpath)
    print(f"Saved grid to {outpath}")


# --- optional helper from earlier ---
def _normalize_idx(idx, N, device):
    if isinstance(idx, (int, slice)):
        return idx
    if torch.is_tensor(idx):
        if idx.dtype == torch.bool:
            if idx.numel() != N:
                raise ValueError(f"Bool mask length {idx.numel()} != batch size {N}")
            return idx.to(device=device)
        return idx.to(device=device, dtype=torch.long).view(-1)
    if isinstance(idx, (list, tuple)):
        if all(torch.is_tensor(i) for i in idx):
            idx = torch.stack([i.to(dtype=torch.long, device=device).view(-1) for i in idx]).view(-1)
            return idx
        return torch.as_tensor(idx, device=device, dtype=torch.long).view(-1)
    raise TypeError(f"Unsupported idx type: {type(idx)}")

def show_batch(
    images: torch.Tensor,
    nrows: int = 4,
    ncols: int = 8,
    padding: int = 2,
    normalize: bool = True,  # expects a user-defined `denormalize` that maps to [0,1]
    idx: Optional[Union[int, slice, Sequence[int], torch.Tensor]] = None,
    # --- plotting extras ---
    plot: bool = False,
    title_fontsize: int = 12, 
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    dpi: int = 120,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Build an nrows × ncols grid (and optionally plot it).
    If idx is empty, shows an empty (blank) grid.
    If fewer than nrows*ncols images are selected, pads with blanks.

    Returns
    -------
    grid : (C,H,W) torch.Tensor in [0,1] if normalize=True
    (fig, ax) is also returned when plot=True.
    """
    def _infer_chw(imgs: torch.Tensor) -> Tuple[int,int,int]:
        if imgs.ndim == 4:
            return int(imgs.shape[1]), int(imgs.shape[2]), int(imgs.shape[3])
        elif imgs.ndim == 3:
            return int(imgs.shape[0]), int(imgs.shape[1]), int(imgs.shape[2])
        else:
            raise ValueError(f"images must be (N,C,H,W) or (C,H,W); got shape {tuple(imgs.shape)}")

    x = images
    # --- apply selection ---
    if idx is not None:
        try:
            idx_norm = _normalize_idx(idx, N=x.size(0), device=x.device)
            x = x[idx_norm]
        except NameError:
            x = x[idx]

    if x.ndim == 3:  # (C,H,W) -> (1,C,H,W)
        x = x.unsqueeze(0)

    total_slots = int(nrows) * int(ncols)
    C, H, W = _infer_chw(images)

    # --- empty selection → blank grid ---
    if x.numel() == 0 or x.size(0) == 0:
        x = torch.ones((total_slots, C, H, W), device=images.device, dtype=torch.float32)
        grid = vutils.make_grid(x, nrow=ncols, padding=padding, pad_value=1.0)
        if not plot:
            return grid

        img = grid.detach().float().cpu()
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        else:
            fig = ax.figure
        if img.size(0) == 1:
            ax.imshow(img.squeeze(0), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(img.permute(1, 2, 0))
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=title_fontsize)
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        if show:
            plt.show()
        return grid, (fig, ax)

    # --- optional de-normalization ---
    if normalize:
        try:
            x = denormalize(x).clamp(0, 1)
        except NameError:
            x = x.detach().float().clamp(0, 1)

    # --- pad/truncate to nrows*ncols ---
    n_sel = x.size(0)
    if n_sel < total_slots:
        pad = torch.ones((total_slots - n_sel, x.size(1), x.size(2), x.size(3)),
                         device=x.device, dtype=x.dtype)
        x = torch.cat([x, pad], dim=0)
    elif n_sel > total_slots:
        x = x[:total_slots]

    # --- grid ---
    grid = vutils.make_grid(x, nrow=ncols, padding=padding, pad_value=1.0)

    if not plot:
        return grid

    img = grid.detach().float().cpu()
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if img.size(0) == 1:
        ax.imshow(img.squeeze(0), cmap="gray", vmin=0, vmax=1)
    else:
        ax.imshow(img.permute(1, 2, 0))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    if show:
        plt.show()

    return grid, (fig, ax)
# ---------------------------
# ---- pretrained DCGAN  ----
# ---------------------------
# ---- Optional HF import (we can still run fully offline if absent) ----
try:
    from huggingface_hub import hf_hub_download
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False
# --- The DCGAN Generator architecture (from the HF repo's dcgan.py) ---
# Ref: https://huggingface.co/hussamalafandi/DCGAN_CelebA (Generator uses nz=100, ngf=64, nc=3)
# pretrained_gan_celeba64.py
import os, shutil, torch, torch.nn as nn
from typing import List, Optional

# ---- Optional HF import (we can still run fully offline if absent) ----
try:
    from huggingface_hub import hf_hub_download
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False

# ---- DCGAN Generator (CelebA 64x64; nz=100, ngf=64, nc=3) ----
class DCGANGenerator(nn.Module):
    def __init__(self, nz: int = 100, ngf: int = 64, nc: int = 3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),  # -> [-1,1], [B,3,64,64]
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.main(z)

# ---- Resolve checkpoint path with local-first logic ----
def _resolve_checkpoint_path(
    repo_id: str,
    filename: str = "generator.pth",
    local_dir: Optional[str] = None,
    allow_download: bool = True,
    cache_dir: Optional[str] = None,
) -> str:
    """
    1) If local_dir/filename exists, return it.
    2) Else if allow_download and HF is available, download to HF cache, then copy into local_dir and return that.
    3) Else raise with a helpful message.
    """
    local_path = None
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        if os.path.isfile(local_path):
            return local_path

    if not allow_download:
        raise FileNotFoundError(
            f"Checkpoint not found locally at {local_path}. Set allow_download=True or place the file there."
        )

    if not _HF_AVAILABLE:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it (`pip install huggingface_hub`) "
            f"or place {filename} under {local_dir}."
        )

    # Download to HF cache (~/.cache/huggingface/hub by default)
    ckpt_cache_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    if local_dir:
        if not os.path.isfile(local_path):
            shutil.copy2(ckpt_cache_path, local_path)
        return local_path
    return ckpt_cache_path

# ---- Public loader: never re-download if local copy exists ----
def load_pretrained_dcgan_celeba64(
    device: str = "cuda",
    repo_id: str = "hussamalafandi/DCGAN_CelebA",
    filename: str = "generator.pth",
    local_dir: Optional[str] = None,   # e.g., "/hpc/group/mastatlab/yx306/pretrained/DCGAN_CelebA"
    allow_download: bool = True,
    nz: int = 100,
    ngf: int = 64,
    nc: int = 3,
) -> DCGANGenerator:
    """
    Loads the pretrained generator. If local_dir/filename exists, uses it.
    Otherwise downloads once and persists it to local_dir for future runs.
    """
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    G = DCGANGenerator(nz=nz, ngf=ngf, nc=nc).to(dev).eval()

    ckpt_path = _resolve_checkpoint_path(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        allow_download=allow_download,
    )
    state = torch.load(ckpt_path, map_location=dev)
    G.load_state_dict(state)
    return G

# ---- Sampling helpers ----
@torch.no_grad()
def sample_latents(
    batch_size: int,
    nz: int = 100,
    device: Optional[torch.device] = None,
    dist: Literal["normal","uniform"] = "normal",
) -> torch.Tensor:
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dist == "normal":
        z = torch.randn(batch_size, nz, 1, 1, device=dev)
    else:  # "uniform"
        z = torch.rand(batch_size, nz, 1, 1, device=dev) * 2.0 - 1.0
    return z

@torch.no_grad()
def generate_images(
    G: nn.Module,
    z: Optional[torch.Tensor] = None,
    batch_size: int = 64,
    nz: int = 100,
    latent_dist: Literal["normal","uniform"] = "normal",
    to_cpu: bool = True,
) -> torch.Tensor:
    """
    Return images in [-1,1], shape [B,3,64,64]. Make sure G.eval() is set.
    """
    G = G.eval()  # important for BatchNorm
    dev = next(G.parameters()).device
    if z is None:
        z = sample_latents(batch_size=batch_size, nz=nz, device=dev, dist=latent_dist)
    imgs = G(z)  # expect tanh output in [-1,1]
    return imgs.cpu() if to_cpu else imgs

@torch.no_grad()
def generate_from_seeds(G: nn.Module, seeds: List[int], nz: int = 100) -> torch.Tensor:
    dev = next(G.parameters()).device
    outs = []
    for s in seeds:
        g = torch.Generator(device=dev).manual_seed(s)
        z = torch.randn(1, nz, 1, 1, device=dev, generator=g)
        outs.append(G(z))
    return torch.cat(outs, dim=0).cpu()

# #### covariate association #####

# Optional libs: install if missing (pip install statsmodels scikit-learn)
from sklearn.linear_model import LogisticRegression, RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.cross_decomposition import CCA
import statsmodels.api as sm

# ---------- data prep ----------
def _to_numpy_1d(x):
    if torch.is_tensor(x):
        x = x.detach().view(-1).float().cpu().numpy()
    return np.asarray(x).reshape(-1)

def _to_numpy_2d(x):
    if torch.is_tensor(x):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x)
    return x

def prepare_gp_attrs(g_p, attrs):
    """
    g_p: (N,) tensor or array  (ratio or log-ratio per image)
    attrs: (N,40) tensor/array in {-1,+1} or {0,1}
    Returns gp (N,), A01 (N,40 in {0,1})
    """
    gp = _to_numpy_1d(g_p)
    A = _to_numpy_2d(attrs)
    # map {-1,+1} -> {0,1} if needed
    uniq = np.unique(A)
    if set(uniq.tolist()) <= {-1, 1}:
        A01 = (A == 1).astype(np.float32)
    else:
        A01 = A.astype(np.float32)
    # drop non-finite rows if any
    mask = np.isfinite(gp)
    if not mask.all():
        gp, A01 = gp[mask], A01[mask]
    return gp, A01



def run_linear(gp, A01, attr_names=None, print_summary=False, robust=None):
    """
    OLS: g_p ~ attributes (+ intercept)
    Returns a DataFrame sorted by p-value (ascending), indexed by attribute labels.
    
    Parameters
    ----------
    gp : array-like (N,)
        Per-image scalar (e.g., ratio or log-ratio).
    A01 : array-like (N, K)
        Attribute matrix in {0,1} (or any numeric). K should be 40 for CelebA.
    attr_names : list[str] or None
        Row labels for attributes (length K). If None, uses x1..xK.
    print_summary : bool
        If True, also print the statsmodels summary().
    robust : None or str
        If not None, use robust covariance. E.g., "HC3", "HC1".
    
    Returns
    -------
    df_ranked : pd.DataFrame with columns:
        ['coef', 'se', 't', 'pvalue', 'ci_low', 'ci_high']
        Sorted by 'pvalue' ascending, indexed by attribute labels.
    results : statsmodels RegressionResults
    """
    gp = np.asarray(gp).reshape(-1)
    A01 = np.asarray(A01)
    N, K = A01.shape

    if attr_names is None:
        attr_names = [f"x{j+1}" for j in range(K)]
    else:
        assert len(attr_names) == K, "attr_names length must match number of columns in A01"

    # Add intercept
    X = sm.add_constant(A01, has_constant='add')

    # Fit
    model = sm.OLS(gp, X)
    if robust is None:
        results = model.fit()
    else:
        results = model.fit(cov_type=robust)

    if print_summary:
        print(results.summary())

    # Pull params (skip intercept at index 0)
    params = results.params[1:]
    bse    = results.bse[1:]
    tvals  = results.tvalues[1:]
    pvals  = results.pvalues[1:]
    ci     = results.conf_int()[1:]  # (K,2)

    df = pd.DataFrame({
        "coef":   params,
        "se":     bse,
        "t":      tvals,
        "pvalue": pvals,
        "ci_low": ci[:, 0],
        "ci_high":ci[:, 1],
    }, index=attr_names)

    # Sort by p-value ascending
    df_ranked = df.sort_values("pvalue", ascending=True)
    return df_ranked, results

def run_beta_on_02(gp, A01, upper = 2.5, attr_names=None, print_summary=False, robust=None, eps=1e-6):
    """
    Beta (or fractional logit) regression for a bounded outcome in [0,2].
    We rescale to (0,1) and fit a mean submodel with a logit link.

    Parameters
    ----------
    gp : array-like (N,)
        Outcome bounded in [0,2].
    A01 : array-like (N, K)
        Design matrix (e.g., attributes).
    attr_names : list[str] or None
        Names for the K attributes (row labels in the output).
    print_summary : bool
        If True, print statsmodels' summary.
    robust : None or str
        If not None, use robust covariance (e.g., "HC3", "HC1").
    eps : float
        Small value to shrink away from 0 and 1 for numerical stability.

    Returns
    -------
    df_ranked : pd.DataFrame
        Columns: ['coef','se','z','pvalue','ci_low','ci_high'] for the mean submodel,
        sorted by p-value ascending, indexed by attribute labels.
    results : statsmodels results object
        Fitted model results (BetaModel if available; GLM otherwise).
    """
    gp = np.asarray(gp).reshape(-1)
    A01 = np.asarray(A01)
    N, K = A01.shape

    if attr_names is None:
        attr_names = [f"x{j+1}" for j in range(K)]
    else:
        assert len(attr_names) == K, "attr_names length must match number of columns in A01"

    # 1) Rescale to (0,1) and nudge away from boundaries
    y = gp / upper
    y = np.clip(y, eps, 1 - eps)

    # 2) Add intercept
    X = sm.add_constant(A01, has_constant='add')

    # 3) Try Beta regression; fall back to fractional logit GLM
    beta_fit_ok = False
    results = None
    try:
        # statsmodels >= 0.14: BetaModel is available here
        from statsmodels.miscmodels.regression import BetaModel  # type: ignore
        model = BetaModel(y, X, link=sm.genmod.families.links.logit())
        results = model.fit()
        beta_fit_ok = True
    except Exception:
        beta_fit_ok = False

    if not beta_fit_ok:
        # Fractional logit (quasi-likelihood): E[y|X] = logistic(X beta)
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        if robust is None:
            results = glm.fit()
        else:
            results = glm.fit(cov_type=robust)

    # 4) Extract mean-submodel coefficients (exclude intercept)
    # For BetaModel, params typically contain mean-coefs then precision term.
    # We detect how many mean coefficients there should be (= X.shape[1]).
    p = X.shape[1]

    params = results.params[:p]        # first p are mean-submodel coefs
    bse    = results.bse[:p]
    # z/t values depend on model, but results.tvalues exists for GLM too
    zvals  = getattr(results, "tvalues", getattr(results, "zvalues", np.nan*np.zeros_like(params)))[:p]
    pvals  = results.pvalues[:p]
    ci     = results.conf_int()[:p]    # (p,2)

    # Drop intercept (index 0) for the attribute table
    params_ = params[1:]
    bse_    = bse[1:]
    zvals_  = zvals[1:]
    pvals_  = pvals[1:]
    ci_     = ci[1:]

    df = pd.DataFrame({
        "coef":   params_,
        "se":     bse_,
        "z":      zvals_,
        "pvalue": pvals_,
        "ci_low": ci_[:, 0],
        "ci_high":ci_[:, 1],
    }, index=attr_names)

    df_ranked = df.sort_values("pvalue", ascending=True)

    if print_summary:
        print(results.summary())

    return df_ranked, results

def run_logistic(gp, A01, attr_names=None, print_summary=False, robust=None, threshold=1.0):
    """
    Logistic: 1{g_p > threshold} ~ attributes (+ intercept).
    Returns a DataFrame sorted by p-value (ascending), indexed by attribute labels.

    Parameters
    ----------
    gp : array-like (N,)
        Per-image scalar in [0,2].
    A01 : array-like (N, K)
        Attribute matrix (e.g., {0,1}). K should be 40 for CelebA.
    attr_names : list[str] or None
        Row labels for attributes (length K). If None, uses x1..xK.
    print_summary : bool
        If True, print the statsmodels summary().
    robust : None or str
        If not None, use robust covariance, e.g., "HC3", "HC1".
    threshold : float
        Use 1{gp > threshold} as the logistic outcome. Default 1.0.

    Returns
    -------
    df_ranked : pd.DataFrame with columns:
        ['coef', 'se', 'z', 'pvalue', 'ci_low', 'ci_high', 'odds_ratio']
        Sorted by 'pvalue' ascending, indexed by attribute labels.
    results : statsmodels.discrete.discrete_model.BinaryResults (or GLMResults fallback)
    """
    gp = np.asarray(gp).reshape(-1)
    A01 = np.asarray(A01)
    N, K = A01.shape

    if attr_names is None:
        attr_names = [f"x{j+1}" for j in range(K)]
    else:
        assert len(attr_names) == K, "attr_names length must match number of columns in A01"

    # Binary target: 1 if gp > threshold
    y = (gp > threshold).astype(float)

    # Add intercept
    X = sm.add_constant(A01, has_constant='add')

    # Fit Logit; fall back to GLM(Binomial) if perfect separation / failure
    model = sm.Logit(y, X)
    try:
        if robust is None:
            results = model.fit(disp=False)
        else:
            results = model.fit(disp=False, cov_type=robust)
    except Exception:
        # Fallback (often more stable numerically)
        glm = sm.GLM(y, X, family=sm.families.Binomial())
        if robust is None:
            results = glm.fit()
        else:
            results = glm.fit(cov_type=robust)

    if print_summary:
        print(results.summary())

    # Pull params (skip intercept at index 0)
    params = results.params[1:]
    bse    = results.bse[1:]
    # For GLM, .tvalues are z-stats under large-sample normality
    zvals  = getattr(results, "tvalues", results.tvalues)[1:]
    pvals  = results.pvalues[1:]
    ci     = results.conf_int()[1:]  # (K,2)

    df = pd.DataFrame({
        "coef":       params,
        "se":         bse,
        "z":          zvals,
        "pvalue":     pvals,
        "ci_low":     ci[:, 0],
        "ci_high":    ci[:, 1],
        "odds_ratio": np.exp(params),
    }, index=attr_names)

    df_ranked = df.sort_values("pvalue", ascending=True)
    return df_ranked, results

# ------------------- DDIM sampler from disk ---------------------- #

class DDIMDiskSampler:
    def __init__(
        self,
        real_loader,
        ddim_data_dir: str,
        gen_frac: float = 0.5,
        device: str = "cuda",
        fake_key: Optional[str] = None,
        assume_chw: bool = True,
        shuffle_batch: bool = True,
        max_shards_in_mem: int = 1,                 # keep memory bounded
        fake_cpu_dtype: Optional[torch.dtype] = torch.float16,  # cut host RAM
        scale_to: Optional[tuple] = (-1.0, 1.0),    # set to None to skip scaling

        # ---- NEW: split controls ----
        split: Optional[str] = None,                # "train" | "test" | None
        shard_ids: Optional[Sequence[int]] = None,  # explicit ids, e.g. range(0,16)
        id_range: Optional[Tuple[int,int]] = None,  # inclusive (start,end), e.g. (0,15)
    ):
        """
        Mixture sampler: gen_frac * fake_from_disk + (1-gen_frac) * real_from_loader
        - Streams shards: never loads more than `max_shards_in_mem`.
        - Accepts shard tensors or dicts/tuples with common keys.
        - NEW: train/test/custom shard filtering.
        """
        self.real_loader = real_loader
        self.real_iter = iter(real_loader)
        self.gen_frac = float(gen_frac)
        self.device = device
        self.fake_key = fake_key
        self.assume_chw = assume_chw
        self.shuffle_batch = shuffle_batch
        self.max_shards_in_mem = max(1, int(max_shards_in_mem))
        self.fake_cpu_dtype = fake_cpu_dtype
        self.scale_to = scale_to

        # ---- discover shard files + ids ----
        all_paths: List[str] = []
        all_ids: List[int] = []
        pat = re.compile(r"^shard_(\d+)\.pt$")
        for f in os.listdir(ddim_data_dir):
            m = pat.match(f)
            if m:
                all_paths.append(os.path.join(ddim_data_dir, f))
                all_ids.append(int(m.group(1)))
        assert all_paths, f"No shard_XXXXX.pt files found in {ddim_data_dir!r}"

        # ---- choose which ids to keep ----
        if shard_ids is not None:
            keep_ids = set(int(x) for x in shard_ids)
        elif id_range is not None:
            a, b = id_range
            assert a <= b, "id_range must be (start<=end)"
            keep_ids = set(range(int(a), int(b) + 1))
        elif isinstance(split, str):
            s = split.lower()
            if s == "train":
                keep_ids = set(range(0, 16))   # 00000..00015
            elif s == "test":
                keep_ids = set(range(16, 20))  # 00016..00019
            else:
                raise ValueError("split must be 'train', 'test', or None")
        else:
            keep_ids = set(all_ids)            # default: use all

        paired = sorted(
            ((sid, p) for sid, p in zip(all_ids, all_paths) if sid in keep_ids),
            key=lambda t: t[0],
        )
        assert paired, f"No shards left after filtering with split/ids: {sorted(keep_ids)}"
        self.shard_ids = [sid for sid, _ in paired]
        self.shards = [p for _, p in paired]

        # simple bounded cache (path -> tensor_on_cpu)
        self._cache_paths = []
        self._cache = {}

    # -------- internals --------
    def _load_shard_bounded(self, shard_path: str):
        if shard_path in self._cache:
            return self._cache[shard_path]

        # evict if needed
        while len(self._cache_paths) >= self.max_shards_in_mem:
            evict = self._cache_paths.pop(0)
            self._cache.pop(evict, None)
            torch.cuda.empty_cache()  # safe even if no CUDA

        obj = torch.load(shard_path, map_location="cpu")
        x = self._extract_tensor_from_obj(obj)

        # optional scaling (only if looks like [0,1] or integer)
        if self.scale_to is not None:
            lo, hi = self.scale_to
            if not torch.is_floating_point(x):
                x = x.float().div_(255.0)
                x = x.mul_(hi - lo).add_(lo)
            else:
                xmin, xmax = x.min(), x.max()
                if xmin >= 0 and xmax <= 1:
                    x = x.mul_(hi - lo).add_(lo)

        # downcast on CPU to save RAM
        if self.fake_cpu_dtype is not None:
            x = x.to(self.fake_cpu_dtype, copy=False)

        self._cache[shard_path] = x
        self._cache_paths.append(shard_path)
        return x

    def _extract_tensor_from_obj(self, obj) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            x = obj
        elif isinstance(obj, dict):
            if self.fake_key is not None:
                if self.fake_key not in obj:
                    raise KeyError(f"{self.fake_key!r} not in shard keys {list(obj.keys())}")
                x = obj[self.fake_key]
            else:
                for k in ("images", "x", "data", "samples"):
                    if k in obj and isinstance(obj[k], torch.Tensor):
                        x = obj[k]; break
                else:
                    raise KeyError(f"No tensor found in shard dict keys {list(obj.keys())}; set fake_key.")
        elif isinstance(obj, (tuple, list)):
            x = next((t for t in obj if isinstance(t, torch.Tensor) and t.ndim >= 3), None)
            if x is None:
                raise TypeError("Shard tuple/list lacks a tensor-like images entry.")
        else:
            raise TypeError(f"Unsupported shard object type: {type(obj)}")

        # ensure NCHW
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}")
        N, A, B, C = x.shape
        if (not self.assume_chw) and C in (1,3):
            x = x.permute(0,3,1,2).contiguous()
        elif self.assume_chw and A not in (1,3) and C in (1,3):
            x = x.permute(0,3,1,2).contiguous()
        return x

    def _sample_fake(self, n: int) -> torch.Tensor:
        shard_path = random.choice(self.shards)
        X = self._load_shard_bounded(shard_path)   # CPU tensor (possibly half)
        m = X.shape[0]
        idx = torch.randint(0, m, (n,))
        return X[idx]

    def _sample_real(self, n: int) -> torch.Tensor:
        out, got = [], 0
        while got < n:
            try:
                batch = next(self.real_iter)
            except StopIteration:
                self.real_iter = iter(self.real_loader)
                batch = next(self.real_iter)
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            need = n - got
            out.append(batch[:need] if batch.shape[0] > need else batch)
            got += min(batch.shape[0], need)
        return torch.cat(out, dim=0)

    # -------- public API --------
    def __call__(self, batch_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
        tgt_device = device if device is not None else self.device
        n_fake = int(round(batch_size * self.gen_frac))
        n_real = batch_size - n_fake

        x_fake_cpu = self._sample_fake(n_fake) if n_fake > 0 else None
        x_real_cpu = self._sample_real(n_real) if n_real > 0 else None

        x = x_real_cpu if x_fake_cpu is None else (x_fake_cpu if x_real_cpu is None else torch.cat([x_real_cpu, x_fake_cpu], dim=0))
        if self.shuffle_batch:
            x = x[torch.randperm(x.shape[0])]

        return x.to(device=tgt_device, dtype=dtype, non_blocking=True)
class IndexedCelebA(torch.utils.data.Dataset):
    def __init__(self, base):
        self.base = base
        self.attr_names = getattr(base, "attr_names", None)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, attrs = self.base[i]         # (img, 40)
        return img, attrs, i              # return the absolute dataset index

class DDIMTrainSubset(Dataset):
    """
    Deterministic list of generated files from shards 00..15 (160k files total).
    Returns (img_tensor, idx_flat, path, shard_id, local_id).
    """
    def __init__(self, root, split="train", shard_ids=range(16), transform=None, pattern="*.png"):
        self.transform = transform
        self.paths = []
        self.meta = []   # (shard_id, local_id) for each file

        # Example root layout guess: {root}/{split}/{shard:02d}/*.png
        # Adjust if yours is different
        for s in shard_ids:
            shard_dir = os.path.join(root, split, f"{s:02d}")
            files = sorted(glob.glob(os.path.join(shard_dir, pattern)))
            # Keep exactly 10,000 per shard if there are extras
            files = files[:10_000]
            for j, p in enumerate(files):
                self.paths.append(p)
                self.meta.append((s, j))

        assert len(self.paths) == 160_000, f"Expected 160k, got {len(self.paths)}"

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        shard_id, local_id = self.meta[i]
        return img, i, path, shard_id, local_id

class DDIMFakeOnlySampler:
    def __init__(
        self,
        ddim_data_dir: str,
        device: str = "cuda",
        *,
        # ---- NEW: split controls ----
        split: Optional[str] = None,                # "train" | "test" | None
        shard_ids: Optional[Sequence[int]] = None,  # explicit ids, e.g. range(0,16)
        id_range: Optional[Tuple[int,int]] = None,  # inclusive (start, end), e.g. (0,15)

        # data layout & scaling
        fake_key: Optional[str] = None,             # key if shards are dicts
        assume_chw: bool = True,                    # auto-fix NHWC → NCHW
        cpu_cast_dtype: Optional[torch.dtype] = torch.float16,  # shrink host RAM
        scale_to: Optional[Tuple[float,float]] = (-1.0, 1.0),   # None if already [-1,1]

        # transfer & epoching
        chunked_device_transfer: bool = True,
        device_chunk_size: int = 256,
        reshuffle_each_epoch: bool = True,
        rng: Optional[torch.Generator] = None,
    ):
        """
        Fake-only sampler from DDIM shard_*.pt files, sampling WITHOUT REPLACEMENT
        across calls until all chosen shards are exhausted (then a new epoch starts).
        Only the selected shard subset (train/test/custom) is used.
        """
        self.device = device
        self.fake_key = fake_key
        self.assume_chw = assume_chw
        self.cpu_cast_dtype = cpu_cast_dtype
        self.scale_to = scale_to
        self.chunked_device_transfer = chunked_device_transfer
        self.device_chunk_size = max(1, int(device_chunk_size))
        self.reshuffle_each_epoch = reshuffle_each_epoch
        self.rng = rng

        # 1) Discover all shard files + ids
        all_paths: List[str] = []
        all_ids: List[int] = []
        pat = re.compile(r"^shard_(\d+)\.pt$")
        for f in os.listdir(ddim_data_dir):
            m = pat.match(f)
            if m:
                all_paths.append(os.path.join(ddim_data_dir, f))
                all_ids.append(int(m.group(1)))
        assert all_paths, f"No shard_XXXXX.pt files found in {ddim_data_dir!r}"

        # 2) Determine which shard ids to keep
        if shard_ids is not None:
            keep_ids = set(int(x) for x in shard_ids)
        elif id_range is not None:
            a, b = id_range
            assert a <= b, "id_range must be (start<=end)"
            keep_ids = set(range(int(a), int(b) + 1))
        elif isinstance(split, str):
            s = split.lower()
            if s == "train":
                keep_ids = set(range(0, 16))        # 00000..00015
            elif s == "test":
                keep_ids = set(range(16, 20))       # 00016..00019
            else:
                raise ValueError("split must be 'train', 'test', or None")
        else:
            keep_ids = set(all_ids)  # default: use all

        # 3) Filter and sort by shard id
        paired = sorted(
            ((sid, p) for sid, p in zip(all_ids, all_paths) if sid in keep_ids),
            key=lambda t: t[0],
        )
        assert paired, f"No shards left after filtering with split/ids: {sorted(keep_ids)}"
        self.shard_ids = [sid for sid, _ in paired]
        self.shards = [p for _, p in paired]

        # 4) Per-shard state: m (len), perm (randperm), ptr
        self._meta = [{"m": None, "perm": None, "ptr": 0} for _ in self.shards]

        # One-shard RAM cache (OOM-safe)
        self._loaded_idx: Optional[int] = None
        self._loaded_tensor: Optional[torch.Tensor] = None  # CPU, compact dtype

        # Round-robin pointer + epoch count
        self._rr = 0
        self.epoch = 0

    # --------- internals ---------
    def _extract_tensor(self, obj) -> torch.Tensor:
        if isinstance(obj, torch.Tensor):
            x = obj
        elif isinstance(obj, dict):
            if self.fake_key is not None:
                if self.fake_key not in obj:
                    raise KeyError(f"{self.fake_key!r} not in shard dict keys {list(obj.keys())}")
                x = obj[self.fake_key]
            else:
                for k in ("images","x","data","samples"):
                    if k in obj and isinstance(obj[k], torch.Tensor):
                        x = obj[k]; break
                else:
                    raise KeyError(f"No tensor found in shard dict keys {list(obj.keys())}; set fake_key.")
        elif isinstance(obj, (tuple, list)):
            x = next((t for t in obj if isinstance(t, torch.Tensor) and t.ndim >= 3), None)
            if x is None:
                raise TypeError("Shard tuple/list lacks a tensor-like images entry.")
        else:
            raise TypeError(f"Unsupported shard object type: {type(obj)}")

        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor, got shape {tuple(x.shape)}")

        # NHWC → NCHW if needed
        N, A, B, C = x.shape
        if (not self.assume_chw) and C in (1,3):
            x = x.permute(0,3,1,2).contiguous()
        elif self.assume_chw and A not in (1,3) and C in (1,3):
            x = x.permute(0,3,1,2).contiguous()
        return x

    def _load_shard_cpu(self, shard_idx: int) -> torch.Tensor:
        # reuse if same shard is already resident
        if self._loaded_idx == shard_idx and self._loaded_tensor is not None:
            return self._loaded_tensor

        # evict previous (keep RAM bounded)
        self._loaded_idx = None
        self._loaded_tensor = None
        torch.cuda.empty_cache()

        obj = torch.load(self.shards[shard_idx], map_location="cpu")
        x = self._extract_tensor(obj)

        # Optional scaling to [-1,1]
        if self.scale_to is not None:
            lo, hi = self.scale_to
            if not torch.is_floating_point(x):
                x = x.float().div_(255.0)
            else:
                xmin, xmax = x.min(), x.max()
                if xmin >= 0 and xmax <= 1:
                    x = x.mul_(hi - lo).add_(lo)

        # Downcast on CPU to save RAM
        if self.cpu_cast_dtype is not None:
            x = x.to(self.cpu_cast_dtype, copy=False)

        # Initialize shard length if unknown
        meta = self._meta[shard_idx]
        if meta["m"] is None:
            meta["m"] = x.shape[0]

        self._loaded_idx = shard_idx
        self._loaded_tensor = x
        return x

    def _ensure_perm(self, shard_idx: int):
        meta = self._meta[shard_idx]
        if meta["perm"] is None:
            m = meta["m"]
            assert m is not None, "Shard length unknown before permutation init."
            meta["perm"] = torch.randperm(m, generator=self.rng)
            meta["ptr"] = 0

    def _shard_exhausted(self, shard_idx: int) -> bool:
        meta = self._meta[shard_idx]
        return (meta["m"] is not None) and (meta["ptr"] >= meta["m"])

    def _all_shards_exhausted(self) -> bool:
        for meta in self._meta:
            if meta["m"] is None or meta["ptr"] < meta["m"]:
                return False
        return True

    def _start_new_epoch(self):
        for meta in self._meta:
            if meta["m"] is None:
                continue
            if self.reshuffle_each_epoch:
                meta["perm"] = torch.randperm(meta["m"], generator=self.rng)
            meta["ptr"] = 0
        self.epoch += 1

    def _advance_rr(self):
        self._rr = (self._rr + 1) % len(self.shards)

    def _take_from_shard(self, shard_idx: int, need: int) -> torch.Tensor:
        X = self._load_shard_cpu(shard_idx)  # CPU tensor
        meta = self._meta[shard_idx]
        self._ensure_perm(shard_idx)
        take = min(need, meta["m"] - meta["ptr"])
        idx = meta["perm"][meta["ptr"]: meta["ptr"] + take]
        meta["ptr"] += take
        return X[idx]

    def _to_device_chunked(self, x_cpu: torch.Tensor, device, dtype) -> torch.Tensor:
        if not self.chunked_device_transfer:
            return x_cpu.to(device=device, dtype=dtype, non_blocking=True)
        B = x_cpu.shape[0]
        out = torch.empty((B, *x_cpu.shape[1:]), device=device, dtype=dtype)
        s = self.device_chunk_size
        for i in range(0, B, s):
            out[i:i+s] = x_cpu[i:i+s].to(device=device, dtype=dtype, non_blocking=True)
        return out

    # --------- public API ---------
    # --- add to the signature ---
    def __call__(self, batch_size: int, device=None, dtype=torch.float32, return_meta: bool = False):
        """
        Returns (B, 3, 64, 64) with NO replacement (within-epoch).
        If return_meta=True, also returns a dict with:
            - 'shard': LongTensor[B]        (global shard id, e.g., 0..19)
            - 'idx_in_shard': LongTensor[B] (0..m_s-1 index inside that shard)
        """
        tgt_device = device if device is not None else self.device
        need = int(batch_size)

        cpu_chunks = []
        meta_pairs = []   # list of (shard_id, idx_tensor) in concat order

        while need > 0:
            if self._all_shards_exhausted():
                self._start_new_epoch()

            # hop to next available shard
            hops = 0
            while hops < len(self.shards) and self._shard_exhausted(self._rr):
                self._advance_rr()
                hops += 1
            if hops >= len(self.shards):
                self._start_new_epoch()

            # take from current shard
            shard_idx = self._rr
            sid = self.shard_ids[shard_idx]

            X = self._load_shard_cpu(shard_idx)  # CPU tensor
            meta = self._meta[shard_idx]
            self._ensure_perm(shard_idx)

            take = min(need, meta["m"] - meta["ptr"])
            idx_local = meta["perm"][meta["ptr"]: meta["ptr"] + take]
            meta["ptr"] += take

            cpu_chunks.append(X[idx_local])
            meta_pairs.append((sid, idx_local.clone()))
            need -= take
            self._advance_rr()

        # concatenate data
        x_cpu = cpu_chunks[0] if len(cpu_chunks) == 1 else torch.cat(cpu_chunks, dim=0)

        # build per-item metadata in concat order
        if return_meta:
            shards_concat = torch.cat([torch.full((idx.shape[0],), sid, dtype=torch.long) for sid, idx in meta_pairs], dim=0)
            idxs_concat   = torch.cat([idx for _, idx in meta_pairs], dim=0)
            assert shards_concat.shape[0] == x_cpu.shape[0] == idxs_concat.shape[0]

        # within-batch shuffle (keep meta aligned)
        perm_b = torch.randperm(x_cpu.shape[0], generator=self.rng)
        x_cpu = x_cpu[perm_b]
        if return_meta:
            shards_concat = shards_concat[perm_b]
            idxs_concat   = idxs_concat[perm_b]

        # device move with chunking / OOM fallback
        try:
            x_dev = self._to_device_chunked(x_cpu, tgt_device, dtype)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if self.chunked_device_transfer and self.device_chunk_size > 1:
                self.device_chunk_size = max(1, self.device_chunk_size // 2)
                x_dev = self._to_device_chunked(x_cpu, tgt_device, dtype)
            else:
                print("[warn] CUDA OOM in sampler; returning CPU tensor.")
                x_dev = x_cpu.to(dtype=dtype)

        if return_meta:
            return x_dev, {"shard": shards_concat, "idx_in_shard": idxs_concat}
        else:
            return x_dev


def _extract_tensor_from_obj(obj, fake_key=None, assume_chw=True):
    if isinstance(obj, torch.Tensor):
        X = obj
    elif isinstance(obj, dict):
        if fake_key is not None:
            X = obj[fake_key]
        else:
            for k in ("images","x","data","samples"):
                if k in obj and isinstance(obj[k], torch.Tensor):
                    X = obj[k]; break
            else:
                raise KeyError("No tensor-like key found; set fake_key.")
    elif isinstance(obj, (tuple, list)):
        X = next((t for t in obj if isinstance(t, torch.Tensor) and t.ndim >= 3), None)
        if X is None:
            raise TypeError("No tensor found in shard tuple/list.")
    else:
        raise TypeError(f"Unsupported shard object type: {type(obj)}")

    # NHWC → NCHW if needed
    N, A, B, C = X.shape
    if (not assume_chw) and C in (1,3):
        X = X.permute(0,3,1,2).contiguous()
    elif assume_chw and A not in (1,3) and C in (1,3):
        X = X.permute(0,3,1,2).contiguous()
    return X

def _find_shard_path(ddim_data_dir: str, sid: int) -> str:
    """Prefer zero-padded name; fall back to regex search."""
    preferred = os.path.join(ddim_data_dir, f"shard_{sid:05d}.pt")
    if os.path.exists(preferred):
        return preferred
    # fallback: search any shard_*.pt with matching numeric id
    pat = re.compile(r"shard_(\d+)\.pt$")
    for p in glob.glob(os.path.join(ddim_data_dir, "shard_*.pt")):
        m = pat.search(os.path.basename(p))
        if m and int(m.group(1)) == int(sid):
            return p
    raise FileNotFoundError(f"No shard file for id={sid} under {ddim_data_dir}")

def load_fake_by_key(ddim_data_dir, shard_id: int, idx_in_shard: int, *, fake_key=None, assume_chw=True):
    path = _find_shard_path(ddim_data_dir, shard_id)
    obj = torch.load(path, map_location="cpu")
    X = _extract_tensor_from_obj(obj, fake_key=fake_key, assume_chw=assume_chw)
    return X[idx_in_shard]


# visualize extreme point images in the entire training set
@torch.no_grad()
def plot_generated_extremes_by_targets(
    scores: torch.Tensor,
    fetch_fn,                         # callable: indices -> (B,C,H,W) tensor (CPU ok)
    targets=(0.0, 1.0, 2.0),
    k_each: int = 25,
    nrow: int = 5,
    dpi: int = 120,
    title_prefix: str = "",
    thresh: float = 0.5,
    score_domain=(0.0, 2.0),          # clip for general case
):
    """
    Show up to k_each images nearest to each target, but ONLY if their scores lie
    within a target-specific band. For targets=(0,1,2) and thresh=0.5, the bands are:
        0 -> [0, 0.5), 1 -> (0.5, 1.5), 2 -> (1.5, 2]
    For other targets, falls back to [t-thresh, t+thresh] (clipped to score_domain).
    """
    scores = scores.view(-1).detach().cpu()
    N = scores.numel()

    def band_for_target(t):
        # Special bands for (0,1,2) with open/closed ends as requested
        if len(targets) == 3 and tuple(float(x) for x in targets) == (0.0, 1.0, 2.0) and abs(thresh-0.5) < 1e-9:
            if t == 0.0:
                # [0, 0.5)
                return ("closed", 0.0, 0.5, "open")
            elif t == 1.0:
                # (0.5, 1.5)
                return ("open", 0.5, 1.5, "open")
            elif t == 2.0:
                # (1.5, 2]
                return ("open", 1.5, 2.0, "closed")

        # General symmetric band: [t - thresh, t + thresh], clipped
        lo = max(score_domain[0], float(t) - thresh)
        hi = min(score_domain[1], float(t) + thresh)
        return ("closed", lo, hi, "closed")

    def filter_indices_by_band(band):
        left_mode, lo, hi, right_mode = band
        if left_mode == "closed":
            left_mask = (scores >= lo)
        else:  # "open"
            left_mask = (scores > lo)
        if right_mode == "closed":
            right_mask = (scores <= hi)
        else:  # "open"
            right_mask = (scores < hi)
        return torch.nonzero(left_mask & right_mask).view(-1)

    # layout
    T = len(targets)
    fig, axes = plt.subplots(1, T, figsize=(6*T, 6), dpi=dpi, constrained_layout=True)
    if T == 1:
        axes = [axes]

    for ax, t in zip(axes, targets):
        band = band_for_target(float(t))
        valid = filter_indices_by_band(band)

        if valid.numel() == 0:
            ax.axis("off")
            prefix = (title_prefix + ": ") if title_prefix else ""
            ax.set_title(f"{prefix}Nearest to {t:g} (0 in band)")
            continue

        # Among VALID indices, pick those nearest to target
        d = (scores[valid] - float(t)).abs()
        k = min(k_each, valid.numel())
        # smallest distances
        _, ord_idx = torch.topk(d, k, largest=False, sorted=True)
        sel_idx = valid[ord_idx]

        # fetch & display
        imgs = fetch_fn(sel_idx.long())
        if isinstance(imgs, torch.Tensor) and imgs.dtype != torch.float32:
            imgs = imgs.float()
        # If in [-1,1], map to [0,1] for display
        if imgs.min() < 0.0:
            imgs = (imgs + 1.0) / 2.0
        imgs = imgs.clamp(0, 1)

        grid = make_grid(imgs, nrow=nrow, padding=2)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.axis("off")

        sel_scores = scores[sel_idx]
        prefix = (title_prefix + ": ") if title_prefix else ""
        ax.set_title(
            f"{prefix}Nearest to {t:g} \n "
            f"(n={sel_idx.numel()}, min={sel_scores.min():.3g}, "
            f"max={sel_scores.max():.3g})"
        )

    plt.show()

def make_fake_fetcher(ddim_data_dir, q_shard, q_idx_local, *, fake_key=None, assume_chw=True):
    """
    Returns fetch_fn(indices) that loads images by (shard, idx_in_shard).
    Efficient: loads each shard once per call; uses zero-padded names.
    """
    q_shard = q_shard.cpu().long()
    q_idx_local = q_idx_local.cpu().long()

    def fetch_fake(indices: torch.Tensor):
        indices = indices.view(-1).long()
        # group requests per shard
        by_shard = {}
        for pos in indices.tolist():
            sid  = int(q_shard[pos])
            iloc = int(q_idx_local[pos])
            by_shard.setdefault(sid, []).append((pos, iloc))

        out = [None] * indices.numel()
        pos_map = {p:i for i,p in enumerate(indices.tolist())}

        for sid, plist in by_shard.items():
            shard_path = _find_shard_path(ddim_data_dir, sid)
            obj = torch.load(shard_path, map_location="cpu")
            X = _extract_tensor_from_obj(obj, fake_key=fake_key, assume_chw=assume_chw)
            for p, iloc in plist:
                out[pos_map[p]] = X[iloc].cpu()

        return torch.stack(out, dim=0)

    return fetch_fake



@torch.no_grad()
def nearest_pixel_l2_stream(
    gen_imgs,
    real_loader,
    device: str = "cuda",
    use_amp: bool = False,
    k: int = 1,
    treat_second_as_labels: bool = True,   # if True, synthesize indices when loader doesn't provide them
    progress: bool = True,
    pbar_desc: str = "Nearest L2",
):
    """
    Returns
    -------
    best_d : np.ndarray, shape (B, k)
        L2 distances for the top-k matches per query.
    best_i : np.ndarray, shape (B, k)
        Indices into the *dataset* (absolute) for the top-k matches per query.
    Also prints per-batch diagnostics:
      - how many *new* indices from this batch entered the running top-k
      - how many slots are filled per query (out of k)
    """
    if gen_imgs.ndim == 3:
        gen_imgs = gen_imgs.unsqueeze(0)
    assert gen_imgs.ndim == 4, f"Expected (B,C,H,W) or (C,H,W); got {tuple(gen_imgs.shape)}"

    gen_imgs = gen_imgs.to(device, non_blocking=True).float()
    Q  = gen_imgs.flatten(1)                # (B, CHW)
    q2 = (Q*Q).sum(1, keepdim=True)         # (B, 1)
    B  = Q.size(0)

    best_d = torch.full((B, k), float("inf"), device="cpu")
    best_i = torch.full((B, k), -1, dtype=torch.long, device="cpu")

    running_base = 0
    iterable = real_loader
    # tqdm over batches
    pbar = tqdm(iterable, desc=pbar_desc, total=len(real_loader) if hasattr(real_loader, "__len__") else None,
                disable=not progress)

    for batch_idx, batch in enumerate(pbar):
        # --- unpack & pick indices ---
        if isinstance(batch, (list, tuple)):
            real = batch[0]
            if treat_second_as_labels or len(batch) == 1:
                # synthesize absolute indices from running_base (shuffle=False assumed)
                idx = torch.arange(running_base, running_base + real.size(0), dtype=torch.long)
            else:
                idx = batch[1]
                if isinstance(idx, torch.Tensor):
                    idx = idx.detach().cpu().view(-1).long()
                else:
                    idx = torch.as_tensor(idx, dtype=torch.long).view(-1)
        else:
            real = batch
            idx  = torch.arange(running_base, running_base + real.size(0), dtype=torch.long)

        running_base += real.size(0)

        real = real.to(device, non_blocking=True).float()
        X    = real.flatten(1)               # (N, CHW)

        with torch.cuda.amp.autocast(enabled=use_amp):
            x2   = (X*X).sum(1).unsqueeze(0)    # (1, N)
            dot  = Q @ X.t()                    # (B, N)
            dist = (q2 + x2 - 2.0*dot).clamp_min_(0)  # (B, N)

        k_here = min(k, dist.size(1))
        vals, loc = torch.topk(dist, k=k_here, largest=False, dim=1)  # each row: best within this batch
        idx_this  = idx.to(vals.device)[loc].to("cpu")                # (B, k_here)

        # --- merge with global best (concat then take k smallest by (distance,index)) ---
        prev_best_i = best_i.clone()

        cat_d = torch.cat([best_d, vals.cpu()], dim=1)   # (B, k + k_here)
        cat_i = torch.cat([best_i, idx_this], dim=1)     # (B, k + k_here)

        # deterministic tie-breaker: key = (distance, index*eps)
        eps = 1e-9
        key = cat_d + eps * cat_i.to(cat_d.dtype)

        new_key, order = torch.topk(key, k=k, largest=False, dim=1)
        ar = torch.arange(cat_d.size(0)).unsqueeze(1)
        best_d = cat_d[ar, order]
        best_i = cat_i[ar, order]

        # --- diagnostics for this batch ---
        # 1) how many slots filled per query (>=0 means filled)
        filled_per_query = (best_i >= 0).sum(dim=1)  # (B,)

        # 2) how many *new* indices from THIS batch actually entered the running top-k?
        #    (compare sets row-by-row; ignore -1 placeholders)
        new_from_batch = 0
        for b in range(B):
            prev = set(prev_best_i[b].tolist())
            now  = set(best_i[b].tolist())
            if -1 in prev: prev.remove(-1)
            if -1 in now:  now.remove(-1)
            new_from_batch += len(now - prev)

        # # Print via tqdm to avoid breaking the bar
        # tqdm.write(
        #     f"[batch {batch_idx}] took {k_here} per query from this batch; "
        #     f"newly accepted into global top-{k}: {new_from_batch} "
        #     f"(filled per query: {filled_per_query.tolist()} / k={k})"
        # )

    if progress:
        pbar.close()

    return best_d.numpy(), best_i.numpy()


@torch.no_grad()
def pack_real_tensor(real_loader, device="cpu"):
    """
    Load entire dataset from loader into a tensor (N,C,H,W).
    Also returns index map (row -> dataset index).
    """
    imgs_all, idx_all = [], []
    for batch in real_loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            imgs, idx = batch
            imgs_all.append(imgs)
            idx_all.append(idx.view(-1).cpu().long())
        else:
            imgs = batch
            start = sum(x.size(0) for x in imgs_all)
            idx = torch.arange(start, start + imgs.size(0))
            imgs_all.append(imgs)
            idx_all.append(idx)
    X = torch.cat(imgs_all, dim=0).to(device)
    I = torch.cat(idx_all, dim=0)
    return X, I

def fetch_by_indices(dataset, indices, device="cuda"):
    """
    Grab samples from a Dataset by integer indices.
    Returns a (B,C,H,W) tensor on the given device.
    """
    imgs = [dataset[i][0] if isinstance(dataset[i], (tuple,list)) else dataset[i] 
            for i in indices]
    return torch.stack(imgs, dim=0).to(device)

@torch.no_grad()
def wasserstein_barycenter_2d(
    imgs,                      # (N, C, H, W) tensor on any device
    epsilon=0.02,              # entropic regularization (>0). Larger = smoother/faster
    max_iter=300,
    tol=1e-6,
    weights=None,              # None ⇒ uniform over N
    nonneg_shift=True,         # shift/clip to make masses ≥0 if your imgs are in [-1,1]
    per_channel=True,          # barycenter each channel independently
    device=None,               # computation device; default = imgs.device
    verbose=False,
):
    """
    Returns:
      bc: (C, H, W) tensor (same device as 'device'); each channel sums to 1 (mass).
    Notes:
      - We treat each (channel) image as a probability measure over pixels.
      - If you want a displayable image, you can rescale bc back to [0,1] or [-1,1].
    """
    assert imgs.ndim == 4, "imgs must be (N,C,H,W)"
    N, C, H, W = imgs.shape
    dev = device or imgs.device

    # Prepare measures A: (N, C, H*W) nonnegative, each (N,c) sums to 1
    X = imgs.to(dev, non_blocking=True).float()
    if nonneg_shift:
        # Map to nonnegative (e.g., if inputs are in [-1,1])
        # Simple and effective: shift to [0,1] then clip
        X = (X - X.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0])
        # If your data are already nonnegative, this is a no-op up to a constant shift

    X = torch.clamp(X, min=0.0)
    A = X.view(N, C, H * W)
    # Normalize each (N,c) to sum 1; add tiny mass if empty
    A_sum = A.sum(dim=-1, keepdim=True).clamp_min_(1e-12)
    A = A / A_sum

    # Uniform weights if none
    if weights is None:
        w = torch.full((N,), 1.0 / N, device=dev)
    else:
        w = torch.as_tensor(weights, dtype=torch.float32, device=dev)
        w = w / w.sum()

    # Build 2D squared-Euclidean cost on the grid (H*W, H*W)
    # Coordinates in [0,1] for numerical scale
    ys = torch.linspace(0.0, 1.0, H, device=dev)
    xs = torch.linspace(0.0, 1.0, W, device=dev)
    Y, Xg = torch.meshgrid(ys, xs, indexing="ij")
    P = torch.stack([Y.reshape(-1), Xg.reshape(-1)], dim=1)  # (HW, 2)
    # C_ij = ||p_i - p_j||^2
    # (HW,HW) = (HW,2)·(2,HW) via (a-b)^2 trick
    # Compute pairwise squared distances efficiently
    P2 = (P * P).sum(dim=1, keepdim=True)          # (HW,1)
    Cmat = (P2 - 2.0 * (P @ P.t()) + P2.t()).clamp_min_(0.0)

    # Gibbs kernel K = exp(-C/epsilon)
    K = torch.exp(-Cmat / max(epsilon, 1e-6))      # (HW,HW)
    KT = K.t()

    # Initialize scaling vectors u_i (N, HW)
    # We’ll do per-channel updates, but we can reuse K for all channels.
    def barycenter_channel(Ac):  # Ac: (N, HW)
        # Avoid zeros for log
        Ac = Ac.clamp_min(1e-12)
        U = torch.ones_like(Ac)  # (N, HW)

        prev_b = None
        for it in range(max_iter):
            KU = U @ K.t()                     # (N, HW), i-th row = K @ u_i
            KU = KU.clamp_min_(1e-30)

            # geometric (weighted) mean: b_j ∝ Π_i (KU_ij)^{w_i}
            logKU = KU.log()                   # (N, HW)
            b = torch.exp((w.view(-1, 1) * logKU).sum(dim=0))  # (HW,)
            b = b / b.sum()                    # normalize mass

            # Update each u_i = a_i / (K^T @ (b / (K u_i)))
            V = (b / KU).clamp_min_(1e-30)     # (N, HW)
            denom = V @ K                      # (N, HW) == (K^T @ V^T)^T
            denom = denom.clamp_min_(1e-30)
            U = (Ac / denom).clamp_min_(1e-30)

            if tol and it % 10 == 0:
                if prev_b is not None:
                    delta = (b - prev_b).abs().sum().item()
                    if verbose:
                        print(f"[iter {it}] Δb L1 = {delta:.3e}")
                    if delta < tol:
                        break
                prev_b = b.clone()

        return b  # (HW,)

    # Compute barycenter per channel
    bc_list = []
    if per_channel:
        for c in range(C):
            bc = barycenter_channel(A[:, c, :])
            bc_list.append(bc.view(H, W))
    else:
        # If you prefer a single grayscale barycenter:
        Ac = A.sum(dim=1)  # (N, HW) sum over channels then renormalize
        Ac = (Ac / Ac.sum(dim=1, keepdim=True).clamp_min_(1e-12))
        bc = barycenter_channel(Ac).view(H, W)
        # replicate to C channels for convenience
        bc_list = [bc for _ in range(C)]

    bc = torch.stack(bc_list, dim=0)  # (C, H, W); each channel sums to 1
    return bc

@torch.no_grad()
def to_grayscale_1ch(imgs, weights=(0.2126, 0.7152, 0.0722)):
    """
    imgs: (N,C,H,W) with C=3 (RGB) or already 1
    returns: (N,1,H,W) grayscale via luminance weights
    """
    assert imgs.ndim == 4, "imgs must be (N,C,H,W)"
    N, C, H, W = imgs.shape
    if C == 1:
        return imgs
    w = torch.tensor(weights, dtype=imgs.dtype, device=imgs.device).view(1, C, 1, 1)
    gray = (imgs * w).sum(dim=1, keepdim=True)  # (N,1,H,W)
    return gray

@torch.no_grad()
def wasserstein_barycenter_gray(imgs_gray, **kwargs):
    """
    imgs_gray: (N,1,H,W) on any device
    returns: (1,H,W) barycenter (mass map, sums to 1)
    kwargs are passed to your wasserstein_barycenter_2d (epsilon, max_iter, tol, device, etc.)
    """
    assert imgs_gray.ndim == 4 and imgs_gray.size(1) == 1, "expect (N,1,H,W)"
    # Reuse your previous barycenter impl; C=1 so per_channel=True does one channel
    bc = wasserstein_barycenter_2d(imgs_gray, per_channel=True, **kwargs)  # (1,H,W)
    return bc

# try to find the most similar images "LPIPS"
# pip install lpips
import torch, lpips
from torch.cuda.amp import autocast

@torch.no_grad()
def nearest_by_lpips_stream(gen_imgs, real_loader, device="cuda", k=1, net="vgg"):
    """
    gen_imgs: (B,C,H,W) or (C,H,W), normalized to [-1,1]
    real_loader: NOT shuffled, yields imgs or (imgs, anything)
    Returns: (dists(B,k), idx(B,k)) with smallest LPIPS first
    """
    if gen_imgs.ndim == 3:
        gen_imgs = gen_imgs.unsqueeze(0)
    gen = gen_imgs.to(device).float()

    loss_fn = lpips.LPIPS(net=net).to(device).eval()  # 'vgg' is typical

    B = gen.size(0)
    best_d = torch.full((B, k), float("inf"), device="cpu")
    best_i = torch.full((B, k), -1, dtype=torch.long, device="cpu")

    running_base = 0
    for batch in real_loader:
        real = batch[0] if isinstance(batch, (list,tuple)) else batch
        idx = torch.arange(running_base, running_base+real.size(0), dtype=torch.long)
        running_base += real.size(0)

        real = real.to(device).float()
        # LPIPS expects [-1,1]; ensure your real pipeline matches gen preprocessing.

        # Compute pairwise LPIPS for all queries vs this batch
        # Do per-query to avoid huge memory spikes (B usually small here)
        vals_all = []
        for b in range(B):
            with autocast(enabled=False):  # AMP can underflow LPIPS; keep fp32
                d = loss_fn(gen[b:b+1], real)              # (Br,1,1,1) or (Br,)
            vals_all.append(d.view(-1))                     # (Br,)
        dmat = torch.stack(vals_all, dim=0)                 # (B,Br)

        # top-k within this chunk
        k_here = min(k, dmat.size(1))
        vals, loc = torch.topk(dmat, k=k_here, largest=False, dim=1)  # (B,k_here)

        # map to dataset indices
        idx_this = idx[loc]                                  # (B,k_here)

        # merge with global best
        cat_d = torch.cat([best_d, vals.cpu()], dim=1)
        cat_i = torch.cat([best_i, idx_this.cpu()], dim=1)
        new_vals, order = torch.topk(cat_d, k=k, largest=False, dim=1)
        ar = torch.arange(B).unsqueeze(1)
        best_d, best_i = new_vals, cat_i[ar, order]

    return best_d.numpy(), best_i.numpy()

def _to_1d_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x).reshape(-1)
    return x

def closest_pairs_near_targets(
    x,
    targets=(0.0, 1.0, 2.0),
    k_per_target=1,
    proximity_weight=1.0,
):
    """
    For each target t, find k_per_target pairs (i,j) that minimize:
        score = |x_i - x_j| + proximity_weight * 0.5*(|x_i - t| + |x_j - t|)
    Only adjacent pairs in the sorted order are considered (optimal for the |x_i - x_j| term).

    Parameters
    ----------
    x : array-like or torch.Tensor, shape (n,)
    targets : iterable of floats
    k_per_target : int, number of pairs to return for each target
    proximity_weight : float, trade-off between pairwise closeness and nearness to target.
        - Larger -> prioritizes being near the target more strongly.

    Returns
    -------
    result : dict
        {
          t: {
            "pairs": [(i,j), ...],            # original indices
            "values": [(x_i, x_j), ...],
            "pair_diffs": [|x_i - x_j|, ...],
            "target_dists": [0.5*(|x_i-t|+|x_j-t|), ...],
            "scores": [score, ...]
          },
          ...
        }
    """
    x = _to_1d_numpy(x)
    n = x.shape[0]
    if n < 2:
        raise ValueError("Need at least two elements to form a pair.")

    # Sort once; adjacent pairs minimize |xi - xj|
    order = np.argsort(x)
    xs = x[order]

    # Precompute adjacent-pair info
    i_left = order[:-1]
    i_right = order[1:]
    v_left = xs[:-1]
    v_right = xs[1:]
    pair_diff = np.abs(v_left - v_right)  # shape (n-1,)

    result = {}
    for t in targets:
        target_dist = 0.5 * (np.abs(v_left - t) + np.abs(v_right - t))
        score = pair_diff + proximity_weight * target_dist

        # pick top-k by score (stable argsort)
        idx = np.argsort(score)[:k_per_target]

        pairs = list(zip(i_left[idx].tolist(), i_right[idx].tolist()))
        vals = list(zip(v_left[idx].tolist(), v_right[idx].tolist()))
        result[t] = {
            "pairs": pairs,
            "values": vals,
            "pair_diffs": pair_diff[idx].tolist(),
            "target_dists": target_dist[idx].tolist(),
            "scores": score[idx].tolist(),
        }
    return result

def find_pairs_for_groups(
    g_p_all,
    g_q_all,
    targets=(0.0, 1.0, 2.0),
    k_per_target=1,
    proximity_weight=1.0,
):
    """
    Convenience wrapper to run the search for both groups.
    """
    out = {
        "g_p": closest_pairs_near_targets(
            g_p_all, targets=targets, k_per_target=k_per_target, proximity_weight=proximity_weight
        ),
        "g_q": closest_pairs_near_targets(
            g_q_all, targets=targets, k_per_target=k_per_target, proximity_weight=proximity_weight
        ),
    }
    return out

    
def _hwc01(img):
    """Accepts torch/np image in CHW/HWC/HW; returns HxWx3 in [0,1]."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] in (1,3):     # CHW -> HWC
        arr = np.transpose(arr, (1,2,0))
    elif arr.ndim == 2:                              # HW -> HWC
        arr = arr[..., None]
    elif arr.ndim == 3 and arr.shape[2] in (1,3):    # already HWC
        pass
    else:
        raise ValueError(f"Unexpected image shape {arr.shape}")
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo + 1e-12:
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = (arr - lo) / (hi - lo)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return arr.astype(np.float32)

# -------------------- pairing (cross-group) --------------------

def cross_pairs_near_targets(
    g_p_all,
    g_q_all,
    targets=(0.0, 1.0, 2.0),
    k_per_target=5,
    proximity_weight=1.0,
    unique_indices=True,
):
    """
    Build pairs (i_p, i_q) with one real and one fake such that:
      score = |g_p - g_q| + proximity_weight * 0.5*(|g_p - t| + |g_q - t|)
    for each target t in {0,1,2}. Returns top-k per target.

    Uses a fast nearest-neighbor-on-sorted approach:
      For each element in the smaller group, check the two nearest in the other group.
    """
    gp = _to_1d_numpy(g_p_all)
    gq = _to_1d_numpy(g_q_all)

    # sort and keep original indices
    order_p = np.argsort(gp)
    order_q = np.argsort(gq)
    sp, sq = gp[order_p], gq[order_q]

    # choose the smaller group to iterate
    iterate_p = len(sp) <= len(sq)

    candidates = {t: [] for t in targets}

    if iterate_p:
        # for each sp[k], test nearest neighbor(s) in sq via bisect
        for k, vp in enumerate(sp):
            j = bisect.bisect_left(sq, vp)
            cand_js = [j-1, j]
            for jj in cand_js:
                if 0 <= jj < len(sq):
                    vq = sq[jj]
                    i_p = int(order_p[k])
                    i_q = int(order_q[jj])
                    for t in targets:
                        score = abs(vp - vq) + proximity_weight * 0.5*(abs(vp - t) + abs(vq - t))
                        candidates[t].append((score, i_p, i_q, float(vp), float(vq)))
    else:
        for k, vq in enumerate(sq):
            i = bisect.bisect_left(sp, vq)
            cand_is = [i-1, i]
            for ii in cand_is:
                if 0 <= ii < len(sp):
                    vp = sp[ii]
                    i_p = int(order_p[ii])
                    i_q = int(order_q[k])
                    for t in targets:
                        score = abs(vp - vq) + proximity_weight * 0.5*(abs(vp - t) + abs(vq - t))
                        candidates[t].append((score, i_p, i_q, float(vp), float(vq)))

    # pick top-k per target (optionally enforcing unique indices)
    out = {}
    for t in targets:
        cand = sorted(candidates[t], key=lambda z: z[0])
        chosen = []
        used_p, used_q = set(), set()
        for sc, ip, iq, vp, vq in cand:
            if unique_indices and (ip in used_p or iq in used_q):
                continue
            chosen.append({"pair": (ip, iq), "vals": (vp, vq), "score": sc,
                           "pair_diff": abs(vp - vq),
                           "target_dist": 0.5*(abs(vp - t) + abs(vq - t))})
            used_p.add(ip); used_q.add(iq)
            if len(chosen) >= k_per_target:
                break
        out[t] = chosen
    return out

# -------------------- visualization via your fetchers --------------------

def visualize_cross_pairs_oneplot(
    g_p_all,
    g_q_all,
    fetch_real,
    fetch_fake,
    targets=(0.0, 1.0, 2.0),
    k_per_target=5,
    proximity_weight=1.0,
    figsize=(12, 10),
    unique_indices=True,
    title_prefix="Real/Fake"
):
    """
    One combined figure: rows = pairs, columns = [Real, Fake, ...] grouped by target.
    For each target, shows k_per_target rows of Real-Fake pairs.
    """
    sel = cross_pairs_near_targets(
        g_p_all, g_q_all,
        targets=targets,
        k_per_target=k_per_target,
        proximity_weight=proximity_weight,
        unique_indices=unique_indices,
    )

    nrows = k_per_target
    ncols = len(targets) * 2  # 2 images per target
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    for t_idx, t in enumerate(targets):
        picks = sel[t]
        if len(picks) == 0:
            continue

        idx_p = torch.tensor([p["pair"][0] for p in picks], dtype=torch.long)
        idx_q = torch.tensor([p["pair"][1] for p in picks], dtype=torch.long)

        imgs_p = fetch_real(idx_p)
        imgs_q = fetch_fake(idx_q)

        for r, pack in enumerate(picks):
            ip, iq = pack["pair"]
            vp, vq = pack["vals"]

            axL = axes[r, 2*t_idx]
            axR = axes[r, 2*t_idx + 1]

            axL.imshow(_hwc01(imgs_p[r])); axL.axis("off")
            axL.set_title(f"Real idx={ip}\n g={vp:.3f}", fontsize=8)

            axR.imshow(_hwc01(imgs_q[r])); axR.axis("off")
            axR.set_title(f"Fake idx={iq}\n g={vq:.3f}", fontsize=8)

            if t_idx == 0:
                axL.set_ylabel(f"Δ={pack['pair_diff']:.3g}\nμ_t={pack['target_dist']:.3g}",
                               fontsize=8, rotation=0, labelpad=25, va="center")

    # column headers for targets
    for t_idx, t in enumerate(targets):
        fig.text(
            (2*t_idx + 1)/ncols,
            1.02,
            f"Target {t}",
            ha="center",
            va="bottom",
            fontsize=12,
            weight="bold"
        )

    fig.suptitle(f"{title_prefix}: cross pairs near targets {targets}", fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])
    plt.show()

    return sel


@torch.no_grad()
def nearest_wasserstein_stream(
    gen_imgs,
    real_loader,
    device: str = "cuda",
    k: int = 1,
    use_amp: bool = False,
    treat_second_as_labels: bool = True,   # synthesize indices if loader doesn't provide them
    progress: bool = True,
    pbar_desc: str = "Nearest W1 (EMD proxy)",
    eps: float = 1e-8,
    map_mode: str = "auto",                # "auto" maps [-1,1] -> [0,1] else clamps to [0,1]
):
    """
    Streaming nearest neighbors under a fast Wasserstein-1 proxy.

    Distance = sum_{j} |CDF_q[j] - CDF_x[j]|  on the flattened (row-major) pixel mass.
    Pixel mass is formed by mapping image to [0,1], summing channels, clamping >=0, and normalizing to 1.

    Parameters
    ----------
    gen_imgs : Tensor (B,C,H,W) or (C,H,W) or (B,H,W,C)
        Query images (fakes).
    real_loader : DataLoader over the real dataset (shuffle=False recommended).
        Batch can be (img, label) or (img, label, abs_idx). If abs_idx is present, it will be used.
    device : str
        CUDA device for compute ("cuda") or "cpu".
    k : int
        Keep top-k nearest reals for each query.
    use_amp : bool
        Mixed precision for batch-side ops (not very impactful here but supported).
    treat_second_as_labels : bool
        If True or loader batch has only one element, synthesize absolute indices using running_base.
    progress : bool
        Show tqdm bar.
    pbar_desc : str
        Progress bar description.
    eps : float
        Small positive to avoid zero-mass divisions.
    map_mode : {"auto", "clamp"}
        "auto": if values look like [-1,1], map via (x+1)/2; else clamp [0,1].

    Returns
    -------
    best_d : np.ndarray, shape (B, k)
        Wasserstein proxy distances.
    best_i : np.ndarray, shape (B, k)
        Absolute indices into the *dataset* for the selected reals.
    """

    # ---------- helpers ----------
    def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            # (C,H,W) or (H,W,C) -> add batch
            if x.shape[0] in (1,3):
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(0).permute(0,3,1,2).contiguous()
        elif x.ndim == 4:
            if x.shape[1] in (1,3):
                pass
            elif x.shape[-1] in (1,3):
                x = x.permute(0,3,1,2).contiguous()
            else:
                raise ValueError(f"Cannot infer channels, shape={tuple(x.shape)}")
        else:
            raise ValueError(f"Expected 3D/4D tensor, got shape={tuple(x.shape)}")
        return x

    def _to_mass01(x: torch.Tensor) -> torch.Tensor:
        """
        Map image to mass in [0,1] and normalize so each sample sums to 1.
        Sum over channels -> grayscale mass.
        """
        x = x.to(device=device, dtype=torch.float32)
        if map_mode == "auto":
            # Heuristic: if range exceeds [0,1] by a bit, assume [-1,1]
            x_min = torch.min(x).item()
            x_max = torch.max(x).item()
            if x_min < -0.05 or x_max > 1.05:
                x = 0.5 * (x + 1.0)  # [-1,1] -> [0,1]
            else:
                x = x.clamp(0, 1)
        else:
            x = x.clamp(0, 1)

        mass = x.sum(dim=1)                 # (B,H,W)
        mass = torch.relu(mass) + eps       # ensure nonnegative + epsilon
        mass = mass.flatten(1)              # (B, HW)
        mass = mass / (mass.sum(dim=1, keepdim=True))  # normalize to 1
        return mass

    # ---------- prep queries ----------
    gen_imgs = _ensure_bchw(gen_imgs)
    B = gen_imgs.size(0)

    q_mass = _to_mass01(gen_imgs)           # (B, HW)
    q_cdf  = torch.cumsum(q_mass, dim=1)    # (B, HW)

    # global top-k buffers (CPU)
    best_d = torch.full((B, k), float("inf"), device="cpu")
    best_i = torch.full((B, k), -1, dtype=torch.long, device="cpu")

    running_base = 0
    pbar = tqdm(real_loader, total=len(real_loader) if hasattr(real_loader, "__len__") else None,
                desc=pbar_desc, disable=not progress)

    for batch_idx, batch in enumerate(pbar):
        # unpack and form absolute indices
        if isinstance(batch, (list, tuple)):
            real = batch[0]
            if not treat_second_as_labels and len(batch) >= 2:
                idx = batch[1]
                idx = idx.detach().cpu().view(-1).long() if torch.is_tensor(idx) \
                      else torch.as_tensor(idx, dtype=torch.long).view(-1)
            else:
                idx = torch.arange(running_base, running_base + real.size(0), dtype=torch.long)
        else:
            real = batch
            idx  = torch.arange(running_base, running_base + real.size(0), dtype=torch.long)

        running_base += real.size(0)

        real = _ensure_bchw(real)
        with torch.cuda.amp.autocast(enabled=use_amp):
            x_mass = _to_mass01(real)            # (N, HW)
            x_cdf  = torch.cumsum(x_mass, dim=1) # (N, HW)

            # pairwise |CDF_q - CDF_x| summed over HW
            # result shape: (B, N)
            # Broadcast: (B,1,HW) vs (1,N,HW) -> (B,N,HW)
            diff = torch.abs(q_cdf[:, None, :] - x_cdf[None, :, :])
            dists = diff.sum(dim=2).to("cpu")    # (B, N) on CPU for stable concat below

        # select k-best within this batch, then merge
        k_here = min(k, dists.size(1))
        vals, loc = torch.topk(dists, k=k_here, largest=False, dim=1)   # (B, k_here)
        idx_this  = idx[loc]                                            # (B, k_here)

        prev_best_i = best_i.clone()

        cat_d = torch.cat([best_d, vals], dim=1)     # (B, k + k_here)
        cat_i = torch.cat([best_i, idx_this], dim=1) # (B, k + k_here)

        # stable tie-break: (distance, index)
        eps_key = 1e-9
        key = cat_d + eps_key * cat_i.to(cat_d.dtype)

        _, order = torch.topk(key, k=k, largest=False, dim=1)
        ar = torch.arange(cat_d.size(0)).unsqueeze(1)
        best_d = cat_d[ar, order]
        best_i = cat_i[ar, order]

        # diagnostics for this batch
        new_from_batch = 0
        for b in range(B):
            prev = set(prev_best_i[b].tolist()); prev.discard(-1)
            now  = set(best_i[b].tolist());      now.discard(-1)
            new_from_batch += len(now - prev)

        filled = (best_i >= 0).sum(dim=1).tolist()
        # tqdm.write(f"[batch {batch_idx}] new accepted from this batch: {new_from_batch} ; filled per query: {filled} / k={k}")

    if progress:
        pbar.close()

    return best_d.numpy(), best_i.numpy()



@torch.no_grad()
def nearest_w2_stream(
    gen_imgs,
    real_loader,
    device: str = "cuda",
    k: int = 25,
    n_projs: int = 64,                 # ↑ => tighter sliced W2^2
    map_mode: str = "auto",            # "auto": [-1,1]->[0,1], else clamp [0,1]
    use_amp: bool = False,
    treat_second_as_labels: bool = True,
    progress: bool = True,
    pbar_desc: str = "Nearest W2 (sliced)",
    color_mode: str = "per_channel",   # "per_channel" or "grayscale"
    channel_weights=None,              # None => uniform avg across channels
):
    """
    Streaming nearest neighbors under sliced 2-Wasserstein squared (W2^2).

    color_mode:
      - "grayscale": sum channels => color-blind (previous behavior).
      - "per_channel": compute W2^2 independently per channel, then average (or weighted) across channels.

    Returns:
      best_d : np.ndarray (B,k) of W2^2 distances
      best_i : np.ndarray (B,k) of absolute dataset indices
    """
    # ---------- small utils ----------
    def _ensure_bchw(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            if x.shape[0] in (1,3): x = x.unsqueeze(0)
            else: x = x.unsqueeze(0).permute(0,3,1,2).contiguous()
        elif x.ndim == 4:
            if x.shape[1] in (1,3): pass
            elif x.shape[-1] in (1,3): x = x.permute(0,3,1,2).contiguous()
            else: raise ValueError(f"Cannot infer channels from shape={tuple(x.shape)}")
        else:
            raise ValueError(f"Expected 3D/4D tensor, got {tuple(x.shape)}")
        return x

    def _map01(x: torch.Tensor) -> torch.Tensor:
        # Map to [0,1] for mass creation
        if map_mode == "auto":
            x_min, x_max = float(x.min().item()), float(x.max().item())
            if x_min < -0.05 or x_max > 1.05:
                x = 0.5*(x + 1.0)  # [-1,1] -> [0,1]
            else:
                x = x.clamp(0, 1)
        else:
            x = x.clamp(0, 1)
        return x

    # ---------- prep query & grid ----------
    gen_imgs = _ensure_bchw(gen_imgs).to(device=device, dtype=torch.float32)
    Bq, C, H, W = gen_imgs.shape
    assert Bq >= 1, "Empty query batch."

    # build spatial coords in [0,1]^2
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # (HW,2)

    # random unit directions in 2D
    theta = torch.randn(n_projs, 2, device=device)
    theta = theta / theta.norm(dim=1, keepdim=True).clamp_min_(1e-12)  # (n_projs,2)

    # project grid & precompute sort perms + Δs (shared by all images)
    s_all  = coords @ theta.T                    # (HW, n_projs)
    perm   = torch.argsort(s_all, dim=0)         # (HW, n_projs)
    s_sort = torch.gather(s_all, 0, perm)        # (HW, n_projs)
    delta_s = (s_sort[1:, :] - s_sort[:-1, :])   # (HW-1, n_projs)

    # query masses (per channel if requested)
    q = _map01(gen_imgs)
    if color_mode == "grayscale":
        q_mass = q.sum(dim=1).flatten(1)                                  # (Bq, HW)
    elif color_mode == "per_channel":
        q_mass = q.flatten(2)                                             # (Bq, C, HW)
        if channel_weights is None:
            cw = torch.full((C,), 1.0 / C, device=device)
        else:
            cw = torch.as_tensor(channel_weights, dtype=torch.float32, device=device)
            cw = cw / cw.sum()
    else:
        raise ValueError("color_mode must be 'grayscale' or 'per_channel'")

    # normalize to probability mass
    if color_mode == "grayscale":
        q_mass = q_mass / q_mass.sum(dim=1, keepdim=True).clamp_min_(1e-12)
    else:
        q_mass = q_mass / q_mass.sum(dim=2, keepdim=True).clamp_min_(1e-12)

    # global top-k on CPU
    best_d = torch.full((Bq, k), float("inf"), device="cpu")
    best_i = torch.full((Bq, k), -1, dtype=torch.long, device="cpu")

    running_base = 0
    pbar = tqdm(real_loader, total=len(real_loader) if hasattr(real_loader, "__len__") else None,
                desc=pbar_desc, disable=not progress)

    for bidx, batch in enumerate(pbar):
        # unpack batch + absolute indices
        if isinstance(batch, (list, tuple)):
            real = batch[0]
            if not treat_second_as_labels and len(batch) >= 2:
                idx = batch[1]
                idx = idx.detach().cpu().view(-1).long() if torch.is_tensor(idx) \
                      else torch.as_tensor(idx, dtype=torch.long).view(-1)
            else:
                idx = torch.arange(running_base, running_base + real.size(0), dtype=torch.long)
        else:
            real = batch
            idx  = torch.arange(running_base, running_base + real.size(0), dtype=torch.long)
        running_base += real.size(0)

        real = _ensure_bchw(real).to(device=device, dtype=torch.float32)
        if real.shape[-2:] != (H, W):
            real = F.interpolate(real, size=(H, W), mode="bilinear", align_corners=False)

        with torch.cuda.amp.autocast(enabled=use_amp):
            x = _map01(real)
            if color_mode == "grayscale":
                x_mass = x.sum(dim=1).flatten(1)                             # (Nb, HW)
                x_mass = x_mass / x_mass.sum(dim=1, keepdim=True).clamp_min_(1e-12)
            else:
                x_mass = x.flatten(2)                                        # (Nb, C, HW)
                x_mass = x_mass / x_mass.sum(dim=2, keepdim=True).clamp_min_(1e-12)

            Nb = x_mass.size(0)

            # accumulate W2^2 across projections to keep memory modest
            d_accum = torch.zeros((Bq, Nb), device=device)

            for p in range(n_projs):
                pp = perm[:, p]                           # (HW,)
                ds = delta_s[:, p]                        # (HW-1,)

                if color_mode == "grayscale":
                    q_ord = q_mass[:, pp]                 # (Bq, HW)
                    x_ord = x_mass[:, pp]                 # (Nb, HW)

                    q_cdf = torch.cumsum(q_ord, dim=1)[:, :-1]   # (Bq, HW-1)
                    x_cdf = torch.cumsum(x_ord, dim=1)[:, :-1]   # (Nb, HW-1)

                    # (Bq,Nb,HW-1)
                    diff = q_cdf[:, None, :] - x_cdf[None, :, :]
                    # integrate squared difference with trapezoid weights (here just Δs)
                    d_p = (diff**2 * ds[None, None, :]).sum(dim=2)  # (Bq,Nb)

                else:  # per_channel
                    # reorder per channel
                    q_ord = q_mass[:, :, pp]              # (Bq, C, HW)
                    x_ord = x_mass[:, :, pp]              # (Nb, C, HW)

                    q_cdf = torch.cumsum(q_ord, dim=2)[:, :, :-1]   # (Bq,C,HW-1)
                    x_cdf = torch.cumsum(x_ord, dim=2)[:, :, :-1]   # (Nb,C,HW-1)

                    # (Bq,Nb,C,HW-1)
                    diff = q_cdf[:, None, :, :] - x_cdf[None, :, :, :]
                    d_ch = (diff**2 * ds[None, None, None, :]).sum(dim=3)  # (Bq,Nb,C)

                    # weighted average across channels
                    d_p = (d_ch * cw[None, None, :]).sum(dim=2)  # (Bq,Nb)

                d_accum += d_p

            d_batch = (d_accum / float(n_projs)).to("cpu")  # (Bq,Nb) W2^2

        # select within batch and merge with global best
        k_here = min(k, d_batch.size(1))
        vals, loc = torch.topk(d_batch, k=k_here, largest=False, dim=1)
        idx_this  = idx[loc]                                  # (Bq,k_here)

        prev_best_i = best_i.clone()
        cat_d = torch.cat([best_d, vals], dim=1)
        cat_i = torch.cat([best_i, idx_this], dim=1)

        key = cat_d + (1e-9) * cat_i.to(cat_d.dtype)
        _, order = torch.topk(key, k=k, largest=False, dim=1)
        ar = torch.arange(cat_d.size(0)).unsqueeze(1)
        best_d = cat_d[ar, order]
        best_i = cat_i[ar, order]

        # diagnostics
        new_from_batch = 0
        for qq in range(Bq):
            prev = set(prev_best_i[qq].tolist()); prev.discard(-1)
            now  = set(best_i[qq].tolist());     now.discard(-1)
            new_from_batch += len(now - prev)
        filled = (best_i >= 0).sum(dim=1).tolist()
        # tqdm.write(f"[batch {bidx}] new accepted: {new_from_batch}; filled per query: {filled} / k={k}")

    if progress:
        pbar.close()

    return best_d.numpy(), best_i.numpy()


def all_q_indices_below_min_p(
    g_p_all,
    g_q_all,
    q_shard=None,            # optional: align-length tensor/array
    q_idx_local=None,        # optional: align-length tensor/array
    sort=True,               # sort by g_q ascending
    return_vals=True,        # include g_q values in output dict
):
    """
    Find ALL positions j where g_q_all[j] < min(g_p_all).

    Returns
    -------
    out : dict with keys:
        'idx'      : LongTensor [M] — positions in g_q_all
        'gq'       : FloatTensor [M] (only if return_vals=True)
        'shard'    : LongTensor [M] (only if q_shard provided)
        'idx_local': LongTensor [M] (only if q_idx_local provided)
        'min_p'    : float
    """
    gp = torch.as_tensor(g_p_all).view(-1).cpu()
    gq = torch.as_tensor(g_q_all).view(-1).cpu()

    min_p = gp.min().item()
    mask = gq < min_p
    idx = torch.nonzero(mask, as_tuple=False).view(-1)  # [M]

    if sort and idx.numel() > 0:
        order = torch.argsort(gq[idx])                  # ascending by g_q
        idx = idx[order]

    out = {"idx": idx, "min_p": float(min_p)}

    if return_vals:
        out["gq"] = gq[idx]

    if q_shard is not None:
        q_shard = torch.as_tensor(q_shard).view(-1).cpu()
        out["shard"] = q_shard[idx]

    if q_idx_local is not None:
        q_idx_local = torch.as_tensor(q_idx_local).view(-1).cpu()
        out["idx_local"] = q_idx_local[idx]

    return out