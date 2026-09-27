# sampler_ddim_celeba64.py
import os, sys, yaml, torch, inspect
import torch.nn.functional as F
from types import SimpleNamespace



def to_ns(d):  # dict -> SimpleNamespace (recursively)
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_ns(v) for k, v in d.items()})
    return d

def _add_repo_to_path(repo_dir):
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

def _load_config(cfg_path):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

@torch.no_grad()
def ddim_sampler_celeba64(repo_dir, ckpt_path, n=512, steps=50, eta=0.0, batch_size=128, device="cuda"):
    """
    Sample CelebA 64x64 images from the ErmonGroup DDIM pretrained model.

    Parameters
    ----------
    repo_dir : str
        Path to local clone of https://github.com/ermongroup/ddim (root with models/, configs/, etc.)
    ckpt_path : str
        Path to pretrained checkpoint (the CelebA 64x64 ckpt that loads as a list of length 5).
    n : int
        Number of samples to generate.
    steps : int
        Number of DDIM steps (<= training T).
    eta : float
        DDIM stochasticity (0.0 = deterministic).
    batch_size : int
        Per-batch generation size.
    device : str
        "cuda" or "cpu".
    """
    import sys, os, yaml, torch
    from types import SimpleNamespace
    import inspect

    # ----- repo import path -----
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from models.diffusion import Model  # noqa

    # ----- load YAML config -----
    cfg_path = os.path.join(repo_dir, "configs", "celeba.yml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # convert dict -> object with attribute access (config.model.*, config.diffusion.*, etc.)
    def _ns(d):
        return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    config = _ns(cfg)

    # image size
    image_size = (
        cfg.get("data", {}).get("image_size")
        or cfg.get("model", {}).get("image_size")
        or 64
    )

    # ----- build model (Model expects a single 'config' arg) -----
    # (Some forks define Model(config), not Model(**cfg["model"]))
    # Keep this try/except for safety across minor code variations.
    try:
        model = Model(config).to(device).eval()
    except TypeError:
        # fallback: if this fork actually uses kwargs, filter them
        model_kwargs_raw = dict(cfg.get("model", {}))
        allowed = {k for k in inspect.signature(Model.__init__).parameters if k != "self"}
        model_kwargs = {k: v for k, v in model_kwargs_raw.items() if k in allowed}
        model = Model(**model_kwargs).to(device).eval()

    # ----- load checkpoint (prefer list index 4 with clean keys) -----
    raw = torch.load(ckpt_path, map_location="cpu")
    if isinstance(raw, (list, tuple)) and len(raw) >= 5:
        state_dict = raw[4]  # clean keys like 'conv_in.weight', ...
    else:
        state_dict = raw

    # strip 'module.' if present
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
        state_dict = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    # ----- diffusion schedule from config -----
    diff = cfg.get("diffusion", {})
    # Training horizon
    T = int(diff.get("num_steps", diff.get("timesteps", 1000)))
    beta_1 = float(diff.get("beta_1", 1e-4))
    beta_T = float(diff.get("beta_T", 2e-2))
    betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # a_t

    # choose a decreasing sequence of timesteps (uniform subsample)
    t_seq  = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)   # e.g., [999, ..., 0]
    t_next = torch.cat([t_seq[1:], torch.tensor([0], device=device, dtype=torch.long)])

    def _extract(a, t, x_shape):
        out = a.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    # ----- DDIM loop -----
    imgs, remaining = [], n
    while remaining > 0:
        bs = min(batch_size, remaining)
        x  = torch.randn(bs, 3, image_size, image_size, device=device)

        for ti, ti_next in zip(t_seq, t_next):
            t = torch.full((bs,), int(ti.item()), device=device, dtype=torch.long)
            eps = model(x, t)  # predict noise

            a_t = _extract(alphas_cumprod, t, x.shape)
            x0  = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t).clamp(min=1e-12)

            t_next_ = torch.full((bs,), int(ti_next.item()), device=device, dtype=torch.long)
            a_s     = _extract(alphas_cumprod, t_next_, x.shape)

            # DDIM sigma (η=0 => deterministic)
            sigma_t = eta * torch.sqrt(
                (1.0 - a_s).clamp_min(1e-12) / (1.0 - a_t).clamp_min(1e-12)
                * (1.0 - (a_t / a_s)).clamp_min(0.0)
            )

            # x_{t-1} = sqrt(a_s)*x0 + sqrt(1 - a_s - sigma_t^2)*eps + sigma_t*z
            dir_xt = torch.sqrt(a_s).clamp_min(1e-12) * x0
            c_eps  = torch.sqrt((1.0 - a_s - sigma_t**2).clamp_min(0.0))
            noise  = torch.randn_like(x) if eta > 0 else 0.0
            x = dir_xt + c_eps * eps + sigma_t * noise

        imgs.append(x.clamp(-1, 1).cpu())
        remaining -= bs

    return torch.cat(imgs, dim=0)
