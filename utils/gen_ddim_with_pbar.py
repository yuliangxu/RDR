# %%
# --- CONFIG: edit these paths/knobs and run this cell ---
REPO        = "/hpc/group/mastatlab/yx306/CelebA/DDIM"        # path to cloned ermongroup/ddim repo
CKPT        = "/hpc/group/mastatlab/yx306/CelebA/ckpt.pth"    # pretrained CelebA64 checkpoint
OUTDIR      = "/hpc/group/mastatlab/yx306/CelebA/DDIM/data"   # where to write shards

TOTAL       = 200_000     # total images to generate
SHARD_SIZE  = 10_000      # images per .pt shard (≈235 MiB each in float16)
STEPS       = 50          # DDIM steps
ETA         = 0.0         # 0 = deterministic DDIM
BATCH_SIZE  = 512         # generation batch size (reduce if you hit OOM)
DEVICE      = "cuda"      # "cuda" or "cpu"
SHOW_STEP_BAR = False     # True: show inner DDIM step bar

# --- imports ---
import os, sys, math, yaml, inspect, torch
from types import SimpleNamespace
from tqdm.auto import tqdm
os.makedirs(OUTDIR, exist_ok=True)
# %%
# ----------------------------
# Streaming DDIM sampler (yields batches) with optional per-step progress bar
# ----------------------------
@torch.no_grad()
def ddim_sampler_celeba64_streaming(
    repo_dir: str,
    ckpt_path: str,
    total: int,
    steps: int = 50,
    eta: float = 0.0,
    batch_size: int = 256,
    device: str = "cuda",
    show_step_bar: bool = False,
):
    """Yield batches of shape (B,3,H,W) in [-1,1] until `total` images are produced."""
    # add repo to path and import their Model
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    from models.diffusion import Model

    # read celeba.yml
    cfg_path = os.path.join(repo_dir, "configs", "celeba.yml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # image size (fallback 64)
    image_size = (
        cfg.get("data", {}).get("image_size")
        or cfg.get("model", {}).get("image_size")
        or 64
    )

    # build Model(config) (handle both styles: Model(config) or Model(**kwargs))
    def _ns(d): 
        return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})
    config = _ns(cfg)
    try:
        model = Model(config).to(device).eval()
    except TypeError:
        allowed = {k for k in inspect.signature(Model.__init__).parameters if k != "self"}
        model_kwargs = {k: v for k, v in dict(cfg.get("model", {})).items() if k in allowed}
        model = Model(**model_kwargs).to(device).eval()

    # load checkpoint (prefer list index 4 with clean keys)
    raw = torch.load(ckpt_path, map_location="cpu")
    state_dict = raw[4] if isinstance(raw, (list, tuple)) and len(raw) >= 5 else raw
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict):
        state_dict = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    # diffusion schedule
    diff = cfg.get("diffusion", {})
    T = int(diff.get("num_steps", diff.get("timesteps", 1000)))
    beta_1 = float(diff.get("beta_1", 1e-4))
    beta_T = float(diff.get("beta_T", 2e-2))
    betas = torch.linspace(beta_1, beta_T, T, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)  # a_t

    # step schedule (uniform subsample from T-1 -> 0)
    t_seq  = torch.linspace(T - 1, 0, steps, dtype=torch.long, device=device)
    t_next = torch.cat([t_seq[1:], torch.tensor([0], device=device, dtype=torch.long)])

    def _extract(a, t, x_shape):
        out = a.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))

    remaining = total
    while remaining > 0:
        bs = min(batch_size, remaining)
        x  = torch.randn(bs, 3, image_size, image_size, device=device)

        step_iter = range(len(t_seq))
        if show_step_bar:
            step_iter = tqdm(step_iter, desc="DDIM steps", leave=False)

        for i in step_iter:
            ti, ti_next = t_seq[i], t_next[i]
            t = torch.full((bs,), int(ti.item()), device=device, dtype=torch.long)
            eps = model(x, t)  # predict noise

            a_t = _extract(alphas_cumprod, t, x.shape)
            x0  = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t).clamp(min=1e-12)

            t_next_ = torch.full((bs,), int(ti_next.item()), device=device, dtype=torch.long)
            a_s     = _extract(alphas_cumprod, t_next_, x.shape)

            sigma_t = eta * torch.sqrt(
                (1.0 - a_s).clamp_min(1e-12) / (1.0 - a_t).clamp_min(1e-12) * (1.0 - (a_t / a_s)).clamp_min(0.0)
            )
            dir_xt = torch.sqrt(a_s).clamp_min(1e-12) * x0
            c_eps  = torch.sqrt((1.0 - a_s - sigma_t**2).clamp_min(0.0))
            noise  = torch.randn_like(x) if eta > 0 else 0.0
            x = dir_xt + c_eps * eps + sigma_t * noise

        yield x.clamp(-1, 1).detach().cpu()
        remaining -= bs
# %%
# ----------------------------
# Generation loop with progress bars (no CLI)
# ----------------------------
# read image size once for prealloc
cfg_path = os.path.join(REPO, "configs", "celeba.yml")
with open(cfg_path, "r") as f:
    _cfg = yaml.safe_load(f)
IM_SZ = (
    _cfg.get("data", {}).get("image_size")
    or _cfg.get("model", {}).get("image_size")
    or 64
)

produced = 0
shard_idx = 0

with tqdm(total=TOTAL, desc="Total samples", unit="img") as p_overall:
    while produced < TOTAL:
        shard_n = min(SHARD_SIZE, TOTAL - produced)
        out_path = os.path.join(OUTDIR, f"shard_{shard_idx:05d}.pt")

        # resume if shard exists
        if os.path.exists(out_path):
            d = torch.load(out_path, map_location="cpu")
            k = int(d["images"].shape[0])
            produced += k
            p_overall.update(k)
            shard_idx += 1
            continue

        with tqdm(total=shard_n, desc=f"Shard {shard_idx:05d}", unit="img", leave=False) as p_shard:
            # preallocate CPU float16 buffer to cap RAM
            shard_buf = torch.empty(shard_n, 3, IM_SZ, IM_SZ, dtype=torch.float16)
            filled = 0
            cur_bs = BATCH_SIZE

            while filled < shard_n:
                try:
                    need = shard_n - filled
                    b = min(cur_bs, need)
                    stream = ddim_sampler_celeba64_streaming(
                        repo_dir=REPO, ckpt_path=CKPT, total=b,
                        steps=STEPS, eta=ETA, batch_size=b,
                        device=DEVICE, show_step_bar=SHOW_STEP_BAR,
                    )
                    for batch in stream:
                        B = batch.shape[0]  # (B,3,IM_SZ,IM_SZ), float32 CPU
                        shard_buf[filled:filled+B].copy_(batch.half())
                        filled   += B
                        produced += B
                        p_shard.update(B)
                        p_overall.update(B)
                        del batch
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and cur_bs > 64:
                        cur_bs = max(64, cur_bs // 2)
                        torch.cuda.empty_cache()
                        print(f"[warn] CUDA OOM; reducing batch_size to {cur_bs} and retrying shard {shard_idx}")
                        continue
                    else:
                        raise

        torch.save({"images": shard_buf}, out_path)
        print(f"[ok] wrote {out_path}  shape={tuple(shard_buf.shape)}  dtype=float16")
        shard_idx += 1

print(f"[done] Generated {produced} samples into {OUTDIR}")
# %%
