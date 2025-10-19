import torch, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Tuple, Optional, Union, Dict, Any
from itertools import cycle
import itertools
from torch.utils.data import DataLoader


def _model_device_dtype(model):
    p = next(model.parameters())
    return p.device, p.dtype

def _to_model(x, model):
    if x is None:
        return None
    device, dtype = _model_device_dtype(model)
    return x.to(device=device, dtype=dtype)

# ------------------------------------------------------------------------------------------------
#################----------------- Density Ratio Loss functions -----------------#################
# ------------------------------------------------------------------------------------------------

def Hellinger_loss(model, x_p, x_q, xi=0.5, log_scale = False):
    """
    Approximates the loss:
       L(w) = E_{x ~ p}[w(x)^{-1/2}] + E_{x ~ q}[w(x)^{1/2}],
    using the empirical averages of samples from p and q.
    """
    # --- ensure device/dtype match the model ---
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    x_p = x_p.to(device=device, dtype=dtype)
    x_q = x_q.to(device=device, dtype=dtype)
    # Compute w(x) for samples from p and q.

    # to avoid batchnorm jumps:
    X = torch.cat([x_p, x_q], dim=0)
    w = model(X)  # BN sees the same mixed distribution once
    w_p, w_q = w[:x_p.size(0)], w[x_p.size(0):]

    # # old:
    # w_p = model(x_p)  # shape: (N_p, 1)
    # w_q = model(x_q)  # shape: (N_q, 1)
    
    # Compute the empirical averages:
    # For p(x): the inverse square root, w(x)^{-1/2}
    if log_scale:
        loss_p = torch.mean(torch.exp( -0.5 * w_p))    
        loss_q = torch.mean(torch.exp(0.5*w_q))
    else:
        loss_p = torch.mean(w_p.pow(-0.5))
        # For q(x): the square root, w(x)^{1/2}
        loss_q = torch.mean(w_q.pow(0.5))
            
    # Total loss is the sum of the two terms.
    loss = xi*loss_p + (1-xi)*loss_q -1
    return loss

def KL_loss(model, x_p, x_q):
    """
    Maximize the loss:
       L(w) = 1+ E_{x ~ p}[log(w)] - E_{x ~ q}[w],
    using the empirical averages of samples from p and q.
    """
    # Compute w(x) for samples from p and q.
    w_p = model(x_p)  # shape: (N_p, 1)
    w_q = model(x_q)  # shape: (N_q, 1)
    
    # Compute the empirical averages:
    # For p(x): int log(w(x)) dx
    loss_p = torch.mean(torch.log(w_p))
    
    # For q(x): -int w(x) dx
    loss_q = - torch.mean(w_q)
    
    # Total loss is the sum of the two terms.
    loss = - (1+ loss_p + loss_q)
    return loss

def Chisq_loss(model, x_p, x_q):
    """
    Maximize the loss:
       L(w) = 2*E_{x ~ p}[w] - E_{x ~ q}[w^2] + 1,
    using the empirical averages of samples from p and q.
    """
    # Compute w(x) for samples from p and q.
    w_p = model(x_p)  # shape: (N_p, 1)
    w_q = model(x_q)  # shape: (N_q, 1)
    
    # Compute the empirical averages:
    # For p(x): 2*E_{x ~ p}[w]
    loss_p = 2 * torch.mean(w_p)
    
    
    # For q(x):  - E_{x ~ q}[w^2 + w]
    loss_q = - torch.mean(w_q.pow(2)) 

    # Total loss is the sum of the two terms.
    loss = - (loss_p + loss_q + 1)
    return loss

# ------------------------------------------------------------------------------------------------
#################----------------- Neural Network Architechture -----------------#################
# ------------------------------------------------------------------------------------------------
class BoundedSoftplus(nn.Module):
    """Maps input to (0,2) using bounded softplus transformation"""
    def forward(self, x):
        sp = F.softplus(x)        # (0, ∞)
        return 2 * sp / (1 + sp)  # (0,2)

# ======= MLP (for 2D example)===============
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            # nn.Softplus()  # ensures the output is > 0
            BoundedSoftplus()
        )
    
    def forward(self, x):
        return self.model(x)



# ======= UNet ===============
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(c_in, c_out)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, 2, stride=2)
        self.conv = ConvBlock(c_in, c_out)
    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed (odd sizes)
        dh, dw = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        x = F.pad(x, (dw//2, dw - dw//2, dh//2, dh - dh//2))
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, base=32):
        super().__init__()
        self.inc  = ConvBlock(c_in, base)
        self.d1   = Down(base, base*2)
        self.d2   = Down(base*2, base*4)
        self.bot  = ConvBlock(base*4, base*8)
        self.u2   = Up(base*8, base*4)
        self.u1   = Up(base*4, base*2)
        self.u0   = Up(base*2, base)
        self.outc = nn.Conv2d(base, c_out, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        xb = self.bot(x3)
        x  = self.u2(xb, x3)
        x  = self.u1(x,  x2)
        x  = self.u0(x,  x1)
        return self.outc(x)

# --------------------------------------------------------
# MNIST: DCGAN architecture
# --------------------------------------------------------


class DREConvNet_DCGAN_MNIST(nn.Module):
    """
    DCGAN Discriminator-style backbone (as in csinva/mnist_dcgan) with a 1D head for ratio/log-ratio.
    - If log_scale=False: returns w(x) > 0 via BoundedSoftplus().
    - If log_scale=True:  returns log w(x) in R (linear).
    """
    def __init__(self, in_ch=1, base=64, log_scale: bool = False, img_hw: int = 28):
        super().__init__()
        self.log_scale = log_scale
        self.img_hw = img_hw

        # DCGAN D blocks (MNIST variant): 4x4 convs, stride=2, pad=1
        # 28 -> 14 -> 7 -> 4 (for img_hw=28)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, base, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(base, base * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Build a dummy pass to infer the remaining spatial size (e.g., 4x4 for 28x28 inputs)
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, img_hw, img_hw)
            h = self.block3(self.block2(self.block1(dummy)))
            _, c, h_sp, w_sp = h.shape
            if h_sp != w_sp:
                raise RuntimeError(f"Expected square feature map, got {h_sp}x{w_sp}.")
            self._feat_hw = h_sp
            self._enc_out_shape = h.shape  # [1, C, H, W]

        # Final conv to collapse HxW -> 1x1 (kernel = current spatial size; stride=1, pad=0)
        self.conv_last = nn.Conv2d(base * 4, 1, kernel_size=self._feat_hw, stride=1, padding=0, bias=True)

        # For the positive head
        self.bounded_softplus = BoundedSoftplus()

    def _ensure_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            if x.size(1) != self.img_hw * self.img_hw:
                raise ValueError(f"Got flat dim {x.size(1)}, expected {self.img_hw**2}.")
            x = x.view(-1, 1, self.img_hw, self.img_hw)
        elif x.ndim == 3:
            if x.shape[1:] != (self.img_hw, self.img_hw):
                raise ValueError(f"Got shape {tuple(x.shape)}, expected (N,{self.img_hw},{self.img_hw}).")
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            if x.shape[1:] != (1, self.img_hw, self.img_hw):
                raise ValueError(f"Got {tuple(x.shape)}, expected (N,1,{self.img_hw},{self.img_hw}).")
        else:
            raise ValueError(f"Expected 2D/3D/4D tensor, got dim={x.ndim}")
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._ensure_nchw(x)

        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)

        # safety: encoder output must match what we probed at init
        if tuple(h.shape[1:]) != tuple(self._enc_out_shape[1:]):
            raise RuntimeError(
                f"Encoder out {tuple(h.shape)} != expected {tuple(self._enc_out_shape)}."
            )

        # Collapse to 1x1 and flatten to (N, 1)
        out = self.conv_last(h)      # (N, 1, 1, 1)
        out = out.view(out.size(0), 1)

        # Head: log-scale vs positive
        return out if self.log_scale else self.bounded_softplus(out)

# --------------------------------------------------------
# CelebA: DCGAN architecture
# --------------------------------------------------------

def dcgan_weights_init(m):
    # DCGAN default init: N(0, 0.02) for conv/convT; gamma~N(1,0.02), beta=0 for BN
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)
        
class RatioNetCelebA64(nn.Module):
    """
    DCGAN-style ratio model for 64x64 RGB (CelebA-64).
      - Body identical to the reference discriminator.
      - Head emits w(x) (positive) or log w(x) depending on `log_scale`.
    """
    def __init__(self, in_ch: int = 3, ndf: int = 64, log_scale: bool = False):
        super().__init__()
        self.log_scale = log_scale
        self.in_ch = in_ch
        self.ndf = ndf

        # ---- DCGAN Discriminator body (64x64 -> 4x4) ----
        blocks = [
            # (nc) x 64 x 64 -> (ndf) x 32 x 32
            nn.Conv2d(in_ch, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # -> (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        self.backbone = nn.Sequential(*blocks)

        # ---- 4x4 -> 1x1 head (no sigmoid) ----
        self.head = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)

        # post-activation for ratio
        if self.log_scale:
            self.out_act = nn.Identity()           # outputs log w(x)
        else:
            self.out_act = BoundedSoftplus()       # outputs w(x) > 0

        self.apply(dcgan_weights_init)

    def forward(self, x):
        """
        Input:  x in [-1,1], shape (N,3,64,64)
        Output: shape (N,) — w(x) if log_scale=False, else log w(x)
        """
        h = self.backbone(x)
        z = self.head(h)               # (N,1,1,1)
        z = z.view(z.size(0))          # (N,)
        return self.out_act(z)

# --------------------------------------------------------------------------------------------
#################----------------- Density Ratio Estimator -----------------#################
# --------------------------------------------------------------------------------------------

# for generaral f-divergences
def run_DRE_fdiv(x_p,x_q,xi=0.5,num_epochs = 2000, 
                x_p_val=None, x_q_val=None,
                 loss_method='Hellinger',
                 NN="MLP",
                 log_scale = False,
                 optimizer=None,
                 scheduler=None,
                 clip_max_norm: float = 1.0):
     
    
    device = x_p.device if isinstance(x_p, torch.Tensor) else 'cpu'

    input_dim = x_p.shape[1]

    if NN == "UNet":
        model = DynamicMLP(input_dim=input_dim, hidden_dims=[32, 32])
    elif NN == "CNN_DCGAN":
        model = DREConvNet_DCGAN(in_ch=1, base=64, log_scale=log_scale).to(device)
    else:
        # Default to MLP
        model = MLP(input_dim=input_dim)

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
            optimizer, mode='min', factor=0.5, patience=5, cooldown=2, verbose=True
        )
    
    # ensure tensors are on the same device
    def _prep(X):
        if isinstance(X, torch.Tensor):
            return X.to(device)
        return X
        
    x_p, x_q = _prep(x_p), _prep(x_q)

    
    x_p_val = _prep(x_p_val) if x_p_val is not None else None
    x_q_val = _prep(x_q_val) if x_q_val is not None else None

    
    losses = []
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
        else:
            raise ValueError(f"Unknown loss_method: {loss_method}")

        loss.backward()
        # clip & (optionally) log grad norm
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        optimizer.step()

        losses.append(loss.item())

        # ---- choose metric for scheduler: val loss if provided, else train loss ----
        metric = loss.detach()
        if (x_p_val is not None) and (x_q_val is not None):
            model.eval()
            with torch.no_grad():
                if loss_method == 'Hellinger':
                    val_loss = Hellinger_loss(model, x_p_val, x_q_val, xi, log_scale=log_scale)
                elif loss_method == 'KL':
                    val_loss = KL_loss(model, x_p_val, x_q_val)
                else:  # Chisq
                    val_loss = Chisq_loss(model, x_p_val, x_q_val)
            model.train()
            metric = val_loss.detach()

        scheduler.step(metric)


        
        if (epoch % show_iter) == 0:
            print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}")

        if torch.isnan(loss):
            print("NaN loss encountered; stopping.")
            break

    return model,losses




class ViewToMNIST(nn.Module):
    """
    Adapter that reshapes input into N×1×28×28 for MNIST CNNs.
    Accepts flat [N,784], 2D [N,28,28], or already [N,1,28,28].
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:  # flat
            if x.shape[1] != 28*28:
                raise ValueError(f"Expected 784 features, got {x.shape}")
            x = x.view(-1, 1, 28, 28)
        elif x.ndim == 3:  # no channel
            if x.shape[1:] != (28, 28):
                raise ValueError(f"Expected (N,28,28), got {x.shape}")
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            if x.shape[1:] != (1, 28, 28):
                raise ValueError(f"Expected (N,1,28,28), got {x.shape}")
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")
        return x

        
def run_DRE_fdiv_cnn(
    x_p, x_q, xi=0.5, num_epochs=2000, x_p_val=None, x_q_val=None,
    loss_method='Hellinger', log_scale=False, optimizer=None, scheduler=None,
    clip_max_norm: float = 1.0, *, in_ch: int = 1, img_hw: tuple = (28,28), base: int = 64
):
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



# evaluated based on mixed ratio
def evaluate_model(
    model: torch.nn.Module,
    x_p: torch.Tensor,
    x_q: torch.Tensor,
    x_mixed: Optional[torch.Tensor] = None,
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """
    Evaluate a model on two or three sets of inputs.

    Parameters
    ----------
    model : torch.nn.Module
        The trained model (e.g., DCGAN discriminator).
    x_p, x_q : torch.Tensor
        Input batches.
    x_mixed : torch.Tensor, optional
        Input batch for mixture. If None, it will be skipped.

    Returns
    -------
    (g_p, g_q, g_mixed) if x_mixed is not None,
    otherwise (g_p, g_q).
    """
    model.eval()
    with torch.no_grad():
        p = next(model.parameters())
        device, dtype = p.device, p.dtype

        x_p_ = x_p.to(device=device, dtype=dtype)
        x_q_ = x_q.to(device=device, dtype=dtype)

        g_p = model(x_p_).squeeze(-1)
        g_q = model(x_q_).squeeze(-1)

        if x_mixed is not None:
            x_mixed_ = x_mixed.to(device=device, dtype=dtype)
            g_mixed = model(x_mixed_).squeeze(-1)
            return g_p, g_q, g_mixed
        else:
            return g_p, g_q

# visulize extreme point samples for saved x_q 
def plot_generated_extremes(
    g_q_all,
    x_q,
    thresh: float = 0.5,
    k_top: int = 25,
    k_near: int = 25,
    k_small: int = 25,
    dpi: int = 120,
):
    """
    Plot extreme subsets of generated scores: smallest, near-one, largest.

    Parameters
    ----------
    g_q_all : torch.Tensor
        All g_q scores (any shape, will be flattened).
    x_q : torch.Tensor
        Generated images (N,C,H,W).
    thresh : float
        Threshold for `help.select_extremes`.
    k_top, k_near, k_small : int
        Number of samples to show for each category.
    dpi : int
        DPI for matplotlib figure.
    """
    g_q_flat = g_q_all.view(-1)

    # Select extremes using your helper
    res = help.select_extremes(g_q_flat, thresh=thresh,
                               k_top=k_top, k_near=k_near, k_small=k_small)

    largest_vals, largest_idx   = res["largest"]["values"],  res["largest"]["indices"]
    closest_vals, closest_idx   = res["near_one"]["values"], res["near_one"]["indices"]
    smallest_vals, smallest_idx = res["smallest"]["values"], res["smallest"]["indices"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=dpi, constrained_layout=True)

    title_mid = f"Generated:Close-to-1 scores ({help.minmax_text_from_idx(g_q_flat, closest_idx)})"
    celeb.show_batch(x_q, nrow=5, idx=closest_idx, plot=True, ax=axes[1], show=False, title=title_mid)

    title_left = f"Generated:Smallest scores ({help.minmax_text_from_idx(g_q_flat, smallest_idx)})"
    celeb.show_batch(x_q, nrow=5, idx=smallest_idx, plot=True, ax=axes[0], show=False, title=title_left)

    title_right = f"Generated:Largest scores ({help.minmax_text_from_idx(g_q_flat, largest_idx)})"
    celeb.show_batch(x_q, nrow=5, idx=largest_idx, plot=True, ax=axes[2], show=False, title=title_right)

    plt.show()





# #################### try preconditioning #########################

from collections import defaultdict

# --- small helpers ------------------------------------------------------------
def _u_from_model_out(y, log_scale: bool):
    """
    Return u = log g. If model already outputs log g (log_scale=True), use it.
    If model outputs g>0 (e.g., via Softplus), take log(y + eps) safely.
    """
    if log_scale:
        return y
    else:
        eps = 1e-8
        return torch.log(y + eps)

@torch.no_grad()
def _renorm_on_q(model, x_q, log_scale: bool, strength: float = 1.0):
    """
    Optional tiny re-centering so that E_q[g] ≈ 1 on the current batch.
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
    # else: skip quietly if the model has no obvious additive bias

# --- main ---------------------------------------------------------------------
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
    """
    Adds a balanced diagonal 'natural gradient' preconditioner:
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
            optimizer, mode='min', factor=0.5, patience=5, cooldown=2, verbose=True
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

# ------------------------------------------------------------
# Minibatch f-div trainer that works with EMBEDDING LOADERS
# ------------------------------------------------------------
def _first_batch_dim(loader):
    it = iter(loader)
    batch = next(it)
    z, _ = _unpack_loader_batch(batch)
    if z.ndim > 2:
        z = z.view(z.size(0), -1)
    return z.shape[1]


def _model_device_dtype(model):
    p = next(model.parameters())
    return p.device, p.dtype


def _to_model(x, model):
    # NOTE: pass only the EMBEDDINGS here, not (z, imgs)
    if x is None:
        return None
    device, dtype = _model_device_dtype(model)
    return x.to(device=device, dtype=dtype)


# # Simple MLP fallback (use your own if already defined)
# class MLP(torch.nn.Module):
#     def __init__(self, input_dim, hidden=(512,512), out_dim=1):
#         super().__init__()
#         layers = []
#         last = input_dim
#         for h in hidden:
#             layers += [torch.nn.Linear(last, h), torch.nn.GELU()]
#             last = h
#         layers += [torch.nn.Linear(last, out_dim)]
#         self.net = torch.nn.Sequential(*layers)
#     def forward(self, x):  # x: (B, D)
#         return self.net(x).squeeze(-1)

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
):
    """
    Trains f-div ratio net on *embedding* minibatches from loaders.
    Returns (model, epoch_losses).
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
            optimizer, mode='min', factor=0.5, patience=5, cooldown=2, verbose=True
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

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    losses = []
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
                with torch.cuda.amp.autocast(dtype=torch.float16):
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
                # one epoch-level val estimate using a few minibatches (e.g., 8)
                val_steps = 8
                val_sum = 0.0
                for _ in range(val_steps):
                    xp = next(pv_it); xq = next(qv_it)
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
                val_loss = val_sum / val_steps
                metric = torch.as_tensor(val_loss, device=device)

        scheduler.step(metric)

        losses.append(epoch_loss)
        if (epoch % show_iter) == 0:
            print(f"[embed] Epoch {epoch:4d}: train_loss={epoch_loss:.6f}"
                  + (f", val_metric={metric.item():.6f}" if (p_val_loader is not None and q_val_loader is not None) else ""))

    return model.eval(), losses


def _unpack_loader_batch(batch):
    # returns (z, imgs_or_None)
    if isinstance(batch, dict):
        z = batch.get('z', batch.get('emb', None))
        if z is None:
            for v in batch.values():
                if torch.is_tensor(v):
                    z = v; break
        imgs = batch.get('imgs', None)
        return z, imgs
    if isinstance(batch, (list, tuple)):
        # common case: (z, imgs)
        if len(batch) >= 1 and torch.is_tensor(batch[0]):
            z = batch[0]
            imgs = batch[1] if (len(batch) >= 2 and torch.is_tensor(batch[1])) else None
            return z, imgs
        # fallback: first tensor found
        for v in batch:
            if torch.is_tensor(v): return v, None
    if torch.is_tensor(batch):
        return batch, None
    raise TypeError(f"Unsupported batch type: {type(batch)}")

def evaluate_model_embed(
    model: torch.nn.Module,
    p_embed_loader,
    q_embed_loader,
    mixed_embed_loader,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Evaluate a trained ratio model on embeddings from three loaders.
    If loaders yield images alongside embeddings, also return those images.

    Returns
    -------
    g_p, g_q, g_mixed : (B,) tensors of model outputs
    x_p, x_q, x_mixed : (B, C, H, W) tensors of original images, or None if unavailable
    """
    model.eval()
    with torch.no_grad():
        # model device/dtype
        p = next(model.parameters())
        device, dtype = p.device, p.dtype

        # pull one batch from each loader
        batch_p     = next(iter(p_embed_loader))
        batch_q     = next(iter(q_embed_loader))
        batch_mixed = next(iter(mixed_embed_loader))

        z_p, x_p     = _unpack_loader_batch(batch_p)
        z_q, x_q     = _unpack_loader_batch(batch_q)
        z_m, x_mixed = _unpack_loader_batch(batch_mixed)

        # flatten embeddings if needed
        if z_p.ndim > 2: z_p = z_p.view(z_p.size(0), -1)
        if z_q.ndim > 2: z_q = z_q.view(z_q.size(0), -1)
        if z_m.ndim > 2: z_m = z_m.view(z_m.size(0), -1)

        # move to model device/dtype
        z_p = z_p.to(device=device, dtype=dtype)
        z_q = z_q.to(device=device, dtype=dtype)
        z_m = z_m.to(device=device, dtype=dtype)

        # forward
        g_p     = model(z_p).squeeze(-1)
        g_q     = model(z_q).squeeze(-1)
        g_mixed = model(z_m).squeeze(-1)

    return g_p, g_q, g_mixed, x_p, x_q, x_mixed


# ------------------------------------------------------------------#
# ----------------------batch update for AGP data-------------------#
# ------------------------------------------------------------------#

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
    device: Optional[torch.device] = None,
):
    # ---- infer device and input dim ----
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
        optimizer, mode="min", factor=0.5, patience=5, cooldown=2, verbose=True
    )

    # ---- helpers for q-source ----
    def _make_iter(src, fallback_len=None):
        """Return an iterator over q-batches for one epoch."""
        if callable(src):
            # callable: we'll call with current p-batch size inside the loop
            return None  # signal "call per step"
        else:
            return iter(src)

    def _next_q_batch(q_iter, want_bsz: int):
        if callable(q_source):
            qb = q_source(want_bsz)  # e.g., q_mixed(batch_size)
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

            batch_q = _next_q_batch(q_iter, batch_p.size(0))
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
                    vq = _next_q_batch(vq_iter, vp.size(0))
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
        else:
            scheduler.step(train_loss)

        if (epoch % max(1, num_epochs // 10)) == 0:
            msg = f"[{epoch:04d}] train={train_loss:.6f}"
            if history["val_loss"]:
                msg += f"  val={history['val_loss'][-1]:.6f}"
            print(msg)

    return model, history


def _flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1) if x.ndim > 2 else x

def _ensure_each_side(n_total: int, p_weight: float):
    n_p = int(n_total * p_weight)
    n_q = n_total - n_p
    if n_total > 0 and n_p == 0: n_p, n_q = 1, n_total - 1
    if n_total > 0 and n_q == 0: n_q, n_p = 1, n_total - 1
    return n_p, n_q

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