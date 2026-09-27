"""Reusable tensor architectures and initializers for density-ratio models.

No datasets, feature extractors, or pretrained weights are loaded here. Names
containing MNIST or CelebA are historical checkpoint APIs; their tensor shapes
and output parameterizations are documented below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'BoundedSoftplus',
    'BoundedSigmoid',
    'MLP',
    'ConvBlock',
    'Down',
    'Up',
    'UNet',
    'DREConvNet_DCGAN_MNIST',
    'dcgan_weights_init',
    'RatioNetCelebA64',
    'ViewToMNIST',
    'dcgan_init',
    'RatioMLP',
    'TwoLayerBlock',
]


# Architectures and initialization

class BoundedSoftplus(nn.Module):
    """Map tensors to (0, 2) via 2*softplus(x)/(1+softplus(x)).

    """
    def forward(self, x):
        sp = F.softplus(x)        # (0, ∞)
        return 2 * sp / (1 + sp)  # (0,2)


class BoundedSigmoid(nn.Module):
    """Map tensors to (0, scale) via scale*sigmoid(alpha*x).

    """
    def __init__(self, alpha=2.0, scale=2.0):
        super().__init__()
        self.alpha = alpha
        self.scale = scale
    def forward(self, x):
        return self.scale * torch.sigmoid(self.alpha * x)


class MLP(nn.Module):
    """Three hidden ReLU layers mapping (N, input_dim) to bounded outputs.

    The historical default is width 32 and one output in (0,2). ``output_alpha``
    controls the sigmoid slope. The ``model.*`` state-dictionary keys are kept
    unchanged for checkpoint compatibility.
    """
    def __init__(self, input_dim, hidden_dim=32, output_dim=1, output_alpha=2.0):
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
            # BoundedSoftplus()
            BoundedSigmoid(alpha=output_alpha, scale=2.0)
        )

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    """Two padded convolutions with BatchNorm and ReLU; preserve spatial size.

    """
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
    """Halve spatial dimensions with max pooling, then apply a convolution block.

    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(c_in, c_out)
    def forward(self, x): return self.conv(self.pool(x))


class Up(nn.Module):
    """Transpose-convolve, align to a skip tensor by padding, and concatenate.

    """
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
    """Convolutional encoder/decoder returning an unbounded spatial map.

    Accepts (N, c_in, H, W) and returns (N, c_out, H, W) for compatible
    spatial dimensions. This historical architecture is not a scalar ratio
    head; its layer names and skip connections are preserved.
    """
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


class DREConvNet_DCGAN_MNIST(nn.Module):
    """DCGAN-style scalar ratio model for square single-channel images.

    Accepts (N, img_hw**2), (N, img_hw, img_hw), or (N,1,img_hw,img_hw)
    and returns (N,1). The shape adapter historically requires one channel
    even though ``in_ch`` is a constructor argument. The default size is 28.
    ``log_scale=False`` emits a bounded ratio in (0,2); True emits a linear
    log ratio. The historical class name and parameter keys are retained.
    A dummy forward at construction also updates the initial BatchNorm buffers.
    """
    def __init__(
        self,
        in_ch=1,
        base=64,
        log_scale: bool = False,
        img_hw: int = 28,
        output_alpha: float = 0.5,
    ):
        super().__init__()
        self.log_scale = log_scale
        self.img_hw = img_hw
        self.output_alpha = output_alpha

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
        self.bounded_softplus = BoundedSigmoid(alpha=output_alpha)

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


def dcgan_weights_init(m):
    # DCGAN default init: N(0, 0.02) for conv/convT; gamma~N(1,0.02), beta=0 for BN
    """Initialize convolution weights and BatchNorm affine parameters.

    Uses the historical class-name matching convention; unlike ``dcgan_init``,
    this initializer does not initialize Linear layers.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.zeros_(m.bias.data)


class RatioNetCelebA64(nn.Module):
    """DCGAN-style ratio model for (N,3,64,64) tensors, normally in [-1,1].

    Returns (N,) bounded ratios in (0,2), or linear log ratios when
    ``log_scale=True``. Constructor ``in_ch`` can configure channel count.
    The historical class name, initialization, and parameter keys are retained.
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
            self.out_act = BoundedSigmoid()       # outputs w(x) > 0

        self.apply(dcgan_weights_init)

    def forward(self, x):
        """Input:  x in [-1,1], shape (N,3,64,64)
        Output: shape (N,) — w(x) if log_scale=False, else log w(x)
        """
        h = self.backbone(x)
        z = self.head(h)               # (N,1,1,1)
        z = z.view(z.size(0))          # (N,)
        return self.out_act(z)


class ViewToMNIST(nn.Module):
    """Reshape single-channel 28x28 inputs to (N,1,28,28).

    Accepts (N,784), (N,28,28), or (N,1,28,28). The historical name is
    retained for saved model compatibility.
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


def dcgan_init(m):
    """Initialize convolution and Linear weights plus BatchNorm parameters.

    Weights use N(0,0.02), BatchNorm scales use N(1,0.02), and biases are zero.
    Kept separate from ``dcgan_weights_init`` to preserve historical recipes.
    """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class RatioMLP(nn.Module):
    """Three-layer ReLU MLP with (N,1) outputs in (0,2).

    Uses historical ``network.*`` checkpoint keys and a separate scalar
    output_alpha. Kept distinct from MLP to preserve architecture metadata.
    """
    def __init__(self, input_dimension: int, hidden_dimension: int = 512, output_alpha: float = 2.0):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 1),
        )
        self.output_alpha = float(output_alpha)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.sigmoid(self.output_alpha * self.network(values))


class TwoLayerBlock(nn.Module):
    """Matched deep/residual block: the skip addition is the only difference.
    """

    def __init__(self, width, skip):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.skip = skip

    def forward(self, x):
        transformed = self.linear2(torch.relu(self.linear1(x)))
        return torch.relu(transformed + x if self.skip else transformed)


RATIO_ARCHITECTURES = (
    "baseline32", "wide64", "deep64", "residual64", "bottleneck8_residual64",
)


class ControlledRatioMLP(nn.Module):
    """Scalar RDR network with matched plain/residual architecture variants.

    Input is (N, dimension); output is 2*sigmoid(output_alpha*z). The optional
    learned linear projection has min(8, dimension) outputs. At alpha=2 these
    layer names and initializations match the saved architecture pilot models.
    Floating-point sigmoid saturation can reach 0 or 2; the trainer rejects
    nonfinite objectives rather than silently changing the loss target.
    """

    def __init__(self, dimension, architecture, output_alpha=2.):
        super().__init__()
        self.dimension, self.method = dimension, architecture
        self.output_alpha = float(output_alpha)
        self.bottleneck_dimension = min(8, dimension) if architecture == "bottleneck8_residual64" else None
        input_width = self.bottleneck_dimension or dimension
        self.projection = nn.Linear(dimension, input_width) if self.bottleneck_dimension else nn.Identity()
        width = 32 if architecture == "baseline32" else 64
        self.stem = nn.Linear(input_width, width)
        if architecture in ("baseline32", "wide64"):
            self.hidden = nn.Sequential(nn.Linear(width, width), nn.ReLU(),
                                        nn.Linear(width, width), nn.ReLU())
        else:
            self.hidden = nn.Sequential(*(TwoLayerBlock(width, architecture != "deep64") for _ in range(3)))
        self.output = nn.Linear(width, 1)

    def forward(self, x):
        hidden = self.hidden(torch.relu(self.stem(self.projection(x))))
        return 2 * torch.sigmoid(self.output_alpha * self.output(hidden))


def make_ratio_mlp(dimension, architecture="baseline32", output_alpha=2., seed=0):
    """Construct a CPU model without consuming the caller's random stream."""
    import math
    from numbers import Integral

    for name, value, minimum in (("dimension", dimension, 1), ("seed", seed, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if architecture not in RATIO_ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    if not math.isfinite(output_alpha) or output_alpha <= 0:
        raise ValueError("output_alpha must be finite and positive")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        return ControlledRatioMLP(int(dimension), architecture, output_alpha)


__all__ += ["RATIO_ARCHITECTURES", "ControlledRatioMLP", "make_ratio_mlp"]
