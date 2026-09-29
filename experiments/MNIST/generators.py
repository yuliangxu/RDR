"""Inference architectures for the retained pretrained MNIST generators.

Weights and original architectures: csinva/gan-vae-pretrained-pytorch,
https://github.com/csinva/gan-vae-pretrained-pytorch
(`mnist_dcgan/netG_epoch_99.pth` and `mnist_vae/vae_epoch_25.pth`).
Generator pretraining is separate from the RDR experiments. The DCGAN emits
images in [-1, 1]; VAE.decode emits flattened images in [0, 1]. The experiment
runner performs the VAE reshape and conversion to model scale [-1, 1].
"""

import torch
from torch import nn
from torch.nn import functional as F


class DCGAN_G_MNIST(nn.Module):
    """Original 100 -> 512 -> 256 -> 128 -> 64 -> 1 MNIST DCGAN."""

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
            # Original final layer crops 32 x 32 to 28 x 28.
            nn.ConvTranspose2d(n_planes, nc, kernel_size=1, stride=1, padding=2, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        if z.ndim == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.main(z)


def build_dcgan28(weights_path, nz=100, n_planes=64, nc=1, device="cpu"):
    """Load the original state dictionary strictly and enter inference mode."""
    generator = DCGAN_G_MNIST(nz=nz, n_planes=n_planes, nc=nc).to(device).eval()
    state = torch.load(str(weights_path), map_location=device, weights_only=True)
    generator.load_state_dict(state, strict=True)
    return generator, nz


class VAE(nn.Module):
    """Original 784/400/20 MNIST VAE, without training-script side effects."""

    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.reshape(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
