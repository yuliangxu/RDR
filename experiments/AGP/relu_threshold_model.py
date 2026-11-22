"""Clipped feedforward ReLU RDR on row-normalized taxon abundances."""

import numpy as np
import torch
from torch import nn


class ThresholdRDR(nn.Module):
    """An additive hinge model with a two-hidden-layer interaction term.

    Callers apply mask_metrics.close before constructing or scoring the model.
    A column index per hinge implements coordinate-specific thresholds without
    allocating a dense taxa-by-hinges weight matrix.
    """

    def __init__(self, scale, taxon_index, thresholds, *, width=128, a=.01, b=1.99):
        super().__init__()
        if not 0 < a < 1 < b < 2:
            raise ValueError("Clipping endpoints must satisfy 0 < a < 1 < b < 2")
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32).clone())
        self.register_buffer("taxon_index", torch.as_tensor(taxon_index, dtype=torch.long).clone())
        self.register_buffer("bounds", torch.tensor([a, b], dtype=torch.float32))
        self.thresholds = nn.Parameter(torch.as_tensor(thresholds, dtype=torch.float32).clone())
        n_features = len(scale) + len(taxon_index)
        self.additive = nn.Linear(n_features, 1)
        self.interaction = nn.Sequential(
            nn.Linear(n_features, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1, bias=False),
        )
        # Small nonzero readout weights keep both paths and knots differentiable
        # on the first step, while the additive intercept centers f near one.
        nn.init.constant_(self.additive.bias, 1.)
        nn.init.normal_(self.additive.weight, std=.001 / np.sqrt(n_features))
        nn.init.normal_(self.interaction[-1].weight, std=.001 / np.sqrt(width))

    @classmethod
    def from_training(cls, p_train, q_train, *, max_knots=8, width=128, a=.01, b=1.99):
        """Initialize from pooled, closed train abundances of the two sources.

        For k=1,...,K use observed inverse-ECDF quantiles k/(K+1) of strictly
        positive values. Drop ties and the maximum (no strictly larger training
        value can activate that hinge). All-zero, singleton-positive and
        constant-positive taxa retain a linear feature and have zero hinges.
        """
        if not isinstance(max_knots, int) or not 0 <= max_knots <= 8:
            raise ValueError("max_knots must be an integer from zero to eight")
        p, q = np.asarray(p_train), np.asarray(q_train)
        if p.ndim != 2 or q.ndim != 2 or min(len(p), len(q)) == 0 or p.shape[1] != q.shape[1]:
            raise ValueError("Provide nonempty training matrices with matching taxa")
        if p.shape[1] == 0:
            raise ValueError("Training matrices require at least one taxon")
        rows = np.concatenate((p, q)).astype(np.float32, copy=False)
        if not np.isfinite(rows).all() or (rows < 0).any() or not np.allclose(rows.sum(1), 1.):
            raise ValueError("Training rows must be finite, nonnegative, and row-normalized")
        scales = np.ones(rows.shape[1], dtype=np.float32)
        taxon_index, thresholds = [], []
        for taxon, col in enumerate(np.ascontiguousarray(rows.T)):
            positive = np.sort(col[col > 0])
            if len(positive) == 0:
                continue
            # Divide by an upper positive quantile for conditioning. The
            # max/32 term bounds training coordinates even with extreme tails.
            n = len(positive)
            scale = max(positive[(9*n-1)//10], positive[-1]/32., 1e-8)
            scales[taxon] = scale
            ranks = (n*np.arange(1, max_knots+1)-1)//(max_knots+1)
            knots = np.unique(positive[ranks])
            knots = np.unique(knots[knots < positive[-1]] / scales[taxon])
            taxon_index.extend([taxon]*len(knots))
            thresholds.extend(knots)
        return cls(scales, taxon_index, thresholds, width=width, a=a, b=b)

    @classmethod
    def from_state_dict(cls, state):
        """Rebuild with checkpoint geometry, without access to training data."""
        a, b = state["bounds"].detach().cpu().tolist()
        # Reconstruct on the CPU even when the input checkpoint is on a GPU.
        model = cls(state["scale"].detach().cpu(), state["taxon_index"].detach().cpu(),
                    state["thresholds"].detach().cpu(), width=state["interaction.0.weight"].shape[0],
                    a=a, b=b)
        model.load_state_dict(state)
        return model

    def forward(self, x):
        scaled = x / self.scale
        hinges = torch.relu(scaled.index_select(1, self.taxon_index) - self.thresholds)
        features = torch.cat((scaled, hinges), dim=1)
        f = (self.additive(features) + self.interaction(features)).flatten()
        return torch.clamp(f, min=self.bounds[0], max=self.bounds[1])


def balanced_js(r_p, r_q):
    """Equal-source classification JS risk evaluated from the clipped RDR."""
    return -.5 * (torch.log(r_p / 2).mean() + torch.log1p(-r_q / 2).mean())


def balanced_brier_rdr(r_p, r_q):
    """Equal-source Brier score, treating r/2 as the real-source probability."""
    return .5 * ((1 - r_p / 2).square().mean() + (r_q / 2).square().mean())


def endpoint_fractions(r, a, b):
    """Return the fractions at (lower endpoint, upper endpoint) for one source."""
    return (r == a).float().mean().item(), (r == b).float().mean().item()
