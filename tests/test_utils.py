"""Scientific contracts for the reusable RDR API, independent of experiment data."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

from utils import DRE_batch, DRE_func, losses, networks, training


class TwoPointScores(nn.Module):
    """A free positive score at each of two atoms; no network approximation."""

    def __init__(self, scores):
        super().__init__()
        self.scores = nn.Parameter(torch.as_tensor(scores, dtype=torch.float64).clone())

    def forward(self, x):
        return self.scores[x[:, 0].long()].unsqueeze(1)


def atom_sample(counts):
    return torch.repeat_interleave(torch.arange(2), torch.tensor(counts)).view(-1, 1)


def assert_population_minimum(objective, target):
    """Check stationarity and strict increase in each independent score direction."""
    model = TwoPointScores(target)
    value = objective(model)
    gradient, = torch.autograd.grad(value, model.scores)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), rtol=0, atol=2e-14)
    for atom in range(2):
        for direction in (-1, 1):
            shifted = target.clone()
            shifted[atom] += direction * 0.02
            assert objective(TwoPointScores(shifted)).item() > value.item()


@pytest.mark.parametrize("name,xi", [
    ("Hellinger_loss", 0.3), ("Hellinger_loss", 0.5), ("Hellinger_loss", 0.6),
    ("KL_loss", None), ("Chisq_loss", None), ("JS_loss", None),
])
def test_model_losses_target_supplied_denominator(name, xi):
    # Exact quadrature for P=(.3,.7), R=(.6,.4), using unequal counts 10 and 30.
    p, r = atom_sample([3, 7]), atom_sample([18, 12])
    p_mass = torch.tensor([0.3, 0.7], dtype=torch.float64)
    r_mass = torch.tensor([0.6, 0.4], dtype=torch.float64)
    target = p_mass / r_mass
    kwargs = {} if xi is None else {"xi": xi}
    if xi is not None:
        target *= xi / (1 - xi)
    assert_population_minimum(lambda model: getattr(losses, name)(model, p, r, **kwargs), target)


@pytest.mark.parametrize("name", ["hellinger_stable", "kl_from_outputs", "chisq_from_outputs"])
def test_output_losses_distinguish_ratio_and_posterior_targets(name):
    p, r = atom_sample([3, 7]), atom_sample([18, 12])
    p_mass = torch.tensor([0.3, 0.7], dtype=torch.float64)
    r_mass = torch.tensor([0.6, 0.4], dtype=torch.float64)
    # The historical output chi-square recipe estimates a posterior probability.
    target = p_mass / (p_mass + r_mass) if name == "chisq_from_outputs" else p_mass / r_mass

    def objective(model):
        wp, wr = model(p), model(r)
        if name == "kl_from_outputs":
            wp, wr = wp.log(), wr.log()
        return getattr(losses, name)(wp, wr)

    assert_population_minimum(objective, target)


@pytest.mark.parametrize("name", [
    "midpoint_hellinger_loss", "concatenated_midpoint_hellinger_loss",
    "original_hellinger_midpoint_loss",
])
def test_midpoint_losses_keep_equal_mixture_weights_with_unequal_counts(name):
    p, q = atom_sample([3, 7]), atom_sample([18, 12])
    p_mass = torch.tensor([0.3, 0.7], dtype=torch.float64)
    q_mass = torch.tensor([0.6, 0.4], dtype=torch.float64)
    target = p_mass / ((p_mass + q_mass) / 2)
    # Both atoms are strictly inside the clipping bounds of the clipped variant.
    assert_population_minimum(lambda model: getattr(losses, name)(model, p, q), target)


@pytest.mark.parametrize("name", ["MLP", "BoundedSigmoid", "DREConvNet_DCGAN_MNIST", "RatioNetCelebA64"])
def test_historical_model_imports_resolve_same_classes(name):
    assert getattr(DRE_func, name) is getattr(networks, name)


def test_historical_training_and_loss_imports_resolve_shared_implementations():
    assert DRE_func.Hellinger_loss is losses.Hellinger_loss
    assert DRE_func.run_DRE_fdiv is training.run_DRE_fdiv
    assert DRE_batch.run_DRE_fdiv_cnn_minibatch is training.run_DRE_fdiv_cnn_minibatch
    assert DRE_batch.chisq_from_outputs is losses.chisq_from_outputs


@pytest.mark.parametrize("class_name,prefix,kwargs", [
    ("MLP", "model", {"input_dim": 2, "hidden_dim": 4}),
    ("RatioMLP", "network", {"input_dimension": 2, "hidden_dimension": 4}),
])
def test_checkpoint_keys_and_strict_reload(class_name, prefix, kwargs):
    model = getattr(networks, class_name)(**kwargs)
    expected = {f"{prefix}.{layer}.{kind}" for layer in (0, 2, 4, 6) for kind in ("weight", "bias")}
    assert set(model.state_dict()) == expected
    restored = getattr(networks, class_name)(**kwargs)
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.tensor([[0.2, -0.1], [0.4, 0.9]])
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


def test_core_imports_are_independent_of_working_directory_and_experiments(tmp_path):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    script = """
import os
import sys
before = os.getcwd()
import utils
assert 'torch' not in sys.modules  # Package initialization remains lightweight.
from utils import losses, networks, training, diagnostics, calibration
from utils import DRE_func, DRE_batch
assert os.getcwd() == before
assert not any(name == 'experiments' or name.startswith('experiments.') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=environment,
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
