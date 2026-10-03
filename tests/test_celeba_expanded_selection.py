import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.CelebA.selection_training import scaled_training_risk, clip_scaled_gradients
from utils.losses import midpoint_loss_from_logits


@pytest.mark.parametrize('loss', ['hellinger', 'kl', 'chisq', 'js'])
@pytest.mark.parametrize('negative', [-3., -100., -400., -1000.])
def test_scaled_gradients_match_float64_clipped_reference(loss, negative):
    values = [negative, 1.2, -0.8, 4., -6.]
    reference = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    risk = midpoint_loss_from_logits(reference[:2], reference[2:], loss)
    risk.backward()
    maximum = reference.grad.abs().max()
    norm = (reference.grad / maximum).norm() * maximum
    expected = reference.grad * min(1., 1. / float(norm))
    actual = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
    value, backward, shift = scaled_training_risk(actual, 2, loss)
    backward.backward()
    clip_scaled_gradients([actual], shift)
    assert torch.isfinite(value)
    torch.testing.assert_close(actual.grad.double(), expected, atol=2e-7, rtol=3e-6)
    assert float(actual.grad.double().norm()) <= 1.000001


def test_float32_norm_overflow_is_clipped_without_zeroing():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([3e20, 4e20])
    assert torch.isinf(parameter.grad.norm())
    clip_scaled_gradients([parameter], 0.)
    torch.testing.assert_close(parameter.grad, torch.tensor([.6, .8]))


def test_nonfinite_components_are_not_accepted():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([float('inf'), 0.])
    with pytest.raises(FloatingPointError):
        clip_scaled_gradients([parameter], 0.)


def test_selection_only_advance_never_registers_other_stages(tmp_path, monkeypatch):
    module = importlib.import_module('experiments.CelebA.model_selection')
    calls = []
    monkeypatch.setattr(module, 'load_protocol', lambda output: {'config': {'selection_only': True}})
    monkeypatch.setattr(module, 'freeze_baseline', lambda *args: calls.append('freeze'))
    monkeypatch.setattr(module, 'report', lambda output, require_complete=False: calls.append(('report', require_complete)))
    monkeypatch.setattr(module, 'submit_stage', lambda *args: pytest.fail('No further jobs authorized'))
    module.advance(tmp_path, 'baseline', submit=True)
    assert calls == ['freeze', ('report', True)]
    with pytest.raises(ValueError):
        module.advance(tmp_path, 'sensitivity', submit=True)
