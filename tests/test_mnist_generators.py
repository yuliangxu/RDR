"""Portable loader checks and optional migration checks against original weights.

Set MNIST_PRETRAINED_ROOT to the directory containing mnist_dcgan/ and mnist_vae/
and MNIST_LEGACY_HELPERS to the preserved historical helpers.py to run both
migration checks. References are read locally; tests never download inputs.
"""

import ast
import os
from pathlib import Path

import pytest
import torch

from experiments.MNIST.generators import DCGAN_G_MNIST, VAE, build_dcgan28


def test_dcgan_checkpoint_loader_preserves_eval_and_rejects_wrong_architecture(tmp_path):
    original = DCGAN_G_MNIST(nz=7, n_planes=2).eval()
    checkpoint = tmp_path / "generator.pt"
    torch.save(original.state_dict(), checkpoint)
    restored, latent = build_dcgan28(checkpoint, nz=7, n_planes=2)
    assert latent == 7 and not restored.training
    z = torch.randn(3, 7, generator=torch.Generator().manual_seed(187))
    with torch.no_grad():
        expected, actual = original(z), restored(z)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(restored(z[:, :, None, None]), expected, atol=0, rtol=0)
    assert actual.shape == (3, 1, 28, 28)
    assert torch.isfinite(actual).all() and actual.min() >= -1 and actual.max() <= 1
    with pytest.raises(RuntimeError, match="size mismatch"):
        build_dcgan28(checkpoint, nz=8, n_planes=2)


def test_vae_decode_uses_supplied_latents_without_consuming_global_randomness():
    model = VAE().eval()
    z = torch.randn(3, 20, generator=torch.Generator().manual_seed(831))
    before = torch.get_rng_state().clone()
    with torch.no_grad():
        images = model.decode(z)
        repeated = model.decode(z)
    assert torch.equal(torch.get_rng_state(), before)
    torch.testing.assert_close(images, repeated, atol=0, rtol=0)
    assert images.shape == (3, 784)
    assert torch.isfinite(images).all() and images.min() >= 0 and images.max() <= 1
    normalized = images.reshape(3, 1, 28, 28).mul(2).sub(1)
    assert normalized.min() >= -1 and normalized.max() <= 1


def _input(variable, relative=None):
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"Optional local migration input: set {variable}")
    path = Path(value) / relative if relative else Path(value)
    if not path.is_file():
        pytest.fail(f"Configured migration input is absent: {path}")
    return path


def _reference_class(path, name):
    # Load only the original class, avoiding historical training/import side effects.
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@torch.no_grad()
def test_existing_dcgan_weights_exactly_match_preserved_generator():
    weights = _input("MNIST_PRETRAINED_ROOT", "mnist_dcgan/netG_epoch_99.pth")
    source = _input("MNIST_LEGACY_HELPERS")
    original = _reference_class(source, "DCGAN_G_MNIST")().eval()
    original.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    restored, _ = build_dcgan28(weights)
    z = torch.randn(7, 100, generator=torch.Generator().manual_seed(1049))
    before = torch.get_rng_state().clone()
    expected, actual = original(z), restored(z)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), before)
    assert actual.shape == (7, 1, 28, 28) and torch.isfinite(actual).all()
    assert actual.min() >= -1 and actual.max() <= 1


@torch.no_grad()
def test_existing_vae_weights_exactly_match_original_decode_and_forward():
    weights = _input("MNIST_PRETRAINED_ROOT", "mnist_vae/vae_epoch_25.pth")
    source = _input("MNIST_PRETRAINED_ROOT", "mnist_vae/vae.py")
    original = _reference_class(source, "VAE")().eval()
    restored = VAE().eval()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    original.load_state_dict(state, strict=True)
    restored.load_state_dict(state, strict=True)
    z = torch.randn(7, 20, generator=torch.Generator().manual_seed(5813))
    before = torch.get_rng_state().clone()
    expected, actual = original.decode(z), restored.decode(z)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), before)
    normalized = actual.reshape(7, 1, 28, 28).mul(2).sub(1)
    assert torch.isfinite(normalized).all() and normalized.min() >= -1 and normalized.max() <= 1
    x = torch.rand(7, 1, 28, 28, generator=torch.Generator().manual_seed(902))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4489)
        expected_forward = original(x)
        torch.manual_seed(4489)
        actual_forward = restored(x)
    for actual_tensor, expected_tensor in zip(actual_forward, expected_forward):
        torch.testing.assert_close(actual_tensor, expected_tensor, atol=0, rtol=0)
