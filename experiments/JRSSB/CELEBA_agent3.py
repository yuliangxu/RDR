"""Shared numerical and data code for the final Agent 3 workflow."""

from __future__ import annotations

import json

import os

import subprocess

import sys

import urllib.request

from pathlib import Path

from typing import Mapping, Optional, Sequence, Union

import numpy as np

import pandas as pd

import torch

from PIL import Image

from CELEBA_data import load_ddim_images, manifest_fingerprint, sha256_file

PAIR_BRANCHES = ("lower", "upper")

def _run(command: Sequence[str], cwd: Optional[Path] = None) -> str:
    result = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()

def ensure_repository(generator_config: Mapping, allow_download: bool) -> tuple[Path, str]:
    repository = Path(generator_config["repository_path"])
    expected_commit = str(generator_config["repository_commit"])
    if not repository.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"Missing Diffusion-GAN repository {repository}; rerun without --offline."
            )
        repository.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", str(generator_config["repository_url"]), str(repository)])
    if not (repository / ".git").is_dir():
        raise RuntimeError(f"Generator repository path is not a git checkout: {repository}")
    status = _run(["git", "status", "--porcelain"], cwd=repository)
    if status:
        raise RuntimeError(
            f"External Diffusion-GAN checkout is dirty and will not be modified: {repository}"
        )
    try:
        _run(["git", "cat-file", "-e", f"{expected_commit}^{{commit}}"], cwd=repository)
    except subprocess.CalledProcessError:
        if not allow_download:
            raise RuntimeError(f"Pinned commit {expected_commit} is absent from {repository}.")
        _run(["git", "fetch", "origin", expected_commit], cwd=repository)
    observed = _run(["git", "rev-parse", "HEAD"], cwd=repository)
    if observed != expected_commit:
        _run(["git", "checkout", "--detach", expected_commit], cwd=repository)
        observed = _run(["git", "rev-parse", "HEAD"], cwd=repository)
    if observed != expected_commit:
        raise RuntimeError(f"Generator code revision mismatch: expected {expected_commit}, got {observed}.")
    return repository, observed

def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    request = urllib.request.Request(url, headers={"User-Agent": "JRSSB-Agent3/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def ensure_checkpoint(generator_config: Mapping, allow_download: bool) -> tuple[Path, str]:
    checkpoint = Path(generator_config["checkpoint_path"])
    expected = str(generator_config["checkpoint_sha256"]).lower()
    if checkpoint.is_file():
        observed = sha256_file(checkpoint)
        if observed.lower() != expected:
            if not allow_download:
                raise RuntimeError(
                    f"Generator checkpoint SHA256 mismatch for {checkpoint}: "
                    f"expected {expected}, got {observed}."
                )
            checkpoint.unlink()
    if not checkpoint.is_file():
        if not allow_download:
            raise FileNotFoundError(
                f"Missing Diffusion-StyleGAN2 checkpoint {checkpoint}; rerun without --offline."
            )
        _download_file(str(generator_config["checkpoint_url"]), checkpoint)
    observed = sha256_file(checkpoint)
    if observed.lower() != expected:
        raise RuntimeError(
            f"Generator checkpoint SHA256 mismatch for {checkpoint}: expected {expected}, got {observed}."
        )
    return checkpoint, observed

def prepare_generator_assets(
    generator_config: Mapping,
    output_root: Union[str, Path],
    allow_download: bool,
) -> dict:
    repository, commit = ensure_repository(generator_config, allow_download=allow_download)
    checkpoint, checkpoint_sha = ensure_checkpoint(
        generator_config, allow_download=allow_download
    )
    code_root = repository / str(generator_config["code_subdirectory"])
    if not (code_root / "legacy.py").is_file():
        raise FileNotFoundError(f"Pinned generator code is incomplete: {code_root}")
    payload = {
        "name": str(generator_config["name"]),
        "repository_url": str(generator_config["repository_url"]),
        "repository_path": str(repository),
        "repository_commit": commit,
        "code_root": str(code_root),
        "checkpoint_url": str(generator_config["checkpoint_url"]),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
    }
    provenance_path = Path(output_root) / "provenance" / "generator_assets.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    with provenance_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return payload

def load_stylegan_generator(asset_provenance: Mapping, generator_config: Mapping, device: torch.device):
    code_root = str(asset_provenance["code_root"])
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    try:
        import dnnlib
        import legacy
    except Exception as error:
        raise RuntimeError(
            "Could not import the pinned Diffusion-StyleGAN2 runtime. Check its environment dependencies."
        ) from error
    checkpoint = str(asset_provenance["checkpoint_path"])
    with dnnlib.util.open_url(checkpoint, verbose=False) as handle:
        payload = legacy.load_network_pkl(handle)
    if "G_ema" not in payload:
        raise KeyError("The pinned checkpoint has no G_ema network.")
    generator = payload["G_ema"].to(device).eval().requires_grad_(False)
    resolution = int(getattr(generator, "img_resolution", 0))
    channels = int(getattr(generator, "img_channels", 0))
    if resolution != int(generator_config["expected_resolution"]):
        raise RuntimeError(f"Generator resolution is {resolution}, expected 64.")
    if channels != int(generator_config["expected_channels"]):
        raise RuntimeError(f"Generator channels are {channels}, expected 3.")
    if bool(generator_config.get("force_reference_ops", False)):
        from torch_utils.ops import bias_act, upfirdn2d

        # The pinned checkout repeatedly retries unavailable CUDA extensions.
        # Marking them initialized selects the same reference path as its fallback.
        bias_act._inited = True
        bias_act._plugin = None
        upfirdn2d._inited = True
        upfirdn2d._plugin = None
    return generator

def latent_batch(seeds: Sequence[int], dimension: int, device: torch.device) -> torch.Tensor:
    values = np.stack(
        [np.random.RandomState(int(seed)).randn(int(dimension)) for seed in seeds]
    ).astype(np.float32)
    return torch.from_numpy(values).to(device)

@torch.inference_mode()
def generate_stylegan_batch(
    generator,
    seeds: Sequence[int],
    truncation_psi: float,
    noise_mode: str,
) -> torch.Tensor:
    z = latent_batch(seeds, int(generator.z_dim), next(generator.parameters()).device)
    label = torch.zeros((len(seeds), int(generator.c_dim)), device=z.device)
    images = generator(
        z,
        label,
        truncation_psi=float(truncation_psi),
        noise_mode=str(noise_mode),
    )
    return (images * 127.5 + 128.0).clamp(0, 255).to(torch.uint8).cpu()

def atomic_torch_save(payload, path: Union[str, Path]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_csv(frame: pd.DataFrame, path: Union[str, Path]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def atomic_json(payload: Mapping, path: Union[str, Path]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

def load_agent3_images(frame: pd.DataFrame, transform) -> torch.Tensor:
    tensor_paths = frame["tensor_path"].fillna("").astype(str)
    if (tensor_paths != "").all():
        return load_ddim_images(frame, output_range="zero_one")
    images = []
    for path in frame["real_path"].astype(str):
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    return torch.stack(images)

def manifest_digest(frame: pd.DataFrame) -> str:
    return manifest_fingerprint(frame)
