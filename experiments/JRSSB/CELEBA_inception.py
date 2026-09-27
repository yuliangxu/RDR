"""Canonical pytorch-fid Inception feature extraction shared by JRSSB stages."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import torch
import torch.nn.functional as F
from pytorch_fid.inception import FID_WEIGHTS_URL, InceptionV3

from CELEBA_data import sha256_file


BACKEND = "pytorch_fid_0.3.0"


def build_inception(dimensions: int, device: torch.device) -> InceptionV3:
    try:
        block = InceptionV3.BLOCK_INDEX_BY_DIM[int(dimensions)]
    except KeyError as error:
        raise ValueError(f"Unsupported pytorch-fid feature dimension: {dimensions}") from error
    return InceptionV3([block], resize_input=True, normalize_input=True).to(device).eval()


@torch.inference_mode()
def embed_batches(
    model: InceptionV3,
    batches: Iterable[torch.Tensor],
    device: torch.device,
    progress_every: int = 10_000,
    progress_label: str = "Embedded",
) -> torch.Tensor:
    features = []
    count = 0
    next_progress = int(progress_every)
    for images in batches:
        images = images.to(device=device, dtype=torch.float32, non_blocking=True)
        prediction = model(images)[0]
        if prediction.shape[-2:] != (1, 1):
            prediction = F.adaptive_avg_pool2d(prediction, output_size=(1, 1))
        features.append(prediction.squeeze(-1).squeeze(-1).cpu())
        count += len(images)
        if progress_every > 0 and count >= next_progress:
            print(f"{progress_label} {count:,} images", flush=True)
            while next_progress <= count:
                next_progress += int(progress_every)
    if not features:
        raise ValueError("Cannot embed an empty image stream.")
    return torch.cat(features, dim=0).float()


def inception_provenance() -> dict:
    weight_path = (
        Path(torch.hub.get_dir())
        / "checkpoints"
        / Path(urlparse(FID_WEIGHTS_URL).path).name
    )
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"pytorch-fid weights were not found after model construction: {weight_path}"
        )
    return {
        "backend": BACKEND,
        "pytorch_fid_version": importlib.metadata.version("pytorch-fid"),
        "inception_weights_url": FID_WEIGHTS_URL,
        "inception_weights_sha256": sha256_file(weight_path),
    }
