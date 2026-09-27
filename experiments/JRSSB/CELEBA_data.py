"""Shared numerical and data code for the final Agent 3 workflow."""

from __future__ import annotations

import hashlib

import random

from pathlib import Path

from typing import Mapping, Union

import numpy as np

import pandas as pd

import torch

import yaml

from torchvision import transforms

from torchvision.transforms import InterpolationMode

def load_yaml(path: Union[str, Path]) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sha256_file(path: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()

def _extract_shard_tensor(obj) -> torch.Tensor:
    if torch.is_tensor(obj):
        tensor = obj
    elif isinstance(obj, dict):
        tensor = next(
            (obj[key] for key in ("images", "x", "data", "samples") if key in obj),
            None,
        )
        if tensor is None:
            tensor = next(
                (value for value in obj.values() if torch.is_tensor(value) and value.ndim == 4),
                None,
            )
    elif isinstance(obj, (tuple, list)):
        tensor = next((value for value in obj if torch.is_tensor(value) and value.ndim == 4), None)
    else:
        tensor = None
    if tensor is None or tensor.ndim != 4:
        raise TypeError("DDIM shard does not contain a four-dimensional image tensor.")
    if tensor.shape[1] not in (1, 3) and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor

def manifest_fingerprint(frame: pd.DataFrame) -> str:
    columns = ["source", "role", "source_id", "real_path", "shard", "idx_in_shard"]
    normalized = frame[columns].astype("string").fillna("")
    payload = normalized.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def interpolation_mode(name: str) -> InterpolationMode:
    normalized = str(name).lower().replace("_antialiased", "")
    choices = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST,
    }
    if normalized not in choices:
        raise ValueError(f"Unsupported interpolation: {name}")
    return choices[normalized]

def image_transform(config: Mapping, pixel_range: str = "zero_one"):
    feature = config["fid_features"]
    operations = [
        transforms.CenterCrop(int(feature["source_center_crop"])),
        transforms.Resize(
            (int(feature["analysis_resize"]), int(feature["analysis_resize"])),
            interpolation=interpolation_mode(feature["interpolation"]),
            antialias=True,
        ),
        transforms.ToTensor(),
    ]
    if pixel_range == "minus_one_one":
        operations.append(transforms.Normalize([0.5] * 3, [0.5] * 3))
    return transforms.Compose(operations)

def _to_zero_one(images: torch.Tensor) -> torch.Tensor:
    images = images.float()
    minimum = float(images.min())
    maximum = float(images.max())
    if minimum < -1e-4:
        images = (images + 1.0) / 2.0
    elif maximum > 1.0 + 1e-4:
        images = images / 255.0
    return images.clamp(0.0, 1.0)

def load_ddim_images(manifest: pd.DataFrame, output_range: str = "zero_one") -> torch.Tensor:
    manifest = manifest.reset_index(drop=True)
    result = torch.empty((len(manifest), 3, 64, 64), dtype=torch.float32)
    for tensor_path, group in manifest.groupby("tensor_path", sort=True):
        obj = torch.load(tensor_path, map_location="cpu")
        shard = _extract_shard_tensor(obj)
        positions = group.index.to_numpy()
        local_indices = torch.as_tensor(group["idx_in_shard"].to_numpy(), dtype=torch.long)
        selected = _to_zero_one(shard[local_indices])
        if selected.shape[-2:] != (64, 64):
            selected = torch.nn.functional.interpolate(
                selected, size=(64, 64), mode="bilinear", align_corners=False, antialias=True
            )
        result[torch.as_tensor(positions)] = selected
        del obj, shard, selected
    if output_range == "minus_one_one":
        result = result.mul(2.0).sub(1.0)
    return result
