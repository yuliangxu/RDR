"""Verified manifest reads and image/feature access in the original row order.

Images use uint8 storage, then [0, 1] for Inception or [-1, 1] for pixel RDR.
With ``fresh=True``, every generated shard/feature cache must exist in work_root.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from CELEBA_data import image_transform, load_yaml, sha256_file
from .CELEBA_paths import HERE


def read_json(path):
    """Read a manifest or provenance record."""
    return json.loads(Path(path).read_text())


def verify(path, expected=None):
    """Require an existing input and, when supplied, its recorded SHA256."""
    path = Path(path)
    if not path.is_file() or (expected and sha256_file(path) != expected):
        raise RuntimeError(f"Missing or changed input: {path}")
    return path


def read_selected(item):
    """Load selected rows only after checking their recorded file hash."""
    return pd.read_csv(
        verify(item["selected_manifest_path"], item.get("selected_manifest_sha256")),
        low_memory=False,
    )


def image_shard(work, original):
    """Map an original shard path to a stable filename within the new work root."""
    return (
        work / "images" / (hashlib.sha256(str(original).encode()).hexdigest() + ".pt")
    )


class ManifestImages(Dataset):
    """Bounded shard cache; identical uint8 preprocessing and model input scale."""

    def __init__(self, frame, work, fresh=False, zero_one=False):
        self.frame = frame.reset_index(drop=True)
        self.work, self.fresh, self.zero_one = work, fresh, zero_one
        self.transform = image_transform(
            load_yaml(HERE / "configs/design.yaml"), "zero_one"
        )
        self.cached_path, self.cached_images = None, None
        self.images = None

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        if self.images is not None:
            x = self.images[index].float() / 255
            return x if self.zero_one else 2 * x - 1
        row = self.frame.iloc[index]
        if row.source == "real":
            with Image.open(row.real_path) as im:
                x = (self.transform(im.convert("RGB")) * 255).round().to(torch.uint8)
        else:
            path = image_shard(self.work, row.tensor_path)
            if not path.exists():
                if self.fresh:
                    raise FileNotFoundError(f"Run generate first: {path}")
                path = Path(row.tensor_path)
            if path != self.cached_path:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                self.cached_images = (
                    payload["images"] if isinstance(payload, dict) else payload
                )
                self.cached_path = path
            x = self.cached_images[int(row.idx_in_shard)]
        x = x.float() / 255
        return x if self.zero_one else 2 * x - 1

    def preload(self):
        """Decode each selected image once, as in the reported pixel fits."""
        images = torch.empty((len(self), 3, 64, 64), dtype=torch.uint8)
        for i in range(len(self)):
            x = self[i]
            if not self.zero_one:
                x = (x + 1) / 2
            images[i] = (255 * x).round().clamp(0, 255).to(torch.uint8)
        self.images = images
        self.cached_path, self.cached_images = None, None
        return self


def feature_values(frame, args, spec):
    """Gather feature rows in manifest order, using verified archived caches if allowed."""
    values = np.empty((len(frame), 2048), dtype=np.float32)
    for key, group in frame.reset_index(drop=True).groupby("feature_key", sort=False):
        path = args.work_root / "features" / f"{key}.pt"
        if not path.exists():
            if args.fresh:
                raise FileNotFoundError(f"Run features first: {path}")
            item = spec["full"][key]
            path = verify(item["original_features"], item["original_features_sha256"])
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        all_values = payload.get("features", payload.get("z"))
        values[group.index] = all_values[group.feature_row.to_numpy(dtype=int)].numpy()
        del payload, all_values
    return values
