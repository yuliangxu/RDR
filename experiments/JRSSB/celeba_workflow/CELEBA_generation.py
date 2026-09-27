"""Regenerate fixed latent-seed images and extract the pinned Inception features."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from CELEBA_agent3 import (
    atomic_torch_save,
    generate_stylegan_batch,
    load_stylegan_generator,
    prepare_generator_assets,
)
from CELEBA_data import load_yaml
from CELEBA_inception import build_inception, embed_batches
from .CELEBA_data import ManifestImages, image_shard
from .CELEBA_paths import HERE


def generate(args, spec):
    """Regenerate complete image shards using the recorded latent seeds and truncation."""
    config = load_yaml(HERE / "configs/agent3.yaml")["generator"]
    provenance = prepare_generator_assets(config, args.work_root, allow_download=False)
    generator = load_stylegan_generator(provenance, config, args.device)
    for key, item in spec["full"].items():
        if key.startswith("real_") or (
            args.branch and not key.startswith(args.branch + "_")
        ):
            continue
        frame = pd.read_csv(item["manifest"])
        for original, group in frame.groupby("tensor_path", sort=False):
            path = image_shard(args.work_root, original)
            if path.exists():
                continue
            group = group.sort_values("idx_in_shard")
            if not np.array_equal(group.idx_in_shard, np.arange(len(group))):
                raise RuntimeError(
                    "Generation requires complete original shard manifests"
                )
            if args.limit:
                group = group.iloc[: args.limit]
            batches = []
            for start in range(0, len(group), 64):
                part = group.iloc[start : start + 64]
                batches.append(
                    generate_stylegan_batch(
                        generator,
                        part.latent_seed.astype(int).tolist(),
                        float(part.truncation_psi.iloc[0]),
                        config["noise_mode"],
                    )
                )
            atomic_torch_save(
                {"images": torch.cat(batches), "source_ids": group.source_id.tolist()},
                path,
            )
            print(path, flush=True)
            if args.limit:
                return


def extract_features(args, spec):
    """Embed each full source manifest with the pinned 2048-dimensional Inception model."""
    model = build_inception(2048, args.device)
    for key, item in spec["full"].items():
        if args.branch and not key.startswith(args.branch + "_"):
            continue
        frame = pd.read_csv(item["manifest"])
        if args.limit:
            frame = frame.iloc[: args.limit]
        dataset = ManifestImages(frame, args.work_root, args.fresh, zero_one=True)
        values = embed_batches(
            model, DataLoader(dataset, batch_size=64, num_workers=0), args.device
        )
        atomic_torch_save(
            {"features": values, "backend": "pytorch_fid_0.3.0"},
            args.work_root / "features" / f"{key}.pt",
        )
        print(f"Features {key}: {len(values)}", flush=True)
        if args.limit:
            return
