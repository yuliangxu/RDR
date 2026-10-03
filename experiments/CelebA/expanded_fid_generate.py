#!/usr/bin/env python3
"""Generate four frozen 20k CelebA pools without changing the approved pair.

Preparation is offline: the pinned external checkout and checkpoint must
already exist, and the checkout must already be clean at the expected HEAD.
Generation publishes each image/feature shard together with a hash receipt
by atomic directory rename, so interrupted shards are never resume inputs.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "JRSSB"))

import numpy as np
import pandas as pd
import torch
import yaml

from CELEBA_agent3 import (atomic_csv, atomic_json, atomic_torch_save,
    generate_stylegan_batch, load_stylegan_generator, prepare_generator_assets)
from CELEBA_data import manifest_fingerprint, set_seed, sha256_file
from CELEBA_inception import build_inception, embed_batches, inception_provenance

DEFAULT_CONFIG = ROOT / "experiments/JRSSB/configs/agent3.yaml"
SOURCE_FILES = ("experiments/CelebA/expanded_fid_generate.py",
                "experiments/CelebA/expanded_fid_report.py", "experiments/CelebA/expanded_fid.slurm",
                "experiments/JRSSB/CELEBA_agent3.py", "experiments/JRSSB/CELEBA_data.py",
                "experiments/JRSSB/CELEBA_inception.py")
TARGETS = (("validation", "lower", 20_000_000), ("validation", "upper", 21_000_000),
           ("final", "lower", 22_000_000), ("final", "upper", 23_000_000))
COUNT, SHARD_SIZE, BATCH_SIZE, DIMENSIONS = 20_000, 1_000, 64, 2048


def _read(path):
    return json.loads(Path(path).read_text())


def _git(repository, *args):
    return subprocess.run(["git", "-C", str(repository), *args], check=True, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()


def verify_repository(generator):
    """Check before the historical helper, which would otherwise checkout HEAD."""
    repository = Path(generator["repository_path"])
    if not repository.is_dir():
        raise FileNotFoundError(f"Missing pinned generator checkout: {repository}")
    if _git(repository, "rev-parse", "HEAD") != generator["repository_commit"]:
        raise RuntimeError("Generator HEAD differs from the pinned commit; no checkout was attempted")
    if _git(repository, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("Generator checkout has modified or untracked files")


def _packages():
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "torchvision", "numpy", "pandas", "pytorch-fid")}
    if packages["pytorch-fid"] != "0.3.0":
        raise RuntimeError("Expanded FID generation requires pytorch-fid==0.3.0")
    return packages


def audit_seed_namespaces(source_root, tasks):
    """Check all historical manifest seed columns, not only selected subsets."""
    paths = sorted((Path(source_root) / "manifests").rglob("*.csv"))
    if not paths:
        raise FileNotFoundError("No historical manifests available for the seed-overlap audit")
    observed, receipts, rows = set(), {}, 0
    for path in paths:
        columns = pd.read_csv(path, nrows=0).columns
        seed_columns = [name for name in ("latent_seed", "seed") if name in columns]
        receipts[str(path.resolve())] = sha256_file(path)
        if not seed_columns:
            continue
        frame = pd.read_csv(path, usecols=seed_columns)
        rows += len(frame)
        for name in seed_columns:
            numeric = pd.to_numeric(frame[name].dropna(), errors="raise").to_numpy()
            if not np.isfinite(numeric).all() or np.any(numeric != np.floor(numeric)):
                raise ValueError(f"Invalid historical seed in {path}")
            observed.update(map(int, numeric))
    assigned = set()
    for task in tasks:
        proposed = set(range(task["seed_start"], task["seed_start"] + task["count"]))
        if proposed & observed or proposed & assigned:
            raise RuntimeError(f"Seed namespace overlaps prior images or another target: {task}")
        assigned.update(proposed)
    return {"manifest_sha256": receipts, "manifest_count": len(paths), "seed_rows": rows,
            "distinct_historical_seeds": len(observed),
            "historical_seed_min": min(observed) if observed else None,
            "historical_seed_max": max(observed) if observed else None,
            "new_seed_count": len(assigned), "overlap_count": 0}


def _validate_pair(config, pair):
    if pair.get("status") != "approved_pair":
        raise ValueError("A frozen approved generator pair is required")
    for field in ("checkpoint_sha256", "repository_commit", "name"):
        if config["generator"][field] != pair["generator_assets"][field]:
            raise ValueError(f"Generator configuration disagrees with pair lock: {field}")
    if config["generator"]["noise_mode"] != pair["noise_mode"] or pair["noise_mode"] != "const":
        raise ValueError("Generation requires the approved constant-noise setting")
    if bool(config["generator"].get("force_reference_ops", False)) != pair["force_reference_ops"]:
        raise ValueError("Reference-op setting differs from the frozen pair")
    if config["fid_features"]["package_version"] != "0.3.0" or int(config["fid_features"]["dimensions"]) != DIMENSIONS:
        raise ValueError("Pinned FID backend or dimensions changed")
    for branch in ("lower", "upper"):
        item = config["full_generation"]["sources"][branch]
        if any(item[field] != pair["branches"][branch][field] for field in ("source", "truncation_psi")):
            raise ValueError(f"Generator branch differs from the pair lock: {branch}")


def _copy_immutable(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256_file(source) != sha256_file(target):
            raise RuntimeError(f"Existing frozen input differs: {target}")
    else:
        shutil.copyfile(source, target)
    return sha256_file(target)


def prepare(output_dir, config_path=DEFAULT_CONFIG, pair_lock=None, source_root=None):
    output = Path(output_dir).resolve()
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text())
    source_root = Path(source_root or config["output_root"]).resolve()
    pair_lock = Path(pair_lock or source_root / "locks/generator_pair_lock.json").resolve()
    if (output / "plan.json").exists():
        plan = load_plan(output)
        if plan["config_sha256"] != sha256_file(config_path) or plan["pair_lock_sha256"] != sha256_file(pair_lock):
            raise RuntimeError("Existing generation plan differs from requested inputs")
        return plan
    pair = _read(pair_lock)
    _validate_pair(config, pair)
    verify_repository(config["generator"])
    packages, inception = _packages(), inception_provenance()
    tasks = [{"task_index": i, "role": role, "branch": branch, "seed_start": seed,
              "count": COUNT, "source": pair["branches"][branch]["source"],
              "truncation_psi": pair["branches"][branch]["truncation_psi"]}
             for i, (role, branch, seed) in enumerate(TARGETS)]
    audit = audit_seed_namespaces(source_root, tasks)
    output.mkdir(parents=True, exist_ok=True)
    # All prechecks happen before the legacy helper; allow_download=False
    # prevents checkpoint replacement/download and HEAD already matches.
    assets = prepare_generator_assets(config["generator"], output, allow_download=False)
    files = {"inputs/agent3.yaml": _copy_immutable(config_path, output / "inputs/agent3.yaml"),
             "inputs/generator_pair_lock.json": _copy_immutable(pair_lock, output / "inputs/generator_pair_lock.json")}
    for relative in SOURCE_FILES:
        files["source/" + relative] = _copy_immutable(ROOT / relative, output / "source" / relative)
    files["provenance/generator_assets.json"] = sha256_file(output / "provenance/generator_assets.json")
    plan = {"schema": 1, "status": "prepared", "study": "expanded_fixed_pair_fid_samples",
            "tasks": tasks, "total_new_images": COUNT * len(tasks), "shard_size": SHARD_SIZE,
            "batch_size": BATCH_SIZE, "dimensions": DIMENSIONS, "files": files,
            "source_root": str(source_root), "original_config_path": str(config_path),
            "config_sha256": files["inputs/agent3.yaml"], "pair_lock_path": str(pair_lock),
            "pair_lock_sha256": files["inputs/generator_pair_lock.json"], "generator_assets": assets,
            "force_reference_ops": bool(config["generator"].get("force_reference_ops", False)),
            "inception": inception, "packages": packages, "seed_audit": audit,
            "image_quantization": "(generator_output*127.5+128).clamp(0,255).to(uint8)",
            "feature_input": "saved uint8 NCHW RGB divided by255; canonical pytorch-fid resize/normalization",
            "sample_scope": "additional fixed-generator samples; no truncation or model selection"}
    atomic_json(plan, output / "plan.json")
    (output / "plan.sha256").write_text(sha256_file(output / "plan.json") + "\n")
    return plan


def load_plan(output_dir):
    output = Path(output_dir).resolve()
    if sha256_file(output / "plan.json") != (output / "plan.sha256").read_text().strip():
        raise RuntimeError("Generation plan checksum mismatch")
    plan = _read(output / "plan.json")
    for relative, expected in plan["files"].items():
        if sha256_file(output / relative) != expected:
            raise RuntimeError(f"Frozen generation input changed: {relative}")
        if relative.startswith("source/") and sha256_file(ROOT / Path(relative).relative_to("source")) != expected:
            raise RuntimeError("Executing source differs from frozen source; use the saved source entrypoint")
    if sha256_file(plan["pair_lock_path"]) != plan["pair_lock_sha256"]:
        raise RuntimeError("Original generator pair lock changed")
    config = yaml.safe_load((output / "inputs/agent3.yaml").read_text())
    _validate_pair(config, _read(output / "inputs/generator_pair_lock.json"))
    verify_repository(config["generator"])
    if sha256_file(config["generator"]["checkpoint_path"]) != plan["generator_assets"]["checkpoint_sha256"]:
        raise RuntimeError("Generator checkpoint changed")
    if _packages() != plan["packages"] or inception_provenance() != plan["inception"]:
        raise RuntimeError("Generation environment or Inception weights changed")
    for path, expected in plan["seed_audit"]["manifest_sha256"].items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"Audited historical manifest changed: {path}")
    observed_paths = {str(path.resolve()) for path in (Path(plan["source_root"]) / "manifests").rglob("*.csv")}
    if observed_paths != set(plan["seed_audit"]["manifest_sha256"]):
        raise RuntimeError("Historical manifest inventory changed after seed audit")
    return plan


def _context(plan, task, plan_sha):
    return {"schema": 1, "plan_sha256": plan_sha, "pair_lock_sha256": plan["pair_lock_sha256"],
            "role": task["role"], "branch": task["branch"], "source": task["source"],
            "truncation_psi": task["truncation_psi"],
            "checkpoint_sha256": plan["generator_assets"]["checkpoint_sha256"],
            "repository_commit": plan["generator_assets"]["repository_commit"],
            "force_reference_ops": plan["force_reference_ops"],
            "noise_mode": "const", "inception": plan["inception"]}


def _load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _verify_shard(directory, context, seeds):
    receipt = _read(directory / "receipt.json")
    expected = {**context, "seed_start": seeds[0], "count": len(seeds)}
    if receipt["context"] != expected or receipt.get("status") != "generated_and_validated":
        raise RuntimeError(f"Shard receipt metadata mismatch: {directory}")
    for name in ("images.pt", "features.pt"):
        if sha256_file(directory / name) != receipt["sha256"][name]:
            raise RuntimeError(f"Shard checksum mismatch: {directory / name}")
    image_payload, feature_payload = [_load_tensor(directory / name) for name in ("images.pt", "features.pt")]
    for payload in (image_payload, feature_payload):
        if payload["context"] != expected or payload["latent_seeds"].tolist() != seeds:
            raise RuntimeError(f"Shard payload differs from manifest seeds/provenance: {directory}")
    images, features = image_payload["images"], feature_payload["features"]
    if images.dtype != torch.uint8 or tuple(images.shape) != (len(seeds), 3, 64, 64):
        raise RuntimeError("Invalid generated image shard")
    if features.dtype != torch.float32 or tuple(features.shape) != (len(seeds), DIMENSIONS) or not torch.isfinite(features).all():
        raise RuntimeError("Invalid generated feature shard")
    return receipt, features


def _frame(task, output):
    rows = []
    key = f"extra_{task['role']}_{task['branch']}"
    for offset in range(task["count"]):
        seed = task["seed_start"] + offset
        shard = f"shard_{offset // SHARD_SIZE:04d}"
        rows.append({"source": task["source"], "role": task["role"],
                     "source_id": f"{task['source']}_extra_{task['role']}_seed_{seed:09d}",
                     "real_path": "", "shard": shard, "idx_in_shard": offset % SHARD_SIZE,
                     "tensor_path": str(output / "shards" / key / shard / "images.pt"),
                     "latent_seed": seed, "seed": seed, "truncation_psi": task["truncation_psi"],
                     "psi": task["truncation_psi"]})
    return pd.DataFrame(rows)


def verify_complete(output, task, plan, plan_sha):
    key = f"extra_{task['role']}_{task['branch']}"
    lock = _read(output / "locks" / f"{key}.json")
    if any(lock.get(name) != value for name, value in
           (("status", "generated_and_validated"), ("role", task["role"]), ("branch", task["branch"]),
            ("count", COUNT), ("pair_lock_sha256", plan["pair_lock_sha256"]), ("plan_sha256", plan_sha))):
        raise RuntimeError("Completed extra-pool receipt metadata mismatch")
    for path_key, sha_key in (("feature_cache_path", "feature_cache_sha256"), ("manifest_path", "manifest_sha256")):
        if sha256_file(lock[path_key]) != lock[sha_key]:
            raise RuntimeError("Completed extra-pool artifact checksum mismatch")
    frame = pd.read_csv(lock["manifest_path"])
    if (len(frame) != COUNT or frame.source_id.duplicated().any()
            or manifest_fingerprint(frame) != lock["manifest_fingerprint"]
            or frame.latent_seed.tolist() != list(range(task["seed_start"], task["seed_start"] + COUNT))):
        raise RuntimeError("Completed manifest count, fingerprint, or seed order mismatch")
    context = _context(plan, task, plan_sha)
    if len(lock["shards"]) != COUNT // SHARD_SIZE:
        raise RuntimeError("Incomplete shard inventory")
    for index, item in enumerate(lock["shards"]):
        expected_directory = output / "shards" / key / f"shard_{index:04d}"
        seeds = list(range(task["seed_start"] + index * SHARD_SIZE, task["seed_start"] + (index + 1) * SHARD_SIZE))
        receipt, _ = _verify_shard(expected_directory, context, seeds)
        if item != {"path": str(expected_directory), "receipt_sha256": sha256_file(expected_directory / "receipt.json"),
                    "image_sha256": receipt["sha256"]["images.pt"], "feature_sha256": receipt["sha256"]["features.pt"]}:
            raise RuntimeError("Completed receipt shard inventory mismatch")
    return lock


def generate(output_dir, task_index, device="cuda"):
    output = Path(output_dir).resolve()
    plan = load_plan(output)
    if not 0 <= int(task_index) < len(plan["tasks"]):
        raise ValueError("task-index must be in0..3")
    task = plan["tasks"][int(task_index)]
    if task["count"] != COUNT or plan["shard_size"] != SHARD_SIZE or plan["batch_size"] != BATCH_SIZE:
        raise ValueError("Generation count or canonical batch/shard size changed")
    plan_sha = sha256_file(output / "plan.json")
    key = f"extra_{task['role']}_{task['branch']}"
    if (output / "locks" / f"{key}.json").exists():
        return verify_complete(output, task, plan, plan_sha)
    config = yaml.safe_load((output / "inputs/agent3.yaml").read_text())
    device = torch.device(device)
    set_seed(task["seed_start"])
    generator = load_stylegan_generator(plan["generator_assets"], config["generator"], device)
    inception = build_inception(DIMENSIONS, device)
    if inception_provenance() != plan["inception"]:
        raise RuntimeError("Loaded Inception weights differ from preparation")
    context, features, shards = _context(plan, task, plan_sha), [], []
    shard_root = output / "shards" / key
    shard_root.mkdir(parents=True, exist_ok=True)
    for offset in range(0, COUNT, SHARD_SIZE):
        directory = shard_root / f"shard_{offset // SHARD_SIZE:04d}"
        seeds = list(range(task["seed_start"] + offset, task["seed_start"] + offset + SHARD_SIZE))
        if not directory.exists():
            staging = Path(tempfile.mkdtemp(prefix=".pending-shard-", dir=shard_root))
            images = torch.cat([generate_stylegan_batch(generator, seeds[start:start + BATCH_SIZE],
                               task["truncation_psi"], "const") for start in range(0, len(seeds), BATCH_SIZE)])
            embedded = embed_batches(inception, (chunk.float().div(255) for chunk in images.split(BATCH_SIZE)),
                                     device, progress_every=0)
            shard_context = {**context, "seed_start": seeds[0], "count": len(seeds)}
            shared = {**context, "context": shard_context, "latent_seeds": torch.tensor(seeds, dtype=torch.int64),
                      "force_reference_ops": bool(config["generator"].get("force_reference_ops", False))}
            atomic_torch_save({**shared, "images": images}, staging / "images.pt")
            atomic_torch_save({**shared, "features": embedded}, staging / "features.pt")
            receipt = {"status": "generated_and_validated", "context": shard_context,
                       "sha256": {name: sha256_file(staging / name) for name in ("images.pt", "features.pt")}}
            atomic_json(receipt, staging / "receipt.json")
            _verify_shard(staging, context, seeds)
            os.rename(staging, directory)
        receipt, embedded = _verify_shard(directory, context, seeds)
        features.append(embedded)
        shards.append({"path": str(directory), "receipt_sha256": sha256_file(directory / "receipt.json"),
                       "image_sha256": receipt["sha256"]["images.pt"], "feature_sha256": receipt["sha256"]["features.pt"]})
        print(f"{key}: verified {offset + SHARD_SIZE:,}/{COUNT:,}", flush=True)
    frame = _frame(task, output)
    frame["checkpoint"] = plan["generator_assets"]["checkpoint_path"]
    frame["checkpoint_sha256"] = plan["generator_assets"]["checkpoint_sha256"]
    if len(frame) != COUNT or frame.source_id.duplicated().any() or frame.latent_seed.duplicated().any():
        raise RuntimeError("Extra-pool manifest is not exactly20,000 unique observations")
    manifest = output / "manifests" / f"{key}.csv"
    if manifest.exists():
        if manifest.read_text() != frame.to_csv(index=False):
            raise RuntimeError("Existing extra-pool manifest differs")
    else:
        atomic_csv(frame, manifest)
    fingerprint = manifest_fingerprint(frame)
    cache = output / "features" / f"{key}_2048.pt"
    values = torch.cat(features)
    feature_context = {**context, "count": COUNT, "manifest_fingerprint": fingerprint}
    if cache.exists():
        saved = _load_tensor(cache)
        if saved["context"] != feature_context or saved["source_ids"] != frame.source_id.tolist() or not torch.equal(saved["features"], values):
            raise RuntimeError("Existing combined feature cache differs from verified shards")
    else:
        atomic_torch_save({"features": values, "source_ids": frame.source_id.tolist(),
                          "manifest_fingerprint": fingerprint, "manifest_path": str(manifest),
                          "context": feature_context, "inception_provenance": plan["inception"]}, cache)
    lock = {"status": "generated_and_validated", "role": task["role"], "branch": task["branch"],
            "count": COUNT, "seed_start": task["seed_start"], "seed_end_inclusive": task["seed_start"] + COUNT - 1,
            "feature_cache_path": str(cache), "feature_cache_sha256": sha256_file(cache),
            "manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest),
            "manifest_fingerprint": fingerprint, "pair_lock_sha256": plan["pair_lock_sha256"],
            "plan_sha256": plan_sha, "inception": plan["inception"], "shards": shards,
            "generator_assets": plan["generator_assets"], "truncation_psi": task["truncation_psi"]}
    atomic_json(lock, output / "locks" / f"{key}.json")
    return verify_complete(output, task, plan, plan_sha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output-dir", type=Path, required=True)
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep.add_argument("--source-root", type=Path)
    prep.add_argument("--pair-lock", type=Path)
    run = commands.add_parser("generate")
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--task-index", type=int, required=True)
    run.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = prepare(args.output_dir, args.config, args.pair_lock, args.source_root) if args.command == "prepare" else generate(args.output_dir, args.task_index, args.device)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir),
                      "count": result.get("count", result.get("total_new_images"))}), flush=True)


if __name__ == "__main__":
    main()
