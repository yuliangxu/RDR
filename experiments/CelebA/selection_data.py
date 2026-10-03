"""Freeze CelebA source rows and prepare verified, reusable selection inputs.

Historical files are read only. Preparation reads test *manifests*, but test
features/images remain inaccessible until a hash-verified selection freeze.
Arrays contain raw pool3 features or uint8 NCHW pixels; scaling belongs to the
trainer and must be fitted on training observations only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_ROLES = ("train", "earlystop", "selection_calibration", "selection_evaluation")
TEST_ROLES = ("test_calibration", "test_evaluation")
SOURCES = ("real", "lower", "upper")
DEFAULT_SOURCE_ROOT = "/cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _receipt(path, expected=None):
    path = Path(path).resolve()
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise RuntimeError(f"SHA256 mismatch: {path}")
    stat = path.stat()
    return dict(path=str(path), sha256=actual, size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def verify_receipt(receipt, full=False, base=None):
    """Cheap immutable-cache check; use full=True for an independent hash audit."""
    path = Path(receipt["path"])
    if not path.is_absolute():
        if base is None:
            raise ValueError("A run directory is required for a relative prepared receipt")
        path = Path(base) / path
    stat = path.stat()
    if stat.st_size != receipt["size"] or stat.st_mtime_ns != receipt["mtime_ns"]:
        raise RuntimeError(f"Prepared input changed: {path}")
    if full and sha256_file(path) != receipt["sha256"]:
        raise RuntimeError(f"Prepared input SHA256 mismatch: {path}")
    return path


def _output_receipt(path, output):
    receipt = _receipt(path)
    receipt["path"] = str(Path(path).relative_to(output))
    return receipt


def _fingerprint(frame):
    columns = ["source", "role", "source_id", "real_path", "shard", "idx_in_shard"]
    normalized = frame[columns].astype("string").fillna("")
    return hashlib.sha256(normalized.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _resolve(path, source_root, path_map):
    """Support relocated archives without changing any historical manifest."""
    mapping = {DEFAULT_SOURCE_ROOT: str(source_root), **dict(path_map)}
    value = str(path)
    for old in sorted(mapping, key=len, reverse=True):
        if value == old or value.startswith(old.rstrip("/") + "/"):
            return Path(mapping[old]) / value[len(old):].lstrip("/")
    return Path(value)


def _identity_parts(frame, fractions, seed):
    """Shuffle whole identities, then cut closest to cumulative image targets."""
    if frame.identity.isna().any():
        raise ValueError("Real identity is missing")
    if not np.isclose(sum(fractions), 1) or min(fractions) <= 0:
        raise ValueError("Positive partition fractions must sum to one")
    identities = np.asarray(sorted(frame.identity.unique()))
    if len(identities) < len(fractions):
        raise ValueError("Too few identities to populate independent roles")
    identities = np.random.default_rng(seed).permutation(identities)
    counts = frame.groupby("identity").size().reindex(identities).to_numpy()
    cumulative = np.cumsum(counts)
    boundaries, previous = [0], 0
    for j, fraction in enumerate(np.cumsum(fractions)[:-1]):
        # Leave at least one identity in every remaining role.
        choices = np.arange(previous + 1, len(identities) - (len(fractions) - j - 1) + 1)
        boundary = int(choices[np.argmin(np.abs(cumulative[choices - 1] - fraction * len(frame)))])
        boundaries.append(boundary)
        previous = boundary
    boundaries.append(len(identities))
    return [frame[frame.identity.isin(identities[a:b])].copy().reset_index(drop=True)
            for a, b in zip(boundaries[:-1], boundaries[1:])]


def _row_parts(frame, counts, seed):
    if sum(counts) != len(frame) or min(counts) <= 0:
        raise ValueError("Partition counts must use every row and populate every role")
    order = np.random.default_rng(seed).permutation(len(frame))
    edges = np.cumsum([0] + list(counts))
    return [frame.iloc[np.sort(order[a:b])].copy().reset_index(drop=True)
            for a, b in zip(edges[:-1], edges[1:])]


def _smoke_subset(frame, source, count, seed):
    if source != "real":
        order = np.random.default_rng(seed).permutation(len(frame))[:count]
        return frame.iloc[np.sort(order)].copy().reset_index(drop=True)
    identities = np.random.default_rng(seed).permutation(sorted(frame.identity.unique()))
    sizes = frame.groupby("identity").size().reindex(identities).to_numpy()
    stop = min(len(identities), int(np.searchsorted(np.cumsum(sizes), count)) + 1)
    return frame[frame.identity.isin(identities[:stop])].copy().reset_index(drop=True)


def _assert_disjoint(roles):
    for source in SOURCES:
        previous_ids, previous_groups = set(), set()
        key = "identity" if source == "real" else "latent_seed"
        for role, distributions in roles.items():
            frame = distributions[source]
            if frame.empty or frame.source_id.isna().any() or frame.source_id.duplicated().any():
                raise ValueError(f"Empty, missing or duplicate source IDs: {source}/{role}")
            if frame[key].isna().any():
                raise ValueError(f"Missing {key}: {source}/{role}")
            ids, groups = set(frame.source_id), set(frame[key])
            if previous_ids & ids or previous_groups & groups:
                raise ValueError(f"Overlapping source IDs or {key}: {source}/{role}")
            if source != "real" and frame[key].duplicated().any():
                raise ValueError(f"Duplicate generated seeds: {source}/{role}")
            previous_ids.update(ids)
            previous_groups.update(groups)


def _freeze_roles(original, seed, smoke_rows=None):
    roles = {role: {} for role in TRAIN_ROLES + TEST_ROLES}
    for offset, source in enumerate(SOURCES):
        roles["train"][source] = original["train"][source].copy()
        validation = original["validation"][source]
        test = pd.concat([original["design"][source], original["test"][source]], ignore_index=True)
        if source == "real":
            val_parts = _identity_parts(validation, [.5, .25, .25], seed)
            test_parts = _identity_parts(test, [.5, .5], seed + 10)
        else:
            n = len(validation)
            val_parts = _row_parts(validation, [n // 2, n // 4, n - n // 2 - n // 4], seed + offset)
            test_parts = _row_parts(test, [len(test) // 2, len(test) - len(test) // 2], seed + 10 + offset)
        for role, frame in zip(TRAIN_ROLES[1:], val_parts):
            roles[role][source] = frame
        for role, frame in zip(TEST_ROLES, test_parts):
            roles[role][source] = frame
        for index, (role, distributions) in enumerate(roles.items()):
            frame = distributions[source]
            if smoke_rows is not None:
                frame = _smoke_subset(frame, source, smoke_rows, seed + 100 + index * 3 + offset)
            distributions[source] = frame.assign(role=role).reset_index(drop=True)
    _assert_disjoint(roles)
    return roles


def _audit_sources(source_root, cfg):
    resolve = lambda path: _resolve(path, source_root, cfg.get("path_map", {}))
    lock_names = ("real_generator_rdr_selection.json", "agent3_final_test_selection.json", "real_generator_real_pixel_cache.json")
    locks, receipts = {}, {}
    for name in lock_names:
        path = source_root / "locks" / name
        receipts[str(path)] = _receipt(path)
        locks[name] = _json(path)
    primary, final, pixels = (locks[name] for name in lock_names)
    primary_sha = receipts[str(source_root / "locks" / lock_names[0])]["sha256"]
    for record, field in ((final, "design_selection_lock_sha256"), (pixels, "selection_lock_sha256")):
        if record[field] != primary_sha:
            raise RuntimeError("Historical input locks do not describe the same selection")
    pair_selection_path = resolve(primary["pair_selection_lock_path"])
    receipts[str(pair_selection_path)] = _receipt(pair_selection_path, primary["pair_selection_lock_sha256"])
    pair_selection = _json(pair_selection_path)
    pair_path = source_root / "locks" / "generator_pair_lock.json"
    receipts[str(pair_path)] = _receipt(pair_path, pair_selection["pair_lock_sha256"])
    pair = _json(pair_path)
    expected_generator = {field: pair["generator_assets"][field] for field in ("checkpoint_sha256", "repository_commit")}
    expected_generator.update(force_reference_ops=pair["force_reference_ops"], noise_mode=pair["noise_mode"])
    originals, inputs = {}, {}
    for origin in ("train", "validation", "design", "test"):
        originals[origin] = {}
        for source in SOURCES:
            item = (final["selection"] if origin == "test" else primary["selection"][origin])[source]
            selected_path, full_path = resolve(item["selected_manifest_path"]), resolve(item["input_manifest_path"])
            for path, expected in ((selected_path, item["selected_manifest_sha256"]), (full_path, item.get("input_manifest_sha256"))):
                receipts[str(path)] = _receipt(path, expected)
            selected, full = pd.read_csv(selected_path), pd.read_csv(full_path)
            if _fingerprint(selected) != item["selected_manifest_fingerprint"] or _fingerprint(full) != item["input_manifest_fingerprint"]:
                raise RuntimeError(f"Manifest fingerprint mismatch: {source}/{origin}")
            expected_count = {"train": 60000, "validation": 20000, "design": 9000, "test": 9000}[origin]
            if len(selected) != expected_count or len(full) != item["pool_size"]:
                raise RuntimeError(f"Historical source counts changed: {source}/{origin}")
            if full.source_id.duplicated().any() or selected.source_id.duplicated().any():
                raise RuntimeError("Source IDs must uniquely identify feature rows")
            row_map = pd.Series(np.arange(len(full)), index=full.source_id)
            positions = selected.source_id.map(row_map)
            if positions.isna().any():
                raise RuntimeError("Selected source ID absent from input manifest")
            if "feature_row" in selected and not np.array_equal(selected.feature_row, positions):
                raise RuntimeError("Recorded feature rows disagree with source IDs")
            key = f"{source}_{origin}"
            selected = selected.assign(feature_row=positions.to_numpy(dtype=int), feature_key=key,
                                       original_role=origin, original_selected_row=np.arange(len(selected)))
            originals[origin][source] = selected
            inputs[key] = dict(feature_path=str(resolve(item["feature_cache_path"])), feature_sha256=item["feature_cache_sha256"],
                               input_fingerprint=item["input_manifest_fingerprint"], selected_fingerprint=item["selected_manifest_fingerprint"],
                               selected_manifest=receipts[str(selected_path)], input_manifest=receipts[str(full_path)],
                               input_count=len(full), selected_count=len(selected))
            if source != "real":
                branch = pair["branches"][source]
                if not selected.source.eq(branch["source"]).all() or not np.allclose(selected.truncation_psi, branch["truncation_psi"]):
                    raise RuntimeError("Generated source manifest disagrees with the frozen generator pair")
                inputs[key]["expected_generator"] = {**expected_generator, "truncation_psi": branch["truncation_psi"]}
            if source == "real":
                pixel = final["real_pixel_cache"] if origin == "test" else pixels["roles"][origin]
                inputs[key].update(pixel_path=str(resolve(pixel.get("path", pixel.get("image_cache_path")))),
                                   pixel_sha256=pixel.get("sha256", pixel.get("image_cache_sha256")))
                if origin != "test" and pixel["manifest_sha256"] != item["selected_manifest_sha256"]:
                    raise RuntimeError("Real pixel cache refers to different source rows")
    # Original design and test identities are independently allocated, too.
    _assert_disjoint(originals)
    return originals, inputs, receipts


def _torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def _gather_features(frame, inputs, target, receipts):
    for key, group in frame.groupby("feature_key", sort=False):
        item = inputs[key]
        path = item["feature_path"]
        if path not in receipts:
            receipts[path] = _receipt(path, item["feature_sha256"])
        payload = _torch_load(path)
        if payload.get("manifest_fingerprint") != item["input_fingerprint"]:
            raise RuntimeError("Feature cache fingerprint does not match its row manifest")
        values = payload.get("features", payload.get("z"))
        if tuple(values.shape) != (item["input_count"], 2048):
            raise RuntimeError("Expected full 2048-dimensional pool3 feature cache")
        for start in range(0, len(group), 2048):
            chunk = group.iloc[start:start + 2048]
            gathered = values[chunk.feature_row.to_numpy(dtype=int)].numpy()
            if not np.isfinite(gathered).all():
                raise RuntimeError("Nonfinite feature input")
            target[chunk.index] = gathered


def _gather_pixels(frame, source, inputs, target, receipts, source_root, path_map):
    import torch
    if source == "real":
        if all(inputs[key].get("pixel_from_jpeg") for key in frame.feature_key.unique()):
            from experiments.CelebA.expanded_selection_data import gather_real_pixels
            gather_real_pixels(frame, target, receipts)
            return
        for key, group in frame.groupby("feature_key", sort=False):
            item, path = inputs[key], inputs[key]["pixel_path"]
            if path not in receipts:
                receipts[path] = _receipt(path, item["pixel_sha256"])
            payload = _torch_load(path)
            if payload.get("manifest_fingerprint") != item["selected_fingerprint"]:
                raise RuntimeError("Real pixel cache row order differs from selected manifest")
            images = payload["images"]
            if tuple(images.shape) != (item["selected_count"], 3, 64, 64) or images.dtype != torch.uint8:
                raise RuntimeError("Real pixel cache must be selected uint8 NCHW images")
            for start in range(0, len(group), 2048):
                chunk = group.iloc[start:start + 2048]
                target[chunk.index] = images[chunk.original_selected_row.to_numpy(dtype=int)].numpy()
        return
    for original_path, group in frame.groupby("tensor_path", sort=False):
        path = str(_resolve(original_path, source_root, path_map))
        expected_hashes = {inputs[key].get("shard_sha256", {}).get(path)
                           for key in group.feature_key.unique()} - {None}
        if len(expected_hashes) > 1:
            raise RuntimeError("Conflicting source shard hashes")
        if path not in receipts:
            expected = next(iter(expected_hashes), None)
            receipts[path] = {**_receipt(path, expected), "hash_basis": "completion_receipt" if expected else "observed_at_preparation"}
        payload = _torch_load(path)
        for key in group.feature_key.unique():
            for field, expected in inputs[key]["expected_generator"].items():
                if payload.get(field) != expected:
                    raise RuntimeError(f"Generated shard {field} differs from the frozen generator")
        images, seeds = payload["images"], payload["latent_seeds"]
        indices = group.idx_in_shard.to_numpy(dtype=int)
        if images.dtype != torch.uint8 or tuple(images.shape[1:]) != (3, 64, 64):
            raise RuntimeError("Generated shard must contain uint8 NCHW images")
        if not np.array_equal(seeds[indices].numpy(), group.latent_seed.to_numpy(dtype=np.int64)):
            raise RuntimeError("Generated shard seeds do not match selected rows")
        if not np.allclose(group.truncation_psi, payload["truncation_psi"]) or payload["noise_mode"] != "const":
            raise RuntimeError("Generated shard does not match the frozen generator branch")
        target[group.index] = images[indices].numpy()


def _build_assets(output, metadata, roles):
    receipts = metadata["source_receipts"]
    for level in metadata["levels"]:
        if level not in ("feature", "pixel"):
            raise ValueError(f"Unknown representation: {level}")
        metadata["arrays"].setdefault(level, {})
        for role in roles:
            metadata["arrays"][level].setdefault(role, {})
            for source in SOURCES:
                manifest = metadata["roles"][role][source]
                frame = pd.read_csv(verify_receipt(manifest["receipt"], full=True, base=output))
                path = output / "data" / "arrays" / level / role / f"{source}.npy"
                receipt_path = path.with_suffix(".json")
                if path.exists() and receipt_path.exists():
                    receipt = _json(receipt_path)
                    if receipt["manifest_sha256"] != manifest["receipt"]["sha256"]:
                        raise RuntimeError(f"Existing asset has different source rows: {path}")
                    verify_receipt(receipt, full=True, base=output)
                    for source_path, source_receipt in receipt["source_receipts"].items():
                        verify_receipt(source_receipt)
                        receipts[source_path] = source_receipt
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
                    shape = (len(frame), 2048) if level == "feature" else (len(frame), 3, 64, 64)
                    values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32 if level == "feature" else np.uint8, shape=shape)
                    before = set(receipts)
                    if level == "feature":
                        _gather_features(frame, metadata["inputs"], values, receipts)
                        used = [metadata["inputs"][key]["feature_path"] for key in frame.feature_key.unique()]
                    else:
                        _gather_pixels(frame, source, metadata["inputs"], values, receipts,
                                       Path(metadata["source_root"]), metadata["config"].get("path_map", {}))
                        used = ([metadata["inputs"][key]["pixel_path"] for key in frame.feature_key.unique()
                                 if "pixel_path" in metadata["inputs"][key]] if source == "real" else
                                [str(_resolve(p, metadata["source_root"], metadata["config"].get("path_map", {}))) for p in frame.tensor_path.unique()])
                    values.flush()
                    del values
                    temporary.replace(path)
                    receipt = {**_output_receipt(path, output), "manifest_sha256": manifest["receipt"]["sha256"],
                               "shape": list(shape), "dtype": "float32" if level == "feature" else "uint8",
                               "source_receipts": {p: receipts[p] for p in set(used) | (set(receipts) - before)}}
                    _write_json(receipt_path, receipt)
                metadata["arrays"][level][role][source] = receipt
                print(f"Prepared {level}/{role}/{source}: {len(frame):,}", flush=True)


def prepare_data(output, source_root=DEFAULT_SOURCE_ROOT, cfg=None, smoke=False):
    """Audit/freeze all row roles and build development arrays without test access."""
    output, source_root, cfg = Path(output).resolve(), Path(source_root).resolve(), dict(cfg or {})
    if output == source_root or source_root in output.parents:
        raise ValueError("New selection outputs must be outside the historical source tree")
    metadata_path = output / "data" / "metadata.json"
    data_config = dict(split_seed=int(cfg.get("split_seed", 20260929)), smoke_rows=int(cfg.get("smoke_rows", 64)),
                       path_map=dict(cfg.get("path_map", {})))
    if cfg.get("expanded_fid_root"):
        data_config["expanded_fid_root"] = str(Path(cfg["expanded_fid_root"]).resolve())
    levels = list(cfg.get("levels", ["feature", "pixel"]))
    if not levels or len(set(levels)) != len(levels) or set(levels) - {"feature", "pixel"}:
        raise ValueError("levels must contain feature and/or pixel once")
    if data_config["smoke_rows"] < 1:
        raise ValueError("smoke_rows must be positive")
    if metadata_path.exists():
        metadata = _json(metadata_path)
        if metadata["config"] != data_config or metadata["levels"] != levels or metadata["smoke"] != bool(smoke) or metadata["source_root"] != str(source_root):
            raise RuntimeError("Prepared data configuration is immutable; use a new run directory")
        for level in levels:
            for role in TRAIN_ROLES:
                for source in SOURCES:
                    verify_receipt(metadata["arrays"][level][role][source], base=output)
        return metadata
    if data_config.get("expanded_fid_root"):
        from experiments.CelebA.expanded_selection_data import audit_expanded_sources
        original, inputs, receipts = audit_expanded_sources(source_root, data_config)
    else:
        original, inputs, receipts = _audit_sources(source_root, data_config)
    roles = _freeze_roles(original, data_config["split_seed"], data_config["smoke_rows"] if smoke else None)
    metadata = dict(schema_version=1, source_root=str(source_root), config=data_config, levels=levels, smoke=bool(smoke),
                    test_assets_loaded=False, retrospective=True,
                    historical_reuse="Old validation was previously used for fitting; old design/test were inspected. This is a retrospective study.",
                    pixel_preprocessing="Verified historical center-crop178, bilinear-antialiased resize64 uint8 caches; trainer maps to [-1,1].",
                    feature_preprocessing="Frozen pytorch-fid 0.3.0 pool3 2048; fit standardization using training rows only.",
                    role_definitions={"train": "Historical training", "earlystop": "Half historical validation, for checkpoint stopping",
                                      "selection_calibration": "Quarter historical validation, for score-cell interval construction",
                                      "selection_evaluation": "Quarter historical validation, for BS and score-cell prediction means",
                                      "test_calibration": "Half inspected historical design plus test union, for final interval construction",
                                      "test_evaluation": "Other half inspected historical design plus test union, for final BS and score-cell prediction means"},
                    inputs=inputs, source_receipts=receipts, roles={}, arrays={})
    if data_config.get('expanded_fid_root'):
        metadata['pixel_preprocessing'] = 'Real JPEGs: center-crop178, PIL bilinear-antialiased resize64, ToTensor times255 rounded uint8; generated verified uint8 shards; trainer maps to [-1,1].'
        metadata['role_definitions'] = {
            'train': 'Full historical training pools',
            'earlystop': 'Half expanded validation; checkpoint selection only',
            'selection_calibration': 'Quarter expanded validation; cell RDR estimates',
            'selection_evaluation': 'Quarter expanded validation; candidate Brier and local Gap',
            'test_calibration': 'Reserved half of expanded final pool; not loaded for model selection',
            'test_evaluation': 'Reserved other half of expanded final pool; not loaded for model selection',
        }
    for role, distributions in roles.items():
        metadata["roles"][role] = {}
        for source, frame in distributions.items():
            path = output / "data" / "manifests" / f"{role}_{source}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            metadata["roles"][role][source] = dict(path=str(path.relative_to(output)), count=len(frame), identity_count=int(frame.identity.nunique()) if source == "real" else None,
                                                  receipt=_output_receipt(path, output), original_role_counts={str(k): int(v) for k, v in frame.original_role.value_counts().items()})
    _build_assets(output, metadata, TRAIN_ROLES)
    _write_json(metadata_path, metadata)
    return metadata


def _verified_freeze(output):
    path, hash_path = output / "freeze.json", output / "freeze.sha256"
    if not path.is_file() or not hash_path.is_file():
        raise RuntimeError("Final assets require freeze.json and freeze.sha256")
    tokens = hash_path.read_text().strip().split()
    if not tokens:
        raise RuntimeError("Empty freeze.sha256")
    receipt = _receipt(path, tokens[0])
    receipt["path"] = "freeze.json"
    return receipt


def prepare_evaluation_data(output, metadata=None):
    """Build final assets after freeze; never rewrite sealed development metadata."""
    output = Path(output).resolve()
    freeze = _verified_freeze(output)
    path = output / "data" / "metadata.json"
    sealed = _json(path)
    if metadata is not None and metadata != sealed:
        raise RuntimeError("Supplied metadata differs from sealed development metadata")
    evaluation_path = output / "data" / "evaluation_metadata.json"
    if evaluation_path.exists():
        existing = _json(evaluation_path)
        if existing["freeze_receipt"]["sha256"] != freeze["sha256"] or existing["development_metadata"]["sha256"] != sha256_file(path):
            raise RuntimeError("Evaluation assets belong to a different frozen selection")
        return existing
    result = json.loads(json.dumps(sealed))
    result.update(freeze_receipt=freeze, development_metadata=_output_receipt(path, output), test_assets_loaded=True)
    _build_assets(output, result, TEST_ROLES)
    _write_json(evaluation_path, result)
    return result


def load_arrays(output, level, branch, roles, metadata=None):
    """Open immutable prepared arrays, never implicitly build or standardize data."""
    output = Path(output).resolve()
    if branch not in ("lower", "upper") or level not in ("feature", "pixel"):
        raise ValueError("Expected lower/upper branch and feature/pixel level")
    roles = tuple(roles)
    if not roles or set(roles) - set(TRAIN_ROLES + TEST_ROLES):
        raise ValueError("Unknown or empty data roles")
    final = bool(set(roles) & set(TEST_ROLES))
    if final:
        freeze = _verified_freeze(output)
    if metadata is None:
        metadata = _json(output / "data" / ("evaluation_metadata.json" if final else "metadata.json"))
    if final and metadata.get("freeze_receipt", {}).get("sha256") != freeze["sha256"]:
        raise RuntimeError("Final arrays require matching frozen evaluation metadata")
    result = {}
    for role in roles:
        result[role] = {}
        for label, source in (("p", "real"), ("q", branch)):
            receipt = metadata["arrays"][level][role][source]
            path = verify_receipt(receipt, base=output)
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(values.shape) != receipt["shape"] or str(values.dtype) != receipt["dtype"]:
                raise RuntimeError(f"Prepared array schema changed: {path}")
            if len(values) != metadata["roles"][role][source]["count"]:
                raise RuntimeError("Prepared arrays no longer align with frozen source rows")
            result[role][label] = values
    return result
