"""Rebuild and verify fixed primary splits and the 25 auxiliary null folds.

Historical design/test labels are source provenance. The primary evaluation role
is their 18,000-row union; training and validation IDs remain unchanged.
"""

import numpy as np
import pandas as pd
from CELEBA_agent3 import atomic_csv, atomic_json
from CELEBA_data import sha256_file, manifest_fingerprint
from CELEBA_split_data import fixed_real_fold_manifests
from .CELEBA_data import read_json, read_selected, verify


def prepare(args):
    """Freeze the exact supplied split inputs, including the original null folds."""
    source, work = args.source_root, args.work_root
    primary = read_json(source / "locks/real_generator_rdr_selection.json")
    final = read_json(source / "locks/agent3_final_test_selection.json")
    spec = {"primary": {}, "full": {}, "null": {}, "input_sha256": {}}
    prepare_primary_splits(source, work, primary, final, spec)
    prepare_generated_null_splits(source, work, spec)
    prepare_real_null_splits(args.reference_root, work, spec)
    # Store source input hashes, but do not require old fitted models or predictions.
    spec["manifest_sha256"] = {
        str(p): sha256_file(p) for p in (work / "manifests").rglob("*.csv")
    }
    atomic_json(spec, work / "inputs.json")
    print("Prepared fixed primary and 25 null experiment splits", flush=True)


def prepare_primary_splits(source, work, primary, final, spec):
    """Verify train/validation/test rows and cross-role identity or seed separation."""
    for name in ("real", "lower", "upper"):
        spec["primary"][name] = {}
        for role in ("train", "validation", "test"):
            origins = ("design", "test") if role == "test" else (role,)
            parts = []
            for origin in origins:
                item = (
                    final["selection"][name]
                    if origin == "test"
                    else primary["selection"][origin][name]
                )
                frame = read_selected(item)
                spec["input_sha256"][item["selected_manifest_path"]] = sha256_file(
                    item["selected_manifest_path"]
                )
                full_key = f"{name}_{origin}"
                full_path = work / "manifests/full" / f"{full_key}.csv"
                full = pd.read_csv(item["input_manifest_path"], low_memory=False)
                if manifest_fingerprint(full) != item["input_manifest_fingerprint"]:
                    raise RuntimeError(
                        "Full input manifest differs from the frozen split"
                    )
                seed = item.get("row_seed")
                if seed is None and name == "real" and origin == "validation":
                    seed = 2026082503
                if seed is not None:
                    positions = np.sort(
                        np.random.default_rng(int(seed)).choice(
                            len(full), len(frame), replace=False
                        )
                    )
                    if not np.array_equal(
                        full.iloc[positions].source_id, frame.source_id
                    ):
                        raise RuntimeError(
                            f"Row selection does not reproduce frozen IDs: {name}/{origin}"
                        )
                atomic_csv(full, full_path)
                spec["full"][full_key] = dict(
                    manifest=str(full_path),
                    original_features=item["feature_cache_path"],
                    original_features_sha256=item["feature_cache_sha256"],
                )
                if "feature_row" not in frame:
                    row_map = dict(zip(full.source_id, range(len(full))))
                    frame["feature_row"] = frame.source_id.map(row_map)
                frame["feature_key"] = full_key
                frame["original_role"], frame["role"] = origin, role
                parts.append(frame)
            combined = pd.concat(parts, ignore_index=True)
            expected = {"train": 60000, "validation": 20000, "test": 18000}[role]
            if len(combined) != expected or combined.source_id.duplicated().any():
                raise RuntimeError(f"Invalid {name}/{role} split")
            path = work / "manifests/primary" / f"{role}_{name}.csv"
            atomic_csv(combined, path)
            spec["primary"][name][role] = str(path)
        groups = [
            pd.read_csv(spec["primary"][name][role])
            for role in ("train", "validation", "test")
        ]
        key = "identity" if name == "real" else "latent_seed"
        for i in range(3):
            for j in range(i):
                if set(groups[i][key]) & set(groups[j][key]):
                    raise RuntimeError(f"Overlapping {key}: {name}")


def prepare_generated_null_splits(source, work, spec):
    """Reproduce five pairs of same-source folds for each generated branch."""
    for branch in ("lower", "upper"):
        spec["null"][branch] = {}
        for repeat in range(5):
            spec["null"][branch][str(repeat)] = {}
            for role, origin in (
                ("train", "train"),
                ("validation", "validation"),
                ("test", "design"),
            ):
                spec["null"][branch][str(repeat)][role] = {}
                full_key = f"{branch}_{origin}"
                full = pd.read_csv(spec["full"][full_key]["manifest"])
                row_map = dict(zip(full.source_id, range(len(full))))
                for fold in ("A", "B"):
                    p = (
                        source
                        / f"manifests/pair_feature_null/{branch}/repeat_{repeat:02d}/{origin}_fold_{fold}.csv"
                    )
                    frame = pd.read_csv(p)
                    fold_seed = (
                        2026083100
                        + (100 if branch == "upper" else 0)
                        + 10 * repeat
                        + {"train": 0, "validation": 1, "test": 2}[role]
                    )
                    n = {"train": 60000, "validation": 10000, "test": 5000}[role]
                    draw = np.random.default_rng(fold_seed).choice(
                        len(full), 2 * n, replace=False
                    )
                    positions = np.sort(draw[:n] if fold == "A" else draw[n:])
                    if not np.array_equal(
                        full.iloc[positions].source_id, frame.source_id
                    ):
                        raise RuntimeError(
                            "Generated null fold selection does not reproduce frozen IDs"
                        )
                    spec["input_sha256"][str(p)] = sha256_file(p)
                    frame["feature_key"], frame["feature_row"] = (
                        full_key,
                        frame.source_id.map(row_map),
                    )
                    frame["original_role"], frame["role"] = origin, role
                    path = work / f"manifests/null/{branch}/{repeat}/{role}_{fold}.csv"
                    atomic_csv(frame, path)
                    spec["null"][branch][str(repeat)][role][fold] = str(path)


def prepare_real_null_splits(reference, work, spec):
    """Reproduce the real-image null folds from the recorded identity seeds."""
    real_lock = read_json(reference / "rdr/pixel/real_fold_cache/cache_lock.json")
    spec["null"]["real"] = {}
    for role in ("train", "validation", "test"):
        spec["null"]["real"][role] = {}
        n = {"train": 60000, "validation": 19000, "test": 9000}[role]
        rebuilt = fixed_real_fold_manifests(
            reference,
            role,
            n,
            20960824 + {"train": 0, "validation": 1, "test": 2}[role],
        )
        for fold in ("A", "B"):
            item = real_lock["roles"][role]["folds"][fold]
            p = verify(item["manifest_path"], item["manifest_sha256"])
            frame = pd.read_csv(p)
            if not np.array_equal(
                rebuilt[0 if fold == "A" else 1].source_id, frame.source_id
            ):
                raise RuntimeError(
                    "Real null fold selection does not reproduce frozen IDs"
                )
            spec["input_sha256"][str(p)] = sha256_file(p)
            path = work / f"manifests/null/real/{role}_{fold}.csv"
            atomic_csv(frame, path)
            spec["null"]["real"][role][fold] = str(path)
