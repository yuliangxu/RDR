#!/usr/bin/env python3
"""Aggregate full-pool CelebA FID from verified original and extra features.

CPU example (request 8 CPUs and 64 GB from the scheduler)::

    python3 expanded_fid_report.py --output-dir /path/to/output --threads 8

Extra generation must finish before validation/final results are published.
FID uses raw pool3 features, pooled float64 moments and sample covariance.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


SOURCE_ROOT = Path("/cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid")
ROLES = ("train", "validation", "final")
BRANCHES = ("lower", "upper")
SOURCE_ROLES = {"train": ("train",), "validation": ("validation",), "final": ("design", "test")}
EXPECTED = {"train": {"p": 122984, "q": 120000},
            "validation": {"p": 39786, "q": 40000},
            "final": {"p": 39829, "q": 40000}}
DIMENSION = 2048
EXTRA_COUNT = 20000


class PendingInputs(RuntimeError):
    """An extra generation completion receipt has not arrived yet."""


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(value)
    temporary.replace(path)


def write(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def checked_receipt(path, expected=None):
    path = Path(path).resolve()
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Source hash differs from frozen receipt: {path}")
    return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}


def pooled_moments(arrays, expected_count, dimension=DIMENSION, chunk_size=4096):
    """Stable chunk merging, including between-chunk mean variation; ddof=1."""
    if expected_count < 2 or dimension < 1 or chunk_size < 1:
        raise ValueError("Moments require at least two observations and positive dimensions/chunk size")
    count, mean, m2 = 0, np.zeros(dimension, dtype=np.float64), np.zeros((dimension, dimension), dtype=np.float64)
    for array in arrays:
        if array.ndim != 2 or array.shape[1] != dimension:
            raise ValueError(f"Expected feature matrix with {dimension} columns, got {array.shape}")
        for start in range(0, len(array), chunk_size):
            values = np.asarray(array[start:start + chunk_size], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError("Feature matrix contains nonfinite values")
            n = len(values)
            center = values.mean(axis=0)
            residual = values - center
            delta = center - mean
            updated = count + n
            m2 += residual.T @ residual + np.outer(delta, delta) * (count * n / updated)
            mean += delta * (n / updated)
            count = updated
    if count != expected_count:
        raise ValueError(f"Feature count mismatch: observed {count}, expected {expected_count}")
    covariance = m2 / (count - 1)
    return mean, (covariance + covariance.T) * .5


def _features(path, expected_hash, expected_count, provenance, expected_metadata=None):
    import torch
    receipt = checked_receipt(path, expected_hash)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError(f"Feature cache lacks a features tensor: {path}")
    values = payload["features"]
    if not isinstance(values, torch.Tensor) or tuple(values.shape) != (expected_count, DIMENSION):
        raise ValueError(f"Feature cache count/dimension mismatch: {path}")
    for key, expected in (expected_metadata or {}).items():
        if payload.get(key) != expected:
            raise ValueError(f"Feature metadata differs for {key}: {path}")
    receipt.update(count=len(values), dimension=DIMENSION,
                   metadata={k: v for k, v in payload.items() if k != "features"})
    provenance.append(receipt)
    return values.numpy()


def _inputs(output, role, source_root):
    """Load completion locks first; never compute from unfinished extra pools."""
    extra_paths = [output / "locks" / f"extra_{role}_{branch}.json" for branch in BRANCHES] if role != "train" else []
    missing = [str(path) for path in extra_paths if not path.is_file()]
    if missing:
        raise PendingInputs("Missing completed extra-generation receipts: " + ", ".join(missing))
    provenance, locks = [], []
    primary_path = source_root / "locks/real_generator_rdr_selection.json"
    primary_receipt = checked_receipt(primary_path)
    locks.append(primary_receipt)
    primary = read(primary_path)
    pair_path = source_root / "locks/generator_pair_lock.json"
    pair_receipt = checked_receipt(pair_path)
    locks.append(pair_receipt)
    pair = read(pair_path)
    final = None
    if role == "final":
        final_path = source_root / "locks/agent3_final_test_selection.json"
        locks.append(checked_receipt(final_path))
        final = read(final_path)
        if final["design_selection_lock_sha256"] != primary_receipt["sha256"]:
            raise ValueError("Historical real-data selection locks disagree")
    p_arrays, q_arrays = [], {branch: [] for branch in BRANCHES}
    for origin in SOURCE_ROLES[role]:
        real = (final["selection"] if origin == "test" else primary["selection"][origin])["real"]
        p_arrays.append(_features(real["feature_cache_path"], real["feature_cache_sha256"],
                                  real["pool_size"], provenance,
                                  {"role": origin, "manifest_fingerprint": real["input_manifest_fingerprint"],
                                   "backend": "pytorch_fid_0.3.0", "pytorch_fid_version": "0.3.0"}))
        generated_path = source_root / "locks" / f"generated_pair_{origin}.json"
        locks.append(checked_receipt(generated_path))
        generated = read(generated_path)
        if (generated["status"] != "generated_and_validated" or generated["role"] != origin
                or generated["pair_lock_sha256"] != pair_receipt["sha256"]):
            raise ValueError(f"Generated source completion/pair lineage mismatch: {generated_path}")
        for branch in BRANCHES:
            source = generated["branches"][branch]
            expected = {"role": origin, "manifest_fingerprint": source["manifest_fingerprint"],
                        "source": pair["branches"][branch]["source"],
                        "truncation_psi": pair["branches"][branch]["truncation_psi"],
                        "noise_mode": pair["noise_mode"], "force_reference_ops": pair["force_reference_ops"],
                        "checkpoint_sha256": pair["generator_assets"]["checkpoint_sha256"],
                        "repository_commit": pair["generator_assets"]["repository_commit"],
                        "pytorch_fid_version": "0.3.0", "backend": "pytorch_fid_0.3.0"}
            q_arrays[branch].append(_features(source["feature_cache_path"], source["feature_cache_sha256"],
                                             source["pool_size"], provenance, expected))
    for branch, path in zip(BRANCHES, extra_paths):
        locks.append(checked_receipt(path))
        extra = read(path)
        if (extra.get("status") != "generated_and_validated" or extra.get("role") != role
                or extra.get("branch") != branch or extra.get("count") != EXTRA_COUNT
                or extra.get("pair_lock_sha256") != pair_receipt["sha256"]):
            raise ValueError(f"Extra generation completion/pair lineage mismatch: {path}")
        feature_path = Path(extra["feature_cache_path"])
        if not feature_path.is_absolute():
            feature_path = output / feature_path
        expected_path = output / "features" / f"extra_{role}_{branch}_2048.pt"
        if feature_path.resolve() != expected_path.resolve():
            raise ValueError(f"Unexpected extra feature path: {feature_path}")
        q_arrays[branch].append(_features(feature_path, extra["feature_cache_sha256"], EXTRA_COUNT, provenance))
        locks[-1]["generation_record"] = extra
    for source, arrays, count in [("P", p_arrays, EXPECTED[role]["p"])] + [
            (branch, q_arrays[branch], EXPECTED[role]["q"]) for branch in BRANCHES]:
        observed = sum(len(array) for array in arrays)
        if observed != count:
            raise ValueError(f"{role}/{source} count mismatch: observed {observed}, expected {count}")
    return p_arrays, q_arrays, {"features": provenance, "locks": locks}


def compute_role(output, role, source_root=SOURCE_ROOT):
    if role not in ROLES:
        raise ValueError(f"Unknown expanded FID role: {role}")
    if importlib.metadata.version("pytorch-fid") != "0.3.0":
        raise ValueError("Expanded FID requires pytorch-fid 0.3.0")
    from pytorch_fid.fid_score import calculate_frechet_distance
    p_arrays, q_arrays, provenance = _inputs(Path(output), role, Path(source_root))
    expected = EXPECTED[role]
    p_mean, p_cov = pooled_moments(p_arrays, expected["p"])
    results = {}
    for branch in BRANCHES:
        q_mean, q_cov = pooled_moments(q_arrays[branch], expected["q"])
        fid = float(calculate_frechet_distance(p_mean, p_cov, q_mean, q_cov))
        if not math.isfinite(fid) or fid < -1e-6:
            raise ValueError(f"Nonfinite or negative FID for {role}/{branch}: {fid}")
        mean_term = float(np.sum((p_mean - q_mean) ** 2))
        results[branch] = {"fid": fid, "n_p": expected["p"], "n_q": expected["q"],
                           "mean_term": mean_term, "covariance_term": fid - mean_term}
    return {"status": "complete", "role": role, "results": results,
            "fid_upper_minus_lower": results["upper"]["fid"] - results["lower"]["fid"],
            "absolute_fid_difference": abs(results["upper"]["fid"] - results["lower"]["fid"]),
            "source_roles": SOURCE_ROLES[role], "extra_per_branch": 0 if role == "train" else EXTRA_COUNT,
            "method": "pytorch_fid.fid_score.calculate_frechet_distance; raw pool3; float64 pooled mean/covariance; ddof=1",
            "pytorch_fid_version": "0.3.0", "dimension": DIMENSION,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "provenance": provenance}


def combined_report(output):
    output = Path(output)
    records, rows = {}, []
    for role in ROLES:
        path = output / "fid" / f"{role}.json"
        record = read(path) if path.exists() else {"status": "pending", "role": role}
        records[role] = record
        if record["status"] == "complete":
            for branch in BRANCHES:
                rows.append({"role": role, "branch": branch, **record["results"][branch],
                             "fid_upper_minus_lower": record["fid_upper_minus_lower"],
                             "absolute_fid_difference": record["absolute_fid_difference"]})
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("role", "branch", "n_p", "n_q", "fid", "mean_term", "covariance_term",
                                               "fid_upper_minus_lower", "absolute_fid_difference"))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output / "FID_RESULTS.csv", buffer.getvalue())
    complete = all(record["status"] == "complete" for record in records.values())
    lines = ["# CelebA expanded-pool FID", "", f"**Status: {'complete' if complete else 'incomplete; pending roles listed below'}.**", "",
             "Raw pytorch-fid 0.3.0 pool3 features; float64 pooled moments and sample covariance (ddof=1). "
             "The same P pool is used for both generator comparisons within each role. "
             "Final combines the historical design and test pools. Validation/final Q add 20,000 new images per branch.", "",
             "| Role | Status | P count | Q count per branch | FID lower | FID upper | Upper minus lower | Absolute difference |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for role, record in records.items():
        expected = EXPECTED[role]
        if record["status"] == "complete":
            result = record["results"]
            values = [f"{result[branch]['fid']:.6f}" for branch in BRANCHES]
            values += [f"{record['fid_upper_minus_lower']:.6f}", f"{record['absolute_fid_difference']:.6f}"]
        else:
            values = ["Pending"] * 4
        lines.append(f"| {role} | {record['status']} | {expected['p']} | {expected['q']} | " + " | ".join(values) + " |")
    lines.extend(["", "Counts in pending rows are planned targets. These empirical FID differences are descriptive; "
                  "they do not establish formal FID equivalence. Historical data and generator choices were inspected "
                  "previously, so this is a retrospective comparison. FID is separate from RDR Brier/local-Gap model selection.", "",
                  "[Numeric results](FID_RESULTS.csv) · [Train provenance](fid/train.json) · "
                  "[Validation provenance](fid/validation.json) · [Final provenance](fid/final.json)", ""])
    atomic_text(output / "FID_RESULTS.md", "\n".join(lines))
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--roles", choices=ROLES, nargs="+", default=list(ROLES))
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(args.threads)
    output = args.output_dir.resolve()
    errors = []
    with threadpool_limits(limits=args.threads):
        for role in dict.fromkeys(args.roles):
            try:
                record = compute_role(output, role, args.source_root)
            except PendingInputs as error:
                record = {"status": "pending", "role": role, "reason": str(error)}
                errors.append(role)
            except Exception as error:
                record = {"status": "failed", "role": role, "error_type": type(error).__name__, "reason": str(error)}
                errors.append(role)
            write(output / "fid" / f"{role}.json", record)
            print(json.dumps({k: v for k, v in record.items() if k != "provenance"}), flush=True)
    combined_report(output)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
