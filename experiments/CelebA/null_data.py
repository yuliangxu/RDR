"""Disjoint same-source halves of the frozen expanded CelebA input roles.

This adapter writes only row indices and auditable manifests. Existing arrays
remain read-only and are indexed lazily, so neither pixels nor pool3 features
are duplicated. A single partition is shared by every training repetition and
by the feature/pixel versions of a generated source.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.CelebA.selection_data import (
    SOURCES, TEST_ROLES, TRAIN_ROLES, _assert_disjoint, _identity_parts, _json,
    _output_receipt, _receipt, _row_parts, _write_json, verify_receipt,
)


ROLES = TRAIN_ROLES + TEST_ROLES
LEVEL_SOURCES = {"feature": ("lower", "upper"), "pixel": SOURCES}


class IndexedRows:
    """Read-only first-axis indexing with bounded batch materialization.

    The training scaler and prediction loop consume slices; minibatches use
    integer arrays. Both access patterns retrieve only the requested rows from
    the original memory map. There is deliberately no eager array conversion.
    """

    def __init__(self, values, indices):
        if not isinstance(values, np.ndarray) or values.ndim < 1:
            raise ValueError("IndexedRows requires a NumPy array with a row axis")
        indices = np.asarray(indices)
        if indices.ndim != 1 or indices.dtype.kind not in "iu" or not len(indices):
            raise ValueError("Row indices must be a nonempty integer vector")
        if indices.min() < 0 or indices.max() >= len(values):
            raise ValueError("Row indices are outside the source array")
        if len(np.unique(indices)) != len(indices):
            raise ValueError("Duplicate row indices")
        self._values = values.view()
        self._values.flags.writeable = False
        self._indices = indices.view()
        self._indices.flags.writeable = False
        self.shape = (len(indices),) + values.shape[1:]
        self.dtype = values.dtype
        self.ndim = values.ndim

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            first, remaining = (key[0], key[1:]) if key else (slice(None), ())
        else:
            first, remaining = key, ()
        if first is Ellipsis:
            first = slice(None)
        selected = self._values[(self._indices[first],) + remaining]
        if isinstance(selected, np.ndarray):
            selected.flags.writeable = False
        return selected


def _absolute_receipt(receipt, base, full=True):
    path = verify_receipt(receipt, full=full, base=base).resolve()
    return {**receipt, "path": str(path)}


def _array(receipt, count, level, base=None, full=False):
    path = verify_receipt(receipt, full=full, base=base)
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    expected_tail = (2048,) if level == "feature" else (3, 64, 64)
    expected_dtype = "float32" if level == "feature" else "uint8"
    if (list(values.shape) != receipt["shape"] or str(values.dtype) != receipt["dtype"]
            or values.shape != (count,) + expected_tail or str(values.dtype) != expected_dtype):
        raise RuntimeError(f"Prepared {level} array schema or row count changed: {path}")
    return values


def _audit_parent(parent):
    development_path = parent / "data/metadata.json"
    evaluation_path = parent / "data/evaluation_metadata.json"
    development, evaluation = _json(development_path), _json(evaluation_path)
    if development.get("test_assets_loaded") or not evaluation.get("test_assets_loaded"):
        raise RuntimeError("Expected sealed development inputs and materialized final inputs")
    development_receipt = _absolute_receipt(evaluation["development_metadata"], parent)
    if Path(development_receipt["path"]) != development_path:
        raise RuntimeError("Final inputs reference a different development metadata file")
    freeze_receipt = _absolute_receipt(evaluation["freeze_receipt"], parent)
    if Path(freeze_receipt["path"]) != parent / "freeze.json":
        raise RuntimeError("Final inputs reference a different selection freeze")
    parent_receipts = {
        "development_metadata": development_receipt,
        "evaluation_metadata": _receipt(evaluation_path),
        "freeze": freeze_receipt,
    }
    roles, manifests = {}, {}
    all_ids, generated_seeds = set(), set()
    for role in ROLES:
        roles[role], manifests[role] = {}, {}
        for source in SOURCES:
            info = development["roles"][role][source]
            if info != evaluation["roles"][role][source]:
                raise RuntimeError(f"Development and final manifests disagree: {role}/{source}")
            receipt = _absolute_receipt(info["receipt"], parent)
            if Path(receipt["path"]) != (parent / info["path"]).resolve():
                raise RuntimeError("Manifest path and receipt disagree")
            frame = pd.read_csv(receipt["path"])
            if len(frame) != info["count"] or len(frame) < 2:
                raise RuntimeError(f"Parent manifest row count changed: {role}/{source}")
            if not frame.role.eq(role).all():
                raise RuntimeError(f"Parent manifest has incorrect roles: {role}/{source}")
            if all_ids.intersection(frame.source_id):
                raise ValueError(f"Overlapping source IDs between parent pools: {role}/{source}")
            all_ids.update(frame.source_id)
            if source == "real":
                if int(frame.identity.nunique()) != info["identity_count"]:
                    raise RuntimeError(f"Parent identity count changed: {role}/{source}")
            else:
                if generated_seeds.intersection(frame.latent_seed):
                    raise ValueError(f"Overlapping generated seeds between parent pools: {role}/{source}")
                generated_seeds.update(frame.latent_seed)
            roles[role][source], manifests[role][source] = frame, receipt
    _assert_disjoint(roles)
    arrays = {level: {} for level in LEVEL_SOURCES}
    for level, sources in LEVEL_SOURCES.items():
        for role in ROLES:
            origin = development if role in TRAIN_ROLES else evaluation
            arrays[level][role] = {}
            for source in sources:
                receipt = _absolute_receipt(origin["arrays"][level][role][source], parent)
                if receipt["manifest_sha256"] != manifests[role][source]["sha256"]:
                    raise RuntimeError(f"Array refers to different manifest rows: {level}/{role}/{source}")
                # Hash the complete source array before recording a reusable reference.
                values = _array(receipt, len(roles[role][source]), level)
                del values
                arrays[level][role][source] = receipt
    return roles, manifests, arrays, parent_receipts


def _indices(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def _partition(frame, source, seed):
    indexed = frame.assign(parent_row=np.arange(len(frame), dtype=np.int64))
    if source == "real":
        return _identity_parts(indexed, [.5, .5], seed)
    return _row_parts(indexed, [len(frame) // 2, len(frame) - len(frame) // 2], seed)


def _load_indices(info, output):
    indices = {}
    for side in ("p", "q"):
        details = info["sides"][side]
        path = verify_receipt(details["index_receipt"], full=True, base=output)
        verify_receipt(details["manifest_receipt"], base=output)
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 1 or values.dtype != np.int64 or len(values) != details["count"]:
            raise RuntimeError("Null indices no longer match the frozen half")
        indices[side] = values
    joined = np.concatenate(list(indices.values()))
    if not np.array_equal(np.sort(joined), np.arange(info["parent_count"])):
        raise RuntimeError("Null halves must use every parent row exactly once")
    return indices


def _verify_prepared(output, metadata):
    for receipt in metadata["parent_receipts"].values():
        verify_receipt(receipt, full=True)
    for role in ROLES:
        for source in SOURCES:
            info = metadata["roles"][role][source]
            verify_receipt(info["parent_manifest_receipt"], full=True)
            for side in ("p", "q"):
                verify_receipt(info["sides"][side]["manifest_receipt"], full=True, base=output)
            _load_indices(info, output)
    for level, sources in LEVEL_SOURCES.items():
        for role in ROLES:
            for source in sources:
                receipt = metadata["arrays"][level][role][source]
                count = metadata["roles"][role][source]["parent_count"]
                _array(receipt, count, level, full=True)


def prepare_null_data(output: Path, parent: Path, seed: int) -> dict:
    """Audit all original arrays and freeze fixed A/B halves for six roles.

    Full array hashes are checked during preparation, including the authorized
    final roles. This performs no fitting and writes exclusively under output.
    Subsequent training opens development arrays only; evaluation must request
    the final roles explicitly through :func:`load_null_arrays`.
    """
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output == parent or parent in output.parents:
        raise ValueError("Null outputs must be outside the immutable parent run")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Null split seed must be a nonnegative integer")
    seed = int(seed)
    target = output / "data/null_data.json"
    if target.exists():
        metadata = _json(target)
        if metadata["parent"] != str(parent) or metadata["split_seed"] != seed:
            raise RuntimeError("Null data configuration is immutable; use a new output directory")
        _verify_prepared(output, metadata)
        return metadata
    original, manifests, arrays, parent_receipts = _audit_parent(parent)
    metadata = dict(
        schema_version=1, parent=str(parent), split_seed=seed,
        fixed_across_repeats=True, shared_across_representations=True,
        side_definitions={"p": "A", "q": "B"},
        partition_rule="Whole-identity nearest-image-target halves for real; deterministic shuffled row halves for generated sources. Every row is used exactly once within its parent role.",
        role_seed_rule="split_seed + 100 * role_index + source_index; roles and sources follow role_order and source_order",
        role_order=list(ROLES), source_order=list(SOURCES),
        level_sources={level: list(sources) for level, sources in LEVEL_SOURCES.items()},
        role_definitions={
            "train": "Fit null models; only training rows determine feature standardization",
            "earlystop": "Select a checkpoint by balanced Brier; no final access",
            "selection_calibration": "Development cell diagnostics; no model or setting selection",
            "selection_evaluation": "Development score diagnostics; no model or setting selection",
            "test_calibration": "Final score-cell RDR estimates and C.1/C.2 intervals",
            "test_evaluation": "Final Brier, null-RDR error and cell prediction means",
        },
        retrospective=True,
        historical_reuse="Subdivisions of the already inspected expanded study roles; these are retrospective same-source controls, not fresh confirmatory data.",
        parent_receipts=parent_receipts, arrays=arrays, roles={},
        original_arrays_verified=sum(len(v) for r in arrays.values() for v in r.values()),
    )
    halves = {}
    for role_index, role in enumerate(ROLES):
        metadata["roles"][role] = {}
        for side in ("p", "q"):
            halves[f"{role}_{side}"] = {}
        for source_index, source in enumerate(SOURCES):
            split_seed = seed + 100 * role_index + source_index
            parent_frame = original[role][source]
            info = dict(parent_count=len(parent_frame), parent_manifest_receipt=manifests[role][source],
                        split_seed=split_seed, sides={})
            for side, frame in zip(("p", "q"), _partition(parent_frame, source, split_seed)):
                frame = frame.assign(null_side=side)
                halves[f"{role}_{side}"][source] = frame
                index_path = output / "data/indices" / source / role / f"{side}.npy"
                manifest_path = output / "data/manifests" / source / role / f"{side}.csv"
                _indices(index_path, frame.parent_row.to_numpy(dtype=np.int64))
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(manifest_path, index=False)
                info["sides"][side] = dict(
                    count=len(frame), identity_count=int(frame.identity.nunique()) if source == "real" else None,
                    index_receipt=_output_receipt(index_path, output),
                    manifest_receipt=_output_receipt(manifest_path, output),
                )
            _load_indices(info, output)
            metadata["roles"][role][source] = info
    _assert_disjoint(halves)
    _write_json(target, metadata)
    return metadata


def load_null_arrays(output, level, source, roles):
    """Open only requested roles as flattened ``role_p``/``role_q`` arrays."""
    output = Path(output).resolve()
    if level not in LEVEL_SOURCES or source not in LEVEL_SOURCES[level]:
        raise ValueError("Expected a generated feature null or real/generated pixel null")
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles) or set(roles) - set(ROLES):
        raise ValueError("Unknown, repeated or empty null data roles")
    metadata = _json(output / "data/null_data.json")
    result = {}
    for role in roles:
        info = metadata["roles"][role][source]
        verify_receipt(info["parent_manifest_receipt"])
        values = _array(metadata["arrays"][level][role][source], info["parent_count"], level)
        indices = _load_indices(info, output)
        for side in ("p", "q"):
            result[f"{role}_{side}"] = IndexedRows(values, indices[side])
    return result
