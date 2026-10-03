import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.CelebA import null_data
from experiments.CelebA.selection_data import _output_receipt, _write_json


def make_parent(root, transform=None):
    """Small real-schema frozen parent, including independently sealed finals."""
    frames = {}
    for role_index, role in enumerate(null_data.ROLES):
        frames[role] = {}
        for source_index, source in enumerate(null_data.SOURCES):
            if source == "real":
                identities = np.repeat(np.arange(4) + 100 * role_index, [1, 4, 2, 3])
                count = len(identities)
            else:
                count, identities = 11, np.full(11, np.nan)
            frame = pd.DataFrame(dict(
                source=[source] * count, role=[role] * count,
                source_id=[f"{source}:{role}:{i}" for i in range(count)],
                identity=identities,
            ))
            if source != "real":
                frame["latent_seed"] = np.arange(count) + 10000 * source_index + 100 * role_index
            frames[role][source] = frame
    if transform:
        transform(frames)
    roles, arrays = {}, {level: {} for level in null_data.LEVEL_SOURCES}
    for role in null_data.ROLES:
        roles[role] = {}
        for source in null_data.SOURCES:
            frame = frames[role][source]
            path = root / "data/manifests" / f"{role}_{source}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False)
            roles[role][source] = dict(
                path=str(path.relative_to(root)), count=len(frame),
                identity_count=int(frame.identity.nunique()) if source == "real" else None,
                receipt=_output_receipt(path, root),
            )
        for level, sources in null_data.LEVEL_SOURCES.items():
            arrays[level][role] = {}
            for source in sources:
                count = len(frames[role][source])
                values = (np.arange(count * 2048, dtype=np.float32).reshape(count, 2048)
                          if level == "feature" else
                          np.arange(count * 3 * 64 * 64, dtype=np.uint8).reshape(count, 3, 64, 64))
                path = root / "data/arrays" / level / role / f"{source}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, values)
                receipt = _output_receipt(path, root)
                # Exercise absolute development-array references in the real parent.
                if role in null_data.TRAIN_ROLES:
                    receipt["path"] = str(path)
                arrays[level][role][source] = dict(
                    **receipt, shape=list(values.shape), dtype=str(values.dtype),
                    manifest_sha256=roles[role][source]["receipt"]["sha256"],
                )
    development = dict(roles=roles, test_assets_loaded=False, arrays={
        level: {role: arrays[level][role] for role in null_data.TRAIN_ROLES}
        for level in arrays
    })
    _write_json(root / "data/metadata.json", development)
    _write_json(root / "freeze.json", {"settings": "selected JS models"})
    evaluation = dict(
        **{key: copy.deepcopy(value) for key, value in development.items() if key != "arrays"},
        arrays=arrays,
        development_metadata=_output_receipt(root / "data/metadata.json", root),
        freeze_receipt=_output_receipt(root / "freeze.json", root),
    )
    evaluation["test_assets_loaded"] = True
    _write_json(root / "data/evaluation_metadata.json", evaluation)
    return root


def test_fixed_halves_exhaust_rows_and_keep_all_identities_disjoint(tmp_path):
    parent = make_parent(tmp_path / "parent")
    output = tmp_path / "null"
    metadata = null_data.prepare_null_data(output, parent, 2026100303)
    assert metadata["original_arrays_verified"] == 30
    assert metadata["fixed_across_repeats"] and metadata["shared_across_representations"]
    assert not (output / "data/arrays").exists()
    seen_identities = set()
    seen_seeds = set()
    seen_ids = set()
    for role in null_data.ROLES:
        for source in null_data.SOURCES:
            info = metadata["roles"][role][source]
            merged = []
            for side in ("p", "q"):
                details = info["sides"][side]
                indices = np.load(output / details["index_receipt"]["path"])
                frame = pd.read_csv(output / details["manifest_receipt"]["path"])
                assert np.array_equal(indices, frame.parent_row)
                assert len(frame) == details["count"]
                assert frame.null_side.eq(side).all()
                assert not seen_ids.intersection(frame.source_id)
                seen_ids.update(frame.source_id)
                if source == "real":
                    identities = set(frame.identity)
                    assert not seen_identities.intersection(identities)
                    seen_identities.update(identities)
                    assert len(identities) == details["identity_count"]
                else:
                    assert not seen_seeds.intersection(frame.latent_seed)
                    seen_seeds.update(frame.latent_seed)
                merged.extend(indices)
            assert sorted(merged) == list(range(info["parent_count"]))
            if source != "real":
                assert [info["sides"][s]["count"] for s in ("p", "q")] == [5, 6]


def test_partition_is_deterministic_shared_and_preparation_is_idempotent(tmp_path):
    parent = make_parent(tmp_path / "parent")
    first, second = tmp_path / "first", tmp_path / "second"
    metadata = null_data.prepare_null_data(first, parent, 23)
    other = null_data.prepare_null_data(second, parent, 23)
    assert null_data.prepare_null_data(first, parent, 23) == metadata
    for role in null_data.ROLES:
        for source in null_data.SOURCES:
            for side in ("p", "q"):
                a = metadata["roles"][role][source]["sides"][side]
                b = other["roles"][role][source]["sides"][side]
                assert a["index_receipt"]["sha256"] == b["index_receipt"]["sha256"]
                assert a["manifest_receipt"]["sha256"] == b["manifest_receipt"]["sha256"]
    feature = null_data.load_null_arrays(first, "feature", "lower", ["train"])
    pixel = null_data.load_null_arrays(first, "pixel", "lower", ["train"])
    for side in ("p", "q"):
        np.testing.assert_array_equal(feature[f"train_{side}"]._indices, pixel[f"train_{side}"]._indices)
    with pytest.raises(RuntimeError, match="immutable"):
        null_data.prepare_null_data(first, parent, 24)


def test_indexed_rows_batches_match_source_and_are_read_only(tmp_path):
    path = tmp_path / "original.npy"
    values = np.arange(60, dtype=np.float32).reshape(10, 6)
    np.save(path, values)
    source = np.load(path, mmap_mode="r")
    index = np.array([1, 3, 7, 8])
    rows = null_data.IndexedRows(source, index)
    assert rows.shape == (4, 6) and rows.ndim == 2 and rows.dtype == np.float32
    assert isinstance(rows._values, np.memmap)
    assert np.shares_memory(rows._values, source)
    assert len(rows) == 4
    for key in (0, -1, slice(1, 3), np.array([2, 0]), np.array([False, True, False, True])):
        actual = rows[key]
        np.testing.assert_array_equal(actual, values[index][key])
        assert not actual.flags.writeable
    np.testing.assert_array_equal(rows[1:3, :2], values[index][1:3, :2])
    with pytest.raises(TypeError):
        rows[0] = values[0]


@pytest.mark.parametrize("indices", [[0, 0], [-1, 1], [0, 10], [], [0., 1.], [[0, 1]]])
def test_invalid_indices_rejected(indices):
    with pytest.raises(ValueError):
        null_data.IndexedRows(np.zeros((10, 2)), np.asarray(indices))


def test_training_loading_does_not_open_final_arrays(tmp_path, monkeypatch):
    parent = make_parent(tmp_path / "parent")
    output = tmp_path / "null"
    null_data.prepare_null_data(output, parent, 23)
    original_load = np.load
    opened = []

    def checked_load(path, *args, **kwargs):
        opened.append(str(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", checked_load)
    arrays = null_data.load_null_arrays(output, "feature", "lower", null_data.TRAIN_ROLES)
    assert len(arrays) == 8
    assert not any("test_calibration" in path or "test_evaluation" in path for path in opened)
    opened.clear()
    final = null_data.load_null_arrays(output, "feature", "lower", null_data.TEST_ROLES)
    assert len(final) == 4
    assert not any("/train/" in path for path in opened)


@pytest.mark.parametrize("already_prepared", [False, True])
def test_original_array_hash_mismatch_rejected_even_with_preserved_stat(tmp_path, already_prepared):
    parent = make_parent(tmp_path / "parent")
    output = tmp_path / "null"
    if already_prepared:
        null_data.prepare_null_data(output, parent, 23)
    path = parent / "data/arrays/feature/train/lower.npy"
    stat = path.stat()
    array = np.load(path, mmap_mode="r+")
    array[0, 0] = 9.
    array.flush()
    del array
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        null_data.prepare_null_data(output, parent, 23)


def test_parent_manifest_hash_mismatch_rejected(tmp_path):
    parent = make_parent(tmp_path / "parent")
    path = parent / "data/manifests/train_real.csv"
    stat = path.stat()
    path.write_text(path.read_text().replace("real:train:0", "real:train:X", 1))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        null_data.prepare_null_data(tmp_path / "null", parent, 23)


@pytest.mark.parametrize("invalid", ["identity", "duplicate_seed", "cross_source_seed", "source_id"])
def test_parent_pool_overlap_rejected(tmp_path, invalid):
    def transform(frames):
        if invalid == "identity":
            frames["earlystop"]["real"].loc[0, "identity"] = frames["train"]["real"].identity.iloc[0]
        elif invalid == "duplicate_seed":
            frames["train"]["lower"].loc[1, "latent_seed"] = frames["train"]["lower"].latent_seed.iloc[0]
        elif invalid == "cross_source_seed":
            frames["train"]["upper"].loc[0, "latent_seed"] = frames["train"]["lower"].latent_seed.iloc[0]
        else:
            frames["earlystop"]["lower"].loc[0, "source_id"] = frames["train"]["lower"].source_id.iloc[0]
    parent = make_parent(tmp_path / "parent", transform)
    with pytest.raises(ValueError, match="Overlapping|Duplicate"):
        null_data.prepare_null_data(tmp_path / "null", parent, 23)


def test_streaming_scaler_consumes_indexed_training_rows(tmp_path):
    torch = pytest.importorskip("torch")
    from experiments.CelebA.selection_training import fit_feature_scaler
    values = np.arange(48, dtype=np.float32).reshape(12, 4)
    p = null_data.IndexedRows(values, np.array([0, 2, 3, 7]))
    q = null_data.IndexedRows(values, np.array([1, 4, 6, 8, 11]))
    model = type("Scaler", (), {"input_mean": torch.zeros(4), "input_scale": torch.zeros(4)})()
    receipt = fit_feature_scaler(model, p, q, batch_size=3)
    expected = values[[0, 2, 3, 7, 1, 4, 6, 8, 11]].astype(np.float64)
    np.testing.assert_allclose(model.input_mean.numpy(), expected.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(model.input_scale.numpy(), expected.std(axis=0), rtol=1e-6)
    assert receipt["n_p"] == 4 and receipt["n_q"] == 5


def test_invalid_representation_role_and_original_output_rejected(tmp_path):
    for level, source, roles in [("feature", "real", ["train"]), ("pixel", "lower", []),
                                 ("pixel", "upper", ["train", "train"]),
                                 ("pixel", "real", ["unknown"])]:
        with pytest.raises(ValueError):
            null_data.load_null_arrays(tmp_path, level, source, roles)
    with pytest.raises(ValueError, match="outside"):
        null_data.prepare_null_data(tmp_path / "parent/null", tmp_path / "parent", 23)
