"""Identity/seed independence, source alignment, and the final-data access gate."""

import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.CelebA import selection_data as d


def originals():
    result = {}
    for origin_index, (origin, n) in enumerate((("train", 12), ("validation", 16), ("design", 8), ("test", 8))):
        result[origin] = {}
        for source_index, source in enumerate(d.SOURCES):
            row = np.arange(n)
            result[origin][source] = pd.DataFrame(dict(
                source="real" if source == "real" else f"stylegan2_{source}",
                source_id=[f"{source}-{origin}-{i}" for i in row], role=origin,
                identity=origin_index * 100 + row // 2 if source == "real" else np.nan,
                latent_seed=source_index * 10000 + origin_index * 100 + row,
                feature_key=f"{source}_{origin}", feature_row=row,
                original_role=origin, original_selected_row=row,
                idx_in_shard=row, truncation_psi=.65 if source == "lower" else 1.569,
                tensor_path="unused", real_path="unused", shard=0,
            ))
    return result


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    source = tmp_path / "history"
    source.mkdir()
    frames, inputs = originals(), {}
    for origin, distributions in frames.items():
        for source_name, frame in distributions.items():
            key = f"{source_name}_{origin}"
            feature_path, pixel_path = source / f"{key}_features.pt", source / f"{key}_pixels.pt"
            n = len(frame)
            torch.save(dict(features=torch.from_numpy(np.repeat(np.arange(n, dtype=np.float32)[:, None], 2048, axis=1)),
                            manifest_fingerprint=key), feature_path)
            images = torch.from_numpy(np.repeat(np.arange(n, dtype=np.uint8)[:, None], 3 * 64 * 64, axis=1).reshape(n, 3, 64, 64))
            torch.save(dict(images=images, latent_seeds=torch.from_numpy(frame.latent_seed.to_numpy()),
                            manifest_fingerprint=key, noise_mode="const", truncation_psi=float(frame.truncation_psi.iloc[0]),
                            checkpoint_sha256="fixture_checkpoint", repository_commit="fixture_commit", force_reference_ops=True), pixel_path)
            frame["tensor_path"] = str(pixel_path)
            inputs[key] = dict(feature_path=str(feature_path), feature_sha256=d.sha256_file(feature_path),
                               input_fingerprint=key, selected_fingerprint=key, input_count=n, selected_count=n,
                               pixel_path=str(pixel_path), pixel_sha256=d.sha256_file(pixel_path),
                               expected_generator=dict(checkpoint_sha256="fixture_checkpoint", repository_commit="fixture_commit",
                                                       force_reference_ops=True, noise_mode="const", truncation_psi=float(frame.truncation_psi.iloc[0])))
    monkeypatch.setattr(d, "_audit_sources", lambda *args: (frames, inputs, {}))
    opened = []
    actual_load = d._torch_load

    def guarded_load(path):
        opened.append(str(path))
        return actual_load(path)

    monkeypatch.setattr(d, "_torch_load", guarded_load)
    output = tmp_path / "run"
    metadata = d.prepare_data(output, source, {"split_seed": 7})
    return output, source, metadata, opened


def test_roles_use_every_row_and_keep_real_identities_together():
    original = originals()
    roles = d._freeze_roles(original, 7)
    assert list(roles) == list(d.TRAIN_ROLES + d.TEST_ROLES)
    d._assert_disjoint(roles)
    for source in d.SOURCES:
        assert sum(len(x[source]) for x in roles.values()) == sum(len(x[source]) for x in original.values())
    assert [len(roles[r]["lower"]) for r in d.TRAIN_ROLES[1:]] == [8, 4, 4]
    for distributions in roles.values():
        assert (distributions["real"].groupby("identity").size() == 2).all()
    # A seed is an independence unit even if its source ID is different.
    broken = copy.deepcopy(roles)
    broken["earlystop"]["lower"].loc[0, "latent_seed"] = broken["train"]["lower"].latent_seed.iloc[0]
    with pytest.raises(ValueError, match="Overlapping"):
        d._assert_disjoint(broken)


def test_identity_partition_rejects_leakage_and_smoke_keeps_whole_people():
    roles = d._freeze_roles(originals(), 7, smoke_rows=3)
    assert all(len(distributions["real"]) == 4 for distributions in roles.values())
    assert all(len(distributions["lower"]) == 3 for distributions in roles.values())
    broken = copy.deepcopy(roles)
    broken["earlystop"]["real"].loc[0, "identity"] = broken["train"]["real"].identity.iloc[0]
    with pytest.raises(ValueError, match="Overlapping"):
        d._assert_disjoint(broken)
    with pytest.raises(ValueError, match="Too few identities"):
        d._identity_parts(originals()["train"]["real"].iloc[:2], [.5, .25, .25], 7)


def test_prepare_opens_development_only_and_p_is_shared_and_rows_align(prepared):
    output, source, metadata, opened = prepared
    assert opened and all("_design_" not in path and "_test_" not in path for path in opened)
    assert not metadata["test_assets_loaded"]
    assert not (output / "data/arrays/feature/test_calibration").exists()
    for level in ("feature", "pixel"):
        lower = d.load_arrays(output, level, "lower", d.TRAIN_ROLES)
        upper = d.load_arrays(output, level, "upper", d.TRAIN_ROLES)
        for role in d.TRAIN_ROLES:
            assert lower[role]["p"].filename == upper[role]["p"].filename
            for label, distribution in (("p", "real"), ("q", "lower")):
                manifest = pd.read_csv(output / metadata["roles"][role][distribution]["path"])
                actual = lower[role][label]
                np.testing.assert_array_equal(actual[:, 0] if level == "feature" else actual[:, 0, 0, 0], manifest.feature_row)
                assert not actual.flags.writeable


def test_final_gate_checks_hash_then_preserves_sealed_metadata(prepared):
    output, _, metadata, opened = prepared
    before = (output / "data/metadata.json").read_bytes()
    opened.clear()
    with pytest.raises(RuntimeError, match="require freeze"):
        d.prepare_evaluation_data(output)
    assert not opened
    (output / "freeze.json").write_text('{"selected": "fixture"}\n')
    (output / "freeze.sha256").write_text("0" * 64)
    with pytest.raises(RuntimeError, match="SHA256"):
        d.prepare_evaluation_data(output)
    assert not opened
    (output / "freeze.sha256").write_text(d.sha256_file(output / "freeze.json") + "\n")
    evaluated = d.prepare_evaluation_data(output, metadata)
    assert evaluated["test_assets_loaded"]
    assert any("_design_" in path for path in opened) and any("_test_" in path for path in opened)
    assert (output / "data/metadata.json").read_bytes() == before
    assert (output / "data/evaluation_metadata.json").is_file()
    for level in ("feature", "pixel"):
        final = d.load_arrays(output, level, "upper", d.TEST_ROLES)
        assert set(final) == set(d.TEST_ROLES)


def test_prepared_arrays_are_portable_and_tampering_is_detected(prepared, tmp_path):
    output, _, metadata, _ = prepared
    relocated = tmp_path / "relocated"
    shutil.copytree(output, relocated)
    arrays = d.load_arrays(relocated, "feature", "lower", ["train"])
    assert str(arrays["train"]["p"].filename).startswith(str(relocated))
    path = relocated / metadata["arrays"]["feature"]["train"]["real"]["path"]
    with path.open("ab") as handle:
        handle.write(b"altered")
    with pytest.raises(RuntimeError, match="changed"):
        d.load_arrays(relocated, "feature", "lower", ["train"])


def test_source_hash_and_payload_row_identity_are_verified(prepared):
    _, source, metadata, _ = prepared
    frame = originals()["train"]["real"].iloc[[4, 0, 2]].reset_index(drop=True)
    target = np.empty((3, 2048), dtype=np.float32)
    d._gather_features(frame, metadata["inputs"], target, {})
    np.testing.assert_array_equal(target[:, 0], [4, 0, 2])
    inputs = copy.deepcopy(metadata["inputs"])
    inputs["real_train"]["feature_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="SHA256"):
        d._gather_features(frame, inputs, target, {})
    inputs = copy.deepcopy(metadata["inputs"])
    inputs["real_train"]["input_fingerprint"] = "wrong row order"
    with pytest.raises(RuntimeError, match="fingerprint"):
        d._gather_features(frame, inputs, target, {})
    generated = originals()["train"]["lower"].iloc[:2].copy()
    generated["tensor_path"] = str(source / "lower_train_pixels.pt")
    generated.loc[0, "latent_seed"] += 1
    pixels = np.empty((2, 3, 64, 64), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="seeds"):
        d._gather_pixels(generated, "lower", inputs, pixels, {}, source, {})


def test_existing_data_config_and_historical_output_location_are_protected(prepared):
    output, source, metadata, _ = prepared
    assert d.prepare_data(output, source, {"split_seed": 7}) == metadata
    with pytest.raises(RuntimeError, match="immutable"):
        d.prepare_data(output, source, {"split_seed": 8})
    with pytest.raises(ValueError, match="outside the historical"):
        d.prepare_data(source / "new_run", source, {"split_seed": 7})
    with pytest.raises(RuntimeError, match="require freeze"):
        d.load_arrays(output, "feature", "lower", ["test_evaluation"], metadata)


@pytest.mark.parametrize("field,value", [("checkpoint_sha256", "different_checkpoint"),
                                        ("repository_commit", "different_code"), ("force_reference_ops", False)])
def test_generated_shards_must_use_the_frozen_generator(prepared, field, value):
    _, source, metadata, _ = prepared
    path = source / "lower_train_pixels.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload[field] = value
    torch.save(payload, path)
    generated = originals()["train"]["lower"].iloc[:2].copy()
    generated["tensor_path"] = str(path)
    with pytest.raises(RuntimeError, match=field):
        d._gather_pixels(generated, "lower", metadata["inputs"], np.empty((2, 3, 64, 64), dtype=np.uint8), {}, source, {})
