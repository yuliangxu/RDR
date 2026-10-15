"""Full-template MIDASim provenance, original P roles, and selection-before-test gates."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.AGP import midasim_full as full


def _save_parent_study(parent, name, parts):
    folder = parent / "data" / name
    folder.mkdir(parents=True)
    frames, arrays, roles = [], {}, []
    for role, side, x, identities in parts:
        if role not in roles:
            roles.append(role)
        target = folder / f"{role}_{side.lower()}.npy"
        np.save(target, x, allow_pickle=False)
        rows = identities.copy()
        rows["distribution"], rows["role"] = side, role
        rows["array_row"] = np.arange(len(rows))
        rows["row_sha256"] = full.row_hashes(x)
        frames.append(rows)
        arrays[f"{role}_{side.lower()}"] = dict(full.receipt(parent, target), shape=list(x.shape), dtype=str(x.dtype))
    manifest = folder / "manifest.csv"
    pd.concat(frames, ignore_index=True).to_csv(manifest, index=False)
    full.write(folder / "audit.json", {"fixture": "subject-disjoint synthetic source"})
    return dict(roles=roles, arrays=arrays, manifest=full.receipt(parent, manifest),
                audit=full.receipt(parent, folder / "audit.json"))


@pytest.fixture
def original_parent(tmp_path):
    parent, inputs = tmp_path / "parent", tmp_path / "originals"
    inputs.mkdir()
    source = inputs / "yx1_xtrain_raw.csv"
    dev = np.zeros((10443, 614), dtype=np.float64)
    dev[np.arange(len(dev)), 1+np.arange(len(dev)) % 613] = 1
    np.savetxt(source, dev, delimiter=",", fmt="%.1f")
    # Training and stopping deliberately interleave original source rows.
    permutation = np.random.default_rng(91).permutation(len(dev))
    ids_train, ids_stop = np.sort(permutation[:9162]), np.sort(permutation[9162:])
    test = np.roll(dev[:2002], 1, axis=1).astype(np.float32)

    def identities(indices, is_test=False):
        role = "test" if is_test else "development"
        return pd.DataFrame(dict(sample_id=[f"{role}:sample:{i}" for i in indices],
            subject_id=[f"{role}:subject:{i}" for i in indices],
            source_file=str(inputs / "yx1_xtest_raw.csv") if is_test else str(source),
            source_row=indices, original_distribution="P", original_role=role))

    main = dict(train=(dev[ids_train].astype(np.float32), identities(ids_train)),
                stopping=(dev[ids_stop].astype(np.float32), identities(ids_stop)),
                test=(test, identities(np.arange(len(test)), True)))
    final = [(role, "P", x, frame) for role, (x, frame) in main.items()]
    boundaries = [0, 7310, 8354, 9399, 10443]
    selection = [(role, "P", dev[boundaries[i]:boundaries[i+1]].astype(np.float32),
                  identities(np.arange(boundaries[i], boundaries[i+1])))
                 for i, role in enumerate(("train", "stopping", "calibration", "evaluation"))]
    null = []
    for role, (x, frame) in main.items():
        n = len(x)//2
        null.extend([(role, "P", x[:n], frame.iloc[:n]), (role, "Q", x[n:], frame.iloc[n:])])
    studies = {name: _save_parent_study(parent, name, parts)
               for name, parts in (("selection", selection), ("final", final), ("null", null))}
    taxonomy = parent / "data/taxa.csv"
    pd.DataFrame(dict(taxon_index=range(614), taxon_id=[f"t{i}" for i in range(614)],
                      phylum="Synthetic")).to_csv(taxonomy, index=False)
    data = dict(data_root=str(inputs), input_sha256={source.name: full.sha(source)}, studies=studies,
                taxa=full.receipt(parent, taxonomy))
    full.write(parent / "data/protocol.json", data)
    config = full.read(Path(full.shared.__file__).with_name("selection_config.json"))
    full.write(parent / "config.json", config)
    (parent / "source").mkdir()
    (parent / "source/fixture.txt").write_text("Immutable synthetic source")
    full.write(parent / "protocol.json", dict(config_sha256=full.sha(parent / "config.json"),
        data_protocol_sha256=full.sha(parent / "data/protocol.json"),
        source_sha256={"fixture.txt": full.sha(parent / "source/fixture.txt")}))
    full.write(parent / "COMPLETE.json", dict(status="complete", protocol_sha256=full.sha(parent / "protocol.json"),
        outputs={"config.json": full.sha(parent / "config.json")}))
    return parent, data, dev


@pytest.fixture
def generator_ready(original_parent, tmp_path, monkeypatch):
    parent, data, dev = original_parent
    output, pilot = tmp_path / "new_study", tmp_path / "pilot"
    tarball = pilot / "vendor/MIDASim_2.0.tar.gz"
    tarball.parent.mkdir(parents=True)
    tarball.write_bytes(b"Synthetic vendor tarball; no R code is executed by these tests.")
    actual_sha = full.sha

    def fixture_sha(path):
        if Path(path) == tarball:
            return "841f0d14bb8b64176fc7fea34136a691fbe49e5e703fabc16571d0282bbef841"
        return actual_sha(path)

    monkeypatch.setattr(full, "sha", fixture_sha)
    full.prepare_generator(output, parent, pilot)
    return output, parent, data, dev


def _seal_generation(output):
    gen = output / "generator"
    directory = gen / "generation"
    directory.mkdir()
    for index, (label, n) in enumerate(full.ROLE_SIZES.items()):
        x = np.zeros((n, 614))
        x[:, 1+(index % 613)] = .25
        x[:, 1+((index+1) % 613)] = .75
        np.savetxt(directory / f"{label}.csv", x, delimiter=",", fmt="%.2f")
        pd.DataFrame(dict(output_row=np.arange(n), generated_row=np.arange(n))).to_csv(
            directory / f"{label}_indices.csv", index=False)
        full.write(directory / f"{label}_metrics.json", {"fixture": label})
    for name in ("fitted.rds", "modified.rds", "sessionInfo.txt", "taxon_map.csv"):
        (directory / name).write_text("Sealed fixture artifact")
    for name in ("metrics.json", "package_versions.json", "input_metadata.json"):
        full.write(directory / name, {"fixture": name})
    full.write(directory / "COMPLETE.json", dict(status="complete",
        config_sha256=full.sha(gen / "config.json"), protocol_sha256=full.sha(gen / "protocol.json"),
        outputs={str(path.relative_to(gen)): full.sha(path) for path in directory.iterdir() if path.is_file()}))


def test_full_template_preserves_original_row_order_and_does_not_include_test(generator_ready):
    output, parent, data, dev = generator_ready
    generated = output / "generator"
    np.testing.assert_array_equal(np.loadtxt(generated / "data/generator_train.csv", delimiter=","), dev)
    original = Path(data["data_root"]) / "yx1_xtrain_raw.csv"
    assert full.sha(generated / "data/generator_train.csv") == full.sha(original)
    manifest = pd.read_csv(generated / "data/manifest.csv", keep_default_na=False)
    assert manifest.source_row.tolist() == list(range(10443))
    assert manifest.sample_id.str.startswith("development:").all()
    protocol = full.read(generated / "data/protocol.json")
    assert protocol["restored_zero_taxa"] == [0]
    assert len(protocol["active_taxa"]) == 613
    config = full.read(generated / "config.json")
    assert config["generation_seed_offset"] == 100000
    assert config["roles"] == full.ROLE_SIZES
    assert config["role_order"] == list(full.ROLE_SIZES)
    full.verify_generator(output, complete=False)


def test_analysis_waits_for_sealed_generation_before_creating_data(generator_ready):
    output, *_ = generator_ready
    with pytest.raises(FileNotFoundError):
        full.prepare_analysis(output)
    assert not (output / "analysis").exists()


def test_analysis_preserves_all_p_and_null_values_and_uses_fresh_q_identities(generator_ready, monkeypatch):
    output, parent, original, _ = generator_ready
    _seal_generation(output)
    full.prepare_analysis(output)
    analysis = output / "analysis"
    config = full.read(analysis / "config.json")
    assert config["seed"] == full.read(parent / "config.json")["seed"] == 2026100400
    assert len(full.shared.selection_tasks(config)) == 80
    prepared = full.read(analysis / "data/protocol.json")
    for stage, entry in prepared["studies"].items():
        frame = pd.read_csv(analysis / entry["manifest"]["path"], keep_default_na=False)
        for key, spec in entry["arrays"].items():
            role, side = key.rsplit("_", 1)
            rows = frame[frame.role.eq(role) & frame.distribution.eq(side.upper())]
            actual = np.load(analysis / spec["path"], allow_pickle=False)
            assert actual.dtype == np.float32
            assert rows.row_sha256.tolist() == full.row_hashes(actual)
            assert rows.array_row.tolist() == list(range(len(actual)))
            if side == "p" or stage == "null":
                source = original["studies"][stage]["arrays"][key]
                assert full.sha(analysis / spec["path"]) == source["sha256"]
                old = pd.read_csv(parent / original["studies"][stage]["manifest"]["path"], keep_default_na=False)
                old = old[old.role.eq(role) & old.distribution.eq(side.upper())]
                assert rows.sample_id.tolist() == old.sample_id.tolist()
                assert rows.subject_id.tolist() == old.subject_id.tolist()
            else:
                assert len(actual) == full.ROLE_SIZES[stage+"_"+role]
                assert rows.sample_id.str.startswith(f"MIDASim:{stage}:{role}:").all()
                assert rows.subject_id.eq("").all()
                assert rows.source_file.str.endswith(f"generation/{stage}_{role}.csv").all()
                assert not actual[:,0].any()
                if "original_manifest_row" in rows:
                    assert rows.original_manifest_row.eq("").all()
    assert full.analysis_gate(output) == analysis
    # The preserved shared runner must reject final/null work before loading any arrays.
    monkeypatch.setattr(full.shared, "load_arrays", lambda *args, **kwargs:
                        pytest.fail("Assessment arrays loaded before selection was frozen"))
    for stage in ("final", "null"):
        with pytest.raises(FileNotFoundError):
            full.shared.run_task(analysis, stage, 0)
        assert not (analysis / stage / "00").exists()
    manifest = analysis / prepared["studies"]["selection"]["manifest"]["path"]
    manifest.write_text(manifest.read_text()+"\n")
    with pytest.raises(ValueError, match="artifact|receipt|manifest|data"):
        full.analysis_gate(output)


def test_generation_gate_rejects_missing_role_receipt_and_changed_provenance(generator_ready):
    output, *_ = generator_ready
    _seal_generation(output)
    gen = output / "generator"
    full.verify_generator(output)
    marker = full.read(gen / "generation/COMPLETE.json")
    original = dict(marker["outputs"])
    for filename in ("selection_evaluation.csv", "selection_train_indices.csv", "modified.rds"):
        marker["outputs"] = dict(original)
        marker["outputs"].pop("generation/"+filename)
        full.write(gen / "generation/COMPLETE.json", marker)
        with pytest.raises(ValueError, match="roles|receipt|artifact"):
            full.verify_generator(output)
    marker["outputs"] = original
    marker["config_sha256"] = "changed"
    full.write(gen / "generation/COMPLETE.json", marker)
    with pytest.raises(ValueError, match="provenance"):
        full.verify_generator(output)


def test_analysis_rejects_parent_completion_changed_after_generator_preparation(generator_ready):
    output, parent, *_ = generator_ready
    _seal_generation(output)
    marker = full.read(parent / "COMPLETE.json")
    marker["changed_after_generator_preparation"] = True
    full.write(parent / "COMPLETE.json", marker)
    with pytest.raises(ValueError, match="parent|Parent"):
        full.prepare_analysis(output)
    assert not (output / "analysis").exists()
