"""Scientific pairing, immutable provenance, and four-loss convergence checks."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from experiments.simulations import convergence as c
from experiments.simulations import workflow as w


@pytest.fixture
def prepared(tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = w.read_json(w.DEFAULT_CONFIG)
    cfg.update(cases=["D2_noise0.00"], selection_repeats=1, convergence_repeats=2,
               losses=["hellinger"], architectures=["bottleneck8_residual64"], output_alphas=[2],
               train_n=16, validation_n=16, calibration_n=24, evaluation_n=24,
               sample_sizes=[8, 16], max_epochs=2, min_epochs=1, patience=1,
               bootstrap_repetitions=20, min_cell_count=2)
    original, protocol = w.prepare(tmp_path / "original", cfg, "selection")
    for index in range(len(w.task_specs(protocol))):
        w.run_task(original, protocol, index)
    w.freeze(original)
    output, protocol = c.prepare(original, tmp_path / "convergence")
    return original, output, protocol


def test_paired_losses_sizes_and_separate_selection_streams(prepared):
    original, output, protocol = prepared
    before = {str(p.relative_to(original)): w.digest(p) for p in original.rglob("*") if p.is_file()}
    assert [row["loss"] for row in protocol["candidates"]] == list(c.LOSSES)
    assert all(row["architecture"] == protocol["selected"]["architecture"]
               and row["output_alpha"] == protocol["selected"]["output_alpha"]
               for row in protocol["candidates"])
    training_by_size = {}
    for index, (case, repeat, n) in enumerate(w.task_specs(protocol)):
        c.run_task(output, index)
        records = [w.read_json(w.task_path(output, case, repeat, n, candidate) / "training.json")
                   for candidate in protocol["candidates"]]
        for record in records:
            assert record["data_seed_vectors"] == records[0]["data_seed_vectors"]
            assert record["model_seed"] == records[0]["model_seed"]
            assert record["initialization_seed_vector"][1] == 2
            assert all(seed[1] == 2 for seed in record["data_seed_vectors"].values())
        training_by_size[(repeat, n)] = records[0]
        # Fresh evaluation draws differ from the design-stage selection draws.
        arrays = w.samples(protocol["config"], "convergence", case, repeat, n)[1]
        old_arrays = w.samples(protocol["config"], "selection", case, repeat, n)[1]
        assert all(not np.array_equal(arrays[key], old_arrays[key]) for key in arrays)
    for repeat in range(2):
        a, b = [training_by_size[(repeat, n)] for n in (8, 16)]
        assert a["model_seed"] == b["model_seed"]
        for role in ("validation", "calibration", "evaluation"):
            for source in ("P", "Q"):
                assert a["data_seed_vectors"][f"{role}_{source}"] == b["data_seed_vectors"][f"{role}_{source}"]
        for source in ("P", "Q"):
            assert a["data_seed_vectors"][f"train_{source}"] != b["data_seed_vectors"][f"train_{source}"]
    _, rows, local = c.collect(output)
    assert len(rows) == len(local) == 16
    assert all(np.isfinite(row[metric]) for row in rows
               for metric in ("validation_brier", "evaluation_brier", "evaluation_mse"))
    assert before == {str(p.relative_to(original)): w.digest(p) for p in original.rglob("*") if p.is_file()}


def test_hellinger_scientific_steps_match_original_workflow_exactly(prepared, tmp_path):
    original, output, protocol = prepared
    baseline, baseline_protocol = w.prepare(tmp_path / "original_convergence", protocol["config"],
                                          "convergence", original / "selection.json")
    for index in (0, 1):
        c.run_task(output, index)
        w.run_task(baseline, baseline_protocol, index)
        case, repeat, n = w.task_specs(protocol)[index]
        new = w.task_path(output, case, repeat, n, protocol["selected"])
        old = w.task_path(baseline, case, repeat, n, protocol["selected"])
        for name in ("metrics.json", "cells.json", "history.json"):
            assert w.read_json(new / name) == w.read_json(old / name)
        a, b = w.read_json(new / "training.json"), w.read_json(old / "training.json")
        a.pop("protocol_sha256")
        b.pop("protocol_sha256")
        assert a == b
        with np.load(new / "predictions.npz") as a, np.load(old / "predictions.npz") as b:
            assert a.files == b.files
            assert all(np.array_equal(a[key], b[key]) for key in a.files)
        a, b = [torch.load(folder / "model.pt", weights_only=True) for folder in (new, old)]
        assert all(torch.equal(a[key], b[key]) for key in a)


def test_incomplete_fit_grid_cannot_publish(prepared):
    _, output, _ = prepared
    c.run_task(output, 0)
    with pytest.raises(ValueError, match="Incomplete task"):
        c.report(output)
    assert not (output / c.COMPLETE).exists()
    assert not (output / "report.md").exists()


@pytest.mark.parametrize("target", ["selection.json", "selection.sha256", "protocol.json", "source"])
def test_reference_or_snapshot_tamper_rejected_before_fit(prepared, target):
    _, output, _ = prepared
    path = (output / "source/utils/losses.py") if target == "source" else (output / "reference" / target)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        c.run_task(output, 0)
    assert not (output / "fits").exists()


def test_loss_architecture_cannot_be_changed_in_a_resealed_protocol(prepared):
    _, output, protocol = prepared
    protocol["candidates"][1]["architecture"] = "wide64"
    (output / "protocol.json").write_text(json.dumps(protocol))
    (output / "protocol.sha256").write_text(w.digest(output / "protocol.json") + "\n")
    with pytest.raises(ValueError, match="differs from its frozen reference"):
        c.run_task(output, 0)
    assert not (output / "fits").exists()


def test_fit_identity_is_checked_even_when_artifact_hashes_are_resealed(prepared):
    _, output, protocol = prepared
    c.run_task(output, 0)
    case, repeat, n = w.task_specs(protocol)[0]
    folder = w.task_path(output, case, repeat, n, protocol["candidates"][0])
    training = w.read_json(folder / "training.json")
    training["data_seed_vectors"]["evaluation_P"][1] = 1
    (folder / "training.json").write_text(json.dumps(training))
    manifest = w.read_json(folder / "complete.json")
    manifest["training.json"] = w.digest(folder / "training.json")
    (folder / "complete.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="seed/protocol identity"):
        c.run_task(output, 0)


def test_snapshot_runs_without_original_from_an_unrelated_directory(prepared, tmp_path):
    original, output, protocol = prepared
    original.rename(original.with_name("selection_not_available"))
    script = output / "source/experiments/simulations/convergence.py"
    result = subprocess.run([sys.executable, "-B", str(script), "fit", "--output", str(output),
                             "--task-index", "0", "--threads", "1"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    case, repeat, n = w.task_specs(protocol)[0]
    for candidate in protocol["candidates"]:
        assert (w.task_path(output, case, repeat, n, candidate) / "complete.json").is_file()


def test_completed_report_pins_all_fits_and_artifacts(prepared):
    _, output, protocol = prepared
    for index in range(len(w.task_specs(protocol))):
        c.run_task(output, index)
    assert c.report(output) == output / "report.md"
    marker = w.read_json(output / c.COMPLETE)
    assert marker["fit_count"] == 16 and marker["task_count"] == 4
    assert len(marker["fit_manifest_sha256"]) == 16
    assert "report.md" in marker["artifact_sha256"]
    assert c.report(output).is_file()
    report = output / "report.md"
    report.write_text(report.read_text() + "modified")
    with pytest.raises(ValueError, match="report artifact changed"):
        c.report(output)


def test_prepare_rejects_environment_and_existing_directory(prepared, tmp_path):
    original, output, _ = prepared
    with pytest.raises(FileExistsError):
        c.prepare(original, output)
    torch.set_num_threads(2)
    try:
        with pytest.raises(ValueError, match="environment differs"):
            c.prepare(original, tmp_path / "bad_environment")
    finally:
        torch.set_num_threads(1)
    assert not (tmp_path / "bad_environment").exists()
