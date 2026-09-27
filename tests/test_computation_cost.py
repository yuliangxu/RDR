"""Measured fitting boundaries, exact replay, and immutable cost provenance."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from experiments.simulations import computation_cost as cost
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
    selection, protocol = w.prepare(tmp_path / "selection", cfg, "selection")
    for index in range(len(w.task_specs(protocol))):
        w.run_task(selection, protocol, index)
    w.freeze(selection)
    reference, original = c.prepare(selection, tmp_path / "convergence")
    output, protocol = cost.prepare(reference, tmp_path / "cost", repeats=2)
    return reference, original, output, protocol


def complete_reference(reference, protocol):
    """Produce a tiny complete scientific grid and its real rendered report."""
    for index in range(len(w.task_specs(protocol))):
        c.run_task(reference, index)
    c.report(reference)


def complete_cost(output, protocol):
    for index in range(len(cost.task_specs(protocol))):
        cost.run_task(output, index)


def test_exact_replay_full_grid_and_production_unchanged(prepared):
    reference, original, output, protocol = prepared
    complete_reference(reference, original)
    before = {str(path.relative_to(reference)): w.digest(path) for path in reference.rglob("*") if path.is_file()}
    complete_cost(output, protocol)
    _, rows = cost.collect(output)
    assert len(rows) == 16
    assert {row["loss"] for row in rows} == set(c.LOSSES)
    assert all(row["wall_seconds"] > 0 and row["cpu_seconds"] > 0 for row in rows)
    assert all(row["warmup"]["timed"] is False and row["epochs"] == 2 for row in rows)
    for index, spec in enumerate(cost.task_specs(protocol)):
        task_rows = [row for row in rows if (row["case"], row["repeat"], row["train_n"]) == spec]
        assert all(row["model_seed"] == task_rows[0]["model_seed"] for row in task_rows)
        assert all(row["data_seed_vectors"] == task_rows[0]["data_seed_vectors"] for row in task_rows)
        assert task_rows[0]["execution_order"] == list(c.LOSSES[index % 4:] + c.LOSSES[:index % 4])
    assert before == {str(path.relative_to(reference)): w.digest(path) for path in reference.rglob("*") if path.is_file()}
    cost.run_task(output, 0)  # Intact completed tasks resume without overwriting.
    assert cost.report(output) == output / "report.md"
    marker = w.read_json(output / cost.COMPLETE)
    assert marker["fit_count"] == 16 and marker["task_count"] == 4
    assert cost.report(output).is_file()
    assert (output / "convergence_with_cost/report.md").is_file()
    assert before == {str(path.relative_to(reference)): w.digest(path) for path in reference.rglob("*") if path.is_file()}


def test_timer_excludes_warmup_and_sampling(prepared, monkeypatch):
    _, _, output, protocol = prepared
    real_fit = w.fit_ratio_mlp
    calls = []
    clock = {"wall": 0., "cpu": 0.}
    def measured_fit(*args, **kwargs):
        calls.append(kwargs["max_epochs"])
        result = real_fit(*args, **kwargs)
        clock["wall"] += 100. if kwargs["max_epochs"] == 1 else 7.
        clock["cpu"] += 50. if kwargs["max_epochs"] == 1 else 3.
        return result
    monkeypatch.setattr(w, "fit_ratio_mlp", measured_fit)
    monkeypatch.setattr(cost.time, "perf_counter", lambda: clock["wall"])
    monkeypatch.setattr(cost.time, "process_time", lambda: clock["cpu"])
    cost.run_task(output, 0)
    assert calls == [1] * 4 + [2] * 4
    case, repeat, n = cost.task_specs(protocol)[0]
    for candidate in protocol["candidates"]:
        row = cost.verify_timing(output, protocol, 0, case, repeat, n, candidate)
        assert row["wall_seconds"] == 7. and row["cpu_seconds"] == 3.


def test_publication_requires_original_completion(prepared):
    _, _, output, protocol = prepared
    complete_cost(output, protocol)
    with pytest.raises(ValueError, match="Original convergence study is not complete"):
        cost.collect(output)
    assert not (output / cost.COMPLETE).exists()


@pytest.mark.parametrize("changed", ["model", "history", "training"])
def test_reference_parameter_history_or_metadata_mismatch_rejected(prepared, changed):
    reference, original, output, protocol = prepared
    complete_reference(reference, original)
    complete_cost(output, protocol)
    case, repeat, n = cost.task_specs(protocol)[0]
    folder = w.task_path(output, case, repeat, n, protocol["candidates"][0])
    row = w.read_json(folder / "timing.json")
    # The timing side is resealed: equality against the immutable scientific fit
    # must still reject it, beyond ordinary artifact checksum verification.
    if changed == "training":
        row["training"]["restored_val_loss"] += 1.
    else:
        row[f"{changed}_sha256"] = "0" * 64
    (folder / "timing.json").write_text(json.dumps(row))
    (folder / "complete.json").write_text(json.dumps({"timing.json": w.digest(folder / "timing.json")}))
    with pytest.raises(ValueError, match="does not exactly match"):
        cost.collect(output)


def test_source_snapshot_change_rejected(prepared):
    _, _, output, _ = prepared
    source = output / "source/experiments/simulations/computation_cost.py"
    source.write_text(source.read_text() + "\n# modified\n")
    with pytest.raises(ValueError, match="source snapshot changed"):
        cost.run_task(output, 0)
    assert not (output / "fits").exists()


def test_resealed_seed_identity_rejected(prepared):
    _, _, output, protocol = prepared
    cost.run_task(output, 0)
    case, repeat, n = cost.task_specs(protocol)[0]
    folder = w.task_path(output, case, repeat, n, protocol["candidates"][0])
    row = w.read_json(folder / "timing.json")
    row["data_seed_vectors"]["train_P"][1] = 1
    (folder / "timing.json").write_text(json.dumps(row))
    (folder / "complete.json").write_text(json.dumps({"timing.json": w.digest(folder / "timing.json")}))
    with pytest.raises(ValueError, match="seed/protocol identity"):
        cost.run_task(output, 0)


def test_snapshot_executes_from_unrelated_directory(prepared, tmp_path):
    _, _, output, protocol = prepared
    script = output / "source/experiments/simulations/computation_cost.py"
    result = subprocess.run([sys.executable, "-B", str(script), "fit", "--output", str(output),
                             "--task-index", "0", "--threads", "1"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    case, repeat, n = cost.task_specs(protocol)[0]
    assert all((w.task_path(output, case, repeat, n, candidate) / "complete.json").is_file()
               for candidate in protocol["candidates"])


def test_cpu_only_and_repeats_bound_and_existing_output(prepared, tmp_path):
    reference, _, output, _ = prepared
    with pytest.raises(FileExistsError):
        cost.prepare(reference, output, repeats=1)
    for repeats in (0, 3, True):
        with pytest.raises(ValueError, match="subset"):
            cost.prepare(reference, tmp_path / f"bad-{repeats}", repeats=repeats)


def test_parameter_digest_independent_of_mapping_order():
    state = {"a": torch.tensor([1., 2.]), "b": torch.ones((2, 2))}
    assert cost.parameter_digest(state) == cost.parameter_digest(dict(reversed(list(state.items()))))
    state["a"][0] += 1
    assert cost.parameter_digest(state) != cost.parameter_digest({"a": torch.tensor([1., 2.]), "b": torch.ones((2, 2))})
