"""Exercise architecture expansion with completed, hashed fixture fits."""

import pytest

from experiments.CelebA import model_selection as workflow


def _record_fit(output, task, brier):
    directory = workflow.task_directory(output, task)
    directory.mkdir(parents=True)
    row = {**task, "brier": brier, "local_gap": .1, "supported_mass": 1.}
    for name in workflow.FIT_FILES:
        if name == "metrics.json":
            workflow.write(directory / name, row)
        else:
            (directory / name).write_bytes(b"fixture artifact; no model training\n")
    workflow.write(directory / "COMPLETE.json", {
        "task_identity": workflow.task_identity(task),
        "protocol_sha256": workflow.sha(output / "protocol.json"),
        "files": {name: workflow.sha(directory / name) for name in workflow.FIT_FILES},
    })


def _prepared_expansion(output, monkeypatch):
    cfg = {"levels": ["feature"], "branches": ["lower", "upper"],
           "losses": ["js", "kl"], "output_alphas": [1, 2], "repeats": 2,
           "sensitivity": {"architecture": "small", "leaders_per_pair": 2,
                           "expansion_minimum_brier_gain": .002,
                           "expansion_positive_repeat_fraction": .8,
                           "expansion_paired_se_multiplier": 1.}}
    protocol = {"config": cfg, "tasks": workflow.grid(cfg), "files": {}, "smoke": True}
    workflow.seal(output / "protocol.json", protocol)
    for task in protocol["tasks"]:
        setting = 2 * cfg["losses"].index(task["loss"]) + cfg["output_alphas"].index(task["output_alpha"])
        _record_fit(output, task, .2 + .01 * setting + .001 * task["repeat"])
    workflow.freeze_baseline(output, protocol)
    sensitivity = workflow.tasks_for(output, protocol, "sensitivity")
    assert {(t["loss"], t["output_alpha"]) for t in sensitivity} == {("js", 1), ("js", 2)}
    for task in sensitivity:
        # Only the lower pair triggers; both pairs must subsequently expand.
        value = .197 if task["branch"] == "lower" else .202
        _record_fit(output, task, value + .01 * (task["output_alpha"] - 1) + .001 * task["repeat"])
    monkeypatch.setattr(workflow, "report", lambda *args, **kwargs: None)
    from experiments.CelebA import selection_data
    monkeypatch.setattr(selection_data, "prepare_evaluation_data",
                        lambda *args, **kwargs: pytest.fail("Test assets read before final freeze"))
    workflow.advance(output, "sensitivity")
    return protocol


def _complete_expansion(output, protocol):
    tasks = workflow.tasks_for(output, protocol, "expanded")
    for task in tasks:
        _record_fit(output, task, .19 + .01 * (task["output_alpha"] - 1) + .001 * task["repeat"])
    return tasks


def test_trigger_expands_both_pairs_and_freeze_requires_full_grid(tmp_path, monkeypatch):
    protocol = _prepared_expansion(tmp_path, monkeypatch)
    decision = workflow.read_sealed(tmp_path / "expansion.json")
    assert decision["expanded_representations"] == ["feature"]
    triggered = [row for row in decision["diagnostics"] if row["expand"]]
    assert {row["branch"] for row in triggered} == {"lower"}
    assert triggered[0]["mean_brier_gain"] == pytest.approx(.003)
    expanded = workflow.tasks_for(tmp_path, protocol, "expanded")
    assert {task["branch"] for task in expanded} == {"lower", "upper"}
    assert len(expanded) == 8
    assert not (tmp_path / "freeze.json").exists()
    with pytest.raises(ValueError, match="Missing completed task"):
        workflow.freeze_final(tmp_path, protocol)
    assert not (tmp_path / "freeze.json").exists()
    for task in expanded[:-1]:
        _record_fit(tmp_path, task, .19 + .01 * (task["output_alpha"] - 1) + .001 * task["repeat"])
    with pytest.raises(ValueError, match="Missing completed task"):
        workflow.freeze_final(tmp_path, protocol)
    last = expanded[-1]
    _record_fit(tmp_path, last, .19 + .01 * (last["output_alpha"] - 1) + .001 * last["repeat"])
    workflow.freeze_final(tmp_path, protocol)
    frozen = workflow.read_sealed(tmp_path / "freeze.json")
    assert {record["chosen"]["candidate"] for record in frozen["selections"].values()} == {"small_kl_a1"}
    evaluation = workflow.tasks_for(tmp_path, protocol, "evaluation")
    assert len(evaluation) == 4
    assert {task["architecture"] for task in evaluation} == {"small"}
    assert len(frozen["selected_fit_receipts"]) == 4
    for relative, digest in frozen["selected_fit_receipts"].items():
        assert workflow.sha(tmp_path / relative / "COMPLETE.json") == digest
    baseline = workflow.read_sealed(tmp_path / "baseline_selection.json")
    assert {record["chosen"]["architecture"] for record in baseline["selections"].values()} == {"baseline"}


def test_truncated_expanded_manifest_cannot_bypass_full_grid_check(tmp_path, monkeypatch):
    protocol = _prepared_expansion(tmp_path, monkeypatch)
    _complete_expansion(tmp_path, protocol)
    path = tmp_path / "expanded_tasks.json"
    record = workflow.read_sealed(path)
    record["tasks"] = record["tasks"][:-1]
    workflow.write(path, record)
    path.with_suffix(".sha256").write_text(workflow.sha(path) + "\n")
    with pytest.raises(ValueError, match="Expanded architecture grid is incomplete"):
        workflow.freeze_final(tmp_path, protocol)
    assert not (tmp_path / "freeze.json").exists()


@pytest.mark.parametrize("stage,parent_key", [
    ("sensitivity", "baseline_selection_sha256"),
    ("expanded", "expansion_sha256"),
    ("evaluation", "freeze_sha256"),
])
def test_stage_self_hash_does_not_replace_parent_lineage(tmp_path, monkeypatch, stage, parent_key):
    protocol = _prepared_expansion(tmp_path, monkeypatch)
    if stage == "evaluation":
        _complete_expansion(tmp_path, protocol)
        workflow.freeze_final(tmp_path, protocol)
    path = tmp_path / f"{stage}_tasks.json"
    record = workflow.read_sealed(path)
    record[parent_key] = "0" * 64
    # Simulate a correctly self-hashed task record copied from the wrong parent.
    workflow.write(path, record)
    path.with_suffix(".sha256").write_text(workflow.sha(path) + "\n")
    assert workflow.read_sealed(path) == record
    with pytest.raises(ValueError, match="lineage changed"):
        workflow.tasks_for(tmp_path, protocol, stage)


def test_duplicate_stage_submission_never_calls_slurm(tmp_path, monkeypatch):
    workflow.write(tmp_path / "jobs" / "sensitivity-feature-123.json", {"job_id": "123"})
    monkeypatch.setattr(workflow.subprocess, "check_output",
                        lambda *args, **kwargs: pytest.fail("Duplicate submission reached Slurm"))
    with pytest.raises(RuntimeError, match="already submitted"):
        workflow.submit_stage(tmp_path, {}, "sensitivity")
