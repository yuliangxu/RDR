"""Scientific role separation and artifact gates for selected AGP experiments."""

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.AGP import selection_data as data
from experiments.AGP import selected_workflow as workflow
from utils.losses import midpoint_loss_from_logits


@pytest.fixture
def config():
    cfg = json.loads(Path(workflow.__file__).with_name("selection_config.json").read_text())
    cfg.update(output_alphas=[.5, 1.], repeats=2, hidden_dimension=5, blocks=1,
               min_epochs=3, max_epochs=3, patience=5, threads=1)
    return workflow.validate_config(cfg)


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(previous)


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    """Repeated real subjects make accidental rowwise allocations detectable."""
    source = tmp_path / "inputs"
    source.mkdir()
    taxonomy = source / "yx1_abundance.csv"
    taxonomy.write_text("taxon,sample\n" + "".join(
        f"k__Bacteria;p__Phylum{i};g__Genus{i},1\n" for i in range(4)))
    rng = np.random.default_rng(141)
    arrays, records = {}, []
    sizes = {"P": {"train": 24, "validation": 8, "test": 8},
             "Q": {"train": 20, "validation": 8, "test": 8}}
    for side, roles in sizes.items():
        for role, size in roles.items():
            values = rng.gamma(2, 1, (size, 4)).astype(np.float32)
            arrays[side, role] = values
            for index, digest in enumerate(data._row_hashes(values)):
                records.append({"sample_id": f"{side}-{role}-{index}",
                    "subject_id": f"real-{role}-{index // 2}" if side == "P" else "",
                    "distribution": side, "role": role,
                    "source_file": f"{side}-{role}.csv", "source_row": index,
                    "row_sha256": digest})
    for index in range(2):
        records.append({"sample_id": f"excluded-{index}", "subject_id": "excluded",
            "distribution": "P", "role": "excluded", "source_file": "P-excluded.csv",
            "source_row": index, "row_sha256": "excluded-not-loaded"})
    manifest = pd.DataFrame(records)
    audit = {"feature_dimension": 4,
             "input_sha256": {taxonomy.name: data._file_sha256(taxonomy)},
             "limitations": ["Synthetic test fixture; repeated real subjects."]}
    monkeypatch.setattr(data, "EXPECTED_EXCLUSIONS", 2)
    monkeypatch.setattr(data, "prepare_712_data", lambda *args, **kwargs:
                        (arrays, manifest.copy(), audit))
    return source, arrays, manifest


def test_prepared_roles_preserve_all_rows_and_keep_real_subjects_disjoint(synthetic_source, tmp_path):
    source, original_arrays, original = synthetic_source
    output = tmp_path / "study"
    cfg = {"split_seed": 41, "null_split_seed": 47}
    protocol = data.prepare(output, source, cfg)
    assert data.prepare(output, source, cfg) == protocol
    final_test = set(original.loc[original.role.eq("test"), "sample_id"])
    for study, entry in protocol["studies"].items():
        manifest = pd.read_csv(output / entry["manifest"]["path"], keep_default_na=False)
        assert not manifest.sample_id.duplicated().any()
        real = manifest.loc[manifest.original_distribution.eq("P")]
        assignments = real.role + ":" + real.distribution
        assert assignments.groupby(real.subject_id).nunique().eq(1).all()
        if study == "selection":
            expected = original.loc[original.role.isin(["train", "validation"]), "sample_id"]
            assert set(manifest.sample_id) == set(expected)
            assert not set(manifest.sample_id) & final_test
            assert not any("test" in key for key in entry["arrays"])
        else:
            expected = original.loc[~original.role.eq("excluded")]
            if study == "null":
                expected = expected.loc[expected.distribution.eq("P")]
                assert manifest.original_distribution.eq("P").all()
            assert set(manifest.sample_id) == set(expected.sample_id)
            assert set(manifest.loc[manifest.role.eq("test"), "sample_id"]) == set(
                expected.loc[expected.role.eq("test"), "sample_id"])
        for key, receipt in entry["arrays"].items():
            role, side = key.rsplit("_", 1)
            rows = manifest.loc[manifest.role.eq(role) & manifest.distribution.eq(side.upper())]
            values = np.load(output / receipt["path"], allow_pickle=False)
            assert values.dtype == np.float32
            assert rows.array_row.tolist() == list(range(len(rows)))
            assert data._row_hashes(values) == rows.row_sha256.tolist()
            assert entry["role_counts"][role][side] == len(rows)
    # Final refits retain the historical training/stopping allocation exactly.
    final = protocol["studies"]["final"]
    for original_role, final_role in (("train", "train"), ("validation", "stopping"), ("test", "test")):
        for side in ("P", "Q"):
            saved = output / final["arrays"][f"{final_role}_{side.lower()}"]["path"]
            np.testing.assert_array_equal(np.load(saved), original_arrays[side, original_role])


def test_prepared_data_reject_changed_splits_and_tampered_array(synthetic_source, tmp_path):
    source, _, _ = synthetic_source
    output = tmp_path / "study"
    cfg = {"split_seed": 41, "null_split_seed": 47}
    protocol = data.prepare(output, source, cfg)
    with pytest.raises(RuntimeError, match="split settings changed"):
        data.prepare(output, source, {**cfg, "split_seed": 42})
    receipt = protocol["studies"]["selection"]["arrays"]["train_p"]
    path = output / receipt["path"]
    changed = np.load(path)
    changed[0, 0] += 1
    np.save(path, changed)
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        data.prepare(output, source, cfg)


def test_partition_audit_rejects_subject_leakage_despite_unique_samples(synthetic_source):
    _, _, original = synthetic_source
    frame = original.loc[original.role.isin(["train", "validation", "test"])].copy()
    frame["original_distribution"] = frame.distribution
    train_subject = frame.loc[frame.distribution.eq("P") & frame.role.eq("train"), "subject_id"].iloc[0]
    first_test = frame.index[frame.distribution.eq("P") & frame.role.eq("test")][0]
    frame.loc[first_test, "subject_id"] = train_subject
    with pytest.raises(AssertionError, match="subject crosses"):
        data._assert_partition(frame, ("train", "validation", "test"), "fixture")


@pytest.fixture
def prepared_run(synthetic_source, tmp_path, config):
    source, _, _ = synthetic_source
    output = tmp_path / "prepared"
    data.prepare(output, source, config)
    workflow.write_json(output / "config.json", config)
    files = {}
    for relative in ("experiments/AGP/selected_workflow.py", "experiments/AGP/selection_data.py",
                     "utils/losses.py", "utils/model_selection.py", "utils/calibration.py"):
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workflow.REPO / relative, target)
        files[relative] = workflow.sha256(target)
    workflow.write_json(output / "protocol.json", {
        "config_sha256": workflow.sha256(output / "config.json"),
        "data_protocol_sha256": workflow.sha256(output / "data/protocol.json"),
        "source_sha256": files,
    })
    (output / "selection").mkdir()
    return output


def test_production_grid_and_initializations_are_paired(config):
    production = json.loads(Path(workflow.__file__).with_name("selection_config.json").read_text())
    tasks = workflow.selection_tasks(workflow.validate_config(production))
    assert len(tasks) == 80
    assert len({task["candidate"] for task in tasks}) == 16
    for candidate in {task["candidate"] for task in tasks}:
        assert sorted(task["repeat"] for task in tasks if task["candidate"] == candidate) == list(range(5))
    before = torch.get_rng_state().clone()
    first = workflow.make_model(4, config, 0, "selection")
    second = workflow.make_model(4, config, 0, "selection")
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    for key, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], rtol=0, atol=0)
    assert isinstance(first.model[-1], torch.nn.Linear)
    seeds = [workflow.seed_for(production, repeat, stage)
             for stage in ("selection", "final", "null") for repeat in range(5)]
    assert len(set(seeds)) == len(seeds)
    final = workflow.make_model(4, config, 0, "final")
    assert any(not torch.equal(value, final.state_dict()[key]) for key, value in first.state_dict().items())


def test_ilr_uses_balanced_training_moments_and_is_invariant_to_row_totals(config):
    p = np.array([[8., 1., 0., 1.], [4., 1., 1., 2.], [9., 1., 0., 0.]])
    q = np.array([[0., 2., 7., 1.], [1., 1., 4., 4.]])
    epsilon = config["epsilon"]

    def clr(values):
        relative = values / values.sum(axis=1, keepdims=True)
        logarithms = np.log((relative + epsilon) / (1 + values.shape[1] * epsilon))
        return logarithms - logarithms.mean(axis=1, keepdims=True)

    cp, cq = clr(p), clr(q)
    expected_center = .5 * (cp.mean(axis=0) + cq.mean(axis=0))
    expected_scale = np.sqrt(.5 * (np.square(cp-expected_center).mean() + np.square(cq-expected_center).mean()))
    state = workflow.fit_ilr(p, q, epsilon)
    np.testing.assert_allclose(state["center"], expected_center, rtol=0, atol=1e-12)
    assert state["scale"] == pytest.approx(expected_scale, abs=1e-12)
    np.testing.assert_allclose(state["basis"] @ state["basis"].T, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(workflow.transform_ilr(p, state),
                               workflow.transform_ilr(p * np.array([1., 100., .01])[:, None], state),
                               rtol=0, atol=1e-6)
    assert not np.allclose(expected_center, np.concatenate([cp, cq]).mean(axis=0))


@pytest.mark.parametrize("loss", ["hellinger", "kl", "chisq", "js"])
def test_local_scaled_gradients_match_float64_objective_at_extreme_logits(loss):
    values = [-400., 1.2, -.8, 4., -6.]
    reference = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    risk = midpoint_loss_from_logits(reference[:2], reference[2:], loss)
    risk.backward()
    expected = reference.grad * min(1., 1 / float(reference.grad.norm()))
    actual = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
    calculated, backward, scale = workflow.stable_risk(actual, 2, loss)
    backward.backward()
    workflow.clip_scaled_gradients([actual], scale, 1.)
    torch.testing.assert_close(calculated, risk, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(actual.grad.double(), expected, rtol=3e-6, atol=2e-7)
    assert float(actual.grad.double().norm()) <= 1.000001


def test_fit_restores_absolute_minimum_brier_and_ignores_other_roles(config, monkeypatch):
    rng = np.random.default_rng(35)
    raw = {key: rng.gamma(2, 1, (n, 4)).astype(np.float32)
           for key, n in (("train_p", 13), ("train_q", 7), ("stopping_p", 5), ("stopping_q", 3))}

    class Forbidden:
        def __array__(self, *args, **kwargs):
            pytest.fail("Fitting attempted to inspect a non-training/stopping role")

    raw["test_p"] = Forbidden()
    raw["calibration_q"] = Forbidden()
    states, calls = [], {"number": 0}

    def controlled_stopping(model, x, alpha):
        call = calls["number"]
        calls["number"] += 1
        epoch, source = divmod(call, 2)
        if source == 0:
            states.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
        error = [.4, .2, .6, .2][epoch]
        return np.full(len(x), 2-error if source == 0 else error)

    monkeypatch.setattr(workflow, "_tensor_scores", controlled_stopping)
    task = {"candidate": "js_a1", "loss": "js", "output_alpha": 1., "repeat": 0}
    model, state, history, fitted = workflow.fit_model(raw, config, task, "selection", "cpu")
    assert fitted["best_epoch"] == 2 and fitted["epochs_run"] == 3
    assert fitted["stopping_brier"] == pytest.approx(.01)
    assert fitted["restored_stopping_brier"] == pytest.approx(.01)
    assert fitted["train_p"] == 13 and fitted["train_q"] == 7
    np.testing.assert_allclose(history.stopping_brier, [.04, .01, .09])
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, states[1][key], atol=0, rtol=0)
        torch.testing.assert_close(states[-1][key], states[1][key], atol=0, rtol=0)
    assert any(not torch.equal(value, states[2][key]) for key, value in states[1].items())
    expected = workflow.fit_ilr(raw["train_p"], raw["train_q"], config["epsilon"])
    np.testing.assert_array_equal(state["center"], expected["center"])


@pytest.mark.parametrize("stage", ["final", "null"])
def test_assessment_refuses_before_selection_without_loading_arrays(prepared_run, monkeypatch, stage):
    def forbidden(*args, **kwargs):
        pytest.fail("Assessment data or fitting accessed before selection was frozen")

    monkeypatch.setattr(workflow, "load_arrays", forbidden)
    monkeypatch.setattr(workflow, "fit_model", forbidden)
    with pytest.raises(FileNotFoundError):
        workflow.run_task(prepared_run, stage, 0)
    assert not (prepared_run / stage).exists()


def test_freeze_refuses_any_missing_candidate_repeat(prepared_run):
    with pytest.raises(FileNotFoundError):
        workflow.freeze_selection(prepared_run)
    assert not (prepared_run / "selection/freeze.json").exists()


def test_selective_array_loading_does_not_open_other_roles(prepared_run, monkeypatch):
    original = np.load
    opened = []

    def inspect(path, *args, **kwargs):
        opened.append(Path(path))
        assert Path(path).name in ("train_p.npy", "train_q.npy", "stopping_p.npy", "stopping_q.npy")
        assert Path(path).parent.name == "selection"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", inspect)
    arrays = workflow.load_arrays(prepared_run, "selection", ("train", "stopping"))
    assert set(arrays) == {"train_p", "train_q", "stopping_p", "stopping_q"}
    assert len(opened) == 4


def test_selection_fit_completes_before_diagnostics_and_protects_saved_metrics(prepared_run, config, monkeypatch):
    original = workflow.load_arrays
    task = workflow.selection_tasks(config)[0]
    directory = prepared_run / "selection/runs" / task["candidate"] / "00"
    calls = []

    def guarded_load(output, study, roles):
        calls.append((study, roles))
        assert study == "selection" and "test" not in roles
        if "calibration" in roles:
            assert (directory / "model.pt").exists()
            assert (directory / "history.csv").exists()
        return original(output, study, roles)

    monkeypatch.setattr(workflow, "load_arrays", guarded_load)
    metrics = workflow.run_task(prepared_run, "selection", 0)
    assert calls == [("selection", ("train", "stopping")),
                     ("selection", ("calibration", "evaluation"))]
    marker = workflow.verify_expected_task(prepared_run, directory, "selection", task)
    assert marker["task"] == task
    assert metrics["same_sample_calibration_evaluation"] is False
    predictions = np.load(directory / "predictions.npz")
    expected = .5*np.mean((predictions["evaluation_p"]/2-1)**2) + .5*np.mean((predictions["evaluation_q"]/2)**2)
    assert metrics["brier"] == pytest.approx(expected, abs=1e-14)
    scores = pd.read_csv(directory / "scores.csv")
    assert set(scores.role) == {"calibration", "evaluation"}
    assert not scores.sample_id.duplicated().any()
    assert workflow.run_task(prepared_run, "selection", 0) == metrics
    assert len(calls) == 2
    workflow.write_json(directory / "metrics.json", {**metrics, "brier": 0.})
    with pytest.raises(ValueError, match="Changed completed artifact"):
        workflow.run_task(prepared_run, "selection", 0)


def _fake_completion(output, directory, task, stage="selection", **metrics):
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("model.pt", "transform.npz", "history.csv", "cells.csv", "predictions.npz", "scores.csv"):
        (directory / name).write_text("Synthetic receipt fixture; never used for model evaluation.\n")
    workflow.write_json(directory / "metrics.json", {**task, **metrics})
    metadata = {"task": task, "stage": stage,
                "config_sha256": workflow.sha256(output / "config.json"),
                "protocol_sha256": workflow.sha256(output / "protocol.json")}
    if stage != "selection":
        metadata["selection_freeze_sha256"] = workflow.sha256(output / "selection/freeze.json")
    workflow.complete_task(directory, metadata)


@pytest.fixture
def frozen_grid(prepared_run, config):
    for task in workflow.selection_tasks(config):
        directory = prepared_run / "selection/runs" / task["candidate"] / f"{task['repeat']:02d}"
        # JS has slightly worse mean Brier, but is within paired one-SE and
        # has the smaller Gap. This distinguishes joint selection from Brier-only.
        local_gap = .01 if task["candidate"] == "js_a1" else .3
        brier = [.095, .107][task["repeat"]] if task["candidate"] == "js_a1" else .1
        _fake_completion(prepared_run, directory, task, brier=brier,
                         local_gap=local_gap, supported_mass=1.)
    selected = workflow.freeze_selection(prepared_run)
    assert selected["chosen"]["candidate"] == "js_a1"
    return prepared_run, selected


def test_complete_grid_uses_joint_rule_and_freeze_detects_changed_receipts(frozen_grid):
    output, selected = frozen_grid
    assert workflow.checked_selection(output) == selected["chosen"]
    assert workflow.freeze_selection(output) == selected
    lock = workflow.read_json(output / "selection/freeze.json")
    assert lock["test_scores_read"] is False
    assert lock["expected_fits"] == lock["completed_fits"] == 16
    path = output / next(iter(lock["selection_completions"]))
    changed = workflow.read_json(path)
    changed["finished_utc"] = "changed"
    workflow.write_json(path, changed)
    with pytest.raises(ValueError, match="Selection completion changed"):
        workflow.checked_selection(output)


def test_final_receipt_must_match_selected_recipe_repeat_and_lock(frozen_grid):
    output, selected = frozen_grid
    task = {**selected["chosen"], "repeat": 0}
    directory = output / "final/00"
    _fake_completion(output, directory, task, stage="final", brier=.1)
    workflow.verify_expected_task(output, directory, "final", task)
    with pytest.raises(ValueError, match="identity or provenance"):
        workflow.verify_expected_task(output, directory, "final", {**task, "repeat": 1})
    with pytest.raises(ValueError, match="identity or provenance"):
        workflow.verify_expected_task(output, directory, "null", task)
    marker = workflow.read_json(directory / "COMPLETE.json")
    marker["selection_freeze_sha256"] = "wrong-lock"
    workflow.write_json(directory / "COMPLETE.json", marker)
    with pytest.raises(ValueError, match="does not match frozen selection"):
        workflow.verify_expected_task(output, directory, "final", task)


def test_empty_completion_and_out_of_grid_indices_are_rejected(prepared_run):
    fake = prepared_run / "fake"
    workflow.write_json(fake / "COMPLETE.json", {"status": "complete", "outputs": {}})
    with pytest.raises(ValueError, match="Incomplete artifact manifest"):
        workflow.verify_task(fake)
    for index in (-1, 999):
        with pytest.raises(ValueError, match="outside declared grid"):
            workflow.run_task(prepared_run, "selection", index)


def test_changed_configuration_snapshot_and_executing_source_are_rejected(prepared_run, monkeypatch, tmp_path):
    workflow.verify_snapshot(prepared_run)
    config_path = prepared_run / "config.json"
    original = config_path.read_bytes()
    cfg = workflow.read_json(config_path)
    workflow.write_json(config_path, {**cfg, "seed": cfg["seed"] + 1})
    with pytest.raises(ValueError, match="Configuration changed"):
        workflow.verify_snapshot(prepared_run)
    config_path.write_bytes(original)
    other = tmp_path / "changed_workflow.py"
    other.write_text("# Different executing workflow\n")
    actual_workflow = workflow.__file__
    monkeypatch.setattr(workflow, "__file__", str(other))
    with pytest.raises(ValueError, match="Executing workflow differs"):
        workflow.verify_snapshot(prepared_run)
    monkeypatch.setattr(workflow, "__file__", actual_workflow)
    losses = workflow.sys.modules["utils.losses"]
    actual_losses = losses.__file__
    monkeypatch.setattr(losses, "__file__", str(other))
    with pytest.raises(ValueError, match="Executing utility differs"):
        workflow.verify_snapshot(prepared_run)
    monkeypatch.setattr(losses, "__file__", actual_losses)
    frozen = prepared_run / "source/utils/losses.py"
    frozen.write_text(frozen.read_text() + "\n# Changed frozen source\n")
    with pytest.raises(ValueError, match="Frozen source changed"):
        workflow.verify_snapshot(prepared_run)
