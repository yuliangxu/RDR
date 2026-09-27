"""Matched-loss reference integrity, paired streams, and executable reporting."""

import copy
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from experiments.simulations import loss_followup as f
from experiments.simulations import workflow as w


@pytest.fixture
def prepared(tmp_path):
    torch.set_num_threads(1)
    cfg = w.read_json(w.DEFAULT_CONFIG)
    cfg.update(cases=["D2_noise0.00"], selection_repeats=2, convergence_repeats=1,
               losses=["hellinger"], architectures=["bottleneck8_residual64"], output_alphas=[2],
               train_n=16, validation_n=16, calibration_n=24, evaluation_n=24,
               sample_sizes=[8,16], max_epochs=2, min_epochs=1, patience=1,
               bootstrap_repetitions=20, min_cell_count=2)
    original, protocol = w.prepare(tmp_path/"original", cfg, "selection")
    for index in range(len(w.task_specs(protocol))):
        w.run_task(original, protocol, index)
    w.freeze(original)
    followup = f.prepare(original, tmp_path/"followup")
    return original, followup


def test_followup_reuses_reference_and_matches_every_stream(prepared):
    original, followup = prepared
    source_before = {str(p.relative_to(original)): w.digest(p) for p in original.rglob("*") if p.is_file()}
    protocol = w.read_json(followup/"protocol.json")
    assert [c["loss"] for c in protocol["candidates"]] == ["kl", "chisq", "js"]
    assert protocol["config"] == w.read_json(original/"protocol.json")["config"]
    for index in range(2):
        f.fit(followup, index)
    _, rows, _ = f.verified_results(followup)
    assert len(rows) == 8
    assert sum(r["origin"] == "reused_Hellinger" for r in rows) == 2
    for case, repeat, n in w.task_specs(protocol):
        old_folder = w.task_path(followup/"reference", case, repeat, n, protocol["reference_candidate"])
        training = w.read_json(old_folder/"training.json")
        expected_inputs = w.samples(protocol["config"], "selection", case, repeat, n)[1]
        for candidate in protocol["candidates"]:
            folder = w.task_path(followup, case, repeat, n, candidate)
            current = w.read_json(folder/"training.json")
            assert current["data_seed_vectors"] == training["data_seed_vectors"]
            assert current["model_seed"] == training["model_seed"]
            assert current["protocol_sha256"] != training["protocol_sha256"]
            original_inputs = w.samples(w.read_json(original/"protocol.json")["config"], "selection", case, repeat, n)[1]
            assert all(np.array_equal(original_inputs[key], expected_inputs[key]) for key in expected_inputs)
    assert f.report(followup).is_file()
    marker = w.read_json(followup/f.COMPLETE)
    assert marker["new_fits"] == 6 and marker["reused_reference_fits"] == 2
    assert f.report(followup).is_file()
    assert source_before == {str(p.relative_to(original)): w.digest(p) for p in original.rglob("*") if p.is_file()}
    assert not (followup/"selection.json").exists()


def test_incomplete_followup_cannot_report_or_mark_complete(prepared):
    _, output = prepared
    f.fit(output, 0)
    with pytest.raises(ValueError, match="Incomplete task"):
        f.report(output)
    assert not (output/f.COMPLETE).exists()
    assert not (output/"report.md").exists()


@pytest.mark.parametrize("target", ["fit", "selection", "source", "manifest"])
def test_reference_tamper_is_rejected_before_fitting(prepared, target):
    _, output = prepared
    protocol = w.read_json(output/"protocol.json")
    if target == "fit":
        case, repeat, n = w.task_specs(protocol)[0]
        path = w.task_path(output/"reference", case, repeat, n, protocol["reference_candidate"])/"metrics.json"
    elif target == "selection":
        path = output/"reference"/"selection.json"
    elif target == "source":
        path = output/"reference"/"source"/"utils"/"losses.py"
    else:
        path = output/"reference_manifest.json"
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError):
        f.fit(output, 0)
    assert not (output/"fits").exists()


def test_self_contained_snapshot_runs_without_original(prepared, tmp_path):
    original, output = prepared
    # The original can be unavailable: only the copied, pinned reference is used.
    renamed = original.with_name("original_moved")
    original.rename(renamed)
    script = output/"source"/"experiments"/"simulations"/"loss_followup.py"
    subprocess.run([sys.executable,"-B",str(script),"fit","--output",str(output),"--task-index","0"],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    assert len(list((output/"fits").glob("*/repeat000/n16/*/complete.json"))) == 3


def test_followup_cannot_change_the_matched_architecture(prepared):
    _, output = prepared
    protocol = w.read_json(output/"protocol.json")
    protocol["candidates"][0]["architecture"] = "wide64"
    (output/"protocol.json").write_text(__import__("json").dumps(protocol))
    with pytest.raises(ValueError, match="Reference protocol/candidate"):
        f.fit(output, 0)
    assert not (output/"fits").exists()


def test_paired_se_uses_within_case_differences():
    rows=[]
    for case, baseline in (("D2_noise0.00",0.),("D20_noise0.10",100.)):
        for repeat, difference in enumerate((1.,3.)):
            for loss in ("hellinger","kl","chisq","js"):
                row={"case":case,"repeat":repeat,"loss":loss}
                row.update({metric: baseline+difference if loss=="hellinger" else baseline for metric in f.paper.METRICS})
                rows.append(row)
    result=f.paired_results(rows,["D2_noise0.00","D20_noise0.10"])
    assert all(r["evaluation_mse_mean"] == pytest.approx(2.) for r in result)
    assert all(r["evaluation_mse_se"] == pytest.approx(1/np.sqrt(2)) for r in result)


def test_archived_render_accepts_updated_presentation_and_rejects_raw_tamper(prepared, tmp_path):
    """Presentation evolves independently; archived fit verification stays strict."""
    _, archive = prepared
    for index in range(2):
        f.fit(archive, index)
    f.report(archive)
    before = {str(p.relative_to(archive)): w.digest(p) for p in archive.rglob("*") if p.is_file()}

    current = tmp_path / "updated-presentation"
    shutil.copytree(archive / "source", current)
    helper = current / "experiments/simulations/report.py"
    helper.write_text(helper.read_text() + "\n# A later presentation-only revision.\n")
    script = current / "experiments/simulations/loss_followup.py"
    strict = subprocess.run([sys.executable, "-B", str(script), "report", "--output", str(archive)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert strict.returncode != 0 and "Executing source differs" in strict.stderr

    output = tmp_path / "rendered"
    command = [sys.executable, "-B", str(script), "render", "--reference", str(archive), "--output"]
    subprocess.run([*command, str(output)], cwd=tmp_path, check=True, capture_output=True, text=True)
    manifest = w.read_json(output / "rendering_manifest.json")
    assert manifest["role"] == "presentation_only" and manifest["verified_fits"] == 8
    assert not (output / f.COMPLETE).exists()
    assert "Definition of the local gap" in (output / "report.md").read_text()
    assert w.digest(output / "aggregation.csv") == w.digest(archive / "aggregation.csv")
    assert manifest["rendering_source_sha256"]["experiments/simulations/report.py"] == w.digest(helper)
    assert any(name.startswith("fits/") and name.endswith("predictions.npz")
               for name in manifest["consumed_sha256"])
    assert before == {str(p.relative_to(archive)): w.digest(p) for p in archive.rglob("*") if p.is_file()}

    damaged = next((archive / "fits").glob("*/repeat000/n16/*/metrics.json"))
    damaged.write_text(damaged.read_text() + " ")
    rejected = tmp_path / "rejected"
    result = subprocess.run([*command, str(rejected)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0 and "Archived fit verification failed" in result.stderr
    assert not rejected.exists()
