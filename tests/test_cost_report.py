"""Timing-report aggregation, completeness and immutable combined presentation."""

import csv
import math
from pathlib import Path
import re

import pytest
from experiments.simulations import cost_report as cr
from experiments.simulations import workflow as w


@pytest.fixture
def timing_design(tmp_path):
    original = tmp_path / "scientific"
    original.mkdir()
    cfg = {"cases": ["D2_noise0.00"], "sample_sizes": [100, 200], "convergence_repeats": 100}
    reference_protocol = {"stage": "convergence", "config": cfg}
    w.save_json(original / "protocol.json", reference_protocol)
    (original / "report.md").write_text("# Scientific results\n\n[Appendix](appendix.md)\n")
    (original / "report.html").write_text("<html><main><h1>Scientific results</h1></main></html>")
    (original / "tables.tex").write_text("% Scientific tables\n")
    (original / "appendix.md").write_text("# Scientific appendix\n")
    original_artifacts = {name: w.digest(original / name) for name in
                          ("report.md", "report.html", "tables.tex", "appendix.md")}
    w.save_json(original / "CONVERGENCE_COMPLETE.json", {
        "status": "complete", "protocol_sha256": w.digest(original / "protocol.json"),
        "artifact_sha256": original_artifacts,
    })
    output = tmp_path / "cost"
    output.mkdir()
    candidates = [{"loss": loss, "architecture": "bottleneck8_residual64", "output_alpha": 2}
                  for loss in cr.LOSSES]
    protocol = {"stage": "computation_cost", "config": {**cfg, "convergence_repeats": 2},
                "reference_path": str(original), "reference_protocol": reference_protocol,
                "reference_protocol_sha256": w.digest(original / "protocol.json"),
                "candidates": candidates, "repeats": 2, "threads": 1, "device": "cpu",
                "versions": {"torch": "test"}}
    w.save_json(output / "protocol.json", protocol)
    w.save_json(output / "submission.json", {"array_concurrency": 4})
    rows = []
    for repeat in range(2):
        for n in cfg["sample_sizes"]:
            for index, candidate in enumerate(candidates):
                wall = 1 + 2*repeat + n/100 + index
                rows.append({"case": cfg["cases"][0], "repeat": repeat, "train_n": n,
                             "candidate": w.candidate_id(candidate), **candidate,
                             "wall_seconds": wall, "cpu_seconds": wall - .1,
                             "epochs": 300 + repeat*10, "best_epoch": 280,
                             "n_parameters": 1234,
                             "hardware": {"node": "test-node", "cpu_model": "test-cpu",
                                          "slurm": {"SLURM_ARRAY_TASK_ID": str(len(rows))}}})
    return original, output, protocol, rows


def test_sample_sd_is_not_monte_carlo_standard_error(timing_design):
    _, _, protocol, rows = timing_design
    checked, repeats = cr.checked_rows(protocol, rows)
    summary = cr.summarize(checked, protocol["config"]["cases"], protocol["config"]["sample_sizes"])
    first = summary[0]
    assert repeats == [0, 1]
    assert first["wall_seconds_mean"] == 3
    assert first["wall_seconds_sd"] == pytest.approx(math.sqrt(2))
    assert first["timing_repeats"] == 2
    assert len(summary) == 8


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "nonfinite", "candidate"])
def test_invalid_timing_grid_cannot_publish(timing_design, mutation):
    _, output, protocol, rows = timing_design
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "nonfinite":
        rows[0]["cpu_seconds"] = float("nan")
    else:
        rows[0]["loss"] = "js"
    with pytest.raises(ValueError):
        cr.render(output, protocol, rows)
    assert not (output / "report.md").exists()
    assert not (output / "figures").exists()


def test_modified_scientific_report_cannot_publish(timing_design):
    original, output, protocol, rows = timing_design
    (original / "report.md").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        cr.render(output, protocol, rows)
    assert not (output / "report.md").exists()


def test_combined_report_preserves_originals_and_has_local_links(timing_design):
    original, output, protocol, rows = timing_design
    before = {str(path.relative_to(original)): w.digest(path) for path in original.rglob("*") if path.is_file()}
    artifacts = cr.render(output, protocol, rows)
    after = {str(path.relative_to(original)): w.digest(path) for path in original.rglob("*") if path.is_file()}
    assert before == after
    assert all(w.digest(output / name) == digest for name, digest in artifacts.items())
    with (output / "cost_summary.csv").open() as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 8
    assert float(summary[0]["wall_seconds_sd"]) == pytest.approx(math.sqrt(2))
    report = (output / "report.md").read_text()
    assert "sample standard deviation" in report
    assert "100 repetitions" in report
    assert "only 2 prespecified timing repetitions" in report
    assert "at most 4 concurrent benchmark workers" in report
    assert report.count("test-node: test-cpu") == 1  # Aggregate task-specific hardware records.
    combined = output / "convergence_with_cost"
    assert (combined / "report.md").read_text().startswith((original / "report.md").read_text())
    assert (combined / "appendix.md").read_bytes() == (original / "appendix.md").read_bytes()
    assert "Average computation cost" in (combined / "report.html").read_text()
    for name in ("report.md", "cost_report.md", "cost_appendix.md", "convergence_with_cost/report.md"):
        path = output / name
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            assert (path.parent / target).is_file(), (name, target)
    with pytest.raises(FileExistsError):
        cr.render(output, protocol, rows)
