"""Completion and provenance boundaries for the CelebA selection report."""

import json

import pytest

from experiments.CelebA import selection_report as report


def protocol():
    return {"smoke": True, "config": {"levels": ["feature"], "branches": ["lower", "upper"],
            "losses": ["js"], "output_alphas": [2], "repeats": 2, "bins": 20},
            "data": {"roles": {"train": {"p_count": 10, "lower_count": 11, "upper_count": 12}}}}


def rows(architecture="baseline"):
    return [dict(representation="feature", branch=branch, architecture=architecture,
                 candidate=f"{architecture}_js_a2", loss="js", output_alpha=2, repeat=repeat,
                 brier=.2 + .01 * repeat, local_gap=.1, supported_mass=1., middle_mass=.4,
                 middle_supported_mass=.4, middle_supported_fraction=1., middle_local_gap=.05)
            for branch in ("lower", "upper") for repeat in range(2)]


def selection(architecture="baseline"):
    return {"selections": {f"feature/{branch}": {"chosen": {
        "candidate": f"{architecture}_js_a2", "architecture": architecture,
        "loss": "js", "output_alpha": 2}} for branch in ("lower", "upper")}}


@pytest.fixture
def no_plots(monkeypatch):
    monkeypatch.setattr(report, "_plots", lambda *args: [])


def test_partial_grid_stays_pending_without_averaging_subset(tmp_path, no_plots):
    status = report.build_report(tmp_path, protocol(), rows()[:-1])
    assert not status["complete"]
    assert status["missing_baseline_fits"] == 1
    assert not status["sensitivity_registered"]
    text = (tmp_path / "RESULTS.md").read_text()
    assert "pending; study is incomplete" in text
    assert "Unavailable (1/2)" in text
    assert "p_count=10; lower_count=11; upper_count=12" in text
    assert "retrospective" in text
    assert "conditional" in text


def test_all_required_stages_gate_completion_and_plots(tmp_path):
    pytest.importorskip("matplotlib")
    (tmp_path / "sensitivity_tasks.json").write_text(json.dumps({"tasks": rows("small")}))
    evaluations = [{**r, "whole_test_brier": .19, "evaluation_brier": r["brier"]} for r in rows()]
    status = report.build_report(tmp_path, protocol(), rows() + rows("small"),
                                 baseline_selection=selection(), freeze=selection(), evaluation_rows=evaluations)
    assert status["complete"]
    assert status["smoke"]
    for name in ("RESULTS.md", "report_status.json", "per_fit.csv", "per_candidate.csv",
                 "selection_grid_feature.png", "selection_grid_feature.pdf"):
        assert (tmp_path / name).stat().st_size > 0
    text = (tmp_path / "RESULTS.md").read_text()
    assert "plumbing only, not scientific evidence" in text
    assert "Whole-test balanced Brier | Independent evaluation Brier" in text
    assert "0.190000 (0.000000)" in text
    assert "0.205000 (0.007071)" in text


def test_partial_assessment_and_expansion_do_not_complete(tmp_path, no_plots):
    (tmp_path / "sensitivity_tasks.json").write_text(json.dumps(rows("small")[:2]))
    (tmp_path / "expanded_tasks.json").write_text(json.dumps(rows("small")[2:]))
    status = report.build_report(tmp_path, protocol(), rows() + rows("small")[:2],
                                 baseline_selection=selection(), freeze=selection(), evaluation_rows=rows()[:-1])
    assert status["sensitivity_complete"]
    assert not status["expansion_complete"]
    assert not status["final_complete"]
    assert not status["complete"]
    assert status["expected_expanded_fits"] == 2


def test_test_candidate_or_architecture_cannot_change_frozen_selection(tmp_path, no_plots):
    with pytest.raises(ValueError, match="frozen selected"):
        report.build_report(tmp_path, protocol(), rows(), freeze=selection(), evaluation_rows=rows("small"))
    bad = [{**r, "architecture": "small"} for r in rows()]
    with pytest.raises(ValueError, match="frozen selected"):
        report.build_report(tmp_path, protocol(), rows(), freeze=selection(), evaluation_rows=bad)


def test_duplicate_invalid_and_unregistered_records_fail(tmp_path, no_plots):
    with pytest.raises(ValueError, match="Duplicate"):
        report.build_report(tmp_path, protocol(), rows() + rows()[:1])
    with pytest.raises(ValueError, match="Invalid report metric"):
        report.build_report(tmp_path, protocol(), [{**r, "brier": float("nan")} for r in rows()])
    with pytest.raises(ValueError, match="outside the registered tasks"):
        report.build_report(tmp_path, protocol(), rows() + rows("small"))


def test_missing_middle_diagnostic_remains_unavailable(tmp_path, no_plots):
    values = rows()
    values[0].update(middle_mass=0., middle_supported_mass=0.,
                     middle_local_gap=None, middle_supported_fraction=None)
    report.build_report(tmp_path, protocol(), values)
    summary = report._aggregate(values, 2, "selection")[0]
    assert summary["middle_local_gap_available_repeats"] == 1
    assert summary["middle_local_gap_mean"] is None
