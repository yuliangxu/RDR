"""Scientific regression tests for label-only MNIST selection diagnostics."""

import numpy as np
import pytest

from experiments.MNIST.selection_metrics import balanced_brier, calibration_diagnostics, choose_configuration


def test_balanced_brier_uses_labels_and_source_weights():
    assert balanced_brier([2], [0, 0, 0]) == 0
    assert balanced_brier([0], [2, 2, 2]) == 1
    assert balanced_brier([1], [1, 1, 1]) == .25
    assert balanced_brier([2, 0], [0]) == .25
    with pytest.raises(ValueError, match="finite"):
        balanced_brier([np.nan], [0])


def test_unequal_counts_have_balanced_cell_masses_and_means():
    # Each source has half its mass in each occupied cell, despite nP != nQ.
    summary, cells = calibration_diagnostics(
        [1.01, 1.51], [1.02, 1.04, 1.52, 1.54],
        [1.01, 1.51], [1.02, 1.04, 1.52, 1.54])
    low, high = cells[10], cells[15]
    assert low["eval_mass"] == high["eval_mass"] == .5
    assert low["neural_mean"] == pytest.approx(1.02)
    assert high["neural_mean"] == pytest.approx(1.52)
    assert low["calibrated_rdr"] == high["calibrated_rdr"] == 1
    assert summary["local_gap"] == pytest.approx(.27)
    assert summary["maximum_gap"] == pytest.approx(.52)
    assert summary["supported_mass"] == 1
    assert summary["middle_mass"] == .5
    assert summary["middle_local_gap"] == pytest.approx(.02)


def test_empty_calibration_does_not_create_a_fake_point_estimate():
    summary, cells = calibration_diagnostics([1], [1], [0, 1], [0, 1])
    assert cells[0]["calibration_empty"]
    assert cells[0]["calibrated_rdr"] is None
    assert cells[0]["gap"] is None
    assert cells[0]["c2_lower"] == 0 and cells[0]["c2_upper"] == 2
    assert summary["supported_mass"] == summary["unsupported_mass"] == .5
    assert summary["local_gap"] == 0
    assert summary["c2_width"] >= 1
    unsupported, _ = calibration_diagnostics([1], [1], [0], [0])
    assert unsupported["local_gap"] is None
    assert unsupported["supported_mass"] == 0


def test_all_fixed_cells_and_endpoints_are_retained():
    summary, cells = calibration_diagnostics([0, 2], [0, 2], [0, 2], [0, 2])
    assert len(cells) == 20
    assert cells[0]["eval_mass"] == cells[19]["eval_mass"] == .5
    assert cells[0]["cal_count_p"] == cells[19]["cal_count_p"] == 1
    assert summary["local_gap"] == 1
    assert summary["middle_mass"] == 0
    assert summary["middle_local_gap"] is None
    with pytest.raises(ValueError, match="finite"):
        calibration_diagnostics([-1e-9], [0], [0], [0])


def test_ci_distance_and_gap_are_distinct_and_directional():
    # theta=1, neural mean=1.55; a large independent calibration sample makes
    # the cell CI narrow enough to expose a positive interval distance.
    scores = np.full(10000, 1.55)
    summary, cells = calibration_diagnostics(scores, scores, [1.55], [1.55])
    cell = cells[15]
    assert cell["calibrated_rdr"] == 1
    assert cell["gap"] == pytest.approx(.55)
    assert 0 < cell["c2_distance"] < cell["absolute_gap"]
    assert summary["c2_distance"] == cell["c2_distance"]


def selection_rows(candidate, briers, gap=.2, support=1.):
    return [dict(candidate=candidate, repeat=repeat, loss="hellinger", output_alpha=1,
                 brier=brier, local_gap=gap, supported_mass=support)
            for repeat, brier in enumerate(briers)]


def test_paired_one_se_shortlist_then_local_gap_selection():
    rows = selection_rows("reference", [.1, .2, .3], .2)
    rows += selection_rows("eligible", [.09, .22, .32], .1)
    rows += selection_rows("ineligible", [.11, .21, .31], .001)
    result = choose_configuration(rows)
    assert result["brier_reference"] == "reference"
    assert result["chosen"]["candidate"] == "eligible"
    ranking = {row["candidate"]: row for row in result["ranking"]}
    assert ranking["eligible"]["brier_eligible"]
    assert not ranking["ineligible"]["brier_eligible"]
    assert ranking["ineligible"]["paired_brier_difference_se"] == pytest.approx(0, abs=1e-16)


def test_support_requirement_applies_to_each_repeat():
    rows = selection_rows("reference", [.1, .2], .2)
    rows += selection_rows("sparse", [.1, .2], .01)
    rows[-1]["supported_mass"] = .98
    result = choose_configuration(rows)
    assert result["chosen"]["candidate"] == "reference"
    sparse = next(row for row in result["ranking"] if row["candidate"] == "sparse")
    assert sparse["mean_supported_mass"] == .99
    assert not sparse["support_eligible"]


def test_selection_rejects_incomplete_duplicates_and_nonfinite_records():
    rows = selection_rows("a", [.1, .2]) + selection_rows("b", [.1, .2])
    with pytest.raises(ValueError, match="Incomplete"):
        choose_configuration(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        choose_configuration(rows + [rows[0]])
    with pytest.raises(ValueError, match="finite"):
        choose_configuration([{**row, "local_gap": np.nan} for row in rows])
    with pytest.raises(ValueError, match="supported mass"):
        choose_configuration([{**row, "supported_mass": .98} for row in rows])


def test_deterministic_tie_break_and_single_repeat():
    rows = selection_rows("z", [.1], .1) + selection_rows("a", [.1], .1)
    rows += selection_rows("better_gap_worse_brier", [.101], 0)
    assert choose_configuration(rows)["chosen"]["candidate"] == "a"
