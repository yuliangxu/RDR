"""Report gates and aggregation semantics, without training or private inputs."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments.AGP import mask_rdr_report as report


def test_report_refuses_unverified_archive_without_creating_summary(tmp_path):
    with pytest.raises(ValueError, match="independent verification"):
        report.report(tmp_path)
    assert not (tmp_path / "summary").exists()
    (tmp_path / "verification.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="independent verification"):
        report.report(tmp_path)


def test_auc_uses_real_as_positive_and_does_not_flip_direction():
    frame = pd.DataFrame({"distribution": ["P", "P", "Q", "Q"], "rdr": [0., .1, 1.9, 2.]})
    assert report._auc(frame) == 0
    frame["rdr"] = 1.
    assert report._auc(frame) == .5


def test_aggregation_is_conditional_sample_sd_and_preserves_undefined_gap():
    frame = pd.DataFrame({"method": ["bootstrap_icfm"]*5, "study": ["final"]*5,
                          "repeat": range(5), "brier": [.1, .2, .3, .4, .5],
                          "middle_local_gap": [np.nan]*5})
    result = report._aggregate(frame)
    row = result[result.metric == "brier"].iloc[0]
    assert row["mean"] == pytest.approx(.3)
    assert row["sd"] == pytest.approx(np.std(frame.brier, ddof=1))
    assert row.repeats == 5
    assert "middle_local_gap" not in set(result.metric)
    with pytest.raises(ValueError, match="five unique"):
        report._aggregate(frame.iloc[:4])
    frame["repeat"] = [0, 0, 1, 2, 3]
    with pytest.raises(ValueError, match="five unique"):
        report._aggregate(frame)


def test_export_rejects_scores_and_changed_verification(tmp_path):
    (tmp_path / "verification.json").write_text(json.dumps({"status": "passed"}))
    (tmp_path / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="outer archive seal"):
        report.export(tmp_path, tmp_path / "public")
    (tmp_path / "COMPLETE.json").write_text(json.dumps({
        "status": "complete", "verification_sha256": report.sha(tmp_path / "verification.json"),
        "outputs_sha256": {}}))
    (tmp_path / "summary").mkdir()
    manifest = {"status": "complete", "verification_sha256": report.sha(tmp_path / "verification.json"),
                "protocol_sha256": report.sha(tmp_path / "protocol.json"),
                "files_sha256": {"scores.csv": "irrelevant"}}
    (tmp_path / "summary/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unapproved"):
        report.export(tmp_path, tmp_path / "public")
    assert not (tmp_path / "public").exists()
    manifest["verification_sha256"] = "changed"
    (tmp_path / "summary/manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="verification has changed"):
        report.export(tmp_path, tmp_path / "public")
