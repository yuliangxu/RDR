"""Public compact evidence must recover the frozen choice without HPC files."""
import json
from pathlib import Path
import shutil

import pytest
from experiments.MNIST.selection_evidence import load_selected, SELECTION_SHA256


def test_published_evidence_recomputes_all_candidates():
    result = load_selected()
    assert result["selection"]["chosen"] == {"candidate": "js_a2", "loss": "js", "output_alpha": 2.}
    assert len(result["selection"]["ranking"]) == 16
    assert all(row["repeats"] == 5 for row in result["selection"]["ranking"])
    assert result["files"]["selection.sha256"].read_text().strip() == SELECTION_SHA256


@pytest.mark.parametrize("filename", ["per_fit.csv", "selection.json", "splits.json"])
def test_changed_published_evidence_is_rejected(tmp_path, filename):
    evidence = load_selected()
    for name, path in evidence["files"].items():
        shutil.copyfile(path, tmp_path / name)
    with (tmp_path / filename).open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed|differs"):
        load_selected(tmp_path)
