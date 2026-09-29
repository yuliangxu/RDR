"""Verify the compact, published DCGAN selection evidence for fresh refits.

Full-study checkpoint audits remain in the frozen selection package. This
loader independently reapplies the registered rule to all 80 published rows
and checks the original selection/protocol locks, without private HPC paths.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from experiments.MNIST.selection_metrics import choose_configuration

ROOT = Path(__file__).resolve().parents[2]
SELECTION_SHA256 = "48807a984a9ad3c9dd3238c3be4465cbf3b486b1c0646354fc8bd9c01681ff60"
PROTOCOL_SHA256 = "dd473e1282933bd104cc7cb75dd2db71bfa37bebc3d541da6df3d11b65d13b0c"
REQUIRED = ("selection.json", "selection.sha256", "protocol.json", "protocol.sha256",
            "per_fit.csv", "splits.json", "manifest.json")


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selected(evidence_root=None):
    root = Path(evidence_root) if evidence_root is not None else ROOT / "results/MNIST/model_selection"
    files = {name: root / name for name in REQUIRED}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    for stem, expected in (("selection", SELECTION_SHA256), ("protocol", PROTOCOL_SHA256)):
        if _sha(files[stem + ".json"]) != expected or files[stem + ".sha256"].read_text().strip() != expected:
            raise ValueError(f"Published {stem} differs from the frozen selected study")
    manifest = json.loads(files["manifest.json"].read_text())
    for name, path in files.items():
        if name != "manifest.json" and _sha(path) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Published evidence changed: {name}")
    protocol = json.loads(files["protocol.json"].read_text())
    selection = json.loads(files["selection.json"].read_text())
    if protocol["smoke"] or selection["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("Expected the completed full DCGAN selection study")
    with files["per_fit.csv"].open(newline="") as stream:
        rows = []
        for row in csv.DictReader(stream):
            if row["stage"] != "selection":
                continue
            row["repeat"] = int(row["repeat"])
            for key in ("output_alpha", "brier", "local_gap", "supported_mass"):
                row[key] = float(row[key])
            rows.append(row)
    identity = lambda row: (row["candidate"], row["repeat"], row["loss"], row["output_alpha"])
    if len(rows) != len(protocol["tasks"]) or {identity(row) for row in rows} != {identity(row) for row in protocol["tasks"]}:
        raise ValueError("Published selection rows do not match the complete registered grid")
    recomputed = choose_configuration(rows)
    for key, value in recomputed.items():
        if selection[key] != value:
            raise ValueError(f"Published {key} does not follow the registered selection rule")
    return {"selection": selection, "config": protocol["config"], "files": files}
