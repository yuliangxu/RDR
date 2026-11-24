"""Verify the published neural baseline on the frozen AGP assessment rows."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.AGP import four_generator_pcoa as assessment
from utils.model_selection import balanced_brier


def pin(path, expected):
    if assessment.sha(path) != expected:
        raise ValueError(f"Archived input checksum differs: {path}")


def neural_fits(method, root, inputs):
    """The assessment loader has already checked ordered rows and the 20 prediction files."""
    results = []
    for repeat in range(5):
        fit = root / "final" / f"{repeat:02d}"
        receipt = assessment.read(fit / "COMPLETE.json")
        path = fit / "metrics.json"
        pin(path, receipt["outputs"]["metrics.json"])
        inputs[str(path)] = receipt["outputs"]["metrics.json"]
        metrics = assessment.read(path)
        with np.load(fit / "predictions.npz", allow_pickle=False) as saved:
            score = balanced_brier(saved["test_p"], saved["test_q"])
        if (metrics["repeat"] != repeat or metrics["assessment_role"] != "test"
                or not np.isclose(score, metrics["brier"], rtol=0, atol=1e-12)):
            raise ValueError(f"Published assessment mismatch at {fit}")
        results.append(dict(method=method, repeat=repeat, seed=metrics["seed_initialization"],
                            brier=score))
    return results


def run(output):
    _, ensemble, _, inputs = assessment.load_assessment()
    rows, all_seeds = [], []
    for method, root in zip(assessment.METHODS, assessment.ROOTS):
        seeds = neural_fits(method, root, inputs)
        all_seeds.extend(seeds)
        scores = [row["brier"] for row in seeds]
        rows.append(dict(method=method, published_neural_brier=np.mean(scores),
                         published_neural_sd=np.std(scores, ddof=1),
                         published_ensemble_brier=balanced_brier(ensemble[method+"_p"],
                                                                ensemble[method+"_q"])))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "baselines.csv", index=False)
    pd.DataFrame(all_seeds).to_csv(output / "published_per_seed.csv", index=False)
    (output / "baseline_audit.json").write_text(json.dumps(
        {"published_inputs_sha256": inputs}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
