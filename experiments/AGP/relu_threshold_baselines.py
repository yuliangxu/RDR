"""Audit the published neural and composition-tree baselines on the frozen AGP rows."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

from experiments.AGP import four_generator_pcoa as assessment
from experiments.AGP import mask_metrics
from utils.model_selection import balanced_brier


TREE_RUN = assessment.BASE / "mask_comparison_20261012"
PUBLIC = Path(__file__).resolve().parents[2] / "results/AGP/retained_mask_comparison_20261013"
RECIPE = "hgb_composition"


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


def tree_score(method, arrays, identities):
    """Match ordered inputs, then reproduce the archived equal-prior classifier loss."""
    directory = TREE_RUN / "metrics"
    release = assessment.read(TREE_RUN / "COMPLETE.json")
    receipt_path = TREE_RUN / "assessment_complete.json"
    pin(receipt_path, release["outputs_sha256"]["assessment_complete.json"])
    artifacts = assessment.read(receipt_path)["outputs_sha256"]
    pin(directory / "COMPLETE.json", artifacts["COMPLETE.json"])
    receipt = assessment.read(directory / "COMPLETE.json")
    pin(directory / "summary/classifier_metrics.csv", receipt["summary_hashes"]["classifier_metrics.csv"])
    summary = pd.read_csv(directory / "summary/classifier_metrics.csv")
    reference = summary[(summary.method == method) & (summary.recipe == RECIPE)].iloc[0]
    public = pd.read_csv(PUBLIC / "classifier_metrics.csv")
    public_loss = public.loc[(public.method == method) & (public.recipe == RECIPE),
                             "balanced_log_loss"].item()
    if not np.isclose(public_loss, reference.balanced_log_loss, atol=1e-12, rtol=0):
        raise ValueError(f"Published tree loss differs from the original archive for {method}")

    p, q = arrays["real"], arrays[method]
    input_hashes = receipt["test_input_hashes"]
    if mask_metrics.array_hash(p) != input_hashes["real"]:
        raise ValueError("Real test rows differ between the tree and neural fits")
    alignment = "identical array hashes for the two sources"
    if mask_metrics.array_hash(q) != input_hashes[method]:
        # mask_rdr.prepare wrote exact float32 casts of the two mask-sampler banks.
        source = TREE_RUN / "generated" / method / "test.npy"
        identity = identities[method]  # sample, subject, source file, source row, row hash
        original = np.load(source, allow_pickle=False)
        if (not np.all(identity[:, 2] == str(source))
                or not np.array_equal(identity[:, 3].astype(int), np.arange(len(q)))
                or mask_metrics.array_hash(original) != input_hashes[method]
                or not np.array_equal(original.astype(np.float32), q)):
            raise ValueError(f"Generated tree rows do not match the neural source rows for {method}")
        q = original
        alignment = "identical real array; original ordered generated rows match neural float32 casts exactly"
    if (len(p), len(q)) != (reference.n_p, reference.n_q):
        raise ValueError(f"Unexpected assessment counts for {method}")

    pred = f"private/{method}__{RECIPE}_test_predictions.npz"
    if (directory / pred).is_file():
        pin(directory / pred, artifacts[pred])
        with np.load(directory / pred, allow_pickle=False) as saved:
            pr_p, pr_q = saved["p"], saved["q"]
        source = pred
    else:
        source = f"private/{method}__{RECIPE}.joblib"
        pin(directory / source, artifacts[source])
        fit = joblib.load(directory / source)
        with threadpool_limits(limits=2):
            pr_p = mask_metrics.predict_classifier(fit, p)
            pr_q = mask_metrics.predict_classifier(fit, q)
    if pr_p.shape != (len(p),) or pr_q.shape != (len(q),):
        raise ValueError(f"Invalid tree probability shapes for {method}")
    probability = np.r_[pr_p, pr_q]
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError(f"Invalid tree probabilities for {method}")
    labels = np.r_[np.ones(len(p)), np.zeros(len(q))]
    weights = np.r_[np.full(len(p), 0.5 / len(p)), np.full(len(q), 0.5 / len(q))]
    loss = float(log_loss(labels, probability, sample_weight=weights, labels=[0, 1]))
    if not np.isclose(loss, public_loss, rtol=0, atol=1e-12):
        raise ValueError(f"Archived tree log loss was {public_loss}, recomputed {loss} for {method}")
    # These are probabilities already. The published neural helper consumes RDRs.
    score = balanced_brier(2 * pr_p, 2 * pr_q)
    audit = dict(source=str(directory / source), source_sha256=artifacts[source],
                 archived_log_loss=public_loss, reproduced_log_loss=loss, alignment=alignment,
                 ordered_real_hash=input_hashes["real"], ordered_generated_hash=input_hashes[method],
                 real_count=len(p), generated_count=len(q))
    return score, loss, audit


def run(output):
    arrays, ensemble, identities, inputs = assessment.load_assessment()
    rows, all_seeds, audited_tree = [], [], {}
    for method, root in zip(assessment.METHODS, assessment.ROOTS):
        seeds = neural_fits(method, root, inputs)
        all_seeds.extend(seeds)
        scores = [row["brier"] for row in seeds]
        try:
            tree, loss, audit = tree_score(method, arrays, identities)
        except FileNotFoundError as exc:
            tree, loss, audit = None, None, {"unavailable": str(exc)}
        audited_tree[method] = audit
        rows.append(dict(method=method, published_neural_brier=np.mean(scores),
                         published_neural_sd=np.std(scores, ddof=1),
                         published_ensemble_brier=balanced_brier(ensemble[method+"_p"],
                                                                ensemble[method+"_q"]),
                         tree_brier=tree, tree_log_loss=loss))
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "baselines.csv", index=False)
    pd.DataFrame(all_seeds).to_csv(output / "published_per_seed.csv", index=False)
    (output / "baseline_audit.json").write_text(json.dumps(
        {"published_inputs_sha256": inputs, "tree": audited_tree}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
