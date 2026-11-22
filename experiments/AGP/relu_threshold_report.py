"""Summarize the frozen four-generator assessment from saved predictions."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.model_selection import balanced_brier


METHODS = {
    "bernoulli_mixture_icfm": "Mixture + ICFM",
    "autoregressive_icfm": "NADE + ICFM",
    "midasim_mask_icfm": "MIDASim mask + ICFM",
    "midasim": "Full MIDASim",
}
PREDICTIONS = ("threshold_p", "threshold_q", "control_p", "control_q")
ENDPOINTS = [f"threshold_{limit}_{side}" for side in ("p", "q") for limit in ("lower", "upper")]
MODEL_NAMES = {"published_neural_brier": "Published ILR neural", "threshold_brier": "Clipped ReLU",
               "tree_brier": "Boosted tree", "control_brier": "Tree-feature MLP"}


def verify_number(observed, expected, name):
    if not math.isclose(float(observed), float(expected), abs_tol=1e-10, rel_tol=1e-9):
        raise ValueError(f"{name} disagrees: {observed} versus {expected}")


def winner(scores, include_control):
    if pd.isna(scores["tree_brier"]):
        return "Unavailable"
    keys = [k for k in MODEL_NAMES if include_control or k != "control_brier"]
    best = min(scores[k] for k in keys)
    return " / ".join(MODEL_NAMES[k] for k in keys if scores[k] == best)


def summarize_method(output, method, baseline, published, protocol):
    reports = []
    accumulated = {}
    arrays = protocol["inputs"][method]["arrays"]
    expected_hashes = {f"test_{s}": arrays[f"test_{s}"]["sha256"] for s in ("p", "q")}
    bounds = {"lower": np.float32(protocol["config"]["a"]), "upper": np.float32(protocol["config"]["b"])}
    if len(published) != 5 or set(published.index) != set(range(5)) or published.seed.nunique() != 5:
        raise ValueError(f"{method}: expected five distinct archived final seeds")
    for repeat in range(5):
        path = output / "assessment" / method / f"{repeat:02d}"
        record = json.loads((path / "scores.json").read_text())
        seed = int(published.loc[repeat, "seed"])
        if (record["method"], record["repeat"], record["seed"]) != (method, repeat, seed):
            raise ValueError(f"{method}/{repeat}: seed or generator mismatch")
        if record["test_hashes"] != expected_hashes:
            raise ValueError(f"{path}: test bank hashes disagree with the frozen protocol")
        row = dict(method=method, repeat=repeat, seed=seed,
                   published_neural_brier=float(published.loc[repeat, "brier"]))
        with np.load(path / "predictions.npz", allow_pickle=False) as data:
            predictions = {k: data[k] for k in PREDICTIONS}
        for side in ("p", "q"):
            scores = predictions[f"threshold_{side}"]
            expected_size = arrays[f"test_{side}"]["shape"][0]
            if scores.shape != (expected_size,) or scores.shape != predictions[f"control_{side}"].shape:
                raise ValueError(f"{path}: score length disagrees with the frozen arrays")
            if np.any((scores < bounds["lower"]) | (scores > bounds["upper"])):
                raise ValueError(f"{path}: saved RDR outside frozen clipping interval")
            for limit, bound in bounds.items():
                key = f"threshold_{limit}_{side}"
                row[key] = float(np.mean(scores == bound))
                verify_number(row[key], record[key], f"{path} {key}")
        for name in ("threshold", "control"):
            brier = balanced_brier(predictions[name + "_p"], predictions[name + "_q"])
            verify_number(brier, record[name + "_brier"], f"{path} {name} Brier")
            row[name + "_brier"] = brier
        for key, array in predictions.items():
            if repeat == 0:
                accumulated[key] = np.zeros_like(array, dtype=np.float64)
            if array.shape != accumulated[key].shape:
                raise ValueError(f"{path}: dimensions changed between seeds")
            accumulated[key] += array
        reports.append(row)
    per_seed = pd.DataFrame(reports)
    result = dict(method=method, generator=METHODS[method], tree_brier=baseline.tree_brier,
                  tree_log_loss=baseline.tree_log_loss, published_ensemble_brier=baseline.published_ensemble_brier)
    for name in ("published_neural", "threshold", "control"):
        scores = per_seed[name + "_brier"]
        result[name + "_brier"] = float(scores.mean())
        result[name + "_sd"] = float(scores.std(ddof=1))
    verify_number(result["published_neural_brier"], baseline.published_neural_brier, method + " published mean")
    verify_number(result["published_neural_sd"], baseline.published_neural_sd, method + " published SD")
    for name in ("threshold", "control"):
        result[name + "_ensemble_brier"] = balanced_brier(
            accumulated[name + "_p"] / 5, accumulated[name + "_q"] / 5)
    result.update({column: float(per_seed[column].mean()) for column in ENDPOINTS})
    result["delta_vs_tree"] = result["threshold_brier"] - result["tree_brier"]
    result["delta_vs_published"] = result["threshold_brier"] - result["published_neural_brier"]
    result["primary_winner"] = winner(result, False)
    result["all_methods_winner"] = winner(result, True)
    return result, per_seed


def format_number(value, signed=False):
    if pd.isna(value):
        return "Unavailable"
    return format(value, "+.6f" if signed else ".6f")


def render(summary):
    lines = [
        "# Retrospective AGP comparison of relative density ratio estimators", "",
        "The table reports held-out balanced Brier scores for the four separate real-versus-generated comparisons. "
        "Lower scores indicate more accurate real-source probabilities. Neural columns give the mean and sample SD "
        "of five individually fitted models using the archived final seeds. The tree column gives the score of "
        "the single frozen composition tree. ΔBS subtracts its score from the clipped ReLU mean.", "",
        "| Generator | Published ILR neural | Clipped ReLU | Boosted tree | ΔBS | Tree-feature MLP | Winner among primary three | Winner including control |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in summary.itertuples():
        lines.append(f"| {r.generator} | {r.published_neural_brier:.6f} ± {r.published_neural_sd:.6f} | "
                     f"{r.threshold_brier:.6f} ± {r.threshold_sd:.6f} | {format_number(r.tree_brier)} | "
                     f"{format_number(r.delta_vs_tree, True)} | {r.control_brier:.6f} ± {r.control_sd:.6f} | "
                     f"{r.primary_winner} | {r.all_methods_winner} |")
    n_pairs = len(summary)
    n_ilr = int((summary.threshold_brier < summary.published_neural_brier).sum())
    n_control = int((summary.threshold_brier < summary.control_brier).sum())
    closest = summary.loc[summary.delta_vs_published.abs().idxmin()]
    lines.extend(["", f"The clipped feedforward ReLU estimator has a lower five-fit mean than the published "
                  f"ILR estimator for {n_ilr} of {n_pairs} generators and than the tree-feature MLP for "
                  f"{n_control} of {n_pairs}. The smallest absolute difference from ILR is for "
                  f"{closest.generator} (clipped ReLU minus ILR = {closest.delta_vs_published:+.6f})."])
    lines.extend(["", "A missing verified tree score makes ΔBS and the winner unavailable."
                  if summary.tree_brier.isna().any() else
                  "The frozen tree's real-source predictions reproduce the archived log loss for each generator.",
                  "", "The clipped feedforward ReLU estimator uses abundance and taxon-specific positive-abundance "
                  "hinges, an additive output, and two width-128 ReLU interaction layers. Each generator has its own "
                  "fitted weights. The control uses the published deep sigmoid MLP on the tree's square-root abundance "
                  "and presence features, with the published objective and slope for each generator. "
                  "The proposed model also changes the architecture and output link, so a score difference cannot "
                  "be attributed solely to thresholds.", "",
                  "The published assessment results motivated this architecture. These comparisons are retrospective. "
                  "Variation across seeds measures initialization and training variability on fixed data. "
                  "Theorem 3's distributional and optimization assumptions have not been established for AGP.", "",
                  "## Ensembles and clipping", "",
                  "For an ensemble, we first average the five real-source probabilities for each observation, then "
                  "compute one balanced Brier score. Endpoint percentages give the fraction clipped to each endpoint "
                  "within each source on the test set, averaged across the five clipped ReLU fits.", "",
                  "| Generator | Published ensemble BS | Clipped ReLU ensemble BS | Control ensemble BS | "
                  "Real lower | Real upper | Generated lower | Generated upper |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for r in summary.itertuples():
        lines.append(f"| {r.generator} | {r.published_ensemble_brier:.6f} | {r.threshold_ensemble_brier:.6f} | "
                     f"{r.control_ensemble_brier:.6f} | {r.threshold_lower_p:.2%} | {r.threshold_upper_p:.2%} | "
                     f"{r.threshold_lower_q:.2%} | {r.threshold_upper_q:.2%} |")
    changes = []
    for r in summary.itertuples():
        if pd.isna(r.tree_brier):
            continue
        ensemble_scores = {"Clipped ReLU": r.threshold_ensemble_brier, "Published ILR neural": r.published_ensemble_brier,
                           "Tree-feature MLP": r.control_ensemble_brier, "Boosted tree": r.tree_brier}
        lowest = min(ensemble_scores, key=ensemble_scores.get)
        if lowest != r.all_methods_winner:
            changes.append(f"For {r.generator}, {lowest} has the lowest score under the ensemble comparison "
                           f"({ensemble_scores[lowest]:.6f}).")
    if changes:
        lines.extend(["", " ".join(changes)])
    lines.extend(["", "[Per-seed results](per_seed.csv) · [Full summary](summary.csv)", ""])
    return "\n".join(lines)


def generate(output, report_dir):
    protocol = json.loads((output / "protocol.json").read_text())
    baseline_dir = output / "baselines"
    baseline = pd.read_csv(baseline_dir / "baselines.csv").set_index("method", verify_integrity=True)
    per_seed = pd.read_csv(baseline_dir / "published_per_seed.csv")
    if set(baseline.index) != set(METHODS) or set(per_seed.method) != set(METHODS):
        raise ValueError("Baseline set differs from the four requested generators")
    results, seeds = [], []
    for method in METHODS:
        published = per_seed.loc[per_seed.method == method].set_index("repeat", verify_integrity=True)
        result, fits = summarize_method(output, method, baseline.loc[method], published, protocol)
        results.append(result)
        seeds.append(fits)
    summary = pd.DataFrame(results)
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, content in (("summary.csv", summary.to_csv(index=False)),
                          ("per_seed.csv", pd.concat(seeds).to_csv(index=False)),
                          ("report.md", render(summary))):
        path = report_dir / name
        temporary = path.with_name(f".{name}.{os.getpid()}.tmp")
        temporary.write_text(content)
        temporary.replace(path)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output, args.report_dir)
