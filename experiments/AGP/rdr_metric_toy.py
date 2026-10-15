#!/usr/bin/env python3
"""Exact two-state example: equal global summaries, opposite local mismatch.

This module uses population probabilities, not AGP or finite sampled data.
Its Gaussian Frechet distance is computed in the declared scalar feature
space {-1, +1}; it is not an Inception-feature FID. The illustrative Inception
Score assumes an ideal deterministic classifier that labels the two states.
No such classifier is assumed or constructed for AGP.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


STATES = np.array([-1.0, 1.0])
P = np.array([0.5, 0.5])
Q = {"Q1": np.array([0.8, 0.2]), "Q2": np.array([0.2, 0.8])}
COLORS = {"P": "#555555", "Q1": "#0072B2", "Q2": "#D55E00"}


def probabilities(value):
    value = np.asarray(value, dtype=np.float64)
    if (value.ndim != 1 or not len(value) or not np.isfinite(value).all()
            or np.any(value < 0) or not np.isclose(value.sum(), 1, rtol=0, atol=1e-12)):
        raise ValueError("Expected a finite nonnegative probability vector summing to one")
    return value


def gaussian_frechet_1d(p, q, states=STATES):
    """Squared 2-Wasserstein distance of population Gaussian moment matches."""
    p, q = probabilities(p), probabilities(q)
    states = np.asarray(states, dtype=np.float64)
    if p.shape != q.shape or states.shape != p.shape or not np.isfinite(states).all():
        raise ValueError("States and probability vectors must have equal finite shape")
    mean_p, mean_q = p @ states, q @ states
    var_p = p @ np.square(states - mean_p)
    var_q = q @ np.square(states - mean_q)
    return float((mean_p - mean_q) ** 2 + (np.sqrt(var_p) - np.sqrt(var_q)) ** 2)


def ideal_label_is(q):
    """exp(H(q)) when a deterministic ideal classifier labels each state."""
    q = probabilities(q)
    positive = q[q > 0]
    return float(np.exp(-positive @ np.log(positive)))


def analytic_prd(p, q, slopes):
    """Population PRD boundary: alpha=sum min(lambda*p,q), beta=alpha/lambda.

    The full PRD set includes all coordinatewise contractions of this boundary.
    Slopes must be positive and finite; exact limiting endpoints are supplied
    separately by ``example_tables``. No clustering or metric feature learning
    enters this two-state population construction.
    """
    p, q = probabilities(p), probabilities(q)
    slopes = np.asarray(slopes, dtype=np.float64)
    if (p.shape != q.shape or slopes.ndim != 1 or not len(slopes)
            or not np.isfinite(slopes).all() or np.any(slopes <= 0)):
        raise ValueError("Expected aligned probabilities and finite positive slopes")
    precision = np.minimum(slopes[:, None] * p[None, :], q[None, :]).sum(axis=1)
    recall = precision / slopes
    return precision, recall


def example_tables():
    """Return exact-population global, regional, and sampled analytic-curve tables."""
    # Include both analytic breakpoints as well as a dense plotting grid.
    slopes = np.unique(np.r_[np.logspace(-4, 4, 401), 0.4, 1.0, 1.6])
    global_rows, region_rows, curve_rows = [], [], []
    for method, q in Q.items():
        global_rows.append({
            "generator": method,
            "population_gaussian_frechet": gaussian_frechet_1d(P, q),
            "ideal_two_label_is": ideal_label_is(q),
            "population_total_variation": float(np.abs(P-q).sum()/2),
            "p_feature_mean": float(P @ STATES),
            "q_feature_mean": float(q @ STATES),
            "p_feature_variance": float(P @ np.square(STATES-P @ STATES)),
            "q_feature_variance": float(q @ np.square(STATES-q @ STATES)),
        })
        for region, coordinate, p_mass, q_mass in zip(["A", "B"], STATES, P, q):
            region_rows.append({
                "generator": method, "community": region, "feature_coordinate": coordinate,
                "p_probability": p_mass, "q_probability": q_mass,
                "population_region_rdr": 2*p_mass/(p_mass+q_mass),
                "q_relative_representation": "overrepresented" if q_mass > p_mass else "underrepresented",
            })
        precision, recall = analytic_prd(P, q, slopes)
        curve_rows.append({"generator": method, "slope": "0_limit", "precision": 0., "recall": 1.})
        curve_rows.extend({"generator": method, "slope": float(slope), "precision": float(a), "recall": float(b)}
                          for slope, a, b in zip(slopes, precision, recall))
        curve_rows.append({"generator": method, "slope": "infinity_limit", "precision": 1., "recall": 0.})
    return pd.DataFrame(global_rows), pd.DataFrame(region_rows), pd.DataFrame(curve_rows)


def make_plot(directory, global_table, regional, curves):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir())/"agp-metric-toy-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.65), gridspec_kw={"wspace": .39})
        x = np.arange(2)
        for offset, (name, value) in zip([-.23, 0, .23], [("P", P), *Q.items()]):
            bars = axes[0].bar(x+offset, value, width=.22, label=name, color=COLORS[name])
            axes[0].bar_label(bars, labels=[f"{v:.0%}" for v in value], fontsize=9, padding=3)
        axes[0].set(xticks=x, xticklabels=["A: feature = -1", "B: feature = +1"],
                    ylim=(0, .97), ylabel="Population probability", title="A   Opposite community imbalance")
        axes[0].legend(frameon=False, ncol=3, loc="upper center", fontsize=9)
        for method, style in [("Q1", "-"), ("Q2", "--")]:
            # Preserve increasing-slope order: sorting rounded recall values can
            # permute the vertical segment at recall=1 and create false zigzags.
            rows = curves[curves.generator.eq(method)]
            axes[1].plot(rows.recall, rows.precision, style, color=COLORS[method],
                         lw=2.7 if method == "Q1" else 1.8, label=method)
        axes[1].set(xlim=(0, 1.02), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision",
                    title="B   Identical population PRD")
        axes[1].legend(frameon=False, loc="lower left")
        score = global_table.iloc[0]
        axes[1].text(.50, .30, f"Both generators\nGaussian Fréchet = {score.population_gaussian_frechet:.3f}"
                     f"\nIdeal-label IS = {score.ideal_two_label_is:.3f}",
                     ha="center", va="center", transform=axes[1].transAxes, fontsize=9,
                     bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=.35"})
        for offset, method in [(-.18, "Q1"), (.18, "Q2")]:
            rows = regional[regional.generator.eq(method)]
            bars = axes[2].bar(x+offset, rows.population_region_rdr, width=.34,
                               color=COLORS[method], label=method)
            axes[2].bar_label(bars, labels=[f"{value:.3f}" for value in rows.population_region_rdr],
                              fontsize=9, padding=3)
        axes[2].axhline(1., color="#555555", ls=":", lw=1)
        axes[2].set(xticks=x, xticklabels=["Community A", "Community B"], ylim=(0, 1.7),
                    ylabel=r"Regional RDR $2P(A)/(P(A)+Q(A))$", title="C   Opposite local mismatch")
        axes[2].legend(frameon=False, ncol=2, loc="upper center", fontsize=9)
        fig.subplots_adjust(bottom=.20, top=.86)
        for suffix in ["png", "pdf"]:
            kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {"dpi": 250}
            fig.savefig(directory/f"global_local_population_example.{suffix}", bbox_inches="tight", **kwargs)
        plt.close(fig)


CAPTION = """Exact population example: identical global summaries can conceal opposite locations of mismatch.
P assigns probability 0.5 to each of two communities, represented by scalar features -1 and +1.
Q1 assigns probabilities (0.8, 0.2), and Q2 assigns (0.2, 0.8).
A: the generators overrepresent opposite communities. B: their complete analytic precision–recall
for distributions (PRD) boundaries coincide because permuting the two states preserves P and
permutes Q. Both have population Gaussian Fréchet distance 0.4 in this scalar feature space.
An ideal deterministic two-label classifier would give both an Inception Score of exp(H(0.8, 0.2))
= 1.649385; no corresponding classifier or IS is claimed for AGP. C: population regional RDR is
10/13 and 10/7 for Q1 and reverses for Q2, identifying the underrepresented community.
These are exact population quantities, not finite-sample estimates; empirical metric estimates
will fluctuate. The Gaussian distance is not image/Inception-feature FID. The PRD statement is
about the aggregate PRD curve, not every precision–recall estimator or an individual-sample
realism/support diagnostic. Conditional metrics supplied with community labels can also locate
the imbalance. The example establishes the loss of location information in these particular
global summaries, not that RDR is the only possible local diagnostic.

PRD definition: Sajjadi et al. (2018), https://arxiv.org/abs/1806.00035.
Gaussian Fréchet/FID definition: Heusel et al. (2017), https://arxiv.org/abs/1706.08500.
IS definition: Salimans et al. (2016), https://arxiv.org/abs/1606.03498.
"""


def run(output):
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing toy artifacts: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    global_table, regional, curves = example_tables()
    global_table.to_csv(directory/"population_global_metrics.csv", index=False)
    regional.to_csv(directory/"population_regional_rdr.csv", index=False)
    curves.to_csv(directory/"population_prd_curve.csv", index=False)
    precision, recall = analytic_prd(P, Q["Q1"], np.array([.1, .4, 1., 1.6, 10.]))
    mirrored = analytic_prd(P, Q["Q2"], np.array([.1, .4, 1., 1.6, 10.]))
    np.testing.assert_allclose(global_table.population_gaussian_frechet, .4, rtol=0, atol=2e-16)
    np.testing.assert_array_equal(precision, mirrored[0])
    np.testing.assert_array_equal(recall, mirrored[1])
    metadata = {
        "status": "complete", "quantities": "exact_population",
        "feature_coordinates": STATES.tolist(), "p": P.tolist(), "q": {key: value.tolist() for key, value in Q.items()},
        "gaussian_frechet_exact": "2/5", "region_rdr_q1_exact": ["10/13", "10/7"],
        "region_rdr_q2_exact": ["10/7", "10/13"],
        "ideal_classifier_is_formula": "exp(-0.8*log(0.8)-0.2*log(0.2))",
        "prd_alpha_formula": "sum_i min(lambda*p_i,q_i)", "prd_beta_formula": "alpha/lambda",
        "prd_identical_for_all_positive_slopes": True,
        "prd_symmetry_reason": "Swapping the two coordinates preserves P and maps Q1 to Q2.",
        "gaussian_frechet_is_image_fid": False, "is_used_on_agp": False,
        "limitation": "Equal global summaries do not imply equal distributions or exclude other local diagnostics.",
    }
    (directory/"population_example.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False)+"\n")
    (directory/"global_local_population_example_caption.md").write_text(CAPTION)
    make_plot(directory, global_table, regional, curves)
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.iterdir()) if path.is_file()}
    receipt = {"status": "complete", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "files": files}
    (directory/"COMPLETE.json").write_text(json.dumps(receipt, indent=2, sort_keys=True)+"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
