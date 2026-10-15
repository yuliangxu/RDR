#!/usr/bin/env python3
"""Recreate the AGP example using rescaled Bernoulli distributions.

Run from the repository root with ``python -B -m
experiments.simulations.global_local_toy --output runs/global-local-toy``.
All quantities are analytic; no samples, fitted networks, or cluster jobs enter.
The original AGP module supplies the unchanged population and metric formulas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np
import pandas as pd

from experiments.AGP import rdr_metric_toy as population


LABELS = {"P": r"$P$", "Q1": r"$Q_1$", "Q2": r"$Q_2$"}
CAPTION = """Identical global summaries, opposite local mismatch. (a) Discrete probability
distributions P, Q1, and Q2 have common support {-1,+1} and probability mass
functions p, q1, and q2. In the order x=-1,+1, their probability masses are
(0.5,0.5), (0.8,0.2), and (0.2,0.8), respectively.
Equivalently, X=2B-1 for B~Bernoulli(theta), with theta=0.5,0.2,0.8.
(b) The entire population precision-recall-for-distributions (PRD) curves
coincide. Both generators have scalar-feature Gaussian Frechet distance
(the FID formula applied to these features) 0.4, and Inception Score 1.649385
under an ideal deterministic classifier that assigns distinct labels to -1 and +1.
(c) Exact RDR, r_k(x)=2p(x)/(p(x)+q_k(x)), is (10/13,10/7) for Q1 and
(10/7,10/13) for Q2, in the same support order. RDR below one means q_k(x)>p(x);
RDR above one means q_k(x)<p(x). The dotted line marks equal probability mass.

The global metrics detect mismatch here but do not identify its location.
These are exact population quantities, not empirical image/Inception metrics
or estimated RDR. The PRD claim concerns the population PRD definition of
Sajjadi et al.; other conditional or local metrics can also locate imbalance.
Finite-sample estimates need not coincide. The unchanged construction was
originally reported in results/AGP/rdr_metric_comparison_20261009/toy/.

PRD: https://arxiv.org/abs/1806.00035
Gaussian Frechet/FID: https://arxiv.org/abs/1706.08500
IS: https://arxiv.org/abs/1606.03498
"""


def make_figure(global_table, regional, curves):
    """Return the compact three-panel figure, also usable interactively."""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rdr-global-local-mpl"))
    import matplotlib.pyplot as plt

    style = {
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.3))
        fig.subplots_adjust(left=.065, right=.995, bottom=.21, top=.86, wspace=.43)
        x = np.arange(2)
        for offset, (name, masses) in zip([-.23, 0, .23], [("P", population.P), *population.Q.items()]):
            bars = axes[0].bar(x + offset, masses, width=.22,
                               color=population.COLORS[name], label=LABELS[name])
            axes[0].bar_label(bars, labels=[f"{mass:.1f}" for mass in masses],
                              fontsize=7, padding=2)
        axes[0].set(xticks=x, xticklabels=[r"$-1$", r"$+1$"],
                    ylim=(0, 1.08), yticks=[0, .5, 1], xlabel=r"$x$",
                    ylabel="Probability", title="(a) Probability mass")
        axes[0].legend(frameon=False, ncol=3, loc="upper center", borderaxespad=.1,
                       handlelength=1, handletextpad=.3, columnspacing=.7)

        for name, line_style, width in [("Q1", "-", 2.6), ("Q2", "--", 1.6)]:
            rows = curves[curves.generator.eq(name)]
            # Keep increasing slope order, including both limiting endpoints.
            # Sorting recall alone scrambles the vertical segment at recall=1.
            axes[1].plot(rows.recall, rows.precision, line_style, lw=width,
                          color=population.COLORS[name], label=LABELS[name])
        axes[1].set(xlim=(-.02, 1.02), ylim=(-.02, 1.06), xticks=[0, .5, 1],
                    yticks=[0, .5, 1], xlabel="Recall", ylabel="Precision",
                    title="(b) Identical PRD")
        axes[1].legend(frameon=False, loc="lower left", borderaxespad=.3,
                       handlelength=2.2, labelspacing=.2)
        score = global_table.iloc[0]
        axes[1].text(.51, .40,
                     "Both generators\n"
                     f"Feature-space FID = {score.population_gaussian_frechet:.1f}\n"
                     f"Ideal-label IS = {score.ideal_two_label_is:.3f}",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=6.6, linespacing=1.4)

        for offset, name in [(-.18, "Q1"), (.18, "Q2")]:
            rows = regional[regional.generator.eq(name)]
            bars = axes[2].bar(x + offset, rows.population_region_rdr, width=.34,
                               color=population.COLORS[name], label=LABELS[name])
            axes[2].bar_label(bars, labels=[f"{v:.3f}" for v in rows.population_region_rdr],
                              fontsize=7, padding=2)
        axes[2].axhline(1, color="#444444", ls=":", lw=1)
        axes[2].set(xticks=x, xticklabels=[r"$-1$", r"$+1$"], ylim=(0, 2),
                    yticks=[0, 1, 2], xlabel=r"$x$", ylabel=r"RDR $2p/(p+q_k)$",
                    title="(c) Local RDR")
        axes[2].legend(frameon=False, ncol=2, loc="upper center", borderaxespad=.2,
                       handlelength=1, handletextpad=.4)
    return fig


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output):
    directory = Path(output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    global_table, regional, curves = population.example_tables()
    global_table.to_csv(directory / "population_global_metrics.csv", index=False)
    regional.to_csv(directory / "population_regional_rdr.csv", index=False)
    curves.to_csv(directory / "population_prd_curve.csv", index=False)
    fig = make_figure(global_table, regional, curves)
    import matplotlib
    import matplotlib.pyplot as plt
    for suffix in ["png", "pdf"]:
        options = {"dpi": 300} if suffix == "png" else {
            "metadata": {"CreationDate": None, "ModDate": None}}
        fig.savefig(directory / f"global_local_population_example.{suffix}",
                    bbox_inches="tight", pad_inches=.03, **options)
    plt.close(fig)
    (directory / "caption.md").write_text(CAPTION)
    metadata = {
        "quantities": "exact_population", "fitted_models": 0,
        "family": "X=2B-1 with B~Bernoulli(theta)",
        "bernoulli_theta": {"P": 0.5, "Q1": 0.2, "Q2": 0.8},
        "support": population.STATES.tolist(), "p": population.P.tolist(),
        "q": {key: value.tolist() for key, value in population.Q.items()},
        "pmf_definition": "p(x)=P({x}), q_k(x)=Q_k({x}); arrays follow support order",
        "gaussian_frechet_exact": "2/5",
        "rdr_q1_exact": ["10/13", "10/7"], "rdr_q2_exact": ["10/7", "10/13"],
        "ideal_label_is_formula": "exp(-0.8*log(0.8)-0.2*log(0.2))",
        "prd_alpha_formula": "sum_i min(lambda*p_i,q_i)",
        "prd_beta_formula": "alpha/lambda",
        "prd_identity_reason": "Reflection x -> -x preserves P and interchanges Q1 and Q2.",
        "scope": "Equal global summaries conceal the location of nonzero mismatch.",
        "fid_features": "scalar coordinates -1,+1; no Inception feature extractor",
        "is_classifier": "ideal deterministic classifier assigning distinct labels to -1 and +1",
        "original_example": "results/AGP/rdr_metric_comparison_20261009/toy",
    }
    (directory / "population_example.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n")
    root = Path(__file__).resolve().parents[2]
    sources = [Path(__file__).resolve(), Path(population.__file__).resolve()]
    receipt = {
        "status": "complete",
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in sources},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "files": {path.name: sha256(path) for path in sorted(directory.iterdir()) if path.is_file()},
    }
    (directory / "COMPLETE.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
