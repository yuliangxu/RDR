"""Replay figures from frozen primary scores and conditional calibration cells."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from experiments.MNIST import model_selection as ms

COLORS = {"p": "#3677a8", "q": "#dc8048"}


def _save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white",
                metadata={"Creator": "MNIST controls replay", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def _hist(ax, record, title, labels):
    for side, label in zip(("p", "q"), labels):
        values = record[side]["scores"].numpy()
        ax.hist(values, bins=np.linspace(0, 2, 41), weights=np.full(len(values), 1/len(values)),
                histtype="step", linewidth=1.8, color=COLORS[side], label=label)
    ax.axvline(1, color=".4", linestyle="--", linewidth=1)
    ax.set(title=title, xlabel="Estimated RDR", ylabel="Fraction within each source", xlim=(0, 2))
    ax.legend(fontsize=8, frameon=False)


def render(output: Path):
    output = Path(output)
    names = ("perturbation", "null")
    metrics = {name: ms.read(output / name / "metrics.json") for name in names}
    scores = {name: torch.load(output / name / "scores.pt", map_location="cpu", weights_only=True) for name in names}
    digits = {name: ms.read(output / name / "digits.json") for name in names}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), constrained_layout=True)
    m = metrics["perturbation"]
    _hist(axes[0, 0], scores["perturbation"]["test"], f"Digit perturbation: held-out Brier {m['primary_brier']:.4f}",
          ("P real", "Q digit-resampled"))
    record = scores["perturbation"]["test"]["p"]
    ax = axes[0, 1]
    values, labels = record["scores"].numpy(), record["labels"].numpy()
    groups = [(digit, values[labels == digit]) for digit in range(10)]
    viable = [(digit, group) for digit, group in groups if len(group) > 1 and np.ptp(group) > 0]
    if viable:
        parts = ax.violinplot([group for _, group in viable], positions=[digit for digit, _ in viable],
                              widths=.75, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(COLORS["p"])
            body.set_alpha(.22)
    for side, marker in (("p", "o"), ("q", "s")):
        rows = [row for row in digits["perturbation"] if row["role"] == "test" and row["side"] == side]
        available = [row for row in rows if row["mean_rdr"] is not None]
        ax.plot([row["digit"] for row in available], [row["mean_rdr"] for row in available],
                marker=marker, color=COLORS[side], linewidth=1.3, markersize=4, label=f"{side.upper()} mean")
    reference = [row for row in digits["perturbation"] if row["role"] == "test" and row["side"] == "p"]
    ax.plot(range(10), [row["nominal_reference"] for row in reference], "k--", linewidth=1.1, label="Uniform-P reference")
    ax.plot(range(10), [row["empirical_reference"] for row in reference], color=".4", linestyle=":", linewidth=1.2, label="Empirical-pool reference")
    ax.set(title="Perturbation: held-out digit scores", xlabel="Digit", ylabel="Estimated RDR",
           xticks=range(10), ylim=(-.03, 2.07))
    ax.legend(fontsize=7.5, ncol=2, frameon=False, loc="lower left")
    m = metrics["null"]
    _hist(axes[1, 0], scores["null"]["validation"], f"Real versus real: validation Brier {m['primary_brier']:.4f}",
          ("P real half", "Q real half"))
    ax = axes[1, 1]
    for name, color in (("perturbation", COLORS["q"]), ("null", COLORS["p"])):
        history = ms.read(output / name / "history.json")
        epochs = [row["epoch"] for row in history]
        ax.plot(epochs, [row["earlystop_brier"] for row in history], "o-", color=color, markersize=3, label=name)
        selected = metrics[name]
        ax.scatter([selected["best_epoch"]], [selected["earlystop_brier"]], color=color, s=80,
                   facecolors="none", linewidths=1.4)
    ax.axhline(.25, color=".5", linestyle="--", linewidth=1, label="Constant r=1 Brier")
    ax.set(title="Checkpoint selection by validation Brier", xlabel="Epoch", ylabel="Balanced Brier")
    ax.legend(fontsize=8, frameon=False)
    chosen = ms.read(output / "protocol.json")["chosen"]
    fig.suptitle(f"MNIST controls · {chosen['loss'].upper()} loss · r = 2 sigmoid({chosen['output_alpha']:g}z)\n"
                 "Fresh configuration transfer; real-versus-real is validation only", fontsize=14)
    _save(fig, output, "mnist_selected_controls")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    for col, name in enumerate(names):
        cells = ms.read(output / name / "cells.json")
        ax, mass_ax = axes[0, col], axes[1, col]
        for cell in cells:
            center = (cell["left"]+cell["right"])/2
            if cell["eval_mass"] == 0:
                continue
            ax.vlines(center, cell["c2_lower"], cell["c2_upper"], color="#849cae", linewidth=2, alpha=.75)
            if cell["calibrated_rdr"] is not None:
                ax.scatter(center, cell["calibrated_rdr"], color=COLORS["p"], s=25, zorder=3)
            if cell["neural_mean"] is not None:
                ax.scatter(center, cell["neural_mean"], marker="x", color=COLORS["q"], s=32, zorder=4)
        ax.plot([0, 2], [0, 2], color=".6", linewidth=.8, linestyle=":")
        m = metrics[name]
        gap = "NA" if m["local_gap"] is None else f"{m['local_gap']:.4f}"
        ax.set(title=f"{name.capitalize()}: conditional local gap {gap}", ylabel="Cell-average RDR", ylim=(-.03, 2.03), xlim=(0, 2))
        ax.plot([], [], color="#849cae", linewidth=2, label="Conditional C.2 interval")
        ax.scatter([], [], color=COLORS["p"], label="Calibration estimate")
        ax.scatter([], [], color=COLORS["q"], marker="x", label="Evaluation neural mean")
        ax.legend(fontsize=8, frameon=False, loc="upper left")
        mass_ax.bar([(cell["left"]+cell["right"])/2 for cell in cells],
                    [cell["eval_mass"] for cell in cells], width=.085, color="#90a4b3")
        mass_ax.axvspan(.7, 1.2, color="#e4c06b", alpha=.15)
        mass_ax.set(xlabel="Fixed score-cell midpoint", ylabel="Mixture mass", xlim=(0, 2))
    fig.suptitle("Conditional finite-pool calibration diagnostics\n"
                 "Fresh resampling uncertainty only; reference-image uncertainty is excluded", fontsize=13)
    _save(fig, output, "mnist_controls_calibration")
    input_names = [f"{name}/{file}" for name in names for file in ("scores.pt", "metrics.json", "digits.json", "history.json", "cells.json")]
    output_names = [f"{stem}.{extension}" for stem in ("mnist_selected_controls", "mnist_controls_calibration") for extension in ("png", "pdf")]
    manifest = {"description": "Figures replayed solely from the saved new primary scores and conditional calibration cells",
                "inputs": {name: ms.sha(output / name) for name in input_names},
                "outputs": {name: ms.sha(output / name) for name in output_names},
                "renderer_sha256": ms.sha(Path(__file__)),
                "null_role": "validation only", "ci_scope": "conditional finite-pool Monte Carlo uncertainty"}
    ms.write(output / "figure_manifest.json", manifest)
    return manifest
