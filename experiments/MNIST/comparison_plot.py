"""Render the retrained MNIST comparison from saved images and RDR scores.

No generator, training code, or dataset loader is imported by this renderer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


PANEL_TITLES = {
    "smallest": "Smallest scores",
    "closest_to_one": "Closest to 1",
    "largest": "Largest scores",
}
STEM = "mnist_vae_dcgan_comparison"


def _array(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def selection_indices(scores, k=40):
    """Return each ordered panel, breaking every score tie by test position."""
    scores = _array(scores).astype(np.float64, copy=False)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite one-dimensional array")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("Panel size must be a positive integer")
    return {
        "smallest": np.argsort(scores, kind="stable")[:k],
        "closest_to_one": np.argsort(np.abs(scores - 1), kind="stable")[:k],
        "largest": np.argsort(-scores, kind="stable")[:k],
    }


def display_images(images, role):
    """Convert real uint8 or generated model-space images to [0,1] grayscale."""
    images = _array(images)
    if images.ndim != 4 or images.shape[1:] != (1, 28, 28) or not len(images):
        raise ValueError("Images must have nonempty shape (N,1,28,28)")
    if role == "p":
        if images.dtype != np.uint8:
            raise ValueError("Real display images must be original uint8 pixels")
        return images[:, 0].astype(np.float32) / 255
    if role != "q":
        raise ValueError("Image role must be p or q")
    if not np.issubdtype(images.dtype, np.floating) or not np.isfinite(images).all():
        raise ValueError("Generated display images must be finite model-space floats")
    if images.min() < -1.000001 or images.max() > 1.000001:
        raise ValueError("Generated model-space images must lie in [-1,1]")
    return np.clip(images[:, 0].astype(np.float32) / 2 + .5, 0, 1)


def image_mosaic(images, indices):
    """Build the historical five-row/eight-column grid, leaving blanks if small."""
    if len(indices) > 40:
        raise ValueError("A comparison panel holds at most 40 images")
    mosaic = np.zeros((5 * 28, 8 * 28), dtype=np.float32)
    for cell, index in enumerate(indices):
        row, column = divmod(cell, 8)
        mosaic[row * 28:(row + 1) * 28, column * 28:(column + 1) * 28] = images[index]
    return mosaic


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(folder):
    bundle = torch.load(folder / "evaluation.pt", map_location="cpu", weights_only=True)
    metrics = json.loads((folder / "metrics.json").read_text())
    result = {}
    for role in ("p", "q"):
        result[f"{role}_images"] = display_images(bundle[f"{role}_images"], role)
        scores = _array(bundle[f"{role}_scores"]).astype(np.float64, copy=False)
        selection_indices(scores)
        if len(scores) != len(result[f"{role}_images"]) or (scores < 0).any() or (scores > 2).any():
            raise ValueError(f"{folder.name}: image/score counts or score bounds disagree")
        result[f"{role}_scores"] = scores
    labels, indices = (_array(bundle[key]) for key in ("p_labels", "p_indices"))
    for name, values in (("labels", labels), ("indices", indices)):
        if values.ndim != 1 or len(values) != len(result["p_scores"]) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{folder.name}: real {name} must be aligned integer vectors")
    if (labels < 0).any() or (labels > 9).any():
        raise ValueError("Real digit labels must lie in [0,9]")
    if (indices < 0).any() or len(np.unique(indices)) != len(indices):
        raise ValueError("Real test indices must be unique and nonnegative")
    result.update(p_labels=labels, p_indices=indices)
    brier = .5 * (np.mean((result["p_scores"] / 2 - 1) ** 2)
                   + np.mean((result["q_scores"] / 2) ** 2))
    if not math.isclose(brier, metrics["test_brier"], abs_tol=1e-8, rel_tol=1e-6):
        raise ValueError(f"{folder.name}: reported test Brier disagrees with saved scores")
    for key in ("output_alpha", "earlystop_brier"):
        if not math.isfinite(float(metrics[key])):
            raise ValueError(f"{folder.name}: {key} must be finite")
    if metrics["output_alpha"] <= 0:
        raise ValueError("Output sigmoid slope must be positive")
    return result, metrics


def _digit_violins(ax, scores, labels):
    # KDE is undefined for absent, singleton, or constant-score groups. Show
    # constant groups directly and leave absent digits empty on the fixed axis.
    positions, values = [], []
    for digit in range(10):
        group = scores[labels == digit]
        if not len(group):
            continue
        if len(group) < 2 or np.ptp(group) == 0:
            ax.plot([digit - .3, digit + .3], [group[0], group[0]], color="#327ba5", lw=2)
        else:
            positions.append(digit)
            values.append(group)
    if values:
        parts = ax.violinplot(values, positions=positions, widths=.82,
                              showextrema=False, quantiles=[[.25, .5, .75]] * len(values),
                              bw_method=.3)
        for body in parts["bodies"]:
            body.set_facecolor("#327ba5")
            body.set_edgecolor("#333333")
            body.set_alpha(.95)
            body.set_linewidth(.6)
        parts["cquantiles"].set_color("#333333")
        parts["cquantiles"].set_linewidth(.6)
    ax.set(ylim=(0, 2), xlim=(-.8, 9.8), xticks=range(10), xlabel="Digit",
           ylabel=r"$r(X_P)$", title="Real test scores by digit")


def render(output: Path):
    """Render both retrained models and return the saved figure manifest."""
    output = Path(output).resolve()
    # Load and validate both inputs before creating any output files.
    loaded = {name: _load(output / name) for name in ("vae", "dcgan")}
    for key in ("p_indices", "p_labels", "p_images"):
        if not np.array_equal(loaded["vae"][0][key], loaded["dcgan"][0][key]):
            raise ValueError(f"The two models must use identical ordered real test rows: {key}")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mnist-comparison-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    manifest = {
        "schema": 1,
        "method": "All histograms, digit violins, and image mosaics use newly saved evaluation images and scores.",
        "selection": "Illustrative extremes and scores closest to 1; panels do not estimate prevalence. Ties use increasing evaluation position.",
        "display_conversion": {"p": "original uint8 / 255", "q": "model-space float / 2 + 0.5"},
        "histogram_bins": np.linspace(0, 2, 51).tolist(),
        "grid": {"rows": 5, "columns": 8, "maximum_images": 40, "unused_cells": "black"},
        "brier": "0.5 mean_P[(r/2-1)^2] + 0.5 mean_Q[(r/2)^2]",
        "panels": [], "inputs": {}, "outputs": {},
    }
    for name in loaded:
        for filename in ("evaluation.pt", "metrics.json"):
            path = output / name / filename
            manifest["inputs"][str(path)] = _sha(path)
    manifest["inputs"][str(Path(__file__).resolve())] = _sha(Path(__file__))
    selection = []
    style = {"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
             "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "pdf.fonttype": 42}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(18, 9.1), facecolor="white")
        fig.add_artist(plt.Line2D([.505, .505], [.055, .94], color=".82", lw=1))
        for name, left, centre in (("vae", .055, .26), ("dcgan", .545, .755)):
            data, metrics = loaded[name]
            label = "VAE" if name == "vae" else "DCGAN"
            fig.text(centre, .975, label, ha="center", fontsize=22)
            fig.text(centre, .943, f"{metrics['loss'].upper()} loss; "
                     + rf"$r(x)=2\,\mathrm{{sigmoid}}({metrics['output_alpha']:g}z(x))$",
                     ha="center", fontsize=11)
            ax = fig.add_axes([left, .685, .19, .215])
            for role, color, legend in (("p", "#9876b4", "Real test"),
                                       ("q", "#66a3c4" if name == "vae" else "#70b578", label)):
                ax.hist(data[f"{role}_scores"], bins=manifest["histogram_bins"], density=True,
                        color=color, edgecolor="#555555", linewidth=.4, alpha=.48, label=legend)
            ax.set(xlim=(-.025, 2.025), xlabel=r"RDR $r(x)$", ylabel="Density",
                   title=f"Test scores; Brier = {metrics['test_brier']:.4f}")
            ax.legend(fontsize=9, loc="upper center" if name == "vae" else "upper right", framealpha=.9)
            fig.text(left - .024, .909, "A" if name == "vae" else "D", fontsize=19)
            ax = fig.add_axes([left + .245, .685, .19, .215])
            _digit_violins(ax, data["p_scores"], data["p_labels"])
            fig.text(left + .22, .909, "B" if name == "vae" else "E", fontsize=19)
            for role, row_y in (("p", .365), ("q", .08)):
                scores = data[f"{role}_scores"]
                for column, (kind, indices) in enumerate(selection_indices(scores).items()):
                    ax = fig.add_axes([left + column * .148, row_y, .137, .18])
                    ax.imshow(image_mosaic(data[f"{role}_images"], indices), cmap="gray",
                              vmin=0, vmax=1, interpolation="none", aspect="auto")
                    ax.set_axis_off()
                    low, high = float(scores[indices].min()), float(scores[indices].max())
                    ax.set_title(f"{PANEL_TITLES[kind]}\n(min={low:.4g}, max={high:.4g})", fontsize=10, pad=8)
                    for x in np.arange(1, 8) * 28:
                        ax.axvline(x - .5, color=".8", lw=.25)
                    for y in np.arange(1, 5) * 28:
                        ax.axhline(y - .5, color=".8", lw=.25)
                    manifest["panels"].append({"generator": name, "role": role,
                        "selection": kind, "n": len(indices), "score_min": low, "score_max": high,
                        "evaluation_positions": indices.tolist()})
                    for rank, index in enumerate(indices):
                        selection.append({"generator": name, "role": role, "selection": kind,
                            "rank": rank, "test_position": int(index), "rdr": float(scores[index]),
                            "official_test_index": int(data["p_indices"][index]) if role == "p" else "",
                            "digit": int(data["p_labels"][index]) if role == "p" else ""})
                fig.text(left - .012, row_y + .09, "Real" if role == "p" else "Generated",
                         rotation=90, va="center", ha="right", fontsize=16)
                if role == "p":
                    fig.text(left - .024, row_y + .24, "C" if name == "vae" else "F", fontsize=19)
            manifest[name] = {"test_p_n": len(data["p_scores"]), "test_q_n": len(data["q_scores"]),
                **{key: metrics[key] for key in ("loss", "output_alpha", "best_epoch", "earlystop_brier", "test_brier")}}
        try:
            for suffix in ("png", "pdf"):
                fig.savefig(output / f"{STEM}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.12)
        finally:
            plt.close(fig)
    with (output / "panel_selection.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selection[0]))
        writer.writeheader()
        writer.writerows(selection)
    for filename in (f"{STEM}.png", f"{STEM}.pdf", "panel_selection.csv"):
        manifest["outputs"][filename] = _sha(output / filename)
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
    print(args.output / f"{STEM}.png")


if __name__ == "__main__":
    main()
