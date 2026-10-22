"""Render the 2D simulation from a frozen selection run, without fitting.

Uses repetition zero of the selected Hellinger model and a deterministic,
balanced subset of its independent evaluation observations in all three panels.
"""

import argparse
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np

from experiments.simulations import workflow as w


def render(run, output, points_per_source=1000, plot_seed=20261022):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new output directory for the scatter figure")
    protocol = w.read_json(run / "protocol.json")
    selection, selection_hash = w.load_frozen(run / "selection.json")
    if protocol["stage"] != "selection" or selection["protocol_sha256"] != w.digest(run / "protocol.json"):
        raise ValueError("Expected the intact frozen model-selection run")
    # These sources define the coordinates and seed streams reconstructed below.
    for name in ("population.py", "workflow.py"):
        relative = f"experiments/simulations/{name}"
        if w.digest(Path(__file__).with_name(name)) != protocol["source_sha256"][relative]:
            raise ValueError(f"Scientific source differs from the saved run: {name}")
    if np.__version__ != protocol["versions"]["numpy"]:
        raise ValueError("Use the run's recorded NumPy version to reconstruct observations")

    case, repeat = "D2_noise0.00", 0
    cfg, candidate = protocol["config"], selection["selected"]
    n = cfg["train_n"]
    if not 1 <= points_per_source <= cfg["evaluation_n"]:
        raise ValueError("points_per_source must be between 1 and evaluation_n")
    task_index = w.task_specs(protocol).index((case, repeat, n))
    folder = w.task_path(run, case, repeat, n, candidate)
    fit_manifest = str((folder / "complete.json").relative_to(run))
    if w.digest(folder / "complete.json") != selection["fit_manifest_sha256"][fit_manifest]:
        raise ValueError("Fit manifest differs from the frozen selection")
    metrics = w.verify_fit(run, protocol, task_index, case, repeat, n, candidate)
    population, arrays, seeds = w.samples(cfg, "selection", case, repeat, n)
    with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
        fitted = {source: saved[f"evaluation_{source}"] for source in ("P", "Q")}
        truth = {source: population.truth(arrays[f"evaluation_{source}"])
                 for source in ("P", "Q")}
        for source in ("P", "Q"):
            if fitted[source].shape != (cfg["evaluation_n"],):
                raise ValueError("Unexpected saved evaluation prediction shape")
            np.testing.assert_allclose(truth[source], saved[f"truth_evaluation_{source}"],
                                       rtol=1e-13, atol=1e-13)
    full_mse = float(np.mean(np.concatenate([
        (fitted[source] - truth[source]) ** 2 for source in ("P", "Q")])))
    np.testing.assert_allclose(full_mse, metrics["evaluation_mse"], rtol=1e-12, atol=1e-14)

    rng = np.random.default_rng(plot_seed)
    indices = {source: rng.choice(cfg["evaluation_n"], points_per_source, replace=False)
               for source in ("P", "Q")}
    # Random drawing order prevents either source from always covering the other.
    order = rng.permutation(2 * points_per_source)
    xy = np.vstack([arrays[f"evaluation_{source}"][indices[source]]
                    for source in ("P", "Q")])[order]
    labels = np.repeat([1, 0], points_per_source)[order]
    fitted_plot, truth_plot = [np.concatenate([values[source][indices[source]]
                              for source in ("P", "Q")])[order]
                              for values in (fitted, truth)]
    if not all(np.isfinite(values).all() for values in (xy, fitted_plot, truth_plot)):
        raise ValueError("Scatter inputs must be finite")
    if any(np.any((values < 0) | (values > 2)) for values in (fitted_plot, truth_plot)):
        raise ValueError("RDR values must be in [0, 2]")

    cmap, norm = plt.get_cmap("RdBu_r"), Normalize(0, 2)
    p_color, q_color = cmap(norm(1.75)), cmap(norm(0.25))
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    center = (lower + upper) / 2
    half_width = float(np.max(upper - lower) * 0.54)
    with plt.rc_context({"font.size": 9, "axes.titlesize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharex=True, sharey=True,
                                 layout="compressed")
        axes[0].scatter(*xy.T, c=np.array([q_color, p_color])[labels],
                        s=4, alpha=0.65, linewidths=0)
        axes[0].set_title("(a) P and Q")
        axes[0].legend(handles=[
            Line2D([], [], marker="o", ls="", color=color, markersize=4, label=label)
            for color, label in ((p_color, "$P$"), (q_color, "$Q$"))
        ], loc="lower right", frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.8)
        for ax, values, title in zip(axes[1:], (fitted_plot, truth_plot),
                                     ("(b) Fitted RDR", "(c) True RDR")):
            scatter = ax.scatter(*xy.T, c=values, cmap=cmap, norm=norm, s=4, linewidths=0)
            ax.set_title(title)
        for ax in axes:
            ax.set(xlim=(center[0] - half_width, center[0] + half_width),
                   ylim=(center[1] - half_width, center[1] + half_width),
                   xlabel="$x_1$", aspect="equal")
            ax.tick_params(length=3, width=0.7)
        axes[0].set_ylabel("$x_2$")
        colorbar = fig.colorbar(scatter, ax=axes[1:], fraction=0.025, pad=0.025,
                               ticks=[0, 1, 2], aspect=25)
        colorbar.ax.set_title("RDR", fontsize=9, pad=6)
        colorbar.outline.set_linewidth(0.6)
        output.mkdir(parents=True)
        fig.savefig(output / "scatter_2d.png", dpi=300, facecolor="white")
        fig.savefig(output / "scatter_2d.pdf", facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

    w.save_json(output / "provenance.json", {
        "purpose": "Three-panel illustration of the documented 2D Gaussian-mixture simulation",
        "run": str(run), "fit": str(folder.relative_to(run)),
        "case": case, "repeat": repeat, "candidate": candidate,
        "train_n_per_source": n, "evaluation_n_per_source": cfg["evaluation_n"],
        "plotted_n_per_source": points_per_source, "plot_seed": plot_seed,
        "repetition_rule": "Fixed repetition zero, not selected by evaluation error",
        "evaluation_seed_vectors": {source: seeds[f"evaluation_{source}"] for source in ("P", "Q")},
        "evaluation_indices": {source: indices[source].tolist() for source in ("P", "Q")},
        "drawing_order": order.tolist(), "full_evaluation_mse": full_mse,
        "truth": "2p/(p+q), recomputed at the reconstructed evaluation coordinates",
        "checks": ["frozen selection and fit hashes", "fit identity and sample seeds",
                   "reconstructed analytic truth matches saved truth", "full evaluation MSE matches saved metric"],
        "inputs_sha256": {
            "selection.json": selection_hash,
            "protocol.json": w.digest(run / "protocol.json"),
            fit_manifest: w.digest(folder / "complete.json"),
            **{str((folder / name).relative_to(run)): w.digest(folder / name)
               for name in ("predictions.npz", "metrics.json", "training.json")}},
        "source_sha256": {str(path.relative_to(w.ROOT)): w.digest(path) for path in
                          (Path(__file__).resolve(), Path(w.__file__),
                           Path(__file__).with_name("population.py"))},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "outputs_sha256": {path.name: w.digest(path) for path in sorted(output.iterdir())},
    })
    return output / "scatter_2d.png"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed frozen selection run")
    parser.add_argument("--output", type=Path, required=True, help="New figure output directory")
    parser.add_argument("--points-per-source", type=int, default=1000)
    parser.add_argument("--plot-seed", type=int, default=20261022)
    args = parser.parse_args()
    print(render(args.run, args.output, args.points_per_source, args.plot_seed))


if __name__ == "__main__":
    main()
