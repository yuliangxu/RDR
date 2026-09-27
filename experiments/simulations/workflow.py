"""Paired model selection, immutable Hellinger selection, then convergence.

Run ``python experiments/simulations/workflow.py --help`` from any directory.
Each task fits all candidates on the same P/Q draws. Selection uses mean
validation Brier only; its oracle and calibration outputs are design diagnostics.
Convergence has a new random namespace after selection is frozen. All sample
counts are per distribution, and each role/source has an independent RNG.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy
import torch
from experiments.simulations.population import Mixture, SETTINGS
from utils.calibration import bin_scores, calibrate_counts, calibrate_merged_counts
from utils.diagnostics import plot_calibration_cells
from utils.networks import RATIO_ARCHITECTURES, make_ratio_mlp
from utils.training import fit_ratio_mlp

EDGES = np.linspace(0., 2., 21)
ROLES = {"train": 1, "validation": 2, "calibration": 3, "evaluation": 4, "initialization": 5}
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def digest(path):
    """Hash files in bounded chunks, including potentially large predictions."""
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, payload):
    # Exclusive creation prevents silent replacement of partial or completed work.
    with Path(path).open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def seal(folder):
    save_json(folder / "complete.json", {p.name: digest(p) for p in sorted(folder.iterdir()) if p.is_file()})


def verify(folder):
    path = folder / "complete.json"
    if not path.is_file():
        raise ValueError(f"Incomplete task; preserve it and use a new output directory: {folder}")
    hashes = read_json(path)
    actual = {p.name for p in folder.iterdir() if p.is_file() and p.name != "complete.json"}
    if actual != set(hashes):
        raise ValueError(f"Task file list changed: {folder}")
    for name, expected in hashes.items():
        if digest(folder / name) != expected:
            raise ValueError(f"Artifact hash changed: {folder / name}")


def validate_config(cfg):
    if set(cfg) != set(read_json(DEFAULT_CONFIG)):
        raise ValueError("Configuration must have exactly the documented config.json keys")
    for name in ("seed", "selection_repeats", "convergence_repeats", "train_n", "validation_n",
                 "calibration_n", "evaluation_n", "max_epochs", "min_epochs", "patience",
                 "min_cell_count", "bootstrap_repetitions"):
        if type(cfg[name]) is not int or cfg[name] < (0 if name == "seed" else 1):
            raise ValueError(f"Invalid nonnegative seed or positive integer: {name}")
    if cfg["min_epochs"] > cfg["max_epochs"]:
        raise ValueError("min_epochs exceeds max_epochs")
    for name, allowed in (("cases", SETTINGS), ("losses", ("hellinger", "kl", "chisq", "js")),
                          ("architectures", RATIO_ARCHITECTURES), ("output_alphas", (1, 2, 3, 4))):
        if not cfg[name] or len(set(cfg[name])) != len(cfg[name]) or any(v not in allowed for v in cfg[name]):
            raise ValueError(f"Invalid or duplicated choices: {name}")
    if "hellinger" not in cfg["losses"]:
        raise ValueError("Hellinger is required for the prespecified final model")
    if not cfg["sample_sizes"] or any(type(n) is not int or n < 1 for n in cfg["sample_sizes"]) or len(set(cfg["sample_sizes"])) != len(cfg["sample_sizes"]):
        raise ValueError("sample_sizes must be unique positive integers")
    if type(cfg["adaptive"]) is not bool or not 0 < cfg["ci_alpha"] < 1:
        raise ValueError("Invalid adaptive flag or CI alpha")


def candidate_grid(cfg):
    """Four losses at baseline/alpha=2; confirm all Hellinger alpha x arch pairs."""
    grid = [{"loss": loss, "architecture": "baseline32", "output_alpha": 2}
            for loss in cfg["losses"]]
    grid += [{"loss": "hellinger", "architecture": arch, "output_alpha": alpha}
             for arch in cfg["architectures"] for alpha in cfg["output_alphas"]]
    return list({candidate_id(row): row for row in grid}.values())


def candidate_id(candidate):
    return f'{candidate["loss"]}__{candidate["architecture"]}__a{candidate["output_alpha"]}'


def seed_for(cfg, stage, case, repeat, role, source=0, n=0):
    """Explicit seed vectors make pairing and separation inspectable."""
    return [cfg["seed"], {"selection": 1, "convergence": 2}[stage],
            list(SETTINGS).index(case), repeat, ROLES[role], source, n if role == "train" else 0]


def samples(cfg, stage, case, repeat, n):
    population, arrays, seeds = Mixture(case), {}, {}
    for role in ("train", "validation", "calibration", "evaluation"):
        count = n if role == "train" else cfg[f"{role}_n"]
        for source_id, source in enumerate(("P", "Q")):
            key = f"{role}_{source}"
            seeds[key] = seed_for(cfg, stage, case, repeat, role, source_id, n)
            arrays[key] = population.draw(np.random.default_rng(np.random.SeedSequence(seeds[key])), count, source)
    return population, arrays, seeds


def score(model, x):
    parameter = next(model.parameters())
    with torch.inference_mode():
        return np.concatenate([model(torch.as_tensor(x[start:start+32768], dtype=parameter.dtype,
                                                    device=parameter.device)).cpu().numpy().ravel()
                               for start in range(0, len(x), 32768)]).astype(float)


def balanced_brier(p, q):
    return float(.5*np.mean((p/2-1)**2) + .5*np.mean((q/2)**2))


def cell_tables(predictions, cfg, seed):
    """Label-only calibration plus separately named finite-sample oracle checks.

    Neural and truth cell means use independent evaluation observations.
    CI containment of an estimated neural mean is descriptive: its own sampling
    uncertainty is not part of C.1/C.2. No Monte Carlo mean is asserted to be the
    exact population cell target, and no coverage rate is computed here.
    """
    kp, kq = [np.bincount(bin_scores(predictions[f"calibration_{s}"], EDGES), minlength=20) for s in ("P", "Q")]
    fixed = calibrate_counts(kp, kq, cfg["calibration_n"], cfg["calibration_n"], cfg["ci_alpha"])
    partitions = [("fixed", fixed, np.arange(20), np.arange(1, 21))]
    if cfg["adaptive"]:
        adaptive = calibrate_merged_counts(kp, kq, cfg["calibration_n"], cfg["calibration_n"],
            cfg["ci_alpha"], cfg["min_cell_count"], cfg["bootstrap_repetitions"], seed)
        chosen = adaptive["selected_indices"]
        cells = {key: value[chosen] for key, value in adaptive["candidates"].items()}
        partitions.append(("adaptive", cells, cells["start"], cells["stop"]))
    evaluation = np.concatenate([predictions[f"evaluation_{s}"] for s in ("P", "Q")])
    truth = np.concatenate([predictions[f"truth_evaluation_{s}"] for s in ("P", "Q")])
    bins, rows = bin_scores(evaluation, EDGES), []
    for partition, cells, starts, stops in partitions:
        for index, (start, stop) in enumerate(zip(starts, stops)):
            mask = (bins >= start) & (bins < stop)
            mean = float(evaluation[mask].mean()) if mask.any() else None
            row = {"partition": partition, "start": int(start), "stop": int(stop),
                   "left": float(EDGES[start]), "right": float(EDGES[stop]),
                   "count_p": int(kp[start:stop].sum()), "count_q": int(kq[start:stop].sum()),
                   "evaluation_count": int(mask.sum()), "evaluation_mass": float(mask.mean()),
                   "neural_mean": mean, "calibrated_rdr": float(cells["estimate"][index]),
                   "calibration_mass": float((kp[start:stop].sum()+kq[start:stop].sum())/(2*cfg["calibration_n"]))}
            for method in ("c1", "c2"):
                lower, upper = float(cells[f"{method}_lower"][index]), float(cells[f"{method}_upper"][index])
                row.update({f"{method}_lower": lower, f"{method}_upper": upper, f"{method}_width": upper-lower,
                            f"{method}_contains_neural_mean_descriptive": None if mean is None else bool(lower <= mean <= upper)})
            row["c1_unavailable"] = bool(cells["c1_unavailable"][index])
            empty_calibration = row["count_p"] + row["count_q"] == 0
            row["calibration_empty"] = empty_calibration
            if empty_calibration:
                # Shared CI code uses estimate=1 as a storage placeholder only.
                row["calibrated_rdr"] = None
            row["neural_minus_calibrated"] = None if empty_calibration or mean is None else mean-row["calibrated_rdr"]
            row["oracle_evaluation_true_mean"] = float(truth[mask].mean()) if mask.any() else None
            row["oracle_evaluation_true_variance"] = float(truth[mask].var()) if mask.any() else None
            rows.append(row)
    return rows


def source_paths():
    return sorted(set(ROOT.glob("utils/*.py")) | {Path(__file__).resolve(),
        Path(__file__).with_name("population.py"), DEFAULT_CONFIG,
        Path(__file__).with_name("requirements.txt"),
        Path(__file__).with_name("__init__.py"), ROOT / "experiments" / "__init__.py"})


def load_frozen(path):
    path = Path(path).resolve()
    expected = path.with_suffix(".sha256").read_text().strip()
    if digest(path) != expected:
        raise ValueError("Frozen selection changed")
    selection = read_json(path)
    if selection["selected"]["loss"] != "hellinger":
        raise ValueError("Frozen model must use Hellinger")
    return selection, expected


def verify_input_selection(output, protocol):
    """Check the retained selection dependency even when its original is gone."""
    if protocol["stage"] != "convergence":
        return
    path = output / "input_selection.json"
    if not path.is_file() or digest(path) != protocol["selection_sha256"]:
        raise ValueError("Copied input selection is missing or changed")
    selection = read_json(path)
    if (selection["selected"] != protocol["selected"]
            or protocol["candidates"] != [selection["selected"]]
            or any(selection[key] != protocol[key] for key in ("source_sha256", "device", "threads", "versions"))
            or any(value != selection["config"][key] for key, value in protocol["config"].items()
                   if key not in ("convergence_repeats", "sample_sizes"))):
        raise ValueError("Copied input selection disagrees with convergence protocol")


def prepare(output, cfg, stage, selection=None, device="cpu"):
    validate_config(cfg)
    output = Path(output).resolve()
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths()}
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "torch": torch.__version__,
                "matplotlib": importlib.metadata.version("matplotlib")}
    selected, selection_hash = load_frozen(selection) if selection else (None, None)
    if stage == "convergence" and selected is None:
        raise ValueError("Convergence requires --selection pointing to a frozen selection.json")
    if selected:
        original = selected["config"]
        for key in cfg:
            if key not in ("convergence_repeats", "sample_sizes") and cfg[key] != original[key]:
                raise ValueError(f"Convergence must retain selection protocol: {key}")
        if selected["source_sha256"] != hashes:
            raise ValueError("Convergence source differs from frozen selection; use its saved source snapshot")
        if selected["device"] != device or selected["threads"] != torch.get_num_threads():
            raise ValueError("Convergence must use the frozen selection's device/thread settings")
        if selected["versions"] != versions:
            raise ValueError("Convergence package versions differ from frozen selection")
    protocol = {"schema": 1, "stage": stage, "config": cfg, "source_sha256": hashes,
                "selection_sha256": selection_hash, "selected": selected["selected"] if selected else None,
                "candidates": [selected["selected"]] if selected else candidate_grid(cfg),
                "device": device, "threads": torch.get_num_threads(), "versions": versions,
                "target": "2p/(p+q)", "edges": EDGES.tolist(),
                "selection_rule": "smallest mean validation Brier across equally weighted settings/repeats; Hellinger constrained",
                "evaluation_role": "design diagnostics" if stage == "selection" else "final assessment of locked configuration"}
    if output.exists():
        if not (output / "protocol.json").is_file() or read_json(output / "protocol.json") != protocol:
            raise ValueError("Output exists with a partial/different protocol; use a new output directory")
        for name, expected in hashes.items():
            if digest(output / "source" / name) != expected:
                raise ValueError(f"Source snapshot changed: {name}")
    else:
        output.mkdir(parents=True)
        for path in source_paths():
            target = output / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        if selection:
            shutil.copy2(selection, output / "input_selection.json")
        save_json(output / "protocol.json", protocol)
    verify_input_selection(output, protocol)
    return output, protocol


def task_specs(protocol):
    cfg, stage = protocol["config"], protocol["stage"]
    sizes = [cfg["train_n"]] if stage == "selection" else cfg["sample_sizes"]
    return [(case, repeat, n) for case in cfg["cases"] for repeat in range(cfg[f"{stage}_repeats"]) for n in sizes]


def task_path(output, case, repeat, n, candidate):
    return output / "fits" / case / f"repeat{repeat:03d}" / f"n{n}" / candidate_id(candidate)


def verify_fit(output, protocol, task_index, case, repeat, n, candidate):
    """Bind an intact artifact to its planned experimental unit, not only bytes."""
    folder = task_path(output, case, repeat, n, candidate)
    verify(folder)
    metrics, training = read_json(folder / "metrics.json"), read_json(folder / "training.json")
    expected = {"case": case, "repeat": repeat, "train_n": n, "candidate": candidate_id(candidate), **candidate}
    if any(metrics.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Fit identity differs from planned task: {folder}")
    cfg, stage = protocol["config"], protocol["stage"]
    expected_seeds = {f"{role}_{source}": seed_for(cfg, stage, case, repeat, role, source_id, n)
                      for role in ("train", "validation", "calibration", "evaluation")
                      for source_id, source in enumerate(("P", "Q"))}
    initialization = seed_for(cfg, stage, case, repeat, "initialization")
    expected_training = {"task_index": task_index, "protocol_sha256": digest(output / "protocol.json"),
                         "data_seed_vectors": expected_seeds, "initialization_seed_vector": initialization,
                         "model_seed": int(np.random.SeedSequence(initialization).generate_state(1)[0])}
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError(f"Fit seed/protocol identity differs from planned task: {folder}")
    return metrics


def run_task(output, protocol, task_index):
    verify_input_selection(output, protocol)
    cfg, stage = protocol["config"], protocol["stage"]
    tasks = task_specs(protocol)
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"task-index must be in [0,{len(tasks)-1}]")
    case, repeat, n = tasks[task_index]
    population, arrays, seeds = samples(cfg, stage, case, repeat, n)
    init_seed = seed_for(cfg, stage, case, repeat, "initialization")
    model_seed = int(np.random.SeedSequence(init_seed).generate_state(1)[0])
    for candidate in protocol["candidates"]:
        folder = task_path(output, case, repeat, n, candidate)
        if folder.exists():
            verify_fit(output, protocol, task_index, case, repeat, n, candidate)
            continue
        folder.mkdir(parents=True)
        model = make_ratio_mlp(population.dimension, candidate["architecture"], candidate["output_alpha"], model_seed).to(protocol["device"])
        model, training, history = fit_ratio_mlp(model, arrays["train_P"], arrays["train_Q"],
            arrays["validation_P"], arrays["validation_Q"], loss=candidate["loss"],
            max_epochs=cfg["max_epochs"], min_epochs=cfg["min_epochs"], patience=cfg["patience"])
        predictions = {key: score(model, value) for key, value in arrays.items() if not key.startswith("train_")}
        for source in ("P", "Q"):
            predictions[f"truth_evaluation_{source}"] = population.truth(arrays[f"evaluation_{source}"])
        metrics = {"case": case, "repeat": repeat, "train_n": n, "candidate": candidate_id(candidate), **candidate,
                   "validation_brier": balanced_brier(predictions["validation_P"], predictions["validation_Q"]),
                   "evaluation_brier": balanced_brier(predictions["evaluation_P"], predictions["evaluation_Q"]),
                   "evaluation_mse": float(.5*sum(np.mean((predictions[f"evaluation_{s}"]-predictions[f"truth_evaluation_{s}"])**2) for s in ("P", "Q")))}
        save_json(folder / "metrics.json", metrics)
        save_json(folder / "training.json", {**training, "data_seed_vectors": seeds, "initialization_seed_vector": init_seed,
            "model_seed": model_seed, "task_index": task_index, "protocol_sha256": digest(output / "protocol.json")})
        save_json(folder / "history.json", history)
        save_json(folder / "cells.json", cell_tables(predictions, cfg, model_seed))
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, folder / "model.pt")
        np.savez_compressed(folder / "predictions.npz", **predictions)
        seal(folder)
        print(f"Completed {case} repeat={repeat} n={n} {candidate_id(candidate)}", flush=True)


def collect(output, protocol):
    verify_input_selection(output, protocol)
    for name, expected in protocol["source_sha256"].items():
        if digest(output / "source" / name) != expected:
            raise ValueError(f"Source snapshot changed: {name}")
    rows = []
    for task_index, (case, repeat, n) in enumerate(task_specs(protocol)):
        for candidate in protocol["candidates"]:
            rows.append(verify_fit(output, protocol, task_index, case, repeat, n, candidate))
    return rows


def freeze(output):
    output = Path(output).resolve()
    protocol = read_json(output / "protocol.json")
    if protocol["stage"] != "selection":
        raise ValueError("Only a selection run can be frozen")
    rows = collect(output, protocol)
    ranking = []
    for candidate in protocol["candidates"]:
        values = [row["validation_brier"] for row in rows if row["candidate"] == candidate_id(candidate)]
        ranking.append({**candidate, "mean_validation_brier": float(np.mean(values))})
    ranking.sort(key=lambda row: (row["mean_validation_brier"], candidate_id(row)))
    best = next(row for row in ranking if row["loss"] == "hellinger")
    selection = {"status": "locked", "selected": {key: best[key] for key in ("loss", "architecture", "output_alpha")},
                 "ranking": ranking, "config": protocol["config"], "protocol_sha256": digest(output / "protocol.json"),
                 "source_sha256": protocol["source_sha256"], "device": protocol["device"], "threads": protocol["threads"],
                 "versions": protocol["versions"],
                 "criterion": protocol["selection_rule"], "selection_uses": ["validation_brier"],
                 "ties": "lexicographic candidate id", "fit_count": len(rows)}
    selection["fit_manifest_sha256"] = {
        str((folder / "complete.json").relative_to(output)): digest(folder / "complete.json")
        for case, repeat, n in task_specs(protocol) for candidate in protocol["candidates"]
        for folder in [task_path(output, case, repeat, n, candidate)]}
    path = output / "selection.json"
    if path.exists():
        previous, _ = load_frozen(path)
        if previous != selection:
            raise ValueError("Existing frozen selection differs")
        return previous
    save_json(path, selection)
    with path.with_suffix(".sha256").open("x") as stream:
        stream.write(digest(path) + "\n")
    return selection


def local_diagnostics(output, rows):
    """Summarize label-only calibration, keeping unsupported mass visible.

    Absolute gaps contain calibration and evaluation sampling error. They are
    descriptive summaries, not estimators of a noise-free calibration norm.
    """
    summaries, tables = [], {}
    for row in rows:
        folder = task_path(output, row["case"], row["repeat"], row["train_n"], row)
        cells = read_json(folder / "cells.json")
        tables[(row["case"], row["repeat"], row["train_n"], row["candidate"])] = cells
        for partition in sorted({cell["partition"] for cell in cells}):
            selected = [cell for cell in cells if cell["partition"] == partition]
            evaluable = [cell for cell in selected if cell["neural_minus_calibrated"] is not None]
            mass = sum(cell["evaluation_mass"] for cell in evaluable)
            summaries.append({**{key: row[key] for key in ("case", "repeat", "train_n", "candidate")},
                "partition": partition, "evaluation_mass_with_estimable_gap": mass,
                "mass_weighted_absolute_gap": sum(cell["evaluation_mass"] * abs(cell["neural_minus_calibrated"])
                                                   for cell in evaluable)/mass if mass else None,
                "mass_weighted_c2_width": sum(cell["evaluation_mass"] * cell["c2_width"] for cell in selected)})
    return summaries, tables


def render_figures(output, protocol, rows, tables, candidate):
    """Use repeat zero for illustration; never select a panel by its errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    paths = []
    n = max(row["train_n"] for row in rows)
    for case in protocol["config"]["cases"]:
        cells = tables[(case, 0, n, candidate)]
        for partition in sorted({cell["partition"] for cell in cells}):
            selected = [cell for cell in cells if cell["partition"] == partition]
            column = lambda name: [cell[name] for cell in selected]
            fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
            for ax, method in zip(axes, ("c2", "c1")):
                plot_calibration_cells(column("neural_mean"), column("calibrated_rdr"),
                    column(f"{method}_lower"), column(f"{method}_upper"),
                    counts_p=column("count_p"), counts_q=column("count_q"),
                    masses=column("evaluation_mass"), ax=ax, title=f"{method.upper()} — {partition}")
            fig.suptitle(f"{case}; n={n}; repeat 0\n{candidate}", fontsize=11)
            path = destination / f"calibration_{case}_{partition}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(path.relative_to(output))
    if protocol["stage"] == "convergence":
        cases = protocol["config"]["cases"]
        fig, axes = plt.subplots(len(cases), 2, squeeze=False,
                                 figsize=(10, 3*len(cases)), constrained_layout=True)
        sizes = sorted(protocol["config"]["sample_sizes"])
        for case, axes_row in zip(cases, axes):
            for ax, metric in zip(axes_row, ("evaluation_mse", "evaluation_brier")):
                groups = [[row[metric] for row in rows if row["case"] == case and row["train_n"] == size]
                          for size in sizes]
                errors = [np.std(group, ddof=1)/np.sqrt(len(group)) if len(group)>1 else 0 for group in groups]
                ax.errorbar(sizes, [np.mean(group) for group in groups], yerr=errors, marker="o", capsize=3)
                ax.set(xscale="log", xlabel="Training count per distribution", ylabel=metric,
                       title=case, xticks=sizes)
                ax.set_xticklabels(sizes)
                ax.grid(alpha=.2)
        path = destination / "convergence.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.insert(0, path.relative_to(output))
    return paths


def report(output):
    """Require all planned fits; report Monte Carlo means and across-fit SEs."""
    output = Path(output).resolve()
    protocol = read_json(output / "protocol.json")
    rows = collect(output, protocol)
    local, tables = local_diagnostics(output, rows)
    lines = ["# Simulation results", "", f'Stage: {protocol["stage"]}. Fits: {len(rows)}.', "",
        "Selection uses validation Brier only. MSE and CI diagnostics from the selection stage are design evidence.",
        "C.1/C.2 target population cell-average RDR, not individual truth or the network's predictions.",
        "Neural-mean containment is descriptive because evaluation means have sampling uncertainty.", "",
        "| Setting | n per distribution | Model | Validation Brier | Evaluation Brier | Evaluation MSE |",
        "|---|---:|---|---:|---:|---:|"]
    for case, n, candidate in sorted({(row["case"], row["train_n"], row["candidate"]) for row in rows}):
        group = [row for row in rows if (row["case"], row["train_n"], row["candidate"]) == (case, n, candidate)]
        values = []
        for metric in ("validation_brier", "evaluation_brier", "evaluation_mse"):
            vector = np.array([row[metric] for row in group])
            values.append(f"{vector.mean():.6f}" + (f" ± {vector.std(ddof=1)/np.sqrt(len(vector)):.6f}" if len(vector)>1 else " (one fit)"))
        lines.append(f"| {case} | {n} | {candidate} | " + " | ".join(values) + " |")
    lines += ["", "Entries show mean ± standard error across independent fits. All CI tables are in each fit's cells.json.",
              "Fixed bins are primary; optional adaptive bins protect all 210 contiguous candidate regions.", "",
              "## Local calibration diagnostics", "",
              "The following fixed-bin summaries use P/Q labels and neural scores, without oracle truth.",
              "Absolute gaps include sampling error and are averaged over cells with an estimable gap;",
              "the corresponding evaluation mass is shown separately. C.2 width is weighted by evaluation mass over all cells.",
              "Gaps are evaluation-mass-weighted, then averaged over fits with an estimable gap.",
              "Mass and width average over all fits. These quantities do not enter model selection.", "",
              "| Setting | n | Model | Weighted absolute gap | Evaluable mass | C.2 width |",
              "|---|---:|---|---:|---:|---:|"]
    for case, n, candidate in sorted({(row["case"], row["train_n"], row["candidate"]) for row in rows}):
        group = [row for row in local if row["partition"] == "fixed" and
                 (row["case"], row["train_n"], row["candidate"]) == (case, n, candidate)]
        gaps = [row["mass_weighted_absolute_gap"] for row in group if row["mass_weighted_absolute_gap"] is not None]
        gap_text = f"{np.mean(gaps):.6f}" if gaps else "unavailable"
        lines.append(f"| {case} | {n} | {candidate} | {gap_text} | "
                     f"{np.mean([row['evaluation_mass_with_estimable_gap'] for row in group]):.4f} | "
                     f"{np.mean([row['mass_weighted_c2_width'] for row in group]):.4f} |")
    if protocol["stage"] == "selection":
        candidate = min((candidate_id(c) for c in protocol["candidates"] if c["loss"] == "hellinger"),
            key=lambda key: (np.mean([r["validation_brier"] for r in rows if r["candidate"] == key]), key))
    else:
        candidate = candidate_id(protocol["selected"])
    lines += ["", f"Panels use `{candidate}`: the validation-Brier-selected Hellinger configuration.",
              "Repeat zero and the largest training size are specified in advance for these illustrative panels.",
              "Marker sizes show evaluation cell mass; counts and individual intervals are retained in cells.json.",
              "C.1 may be unavailable in sparse cells. Neural-mean agreement with a band is descriptive.",
              "All fixed/adaptive per-fit summaries are in [calibration.csv](calibration.csv).", ""]
    figure_paths = ([Path("figures/convergence.png")] if protocol["stage"] == "convergence" else [])
    figure_paths += [Path(f"figures/calibration_{case}_{partition}.png") for case in protocol["config"]["cases"]
                     for partition in (["adaptive", "fixed"] if protocol["config"]["adaptive"] else ["fixed"])]
    lines += [f"![{path.stem}]({path.as_posix()})\n" for path in figure_paths]
    text = "\n".join(lines)
    path = output / "report.md"
    if path.exists() and path.read_text() != text:
        raise ValueError("Existing report differs; preserve it before regenerating")
    required = figure_paths + [Path("metrics.csv"), Path("calibration.csv")]
    if path.exists() and any(not (output / name).is_file() for name in required):
        raise ValueError("Existing report has missing derived artifacts; preserve it before regenerating")
    if not path.exists():
        render_figures(output, protocol, rows, tables, candidate)
        for name, data in (("metrics.csv", rows), ("calibration.csv", local)):
            with (output / name).open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
        with path.open("x") as stream:
            stream.write(text)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("selection", "freeze", "convergence", "report"))
    parser.add_argument("--output", type=Path, required=True, help="New output directory; partial artifacts are never overwritten")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--selection", type=Path, help="Frozen selection.json required for convergence")
    parser.add_argument("--task-index", type=int, help="Run one case/repeat/n task (all candidates); useful for Slurm arrays")
    parser.add_argument("--prepare-only", action="store_true", help="Freeze protocol/source before submitting parallel tasks")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--threads", default=1, type=int)
    parser.add_argument("--smoke", action="store_true", help="Tiny plumbing check, not scientific evidence")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.stage in ("freeze", "report"):
        result = freeze(args.output) if args.stage == "freeze" else report(args.output)
        print(result)
        return
    cfg = read_json(args.config)
    if args.smoke:
        cfg.update(cases=["D2_noise0.00"], selection_repeats=1, convergence_repeats=1,
                   architectures=["baseline32"], output_alphas=[1, 2], train_n=24,
                   validation_n=20, calibration_n=40, evaluation_n=40, sample_sizes=[12, 24],
                   max_epochs=3, min_epochs=1, patience=2, min_cell_count=3, bootstrap_repetitions=100)
    output, protocol = prepare(args.output, cfg, args.stage, args.selection, args.device)
    if args.prepare_only:
        print(f"Prepared {len(task_specs(protocol))} tasks in {output}")
        return
    indices = range(len(task_specs(protocol))) if args.task_index is None else [args.task_index]
    for index in indices:
        run_task(output, protocol, index)


if __name__ == "__main__":
    main()
