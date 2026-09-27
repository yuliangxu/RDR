"""Fresh, paired sample-size assessment of four losses at a frozen architecture.

The original model selection locks the architecture and output activation. This
follow-up varies only the training size and loss. It retains the original
optimizer, loss-specific validation checkpoint rule, and independent data roles.
All fits use the existing convergence random namespace, distinct from selection.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.simulations import workflow as w

LOSSES = ("hellinger", "kl", "chisq", "js")
COMPLETE = "CONVERGENCE_COMPLETE.json"
REFERENCE_RECORDS = ("protocol.json", "selection.json", "selection.sha256")
CONFIG_OVERRIDES = ("convergence_repeats", "sample_sizes")


def scientific_sources():
    """The original scientific closure is unchanged by this orchestration."""
    return {str(path.relative_to(ROOT)): w.digest(path) for path in w.source_paths()}


def convergence_sources():
    paths = [*w.source_paths(), Path(__file__).resolve(),
             Path(__file__).with_name("convergence_report.py"),
             Path(__file__).with_name("report.py")]
    return {str(path.relative_to(ROOT)): w.digest(path) for path in paths}


def versions():
    return {"python": w.platform.python_version(), "numpy": np.__version__,
            "scipy": w.scipy.__version__, "torch": torch.__version__,
            "matplotlib": w.importlib.metadata.version("matplotlib")}


def read_reference(reference):
    """Bind the locked choice to its original complete selection manifest."""
    original = w.read_json(reference / "protocol.json")
    selection, checksum = w.load_frozen(reference / "selection.json")
    if (original["stage"] != "selection" or selection.get("status") != "locked"
            or selection["protocol_sha256"] != w.digest(reference / "protocol.json")):
        raise ValueError("Reference is not a matching frozen selection study")
    for key in ("config", "source_sha256", "device", "threads", "versions"):
        if selection[key] != original[key]:
            raise ValueError(f"Reference selection/protocol disagree: {key}")
    w.validate_config(original["config"])
    if original["candidates"] != w.candidate_grid(original["config"]):
        raise ValueError("Reference selection candidate grid changed")
    expected = min((row for row in selection["ranking"] if row["loss"] == "hellinger"),
                   key=lambda row: (row["mean_validation_brier"], w.candidate_id(row)))
    if selection["selected"] != {key: expected[key] for key in ("loss", "architecture", "output_alpha")}:
        raise ValueError("Reference candidate differs from frozen Hellinger ranking")
    manifests = {str((w.task_path(reference, case, repeat, n, candidate) / "complete.json").relative_to(reference))
                 for case, repeat, n in w.task_specs(original) for candidate in original["candidates"]}
    if set(selection["fit_manifest_sha256"]) != manifests or selection["fit_count"] != len(manifests):
        raise ValueError("Frozen selection does not bind its complete planned fit grid")
    return original, selection, checksum


def verify_reference(output, protocol):
    reference = output / "reference"
    if set(protocol["reference_sha256"]) != set(REFERENCE_RECORDS):
        raise ValueError("Convergence reference record list changed")
    for name, expected in protocol["reference_sha256"].items():
        path = reference / name
        if not path.is_file() or w.digest(path) != expected:
            raise ValueError(f"Convergence reference changed or is missing: {name}")
    original, selection, checksum = read_reference(reference)
    candidates = [{**selection["selected"], "loss": loss} for loss in LOSSES]
    if (protocol["stage"] != "convergence" or protocol["selected"] != selection["selected"]
            or protocol["candidates"] != candidates or protocol["selection_sha256"] != checksum
            or protocol["reference_protocol_sha256"] != selection["protocol_sha256"]
            or any(protocol[key] != original[key] for key in ("source_sha256", "device", "threads", "versions"))
            or any(value != original["config"][key] for key, value in protocol["config"].items()
                   if key not in CONFIG_OVERRIDES)):
        raise ValueError("Convergence protocol differs from its frozen reference")
    w.validate_config(protocol["config"])


def verify_execution(output, protocol):
    """Verify both the executing code and retained snapshot before any fit."""
    if (not (output / "protocol.sha256").is_file()
            or (output / "protocol.sha256").read_text().strip() != w.digest(output / "protocol.json")):
        raise ValueError("Prepared convergence protocol changed")
    if (scientific_sources() != protocol["source_sha256"]
            or convergence_sources() != protocol["convergence_source_sha256"]):
        raise ValueError("Executing source differs from prepared convergence; use its frozen snapshot")
    for name, expected in protocol["convergence_source_sha256"].items():
        path = output / "source" / name
        if not path.is_file() or w.digest(path) != expected:
            raise ValueError(f"Convergence source snapshot changed: {name}")
    if versions() != protocol["versions"] or torch.get_num_threads() != protocol["threads"]:
        raise ValueError("Execution versions/threads differ from original selection")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Convergence requires deterministic Torch algorithms")
    verify_reference(output, protocol)


def prepare(reference, output, repeats=None, sample_sizes=None, device="cpu"):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Convergence output must be a new directory")
    original, selection, selected_hash = read_reference(reference)
    if scientific_sources() != original["source_sha256"]:
        raise ValueError("Scientific source differs from frozen selection")
    if (versions() != original["versions"] or torch.get_num_threads() != original["threads"]
            or device != original["device"]):
        raise ValueError("Convergence execution environment differs from frozen selection")
    if not torch.are_deterministic_algorithms_enabled():
        raise ValueError("Convergence requires deterministic Torch algorithms")
    for name, expected in original["source_sha256"].items():
        path = reference / "source" / name
        if not path.is_file() or w.digest(path) != expected:
            raise ValueError(f"Original selection source snapshot changed: {name}")
    cfg = copy.deepcopy(original["config"])
    if repeats is not None:
        cfg["convergence_repeats"] = repeats
    if sample_sizes is not None:
        cfg["sample_sizes"] = list(sample_sizes)
    w.validate_config(cfg)
    protocol = {"schema": 1, "stage": "convergence", "config": cfg,
                "selected": selection["selected"],
                "candidates": [{**selection["selected"], "loss": loss} for loss in LOSSES],
                "source_sha256": original["source_sha256"],
                "convergence_source_sha256": convergence_sources(),
                "reference_sha256": {name: w.digest(reference / name) for name in REFERENCE_RECORDS},
                "selection_sha256": selected_hash,
                "reference_protocol_sha256": selection["protocol_sha256"],
                "device": device, "threads": torch.get_num_threads(), "versions": versions(),
                "target": "2p/(p+q)", "edges": w.EDGES.tolist(),
                "design": "User-requested four-loss convergence at the frozen selected architecture and activation",
                "selection_rule": "No new selection; freeze architecture/activation, compare all four losses",
                "evaluation_role": "fresh assessment of the locked four-loss design; independent convergence streams",
                "pairing": "Within case/repeat, initialization and validation/calibration/evaluation draws are shared across losses and sizes; training draws are shared across losses and independently seeded by size",
                "checkpoint_rule": "Each loss restores its own minimum validation-loss checkpoint; unchanged original optimizer and stopping settings",
                "sample_size_scope": "Only training P/Q counts vary; validation/calibration/evaluation counts remain fixed"}
    output.mkdir(parents=True)
    (output / "reference").mkdir()
    for name in REFERENCE_RECORDS:
        shutil.copy2(reference / name, output / "reference" / name)
    for name in protocol["convergence_source_sha256"]:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    w.save_json(output / "protocol.json", protocol)
    with (output / "protocol.sha256").open("x") as stream:
        stream.write(w.digest(output / "protocol.json") + "\n")
    verify_execution(output, protocol)
    return output, protocol


def run_task(output, task_index):
    """Fit four paired losses with the unchanged original scientific steps.

    The original workflow's convergence guard intentionally accepts Hellinger
    alone. This explicit loop adds four-loss orchestration while reusing its
    sampling, model/training, metrics, cells, and artifact verification routines.
    """
    output = Path(output).resolve()
    protocol = w.read_json(output / "protocol.json")
    verify_execution(output, protocol)
    cfg, stage = protocol["config"], protocol["stage"]
    tasks = w.task_specs(protocol)
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"task-index must be in [0,{len(tasks)-1}]")
    case, repeat, n = tasks[task_index]
    population, arrays, seeds = w.samples(cfg, stage, case, repeat, n)
    init_seed = w.seed_for(cfg, stage, case, repeat, "initialization")
    model_seed = int(np.random.SeedSequence(init_seed).generate_state(1)[0])
    for candidate in protocol["candidates"]:
        folder = w.task_path(output, case, repeat, n, candidate)
        if folder.exists():
            w.verify_fit(output, protocol, task_index, case, repeat, n, candidate)
            continue
        folder.mkdir(parents=True)
        model = w.make_ratio_mlp(population.dimension, candidate["architecture"], candidate["output_alpha"], model_seed).to(protocol["device"])
        model, training, history = w.fit_ratio_mlp(model, arrays["train_P"], arrays["train_Q"],
            arrays["validation_P"], arrays["validation_Q"], loss=candidate["loss"],
            max_epochs=cfg["max_epochs"], min_epochs=cfg["min_epochs"], patience=cfg["patience"])
        predictions = {key: w.score(model, value) for key, value in arrays.items() if not key.startswith("train_")}
        for source in ("P", "Q"):
            predictions[f"truth_evaluation_{source}"] = population.truth(arrays[f"evaluation_{source}"])
        metrics = {"case": case, "repeat": repeat, "train_n": n, "candidate": w.candidate_id(candidate), **candidate,
                   "validation_brier": w.balanced_brier(predictions["validation_P"], predictions["validation_Q"]),
                   "evaluation_brier": w.balanced_brier(predictions["evaluation_P"], predictions["evaluation_Q"]),
                   "evaluation_mse": float(.5*sum(np.mean((predictions[f"evaluation_{s}"]-predictions[f"truth_evaluation_{s}"])**2) for s in ("P", "Q")))}
        w.save_json(folder / "metrics.json", metrics)
        w.save_json(folder / "training.json", {**training, "data_seed_vectors": seeds, "initialization_seed_vector": init_seed,
            "model_seed": model_seed, "task_index": task_index, "protocol_sha256": w.digest(output / "protocol.json")})
        w.save_json(folder / "history.json", history)
        w.save_json(folder / "cells.json", w.cell_tables(predictions, cfg, model_seed))
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, folder / "model.pt")
        np.savez_compressed(folder / "predictions.npz", **predictions)
        w.seal(folder)
        print(f"Completed {case} repeat={repeat} n={n} {w.candidate_id(candidate)}", flush=True)


def collect(output):
    """Return only a complete verified four-loss grid and its local summaries."""
    output = Path(output).resolve()
    protocol = w.read_json(output / "protocol.json")
    verify_execution(output, protocol)
    rows = []
    for index, (case, repeat, n) in enumerate(w.task_specs(protocol)):
        for candidate in protocol["candidates"]:
            rows.append(w.verify_fit(output, protocol, index, case, repeat, n, candidate))
    # Avoid retaining 240,000 cell dictionaries when rendering the full study.
    local = [summary for row in rows for summary in w.local_diagnostics(output, [row])[0]]
    return protocol, rows, local


def report(output):
    """Publish a completion marker only after every fit and report is verified."""
    from experiments.simulations import convergence_report

    output = Path(output).resolve()
    protocol, rows, local = collect(output)
    manifests = {str((folder / "complete.json").relative_to(output)): w.digest(folder / "complete.json")
                 for case, repeat, n in w.task_specs(protocol) for candidate in protocol["candidates"]
                 for folder in [w.task_path(output, case, repeat, n, candidate)]}
    identity = {"status": "complete", "fit_count": len(rows), "task_count": len(w.task_specs(protocol)),
                "protocol_sha256": w.digest(output / "protocol.json"),
                "selection_sha256": protocol["selection_sha256"], "fit_manifest_sha256": manifests}
    marker = output / COMPLETE
    if marker.exists():
        previous = w.read_json(marker)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing convergence completion identity differs")
        for name, expected in previous["artifact_sha256"].items():
            if not (output / name).is_file() or w.digest(output / name) != expected:
                raise ValueError(f"Completed convergence report artifact changed: {name}")
        return output / "report.md"
    artifacts = convergence_report.render(output, protocol, rows, local)
    if "report.md" not in artifacts:
        raise ValueError("Convergence renderer did not return a report")
    for name, expected in artifacts.items():
        path = output / name
        if not path.is_file() or w.digest(path) != expected:
            raise ValueError(f"Rendered convergence artifact hash differs: {name}")
    w.save_json(marker, {**identity, "artifact_sha256": artifacts})
    return output / "report.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Completed original frozen selection run")
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--repeats", type=int, help="Override only the convergence repetition count")
    parser.add_argument("--sample-sizes", type=int, nargs="+", help="Training counts per distribution")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.command == "prepare":
        if args.reference is None:
            parser.error("prepare requires --reference")
        output, protocol = prepare(args.reference, args.output, args.repeats, args.sample_sizes, args.device)
        print(f"Prepared {len(w.task_specs(protocol))} tasks / {4*len(w.task_specs(protocol))} fits in {output}")
    elif args.command == "fit":
        if args.task_index is None:
            parser.error("fit requires --task-index")
        if args.reference is not None or args.repeats is not None or args.sample_sizes is not None:
            parser.error("fit uses its frozen protocol; prepare options cannot be supplied")
        run_task(args.output, args.task_index)
    else:
        if args.reference is not None or args.repeats is not None or args.sample_sizes is not None:
            parser.error("report uses its frozen protocol; prepare options cannot be supplied")
        print(report(args.output))


if __name__ == "__main__":
    main()
