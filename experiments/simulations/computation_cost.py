"""Measure fit cost by replaying a prespecified subset of convergence fits.

Timing is separate from the frozen scientific experiment. Each CPU replay uses
its original samples and initialization. Publication requires exact equality of
parameters, optimization history, and training metadata to the completed study.
Wall and process CPU timers cover fit_ratio_mlp only, including validation and
checkpoint restoration, excluding sampling, initialization, diagnostics and I/O.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.simulations import convergence as c
from experiments.simulations import workflow as w

COMPLETE = "COMPUTATION_COST_COMPLETE.json"
WARMUP = {"epochs": 1, "samples_per_distribution": 8, "seed": 0,
          "losses": list(c.LOSSES), "timed": False}
TIMING_SCOPE = ("Wall and process CPU seconds around fit_ratio_mlp, including input tensor "
                "conversion, optimizer creation, validation and best-checkpoint restoration; "
                "excluding process startup, imports, warm-up, sampling, model initialization, "
                "evaluation, calibration diagnostics and artifact I/O")


def canonical_json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def parameter_digest(state):
    """Hash sorted tensor names, dtypes, shapes and bytes, independent of torch.save."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        header = json.dumps([name, array.dtype.str, list(array.shape)],
                            separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def sources():
    paths = set(c.convergence_sources()) | {str(Path(__file__).resolve().relative_to(ROOT))}
    renderer = Path(__file__).with_name("cost_report.py")
    paths.add(str(renderer.relative_to(ROOT)))
    return {name: w.digest(ROOT / name) for name in sorted(paths)}


def task_specs(protocol):
    return w.task_specs({"stage": "convergence", "config": protocol["config"]})


def hardware():
    model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {"cpu_model": model, "platform": platform.platform(), "node": socket.gethostname(),
            "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "torch_threads": torch.get_num_threads(),
            "thread_environment": {key: os.environ.get(key) for key in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "slurm": {key: os.environ.get(key) for key in
                      ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_CPUS_PER_TASK")}}


def verify_execution(output, protocol):
    output = Path(output)
    if (not (output / "protocol.sha256").is_file()
            or (output / "protocol.sha256").read_text().strip() != w.digest(output / "protocol.json")):
        raise ValueError("Prepared computation-cost protocol changed")
    if sources() != protocol["cost_source_sha256"]:
        raise ValueError("Executing source differs from prepared computation cost; use its frozen snapshot")
    for name, checksum in protocol["cost_source_sha256"].items():
        path = output / "source" / name
        if not path.is_file() or w.digest(path) != checksum:
            raise ValueError(f"Computation-cost source snapshot changed: {name}")
    reference = Path(protocol["reference_path"])
    original = w.read_json(reference / "protocol.json")
    if (w.digest(reference / "protocol.json") != protocol["reference_protocol_sha256"]
            or original != protocol["reference_protocol"]):
        raise ValueError("Original convergence protocol changed")
    for name, checksum in protocol["reference_sha256"].items():
        path = output / "reference" / name
        if not path.is_file() or w.digest(path) != checksum:
            raise ValueError(f"Copied reference record changed: {name}")
    if protocol["reference_sha256"] != {"protocol.json": protocol["reference_protocol_sha256"],
                                        "protocol.sha256": w.digest(reference / "protocol.sha256")}:
        raise ValueError("Reference record list differs from convergence")
    c.verify_execution(reference, original)
    expected_cfg = copy.deepcopy(original["config"])
    repeats = protocol["repeats"]
    if type(repeats) is not int or not 1 <= repeats <= original["config"]["convergence_repeats"]:
        raise ValueError("Cost repetition count must be a subset of original repetitions")
    expected_cfg["convergence_repeats"] = repeats
    if (protocol["stage"] != "computation_cost" or protocol["config"] != expected_cfg
            or protocol["candidates"] != original["candidates"]
            or protocol["device"] != "cpu" or original["device"] != "cpu"
            or protocol["threads"] != original["threads"] or protocol["versions"] != original["versions"]
            or protocol["warmup"] != WARMUP or protocol["timing_scope"] != TIMING_SCOPE
            or protocol["execution_order"] != "Rotate original four-loss order by task_index modulo 4"):
        raise ValueError("Cost design differs from frozen convergence or documented timing recipe")


def prepare(reference, output, repeats=10):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Computation-cost output must be a new directory")
    original = w.read_json(reference / "protocol.json")
    c.verify_execution(reference, original)
    if original["device"] != "cpu":
        raise ValueError("Computation-cost replay currently supports CPU only")
    if type(repeats) is not int or not 1 <= repeats <= original["config"]["convergence_repeats"]:
        raise ValueError("Cost repetition count must be a subset of original repetitions")
    cfg = copy.deepcopy(original["config"])
    cfg["convergence_repeats"] = repeats
    protocol = {"schema": 1, "stage": "computation_cost", "config": cfg, "repeats": repeats,
                "candidates": original["candidates"], "reference_path": str(reference),
                "reference_protocol": original, "reference_protocol_sha256": w.digest(reference / "protocol.json"),
                "reference_sha256": {name: w.digest(reference / name) for name in ("protocol.json", "protocol.sha256")},
                "cost_source_sha256": sources(), "device": "cpu", "threads": original["threads"],
                "versions": original["versions"], "warmup": WARMUP, "timing_scope": TIMING_SCOPE,
                "execution_order": "Rotate original four-loss order by task_index modulo 4",
                "repetition_rule": "First fixed repetitions 0 through repeats-1; same convergence samples and initialization",
                "evidence_role": "Matched timing replays, not additional independent scientific repetitions"}
    output.mkdir(parents=True)
    (output / "reference").mkdir()
    for name in protocol["reference_sha256"]:
        shutil.copy2(reference / name, output / "reference" / name)
    for name in protocol["cost_source_sha256"]:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    w.save_json(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(w.digest(output / "protocol.json") + "\n")
    verify_execution(output, protocol)
    return output, protocol


def warmup(dimension, candidate):
    """Initialize lazy optimizer/loss machinery before any measured fit."""
    rng = np.random.default_rng(WARMUP["seed"])
    arrays = [rng.normal(size=(WARMUP["samples_per_distribution"], dimension)) for _ in range(4)]
    for loss in c.LOSSES:
        model = w.make_ratio_mlp(dimension, candidate["architecture"], candidate["output_alpha"], WARMUP["seed"])
        w.fit_ratio_mlp(model, *arrays, loss=loss, max_epochs=1, min_epochs=1, patience=1)


def seal(folder):
    """Publish complete.json atomically, without overwriting an existing marker."""
    temporary = folder / ".complete.pending"
    w.save_json(temporary, {"timing.json": w.digest(folder / "timing.json")})
    os.link(temporary, folder / "complete.json")
    temporary.unlink()


def expected_identity(output, protocol, index, case, repeat, n, candidate):
    cfg = protocol["reference_protocol"]["config"]
    initialization = w.seed_for(cfg, "convergence", case, repeat, "initialization")
    order = protocol["candidates"][index % 4:] + protocol["candidates"][:index % 4]
    return {"case": case, "repeat": repeat, "train_n": n, "candidate": w.candidate_id(candidate), **candidate,
            "task_index": index, "protocol_sha256": w.digest(Path(output) / "protocol.json"),
            "reference_protocol_sha256": protocol["reference_protocol_sha256"],
            "model_seed": int(np.random.SeedSequence(initialization).generate_state(1)[0]),
            "initialization_seed_vector": initialization,
            "data_seed_vectors": {f"{role}_{source}": w.seed_for(cfg, "convergence", case, repeat, role, source_id, n)
                                  for role in ("train", "validation", "calibration", "evaluation")
                                  for source_id, source in enumerate(("P", "Q"))},
            "warmup": WARMUP, "execution_order": [row["loss"] for row in order]}


def verify_timing(output, protocol, index, case, repeat, n, candidate):
    folder = w.task_path(output, case, repeat, n, candidate)
    w.verify(folder)
    if set(w.read_json(folder / "complete.json")) != {"timing.json"}:
        raise ValueError("Unexpected timing artifact list")
    row = w.read_json(folder / "timing.json")
    expected = expected_identity(output, protocol, index, case, repeat, n, candidate)
    if any(row.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Timing seed/protocol identity differs: {folder}")
    if any(not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] <= 0
           for key in ("wall_seconds", "cpu_seconds")):
        raise ValueError("Timing values must be finite and positive")
    if any(row.get(key) != row["training"].get(key) for key in ("epochs", "best_epoch", "n_parameters", "loss")):
        raise ValueError("Timing training metadata differs")
    if row["hardware"]["torch_threads"] != protocol["threads"]:
        raise ValueError("Timing worker thread count differs")
    return row


def run_task(output, task_index):
    output = Path(output).resolve()
    protocol = w.read_json(output / "protocol.json")
    verify_execution(output, protocol)
    tasks = task_specs(protocol)
    if not 0 <= task_index < len(tasks):
        raise ValueError(f"task-index must be in [0,{len(tasks)-1}]")
    case, repeat, n = tasks[task_index]
    cfg = protocol["reference_protocol"]["config"]
    population, arrays, _ = w.samples(cfg, "convergence", case, repeat, n)
    warmup(population.dimension, protocol["candidates"][0])
    order = protocol["candidates"][task_index % 4:] + protocol["candidates"][:task_index % 4]
    worker = hardware()
    for candidate in order:
        folder = w.task_path(output, case, repeat, n, candidate)
        if folder.exists():
            verify_timing(output, protocol, task_index, case, repeat, n, candidate)
            continue
        folder.mkdir(parents=True)
        identity = expected_identity(output, protocol, task_index, case, repeat, n, candidate)
        model = w.make_ratio_mlp(population.dimension, candidate["architecture"], candidate["output_alpha"], identity["model_seed"])
        wall_start, cpu_start = time.perf_counter(), time.process_time()
        model, training, history = w.fit_ratio_mlp(model, arrays["train_P"], arrays["train_Q"],
            arrays["validation_P"], arrays["validation_Q"], loss=candidate["loss"],
            max_epochs=cfg["max_epochs"], min_epochs=cfg["min_epochs"], patience=cfg["patience"])
        cpu_seconds, wall_seconds = time.process_time() - cpu_start, time.perf_counter() - wall_start
        row = {**identity, "wall_seconds": wall_seconds, "cpu_seconds": cpu_seconds,
               **{key: training[key] for key in ("epochs", "best_epoch", "n_parameters")},
               "training": training, "hardware": worker, "model_sha256": parameter_digest(model.state_dict()),
               "history_sha256": canonical_json_digest(history)}
        w.save_json(folder / "timing.json", row)
        seal(folder)
        print(f"Timed {case} repeat={repeat} n={n} {candidate['loss']}: {wall_seconds:.3f} wall s, {cpu_seconds:.3f} CPU s", flush=True)


def verify_reference_complete(protocol):
    """Verify the full original completion identity before publishing timing results."""
    reference = Path(protocol["reference_path"])
    marker_path = reference / c.COMPLETE
    if not marker_path.is_file():
        raise ValueError("Original convergence study is not complete; timing report cannot be published")
    marker = w.read_json(marker_path)
    original = protocol["reference_protocol"]
    tasks = w.task_specs(original)
    manifests = {str((w.task_path(reference, case, repeat, n, candidate) / "complete.json").relative_to(reference))
                 for case, repeat, n in tasks for candidate in original["candidates"]}
    if (marker.get("status") != "complete" or marker.get("protocol_sha256") != protocol["reference_protocol_sha256"]
            or marker.get("selection_sha256") != original["selection_sha256"]
            or marker.get("task_count") != len(tasks) or marker.get("fit_count") != len(manifests)
            or set(marker.get("fit_manifest_sha256", {})) != manifests):
        raise ValueError("Original convergence completion identity differs from its complete grid")
    if "report.md" not in marker.get("artifact_sha256", {}):
        raise ValueError("Original convergence completion has no report artifact")
    for name, expected in marker["artifact_sha256"].items():
        if not (reference / name).is_file() or w.digest(reference / name) != expected:
            raise ValueError(f"Original convergence report artifact changed: {name}")
    return marker


def verify_matched_fit(protocol, row, marker=None):
    """Check one replay against an intact original fit, also useful for preflight.

    A production report additionally passes the verified full-study marker. A
    preflight may compare already finished fits while the study is still running.
    This helper alone does not establish that the full study is complete.
    """
    reference = Path(protocol["reference_path"])
    original = protocol["reference_protocol"]
    case, repeat, n = row["case"], row["repeat"], row["train_n"]
    index = w.task_specs(original).index((case, repeat, n))
    candidate = {key: row[key] for key in ("loss", "architecture", "output_alpha")}
    if candidate not in original["candidates"]:
        raise ValueError("Timing candidate does not belong to original convergence")
    folder = w.task_path(reference, case, repeat, n, candidate)
    relative = str((folder / "complete.json").relative_to(reference))
    if marker is not None and w.digest(folder / "complete.json") != marker["fit_manifest_sha256"][relative]:
        raise ValueError(f"Original fit completion manifest changed: {relative}")
    w.verify_fit(reference, original, index, case, repeat, n, candidate)
    training = w.read_json(folder / "training.json")
    metadata_keys = ("task_index", "protocol_sha256", "data_seed_vectors", "initialization_seed_vector", "model_seed")
    actual_training = {key: value for key, value in training.items() if key not in metadata_keys}
    if (actual_training != row["training"]
            or any(training[key] != row[key] for key in metadata_keys[2:])
            or canonical_json_digest(w.read_json(folder / "history.json")) != row["history_sha256"]
            or parameter_digest(torch.load(folder / "model.pt", map_location="cpu", weights_only=True)) != row["model_sha256"]):
        raise ValueError(f"Timing replay does not exactly match original fitted model/history/metadata: {folder}")


def collect(output):
    """Return complete replay rows only after exact matched-production verification."""
    output = Path(output).resolve()
    protocol = w.read_json(output / "protocol.json")
    verify_execution(output, protocol)
    rows = [verify_timing(output, protocol, index, case, repeat, n, candidate)
            for index, (case, repeat, n) in enumerate(task_specs(protocol)) for candidate in protocol["candidates"]]
    marker = verify_reference_complete(protocol)
    for row in rows:
        verify_matched_fit(protocol, row, marker)
    return protocol, rows


def report(output):
    from experiments.simulations import cost_report
    output = Path(output).resolve()
    protocol, rows = collect(output)
    identity = {"status": "complete", "fit_count": len(rows), "task_count": len(task_specs(protocol)),
                "protocol_sha256": w.digest(output / "protocol.json"),
                "reference_completion_sha256": w.digest(Path(protocol["reference_path"]) / c.COMPLETE),
                "fit_manifest_sha256": {str((folder / "complete.json").relative_to(output)): w.digest(folder / "complete.json")
                    for case, repeat, n in task_specs(protocol) for candidate in protocol["candidates"]
                    for folder in [w.task_path(output, case, repeat, n, candidate)]}}
    marker = output / COMPLETE
    if marker.exists():
        previous = w.read_json(marker)
        if any(previous.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing computation-cost completion identity differs")
        artifacts = previous["artifact_sha256"]
    else:
        artifacts = cost_report.render(output, protocol, rows)
    if "report.md" not in artifacts:
        raise ValueError("Cost renderer did not return report.md")
    for name, expected in artifacts.items():
        if not (output / name).is_file() or w.digest(output / name) != expected:
            raise ValueError(f"Computation-cost report artifact changed: {name}")
    if not marker.exists():
        w.save_json(marker, {**identity, "artifact_sha256": artifacts})
    return output / "report.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit", "verify", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.command == "prepare":
        if args.reference is None:
            parser.error("prepare requires --reference")
        output, protocol = prepare(args.reference, args.output, 10 if args.repeats is None else args.repeats)
        print(f"Prepared {len(task_specs(protocol))} tasks / {4*len(task_specs(protocol))} timing replays in {output}")
    else:
        if args.reference is not None or args.repeats is not None:
            parser.error("fit/verify/report use the frozen cost protocol")
        if args.command == "fit":
            if args.task_index is None:
                parser.error("fit requires --task-index")
            run_task(args.output, args.task_index)
        elif args.command == "verify":
            _, rows = collect(args.output)
            print(f"Verified {len(rows)} timing replays exactly match the original convergence fits")
        else:
            print(report(args.output))


if __name__ == "__main__":
    main()
