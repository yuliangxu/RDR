"""Opt-in retrospective AGP threshold experiment. Run from the repository root with -m."""

import argparse
import csv
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from experiments.AGP import four_generator_metrics as frozen
from experiments.AGP import mask_metrics, relu_threshold_control as control
from experiments.AGP import selected_workflow as published
from experiments.AGP.relu_threshold_model import ThresholdRDR, balanced_brier_rdr
from utils.model_selection import balanced_brier


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARMS = ("threshold", "control")
METHODS = tuple(frozen.ROOTS)
CONFIG = HERE / "relu_threshold_config.json"
SOURCES = [HERE / f for f in ("relu_threshold_experiment.py", "relu_threshold_model.py",
           "relu_threshold_control.py", "selected_workflow.py", "four_generator_metrics.py",
           "mask_metrics.py", "rdr_metric_utils.py")]
SOURCES += [REPO / "utils" / f for f in ("losses.py", "model_selection.py", "calibration.py")]


def prepare(out, config):
    cfg = frozen.read(config)
    if not 0 < cfg["a"] < 1 < cfg["b"] < 2 or cfg["gradient_chunk_rows"] < 1:
        raise ValueError("Invalid bounds or gradient chunk size")
    if not 1 <= cfg["min_epochs"] <= cfg["max_epochs"]:
        raise ValueError("Invalid training epoch range")
    sources = {str(path.relative_to(REPO)): frozen.sha(path) for path in SOURCES}
    inputs = {}
    for method, root in frozen.ROOTS.items():
        data = frozen.metadata(root)
        metadata = frozen.read(root / "protocol.json")
        frozen.checked(root / "config.json", metadata["config_sha256"])
        old_cfg = frozen.read(root / "config.json")
        if old_cfg["repeats"] != 5:
            raise ValueError("Expected five frozen final repeats")
        complete = frozen.read(root / "COMPLETE.json")
        tasks = []
        for repeat in range(5):
            relative = f"final/{repeat:02d}/COMPLETE.json"
            frozen.checked(root / relative, complete["assessment_completions"][relative])
            task = frozen.read(root / relative)["task"]
            if task["repeat"] != repeat:
                raise ValueError("Unexpected archived repeat")
            tasks.append(task)
        inputs[method] = dict(root=str(root), arrays=data["studies"]["final"]["arrays"],
                              manifest=data["studies"]["final"]["manifest"], tasks=tasks,
                              config=old_cfg, taxa=data["taxa"],
                              root_complete_sha256=frozen.sha(root/"COMPLETE.json"))
    for key in ("train_p", "stopping_p", "test_p"):
        if len({spec["arrays"][key]["sha256"] for spec in inputs.values()}) != 1:
            raise ValueError(f"Real rows differ between generators: {key}")
    if len({spec["taxa"]["sha256"] for spec in inputs.values()}) != 1:
        raise ValueError("Taxon order differs between generators")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp)/"run"
        stage.mkdir()
        (stage/"logs").mkdir()
        frozen.write(stage/"protocol.json", dict(config=cfg, sources=sources, inputs=inputs,
                     seeds=[published.seed_for(next(iter(inputs.values()))["config"], i, "final")
                            for i in range(5)], torch_version=str(torch.__version__),
                     assessment="retrospective; checkpoint selection uses stopping Brier only"))
        for path in SOURCES + [config]:
            dest = stage/"source"/path.relative_to(REPO) if path in SOURCES else stage/"config.json"
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())
        stage.rename(out)


def context(out, index):
    if not 0 <= index < 5 * len(METHODS):
        raise ValueError("Index outside the 20 final pairs")
    protocol = frozen.read(out/"protocol.json")
    for path, digest in protocol["sources"].items():
        frozen.checked(REPO/path, digest)
    method, repeat = METHODS[index % len(METHODS)], index // len(METHODS)
    entry = protocol["inputs"][method]
    task = entry["tasks"][repeat]
    seed = published.seed_for(entry["config"], repeat, "final")
    if seed != protocol["seeds"][repeat]:
        raise ValueError("Final seeds differ from the frozen configuration")
    return protocol["config"], method, entry, task, seed


def raw_arrays(entry, roles):
    arrays = {}
    for role in roles:
        for side in ("p", "q"):
            key = f"{role}_{side}"
            spec = entry["arrays"][key]
            path = Path(entry["root"])/spec["path"]
            frozen.checked(path, spec["sha256"])
            x = np.load(path, allow_pickle=False)
            if list(x.shape) != spec["shape"] or str(x.dtype) != spec["dtype"] or x.shape[1] != 614:
                raise ValueError(f"Invalid archived array: {path}")
            arrays[key] = x
    return arrays


def closed(raw):
    return np.asarray(mask_metrics.close(raw), dtype=np.float32)


@torch.no_grad()
def predict_tensor(model, inputs):
    model.eval()
    return torch.cat([model(block) for block in inputs.split(1024)])


def predict(model, raw, device):
    return predict_tensor(model, torch.as_tensor(closed(raw), device=device)).cpu().numpy()


def endpoints(r, bounds):
    # Compare in the prediction dtype (the network and its clamp bounds are float32).
    return {name: float(np.mean(r == value)) for name, value in zip(("lower", "upper"), bounds)}


def train_threshold(raw, cfg, seed, device):
    """One full empirical JS gradient and one AdamW update per epoch.

    Source-specific row-count weights keep this identical in objective to a
    single full batch when the two sources and final chunks differ in size.
    """
    started = time.monotonic()
    x = {key: closed(value) for key, value in raw.items()}
    torch.manual_seed(seed)
    model = ThresholdRDR.from_training(x["train_p"], x["train_q"], width=cfg["width"],
            max_knots=cfg["max_knots"], a=cfg["a"], b=cfg["b"]).to(device)
    x = {key: torch.as_tensor(value, device=device) for key, value in x.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5,
                            patience=cfg["plateau_patience"], cooldown=cfg["plateau_cooldown"])
    history, best, best_state, stale = [], math.inf, None, 0
    bounds = model.bounds.detach().cpu().numpy()
    with torch.no_grad():
        initial = predict_tensor(model, x["train_p"]).cpu().numpy()
        initial_q = predict_tensor(model, x["train_q"]).cpu().numpy()
    startup = {s: endpoints(r, bounds) for s, r in (("p",initial), ("q",initial_q))}
    if any(sum(v.values()) == 1 for v in startup.values()):
        raise RuntimeError("All initial predictions clipped for a training source")
    n_p, n_q = len(x["train_p"]), len(x["train_q"])
    chunks = math.ceil(max(n_p, n_q)/cfg["gradient_chunk_rows"])
    p_parts, q_parts = x["train_p"].tensor_split(chunks), x["train_q"].tensor_split(chunks)
    if chunks > min(n_p, n_q):
        raise ValueError("Gradient chunk size too small for these unequal source counts")
    for epoch in range(1, cfg["max_epochs"]+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk = torch.zeros((), dtype=torch.float64, device=device)
        for p, q in zip(p_parts, q_parts):
            r_p, r_q = model(torch.cat((p, q))).double().split([len(p), len(q)])
            contribution = -.5 * (torch.log(r_p / 2).sum() / n_p
                                   + torch.log1p(-r_q / 2).sum() / n_q)
            contribution.backward()
            risk += contribution.detach()
        if not torch.isfinite(risk):
            raise FloatingPointError("Nonfinite equal-mixture JS objective")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_max_norm"], error_if_nonfinite=True)
        optimizer.step()
        sp, sq = (predict_tensor(model, x["stopping_"+s]) for s in ("p","q"))
        stopping = balanced_brier_rdr(sp.double(), sq.double()).item()
        if not math.isfinite(stopping):
            raise FloatingPointError("Nonfinite stopping Brier")
        if stopping < best:
            best, best_epoch, stale = stopping, epoch, 0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
        log = dict(epoch=epoch, training_objective=risk.item(), stopping_brier=stopping,
                   learning_rate=optimizer.param_groups[0]["lr"])
        for side, scores in (("p", sp), ("q", sq)):
            for endpoint, fraction in endpoints(scores.cpu().numpy(), bounds).items():
                log[f"stopping_{endpoint}_{side}"] = fraction
        history.append(log)
        scheduler.step(stopping)
        if epoch == 1 or epoch % 100 == 0:
            print(f"threshold seed={seed} epoch={epoch} stopping Brier={stopping:.8f}", flush=True)
        if epoch >= cfg["min_epochs"] and stale >= cfg["patience"]:
            break
    model.load_state_dict(best_state)
    return model, history, dict(best_epoch=best_epoch, epochs_run=epoch, stopping_brier=best,
               fit_seconds=time.monotonic()-started, seed_initialization=seed, startup_endpoints=startup,
               hinges=model.thresholds.numel(), parameter_count=sum(v.numel() for v in model.parameters()))


def load_model(path, arm, entry, task, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if arm == "threshold":
        model = ThresholdRDR.from_state_dict(state)
    else:
        cfg = entry["config"]
        model = control.make_control(614, cfg["hidden_dimension"], cfg["blocks"], task["output_alpha"])
        model.load_state_dict(state)
    return model.eval().to(device)


def scores(model, raw, role, arm, device):
    predictor = predict if arm == "threshold" else control.predict_control
    return tuple(predictor(model, raw[role+"_"+s], device) for s in ("p","q"))


def verify_directory(path, protocol_hash):
    marker = frozen.read(path/"COMPLETE.json")
    if marker["protocol_sha256"] != protocol_hash:
        raise ValueError(f"Run protocol changed: {path}")
    for name, digest in marker["files"].items():
        frozen.checked(path/name, digest)


def save_directory(stage, target, protocol_hash):
    files = {p.name:frozen.sha(p) for p in stage.iterdir() if p.is_file()}
    frozen.write(stage/"COMPLETE.json", dict(protocol_sha256=protocol_hash, files=files))
    try:
        stage.rename(target)
    except OSError:
        if not target.is_dir():
            raise
        verify_directory(target, protocol_hash)


def fit(out, index, device, arms):
    cfg, method, entry, task, seed = context(out, index)
    protocol_hash = frozen.sha(out/"protocol.json")
    raw = raw_arrays(entry, ("train","stopping"))
    parent = out/"fits"/method/f'{task["repeat"]:02d}'
    parent.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        target = parent/arm
        if target.exists():
            verify_directory(target, protocol_hash)
            continue
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            stage = Path(temp)/"fit"
            stage.mkdir()
            if arm == "threshold":
                model, history, training = train_threshold(raw, cfg, seed, device)
            else:
                model, history, training = control.fit_control(raw, entry["config"], task, device)
            if training["seed_initialization"] != seed:
                raise ValueError("Unexpected initialization seed")
            torch.save(model.state_dict(), stage/"model.pt")
            with (stage/"history.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            before = scores(model, raw, "stopping", arm, device)
            reloaded = load_model(stage/"model.pt", arm, entry, task, device)
            after = scores(reloaded, raw, "stopping", arm, device)
            if not all(np.array_equal(a,b) for a,b in zip(before, after)):
                raise ValueError("Checkpoint reload changed stopping predictions")
            observed = balanced_brier(*(arr.astype(np.float64) for arr in after))
            if abs(observed-training["stopping_brier"]) > 1e-7:
                raise ValueError("Saved checkpoint differs from minimum stopping Brier")
            frozen.write(stage/"training.json", dict(**training, method=method, repeat=task["repeat"],
                         device=str(device), reload_brier=observed, reload_predictions_identical=True))
            save_directory(stage, target, protocol_hash)
            print(f"saved {arm} {method} seed={seed} epoch={training['best_epoch']}", flush=True)


def assess(out, index, device):
    cfg, method, entry, task, seed = context(out, index)
    protocol_hash = frozen.sha(out/"protocol.json")
    # All methods and seeds must have frozen checkpoints before unblinding.
    for name in METHODS:
        for rep in range(5):
            for arm in ARMS:
                marker = out/"fits"/name/f"{rep:02d}"/arm/"COMPLETE.json"
                if frozen.read(marker)["protocol_sha256"] != protocol_hash:
                    raise ValueError(f"Unfrozen fit: {marker}")
    pair = Path(method)/f'{task["repeat"]:02d}'
    target = out/"assessment"/pair
    if target.exists():
        return verify_directory(target, protocol_hash)
    raw = raw_arrays(entry, ("test",))
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as temp:
        stage = Path(temp)/"assessment"
        stage.mkdir()
        record = dict(method=method, repeat=task["repeat"], seed=seed,
                      test_hashes={k:entry["arrays"][k]["sha256"] for k in raw})
        predictions = {}
        for arm in ARMS:
            folder = out/"fits"/pair/arm
            verify_directory(folder, protocol_hash)
            model = load_model(folder/"model.pt", arm, entry, task, device)
            p, q = scores(model, raw, "test", arm, device)
            record[arm+"_brier"] = balanced_brier(p.astype(np.float64), q.astype(np.float64))
            for side, r in (("p",p), ("q",q)):
                if not np.isfinite(r).all() or r.min()<0 or r.max()>2:
                    raise ValueError("Invalid test RDR")
                if arm == "threshold":
                    bounds = model.bounds.cpu().numpy()
                    if r.min()<bounds[0] or r.max()>bounds[1]:
                        raise ValueError("Test score exceeds configured clip interval")
                    for endpoint, fraction in endpoints(r,bounds).items():
                        record[f"threshold_{endpoint}_{side}"] = fraction
                predictions[f"{arm}_{side}"] = r
        np.savez_compressed(stage/"predictions.npz", **predictions)
        frozen.write(stage/"scores.json", record)
        save_directory(stage, target, protocol_hash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","fit","assess"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", choices=("cpu","cuda"), default="cpu")
    parser.add_argument("--arm", choices=(*ARMS,"both"), default="both")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.action == "prepare":
        return prepare(args.output, args.config)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.action == "fit":
        fit(args.output, args.index, args.device, ARMS if args.arm == "both" else (args.arm,))
    else:
        assess(args.output, args.index, args.device)


if __name__ == "__main__":
    main()
