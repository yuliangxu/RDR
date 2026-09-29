#!/usr/bin/env python3
"""Paired DCGAN loss/sigmoid selection with locked sample roles and inputs.

The CNN emits a raw logit z; its RDR is 2 sigmoid(alpha*z). All four risks
target P / ((P+Q)/2). Brier chooses checkpoints. Selection uses Brier and
cell-average calibration discrepancy only; official-test images are accessed
by ``evaluate`` only after a complete selection has been frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mnist-selection-matplotlib")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from experiments.MNIST.generators import build_dcgan28
from experiments.MNIST.selection_metrics import (
    balanced_brier, calibration_diagnostics, choose_configuration,
)
from utils.networks import DREConvNet_DCGAN_MNIST, dcgan_init

DEFAULT_CONFIG = Path(__file__).with_name("selection_config.json")
ROLE_IDS = {"initialization": 1, "training_q": 2, "training_order": 3,
            "earlystop": 4, "selection_calibration": 5,
            "selection_evaluation": 6, "final_calibration": 7,
            "final_evaluation": 8}
FIT_FILES = ("model.pt", "predictions.npz", "metrics.json", "cells.json", "history.json")
FINAL_FILES = ("metrics.json", "cells.json", "predictions.npz")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seed_for(cfg, repeat, role):
    return int(np.random.SeedSequence([cfg["seed"], repeat, ROLE_IDS[role]])
               .generate_state(1)[0])


def candidates(cfg):
    return [{"candidate": f"{loss}_a{alpha:g}".replace(".", "p"),
             "loss": loss, "output_alpha": alpha}
            for loss in cfg["losses"] for alpha in cfg["output_alphas"]]


def validate_config(cfg):
    if not cfg["losses"] or len(set(cfg["losses"])) != len(cfg["losses"]) or not set(cfg["losses"]) <= {"hellinger", "kl", "chisq", "js"}:
        raise ValueError("Use distinct supported midpoint losses")
    if not cfg["output_alphas"] or len(set(cfg["output_alphas"])) != len(cfg["output_alphas"]) or any(a not in (.5, 1, 2, 4) for a in cfg["output_alphas"]):
        raise ValueError("The approved sigmoid slopes are 0.5, 1, 2, 4")
    for key in ("repeats", "train_n", "earlystop_n", "selection_calibration_n",
                "selection_evaluation_n", "final_calibration_n", "final_evaluation_n",
                "batch_size", "max_epochs", "min_epochs", "patience", "base_channels", "bins"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg["repeats"] < 2 or cfg["min_epochs"] > cfg["max_epochs"] or cfg["batch_size"] < 2:
        raise ValueError("Need >=2 repetitions and valid epoch/batch limits")
    if sum(cfg[f"{r}_n"] for r in ("train", "earlystop", "selection_calibration", "selection_evaluation")) > 60000:
        raise ValueError("Training-source role counts exceed 60,000")
    if cfg["final_calibration_n"] + cfg["final_evaluation_n"] > 10000:
        raise ValueError("Final role counts exceed 10,000")
    if cfg["bins"] != 20 or not 0 < cfg["ci_alpha"] < 1 or not 0 < cfg["max_lr"] < 1:
        raise ValueError("Invalid fixed partition or numerical settings")


def make_splits(cfg):
    result = {}
    for source, total, roles, tag in (
        ("official_train", 60000, ("train", "earlystop", "selection_calibration", "selection_evaluation"), 101),
        ("official_test", 10000, ("final_calibration", "final_evaluation"), 102),
    ):
        order = np.random.default_rng(np.random.SeedSequence([cfg["seed"], tag])).permutation(total)
        offset = 0
        for role in roles:
            n = cfg[f"{role}_n"]
            result[role] = {"source": source, "indices": order[offset:offset+n].tolist()}
            offset += n
    return result


def prepare(output, config, data_root, smoke=False):
    cfg = read(config)
    if smoke:
        cfg.update(losses=["hellinger", "kl", "chisq", "js"], output_alphas=[.5, 4],
                   repeats=2, train_n=32, earlystop_n=20, selection_calibration_n=40,
                   selection_evaluation_n=40, final_calibration_n=40, final_evaluation_n=40,
                   batch_size=16, max_epochs=2, min_epochs=1, patience=1, base_channels=4)
    validate_config(cfg)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    # Validate every acquisition input before creating the run directory.
    inputs = {"assets/netG_epoch_99.pth": data_root / "mnist_dcgan/netG_epoch_99.pth"}
    for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        inputs[f"assets/{name}"] = data_root / "MNIST/raw" / name
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    files = {}
    sources = list((ROOT / "utils").glob("*.py")) + [
        ROOT / "experiments/__init__.py", Path(__file__), DEFAULT_CONFIG,
        Path(__file__).with_name("selection_metrics.py"),
        Path(__file__).with_name("selection_report.py"),
        Path(__file__).with_name("selection.slurm"),
        Path(__file__).with_name("generators.py"), Path(__file__).with_name("__init__.py"),
    ]
    for source in sources:
        inputs["source/" + str(source.relative_to(ROOT))] = source
    for relative, source in inputs.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[relative] = {"sha256": sha(target), "bytes": target.stat().st_size,
                           "original": str(source.resolve())}
    write(output / "splits.json", make_splits(cfg))
    files["splits.json"] = {"sha256": sha(output / "splits.json")}
    grid = candidates(cfg)
    tasks = [{"task_index": i, "repeat": repeat, **candidate}
             for i, (repeat, candidate) in enumerate((r, c) for r in range(cfg["repeats"]) for c in grid)]
    protocol = {"schema": 1, "study": "dcgan_midpoint_loss_sigmoid_selection", "smoke": smoke,
                "config": cfg, "candidates": grid, "tasks": tasks, "files": files,
                "target": "r=2p/(p+q), midpoint denominator M=(P+Q)/2",
                "input_scale": "P and DCGAN Q are both [-1,1] for every role",
                "checkpoint_rule": "absolute minimum earlystop balanced Brier; patience uses Brier only",
                "selection_rule": "paired one-SE Brier shortlist, then minimum local absolute gap; require >=0.99 supported mass",
                "sampling": "fixed disjoint real roles across repetitions; independent Q/initialization per repetition; candidates paired",
                "ci_scope": "nominal 95% C.2 across20 cells per frozen model, not across selected models; neural means have sampling uncertainty",
                "limitations": "generator pretraining membership unaudited; official test previously inspected in historical studies",
                "python": sys.version, "packages": {p: importlib.metadata.version(p) for p in ("torch", "torchvision", "numpy", "scipy", "matplotlib")}}
    write(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(sha(output / "protocol.json") + "\n")
    print(json.dumps({"prepared": str(output), "fits": len(tasks), "smoke": smoke}))


def load_protocol(output):
    if sha(output / "protocol.json") != (output / "protocol.sha256").read_text().strip():
        raise ValueError("Protocol changed")
    protocol = read(output / "protocol.json")
    for name, metadata in protocol["files"].items():
        if sha(output / name) != metadata["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
        if name.startswith("source/") and name.endswith(".py"):
            executing_source = ROOT / Path(name).relative_to("source")
            if sha(executing_source) != metadata["sha256"]:
                raise ValueError(f"Executing source differs from frozen input: {name}; use the saved source entrypoint")
    return protocol


def midpoint_loss_from_logits(p_logits, q_logits, loss):
    """Expanded midpoint objectives, evaluated stably without ratio clipping.

    log r = log2 + logsigmoid(t); log(2-r) = log2 + logsigmoid(-t).
    Additive constants match utils.losses.midpoint_rdr_loss. The separate
    means retain the correct P/Q weights for unequal minibatch sizes.
    """
    lp, lq = math.log(2) + F.logsigmoid(p_logits), math.log(2) + F.logsigmoid(q_logits)
    if loss == "hellinger":
        return .5 * (-.5*lp).exp().mean() + .25 * (.5*lp).exp().mean() + .25 * (.5*lq).exp().mean() - 1
    if loss == "kl":
        return .5*lp.exp().mean() + .5*lq.exp().mean() - lp.mean() - 1
    if loss == "chisq":
        return .5*(2*lp).exp().mean() + .5*(2*lq).exp().mean() - 2*lp.exp().mean() - 1
    if loss == "js":
        return -.5*lp.mean() - .5*(math.log(2) + F.logsigmoid(-q_logits)).mean()
    raise ValueError(loss)


def make_model(cfg, repeat, alpha, device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_for(cfg, repeat, "initialization"))
        # The shared class's log_scale=True interface supplies raw z here.
        model = DREConvNet_DCGAN_MNIST(base=cfg["base_channels"], log_scale=True,
                                     output_alpha=alpha)
        model.apply(dcgan_init)
    return model.to(device)


def real_images(output, role):
    split = read(output / "splits.json")[role]
    name = "train-images-idx3-ubyte" if split["source"] == "official_train" else "t10k-images-idx3-ubyte"
    data = np.frombuffer((output / "assets" / name).read_bytes(), dtype=np.uint8, offset=16).reshape(-1, 1, 28, 28)
    return torch.from_numpy(data[split["indices"]].copy())


def to_pixels(x, device):
    x = x.to(device)
    return x.float().div(127.5).sub(1) if x.dtype == torch.uint8 else x


@torch.no_grad()
def generated(generator, n, rng, batch_size, device):
    batches = []
    for start in range(0, n, batch_size):
        x = generator(torch.randn(min(batch_size, n-start), 100, 1, 1, generator=rng, device=device))
        if not torch.isfinite(x).all() or x.min() < -1 or x.max() > 1:
            raise ValueError("DCGAN output is not finite model-space data")
        batches.append(x.cpu())
    return torch.cat(batches)


@torch.no_grad()
def predict(model, x, alpha, batch_size, device):
    model.eval()
    outputs = [2*torch.sigmoid(alpha*model(to_pixels(chunk, device))).flatten().cpu()
               for chunk in x.split(batch_size)]
    result = torch.cat(outputs).double().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite predictions")
    return result


def role_samples(output, cfg, repeat, roles, generator, device):
    samples = {}
    for role in roles:
        samples[role + "_p"] = real_images(output, role)
        rng = torch.Generator(device=device).manual_seed(seed_for(cfg, repeat, role))
        samples[role + "_q"] = generated(generator, cfg[role + "_n"], rng, cfg["batch_size"], device)
    return samples


def run_path(output, task):
    return output / "runs" / task["candidate"] / f"repeat_{task['repeat']:02d}"


def complete(folder, output, filenames):
    write(folder / "complete.json", {"protocol_sha256": sha(output / "protocol.json"),
          "files": {name: sha(folder / name) for name in filenames}})


def verify_result(folder, output):
    result = read(folder / "complete.json")
    if result["protocol_sha256"] != sha(output / "protocol.json"):
        raise ValueError("Result belongs to a different protocol")
    required = FINAL_FILES if folder.parent.name == "final" else FIT_FILES
    if set(result["files"]) != set(required):
        raise ValueError(f"Incomplete result manifest: {folder}")
    for name, digest in result["files"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Result changed: {folder/name}")
    return read(folder / "metrics.json")


def fit(output, task_index, device):
    protocol = load_protocol(output)
    if not 0 <= task_index < len(protocol["tasks"]):
        raise ValueError("Task index outside protocol")
    cfg, task = protocol["config"], protocol["tasks"][task_index]
    folder = run_path(output, task)
    if (folder / "complete.json").exists():
        verify_result(folder, output)
        print(f"Already complete: {folder}")
        return
    folder.mkdir(parents=True, exist_ok=True)
    repeat, alpha = task["repeat"], task["output_alpha"]
    generator, _ = build_dcgan28(output / "assets/netG_epoch_99.pth", device=device)
    generator.requires_grad_(False)
    model = make_model(cfg, repeat, alpha, device)
    train = real_images(output, "train")
    samples = role_samples(output, cfg, repeat, ("earlystop",), generator, device)
    batch_size = cfg["batch_size"]
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["max_lr"],
        total_steps=cfg["max_epochs"]*math.ceil(len(train)/batch_size), pct_start=.1,
        anneal_strategy="cos", div_factor=10, final_div_factor=1000)
    latent_rng = torch.Generator(device=device).manual_seed(seed_for(cfg, repeat, "training_q"))
    order_rng = torch.Generator().manual_seed(seed_for(cfg, repeat, "training_order"))
    best, best_state, best_epoch, stale, history = float("inf"), None, None, 0, []
    start_time = time.monotonic()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        if epoch >= cfg["bn_freeze_epoch"]:
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        order = torch.randperm(len(train), generator=order_rng)
        total_risk = 0.
        for indices in order.split(batch_size):
            p = to_pixels(train[indices], device)
            with torch.no_grad():
                q = generator(torch.randn(len(p), 100, 1, 1, generator=latent_rng, device=device))
            logits = alpha * model(torch.cat((p, q)))
            risk = midpoint_loss_from_logits(logits[:len(p)], logits[len(p):], task["loss"])
            if not torch.isfinite(risk):
                raise FloatingPointError(f"Nonfinite training risk at epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            risk.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total_risk += float(risk.detach()) * len(p)
        ep, eq = [predict(model, samples[f"earlystop_{s}"], alpha, batch_size, device) for s in ("p", "q")]
        brier = balanced_brier(ep, eq)
        if brier < best:
            best, best_epoch, stale = brier, epoch + 1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch+1, "training_objective": total_risk/len(train),
                        "earlystop_brier": brier, "lr": optimizer.param_groups[0]["lr"]})
        print(f"{task['candidate']} repeat={repeat} epoch={epoch+1} Brier={brier:.7f}", flush=True)
        if epoch+1 >= cfg["min_epochs"] and stale >= cfg["patience"]:
            break
    fit_seconds = time.monotonic() - start_time
    model.load_state_dict(best_state)
    restored = balanced_brier(*[predict(model, samples[f"earlystop_{s}"], alpha, batch_size, device) for s in ("p", "q")])
    if abs(restored-best) > 1e-10:
        raise AssertionError("Best-Brier checkpoint was not restored")
    samples = role_samples(output, cfg, repeat, ("selection_calibration", "selection_evaluation"), generator, device)
    predictions = {name: predict(model, value, alpha, batch_size, device) for name, value in samples.items()}
    summary, cells = calibration_diagnostics(*[predictions[f"{role}_{side}"] for role in ("selection_calibration", "selection_evaluation") for side in ("p", "q")], alpha=cfg["ci_alpha"], bins=cfg["bins"])
    metrics = {**task, **summary, "brier": balanced_brier(predictions["selection_evaluation_p"], predictions["selection_evaluation_q"]),
               "best_epoch": best_epoch, "epochs": len(history), "earlystop_brier": best,
               "fit_seconds": fit_seconds, "device": str(device),
               "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
               "train_p_per_epoch": len(train), "train_q_per_epoch": len(train)}
    torch.save({"model": best_state, "task": task, "protocol_sha256": sha(output / "protocol.json"), "best_epoch": best_epoch}, folder / "model.pt")
    np.savez_compressed(folder / "predictions.npz", **predictions)
    write(folder / "metrics.json", metrics)
    write(folder / "cells.json", cells)
    write(folder / "history.json", history)
    complete(folder, output, FIT_FILES)


def all_rows(output, protocol):
    rows = []
    for task in protocol["tasks"]:
        folder = run_path(output, task)
        row = verify_result(folder, output)
        if any(row[key] != task[key] for key in task):
            raise ValueError(f"Incorrect task identity in {folder}")
        rows.append(row)
    return rows


def freeze(output):
    protocol = load_protocol(output)
    rows = all_rows(output, protocol)
    selection = choose_configuration(rows)
    selection.update(protocol_sha256=sha(output / "protocol.json"),
                     checkpoints={str(task["repeat"]): sha(run_path(output, task) / "model.pt")
                       for task in protocol["tasks"] if task["candidate"] == selection["chosen"]["candidate"]})
    path = output / "selection.json"
    if path.exists():
        if read(path) != selection or sha(path) != (output / "selection.sha256").read_text().strip():
            raise ValueError("Existing frozen selection differs")
    else:
        write(path, selection)
        (output / "selection.sha256").write_text(sha(path) + "\n")
    print(json.dumps(selection["chosen"]))


def load_selection(output, protocol):
    path = output / "selection.json"
    if sha(path) != (output / "selection.sha256").read_text().strip():
        raise ValueError("Frozen selection changed")
    selection = read(path)
    if selection["protocol_sha256"] != sha(output / "protocol.json"):
        raise ValueError("Selection protocol mismatch")
    expected = choose_configuration(all_rows(output, protocol))
    for key in expected:
        if expected[key] != selection[key]:
            raise ValueError("Selection does not follow the registered rule")
    return selection


def evaluate(output, repeat, device):
    protocol = load_protocol(output)
    selection = load_selection(output, protocol)
    cfg, chosen = protocol["config"], selection["chosen"]
    if not 0 <= repeat < cfg["repeats"]:
        raise ValueError("Repeat outside protocol")
    folder = output / "final" / f"repeat_{repeat:02d}"
    task = next(t for t in protocol["tasks"] if t["repeat"] == repeat and t["candidate"] == chosen["candidate"])
    if (folder / "complete.json").exists():
        existing = verify_result(folder, output)
        if any(existing[key] != task[key] for key in task) or existing["selection_sha256"] != sha(output / "selection.json"):
            raise ValueError("Final result belongs to a different selected model")
        return
    checkpoint = run_path(output, task) / "model.pt"
    if sha(checkpoint) != selection["checkpoints"][str(repeat)]:
        raise ValueError("Selected checkpoint changed")
    model = make_model(cfg, repeat, chosen["output_alpha"], device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"])
    generator, _ = build_dcgan28(output / "assets/netG_epoch_99.pth", device=device)
    samples = role_samples(output, cfg, repeat, ("final_calibration", "final_evaluation"), generator, device)
    predictions = {name: predict(model, value, chosen["output_alpha"], cfg["batch_size"], device) for name, value in samples.items()}
    summary, cells = calibration_diagnostics(*[predictions[f"{role}_{side}"] for role in ("final_calibration", "final_evaluation") for side in ("p", "q")], alpha=cfg["ci_alpha"], bins=cfg["bins"])
    metrics = {**task, **summary, "brier": balanced_brier(predictions["final_evaluation_p"], predictions["final_evaluation_q"]),
               "selection_sha256": sha(output / "selection.json"), "role": "retrospective final assessment after selection lock"}
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / "metrics.json", metrics)
    write(folder / "cells.json", cells)
    np.savez_compressed(folder / "predictions.npz", **predictions)
    complete(folder, output, FINAL_FILES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "fit", "freeze", "evaluate", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if args.command == "prepare":
        prepare(output, args.config, args.data_root, args.smoke)
    elif args.command == "fit":
        fit(output, args.task_index, device)
    elif args.command == "freeze":
        freeze(output)
    elif args.command == "evaluate":
        evaluate(output, args.repeat, device)
    else:
        from experiments.MNIST.selection_report import report
        protocol = load_protocol(output)
        all_rows(output, protocol)
        load_selection(output, protocol)
        report(output)


if __name__ == "__main__":
    main()
