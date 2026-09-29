#!/usr/bin/env python3
"""Reproducible selected-model MNIST digit perturbation and null validation.

Historical primary rows are preserved. Fresh calibration/evaluation resamples
give conditional finite-pool diagnostics, not population confidence intervals.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mnist-controls-matplotlib")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.MNIST import model_selection as ms
from experiments.MNIST.selection_metrics import balanced_brier, calibration_diagnostics

NAMES = ("perturbation", "null")
FILES = ("model.pt", "scores.pt", "diagnostics.pt", "metrics.json", "history.json",
         "cells.json", "digits.json")
RAW_NAMES = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
Q_PROBS = torch.tensor([0., 0., .025, .025, .1, .1, .1, .1, .275, .275], dtype=torch.float64)
CI_SCOPE = ("Conditional finite-pool Monte Carlo intervals after freezing the fitted network: "
            "the reference images are fixed; only independent resampling draws are random. "
            "Repeated images are not additional independent source images. These intervals "
            "exclude reference-image sampling uncertainty and are not population or pointwise guarantees.")


def sample_indices(labels, source_indices, probs, n, seed):
    """Original two-stage label sampler, with its own RNG and saved indices."""
    labels = torch.as_tensor(labels, dtype=torch.long)
    source = torch.as_tensor(source_indices, dtype=torch.long)
    probs = torch.as_tensor(probs, dtype=torch.float64)
    if not len(source) or n <= 0 or probs.shape != (10,) or not torch.isfinite(probs).all():
        raise ValueError("A nonempty source and ten finite probabilities are required")
    if (probs < 0).any() or not torch.isclose(probs.sum(), torch.tensor(1., dtype=probs.dtype)):
        raise ValueError("Label probabilities must be nonnegative and sum to one")
    rng = torch.Generator().manual_seed(seed)
    draws = torch.multinomial(probs, n, replacement=True, generator=rng)
    result = torch.empty(n, dtype=torch.long)
    source_labels = labels[source]
    for digit, prob in enumerate(probs):
        choices = source[source_labels == digit]
        if prob > 0 and not len(choices):
            raise ValueError(f"No source images for positive-probability digit {digit}")
        positions = (draws == digit).nonzero(as_tuple=True)[0]
        if len(positions):
            result[positions] = choices[torch.randint(len(choices), (len(positions),), generator=rng)]
    return result.tolist()


def uniform_indices(source, n, seed):
    source = torch.as_tensor(source, dtype=torch.long)
    return source[torch.randint(len(source), (n,), generator=torch.Generator().manual_seed(seed))].tolist()


def make_splits(train_labels, test_labels, smoke=False):
    """Recreate the exact retained seed-123 historical primary role rows."""
    if len(train_labels) != 60000 or len(test_labels) != 10000:
        raise ValueError("Expected official MNIST 60,000/10,000 labels")
    train_order = torch.randperm(60000, generator=torch.Generator().manual_seed(123)).tolist()
    test_order = torch.randperm(10000, generator=torch.Generator().manual_seed(133)).tolist()
    train_p = list(range(60000))
    val_p, test_p = test_order[:5000], test_order[5000:]
    if smoke:
        train_p, val_p, test_p = train_p[:160], val_p[:80], test_p[:80]
    perturbation = {
        "train": {"source": "official_train", "p": train_p,
                  "q": sample_indices(train_labels, train_p, Q_PROBS, len(train_p), 143)},
        "validation": {"source": "official_test", "p": val_p,
                       "q": sample_indices(test_labels, val_p, Q_PROBS, len(val_p), 153)},
        "test": {"source": "official_test", "p": test_p,
                 "q": sample_indices(test_labels, test_p, Q_PROBS, len(test_p), 163)},
        "reference": {"source": "official_test", "indices": test_p,
                      "description": "held-out perturbation source pool; P uniform; Q label-weighted"},
    }
    null = {
        "train": {"source": "official_train", "p": train_order[:30000], "q": train_order[30000:]},
        "validation": {"source": "official_test", "p": test_order[:5000], "q": test_order[5000:]},
    }
    if smoke:
        for role in ("train", "validation"):
            count = 96 if role == "train" else 80
            null[role]["p"], null[role]["q"] = null[role]["p"][:count], null[role]["q"][:count]
    null["reference"] = {"source": "official_test",
                         "indices": sorted(null["validation"]["p"] + null["validation"]["q"]),
                         "description": "pooled test-derived validation images; P=Q uniform"}
    return {"perturbation": perturbation, "null": null}


def raw_directory(data_root):
    for candidate in (data_root / "MNIST/raw", data_root / "raw", data_root):
        if all((candidate / name).is_file() for name in RAW_NAMES):
            return candidate
    raise FileNotFoundError(f"Four uncompressed MNIST IDX files not found below {data_root}")


def read_labels(path):
    with path.open("rb") as stream:
        magic, n = struct.unpack(">II", stream.read(8))
    if magic != 2049 or path.stat().st_size != n + 8:
        raise ValueError(f"Invalid MNIST label file {path}")
    return torch.from_numpy(np.fromfile(path, dtype=np.uint8, offset=8).astype(np.int64))


def read_images(output, source, indices):
    prefix = {"official_train": "train", "official_test": "t10k"}[source]
    path = output / "assets" / f"{prefix}-images-idx3-ubyte"
    with path.open("rb") as stream:
        magic, n, rows, cols = struct.unpack(">IIII", stream.read(16))
    if magic != 2051 or (rows, cols) != (28, 28) or path.stat().st_size != 16+n*784:
        raise ValueError(f"Invalid MNIST image file {path}")
    # Only the requested role's rows are materialized; final rows are requested
    # by train_branch only after saving and restoring its selected checkpoint.
    mapped = np.memmap(path, dtype=np.uint8, mode="r", offset=16, shape=(n, 1, 28, 28))
    return torch.from_numpy(np.array(mapped[np.asarray(indices, dtype=np.int64)], copy=True))


def prepare(output, data_root, evidence_root=None, smoke=False):
    from experiments.MNIST.selection_evidence import load_selected
    output, data_root = Path(output), Path(data_root)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    evidence = load_selected(evidence_root)
    selection, cfg = evidence["selection"], dict(evidence["config"])
    chosen = selection["chosen"]
    if (chosen["loss"], chosen["output_alpha"]) != ("js", 2.):
        raise ValueError("These registered controls require the selected JS/slope-2 configuration")
    cfg.update(seed=2026092901, diagnostic_n=5000)
    if smoke:
        cfg.update(batch_size=16, max_epochs=2, min_epochs=1, patience=1, base_channels=4, diagnostic_n=80)
    raw = raw_directory(data_root)
    splits = make_splits(read_labels(raw / RAW_NAMES[1]), read_labels(raw / RAW_NAMES[3]), smoke)
    inputs = {f"assets/{name}": raw / name for name in RAW_NAMES}
    inputs.update({f"assets/evidence/{relative}": Path(path) for relative, path in evidence["files"].items()})
    sources = list((ROOT / "utils").glob("*.py"))
    sources += [ROOT / "experiments/__init__.py", ROOT / "experiments/MNIST/__init__.py"]
    sources += [ROOT / "experiments/MNIST" / name for name in (
        "generators.py", "model_selection.py", "selection_metrics.py", "selection_report.py",
        "selection_evidence.py", "controls.py", "controls_plot.py", "controls.slurm")]
    inputs.update({"source/" + str(path.relative_to(ROOT)): path for path in sources})
    for source in inputs.values():
        if not source.is_file():
            raise FileNotFoundError(source)
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    files = {}
    for relative, source in inputs.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[relative] = {"sha256": ms.sha(target), "original": str(source.resolve())}
    ms.write(output / "splits.json", splits)
    files["splits.json"] = {"sha256": ms.sha(output / "splits.json")}
    ms.write(output / "selection.json", selection)
    files["selection.json"] = {"sha256": ms.sha(output / "selection.json")}
    protocol = {
        "study": "selected_mnist_perturbation_and_null", "smoke": smoke,
        "chosen": chosen, "config": cfg, "files": files,
        "selection_sha256": ms.sha(output / "selection.json"),
        "target": "r=2p/(p+q); network inputs [-1,1]",
        "primary_splits": "Historical seed 123 train halves and seed 133 test halves; Q resampling seeds 143/153/163",
        "checkpoint_rule": "minimum validation balanced Brier; checkpoint saved/restored before final or fresh diagnostic scoring",
        "sampling": "fixed historical primary Q rows; all training rows each epoch; independent shuffles for P and Q",
        "diagnostic_scope": CI_SCOPE,
        "null_role": "validation diagnostic, not an independent final-test experiment",
        "reference_roles": "perturbation held-out 5,000; null pooled 10,000 validation; smoke pools smaller",
        "limitations": "post-selection transfer without further search; official test previously inspected; historical early stopping used Hellinger",
        "python": sys.version, "packages": {p: importlib.metadata.version(p)
            for p in ("torch", "torchvision", "numpy", "scipy", "matplotlib")},
    }
    ms.write(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(ms.sha(output / "protocol.json") + "\n")
    print(f"Prepared {output}; chosen={chosen}; smoke={smoke}", flush=True)


def load_protocol(output):
    protocol = ms.load_protocol(output)
    if ms.sha(output / "selection.json") != protocol["selection_sha256"]:
        raise ValueError("Selection lock mismatch")
    if ms.read(output / "selection.json")["chosen"] != protocol["chosen"]:
        raise ValueError("Refit configuration differs from the selection lock")
    return protocol


def stream_seed(cfg, name, role):
    branches = {"perturbation": 301, "null": 302}
    roles = {"training_p": 1, "training_q": 2, "calibration_p": 3,
             "calibration_q": 4, "evaluation_p": 5, "evaluation_q": 6}
    return int(np.random.SeedSequence([cfg["seed"], branches[name], roles[role]]).generate_state(1)[0])


def diagnostic_indices(cfg, name, labels, reference):
    result = {}
    for role in ("calibration", "evaluation"):
        result[role] = {}
        for side in ("p", "q"):
            seed = stream_seed(cfg, name, f"{role}_{side}")
            if name == "perturbation" and side == "q":
                draws = sample_indices(labels, reference, Q_PROBS, cfg["diagnostic_n"], seed)
            else:
                draws = uniform_indices(reference, cfg["diagnostic_n"], seed)
            result[role][side] = draws
    return result


def score_record(scores, labels, indices, source):
    indices = torch.tensor(indices, dtype=torch.long)
    return {"scores": torch.as_tensor(scores, dtype=torch.float64),
            "labels": labels[indices].clone(), "original_indices": indices,
            "source": source}


def digit_rows(scores, name, reference_labels):
    frequencies = torch.bincount(reference_labels, minlength=10).double().numpy() / len(reference_labels)
    nominal = np.full(10, .1)
    q_probs = Q_PROBS.numpy() if name == "perturbation" else frequencies
    rows = []
    for role, record in scores.items():
        for side in ("p", "q"):
            values, labels = record[side]["scores"].numpy(), record[side]["labels"].numpy()
            for digit in range(10):
                selected = values[labels == digit]
                empirical = 2*frequencies[digit]/(frequencies[digit]+q_probs[digit]) if frequencies[digit]+q_probs[digit] else None
                rows.append({"role": role, "side": side, "digit": digit, "n": len(selected),
                    "mean_rdr": float(selected.mean()) if len(selected) else None,
                    "median_rdr": float(np.median(selected)) if len(selected) else None,
                    "nominal_reference": float(2*nominal[digit]/(nominal[digit]+q_probs[digit])) if name == "perturbation" else 1.,
                    "empirical_reference": float(empirical) if empirical is not None else None,
                    "reference_p_digit_probability": float(frequencies[digit]),
                    "q_digit_probability": float(q_probs[digit]),
                    "reference_scope": "label-only target assumes common within-digit law; nominal additionally assumes uniform P" if name == "perturbation" else "common-source null target"})
    return rows


def verify_branch(output, name):
    marker = ms.read(output / name / "complete.json")
    if marker["protocol_sha256"] != ms.sha(output / "protocol.json") or set(marker["files"]) != set(FILES):
        raise ValueError(f"Wrong result manifest: {name}")
    for relative, digest in marker["files"].items():
        if ms.sha(output / name / relative) != digest:
            raise ValueError(f"Result changed: {name}/{relative}")
    return ms.read(output / name / "metrics.json")


def train_branch(output, name, device):
    if name not in NAMES:
        raise ValueError(name)
    protocol = load_protocol(output)
    cfg, chosen = protocol["config"], protocol["chosen"]
    split = ms.read(output / "splits.json")[name]
    folder = output / name
    if (folder / "complete.json").is_file():
        return verify_branch(output, name)
    folder.mkdir(parents=True, exist_ok=True)
    alpha, batch_size = chosen["output_alpha"], cfg["batch_size"]
    model = ms.make_model(cfg, 0 if name == "perturbation" else 1, alpha, device)
    train = [read_images(output, split["train"]["source"], split["train"][side]) for side in ("p", "q")]
    early = [read_images(output, split["validation"]["source"], split["validation"][side]) for side in ("p", "q")]
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["max_lr"],
        total_steps=cfg["max_epochs"]*math.ceil(len(train[0])/batch_size), pct_start=.1,
        anneal_strategy="cos", div_factor=10, final_div_factor=1000)
    rngs = [torch.Generator().manual_seed(stream_seed(cfg, name, f"training_{side}")) for side in ("p", "q")]
    best, best_state, best_epoch, stale, history = float("inf"), None, None, 0, []
    started = time.monotonic()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        if epoch >= cfg["bn_freeze_epoch"]:
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        orders = [torch.randperm(len(images), generator=rng).split(batch_size) for images, rng in zip(train, rngs)]
        total_risk = 0.
        for p_indices, q_indices in zip(*orders):
            p, q = [ms.to_pixels(images[indices], device) for images, indices in zip(train, (p_indices, q_indices))]
            logits = alpha*model(torch.cat((p, q)))
            risk = ms.midpoint_loss_from_logits(logits[:len(p)], logits[len(p):], chosen["loss"])
            if not torch.isfinite(risk):
                raise FloatingPointError(f"Nonfinite objective at epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            risk.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total_risk += float(risk.detach())*len(p)
        brier = balanced_brier(*[ms.predict(model, images, alpha, batch_size, device) for images in early])
        if brier < best:
            best, best_epoch, stale = brier, epoch+1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch+1, "training_objective": total_risk/len(train[0]),
                        "earlystop_brier": brier, "lr": optimizer.param_groups[0]["lr"]})
        print(f"{name} epoch={epoch+1} validation Brier={brier:.7f}", flush=True)
        if epoch+1 >= cfg["min_epochs"] and stale >= cfg["patience"]:
            break
    fit_seconds = time.monotonic()-started
    torch.save({"model": best_state, "chosen": chosen, "best_epoch": best_epoch,
                "protocol_sha256": ms.sha(output / "protocol.json")}, folder / "model.pt")
    model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True)["model"])
    restored = balanced_brier(*[ms.predict(model, images, alpha, batch_size, device) for images in early])
    if abs(restored-best) > 1e-10:
        raise AssertionError("Minimum-Brier checkpoint was not restored")
    # No final images or fresh diagnostic scores are accessed above this point.
    labels = {"official_train": read_labels(output / "assets" / RAW_NAMES[1]),
              "official_test": read_labels(output / "assets" / RAW_NAMES[3])}
    scores = {}
    for role in ("train", "validation", "test"):
        if role not in split:
            continue
        source = split[role]["source"]
        scores[role] = {}
        for side_index, side in enumerate(("p", "q")):
            indices = split[role][side]
            images = train[side_index] if role == "train" else early[side_index] if role == "validation" else read_images(output, source, indices)
            values = ms.predict(model, images, alpha, batch_size, device)
            scores[role][side] = score_record(values, labels[source], indices, source)
    reference = split["reference"]
    reference_indices, source = reference["indices"], reference["source"]
    reference_images = read_images(output, source, reference_indices)
    reference_scores = ms.predict(model, reference_images, alpha, batch_size, device)
    lookup = {index: position for position, index in enumerate(reference_indices)}
    draws = diagnostic_indices(cfg, name, labels[source], reference_indices)
    diagnostics = {"scope": CI_SCOPE, "reference": score_record(reference_scores, labels[source], reference_indices, source)}
    for role in ("calibration", "evaluation"):
        diagnostics[role] = {}
        for side in ("p", "q"):
            indices = draws[role][side]
            values = reference_scores[[lookup[index] for index in indices]]
            diagnostics[role][side] = score_record(values, labels[source], indices, source)
    summary, cells = calibration_diagnostics(
        *[diagnostics[role][side]["scores"].numpy() for role in ("calibration", "evaluation") for side in ("p", "q")],
        alpha=cfg["ci_alpha"], bins=cfg["bins"])
    primary_role = "test" if name == "perturbation" else "validation"
    primary_brier = balanced_brier(*[scores[primary_role][side]["scores"].numpy() for side in ("p", "q")])
    all_draws = [set(draws[role][side]) for role in ("calibration", "evaluation") for side in ("p", "q")]
    metrics = {**chosen, **summary, "experiment": name, "best_epoch": best_epoch, "epochs": len(history),
        "earlystop_brier": best, "primary_role": primary_role, "primary_brier": primary_brier,
        "test_brier": primary_brier if name == "perturbation" else None,
        "train_brier": balanced_brier(*[scores["train"][side]["scores"].numpy() for side in ("p", "q")]),
        "diagnostic_evaluation_brier": balanced_brier(*[diagnostics["evaluation"][side]["scores"].numpy() for side in ("p", "q")]),
        "diagnostic_scope": CI_SCOPE, "reference_n": len(reference_indices),
        "counts": {role: {side: len(record[side]["scores"]) for side in ("p", "q")} for role, record in scores.items()},
        "unique_primary_images": {role: {side: len(set(record[side]["original_indices"].tolist())) for side in ("p", "q")} for role, record in scores.items()},
        "unique_diagnostic_images": {role: {side: len(set(draws[role][side])) for side in ("p", "q")} for role in draws},
        "calibration_evaluation_image_overlap": len((all_draws[0] | all_draws[1]) & (all_draws[2] | all_draws[3])),
        "has_independent_final_role": name == "perturbation", "fit_seconds": fit_seconds,
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "selection_sha256": protocol["selection_sha256"], "smoke": protocol["smoke"]}
    torch.save(scores, folder / "scores.pt")
    torch.save(diagnostics, folder / "diagnostics.pt")
    ms.write(folder / "metrics.json", metrics)
    ms.write(folder / "history.json", history)
    ms.write(folder / "cells.json", cells)
    ms.write(folder / "digits.json", digit_rows(scores, name, labels[source][reference_indices]))
    ms.complete(folder, output, FILES)
    return metrics


def report(output):
    protocol = load_protocol(output)
    rows = [verify_branch(output, name) for name in NAMES]
    from experiments.MNIST.controls_plot import render
    render(output)
    chosen = protocol["chosen"]
    lines = ["# MNIST selected-model perturbation and null", "",
        f"Transferred DCGAN selection: **{chosen['loss'].upper()}, r = 2 sigmoid({chosen['output_alpha']:g} z)**.",
        "Both networks were initialized and trained afresh, without another configuration search.", "",
        "| Experiment | Primary role | Best epoch | Validation Brier | Primary Brier | Conditional local gap | Supported mass |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['primary_role']} | {row['best_epoch']} | {row['earlystop_brier']:.6f} | {row['primary_brier']:.6f} | {row['local_gap']:.6f} | {row['supported_mass']:.6f} |")
    lines += ["", "## Role counts and checkpoint selection", "",
        "Full perturbation protocol: 60,000 P and 60,000 fixed resampled Q training rows; 5,000 P/Q validation rows; 5,000 P/Q held-out evaluation rows.",
        "Its training source is official MNIST train; the two official-test halves provide disjoint validation and evaluation source images. Repeated Q rows are not additional source images.",
        "Full null protocol: disjoint 30,000/30,000 official-training halves; disjoint 5,000/5,000 test-derived validation halves. **The null has no independent final-test evaluation.**",
        "The saved primary indices reproduce historical seed 123/133 partitions and Q seeds 143/153/163. Smoke runs explicitly use smaller pools.",
        "Training uses the selected CNN, batch size 512, Adam/OneCycle (max_lr 0.0006), maximum 20/minimum 6 epochs, patience 5, and BN frozen from epoch index 5. Every training row is used each epoch; historical drop_last=True omitted 96 perturbation rows and 304 null rows per source per epoch.",
        "Checkpoint selection uses validation balanced Brier only. Its state is saved and reloaded before final or fresh diagnostic scoring. All model inputs use [-1,1].", "",
        "## Calibration-CI interpretation", "", CI_SCOPE,
        "Each calibration and diagnostic-evaluation role draws 5,000 P and 5,000 Q rows independently using separate frozen RNG seeds (smoke: 80 per side). They can repeat and share image identities while their random draws are independent conditional on the reference pool.",
        "Perturbation reference: its 5,000 held-out source images, P uniform and Q sampled at the controlled digit probabilities. Null reference: its pooled 10,000 test-derived validation images, P=Q uniform. The latter is an auxiliary common-source validation diagnostic.",
        "C.1/C.2 intervals target cell-average RDR for these empirical distributions. Neural cell means have separate evaluation sampling uncertainty. Broad intervals and small overall gaps do not establish local or individual-image accuracy.",
        "The fixed 20 bins span [0,2]. The following region [0.7,1.2) is reported separately:", "",
        "| Experiment | Middle mixture mass | Supported fraction | Conditional local gap |",
        "| --- | ---: | ---: | ---: |"]
    def number(value):
        return "NA" if value is None else f"{value:.6f}"
    for row in rows:
        lines.append(f"| {row['experiment']} | {number(row['middle_mass'])} | {number(row['middle_supported_fraction'])} | {number(row['middle_local_gap'])} |")
    lines += ["", "Controlled Q digit probabilities are [0,0,.025,.025,.10,.10,.10,.10,.275,.275]. The nominal reference ratios are [2,2,1.6,1.6,1,1,1,1,0.533333,0.533333], assuming uniform P and a common within-digit distribution. The digit plot also shows the empirical source-pool frequency reference; no MAE/MSE criterion is used.",
        "Under the common-source null the target is r=1 and its population Brier baseline is 0.25. Historical official-test inspection makes these post-selection analyses retrospective.", "",
        "![Digit perturbation and validation null](mnist_selected_controls.png)", "",
        "![Conditional finite-pool calibration cells](mnist_controls_calibration.png)", "",
        "The run contains frozen raw inputs, source, index manifest, checkpoints, all primary and diagnostic scores/indices, digit tables, histories, cells, and SHA256 completion manifests. `report` replays tables/figures from saved scores without training.",
        f"Selection SHA256: `{protocol['selection_sha256']}`.",
        f"Protocol SHA256: `{ms.sha(output / 'protocol.json')}`.",
        f"Smoke run: `{protocol['smoke']}`.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    outputs = ("RESULTS.md", "figure_manifest.json", "mnist_selected_controls.png", "mnist_selected_controls.pdf",
               "mnist_controls_calibration.png", "mnist_controls_calibration.pdf", "perturbation/complete.json", "null/complete.json")
    ms.write(output / "complete.json", {"protocol_sha256": ms.sha(output / "protocol.json"),
        "files": {name: ms.sha(output / name) for name in outputs}})
    (output / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"Completed {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--experiment", choices=NAMES, default="perturbation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    output, device = args.output.resolve(), torch.device(args.device)
    if args.command == "prepare":
        prepare(output, args.data_root.resolve(), args.evidence_root, args.smoke)
    elif args.command == "train":
        train_branch(output, args.experiment, device)
    elif args.command == "report":
        report(output)
    else:
        for name in NAMES:
            train_branch(output, name, device)
        report(output)


if __name__ == "__main__":
    main()
