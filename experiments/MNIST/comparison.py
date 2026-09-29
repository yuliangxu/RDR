#!/usr/bin/env python3
"""Refit both MNIST RDRs with the frozen DCGAN-selected configuration.

Preserve the historical 55k/5k/10k real split; select checkpoints by validation
Brier and render new test images/scores. Generator weights remain frozen.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mnist-comparison-matplotlib")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.MNIST import model_selection as ms
from experiments.MNIST.generators import VAE, build_dcgan28
from experiments.MNIST.selection_evidence import load_selected
from experiments.MNIST.selection_metrics import balanced_brier, calibration_diagnostics

FILES = ("model.pt", "evaluation.pt", "metrics.json", "history.json", "cells.json")

def historical_manifest():
    """Reconstruct the exact published 55k/5k/10k split without private files."""
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(52)).tolist()
    return {"seed": 42,
            "train": {"source": "MNIST train minus validation-loss subset", "n": 55000},
            "validation_loss": {"source": "deterministic validation subset of MNIST train", "n": 5000},
            "test": {"source": "all official MNIST test observations", "n": 10000},
            "train_indices": order[5000:], "validation_loss_indices": order[:5000],
            "test_indices": list(range(10000))}


def make_splits(historical, selected, smoke=False):
    train = historical["train_indices"]
    earlystop = historical["validation_loss_indices"]
    test = historical["test_indices"]
    if (len(train) != 55000 or len(earlystop) != 5000
            or len(set(train + earlystop)) != 60000
            or set(train + earlystop) != set(range(60000))
            or test != list(range(10000))):
        raise ValueError("Historical 55k/5k/10k split changed")
    cal = selected["final_calibration"]["indices"]
    evaluate = selected["final_evaluation"]["indices"]
    if len(cal) != 5000 or len(evaluate) != 5000 or set(cal + evaluate) != set(test):
        raise ValueError("Expected disjoint 5k/5k final diagnostic partition")
    if smoke:
        train, earlystop = train[:32], earlystop[:20]
        cal, evaluate = cal[:64], evaluate[:64]
        test = sorted(cal + evaluate)
    positions = {index: i for i, index in enumerate(test)}
    return {"train": {"source": "official_train", "indices": train},
            "earlystop": {"source": "official_train", "indices": earlystop},
            "test": {"source": "official_test", "indices": test},
            "calibration_positions": [positions[i] for i in cal],
            "evaluation_positions": [positions[i] for i in evaluate]}


def prepare(output, selection_root, historical_root, data_root, smoke=False):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    evidence = load_selected(selection_root)
    selected_protocol = ms.read(evidence["files"]["protocol.json"])
    selection = evidence["selection"]
    historical = historical_manifest()
    if historical_root is not None:
        supplied = ms.read(historical_root / "split_manifest.json")
        if any(supplied[key] != historical[key] for key in ("train_indices", "validation_loss_indices", "test_indices")):
            raise ValueError("Supplied historical split differs from the published seed-52 split")
    splits = make_splits(historical, ms.read(evidence["files"]["splits.json"]), smoke)
    cfg = dict(evidence["config"])
    cfg.update(seed=2026092801, train_n=len(splits["train"]["indices"]),
               earlystop_n=len(splits["earlystop"]["indices"]),
               test_n=len(splits["test"]["indices"]))
    if smoke:
        cfg.update(batch_size=16, max_epochs=2, min_epochs=1, patience=1, base_channels=4)
    inputs = {
        "assets/netG_epoch_99.pth": data_root / "mnist_dcgan/netG_epoch_99.pth",
        "assets/vae_epoch_25.pth": data_root / "mnist_vae/vae_epoch_25.pth",
        "selection.json": evidence["files"]["selection.json"],
        "selection.sha256": evidence["files"]["selection.sha256"],
        "selection_protocol.json": evidence["files"]["protocol.json"],
    }
    for name in ("train-images-idx3-ubyte", "train-labels-idx1-ubyte",
                 "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"):
        inputs[f"assets/{name}"] = data_root / "MNIST/raw" / name
    for relative, metadata in selected_protocol["files"].items():
        if relative.startswith("assets/") and ms.sha(inputs[relative]) != metadata["sha256"]:
            raise ValueError(f"Input differs from the selected study: {relative}")
    for name, source in evidence["files"].items():
        inputs["selection_evidence/" + name] = source
    for source in list((ROOT / "utils").glob("*.py")) + [ROOT / "experiments/__init__.py"]:
        inputs["source/" + str(source.relative_to(ROOT))] = source
    for name in ("__init__.py", "comparison.py", "comparison_plot.py", "comparison.slurm",
                 "model_selection.py", "selection_metrics.py", "selection_report.py",
                 "selection_config.json", "generators.py", "selection_evidence.py"):
        inputs["source/experiments/MNIST/" + name] = Path(__file__).with_name(name)
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
    ms.write(output / "historical_split_manifest.json", historical)
    files["historical_split_manifest.json"] = {
        "sha256": ms.sha(output / "historical_split_manifest.json"),
        "construction": "torch.randperm(60000, private Generator seed52); validation first5000"}
    ms.write(output / "splits.json", splits)
    files["splits.json"] = {"sha256": ms.sha(output / "splits.json")}
    protocol = {
        "study": "post_selection_vae_dcgan_refit", "smoke": smoke,
        "config": cfg, "chosen": selection["chosen"], "files": files,
        "selection_root": str(evidence["files"]["selection.json"].parent),
        "selection_sha256": ms.sha(evidence["files"]["selection.json"]),
        "target": "r=2p/(p+q); all model inputs [-1,1]",
        "checkpoint_rule": "minimum fixed validation balanced Brier; test access after checkpoint save and restore",
        "sampling": "new initialization; matched real order; branch-specific private latent streams for training, validation and test",
        "comparison_scope": "DCGAN-selected configuration transferred to VAE without VAE search; both max20/min6/patience5/BNfreeze5",
        "limitations": "post-selection refit; official test inspected previously; generator pretraining membership unaudited; not a loss-only comparison with historical fits",
        "python": sys.version, "packages": {p: importlib.metadata.version(p)
            for p in ("torch", "torchvision", "numpy", "scipy", "matplotlib")},
    }
    ms.write(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(ms.sha(output / "protocol.json") + "\n")
    print(f"Prepared {output}; chosen={selection['chosen']}; smoke={smoke}", flush=True)


def load_protocol(output):
    protocol = ms.load_protocol(output)
    if ms.sha(output / "selection.json") != protocol["selection_sha256"]:
        raise ValueError("Selection lock mismatch")
    if ms.read(output / "selection.json")["chosen"] != protocol["chosen"]:
        raise ValueError("Refit configuration differs from selected configuration")
    return protocol


def load_generator(output, name, device):
    if name == "dcgan":
        generator, _ = build_dcgan28(output / "assets/netG_epoch_99.pth", device=device)
        latent = 100
    elif name == "vae":
        generator = VAE().to(device)
        generator.load_state_dict(torch.load(output / "assets/vae_epoch_25.pth", map_location=device, weights_only=True))
        latent = 20
    else:
        raise ValueError(name)
    return generator.eval().requires_grad_(False), latent


def stream_seed(cfg, name, role):
    return int(np.random.SeedSequence([cfg["seed"], {"vae": 201, "dcgan": 202}[name],
        {"training_q": 1, "earlystop": 2, "test": 3}[role]]).generate_state(1)[0])


@torch.no_grad()
def sample_batch(generator, name, n, rng, device):
    if name == "vae":
        z = torch.randn(n, 20, generator=rng, device=device)
        images = generator.decode(z).reshape(n, 1, 28, 28).mul(2).sub(1)
    elif name == "dcgan":
        images = generator(torch.randn(n, 100, 1, 1, generator=rng, device=device))
    else:
        raise ValueError(name)
    if images.shape != (n, 1, 28, 28) or not torch.isfinite(images).all() or images.min() < -1 or images.max() > 1:
        raise ValueError("Generator produced invalid model-space images")
    return images


def sample_images(generator, name, n, rng, batch_size, device):
    return torch.cat([sample_batch(generator, name, min(batch_size, n-i), rng, device).cpu()
                      for i in range(0, n, batch_size)])


def verify_branch(output, name):
    folder = output / name
    marker = ms.read(folder / "complete.json")
    if marker["protocol_sha256"] != ms.sha(output / "protocol.json") or set(marker["files"]) != set(FILES):
        raise ValueError(f"Wrong result manifest: {name}")
    for file, digest in marker["files"].items():
        if ms.sha(folder / file) != digest:
            raise ValueError(f"Result changed: {folder/file}")
    return ms.read(folder / "metrics.json")


def train_branch(output, name, device):
    protocol = load_protocol(output)
    cfg, chosen = protocol["config"], protocol["chosen"]
    folder = output / name
    if (folder / "complete.json").exists():
        return verify_branch(output, name)
    folder.mkdir(parents=True, exist_ok=True)
    generator, _ = load_generator(output, name, device)
    alpha, batch_size = chosen["output_alpha"], cfg["batch_size"]
    model = ms.make_model(cfg, 0, alpha, device)
    train = ms.real_images(output, "train")
    early_p = ms.real_images(output, "earlystop")
    def rng(role):
        return torch.Generator(device=device).manual_seed(stream_seed(cfg, name, role))
    early_q = sample_images(generator, name, len(early_p), rng("earlystop"), batch_size, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(.9, .999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg["max_lr"],
        total_steps=cfg["max_epochs"]*math.ceil(len(train)/batch_size), pct_start=.1,
        anneal_strategy="cos", div_factor=10, final_div_factor=1000)
    latent_rng = rng("training_q")
    order_rng = torch.Generator().manual_seed(ms.seed_for(cfg, 0, "training_order"))
    best, best_state, best_epoch, stale, history = float("inf"), None, None, 0, []
    started = time.monotonic()
    for epoch in range(cfg["max_epochs"]):
        model.train()
        if epoch >= cfg["bn_freeze_epoch"]:
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        total_risk = 0.
        for indices in torch.randperm(len(train), generator=order_rng).split(batch_size):
            p = ms.to_pixels(train[indices], device)
            q = sample_batch(generator, name, len(p), latent_rng, device)
            logits = alpha * model(torch.cat((p, q)))
            risk = ms.midpoint_loss_from_logits(logits[:len(p)], logits[len(p):], chosen["loss"])
            if not torch.isfinite(risk):
                raise FloatingPointError(f"Nonfinite training objective, epoch {epoch+1}")
            optimizer.zero_grad(set_to_none=True)
            risk.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total_risk += float(risk.detach())*len(p)
        brier = balanced_brier(*[ms.predict(model, x, alpha, batch_size, device) for x in (early_p, early_q)])
        if brier < best:
            best, best_epoch, stale = brier, epoch+1, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        history.append({"epoch": epoch+1, "training_objective": total_risk/len(train),
                        "earlystop_brier": brier, "lr": optimizer.param_groups[0]["lr"]})
        print(f"{name} epoch={epoch+1} validation Brier={brier:.7f}", flush=True)
        if epoch+1 >= cfg["min_epochs"] and stale >= cfg["patience"]:
            break
    fit_seconds = time.monotonic()-started
    # Save and reload the selected checkpoint before opening any final images.
    torch.save({"model": best_state, "chosen": chosen, "best_epoch": best_epoch,
                "protocol_sha256": ms.sha(output / "protocol.json")}, folder / "model.pt")
    model.load_state_dict(torch.load(folder / "model.pt", map_location=device, weights_only=True)["model"])
    restored = balanced_brier(*[ms.predict(model, x, alpha, batch_size, device) for x in (early_p, early_q)])
    if abs(restored-best) > 1e-10:
        raise AssertionError("Best-Brier checkpoint was not restored")
    p_images = ms.real_images(output, "test")
    q_images = sample_images(generator, name, len(p_images), rng("test"), batch_size, device)
    p_scores, q_scores = [ms.predict(model, x, alpha, batch_size, device) for x in (p_images, q_images)]
    splits = ms.read(output / "splits.json")
    cal, evaluate = splits["calibration_positions"], splits["evaluation_positions"]
    ncal = len(cal)
    summary, cells = calibration_diagnostics(p_scores[cal], q_scores[:ncal],
        p_scores[evaluate], q_scores[ncal:], alpha=cfg["ci_alpha"], bins=cfg["bins"])
    labels = np.frombuffer((output / "assets/t10k-labels-idx1-ubyte").read_bytes(), dtype=np.uint8, offset=8)
    indices = splits["test"]["indices"]
    torch.save({"p_images": p_images, "q_images": q_images,
                "p_scores": torch.from_numpy(p_scores), "q_scores": torch.from_numpy(q_scores),
                "p_labels": torch.from_numpy(labels[indices].astype(np.int64)),
                "p_indices": torch.tensor(indices)}, folder / "evaluation.pt")
    metrics = {**chosen, **summary, "generator": name, "best_epoch": best_epoch,
        "epochs": len(history), "earlystop_brier": best,
        "test_brier": balanced_brier(p_scores, q_scores),
        "diagnostic_evaluation_brier": balanced_brier(p_scores[evaluate], q_scores[ncal:]),
        "train_p_per_epoch": len(train), "train_q_per_epoch": len(train),
        "earlystop_p": len(early_p), "earlystop_q": len(early_q),
        "test_p": len(p_images), "test_q": len(q_images),
        "calibration_p": ncal, "calibration_q": ncal,
        "diagnostic_evaluation_p": len(evaluate), "diagnostic_evaluation_q": len(q_scores)-ncal,
        "fit_seconds": fit_seconds, "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "selection_sha256": protocol["selection_sha256"], "smoke": protocol["smoke"]}
    ms.write(folder / "metrics.json", metrics)
    ms.write(folder / "history.json", history)
    ms.write(folder / "cells.json", cells)
    ms.complete(folder, output, FILES)
    return metrics


def report(output):
    protocol = load_protocol(output)
    rows = [verify_branch(output, name) for name in ("vae", "dcgan")]
    from experiments.MNIST.comparison_plot import render
    render(output)
    chosen = protocol["chosen"]
    lines = ["# MNIST comparison after model selection", "",
        f"Frozen DCGAN selection: **{chosen['loss'].upper()}, r = 2 sigmoid({chosen['output_alpha']:g} z)**.",
        "The same configuration is transferred to VAE without another model search.",
        "Both RDR networks were initialized and trained afresh; pretrained generators stayed fixed.", "",
        "| Generator | Best epoch | Validation Brier | Whole-test Brier | Local absolute gap | Supported mass |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['generator'].upper()} | {row['best_epoch']} | {row['earlystop_brier']:.6f} | {row['test_brier']:.6f} | {row['local_gap']:.6f} | {row['supported_mass']:.6f} |")
    cfg = protocol["config"]
    lines += ["", f"Real data: {cfg['train_n']:,} fitting, {cfg['earlystop_n']:,} fixed validation, {cfg['test_n']:,} official test images.",
        "The real split exactly follows the historical comparison (except explicitly marked smoke runs).",
        "Q has the same counts: fresh draws each training epoch, fixed independent validation draws, and independent test draws.",
        "Both branches use the selected CNN/Adam/OneCycle recipe: maximum 20 epochs, minimum 6, Brier patience 5, BN frozen from zero-based epoch 5.",
        "The checkpoint is saved and restored before final-data scoring. All model inputs use [-1,1]; display alone uses [0,1].", "",
        "Whole-test Brier uses all 10,000 P and 10,000 Q scores. Local discrepancy uses disjoint 5,000-P/5,000-Q calibration and 5,000-P/5,000-Q evaluation roles (smoke counts are smaller).",
        "The real diagnostic partition is the frozen selection study's official-test partition; Q is split into two independent halves of its new test draw.",
        "C.1/C.2 intervals concern cell-average true RDR; neural cell means have sampling uncertainty. The saved cells and coverage diagnostics do not provide pointwise image guarantees.", "",
        "This is a post-selection refit. The official test set has been inspected in earlier studies, and generator pretraining membership is unaudited.",
        "Historical RDRs used Hellinger (including a 3-epoch VAE run); this new matched 20-epoch-cap procedure is not a loss-only controlled comparison. Historical results remain preserved.",
        "The mosaics show 40 smallest, closest-to-one, and largest scores per side. These selected illustrations do not estimate prevalence.", "",
        "![Fresh VAE/DCGAN comparison](mnist_vae_dcgan_comparison.png)", "",
        "The PDF, exact panel positions, full saved images/scores, checkpoints, history, calibration cells, frozen inputs/source, and SHA manifests accompany this report.",
        f"Selection SHA256: `{protocol['selection_sha256']}`.",
        f"Protocol SHA256: `{ms.sha(output / 'protocol.json')}`.",
        f"Smoke run: `{protocol['smoke']}`.", ""]
    (output / "RESULTS.md").write_text("\n".join(lines))
    ms.write(output / "complete.json", {"protocol_sha256": ms.sha(output / "protocol.json"),
        "files": {name: ms.sha(output / name) for name in ("RESULTS.md", "figure_manifest.json", "panel_selection.csv", "mnist_vae_dcgan_comparison.png", "mnist_vae_dcgan_comparison.pdf", "vae/complete.json", "dcgan/complete.json")}})
    (output / "COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")
    print(f"Completed {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "train", "report", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, help="Compact published evidence directory; defaults to results/MNIST/model_selection")
    parser.add_argument("--historical-root", type=Path, help="Optional historical split directory for an additional exact-row check")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--generator", choices=("vae", "dcgan"), default="dcgan")
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
        prepare(output, args.selection_root, args.historical_root, args.data_root, args.smoke)
    elif args.command == "train":
        train_branch(output, args.generator, device)
    elif args.command == "report":
        report(output)
    else:
        for name in ("vae", "dcgan"):
            train_branch(output, name, device)
        report(output)


if __name__ == "__main__":
    main()
