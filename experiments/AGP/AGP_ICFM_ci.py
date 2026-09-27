#!/usr/bin/env python3
"""Strict-split AGP real-versus-ICFM RDR with fixed-bin calibration CIs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time

import numpy as np
import pandas as pd
import scipy
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils.calibration import bin_scores, calibrate_counts
from utils.networks import MLP
from utils.losses import concatenated_midpoint_hellinger_loss as midpoint_loss
from AGP_ci_data import prepare_data

DEFAULT_DATA = Path("/hpc/group/mastatlab/yx306/AGP/data")
DEFAULT_OUTPUT = Path("/cwork/yx306/RDR/JRSSB/CI/agp_icfm_20260918")
ROLES = ("train", "validation", "calibration", "query")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fit(arrays, args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    tensors = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in arrays.items()
               if key[1] in ("train", "validation")}
    model = MLP(input_dim=arrays["P", "train"].shape[1], hidden_dim=32, output_alpha=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=5, cooldown=2)
    best, best_epoch, best_state = float("inf"), None, None
    stopping_best, stale = float("inf"), 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = midpoint_loss(model, tensors["P", "train"], tensors["Q", "train"])
        require(bool(torch.isfinite(loss)), "Nonfinite training objective")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = midpoint_loss(model, tensors["P", "validation"], tensors["Q", "validation"])
        value = float(validation)
        require(np.isfinite(value), "Nonfinite validation objective")
        history.append({"epoch": epoch, "train_loss": float(loss.detach()),
                        "validation_loss": value, "learning_rate": optimizer.param_groups[0]["lr"]})
        if value < best:
            best, best_epoch, best_state = value, epoch, copy.deepcopy(model.state_dict())
        if epoch >= args.min_epochs:
            if value < stopping_best - 1e-5:
                stopping_best, stale = value, 0
            else:
                stale += 1
        scheduler.step(value)
        if epoch == 1 or epoch % 100 == 0:
            print(f"Epoch {epoch}: train={float(loss.detach()):.6f}, validation={value:.6f}", flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored = float(midpoint_loss(model, tensors["P", "validation"], tensors["Q", "validation"]))
    require(abs(restored-best) < 1e-7, "Restored model is not the minimum-validation checkpoint")
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "best_validation_loss": best,
        "epochs_run": len(history), "restore_mode": "absolute_minimum_validation",
        "stop_reason": "validation_patience" if len(history) < args.max_epochs else "max_epochs"}


def score_rows(model, arrays, manifest, edges):
    parts = []
    for distribution in ("P", "Q"):
        for role in ROLES:
            values = arrays[distribution, role]
            with torch.no_grad():
                scores = np.concatenate([model(torch.as_tensor(chunk, dtype=torch.float32))[:, 0].numpy()
                                         for chunk in np.array_split(values, max(1, (len(values)+2047)//2048))])
            rows = manifest[manifest.distribution.eq(distribution) & manifest.role.eq(role)].copy()
            require(len(rows) == len(scores), "Manifest and tensor row counts differ")
            rows["rdr"] = scores.astype(float)
            rows["bin_index"] = bin_scores(scores, edges)
            parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def calibrate(scores, edges, alpha):
    cal = scores[scores.role.eq("calibration")]
    p = cal[cal.distribution.eq("P")]
    q = cal[cal.distribution.eq("Q")]
    kp = np.bincount(p.bin_index, minlength=len(edges)-1)
    kq = np.bincount(q.bin_index, minlength=len(edges)-1)
    result = calibrate_counts(kp, kq, len(p), len(q), alpha)
    cells = pd.DataFrame({"bin_index": np.arange(len(edges)-1), "left": edges[:-1], "right": edges[1:],
                          "right_closed": np.arange(len(edges)-1) == len(edges)-2,
                          "n_p_calibration": len(p), "n_q_calibration": len(q), "k_p": kp, "k_q": kq,
                          "p_hat": kp/len(p), "q_hat": kq/len(q), **result})
    query = scores[scores.role.eq("query")].merge(cells, on="bin_index", how="left", validate="many_to_one")
    summary = []
    for method in ("c1", "c2"):
        excluded = (cells[f"{method}_lower"] > 1) | (cells[f"{method}_upper"] < 1)
        for distribution in ("P", "Q", "equal_mixture"):
            selected = query if distribution == "equal_mixture" else query[query.distribution.eq(distribution)]
            if distribution == "equal_mixture":
                weights = np.where(selected.distribution.eq("P"), .5 / sum(selected.distribution.eq("P")),
                                   .5 / sum(selected.distribution.eq("Q")))
            else:
                weights = np.full(len(selected), 1/len(selected))
            lo, hi = selected[f"{method}_lower"], selected[f"{method}_upper"]
            summary.append({"method": method, "distribution": distribution, "n_query": len(selected),
                            "mean_width": float(np.sum(weights*(hi-lo))),
                            "fraction_excluding_one": float(np.sum(weights*((lo > 1) | (hi < 1)))),
                            "n_cells_excluding_one": int(excluded.sum()),
                            "n_full_range_cells": int(((cells[f"{method}_lower"] == 0) &
                                                       (cells[f"{method}_upper"] == 2)).sum()),
                            "c1_unavailable_query_fraction": float(np.sum(weights*selected.c1_unavailable))})
    return cells, query, pd.DataFrame(summary)


def plot(out, cells, scores, history, alpha):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.r_[cells.left, cells.right.iloc[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax = axes[0]
    for method, color in (("c2", "#b9cde0"), ("c1", "#4189bb")):
        low, high = cells[f"{method}_lower"].to_numpy(), cells[f"{method}_upper"].to_numpy()
        ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post", color=color,
                        alpha=.65, label="C.2 simultaneous" if method == "c2" else "C.1 marginal")
    ax.stairs(cells.estimate, edges, color="#182f44", label="Calibrated cell estimate")
    ax.plot([0, 2], [0, 2], ":", color=".4")
    ax.axhline(1, ls="--", color=".4", lw=.8)
    ax.set(xlim=(0, 2), ylim=(0, 2), xlabel="Frozen AGP RDR score", ylabel="Cell-average RDR")
    ax.legend(fontsize=9)
    cal = scores[scores.role.eq("calibration")]
    axes[1].hist([cal[cal.distribution.eq(d)].rdr for d in ("P", "Q")], bins=edges,
                 label=["P: observed AGP", "Q: ICFM"], alpha=.65)
    axes[1].set(xlim=(0, 2), xlabel="Frozen AGP RDR score", ylabel="Calibration count")
    axes[1].legend(fontsize=9)
    fig.suptitle(f"AGP: nominal {100*(1-alpha):g}% calibration CIs under IID sampling")
    fig.tight_layout()
    fig.savefig(out / "agp_icfm_ci.png", dpi=180)
    fig.savefig(out / "agp_icfm_ci.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(history.epoch, history.train_loss, label="Training")
    ax.plot(history.epoch, history.validation_loss, label="Validation")
    ax.axvline(history.loc[history.validation_loss.idxmin(), "epoch"], color=".4", ls="--")
    ax.set(xlabel="Epoch", ylabel="Hellinger objective")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_history.png", dpi=160)
    plt.close(fig)


def report(out, protocol, training, cells, summary):
    manifest = pd.read_csv(out / "split_manifest.csv", dtype={"sample_id": str, "subject_id": str})
    text = ["# AGP: real versus ICFM calibration confidence intervals", "",
        "This is a new RDR fit with disjoint training, validation, calibration, and query roles, using "
        "archived 614-taxon observed compositions and one archived ICFM sample bank. It is a retrospective "
        "application to existing data, not a reproduction of the historical fitted RDR or a newly sealed experiment.", "",
        "## Target and loss", "",
        "For $M=(P+Q)/2$, $r_0(x)=2p(x)/(p(x)+q(x))$. With frozen score cells "
        "$A_j=\\{x:\\hat r(x)\\in I_j\\}$, the CI target is", "",
        "$$\\theta_j=\\frac{2P(A_j)}{P(A_j)+Q(A_j)}=\\mathbb E_M[r_0(X)\\mid X\\in A_j].$$", "",
        "Training uses the original 614–32–32–32–1 MLP with output $2\\operatorname{sigmoid}(2a)$, "
        "raw compositions without new normalization, AdamW (learning rate 0.0005, weight decay 0.01), "
        "gradient clipping at 1, and the original plateau scheduler. The empirical midpoint objective is", "",
        "$$L=\\tfrac12\\overline{r_P^{-1/2}}+\\tfrac14\\overline{\\sqrt{r_P}}"
        "+\\tfrac14\\overline{\\sqrt{r_Q}}-1.$$", "",
        "Each distribution's mean has its own denominator. Thus the target remains the equally weighted midpoint "
        "despite unequal sample sizes. This is the existing Hellinger objective written as separate P and Q means; "
        "there is no classifier-to-RDR conversion. The absolute minimum-validation checkpoint is restored.", "",
        "## Split and provenance", "",
        "The historical notebook trained RDR on nominal test observations and reused that mixture for validation. "
        "Those fits are not reused. Canonical real row IDs are reconstructed from abundance columns and the original "
        "seed-1 train/test split, verified against the raw matrices. Subject metadata is joined by exact sample ID; "
        "the archived metadata row order is not trusted.", "",
        "Original real training subjects are split between RDR fitting and validation. Original real test rows whose "
        "subjects appeared in generator training are excluded; the remaining subjects are split between calibration "
        "and query, keeping all samples from one subject together. P remains a sample-level target; it is not "
        "silently replaced by a one-sample-per-person distribution. Inference pertains to this held-out AGP cohort, "
        "not an established claim for all human microbiomes.", "",
        "Only `yx1_icfm_recon_train.csv` supplies Q for all four roles, with random proportions 60%/20%/10%/10%. "
        "The generated test bank is excluded because historical generation reused seed 2 and its paired prefix is "
        "strongly dependent on the training bank. No ICFM model is retrained. Archived generator-training provenance "
        "comes from the preprocessing/generation code; complete historical latent-seed manifests are unavailable.", "",
        "| Role | P samples | P subjects | Q samples |", "|---|---:|---:|---:|"]
    for role in ROLES:
        p = manifest[manifest.distribution.eq("P") & manifest.role.eq(role)]
        q = manifest[manifest.distribution.eq("Q") & manifest.role.eq(role)]
        text.append(f"| {role} | {len(p)} | {p.subject_id.nunique()} | {len(q)} |")
    text.extend(["", f"Selected epoch: **{training['best_epoch']}** of {training['epochs_run']} epochs; "
                 f"validation loss {training['best_validation_loss']:.6f}. Calibration and query values never "
                 "enter optimization, early stopping, or partition choice. All 20 equal-width bins on [0,2] "
                 "are fixed before fitting; the last includes score 2.", "",
        "## Intervals and limitations", "",
        "C.1 gives nominal 95% asymptotic marginal cell intervals. C.2 uses the unchanged Clopper–Pearson algorithm "
        "with Bonferroni allocation across all 20 bins, including empty bins, for nominal 95% simultaneous coverage "
        "under independent P/Q sampling. No pseudocounts are used. Empty or zero-standard-error cells receive "
        "the prescribed C.1 [0,2] fallback; empty-cell estimate 1 is only a storage convention. C.2 uses its "
        "binomial bounds even in endpoint cases.", "",
        "**Sampling limitation:** subject separation prevents cross-role subject leakage but multiple stool samples "
        "from a subject can remain within a role. These unchanged intervals do not adjust for within-subject dependence. "
        "They are nominal under independent-sample and fixed-generator assumptions, whose full historical provenance "
        "cannot be certified from these archived inputs. Feature selection/preprocessing also preceded this rerun. "
        "The intervals quantify calibration uncertainty conditional on the fitted score map; they do not cover "
        "individual true RDR values, Hellinger divergence, or model-training variation. True AGP cell ratios are "
        "unknown, so no observed coverage rate is claimed.", "",
        "## Results", "", "![AGP calibration CIs](agp_icfm_ci.png)", "",
        "Equal-mixture summaries give P and Q total weight 1/2 each, using only query samples. "
        "Excluding 1 concerns a cell average; containing 1 does not establish equality.", "",
        "| Method | Query group | Mean width | Fraction excluding 1 | Cells excluding 1 | Full [0,2] cells |",
        "|---|---|---:|---:|---:|---:|"])
    for row in summary.itertuples():
        text.append(f"| {row.method} | {row.distribution} | {row.mean_width:.4f} | "
                    f"{row.fraction_excluding_one:.4f} | {row.n_cells_excluding_one} / 20 | {row.n_full_range_cells} |")
    text.extend(["", "| Score cell | P count | Q count | Cell estimate | C.1 | C.2 |",
                 "|---|---:|---:|---:|---|---|"])
    for row in cells.itertuples():
        text.append(f"| [{row.left:.1f}, {row.right:.1f}{']' if row.right_closed else ')'} | {row.k_p} | {row.k_q} | "
                    f"{row.estimate:.4f} | [{row.c1_lower:.4f}, {row.c1_upper:.4f}] | "
                    f"[{row.c2_lower:.4f}, {row.c2_upper:.4f}] |")
    command = ["python3", "-B", "experiments/AGP/AGP_ICFM_ci.py", "--data-root", protocol["data_root"],
               "--seed", str(protocol["seed"]), "--max-epochs", str(protocol["max_epochs"]),
               "--min-epochs", str(protocol["min_epochs"]), "--patience", str(protocol["patience"]),
               "--threads", str(protocol["threads"]), "--output-dir", str(DEFAULT_OUTPUT) + "_rerun"]
    text.extend(["", "## Reproduction", "", "Use a new output directory:", "", "```bash", shlex.join(command), "```", "",
        "Add `--replay-from PATH_TO_COMPLETED_OUTPUT` to reuse its verified checkpoint and loss history instead "
        "of fitting again. `protocol.json`, `split_manifest.csv`, `scores.csv`, `model.pt`, and `loss_history.csv` "
        "record the complete fit and split. Cell and query CSVs contain all endpoints. Code snapshots are under "
        "`source/`; `result.json` is written last with artifact hashes.", ""])
    (out / "report.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--min-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    require(0 < args.min_epochs <= args.max_epochs and args.patience > 0 and args.threads > 0, "Invalid fit controls")
    out = args.output_dir.resolve()
    require(out.is_relative_to(Path("/tmp")) or out.is_relative_to(Path("/cwork")), "Outputs must be under /tmp or /cwork")
    require(not out.exists() or not any(out.iterdir()), "Output directory must be new or empty")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    arrays, manifest, audit = prepare_data(args.data_root.resolve(), args.seed)
    edges = np.linspace(0, 2, 21)
    protocol = {"analysis": "agp_real_vs_icfm_strict_split_calibration", "data_root": str(args.data_root.resolve()),
                "seed": args.seed, "max_epochs": args.max_epochs, "min_epochs": args.min_epochs,
                "patience": args.patience, "threads": args.threads, "alpha": .05, "bins": 20, "edges": edges.tolist(),
                "data_audit": audit, "model_architecture": "MLP 614-32-32-32-1; 2*sigmoid(2*logit)",
                "loss": ".5*mean_P(r^-1/2)+.25*mean_P(sqrt(r))+.25*mean_Q(sqrt(r))-1",
                "same_population_midpoint_target_with_unequal_sample_sizes": True,
                "retrospective": True, "true_coverage_measured": False, "within_subject_correlation_adjusted": False,
                "code_sha256": {}, "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                                                "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__}}
    source_files = ["experiments/AGP/AGP_ICFM_ci.py", "experiments/AGP/AGP_ci_data.py",
                    "utils/calibration.py", "utils/__init__.py", "utils/networks.py", "utils/losses.py",
                    "experiments/AGP/AGP1_data_preprocessing.py", "experiments/AGP/AGP2_ICFM.py"]
    protocol["code_sha256"] = {name: sha256(REPO/name) for name in source_files}
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "protocol.json", protocol)
    manifest.to_csv(out / "split_manifest.csv", index=False)
    for name in source_files:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    if args.replay_from:
        previous = args.replay_from.resolve()
        result = json.loads((previous / "result.json").read_text())
        require(result["status"] == "complete", "Replay source is incomplete")
        for name, digest in result["outputs"].items():
            require(sha256(previous / name) == digest, f"Replay artifact hash mismatch: {name}")
        require((previous / "split_manifest.csv").read_bytes() == (out / "split_manifest.csv").read_bytes(), "Replay split differs")
        prior_protocol = json.loads((previous / "protocol.json").read_text())
        require(prior_protocol == protocol, "Replay inputs, code, runtime, or protocol differs")
        checkpoint = torch.load(previous / "model.pt", map_location="cpu", weights_only=False)
        require(checkpoint["protocol_sha256"] == sha256(previous / "protocol.json"), "Checkpoint protocol differs")
        model = MLP(input_dim=arrays["P", "train"].shape[1], hidden_dim=32, output_alpha=2)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        history = pd.read_csv(previous / "loss_history.csv")
        training = checkpoint["training"]
        shutil.copyfile(previous / "model.pt", out / "model.pt")
    else:
        model, history, training = fit(arrays, args)
        torch.save({"state_dict": model.state_dict(), "training": training,
                    "protocol_sha256": sha256(out / "protocol.json")}, out / "model.pt")
    history.to_csv(out / "loss_history.csv", index=False)
    scores = score_rows(model, arrays, manifest, edges)
    scores.to_csv(out / "scores.csv", index=False)
    cells, query, summary = calibrate(scores, edges, .05)
    cells.to_csv(out / "cell_intervals.csv", index=False)
    query.to_csv(out / "query_intervals.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    save_json(out / "training.json", training)
    plot(out, cells, scores, history, .05)
    report(out, protocol, training, cells, summary)
    files = [p for p in out.rglob("*") if p.is_file() and ".matplotlib" not in p.parts]
    save_json(out / "result.json", {"status": "complete", "analysis": protocol["analysis"],
        "training": training, "elapsed_seconds": time.monotonic()-started,
        "replay_from": str(args.replay_from.resolve()) if args.replay_from else None,
        "query_rows": len(query), "cells": len(cells), "outputs": {str(p.relative_to(out)): sha256(p) for p in sorted(files)}})
    print(json.dumps({"status": "complete", "output": str(out), "training": training,
                      "query_rows": len(query), "cells": len(cells)}, indent=2))


if __name__ == "__main__":
    main()
