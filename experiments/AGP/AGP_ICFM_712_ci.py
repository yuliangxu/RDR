#!/usr/bin/env python3
"""Refit AGP real-versus-ICFM RDR with a roughly 7:1:2 split and full-test CIs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import time

import numpy as np
import pandas as pd
import scipy
import torch

from AGP_ICFM_ci import (REPO, DEFAULT_DATA, MLP, bin_scores, calibrate,
                         fit, midpoint_loss, require, save_json, sha256)
from AGP_ICFM_test_ci import plot as plot_ci
from AGP_712_data import prepare_712_data

DEFAULT_OUTPUT = Path("/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_20260920")
DEFAULT_BASELINE = Path("/cwork/yx306/RDR/JRSSB/CI/agp_icfm_full_test_20260918")
ROLES = ("train", "validation", "test")


def verify_result(root):
    result = json.loads((root / "result.json").read_text())
    require(result["status"] == "complete", f"Incomplete source: {root}")
    for name, digest in result["outputs"].items():
        require(sha256(root / name) == digest, f"Source hash mismatch: {root / name}")
    return result


def score_rows(model, arrays, manifest, edges):
    parts = []
    for distribution in ("P", "Q"):
        for role in ROLES:
            values = arrays[distribution, role]
            rows = manifest[manifest.distribution.eq(distribution) & manifest.role.eq(role)].copy()
            require(len(rows) == len(values), "Manifest/tensor row count mismatch")
            with torch.no_grad():
                scores = np.concatenate([model(torch.as_tensor(chunk, dtype=torch.float32))[:, 0].numpy()
                    for chunk in np.array_split(values, max(1, (len(values)+2047)//2048))])
            rows["rdr"] = scores.astype(float)
            rows["bin_index"] = bin_scores(scores, edges)
            parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def diagnostics(scores, edges):
    """Class-normalized empirical calibration; no inferential claims for fit roles."""
    cells, summaries = [], []
    for role in ROLES:
        selected = scores[scores.role.eq(role)]
        p, q = (selected[selected.distribution.eq(d)] for d in ("P", "Q"))
        kp = np.bincount(p.bin_index, minlength=len(edges)-1)
        kq = np.bincount(q.bin_index, minlength=len(edges)-1)
        ph, qh = kp/len(p), kq/len(q)
        total = ph + qh
        ratio = np.divide(2*ph, total, out=np.full(len(total), np.nan), where=total > 0)
        score_sum = (np.bincount(p.bin_index, weights=p.rdr, minlength=len(total))/len(p)
                     + np.bincount(q.bin_index, weights=q.rdr, minlength=len(total))/len(q))
        score_mean = np.divide(score_sum, total, out=np.full(len(total), np.nan), where=total > 0)
        mass = total/2
        for j in range(len(total)):
            cells.append({"role": role, "bin_index": j, "left": edges[j], "right": edges[j+1],
                "n_p": len(p), "n_q": len(q), "k_p": kp[j], "k_q": kq[j], "mixture_mass": mass[j],
                "cell_rdr": ratio[j], "mean_model_score": score_mean[j],
                "calibration_gap": ratio[j]-score_mean[j], "empty": bool(total[j] == 0),
                "diagnostic_only": True})
        summaries.append({"role": role, "n_p": len(p), "n_q": len(q),
            "mean_score_p": float(p.rdr.mean()), "mean_score_q": float(q.rdr.mean()),
            "equal_mixture_mean_score": float(.5*(p.rdr.mean()+q.rdr.mean())),
            "empirical_bin_absolute_gap": float(np.nansum(mass*np.abs(ratio-score_mean))),
            "cells_above_mean_model_score": int(np.sum(ratio > score_mean)),
            "nonempty_cells": int(np.sum(total > 0)), "diagnostic_only": True})
    return pd.DataFrame(cells), pd.DataFrame(summaries)


def compare_baseline(scores, summary, baseline, edges):
    old_scores = pd.read_csv(baseline / "scores.csv", dtype={"sample_id": str, "subject_id": str},
                             keep_default_na=False)
    previous_test = old_scores[old_scores.role.eq("test")]
    test = scores[scores.role.eq("test")]
    keys = ["distribution", "source_file", "source_row", "sample_id", "subject_id", "row_sha256"]
    require(not previous_test.duplicated(keys).any() and not test.duplicated(keys).any(), "Duplicate test IDs")
    merged = previous_test[keys].merge(test[keys], on=keys, how="outer", indicator=True, validate="one_to_one")
    require(merged._merge.eq("both").all(), "New and previous test observations differ")
    _, old_diagnostics = diagnostics(old_scores, edges)
    _, new_diagnostics = diagnostics(scores, edges)
    old_summary = pd.read_csv(baseline / "summary.csv")
    comparisons = []
    for label, table, desc in (("previous", old_summary, old_diagnostics), ("rerun_712", summary, new_diagnostics)):
        row = desc[desc.role.eq("test")].iloc[0]
        for method in ("c1", "c2"):
            interval = table[table.method.eq(method) & table.distribution.eq("equal_mixture")].iloc[0]
            comparisons.append({"run": label, "method": method,
                "test_n_p": int(row.n_p), "test_n_q": int(row.n_q),
                "equal_mixture_mean_score": row.equal_mixture_mean_score,
                "empirical_bin_absolute_gap": row.empirical_bin_absolute_gap,
                "mean_interval_width": interval.mean_width,
                "fraction_excluding_one": interval.fraction_excluding_one,
                "n_cells_excluding_one": int(interval.n_cells_excluding_one),
                "same_test_observations": True, "same_fitted_score_map": False})
    return pd.DataFrame(comparisons)


def plot_diagnostics(out, history, diagnostics_table):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, role in zip(axes, ROLES):
        rows = diagnostics_table[diagnostics_table.role.eq(role)]
        valid = ~rows["empty"]
        ax.plot(rows.loc[valid, "mean_model_score"], rows.loc[valid, "cell_rdr"], "o-", ms=3)
        ax.plot([0, 2], [0, 2], ":", color=".5")
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel="Within-bin mean model score",
               title=role.title() + (" (used for fitting)" if role == "train" else
                                     " (used for selection)" if role == "validation" else " = calibration"))
    axes[0].set_ylabel("Empirical cell RDR from P/Q frequencies")
    fig.suptitle("Descriptive calibration comparison: common bins, separately normalized P/Q counts")
    fig.tight_layout()
    fig.savefig(out / "split_calibration.png", dpi=180)
    fig.savefig(out / "split_calibration.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(history.epoch, history.train_loss, label="Training")
    ax.plot(history.epoch, history.validation_loss, label="Validation")
    ax.axvline(history.loc[history.validation_loss.idxmin(), "epoch"], color=".4", ls="--", label="Selected epoch")
    ax.set(xlabel="Epoch", ylabel="Midpoint Hellinger objective")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "loss_history.png", dpi=160)
    plt.close(fig)


def report(out, protocol, training, manifest, cells, summary, diagnostic_summary, comparison):
    text = ["# AGP real versus ICFM: approximately 7:1:2 rerun", "",
        "The RDR is refitted from its original seeded initialization after reallocating training and validation. "
        "Test and calibration use the same observations. The ICFM generator, raw 614-taxon inputs, model architecture, "
        "training objective, optimizer, stopping rule, and 20 equal-width score bins are retained.", "",
        "## Split and provenance", "", "| Role | P samples | P subjects | P fraction | Q samples | Q fraction |",
        "|---|---:|---:|---:|---:|---:|"]
    totals = {d: sum(manifest.distribution.eq(d) & manifest.role.isin(ROLES)) for d in ("P", "Q")}
    for role in ROLES:
        p = manifest[manifest.distribution.eq("P") & manifest.role.eq(role)]
        q = manifest[manifest.distribution.eq("Q") & manifest.role.eq(role)]
        text.append(f"| {role} | {len(p):,} | {p.subject_id.nunique():,} | {100*len(p)/totals['P']:.2f}% | "
                    f"{len(q):,} | {100*len(q)/totals['Q']:.2f}% |")
    unused = sum(manifest.distribution.eq("P") & manifest.role.eq("unused"))
    text.extend(["", ("All eligible P samples are retained. With the fixed 2,002-sample generator-unseen test cohort, "
                      "the remaining P pool is divided approximately 7:1 between training and validation; thus its "
                      "overall fractions differ from exactly 70/10/20."
                      if protocol["p_policy"] == "keep-all" else
                      f"P uses approximately 70/10/20 by leaving {unused:,} fit-pool samples unused; complete subject "
                      "groups are allocated using a seeded order and cumulative sample counts."), "",
        "Both P and Q test IDs and input row hashes are exactly the same as the previous full-test analysis. "
        "No test observations enter RDR fitting, checkpoint selection, or bin construction. The 609 real test "
        "observations whose subjects occurred in generator training remain excluded. P subjects are disjoint "
        "between the three current roles. Q uses a single archived bank; its historical second bank remains "
        "excluded because of shared generation seeds. `previous_role` records the earlier role allocation.", "",
        "## Fit and statistical target", "",
        "The model remains the 614–32–32–32–1 MLP with output $2\\operatorname{sigmoid}(2a)$. "
        "The exact empirical objective is", "",
        "$$L=\\tfrac12\\overline{r_P^{-1/2}}+\\tfrac14\\overline{\\sqrt{r_P}}"
        "+\\tfrac14\\overline{\\sqrt{r_Q}}-1.$$", "",
        "Separate P and Q means preserve the equally weighted midpoint despite unequal sample sizes. "
        "The population derivative is proportional to $-2p+(p+q)r$, so the unrestricted minimizer is "
        "$r_0=2p/(p+q)$. This identifies the intended target, not a guarantee that finite neural training estimates it accurately.", "",
        f"Selected epoch: **{training['best_epoch']}** of **{training['epochs_run']}**; restored validation loss "
        f"**{training['best_validation_loss']:.6f}**. AdamW, learning rate 0.0005, weight decay 0.01, gradient "
        "clipping at 1, and the plateau scheduler match the preceding fit. The absolute minimum-validation "
        "checkpoint is restored; patience begins after 300 epochs in the default run.", "",
        "For $A_j=\\{x:\\hat r(x)\\in I_j\\}$, unchanged C.1/C.2 estimate the population cell average", "",
        "$$\\theta_j=\\frac{2P(A_j)}{P(A_j)+Q(A_j)}=\\mathbb E_{(P+Q)/2}[r_0(X)\\mid X\\in A_j].$$", "",
        "C.1 is an asymptotic marginal interval. C.2 allocates alpha=0.05 across all 20 prespecified cells "
        "using Clopper–Pearson bounds for nominal 95% simultaneous coverage under independent P/Q sampling. "
        "The same test observations calibrate and receive the intervals; this is not independent evaluation of "
        "calibration performance. Neither interval estimates individual-RDR uncertainty or training uncertainty.", "",
        "This is a retrospective rerun after inspecting earlier results. The test cohort is unchanged but is "
        "not newly unexamined confirmation. Repeated stool samples within subjects, historical whole-table "
        "preprocessing, and archived generator provenance retain the earlier sampling limitations. The unchanged "
        "binomial intervals do not adjust for subject clustering or prior protocol selection.", "",
        "## Test/calibration intervals", "", "![AGP test CIs](agp_icfm_ci.png)", "",
        "| Method | Test group | Mean width | Fraction excluding 1 | Cells excluding 1 |",
        "|---|---|---:|---:|---:|"])
    for row in summary.itertuples():
        text.append(f"| {row.method} | {row.distribution} | {row.mean_width:.6f} | "
                    f"{row.fraction_excluding_one:.6f} | {row.n_cells_excluding_one}/20 |")
    text.extend(["", "## Calibration pattern across roles", "", "![Descriptive role comparison](split_calibration.png)", "",
        "These curves use class-normalized P/Q frequencies. Their horizontal coordinates are the within-cell "
        "model-score means under equal P/Q mixture weights. Training and validation curves are descriptive: "
        "those observations influenced fitting and selection. The empirical absolute gap is a bin-weighted "
        "diagnostic, not an unbiased population calibration error or a measured coverage rate.", "",
        "| Role | Mean model score under equal mixture | Empirical bin absolute gap | Cells above mean model score |",
        "|---|---:|---:|---:|"])
    for row in diagnostic_summary.itertuples():
        text.append(f"| {row.role} | {row.equal_mixture_mean_score:.6f} | {row.empirical_bin_absolute_gap:.6f} | "
                    f"{row.cells_above_mean_model_score}/{row.nonempty_cells} |")
    text.extend(["", "## Comparison with the previous fitted model", "",
        "The test observations are identical. The fitted score map changes, so equal numeric score bins can "
        "contain different observations and have different population targets. Changes in interval width are "
        "descriptive and cannot be attributed solely to calibration sample size, which is unchanged. This "
        "single rerun does not establish a general improvement or distinguish distribution shift from estimation error.", "",
        "| Fit | Method | Mean test score | Empirical bin absolute gap | Mean CI width |",
        "|---|---|---:|---:|---:|"])
    for row in comparison.itertuples():
        text.append(f"| {row.run} | {row.method} | {row.equal_mixture_mean_score:.6f} | "
                    f"{row.empirical_bin_absolute_gap:.6f} | {row.mean_interval_width:.6f} |")
    text.extend(["", "## All test cells", "", "| Score cell | P count | Q count | Cell estimate | C.1 | C.2 |",
                 "|---|---:|---:|---:|---|---|"])
    for row in cells.itertuples():
        estimate = "undefined (empty)" if row.c1_empty else f"{row.estimate:.5f}"
        c1 = "unavailable; [0,2]" if row.c1_unavailable else f"[{row.c1_lower:.5f}, {row.c1_upper:.5f}]"
        text.append(f"| [{row.left:.1f}, {row.right:.1f}{']' if row.right_closed else ')'} | {row.k_p} | "
                    f"{row.k_q} | {estimate} | {c1} | [{row.c2_lower:.5f}, {row.c2_upper:.5f}] |")
    command = ["python3", "-B", "experiments/AGP/AGP_ICFM_712_ci.py", "--p-policy", protocol["p_policy"],
        "--data-root", protocol["data_root"], "--baseline", protocol["baseline"], "--seed", str(protocol["seed"]),
        "--max-epochs", str(protocol["max_epochs"]), "--min-epochs", str(protocol["min_epochs"]),
        "--patience", str(protocol["patience"]), "--threads", str(protocol["threads"]),
        "--output-dir", str(DEFAULT_OUTPUT) + "_rerun"]
    text.extend(["", "## Reproduction", "", "```bash", shlex.join(command), "```", "",
        "Use a new output directory. Add `--replay-from PATH_TO_THIS_OUTPUT` to verify and reuse the saved "
        "checkpoint instead of training again. The source snapshots support the same command from their "
        "`source/` directory. `protocol.json` records data, split, baseline, code, and runtime provenance. "
        "`result.json` is written last with all artifact hashes.", "", "![Training history](loss_history.png)", ""])
    (out / "report.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--p-policy", choices=("keep-all", "match-ratio"), default="keep-all")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--max-epochs", type=int, default=2000)
    parser.add_argument("--min-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--replay-from", type=Path)
    args = parser.parse_args()
    require(0 < args.min_epochs <= args.max_epochs and args.patience > 0 and args.threads > 0, "Invalid fit controls")
    out, baseline = args.output_dir.resolve(), args.baseline.resolve()
    require(out.is_relative_to(Path("/tmp")) or out.is_relative_to(Path("/cwork")), "Use /tmp or /cwork for outputs")
    require(not out.exists(), "Output directory already exists")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    verify_result(baseline)
    arrays, manifest, audit = prepare_712_data(args.data_root.resolve(), args.seed, args.p_policy)
    edges = np.linspace(0, 2, 21)
    files = ["experiments/AGP/AGP_ICFM_712_ci.py", "experiments/AGP/AGP_712_data.py",
        "experiments/AGP/AGP_ICFM_ci.py", "experiments/AGP/AGP_ci_data.py", "experiments/AGP/AGP_ICFM_test_ci.py",
        "utils/calibration.py", "utils/__init__.py", "utils/networks.py", "utils/losses.py",
        "experiments/AGP/AGP1_data_preprocessing.py", "experiments/AGP/AGP2_ICFM.py"]
    protocol = {"analysis": "agp_real_vs_icfm_712_full_test_calibration", "data_root": str(args.data_root.resolve()),
        "p_policy": args.p_policy, "seed": args.seed, "max_epochs": args.max_epochs, "min_epochs": args.min_epochs,
        "patience": args.patience, "threads": args.threads, "alpha": .05, "edges": edges.tolist(),
        "baseline": str(baseline), "baseline_result_sha256": sha256(baseline / "result.json"),
        "data_audit": audit, "calibration_equals_test": True, "test_ids_unchanged": True,
        "generator_unchanged": True, "model_architecture": "MLP 614-32-32-32-1; 2*sigmoid(2*logit)",
        "loss": ".5*mean_P(r^-1/2)+.25*mean_P(sqrt(r))+.25*mean_Q(sqrt(r))-1",
        "optimizer": "AdamW(lr=5e-4, weight_decay=.01), clip_grad_norm=1; original plateau scheduler",
        "retrospective": True, "true_coverage_measured": False, "within_subject_correlation_adjusted": False,
        "code_sha256": {name: sha256(REPO / name) for name in files},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "torch": torch.__version__, "scipy": scipy.__version__}}
    out.mkdir(parents=True)
    save_json(out / "protocol.json", protocol)
    manifest.to_csv(out / "split_manifest.csv", index=False)
    for name in files:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    if args.replay_from:
        previous = args.replay_from.resolve()
        verify_result(previous)
        require(json.loads((previous / "protocol.json").read_text()) == protocol, "Replay protocol differs")
        require((previous / "split_manifest.csv").read_bytes() == (out / "split_manifest.csv").read_bytes(), "Replay split differs")
        checkpoint = torch.load(previous / "model.pt", map_location="cpu", weights_only=False)
        require(checkpoint["protocol_sha256"] == sha256(previous / "protocol.json"), "Checkpoint protocol mismatch")
        model = MLP(input_dim=arrays["P", "train"].shape[1], hidden_dim=32, output_alpha=2)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        training = checkpoint["training"]
        history = pd.read_csv(previous / "loss_history.csv")
        with torch.no_grad():
            restored = float(midpoint_loss(model, torch.as_tensor(arrays["P", "validation"]),
                                         torch.as_tensor(arrays["Q", "validation"])))
        require(abs(restored-training["best_validation_loss"]) < 1e-7, "Replay validation loss differs")
        shutil.copyfile(previous / "model.pt", out / "model.pt")
    else:
        model, history, training = fit(arrays, args)
        torch.save({"state_dict": model.state_dict(), "training": training,
                    "protocol_sha256": sha256(out / "protocol.json")}, out / "model.pt")
    history.to_csv(out / "loss_history.csv", index=False)
    save_json(out / "training.json", training)
    scores = score_rows(model, arrays, manifest, edges)
    test = scores[scores.role.eq("test")]
    cells, attached, summary = calibrate(pd.concat([test.assign(role="calibration"), test.assign(role="query")]), edges, .05)
    attached["role"] = "test"
    summary = summary.rename(columns={"n_query": "n_test", "c1_unavailable_query_fraction": "c1_unavailable_test_fraction"})
    diagnostic_cells, diagnostic_summary = diagnostics(scores, edges)
    comparison = compare_baseline(scores, summary, baseline, edges)
    for name, table in (("scores", scores), ("cell_intervals", cells), ("test_intervals", attached),
                        ("summary", summary), ("split_calibration_cells", diagnostic_cells),
                        ("split_calibration_summary", diagnostic_summary), ("baseline_comparison", comparison)):
        table.to_csv(out / f"{name}.csv", index=False)
    plot_ci(out, cells, .05)
    plot_diagnostics(out, history, diagnostic_cells)
    report(out, protocol, training, manifest, cells, summary, diagnostic_summary, comparison)
    artifacts = [p for p in out.rglob("*") if p.is_file() and ".matplotlib" not in p.parts]
    save_json(out / "result.json", {"status": "complete", "analysis": protocol["analysis"],
        "training": training, "test_rows": len(test), "cells": len(cells), "elapsed_seconds": time.monotonic()-started,
        "replay_from": str(args.replay_from.resolve()) if args.replay_from else None,
        "outputs": {str(p.relative_to(out)): sha256(p) for p in sorted(artifacts)}})
    print(json.dumps({"status": "complete", "output": str(out), "training": training,
                      "diagnostics": diagnostic_summary.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
