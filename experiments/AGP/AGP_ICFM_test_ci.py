#!/usr/bin/env python3
"""Use the full held-out AGP test pool for calibration of a verified frozen fit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import shutil

import numpy as np
import pandas as pd
import scipy

from AGP_ICFM_ci import REPO, bin_scores, calibrate, require, save_json, sha256

DEFAULT_FIT = Path("/cwork/yx306/RDR/JRSSB/CI/agp_icfm_20260918")
DEFAULT_OUTPUT = Path("/cwork/yx306/RDR/JRSSB/CI/agp_icfm_full_test_20260918")


def full_test(source):
    """Verify the saved fit and unite only its two held-out roles."""
    result = json.loads((source / "result.json").read_text())
    require(result["status"] == "complete", "Fit source is incomplete")
    for name, digest in result["outputs"].items():
        require(sha256(source / name) == digest, f"Source artifact hash mismatch: {name}")
    prior = json.loads((source / "protocol.json").read_text())
    require(prior["analysis"] == "agp_real_vs_icfm_strict_split_calibration", "Unsupported fit protocol")
    # Reuse the exact original calibration implementation; no training is run here.
    dependencies = ("experiments/AGP/AGP_ICFM_ci.py", "experiments/AGP/AGP_ci_data.py",
                    "utils/calibration.py", "utils/__init__.py", "utils/networks.py", "utils/losses.py")
    for name in dependencies:
        require(sha256(REPO / name) == prior["code_sha256"].get(name),
                f"Fit dependency changed: {name}; replay this historical fit with its archived source or rollback checkout")
    dtypes = {"sample_id": str, "subject_id": str, "row_sha256": str}
    scores = pd.read_csv(source / "scores.csv", dtype=dtypes, keep_default_na=False)
    manifest = pd.read_csv(source / "split_manifest.csv", dtype=dtypes, keep_default_na=False)
    keys = ["distribution", "source_file", "source_row"]
    require(not scores.duplicated(keys).any() and not manifest.duplicated(keys).any(), "Duplicate observations")
    eligible = manifest[manifest.role.isin(["train", "validation", "calibration", "query"])]
    matched = scores.merge(eligible, on=keys, suffixes=("_score", "_manifest"), validate="one_to_one")
    require(len(matched) == len(scores) == len(eligible), "Score/manifest observation mismatch")
    for name in ("sample_id", "subject_id", "row_sha256", "role"):
        require((matched[name + "_score"] == matched[name + "_manifest"]).all(), f"Manifest mismatch: {name}")
    edges = np.asarray(prior["edges"])
    require(np.array_equal(scores.bin_index, bin_scores(scores.rdr.to_numpy(), edges)), "Saved score bins differ")
    for table in (scores, manifest):
        table["original_role"] = table.role
        table.loc[table.role.isin(["calibration", "query"]), "role"] = "test"
    test = scores[scores.role.eq("test")].copy()
    require(set(test.distribution) == {"P", "Q"}, "Both test distributions required")
    for distribution in ("P", "Q"):
        held = test[test.distribution.eq(distribution)]
        fitting = scores[scores.distribution.eq(distribution) & scores.role.isin(["train", "validation"])]
        require(not set(held.sample_id) & set(fitting.sample_id), "Test sample in training/validation")
        if distribution == "P":
            require(not set(held.subject_id) & set(fitting.subject_id), "Test subject in training/validation")
    return result, prior, scores, manifest, test, dependencies


def width_comparison(old_cells, cells, test):
    weights = np.where(test.distribution.eq("P"), .5 / sum(test.distribution.eq("P")),
                       .5 / sum(test.distribution.eq("Q")))
    rows = []
    for method in ("c1", "c2"):
        widths = []
        for table in (old_cells, cells):
            table = table.set_index("bin_index").loc[test.bin_index]
            widths.append(float(np.sum(weights * (table[f"{method}_upper"].to_numpy() -
                                                  table[f"{method}_lower"].to_numpy()))))
        rows.append({"method": method, "weighting": "same_full_test_equal_mixture",
                     "previous_mean_width": widths[0], "full_test_mean_width": widths[1]})
    return pd.DataFrame(rows)


def plot(out, cells, alpha):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    edges = np.r_[cells.left, cells.right.iloc[-1]]
    centers = (cells.left.to_numpy() + cells.right.to_numpy()) / 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, method in zip(axes[:2], ("c2", "c1")):
        low = cells[f"{method}_lower"].to_numpy().copy()
        high = cells[f"{method}_upper"].to_numpy().copy()
        if method == "c1":
            unavailable = cells.c1_unavailable.to_numpy()
            low[unavailable] = np.nan
            high[unavailable] = np.nan
            if unavailable.any():
                ax.scatter(centers[unavailable], np.ones(sum(unavailable)), marker="x", color=".5",
                           label="C.1 unavailable ([0,2] in CSV)")
        ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                        color="#88b9d9", alpha=.65, label="Interval")
        estimate = cells.estimate.to_numpy().copy()
        estimate[cells.c1_empty.to_numpy()] = np.nan
        ax.stairs(estimate, edges, color="#182f44", label="Cell estimate")
        ax.plot([0, 2], [0, 2], ":", color=".5", lw=.8)
        ax.axhline(1, ls="--", color=".5", lw=.8)
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel="Frozen AGP RDR score", ylabel="Cell-average RDR",
               title="C.2 simultaneous" if method == "c2" else "C.1 marginal, available cells")
        ax.legend(fontsize=7)
    width = np.diff(edges) * .4
    axes[2].bar(centers - width/2, cells.k_p, width=width, label="P: observed AGP")
    axes[2].bar(centers + width/2, cells.k_q, width=width, label="Q: ICFM")
    axes[2].set(xlim=(0, 2), yscale="log", xlabel="Frozen AGP RDR score", ylabel="Test/calibration count (log)",
                title=f"Full test: {int(cells.n_p_calibration.iloc[0]):,} P / {int(cells.n_q_calibration.iloc[0]):,} Q")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"AGP real vs ICFM: nominal {100*(1-alpha):g}% cell intervals")
    fig.tight_layout()
    fig.savefig(out / "agp_icfm_ci.png", dpi=180)
    fig.savefig(out / "agp_icfm_ci.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def report(out, protocol, manifest, cells, summary, comparison):
    text = ["# AGP real versus ICFM: full-test calibration", "",
        "The calibration set is now the entire eligible test set. The earlier calibration and query subsets "
        "are united without filtering on scores. The verified fitted model, validation-selected epoch, "
        "and 20 equal-width score bins are unchanged. No model is retrained.", "",
        "| Role | P samples | P subjects | Q samples |", "|---|---:|---:|---:|"]
    for role in ("train", "validation", "test"):
        p = manifest[manifest.distribution.eq("P") & manifest.role.eq(role)]
        q = manifest[manifest.distribution.eq("Q") & manifest.role.eq(role)]
        text.append(f"| {role} | {len(p):,} | {p.subject_id.nunique():,} | {len(q):,} |")
    text.extend(["", "Test and calibration refer to exactly the same observations. The 609 real observations "
        "from generator-training subjects remain excluded. Q uses the unused portion of the single archived "
        "ICFM bank; the historical generated test bank remains excluded because its random starts were shared "
        "with that bank. Real test subjects remain disjoint from training and validation subjects.", "",
        "## Target and interpretation", "",
        "For the frozen score cells $A_j=\\{x:\\hat r(x)\\in I_j\\}$, the target is", "",
        "$$\\theta_j=\\frac{2P(A_j)}{P(A_j)+Q(A_j)}=\\mathbb E_{(P+Q)/2}[r_0(X)\\mid X\\in A_j].$$", "",
        "C.1 and C.2 use the original algorithms without pseudocounts. C.2 allocates the error budget across "
        "all 20 cells. Under independent P/Q sampling and a score map/partition fixed independently of the "
        "calibration observations, simultaneous coverage of every cell also covers the cell selected by any "
        "of those observations. Thus no extra query holdout is needed to attach C.2 intervals to the test set. "
        "This does not turn the intervals into individual-RDR intervals or an independent assessment of "
        "calibration performance. C.1 remains asymptotic and marginal for a fixed cell, without C.2's "
        "simultaneous guarantee for data-selected cells.", "",
        "Repeated samples within a subject, historical whole-table preprocessing, and incomplete generator "
        "provenance retain the original report's sampling limitations. This is retrospective nominal "
        "inference conditional on the fitted score map, without a measured coverage rate or uncertainty "
        "for model training or Hellinger divergence. P describes the retained held-out AGP cohort.", "",
        "## Results", "", "![Full-test AGP intervals](agp_icfm_ci.png)", "",
        "C.1 unavailable cells are marked separately in the plot; their prescribed [0,2] CSV bounds are retained. "
        "Empty cells have no displayed point estimate. Count axes use a log scale.", "",
        "| Method | Test group | Mean width | Fraction excluding 1 | Cells excluding 1 |",
        "|---|---|---:|---:|---:|"])
    for row in summary.itertuples():
        text.append(f"| {row.method} | {row.distribution} | {row.mean_width:.6f} | "
                    f"{row.fraction_excluding_one:.6f} | {row.n_cells_excluding_one}/20 |")
    text.extend(["", "Equal-mixture summaries give total weight 1/2 to each distribution. The following "
        "comparison evaluates both sets of intervals using the same full test weights; it is descriptive, "
        "not a coverage estimate. Larger samples need not narrow every realized cell interval.", "",
        "| Method | Previous calibration mean width | Full-test calibration mean width |",
        "|---|---:|---:|"])
    for row in comparison.itertuples():
        text.append(f"| {row.method} | {row.previous_mean_width:.6f} | {row.full_test_mean_width:.6f} |")
    text.extend(["", "| Score cell | P count | Q count | Cell estimate | C.1 | C.2 |",
                 "|---|---:|---:|---:|---|---|"])
    for row in cells.itertuples():
        estimate = "undefined (empty)" if row.c1_empty else f"{row.estimate:.5f}"
        c1 = "unavailable; [0,2]" if row.c1_unavailable else f"[{row.c1_lower:.5f}, {row.c1_upper:.5f}]"
        text.append(f"| [{row.left:.1f}, {row.right:.1f}{']' if row.right_closed else ')'} | "
                    f"{row.k_p} | {row.k_q} | {estimate} | {c1} | [{row.c2_lower:.5f}, {row.c2_upper:.5f}] |")
    command = ["python3", "-B", "experiments/AGP/AGP_ICFM_test_ci.py", "--fit-from", protocol["fit_from"],
               "--output-dir", str(DEFAULT_OUTPUT) + "_rerun"]
    text.extend(["", "## Reproduction and provenance", "", "```bash", shlex.join(command), "```", "",
        "All previous artifact hashes are verified before reuse. `fit_protocol.json` and `fit_result.json` "
        "preserve the original fit metadata; `model.pt` and its history are byte-for-byte copies. "
        "`split_manifest.csv` and `scores.csv` retain `original_role`; `test_intervals.csv` attaches intervals "
        "to every calibration/test row. `protocol.json` records source hashes and the new roles. "
        "Source snapshots are under `source/`, and `result.json` is written last with output hashes.", ""])
    (out / "report.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-from", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source, out = args.fit_from.resolve(), args.output_dir.resolve()
    require(out.is_relative_to(Path("/tmp")) or out.is_relative_to(Path("/cwork")), "Use /tmp or /cwork for outputs")
    require(not out.exists(), "Output directory already exists")
    result, prior, scores, manifest, test, dependencies = full_test(source)
    edges, alpha = np.asarray(prior["edges"]), prior["alpha"]
    # The existing helper takes separate tables. Both views here contain the same
    # test rows; each observation is counted exactly once when fitting each cell.
    views = pd.concat([test.assign(role="calibration"), test.assign(role="query")], ignore_index=True)
    cells, attached, summary = calibrate(views, edges, alpha)
    attached["role"] = "test"
    summary = summary.rename(columns={"n_query": "n_test", "c1_unavailable_query_fraction": "c1_unavailable_test_fraction"})
    comparison = width_comparison(pd.read_csv(source / "cell_intervals.csv"), cells, test)
    source_files = ("experiments/AGP/AGP_ICFM_test_ci.py", *dependencies)
    protocol = {"analysis": "agp_real_vs_icfm_full_test_calibration", "fit_from": str(source),
        "fit_result_sha256": sha256(source / "result.json"), "fit_protocol_sha256": sha256(source / "protocol.json"),
        "frozen_checkpoint_sha256": sha256(source / "model.pt"), "alpha": alpha, "edges": edges.tolist(),
        "calibration_equals_test": True, "test_pool": "union of previous calibration and query, no score filtering",
        "n_p_test": int(sum(test.distribution.eq("P"))), "n_q_test": int(sum(test.distribution.eq("Q"))),
        "training_unchanged": True, "validation_unchanged": True, "partition_unchanged": True,
        "independent_query_evaluation": False, "true_coverage_measured": False,
        "within_subject_correlation_adjusted": False, "retrospective": True,
        "code_sha256": {name: sha256(REPO / name) for name in source_files},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                    "scipy": scipy.__version__}}
    out.mkdir(parents=True)
    save_json(out / "protocol.json", protocol)
    for filename, target in (("protocol.json", "fit_protocol.json"), ("result.json", "fit_result.json"),
                             ("model.pt", "model.pt"), ("loss_history.csv", "loss_history.csv"),
                             ("training.json", "training.json")):
        shutil.copyfile(source / filename, out / target)
    for name in source_files:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    manifest.to_csv(out / "split_manifest.csv", index=False)
    scores.to_csv(out / "scores.csv", index=False)
    cells.to_csv(out / "cell_intervals.csv", index=False)
    attached.to_csv(out / "test_intervals.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    comparison.to_csv(out / "width_comparison.csv", index=False)
    plot(out, cells, alpha)
    report(out, protocol, manifest, cells, summary, comparison)
    files = [p for p in out.rglob("*") if p.is_file() and ".matplotlib" not in p.parts]
    save_json(out / "result.json", {"status": "complete", "analysis": protocol["analysis"],
        "training": result["training"], "test_rows": len(test), "cells": len(cells),
        "outputs": {str(p.relative_to(out)): sha256(p) for p in sorted(files)}})
    print(json.dumps({"status": "complete", "output": str(out), "n_p_test": protocol["n_p_test"],
                      "n_q_test": protocol["n_q_test"], "comparison": comparison.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
