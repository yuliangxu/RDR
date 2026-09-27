#!/usr/bin/env python3
"""Recalibrate the frozen AGP 7:1:2-policy fit with shared adjacent-cell merging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys

import numpy as np
import pandas as pd
import scipy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils.calibration import bin_scores, calibrate_merged_counts

DEFAULT_FIT = Path("/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_20260920")
ROLES = ("train", "validation", "test")


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


def verified_fit(source):
    """Reuse sealed predictions; changing calibration never requires a new fit."""
    result = json.loads((source / "result.json").read_text())
    require(result["status"] == "complete", "Source fit is incomplete")
    for name, digest in result["outputs"].items():
        require(sha256(source / name) == digest, f"Source artifact hash mismatch: {name}")
    prior = json.loads((source / "protocol.json").read_text())
    require(prior["analysis"] == "agp_real_vs_icfm_712_full_test_calibration", "Unsupported fit protocol")
    dtypes = {"sample_id": str, "subject_id": str, "row_sha256": str}
    scores = pd.read_csv(source / "scores.csv", dtype=dtypes, keep_default_na=False,
                         float_precision="round_trip")
    manifest = pd.read_csv(source / "split_manifest.csv", dtype=dtypes, keep_default_na=False)
    keys = ["distribution", "source_file", "source_row"]
    eligible = manifest[manifest.role.isin(ROLES)]
    require(not scores.duplicated(keys).any() and not manifest.duplicated(keys).any(), "Duplicate source rows")
    matched = scores.merge(eligible, on=keys, suffixes=("_score", "_manifest"), validate="one_to_one")
    require(len(matched) == len(scores) == len(eligible), "Score/manifest row mismatch")
    for name in ("sample_id", "subject_id", "row_sha256", "role"):
        require((matched[name + "_score"] == matched[name + "_manifest"]).all(), f"Manifest mismatch: {name}")
    edges = np.asarray(prior["edges"], dtype=float)
    require(np.array_equal(scores.bin_index, bin_scores(scores.rdr.to_numpy(), edges)), "Source score/bin mismatch")
    require(set(scores.role) == set(ROLES) and set(scores.distribution) == {"P", "Q"}, "Unexpected score roles")
    for distribution in ("P", "Q"):
        subset = scores[scores.distribution.eq(distribution)]
        test = subset[subset.role.eq("test")]
        fitting = subset[~subset.role.eq("test")]
        require(not set(test.sample_id) & set(fitting.sample_id), "Test sample reused for fitting")
        if distribution == "P":
            require(not set(test.subject_id) & set(fitting.subject_id), "Test subject reused for fitting")
        for role in ROLES:
            require(sum(subset.role.eq(role)) == prior["data_audit"]["counts"][f"{distribution}_{role}"]["rows"],
                    "Source protocol sample count mismatch")
    return result, prior, scores, manifest, edges


def split_diagnostics(scores, cells, lookup):
    """Use the same test-selected regions for all three descriptive curves."""
    rows, summaries = [], []
    for role in ROLES:
        subset = scores[scores.role.eq(role)]
        p, q = (subset[subset.distribution.eq(d)] for d in ("P", "Q"))
        pbin, qbin = lookup[p.bin_index.to_numpy()], lookup[q.bin_index.to_numpy()]
        kp, kq = (np.bincount(b, minlength=len(cells)) for b in (pbin, qbin))
        ph, qh = kp / len(p), kq / len(q)
        total = ph + qh
        sums = (np.bincount(pbin, weights=p.rdr, minlength=len(cells)) / len(p)
                + np.bincount(qbin, weights=q.rdr, minlength=len(cells)) / len(q))
        mean = np.divide(sums, total, out=np.full_like(total, np.nan), where=total > 0)
        ratio = np.divide(2 * ph, total, out=np.full_like(total, np.nan), where=total > 0)
        for j, cell in enumerate(cells.itertuples()):
            rows.append({"role": role, "bin_index": j, "candidate_index": cell.candidate_index,
                         "left": cell.left, "right": cell.right, "n_p": len(p), "n_q": len(q),
                         "k_p": kp[j], "k_q": kq[j], "mixture_mass": total[j] / 2,
                         "mean_model_score": mean[j], "cell_rdr": ratio[j],
                         "calibration_gap": ratio[j] - mean[j], "diagnostic_only": True})
        summaries.append({"role": role, "n_p": len(p), "n_q": len(q),
                          "equal_mixture_mean_score": .5 * (p.rdr.mean() + q.rdr.mean()),
                          "empirical_bin_absolute_gap": np.nansum(total / 2 * np.abs(ratio - mean)),
                          "cells_above_mean_model_score": int(np.sum(ratio > mean)), "n_cells": len(cells),
                          "partition_selected_from": "test_calibration"})
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def interval_summary(attached, cells):
    rows = []
    for method in ("c1", "c2"):
        excluded = (cells[f"{method}_lower"] > 1) | (cells[f"{method}_upper"] < 1)
        for distribution in ("P", "Q", "equal_mixture"):
            selected = attached if distribution == "equal_mixture" else attached[attached.distribution.eq(distribution)]
            weights = (np.where(selected.distribution.eq("P"), .5 / sum(selected.distribution.eq("P")),
                                .5 / sum(selected.distribution.eq("Q")))
                       if distribution == "equal_mixture" else np.full(len(selected), 1 / len(selected)))
            lo, hi = selected[f"{method}_lower"], selected[f"{method}_upper"]
            rows.append({"method": method, "distribution": distribution, "n_test": len(selected),
                         "mean_width": np.sum(weights * (hi - lo)),
                         "fraction_excluding_one": np.sum(weights * ((lo > 1) | (hi < 1))),
                         "n_cells": len(cells), "n_cells_excluding_one": int(excluded.sum()),
                         "n_full_range_cells": int(((cells[f"{method}_lower"] == 0)
                                                    & (cells[f"{method}_upper"] == 2)).sum()),
                         "c1_unavailable_test_fraction": np.sum(weights * selected.c1_unavailable)})
    return pd.DataFrame(rows)


def make_plots(out, cells, diagnostics, metadata):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges = np.r_[cells.left, cells.right.iloc[-1]]
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, method, title in zip(axes[:2], ("c1", "c2"), ("C.1 joint multiplier bootstrap", "C.2 exact candidate correction")):
        low, high = (cells[f"{method}_{bound}"].to_numpy() for bound in ("lower", "upper"))
        ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post", alpha=.4,
                        color="#4189bb", label="Simultaneous interval")
        estimate = cells.estimate.to_numpy().copy()
        estimate[cells.c1_empty] = np.nan
        ax.stairs(estimate, edges, color="#182f44", label="Merged cell estimate")
        ax.plot([0, 2], [0, 2], ":", color=".5", lw=.9)
        ax.axhline(1, ls="--", color=".5", lw=.8)
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel="Frozen AGP RDR score", ylabel="Merged-cell average RDR", title=title)
        ax.legend(fontsize=8)
    width = np.diff(edges) * .36
    axes[2].bar(centers - width / 2, cells.k_p, width=width, label="P: observed AGP")
    axes[2].bar(centers + width / 2, cells.k_q, width=width, label="Q: ICFM")
    axes[2].axhline(metadata["min_count"], color=".4", ls="--", label=f"h = {metadata['min_count']} per distribution")
    axes[2].set(xlim=(0, 2), yscale="log", xlabel="Frozen AGP RDR score", ylabel="Test/calibration count (log)",
                title=f"{len(cells)} selected regions; {metadata['n_candidates']} candidates")
    axes[2].legend(fontsize=8)
    fig.suptitle(f"AGP real vs ICFM: nominal {100 * (1 - metadata['alpha']):g}% merged-cell intervals")
    fig.tight_layout()
    fig.savefig(out / "agp_icfm_merged_ci.png", dpi=180)
    fig.savefig(out / "agp_icfm_merged_ci.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, role, title in zip(axes, ROLES, ("Train (used for fitting)", "Validation (used for selection)", "Test = calibration")):
        rows = diagnostics[diagnostics.role.eq(role)]
        ax.plot(rows.mean_model_score, rows.cell_rdr, "o-", ms=4)
        ax.plot([0, 2], [0, 2], ":", color=".5")
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel="Within-region mean model score", title=title)
    axes[0].set_ylabel("Empirical cell RDR from P/Q frequencies")
    fig.suptitle("Descriptive comparison: common merged regions selected from test/calibration counts")
    fig.tight_layout()
    fig.savefig(out / "split_calibration.png", dpi=180)
    fig.savefig(out / "split_calibration.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def ci_panel_axis_description(n_p, n_q):
    """Explain the coordinates of the two merged-CI staircase panels."""
    return "\n".join([
        "### Axes in the first two panels", "",
        "Both panels use the same horizontal coordinate and the same central step curve. "
        "Their shaded confidence intervals are constructed differently.", "",
        r"**Horizontal axis:** for a microbiome profile $z$, the frozen network score is", "",
        "$$", r"t=\hat r_{\mathrm{tr}}(z)\in[0,2].", "$$", "",
        r"Let $I_j=[a_j,b_j)$ be a selected merged score interval (including $2$ at the final right endpoint), "
        r"and define its input-space region by $A_j=\{z:\hat r_{\mathrm{tr}}(z)\in I_j\}$. "
        "The horizontal step spans the entire score interval. Its position is not a within-region mean score; "
        "that different horizontal summary is used in the later three-panel diagnostic.", "",
        r"**Vertical axis:** the dark line is the calibrated estimate $\hat\theta_j$ throughout $I_j$:", "",
        "$$",
        r"y(t)=\hat\theta_j"
        r"=\frac{2\hat p_j}{\hat p_j+\hat q_j}"
        r"=\frac{2k_P(A_j)/n}{k_P(A_j)/n+k_Q(A_j)/m},\qquad t\in I_j.",
        "$$", "",
        r"For a profile $z$, its calibrated estimate is $\hat r_{\mathrm{cali}}(z)=y(\hat r_{\mathrm{tr}}(z))$. "
        f"Here $n={n_p}$ and $m={n_q}$ are the total P and Q calibration sample sizes; "
        r"$k_P(A_j)$ and $k_Q(A_j)$ count samples whose network scores fall in $I_j$. "
        r"The separately normalized proportions are $\hat p_j=k_P(A_j)/n$ and $\hat q_j=k_Q(A_j)/m$.", "",
        "The shaded interval in each region concerns the population cell-average RDR", "",
        "$$",
        r"\theta_j=\frac{2P(A_j)}{P(A_j)+Q(A_j)}"
        r"=\mathbb E_{Z\sim M}[r_0(Z)\mid Z\in A_j],"
        r"\qquad M=(P+Q)/2,\quad r_0=\frac{2p}{p+q}.",
        "$$", "",
        "This expectation treats the realized network and region as fixed and averages over a fresh population "
        "observation. Panel 1 uses the C.1 joint multiplier-bootstrap interval; panel 2 uses the C.2 exact "
        "simultaneous interval under the stated IID assumptions. These intervals concern the cell average, "
        "not every individual RDR within the region.", "",
        r"The dotted diagonal is $y=t$, and the dashed horizontal reference is $y=1$.",
    ])


def write_report(out, protocol, manifest, cells, summary, comparison):
    meta = protocol["calibration"]
    text = ["# AGP real versus ICFM: adjacent-cell merging", "",
            f"The frozen epoch-{protocol['source_training']['best_epoch']} fit and the same "
            f"{protocol['n_p']:,} P / {protocol['n_q']:,} Q test/calibration observations are reused. "
            f"The user-selected threshold is **h = {meta['min_count']} from each distribution**. "
            f"The {meta['n_elementary_bins']} elementary bins produce **{len(cells)} merged regions**; "
            f"both methods protect all **{meta['n_candidates']} contiguous candidate regions**.", "",
            "| Role | P samples | P subjects | Q samples |", "|---|---:|---:|---:|"]
    for role in ROLES:
        p, q = (manifest[manifest.role.eq(role) & manifest.distribution.eq(d)] for d in ("P", "Q"))
        text.append(f"| {role} | {len(p):,} | {p.subject_id.nunique():,} | {len(q):,} |")
    text.extend(["", "## Method and target", "",
        "Scan left to right, close a region when both P and Q counts reach h, and merge any remaining tail "
        "into the preceding region. If none qualifies, use the whole range. The elementary boundaries and "
        "candidate family stay fixed; calibration counts select the reported regions.", "",
        r"For a realized selected region $A$, the target is $\theta_A=2P(A)/(P(A)+Q(A))=\mathbb E_{(P+Q)/2}[r_0(X)\mid X\in A]$.", "",
        f"C.1 uses {meta['bootstrap_repetitions']:,} joint Gaussian multiplier repetitions, seed {meta['seed']}, "
        f"with shared multipliers across every candidate. Its simultaneous critical value is **{meta['critical_value']:.6f}** "
        f"(NumPy quantile method `{meta['quantile_method']}`). Zero-SE/undefined candidates use [0,2] and are omitted "
        "from studentization; the structural whole-range candidate uses [1,1]. C.1 is an asymptotic simultaneous "
        "extension, subject to uniform approximation and nondegeneracy conditions; the count threshold alone "
        "does not establish them.", "",
        f"C.2 applies Clopper-Pearson probability intervals with each tail equal to alpha/(4J) = "
        f"{meta['c2_tail_probability']:.10g}. J remains {meta['n_candidates']} after selection. "
        "The simultaneous event covers every selected positive-mass candidate under independent IID P/Q sampling. "
        "The structural whole-range candidate has the exact interval [1,1].", "",
        "Merging changes the population cell target and reduces score resolution. These are intervals for cell "
        "averages, not individual RDRs; the procedure does not enforce monotonicity. The unchanged network scores "
        "can still differ from the calibrated estimates. Whole-range [1,1] would describe a known global average, "
        "not equality of P and Q.", "",
        "## Results", "", "![Merged calibration intervals](agp_icfm_merged_ci.png)", "",
        ci_panel_axis_description(protocol["n_p"], protocol["n_q"]), "",
        "| Score region | P | Q | Estimate | C.1 joint interval | C.2 exact interval |",
        "|---|---:|---:|---:|---|---|"])
    for row in cells.itertuples():
        interval = f"[{row.left:.1f}, {row.right:.1f}{']' if row.right_closed else ')'}"
        text.append(f"| {interval} | {row.k_p:,} | {row.k_q:,} | {row.estimate:.6f} | "
                    f"[{row.c1_lower:.6f}, {row.c1_upper:.6f}] | [{row.c2_lower:.6f}, {row.c2_upper:.6f}] |")
    text.extend(["", "| Method | Test group | Mean width | Fraction excluding 1 | Regions excluding 1 |",
                 "|---|---|---:|---:|---:|"])
    for row in summary.itertuples():
        text.append(f"| {row.method} | {row.distribution} | {row.mean_width:.6f} | {row.fraction_excluding_one:.6f} | "
                    f"{row.n_cells_excluding_one}/{row.n_cells} |")
    text.extend(["", "Equal-mixture summaries give total weight 1/2 to P and 1/2 to Q.", "",
        "| Method | Previous mean width | Merged mean width | Previous / merged cells |",
        "|---|---:|---:|---:|"])
    for row in comparison.itertuples():
        text.append(f"| {row.method} | {row.previous_mean_width:.6f} | {row.merged_mean_width:.6f} | "
                    f"{row.previous_n_cells} / {row.merged_n_cells} |")
    text.extend(["", "These width comparisons use identical test observations and weights, but different population "
        "cell targets. In addition, the previous C.1 was marginal whereas the new C.1 is simultaneous. "
        "A width change is therefore descriptive, not a comparison at fixed target and guarantee.", "",
        "![Calibration comparison on common merged regions](split_calibration.png)", "",
        "Each point averages model predictions on the horizontal axis and normalized P/Q counts on the vertical "
        "axis. All three panels use the same regions selected from test/calibration counts. The fitting-role "
        "panels and same-sample test curve are descriptive; they do not independently validate calibration.", "",
        "## Sampling limitations and reproducibility", "",
        "The existing subject separation and 609 excluded P rows are retained. Repeated samples within subjects "
        "are not corrected by the IID intervals. Historical whole-table preprocessing, recovered generator "
        "provenance, and prior examination of this test set remain limitations. These are retrospective nominal "
        "intervals; no true coverage rate is measured for AGP.", "",
        f"Verified source: [{protocol['source_fit']}]({protocol['source_fit']}/report.md). "
        "The model, input scores, split manifest, training history, and fit protocol are copied without modification. "
        "The checkpoint's embedded protocol hash refers to `fit_protocol.json`; `protocol.json` records this new calibration.", "",
        "[All candidate intervals](candidate_intervals.csv), [selected intervals](cell_intervals.csv), "
        "[elementary counts and lookup](elementary_bin_counts.csv), [test attachments](test_intervals.csv), "
        "[split diagnostics](split_calibration_cells.csv), [bootstrap maxima](bootstrap_maxima.npy), "
        "[protocol](protocol.json), and [completion hashes](result.json).", "",
        "Reproduce from the saved source with a fresh output directory:", "", "```bash",
        protocol["reproduce_command"], "```", ""])
    (out / "report.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fit", type=Path, default=DEFAULT_FIT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--min-count", type=int, default=20)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--preflight", action="store_true", help="Allow a disposable output directory under /tmp")
    args = parser.parse_args()
    source = args.source_fit.resolve()
    out = (args.output_dir or DEFAULT_FIT.with_name(f"agp_icfm_712_merged_h{args.min_count}_20260920")).resolve()
    allowed = Path("/tmp") if args.preflight else Path("/cwork/yx306/RDR")
    require(allowed in out.parents, f"Output must be below {allowed}")
    require(not out.exists(), "Output already exists; select a fresh directory")
    result, prior, scores, manifest, edges = verified_fit(source)
    test = scores[scores.role.eq("test")].copy()
    kp, kq = (np.bincount(test[test.distribution.eq(d)].bin_index, minlength=len(edges)-1) for d in ("P", "Q"))
    old_cells = pd.read_csv(source / "cell_intervals.csv", float_precision="round_trip")
    require(np.array_equal(kp, old_cells.k_p) and np.array_equal(kq, old_cells.k_q), "Source calibration counts differ")
    calibrated = calibrate_merged_counts(kp, kq, int(kp.sum()), int(kq.sum()), alpha=args.alpha,
        min_count=args.min_count, bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed)
    candidates = pd.DataFrame(calibrated["candidates"])
    candidates.insert(0, "candidate_index", np.arange(len(candidates)))
    candidates["left"], candidates["right"] = edges[candidates.start], edges[candidates.stop]
    candidates["right_closed"] = candidates.stop.eq(len(edges)-1)
    cells = candidates.iloc[calibrated["selected_indices"]].copy().reset_index(drop=True)
    cells.insert(0, "bin_index", np.arange(len(cells)))
    cells["n_p_calibration"], cells["n_q_calibration"] = kp.sum(), kq.sum()
    lookup = calibrated["elementary_to_merged"]
    test = test.rename(columns={"bin_index": "elementary_bin_index"})
    test["bin_index"] = lookup[test.elementary_bin_index.to_numpy()]
    attached = test.merge(cells, on="bin_index", how="left", validate="many_to_one")
    require(len(attached) == len(test) and attached.candidate_index.notna().all(), "Incomplete query attachment")
    require(np.array_equal(np.bincount(attached[attached.distribution.eq("P")].bin_index), cells.k_p), "Merged P counts differ")
    require(np.array_equal(np.bincount(attached[attached.distribution.eq("Q")].bin_index), cells.k_q), "Merged Q counts differ")
    require(np.array_equal(bin_scores(test.rdr.to_numpy(), np.r_[cells.left, cells.right.iloc[-1]]), test.bin_index),
            "Query-to-merged-boundary lookup differs")
    summary = interval_summary(attached, cells)
    diagnostics, diagnostic_summary = split_diagnostics(scores, cells, lookup)
    old_summary = pd.read_csv(source / "summary.csv")
    comparison = []
    for method in ("c1", "c2"):
        old = old_summary[old_summary.method.eq(method) & old_summary.distribution.eq("equal_mixture")].iloc[0]
        new = summary[summary.method.eq(method) & summary.distribution.eq("equal_mixture")].iloc[0]
        comparison.append({"method": method, "previous_mean_width": old.mean_width,
                           "merged_mean_width": new.mean_width, "previous_n_cells": len(kp),
                           "merged_n_cells": len(cells), "same_test_observations": True,
                           "same_score_map": True, "same_cell_targets": False,
                           "previous_guarantee": "marginal" if method == "c1" else "simultaneous",
                           "merged_guarantee": "simultaneous"})
    comparison = pd.DataFrame(comparison)
    elementary = pd.DataFrame({"elementary_bin_index": np.arange(len(kp)), "left": edges[:-1],
        "right": edges[1:], "k_p": kp, "k_q": kq, "merged_bin_index": lookup,
        "candidate_index": calibrated["selected_indices"][lookup]})
    paths = ["experiments/AGP/AGP_ICFM_merged_ci.py", "utils/calibration.py", "utils/__init__.py",
             "tests/test_calibration.py", "tests/test_merged_calibration.py",
             "experiments/JRSSB/CI/agent_CI.md"]
    code_hashes = {name: sha256(REPO / name) for name in paths}
    out.mkdir(parents=True)
    for name in paths:
        destination = out / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, destination)
    for name in ("model.pt", "training.json", "loss_history.csv", "scores.csv", "split_manifest.csv"):
        shutil.copyfile(source / name, out / name)
        require(sha256(out / name) == result["outputs"][name], f"Copy changed source artifact: {name}")
    shutil.copyfile(source / "protocol.json", out / "fit_protocol.json")
    shutil.copyfile(source / "result.json", out / "source_fit_result.json")
    protocol = {"analysis": "agp_real_vs_icfm_712_merged_calibration", "source_fit": str(source),
        "source_training": result["training"],
        "source_result_sha256": sha256(source / "result.json"), "source_model_sha256": sha256(source / "model.pt"),
        "source_scores_sha256": sha256(source / "scores.csv"), "elementary_edges": edges.tolist(),
        "calibration": calibrated["metadata"], "n_p": int(kp.sum()), "n_q": int(kq.sum()),
        "selected_candidates": calibrated["selected_indices"].tolist(), "n_selected_regions": len(cells),
        "model_retrained": False, "calibration_equals_test": True, "retrospective": True,
        "true_coverage_measured": False, "within_subject_correlation_adjusted": False,
        "code_sha256": code_hashes, "runtime": {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__}, "preflight": args.preflight}
    protocol["reproduce_command"] = shlex.join(["python3", "-B", str(out / "source/experiments/AGP/AGP_ICFM_merged_ci.py"),
        "--source-fit", str(source), "--output-dir", str(out.with_name(out.name + "_replay")),
        "--min-count", str(args.min_count), "--bootstrap-repetitions", str(args.bootstrap_repetitions),
        "--seed", str(args.seed), "--alpha", str(args.alpha)] + (["--preflight"] if args.preflight else []))
    save_json(out / "protocol.json", protocol)
    for name, frame in (("candidate_intervals", candidates), ("cell_intervals", cells), ("test_intervals", attached),
                        ("elementary_bin_counts", elementary), ("summary", summary), ("baseline_comparison", comparison),
                        ("split_calibration_cells", diagnostics), ("split_calibration_summary", diagnostic_summary)):
        frame.to_csv(out / f"{name}.csv", index=False)
    np.save(out / "bootstrap_maxima.npy", calibrated["bootstrap_maxima"], allow_pickle=False)
    make_plots(out, cells, diagnostics, calibrated["metadata"])
    write_report(out, protocol, manifest, cells, summary, comparison)
    artifacts = sorted(p for p in out.rglob("*") if p.is_file() and ".matplotlib" not in p.parts)
    save_json(out / "result.json", {"status": "complete", "analysis": protocol["analysis"],
        "source_verified_artifacts": len(result["outputs"]), "test_rows": len(test), "cells": len(cells),
        "candidates": len(candidates), "calibration": calibrated["metadata"],
        "outputs": {str(p.relative_to(out)): sha256(p) for p in artifacts}})
    print(json.dumps({"status": "complete", "output": str(out), "calibration": calibrated["metadata"],
                      "selected_regions": cells[["left", "right", "k_p", "k_q"]].to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
