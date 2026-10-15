#!/usr/bin/env python3
"""Render the four-generator AGP case study from completed aggregate studies.

The selected opening table and figure are copied byte-for-byte. New figures
compare frozen global metrics, common interpretable regions, and paired fitted
scores on the same real cohort. No row scores or subject identifiers are exported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
import numpy as np
import pandas as pd

METHODS = ("bernoulli_mixture_icfm", "autoregressive_icfm", "midasim_mask_icfm", "midasim")
LABELS = ("Bernoulli mixture + ICFM", "NADE mask + ICFM", "MIDASim mask + ICFM", "MIDASim")
SHORT = ("Bernoulli mixture\n+ ICFM", "NADE mask\n+ ICFM", "MIDASim mask\n+ ICFM", "MIDASim")
COLORS = ("#0072B2", "#009E73", "#CC79A7", "#D55E00")
REPOSITORY = Path(__file__).resolve().parents[2]
OPENING = REPOSITORY / "results/AGP/retained_mask_comparison_20261013"
OPENING_FILES = ("comparison.csv", "rdr_comparison.png", "rdr_comparison.pdf", "conventional_comparison.csv",
                 "fixed_width_cells.csv", "split_counts.csv", "null_controls.csv")
COMMUNITY_FILES = ("shared_communities.csv", "tree_fidelity.csv", "paired_real_scores.csv", "score_replay.csv", "community_rules.csv")
GLOBAL_FILES = ("global_metrics.csv", "global_metrics_aggregate.csv", "prd_curves.csv", "prd_curve_summary.csv",
                "prd_histograms.csv", "fd_reference_check.csv")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            result.update(part)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pin(path, inputs, expected=None):
    path = Path(path).resolve()
    actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Changed source: {path}")
    if str(path) in inputs and inputs[str(path)] != actual:
        raise ValueError(f"Source changed during rendering: {path}")
    inputs[str(path)] = actual


def save(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def markdown(columns, rows):
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(clean, columns)) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines += ["| " + " | ".join(map(clean, row)) + " |" for row in rows]
    return "\n".join(lines)


def global_figures(frame, curves, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True)
    for row, representation in enumerate(("sqrt", "clr")):
        table = frame[frame.representation.eq(representation)].set_index("method").loc[list(METHODS)]
        for col, (metric, title) in enumerate((("frechet", "Gaussian Fréchet analogue"), ("prd_max_f1", "Maximum PRD F1"))):
            ax = axes[row, col]
            for i, method in enumerate(METHODS):
                ax.errorbar(table.loc[method, metric + "_mean"], i, xerr=table.loc[method, metric + "_sd"],
                            fmt="o", ms=7, capsize=4, color=COLORS[i])
            ax.set_yticks(range(4), LABELS)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=.2)
            ax.set_title(("Square-root: primary" if row == 0 else "CLR: sensitivity") + " · " + title)
            ax.set_xlabel("Distance (smaller = closer moments)" if col == 0 else "F1 (larger = greater quantized overlap)")
            if col == 1:
                ax.set_xlim(0, 1)
    save(fig, output, "global_baselines")
    for representation in ("sqrt", "clr"):
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharex=True, sharey=True, constrained_layout=True)
        for i, (method, ax) in enumerate(zip(METHODS, axes)):
            subset = curves[curves.method.eq(method) & curves.representation.eq(representation)]
            for _, repeat in subset.groupby("seed", sort=True):
                ax.plot(repeat.recall, repeat.precision, color=COLORS[i], alpha=.17, lw=1)
            mean = subset.groupby("angle")[["precision", "recall"]].mean()
            ax.plot(mean.recall, mean.precision, color=COLORS[i], lw=2)
            ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", title=SHORT[i])
            ax.grid(alpha=.18)
        axes[0].set_ylabel("Precision")
        fig.suptitle("Full PRD curves · " + ("square-root primary representation" if representation == "sqrt" else "CLR sensitivity representation"))
        save(fig, output, "prd_" + representation)


def community_figures(frame, output):
    order = sorted(frame.community.unique())
    p = frame[frame.method.eq(METHODS[0])].set_index("community").loc[order]
    ticklabels = [f"{community}   P {row.p_fraction:.1%} (n={row.p_count})" for community, row in p.iterrows()]
    cmap = LinearSegmentedColormap.from_list("signed_rdr", ["#D55E00", "#faf9f6", "#0072B2"])
    fig, axes = plt.subplots(1, 2, figsize=(15, max(4.3, .48 * len(order) + 2)), constrained_layout=True)
    for ax, variable, title in zip(axes, ("empirical_region_rdr", "mean_fitted_rdr"),
                                   ("Empirical regional RDR", "Mean fitted RDR in region")):
        table = frame.pivot(index="community", columns="method", values=variable).loc[order, list(METHODS)]
        ax.imshow(table, cmap=cmap, vmin=0, vmax=2, aspect="auto")
        for row in range(len(order)):
            for column in range(4):
                value = table.iloc[row, column]
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if value < .35 or value > 1.7 else "#222222")
        ax.set_xticks(range(4), SHORT, fontsize=10)
        ax.set_yticks(range(len(order)), ticklabels if ax is axes[0] else order, fontsize=10)
        ax.set_title(title, pad=12)
        ax.set_xticks(np.arange(-.5, 4), minor=True)
        ax.set_yticks(np.arange(-.5, len(order)), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 2), cmap=cmap), ax=axes, shrink=.8, pad=.02)
    colorbar.set_ticks([0, 1, 2], labels=["0: Q excess", "1: equal mass", "2: Q deficit"])
    fig.suptitle("Common real communities · fixed leaf order", fontsize=15)
    save(fig, output, "community_rdr")


def paired_figure(private_path, summary, output):
    with np.load(private_path, allow_pickle=False) as private:
        scores = private["real_scores"]
    if scores.shape != (2002, 4) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 2)):
        raise ValueError("Invalid paired real-score matrix")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    plots = []
    for ax, (i, j) in zip(axes.flat, itertools.combinations(range(4), 2)):
        row = summary[summary.method_a.eq(METHODS[i]) & summary.method_b.eq(METHODS[j])].iloc[0]
        plots.append(ax.hexbin(scores[:, i], scores[:, j], gridsize=40, extent=(0, 2, 0, 2), mincnt=1,
                               cmap="Blues", norm=LogNorm(vmin=1, vmax=2002), linewidths=0, rasterized=True))
        ax.plot([0, 2], [0, 2], "--", color=".45", lw=1)
        ax.axhline(1, color=".5", ls=":", lw=.9)
        ax.axvline(1, color=".5", ls=":", lw=.9)
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel=LABELS[i] + " · fitted RDR", ylabel=LABELS[j] + " · fitted RDR")
        ax.set_title(f"Spearman {row.spearman:.2f} · opposite sides of 1: {row.opposite_fitted_direction_fraction:.1%}", fontsize=10)
    common_norm = LogNorm(vmin=1, vmax=max(float(plot.get_array().max()) for plot in plots))
    for plot in plots:
        plot.set_norm(common_norm)
    cb = fig.colorbar(plots[0], ax=axes, shrink=.85, pad=.02)
    cb.set_label("Real samples per hexagon (log scale)")
    fig.suptitle("Same real samples, different fitted RDRs\nFive-fit means; descriptive predictions, not oracle density ratios", fontsize=15)
    save(fig, output, "paired_real_scores")


def report(output, opening, global_table, regions, fidelity, paired, frozen, metrics_protocol):
    selected = {"js_a0.5": "JS / 0.5", "chisq_a0.5": "Chi-square / 0.5"}
    main_rows = []
    for row in opening.itertuples():
        main_rows.append([row.label, selected[row.selected_candidate],
            f"{row.auc_mean:.6f} ± {row.auc_sd:.6f}", f"{row.brier_mean:.6f} ± {row.brier_sd:.6f}",
            f"{row.local_gap_mean:.6f} ± {row.local_gap_sd:.6f}"])
    global_rows = []
    for method, label in zip(METHODS, LABELS):
        values = global_table[global_table.method.eq(method)].set_index("representation")
        global_rows.append([label, f"{values.loc['sqrt', 'frechet_mean']:.5f}",
                           f"{values.loc['sqrt', 'prd_max_f1_mean']:.4f} ± {values.loc['sqrt', 'prd_max_f1_sd']:.4f}",
                           f"{values.loc['clr', 'frechet_mean']:.2f}",
                           f"{values.loc['clr', 'prd_max_f1_mean']:.4f} ± {values.loc['clr', 'prd_max_f1_sd']:.4f}"])
    count_rows, rules_rows = [], []
    for community in sorted(regions.community.unique()):
        rows = regions[regions.community.eq(community)].set_index("method").loc[list(METHODS)]
        first = rows.iloc[0]
        count_rows.append([community, f"{first.p_fraction:.2%} ({first.p_count})"] +
                          [f"{row.q_fraction:.2%} ({row.q_count})" for row in rows.itertuples()])
        rules_rows.append([community, first.rule])
    extrema = []
    for method, label in zip(METHODS, LABELS):
        subset = regions[regions.method.eq(method)]
        low = subset.loc[subset.empirical_region_rdr.idxmin()]
        high = subset.loc[subset.empirical_region_rdr.idxmax()]
        extrema.append([label, f"{low.community}: {low.empirical_region_rdr:.3f}",
                        f"{high.community}: {high.empirical_region_rdr:.3f}",
                        f"{subset.empirical_abs_gap.max():.3f}"])
    fidelity_rows = [[dict(zip(METHODS, LABELS))[row.method], f"{row.real_assessment_mse:.4f}", f"{row.real_assessment_r2:.3f}"]
                     for row in fidelity.itertuples()]
    paired_rows = [[dict(zip(METHODS, LABELS))[row.method_a], dict(zip(METHODS, LABELS))[row.method_b],
                   f"{row.spearman:.3f}", f"{row.mean_absolute_score_difference:.3f}",
                   f"{row.opposite_fitted_direction_fraction:.2%}"] for row in paired.itertuples()]
    small = int(regions.small_count.sum())
    selections = frozen["selected"]
    sections = ["# AGP: a four-generator case study of local representation\n",
        "The case study compares Bernoulli-mixture-mask + ICFM, NADE-mask + ICFM, MIDASim-mask + ICFM and full MIDASim with equal emphasis. The question is which observed microbial communities each generator underrepresents or overrepresents, and whether the same real observations receive different fitted RDRs across generators. The opening comparison is unchanged. All later analyses reuse the existing generators and separately selected RDRs; no generator or RDR was refitted.",
        "## The selected RDR comparison\n",
        markdown(["Generator", "Selected loss / slope", "RDR source AUC", "Balanced Brier", "Local absolute gap"], main_rows),
        "Values are mean ± sample SD over five initializations on the same data. The numerical source table [comparison.csv](comparison.csv) and the opening figure are copied byte-for-byte from the retained comparison, with their recorded SHA-256 hashes verified. Source AUC and Brier measure this evaluator's source discrimination; neither is a generator-quality score. Local gap measures empirical calibration in fixed score cells, not distributional equality or pointwise accuracy.",
        "![Selected RDR score distributions and fixed-width calibration](rdr_comparison.png)",
        "Each column is one generator. Top: fitted observation-level RDR distributions for real P and generated Q. Bottom: mean fitted RDR against empirical RDR in 20 fixed score cells. Fit 0 is emphasized; four other fits are faint. Dark C.1 and light C.2 intervals are nominal cell-average diagnostics; they do not give confidence intervals for individual true density ratios. All final fits have supported mass one. These retrospective calibration diagnostics use the same assessment observations for empirical frequencies and neural-score means.",
        "## What information RDR adds\n",
        r"For generator $Q_m$, the population target is $r_m(x)=2p(x)/(p(x)+q_m(x))$. Values above one indicate local underrepresentation by $Q_m$; values below one indicate local overrepresentation. Its fitted version supplies a signed score for each observation, enabling a comparison of the same real observation across generators. These directions are interpretations of an estimate, conditional on evaluator quality, not verified individual density ratios.",
        "The reported scalar Gaussian Fréchet distance and Inception Score, or an unlabeled global precision–recall curve, do not name the biological locations of mismatch. The additional deliverable here is a shared, interpretable map that attaches signed representation differences and source mass to communities, plus a paired map of fitted scores on common real samples. This is a claim about the information in the reported outputs, not an exclusive mathematical capability: the histograms underlying PRD, local precision/recall, or other regional metrics can also localize mismatch when their regions are retained and interpreted. Indeed the empirical community ratios below are directly computable from region counts; RDR contributes to discovering the common partition and to scoring individual observations.",
        "[FID](https://papers.neurips.cc/paper/7240-gans-trained-by-a-two-time-scale-update-rule-converge-to-a-local-nash-equilibrium.pdf) compares Gaussian feature summaries; the [original Inception Score](https://arxiv.org/html/1606.03498v1) uses a pretrained sample-class predictor on generated samples; [PRD](https://papers.neurips.cc/paper/7769-assessing-generative-models-via-precision-and-recall.pdf) describes global precision–recall tradeoffs. AGP has no justified Inception classifier here, so **no numerical IS is reported**. Our distance is a Gaussian Fréchet analogue in declared compositional features, not image Inception FID. Related work also defines [individual realism scores](https://arxiv.org/abs/1904.06991), so samplewise quality information is not unique to RDR in the broader literature.",
        "## Global baselines for all four generators\n",
        markdown(["Generator", "Square-root FD", "Square-root PRD max F1", "CLR FD", "CLR PRD max F1"], global_rows),
        "Square-root composition is primary; CLR with pseudocount 1e-6 is a representation sensitivity analysis. Each pair-specific PRD quantizer has 64 clusters and was fitted to balanced development-only subsets of 2,048 P and 2,048 Q observations, using five prespecified seeds. All quantizers were frozen before assessment arrays were read by this extension. FD uses all assessment rows and unbiased sample covariances. FD is identical across quantizer seeds; PRD SD describes quantizer initialization/subsampling variation on the fixed cohort, not sampling uncertainty. Scales differ between representations, so their distances should not be compared directly.",
        "![Global baseline summaries](global_baselines.png)",
        "![Complete primary PRD curves for all four generators](prd_sqrt.png)",
        "Full curves retain every one of the 201 prespecified angle points. Five curves are faint and their anglewise mean is emphasized. Maximum F1 is only a supplemental scalar summary. These are distribution PRD curves, not the classification precision–recall curve of the RDR network. The [raw curves](global/prd_curves.csv) and [cluster counts](global/prd_histograms.csv) remain available.",
        "![Complete CLR sensitivity PRD curves](prd_clr.png)",
        "## A common map of real communities\n",
        f"A single multioutput tree summarizes all four frozen five-fit mean RDRs using the original {frozen['real_stopping_n']:,} real RDR stopping observations. A fixed subject-disjoint 70%/30% split (seed {frozen['tree_subject_split_seed']}) gives {frozen['tree_training_n']:,} observations to fit the tree and {frozen['tree_validation_n']:,} to select its depth. These observations were excluded from RDR gradient updates but participated in RDR checkpoint choice; they are not a new independent teacher-validation cohort. The tree uses {frozen['feature_count']} interpretable composition summaries (richness, entropy, dominant abundance, taxonomic groups defined from the original {frozen['real_train_n']:,} real training observations, and log balances). Depths 2, 3 and 4 were compared by average squared error across all four targets on the tree-validation subset. Depth {selections['depth']} with {selections['leaves']} leaves was selected; all rules and leaf order were frozen before this extension read assessment compositions or scores. Every leaf is shown, in increasing leaf-index order, without assessment-based filtering or reordering.",
        r"For each fixed community $C$, the empirical regional target is $\hat\theta_m(C)=2\hat P(C)/(\hat P(C)+\hat Q_m(C))$. The corresponding fitted mean uses equal-source mixture weights: $(n_P^{-1}\sum_{i\in P\cap C}\hat r_m(x_i)+n_Q^{-1}\sum_{j\in Q_m\cap C}\hat r_m(x_j))/(\hat P(C)+\hat Q_m(C))$. The formula matters because source sample sizes differ. The left heatmap measures empirical community occupancy; the right summarizes the fitted RDR. Agreement between them must be checked rather than assumed.",
        "![Common community empirical regional RDR and mean fitted RDR](community_rdr.png)",
        "Blue values above one indicate a Q deficit; orange values below one indicate a Q excess. Both panels use the same scale, all four methods and every frozen community. Labels include P fraction and P count. A community value is a regional average, not a constant individual RDR throughout that region.",
        markdown(["Community", "Real P fraction (n)"] + [label + " Q fraction (n)" for label in LABELS], count_rows),
        "The following rules define the shared regions. `log mass/rest` is the log of group mass plus 1e-6 divided by complementary mass plus 1e-6; `log balance` compares two group masses with the same pseudocount. Rules are reproducible mathematical descriptions, not clinical labels or established microbial community classes.",
        markdown(["Community", "Frozen rule"], rules_rows),
        "Three descriptive comparisons make the common-region information concrete. **C01**, low Lachnospiraceae and low Streptococcaceae (approximately at most 3.85% and 6.12% mass respectively), contains 6.49% of real observations. The three ICFM methods assign it only 1.20%, 1.01% and 0.72% of generated mass, whereas full MIDASim assigns 5.98%. Its empirical regional RDRs are 1.689, 1.732, 1.801 and 1.041 in the four-generator order. Thus the near-matched full-MIDASim regional frequency differs from the shared deficit in the three ICFM methods, without implying that samples within this region match in distribution.",
        "**C02**, low Lachnospiraceae and higher Streptococcaceae, contains 3.15% of real observations. The Bernoulli mixture places 6.56% of its samples there, while NADE, MIDASim-mask + ICFM and full MIDASim place 1.87%, 1.34% and 1.01%. The same community therefore has an empirical excess for the mixture and empirical deficits for the other three: regional RDRs 0.649, 1.255, 1.403 and 1.516. Fitted regional means have the same directions (0.887, 1.402, 1.567 and 1.501), but the mixture's absolute gap is 0.239. The hybrid and full MIDASim have only 28 and 21 generated observations in this community; the low counts remain explicit.",
        "**C08** contains 41.06% of real observations and, under its full rule above, has larger Lachnospiraceae and Ruminococcaceae masses. The three ICFM methods allocate 59.60%, 55.34% and 57.25% there, versus full MIDASim's 37.19%; empirical regional RDRs are 0.816, 0.852, 0.835 and 1.049. All eight communities, including those with smaller contrasts, remain visible. These examples describe estimated regional frequencies, not causal family effects or statistically significant direction changes.",
        "The extrema below are descriptive summaries across all displayed leaves, not hypothesis tests or independently selected biological discoveries:",
        markdown(["Generator", "Lowest empirical regional RDR", "Highest empirical regional RDR", "Largest absolute fitted–empirical gap"], extrema),
        f"There are {small} generator–community cells with fewer than 30 observations from at least one source. All are retained and flagged in [shared_communities.csv](communities/shared_communities.csv), which includes P/Q counts and fractions, balanced mass, empirical RDR, fitted means, five-fit SD and absolute gaps. No new regional confidence intervals or multiple-testing claims are made. The partition was learned from P only: it describes real communities and does not discover modes supported exclusively by generated Q.",
        "The small tree is an interpretable summary of the neural score, not the neural evaluator itself. Its fidelity on real assessment samples is:",
        markdown(["Generator", "Tree mean squared error", "Tree R²"], fidelity_rows),
        "Low or negative R² means the interpretable partition captures little of that evaluator's samplewise variation. Large fitted–empirical regional gaps mean that fitted RDR magnitudes should not be used as reliable community mass estimates; report empirical occupancies alongside them. The same source observations contribute to both quantities, so this comparison is descriptive, not an independent validation of oracle RDR recovery.",
        "## The same real observations across generators\n",
        "![Paired fitted scores on the common real cohort](paired_real_scores.png)",
        "Each panel compares the five-fit mean predictions for the same 2,002 real assessment observations. Hexagons contain counts, with no sample labels exported. The dashed diagonal marks equal fitted scores; dotted lines mark one. Off-diagonal points and rank disagreement show that fitted RDR can assign generator-specific scores to the same real observation. Opposite sides of one are opposite fitted directions only; they do not establish true local underrepresentation versus overrepresentation.",
        markdown(["Generator A", "Generator B", "Spearman", "Mean absolute score difference", "Opposite fitted directions"], paired_rows),
        "The [paired summary](communities/paired_real_scores.csv) covers all six pairs, with no favorable pair selected for display. These 2,002 rows form one cohort; the 20 reused RDR fits are four sets of five initializations, not independent samples of biological populations.",
        "## Scope, limitations and reproducibility\n",
        "The case study is retrospective: the assessment cohort had been examined in earlier analyses. New tree and quantizer choices were frozen before this extension's assessment, but that does not turn the existing cohort into a new confirmatory sample. Real subjects are disjoint between development and assessment; generator training overlaps real evaluator training/stopping. Each comparison uses 2,002 real and 2,089 generated assessment observations on 614 taxa. Resampled observed masks are excluded by the prior paper-scope decision, with their original records preserved elsewhere.",
        "A preparation-only tree trained on in-fit RDR training predictions showed negative stopping R² for three methods. Before any shared-community assessment, its development design was revised to use the subject-disjoint inner split of original RDR stopping observations described above. That initial preparation is preserved separately in the private staging area, and the final freeze records this revision. No assessment-guided tree revision followed.",
        "Evaluator dependence remains substantial. In the unchanged conventional comparison, NADE's boosted-tree composition AUC is 0.9896 while its neural RDR source AUC is 0.7334. Thus the neural comparison does not establish NADE fidelity or an ordering that is invariant to evaluator choice. The two MIDASim-mask methods share the assessed occurrence masks by construction; differences in their local representations compare their abundance-generating mechanisms conditional on that common mask design. This is not a causal decomposition of the networks' learned features.",
        "The opening table/figure and reused conventional summaries are unchanged. All 20 assessment score replays match the stored predictions exactly. The new public package exports aggregate tables, figures, rules and source snapshots only: no subject IDs, sample IDs, row-level score tables, abundance arrays or fitted checkpoints. Individual score coordinates are summarized in a hexagonal count image, with no vector point layer. Source and input hashes, completion gates, counts and output hashes are recorded in [manifest.json](manifest.json). Independent package audit is a separate step; rendering completion alone does not certify that audit.",
        "The primary paper sequence is the unchanged four-generator RDR table/figure, global baseline context, the common community map with occupancy evidence, and the common-real paired fitted-score figure. Controlled perturbation is not the central evidence in this AGP study. The supported contribution is interpretable, signed, generator-specific diagnostic information beyond the listed global summaries, with calibration and evaluator limitations visible alongside it."]
    (output / "report.md").write_text("\n\n".join(sections) + "\n")


def run(communities, metrics, output):
    communities, metrics, output = map(lambda p: Path(p).resolve(), (communities, metrics, output))
    if output.exists():
        raise FileExistsError("Use a fresh output; existing artifacts are preserved")
    inputs = {}
    # Both complete gates are required before loading any assessment table.
    community_seal = read(communities / "ASSESSMENT_COMPLETE.json")
    global_seal = read(metrics / "summary/COMPLETE.json")
    if community_seal["status"] != "complete" or global_seal["status"] != "complete":
        raise ValueError("Both assessment studies must be complete")
    if community_seal["score_replays"] != 20 or community_seal["max_score_replay_difference"] != 0:
        raise ValueError("Incomplete or changed RDR score replays")
    pin(communities / "ASSESSMENT_COMPLETE.json", inputs)
    pin(communities / "FREEZE.json", inputs, community_seal["freeze_sha256"])
    pin(metrics / "summary/COMPLETE.json", inputs)
    pin(metrics / "FREEZE.json", inputs, global_seal["freeze_sha256"])
    pin(metrics / "protocol.json", inputs, read(metrics / "FREEZE.json")["protocol_sha256"])
    for name, digest in community_seal["files_sha256"].items():
        pin(communities / name, inputs, digest)
    for name, digest in global_seal["files"].items():
        pin(metrics / "summary" / name, inputs, digest)
    opening_seal = read(OPENING / "manifest.json")
    if opening_seal["status"] != "complete" or tuple(opening_seal["retained_methods"]) != METHODS:
        raise ValueError("Opening comparison is not the retained four-method comparison")
    pin(OPENING / "manifest.json", inputs)
    for name in OPENING_FILES:
        pin(OPENING / name, inputs, opening_seal["files_sha256"][name])
    sources = [Path(__file__).resolve(), Path(__file__).with_name("four_generator_communities.py").resolve(),
               Path(__file__).with_name("four_generator_metrics.py").resolve()]
    for path in sources:
        pin(path, inputs)
    frame = pd.read_csv(communities / "summary/shared_communities.csv")
    common = frame[frame.method.eq(METHODS[0])].set_index("community")
    for method in METHODS:
        part = frame[frame.method.eq(method)].set_index("community").loc[common.index]
        np.testing.assert_array_equal(part[["leaf", "p_count", "p_fraction", "rule"]].to_numpy(),
                                      common[["leaf", "p_count", "p_fraction", "rule"]].to_numpy())
    if set(frame.method) != set(METHODS) or len(frame) != 4 * community_seal["communities"]:
        raise ValueError("Incomplete community grid")
    opening = pd.read_csv(OPENING / "comparison.csv")
    global_table = pd.read_csv(metrics / "summary/global_metrics_aggregate.csv")
    curves = pd.read_csv(metrics / "summary/prd_curves.csv")
    fidelity = pd.read_csv(communities / "summary/tree_fidelity.csv")
    paired = pd.read_csv(communities / "summary/paired_real_scores.csv")
    if len(global_table) != 8 or len(curves) != 8040 or len(paired) != 6:
        raise ValueError("Incomplete four-generator summaries")
    output.mkdir(parents=True)
    for folder in ("communities", "global", "provenance", "source"):
        (output / folder).mkdir()
    for name in OPENING_FILES:
        shutil.copyfile(OPENING / name, output / name)
        if sha(output / name) != inputs[str((OPENING / name).resolve())]:
            raise ValueError("Opening copy changed")
    for name in COMMUNITY_FILES:
        shutil.copyfile(communities / "summary" / name, output / "communities" / name)
    for name in GLOBAL_FILES:
        shutil.copyfile(metrics / "summary" / name, output / "global" / name)
    for source in sources:
        shutil.copyfile(source, output / "source" / source.name)
    for source, name in ((communities / "FREEZE.json", "communities_freeze.json"),
                         (communities / "ASSESSMENT_COMPLETE.json", "communities_assessment_complete.json"),
                         (metrics / "FREEZE.json", "metrics_freeze.json"),
                         (metrics / "protocol.json", "metrics_protocol.json"),
                         (metrics / "summary/COMPLETE.json", "metrics_assessment_complete.json"),
                         (OPENING / "manifest.json", "opening_manifest.json")):
        shutil.copyfile(source, output / "provenance" / name)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    global_figures(global_table, curves, output)
    community_figures(frame, output)
    paired_figure(communities / "assessment_private.npz", paired, output)
    report(output, opening, global_table, frame, fidelity, paired, read(communities / "FREEZE.json"), read(metrics / "protocol.json"))
    for name, digest in inputs.items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Input changed during rendering: {name}")
    manifest = {"status": "complete", "kind": "aggregate_case_study_rendering", "created_utc": datetime.now(timezone.utc).isoformat(),
                "retained_methods": list(METHODS), "inputs_sha256": inputs,
                "opening_files_copied_unchanged": {name: sha(output / name) for name in OPENING_FILES},
                "copied_opening_sha256": {str(OPENING / name): {"destination": name, "sha256": sha(output / name)} for name in OPENING_FILES},
                "aggregate_source": {**{str(communities / "summary" / name): "communities/" + name for name in COMMUNITY_FILES},
                                     **{str(metrics / "summary" / name): "global/" + name for name in GLOBAL_FILES}},
                "assessment_rows_per_method": {"P": 2002, "Q": 2089}, "taxa": 614,
                "shared_communities": community_seal["communities"], "regional_rows": len(frame), "global_metric_rows": 40,
                "full_prd_curve_rows": len(curves), "paired_real_comparisons": len(paired),
                "score_replays": 20, "max_score_replay_difference": 0, "new_generator_or_rdr_fits": 0,
                "independent_package_audit": "not performed by this renderer",
                "scope": "Retrospective fitted-score diagnostics and empirical regional frequencies, not oracle pointwise RDR recovery or causal attribution",
                "private_excluded": ["row-level score tables", "subject IDs", "sample IDs", "abundance arrays", "checkpoints", "assessment_private.npz"],
                "files_sha256": {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}}
    write(output / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "output": str(output), "files": len(manifest["files_sha256"]),
                      "communities": community_seal["communities"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--communities", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.communities, args.metrics, args.output)
