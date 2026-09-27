"""Render the AGP compositional-input comparison from saved diagnostic tables."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shlex

import numpy as np
import pandas as pd


LABELS = {"archived_raw": "Archived raw MLP", "raw": "Raw MLP",
          "closed_raw": "Closed raw MLP", "ilr": "ILR + MLP",
          "logcontrast": "Learned log-contrast", "philr": "Phylogenetic ILR + MLP"}
COLORS = dict(zip(LABELS, ("#777777", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd")))
ROLES = ("train", "validation", "test")
REFERENCE_SEED = 20260918
MEASURES = ("empirical_bin_absolute_gap", "empirical_bin_rmse", "balanced_brier", "auc",
            "equal_mixture_mean_score", "midpoint_loss")


def _table(frame, columns, names=None):
    """Markdown without an optional tabulate dependency."""
    lines = ["| " + " | ".join(names or columns) + " |", "|" + "---|" * len(columns)]
    for row in frame[columns].itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.6g}" if np.isfinite(value) else "—")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _save(fig, out, name, plt):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=180)
    fig.savefig(out / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def _reference(metrics, seed=REFERENCE_SEED):
    reference = metrics[metrics.primary & metrics.seed.eq(seed)].copy()
    # An exact baseline reproduction needs only one line in the main figure.
    old = reference[reference.method.eq("archived_raw")].set_index("role")
    new = reference[reference.method.eq("raw")].set_index("role")
    if len(old) and set(old.index) == set(new.index):
        if np.allclose(old.loc[new.index, list(MEASURES)], new[list(MEASURES)],
                       rtol=1e-7, atol=1e-8, equal_nan=True):
            reference = reference[~reference.method.eq("archived_raw")]
    return reference


def _score_regions(cells):
    """Post-hoc descriptive mass/error decomposition; never a selection criterion."""
    records = []
    selected = cells[cells.primary & ~cells.method.eq("archived_raw")]
    for (run_id, role), rows in selected.groupby(["run_id", "role"]):
        first = rows.iloc[0]
        middle = rows.left.ge(.2 - 1e-12) & rows.right.le(1.8 + 1e-12)
        mass = rows.mixture_mass
        gap = rows.calibration_gap
        if not np.isclose(mass.sum(), 1., rtol=0, atol=1e-10) or gap[mass > 0].isna().any():
            raise ValueError("Invalid cell mass or missing nonempty-cell calibration gap")
        mid_mass = float(mass[middle].sum())
        mid_contribution = float((mass[middle] * gap[middle].abs()).sum())
        end_contribution = float((mass[~middle] * gap[~middle].abs()).sum())
        records.append({"run_id": run_id, "method": first.method, "seed": first.seed,
            "epsilon": first.epsilon, "primary": True, "role": role, "posthoc_descriptive": True,
            "middle_left": .2, "middle_right_exclusive": 1.8, "middle_mass": mid_mass,
            "middle_conditional_absolute_gap": mid_contribution / mid_mass if mid_mass else np.nan,
            "middle_conditional_signed_gap": float((mass[middle] * gap[middle]).sum()) / mid_mass if mid_mass else np.nan,
            "middle_gap_contribution": mid_contribution, "endpoint_mass": float(mass[~middle].sum()),
            "endpoint_gap_contribution": end_contribution,
            "all_bin_absolute_gap": mid_contribution + end_contribution})
    return pd.DataFrame(records)


def render_report(out, metrics, cells, history, protocol):
    """Write plots, a seed-summary CSV, and report.md; never fit or select models."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference_seed = protocol.get("representative_seed", REFERENCE_SEED)
    primary_epsilon = protocol.get("config", {}).get("epsilon", 1e-6)
    reference = _reference(metrics, reference_seed)
    run_ids = set(reference.run_id)
    methods = [method for method in LABELS if method in set(reference.method)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, role, title in zip(axes, ROLES, ("Training", "Validation", "Test / calibration")):
        for method in methods:
            rows = cells[cells.run_id.isin(run_ids) & cells.method.eq(method) & cells.role.eq(role)]
            rows = rows[~rows["empty"]].sort_values("bin_index")
            ax.plot(rows.mean_model_score, rows.cell_rdr, "o-", ms=3, lw=1.2,
                    color=COLORS[method], label=LABELS[method])
        ax.plot([0, 2], [0, 2], ":", color=".4", lw=1)
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel="Balanced within-bin mean neural RDR", title=title)
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Frequency RDR from separately normalized P/Q counts")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"AGP score versus frequency: fixed seed {reference_seed}; 20 fixed score bins")
    _save(fig, out, "score_vs_frequency", plt)

    region_summary = _score_regions(cells)
    region_summary.to_csv(out / "score_region_summary.csv", index=False)
    region_columns = ["middle_mass", "middle_conditional_absolute_gap", "middle_gap_contribution",
                      "endpoint_mass", "endpoint_gap_contribution", "all_bin_absolute_gap"]
    region_means = region_summary.groupby(["method", "role"])[region_columns].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
    positive_mass = cells.loc[cells.run_id.isin(run_ids) & cells.mixture_mass.gt(0), "mixture_mass"]
    for ax, role in zip(axes, ROLES):
        for method in methods:
            rows = cells[cells.run_id.isin(run_ids) & cells.method.eq(method) & cells.role.eq(role)].sort_values("bin_index")
            edges = np.r_[rows.left, rows.right.iloc[-1]]
            mass = rows.mixture_mass.to_numpy()
            ax.stairs(np.where(mass > 0, mass, np.nan), edges, color=COLORS[method], label=LABELS[method])
        ax.axvspan(.2, 1.8, color=".8", alpha=.15)
        ax.set(xlim=(0, 2), yscale="log", ylim=(positive_mass.min() / 2, 1),
               xlabel="Neural RDR score", title=role.title())
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Balanced mixture mass per score bin (log)")
    axes[1].legend(fontsize=8, loc="lower center")
    fig.suptitle(f"Where the samples lie: fixed seed {reference_seed}; shaded range [0.2, 1.8)")
    _save(fig, out, "score_bin_mass", plt)

    primary = metrics[metrics.primary & ~metrics.method.eq("archived_raw")]
    n_primary_seeds = primary.seed.nunique()
    grouped = primary.groupby(["method", "role"])
    summary = grouped[list(MEASURES)].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary["n_seeds"] = grouped.seed.nunique().to_numpy()
    summary.to_csv(out / "seed_summary.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    plot_methods = [method for method in LABELS if method in set(primary.method)]
    for column, role in enumerate(ROLES):
        for row, measure in enumerate(("empirical_bin_absolute_gap", "balanced_brier")):
            ax = axes[row, column]
            for i, method in enumerate(plot_methods):
                values = primary[primary.method.eq(method) & primary.role.eq(role)][measure].to_numpy()
                ax.scatter(np.full(len(values), i), values, color=COLORS[method], s=25, alpha=.7)
                ax.plot([i-.2, i+.2], [np.mean(values)]*2, color="black", lw=1.5)
            ax.set_xticks(range(len(plot_methods)), [LABELS[m] for m in plot_methods], rotation=30, ha="right")
            ax.grid(axis="y", alpha=.2)
            if row == 0:
                ax.set_title(role.title())
            if column == 0:
                ax.set_ylabel("Weighted absolute gap (RDR units)" if row == 0 else "Balanced Brier score")
    fig.suptitle("Primary runs: individual seeds and their mean; lower values are better")
    _save(fig, out, "seed_comparison", plt)

    sensitivity = metrics[metrics.seed.eq(reference_seed) & metrics.role.eq("validation")
                          & metrics.method.isin(("ilr", "logcontrast", "philr"))]
    if len(sensitivity):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
        for ax, measure, title in zip(axes, ("empirical_bin_absolute_gap", "balanced_brier", "auc"),
                                     ("Absolute calibration gap", "Balanced Brier score", "AUC")):
            for method in LABELS:
                rows = sensitivity[sensitivity.method.eq(method)].sort_values("epsilon")
                if len(rows):
                    ax.plot(rows.epsilon, rows[measure], "o-", color=COLORS[method], label=LABELS[method])
            ax.set(xscale="log", xlabel="Additive epsilon after closure", title=title)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"Validation-only epsilon sensitivity; primary epsilon = {primary_epsilon:g}")
        _save(fig, out, "epsilon_sensitivity", plt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method in methods:
        rows = history[history.run_id.isin(run_ids) & history.method.eq(method)].sort_values("epoch")
        if len(rows):
            for ax, measure in zip(axes, ("train_loss", "validation_loss")):
                ax.plot(rows.epoch, rows[measure], color=COLORS[method], label=LABELS[method], lw=1.2)
    for ax, title in zip(axes, ("Training objective", "Validation objective")):
        ax.set(xlabel="Epoch", ylabel="Midpoint Hellinger objective", title=title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    _save(fig, out, "loss_histories", plt)

    selected_id = protocol.get("selection", {}).get("selected_run_id")
    selected = metrics[metrics.run_id.eq(selected_id)] if selected_id else metrics.iloc[:0]
    preflight = protocol.get("config", {}).get("preflight", False)
    test_means = summary[summary.role.eq("test")].set_index("method")
    transformed = [m for m in ("ilr", "logcontrast", "philr") if m in test_means.index]
    improves = bool(transformed) and "raw" in test_means.index and all(
        test_means.loc[m, metric + "_mean"] < test_means.loc["raw", metric + "_mean"]
        for m in transformed for metric in ("balanced_brier", "empirical_bin_absolute_gap"))
    region_test = region_means[region_means.role.eq("test")].set_index("method")
    unresolved = bool(transformed) and "raw" in region_test.index and all(
        region_test.loc[m, "middle_conditional_absolute_gap"] > region_test.loc["raw", "middle_conditional_absolute_gap"]
        and region_test.loc[m, "middle_mass"] < region_test.loc["raw", "middle_mass"] for m in transformed)
    result_intro = ("This is a preflight run for implementation checks, not evidence about model performance. "
        "The decomposition below checks the distinction between score-bin mass and conditional calibration gaps."
        if preflight else
        ("**The transformations do not uniformly improve score calibration.** Their overall weighted gaps "
         "and balanced Brier scores improve in this experiment, but their conditional middle-score gaps are larger. "
         "Several intermediate-score bins remain well above the diagonal: neural RDR still underpredicts "
         "frequency RDR there. Much less probability mass remains in those bins, while most samples move toward "
         "scores near 0 and 2. The smaller global weighted gap does not show that the middle discrepancy is resolved."
         if improves and unresolved else
         "The tables distinguish global weighted calibration gaps from conditional gaps at intermediate scores. "
         "An overall improvement alone does not establish better calibration throughout the score range."))
    text = ["# AGP compositional-input RDR diagnostics", "",
        "The observed AGP and archived ICFM samples, train/validation/test assignments, Hellinger objective, "
        "bounded RDR output, and fitting schedule are retained. The comparison changes the input representation "
        "and first-layer parameterization. This is a retrospective diagnostic experiment; the test cohort was "
        "already examined in earlier analyses and is not a newly untouched confirmation set.", "",
        "## What improved, and what remains unresolved", "",
        result_intro, "",
        f"The table below averages over the {n_primary_seeds} primary training seeds. The middle is defined as scores "
        "in **[0.2, 1.8)**; its conditional gap averages only over the mixture mass in that range.", "",
        _table(region_means[region_means.role.eq("test")],
               ["method", "middle_mass", "middle_conditional_absolute_gap", "all_bin_absolute_gap"],
               ["Method", "Test mass in middle", "Conditional middle gap", "Overall weighted gap"]), "",
        "This score-range decomposition was chosen after inspecting the plots and is **post-hoc descriptive "
        "context, not a model-selection criterion**. A network's middle-score region differs from another "
        "network's region; these are not comparisons of the same population subset.", "",
        "## Main score-versus-frequency comparison", "",
        "![Score versus frequency](score_vs_frequency.png)", "",
        f"The main figure uses the prespecified seed **{reference_seed}**, with **epsilon = {primary_epsilon:g}** for "
        "log-ratio models. It does not choose the best-looking seed. An archived raw curve is omitted from the "
        f"figure when its numerical diagnostics match the reproduced raw fit. The {n_primary_seeds}-seed comparison is below.", "",
        "The 20 elementary score intervals on [0,2] are fixed for every model and split. We use these fixed "
        "bins for the architecture comparison, rather than the earlier h=40 merging. The numeric intervals "
        "are shared, but different networks can assign different observations to them. Empty bins are omitted; "
        "small populated bins can produce noisy frequency estimates. These are descriptive diagnostics, not CIs.", "",
        "### Exact axes and summary metrics", "",
        r"Within one split, let $n,m$ be its P/Q sizes, $A_j=\{z:\hat r(z)\in I_j\}$, "
        r"$\hat p_j=k_{Pj}/n$, and $\hat q_j=k_{Qj}/m$. The plotted coordinates are", "",
        "$$", r"x_j=\frac{n^{-1}\sum_{i=1}^{n}\hat r(X_i)\mathbf1\{X_i\in A_j\}"
        r"+m^{-1}\sum_{\ell=1}^{m}\hat r(Y_\ell)\mathbf1\{Y_\ell\in A_j\}}{\hat p_j+\hat q_j},"
        r"\qquad y_j=\frac{2\hat p_j}{\hat p_j+\hat q_j}.", "$$", "",
        "Each distribution receives total weight 1/2 despite unequal sample sizes. The horizontal coordinate "
        "averages neural scores; the vertical coordinate uses empirical P/Q frequencies. Using the same "
        "training observations for both calculations does not force the two values to agree.", "",
        "$$", r"G=\sum_{j:\hat p_j+\hat q_j>0}\frac{\hat p_j+\hat q_j}{2}|y_j-x_j|,"
        r"\qquad G_2=\sqrt{\sum_j\frac{\hat p_j+\hat q_j}{2}(y_j-x_j)^2}.", "$$", "",
        "Lower empirical gaps indicate closer agreement in these bins. They are in RDR units, not probability "
        "units, and are not unbiased population calibration errors. A constant score of 1 has zero balanced "
        "empirical gap but no discrimination, so we also report balanced Brier score and AUC:", "",
        "$$", r"B=\tfrac12\overline{(1-\hat r_P/2)^2}+\tfrac12\overline{(\hat r_Q/2)^2}.", "$$", "",
        "Lower Brier and higher AUC are better; the constant-1 RDR has Brier 0.25 and AUC 0.5. "
        "The equal-mixture mean score should approach 1 for a correct RDR. Training and validation "
        "diagnostics use observations that influenced fitting and selection.", "",
        "## Sample mass and the remaining middle-score discrepancy", "",
        "![Balanced score-bin mass](score_bin_mass.png)", "",
        "Each line shows the balanced empirical mixture mass of every elementary score bin; zero-mass bins "
        "are omitted on the logarithmic axis. The gray area marks the post-hoc middle-score range. The "
        "following table averages each run's quantities over primary seeds, separately within each role.", "",
        "$$", r"w_j=\frac{\hat p_j+\hat q_j}{2},\quad J_{\mathrm{mid}}=\{j:I_j\subset[0.2,1.8)\},\quad "
        r"w_{\mathrm{mid}}=\sum_{j\in J_{\mathrm{mid}}}w_j,\quad "
        r"G_{\mathrm{mid}}=\frac{\sum_{j\in J_{\mathrm{mid}}}w_j|y_j-x_j|}{w_{\mathrm{mid}}},", "$$", "",
        "$$", r"G=w_{\mathrm{mid}}G_{\mathrm{mid}}+\sum_{j\notin J_{\mathrm{mid}}}w_j|y_j-x_j|.", "$$", "",
        _table(region_means, ["method", "role", *region_columns],
               ["Method", "Split", "Middle mass", "Conditional middle gap", "Middle contribution",
                "Endpoint mass", "Endpoint contribution", "Overall gap"]), "",
        "`score_region_summary.csv` records every primary run's decomposition, including the signed middle "
        "gap. Conditional middle gaps are undefined when that range contains no observations. The identity "
        "above holds within each run; products of separate seed means need not reproduce the mean contribution.", "",
        "## Quantitative results", "",
        _table(reference.sort_values(["role", "method"]),
               ["method", "role", "empirical_bin_absolute_gap", "balanced_brier", "auc",
                "equal_mixture_mean_score", "best_epoch", "epochs_run"],
               ["Method", "Split", "Absolute gap", "Brier", "AUC", "Mixture mean score", "Best epoch", "Epochs run"]), ""]
    if len(selected):
        text.extend([f"Selected primary run: **`{selected_id}`**, using validation balanced Brier only. "
                     "Each run's checkpoint was first selected by the original validation Hellinger objective. "
                     "Epsilon sensitivity runs did not enter selection; test results did not enter selection.", "",
                     _table(selected, ["role", "empirical_bin_absolute_gap", "balanced_brier", "auc"]), ""])
        test = selected[selected.role.eq("test")]
        baseline = primary[primary.method.eq("raw") & primary.role.eq("test")
                           & primary.seed.eq(selected.seed.iloc[0])]
        if len(test) == len(baseline) == 1:
            a, b = test.iloc[0], baseline.iloc[0]
            text.extend([f"For that selected run, the test absolute gap is {a.empirical_bin_absolute_gap:.5f}, "
                f"compared with {b.empirical_bin_absolute_gap:.5f} for the raw MLP using the same seed. "
                f"Its test Brier score is {a.balanced_brier:.5f} versus {b.balanced_brier:.5f}, and AUC "
                f"is {a.auc:.5f} versus {b.auc:.5f}. These are descriptive comparisons on the same observations; "
                "selection was fixed using validation before these test metrics were computed.", ""])
    text.extend(["## Variation across training seeds", "", "![Seed comparison](seed_comparison.png)", "",
        "Dots are individual primary runs and black horizontal marks are their means. Training seeds reuse "
        "the same data and splits; they measure fitting variation, not independent sampling uncertainty.", "",
        _table(summary, ["method", "role", "n_seeds", "empirical_bin_absolute_gap_mean",
                         "empirical_bin_absolute_gap_std", "balanced_brier_mean", "auc_mean"]), "",
        "For a compact comparison, the following ratios divide each method's mean absolute gap by the "
        "raw MLP's mean for the same split. A ratio below 1 indicates a smaller diagnostic gap; use the "
        "Brier and AUC results above to distinguish calibration from discrimination.", ""])
    ratios = []
    for role in ("validation", "test"):
        values = summary[summary.role.eq(role)].set_index("method")
        if "raw" in values.index and values.loc["raw", "empirical_bin_absolute_gap_mean"] > 0:
            for method in values.index:
                ratios.append({"method": method, "role": role, "gap_ratio_to_raw":
                    values.loc[method, "empirical_bin_absolute_gap_mean"] / values.loc["raw", "empirical_bin_absolute_gap_mean"]})
    if ratios:
        text.extend([_table(pd.DataFrame(ratios), ["method", "role", "gap_ratio_to_raw"]), ""])
    text.extend([
        "## Representations and target", "",
        r"The raw control uses archived abundances. The closed raw control uses $c_j=x_j/\sum_kx_k$. "
        r"For all log-ratio models, the common positive transform is $\tilde c_j=(c_j+\epsilon)/(1+D\epsilon)$, "
        r"with $D=614$. Write $u=\operatorname{clr}(\tilde c)=\log\tilde c-D^{-1}\mathbf1\sum_j\log\tilde c_j$.", "",
        r"Training-only preprocessing uses the equal-mixture CLR mean $\mu=\tfrac12\overline u_P+\tfrac12\overline u_Q$ "
        r"and a common positive scalar RMS $s$; the standardized contrast vector is $v=(u-\mu)/s$.", "",
        "$$", r"s^2=\frac{1}{2nD}\sum_{i=1}^n\|u(X_i)-\mu\|^2"
        r"+\frac{1}{2mD}\sum_{\ell=1}^m\|u(Y_\ell)-\mu\|^2.", "$$", "",
        r"ILR uses $vV$, where $V^\top V=I_{D-1}$ and $V^\top\mathbf1=0$. Phylogenetic ILR uses the "
        r"corresponding tree-balance basis. The learned log-contrast first layer uses $Wv+b$ with "
        r"$W=\widetilde W-\operatorname{rowmean}(\widetilde W)\mathbf1^\top$, so $W\mathbf1=0$ exactly "
        "up to numerical precision. It replaces the first hidden layer; it does not add an extra layer.", "",
        "Full ILR, full phylogenetic ILR, and a zero-sum log-contrast first layer span the same linear contrast "
        "space and therefore the same three-hidden-layer function class. Their optimization coordinates differ. "
        "Function-matched initialization makes their initial predictions agree. The shared scalar normalization "
        "preserves orthonormal basis comparisons; preprocessing statistics are fitted on training inputs only.", "",
        r"The output remains $\hat r=2\operatorname{sigmoid}(2a)$, and every method minimizes", "",
        "$$", r"L=\tfrac12\overline{\hat r_P^{-1/2}}+\tfrac14\overline{\sqrt{\hat r_P}}"
        r"+\tfrac14\overline{\sqrt{\hat r_Q}}-1.", "$$", "",
        "A common invertible log-ratio transformation preserves the population RDR of the closed compositions. "
        "Fixed additive smoothing is also injective on the closed simplex in exact arithmetic. Closing the "
        "original profiles discards their total retained abundance, so its effect is separated by the closed raw "
        "control. No method is guaranteed to improve finite-sample fitting or calibration. In particular, the "
        "substantial observed-versus-generated zero-frequency difference can affect discrimination.", "",
        "## Epsilon sensitivity", ""])
    if len(sensitivity):
        text.extend(["![Validation epsilon sensitivity](epsilon_sensitivity.png)", "",
            "Only validation metrics are displayed for the fixed reference seed. Additional epsilon values "
            "are sensitivity checks around the prespecified primary choice, not test-driven tuning. "
            "Compare both predictive risk and calibration gaps across epsilon: the zero-handling choice can "
            "substantially affect these fits.", "",
            _table(sensitivity.sort_values(["method", "epsilon"]),
                   ["method", "epsilon", "empirical_bin_absolute_gap", "balanced_brier", "auc"]), ""])
        for method, rows in sensitivity.groupby("method"):
            if rows.epsilon.nunique() > 1:
                text.extend([f"Across the checked epsilon values, {LABELS[method]} has validation Brier "
                    f"{rows.balanced_brier.min():.5f}–{rows.balanced_brier.max():.5f} and absolute gap "
                    f"{rows.empirical_bin_absolute_gap.min():.5f}–{rows.empirical_bin_absolute_gap.max():.5f}.", ""])
    text.extend([("The transformed inputs improve predictive risk here, but they do not resolve the neural "
        "score-versus-frequency discrepancy at intermediate scores. " if improves and unresolved and not preflight else "")
        + "These bin diagnostics cannot establish pointwise accuracy of the individual true RDR.", "",
        "## Training and provenance", "", "![Loss histories](loss_histories.png)", "",
        "The full-batch optimizer, clipping, scheduler, stopping controls, and minimum-validation checkpoint "
        "restoration follow the baseline protocol. The stopping patience can limit the opportunity to benefit "
        "from a subsequent plateau learning-rate reduction. Compare the actual training curves before "
        "attributing a performance difference solely to the representation.", "",
        f"Source fit: `{protocol.get('source_fit', '')}`.", "",
        "`metrics.csv`, `diagnostic_cells.csv`, `loss_history.csv`, and `seed_summary.csv` provide reproducible numeric tables. "
        "`protocol.json` records split provenance, preprocessing, candidate configurations, and selection. "
        "P subjects remain separated across roles; repeated samples within a role and historical preprocessing "
        "retain the preceding experiment's limitations. No independent true-RDR error or coverage rate is measured.", "",
        "### Input audit", ""])
    audit_rows = [{"distribution_split": key, "n": value["shape"][0], "D": value["shape"][1],
                   "zero_fraction": value["zero_fraction"], "row_sum_min": value["row_sum_min"],
                   "row_sum_max": value["row_sum_max"]} for key, value in protocol.get("input_audit", {}).items()]
    if audit_rows:
        text.extend([_table(pd.DataFrame(audit_rows), ["distribution_split", "n", "D", "zero_fraction",
                                                      "row_sum_min", "row_sum_max"]), ""])
    if protocol.get("phylo_basis", {}).get("tree_provenance"):
        text.extend(["Phylogenetic basis provenance: " + protocol["phylo_basis"]["tree_provenance"], ""])
    text.extend(["### Recorded configuration", "", "```json", json.dumps(protocol.get("config", {}), indent=2, default=str),
        "```", "", "### All run metrics", "",
        _table(metrics.sort_values(["method", "seed", "epsilon", "role"]),
               ["run_id", "role", "primary", "epsilon", "n_p", "n_q", *MEASURES, "best_epoch", "epochs_run"]), ""])
    command = protocol.get("reproduction_command") or protocol.get("command")
    if not command:
        config = protocol.get("config", {})
        command_parts = ["python3", "-B", "experiments/AGP/AGP_composition_diagnostics.py",
                         "--output-dir", str(out) + "_rerun"]
        for key in ("data_root", "source_fit", "seeds", "epsilon", "sensitivity_epsilons",
                    "max_epochs", "min_epochs", "patience", "threads"):
            if key in config:
                values = config[key] if isinstance(config[key], list) else [config[key]]
                command_parts.extend(["--" + key.replace("_", "-"), *map(str, values)])
        if config.get("preflight"):
            command_parts.append("--preflight")
        command = shlex.join(command_parts)
    if command:
        text.extend(["## Reproduction", "", "Use a fresh output directory; preserve the saved source fit and "
                     "candidate configuration.", "", "```bash", str(command), "```", ""])
    else:
        text.extend(["## Reproduction", "", "Run the accompanying experiment runner using the source fit and "
                     "configuration recorded in `protocol.json`, with a fresh output directory. The report can "
                     "be regenerated by calling `render_report` with the saved CSV tables and protocol.", ""])
    (out / "report.md").write_text("\n".join(text))
