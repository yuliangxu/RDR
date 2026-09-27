"""Render AGP architecture, validation calibration, and merged-region CI diagnostics.

This module consumes saved results only: it neither fits nor selects a model.
The command-line entry point regenerates a completed run's report and figures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex

import numpy as np
import pandas as pd


LABELS = {
    "baseline_p5": "Baseline, patience 5",
    "baseline_p30": "Baseline, patience 30",
    "wide_p30": "Wide MLP, patience 30",
    "deep_p30": "Deep MLP, patience 30",
    "residual_p30": "Residual MLP, patience 30",
}
COLORS = dict(zip(LABELS, ("#777777", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")))
ROLES = ("train", "validation", "test")
MEASURES = ("empirical_bin_absolute_gap", "empirical_bin_rmse", "balanced_brier", "auc",
            "equal_mixture_mean_score", "midpoint_loss")


def _table(frame, columns, names=None):
    """Render Markdown without tabulate as an additional dependency."""
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


def _run_id(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return value.get("selected_run_id", value.get("run_id"))
    return None


def _selection(protocol, metrics):
    selection = protocol["selection"]
    selected_id = _run_id(selection)
    if selected_id is None:
        selected_id = _run_id(selection.get("global"))
    representatives = {method: _run_id(value)
                       for method, value in selection["by_method"].items()}
    available = set(metrics.run_id)
    if selected_id not in available or not set(representatives.values()).issubset(available):
        raise ValueError("Saved selection refers to absent run IDs")
    return selected_id, representatives


def _score_regions(cells):
    """Decompose gap by a prespecified middle score range, within each fitted run."""
    records = []
    for (run_id, role), rows in cells.groupby(["run_id", "role"], sort=False):
        first = rows.iloc[0]
        middle = rows.left.ge(.2 - 1e-12) & rows.right.le(1.8 + 1e-12)
        # These ranges must be unions of the saved elementary bins.
        for boundary in (.2, 1.8):
            if ((rows.left < boundary - 1e-12) & (rows.right > boundary + 1e-12)).any():
                raise ValueError("Middle-range boundary crosses an elementary score bin")
        mass, gap = rows.mixture_mass, rows.calibration_gap
        if not np.isclose(mass.sum(), 1., rtol=0, atol=1e-10) or gap[mass > 0].isna().any():
            raise ValueError("Invalid cell masses or missing nonempty-cell calibration gaps")
        mid_mass = float(mass[middle].sum())
        mid_abs = float((mass[middle] * gap[middle].abs()).sum())
        mid_signed = float((mass[middle] * gap[middle]).sum())
        end_abs = float((mass[~middle] * gap[~middle].abs()).sum())
        records.append({
            "run_id": run_id, "method": first.method, "seed": first.seed, "role": role,
            "epsilon": first.get("epsilon", 1e-6), "primary": True,
            "middle_left": .2, "middle_right_exclusive": 1.8,
            "middle_mass": mid_mass,
            "middle_conditional_signed_gap": mid_signed / mid_mass if mid_mass else np.nan,
            "middle_conditional_absolute_gap": mid_abs / mid_mass if mid_mass else np.nan,
            "middle_gap_contribution": mid_abs, "endpoint_mass": float(mass[~middle].sum()),
            "endpoint_gap_contribution": end_abs, "all_bin_absolute_gap": mid_abs + end_abs,
        })
    return pd.DataFrame(records)


def _tail_diagnostics(out, metrics):
    """Audit observed test-score tails, without changing the saved model selection."""
    records = []
    test = metrics[metrics.role.eq("test")]
    paths = [out / "runs" / run_id / "test_scores.csv" for run_id in test.run_id]
    if not any(path.exists() for path in paths):
        return pd.DataFrame()
    if not all(path.exists() for path in paths):
        raise ValueError("Partial test-score archive cannot support the complete tail diagnostic")
    for row, path in zip(test.itertuples(index=False), paths):
        scores = pd.read_csv(path, usecols=["distribution", "rdr"], float_precision="round_trip")
        p, q = (scores.loc[scores.distribution.eq(d), "rdr"].to_numpy() for d in ("P", "Q"))
        if len(p) != row.n_p or len(q) != row.n_q or not np.isfinite(np.r_[p, q]).all() or min(p.min(), q.min()) <= 0:
            raise ValueError("Invalid archived test scores or split counts")
        loss = .5*np.mean(p**-.5)+.25*np.mean(np.sqrt(p))+.25*np.mean(np.sqrt(q))-1
        if not np.isclose(loss, row.midpoint_loss, atol=1e-10, rtol=1e-8):
            raise ValueError("Archived test scores do not reproduce the reported Hellinger objective")
        records.append({"run_id": row.run_id, "method": row.method, "seed": row.seed,
            "n_p": len(p), "n_q": len(q), "min_rdr_p": p.min(),
            "p_count_rdr_lt_0_01": int((p < .01).sum()), "p_count_rdr_lt_0_1": int((p < .1).sum()),
            "max_single_p_inverse_sqrt_contribution": .5 / len(p) / np.sqrt(p.min()),
            "p_inverse_sqrt_term": .5*np.mean(p**-.5), "midpoint_loss": loss})
    result = pd.DataFrame(records)
    result.to_csv(out / "tail_diagnostics.csv", index=False)
    return result


def _ci_axis(ax, rows, title, legend=False):
    rows = rows.sort_values("left")
    for row in rows.itertuples(index=False):
        ax.plot([row.left, row.right], [row.estimate, row.estimate], color=".5", lw=1, alpha=.7)
    x = rows.mean_model_score.to_numpy()
    y = rows.estimate.to_numpy()
    # Bars are at the exact mean score; distinct widths show nested intervals.
    for label, color, width in (("C.2", "#88bde6", 6), ("C.1", "#234f78", 2)):
        prefix = label.lower().replace(".", "")
        lower, upper = rows[prefix + "_lower"], rows[prefix + "_upper"]
        ax.vlines(x, lower, upper, color=color, linewidth=width, label=label, alpha=.9)
    ax.scatter(x, y, color="black", s=14, zorder=4, label="Frequency RDR")
    ax.plot([0, 2], [0, 2], ":", color=".4", lw=1)
    ax.set(xlim=(0, 2), ylim=(0, 2), title=title,
           xlabel="Balanced within-region mean neural RDR")
    ax.grid(alpha=.15)
    if legend:
        ax.legend(fontsize=8, loc="lower right")


def render_report(out, metrics, cells, history, protocol, ci_cells, ci_summary):
    """Write numeric summaries, PNG/PDF figures, and report.md from completed fits."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected_id, representatives = _selection(protocol, metrics)
    methods = [method for method in LABELS if method in set(metrics.method)]
    selected = metrics[metrics.run_id.eq(selected_id)]
    selected_test = selected[selected.role.eq("test")].iloc[0]
    representative_ids = set(representatives.values())
    representative_metrics = metrics[metrics.run_id.isin(representative_ids)]
    group = metrics.groupby(["method", "role"], sort=False)
    summary = group[list(MEASURES)].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    seed_counts = group.seed.nunique().rename("n_seeds").reset_index()
    summary = summary.merge(seed_counts, on=["method", "role"], validate="one_to_one")
    summary.to_csv(out / "seed_summary.csv", index=False)

    regions = _score_regions(cells)
    regions.to_csv(out / "score_region_summary.csv", index=False)
    tails = _tail_diagnostics(out, metrics)
    region_columns = ["middle_mass", "middle_conditional_signed_gap", "middle_conditional_absolute_gap",
                      "middle_gap_contribution", "endpoint_mass", "endpoint_gap_contribution",
                      "all_bin_absolute_gap"]
    region_means = regions.groupby(["method", "role"], sort=False)[region_columns].mean().reset_index()
    check = regions.merge(metrics[["run_id", "role", "empirical_bin_absolute_gap"]],
                          on=["run_id", "role"], validate="one_to_one")
    if not np.allclose(check.all_bin_absolute_gap, check.empirical_bin_absolute_gap, atol=1e-10, rtol=1e-8):
        raise ValueError("Region decomposition does not reproduce saved whole-score gap")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True, sharey=True)
    for ax, role in zip(axes, ROLES):
        for method in methods:
            rows = cells[cells.run_id.eq(representatives[method]) & cells.role.eq(role)]
            rows = rows[rows.mixture_mass.gt(0)].sort_values("bin_index")
            ax.plot(rows.mean_model_score, rows.cell_rdr, "o-", ms=3, lw=1.1,
                    color=COLORS[method], label=LABELS[method])
        ax.plot([0, 2], [0, 2], ":", color=".4", lw=1)
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel="Balanced within-bin mean neural RDR", title=role.title())
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Frequency RDR from normalized P/Q counts")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("AGP architecture diagnostics: each method's validation-selected seed")
    _save(fig, out, "score_vs_frequency", plt)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharex=True, sharey=True)
    positive_mass = cells[cells.run_id.isin(representative_ids) & cells.mixture_mass.gt(0)].mixture_mass
    for ax, role in zip(axes, ROLES):
        for method in methods:
            rows = cells[cells.run_id.eq(representatives[method]) & cells.role.eq(role)].sort_values("bin_index")
            edges = np.r_[rows.left, rows.right.iloc[-1]]
            mass = rows.mixture_mass.to_numpy()
            ax.stairs(np.where(mass > 0, mass, np.nan), edges, color=COLORS[method], label=LABELS[method])
        ax.axvspan(.2, 1.8, color=".8", alpha=.2)
        ax.set(xlim=(0, 2), yscale="log", ylim=(positive_mass.min() / 2, 1),
               xlabel="Neural RDR score", title=role.title())
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Balanced mixture mass per score bin (log)")
    axes[1].legend(fontsize=8, loc="lower center")
    fig.suptitle("Score-bin mass; shaded range [0.2, 1.8)")
    _save(fig, out, "score_bin_mass", plt)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    selected_ci = ci_cells[ci_cells.run_id.eq(selected_id)].sort_values("left")
    if selected_ci.empty:
        raise ValueError("Selected run has no CI results")
    _ci_axis(axes[0], selected_ci, "Selected model: test C.1 / C.2", legend=True)
    axes[0].set_ylabel("Cell-average frequency RDR and confidence interval")
    vrows = cells[cells.run_id.eq(selected_id) & cells.role.eq("validation") & cells.mixture_mass.gt(0)]
    trows = cells[cells.run_id.eq(selected_id) & cells.role.eq("test") & cells.mixture_mass.gt(0)]
    for rows, label, color in ((vrows, "Validation (descriptive)", "#e08214"),
                               (trows, "Test (descriptive)", "#1f78b4")):
        axes[1].plot(rows.mean_model_score, rows.cell_rdr, "o-", ms=3, lw=1, color=color, label=label)
    axes[1].plot([0, 2], [0, 2], ":", color=".4", lw=1)
    axes[1].set(xlim=(0, 2), ylim=(0, 2), title="Selected model: elementary-bin diagnostics",
                xlabel="Balanced within-bin mean neural RDR", ylabel="Frequency RDR")
    axes[1].legend(fontsize=8, loc="lower right")
    axes[1].grid(alpha=.15)
    fig.suptitle(f"Validation-selected model: {selected_id}")
    _save(fig, out, "selected_model_ci", plt)

    elementary_ci = None
    candidate_path = out / "ci_candidates.csv"
    if candidate_path.exists():
        candidates = pd.read_csv(candidate_path)
        elementary_ci = candidates[candidates.run_id.eq(selected_id) & (candidates.stop-candidates.start).eq(1)].copy()
        descriptive = cells[cells.run_id.eq(selected_id) & cells.role.eq("test")]
        elementary_ci = elementary_ci.drop(columns=["mean_model_score", "mixture_mass"], errors="ignore").merge(
            descriptive[["bin_index", "mean_model_score", "mixture_mass"]],
            left_on="start", right_on="bin_index", validate="one_to_one")
        if len(elementary_ci) != len(descriptive):
            raise ValueError("Candidate CI table does not contain all elementary score bins")
        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        candidate_count = len(descriptive) * (len(descriptive) + 1) // 2
        _ci_axis(ax, elementary_ci[elementary_ci.mixture_mass.gt(0)],
                 f"Elementary score bins: same {candidate_count}-candidate correction", legend=True)
        ax.set_ylabel("Cell-average RDR and confidence interval")
        fig.suptitle(f"Validation-selected model: {selected_id}")
        _save(fig, out, "selected_model_elementary_ci", plt)

    ncols = 3
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.4 * nrows), squeeze=False)
    for ax, method in zip(axes.flat, methods):
        rows = ci_cells[ci_cells.run_id.eq(representatives[method])]
        seed = metrics[metrics.run_id.eq(representatives[method])].seed.iloc[0]
        _ci_axis(ax, rows, f"{LABELS[method]}\nseed {seed}", legend=True)
        ax.set_ylabel("Cell-average RDR")
    for ax in list(axes.flat)[len(methods):]:
        ax.axis("off")
    fig.suptitle("Test merged-region intervals: validation-selected seed within each method")
    _save(fig, out, "architecture_ci_comparison", plt)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for method in methods:
        rows = history[history.run_id.eq(representatives[method])].sort_values("epoch")
        for ax, measure in zip(axes.flat, ("train_loss", "validation_loss", "learning_rate", "gradient_norm")):
            ax.plot(rows.epoch, rows[measure], color=COLORS[method], lw=1.1, label=LABELS[method])
    for ax, title, ylabel in zip(axes.flat,
                                ("Training objective", "Validation objective", "Learning-rate schedule", "Gradient norm"),
                                ("Hellinger objective", "Hellinger objective", "Learning rate", "Recorded gradient norm")):
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.grid(alpha=.2)
    axes[1, 0].set_yscale("log")
    axes[0, 0].legend(fontsize=8)
    _save(fig, out, "training_histories", plt)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for column, role in enumerate(("validation", "test")):
        for row, measure in enumerate(("empirical_bin_absolute_gap", "balanced_brier")):
            ax = axes[row, column]
            for i, method in enumerate(methods):
                values = metrics[metrics.method.eq(method) & metrics.role.eq(role)][measure].to_numpy()
                ax.scatter(np.full(len(values), i), values, color=COLORS[method], s=23, alpha=.7)
                ax.plot([i-.2, i+.2], [values.mean()]*2, color="black", lw=1.5)
            ax.set_xticks(range(len(methods)), [LABELS[m] for m in methods], rotation=28, ha="right")
            ax.set_ylabel("Weighted absolute RDR gap" if row == 0 else "Balanced Brier score")
            if row == 0:
                ax.set_title(role.title())
            ax.grid(axis="y", alpha=.2)
    fig.suptitle("Individual training seeds and their means; lower values are better")
    _save(fig, out, "seed_comparison", plt)

    counts = metrics.groupby("role", sort=False)[["n_p", "n_q"]].first().reset_index()
    architecture_columns = ["method", "parameter_count", "patience"]
    architecture_table = metrics[architecture_columns].drop_duplicates()
    selected_summary = ci_summary[ci_summary.run_id.eq(selected_id)].iloc[0]
    above_c1 = int(selected_summary.c1_regions_entirely_above_scores)
    above_c2 = int(selected_summary.c2_regions_entirely_above_scores)
    selected_region = regions[regions.run_id.eq(selected_id) & regions.role.eq("test")].iloc[0]
    preflight = protocol.get("config", {}).get("preflight", False)
    result_intro = ("**Implementation preflight only. These shortened fits are not a model-performance conclusion.**"
                    if preflight else
                    f"**Validation selected `{selected_id}`.** Its retrospective test weighted absolute gap is "
                    f"{selected_test.empirical_bin_absolute_gap:.5f}, balanced Brier score is "
                    f"{selected_test.balanced_brier:.5f}, and AUC is {selected_test.auc:.5f}. "
                    f"The middle-score range contains {selected_region.middle_mass:.2%} of balanced test mass; "
                    f"its conditional signed gap is {selected_region.middle_conditional_signed_gap:.5f} and "
                    f"conditional absolute gap is {selected_region.middle_conditional_absolute_gap:.5f} RDR units.")
    result_ci = (f"For the selected model, {above_c1} merged regions have a C.1 lower endpoint above the "
                 f"region's upper score cutoff, and {above_c2} have this property under C.2. ")
    result_ci += ("These preflight intervals only exercise the reporting implementation. " if preflight else
                  "The regional underprediction diagnostic therefore persists for this fit. "
                  if above_c1 or above_c2 else
                  "No merged region meets that stronger underprediction criterion; this does not establish "
                  "calibration or individual-RDR accuracy. ")
    result_ci += ("These intervals concern cell-average RDR, with the retrospective sampling limitations below.")
    ci_config = protocol.get("ci_config", {})
    alpha = ci_config.get("alpha", .05)
    confidence = 100 * (1 - alpha)
    n_elementary = int(cells.groupby(["run_id", "role"]).size().iloc[0])
    n_candidates = n_elementary * (n_elementary + 1) // 2
    selected_ci_table = selected_ci.copy()
    selected_ci_table["c1_entirely_above_scores"] = selected_ci_table.c1_lower > selected_ci_table.right
    selected_ci_table["c2_entirely_above_scores"] = selected_ci_table.c2_lower > selected_ci_table.right

    collapsed_ids = []
    for run_id, rows in ci_cells.groupby("run_id"):
        if (len(rows) == 1 and np.isclose(rows.left.iloc[0], 0) and np.isclose(rows.right.iloc[0], 2)
                and np.allclose(rows[["estimate", "c1_lower", "c1_upper", "c2_lower", "c2_upper"]], 1)):
            collapsed_ids.append(run_id)
    if len(collapsed_ids) == metrics.run_id.nunique():
        result_ci = (f"**The h={ci_config.get('min_count', 40)} merged CIs have no regional resolution in this run.** "
            f"All {len(collapsed_ids)} fits collapse to the single score region [0,2], whose population "
            "cell-average RDR is structurally known to equal 1. Thus C.1 and C.2 return [1,1]; this is "
            "the known global average, not evidence that the neural scores are calibrated. The original "
            f"{n_elementary}-bin candidate bands retain score resolution, but middle-score bins have few "
            "observations and correspondingly wide intervals. The elementary-bin figure and validation "
            "frequency diagnostics are essential for interpreting the remaining regional discrepancy.")

    tradeoff_sections = []
    region_comparison_sections = []
    if not preflight and {"baseline_p5", "deep_p30", "residual_p30"}.issubset(set(region_means.method)):
        regional = region_means.set_index(["method", "role"])
        baseline_val = regional.loc["baseline_p5", "validation"]
        baseline_test = regional.loc["baseline_p5", "test"]
        residual_val = regional.loc["residual_p30", "validation"]
        residual_test = regional.loc["residual_p30", "test"]
        deep_test = regional.loc["deep_p30", "test"]
        region_comparison_sections = [
            "**The regional-calibration comparison differs from the Brier ranking.** Across training seeds, "
            f"the residual MLP's conditional absolute middle-score gap is {residual_val.middle_conditional_absolute_gap:.6f} "
            f"on validation and {residual_test.middle_conditional_absolute_gap:.6f} on test, compared with "
            f"{baseline_val.middle_conditional_absolute_gap:.6f} and {baseline_test.middle_conditional_absolute_gap:.6f} "
            f"for `baseline_p5`. Its mean test middle mass is {residual_test.middle_mass:.6f}, versus "
            f"{baseline_test.middle_mass:.6f} for the baseline. The deep MLP's mean test conditional "
            f"absolute middle gap is {deep_test.middle_conditional_absolute_gap:.6f}, with middle mass "
            f"{deep_test.middle_mass:.6f}. These are descriptive comparisons of model-dependent score "
            "regions, not a common fixed population subset or a new model-selection rule. The saved "
            "validation-Brier winner remains unchanged.", "",
        ]
    baseline_id = representatives.get("baseline_p5")
    baseline = metrics[metrics.run_id.eq(baseline_id) & metrics.role.eq("test")]
    if not preflight and len(baseline):
        baseline = baseline.iloc[0]
        selected_method_mean = summary[summary.method.eq(selected_test.method) & summary.role.eq("test")].iloc[0]
        baseline_method_mean = summary[summary.method.eq("baseline_p5") & summary.role.eq("test")].iloc[0]
        tradeoff_sections = [
            "**Brier selection and Hellinger risk expose different behavior.** The selected run's test "
            f"Hellinger objective is {selected_test.midpoint_loss:.6f}, compared with {baseline.midpoint_loss:.6f} "
            f"for the validation-selected baseline `{baseline_id}`; **lower is better**. Across all seeds, "
            f"the `{selected_test.method}` test mean is {selected_method_mean.midpoint_loss_mean:.6f}, versus "
            f"{baseline_method_mean.midpoint_loss_mean:.6f} for `baseline_p5`. The "
            r"$\tfrac12\overline{\hat r_P^{-1/2}}$ term is highly sensitive to very small scores on observed P profiles. "
            "Brier gains alone therefore do not establish improvement under the training objective. "
            "The validation-Brier selection remains locked; these retrospective test diagnostics do not "
            "select a replacement winner.", "",
        ]
        if len(tails):
            selected_tail = tails[tails.run_id.eq(selected_id)].iloc[0]
            tradeoff_sections += [
                f"The smallest selected-model score among {int(selected_tail.n_p):,} test P observations is "
                f"**{selected_tail.min_rdr_p:.12g}**. That single observation contributes "
                f"**{selected_tail.max_single_p_inverse_sqrt_contribution:.6f}** to "
                r"$\tfrac12\overline{\hat r_P^{-1/2}}$ (its contribution is "
                r"$0.5/[n_P\sqrt{\hat r_P}]$). These are exact diagnostics of the archived observed scores "
                "and empirical loss; the true individual RDR is unavailable, so this is not a measured "
                "pointwise RDR error.", "",
                _table(tails[tails.run_id.isin((selected_id, baseline_id))],
                       ["run_id", "n_p", "n_q", "min_rdr_p", "p_count_rdr_lt_0_01", "p_count_rdr_lt_0_1",
                        "max_single_p_inverse_sqrt_contribution", "midpoint_loss"],
                       ["Run", "P count", "Q count", "Minimum P score", "P scores < 0.01", "P scores < 0.1",
                        "Largest single inverse-root loss contribution", "Test Hellinger objective"]), "",
                "[tail_diagnostics.csv](tail_diagnostics.csv) records these quantities for every fit. "
                "This tail audit was added after inspecting the completed results; it changes no fit, "
                "checkpoint, score, CI, or selection.", "",
            ]

    sections = [
        "# AGP architecture comparison with validation RDR and confidence intervals", "",
        result_intro, "", *region_comparison_sections, result_ci, "", *tradeoff_sections,
        "This experiment fixes the smoothed ILR representation, archived P/Q data and split assignments, "
        "Hellinger objective, bounded output, and optimizer. It compares the original narrow MLP, a "
        "longer-patience narrow control, a wider MLP, a deep MLP, and a residual MLP. The test cohort "
        "was examined in preceding AGP analyses; this is a retrospective diagnostic comparison.", "",
        "## Selection and sample accounting", "",
        f"Completed fits: **{metrics.run_id.nunique()}**, across **{len(methods)} methods** and "
        f"**{metrics.seed.nunique()} training seeds**. The checkpoint in each fit is selected by validation "
        "Hellinger loss. The saved overall run and the representative seed within each method are selected "
        "by validation balanced Brier score. Test metrics and CIs are diagnostic outputs and do not enter "
        "this selection. Validation scores are consequently development diagnostics, not independent "
        "post-selection evidence.", "", _table(counts, ["role", "n_p", "n_q"], ["Role", "P observations", "Q observations"]), "",
        _table(architecture_table, architecture_columns, ["Method", "Trainable parameters", "Stopping patience"]), "",
        "The patience-5 versus patience-30 narrow models separate stopping behavior from capacity. Wide, "
        "deep, and residual models use the same longer stopping patience; deep versus residual isolates "
        "the skip connections for the recorded depth and width. Training curves below show whether "
        "learning-rate reductions had time to affect fitting.", "",
        "Recorded architecture definitions:", "", "```json", json.dumps(protocol.get("architectures", {}), indent=2, default=str), "```", "",
        "## Validation neural score versus frequency RDR", "",
        "![Score versus frequency](score_vs_frequency.png)", "",
        "Each panel shows the validation-selected seed within each method. All elementary score cutoffs "
        "are fixed on [0,2], but the corresponding input-space cells depend on the fitted network. A point "
        "above the diagonal means the frequency RDR exceeds the mean neural score in that cell. Small "
        "populated cells can have noisy frequency estimates, and empty cells are omitted. No validation "
        "CIs are reported because validation influenced stopping and model selection.", "",
        _table(representative_metrics[representative_metrics.role.eq("validation")],
               ["method", "run_id", "empirical_bin_absolute_gap", "balanced_brier", "auc", "best_epoch", "epochs_run"],
               ["Method", "Representative run", "Weighted absolute gap", "Brier", "AUC", "Best epoch", "Epochs run"]), "",
        "For each split, write n and m for its P/Q sizes, and let a cell be "
        r"$A_j=\{z:\hat r(z)\in I_j\}$. Its empirical masses are $\hat p_j=k_{Pj}/n$, $\hat q_j=k_{Qj}/m$; "
        "normalization is separate for P and Q despite unequal sample sizes:", "",
        "$$", r"x_j=\frac{n^{-1}\sum_i\hat r(X_i)1_{A_j}(X_i)+m^{-1}\sum_\ell\hat r(Y_\ell)1_{A_j}(Y_\ell)}{\hat p_j+\hat q_j},"
        r"\quad y_j=\frac{2\hat p_j}{\hat p_j+\hat q_j},\quad w_j=\frac{\hat p_j+\hat q_j}{2}.", "$$", "",
        r"The signed gap is $y_j-x_j$; positive values indicate regional neural underprediction. "
        r"The weighted absolute gap is $G=\sum_jw_j|y_j-x_j|$, and the RMSE is "
        r"$\sqrt{\sum_jw_j(y_j-x_j)^2}$. These are empirical diagnostics in RDR units, not "
        "unbiased estimates of population calibration error.", "",
        r"Balanced Brier is $\tfrac12\overline{(1-\hat r_P/2)^2}+\tfrac12\overline{(\hat r_Q/2)^2}$. "
        "A constant RDR of 1 has zero balanced empirical gap, Brier 0.25, and AUC 0.5. Brier and AUC "
        "therefore accompany calibration gaps. The equal-mixture mean score should be 1 for the true RDR.", "",
        "## Test confidence intervals", "", "![Selected model CIs](selected_model_ci.png)", "",
        "![Architecture CI comparison](architecture_ci_comparison.png)", "",
        "Black points are empirical cell-average RDR estimates. Thin dark bars are C.1; wide pale bars are "
        "C.2. Horizontal gray segments show the score interval defining each merged region. The CIs "
        "target the population cell average "
        r"$\theta_A=2P(A)/(P(A)+Q(A))=E_{(P+Q)/2}[r_0(Z)\mid Z\in A]$, "
        "not the individual true RDR at each profile. Horizontal means are empirical neural-score "
        "summaries and have no uncertainty bars. Comparing a CI with the whole score interval avoids "
        "treating that empirical mean as fixed truth.", "",
        f"The nominal confidence level is {confidence:g}% **within each fitted model**, with all "
        f"{n_candidates} contiguous unions of {n_elementary} elementary bins protected before the "
        "count-based merging step. C.1 uses the joint Gaussian-multiplier maximum and is asymptotic. "
        "C.2 uses Bonferroni-adjusted Clopper–Pearson probability bounds; under independent IID P and Q "
        "sampling its finite-sample simultaneous guarantee survives the adaptive merging. The correction "
        f"is not a joint correction across the {metrics.run_id.nunique()} networks. A method/seed selected "
        "by inspecting these test intervals would need an additional selection adjustment.", "",
        "Here P includes repeated observations within subjects, and the reused cohort and historical "
        "protocol have already influenced the research choices. These are **nominal observation-level "
        "retrospective intervals**; neither IID coverage for this cohort nor independent final-test "
        "confirmation is asserted. Absence of a significant region can reflect interval width or merging "
        "and is not evidence of accurate pointwise RDR.", "",
        "Selected model, all merged regions:", "",
        _table(selected_ci_table, ["left", "right", "k_p", "k_q", "mixture_mass", "mean_model_score", "estimate",
                                  "c1_lower", "c1_upper", "c2_lower", "c2_upper", "c1_entirely_above_scores", "c2_entirely_above_scores"],
               ["Left", "Right", "P count", "Q count", "Mixture mass", "Mean neural RDR", "Frequency RDR",
                "C.1 lower", "C.1 upper", "C.2 lower", "C.2 upper", "C.1 > whole score region", "C.2 > whole score region"]), "",
        "The flags require the lower endpoint to exceed the region's upper score cutoff. This supports "
        "a discrepancy between its population cell-average RDR and all neural scores in that region; "
        "it does not show that every individual true RDR is underestimated. Counts of flagged regions "
        "are descriptive because networks induce different regions and merging patterns.", "",
        _table(ci_summary, ["run_id", "n_regions", "c1_mean_width", "c2_mean_width",
                            "c1_regions_entirely_above_scores", "c2_regions_entirely_above_scores"]), "",
        "## Score mass and middle-score gaps", "", "![Score-bin mass](score_bin_mass.png)", "",
        "The range [0.2,1.8) was fixed before this architecture comparison from the preceding diagnostics. "
        "For each run, its conditional signed and absolute middle gaps average elementary-bin gaps only "
        "over mass within that range. Positive signed gaps indicate net underprediction. Opposing gaps "
        "can cancel in the signed measure, so the absolute measure is also reported. The same numeric "
        "score range generally contains different profiles for different networks.", "",
        _table(region_means[region_means.role.isin(("validation", "test"))],
               ["method", "role", "middle_mass", "middle_conditional_signed_gap", "middle_conditional_absolute_gap", "all_bin_absolute_gap"],
               ["Method", "Role", "Middle mass", "Conditional signed gap", "Conditional absolute gap", "Overall weighted gap"]), "",
        "These are means of each run's quantities. Conditional gaps are undefined for an empty middle "
        "range. A smaller overall gap may coincide with moving mass toward 0 and 2 while retaining large "
        "middle-score errors. `score_region_summary.csv` records the complete per-run decomposition; "
        "each run's middle and endpoint contributions sum to its overall gap.", "",
        "## Variation across training seeds", "", "![Seed comparison](seed_comparison.png)", "",
        "Dots show each fit and black horizontal marks their means. The training seeds share data and "
        "splits: their standard deviations describe fitting variation and are not sampling standard errors "
        "or confidence intervals. Mean results include every completed seed, rather than only the "
        "validation-selected representative.", "",
        _table(summary[summary.role.isin(("validation", "test"))],
               ["method", "role", "n_seeds", "empirical_bin_absolute_gap_mean", "empirical_bin_absolute_gap_std",
                "balanced_brier_mean", "balanced_brier_std", "midpoint_loss_mean", "midpoint_loss_std",
                "auc_mean", "equal_mixture_mean_score_mean"],
               ["Method", "Role", "Seeds", "Gap mean", "Gap SD", "Brier mean", "Brier SD", "Hellinger mean",
                "Hellinger SD", "AUC mean", "Mixture mean score"]), "",
        "Lower Hellinger objective values are better. Its inverse-root P-score term can reveal extreme "
        "observed-score tails that contribute little to an average Brier score or a weighted calibration gap.", "",
        "## Training, target, and reproducibility", "", "![Training histories](training_histories.png)", "",
        r"The bounded output is $\hat r=2\operatorname{sigmoid}(2a)$. The unchanged objective is", "",
        "$$", r"L=\tfrac12\overline{\hat r_P^{-1/2}}+\tfrac14\overline{\sqrt{\hat r_P}}"
        r"+\tfrac14\overline{\sqrt{\hat r_Q}}-1.", "$$", "",
        "Each P/Q expectation is normalized by that distribution's own training count. This objective "
        r"targets $r_0=2p/(p+q)$ on the shared transformed input space. Closure, additive smoothing, "
        "the ILR basis, and training-only normalization follow the preceding composition experiment. "
        "The result is the RDR of the retained compositions; closure discards total retained abundance. "
        "All architecture comparisons hold this representation and its epsilon fixed.", "",
        f"Source fit: `{protocol.get('source_fit', '')}`. Source composition experiment: "
        f"`{protocol.get('source_composition', '')}`.", "",
        "`metrics.csv`, `diagnostic_cells.csv`, `loss_history.csv`, `seed_summary.csv`, and "
        "`score_region_summary.csv` contain the reproducible diagnostics. The CI tables retain region "
        "counts, endpoints, and run IDs; `protocol.json` freezes the fitting configuration and provenance. "
        "`report_protocol.json` and `selection.json` add the completed validation selection and model hashes. "
        "PNG and PDF copies of each figure are saved next to this report. Saved source snapshots and "
        "checkpoints should be retained when rerunning or extending the experiment.", "",
        "### Recorded limitations", "", "```json", json.dumps(protocol.get("limitations", []), indent=2, default=str), "```", "",
        "### CI configuration", "", "```json", json.dumps(ci_config, indent=2, default=str), "```", "",
        "### Optimizer and objective", "", "```json", json.dumps(protocol.get("training", {}), indent=2, default=str), "```", "",
        "### Training configuration", "", "```json", json.dumps(protocol.get("config", {}), indent=2, default=str), "```", "",
        "### All fit metrics", "", _table(metrics, ["run_id", "role", "n_p", "n_q", *MEASURES, "best_epoch", "epochs_run"]), "",
    ]
    if elementary_ci is not None:
        ci_index = sections.index("![Architecture CI comparison](architecture_ci_comparison.png)") + 2
        sections[ci_index:ci_index] = [
            "![Selected-model elementary-bin intervals](selected_model_elementary_ci.png)", "",
            f"This additional plot retains the original {n_elementary} elementary-bin resolution for the "
            f"selected model. It uses the **same {n_candidates}-candidate adjusted intervals** computed "
            "before merging, so it introduces no additional inferential family within that model. "
            "Only populated bins are plotted. Sparse bins can have wide intervals, and C.1 falls back "
            "to [0,2] when its variance estimate is unavailable. This plot reveals fine-score "
            "uncertainty that count-based merging can conceal; the merged regions remain the primary CI "
            "diagnostic. `ci_candidates.csv` preserves the full candidate family for every fitted model.", "",
        ]
    command = protocol.get("reproduction_command") or protocol.get("command")
    if command:
        sections.extend(["### Reproduction command", "", "Use a fresh output directory and the recorded source snapshot.",
                         "", "```bash", str(command), "```", ""])
    renderer = Path(__file__).resolve()
    sections.extend(["### Regenerate this report", "", "```bash",
                     shlex.join(["python3", "-B", str(renderer), "--output-dir", str(out)]),
                     "```", ""])
    (out / "report.md").write_text("\n".join(sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    protocol_path = out / "report_protocol.json"
    if not protocol_path.exists():
        protocol_path = out / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    frames = [pd.read_csv(out / filename) for filename in
              ("metrics.csv", "diagnostic_cells.csv", "loss_history.csv", "ci_cells.csv", "ci_summary.csv")]
    render_report(out, frames[0], frames[1], frames[2], protocol, frames[3], frames[4])


if __name__ == "__main__":
    main()
