"""Compact tables and curves for a verified, complete four-loss convergence run.

The caller verifies frozen sources, fit hashes, and matched random streams.
``render`` additionally requires every case/repetition/sample-size/loss cell,
never overwrites a partial presentation, and returns hashes for its artifacts.
It does not fit models, change the selection, or write a completion marker.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path

import numpy as np
from experiments.simulations import report as paper
from experiments.simulations import workflow as w
from experiments.simulations.population import SETTINGS

LOSSES = ("hellinger", "kl", "chisq", "js")
ERRORS = ("evaluation_mse", "evaluation_brier", "local_gap")
LABELS = dict(zip(paper.METRICS, paper.LABELS))
FILES = ("metrics.csv", "calibration.csv", "summary.csv", "aggregation.csv",
         "paired_comparisons.csv", "report.md", "report.html", "appendix.md",
         "appendix.html", "tables.tex", "appendix_tables.tex")


def identity(row):
    return tuple(row[key] for key in ("case", "repeat", "train_n", "candidate"))


def joined_rows(protocol, rows, local):
    """Check the complete design and join each fit to its fixed-bin diagnostic."""
    cfg, candidates = protocol["config"], protocol["candidates"]
    if (protocol["stage"] != "convergence" or len(candidates) != len(LOSSES)
            or {candidate["loss"] for candidate in candidates} != set(LOSSES)
            or len({(c["architecture"], c["output_alpha"]) for c in candidates}) != 1):
        raise ValueError("Expected four losses at one fixed convergence architecture/activation")
    expected = {(case, repeat, n, w.candidate_id(candidate))
                for case in cfg["cases"] for repeat in range(cfg["convergence_repeats"])
                for n in cfg["sample_sizes"] for candidate in candidates}
    lookup = {identity(row): row for row in rows}
    fixed = [row for row in local if row["partition"] == "fixed"]
    local_lookup = {identity(row): row for row in fixed}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("Convergence presentation requires every fit exactly once")
    if len(local_lookup) != len(fixed) or set(local_lookup) != expected:
        raise ValueError("Convergence presentation requires every fixed-bin diagnostic exactly once")
    if any(identity(row) not in expected for row in local):
        raise ValueError("Unexpected calibration diagnostic outside the convergence design")
    by_name = {w.candidate_id(candidate): candidate for candidate in candidates}
    joined = []
    for key in sorted(expected):
        row, diagnostic = dict(lookup[key]), local_lookup[key]
        if any(row[name] != value for name, value in by_name[row["candidate"]].items()):
            raise ValueError("Convergence row factors disagree with its candidate")
        row.update(local_gap=diagnostic["mass_weighted_absolute_gap"],
                   c2_width=diagnostic["mass_weighted_c2_width"],
                   estimable_mass=diagnostic["evaluation_mass_with_estimable_gap"])
        if any(row[name] is not None and not np.isfinite(row[name]) for name in paper.METRICS):
            raise ValueError("Nonfinite convergence metric")
        joined.append(row)
    return joined


def summarize(rows, cases, candidates, sizes):
    """Keep sample sizes separate and fixed settings equally weighted."""
    individual, aggregate = [], []
    for n in sizes:
        summaries = paper.summarize([row for row in rows if row["train_n"] == n], cases, candidates)
        # The shared helper labels a one-setting aggregate as that setting.
        # Explicitly label this block so small executable smoke runs remain valid.
        for row in summaries:
            row["train_n"] = n
        for row in summaries[-len(candidates):]:
            row.update(case="equal_setting_average", dimension=None, noise_sd=None)
        individual.extend(summaries[:-len(candidates)])
        aggregate.extend(summaries[-len(candidates):])
    return individual, aggregate


def paired_results(rows, cases, sizes):
    """Monte Carlo errors of matched differences, never independent-model SEs."""
    lookup = {(r["case"], r["repeat"], r["train_n"], r["loss"]): r for r in rows}
    repeats = sorted({row["repeat"] for row in rows})
    output = []
    for group in [[case] for case in cases] + [cases]:
        for n in sizes:
            for loss in LOSSES[1:]:
                dimension, noise = SETTINGS[group[0]] if len(group) == 1 else (None, None)
                record = {"case": group[0] if len(group) == 1 else "equal_setting_average",
                          "dimension": dimension, "noise_sd": noise, "train_n": n,
                          "alternative_loss": loss, "comparison": f"Hellinger minus {paper.LOSS[loss]}"}
                for metric in paper.METRICS:
                    values = []
                    for case in group:
                        pairs = [(lookup[(case, repeat, n, "hellinger")][metric],
                                  lookup[(case, repeat, n, loss)][metric]) for repeat in repeats]
                        values.append([left-right for left, right in pairs if left is not None and right is not None])
                    mean, se, count = paper.statistics(values)
                    record.update({f"{metric}_mean": mean, f"{metric}_se": se, f"{metric}_count": count})
                output.append(record)
    # Preserve an unambiguous aggregate in a one-setting smoke design as well.
    for record in output[-len(sizes)*len(LOSSES[1:]):]:
        record.update(case="equal_setting_average", dimension=None, noise_sd=None)
    return output


def metric_table(title, values, sizes, metric, caption, rank=True):
    lookup = {(row["train_n"], row["loss"]): row for row in values}
    rows = []
    for n in sizes:
        current = [lookup[(n, loss)] for loss in LOSSES]
        finite = [row[f"{metric}_mean"] for row in current if row[f"{metric}_mean"] is not None]
        best = min(finite) if finite else None
        cells = [f"{n:,}"]
        for row in current:
            value = row[f"{metric}_mean"]
            bold = rank and best is not None and value is not None and np.isclose(value, best, rtol=0, atol=1e-12)
            cells.append(paper.number(row, metric, bold=bold))
        rows.append((cells, False))
    emphasis = (" Bold marks the lowest observed mean at each sample size (ties included), without a significance claim."
                if rank else " This contextual diagnostic has no winner ranking.")
    return paper.table(title, caption + emphasis, ["Train n per P/Q", *[paper.LOSS[loss] for loss in LOSSES]], rows)


def curve_figures(output, cases, sizes, individual, aggregate):
    """Plot means with one-MCSE ribbons; use linear metric axes and log sample size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    colors = {"hellinger": "#0072B2", "kl": "#D55E00", "chisq": "#009E73", "js": "#CC79A7"}
    styles = {"hellinger": "o", "kl": "s", "chisq": "^", "js": "D"}

    def panel(ax, values, metric):
        lookup = {(row["train_n"], row["loss"]): row for row in values}
        for loss in LOSSES:
            mean = np.array([lookup[(n, loss)][f"{metric}_mean"] for n in sizes], dtype=float)
            se = np.array([lookup[(n, loss)][f"{metric}_se"] for n in sizes], dtype=float)
            ax.plot(sizes, mean, color=colors[loss], marker=styles[loss], ms=4, lw=1.5, label=paper.LOSS[loss])
            ax.fill_between(sizes, mean-se, mean+se, color=colors[loss], alpha=.12, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks(sizes, labels=[f"{n:,}" for n in sizes], rotation=30)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_ylabel(LABELS[metric])
        ax.set_xlabel("Training n per distribution")
        ax.grid(alpha=.22)
        ax.spines[["right", "top"]].set_visible(False)

    paths = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        for name, groups in (("convergence_overall", [("Equal-setting average", aggregate)]),
                             ("convergence_by_setting", [(f"D={SETTINGS[case][0]}, noise SD={SETTINGS[case][1]:.2f}",
                               [row for row in individual if row["case"] == case]) for case in cases])):
            fig, axes = plt.subplots(len(groups), len(ERRORS), figsize=(12, 3.35*len(groups)), squeeze=False)
            for row_index, (label, values) in enumerate(groups):
                for column, metric in enumerate(ERRORS):
                    panel(axes[row_index, column], values, metric)
                    axes[row_index, column].set_title(label)
            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
            fig.tight_layout(rect=(0, 0, 1, 1-.13/len(groups)))
            relative = Path("figures") / f"{name}.png"
            fig.savefig(output/relative, dpi=200, facecolor="white")
            fig.savefig((output/relative).with_suffix(".pdf"), facecolor="white",
                        metadata={"CreationDate": None, "ModDate": None})
            plt.close(fig)
            paths.append(relative.as_posix())
    return paths


def local_gap_definition(path, cfg, rows):
    """State the estimand, unsupported-bin rule, and fixed calibration budget."""
    minimum = min(row["estimable_mass"] for row in rows)
    support = ("Supported evaluation mass was 100% for every fit, so the denominator was 1."
               if np.isclose(minimum, 1, rtol=0, atol=1e-12) else
               f"The minimum supported evaluation mass across fits was {100*minimum:.6f}%; each fit is normalized over its own supported mass.")
    text = (f"For each frozen network, 20 equal-width score bins on [0,2] define input regions A_j. "
            f"Using {cfg['calibration_n']:,} independent calibration observations from each of P and Q, "
            "the count estimate is theta_hat_j = 2 k_Pj / (k_Pj + k_Qj). It estimates "
            "E_M[r_0(X) | X in A_j], where M = (P+Q)/2 and r_0 = dP/dM. "
            f"On a separate evaluation sample with {cfg['evaluation_n']:,} observations per distribution, "
            "r_bar_j is the mean network score and w_j is the fraction of evaluation observations in A_j. "
            "J contains bins with both calibration and evaluation observations.")
    interpretation = ("The gap uses predictions and P/Q labels without analytic truth; CI endpoints do not enter it. "
                      "Calibration and evaluation sampling noise contribute to this binned discrepancy. Their sample sizes "
                      "remain fixed as training n increases, so the measured local gap need not approach zero. "
                      "Score cutoffs are fixed, but input regions depend on the fitted network. This is not pointwise RDR error.")
    if path.suffix == ".md":
        block = ("\n\n### Definition of the local gap\n\n" + text + "\n\n"
                 + r"$$\operatorname{Local\ gap}=\frac{\sum_{j\in J}w_j|\bar r_j-\hat\theta_j|}{\sum_{j\in J}w_j},"
                 + r"\qquad \hat\theta_j=\frac{2k_{Pj}}{k_{Pj}+k_{Qj}},\qquad w_j=\frac{N_j}{2n_{\mathrm{eval}}}.$$"
                 + "\n\n" + support + "\n\n" + interpretation + "\n")
        path.write_text(path.read_text() + block)
    else:
        block = ("<section><h2>Definition of the local gap</h2><p>" + html.escape(text)
                 + "</p><p><strong>Local gap = [∑<sub>j ∈ J</sub> w<sub>j</sub> |r̄<sub>j</sub> − θ̂<sub>j</sub>|] / "
                   "[∑<sub>j ∈ J</sub> w<sub>j</sub>]</strong>; w<sub>j</sub> = N<sub>j</sub> / (2n<sub>eval</sub>).</p><p>"
                 + html.escape(support) + "</p><p>" + html.escape(interpretation) + "</p></section>")
        path.write_text(path.read_text().replace("</main>", block + "</main>", 1))


def render(output, protocol, rows, local):
    """Write a complete, non-overwriting publication presentation and its hashes."""
    output = Path(output)
    if not output.is_dir():
        raise FileNotFoundError("Prepare the convergence output directory first")
    if any((output/name).exists() for name in FILES) or (output/"figures").exists():
        raise FileExistsError("Partial presentation exists; preserve it before retrying")
    rows = joined_rows(protocol, rows, local)
    cfg = protocol["config"]
    cases = sorted(cfg["cases"], key=lambda case: SETTINGS[case])
    sizes = sorted(cfg["sample_sizes"])
    candidates = sorted(protocol["candidates"], key=lambda candidate: LOSSES.index(candidate["loss"]))
    individual, aggregate = summarize(rows, cases, candidates, sizes)
    paired = paired_results(rows, cases, sizes)
    selected = candidates[0]
    fixed = f"Fixed NN architecture {paper.ARCH[selected['architecture']]} and output activation 2 sigmoid({selected['output_alpha']} z)."
    caption = fixed + f" Equal-weight mean over {len(cases)} fixed settings; parentheses give Monte Carlo standard errors."
    tables = [metric_table(LABELS[metric], aggregate, sizes, metric, caption) for metric in ERRORS]
    appendix = [metric_table("Validation Brier", aggregate, sizes, "validation_brier", caption)]
    appendix += [metric_table(LABELS[metric], aggregate, sizes, metric, caption, rank=False)
                 for metric in ("c2_width", "estimable_mass")]
    for case in cases:
        dimension, noise = SETTINGS[case]
        values = [row for row in individual if row["case"] == case]
        for metric in ERRORS:
            appendix.append(metric_table(f"D={dimension}, noise SD={noise:.2f}: {LABELS[metric]}", values, sizes, metric,
                fixed + f" Mean (Monte Carlo standard error) across {cfg['convergence_repeats']} matched repetitions."))
    intro = (f"{len(rows):,} verified fits compare four losses at {len(sizes)} training sample sizes in {len(cases)} "
             f"Gaussian-mixture settings, with {cfg['convergence_repeats']} repetitions per setting and sample size. "
             + fixed + " Architecture and activation are inherited from the completed selection study and remain fixed for every loss and sample size.")
    settings = "; ".join(f"D={SETTINGS[case][0]}, noise SD={SETTINGS[case][1]:.2f}" for case in cases)
    notes = [f"Settings: {settings}. Training n is the number from each distribution (total training size 2n). "
             f"Each P/Q sample has {cfg['validation_n']:,} validation, {cfg['calibration_n']:,} calibration and {cfg['evaluation_n']:,} evaluation observations at every n.",
             "Convergence uses fresh random streams distinct from model selection and the matched-loss follow-up. "
             "Within a setting/repetition/sample size, all four losses share data and initialization. Within a setting/repetition, "
             "validation, calibration and evaluation samples are reused across training sizes; training samples use size-specific random streams and are not nested. "
             "Observations in different sample roles are independent. Curves across n are paired, not independent experiments.",
             "The optimizer, scheduler, stopping rule and epoch budget are held fixed across losses and sample sizes. Each loss restores its own "
             "minimum-validation-loss checkpoint. There is no retuning at each n or a separate architecture search for each loss.",
             "Evaluation MSE uses analytic RDR truth. Balanced evaluation Brier uses observed P/Q labels and score r/2. "
             "Local gaps use the independent calibration sample and evaluation predictions; their definition appears below.",
             f"Per-setting means use {cfg['convergence_repeats']} repetitions. Equal-setting means weight the {len(cases)} fixed settings equally. "
             "Their Monte Carlo SE is sqrt(sum_k s_k²/R_k)/K, using within-setting variances only. Ribbons show one Monte Carlo SE, "
             "not simultaneous confidence bands. SE is unavailable with a single repetition. Available-fit counts for every metric are retained in the CSVs.",
             "Paired-comparison CSV entries are Hellinger minus each alternative at the same setting/repetition/n. Their SE uses paired differences "
             "before averaging fixed settings. Positive MSE, Brier or local-gap differences favor the alternative; table boldface alone does not establish statistical superiority.",
             "C.2 widths and estimable mass are contextual diagnostics without a winner ranking. C.2 intervals target population cell-average RDR "
             "and are simultaneous over the fixed score bins of one frozen model. No guarantee simultaneously across all losses, sample sizes or fitted models is asserted. "
             "A shorter interval does not by itself establish a better neural model.",
             "B8-Res64 uses a learned linear min(D,8)-dimensional bottleneck before a width-64 residual network. "
             "The curves describe this finite training protocol; no monotonicity, asymptotic rate, or loss equivalence is imposed."]
    for filename, values in (("metrics.csv", rows), ("calibration.csv", local), ("summary.csv", individual),
                             ("aggregation.csv", aggregate), ("paired_comparisons.csv", paired)):
        with (output/filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    images = paper.render_table_figures(tables+appendix, output)
    curves = curve_figures(output, cases, sizes, individual, aggregate)
    for extension in ("md", "html"):
        links = [("Per-setting appendix", f"appendix.{extension}"), ("All setting summaries", "summary.csv"),
                 ("Equal-setting averages", "aggregation.csv"), ("Paired comparisons", "paired_comparisons.csv"),
                 ("LaTeX tables", "tables.tex")]
        paper.write_document(output/f"report.{extension}", "Four-loss sample-size convergence", intro, tables, notes, links,
            figures=[("Convergence across fixed settings", curves[0]), ("Convergence within each setting", curves[1])],
            table_images=images[:len(tables)] if extension == "md" else ())
        local_gap_definition(output/f"report.{extension}", cfg, rows)
        paper.write_document(output/f"appendix.{extension}", "Four-loss convergence: setting-specific tables", intro,
            appendix, notes, [("Main report", f"report.{extension}"), ("All summaries", "summary.csv"),
                              ("LaTeX appendix", "appendix_tables.tex")],
            table_images=images[len(tables):] if extension == "md" else ())
    for filename, values in (("tables.tex", tables), ("appendix_tables.tex", appendix)):
        (output/filename).write_text("% Requires \\usepackage{booktabs}; entries are means (Monte Carlo SEs).\n\n"
                                    + "\n\n".join(paper.render_table(value, "tex") for value in values) + "\n")
    artifacts = [*FILES, *[str(path.relative_to(output)) for path in sorted((output/"figures").iterdir())]]
    return {name: w.digest(output/name) for name in artifacts}
