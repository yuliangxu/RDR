"""Report paired CelebA selection and frozen-model retrospective assessment.

This module only renders verified records supplied by the workflow. It never
selects candidates from assessment metrics and can render an incomplete study.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
from pathlib import Path

import numpy as np


METRICS = (
    "brier", "local_gap", "supported_mass", "middle_mass",
    "middle_supported_mass", "middle_supported_fraction", "middle_local_gap",
    "c2_distance", "c2_width", "whole_test_brier", "evaluation_brier",
)
LABELS = {
    "brier": "Balanced Brier", "local_gap": "Local absolute gap",
    "supported_mass": "Supported mass", "middle_mass": "Middle mass",
    "middle_supported_mass": "Middle supported mass",
    "middle_supported_fraction": "Middle supported fraction",
    "middle_local_gap": "Middle local gap", "c2_distance": "C.2 distance",
    "c2_width": "C.2 width", "whole_test_brier": "Whole-test balanced Brier",
    "evaluation_brier": "Independent evaluation Brier",
}
IDENTITY = ("representation", "branch", "architecture", "loss", "output_alpha", "repeat")


def _identity(row):
    return tuple(row[key] for key in IDENTITY)


def _validate(rows):
    seen, definitions = set(), {}
    required = set(IDENTITY) | {"candidate", "brier", "local_gap", "supported_mass"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"Report row missing {sorted(missing)}")
        identity = _identity(row)
        if identity in seen:
            raise ValueError(f"Duplicate report row: {identity}")
        seen.add(identity)
        candidate_key = (row["representation"], row["branch"], row["candidate"])
        definition = tuple(row[key] for key in ("architecture", "loss", "output_alpha"))
        if definitions.setdefault(candidate_key, definition) != definition:
            raise ValueError(f"Inconsistent candidate definition: {candidate_key}")
        if (isinstance(row["repeat"], bool) or not isinstance(row["repeat"], int)
                or row["repeat"] < 0):
            raise ValueError("Invalid report repetition")
        for key in METRICS:
            value = row.get(key)
            if value is None:
                if key in ("brier", "supported_mass"):
                    raise ValueError(f"Unavailable required metric {key}")
                if key == "local_gap" and row["supported_mass"] > 0:
                    raise ValueError("Local gap unavailable with positive supported mass")
                continue
            if isinstance(value, (bool, str)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid report metric {key}")
            upper = 2 if key in ("local_gap", "middle_local_gap", "c2_distance", "c2_width") else 1
            if not 0 <= value <= upper + 1e-12:
                raise ValueError(f"Report metric {key} outside its range")
        if row.get("middle_supported_mass") is not None and row.get("middle_mass") is not None:
            if row["middle_supported_mass"] > row["middle_mass"] + 1e-12:
                raise ValueError("Middle supported mass exceeds middle mass")


def _aggregate(rows, expected_repeats, stage):
    result = []
    groups = sorted({(r["representation"], r["branch"], r["candidate"]) for r in rows})
    for representation, branch, candidate in groups:
        group = [r for r in rows if (r["representation"], r["branch"], r["candidate"])
                 == (representation, branch, candidate)]
        first = group[0]
        summary = {"stage": stage, "representation": representation, "branch": branch,
                   "candidate": candidate, "architecture": first["architecture"],
                   "loss": first["loss"], "output_alpha": first["output_alpha"],
                   "repeats": len(group), "expected_repeats": expected_repeats,
                   "complete": sorted(r["repeat"] for r in group) == list(range(expected_repeats))}
        for key in METRICS:
            values = [r[key] for r in group if r.get(key) is not None]
            summary[key + "_available_repeats"] = len(values)
            # Incomplete diagnostics are not silently averaged over fewer runs.
            available = len(values) == len(group) and summary["complete"]
            summary[key + "_mean"] = float(np.mean(values)) if available else None
            summary[key + "_sd"] = float(np.std(values, ddof=1)) if available and len(values) > 1 else None
        result.append(summary)
    return result


def _csv(path, rows, preferred):
    fields = list(preferred) + sorted(set().union(*(r.keys() for r in rows)) - set(preferred))
    # Exclude nested provenance/cell arrays, which have their own JSON artifacts.
    fields = [key for key in fields if not any(isinstance(r.get(key), (dict, list)) for r in rows)]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format(row, key):
    value = row[key + "_mean"]
    if value is None:
        return f"Unavailable ({row[key + '_available_repeats']}/{row['expected_repeats']})"
    sd = row[key + "_sd"]
    return f"{value:.6f} ({sd:.6f})" if sd is not None else f"{value:.6f} (SD unavailable)"


def _table(summaries, metrics):
    headers = ["Representation / pair", "Candidate", "Repeats"] + [LABELS[k] for k in metrics]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in summaries:
        cells = [f"{row['representation']} / P vs Q_{row['branch']}", f"`{row['candidate']}`",
                 f"{row['repeats']}/{row['expected_repeats']}"] + [_format(row, key) for key in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    if not summaries:
        lines.append("| Pending | — | 0 | " + " | ".join(["—"] * len(metrics)) + " |")
    return lines


def _chosen(selection, key):
    item = (selection or {}).get("selections", {}).get(key, {})
    return item.get("chosen", item) if isinstance(item, dict) else {}


def _expected_baseline(config):
    return set(itertools.product(config["levels"], config["branches"], ["baseline"],
                                 config["losses"], config["output_alphas"], range(config["repeats"])))


def _load_tasks(output, filename):
    path = output / filename
    if not path.is_file():
        return None
    value = json.loads(path.read_text())
    return value.get("tasks", []) if isinstance(value, dict) else value


def _split_accounting(output, protocol):
    metadata = protocol.get("data", {})
    if (output / "data" / "metadata.json").is_file():
        metadata = json.loads((output / "data" / "metadata.json").read_text())
    roles = metadata.get("roles", {})
    if roles and all(isinstance(sources.get("real"), dict) and "count" in sources["real"]
                     for sources in roles.values()):
        lines = ["| Role | Historical source / use | P images | P identities | Q_lower images | Q_upper images |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for role, sources in roles.items():
            definition = metadata.get("role_definitions", {}).get(role, "See prepared manifest")
            counts = [sources.get(source, {}).get("count", "Pending") for source in ("real", "lower", "upper")]
            identities = sources["real"].get("identity_count", "Unavailable")
            lines.append(f"| {role} | {definition} | {counts[0]} | {identities} | {counts[1]} | {counts[2]} |")
        return lines
    lines = []
    # Include exact role accounting wherever the preparation metadata stores it.
    def visit(value, prefix=""):
        if isinstance(value, dict):
            scalar = {k: v for k, v in value.items() if isinstance(v, (str, int, float, bool)) or v is None}
            if scalar and any(token in prefix.lower() for token in
                              ("role", "train", "stop", "selection", "calibration", "evaluation", "test", "count")):
                text = "; ".join(f"{k}={v}" for k, v in scalar.items())
                lines.append(f"| {prefix or 'data'} | {text.replace('|', '/')} |")
            for key, item in value.items():
                if isinstance(item, dict):
                    visit(item, f"{prefix}/{key}".strip("/"))
    visit(metadata)
    if not lines:
        lines = ["| Pending | Exact P/Q role counts appear here once data preparation finishes. |"]
    return ["| Role / source | Prepared-data accounting |", "| --- | --- |", *lines]


def _plots(output, summaries, config, baseline_selection, selection_label='baseline selection'):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/celeba-selection-report-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    artifacts = []
    for representation in config["levels"]:
        fig, axes = plt.subplots(len(config["branches"]), 2, squeeze=False,
                                 figsize=(11, 4.2 * len(config["branches"])), constrained_layout=True)
        for i, branch in enumerate(config["branches"]):
            lookup = {(r["loss"], r["output_alpha"]): r for r in summaries
                      if r["representation"] == representation and r["branch"] == branch
                      and r["architecture"] == "baseline"}
            chosen = _chosen(baseline_selection, f"{representation}/{branch}").get("candidate")
            for ax, metric in zip(axes[i], ("brier", "local_gap")):
                values = np.full((len(config["losses"]), len(config["output_alphas"])), np.nan)
                for y, loss in enumerate(config["losses"]):
                    for x, alpha in enumerate(config["output_alphas"]):
                        value = lookup.get((loss, alpha), {}).get(metric + "_mean")
                        if value is not None:
                            values[y, x] = value
                cmap = plt.get_cmap("viridis_r").copy()
                cmap.set_bad("#eeeeee")
                im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap)
                ax.set(xticks=range(len(config["output_alphas"])),
                       xticklabels=[f"{a:g}" for a in config["output_alphas"]],
                       yticks=range(len(config["losses"])), yticklabels=config["losses"],
                       xlabel="Sigmoid slope a in r = 2 sigmoid(a z)",
                       title=f"P vs Q_{branch}: {LABELS[metric]}")
                for y, loss in enumerate(config["losses"]):
                    for x, alpha in enumerate(config["output_alphas"]):
                        value = values[y, x]
                        rgba = im.cmap(im.norm(value)) if np.isfinite(value) else (1, 1, 1, 1)
                        brightness = .299 * rgba[0] + .587 * rgba[1] + .114 * rgba[2]
                        summary = lookup.get((loss, alpha), {})
                        candidate_id = summary.get('candidate', f'baseline_{loss}_a{alpha:g}'.replace('.', 'p'))
                        directory = output / 'fits' / representation / branch / candidate_id
                        failed = sum((directory / f'repeat_{repeat:02d}' / 'FAILED.json').exists()
                                     and not (directory / f'repeat_{repeat:02d}' / 'COMPLETE.json').exists()
                                     for repeat in range(config['repeats']))
                        label = (f"{value:.4f}" if np.isfinite(value) else
                                 f"Failed\n({failed}/{config['repeats']})" if failed else
                                 'Unavailable' if summary.get('complete') else 'Pending')
                        ax.text(x, y, label,
                                ha="center", va="center", color="black" if brightness > .5 else "white", fontsize=9)
                        if chosen and lookup.get((loss, alpha), {}).get("candidate") == chosen:
                            ax.add_patch(Rectangle((x-.47, y-.47), .94, .94, fill=False,
                                                   edgecolor="red", linewidth=2))
                fig.colorbar(im, ax=ax, shrink=.8)
        fig.suptitle(f"CelebA {representation}: baseline architecture; {config['repeats']} paired repeats\n"
                     f"Red box = {selection_label}; each pair has its own color scale\n"
                     "Failed (k/n): k fits failed; no n-repeat mean is reported")
        stem = f"selection_grid_{representation}"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        artifacts.append(stem)
    return artifacts


def build_report(output, protocol, rows, baseline_selection=None, freeze=None, evaluation_rows=None):
    """Write diagnostics, completion accounting and plots; return report status.

    ``rows`` and ``evaluation_rows`` must already have passed the workflow's
    completion/provenance verification. Missing fits remain explicitly pending.
    """
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = protocol["config"]
    rows = list(rows)
    evaluation_rows = [{**r, "evaluation_brier": r.get("evaluation_brier", r["brier"])}
                       for r in evaluation_rows or []]
    _validate(rows)
    _validate(evaluation_rows)
    repeats = config["repeats"]
    expected_baseline = _expected_baseline(config)
    actual_baseline = {_identity(r) for r in rows if r["architecture"] == "baseline"}
    if actual_baseline - expected_baseline:
        raise ValueError("Report contains baseline fits outside the registered grid")
    baseline_complete = actual_baseline == expected_baseline
    sensitivity_tasks = _load_tasks(output, "sensitivity_tasks.json")
    expanded_tasks = _load_tasks(output, "expanded_tasks.json")
    expected_sensitivity = {_identity(r) for r in sensitivity_tasks} if sensitivity_tasks is not None else set()
    expected_expanded = {_identity(r) for r in expanded_tasks} if expanded_tasks is not None else set()
    actual_sensitivity = {_identity(r) for r in rows if r["architecture"] != "baseline"}
    if actual_sensitivity - (expected_sensitivity | expected_expanded):
        raise ValueError("Report contains sensitivity fits outside the registered tasks")
    sensitivity_complete = sensitivity_tasks is not None and expected_sensitivity <= actual_sensitivity
    expansion_complete = expected_expanded <= actual_sensitivity
    expected_evaluation = set()
    for representation, branch in itertools.product(config["levels"], config["branches"]):
        chosen = _chosen(freeze, f"{representation}/{branch}")
        if chosen:
            for repeat in range(repeats):
                expected_evaluation.add((representation, branch, chosen["architecture"],
                                         chosen["loss"], chosen["output_alpha"], repeat, chosen["candidate"]))
    actual_evaluation = {_identity(r) + (r["candidate"],) for r in evaluation_rows}
    expected_final_count = len(config["levels"]) * len(config["branches"]) * repeats
    if actual_evaluation - expected_evaluation:
        raise ValueError("Assessment rows do not match the frozen selected candidates")
    final_complete = bool(freeze) and len(expected_evaluation) == expected_final_count and actual_evaluation == expected_evaluation
    complete = baseline_complete and sensitivity_complete and expansion_complete and final_complete
    summaries = _aggregate(rows, repeats, "selection")
    final_summaries = _aggregate(evaluation_rows, repeats, "retrospective_assessment")
    per_fit = [{**r, "stage": "selection"} for r in rows]
    per_fit += [{**r, "stage": "retrospective_assessment"} for r in evaluation_rows]
    _csv(output / "per_fit.csv", per_fit, ("stage", "task_index", "representation", "branch", "candidate", "repeat") + METRICS)
    _csv(output / "per_candidate.csv", summaries + final_summaries,
         ("stage", "representation", "branch", "candidate", "architecture", "loss", "output_alpha", "repeats", "expected_repeats", "complete"))
    artifacts = _plots(output, summaries, config, baseline_selection)
    status = {
        "complete": complete, "smoke": bool(protocol.get("smoke", False)),
        "baseline_complete": baseline_complete, "baseline_fits": len(actual_baseline),
        "expected_baseline_fits": len(expected_baseline),
        "missing_baseline_fits": len(expected_baseline - actual_baseline),
        "sensitivity_registered": sensitivity_tasks is not None, "sensitivity_complete": sensitivity_complete,
        "sensitivity_fits": len(actual_sensitivity & expected_sensitivity),
        "expected_sensitivity_fits": len(expected_sensitivity) if sensitivity_tasks is not None else None,
        "expansion_registered": expanded_tasks is not None, "expansion_complete": expansion_complete,
        "expanded_fits": len(actual_sensitivity & expected_expanded),
        "expected_expanded_fits": len(expected_expanded),
        "final_complete": final_complete, "evaluation_fits": len(actual_evaluation),
        "expected_evaluation_fits": expected_final_count,
    }
    sensitivity_count = str(len(expected_sensitivity)) if sensitivity_tasks is not None else "pending registration"
    lines = ["# CelebA paired model selection", "",
             f"**Status: {'complete; all frozen-model assessments verified' if complete else 'pending; study is incomplete'}.**",
             f"**Smoke study: {'yes; plumbing only, not scientific evidence' if protocol.get('smoke') else 'no'}.**", "",
             f"Baseline grid: **{len(actual_baseline)}/{len(expected_baseline)}** verified fits "
             f"({len(expected_baseline - actual_baseline)} missing). "
             f"Architecture sensitivity: **{len(actual_sensitivity & expected_sensitivity)}/{sensitivity_count}** fits. "
             f"Additional expanded grid: **{len(actual_sensitivity & expected_expanded)}/{len(expected_expanded)}** fits. "
             f"Frozen-model retrospective assessment: **{len(actual_evaluation)}/{expected_final_count}** fits.", "",
             "## Registered comparison", "",
             "Separate weights are trained for P vs Q_lower and P vs Q_upper. The same loss × activation grid "
             "is evaluated within each fixed architecture. Checkpoints use stopping-set balanced Brier. "
             "For each pair, the minimum-mean-Brier reference defines the paired one-standard-error shortlist; "
             "eligible candidates require at least 0.99 supported mass in every repeat. Smallest mean local "
             "absolute gap breaks the Brier shortlist, followed by mean Brier and candidate ID. "
             "Architecture sensitivity crosses a narrower alternative with the union of two leaders per pair: "
             "the Brier reference and the baseline selection, adding the next support-eligible Brier-ranked "
             "setting when these coincide. The union is applied to both pairs of each representation. "
             "Assessment data never enter selection.", "",
             f"Expand the narrower architecture to the full loss × activation grid for both pairs of a "
             f"representation if any alternative has supported mass at least 0.99 in every repeat, beats its "
             f"pair's baseline-selected Brier by at least "
             f"{config.get('sensitivity', {}).get('expansion_minimum_brier_gain', .002):g}, "
             f"exceeds {config.get('sensitivity', {}).get('expansion_paired_se_multiplier', 1):g} paired "
             f"standard error of the Brier difference, and improves Brier in at least "
             f"{100 * config.get('sensitivity', {}).get('expansion_positive_repeat_fraction', .8):g}% of repeats. "
             "Without this trigger, baseline selection remains primary. Following expansion, apply the same "
             "Brier/Gap rule to the complete baseline and alternative grids. This trigger is a conditional "
             "repeat-variability heuristic, not an independent-data significance test.", "",
             "Balanced Brier uses P=1, Q=0 and predicted probability r/2, weighting the two sources equally. "
             "Local Gap is the evaluation-mixture-mass-weighted absolute difference between independent "
             "calibration cell RDR estimates and evaluation neural cell means, normalized over supported mass. "
             "The middle score region is [0.7, 1.2). Its mass, supported fraction and Gap are separate diagnostics.", "",
             "## Selected configurations", "",
             "| Representation / pair | Baseline choice | Frozen final choice |", "| --- | --- | --- |"]
    for representation, branch in itertools.product(config["levels"], config["branches"]):
        key = f"{representation}/{branch}"
        baseline = _chosen(baseline_selection, key)
        final = _chosen(freeze, key)
        lines.append(f"| {key} | {baseline.get('candidate', 'Pending')} | {final.get('candidate', 'Pending')} |")
    lines.extend(["", "Common configurations are supported only when eligible for both pairs; "
                  "they still use separately trained weights.", ""])
    for label, selection in (("Baseline", baseline_selection), ("Final", freeze)):
        for representation, value in (selection or {}).get("shared_configurations", {}).items():
            if isinstance(value, dict):
                choice = value.get("chosen", value.get("candidate"))
                if isinstance(choice, dict):
                    choice = choice.get("candidate", choice)
                eligible = value.get("eligible_candidates", value.get("common_candidates"))
                text = str(choice) if choice else "no common choice recorded"
                if eligible is not None:
                    text += f"; shared eligible candidates: {eligible}"
            else:
                text = str(value)
            lines.append(f"- {label}, {representation}: {text}.")
    lines.extend(["", "## Split accounting", "", *_split_accounting(output, protocol), "",
                  "The same P observations may be shared between pairs only in matching roles. "
                  "Training, stopping, selection calibration and selection evaluation use separate roles. "
                  "Real identities remain grouped by the prepared split manifest. "
                  "Repeated seeds are conditional on the fixed prepared data; repeat SDs are descriptive "
                  "and are not independent-data sampling confidence intervals.", "",
                  "## Selection diagnostics", "",
                  "Entries are mean (sample SD) over all registered repeats. Missing fits or unavailable "
                  "diagnostics are shown with available/expected counts; they are never silently dropped.", "",
                  *_table(summaries, ("brier", "local_gap", "supported_mass")), "",
                  *_table(summaries, ("middle_mass", "middle_supported_fraction", "middle_local_gap")), ""])
    for stem in artifacts:
        lines.extend([f"![Baseline Brier and local-Gap grid]({stem}.png)", f"[PDF]({stem}.pdf)", ""])
    lines.extend(["## Assessment after selection was frozen", ""])
    if not final_complete:
        lines.extend(["**Pending. Partial rows below do not constitute a completed assessment.**", ""])
    lines.extend(["Whole-test balanced Brier is the primary aggregate retrospective score, using the union "
                  "of test_calibration and test_evaluation. Independent evaluation Brier uses only "
                  "test_evaluation, whose observations also estimate neural cell means and mixture masses "
                  "for comparison with test_calibration cell RDR estimates. Neither score enters selection.", "",
                  *_table(final_summaries, ("whole_test_brier", "evaluation_brier", "local_gap", "supported_mass"))])
    lines.extend(["", *_table(final_summaries, ("middle_mass", "middle_supported_fraction", "middle_local_gap")), "",
                  "## Interpretation and provenance", "",
                  f"The {config.get('bins', 20)} score cutoffs on [0, 2] are fixed; the corresponding input-space "
                  "cells depend on each fitted network. Calibration intervals target population cell-average "
                  "RDR, not individual-image RDR. Nominal image-level CIs rely on the calibration sampling "
                  "assumptions and do not account for residual within-identity dependence, neural-mean sampling "
                  "uncertainty or selection across models. Historical validation had already been used for fitting, "
                  "and historical design/test data were already inspected; "
                  "their reuse here is retrospective assessment, even though they are excluded from this selection.", "",
                  "Compare candidate Brier differences within each P/Q pair. Raw Brier values across pairs "
                  "also reflect different classification difficulty and are not a direct comparison of RDR "
                  "estimation quality. Agreement across these two pairs does not establish transfer to arbitrary Q. "
                  "Architecture conclusions cover only the registered alternatives. Local calibration diagnostics "
                  "complement overall Brier and do not replace it.", "",
                  "[Per-fit metrics](per_fit.csv) · [Per-candidate metrics](per_candidate.csv) · "
                  "[Frozen protocol](protocol.json) · [Prepared-data metadata](data/metadata.json) · "
                  "[Report status](report_status.json)", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    (output / "report_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status
