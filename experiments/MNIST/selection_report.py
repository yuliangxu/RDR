"""Render the verified DCGAN Brier and local-calibration selection results."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import numpy as np

METRICS = (
    "brier", "local_gap", "c2_distance", "c2_width", "supported_mass",
    "middle_mass", "middle_supported_mass", "middle_local_gap",
)
LABELS = {
    "brier": "Brier", "local_gap": "Local absolute gap",
    "c2_distance": "C.2 distance", "c2_width": "C.2 width",
    "supported_mass": "Supported mass", "middle_mass": "Middle mass",
    "middle_supported_mass": "Middle supported mass", "middle_local_gap": "Middle gap",
}


def _validate_metrics(row, cfg, calibration_role, evaluation_role):
    for key in METRICS:
        if key not in row:
            raise ValueError(f"Missing reported metric {key}")
        value = row[key]
        if value is None:
            if key == "middle_local_gap" and row.get("middle_supported_mass") == 0:
                continue
            raise ValueError(f"Unavailable reported metric {key}")
        if isinstance(value, (bool, str)) or not math.isfinite(float(value)):
            raise ValueError(f"Invalid reported metric {key}")
        upper = 1 if key in ("brier", "supported_mass", "middle_mass", "middle_supported_mass") else 2
        if not 0 <= value <= upper + 1e-12:
            raise ValueError(f"Reported metric {key} is outside its range")
    if row["middle_supported_mass"] > row["middle_mass"] + 1e-12:
        raise ValueError("Middle support exceeds middle mass")
    for key in ("n_cal_p", "n_cal_q"):
        if row.get(key) != cfg[calibration_role + "_n"]:
            raise ValueError(f"Incorrect calibration count {key}")
    for key in ("n_eval_p", "n_eval_q"):
        if row.get(key) != cfg[evaluation_role + "_n"]:
            raise ValueError(f"Incorrect evaluation count {key}")
    if row.get("bins") != cfg["bins"] or row.get("alpha") != cfg["ci_alpha"]:
        raise ValueError("Calibration settings differ from protocol")


def _aggregate(rows, cfg, stage):
    candidates = sorted({row["candidate"] for row in rows})
    result = []
    for candidate in candidates:
        values = [row for row in rows if row["candidate"] == candidate]
        if sorted(row["repeat"] for row in values) != list(range(cfg["repeats"])):
            raise ValueError(f"Missing or duplicate repetitions for {candidate}")
        first = values[0]
        summary = {"stage": stage, "candidate": candidate, "loss": first["loss"],
                   "output_alpha": first["output_alpha"], "repeats": len(values)}
        for key in METRICS:
            available = [row[key] for row in values if row[key] is not None]
            # An absent middle-region diagnostic remains unavailable; do not
            # turn the available repetitions into a silently smaller study.
            summary[key + "_available_repeats"] = len(available)
            complete = len(available) == len(values)
            summary[key + "_mean"] = float(np.mean(available)) if complete else None
            summary[key + "_sd"] = float(np.std(available, ddof=1)) if complete and len(values) > 1 else None
        result.append(summary)
    return result


def _csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _formatted(row, key):
    mean, sd = row[key + "_mean"], row[key + "_sd"]
    if mean is None:
        return f"Unavailable ({row[key + '_available_repeats']}/{row['repeats']} available)"
    return f"{mean:.6f} ({sd:.6f})" if sd is not None else f"{mean:.6f} (SD unavailable)"


def _table(rows, metrics):
    columns = ["Candidate"] + [LABELS[key] for key in metrics]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"]*len(columns)) + " |"]
    lines.extend("| " + " | ".join([row["candidate"]] + [_formatted(row, key) for key in metrics]) + " |" for row in rows)
    return "\n".join(lines)


def _save(fig, output, stem):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{extension}", dpi=180, bbox_inches="tight")


def _grid_plot(plt, output, summaries, cfg, chosen):
    from matplotlib.patches import Rectangle
    lookup = {(row["loss"], row["output_alpha"]): row for row in summaries}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, key in zip(axes, ("brier", "local_gap")):
        values = np.array([[lookup[(loss, alpha)][key + "_mean"] for alpha in cfg["output_alphas"]] for loss in cfg["losses"]])
        im = ax.imshow(values, aspect="auto", cmap="viridis_r")
        ax.set(xticks=range(len(cfg["output_alphas"])), xticklabels=[f"{a:g}" for a in cfg["output_alphas"]],
               yticks=range(len(cfg["losses"])), yticklabels=cfg["losses"], xlabel="Sigmoid slope alpha",
               title=f"Mean {LABELS[key].lower()} (lower is better)")
        for i, loss in enumerate(cfg["losses"]):
            for j, alpha in enumerate(cfg["output_alphas"]):
                ax.text(j, i, f"{values[i,j]:.4f}", ha="center", va="center",
                        color="white" if values[i,j] > np.nanmean(values) else "black")
                if lookup[(loss, alpha)]["candidate"] == chosen:
                    ax.add_patch(Rectangle((j-.48, i-.48), .96, .96, fill=False, edgecolor="red", linewidth=2))
        fig.colorbar(im, ax=ax, shrink=.85)
    fig.suptitle(f"DCGAN selection: {cfg['repeats']} paired repetitions; red box = selected")
    _save(fig, output, "selection_grid")
    plt.close(fig)


def _cell_plot(plt, output, cells, cfg, stem, title):
    if len(cells) != cfg["bins"] or [cell.get("bin") for cell in cells] != list(range(cfg["bins"])):
        raise ValueError("Incomplete or reordered calibration cells")
    edges = np.linspace(0, 2, cfg["bins"]+1)
    for i, cell in enumerate(cells):
        if not np.isclose(cell["left"], edges[i]) or not np.isclose(cell["right"], edges[i+1]):
            raise ValueError("Unexpected calibration score partition")
        for key in ("c2_lower", "c2_upper", "eval_mass"):
            if not math.isfinite(cell[key]):
                raise ValueError(f"Invalid cell {key}")
        if not 0 <= cell["c2_lower"] <= cell["c2_upper"] <= 2:
            raise ValueError("Invalid C.2 interval")
        if not 0 <= cell["eval_mass"] <= 1:
            raise ValueError("Invalid cell evaluation mass")
        for key, available in (("neural_mean", cell["eval_mass"] > 0),
                               ("calibrated_rdr", cell["cal_count_p"] + cell["cal_count_q"] > 0)):
            value = cell[key]
            if (value is None) != (not available):
                raise ValueError(f"Inconsistent availability of {key}")
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 2):
                raise ValueError(f"Invalid cell {key}")
    lower, upper = [np.array([cell[key] for cell in cells]) for key in ("c2_lower", "c2_upper")]
    centers = .5*(edges[1:] + edges[:-1])
    neural, calibrated = [np.array([np.nan if cell[key] is None else cell[key] for cell in cells])
                          for key in ("neural_mean", "calibrated_rdr")]
    fig, (ax, mass_ax) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True)
    ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step="post",
                    color="steelblue", alpha=.22, label=f"C.2 {100*(1-cfg['ci_alpha']):g}% cell bands")
    ax.plot(centers, calibrated, "o", color="navy", markersize=4, label="Calibration cell RDR estimate")
    ax.plot(centers, neural, "s", color="darkorange", markersize=4, label="Independent evaluation neural mean")
    ax.set(ylim=(-.03, 2.03), ylabel="Cell-average RDR", title=title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=.2)
    mass_ax.bar(centers, [cell["eval_mass"] for cell in cells], width=np.diff(edges)*.85, color="slategray")
    mass_ax.axvspan(.7, 1.2, color="orange", alpha=.15, label="Middle region [0.7, 1.2)")
    mass_ax.set(xlim=(0, 2), xlabel="Frozen neural score cell", ylabel="Mixture mass")
    mass_ax.legend(fontsize=8)
    _save(fig, output, stem)
    plt.close(fig)


def report(output):
    from experiments.MNIST import model_selection as w
    output = Path(output).resolve()
    protocol = w.load_protocol(output)
    cfg = protocol["config"]
    rows = w.all_rows(output, protocol)
    selection = w.load_selection(output, protocol)
    expected = {(candidate["candidate"], repeat) for candidate in protocol["candidates"] for repeat in range(cfg["repeats"])}
    if len(rows) != len(expected) or {(row["candidate"], row["repeat"]) for row in rows} != expected:
        raise ValueError("Selection grid is incomplete or duplicated")
    for row in rows:
        _validate_metrics(row, cfg, "selection_calibration", "selection_evaluation")
    chosen = selection["chosen"]
    selected_task = next(task for task in protocol["tasks"] if task["candidate"] == chosen["candidate"] and task["repeat"] == 0)
    selected_cells = w.read(w.run_path(output, selected_task) / "cells.json")
    final_rows = []
    final_folders = [output / "final" / f"repeat_{repeat:02d}" for repeat in range(cfg["repeats"])]
    if (output / "final").exists():
        unexpected = [path for path in (output / "final").iterdir() if path.is_dir() and path not in final_folders]
        if unexpected:
            raise ValueError(f"Unexpected final repetition folders: {unexpected}")
    if any(path.exists() for path in final_folders):
        if not all((path / "complete.json").is_file() for path in final_folders):
            raise ValueError("Final assessment is incomplete; all selected repetitions must finish before reporting it")
        for repeat, folder in enumerate(final_folders):
            row = w.verify_result(folder, output)
            if row.get("repeat") != repeat or any(row.get(key) != chosen[key] for key in chosen):
                raise ValueError(f"Incorrect selected-model identity in {folder}")
            if row.get("selection_sha256") != w.sha(output / "selection.json"):
                raise ValueError("Final assessment belongs to a different frozen selection")
            _validate_metrics(row, cfg, "final_calibration", "final_evaluation")
            final_rows.append(row)
    summaries = _aggregate(rows, cfg, "selection")
    final_summaries = _aggregate(final_rows, cfg, "final") if final_rows else []
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mnist-selection-report-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _grid_plot(plt, output, summaries, cfg, chosen["candidate"])
    _cell_plot(plt, output, selected_cells, cfg, "selected_selection_cells",
               f"{chosen['candidate']}: selection cells, repetition 0")
    if final_rows:
        _cell_plot(plt, output, w.read(final_folders[0] / "cells.json"), cfg, "selected_final_cells",
                   f"{chosen['candidate']}: final assessment cells, repetition 0")
    per_fit = [{"stage": "selection", **row} for row in rows] + [{"stage": "final", **row} for row in final_rows]
    _csv(output / "per_fit.csv", per_fit, ("stage", "candidate", "loss", "output_alpha", "repeat") + METRICS)
    all_summaries = summaries + final_summaries
    _csv(output / "summary.csv", all_summaries, list(all_summaries[0]))
    status = "Selection and all selected-model final assessments complete" if final_rows else "Selection complete; final assessment not run"
    role_rows = []
    for role, source in (("train", "official training"), ("earlystop", "official training"),
                         ("selection_calibration", "official training"), ("selection_evaluation", "official training"),
                         ("final_calibration", "official test"), ("final_evaluation", "official test")):
        q_count = f"{cfg[role+'_n']:,} fresh draws per epoch" if role == "train" else f"{cfg[role+'_n']:,}"
        role_rows.append(f"| {role} | {source} | {cfg[role+'_n']:,} | {q_count} |")
    lines = ["# MNIST DCGAN model selection", "", f"**Status: {status}.**",
             f"**Smoke study: {'yes; plumbing only, not scientific results' if protocol['smoke'] else 'no'}.**",
             f"Verified {len(rows)}/{len(expected)} selection fits and {len(final_rows)}/{cfg['repeats']} final assessments.", "",
             f"**Selected:** `{chosen['candidate']}` ({chosen['loss']}; r=2 sigmoid({chosen['output_alpha']:g} z)).",
             f"Brier-reference candidate: `{selection['brier_reference']}`.", "", "## Registered selection procedure", "",
             "Each checkpoint is chosen by the smallest early-stopping balanced Brier. " + selection["rule"], "",
             "Brier uses labels P=1, Q=0 and probability r/2, with equal source weights. "
             "Local gap is the evaluation-mixture-mass-weighted absolute difference between the independently "
             "estimated neural cell mean and calibration cell RDR estimate, normalized over supported mass. "
             "C.2 distance measures distance of that neural mean from the calibration interval; width is "
             "weighted across all evaluation mass. Support records evaluation mass with both a neural mean "
             "and a calibration point estimate. The middle region is [0.7, 1.2).", "",
             "## Split accounting", "", "| Role | Real-image source | P count | Q count |", "| --- | --- | --- | --- |",
             *role_rows, "", "Real-image roles are disjoint. All candidates use paired real observations, "
             "initializations and generator seeds within each repetition. Real splits stay fixed across repetitions; "
             "initializations and Q samples change. P and Q input pixels use [-1,1].", "",
             "## Selection diagnostics", "",
             f"Entries are mean (sample SD) across {cfg['repeats']} paired repetitions, conditional on the fixed "
             "real data. These are not sampling confidence intervals. Unavailable middle-region values are "
             "reported with their available repetition count and are not averaged after dropping missing values.", "",
             _table(summaries, ("brier", "local_gap", "c2_distance", "c2_width", "supported_mass")), "",
             _table(summaries, ("middle_mass", "middle_supported_mass", "middle_local_gap")), "",
             "![Brier and local-gap grid](selection_grid.png)", "[Grid PDF](selection_grid.pdf)", "",
             "![Selected model calibration cells](selected_selection_cells.png)",
             "[Selection cells PDF](selected_selection_cells.pdf)", ""]
    if final_rows:
        lines.extend(["## Assessment after selection was frozen", "",
                      _table(final_summaries, ("brier", "local_gap", "c2_distance", "c2_width", "supported_mass")), "",
                      _table(final_summaries, ("middle_mass", "middle_supported_mass", "middle_local_gap")), "",
                      "![Selected model final cells](selected_final_cells.png)",
                      "[Final cells PDF](selected_final_cells.pdf)", ""])
    lines.extend(["## Interpretation and provenance", "",
                  f"The {cfg['bins']} score cutoffs are fixed, but their input-space cells depend on each fitted "
                  f"network. The nominal {100*(1-cfg['ci_alpha']):g}% C.2 intervals are simultaneous over cells for "
                  "one fixed model under the calibration assumptions. They target population cell-average RDR, "
                  "not individual-image RDR. Selection-stage intervals are unadjusted across candidate models; "
                  "they do not provide a joint guarantee after model selection. Neural cell means also have "
                  "sampling uncertainty that the displayed calibration intervals do not include. Repetition-0 "
                  "cell figures are descriptive examples; all repetitions enter the tables.", "",
                  "Official-test images were inspected in historical MNIST studies. Final assessment here is "
                  "retrospective, although these roles are excluded from this selection procedure. Frozen DCGAN "
                  "pretraining membership has not been audited. This study does not replace the retained VAE, "
                  "DCGAN, digit-perturbation results or the validation-only null diagnostic.", "",
                  "[Per-fit metrics](per_fit.csv) · [Summary metrics](summary.csv) · "
                  "[Frozen protocol](protocol.json) · [Split manifest](splits.json) · [Frozen selection](selection.json)", ""])
    (output / "RESULTS.md").write_text("\n".join(lines))
    w.write(output / "report_status.json", {"selection_complete": True, "selection_fits": len(rows),
            "expected_selection_fits": len(expected), "final_complete": bool(final_rows),
            "final_repeats": len(final_rows), "expected_final_repeats": cfg["repeats"],
            "smoke": protocol["smoke"], "chosen": chosen, "protocol_sha256": w.sha(output / "protocol.json"),
            "selection_sha256": w.sha(output / "selection.json")})
    print(output / "RESULTS.md")
