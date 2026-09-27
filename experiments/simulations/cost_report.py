"""Publication tables for verified, timed replays of the convergence fits.

The runner checks the complete timing grid against the original fitted models,
histories and stopping metadata before calling ``render``. This module reports
sample standard deviations, not the Monte Carlo SEs of the scientific report.
It also creates a combined scientific-and-cost report without modifying either
the original convergence fits or its frozen presentation.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

import numpy as np
from experiments.simulations import report as paper
from experiments.simulations import workflow as w
from experiments.simulations.population import SETTINGS

LOSSES = ("hellinger", "kl", "chisq", "js")
MEASURES = ("wall_seconds", "cpu_seconds", "epochs")
LABELS = {"wall_seconds": "Training wall time (s)",
          "cpu_seconds": "Training CPU time (s)", "epochs": "Training epochs"}
PRESENTATION_FILES = ("report.md", "report.html", "cost_report.md", "cost_report.html",
                      "cost_appendix.md", "cost_appendix.html", "cost_results.csv",
                      "cost_summary.csv", "cost_hardware.json", "cost_tables.tex",
                      "cost_appendix_tables.tex", "cost_rendering_manifest.json")


def checked_rows(protocol, rows):
    """Reject duplicate, incomplete or nonfinite per-setting timing records."""
    cfg, candidates = protocol["config"], protocol["candidates"]
    if (protocol["stage"] != "computation_cost" or len(candidates) != 4
            or {row["loss"] for row in candidates} != set(LOSSES)
            or len({(row["architecture"], row["output_alpha"]) for row in candidates}) != 1):
        raise ValueError("Cost report requires the fixed four-loss convergence model")
    if protocol.get("threads") != 1 or protocol.get("device") != "cpu":
        raise ValueError("This timing presentation requires a single-thread CPU protocol")
    repeats = protocol.get("repeats", cfg["convergence_repeats"])
    repeats = list(range(repeats)) if isinstance(repeats, int) else list(repeats)
    if len(repeats) < 2 or len(set(repeats)) != len(repeats):
        raise ValueError("Cost summaries require at least two distinct timing repetitions")
    expected = {(case, repeat, n, w.candidate_id(candidate))
                for case in cfg["cases"] for repeat in repeats
                for n in cfg["sample_sizes"] for candidate in candidates}
    identity = lambda row: tuple(row[key] for key in ("case", "repeat", "train_n", "candidate"))
    lookup = {identity(row): row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError("Cost report requires the complete timing grid exactly once")
    by_candidate = {w.candidate_id(row): row for row in candidates}
    checked = []
    for key in sorted(expected):
        row = dict(lookup[key])
        if any(row[name] != value for name, value in by_candidate[row["candidate"]].items()):
            raise ValueError("Timing row factors disagree with its candidate")
        if any(not np.isfinite(row[name]) or row[name] <= 0
               for name in (*MEASURES, "n_parameters")):
            raise ValueError("Timing records must have positive finite measurements")
        row.update(dimension=SETTINGS[row["case"]][0], noise_sd=SETTINGS[row["case"]][1])
        checked.append(row)
    return checked, repeats


def summarize(rows, cases, sizes):
    """One row per case/n/loss, with sample SD (ddof=1) over timed replays."""
    summaries = []
    for case in cases:
        for n in sizes:
            for loss in LOSSES:
                values = [row for row in rows
                          if (row["case"], row["train_n"], row["loss"]) == (case, n, loss)]
                first = values[0]
                if len({row["n_parameters"] for row in values}) != 1:
                    raise ValueError("Parameter count changed within a timing setting")
                result = {key: first[key] for key in
                          ("case", "dimension", "noise_sd", "train_n", "candidate", "loss",
                           "architecture", "output_alpha", "n_parameters")}
                result["timing_repeats"] = len(values)
                for metric in MEASURES:
                    samples = np.asarray([row[metric] for row in values], dtype=float)
                    result[f"{metric}_mean"] = float(samples.mean())
                    result[f"{metric}_sd"] = float(samples.std(ddof=1))
                summaries.append(result)
    for case in cases:
        if len({row["n_parameters"] for row in summaries if row["case"] == case}) != 1:
            raise ValueError("Fixed architecture has inconsistent parameter counts")
    return summaries


def timing_table(case, values, sizes, metric, repeats):
    """Keep the six sample sizes in rows and the four losses in columns."""
    dimension, noise = SETTINGS[case]
    lookup = {(row["train_n"], row["loss"]): row for row in values if row["case"] == case}
    rows = []
    for n in sizes:
        cells = [f"{n:,}"]
        for loss in LOSSES:
            row = lookup[(n, loss)]
            # Number is only the shared two-line formatter: its second value is
            # explicitly a sample SD here; no scientific-report SE is reused.
            cells.append(paper.Number(row[f"{metric}_mean"], row[f"{metric}_sd"],
                                      digits=1 if metric == "epochs" else 2))
        rows.append((cells, False))
    caption = (f"Mean (sample standard deviation) across {repeats} timed replays per cell; "
               "parentheses contain SD, not standard error. Training n is per distribution.")
    return paper.table(f"D={dimension}, noise SD={noise:.2f}: {LABELS[metric]}", caption,
                       ["Train n per P/Q", *[paper.LOSS[loss] for loss in LOSSES]], rows)


def hardware_records(rows):
    """Retain every observed host/CPU/software record and its measurement count."""
    counts, records = {}, {}
    for row in rows:
        encoded = json.dumps(row["hardware"], sort_keys=True)
        counts[encoded] = counts.get(encoded, 0) + 1
        records[encoded] = row["hardware"]
    return [{"timed_fits": counts[key], "hardware": records[key]} for key in sorted(records)]


def hardware_note(records):
    counts = {}
    for record in records:
        hardware = record["hardware"]
        host = hardware.get("node", hardware.get("hostname", hardware.get("host", "host recorded in JSON")))
        cpu = hardware.get("cpu_model", hardware.get("cpu_model_name", hardware.get("model_name", "CPU recorded in JSON")))
        counts[(host, cpu)] = counts.get((host, cpu), 0) + record["timed_fits"]
    labels = [f"{host}: {cpu} ({count:,} timed fits)" for (host, cpu), count in sorted(counts.items())]
    return "Observed hardware: " + "; ".join(labels) + ". Full per-fit hardware and software metadata accompany the CSV."


def csv_file(path, rows):
    """Keep nested per-fit provenance as JSON in otherwise ordinary CSV cells."""
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple))
                          else value for key, value in row.items()} for row in rows)


def verified_original(protocol):
    """Check all original presentation hashes before constructing the new view."""
    reference = Path(protocol["reference_path"])
    marker = reference / "CONVERGENCE_COMPLETE.json"
    completed = w.read_json(marker)
    if (completed.get("status") != "complete"
            or completed.get("protocol_sha256") != protocol["reference_protocol_sha256"]
            or w.digest(reference / "protocol.json") != protocol["reference_protocol_sha256"]):
        raise ValueError("Original convergence report is not complete at the expected protocol")
    artifacts = dict(completed["artifact_sha256"])
    if not {"report.md", "report.html", "tables.tex"} <= artifacts.keys():
        raise ValueError("Original convergence report lacks required publication artifacts")
    for name, expected in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe original presentation artifact path")
        if not (reference / name).is_file() or w.digest(reference / name) != expected:
            raise ValueError(f"Original presentation artifact changed: {name}")
    return reference, marker, artifacts


def combined_report(output, protocol, tables, images, notes, original):
    """Copy the verified scientific presentation and append computation costs."""
    reference, marker, original_artifacts = original
    combined = output / "convergence_with_cost"
    combined.mkdir()
    for name in original_artifacts:
        target = combined / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(reference / name, target)
    shutil.copy2(reference / "protocol.json", combined / "original_protocol.json")
    shutil.copy2(marker, combined / "original_completion.json")
    intro = ("Computation costs below come from a separate, verified timing replay of a prespecified subset "
             "of the convergence repetitions. The scientific results above retain all original repetitions.")
    md = ["\n## Average computation cost\n", intro,
          "[Timing protocol and hardware](../cost_report.md) · [Per-setting cost summaries](../cost_summary.csv) · "
          "[CPU time and epoch tables](../cost_appendix.md)"]
    for table, image in zip(tables, images):
        md.append(f"### {table['title']}\n\n{table['caption']}\n\n![{table['title']}](../{image})")
    md.extend(notes)
    path = combined / "report.md"
    path.write_text(path.read_text() + "\n\n".join(md) + "\n")
    block = ("<section><h2>Average computation cost</h2><p>" + html.escape(intro) + "</p>"
             '<p><a href="../cost_report.html">Timing protocol and hardware</a> · '
             '<a href="../cost_summary.csv">Per-setting cost summaries</a> · '
             '<a href="../cost_appendix.html">CPU time and epoch tables</a></p></section>'
             + "".join(paper.render_table(table, "html") for table in tables)
             + "".join("<p class='notes'>" + html.escape(note) + "</p>" for note in notes))
    path = combined / "report.html"
    text = path.read_text()
    if text.count("</main>") != 1:
        raise ValueError("Original HTML report does not have one main container")
    path.write_text(text.replace("</main>", block + "</main>"))
    path = combined / "tables.tex"
    path.write_text(path.read_text() + "\n\n% Computation costs: entries are mean (sample SD), not MCSE.\n"
                    + "\n\n".join(paper.render_table(table, "tex") for table in tables) + "\n")
    w.save_json(combined / "combined_provenance.json", {
        "original_run": str(reference), "original_protocol_sha256": protocol["reference_protocol_sha256"],
        "original_completion_sha256": w.digest(marker), "original_artifact_sha256": original_artifacts,
        "cost_protocol_sha256": w.digest(output / "protocol.json"),
        "cost_results_sha256": w.digest(output / "cost_results.csv"),
        "cost_summary_sha256": w.digest(output / "cost_summary.csv"),
        "modified_copies": ["report.md", "report.html", "tables.tex"],
        "scope": "Scientific fits and source report unchanged; timing appended only to this copied presentation.",
    })


def render(output, protocol, rows):
    """Render complete per-setting timing tables and return every artifact hash."""
    output = Path(output)
    if not output.is_dir():
        raise FileNotFoundError("Prepare the computation-cost output directory first")
    if any((output / name).exists() for name in (*PRESENTATION_FILES, "figures", "convergence_with_cost")):
        raise FileExistsError("Preserve the existing cost presentation before retrying")
    rows, repeats = checked_rows(protocol, rows)
    original = verified_original(protocol)
    cfg = protocol["config"]
    cases = sorted(cfg["cases"], key=lambda case: SETTINGS[case])
    sizes = sorted(cfg["sample_sizes"])
    summaries = summarize(rows, cases, sizes)
    hardware = hardware_records(rows)
    tables = [timing_table(case, summaries, sizes, "wall_seconds", len(repeats)) for case in cases]
    appendix = [timing_table(case, summaries, sizes, metric, len(repeats))
                for metric in ("cpu_seconds", "epochs") for case in cases]
    parameters = [(list(map(str, [SETTINGS[case][0], f"{SETTINGS[case][1]:.2f}",
                                  next(row["n_parameters"] for row in summaries if row["case"] == case)])), False)
                  for case in cases]
    appendix.append(paper.table("Model size", "Trainable parameter count is constant across losses and sample sizes within each setting.",
                                ["Dimension", "Noise SD", "Parameters"], parameters))
    selected = protocol["candidates"][0]
    scientific_repeats = protocol["reference_protocol"]["config"]["convergence_repeats"]
    intro = (f"{len(rows):,} verified timing replays: {len(cases)} settings × {len(sizes)} training sizes × four losses × "
             f"{len(repeats)} repetitions. Fixed model: {paper.ARCH[selected['architecture']]} with output "
             f"2 sigmoid({selected['output_alpha']} z). Scientific convergence summaries use {scientific_repeats} repetitions; "
             f"these cost summaries use only {len(repeats)} prespecified timing repetitions ({', '.join(map(str, repeats))}).")
    submission_path = output / "submission.json"
    submission = w.read_json(submission_path) if submission_path.is_file() else {}
    concurrency = submission.get("array_concurrency")
    concurrency_note = (f"The Slurm array requested at most {concurrency} concurrent benchmark workers. "
                        if isinstance(concurrency, int) and concurrency > 0 else
                        "The requested benchmark-worker concurrency is not recorded in submission.json. ")
    notes = ["Time covers fit_ratio_mlp, including optimization, validation each epoch, early stopping and restoration of the best checkpoint. "
             "It excludes data generation, model initialization, evaluation, calibration diagnostics, artifact I/O and Slurm queueing. "
             "This is training cost per fitted model, not end-to-end pipeline time or allocated Slurm CPU-hours.",
             "Each timing task first performs an untimed one-epoch warm-up for every loss. Loss execution order rotates across tasks. "
             "Reported wall time is warmed elapsed time and CPU time is process CPU time. All timed fits use a single CPU with numerical-library threads set to one. "
             "Concurrent work and shared-node contention can affect wall time; observed hardware is recorded for every fit.",
             "Entries give arithmetic mean (sample standard deviation, ddof=1) over the timed repetitions within each dimension/noise/n/loss setting. "
             "SD measures replay-to-replay variability, not uncertainty of the mean. Timing variation reflects both data-dependent stopping epochs and execution variability. "
             "The scientific tables use Monte Carlo standard errors, so their parentheses have a different meaning.",
             "The replay retains the original data, initialization, training recipe and each loss's own minimum-validation-loss checkpoint. "
             "Before publication, the runner requires exact model-parameter, training-history and stopping-metadata agreement with the corresponding original convergence fits. "
             "Costs were measured separately; timestamps of the scientific output files were not used as training-time estimates.",
             concurrency_note + "Other work on the shared cluster may run concurrently.", hardware_note(hardware),
             "Software versions: " + json.dumps(protocol.get("versions", {}), sort_keys=True) + "."]
    csv_file(output / "cost_results.csv", rows)
    csv_file(output / "cost_summary.csv", summaries)
    w.save_json(output / "cost_hardware.json", {"hardware_records": hardware, "versions": protocol.get("versions"),
                                                "threads": protocol.get("threads"), "device": protocol.get("device")})
    images = paper.render_table_figures(tables + appendix, output)
    for extension in ("md", "html"):
        links = [("Convergence results with computation costs", f"convergence_with_cost/report.{extension}"),
                 ("CPU time, epochs and parameters", f"cost_appendix.{extension}"),
                 ("Per-setting summaries", "cost_summary.csv"), ("All timing records", "cost_results.csv"),
                 ("Hardware records", "cost_hardware.json"), ("LaTeX tables", "cost_tables.tex")]
        paper.write_document(output / f"cost_report.{extension}", "Convergence computation costs", intro, tables, notes, links,
                             table_images=images[:len(tables)] if extension == "md" else ())
        shutil.copy2(output / f"cost_report.{extension}", output / f"report.{extension}")
        paper.write_document(output / f"cost_appendix.{extension}", "Convergence computation costs: CPU time and model size",
                             intro, appendix, notes, [("Training wall time", f"cost_report.{extension}"),
                             ("Per-setting summaries", "cost_summary.csv"), ("LaTeX tables", "cost_appendix_tables.tex")],
                             table_images=images[len(tables):] if extension == "md" else ())
    for name, values in (("cost_tables.tex", tables), ("cost_appendix_tables.tex", appendix)):
        (output / name).write_text("% Requires \\usepackage{booktabs}; entries are mean (sample SD), not MCSE.\n\n"
                                   + "\n\n".join(paper.render_table(table, "tex") for table in values) + "\n")
    combined_report(output, protocol, tables, images[:len(tables)], notes, original)
    artifacts = [name for name in PRESENTATION_FILES if name != "cost_rendering_manifest.json"]
    artifacts += [str(path.relative_to(output)) for directory in ("figures", "convergence_with_cost")
                  for path in sorted((output / directory).rglob("*")) if path.is_file()]
    hashes = {name: w.digest(output / name) for name in artifacts}
    w.save_json(output / "cost_rendering_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "renderer_sha256": w.digest(Path(__file__)),
        "protocol_sha256": w.digest(output / "protocol.json"), "timed_fit_count": len(rows),
        "summary_count": len(summaries), "timing_repeats": repeats,
        "aggregation": "arithmetic mean and sample SD (ddof=1) within each case/train_n/loss",
        "original_protocol_sha256": protocol["reference_protocol_sha256"], "artifact_sha256": hashes,
        "submission_sha256": w.digest(submission_path) if submission_path.is_file() else None,
    })
    hashes["cost_rendering_manifest.json"] = w.digest(output / "cost_rendering_manifest.json")
    return hashes
