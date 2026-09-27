"""Matched KL/chi-square/JS follow-up, reusing the frozen Hellinger reference.

prepare --reference ORIGINAL --output NEW copies the selected reference fits.
fit --output NEW --task-index N trains three losses on the original task draws.
report --output NEW verifies every pair before writing descriptive comparisons.
render --reference COMPLETED --output NEW regenerates presentation from verified fits.
No stage refits Hellinger, changes the original run, or launches convergence.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from experiments.simulations import report as paper
from experiments.simulations import workflow as w
from experiments.simulations.population import SETTINGS

LOSSES = ("kl", "chisq", "js")
COMPLETE = "FOLLOWUP_COMPLETE.json"


def scientific_sources():
    return {str(path.relative_to(ROOT)): w.digest(path) for path in w.source_paths()}


def rendering_sources():
    paths = [*w.source_paths(), Path(__file__).resolve(), Path(paper.__file__).resolve()]
    return {str(path.relative_to(ROOT)): w.digest(path) for path in paths}


def verify_execution(output, protocol):
    """Jobs execute exactly the prepared snapshot, including reporting helpers."""
    if scientific_sources() != protocol["source_sha256"] or rendering_sources() != protocol["followup_source_sha256"]:
        raise ValueError("Executing source differs from prepared follow-up; use its frozen snapshot")
    for name, expected in protocol["followup_source_sha256"].items():
        path = output / "source" / name
        if not path.is_file() or w.digest(path) != expected:
            raise ValueError(f"Follow-up source snapshot changed: {name}")
    versions = {"python": w.platform.python_version(), "numpy": np.__version__, "scipy": w.scipy.__version__,
                "torch": torch.__version__, "matplotlib": w.importlib.metadata.version("matplotlib")}
    if versions != protocol["versions"] or torch.get_num_threads() != protocol["threads"]:
        raise ValueError("Execution versions/threads differ from original study")


def reference_rows(root, reference_protocol, selection):
    """Verify only the selected original configuration, preserving its metadata."""
    rows, manifests = [], {}
    for index, (case, repeat, n) in enumerate(w.task_specs(reference_protocol)):
        candidate = selection["selected"]
        rows.append(w.verify_fit(root, reference_protocol, index, case, repeat, n, candidate))
        folder = w.task_path(root, case, repeat, n, candidate)
        name = str((folder / "complete.json").relative_to(root))
        manifests[name] = w.digest(folder / "complete.json")
        if selection["fit_manifest_sha256"].get(name) != manifests[name]:
            raise ValueError("Reference fit does not match its frozen selection manifest")
    return rows, manifests


def verify_reference(output, protocol):
    root = output / "reference"
    manifest_path = output / "reference_manifest.json"
    if not manifest_path.is_file() or w.digest(manifest_path) != protocol["reference_manifest_sha256"]:
        raise ValueError("Reference manifest changed or is missing")
    manifest = w.read_json(manifest_path)
    for name, expected in manifest["records_sha256"].items():
        path = root / name
        if not path.is_file() or w.digest(path) != expected:
            raise ValueError(f"Reference record changed or is missing: {name}")
    reference_protocol = w.read_json(root / "protocol.json")
    selection, _ = w.load_frozen(root / "selection.json")
    expected_candidates = [{**selection["selected"], "loss": loss} for loss in LOSSES]
    if (reference_protocol["stage"] != "selection" or protocol["stage"] != "selection"
            or selection["selected"] != protocol["reference_candidate"]
            or protocol["candidates"] != expected_candidates
            or any(reference_protocol[key] != protocol[key]
                   for key in ("config", "source_sha256", "device", "threads", "versions"))):
        raise ValueError("Reference protocol/candidate differs from follow-up")
    rows, manifests = reference_rows(root, reference_protocol, selection)
    if manifests != manifest["fit_manifest_sha256"]:
        raise ValueError("Copied reference fit set differs from preparation")
    return reference_protocol, rows


def prepare(reference, output):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Follow-up output must be a new directory")
    original = w.read_json(reference / "protocol.json")
    selected, selected_hash = w.load_frozen(reference / "selection.json")
    if original["stage"] != "selection" or selected["protocol_sha256"] != w.digest(reference / "protocol.json"):
        raise ValueError("Reference is not a matching frozen selection study")
    for key in ("config", "source_sha256", "device", "threads", "versions"):
        if selected[key] != original[key]:
            raise ValueError(f"Reference selection/protocol disagree: {key}")
    if scientific_sources() != original["source_sha256"]:
        raise ValueError("Scientific source differs from original selection study")
    expected = min((r for r in selected["ranking"] if r["loss"] == "hellinger"),
                   key=lambda r: (r["mean_validation_brier"], w.candidate_id(r)))
    if selected["selected"] != {key: expected[key] for key in ("loss", "architecture", "output_alpha")}:
        raise ValueError("Reference candidate differs from the frozen Hellinger ranking")
    rows, manifests = reference_rows(reference, original, selected)
    if float(np.mean([row["validation_brier"] for row in rows])) != expected["mean_validation_brier"]:
        raise ValueError("Reference validation scores differ from frozen ranking")
    protocol = copy.deepcopy(original)
    protocol.update(candidates=[{**selected["selected"], "loss": loss} for loss in LOSSES],
                    reference_candidate=selected["selected"], reference_selection_sha256=selected_hash,
                    followup_source_sha256=rendering_sources(),
                    selection_rule="No new selection: matched loss comparison at the original frozen architecture/activation",
                    evaluation_role="follow-up design evidence after inspecting the original study")
    output.mkdir(parents=True)
    retained = output / "reference"
    retained.mkdir()
    records = {}
    for name in ("protocol.json", "selection.json", "selection.sha256"):
        shutil.copy2(reference / name, retained / name)
        records[name] = w.digest(retained / name)
    for name, checksum in original["source_sha256"].items():
        source = reference / "source" / name
        if w.digest(source) != checksum:
            raise ValueError(f"Original source snapshot changed: {name}")
        target = retained / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        records[f"source/{name}"] = checksum
    for case, repeat, n in w.task_specs(original):
        source = w.task_path(reference, case, repeat, n, selected["selected"])
        shutil.copytree(source, retained / source.relative_to(reference))
    w.save_json(output / "reference_manifest.json", {"records_sha256": records, "fit_manifest_sha256": manifests})
    protocol["reference_manifest_sha256"] = w.digest(output / "reference_manifest.json")
    for name in protocol["followup_source_sha256"]:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    w.save_json(output / "protocol.json", protocol)
    verify_execution(output, protocol)
    verify_reference(output, protocol)
    return output


def fit(output, task_index):
    output = Path(output).resolve()
    protocol = w.read_json(output / "protocol.json")
    verify_execution(output, protocol)
    verify_reference(output, protocol)
    w.run_task(output, protocol, task_index)


def verified_results(output):
    protocol = w.read_json(output / "protocol.json")
    verify_execution(output, protocol)
    original, reference = verify_reference(output, protocol)
    fresh = w.collect(output, protocol)
    local_fresh, _ = w.local_diagnostics(output, fresh)
    local_reference, _ = w.local_diagnostics(output / "reference", reference)
    for index, (case, repeat, n) in enumerate(w.task_specs(protocol)):
        reference_folder = w.task_path(output / "reference", case, repeat, n, protocol["reference_candidate"])
        reference_training = w.read_json(reference_folder / "training.json")
        for candidate in protocol["candidates"]:
            folder = w.task_path(output, case, repeat, n, candidate)
            training = w.read_json(folder / "training.json")
            for key in ("data_seed_vectors", "initialization_seed_vector", "model_seed"):
                if training[key] != reference_training[key]:
                    raise ValueError(f"New/reference random streams differ: {case}, {repeat}, {key}")
    local = local_reference + local_fresh
    lookup = {(r["case"], r["repeat"], r["train_n"], r["candidate"]): r for r in local if r["partition"] == "fixed"}
    rows = []
    for origin, values in (("reused_Hellinger", reference), ("new_followup", fresh)):
        for value in values:
            row = {**value, "origin": origin}
            diagnostics = lookup[(row["case"], row["repeat"], row["train_n"], row["candidate"])]
            row.update(local_gap=diagnostics["mass_weighted_absolute_gap"], c2_width=diagnostics["mass_weighted_c2_width"],
                       estimable_mass=diagnostics["evaluation_mass_with_estimable_gap"])
            rows.append(row)
    return protocol, rows, local


def paired_results(rows, cases):
    index = {(r["case"], r["repeat"], r["loss"]): r for r in rows}
    paired = []
    for loss in LOSSES:
        result = {"comparison": f"Hellinger minus {paper.LOSS[loss]}", "alternative_loss": loss}
        for metric in paper.METRICS:
            groups = []
            for case in cases:
                repeats = sorted({r["repeat"] for r in rows if r["case"] == case})
                groups.append([index[(case, repeat, "hellinger")][metric]-index[(case, repeat, loss)][metric]
                               for repeat in repeats if index[(case, repeat, "hellinger")][metric] is not None
                               and index[(case, repeat, loss)][metric] is not None])
            mean, se, count = paper.statistics(groups)
            result.update({f"{metric}_mean": mean, f"{metric}_se": se, f"{metric}_count": count})
        paired.append(result)
    return paired


def comparison_tables(protocol, cases, individual, aggregate, paired):
    """Highlight the lowest mean error per column; interval width is contextual."""
    fixed = f"Fixed architecture {paper.ARCH[protocol['reference_candidate']['architecture']]} and output activation α={protocol['reference_candidate']['output_alpha']}."
    emphasis = " Bold identifies the lowest mean in each error column (ties included), without a significance claim."
    def winner(row, metric, values):
        means = [r[f"{metric}_mean"] for r in values if r[f"{metric}_mean"] is not None]
        value = row[f"{metric}_mean"]
        best = bool(means) and value is not None and np.isclose(value, min(means), rtol=0, atol=1e-12)
        return paper.number(row, metric, bold=best)
    columns = ["Loss", "Activation α", "NN architecture", "Validation Brier", "Evaluation MSE", "Evaluation Brier", "Local gap"]
    def comparison_table(title, values, caption):
        return paper.table(title, caption + emphasis, columns,
            [([*paper.factor_cells(row), *[winner(row, m, values) for m in paper.METRICS[:4]]], False) for row in values])
    tables = [comparison_table("Matched loss comparison", aggregate, fixed + f" Equal-weight mean over {len(cases)} fixed settings. Parentheses give Monte Carlo standard errors.")]
    tables.append(paper.table("Local interval width and support", fixed + " Fixed 20-bin C.2 diagnostics; evaluation-mass-weighted summaries. Bold marks the lowest local gap; interval width and support are descriptive, without a winner ranking.",
        ["Loss", "Local gap", "C.2 width", "Estimable mass (%)"],
        [([paper.LOSS[r["loss"]], winner(r, "local_gap", aggregate), *[paper.number(r, m) for m in paper.METRICS[4:]]], False) for r in aggregate]))
    tables.append(paper.table("Paired Hellinger-minus-alternative differences", "Positive values mean a smaller reported quantity for the alternative; local gap remains descriptive. SEs use paired within-repetition differences, then equal weighting over the fixed settings; no significance decision is applied.",
        ["Comparison", "Validation Brier", "Evaluation MSE", "Evaluation Brier", "Local gap"],
        [([r["comparison"].replace(" minus ", " minus\n"), *[paper.number(r, m) for m in paper.METRICS[:4]]], False) for r in paired]))
    for case in cases:
        dimension, noise = SETTINGS[case]
        tables.append(comparison_table(f"Dimension {dimension}; noise SD {noise:.2f}", [r for r in individual if r["case"] == case],
            fixed + " Mean (standard error) across matched training repetitions."))
    return tables


def insert_local_gap_definition(path, cfg, supported_mass):
    """Document the existing fixed-bin diagnostic without recomputing any results.

    Markers allow presentation-only replays to replace this section idempotently.
    ``supported_mass`` is the minimum estimable evaluation mass across fixed-bin
    fits, so a value of one supports the statement that every fit has full mass.
    """
    path = Path(path)
    if path.suffix not in (".md", ".html"):
        raise ValueError("The local-gap definition supports Markdown and HTML reports")
    ncal, neval = cfg["calibration_n"], cfg["evaluation_n"]
    repeats, settings = cfg["selection_repeats"], len(cfg["cases"])
    if np.isclose(supported_mass, 1.0, rtol=0, atol=1e-12):
        support = "Supported evaluation mass was 100% for every fitted model in this report, so the denominator was 1."
    else:
        support = (f"The minimum supported evaluation mass across fitted models was {100*supported_mass:.6f}%; "
                   "the denominator normalizes each gap over its own supported mass.")
    interpretation = ("The statistic uses observed P/Q labels and network predictions, without analytic truth. "
                      "CI endpoints do not enter its calculation. Calibration and evaluation sampling noise "
                      "contribute to the gap, so even a calibrated model can have a positive value. "
                      "It measures binned calibration discrepancy, not pointwise RDR error.")
    aggregation = (f"Each setting's table averages the gap over {repeats} matched repetitions; "
                   f"the overall table then averages equally across the {settings} settings.")
    if path.suffix == ".md":
        body = rf"""### Definition of the local gap

For each fitted network $\hat r$, divide the score range $[0,2]$ into 20 equal-width bins $I_j$. These fixed score cutoffs define model-dependent input regions $A_j=\{{x:\hat r(x)\in I_j\}}$. Let $M=(P+Q)/2$.

The independent calibration sample contains {ncal:,} observations from each of $P$ and $Q$. If their counts in $A_j$ are $k_{{Pj}}$ and $k_{{Qj}}$, the count-based estimate is

$$
\hat\theta_j=\frac{{2k_{{Pj}}}}{{k_{{Pj}}+k_{{Qj}}}}.
$$

It estimates the population cell-average RDR $\theta_j=\mathbb{{E}}_M[r_0(X)\mid X\in A_j]=2P(A_j)/[P(A_j)+Q(A_j)]$, where $r_0=dP/dM$.

Use a separate evaluation sample with {neval:,} observations from each distribution. With $N_j$ evaluation observations in $A_j$, compute its mean neural prediction and empirical mixture mass:

$$
\bar r_j=\frac{{1}}{{N_j}}\sum_{{i:X_i\in A_j}}\hat r(X_i),
\qquad w_j=\frac{{N_j}}{{2n_{{\mathrm{{eval}}}}}},
\qquad n_{{\mathrm{{eval}}}}={neval}.
$$

Let $J=\{{j:k_{{Pj}}+k_{{Qj}}>0,\ N_j>0\}}$ contain bins with both calibration and evaluation observations. The reported local gap is

$$
\operatorname{{Local\ gap}}=
\frac{{\sum_{{j\in J}}w_j\left|\bar r_j-\hat\theta_j\right|}}
{{\sum_{{j\in J}}w_j}}.
$$

{support}

{aggregation}

{interpretation}"""
        anchor = "This follow-up was chosen"
    else:
        body = f"""<section id="local-gap-definition"><h2>Definition of the local gap</h2>
<p>For each fitted network r̂, divide the score range [0, 2] into 20 equal-width bins I<sub>j</sub>. These fixed score cutoffs define model-dependent input regions A<sub>j</sub> = {{x: r̂(x) ∈ I<sub>j</sub>}}. Let M = (P + Q)/2.</p>
<p>The independent calibration sample contains {ncal:,} observations from each of P and Q. If their counts in A<sub>j</sub> are k<sub>Pj</sub> and k<sub>Qj</sub>, the count-based estimate is</p>
<p class="formula">θ̂<sub>j</sub> = 2k<sub>Pj</sub> / (k<sub>Pj</sub> + k<sub>Qj</sub>).</p>
<p>It estimates the population cell-average RDR θ<sub>j</sub> = E<sub>M</sub>[r<sub>0</sub>(X) | X ∈ A<sub>j</sub>] = 2P(A<sub>j</sub>) / [P(A<sub>j</sub>) + Q(A<sub>j</sub>)], where r<sub>0</sub> = dP/dM.</p>
<p>Use a separate evaluation sample with {neval:,} observations from each distribution. With N<sub>j</sub> evaluation observations in A<sub>j</sub>, compute its mean neural prediction and empirical mixture mass:</p>
<p class="formula">r̄<sub>j</sub> = (1/N<sub>j</sub>) ∑<sub>i: Xᵢ ∈ Aⱼ</sub> r̂(Xᵢ), &nbsp; w<sub>j</sub> = N<sub>j</sub> / (2n<sub>eval</sub>), &nbsp; n<sub>eval</sub> = {neval:,}.</p>
<p>Let J = {{j: k<sub>Pj</sub> + k<sub>Qj</sub> &gt; 0, N<sub>j</sub> &gt; 0}} contain bins with both calibration and evaluation observations. The reported local gap is</p>
<p class="formula"><strong>Local gap = [∑<sub>j ∈ J</sub> w<sub>j</sub> |r̄<sub>j</sub> − θ̂<sub>j</sub>|] / [∑<sub>j ∈ J</sub> w<sub>j</sub>].</strong></p>
<p>{support}</p><p>{aggregation}</p><p>{interpretation}</p></section>"""
        anchor = "<p class='notes'>This follow-up was chosen"
    start, end = "<!-- local-gap-definition:start -->", "<!-- local-gap-definition:end -->"
    block = f"{start}\n{body}\n{end}"
    document = path.read_text()
    if start in document or end in document:
        if document.count(start) != 1 or document.count(end) != 1 or document.index(start) > document.index(end):
            raise ValueError("Malformed local-gap definition markers")
        left, right = document.index(start), document.index(end) + len(end)
        document = document[:left] + block + document[right:]
    else:
        if anchor not in document:
            raise ValueError("Could not find the follow-up report notes")
        document = document.replace(anchor, block + "\n\n" + anchor, 1)
    path.write_text(document)
    return path


def write_presentation(output, protocol, rows, local):
    """Write the same tables for freshly completed fits and archived evidence."""
    filenames = ("metrics.csv", "calibration.csv", "summary.csv", "aggregation.csv", "paired_comparisons.csv", "report.md", "report.html", "tables.tex")
    if any((output/name).exists() for name in filenames) or (output/"figures").exists():
        raise FileExistsError("Partial report exists; preserve it before retrying")
    cases = sorted(protocol["config"]["cases"], key=lambda case: SETTINGS[case])
    candidates = [protocol["reference_candidate"], *protocol["candidates"]]
    summaries = paper.summarize(rows, cases, candidates)
    # The publication helper assumes multiple settings; distinguish its final
    # aggregate block explicitly for a one-setting executable smoke check.
    for row in summaries[-len(candidates):]:
        row.update(case="equal_setting_average", dimension=None, noise_sd=None)
    individual, aggregate = summaries[:-len(candidates)], summaries[-len(candidates):]
    paired = paired_results(rows, cases)
    tables = comparison_tables(protocol, cases, individual, aggregate, paired)
    cfg = protocol["config"]
    intro = (f"{len(w.task_specs(protocol))*3} new fits compare KL, chi-square and JS with {len(w.task_specs(protocol))} reused Hellinger fits. "
             "Every comparison uses identical case/repetition train, validation, calibration and evaluation draws, and identical initial network parameters.")
    notes = ["This follow-up was chosen after inspecting the original architecture/activation study. It is design evidence, not a new independent final test or an architecture search for every loss. The original Hellinger configuration remains a reference; this report does not freeze a new model or launch convergence.",
             f"Each distribution supplies {cfg['train_n']:,} training, {cfg['validation_n']:,} validation, {cfg['calibration_n']:,} calibration and {cfg['evaluation_n']:,} evaluation observations. Training uses the original AdamW/gradient-clipping/scheduler recipe. Each objective restores its own minimum-validation-loss checkpoint; cross-loss validation Brier is reported afterward.",
             "Equal-setting SE = sqrt(sum_k s_k²/n_k)/K. Paired tables use variances of Hellinger-minus-alternative differences within each setting, not independent-model SEs. With one repetition, SE is unavailable.",
             "Local gaps use observed labels and neural scores, include sampling error, and are normalized over estimable evaluation mass. C.2 width weights all evaluation mass. Bands target population cell-average RDR, not individual truth or uncertainty in neural means; fixed score cutpoints induce different input regions across models. No simultaneous guarantee across the model comparison is asserted.",
             "Chi-square empirical loss equals four times balanced Brier minus three. JS equals balanced P-versus-Q binary cross-entropy up to a constant. B8-Res64 denotes a learned linear min(D,8)-dimensional bottleneck followed by the width-64 residual network."]
    for name, values in (("metrics.csv", rows), ("calibration.csv", local), ("summary.csv", individual), ("aggregation.csv", aggregate), ("paired_comparisons.csv", paired)):
        with (output/name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    table_images = paper.render_table_figures(tables, output)
    supported_mass = min(row["evaluation_mass_with_estimable_gap"] for row in local if row["partition"] == "fixed")
    for extension in ("md", "html"):
        paper.write_document(output/f"report.{extension}", "Matched loss follow-up", intro, tables, notes,
            [("All setting summaries", "summary.csv"), ("Paired comparisons", "paired_comparisons.csv"), ("LaTeX tables", "tables.tex")],
            table_images=table_images if extension == "md" else ())
        insert_local_gap_definition(output/f"report.{extension}", cfg, supported_mass)
    (output/"tables.tex").write_text("% Requires \\usepackage{booktabs}.\n\n"+"\n\n".join(paper.render_table(t,"tex") for t in tables)+"\n")
    return {name: w.digest(output/name) for name in [*filenames,
            *[str(path.relative_to(output)) for path in sorted((output/"figures").iterdir())]]}


def report(output):
    output = Path(output).resolve()
    protocol, rows, local = verified_results(output)
    marker = output / COMPLETE
    if marker.exists():
        for name, expected in w.read_json(marker)["report_sha256"].items():
            if not (output/name).is_file() or w.digest(output/name) != expected:
                raise ValueError(f"Completed report changed or is missing: {name}")
        return output / "report.md"
    hashes = write_presentation(output, protocol, rows, local)
    # A completion marker is created only after every reference and new fit was
    # verified and every report artifact was written successfully.
    w.save_json(marker, {"status": "complete", "new_fits": len(rows)-len(w.task_specs(protocol)),
        "reused_reference_fits": len(w.task_specs(protocol)), "protocol_sha256": w.digest(output/"protocol.json"),
        "report_sha256": hashes,
        "reference_manifest_sha256": protocol["reference_manifest_sha256"],
        "followup_source_sha256": protocol["followup_source_sha256"]})
    return output / "report.md"


def archived_results(reference):
    """Verify frozen code before importing it in a separate, isolated process.

    Rendering may change without invalidating completed fits. Their scientific
    verification must still use their original code, package versions and thread
    count. The child only reads artifacts and returns verified scalar summaries.
    """
    protocol = w.read_json(reference / "protocol.json")
    marker = w.read_json(reference / COMPLETE)
    if (marker.get("status") != "complete"
            or marker["protocol_sha256"] != w.digest(reference / "protocol.json")
            or marker["reference_manifest_sha256"] != protocol["reference_manifest_sha256"]
            or marker["followup_source_sha256"] != protocol["followup_source_sha256"]):
        raise ValueError("Completed follow-up does not match its frozen protocol")
    sources = protocol["followup_source_sha256"]
    expected_names = set(protocol["source_sha256"]) | {
        "experiments/simulations/loss_followup.py", "experiments/simulations/report.py"}
    if (set(sources) != expected_names
            or any(sources[name] != value for name, value in protocol["source_sha256"].items())):
        raise ValueError("Archived source closure differs from its scientific protocol")
    actual_python = {str(path.relative_to(reference / "source"))
                     for path in (reference / "source").rglob("*.py")}
    if actual_python != {name for name in sources if name.endswith(".py")}:
        raise ValueError("Archived Python source closure contains missing or unexpected files")
    consumed = {"protocol.json": marker["protocol_sha256"], COMPLETE: w.digest(reference / COMPLETE),
                "reference_manifest.json": protocol["reference_manifest_sha256"],
                **{f"source/{name}": value for name, value in sources.items()},
                **marker["report_sha256"]}

    def check_inputs():
        for name, expected in consumed.items():
            path = (reference / name).resolve()
            if reference not in path.parents or not path.is_file() or w.digest(path) != expected:
                raise ValueError(f"Archived artifact changed, missing, or outside run: {name}")

    check_inputs()
    # -E ignores PYTHONPATH while retaining installed user-site dependencies,
    # which HPC environments often need. The checkout/cwd is removed below;
    # -B prevents bytecode writes into the frozen scientific source tree.
    program = """import json, sys
from pathlib import Path
sys.path = [sys.argv[1], *[path for path in sys.path
                         if path and Path(path).resolve() != Path.cwd()]]
from experiments.simulations import loss_followup as f
root = Path(sys.argv[2])
protocol = f.w.read_json(root / 'protocol.json')
f.torch.set_num_threads(protocol['threads'])
f.torch.use_deterministic_algorithms(True)
protocol, rows, local = f.verified_results(root)
consumed = {}
reference_manifest = f.w.read_json(root / 'reference_manifest.json')
consumed.update({'reference/' + name: checksum for name, checksum
                 in reference_manifest['records_sha256'].items()})
for origin, current in ((root, protocol), (root / 'reference', f.w.read_json(root / 'reference/protocol.json'))):
    candidates = protocol['candidates'] if origin == root else [protocol['reference_candidate']]
    for case, repeat, n in f.w.task_specs(current):
        for candidate in candidates:
            folder = f.w.task_path(origin, case, repeat, n, candidate)
            consumed[str((folder / 'complete.json').relative_to(root))] = f.w.digest(folder / 'complete.json')
            consumed.update({str((folder / name).relative_to(root)): checksum
                             for name, checksum in f.w.read_json(folder / 'complete.json').items()})
json.dump({'protocol': protocol, 'rows': rows, 'local': local, 'consumed_sha256': consumed},
          sys.stdout, allow_nan=False)
"""
    completed = subprocess.run([sys.executable, "-E", "-B", "-c", program,
                               str(reference / "source"), str(reference)],
                              cwd=ROOT, capture_output=True, text=True, timeout=180)
    if completed.returncode:
        raise ValueError("Archived fit verification failed:\n" + completed.stderr.strip())
    result = json.loads(completed.stdout)
    if result["protocol"] != protocol:
        raise ValueError("Archived protocol changed during verification")
    consumed.update(result["consumed_sha256"])
    check_inputs()
    return protocol, result["rows"], result["local"], consumed


def render(reference, output):
    """Render completed evidence into a new directory without refitting."""
    reference, output = Path(reference).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Presentation output must be a new directory")
    if reference in output.parents:
        raise ValueError("Presentation output must be outside the archived run")
    protocol, rows, local, consumed = archived_results(reference)
    output.mkdir(parents=True)
    hashes = write_presentation(output, protocol, rows, local)
    w.save_json(output / "rendering_manifest.json", {
        "status": "complete", "role": "presentation_only", "run": str(reference),
        "verified_fits": len(rows), "protocol_sha256": consumed["protocol.json"],
        "consumed_sha256": consumed, "rendering_source_sha256": rendering_sources(),
        "scientific_versions": protocol["versions"],
        "rendering_versions": {"python": w.platform.python_version(), "numpy": np.__version__,
                               "matplotlib": w.importlib.metadata.version("matplotlib")},
        "outputs_sha256": hashes})
    return output / "report.md"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "fit", "report", "render"))
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    if args.stage in ("prepare", "render"):
        if args.reference is None:
            parser.error(f"{args.stage} requires --reference")
        result = (prepare if args.stage == "prepare" else render)(args.reference, args.output)
    elif args.stage == "fit":
        if args.task_index is None:
            parser.error("fit requires --task-index")
        result = fit(args.output, args.task_index)
    else:
        result = report(args.output)
    print(result if result is not None else "Task complete")


if __name__ == "__main__":
    main()
