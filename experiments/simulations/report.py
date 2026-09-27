"""Render verified selection results as compact paper tables; never fit models.

Usage: python experiments/simulations/report.py --run RUN --output NEW_DIRECTORY
The output directory must not exist. Every scientific input remains unchanged.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from experiments.simulations import workflow as w
from experiments.simulations.population import SETTINGS

ARCH = {"baseline32": "MLP32", "wide64": "MLP64", "deep64": "Deep64",
        "residual64": "Res64", "bottleneck8_residual64": "B8-Res64"}
LOSS = {"hellinger": "Hellinger", "kl": "KL", "chisq": "Chi-square", "js": "JS"}
METRICS = ("validation_brier", "evaluation_mse", "evaluation_brier", "local_gap", "c2_width", "estimable_mass")
LABELS = ("Validation Brier", "Evaluation MSE", "Evaluation Brier", "Local gap", "C.2 width", "Estimable mass (%)")


@dataclass
class Number:
    mean: float | None
    se: float | None
    digits: int = 4
    bold: bool = False


def tex_escape(text):
    return (str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_")
            .replace("%", r"\%").replace("&", r"\&").replace("α", r"$\alpha$")
            .replace("–", "--").replace("—", "---"))


def cell(value, style, bold=False):
    if isinstance(value, Number):
        bold = bold or value.bold
        top = "Unavailable" if value.mean is None else f"{value.mean:.{value.digits}f}"
        bottom = "SE unavailable" if value.se is None else f"({value.se:.{value.digits}f})"
        if style == "tex":
            text = r"\shortstack{" + top + r"\\" + bottom + "}"
        elif style == "html":
            text = f'{top}<small class="se">{bottom}</small>'
        else:
            text = top + "<br>" + bottom
    else:
        text = tex_escape(value) if style == "tex" else html.escape(str(value))
    if bold:
        return (r"\textbf{" + text + "}") if style == "tex" else (f"<strong>{text}</strong>" if style == "html" else f"**{text}**")
    return text


def table(title, caption, headers, rows):
    """Rows are (cells, selected-row flag); cells may carry individual bold flags."""
    return {"title": title, "caption": caption, "headers": headers, "rows": rows}


def render_table(data, style):
    headers, rows = data["headers"], data["rows"]
    if style == "tex":
        label = re.sub(r"[^a-z0-9]+", "-", data["title"].lower()).strip("-")
        lines = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
                 r"\caption{" + tex_escape(data["title"] + ". " + data["caption"]) + "}",
                 r"\label{tab:sim-" + label + "}",
                 r"\begin{tabular}{" + "l" + "c"*(len(headers)-1) + "}", r"\toprule",
                 " & ".join(tex_escape(h) for h in headers) + r" \\", r"\midrule"]
        lines += [" & ".join(cell(v, style, strong) for v in values) + r" \\" for values, strong in rows]
        return "\n".join(lines + [r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    if style == "html":
        head = "".join(f"<th scope='col'>{html.escape(h)}</th>" for h in headers)
        body = "".join("<tr>" + "".join(f"<td>{cell(v, style, strong)}</td>" for v in values) + "</tr>" for values, strong in rows)
        return f"<section><h2>{html.escape(data['title'])}</h2><p class='caption'>{html.escape(data['caption'])}</p><div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div></section>"
    lines = [f"### {data['title']}", "", data["caption"], "", "| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(v, style, strong) for v in values) + " |" for values, strong in rows]
    return "\n".join(lines)


def verified_inputs(run):
    protocol = w.read_json(run / "protocol.json")
    if protocol["stage"] != "selection":
        raise ValueError("This renderer requires a completed model-selection run")
    rows = w.collect(run, protocol)
    selection, selection_hash = w.load_frozen(run / "selection.json")
    if selection["protocol_sha256"] != w.digest(run / "protocol.json"):
        raise ValueError("Frozen selection does not match the protocol")
    for name in ("config", "source_sha256", "device", "threads", "versions"):
        if selection[name] != protocol[name]:
            raise ValueError(f"Frozen selection disagrees with protocol: {name}")
    ranking = [{**candidate, "mean_validation_brier": float(np.mean([
        row["validation_brier"] for row in rows if row["candidate"] == w.candidate_id(candidate)]))}
        for candidate in protocol["candidates"]]
    ranking.sort(key=lambda row: (row["mean_validation_brier"], w.candidate_id(row)))
    if selection["ranking"] != ranking or selection["fit_count"] != len(rows):
        raise ValueError("Frozen ranking or fit count differs from verified results")
    best = next(row for row in ranking if row["loss"] == "hellinger")
    if selection["selected"] != {key: best[key] for key in ("loss", "architecture", "output_alpha")}:
        raise ValueError("Frozen candidate does not match the Hellinger-constrained validation rule")
    consumed = {name: w.digest(run / name) for name in ("protocol.json", "selection.json", "selection.sha256")}
    manifests = {}
    for row in rows:
        folder = w.task_path(run, row["case"], row["repeat"], row["train_n"], row)
        name = str((folder / "complete.json").relative_to(run))
        manifests[name] = w.digest(folder / "complete.json")
        for filename, checksum in w.read_json(folder / "complete.json").items():
            consumed[str((folder / filename).relative_to(run))] = checksum
    if selection["fit_manifest_sha256"] != manifests:
        raise ValueError("Frozen fit-manifest hashes differ from verified artifacts")
    consumed.update(manifests)
    consumed.update({f"source/{name}": checksum for name, checksum in protocol["source_sha256"].items()})
    local, _ = w.local_diagnostics(run, rows)
    lookup = {(r["case"], r["repeat"], r["train_n"], r["candidate"]): r for r in local if r["partition"] == "fixed"}
    for row in rows:
        values = lookup[(row["case"], row["repeat"], row["train_n"], row["candidate"])]
        row.update(local_gap=values["mass_weighted_absolute_gap"], c2_width=values["mass_weighted_c2_width"],
                   estimable_mass=values["evaluation_mass_with_estimable_gap"])
    return protocol, selection, rows, consumed, selection_hash


def statistics(groups):
    """Equal fixed-setting means; only within-setting Monte Carlo variability.

    SE = sqrt(sum_k sample_variance_k/n_k)/K. A missing setting or a group with
    fewer than two observations cannot provide this complete-grid mean or SE.
    """
    arrays = [np.array([v for v in group if v is not None], dtype=float) for group in groups]
    if not arrays or any(not len(a) for a in arrays):
        return None, None, sum(map(len, arrays))
    mean = float(np.mean([a.mean() for a in arrays]))
    se = float(np.sqrt(sum(a.var(ddof=1)/len(a) for a in arrays))/len(arrays)) if all(len(a)>1 for a in arrays) else None
    return mean, se, sum(map(len, arrays))


def summarize(rows, cases, candidates):
    output = []
    for selected_cases in [[case] for case in cases] + [cases]:
        for candidate in candidates:
            name = w.candidate_id(candidate)
            per_case = [[r for r in rows if r["case"] == case and r["candidate"] == name] for case in selected_cases]
            dimension, noise = SETTINGS[selected_cases[0]] if len(selected_cases)==1 else (None, None)
            row = {"case": selected_cases[0] if len(selected_cases)==1 else "equal_setting_average",
                   "dimension": dimension, "noise_sd": noise, **candidate, "architecture_label": ARCH[candidate["architecture"]],
                   "candidate": name, "settings": len(selected_cases), "fits": sum(map(len, per_case))}
            for metric in METRICS:
                mean, se, count = statistics([[r[metric] for r in group] for group in per_case])
                row.update({f"{metric}_mean": mean, f"{metric}_se": se, f"{metric}_count": count})
            output.append(row)
    return output


def number(row, metric, bold=False):
    scale = 100 if metric == "estimable_mass" else 1
    mean, se = row[f"{metric}_mean"], row[f"{metric}_se"]
    return Number(None if mean is None else scale*mean, None if se is None else scale*se,
                  5 if metric == "validation_brier" else 3 if metric == "estimable_mass" else 4, bold)


def factor_cells(candidate):
    return [LOSS[candidate["loss"]], str(candidate["output_alpha"]), ARCH[candidate["architecture"]]]


def main_tables(summary, cases, selected):
    name = w.candidate_id(selected)
    by_case = {(r["case"], r["candidate"]): r for r in summary}
    overall = {r["candidate"]: r for r in summary if r["case"] == "equal_setting_average"}
    selected_rows = [by_case[(case, name)] for case in cases]
    tables = [table("Selected Hellinger configuration", "Mean (standard error) across ten independent training repetitions within each setting. Loss, activation and architecture are held fixed across settings.",
        ["Dimension", "Noise SD", "Loss", "Activation α", "NN architecture", "Evaluation MSE", "Evaluation Brier"],
        [([r["dimension"], f'{r["noise_sd"]:.2f}', *factor_cells(selected), number(r, "evaluation_mse"), number(r, "evaluation_brier")], False) for r in selected_rows])]
    slices = [("Loss comparison", [r for r in overall.values() if r["architecture"] == "baseline32" and r["output_alpha"] == 2],
               "Architecture MLP32 and activation α=2 are fixed."),
              ("Activation comparison", [r for r in overall.values() if r["loss"] == "hellinger" and r["architecture"] == selected["architecture"]],
               f"Hellinger loss and architecture {ARCH[selected['architecture']]} are fixed."),
              ("Architecture comparison", [r for r in overall.values() if r["loss"] == "hellinger" and r["output_alpha"] == selected["output_alpha"]],
               f"Hellinger loss and activation α={selected['output_alpha']} are fixed.")]
    for title, values, fixed in slices:
        caption = fixed + f" Equal-weight average over the {len(cases)} dimension/noise settings above. Entries are mean (standard error); bold marks the frozen selected configuration."
        tables.append(table(title, caption, ["Loss", "Activation α", "NN architecture", "Validation Brier", "Evaluation MSE", "Evaluation Brier", "Local gap"],
            [([*factor_cells(r), *[number(r, metric) for metric in METRICS[:4]]], r["candidate"] == name) for r in values]))
    tables.append(table("Local diagnostics for the selected configuration", "Fixed 20-bin diagnostics. Local gap is the evaluation-mass-weighted absolute difference between neural and calibrated cell means, normalized over cells with an estimable gap. C.2 width is weighted over all evaluation mass. Mean (standard error) across repetitions.",
        ["Dimension", "Noise SD", "Local gap", "C.2 width", "Estimable mass (%)"],
        [([r["dimension"], f'{r["noise_sd"]:.2f}', *[number(r, metric) for metric in METRICS[3:]]], False) for r in selected_rows]))
    return tables


def appendix_tables(summary, cases, selected, protocol):
    index = {(r["case"], r["candidate"]): r for r in summary}
    tables, name = [], w.candidate_id(selected)
    for case in cases:
        dimension, noise = SETTINGS[case]
        prefix = f"Dimension {dimension}, noise SD {noise:.2f}"
        losses = [index[(case, w.candidate_id(c))] for c in protocol["candidates"] if c["architecture"] == "baseline32" and c["output_alpha"] == 2]
        tables.append(table(prefix + ": loss comparison", "MLP32; activation α=2. Mean (standard error) across training repetitions. Only the frozen selected configuration is bold.",
            ["Loss", "Activation α", "NN architecture", "Validation Brier", "Evaluation MSE", "Evaluation Brier", "Local gap"],
            [([*factor_cells(r), *[number(r, metric) for metric in METRICS[:4]]], r["candidate"] == name) for r in losses]))
        for metric, label in zip(METRICS, LABELS):
            matrix = []
            for architecture in protocol["config"]["architectures"]:
                cells = [ARCH[architecture]]
                for alpha in sorted(protocol["config"]["output_alphas"]):
                    candidate = w.candidate_id(dict(loss="hellinger", architecture=architecture, output_alpha=alpha))
                    cells.append(number(index[(case, candidate)], metric, candidate == name))
                matrix.append((cells, False))
            tables.append(table(prefix + ": " + label, "Hellinger loss. Columns give output sigmoid slope α; rows give NN architecture. Mean (standard error). Bold identifies the global frozen configuration only.",
                ["NN architecture", *[f"α={alpha}" for alpha in sorted(protocol["config"]["output_alphas"])]], matrix))
    return tables


CSS = """body{margin:0;background:#f2f3f4;color:#18202b;font:16px/1.55 Georgia,serif}main{max-width:1120px;margin:32px auto;padding:38px 46px;background:white;box-shadow:0 2px 18px #0001}h1{font-size:30px;line-height:1.2;margin:0 0 16px}h2{font:600 19px/1.3 system-ui,sans-serif;margin:30px 0 6px}.caption,.notes{font-size:13px;color:#46515d}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font:13px/1.3 system-ui,sans-serif;font-variant-numeric:tabular-nums;margin:12px 0 25px}th{padding:9px 8px;border-top:2px solid #263645;border-bottom:1px solid #6d7884;text-align:center}td{padding:8px;text-align:center;white-space:nowrap;border-bottom:1px solid #e5e8eb}td:first-child,th:first-child{text-align:left}tbody tr:last-child td{border-bottom:2px solid #263645}.se{display:block;font-size:11px;color:#65717d;font-weight:400;margin-top:3px}a{color:#175a84}img{max-width:100%;height:auto}nav{font:13px system-ui,sans-serif;margin:20px 0}section{break-inside:avoid}.provenance{font:12px/1.5 monospace;overflow-wrap:anywhere;color:#65717d}@media print{body{background:white}main{box-shadow:none;margin:0;padding:0;max-width:none}nav{display:none}table{font-size:9pt}.se{font-size:8pt}th,td{padding:5px}h2{font-size:14px}.caption,.notes{font-size:10px}@page{size:A4 landscape;margin:16mm}}"""


def render_table_figures(tables, output):
    """Render compact booktabs-style tables for Markdown and paper figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = output / "figures"
    destination.mkdir(exist_ok=True)
    paths = []
    for index, data in enumerate(tables, 1):
        def display(value):
            if not isinstance(value, Number):
                return str(value)
            mean = "Unavailable" if value.mean is None else f"{value.mean:.{value.digits}f}"
            se = "SE unavailable" if value.se is None else f"({value.se:.{value.digits}f})"
            return mean + "\n" + se

        headers = [h.replace(" ", "\n", 1) if " " in h else h for h in data["headers"]]
        values = [[display(value) for value in row] for row, _ in data["rows"]]
        widths = [1.1 if h in ("Loss", "NN architecture") else .82 if h in
                  ("Dimension", "Noise SD", "Activation α") else 1.28 for h in data["headers"]]
        width = sum(widths)
        height = .55 + .52 * len(values)
        with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 10}):
            fig, ax = plt.subplots(figsize=(width, height))
            fig.subplots_adjust(left=.012, right=.988, bottom=.035, top=.965)
            ax.axis("off")
            artist = ax.table(cellText=values, colLabels=headers, cellLoc="center",
                              colWidths=np.array(widths)/width, bbox=[0, 0, 1, 1])
            artist.auto_set_font_size(False)
            artist.set_fontsize(10)
            for (row, column), entry in artist.get_celld().items():
                entry.set_facecolor("white")
                entry.set_edgecolor("#24313b")
                entry.visible_edges = "TB" if row == 0 else "B" if row == len(values) else ""
                entry.set_linewidth(1.1 if row in (0, len(values)) else .6)
                value = data["rows"][row-1][0][column] if row else None
                if row == 0 or data["rows"][row-1][1] or (isinstance(value, Number) and value.bold):
                    entry.get_text().set_weight("bold")
                entry.get_text().set_linespacing(1.35)
            relative = Path("figures") / f"table_{index:02d}.png"
            fig.savefig(output / relative, dpi=200, facecolor="white")
            fig.savefig((output / relative).with_suffix(".pdf"), facecolor="white",
                        metadata={"CreationDate": None, "ModDate": None})
            plt.close(fig)
            paths.append(relative.as_posix())
    return paths


def write_document(path, title, intro, tables, notes, links, figures=(), table_images=()):
    style = "html" if path.suffix == ".html" else "md"
    if style == "html":
        body = f"<h1>{html.escape(title)}</h1><p>{html.escape(intro)}</p><nav>" + " · ".join(f'<a href="{target}">{html.escape(label)}</a>' for label, target in links) + "</nav>"
        body += "".join(render_table(t, style) for t in tables)
        body += "".join(f"<p class='notes'>{html.escape(note)}</p>" for note in notes)
        body += "".join(f"<section><h2>{html.escape(label)}</h2><img src='{target}' alt='{html.escape(label)}'></section>" for label, target in figures)
        text = "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>" + html.escape(title) + "</title><style>" + CSS + "</style></head><body><main>" + body + "</main></body></html>\n"
    else:
        parts = [f"# {title}", intro, " · ".join(f"[{label}]({target})" for label, target in links)]
        for index, t in enumerate(tables):
            if table_images:
                parts.append(f"### {t['title']}\n\n{t['caption']}\n\n![{t['title']}]({table_images[index]})")
            else:
                parts.append(render_table(t, style))
        parts += list(notes)
        parts += [f"### {label}\n\n![{label}]({target})" for label, target in figures]
        text = "\n\n".join(parts) + "\n"
    path.write_text(text)


def render(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Output must be a new directory; existing results are never overwritten")
    protocol, selection, rows, consumed, selection_hash = verified_inputs(run)
    cases = sorted(protocol["config"]["cases"], key=lambda case: SETTINGS[case])
    if len(cases) != 5 or len(protocol["candidates"]) != 23 or protocol["config"]["selection_repeats"] != 10:
        raise ValueError("Paper renderer expects the complete five-setting, 23-candidate, ten-repetition design")
    summary = summarize(rows, cases, protocol["candidates"])
    selected, main = selection["selected"], main_tables(summary, cases, selection["selected"])
    supplement = appendix_tables(summary, cases, selected, protocol)
    output.mkdir(parents=True)
    for filename, values in (("summary.csv", [r for r in summary if r["case"] != "equal_setting_average"]),
                             ("aggregation.csv", [r for r in summary if r["case"] == "equal_setting_average"])):
        with (output / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    (output / "tables.tex").write_text("% Requires \\usepackage{booktabs}; mean (SE), not confidence intervals.\n\n" + "\n\n".join(render_table(t, "tex") for t in main) + "\n")
    table_images = render_table_figures(main, output)
    figures = []
    for case in cases:
        source = run / "figures" / f"calibration_{case}_fixed.png"
        if source.is_file():
            (output / "figures").mkdir(exist_ok=True)
            target = Path("figures") / source.name
            shutil.copy2(source, output / target)
            consumed[str(source.relative_to(run))] = w.digest(source)
            dimension, noise = SETTINGS[case]
            figures.append((f"Selected configuration: dimension {dimension}, noise SD {noise:.2f}; repeat zero", target.as_posix()))
    intro = f"Paired model selection over five fixed Gaussian-mixture settings, 23 candidates and ten independent repetitions per setting (1,150 fits). The frozen Hellinger configuration is {ARCH[selected['architecture']]} with output 2 sigmoid({selected['output_alpha']} z). Selection uses mean validation Brier only."
    cfg = protocol["config"]
    notes = ["Entries show a mean with its Monte Carlo standard error on the second line. For an equal-weight average across K fixed settings, SE = sqrt(sum_k s_k²/n_k)/K, using within-setting repetition variances; variability between the fixed settings is not included. These SEs are not confidence intervals or paired significance tests.",
             "The final Hellinger configuration is chosen jointly from all 20 activation–architecture combinations. The other losses are compared at MLP32 and α=2. Conditional tables do not establish a best loss at every architecture. Bold denotes the exact frozen configuration only, not a significant advantage.",
             f"Evaluation MSE uses analytic RDR truth. Balanced Brier uses observed P/Q labels and prediction r/2. Each distribution supplies {cfg['train_n']:,} training, {cfg['validation_n']:,} validation, {cfg['calibration_n']:,} independent calibration and {cfg['evaluation_n']:,} independent evaluation observations. Evaluation results in this selection stage are design evidence, not final evidence for a subsequently chosen procedure.",
             "Local gaps use labels and neural scores without oracle truth; they include calibration/evaluation sampling error. Gaps are normalized over estimable evaluation mass, with available-fit counts and support retained in the CSVs. C.2 bands are simultaneous across 20 fixed score bins for each frozen model; they target population cell-average RDR, not individual RDR or the neural prediction. Neural means have sampling uncertainty. Regions differ across networks. No guarantee is asserted simultaneously across model selection.",
             "B8-Res64 uses a learned linear bottleneck of min(D,8) dimensions before the width-64 residual network. Full architecture definitions and the complete joint grid appear in the supplement."]
    provenance = f"Verified protocol SHA-256: {w.digest(run/'protocol.json')}. Frozen selection SHA-256: {selection_hash}. All raw fits and frozen source files are unchanged; rendering provenance is recorded in rendering_manifest.json."
    definitions = ["Architecture key: MLP32 = three width-32 ReLU hidden layers; MLP64 = three width-64 layers; Deep64 = width-64 stem plus three plain two-layer blocks; Res64 = corresponding residual blocks; B8-Res64 = learned linear bottleneck of min(D,8) dimensions before Res64. Activation is 2 sigmoid(αz). JS equals balanced P-versus-Q binary cross-entropy up to an additive constant; it is not the historical P-versus-midpoint classifier.",
                   "Matrices retain all 20 Hellinger combinations within every setting; each loss table adds KL, chi-square and JS at the common baseline. All 115 setting-by-candidate summaries, including support and widths, are in summary.csv. The gallery reuses prespecified repeat-zero fixed-bin panels; panels were not chosen for attractive evaluation errors."]
    for extension in ("md", "html"):
        links = [("Full-grid appendix", f"appendix.{extension}"), ("All 115 summaries", "summary.csv"), ("Equal-setting averages", "aggregation.csv"), ("LaTeX tables", "tables.tex"), ("Reproducibility record", "rendering_manifest.json")]
        write_document(output/f"report.{extension}", "Simulation model selection", intro, main, notes, links,
                       table_images=table_images if extension == "md" else ())
        write_document(output/f"appendix.{extension}", "Simulation model selection: full-grid supplement", intro, supplement, notes+definitions+[provenance],
                       [("Main report", f"report.{extension}"), ("All summaries", "summary.csv")], figures)
    w.save_json(output/"rendering_manifest.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "run": str(run),
        "renderer_sha256": w.digest(Path(__file__)), "consumed_sha256": consumed,
        "rendering_dependencies_sha256": {str(path.relative_to(ROOT)): w.digest(path)
                                          for path in [*w.source_paths(), Path(__file__).resolve()]},
        "rendering_versions": {"python": w.platform.python_version(), "numpy": np.__version__,
                               "matplotlib": w.importlib.metadata.version("matplotlib")},
        "aggregation_rule": "equal fixed-setting means; sqrt(sum(within-setting sample variance / n))/K",
        "outputs_sha256": {str(p.relative_to(output)): w.digest(p) for p in sorted(output.rglob("*")) if p.is_file()}})
    return output / "report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(render(args.run, args.output))


if __name__ == "__main__":
    main()
