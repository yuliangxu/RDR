#!/usr/bin/env python3
"""Present the retained AGP mask comparisons without refitting or reassessment.

Read sealed aggregate exports; use receipt-verified private scores only to draw
densities. Preserve the original exports and archives and require a fresh output
directory. The user-directed method exclusion is a posthoc change of paper scope,
not a metric-based model selection decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/agp-retained-mask-presentation")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


METHODS = ("bernoulli_mixture_icfm", "autoregressive_icfm", "midasim_mask_icfm", "midasim")
NEW_METHODS = METHODS[:2]
LABELS = ("Bernoulli mixture + ICFM", "NADE mask + ICFM", "MIDASim mask + ICFM", "MIDASim")
SHORT = ("Bernoulli mixture\n+ ICFM", "NADE mask\n+ ICFM", "MIDASim mask\n+ ICFM", "MIDASim")
COLORS = ("#E69F00", "#CC79A7", "#0072B2", "#D55E00")
RDR_TABLES = ("comparison.csv", "null_controls.csv", "metrics.csv", "metrics_aggregate.csv",
              "fixed_width_cells.csv", "split_counts.csv", "fit_stopping.csv", "epoch_cap_flags.csv",
              "selection_ranking.csv", "selection_rows.csv")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def read_table(path):
    # Pandas otherwise interprets the literal study label "null" as missing.
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


class Receipts:
    def __init__(self):
        self.inputs = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Changed input: {path}")
        self.inputs[str(path)] = actual
        return actual

    def manifest(self, root, name, field="files_sha256"):
        self.check(root / name)
        manifest = read(root / name)
        for filename, expected in manifest[field].items():
            self.check(root / filename, expected)
        return manifest

    def unchanged(self):
        for filename, expected in self.inputs.items():
            if sha(filename) != expected:
                raise ValueError(f"Input changed during presentation: {filename}")


def filtered_csv(source, destination, allowed):
    """Copy retained records while preserving every field's source text."""
    with source.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        column = header.index("method")
        rows = [row for row in reader if row[column] in allowed]
    if not rows:
        raise ValueError(f"No retained records: {source}")
    with destination.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    original = read_table(source)
    expected = original[original.method.isin(allowed)].reset_index(drop=True)
    result = read_table(destination)
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    return result


def load_scores(public, archive, receipts):
    protocol = read(public / "protocol.json")
    outer = read(public / "COMPLETE.json")
    if outer["status"] != "complete" or outer["verification_sha256"] != sha(public / "verification.json"):
        raise ValueError("RDR archive completion or verification receipt is invalid")
    receipts.check(archive / "COMPLETE.json", sha(public / "COMPLETE.json"))
    roots = {method: archive / "analyses" / method for method in NEW_METHODS}
    roots.update({method: Path(protocol["references"][method]) for method in METHODS[2:]})
    scores = {}
    for method, root in roots.items():
        complete = root / "COMPLETE.json"
        expected = (outer["outputs_sha256"][str(complete.relative_to(archive))]
                    if method in NEW_METHODS else protocol["inputs_sha256"][str(complete)])
        receipts.check(complete, expected)
        seal = read(complete)
        if seal["status"] != "complete" or seal["final_fits"] != 5 or seal["null_fits"] != 5:
            raise ValueError(f"Incomplete reference: {root}")
        scores[method] = {}
        for study in ("final", "null"):
            scores[method][study] = {}
            for repeat in range(5):
                relative = f"{study}/{repeat:02d}/COMPLETE.json"
                receipts.check(root / relative, seal["assessment_completions"][relative])
                marker = read(root / relative)
                if marker["status"] != "complete" or marker["stage"] != study:
                    raise ValueError(f"Incomplete score source: {root / relative}")
                source = root / study / f"{repeat:02d}/scores.csv"
                receipts.check(source, marker["outputs"]["scores.csv"])
                # Do not load identifiers or other private metadata columns.
                frame = pd.read_csv(source, usecols=["distribution", "rdr"])
                expected_counts = {"P": 2002, "Q": 2089} if study == "final" else {"P": 1001, "Q": 1001}
                if frame.distribution.value_counts().to_dict() != expected_counts:
                    raise ValueError(f"Unexpected score counts: {source}")
                if not np.isfinite(frame.rdr).all() or not frame.rdr.between(0, 2).all():
                    raise ValueError(f"Invalid fitted scores: {source}")
                scores[method][study][repeat] = frame
    return scores


def save(fig, destination, name):
    for suffix in ("png", "pdf"):
        kwargs = {"metadata": {"CreationDate": None, "ModDate": None}} if suffix == "pdf" else {}
        fig.savefig(destination / f"{name}.{suffix}", dpi=200, bbox_inches="tight", **kwargs)
    plt.close(fig)


def rdr_plots(destination, scores, cells):
    for study in ("final", "null"):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6.2), constrained_layout=True)
        for column, method in enumerate(METHODS):
            top, bottom = axes[:, column]
            for repeat in reversed(range(5)):
                frame = scores[method][study][repeat]
                for source, color in (("P", "#0072B2"), ("Q", "#D55E00")):
                    density, edges = np.histogram(frame.loc[frame.distribution.eq(source), "rdr"],
                                                 bins=np.linspace(0, 2, 41), density=True)
                    top.stairs(density, edges, color=color, alpha=1 if repeat == 0 else .17,
                               lw=1.9 if repeat == 0 else .8)
            top.legend(handles=[plt.Line2D([], [], color="#0072B2", label="Real P"),
                                plt.Line2D([], [], color="#D55E00", label="Real Q" if study == "null" else "Generated Q")],
                       fontsize=8, loc="upper left")
            top.set(title=SHORT[column], xlim=(0, 2), xlabel="Fitted observation-level RDR")
            top.axvline(1, ls=":", color=".55", lw=.8)
            for repeat in reversed(range(5)):
                frame = cells[cells.method.eq(method) & cells.study.eq(study) & cells["repeat"].eq(repeat)]
                if len(frame) != 20:
                    raise ValueError("Expected 20 fixed calibration cells per fit")
                visible = frame[frame.eval_mass.gt(0) & frame.calibrated_rdr.notna() & frame.neural_mean.notna()]
                if repeat:
                    bottom.scatter(visible.neural_mean, visible.calibrated_rdr, s=12, c=".6", alpha=.25)
                else:
                    bottom.vlines(visible.neural_mean, visible.c2_lower, visible.c2_upper, color="#9ECAE1", lw=4, zorder=2)
                    c1 = visible[~visible.c1_unavailable.astype(bool)]
                    bottom.vlines(c1.neural_mean, c1.c1_lower, c1.c1_upper, color="#08519C", lw=1.2, zorder=3)
                    bottom.scatter(visible.neural_mean, visible.calibrated_rdr, s=20 + 90 * visible.eval_mass, c="black", zorder=4)
            bottom.plot([0, 2], [0, 2], "--", color=".5", lw=.8)
            if study == "null":
                bottom.axhline(1, ls=":", color=".5", lw=.8)
            bottom.set(xlim=(0, 2), ylim=(-.03, 2.03), xlabel="Within-cell mean fitted RDR")
            for axis in (top, bottom):
                axis.set_xticks([0, .5, 1, 1.5, 2])
                axis.grid(axis="y", alpha=.15)
                axis.set_axisbelow(True)
        axes[0, 0].set_ylabel("Score density")
        axes[1, 0].set_ylabel("Empirical cell-average RDR")
        title = "Real AGP versus generated samples" if study == "final" else "Real-versus-real controls for each selected recipe"
        fig.suptitle(title + "\nPrimary fit 0; four other fits shown faintly", fontsize=12)
        save(fig, destination, "rdr_comparison" if study == "final" else "rdr_null_controls")


def selection_plot(destination, ranking, configs):
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), constrained_layout=True)
    for row, method in enumerate(NEW_METHODS):
        losses, alphas = configs[method]["losses"], configs[method]["output_alphas"]
        table = ranking[ranking.method.eq(method)].set_index("candidate")
        for col, (metric, label) in enumerate((("mean_brier", "Selection Brier"), ("mean_local_gap", "Selection local gap"))):
            axis = axes[row, col]
            values = np.array([[table.loc[f"{loss}_a{alpha:g}", metric] for alpha in alphas] for loss in losses])
            im = axis.imshow(values, cmap="viridis_r", aspect="auto")
            fig.colorbar(im, ax=axis, shrink=.85)
            midpoint = (values.max() + values.min()) / 2
            for i, loss in enumerate(losses):
                for j, alpha in enumerate(alphas):
                    candidate = f"{loss}_a{alpha:g}"
                    axis.text(j, i, f"{values[i, j]:.4f}", ha="center", va="center",
                              color="white" if values[i, j] > midpoint else "black", fontsize=9)
                    if table.loc[candidate, "eligible"]:
                        axis.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False, edgecolor="#F0E442", lw=2))
                    if table.loc[candidate, "selected"]:
                        axis.text(j+.32, i-.25, "★", color="#D55E00", fontsize=18, ha="center")
            axis.set(xticks=range(len(alphas)), xticklabels=[f"{a:g}" for a in alphas],
                     yticks=range(len(losses)), yticklabels=losses, xlabel="Output sigmoid slope",
                     title=f"{LABELS[row]}\n{label}")
    fig.suptitle("Separate development selection for the two additional retained masks\nFive-fit means; yellow = eligible, star = selected", fontsize=12)
    save(fig, destination, "rdr_model_selection")


def conventional_plot(destination, classifiers, fidelity):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.6), constrained_layout=True)
    for axis, recipe, title in zip(axes[0], ("hgb_composition", "hgb_mask"), ("A  Composition AUC", "B  Mask-only AUC")):
        table = classifiers[classifiers.recipe.eq(recipe)].set_index("method")
        axis.axvline(.5, color=".45", ls="--", lw=1, label="Chance")
        null = table.loc["real_real", "auc"]
        axis.axvline(null, color=".45", ls=":", lw=1.5, label=f"Real–real: {null:.3f}")
        for i, method in enumerate(METHODS):
            auc = table.loc[method, "auc"]
            axis.scatter(auc, i, color=COLORS[i], s=75, zorder=3)
            right = auc < .96
            axis.text(auc + (.012 if right else -.012), i, f"{auc:.3f}",
                      ha="left" if right else "right", va="center", fontsize=9)
        axis.set(yticks=range(4), yticklabels=LABELS, ylim=(4.05, -.6), xlim=(.46, 1.025),
                 xlabel="Held-out AUC (higher = more separable)", title=title)
        axis.grid(axis="x", alpha=.13)
        axis.legend(loc="lower center", ncol=2, frameon=False, fontsize=8)
    table = fidelity.set_index("method")
    for axis, metric, title, xlabel in zip(axes[1], ("richness", "shannon"),
            ("C  Mean taxon richness", "D  Mean Shannon diversity"),
            ("Mean number of present taxa", "Mean Shannon entropy (natural log)")):
        reference = table.loc[METHODS[0], f"{metric}_mean_p"]
        axis.axvline(reference, color="black", ls="--", lw=1.4, label=f"Observed P: {reference:.2f}")
        for i, method in enumerate(METHODS):
            value = table.loc[method, f"{metric}_mean_q"]
            axis.scatter(value, i, color=COLORS[i], s=75, zorder=3)
            axis.annotate(f"{value:.2f}", (value, i), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9,
                          bbox={"facecolor": "white", "edgecolor": "none", "pad": .5})
        values = table.loc[list(METHODS), f"{metric}_mean_q"].tolist() + [reference]
        span = max(values) - min(values)
        axis.set(yticks=range(4), yticklabels=LABELS, ylim=(4.05, -.6),
                 xlim=(min(values) - .2 * span, max(values) + .5 * span), xlabel=xlabel, title=title)
        axis.grid(axis="x", alpha=.13)
        axis.legend(loc="lower center", frameon=False, fontsize=8)
    fig.suptitle("Frozen conventional diagnostics: four retained generators\nBottom panels show means only; distributional distances remain in the CSV tables", fontsize=12)
    save(fig, destination, "mask_comparison_readable")


def render(conventional, public, archive, destination):
    conventional, public, archive, destination = [Path(p).resolve() for p in (conventional, public, archive, destination)]
    if destination.exists():
        raise FileExistsError("Use a fresh presentation directory; original evidence must be preserved")
    receipts = Receipts()
    conventional_manifest = receipts.manifest(conventional, "paper_files.json")
    rdr_manifest = receipts.manifest(public, "export_manifest.json")
    if rdr_manifest["status"] != "exported":
        raise ValueError("RDR public export is incomplete")
    for root in (conventional, public):
        if read(root / "verification.json")["status"] != "passed":
            raise ValueError(f"Independent verification did not pass: {root}")
    conventional_seal = read(conventional / conventional_manifest["active_completion_receipt"])
    if conventional_seal["status"] != "complete":
        raise ValueError("Conventional archive is incomplete")
    for name in ("comparison.csv", "classifier_metrics.csv", "fidelity_metrics.csv"):
        receipts.check(conventional / name, conventional_seal["outputs_sha256"][f"metrics/summary/{name}"])
    for name in RDR_TABLES:
        receipts.check(public / name, read(public / "manifest.json")["files_sha256"][name])
    scores = load_scores(public, archive, receipts)
    configs = {method: read(public / "recipes" / method / "config.json") for method in NEW_METHODS}
    destination.mkdir(parents=True)
    tables = {name: filtered_csv(public / name, destination / name, METHODS) for name in RDR_TABLES}
    for source, target, allowed in (("comparison.csv", "conventional_comparison.csv", METHODS),
                                  ("classifier_metrics.csv", "classifier_metrics.csv", (*METHODS, "real_real")),
                                  ("fidelity_metrics.csv", "fidelity_metrics.csv", METHODS),
                                  ("mask_candidate_validation.csv", "mask_candidate_validation.csv", ("bernoulli_mixture", "autoregressive"))):
        tables[target] = filtered_csv(conventional / source, destination / target, allowed)
    for name in ("comparison.csv", "null_controls.csv", "conventional_comparison.csv", "fidelity_metrics.csv"):
        if set(tables[name].method) != set(METHODS):
            raise ValueError(f"Incorrect retained methods: {name}")
    if len(tables["classifier_metrics.csv"]) != 15 or len(tables["fixed_width_cells.csv"]) != 800:
        raise ValueError("Incorrect retained classifier or fixed-cell count")
    new_fits = tables["fit_stopping.csv"][tables["fit_stopping.csv"].method.isin(NEW_METHODS)]
    counts = new_fits.study.value_counts().to_dict()
    if counts != {"selection": 160, "final": 10, "null": 10}:
        raise ValueError("Unexpected retained new-fit counts")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    rdr_plots(destination, scores, tables["fixed_width_cells.csv"])
    selection_plot(destination, tables["selection_ranking.csv"], configs)
    conventional_plot(destination, tables["classifier_metrics.csv"], tables["fidelity_metrics.csv"])
    receipts.unchanged()
    manifest = {
        "status": "complete", "kind": "presentation_only", "created_date": "2026-10-13",
        "retained_methods": list(METHODS), "reference_studies_reused": list(METHODS[2:]),
        "existing_new_fit_counts_in_retained_subset": counts,
        "new_fits_run_by_this_renderer": 0, "new_assessments_run_by_this_renderer": 0,
        "exclusion": {"method": "bootstrap_icfm", "label": "Resampled mask + ICFM",
            "rationale": "Finite empirical mask support and overlap of evaluator development subjects with the generator mask bank; a fitted evaluator can miss exact-bank membership separation on held-out real subjects.",
            "decision": "User-directed posthoc paper scope; not selection based on favorable or unfavorable evaluation metrics.",
            "preservation": "Original sealed reports, numerical results, and archive artifacts remain unchanged."},
        "inference_scope": "Retrospective assessment; five-fit SD describes initialization variation on the same observations. Fixed-cell CIs are descriptive cell-average diagnostics, not pointwise RDR confidence intervals.",
        "figure_notes": {"rdr_comparison": "Primary fit 0 with four other fits faint; nominal C.1 dark and C.2 light over 20 fixed cells.",
            "rdr_null_controls": "Real-versus-real controls reuse existing subject-separated cohorts and fitted recipes.",
            "rdr_model_selection": "Two additional retained mask methods only; selection ranking CSV also retains the two reused reference methods.",
            "mask_comparison_readable": "Frozen aggregate AUCs and means; real-real AUC controls retained. Bottom panels show means, not ECDFs."},
        "private_artifacts_excluded": ["row-level scores", "sample IDs", "subject IDs", "abundance arrays", "checkpoints"],
        "filter_validation": "All retained CSV fields match their source records exactly; numeric frame equality verified.",
        "input_verification": "Public export manifests, conventional completion seal, RDR summary manifest, and score completion chains checked; all recorded inputs rehashed after rendering.",
        "renderer_sha256": sha(__file__), "inputs_sha256": receipts.inputs,
        "table_rows": {name: len(frame) for name, frame in tables.items()},
        "files_sha256": {p.name: sha(p) for p in sorted(destination.iterdir()) if p.is_file()},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"destination": str(destination), "status": "complete", "tables": len(tables),
            "figures": 4, "input_files_verified": len(receipts.inputs), "retained_new_fit_counts": counts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conventional", type=Path, default=Path("results/AGP/mask_comparison_20261012"))
    parser.add_argument("--rdr", type=Path, default=Path("results/AGP/mask_rdr_20261012"))
    parser.add_argument("--archive", type=Path, default=Path("/cwork/yx306/RDR/AGP/mask_rdr_20261012"))
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.conventional, args.rdr, args.archive, args.destination), sort_keys=True))
