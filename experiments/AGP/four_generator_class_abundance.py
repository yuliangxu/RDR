#!/usr/bin/env python3
"""Class-level stacked abundances for the four frozen AGP generators and P.

Reuse all assessment observations and the historical class colors/sample sort.
The class stack order is shared across panels and follows real mean abundance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.AGP import four_generator_pcoa as original

SOURCES = ("real", *original.METHODS)
LABELS = ("Real data", "Bernoulli mixture + ICFM", "NADE mask + ICFM",
          "MIDASim mask + ICFM", "MIDASim")
COMPACT_LABELS = ("Real data", "Mixture + ICFM", "NADE + ICFM",
                  "MIDASim mask + ICFM", "Full MIDASim")
COMPACT_PALETTE = ("#0072B2", "#E69F00", "#009E73", "#CC79A7",
                   "#56B4E9", "#D55E00", "#F0E442", "#332288", "#D9D9D9")
COMPACT_STEM = "class_relative_abundance_compact"
FULL_FIGURE = Path("results/AGP/four_generator_class_abundance_20261022")
COMPACT_CAPTION = (
    "Class-level relative abundance in real assessment data ($n=2{,}002$) and "
    "four generators ($n=2{,}089$ each). The eight classes with the highest "
    "mean abundance in the real assessment data are shown separately; the "
    "remaining 40 classes are summed as Other. All observations are retained, "
    "independently sorted within each source by decreasing largest-class "
    "abundance in the original 48-class compositions, before aggregation. "
    "Horizontal ranks are not matched observations. This descriptive view "
    "compares broad composition patterns; visual similarity alone does not "
    "establish distributional agreement. See Supplementary "
    "Fig.~\\ref{fig:agp-class-abundance-full} for the full 48-class view."
)


def load_inputs():
    arrays, _, _, inputs = original.load_assessment()
    taxa = None
    for root in original.ROOTS:
        spec = original.read(root / "data/protocol.json")["taxa"]
        path = root / spec["path"]
        digest = original.sha(path)
        if digest != spec["sha256"]:
            raise ValueError(f"Taxonomy hash changed: {path}")
        inputs[str(path)] = digest
        table = pd.read_csv(path, keep_default_na=False)
        if table.taxon_index.tolist() != list(range(614)) or table.taxon_id.nunique() != 614:
            raise ValueError("Taxonomy must retain all 614 feature indices in order")
        if taxa is not None:
            pd.testing.assert_frame_equal(table, taxa)
        taxa = table
    # Independently confirm that the recorded class matches each feature ID.
    for row in taxa.to_dict("records"):
        labels = [part.strip()[3:].strip() or "Unclassified"
                  for part in row["taxon_id"].split(";") if part.strip().startswith("c__")]
        if labels != [row["class"]]:
            raise ValueError("Class label disagrees with the taxon identifier")
    return arrays, taxa, inputs


def aggregate_classes(arrays, taxa):
    classes = sorted(taxa["class"].unique())
    grouped, diagnostics = {}, {}
    indices = {name: np.flatnonzero(taxa["class"].eq(name)) for name in classes}
    if sum(map(len, indices.values())) != 614:
        raise ValueError("Every feature must belong to exactly one class")
    for source in SOURCES:
        raw = np.asarray(arrays[source], dtype=np.float64)
        totals = raw.sum(axis=1)
        if not np.isfinite(raw).all() or (raw < 0).any() or (totals <= 0).any():
            raise ValueError(f"Invalid composition: {source}")
        # Close the saved compositions once; do not discard rare/unclassified taxa.
        closed = raw / totals[:, None]
        values = np.column_stack([closed[:, indices[name]].sum(axis=1) for name in classes])
        np.testing.assert_allclose(values.sum(axis=1), 1, rtol=0, atol=2e-15)
        grouped[source] = values
        diagnostics[source] = {
            "rows": len(raw), "taxa": raw.shape[1], "classes": len(classes),
            "original_row_sum_min": float(totals.min()),
            "original_row_sum_max": float(totals.max()),
            "max_original_row_sum_error": float(np.max(np.abs(totals - 1))),
            "max_class_row_sum_error": float(np.max(np.abs(values.sum(axis=1) - 1))),
        }
    # Ties use alphabetical class order; ordering is shared by all five sources.
    order = np.argsort(-grouped["real"].mean(axis=0), kind="stable")
    classes = [classes[index] for index in order]
    return {source: values[:, order] for source, values in grouped.items()}, classes, diagnostics


def class_colors(classes):
    # The old V3 labels had c__ prefixes and '-' for unspecified classes.
    # Preserve that alphabetic mapping, including the familiar dominant hues.
    key = lambda name: "-" if name == "Unclassified" else "c__" + name
    alphabetical = sorted(classes, key=key)
    cmap = plt.get_cmap("tab20" if len(classes) <= 20 else "tab20b")
    return {name: to_hex(cmap(index / max(1, len(classes) - 1)))
            for index, name in enumerate(alphabetical)}


def draw(grouped, classes, colors, output):
    with plt.rc_context({"font.size": 9, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 8,
                         "pdf.fonttype": 42, "axes.linewidth": 0.6}):
        fig = plt.figure(figsize=(14.5, 11.6))
        grid = fig.add_gridspec(5, 2, width_ratios=(1, 0.23),
                                left=0.065, right=0.995, bottom=0.055, top=0.97,
                                wspace=0.035, hspace=0.34)
        for row, (source, label) in enumerate(zip(SOURCES, LABELS)):
            ax = fig.add_subplot(grid[row, 0])
            values = grouped[source]
            values = values[np.argsort(-values.max(axis=1), kind="stable")]
            n = len(values)
            edges = np.arange(n + 1)
            bottom = np.zeros(n)
            # Step polygons give exact unit-width stacked bars without hundreds
            # of thousands of individual Rectangle objects. Rasterize only bars.
            for column, name in enumerate(classes):
                top = bottom + values[:, column]
                ax.fill_between(edges, np.r_[bottom, bottom[-1]], np.r_[top, top[-1]],
                                step="post", color=colors[name], linewidth=0,
                                antialiased=False, rasterized=True)
                bottom = top
            ax.set(xlim=(0, n), ylim=(0, 1), yticks=[0, 0.25, 0.5, 0.75, 1],
                   ylabel="Relative abundance", xticks=[0, 500, 1000, 1500, 2000])
            ax.set_title(f"({chr(97 + row)}) {label}  (n = {n:,})", loc="left", pad=6)
            ax.tick_params(length=2.5, width=0.6, pad=3)
            if row == 4:
                ax.set_xlabel("Sample index (sorted within each source)", labelpad=5)
        legend_ax = fig.add_subplot(grid[:, 1])
        legend_ax.axis("off")
        legend_ax.legend(handles=[Patch(facecolor=colors[name], label=name) for name in classes],
                         loc="center left", title="Taxonomic class", frameon=False,
                         handlelength=1.3, handleheight=0.8, handletextpad=0.6,
                         borderaxespad=0, labelspacing=0.48)
        fig.savefig(output / "class_relative_abundance.png", dpi=300, facecolor="white")
        fig.savefig(output / "class_relative_abundance.pdf", dpi=300, facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)


def compact_compositions(grouped, classes):
    """Sort on all 48 classes, then sum the last 40 without renormalizing."""
    if len(classes) != 48:
        raise ValueError("The compact assessment view expects the original 48 classes")
    displayed = [*classes[:8], "Other"]
    compact, diagnostics = {}, {}
    for source in SOURCES:
        values = grouped[source]
        order = np.argsort(-values.max(axis=1), kind="stable")
        sorted_values = values[order]
        reduced = np.column_stack((sorted_values[:, :8], sorted_values[:, 8:].sum(axis=1)))
        np.testing.assert_allclose(reduced.sum(axis=1), sorted_values.sum(axis=1),
                                   rtol=0, atol=2e-15)
        compact[source] = reduced
        diagnostics[source] = {
            "rows": len(values),
            "max_aggregation_row_sum_error": float(np.max(np.abs(
                reduced.sum(axis=1) - sorted_values.sum(axis=1)))),
            "original_48_class_sort_indices_sha256": hashlib.sha256(
                order.astype("<i8").tobytes()).hexdigest(),
        }
    return compact, displayed, diagnostics


def draw_compact(compact, classes, colors, output):
    """Fixed manuscript dimensions; only the abundance polygons rasterize."""
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 9, "legend.fontsize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "axes.linewidth": 0.5}):
        fig, axes = plt.subplots(5, 1, figsize=(6.8, 3.6), sharex=True)
        fig.subplots_adjust(left=0.088, right=0.993, bottom=0.23,
                            top=0.94, hspace=0.50)
        largest_source = max(len(values) for values in compact.values())
        for row, (ax, source, label) in enumerate(zip(axes, SOURCES, COMPACT_LABELS)):
            values = compact[source]
            n = len(values)
            # Bin centers are the 1-based ranks 1,...,n; every observation
            # contributes its own unit-width step, without averaging.
            edges = np.arange(n + 1) + 0.5
            bottom = np.zeros(n)
            for column, name in enumerate(classes):
                top = bottom + values[:, column]
                ax.fill_between(edges, np.r_[bottom, bottom[-1]], np.r_[top, top[-1]],
                                step="post", color=colors[name], linewidth=0,
                                antialiased=False, rasterized=True)
                bottom = top
            ax.set(xlim=(0.5, largest_source + 0.5), ylim=(0, 1), yticks=[0, 0.5, 1],
                   yticklabels=["0", "0.5", "1"], xticks=[1, 500, 1000, 1500, 2000])
            ax.set_title(label, loc="left", pad=2)
            ax.tick_params(length=2, width=0.5, pad=2)
            ax.tick_params(axis="x", bottom=(row == 4), labelbottom=(row == 4))
        axes[-1].set_xlabel("sample  index", labelpad=3)
        fig.supylabel("Relative abundance", x=0.015, y=0.585, fontsize=9)
        fig.legend(handles=[Patch(facecolor=colors[name], label=name) for name in classes],
                   loc="lower center", bbox_to_anchor=(0.54, 0.008), ncol=3,
                   frameon=False, handlelength=1.3, handleheight=0.8,
                   handletextpad=0.5, columnspacing=1.4, labelspacing=0.4,
                   borderaxespad=0, borderpad=0)
        fig.savefig(output / (COMPACT_STEM + ".png"), dpi=600, facecolor="white")
        fig.savefig(output / (COMPACT_STEM + ".pdf"), dpi=600, facecolor="white",
                    metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)


def build_compact(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new compact class-abundance output directory")
    arrays, taxa, inputs = load_inputs()
    grouped, original_classes, source_diagnostics = aggregate_classes(arrays, taxa)
    compact, classes, aggregation_diagnostics = compact_compositions(grouped, original_classes)
    colors = dict(zip(classes, COMPACT_PALETTE))
    output.mkdir(parents=True)
    draw_compact(compact, classes, colors, output)
    rows = []
    for source, label in zip(SOURCES, COMPACT_LABELS):
        for index, name in enumerate(classes):
            members = [name] if name != "Other" else original_classes[8:]
            rows.append({"source": source, "label": label, "sample_count": len(compact[source]),
                         "stack_order": index, "class": name, "color": colors[name],
                         "original_class_count": len(members),
                         "taxon_count": int(taxa["class"].isin(members).sum()),
                         "mean_relative_abundance": float(compact[source][:, index].mean())})
    pd.DataFrame(rows).to_csv(output / "class_mean_abundance_compact.csv", index=False,
                             float_format="%.17g")
    # Paths below are relative to the repository root. The full supplementary
    # figure keeps its existing files, colors, dimensions, and 48-class legend.
    graphic = (output / (COMPACT_STEM + ".pdf")).relative_to(REPO).as_posix() \
        if output.is_relative_to(REPO) else (output / (COMPACT_STEM + ".pdf")).as_posix()
    (output / "figure_compact.tex").write_text(
        "% Requires \\usepackage{graphicx}; compile from the repository root.\n"
        "% Keep the figure at 6.8 inches wide to retain the 8 pt minimum text size.\n"
        "\\begin{figure*}[t]\n  \\centering\n"
        f"  \\includegraphics[width=6.8in]{{{graphic}}}\n"
        f"  \\caption{{{COMPACT_CAPTION}}}\n"
        "  \\label{fig:agp-class-abundance-compact}\n\\end{figure*}\n\n"
        "% Add this figure to the supplement to resolve the caption reference.\n"
        "\\begin{figure*}[p]\n  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{FULL_FIGURE.as_posix()}/class_relative_abundance.pdf}}\n"
        "  \\caption{Full 48-class relative abundances for the same assessment "
        "observations, with the same independent within-source sorting as "
        "Fig.~\\ref{fig:agp-class-abundance-compact}.}\n"
        "  \\label{fig:agp-class-abundance-full}\n\\end{figure*}\n"
    )
    original.write(output / "diagnostics_compact.json", {
        "sources": source_diagnostics, "aggregation": aggregation_diagnostics,
        "original_class_count": len(original_classes), "displayed_category_count": len(classes),
        "class_order": classes, "colors": colors, "other_members": original_classes[8:],
        "class_selection_rule": "Eight largest real-assessment mean abundances; alphabetical tie-break",
        "stack_order_rule": "Descending real-assessment mean for selected classes, then Other",
        "sample_order_rule": "Descending largest ORIGINAL 48-class abundance independently within each source; stable ties; BEFORE Other aggregation",
        "input_closure": "Each saved composition closed once in float64 before summing taxa by class",
        "renormalization_after_aggregation": False,
        "all_classes_retained": True, "subsampling": False, "smoothing": False,
        "sample_positions_are_paired_between_sources": False,
        "rank_convention": "One-based within-source rank at each unit-width step center",
        "shared_rank_axis_limits": [0.5, max(len(values) for values in compact.values()) + 0.5],
        "figure_size_inches": [6.8, 3.6], "minimum_font_size_pt": 8,
        "raster_dpi": 600, "rasterized_artists": "Stacked abundance regions only",
        "individual_abundance_arrays_or_identifiers_exported": False,
    })
    for name, expected in inputs.items():
        if original.sha(name) != expected:
            raise ValueError(f"Source changed during rendering: {name}")
    original.write(output / "manifest_compact.json", {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Compact descriptive main-text abundance view for docs/AGP_RDR.md",
        "source_order": list(SOURCES), "panel_labels": list(COMPACT_LABELS),
        "rows": sum(len(arrays[s]) for s in SOURCES), "taxa": 614,
        "original_classes": len(original_classes), "displayed_categories": len(classes), "new_fits": 0,
        "inputs_sha256": inputs,
        "source_sha256": {str(path.relative_to(REPO)): original.sha(path) for path in
                          (Path(__file__).resolve(), Path(original.__file__).resolve())},
        "supplementary_figure_sha256": {
            str(path.relative_to(REPO)): original.sha(path)
            for path in sorted((REPO / FULL_FIGURE).glob("class_relative_abundance.*"))},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "files_sha256": {path.name: original.sha(path) for path in sorted(output.iterdir())},
    })
    print(output / (COMPACT_STEM + ".png"))


def build(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a new class-abundance output directory")
    arrays, taxa, inputs = load_inputs()
    grouped, classes, diagnostics = aggregate_classes(arrays, taxa)
    colors = class_colors(classes)
    output.mkdir(parents=True)
    draw(grouped, classes, colors, output)
    rows = []
    for source, label in zip(SOURCES, LABELS):
        for index, name in enumerate(classes):
            rows.append({"source": source, "label": label, "sample_count": len(grouped[source]),
                         "stack_order": index, "class": name, "color": colors[name],
                         "taxon_count": int(taxa["class"].eq(name).sum()),
                         "mean_relative_abundance": float(grouped[source][:, index].mean())})
    pd.DataFrame(rows).to_csv(output / "class_mean_abundance.csv", index=False, float_format="%.17g")
    original.write(output / "diagnostics.json", {
        "sources": diagnostics, "class_count": len(classes), "class_order": classes,
        "class_order_rule": "Descending real-assessment mean class abundance; alphabetical tie-break",
        "sample_order_rule": "Descending largest class abundance independently within each source; stable ties",
        "input_closure": "Each saved composition closed once in float64 before summing taxa by class",
        "palette": "Historical tab20b, indexed by alphabetical c__ labels with '-' for Unclassified",
        "all_classes_retained": True, "subsampling": False,
        "sample_positions_are_paired_between_sources": False,
        "individual_abundance_arrays_or_identifiers_exported": False,
    })
    for name, expected in inputs.items():
        if original.sha(name) != expected:
            raise ValueError(f"Source changed during rendering: {name}")
    original.write(output / "manifest.json", {
        "status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Five-panel class-level assessment abundance illustration for docs/AGP_RDR.md",
        "source_order": list(SOURCES), "rows": sum(len(arrays[s]) for s in SOURCES),
        "taxa": 614, "classes": len(classes), "new_fits": 0,
        "inputs_sha256": inputs,
        "source_sha256": {str(path.relative_to(REPO)): original.sha(path) for path in
                          (Path(__file__).resolve(), Path(original.__file__).resolve())},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "files_sha256": {path.name: original.sha(path) for path in sorted(output.iterdir())},
    })
    print(output / "class_relative_abundance.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New figure output directory")
    parser.add_argument("--compact", action="store_true",
                        help="Main-text 6.8 x 3.6 inch view: top eight real classes plus Other")
    args = parser.parse_args()
    (build_compact if args.compact else build)(args.output)
