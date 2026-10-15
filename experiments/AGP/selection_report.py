"""Rebuild the fixed-bin AGP paper report from completed, frozen model outputs.

No training or selection is performed here. The saved selection is verified,
and every test summary is independently reconstructed from archived scores.
Run with ``python -m experiments.AGP.selection_report --out RUN_DIRECTORY``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils.model_selection import (
    balanced_brier,
    calibration_diagnostics,
    choose_configuration,
)


SOURCE_COLORS = {"P": "#0072B2", "Q": "#D55E00"}
STUDY_LABELS = {"final": "Real AGP versus archived ICFM", "null": "Real AGP versus real AGP"}
REGION_EDGES = np.array([0., .5, 1., 1.5, 2.])
REGION_LABELS = ["[0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2]"]
METRICS = [
    "brier", "local_gap", "supported_mass", "c2_distance", "c2_width",
    "middle_mass", "middle_local_gap", "middle_c2_distance",
    "agp_middle_mass", "agp_middle_supported_mass", "agp_middle_local_gap",
    "mean_rdr_p", "mean_rdr_q", "equal_mixture_mean_rdr", "null_rdr_mse",
    "c2_excludes_one_bins",
]


def _read_json(path):
    return json.loads(Path(path).read_text())


def _read_csv(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=True)


def _number(value, digits=6):
    if value is None or pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}g}"
    return str(value).replace("|", "\\|")


def _table(frame, columns=None, labels=None):
    columns = list(frame.columns) if columns is None else columns
    lines = ["| " + " | ".join(labels or columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_number(v) for v in row) + " |"
              for row in frame[columns].itertuples(index=False, name=None)]
    return "\n".join(lines)


def _same_float(actual, expected, message, atol=1e-10):
    if not np.isclose(float(actual), float(expected), atol=atol, rtol=1e-8, equal_nan=True):
        raise ValueError(f"{message}: {actual} != {expected}")


def _verify_selection(out, config):
    rows = _read_csv(out / "selection/rows.csv")
    required = {"candidate", "repeat", "loss", "output_alpha", "brier", "local_gap", "supported_mass"}
    if required - set(rows.columns):
        raise ValueError(f"Selection rows missing {required - set(rows.columns)}")
    expected = {f"{loss}_a{alpha:g}" for loss in config["losses"]
                for alpha in config["output_alphas"]}
    repeats = list(range(int(config["repeats"])))
    if set(rows.candidate) != expected or set(rows["repeat"]) != set(repeats):
        raise ValueError("Selection archive does not match the configured candidate/repeat grid")
    # to_dict yields Python scalars, as required by the shared strict selector.
    recomputed = choose_configuration(rows.to_dict("records"))
    saved = _read_json(out / "selection/selection.json")
    if saved["chosen"] != recomputed["chosen"] or saved["brier_reference"] != recomputed["brier_reference"]:
        raise ValueError("Saved model selection does not reproduce from selection/rows.csv")
    previous = {row["candidate"]: row for row in saved["ranking"]}
    for row in recomputed["ranking"]:
        if row["candidate"] not in previous:
            raise ValueError("Saved ranking omits a candidate")
        for key, value in row.items():
            if isinstance(value, float):
                _same_float(previous[row["candidate"]][key], value, f"Selection {key}")
            elif previous[row["candidate"]][key] != value:
                raise ValueError(f"Saved ranking differs at {row['candidate']}/{key}")
    ranking = pd.DataFrame(recomputed["ranking"])
    dispersions = rows.groupby("candidate").agg(
        sd_brier=("brier", "std"), sd_local_gap=("local_gap", "std"),
    ).reset_index()
    return rows, ranking.merge(dispersions, on="candidate", validate="one_to_one"), saved, repeats


def _manifest(out, study):
    # IDs must remain strings; sample identifiers can contain NA-like tokens.
    manifest = pd.read_csv(out / f"data/{study}/manifest.csv", dtype=str, keep_default_na=False)
    required = {"distribution", "role", "sample_id", "subject_id", "source_file", "source_row", "row_sha256"}
    if required - set(manifest.columns):
        raise ValueError(f"{study} manifest missing {required - set(manifest.columns)}")
    if manifest.duplicated(["distribution", "role", "sample_id"]).any():
        raise ValueError(f"Duplicate sample IDs within {study} manifest source/role")
    if set(manifest.distribution) != {"P", "Q"}:
        raise ValueError(f"Unexpected source labels in {study} manifest")
    return manifest


def _scores(out, study, repeat, manifest):
    path = out / study / f"{repeat:02d}"
    scores = pd.read_csv(path / "scores.csv", dtype=str, keep_default_na=False)
    fields = ["distribution", "role", "sample_id", "subject_id", "source_file", "source_row", "row_sha256"]
    if set(fields + ["rdr"]) - set(scores.columns):
        raise ValueError(f"Missing score provenance fields in {path}")
    scores["rdr"] = pd.to_numeric(scores.rdr, errors="raise")
    if not np.isfinite(scores.rdr).all() or not scores.rdr.between(0, 2).all():
        raise ValueError(f"Invalid RDR scores in {path}")
    keys = ["distribution", "role", "sample_id"]
    if scores.duplicated(keys).any():
        raise ValueError(f"Duplicate scored sample in {path}")
    # Support a score archive containing test only or all train/stopping/test roles.
    roles = set(scores.role)
    if "test" not in roles or not roles.issubset(set(manifest.role)):
        raise ValueError(f"Invalid score roles in {path}")
    expected = manifest[manifest.role.isin(roles)][fields].sort_values(keys).reset_index(drop=True)
    observed = scores[fields].sort_values(keys).reset_index(drop=True)
    if not observed.equals(expected):
        raise ValueError(f"Archived score identities differ from the manifest in {path}")
    return scores[scores.role.eq("test")].copy()


def _study(out, study, repeats, manifest, config):
    records, scores_by_repeat, cells_by_repeat = [], {}, {}
    for repeat in repeats:
        path = out / study / f"{repeat:02d}"
        scores = _scores(out, study, repeat, manifest)
        p, q = (scores.loc[scores.distribution.eq(source), "rdr"].to_numpy() for source in ("P", "Q"))
        summary, calculated = calibration_diagnostics(
            p, q, p, q, alpha=float(config["ci_alpha"]), bins=int(config["bins"]))
        cells = _read_csv(path / "cells.csv").sort_values("bin").reset_index(drop=True)
        reference = pd.DataFrame(calculated).sort_values("bin").reset_index(drop=True)
        if len(cells) != int(config["bins"]):
            raise ValueError(f"Wrong number of calibration cells in {path}")
        for column in reference.columns:
            if column not in cells:
                raise ValueError(f"Missing cell column {column} in {path}")
            if not np.allclose(cells[column].to_numpy(dtype=float), reference[column].to_numpy(dtype=float),
                               atol=1e-10, rtol=1e-8, equal_nan=True):
                raise ValueError(f"Fixed-bin {column} does not reproduce in {path}")
        summary.update({"brier": balanced_brier(p, q), "mean_rdr_p": float(p.mean()),
                        "mean_rdr_q": float(q.mean()), "equal_mixture_mean_rdr": float(.5*(p.mean()+q.mean()))})
        middle = reference[reference.left.ge(.2-1e-12) & reference.right.le(1.8+1e-12)]
        supported_middle = middle[middle.absolute_gap.notna()]
        support = float(supported_middle.eval_mass.sum())
        summary.update({"agp_middle_mass": float(middle.eval_mass.sum()),
                        "agp_middle_supported_mass": support,
                        "agp_middle_local_gap": float((supported_middle.eval_mass * supported_middle.absolute_gap).sum()/support)
                        if support else None})
        if study == "null":
            summary["null_rdr_mse"] = float(.5*(np.mean((p-1)**2)+np.mean((q-1)**2)))
            summary["c2_excludes_one_bins"] = int(((reference.c2_lower > 1) | (reference.c2_upper < 1)).sum())
        saved = _read_json(path / "metrics.json")
        for key, value in summary.items():
            if key in saved and value is not None and saved[key] is not None:
                _same_float(saved[key], value, f"{path.name}/{key}")
        for source in ("p", "q"):
            if f"mean_score_{source}" in saved:
                _same_float(saved[f"mean_score_{source}"], summary[f"mean_rdr_{source}"], f"{path.name}/mean score {source}")
        if study == "null" and "true_rdr_mse" in saved:
            _same_float(saved["true_rdr_mse"], summary["null_rdr_mse"], f"{path.name}/null MSE")
        record = {"study": study, "repeat": repeat, "n_p": len(p), "n_q": len(q), **summary}
        for key in ("auc", "AUC", "best_epoch", "epochs_run", "seed"):
            if key in saved:
                record[key.lower()] = saved[key]
        records.append(record)
        scores_by_repeat[repeat], cells_by_repeat[repeat] = scores, cells
    result = pd.DataFrame(records)
    for column in result.columns:
        if column != "study":
            result[column] = pd.to_numeric(result[column], errors="raise")
    return result, scores_by_repeat, cells_by_repeat


def _aggregate(metrics):
    rows = []
    for study, frame in metrics.groupby("study", sort=False):
        for column in METRICS + ["auc", "best_epoch", "epochs_run"]:
            if column not in frame:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not len(values):
                continue
            rows.append({"study": study, "metric": column, "repeats": len(values),
                         "mean": values.mean(), "sd": values.std(ddof=1) if len(values) > 1 else np.nan,
                         "minimum": values.min(), "maximum": values.max()})
    return pd.DataFrame(rows)


def _save(fig, summary, name, plt):
    fig.savefig(summary / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(summary / f"{name}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def _selection_figure(rows, ranking, selection, config, summary, plt):
    losses, alphas = config["losses"], config["output_alphas"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ("mean_brier", "mean_local_gap"),
                               ("Selection balanced Brier", "Selection local absolute gap")):
        matrix = np.array([[ranking.set_index("candidate").loc[f"{loss}_a{alpha:g}", metric]
                            for alpha in alphas] for loss in losses])
        image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
        fig.colorbar(image, ax=ax, shrink=.83)
        for i, loss in enumerate(losses):
            for j, alpha in enumerate(alphas):
                candidate = f"{loss}_a{alpha:g}"
                row = ranking.loc[ranking.candidate.eq(candidate)].iloc[0]
                ax.text(j, i, f"{matrix[i,j]:.4f}", ha="center", va="center", fontsize=9,
                        color="white" if matrix[i,j] > (matrix.max()+matrix.min())/2 else "black")
                if row.eligible:
                    from matplotlib.patches import Rectangle
                    ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False, lw=1.8,
                                           edgecolor="#F0E442"))
                if candidate == selection["chosen"]["candidate"]:
                    ax.text(j+.34, i-.3, "★", ha="center", va="center", color="#D55E00", fontsize=18)
        ax.set(xticks=np.arange(len(alphas)), xticklabels=[f"{a:g}" for a in alphas],
               yticks=np.arange(len(losses)), yticklabels=losses, xlabel="Output sigmoid slope", title=title)
    fig.suptitle(f"Mean across {config['repeats']} paired fits; yellow: eligible, star: selected", fontsize=11)
    _save(fig, summary, "model_selection", plt)


def _scientific_figure(study, scores, cells, summary, config, plt):
    primary = min(scores)
    edges = np.linspace(0, 2, int(config["bins"])+1)
    centers, width = (edges[:-1]+edges[1:])/2, edges[1]-edges[0]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1.15]}, constrained_layout=True)
    hist, ci, mass, cell_mass = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    source_labels = {"P": "Real P", "Q": "ICFM Q" if study == "final" else "Real Q"}
    for repeat, frame in scores.items():
        for source in ("P", "Q"):
            values = frame.loc[frame.distribution.eq(source), "rdr"].to_numpy()
            density, _ = np.histogram(values, bins=edges, density=True)
            hist.stairs(density, edges, color=SOURCE_COLORS[source],
                        alpha=1 if repeat == primary else .16, lw=2 if repeat == primary else .8,
                        label=source_labels[source] if repeat == primary else None,
                        zorder=3 if repeat == primary else 1)
    hist.set(ylabel="Density", title="Frozen test-score distributions", xlim=(0, 2))
    hist.legend(loc="upper center", fontsize=9)
    rows = cells[primary]
    visible = rows.eval_mass.gt(0) & rows.calibrated_rdr.notna() & rows.neural_mean.notna()
    for repeat, frame in cells.items():
        if repeat != primary:
            active = frame.eval_mass.gt(0) & frame.calibrated_rdr.notna()
            ci.scatter(frame.loc[active, "neural_mean"], frame.loc[active, "calibrated_rdr"],
                       s=13, color=".5", alpha=.18, zorder=1)
    active = rows[visible]
    ci.vlines(active.neural_mean, active.c2_lower, active.c2_upper,
              color="#9ECAE1", lw=5, label="C.2 (within-model simultaneous)", zorder=2)
    c1 = active[~active.c1_unavailable.astype(bool)]
    ci.vlines(c1.neural_mean, c1.c1_lower, c1.c1_upper,
              color="#08519C", lw=1.5, label="C.1 (marginal)", zorder=3)
    ci.scatter(active.neural_mean, active.calibrated_rdr, color="black", s=18,
               label="Empirical cell-average RDR", zorder=4)
    ci.plot([0, 2], [0, 2], "--", color=".5", lw=1, zorder=0)
    ci.set(xlim=(0, 2), ylim=(-.03, 2.03), ylabel="Calibrated cell-average RDR",
           title=f"{int(config['bins'])} fixed-width score cells")
    ci.legend(fontsize=7, loc="upper left")
    for source, count_col, offset in (("P", "eval_count_p", -.22), ("Q", "eval_count_q", .22)):
        counts = rows[count_col].to_numpy()
        mass.bar(centers+offset*width, counts/counts.sum(), width=.42*width,
                 color=SOURCE_COLORS[source], alpha=.85)
    cell_mass.bar(centers, rows.eval_mass, width=.85*width, color="#6B8B8E")
    mass.set(ylabel="Source mass", xlabel="Neural RDR", xlim=(0, 2))
    cell_mass.set(ylabel="Balanced mass", xlabel="Within-cell mean neural RDR / score cell", xlim=(0, 2))
    # Cell masses use fixed numeric bin centers; the CI above uses the observed cell mean.
    for ax in axes.flat:
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    if study == "null":
        hist.axvline(1, color=".35", linestyle="--", lw=1)
        ci.axhline(1, color=".35", linestyle=":", lw=1)
    fig.suptitle(f"{STUDY_LABELS[study]} — primary repeat {primary}; other fits shown faintly", fontsize=12)
    _save(fig, summary, f"{study}_fixed_width", plt)


def _closed_array(out, study, role, source, manifest, taxa_count):
    values = np.load(out / f"data/{study}/{role}_{source.lower()}.npy", allow_pickle=False)
    group = manifest[manifest.distribution.eq(source) & manifest.role.eq(role)]
    if "array_row" in group and not np.array_equal(group.array_row.astype(int).to_numpy(), np.arange(len(group))):
        raise ValueError("Manifest array_row does not match the archived array ordering")
    if values.shape != (len(group), taxa_count):
        raise ValueError(f"Abundance array shape differs from manifest/taxa: {study}/{role}/{source}")
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values.sum(axis=1) <= 0):
        raise ValueError("Invalid raw nonnegative abundance data")
    return values / values.sum(axis=1, keepdims=True), group


def _taxonomy(out, manifest, scores, summary, plt):
    taxa = pd.read_csv(out / "data/taxa.csv", keep_default_na=False)
    index_column = "taxon_index" if "taxon_index" in taxa else "feature_index"
    if index_column not in taxa or "phylum" not in taxa:
        raise ValueError("Taxonomy needs taxon_index (or feature_index) and phylum")
    taxa = taxa.sort_values(index_column).reset_index(drop=True)
    if not np.array_equal(taxa[index_column], np.arange(len(taxa))):
        raise ValueError("Taxonomy indices are not a complete zero-based feature order")
    taxa["phylum"] = taxa.phylum.astype(str).str.strip().replace({"": "Unclassified", "nan": "Unclassified"})
    phyla = sorted(taxa.phylum.unique())
    membership = np.column_stack([taxa.phylum.eq(phylum).to_numpy(dtype=float) for phylum in phyla])
    training = {}
    for source in ("P", "Q"):
        abundances, _ = _closed_array(out, "final", "train", source, manifest, len(taxa))
        training[source] = (abundances @ membership).mean(axis=0)
    training_means = .5*(training["P"]+training["Q"])
    order = sorted(range(len(phyla)), key=lambda i: (-training_means[i], phyla[i]))
    top_indices = order[:8]
    top_phyla = [phyla[i] for i in top_indices]
    pd.DataFrame({"phylum": phyla, "train_mean_p": training["P"], "train_mean_q": training["Q"],
                  "balanced_train_mean": training_means, "shown_separately": [p in top_phyla for p in phyla]}
                 ).sort_values("balanced_train_mean", ascending=False).to_csv(summary / "phylum_display_selection.csv", index=False)
    primary = min(scores)
    test_scores = scores[primary]
    rows, masses, display = [], [], []
    for source in ("P", "Q"):
        abundances, group = _closed_array(out, "final", "test", source, manifest, len(taxa))
        phylum_abundance = abundances @ membership
        source_scores = test_scores[test_scores.distribution.eq(source)].set_index("sample_id")
        ordered_scores = source_scores.loc[group.sample_id, "rdr"].to_numpy()
        indices = np.minimum(np.searchsorted(REGION_EDGES, ordered_scores, side="right")-1, 3)
        if (indices < 0).any() or (indices > 3).any():
            raise ValueError("Invalid score-region assignment")
        for region, label in enumerate(REGION_LABELS):
            mask = indices == region
            count = int(mask.sum())
            proportion = count / len(ordered_scores)
            means = phylum_abundance[mask].mean(axis=0) if count else np.full(len(phyla), np.nan)
            masses.append({"repeat": primary, "distribution": source, "region": region,
                           "region_label": label, "left": REGION_EDGES[region], "right": REGION_EDGES[region+1],
                           "count": count, "source_count": len(ordered_scores), "source_mass": proportion,
                           "balanced_mass_contribution": .5*proportion,
                           "mean_rdr": float(ordered_scores[mask].mean()) if count else np.nan})
            for phylum, mean in zip(phyla, means):
                rows.append({**masses[-1], "phylum": phylum, "mean_relative_abundance": mean})
            other = float(means[[i for i in range(len(phyla)) if i not in top_indices]].sum()) if count else np.nan
            display.append([*(means[top_indices]), other])
    masses = pd.DataFrame(masses)
    taxonomy = pd.DataFrame(rows)
    masses.to_csv(summary / "score_region_masses.csv", index=False)
    taxonomy.to_csv(summary / "phylum_by_score_region.csv", index=False)
    display = np.array(display)
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 6.7), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 3]}, constrained_layout=True)
    x = np.r_[np.arange(4), np.arange(4)+4.6]
    axes[0].bar(x, masses.source_mass, width=.72, color=[SOURCE_COLORS[s] for s in masses.distribution])
    for i, row in masses.iterrows():
        axes[0].text(x[i], row.source_mass+.015, f"n={row['count']:,}", ha="center", fontsize=8)
    axes[0].set(ylabel="Within-source mass", ylim=(0, min(1.2, max(.2, masses.source_mass.max()+.17))),
                title=f"Fixed score regions of the primary frozen model (repeat {primary})")
    colors = list(plt.get_cmap("tab10").colors[:len(top_phyla)]) + [(0.72, .72, .72)]
    baseline = np.zeros(len(display))
    for j, (label, color) in enumerate(zip(top_phyla+["Other"], colors)):
        heights = np.nan_to_num(display[:, j], nan=0.)
        axes[1].bar(x, heights, width=.72, bottom=baseline, color=color, label=label)
        baseline += heights
    for i, count in enumerate(masses["count"]):
        if not count:
            axes[1].bar(x[i], 1, width=.72, facecolor="white", edgecolor=".7", hatch="///")
            axes[1].text(x[i], .5, "Empty", ha="center", rotation=90, color=".4", fontsize=9)
    axes[1].set(xticks=x, xticklabels=REGION_LABELS*2, ylabel="Mean relative abundance", ylim=(0, 1),
                xlabel="Real P                                     Neural RDR region                                     ICFM Q")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, title="Phylum")
    for ax in axes:
        ax.axvline(3.8, color=".8", lw=1)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
    _save(fig, summary, "taxonomic_score_regions", plt)
    return masses, taxonomy, top_phyla


def _counts(manifests):
    records = []
    for study, manifest in manifests.items():
        for (role, source), frame in manifest.groupby(["role", "distribution"], sort=False):
            subjects = frame.subject_id[frame.subject_id.ne("")]
            records.append({"study": study, "role": role, "distribution": source,
                            "observations": len(frame), "nonempty_subject_ids": subjects.nunique()})
    return pd.DataFrame(records)


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_archive(out, config, protocol):
    """Completion and receipt gates also apply to standalone regeneration."""
    from experiments.AGP.selected_workflow import (
        checked_selection, selection_tasks, verify_expected_task, verify_snapshot,
    )
    verify_snapshot(out)
    chosen = checked_selection(out)
    tasks = [("selection", task, out / "selection/runs" / task["candidate"] / f"{task['repeat']:02d}")
             for task in selection_tasks(config)]
    tasks += [(study, {**chosen, "repeat": repeat}, out / study / f"{repeat:02d}")
              for study in STUDY_LABELS for repeat in range(int(config["repeats"]))]
    for stage, task, directory in tasks:
        verify_expected_task(out, directory, stage, task)
    receipts = [protocol[key] for key in ("original_manifest", "original_audit", "taxa")]
    for study in protocol["studies"].values():
        receipts += [study["manifest"], study["audit"], *study["arrays"].values()]
    for receipt in receipts:
        if _hash(out / receipt["path"]) != receipt["sha256"]:
            raise ValueError(f"Prepared data receipt does not match: {receipt['path']}")


def report(out: Path):
    """Validate a complete study, then write summary/report.md and its artifacts."""
    out = Path(out).resolve()
    config = _read_json(out / "config.json")
    protocol = _read_json(out / "data/protocol.json")
    _verify_archive(out, config, protocol)
    rows, ranking, selected, repeats = _verify_selection(out, config)
    if int(config["bins"]) != 20:
        raise ValueError("This paper report requires the prespecified 20 equal-width bins")
    manifests = {study: _manifest(out, study) for study in ("selection", "final", "null")}
    studies = {study: _study(out, study, repeats, manifests[study], config) for study in STUDY_LABELS}
    metrics = pd.concat([item[0] for item in studies.values()], ignore_index=True)
    aggregate = _aggregate(metrics)
    summary = out / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(summary / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    metrics.to_csv(summary / "metrics.csv", index=False)
    aggregate.to_csv(summary / "metrics_aggregate.csv", index=False)
    ranking.to_csv(summary / "selection_summary.csv", index=False)
    counts = _counts(manifests)
    counts.to_csv(summary / "split_counts.csv", index=False)
    all_cells = pd.concat([frame.assign(study=study, repeat=repeat)
                           for study, (_, _, cells) in studies.items()
                           for repeat, frame in cells.items()], ignore_index=True)
    all_cells.to_csv(summary / "fixed_width_cells.csv", index=False)
    _selection_figure(rows, ranking, selected, config, summary, plt)
    for study, (_, scores, cells) in studies.items():
        _scientific_figure(study, scores, cells, summary, config, plt)
    region_masses, taxonomy, top_phyla = _taxonomy(out, manifests["final"], studies["final"][1], summary, plt)
    chosen = selected["chosen"]
    chosen_row = ranking[ranking.candidate.eq(chosen["candidate"])].iloc[0]
    metric_display = aggregate.copy()
    metric_display["mean ± SD"] = [f"{row['mean']:.6g} ± {row['sd']:.3g}" if pd.notna(row['sd'])
                                    else f"{row['mean']:.6g} (one available repeat)"
                                    for _, row in metric_display.iterrows()]
    main_metrics = ["brier", "local_gap", "supported_mass", "c2_width", "middle_mass", "middle_local_gap",
                    "agp_middle_mass", "agp_middle_local_gap", "auc", "null_rdr_mse", "c2_excludes_one_bins"]
    metric_display = metric_display[metric_display.metric.isin(main_metrics)]
    primary = min(repeats)
    lines = [
        "# AGP: fixed-width calibration and paper results", "",
        f"Selected recipe: **{chosen['loss']}**, output `2 sigmoid({chosen['output_alpha']:g} a)`, "
        f"{config.get('representation', 'ILR')} input, {config.get('architecture', 'deep')} network. "
        f"All {len(rows)} selection fits and {2*len(repeats)} final/control fits are present and verified. "
        f"Selection balanced Brier is {chosen_row.mean_brier:.6g} (seed SD {chosen_row.sd_brier:.3g}); "
        f"local absolute gap is {chosen_row.mean_local_gap:.6g} (seed SD {chosen_row.sd_local_gap:.3g}).", "",
        "## Design and provenance", "",
        "P is real AGP and Q is the fixed archived ICFM generator for the main application. "
        "For the control, P and Q are subject-disjoint random subsets of the real data. "
        "The RDR target is `r0(x)=2p(x)/(p(x)+q(x))`; the corresponding balanced source probability is `r0/2`. "
        "P and Q receive weight one half regardless of observed sample counts.", "",
        "The following counts are read directly from the saved manifests. Unique nonempty subject IDs "
        "are provenance counts; a generator row identifier does not establish an independent biological subject.", "",
        _table(counts), "",
        "Selection uses separate stopping, calibration and evaluation roles as recorded in "
        "[the protocol](../data/protocol.json). Final training/stopping/test allocations are recorded separately. "
        "The test cohort was not used by this selection, but it was examined in historical AGP studies; "
        "this is a retrospective assessment. Historical whole-table preprocessing, incomplete generator provenance "
        "and repeated observations within subjects limit an IID interpretation. Subject separation between roles "
        "does not remove within-role dependence.", "",
        f"The primary illustrative fit is repeat **{primary}**, chosen by its fixed repeat index before test assessment. "
        "The other fitted seeds are shown faintly in the score and calibration figures. "
        "Means and standard deviations below describe variability across fitted seeds on fixed study data; "
        "they are not confidence intervals and do not estimate independent-data sampling uncertainty.", "",
        "Inputs and configuration: [config.json](../config.json), [data protocol](../data/protocol.json), "
        "[main manifest](../data/final/manifest.csv), [null manifest](../data/null/manifest.csv). "
        "The report verifies completion markers, frozen-source/configuration receipts, cached data hashes, "
        "manifest identities, the saved selection, and every test fixed-bin diagnostic against archived scores.", "",
        "## Loss and output-activation selection", "",
        selected["rule"], "",
        f"The Brier reference is `{selected['brier_reference']}`; the selected candidate is `{chosen['candidate']}`. "
        "Selection local gap uses disjoint calibration and evaluation observations. "
        "The loss/output-slope comparison holds the representation and architecture fixed. "
        "It does not establish that this network is optimal across all architectures.", "",
        _table(ranking, ["candidate", "mean_brier", "sd_brier", "mean_local_gap", "sd_local_gap",
                         "minimum_supported_mass", "paired_mean_brier_difference", "paired_brier_difference_se", "eligible"]), "",
        "![Loss and output-slope selection](model_selection.png)", "",
        "[Selection table](selection_summary.csv) · [PDF](model_selection.pdf)", "",
        "## 1. Real AGP versus ICFM: selected fit", "",
        _table(metric_display[metric_display.study.eq("final")], ["metric", "repeats", "mean ± SD"]), "",
        "![Main application with fixed-width calibration](final_fixed_width.png)", "",
        "[Main figure PDF](final_fixed_width.pdf) · [All test cells](fixed_width_cells.csv) · "
        "[Per-repeat metrics](metrics.csv) · [Aggregate metrics](metrics_aggregate.csv)", "",
        f"Calibration uses **20 fixed equal-width bins on [0,2]**, with nominal alpha={float(config['ci_alpha']):g}. "
        "The last bin includes 2; the other right endpoints are excluded. There is no adaptive merging. "
        "Test observations supply both cell frequencies and the descriptive neural cell means. "
        "Thus final local gaps are same-sample descriptive diagnostics; unlike selection gaps, their "
        "calibration and evaluation samples are not independent.", "",
        "C.1 is a marginal asymptotic interval; C.2 is simultaneous over the 20 cells of one frozen model. "
        "The intervals target `theta_j = E_M[r0(X) | rhat(X) in I_j]`, where `I_j` is a fixed numeric score interval "
        "and `M=(P+Q)/2`. "
        "They do not give confidence intervals for individual RDRs. They do not include uncertainty in "
        "the empirical neural cell mean, cover all fitted seeds jointly, or establish verified AGP coverage. "
        "C.1 intervals are omitted where unavailable. Empty score cells remain in the exported table. "
        "The lower-right bars use fixed bin centers; interval x-coordinates use the actual balanced cell means.", "",
        "Local gap is the mixture-mass-weighted absolute difference between the empirical neural cell mean "
        "and estimated cell-average RDR, normalized over supported mass. Middle-region metrics use the "
        "fixed cells in [0.7,1.2). The separately labeled `agp_middle_*` metrics use the historical AGP "
        "region [0.2,1.8); both regions are unions of the same fixed cells. These definitions must not be pooled. "
        "Reported middle mass and conditional gap expose discrepancies that "
        "a low overall gap can conceal. Global Brier remains the fitting-risk criterion: a constant score "
        "can have a small local gap while losing discrimination. True RDR MSE is unknown for the ICFM comparison.", "",
        "## 2. Taxonomic composition across fixed score regions", "",
        "![Phylum composition and region masses](taxonomic_score_regions.png)", "",
        "[Taxonomy figure PDF](taxonomic_score_regions.pdf) · [Every phylum and region](phylum_by_score_region.csv) · "
        "[Region counts and masses](score_region_masses.csv) · [Training-only display selection](phylum_display_selection.csv)", "",
        "Raw abundances are closed to sum one within each observation and then aggregated to phylum. "
        "The displayed phyla are the eight with largest balanced P/Q training mean abundance (or all phyla "
        "if fewer than eight); remaining phyla are grouped as Other. "
        "Display selection uses training abundances only. Test observations are grouped by the "
        "primary frozen model's four prespecified score regions [0,0.5), [0.5,1), [1,1.5), [1.5,2]. "
        "These four descriptive regions do not replace the 20 calibration cells. Empty groups are explicitly marked.", "",
        _table(region_masses, ["distribution", "region_label", "count", "source_mass", "mean_rdr"]), "",
        "These panels describe the composition of observations assigned to different score regions. "
        "They do not identify causal taxa or establish taxon-specific associations; source membership "
        "and compositional dependence can explain differences. No association p-values or significance screens are used.", "",
        "## 3. Same-source real-versus-real control", "",
        _table(metric_display[metric_display.study.eq("null")], ["metric", "repeats", "mean ± SD"]), "",
        "![Real-versus-real fixed-width control](null_fixed_width.png)", "",
        "[Control figure PDF](null_fixed_width.pdf)", "",
        "The control uses the main application's selected loss, activation, representation and architecture; "
        "it does not select a separate recipe on null results. Under the same-source null, `r0=1` and the "
        "constant-source probability 0.5 has balanced Brier 0.25. The exported null RDR MSE is "
        "`0.5 mean_P[(rhat-1)^2] + 0.5 mean_Q[(rhat-1)^2]`. "
        "Finite empirical Brier can lie below 0.25 by chance. A random subject split is a diagnostic control, "
        "not a demonstration of nominal CI coverage or a guarantee of identical empirical subject mixtures.", "",
        "## Paper organization", "",
        "Use the selected real-versus-ICFM score/calibration figure and the phylum interpretation panel "
        "as the main AGP application. Include the real-versus-real control with the application or in a compact "
        "supplementary panel. Put the complete loss/slope ranking, per-seed tables, and full fixed-bin tables "
        "in the supplement. Historical representation, smoothing, architecture, diversity and dependence "
        "analyses remain separate supplementary studies; they use their own sample allocations and "
        "selection histories and should not be pooled with these results. Adaptive-bin and variance-bound "
        "extensions are not part of this fixed-width package.", "",
        "## Reproduction", "",
        f"Regenerate this report with `python -m experiments.AGP.selection_report --out {out}`. "
        "This reads saved model outputs and does not refit models or alter the chosen recipe. "
        "[Artifact hashes](artifact_sha256.csv) cover the generated numerical tables and figures. "
        "The run configuration and workflow provenance record the training commands and source snapshots.", "",
    ]
    (summary / "report.md").write_text("\n".join(lines))
    artifacts = [path for path in sorted(summary.iterdir())
                 if path.is_file() and path.name not in {"artifact_sha256.csv", "report_status.json"}]
    pd.DataFrame([{"file": path.name, "bytes": path.stat().st_size, "sha256": _hash(path)}
                  for path in artifacts]).to_csv(summary / "artifact_sha256.csv", index=False)
    status = {"status": "complete", "selection_fits": len(rows), "main_fits": len(repeats),
              "null_fits": len(repeats), "primary_repeat": primary, "selected": chosen,
              "bins": int(config["bins"]), "calibration": "fixed_width", "artifacts": len(artifacts),
              "source_run": str(out), "verified": ["completion_markers", "source_config_receipts", "data_receipts",
                  "selection", "score_manifest_identities", "fixed_bin_cells", "test_metrics"]}
    (summary / "report_status.json").write_text(json.dumps(status, indent=2)+"\n")
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(report(arguments.out), indent=2))
