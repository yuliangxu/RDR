"""Aggregate the frozen five-generator AGP RDR comparison, without fitting.

New masks have independent loss/slope selection. The two existing generators
reuse their sealed studies. Only aggregate tables and figures are exported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from experiments.AGP import selection_report as shared


NEW_METHODS = ("bootstrap_icfm", "bernoulli_mixture_icfm", "autoregressive_icfm")
METHODS = (*NEW_METHODS, "midasim_mask_icfm", "midasim")
LABELS = {
    "bootstrap_icfm": "Resampled mask + ICFM",
    "bernoulli_mixture_icfm": "Bernoulli mixture + ICFM",
    "autoregressive_icfm": "NADE mask + ICFM",
    "midasim_mask_icfm": "MIDASim mask + ICFM",
    "midasim": "MIDASim",
}
SHORT = {
    "bootstrap_icfm": "Resampled mask\n+ ICFM",
    "bernoulli_mixture_icfm": "Bernoulli mixture\n+ ICFM",
    "autoregressive_icfm": "NADE mask\n+ ICFM",
    "midasim_mask_icfm": "MIDASim mask\n+ ICFM",
    "midasim": "MIDASim",
}
REFERENCE_ROOTS = {
    "midasim_mask_icfm": "/cwork/yx306/RDR/AGP/icfm_zero_20261007/analysis",
    "midasim": "/cwork/yx306/RDR/AGP/midasim_full_20261004/analysis",
}
METRICS = [
    "brier", "auc", "local_gap", "supported_mass", "middle_mass",
    "middle_local_gap", "agp_middle_mass", "agp_middle_local_gap",
    "mean_rdr_p", "mean_rdr_q", "equal_mixture_mean_rdr", "null_rdr_mse",
    "c2_excludes_one_bins", "best_epoch", "epochs_run", "c2_width",
]
PUBLIC_ARTIFACTS = {
    "report.md", "metrics.csv", "metrics_aggregate.csv", "comparison.csv", "null_controls.csv",
    "fixed_width_cells.csv", "split_counts.csv", "fit_stopping.csv", "epoch_cap_flags.csv",
    "selection_ranking.csv", "selection_rows.csv",
    *{f"{name}.{suffix}" for name in ("rdr_comparison", "rdr_null_controls", "rdr_model_selection")
      for suffix in ("png", "pdf")},
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _gate(out):
    path = Path(out) / "verification.json"
    if not path.is_file() or read(path).get("status") != "passed":
        raise ValueError("Report requires independent verification.json status='passed'")
    return path


def _check_receipts(root, receipts):
    for name, digest in receipts.items():
        path = root / name
        if not path.is_file() or sha(path) != digest:
            raise ValueError(f"Changed or missing report input: {path}")


def _auc(scores):
    return float(roc_auc_score(scores.distribution.eq("P"), scores.rdr))


def _load_method(root, method):
    """Reconstruct all assessment diagnostics and reject incomplete recipes."""
    root = Path(root)
    config = read(root / "config.json")
    if int(config["bins"]) != 20 or int(config["repeats"]) != 5:
        raise ValueError("Five-generator report requires 20 fixed bins and five fits")
    rows, ranking, selected, repeats = shared._verify_selection(root, config)
    if len(rows) != 80 or rows.duplicated(["candidate", "repeat"]).any():
        raise ValueError("Every generator requires 80 distinct selection fits")
    freeze = read(root / "selection/freeze.json")
    if (freeze.get("status") != "frozen" or freeze.get("expected_fits") != 80
            or freeze.get("completed_fits") != 80 or freeze.get("test_scores_read") is not False):
        raise ValueError("Selection was not completely frozen before assessment")
    _check_receipts(root, freeze["outputs"])
    _check_receipts(root, freeze["selection_completions"])
    fit_records = []
    for row in rows.to_dict("records"):
        metrics = read(root / "selection/runs" / row["candidate"] / f"{row['repeat']:02d}" / "metrics.json")
        fit_records.append({"method": method, "study": "selection", "candidate": row["candidate"],
                            "repeat": row["repeat"], "selected": row["candidate"] == selected["chosen"]["candidate"],
                            "best_epoch": metrics["best_epoch"], "epochs_run": metrics["epochs_run"],
                            "max_epochs": config["max_epochs"], "stopping_reason": metrics["stopping_reason"],
                            "hit_epoch_cap": metrics["stopping_reason"] == "max_epochs"})
    manifests, studies = {}, {}
    for study in ("final", "null"):
        manifests[study] = shared._manifest(root, study)
        for repeat in repeats:
            directory = root / study / f"{repeat:02d}"
            marker = read(directory / "COMPLETE.json")
            if (marker.get("status") != "complete" or marker.get("stage") != study
                    or marker.get("task") != {**selected["chosen"], "repeat": repeat}
                    or marker.get("selection_freeze_sha256") != sha(root / "selection/freeze.json")):
                raise ValueError(f"Invalid completion or recipe: {directory}")
            _check_receipts(directory, marker["outputs"])
            metrics = read(directory / "metrics.json")
            fit_records.append({"method": method, "study": study, "candidate": selected["chosen"]["candidate"],
                                "repeat": repeat, "selected": True, "best_epoch": metrics["best_epoch"],
                                "epochs_run": metrics["epochs_run"], "max_epochs": config["max_epochs"],
                                "stopping_reason": metrics["stopping_reason"],
                                "hit_epoch_cap": metrics["stopping_reason"] == "max_epochs"})
        metrics, scores, cells = shared._study(root, study, repeats, manifests[study], config)
        expected = (2002, 2089) if study == "final" else (1001, 1001)
        if not ((metrics.n_p == expected[0]) & (metrics.n_q == expected[1])).all():
            raise ValueError(f"Unexpected {study} assessment counts for {method}")
        for repeat, frame in scores.items():
            auc = _auc(frame)
            saved = metrics.loc[metrics["repeat"].eq(repeat), "auc"].iloc[0]
            if not np.isclose(auc, saved, atol=1e-12, rtol=1e-10):
                raise ValueError(f"Saved AUC does not reproduce for {method}/{study}/{repeat}")
        studies[study] = (metrics.assign(method=method, reused=method not in NEW_METHODS), scores, cells)
    ranking = ranking.assign(method=method, selected=ranking.candidate.eq(selected["chosen"]["candidate"]))
    return {"root": root, "config": config, "selection": selected, "ranking": ranking,
            "selection_rows": rows.assign(method=method), "studies": studies,
            "fit_records": pd.DataFrame(fit_records), "manifests": manifests}


def _aggregate(metrics):
    records = []
    for (method, study), group in metrics.groupby(["method", "study"], sort=False):
        if len(group) != 5 or set(group["repeat"]) != set(range(5)):
            raise ValueError(f"Exactly five unique fitted repeats required: {method}/{study}")
        for metric in METRICS:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="raise").dropna()
            if len(values):
                records.append(dict(method=method, label=LABELS[method], study=study, metric=metric,
                                    repeats=len(values), mean=float(values.mean()), sd=float(values.std(ddof=1)),
                                    minimum=float(values.min()), maximum=float(values.max())))
    return pd.DataFrame(records)


def _wide(aggregate, study):
    rows = []
    subset = aggregate[aggregate.study.eq(study)]
    for method in METHODS:
        row = {"method": method, "label": LABELS[method], "reused": method not in NEW_METHODS}
        for record in subset[subset.method.eq(method)].to_dict("records"):
            for stat in ("mean", "sd", "repeats"):
                row[f"{record['metric']}_{stat}"] = record[stat]
        rows.append(row)
    return pd.DataFrame(rows)


def _display(comparison, selected, metrics):
    rows = []
    for row in comparison.to_dict("records"):
        record = {"Generator": row["label"], "Selected loss/slope": selected[row["method"]],
                  "Fit source": "Reused" if row["reused"] else "New"}
        for metric, label in metrics:
            mean, sd = row.get(metric + "_mean"), row.get(metric + "_sd")
            record[label] = "—" if mean is None or pd.isna(mean) else f"{mean:.4f} ± {sd:.4f}"
        rows.append(record)
    return pd.DataFrame(rows)


def _plots(loaded, summary):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/agp-mask-rdr-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})

    def save(fig, name):
        fig.savefig(summary / f"{name}.png", dpi=200, bbox_inches="tight")
        fig.savefig(summary / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)

    for study in ("final", "null"):
        fig, axes = plt.subplots(2, 5, figsize=(16.8, 6.4), constrained_layout=True)
        for col, method in enumerate(METHODS):
            _, scores, cells = loaded[method]["studies"][study]
            top, bottom = axes[:, col]
            for repeat in sorted(scores, reverse=True):
                frame = scores[repeat]
                for source, color in (("P", "#0072B2"), ("Q", "#D55E00")):
                    values = frame.loc[frame.distribution.eq(source), "rdr"].to_numpy()
                    density, edges = np.histogram(values, bins=np.linspace(0, 2, 41), density=True)
                    top.stairs(density, edges, color=color, alpha=1 if repeat == 0 else .17,
                               lw=1.9 if repeat == 0 else .8)
            # Explicit handles avoid duplicate labels from nonprimary curves.
            top.legend(handles=[plt.Line2D([], [], color="#0072B2", label="Real P"),
                                plt.Line2D([], [], color="#D55E00", label="Real Q" if study == "null" else "Generated Q")],
                       fontsize=8, loc="upper left")
            top.set(title=SHORT[method], xlim=(0, 2), xlabel="Fitted observation-level RDR")
            top.axvline(1, ls=":", color=".55", lw=.8)
            for repeat, frame in cells.items():
                valid = frame.eval_mass.gt(0) & frame.calibrated_rdr.notna() & frame.neural_mean.notna()
                visible = frame[valid]
                if repeat:
                    bottom.scatter(visible.neural_mean, visible.calibrated_rdr, s=12, c=".6", alpha=.25)
                else:
                    bottom.vlines(visible.neural_mean, visible.c2_lower, visible.c2_upper,
                                  color="#9ECAE1", lw=4, zorder=2)
                    c1 = visible[~visible.c1_unavailable.astype(bool)]
                    bottom.vlines(c1.neural_mean, c1.c1_lower, c1.c1_upper, color="#08519C", lw=1.2, zorder=3)
                    bottom.scatter(visible.neural_mean, visible.calibrated_rdr,
                                   s=20 + 90*visible.eval_mass, c="black", zorder=4)
            bottom.plot([0, 2], [0, 2], "--", color=".5", lw=.8)
            if study == "null":
                bottom.axhline(1, ls=":", color=".5", lw=.8)
            bottom.set(xlim=(0, 2), ylim=(-.03, 2.03), xlabel="Within-cell mean fitted RDR")
            for ax in (top, bottom):
                ax.set_xticks([0, .5, 1, 1.5, 2])
                ax.grid(axis="y", alpha=.15)
                ax.set_axisbelow(True)
        axes[0, 0].set_ylabel("Score density")
        axes[1, 0].set_ylabel("Empirical cell-average RDR")
        title = "Real AGP versus generated samples" if study == "final" else "Real-versus-real controls for each selected recipe"
        fig.suptitle(title + " — primary fit 0; four other fits shown faintly", fontsize=12)
        save(fig, "rdr_comparison" if study == "final" else "rdr_null_controls")

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for row, method in enumerate(NEW_METHODS):
        item = loaded[method]
        losses, alphas = item["config"]["losses"], item["config"]["output_alphas"]
        ranking = item["ranking"].set_index("candidate")
        for col, (metric, label) in enumerate((("mean_brier", "Selection Brier"),
                                               ("mean_local_gap", "Selection local gap"))):
            ax = axes[row, col]
            values = np.array([[ranking.loc[f"{loss}_a{alpha:g}", metric] for alpha in alphas] for loss in losses])
            im = ax.imshow(values, cmap="viridis_r", aspect="auto")
            fig.colorbar(im, ax=ax, shrink=.85)
            midpoint = (values.max()+values.min()) / 2
            for i, loss in enumerate(losses):
                for j, alpha in enumerate(alphas):
                    candidate = f"{loss}_a{alpha:g}"
                    ax.text(j, i, f"{values[i,j]:.4f}", ha="center", va="center",
                            color="white" if values[i,j] > midpoint else "black", fontsize=9)
                    if ranking.loc[candidate, "eligible"]:
                        ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False, edgecolor="#F0E442", lw=2))
                    if candidate == item["selection"]["chosen"]["candidate"]:
                        ax.text(j+.32, i-.25, "★", color="#D55E00", fontsize=18, ha="center")
            ax.set(xticks=range(len(alphas)), xticklabels=[f"{a:g}" for a in alphas],
                   yticks=range(len(losses)), yticklabels=losses, xlabel="Output sigmoid slope",
                   title=f"{LABELS[method]}: {label}")
    fig.suptitle("Separate selection for each new mask: five-fit means; yellow = eligible, star = selected", fontsize=12)
    save(fig, "rdr_model_selection")


def report(out: Path):
    out = Path(out).resolve()
    verification = _gate(out)
    protocol = read(out / "protocol.json")
    references = protocol.get("references", REFERENCE_ROOTS)
    roots = {method: out / "analyses" / method for method in NEW_METHODS}
    for method, default in REFERENCE_ROOTS.items():
        spec = references.get(method, default)
        roots[method] = Path(spec if isinstance(spec, str) else spec.get("analysis_root", spec.get("path", default)))
    loaded = {method: _load_method(roots[method], method) for method in METHODS}
    all_metrics = pd.concat([loaded[m]["studies"][s][0] for m in METHODS for s in ("final", "null")], ignore_index=True)
    aggregate = _aggregate(all_metrics)
    comparison, controls = _wide(aggregate, "final"), _wide(aggregate, "null")
    fit_records = pd.concat([loaded[m]["fit_records"] for m in METHODS], ignore_index=True)
    cap_flags = fit_records.groupby(["method", "study"], sort=False).agg(
        fits=("repeat", "size"), epoch_cap_fits=("hit_epoch_cap", "sum"),
        maximum_epochs_run=("epochs_run", "max"), maximum_best_epoch=("best_epoch", "max")).reset_index()
    if int(fit_records.method.isin(NEW_METHODS).sum()) != 270:
        raise ValueError("Expected 240 selection + 15 final + 15 null new fits")
    all_cells = pd.concat([frame.assign(method=m, study=s, repeat=r)
                          for m in METHODS for s in ("final", "null")
                          for r, frame in loaded[m]["studies"][s][2].items()], ignore_index=True)
    counts = pd.concat([shared._counts(loaded[m]["manifests"]).assign(method=m) for m in METHODS], ignore_index=True)
    selected = {m: loaded[m]["selection"]["chosen"]["candidate"] for m in METHODS}
    for table in (comparison, controls):
        table.insert(2, "selected_candidate", table.method.map(selected))
        stage = "final" if table is comparison else "null"
        table["epoch_cap_fits"] = [int(cap_flags[(cap_flags.method == m) & (cap_flags.study == stage)].epoch_cap_fits.iloc[0])
                                   for m in table.method]
    summary = out / "summary"
    summary.mkdir(parents=True, exist_ok=True)
    tables = {
        "metrics.csv": all_metrics, "metrics_aggregate.csv": aggregate,
        "comparison.csv": comparison, "null_controls.csv": controls,
        "fixed_width_cells.csv": all_cells, "split_counts.csv": counts,
        "fit_stopping.csv": fit_records, "epoch_cap_flags.csv": cap_flags,
        "selection_ranking.csv": pd.concat([loaded[m]["ranking"] for m in METHODS], ignore_index=True),
        "selection_rows.csv": pd.concat([loaded[m]["selection_rows"] for m in METHODS], ignore_index=True),
    }
    for filename, frame in tables.items():
        frame.to_csv(summary / filename, index=False)
    _plots(loaded, summary)
    main_table = _display(comparison, selected, [("brier", "Brier"), ("auc", "AUC"), ("local_gap", "Local gap"),
                                                ("mean_rdr_p", "Mean RDR: P"), ("mean_rdr_q", "Mean RDR: Q")])
    support_table = _display(comparison, selected, [("supported_mass", "Supported mass"),
                        ("middle_mass", "Middle mass [0.7,1.2)"), ("middle_local_gap", "Middle local gap"),
                        ("agp_middle_mass", "AGP middle mass [0.2,1.8)"), ("agp_middle_local_gap", "AGP middle local gap")])
    null_table = _display(controls, selected, [("brier", "Brier"), ("auc", "AUC"), ("local_gap", "Local gap"),
                                              ("null_rdr_mse", "MSE from true RDR 1")])
    null_table["Epoch-cap fits / 5"] = controls.epoch_cap_fits.to_numpy()
    lines = [
        "# AGP: RDR evaluation of five mask/generator methods", "",
        "Three additional mask strategies are evaluated with separately selected RDR fits: whole-mask resampling, "
        "a finite Bernoulli mixture, and autoregressive NADE. They share the frozen conditional ICFM abundance flow. "
        "The current MIDASim-mask ICFM and full MIDASim reuse their completed RDR studies; they were not refitted.", "",
        "All **240 new selection fits, 15 final fits and 15 real-versus-real control fits** are complete. "
        "Each generator is assessed on **2,002 real samples and 2,089 generated samples**, across five RDR initializations. "
        "The loss/slope selection uses development data only, before the new RDR test scores are computed. "
        "The assessment cohort and these generated samples have already been inspected in preceding analyses, "
        "so this is a retrospective comparison.", "",
        "## Selected RDR comparisons", "",
        shared._table(main_table), "",
        "Entries are mean ± sample SD across five fits on the same observations. These SDs quantify initialization "
        "variation, not independent-data uncertainty. RDR is fitted as `r(x)=2p(x)/(p(x)+q(x))`: values above 1 favor "
        "real P and values below 1 favor generated Q. AUC uses real P as the positive class. Lower Brier and higher AUC "
        "can indicate easier source discrimination; lower Brier is not evidence of better generator fidelity. "
        "Local gap assesses calibration of the fitted RDR, not distributional distance.", "",
        "![Five-generator fitted RDR and fixed-cell diagnostics](rdr_comparison.png)", "",
        "Top: observation-level fitted-score densities. Bottom: empirical cell-average RDR versus within-cell mean "
        "fitted RDR for 20 equal-width cells on [0,2]. Black point area increases with cell mass. "
        "Fit 0 is the prespecified illustration; four other fits are faint. Dark intervals are nominal marginal C.1; "
        "light intervals are nominal C.2, simultaneous over cells within one fit, not over methods or fits. "
        "Cell counts and neural means use the same assessment observations; these gaps and intervals are descriptive. "
        "They do not give pointwise confidence intervals for individual observations' true RDR.", "",
        shared._table(support_table), "",
        "The middle-region gap is conditional on supported cells in that region and must be interpreted with its "
        "mass. The common region is [0.7,1.2); the wider historical AGP region is [0.2,1.8).", "",
        "## What resampling can and cannot establish", "",
        "RDR retains the same development-data limitation as the earlier ordinary classifier comparison. "
        "The real RDR training and stopping subjects also supplied the 10,443-row generator-training mask bank. "
        "For whole-mask resampling, both development sources therefore use training-bank masks; the held-out "
        "real subjects have new masks. Switching the loss or calling the fitted score RDR does not remove this "
        "change between development and assessment. Fitted scores near 1 or weak discrimination do not establish "
        "novel generation or equality of P and Q.", "",
        "The separately documented posthoc exact-mask membership check finds 0/2,002 held-out real masks and "
        "2,089/2,089 resampled generated masks in the original training bank, giving membership AUC 1.000 on "
        "those samples. That bank-aware diagnostic establishes a difference the learned evaluator can miss. "
        "Its perfect empirical AUC is a retrospective sample result, not a population guarantee. A prospectively "
        "independent evaluator cohort would need real training, stopping and assessment subjects excluded from "
        "generator fitting. This comparison intentionally preserves the historical protocol to match the reused studies.", "",
        "## Separate development selection", "",
        "Each new P-versus-Q pair uses its own 4-loss × 4-output-slope × 5-initialization grid. "
        + loaded[NEW_METHODS[0]]["selection"]["rule"], "",
        "![Selection Brier and local gap for the three new masks](rdr_model_selection.png)", "",
        "The representation and neural architecture are fixed to the existing ILR/deep RDR recipe. "
        "This is selection among loss/output-slope choices, not an architecture search. All 20 bins are fixed "
        "in advance. Full rankings and seed-level rows are included as CSVs.", "",
        "## Real-versus-real controls and stopping limits", "",
        shared._table(null_table), "",
        "Each selected recipe is also fitted to the same subject-separated real-versus-real split "
        "(1,001 versus 1,001 assessment rows), where the population RDR target is 1. "
        "Reused controls and any identical selected recipes share data and initialization seeds; they are not "
        "independent replications. Epoch-cap flags describe numerical stopping limits and are retained rather "
        "than interpreted as additional evidence of convergence.", "",
        shared._table(cap_flags.assign(label=cap_flags.method.map(LABELS))[
            ["label", "study", "fits", "epoch_cap_fits", "maximum_epochs_run", "maximum_best_epoch"]]), "",
        "![Real-versus-real score and cell diagnostics](rdr_null_controls.png)", "",
        "## Provenance and files", "",
        f"Private study archive: `{out}`. Independent verification passed before report generation. "
        "The report separately reproduces the saved selection, Brier, AUC, fixed-cell quantities and null MSE "
        "from archived scores and checks fit completion receipts. Original reference archives are reused in place.", "",
        *[f"- {LABELS[m]}: `{roots[m]}`" for m in METHODS], "",
        "Aggregate files: [comparison](comparison.csv), [all metrics](metrics_aggregate.csv), "
        "[controls](null_controls.csv), [selection ranking](selection_ranking.csv), "
        "[fixed-width cells](fixed_width_cells.csv), [stopping flags](epoch_cap_flags.csv), "
        "[split counts](split_counts.csv). Public export excludes row-level scores, IDs, abundance arrays and checkpoints.", "",
    ]
    (summary / "report.md").write_text("\n".join(lines))
    files = {p.name: sha(p) for p in sorted(summary.iterdir()) if p.is_file() and p.name != "manifest.json"}
    manifest = {"status": "complete", "new_selection_fits": 240, "new_final_fits": 15, "new_null_fits": 15,
                "reference_methods_reused": list(REFERENCE_ROOTS), "selected": selected,
                "verification_sha256": sha(verification), "protocol_sha256": sha(out / "protocol.json"),
                "files_sha256": files}
    write(summary / "manifest.json", manifest)
    return manifest


def export(out: Path, destination: Path):
    """Copy only a completed aggregate report into a fresh public directory."""
    out, destination = Path(out).resolve(), Path(destination).resolve()
    verification = _gate(out)
    seal_path = out / "COMPLETE.json"
    if not seal_path.is_file() or read(seal_path).get("status") != "complete":
        raise ValueError("Public export requires the completed outer archive seal")
    seal = read(seal_path)
    if seal.get("verification_sha256") != sha(verification):
        raise ValueError("Outer seal verification receipt changed")
    _check_receipts(out, seal["outputs_sha256"])
    summary = out / "summary"
    manifest = read(summary / "manifest.json")
    if manifest.get("status") != "complete" or manifest["verification_sha256"] != sha(verification):
        raise ValueError("Report manifest is incomplete or verification has changed")
    if manifest["protocol_sha256"] != sha(out / "protocol.json"):
        raise ValueError("Protocol changed after report generation")
    for name in manifest["files_sha256"]:
        if name not in PUBLIC_ARTIFACTS:
            raise ValueError("Public report manifest includes an unapproved artifact")
    _check_receipts(summary, manifest["files_sha256"])
    if destination.exists():
        raise FileExistsError("Use a fresh public report destination")
    # All metadata are study-level receipts/settings; no array manifests or IDs.
    metadata = {name: out / name for name in ("protocol.json", "verification.json", "COMPLETE.json",
                                             "generation_complete.json", "fits_complete.json", "rdr_config.json")
                if (out / name).is_file()}
    protocol = read(out / "protocol.json")
    references = protocol.get("references", REFERENCE_ROOTS)
    roots = {method: out / "analyses" / method for method in NEW_METHODS}
    for method, default in REFERENCE_ROOTS.items():
        spec = references.get(method, default)
        roots[method] = Path(spec if isinstance(spec, str) else spec.get("analysis_root", spec.get("path", default)))
    for method, root in roots.items():
        if method not in NEW_METHODS:
            _check_receipts(root, read(root / "selection/freeze.json")["outputs"])
        for name in ("config.json", "selection/selection.json", "selection/freeze.json"):
            source = root / name
            if not source.is_file():
                raise ValueError(f"Missing public recipe metadata: {source}")
            if method not in NEW_METHODS and name in ("config.json", "selection/freeze.json"):
                expected = protocol.get("inputs_sha256", {}).get(str(source))
                if expected is not None and sha(source) != expected:
                    raise ValueError(f"Changed reference metadata: {source}")
            metadata[f"recipes/{method}/{name}"] = source
    destination.mkdir(parents=True)
    for name in (*manifest["files_sha256"], "manifest.json"):
        shutil.copy2(summary / name, destination / name)
    for name, source in metadata.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _check_receipts(destination, manifest["files_sha256"])
    outputs = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    receipt = {"status": "exported", "destination": str(destination), "files": len(outputs),
               "archive_seal_sha256": sha(seal_path), "files_sha256": outputs,
               "private_artifacts_excluded": ["scores", "sample IDs", "abundance arrays", "model checkpoints"]}
    write(destination / "export_manifest.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["report", "export"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "export" and args.destination is None:
        parser.error("export requires --destination")
    result = report(args.out) if args.command == "report" else export(args.out, args.destination)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
