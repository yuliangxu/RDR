#!/usr/bin/env python3
"""Retrospective, fixed-score-partition C.1/C.2 calibration for CelebA RDR.

CPU only: reuse hash-verified saved scores from frozen feature or pixel models.
No training, score extraction, partition tuning, or evaluation of true coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from utils.calibration import bin_scores, calibrate_counts


SOURCE_ROOT = Path("/cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid")
DEFAULT_OUTPUT = Path("/cwork/yx306/RDR/JRSSB/CI/celeba_feature_20260918")
DEFAULT_PIXEL_OUTPUT = DEFAULT_OUTPUT.with_name("celeba_pixel_20260918")
SOURCE_RESULT_SHA256 = "d25183e20a9338a947e1fee4c0c765f8447dd59ee491130975af410c4742a0be"
BRANCHES = ("lower", "upper")
METHODS = ("c1", "c2", "c2_joint_two_contrasts")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path}")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def integer(value):
    number = float(value)
    require(np.isfinite(number) and number == int(number), f"Invalid integer: {value}")
    return int(number)


def unique_map(rows, key="source_id"):
    result = {row[key]: row for row in rows}
    require(len(result) == len(rows), f"Duplicate {key}")
    return result


class Provenance:
    def __init__(self):
        self.hashes = {}

    def verify(self, path, expected):
        path = Path(path)
        actual = sha256(path)
        require(actual == expected, f"SHA256 mismatch: {path}")
        self.hashes[str(path)] = actual
        return path

    def json(self, path, expected):
        return json.loads(self.verify(path, expected).read_text())


def load_sources(root, level="feature", calibration_mode="split"):
    """Audit score provenance and image/identity/seed separation before splitting."""
    require(level in ("feature", "pixel"), "Unknown RDR level")
    score_field = f"{level}_rdr"
    representation = "pool3_2048" if level == "feature" else "pixels64_cnn"
    full_test = calibration_mode == "full_test"
    provenance = Provenance()
    published = root / "rdr/real_generator/single_test"
    source = provenance.json(published / "result.json", SOURCE_RESULT_SHA256)
    require(source["status"] == "complete", "Source analysis is incomplete")
    inputs = source["input_sha256"]

    def locked_json(relative):
        path = root / relative
        return provenance.json(path, inputs[str(path)])

    design = locked_json("locks/real_generator_rdr_selection.json")
    selection = locked_json("locks/agent3_final_test_selection.json")
    final = locked_json("locks/agent3_final_test_result.json")
    provenance.verify(selection["design_selection_lock_path"], selection["design_selection_lock_sha256"])
    provenance.verify(final["final_test_selection_path"], final["final_test_selection_sha256"])
    provenance.verify(selection["final_test_protocol_path"], selection["final_test_protocol_sha256"])
    provenance.verify(selection["generated_test_role_lock_path"], selection["generated_test_role_lock_sha256"])
    pair = provenance.json(design["pair_selection_lock_path"], design["pair_selection_lock_sha256"])
    generator = provenance.json(root / "locks/generator_pair_lock.json", pair["pair_lock_sha256"])
    require(selection["selection_used_fid_or_rdr_scores"] is False, "Final-test selection used scores")
    require(selection["sample_size_per_distribution"] == 9000, "Unexpected final-test size")

    manifests = {}
    audit = []
    for distribution in ("real", *BRANCHES):
        item = selection["selection"][distribution]
        path = provenance.verify(item["selected_manifest_path"], item["selected_manifest_sha256"])
        test = read_csv(path)
        require(len(test) == 9000, f"Unexpected {distribution} test size")
        require(all(row["role"] == "test" for row in test), "Wrong original manifest role")
        unique_map(test)
        manifests[distribution] = test
        extra_key = "identity" if distribution == "real" else "latent_seed"
        test_ids = {row["source_id"] for row in test}
        test_extra = {integer(row[extra_key]) for row in test}
        if distribution != "real":
            require(len(test_extra) == len(test), "Repeated generated latent seed")
        prior_manifests = {}
        for role in ("train", "validation", "design"):
            prior_item = design["selection"][role][distribution]
            prior_path = provenance.verify(prior_item["selected_manifest_path"], prior_item["selected_manifest_sha256"])
            prior = read_csv(prior_path)
            require(len(prior) == prior_item["sample_size"], "Prior sample-size mismatch")
            require(all(row["role"] == role for row in prior), "Prior-role mismatch")
            unique_map(prior)
            prior_manifests[role] = prior
            id_overlap = len(test_ids & {row["source_id"] for row in prior})
            extra_overlap = len(test_extra & {integer(row[extra_key]) for row in prior})
            require(id_overlap == extra_overlap == 0, f"Test overlaps {role}: {distribution}")
            audit.append({"distribution": distribution, "prior_role": role, "prior_n": len(prior),
                          "test_n": len(test), "source_id_overlap": id_overlap,
                          "additional_key": extra_key, "additional_key_overlap": extra_overlap})
        if full_test:
            historical_design = prior_manifests["design"]
            require(len(historical_design) == 9000, "Unexpected historical design size")
            manifests[distribution] = historical_design + test
            unique_map(manifests[distribution])
            for role in ("train", "validation"):
                prior = prior_manifests[role]
                id_overlap = len({row["source_id"] for row in historical_design}
                                 & {row["source_id"] for row in prior})
                extra_overlap = len({integer(row[extra_key]) for row in historical_design}
                                    & {integer(row[extra_key]) for row in prior})
                require(id_overlap == extra_overlap == 0, f"Historical design overlaps {role}: {distribution}")
                audit.append({"distribution": distribution, "prior_role": role, "prior_n": len(prior),
                              "audited_role": "historical_design", "test_n": len(historical_design),
                              "source_id_overlap": id_overlap, "additional_key": extra_key,
                              "additional_key_overlap": extra_overlap})
            if distribution != "real":
                require(len({integer(row[extra_key]) for row in manifests[distribution]}) == 18000,
                        "Repeated generated latent seed in full test set")

    scores, checkpoints = {}, {}
    for branch in BRANCHES:
        lock_path = root / f"locks/real_vs_{branch}_{level}_result.json"
        lock = locked_json(f"locks/real_vs_{branch}_{level}_result.json")
        require(final["model_result_lock_sha256"][f"{level}_{branch}"] == sha256(lock_path), "Model lock mismatch")
        require(lock["status"] == "complete" and lock["representation"] == representation, "Wrong model")
        require(lock["p_distribution"] == "real" and lock["q_distribution"] == branch, "Wrong model orientation")
        provenance.verify(lock["checkpoint_path"], lock["checkpoint_sha256"])
        provenance.verify(lock["selection_lock_path"], lock["selection_lock_sha256"])
        checkpoints[branch] = {key: lock[key] for key in (
            "checkpoint_path", "checkpoint_sha256", "best_epoch", "restore_mode", "representation",
            "n_train_p", "n_train_q", "n_validation_p", "n_validation_q")}
        score_name = f"test_scores_real_vs_{branch}.csv"
        score_path = provenance.verify(published / score_name, source["outputs"][score_name])
        all_scores = read_csv(score_path)
        require(len(all_scores) == 36000, "Unexpected combined score count")
        require({row["original_role"] for row in all_scores} == {"design", "test"}, "Unexpected original roles")
        retained = all_scores if full_test else [row for row in all_scores if row["original_role"] == "test"]
        require(len(retained) == (36000 if full_test else 18000), "Unexpected retained score count")
        by_id = unique_map(retained)
        for distribution in ("real", branch):
            expected = unique_map(manifests[distribution])
            found = {row["source_id"]: row for row in retained if row["distribution"] == distribution}
            require(set(found) == set(expected), f"Score/manifest ID mismatch: {distribution}")
            extra = "identity" if distribution == "real" else "latent_seed"
            for source_id, row in found.items():
                require(integer(row[extra]) == integer(expected[source_id][extra]), "Identity/seed mismatch")
                require(row["source"] == expected[source_id]["source"], "Source-label mismatch")
                require(row["original_role"] == expected[source_id]["role"], "Original-role mismatch")
        original_item = final["outputs"][f"predictions_{branch}"]
        original = read_csv(provenance.verify(original_item["path"], original_item["sha256"]))
        test_ids = {row["source_id"] for row in retained if row["original_role"] == "test"}
        require(set(unique_map(original)) == test_ids, "Original prediction IDs differ")
        for row in original:
            require(np.isclose(float(row[score_field]), float(by_id[row["source_id"]][score_field]),
                               rtol=0, atol=6e-8), "Saved score differs beyond float32 CSV roundoff")
        if full_test:
            original_design = read_csv(provenance.verify(lock["prediction_path"], lock["prediction_sha256"]))
            require(inputs[lock["prediction_path"]] == lock["prediction_sha256"], "Design score lock mismatch")
            design_ids = {row["source_id"] for row in retained if row["original_role"] == "design"}
            require(set(unique_map(original_design)) == design_ids, "Original design prediction IDs differ")
            for row in original_design:
                require(row["distribution"] == by_id[row["source_id"]]["distribution"], "Design distribution mismatch")
                require(np.isclose(float(row["rdr"]), float(by_id[row["source_id"]][score_field]),
                                   rtol=0, atol=6e-8), "Design score differs beyond float32 CSV roundoff")
        # bin_scores also checks finite values and the exact [0,2] support.
        bin_scores([float(row[score_field]) for row in retained], [0, 2])
        scores[branch] = retained
    return provenance, scores, manifests, checkpoints, audit, generator


def split_rows(scores, manifests, seed, edges, level="feature", calibration_mode="split"):
    score_field = f"{level}_rdr"
    full_test = calibration_mode == "full_test"
    attached_role = "test" if full_test else "query"
    identities = np.sort(np.array([integer(row["identity"]) for row in manifests["real"]]))
    identities = np.unique(identities)
    permuted = identities if full_test else np.random.default_rng(seed).permutation(identities)
    calibration_identities = set(permuted[:len(permuted) // 2].tolist())
    calibration, query, counts, split_audit = [], [], {}, []
    for branch_index, branch in enumerate(BRANCHES, 1):
        q_ids = np.array(sorted(row["source_id"] for row in manifests[branch]))
        permuted_q = q_ids if full_test else np.random.default_rng(seed + branch_index).permutation(q_ids)
        calibration_q_ids = set(permuted_q[:len(permuted_q) // 2])
        for row in sorted(scores[branch], key=lambda item: (item["distribution"], item["source_id"])):
            real = row["distribution"] == "real"
            is_calibration = full_test or (integer(row["identity"]) in calibration_identities if real
                                           else row["source_id"] in calibration_q_ids)
            score = float(row[score_field])
            record = {"branch": branch, "contrast": f"P_vs_Q_{branch}",
                      "distribution": "P" if real else f"Q_{branch}", "source": row["source"],
                      "source_id": row["source_id"], "identity": integer(row["identity"]) if real else "",
                      "latent_seed": "" if real else integer(row["latent_seed"]),
                      "original_role": row["original_role"], "ci_role": "calibration" if is_calibration else "query",
                      score_field: score, "bin_index": int(bin_scores(score, edges))}
            (calibration if is_calibration else query).append(record)
            if full_test:
                query.append({**record, "ci_role": "test_attached_ci"})
        counts[branch] = {}
        for role, records in (("calibration", calibration), (attached_role, query)):
            subset = [row for row in records if row["branch"] == branch]
            counts[branch][role] = {"n_p": sum(row["distribution"] == "P" for row in subset),
                                    "n_q": sum(row["distribution"] != "P" for row in subset),
                                    "p_identities": len({row["identity"] for row in subset if row["distribution"] == "P"})}
        cal_ids = {row["source_id"] for row in calibration if row["branch"] == branch}
        query_ids = {row["source_id"] for row in query if row["branch"] == branch}
        require(cal_ids == query_ids if full_test else not cal_ids & query_ids,
                "Wrong calibration/attachment ID relation")
        for distribution in ("P", f"Q_{branch}"):
            extra = "identity" if distribution == "P" else "latent_seed"
            cal_keys = {row[extra] for row in calibration if row["branch"] == branch and row["distribution"] == distribution}
            query_keys = {row[extra] for row in query if row["branch"] == branch and row["distribution"] == distribution}
            require(cal_keys == query_keys if full_test else not cal_keys & query_keys,
                    f"Wrong calibration/attachment {extra} relation")
            split_audit.append({"branch": branch, "distribution": distribution,
                                "source_id_overlap": sum(row["branch"] == branch and row["distribution"] == distribution
                                                         for row in calibration) if full_test else 0,
                                "additional_key": extra, "additional_key_overlap": len(cal_keys & query_keys)})
    require({row["source_id"] for row in calibration if row["branch"] == "lower" and row["distribution"] == "P"}
            == {row["source_id"] for row in calibration if row["branch"] == "upper" and row["distribution"] == "P"},
            "Real split differs between contrasts")
    return calibration, query, counts, split_audit


def intervals(calibration, query, counts, edges, alpha, attached_role="query"):
    cells = []
    for branch in BRANCHES:
        rows = [row for row in calibration if row["branch"] == branch]
        k_p = np.bincount([row["bin_index"] for row in rows if row["distribution"] == "P"], minlength=len(edges)-1)
        k_q = np.bincount([row["bin_index"] for row in rows if row["distribution"] != "P"], minlength=len(edges)-1)
        n_p, n_q = counts[branch]["calibration"]["n_p"], counts[branch]["calibration"]["n_q"]
        result = calibrate_counts(k_p, k_q, n_p, n_q, alpha=alpha)
        joint = calibrate_counts(k_p, k_q, n_p, n_q, alpha=alpha/2)
        for index in range(len(edges)-1):
            row = {"branch": branch, "bin_index": index, "left": float(edges[index]),
                   "right": float(edges[index+1]), "right_closed": index == len(edges)-2,
                   "n_p_calibration": n_p, "n_q_calibration": n_q,
                   "k_p": int(k_p[index]), "k_q": int(k_q[index]),
                   "p_hat": float(k_p[index]/n_p), "q_hat": float(k_q[index]/n_q)}
            row.update({key: value[index].item() for key, value in result.items()})
            row.update({f"c2_joint_two_contrasts_{side}": float(joint[f"c2_{side}"][index]) for side in ("lower", "upper")})
            cells.append(row)
    lookup = {(row["branch"], row["bin_index"]): row for row in cells}
    queries = [{**row, **{key: value for key, value in lookup[row["branch"], row["bin_index"]].items()
                         if key not in ("branch", "bin_index")}} for row in query]
    summary = []
    for branch in BRANCHES:
        branch_cells = [row for row in cells if row["branch"] == branch]
        pooled_name = f"pooled_{attached_role}"
        for distribution in ("P", f"Q_{branch}", pooled_name, "equal_mixture"):
            selected = [row for row in queries if row["branch"] == branch and
                        (distribution in (pooled_name, "equal_mixture") or row["distribution"] == distribution)]
            weights = np.ones(len(selected)) / len(selected)
            if distribution == "equal_mixture":
                weights = np.array([.5/counts[branch][attached_role]["n_p"] if row["distribution"] == "P"
                                    else .5/counts[branch][attached_role]["n_q"] for row in selected])
            for method in METHODS:
                low = np.array([row[f"{method}_lower"] for row in selected])
                high = np.array([row[f"{method}_upper"] for row in selected])
                widths = high-low
                order = np.argsort(widths)
                weighted_median = widths[order[min(np.searchsorted(np.cumsum(weights[order]), .5), len(order)-1)]]
                summary.append({"branch": branch, "method": method, "distribution": distribution,
                                f"n_{attached_role}": len(selected), "mean_width": float(np.dot(weights, widths)),
                                "median_width": float(weighted_median),
                                "fraction_excluding_one": float(np.dot(weights, (low > 1) | (high < 1))),
                                "fraction_below_one": float(np.dot(weights, high < 1)),
                                "fraction_above_one": float(np.dot(weights, low > 1)),
                                "n_cells_excluding_one": sum(row[f"{method}_lower"] > 1 or row[f"{method}_upper"] < 1 for row in branch_cells),
                                "n_cells": len(branch_cells),
                                f"c1_unavailable_{attached_role}_fraction": float(np.dot(weights, [row["c1_unavailable"] for row in selected]))})
    return cells, queries, summary


def plot_intervals(out, cells, edges, counts, alpha, level="feature", calibration_mode="split"):
    # Keep Matplotlib's cache inside the explicitly requested artifact directory.
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if calibration_mode == "full_test":
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.1]})
        centers = (edges[:-1] + edges[1:]) / 2
        for column, branch in enumerate(("upper", "lower")):
            ax, count_ax = axes[:, column]
            rows = [row for row in cells if row["branch"] == branch]
            low = np.array([row["c2_lower"] for row in rows])
            high = np.array([row["c2_upper"] for row in rows])
            ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                            color="#80b2d4", alpha=.55, label="C.2 simultaneous per contrast")
            ax.stairs(low, edges, color="#276795", lw=.8)
            ax.stairs(high, edges, color="#276795", lw=.8)
            available = np.array([not row["c1_unavailable"] for row in rows])
            estimate = np.array([row["estimate"] for row in rows])
            c1_low = np.array([row["c1_lower"] for row in rows])
            c1_high = np.array([row["c1_upper"] for row in rows])
            ax.errorbar(centers[available], estimate[available],
                        yerr=[estimate[available]-c1_low[available], c1_high[available]-estimate[available]],
                        fmt="none", ecolor="#a85015", elinewidth=1.4, capsize=3, label="C.1 when available")
            if np.any(~available):
                ax.scatter(centers[~available], np.full(np.sum(~available), .025), marker="x", color="#a85015",
                           transform=ax.get_xaxis_transform(), label="C.1 unavailable ([0,2] in CSV)", zorder=5)
            estimate[np.array([row["c1_empty"] for row in rows])] = np.nan
            ax.stairs(estimate, edges, baseline=None, color="#182f44", lw=1.5, label="Cell estimate (nonempty cells)")
            ax.plot([0, 2], [0, 2], ls=":", color=".45", lw=1, label="Original score reference")
            ax.axhline(1, color=".35", ls="--", lw=.8)
            label = "U" if branch == "upper" else "L"
            n = counts[branch]["calibration"]
            ax.set(title=f"$P$ vs $Q_{label}$ — full test calibration\n{n['n_p']:,} P, {n['n_q']:,} Q",
                   xlim=(0, 2), ylim=(0, 2))
            ax.grid(alpha=.15)
            for offset, field, color, label in ((-.018, "k_p", "#3b748f", "P"), (.018, "k_q", "#c68f48", "Q")):
                count_ax.bar(centers+offset, [row[field] for row in rows], width=.035, color=color, label=label)
            count_ax.set_yscale("symlog", linthresh=1)
            count_ax.set(xlabel=f"Frozen {level} RDR score", ylabel="Calibration count")
            count_ax.grid(axis="y", alpha=.15)
            count_ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
        axes[0, 0].set_ylabel(f"Cell-average {level} RDR")
        legend = {}
        for ax in axes[0]:
            handles, labels = ax.get_legend_handles_labels()
            legend.update(zip(labels, handles))
        fig.legend(legend.values(), legend.keys(), loc="lower center", ncol=3, frameon=False, fontsize=8)
        fig.suptitle(f"Nominal {100*(1-alpha):g}% intervals: fixed-map IID model; retrospective calibration", fontsize=11)
        fig.tight_layout(rect=(0, .11, 1, .95))
        fig.savefig(out / f"{level}_rdr_ci.png", dpi=200)
        fig.savefig(out / f"{level}_rdr_ci.pdf", metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharex=True, sharey=True)
    for ax, branch in zip(axes, ("upper", "lower")):
        rows = [row for row in cells if row["branch"] == branch]
        for method, color, label, opacity in (("c2", "#afc8de", "C.2 simultaneous per contrast", .65),
                                              ("c1", "#3d82b2", "C.1 marginal", .65)):
            low = np.array([row[f"{method}_lower"] for row in rows])
            high = np.array([row[f"{method}_upper"] for row in rows])
            ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                            color=color, alpha=opacity, label=label)
        ax.stairs([row["estimate"] for row in rows], edges, color="#182f44", lw=1.5, label="Calibrated cell estimate")
        ax.plot([0, 2], [0, 2], ls=":", color="0.4", lw=1, label="Original score reference")
        ax.axhline(1, color="0.3", ls="--", lw=.8)
        label = "U" if branch == "upper" else "L"
        n = counts[branch]["calibration"]
        ax.set_title(f"$P$ vs $Q_{label}$\nCalibration: {n['n_p']:,} P, {n['n_q']:,} Q")
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel=f"Frozen {level} RDR score")
        ax.grid(alpha=.15)
    axes[0].set_ylabel(f"Cell-average {level} RDR")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"Nominal {100*(1-alpha):g}% intervals under the IID image model", fontsize=12)
    fig.tight_layout(rect=(0, .15, 1, .95))
    fig.savefig(out / f"{level}_rdr_ci.png", dpi=200)
    fig.savefig(out / f"{level}_rdr_ci.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def report(out, protocol, cells, summary):
    level = protocol["level"]
    full_test = protocol.get("calibration_mode", "split") == "full_test"
    attached_role = "test" if full_test else "query"
    symbol, random_variable = ("z", "Z") if level == "feature" else ("x", "X")
    representation = "the Inception pool3 feature" if level == "feature" else "the complete 64 by 64 RGB image"
    default_output = DEFAULT_OUTPUT if level == "feature" else DEFAULT_PIXEL_OUTPUT
    if full_test:
        default_output = default_output.with_name(f"celeba_{level}_full_test_20260918")
    confidence = 100 * (1-protocol["alpha"])
    reproduction_command = shlex.join([
        "python3", f"experiments/JRSSB/CELEBA_{level}_ci.py", "--source-root", protocol["source_root"],
        *(["--calibration-mode", "full-test"] if full_test else []),
        *([] if full_test else ["--seed", str(protocol["seed"])]), "--alpha", str(protocol["alpha"]),
        "--output-dir", str(default_output.with_name(default_output.name + "_rerun"))])
    labels = {"upper": "$P$ vs $Q_U$", "lower": "$P$ vs $Q_L$"}
    text = [f"# CelebA {level} RDR: calibration confidence intervals", "",
            "These are retrospective, nominal image-level intervals for score-cell population averages. "
            f"The two frozen {level} RDR models compare real CelebA $P$ with $Q_U$ and $Q_L$.", "",
            "## Target and interpretation", "",
            f"For branch $b$, let ${random_variable}$ be {representation}, $M_b=(P+Q_b)/2$, "
            f"$r_{{0,b}}({symbol})=2p({symbol})/(p({symbol})+q_b({symbol}))$, "
            f"and $A_{{b,j}}=\\{{{symbol}:\\hat r_b({symbol})\\in I_j\\}}$. The target is", "",
            "$$\\theta_{b,j}=\\frac{2P(A_{b,j})}{P(A_{b,j})+Q_b(A_{b,j})}"
            f"=\\mathbb{{E}}_{{M_b}}[r_{{0,b}}({random_variable})\\mid {random_variable}\\in A_{{b,j}}].$$", "",
            "This target is defined for positive mixture-mass cells. Every prespecified cell remains in the output. "
            "These are not confidence intervals for an individual true RDR or for $H^2$. They are conditional "
            "on the fitted networks and fixed partition; model-training uncertainty is not included.", "",
            "## Frozen protocol and provenance", "",
            f"The partition consists of {protocol['bins']} equal-width bins on $[0,2]$, left closed and right open "
            "except that the last bin includes 2. " +
            ("No scores, calibration counts, or test attachments determine the boundaries." if full_test else
             "No scores, calibration counts, or query results determine the boundaries or split."), "",
            ("Calibration uses the complete current test set: 18,000 images per distribution, comprising 9,000 "
             "historical design images and 9,000 historical final-test images. Both components are disjoint by source "
             "ID and real identity/generated latent seed from the 60,000 training and 20,000 early-stopping validation "
             "images per distribution. Original roles remain recorded. The historical design component was used "
             "in earlier design diagnostics and protocol decisions; calling the combined pool the current test set "
             "does not erase that influence. This is retrospective calibration, not independent confirmation."
             if full_test else
            "Only the historical `original_role == test` rows are eligible: 9,000 images per distribution. "
            "The other 9,000 historical design images per distribution in the merged score files are excluded. "
            "Those test scores have already appeared in earlier analyses, so this is a retrospective reuse, "
            "not newly collected independent confirmation. Calibration and query images are disjoint from model training, "
            "early-stopping validation, historical design, and one another."), "",
            ("There is no calibration/query split and no random selection: all current test observations calibrate "
             "the cells. Those same observations receive their corresponding cell intervals in `test_intervals.csv`. "
             "These test-attached intervals reuse the calibration observations and do not constitute a separate "
             "evaluation of coverage. The same 18,000 real images are used in both contrasts."
             if full_test else
            f"Sorted numerical real identities are permuted with NumPy `default_rng({protocol['seed']})`; "
            "the first floor-half are calibration identities. All their images stay together, and the same real split "
            "is used in both contrasts. Sorted generated source IDs are independently permuted with seeds "
            f"{protocol['seed']+1} (lower) and {protocol['seed']+2} (upper), then split in half."), "",
            f"| Contrast | Calibration P images / identities | Calibration Q | {attached_role.title()} P images / identities | {attached_role.title()} Q |", "|---|---:|---:|---:|---:|"]
    for branch in ("upper", "lower"):
        c, q = protocol["counts"][branch]["calibration"], protocol["counts"][branch][attached_role]
        text.append(f"| {labels[branch]} | {c['n_p']} / {c['p_identities']} | {c['n_q']} | {q['n_p']} / {q['p_identities']} | {q['n_q']} |")
    text.extend(["", "Training and validation remain 60,000 and 20,000 images per distribution, respectively. "
                 f"All relevant source score, split, selection-lock, and {level}-model checkpoint hashes are verified; "
                 + ("the original design and final-test predictions agree within float32 CSV roundoff. " if full_test else
                    "the original final-test predictions agree within float32 CSV roundoff. ") +
                 "`protocol.json` records source hashes, exact counts, runtime versions, and role-overlap audits "
                 "before intervals are computed.", "",
                 "## Interval methods and limitations", "",
                 f"C.1 is the existing asymptotic {confidence:g}% marginal delta-method interval for each cell. "
                 f"C.2 uses Clopper–Pearson probability limits with Bonferroni allocation across all 20 cells "
                 f"and has nominal {confidence:g}% simultaneous coverage within each contrast under independent IID "
                 "P and Q image sampling. The additional `c2_joint_two_contrasts` columns call the same C.2 algorithm "
                 f"with alpha/2 for each contrast, yielding nominal {confidence:g}% simultaneous coverage for all 40 "
                 "cells under that model; this union bound does not require the two contrasts to be independent. "
                 "Neither claim is an empirically measured coverage rate here, because the true cell targets are unknown.", "",
                 *( ["**Design-reuse limitation:** these are nominal intervals under a fixed-map IID sampling model. "
                      "The historical design observations influenced earlier diagnostics and protocol decisions. "
                      "Conditioning on the currently frozen map does not establish independence after that prior reuse, "
                      "so unconditional post-selection coverage is not established for this combined pool.", ""]
                    if full_test else []),
                 ("**CelebA sampling limitation:** multiple images of one person remain within calibration; "
                  "separation from training/validation does not make calibration images IID. " if full_test else
                 "**CelebA sampling limitation:** multiple images of one person remain within calibration. Identity separation "
                 "prevents shared people across calibration and query, but does not make images within calibration IID. "
                 ) +
                 "Accordingly, exact finite-sample C.2 coverage is conditional on an IID image model and is not established "
                 "for identity-clustered CelebA sampling. C.1 also needs its usual large-sample regularity. "
                 "The unchanged C.1/C.2 algorithms do not correct identity clustering.", "",
                 "The estimate uses $\\hat p_j=k_{P,j}/n_P$, $\\hat q_j=k_{Q,j}/n_Q$, and "
                 "$\\hat\\theta_j=2\\hat p_j/(\\hat p_j+\\hat q_j)$; unequal sample sizes are normalized separately. "
                 "No pseudocounts are used. Empty calibration cells get estimate 1 as a storage convention and interval [0,2]; "
                 "zero estimated variance also makes C.1 unavailable with interval [0,2]. "
                 "Saved scores of exactly 0 or 2 are retained without clipping or jitter and enter the first or last cell.", "",
                 "## Results", "", f"![{level.title()} RDR calibration intervals]({level}_rdr_ci.png)", "",
                 ("The figure emphasizes per-contrast C.2 in blue and displays available C.1 intervals as orange "
                  "whiskers. Orange crosses mark C.1 unavailable cells; their CSV intervals remain [0,2]. Empty-cell "
                  "storage estimates of 1 are omitted from the estimate line. Lower panels show P and Q calibration "
                  "counts on a symmetric-log axis (linear at counts up to 1). The all-cell tables below retain exact "
                  "counts. Joint-two-contrast C.2 is available in the CSVs. Test summaries describe attachments on "
                  "the same observations used for calibration, with equal image weights within each distribution. "
                  "Pooled-test and equal-mixture weights agree because P and Q each contribute 18,000 observations. "
                  "Median widths are inverse-CDF weighted medians."
                  if full_test else
                 "The figure shows C.1 and per-contrast C.2; joint-two-contrast C.2 is available in the CSVs. "
                 "Within each distribution, query summaries weight each held-out image equally. Pooled-query summaries "
                 "use the actual query sample proportions. `equal_mixture` gives each distribution total weight 1/2, "
                 "matching the mixture $M_b$ despite unequal query sample sizes. Median widths are inverse-CDF "
                 "weighted medians."), "",
                 f"| Contrast | Method | {attached_role.title()} group | Mean width | Fraction excluding 1 | Cells excluding 1 / 20 |",
                 "|---|---|---|---:|---:|---:|"])
    for row in summary:
        text.append(f"| {labels[row['branch']]} | {row['method']} | {row['distribution']} | "
                    f"{row['mean_width']:.4f} | {row['fraction_excluding_one']:.4f} | {row['n_cells_excluding_one']} |")
    for branch in ("upper", "lower"):
        text.extend(["", f"### {labels[branch]}: all prespecified cells", "",
                     "| Score cell | P count | Q count | Cell estimate | C.1 | C.2 (per contrast) |", "|---|---:|---:|---:|---|---|"])
        for row in cells:
            if row["branch"] == branch:
                closing = "]" if row["right_closed"] else ")"
                estimate_text = "undefined (empty)" if full_test and row["c1_empty"] else f"{row['estimate']:.4f}"
                c1_text = ("unavailable [0,2]" if full_test and row["c1_unavailable"] else
                           f"[{row['c1_lower']:.4f}, {row['c1_upper']:.4f}]")
                text.append(f"| [{row['left']:.1f}, {row['right']:.1f}{closing} | {row['k_p']} | {row['k_q']} | "
                            f"{estimate_text} | {c1_text} | "
                            f"[{row['c2_lower']:.4f}, {row['c2_upper']:.4f}] |")
    text.extend(["", "## Reproduction and files", "",
                 "Run from the repository root, using a new output directory:", "", "```bash",
                 reproduction_command, "```", "",
                 "`source/experiments/JRSSB/CELEBA_feature_ci.py` and `source/utils/calibration.py` preserve the shared code used. "
                 + ("`source/experiments/JRSSB/CELEBA_pixel_ci.py` preserves the pixel entry point. " if level == "pixel" else "") +
                 "The saved copy can be run with the same command-line options. NumPy, SciPy, and Matplotlib are "
                 "required; saved scores suffice and no GPU, Torch, full input caches, or retraining is needed.", "",
                 "`cell_intervals.csv` contains all 40 cell estimates, normalized counts, probability bounds, "
                 f"C.1/C.2 and joint-two-contrast C.2. `{attached_role}_intervals.csv` maps every {attached_role} score to its cell interval. "
                 f"`calibration_manifest.csv` and `{attached_role}_manifest.csv` identify every image, identity/seed, score, and role. "
                 "`summary.csv` reports interval width and exclusion of 1, not true coverage. "
                 "`result.json` is written last and hashes every analysis artifact.", ""])
    (out / "report.md").write_text("\n".join(text))


def main(default_level="feature"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("feature", "pixel"), default=default_level)
    parser.add_argument("--calibration-mode", choices=("split", "full-test"), default="split",
                        help="split: historical final-test calibration/query halves; full-test: all current test rows calibrate and receive CIs")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--alpha", type=float, default=.05)
    args = parser.parse_args()
    args.calibration_mode = args.calibration_mode.replace("-", "_")
    full_test = args.calibration_mode == "full_test"
    attached_role = "test" if full_test else "query"
    require(0 < args.alpha < 1, "alpha must be strictly between 0 and 1")
    require(0 <= args.seed < 2**32-2, "seed must allow two additional nonnegative RNG seeds")
    default_output = DEFAULT_OUTPUT if args.level == "feature" else DEFAULT_PIXEL_OUTPUT
    if full_test:
        default_output = default_output.with_name(f"celeba_{args.level}_full_test_20260918")
    out = (args.output or default_output).resolve()
    require(out.is_relative_to(Path("/tmp")) or out.is_relative_to(Path("/cwork")),
            "Artifacts must be staged under /tmp or saved durably under /cwork")
    require(not out.exists(), "Output must be a new directory")
    edges = np.linspace(0, 2, 21)
    provenance, scores, manifests, checkpoints, audit, generator = load_sources(
        args.source_root.resolve(), args.level, args.calibration_mode)
    calibration, query, counts, split_audit = split_rows(scores, manifests, args.seed, edges, args.level, args.calibration_mode)
    script = Path(__file__).resolve()
    calibration_source = REPO / "utils/calibration.py"
    code_sources = [script, calibration_source, REPO / "utils/__init__.py"]
    if args.level == "pixel":
        code_sources.append(script.with_name("CELEBA_pixel_ci.py"))
    protocol = {"schema_version": 1, "analysis": f"retrospective_celeba_{args.level}_rdr_calibration",
                "level": args.level, "score_field": f"{args.level}_rdr",
                "calibration_mode": args.calibration_mode,
                "source_root": str(args.source_root.resolve()), "seed": args.seed, "alpha": args.alpha,
                "bins": 20, "edges": edges.tolist(), "bin_rule": "[left,right); final cell includes 2",
                "eligible_original_role": "test", "excluded_original_role": "design",
                "original_test_n_per_distribution": 9000, "historical_design_excluded_per_distribution": 9000,
                "real_split_rule": "default_rng(seed).permutation(sorted numerical unique identities), first floor(N/2) calibration",
                "generated_split_rule": "default_rng(seed+1 lower, seed+2 upper).permutation(sorted source_id), first floor(N/2) calibration",
                "score_dependent_split_or_partition": False, "same_real_split_across_contrasts": True,
                "counts": counts, "prior_role_overlap_audit": audit, "calibration_query_overlap_audit": split_audit,
                "checkpoint_provenance": checkpoints,
                "generator_truncation_psi": {branch: generator["branches"][branch]["truncation_psi"] for branch in BRANCHES},
                "target": f"2 P(A_bj)/(P(A_bj)+Q_b(A_bj)); {args.level}-score-cell mixture-average RDR",
                "methods": {"c1": "asymptotic marginal cell CI, alpha", "c2": "simultaneous 20 cells per contrast, alpha",
                            "c2_joint_two_contrasts": "simultaneous all 40 cells, alpha/2 per contrast"},
                "coverage_assumption": "Independent IID images within P and Q; not guaranteed with within-identity clustering",
                "retrospective_reuse": True, "training_uncertainty_included": False, "true_coverage_measured": False,
                "source_sha256": provenance.hashes,
                "code_sha256": {str(path): sha256(path) for path in code_sources},
                "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                            "matplotlib": version("matplotlib")}}
    if full_test:
        protocol.update({
            "analysis": f"retrospective_celeba_{args.level}_rdr_full_test_calibration",
            "seed": None, "eligible_original_role": ["design", "test"], "excluded_original_role": [],
            "current_test_n_per_distribution": 18000, "historical_design_included_per_distribution": 9000,
            "historical_design_excluded_per_distribution": 0,
            "real_split_rule": "No split; every current-test real image calibrates and receives its cell CI",
            "generated_split_rule": "No split; every current-test generated image calibrates and receives its cell CI",
            "same_real_split_across_contrasts": True, "calibration_equals_test": True,
            "test_attachments_independent_of_calibration": False, "historical_design_protocol_influence": True,
            "unconditional_post_selection_coverage_established": False,
            "coverage_assumption": "Fixed-map IID P/Q sampling; prior design/protocol reuse and within-identity dependence prevent an unconditional coverage claim",
            "calibration_test_attachment_audit": protocol.pop("calibration_query_overlap_audit"),
        })
    out.mkdir(parents=True, exist_ok=True)
    # Commit the rules and exact split accounting before any interval calculation.
    write_json(out / "protocol.json", protocol)
    write_csv(out / "calibration_manifest.csv", calibration)
    write_csv(out / f"{attached_role}_manifest.csv", query)
    for source in code_sources:
        target = out / "source" / source.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    cells, queries, summary = intervals(calibration, query, counts, edges, args.alpha, attached_role)
    write_csv(out / "cell_intervals.csv", cells)
    write_csv(out / f"{attached_role}_intervals.csv", queries)
    write_csv(out / "summary.csv", summary)
    plot_intervals(out, cells, edges, counts, args.alpha, args.level, args.calibration_mode)
    report(out, protocol, cells, summary)
    artifact_paths = [path for path in out.rglob("*") if path.is_file() and ".matplotlib" not in path.parts]
    output_hashes = {str(path.relative_to(out)): sha256(path) for path in sorted(artifact_paths)}
    write_json(out / "result.json", {"status": "complete", "analysis": protocol["analysis"],
                                    "counts": counts, "cells": len(cells), f"{attached_role}_rows": len(queries),
                                    "calibration_rows": len(calibration), "outputs": output_hashes})
    print(json.dumps({"output": str(out), "counts": counts, "cells": len(cells), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
