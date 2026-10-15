#!/usr/bin/env python3
"""Freeze shared real-community rules, then assess four existing AGP RDRs.

No generator or RDR is refitted. A multioutput tree summarizes the four frozen
ensemble scores on common real development observations. All communities and
assessment counts are retained; scores and subject identifiers stay private.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import four_generator_features as features
import selected_workflow as shared

METHODS = ("bernoulli_mixture_icfm", "autoregressive_icfm", "midasim_mask_icfm", "midasim")
ROOTS = {
    METHODS[0]: Path("/cwork/yx306/RDR/AGP/mask_rdr_20261012/analyses") / METHODS[0],
    METHODS[1]: Path("/cwork/yx306/RDR/AGP/mask_rdr_20261012/analyses") / METHODS[1],
    METHODS[2]: Path("/cwork/yx306/RDR/AGP/icfm_zero_20261007/analysis"),
    METHODS[3]: Path("/cwork/yx306/RDR/AGP/midasim_full_20261004/analysis"),
}
sha, read, write = shared.sha256, shared.read_json, shared.write_json


def pin(path, inputs, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError(f"Changed input: {path}")
    if str(path) in inputs and inputs[str(path)] != digest:
        raise ValueError(f"Input changed during analysis: {path}")
    inputs[str(path)] = digest


def array(method, role, side, inputs):
    root = ROOTS[method]
    path = root / "data/protocol.json"
    pin(path, inputs)
    spec = read(path)["studies"]["final"]["arrays"][f"{role}_{side}"]
    source = root / spec["path"]
    pin(source, inputs, spec["sha256"])
    result = np.load(source, allow_pickle=False)
    if list(result.shape) != spec["shape"] or not np.isfinite(result).all():
        raise ValueError(f"Invalid input array: {source}")
    return result


def manifest(method, inputs):
    root = ROOTS[method]
    spec = read(root / "data/protocol.json")["studies"]["final"]["manifest"]
    path = root / spec["path"]
    pin(path, inputs, spec["sha256"])
    return pd.read_csv(path, keep_default_na=False, dtype=str)


def real_identity(method, role, inputs):
    table = manifest(method, inputs)
    table = table[table.distribution.eq("P") & table.role.eq(role)].copy()
    table["array_row"] = table.array_row.astype(int)
    table = table.sort_values("array_row")
    if table.array_row.tolist() != list(range(len(table))):
        raise ValueError("Manifest array order differs")
    return table.subject_id.to_numpy(str)


def scores(method, raw, inputs):
    root = ROOTS[method]
    pin(root / "COMPLETE.json", inputs)
    seal = read(root / "COMPLETE.json")
    if seal["status"] != "complete" or seal["final_fits"] != 5:
        raise ValueError("Upstream RDR study is incomplete")
    columns = []
    for repeat in range(5):
        folder = root / "final" / f"{repeat:02d}"
        pin(folder / "COMPLETE.json", inputs, seal["assessment_completions"][f"final/{repeat:02d}/COMPLETE.json"])
        receipt = read(folder / "COMPLETE.json")
        if receipt["status"] != "complete" or receipt["stage"] != "final":
            raise ValueError("Upstream fit incomplete")
        for name in ("model.pt", "transform.npz"):
            pin(folder / name, inputs, receipt["outputs"][name])
        saved = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
        state = dict(np.load(folder / "transform.npz", allow_pickle=False))
        model = shared.make_model(raw.shape[1], saved["config"], repeat, "final")
        model.load_state_dict(saved["model"])
        columns.append(shared.predict(model, raw, state, saved["task"]["output_alpha"], "cpu"))
    return np.column_stack(columns)


def verify_sources(inputs):
    for name, expected in inputs.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def prepare(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh shared-community directory")
    output.mkdir(parents=True)
    inputs = {}
    for path in (Path(__file__), Path(features.__file__), Path(shared.__file__)):
        pin(path, inputs)
    train = array(METHODS[0], "train", "p", inputs)
    stop = array(METHODS[0], "stopping", "p", inputs)
    taxa_spec = read(ROOTS[METHODS[0]] / "data/protocol.json")["taxa"]
    taxa_path = ROOTS[METHODS[0]] / taxa_spec["path"]
    pin(taxa_path, inputs, taxa_spec["sha256"])
    groups = features.build_groups(train, pd.read_csv(taxa_path, keep_default_na=False))
    heldout_x, names = features.feature_matrix(stop, groups)
    train_subjects = real_identity(METHODS[0], "train", inputs)
    stop_subjects = real_identity(METHODS[0], "stopping", inputs)
    if set(train_subjects) & set(stop_subjects):
        raise ValueError("Development subject overlap")
    heldout_y = []
    for method in METHODS:
        if not np.array_equal(train, array(method, "train", "p", inputs)) or not np.array_equal(stop, array(method, "stopping", "p", inputs)):
            raise ValueError("The four real development arrays differ")
        if not np.array_equal(train_subjects, real_identity(method, "train", inputs)) or not np.array_equal(stop_subjects, real_identity(method, "stopping", inputs)):
            raise ValueError("Real development subject identities differ")
        print(f"Scoring frozen RDR development: {method}", flush=True)
        heldout_y.append(scores(method, stop, inputs).mean(axis=1))
    heldout_y = np.column_stack(heldout_y)
    # The existing RDR stopping observations were excluded from gradient updates.
    # Split their subjects again to fit and select the explanatory tree. Their
    # earlier use for RDR checkpoint choice remains an explicit limitation.
    subject_order = np.random.default_rng(2026101302).permutation(np.unique(stop_subjects))
    discovery_subjects = set(subject_order[:int(.7 * len(subject_order))])
    discovery = np.array([subject in discovery_subjects for subject in stop_subjects])
    validation = ~discovery
    train_x, stop_x = heldout_x[discovery], heldout_x[validation]
    train_y, stop_y = heldout_y[discovery], heldout_y[validation]
    candidates, trees = [], []
    for depth in (2, 3, 4):
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=.05, random_state=2026101301)
        tree.fit(train_x, train_y)
        mse = np.mean((tree.predict(stop_x) - stop_y) ** 2, axis=0)
        candidates.append({"depth": depth, "leaves": int(tree.get_n_leaves()), "stopping_mse": float(mse.mean()),
                           "mse_by_method": dict(zip(METHODS, mse.tolist()))})
        trees.append(tree)
    index = min(range(3), key=lambda i: (candidates[i]["stopping_mse"], candidates[i]["depth"]))
    tree = trees[index]
    joblib.dump(tree, output / "tree.joblib")
    leaf_rules = features.rules(tree, names)
    ordered = sorted(map(int, leaf_rules))
    community_ids = {str(leaf): f"C{i+1:02d}" for i, leaf in enumerate(ordered)}
    write(output / "rules.json", {community_ids[str(leaf)]: {"leaf": leaf, "rule": leaf_rules[str(leaf)]} for leaf in ordered})
    write(output / "groups.json", groups)
    np.savez_compressed(output / "development.npz", train_x=train_x, stop_x=stop_x, train_y=train_y, stop_y=stop_y,
                        train_subjects=train_subjects, stop_subjects=stop_subjects,
                        discovery_indices=np.flatnonzero(discovery), validation_indices=np.flatnonzero(validation),
                        discovery_subjects=stop_subjects[discovery], validation_subjects=stop_subjects[validation])
    protocol = {"status": "frozen", "created_utc": shared.now(), "methods": list(METHODS), "selected": candidates[index],
                "candidates": candidates, "feature_names": names, "groups": groups, "real_train_n": len(train),
                "real_stopping_n": len(stop), "feature_count": len(names), "minimum_training_leaf_fraction": .05,
                "tree_training_n": len(train_x), "tree_validation_n": len(stop_x),
                "tree_subject_split_seed": 2026101302, "tree_training_subject_fraction": .7,
                "tree_sample_source": "Original RDR stopping P; excluded from gradient updates but used for RDR checkpoint choice",
                "training_target": "Four outputs: each separately selected RDR ensemble mean on the same real stopping observations",
                "selection": "Smallest mean squared stopping error across the four RDR outputs; smaller depth breaks ties",
                "development_revision": "A preparation-only tree on in-fit RDR training scores showed negative stopping R2 in three methods. Before shared-community assessment, the partition was revised to use a fixed subject split of RDR stopping observations. The initial development trial is preserved separately; no assessment-guided tree revision occurred.",
                "assessment_compositions_or_scores_used_for_selection": False, "retrospective": True,
                "region_order": "Increasing tree leaf index, fixed before assessment",
                "scope": "Shared real-community partition; P-only tree does not discover exclusively generated modes. Fixed fitted-score interpretation, not causal attribution or true pointwise RDR recovery.",
                "upstream": inputs, "files_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    verify_sources(inputs)
    write(output / "FREEZE.json", protocol)
    print(json.dumps({"status": "frozen", "selected": candidates[index], "output": str(output)}), flush=True)


def region_summary(p_scores, q_scores, p_labels, q_labels, leaf, method, community, rule):
    pm, qm = p_labels == leaf, q_labels == leaf
    pp, qq = float(pm.mean()), float(qm.mean())
    total = pp + qq
    if total == 0:
        raise ValueError("Empty assessment region in both sources")
    weighted = (p_scores[pm].sum(axis=0) / len(p_scores) + q_scores[qm].sum(axis=0) / len(q_scores)) / total
    theta = 2 * pp / total
    return {"method": method, "community": community, "leaf": leaf, "rule": rule,
            "n_p": len(p_scores), "n_q": len(q_scores), "p_count": int(pm.sum()), "q_count": int(qm.sum()),
            "p_fraction": pp, "q_fraction": qq, "q_minus_p": qq - pp, "balanced_mass": total / 2,
            "empirical_region_rdr": theta, "mean_fitted_rdr": float(weighted.mean()),
            "fitted_rdr_seed_sd": float(weighted.std(ddof=1)), "empirical_abs_gap": abs(float(weighted.mean()) - theta),
            "mean_real_rdr": float(p_scores[pm].mean()) if pm.any() else None,
            "mean_generated_rdr": float(q_scores[qm].mean()) if qm.any() else None,
            "small_count": bool(min(pm.sum(), qm.sum()) < 30)}


def assess(output):
    output = Path(output).resolve()
    if (output / "ASSESSMENT_COMPLETE.json").exists():
        raise FileExistsError("Assessment already completed")
    frozen = read(output / "FREEZE.json")
    if frozen["status"] != "frozen" or frozen["assessment_compositions_or_scores_used_for_selection"]:
        raise ValueError("Shared rules were not frozen before assessment")
    verify_sources(frozen["upstream"])
    for name, digest in frozen["files_sha256"].items():
        if sha(output / name) != digest:
            raise ValueError(f"Shared community decision changed: {name}")
    inputs = dict(frozen["upstream"])
    inputs[str(output / "FREEZE.json")] = sha(output / "FREEZE.json")
    p = array(METHODS[0], "test", "p", inputs)
    subjects = real_identity(METHODS[0], "test", inputs)
    development = np.load(output / "development.npz", allow_pickle=False)
    if set(subjects) & (set(development["train_subjects"]) | set(development["stop_subjects"])):
        raise ValueError("Assessment subject overlap")
    tree = joblib.load(output / "tree.joblib")
    px, _ = features.feature_matrix(p, frozen["groups"])
    plabels = tree.apply(px)
    tree_pred = tree.predict(px)
    rules = read(output / "rules.json")
    rows, fidelity, replays, real_scores = [], [], [], []
    private = {"p_labels": plabels, "subjects": subjects}
    for column, method in enumerate(METHODS):
        if not np.array_equal(p, array(method, "test", "p", inputs)) or not np.array_equal(subjects, real_identity(method, "test", inputs)):
            raise ValueError("Four-way real assessment join differs")
        q = array(method, "test", "q", inputs)
        qx, _ = features.feature_matrix(q, frozen["groups"])
        qlabels = tree.apply(qx)
        ps, qs = scores(method, p, inputs), scores(method, q, inputs)
        root = ROOTS[method]
        for repeat in range(5):
            folder = root / "final" / f"{repeat:02d}"
            marker = read(folder / "COMPLETE.json")
            pin(folder / "predictions.npz", inputs, marker["outputs"]["predictions.npz"])
            with np.load(folder / "predictions.npz", allow_pickle=False) as stored:
                difference = max(float(np.max(np.abs(ps[:, repeat] - stored["test_p"]))),
                                 float(np.max(np.abs(qs[:, repeat] - stored["test_q"]))))
                if difference > 1e-12:
                    raise ValueError("Existing assessment score replay differs")
            replays.append({"method": method, "repeat": repeat, "max_abs_difference": difference})
        real_scores.append(ps.mean(axis=1))
        private[method + "_p_scores"] = ps
        private[method + "_q_scores"] = qs
        private[method + "_q_labels"] = qlabels
        for community, specification in rules.items():
            rows.append(region_summary(ps, qs, plabels, qlabels, specification["leaf"], method, community, specification["rule"]))
        target = ps.mean(axis=1)
        mse = float(np.mean((tree_pred[:, column] - target) ** 2))
        fidelity.append({"method": method, "real_assessment_mse": mse,
                         "real_assessment_r2": 1 - mse / float(np.var(target)), "real_n": len(p),
                         "target": "Ensemble mean fitted RDR on real P only"})
        print(f"Assessed shared communities: {method}", flush=True)
    real_scores = np.column_stack(real_scores)
    private["real_scores"] = real_scores
    comparisons = []
    for i, j in itertools.combinations(range(4), 2):
        a, b = real_scores[:, i], real_scores[:, j]
        comparisons.append({"method_a": METHODS[i], "method_b": METHODS[j], "real_n": len(p),
                            "spearman": float(spearmanr(a, b).statistic),
                            "mean_absolute_score_difference": float(np.abs(a - b).mean()),
                            "opposite_fitted_direction_fraction": float(((a - 1) * (b - 1) < 0).mean())})
    summary = output / "summary"
    summary.mkdir()
    frame = pd.DataFrame(rows)
    totals = frame.groupby("method")[["p_count", "q_count", "p_fraction", "q_fraction", "balanced_mass"]].sum()
    if not np.allclose(totals.to_numpy(), np.tile([2002, 2089, 1, 1, 1], (4, 1)), atol=1e-12):
        raise ValueError("Regions fail to partition assessment sources")
    frame.to_csv(summary / "shared_communities.csv", index=False)
    pd.DataFrame(fidelity).to_csv(summary / "tree_fidelity.csv", index=False)
    pd.DataFrame(comparisons).to_csv(summary / "paired_real_scores.csv", index=False)
    pd.DataFrame(replays).to_csv(summary / "score_replay.csv", index=False)
    pd.DataFrame([{"community": k, **v} for k, v in rules.items()]).to_csv(summary / "community_rules.csv", index=False)
    np.savez_compressed(output / "assessment_private.npz", **private)
    verify_sources(inputs)
    write(output / "ASSESSMENT_COMPLETE.json", {"status": "complete", "created_utc": shared.now(),
          "freeze_sha256": sha(output / "FREEZE.json"), "inputs_sha256": inputs,
          "communities": len(rules), "assessment_rows_per_method": {"P": len(p), "Q": len(q)},
          "real_subject_overlap": 0, "score_replays": len(replays), "max_score_replay_difference": max(r["max_abs_difference"] for r in replays),
          "files_sha256": {str(path.relative_to(output)): sha(path) for path in [output / "assessment_private.npz", *summary.iterdir()]}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "assess"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        (prepare if arguments.action == "prepare" else assess)(arguments.output)
