#!/usr/bin/env python3
"""Independent arithmetic and provenance audit of the four-generator AGP study.

Reads completed child archives without modifying them. Reconstructs community
features, tree selection, membership, source-normalized summaries and PRD curves.
The separate metric audit supplies the independent QR/nuclear-norm FD check.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

METHODS = ("bernoulli_mixture_icfm", "autoregressive_icfm", "midasim_mask_icfm", "midasim")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_hashes(mapping, base=None):
    for name, expected in mapping.items():
        path = Path(name) if base is None else Path(base) / name
        if sha(path) != expected:
            raise ValueError(f"Hash mismatch: {path}")


def close(actual, expected, atol=2e-12, rtol=2e-12):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True)


def load_raw(root, role, side):
    spec = read(root / "data/protocol.json")["studies"]["final"]["arrays"][f"{role}_{side}"]
    path = root / spec["path"]
    check_hashes({str(path): spec["sha256"]})
    x = np.load(path, allow_pickle=False)
    assert list(x.shape) == spec["shape"] and np.isfinite(x).all() and (x >= 0).all()
    return x


def identities(root, role):
    spec = read(root / "data/protocol.json")["studies"]["final"]["manifest"]
    path = root / spec["path"]
    check_hashes({str(path): spec["sha256"]})
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame[(frame.distribution == "P") & (frame.role == role)].copy()
    frame["array_row"] = frame.array_row.astype(int)
    frame = frame.sort_values("array_row")
    assert frame.array_row.tolist() == list(range(len(frame)))
    return frame.subject_id.to_numpy(str)


def features(raw, groups):
    """Reconstruct all 105 features without importing the producer module."""
    x = np.asarray(raw, float)
    x = x / x.sum(axis=1)[:, None]
    present = x > 0
    logs = np.zeros_like(x)
    logs[present] = np.log(x[present])
    columns = [present.sum(axis=1), -(x * logs).sum(axis=1), x.max(axis=1)]
    names = ["Richness", "Shannon entropy", "Maximum taxon abundance"]
    masses = []
    for group in groups:
        mass = x[:, group["indices"]].sum(axis=1)
        masses.append(mass)
        columns.extend((np.log((mass + 1e-6) / (1 - mass + 1e-6)), mass > 0,
                        present[:, group["indices"]].sum(axis=1)))
        names.extend(group["name"] + suffix for suffix in (" log mass/rest", " present", " richness"))
    for i, group in enumerate(groups):
        for j in range(i):
            columns.append(np.log((masses[i] + 1e-6) / (masses[j] + 1e-6)))
            names.append(group["name"] + "/" + groups[j]["name"] + " log balance")
    return np.column_stack(columns), names


def verify_family_selection(raw, root, groups):
    specification = read(root / "data/protocol.json")["taxa"]
    path = root / specification["path"]
    check_hashes({str(path): specification["sha256"]})
    taxa = pd.read_csv(path, keep_default_na=False)
    np.testing.assert_array_equal(taxa.taxon_index, np.arange(raw.shape[1]))
    normalized = np.asarray(raw, dtype=np.float64)
    normalized = normalized / normalized.sum(axis=1)[:, None]
    families = []
    for lineage, frame in taxa.groupby(["kingdom", "phylum", "class", "order", "family"], dropna=False):
        if pd.isna(lineage[-1]) or str(lineage[-1]) == "Unclassified":
            continue
        indices = frame.taxon_index.to_numpy(int)
        mass = normalized[:, indices].sum(axis=1)
        families.append((float(mass.mean()), tuple(map(str, lineage)), indices, float((mass > 0).mean())))
    selected = sorted(families, key=lambda row: (-row[0], row[1]))[:12]
    for group, (mean, lineage, indices, prevalence) in zip(groups, selected):
        assert group["lineage"] == list(lineage) and group["indices"] == indices.tolist()
        close(group["train_mean_abundance"], mean)
        close(group["train_prevalence"], prevalence)


def independent_leaves(tree, values):
    """Replay sklearn's float32 traversal directly against frozen thresholds."""
    x = np.asarray(values, dtype=np.float32)
    nodes = np.zeros(len(x), dtype=int)
    for i, row in enumerate(x):
        node = 0
        while tree.tree_.children_left[node] != tree.tree_.children_right[node]:
            node = (tree.tree_.children_left[node] if row[tree.tree_.feature[node]] <= tree.tree_.threshold[node]
                    else tree.tree_.children_right[node])
        nodes[i] = node
    return nodes


def independent_rules(tree, names):
    result = {}
    pending = [(0, [])]
    while pending:
        node, path = pending.pop()
        left, right = tree.tree_.children_left[node], tree.tree_.children_right[node]
        if left == right:
            result[int(node)] = " AND ".join(path) if path else "All observations"
        else:
            feature, threshold = names[tree.tree_.feature[node]], tree.tree_.threshold[node]
            pending.extend([(int(left), path + [f"{feature} <= {threshold:.6g}"]),
                            (int(right), path + [f"{feature} > {threshold:.6g}"])])
    return result


def verify_selection_source(path, freeze):
    """Static inspection supplements hash gates; it is not a runtime access log."""
    source = ast.parse(path.read_text())
    prepare = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "prepare")
    roles = []
    for node in ast.walk(prepare):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "array":
            assert isinstance(node.args[1], ast.Constant)
            roles.append(node.args[1].value)
    assert set(roles) == {"train", "stopping"}
    assert not freeze["assessment_compositions_or_scores_used_for_selection"]
    for name in freeze["upstream"]:
        lower = str(name).lower()
        assert not lower.endswith("predictions.npz")
        assert not (lower.endswith(".npy") and ("test_p" in lower or "test_q" in lower))
    return "Prepare array calls are limited to train/stopping; frozen upstream excludes assessment arrays and predictions. Cohort previously examined; not an access-log proof."


def verify_communities(folder, roots):
    freeze = read(folder / "FREEZE.json")
    complete = read(folder / "ASSESSMENT_COMPLETE.json")
    assert freeze["status"] == "frozen" and complete["status"] == "complete"
    assert tuple(freeze["methods"]) == METHODS and freeze["feature_count"] == 105
    assert freeze["real_train_n"] == 9162 and freeze["real_stopping_n"] == 1281
    check_hashes(freeze["upstream"])
    check_hashes(freeze["files_sha256"], folder)
    check_hashes(complete["inputs_sha256"])
    check_hashes(complete["files_sha256"], folder)
    assert sha(folder / "FREEZE.json") == complete["freeze_sha256"]
    assert datetime.fromisoformat(freeze["created_utc"]) < datetime.fromisoformat(complete["created_utc"])
    source = next(Path(name) for name in freeze["upstream"] if Path(name).name == "four_generator_communities.py")
    source_audit = verify_selection_source(source, freeze)
    groups = read(folder / "groups.json")
    assert len(groups) == 12 and groups == freeze["groups"]
    development = dict(np.load(folder / "development.npz", allow_pickle=False))
    private = dict(np.load(folder / "assessment_private.npz", allow_pickle=False))
    reference = roots[METHODS[0]]
    raw = {role: load_raw(reference, role, "p") for role in ("train", "stopping", "test")}
    subject = {role: identities(reference, role) for role in raw}
    for a, b in itertools.combinations(subject.values(), 2):
        assert not set(a) & set(b)
    np.testing.assert_array_equal(subject["train"], development["train_subjects"])
    np.testing.assert_array_equal(subject["stopping"], development["stop_subjects"])
    np.testing.assert_array_equal(subject["test"], private["subjects"])
    verify_family_selection(raw["train"], reference, groups)
    heldout_x, names = features(raw["stopping"], groups)
    assert names == freeze["feature_names"]
    assert freeze["tree_subject_split_seed"] == 2026101302 and freeze["tree_training_subject_fraction"] == .7
    subject_order = np.random.default_rng(2026101302).permutation(np.unique(subject["stopping"]))
    fitting_subjects = subject_order[:int(.7 * len(subject_order))]
    fitting_mask = np.isin(subject["stopping"], fitting_subjects)
    fitting_indices, validation_indices = np.flatnonzero(fitting_mask), np.flatnonzero(~fitting_mask)
    np.testing.assert_array_equal(development["discovery_indices"], fitting_indices)
    np.testing.assert_array_equal(development["validation_indices"], validation_indices)
    np.testing.assert_array_equal(development["discovery_subjects"], subject["stopping"][fitting_mask])
    np.testing.assert_array_equal(development["validation_subjects"], subject["stopping"][~fitting_mask])
    assert not set(development["discovery_subjects"]) & set(development["validation_subjects"])
    assert sorted(np.r_[fitting_indices, validation_indices]) == list(range(1281))
    train_x, stop_x = heldout_x[fitting_mask], heldout_x[~fitting_mask]
    assert len(train_x) == freeze["tree_training_n"] and len(stop_x) == freeze["tree_validation_n"]
    close(train_x, development["train_x"], atol=0, rtol=0)
    close(stop_x, development["stop_x"], atol=0, rtol=0)
    train_y, stop_y = development["train_y"], development["stop_y"]
    assert train_y.shape == (len(train_x), 4) and stop_y.shape == (len(stop_x), 4)
    candidates, trees = [], []
    for depth in (2, 3, 4):
        candidate = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=.05, random_state=2026101301)
        candidate.fit(train_x, train_y)
        mse = ((candidate.predict(stop_x) - stop_y) ** 2).mean(axis=0)
        record = next(row for row in freeze["candidates"] if row["depth"] == depth)
        close(mse, [record["mse_by_method"][m] for m in METHODS])
        close(mse.mean(), record["stopping_mse"])
        assert record["leaves"] == candidate.get_n_leaves()
        candidates.append((float(mse.mean()), depth))
        trees.append(candidate)
    selected = min(range(3), key=lambda i: candidates[i])
    assert freeze["selected"]["depth"] == candidates[selected][1]
    tree = joblib.load(folder / "tree.joblib")
    for attribute in ("children_left", "children_right", "feature", "threshold", "value", "n_node_samples"):
        np.testing.assert_array_equal(getattr(tree.tree_, attribute), getattr(trees[selected].tree_, attribute))
    rules = read(folder / "rules.json")
    calculated_rules = independent_rules(tree, names)
    assert set(calculated_rules) == {row["leaf"] for row in rules.values()}
    for index, (community, specification) in enumerate(sorted(rules.items())):
        assert community == f"C{index+1:02d}"
        assert specification["leaf"] == sorted(calculated_rules)[index]
        assert specification["rule"] == calculated_rules[specification["leaf"]]
    p_features, _ = features(raw["test"], groups)
    p_labels = independent_leaves(tree, p_features)
    np.testing.assert_array_equal(p_labels, private["p_labels"])
    rows = pd.read_csv(folder / "summary/shared_communities.csv")
    fidelity = pd.read_csv(folder / "summary/tree_fidelity.csv").set_index("method")
    replay = pd.read_csv(folder / "summary/score_replay.csv").set_index(["method", "repeat"])
    assert len(rows) == len(rules) * 4 and len(replay) == 20
    max_difference = 0.0
    means = []
    for column, method in enumerate(METHODS):
        root = roots[method]
        for role in raw:
            np.testing.assert_array_equal(raw[role], load_raw(root, role, "p"))
            np.testing.assert_array_equal(subject[role], identities(root, role))
        q = load_raw(root, "test", "q")
        assert raw["test"].shape == (2002, 614) and q.shape == (2089, 614)
        q_features, _ = features(q, groups)
        q_labels = independent_leaves(tree, q_features)
        np.testing.assert_array_equal(q_labels, private[method + "_q_labels"])
        p_scores, q_scores = private[method + "_p_scores"], private[method + "_q_scores"]
        assert p_scores.shape == (2002, 5) and q_scores.shape == (2089, 5)
        for repeat in range(5):
            final = root / "final" / f"{repeat:02d}"
            seal = read(root / "COMPLETE.json")
            check_hashes({str(final / "COMPLETE.json"): seal["assessment_completions"][f"final/{repeat:02d}/COMPLETE.json"]})
            receipt = read(final / "COMPLETE.json")
            check_hashes({str(final / "predictions.npz"): receipt["outputs"]["predictions.npz"]})
            with np.load(final / "predictions.npz", allow_pickle=False) as stored:
                for scores, key in ((p_scores, "test_p"), (q_scores, "test_q")):
                    difference = float(np.max(np.abs(scores[:, repeat] - stored[key])))
                    max_difference = max(max_difference, difference)
                    close(scores[:, repeat], stored[key], atol=1e-12, rtol=0)
            close(replay.loc[(method, repeat), "max_abs_difference"], 0, atol=1e-12, rtol=0)
        for community, specification in rules.items():
            selected_rows = rows[(rows.method == method) & (rows.community == community)]
            assert len(selected_rows) == 1
            row = selected_rows.iloc[0]
            assert row.rule == specification["rule"] and row.leaf == specification["leaf"]
            pm, qm = p_labels == specification["leaf"], q_labels == specification["leaf"]
            pp, qq = pm.sum() / len(pm), qm.sum() / len(qm)
            assert pp + qq > 0
            # Balance the sources globally, then condition these weights on A.
            weights = np.r_[np.full(pm.sum(), 1 / len(pm)), np.full(qm.sum(), 1 / len(qm))]
            samples = np.vstack((p_scores[pm], q_scores[qm]))
            region_means = np.average(samples, axis=0, weights=weights)
            theta = 2 * pp / (pp + qq)
            expected = dict(n_p=len(pm), n_q=len(qm), p_count=pm.sum(), q_count=qm.sum(), p_fraction=pp,
                            q_fraction=qq, q_minus_p=qq-pp, balanced_mass=(pp+qq)/2,
                            empirical_region_rdr=theta, mean_fitted_rdr=region_means.mean(),
                            fitted_rdr_seed_sd=region_means.std(ddof=1),
                            empirical_abs_gap=abs(region_means.mean()-theta),
                            mean_real_rdr=p_scores[pm].mean() if pm.any() else np.nan,
                            mean_generated_rdr=q_scores[qm].mean() if qm.any() else np.nan)
            for name, value in expected.items():
                close(row[name], value)
            assert bool(row.small_count) == (min(pm.sum(), qm.sum()) < 30)
        selected_rows = rows[rows.method == method]
        close(selected_rows[["p_count", "q_count", "p_fraction", "q_fraction", "balanced_mass"]].sum(), [2002, 2089, 1, 1, 1])
        target = p_scores.mean(axis=1)
        prediction = tree.tree_.value[p_labels, column, 0]
        mse = np.mean((prediction - target) ** 2)
        close(fidelity.loc[method, "real_assessment_mse"], mse)
        close(fidelity.loc[method, "real_assessment_r2"], 1 - mse / np.var(target))
        means.append(target)
    means = np.column_stack(means)
    np.testing.assert_array_equal(means, private["real_scores"])
    paired = pd.read_csv(folder / "summary/paired_real_scores.csv").set_index(["method_a", "method_b"])
    assert len(paired) == 6
    for i, j in itertools.combinations(range(4), 2):
        a, b = means[:, i], means[:, j]
        row = paired.loc[(METHODS[i], METHODS[j])]
        close(row.spearman, np.corrcoef(rankdata(a), rankdata(b))[0, 1])
        close(row.mean_absolute_score_difference, np.abs(a-b).mean())
        close(row.opposite_fitted_direction_fraction, np.mean((a-1)*(b-1) < 0))
    return dict(status="passed", methods=4, communities=len(rules), community_rows=len(rows),
                reconstructed_feature_count=len(names), tree_candidates_replayed=3,
                selected_depth=freeze["selected"]["depth"], assessment_score_replays=20,
                maximum_upstream_score_difference=max_difference, common_real_rows_per_role={k: len(v) for k, v in raw.items()},
                tree_source="Original RDR stopping P, used previously for checkpoint selection, split by subjects with seed 2026101302",
                tree_subject_split_counts={"fit_rows": len(train_x), "validation_rows": len(stop_x),
                                           "fit_subjects": len(set(development["discovery_subjects"])),
                                           "validation_subjects": len(set(development["validation_subjects"]))},
                source_normalized_region_means="Independently weighted each observation by inverse source size before conditioning on its common leaf.",
                selection_source_audit=source_audit,
                development_target_scope="Replayed tree candidates on frozen development response arrays; did not independently rerun neural development predictions.",
                assessment_complete_sha256=sha(folder / "ASSESSMENT_COMPLETE.json"))


def verify_metrics(folder):
    freeze, protocol = read(folder / "FREEZE.json"), read(folder / "protocol.json")
    complete, previous = read(folder / "summary/COMPLETE.json"), read(folder / "verification.json")
    assert freeze["status"] == "frozen" and freeze["n_models"] == 40 and freeze["no_test_arrays_read"]
    assert complete["status"] == "complete" and previous["status"] == "passed"
    assert tuple(protocol["methods"]) == METHODS and complete["no_numeric_inception_score"]
    assert previous["completion_sha256"] == sha(folder / "summary/COMPLETE.json")
    assert sha(folder / "FREEZE.json") == complete["freeze_sha256"]
    assert sha(folder / "protocol.json") == freeze["protocol_sha256"]
    assert sha(folder / "ASSESSMENT_RELEASE.json") == complete["release_sha256"]
    check_hashes(freeze["files"], folder)
    check_hashes(freeze["training_inputs"])
    check_hashes(protocol["sources"])
    check_hashes(protocol["metadata_inputs"])
    check_hashes(complete["assessment_inputs"])
    check_hashes(complete["published_reference_inputs"])
    check_hashes(complete["files"], folder / "summary")
    frame = pd.read_csv(folder / "summary/global_metrics.csv")
    hist = pd.read_csv(folder / "summary/prd_histograms.csv")
    curves = pd.read_csv(folder / "summary/prd_curves.csv")
    assert len(frame) == 40 and len(curves) == 8040 and len(hist) == 2560
    assert not frame.duplicated(["method", "representation", "seed"]).any()
    slopes = np.tan(np.linspace(1e-10, np.pi/2-1e-10, 201))
    maximum = 0.0
    for row in frame.itertuples(index=False):
        mask = (hist.method == row.method) & (hist.representation == row.representation) & (hist.seed == row.seed)
        h = hist[mask].sort_values("cluster")
        assert h.cluster.tolist() == list(range(64)) and h.count_p.sum() == 2002 and h.count_q.sum() == 2089
        assert (h[["count_p", "count_q"]].to_numpy() >= 0).all()
        p, q = h.count_p.to_numpy()/2002, h.count_q.to_numpy()/2089
        close([p.sum(), q.sum()], [1, 1])
        precision = np.array([np.minimum(scale*p, q).sum() for scale in slopes])
        recall = np.array([np.minimum(p, q/scale).sum() for scale in slopes])
        c = curves[(curves.method == row.method) & (curves.representation == row.representation) & (curves.seed == row.seed)].sort_values("angle")
        assert c.angle.tolist() == list(range(201))
        close(c.precision, precision)
        close(c.recall, recall)
        maximum = max(maximum, float(np.abs(c.precision.to_numpy()-precision).max()), float(np.abs(c.recall.to_numpy()-recall).max()))
        f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(201), where=precision+recall > 0)
        close(row.prd_max_f1, f1.max())
    aggregate = pd.read_csv(folder / "summary/global_metrics_aggregate.csv").set_index(["method", "representation"])
    for key, subset in frame.groupby(["method", "representation"]):
        row = aggregate.loc[key]
        assert len(subset) == row.seeds == 5
        for metric in ("frechet", "prd_max_f1"):
            close(row[metric+"_mean"], subset[metric].mean())
            close(row[metric+"_sd"], subset[metric].std(ddof=1))
    curve_summary = pd.read_csv(folder / "summary/prd_curve_summary.csv").set_index(["method", "representation", "angle"])
    assert len(curve_summary) == 1608
    for key, subset in curves.groupby(["method", "representation", "angle"]):
        assert len(subset) == 5
        row = curve_summary.loc[key]
        for metric in ("precision", "recall"):
            close(row[metric+"_mean"], subset[metric].mean())
            close(row[metric+"_sd"], subset[metric].std(ddof=1))
    reference = pd.read_csv(folder / "summary/fd_reference_check.csv")
    assert len(reference) == 4 and reference.passed.all()
    published = pd.read_csv(Path(protocol["published_reference_root"]) / "global/global_metrics.csv")
    for row in reference.itertuples(index=False):
        historical = "scope" in published.columns
        method = "zero_aware" if historical and row.method == "midasim_mask_icfm" else row.method
        selected = (published.method == method) & (published.representation == row.representation)
        if historical:
            selected &= published.scope == "full"
        old = published[selected]
        assert len(old) == 5
        close(row.published_frechet, old.frechet.iloc[0])
        close(row.frechet, row.published_frechet, atol=1e-8, rtol=1e-9)
    return dict(status="passed", global_rows=40, prd_curves=40, prd_points=8040, histogram_cells=2560,
                maximum_prd_difference=maximum, published_frechet_checks=4,
                independent_fd_audit=previous, independent_fd_audit_sha256=sha(folder / "verification.json")), {k: Path(v) for k, v in protocol["roots"].items()}


def verify_public(folder):
    manifest = read(folder / "manifest.json")
    assert manifest.get("status") == "complete"
    mapping = manifest.get("files_sha256", manifest.get("files"))
    if not isinstance(mapping, dict):
        raise ValueError("Public manifest needs a relative files_sha256 map")
    check_hashes(mapping, folder)
    actual_files = {str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file()}
    assert actual_files == set(mapping) | {"manifest.json"}
    assert tuple(manifest["retained_methods"]) == METHODS
    if "inputs_sha256" in manifest:
        check_hashes(manifest["inputs_sha256"])
    opening = manifest.get("opening_files_copied_unchanged", {})
    source = Path(__file__).resolve().parents[2] / "results/AGP/retained_mask_comparison_20261013"
    assert {"comparison.csv", "rdr_comparison.png", "rdr_comparison.pdf"} <= set(opening)
    for name, expected in opening.items():
        check_hashes({str(source / name): expected, str(folder / name): expected})
    aggregates = manifest.get("aggregate_source", {})
    assert len(aggregates) == 11
    for original, destination in aggregates.items():
        assert destination in mapping
        assert sha(original) == sha(folder / destination)
    allowed_csv = {name for name in opening if name.endswith(".csv")} | set(aggregates.values())
    for file in folder.rglob("*"):
        if not file.is_file():
            continue
        assert file.suffix.lower() not in {".npy", ".npz", ".pt", ".pth", ".joblib", ".pkl"}
        if file.suffix == ".csv":
            assert str(file.relative_to(folder)) in allowed_csv
            frame = pd.read_csv(file)
            assert not set(frame.columns) & {"subject_id", "sample_id", "array_row"}
            assert not frame.astype(str).apply(lambda col: col.str.contains("bootstrap_icfm|Resampled mask", regex=True)).any().any()
    return dict(status="passed", files_hashed=len(mapping), unchanged_opening_files=len(opening),
                unchanged_aggregate_tables=len(aggregates),
                private_array_and_id_columns="absent", manifest_sha256=sha(folder / "manifest.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/tmp/agp-four-generator-20261013"))
    parser.add_argument("--public", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "verification.json"
    if output.exists():
        raise FileExistsError("Audit output exists; select a fresh output")
    if not (args.root / "communities/ASSESSMENT_COMPLETE.json").is_file():
        raise ValueError("Community assessment must complete before auditing")
    with threadpool_limits(limits=2):
        metrics, roots = verify_metrics(args.root / "metrics")
        communities = verify_communities(args.root / "communities", roots)
    result = dict(status="passed", verified_utc=datetime.now(timezone.utc).isoformat(),
                  verifier_sha256=sha(__file__), metrics=metrics, communities=communities,
                  scope="Retrospective arithmetic/provenance audit; no pointwise RDR accuracy, interval coverage, generator-refit uncertainty or causal attribution claim.")
    if args.public is not None:
        result["public_export"] = verify_public(args.public)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": "passed", "output": str(output), "community_rows": communities["community_rows"], "prd_points": metrics["prd_points"]}), flush=True)


if __name__ == "__main__":
    main()
