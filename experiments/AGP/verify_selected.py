#!/usr/bin/env python3
"""Independently audit a completed AGP selected-model archive without refitting.

Only the requested verification receipt is written (a fresh /tmp file by
default). Model execution uses the archive's frozen source, never current
workflow code. All array access follows completion and selection-lock checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from scipy.stats import beta, norm
import torch
from threadpoolctl import threadpool_limits


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, atol=2e-11, rtol=2e-9):
    a = np.asarray(actual, dtype=float)
    b = np.asarray(expected, dtype=float)
    require(a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True),
            f"Numeric mismatch: {label}")


def _frozen_runner(output):
    source = output / "source/experiments/AGP/selected_workflow.py"
    require(source.is_file(), "Missing frozen workflow source")
    spec = importlib.util.spec_from_file_location("_agp_frozen_verification_workflow", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(Path(module.__file__).resolve() == source.resolve(), "Workflow was not loaded from the frozen archive")
    return module


def _receipts(output, receipts):
    for receipt in receipts:
        path = output / receipt["path"]
        require(sha(path) == receipt["sha256"], f"Data receipt mismatch: {path}")


def _brier(p, q):
    return float(.5*np.square(p/2-1).mean() + .5*np.square(q/2).mean())


def _cp(k, n, tail):
    # Deliberately use direct beta quantiles rather than the shared CP helper.
    lower = 0. if k == 0 else float(beta.ppf(tail, k, n-k+1))
    upper = 1. if k == n else float(beta.ppf(1-tail, k+1, n-k))
    return lower, upper


def _diagnostics(cp, cq, ep, eq, alpha, bins):
    """Independent balanced cell estimates, gaps, C.1 and direct-quantile C.2."""
    edges = np.linspace(0., 2., bins+1)
    for values in (cp, cq, ep, eq):
        require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all()
                and ((values >= 0) & (values <= 2)).all(), "Invalid archived prediction array")
    counts = [np.histogram(values, edges)[0] for values in (cp, cq, ep, eq)]
    kp, kq, ne_p, ne_q = counts
    p, q = kp/len(cp), kq/len(cq)
    mass = .5*(ne_p/len(ep) + ne_q/len(eq))
    ps = np.histogram(ep, edges, weights=ep)[0] / len(ep)
    qs = np.histogram(eq, edges, weights=eq)[0] / len(eq)
    means = np.divide(.5*(ps+qs), mass, out=np.full(bins, np.nan), where=mass > 0)
    theta = np.divide(2*p, p+q, out=np.full(bins, np.nan), where=p+q > 0)
    rows = []
    for j in range(bins):
        total = p[j]+q[j]
        variance = (4*q[j]**2*p[j]*(1-p[j])/len(cp) +
                    4*p[j]**2*q[j]*(1-q[j])/len(cq))/total**4 if total else 0.
        unavailable = total == 0 or variance == 0
        radius = norm.ppf(1-alpha/2)*np.sqrt(variance)
        c1 = (0., 2.) if unavailable else (max(0., theta[j]-radius), min(2., theta[j]+radius))
        pl, pu = _cp(int(kp[j]), len(cp), alpha/(4*bins))
        ql, qu = _cp(int(kq[j]), len(cq), alpha/(4*bins))
        lo, hi = 2*pl/(pl+qu), 2*pu/(pu+ql)
        gap = means[j]-theta[j]
        rows.append(dict(bin=j, left=edges[j], right=edges[j+1],
            cal_count_p=int(kp[j]), cal_count_q=int(kq[j]), calibration_mass=total/2,
            eval_count_p=int(ne_p[j]), eval_count_q=int(ne_q[j]), eval_mass=mass[j],
            neural_mean=means[j], calibrated_rdr=theta[j], gap=gap, absolute_gap=abs(gap),
            c1_lower=c1[0], c1_upper=c1[1], c1_unavailable=unavailable,
            c2_lower=lo, c2_upper=hi, c2_width=hi-lo,
            c2_distance=max(lo-means[j], means[j]-hi, 0.) if mass[j] else np.nan,
            calibration_empty=total == 0))
    cells = pd.DataFrame(rows)
    supported = cells.eval_mass.gt(0) & cells.calibrated_rdr.notna()
    support = float(cells.loc[supported, "eval_mass"].sum())
    summary = dict(brier=_brier(ep, eq), supported_mass=support,
                   unsupported_mass=float(cells.loc[~supported, "eval_mass"].sum()),
                   local_gap=float((cells.loc[supported, "eval_mass"] *
                                    cells.loc[supported, "absolute_gap"]).sum()/support) if support else None,
                   signed_gap=float((cells.loc[supported, "eval_mass"] *
                                     cells.loc[supported, "gap"]).sum()/support) if support else None,
                   c2_width=float((cells.eval_mass*cells.c2_width).sum()))
    for prefix, left, right in (("middle", .7, 1.2), ("agp_middle", .2, 1.8)):
        mask = cells.left.ge(left-1e-12) & cells.right.le(right+1e-12)
        selected = mask & supported
        sm = float(cells.loc[selected, "eval_mass"].sum())
        summary[prefix+"_mass"] = float(cells.loc[mask, "eval_mass"].sum())
        summary[prefix+"_supported_mass"] = sm
        summary[prefix+"_local_gap"] = float((cells.loc[selected, "eval_mass"] *
                                                cells.loc[selected, "absolute_gap"]).sum()/sm) if sm else None
    return summary, cells


def _array_audit(output, protocol):
    """Validate whole-source preservation, disjoint subjects, and row-level bytes."""
    original = pd.read_csv(output/protocol["original_manifest"]["path"], dtype=str, keep_default_na=False)
    require(not original.sample_id.duplicated().any(), "Original sample IDs are duplicated")
    retained = original.loc[~original.role.eq("excluded")]
    require(int(original.role.eq("excluded").sum()) == protocol["original_excluded_real_test_rows"],
            "Original exclusion count changed")
    manifests, arrays, exact_counts = {}, {}, {}
    source_fields = ["sample_id", "subject_id", "source_file", "source_row", "row_sha256"]
    for study, spec in protocol["studies"].items():
        frame = pd.read_csv(output/spec["manifest"]["path"], dtype=str, keep_default_na=False)
        manifests[study] = frame
        require(not frame.sample_id.duplicated().any(), f"{study}: duplicated samples")
        require(not frame.duplicated(["source_file", "source_row"]).any(), f"{study}: duplicated source rows")
        real = frame.loc[frame.original_distribution.eq("P")]
        assignments = real.role+":"+real.distribution
        require(assignments.groupby(real.subject_id).nunique().eq(1).all(), f"{study}: subject leakage")
        require(real.subject_id.ne("").all(), f"{study}: missing real subject")
        matching = original.iloc[frame.original_manifest_row.astype(int).to_numpy()].reset_index(drop=True)
        require(frame[source_fields].reset_index(drop=True).equals(matching[source_fields]), f"{study}: original identity mismatch")
        require(np.array_equal(frame.original_role, matching.role), f"{study}: original roles changed")
        require(np.array_equal(frame.original_distribution, matching.distribution), f"{study}: original sources changed")
        expected = retained
        if study == "selection":
            expected = expected.loc[expected.role.isin(["train", "validation"])]
            require(not frame.original_role.eq("test").any(), "Selection contains historical test data")
        elif study == "null":
            expected = expected.loc[expected.distribution.eq("P")]
            require(frame.original_distribution.eq("P").all(), "Null includes generated observations")
        require(set(frame.sample_id) == set(expected.sample_id), f"{study}: source population changed")
        if study != "selection":
            require(set(frame.loc[frame.role.eq("test"), "sample_id"]) ==
                    set(expected.loc[expected.role.eq("test"), "sample_id"]), f"{study}: original test cohort changed")
        arrays[study], exact_counts[study] = {}, {}
        for key, receipt in spec["arrays"].items():
            role, side = key.rsplit("_", 1)
            group = frame.loc[frame.role.eq(role) & frame.distribution.eq(side.upper())]
            require(group.array_row.astype(int).tolist() == list(range(len(group))), f"{study}/{key}: row order changed")
            values = np.load(output/receipt["path"], allow_pickle=False, mmap_mode="r")
            require(values.dtype == np.float32 and list(values.shape) == receipt["shape"], f"{study}/{key}: array shape/dtype changed")
            require(len(values) == len(group) == spec["counts"][key]["rows"], f"{study}/{key}: count mismatch")
            require(len(group) == spec["role_counts"][role][side], f"{study}/{key}: role count mismatch")
            row_hashes = [hashlib.sha256(row.astype("<f4").tobytes()).hexdigest() for row in values]
            require(row_hashes == group.row_sha256.tolist(), f"{study}/{key}: raw rows changed")
            if group.original_distribution.eq("P").all():
                require(group.subject_id.nunique() == spec["counts"][key]["subjects"], f"{study}/{key}: subject count mismatch")
            arrays[study][key] = values
            exact_counts[study][key] = dict(rows=len(group), subjects=spec["counts"][key]["subjects"])
    return manifests, arrays, exact_counts


def _expected_transform(raw, epsilon):
    transformed = []
    for key in ("train_p", "train_q"):
        x = np.asarray(raw[key], dtype=float)
        x = x/x.sum(axis=1, keepdims=True)
        logarithm = np.log((x+epsilon)/(1+x.shape[1]*epsilon))
        transformed.append(logarithm-logarithm.mean(axis=1, keepdims=True))
    p, q = transformed
    center = .5*(p.mean(0)+q.mean(0))
    scale = np.sqrt(.5*(np.square(p-center).mean()+np.square(q-center).mean()))
    return dict(center=center, scale=scale, epsilon=epsilon, basis=helmert(p.shape[1], full=False))


def _independent_choice(rows):
    candidates = sorted({row["candidate"] for row in rows})
    grouped = {candidate: sorted([row for row in rows if row["candidate"] == candidate],
                                key=lambda r: r["repeat"]) for candidate in candidates}
    briers = {candidate: np.array([r["brier"] for r in group]) for candidate, group in grouped.items()}
    reference = min(candidates, key=lambda c: (briers[c].mean(), c))
    eligible = []
    for candidate, group in grouped.items():
        delta = briers[candidate]-briers[reference]
        threshold = delta.std(ddof=1)/np.sqrt(len(delta))
        if delta.mean() <= threshold+1e-12 and min(r["supported_mass"] for r in group) >= .99:
            eligible.append(candidate)
    require(bool(eligible), "No independently eligible configuration")
    chosen = min(eligible, key=lambda c: (np.mean([r["local_gap"] for r in grouped[c]]), briers[c].mean(), c))
    return reference, chosen, eligible


def verify(output):
    output = Path(output).resolve()
    started = time.monotonic()
    root_completion = read(output/"COMPLETE.json")
    require(root_completion.get("status") == "complete", "Root completion marker is missing or incomplete")
    runner = _frozen_runner(output)
    protocol = runner.verify_snapshot(output)
    config = runner.validate_config(read(output/"config.json"))
    data_protocol = read(output/"data/protocol.json")
    chosen = runner.checked_selection(output)
    tasks = runner.selection_tasks(config)
    expected_grid = {(loss, float(alpha), repeat) for loss in config["losses"]
                     for alpha in config["output_alphas"] for repeat in range(config["repeats"])}
    require({(t["loss"], t["output_alpha"], t["repeat"]) for t in tasks} == expected_grid,
            "Selection task grid changed")
    require(len(tasks) == len(expected_grid) == root_completion["selection_fits"] == protocol["expected_selection_fits"],
            "Selection fit count mismatch")
    require(root_completion["final_fits"] == root_completion["null_fits"] == config["repeats"], "Final fit count mismatch")
    for relative, digest in root_completion["outputs"].items():
        require(sha(output/relative) == digest, f"Root completion artifact changed: {relative}")
    work = [("selection", task, output/"selection/runs"/task["candidate"]/f"{task['repeat']:02d}") for task in tasks]
    work += [(stage, {**chosen, "repeat": repeat}, output/stage/f"{repeat:02d}")
             for stage in ("final", "null") for repeat in range(config["repeats"])]
    for stage, task, directory in work:
        runner.verify_expected_task(output, directory, stage, task)
        require(not (directory/"FAILED.json").exists(), f"Failure marker remains: {directory}")
    expected_markers = {directory/"COMPLETE.json" for _, _, directory in work}
    actual_markers = set(output.glob("selection/runs/*/*/COMPLETE.json"))
    actual_markers |= set(output.glob("final/*/COMPLETE.json")) | set(output.glob("null/*/COMPLETE.json"))
    require(actual_markers == expected_markers, "Unexpected or absent task completion markers")
    receipts = [data_protocol[k] for k in ("original_manifest", "original_audit", "taxa")]
    for spec in data_protocol["studies"].values():
        receipts += [spec["manifest"], spec["audit"], *spec["arrays"].values()]
    _receipts(output, receipts)
    for relative, digest in data_protocol["input_sha256"].items():
        require(sha(Path(data_protocol["data_root"])/relative) == digest, f"Original AGP input changed: {relative}")
    manifests, raw, counts = _array_audit(output, data_protocol)
    expected_transforms = {stage: _expected_transform(values, config["epsilon"]) for stage, values in raw.items()}
    verified, selection_rows = [], []
    max_prediction_difference, max_stopping_difference = 0., 0.
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        for index, (stage, task, directory) in enumerate(work):
            metrics = read(directory/"metrics.json")
            require(all(metrics.get(key) == value for key, value in task.items()), f"{directory}: task metric identity mismatch")
            seed = int(config["seed"])+{"selection": 0, "final": 100000, "null": 200000}[stage]+1009*task["repeat"]
            require(metrics["seed_initialization"] == seed, f"{directory}: initialization seed mismatch")
            with np.load(directory/"predictions.npz", allow_pickle=False) as saved:
                predictions = {key: saved[key] for key in saved.files}
            roles = ("calibration", "evaluation") if stage == "selection" else ("test",)
            require(set(predictions) == {f"{role}_{side}" for role in roles for side in ("p", "q")}, "Prediction roles changed")
            scores = pd.read_csv(directory/"scores.csv", dtype=str, keep_default_na=False)
            for role in roles:
                for side in ("p", "q"):
                    key = f"{role}_{side}"
                    group = scores.loc[scores.role.eq(role) & scores.distribution.eq(side.upper())]
                    expected = manifests[stage].loc[manifests[stage].role.eq(role) &
                                                   manifests[stage].distribution.eq(side.upper())].sort_values("array_row", key=lambda x: x.astype(int))
                    require(set(scores.columns) == set(expected.columns) | {"rdr"}, "Score manifest columns changed")
                    require(group[expected.columns].reset_index(drop=True).equals(expected.reset_index(drop=True)), "Score identities or order changed")
                    close(group.rdr.astype(float).to_numpy(), predictions[key], f"{directory}/{key}: CSV predictions", atol=1e-14, rtol=1e-13)
                    require(len(predictions[key]) == len(raw[stage][key]), "Prediction observation count mismatch")
            eval_role = "evaluation" if stage == "selection" else "test"
            cal_role = "calibration" if stage == "selection" else "test"
            alpha = config["ci_alpha"]/(len(config["losses"])*len(config["output_alphas"])) if stage == "selection" else config["ci_alpha"]
            diagnostic, cells = _diagnostics(predictions[cal_role+"_p"], predictions[cal_role+"_q"],
                                             predictions[eval_role+"_p"], predictions[eval_role+"_q"], alpha, config["bins"])
            for key, value in diagnostic.items():
                require(key in metrics, f"Missing diagnostic metric {key}")
                close(metrics[key], value, f"{directory}: independent {key}")
            saved_cells = pd.read_csv(directory/"cells.csv", float_precision="round_trip").sort_values("bin")
            require(len(saved_cells) == config["bins"], "Fixed-bin count changed")
            for column in cells:
                close(saved_cells[column].to_numpy(), cells[column].to_numpy(), f"{directory}: independent {column}", atol=2e-9)
            checkpoint = torch.load(directory/"model.pt", map_location="cpu", weights_only=True)
            require(checkpoint["task"] == task and checkpoint["config"] == config, "Checkpoint configuration/task changed")
            history = pd.read_csv(directory/"history.csv", float_precision="round_trip")
            require(history.epoch.tolist() == list(range(1, len(history)+1)), "History epochs are not complete and sequential")
            best = history.loc[history.stopping_brier.idxmin()]
            require(metrics["best_epoch"] == checkpoint["training"]["best_epoch"] == int(best.epoch), "Checkpoint is not first absolute Brier minimum")
            require(metrics["epochs_run"] == len(history), "History length differs from fitted metadata")
            close(metrics["stopping_brier"], best.stopping_brier, "History minimum Brier")
            close(checkpoint["training"]["restored_stopping_brier"], best.stopping_brier, "Checkpoint restored Brier metadata")
            with np.load(directory/"transform.npz", allow_pickle=False) as saved:
                state = {key: saved[key] for key in saved.files}
            for key, value in expected_transforms[stage].items():
                close(state[key], value, f"{directory}: training-only ILR {key}", atol=2e-12, rtol=2e-12)
            model = runner.make_model(data_protocol["feature_dimension"], config, task["repeat"], stage)
            model.load_state_dict(checkpoint["model"], strict=True)
            stop_scores = [runner.predict(model, raw[stage]["stopping_"+side], state, task["output_alpha"], "cpu") for side in ("p", "q")]
            stopped = _brier(*stop_scores)
            max_stopping_difference = max(max_stopping_difference, abs(stopped-float(best.stopping_brier)))
            close(stopped, best.stopping_brier, f"{directory}: rescored stopping Brier", atol=2e-8, rtol=2e-6)
            test_difference = None
            if stage != "selection":
                test_difference = 0.
                for side in ("p", "q"):
                    expected = predictions["test_"+side]
                    rescored = runner.predict(model, raw[stage]["test_"+side], state, task["output_alpha"], "cpu")
                    close(rescored, expected, f"{directory}: rescored full test {side}", atol=3e-7, rtol=2e-6)
                    test_difference = max(test_difference, float(np.max(np.abs(rescored-expected))))
                max_prediction_difference = max(max_prediction_difference, test_difference)
            else:
                selection_rows.append({**task, **diagnostic})
            if stage == "null":
                null_mse = .5*(np.square(predictions["test_p"]-1).mean()+np.square(predictions["test_q"]-1).mean())
                close(metrics["true_rdr_mse"], null_mse, "Null true-RDR MSE")
            verified.append(dict(stage=stage, **task, seed=seed, best_epoch=int(best.epoch), epochs=len(history),
                                 stopping_brier=stopped, brier=diagnostic["brier"], local_gap=diagnostic["local_gap"],
                                 maximum_test_rescore_difference=test_difference))
            if (index+1) % 10 == 0 or index+1 == len(work):
                print(f"Verified {index+1}/{len(work)} checkpoints and diagnostics", flush=True)
    reference, independent_chosen, eligible = _independent_choice(selection_rows)
    saved_selection = read(output/"selection/selection.json")
    require(independent_chosen == chosen["candidate"] and reference == saved_selection["brier_reference"], "Independent selection decision mismatch")
    require(set(eligible) == {row["candidate"] for row in saved_selection["ranking"] if row["eligible"]}, "Independent shortlist mismatch")
    return dict(status="passed", output=str(output), checked_source="frozen output/source",
        root_completion_sha256=sha(output/"COMPLETE.json"), protocol_sha256=sha(output/"protocol.json"),
        selection_freeze_sha256=sha(output/"selection/freeze.json"), chosen=chosen,
        brier_reference=reference, independently_eligible_candidates=sorted(eligible), exact_counts=counts,
        selection_fits=len(tasks), final_fits=config["repeats"], null_fits=config["repeats"],
        checkpoint_stopping_rescores=len(work), final_null_test_rescores=2*config["repeats"],
        independently_checked_c2_cells=len(work)*config["bins"], original_input_hashes=len(data_protocol["input_sha256"]),
        maximum_test_prediction_difference=max_prediction_difference,
        maximum_stopping_brier_difference=max_stopping_difference, models=verified,
        elapsed_seconds=time.monotonic()-started,
        interpretation="Numerical and provenance verification only. C.1 is nominal marginal and C.2 nominal per-model simultaneous cell-average inference; no AGP coverage or joint-across-fits guarantee is asserted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verification-file", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc)
    receipt = args.verification_file or Path("/tmp")/f"agp-selected-verification-{timestamp:%Y%m%dT%H%M%S}-{os.getpid()}.json"
    require(not receipt.exists(), f"Refusing to replace an existing verification receipt: {receipt}")
    require(receipt.parent.is_dir(), f"Verification receipt parent does not exist: {receipt.parent}")
    try:
        result = verify(args.output)
        exit_code = 0
    except Exception as error:
        result = dict(status="failed", output=str(args.output.resolve()), error=f"{type(error).__name__}: {error}")
        exit_code = 1
    result.update(verified_utc=datetime.now(timezone.utc).isoformat(),
                  verifier_source=str(Path(__file__).resolve()), verifier_source_sha256=sha(__file__))
    with receipt.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(status=result["status"], verification_file=str(receipt), error=result.get("error"))), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
