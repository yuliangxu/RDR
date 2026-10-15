#!/usr/bin/env python3
"""Independently verify a completed full-template MIDASim selection study.

Reads the outer generator/analysis archive and its frozen source snapshots.
Reconstructs every checkpoint and selection decision without fitting models.
Only a fresh, explicitly requested receipt outside the archive is written.
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
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "2")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
from threadpoolctl import threadpool_limits


ROLE_SIZES = {
    "selection": dict(train_p=7310, train_q=5847, stopping_p=1044, stopping_q=836,
                      calibration_p=1045, calibration_q=835, evaluation_p=1044, evaluation_q=836),
    "final": dict(train_p=9162, train_q=7310, stopping_p=1281, stopping_q=1044, test_p=2002, test_q=2089),
    "null": dict(train_p=4581, train_q=4581, stopping_p=640, stopping_q=641, test_p=1001, test_q=1001),
}
GENERATOR_ROLES = ("selection_train", "selection_stopping", "selection_calibration",
                   "selection_evaluation", "final_train", "final_stopping", "final_test")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, label, atol=2e-10, rtol=2e-9):
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True), label)


def hashes(root, records):
    require(bool(records), f"Empty receipt list: {root}")
    for name, digest in records.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"Invalid receipt path: {name}")
        require(sha(Path(root)/relative) == digest, f"Changed artifact: {Path(root)/relative}")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rows(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def subset(frame, role, side):
    return frame[frame.role.eq(role) & frame.distribution.eq(side.upper())].sort_values(
        "array_row", key=lambda x: x.astype(int)).reset_index(drop=True)


def generator_audit(outer, template, parent, parent_data):
    root = outer/"generator"
    config, metrics = read(root/"config.json"), read(root/"generation/metrics.json")
    require(metrics["status"] == "complete" and metrics["version"] == "2.0", "Wrong generator completion/version")
    require(metrics["mode"] == "nonparametric" and metrics["only_relative_abundance"] is True,
            "Wrong MIDASim generation mode")
    require(not metrics["count_data_generated"] and not metrics["fabricated_library_depths"]
            and metrics["library_size_metadata_removed"], "Sequencing-depth/count path changed")
    require(metrics["seed"] == metrics["setup_seed"] == config["generation_seed"] == 2026100601,
            "Generator setup seed differs")
    require(metrics["generation_seed_offset"] == 100000 and metrics["tie_break_repetitions"] == 100,
            "Generator fitting or seed schedule differs")
    require(tuple(config["role_order"]) == tuple(metrics["role_order"]) == GENERATOR_ROLES,
            "Generator role order changed")
    require(metrics["fitted_taxa"] == 613 and metrics["restored_zero_taxa_indices"] == [0],
            "Template-only zero-taxon policy differs")
    hashes(Path(metrics["installed_package"]["path"]), metrics["installed_package"]["sha256"])
    require(sha(metrics["package_source"]["path"]) == metrics["package_source"]["sha256"],
            "MIDASim vendor archive changed")
    require(sha(root/"data/generator_train.csv") == template["original_source_sha256"]
            == sha(template["original_source"]), "Generator template differs from exact original CSV")
    values = np.loadtxt(root/"data/generator_train.csv", delimiter=",")
    require(values.shape == (10443, 614) and np.isfinite(values).all() and (values >= 0).all(), "Invalid template")
    require(np.flatnonzero(~(values > 0).any(0)).tolist() == [0], "Unexpected template-empty taxa")
    manifest = rows(root/"data/manifest.csv")
    require(manifest.source_row.astype(int).tolist() == list(range(10443)), "Original generator row order changed")
    require(manifest.subject_id.nunique() == 8732 and not manifest.sample_id.duplicated().any(), "Template subject/sample counts differ")
    original = rows(parent/parent_data["studies"]["final"]["manifest"]["path"])
    development = original[original.distribution.eq("P") & original.role.isin(["train", "stopping"])]
    development = development.sort_values("source_row", key=lambda x: x.astype(int)).reset_index(drop=True)
    for field in ("sample_id", "subject_id", "source_file", "source_row", "row_sha256"):
        require(manifest[field].tolist() == development[field].tolist(), f"Generator template identities differ: {field}")
    require([hashlib.sha256(row.astype("<f4").tobytes()).hexdigest() for row in values]
            == manifest.row_sha256.tolist(), "Generator raw rows differ from archived float32 coordinates")
    tests = original[original.distribution.eq("P") & original.role.eq("test")]
    require(not set(tests.subject_id) & set(manifest.subject_id), "Generator/test subjects overlap")
    seen, generated, verified = {config["generation_seed"]}, {}, []
    for index, name in enumerate(GENERATOR_ROLES):
        stage, role = name.split("_", 1)
        n = ROLE_SIZES[stage][role+"_q"]
        require(config["roles"][name] == n, f"Wrong generated role size: {name}")
        entry = read(root/"generation"/(name+"_metrics.json"))
        expected_seed = config["generation_seed"]+100000+1000*index
        require(entry["generation_seed"] == expected_seed and entry["subset_seed"] == expected_seed+1,
                f"Wrong generation/subset seeds: {name}")
        require(not {expected_seed, expected_seed+1} & seen, "Repeated fitting/generation seed")
        seen |= {expected_seed, expected_seed+1}
        indices = pd.read_csv(root/"generation"/(name+"_indices.csv"))
        require(indices.output_row.tolist() == list(range(n)) and indices.generated_row.nunique() == n
                and indices.generated_row.between(0, 10442).all(), f"Invalid uniform-subset indices: {name}")
        require(entry["generated_rows_before_subsampling"] == 10443 and entry["rows"] == n,
                f"Wrong full-draw/subset count: {name}")
        require(entry["sha256"] == sha(root/"generation"/(name+".csv"))
                and entry["indices_sha256"] == sha(root/"generation"/(name+"_indices.csv")), "Role provenance differs")
        x = np.loadtxt(root/"generation"/(name+".csv"), delimiter=",")
        require(x.shape == (n, 614) and np.isfinite(x).all() and (x >= 0).all() and not x[:, 0].any(),
                f"Invalid generated compositions: {name}")
        close(x.sum(1), np.ones(n), f"Generated rows are not closed: {name}", atol=1e-10, rtol=0)
        generated[stage+":"+role] = x
        verified.append(dict(role=name, rows=n, generation_seed=expected_seed, subset_seed=expected_seed+1))
    return generated, dict(template_rows=10443, template_subjects=8732, fitted_taxa=613, restored_zero_taxa=[0],
        original_float64_csv_exact=True, template_test_subjects_disjoint=True, independent_role_draws=7,
        distinct_setup_generation_subset_seeds=len(seen), fixed_generated_tables_across_repeats=True, roles=verified)


def data_audit(analysis, protocol, parent, parent_data, generated):
    raw, manifests, counts = {}, {}, {}
    require(set(protocol["studies"]) == set(ROLE_SIZES), "Unexpected analysis studies")
    for stage, expected in ROLE_SIZES.items():
        spec = protocol["studies"][stage]
        require(set(spec["arrays"]) == set(expected), f"Unexpected cached roles: {stage}")
        manifest = rows(analysis/spec["manifest"]["path"])
        require(not manifest.sample_id.duplicated().any(), f"Duplicate sample identity: {stage}")
        require(not manifest.duplicated(["source_file", "source_row"]).any(), f"Duplicate source rows: {stage}")
        real = manifest[manifest.original_distribution.eq("P")]
        require(real.subject_id.ne("").all(), f"Missing real subject identity: {stage}")
        require((real.role+":"+real.distribution).groupby(real.subject_id).nunique().eq(1).all(),
                f"Real subjects cross RDR roles: {stage}")
        parent_manifest = rows(parent/parent_data["studies"][stage]["manifest"]["path"])
        raw[stage], counts[stage] = {}, {}
        for key, n in expected.items():
            role, side = key.rsplit("_", 1)
            receipt = spec["arrays"][key]
            x = np.load(analysis/receipt["path"], allow_pickle=False)
            require(x.shape == (n, 614) and x.dtype == np.float32, f"Cached dimension/dtype differs: {stage}/{key}")
            group = subset(manifest, role, side)
            require(group.array_row.astype(int).tolist() == list(range(n)), f"Cache row identities differ: {stage}/{key}")
            require([hashlib.sha256(row.astype("<f4").tobytes()).hexdigest() for row in x]
                    == group.row_sha256.tolist(), f"Cached row bytes differ: {stage}/{key}")
            if side == "p" or stage == "null":
                original = parent_data["studies"][stage]["arrays"][key]
                require(sha(analysis/receipt["path"]) == original["sha256"], f"Original P/null cache changed: {stage}/{key}")
                original_rows = subset(parent_manifest, role, side)
                for field in ("sample_id", "subject_id", "source_file", "source_row", "row_sha256",
                              "original_role", "original_distribution"):
                    require(group[field].tolist() == original_rows[field].tolist(), f"Original real identity differs: {stage}/{key}/{field}")
            else:
                require(np.array_equal(x, generated[stage+":"+role].astype(np.float32)), f"Float64-to-float32 import differs: {stage}/{key}")
                require(group.subject_id.eq("").all() and group.sample_id.tolist() ==
                        [f"MIDASim:{stage}:{role}:{i:06d}" for i in range(n)], "Generated rows have incorrect identities")
                require(group.source_row.astype(int).tolist() == list(range(n)), "Generated source row indices changed")
                require(group.source_file.str.endswith("generation/"+stage+"_"+role+".csv").all(), "Generated source path differs")
                if "original_manifest_row" in group:
                    require(group.original_manifest_row.eq("").all(), "Generated rows falsely reference archived ICFM identities")
            require(spec["counts"][key]["rows"] == spec["role_counts"][role][side] == n, "Protocol/cache count mismatch")
            raw[stage][key] = x
            counts[stage][key] = dict(rows=n, subjects=int(group.loc[group.subject_id.ne(""), "subject_id"].nunique()))
        manifests[stage] = manifest
    require(not set(manifests["selection"].sample_id) & set(
        manifests["final"].loc[manifests["final"].role.eq("test"), "sample_id"]), "Selection contains final test rows")
    return raw, manifests, counts


def model_from_weights(config, weights):
    width = config["hidden_dimension"]
    layers = [torch.nn.Linear(613, width), torch.nn.ReLU()]
    for _ in range(config["blocks"]):
        layers += [torch.nn.Linear(width, width), torch.nn.ReLU(), torch.nn.Linear(width, width), torch.nn.ReLU()]
    layers.append(torch.nn.Linear(width, 1))
    model = torch.nn.Sequential(*layers)
    model.load_state_dict({name.removeprefix("model."): value for name, value in weights.items()}, strict=True)
    return model.eval()


def predict(model, raw, state, alpha):
    values = np.asarray(raw, dtype=np.float64)
    values = values/values.sum(1, keepdims=True)
    logarithms = np.log((values+float(state["epsilon"]))/ (1+614*float(state["epsilon"])))
    clr = logarithms-logarithms.mean(1, keepdims=True)
    ilr = (((clr-state["center"])/float(state["scale"])) @ state["basis"].T).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(ilr)).flatten().double()
        return (2*torch.sigmoid(float(alpha)*logits)).numpy()


def checkpoint_audit(analysis, config, chosen, work, raw, manifests, math):
    transforms = {stage: math._expected_transform(values, config["epsilon"]) for stage, values in raw.items()}
    verified, selection_rows = [], []
    for index, (stage, task, directory) in enumerate(work):
        checkpoint = torch.load(directory/"model.pt", map_location="cpu", weights_only=True)
        require(checkpoint["task"] == task and checkpoint["config"] == config, "Checkpoint identity/configuration differs")
        training, metrics = checkpoint["training"], read(directory/"metrics.json")
        for key, value in training.items():
            require(metrics[key] == value, f"Checkpoint/training metadata differ: {directory}/{key}")
        require(all(metrics[key] == value for key, value in task.items()), "Task metric identity differs")
        seed = config["seed"]+{"selection": 0, "final": 100000, "null": 200000}[stage]+1009*task["repeat"]
        require(training["seed_initialization"] == seed, "Raw model initialization seed differs")
        history = pd.read_csv(directory/"history.csv", float_precision="round_trip")
        require(history.epoch.tolist() == list(range(1, len(history)+1)), "Incomplete epoch history")
        require(config["min_epochs"] <= len(history) <= config["max_epochs"] and training["epochs_run"] == len(history),
                "Training epoch count/budget differs")
        best = history.loc[history.stopping_brier.idxmin()]
        require(training["best_epoch"] == int(best.epoch), "Checkpoint is not first absolute stopping-Brier minimum")
        close(training["stopping_brier"], best.stopping_brier, "Stopping history minimum differs")
        close(training["restored_stopping_brier"], best.stopping_brier, "Restored Brier metadata differs")
        require(training["stopping_reason"] == ("patience" if history.iloc[-1].epochs_without_improvement >= config["patience"]
                                               else "max_epochs"), "Stopping reason differs")
        with np.load(directory/"transform.npz", allow_pickle=False) as saved:
            state = {key: saved[key] for key in saved.files}
        require(set(state) == set(transforms[stage]), "Unexpected transform state")
        for key, value in transforms[stage].items():
            close(state[key], value, f"Independent train-only ILR differs: {stage}/{key}", atol=2e-12, rtol=2e-12)
        model = model_from_weights(config, checkpoint["model"])
        stop = [predict(model, raw[stage]["stopping_"+side], state, task["output_alpha"]) for side in ("p", "q")]
        stopping_brier = math._brier(*stop)
        close(stopping_brier, best.stopping_brier, "Independently rescored stopping Brier differs", atol=2e-8, rtol=2e-6)
        roles = ("calibration", "evaluation") if stage == "selection" else ("test",)
        with np.load(directory/"predictions.npz", allow_pickle=False) as saved:
            archived_predictions = {key: saved[key] for key in saved.files}
        require(set(archived_predictions) == {role+"_"+side for role in roles for side in ("p", "q")}, "Diagnostic roles differ")
        scores, predictions, maximum_prediction_error = rows(directory/"scores.csv"), {}, 0.
        for role in roles:
            for side in ("p", "q"):
                key = role+"_"+side
                predictions[key] = predict(model, raw[stage][key], state, task["output_alpha"])
                close(predictions[key], archived_predictions[key], f"Independently rescored predictions differ: {directory}/{key}",
                      atol=3e-7, rtol=2e-6)
                maximum_prediction_error = max(maximum_prediction_error,
                    float(np.max(abs(predictions[key]-archived_predictions[key]))))
                expected_rows = subset(manifests[stage], role, side)
                score_rows = subset(scores, role, side)
                require(set(score_rows.columns) == set(expected_rows.columns) | {"rdr"}, "Score identity columns differ")
                require(score_rows[expected_rows.columns].equals(expected_rows), "Score row identities differ")
                close(score_rows.rdr.astype(float), archived_predictions[key], "CSV score values differ", atol=1e-14, rtol=1e-13)
        cal, evaluate = ("calibration", "evaluation") if stage == "selection" else ("test", "test")
        alpha = config["ci_alpha"]/16 if stage == "selection" else config["ci_alpha"]
        # Use archived exact scores for count-boundary comparisons after checkpoint rescore checks.
        p, q = archived_predictions[evaluate+"_p"], archived_predictions[evaluate+"_q"]
        diagnostic, cells = math._diagnostics(archived_predictions[cal+"_p"], archived_predictions[cal+"_q"], p, q, alpha, 20)
        ranks = rankdata(np.r_[p, q], method="average")
        diagnostic["auc"] = float((ranks[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(q)))
        for key, value in diagnostic.items():
            close(metrics[key], value, f"Independent diagnostic differs: {directory}/{key}")
        saved_cells = pd.read_csv(directory/"cells.csv", float_precision="round_trip").sort_values("bin")
        require(len(saved_cells) == 20, "Fixed-bin count differs")
        for column in cells:
            close(saved_cells[column], cells[column], f"Independent cell differs: {directory}/{column}", atol=2e-9)
        if stage == "selection":
            selection_rows.append({**task, **diagnostic})
        elif stage == "null":
            close(metrics["true_rdr_mse"], .5*(np.square(p-1).mean()+np.square(q-1).mean()), "Null RDR MSE differs")
        verified.append(dict(stage=stage, **task, seed=seed, best_epoch=int(best.epoch), epochs_run=len(history),
            stopping_reason=training["stopping_reason"], stopping_brier=stopping_brier,
            maximum_stopping_brier_difference=abs(stopping_brier-float(best.stopping_brier)),
            maximum_prediction_difference=maximum_prediction_error, **diagnostic))
        if (index+1) % 10 == 0:
            print(f"Verified {index+1}/{len(work)} checkpoints and fixed-width diagnostics", flush=True)
    reference, candidate, eligible = math._independent_choice(selection_rows)
    decision = read(analysis/"selection/selection.json")
    require(candidate == chosen["candidate"] and reference == decision["brier_reference"], "Independent selection decision differs")
    require(set(eligible) == {r["candidate"] for r in decision["ranking"] if r["eligible"]}, "Independent Brier/Gap shortlist differs")
    return verified, dict(chosen=chosen, brier_reference=reference, independently_eligible_candidates=sorted(eligible))


def verify(output):
    started = time.monotonic()
    outer = Path(output).resolve()
    analysis, gen = outer/"analysis", outer/"generator"
    source = analysis/"source"
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source/"experiments/AGP"))
    wrapper = module("_frozen_full_midasim_audit_runner", source/"experiments/AGP/midasim_full.py")
    require(Path(wrapper.shared.__file__).resolve() == source/"experiments/AGP/selected_workflow.py", "Unfrozen shared runner loaded")
    completion = wrapper.verify_analysis(outer)
    protocol = read(analysis/"protocol.json")
    relative = "experiments/AGP/verify_midasim_full.py"
    require(sha(__file__) == protocol["source_sha256"].get(relative), "Executing verifier differs from frozen verifier source")
    completion_before, generator_before = sha(analysis/"COMPLETE.json"), sha(gen/"generation/COMPLETE.json")
    math = module("_frozen_full_midasim_independent_math", source/"experiments/AGP/verify_selected.py")
    pilot_audit = module("_frozen_full_midasim_parent_audit", source/"experiments/AGP/verify_midasim.py")
    template = read(gen/"data/protocol.json")
    parent, parent_data, parent_result = pilot_audit.parent_audit(outer, template)
    config, data = read(analysis/"config.json"), read(analysis/"data/protocol.json")
    require(config["seed"] == 2026100400 and config["repeats"] == 5 and config["bins"] == 20
            and config["representation"] == "ilr" and config["architecture"] == "deep"
            and config["hidden_dimension"] == 128 and config["blocks"] == 3 and config["epsilon"] == 1e-6
            and config["min_epochs"] == 100 and config["max_epochs"] == 1000, "Declared model/training protocol differs")
    expected_grid = {(loss, alpha, repeat) for loss in ("hellinger", "kl", "chisq", "js")
                     for alpha in (.5, 1., 2., 4.) for repeat in range(5)}
    tasks = wrapper.shared.selection_tasks(config)
    require(len(tasks) == 80 and {(t["loss"], t["output_alpha"], t["repeat"]) for t in tasks} == expected_grid,
            "Candidate/repeat grid differs")
    chosen = wrapper.shared.checked_selection(analysis)
    work = [("selection", task, analysis/"selection/runs"/task["candidate"]/f"{task['repeat']:02d}") for task in tasks]
    work += [(stage, {**chosen, "repeat": repeat}, analysis/stage/f"{repeat:02d}")
             for stage in ("final", "null") for repeat in range(5)]
    expected_markers = {directory/"COMPLETE.json" for _, _, directory in work}
    actual = set(analysis.glob("selection/runs/*/*/COMPLETE.json"))
    actual |= set(analysis.glob("final/*/COMPLETE.json")) | set(analysis.glob("null/*/COMPLETE.json"))
    require(expected_markers == actual, "Missing or extra completed fits")
    require(not list(gen.rglob("FAILED.json")) and not list(analysis.rglob("FAILED.json")), "A failed-task marker remains")
    require(set(completion["assessment_completions"]) ==
            {f"{stage}/{repeat:02d}/COMPLETE.json" for stage in ("final", "null") for repeat in range(5)},
            "Root final/null task receipt set differs")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        generated, generation_result = generator_audit(outer, template, parent, parent_data)
        raw, manifests, counts = data_audit(analysis, data, parent, parent_data, generated)
        verified, selection = checkpoint_audit(analysis, config, chosen, work, raw, manifests, math)
    require(sha(analysis/"COMPLETE.json") == completion_before and sha(gen/"generation/COMPLETE.json") == generator_before,
            "Study completion changed during verification")
    require(sha(parent/"COMPLETE.json") == parent_result["completion_sha256"], "Parent archive changed during verification")
    return dict(status="passed", output=str(outer), analysis_completion_sha256=completion_before,
        generator_completion_sha256=generator_before, protocol_sha256=sha(analysis/"protocol.json"),
        selection_freeze_sha256=sha(analysis/"selection/freeze.json"), parent=parent_result,
        generator=generation_result, exact_counts=counts, selection=selection, selection_fits=80, final_fits=5, null_fits=5,
        checkpoint_stopping_rescores=90, selection_calibration_evaluation_rescores=80, final_null_test_rescores=10,
        independently_checked_fixed_width_cells=1800, independently_checked_clopper_pearson_c2_cells=1800,
        maximum_prediction_difference=max(v["maximum_prediction_difference"] for v in verified),
        maximum_stopping_brier_difference=max(v["maximum_stopping_brier_difference"] for v in verified),
        cap_limited_fits=[{k: v[k] for k in ("stage", "candidate", "repeat", "epochs_run")}
                          for v in verified if v["epochs_run"] == config["max_epochs"]],
        models=verified, elapsed_seconds=time.monotonic()-started,
        helper_source_sha256={name: sha(source/"experiments/AGP"/name) for name in ("verify_selected.py", "verify_midasim.py")},
        numerical_method="Independent CLR/ILR and Sequential checkpoint reconstruction, float64 sigmoid, rank AUC, histogram Brier/Gap and direct beta-quantile C2; independent paired one-SE shortlist",
        interpretation="Descriptive conditional RDR study. Generated rows within a role are dependent; generator training overlaps real development roles; historical test and preprocessing are retrospective. No empirical coverage or IID finite-sample guarantee is asserted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Outer directory containing generator/ and analysis/")
    parser.add_argument("--receipt", type=Path, required=True, help="Fresh JSON file outside the experiment archive")
    args = parser.parse_args()
    output, receipt = args.output.resolve(), args.receipt.resolve()
    require(not receipt.exists(), "Refusing to overwrite an existing verification receipt")
    require(receipt.parent.is_dir(), "Verification receipt parent directory does not exist")
    require(output not in receipt.parents, "Receipt must be outside the experiment archive")
    try:
        result, status = verify(output), 0
    except Exception as error:
        result, status = dict(status="failed", output=str(output), error=f"{type(error).__name__}: {error}"), 1
    result.update(verified_utc=datetime.now(timezone.utc).isoformat(), verifier_source=str(Path(__file__).resolve()),
                  verifier_source_sha256=sha(__file__))
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], receipt=str(receipt), error=result.get("error"))), flush=True)
    raise SystemExit(status)


if __name__ == "__main__":
    main()
