#!/usr/bin/env python3
"""Frozen global baselines for the four retained AGP generators.

Prepare fits balanced development-only PRD quantizers and freezes all settings
before test arrays are read. Assess uses that lock for declared-feature Gaussian
Frechet distances and complete PRD curves. No generation or RDR fitting occurs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

try:
    from . import rdr_metric_utils as metric
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rdr_metric_utils as metric


BASE = Path("/cwork/yx306/RDR/AGP")
ROOTS = {
    "bernoulli_mixture_icfm": BASE / "mask_rdr_20261012/analyses/bernoulli_mixture_icfm",
    "autoregressive_icfm": BASE / "mask_rdr_20261012/analyses/autoregressive_icfm",
    "midasim_mask_icfm": BASE / "icfm_zero_20261007/analysis",
    "midasim": BASE / "midasim_full_20261004/analysis",
}
METHODS = tuple(ROOTS)
REPRESENTATIONS = ("sqrt", "clr")
SEEDS = tuple(2026100901 + i for i in range(5))
N_CLUSTERS = 64
MAX_TRAIN_PER_SOURCE = 2048
REPOSITORY = Path(__file__).resolve().parents[2]
PUBLISHED = REPOSITORY / "results/AGP/four_generator_case_study_20261013"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Changed or invalid upstream input: {path}")
    return actual


def metadata(root):
    """Verify completion -> root protocol -> data protocol, reading no arrays."""
    complete = read(root / "COMPLETE.json")
    if complete["status"] != "complete":
        raise ValueError(f"Incomplete source study: {root}")
    checked(root / "protocol.json", complete["protocol_sha256"])
    protocol = read(root / "protocol.json")
    checked(root / "data/protocol.json", protocol["data_protocol_sha256"])
    data = read(root / "data/protocol.json")
    if data["feature_dimension"] != 614:
        raise ValueError("Expected the audited 614-taxon AGP panel")
    return data


def load_array(root, spec):
    source = root / spec["path"]
    checked(source, spec["sha256"])
    values = np.load(source, allow_pickle=False)
    if list(values.shape) != spec["shape"] or str(values.dtype) != spec["dtype"]:
        raise ValueError(f"Unexpected array shape or dtype: {source}")
    return metric.close(values)


def model_name(method, representation, seed):
    return f"{method}_{representation}_{seed}.joblib"


def prepare(output, published_reference=PUBLISHED):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing outputs; preparation requires a fresh directory")
    data = {method: metadata(root) for method, root in ROOTS.items()}
    arrays = {method: {key: value for key, value in spec["studies"]["final"]["arrays"].items()
                       if key in ("train_p", "train_q", "test_p", "test_q")}
              for method, spec in data.items()}
    # Same P cohort and taxon order are required for the four-method comparison.
    if len({spec["taxa"].__str__() for spec in data.values()}) != 1:
        raise ValueError("Taxon order differs across generator studies")
    for role in ("train_p", "test_p"):
        if len({spec[role]["sha256"] for spec in arrays.values()}) != 1:
            raise ValueError("Real-data array receipts differ across generator studies")
    inputs = {str(root / name): sha(root / name) for root in ROOTS.values()
              for name in ("COMPLETE.json", "protocol.json", "data/protocol.json")}
    output.mkdir(parents=True)
    (output / "models").mkdir()
    (output / "source").mkdir()
    sources = [Path(__file__).resolve(), Path(metric.__file__).resolve()]
    for source in sources:
        shutil.copyfile(source, output / "source" / source.name)
    quantizer = dict(n_clusters=N_CLUSTERS, n_init=1, max_iter=200, batch_size=1024,
                     max_no_improvement=10, reassignment_ratio=.01, init_size=None, tol=0.)
    protocol = dict(status="prespecified_retrospective_metric_extension", created_utc=now(),
        roots={k: str(v) for k, v in ROOTS.items()}, methods=METHODS, representations=REPRESENTATIONS,
        primary="sqrt", sensitivity="clr", epsilon=1e-6, seeds=SEEDS, quantizer=quantizer,
        max_train_per_source=MAX_TRAIN_PER_SOURCE, threads=2,
        sampling="Pair-specific balanced uniform subsets without replacement; 2048 per source; same seed indices across representations.",
        development_role="Existing final/train P and Q arrays; stopping and test arrays excluded from fitting.",
        fd="Gaussian Frechet analogue in declared composition features, using unbiased sample covariances; not image Inception FID.",
        prd="P reference, Q evaluated; frozen development-only quantizer; 201 angles in [1e-10, pi/2-1e-10].",
        prd_uncertainty="Mean and sample SD describe quantizer-seed variation on fixed assessment observations, not sampling uncertainty.",
        fd_uncertainty="FD uses all assessment observations and is invariant across PRD quantizer seeds; its reported seed SD is zero.",
        inception_score="Not reported: no externally validated, scientifically justified AGP sample-class classifier.",
        scope="Retrospective existing cohort; no new generator or RDR fits, no controlled perturbation, no selection by assessment metrics.",
        no_test_arrays_read=True, arrays=arrays,
        role_counts={m: data[m]["studies"]["final"]["counts"] for m in METHODS},
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas", "joblib")},
        sources={str(p): sha(p) for p in sources}, metadata_inputs=inputs,
        published_reference_root=str(Path(published_reference).resolve()),
        published_reference_purpose="Byte-pinned regression reference for the two reused MIDASim comparisons; the publication default is the completed four-generator aggregate, not an independent new validation.")
    write(output / "protocol.json", protocol)
    rows, training_inputs = [], {}
    with threadpool_limits(limits=2):
        for method, root in ROOTS.items():
            p, q = [load_array(root, arrays[method][f"train_{side}"]) for side in ("p", "q")]
            if (len(p), len(q), p.shape[1]) != (9162, 7310, 614):
                raise ValueError("Unexpected final training-role counts")
            for side in ("p", "q"):
                spec = arrays[method][f"train_{side}"]
                training_inputs[str(root / spec["path"])] = spec["sha256"]
            for representation in REPRESENTATIONS:
                xp, xq = metric.embed(p, representation), metric.embed(q, representation)
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    n = min(len(p), len(q), MAX_TRAIN_PER_SOURCE)
                    ip, iq = rng.choice(len(p), n, replace=False), rng.choice(len(q), n, replace=False)
                    model = MiniBatchKMeans(random_state=seed, **quantizer)
                    model.fit(np.concatenate([xp[ip], xq[iq]]))
                    name = model_name(method, representation, seed)
                    joblib.dump(model, output / "models" / name)
                    np.savez_compressed(output / "models" / name.replace(".joblib", "_rows.npz"), p=ip, q=iq)
                    rows.append(dict(method=method, representation=representation, seed=seed,
                        n_train_p=n, n_train_q=n, dimensions=p.shape[1], n_iterations=int(model.n_iter_),
                        n_steps=int(model.n_steps_), cap_reached=bool(model.n_iter_ >= quantizer["max_iter"])))
            print(f"Frozen quantizers: {method}", flush=True)
    pd.DataFrame(rows).to_csv(output / "quantizer_fits.csv", index=False)
    if len(rows) != 40:
        raise ValueError("Expected 40 frozen quantizers")
    for source, expected in {**inputs, **training_inputs, **protocol["sources"]}.items():
        checked(source, expected)
    freeze = dict(status="frozen", created_utc=now(), no_test_arrays_read=True, n_models=len(rows),
        protocol_sha256=sha(output / "protocol.json"), training_inputs=training_inputs,
        files={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()})
    write(output / "FREEZE.json", freeze)
    return dict(status="frozen", n_models=len(rows), output=str(output))


def verify_freeze(output):
    freeze, protocol = read(output / "FREEZE.json"), read(output / "protocol.json")
    if freeze["status"] != "frozen" or not freeze["no_test_arrays_read"] or freeze["n_models"] != 40:
        raise ValueError("Incomplete development freeze")
    checked(output / "protocol.json", freeze["protocol_sha256"])
    for name, expected in freeze["files"].items():
        checked(output / name, expected)
    for source, expected in {**protocol["sources"], **protocol["metadata_inputs"], **freeze["training_inputs"]}.items():
        checked(source, expected)
    current = {name: importlib.metadata.version(name) for name in protocol["versions"]}
    if current != protocol["versions"]:
        raise ValueError("Package versions changed since quantizer preparation")
    return protocol


def check_published(summary, published):
    """Check reused-pair FD against either published four-method or legacy data.

    This is a numerical regression check. The four-generator publication ships
    its own completed aggregate, so new reproductions need no perturbation data.
    Legacy frozen protocols still retain their original reference and receipts.
    """
    published = Path(published)
    path = published / "global/global_metrics.csv"
    manifest_path = published / "manifest.json"
    if manifest_path.is_file():
        seal_path = manifest_path
        seal = read(seal_path)
        if seal["status"] != "complete" or tuple(seal["retained_methods"]) != METHODS:
            raise ValueError("Published four-generator comparison is incomplete")
        checked(path, seal["files_sha256"]["global/global_metrics.csv"])
        historical = False
    else:
        seal_path = published / "COMPLETE.json"
        seal = read(seal_path)
        if seal["status"] != "complete":
            raise ValueError("Published comparison is incomplete")
        checked(path, seal["files"]["global/global_metrics.csv"])
        historical = True
    frame = pd.read_csv(path)
    rows = []
    for method in ("midasim_mask_icfm", "midasim"):
        previous = "zero_aware" if historical and method == "midasim_mask_icfm" else method
        for representation in REPRESENTATIONS:
            selected = frame.method.eq(previous) & frame.representation.eq(representation)
            if historical:
                selected &= frame.scope.eq("full")
            reference = frame[selected]
            values = summary[summary.method.eq(method) & summary.representation.eq(representation)]
            if len(reference) != 5 or len(values) != 5:
                raise ValueError("Published reference counts differ")
            actual, expected = float(values.frechet.iloc[0]), float(reference.frechet.iloc[0])
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-9)
            rows.append(dict(method=method, representation=representation, frechet=actual,
                             published_frechet=expected, absolute_difference=abs(actual - expected), passed=True))
    return pd.DataFrame(rows), {str(path): sha(path), str(seal_path): sha(seal_path)}


def assess(output):
    output = Path(output).resolve()
    protocol = verify_freeze(output)
    if (output / "summary").exists() or (output / "ASSESSMENT_RELEASE.json").exists():
        raise FileExistsError("Assessment already started; preserve the existing run")
    write(output / "ASSESSMENT_RELEASE.json", dict(status="released", created_utc=now(),
          freeze_sha256=sha(output / "FREEZE.json"), protocol_sha256=sha(output / "protocol.json")))
    summary = output / "summary"
    summary.mkdir()
    rows, curves, histograms, assessment_inputs = [], [], [], {}
    with threadpool_limits(limits=2):
        for method in protocol["methods"]:
            root = Path(protocol["roots"][method])
            arrays = protocol["arrays"][method]
            p, q = [load_array(root, arrays[f"test_{side}"]) for side in ("p", "q")]
            if (len(p), len(q), p.shape[1]) != (2002, 2089, 614):
                raise ValueError("Unexpected assessment-role counts")
            for side in ("p", "q"):
                spec = arrays[f"test_{side}"]
                assessment_inputs[str(root / spec["path"])] = spec["sha256"]
            for representation in protocol["representations"]:
                xp, xq = metric.embed(p, representation), metric.embed(q, representation)
                fd = metric.frechet(xp, xq)
                for seed in protocol["seeds"]:
                    model = joblib.load(output / "models" / model_name(method, representation, seed))
                    ph = np.bincount(model.predict(xp), minlength=N_CLUSTERS)
                    qh = np.bincount(model.predict(xq), minlength=N_CLUSTERS)
                    precision, recall = metric.prd(ph, qh)
                    key = dict(method=method, representation=representation, seed=seed)
                    rows.append(dict(**key, n_p=len(p), n_q=len(q), frechet=fd,
                                     prd_max_f1=metric.prd_max_f1(precision, recall)))
                    for angle, (prec, rec) in enumerate(zip(precision, recall)):
                        curves.append(dict(**key, angle=angle, precision=float(prec), recall=float(rec)))
                    for cluster in range(N_CLUSTERS):
                        histograms.append(dict(**key, cluster=cluster, count_p=int(ph[cluster]), count_q=int(qh[cluster])))
            print(f"Assessed frozen metrics: {method}", flush=True)
    frame, curve = pd.DataFrame(rows), pd.DataFrame(curves)
    if len(frame) != 40 or len(curve) != 8040:
        raise ValueError("Incomplete FD/PRD assessment")
    aggregate = frame.groupby(["method", "representation"], sort=False).agg(
        seeds=("seed", "size"), n_p=("n_p", "first"), n_q=("n_q", "first"),
        frechet_mean=("frechet", "mean"), frechet_sd=("frechet", "std"),
        prd_max_f1_mean=("prd_max_f1", "mean"), prd_max_f1_sd=("prd_max_f1", "std")).reset_index()
    curve_summary = curve.groupby(["method", "representation", "angle"], sort=False).agg(
        precision_mean=("precision", "mean"), precision_sd=("precision", "std"),
        recall_mean=("recall", "mean"), recall_sd=("recall", "std")).reset_index()
    reference, published_inputs = check_published(frame, Path(protocol["published_reference_root"]))
    outputs = {"global_metrics.csv": frame, "global_metrics_aggregate.csv": aggregate,
               "prd_curves.csv": curve, "prd_curve_summary.csv": curve_summary,
               "prd_histograms.csv": pd.DataFrame(histograms), "fd_reference_check.csv": reference}
    for name, table in outputs.items():
        table.to_csv(summary / name, index=False)
    verify_freeze(output)
    for source, expected in assessment_inputs.items():
        checked(source, expected)
    complete = dict(status="complete", completed_utc=now(), freeze_sha256=sha(output / "FREEZE.json"),
        release_sha256=sha(output / "ASSESSMENT_RELEASE.json"), assessment_inputs=assessment_inputs,
        published_reference_inputs=published_inputs, tables={name: len(table) for name, table in outputs.items()},
        files={name: sha(summary / name) for name in outputs}, no_generator_or_rdr_fits=True,
        no_numeric_inception_score=True, retrospective=True)
    write(summary / "COMPLETE.json", complete)
    return dict(status="complete", output=str(output), tables=complete["tables"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "assess", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--published-reference", type=Path, default=PUBLISHED,
                        help="Frozen FD regression reference for prepare/run; defaults to the shipped four-generator aggregate")
    args = parser.parse_args()
    if args.phase == "run":
        print(json.dumps(prepare(args.output, args.published_reference), sort_keys=True), flush=True)
        result = assess(args.output)
    elif args.phase == "prepare":
        result = prepare(args.output, args.published_reference)
    else:
        result = assess(args.output)
    print(json.dumps(result, sort_keys=True), flush=True)
