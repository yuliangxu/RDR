#!/usr/bin/env python3
"""Independently audit completed four-generator AGP global metrics.

Reconstruct FD from centered-data QR factors and PRD from explicit distances to
frozen cluster centers. Replay all balanced development subsets from their seeds.
No producer module is imported and no fitted artifact or previous receipt is
modified. By default, write a fresh OUTPUT/verification.json; --receipt permits
an additional audit of an already verified archive without replacing evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits


METHODS = ("bernoulli_mixture_icfm", "autoregressive_icfm", "midasim_mask_icfm", "midasim")
REPRESENTATIONS = ("sqrt", "clr")
SEEDS = tuple(2026100901 + i for i in range(5))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_hashes(mapping, base=None):
    for name, expected in mapping.items():
        path = Path(name) if base is None else Path(base) / name
        if sha(path) != expected:
            raise ValueError(f"Changed audit input: {path}")


def features(root, spec, representation):
    path = root / spec["path"]
    check_hashes({str(path): spec["sha256"]})
    raw = np.load(path, allow_pickle=False)
    if list(raw.shape) != spec["shape"] or str(raw.dtype) != spec["dtype"]:
        raise ValueError(f"Invalid assessment-array schema: {path}")
    x = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(x).all() or (x < 0).any() or (x.sum(axis=1) <= 0).any():
        raise ValueError(f"Invalid assessment composition: {path}")
    # Match the declared loading closure followed by embedding closure while
    # independently constructing both transformations without the producer.
    x = x / x.sum(axis=1, keepdims=True)
    x = x / x.sum(axis=1, keepdims=True)
    if representation == "sqrt":
        return np.sqrt(x)
    logs = np.log(x + 1e-6)
    return logs - logs.mean(axis=1, keepdims=True)


def gaussian_distance(x, y):
    """Independent QR/nuclear-norm formula; handles singular covariances."""
    xc = (x - x.mean(axis=0)) / np.sqrt(len(x) - 1)
    yc = (y - y.mean(axis=0)) / np.sqrt(len(y) - 1)
    rx, ry = np.linalg.qr(xc, mode="r"), np.linalg.qr(yc, mode="r")
    value = float(np.square(x.mean(axis=0) - y.mean(axis=0)).sum()
                  + np.square(xc).sum() + np.square(yc).sum()
                  - 2 * np.linalg.svd(rx @ ry.T, compute_uv=False).sum())
    if value < -1e-7:
        raise FloatingPointError("Independent FD is materially negative")
    return max(0., value)


def verify(output, receipt=None):
    output = Path(output).resolve()
    receipt = Path(receipt).resolve() if receipt is not None else output / "verification.json"
    if receipt.exists():
        raise FileExistsError("Preserve the existing audit receipt; use --receipt with a fresh path")
    summary = output / "summary"
    protocol, freeze = read(output / "protocol.json"), read(output / "FREEZE.json")
    release, complete = read(output / "ASSESSMENT_RELEASE.json"), read(summary / "COMPLETE.json")
    if (freeze["status"] != "frozen" or release["status"] != "released"
            or complete["status"] != "complete" or not freeze["no_test_arrays_read"]
            or freeze["n_models"] != 40):
        raise ValueError("Incomplete development or assessment stage")
    if not (freeze["created_utc"] <= release["created_utc"] <= complete["completed_utc"]):
        raise ValueError("Invalid freeze/release/completion ordering")
    check_hashes({str(output / "protocol.json"): freeze["protocol_sha256"]})
    check_hashes({str(output / "protocol.json"): release["protocol_sha256"],
                  str(output / "FREEZE.json"): release["freeze_sha256"]})
    check_hashes({str(output / "FREEZE.json"): complete["freeze_sha256"],
                  str(output / "ASSESSMENT_RELEASE.json"): complete["release_sha256"]})
    check_hashes(freeze["files"], output)
    check_hashes(complete["files"], summary)
    dependencies = {**protocol["sources"], **protocol["metadata_inputs"],
                    **freeze["training_inputs"], **complete["assessment_inputs"],
                    **complete["published_reference_inputs"]}
    check_hashes(dependencies)
    if (tuple(protocol["methods"]) != METHODS or tuple(protocol["representations"]) != REPRESENTATIONS
            or tuple(protocol["seeds"]) != SEEDS or protocol["epsilon"] != 1e-6
            or protocol["quantizer"]["n_clusters"] != 64 or protocol["max_train_per_source"] != 2048):
        raise ValueError("Unexpected declared metric design")
    metrics = pd.read_csv(summary / "global_metrics.csv")
    curves = pd.read_csv(summary / "prd_curves.csv")
    histograms = pd.read_csv(summary / "prd_histograms.csv")
    if len(metrics) != 40 or len(curves) != 8040 or len(histograms) != 2560:
        raise ValueError("Incomplete metric, curve, or histogram tables")
    fd_error, curve_error, n_models = 0., 0., 0
    with threadpool_limits(limits=2):
        for method in METHODS:
            root = Path(protocol["roots"][method])
            arrays = protocol["arrays"][method]
            for representation in REPRESENTATIONS:
                x, y = [features(root, arrays[f"test_{side}"], representation) for side in ("p", "q")]
                if x.shape != (2002, 614) or y.shape != (2089, 614):
                    raise ValueError("Unexpected assessment-role counts")
                fd = gaussian_distance(x, y)
                rows = metrics[metrics.method.eq(method) & metrics.representation.eq(representation)]
                if set(rows.seed) != set(SEEDS) or len(rows) != 5:
                    raise ValueError("Metric table lacks five unique seed rows")
                np.testing.assert_allclose(rows.frechet, fd, atol=1e-8, rtol=1e-9)
                fd_error = max(fd_error, float(np.abs(rows.frechet - fd).max()))
                for seed in SEEDS:
                    stem = f"{method}_{representation}_{seed}"
                    model = joblib.load(output / "models" / f"{stem}.joblib")
                    ph = np.bincount(cdist(x, model.cluster_centers_).argmin(axis=1), minlength=64)
                    qh = np.bincount(cdist(y, model.cluster_centers_).argmin(axis=1), minlength=64)
                    saved = histograms[histograms.method.eq(method) & histograms.representation.eq(representation)
                                       & histograms.seed.eq(seed)].sort_values("cluster")
                    np.testing.assert_array_equal(saved.cluster, np.arange(64))
                    np.testing.assert_array_equal(ph, saved.count_p)
                    np.testing.assert_array_equal(qh, saved.count_q)
                    slopes = np.tan(np.linspace(1e-10, np.pi / 2 - 1e-10, 201))
                    precision = np.minimum(slopes[:, None] * (ph / ph.sum()), qh / qh.sum()).sum(axis=1)
                    recall = precision / slopes
                    saved = curves[curves.method.eq(method) & curves.representation.eq(representation)
                                   & curves.seed.eq(seed)].sort_values("angle")
                    np.testing.assert_array_equal(saved.angle, np.arange(201))
                    np.testing.assert_allclose(saved.precision, precision, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(saved.recall, recall, atol=1e-12, rtol=0)
                    curve_error = max(curve_error, float(np.abs(saved.precision - precision).max()),
                                      float(np.abs(saved.recall - recall).max()))
                    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(precision),
                                   where=precision + recall > 0).max()
                    np.testing.assert_allclose(rows[rows.seed.eq(seed)].prd_max_f1, f1, atol=1e-12, rtol=0)
                    rng = np.random.default_rng(seed)
                    with np.load(output / "models" / f"{stem}_rows.npz", allow_pickle=False) as indices:
                        for side, count in (("p", 9162), ("q", 7310)):
                            if arrays[f"train_{side}"]["shape"][0] != count:
                                raise ValueError("Unexpected declared development count")
                            np.testing.assert_array_equal(indices[side], rng.choice(count, 2048, replace=False))
                    n_models += 1
            print(f"Independent metric verification passed: {method}", flush=True)
    # Confirm that the audit itself left every frozen or assessed input intact.
    check_hashes(freeze["files"], output)
    check_hashes(complete["files"], summary)
    check_hashes(dependencies)
    result = dict(status="passed", verified_utc=datetime.now(timezone.utc).isoformat(),
        completion_sha256=sha(summary / "COMPLETE.json"), methods=4, models=n_models,
        fd_values=8, full_curve_points=len(curves),
        maximum_independent_fd_absolute_difference=fd_error,
        maximum_independent_prd_curve_absolute_difference=curve_error,
        fd_method="Independent centered-data QR factors and nuclear norm",
        prd_method="Explicit scipy cdist nearest centers, histogram counts, independent angle formula",
        training_subsets="Exact seeded balanced subset replay verified for all 40 models",
        freeze_and_completion_hashes="passed", verifier_source=str(Path(__file__).resolve()),
        verifier_sha256=sha(__file__),
        versions={name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "pandas", "joblib")})
    # Exclusive creation also protects against another process writing a receipt
    # after the initial check. Existing evidence is never overwritten.
    with receipt.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return dict(status="passed", receipt=str(receipt), models=n_models,
                maximum_independent_fd_absolute_difference=fd_error,
                maximum_independent_prd_curve_absolute_difference=curve_error,
                verifier_sha256=result["verifier_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Completed metrics directory")
    parser.add_argument("--receipt", type=Path, help="Fresh audit JSON path; default OUTPUT/verification.json")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, args.receipt), sort_keys=True), flush=True)
