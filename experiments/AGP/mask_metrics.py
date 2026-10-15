"""Conventional source-classifier and composition checks for AGP mask samplers.

This module fits ordinary binary classifiers. It does not estimate or calibrate
RDR. ``prepare`` selects boosting iteration using train/stop only and freezes all
classifiers before ``evaluate`` uses assessment observations. All methods use
the same recipes and real observations; the historical RDR AUCs are not reused.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score


METHODS = ("bootstrap_icfm", "bernoulli_mixture_icfm", "autoregressive_icfm",
           "midasim_mask_icfm", "midasim")
LABELS = {"bootstrap_icfm": "Resampled mask + ICFM",
          "bernoulli_mixture_icfm": "Bernoulli mixture + ICFM",
          "autoregressive_icfm": "Autoregressive mask + ICFM",
          "midasim_mask_icfm": "MIDASim mask + ICFM",
          "midasim": "MIDASim", "real_real": "Real–real null"}
RECIPES = ("hgb_composition", "hgb_mask", "logistic_composition")
COLORS = dict(zip(METHODS, ("#009e73", "#e69f00", "#cc79a7", "#0072b2", "#d55e00")))


def default_config():
    return {"seed": 2026101107,
            "hgb": {"max_iter": 300, "checkpoint_every": 10,
                    "learning_rate": .05, "max_leaf_nodes": 15,
                    "max_depth": 4, "min_samples_leaf": 40,
                    "l2_regularization": 10., "max_bins": 63},
            "logistic": {"C": 1., "max_iter": 2000, "tol": 1e-6},
            "recipes": list(RECIPES), "make_plots": True}


def merged_config(config=None):
    result = default_config()
    for key, value in (config or {}).items():
        if key not in result:
            raise ValueError(f"unknown classifier configuration: {key}")
        if isinstance(result[key], dict):
            unknown = set(value)-set(result[key])
            if unknown:
                raise ValueError(f"unknown {key} configuration: {sorted(unknown)}")
            result[key].update(value)
        else:
            result[key] = value
    if not result["recipes"] or set(result["recipes"])-set(RECIPES):
        raise ValueError("recipes must be a nonempty subset of declared classifiers")
    if result["hgb"]["max_iter"] < 1 or result["hgb"]["checkpoint_every"] < 1:
        raise ValueError("positive boosting iteration/checkpoint counts required")
    return result


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+"\n")


def array_hash(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.shape).encode())
    h.update(array.dtype.str.encode())
    h.update(array.tobytes())
    return h.hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(array):
    x = np.asarray(array, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or x.shape[1] < 2:
        raise ValueError("composition array must contain at least two rows and taxa")
    if not np.isfinite(x).all() or (x < 0).any() or (x.sum(1) <= 0).any():
        raise ValueError("finite nonnegative compositions with positive row sums required")
    return x/x.sum(1, keepdims=True)


def features(array, recipe):
    x = close(array)
    mask = (x > 0).astype(np.float32)
    if recipe == "hgb_composition":
        return np.column_stack((np.sqrt(x), mask)).astype(np.float32)
    if recipe == "hgb_mask":
        return np.column_stack((mask, mask.sum(1))).astype(np.float32)
    if recipe == "logistic_composition":
        return np.sqrt(x).astype(np.float32)
    raise ValueError(f"unknown classifier recipe {recipe}")


def binary_pair(p, q):
    if p.shape[1] != q.shape[1]:
        raise ValueError("source feature dimensions disagree")
    y = np.r_[np.ones(len(p), dtype=int), np.zeros(len(q), dtype=int)]
    weights = np.r_[np.full(len(p), len(y)/(2*len(p))),
                    np.full(len(q), len(y)/(2*len(q)))]
    return np.vstack((p, q)), y, weights


def null_partition(n, seed, subjects=None):
    """Disjoint real halves; all rows of a subject stay in the same half."""
    if subjects is None:
        units = np.arange(n)
    else:
        if len(subjects) != n:
            raise ValueError("subject count disagrees with real observation count")
        units = np.asarray(subjects).astype(str)
    unique = np.unique(units)
    if len(unique) < 4:
        raise ValueError("real-real null requires at least four independent units")
    order = np.random.default_rng(seed).permutation(len(unique))
    first = np.isin(units, unique[order[:len(unique)//2]])
    a, b = np.flatnonzero(first), np.flatnonzero(~first)
    if min(len(a), len(b)) < 2:
        raise ValueError("real-real null halves must have at least two observations")
    return a, b


def fit_classifier(p_train, q_train, p_stop, q_stop, recipe, config):
    """Fit a fixed classifier recipe and choose HGB iteration on stopping data."""
    x, y, weights = binary_pair(features(p_train, recipe), features(q_train, recipe))
    stop_x, stop_y, stop_w = binary_pair(features(p_stop, recipe), features(q_stop, recipe))
    started = time.monotonic()
    if recipe.startswith("hgb_"):
        settings = copy.deepcopy(config["hgb"])
        checkpoint = int(settings.pop("checkpoint_every"))
        model = HistGradientBoostingClassifier(**settings, loss="log_loss",
                                               early_stopping=False, random_state=config["seed"])
        model.fit(x, y, sample_weight=weights)
        grid = set(range(checkpoint, settings["max_iter"]+1, checkpoint)) | {settings["max_iter"]}
        trace = [{"iteration": 0, "stop_balanced_log_loss": float(np.log(2.))}]
        best_iteration, best_loss = 0, float(np.log(2.))
        for iteration, probabilities in enumerate(model.staged_predict_proba(stop_x), 1):
            if iteration in grid:
                loss = float(log_loss(stop_y, probabilities[:, 1], sample_weight=stop_w, labels=[0, 1]))
                trace.append({"iteration": iteration, "stop_balanced_log_loss": loss})
                # A strict improvement keeps the earliest checkpoint on a tie.
                if loss < best_loss-1e-12:
                    best_iteration, best_loss = iteration, loss
        n_iter, convergence_warning = settings["max_iter"], False
    else:
        model = LogisticRegression(**config["logistic"], solver="lbfgs", random_state=config["seed"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y, sample_weight=weights)
        convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught)
        best_loss = float(log_loss(stop_y, model.predict_proba(stop_x)[:, 1], sample_weight=stop_w,
                                   labels=[0, 1]))
        n_iter = int(model.n_iter_.max())
        best_iteration, trace = None, [{"iteration": n_iter, "stop_balanced_log_loss": best_loss}]
    return {"model": model, "recipe": recipe, "selected_iteration": best_iteration,
            "stop_balanced_log_loss": best_loss, "trace": trace,
            "trained_iterations": int(n_iter), "convergence_warning": convergence_warning,
            "selection_at_cap": bool(best_iteration == config["hgb"]["max_iter"])
            if recipe.startswith("hgb_") else False,
            "fit_seconds": float(time.monotonic()-started), "feature_count": int(x.shape[1])}


def predict_classifier(fit, array):
    x = features(array, fit["recipe"])
    selected = fit["selected_iteration"]
    if selected == 0:
        return np.full(len(x), .5)
    if selected is None:
        return fit["model"].predict_proba(x)[:, 1]
    for iteration, probabilities in enumerate(fit["model"].staged_predict_proba(x), 1):
        if iteration == selected:
            return probabilities[:, 1]
    raise RuntimeError("selected boosting iteration missing from fitted model")


def _training_inputs(real, generated):
    hashes = {}
    width = None
    for name, bank in [("real", real), *generated.items()]:
        for role in ("train", "stop"):
            x = close(bank[role])
            width = x.shape[1] if width is None else width
            if x.shape[1] != width:
                raise ValueError("all sources and roles must share taxa in the same order")
            hashes[f"{name}/{role}"] = array_hash(np.asarray(bank[role]))
    return hashes


def prepare(real, generated, output, config=None, real_subjects=None):
    """Read train/stop only; persist every fit before any assessment scoring."""
    output = Path(output)
    if (output/"FREEZE.json").exists() or (output/"private").exists():
        raise FileExistsError("classifier preparation already exists")
    config = merged_config(config)
    if not generated or set(generated)-set(METHODS):
        raise ValueError("unknown or empty generator comparison")
    methods = [method for method in METHODS if method in generated]
    input_hashes = _training_inputs(real, generated)
    private = output/"private"
    private.mkdir(parents=True)
    write_json(output/"config.json", config)
    splits, subject_hashes = {}, {}
    for j, role in enumerate(("train", "stop")):
        subjects = None if real_subjects is None else real_subjects[role]
        splits[role] = null_partition(len(real[role]), config["seed"]+j, subjects)
        subject_hashes[role] = None if subjects is None else array_hash(np.asarray(subjects).astype(str))
    np.savez_compressed(private/"null_training_indices.npz",
                        **{f"{role}_{side}": index for role, pair in splits.items()
                           for side, index in zip(("p", "q"), pair)})
    records, paths = [], {}
    for method in [*methods, "real_real"]:
        if method == "real_real":
            pairs = {role: (np.asarray(real[role])[splits[role][0]], np.asarray(real[role])[splits[role][1]])
                     for role in ("train", "stop")}
        else:
            pairs = {role: (real[role], generated[method][role]) for role in ("train", "stop")}
        for recipe in config["recipes"]:
            fit = fit_classifier(*pairs["train"], *pairs["stop"], recipe, config)
            filename = f"{method}__{recipe}.joblib"
            joblib.dump(fit, private/filename, compress=3)
            paths[filename] = file_hash(private/filename)
            record = {"method": method, "recipe": recipe,
                      **{key: value for key, value in fit.items() if key not in ("model", "trace")},
                      "n_train_p": len(pairs["train"][0]), "n_train_q": len(pairs["train"][1]),
                      "n_stop_p": len(pairs["stop"][0]), "n_stop_q": len(pairs["stop"][1])}
            records.append(record)
            write_json(private/f"{method}__{recipe}_trace.json", fit["trace"])
            print(f"Classifier ready: {method}/{recipe}; iteration={fit['selected_iteration']}; "
                  f"stopping log loss={fit['stop_balanced_log_loss']:.6f}; "
                  f"seconds={fit['fit_seconds']:.1f}", flush=True)
    pd.DataFrame(records).to_csv(output/"classifier_training.csv", index=False)
    freeze = {"status": "classifiers_frozen", "no_test_data_used": True,
              "methods": methods, "recipes": config["recipes"], "n_classifiers": len(records),
              "training_inputs": input_hashes, "subject_hashes": subject_hashes,
              "null_partition_unit": "subject" if real_subjects is not None else "row",
              "source_sha256": file_hash(__file__), "config_sha256": file_hash(output/"config.json"),
              "model_hashes": paths, "null_indices_sha256": file_hash(private/"null_training_indices.npz")}
    write_json(output/"FREEZE.json", freeze)
    return freeze


def _occurrence_correlations(mask, selected):
    b = np.asarray(mask[:, selected], dtype=np.float64)
    mean = b.mean(0)
    sd = np.sqrt(mean*(1-mean))
    cov = b.T@b/len(b)-np.outer(mean, mean)
    denominator = np.outer(sd, sd)
    result = np.full_like(cov, np.nan)
    np.divide(cov, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1, 1)


def mask_novelty(reference, generated):
    """Fraction of generated exact binary patterns absent from real training."""
    ref = {row.tobytes() for row in np.packbits(reference > 0, axis=1)}
    rows = [row.tobytes() for row in np.packbits(generated > 0, axis=1)]
    return float(np.mean([row not in ref for row in rows])), len(set(rows))/len(rows)


def fidelity_metrics(p, q, training, mask_reference=None):
    p, q, training = close(p), close(q), close(training)
    mask_reference = training if mask_reference is None else close(mask_reference)
    if p.shape[1] != q.shape[1] or p.shape[1] != training.shape[1]:
        raise ValueError("fidelity inputs must share taxa")
    bp, bq, bt = p > 0, q > 0, training > 0
    rp, rq = bp.sum(1), bq.sum(1)
    hp, hq = -xlogy(p, p).sum(1), -xlogy(q, q).sum(1)
    selected = np.flatnonzero((bt.mean(0) > 0) & (bt.mean(0) < 1))
    cp, cq = _occurrence_correlations(bp, selected), _occurrence_correlations(bq, selected)
    triangle = np.triu(np.ones_like(cp, dtype=bool), k=1)
    valid = triangle & np.isfinite(cp) & np.isfinite(cq)
    novelty, unique_fraction = mask_novelty(mask_reference, q)
    reference_richness = (mask_reference > 0).sum(1)
    result = {"n_p": len(p), "n_q": len(q), "n_taxa": p.shape[1],
              "zero_fraction_p": float(1-bp.mean()), "zero_fraction_q": float(1-bq.mean()),
              "richness_mean_p": float(rp.mean()), "richness_mean_q": float(rq.mean()),
              "richness_ks": float(ks_2samp(rp, rq).statistic),
              "shannon_mean_p": float(hp.mean()), "shannon_mean_q": float(hq.mean()),
              "shannon_ks": float(ks_2samp(hp, hq).statistic),
              "prevalence_mae": float(np.abs(bp.mean(0)-bq.mean(0)).mean()),
              "mean_abundance_mae": float(np.abs(p.mean(0)-q.mean(0)).mean()),
              "occurrence_correlation_mae": float(np.abs(cp[valid]-cq[valid]).mean()) if valid.any() else None,
              "occurrence_variable_taxa_training": len(selected),
              "occurrence_pairs_possible": int(triangle.sum()),
              "occurrence_pairs_compared": int(valid.sum()),
              "constant_p_taxa_in_training_variable_set": int(np.sum(np.all(bp[:, selected] == bp[0, selected], axis=0))),
              "constant_q_taxa_in_training_variable_set": int(np.sum(np.all(bq[:, selected] == bq[0, selected], axis=0))),
              "mask_novel_fraction": novelty, "mask_unique_fraction": float(unique_fraction),
              "mask_reference_rows": len(mask_reference),
              "flow_training_richness_min": int(reference_richness.min()),
              "flow_training_richness_max": int(reference_richness.max()),
              "q_richness_below_flow_training_fraction": float(np.mean(rq < reference_richness.min())),
              "q_richness_above_flow_training_fraction": float(np.mean(rq > reference_richness.max()))}
    return result


def evaluate(real, generated, output, real_subjects=None):
    """Assess all frozen classifiers once, and create aggregate figures/report."""
    output = Path(output)
    if (output/"COMPLETE.json").exists() or (output/"summary").exists():
        raise FileExistsError("classifier assessment already exists")
    freeze = json.loads((output/"FREEZE.json").read_text())
    config = json.loads((output/"config.json").read_text())
    if file_hash(__file__) != freeze["source_sha256"] or file_hash(output/"config.json") != freeze["config_sha256"]:
        raise ValueError("frozen classifier source/configuration changed")
    if _training_inputs(real, generated) != freeze["training_inputs"]:
        raise ValueError("frozen train/stop inputs changed")
    if set(generated) != set(freeze["methods"]):
        raise ValueError("assessment generator names differ from frozen comparison")
    for name, digest in freeze["model_hashes"].items():
        if file_hash(output/"private"/name) != digest:
            raise ValueError(f"frozen classifier changed: {name}")
    if file_hash(output/"private/null_training_indices.npz") != freeze["null_indices_sha256"]:
        raise ValueError("frozen null indices changed")
    if (real_subjects is None) != (freeze["null_partition_unit"] == "row"):
        raise ValueError("null partition must retain the frozen subject/row convention")
    for role in ("train", "stop"):
        if real_subjects is not None and array_hash(np.asarray(real_subjects[role]).astype(str)) != freeze["subject_hashes"][role]:
            raise ValueError("frozen real subject identifiers changed")
    # No test array has been accessed before all of the freeze checks above.
    p = close(real["test"])
    null_subjects = None if real_subjects is None else real_subjects["test"]
    null_a, null_b = null_partition(len(p), config["seed"]+2, null_subjects)
    np.savez_compressed(output/"private/null_test_indices.npz", p=null_a, q=null_b)
    rows, fidelity = [], []
    mask_reference = np.vstack((np.asarray(real["train"]), np.asarray(real["stop"])))
    tests = {}
    for method in [*freeze["methods"], "real_real"]:
        if method == "real_real":
            pp, qq = p[null_a], p[null_b]
        else:
            pp, qq = p, close(generated[method]["test"])
            fidelity.append({"method": method, **fidelity_metrics(pp, qq, real["train"], mask_reference=mask_reference)})
            tests[method] = qq
        if pp.shape[1] != qq.shape[1] or pp.shape[1] != np.asarray(real["train"]).shape[1]:
            raise ValueError("assessment taxa differ from training")
        for recipe in freeze["recipes"]:
            fit = joblib.load(output/"private"/f"{method}__{recipe}.joblib")
            pr_p, pr_q = predict_classifier(fit, pp), predict_classifier(fit, qq)
            probabilities = np.r_[pr_p, pr_q]
            _, y, weights = binary_pair(pp, qq)
            auc = float(roc_auc_score(y, probabilities))
            rows.append({"method": method, "recipe": recipe, "auc": auc,
                         "balanced_log_loss": float(log_loss(y, probabilities, sample_weight=weights, labels=[0, 1])),
                         "stop_balanced_log_loss": fit["stop_balanced_log_loss"],
                         "selected_iteration": fit["selected_iteration"], "selection_at_cap": fit["selection_at_cap"],
                         "convergence_warning": fit["convergence_warning"], "n_p": len(pp), "n_q": len(qq)})
            np.savez_compressed(output/"private"/f"{method}__{recipe}_test_predictions.npz", p=pr_p, q=pr_q)
    summary = output/"summary"
    summary.mkdir()
    classifiers = pd.DataFrame(rows)
    compositions = pd.DataFrame(fidelity)
    classifiers.to_csv(summary/"classifier_metrics.csv", index=False)
    compositions.to_csv(summary/"fidelity_metrics.csv", index=False)
    comparison = compositions.merge(classifiers.pivot(index="method", columns="recipe", values="auc").add_suffix("_auc"),
                                    left_on="method", right_index=True)
    comparison.to_csv(summary/"comparison.csv", index=False)
    checks = {}
    if {"midasim_mask_icfm", "midasim"}.issubset(tests):
        for role in ("train", "stop", "test"):
            a, b = np.asarray(generated["midasim_mask_icfm"][role]) > 0, np.asarray(generated["midasim"][role]) > 0
            checks[f"midasim_hybrid_{role}_masks_identical"] = bool(a.shape == b.shape and np.array_equal(a, b))
            if not checks[f"midasim_hybrid_{role}_masks_identical"]:
                raise ValueError(f"current hybrid no longer shares the MIDASim {role} masks")
        if "hgb_mask" in freeze["recipes"]:
            paths = [output/"private"/f"{method}__hgb_mask_test_predictions.npz"
                     for method in ("midasim_mask_icfm", "midasim")]
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                checks["midasim_hybrid_mask_classifier_predictions_identical"] = all(np.array_equal(a[key], b[key]) for key in ("p", "q"))
            if not checks["midasim_hybrid_mask_classifier_predictions_identical"]:
                raise AssertionError("identical masks produced different fixed-classifier predictions")
    if config["make_plots"]:
        make_figure(summary, classifiers, p, tests)
    make_report(summary, classifiers, compositions, config, freeze, checks)
    result = {"status": "complete", "n_classifiers": len(rows), "methods": freeze["methods"],
              "test_input_hashes": {"real": array_hash(np.asarray(real["test"])),
                                    **{m: array_hash(np.asarray(generated[m]["test"])) for m in freeze["methods"]}},
              "test_subjects_sha256": None if null_subjects is None else array_hash(np.asarray(null_subjects).astype(str)),
              "checks": checks, "rdr_fitted_or_evaluated": False,
              "selected_at_boosting_cap": int(classifiers.selection_at_cap.sum()),
              "logistic_convergence_warnings": int(classifiers.convergence_warning.sum()),
              "summary_hashes": {path.name: file_hash(path) for path in sorted(summary.iterdir())}}
    write_json(output/"COMPLETE.json", result)
    return result


def make_figure(folder, classifiers, p, tests):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/agp-mask-metrics-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    methods = list(tests)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), constrained_layout=True)
    for col, recipe, title in ((0, "hgb_composition", "A  Composition source classification"),
                               (1, "hgb_mask", "B  Mask source classification")):
        ax = axes[0, col]
        subset = classifiers[classifiers.recipe.eq(recipe)].set_index("method")
        if len(subset):
            for i, method in enumerate(methods):
                auc = subset.loc[method, "auc"]
                ax.barh(i, auc, color=COLORS[method], height=.65)
                ax.text(min(auc+.012, .99), i, f"{auc:.3f}", va="center", ha="left" if auc < .94 else "right")
            ax.axvline(.5, ls="--", color=".4", lw=1, label="Chance")
            if "real_real" in subset.index:
                ax.axvline(subset.loc["real_real", "auc"], color=".3", ls=":", lw=1.5, label="Real–real null")
        ax.set(yticks=np.arange(len(methods)), yticklabels=[LABELS[m] for m in methods], xlim=(0, 1.055),
               xlabel="Held-out AUC (higher = more separable)", title=title)
        ax.invert_yaxis()
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    for col, kind, title in ((0, "richness", "C  Taxon richness"), (1, "shannon", "D  Shannon diversity")):
        ax = axes[1, col]
        for method, values in [("real", p), *tests.items()]:
            x = np.sort((values > 0).sum(1) if kind == "richness" else -xlogy(values, values).sum(1))
            ax.step(x, np.arange(1, len(x)+1)/len(x), where="post", label="Observed P" if method == "real" else LABELS[method],
                    color="black" if method == "real" else COLORS[method], lw=1.7 if method == "real" else 1.2,
                    ls="--" if method == "midasim_mask_icfm" else "-")
        ax.set(xlabel="Number of present taxa" if kind == "richness" else "Shannon entropy (natural log)",
               ylabel="Empirical CDF", title=title, ylim=(0, 1.02))
        ax.grid(alpha=.15)
        if col == 1:
            ax.legend(frameon=False, fontsize=8)
    fig.savefig(folder/"mask_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(folder/"mask_comparison.pdf", bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def make_report(folder, classifiers, fidelity, config, freeze, checks):
    def fmt(value, digits=4):
        return "—" if value is None or pd.isna(value) else f"{value:.{digits}f}"
    auc = classifiers.pivot(index="method", columns="recipe", values="auc")
    rows = ["| Method | Composition AUC | Mask AUC | Logistic AUC | Zero fraction | Richness | Shannon |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    first = fidelity.iloc[0]
    rows.append(f"| Observed P | — | — | — | {first.zero_fraction_p:.4f} | {first.richness_mean_p:.2f} | {first.shannon_mean_p:.3f} |")
    for row in fidelity.itertuples():
        a = auc.loc[row.method]
        rows.append(f"| {LABELS[row.method]} | {fmt(a.get('hgb_composition'))} | {fmt(a.get('hgb_mask'))} | "
                    f"{fmt(a.get('logistic_composition'))} | {row.zero_fraction_q:.4f} | {row.richness_mean_q:.2f} | {row.shannon_mean_q:.3f} |")
    a = auc.loc["real_real"]
    rows.append(f"| Real–real null | {fmt(a.get('hgb_composition'))} | {fmt(a.get('hgb_mask'))} | {fmt(a.get('logistic_composition'))} | — | — | — |")
    discrepancy = ["| Method | Prevalence MAE | Mean abundance MAE | Richness KS | Shannon KS | Occurrence correlation MAE | Novel masks |",
                   "|---|---:|---:|---:|---:|---:|---:|"]
    for row in fidelity.itertuples():
        discrepancy.append(f"| {LABELS[row.method]} | {row.prevalence_mae:.5f} | {row.mean_abundance_mae:.6f} | {row.richness_ks:.4f} | "
                           f"{row.shannon_ks:.4f} | {fmt(row.occurrence_correlation_mae, 5)} | {row.mask_novel_fraction:.3%} |")
    text = ["# AGP mask-sampler comparison", "",
            "Five generator variants are assessed with ordinary held-out real-versus-generated source classification and simple composition statistics. "
            "No RDR model, RDR loss, density-ratio score, or calibration diagnostic is fitted or evaluated in this comparison.", "",
            *rows, "", "![Mask comparison](mask_comparison.png)" if config["make_plots"] else "", "", *discrepancy, "",
            "## Assessment protocol", "",
            "AUC uses real=1 and generated=0; 0.5 denotes chance discrimination, and larger values mean that this fitted classifier separates the sources more easily. "
            "We retain AUC below 0.5 without flipping labels or selecting test direction. Low AUC alone does not establish distributional equivalence or overall generator quality.", "",
            "Composition HGB uses the square root of closed abundances concatenated with their binary presence masks. Mask HGB uses the binary masks plus richness. "
            "The sensitivity classifier is L2 logistic regression on square-root composition, without feature standardization. "
            "All sources have equal total training weight; stopping log loss is also equally weighted by source. "
            f"Boosting fits at most {config['hgb']['max_iter']} iterations, checking stopping loss every {config['hgb']['checkpoint_every']} iterations and at the cap, "
            "including the constant probability 0.5 candidate. The earliest minimum is retained, treating loss differences within 1e-12 as ties. No assessment labels select a model or iteration. "
            "The same recipe and fixed seed are used for every pair, with one fitted classifier per recipe/pair; these are new conventional classifier AUCs, not historical neural RDR AUCs.", "",
            f"All {freeze['n_classifiers']} classifiers were frozen before assessment scoring. "
            f"The real–real null divides each role's real {freeze['null_partition_unit']} units into two disjoint halves. "
            "It has fewer observations per source than the generator comparisons, so it is a sanity check rather than a sample-size-matched performance bound. "
            "The reported values are descriptive for these fixed generator fits, classifier fits, and assessment samples; no repeated-run or population uncertainty is claimed.", "",
            "Taxon prevalence and mean abundance MAE average absolute differences over all taxa. Richness and Shannon KS are two-sample empirical CDF distances; "
            "Shannon uses natural logarithms and zero contributes zero. Occurrence correlations use only taxa with nonconstant presence in real classifier training. "
            "A pair is compared only when both taxa vary in each assessment source; the CSV records possible/compared pair counts and constant-taxon counts, "
            "because prevalence errors and lost variation must not be hidden by this exclusion. Novelty is the fraction of generated assessment masks absent "
            "from the union of real train and stopping observations, the full 10,443-observation generator-training bank in the AGP study. "
            "The CSV also records generated richness below/above the observed generator-training range; matching this range does not guarantee that a new mask is well supported by the abundance flow.", "",
            "The existing hybrid and MIDASim reuse exactly the same masks in every role; their mask AUC, richness distribution, prevalence, and occurrence-correlation statistics "
            "therefore coincide by construction. The four ICFM variants must be interpreted conditional on the common frozen abundance flow; the full MIDASim comparison changes the abundance generator as well.", "",
            f"Boosting selections at the iteration cap: {int(classifiers.selection_at_cap.sum())}. "
            f"Logistic convergence warnings: {int(classifiers.convergence_warning.sum())}. "
            "These are reported explicitly rather than treated as proof that optimization or statistical power is exhausted.", "",
            "Machine-readable details: [classifier metrics](classifier_metrics.csv), [composition metrics](fidelity_metrics.csv), [combined table](comparison.csv).", ""]
    (folder/"report.md").write_text("\n".join(text))


def assess(real, generated, output, config=None, real_subjects=None):
    prepare(real, generated, output, config=config, real_subjects=real_subjects)
    return evaluate(real, generated, output, real_subjects=real_subjects)
