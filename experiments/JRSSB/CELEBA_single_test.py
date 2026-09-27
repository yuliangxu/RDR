#!/usr/bin/env python3
"""Reproduce Agent 3 on one 18k held-out test split, using learned checkpoints."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/agent3-single-test-{os.getpid()}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

from CELEBA_plots import FIGURE_SPECS, save_score_panel
from CELEBA_plots import deterministic_sample
from CELEBA_null_data import learned_null_specs, load_row
from CELEBA_plots import save_metric_summary, save_score_map, save_score_distributions
from CELEBA_agent3 import load_agent3_images, manifest_digest
from CELEBA_data import image_transform, load_yaml, sha256_file
from CELEBA_final_test import (
    prepare_rdr_bootstrap_components,
    rdr_bootstrap_row,
    summarize_percentile_intervals,
)
from CELEBA_rdr import (
    estimate_midpoint_hellinger,
    estimate_phi_divergences,
    summarize_ratio_scores,
)
from CELEBA_selection import FeatureMoments, fid_from_moments

HERE = Path(__file__).resolve().parent
ROOT = Path("/cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to SOURCE_ROOT/rdr/real_generator/single_test under /cwork.",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    require(args.bootstrap_repeats > 0, "Positive bootstrap count required")
    root = args.source_root
    out = args.output_dir or root / "rdr/real_generator/single_test"
    if not str(out.resolve()).startswith("/cwork/"):
        raise ValueError("Scientific outputs must be saved under /cwork")
    out.mkdir(parents=True, exist_ok=True)
    inputs = {}

    def verified(path, expected=None):
        path = Path(path)
        digest = sha256_file(path)
        require(expected is None or digest == expected, f"Input hash mismatch: {path}")
        inputs[str(path)] = digest
        return path

    def read_json(path, expected=None):
        return json.loads(verified(path, expected).read_text())

    def read_csv(path, expected=None):
        return pd.read_csv(verified(path, expected))

    final = read_json(root / "locks/agent3_final_test_result.json")
    require(final["status"] == "complete", "Original test incomplete")
    final_selection = read_json(
        final["final_test_selection_path"], final["final_test_selection_sha256"]
    )
    old_selection = read_json(
        final_selection["design_selection_lock_path"],
        final_selection["design_selection_lock_sha256"],
    )
    source_items = {
        "design": old_selection["selection"]["design"],
        "test": final_selection["selection"],
    }
    pools, audit, training = {}, [], []
    for name in ("real", "lower", "upper"):
        parts = []
        for origin in ("design", "test"):
            item = source_items[origin][name]
            part = read_csv(
                item["selected_manifest_path"], item["selected_manifest_sha256"]
            )
            require(
                len(part) == 9000 and part.role.eq(origin).all(),
                "Unexpected source role/count",
            )
            parts.append(
                part.assign(original_role=origin, role="test", distribution=name)
            )
        combined = pd.concat(parts, ignore_index=True)
        require(not combined.source_id.duplicated().any(), "Repeated test image")
        if name == "real":
            require(
                not (set(parts[0].identity) & set(parts[1].identity)),
                "Old roles share real identities",
            )
        else:
            require(
                not combined.latent_seed.duplicated().any(), "Repeated test latent seed"
            )
        for role in ("train", "validation"):
            item = old_selection["selection"][role][name]
            prior = read_csv(
                item["selected_manifest_path"], item["selected_manifest_sha256"]
            )
            id_overlap = len(set(prior.source_id) & set(combined.source_id))
            key = "identity" if name == "real" else "latent_seed"
            key_overlap = len(set(prior[key].dropna()) & set(combined[key].dropna()))
            require(id_overlap == key_overlap == 0, f"Overlap with {role}: {name}")
            audit.append(
                dict(
                    distribution=name,
                    prior_role=role,
                    n_prior=len(prior),
                    n_test=len(combined),
                    image_overlap=id_overlap,
                    **{f"{key}_overlap": key_overlap},
                )
            )
        combined.to_csv(out / f"test_{name}_manifest.csv", index=False)
        pools[name] = combined

    frames, scores, rows, phi, summaries = {}, {}, [], [], []
    for branch in ("lower", "upper"):
        design_parts = []
        for level in ("feature", "pixel"):
            lock = read_json(
                root / f"locks/real_vs_{branch}_{level}_result.json",
                final["model_result_lock_sha256"][f"{level}_{branch}"],
            )
            verified(lock["checkpoint_path"], lock["checkpoint_sha256"])
            require(
                lock["restore_mode"] == "best_validation" and lock["best_epoch"] > 0,
                "Checkpoint is not learned",
            )
            history = read_csv(
                Path(lock["checkpoint_path"]).parent / "loss_history.csv"
            )
            require(
                history.iloc[0].validation_loss < 0,
                "Baseline may have changed early stopping",
            )
            require(
                int(history.loc[history.validation_loss.idxmin(), "epoch"])
                == lock["best_epoch"],
                "Best learned epoch differs",
            )
            training.append(
                dict(
                    branch=branch,
                    representation=level,
                    first_validation_loss=float(history.iloc[0].validation_loss),
                    best_epoch=lock["best_epoch"],
                    best_validation_loss=lock["best_validation_loss"],
                    baseline_changes_selection_or_stopping=False,
                    checkpoint_path=lock["checkpoint_path"],
                    checkpoint_sha256=lock["checkpoint_sha256"],
                )
            )
            pred = read_csv(lock["prediction_path"], lock["prediction_sha256"])
            pred = pred[["source", "source_id", "distribution", "rdr"]].rename(
                columns={"rdr": f"{level}_rdr"}
            )
            design_parts.append(pred)
        design_scores = design_parts[0].merge(
            design_parts[1],
            on=["source", "source_id", "distribution"],
            validate="one_to_one",
            how="outer",
        )
        require(len(design_scores) == 18000, "Design score join lost rows")
        item = final["outputs"][f"predictions_{branch}"]
        test_scores = read_csv(item["path"], item["sha256"])
        predictions = pd.concat(
            [
                design_scores.assign(original_role="design"),
                test_scores.assign(original_role="test"),
            ],
            ignore_index=True,
        )
        keys = ["source", "source_id", "distribution", "original_role"]
        manifests = pd.concat([pools["real"], pools[branch]], ignore_index=True)
        frame = manifests.merge(
            predictions[keys + ["feature_rdr", "pixel_rdr"]],
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        require(
            len(frame) == 36000 and frame["_merge"].eq("both").all(),
            "Test manifest/score mismatch",
        )
        frame = frame.drop(columns="_merge")
        require(
            np.isfinite(frame[["feature_rdr", "pixel_rdr"]]).all().all(),
            "Nonfinite scores",
        )
        frame.to_csv(out / f"test_scores_real_vs_{branch}.csv", index=False)
        frames[branch], scores[branch] = frame, {}
        for level in ("feature", "pixel"):
            p = frame.loc[frame.distribution.eq("real"), f"{level}_rdr"].to_numpy()
            q = frame.loc[frame.distribution.eq(branch), f"{level}_rdr"].to_numpy()
            require(len(p) == len(q) == 18000, "Unequal evaluation counts")
            scores[branch][level] = {"p": p, "q": q}
            estimate = estimate_midpoint_hellinger(p, q, 1e-6)
            # Equal-sized union must reproduce the mean of the two source-role estimates.
            role_estimates = []
            for origin in ("design", "test"):
                sub = frame[frame.original_role.eq(origin)]
                role_estimates.append(
                    estimate_midpoint_hellinger(
                        sub.loc[sub.distribution.eq("real"), f"{level}_rdr"].to_numpy(),
                        sub.loc[sub.distribution.eq(branch), f"{level}_rdr"].to_numpy(),
                        1e-6,
                    )["variational_lower_bound"]
                )
            require(
                abs(estimate["variational_lower_bound"] - np.mean(role_estimates))
                < 1e-12,
                "Pooled estimator accounting failed",
            )
            rows.append(
                dict(
                    contrast=f"real_vs_{branch}",
                    representation=level,
                    evaluation_role="test",
                    **estimate,
                )
            )
            phi.append(
                estimate_phi_divergences(p, q, 1e-6).assign(
                    contrast=f"real_vs_{branch}", representation=level
                )
            )
            summaries.append(
                summarize_ratio_scores(p, q).assign(
                    contrast=f"real_vs_{branch}", representation=level
                )
            )
    require(
        np.array_equal(
            frames["lower"].loc[frames["lower"].distribution.eq("real"), "source_id"],
            frames["upper"].loc[frames["upper"].distribution.eq("real"), "source_id"],
        ),
        "Unpaired real scores",
    )
    midpoint = pd.DataFrame(rows)
    midpoint.to_csv(out / "test_midpoint_hellinger.csv", index=False)
    pd.concat(phi).to_csv(out / "test_phi_divergences.csv", index=False)
    pd.concat(summaries).to_csv(out / "test_score_summary.csv", index=False)
    pd.DataFrame(training).to_csv(out / "checkpoint_audit.csv", index=False)
    pd.DataFrame(audit).to_csv(out / "split_audit.csv", index=False)
    estimates = {}
    for branch in ("lower", "upper"):
        for level in ("feature", "pixel"):
            row = midpoint[
                (midpoint.contrast == f"real_vs_{branch}")
                & (midpoint.representation == level)
            ].iloc[0]
            estimates[f"{level}_h2_real_vs_{branch}"] = float(
                row.variational_lower_bound
            )
            estimates[f"{level}_plugin_real_vs_{branch}"] = float(
                row.plugin_midpoint_hellinger
            )
        estimates[f"pixel_minus_feature_real_vs_{branch}"] = (
            estimates[f"pixel_h2_real_vs_{branch}"]
            - estimates[f"feature_h2_real_vs_{branch}"]
        )
    for level in ("feature", "pixel"):
        estimates[f"{level}_h2_lower_minus_upper"] = (
            estimates[f"{level}_h2_real_vs_lower"]
            - estimates[f"{level}_h2_real_vs_upper"]
        )
    estimates["increment_lower_minus_upper"] = (
        estimates["pixel_minus_feature_real_vs_lower"]
        - estimates["pixel_minus_feature_real_vs_upper"]
    )
    components = prepare_rdr_bootstrap_components(scores, 1e-6)
    boot = pd.DataFrame(
        [
            rdr_bootstrap_row(components, args.seed, i)
            for i in range(args.bootstrap_repeats)
        ]
    )
    boot.to_csv(out / "test_rdr_bootstrap.csv", index=False)
    intervals = summarize_percentile_intervals(estimates, boot, 0.95)
    intervals.to_csv(out / "test_rdr_intervals.csv", index=False)
    print("Pooled scores, checkpoint audit, and RDR bootstrap complete", flush=True)

    fid = compute_test_fid(source_items, pools, read_csv, verified, out)
    summarize_nulls(root, verified, out)
    render_figures(frames, pools, estimates, fid, args, verified, out)
    summary_rows = write_report(intervals, fid, args, estimates, out)

    script_paths = [
        Path(__file__),
        HERE / "CELEBA_plots.py",
        HERE / "CELEBA_rdr.py",
        HERE / "CELEBA_final_test.py",
        HERE / "CELEBA_selection.py",
    ]
    result = dict(
        status="complete",
        analysis="single_heldout_test_learned_only",
        n_train_per_distribution=60000,
        n_validation_per_distribution=20000,
        n_test_per_distribution=18000,
        retrained=False,
        stabilization=False,
        numerical_ratio_epsilon=1e-6,
        historical_test_union=["design", "test"],
        bootstrap_repeats=args.bootstrap_repeats,
        bootstrap_seed=args.seed,
        fid_bootstrap_repeats=0,
        input_sha256=inputs,
        script_sha256={str(p): sha256_file(p) for p in script_paths},
        outputs={
            p.name: sha256_file(p)
            for p in sorted(out.iterdir())
            if p.is_file() and p.name != "result.json"
        },
    )
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    print(out / "agent3_single_test_report.md", flush=True)


def compute_test_fid(source_items, pools, read_csv, verified, out):
    """Recompute pooled feature moments and FID with bounded memory."""
    # Recompute FID from the union of the features, never average split FIDs.
    moments = {}
    for name in ("real", "lower", "upper"):
        part_moments = []
        for origin in ("design", "test"):
            item = source_items[origin][name]
            full = read_csv(
                item["input_manifest_path"], item.get("input_manifest_sha256")
            )
            require(
                manifest_digest(full) == item["input_manifest_fingerprint"],
                "Feature input manifest mismatch",
            )
            path = verified(item["feature_cache_path"], item["feature_cache_sha256"])
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require(payload["backend"] == "pytorch_fid_0.3.0", "Wrong feature backend")
            features = payload.get("features", payload.get("z"))
            require(
                tuple(features.shape) == (len(full), 2048), "Wrong feature dimensions"
            )
            selected = pools[name][pools[name].original_role.eq(origin)]
            indices = selected.feature_row.to_numpy(dtype=int)
            mean = np.zeros(2048, dtype=np.float64)
            for start in range(0, len(indices), 256):
                block = (
                    features[indices[start : start + 256]].numpy().astype(np.float64)
                )
                mean += block.sum(axis=0)
            mean /= len(indices)
            scatter = np.zeros((2048, 2048), dtype=np.float64)
            for start in range(0, len(indices), 256):
                block = (
                    features[indices[start : start + 256]].numpy().astype(np.float64)
                    - mean
                )
                scatter += block.T @ block
            part_moments.append(
                FeatureMoments(mean, scatter / (len(indices) - 1), len(indices))
            )
            del features, payload, block, scatter
        a, b = part_moments
        n = a.sample_size + b.sample_size
        difference = a.mean - b.mean
        covariance = (
            (a.sample_size - 1) * a.covariance
            + (b.sample_size - 1) * b.covariance
            + a.sample_size * b.sample_size / n * np.outer(difference, difference)
        ) / (n - 1)
        moments[name] = FeatureMoments(
            (a.sample_size * a.mean + b.sample_size * b.mean) / n, covariance, n
        )
        print(f"Computed pooled {name} feature moments", flush=True)
    fid_rows = []
    for p, q in (("real", "lower"), ("real", "upper"), ("lower", "upper")):
        value = fid_from_moments(moments[p], moments[q])
        fid_rows.append(
            dict(
                comparison=f"{p}_vs_{q}",
                fid=value.fid,
                mean_term=value.mean_term,
                covariance_term=value.covariance_term,
                n_p=18000,
                n_q=18000,
            )
        )
        print(f"FID {p} vs {q}: {value.fid:.6f}", flush=True)
    fid = pd.DataFrame(fid_rows)
    fid.to_csv(out / "test_fid.csv", index=False)

    return fid


def summarize_nulls(root, verified, out):
    """Evaluate the 25 archived learned null fits and record their input hashes."""
    # Auxiliary nulls retain learned predictions, with their own fold accounting.
    null_rows = []
    specs = learned_null_specs(
        root, Path("/cwork/yx306/RDR/JRSSB/real_ddim_fid_matched")
    )
    for spec in specs:
        predictions, _ = load_row(spec)
        for repeat, frame in predictions.items():
            verified(spec["root"] / f"repeat_{repeat:02d}" / spec["prediction_name"])
            p = frame.loc[frame[spec["fold_column"]].eq("A"), "rdr"].to_numpy()
            q = frame.loc[frame[spec["fold_column"]].eq("B"), "rdr"].to_numpy()
            null_rows.append(
                dict(
                    comparison=spec["label"],
                    repeat=repeat,
                    estimator="learned_only",
                    **estimate_midpoint_hellinger(p, q, 1e-6),
                )
            )
    pd.DataFrame(null_rows).to_csv(out / "learned_only_nulls.csv", index=False)


def render_figures(frames, pools, estimates, fid, args, verified, out):
    """Render the eight paper figures and save the exact selected image IDs."""
    config_path = verified(HERE / "configs/design.yaml")
    transform = image_transform(load_yaml(config_path), pixel_range="zero_one")
    manifests = []
    for i, spec in enumerate(FIGURE_SPECS):
        name = f"{spec['prefix']}_{spec['level']}_rdr_real_vs_{spec['branch']}.png"
        chosen = save_score_panel(
            out / name,
            frames[spec["branch"]],
            level=spec["level"],
            score_column=spec["score_column"],
            branch=spec["branch"],
            hellinger=estimates[f"{spec['level']}_h2_real_vs_{spec['branch']}"],
            maximum=40,
            seed=args.seed + 1000 * i,
            transform=transform,
            expected_rows=18000,
        )
        chosen["paper_level"] = spec["level"]
        manifests.append(chosen)
    pd.concat(manifests).to_csv(out / "paper_example_panel_ids.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=220)
    random = []
    for i, (name, label) in enumerate(
        (
            ("real", r"$P$: real CelebA"),
            ("lower", r"$Q_L$: generated lower"),
            ("upper", r"$Q_U$: generated upper"),
        )
    ):
        frame = deterministic_sample(pools[name], 40, args.seed + i)
        images = load_agent3_images(frame, transform)
        axes[i].imshow(
            make_grid(images, nrow=8, padding=2, pad_value=1)
            .permute(1, 2, 0)
            .clamp(0, 1)
        )
        axes[i].set_title(label)
        axes[i].axis("off")
        random.append(frame.assign(draw=np.arange(1, 41), sampling_seed=args.seed + i))
    fig.suptitle("40 random test images per distribution")
    fig.tight_layout()
    fig.savefig(out / "12_random_draws_P_QL_QU_40.png", bbox_inches="tight")
    plt.close(fig)
    pd.concat(random).to_csv(out / "random_draw_ids.csv", index=False)
    save_score_map(out / "05_feature_pixel_score_map.png", frames)
    save_score_distributions(out / "06_score_distributions.png", frames)
    fid_values = {r.comparison: r.fid for r in fid.itertuples()}
    save_metric_summary(
        out / "01_metric_hierarchy.png",
        {f"fid_real_{b}": fid_values[f"real_vs_{b}"] for b in ("lower", "upper")},
        {
            b: {"variational_lower_bound": estimates[f"feature_h2_real_vs_{b}"]}
            for b in ("lower", "upper")
        },
        {
            b: {"variational_lower_bound": estimates[f"pixel_h2_real_vs_{b}"]}
            for b in ("lower", "upper")
        },
    )


def write_report(intervals, fid, args, estimates, out):
    """Write the scientific interpretation, linked figures, and summary table."""
    fid_values = {r.comparison: r.fid for r in fid.itertuples()}

    def interval(key):
        r = intervals[intervals.metric.eq(key)].iloc[0]
        return f"{r.estimate:.6f} [{r.interval_low:.6f}, {r.interval_high:.6f}]"

    report = [
        "# Agent 3: one held-out test split, learned estimators",
        "",
        "The primary experiment has three roles: 60,000 training, 20,000 validation, and 18,000 test images per distribution. The test split is the complete union of the previous 9,000 design and 9,000 final-test rows, with no score-based filtering. The same real test images are used in both generator contrasts.",
        "",
        "Training and validation remain unchanged. All four first-epoch validation losses are negative; the learned minimum and stopping trajectory are therefore identical with or without the constant-ratio baseline. This reproduction reuses the identical hashed learned checkpoints and their saved predictions. It applies no constant-ratio substitution and no stabilization-based null gate.",
        "",
        "The single test split is disjoint from training and validation in image IDs, real identities, and generated seeds. Its historical source role is retained only for provenance. These data have previously been inspected: this is a retrospective held-out reanalysis, not a newly sealed confirmatory experiment.",
        "",
        "## Estimator and uncertainty",
        "",
        r"For each branch, $M=(P+Q)/2$ and $r=dP/dM\in[0,2]$. The primary estimate is $\widehat H^2=1-\tfrac12\overline{r_P^{-1/2}}-\tfrac14\overline{\sqrt{r_P}}-\tfrac14\overline{\sqrt{r_Q}}$, with the original numerical clipping epsilon $10^{-6}$. Each mean uses all 18,000 observations of its source. Numerical clipping is distinct from replacing a learned model by $r=1$; it is retained to evaluate the original loss at saturated outputs.",
        "",
        f"Intervals use {args.bootstrap_repeats:,} paired image-level percentile bootstrap replicates, sharing real resamples across branches and image resamples across feature/pixel scores. They condition on the fitted models and do not include training variability or within-identity clustering. FID is recomputed from all 18,000 pool3 vectors per source; it is not the average of previous FIDs. FID is reported as a point estimate without a new equivalence test or bootstrap interval.",
        "",
        "| Contrast | Test P / Q | FID | Feature H2 [95% interval] | Pixel H2 [95% interval] | Pixel minus feature [95% interval] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary_rows = []
    for b in ("lower", "upper"):
        report.append(
            f"| Real vs {b} | 18,000 / 18,000 | {fid_values[f'real_vs_{b}']:.6f} | {interval(f'feature_h2_real_vs_{b}')} | {interval(f'pixel_h2_real_vs_{b}')} | {interval(f'pixel_minus_feature_real_vs_{b}')} |"
        )
        summary_rows.append(
            dict(
                contrast=f"real_vs_{b}",
                n_test_p=18000,
                n_test_q=18000,
                fid=fid_values[f"real_vs_{b}"],
                feature_h2=estimates[f"feature_h2_real_vs_{b}"],
                pixel_h2=estimates[f"pixel_h2_real_vs_{b}"],
                pixel_minus_feature=estimates[f"pixel_minus_feature_real_vs_{b}"],
            )
        )
    report += [
        "",
        f"FID gap (lower minus upper): {fid_values['real_vs_lower']-fid_values['real_vs_upper']:.6f}. This is descriptive, not evidence of formal FID equivalence.",
        "",
        f"Lower-minus-upper feature H2: {interval('feature_h2_lower_minus_upper')}; pixel H2: {interval('pixel_h2_lower_minus_upper')}; increment difference: {interval('increment_lower_minus_upper')}.",
        "",
        "The feature/pixel differences are diagnostics from separately fitted estimators, not exact conditional-divergence decompositions. Population information ordering does not guarantee ordering of finite fitted estimates.",
        "",
        "## Learned-only null diagnostics",
        "",
        "All 25 previously completed auxiliary null fits are recomputed from their learned scores in [learned_only_nulls.csv](learned_only_nulls.csv). They are separate same-source experiments, not another primary evaluation split. Generated nulls have 5,000 observations per held-out fold; real pixel nulls have 9,000 per fold. No exact-zero substitution or calibration-pass claim is applied. The real-pixel learned-only null discrepancy remains visible; these diagnostics do not establish calibrated neural divergence estimation.",
        "",
        "## Test figures",
        "",
        "Random images are sampled uniformly; score-bin examples use up to 40 images nearest 0, 1, or 2 within the fixed score intervals. They are descriptive examples, not prevalence estimates.",
        "",
    ]
    for filename in (
        ["01_metric_hierarchy.png", "12_random_draws_P_QL_QU_40.png"]
        + [
            f"{s['prefix']}_{s['level']}_rdr_real_vs_{s['branch']}.png"
            for s in FIGURE_SPECS
        ]
        + ["05_feature_pixel_score_map.png", "06_score_distributions.png"]
    ):
        report += [f"![{filename}]({filename})", ""]
    report += [
        "## Reproduction",
        "",
        "Run `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 experiments/JRSSB/CELEBA_single_test.py` from the repository root. The output directory contains test manifests, predictions, bootstrap replicates, exact thumbnail selections, checkpoint/split audits, and input/output hashes in `result.json`.",
        "",
    ]
    (out / "agent3_single_test_report.md").write_text("\n".join(report))
    pd.DataFrame(summary_rows).to_csv(out / "test_summary.csv", index=False)
    return summary_rows


if __name__ == "__main__":
    main()
