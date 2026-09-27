"""Report freshly trained/scored final-workflow models, without historical predictions."""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from CELEBA_agent3 import atomic_csv, atomic_json, load_agent3_images
from CELEBA_data import image_transform, load_yaml, sha256_file
from CELEBA_final_test import (
    prepare_rdr_bootstrap_components,
    rdr_bootstrap_row,
    summarize_percentile_intervals,
)
from CELEBA_rdr import estimate_midpoint_hellinger, estimate_phi_divergences
from CELEBA_selection import FeatureMoments, fid_from_moments
from CELEBA_plots import (
    FIGURE_SPECS,
    save_score_panel,
    deterministic_sample,
    save_score_map,
    save_score_distributions,
    save_metric_summary,
)


from .CELEBA_data import feature_values, image_shard
from .CELEBA_paths import HERE


def report(args, spec):
    """Build tables, uncertainty intervals, null diagnostics, and paper figures."""
    out = args.work_root / "report"
    out.mkdir(parents=True, exist_ok=True)
    frames, scores, estimates, inputs, metrics, phi = primary_scores(args)
    intervals = bootstrap_intervals(out, scores, estimates, metrics, phi)
    pools, fid = test_fid(args, spec, frames, out)
    null_diagnostics(args, out, inputs)
    render_figures(args, frames, pools, estimates, fid, out)
    write_report(out, intervals, fid, inputs)


def primary_scores(args):
    """Load paired feature/pixel predictions and compute the four primary estimates."""
    frames = {}
    scores = {}
    estimates = {}
    inputs = {}
    metrics = []
    phi = []
    for branch in ("lower", "upper"):
        merged = None
        for level in ("feature", "pixel"):
            path = (
                args.work_root / f"models/primary/{branch}/{level}/test_predictions.csv"
            )
            inputs[str(path)] = sha256_file(path)
            frame = pd.read_csv(path, low_memory=False)
            frame["distribution"] = np.where(frame.fold.eq("A"), "real", branch)
            if frame.groupby("fold").size().to_dict() != {"A": 18000, "B": 18000}:
                raise RuntimeError("Report requires the complete 18000/18000 test")
            if merged is None:
                merged = frame.rename(columns={"rdr": f"{level}_rdr"})
            else:
                merged = merged.merge(
                    frame[["source_id", "fold", "rdr"]].rename(
                        columns={"rdr": f"{level}_rdr"}
                    ),
                    on=["source_id", "fold"],
                    validate="one_to_one",
                )
            p = frame.loc[frame.fold.eq("A"), "rdr"].to_numpy()
            q = frame.loc[frame.fold.eq("B"), "rdr"].to_numpy()
            scores.setdefault(branch, {})[level] = {"p": p, "q": q}
            m = estimate_midpoint_hellinger(p, q, 1e-6)
            metrics.append(dict(contrast=branch, representation=level, **m))
            estimates[f"{level}_h2_real_vs_{branch}"] = m["variational_lower_bound"]
            estimates[f"{level}_plugin_real_vs_{branch}"] = m[
                "plugin_midpoint_hellinger"
            ]
            phi.append(
                estimate_phi_divergences(p, q, 1e-6).assign(
                    contrast=branch, representation=level
                )
            )
        frames[branch] = merged
        estimates[f"pixel_minus_feature_real_vs_{branch}"] = (
            estimates[f"pixel_h2_real_vs_{branch}"]
            - estimates[f"feature_h2_real_vs_{branch}"]
        )
    if not np.array_equal(
        frames["lower"].query("fold=='A'").source_id,
        frames["upper"].query("fold=='A'").source_id,
    ):
        raise RuntimeError("Real rows must align across branches")
    return frames, scores, estimates, inputs, metrics, phi


def bootstrap_intervals(out, scores, estimates, metrics, phi):
    """Write paired bootstrap intervals conditional on the fitted models."""
    components = prepare_rdr_bootstrap_components(scores, 1e-6)
    bootstrap = pd.DataFrame(
        [rdr_bootstrap_row(components, 20260913, i) for i in range(5000)]
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
    intervals = summarize_percentile_intervals(estimates, bootstrap, 0.95)
    atomic_csv(bootstrap, out / "test_rdr_bootstrap.csv")
    atomic_csv(intervals, out / "test_rdr_intervals.csv")
    atomic_csv(pd.DataFrame(metrics), out / "test_midpoint_hellinger.csv")
    atomic_csv(pd.concat(phi), out / "test_phi_divergences.csv")
    return intervals


def test_fid(args, spec, frames, out):
    """Compute FID from all test feature vectors using bounded covariance blocks."""
    pools = {
        "real": frames["lower"].query("fold=='A'").copy(),
        "lower": frames["lower"].query("fold=='B'").copy(),
        "upper": frames["upper"].query("fold=='B'").copy(),
    }
    moments = {}
    for name, frame in pools.items():
        values = feature_values(frame, args, spec)
        mean = values.mean(axis=0, dtype=np.float64)
        scatter = np.zeros((2048, 2048))
        for start in range(0, len(values), 256):
            block = values[start : start + 256].astype(np.float64) - mean
            scatter += block.T @ block
        moments[name] = FeatureMoments(mean, scatter / (len(values) - 1), len(values))
    fid = {
        b: fid_from_moments(moments["real"], moments[b]).fid for b in ("lower", "upper")
    }
    atomic_csv(
        pd.DataFrame(
            [dict(contrast=b, fid=v, n_p=18000, n_q=18000) for b, v in fid.items()]
        ),
        out / "test_fid.csv",
    )
    return pools, fid


def null_diagnostics(args, out, inputs):
    """Summarize all 25 learned-only null runs with their exact held-out counts."""
    null = []
    for branch in ("lower", "upper", "real"):
        for level in ("pixel",) if branch == "real" else ("feature", "pixel"):
            for repeat in range(5):
                path = (
                    args.work_root
                    / f"models/null/{branch}/{level}/{repeat}/test_predictions.csv"
                )
                inputs[str(path)] = sha256_file(path)
                f = pd.read_csv(path)
                expected = 9000 if branch == "real" else 5000
                if f.groupby("fold").size().to_dict() != {"A": expected, "B": expected}:
                    raise RuntimeError("Report requires complete held-out null folds")
                null.append(
                    dict(
                        branch=branch,
                        level=level,
                        repeat=repeat,
                        **estimate_midpoint_hellinger(
                            f.loc[f.fold.eq("A"), "rdr"].to_numpy(),
                            f.loc[f.fold.eq("B"), "rdr"].to_numpy(),
                            1e-6,
                        ),
                    )
                )
    atomic_csv(pd.DataFrame(null), out / "learned_only_nulls.csv")


def render_figures(args, frames, pools, estimates, fid, out):
    """Render the eight published panels with fixed score bins and sampling seeds."""
    # Select exactly the same score bins and random seed as the published renderer.
    for frame in list(frames.values()) + list(pools.values()):
        for i, row in frame[~frame.source.eq("real")].iterrows():
            new = image_shard(args.work_root, row.tensor_path)
            if new.exists():
                frame.at[i, "tensor_path"] = str(new)
    transform = image_transform(load_yaml(HERE / "configs/design.yaml"), "zero_one")
    selections = []
    for i, s in enumerate(FIGURE_SPECS):
        path = out / f"{s['prefix']}_{s['level']}_rdr_real_vs_{s['branch']}.png"
        selections.append(
            save_score_panel(
                path,
                frames[s["branch"]],
                level=s["level"],
                score_column=s["score_column"],
                branch=s["branch"],
                hellinger=estimates[f"{s['level']}_h2_real_vs_{s['branch']}"],
                maximum=40,
                seed=20260913 + 1000 * i,
                transform=transform,
                expected_rows=18000,
            )
        )
    atomic_csv(pd.concat(selections), out / "paper_example_panel_ids.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=220)
    draws = []
    for i, (name, label) in enumerate(
        (
            ("real", r"$P$: real CelebA"),
            ("lower", r"$Q_L$: generated lower"),
            ("upper", r"$Q_U$: generated upper"),
        )
    ):
        f = deterministic_sample(pools[name], 40, 20260913 + i)
        draws.append(f)
        axes[i].imshow(
            make_grid(load_agent3_images(f, transform), nrow=8, padding=2, pad_value=1)
            .permute(1, 2, 0)
            .clamp(0, 1)
        )
        axes[i].set_title(label)
        axes[i].axis("off")
    fig.suptitle("40 random test images per distribution")
    fig.tight_layout()
    fig.savefig(out / "12_random_draws_P_QL_QU_40.png", bbox_inches="tight")
    plt.close(fig)
    atomic_csv(pd.concat(draws), out / "random_draw_ids.csv")
    save_score_map(out / "05_feature_pixel_score_map.png", frames)
    save_score_distributions(out / "06_score_distributions.png", frames)
    save_metric_summary(
        out / "01_metric_hierarchy.png",
        {f"fid_real_{b}": fid[b] for b in fid},
        {
            b: {"variational_lower_bound": estimates[f"feature_h2_real_vs_{b}"]}
            for b in fid
        },
        {
            b: {"variational_lower_bound": estimates[f"pixel_h2_real_vs_{b}"]}
            for b in fid
        },
    )


def write_report(out, intervals, fid, inputs):
    """Write the linked Markdown report and input/output checksum manifest."""
    lines = [
        "# Agent 3 workflow results",
        "",
        "Training: 60000; validation: 20000; test: 18000 per distribution. Learned-only estimators; no constant-ratio substitution. Intervals are paired image bootstrap intervals conditional on these fitted models. The test includes previously inspected data. FID is a point estimate.",
        "",
        intervals.to_string(index=False),
        "",
        "FID: " + json.dumps(fid),
        "",
        "25 raw learned-only null runs are recorded in learned_only_nulls.csv.",
    ]
    for p in sorted(out.glob("*.png")):
        lines += ["", f"![{p.stem}]({p.name})"]
    (out / "agent3.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        dict(
            status="complete",
            input_sha256=inputs,
            outputs={
                p.name: sha256_file(p)
                for p in out.iterdir()
                if p.is_file() and p.name != "result.json"
            },
        ),
        out / "result.json",
    )
    print(out / "agent3.md")
