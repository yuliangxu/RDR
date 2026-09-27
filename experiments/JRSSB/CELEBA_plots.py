"""Final Agent 3 figures and deterministic thumbnail selection."""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/agent3-matplotlib")
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from CELEBA_agent3 import load_agent3_images, PAIR_BRANCHES
MAXIMUM_RATIO = 2.0

GRID_COLUMNS = 10

GRID_ROWS = 4

BIN_SPECS = (
    {"target": 0.0, "low": 0.0, "high": 0.5, "include_high": False},
    {"target": 1.0, "low": 0.5, "high": 1.5, "include_high": False},
    {"target": 2.0, "low": 1.5, "high": MAXIMUM_RATIO, "include_high": True},
)

FIGURE_SPECS = (
    {
        "prefix": "08",
        "level": "pixel",
        "score_column": "pixel_rdr",
        "branch": "upper",
    },
    {
        "prefix": "09",
        "level": "pixel",
        "score_column": "pixel_rdr",
        "branch": "lower",
    },
    {
        "prefix": "10",
        "level": "feature",
        "score_column": "feature_rdr",
        "branch": "upper",
    },
    {
        "prefix": "11",
        "level": "feature",
        "score_column": "feature_rdr",
        "branch": "lower",
    },
)

def format_score(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3:
        return f"{value:.2e}"
    return f"{value:.3g}"

def interval_label(spec: dict) -> str:
    right = "]" if spec["include_high"] else ")"
    return f"[{spec['low']:g}, {spec['high']:g}{right}"

def deterministic_tie_sample(
    frame: pd.DataFrame,
    count: int,
    seed: int,
) -> pd.DataFrame:
    if count >= len(frame):
        return frame.copy()
    rng = np.random.default_rng(int(seed))
    positions = np.sort(rng.choice(len(frame), size=int(count), replace=False))
    return frame.iloc[positions].copy()

def select_nearest_in_bin(
    frame: pd.DataFrame,
    score_column: str,
    spec: dict,
    maximum: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    values = frame[score_column].astype(float)
    mask = values.ge(float(spec["low"]))
    if bool(spec["include_high"]):
        mask &= values.le(float(spec["high"]))
    else:
        mask &= values.lt(float(spec["high"]))
    pool = frame.loc[mask].copy()
    pool["paper_target_distance"] = (
        pool[score_column].astype(float) - float(spec["target"])
    ).abs()
    pool_size = len(pool)
    if pool_size <= maximum:
        selected = pool
    else:
        ordered = pool.sort_values(
            ["paper_target_distance", "source_id"], kind="stable"
        )
        cutoff = float(ordered.iloc[int(maximum) - 1]["paper_target_distance"])
        closer = ordered[ordered["paper_target_distance"] < cutoff]
        tied = ordered[ordered["paper_target_distance"] == cutoff]
        needed = int(maximum) - len(closer)
        boundary = deterministic_tie_sample(tied, needed, seed)
        selected = pd.concat([closer, boundary], ignore_index=True)
    selected = selected.sort_values(
        ["paper_target_distance", "source_id"], kind="stable"
    ).reset_index(drop=True)
    selected["paper_score_target"] = float(spec["target"])
    selected["paper_score_bin"] = interval_label(spec)
    selected["paper_bin_pool_size"] = int(pool_size)
    selected["paper_selection_rank"] = np.arange(1, len(selected) + 1)
    return selected, pool_size

def group_title(
    frame: pd.DataFrame,
    source_label: str,
    score_column: str,
    spec: dict,
) -> str:
    target = format_score(float(spec["target"]))
    if frame.empty:
        summary = f"n=0 in {interval_label(spec)}"
    else:
        values = frame[score_column].astype(float)
        summary = (
            f"n={len(frame)}, min={format_score(values.min())}, "
            f"max={format_score(values.max())}"
        )
    return f"{source_label}: Nearest to {target}\n({summary})"

def draw_thumbnail_grid(
    fig,
    frame: pd.DataFrame,
    transform,
    *,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> None:
    if frame.empty:
        return
    if len(frame) > GRID_COLUMNS * GRID_ROWS:
        raise ValueError("A paper subpanel cannot contain more than 40 images.")
    images = load_agent3_images(frame, transform)
    figure_width, figure_height = fig.get_size_inches()
    gap_inches = 0.025
    available_width = width * figure_width - gap_inches * (GRID_COLUMNS - 1)
    available_height = height * figure_height - gap_inches * (GRID_ROWS - 1)
    cell_inches = min(
        available_width / GRID_COLUMNS,
        available_height / GRID_ROWS,
    )
    cell_width = cell_inches / figure_width
    cell_height = cell_inches / figure_height
    horizontal_gap = gap_inches / figure_width
    vertical_gap = gap_inches / figure_height
    grid_width = GRID_COLUMNS * cell_width + (GRID_COLUMNS - 1) * horizontal_gap
    grid_left = left + (width - grid_width) / 2.0
    grid_top = bottom + height
    for index, image in enumerate(images):
        row = index // GRID_COLUMNS
        column = index % GRID_COLUMNS
        axis = fig.add_axes(
            [
                grid_left + column * (cell_width + horizontal_gap),
                grid_top - (row + 1) * cell_height - row * vertical_gap,
                cell_width,
                cell_height,
            ]
        )
        axis.imshow(image.permute(1, 2, 0).clamp(0.0, 1.0))
        axis.set_axis_off()

def save_score_panel(
    output_path: Path,
    scores: pd.DataFrame,
    *,
    level: str,
    score_column: str,
    branch: str,
    hellinger: float,
    maximum: int,
    seed: int,
    transform,
    expected_rows: int = 9000,
) -> pd.DataFrame:
    branch_symbol = {"lower": "L", "upper": "U"}[branch]
    comparison_label = rf"$P$ vs $Q_{branch_symbol}$"
    generated_label = rf"$Q_{branch_symbol}$"
    selections = {}
    manifest_parts = []
    for distribution_index, distribution in enumerate(("real", branch)):
        source_scores = scores[scores["distribution"] == distribution]
        if len(source_scores) != expected_rows:
            raise RuntimeError(
                f"Expected {expected_rows:,} {distribution} rows for real vs {branch}."
            )
        for bin_index, spec in enumerate(BIN_SPECS):
            selected, pool_size = select_nearest_in_bin(
                source_scores,
                score_column,
                spec,
                maximum,
                seed + 100 * distribution_index + bin_index,
            )
            selected["paper_level"] = level
            selected["paper_branch"] = branch
            selected["paper_score_column"] = score_column
            selections[(distribution, float(spec["target"]))] = selected
            manifest_parts.append(selected)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.linewidth": 0.0,
            "figure.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(18.0, 7.6), dpi=220)
    fig.text(
        0.5,
        0.987,
        rf"{level.capitalize()}-level RDR: {comparison_label}, "
        rf"$\widehat{{H}}^2={hellinger:.3f}$",
        ha="center",
        va="top",
        fontsize=24,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.925,
        r"$P$: real images",
        ha="center",
        va="top",
        fontsize=25,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.475,
        f"{generated_label}: generated images",
        ha="center",
        va="top",
        fontsize=25,
        fontweight="bold",
    )

    panel_lefts = (0.014, 0.345, 0.676)
    panel_width = 0.310
    row_layout = {
        "real": {"title_y": 0.850, "grid_bottom": 0.525},
        branch: {"title_y": 0.405, "grid_bottom": 0.080},
    }
    source_labels = {"real": r"$P$", branch: generated_label}
    for distribution in ("real", branch):
        for panel_index, spec in enumerate(BIN_SPECS):
            frame = selections[(distribution, float(spec["target"]))]
            fig.text(
                panel_lefts[panel_index] + panel_width / 2.0,
                row_layout[distribution]["title_y"],
                group_title(
                    frame,
                    source_labels[distribution],
                    score_column,
                    spec,
                ),
                ha="center",
                va="top",
                fontsize=11.5,
            )
            draw_thumbnail_grid(
                fig,
                frame,
                transform,
                left=panel_lefts[panel_index],
                bottom=row_layout[distribution]["grid_bottom"],
                width=panel_width,
                height=0.275,
            )
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return pd.concat(manifest_parts, ignore_index=True)

def deterministic_sample(frame: pd.DataFrame, count: int, seed: int) -> pd.DataFrame:
    if count > len(frame):
        raise ValueError(f"Cannot select {count} rows from a pool of {len(frame)}.")
    rng = np.random.default_rng(int(seed))
    positions = rng.choice(len(frame), size=int(count), replace=False)
    return frame.iloc[positions].copy().reset_index(drop=True)

MIDPOINT_MAXIMUM = 1.0 - 1.0 / np.sqrt(2.0)

COLORS = {
    "real": "#4D4D4D",
    "lower": "#17866B",
    "upper": "#C04A67",
    "feature": "#C89020",
    "pixel": "#3974B7",
}

def distribution_label(distribution: str) -> str:
    labels = {
        "real": "P: real",
        "lower": r"$Q_L$: generated lower",
        "upper": r"$Q_U$: generated upper",
    }
    try:
        return labels[str(distribution)]
    except KeyError as error:
        raise ValueError(f"Unknown displayed distribution: {distribution}") from error

def save_metric_summary(
    path: Path,
    fid: dict,
    feature_locks: dict[str, dict],
    pixel_locks: dict[str, dict],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3), dpi=180)
    fid_values = [float(fid["fid_real_lower"]), float(fid["fid_real_upper"])]
    bars = axes[0].bar(
        ["lower", "upper"],
        fid_values,
        color=[COLORS["lower"], COLORS["upper"]],
        width=0.62,
    )
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].set_ylim(0, max(fid_values) * 1.22)
    axes[0].set_ylabel("FID to real")
    axes[0].set_title("1. One pool3 moment score")
    axes[0].text(
        0.5,
        0.94,
        f"gap = {abs(fid_values[0] - fid_values[1]):.2f}",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )

    feature_values = [
        float(feature_locks[branch]["variational_lower_bound"])
        for branch in PAIR_BRANCHES
    ]
    bars = axes[1].bar(
        ["real vs lower", "real vs upper"],
        feature_values,
        color=[COLORS["lower"], COLORS["upper"]],
        width=0.62,
    )
    axes[1].bar_label(bars, fmt="%.3f", padding=3)
    axes[1].axhline(MIDPOINT_MAXIMUM, color="#222222", linestyle="--", linewidth=1.1)
    axes[1].set_ylabel(r"variational $H^2(P_Z,M_{j,Z})$")
    axes[1].set_title("2. Full pool3 distributions")
    axes[1].tick_params(axis="x", rotation=10)

    x = np.arange(2)
    width = 0.34
    pixel_values = [
        float(pixel_locks[branch]["variational_lower_bound"])
        for branch in PAIR_BRANCHES
    ]
    axes[2].bar(
        x - width / 2,
        feature_values,
        width,
        label="feature",
        color=COLORS["feature"],
    )
    axes[2].bar(
        x + width / 2,
        pixel_values,
        width,
        label="pixel",
        color=COLORS["pixel"],
    )
    axes[2].axhline(MIDPOINT_MAXIMUM, color="#222222", linestyle="--", linewidth=1.1)
    axes[2].set_xticks(x, ["real vs lower", "real vs upper"], rotation=10)
    axes[2].set_ylabel(r"variational midpoint $H^2$")
    axes[2].set_title("3. Information before feature mapping")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.text(
        0.5,
        0.01,
        "FID and Hellinger use separate axes. Dashed lines mark the 0.292893 midpoint-Hellinger maximum.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def save_score_map(path: Path, scores: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), dpi=180, sharex=True, sharey=True)
    for axis, branch in zip(axes, PAIR_BRANCHES):
        frame = scores[branch]
        for distribution in ("real", branch):
            subset = frame[frame["distribution"] == distribution]
            axis.scatter(
                subset["feature_rdr"],
                subset["pixel_rdr"],
                s=5,
                alpha=0.16,
                color=COLORS[distribution],
                label=distribution_label(distribution),
                linewidths=0,
            )
        axis.axhline(1.0, color="#777777", linewidth=0.8)
        axis.axvline(1.0, color="#777777", linewidth=0.8)
        axis.plot([0, 2], [0, 2], color="#222222", linestyle="--", linewidth=1.0)
        axis.set_xlim(0, 2)
        axis.set_ylim(0, 2)
        axis.set_title(f"real vs {branch}")
        axis.set_xlabel(r"feature ratio $r_Z$")
        axis.legend(frameon=False, markerscale=3)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"pixel ratio $r_X$")
    fig.suptitle("Feature and pixel evidence within each real-generator contrast")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def save_score_distributions(path: Path, scores: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.6), dpi=180, sharex=True)
    for row, branch in enumerate(PAIR_BRANCHES):
        frame = scores[branch]
        for column, (score, label) in enumerate(
            (("feature_rdr", "feature"), ("pixel_rdr", "pixel"))
        ):
            axis = axes[row, column]
            for distribution in ("real", branch):
                axis.hist(
                    frame.loc[frame["distribution"] == distribution, score],
                    bins=np.linspace(0, 2, 61),
                    density=True,
                    histtype="step",
                    linewidth=1.7,
                    color=COLORS[distribution],
                    label=distribution_label(distribution),
                )
            axis.axvline(1.0, color="#777777", linewidth=0.8)
            axis.set_title(f"real vs {branch}: {label} RDR")
            axis.set_ylabel("density")
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(frameon=False)
    for axis in axes[-1]:
        axis.set_xlabel("midpoint density ratio")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
