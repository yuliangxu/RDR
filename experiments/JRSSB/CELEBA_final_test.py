"""Paired held-out RDR bootstrap, conditional on fitted models."""
from __future__ import annotations
from typing import Mapping
import numpy as np
import pandas as pd
PAIR_BRANCHES = ("lower", "upper")
REPRESENTATIONS = ("feature", "pixel")
def _score_components(scores: np.ndarray, epsilon: float, group: str) -> dict:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid final-test ratio scores for {group}.")
    clipped = np.clip(values, float(epsilon), 2.0 - float(epsilon))
    square_root = np.sqrt(clipped)
    if group == "p":
        objective = 0.5 / square_root + 0.25 * square_root
    elif group == "q":
        objective = 0.25 * square_root
    else:
        raise ValueError(f"Unknown midpoint group: {group!r}")
    return {
        "objective": objective,
        "plugin": 0.5 * square_root,
    }

def prepare_rdr_bootstrap_components(scores: Mapping, epsilon: float) -> dict:
    components = {}
    expected_real_size = None
    for branch in PAIR_BRANCHES:
        components[branch] = {}
        expected_q_size = None
        for representation in REPRESENTATIONS:
            item = scores[branch][representation]
            p_values = np.asarray(item["p"], dtype=np.float64).reshape(-1)
            q_values = np.asarray(item["q"], dtype=np.float64).reshape(-1)
            if expected_real_size is None:
                expected_real_size = len(p_values)
            if len(p_values) != expected_real_size:
                raise ValueError("All final-test real score vectors must have equal length.")
            if expected_q_size is None:
                expected_q_size = len(q_values)
            if len(q_values) != expected_q_size:
                raise ValueError(
                    f"Final feature/pixel Q sizes differ for real_vs_{branch}."
                )
            components[branch][representation] = {
                "p": _score_components(p_values, epsilon, "p"),
                "q": _score_components(q_values, epsilon, "q"),
            }
    return components

def _component_estimate(item: Mapping, p_rows: np.ndarray, q_rows: np.ndarray) -> tuple[float, float]:
    variational = 1.0 - float(item["p"]["objective"][p_rows].mean()) - float(
        item["q"]["objective"][q_rows].mean()
    )
    plugin = 1.0 - float(item["p"]["plugin"][p_rows].mean()) - float(
        item["q"]["plugin"][q_rows].mean()
    )
    return variational, plugin

def rdr_bootstrap_row(
    components: Mapping,
    seed: int,
    repeat: int,
) -> dict:
    real_size = len(components["lower"]["feature"]["p"]["objective"])
    q_sizes = {
        branch: len(components[branch]["feature"]["q"]["objective"])
        for branch in PAIR_BRANCHES
    }
    rng = np.random.default_rng(int(seed) + int(repeat))
    real_rows = rng.integers(0, real_size, size=real_size)
    q_rows = {
        branch: rng.integers(0, q_sizes[branch], size=q_sizes[branch])
        for branch in PAIR_BRANCHES
    }
    row = {"repeat": int(repeat)}
    for branch in PAIR_BRANCHES:
        estimates = {}
        for representation in REPRESENTATIONS:
            variational, plugin = _component_estimate(
                components[branch][representation], real_rows, q_rows[branch]
            )
            estimates[representation] = variational
            row[f"{representation}_h2_real_vs_{branch}"] = variational
            row[f"{representation}_plugin_real_vs_{branch}"] = plugin
        row[f"pixel_minus_feature_real_vs_{branch}"] = (
            estimates["pixel"] - estimates["feature"]
        )
    for representation in REPRESENTATIONS:
        row[f"{representation}_h2_lower_minus_upper"] = (
            row[f"{representation}_h2_real_vs_lower"]
            - row[f"{representation}_h2_real_vs_upper"]
        )
    row["increment_lower_minus_upper"] = (
        row["pixel_minus_feature_real_vs_lower"]
        - row["pixel_minus_feature_real_vs_upper"]
    )
    return row

def summarize_percentile_intervals(
    point_estimates: Mapping[str, float],
    bootstrap: pd.DataFrame,
    confidence_level: float,
) -> pd.DataFrame:
    confidence = float(confidence_level)
    alpha = 1.0 - confidence
    rows = []
    for metric, estimate in point_estimates.items():
        if metric not in bootstrap:
            raise KeyError(f"Bootstrap output is missing metric {metric!r}.")
        values = bootstrap[metric].to_numpy(dtype=np.float64)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"Bootstrap metric is invalid: {metric}")
        low, high = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
        rows.append(
            {
                "metric": metric,
                "estimate": float(estimate),
                "confidence_level": confidence,
                "interval_low": float(low),
                "interval_high": float(high),
                "bootstrap_repeats": int(len(values)),
                "interval_method": "paired_percentile_test_sample_bootstrap",
                "conditional_on_locked_models": True,
            }
        )
    return pd.DataFrame(rows)
