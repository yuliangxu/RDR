"""Exact identity-disjoint real null split construction from canonical real manifests."""

from __future__ import annotations

from pathlib import Path

from typing import Iterable, Optional, Union

import numpy as np

import pandas as pd

def load_manifest(output_root: Union[str, Path], role: str, source: str, selected: bool = False) -> pd.DataFrame:
    kind = "selected" if selected else "input"
    path = Path(output_root) / "manifests" / kind / f"{role}_{source}.csv"
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Manifest is empty: {path}")
    return frame

def validate_disjoint_manifests(frames: Iterable[pd.DataFrame]) -> None:
    seen = set()
    for frame in frames:
        keys = set(zip(frame["source"], frame["source_id"]))
        overlap = seen.intersection(keys)
        if overlap:
            example = next(iter(overlap))
            raise ValueError(f"Manifest overlap detected, for example {example}.")
        seen.update(keys)

def _split_manifest_rows(
    manifest: pd.DataFrame,
    sample_size: int,
    seed: int,
    eligible_rows: Optional[np.ndarray] = None,
    group_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = (
        np.arange(len(manifest), dtype=np.int64)
        if eligible_rows is None
        else np.asarray(eligible_rows, dtype=np.int64)
    )
    if len(np.unique(rows)) != len(rows):
        raise ValueError("Eligible manifest rows contain duplicates.")
    if len(rows) < 2 * sample_size:
        raise ValueError(
            f"The eligible pool has {len(rows)} rows; at least {2 * sample_size} "
            "are required for two disjoint folds."
        )
    rng = np.random.default_rng(seed)
    if group_column is None:
        chosen = rng.choice(rows, size=2 * sample_size, replace=False)
        fold_rows = (chosen[:sample_size], chosen[sample_size:])
    else:
        if group_column not in manifest:
            raise KeyError(f"Manifest has no grouping column {group_column!r}.")
        eligible = manifest.iloc[rows].copy()
        eligible["_manifest_row"] = rows
        if eligible[group_column].isna().any():
            raise ValueError(f"Grouping column {group_column!r} contains missing values.")
        grouped = [
            (group_value, group["_manifest_row"].to_numpy(dtype=np.int64))
            for group_value, group in eligible.groupby(group_column, sort=False)
        ]
        tie_breaks = rng.random(len(grouped))
        order = sorted(
            range(len(grouped)),
            key=lambda index: (-len(grouped[index][1]), tie_breaks[index]),
        )
        fold_group_rows: list[list[np.ndarray]] = [[], []]
        fold_totals = [0, 0]
        for index in order:
            target = 0 if fold_totals[0] <= fold_totals[1] else 1
            group_rows = grouped[index][1]
            fold_group_rows[target].append(group_rows)
            fold_totals[target] += len(group_rows)
        if min(fold_totals) < sample_size:
            raise ValueError(
                f"Identity-disjoint split produced fold capacities {fold_totals}; "
                f"each must contain at least {sample_size} images."
            )
        fold_rows = tuple(
            rng.choice(np.concatenate(group_rows), size=sample_size, replace=False)
            for group_rows in fold_group_rows
        )
    first = manifest.iloc[fold_rows[0]].copy().reset_index(drop=True)
    second = manifest.iloc[fold_rows[1]].copy().reset_index(drop=True)
    first["null_fold"] = "A"
    second["null_fold"] = "B"
    validate_disjoint_manifests((first, second))
    if group_column is not None:
        overlap = set(first[group_column]).intersection(second[group_column])
        if overlap:
            raise ValueError(f"Null folds overlap in {group_column}: {next(iter(overlap))}")
    return first, second

def fixed_real_fold_manifests(
    output_root: Union[str, Path],
    role: str,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fixed image- and identity-disjoint folds from one real role."""
    manifest = load_manifest(output_root, role, "real", selected=False)
    return _split_manifest_rows(
        manifest,
        sample_size,
        seed,
        group_column="identity",
    )
