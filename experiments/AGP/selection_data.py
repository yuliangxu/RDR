"""Freeze audited AGP inputs for selection, the final fit, and a real-data null.

Selection partitions only the historical development pool.  Final and null
studies deliberately reuse development observations after selection; their
historical test observations never enter model selection.  Preparation audits
all raw inputs, while training can open only the arrays for its own roles.
"""

from __future__ import annotations

import csv
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# The preserved audit modules predate package imports and use bare imports.
_AGP_DIRECTORY = str(Path(__file__).resolve().parent)
if _AGP_DIRECTORY not in sys.path:
    sys.path.insert(0, _AGP_DIRECTORY)
from AGP_712_data import prepare_712_data
from AGP_ci_data import _file_sha256, _row_hashes


ORIGINAL_SEED = 20260918
EXPECTED_EXCLUSIONS = 609
STUDY_ROLES = {
    "selection": ("train", "stopping", "calibration", "evaluation"),
    "final": ("train", "stopping", "test"),
    "null": ("train", "stopping", "test"),
}


def _write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _receipt(path: Path, output: Path) -> dict:
    return {"path": str(path.relative_to(output)), "sha256": _file_sha256(path)}


def _subject_parts(frame: pd.DataFrame, fractions, seed: int):
    """Shuffle sorted subjects and choose nearest cumulative sample targets.

    Ties select the shorter prefix.  Each role retains at least one subject.
    Returned frames preserve their order in the source manifest.
    """
    fractions = np.asarray(fractions, dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Positive subject-partition fractions must sum to one.")
    if frame.subject_id.isna().any() or frame.subject_id.astype(str).eq("").any():
        raise ValueError("Real samples require nonmissing subject IDs.")
    subjects = np.sort(frame.subject_id.unique())
    if len(subjects) < len(fractions):
        raise ValueError("Too few subjects for nonempty subject-disjoint roles.")
    subjects = np.random.default_rng(seed).permutation(subjects)
    counts = frame.groupby("subject_id").size().reindex(subjects).to_numpy()
    cumulative = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    cuts = [0]
    for i, fraction in enumerate(np.cumsum(fractions)[:-1]):
        # Leave at least one complete subject for every remaining role.
        possible = np.arange(cuts[-1] + 1, len(subjects) - (len(fractions) - i - 1) + 1)
        cut = possible[np.argmin(np.abs(cumulative[possible] - fraction * len(frame)))]
        cuts.append(int(cut))
    cuts.append(len(subjects))
    return [frame.loc[frame.subject_id.isin(subjects[a:b])].copy()
            for a, b in zip(cuts[:-1], cuts[1:])]


def _row_parts(frame: pd.DataFrame, fractions, seed: int):
    fractions = np.asarray(fractions, dtype=float)
    if np.any(fractions <= 0) or not np.isclose(fractions.sum(), 1):
        raise ValueError("Positive row-partition fractions must sum to one.")
    order = np.random.default_rng(seed).permutation(len(frame))
    # Decimal input fractions describe exact cut targets: 0.7 + 0.1 must not
    # become 0.7999999999999999 and lose a row when the target is an integer.
    cuts, cumulative = [0], Fraction(0)
    for fraction in fractions[:-1]:
        cumulative += Fraction(str(float(fraction)))
        cuts.append(cumulative.numerator * len(frame) // cumulative.denominator)
    cuts.append(len(frame))
    if min(np.diff(cuts)) <= 0:
        raise ValueError("Too few rows for nonempty selection roles.")
    return [frame.iloc[np.sort(order[a:b])].copy() for a, b in zip(cuts[:-1], cuts[1:])]


def _taxa_table(path: Path, dimension: int) -> pd.DataFrame:
    """Read only identifiers, preserving the original abundance feature order."""
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        identifiers = [row[0] for row in reader]
    if len(identifiers) != dimension or len(set(identifiers)) != dimension:
        raise ValueError("Original abundance taxonomy dimension or uniqueness changed.")
    ranks = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
    prefixes = dict(zip(("k", "p", "c", "o", "f", "g", "s"), ranks))
    records = []
    for index, identifier in enumerate(identifiers):
        record = dict(taxon_index=index, taxon_id=identifier)
        record.update({rank: "Unclassified" for rank in ranks})
        seen = set()
        for item in identifier.split(";"):
            prefix, separator, label = item.strip().partition("__")
            rank = prefixes.get(prefix.lower())
            if not separator or rank is None:
                raise ValueError(f"Unrecognized taxonomy component: {item!r}")
            if rank in seen:
                raise ValueError(f"Repeated taxonomy rank in {identifier!r}")
            seen.add(rank)
            record[rank] = label.strip() or "Unclassified"
        records.append(record)
    return pd.DataFrame(records)


def _assert_partition(frame: pd.DataFrame, roles, study: str):
    if set(frame.role) != set(roles):
        raise AssertionError(f"{study}: missing or unexpected analysis roles.")
    if frame.sample_id.duplicated().any():
        raise AssertionError(f"{study}: a sample occurs more than once.")
    if frame.duplicated(["source_file", "source_row"]).any():
        raise AssertionError(f"{study}: a source row occurs more than once.")
    real = frame.loc[frame.original_distribution.eq("P")]
    assignment = real.role + ":" + real.distribution
    if assignment.groupby(real.subject_id).nunique().max() != 1:
        raise AssertionError(f"{study}: a subject crosses roles or distributions.")
    for role in roles:
        for side in ("P", "Q"):
            if not ((frame.role == role) & (frame.distribution == side)).any():
                raise AssertionError(f"{study}: empty {role}/{side}.")


def _write_study(output: Path, study: str, frame: pd.DataFrame,
                 values: np.ndarray, seed: int, test_ids: set) -> dict:
    directory = output / "data" / study
    directory.mkdir(parents=True, exist_ok=True)
    frame = frame.reset_index(drop=True).copy()
    roles = STUDY_ROLES[study]
    _assert_partition(frame, roles, study)
    if study == "selection":
        if set(frame.sample_id) & test_ids or frame.original_role.eq("test").any():
            raise AssertionError("Historical test observations entered selection.")
    elif set(frame.loc[frame.role.eq("test"), "sample_id"]) != (
            test_ids if study == "final" else
            set(frame.loc[frame.original_role.eq("test"), "sample_id"])):
        raise AssertionError(f"{study}: original test observations changed.")
    arrays, counts, role_counts = {}, {}, {}
    frame["array_row"] = -1
    for role in roles:
        role_counts[role] = {}
        for side in ("P", "Q"):
            key = f"{role}_{side.lower()}"
            mask = frame.role.eq(role) & frame.distribution.eq(side)
            rows = values[frame.loc[mask, "original_manifest_row"].to_numpy(dtype=int)]
            if rows.dtype != np.float32 or _row_hashes(rows) != frame.loc[mask, "row_sha256"].tolist():
                raise AssertionError(f"{study}/{key}: row/manifest hash disagreement.")
            frame.loc[mask, "array_row"] = np.arange(len(rows))
            path = directory / f"{key}.npy"
            np.save(path, rows, allow_pickle=False)
            # Independently read back the array written on disk.
            reread = np.load(path, mmap_mode="r", allow_pickle=False)
            if reread.shape != rows.shape or _row_hashes(reread) != frame.loc[mask, "row_sha256"].tolist():
                raise AssertionError(f"{study}/{key}: cached array hash disagreement.")
            arrays[key] = dict(_receipt(path, output), shape=list(rows.shape), dtype=str(rows.dtype))
            real = frame.loc[mask, "original_distribution"].eq("P").all()
            counts[key] = {"rows": int(mask.sum()),
                           "subjects": int(frame.loc[mask, "subject_id"].nunique()) if real else None}
            role_counts[role][side.lower()] = int(mask.sum())
    manifest_path = directory / "manifest.csv"
    frame.to_csv(manifest_path, index=False)
    audit = {
        "study": study, "seed": int(seed), "counts": counts,
        "sample_ids_and_source_rows_unique": True,
        "real_subjects_disjoint_across_roles_and_sides": True,
        "arrays_match_manifest_float32_hashes": True,
        "array_row_zero_based_within_role_and_distribution": True,
        "test_rows_used_for_model_selection": False,
        "original_test_rows_preserved": study != "selection",
        "shared_development_rows_across_studies_are_intentional": True,
    }
    audit_path = directory / "audit.json"
    _write_json(audit_path, audit)
    return dict(roles=list(roles), counts=counts, role_counts=role_counts,
                arrays=arrays, manifest=_receipt(manifest_path, output),
                audit=_receipt(audit_path, output))


def _verify_existing(output: Path, protocol: dict, settings: dict):
    if protocol["settings"] != settings:
        raise RuntimeError("Prepared AGP data are immutable; split settings changed.")
    for filename, expected in protocol["input_sha256"].items():
        path = Path(protocol["data_root"]) / filename
        if _file_sha256(path) != expected:
            raise RuntimeError(f"Original AGP input SHA256 mismatch: {path}")
    receipts = [protocol[key] for key in ("original_manifest", "original_audit", "taxa")]
    for study in protocol["studies"].values():
        receipts += [study["manifest"], study["audit"], *study["arrays"].values()]
    for receipt in receipts:
        if _file_sha256(output / receipt["path"]) != receipt["sha256"]:
            raise RuntimeError(f"Prepared AGP input SHA256 mismatch: {receipt['path']}")


def prepare(output: Path, data_root: Path, config: dict) -> dict:
    """Prepare raw float32 arrays and provenance; return ``data/protocol.json``.

    Only split seeds affect these caches.  Existing completed caches are fully
    hash-verified, and cannot be silently replaced with a different protocol.
    """
    output, data_root = Path(output).resolve(), Path(data_root).resolve()
    settings = dict(data_root=str(data_root), original_seed=ORIGINAL_SEED,
                    split_seed=int(config.get("split_seed", 2026100401)),
                    null_split_seed=int(config.get("null_split_seed", 2026100402)),
                    p_policy="keep-all", selection_fractions=[0.7, 0.1, 0.1, 0.1])
    protocol_path = output / "data" / "protocol.json"
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text())
        _verify_existing(output, protocol, settings)
        return protocol

    arrays, original, audit = prepare_712_data(data_root, seed=ORIGINAL_SEED, p_policy="keep-all")
    if int(original.role.eq("excluded").sum()) != EXPECTED_EXCLUSIONS:
        raise AssertionError("The original 609 real-test exclusions changed.")
    excluded = original.loc[original.role.eq("excluded")]
    if not excluded.distribution.eq("P").all():
        raise AssertionError("Only original real rows may be excluded.")
    dimension = int(audit["feature_dimension"])
    values = np.zeros((len(original), dimension), dtype=np.float32)
    available = np.zeros(len(original), dtype=bool)
    for (side, role), rows in arrays.items():
        mask = original.distribution.eq(side) & original.role.eq(role)
        if len(rows) != int(mask.sum()) or _row_hashes(rows) != original.loc[mask, "row_sha256"].tolist():
            raise AssertionError("Audited source arrays disagree with original manifest.")
        values[mask.to_numpy()] = rows
        available[mask.to_numpy()] = True
    if not np.array_equal(available, ~original.role.eq("excluded").to_numpy()):
        raise AssertionError("Every retained original row must have an audited raw value.")

    directory = output / "data"
    directory.mkdir(parents=True, exist_ok=True)
    original_path = directory / "original_manifest.csv"
    original.to_csv(original_path, index=False)
    audit_path = directory / "original_audit.json"
    _write_json(audit_path, audit)
    taxonomy_path = directory / "taxa.csv"
    _taxa_table(data_root / "yx1_abundance.csv", dimension).to_csv(taxonomy_path, index=False)

    base = original.assign(original_manifest_row=np.arange(len(original)),
                           original_role=original.role,
                           original_distribution=original.distribution)
    retained = base.loc[~base.role.eq("excluded")].copy()
    development = retained.loc[retained.role.isin(["train", "validation"])]
    test_ids = set(retained.loc[retained.role.eq("test"), "sample_id"])
    selection_frames = []
    for side in ("P", "Q"):
        frame = development.loc[development.distribution.eq(side)]
        parts = (_subject_parts(frame, settings["selection_fractions"], settings["split_seed"])
                 if side == "P" else
                 _row_parts(frame, settings["selection_fractions"], settings["split_seed"] + 1))
        for role, part in zip(STUDY_ROLES["selection"], parts):
            selection_frames.append(part.assign(role=role))
    selection = pd.concat(selection_frames, ignore_index=True)
    if set(selection.sample_id) != set(development.sample_id):
        raise AssertionError("Selection roles must partition the complete development pool.")
    final = retained.copy()
    final.loc[final.role.eq("validation"), "role"] = "stopping"
    null_frames = []
    for index, role in enumerate(STUDY_ROLES["null"]):
        real = final.loc[final.role.eq(role) & final.distribution.eq("P")]
        for side, part in zip(("P", "Q"), _subject_parts(
                real, [.5, .5], settings["null_split_seed"] + index)):
            null_frames.append(part.assign(distribution=side))
    null = pd.concat(null_frames, ignore_index=True)
    if set(null.sample_id) != set(retained.loc[retained.distribution.eq("P"), "sample_id"]):
        raise AssertionError("Null halves must partition every retained real observation.")
    if set(null.loc[null.role.eq("test"), "sample_id"]) != set(
            retained.loc[retained.role.eq("test") & retained.distribution.eq("P"), "sample_id"]):
        raise AssertionError("Null test halves changed the original real test cohort.")

    studies = {}
    for name, frame, seed in (("selection", selection, settings["split_seed"]),
                              ("final", final, ORIGINAL_SEED),
                              ("null", null, settings["null_split_seed"])):
        studies[name] = _write_study(output, name, frame, values, seed, test_ids)
    protocol = dict(
        version=1, settings=settings, data_root=str(data_root),
        feature_dimension=dimension, split_seed=settings["split_seed"],
        null_split_seed=settings["null_split_seed"], original_seed=ORIGINAL_SEED,
        input_sha256=audit["input_sha256"],
        original_manifest=_receipt(original_path, output), original_audit=_receipt(audit_path, output),
        taxa=_receipt(taxonomy_path, output), studies=studies,
        original_excluded_real_test_rows=EXPECTED_EXCLUSIONS,
        original_final_test_ids_preserved=True, selection_uses_test_rows=False,
        selection_uses_rdr_scores=False, raw_test_values_audited_during_preparation=True,
        subject_allocation="Shuffle sorted subject IDs, then choose whole-subject prefixes nearest cumulative sample-count targets; ties choose the shorter prefix.",
        selection_q_allocation="Shuffle development-bank rows with split_seed+1; floor cumulative fractions 0.7/0.8/0.9 and retain the remaining rows.",
        null_allocation="Independently split original real train/validation/test subjects into sample-count-balanced halves using null_split_seed + role index.",
        taxonomy_source="Original yx1_abundance.csv taxonomic index in exact feature order; blank ranks are Unclassified.",
        limitations=audit["limitations"] + [
            "Development data are reused across selection, final fitting, and the real-versus-real control by design.",
            "The historical test cohort was inspected previously; final analyses remain retrospective.",
        ],
    )
    _write_json(protocol_path, protocol)
    return protocol
