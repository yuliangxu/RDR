"""Audited, frozen AGP splits for calibration of real-versus-ICFM RDR.

The old metadata train/test files are not aligned with the abundance rows.
Recover sample IDs from the original abundance column order and split seed,
then join metadata by sample ID.  All output arrays retain the original
614-taxon representation; no normalization or selection uses RDR scores.
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _row_hashes(values):
    """Hash the actual float32 input rows using little-endian C order."""
    return [hashlib.sha256(row.astype("<f4").tobytes()).hexdigest() for row in values]


def prepare_data(data_root: Path, seed: int = 20260918):
    """Return ``(arrays, manifest, audit)`` without writing any files.

    ``arrays[(distribution, role)]`` contains float32 observations, where
    distribution is P or Q and role is train/validation/calibration/query.
    Manifest source_row is zero-based within the named source CSV.  Excluded
    real test rows are recorded separately with role="excluded".

    The sample is the target unit.  Subject-level allocation prevents subjects
    crossing roles but preserves repeated samples within each role, so the
    subsequent binomial CIs retain an explicit within-subject dependence caveat.
    """
    root = Path(data_root).resolve()
    files = {
        "P_train": "yx1_xtrain_raw.csv",
        "P_test": "yx1_xtest_raw.csv",
        "Q_bank": "yx1_icfm_recon_train.csv",
        "Q_legacy_test": "yx1_icfm_recon_test.csv",
        "abundance": "yx1_abundance.csv",
        "metadata": "yx2_metadata.csv",
        "generator_checkpoint": "icfm_checkpoint.pth",
    }
    for filename in files.values():
        if not (root / filename).is_file():
            raise FileNotFoundError(root / filename)

    abundance = pd.read_csv(root / files["abundance"], index_col=0)
    sample_ids = abundance.columns.astype(str).to_numpy()
    if len(np.unique(sample_ids)) != len(sample_ids):
        raise ValueError("Abundance sample IDs must be unique.")
    original_train, original_test = train_test_split(
        np.arange(len(sample_ids)), test_size=0.2, random_state=1
    )
    raw64 = {
        key: np.loadtxt(root / files[key], delimiter=",", dtype=np.float64)
        for key in ("P_train", "P_test", "Q_bank", "Q_legacy_test")
    }
    reference = abundance.to_numpy(dtype=np.float64).T / 100.0
    alignment = {}
    for key, indices in (("P_train", original_train), ("P_test", original_test)):
        raw = raw64[key]
        if raw.shape != reference[indices].shape:
            raise ValueError(f"{key} does not match the reconstructed original split.")
        maximum = float(np.max(np.abs(raw - reference[indices])))
        if not np.allclose(raw, reference[indices], atol=1e-14, rtol=1e-13):
            raise ValueError(f"{key} row order disagrees with abundance IDs.")
        alignment[key] = maximum
    dimension = reference.shape[1]
    for key, raw in raw64.items():
        if raw.ndim != 2 or raw.shape[1] != dimension:
            raise ValueError(f"{key} has an inconsistent feature dimension.")
        if not np.isfinite(raw).all() or (raw < 0).any():
            raise ValueError(f"{key} contains nonfinite or negative abundances.")

    metadata = pd.read_csv(
        root / files["metadata"], usecols=["#SampleID", "HOST_SUBJECT_ID"],
        dtype=str, keep_default_na=False,
    )
    if not metadata["#SampleID"].is_unique:
        raise ValueError("Metadata sample IDs are ambiguous.")
    metadata_order_matches = bool(np.array_equal(metadata["#SampleID"], sample_ids))
    indexed = metadata.set_index("#SampleID")
    if not pd.Index(sample_ids).isin(indexed.index).all():
        raise ValueError("Some abundance sample IDs lack metadata.")
    subjects = indexed.loc[sample_ids, "HOST_SUBJECT_ID"].str.strip()
    invalid_subjects = subjects.str.lower().isin(
        ["", "na", "nan", "none", "unknown", "unspecified", "not provided",
         "not applicable", "missing", "no_data", "null"]
    )
    if invalid_subjects.any():
        raise ValueError("Missing or ambiguous subject IDs prevent a subject-disjoint split.")
    subject_ids = subjects.to_numpy()
    p_train_subjects = subject_ids[original_train]
    p_test_subjects = subject_ids[original_test]

    generator_subjects = np.unique(p_train_subjects)
    train_order = np.random.default_rng(seed + 1).permutation(generator_subjects)
    train_cut = int(0.8 * len(train_order))
    train_mask = np.isin(p_train_subjects, train_order[:train_cut])
    eligible_mask = ~np.isin(p_test_subjects, generator_subjects)
    eligible_subjects = np.unique(p_test_subjects[eligible_mask])
    test_order = np.random.default_rng(seed).permutation(eligible_subjects)
    cal_cut = len(test_order) // 2
    cal_mask = eligible_mask & np.isin(p_test_subjects, test_order[:cal_cut])
    query_mask = eligible_mask & ~cal_mask
    q_order = np.random.default_rng(seed + 2).permutation(len(raw64["Q_bank"]))
    q_cuts = [0] + [int(f * len(q_order)) for f in (0.6, 0.8, 0.9)] + [len(q_order)]
    roles = ("train", "validation", "calibration", "query")
    selected = {
        ("P", "train"): ("P_train", np.flatnonzero(train_mask)),
        ("P", "validation"): ("P_train", np.flatnonzero(~train_mask)),
        ("P", "calibration"): ("P_test", np.flatnonzero(cal_mask)),
        ("P", "query"): ("P_test", np.flatnonzero(query_mask)),
        ("P", "excluded"): ("P_test", np.flatnonzero(~eligible_mask)),
    }
    for index, role in enumerate(roles):
        selected[("Q", role)] = ("Q_bank", q_order[q_cuts[index]:q_cuts[index + 1]])

    raw32 = {key: values.astype(np.float32) for key, values in raw64.items()}
    hashes = {key: _row_hashes(raw32[key]) for key in ("P_train", "P_test", "Q_bank")}
    if len(set(hashes["Q_bank"])) != len(hashes["Q_bank"]):
        raise ValueError("The ICFM bank contains duplicate float32 input rows.")
    if set(hashes["P_train"]) & set(hashes["P_test"]):
        raise ValueError("Real official train/test contain duplicated input rows.")
    arrays, frames = {}, []
    for (distribution, role), (source, indices) in selected.items():
        if role != "excluded":
            if not len(indices):
                raise ValueError(f"Empty {distribution}/{role} split.")
            arrays[(distribution, role)] = raw32[source][indices]
        if distribution == "P":
            original = original_train if source == "P_train" else original_test
            ids = sample_ids[original[indices]]
            grouped = subject_ids[original[indices]]
        else:
            ids = [f"icfm_train:{row:08d}" for row in indices]
            grouped = [""] * len(indices)
        frames.append(pd.DataFrame({
            "distribution": distribution, "role": role,
            "source_file": str(root / files[source]), "source_row": indices,
            "sample_id": ids, "subject_id": grouped,
            "row_sha256": [hashes[source][row] for row in indices],
            "exclusion_reason": "subject_seen_in_generator_training" if role == "excluded" else "",
        }))
    manifest = pd.concat(frames, ignore_index=True)
    real_used = manifest[(manifest.distribution == "P") & (manifest.role != "excluded")]
    if real_used.groupby("subject_id").role.nunique().max() != 1:
        raise AssertionError("A real subject crosses analysis roles.")
    if real_used.sample_id.duplicated().any():
        raise AssertionError("A real sample crosses analysis roles.")
    heldout = real_used[real_used.role.isin(["calibration", "query"])]
    if heldout.subject_id.isin(generator_subjects).any():
        raise AssertionError("A calibration/query subject was used to fit the generator.")
    q_used = manifest[manifest.distribution == "Q"]
    if q_used.source_row.duplicated().any() or len(q_used) != len(raw32["Q_bank"]):
        raise AssertionError("ICFM bank allocation is not a partition.")

    prefix_n = min(len(raw64["Q_bank"]), len(raw64["Q_legacy_test"]))
    prefix_diff = np.abs(raw64["Q_bank"][:prefix_n] - raw64["Q_legacy_test"][:prefix_n])
    random_rows = np.random.default_rng(1).permutation(len(raw64["Q_bank"]))[:prefix_n]
    random_diff = np.abs(raw64["Q_bank"][random_rows] - raw64["Q_legacy_test"][:prefix_n])
    counts = {}
    for (distribution, role), frame in manifest.groupby(["distribution", "role"], sort=True):
        counts[f"{distribution}_{role}"] = {
            "rows": int(len(frame)),
            "subjects": int(frame.subject_id.nunique()) if distribution == "P" else None,
        }
    audit = {
        "data_root": str(root), "seed": int(seed), "feature_dimension": int(dimension),
        "input_sha256": {files[key]: _file_sha256(root / files[key]) for key in files},
        "counts": counts,
        "seeds": {"P_calibration_query": int(seed), "P_train_validation": int(seed + 1),
                  "Q_all_roles": int(seed + 2)},
        "original_split": {"method": "sklearn.model_selection.train_test_split",
                           "test_size": 0.2, "random_state": 1},
        "raw_alignment_max_abs_error": alignment,
        "metadata_original_order_matches_abundance": metadata_order_matches,
        "metadata_join": "Join #SampleID strings to original abundance column IDs; ignore legacy metadata_train/test row order.",
        "missing_metadata_policy": "Fail on missing or duplicate sample IDs or blank/placeholder subject IDs; strip subject whitespace and preserve case.",
        "missing_sample_ids": 0, "missing_subject_ids": 0,
        "row_sha256_encoding": "C-order little-endian float32 input row bytes",
        "target_unit": "Original microbiome sample; all retained samples are used without subject averaging or abundance renormalization.",
        "generator_training_provenance": "AGP1_data_preprocessing.py creates x_lt from official xtrain_raw; AGP2_ICFM.py trains ICFM on x_lt_torch and reloads icfm_checkpoint.pth. Historical checkpoint lacks an independent split manifest.",
        "generator_training_rows": int(len(original_train)),
        "generator_training_subjects": int(len(generator_subjects)),
        "excluded_real_test_rows": int((~eligible_mask).sum()),
        "excluded_real_test_subjects": int(len(np.unique(p_test_subjects[~eligible_mask]))),
        "subject_disjoint_across_analysis_roles": True,
        "calibration_query_subjects_disjoint_generator_training": True,
        "q_single_bank_partition_no_duplicates": True,
        "q_legacy_test_unused": True,
        "q_prefix_audit": {
            "rows_compared": int(prefix_n),
            "paired_prefix_l1_median": float(np.median(prefix_diff.sum(axis=1))),
            "random_pair_l1_median": float(np.median(random_diff.sum(axis=1))),
            "paired_prefix_max_absolute_difference": float(prefix_diff.max()),
            "reason_excluded": "Historical train/test generation both use seed=2; close paired prefixes indicate shared latent draws despite numerical differences. Use only the generated train bank, repartitioned once.",
        },
        "limitations": [
            "Nominal C.1/C.2 coverage assumes independent samples. Repeated samples within retained subjects are not corrected by subject-level allocation.",
            "Historical feature filtering and representation preprocessing used the complete original AGP abundance table; this is a retrospective calibration using fixed preprocessing.",
            "Historical generator and generation provenance are recovered from source and frozen artifacts, not a contemporary split/generation manifest.",
            "Excluding subjects represented in generator training changes the real evaluation population to retained samples from previously unseen subjects.",
        ],
    }
    return arrays, manifest, audit
