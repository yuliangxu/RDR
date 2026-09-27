"""Audited 7:1:2-style AGP splits with the previous test pool preserved.

The original helper remains unchanged so archived fits can still be replayed.
Only source IDs, subject IDs, and sample counts determine this allocation.
"""

from pathlib import Path

import numpy as np

from AGP_ci_data import _row_hashes, prepare_data


def _nearest_prefix(counts, target):
    """Choose the closest cumulative sample count, keeping whole subjects."""
    cumulative = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    cut = int(np.argmin(np.abs(cumulative - target)))
    return cut, int(cumulative[cut])


def prepare_712_data(data_root: Path, seed: int = 20260918,
                     p_policy: str = "keep-all"):
    """Return arrays, a complete manifest, and the allocation audit.

    Arrays use ``(distribution, role)`` keys for P/Q and train/validation/test.
    The test set is exactly the old calibration/query union.  ``keep-all``
    divides the real fit pool approximately 7:1 while retaining every eligible
    sample; ``match-ratio`` retains approximately four times as many real fit
    samples as test samples, reserving the remaining whole subjects as unused.
    """
    if p_policy not in ("keep-all", "match-ratio"):
        raise ValueError("p_policy must be 'keep-all' or 'match-ratio'.")
    old_arrays, previous, original_audit = prepare_data(Path(data_root), seed)
    manifest = previous.copy()
    manifest["previous_role"] = manifest.role
    values = np.empty((len(manifest), original_audit["feature_dimension"]),
                      dtype=np.float32)
    available = np.zeros(len(manifest), dtype=bool)
    for (distribution, role), rows in old_arrays.items():
        mask = ((manifest.distribution == distribution)
                & (manifest.previous_role == role)).to_numpy()
        if int(mask.sum()) != len(rows):
            raise AssertionError("Original arrays and manifest disagree.")
        values[mask] = rows
        available[mask] = True

    old_test = manifest.previous_role.isin(["calibration", "query"])
    manifest.loc[old_test, "role"] = "test"
    p_fit = ((manifest.distribution == "P")
             & manifest.previous_role.isin(["train", "validation"]))
    subject_counts = manifest.loc[p_fit].groupby("subject_id").size()
    subject_order = np.random.default_rng(seed + 1).permutation(
        np.sort(subject_counts.index.to_numpy()))
    ordered_counts = subject_counts.loc[subject_order].to_numpy()
    n_p_test = int(((manifest.distribution == "P") & old_test).sum())
    n_p_fit = int(p_fit.sum())
    target_fit = n_p_fit if p_policy == "keep-all" else 4 * n_p_test
    if target_fit > n_p_fit:
        raise ValueError("The real fit pool is too small for the requested ratio.")
    target_train = int(round(7 * target_fit / 8))
    train_cut, n_p_train = _nearest_prefix(ordered_counts, target_train)
    if p_policy == "keep-all":
        fit_cut = len(subject_order)
    else:
        fit_cut, _ = _nearest_prefix(ordered_counts, target_fit)
    if not 0 < train_cut < fit_cut <= len(subject_order):
        raise AssertionError("Subject allocation produced an empty fit role.")
    manifest.loc[p_fit, "role"] = "unused"
    manifest.loc[p_fit & manifest.subject_id.isin(subject_order[:train_cut]),
                 "role"] = "train"
    manifest.loc[p_fit & manifest.subject_id.isin(subject_order[train_cut:fit_cut]),
                 "role"] = "validation"
    manifest.loc[manifest.role == "unused", "exclusion_reason"] = (
        "reserved_to_match_7_1_2_without_changing_test_pool")

    q_mask = manifest.distribution == "Q"
    n_q = int(q_mask.sum())
    q_order = np.random.default_rng(seed + 2).permutation(n_q)
    q_cuts = [0, int(0.7 * n_q), int(0.8 * n_q), n_q]
    for index, role in enumerate(("train", "validation", "test")):
        selected = manifest.source_row.isin(q_order[q_cuts[index]:q_cuts[index + 1]])
        manifest.loc[q_mask & selected, "role"] = role

    roles = ("train", "validation", "test")
    arrays, counts, fractions = {}, {}, {}
    for distribution in ("P", "Q"):
        for role in roles:
            mask = ((manifest.distribution == distribution)
                    & (manifest.role == role)).to_numpy()
            if not mask.any() or not available[mask].all():
                raise AssertionError("Missing source rows in a retained split.")
            rows = values[mask].copy()
            if _row_hashes(rows) != manifest.loc[mask, "row_sha256"].tolist():
                raise AssertionError("Float32 rows disagree with source hashes.")
            arrays[(distribution, role)] = rows
        selected = manifest[(manifest.distribution == distribution)
                            & manifest.role.isin(roles)]
        if selected.sample_id.duplicated().any():
            raise AssertionError("A sample crosses analysis roles.")
        old_ids = set(manifest.loc[(manifest.distribution == distribution)
                                  & old_test, "sample_id"])
        new_ids = set(selected.loc[selected.role == "test", "sample_id"])
        if old_ids != new_ids:
            raise AssertionError("The original full test pool changed.")
        fractions[distribution] = {
            role: float((selected.role == role).sum() / len(selected))
            for role in roles
        }
    real = manifest[manifest.distribution == "P"]
    if real[real.role != "excluded"].groupby("subject_id").role.nunique().max() != 1:
        raise AssertionError("A real subject crosses allocation roles.")
    generator_subjects = set(subject_counts.index)
    if set(real.loc[real.role == "test", "subject_id"]) & generator_subjects:
        raise AssertionError("A test subject appeared in generator training.")
    if not manifest.loc[manifest.role == "excluded"].equals(
            previous.loc[previous.role == "excluded"].assign(previous_role="excluded")):
        raise AssertionError("The original excluded rows changed.")
    for (distribution, role), frame in manifest.groupby(
            ["distribution", "role"], sort=True):
        counts[f"{distribution}_{role}"] = {
            "rows": int(len(frame)),
            "subjects": int(frame.subject_id.nunique()) if distribution == "P" else None,
        }
    if counts["P_train"]["rows"] != n_p_train:
        raise AssertionError("Real training allocation disagrees with its prefix.")
    audit = {
        "original_data_audit": original_audit,
        "data_root": original_audit["data_root"],
        "input_sha256": original_audit["input_sha256"],
        "seed": int(seed), "feature_dimension": original_audit["feature_dimension"],
        "p_policy": p_policy, "counts": counts, "retained_role_fractions": fractions,
        "requested_train_validation_test_ratio": [7, 1, 2],
        "p_fit_target_rows": int(target_fit), "p_train_target_rows": target_train,
        "p_allocation": "Shuffle sorted generator-training subject IDs with seed+1; choose cumulative whole-subject prefixes nearest the prespecified sample-count targets; ties choose the shorter prefix.",
        "q_allocation": "Shuffle the single generated bank with seed+2; cut at floor(0.7*N) and floor(0.8*N), preserving the original full test tail.",
        "selection_uses_rdr_scores": False,
        "test_equals_calibration": True,
        "original_full_test_ids_preserved": True,
        "original_excluded_rows_preserved": True,
        "subject_disjoint_across_retained_and_unused_roles": True,
        "test_subjects_disjoint_generator_training": True,
        "arrays_match_manifest_float32_hashes": True,
        "q_legacy_test_unused": True,
        "limitations": original_audit["limitations"] + [
            "The test pool has been examined in earlier analyses; this rerun is retrospective, not a newly untouched confirmation sample.",
            "Real test size is constrained by the previously unseen-subject pool; keep-all therefore approximates the requested ratio by splitting the remaining real fit pool 7:1.",
        ],
    }
    return arrays, manifest, audit
