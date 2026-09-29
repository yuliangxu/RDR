"""Label-only DCGAN RDR selection diagnostics on independent sample roles.

The probability of a P label in a balanced P/Q mixture is r/2. Calibration
intervals target population score-cell averages, not individual RDR values.
Comparisons with estimated neural cell means are descriptive: their sampling
uncertainty is not included in C.1/C.2. Callers allocate alpha across models
when simultaneous protection across the candidate grid is wanted.
"""

from numbers import Integral

import numpy as np

from utils.calibration import bin_scores, calibrate_counts


def _scores(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a nonempty one-dimensional score array")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 2)):
        raise ValueError(f"{name} scores must be finite and in [0, 2]")
    return values


def balanced_brier(p, q):
    """Balanced binary Brier with P=1, Q=0 and probability r/2.

    Each source has weight one half, regardless of its sample count. Population
    excess Brier equals one quarter of E_(P+Q)/2[(r-r0)**2].
    """
    p, q = _scores(p, "P"), _scores(q, "Q")
    return float(.5 * np.mean((p / 2 - 1) ** 2) + .5 * np.mean((q / 2) ** 2))


def calibration_diagnostics(cal_p, cal_q, eval_p, eval_q, alpha=.05, bins=20):
    """Return ``(summary, cells)`` for a frozen equal-width score partition.

    Calibration samples estimate theta_j=2P(A_j)/(P(A_j)+Q(A_j)); independent
    evaluation samples estimate the neural mean and mixture mass in each cell.
    Both use balanced source weights even with unequal P/Q sample sizes.
    Empty calibration cells have no point estimate. Weighted gaps and distances
    normalize over supported evaluation mass; widths include all evaluation
    mass. The middle region consists of whole fixed cells inside [0.7, 1.2).
    """
    if isinstance(bins, bool) or not isinstance(bins, Integral) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    cp, cq, ep, eq = [_scores(values, name) for values, name in zip(
        (cal_p, cal_q, eval_p, eval_q), ("calibration P", "calibration Q", "evaluation P", "evaluation Q"))]
    edges = np.linspace(0., 2., int(bins) + 1)
    cp_bins, cq_bins, ep_bins, eq_bins = [bin_scores(values, edges) for values in (cp, cq, ep, eq)]
    kp, kq, np_eval, nq_eval = [np.bincount(index, minlength=bins) for index in
                               (cp_bins, cq_bins, ep_bins, eq_bins)]
    intervals = calibrate_counts(kp, kq, len(cp), len(cq), alpha)
    p_sums = np.bincount(ep_bins, weights=ep, minlength=bins)
    q_sums = np.bincount(eq_bins, weights=eq, minlength=bins)
    masses = .5 * (np_eval / len(ep) + nq_eval / len(eq))
    rows = []
    for j, mass in enumerate(masses):
        empty = int(kp[j] + kq[j]) == 0
        neural = float(.5 * (p_sums[j] / len(ep) + q_sums[j] / len(eq)) / mass) if mass else None
        theta = None if empty else float(intervals["estimate"][j])
        gap = None if neural is None or theta is None else neural - theta
        lower, upper = float(intervals["c2_lower"][j]), float(intervals["c2_upper"][j])
        distance = None if neural is None else max(lower - neural, neural - upper, 0.)
        rows.append({
            "bin": j, "left": float(edges[j]), "right": float(edges[j + 1]),
            "cal_count_p": int(kp[j]), "cal_count_q": int(kq[j]),
            "calibration_mass": float(.5 * (kp[j] / len(cp) + kq[j] / len(cq))),
            "eval_count_p": int(np_eval[j]), "eval_count_q": int(nq_eval[j]),
            "eval_mass": float(mass), "neural_mean": neural, "calibrated_rdr": theta,
            "gap": gap, "absolute_gap": None if gap is None else abs(gap),
            "c1_lower": float(intervals["c1_lower"][j]),
            "c1_upper": float(intervals["c1_upper"][j]),
            "c1_unavailable": bool(intervals["c1_unavailable"][j]),
            "c2_lower": lower, "c2_upper": upper, "c2_width": upper - lower,
            "c2_distance": distance, "calibration_empty": empty,
            "middle_region": bool(edges[j] >= .7 - 1e-12 and edges[j + 1] <= 1.2 + 1e-12),
        })

    def aggregate(cells):
        mass = float(sum(cell["eval_mass"] for cell in cells))
        supported = [cell for cell in cells if cell["gap"] is not None]
        support = float(sum(cell["eval_mass"] for cell in supported))
        average = lambda key: float(sum(cell["eval_mass"] * cell[key] for cell in supported) / support) if support else None
        return {"mass": mass, "supported_mass": support,
                "local_gap": average("absolute_gap"), "c2_distance": average("c2_distance"),
                "signed_gap": average("gap"),
                "maximum_gap": max((cell["absolute_gap"] for cell in supported), default=None)}

    full = aggregate(rows)
    middle = aggregate([cell for cell in rows if cell["middle_region"]])
    summary = {key: full[key] for key in ("local_gap", "c2_distance", "supported_mass", "maximum_gap", "signed_gap")}
    summary.update({
        "unsupported_mass": float(sum(cell["eval_mass"] for cell in rows if cell["gap"] is None)),
        "c2_width": float(sum(cell["eval_mass"] * cell["c2_width"] for cell in rows)),
        "middle_mass": middle["mass"], "middle_supported_mass": middle["supported_mass"],
        "middle_supported_fraction": middle["supported_mass"] / middle["mass"] if middle["mass"] else None,
        "middle_local_gap": middle["local_gap"], "middle_c2_distance": middle["c2_distance"],
        "middle_signed_gap": middle["signed_gap"], "middle_maximum_gap": middle["maximum_gap"],
        "bins": int(bins), "alpha": float(alpha),
        "n_cal_p": len(cp), "n_cal_q": len(cq), "n_eval_p": len(ep), "n_eval_q": len(eq),
    })
    return summary, rows


SELECTION_RULE = (
    "Reference: minimum mean balanced Brier. Shortlist: mean paired Brier "
    "difference from reference <= one standard error of paired differences "
    "+ 1e-12, with supported evaluation mass >= 0.99 in every repeat. "
    "Choose smallest mean local absolute gap, then mean Brier, then candidate ID. "
    "The repeat standard error is descriptive conditional on the fixed MNIST "
    "data; it is not a confidence interval or an independent-data uncertainty estimate."
)


def choose_configuration(rows):
    """Select from a complete paired candidate-by-repeat table.

    Invalid, missing, duplicate or nonfinite records fail explicitly. Repeat
    labels and all corresponding sample roles must be paired by the caller.
    A one-repeat study has zero paired SE and only exact Brier ties can enter
    the shortlist. The 0.99 support rule applies to every repeat separately.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("Selection requires candidate records")
    grouped, repeats = {}, set()
    for source in rows:
        row = dict(source)
        required = {"candidate", "repeat", "loss", "output_alpha", "brier", "local_gap", "supported_mass"}
        if required - row.keys():
            raise ValueError(f"Selection record is missing {sorted(required - row.keys())}")
        candidate, repeat = row["candidate"], row["repeat"]
        if not isinstance(candidate, str) or not candidate:
            raise ValueError("candidate must be a nonempty string")
        if isinstance(repeat, bool) or not isinstance(repeat, Integral) or repeat < 0:
            raise ValueError("repeat must be a nonnegative integer")
        if not isinstance(row["loss"], str) or not row["loss"]:
            raise ValueError("loss must be a nonempty string")
        for key in ("output_alpha", "brier", "local_gap", "supported_mass"):
            value = row[key]
            if isinstance(value, (bool, str)) or value is None or not np.isscalar(value):
                raise ValueError(f"{key} must be a finite number")
            try:
                row[key] = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{key} must be a finite number") from error
            if not np.isfinite(row[key]):
                raise ValueError(f"{key} must be finite")
        if (row["output_alpha"] <= 0 or not 0 <= row["brier"] <= 1
                or not 0 <= row["local_gap"] <= 2 or not 0 <= row["supported_mass"] <= 1 + 1e-12):
            raise ValueError("Selection metrics or activation slope are outside their valid ranges")
        records = grouped.setdefault(candidate, {})
        if repeat in records:
            raise ValueError(f"Duplicate candidate/repeat: {candidate}/{repeat}")
        if records:
            previous = next(iter(records.values()))
            if (row["loss"], row["output_alpha"]) != (previous["loss"], previous["output_alpha"]):
                raise ValueError(f"Candidate definition differs across repeats: {candidate}")
        records[repeat] = row
        repeats.add(repeat)
    for candidate, records in grouped.items():
        if set(records) != repeats:
            raise ValueError(f"Incomplete paired repeats for candidate {candidate}")
    repeats = sorted(repeats)
    brier = {candidate: np.array([records[repeat]["brier"] for repeat in repeats])
             for candidate, records in grouped.items()}
    reference = min(grouped, key=lambda candidate: (float(brier[candidate].mean()), candidate))
    ranking = []
    for candidate, records in grouped.items():
        difference = brier[candidate] - brier[reference]
        delta = float(difference.mean())
        se = float(difference.std(ddof=1) / np.sqrt(len(repeats))) if len(repeats) > 1 else 0.
        brier_eligible = delta <= se + 1e-12
        support_eligible = all(record["supported_mass"] >= .99 for record in records.values())
        first = records[repeats[0]]
        ranking.append({
            "candidate": candidate, "loss": first["loss"], "output_alpha": first["output_alpha"],
            "repeats": len(repeats), "mean_brier": float(brier[candidate].mean()),
            "mean_local_gap": float(np.mean([record["local_gap"] for record in records.values()])),
            "mean_supported_mass": float(np.mean([record["supported_mass"] for record in records.values()])),
            "minimum_supported_mass": min(record["supported_mass"] for record in records.values()),
            "paired_mean_brier_difference": delta, "paired_brier_difference_se": se,
            "brier_eligible": brier_eligible, "support_eligible": support_eligible,
            "eligible": brier_eligible and support_eligible,
        })
    ranking.sort(key=lambda row: (not row["eligible"], row["mean_local_gap"], row["mean_brier"], row["candidate"]))
    eligible = [row for row in ranking if row["eligible"]]
    if not eligible:
        raise ValueError("No Brier-shortlisted candidate has supported mass >= 0.99 in every repeat")
    chosen = {key: eligible[0][key] for key in ("candidate", "loss", "output_alpha")}
    return {"chosen": chosen, "ranking": ranking, "rule": SELECTION_RULE, "brier_reference": reference}
