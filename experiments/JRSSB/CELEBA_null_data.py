"""Raw learned-only null predictions; no legacy aggregate or calibration gates."""
from pathlib import Path
import pandas as pd


def learned_null_specs(output_root, real_output_root):
    specs = []
    for level, symbol in (("feature", "Z"), ("pixel", "X")):
        for branch, label in (("lower", "L"), ("upper", "U")):
            specs.append(dict(label=rf"$Q_{{{label},{symbol}}}$ vs $Q_{{{label},{symbol}}}$",
                              root=Path(output_root) / "rdr" / f"pair_{level}_null_design" / branch,
                              prediction_name="design_predictions.csv", fold_column="null_fold"))
    specs.append(dict(label=r"$P_X$ vs $P_X$", root=Path(real_output_root) / "rdr/pixel/real_fold_null",
                      prediction_name="test_predictions.csv", fold_column="fold"))
    return specs


def load_row(spec):
    predictions = {}
    for repeat in range(5):
        path = spec["root"] / f"repeat_{repeat:02d}" / spec["prediction_name"]
        frame = pd.read_csv(path)
        if set(frame[spec["fold_column"]]) != {"A", "B"}:
            raise ValueError(f"Invalid null folds: {path}")
        predictions[repeat] = frame
    return predictions, {}
