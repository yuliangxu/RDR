"""Compatibility import for the shared :mod:`utils.calibration` implementation.

New code should import ``utils.calibration``. This path remains available to
historical experiment scripts while the dataset folders are reorganized.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils import calibration as _calibration
from utils.calibration import (
    bin_scores,
    calibrate_counts,
    calibrate_merged_counts,
    contiguous_candidates,
    merge_adjacent_counts,
    variance_upper_bound,
)


def __getattr__(name):
    # Preserve explicit imports of historical internal helpers used by tests.
    return getattr(_calibration, name)
