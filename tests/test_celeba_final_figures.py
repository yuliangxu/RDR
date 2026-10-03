"""Image bins must preserve score endpoints and must not borrow sparse rows."""
import numpy as np
import pytest

from experiments.CelebA.final_figures import select_rows, same_cells, montage
from utils.calibration import bin_scores


def test_every_score_has_exactly_one_matching_ci_bin():
    edges = np.linspace(0, 2, 21)
    scores = np.r_[edges, np.nextafter(edges[1:-1], 0)]
    assigned = bin_scores(scores, edges)
    seen = []
    for i in range(20):
        selected, size = select_rows(scores, edges[i], edges[i+1], i == 19, 100, 9)
        assert size == np.sum(assigned == i)
        assert np.all(assigned[selected] == i)
        seen.extend(selected)
    assert sorted(seen) == list(range(len(scores)))


def test_empty_bins_stay_empty_and_nearest_examples_are_in_range():
    scores = np.array([0., .1, .8, 1.1, 1.8, 2.])
    selected, size = select_rows(scores, .2, .3, False, 40, 2)
    assert size == 0 and len(selected) == 0
    selected, size = select_rows(scores, .5, 1.5, False, 1, 2, target=1)
    assert size == 2 and scores[selected[0]] == 1.1
    selected, size = select_rows(scores, 1.5, 2., True, 40, 2, target=2)
    assert list(scores[selected]) == [2., 1.8]


def test_uniform_draw_reproducible_without_duplicates_and_no_fabricated_tiles():
    scores = np.linspace(0, .49, 100)
    first, _ = select_rows(scores, 0, .5, False, 4, 3)
    second, _ = select_rows(scores, 0, .5, False, 4, 3)
    np.testing.assert_array_equal(first, second)
    assert len(set(first)) == 4
    images = np.full((1, 3, 64, 64), 17, dtype=np.uint8)
    canvas = montage(images, [0], 4, 4)
    assert np.all(canvas[2:66, 2:66] == 17)
    assert np.all(canvas[:, 68:] == 245)


def test_unavailable_estimates_are_not_replaced_with_one():
    saved = [{'bin': 0, 'calibrated_rdr': None, 'c2_lower': 0., 'c2_upper': 2.}]
    same_cells(saved, saved)
    with pytest.raises(ValueError, match='Cell diagnostic differs'):
        same_cells(saved, [{**saved[0], 'calibrated_rdr': 1.}])
