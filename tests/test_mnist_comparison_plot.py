"""Selection and pixel-scale checks for newly rendered MNIST panels."""
import json

import numpy as np
import pytest
import torch

from experiments.MNIST.comparison_plot import display_images, image_mosaic, render, selection_indices


def test_selection_orders_extremes_and_breaks_all_ties_by_position():
    scores = np.array([1, 2, 0, 1, 2, 0, .5, 1.5])
    selected = selection_indices(scores, 5)
    assert selected["smallest"].tolist() == [2, 5, 6, 0, 3]
    assert selected["closest_to_one"].tolist() == [0, 3, 6, 7, 1]
    assert selected["largest"].tolist() == [1, 4, 7, 0, 3]
    assert len(selection_indices(scores)["largest"]) == len(scores)
    with pytest.raises(ValueError, match="finite"):
        selection_indices([np.nan])


def test_display_conversion_and_mosaic_preserve_actual_saved_pixels():
    real = torch.stack([torch.full((1, 28, 28), value, dtype=torch.uint8) for value in (0, 128, 255)])
    fake = torch.stack([torch.full((1, 28, 28), value, dtype=torch.float32) for value in (-1, 0, 1)])
    p, q = display_images(real, "p"), display_images(fake, "q")
    np.testing.assert_allclose(p[:, 0, 0], [0, 128 / 255, 1])
    np.testing.assert_array_equal(q[:, 0, 0], [0, .5, 1])
    mosaic = image_mosaic(q, [2, 1, 0])
    np.testing.assert_array_equal(mosaic[:28, :28], 1)
    np.testing.assert_array_equal(mosaic[:28, 28:56], .5)
    np.testing.assert_array_equal(mosaic[28:], 0)
    with pytest.raises(ValueError, match="uint8"):
        display_images(real.float(), "p")
    with pytest.raises(ValueError, match="\\[-1,1\\]"):
        display_images(fake + 1, "q")


def test_render_small_saved_bundles_including_absent_and_constant_digits(tmp_path):
    scores = torch.tensor([0, .5, 1, 1, 1.5, 2.])
    for name in ("vae", "dcgan"):
        folder = tmp_path / name
        folder.mkdir()
        torch.save({"p_images": torch.zeros((6, 1, 28, 28), dtype=torch.uint8),
                    "q_images": torch.zeros((6, 1, 28, 28)),
                    "p_scores": scores, "q_scores": scores.flip(0),
                    "p_labels": torch.tensor([0, 1, 2, 2, 3, 3]),
                    "p_indices": torch.arange(100, 106)}, folder / "evaluation.pt")
        p = scores.double().numpy()
        brier = .5 * np.mean((p / 2 - 1) ** 2) + .5 * np.mean((p / 2) ** 2)
        (folder / "metrics.json").write_text(json.dumps({"best_epoch": 1, "earlystop_brier": .2,
            "test_brier": brier, "loss": "js", "output_alpha": 2}))
    manifest = render(tmp_path)
    assert len(manifest["panels"]) == 12
    assert all(panel["n"] == 6 for panel in manifest["panels"])
    assert len(manifest["outputs"]) == 3
    assert manifest["vae"]["test_p_n"] == 6
    assert len((tmp_path / "panel_selection.csv").read_text().splitlines()) == 73
    assert (tmp_path / "figure_manifest.json").is_file()
