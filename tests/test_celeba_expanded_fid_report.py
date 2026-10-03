"""Pooled covariance and completion boundaries for full-pool CelebA FID."""

import csv

import numpy as np
import pytest

from experiments.CelebA import expanded_fid_report as report


def test_pooled_covariance_matches_concatenation_with_between_pool_variation():
    rng = np.random.default_rng(2026)
    a = rng.normal(size=(7, 3)) - 20
    b = rng.normal(size=(11, 3)) + 50
    expected = np.concatenate([a, b])
    mean, covariance = report.pooled_moments([a, b], 18, dimension=3, chunk_size=4)
    np.testing.assert_allclose(mean, expected.mean(axis=0), rtol=1e-14, atol=1e-13)
    np.testing.assert_allclose(covariance, np.cov(expected, rowvar=False, ddof=1), rtol=1e-14, atol=1e-12)
    # This would fail if within-pool covariances were merely averaged.
    assert covariance[0, 0] > 1000


def test_covariance_uses_ddof_one_and_is_chunk_size_independent():
    features = np.array([[0., 1.], [2., 5.], [4., 9.]])
    for chunk_size in (1, 2, 4096):
        mean, covariance = report.pooled_moments([features], 3, dimension=2, chunk_size=chunk_size)
        np.testing.assert_allclose(mean, [2., 5.])
        np.testing.assert_allclose(covariance, [[4., 8.], [8., 16.]])
        assert covariance.dtype == np.float64


def test_count_dimension_and_nonfinite_features_fail():
    with pytest.raises(ValueError, match="count mismatch"):
        report.pooled_moments([np.ones((3, 2))], 4, dimension=2)
    with pytest.raises(ValueError, match="columns"):
        report.pooled_moments([np.ones((3, 2))], 3, dimension=3)
    with pytest.raises(ValueError, match="nonfinite"):
        report.pooled_moments([np.array([[0., 1.], [np.nan, 2.]])], 2, dimension=2)


def test_feature_hash_and_cache_shape_are_verified_before_aggregation(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    path = tmp_path / "features.pt"
    torch.save({"features": torch.ones(3, 2), "role": "validation"}, path)
    monkeypatch.setattr(report, "DIMENSION", 2)
    digest = report.sha256(path)
    with pytest.raises(ValueError, match="hash differs"):
        report._features(path, "0" * 64, 3, [])
    with pytest.raises(ValueError, match="count/dimension"):
        report._features(path, digest, 4, [])
    with pytest.raises(ValueError, match="metadata differs"):
        report._features(path, digest, 3, [], {"role": "final"})
    receipts = []
    values = report._features(path, digest, 3, receipts, {"role": "validation"})
    assert values.shape == (3, 2)
    assert receipts[0]["sha256"] == digest


def test_extra_features_without_generation_receipts_remain_pending(tmp_path):
    (tmp_path / "features").mkdir()
    for branch in report.BRANCHES:
        (tmp_path / "features" / f"extra_validation_{branch}_2048.pt").write_bytes(b"unfinished")
    with pytest.raises(report.PendingInputs, match="completion|completed"):
        report._inputs(tmp_path, "validation", tmp_path / "unread_sources")


def test_combined_report_never_promotes_missing_roles_to_complete(tmp_path):
    report.write(tmp_path / "fid/train.json", {
        "status": "complete", "role": "train", "fid_upper_minus_lower": -.5,
        "absolute_fid_difference": .5,
        "results": {branch: {"fid": fid, "n_p": 122984, "n_q": 120000,
                              "mean_term": 1., "covariance_term": fid - 1}
                    for branch, fid in (("lower", 10.), ("upper", 9.5))},
    })
    assert not report.combined_report(tmp_path)
    text = (tmp_path / "FID_RESULTS.md").read_text()
    assert "incomplete; pending roles" in text
    assert "validation | pending | 39786 | 40000 | Pending" in text
    assert "final | pending | 39829 | 40000 | Pending" in text
    with (tmp_path / "FID_RESULTS.csv").open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 2
    assert {row["role"] for row in records} == {"train"}
