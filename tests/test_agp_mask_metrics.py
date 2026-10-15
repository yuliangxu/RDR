"""Meaningful contracts for conventional source AUC and mask fidelity checks."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from experiments.AGP import mask_metrics as m


def sparse_bank(seed, n, d=6):
    rng = np.random.default_rng(seed)
    x = rng.exponential(size=(n, d))
    x[rng.uniform(size=x.shape) < .55] = 0
    x[:, 0] += .2
    return x/x.sum(1, keepdims=True)


def tiny_config(**extra):
    config = {"hgb": {"max_iter": 6, "checkpoint_every": 2, "min_samples_leaf": 3,
                       "max_leaf_nodes": 5, "max_bins": 16},
              "logistic": {"max_iter": 150}, "make_plots": False}
    config.update(extra)
    return config


def banks():
    real = {role: sparse_bank(i, n) for i, (role, n) in enumerate((("train", 40), ("stop", 20), ("test", 18)))}
    generated = {method: {role: sparse_bank(100+i*20+j, len(x)+2) for j, (role, x) in enumerate(real.items())}
                 for i, method in enumerate(m.METHODS)}
    generated["midasim_mask_icfm"] = {role: m.close(x**1.2) for role, x in generated["midasim"].items()}
    subjects = {role: np.array([f"{role}_s{i//2}" for i in range(len(x))]) for role, x in real.items()}
    return real, generated, subjects


def test_close_and_feature_map_preserve_zero_semantics():
    x = np.array([[0., 2., 3.], [4., 0., 1.]])
    np.testing.assert_allclose(m.close(x).sum(1), 1)
    np.testing.assert_array_equal(m.features(x, "hgb_mask"), [[0, 1, 1, 2], [1, 0, 1, 2]])
    f = m.features(x, "hgb_composition")
    np.testing.assert_array_equal(f[:, 3:], x > 0)
    np.testing.assert_allclose((f[:, :3]**2).sum(1), 1, atol=1e-7)
    for bad in (np.zeros((3, 2)), [[1., -1.], [2., 1.]], [[np.nan, 1.], [2., 1.]]):
        with pytest.raises(ValueError):
            m.close(bad)
    with pytest.raises(ValueError):
        m.features(x, "rdr")


def test_balanced_source_weights_and_auc_direction():
    _, y, weights = m.binary_pair(np.ones((3, 2)), np.zeros((7, 2)))
    assert weights[y == 1].sum() == pytest.approx(weights[y == 0].sum())
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y, np.r_[np.ones(3), np.zeros(7)]) == 1
    assert roc_auc_score(y, np.full(10, .5)) == .5
    assert roc_auc_score(y, np.r_[np.zeros(3), np.ones(7)]) == 0


def test_null_partition_preserves_subjects_and_is_reproducible():
    subjects = np.array(["a", "a", "b", "c", "c", "d", "e", "f", "f"])
    a, b = m.null_partition(len(subjects), 52, subjects)
    assert not (set(subjects[a]) & set(subjects[b]))
    assert set(a) | set(b) == set(range(len(subjects)))
    for first, second in zip((a, b), m.null_partition(len(subjects), 52, subjects)):
        np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        m.null_partition(5, 12, [1, 1, 1, 2, 2])


def test_exact_null_selects_constant_and_replays_probabilities():
    train, stop = sparse_bank(43, 30), sparse_bank(44, 16)
    fit = m.fit_classifier(train, train, stop, stop, "hgb_mask", m.merged_config(tiny_config()))
    assert fit["selected_iteration"] == 0
    assert fit["stop_balanced_log_loss"] == pytest.approx(np.log(2))
    np.testing.assert_array_equal(m.predict_classifier(fit, stop), .5)


def test_predict_uses_selected_stage_not_final_stage():
    class Stages:
        def staged_predict_proba(self, x):
            for p in (.2, .3, .8):
                yield np.tile([1-p, p], (len(x), 1))
    fit = {"recipe": "hgb_mask", "selected_iteration": 2, "model": Stages()}
    np.testing.assert_array_equal(m.predict_classifier(fit, sparse_bank(15, 8)), .3)


def test_fidelity_identity_and_full_generator_reference_novelty():
    p = np.array([[.2, .8, 0], [.8, 0, .2], [.1, .4, .5], [1., 0, 0]])
    result = m.fidelity_metrics(p, p, p)
    for key in ("richness_ks", "shannon_ks", "prevalence_mae", "mean_abundance_mae", "occurrence_correlation_mae", "mask_novel_fraction"):
        assert result[key] == pytest.approx(0)
    train, stop = p[:2], p[2:]
    assert m.fidelity_metrics(stop, stop, train)["mask_novel_fraction"] == 1
    with_full = m.fidelity_metrics(stop, stop, train, mask_reference=p)
    assert with_full["mask_novel_fraction"] == 0
    assert with_full["flow_training_richness_min"] == 1
    assert with_full["flow_training_richness_max"] == 3


def test_correlation_constant_taxa_are_explicitly_counted():
    p = np.array([[1., 0, 0], [0, 1., 0], [.5, 0, .5], [0, .5, .5]])
    q = np.array([[1., 0, 0], [0, 1., 0], [.5, .5, 0], [.2, .8, 0]])
    result = m.fidelity_metrics(p, q, p)
    assert result["occurrence_pairs_possible"] == 3
    assert result["occurrence_pairs_compared"] == 1
    assert result["constant_q_taxa_in_training_variable_set"] == 1
    assert result["prevalence_mae"] > 0


def test_preparation_never_reads_test_and_assessment_requires_freeze(tmp_path):
    real, generated, subjects = banks()
    class Guarded(dict):
        def __getitem__(self, key):
            if key == "test":
                raise AssertionError("preparation touched assessment")
            return super().__getitem__(key)
    r = Guarded(real)
    g = {name: Guarded(bank) for name, bank in generated.items()}
    s = Guarded(subjects)
    freeze = m.prepare(r, g, tmp_path/"run", tiny_config(recipes=["hgb_mask"]), real_subjects=s)
    assert freeze["n_classifiers"] == 6
    assert freeze["no_test_data_used"]
    assert freeze["null_partition_unit"] == "subject"
    result = m.evaluate(real, generated, tmp_path/"run", real_subjects=subjects)
    assert result["rdr_fitted_or_evaluated"] is False
    assert all(result["checks"].values())
    with pytest.raises(FileExistsError):
        m.evaluate(real, generated, tmp_path/"run", real_subjects=subjects)


def test_full_small_comparison_and_prediction_metric_replay(tmp_path):
    real, generated, subjects = banks()
    result = m.assess(real, generated, tmp_path/"run", tiny_config(make_plots=True), real_subjects=subjects)
    assert result["n_classifiers"] == 18
    frame = pd.read_csv(tmp_path/"run/summary/classifier_metrics.csv")
    composition = pd.read_csv(tmp_path/"run/summary/fidelity_metrics.csv")
    assert len(composition) == 5
    assert composition.mask_reference_rows.eq(60).all()
    assert (tmp_path/"run/summary/mask_comparison.pdf").stat().st_size > 1000
    from sklearn.metrics import roc_auc_score
    for row in frame.itertuples():
        with np.load(tmp_path/"run/private"/f"{row.method}__{row.recipe}_test_predictions.npz") as prediction:
            values = np.r_[prediction["p"], prediction["q"]]
            labels = np.r_[np.ones(len(prediction["p"])), np.zeros(len(prediction["q"]))]
            assert row.auc == pytest.approx(roc_auc_score(labels, values))
    fit = joblib.load(tmp_path/"run/private/midasim__hgb_composition.joblib")
    with np.load(tmp_path/"run/private/midasim__hgb_composition_test_predictions.npz") as prediction:
        np.testing.assert_array_equal(m.predict_classifier(fit, real["test"]), prediction["p"])


def test_freeze_guards_config_and_training_data(tmp_path):
    real, generated, subjects = banks()
    subset = {"bootstrap_icfm": generated["bootstrap_icfm"]}
    out = tmp_path/"run"
    m.prepare(real, subset, out, tiny_config(recipes=["hgb_mask"]), real_subjects=subjects)
    previous = real["train"].copy()
    real["train"][0] = np.roll(real["train"][0], 1)
    with pytest.raises(ValueError, match="inputs changed"):
        m.evaluate(real, subset, out, real_subjects=subjects)
    real["train"] = previous
    config = json.loads((out/"config.json").read_text())
    config["seed"] += 1
    (out/"config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration changed"):
        m.evaluate(real, subset, out, real_subjects=subjects)


def test_mismatched_null_unit_and_unknown_configuration_fail(tmp_path):
    real, generated, subjects = banks()
    subset = {"bootstrap_icfm": generated["bootstrap_icfm"]}
    out = tmp_path/"run"
    m.prepare(real, subset, out, tiny_config(recipes=["hgb_mask"]), real_subjects=subjects)
    with pytest.raises(ValueError, match="subject/row convention"):
        m.evaluate(real, subset, out)
    with pytest.raises(ValueError, match="unknown classifier"):
        m.merged_config({"test_tuning": True})
    with pytest.raises(ValueError, match="unknown hgb"):
        m.merged_config({"hgb": {"test_tuning": True}})
