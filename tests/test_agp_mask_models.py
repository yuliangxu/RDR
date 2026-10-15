"""Scientific checks for independent AGP occurrence models."""
import itertools
import json

import numpy as np
import pytest
from scipy.special import expit
import torch

from experiments.AGP import mask_models as m


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    return {
        "seed": 92,
        "mixture": {"components": [1, 2], "starts": 1, "max_iter": 12, "patience": 4},
        "autoregressive": {"hidden": [4, 8], "max_epochs": 8, "patience": 3,
                           "batch_size": 32, "learning_rate": 0.02},
        "sampling": {"batch_size": 64},
    }


def example_masks():
    rng = np.random.default_rng(219)
    c = rng.random(200) < 0.4
    correlated = np.column_stack([c, c ^ (rng.random(200) < 0.03)])
    return np.column_stack([np.ones(200), correlated, np.zeros(200)]).astype(bool)


def test_mixture_likelihood_matches_enumeration_and_normalizes():
    x = np.array(list(itertools.product([0, 1], repeat=3)), dtype=float)
    weights = np.array([0.25, 0.75])
    p = np.array([[0.1, 0.2, 0.4], [0.8, 0.9, 0.3]])
    brute = np.array([sum(w * np.prod(np.where(row, component, 1 - component))
                         for w, component in zip(weights, p)) for row in x])
    computed = np.exp(m.mixture_log_prob(x, weights, p))
    np.testing.assert_allclose(computed, brute, rtol=1e-13)
    assert computed.sum() == pytest.approx(1)


def test_mixture_learns_dependence_beyond_independent_bernoulli():
    rng = np.random.default_rng(42)
    c = rng.random(1000) < 0.5
    x = np.column_stack([c, c ^ (rng.random(1000) < 0.02)]).astype(float)
    config = dict(m.DEFAULT_CONFIG["mixture"], max_iter=60, patience=10)
    fit = m._fit_mixture(x[:700], x[700:], 2, 5, config)
    independent = x[:700].mean(axis=0)[None, :]
    baseline = -m.mixture_log_prob(x[700:], np.ones(1), independent).mean()
    assert fit["valid_nll"] < baseline - 0.3
    population = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    probability = np.exp(m.mixture_log_prob(population, fit["weights"], fit["probabilities"]))
    assert probability[[0, 3]].sum() > 0.9


def test_nade_is_autoregressive_and_joint_distribution_normalizes():
    torch.manual_seed(73)
    model = m.BinaryNADE(4, 3).double()
    with torch.no_grad():
        model.input_weights.normal_()
        model.output_weights.normal_()
    x = torch.tensor(list(itertools.product([0.0, 1.0], repeat=4)), dtype=torch.float64)
    assert model.log_prob(x).exp().sum().item() == pytest.approx(1, abs=1e-12)
    x0 = torch.tensor([[0.3, 0.4, 0.6, 0.8]], dtype=torch.float64, requires_grad=True)
    logits = model(x0)
    for j in range(4):
        gradient, = torch.autograd.grad(logits[0, j], x0, retain_graph=True)
        torch.testing.assert_close(gradient[0, j:], torch.zeros_like(gradient[0, j:]), atol=1e-12, rtol=0)
    # Conditional probabilities match a direct sequential hidden-state calculation.
    hidden = model.hidden_bias.clone()
    for j in range(4):
        expected = torch.sigmoid(hidden) @ model.output_weights[j] + model.output_bias[j]
        torch.testing.assert_close(expected, logits[0, j])
        hidden = hidden + x0[0, j] * model.input_weights[j]


def test_nade_sampling_matches_known_conditionals():
    model = m.BinaryNADE(2, 1)
    with torch.no_grad():
        model.input_weights.copy_(torch.tensor([[8.0], [0.0]]))
        model.output_weights.copy_(torch.tensor([[0.0], [8.0]]))
        model.hidden_bias.fill_(-4)
        model.output_bias.copy_(torch.tensor([0.0, -4.0]))
    values = model.sample(20000, torch.Generator().manual_seed(7)).numpy()
    assert abs(values[:, 0].mean() - 0.5) < 0.015
    for first in (0, 1):
        expected = expit(8 * expit(-4 + 8 * first) - 4)
        assert abs(values[values[:, 0] == first, 1].mean() - expected) < 0.01


def test_schema_constants_and_order_use_only_supplied_training_rows():
    x = np.array([[1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]], dtype=bool)
    schema = m.mask_schema(x)
    assert schema["always_one"] == [0]
    assert schema["always_zero"] == [1]
    assert schema["active"] == [3, 2]
    np.testing.assert_array_equal(m._restore_schema(x[:, schema["active"]], schema), x)


def test_selection_refit_sampling_and_immutable_receipt(tmp_path):
    x = example_masks()
    output = tmp_path / "models"
    result = m.fit_and_select(x[:140], x[140:], x, output, tiny_config())
    assert result["status"] == "complete"
    assert result["sample_counts"] == dict(training=140, stopping=60, full_template=200)
    for method in ("bernoulli_mixture", "autoregressive"):
        metadata = json.loads((output / f"{method}.json").read_text())
        chosen = result["selected"][method]
        unit = "iterations" if method == "bernoulli_mixture" else "epochs"
        assert metadata[f"refit_{unit}"] == chosen[f"selected_{unit}"]
        assert metadata["training_rows"] == 200
    assert m.fit_and_select(x[:140], x[140:], x, output, tiny_config()) == result
    for method in m.METHODS:
        first, info = m.sample_model(output, method, 300, 42)
        second, _ = m.sample_model(output, method, 300, 42)
        np.testing.assert_array_equal(first, second)
        assert first.dtype == bool and first.shape == (300, 4)
        assert first[:, 0].all() and not first[:, 3].any()
        assert first.any(axis=1).all()
        assert info["proposals"] - info["rejected_empty"] == 300
        assert info["rejected_empty"] == 0
        if method == "bootstrap":
            assert set(map(bytes, first)) <= set(map(bytes, x))
    # A preserved receipt prevents silent mutation/reuse of a different study.
    with pytest.raises(ValueError, match="different input"):
        m.fit_and_select(x[:140], x[140:], x, output, dict(tiny_config(), seed=94))
    (output / "bootstrap.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact changed"):
        m.sample_model(output, "bootstrap", 10, 42)


def test_empty_masks_are_rejected_and_accounted_without_bit_flips(tmp_path):
    x = np.tile([[1, 0], [0, 1]], (30, 1)).astype(bool)
    cfg = tiny_config()
    cfg["mixture"].update(components=[1], max_iter=2)
    cfg["autoregressive"].update(hidden=[2], max_epochs=1)
    output = tmp_path / "models"
    m.fit_and_select(x[:40], x[40:], x, output, cfg)
    result, info = m.sample_model(output, "bernoulli_mixture", 1000, 123)
    assert result.any(axis=1).all()
    assert info["rejected_empty"] > 0
    assert info["proposals"] == 1000 + info["rejected_empty"]
    # Independent fair Bernoulli pairs conditioned on nonempty have 1/3 joint presences.
    assert abs(result.all(axis=1).mean() - 1 / 3) < 0.08


def test_support_violation_and_wrong_full_template_fail_before_writing(tmp_path):
    train = np.array([[1, 0], [1, 0]], dtype=bool)
    valid = np.array([[1, 1]], dtype=bool)
    with pytest.raises(ValueError, match="deterministic inner-training"):
        m.fit_and_select(train, valid, np.concatenate([train, valid]), tmp_path / "bad", tiny_config())
    assert not (tmp_path / "bad").exists()
    x = example_masks()
    with pytest.raises(ValueError, match="multiset union"):
        m.fit_and_select(x[:140], x[140:], x[:-1], tmp_path / "wrong", tiny_config())
    with pytest.raises(ValueError, match="all-zero"):
        m.validate_masks([[0, 0]])


def test_all_constant_nonempty_template_is_deterministic(tmp_path):
    x = np.tile([[0, 1, 0]], (10, 1)).astype(bool)
    cfg = tiny_config()
    cfg["mixture"].update(components=[1], max_iter=1)
    cfg["autoregressive"].update(hidden=[2], max_epochs=1)
    m.fit_and_select(x[:6], x[6:], x, tmp_path, cfg)
    for method in m.METHODS:
        sampled, info = m.sample_model(tmp_path, method, 7, 91)
        np.testing.assert_array_equal(sampled, np.tile(x[0], (7, 1)))
        assert info["rejected_empty"] == 0


def test_default_config_is_not_shared_and_unknown_settings_rejected():
    first = m.default_config()
    first["mixture"]["components"].append(999)
    assert 999 not in m.default_config()["mixture"]["components"]
    with pytest.raises(ValueError, match="Unknown"):
        m.normalized_config({"autoregressive": {"bad": True}})
