"""Scientific protocol checks for the selected perturbation/null refits."""
import json

import numpy as np
import pytest
import torch

from experiments.MNIST import controls as c


def labels():
    return torch.arange(60000) % 10, torch.arange(10000) % 10


def original_sampler(labels, source, probs, n, seed):
    """Independent expression of the historical position-based sampler."""
    source = list(source)
    source_labels = labels[source]
    rng = torch.Generator().manual_seed(seed)
    selected_labels = torch.multinomial(probs, n, replacement=True, generator=rng)
    sampled = torch.empty(n, dtype=torch.long)
    for digit in range(10):
        positions = torch.where(selected_labels == digit)[0]
        if not len(positions):
            continue
        choices = torch.where(source_labels == digit)[0]
        sampled[positions] = choices[torch.randint(len(choices), (len(positions),), generator=rng)]
    return [source[position] for position in sampled]


def test_historical_primary_rows_and_q_sampler_are_preserved():
    train, test = labels()
    before = torch.get_rng_state().clone()
    splits = c.make_splits(train, test)
    assert torch.equal(before, torch.get_rng_state())
    train_order = torch.randperm(60000, generator=torch.Generator().manual_seed(123)).tolist()
    test_order = torch.randperm(10000, generator=torch.Generator().manual_seed(133)).tolist()
    perturb, null = splits["perturbation"], splits["null"]
    for role, source, p, seed in (("train", train, list(range(60000)), 143),
                                 ("validation", test, test_order[:5000], 153),
                                 ("test", test, test_order[5000:], 163)):
        assert perturb[role]["p"] == p
        assert perturb[role]["q"] == original_sampler(source, p, c.Q_PROBS, len(p), seed)
        assert set(perturb[role]["q"]).issubset(p)
    assert set(perturb["validation"]["p"]).isdisjoint(perturb["test"]["p"])
    assert null["train"]["p"] == train_order[:30000]
    assert null["train"]["q"] == train_order[30000:]
    assert null["validation"]["p"] == test_order[:5000]
    assert null["validation"]["q"] == test_order[5000:]
    assert "test" not in null
    assert null["reference"]["indices"] == list(range(10000))


def test_fresh_roles_have_private_reproducible_streams_and_controlled_support():
    _, test = labels()
    cfg = {"seed": 12345, "diagnostic_n": 400}
    reference = list(range(100))
    seeds = [c.stream_seed(cfg, name, role) for name in c.NAMES for role in
             ("training_p", "training_q", "calibration_p", "calibration_q", "evaluation_p", "evaluation_q")]
    assert len(set(seeds)) == 12
    before = torch.get_rng_state().clone()
    draws = c.diagnostic_indices(cfg, "perturbation", test, reference)
    assert torch.equal(before, torch.get_rng_state())
    assert draws == c.diagnostic_indices(cfg, "perturbation", test, reference)
    for role in draws:
        for side in draws[role]:
            assert set(draws[role][side]).issubset(reference)
            assert len(draws[role][side]) == 400
        assert not ((test[draws[role]["q"]] == 0) | (test[draws[role]["q"]] == 1)).any()
    assert draws["calibration"] != draws["evaluation"]
    # Shared identities are expected; only random draws are independent given
    # the fixed reference pool, which the report must explicitly state.
    assert set(draws["calibration"]["q"]) & set(draws["evaluation"]["q"])
    null = c.diagnostic_indices(cfg, "null", test, reference)
    assert set(test[null["calibration"]["q"]].tolist()) == set(range(10))


def test_nominal_and_empirical_reference_are_distinct_without_extra_risk_metrics():
    indices = list(range(10))
    labels = torch.arange(10)
    score = c.score_record(np.linspace(0, 2, 10), labels, indices, "official_test")
    record = {"test": {"p": score, "q": score}}
    reference_labels = torch.tensor([0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    rows = c.digit_rows(record, "perturbation", reference_labels)
    nominal = [row["nominal_reference"] for row in rows[:10]]
    np.testing.assert_allclose(nominal, [2, 2, 1.6, 1.6, 1, 1, 1, 1, 8/15, 8/15])
    assert rows[2]["empirical_reference"] != rows[2]["nominal_reference"]
    assert not any("mse" in key or "mae" in key for row in rows for key in row)


@pytest.mark.parametrize("name", c.NAMES)
def test_checkpoint_saved_and_restored_before_final_or_fresh_diagnostics(tmp_path, monkeypatch, name):
    """A worsening final epoch must be rejected before scoring any new roles."""
    torch.set_num_threads(1)
    cfg = {"seed": 99, "batch_size": 2, "max_epochs": 3, "min_epochs": 3,
           "patience": 3, "bn_freeze_epoch": 5, "max_lr": .001,
           "diagnostic_n": 20, "ci_alpha": .05, "bins": 20}
    chosen = {"candidate": "js_a2", "loss": "js", "output_alpha": 2.}
    protocol = {"config": cfg, "chosen": chosen, "selection_sha256": "fake", "smoke": True}
    monkeypatch.setattr(c, "load_protocol", lambda output: protocol)
    (tmp_path / "protocol.json").write_text("{}")
    split = {"train": {"source": "official_train", "p": [0, 1], "q": [2, 3]},
             "validation": {"source": "official_test", "p": [0, 1], "q": [2, 3]},
             "reference": {"source": "official_test", "indices": list(range(10))}}
    if name == "perturbation":
        split["test"] = {"source": "official_test", "p": [4, 5], "q": [6, 7]}
    (tmp_path / "splits.json").write_text(json.dumps({name: split}))
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 1))
    monkeypatch.setattr(c.ms, "make_model", lambda *args: model)
    reads = []
    def images(output, source, indices):
        if len(reads) >= 4:
            assert (tmp_path / name / "model.pt").is_file()
            saved = torch.load(tmp_path / name / "model.pt", weights_only=True)
            assert saved["best_epoch"] == 2
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, saved["model"][key], atol=0, rtol=0)
        reads.append(list(indices))
        return torch.full((len(indices), 1, 28, 28), 127, dtype=torch.uint8)
    monkeypatch.setattr(c, "read_images", images)
    monkeypatch.setattr(c, "read_labels", lambda path: torch.arange(100) % 10)
    calls = []
    def predict(model, images, alpha, batch_size, device):
        calls.append(1)
        epoch = (len(calls)-1)//2
        side = (len(calls)-1) % 2
        if epoch < 4:
            # Epochs1,2,3 then restored checkpoint: Brier .16,.09,.1225,.09.
            p = [.8, .6, .7, .6][epoch]
            return np.full(len(images), p if side else 2-p)
        return np.full(len(images), 1.)
    monkeypatch.setattr(c.ms, "predict", predict)
    metrics = c.train_branch(tmp_path, name, torch.device("cpu"))
    assert metrics["best_epoch"] == 2
    assert metrics["has_independent_final_role"] == (name == "perturbation")
    assert metrics["test_brier"] is None if name == "null" else metrics["test_brier"] is not None
    assert "not population" in metrics["diagnostic_scope"]
    assert c.verify_branch(tmp_path, name) == metrics


def test_sampler_rejects_absent_positive_probability_digit():
    with pytest.raises(ValueError, match="No source images"):
        c.sample_indices(torch.zeros(5, dtype=torch.long), range(5), c.Q_PROBS, 10, 1)
