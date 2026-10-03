"""Scientific target, paired training, and frozen-checkpoint regression checks."""

import json

import numpy as np
import pytest
import torch

from experiments.CelebA import selection_training as training
from utils.losses import midpoint_loss_from_logits, midpoint_rdr_loss


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def config():
    return {"seed": 717, "bins": 4, "ci_alpha": .05,
            "feature": {"input_dimension": 3, "hidden_dimension": 8,
                        "small_hidden_dimension": 4, "batch_size": 4,
                        "evaluation_batch_size": 8, "max_epochs": 3,
                        "min_epochs": 1, "patience": 2},
            "pixel": {"ndf": 2, "small_ndf": 1, "batch_size": 4,
                      "evaluation_batch_size": 8, "max_epochs": 2,
                      "min_epochs": 1, "patience": 2, "bn_freeze_epoch": 1}}


def arrays_for(representation="feature"):
    rng = np.random.default_rng(29)
    shape = (3,) if representation == "feature" else (3, 64, 64)
    arrays = {}
    for role in ("train", "earlystop", "selection_calibration", "selection_evaluation",
                 "test_calibration", "test_evaluation"):
        for side in ("p", "q"):
            n = (7 if side == "p" else 5) if role == "train" else (5 if side == "p" else 3)
            if representation == "feature":
                values = rng.normal(1 if side == "p" else -1, 1, (n, *shape)).astype(np.float32)
            else:
                values = rng.integers(0, 256, (n, *shape), dtype=np.uint8)
            arrays[f"{role}_{side}"] = values
    return arrays


def task_for(representation="feature", **kwargs):
    return {"representation": representation, "architecture": "baseline", "branch": "upper",
            "repeat": 0, "candidate": "baseline_js_alpha1", "output_alpha": 1.,
            "loss": "js", "task_index": 0, **kwargs}


@pytest.mark.parametrize("loss", ["hellinger", "kl", "chisq", "js"])
def test_midpoint_logit_loss_agrees_with_ratio_loss_and_has_correct_target(loss):
    # Unequal source sample sizes encode P=(3/4,1/4), Q=(1/4,3/4).
    p = torch.tensor([0, 0, 0, 1])
    q = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    logits = torch.logit(torch.tensor([.75, .25], dtype=torch.float64)).requires_grad_()

    class Lookup(torch.nn.Module):
        def forward(self, indices):
            return 2 * logits[indices].sigmoid()

    risk = midpoint_loss_from_logits(logits[p], logits[q], loss)
    torch.testing.assert_close(risk, midpoint_rdr_loss(Lookup(), p, q, loss), atol=1e-13, rtol=1e-13)
    risk.backward()
    torch.testing.assert_close(logits.grad, torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
    perturbed = logits.detach() + torch.tensor([.2, -.1])
    assert midpoint_loss_from_logits(perturbed[p], perturbed[q], loss) > risk


@pytest.mark.parametrize("loss", ["kl", "js"])
def test_saturated_outputs_retain_finite_logit_gradients(loss):
    p, q = [torch.tensor([v], requires_grad=True) for v in (-1000., 1000.)]
    risk = midpoint_loss_from_logits(p, q, loss)
    assert torch.isfinite(risk)
    risk.backward()
    assert torch.isfinite(p.grad).all() and p.grad.item() < 0
    assert torch.isfinite(q.grad).all()
    if loss == "js":
        assert q.grad.item() > 0


@pytest.mark.parametrize("representation", ["feature", "pixel"])
def test_raw_head_and_paired_initialization(config, representation):
    before = torch.get_rng_state().clone()
    a = training.make_model(config, 0, representation)
    b = training.make_model(config, 0, representation)
    torch.testing.assert_close(torch.get_rng_state(), before)
    for name, parameter in a.state_dict().items():
        torch.testing.assert_close(parameter, b.state_dict()[name], rtol=0, atol=0)
    small = training.make_model(config, 0, representation, "small")
    assert sum(p.numel() for p in small.parameters()) < sum(p.numel() for p in a.parameters())
    x = torch.zeros(2, 3) if representation == "feature" else torch.zeros(2, 3, 64, 64)
    a.eval()
    if representation == "pixel":
        assert isinstance(a.out_act, torch.nn.Identity)
        torch.testing.assert_close(a(x), a.head(a.backbone(x)).flatten())
    else:
        assert isinstance(a.model[-1], torch.nn.Identity)
    expected = (2 * torch.sigmoid(4 * a(x))).detach().double().numpy()
    np.testing.assert_allclose(training.predict(a, x, 4, 2, "cpu"), expected, rtol=0, atol=0)


@pytest.mark.parametrize("n_p,n_q,batch_size", [(13, 9, 4), (1, 19, 4), (2, 3, 16)])
def test_unequal_epoch_uses_every_observation_once(n_p, n_q, batch_size):
    def epoch():
        return list(training.paired_epoch_indices(n_p, n_q, batch_size,
                    torch.Generator().manual_seed(3), torch.Generator().manual_seed(4)))
    first, second = epoch(), epoch()
    for side, n in ((0, n_p), (1, n_q)):
        joined = torch.cat([batch[side] for batch in first])
        assert all(len(batch[side]) > 0 for batch in first)
        torch.testing.assert_close(joined.sort().values, torch.arange(n))
        torch.testing.assert_close(joined, torch.cat([batch[side] for batch in second]))


def test_scaler_matches_pooled_training_moments_and_constant_columns(config):
    p = np.array([[0., 4., 1.], [2., 4., 3.]], dtype=np.float32)
    q = np.array([[7., 4., -2.]], dtype=np.float32)
    model = training.make_model(config, 0, "feature")
    metadata = training.fit_feature_scaler(model, p, q, batch_size=1)
    pooled = np.concatenate((p, q)).astype(np.float64)
    expected_scale = pooled.std(0)
    expected_scale[expected_scale == 0] = 1
    np.testing.assert_allclose(model.input_mean.numpy(), pooled.mean(0), atol=1e-6)
    np.testing.assert_allclose(model.input_scale.numpy(), expected_scale, atol=1e-6)
    assert metadata["n_p"] == 2 and metadata["n_q"] == 1


def test_fit_restores_absolute_minimum_and_evaluation_uses_frozen_scaler(config, tmp_path, monkeypatch):
    arrays, task = arrays_for(), task_for()
    actual_predict = training.predict
    early_states = []
    current = {"epoch": -1}

    def controlled_earlystop(model, x, alpha, batch_size, device, representation=None):
        if x is arrays["earlystop_p"]:
            current["epoch"] += 1
            early_states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if x is arrays["earlystop_p"] or x is arrays["earlystop_q"]:
            # The final call checks restoration; epoch 1 must beat epochs 2/3.
            mistake = [.2, .4, 1., .2][current["epoch"]]
            return np.full(len(x), 2 - mistake if x is arrays["earlystop_p"] else mistake)
        return actual_predict(model, x, alpha, batch_size, device, representation)

    monkeypatch.setattr(training, "predict", controlled_earlystop)
    result = training.fit_model(task, config, arrays, tmp_path / "fit", "cpu")
    assert result["best_epoch"] == 1 and result["epochs"] == 3
    assert result["earlystop_brier"] == pytest.approx(.01)
    assert result["stopping_reason"] == "patience"
    checkpoint = torch.load(tmp_path / "fit/model.pt", weights_only=True)
    for name, value in early_states[0].items():
        torch.testing.assert_close(checkpoint["model"][name], value, rtol=0, atol=0)
        torch.testing.assert_close(early_states[-1][name], value, rtol=0, atol=0)
    history = json.loads((tmp_path / "fit/history.json").read_text())
    assert all(row["train_p_used"] == 7 and row["train_q_used"] == 5 for row in history)
    assert all(np.isfinite(row["train_brier"]) for row in history)
    # Huge test shifts must not change the scaler saved with the trained model.
    arrays["test_calibration_p"] += 100
    # Evaluation scheduling assigns a new index without changing the fit.
    evaluated = training.evaluate_model({**task, "task_index": 999}, config, arrays, tmp_path / "fit", tmp_path / "eval", "cpu")
    assert np.isfinite(evaluated["whole_test_brier"])
    assert evaluated["scaler"] == result["scaler"]
    frozen = torch.load(tmp_path / "fit/model.pt", weights_only=True)
    torch.testing.assert_close(frozen["model"]["input_mean"], checkpoint["model"]["input_mean"])
    with pytest.raises(ValueError, match="task or configuration"):
        training.evaluate_model({**task, "branch": "lower"}, config, arrays, tmp_path / "fit", tmp_path / "bad", "cpu")
    for field, value in (("output_alpha", 2), ("loss", "hellinger")):
        with pytest.raises(ValueError, match="task or configuration"):
            training.evaluate_model({**task, field: value}, config, arrays, tmp_path / "fit", tmp_path / "bad", "cpu")


def test_pixel_fit_mixes_sources_once_per_step_and_freezes_bn(config, tmp_path, monkeypatch):
    arrays, task = arrays_for("pixel"), task_for("pixel")
    original_make = training.make_model
    captured, training_batch_sizes = {}, []

    def capture(*args, **kwargs):
        model = original_make(*args, **kwargs)
        captured["model"] = model
        model.register_forward_pre_hook(lambda module, inputs:
            training_batch_sizes.append(len(inputs[0])) if module.training else None)
        return model

    monkeypatch.setattr(training, "make_model", capture)
    metrics = training.fit_model(task, config, arrays, tmp_path, "cpu")
    assert metrics["epochs"] == 2
    assert training_batch_sizes == [7, 5, 7, 5]
    assert all(not parameter.requires_grad for module in captured["model"].modules()
               if isinstance(module, torch.nn.BatchNorm2d) for parameter in module.parameters())
    checkpoint = torch.load(tmp_path / "model.pt", weights_only=True)
    # Whichever epoch is restored, stats received exactly the two epoch-1 forwards.
    tracked = [int(value) for name, value in checkpoint["model"].items() if name.endswith("num_batches_tracked")]
    assert tracked and tracked == [2] * len(tracked)


def test_ten_step_pixel_smoke_avoids_zero_length_onecycle_warmup(config, tmp_path):
    config["pixel"].update(max_epochs=5, min_epochs=5, patience=5)
    metrics = training.fit_model(task_for("pixel"), config, arrays_for("pixel"), tmp_path, "cpu")
    assert metrics["epochs"] == 5
    assert metrics["effective_onecycle_pct_start"] == .2
