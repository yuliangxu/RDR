"""CPU smoke run for the clipped feedforward ReLU estimator."""
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from experiments.AGP.mask_metrics import close
from experiments.AGP.relu_threshold_model import (
    ThresholdRDR, balanced_brier_rdr, balanced_js, endpoint_fractions,
)


def toy_rows(values, rare=False):
    # The first three taxa are absent, singleton-positive, and constant.
    rows = np.zeros((len(values), 6))
    rows[:, 2] = .125
    rows[:, 3] = values
    rows[:, 4] = np.resize([.0625, .125, .125, .1875], len(rows))
    if rare:
        rows[0, 1] = .0625
    rows[:, 5] = 1 - rows.sum(1)
    return close(2 * rows).astype(np.float32)


def test_cpu_fit_and_checkpoint():
    p = toy_rows(np.arange(1, 19) / 64, rare=True)
    q = toy_rows(np.arange(2, 13) / 32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(271)
        model = ThresholdRDR.from_training(p, q)
    assert model.scale[0].item() == 1.
    assert (model.taxon_index >= 3).all()
    for taxon in range(p.shape[1]):
        knots = model.thresholds[model.taxon_index == taxon].detach().numpy()
        assert len(knots) == len(np.unique(knots)) <= 8
    torch.testing.assert_close(model.thresholds[model.taxon_index == 4] * model.scale[4],
                               torch.tensor([.0625, .125]))
    assert model.additive.in_features == p.shape[1] + len(model.thresholds)
    assert model.interaction[0].out_features == model.interaction[2].out_features == 128
    p, q = torch.from_numpy(p), torch.from_numpy(q)
    p_stop = torch.from_numpy(toy_rows([.03, .11, .26]))
    q_stop = torch.from_numpy(toy_rows([.22, .28, .39, .40]))
    initial = model(torch.cat((p, q))).detach()
    assert ((initial > .9) & (initial < 1.1)).all()
    assert endpoint_fractions(initial, .01, 1.99) == (0., 0.)
    const_p, const_q = torch.ones(2), torch.ones(7)
    assert balanced_brier_rdr(const_p, const_q).item() == .25
    assert math.isclose(balanced_js(const_p, const_q).item(), math.log(2), rel_tol=1e-6)
    expected = -.5 * (math.log(.75) + .5 * (math.log(.9) + math.log(.8)))
    observed = balanced_js(torch.tensor([1.5]), torch.tensor([.2, .4])).item()
    assert math.isclose(observed, expected, rel_tol=1e-6)

    optimizer = torch.optim.Adam(model.parameters(), lr=.002)
    before_knots = model.thresholds.detach().clone()
    best = math.inf
    with tempfile.TemporaryDirectory(prefix="agp-relu-smoke-") as directory:
        path = Path(directory) / "checkpoint.pt"
        for epoch in range(6):
            optimizer.zero_grad()
            loss = balanced_js(model(p), model(q))
            assert torch.isfinite(loss)
            loss.backward()
            if epoch == 0:
                assert model.thresholds.grad.abs().sum().item() > 0.
            optimizer.step()
            with torch.no_grad():
                scores = model(torch.cat((p_stop, q_stop)))
                brier = balanced_brier_rdr(scores[:len(p_stop)], scores[len(p_stop):]).item()
            if brier < best:
                best, best_scores = brier, scores.clone()
                torch.save(model.state_dict(), path)
        assert not torch.equal(model.thresholds, before_knots)
        loaded = ThresholdRDR.from_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    with torch.no_grad():
        result = loaded(torch.cat((p_stop, q_stop)))
        torch.testing.assert_close(result, best_scores, rtol=0, atol=0)
        assert best == balanced_brier_rdr(result[:len(p_stop)], result[len(p_stop):]).item()
        assert ((result >= .01) & (result <= 1.99)).all()
        for intercept, clipping in [(-100., (1., 0.)), (100., (0., 1.))]:
            loaded.additive.bias.fill_(intercept)
            assert endpoint_fractions(loaded(p_stop), .01, 1.99) == clipping


def test_runner_full_batch_gradient_and_best_epoch(tmp_path, monkeypatch):
    """Unequal source and chunk sizes must give the complete empirical gradient."""
    import json
    from experiments.AGP import relu_threshold_experiment as experiment

    seed = 17
    # Opposing stopping labels force an early minimum as training separates P/Q.
    p = toy_rows(np.linspace(.30, .36, 19), rare=True) * 7.
    q = toy_rows(np.linspace(.05, .13, 11)) * 3.
    raw = dict(train_p=p, train_q=q, stopping_p=q[:4], stopping_q=p[-5:])
    cfg = json.loads(experiment.CONFIG.read_text())
    cfg.update(gradient_chunk_rows=6, min_epochs=5, max_epochs=5, learning_rate=.003)
    actual_gradients = []
    clip = torch.nn.utils.clip_grad_norm_

    def record_first_gradient(parameters, *args, **kwargs):
        parameters = list(parameters)
        if not actual_gradients:
            actual_gradients.extend(v.grad.detach().clone() for v in parameters)
        return clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_first_gradient)
    with torch.random.fork_rng(devices=[]):
        fitted, history, training = experiment.train_threshold(raw, cfg, seed, "cpu")
        torch.manual_seed(seed)
        p, q = (torch.from_numpy(close(raw["train_" + side]).astype(np.float32)) for side in ("p", "q"))
        full = ThresholdRDR.from_training(p.numpy(), q.numpy())
        probabilities = full(torch.cat((p, q))).double() / 2
        probability_p, probability_q = probabilities.split([len(p), len(q)])
        reference = -torch.log(probability_p).mean()/2 - torch.log1p(-probability_q).mean()/2
        reference.backward()
    assert abs(history[0]["training_objective"] - reference.item()) < 1e-7
    assert len(actual_gradients) == len(list(full.parameters()))
    for actual, param in zip(actual_gradients, full.parameters()):
        torch.testing.assert_close(actual, param.grad, atol=5e-8, rtol=1e-4)

    best_index = int(np.argmin([epoch["stopping_brier"] for epoch in history]))
    assert training["best_epoch"] == best_index + 1 < len(history)
    assert training["stopping_brier"] == history[best_index]["stopping_brier"]
    output = tmp_path / "threshold.pt"
    torch.save(fitted.state_dict(), output)
    loaded = experiment.load_model(output, "threshold", {}, {}, "cpu")
    before = experiment.scores(fitted, raw, "stopping", "threshold", "cpu")
    after = experiment.scores(loaded, raw, "stopping", "threshold", "cpu")
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    real, generated = (scores.astype(np.float64) / 2 for scores in after)
    brier = .5*((1-real)**2).mean() + .5*(generated**2).mean()
    assert abs(brier - training["stopping_brier"]) < 1e-12
