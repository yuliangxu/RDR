"""Scientific target and artifact-gating checks for DCGAN model selection."""

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.MNIST import model_selection as m
from utils.losses import midpoint_rdr_loss


@pytest.fixture
def config():
    cfg = m.read(m.DEFAULT_CONFIG)
    cfg.update(losses=["hellinger", "js"], output_alphas=[.5, 1], repeats=2,
               train_n=8, earlystop_n=4, selection_calibration_n=4,
               selection_evaluation_n=4, final_calibration_n=4, final_evaluation_n=4,
               batch_size=4, max_epochs=3, min_epochs=1, patience=2, base_channels=4)
    return cfg


@pytest.fixture
def protocol_run(tmp_path, config):
    output = tmp_path / "study"
    output.mkdir()
    source = output / "source/experiments/MNIST/model_selection.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(Path(m.__file__).read_bytes())
    grid = m.candidates(config)
    tasks = [{"task_index": index, "repeat": repeat, **candidate}
             for index, (repeat, candidate) in enumerate((r, c) for r in range(config["repeats"]) for c in grid)]
    protocol = {"schema": 1, "config": config, "candidates": grid, "tasks": tasks,
                "files": {str(source.relative_to(output)): {"sha256": m.sha(source)}}}
    m.write(output / "protocol.json", protocol)
    (output / "protocol.sha256").write_text(m.sha(output / "protocol.json") + "\n")
    return output, protocol


@pytest.mark.parametrize("loss", ["hellinger", "kl", "chisq", "js"])
def test_logit_risks_agree_with_shared_midpoint_risks_and_correct_target(loss):
    # P=(3/4,1/4), Q=(1/4,3/4), hence r0=(1.5,.5). Q has twice as many
    # observations, checking that each expectation retains its own denominator.
    class Lookup(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.logit(torch.tensor([.75, .25], dtype=torch.float64)))

        def forward(self, indices):
            return (2 * self.logits[indices].sigmoid()).reshape(-1, 1)

    model = Lookup()
    p, q = torch.tensor([0, 0, 0, 1]), torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    risk = m.midpoint_loss_from_logits(model.logits[p], model.logits[q], loss)
    reference = midpoint_rdr_loss(model, p, q, loss)
    torch.testing.assert_close(risk, reference, atol=1e-13, rtol=1e-13)
    risk.backward()
    torch.testing.assert_close(model.logits.grad, torch.zeros(2, dtype=torch.float64), atol=1e-14, rtol=0)
    perturbed = model.logits.detach() + torch.tensor([.2, -.1])
    assert m.midpoint_loss_from_logits(perturbed[p], perturbed[q], loss) > risk


@pytest.mark.parametrize("loss", ["kl", "js"])
def test_stable_logit_risks_keep_gradients_at_saturated_sigmoid_outputs(loss):
    p = torch.tensor([-1000.], requires_grad=True)
    q = torch.tensor([1000.], requires_grad=True)
    risk = m.midpoint_loss_from_logits(p, q, loss)
    assert torch.isfinite(risk)
    risk.backward()
    assert torch.isfinite(p.grad).all() and p.grad.item() < 0
    assert torch.isfinite(q.grad).all()
    if loss == "js":
        assert q.grad.item() > 0


def test_initialization_matches_across_slopes_and_restores_cpu_rng(config):
    torch.set_num_threads(1)
    before = torch.get_rng_state().clone()
    a = m.make_model(config, 0, .5, torch.device("cpu"))
    assert torch.equal(torch.get_rng_state(), before)
    b = m.make_model(config, 0, 4, torch.device("cpu"))
    assert a.state_dict().keys() == b.state_dict().keys()
    for name, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[name], atol=0, rtol=0)
    a.eval()
    images = torch.ones(3, 1, 28, 28)
    logits = a(images).flatten().detach()
    actual = m.predict(a, images, 4, 2, torch.device("cpu"))
    np.testing.assert_allclose(actual, (2 * (4 * logits).sigmoid()).numpy(), atol=1e-7)


def test_real_role_assignments_and_generated_seeds_are_disjoint(config):
    splits = m.make_splits(config)
    assert splits == m.make_splits(config)
    for source, limit in (("official_train", 60000), ("official_test", 10000)):
        role_sets = [set(split["indices"]) for split in splits.values() if split["source"] == source]
        assert sum(map(len, role_sets)) == len(set.union(*role_sets))
        assert all(0 <= index < limit for index in set.union(*role_sets))
    for role, split in splits.items():
        assert len(split["indices"]) == config[f"{role}_n"]
    seeds = [m.seed_for(config, repeat, role) for repeat in range(config["repeats"]) for role in m.ROLE_IDS]
    assert len(seeds) == len(set(seeds))
    assert all(m.seed_for(config, 0, role) == m.seed_for(config, 0, role) for role in m.ROLE_IDS)


def test_evaluation_refuses_before_selection_without_accessing_final_images(protocol_run, monkeypatch):
    output, _ = protocol_run
    def forbidden(*args, **kwargs):
        pytest.fail("Final data/model access occurred before selection was frozen")
    monkeypatch.setattr(m, "real_images", forbidden)
    monkeypatch.setattr(m, "build_dcgan28", forbidden)
    monkeypatch.setattr(m, "make_model", forbidden)
    with pytest.raises(FileNotFoundError):
        m.evaluate(output, 0, torch.device("cpu"))
    assert not (output / "final").exists()


def test_freeze_refuses_missing_fit_grid(protocol_run):
    output, _ = protocol_run
    with pytest.raises(FileNotFoundError):
        m.freeze(output)
    assert not (output / "selection.json").exists()


def test_protocol_and_frozen_source_tampering_are_rejected(protocol_run):
    output, protocol = protocol_run
    m.load_protocol(output)
    original = (output / "protocol.json").read_bytes()
    changed = copy.deepcopy(protocol)
    changed["config"]["seed"] += 1
    m.write(output / "protocol.json", changed)
    with pytest.raises(ValueError, match="Protocol changed"):
        m.load_protocol(output)
    (output / "protocol.json").write_bytes(original)
    frozen = output / next(iter(protocol["files"]))
    frozen.write_text(frozen.read_text() + "\n# altered frozen source\n")
    with pytest.raises(ValueError, match="Frozen input changed"):
        m.load_protocol(output)


def test_complete_result_metrics_are_hash_protected(protocol_run):
    output, _ = protocol_run
    folder = output / "runs/fake/repeat_00"
    folder.mkdir(parents=True)
    for name in ("model.pt", "predictions.npz", "cells.json", "history.json"):
        (folder / name).write_text("test fixture\n")
    m.write(folder / "metrics.json", {"brier": .1})
    names = ("model.pt", "predictions.npz", "metrics.json", "cells.json", "history.json")
    m.complete(folder, output, names)
    assert m.verify_result(folder, output)["brier"] == .1
    m.write(folder / "metrics.json", {"brier": .01})
    with pytest.raises(ValueError, match="Result changed"):
        m.verify_result(folder, output)


def test_executing_source_must_match_the_snapshot(protocol_run, monkeypatch, tmp_path):
    output, _ = protocol_run
    runtime_root = tmp_path / "different_checkout"
    runtime_source = runtime_root / "experiments/MNIST/model_selection.py"
    runtime_source.parent.mkdir(parents=True)
    runtime_source.write_text("# unregistered implementation\n")
    monkeypatch.setattr(m, "ROOT", runtime_root)
    with pytest.raises(ValueError, match="Executing source differs"):
        m.load_protocol(output)


def test_omitting_metrics_from_completion_manifest_is_rejected(protocol_run):
    output, _ = protocol_run
    folder = output / "runs/fake/repeat_00"
    folder.mkdir(parents=True)
    m.write(folder / "metrics.json", {"brier": .1})
    # A syntactically valid completion file cannot opt out of protecting its
    # scores merely by omitting metrics.json from the hash list.
    m.complete(folder, output, ())
    with pytest.raises(ValueError, match="Incomplete result manifest"):
        m.verify_result(folder, output)


def test_negative_task_index_is_rejected_before_fit(protocol_run):
    output, _ = protocol_run
    with pytest.raises(ValueError, match="Task index outside"):
        m.fit(output, -1, torch.device("cpu"))


def test_fit_restores_absolute_best_brier_checkpoint(protocol_run, monkeypatch):
    output, protocol = protocol_run

    class TinyRatio(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.3))

        def forward(self, images):
            return self.weight.expand(len(images), 1)

    class FakeGenerator(torch.nn.Module):
        def forward(self, latent):
            return torch.zeros(len(latent), 1, 28, 28)

    def samples(output, cfg, repeat, roles, generator, device):
        return {f"{role}_{side}": torch.zeros(cfg[f"{role}_n"], 1, 28, 28)
                for role in roles for side in ("p", "q")}

    calls, epoch_states = [], []
    epoch_scores = [(1.4, .6), (1.6, .4), (1.2, .8)]  # Brier .09, .04, .16.

    def prediction(model, images, alpha, batch_size, device):
        call = len(calls)
        calls.append(call)
        if call < 6:
            if call % 2 == 0:
                epoch_states.append(model.weight.detach().clone())
            value = epoch_scores[call // 2][call % 2]
        elif call < 8:
            torch.testing.assert_close(model.weight.detach(), epoch_states[1], atol=0, rtol=0)
            value = epoch_scores[1][call % 2]
        else:
            value = 1.
        return np.full(len(images), value)

    monkeypatch.setattr(m, "make_model", lambda *args: TinyRatio())
    monkeypatch.setattr(m, "build_dcgan28", lambda *args, **kwargs: (FakeGenerator(), 100))
    monkeypatch.setattr(m, "real_images", lambda *args: torch.zeros(8, 1, 28, 28, dtype=torch.uint8))
    monkeypatch.setattr(m, "role_samples", samples)
    monkeypatch.setattr(m, "predict", prediction)
    m.fit(output, 0, torch.device("cpu"))
    folder = m.run_path(output, protocol["tasks"][0])
    metrics = m.verify_result(folder, output)
    assert metrics["best_epoch"] == 2
    assert metrics["epochs"] == 3
    assert metrics["earlystop_brier"] == pytest.approx(.04)
    assert len(calls) == 12  # Three epochs, restored checkpoint, two selection roles.
    checkpoint = torch.load(folder / "model.pt", weights_only=True)
    torch.testing.assert_close(checkpoint["model"]["weight"], epoch_states[1], atol=0, rtol=0)
    assert not torch.equal(epoch_states[0], epoch_states[1])
