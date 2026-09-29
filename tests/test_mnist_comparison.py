"""Split, sampling, and final-data safeguards for the post-selection refit."""

import copy

import numpy as np
import pytest
import torch

from experiments.MNIST import comparison as c


def historical_splits():
    return {"train_indices": list(range(55000)),
            "validation_loss_indices": list(range(55000, 60000)),
            "test_indices": list(range(10000))}


def selected_splits():
    return {"final_calibration": {"indices": list(range(0, 10000, 2))},
            "final_evaluation": {"indices": list(range(1, 10000, 2))}}


def test_historical_rows_and_diagnostic_positions_are_preserved():
    old, selected = historical_splits(), selected_splits()
    originals = copy.deepcopy((old, selected))
    splits = c.make_splits(old, selected)
    assert splits["train"]["indices"] == old["train_indices"]
    assert splits["earlystop"]["indices"] == old["validation_loss_indices"]
    assert splits["test"]["indices"] == old["test_indices"]
    for role in ("calibration", "evaluation"):
        indices = np.array(splits["test"]["indices"])[splits[f"{role}_positions"]]
        np.testing.assert_array_equal(indices, selected[f"final_{role}"]["indices"])
    smoke = c.make_splits(old, selected, smoke=True)
    assert len(smoke["train"]["indices"]) == 32
    assert len(smoke["earlystop"]["indices"]) == 20
    assert len(smoke["test"]["indices"]) == 128
    assert set(smoke["calibration_positions"]).isdisjoint(smoke["evaluation_positions"])
    assert set(smoke["calibration_positions"] + smoke["evaluation_positions"]) == set(range(128))
    assert (old, selected) == originals


@pytest.mark.parametrize("corruption", ["training_overlap", "test_order", "diagnostic_overlap"])
def test_split_changes_are_rejected(corruption):
    old, selected = historical_splits(), selected_splits()
    if corruption == "training_overlap":
        old["train_indices"][0] = old["validation_loss_indices"][0]
    elif corruption == "test_order":
        old["test_indices"].reverse()
    else:
        selected["final_calibration"]["indices"][0] = selected["final_evaluation"]["indices"][0]
    with pytest.raises(ValueError):
        c.make_splits(old, selected)


class TinyGenerator(torch.nn.Module):
    def decode(self, z):
        assert z.ndim == 2 and z.shape[1] == 20
        return z[:, :1].sigmoid().expand(-1, 784)

    def forward(self, z):
        assert z.shape[1:] == (100, 1, 1)
        return z[:, :1].tanh().expand(-1, 1, 28, 28)


@pytest.mark.parametrize("name", ["vae", "dcgan"])
def test_sampling_uses_private_latent_rng_and_consistent_model_scale(name):
    device = torch.device("cpu")
    rng = torch.Generator().manual_seed(983)
    reference = torch.Generator().manual_seed(983)
    before = torch.get_rng_state().clone()
    images = c.sample_batch(TinyGenerator(), name, 7, rng, device)
    if name == "vae":
        z = torch.randn(7, 20, generator=reference)
        expected = (2 * z[:, :1].sigmoid() - 1).reshape(7, 1, 1, 1).expand(-1, 1, 28, 28)
    else:
        z = torch.randn(7, 100, 1, 1, generator=reference)
        expected = z[:, :1].tanh().expand(-1, 1, 28, 28)
    torch.testing.assert_close(images, expected, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), before)
    assert torch.equal(rng.get_state(), reference.get_state())
    assert not images.requires_grad
    # Non-divisible batching must retain every generated image.
    a = c.sample_images(TinyGenerator(), name, 7, torch.Generator().manual_seed(12), 3, device)
    b = c.sample_images(TinyGenerator(), name, 7, torch.Generator().manual_seed(12), 3, device)
    assert a.shape == (7, 1, 28, 28)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(torch.get_rng_state(), before)


def test_generator_branch_and_sample_role_streams_are_distinct():
    cfg = {"seed": 2026092801}
    seeds = [c.stream_seed(cfg, name, role)
             for name in ("vae", "dcgan") for role in ("training_q", "earlystop", "test")]
    assert len(set(seeds)) == 6
    assert seeds == [c.stream_seed(cfg, name, role)
                     for name in ("vae", "dcgan") for role in ("training_q", "earlystop", "test")]


@pytest.mark.parametrize("name", ["vae", "dcgan"])
def test_invalid_generator_scale_is_rejected(name):
    class Invalid(TinyGenerator):
        def decode(self, z):
            return torch.full((len(z), 784), 1.5)

        def forward(self, z):
            return torch.full((len(z), 1, 28, 28), -1.5)

    with pytest.raises(ValueError, match="invalid model-space"):
        c.sample_batch(Invalid(), name, 2, torch.Generator().manual_seed(1), torch.device("cpu"))


@pytest.mark.parametrize("fail_restore_check", [False, True])
def test_best_checkpoint_is_reloaded_before_test_access(tmp_path, monkeypatch, fail_restore_check):
    """A real gradient fit must restore epoch 2 after a worse epoch 3.

    Predetermined validation scores control the checkpoint decision; parameter
    snapshots independently verify restoration and protect the test-data gate.
    """
    torch.set_num_threads(1)
    cfg = {"seed": 19, "batch_size": 2, "max_epochs": 3, "min_epochs": 3,
           "patience": 9, "bn_freeze_epoch": 5, "max_lr": .006,
           "ci_alpha": .05, "bins": 20}
    protocol = {"config": cfg, "chosen": {"candidate": "js_a2", "loss": "js", "output_alpha": 2},
                "selection_sha256": "fixture", "smoke": True}
    c.ms.write(tmp_path / "protocol.json", protocol)
    c.ms.write(tmp_path / "splits.json", {"test": {"indices": list(range(12))},
        "calibration_positions": list(range(6)), "evaluation_positions": list(range(6, 12))})
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets/t10k-labels-idx1-ubyte").write_bytes(bytes(8) + bytes(i % 10 for i in range(12)))
    monkeypatch.setattr(c, "load_protocol", lambda output: protocol)
    monkeypatch.setattr(c, "load_generator", lambda *args: (TinyGenerator(), 20))

    class TinyRatio(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(.01))

        def forward(self, x):
            return self.weight * x.mean((1, 2, 3)).reshape(-1, 1)

    model = TinyRatio()
    monkeypatch.setattr(c.ms, "make_model", lambda *args: model)
    calls, epoch_states, test_access = [], [], []

    def predict(model, images, alpha, batch_size, device):
        call = len(calls)
        calls.append(call)
        if call < 6:
            if call % 2 == 0:
                epoch_states.append(model.weight.detach().clone())
            p_score = (1.5, 1.9, 1.6)[call // 2]
        else:
            torch.testing.assert_close(model.weight.detach(), epoch_states[1], atol=0, rtol=0)
            p_score = 1.4 if fail_restore_check and call < 8 else 1.9
        return np.full(len(images), p_score if call % 2 == 0 else 2 - p_score, dtype=np.float64)

    monkeypatch.setattr(c.ms, "predict", predict)

    def real_images(output, role):
        if role == "test":
            test_access.append(role)
            assert len(calls) == 8  # Three epoch assessments plus restored-state assessment.
            saved = torch.load(output / "vae/model.pt", weights_only=True)
            assert saved["best_epoch"] == 2
            torch.testing.assert_close(model.weight.detach(), saved["model"]["weight"], atol=0, rtol=0)
            return torch.full((12, 1, 28, 28), 255, dtype=torch.uint8)
        return torch.full((4, 1, 28, 28), 255, dtype=torch.uint8)

    monkeypatch.setattr(c.ms, "real_images", real_images)
    if fail_restore_check:
        with pytest.raises(AssertionError, match="checkpoint was not restored"):
            c.train_branch(tmp_path, "vae", torch.device("cpu"))
        assert not test_access
        assert not (tmp_path / "vae/evaluation.pt").exists()
        assert not (tmp_path / "vae/complete.json").exists()
    else:
        metrics = c.train_branch(tmp_path, "vae", torch.device("cpu"))
        assert metrics["best_epoch"] == 2 and metrics["epochs"] == 3
        assert test_access == ["test"]
        assert not torch.equal(epoch_states[1], epoch_states[2])
        assert metrics["test_p"] == metrics["test_q"] == 12
        assert metrics["calibration_p"] == metrics["diagnostic_evaluation_p"] == 6
        assert c.verify_branch(tmp_path, "vae") == metrics
        # Completed results must not silently accept changed scientific evidence.
        (tmp_path / "vae/evaluation.pt").write_bytes(b"modified")
        with pytest.raises(ValueError, match="Result changed"):
            c.verify_branch(tmp_path, "vae")
