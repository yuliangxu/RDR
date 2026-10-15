"""Contracts for mask-conditioned ICFM, exact support and staged RDR assessment."""
import signal
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import pandas as pd

from experiments.AGP import icfm_zero_generator as generator


def test_masked_clr_roundtrip_preserves_support_and_ignores_row_scale():
    raw = np.array([[0., .2, .3, 0., .5], [0., 0., 7., 0., 0.],
                    [1e-8, 0., 1., 2., 0.], [.8, .2, 0., 0., 0.]])
    z, mask = generator.encode_masked_clr(raw)
    assert mask.dtype == np.bool_ and z.dtype == np.float64
    np.testing.assert_array_equal(mask, raw > 0)
    assert np.isfinite(z).all() and not z[~mask].any()
    np.testing.assert_allclose(z.sum(1), 0., atol=4e-15, rtol=0)
    decoded = generator.decode_masked_clr(z, mask)
    np.testing.assert_allclose(decoded, raw/raw.sum(1, keepdims=True), atol=2e-15, rtol=2e-14)
    np.testing.assert_array_equal(decoded > 0, mask)
    np.testing.assert_allclose(decoded.sum(1), 1., atol=2e-16, rtol=0)
    scaled, other_mask = generator.encode_masked_clr(raw*np.array([.1, 100., 7., .3])[:, None])
    np.testing.assert_array_equal(mask, other_mask)
    np.testing.assert_allclose(z, scaled, atol=4e-15, rtol=0)
    np.testing.assert_array_equal(z[1], np.zeros(5))
    np.testing.assert_array_equal(decoded[1], [0., 0., 1., 0., 0.])


def test_empty_masks_and_numerical_loss_of_positive_support_are_rejected():
    with pytest.raises(ValueError):
        generator.encode_masked_clr(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        generator.decode_masked_clr(np.zeros((2, 4)), np.zeros((2, 4), dtype=bool))
    with pytest.raises((ValueError, FloatingPointError)):
        generator.decode_masked_clr(np.array([[1000., -1000.]]), np.ones((1, 2), dtype=bool))


def test_mask_projection_is_centered_idempotent_and_discards_inactive_coordinates():
    mask = torch.tensor([[True, False, True, True], [False, True, False, False]])
    values = torch.tensor([[2., 9876., 4., -3.], [981., 7., -999., 42.]], dtype=torch.float64,
                          requires_grad=True)
    projected = generator.project_masked(values, mask)
    expected = torch.tensor([[1., 0., 3., -4.], [0., 0., 0., 0.]], dtype=torch.float64)
    assert torch.equal(projected, expected)
    assert torch.equal(generator.project_masked(projected, mask), projected)
    projected.square().sum().backward()
    assert torch.isfinite(values.grad).all() and not values.grad[~mask].any()


def test_flow_loss_weights_rows_by_active_dimension_and_ignores_absent_coordinates():
    mask = torch.tensor([[1., 1., 0., 0.], [1., 1., 1., 0.], [0., 1., 0., 0.]])
    target = torch.tensor([[1., -1., 0., 0.], [1., 1., -2., 0.], [0., 0., 0., 0.]])
    predicted = torch.tensor([[0., 0., 999., -999.], [0., 0., 0., 500.], [33., 0., -90., 81.]])
    # Row errors are 2/(2-1), 6/(3-1), and 0/max(1-1,1).
    torch.testing.assert_close(generator.conditional_flow_loss(predicted, target, mask), torch.tensor(5/3))


def test_velocity_receives_mask_as_context_and_remains_tangent_to_its_face():
    torch.set_num_threads(2)
    model = generator.build_model(dict(dim=6, width=16, blocks=2, time_features=8, time_width=16, dropout=.1))
    model.eval()
    torch.nn.init.xavier_uniform_(model.out[-1].weight)
    masks = torch.tensor([[True, False, True, True, False, False],
                          [False, True, True, False, True, False]])
    state_time = torch.zeros(2, 7)
    state_time[:, -1] = .4
    captured = []
    handle = model.input_projection.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
    velocity = model(state_time, masks)
    handle.remove()
    assert captured[0].shape == (2, 12)
    np.testing.assert_array_equal(captured[0][:, 6:].numpy(), masks.numpy())
    assert not velocity[~masks].any()
    torch.testing.assert_close(velocity.sum(1), torch.zeros(2), atol=1e-6, rtol=0)
    assert not torch.equal(captured[0][0], captured[0][1])
    production = generator.build_model({})
    assert production.input_projection.in_features == 1228 and production.input_projection.out_features == 1024
    assert len(production.blocks) == 4
    assert production.out[-1].in_features == 1024 and production.out[-1].out_features == 614


def test_sampler_projects_supplied_noise_and_passes_fixed_masks_to_every_rhs(monkeypatch):
    mask = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0], [1, 1, 1], [0, 1, 1]], dtype=bool)
    noise = np.random.default_rng(19).normal(size=mask.shape).astype(np.float32)
    original = noise.copy()
    seen_masks, seen_initial = [], []

    class Velocity(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dropout = torch.nn.Dropout(.9)

        def forward(self, state_time, masks):
            assert not torch.is_grad_enabled() and not self.training and not self.dropout.training
            seen_masks.append(masks.detach().cpu().numpy())
            values = self.dropout(state_time[:, :-1])
            return generator.project_masked(values, masks)

    class InspectODE(torch.nn.Module):
        def __init__(self, field, *, solver, sensitivity, atol, rtol):
            super().__init__()
            assert solver == "dopri5" and sensitivity == "adjoint" and atol == rtol == 1e-4
            self.field = field

        def trajectory(self, initial, *, t_span):
            assert all(not part.training for part in self.modules())
            np.testing.assert_array_equal(t_span.numpy(), [0., 1.])
            seen_initial.append(initial.detach().cpu().numpy())
            first, second = self.field(torch.tensor(.3), initial), self.field(torch.tensor(.3), initial)
            assert torch.equal(first, second)
            torch.testing.assert_close(first.sum(1), torch.zeros(len(first)), atol=3e-7, rtol=0)
            return torch.stack([initial, initial+first])

    core = types.ModuleType("torchdyn.core")
    core.NeuralODE = InspectODE
    monkeypatch.setitem(sys.modules, "torchdyn.core", core)
    sampled = generator.sample_model(Velocity().train(), noise, mask, batch_size=2)
    expected = generator.project_masked(torch.from_numpy(noise), torch.from_numpy(mask)).numpy()
    np.testing.assert_array_equal(np.concatenate(seen_initial), expected)
    for index, begin in enumerate(range(0, len(mask), 2)):
        np.testing.assert_array_equal(seen_masks[2*index], mask[begin:begin+2])
        np.testing.assert_array_equal(seen_masks[2*index+1], mask[begin:begin+2])
    assert not sampled[~mask].any()
    np.testing.assert_allclose(sampled.sum(1), 0., atol=1e-6, rtol=0)
    np.testing.assert_array_equal(noise, original)


def test_actual_solver_constant_velocity_stays_on_each_mask_face():
    pytest.importorskip("torchdyn.core", reason="Historical generator ODE runtime is optional")
    masks = np.array([[1, 1, 0, 1], [0, 1, 0, 0], [1, 0, 1, 1]], dtype=bool)
    noise = np.random.default_rng(45).normal(size=masks.shape).astype(np.float32)

    class ConstantVelocity(torch.nn.Module):
        def forward(self, state_time, mask):
            constant = torch.tensor([.2, -.1, .4, -.3], dtype=state_time.dtype).expand(len(mask), -1)
            return generator.project_masked(constant, mask)

    first = generator.sample_model(ConstantVelocity(), noise, masks, batch_size=2, atol=1e-7, rtol=1e-7)
    second = generator.sample_model(ConstantVelocity(), noise, masks, batch_size=2, atol=1e-7, rtol=1e-7)
    expected = generator.project_masked(torch.from_numpy(noise)+torch.tensor([.2, -.1, .4, -.3]),
                                        torch.from_numpy(masks)).numpy()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, expected, atol=2e-6, rtol=2e-6)
    assert not first[~masks].any()
    np.testing.assert_allclose(first.sum(1), 0., atol=1e-6, rtol=0)


def test_conditional_fit_resumes_exact_model_ema_optimizer_and_rng(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    rng = np.random.default_rng(38)
    raw = rng.uniform(.1, 1, (13, 6))*(rng.uniform(size=(13, 6)) > .45)
    raw[:, 0] = .2
    raw[0, 1:] = 0
    path = tmp_path/"raw.npy"
    np.save(path, raw/raw.sum(1, keepdims=True))
    cfg = dict(model=dict(dim=6, width=16, blocks=2, time_features=8, time_width=16, dropout=.2),
               training=dict(epochs=3, batch_size=6, device="cpu", checkpoint_epochs=1, warmup_steps=2, seed=59))
    uninterrupted, resumed = tmp_path/"continuous", tmp_path/"resumed"
    generator.train(cfg, path, uninterrupted)
    real_step, calls = generator._step, []

    def interrupt(*args, **kwargs):
        result = real_step(*args, **kwargs)
        calls.append(None)
        if len(calls) == 2:
            signal.raise_signal(signal.SIGUSR1)
        return result

    monkeypatch.setattr(generator, "_step", interrupt)
    with pytest.raises(InterruptedError):
        generator.train(cfg, path, resumed)
    assert not (resumed/"final.pt").exists()
    monkeypatch.setattr(generator, "_step", real_step)
    generator.train(cfg, path, resumed)
    expected, actual = [torch.load(folder/"final.pt", map_location="cpu", weights_only=False)
                        for folder in (uninterrupted, resumed)]

    def same(left, right):
        if isinstance(left, torch.Tensor):
            assert torch.equal(left, right)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                same(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for x, y in zip(left, right):
                same(x, y)
        else:
            assert left == right

    for key in ("identity", "model_state", "ema_state", "optimizer", "scheduler", "rng", "epochs_completed", "updates"):
        same(expected[key], actual[key])
    for left, right in zip(expected["history"], actual["history"]):
        same({key: value for key, value in left.items() if key != "elapsed_seconds"},
             {key: value for key, value in right.items() if key != "elapsed_seconds"})


@pytest.mark.parametrize("defect", ["float32_underflow", "negative_absent"])
def test_analysis_import_rejects_lost_support_and_negative_absent_entries(tmp_path, monkeypatch, defect):
    from experiments.AGP import icfm_zero as workflow
    output, parent = tmp_path/"study", tmp_path/"parent"
    gen = output/"generator"
    (gen/"data").mkdir(parents=True)
    (gen/"generation").mkdir()
    parent.mkdir()
    cfg = workflow.read(Path(workflow.shared.__file__).with_name("selection_config.json"))
    workflow.write(parent/"config.json", cfg)
    workflow.write(gen/"data/protocol.json", dict(parent_run=str(parent), midasim_run=str(parent)))
    (gen/"data/taxa.csv").write_text("taxon_index,taxon_id\n0,t0\n")
    real = np.zeros((1, 614), dtype=np.float32)
    real[0, 1] = 1
    real_path = parent/"real.npy"
    np.save(real_path, real)
    real_frame = pd.DataFrame(dict(distribution=["P"], role=["train"], source_file=[str(real_path)],
        source_row=[0], sample_id=["real:0"], subject_id=["subject:0"], row_sha256=workflow.row_hashes(real),
        original_distribution=["P"], original_role=["train"], array_row=[0]))
    monkeypatch.setattr(workflow, "verify_generation", lambda output: None)
    monkeypatch.setattr(workflow, "parent_check", lambda parent: {"studies": {"selection": {"roles": ["train"]}}})
    monkeypatch.setattr(workflow, "checked_parent_array", lambda *args: (real, real_frame.copy(), real_path))
    values = np.zeros((workflow.ROLE_SIZES["selection_train"], 614))
    values[:, 1:3] = .5
    masks = values > 0
    if defect == "float32_underflow":
        values[0, 1:3] = [1e-50, 1.]
    else:
        values[0, 3] = -1e-8
        values[0, 2] += 1e-8
    np.save(gen/"generation/selection_train.npy", values)
    np.save(gen/"data/mask_selection_train.npy", masks)
    with pytest.raises(ValueError, match="support|Invalid generated"):
        workflow.prepare_analysis(output)
