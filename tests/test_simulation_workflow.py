"""Scientific target, selection separation, and artifact-safety regression tests."""

import copy
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import logsumexp, expit
from scipy.stats import multivariate_normal

from experiments.simulations import workflow as w
from experiments.simulations.population import Mixture
from utils.losses import midpoint_rdr_loss
from utils.networks import make_ratio_mlp
from utils.training import fit_ratio_mlp


@pytest.fixture
def cfg():
    torch.set_num_threads(1)
    result = w.read_json(w.DEFAULT_CONFIG)
    result.update(cases=["D2_noise0.00"], selection_repeats=1, convergence_repeats=1,
                  losses=["hellinger", "js"], architectures=["baseline32"], output_alphas=[1, 2],
                  train_n=12, validation_n=12, calibration_n=20, evaluation_n=20,
                  sample_sizes=[6, 12], max_epochs=2, min_epochs=1, patience=1,
                  min_cell_count=2, bootstrap_repetitions=20)
    return result


@pytest.mark.parametrize("case", ["D2_noise0.00", "D20_noise0.10", "D40_noise0.30"])
def test_truth_matches_full_dimensional_density(case):
    population = Mixture(case)
    x = population.draw(np.random.default_rng(43), 15, "M")
    logs = {}
    for source in ("P", "Q"):
        components = []
        for weight, mean, covariance in zip(population.w, population.means[source], population.covs[source]):
            full_cov = population.k @ covariance @ population.k.T + population.noise**2*np.eye(population.dimension)
            components.append(np.log(weight) + multivariate_normal.logpdf(x, population.k@mean, full_cov))
        logs[source] = logsumexp(components, axis=0)
    np.testing.assert_allclose(population.truth(x), 2*expit(logs["P"]-logs["Q"]), atol=1e-11)


@pytest.mark.parametrize("objective", ["hellinger", "kl", "chisq", "js"])
def test_losses_have_correct_population_stationary_target(objective):
    # Exact discrete population: P=(.75,.25), Q=(.25,.75), r=(1.5,.5).
    class Lookup(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.r = torch.nn.Parameter(torch.tensor([1.5, .5], dtype=torch.float64))

        def forward(self, x):
            return self.r[x.long()].reshape(-1, 1)

    model = Lookup()
    p, q = torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 1, 1])
    loss = midpoint_rdr_loss(model, p, q, objective)
    loss.backward()
    np.testing.assert_allclose(model.r.grad, 0, atol=1e-14)
    with torch.no_grad():
        model.r.add_(torch.tensor([.1, -.1]))
    assert midpoint_rdr_loss(model, p, q, objective) > loss


def test_js_is_balanced_pq_bce_up_to_constant():
    model = make_ratio_mlp(2, seed=7)
    p, q = torch.randn(7, 2), torch.randn(11, 2)
    bce = -.5*(model(p)/2).log().mean() - .5*(1-model(q)/2).log().mean()
    assert float(midpoint_rdr_loss(model, p, q, "js")) == pytest.approx(float(bce)-math.log(2), abs=1e-7)


def test_brier_excess_equals_quarter_mixture_mse():
    p, q = np.array([.75, .25]), np.array([.25, .75])
    truth, estimate = 2*p/(p+q), np.array([1.2, .6])
    risk = lambda r: .5*np.sum(p*(r/2-1)**2) + .5*np.sum(q*(r/2)**2)
    mse = np.sum(.5*(p+q)*(truth-estimate)**2)
    assert risk(estimate)-risk(truth) == pytest.approx(mse/4)


def test_matched_initialization_and_activation():
    before = torch.get_rng_state().clone()
    plain, residual = [make_ratio_mlp(3, name, seed=9) for name in ("deep64", "residual64")]
    assert torch.equal(before, torch.get_rng_state())
    for key in plain.state_dict():
        assert torch.equal(plain.state_dict()[key], residual.state_dict()[key])
    a1, a4 = [make_ratio_mlp(3, output_alpha=alpha, seed=9).double() for alpha in (1, 4)]
    x = torch.ones(2, 3, dtype=torch.float64)
    assert torch.allclose(torch.logit(a4(x)/2), 4*torch.logit(a1(x)/2))


def test_checkpoint_is_absolute_best_even_at_epoch_limit(cfg):
    _, arrays, _ = w.samples(cfg, "selection", cfg["cases"][0], 0, 12)
    model, metadata, history = fit_ratio_mlp(make_ratio_mlp(2), arrays["train_P"], arrays["train_Q"],
        arrays["validation_P"], arrays["validation_Q"], max_epochs=3, min_epochs=3, patience=3)
    assert metadata["best_val_loss"] == min(row["validation_loss"] for row in history)
    assert metadata["restored_val_loss"] == metadata["best_val_loss"]


def test_all_split_seed_vectors_unique_and_stage_separated(cfg):
    case = cfg["cases"][0]
    _, arrays, seeds = w.samples(cfg, "selection", case, 0, 12)
    _, arrays_again, seeds_again = w.samples(cfg, "selection", case, 0, 12)
    assert len({tuple(vector) for vector in seeds.values()}) == 8
    assert seeds == seeds_again
    assert all(np.array_equal(arrays[key], arrays_again[key]) for key in arrays)
    _, fresh, fresh_seeds = w.samples(cfg, "convergence", case, 0, 12)
    assert not set(map(tuple, seeds.values())) & set(map(tuple, fresh_seeds.values()))
    assert all(not np.array_equal(arrays[key], fresh[key]) for key in arrays)
    # Validation/calibration/evaluation draws are paired across convergence n.
    _, other_n, _ = w.samples(cfg, "convergence", case, 0, 6)
    assert np.array_equal(fresh["evaluation_P"], other_n["evaluation_P"])
    assert not np.array_equal(fresh["train_P"][:6], other_n["train_P"])


def test_default_grid_joint_confirmation():
    grid = w.candidate_grid(w.read_json(w.DEFAULT_CONFIG))
    assert len(grid) == 23
    assert len([row for row in grid if row["loss"] == "hellinger"]) == 20
    assert all(row["architecture"] == "baseline32" and row["output_alpha"] == 2 for row in grid if row["loss"] != "hellinger")


def test_complete_workflow_selection_freeze_convergence_and_integrity(tmp_path, cfg):
    cfg["adaptive"] = True
    output, protocol = w.prepare(tmp_path / "selection", cfg, "selection")
    w.run_task(output, protocol, 0)
    rows = w.collect(output, protocol)
    expected = min((row for row in rows if row["loss"] == "hellinger"),
                   key=lambda row: (row["validation_brier"], row["candidate"]))
    chosen = w.freeze(output)
    assert chosen["selected"]["output_alpha"] == expected["output_alpha"]
    assert chosen["selection_uses"] == ["validation_brier"]
    assert w.freeze(output) == chosen
    assert w.report(output).is_file()
    # Complete tasks can be verified/resumed; partial ones are refused.
    w.run_task(output, protocol, 0)
    convergence, cp = w.prepare(tmp_path / "convergence", cfg, "convergence", output / "selection.json")
    for index in range(len(w.task_specs(cp))):
        w.run_task(convergence, cp, index)
    assert len(w.collect(convergence, cp)) == 2
    assert w.report(convergence).is_file()
    assert w.report(convergence).is_file()  # A complete report is idempotent.
    (convergence / "figures" / "convergence.png").unlink()
    with pytest.raises(ValueError, match="missing derived artifacts"):
        w.report(convergence)
    first = w.task_path(output, cfg["cases"][0], 0, cfg["train_n"], protocol["candidates"][0])
    cells = w.read_json(first / "cells.json")
    assert {row["partition"] for row in cells} == {"fixed", "adaptive"}
    assert sum(row["count_p"] for row in cells if row["partition"] == "fixed") == cfg["calibration_n"]
    with (first / "metrics.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="hash changed"):
        w.collect(output, protocol)
    (output / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen selection changed"):
        w.load_frozen(output / "selection.json")


def test_evaluation_scores_cannot_affect_selection(tmp_path, cfg):
    chosen = []
    for run, flip in enumerate((False, True)):
        output, protocol = w.prepare(tmp_path / str(run), cfg, "selection")
        for index, candidate in enumerate(protocol["candidates"]):
            folder = w.task_path(output, cfg["cases"][0], 0, cfg["train_n"], candidate)
            folder.mkdir(parents=True)
            w.save_json(folder / "metrics.json", {"candidate": w.candidate_id(candidate),
                "case": cfg["cases"][0], "repeat": 0, "train_n": cfg["train_n"], **candidate,
                "validation_brier": .1+index*.01, "evaluation_mse": 100-index if flip else index,
                "evaluation_brier": index if flip else 100-index})
            _, _, seeds = w.samples(cfg, "selection", cfg["cases"][0], 0, cfg["train_n"])
            initialization = w.seed_for(cfg, "selection", cfg["cases"][0], 0, "initialization")
            w.save_json(folder / "training.json", {"protocol_sha256": w.digest(output / "protocol.json"),
                "task_index": 0, "data_seed_vectors": seeds, "initialization_seed_vector": initialization,
                "model_seed": int(np.random.SeedSequence(initialization).generate_state(1)[0])})
            w.seal(folder)
        chosen.append(w.freeze(output)["selected"])
    assert chosen[0] == chosen[1]


def test_copied_intact_fit_cannot_impersonate_another_repeat(tmp_path, cfg):
    cfg["selection_repeats"] = 2
    output, protocol = w.prepare(tmp_path / "selection", cfg, "selection")
    w.run_task(output, protocol, 0)
    candidate = protocol["candidates"][0]
    original = w.task_path(output, cfg["cases"][0], 0, cfg["train_n"], candidate)
    copied = w.task_path(output, cfg["cases"][0], 1, cfg["train_n"], candidate)
    shutil.copytree(original, copied)
    w.verify(copied)  # Byte integrity alone cannot establish replicate identity.
    with pytest.raises(ValueError, match="Fit identity"):
        w.collect(output, protocol)
    with pytest.raises(ValueError, match="Fit identity"):
        w.run_task(output, protocol, 1)


@pytest.mark.parametrize("change", ["deleted", "tampered"])
def test_copied_selection_is_required_and_verified_on_resume_and_report(tmp_path, cfg, change):
    selection, sp = w.prepare(tmp_path / "selection", cfg, "selection")
    w.run_task(selection, sp, 0)
    w.freeze(selection)
    original = selection / "selection.json"
    output, protocol = w.prepare(tmp_path / "convergence", cfg, "convergence", original)
    copied = output / "input_selection.json"
    if change == "deleted":
        copied.unlink()
    else:
        copied.write_text(copied.read_text() + " ")
    with pytest.raises(ValueError, match="Copied input selection"):
        w.prepare(output, cfg, "convergence", original)
    with pytest.raises(ValueError, match="Copied input selection"):
        w.collect(output, protocol)
    with pytest.raises(ValueError, match="Copied input selection"):
        w.report(output)
    with pytest.raises(ValueError, match="Copied input selection"):
        w.run_task(output, protocol, 0)


def test_partial_and_mismatched_protocol_refused(tmp_path, cfg):
    output, protocol = w.prepare(tmp_path / "selection", cfg, "selection")
    folder = w.task_path(output, cfg["cases"][0], 0, cfg["train_n"], protocol["candidates"][0])
    folder.mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete task"):
        w.run_task(output, protocol, 0)
    changed = copy.deepcopy(cfg)
    changed["seed"] += 1
    with pytest.raises(ValueError, match="different protocol"):
        w.prepare(output, changed, "selection")


def test_help_from_another_working_directory(tmp_path):
    result = subprocess.run([sys.executable, "-B", str(Path(w.__file__).resolve()), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "convergence" in result.stdout
