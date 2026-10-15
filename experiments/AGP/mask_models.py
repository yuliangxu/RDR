#!/usr/bin/env python3
"""Standalone occurrence models for the AGP conditional abundance flow.

Only training and stopping masks enter model selection. The selected fit length
and architecture are refitted on their union, without further stopping. Empty
generated masks are rejected, never repaired by changing arbitrary taxa.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_CONFIG = {
    "seed": 2026101101,
    "mixture": {
        "components": [8, 16, 32], "starts": 2, "max_iter": 200,
        "patience": 15, "min_delta": 1e-5,
        "beta_pseudocount": 0.5, "weight_pseudocount": 0.5,
    },
    "autoregressive": {
        "hidden": [128, 256], "max_epochs": 150, "patience": 15,
        "batch_size": 256, "learning_rate": 1e-3,
        "weight_decay": 1e-4, "grad_clip": 5.0, "min_delta": 1e-4,
    },
    "sampling": {"batch_size": 1024, "max_draw_multiplier": 100},
}
METHODS = ("bootstrap", "bernoulli_mixture", "autoregressive")


def default_config():
    """Return independent defaults for a caller to freeze in its protocol."""
    return copy.deepcopy(DEFAULT_CONFIG)


def normalized_config(config=None):
    result = copy.deepcopy(DEFAULT_CONFIG)
    for key, value in (config or {}).items():
        if key not in result:
            raise ValueError(f"Unknown mask-model configuration key: {key}")
        if isinstance(result[key], dict):
            unknown = set(value) - set(result[key])
            if unknown:
                raise ValueError(f"Unknown {key} configuration keys: {sorted(unknown)}")
            result[key].update(value)
        else:
            result[key] = value
    mix, ar, sampling = result["mixture"], result["autoregressive"], result["sampling"]
    for key in ("starts", "max_iter", "patience"):
        if int(mix[key]) < 1:
            raise ValueError(f"mixture.{key} must be positive")
    for key in ("max_epochs", "patience", "batch_size"):
        if int(ar[key]) < 1:
            raise ValueError(f"autoregressive.{key} must be positive")
    for key in ("batch_size", "max_draw_multiplier"):
        if int(sampling[key]) < 1:
            raise ValueError(f"sampling.{key} must be positive")
    if not mix["components"] or min(mix["components"]) < 1:
        raise ValueError("Mixture components must be a nonempty positive grid")
    if not ar["hidden"] or min(ar["hidden"]) < 1:
        raise ValueError("Autoregressive hidden widths must be a nonempty positive grid")
    for value in (mix["beta_pseudocount"], mix["weight_pseudocount"],
                  ar["learning_rate"], ar["grad_clip"]):
        if value <= 0 or not np.isfinite(value):
            raise ValueError("Smoothing, learning rate and gradient bound must be positive")
    if ar["weight_decay"] < 0 or min(mix["min_delta"], ar["min_delta"]) < 0:
        raise ValueError("Weight decay and minimum improvements must be nonnegative")
    return result


def validate_masks(values, *, allow_empty=False):
    values = np.asarray(values)
    if (values.ndim != 2 or not len(values) or not values.shape[1]
            or values.dtype.kind not in "biuf" or not np.isfinite(values).all()
            or not np.all((values == 0) | (values == 1))):
        raise ValueError("Masks must be a nonempty finite binary matrix")
    result = np.ascontiguousarray(values, dtype=bool)
    if not allow_empty and np.any(~result.any(axis=1)):
        raise ValueError("Observed all-zero masks are not valid compositions")
    return result


def _hash_array(values):
    h = hashlib.sha256()
    h.update(str(values.shape).encode())
    h.update(np.ascontiguousarray(values, dtype=np.uint8).tobytes())
    return h.hexdigest()


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _row_counts(values):
    return Counter(row.tobytes() for row in np.packbits(values, axis=1))


def mask_schema(train):
    """Determine constants and a prevalence-descending order using these rows only."""
    mean = train.mean(axis=0)
    active = np.flatnonzero((mean > 0) & (mean < 1))
    active = active[np.argsort(-mean[active], kind="stable")]
    return dict(dimension=int(train.shape[1]), active=active.tolist(),
                always_one=np.flatnonzero(mean == 1).tolist(),
                always_zero=np.flatnonzero(mean == 0).tolist(),
                order="descending training prevalence, original index breaks ties")


def _schema_support_violations(values, schema):
    bad = np.zeros(len(values), dtype=bool)
    if schema["always_zero"]:
        bad |= values[:, schema["always_zero"]].any(axis=1)
    if schema["always_one"]:
        bad |= (~values[:, schema["always_one"]]).any(axis=1)
    return int(bad.sum())


def mixture_log_prob(values, weights, probabilities):
    """Exact log likelihood under a mixture of independent Bernoulli vectors."""
    x = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if (x.ndim != 2 or p.ndim != 2 or p.shape[1] != x.shape[1]
            or weights.shape != (len(p),) or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1) or np.any(p <= 0) or np.any(p >= 1)):
        raise ValueError("Invalid Bernoulli mixture parameters")
    log0 = np.log1p(-p)
    joint = x @ (np.log(p) - log0).T + log0.sum(axis=1) + np.log(weights)
    return logsumexp(joint, axis=1)


def _mixture_initialize(x, components, seed):
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(x), components, replace=components > len(x))
    # A training-row prototype plus a strictly interior global distribution.
    global_p = (x.sum(axis=0) + 0.5) / (len(x) + 1)
    p = 0.8 * x[selected] + 0.2 * global_p
    return np.full(components, 1 / components), p


def _mixture_update(x, weights, p, config):
    log0 = np.log1p(-p)
    log_joint = x @ (np.log(p) - log0).T + log0.sum(axis=1) + np.log(weights)
    responsibilities = np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])
    mass = responsibilities.sum(axis=0)
    alpha = float(config["beta_pseudocount"])
    updated = (responsibilities.T @ x + alpha) / (mass[:, None] + 2 * alpha)
    gamma = float(config["weight_pseudocount"])
    weights = (mass + gamma) / (len(x) + len(mass) * gamma)
    return weights, updated


def _fit_mixture(train, valid, components, seed, config, *, fixed_iterations=None):
    start = time.monotonic()
    x = np.asarray(train, dtype=np.float64)
    v = None if valid is None else np.asarray(valid, dtype=np.float64)
    weights, p = _mixture_initialize(x, components, seed)
    limit = int(config["max_iter"] if fixed_iterations is None else fixed_iterations)
    history = []
    best = None
    bad = 0
    for iteration in range(limit + 1):
        if iteration:
            weights, p = _mixture_update(x, weights, p, config)
        train_nll = float(-mixture_log_prob(x, weights, p).mean())
        valid_nll = None if v is None else float(-mixture_log_prob(v, weights, p).mean())
        history.append(dict(iteration=iteration, train_nll=train_nll, valid_nll=valid_nll))
        if v is not None:
            if best is None or valid_nll < best[0] - config["min_delta"]:
                best = (valid_nll, iteration, weights.copy(), p.copy())
                bad = 0
            else:
                bad += 1
            if bad >= config["patience"]:
                break
    if valid is not None:
        score, selected, weights, p = best
    else:
        score, selected = None, limit
    return dict(weights=weights, probabilities=p, history=history,
                selected_iterations=int(selected), valid_nll=score,
                iterations_completed=int(history[-1]["iteration"]),
                cap_hit=bool(v is not None and history[-1]["iteration"] == limit
                             and bad < config["patience"]),
                seconds=time.monotonic() - start)


class BinaryNADE(nn.Module):
    """One-hidden-layer NADE with parallel likelihood and cached sequential sampling.

    Hidden preactivation for taxon j uses only x_0,...,x_(j-1). Columns are
    supplied in a frozen training-derived order by the surrounding schema.
    """
    def __init__(self, dimension, hidden):
        super().__init__()
        self.dimension, self.hidden = int(dimension), int(hidden)
        self.input_weights = nn.Parameter(torch.empty(dimension, hidden))
        self.output_weights = nn.Parameter(torch.empty(dimension, hidden))
        self.hidden_bias = nn.Parameter(torch.zeros(hidden))
        self.output_bias = nn.Parameter(torch.zeros(dimension))
        nn.init.normal_(self.input_weights, std=0.01)
        nn.init.normal_(self.output_weights, std=0.01)

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError("NADE input dimensions differ")
        contributions = x[:, :, None] * self.input_weights[None, :, :]
        # Shift the cumulative sum, so even floating-point evaluation never reads
        # the current or future taxon's value into its own conditional probability.
        prefix = torch.cat((torch.zeros_like(contributions[:, :1]),
                            torch.cumsum(contributions[:, :-1], dim=1)), dim=1)
        hidden = torch.sigmoid(prefix + self.hidden_bias)
        return (hidden * self.output_weights).sum(dim=2) + self.output_bias

    def log_prob(self, x):
        return -F.binary_cross_entropy_with_logits(self(x), x, reduction="none").sum(dim=1)

    @torch.no_grad()
    def sample(self, count, generator):
        device = self.hidden_bias.device
        a = self.hidden_bias.expand(count, self.hidden).clone()
        values = torch.empty((count, self.dimension), device=device)
        for j in range(self.dimension):
            logits = (torch.sigmoid(a) * self.output_weights[j]).sum(dim=1) + self.output_bias[j]
            draw = torch.rand(count, generator=generator, device=device) < torch.sigmoid(logits)
            values[:, j] = draw
            a.add_(draw[:, None] * self.input_weights[j])
        return values


def _seed_torch(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    # Cumsum and elementwise NADE operations avoid nondeterministic scatter kernels.
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def _nade_nll(model, x, batch_size):
    model.eval()
    total = 0.0
    for batch in x.split(batch_size):
        total += float(-model.log_prob(batch).double().sum())
    return total / len(x)


def _cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _fit_nade(train, valid, hidden, seed, config, device, *, fixed_epochs=None):
    start = time.monotonic()
    _seed_torch(seed)
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(np.asarray(train, dtype=np.float32), device=device)
    v = None if valid is None else torch.as_tensor(np.asarray(valid, dtype=np.float32), device=device)
    model = BinaryNADE(x.shape[1], hidden).to(device)
    prevalence = (x.sum(dim=0) + 0.5) / (len(x) + 1)
    with torch.no_grad():
        model.output_bias.copy_(torch.logit(prevalence))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    batch_size = int(config["batch_size"])
    limit = int(config["max_epochs"] if fixed_epochs is None else fixed_epochs)
    if x.shape[1] == 0:
        limit = 0
    history = []
    best, bad = None, 0
    for epoch in range(limit + 1):
        if epoch:
            model.train()
            indices = rng.permutation(len(x))
            for offset in range(0, len(x), batch_size):
                idx = torch.as_tensor(indices[offset:offset + batch_size], device=device)
                batch = x[idx]
                optimizer.zero_grad(set_to_none=True)
                # Ordinary joint Bernoulli NLL: no class weighting or taxon weighting.
                loss = -model.log_prob(batch).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Non-finite NADE likelihood")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"],
                                               error_if_nonfinite=True)
                optimizer.step()
        train_nll = _nade_nll(model, x, batch_size)
        valid_nll = None if v is None else _nade_nll(model, v, batch_size)
        history.append(dict(epoch=epoch, train_nll=train_nll, valid_nll=valid_nll))
        if v is not None:
            if best is None or valid_nll < best[0] - config["min_delta"]:
                best = (valid_nll, epoch, _cpu_state(model))
                bad = 0
            else:
                bad += 1
            if bad >= config["patience"]:
                break
    if valid is not None:
        score, selected, state = best
    else:
        score, selected, state = None, limit, _cpu_state(model)
    return dict(state=state, history=history, selected_epochs=int(selected), valid_nll=score,
                epochs_completed=int(history[-1]["epoch"]),
                cap_hit=bool(v is not None and x.shape[1] > 0 and history[-1]["epoch"] == limit
                             and bad < config["patience"]),
                seconds=time.monotonic() - start)


def _save_mixture(path, fit):
    np.savez_compressed(path, weights=fit["weights"], probabilities=fit["probabilities"])


def fit_and_select(train_masks, valid_masks, full_masks, output, config=None, device="cpu"):
    """Select mask models on stopping NLL, then refit on the full training template.

    The output directory must be empty or hold a verified identical completed fit.
    Dimensions constant in inner training are deterministic during selection;
    a stopping violation of this support raises an error. Final refits recompute
    constants and ordering using the full template. Epoch zero is a valid candidate.
    """
    config = normalized_config(config)
    train, valid, full = (validate_masks(x) for x in (train_masks, valid_masks, full_masks))
    if train.shape[1] != valid.shape[1] or train.shape[1] != full.shape[1]:
        raise ValueError("Training, stopping and full-template dimensions differ")
    if len(full) != len(train) + len(valid) or _row_counts(full) != _row_counts(train) + _row_counts(valid):
        raise ValueError("Full template must be exactly the multiset union of training and stopping")
    development_schema, final_schema = mask_schema(train), mask_schema(full)
    violations = _schema_support_violations(valid, development_schema)
    if violations:
        raise ValueError(f"{violations} stopping rows violate deterministic inner-training columns")
    identity = dict(train_sha256=_hash_array(train), valid_sha256=_hash_array(valid),
                    full_sha256=_hash_array(full), config=config, device=str(device),
                    source_sha256=_sha256(__file__))
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        receipt = output / "COMPLETE.json"
        if not receipt.exists():
            raise FileExistsError("Mask-model output contains an incomplete earlier run")
        saved = json.loads(receipt.read_text())
        if saved.get("identity") != identity:
            raise ValueError("Completed mask model uses different input, source, device or settings")
        for relative, digest in saved["artifacts_sha256"].items():
            if _sha256(output / relative) != digest:
                raise ValueError(f"Completed mask-model artifact changed: {relative}")
        return saved
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    _dump(output / "protocol.json", dict(identity=identity,
          development_schema=development_schema, final_schema=final_schema,
          sample_counts=dict(training=len(train), stopping=len(valid), full_template=len(full)),
          selection="minimum stopping joint Bernoulli NLL; earliest checkpoint within min_delta; "
                    "smaller architecture then lower start index breaks candidate ties",
          refit="selected architecture and exact epoch/iteration count on full template",
          empty_mask_policy="rejection sampling conditional on at least one present taxon",
          likelihood_units="nats per complete mask, sum over varying taxa; constants contribute zero"))
    np.savez_compressed(output / "bootstrap.npz", masks=full)
    _dump(output / "bootstrap.json", dict(method="bootstrap", training_rows=len(full),
          seed=config["seed"], schema=final_schema, seconds=time.monotonic() - start,
          description="Uniform row sampling from the full observed mask matrix"))
    x, v = train[:, development_schema["active"]], valid[:, development_schema["active"]]
    all_x = full[:, final_schema["active"]]
    candidates = []
    mix_candidates = []
    mix_cfg = config["mixture"]
    for components in sorted(set(mix_cfg["components"])):
        for initialization in range(int(mix_cfg["starts"])):
            seed = int(config["seed"]) + 10000 + 100 * int(components) + initialization
            name = f"mixture_k{components}_start{initialization}"
            fit = _fit_mixture(x, v, int(components), seed, mix_cfg)
            _save_mixture(output / f"{name}.npz", fit)
            _write_csv(output / f"{name}_history.csv", fit["history"])
            entry = dict(method="bernoulli_mixture", candidate=name, components=int(components),
                         initialization=initialization, seed=seed, valid_nll=fit["valid_nll"],
                         selected_iterations=fit["selected_iterations"],
                         iterations_completed=fit["iterations_completed"],
                         cap_hit=fit["cap_hit"], seconds=fit["seconds"])
            candidates.append(entry)
            mix_candidates.append(entry)
    mix_best = min(mix_candidates, key=lambda a: (a["valid_nll"], a["components"], a["initialization"]))
    refit = _fit_mixture(all_x, None, mix_best["components"], mix_best["seed"], mix_cfg,
                         fixed_iterations=mix_best["selected_iterations"])
    _save_mixture(output / "bernoulli_mixture.npz", refit)
    _write_csv(output / "bernoulli_mixture_refit_history.csv", refit["history"])
    _dump(output / "bernoulli_mixture.json", dict(method="bernoulli_mixture", schema=final_schema,
          selected=mix_best, refit_iterations=refit["selected_iterations"],
          refit_train_nll=refit["history"][-1]["train_nll"], refit_seconds=refit["seconds"],
          training_rows=len(full), probability_smoothing="posterior mean with symmetric Beta prior",
          beta_pseudocount=mix_cfg["beta_pseudocount"],
          weight_pseudocount=mix_cfg["weight_pseudocount"]))
    ar_candidates = []
    ar_cfg = config["autoregressive"]
    for hidden in sorted(set(ar_cfg["hidden"])):
        seed = int(config["seed"]) + 20000 + int(hidden)
        name = f"nade_h{hidden}"
        fit = _fit_nade(x, v, int(hidden), seed, ar_cfg, device)
        torch.save(dict(state=fit["state"], dimension=x.shape[1], hidden=int(hidden)),
                   output / f"{name}.pt")
        _write_csv(output / f"{name}_history.csv", fit["history"])
        entry = dict(method="autoregressive", candidate=name, hidden=int(hidden), seed=seed,
                     valid_nll=fit["valid_nll"], selected_epochs=fit["selected_epochs"],
                     epochs_completed=fit["epochs_completed"], cap_hit=fit["cap_hit"],
                     seconds=fit["seconds"])
        candidates.append(entry)
        ar_candidates.append(entry)
    ar_best = min(ar_candidates, key=lambda a: (a["valid_nll"], a["hidden"]))
    refit = _fit_nade(all_x, None, ar_best["hidden"], ar_best["seed"], ar_cfg, device,
                      fixed_epochs=ar_best["selected_epochs"])
    torch.save(dict(state=refit["state"], dimension=all_x.shape[1], hidden=ar_best["hidden"]),
               output / "autoregressive.pt")
    _write_csv(output / "autoregressive_refit_history.csv", refit["history"])
    _dump(output / "autoregressive.json", dict(method="autoregressive", schema=final_schema,
          selected=ar_best, refit_epochs=refit["selected_epochs"],
          refit_train_nll=refit["history"][-1]["train_nll"], refit_seconds=refit["seconds"],
          training_rows=len(full), model="one-hidden-layer NADE, sigmoid hidden activation",
          objective="unweighted ordinary joint Bernoulli negative log likelihood"))
    for row in candidates:
        row["selected"] = row["candidate"] in (mix_best["candidate"], ar_best["candidate"])
    _write_csv(output / "candidate_validation.csv", candidates)
    receipt = dict(status="complete", identity=identity, selected=dict(
        bernoulli_mixture=mix_best, autoregressive=ar_best),
        sample_counts=dict(training=len(train), stopping=len(valid), full_template=len(full)),
        development_schema=development_schema, final_schema=final_schema,
        seconds=time.monotonic() - start)
    receipt["artifacts_sha256"] = {p.name: _sha256(p) for p in sorted(output.iterdir()) if p.is_file()}
    _dump(output / "COMPLETE.json", receipt)
    return receipt


def _restore_schema(active_values, schema):
    result = np.zeros((len(active_values), schema["dimension"]), dtype=bool)
    result[:, schema["active"]] = active_values
    result[:, schema["always_one"]] = True
    return result


def sample_model(output, method, n, seed, device="cpu"):
    """Draw nonempty masks reproducibly and return explicit rejection accounting.

    Sampling changes no fitted artifact. Samples are conditional on nonempty
    support; details record every proposal and rejection. At most n*multiplier
    proposals are permitted, so an almost-always-empty model fails explicitly.
    """
    if method not in METHODS or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("Unknown mask model or nonpositive integer sample count")
    output = Path(output)
    receipt = json.loads((output / "COMPLETE.json").read_text())
    if receipt.get("status") != "complete":
        raise ValueError("Mask model is incomplete")
    model_file = {"bootstrap": "bootstrap.npz", "bernoulli_mixture": "bernoulli_mixture.npz",
                  "autoregressive": "autoregressive.pt"}[method]
    for relative in (model_file, f"{method}.json"):
        if _sha256(output / relative) != receipt["artifacts_sha256"][relative]:
            raise ValueError(f"Fitted mask-model artifact changed: {relative}")
    metadata = json.loads((output / f"{method}.json").read_text())
    schema = metadata["schema"]
    cfg = receipt["identity"]["config"]["sampling"]
    rng = np.random.default_rng(int(seed))
    if method == "bootstrap":
        with np.load(output / model_file) as arrays:
            reference = arrays["masks"]
        draw = lambda count: reference[rng.integers(len(reference), size=count)]
    elif method == "bernoulli_mixture":
        with np.load(output / model_file) as arrays:
            weights, p = arrays["weights"], arrays["probabilities"]
        def draw(count):
            c = rng.choice(len(weights), count, p=weights)
            values = rng.random((count, p.shape[1])) < p[c]
            return _restore_schema(values, schema)
    else:
        saved = torch.load(output / model_file, map_location="cpu", weights_only=True)
        model = BinaryNADE(saved["dimension"], saved["hidden"]).to(device).eval()
        model.load_state_dict(saved["state"], strict=True)
        generator = torch.Generator(device=device).manual_seed(int(seed))
        def draw(count):
            values = model.sample(count, generator).cpu().numpy().astype(bool)
            return _restore_schema(values, schema)
    start = time.monotonic()
    accepted, remaining, proposals, rejected = [], int(n), 0, 0
    maximum = int(n) * int(cfg["max_draw_multiplier"])
    while remaining:
        count = min(remaining, int(cfg["batch_size"]), maximum - proposals)
        if count <= 0:
            raise RuntimeError(f"{method}: nonempty-mask rejection limit exceeded after "
                               f"{proposals} proposals and {rejected} empty rejections")
        batch = draw(count)
        good = batch.any(axis=1)
        proposals += count
        rejected += int((~good).sum())
        accepted.append(batch[good])
        remaining -= int(good.sum())
    values = np.concatenate(accepted)
    details = dict(method=method, requested=int(n), seed=int(seed), device=str(device),
                   proposals=proposals, rejected_empty=rejected, rejection_fraction=rejected / proposals,
                   nonempty_conditioning=True, sampling_seconds=time.monotonic() - start,
                   unique_masks=int(len(_row_counts(values))),
                   model_sha256=receipt["artifacts_sha256"][model_file])
    return values, details
