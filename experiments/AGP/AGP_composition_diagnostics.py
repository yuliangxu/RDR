#!/usr/bin/env python3
"""Matched AGP compositional-input experiments with score/frequency diagnostics."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import helmert
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
import torch
from torch import nn
from torch.nn import functional as F

from AGP_ICFM_ci import DEFAULT_DATA, MLP, REPO, midpoint_loss, require, save_json, sha256
from AGP_ICFM_712_ci import verify_result
from AGP_712_data import prepare_712_data

DEFAULT_FIT = Path('/cwork/yx306/RDR/JRSSB/CI/agp_icfm_712_20260920')
DEFAULT_OUTPUT = Path('/cwork/yx306/RDR/JRSSB/CI/agp_composition_20260922')
ROLES = ('train', 'validation', 'test')
EDGES = np.linspace(0, 2, 21)
TRANSFORMED = ('ilr', 'logcontrast', 'philr')


class ContrastLinear(nn.Linear):
    """First-layer weights have exactly zero row sums up to floating-point error."""

    def effective_weight(self):
        return self.weight - self.weight.mean(dim=1, keepdim=True)

    def forward(self, inputs):
        return F.linear(inputs, self.effective_weight(), self.bias)


def close(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 2 and np.isfinite(x).all() and (x >= 0).all(), 'Invalid composition')
    total = x.sum(axis=1, keepdims=True)
    require((total > 0).all(), 'Zero-total composition')
    return x / total


def clr(values, epsilon):
    c = close(values)
    require(epsilon > 0, 'Smoothing must be positive')
    positive = (c + epsilon) / (1 + c.shape[1] * epsilon)
    logs = np.log(positive)
    return logs - logs.mean(axis=1, keepdims=True)


def fit_transform(arrays, method, epsilon, basis=None):
    """Learn only one CLR center and one scalar RMS, using P/Q training equally."""
    state = {'method': method, 'epsilon': epsilon, 'dimension': arrays['P', 'train'].shape[1]}
    if method in TRANSFORMED:
        p, q = (clr(arrays[d, 'train'], epsilon) for d in ('P', 'Q'))
        center = .5 * (p.mean(axis=0) + q.mean(axis=0))
        scale = np.sqrt(.5 * (np.mean((p - center)**2) + np.mean((q - center)**2)))
        require(np.isfinite(scale) and scale > 0, 'Degenerate training log-ratio scale')
        state.update(center=center, scale=float(scale), basis=basis)
    return state


def transform(values, state):
    method = state['method']
    if method == 'raw':
        return np.asarray(values, dtype=np.float32).copy()
    if method == 'closed_raw':
        return close(values).astype(np.float32)
    x = (clr(values, state['epsilon']) - state['center']) / state['scale']
    if method in ('ilr', 'philr'):
        x = x @ state['basis'].T
    require(np.isfinite(x).all(), 'Nonfinite transformed inputs')
    return x.astype(np.float32)


def make_model(dimension, method, seed, basis=None):
    """Match hidden weights and initial functions across log-ratio coordinates."""
    torch.manual_seed(seed)
    model = MLP(dimension, hidden_dim=32, output_alpha=2)
    old = model.model[0]
    if method == 'logcontrast':
        replacement = ContrastLinear(dimension, 32)
        with torch.no_grad():
            replacement.weight.copy_(old.weight - old.weight.mean(dim=1, keepdim=True))
            replacement.bias.copy_(old.bias)
        model.model[0] = replacement
    elif method in ('ilr', 'philr'):
        replacement = nn.Linear(dimension - 1, 32)
        with torch.no_grad():
            replacement.weight.copy_(old.weight @ torch.as_tensor(basis.T, dtype=old.weight.dtype))
            replacement.bias.copy_(old.bias)
        model.model[0] = replacement
    return model


def fit_model(model, arrays, args):
    """Original full-batch loss, optimizer, scheduler, and stopping rule."""
    tensors = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in arrays.items()
               if k[1] in ('train', 'validation')}
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=5, cooldown=2)
    best, best_epoch, best_state = float('inf'), None, None
    stopping_best, stale, history = float('inf'), 0, []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = midpoint_loss(model, tensors['P', 'train'], tensors['Q', 'train'])
        require(bool(torch.isfinite(loss)), 'Nonfinite training objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(midpoint_loss(model, tensors['P', 'validation'], tensors['Q', 'validation']))
        require(np.isfinite(value), 'Nonfinite validation objective')
        history.append({'epoch': epoch, 'train_loss': float(loss.detach()), 'validation_loss': value,
                        'learning_rate': optimizer.param_groups[0]['lr'], 'gradient_norm': float(norm)})
        if value < best:
            best, best_epoch, best_state = value, epoch, copy.deepcopy(model.state_dict())
        if epoch >= args.min_epochs:
            if value < stopping_best - 1e-5:
                stopping_best, stale = value, 0
            else:
                stale += 1
        scheduler.step(value)
        if epoch == 1 or epoch % 200 == 0:
            print(f'  epoch={epoch} train={float(loss.detach()):.7f} val={value:.7f}', flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        restored = float(midpoint_loss(model, tensors['P', 'validation'], tensors['Q', 'validation']))
    require(abs(restored - best) < 1e-7, 'Checkpoint restoration failed')
    return pd.DataFrame(history), {'best_epoch': best_epoch, 'epochs_run': len(history),
        'best_validation_loss': best, 'restore_mode': 'absolute_minimum_validation',
        'stop_reason': 'validation_patience' if len(history) < args.max_epochs else 'max_epochs',
        'parameter_count': sum(x.numel() for x in model.parameters()),
        'contrast_max_row_sum': float(model.model[0].effective_weight().sum(1).abs().max())
            if isinstance(model.model[0], ContrastLinear) else None}


def summarize(scores, spec, training):
    # Existing diagnostics expects all roles; compute directly for whichever have been scored.
    cell_parts, metric_rows = [], []
    for role in scores.role.unique():
        subset = scores[scores.role.eq(role)]
        p, q = (subset[subset.distribution.eq(d)].rdr.to_numpy() for d in ('P', 'Q'))
        bp, bq = (np.searchsorted(EDGES[1:-1], x, side='right') for x in (p, q))
        kp, kq = (np.bincount(b, minlength=20) for b in (bp, bq))
        ph, qh = kp / len(p), kq / len(q)
        total = ph + qh
        y = np.divide(2 * ph, total, out=np.full(20, np.nan), where=total > 0)
        sums = np.bincount(bp, weights=p, minlength=20)/len(p) + np.bincount(bq, weights=q, minlength=20)/len(q)
        x = np.divide(sums, total, out=np.full(20, np.nan), where=total > 0)
        meta = {**spec, 'role': role, 'n_p': len(p), 'n_q': len(q)}
        cells = pd.DataFrame({'bin_index': np.arange(20), 'left': EDGES[:-1], 'right': EDGES[1:],
            'k_p': kp, 'k_q': kq, 'mixture_mass': total/2, 'mean_model_score': x,
            'cell_rdr': y, 'calibration_gap': y-x, 'empty': total == 0})
        for key, value in meta.items():
            cells[key] = value
        cell_parts.append(cells)
        metric_rows.append({**meta, 'equal_mixture_mean_score': .5*(p.mean()+q.mean()),
            'mean_score_p': p.mean(), 'mean_score_q': q.mean(),
            'empirical_bin_absolute_gap': np.nansum(total/2*np.abs(y-x)),
            'empirical_bin_rmse': np.sqrt(np.nansum(total/2*(y-x)**2)),
            'midpoint_loss': .5*np.mean(p**-.5)+.25*np.mean(np.sqrt(p))+.25*np.mean(np.sqrt(q))-1,
            'balanced_brier': .5*np.mean((1-p/2)**2)+.5*np.mean((q/2)**2),
            'auc': roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(q))], np.r_[p, q]),
            'best_epoch': training['best_epoch'], 'epochs_run': training['epochs_run']})
    return pd.concat(cell_parts, ignore_index=True), pd.DataFrame(metric_rows)


def score_roles(model, arrays, manifest, roles):
    parts = []
    for d in ('P', 'Q'):
        for role in roles:
            rows = manifest[manifest.distribution.eq(d) & manifest.role.eq(role)].copy()
            with torch.no_grad():
                values = arrays[d, role]
                scores = np.concatenate([model(torch.as_tensor(chunk))[:, 0].numpy()
                    for chunk in np.array_split(values, max(1, (len(values)+2047)//2048))])
            require(len(rows) == len(scores), 'Score row alignment mismatch')
            require(np.isfinite(scores).all() and (scores > 0).all() and (scores <= 2).all(), 'Invalid RDR scores')
            rows['rdr'] = scores.astype(float)
            rows['bin_index'] = np.searchsorted(EDGES[1:-1], scores.astype(float), side='right')
            parts.append(rows)
    return pd.concat(parts, ignore_index=True)


def refresh_report(out):
    """Add interpretation to a verified completed run without refitting models."""
    old = verify_result(out)
    require(old['analysis'] == 'agp_compositional_input_diagnostics', 'Unexpected completed analysis')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    revision = out/'report_revisions'/stamp
    revision.mkdir(parents=True, exist_ok=False)
    for name in ('report.md', 'result.json'):
        shutil.copyfile(out/name, revision/name)
    report_path = Path('experiments/AGP/AGP_composition_report.py')
    shutil.copyfile(out/'source'/report_path, revision/'previous_renderer.py')
    shutil.copyfile(Path(__file__), revision/'refresh_runner.py')
    shutil.copyfile(REPO/report_path, out/'source'/report_path)
    protocol = json.loads((out/'protocol.json').read_text())
    tables = [pd.read_csv(out/name, float_precision='round_trip') for name in
              ('metrics.csv', 'diagnostic_cells.csv', 'loss_history.csv')]
    from AGP_composition_report import render_report
    render_report(out, *tables, protocol)
    # Existing fits, all observation scores, selection, and diagnostic cells must be untouched.
    unchanged = ['metrics.csv', 'diagnostic_cells.csv', 'loss_history.csv', 'protocol.json',
                 'selection.json', 'split_manifest.csv']
    unchanged += [name for name in old['outputs'] if name.startswith('runs/')]
    for name in unchanged:
        require(sha256(out/name) == old['outputs'][name], f'Report refresh changed numerical artifact: {name}')
    old.setdefault('report_updates', []).append({'timestamp_utc': stamp,
        'reason': 'Add descriptive score-bin mass and middle-range diagnostics to interpret aggregate calibration gains.',
        'previous_artifacts': str(revision.relative_to(out)), 'numerical_fit_artifacts_unchanged': True})
    files = [p for p in out.rglob('*') if p.is_file() and '.matplotlib' not in p.parts and p != out/'result.json']
    old['outputs'] = {str(p.relative_to(out)): sha256(p) for p in sorted(files)}
    save_json(out/'result.json', old)
    print(json.dumps({'status': 'report_refreshed', 'output': str(out), 'numerical_artifacts_unchanged': True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--source-fit', type=Path, default=DEFAULT_FIT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20260918, 20260919, 20260920])
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--sensitivity-epsilons', type=float, nargs='*', default=[1e-7, 1e-5])
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--min-epochs', type=int, default=300)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--refresh-report', action='store_true')
    args = parser.parse_args()
    out, source = args.output_dir.resolve(), args.source_fit.resolve()
    require(out.is_relative_to(Path('/tmp')) if args.preflight else out.is_relative_to(Path('/cwork')),
            'Preflight outputs must use /tmp; production outputs must use /cwork')
    if args.refresh_report:
        refresh_report(out)
        return
    require(not out.exists(), 'Refusing to overwrite an existing experiment')
    require(0 < args.min_epochs <= args.max_epochs and args.patience > 0 and args.threads > 0, 'Invalid fit controls')
    require(args.epsilon > 0 and all(e > 0 and e != args.epsilon for e in args.sensitivity_epsilons), 'Invalid epsilons')
    require(len(set(args.sensitivity_epsilons)) == len(args.sensitivity_epsilons), 'Duplicate sensitivity epsilons')
    require(len(set(args.seeds)) == len(args.seeds), 'Duplicate seeds')
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    threadpool_limits(limits=args.threads)
    torch.use_deterministic_algorithms(True)
    source_result = verify_result(source)
    prior = json.loads((source/'protocol.json').read_text())
    arrays, manifest, audit = prepare_712_data(args.data_root, prior['seed'], prior['p_policy'])
    require(manifest.to_csv(index=False) == (source/'split_manifest.csv').read_text(), 'Archived splits changed')
    require(audit['input_sha256'] == prior['data_audit']['input_sha256'], 'Source data changed')
    from AGP_phylo_basis import load_phylo_basis
    phylo, phylo_metadata = load_phylo_basis(args.data_root)
    dim = audit['feature_dimension']
    bases = {'ilr': helmert(dim, full=False), 'philr': phylo, 'logcontrast': None}
    specs = []
    for seed in args.seeds:
        for method in ('raw', 'closed_raw', *TRANSFORMED):
            epsilon = args.epsilon if method in TRANSFORMED else 0.
            specs.append({'run_id': f'{method}_s{seed}_e{epsilon:g}', 'method': method,
                          'seed': seed, 'epsilon': epsilon, 'primary': True})
    for epsilon in args.sensitivity_epsilons:
        for method in TRANSFORMED:
            specs.append({'run_id': f'{method}_s{args.seeds[0]}_e{epsilon:g}', 'method': method,
                          'seed': args.seeds[0], 'epsilon': epsilon, 'primary': False})
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    input_audit = {f'{d}_{role}': {'shape': list(x.shape), 'zero_fraction': float(np.mean(x == 0)),
        'row_sum_min': float(x.astype(float).sum(1).min()), 'row_sum_max': float(x.astype(float).sum(1).max())}
        for (d, role), x in arrays.items()}
    protocol = {'analysis': 'agp_compositional_input_diagnostics', 'source_fit': str(source),
        'source_result_sha256': sha256(source/'result.json'), 'data_audit': audit, 'config': config,
        'candidate_specs': specs, 'input_audit': input_audit, 'phylo_basis': phylo_metadata,
        'edges': EDGES.tolist(), 'representative_seed': args.seeds[0],
        'selection_rule': 'minimum validation balanced Brier among primary fits; tie by run_id; sensitivity excluded',
        'checkpoint_rule': 'minimum validation original midpoint Hellinger loss',
        'normalization': 'close x; (c+epsilon)/(1+D*epsilon); CLR; equal P/Q train mean and one scalar RMS; optional orthonormal ILR basis',
        'initialization': 'same seeded raw MLP hidden weights; projected first-layer weights give matching initial transformed functions',
        'limitations': audit['limitations'] + ['Three optimization seeds share one data split and are not independent data replicates.',
            'ILR, PhILR and full zero-sum contrast layers have the same function class; coordinates and AdamW optimization differ.',
            'Descriptive fixed-bin gaps compare model-dependent regions and can be small for uninformative constant scores.'],
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'scipy': scipy.__version__}}
    out.mkdir(parents=True)
    save_json(out/'protocol.json', protocol)
    shutil.copyfile(source/'split_manifest.csv', out/'split_manifest.csv')
    source_names = ['experiments/AGP/'+name for name in ['AGP_composition_diagnostics.py', 'AGP_composition_report.py',
        'AGP_phylo_basis.py', 'AGP_ICFM_ci.py', 'AGP_ICFM_712_ci.py', 'AGP_ICFM_test_ci.py', 'AGP_712_data.py', 'AGP_ci_data.py']]
    source_names += ['utils/networks.py', 'utils/losses.py', 'utils/calibration.py', 'utils/__init__.py']
    for name in source_names:
        dest = out/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, dest)
    all_cells, all_metrics, all_history, completed = [], [], [], []
    for spec in specs:
        print(f"Fitting {spec['run_id']}", flush=True)
        run_start = time.monotonic()
        run = out/'runs'/spec['run_id']
        run.mkdir(parents=True)
        state = fit_transform(arrays, spec['method'], spec['epsilon'], bases.get(spec['method']))
        fit_arrays = {k: transform(v, state) for k, v in arrays.items() if k[1] in ('train', 'validation')}
        model = make_model(dim, spec['method'], spec['seed'], bases.get(spec['method']))
        history, training = fit_model(model, fit_arrays, args)
        training['elapsed_seconds'] = time.monotonic()-run_start
        torch.save({'state_dict': model.state_dict(), 'transform': state, 'spec': spec, 'training': training}, run/'model.pt')
        save_json(run/'training.json', training)
        history.to_csv(run/'loss_history.csv', index=False)
        for key, value in spec.items():
            history[key] = value
        all_history.append(history)
        scores = score_roles(model, fit_arrays, manifest, ('train', 'validation'))
        scores.to_csv(run/'fit_scores.csv', index=False)
        cells, metrics = summarize(scores, spec, training)
        all_cells.append(cells)
        all_metrics.append(metrics)
        completed.append((spec, training))
        if spec['method'] == 'raw' and spec['seed'] == prior['seed'] and not args.preflight:
            archived = torch.load(source/'model.pt', map_location='cpu', weights_only=False)
            require(training['best_epoch'] == source_result['training']['best_epoch'], 'Raw refit selected another epoch')
            require(all(torch.equal(value, archived['state_dict'][key]) for key, value in model.state_dict().items()),
                    'Raw reference refit does not reproduce archived checkpoint')
        print(f"Finished {spec['run_id']}: best={training['best_epoch']} val_gap={metrics.loc[metrics.role.eq('validation'), 'empirical_bin_absolute_gap'].iloc[0]:.6f}", flush=True)
        pd.concat(all_metrics, ignore_index=True).to_csv(out/'fit_metrics.csv', index=False)
    fit_metrics = pd.concat(all_metrics, ignore_index=True)
    eligible = fit_metrics[fit_metrics.role.eq('validation') & fit_metrics.primary]
    selected = eligible.sort_values(['balanced_brier', 'run_id']).iloc[0]
    protocol['selection'] = {'selected_run_id': selected.run_id,
        'criterion': 'validation balanced Brier; primary fits only', 'value': float(selected.balanced_brier),
        'test_scoring_started_at_selection': False}
    save_json(out/'selection.json', protocol['selection'])
    # Freeze the comparison before reading any new test predictions.
    for spec, training in completed:
        if not spec['primary']:
            continue
        run = out/'runs'/spec['run_id']
        checkpoint = torch.load(run/'model.pt', map_location='cpu', weights_only=False)
        model = make_model(dim, spec['method'], spec['seed'], bases.get(spec['method']))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        test_arrays = {k: transform(v, checkpoint['transform']) for k, v in arrays.items() if k[1] == 'test'}
        scores = score_roles(model, test_arrays, manifest, ('test',))
        scores.to_csv(run/'test_scores.csv', index=False)
        cells, metrics = summarize(scores, spec, training)
        all_cells.append(cells)
        all_metrics.append(metrics)
    archived_scores = pd.read_csv(source/'scores.csv', float_precision='round_trip')
    archived_spec = {'run_id': 'archived_raw', 'method': 'archived_raw', 'seed': prior['seed'], 'epsilon': 0., 'primary': True}
    cells, metrics = summarize(archived_scores, archived_spec, source_result['training'])
    all_cells.append(cells)
    all_metrics.append(metrics)
    cells, metrics, history = (pd.concat(parts, ignore_index=True) for parts in (all_cells, all_metrics, all_history))
    cells.to_csv(out/'diagnostic_cells.csv', index=False)
    metrics.to_csv(out/'metrics.csv', index=False)
    history.to_csv(out/'loss_history.csv', index=False)
    save_json(out/'protocol.json', protocol)
    from AGP_composition_report import render_report
    render_report(out, metrics, cells, history, protocol)
    artifacts = [p for p in out.rglob('*') if p.is_file() and '.matplotlib' not in p.parts]
    save_json(out/'result.json', {'status': 'complete', 'analysis': protocol['analysis'],
        'n_fits': len(completed), 'preflight': args.preflight, 'selected_run_id': selected.run_id,
        'elapsed_seconds': time.monotonic()-started,
        'outputs': {str(p.relative_to(out)): sha256(p) for p in sorted(artifacts)}})
    print(json.dumps({'status': 'complete', 'output': str(out), 'selection': protocol['selection']}, indent=2))


if __name__ == '__main__':
    main()
