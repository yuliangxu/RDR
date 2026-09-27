#!/usr/bin/env python3
"""Controlled ILR architecture comparison with validation diagnostics and merged CIs."""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from threadpoolctl import threadpool_limits
import torch
from torch import nn

import AGP_composition_diagnostics as composition
from AGP_ICFM_ci import DEFAULT_DATA, REPO, MLP, require, save_json, sha256
from AGP_ICFM_712_ci import verify_result
from AGP_712_data import prepare_712_data
from AGP_ICFM_merged_ci import calibrate_merged_counts, split_diagnostics, make_plots

DEFAULT_OUTPUT = Path('/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925')
DEFAULT_COMPOSITION = Path('/cwork/yx306/RDR/JRSSB/CI/agp_composition_20260922')
METHODS = ('baseline_p5', 'baseline_p30', 'wide_p30', 'deep_p30', 'residual_p30')
ARCHITECTURES = {
    'baseline_p5': '613 -> 32 -> 32 -> 32 -> 1; ReLU; patience 5',
    'baseline_p30': '613 -> 32 -> 32 -> 32 -> 1; ReLU; patience 30',
    'wide_p30': '613 -> 128 -> 128 -> 128 -> 1; ReLU; patience 30',
    'deep_p30': '613 -> 128; 3 two-linear 128-width blocks without skips; scalar head; patience 30',
    'residual_p30': '613 -> 128; 3 two-linear 128-width residual blocks; scalar head; patience 30',
}
EDGES = composition.EDGES
ROLES = composition.ROLES


class TwoLayerBlock(nn.Module):
    def __init__(self, width, residual):
        super().__init__()
        self.linear1 = nn.Linear(width, width)
        self.linear2 = nn.Linear(width, width)
        self.residual = residual

    def forward(self, x):
        y = self.linear2(torch.relu(self.linear1(x)))
        return torch.relu(y + x if self.residual else y)


def make_model(dimension, method, seed, basis):
    """Deep controls share all initial weights; only the skip additions differ."""
    require(method in METHODS, 'Unknown architecture')
    if method.startswith('baseline'):
        return composition.make_model(dimension, 'ilr', seed, basis)
    torch.manual_seed(seed)
    if method == 'wide_p30':
        model = MLP(dimension, hidden_dim=128, output_alpha=2)
        old = model.model[0]
        first = nn.Linear(dimension - 1, 128)
        with torch.no_grad():
            first.weight.copy_(old.weight @ torch.as_tensor(basis.T, dtype=old.weight.dtype))
            first.bias.copy_(old.bias)
        model.model[0] = first
        return model
    # Reuse the exact bounded output module; expose .model for shared fit metadata.
    model = MLP(dimension - 1, hidden_dim=128, output_alpha=2)
    head = model.model[-1]
    model.model = nn.Sequential(nn.Linear(dimension - 1, 128), nn.ReLU(),
        *(TwoLayerBlock(128, method == 'residual_p30') for _ in range(3)),
        nn.Linear(128, 1), head)
    return model


def read_frame(path):
    return pd.read_csv(path, float_precision='round_trip',
                       dtype={'sample_id': str, 'subject_id': str, 'row_sha256': str},
                       keep_default_na=False)


def source_paths():
    names = ['AGP_architecture_diagnostics.py', 'AGP_architecture_report.py',
             'test_AGP_architecture_diagnostics.py', 'AGP_composition_diagnostics.py',
             'AGP_ICFM_ci.py', 'AGP_ICFM_712_ci.py', 'AGP_ICFM_test_ci.py',
             'AGP_ICFM_merged_ci.py', 'AGP_712_data.py', 'AGP_ci_data.py']
    return ['experiments/AGP/' + n for n in names] + ['utils/networks.py', 'utils/losses.py',
        'utils/calibration.py', 'utils/__init__.py', 'experiments/JRSSB/CI/agent_CI.md']


def prepare(args):
    out, source, previous = args.output_dir.resolve(), args.source_fit.resolve(), args.source_composition.resolve()
    require(out.is_relative_to(Path('/tmp')) if args.preflight else out.is_relative_to(Path('/cwork/yx306/RDR')),
            'Use /tmp for preflight and /cwork/yx306/RDR for production')
    require(not out.exists(), 'Refusing to overwrite existing experiment')
    require(0 < args.min_epochs <= args.max_epochs, 'Invalid epoch limits')
    require(len(set(args.seeds)) == len(args.seeds) and all(s >= 0 for s in args.seeds), 'Invalid seeds')
    require(args.epsilon > 0 and args.threads > 0, 'Invalid fit controls')
    verify_result(source)
    prior_composition = verify_result(previous)
    prior = json.loads((source / 'protocol.json').read_text())
    arrays, manifest, audit = prepare_712_data(args.data_root, prior['seed'], prior['p_policy'])
    require(manifest.to_csv(index=False) == (source / 'split_manifest.csv').read_text(), 'Archived splits changed')
    require(audit['input_sha256'] == prior['data_audit']['input_sha256'], 'Source data changed')
    require(args.epsilon == 1e-6 or args.preflight, 'Production epsilon must match archived primary ILR')
    dimension = audit['feature_dimension']
    basis = helmert(dimension, full=False)
    state = composition.fit_transform(arrays, 'ilr', args.epsilon, basis)
    specs = [{'run_id': f'{method}_s{seed}', 'method': method, 'seed': seed,
              'epsilon': args.epsilon, 'primary': True, 'patience': 5 if method == 'baseline_p5' else 30}
             for seed in args.seeds for method in METHODS]
    out.mkdir(parents=True)
    (out / 'cache').mkdir()
    (out / 'runs').mkdir()
    shutil.copyfile(source / 'split_manifest.csv', out / 'split_manifest.csv')
    torch.save(state, out / 'transform.pt')
    for (distribution, role), values in arrays.items():
        # Hold test profiles in raw coordinates until all model selection is frozen.
        value = values if role == 'test' else composition.transform(values, state)
        np.save(out / 'cache' / f'{distribution}_{role}.npy', value, allow_pickle=False)
    for name in source_paths():
        path = REPO / name
        require(path.is_file(), f'Missing source snapshot: {path}')
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    baseline_references = {}
    for seed in args.seeds:
        name = f'runs/ilr_s{seed}_e{args.epsilon:g}/model.pt'
        require(name in prior_composition['outputs'], 'Missing archived ILR seed')
        baseline_references[str(seed)] = {'path': str(previous / name), 'sha256': prior_composition['outputs'][name]}
    protocol = {
        'analysis': 'agp_ilr_architecture_validation_and_merged_ci',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'source_fit': str(source),
        'source_composition': str(previous), 'source_result_sha256': sha256(source / 'result.json'),
        'source_composition_result_sha256': sha256(previous / 'result.json'),
        'candidate_specs': specs, 'architectures': ARCHITECTURES, 'data_audit': audit,
        'split_counts': audit['counts'], 'baseline_references': baseline_references,
        'config': {'seeds': args.seeds, 'epsilon': args.epsilon, 'threads': args.threads,
                   'min_epochs': args.min_epochs, 'max_epochs': args.max_epochs, 'preflight': args.preflight},
        'ci_config': {'alpha': .05, 'min_count': 40, 'bootstrap_repetitions': args.bootstrap_repetitions,
                      'seed': 20260920, 'n_candidates': 210, 'scope': 'simultaneous within each model; not joint across models'},
        'edges': EDGES.tolist(),
        'training': {'loss': '.5*mean_P(r^-1/2)+.25*mean_P(sqrt(r))+.25*mean_Q(sqrt(r))-1',
            'population_target': '2p/(p+q)', 'output': '2*sigmoid(2*a)', 'optimizer': 'full-batch AdamW',
            'learning_rate': .0005, 'weight_decay': .01, 'clip_max_norm': 1.,
            'scheduler': 'ReduceLROnPlateau factor=.5 patience=5 cooldown=2',
            'checkpoint': 'absolute minimum validation Hellinger', 'early_stopping_min_delta': 1e-5,
            'deep_control': 'same weights, width, linear layers and activation placement; skip addition only differs'},
        'selection_rule': 'minimum validation balanced Brier, ties by run_id; select global and within each method before test scoring',
        'limitations': audit['limitations'] + [
            'Existing test cohort already informed prior diagnostics; retrospective nominal intervals only.',
            'Validation influenced checkpoint and model selection; validation RDR curves are descriptive, without CIs.',
            'Seed repeats share one split and do not measure independent sampling variability.',
            'Networks induce different input-space cells despite shared numeric score boundaries.',
            'Per-model simultaneous intervals are not simultaneous across the 15 model fits.'],
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'host': platform.node()},
    }
    command = ['python3', '-B', str(out / 'source/experiments/AGP/AGP_architecture_diagnostics.py'),
        '--stage', 'all', '--output-dir', str(out.with_name(out.name + '_rerun')),
        '--source-fit', str(source), '--source-composition', str(previous), '--data-root', str(args.data_root),
        '--seeds', *map(str, args.seeds), '--epsilon', str(args.epsilon), '--threads', str(args.threads),
        '--min-epochs', str(args.min_epochs), '--max-epochs', str(args.max_epochs),
        '--bootstrap-repetitions', str(args.bootstrap_repetitions)]
    if args.preflight:
        command.append('--preflight')
    protocol['reproduction_command'] = shlex.join(command)
    save_json(out / 'protocol.json', protocol)
    files = [out / 'protocol.json', out / 'transform.pt', out / 'split_manifest.csv',
             *sorted((out / 'cache').glob('*.npy')), *sorted(p for p in (out / 'source').rglob('*') if p.is_file())]
    save_json(out / 'prepared.json', {'status': 'prepared', 'n_fits': len(specs),
        'outputs': {str(p.relative_to(out)): sha256(p) for p in files}})
    print(f'Prepared {len(specs)} fits at {out}', flush=True)


def load_prepared(out):
    prepared = json.loads((out / 'prepared.json').read_text())
    for name, digest in prepared['outputs'].items():
        require(sha256(out / name) == digest, f'Prepared artifact changed: {name}')
    return json.loads((out / 'protocol.json').read_text())


def fit_one(out, index):
    protocol = load_prepared(out)
    require(0 <= index < len(protocol['candidate_specs']), 'Invalid task index')
    spec = protocol['candidate_specs'][index]
    run = out / 'runs' / spec['run_id']
    if (run / 'result.json').exists():
        verify_result(run)
        print(f'Already complete: {spec["run_id"]}', flush=True)
        return
    require(not run.exists(), f'Partial fit exists: {run}; preserve it and use a fresh experiment')
    state = torch.load(out / 'transform.pt', weights_only=False, map_location='cpu')
    arrays = {(d, role): np.load(out / 'cache' / f'{d}_{role}.npy', allow_pickle=False)
              for d in ('P', 'Q') for role in ('train', 'validation')}
    manifest = read_frame(out / 'split_manifest.csv')
    model = make_model(state['dimension'], spec['method'], spec['seed'], state['basis'])
    args = SimpleNamespace(**{k: protocol['config'][k] for k in ('min_epochs', 'max_epochs')}, patience=spec['patience'])
    print(f'Fitting {index + 1}/{len(protocol["candidate_specs"])}: {spec["run_id"]}', flush=True)
    started = time.monotonic()
    history, training = composition.fit_model(model, arrays, args)
    training.update(elapsed_seconds=time.monotonic() - started, patience=spec['patience'])
    if spec['method'] == 'baseline_p5' and not protocol['config']['preflight']:
        reference = protocol['baseline_references'][str(spec['seed'])]
        require(sha256(reference['path']) == reference['sha256'], 'Archived baseline changed')
        old = torch.load(reference['path'], weights_only=False, map_location='cpu')
        require(training['best_epoch'] == old['training']['best_epoch'], 'Baseline checkpoint epoch changed')
        require(all(torch.equal(value, old['state_dict'][key]) for key, value in model.state_dict().items()),
                'Baseline does not reproduce archived ILR weights exactly')
        training['archived_checkpoint_exact_match'] = True
    run.mkdir()
    torch.save({'state_dict': model.state_dict(), 'spec': spec, 'training': training,
                'transform_sha256': sha256(out / 'transform.pt')}, run / 'model.pt')
    save_json(run / 'training.json', training)
    history.to_csv(run / 'loss_history.csv', index=False)
    scores = composition.score_roles(model, arrays, manifest, ('train', 'validation'))
    scores.to_csv(run / 'fit_scores.csv', index=False)
    cells, metrics = composition.summarize(scores, spec, training)
    metrics['parameter_count'] = training['parameter_count']
    cells.to_csv(run / 'fit_cells.csv', index=False)
    metrics.to_csv(run / 'fit_metrics.csv', index=False)
    save_json(run / 'result.json', {'status': 'complete', 'phase': 'fit_only', 'spec': spec,
        'test_scored': False, 'outputs': {p.name: sha256(p) for p in sorted(run.iterdir()) if p.is_file()}})
    print(f'Finished {spec["run_id"]}: epoch={training["best_epoch"]}, '
          f'val Brier={metrics.loc[metrics.role.eq("validation"), "balanced_brier"].iloc[0]:.7f}', flush=True)


def select_models(metrics):
    eligible = metrics.loc[metrics.role.eq('validation')].sort_values(['balanced_brier', 'run_id'])
    require(len(eligible) > 0 and np.isfinite(eligible.balanced_brier).all(), 'Invalid validation selection')
    require(not eligible.run_id.duplicated().any(), 'Duplicate validation run IDs')
    selected = eligible.iloc[0]
    return {'selected_run_id': selected.run_id, 'criterion': 'validation balanced Brier',
        'value': float(selected.balanced_brier),
        'by_method': {method: group.iloc[0].run_id for method, group in eligible.groupby('method', sort=False)},
        'test_scoring_started_at_selection': False, 'selected_utc': datetime.now(timezone.utc).isoformat()}


def merged_intervals(scores, spec, config):
    """Test/calibration only. Validation does not receive inferential intervals."""
    require(set(scores.role) == {'test'}, 'CIs require only frozen test/calibration scores')
    kp, kq = (np.bincount(scores.loc[scores.distribution.eq(d), 'bin_index'], minlength=20) for d in ('P', 'Q'))
    calibrated = calibrate_merged_counts(kp, kq, int(kp.sum()), int(kq.sum()),
        **{k: config[k] for k in ('alpha', 'min_count', 'bootstrap_repetitions', 'seed')})
    candidates = pd.DataFrame(calibrated['candidates'])
    candidates.insert(0, 'candidate_index', np.arange(len(candidates)))
    candidates['left'], candidates['right'] = EDGES[candidates.start], EDGES[candidates.stop]
    cells = candidates.iloc[calibrated['selected_indices']].copy().reset_index(drop=True)
    cells.insert(0, 'bin_index', np.arange(len(cells)))
    lookup = calibrated['elementary_to_merged']
    attached = scores.rename(columns={'bin_index': 'elementary_bin_index'}).copy()
    attached['bin_index'] = lookup[attached.elementary_bin_index.to_numpy()]
    attached = attached.merge(cells, on='bin_index', validate='many_to_one', how='left')
    require(len(attached) == len(scores) and attached.candidate_index.notna().all(), 'CI attachment mismatch')
    sums = np.zeros(len(cells))
    for d, n, counts in (('P', kp.sum(), cells.k_p), ('Q', kq.sum(), cells.k_q)):
        subset = attached[attached.distribution.eq(d)]
        require(np.array_equal(np.bincount(subset.bin_index, minlength=len(cells)), counts), 'Merged count mismatch')
        sums += np.bincount(subset.bin_index, weights=subset.rdr, minlength=len(cells)) / n
    cells['mixture_mass'] = (cells.p + cells.q) / 2
    cells['mean_model_score'] = sums / (cells.p + cells.q)
    cells['calibration_gap'] = cells.estimate - cells.mean_model_score
    for frame in (cells, candidates):
        for key in ('run_id', 'method', 'seed'):
            frame[key] = spec[key]
    summary = {key: spec[key] for key in ('run_id', 'method', 'seed')}
    summary['n_regions'] = len(cells)
    for method in ('c1', 'c2'):
        summary[method + '_mean_width'] = float(np.sum(cells.mixture_mass * (cells[method + '_upper'] - cells[method + '_lower'])))
        summary[method + '_regions_entirely_above_scores'] = int((cells[method + '_lower'] > cells.right).sum())
        summary[method + '_regions_entirely_below_scores'] = int((cells[method + '_upper'] < cells.left).sum())
    return cells, candidates, attached, summary, calibrated


def aggregate(out):
    protocol = load_prepared(out)
    require(not (out / 'result.json').exists(), 'Experiment already complete')
    all_metrics, all_cells, all_history, checkpoints = [], [], [], {}
    for spec in protocol['candidate_specs']:
        run = out / 'runs' / spec['run_id']
        verify_result(run)
        all_metrics.append(pd.read_csv(run / 'fit_metrics.csv', float_precision='round_trip'))
        all_cells.append(pd.read_csv(run / 'fit_cells.csv', float_precision='round_trip'))
        history = pd.read_csv(run / 'loss_history.csv', float_precision='round_trip')
        for key, value in spec.items():
            history[key] = value
        all_history.append(history)
        checkpoints[spec['run_id']] = sha256(run / 'model.pt')
    fit_metrics = pd.concat(all_metrics, ignore_index=True)
    require(set(fit_metrics.run_id) == set(checkpoints), 'Incomplete fitting comparison')
    selection = select_models(fit_metrics)
    selection['model_sha256'] = checkpoints
    require(not (out / 'selection.json').exists(), 'Selection already frozen; use a fresh output for retry')
    save_json(out / 'selection.json', selection)
    fit_metrics.to_csv(out / 'fit_metrics.csv', index=False)
    protocol['selection'] = selection
    state = torch.load(out / 'transform.pt', weights_only=False, map_location='cpu')
    manifest = read_frame(out / 'split_manifest.csv')
    test_arrays = {(d, 'test'): composition.transform(np.load(out / 'cache' / f'{d}_test.npy', allow_pickle=False), state)
                   for d in ('P', 'Q')}
    all_ci, all_candidates, all_ci_summary, all_merged_diagnostics = [], [], [], []
    for spec in protocol['candidate_specs']:
        run = out / 'runs' / spec['run_id']
        require(sha256(run / 'model.pt') == checkpoints[spec['run_id']], 'Frozen checkpoint changed')
        checkpoint = torch.load(run / 'model.pt', weights_only=False, map_location='cpu')
        model = make_model(state['dimension'], spec['method'], spec['seed'], state['basis'])
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        scores = composition.score_roles(model, test_arrays, manifest, ('test',))
        scores.to_csv(run / 'test_scores.csv', index=False)
        cells, metrics = composition.summarize(scores, spec, checkpoint['training'])
        metrics['parameter_count'] = checkpoint['training']['parameter_count']
        all_cells.append(cells)
        all_metrics.append(metrics)
        ci_cells, candidates, attached, summary, calibrated = merged_intervals(scores, spec, protocol['ci_config'])
        all_ci.append(ci_cells)
        all_candidates.append(candidates)
        all_ci_summary.append(summary)
        candidate_path = run / 'candidate_intervals.csv'
        candidates.to_csv(candidate_path, index=False)
        ci_cells.to_csv(run / 'cell_intervals.csv', index=False)
        attached.to_csv(run / 'test_intervals.csv', index=False)
        np.save(run / 'bootstrap_maxima.npy', calibrated['bootstrap_maxima'], allow_pickle=False)
        save_json(run / 'ci_metadata.json', calibrated['metadata'])
        common_scores = pd.concat([read_frame(run / 'fit_scores.csv'), scores], ignore_index=True)
        diagnostics, diagnostic_summary = split_diagnostics(common_scores, ci_cells, calibrated['elementary_to_merged'])
        for key in ('run_id', 'method', 'seed'):
            diagnostics[key] = spec[key]
        diagnostics.to_csv(run / 'merged_split_diagnostics.csv', index=False)
        all_merged_diagnostics.append(diagnostics)
        if spec['run_id'] in selection['by_method'].values():
            make_plots(run, ci_cells, diagnostics, calibrated['metadata'])
        print(f'Calibrated {spec["run_id"]}: {len(ci_cells)} regions', flush=True)
    metrics, cells, history = (pd.concat(parts, ignore_index=True) for parts in (all_metrics, all_cells, all_history))
    ci_cells, ci_summary = pd.concat(all_ci, ignore_index=True), pd.DataFrame(all_ci_summary)
    for name, frame in [('metrics', metrics), ('diagnostic_cells', cells), ('loss_history', history),
                        ('ci_cells', ci_cells), ('ci_summary', ci_summary),
                        ('ci_candidates', pd.concat(all_candidates, ignore_index=True)),
                        ('merged_split_diagnostics', pd.concat(all_merged_diagnostics, ignore_index=True))]:
        frame.to_csv(out / f'{name}.csv', index=False)
    # Keep the immutable fitting protocol hash intact; final reporting extends it separately.
    save_json(out / 'report_protocol.json', protocol)
    from AGP_architecture_report import render_report
    render_report(out, metrics, cells, history, protocol, ci_cells, ci_summary)
    artifacts = sorted(p for p in out.rglob('*') if p.is_file() and '.matplotlib' not in p.parts
                       and p.name != 'run.log' and p != out / 'result.json')
    save_json(out / 'result.json', {'status': 'complete', 'analysis': protocol['analysis'],
        'n_fits': len(protocol['candidate_specs']), 'selected_run_id': selection['selected_run_id'],
        'preflight': protocol['config']['preflight'], 'completed_utc': datetime.now(timezone.utc).isoformat(),
        'outputs': {str(p.relative_to(out)): sha256(p) for p in artifacts}})
    print(json.dumps({'status': 'complete', 'output': str(out), 'selection': selection}, indent=2), flush=True)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'prepare', 'fit', 'aggregate'), default='all')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--source-fit', type=Path, default=composition.DEFAULT_FIT)
    parser.add_argument('--source-composition', type=Path, default=DEFAULT_COMPOSITION)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--seeds', nargs='+', type=int, default=[20260918, 20260919, 20260920])
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--min-epochs', type=int, default=300)
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    parser.add_argument('--task-index', type=int)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if args.stage in ('fit', 'aggregate'):
        args.threads = json.loads((out / 'protocol.json').read_text())['config']['threads']
    torch.set_num_threads(args.threads)
    threadpool_limits(limits=args.threads)
    torch.use_deterministic_algorithms(True)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/agp-architecture-matplotlib')
    if args.stage in ('prepare', 'all'):
        prepare(args)
    with (out / 'run.log').open('a', buffering=1) as log, \
            contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
        if args.stage == 'all':
            n = len(json.loads((out / 'protocol.json').read_text())['candidate_specs'])
            for index in range(n):
                fit_one(out, index)
            aggregate(out)
        elif args.stage == 'fit':
            require(args.task_index is not None, 'fit requires --task-index')
            fit_one(out, args.task_index)
        elif args.stage == 'aggregate':
            aggregate(out)


if __name__ == '__main__':
    main()
