#!/usr/bin/env python3
"""CelebA paired loss/slope selection and prespecified architecture sensitivity.

All stages use frozen source, explicit completion receipts and isolated outputs.
Test assets are materialized only after the entire adaptive selection is locked.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/celeba-selection-mpl")

import numpy as np
from utils.model_selection import choose_configuration

DEFAULT_SOURCE = Path('/cwork/yx306/RDR/JRSSB/ddim_diffusion_stylegan2_equal_fid')
DEFAULT_CONFIG = Path(__file__).with_name('selection_config.json')
FIT_FILES = ('model.pt', 'predictions.npz', 'metrics.json', 'cells.json', 'history.json')
EVAL_FILES = ('predictions.npz', 'metrics.json', 'cells.json')
TRAIN_ROLES = ('train', 'earlystop', 'selection_calibration', 'selection_evaluation')
TEST_ROLES = ('test_calibration', 'test_evaluation')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seal(path, value):
    path = Path(path)
    if path.exists():
        if read_sealed(path) != value:
            raise ValueError(f'Refusing to change frozen record: {path}')
        return
    write(path, value)
    path.with_suffix('.sha256').write_text(sha(path) + '\n')


def read_sealed(path):
    path = Path(path)
    if sha(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError(f'Frozen record changed: {path}')
    return read(path)


def validate_config(cfg):
    for field, allowed in [('levels', {'feature', 'pixel'}), ('branches', {'lower', 'upper'}),
                           ('losses', {'hellinger', 'kl', 'chisq', 'js'}),
                           ('output_alphas', {.5, 1, 2, 4})]:
        values = cfg[field]
        if not values or len(set(values)) != len(values) or not set(values) <= allowed:
            raise ValueError(f'Invalid {field}')
    if cfg['repeats'] < 2 or cfg['bins'] != 20 or not 0 < cfg['ci_alpha'] < 1:
        raise ValueError('Require paired repeats, 20 fixed cells, and valid CI alpha')
    for level in cfg['levels']:
        recipe = cfg[level]
        if not 1 <= recipe['min_epochs'] <= recipe['max_epochs'] or recipe['batch_size'] < 2:
            raise ValueError(f'Invalid training recipe: {level}')
    sensitivity = cfg['sensitivity']
    if (sensitivity['leaders_per_pair'] != 2 or sensitivity['architecture'] != 'small'
            or sensitivity['expansion_minimum_brier_gain'] <= 0
            or not 0 < sensitivity['expansion_positive_repeat_fraction'] <= 1):
        raise ValueError('Invalid prespecified architecture sensitivity')


def configuration_id(loss, alpha):
    return f'{loss}_a{alpha:g}'.replace('.', 'p')


def grid(cfg, architecture='baseline', levels=None, settings=None):
    tasks = []
    for representation in levels or cfg['levels']:
        candidates = settings[representation] if settings else [
            {'loss': loss, 'output_alpha': alpha}
            for loss in cfg['losses'] for alpha in cfg['output_alphas']]
        for branch in cfg['branches']:
            for repeat in range(cfg['repeats']):
                for candidate in candidates:
                    loss, alpha = candidate['loss'], candidate['output_alpha']
                    tasks.append(dict(task_index=len(tasks), representation=representation,
                                      branch=branch, repeat=repeat, architecture=architecture,
                                      candidate=f'{architecture}_{configuration_id(loss, alpha)}',
                                      loss=loss, output_alpha=alpha))
    return tasks


def task_identity(task):
    return tuple(task[key] for key in ('representation', 'branch', 'candidate', 'repeat'))


def task_directory(output, task, evaluation=False):
    return (output / ('evaluation' if evaluation else 'fits') / task['representation'] /
            task['branch'] / task['candidate'] / f"repeat_{task['repeat']:02d}")


def prepare(output, config, source_root, smoke=False):
    from experiments.CelebA.selection_data import prepare_data
    cfg = read(config)
    if smoke:
        cfg.update(repeats=2, output_alphas=[.5, 4], smoke_rows=64)
        for level in cfg['levels']:
            cfg[level].update(batch_size=32, evaluation_batch_size=128, max_epochs=2,
                              min_epochs=1, patience=1)
        cfg['feature'].update(hidden_dimension=8, small_hidden_dimension=4)
        cfg['pixel'].update(ndf=4, small_ndf=2, bn_freeze_epoch=1)
    validate_config(cfg)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    sources = list((ROOT / 'utils').glob('*.py')) + [ROOT / 'experiments/__init__.py']
    sources += [p for p in (ROOT / 'experiments/CelebA').iterdir()
                if p.suffix in ('.py', '.json', '.slurm', '.md')]
    files = {}
    for source in sorted(sources):
        relative = 'source/' + str(source.relative_to(ROOT))
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[relative] = sha(target)
    metadata = prepare_data(output, source_root, cfg, smoke=smoke)
    files['data/metadata.json'] = sha(output / 'data/metadata.json')
    protocol = dict(schema=1, study='celeba_paired_loss_activation_architecture_selection',
                    smoke=smoke, config=cfg, files=files, data=metadata,
                    tasks=grid(cfg), created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    target='r=2p/(p+q), balanced P=1/Q=0 probability r/2',
                    checkpoint_rule='Minimum stopping-set balanced Brier, restored before selection scoring',
                    selection_rule='Paired one-SE Brier shortlist, then minimum supported local absolute gap; support>=.99 each repeat',
                    sensitivity_rule=('Not included: fixed-architecture loss/activation selection only' if cfg.get('selection_only') else
                                      'Union of two leading settings per pair, narrower network, same settings on both pairs; expand only using prespecified selection-Brier gain, paired SE and repeat-consistency thresholds'),
                    test_scope='Reserved retrospective calibration/evaluation pool; exact counts in data metadata; never used by selection',
                    ci_scope='Nominal image-level cell-average intervals; no identity-cluster, neural-mean, or candidate-selection uncertainty correction',
                    repeat_scope='Initialization and ordering variation conditional on fixed real/generated samples; not independent-data uncertainty',
                    source_root=str(source_root), python=sys.version,
                    packages={p: importlib.metadata.version(p) for p in ('numpy', 'torch', 'pandas', 'scipy', 'matplotlib')})
    seal(output / 'protocol.json', protocol)
    report(output)
    print(json.dumps(dict(prepared=str(output), baseline_fits=len(protocol['tasks']), smoke=smoke)))


def load_protocol(output):
    protocol = read_sealed(output / 'protocol.json')
    for name, expected in protocol['files'].items():
        if sha(output / name) != expected:
            raise ValueError(f'Frozen source/metadata changed: {name}')
        if name.startswith('source/') and name.endswith('.py'):
            source = ROOT / Path(name).relative_to('source')
            if not source.exists() or sha(source) != expected:
                raise ValueError(f'Use the frozen source entrypoint; executing source differs: {name}')
    return protocol


def completed(directory, task, protocol_hash, evaluation=False):
    path = directory / 'COMPLETE.json'
    if not path.exists():
        return False
    receipt = read(path)
    if (receipt['task_identity'] != list(task_identity(task))
            or receipt['protocol_sha256'] != protocol_hash):
        raise ValueError(f'Completion provenance mismatch: {directory}')
    expected_files = EVAL_FILES if evaluation else FIT_FILES
    if set(receipt['files']) != set(expected_files):
        raise ValueError(f'Incomplete receipt: {directory}')
    for name, digest in receipt['files'].items():
        if sha(directory / name) != digest:
            raise ValueError(f'Completed artifact changed: {directory / name}')
    return True


def tasks_for(output, protocol, stage):
    if stage == 'baseline':
        return protocol['tasks']
    record = read_sealed(output / f'{stage}_tasks.json')
    if stage == 'sensitivity':
        parent = read_sealed(output / 'baseline_selection.json')
        valid = (parent['protocol_sha256'] == sha(output / 'protocol.json') and
                 record['baseline_selection_sha256'] == sha(output / 'baseline_selection.json'))
    elif stage == 'expanded':
        parent = read_sealed(output / 'expansion.json')
        valid = (parent['protocol_sha256'] == sha(output / 'protocol.json') and
                 parent['baseline_selection_sha256'] == sha(output / 'baseline_selection.json') and
                 parent['sensitivity_tasks_sha256'] == sha(output / 'sensitivity_tasks.json') and
                 record['expansion_sha256'] == sha(output / 'expansion.json'))
    else:
        parent = read_sealed(output / 'freeze.json')
        valid = (parent['protocol_sha256'] == sha(output / 'protocol.json') and
                 record['freeze_sha256'] == sha(output / 'freeze.json'))
    if not valid:
        raise ValueError(f'Frozen {stage} task lineage changed')
    return record['tasks']


def fit(output, stage, index, device):
    from experiments.CelebA.selection_data import load_arrays
    from experiments.CelebA.selection_training import fit_model
    protocol = load_protocol(output)
    tasks = tasks_for(output, protocol, stage)
    if not 0 <= index < len(tasks):
        raise ValueError('Task index outside frozen task list')
    task = tasks[index]
    directory = task_directory(output, task)
    digest = sha(output / 'protocol.json')
    if completed(directory, task, digest):
        print(f'Verified complete: {directory}')
        return
    directory.parent.mkdir(parents=True, exist_ok=True)
    if directory.exists():
        raise RuntimeError(f'Incomplete fit preserved at {directory}; inspect before retrying')
    directory.mkdir()
    try:
        nested = load_arrays(output, task['representation'], task['branch'], TRAIN_ROLES)
        arrays = {f'{role}_{side}': values for role, sides in nested.items() for side, values in sides.items()}
        metrics = fit_model(task, protocol['config'], arrays, directory, device)
        write(directory / 'metrics.json', {**metrics, **task})
        write(directory / 'COMPLETE.json', dict(task_identity=task_identity(task), protocol_sha256=digest,
                                               files={name: sha(directory / name) for name in FIT_FILES}))
    except Exception as error:
        write(directory / 'FAILED.json', dict(type=type(error).__name__, message=str(error)))
        raise


def collect(output, protocol, tasks, require_complete=True):
    rows = []
    digest = sha(output / 'protocol.json')
    identities = set()
    for task in tasks:
        identity = task_identity(task)
        if identity in identities:
            raise ValueError('Duplicate frozen task')
        identities.add(identity)
        directory = task_directory(output, task)
        if not completed(directory, task, digest):
            if require_complete:
                raise ValueError(f'Missing completed task: {directory}')
            continue
        row = read(directory / 'metrics.json')
        if task_identity(row) != identity:
            raise ValueError('Metric/task identity mismatch')
        rows.append(row)
    return rows


def selections(rows, cfg):
    result = {}
    for level in cfg['levels']:
        for branch in cfg['branches']:
            subset = [r for r in rows if r['representation'] == level and r['branch'] == branch]
            selection = choose_configuration(subset)
            for candidate in [selection['chosen']] + selection['ranking']:
                candidate['architecture'] = next(r['architecture'] for r in subset if r['candidate'] == candidate['candidate'])
            result[f'{level}/{branch}'] = selection
    return result


def shared_configurations(selected, cfg):
    result = {}
    for level in cfg['levels']:
        rankings = [{r['candidate']: r for r in selected[f'{level}/{b}']['ranking'] if r['eligible']}
                    for b in cfg['branches']]
        common = set.intersection(*(set(r) for r in rankings))
        ordered = sorted(common, key=lambda c: (np.mean([r[c]['mean_local_gap'] for r in rankings]),
                                                np.mean([r[c]['mean_brier'] for r in rankings]), c))
        result[level] = dict(eligible_on_both=ordered, candidate=ordered[0] if ordered else None,
                             rule='Intersection of per-pair eligible sets; mean local gap then mean Brier; descriptive common-setting option')
    return result


def sensitivity_settings(selected, cfg):
    settings = {}
    for level in cfg['levels']:
        leaders = {}
        for branch in cfg['branches']:
            selection = selected[f'{level}/{branch}']
            ranking = selection['ranking']
            eligible_support = sorted([r for r in ranking if r['support_eligible']],
                                      key=lambda r: (r['mean_brier'], r['candidate']))
            picked = list(dict.fromkeys([selection['chosen']['candidate'], selection['brier_reference']]))
            for row in eligible_support:
                if row['candidate'] not in picked and len(picked) < cfg['sensitivity']['leaders_per_pair']:
                    picked.append(row['candidate'])
            for candidate in picked:
                row = next(r for r in ranking if r['candidate'] == candidate)
                leaders[configuration_id(row['loss'], row['output_alpha'])] = {
                    'loss': row['loss'], 'output_alpha': row['output_alpha']}
        settings[level] = [leaders[key] for key in sorted(leaders)]
    return settings


def expansion_decision(baseline_rows, sensitivity_rows, selected, cfg):
    diagnostics, expanded = [], set()
    rule = cfg['sensitivity']
    for level in cfg['levels']:
        for branch in cfg['branches']:
            chosen = selected[f'{level}/{branch}']['chosen']['candidate']
            base = sorted([r for r in baseline_rows if r['representation'] == level and r['branch'] == branch
                           and r['candidate'] == chosen], key=lambda r: r['repeat'])
            alternatives = [r for r in sensitivity_rows if r['representation'] == level and r['branch'] == branch]
            for candidate in sorted({r['candidate'] for r in alternatives}):
                rows = sorted([r for r in alternatives if r['candidate'] == candidate], key=lambda r: r['repeat'])
                if [r['repeat'] for r in base] != list(range(cfg['repeats'])) or [r['repeat'] for r in rows] != list(range(cfg['repeats'])):
                    raise ValueError('Expansion requires complete paired repeats')
                gain = np.array([a['brier'] - b['brier'] for a, b in zip(base, rows)])
                mean, se = float(gain.mean()), float(gain.std(ddof=1) / math.sqrt(len(gain)))
                positive = float(np.mean(gain > 0))
                support = all(r['supported_mass'] >= .99 for r in rows)
                trigger = bool(support and mean >= rule['expansion_minimum_brier_gain']
                               and mean > rule['expansion_paired_se_multiplier'] * se
                               and positive >= rule['expansion_positive_repeat_fraction'])
                diagnostics.append(dict(representation=level, branch=branch, candidate=candidate,
                                        baseline=chosen, mean_brier_gain=mean, paired_se=se,
                                        positive_repeat_fraction=positive, support_eligible=support, expand=trigger))
                if trigger:
                    expanded.add(level)
    return dict(expanded_representations=sorted(expanded), diagnostics=diagnostics, rule=rule)


def freeze_baseline(output, protocol):
    rows = collect(output, protocol, protocol['tasks'])
    selected = selections(rows, protocol['config'])
    record = dict(selections=selected, shared_configurations=shared_configurations(selected, protocol['config']),
                  protocol_sha256=sha(output / 'protocol.json'),
                  fit_receipts={str(task_directory(output, t).relative_to(output)): sha(task_directory(output, t) / 'COMPLETE.json')
                                for t in protocol['tasks']})
    seal(output / 'baseline_selection.json', record)
    if protocol['config'].get('selection_only'):
        return
    settings = sensitivity_settings(selected, protocol['config'])
    seal(output / 'sensitivity_tasks.json', dict(tasks=grid(protocol['config'], 'small', settings=settings), settings=settings,
                                                baseline_selection_sha256=sha(output / 'baseline_selection.json')))


def freeze_final(output, protocol):
    baseline = read_sealed(output / 'baseline_selection.json')
    decision = read_sealed(output / 'expansion.json')
    if (baseline['protocol_sha256'] != sha(output / 'protocol.json') or
            decision['protocol_sha256'] != sha(output / 'protocol.json') or
            decision['baseline_selection_sha256'] != sha(output / 'baseline_selection.json') or
            decision['sensitivity_tasks_sha256'] != sha(output / 'sensitivity_tasks.json')):
        raise ValueError('Final selection parent lineage changed')
    for directory, digest in baseline['fit_receipts'].items():
        if sha(output / directory / 'COMPLETE.json') != digest:
            raise ValueError('Baseline fit changed after first-stage selection')
    rows = collect(output, protocol, protocol['tasks'])
    sensitivity = collect(output, protocol, tasks_for(output, protocol, 'sensitivity'))
    expanded_levels = decision['expanded_representations']
    if expanded_levels:
        expansion = collect(output, protocol, tasks_for(output, protocol, 'expanded'))
        rows += [r for r in sensitivity if r['representation'] in expanded_levels] + expansion
        expected = grid(protocol['config'], 'small', levels=expanded_levels)
        if {task_identity(t) for t in expected} != {task_identity(r) for r in rows if r['architecture'] == 'small'}:
            raise ValueError('Expanded architecture grid is incomplete')
    selected = selections(rows, protocol['config'])
    record = dict(selections=selected, shared_configurations=shared_configurations(selected, protocol['config']),
                  expansion=decision, baseline_selection_sha256=sha(output / 'baseline_selection.json'),
                  protocol_sha256=sha(output / 'protocol.json'),
                  primary_rule='Baseline remains primary unless prespecified sensitivity triggers the full alternative grid')
    tasks = []
    for level in protocol['config']['levels']:
        for branch in protocol['config']['branches']:
            chosen = selected[f'{level}/{branch}']['chosen']
            for repeat in range(protocol['config']['repeats']):
                tasks.append(dict(task_index=len(tasks), representation=level, branch=branch, repeat=repeat, **chosen))
    record['selected_fit_receipts'] = {str(task_directory(output, t).relative_to(output)): sha(task_directory(output, t) / 'COMPLETE.json')
                                      for t in tasks}
    seal(output / 'freeze.json', record)
    seal(output / 'evaluation_tasks.json', dict(tasks=tasks, freeze_sha256=sha(output / 'freeze.json')))


def evaluate(output, index, device):
    from experiments.CelebA.selection_data import load_arrays
    from experiments.CelebA.selection_training import evaluate_model
    protocol = load_protocol(output)
    frozen = read_sealed(output / 'freeze.json')
    record = read_sealed(output / 'evaluation_tasks.json')
    if record['freeze_sha256'] != sha(output / 'freeze.json') or frozen['protocol_sha256'] != sha(output / 'protocol.json'):
        raise ValueError('Evaluation lock provenance changed')
    task = record['tasks'][index]
    directory = task_directory(output, task, evaluation=True)
    digest = sha(output / 'protocol.json')
    if completed(directory, task, digest, evaluation=True):
        return
    checkpoint = task_directory(output, task)
    if sha(checkpoint / 'COMPLETE.json') != frozen['selected_fit_receipts'][str(checkpoint.relative_to(output))]:
        raise ValueError('Selected checkpoint receipt changed after freeze')
    if not completed(checkpoint, task, digest):
        raise ValueError('Selected fit is incomplete')
    evaluation_data = read_sealed(output / 'evaluation_data.json')
    if (evaluation_data['freeze_sha256'] != sha(output / 'freeze.json') or
            evaluation_data['metadata_sha256'] != sha(output / 'data/evaluation_metadata.json')):
        raise ValueError('Evaluation data provenance changed')
    directory.mkdir(parents=True, exist_ok=False)
    nested = load_arrays(output, task['representation'], task['branch'], TEST_ROLES)
    arrays = {f'{role}_{side}': value for role, sides in nested.items() for side, value in sides.items()}
    metrics = evaluate_model(task, protocol['config'], arrays, checkpoint, directory, device)
    write(directory / 'metrics.json', {**metrics, **task})
    write(directory / 'COMPLETE.json', dict(task_identity=task_identity(task), protocol_sha256=digest,
                                           files={name: sha(directory / name) for name in EVAL_FILES}))


def report(output, require_complete=False):
    from experiments.CelebA.selection_report import build_report
    protocol = load_protocol(output)
    if protocol['config'].get('selection_only'):
        from experiments.CelebA.selection_only_report import build_report as build_selection_report
        rows = collect(output, protocol, protocol['tasks'], require_complete=require_complete)
        selected = read_sealed(output / 'baseline_selection.json') if (output / 'baseline_selection.json').exists() else None
        status = build_selection_report(output, protocol, rows, selected)
        if require_complete:
            if not status['complete']:
                raise ValueError('Loss/activation selection is incomplete')
            write(output / 'COMPLETE.json', dict(scope='loss_and_activation_selection_only',
                  protocol_sha256=sha(output / 'protocol.json'),
                  selection_sha256=sha(output / 'baseline_selection.json'),
                  fits=len(rows), test_assets_loaded=False, report_sha256=sha(output / 'RESULTS.md')))
        return
    tasks = list(protocol['tasks'])
    for stage in ('sensitivity', 'expanded'):
        if (output / f'{stage}_tasks.json').exists():
            tasks += tasks_for(output, protocol, stage)
    rows = collect(output, protocol, tasks, require_complete=require_complete)
    evaluations = []
    if (output / 'evaluation_tasks.json').exists():
        for task in read_sealed(output / 'evaluation_tasks.json')['tasks']:
            directory = task_directory(output, task, evaluation=True)
            if completed(directory, task, sha(output / 'protocol.json'), evaluation=True):
                evaluations.append(read(directory / 'metrics.json'))
            elif require_complete:
                raise ValueError(f'Incomplete selected assessment: {directory}')
    elif require_complete:
        raise ValueError('Selection has not been frozen')
    baseline = read_sealed(output / 'baseline_selection.json') if (output / 'baseline_selection.json').exists() else None
    frozen = read_sealed(output / 'freeze.json') if (output / 'freeze.json').exists() else None
    status = build_report(output, protocol, rows, baseline, frozen, evaluations)
    if require_complete:
        if not status['complete']:
            raise ValueError('Report verification found incomplete or inconsistent study records')
        write(output / 'COMPLETE.json', dict(protocol_sha256=sha(output / 'protocol.json'),
                                            freeze_sha256=sha(output / 'freeze.json'), fits=len(rows),
                                            evaluations=len(evaluations), report_sha256=sha(output / 'RESULTS.md')))


def submit_stage(output, protocol, stage, concurrency=4):
    if (output / 'jobs').exists() and any((output / 'jobs').glob(f'{stage}-*.json')):
        raise RuntimeError('Stage already submitted; inspect job records before resubmission')
    tasks = tasks_for(output, protocol, stage)
    wrapper = output / 'source/experiments/CelebA/selection.slurm'
    jobs = []
    for level in protocol['config']['levels']:
        indices = [str(i) for i, task in enumerate(tasks) if task['representation'] == level]
        if not indices:
            continue
        command = ['sbatch', '--parsable', f'--array={",".join(indices)}%{concurrency}',
                   f'--mem={"24G" if level == "feature" else "48G"}',
                   f'--time={"03:00:00" if level == "feature" else "08:00:00"}',
                   f'--output={output}/logs/{stage}-%A_%a.out', f'--error={output}/logs/{stage}-%A_%a.err',
                   str(wrapper), str(output), 'evaluate' if stage == 'evaluation' else 'fit', stage]
        job = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError(f'Invalid Slurm job ID: {job}')
        jobs.append(job)
        write(output / 'jobs' / f'{stage}-{level}-{job}.json', dict(job_id=job, stage=stage, representation=level, command=command))
    if not jobs:
        raise ValueError('No tasks to submit')
    command = ['sbatch', '--parsable', '--gres=gpu:0', '--mem=24G', '--time=03:00:00',
               f'--dependency=afterok:{":".join(jobs)}',
               f'--output={output}/logs/advance-{stage}-%j.out', f'--error={output}/logs/advance-{stage}-%j.err',
               str(wrapper), str(output), 'advance', stage, str(concurrency)]
    followup = subprocess.check_output(command, text=True).strip().split(';')[0]
    write(output / 'jobs' / f'advance-{stage}-{followup}.json', dict(job_id=followup, stage=stage, command=command))
    print(json.dumps(dict(submitted=stage, arrays=jobs, followup=followup)))


def advance(output, stage, submit=False, concurrency=4):
    from experiments.CelebA.selection_data import prepare_evaluation_data
    protocol = load_protocol(output)
    if protocol['config'].get('selection_only'):
        if stage != 'baseline':
            raise ValueError('Only loss/activation selection is authorized for this run')
        freeze_baseline(output, protocol)
        report(output, require_complete=True)
        return
    if stage == 'baseline':
        freeze_baseline(output, protocol)
        next_stage = 'sensitivity'
    elif stage == 'sensitivity':
        baseline = collect(output, protocol, protocol['tasks'])
        small = collect(output, protocol, tasks_for(output, protocol, 'sensitivity'))
        selected = read_sealed(output / 'baseline_selection.json')['selections']
        decision = expansion_decision(baseline, small, selected, protocol['config'])
        decision.update(protocol_sha256=sha(output / 'protocol.json'),
                        baseline_selection_sha256=sha(output / 'baseline_selection.json'),
                        sensitivity_tasks_sha256=sha(output / 'sensitivity_tasks.json'))
        seal(output / 'expansion.json', decision)
        expanded = decision['expanded_representations']
        if expanded:
            seen = {task_identity(t) for t in tasks_for(output, protocol, 'sensitivity')}
            tasks = [t for t in grid(protocol['config'], 'small', levels=expanded) if task_identity(t) not in seen]
            for i, task in enumerate(tasks):
                task['task_index'] = i
            seal(output / 'expanded_tasks.json', dict(tasks=tasks, expansion_sha256=sha(output / 'expansion.json')))
            next_stage = 'expanded' if tasks else 'evaluation'
        else:
            next_stage = 'evaluation'
        if next_stage == 'evaluation':
            freeze_final(output, protocol)
            prepare_evaluation_data(output)
    elif stage == 'expanded':
        freeze_final(output, protocol)
        prepare_evaluation_data(output)
        next_stage = 'evaluation'
    elif stage == 'evaluation':
        report(output, require_complete=True)
        return
    else:
        raise ValueError(stage)
    if next_stage == 'evaluation':
        seal(output / 'evaluation_data.json', dict(freeze_sha256=sha(output / 'freeze.json'),
                                                  metadata_sha256=sha(output / 'data/evaluation_metadata.json')))
    report(output)
    if submit:
        submit_stage(output, protocol, next_stage, concurrency)
    else:
        print(json.dumps(dict(next_stage=next_stage)))


def run(output, device):
    stages = ('baseline',) if load_protocol(output)['config'].get('selection_only') else ('baseline', 'sensitivity', 'expanded', 'evaluation')
    for stage in stages:
        protocol = load_protocol(output)
        if stage == 'expanded' and not (output / 'expanded_tasks.json').exists():
            continue
        for index in range(len(tasks_for(output, protocol, stage))):
            (evaluate(output, index, device) if stage == 'evaluation' else fit(output, stage, index, device))
        advance(output, stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'fit', 'advance', 'evaluate', 'report', 'run', 'submit', 'status'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--source-root', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--stage', default='baseline', choices=['baseline', 'sensitivity', 'expanded', 'evaluation'])
    parser.add_argument('--task-index', type=int, default=0)
    parser.add_argument('--submit-next', action='store_true')
    parser.add_argument('--concurrency', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.concurrency < 1:
        parser.error('Concurrency must be positive')
    if args.command == 'prepare':
        prepare(output, args.config, args.source_root.resolve(), args.smoke)
    elif args.command == 'fit':
        fit(output, args.stage, args.task_index, args.device)
    elif args.command == 'evaluate':
        evaluate(output, args.task_index, args.device)
    elif args.command == 'advance':
        advance(output, args.stage, args.submit_next, args.concurrency)
    elif args.command == 'run':
        run(output, args.device)
    elif args.command == 'report':
        report(output)
    elif args.command == 'submit':
        protocol = load_protocol(output)
        if (output / 'jobs').exists() and any((output / 'jobs').glob(f'{args.stage}-*.json')):
            raise RuntimeError('Stage already submitted; inspect job records before resubmission')
        submit_stage(output, protocol, args.stage, args.concurrency)
    else:
        protocol = load_protocol(output)
        for stage in ('baseline', 'sensitivity', 'expanded', 'evaluation'):
            if stage != 'baseline' and not (output / f'{stage}_tasks.json').exists():
                continue
            tasks = tasks_for(output, protocol, stage)
            done = sum(completed(task_directory(output, t, stage == 'evaluation'), t,
                                 sha(output / 'protocol.json'), stage == 'evaluation') for t in tasks)
            print(f'{stage}: {done}/{len(tasks)} complete')


if __name__ == '__main__':
    main()
