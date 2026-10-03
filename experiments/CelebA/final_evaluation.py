#!/usr/bin/env python3
"""Evaluate user-accepted expanded CelebA settings without refitting or selection."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/celeba-final-mpl')

from experiments.CelebA.model_selection import (
    EVAL_FILES, completed, load_protocol, read, read_sealed, seal, sha,
    task_directory, task_identity, write,
)
from experiments.CelebA.selection_data import (
    TEST_ROLES, load_arrays, prepare_evaluation_data, verify_receipt,
)


SETTINGS = {'feature': ('js', .5), 'pixel': ('js', 2.)}


def selected_tasks(parent_protocol):
    tasks = [dict(t) for t in parent_protocol['tasks']
             if (t['loss'], t['output_alpha']) == SETTINGS[t['representation']]
             and t['architecture'] == 'baseline']
    expected = {(level, branch, repeat) for level in SETTINGS
                for branch in ('lower', 'upper')
                for repeat in range(parent_protocol['config']['repeats'])}
    actual = [(t['representation'], t['branch'], t['repeat']) for t in tasks]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Selected models do not cover the four paired comparisons')
    return tasks


def prepare(output, parent):
    protocol = load_protocol(parent)
    tasks = selected_tasks(protocol)
    parent_hash = sha(parent / 'protocol.json')
    checkpoints = {}
    for task in tasks:
        directory = task_directory(parent, task)
        if not completed(directory, task, parent_hash):
            raise ValueError(f'Selected checkpoint incomplete: {directory}')
        checkpoints[str(directory)] = sha(directory / 'COMPLETE.json')
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    # Preserve the exact trainer/data implementation used for selection.
    shutil.copytree(parent / 'source', output / 'source')
    for name in ('final_evaluation.py', 'final_evaluation.slurm'):
        shutil.copy2(Path(__file__).with_name(name), output / 'source/experiments/CelebA' / name)
    metadata = read(parent / 'data/metadata.json')
    # Development arrays are referenced read-only and never loaded here.
    for roles in metadata['arrays'].values():
        for sources in roles.values():
            for receipt in sources.values():
                receipt['path'] = str(parent / receipt['path'])
    for sources in metadata['roles'].values():
        for info in sources.values():
            source = verify_receipt(info['receipt'], full=True, base=parent)
            target = output / info['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            verify_receipt(info['receipt'], full=True, base=output)
    write(output / 'data/metadata.json', metadata)
    files = {str(p.relative_to(output)): sha(p)
             for p in (output / 'source').rglob('*') if p.is_file()}
    files['data/metadata.json'] = sha(output / 'data/metadata.json')
    record = dict(schema=1, scope='selected_models_final_evaluation_only',
                  parent=str(parent), parent_protocol_sha256=parent_hash,
                  config=protocol['config'], tasks=tasks, files=files,
                  checkpoint_receipts=checkpoints,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  approval='User accepted JS slope 0.5 feature and JS slope 2 pixel; only evaluate saved models on final splits.',
                  selection_limitation='Parent grid has 319/320 fits; pixel/lower Hellinger slope 4 repeat 03 failed. User accepted the recommended settings; parent selection remains incomplete.',
                  repeat_rule='Evaluate every saved repeat; no best-repeat selection, ensembling, retraining, or test-based tuning.',
                  scope_caveat='Retrospective historically inspected pool; this is the main final assessment, not fresh confirmatory evidence.')
    seal(output / 'protocol.json', record)
    seal(output / 'freeze.json', dict(protocol_sha256=sha(output / 'protocol.json'),
                                     settings=SETTINGS, checkpoint_receipts=checkpoints,
                                     authorization=record['approval']))
    report(output)
    print(json.dumps({'prepared': str(output), 'evaluations': len(tasks)}))


def locked(output):
    protocol = read_sealed(output / 'protocol.json')
    frozen = read_sealed(output / 'freeze.json')
    if frozen['protocol_sha256'] != sha(output / 'protocol.json'):
        raise ValueError('Final evaluation freeze changed')
    if sha(Path(protocol['parent']) / 'protocol.json') != protocol['parent_protocol_sha256']:
        raise ValueError('Parent selection protocol changed')
    for name, digest in protocol['files'].items():
        if sha(output / name) != digest:
            raise ValueError(f'Frozen final-evaluation input changed: {name}')
        if name.startswith('source/') and name.endswith('.py'):
            if sha(ROOT / Path(name).relative_to('source')) != digest:
                raise ValueError(f'Use frozen final-evaluation source: {name}')
    return protocol


def materialize(output):
    locked(output)
    metadata = prepare_evaluation_data(output)
    count = 0
    for level in SETTINGS:
        for role in TEST_ROLES:
            for receipt in metadata['arrays'][level][role].values():
                verify_receipt(receipt, full=True, base=output)
                count += 1
    if count != 12:
        raise ValueError('Expected twelve final arrays')
    seal(output / 'evaluation_data.json', dict(metadata_sha256=sha(output / 'data/evaluation_metadata.json'),
                                             freeze_sha256=sha(output / 'freeze.json'), verified_arrays=count))
    print(json.dumps({'final_arrays_verified': count}))


def evaluate(output, index, device):
    from experiments.CelebA.selection_training import evaluate_model
    protocol = locked(output)
    task = protocol['tasks'][index]
    digest = sha(output / 'protocol.json')
    directory = task_directory(output, task, evaluation=True)
    if completed(directory, task, digest, evaluation=True):
        return
    data = read_sealed(output / 'evaluation_data.json')
    if (data['metadata_sha256'] != sha(output / 'data/evaluation_metadata.json')
            or data['freeze_sha256'] != sha(output / 'freeze.json')):
        raise ValueError('Final data lineage changed')
    checkpoint = task_directory(Path(protocol['parent']), task)
    if sha(checkpoint / 'COMPLETE.json') != protocol['checkpoint_receipts'][str(checkpoint)]:
        raise ValueError('Selected checkpoint changed')
    if not completed(checkpoint, task, protocol['parent_protocol_sha256']):
        raise ValueError('Invalid selected checkpoint')
    directory.mkdir(parents=True, exist_ok=False)
    try:
        nested = load_arrays(output, task['representation'], task['branch'], TEST_ROLES)
        arrays = {f'{role}_{side}': a for role, sides in nested.items() for side, a in sides.items()}
        metrics = evaluate_model(task, protocol['config'], arrays, checkpoint, directory, device)
        write(directory / 'metrics.json', {**metrics, **task})
        write(directory / 'COMPLETE.json', dict(task_identity=task_identity(task), protocol_sha256=digest,
              files={name: sha(directory / name) for name in EVAL_FILES}))
    except Exception as error:
        write(directory / 'FAILED.json', dict(type=type(error).__name__, message=str(error)))
        raise


def report(output, require_complete=False):
    from experiments.CelebA.selection_report import METRICS, _aggregate, _csv, _split_accounting, _table, _validate
    protocol = locked(output)
    digest = sha(output / 'protocol.json')
    rows = []
    for task in protocol['tasks']:
        directory = task_directory(output, task, evaluation=True)
        if completed(directory, task, digest, evaluation=True):
            row = read(directory / 'metrics.json')
            if task_identity(row) != task_identity(task):
                raise ValueError('Final metric/task identity mismatch')
            rows.append(row)
    _validate(rows)
    summaries = _aggregate(rows, protocol['config']['repeats'], 'final_evaluation')
    _csv(output / 'per_fit.csv', rows, ('representation', 'branch', 'candidate', 'repeat') + METRICS)
    _csv(output / 'per_comparison.csv', summaries, ('representation', 'branch', 'candidate', 'repeats'))
    complete = len(rows) == len(protocol['tasks'])
    status = dict(complete=complete, evaluations=len(rows), expected_evaluations=len(protocol['tasks']),
                  training_performed=False, selection_performed=False, retrospective=True)
    lines = ['# CelebA expanded-data final assessment', '',
             f"**Status: {'complete' if complete else 'incomplete'}; {len(rows)}/{len(protocol['tasks'])} verified evaluations.**", '',
             'Selected saved JS models: feature slope 0.5, pixel slope 2, each fitted separately for P versus Q_l and P versus Q_u. '
             'Output r = 2 sigmoid(alpha z). All five repetitions are assessed; no best repeat is chosen.', '',
             '## Final results', '',
             'Means (SD) across five fitted repetitions on the same reserved final rows. '
             'Balanced Brier uses only the final evaluation split, with equal P/Q weight. '
             'Final calibration rows estimate score-cell RDRs; final evaluation rows supply neural means and cell masses. '
             'Local Gap is the supported-mass-weighted absolute cell gap. Incomplete groups have no five-repeat mean.', '',
             *_table(summaries, ('brier', 'local_gap', 'supported_mass')), '',
             '## Cell diagnostics', '',
             *_table(summaries, ('c2_distance', 'c2_width', 'middle_mass', 'middle_local_gap')), '',
             'Per-fit cells.json files contain C.1/C.2 intervals for all 20 fixed score bins. '
             'These are nominal image-level cell-average intervals, not pointwise RDR guarantees, '
             'and do not account for within-identity dependence or neural-mean estimation uncertainty.', '',
             '## Split accounting', '', *_split_accounting(output, protocol), '',
             'Training and development rows are listed for provenance; only the two final roles are evaluated in this run.', '',
             '## Provenance and interpretation', '', protocol['scope_caveat'], '',
             protocol['selection_limitation'], '',
             'Repeat SD measures training variability conditional on fixed data. It is not an independent-data confidence interval. '
             'No refitting, architecture search, test-based selection, global divergence rerun, or image-panel analysis was performed.', '',
             '[Per-fit results](per_fit.csv) · [Per-comparison results](per_comparison.csv) · [Frozen protocol](protocol.json)', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    write(output / 'report_status.json', status)
    if require_complete and not complete:
        raise ValueError('Final assessment is incomplete')
    if complete:
        seal(output / 'COMPLETE.json', dict(protocol_sha256=digest, evaluations=len(rows),
             evaluation_data_sha256=sha(output / 'evaluation_data.json'),
             files={name: sha(output / name) for name in ('RESULTS.md', 'per_fit.csv', 'per_comparison.csv', 'report_status.json')}))
    print(json.dumps(status))


def submit(output):
    protocol = locked(output)
    if (output / 'jobs.json').exists():
        raise ValueError('Final evaluation already submitted')
    wrapper = output / 'source/experiments/CelebA/final_evaluation.slurm'
    jobs = []
    def job(action, extra):
        command = ['sbatch', '--parsable', f'--output={output}/logs/{action}-%A_%a.out',
                   f'--error={output}/logs/{action}-%A_%a.err', *extra, str(wrapper), str(output), action]
        value = subprocess.check_output(command, text=True).strip().split(';')[0]
        if not value.isdigit():
            raise ValueError('Invalid Slurm job ID')
        jobs.append(dict(action=action, job_id=value, command=command))
        write(output / 'jobs.json', jobs)
        return value
    preparation = job('materialize', ['--gres=gpu:0', '--mem=48G', '--time=03:00:00'])
    assessment = job('evaluate', [f'--array=0-{len(protocol["tasks"])-1}%4', '--gres=gpu:1',
                                 '--mem=24G', '--time=02:00:00', f'--dependency=afterok:{preparation}'])
    job('report', ['--gres=gpu:0', '--mem=8G', '--time=01:00:00', f'--dependency=afterok:{assessment}'])
    print(json.dumps(jobs, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'materialize', 'evaluate', 'report', 'submit'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--task-index', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        if args.parent is None:
            parser.error('prepare requires --parent')
        prepare(output, args.parent.resolve())
    elif args.action == 'materialize':
        materialize(output)
    elif args.action == 'evaluate':
        evaluate(output, args.task_index, args.device)
    elif args.action == 'report':
        report(output, args.require_complete)
    else:
        submit(output)


if __name__ == '__main__':
    main()
