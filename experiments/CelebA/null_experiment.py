#!/usr/bin/env python3
"""Expanded CelebA learned same-source nulls at the accepted JS configurations."""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.CelebA.model_selection import (
    FIT_FILES, EVAL_FILES, completed, read, read_sealed, seal, sha, task_identity, write,
)

FAMILIES = (('feature', 'lower'), ('feature', 'upper'),
            ('pixel', 'lower'), ('pixel', 'upper'), ('pixel', 'real'))
DEVELOPMENT = ('train', 'earlystop', 'selection_calibration', 'selection_evaluation')
FINAL = ('test_calibration', 'test_evaluation')


def tasks_for(config):
    tasks = []
    for level, source in FAMILIES:
        for repeat in range(config['repeats']):
            alpha = .5 if level == 'feature' else 2.
            tasks.append(dict(task_index=len(tasks), representation=level, source=source,
                              branch='null_' + source, repeat=repeat, architecture='baseline',
                              candidate='baseline_js_a0p5' if level == 'feature' else 'baseline_js_a2',
                              loss='js', output_alpha=alpha))
    return tasks


def directory(output, task, stage):
    return output / stage / task['representation'] / task['source'] / f"repeat_{task['repeat']:02d}"


def null_metrics(p, q):
    p, q = np.asarray(p, dtype=float), np.asarray(q, dtype=float)
    for scores in (p, q):
        if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 2)):
            raise ValueError('Null scores must be finite nonempty vectors in [0,2]')
    mse = float(.5 * np.mean((p-1)**2) + .5 * np.mean((q-1)**2))
    result = dict(rdr_mean_p=float(p.mean()), rdr_mean_q=float(q.mean()),
                  rdr_bias=float(.5*(p.mean()+q.mean())-1), rdr_mse_one=mse,
                  rdr_rmse_one=float(np.sqrt(mse)),
                  rdr_mae_one=float(.5*np.abs(p-1).mean()+.5*np.abs(q-1).mean()),
                  excess_brier_true=mse/4)
    for tolerance in (.05, .1, .2):
        result[f'fraction_within_{tolerance:g}'] = float(.5*np.mean(abs(p-1) <= tolerance)+.5*np.mean(abs(q-1) <= tolerance))
    return result


def verify_parent(parent):
    completion = read_sealed(parent / 'COMPLETE.json')
    protocol = read_sealed(parent / 'protocol.json')
    if completion['protocol_sha256'] != sha(parent / 'protocol.json') or completion['evaluations'] != 20:
        raise ValueError('Expected completed selected-model parent assessment')
    data_record = read_sealed(parent / 'evaluation_data.json')
    if (sha(parent / 'evaluation_data.json') != completion['evaluation_data_sha256']
            or sha(parent / 'data/evaluation_metadata.json') != data_record['metadata_sha256']
            or sha(parent / 'freeze.json') != data_record['freeze_sha256']):
        raise ValueError('Parent final-data provenance differs from its completion receipt')
    for name, digest in completion['files'].items():
        if sha(parent / name) != digest:
            raise ValueError(f'Parent assessment changed: {name}')
    for name, digest in protocol['files'].items():
        if sha(parent / name) != digest:
            raise ValueError(f'Frozen parent source changed: {name}')
    for task in protocol['tasks']:
        if (task['loss'] != 'js' or task['architecture'] != 'baseline'
                or task['output_alpha'] != (.5 if task['representation'] == 'feature' else 2.)):
            raise ValueError('Parent accepted settings differ')
    return protocol


def prepare(output, parent):
    from experiments.CelebA.null_data import prepare_null_data
    parent_protocol = verify_parent(parent)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    (output / 'logs').mkdir()
    shutil.copytree(parent / 'source', output / 'source')
    for name in ('null_experiment.py', 'null_data.py', 'null_report.py', 'null.slurm'):
        shutil.copy2(Path(__file__).with_name(name), output / 'source/experiments/CelebA' / name)
    legacy = output / 'source/experiments/JRSSB/CELEBA_rdr.py'
    legacy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'experiments/JRSSB/CELEBA_rdr.py', legacy)
    for name in ('experiments/CelebA/selection_training.py', 'experiments/CelebA/selection_data.py',
                 'utils/networks.py', 'utils/losses.py', 'utils/model_selection.py', 'utils/calibration.py'):
        if sha(ROOT / name) != sha(parent / 'source' / name):
            raise ValueError(f'Accepted training/data implementation differs: {name}')
    metadata = prepare_null_data(output, parent, seed=2026100303)
    cfg = dict(parent_protocol['config'])
    cfg.update(seed=2026100304, repeats=5)
    files = {str(path.relative_to(output)): sha(path) for path in (output/'source').rglob('*') if path.is_file()}
    files.update({str(path.relative_to(output)):sha(path) for path in (output/'data').rglob('*') if path.is_file()})
    protocol = dict(schema=1, study='expanded_selected_configuration_learned_same_source_nulls',
                    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    parent=str(parent), parent_completion_sha256=sha(parent/'COMPLETE.json'),
                    accepted_settings_freeze_sha256=sha(parent/'freeze.json'),
                    config=cfg, tasks=tasks_for(cfg), files=files,
                    data_metadata_sha256=sha(output/'data/null_data.json'),
                    data_split_seed=2026100303, training_seed=2026100304,
                    target='For each same-source A/B pair: r=1, population optimal balanced Brier=0.25.',
                    checkpoint_rule='Minimum stopping-set Brier; fresh weights and training-only scaler.',
                    repeat_scope='Five initializations/training orders on the same frozen source halves; conditional training variation.',
                    evaluation_scope='Separate final calibration and evaluation halves. No final-based tuning, no best-repeat selection.',
                    null_policy='Learned scores only. Exact r=1 is a reference, never a substitute or calibration-pass gate.',
                    historical_hellinger='Auxiliary signed variational midpoint diagnostic at epsilon1e-6; record clipping, no zero truncation.',
                    count_note='Splitting the full expanded pools yields60k generated images per training arm, not120k per arm.',
                    selection_performed=False, real_feature_null_included=False)
    seal(output/'protocol.json', protocol)
    report(output)
    print(f'Prepared {len(protocol["tasks"])} fresh null fits in {output}', flush=True)


def locked(output):
    protocol = read_sealed(output/'protocol.json')
    parent = Path(protocol['parent'])
    if (sha(parent/'COMPLETE.json') != protocol['parent_completion_sha256']
            or sha(parent/'freeze.json') != protocol['accepted_settings_freeze_sha256']):
        raise ValueError('Parent accepted assessment lineage changed')
    for name, digest in protocol['files'].items():
        if sha(output/name) != digest:
            raise ValueError(f'Frozen null source/data changed: {name}')
        if name.startswith('source/') and name.endswith('.py'):
            source = ROOT / Path(name).relative_to('source')
            # Parent includes old presentation source: only check code used by this workflow.
            used = name.startswith('source/utils/') or name.endswith(tuple(
                '/'+n for n in ('null_experiment.py','null_data.py','null_report.py','selection_training.py',
                               'selection_data.py','expanded_selection_data.py','model_selection.py','CELEBA_rdr.py')))
            if used and (not source.exists() or sha(source) != digest):
                raise ValueError(f'Use the frozen null entrypoint: {name}')
    return protocol


def fit(output, index, device):
    from experiments.CelebA.null_data import load_null_arrays
    from experiments.CelebA.selection_training import fit_model, evaluate_model
    from experiments.JRSSB.CELEBA_rdr import estimate_midpoint_hellinger
    protocol = locked(output)
    task = protocol['tasks'][index]
    digest = sha(output/'protocol.json')
    training = directory(output, task, 'fits')
    assessment = directory(output, task, 'evaluation')
    if not completed(training, task, digest):
        training.mkdir(parents=True, exist_ok=False)
        try:
            arrays = load_null_arrays(output, task['representation'], task['source'], DEVELOPMENT)
            metrics = fit_model(task, protocol['config'], arrays, training, device)
            write(training/'metrics.json', {**metrics, **task})
            write(training/'COMPLETE.json', dict(task_identity=task_identity(task), protocol_sha256=digest,
                  files={name:sha(training/name) for name in FIT_FILES}))
        except Exception as error:
            write(training/'FAILED.json', dict(type=type(error).__name__, message=str(error)))
            raise
    if completed(assessment, task, digest, evaluation=True):
        if read(assessment/'COMPLETE.json')['training_receipt_sha256'] != sha(training/'COMPLETE.json'):
            raise ValueError('Final null scores no longer match training receipt')
        return
    assessment.mkdir(parents=True, exist_ok=False)
    try:
        arrays = load_null_arrays(output, task['representation'], task['source'], FINAL)
        metrics = evaluate_model(task, protocol['config'], arrays, training, assessment, device)
        with np.load(assessment/'predictions.npz', allow_pickle=False) as predictions:
            p, q = predictions['test_evaluation_p'], predictions['test_evaluation_q']
            diagnostic = estimate_midpoint_hellinger(p, q, 1e-6)
            metrics.update(null_metrics(p,q))
        metrics.update(empirical_brier_excess=metrics['brier']-.25,
                       h2_variational=diagnostic['variational_lower_bound'],
                       h2_plugin=diagnostic['plugin_midpoint_hellinger'],
                       h2_clip_rate_p=diagnostic['p_clip_rate'], h2_clip_rate_q=diagnostic['q_clip_rate'],
                       h2_epsilon=1e-6, null_truth_rdr=1., assessment_scope='expanded_same_source_null_final_evaluation')
        write(assessment/'metrics.json', {**metrics, **task})
        write(assessment/'COMPLETE.json', dict(task_identity=task_identity(task), protocol_sha256=digest,
              training_receipt_sha256=sha(training/'COMPLETE.json'),
              files={name:sha(assessment/name) for name in EVAL_FILES}))
    except Exception as error:
        write(assessment/'FAILED.json', dict(type=type(error).__name__, message=str(error)))
        raise


def report(output, require_complete=False):
    from experiments.CelebA.null_report import build_report
    protocol = locked(output)
    digest = sha(output/'protocol.json')
    rows, failures = [], []
    for task in protocol['tasks']:
        training = directory(output, task, 'fits')
        assessment = directory(output, task, 'evaluation')
        if completed(assessment, task, digest, evaluation=True):
            if (not completed(training, task, digest)
                    or read(assessment/'COMPLETE.json')['training_receipt_sha256'] != sha(training/'COMPLETE.json')):
                raise ValueError('Final null scores no longer match training receipt')
            row = read(assessment/'metrics.json')
            if task_identity(row) != task_identity(task):
                raise ValueError('Null metric identity differs')
            rows.append(row)
        else:
            for stage, path in (('fit',training), ('evaluation',assessment)):
                if (path/'FAILED.json').exists() and not (path/'COMPLETE.json').exists():
                    failures.append({**task, 'stage':stage, **read(path/'FAILED.json')})
    status = build_report(output, protocol, rows, failures)
    if require_complete and not status['complete']:
        raise ValueError('Null study is incomplete')
    if status['complete']:
        artifacts = ['RESULTS.md','report_status.json','per_fit.csv','per_comparison.csv']
        artifacts += [p.name for p in output.glob('null_*') if p.suffix in ('.png', '.pdf')]
        seal(output/'COMPLETE.json', dict(protocol_sha256=digest, fits=len(rows), evaluations=len(rows),
                                        files={name:sha(output/name) for name in artifacts}))
    print(status, flush=True)


def submit(output):
    protocol = locked(output)
    if (output/'jobs.json').exists():
        raise ValueError('Null study already submitted')
    wrapper = output/'source/experiments/CelebA/null.slurm'
    jobs, arrays = [], []
    for level in ('feature','pixel'):
        indices = ','.join(str(i) for i,t in enumerate(protocol['tasks']) if t['representation']==level)
        command = ['sbatch','--parsable',f'--array={indices}%4',f'--mem={"24G" if level=="feature" else "48G"}',
                   f'--time={"03:00:00" if level=="feature" else "08:00:00"}',
                   f'--output={output}/logs/fit-%A_%a.out',f'--error={output}/logs/fit-%A_%a.err',
                   str(wrapper),str(output),'fit']
        job = subprocess.check_output(command,text=True).strip().split(';')[0]
        if not job.isdigit(): raise ValueError('Invalid job ID')
        arrays.append(job); jobs.append(dict(stage=level,job_id=job,command=command))
        write(output/'jobs.json',jobs)
    command = ['sbatch','--parsable','--gres=gpu:0','--mem=8G','--time=01:00:00',
               '--dependency=afterok:'+':'.join(arrays),f'--output={output}/logs/report-%j.out',
               f'--error={output}/logs/report-%j.err',str(wrapper),str(output),'report']
    job = subprocess.check_output(command,text=True).strip().split(';')[0]
    if not job.isdigit(): raise ValueError('Invalid job ID')
    jobs.append(dict(stage='report',job_id=job,command=command)); write(output/'jobs.json',jobs)
    print(jobs,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','fit','report','submit'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--task-index',type=int,default=0)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args(); output=args.output.resolve()
    if args.action=='prepare':
        if args.parent is None: parser.error('prepare requires --parent')
        prepare(output,args.parent.resolve())
    elif args.action=='fit': fit(output,args.task_index,args.device)
    elif args.action=='report': report(output,args.require_complete)
    else: submit(output)


if __name__=='__main__': main()
