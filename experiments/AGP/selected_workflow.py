#!/usr/bin/env python3
"""Audited AGP loss/slope selection and selected fixed-width assessments.

Every production stage consumes a frozen source snapshot and immutable data
cache. Selection workers read development roles only; the completion-gated
selection lock is required before a final or null worker can read its inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.linalg import helmert
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[2]
AGP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(AGP_DIR))
from utils.losses import midpoint_loss_from_logits
from utils.model_selection import balanced_brier, calibration_diagnostics, choose_configuration

# Freeze only the dependencies of the selected RDR study. Unrelated AGP work
# (including concurrent pooling experiments) must never enter its source snapshot.
RDR_SOURCE_FILES = (
    "__init__.py", "selected_workflow.py", "selection_data.py", "AGP_712_data.py",
    "AGP_ci_data.py", "selection_report.py", "verify_selected.py",
)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_config(cfg):
    if set(cfg['losses']) != {'hellinger', 'kl', 'chisq', 'js'} or len(cfg['losses']) != 4:
        raise ValueError('Exactly the four declared midpoint losses are required')
    if len(set(cfg['output_alphas'])) != len(cfg['output_alphas']) or any(
            not math.isfinite(float(a)) or a <= 0 for a in cfg['output_alphas']):
        raise ValueError('Output slopes must be unique, finite and positive')
    if not cfg['output_alphas'] or cfg['repeats'] < 2:
        raise ValueError('An empty grid or unpaired single-repeat study is not permitted')
    if cfg['representation'] != 'ilr' or cfg['architecture'] != 'deep':
        raise ValueError('Representation and architecture must remain fixed')
    if cfg['bins'] != 20 or cfg['representative_repeat'] != 0:
        raise ValueError('This protocol fixes 20 bins and representative repeat zero')
    if not 1 <= cfg['min_epochs'] <= cfg['max_epochs'] or cfg['patience'] <= 0:
        raise ValueError('Invalid stopping configuration')
    if not 0 < cfg['ci_alpha'] < 1 or cfg['epsilon'] <= 0:
        raise ValueError('Invalid CI level or compositional smoothing')
    if cfg['hidden_dimension'] < 1 or cfg['blocks'] < 1:
        raise ValueError('Empty neural architecture')
    if cfg.get('selection_allocation', [.7,.1,.1,.1]) != [.7,.1,.1,.1]:
        raise ValueError('Selection allocation is fixed at 70/10/10/10')
    return cfg


def selection_tasks(cfg):
    return [dict(candidate=f'{loss}_a{alpha:g}', loss=loss, output_alpha=float(alpha), repeat=repeat)
            for loss in cfg['losses'] for alpha in cfg['output_alphas']
            for repeat in range(cfg['repeats'])]


def seed_for(cfg, repeat, stage):
    return int(cfg['seed']) + {'selection': 0, 'final': 100000, 'null': 200000}[stage] + 1009 * int(repeat)


class DeepLogitMLP(nn.Module):
    def __init__(self, dimension, width, blocks):
        super().__init__()
        layers = [nn.Linear(dimension, width), nn.ReLU()]
        for _ in range(blocks):
            layers.extend([nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU()])
        layers.append(nn.Linear(width, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).flatten()


def make_model(dimension, cfg, repeat, stage):
    # Loss and output slope never enter initialization. Raw input dimension
    # decreases by one under the full orthonormal ILR transform.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed_for(cfg, repeat, stage))
        return DeepLogitMLP(dimension - 1, cfg['hidden_dimension'], cfg['blocks'])


def _clr(raw, epsilon):
    x = np.asarray(raw, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all() or (x < 0).any() or (x.sum(1) <= 0).any():
        raise ValueError('Invalid abundance compositions')
    x = x / x.sum(1, keepdims=True)
    logs = np.log((x + epsilon) / (1 + x.shape[1] * epsilon))
    return logs - logs.mean(1, keepdims=True)


def fit_ilr(p, q, epsilon):
    p, q = _clr(p, epsilon), _clr(q, epsilon)
    center = .5 * (p.mean(0) + q.mean(0))
    scale = float(np.sqrt(.5 * (np.square(p-center).mean() + np.square(q-center).mean())))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Degenerate training-only ILR scale')
    return {'center': center, 'scale': scale, 'basis': helmert(p.shape[1], full=False), 'epsilon': epsilon}


def transform_ilr(raw, state):
    x = ((_clr(raw, float(state['epsilon'])) - state['center']) / float(state['scale'])) @ state['basis'].T
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite ILR transformation')
    return x.astype(np.float32)


def stable_risk(logits, n_p, loss):
    """Exact risk with scale-factored Hellinger derivatives, without clipping."""
    t = logits.double()
    if loss != 'hellinger':
        risk = midpoint_loss_from_logits(t[:n_p], t[n_p:], loss)
        return risk, risk, 0.
    log_ratio = math.log(2) + torch.nn.functional.logsigmoid(t)
    lp, lq = log_ratio[:n_p], log_ratio[n_p:]
    shift = max(0., float((-.5 * lp.detach()).max()))
    backward = (.5 * torch.exp(-.5*lp-shift).mean() + .25 * torch.exp(.5*lp-shift).mean()
                + .25 * torch.exp(.5*lq-shift).mean())
    risk = backward.detach() * torch.exp(t.new_tensor(shift)) - 1
    return risk, backward, shift


def clip_scaled_gradients(parameters, shift, max_norm):
    gradients = [p.grad for p in parameters if p.grad is not None]
    norm = float(torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.double()) for g in gradients])))
    if not math.isfinite(norm):
        raise FloatingPointError('Nonfinite gradient norm')
    if norm:
        factor = math.exp(min(shift, math.log(max_norm)-math.log(norm)))
        for g in gradients:
            g.copy_((g.double()*factor).to(g.dtype))
    return norm


@torch.no_grad()
def _tensor_scores(model, x, alpha):
    model.eval()
    return (2 * torch.sigmoid(alpha * model(x).double())).cpu().numpy()


@torch.no_grad()
def predict(model, raw, state, alpha, device):
    model.eval()
    parts = []
    for start in range(0, len(raw), 4096):
        x = torch.as_tensor(transform_ilr(raw[start:start+4096], state), device=device)
        parts.append(_tensor_scores(model, x, alpha))
    return np.concatenate(parts)


def fit_model(raw_arrays, cfg, task, stage, device):
    """Train full-batch and restore the absolute minimum stopping Brier."""
    state = fit_ilr(raw_arrays['train_p'], raw_arrays['train_q'], cfg['epsilon'])
    arrays = {k: torch.as_tensor(transform_ilr(raw_arrays[k], state), device=device)
              for k in ('train_p', 'train_q', 'stopping_p', 'stopping_q')}
    model = make_model(raw_arrays['train_p'].shape[1], cfg, task['repeat'], stage).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=.5, patience=cfg['plateau_patience'], cooldown=cfg['plateau_cooldown'])
    train = torch.cat([arrays['train_p'], arrays['train_q']])
    best, best_epoch, best_state, stale = float('inf'), None, None, 0
    history = []
    started = time.monotonic()
    alpha = task['output_alpha']
    for epoch in range(1, cfg['max_epochs']+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk, backward, shift = stable_risk(alpha*model(train), len(arrays['train_p']), task['loss'])
        if not bool(torch.isfinite(risk)):
            raise FloatingPointError(f'Nonfinite {task["loss"]} risk at epoch {epoch}')
        backward.backward()
        norm = clip_scaled_gradients(model.parameters(), shift, cfg['clip_max_norm'])
        optimizer.step()
        stopping = balanced_brier(*[_tensor_scores(model, arrays['stopping_'+s], alpha) for s in ('p','q')])
        if stopping < best:
            best, best_epoch, stale = stopping, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        history.append({'epoch': epoch, 'training_objective': float(risk), 'stopping_brier': stopping,
                        'learning_rate': optimizer.param_groups[0]['lr'], 'scaled_gradient_norm': norm,
                        'gradient_log_scale': shift, 'epochs_without_improvement': stale})
        scheduler.step(stopping)
        if epoch == 1 or epoch % 100 == 0:
            print(f'{stage} {task["candidate"]} repeat={task["repeat"]} epoch={epoch} Brier={stopping:.8f}', flush=True)
        if epoch >= cfg['min_epochs'] and stale >= cfg['patience']:
            break
    model.load_state_dict(best_state)
    restored = balanced_brier(*[_tensor_scores(model, arrays['stopping_'+s], alpha) for s in ('p','q')])
    if not math.isclose(restored, best, rel_tol=0, abs_tol=1e-12):
        raise AssertionError('Brier checkpoint restoration failed')
    metadata = dict(best_epoch=best_epoch, epochs_run=len(history), stopping_brier=best,
                    restored_stopping_brier=restored, parameter_count=sum(p.numel() for p in model.parameters()),
                    fit_seconds=time.monotonic()-started, seed_initialization=seed_for(cfg,task['repeat'],stage),
                    stopping_reason='patience' if stale >= cfg['patience'] else 'max_epochs',
                    device=str(device), model_dtype='float32', loss_dtype='float64',
                    numerical_gradient_mode='scale_factored_hellinger_float64_norm',
                    train_p=len(arrays['train_p']), train_q=len(arrays['train_q']))
    return model, state, pd.DataFrame(history), metadata


def load_arrays(output, study, roles):
    protocol = read_json(output/'data/protocol.json')['studies'][study]
    arrays = {}
    for role in roles:
        for side in ('p','q'):
            key = role+'_'+side
            spec = protocol['arrays'][key]
            path = output/spec['path']
            if sha256(path) != spec['sha256']:
                raise ValueError(f'Changed data cache: {path}')
            arrays[key] = np.load(path, allow_pickle=False)
    return arrays


def verify_snapshot(output):
    spec = read_json(output/'protocol.json')
    if sha256(output/'config.json') != spec['config_sha256']:
        raise ValueError('Configuration changed after preparation')
    if sha256(output/'data/protocol.json') != spec['data_protocol_sha256']:
        raise ValueError('Data protocol changed after preparation')
    for relative, digest in spec['source_sha256'].items():
        if sha256(output/'source'/relative) != digest:
            raise ValueError(f'Frozen source changed: {relative}')
    own_relative = 'experiments/AGP/selected_workflow.py'
    if own_relative not in spec['source_sha256'] or sha256(__file__) != spec['source_sha256'][own_relative]:
        raise ValueError('Executing workflow differs from frozen source')
    for name in ('utils.losses','utils.model_selection','utils.calibration'):
        relative=name.replace('.','/')+'.py'
        if sha256(sys.modules[name].__file__) != spec['source_sha256'].get(relative):
            raise ValueError(f'Executing utility differs from frozen source: {name}')
    return spec


def complete_task(directory, metadata):
    files = {str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*'))
             if p.is_file() and p.name != 'COMPLETE.json'}
    write_json(directory/'COMPLETE.json', {'status':'complete', 'finished_utc':now(), **metadata, 'outputs':files})


def verify_task(directory):
    marker = read_json(directory/'COMPLETE.json')
    if marker.get('status') != 'complete':
        raise ValueError(f'Incomplete fit: {directory}')
    required = {'model.pt','transform.npz','history.csv','metrics.json','cells.csv','predictions.npz','scores.csv'}
    if not required.issubset(marker.get('outputs',{})):
        raise ValueError(f'Incomplete artifact manifest: {directory}')
    if not {'task','stage','config_sha256','protocol_sha256'}.issubset(marker):
        raise ValueError(f'Missing task provenance: {directory}')
    for relative, digest in marker['outputs'].items():
        if sha256(directory/relative) != digest:
            raise ValueError(f'Changed completed artifact: {directory/relative}')
    return marker


def checked_selection(output):
    lock = read_json(output/'selection/freeze.json')
    for relative, digest in lock['outputs'].items():
        if sha256(output/relative) != digest:
            raise ValueError(f'Selection lock changed: {relative}')
    if not lock.get('selection_completions') or lock['completed_fits'] != lock['expected_fits']:
        raise ValueError('Incomplete frozen selection grid')
    for relative,digest in lock['selection_completions'].items():
        if sha256(output/relative) != digest:
            raise ValueError(f'Selection completion changed: {relative}')
    if len(lock['selection_completions']) != lock['expected_fits']:
        raise ValueError('Selection completion count mismatch')
    return read_json(output/'selection/selection.json')['chosen']


def verify_expected_task(output,directory,stage,task):
    marker=verify_task(directory)
    if (marker['task'] != task or marker['stage'] != stage
            or marker['config_sha256'] != sha256(output/'config.json')
            or marker['protocol_sha256'] != sha256(output/'protocol.json')):
        raise ValueError('Completed task identity or provenance mismatch')
    if stage != 'selection' and marker.get('selection_freeze_sha256') != sha256(output/'selection/freeze.json'):
        raise ValueError('Assessment does not match frozen selection')
    return marker


def add_agp_middle(summary, cells):
    middle = [c for c in cells if c['left'] >= .2-1e-12 and c['right'] <= 1.8+1e-12]
    mass = sum(c['eval_mass'] for c in middle)
    supported = [c for c in middle if c['absolute_gap'] is not None]
    support = sum(c['eval_mass'] for c in supported)
    summary['agp_middle_mass'] = float(mass)
    summary['agp_middle_supported_mass'] = float(support)
    summary['agp_middle_local_gap'] = float(sum(c['eval_mass']*c['absolute_gap'] for c in supported)/support) if support else None
    summary['agp_middle_definition'] = '[0.2,1.8); historical AGP descriptive region'
    return summary


def run_task(output, stage, index, device='cpu'):
    output = Path(output)
    verify_snapshot(output)
    cfg = validate_config(read_json(output/'config.json'))
    if stage == 'selection':
        if not 0 <= index < len(selection_tasks(cfg)):
            raise ValueError('Selection task index outside declared grid')
        task = selection_tasks(cfg)[index]
        target = output/'selection/runs'/task['candidate']/f'{task["repeat"]:02d}'
    elif stage in ('final','null'):
        # This gate deliberately precedes ANY reads from the assessment cache.
        chosen = checked_selection(output)
        if not 0 <= index < cfg['repeats']:
            raise ValueError('Assessment repeat outside declared grid')
        task = {k:chosen[k] for k in ('candidate','loss','output_alpha')}
        task['repeat'] = index
        target = output/stage/f'{index:02d}'
    else:
        raise ValueError('Unknown fitting stage')
    if (target/'COMPLETE.json').exists():
        verify_expected_task(output,target,stage,task)
        return read_json(target/'metrics.json')
    if target.exists():
        raise ValueError(f'Incomplete existing task must be preserved and investigated: {target}')
    target.mkdir(parents=True)
    try:
        torch.set_num_threads(cfg['threads'])
        torch.use_deterministic_algorithms(True)
        if str(device).startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with threadpool_limits(limits=cfg['threads']):
            raw = load_arrays(output,stage,('train','stopping'))
            model,state,history,training = fit_model(raw,cfg,task,stage,device)
            # Persist the fitted checkpoint before accessing diagnostic inputs.
            torch.save({'model':model.cpu().state_dict(),'task':task,'training':training,'config':cfg},target/'model.pt')
            model.to(device)
            np.savez_compressed(target/'transform.npz',**state)
            history.to_csv(target/'history.csv',index=False)
            roles = ('calibration','evaluation') if stage == 'selection' else ('test',)
            diagnostic = load_arrays(output,stage,roles)
            predictions = {k:predict(model,v,state,task['output_alpha'],device) for k,v in diagnostic.items()}
            if stage == 'selection':
                args = [predictions[r+'_'+s] for r in ('calibration','evaluation') for s in ('p','q')]
                ci_alpha = cfg['ci_alpha']/(len(cfg['losses'])*len(cfg['output_alphas']))
                eval_role = 'evaluation'
            else:
                args = [predictions['test_'+s] for _ in range(2) for s in ('p','q')]
                ci_alpha,eval_role = cfg['ci_alpha'],'test'
            summary,cells = calibration_diagnostics(*args,alpha=ci_alpha,bins=cfg['bins'])
            summary = add_agp_middle(summary,cells)
            p,q = predictions[eval_role+'_p'],predictions[eval_role+'_q']
            metrics = {**task,**training,**summary,'stage':stage,'brier':balanced_brier(p,q),
                       'auc':float(roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(q))],np.r_[p,q])),
                       'assessment_role':eval_role,'mean_score_p':float(p.mean()),'mean_score_q':float(q.mean()),
                       'same_sample_calibration_evaluation':stage!='selection',
                       'ci_scope':'per-fit marginal C1 and simultaneous-over-20-bins C2' if stage!='selection' else 'candidate-adjusted selection diagnostics'}
            if stage == 'null':
                metrics.update(constant_brier=.25, true_rdr_mse=float(.5*np.mean((p-1)**2)+.5*np.mean((q-1)**2)),
                    c2_excludes_one_bins=sum(c['c2_lower']>1 or c['c2_upper']<1 for c in cells))
            write_json(target/'metrics.json',metrics)
            pd.DataFrame(cells).to_csv(target/'cells.csv',index=False)
            np.savez_compressed(target/'predictions.npz',**predictions)
            manifest = pd.read_csv(output/'data'/stage/'manifest.csv',dtype={'sample_id':str,'subject_id':str},keep_default_na=False)
            frames=[]
            for role in roles:
                for side in ('p','q'):
                    rows=manifest[manifest.role.eq(role)&manifest.distribution.eq(side.upper())].copy()
                    if 'array_row' in rows:
                        rows=rows.sort_values('array_row')
                    if len(rows)!=len(predictions[role+'_'+side]):
                        raise AssertionError('Score/manifest row count mismatch')
                    rows['rdr']=predictions[role+'_'+side]
                    frames.append(rows)
            pd.concat(frames,ignore_index=True).to_csv(target/'scores.csv',index=False)
        complete_task(target,{'task':task,'stage':stage,'config_sha256':sha256(output/'config.json'),
            'protocol_sha256':sha256(output/'protocol.json'),
            'selection_freeze_sha256':sha256(output/'selection/freeze.json') if stage != 'selection' else None})
        print(json.dumps({'status':'complete','stage':stage,'task':task,'brier':metrics['brier'],'gap':metrics['local_gap']}),flush=True)
        return metrics
    except Exception as exc:
        write_json(target/'FAILED.json',{'error':repr(exc),'utc':now(),'stage':stage,'task':task})
        raise


def freeze_selection(output):
    output=Path(output)
    verify_snapshot(output)
    cfg=read_json(output/'config.json')
    if (output/'selection/freeze.json').exists():
        checked_selection(output)
        return read_json(output/'selection/selection.json')
    rows=[]
    completions={}
    for task in selection_tasks(cfg):
        directory=output/'selection/runs'/task['candidate']/f'{task["repeat"]:02d}'
        marker=verify_expected_task(output,directory,'selection',task)
        if marker['task']!=task or marker['config_sha256']!=sha256(output/'config.json'):
            raise ValueError('Completed task does not match prescribed grid')
        row=read_json(directory/'metrics.json')
        if any(row[k]!=v for k,v in task.items()):
            raise ValueError('Metrics/task mismatch')
        rows.append(row)
        completions[str((directory/'COMPLETE.json').relative_to(output))]=sha256(directory/'COMPLETE.json')
    selection=choose_configuration(rows)
    pd.DataFrame(rows).to_csv(output/'selection/rows.csv',index=False)
    pd.DataFrame(selection['ranking']).to_csv(output/'selection/ranking.csv',index=False)
    write_json(output/'selection/selection.json',selection)
    files=['config.json','data/protocol.json','selection/rows.csv','selection/ranking.csv','selection/selection.json']
    write_json(output/'selection/freeze.json',{'status':'frozen','frozen_utc':now(),'expected_fits':len(selection_tasks(cfg)),
        'completed_fits':len(rows),'test_scores_read':False,'outputs':{p:sha256(output/p) for p in files},
        'selection_completions':completions,'representative_repeat':0})
    print(json.dumps(selection['chosen'],indent=2),flush=True)
    return selection


def prepare(output,data_root,config):
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('Preparation requires a fresh output directory')
    cfg=validate_config(read_json(config))
    output.mkdir(parents=True)
    write_json(output/'config.json',cfg)
    from selection_data import prepare as prepare_data
    data=prepare_data(output,Path(data_root),cfg)
    # A self-contained frozen source tree preserves legacy data-audit imports.
    paths=[AGP_DIR/name for name in RDR_SOURCE_FILES]
    paths += [REPO/'experiments/__init__.py'] if (REPO/'experiments/__init__.py').exists() else []
    paths += [REPO/'utils'/name for name in ('__init__.py','losses.py','model_selection.py','calibration.py')]
    paths += [AGP_DIR/'selection_config.json',AGP_DIR/'selection.slurm']
    frozen={}
    for path in sorted(set(paths)):
        relative=path.relative_to(REPO)
        destination=output/'source'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
        frozen[str(relative)]=sha256(destination)
    (output/'logs').mkdir()
    (output/'selection').mkdir()
    write_json(output/'protocol.json',{'created_utc':now(),'config_sha256':sha256(output/'config.json'),
        'data_protocol_sha256':sha256(output/'data/protocol.json'),'source_sha256':frozen,
        'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'platform':platform.platform()},
        'expected_selection_fits':len(selection_tasks(cfg)),'expected_final_fits':cfg['repeats'],'expected_null_fits':cfg['repeats'],
        'test_scope':'Previously inspected historical cohort; independent of new loss/slope selection, retrospective evidence.',
        'interval_scope':'20 fixed equal-width score bins; no adaptive merging; per-fit nominal cell-average intervals.',
        'roles':data['studies']})
    print(json.dumps({'prepared':str(output),'selection_fits':len(selection_tasks(cfg))}),flush=True)


def finish(output):
    output=Path(output)
    verify_snapshot(output)
    chosen=checked_selection(output)
    cfg=read_json(output/'config.json')
    for stage in ('final','null'):
        for repeat in range(cfg['repeats']):
            task={k:chosen[k] for k in ('candidate','loss','output_alpha')}
            task['repeat']=repeat
            verify_expected_task(output,output/stage/f'{repeat:02d}',stage,task)
    from selection_report import report
    report(output)
    files={str(p.relative_to(output)):sha256(p) for folder in ['summary','selection']
           for p in sorted((output/folder).glob('*')) if p.is_file()}
    write_json(output/'COMPLETE.json',{'status':'complete','finished_utc':now(),
        'selection_fits':len(selection_tasks(cfg)),'final_fits':cfg['repeats'],'null_fits':cfg['repeats'],
        'protocol_sha256':sha256(output/'protocol.json'),'selection_freeze_sha256':sha256(output/'selection/freeze.json'),
        'assessment_completions':{f'{stage}/{repeat:02d}/COMPLETE.json':sha256(output/stage/f'{repeat:02d}/COMPLETE.json')
            for stage in ('final','null') for repeat in range(cfg['repeats'])},
        'outputs':files})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','fit','select','report'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=AGP_DIR/'selection_config.json')
    parser.add_argument('--data-root',type=Path,default=Path('/hpc/group/mastatlab/yx306/AGP/data'))
    parser.add_argument('--stage',choices=['selection','final','null'],default='selection')
    parser.add_argument('--task-index',type=int,default=0)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.output,args.data_root,args.config)
    elif args.action=='fit': run_task(args.output,args.stage,args.task_index,args.device)
    elif args.action=='select': freeze_selection(args.output)
    else: finish(args.output)


if __name__=='__main__':
    main()
