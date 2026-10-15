#!/usr/bin/env python3
"""Separately selected, fixed-bin RDR assessments for three new AGP mask laws."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import selected_workflow as shared

HYBRID = Path('/cwork/yx306/RDR/AGP/icfm_zero_20261007')
MIDASIM = Path('/cwork/yx306/RDR/AGP/midasim_full_20261004')
COMPARISON = Path('/cwork/yx306/RDR/AGP/mask_comparison_20261012')
METHODS = ('bootstrap_icfm', 'bernoulli_mixture_icfm', 'autoregressive_icfm')
SELECTION_SIZES = dict(train=5847, stopping=836, calibration=835, evaluation=836)
read, write, sha = shared.read_json, shared.write_json, shared.sha256


def hashes(root, records):
    for name, digest in records.items():
        p = Path(name) if Path(name).is_absolute() else Path(root)/name
        if sha(p) != digest:
            raise ValueError('Changed pinned artifact: '+str(p))


def inventory(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.log','.out','.err')}


def receipt(root, path):
    return dict(path=str(path.relative_to(root)), sha256=sha(path))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def initialize(out):
    if out.exists():
        raise FileExistsError('Use a fresh RDR study directory')
    # Both existing reference analyses are complete and remain unchanged.
    for ref in (HYBRID/'analysis', MIDASIM/'analysis'):
        complete, protocol = read(ref/'COMPLETE.json'), read(ref/'protocol.json')
        if complete['status'] != 'complete' or complete['protocol_sha256'] != sha(ref/'protocol.json'):
            raise ValueError('Reference analysis completion changed')
        hashes(ref, complete['outputs'])
        hashes(ref, complete['assessment_completions'])
        hashes(ref/'source', protocol['source_sha256'])
    comparison_seal = COMPARISON/('COMPLETE_v2.json' if (COMPARISON/'COMPLETE_v2.json').exists() else 'COMPLETE.json')
    complete = read(comparison_seal)
    if complete['status'] != 'complete':
        raise ValueError('The preceding mask comparison is not complete')
    hashes(COMPARISON, complete['outputs_sha256'])
    gp, gc = read(HYBRID/'generator/protocol.json'), read(HYBRID/'generator/COMPLETE.json')
    if gc['status'] != 'complete' or gc['protocol_sha256'] != sha(HYBRID/'generator/protocol.json'):
        raise ValueError('Conditional abundance flow provenance changed')
    hashes(HYBRID/'generator', gp['data_receipts'])
    hashes(HYBRID/'generator/source', gp['source_sha256'])
    hashes(HYBRID, gp['runtime_receipts'])
    if sha(HYBRID/'generator/training/final.pt') != gc['checkpoint_sha256']:
        raise ValueError('Conditional abundance flow checkpoint changed')
    models = read(COMPARISON/'models/COMPLETE.json')
    hashes(COMPARISON/'models', models['artifacts_sha256'])
    source = out/'source'
    source.mkdir(parents=True)
    own_sources = ['mask_rdr.py', 'mask_rdr.slurm', 'selected_workflow.py', 'selection_report.py',
                   'verify_selected.py', 'verify_midasim_full.py']
    copies = {str(Path('experiments/AGP')/name):HERE/name for name in own_sources}
    copies.update({str(Path('utils')/name):ROOT/'utils'/name
                   for name in ('__init__.py','losses.py','model_selection.py','calibration.py')})
    copies.update({str(Path('experiments/AGP')/name):COMPARISON/'source'/name
                   for name in ('mask_models.py','icfm_zero_generator.py','icfm_corrected_generator.py')})
    for relative, path in copies.items():
        target = source/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    inputs = [comparison_seal, COMPARISON/'protocol.json', COMPARISON/'models/COMPLETE.json',
              HYBRID/'generator/protocol.json', HYBRID/'generator/COMPLETE.json',
              HYBRID/'generator/training/final.pt', HYBRID/'generator/data/generator_train.csv',
              HYBRID/'generator/data/manifest.csv', HYBRID/'generator/data/taxa.csv']
    inputs += [COMPARISON/'models'/name for name in models['artifacts_sha256']]
    for ref in (HYBRID/'analysis', MIDASIM/'analysis'):
        inputs += [ref/name for name in ('COMPLETE.json','protocol.json','config.json','data/protocol.json',
                                        'selection/freeze.json','summary/metrics.csv')]
    parent_data = read(HYBRID/'analysis/data/protocol.json')
    for stage in parent_data['studies'].values():
        inputs += [HYBRID/'analysis'/entry['path'] for entry in [stage['manifest'],stage['audit'],*stage['arrays'].values()]]
    for method in METHODS:
        inputs += [COMPARISON/f'generated/{method}/{role}.npy' for role in ('train','stop','test')]
    inputs += [HYBRID/f'generator/data/noise_selection_{role}.npy' for role in SELECTION_SIZES]
    inputs += [HYBRID/name for name in gp['runtime_receipts']]
    cfg = read(HYBRID/'analysis/config.json')
    shared.validate_config(cfg)
    write(out/'rdr_config.json', cfg)
    write(out/'protocol.json', dict(created_utc=shared.now(), methods=list(METHODS),
          hybrid_root=str(HYBRID), midasim_root=str(MIDASIM), comparison_root=str(COMPARISON),
          references=dict(midasim_mask_icfm=str(HYBRID/'analysis'), midasim=str(MIDASIM/'analysis')),
          source_sha256=inventory(source), inputs_sha256={str(p):sha(p) for p in inputs},
          rdr_config_sha256=sha(out/'rdr_config.json'), selection_q_sizes=SELECTION_SIZES,
          selection_role_order=list(SELECTION_SIZES), mask_seed_base=2027101201,
          flow_sampling=dict(atol=1e-4,rtol=1e-4,batch_size=128,ema=True),
          expected_selection_fits=240, expected_final_fits=15, expected_null_fits=15,
          reference_fits_reused=True,
          scope='Matched retrospective RDR protocol; separately selected per generator; same observed roles and full audited assessment cohort as existing hybrid and MIDASim.',
          limitations=['Real RDR development observations overlap generator training; switching to RDR does not fix the resampled-mask membership blind spot.',
              'Assessment cohort already inspected in prior studies. Selection and scoring are newly staged, not prospectively untouched.',
              'Final calibration is descriptive on the same test rows; five-repeat SD reflects RDR initialization, not independent generated datasets or pointwise ratio accuracy.']))
    (out/'logs').mkdir()
    print('Initialized '+str(out), flush=True)


def gate(out, inputs=False):
    p = read(out/'protocol.json')
    hashes(out/'source', p['source_sha256'])
    if sha(__file__) != p['source_sha256']['experiments/AGP/mask_rdr.py']:
        raise ValueError('Executing wrapper differs from the frozen snapshot')
    if sha(out/'rdr_config.json') != p['rdr_config_sha256']:
        raise ValueError('RDR configuration changed')
    if inputs:
        hashes(out, p['inputs_sha256'])
    return p


def generate(out, device):
    p = gate(out, inputs=True)
    if (out/'generation_complete.json').exists():
        hashes(out/'generated', read(out/'generation_complete.json')['outputs_sha256'])
        return
    model_dir = COMPARISON/'models'
    sampler = module('_mask_rdr_sampler', HERE/'mask_models.py')
    flow = module('_mask_rdr_flow', HERE/'icfm_zero_generator.py')
    model = flow.load_trained_model(HYBRID/'generator/training/final.pt', device=device, ema=True)
    started = time.monotonic()
    for mi, method in enumerate(METHODS):
        folder = out/'generated'/method
        folder.mkdir(parents=True, exist_ok=True)
        for ri, role in enumerate(p['selection_role_order']):
            label = 'selection_'+role
            marker = folder/(label+'.json')
            if marker.exists():
                hashes(folder, read(marker)['outputs_sha256'])
                continue
            seed = p['mask_seed_base']+10000*(mi+1)+1000*(ri+1)
            noise_path = HYBRID/'generator/data'/('noise_'+label+'.npy')
            noise = np.load(noise_path)
            if noise.shape != (SELECTION_SIZES[role],614):
                raise ValueError('Unexpected selection noise dimensions')
            masks, mask_info = sampler.sample_model(model_dir, method.removesuffix('_icfm'), len(noise), seed, device=device)
            latent, flow_info = flow.sample_model(model, noise, masks, device=device, return_metrics=True,
                                                  **{k:v for k,v in p['flow_sampling'].items() if k!='ema'})
            raw = flow.decode_masked_clr(latent,masks)
            for suffix, values in [('',raw),('_mask',masks),('_latent',latent)]:
                np.save(folder/(label+suffix+'.npy'),values,allow_pickle=False)
            write(marker, dict(method=method,role=role,seed=seed,mask=mask_info,ode=flow_info,
                  noise_path=str(noise_path),noise_sha256=sha(noise_path),
                  checkpoint_sha256=sha(HYBRID/'generator/training/final.pt'),
                  outputs_sha256={label+s+'.npy':sha(folder/(label+s+'.npy')) for s in ('','_mask','_latent')}))
            print(f'Generated {method}/{label}: {len(raw)} rows',flush=True)
    write(out/'generation_complete.json',dict(status='complete',finished_utc=shared.now(),
          seconds=time.monotonic()-started,outputs_sha256=inventory(out/'generated'),
          models_receipt_sha256=sha(COMPARISON/'models/COMPLETE.json'),
          comparison_completion_sha256=sha(COMPARISON/'COMPLETE_v2.json')))


def prepare(out):
    root_protocol = gate(out, inputs=True)
    generation = read(out/'generation_complete.json')
    if generation['status'] != 'complete' or generation['models_receipt_sha256'] != sha(COMPARISON/'models/COMPLETE.json'):
        raise ValueError('Selection generation is incomplete or mask models changed')
    hashes(out/'generated',generation['outputs_sha256'])
    parent = HYBRID/'analysis'
    parent_data = read(parent/'data/protocol.json')
    if (out/'analyses').exists():
        raise FileExistsError('Analysis preparation requires fresh caches')
    for method in METHODS:
        analysis = out/'analyses'/method
        (analysis/'data').mkdir(parents=True)
        cfg = read(out/'rdr_config.json')
        cfg.update(generator=method, selection_scope='Independent loss/slope selection for '+method,
                   interval_scope='Fixed20 retrospective descriptive calibration; generator training overlaps real RDR development.')
        write(analysis/'config.json',cfg)
        shutil.copy2(HYBRID/'generator/data/taxa.csv',analysis/'data/taxa.csv')
        studies = {}
        for stage in ('selection','final','null'):
            destination = analysis/'data'/stage
            destination.mkdir()
            parent_study = parent_data['studies'][stage]
            original = pd.read_csv(parent/parent_study['manifest']['path'],dtype={'sample_id':str,'subject_id':str},keep_default_na=False)
            frames, arrays, counts, role_counts = [], {}, {}, {}
            for role in parent_study['roles']:
                role_counts[role] = {}
                for side in ('p','q'):
                    key = role+'_'+side
                    target = destination/(key+'.npy')
                    if side=='p' or stage=='null':
                        entry = parent_study['arrays'][key]
                        source = parent/entry['path']
                        if sha(source) != entry['sha256']:
                            raise ValueError('Changed parent input '+key)
                        shutil.copy2(source,target)
                        values = np.load(target)
                        frame = original[original.role.eq(role)&original.distribution.eq(side.upper())].sort_values('array_row').copy()
                    else:
                        source = (out/'generated'/method/('selection_'+role+'.npy') if stage=='selection'
                                  else COMPARISON/'generated'/method/(('stop' if role=='stopping' else role)+'.npy'))
                        raw = np.load(source)
                        values = raw.astype(np.float32)
                        if (not np.isfinite(values).all() or np.any(values<0) or
                                not np.array_equal(values>0,raw>0) or not np.allclose(values.sum(1),1,atol=2e-6,rtol=0)):
                            raise ValueError('Float32 assessment cache violates composition/support')
                        np.save(target,values,allow_pickle=False)
                        frame = pd.DataFrame(dict(distribution='Q',role=role,source_file=str(source),source_row=np.arange(len(values)),
                              sample_id=[f'{method}:{stage}:{role}:{i:06d}' for i in range(len(values))],subject_id='',
                              row_sha256=[hashlib.sha256(row.astype('<f4').tobytes()).hexdigest() for row in values],
                              exclusion_reason='',previous_role=role,original_role=role,original_distribution='Q',array_row=np.arange(len(values))))
                    if values.shape != tuple(parent_study['arrays'][key]['shape']) or len(frame)!=len(values):
                        raise ValueError('Cached role differs from matched reference design')
                    frames.append(frame)
                    arrays[key] = dict(receipt(analysis,target),shape=list(values.shape),dtype=str(values.dtype))
                    counts[key] = dict(rows=len(values),subjects=int(frame.subject_id.nunique()) if side=='p' or stage=='null' else None)
                    role_counts[role][side] = len(values)
            manifest = pd.concat(frames,ignore_index=True)
            if manifest.sample_id.duplicated().any():
                raise ValueError('Duplicate sample identity in a study')
            real = manifest[manifest.original_distribution.eq('P')]
            if (real.role+':'+real.distribution).groupby(real.subject_id).nunique().max()!=1:
                raise ValueError('Real subjects cross within-study roles')
            manifest.to_csv(destination/'manifest.csv',index=False)
            write(destination/'audit.json',dict(counts=counts,real_roles_identical_to_current_reference=True,
                  subject_disjoint_within_study=True,generator_training_overlaps_real_development=True,
                  selection_test_scores_not_used=True,q_source='fresh selection draws; frozen conventional-comparison final draws; existing real-real null'))
            studies[stage] = dict(roles=parent_study['roles'],arrays=arrays,counts=counts,role_counts=role_counts,
                                 manifest=receipt(analysis,destination/'manifest.csv'),audit=receipt(analysis,destination/'audit.json'))
        write(analysis/'data/protocol.json',dict(studies=studies,feature_dimension=614,
              taxa=receipt(analysis,analysis/'data/taxa.csv'),generator=method,
              hybrid_root=str(HYBRID),comparison_root=str(COMPARISON),
              root_protocol_sha256=sha(out/'protocol.json'),generation_complete_sha256=sha(out/'generation_complete.json'),
              limitations=root_protocol['limitations'],original_excluded_real_test_rows=609))
        shutil.copytree(out/'source',analysis/'source')
        (analysis/'selection').mkdir()
        (analysis/'logs').mkdir()
        write(analysis/'protocol.json',dict(created_utc=shared.now(),config_sha256=sha(analysis/'config.json'),
              data_protocol_sha256=sha(analysis/'data/protocol.json'),source_sha256=inventory(analysis/'source'),
              roles=studies,expected_selection_fits=80,expected_final_fits=5,expected_null_fits=5,
              root_protocol_sha256=sha(out/'protocol.json')))
        print('Prepared '+str(analysis),flush=True)
    write(out/'analysis_prepared.json',dict(status='complete',finished_utc=shared.now(),
          method_protocols={m:sha(out/'analyses'/m/'protocol.json') for m in METHODS}))


def fit(out, stage, index):
    gate(out)
    counts = 80 if stage=='selection' else 5
    if not 0<=index<counts*len(METHODS):
        raise ValueError('Global task index outside grid')
    method, local_index = METHODS[index//counts], index%counts
    shared.run_task(out/'analyses'/method, stage,local_index,'cpu')


def select(out):
    gate(out)
    for method in METHODS:
        shared.freeze_selection(out/'analyses'/method)
    write(out/'selection_frozen.json',dict(status='complete',finished_utc=shared.now(),
          freezes={m:sha(out/'analyses'/m/'selection/freeze.json') for m in METHODS}))


def finish(out):
    gate(out,inputs=True)
    for method in METHODS:
        analysis = out/'analyses'/method
        if (analysis/'COMPLETE.json').exists():
            raise FileExistsError('Analysis already sealed')
        shared.verify_snapshot(analysis)
        chosen = shared.checked_selection(analysis)
        records, assessment = [], {}
        for task in shared.selection_tasks(read(analysis/'config.json')):
            shared.verify_expected_task(analysis,analysis/'selection/runs'/task['candidate']/f'{task["repeat"]:02d}','selection',task)
        for stage in ('final','null'):
            for repeat in range(5):
                task = {k:chosen[k] for k in ('candidate','loss','output_alpha')}
                task['repeat'] = repeat
                folder = analysis/stage/f'{repeat:02d}'
                shared.verify_expected_task(analysis,folder,stage,task)
                assessment[str((folder/'COMPLETE.json').relative_to(analysis))] = sha(folder/'COMPLETE.json')
                records.append(read(folder/'metrics.json'))
        (analysis/'summary').mkdir()
        pd.DataFrame(records).to_csv(analysis/'summary/metrics.csv',index=False)
        outputs = {str(p.relative_to(analysis)):sha(p) for folder in ('summary','selection') for p in (analysis/folder).glob('*') if p.is_file()}
        write(analysis/'COMPLETE.json',dict(status='complete',finished_utc=shared.now(),selection_fits=80,final_fits=5,null_fits=5,
              protocol_sha256=sha(analysis/'protocol.json'),selection_freeze_sha256=sha(analysis/'selection/freeze.json'),
              assessment_completions=assessment,outputs=outputs))
    write(out/'fits_complete.json',dict(status='complete',finished_utc=shared.now(),selection_fits=240,final_fits=15,null_fits=15,
          method_completions={m:sha(out/'analyses'/m/'COMPLETE.json') for m in METHODS}))


def seal(out):
    gate(out,inputs=True)
    if (out/'COMPLETE.json').exists():
        raise FileExistsError('Archive already sealed')
    verification = read(out/'verification.json')
    if verification['status']!='passed' or not (out/'summary/report.md').exists():
        raise ValueError('Independent numerical audit and report must be complete')
    write(out/'COMPLETE.json',dict(status='complete',finished_utc=shared.now(),
          verification_sha256=sha(out/'verification.json'),outputs_sha256=inventory(out),
          new_selection_fits=240,new_final_fits=15,new_null_fits=15,reused_reference_methods=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('initialize','generate','prepare','fit','select','finish','seal'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('selection','final','null'),default='selection')
    parser.add_argument('--task-index',type=int,default=0)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    if args.action=='initialize': initialize(args.output)
    elif args.action=='generate': generate(args.output,args.device)
    elif args.action=='prepare': prepare(args.output)
    elif args.action=='fit': fit(args.output,args.stage,args.task_index)
    elif args.action=='select': select(args.output)
    elif args.action=='finish': finish(args.output)
    elif args.action=='seal': seal(args.output)


if __name__=='__main__': main()
