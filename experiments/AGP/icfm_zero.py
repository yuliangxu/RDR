#!/usr/bin/env python3
"""MIDASim-mask/conditional-ICFM hybrid with separately selected AGP assessment."""
from __future__ import annotations
import argparse
import importlib.metadata
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
import icfm_corrected as corrected
import icfm_zero_generator as generator
from midasim_full import parent_check, checked_parent_array, hashes, snapshot, receipt, row_hashes, ROLE_SIZES, verify_generator as verify_mid_generator

read, write, sha = shared.read_json, shared.write_json, shared.sha256
DEFAULT_OUTPUT = Path('/cwork/yx306/RDR/AGP/icfm_zero_20261007')
DEFAULT_CORRECTED = Path('/cwork/yx306/RDR/AGP/icfm_corrected_20261006')


def source_paths(generation=False):
    names=('icfm_zero.py','icfm_zero_generator.py','icfm_corrected.py',
           'icfm_corrected_generator.py','midasim_full.py','selected_workflow.py')
    paths = [HERE/name for name in (names if generation else
        tuple(dict.fromkeys((*names, *shared.RDR_SOURCE_FILES, 'verify_midasim_full.py'))))]
    paths += [HERE/'icfm_zero.slurm', HERE/'icfm_zero_config.json']
    paths += [ROOT/'utils'/name for name in ('__init__.py', 'losses.py', 'model_selection.py', 'calibration.py')]
    return paths


def prepare(output, corrected_run=DEFAULT_CORRECTED):
    output, corrected_run = Path(output).resolve(), Path(corrected_run).resolve()
    gen = output/'generator'
    if gen.exists() or (output/'runtime').exists():
        raise ValueError('Preparation requires a fresh output directory')
    # Preserve the completed correction study, including its training identities.
    corrected.verify_analysis(corrected_run, 'eval_only')
    corrected.verify_analysis(corrected_run, 'resmlp')
    prior = corrected_run/'generator'
    template = read(prior/'data/protocol.json')
    parent, midasim = Path(template['parent_run']), Path(template['midasim_run'])
    data, mask_data = parent_check(parent), parent_check(midasim)
    raw_path = prior/'data/generator_train.csv'
    raw = np.loadtxt(raw_path, delimiter=',')
    if (raw.shape != (10443,614) or not np.isfinite(raw).all() or (raw<0).any()
            or (raw.sum(1)<=0).any() or sha(raw_path) != template['original_source_sha256']):
        raise ValueError('Original training cohort differs')
    frames = []
    for role in ('train','stopping'):
        x, frame, _ = checked_parent_array(parent, data, 'final', role+'_p')
        if not np.array_equal(raw[frame.source_row.to_numpy(int)].astype(np.float32), x):
            raise ValueError('Generator rows differ from original real development rows')
        frames.append(frame)
    manifest = pd.concat(frames).sort_values('source_row').reset_index(drop=True)
    _, test, _ = checked_parent_array(parent, data, 'final', 'test_p')
    if (manifest.source_row.tolist() != list(range(10443)) or manifest.sample_id.duplicated().any()
            or set(manifest.subject_id) & set(test.subject_id)):
        raise ValueError('Generator identities or assessment separation differ')
    cfg = read(HERE/'icfm_zero_config.json')
    if cfg['roles'] != ROLE_SIZES or set(cfg['role_order']) != set(ROLE_SIZES) or len(cfg['role_order']) != 7:
        raise ValueError('Unexpected generation role design')
    if sha(parent/data['taxa']['path']) != data['taxa']['sha256']:
        raise ValueError('Parent taxonomy changed')
    if sha(parent/data['taxa']['path']) != sha(midasim/mask_data['taxa']['path']):
        raise ValueError('Mask-model taxon order differs')
    # Check full-template MIDASim was fitted to the identical raw observations.
    midgen = midasim.parent/'generator'
    verify_mid_generator(midasim.parent)
    midprov = read(midgen/'data/protocol.json')
    if midprov.get('original_source_sha256') != sha(raw_path):
        raise ValueError('MIDASim mask template differs from the original ICFM template')
    (gen/'data').mkdir(parents=True)
    (gen/'logs').mkdir()
    shutil.copytree(corrected_run/'runtime', output/'runtime', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(raw_path, gen/'data/generator_train.csv')
    shutil.copyfile(parent/data['taxa']['path'], gen/'data/taxa.csv')
    manifest.to_csv(gen/'data/manifest.csv', index=False)
    masks, noises = {}, {}
    for index, label in enumerate(cfg['role_order']):
        stage, role = label.split('_',1)
        q, _, original = checked_parent_array(midasim, mask_data, stage, role+'_q')
        mask = q > 0
        if q.shape != (ROLE_SIZES[label],614) or not mask.any(1).all():
            raise ValueError('Invalid or empty MIDASim support')
        mask_path = gen/'data'/('mask_'+label+'.npy')
        np.save(mask_path, mask, allow_pickle=False)
        masks[label] = dict(receipt(gen,mask_path), source=str(original), source_sha256=sha(original),
            shape=list(mask.shape), dtype='bool', law='Exact mask from existing MIDASim full-template role draw/subset')
        seed = cfg['noise_seed'] + cfg['noise_seed_stride']*index
        noise = np.random.default_rng(seed).standard_normal(mask.shape).astype(np.float32)
        noise_path = gen/'data'/('noise_'+label+'.npy')
        np.save(noise_path,noise,allow_pickle=False)
        noises[label] = dict(receipt(gen,noise_path),seed=seed,shape=list(noise.shape),dtype='float32',
            algorithm='numpy.default_rng(PCG64).standard_normal(float64)->float32')
    upstream = {str(p):sha(p) for p in (parent/'COMPLETE.json',parent/'data/protocol.json',
        corrected_run/'comparison/COMPLETE.json',prior/'protocol.json',midasim/'COMPLETE.json',
        midasim/'data/protocol.json',midgen/'protocol.json',midgen/'data/protocol.json',midgen/'generation/COMPLETE.json')}
    write(gen/'data/protocol.json',dict(parent_run=str(parent),parent_completion_sha256=sha(parent/'COMPLETE.json'),
        corrected_run=str(corrected_run),corrected_completion_sha256=sha(corrected_run/'comparison/COMPLETE.json'),
        midasim_run=str(midasim),midasim_completion_sha256=sha(midasim/'COMPLETE.json'),
        original_source=str(raw_path),original_source_sha256=sha(raw_path),rows=10443,
        subjects=int(manifest.subject_id.nunique()),taxa=614,coordinate_dimension=614,
        mask_model='Completed nonparametric MIDASim full-template dependent presence model',
        masks=masks,noise=noises,mask_pairing=cfg['mask_pairing'],upstream_receipts=upstream,
        zero_handling=cfg['zero_handling'],
        generator_overlap='Original10443 generator-training observations overlap RDR development roles.',
        repeat_scope='One conditional generator fit, fixed paired-mask Q banks, five RDR initialization repeats.',
        limitations=['Retrospective real assessment cohort; all609 subject-overlap exclusions retained.',
            'Masks are paired with MIDASim; support summaries agree by construction, not independently.',
            'MIDASim masks retain within-table dependence from full-template draws and subsets.',
            'Positive-abundance representation changes from smoothed tree ratios to conditional masked CLR.',
            'New614-dimensional noise is not paired with old613-dimensional unconditional ICFM noise.',
            'Seven Q roles remain distinct; archived ICFM reused development Q across stages.']))
    write(gen/'config.json',cfg)
    frozen = snapshot(gen,source_paths(generation=True))
    write(gen/'protocol.json',dict(created_utc=shared.now(),config_sha256=sha(gen/'config.json'),
        source_sha256=frozen,data_receipts={str(p.relative_to(gen)):sha(p) for p in (gen/'data').iterdir()},
        runtime_receipts={str(p.relative_to(output)):sha(p) for p in (output/'runtime').rglob('*') if p.is_file()}))
    print('Prepared zero-aware ICFM: '+str(output),flush=True)


def generator_gate(output):
    output = Path(output).resolve()
    gen = output/'generator'
    protocol = read(gen/'protocol.json')
    if protocol['config_sha256'] != sha(gen/'config.json'):
        raise ValueError('Generator config changed')
    hashes(gen,protocol['data_receipts'])
    hashes(gen/'source',protocol['source_sha256'])
    hashes(output,protocol['runtime_receipts'])
    for name,path in [('icfm_zero.py',__file__),('icfm_zero_generator.py',generator.__file__),
                      ('icfm_corrected_generator.py',generator.shared.__file__)]:
        if sha(path) != protocol['source_sha256']['experiments/AGP/'+name]:
            raise ValueError('Executing generator source differs: '+name)
    for path,digest in read(gen/'data/protocol.json')['upstream_receipts'].items():
        if sha(path) != digest:
            raise ValueError('Upstream study receipt changed: '+path)
    return gen


def train(output):
    gen = generator_gate(output)
    return generator.train(read(gen/'config.json'),gen/'data/generator_train.csv',gen/'training')


def training_gate(gen):
    cfg=generator.normalized_config(read(gen/'config.json'))
    detail=read(gen/'training/metrics.json')
    identity=dict(config_sha256=generator.shared.canonical_hash(cfg),
                  data_sha256=sha(gen/'data/generator_train.csv'),
                  representation='positive-taxon-centered-log-abundance')
    rows=read(gen/'data/protocol.json')['rows']
    updates=cfg['training']['epochs']*(rows//cfg['training']['batch_size'])
    if (detail.get('status')!='complete' or detail.get('identity')!=identity
            or detail.get('epochs_completed')!=cfg['training']['epochs'] or detail.get('updates')!=updates
            or detail.get('rows')!=rows or detail.get('coordinates')!=cfg['model']['dim']
            or detail.get('final_checkpoint_sha256')!=sha(gen/'training/final.pt')
            or detail.get('training_log_sha256')!=sha(gen/'training/training.json')):
        raise ValueError('Conditional training does not match the complete fixed-budget protocol')
    return detail


def generate(output):
    gen = generator_gate(output)
    if (gen/'COMPLETE.json').exists():
        return verify_generation(output)
    cfg, trained = read(gen/'config.json'), training_gate(gen)
    checkpoint = gen/'training/final.pt'
    if trained['status'] != 'complete' or trained['final_checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('Conditional flow training incomplete or changed')
    model = generator.load_trained_model(checkpoint,cfg['sampling']['device'])
    folder = gen/'generation'
    folder.mkdir(exist_ok=True)
    roles, started = {},time.monotonic()
    for role in cfg['role_order']:
        metrics_path = folder/(role+'_metrics.json')
        if metrics_path.exists():
            detail=read(metrics_path)
            hashes(folder,detail['outputs'])
            if detail['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('Partial generation used another checkpoint')
            roles[role]=detail
            continue
        if (folder/(role+'.npy')).exists() or (folder/(role+'_latent.npy')).exists():
            raise ValueError('Preserve and inspect incomplete generation role: '+role)
        noise_path, mask_path = gen/'data'/('noise_'+role+'.npy'),gen/'data'/('mask_'+role+'.npy')
        noise,masks=np.load(noise_path,allow_pickle=False),np.load(mask_path,allow_pickle=False)
        latent,sampling=generator.sample_model(model,noise,masks,**{k:cfg['sampling'][k]
            for k in ('atol','rtol','batch_size','device')},return_metrics=True)
        raw=generator.decode_masked_clr(latent,masks)
        if (raw.shape!=(ROLE_SIZES[role],614) or not np.isfinite(raw).all() or (raw<0).any()
                or not np.array_equal(raw>0,masks) or not np.allclose(raw.sum(1),1,atol=1e-12,rtol=0)):
            raise ValueError('Generated composition violates its prescribed support')
        np.save(folder/(role+'_latent.npy'),latent.astype(np.float32),allow_pickle=False)
        np.save(folder/(role+'.npy'),raw.astype(np.float64),allow_pickle=False)
        detail=dict(role=role,rows=len(raw),taxa=614,sampling=sampling,
            noise_sha256=sha(noise_path),mask_sha256=sha(mask_path),checkpoint_sha256=sha(checkpoint),
            zero_fraction=float((raw==0).mean()),mean_positive_taxa=float((raw>0).sum(1).mean()),
            maximum_closure_error=float(np.abs(raw.sum(1)-1).max()),exact_mid_mask=True,
            outputs={role+suffix:sha(folder/(role+suffix)) for suffix in ('.npy','_latent.npy')})
        write(metrics_path,detail)
        roles[role]=detail
        print(f'{role}: {len(raw)} rows, exact prescribed support, positive taxa={detail["mean_positive_taxa"]:.3f}',flush=True)
    versions={name:importlib.metadata.version(name) for name in ('torch','torchdyn','torchdiffeq','numpy','scipy','pandas','attrs')}
    write(folder/'metrics.json',dict(status='complete',method=cfg['method'],roles=roles,versions=versions,
        elapsed_seconds=time.monotonic()-started,explicit_evaluation_mode=True,paired_mid_masks=True,
        parameter_count=sum(p.numel() for p in model.parameters()),checkpoint_sha256=sha(checkpoint)))
    write(gen/'COMPLETE.json',dict(status='complete',finished_utc=shared.now(),
        config_sha256=sha(gen/'config.json'),protocol_sha256=sha(gen/'protocol.json'),
        checkpoint_sha256=sha(checkpoint),training_metrics_sha256=sha(gen/'training/metrics.json'),
        outputs={str(p.relative_to(gen)):sha(p) for p in folder.iterdir() if p.is_file()}))
    return verify_generation(output)


def verify_generation(output):
    gen=generator_gate(output)
    training_gate(gen)
    marker=read(gen/'COMPLETE.json')
    if marker['status']!='complete':
        raise ValueError('Generator incomplete')
    for key,path in [('config_sha256',gen/'config.json'),('protocol_sha256',gen/'protocol.json'),
                     ('checkpoint_sha256',gen/'training/final.pt'),('training_metrics_sha256',gen/'training/metrics.json')]:
        if marker[key]!=sha(path):
            raise ValueError('Generation provenance changed')
    required={'generation/'+role+suffix for role in ROLE_SIZES for suffix in ('.npy','_latent.npy','_metrics.json')} | {'generation/metrics.json'}
    if set(marker['outputs'])!=required:
        raise ValueError('Incomplete generated inventory')
    hashes(gen,marker['outputs'])
    return marker


def prepare_analysis(output):
    output=Path(output).resolve()
    verify_generation(output)
    gen=output/'generator'
    template=read(gen/'data/protocol.json')
    parent,midasim=Path(template['parent_run']),Path(template['midasim_run'])
    data=parent_check(parent)
    analysis=output/'analysis'
    if analysis.exists():
        raise ValueError('Analysis preparation requires a fresh destination')
    cfg=read(parent/'config.json')
    cfg.update(generator='ICFM_zero_aware',generator_training_rows=10443,
        selection_scope='Separate loss/slope selection for MIDASim-mask/conditional-ICFM hybrid',
        interval_scope='Fixed20 descriptive diagnostics; retrospective cohort and dependent mask rows limit coverage.')
    shared.validate_config(cfg)
    analysis.mkdir(parents=True)
    write(analysis/'config.json',cfg)
    (analysis/'data').mkdir()
    shutil.copyfile(gen/'data/taxa.csv',analysis/'data/taxa.csv')
    studies={}
    for stage in ('selection','final','null'):
        folder=analysis/'data'/stage
        folder.mkdir()
        roles=data['studies'][stage]['roles']
        frames,arrays,counts,role_counts=[],{},{},{}
        for role in roles:
            role_counts[role]={}
            for side in ('p','q'):
                key=role+'_'+side
                target=folder/(key+'.npy')
                if side=='p' or stage=='null':
                    x,frame,source=checked_parent_array(parent,data,stage,key)
                    shutil.copyfile(source,target)
                else:
                    label=stage+'_'+role
                    source=gen/'generation'/(label+'.npy')
                    x64=np.load(source,allow_pickle=False)
                    mask=np.load(gen/'data'/('mask_'+label+'.npy'),allow_pickle=False)
                    if (x64.shape!=(ROLE_SIZES[label],614) or not np.isfinite(x64).all() or (x64<0).any() or
                            not np.array_equal(x64>0,mask) or not np.allclose(x64.sum(1),1,atol=1e-12,rtol=0)):
                        raise ValueError('Invalid generated table on analysis import')
                    x=x64.astype(np.float32)
                    if not np.array_equal(x>0,mask):
                        raise ValueError('Float32 cache loses positive support; preserve and inspect')
                    np.save(target,x,allow_pickle=False)
                    frame=pd.DataFrame(dict(distribution='Q',role=role,source_file=str(source),source_row=np.arange(len(x)),
                        sample_id=[f'ICFM:zero_aware:{stage}:{role}:{i:06d}' for i in range(len(x))],subject_id='',
                        row_sha256=row_hashes(x),exclusion_reason='',previous_role=role,original_role=role,
                        original_distribution='Q',array_row=np.arange(len(x))))
                frames.append(frame)
                arrays[key]=dict(receipt(analysis,target),shape=list(x.shape),dtype=str(x.dtype))
                counts[key]=dict(rows=len(x),subjects=int(frame.subject_id.nunique()) if (side=='p' or stage=='null') else None)
                role_counts[role][side]=len(x)
        manifest=pd.concat(frames,ignore_index=True)
        if manifest.sample_id.duplicated().any():
            raise ValueError('Duplicate sample IDs')
        real=manifest[manifest.original_distribution.eq('P')]
        if (real.role+':'+real.distribution).groupby(real.subject_id).nunique().max()!=1:
            raise ValueError('Subjects cross RDR roles')
        manifest.to_csv(folder/'manifest.csv',index=False)
        write(folder/'audit.json',dict(real_roles_identical_to_main_icfm=True,subject_disjoint_rdr_roles=True,
            q_roles_independently_generated=stage!='null',paired_mid_masks=stage!='null',counts=counts,
            no_test_scores_used_for_selection=True,row_hash_dtype='little-endian float32'))
        studies[stage]=dict(roles=roles,arrays=arrays,counts=counts,role_counts=role_counts,
            manifest=receipt(analysis,folder/'manifest.csv'),audit=receipt(analysis,folder/'audit.json'))
    provenance={k:template[k] for k in ('parent_run','parent_completion_sha256','corrected_run',
        'corrected_completion_sha256','midasim_run','midasim_completion_sha256','limitations','generator_overlap','repeat_scope')}
    provenance.update(generator_run=str(gen),generator_protocol_sha256=sha(gen/'protocol.json'),
        generator_completion_sha256=sha(gen/'COMPLETE.json'),feature_dimension=614,studies=studies,
        taxa=receipt(analysis,analysis/'data/taxa.csv'),original_excluded_real_test_rows=609)
    write(analysis/'data/protocol.json',provenance)
    frozen=snapshot(analysis,source_paths())
    (analysis/'logs').mkdir()
    (analysis/'selection').mkdir()
    write(analysis/'protocol.json',dict(created_utc=shared.now(),config_sha256=sha(analysis/'config.json'),
        data_protocol_sha256=sha(analysis/'data/protocol.json'),source_sha256=frozen,roles=studies,
        expected_selection_fits=80,expected_final_fits=5,expected_null_fits=5,
        generator_completion_sha256=sha(gen/'COMPLETE.json')))
    print('Prepared zero-aware ICFM analysis: '+str(analysis),flush=True)


def analysis_gate(output):
    output=Path(output).resolve()
    analysis=output/'analysis'
    shared.verify_snapshot(analysis)
    protocol=read(analysis/'protocol.json')
    if sha(__file__)!=protocol['source_sha256']['experiments/AGP/icfm_zero.py']:
        raise ValueError('Executing wrapper differs from snapshot')
    data=read(analysis/'data/protocol.json')
    receipts=[data['taxa']]
    for study in data['studies'].values():
        receipts += [study['manifest'],study['audit'],*study['arrays'].values()]
    hashes(analysis,{item['path']:item['sha256'] for item in receipts})
    if data['generator_protocol_sha256']!=sha(output/'generator/protocol.json') or data['generator_completion_sha256']!=sha(output/'generator/COMPLETE.json'):
        raise ValueError('Analysis generator inputs changed')
    verify_generation(output)
    return analysis


def seal_analysis(output):
    analysis=analysis_gate(output)
    if (analysis/'COMPLETE.json').exists():
        return verify_analysis(output)
    chosen=shared.checked_selection(analysis)
    completion,metrics={},[]
    for stage in ('final','null'):
        for repeat in range(5):
            task={k:chosen[k] for k in ('candidate','loss','output_alpha')}
            task['repeat']=repeat
            folder=analysis/stage/f'{repeat:02d}'
            shared.verify_expected_task(analysis,folder,stage,task)
            completion[str((folder/'COMPLETE.json').relative_to(analysis))]=sha(folder/'COMPLETE.json')
            metrics.append(read(folder/'metrics.json'))
    (analysis/'summary').mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv(analysis/'summary/metrics.csv',index=False)
    files={str(p.relative_to(analysis)):sha(p) for name in ('summary','selection') for p in (analysis/name).glob('*') if p.is_file()}
    write(analysis/'COMPLETE.json',dict(status='complete',finished_utc=shared.now(),selection_fits=80,final_fits=5,null_fits=5,
        protocol_sha256=sha(analysis/'protocol.json'),selection_freeze_sha256=sha(analysis/'selection/freeze.json'),
        assessment_completions=completion,outputs=files))
    return verify_analysis(output)


def verify_analysis(output):
    analysis=analysis_gate(output)
    marker=read(analysis/'COMPLETE.json')
    if marker['status']!='complete' or (marker['selection_fits'],marker['final_fits'],marker['null_fits'])!=(80,5,5):
        raise ValueError('Analysis completion counts differ')
    for key,name in [('protocol_sha256','protocol.json'),('selection_freeze_sha256','selection/freeze.json')]:
        if marker[key]!=sha(analysis/name):
            raise ValueError('Analysis completion provenance differs')
    hashes(analysis,marker['outputs'])
    hashes(analysis,marker['assessment_completions'])
    for task in shared.selection_tasks(read(analysis/'config.json')):
        shared.verify_expected_task(analysis,analysis/'selection/runs'/task['candidate']/f'{task["repeat"]:02d}','selection',task)
    chosen=shared.checked_selection(analysis)
    for stage in ('final','null'):
        for repeat in range(5):
            task={k:chosen[k] for k in ('candidate','loss','output_alpha')}
            task['repeat']=repeat
            shared.verify_expected_task(analysis,analysis/stage/f'{repeat:02d}',stage,task)
    return marker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','benchmark','train','generate','prepare-analysis','fit','select','seal-analysis','verify-generation','verify'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--corrected',type=Path,default=DEFAULT_CORRECTED)
    parser.add_argument('--stage',choices=['selection','final','null'],default='selection')
    parser.add_argument('--task-index',type=int,default=0)
    args=parser.parse_args()
    if args.action=='prepare': prepare(args.output,args.corrected)
    elif args.action=='train': train(args.output)
    elif args.action=='benchmark':
        gen=generator_gate(args.output)
        generator.benchmark(read(gen/'config.json'),gen/'data/generator_train.csv',args.output/'benchmark.json')
    elif args.action=='generate': generate(args.output)
    elif args.action=='prepare-analysis': prepare_analysis(args.output)
    elif args.action=='fit': shared.run_task(analysis_gate(args.output),args.stage,args.task_index,'cpu')
    elif args.action=='select': shared.freeze_selection(analysis_gate(args.output))
    elif args.action=='seal-analysis': seal_analysis(args.output)
    elif args.action=='verify-generation': verify_generation(args.output)
    elif args.action=='verify': verify_analysis(args.output)


if __name__=='__main__': main()
