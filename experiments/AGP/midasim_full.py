#!/usr/bin/env python3
"""Full-template MIDASim with the main AGP split and separate RDR selection."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import selected_workflow as shared

read, write, sha = shared.read_json, shared.write_json, shared.sha256
DEFAULT_PARENT = Path('/cwork/yx306/RDR/AGP/fixed_width_selection_20261003')
DEFAULT_PILOT = Path('/cwork/yx306/RDR/AGP/midasim_pilot_20261004')
ROLE_SIZES = dict(selection_train=5847, selection_stopping=836,
                  selection_calibration=835, selection_evaluation=836,
                  final_train=7310, final_stopping=1044, final_test=2089)


def hashes(root, receipts):
    if not receipts:
        raise ValueError('Missing artifact receipts')
    for name, digest in receipts.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or sha(Path(root)/path) != digest:
            raise ValueError(f'Changed or invalid artifact: {name}')


def receipt(root, path):
    return dict(path=str(Path(path).relative_to(root)), sha256=sha(path))


def row_hashes(values):
    return [hashlib.sha256(row.astype('<f4').tobytes()).hexdigest() for row in values]


def parent_check(parent):
    parent = Path(parent)
    completion, protocol = read(parent/'COMPLETE.json'), read(parent/'protocol.json')
    if completion['status'] != 'complete' or completion['protocol_sha256'] != sha(parent/'protocol.json'):
        raise ValueError('Parent study incomplete or changed')
    for key, name in [('config_sha256','config.json'),('data_protocol_sha256','data/protocol.json')]:
        if protocol[key] != sha(parent/name):
            raise ValueError('Changed parent preparation')
    hashes(parent, completion['outputs'])
    hashes(parent/'source', protocol['source_sha256'])
    data = read(parent/'data/protocol.json')
    return data


def checked_parent_array(parent, data, study, key):
    spec = data['studies'][study]['arrays'][key]
    if sha(parent/spec['path']) != spec['sha256']:
        raise ValueError('Changed parent array')
    x = np.load(parent/spec['path'], allow_pickle=False)
    ms = data['studies'][study]['manifest']
    if sha(parent/ms['path']) != ms['sha256']:
        raise ValueError('Changed parent manifest')
    frame = pd.read_csv(parent/ms['path'], dtype={'sample_id':str,'subject_id':str}, keep_default_na=False)
    role, side = key.rsplit('_',1)
    frame = frame[frame.role.eq(role)&frame.distribution.eq(side.upper())].sort_values('array_row').copy()
    if len(frame) != len(x) or not np.array_equal(frame.array_row, np.arange(len(x))):
        raise ValueError('Parent row order/count mismatch')
    if frame.row_sha256.tolist() != row_hashes(x):
        raise ValueError('Parent row hashes mismatch')
    return x, frame, parent/spec['path']


def snapshot(destination, paths):
    result = {}
    for path in sorted(set(paths)):
        path = Path(path)
        if not path.is_file():
            raise ValueError(f'Missing source before freeze: {path}')
        name = str(path.relative_to(ROOT))
        target = destination/'source'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        result[name] = sha(target)
    return result


def prepare_generator(output, parent=DEFAULT_PARENT, pilot=DEFAULT_PILOT):
    output, parent, pilot = [Path(p).resolve() for p in (output,parent,pilot)]
    gen = output/'generator'
    if gen.exists():
        raise ValueError('Generator preparation requires a fresh destination')
    data = parent_check(parent)
    original = Path(data['data_root'])/'yx1_xtrain_raw.csv'
    if sha(original) != data['input_sha256'][original.name]:
        raise ValueError('Changed original generator-training data')
    raw = np.loadtxt(original,delimiter=',')
    if raw.shape != (10443,614) or not np.isfinite(raw).all() or (raw<0).any() or (raw.sum(1)<=0).any():
        raise ValueError('Unexpected full template')
    pieces=[]
    for role in ('train','stopping'):
        x, frame, _ = checked_parent_array(parent,data,'final',role+'_p')
        if not np.array_equal(raw[frame.source_row.to_numpy(int)].astype(np.float32),x):
            raise ValueError('Template does not match main development rows')
        pieces.append(frame)
    manifest=pd.concat(pieces).sort_values('source_row').reset_index(drop=True)
    if not np.array_equal(manifest.source_row,np.arange(len(raw))) or manifest.sample_id.duplicated().any():
        raise ValueError('Template source-row partition mismatch')
    _,test,_=checked_parent_array(parent,data,'final','test_p')
    if set(manifest.subject_id)&set(test.subject_id):
        raise ValueError('Generator and assessment subjects overlap')
    active=np.flatnonzero((raw>0).any(0))
    if len(active)!=613 or np.any((raw>0).sum(0)==1):
        raise ValueError('Unexpected active/singleton taxa')
    (gen/'data').mkdir(parents=True)
    (gen/'vendor').mkdir()
    (gen/'logs').mkdir()
    shutil.copyfile(original,gen/'data/generator_train.csv')
    manifest.to_csv(gen/'data/manifest.csv',index=False)
    taxa=data['taxa']
    if sha(parent/taxa['path'])!=taxa['sha256']:
        raise ValueError('Changed taxonomy')
    shutil.copyfile(parent/taxa['path'],gen/'data/taxa.csv')
    tarball=pilot/'vendor/MIDASim_2.0.tar.gz'
    expected_tar='841f0d14bb8b64176fc7fea34136a691fbe49e5e703fabc16571d0282bbef841'
    if sha(tarball)!=expected_tar:
        raise ValueError('Unexpected MIDASim source distribution')
    shutil.copyfile(tarball,gen/'vendor/MIDASim_2.0.tar.gz')
    write(gen/'data/protocol.json',dict(parent_run=str(parent),parent_completion_sha256=sha(parent/'COMPLETE.json'),
        parent_data_protocol_sha256=sha(parent/'data/protocol.json'),original_source=str(original),
        original_source_sha256=sha(original),rows=10443,subjects=int(manifest.subject_id.nunique()),taxa=614,
        active_taxa=active.tolist(),restored_zero_taxa=sorted(set(range(614))-set(active.tolist())),
        cohort='Current audited main cohort: Ptest2002;609 historical subject-overlap exclusions retained.',
        generator_training='Exact original ICFM generator-training rows in original source order, float64 CSV.',
        overlap='Generator uses all real development rows, including RDR stopping and selection roles, as ICFM did.',
        sampling='One fixed independently drawn Q table per role, shared across all candidate/initialization fits.',
        limitations=['Historical real test cohort and whole-table filtering are retrospective.',
                    'MIDASim rank matching creates dependence within each generated role table.',
                    'ICFM archived generator and MIDASim new fit use the same observations but different algorithms and preprocessing.',
                    'Relative-only model does not generate read counts or model sequencing depths.']))
    write(gen/'config.json',dict(generator='MIDASim',version='2.0',mode='nonparametric',only_relative_abundances=True,
        generation_seed=2026100601,generation_seed_offset=100000,n_break_ties=100,roles=ROLE_SIZES,role_order=list(ROLE_SIZES),
        generator_template_n=10443,feature_dimension=614,threads=2))
    paths=[HERE/name for name in ('midasim_full_fit.R','midasim_fit.R','midasim_full.py','midasim_full.slurm')]
    frozen=snapshot(gen,paths)
    inputs={str(p.relative_to(gen)):sha(p) for folder in ('data','vendor') for p in (gen/folder).iterdir() if p.is_file()}
    write(gen/'protocol.json',dict(created_utc=shared.now(),config_sha256=sha(gen/'config.json'),
        source_sha256=frozen,data_receipts=inputs))
    print(f'Prepared full MIDASim template: {gen}',flush=True)


def verify_generator(output, complete=True):
    gen=Path(output)/'generator'
    protocol=read(gen/'protocol.json')
    if protocol['config_sha256']!=sha(gen/'config.json'):
        raise ValueError('Changed generator configuration')
    hashes(gen,protocol['data_receipts'])
    hashes(gen/'source',protocol['source_sha256'])
    if complete:
        marker=read(gen/'generation/COMPLETE.json')
        if marker['status']!='complete':
            raise ValueError('Incomplete generation')
        for key,name in [('config_sha256','config.json'),('protocol_sha256','protocol.json')]:
            if marker[key]!=sha(gen/name):
                raise ValueError('Generation provenance mismatch')
        hashes(gen,marker['outputs'])
        required={f'generation/{r}{suffix}' for r in ROLE_SIZES
                  for suffix in ('.csv','_indices.csv','_metrics.json')}
        required.update('generation/'+name for name in ('fitted.rds','modified.rds','metrics.json','sessionInfo.txt'))
        if not required.issubset(marker['outputs']):
            raise ValueError('Missing generated roles or fitting provenance')
    return protocol


def prepare_analysis(output):
    output=Path(output).resolve()
    verify_generator(output)
    gen=output/'generator'
    template=read(gen/'data/protocol.json')
    parent=Path(template['parent_run'])
    data=parent_check(parent)
    if (template['parent_completion_sha256']!=sha(parent/'COMPLETE.json') or
            template['parent_data_protocol_sha256']!=sha(parent/'data/protocol.json')):
        raise ValueError('Parent changed since generator preparation')
    analysis=output/'analysis'
    if analysis.exists():
        raise ValueError('Analysis requires a fresh destination')
    cfg=read(parent/'config.json')
    cfg.update(generator='MIDASim',generator_training_rows=10443,
               selection_scope='Independent P-versus-MIDASim loss/slope selection; fixed Q per role.',
               interval_scope='Fixed20 descriptive diagnostics; within-table rank dependence prevents IID coverage claims.')
    shared.validate_config(cfg)
    analysis.mkdir()
    write(analysis/'config.json',cfg)
    (analysis/'data').mkdir()
    shutil.copyfile(gen/'data/taxa.csv',analysis/'data/taxa.csv')
    studies={}
    for stage in ('selection','final','null'):
        folder=analysis/'data'/stage
        folder.mkdir()
        roles=data['studies'][stage]['roles']
        frames=[]
        arrays={}
        counts={}
        role_counts={}
        for role in roles:
            role_counts[role]={}
            for side in ('p','q'):
                key=role+'_'+side
                if side=='p' or stage=='null':
                    x,frame,source=checked_parent_array(parent,data,stage,key)
                    target=folder/(key+'.npy')
                    shutil.copyfile(source,target)
                else:
                    label=stage+'_'+role
                    source=gen/'generation'/(label+'.csv')
                    x64=np.loadtxt(source,delimiter=',',ndmin=2)
                    if (x64.shape!=(ROLE_SIZES[label],614) or not np.isfinite(x64).all()
                            or (x64<0).any() or not np.allclose(x64.sum(1),1,atol=1e-10,rtol=0)
                            or np.any(x64[:,template['restored_zero_taxa']]!=0)):
                        raise ValueError('Invalid generated compositions')
                    x=x64.astype(np.float32)
                    target=folder/(key+'.npy')
                    np.save(target,x,allow_pickle=False)
                    frame=pd.DataFrame(dict(distribution='Q',role=role,source_file=str(source),
                        source_row=np.arange(len(x)),sample_id=[f'MIDASim:{stage}:{role}:{i:06d}' for i in range(len(x))],
                        subject_id='',row_sha256=row_hashes(x),exclusion_reason='',previous_role=role,
                        original_role=role,original_distribution='Q',array_row=np.arange(len(x))))
                frames.append(frame)
                arrays[key]=dict(receipt(analysis,target),shape=list(x.shape),dtype=str(x.dtype))
                counts[key]=dict(rows=len(x),subjects=int(frame.subject_id.nunique()) if (side=='p' or stage=='null') else None)
                role_counts[role][side]=len(x)
        manifest=pd.concat(frames,ignore_index=True)
        if manifest.sample_id.duplicated().any():
            raise ValueError('Duplicate analysis sample IDs')
        real=manifest[manifest.original_distribution.eq('P')]
        if (real.role+':'+real.distribution).groupby(real.subject_id).nunique().max()!=1:
            raise ValueError('Real subjects cross RDR roles')
        manifest.to_csv(folder/'manifest.csv',index=False)
        write(folder/'audit.json',dict(real_roles_identical_to_main_icfm=True,subject_disjoint_rdr_roles=True,
            q_roles_independently_generated=stage!='null',row_hash_dtype='little-endian float32',
            no_test_scores_used_for_selection=True,counts=counts))
        studies[stage]=dict(roles=roles,arrays=arrays,counts=counts,role_counts=role_counts,
            manifest=receipt(analysis,folder/'manifest.csv'),audit=receipt(analysis,folder/'audit.json'))
    write(analysis/'data/protocol.json',dict(parent_run=str(parent),parent_completion_sha256=sha(parent/'COMPLETE.json'),
        generator_run=str(gen),generator_completion_sha256=sha(gen/'generation/COMPLETE.json'),
        generator_protocol_sha256=sha(gen/'protocol.json'),feature_dimension=614,studies=studies,
        taxa=receipt(analysis,analysis/'data/taxa.csv'),original_excluded_real_test_rows=609,
        limitations=template['limitations'],generator_overlap=template['overlap'],repeat_scope='Initialization only; fixed P/Q datasets.',
        q_cache='Generated float64 CSV rounded to float32 to match main ICFM cache convention; raw CSVs retained.'))
    paths=[HERE/name for name in (*shared.RDR_SOURCE_FILES, 'midasim_full.py',
        'verify_midasim_full.py', 'midasim_full_fit.R', 'midasim_fit.R')]
    paths += [HERE/'midasim_full.slurm']+[ROOT/'utils'/name for name in ('__init__.py','losses.py','model_selection.py','calibration.py')]
    frozen=snapshot(analysis,paths)
    (analysis/'logs').mkdir()
    (analysis/'selection').mkdir()
    write(analysis/'protocol.json',dict(created_utc=shared.now(),config_sha256=sha(analysis/'config.json'),
        data_protocol_sha256=sha(analysis/'data/protocol.json'),source_sha256=frozen,
        expected_selection_fits=80,expected_final_fits=5,expected_null_fits=5,roles=studies,
        generator_completion_sha256=sha(gen/'generation/COMPLETE.json'),
        test_scope='Audited historical cohort, previously inspected; withheld from this new selection.',
        interval_scope=cfg['interval_scope']))
    print(f'Prepared MIDASim analysis: {analysis}',flush=True)


def analysis_gate(output):
    output=Path(output)
    analysis=output/'analysis'
    shared.verify_snapshot(analysis)
    own='experiments/AGP/midasim_full.py'
    if sha(__file__)!=read(analysis/'protocol.json')['source_sha256'][own]:
        raise ValueError('Executing full-study wrapper differs from snapshot')
    data=read(analysis/'data/protocol.json')
    receipts=[data['taxa']]
    for study in data['studies'].values():
        receipts += [study['manifest'],study['audit'],*study['arrays'].values()]
    hashes(analysis,{item['path']:item['sha256'] for item in receipts})
    gen=output/'generator'
    if data['generator_completion_sha256']!=sha(gen/'generation/COMPLETE.json'):
        raise ValueError('Generator completion changed')
    if data['generator_protocol_sha256']!=sha(gen/'protocol.json'):
        raise ValueError('Generator preparation changed')
    verify_generator(output)
    return analysis


def seal_analysis(output):
    analysis=analysis_gate(output)
    if (analysis/'COMPLETE.json').exists():
        return verify_analysis(output)
    chosen=shared.checked_selection(analysis)
    cfg=read(analysis/'config.json')
    completion={}
    metrics=[]
    for stage in ('final','null'):
        for i in range(cfg['repeats']):
            task={k:chosen[k] for k in ('candidate','loss','output_alpha')}
            task['repeat']=i
            folder=analysis/stage/f'{i:02d}'
            shared.verify_expected_task(analysis,folder,stage,task)
            completion[str((folder/'COMPLETE.json').relative_to(analysis))]=sha(folder/'COMPLETE.json')
            metrics.append(read(folder/'metrics.json'))
    (analysis/'summary').mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv(analysis/'summary/metrics.csv',index=False)
    files={str(p.relative_to(analysis)):sha(p) for folder in ('summary','selection')
           for p in (analysis/folder).glob('*') if p.is_file()}
    write(analysis/'COMPLETE.json',dict(status='complete',finished_utc=shared.now(),selection_fits=80,final_fits=5,null_fits=5,
        protocol_sha256=sha(analysis/'protocol.json'),selection_freeze_sha256=sha(analysis/'selection/freeze.json'),
        assessment_completions=completion,outputs=files))
    return verify_analysis(output)


def verify_analysis(output):
    analysis=analysis_gate(output)
    marker=read(analysis/'COMPLETE.json')
    if marker['status']!='complete' or (marker['selection_fits'],marker['final_fits'],marker['null_fits'])!=(80,5,5):
        raise ValueError('Incomplete analysis counts')
    for key,name in [('protocol_sha256','protocol.json'),('selection_freeze_sha256','selection/freeze.json')]:
        if marker[key]!=sha(analysis/name):
            raise ValueError('Analysis completion provenance changed')
    hashes(analysis,marker['outputs'])
    hashes(analysis,marker['assessment_completions'])
    cfg=read(analysis/'config.json')
    for task in shared.selection_tasks(cfg):
        shared.verify_expected_task(analysis,analysis/'selection/runs'/task['candidate']/f'{task["repeat"]:02d}','selection',task)
    chosen=shared.checked_selection(analysis)
    for stage in ('final','null'):
        for i in range(5):
            task={k:chosen[k] for k in ('candidate','loss','output_alpha')}
            task['repeat']=i
            shared.verify_expected_task(analysis,analysis/stage/f'{i:02d}',stage,task)
    return marker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare-generator','prepare-analysis','fit','select','seal-analysis','verify'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    parser.add_argument('--pilot',type=Path,default=DEFAULT_PILOT)
    parser.add_argument('--stage',choices=['selection','final','null'],default='selection')
    parser.add_argument('--task-index',type=int,default=0)
    args=parser.parse_args()
    if args.action=='prepare-generator':prepare_generator(args.output,args.parent,args.pilot)
    elif args.action=='prepare-analysis':prepare_analysis(args.output)
    elif args.action=='fit':shared.run_task(analysis_gate(args.output),args.stage,args.task_index,'cpu')
    elif args.action=='select':shared.freeze_selection(analysis_gate(args.output))
    elif args.action=='seal-analysis':seal_analysis(args.output)
    elif args.action=='verify':verify_analysis(args.output)


if __name__=='__main__':main()
