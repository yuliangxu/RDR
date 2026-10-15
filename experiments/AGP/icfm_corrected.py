#!/usr/bin/env python3
"""Correct AGP ICFM inference and velocity capacity with matched RDR assessment."""
from __future__ import annotations
import argparse
import importlib.metadata
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import selected_workflow as shared
from midasim_full import parent_check, checked_parent_array, hashes, snapshot, receipt, row_hashes, ROLE_SIZES
import icfm_corrected_generator as generator

read, write, sha = shared.read_json, shared.write_json, shared.sha256
DEFAULT_OUTPUT = Path('/cwork/yx306/RDR/AGP/icfm_corrected_20261006')
DEFAULT_PARENT = Path('/cwork/yx306/RDR/AGP/fixed_width_selection_20261003')
DEFAULT_MIDASIM = Path('/cwork/yx306/RDR/AGP/midasim_full_20261004/analysis')
VARIANTS = ('eval_only', 'resmlp')


def prepare(output, parent=DEFAULT_PARENT, midasim=DEFAULT_MIDASIM):
    output, parent, midasim = [Path(p).resolve() for p in (output, parent, midasim)]
    gen = output/'generator'
    if gen.exists():
        raise ValueError('Preparation requires a fresh generator directory')
    data = parent_check(parent)
    parent_check(midasim)
    source = Path(data['data_root'])
    raw_path, latent_path = source/'yx1_xtrain_raw.csv', source/'yx1_abundance_lt_train.csv'
    if sha(raw_path) != data['input_sha256'][raw_path.name]:
        raise ValueError('Original generator-training observations changed')
    raw = np.loadtxt(raw_path, delimiter=',')
    latent = np.loadtxt(latent_path, delimiter=',')
    if raw.shape != (10443, 614) or latent.shape != (10443, 613) or not np.isfinite(latent).all():
        raise ValueError('Original training dimensions or coordinates differ')
    frames = []
    for role in ('train', 'stopping'):
        x, frame, _ = checked_parent_array(parent, data, 'final', role+'_p')
        if not np.array_equal(raw[frame.source_row.to_numpy(int)].astype(np.float32), x):
            raise ValueError('Generator and main development samples differ')
        frames.append(frame)
    manifest = pd.concat(frames).sort_values('source_row').reset_index(drop=True)
    _, test, _ = checked_parent_array(parent, data, 'final', 'test_p')
    if (manifest.source_row.tolist() != list(range(10443)) or manifest.sample_id.duplicated().any()
            or set(manifest.subject_id) & set(test.subject_id)):
        raise ValueError('Generator identities or assessment separation changed')
    cfg = read(HERE/'icfm_corrected_config.json')
    if cfg['roles'] != ROLE_SIZES or cfg['variants'] != list(VARIANTS):
        raise ValueError('Unexpected paired generation design')
    if sha(parent/data['taxa']['path']) != data['taxa']['sha256']:
        raise ValueError('Parent taxonomy differs from its frozen receipt')
    (gen/'data').mkdir(parents=True)
    (gen/'vendor').mkdir()
    (gen/'logs').mkdir()
    inputs = {
        'data/generator_train.csv': raw_path,
        'data/generator_train_latent.csv': latent_path,
        'data/tree_children.csv': source/'yx3_chid_node_df.csv',
        'data/tree_taxa.csv': source/'yx4_taxa_name_df.csv',
        'data/abundance.csv': source/'yx1_abundance.csv',
        'data/taxa.csv': parent/data['taxa']['path'],
        'vendor/icfm_checkpoint.pth': source/'icfm_checkpoint.pth',
        'vendor/microbiome_unet.py': Path('/hpc/group/mastatlab/microbiome/clean/microbiome_unet.py'),
    }
    original = {}
    for relative, path in inputs.items():
        shutil.copyfile(path, gen/relative)
        original[relative] = dict(source=str(path), sha256=sha(path))
        if sha(gen/relative) != original[relative]['sha256']:
            raise ValueError('Input copy changed')
    manifest.to_csv(gen/'data/manifest.csv', index=False)
    noise_receipts = {}
    for index, role in enumerate(cfg['role_order']):
        seed = cfg['noise_seed'] + cfg['noise_seed_stride']*index
        noise = np.random.default_rng(seed).standard_normal((ROLE_SIZES[role],613)).astype(np.float32)
        path = gen/'data'/('noise_'+role+'.npy')
        np.save(path, noise, allow_pickle=False)
        noise_receipts[role] = dict(receipt(gen,path), seed=seed, shape=list(noise.shape),
                                   dtype='float32', algorithm='numpy.default_rng(PCG64).standard_normal(float64)->float32')
    provenance = dict(parent_run=str(parent), parent_completion_sha256=sha(parent/'COMPLETE.json'),
        parent_data_protocol_sha256=sha(parent/'data/protocol.json'), midasim_run=str(midasim),
        midasim_completion_sha256=sha(midasim/'COMPLETE.json'), original_inputs=original,
        original_source=str(raw_path), original_source_sha256=sha(raw_path), rows=10443,
        subjects=int(manifest.subject_id.nunique()), taxa=614, latent_dimension=613,
        noise=noise_receipts, zero_handling=cfg['zero_handling'],
        cohort='Audited P test=2002; all609 original subject-overlap exclusions retained.',
        generator_overlap='All original real development observations train the generator, including RDR stopping/selection roles.',
        repeat_scope='One generator checkpoint per variant and fixed generated banks; five RDR initialization repeats.',
        limitations=['Historical cohort already inspected: retrospective assessment.',
                    'New variants share paired Gaussian noise; original archived ICFM and MIDASim banks are unpaired.',
                    'Explicit eval mode is verified in this run; actual historical interactive runtime mode was not recorded.',
                    'Original zero smoothing/inversion remains unchanged and does not explicitly model boundary mass.',
                    'Seven independent Q roles are drawn; archived ICFM reused its Q development bank across stages.'])
    write(gen/'data/protocol.json', provenance)
    write(gen/'config.json', cfg)
    paths = [HERE/name for name in ('icfm_corrected.py','icfm_corrected_generator.py','icfm_corrected.slurm',
             'icfm_corrected_config.json','midasim_full.py','selected_workflow.py')]
    paths += [ROOT/'utils'/name for name in ('__init__.py','losses.py','model_selection.py','calibration.py')]
    frozen = snapshot(gen, paths)
    receipts = {str(p.relative_to(gen)):sha(p) for directory in ('data','vendor') for p in (gen/directory).iterdir() if p.is_file()}
    runtime = output/'runtime'
    runtime_receipts = {str(p.relative_to(output)):sha(p) for p in runtime.rglob('*')
                        if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    write(gen/'protocol.json', dict(created_utc=shared.now(), config_sha256=sha(gen/'config.json'),
        source_sha256=frozen, data_receipts=receipts, runtime_receipts=runtime_receipts))
    print('Prepared corrected ICFM generator: '+str(gen), flush=True)


def generator_gate(output):
    output = Path(output).resolve()
    gen = output/'generator'
    protocol = read(gen/'protocol.json')
    if protocol['config_sha256'] != sha(gen/'config.json'):
        raise ValueError('Generator config changed')
    hashes(gen, protocol['data_receipts'])
    hashes(gen/'source', protocol['source_sha256'])
    if protocol.get('runtime_receipts'):
        hashes(output, protocol['runtime_receipts'])
    for name, path in [('icfm_corrected.py',__file__), ('icfm_corrected_generator.py',generator.__file__)]:
        if sha(path) != protocol['source_sha256']['experiments/AGP/'+name]:
            raise ValueError('Executing generator source differs: '+name)
    return gen


def train(output):
    gen = generator_gate(output)
    return generator.train(read(gen/'config.json'), gen/'data/generator_train_latent.csv', gen/'training/resmlp')


def checkpoint_for(gen, variant):
    return gen/('vendor/icfm_checkpoint.pth' if variant == 'eval_only' else 'training/resmlp/final.pt')


def generate(output, variant):
    gen = generator_gate(output)
    cfg = read(gen/'config.json')
    folder = gen/'generation'/variant
    if (folder/'COMPLETE.json').exists():
        return verify_generation(output, variant)
    checkpoint = checkpoint_for(gen, variant)
    if variant == 'resmlp':
        trained = read(gen/'training/resmlp/metrics.json')
        if trained['status'] != 'complete' or trained['final_checkpoint_sha256'] != sha(checkpoint):
            raise ValueError('New velocity training is incomplete')
    model = (generator.load_legacy_model(checkpoint, gen/'vendor/microbiome_unet.py', cfg['sampling']['device'])
             if variant == 'eval_only' else generator.load_trained_model(checkpoint, cfg['sampling']['device']))
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    roles = {}
    for role in cfg['role_order']:
        metrics_path = folder/(role+'_metrics.json')
        if metrics_path.exists():
            detail = read(metrics_path)
            hashes(folder, detail['outputs'])
            if detail['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('Partial generation used another checkpoint')
            roles[role] = detail
            continue
        if (folder/(role+'.npy')).exists() or (folder/(role+'_latent.npy')).exists():
            raise ValueError('Preserve and inspect incomplete role files: '+role)
        noise_path = gen/'data'/('noise_'+role+'.npy')
        noise = np.load(noise_path, allow_pickle=False)
        latent, sampling = generator.sample_model(model, noise, **{k:cfg['sampling'][k]
                for k in ('atol','rtol','batch_size','device')}, return_metrics=True)
        raw = generator.invert_tree(latent, gen/'data/tree_children.csv', gen/'data/tree_taxa.csv', gen/'data/abundance.csv')
        if (raw.shape != (ROLE_SIZES[role],614) or not np.isfinite(raw).all()
                or (raw < 0).any() or not np.allclose(raw.sum(1),1,atol=1e-10,rtol=0)):
            raise ValueError('Invalid generated compositions')
        np.save(folder/(role+'_latent.npy'), latent.astype(np.float32), allow_pickle=False)
        np.save(folder/(role+'.npy'), raw.astype(np.float64), allow_pickle=False)
        detail = dict(role=role, variant=variant, rows=len(raw), taxa=614, sampling=sampling,
            noise_sha256=sha(noise_path), checkpoint_sha256=sha(checkpoint),
            zero_fraction=float((raw==0).mean()), mean_positive_taxa=float((raw>0).sum(1).mean()),
            maximum_closure_error=float(np.abs(raw.sum(1)-1).max()),
            outputs={role+suffix:sha(folder/(role+suffix)) for suffix in ('.npy','_latent.npy')})
        write(metrics_path, detail)
        roles[role] = detail
        print(f'{variant} {role}: {len(raw)} rows; positive taxa={detail["mean_positive_taxa"]:.3f}', flush=True)
    versions = {}
    for package in ('torch','torchdyn','torchdiffeq','numpy','scipy','pandas','attrs'):
        versions[package] = importlib.metadata.version(package)
    metrics = dict(status='complete', variant=variant, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                   parameter_count=sum(p.numel() for p in model.parameters()), roles=roles, versions=versions,
                   elapsed_seconds=time.monotonic()-started, zero_restoration=False,
                   paired_noise=True, explicit_evaluation_mode=True, solver='torchdyn.dopri5')
    write(folder/'metrics.json', metrics)
    outputs = {p.name:sha(p) for p in folder.iterdir() if p.is_file()}
    write(folder/'COMPLETE.json', dict(status='complete', variant=variant, finished_utc=shared.now(),
        config_sha256=sha(gen/'config.json'), protocol_sha256=sha(gen/'protocol.json'),
        checkpoint_sha256=sha(checkpoint), outputs=outputs))
    return verify_generation(output, variant)


def verify_generation(output, variant):
    gen = generator_gate(output)
    folder = gen/'generation'/variant
    marker = read(folder/'COMPLETE.json')
    if marker['status'] != 'complete' or marker['variant'] != variant:
        raise ValueError('Incomplete or wrong generator variant')
    for key, path in [('config_sha256',gen/'config.json'),('protocol_sha256',gen/'protocol.json'),
                      ('checkpoint_sha256',checkpoint_for(gen,variant))]:
        if marker[key] != sha(path):
            raise ValueError('Generation provenance changed')
    required = {role+suffix for role in ROLE_SIZES for suffix in ('.npy','_latent.npy','_metrics.json')} | {'metrics.json'}
    if set(marker['outputs']) != required:
        raise ValueError('Incomplete generated role inventory')
    hashes(folder,marker['outputs'])
    return marker


def prepare_analysis(output, variant):
    output = Path(output).resolve()
    verify_generation(output,variant)
    gen = output/'generator'
    template = read(gen/'data/protocol.json')
    parent, midasim = Path(template['parent_run']), Path(template['midasim_run'])
    data = parent_check(parent)
    if sha(parent/'COMPLETE.json') != template['parent_completion_sha256'] or sha(midasim/'COMPLETE.json') != template['midasim_completion_sha256']:
        raise ValueError('Reference studies changed')
    analysis = output/'analysis'/variant
    if analysis.exists():
        raise ValueError('Analysis preparation needs a fresh destination')
    cfg = read(parent/'config.json')
    cfg.update(generator='ICFM_'+variant, generator_training_rows=10443,
        selection_scope='Independent loss/slope selection for corrected ICFM '+variant,
        interval_scope='Fixed20 descriptive diagnostics; historical cohort/subject dependence limits coverage claims.')
    shared.validate_config(cfg)
    analysis.mkdir(parents=True)
    write(analysis/'config.json',cfg)
    (analysis/'data').mkdir()
    shutil.copyfile(gen/'data/taxa.csv',analysis/'data/taxa.csv')
    studies = {}
    for stage in ('selection','final','null'):
        folder = analysis/'data'/stage
        folder.mkdir()
        roles = data['studies'][stage]['roles']
        frames, arrays, counts, role_counts = [], {}, {}, {}
        for role in roles:
            role_counts[role] = {}
            for side in ('p','q'):
                key = role+'_'+side
                target = folder/(key+'.npy')
                if side == 'p' or stage == 'null':
                    x,frame,source = checked_parent_array(parent,data,stage,key)
                    shutil.copyfile(source,target)
                else:
                    label = stage+'_'+role
                    source = gen/'generation'/variant/(label+'.npy')
                    x64 = np.load(source,allow_pickle=False)
                    if (x64.shape != (ROLE_SIZES[label],614) or not np.isfinite(x64).all()
                            or (x64 < 0).any() or not np.allclose(x64.sum(1),1,atol=1e-10,rtol=0)):
                        raise ValueError('Invalid generated compositions at analysis import')
                    x = x64.astype(np.float32)
                    np.save(target,x,allow_pickle=False)
                    frame = pd.DataFrame(dict(distribution='Q',role=role,source_file=str(source),
                        source_row=np.arange(len(x)),sample_id=[f'ICFM:{variant}:{stage}:{role}:{i:06d}' for i in range(len(x))],
                        subject_id='',row_sha256=row_hashes(x),exclusion_reason='',previous_role=role,
                        original_role=role,original_distribution='Q',array_row=np.arange(len(x))))
                frames.append(frame)
                arrays[key] = dict(receipt(analysis,target),shape=list(x.shape),dtype=str(x.dtype))
                counts[key] = dict(rows=len(x),subjects=int(frame.subject_id.nunique()) if (side=='p' or stage=='null') else None)
                role_counts[role][side] = len(x)
        manifest = pd.concat(frames,ignore_index=True)
        if manifest.sample_id.duplicated().any():
            raise ValueError('Duplicate sample IDs')
        real = manifest[manifest.original_distribution.eq('P')]
        if (real.role+':'+real.distribution).groupby(real.subject_id).nunique().max()!=1:
            raise ValueError('Subjects cross analysis roles')
        manifest.to_csv(folder/'manifest.csv',index=False)
        write(folder/'audit.json',dict(real_roles_identical_to_main_icfm=True,subject_disjoint_rdr_roles=True,
             q_roles_independently_generated=stage!='null',row_hash_dtype='little-endian float32',counts=counts,
             no_test_scores_used_for_selection=True,paired_noise_across_new_generators=True))
        studies[stage] = dict(roles=roles,arrays=arrays,counts=counts,role_counts=role_counts,
            manifest=receipt(analysis,folder/'manifest.csv'),audit=receipt(analysis,folder/'audit.json'))
    write(analysis/'data/protocol.json',dict(parent_run=str(parent),parent_completion_sha256=sha(parent/'COMPLETE.json'),
        generator_run=str(gen),generator_variant=variant,generator_protocol_sha256=sha(gen/'protocol.json'),
        generator_completion_sha256=sha(gen/'generation'/variant/'COMPLETE.json'),
        midasim_run=str(midasim),midasim_completion_sha256=sha(midasim/'COMPLETE.json'),
        feature_dimension=614,studies=studies,taxa=receipt(analysis,analysis/'data/taxa.csv'),
        original_excluded_real_test_rows=609,limitations=template['limitations'],
        generator_overlap=template['generator_overlap'],repeat_scope=template['repeat_scope']))
    paths = [HERE/name for name in (*shared.RDR_SOURCE_FILES, 'icfm_corrected.py',
        'icfm_corrected_generator.py', 'midasim_full.py', 'verify_midasim_full.py')]
    paths += [HERE/'icfm_corrected.slurm', HERE/'icfm_corrected_config.json']
    paths += [ROOT/'utils'/name for name in ('__init__.py','losses.py','model_selection.py','calibration.py')]
    frozen = snapshot(analysis,paths)
    (analysis/'logs').mkdir()
    (analysis/'selection').mkdir()
    write(analysis/'protocol.json',dict(created_utc=shared.now(),config_sha256=sha(analysis/'config.json'),
        data_protocol_sha256=sha(analysis/'data/protocol.json'),source_sha256=frozen,
        expected_selection_fits=80,expected_final_fits=5,expected_null_fits=5,roles=studies,
        generator_completion_sha256=sha(gen/'generation'/variant/'COMPLETE.json')))
    print('Prepared corrected ICFM analysis: '+str(analysis),flush=True)


def analysis_gate(output,variant):
    analysis = Path(output).resolve()/'analysis'/variant
    shared.verify_snapshot(analysis)
    protocol = read(analysis/'protocol.json')
    if sha(__file__) != protocol['source_sha256']['experiments/AGP/icfm_corrected.py']:
        raise ValueError('Executing wrapper differs from analysis snapshot')
    data = read(analysis/'data/protocol.json')
    receipts = [data['taxa']]
    for study in data['studies'].values():
        receipts += [study['manifest'],study['audit'],*study['arrays'].values()]
    hashes(analysis,{item['path']:item['sha256'] for item in receipts})
    gen = Path(output).resolve()/'generator'
    if (data['generator_protocol_sha256'] != sha(gen/'protocol.json') or
        data['generator_completion_sha256'] != sha(gen/'generation'/variant/'COMPLETE.json')):
        raise ValueError('Analysis generator inputs changed')
    verify_generation(output,variant)
    return analysis


def seal_analysis(output,variant):
    analysis = analysis_gate(output,variant)
    if (analysis/'COMPLETE.json').exists():
        return verify_analysis(output,variant)
    chosen = shared.checked_selection(analysis)
    completion,metrics = {},[]
    for stage in ('final','null'):
        for repeat in range(5):
            task = {k:chosen[k] for k in ('candidate','loss','output_alpha')}
            task['repeat'] = repeat
            folder = analysis/stage/f'{repeat:02d}'
            shared.verify_expected_task(analysis,folder,stage,task)
            completion[str((folder/'COMPLETE.json').relative_to(analysis))] = sha(folder/'COMPLETE.json')
            metrics.append(read(folder/'metrics.json'))
    (analysis/'summary').mkdir(exist_ok=True)
    pd.DataFrame(metrics).to_csv(analysis/'summary/metrics.csv',index=False)
    files = {str(p.relative_to(analysis)):sha(p) for folder in ('summary','selection')
             for p in (analysis/folder).glob('*') if p.is_file()}
    write(analysis/'COMPLETE.json',dict(status='complete',finished_utc=shared.now(),selection_fits=80,final_fits=5,null_fits=5,
        protocol_sha256=sha(analysis/'protocol.json'),selection_freeze_sha256=sha(analysis/'selection/freeze.json'),
        assessment_completions=completion,outputs=files))
    return verify_analysis(output,variant)


def verify_analysis(output,variant):
    analysis = analysis_gate(output,variant)
    marker = read(analysis/'COMPLETE.json')
    if marker['status']!='complete' or (marker['selection_fits'],marker['final_fits'],marker['null_fits'])!=(80,5,5):
        raise ValueError('Analysis completion counts differ')
    for key,name in [('protocol_sha256','protocol.json'),('selection_freeze_sha256','selection/freeze.json')]:
        if marker[key]!=sha(analysis/name):
            raise ValueError('Analysis completion provenance differs')
    hashes(analysis,marker['outputs'])
    hashes(analysis,marker['assessment_completions'])
    for task in shared.selection_tasks(read(analysis/'config.json')):
        shared.verify_expected_task(analysis,analysis/'selection/runs'/task['candidate']/f'{task["repeat"]:02d}','selection',task)
    chosen = shared.checked_selection(analysis)
    for stage in ('final','null'):
        for repeat in range(5):
            task = {k:chosen[k] for k in ('candidate','loss','output_alpha')}
            task['repeat'] = repeat
            shared.verify_expected_task(analysis,analysis/stage/f'{repeat:02d}',stage,task)
    return marker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','benchmark','train','generate','prepare-analysis','fit','select','seal-analysis','verify-generation','verify'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--parent',type=Path,default=DEFAULT_PARENT)
    parser.add_argument('--midasim',type=Path,default=DEFAULT_MIDASIM)
    parser.add_argument('--variant',choices=VARIANTS,default='resmlp')
    parser.add_argument('--stage',choices=['selection','final','null'],default='selection')
    parser.add_argument('--task-index',type=int,default=0)
    args = parser.parse_args()
    if args.action=='prepare': prepare(args.output,args.parent,args.midasim)
    elif args.action=='train': train(args.output)
    elif args.action=='benchmark':
        gen=generator_gate(args.output)
        generator.benchmark(read(gen/'config.json'),gen/'data/generator_train_latent.csv',args.output/'benchmark.json')
    elif args.action=='generate': generate(args.output,args.variant)
    elif args.action=='prepare-analysis': prepare_analysis(args.output,args.variant)
    elif args.action=='fit': shared.run_task(analysis_gate(args.output,args.variant),args.stage,args.task_index,'cpu')
    elif args.action=='select': shared.freeze_selection(analysis_gate(args.output,args.variant))
    elif args.action=='seal-analysis': seal_analysis(args.output,args.variant)
    elif args.action=='verify-generation': verify_generation(args.output,args.variant)
    elif args.action=='verify': verify_analysis(args.output,args.variant)


if __name__=='__main__': main()
