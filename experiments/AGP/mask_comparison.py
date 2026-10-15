#!/usr/bin/env python3
"""Frozen five-generator AGP comparison with ordinary source classifiers, no RDR."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

HYBRID = Path('/cwork/yx306/RDR/AGP/icfm_zero_20261007')
MIDASIM = Path('/cwork/yx306/RDR/AGP/midasim_full_20261004')
ROLES = {'train': 'train', 'stop': 'stopping', 'test': 'test'}
METHODS = ('bootstrap_icfm', 'bernoulli_mixture_icfm', 'autoregressive_icfm',
           'midasim_mask_icfm', 'midasim')
EXPECTED_P = {'train': 9162, 'stop': 1281, 'test': 2002}
EXPECTED_Q = {'train': 7310, 'stop': 1044, 'test': 2089}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def module(name, directory=None):
    path = (Path(directory) if directory else Path(__file__).parent) / (name + '.py')
    spec = importlib.util.spec_from_file_location('_agp_masks_' + name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def receipts(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts}


def check_files(root, expected):
    for name, digest in expected.items():
        p = Path(name) if Path(name).is_absolute() else Path(root) / name
        if not p.is_file() or sha(p) != digest:
            raise ValueError(f'Changed or missing pinned input: {p}')


def protocol(out):
    cfg = read(out / 'protocol.json')
    check_files(out / 'source', cfg['source_sha256'])
    check_files(out, cfg['inputs_sha256'])
    return cfg


def subjects(roles=ROLES):
    frame = pd.read_csv(HYBRID / 'analysis/data/final/manifest.csv', dtype={'subject_id': str})
    result = {}
    for role in roles:
        rows = frame[(frame.distribution == 'P') & (frame.role == ROLES[role])].sort_values('array_row')
        if not np.array_equal(rows.array_row.to_numpy(), np.arange(EXPECTED_P[role])):
            raise ValueError('Observed manifest order is not the pinned array order')
        result[role] = rows.subject_id.to_numpy(dtype=str)
    return result


def initialize(out, seed):
    if out.exists():
        raise FileExistsError('Use a fresh output directory; existing studies are immutable')
    models, metrics = module('mask_models'), module('mask_metrics')
    # Check the archive receipts underlying every reused generator input.
    gp = read(HYBRID / 'generator/protocol.json')
    gc = read(HYBRID / 'generator/COMPLETE.json')
    if gc['status'] != 'complete' or sha(HYBRID / 'generator/protocol.json') != gc['protocol_sha256']:
        raise ValueError('Original flow completion/protocol chain failed')
    if sha(HYBRID / 'generator/training/final.pt') != gc['checkpoint_sha256']:
        raise ValueError('Original conditional flow checkpoint receipt failed')
    check_files(HYBRID / 'generator', gp['data_receipts'])
    check_files(HYBRID / 'generator/source', gp['source_sha256'])
    check_files(HYBRID, gp['runtime_receipts'])
    src = HYBRID / 'generator/source/experiments/AGP'
    paths = [HYBRID / 'generator/training/final.pt', HYBRID / 'generator/protocol.json',
             HYBRID / 'generator/COMPLETE.json', HYBRID / 'generator/data/generator_train.csv',
             HYBRID / 'generator/data/manifest.csv', HYBRID / 'generator/data/taxa.csv',
             HYBRID / 'analysis/data/final/manifest.csv',
             MIDASIM / 'analysis/data/final/manifest.csv']
    for role, original in ROLES.items():
        paths.extend([HYBRID / f'generator/data/noise_final_{original}.npy',
                      HYBRID / f'generator/data/mask_final_{original}.npy'])
        for root in (HYBRID, MIDASIM):
            paths.extend(root / f'analysis/data/final/{original}_{pq}.npy' for pq in ('p', 'q'))
    paths.extend(p for p in (HYBRID / 'runtime').rglob('*') if p.is_file())
    ids = subjects()
    generator_ids = set(pd.read_csv(HYBRID / 'generator/data/manifest.csv').subject_id)
    for a, b in [('train', 'stop'), ('train', 'test'), ('stop', 'test')]:
        if set(ids[a]) & set(ids[b]):
            raise ValueError(f'Observed subject overlap: {a}/{b}')
    if generator_ids != set(ids['train']) | set(ids['stop']) or generator_ids & set(ids['test']):
        raise ValueError('Generator template does not match the allowed development subjects')
    for original in ROLES.values():
        a = np.load(HYBRID / f'analysis/data/final/{original}_p.npy')
        b = np.load(MIDASIM / f'analysis/data/final/{original}_p.npy')
        if not np.array_equal(a, b):
            raise ValueError('Reference generators do not share the observed cohort')
        mask = np.load(HYBRID / f'generator/data/mask_final_{original}.npy')
        for root in (HYBRID, MIDASIM):
            if not np.array_equal(mask, np.load(root / f'analysis/data/final/{original}_q.npy') > 0):
                raise ValueError('Original MIDASim/hybrid paired support changed')
    out.mkdir(parents=True)
    (out / 'source').mkdir()
    for name in ('mask_comparison.py', 'mask_models.py', 'mask_metrics.py', 'mask_comparison.slurm'):
        shutil.copy2(Path(__file__).parent / name, out / 'source' / name)
    for name in ('icfm_zero_generator.py', 'icfm_corrected_generator.py'):
        shutil.copy2(src / name, out / 'source' / name)
    cfg = dict(created_utc=now(), seed=seed, methods=list(METHODS), no_rdr=True,
               hybrid_root=str(HYBRID), midasim_root=str(MIDASIM),
               model_config=models.normalized_config({'seed': seed}),
               metric_config=metrics.default_config(),
               sampling=dict(atol=1e-4, rtol=1e-4, batch_size=128, ema=True),
               p_counts=EXPECTED_P, q_counts=EXPECTED_Q,
               p_subject_counts={k: len(set(v)) for k, v in ids.items()},
               generator_rows=10443, generator_subjects=len(generator_ids), taxa=614,
               selection='Mask architectures and stopping length selected by development NLL; refit on 10443 original generator rows. Classifiers selected by balanced development log loss; all freeze before test assessment.',
               scope='Retrospective existing audited cohort. Fixed conditional abundance flow; comparison isolates mask replacement, not independently optimized generators. One realization per method; conditional descriptive point estimates.',
               source_sha256=receipts(out / 'source'),
               inputs_sha256={str(p): sha(p) for p in paths})
    write(out / 'protocol.json', cfg)
    print(json.dumps(dict(status='initialized', output=str(out), protocol_sha256=sha(out / 'protocol.json'))), flush=True)


def load_real(roles):
    return {role: np.load(HYBRID / f'analysis/data/final/{ROLES[role]}_p.npy') for role in roles}


def load_generated(out, roles):
    result = {}
    for method in METHODS:
        result[method] = {}
        for role in roles:
            root = HYBRID if method == 'midasim_mask_icfm' else MIDASIM if method == 'midasim' else None
            path = (root / f'analysis/data/final/{ROLES[role]}_q.npy' if root
                    else out / 'generated' / method / f'{role}.npy')
            result[method][role] = np.load(path)
    return result


def fit_masks(out, device):
    cfg = protocol(out)
    dev = load_real(('train', 'stop'))
    full = np.loadtxt(HYBRID / 'generator/data/generator_train.csv', delimiter=',') > 0
    result = module('mask_models', out / 'source').fit_and_select(
        dev['train'] > 0, dev['stop'] > 0, full, out / 'models', cfg['model_config'], device=device)
    print(json.dumps(result), flush=True)


def generate(out, device):
    cfg = protocol(out)
    if (out / 'generation_complete.json').exists():
        raise FileExistsError('Generation is already complete')
    if not (out / 'models/COMPLETE.json').exists():
        raise RuntimeError('All mask models must be selected and refitted first')
    mask_module = module('mask_models', out / 'source')
    flow = module('icfm_zero_generator', out / 'source')
    model = flow.load_trained_model(HYBRID / 'generator/training/final.pt', device=device, ema=True)
    stage_start = time.monotonic()
    for j, method in enumerate(METHODS[:3]):
        destination = out / 'generated' / method
        destination.mkdir(parents=True, exist_ok=True)
        for i, role in enumerate(ROLES):
            receipt = destination / f'{role}.json'
            if receipt.exists():
                check_files(destination, read(receipt)['outputs_sha256'])
                continue
            noise_path = HYBRID / f'generator/data/noise_final_{ROLES[role]}.npy'
            noise = np.load(noise_path)
            seed = cfg['seed'] + 10000 * (j + 1) + 1000 * (i + 1)
            masks, mask_metrics = mask_module.sample_model(out / 'models', method.removesuffix('_icfm'),
                                                           len(noise), seed, device=device)
            latent, ode_metrics = flow.sample_model(model, noise, masks, device=device, return_metrics=True,
                                                    **{k: v for k, v in cfg['sampling'].items() if k != 'ema'})
            values = flow.decode_masked_clr(latent, masks)
            if values.shape != (EXPECTED_Q[role], 614) or not np.array_equal(values > 0, masks):
                raise ValueError('Generated shape or support differs from its declared mask')
            np.save(destination / f'{role}.npy', values)
            np.save(destination / f'{role}_mask.npy', masks)
            np.save(destination / f'{role}_latent.npy', latent)
            write(receipt, dict(method=method, role=role, seed=seed, mask=mask_metrics, ode=ode_metrics,
                                noise_path=str(noise_path), noise_sha256=sha(noise_path),
                                checkpoint_sha256=cfg['inputs_sha256'][str(HYBRID / 'generator/training/final.pt')],
                                outputs_sha256={f'{role}{suffix}.npy': sha(destination / f'{role}{suffix}.npy')
                                                for suffix in ('', '_mask', '_latent')}))
            print(f'Generated {method} {role}: {len(values)} rows', flush=True)
    write(out / 'generation_complete.json', dict(status='complete', finished_utc=now(),
          seconds=time.monotonic() - stage_start, models_receipt_sha256=sha(out / 'models/COMPLETE.json'),
          outputs_sha256=receipts(out / 'generated')))


def prepare(out):
    cfg = protocol(out)
    g = read(out / 'generation_complete.json')
    if g['models_receipt_sha256'] != sha(out / 'models/COMPLETE.json'):
        raise ValueError('Mask-model receipt changed after generation')
    check_files(out / 'generated', g['outputs_sha256'])
    metrics = module('mask_metrics', out / 'source')
    result = metrics.prepare(load_real(('train', 'stop')), load_generated(out, ('train', 'stop')),
                             out / 'metrics', config=cfg['metric_config'], real_subjects=subjects(('train', 'stop')))
    write(out / 'classifiers_frozen.json', dict(finished_utc=now(), metrics_receipt=result,
          files_sha256=receipts(out / 'metrics'), protocol_sha256=sha(out / 'protocol.json'),
          generation_complete_sha256=sha(out / 'generation_complete.json')))
    print('All ordinary source classifiers are frozen before test assessment', flush=True)


def evaluate(out):
    protocol(out)
    freeze = read(out / 'classifiers_frozen.json')
    if (freeze['protocol_sha256'] != sha(out / 'protocol.json') or
            freeze['generation_complete_sha256'] != sha(out / 'generation_complete.json')):
        raise ValueError('Protocol or generation receipt changed after classifier freeze')
    check_files(out / 'generated', read(out / 'generation_complete.json')['outputs_sha256'])
    check_files(out / 'metrics', freeze['files_sha256'])
    result = module('mask_metrics', out / 'source').evaluate(
        load_real(ROLES), load_generated(out, ROLES), out / 'metrics', real_subjects=subjects())
    write(out / 'assessment_complete.json', dict(status='complete', finished_utc=now(), result=result,
          freeze_sha256=sha(out / 'classifiers_frozen.json'), outputs_sha256=receipts(out / 'metrics')))
    print('Ordinary classifier and composition assessments complete', flush=True)


def export(out, destination):
    completion = out / ('COMPLETE_v2.json' if (out / 'COMPLETE_v2.json').exists() else 'COMPLETE.json')
    complete = read(completion)
    if complete['status'] != 'complete':
        raise RuntimeError('Export requires successful independent verification')
    check_files(out, complete['outputs_sha256'])
    if destination.exists():
        raise FileExistsError('Use a fresh public package destination')
    destination.mkdir(parents=True)
    for p in sorted((out / 'metrics/summary').iterdir()):
        if p.is_file():
            shutil.copy2(p, destination / p.name)
    for f in ('protocol.json', 'verification.json', 'COMPLETE.json'):
        shutil.copy2(out / f, destination / f)
    if completion.name != 'COMPLETE.json':
        shutil.copy2(completion, destination / completion.name)
    for f in ('candidate_validation.csv', 'bernoulli_mixture.json', 'autoregressive.json',
              'bernoulli_mixture_refit_history.csv', 'autoregressive_refit_history.csv'):
        if (out / 'models' / f).exists():
            shutil.copy2(out / 'models' / f, destination / ('mask_' + f))
    write(destination / 'paper_files.json', dict(archive=str(out), files_sha256=receipts(destination)))
    print(f'Exported verified aggregate package to {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('initialize', 'fit-masks', 'generate', 'prepare', 'evaluate', 'export'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--seed', type=int, default=2026101201)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.action == 'initialize':
        initialize(args.output, args.seed)
    elif args.action == 'fit-masks':
        fit_masks(args.output, args.device)
    elif args.action == 'generate':
        generate(args.output, args.device)
    elif args.action == 'prepare':
        prepare(args.output)
    elif args.action == 'evaluate':
        evaluate(args.output)
    elif args.action == 'export':
        if args.destination is None:
            parser.error('--destination is required for export')
        export(args.output, args.destination)


if __name__ == '__main__':
    main()
