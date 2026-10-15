#!/usr/bin/env python3
"""Independently verify the three added AGP mask-generator RDR studies.

Audits every completion gate, data identity, frozen Brier/local-gap selection,
checkpoint restoration and twenty-cell diagnostic. Existing hybrid and MIDASim
studies are checked by their immutable ninety-fit receipts, not refitted.
Only a fresh successful verification receipt is written; failures write none.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for _variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_variable, '2')

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

METHODS = ('bootstrap_icfm', 'bernoulli_mixture_icfm', 'autoregressive_icfm')
ROLE_SIZES = {
    'selection': dict(train_p=7310, train_q=5847, stopping_p=1044, stopping_q=836,
                      calibration_p=1045, calibration_q=835, evaluation_p=1044, evaluation_q=836),
    'final': dict(train_p=9162, train_q=7310, stopping_p=1281, stopping_q=1044, test_p=2002, test_q=2089),
    'null': dict(train_p=4581, train_q=4581, stopping_p=640, stopping_q=641, test_p=1001, test_q=1001),
}
IDENTITY_FIELDS = ('sample_id', 'subject_id', 'source_file', 'source_row', 'row_sha256',
                   'original_role', 'original_distribution')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def hashes(root, expected, *, absolute=False):
    require(bool(expected), f'Empty receipt inventory: {root}')
    for name, digest in expected.items():
        relative = Path(name)
        require('..' not in relative.parts and (absolute or not relative.is_absolute()),
                f'Invalid receipt path: {name}')
        path = relative if relative.is_absolute() else Path(root) / relative
        require(path.is_file() and sha(path) == digest, f'Pinned artifact changed: {path}')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def rows(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def subset(frame, role, side):
    return frame[frame.role.eq(role) & frame.distribution.eq(side.upper())].sort_values(
        'array_row', key=lambda values: values.astype(int)).reset_index(drop=True)


def selection_tasks(config):
    return [dict(candidate=f'{loss}_a{alpha:g}', loss=loss, output_alpha=float(alpha), repeat=repeat)
            for loss in config['losses'] for alpha in config['output_alphas']
            for repeat in range(config['repeats'])]


def analysis_gate(analysis, *, expected_selection=80, expected_repeats=5):
    """Receipt-only checks, deliberately independent of selected_workflow gates."""
    analysis = Path(analysis)
    complete, protocol = read(analysis/'COMPLETE.json'), read(analysis/'protocol.json')
    config, data = read(analysis/'config.json'), read(analysis/'data/protocol.json')
    require(complete.get('status') == 'complete', f'Incomplete analysis: {analysis}')
    require(complete['protocol_sha256'] == sha(analysis/'protocol.json'), 'Analysis protocol changed')
    require(protocol['config_sha256'] == sha(analysis/'config.json') and
            protocol['data_protocol_sha256'] == sha(analysis/'data/protocol.json'), 'Analysis preparation changed')
    require((complete['selection_fits'], complete['final_fits'], complete['null_fits']) ==
            (expected_selection, expected_repeats, expected_repeats), 'Analysis fit counts differ')
    require((protocol['expected_selection_fits'], protocol['expected_final_fits'], protocol['expected_null_fits']) ==
            (expected_selection, expected_repeats, expected_repeats), 'Prespecified fit counts differ')
    hashes(analysis/'source', protocol['source_sha256'])
    hashes(analysis, complete['outputs'])
    freeze, decision = read(analysis/'selection/freeze.json'), read(analysis/'selection/selection.json')
    require(freeze.get('status') == 'frozen' and freeze.get('test_scores_read') is False and
            freeze['completed_fits'] == freeze['expected_fits'] == expected_selection,
            'Selection was not fully frozen without test scores')
    require(complete['selection_freeze_sha256'] == sha(analysis/'selection/freeze.json'), 'Selection freeze changed')
    hashes(analysis, freeze['outputs'])
    tasks = selection_tasks(config)
    require(len(tasks) == expected_selection and config['repeats'] == expected_repeats, 'Selection grid count differs')
    chosen = decision['chosen']
    require(set(chosen) == {'candidate', 'loss', 'output_alpha'} and
            chosen in [{key: task[key] for key in chosen} for task in tasks], 'Chosen configuration outside grid')
    work = [('selection', task, analysis/'selection/runs'/task['candidate']/f"{task['repeat']:02d}") for task in tasks]
    work += [(stage, {**chosen, 'repeat': repeat}, analysis/stage/f'{repeat:02d}')
             for stage in ('final', 'null') for repeat in range(expected_repeats)]
    selection_receipts = {str((folder/'COMPLETE.json').relative_to(analysis)) for stage, _, folder in work if stage == 'selection'}
    assessment_receipts = {str((folder/'COMPLETE.json').relative_to(analysis)) for stage, _, folder in work if stage != 'selection'}
    require(set(freeze['selection_completions']) == selection_receipts and
            set(complete['assessment_completions']) == assessment_receipts, 'Fit completion receipt inventory differs')
    hashes(analysis, freeze['selection_completions'])
    hashes(analysis, complete['assessment_completions'])
    actual = set(analysis.glob('selection/runs/*/*/COMPLETE.json')) | set(analysis.glob('final/*/COMPLETE.json')) | set(analysis.glob('null/*/COMPLETE.json'))
    require(actual == {folder/'COMPLETE.json' for _, _, folder in work}, 'Extra or absent completed fits')
    require(not list(analysis.rglob('FAILED.json')), 'Failure marker remains')
    required_outputs = {'model.pt', 'transform.npz', 'history.csv', 'metrics.json', 'cells.csv', 'predictions.npz', 'scores.csv'}
    for stage, task, folder in work:
        marker = read(folder/'COMPLETE.json')
        require(marker.get('status') == 'complete' and marker['stage'] == stage and marker['task'] == task,
                'Task completion identity differs')
        require(marker['config_sha256'] == sha(analysis/'config.json') and
                marker['protocol_sha256'] == sha(analysis/'protocol.json'), 'Task configuration/protocol differs')
        require(required_outputs <= set(marker['outputs']), 'Task scientific outputs incomplete')
        hashes(folder, marker['outputs'])
        if stage != 'selection':
            require(marker['selection_freeze_sha256'] == sha(analysis/'selection/freeze.json'),
                    'Assessment checkpoint belongs to another selection lock')
    receipts = [data['taxa']]
    for study in data['studies'].values():
        receipts.extend([study['manifest'], study['audit'], *study['arrays'].values()])
    hashes(analysis, {entry['path']: entry['sha256'] for entry in receipts})
    return config, data, chosen, work, dict(output=str(analysis), status='unchanged',
        completion_sha256=sha(analysis/'COMPLETE.json'), selection_freeze_sha256=sha(analysis/'selection/freeze.json'),
        checkpoint_tasks_hashed=len(work))


def real_subject_audit(frame):
    real = frame[frame.original_distribution.eq('P')]
    require(real.subject_id.ne('').all(), 'Missing real subject identity')
    require((real.role+':'+real.distribution).groupby(real.subject_id).nunique().eq(1).all(),
            'A real subject crosses RDR roles/distributions')


def data_audit(analysis, data, method, parent, parent_data, comparison, generated_root,
               *, role_sizes=None, feature_dimension=614):
    """Verify observed identities and actual Q bytes, never only count summaries."""
    role_sizes = ROLE_SIZES if role_sizes is None else role_sizes
    require(set(data['studies']) == set(role_sizes), 'RDR stage inventory differs')
    raw, manifests, counts, source_receipts = {}, {}, {}, {}
    for stage, expected in role_sizes.items():
        entry = data['studies'][stage]
        require(set(entry['arrays']) == set(expected), 'Cached role inventory differs')
        frame = rows(analysis/entry['manifest']['path'])
        reference = rows(parent/parent_data['studies'][stage]['manifest']['path'])
        require(not frame.sample_id.duplicated().any(), 'Duplicate sample IDs')
        require(not frame.duplicated(['source_file', 'source_row']).any(), 'Duplicate source rows')
        real_subject_audit(frame)
        raw[stage], counts[stage] = {}, {}
        for key, expected_n in expected.items():
            role, side = key.rsplit('_', 1)
            receipt = entry['arrays'][key]
            values = np.load(analysis/receipt['path'], allow_pickle=False)
            group = subset(frame, role, side)
            require(values.shape == (expected_n, feature_dimension) and values.dtype == np.float32,
                    f'Cache dimensions/dtype differ: {method}/{stage}/{key}')
            require(np.isfinite(values).all() and (values >= 0).all() and (values.sum(1) > 0).all(), 'Invalid cached compositions')
            require(group.array_row.astype(int).tolist() == list(range(expected_n)), 'Array row identities differ')
            require(group.row_sha256.tolist() == [hashlib.sha256(row.astype('<f4').tobytes()).hexdigest() for row in values],
                    'Cached row bytes do not match manifest')
            if side == 'p' or stage == 'null':
                original = parent_data['studies'][stage]['arrays'][key]
                require(sha(analysis/receipt['path']) == original['sha256'], 'Original P/null cache changed')
                old = subset(reference, role, side)
                for field in IDENTITY_FIELDS:
                    require(group[field].tolist() == old[field].tolist(), f'Observed identity changed: {field}')
            else:
                require(group.subject_id.eq('').all() and group.original_distribution.eq('Q').all(), 'Generated source labels/subjects differ')
                require(group.source_file.nunique() == 1, 'A generated role has multiple source arrays')
                source = Path(group.source_file.iloc[0]).resolve()
                if stage == 'final':
                    original_role = {'train': 'train', 'stopping': 'stop', 'test': 'test'}[role]
                    expected_source = (comparison/'generated'/method/f'{original_role}.npy').resolve()
                    require(source == expected_source, 'Final Q is not the original mask-comparison sample')
                else:
                    require((generated_root/method).resolve() in source.parents,
                            'Selection Q is not a fresh method-specific generated role')
                original = np.load(source, allow_pickle=False)
                require(group.source_row.astype(int).tolist() == list(range(expected_n)), 'Generated source order differs')
                require(np.array_equal(values, original.astype(np.float32)) and np.array_equal(values > 0, original > 0),
                        'Generated import/support differs from source')
                require(np.allclose(original.sum(1), 1., atol=2e-12, rtol=0), 'Generated compositions are not closed')
                source_receipts[str(source)] = sha(source)
            require(entry['counts'][key]['rows'] == entry['role_counts'][role][side] == expected_n,
                    'Declared role counts differ')
            subjects = int(group.loc[group.subject_id.ne(''), 'subject_id'].nunique())
            if side == 'p' or stage == 'null':
                require(entry['counts'][key]['subjects'] == subjects, 'Declared real subject count differs')
            raw[stage][key] = values
            counts[stage][key] = dict(rows=expected_n, subjects=subjects)
        manifests[stage] = frame
    if 'selection' in manifests and 'final' in manifests:
        test = manifests['final'].loc[manifests['final'].role.eq('test')]
        require(not set(manifests['selection'].sample_id) & set(test.sample_id), 'Selection includes final test rows')
        real_test = set(test.loc[test.original_distribution.eq('P'), 'subject_id'])
        require(not real_test & set(manifests['selection'].loc[manifests['selection'].original_distribution.eq('P'), 'subject_id']),
                'Selection includes final real test subjects')
    return raw, manifests, counts, source_receipts



def independent_decode(latent, masks):
    require(latent.shape == masks.shape and masks.dtype == np.bool_ and masks.any(1).all(),
            'Invalid latent/mask dimensions or empty support')
    require(np.isfinite(latent).all() and not latent[~masks].any(), 'Latent values leave mask support')
    values = np.where(masks, latent.astype(np.float64), -np.inf)
    weights = np.exp(values-values.max(1, keepdims=True))
    return weights/weights.sum(1, keepdims=True)


def replay_masks(model_directory, method, n, seed, config):
    """Independent CPU replay for the two NumPy mask samplers."""
    rng = np.random.default_rng(seed)
    metadata = read(model_directory/(method+'.json'))
    schema = metadata['schema']
    with np.load(model_directory/(method+'.npz'), allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    accepted, proposals, rejected, remaining = [], 0, 0, n
    while remaining:
        count = min(remaining, config['batch_size'], n*config['max_draw_multiplier']-proposals)
        require(count > 0, 'Independent mask replay exceeded rejection limit')
        if method == 'bootstrap':
            batch = arrays['masks'][rng.integers(len(arrays['masks']), size=count)]
        else:
            components = rng.choice(len(arrays['weights']), count, p=arrays['weights'])
            active = rng.random((count, arrays['probabilities'].shape[1])) < arrays['probabilities'][components]
            batch = np.zeros((count, schema['dimension']), dtype=bool)
            batch[:, schema['active']] = active
            batch[:, schema['always_one']] = True
        good = batch.any(1)
        accepted.append(batch[good])
        proposals += count
        rejected += int((~good).sum())
        remaining -= int(good.sum())
    return np.concatenate(accepted), proposals, rejected


def generation_audit(output, protocol):
    comparison = Path(protocol['comparison_root'])
    hybrid = Path(protocol['hybrid_root'])
    models = read(comparison/'models/COMPLETE.json')
    marker = read(output/'generation_complete.json')
    require(marker.get('status') == models.get('status') == 'complete', 'Generation or fitted mask model incomplete')
    hashes(comparison/'models', models['artifacts_sha256'])
    require(marker['models_receipt_sha256'] == sha(comparison/'models/COMPLETE.json'), 'Mask model completion changed')
    comparison_complete = comparison/('COMPLETE_v2.json' if (comparison/'COMPLETE_v2.json').exists() else 'COMPLETE.json')
    require(marker['comparison_completion_sha256'] == sha(comparison_complete), 'Original mask comparison completion changed')
    hashes(comparison, read(comparison_complete)['outputs_sha256'])
    roles = ('train', 'stopping', 'calibration', 'evaluation')
    require(tuple(protocol['selection_role_order']) == roles and protocol['mask_seed_base'] == 2027101201,
            'Independent selection-draw protocol changed')
    expected = {f'{method}/selection_{role}{suffix}' for method in METHODS for role in roles
                for suffix in ('.json', '.npy', '_mask.npy', '_latent.npy')}
    require(set(marker['outputs_sha256']) == expected, 'Selection generation inventory differs')
    hashes(output/'generated', marker['outputs_sha256'])
    checkpoint_sha = sha(hybrid/'generator/training/final.pt')
    verified = []
    for mi, method in enumerate(METHODS):
        mask_method = method.removesuffix('_icfm')
        model_file = mask_method+('.pt' if mask_method == 'autoregressive' else '.npz')
        for ri, role in enumerate(roles):
            folder, label = output/'generated'/method, 'selection_'+role
            detail = read(folder/(label+'.json'))
            n = ROLE_SIZES['selection'][role+'_q']
            seed = protocol['mask_seed_base']+10000*(mi+1)+1000*(ri+1)
            require(detail['method'] == method and detail['role'] == role and detail['seed'] == seed,
                    'Selection generation identity or seed differs')
            require(detail['checkpoint_sha256'] == checkpoint_sha, 'Conditional abundance checkpoint changed')
            require(set(detail['outputs_sha256']) == {label+s+'.npy' for s in ('', '_mask', '_latent')},
                    'Generated role output inventory differs')
            hashes(folder, detail['outputs_sha256'])
            noise_path = hybrid/'generator/data'/('noise_'+label+'.npy')
            require(Path(detail['noise_path']) == noise_path and detail['noise_sha256'] == sha(noise_path),
                    'Gaussian noise is not paired with the current ICFM reference')
            noise = np.load(noise_path, allow_pickle=False)
            values, masks, latent = [np.load(folder/(label+s+'.npy'), allow_pickle=False) for s in ('', '_mask', '_latent')]
            require(values.shape == masks.shape == latent.shape == noise.shape == (n, 614) and
                    values.dtype == np.float64 and latent.dtype == noise.dtype == np.float32,
                    'Selection generation dimensions/dtypes differ')
            decoded = independent_decode(latent, masks)
            require(np.allclose(values, decoded, atol=2e-14, rtol=2e-13) and np.array_equal(values > 0, masks),
                    'Independent masked-softmax decode differs')
            require(np.isfinite(values).all() and (values >= 0).all() and
                    np.allclose(values.sum(1), 1., atol=2e-15, rtol=0), 'Invalid generated composition')
            mask_info, ode = detail['mask'], detail['ode']
            require(mask_info['method'] == mask_method and mask_info['requested'] == n and mask_info['seed'] == seed
                    and mask_info['nonempty_conditioning'] and mask_info['model_sha256'] == models['artifacts_sha256'][model_file],
                    'Generated masks are not bound to the frozen fitted model')
            require(mask_info['proposals']-mask_info['rejected_empty'] == n and
                    mask_info['unique_masks'] == len(np.unique(np.packbits(masks, axis=1), axis=0)),
                    'Mask sampling accounting differs')
            if mask_method != 'autoregressive':
                replay, proposals, rejected = replay_masks(comparison/'models', mask_method, n, seed,
                                                           models['identity']['config']['sampling'])
                require(np.array_equal(replay, masks) and proposals == mask_info['proposals']
                        and rejected == mask_info['rejected_empty'], 'Independent seeded mask replay differs')
            require(ode['evaluation_mode'] and ode['mask_conditioned_velocity'] and ode['pseudocount'] is None
                    and ode['solver'] == 'torchdyn.dopri5' and ode['saved_times'] == [0,1]
                    and ode['atol'] == ode['rtol'] == 1e-4 and ode['batch_size'] == 128,
                    'Prespecified conditional ODE sampling contract differs')
            require(ode['input_mask_array_sha256'] == hashlib.sha256(masks.tobytes()).hexdigest() and
                    ode['input_noise_array_sha256'] == hashlib.sha256(noise.tobytes()).hexdigest(), 'ODE input identity differs')
            verified.append(dict(method=method, role=role, rows=n, mask_seed=seed,
                mask_replayed=mask_method != 'autoregressive', maximum_decode_difference=float(np.max(abs(values-decoded)))))
    return dict(status='passed', completion_sha256=sha(output/'generation_complete.json'),
                independent_mask_replays=8, independent_decodes=12, roles=verified, ode_trajectories_replayed=False,
                gpu_nade_random_draws_replayed=False)


def config_audit(config):
    expected = dict(seed=2026100400, repeats=5, bins=20, hidden_dimension=128, blocks=3,
                    epsilon=1e-6, representation='ilr', architecture='deep', min_epochs=100, max_epochs=1000)
    require(all(config.get(key) == value for key, value in expected.items()), 'Prespecified RDR protocol differs')
    require(set(config['losses']) == {'hellinger', 'kl', 'chisq', 'js'} and
            set(config['output_alphas']) == {.5, 1., 2., 4.}, 'Prespecified loss/slope grid differs')


def verify(output):
    started = time.monotonic()
    output = Path(output).resolve()
    protocol = read(output/'protocol.json')
    require(tuple(protocol['methods']) == METHODS, 'New-method inventory differs')
    hashes(output/'source', protocol['source_sha256'])
    hashes(output, protocol['inputs_sha256'], absolute=True)
    require(protocol['rdr_config_sha256'] == sha(output/'rdr_config.json'), 'Root RDR configuration changed')
    completed = read(output/'fits_complete.json')
    require(completed.get('status') == 'complete' and
            (completed['selection_fits'], completed['final_fits'], completed['null_fits']) == (240,15,15),
            'The three-study fit inventory is incomplete')
    require(set(completed['method_completions']) == set(METHODS), 'Fit completion method inventory differs')
    for method, digest in completed['method_completions'].items():
        require(sha(output/'analyses'/method/'COMPLETE.json') == digest, 'Root method completion binding differs')
    generation = generation_audit(output, protocol)
    hybrid = Path(protocol['hybrid_root'])/'analysis'
    midasim = Path(protocol['midasim_root'])/'analysis'
    comparison = Path(protocol['comparison_root'])
    _, parent_data, _, _, hybrid_receipt = analysis_gate(hybrid)
    _, _, _, _, midasim_receipt = analysis_gate(midasim)
    helpers = output/'source/experiments/AGP'
    for name in ('verify_selected.py', 'verify_midasim_full.py'):
        relative = 'experiments/AGP/'+name
        require(protocol['source_sha256'].get(relative) == sha(helpers/name), 'Independent helper is not frozen')
    math = module('_agp_mask_rdr_independent_math', helpers/'verify_selected.py')
    checkpoint = module('_agp_mask_rdr_independent_checkpoint', helpers/'verify_midasim_full.py')
    results, source_receipts = {}, {}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    with threadpool_limits(limits=2):
        for method in METHODS:
            analysis = output/'analyses'/method
            config, data, chosen, work, provenance = analysis_gate(analysis)
            config_audit(config)
            require(data['root_protocol_sha256'] == sha(output/'protocol.json') and
                    data['generation_complete_sha256'] == generation['completion_sha256'], 'Analysis root/generation binding differs')
            require(read(analysis/'protocol.json')['root_protocol_sha256'] == sha(output/'protocol.json'),
                    'Analysis root protocol binding differs')
            raw, manifests, counts, receipts = data_audit(analysis, data, method, hybrid, parent_data,
                                                        comparison, output/'generated')
            source_receipts.update(receipts)
            models, selection = checkpoint.checkpoint_audit(analysis, config, chosen, work, raw, manifests, math)
            results[method] = dict(**provenance, exact_counts=counts, selection=selection,
                selection_fits=80, final_fits=5, null_fits=5, checkpoint_stopping_rescores=90,
                selection_calibration_evaluation_rescores=80, final_null_test_rescores=10,
                independently_checked_c2_cells=1800, models=models,
                maximum_prediction_difference=max(m['maximum_prediction_difference'] for m in models),
                maximum_stopping_brier_difference=max(m['maximum_stopping_brier_difference'] for m in models))
    hashes(output, protocol['inputs_sha256'], absolute=True)
    hashes(output, source_receipts, absolute=True)
    for result in [hybrid_receipt, midasim_receipt, *results.values()]:
        require(sha(Path(result['output'])/'COMPLETE.json') == result['completion_sha256'], 'Archive changed during numerical verification')
    return dict(status='passed', output=str(output), protocol_sha256=sha(output/'protocol.json'),
                generation=generation, fits_completion_sha256=sha(output/'fits_complete.json'),
                methods=results, reused_references=dict(midasim_mask_icfm=hybrid_receipt, midasim=midasim_receipt),
                newly_verified_fits=270, reused_reference_fits=180, independently_checked_c2_cells=5400,
                checkpoint_stopping_rescores=270, selection_calibration_evaluation_rescores=240,
                final_null_test_rescores=30, generated_source_sha256=source_receipts,
                helper_source_sha256={name: sha(helpers/name) for name in ('verify_selected.py', 'verify_midasim_full.py')},
                elapsed_seconds=time.monotonic()-started,
                interpretation='Numerical/provenance verification only. Same retrospective cohort and overlapping generator/RDR development roles as the archived references; bootstrap mask membership remains a known support limitation. C.1/C.2 target score-cell averages, not pointwise RDR. No interval coverage or independent generator-refit claim. Generator ODE paths were not replayed.')


def write_receipt(receipt, result):
    receipt = Path(receipt)
    require(result.get('status') == 'passed', 'Only a successful verification can be persisted')
    require(not receipt.exists() and receipt.parent.is_dir(), 'Verification receipt must be fresh in an existing directory')
    with receipt.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    receipt = args.receipt or args.output/'verification.json'
    require(not receipt.exists(), 'Refusing to overwrite a verification receipt')
    result = verify(args.output)
    result.update(verified_utc=datetime.now(timezone.utc).isoformat(), verifier_source=str(Path(__file__).resolve()),
                  verifier_source_sha256=sha(__file__))
    write_receipt(receipt, result)
    print(json.dumps(dict(status='passed', receipt=str(receipt), newly_verified_fits=270)), flush=True)


if __name__ == '__main__':
    main()
