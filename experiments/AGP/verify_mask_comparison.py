#!/usr/bin/env python3
"""Independent artifact audit for the five-way AGP mask comparison, without RDR.

Recomputes scalar metrics, replays bootstrap/mixture sampling, checks every
stored flow decode, and scores fixed subsets with the persisted classifiers.
The neural ODE and GPU NADE random draws are not re-executed by this CPU audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.stats import rankdata

METHODS = ('bootstrap_icfm', 'bernoulli_mixture_icfm', 'autoregressive_icfm',
           'midasim_mask_icfm', 'midasim')
ROLES = {'train': 'train', 'stop': 'stopping', 'test': 'test'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def close(values):
    values = np.asarray(values, dtype=np.float64)
    ensure(values.ndim == 2 and np.isfinite(values).all() and np.all(values >= 0)
           and np.all(values.sum(1) > 0), 'Invalid composition')
    return values / values.sum(1, keepdims=True)


def array_hash(values, *, masks=False):
    values = np.asarray(values)
    digest = hashlib.sha256(str(values.shape).encode())
    if masks:
        values = np.ascontiguousarray(values, dtype=np.uint8)
    else:
        values = np.ascontiguousarray(values)
        digest.update(values.dtype.str.encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def hashes(root, expected):
    for name, digest in expected.items():
        path = Path(name) if Path(name).is_absolute() else Path(root) / name
        ensure(path.is_file() and sha(path) == digest, f'Pinned artifact changed: {path}')


def near(a, b, context, *, atol=2e-10):
    ensure(np.allclose(a, b, atol=atol, rtol=2e-11, equal_nan=True),
           f'Numerical audit mismatch: {context}: {a} versus {b}')
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def rank_auc(p, q):
    """Mann--Whitney statistic, with midranks for ties; P is the positive class."""
    p, q = np.asarray(p), np.asarray(q)
    ensure(p.ndim == q.ndim == 1 and min(len(p), len(q)) > 0,
           'AUC needs two nonempty vectors')
    values = np.r_[p, q]
    ensure(np.isfinite(values).all(), 'AUC scores must be finite')
    ranks = rankdata(values, method='average')
    return float((ranks[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(q)))


def balanced_log_loss(p, q):
    p, q = np.asarray(p), np.asarray(q)
    ensure(np.isfinite(p).all() and np.isfinite(q).all()
           and np.all((p >= 0) & (p <= 1)) and np.all((q >= 0) & (q <= 1)),
           'Class probabilities outside [0,1]')
    eps = np.finfo(np.result_type(p.dtype, q.dtype)).eps
    return float(-.5 * (np.log(np.clip(p, eps, 1-eps)).mean()
                       + np.log1p(-np.clip(q, eps, 1-eps)).mean()))


def ks_statistic(a, b):
    a, b = np.sort(a), np.sort(b)
    points = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, points, side='right') / len(a)
                               - np.searchsorted(b, points, side='right') / len(b))))


def row_counts(values):
    return Counter(row.tobytes() for row in np.packbits(values, axis=1))


def schema(values):
    mean = values.mean(0)
    active = np.flatnonzero((mean > 0) & (mean < 1))
    active = sorted(active.tolist(), key=lambda j: (-mean[j], j))
    return dict(dimension=values.shape[1], active=active,
                always_one=np.flatnonzero(mean == 1).tolist(),
                always_zero=np.flatnonzero(mean == 0).tolist(),
                order='descending training prevalence, original index breaks ties')


def fidelity(p, q, training, reference):
    p, q, training, reference = map(close, (p, q, training, reference))
    bp, bq, bt = p > 0, q > 0, training > 0
    rp, rq = bp.sum(1), bq.sum(1)
    def entropy(x):
        terms = np.zeros_like(x)
        positive = x > 0
        terms[positive] = x[positive] * np.log(x[positive])
        return -terms.sum(1)
    hp, hq = entropy(p), entropy(q)
    varying = np.flatnonzero(bt.any(0) & ~bt.all(0))
    valid_p = np.flatnonzero(bp[:, varying].any(0) & ~bp[:, varying].all(0))
    valid_q = np.flatnonzero(bq[:, varying].any(0) & ~bq[:, varying].all(0))
    valid = np.intersect1d(valid_p, valid_q)
    n_pairs = len(valid) * (len(valid)-1) // 2
    if n_pairs:
        cp = np.corrcoef(bp[:, varying[valid]].astype(float), rowvar=False)
        cq = np.corrcoef(bq[:, varying[valid]].astype(float), rowvar=False)
        upper = np.triu_indices(len(valid), 1)
        correlation_mae = float(np.abs(cp[upper] - cq[upper]).mean())
    else:
        correlation_mae = np.nan
    reference_rows = set(row_counts(reference > 0))
    q_rows = row_counts(bq)
    novelty = sum(count for key, count in q_rows.items() if key not in reference_rows) / len(q)
    richness_reference = (reference > 0).sum(1)
    return dict(n_p=len(p), n_q=len(q), n_taxa=p.shape[1],
                zero_fraction_p=1-bp.mean(), zero_fraction_q=1-bq.mean(),
                richness_mean_p=rp.mean(), richness_mean_q=rq.mean(),
                richness_ks=ks_statistic(rp, rq),
                shannon_mean_p=hp.mean(), shannon_mean_q=hq.mean(),
                shannon_ks=ks_statistic(hp, hq),
                prevalence_mae=np.abs(bp.mean(0)-bq.mean(0)).mean(),
                mean_abundance_mae=np.abs(p.mean(0)-q.mean(0)).mean(),
                occurrence_correlation_mae=correlation_mae,
                occurrence_variable_taxa_training=len(varying),
                occurrence_pairs_possible=len(varying)*(len(varying)-1)//2,
                occurrence_pairs_compared=n_pairs,
                constant_p_taxa_in_training_variable_set=len(varying)-len(valid_p),
                constant_q_taxa_in_training_variable_set=len(varying)-len(valid_q),
                mask_novel_fraction=novelty, mask_unique_fraction=len(q_rows)/len(q),
                mask_reference_rows=len(reference),
                flow_training_richness_min=richness_reference.min(),
                flow_training_richness_max=richness_reference.max(),
                q_richness_below_flow_training_fraction=np.mean(rq < richness_reference.min()),
                q_richness_above_flow_training_fraction=np.mean(rq > richness_reference.max()))


def replay_masks(folder, method, n, seed, cfg):
    """Separate NumPy implementation of the two non-neural sampling laws."""
    metadata = read(folder / f'{method}.json')
    rng = np.random.default_rng(seed)
    with np.load(folder / f'{method}.npz') as archive:
        saved = {key: archive[key] for key in archive.files}
    rows, proposals, rejected, remaining = [], 0, 0, n
    while remaining:
        count = min(remaining, cfg['batch_size'], n*cfg['max_draw_multiplier'] - proposals)
        ensure(count > 0, 'Audit mask rejection limit exceeded')
        if method == 'bootstrap':
            sample = saved['masks'][rng.integers(len(saved['masks']), size=count)]
        else:
            components = rng.choice(len(saved['weights']), count, p=saved['weights'])
            random = rng.random((count, saved['probabilities'].shape[1]))
            sample = np.zeros((count, metadata['schema']['dimension']), bool)
            sample[:, metadata['schema']['active']] = random < saved['probabilities'][components]
            sample[:, metadata['schema']['always_one']] = True
        keep = sample.any(1)
        rows.append(sample[keep])
        proposals += count
        rejected += int((~keep).sum())
        remaining -= int(keep.sum())
    return np.concatenate(rows), proposals, rejected


def checkpoint_choice(history, column, min_delta, patience):
    selected, best, bad = None, None, 0
    for row in history.to_dict('records'):
        value = row['valid_nll']
        ensure(np.isfinite(value), 'Nonfinite development NLL')
        if best is None or value < best - min_delta:
            selected, best, bad = int(row[column]), float(value), 0
        else:
            bad += 1
        ensure(bad < patience or row[column] == history.iloc[-1][column],
               'Training continued past declared patience')
    return selected, best, bad


def independent_nade_nll(path, masks):
    """CPU float64 sequential conditionals from saved parameters, without NADE source."""
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = {k: v.numpy().astype(np.float64) for k, v in saved['state'].items()}
    ensure(all(np.isfinite(v).all() for v in state.values()), 'Nonfinite NADE parameters')
    a = np.broadcast_to(state['hidden_bias'], (len(masks), saved['hidden'])).copy()
    total = np.zeros(len(masks))
    for j in range(saved['dimension']):
        logit = (expit(a) * state['output_weights'][j]).sum(1) + state['output_bias'][j]
        total += np.logaddexp(0, logit) - masks[:, j]*logit
        a += masks[:, j, None] * state['input_weights'][j]
    return float(total.mean())


def check_models(out, cfg, real, full):
    folder = out / 'models'
    receipt = read(folder / 'COMPLETE.json')
    ensure(receipt['status'] == 'complete', 'Mask fit incomplete')
    hashes(folder, receipt['artifacts_sha256'])
    identity = receipt['identity']
    ensure(identity['config'] == cfg['model_config'], 'Mask configuration changed')
    ensure(identity['source_sha256'] == sha(out / 'source/mask_models.py'), 'Mask source changed')
    for name, values in [('train', real['train'] > 0), ('valid', real['stop'] > 0), ('full', full)]:
        ensure(identity[name+'_sha256'] == array_hash(values, masks=True), 'Mask fit input differs: '+name)
    ensure(row_counts(full) == row_counts(real['train'] > 0) + row_counts(real['stop'] > 0),
           'Full mask template differs from the multiset union of development roles')
    ensure(receipt['development_schema'] == schema(real['train'] > 0)
           and receipt['final_schema'] == schema(full), 'Mask ordering/constants not derived from declared rows')
    with np.load(folder / 'bootstrap.npz') as archive:
        ensure(np.array_equal(archive['masks'], full), 'Bootstrap donor matrix differs from generator template')
    candidates = pd.read_csv(folder / 'candidate_validation.csv')
    expected = len(set(cfg['model_config']['mixture']['components'])) * cfg['model_config']['mixture']['starts']
    expected += len(set(cfg['model_config']['autoregressive']['hidden']))
    ensure(len(candidates) == expected and candidates.candidate.is_unique, 'Mask candidate grid incomplete')
    largest_nll_error = 0.
    for method in ('bernoulli_mixture', 'autoregressive'):
        frame = candidates[candidates.method == method]
        mixture = method == 'bernoulli_mixture'
        column, suffix = ('iteration', 'iterations') if mixture else ('epoch', 'epochs')
        settings = cfg['model_config']['mixture' if mixture else 'autoregressive']
        cap = settings['max_iter' if mixture else 'max_epochs']
        ordering = ['valid_nll', 'components', 'initialization'] if mixture else ['valid_nll', 'hidden']
        best = frame.sort_values(ordering, kind='stable').iloc[0]
        ensure(receipt['selected'][method]['candidate'] == best.candidate, 'Mask architecture is not minimum development NLL')
        ensure(frame['selected'].sum() == 1 and bool(best['selected']), 'Candidate selection flags disagree')
        metadata = read(folder / f'{method}.json')
        ensure(all(receipt['selected'][method].get(key) == value
                   for key,value in metadata['selected'].items()), 'Selected mask receipt disagreement')
        ensure(metadata['schema'] == receipt['final_schema'], 'Final mask schema mismatch')
        ensure(metadata['training_rows'] == len(full), 'Refit used the wrong sample count')
        if mixture:
            with np.load(folder / 'bernoulli_mixture.npz') as archive:
                weights, probabilities = archive['weights'], archive['probabilities']
            ensure(weights.shape == (int(best.components),)
                   and probabilities.shape == (int(best.components), len(receipt['final_schema']['active']))
                   and np.isfinite(weights).all() and np.isfinite(probabilities).all()
                   and np.all(weights > 0) and np.all((probabilities > 0) & (probabilities < 1)),
                   'Final mixture parameters violate the selected architecture')
            near(weights.sum(), 1., 'Refit mixture weights sum')
        else:
            import torch
            saved = torch.load(folder/'autoregressive.pt', map_location='cpu', weights_only=True)
            ensure(saved['dimension'] == len(receipt['final_schema']['active'])
                   and saved['hidden'] == int(best.hidden)
                   and all(bool(torch.isfinite(tensor).all()) for tensor in saved['state'].values()),
                   'Final NADE parameters violate the selected architecture')
        ensure(metadata['refit_'+suffix] == int(best['selected_'+suffix]), 'Refit budget differs from development selection')
        refit = pd.read_csv(folder / f'{method}_refit_history.csv')
        ensure(np.array_equal(refit[column], np.arange(metadata['refit_'+suffix]+1)), 'Refit history incomplete')
        near(refit.iloc[-1].train_nll, metadata['refit_train_nll'], method+' refit NLL')
        for row in frame.to_dict('records'):
            history = pd.read_csv(folder / f"{row['candidate']}_history.csv")
            ensure(np.array_equal(history[column], np.arange(len(history))), 'Candidate history missing iterations')
            selected, loss, bad = checkpoint_choice(history, column, settings['min_delta'], settings['patience'])
            ensure(selected == row['selected_'+suffix], 'Candidate checkpoint violates development stopping rule')
            near(loss, row['valid_nll'], 'Mask checkpoint NLL')
            ensure(history.iloc[-1][column] == row[suffix+'_completed'], 'Candidate completed-iteration receipt wrong')
            cap_hit = history.iloc[-1][column] == cap and bad < settings['patience']
            ensure(bool(row['cap_hit']) == bool(cap_hit), 'Candidate cap flag is incorrect')
            validation = (real['stop'] > 0)[:, receipt['development_schema']['active']]
            if mixture:
                with np.load(folder / (row['candidate']+'.npz')) as archive:
                    weights, probabilities = archive['weights'], archive['probabilities']
                ensure(np.all(weights > 0) and np.all((probabilities > 0) & (probabilities < 1)),
                       'Invalid smoothed Bernoulli mixture probabilities')
                near(weights.sum(), 1., 'Mixture weights sum')
                values = validation.astype(np.float64)
                ll = values @ np.log(probabilities).T + (1-values) @ np.log1p(-probabilities).T
                checked_nll = float(-logsumexp(ll + np.log(weights), axis=1).mean())
            else:
                checked_nll = independent_nade_nll(folder / (row['candidate']+'.pt'), validation)
            largest_nll_error = max(largest_nll_error, near(checked_nll, row['valid_nll'],
                                      'Independent mask validation likelihood', atol=1e-4))
    return dict(candidates=len(candidates), nll_max_absolute_difference=largest_nll_error,
                selected=receipt['selected'], cap_hits=int(candidates.cap_hit.sum()))


def feature_matrix(values, recipe):
    values = close(values)
    mask = (values > 0).astype(np.float32)
    if recipe == 'hgb_composition':
        return np.concatenate([np.sqrt(values), mask], axis=1).astype(np.float32)
    if recipe == 'hgb_mask':
        return np.concatenate([mask, mask.sum(1)[:, None]], axis=1).astype(np.float32)
    ensure(recipe == 'logistic_composition', 'Unknown frozen classifier representation')
    return np.sqrt(values).astype(np.float32)


def model_predict(fit, values):
    x = feature_matrix(values, fit['recipe'])
    if fit['selected_iteration'] == 0:
        return np.full(len(x), .5)
    if fit['selected_iteration'] is None:
        return fit['model'].predict_proba(x)[:, 1]
    for i, probabilities in enumerate(fit['model'].staged_predict_proba(x), 1):
        if i == fit['selected_iteration']:
            return probabilities[:, 1]
    raise ValueError('Missing classifier checkpoint')


def audit(out):
    cfg = read(out / 'protocol.json')
    ensure(cfg['methods'] == list(METHODS) and cfg['no_rdr'] is True, 'Wrong comparison protocol')
    hashes(out / 'source', cfg['source_sha256'])
    hashes(out, cfg['inputs_sha256'])
    hybrid, midasim = Path(cfg['hybrid_root']), Path(cfg['midasim_root'])
    original_completion = read(hybrid/'generator/COMPLETE.json')
    original_protocol = read(hybrid/'generator/protocol.json')
    ensure(original_completion['status'] == 'complete'
           and original_completion['protocol_sha256'] == sha(hybrid/'generator/protocol.json')
           and original_completion['checkpoint_sha256'] == sha(hybrid/'generator/training/final.pt'),
           'Original conditional flow completion chain is invalid')
    hashes(hybrid/'generator/source',original_protocol['source_sha256'])
    hashes(hybrid,original_protocol['runtime_receipts'])
    hashes(hybrid/'generator',original_protocol['data_receipts'])
    for filename in ('icfm_zero_generator.py','icfm_corrected_generator.py'):
        ensure(sha(out/'source'/filename)
               == original_protocol['source_sha256']['experiments/AGP/'+filename],
               'Reused flow source differs from the completed original fit')
    manifest = pd.read_csv(hybrid / 'analysis/data/final/manifest.csv', dtype={'subject_id': str})
    real, ids, generated = {}, {}, {m: {} for m in METHODS}
    for role, original in ROLES.items():
        real[role] = np.load(hybrid / f'analysis/data/final/{original}_p.npy')
        rows = manifest[(manifest.distribution == 'P') & (manifest.role == original)].sort_values('array_row')
        ensure(np.array_equal(rows.array_row, np.arange(len(real[role]))), 'Manifest array order wrong')
        ids[role] = rows.subject_id.to_numpy(dtype=str)
        ensure(len(real[role]) == cfg['p_counts'][role]
               and len(set(ids[role])) == cfg['p_subject_counts'][role], 'Real cohort count wrong')
        ensure(np.array_equal(real[role], np.load(midasim / f'analysis/data/final/{original}_p.npy')),
               'Reference methods use different real cohorts')
        for method in METHODS:
            source = hybrid if method == 'midasim_mask_icfm' else midasim if method == 'midasim' else None
            path = source / f'analysis/data/final/{original}_q.npy' if source else out / f'generated/{method}/{role}.npy'
            generated[method][role] = np.load(path)
            values = generated[method][role]
            ensure(values.shape == (cfg['q_counts'][role], cfg['taxa']), 'Generated sample count or taxa wrong')
            close(values)
            ensure(np.max(np.abs(values.sum(1, dtype=float)-1)) < 2e-6, 'Generated arrays violate closure')
    for a, b in [('train', 'stop'), ('train', 'test'), ('stop', 'test')]:
        ensure(not set(ids[a]) & set(ids[b]), 'Real subjects overlap between roles')
    gen_manifest = pd.read_csv(hybrid / 'generator/data/manifest.csv', dtype={'subject_id': str})
    generator_ids = set(gen_manifest.subject_id)
    ensure(generator_ids == set(ids['train']) | set(ids['stop'])
           and not generator_ids & set(ids['test']), 'Generator training or mask selection overlaps real assessment')
    full = np.loadtxt(hybrid / 'generator/data/generator_train.csv', delimiter=',') > 0
    ensure(full.shape == (cfg['generator_rows'], cfg['taxa'])
           and len(generator_ids) == cfg['generator_subjects'], 'Generator template size wrong')
    models = check_models(out, cfg, real, full)
    generation = read(out / 'generation_complete.json')
    ensure(generation['status'] == 'complete', 'Generation incomplete')
    hashes(out / 'generated', generation['outputs_sha256'])
    ensure(generation['models_receipt_sha256'] == sha(out / 'models/COMPLETE.json'), 'Generation used different mask fits')
    ensure(cfg['sampling']['ema'] is True, 'Conditional flow must use the frozen EMA checkpoint')
    decode_error, n_replayed = 0., 0
    for j, method in enumerate(METHODS[:3]):
        folder = out / 'generated' / method
        for i, (role, original) in enumerate(ROLES.items()):
            metadata = read(folder / f'{role}.json')
            hashes(folder, metadata['outputs_sha256'])
            mask = np.load(folder / f'{role}_mask.npy')
            latent = np.load(folder / f'{role}_latent.npy')
            values = generated[method][role]
            ensure(mask.dtype == bool and np.array_equal(values > 0, mask) and mask.any(1).all(),
                   'Generated exact support differs from declared nonempty binary mask')
            ensure(latent.shape == mask.shape and np.isfinite(latent).all()
                   and np.all(latent[~mask] == 0), 'Flow latent coordinates violate masked support')
            logs = np.where(mask, latent.astype(float), -np.inf)
            weights = np.exp(logs - logs.max(1, keepdims=True))
            reconstructed = weights / weights.sum(1, keepdims=True)
            decode_error = max(decode_error, near(values, reconstructed, 'Stored flow decode', atol=2e-15))
            expected_seed = cfg['seed'] + 10000*(j+1) + 1000*(i+1)
            ensure(metadata['seed'] == expected_seed and metadata['mask']['seed'] == expected_seed,
                   'Mask sampling seed deviates from frozen schedule')
            noise_path = hybrid / f'generator/data/noise_final_{original}.npy'
            ensure(metadata['noise_path'] == str(noise_path) and metadata['noise_sha256'] == sha(noise_path),
                   'ICFM variants do not reuse the same role-specific Gaussian noise')
            noise = np.load(noise_path)
            ensure(metadata['ode']['input_noise_array_sha256'] == hashlib.sha256(np.ascontiguousarray(noise).tobytes()).hexdigest(),
                   'ODE noise array receipt wrong')
            ensure(metadata['ode']['input_mask_array_sha256'] == hashlib.sha256(np.ascontiguousarray(mask).tobytes()).hexdigest(),
                   'ODE mask array receipt wrong')
            ensure(metadata['checkpoint_sha256'] == cfg['inputs_sha256'][str(hybrid / 'generator/training/final.pt')],
                   'Conditional abundance flow checkpoint changed')
            for name in ('atol', 'rtol', 'batch_size'):
                ensure(metadata['ode'][name] == cfg['sampling'][name], 'ODE setting differs from frozen protocol')
            ensure(metadata['ode']['evaluation_mode'] and metadata['ode']['mask_conditioned_velocity'],
                   'Conditional flow generation was not evaluated with fixed masks')
            details = metadata['mask']
            ensure(details['requested'] == len(mask) and details['proposals']-details['rejected_empty'] == len(mask)
                   and details['nonempty_conditioning'], 'Mask rejection accounting wrong')
            near(details['rejection_fraction'], details['rejected_empty']/details['proposals'], 'Mask rejection fraction')
            ensure(details['unique_masks'] == len(row_counts(mask)), 'Unique mask count wrong')
            sampler = method.removesuffix('_icfm')
            model_file = {'bootstrap':'bootstrap.npz','bernoulli_mixture':'bernoulli_mixture.npz',
                          'autoregressive':'autoregressive.pt'}[sampler]
            ensure(details['model_sha256'] == sha(out/'models'/model_file),
                   'Generated mask model digest differs from selected refit')
            saved_schema = read(out / 'models' / f'{sampler}.json')['schema']
            ensure(np.all(mask[:, saved_schema['always_one']]) and not mask[:, saved_schema['always_zero']].any(),
                   'Generated masks violate deterministic fitted columns')
            if sampler != 'autoregressive':
                replay, proposals, rejected = replay_masks(out/'models', sampler, len(mask), expected_seed,
                                                           cfg['model_config']['sampling'])
                ensure(np.array_equal(replay, mask), 'Seeded independent mask replay differs')
                ensure(proposals == details['proposals'] and rejected == details['rejected_empty'],
                       'Independent rejection accounting differs')
                n_replayed += 1
    for role, original in ROLES.items():
        support = np.load(hybrid / f'generator/data/mask_final_{original}.npy')
        ensure(np.array_equal(generated['midasim_mask_icfm'][role] > 0, support)
               and np.array_equal(generated['midasim'][role] > 0, support), 'Old paired reference masks changed')
    metric_folder = out / 'metrics'
    freeze = read(out / 'classifiers_frozen.json')
    hashes(metric_folder, freeze['files_sha256'])
    ensure(freeze['protocol_sha256'] == sha(out/'protocol.json')
           and freeze['generation_complete_sha256'] == sha(out/'generation_complete.json'), 'Classifier freeze provenance changed')
    assessment = read(out / 'assessment_complete.json')
    hashes(metric_folder, assessment['outputs_sha256'])
    ensure(assessment['status'] == 'complete' and assessment['freeze_sha256'] == sha(out/'classifiers_frozen.json'),
           'Assessment not bound to frozen classifiers')
    ensure(datetime.fromisoformat(freeze['finished_utc']) <= datetime.fromisoformat(assessment['finished_utc']),
           'Assessment predates classifier freeze')
    mf, mc = read(metric_folder/'FREEZE.json'), read(metric_folder/'COMPLETE.json')
    ensure(mf['no_test_data_used'] and not mc['rdr_fitted_or_evaluated'], 'Wrong fit/assessment scope')
    ensure(mf['source_sha256'] == sha(out/'source/mask_metrics.py'), 'Evaluation source changed since freeze')
    ensure(mf['config_sha256'] == sha(metric_folder/'config.json'), 'Classifier configuration changed')
    hashes(metric_folder/'private', mf['model_hashes'])
    hashes(metric_folder/'summary', mc['summary_hashes'])
    expected_inputs = {f'{name}/{role}': array_hash(bank[role])
                       for name, bank in [('real', real), *generated.items()] for role in ('train', 'stop')}
    ensure(mf['training_inputs'] == expected_inputs and mf['null_partition_unit'] == 'subject',
           'Classifier preparation used wrong roles or null partition unit')
    for role in ('train', 'stop'):
        ensure(mf['subject_hashes'][role] == array_hash(ids[role]), 'Classifier subject provenance wrong')
    ensure(mc['test_input_hashes'] == {'real': array_hash(real['test']),
           **{m: array_hash(generated[m]['test']) for m in METHODS}}, 'Test input receipts differ')
    ensure(mc['test_subjects_sha256'] == array_hash(ids['test']), 'Test subject receipt differs')
    config = read(metric_folder/'config.json')
    null = {}
    for role in ROLES:
        filename = 'null_test_indices.npz' if role == 'test' else 'null_training_indices.npz'
        with np.load(metric_folder/'private'/filename) as archive:
            pair = [archive[side if role == 'test' else role+'_'+side] for side in ('p', 'q')]
        a, b = pair
        ensure(np.array_equal(np.sort(np.r_[a,b]), np.arange(len(real[role])))
               and not set(ids[role][a]) & set(ids[role][b]), 'Null halves do not partition disjoint subjects')
        unique = np.unique(ids[role])
        chosen = np.random.default_rng(config['seed']+list(ROLES).index(role)).permutation(len(unique))[:len(unique)//2]
        ensure(np.array_equal(a, np.flatnonzero(np.isin(ids[role], unique[chosen]))), 'Null partition seed mismatch')
        null[role] = pair
    rows = pd.read_csv(metric_folder/'summary/classifier_metrics.csv')
    expected_pairs = {(m,r) for m in (*METHODS,'real_real') for r in config['recipes']}
    ensure(len(rows) == len(expected_pairs) and set(zip(rows.method, rows.recipe)) == expected_pairs,
           'Classifier comparison rows incomplete')
    max_auc_error = max_loss_error = max_prediction_error = 0.
    for row in rows.to_dict('records'):
        name = row['method']+'__'+row['recipe']
        fit = joblib.load(metric_folder/'private'/(name+'.joblib'))
        for key in ('selection_at_cap', 'convergence_warning'):
            ensure(bool(row[key]) == bool(fit[key]), 'Reported classifier flag differs: '+key)
        ensure((pd.isna(row['selected_iteration']) and fit['selected_iteration'] is None)
               or row['selected_iteration'] == fit['selected_iteration'],
               'Reported classifier stopping iteration differs')
        trace = read(metric_folder/'private'/(name+'_trace.json'))
        ensure(trace == fit['trace'], 'Stored classifier stopping trace differs')
        if row['recipe'].startswith('hgb_'):
            best_iteration, best_loss = 0, float(np.log(2))
            for item in trace:
                if item['stop_balanced_log_loss'] < best_loss-1e-12:
                    best_iteration, best_loss = item['iteration'], item['stop_balanced_log_loss']
            ensure(fit['selected_iteration'] == best_iteration, 'Classifier was not selected by minimum stopping log loss')
            ensure(fit['selection_at_cap'] == (best_iteration == config['hgb']['max_iter']), 'Classifier cap flag wrong')
        for role in ('stop', 'test'):
            if row['method'] == 'real_real':
                pp, qq = [close(real[role])[indices] for indices in null[role]]
            else:
                pp, qq = close(real[role]), close(generated[row['method']][role])
            if role == 'stop':
                sp, sq = model_predict(fit, pp), model_predict(fit, qq)
                near(balanced_log_loss(sp,sq), fit['stop_balanced_log_loss'], 'Frozen stopping classifier score')
                near(row['stop_balanced_log_loss'], fit['stop_balanced_log_loss'], 'Reported stopping score')
                continue
            with np.load(metric_folder/'private'/(name+'_test_predictions.npz')) as archive:
                pscore, qscore = archive['p'], archive['q']
            ensure(len(pscore) == len(pp) == row['n_p'] and len(qscore) == len(qq) == row['n_q'], 'Assessment prediction counts wrong')
            max_auc_error = max(max_auc_error, near(rank_auc(pscore,qscore), row['auc'], 'Assessment AUC'))
            max_loss_error = max(max_loss_error, near(balanced_log_loss(pscore,qscore), row['balanced_log_loss'], 'Assessment log loss'))
            for source, scores in ((pp,pscore),(qq,qscore)):
                indices = np.unique(np.linspace(0,len(source)-1,min(64,len(source)),dtype=int))
                max_prediction_error = max(max_prediction_error,
                      near(model_predict(fit,source[indices]),scores[indices], 'Frozen classifier prediction subset', atol=2e-12))
    ensure(mc['selected_at_boosting_cap'] == int(rows.selection_at_cap.sum())
           and mc['logistic_convergence_warnings'] == int(rows.convergence_warning.sum()),
           'Reported classifier cap/convergence totals differ')
    fidelity_rows = pd.read_csv(metric_folder/'summary/fidelity_metrics.csv')
    ensure(len(fidelity_rows) == len(METHODS) and set(fidelity_rows.method) == set(METHODS), 'Missing simple-metric methods')
    metric_error = 0.
    reference = np.vstack([real['train'],real['stop']])
    for row in fidelity_rows.to_dict('records'):
        reproduced = fidelity(real['test'],generated[row['method']]['test'],real['train'],reference)
        ensure(set(reproduced) == set(row)-{'method'}, 'Simple metric schema changed without audit')
        for key,value in reproduced.items():
            error = near(value,row[key],f'{row["method"]} {key}')
            if np.isfinite(error):
                metric_error = max(metric_error,error)
    combined = pd.read_csv(metric_folder/'summary/comparison.csv').set_index('method')
    ensure(set(combined.index) == set(METHODS) and combined.index.is_unique, 'Combined summary methods wrong')
    for row in fidelity_rows.to_dict('records'):
        for key,value in row.items():
            if key != 'method':
                near(value,combined.loc[row['method'],key],'Combined composition table')
    for row in rows[rows.method != 'real_real'].to_dict('records'):
        near(row['auc'],combined.loc[row['method'],row['recipe']+'_auc'],'Combined AUC table')
    for side in ('p','q'):
        a = np.load(metric_folder/'private/midasim_mask_icfm__hgb_mask_test_predictions.npz')[side]
        b = np.load(metric_folder/'private/midasim__hgb_mask_test_predictions.npz')[side]
        ensure(np.array_equal(a,b), 'Identical reference masks produce different mask-classifier predictions')
    hashes(out/'source',cfg['source_sha256'])
    hashes(out,cfg['inputs_sha256'])
    return dict(status='passed', finished_utc=datetime.now(timezone.utc).isoformat(),
                protocol_sha256=sha(out/'protocol.json'), verifier_sha256=sha(__file__),
                methods=list(METHODS), models=models, real_role_counts=cfg['p_counts'],
                generator_training_test_subject_overlap=0, all_subject_roles_disjoint=True,
                generated_role_arrays=15, stored_flow_decodes=9, independent_seeded_mask_replays=n_replayed,
                nade_audit='Independent float64 sequential development likelihood for both candidate checkpoints; parameter and support checks. GPU random draws not replayed.',
                flow_audit='Exact support, shared noise receipts, frozen EMA checkpoint and every stored decode; neural ODE not reintegrated.',
                classifier_rows=len(rows), classifier_prediction_rows_per_source=64,
                auc_max_absolute_difference=max_auc_error, log_loss_max_absolute_difference=max_loss_error,
                classifier_prediction_max_absolute_difference=max_prediction_error,
                fidelity_max_absolute_difference=metric_error, flow_decode_max_absolute_difference=decode_error,
                rdr_fitted_or_evaluated=False,
                scope='Descriptive fixed-generator, fixed-mask-realization and fixed-classifier audit on the retrospective cohort; no uncertainty or universal generator ranking claim.')


def is_live_log(path, out):
    relative = path.relative_to(out)
    return (any(part in ('logs', 'log') for part in relative.parts)
            or path.suffix.lower() in ('.out', '.err', '.log')
            or path.name.startswith('slurm-'))


def seal_files(out, excluded):
    files = {}
    for path in sorted(out.rglob('*')):
        if (not path.is_file() or path in excluded or '__pycache__' in path.parts
                or path.suffix == '.pyc' or is_live_log(path, out)):
            continue
        files[str(path.relative_to(out))] = sha(path)
    return files


def seal(out, verification):
    files = seal_files(out, {out/'COMPLETE.json'})
    write(out/'COMPLETE.json',dict(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
          verification_sha256=sha(out/'verification.json'), verifier_sha256=sha(__file__),
          no_rdr=True, outputs_sha256=files))


def amend_live_log_seal(out):
    """Preserve a passed audit and old receipt while removing mutable log pins.

    Only log changes are admissible among previously sealed artifacts. This
    operation cannot repair changed scientific inputs, outputs, sources or
    verification reports, and it never re-executes the numerical audit.
    """
    out = Path(out)
    original_path, amended_path = out/'COMPLETE.json', out/'COMPLETE_v2.json'
    ensure(not amended_path.exists(), 'Amended receipt already exists; it cannot be overwritten')
    original_digest = sha(original_path)
    original = read(original_path)
    ensure(original['status'] == 'complete', 'Only a completed archive can receive a live-log amendment')
    verification_path = out/'verification.json'
    verification_digest = sha(verification_path)
    ensure(read(verification_path)['status'] == 'passed'
           and verification_digest == original['verification_sha256']
           and original['outputs_sha256'].get('verification.json') == verification_digest,
           'Original passed numerical verification is missing or changed')
    unchanged, log_mismatches, excluded_logs = {}, {}, []
    for name, digest in original['outputs_sha256'].items():
        path = out/name
        ensure(not Path(name).is_absolute() and '..' not in Path(name).parts,
               'Original seal contains a nonlocal artifact path')
        observed = sha(path) if path.is_file() else None
        if is_live_log(path, out):
            excluded_logs.append(name)
            if observed != digest:
                log_mismatches[name] = dict(original_sha256=digest, current_sha256=observed)
        else:
            ensure(observed == digest, f'Non-log artifact changed; amendment prohibited: {name}')
            unchanged[name] = digest
    ensure(excluded_logs, 'Original receipt does not contain live logs needing this amendment')
    files = seal_files(out, {amended_path})
    # Recheck the original immutable scope after enumerating current artifacts.
    hashes(out, unchanged)
    ensure(sha(original_path) == original_digest and sha(verification_path) == verification_digest,
           'Original completion or verification changed during amendment')
    ensure(files.get('COMPLETE.json') == original_digest
           and files.get('verification.json') == verification_digest,
           'Amendment must pin both original receipts')
    result = dict(status='complete', finished_utc=datetime.now(timezone.utc).isoformat(),
                  amendment_reason='The original seal included live audit log output that grew after the seal was written. Exclude live logs while preserving the passed numerical audit and all original non-log artifacts.',
                  original_completion_sha256=original_digest,
                  verification_sha256=verification_digest,
                  original_verifier_sha256=original.get('verifier_sha256'),
                  amender_source=str(Path(__file__).resolve()), amender_source_sha256=sha(__file__),
                  original_nonlog_files_verified=len(unchanged),
                  excluded_original_logs=sorted(excluded_logs),
                  admissible_log_mismatches=log_mismatches,
                  numerical_audit_rerun=False, no_rdr=original.get('no_rdr'),
                  outputs_sha256=files)
    # Exclusive creation prevents accidental replacement of an amendment.
    with amended_path.open('x') as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--amend-live-log-seal', action='store_true',
                        help='Preserve original receipts and write COMPLETE_v2.json only when all originally pinned non-log artifacts are unchanged')
    args = parser.parse_args()
    if args.amend_live_log_seal:
        result = amend_live_log_seal(args.output)
        print(json.dumps(dict(status=result['status'], receipt=str(args.output/'COMPLETE_v2.json'),
              original_nonlog_files_verified=result['original_nonlog_files_verified'],
              admissible_log_mismatches=result['admissible_log_mismatches']), sort_keys=True), flush=True)
        return
    ensure(not (args.output/'COMPLETE.json').exists(), 'Archive is already sealed; verify its hashes without rewriting it')
    try:
        result = audit(args.output)
    except Exception as exc:
        write(args.output/'verification.json',dict(status='failed', error=type(exc).__name__+': '+str(exc),
              verifier_sha256=sha(__file__), finished_utc=datetime.now(timezone.utc).isoformat()))
        raise
    write(args.output/'verification.json', result)
    seal(args.output,result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == '__main__':
    main()
