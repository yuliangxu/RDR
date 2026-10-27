#!/usr/bin/env python3
"""Assemble seven full-cohort AGP metrics and recompute taxon correlation RMSE.

Existing public tables supply FD, PRD, biological marginals, and fitted-RDR
JS(P,M). Correlation recomputation requires the frozen external train/test
arrays listed in the published metric protocol; no models are fitted.
Use --verify PACKAGE to check a saved package without accessing private arrays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/cwork/yx306/RDR/AGP')
DEFAULT_OUTPUT = ROOT / 'results/AGP/seven_global_metrics_20261026'
METHODS = {
    'bernoulli_mixture_icfm': 'Bernoulli mixture + ICFM',
    'autoregressive_icfm': 'NADE mask + ICFM',
    'midasim_mask_icfm': 'MIDASim mask + ICFM',
    'midasim': 'MIDASim',
}
SOURCES = {
    'global_rows': 'results/AGP/four_generator_case_study_20261013/global/global_metrics.csv',
    'global_summary': 'results/AGP/four_generator_case_study_20261013/global/global_metrics_aggregate.csv',
    'protocol': 'results/AGP/four_generator_case_study_20261013/provenance/metrics_protocol.json',
    'biological': 'results/AGP/retained_mask_comparison_20261013/fidelity_metrics.csv',
    'rdr': 'results/AGP/retained_mask_comparison_20261013/metrics.csv',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def close_number(a, b):
    if not math.isfinite(float(a)) or not math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-12):
        raise ValueError(f'Inconsistent metric: {a} versus {b}')


def assemble(correlations):
    global_rows = read_csv(ROOT / SOURCES['global_rows'])
    global_summary = read_csv(ROOT / SOURCES['global_summary'])
    biological = read_csv(ROOT / SOURCES['biological'])
    rdr = read_csv(ROOT / SOURCES['rdr'])
    rows = []
    for method, label in METHODS.items():
        g = [r for r in global_summary if r['method'] == method and r['representation'] == 'sqrt']
        b = [r for r in biological if r['method'] == method]
        c = [r for r in correlations if r['method'] == method]
        if not len(g) == len(b) == len(c) == 1:
            raise ValueError('Expected one full-cohort summary per generator')
        g, b, c = g[0], b[0], c[0]
        for item in (g, b, c):
            if (int(item['n_p']), int(item['n_q'])) != (2002, 2089):
                raise ValueError('Summary sources use different cohort sizes')
        if int(c['taxa']) != 50 or int(c['pairs']) != 1225:
            raise ValueError('Correlation must cover all 1225 selected pairs')
        if int(c['constant_taxa_p']) or int(c['constant_taxa_q']):
            raise ValueError('Correlation RMSE undefined for constant selected taxa')
        samples = [r for r in global_rows if r['method'] == method and r['representation'] == 'sqrt']
        fits = [r for r in rdr if r['method'] == method and r['study'] == 'final']
        if len(samples) != 5 or len({r['seed'] for r in samples}) != 5:
            raise ValueError('Expected five distinct PRD quantizer seeds')
        if len(fits) != 5 or {int(r['repeat']) for r in fits} != set(range(5)):
            raise ValueError('Expected five distinct final RDR fits')
        for r in samples + fits:
            if (int(r['n_p']), int(r['n_q'])) != (2002, 2089):
                raise ValueError('Per-repeat sources use different cohort sizes')
        f1 = [float(r['prd_max_f1']) for r in samples]
        close_number(statistics.mean(f1), g['prd_max_f1_mean'])
        close_number(statistics.stdev(f1), g['prd_max_f1_sd'])
        for r in samples:
            close_number(r['frechet'], g['frechet_mean'])
        js = [float(r['test_js_p_m_variational']) for r in fits]
        if any(not math.isfinite(v) or v > .75 * math.log(4/3) + 1e-12 for v in js):
            raise ValueError('Invalid JS(P,M) variational estimate')
        rows.append(dict(method=method, label=label, n_p=2002, n_q=2089,
            fd_sqrt=float(g['frechet_mean']),
            prd_max_f1_sqrt_mean=float(g['prd_max_f1_mean']),
            prd_max_f1_sqrt_sd=float(g['prd_max_f1_sd']), prd_quantizer_seeds=5,
            prevalence_mae=float(b['prevalence_mae']), richness_ks=float(b['richness_ks']),
            shannon_ks=float(b['shannon_ks']),
            js_p_m_variational_mean=statistics.mean(js), js_p_m_variational_sd=statistics.stdev(js),
            rdr_initializations=5, spearman_rmse=float(c['spearman_rmse']),
            correlation_taxa=50, correlation_pairs=1225))
    return rows


def correlation_metrics(archive_root):
    import numpy as np
    from scipy.stats import rankdata

    protocol = json.loads((ROOT / SOURCES['protocol']).read_text())
    receipts = {}

    def array(method, role):
        spec = protocol['arrays'][method][role]
        relative = Path(protocol['roots'][method]).relative_to(ARCHIVE) / spec['path']
        path = Path(archive_root) / relative
        if sha(path) != spec['sha256']:
            raise ValueError(f'Frozen array changed: {relative}')
        x = np.load(path, allow_pickle=False)
        if list(x.shape) != spec['shape'] or str(x.dtype) != spec['dtype']:
            raise ValueError(f'Array shape/dtype mismatch: {relative}')
        if not np.isfinite(x).all() or (x < 0).any() or (x.sum(axis=1) <= 0).any():
            raise ValueError('Invalid compositions')
        receipts[str(relative)] = dict(sha256=spec['sha256'], shape=spec['shape'], dtype=spec['dtype'])
        return x

    for role in ('train_p', 'test_p'):
        if len({protocol['arrays'][m][role]['sha256'] for m in METHODS}) != 1:
            raise ValueError('Real cohorts differ across methods')
    first = next(iter(METHODS))
    training = array(first, 'train_p')
    prevalence = (training > 0).mean(axis=0)
    selected = np.argsort(-prevalence, kind='stable')[:50]
    taxa = [dict(taxon_index=int(i), training_prevalence=float(prevalence[i])) for i in selected]

    def correlation(raw):
        x = raw.astype(np.float64)
        x /= x.sum(axis=1, keepdims=True)
        ranks = rankdata(x[:, selected], axis=0, method='average')
        ranks -= ranks.mean(axis=0)
        norms = np.sqrt(np.square(ranks).sum(axis=0))
        if (norms == 0).any():
            raise ValueError('Correlation RMSE undefined for constant selected taxa')
        return (ranks.T @ ranks) / (norms[:, None] * norms[None, :])

    p = array(first, 'test_p')
    cp = correlation(p)
    upper = np.triu_indices(50, k=1)
    rows = []
    for method, label in METHODS.items():
        q = array(method, 'test_q')
        rmse = float(np.sqrt(np.square((cp-correlation(q))[upper]).mean()))
        rows.append(dict(method=method, label=label, n_p=len(p), n_q=len(q),
            spearman_rmse=rmse, taxa=50, pairs=1225, constant_taxa_p=0, constant_taxa_q=0))
    return rows, taxa, receipts


def generate(output, archive_root):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    inputs = {path: sha(ROOT/path) for path in SOURCES.values()}
    correlations, taxa, receipts = correlation_metrics(archive_root)
    rows = assemble(correlations)
    for path, expected in inputs.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Source table changed during assembly')
    output.mkdir(parents=True)
    for name, values in (('global_metrics.csv', rows), ('correlation_rmse.csv', correlations), ('correlation_taxa.csv', taxa)):
        write_csv(output/name, values)
    manifest = dict(status='complete', kind='four_generator_full_cohort_seven_metric_summary',
        n_p=2002, n_q_per_generator=2089, methods=list(METHODS), new_model_fits=0,
        metric_count=7, definitions={
            'fd_sqrt': 'Gaussian Frechet distance of square-root relative-abundance features with unbiased sample covariances; no additional outer square root.',
            'prd_max_f1_sqrt': 'Original pair-specific 64-cluster training-only quantizers, 201-angle PRD curve maximum F1; mean and sample SD over five fixed quantizer seeds.',
            'prevalence_mae': 'Mean absolute P/Q nonzero-prevalence difference across all 614 taxa.',
            'richness_ks': 'Two-sample KS distance between numbers of nonzero taxa per composition.',
            'shannon_ks': 'Two-sample KS distance between Shannon entropies of closed compositions; natural logarithms, 0*log(0)=0.',
            'js_p_m_variational': '0.5*mean_P(log(2*r/(1+r)))+0.25*mean_P(log(2/(1+r)))+0.25*mean_Q(log(2/(1+r))); M=(P+Q)/2, r estimates p/m. Nats; mean and sample SD over five existing RDR initializations.',
            'spearman_rmse': 'sqrt(mean_{j<k}((rho_P[j,k]-rho_Q[j,k])^2)); all 1225 unordered pairs among the 50 most prevalent real-training taxa. Stable index ties; float64 row closure; average-rank Spearman; no CLR or pseudocount. Undefined if any selected taxon is constant.'},
        interpretation=['All columns use the same full assessment banks, not matched subsets.',
            'FD, marginal discrepancies, and correlation RMSE are fixed cohort statistics; no artificial repeat SD is assigned.',
            'PRD and JS SDs quantify different conditional sources of variation and are not sampling confidence intervals.',
            'Six metrics compare P with Q; JS compares P with M=(P+Q)/2. Only JS uses fitted RDR scores.',
            'Correlation covers selected pairwise dependence, not all 614 taxa or the entire joint distribution.'],
        sources_sha256=inputs, archive_inputs=receipts,
        source_script='experiments/AGP/seven_global_metrics.py', source_script_sha256=sha(__file__),
        files_sha256={name: sha(output/name) for name in ('global_metrics.csv', 'correlation_rmse.csv', 'correlation_taxa.csv')})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
    verify(output)


def verify(output):
    """Check saved-result arithmetic and hashes; does not replay private arrays."""
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    for relative, expected in manifest['sources_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError(f'Source hash mismatch: {relative}')
    if sha(__file__) != manifest['source_script_sha256']:
        raise ValueError('Source script hash mismatch')
    for relative, expected in manifest['files_sha256'].items():
        if sha(output/relative) != expected:
            raise ValueError(f'Output hash mismatch: {relative}')
    correlations = read_csv(output/'correlation_rmse.csv')
    if len(correlations) != 4 or {r['method'] for r in correlations} != set(METHODS):
        raise ValueError('Invalid correlation method set')
    taxa = read_csv(output/'correlation_taxa.csv')
    if len(taxa) != 50 or len({r['taxon_index'] for r in taxa}) != 50:
        raise ValueError('Invalid taxon list')
    expected, actual = assemble(correlations), read_csv(output/'global_metrics.csv')
    if len(actual) != 4:
        raise ValueError('Expected four summary rows')
    for a, b in zip(actual, expected):
        if set(a) != set(b):
            raise ValueError('Summary schema mismatch')
        for key in b:
            if key in ('method', 'label'):
                if a[key] != b[key]:
                    raise ValueError('Summary method order mismatch')
            else:
                close_number(a[key], b[key])
    print(f'Verified four rows, seven metrics, source/output hashes, and saved-result arithmetic: {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive-root', type=Path, default=ARCHIVE)
    parser.add_argument('--verify', type=Path, help='Verify an existing package without private arrays')
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    else:
        generate(args.output, args.archive_root)
