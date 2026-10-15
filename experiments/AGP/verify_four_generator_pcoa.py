#!/usr/bin/env python3
"""Independently audit the four-generator assessment PCoA and reused scores.

No producer code is imported. Hashes, identities and scores are checked in full;
distance/Gram arithmetic is checked for deterministic rows, including every
cohort boundary. This does not independently solve the full eigenproblem or
certify that the saved minimum eigenpair is globally minimal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits


METHODS = ('bernoulli_mixture_icfm', 'autoregressive_icfm',
           'midasim_mask_icfm', 'midasim')
BASE = Path('/cwork/yx306/RDR/AGP')
ROOTS = {
    METHODS[0]: BASE/'mask_rdr_20261012/analyses'/METHODS[0],
    METHODS[1]: BASE/'mask_rdr_20261012/analyses'/METHODS[1],
    METHODS[2]: BASE/'icfm_zero_20261007/analysis',
    METHODS[3]: BASE/'midasim_full_20261004/analysis',
}
COMMUNITIES = BASE/'four_generator_case_study_20261013/communities'
COUNTS = {'real': 2002, **{method: 2089 for method in METHODS}}
N = sum(COUNTS.values())
SEED = 2026101407


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, name, *, atol=1e-12, rtol=1e-12):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol), name+' differs')


def validate_arrays_and_scores(saved, inputs, check):
    receipt_path = COMMUNITIES/'ASSESSMENT_COMPLETE.json'
    check(receipt_path, inputs[str(receipt_path)])
    receipt = read(receipt_path)
    require(receipt['status'] == 'complete' and receipt['score_replays'] == 20,
            'Shared assessment cache is not complete')
    cache_path = COMMUNITIES/'assessment_private.npz'
    check(cache_path, receipt['files_sha256']['assessment_private.npz'])
    require(inputs[str(cache_path)] == receipt['files_sha256']['assessment_private.npz'],
            'Input manifest and assessment cache receipt disagree')
    cohorts, shared_identity, predictions_checked = {}, None, set()
    with np.load(cache_path, allow_pickle=False) as cache:
        for method in METHODS:
            root = ROOTS[method]
            seal_path = root/'COMPLETE.json'
            protocol_path = root/'data/protocol.json'
            for path in (seal_path, protocol_path):
                require(inputs[str(path)] == receipt['inputs_sha256'][str(path)],
                        'Source receipt mismatch: '+str(path))
                check(path, inputs[str(path)])
            seal = read(seal_path)
            require(seal['status'] == 'complete'
                    and seal['selection_fits'] == 80 and seal['final_fits'] == 5,
                    'Upstream study is incomplete: '+method)
            final = read(protocol_path)['studies']['final']
            manifest_path = root/final['manifest']['path']
            require(inputs[str(manifest_path)] == final['manifest']['sha256'],
                    'Manifest receipt differs: '+method)
            check(manifest_path, final['manifest']['sha256'])
            table = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
            for side, count in (('p', 2002), ('q', 2089)):
                spec = final['arrays']['test_'+side]
                path = root/spec['path']
                require(inputs[str(path)] == spec['sha256']
                        == receipt['inputs_sha256'][str(path)], 'Array provenance mismatch')
                check(path, spec['sha256'])
                raw = np.load(path, allow_pickle=False)
                require(raw.shape == (count, 614) and list(raw.shape) == spec['shape'],
                        'Assessment array shape differs')
                require(np.isfinite(raw).all() and (raw >= 0).all()
                        and (raw.sum(axis=1, dtype=np.float64) > 0).all(),
                        'Invalid assessment composition')
                rows = table[table.role.eq('test') & table.distribution.eq(side.upper())].copy()
                rows['array_row'] = rows.array_row.astype(int)
                rows = rows.sort_values('array_row')
                require(rows.array_row.tolist() == list(range(count)),
                        'Manifest does not describe the exact assessment order')
                identity = rows[['sample_id', 'subject_id', 'source_file',
                                 'source_row', 'row_sha256']].to_numpy(str)
                key = 'real' if side == 'p' else method
                require(np.array_equal(identity, saved[key+'_identity']),
                        'Private ordination identity mismatch: '+key)
                if side == 'p':
                    require(np.array_equal(identity[:, 1], cache['subjects']),
                            'Assessment cache real subjects differ')
                if key in cohorts:
                    require(np.array_equal(cohorts[key], raw)
                            and np.array_equal(shared_identity, identity),
                            'Real assessment rows differ between comparisons')
                else:
                    cohorts[key] = raw
                    if side == 'p':
                        shared_identity = identity
                columns = []
                for repeat in range(5):
                    folder = root/'final'/f'{repeat:02d}'
                    marker_path = folder/'COMPLETE.json'
                    expected = seal['assessment_completions'][f'final/{repeat:02d}/COMPLETE.json']
                    require(inputs[str(marker_path)] == expected, 'Fit completion receipt mismatch')
                    check(marker_path, expected)
                    marker = read(marker_path)
                    require(marker['status'] == 'complete' and marker['stage'] == 'final',
                            'Incomplete assessment fit')
                    prediction_path = folder/'predictions.npz'
                    expected = marker['outputs']['predictions.npz']
                    require(inputs[str(prediction_path)] == expected
                            == receipt['inputs_sha256'][str(prediction_path)],
                            'Prediction provenance mismatch')
                    check(prediction_path, expected)
                    with np.load(prediction_path, allow_pickle=False) as prediction:
                        column = prediction['test_'+side]
                    require(column.shape == (count,), 'Prediction row count differs')
                    columns.append(column)
                    predictions_checked.add(str(prediction_path))
                score_matrix = np.column_stack(columns)
                require(np.isfinite(score_matrix).all() and (score_matrix >= 0).all()
                        and (score_matrix <= 2).all(), 'Scores are outside the fitted RDR scale')
                require(np.array_equal(score_matrix, cache[method+'_'+side+'_scores']),
                        'Completed predictions differ from the assessment cache')
                require(np.array_equal(score_matrix.mean(axis=1), saved[method+'_'+side+'_scores']),
                        'Color vector differs from the five-fit pair-specific arithmetic mean')
    require(len(predictions_checked) == 20, 'Did not verify all 20 prediction files')
    return np.concatenate([cohorts[key].astype(np.float64)
                           for key in ('real', *METHODS)]), len(predictions_checked)


def numerical_audit(raw, saved, diagnostics, sample_rows, threads):
    require(raw.shape == (N, 614), 'Joint cohort is not 10,358 by 614')
    values = raw/raw.sum(axis=1, keepdims=True)
    coords = saved['coordinates']
    vectors = saved['leading_vectors']
    minimum = saved['minimum_vector']
    means = saved['raw_squared_distance_row_means']
    grand = float(saved['raw_squared_distance_grand_mean'])
    top = np.asarray(diagnostics['raw_leading_eigenvalues'], dtype=float)
    low = float(diagnostics['raw_minimum_eigenvalue'])
    correction = float(diagnostics['lingoes_constant'])
    require(coords.shape == vectors.shape == (N, 2) and minimum.shape == means.shape == (N,),
            'Private ordination array shapes differ')
    for name, value in [('coordinates', coords), ('vectors', vectors),
                        ('minimum', minimum), ('means', means), ('top', top)]:
        require(np.isfinite(value).all(), 'Nonfinite '+name)
    require(np.isfinite([grand, low, correction]).all(), 'Nonfinite scalar diagnostic')
    require(top.shape == (2,) and top[0] >= top[1] > 0, 'Leading eigenvalues are invalid')
    require((means >= 0).all() and (means <= 1+1e-12).all(), 'Invalid squared-distance means')
    close(grand, means.mean(), 'Squared-distance grand mean')
    close(correction, max(0., -low), 'Lingoes constant')
    close(vectors.T@vectors, np.eye(2), 'Leading-vector orthonormality', atol=1e-10)
    close(np.linalg.norm(minimum), 1., 'Minimum-vector unit norm', atol=1e-10)
    close(vectors.T@minimum, np.zeros(2), 'Leading/minimum-vector orthogonality', atol=1e-8)
    require(float(np.abs(vectors.sum(axis=0)).max()) <= 1e-6,
            'Leading vectors are not centered')
    if abs(low) > 1e-8:
        require(abs(float(minimum.sum())) <= 1e-6, 'Nonzero minimum eigenvector is not centered')
    for axis in range(2):
        require(vectors[np.argmax(np.abs(vectors[:, axis])), axis] >= 0,
                'Leading-vector sign convention differs')
    close(coords, vectors*np.sqrt(top+correction), 'Lingoes coordinates', atol=1e-12)
    trace_from_hollow_distances = float(means.sum()-.5*N*grand)
    close(diagnostics['raw_trace'], trace_from_hollow_distances, 'Raw Gram trace', atol=1e-9)
    corrected_trace = trace_from_hollow_distances+(N-1)*correction
    require(corrected_trace > 0, 'Nonpositive corrected trace')
    close(diagnostics['corrected_total_inertia'], corrected_trace, 'Lingoes corrected trace', atol=1e-9)
    close(diagnostics['displayed_corrected_inertia_fractions'],
          (top+correction)/corrected_trace, 'Displayed corrected-inertia fractions')
    close(diagnostics['coordinate_centering_max_abs'],
          np.abs(coords.mean(axis=0)).max(), 'Saved coordinate centering', atol=1e-15)

    # Include the first/last row of every cohort, then add seeded rows.
    boundaries = np.r_[0, np.cumsum(list(COUNTS.values()))]
    mandatory = np.unique(np.r_[boundaries[:-1], boundaries[1:]-1])
    target = min(N, max(128, sample_rows))
    pool = np.setdiff1d(np.arange(N), mandatory)
    indices = np.sort(np.r_[mandatory, np.random.default_rng(SEED).choice(
        pool, size=target-len(mandatory), replace=False)])
    scale = max(1., float(np.abs(top).max()), abs(low))
    maximum = dict(row_mean_error=0., direct_distance_error=0.,
                   gram_row_sum_abs=0., leading_residual_abs=0., minimum_residual_abs=0.)
    residual_squared = np.zeros(3)
    direct_columns = np.unique(np.r_[mandatory, np.linspace(0, N-1, 17, dtype=int)])
    with threadpool_limits(limits=threads):
        for start in range(0, len(indices), 16):
            rows = indices[start:start+16]
            distances = cdist(values[rows], values, metric='braycurtis')
            require(np.isfinite(distances).all() and (distances >= 0).all()
                    and (distances <= 1+1e-12).all(), 'Invalid independent Bray-Curtis distances')
            close(distances[np.arange(len(rows)), rows], 0., 'Bray-Curtis diagonal', atol=1e-15)
            # A direct NumPy formula independently checks the SciPy distance kernel.
            a, b = values[rows[0]], values[direct_columns]
            manual = np.abs(a-b).sum(axis=1)/np.abs(a+b).sum(axis=1)
            direct_error = float(np.max(np.abs(manual-distances[0, direct_columns])))
            maximum['direct_distance_error'] = max(maximum['direct_distance_error'], direct_error)
            require(direct_error <= 2e-13, 'Direct Bray-Curtis formula differs')
            squared = distances*distances
            independently_computed_means = squared.mean(axis=1)
            mean_error = float(np.max(np.abs(independently_computed_means-means[rows])))
            maximum['row_mean_error'] = max(maximum['row_mean_error'], mean_error)
            close(independently_computed_means, means[rows], 'Independent squared-distance row means',
                  atol=2e-13, rtol=2e-13)
            gram = -.5*(squared-independently_computed_means[:, None]-means[None, :]+grand)
            maximum['gram_row_sum_abs'] = max(maximum['gram_row_sum_abs'],
                                              float(np.max(np.abs(gram.sum(axis=1)))))
            leading_residual = gram@vectors-vectors[rows]*top
            minimum_residual = gram@minimum-minimum[rows]*low
            maximum['leading_residual_abs'] = max(maximum['leading_residual_abs'],
                                                  float(np.max(np.abs(leading_residual))))
            maximum['minimum_residual_abs'] = max(maximum['minimum_residual_abs'],
                                                  float(np.max(np.abs(minimum_residual))))
            residual_squared[:2] += np.sum(leading_residual**2, axis=0)
            residual_squared[2] += float(np.sum(minimum_residual**2))
    residual_norms = np.sqrt(residual_squared)
    require(maximum['gram_row_sum_abs'] <= 1e-8*scale, 'Independent centered rows do not sum to zero')
    require(float(residual_norms.max()) <= 1e-8*scale, 'Independent sampled eigenpair residual is too large')
    require(diagnostics['eigen_residual_norm'] <= 1e-8*scale
            and diagnostics['centering_row_sum_max_abs'] <= 1e-8*scale,
            'Producer full-matrix residuals exceed tolerance')
    require(float(residual_norms.max()) <= diagnostics['eigen_residual_norm']+1e-10*scale,
            'Independent sampled residual exceeds the reported full residual')
    return dict(sample_rows=len(indices), sample_seed=SEED,
                sample_indices=indices.tolist(), all_cohort_boundaries_included=True,
                checked_distances=int(len(indices)*N), **maximum,
                sampled_eigen_residual_norms=residual_norms.tolist(),
                raw_trace_from_hollow_distance_identity=trace_from_hollow_distances,
                corrected_trace=corrected_trace,
                full_minimum_eigenproblem_independently_solved=False,
                limitation='Distance and Gram checks sample full rows; global extremality of the saved eigenpairs is not independently certified.')


def verify(private, public, sample_rows=160, threads=2):
    checked = {}

    def check(path, expected):
        path = Path(path).resolve()
        require(path.is_file() and not path.is_symlink(), 'Missing or symlinked input: '+str(path))
        actual = sha(path)
        require(actual == expected, 'SHA256 mismatch: '+str(path))
        require(str(path) not in checked or checked[str(path)] == actual,
                'File changed during verification: '+str(path))
        checked[str(path)] = actual

    private, public = private.resolve(), public.resolve()
    completion = read(private/'COMPLETE.json')
    require(completion['status'] == 'complete' and Path(completion['public']).resolve() == public,
            'Private completion does not identify the supplied public directory')
    check(public/'manifest.json', completion['manifest_sha256'])
    manifest = read(public/'manifest.json')
    require(manifest['status'] == 'complete' and manifest['methods'] == list(METHODS)
            and manifest['rows'] == N and manifest['fitted_scores_reused'] == 20,
            'Public manifest cohort or fit counts differ')
    check(private/'inputs.json', completion['inputs_sha256'])
    inputs = read(private/'inputs.json')
    require(inputs == manifest['inputs_sha256'], 'Private and public input receipts differ')
    for path, digest in inputs.items():
        check(path, digest)
    expected_files = set(manifest['files_sha256'])
    actual_files = {str(path.relative_to(public)) for path in public.rglob('*')
                    if path.is_file() and path != public/'manifest.json'}
    require(actual_files == expected_files, 'Public output file inventory differs from manifest')
    for relative, digest in manifest['files_sha256'].items():
        path = (public/relative).resolve()
        require(path.is_relative_to(public), 'Public output escapes its directory')
        check(path, digest)
    producer_inputs = [(path, digest) for path, digest in inputs.items()
                       if Path(path).name == 'four_generator_pcoa.py']
    require(len(producer_inputs) == 1, 'Expected exactly one pinned producer source')
    check(public/'source/four_generator_pcoa.py', producer_inputs[0][1])
    archive = private/'ordination_private.npz'
    require(completion['coordinates_sha256'] == manifest['private_coordinates_sha256'],
            'Private coordinate hash receipts differ')
    check(archive, completion['coordinates_sha256'])
    diagnostics = read(public/'ordination_diagnostics.json')
    require(diagnostics['rows'] == N and diagnostics['taxa'] == 614
            and diagnostics['group_order'] == ['real', *METHODS]
            and diagnostics['group_counts'] == COUNTS, 'Ordination cohort metadata differs')
    for key, expected in {'exact_distances': True, 'structural_zero_shifted': False,
                          'subsampling': False, 'same_real_coordinates_in_every_panel': True,
                          'assessment_only_descriptive': True,
                          'used_for_fitting_or_selection': False}.items():
        require(diagnostics[key] == expected, 'Unexpected diagnostic flag: '+key)
    require(diagnostics['score_scale'] == [0, 2], 'Plot score scale differs')
    with np.load(archive, allow_pickle=False) as saved:
        arrays, predictions = validate_arrays_and_scores(saved, inputs, check)
        numerical = numerical_audit(arrays, saved, diagnostics, sample_rows, threads)
    # Rehash after the arithmetic so the audit also detects changes during its run.
    for path, digest in checked.items():
        require(sha(path) == digest, 'Verified file changed during audit: '+path)
    return dict(status='passed', created_utc=datetime.now(timezone.utc).isoformat(),
                verifier_sha256=sha(__file__), public=str(public), private=str(private),
                manifest_sha256=sha(public/'manifest.json'),
                files_verified=len(checked), input_files_verified=len(inputs),
                rows=N, real_rows=2002, generated_rows_per_method=2089, taxa=614,
                score_prediction_files_verified=predictions, color_vectors_verified=8,
                real_identity_order_verified_across_methods=True,
                exact_score_mean_and_cache_match=True, numerical=numerical)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--sample-rows', type=int, default=160)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    require(args.threads > 0, 'Thread count must be positive')
    if args.output is not None:
        require(not args.output.exists(), 'Use a fresh verification output path')
        require(not args.output.resolve().is_relative_to(args.public.resolve()),
                'Keep the verification receipt outside the sealed public directory')
    result = verify(args.private, args.public, args.sample_rows, args.threads)
    if args.output is not None:
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    print(json.dumps(result, sort_keys=True, allow_nan=False))
