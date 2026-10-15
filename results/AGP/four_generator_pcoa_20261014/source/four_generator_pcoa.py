#!/usr/bin/env python3
"""Matched descriptive Bray-Curtis PCoA for the four frozen AGP studies.

All assessment observations enter one joint ordination. No generator, evaluator,
selection rule, or community is refitted. Row-level coordinates remain private.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
from threadpoolctl import threadpool_limits

METHODS = ('bernoulli_mixture_icfm', 'autoregressive_icfm', 'midasim_mask_icfm', 'midasim')
LABELS = ('Bernoulli mixture + ICFM', 'NADE masks + ICFM', 'MIDASim masks + ICFM', 'Full MIDASim')
SHORT = ('Bernoulli mixture\n+ ICFM', 'NADE masks\n+ ICFM', 'MIDASim masks\n+ ICFM', 'Full MIDASim')
BASE = Path('/cwork/yx306/RDR/AGP')
ROOTS = (BASE/'mask_rdr_20261012/analyses'/METHODS[0],
         BASE/'mask_rdr_20261012/analyses'/METHODS[1],
         BASE/'icfm_zero_20261007/analysis', BASE/'midasim_full_20261004/analysis')
COMMUNITIES = BASE/'four_generator_case_study_20261013/communities'
SEED = 20261014


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def load_assessment():
    inputs = {}

    def pin(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Changed frozen input: '+str(path))
        inputs[str(path)] = digest

    receipt = read(COMMUNITIES/'ASSESSMENT_COMPLETE.json')
    if receipt['status'] != 'complete' or receipt['score_replays'] != 20:
        raise ValueError('Four-generator assessment cache is incomplete')
    pin(COMMUNITIES/'ASSESSMENT_COMPLETE.json')
    pin(COMMUNITIES/'assessment_private.npz', receipt['files_sha256']['assessment_private.npz'])
    arrays, scores, identities = {}, {}, {}
    real_identity = None
    with np.load(COMMUNITIES/'assessment_private.npz', allow_pickle=False) as cache:
        for method, root in zip(METHODS, ROOTS):
            seal = read(root/'COMPLETE.json')
            pin(root/'COMPLETE.json', receipt['inputs_sha256'][str(root/'COMPLETE.json')])
            if seal['status'] != 'complete' or (seal['selection_fits'], seal['final_fits']) != (80, 5):
                raise ValueError('Incomplete upstream study')
            pin(root/'data/protocol.json', receipt['inputs_sha256'][str(root/'data/protocol.json')])
            spec = read(root/'data/protocol.json')['studies']['final']
            pin(root/spec['manifest']['path'], spec['manifest']['sha256'])
            manifest = pd.read_csv(root/spec['manifest']['path'], dtype=str, keep_default_na=False)
            for side, count in (('p', 2002), ('q', 2089)):
                item = spec['arrays']['test_'+side]
                path = root/item['path']
                pin(path, item['sha256'])
                if receipt['inputs_sha256'][str(path)] != item['sha256']:
                    raise ValueError('Score cache and abundance array provenance differ')
                raw = np.load(path, allow_pickle=False)
                if raw.shape != (count, 614) or not np.isfinite(raw).all() or (raw < 0).any():
                    raise ValueError('Invalid assessment composition')
                rows = manifest[manifest.role.eq('test') & manifest.distribution.eq(side.upper())].copy()
                rows['array_row'] = rows.array_row.astype(int)
                rows = rows.sort_values('array_row')
                if rows.array_row.tolist() != list(range(count)):
                    raise ValueError('Assessment row ordering differs')
                identity = rows[['sample_id', 'subject_id', 'source_file', 'source_row', 'row_sha256']].to_numpy(str)
                key = 'real' if side == 'p' else method
                if key in arrays:
                    if not np.array_equal(arrays[key], raw) or not np.array_equal(real_identity, identity):
                        raise ValueError('Real samples differ across generator comparisons')
                else:
                    arrays[key] = raw
                    identities[key] = identity
                    if side == 'p':
                        real_identity = identity
                columns = []
                for repeat in range(5):
                    folder = root/'final'/f'{repeat:02d}'
                    pin(folder/'COMPLETE.json', seal['assessment_completions'][f'final/{repeat:02d}/COMPLETE.json'])
                    fit = read(folder/'COMPLETE.json')
                    pin(folder/'predictions.npz', fit['outputs']['predictions.npz'])
                    with np.load(folder/'predictions.npz', allow_pickle=False) as saved:
                        columns.append(saved['test_'+side])
                values = np.column_stack(columns)
                if values.shape != (count, 5) or not np.array_equal(values, cache[method+'_'+side+'_scores']):
                    raise ValueError('Cached ensemble scores differ from completed predictions')
                if not np.isfinite(values).all() or (values < 0).any() or (values > 2).any():
                    raise ValueError('Invalid fitted RDR values')
                scores[method+'_'+side] = values.mean(axis=1)
    return arrays, scores, identities, inputs


def pcoa_from_distances(distances, threads=4, solver='iterative'):
    """Compute leading Lingoes coordinates, retaining the structural zero."""
    matrix = np.asarray(distances, dtype=np.float64).copy()
    n = len(matrix)
    if n < 4 or matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError('Invalid distance matrix')
    if (matrix < 0).any() or not np.allclose(matrix, matrix.T, atol=1e-14, rtol=0) or np.any(np.diag(matrix)):
        raise ValueError('Distances must be nonnegative, symmetric, and hollow')
    np.square(matrix, out=matrix)
    means = matrix.mean(axis=0)
    grand = float(means.mean())
    matrix -= means[None, :]
    matrix -= means[:, None]
    matrix += grand
    matrix *= -.5
    trace = float(np.trace(matrix))
    row_sum_residual = float(np.max(np.abs(matrix.sum(axis=1))))
    v0 = np.random.default_rng(SEED).normal(size=n)
    v0 -= v0.mean()
    v0 /= np.linalg.norm(v0)
    print(f'Centered {n} x {n} distances; solving leading and minimum eigenpairs', flush=True)
    with threadpool_limits(limits=threads):
        if solver == 'dense':
            eigenvalues, eigenvectors = eigh(matrix, check_finite=False)
            low, low_vector = eigenvalues[0], eigenvectors[:, :1]
            top, vectors = eigenvalues[-2:][::-1], eigenvectors[:, -2:][:, ::-1]
        else:
            top, vectors = eigsh(matrix, k=2, which='LA', tol=1e-11, v0=v0, maxiter=10000)
            order = np.argsort(top)[::-1]
            top, vectors = top[order], vectors[:, order]
            low_values, low_vector = eigsh(matrix, k=1, which='SA', tol=1e-11, v0=v0, maxiter=10000)
            low = float(low_values[0])
        eigen_residual = max(float(np.linalg.norm(matrix@vectors-vectors*top, axis=0).max()),
                             float(np.linalg.norm(matrix@low_vector-low_vector*low)))
    scale = max(1., float(np.abs(top).max()), abs(float(low)))
    if min(top) <= 0 or eigen_residual > 1e-8*scale or row_sum_residual > 1e-8*scale:
        raise ValueError('Ordination eigensolve failed its residual checks')
    if np.max(np.abs(vectors.sum(axis=0))) > 1e-6 or not np.allclose(vectors.T@vectors, np.eye(2), atol=1e-10):
        raise ValueError('Leading eigenvectors are not centered and orthonormal')
    correction = max(0., -float(low))
    corrected_trace = trace+(n-1)*correction
    if corrected_trace <= 0:
        raise ValueError('Nonpositive corrected inertia')
    for axis in range(2):
        anchor = int(np.argmax(np.abs(vectors[:, axis])))
        vectors[:, axis] *= 1 if vectors[anchor, axis] >= 0 else -1
    coords = vectors*np.sqrt(top+correction)
    diagnostics = dict(method='Joint Bray-Curtis PCoA with Lingoes correction', rows=n,
        exact_distances=True, eigensolver=solver, eigensolver_tolerance=1e-11, eigensolver_seed=SEED,
        full_spectrum_computed=False if solver == 'iterative' else True,
        raw_minimum_eigenvalue=float(low), raw_leading_eigenvalues=top.tolist(),
        raw_trace=trace, lingoes_constant=correction, corrected_total_inertia=corrected_trace,
        displayed_corrected_inertia_fractions=((top+correction)/corrected_trace).tolist(),
        eigen_residual_norm=eigen_residual, centering_row_sum_max_abs=row_sum_residual,
        coordinate_centering_max_abs=float(np.abs(coords.mean(axis=0)).max()),
        structural_zero_shifted=False, subsampling=False)
    # These row means permit an independent blockwise distance/eigenvector audit.
    return coords, diagnostics, dict(raw_squared_distance_row_means=means,
        raw_squared_distance_grand_mean=np.array(grand), leading_vectors=vectors,
        minimum_vector=low_vector[:, 0])


def plot(coords, scores, diagnostics, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fractions = 100*np.array(diagnostics['displayed_corrected_inertia_fractions'])
    limits = [(float(coords[:, k].min()), float(coords[:, k].max())) for k in range(2)]
    limits = [(a-.04*(b-a), b+.04*(b-a)) for a, b in limits]
    p = coords[:2002]
    q = {method: coords[2002+i*2089:2002+(i+1)*2089] for i, method in enumerate(METHODS)}

    def style(ax):
        ax.set(xlim=limits[0], ylim=limits[1], xlabel=f'PCo 1 ({fractions[0]:.3f}%)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.15)

    def save(fig, stem):
        for ext in ('png', 'pdf'):
            fig.savefig(output/(stem+'.'+ext), dpi=200, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharex=True, sharey=True, constrained_layout=True)
    order = np.random.default_rng(SEED).permutation(4091)
    colors = np.array(['#0072B2']*2002+['#D55E00']*2089)
    for index, (method, ax) in enumerate(zip(METHODS, axes)):
        xy = np.vstack([p, q[method]])
        ax.scatter(xy[order, 0], xy[order, 1], c=colors[order], s=9, alpha=.55, linewidths=0, rasterized=True)
        ax.set_title(SHORT[index])
        style(ax)
    axes[0].set_ylabel(f'PCo 2 ({fractions[1]:.3f}%)')
    handles = [Line2D([], [], marker='o', linestyle='', color=c, label=label, markersize=6)
               for c, label in [('#0072B2', 'Real P (2,002)'), ('#D55E00', 'Generated Q (2,089)')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 1.07), ncol=2, frameon=False)
    save(fig, 'pcoa_sources')

    cmap = LinearSegmentedColormap.from_list('rdr', ['#b35806', '#f7f7f7', '#0571b0'])
    norm = Normalize(0, 2)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True, constrained_layout=True)
    for index, method in enumerate(METHODS):
        for row, (side, xy) in enumerate([('p', p), ('q', q[method])]):
            ax = axes[row, index]
            sequence = np.random.default_rng(SEED+row).permutation(len(xy))
            artist = ax.scatter(xy[sequence, 0], xy[sequence, 1], c=scores[method+'_'+side][sequence],
                cmap=cmap, norm=norm, s=10, alpha=.8, linewidths=0, rasterized=True)
            style(ax)
            if row == 0:
                ax.set_title(SHORT[index])
        axes[0, 0].set_ylabel(f'Real P\nPCo 2 ({fractions[1]:.3f}%)')
        axes[1, 0].set_ylabel(f'Generated Q\nPCo 2 ({fractions[1]:.3f}%)')
    colorbar = fig.colorbar(artist, ax=axes, shrink=.8, pad=.02, ticks=[0, .5, 1, 1.5, 2])
    colorbar.set_label('Fitted RDR (mean of five fits)')
    save(fig, 'pcoa_rdr')


def build(private, public, threads):
    if private.exists() or public.exists():
        raise FileExistsError('Use fresh private and public ordination directories')
    arrays, scores, identities, inputs = load_assessment()
    values = np.concatenate([arrays[key].astype(np.float64) for key in ('real', *METHODS)])
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError('Empty compositions')
    values /= totals
    private.mkdir(parents=True)
    public.mkdir(parents=True)
    print(f'Computing exact Bray-Curtis distances for {len(values)} observations', flush=True)
    condensed = pdist(values, metric='braycurtis')
    distances = squareform(condensed)
    del condensed
    coords, diagnostics, audit = pcoa_from_distances(distances, threads)
    del distances
    diagnostics.update(group_order=['real', *METHODS], group_counts={'real': 2002, **{m: 2089 for m in METHODS}},
        taxa=614, input_closure='Each assessment row closed once in float64 for Bray-Curtis only',
        score_summary='Arithmetic mean of the five completed pair-specific final RDR fits',
        same_real_coordinates_in_every_panel=True, score_scale=[0, 2],
        assessment_only_descriptive=True, used_for_fitting_or_selection=False)
    np.savez_compressed(private/'ordination_private.npz', coordinates=coords, **audit,
                        **{name+'_scores': value for name, value in scores.items()},
                        **{name+'_identity': value for name, value in identities.items()})
    write(public/'ordination_diagnostics.json', diagnostics)
    plot(coords, scores, diagnostics, public)
    (public/'source').mkdir()
    shutil.copy2(__file__, public/'source/four_generator_pcoa.py')
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    write(private/'inputs.json', inputs)
    fractions = 100*np.array(diagnostics['displayed_corrected_inertia_fractions'])
    report = f'''# Four-generator Bray–Curtis PCoA

![Assessment samples colored by source](pcoa_sources.png)

One joint map includes all 10,358 assessment observations: 2,002 real samples
once, plus 2,089 generated samples from each retained generator. All panels
use the same coordinates and axes. No samples are removed or subsampled.

![The same map colored by fitted RDR](pcoa_rdr.png)

Rows show real P and generated Q separately. Colors use each P-versus-Q pair's
mean fitted RDR over its five completed final initializations, on the same
0–2 scale. The real coordinates are repeated unchanged across columns; their
colors differ because each generator has a separately selected evaluator.
Orange scores below one indicate fitted generated-mass excess; blue scores
above one indicate fitted generated-mass deficit. These are fitted scores,
with the calibration limitations documented in the main AGP report.

Bray–Curtis is computed after row closure in float64. Lingoes correction uses
c = {diagnostics['lingoes_constant']:.10g}. PCo 1 and PCo 2 explain
{fractions[0]:.4f}% and {fractions[1]:.4f}% of corrected inertia, respectively
({fractions.sum():.4f}% together). Projected overlap does not establish equality
of the full distributions, and this descriptive assessment map was not used
for generator fitting, checkpoint choice, RDR selection, or community discovery.

Exact pairwise distances are used. Deterministic iterative eigensolves estimate
the two largest and the smallest eigenpairs; residuals and corrected total
inertia are retained in [diagnostics](ordination_diagnostics.json). No full
eigenvalue spectrum or raw negative-inertia fraction is claimed.

[Source-colored PDF](pcoa_sources.pdf) · [RDR-colored PDF](pcoa_rdr.pdf) ·
[Rendering source](source/four_generator_pcoa.py)

The four sealed studies and their scores are unchanged. Input hashes, private
coordinates, and row identities stay under `{private}`; public outputs contain
figures, aggregate diagnostics, and the rendering source.
'''
    (public/'README.md').write_text(report)
    for name, digest in inputs.items():
        if sha(name) != digest:
            raise ValueError('Input changed while rendering: '+name)
    write(public/'manifest.json', dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        methods=list(METHODS), rows=10358, fitted_scores_reused=20, inputs_sha256=inputs,
        private_coordinates_sha256=sha(private/'ordination_private.npz'),
        files_sha256={str(p.relative_to(public)): sha(p) for p in sorted(public.rglob('*')) if p.is_file()}))
    write(private/'COMPLETE.json', dict(status='complete', public=str(public),
        manifest_sha256=sha(public/'manifest.json'), inputs_sha256=sha(private/'inputs.json'),
        coordinates_sha256=sha(private/'ordination_private.npz')))
    print(json.dumps({'status': 'complete', 'public': str(public), 'diagnostics': diagnostics}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--public', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    build(args.private.resolve(), args.public.resolve(), args.threads)
