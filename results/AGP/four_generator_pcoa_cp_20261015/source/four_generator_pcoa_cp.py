#!/usr/bin/env python3
"""Color the existing AGP PCoA by saved fit-0 C.2 CP score-cell intervals.

No models, intervals, or coordinates are refitted. Public outputs contain
figures and aggregates only; observation identities remain in existing inputs.
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
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import pandas as pd

BASE = Path('/cwork/yx306/RDR/AGP')
REPOSITORY = Path(__file__).resolve().parents[2]
METHODS = ('bernoulli_mixture_icfm', 'autoregressive_icfm', 'midasim_mask_icfm', 'midasim')
SHORT = ('Bernoulli mixture\n+ ICFM', 'NADE masks\n+ ICFM', 'MIDASim masks\n+ ICFM', 'Full MIDASim')
ROOTS = (BASE/'mask_rdr_20261012/analyses'/METHODS[0],
         BASE/'mask_rdr_20261012/analyses'/METHODS[1],
         BASE/'icfm_zero_20261007/analysis', BASE/'midasim_full_20261004/analysis')
CATEGORIES = ('entirely_below_1', 'contains_1', 'entirely_above_1')
COLORS = ('#b35806', '#999999', '#0571b0')
IDENTITY = ['sample_id', 'subject_id', 'source_file', 'source_row', 'row_sha256']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def classify(lower, upper):
    """Closed intervals touching 1 belong to the contains-1 category."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    if lower.shape != upper.shape or not np.isfinite([lower, upper]).all():
        raise ValueError('Invalid interval bounds')
    if (lower < 0).any() or (upper > 2).any() or (lower > upper).any():
        raise ValueError('Invalid RDR intervals')
    return np.where(upper < 1, 0, np.where(lower > 1, 2, 1)).astype(np.uint8)


def load(private, public):
    inputs = {}

    def pin(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError('Changed frozen input: '+str(path))
        inputs[str(path.resolve())] = digest

    seal = read(private/'COMPLETE.json')
    if seal['status'] != 'complete':
        raise ValueError('Incomplete ordination')
    pin(private/'COMPLETE.json')
    pin(public/'manifest.json', seal['manifest_sha256'])
    manifest = read(public/'manifest.json')
    if manifest['status'] != 'complete' or manifest['methods'] != list(METHODS):
        raise ValueError('Wrong ordination methods')
    for name, digest in manifest['files_sha256'].items():
        pin(public/name, digest)
    pin(private/'ordination_private.npz', manifest['private_coordinates_sha256'])
    if seal['coordinates_sha256'] != manifest['private_coordinates_sha256']:
        raise ValueError('Ordination receipts disagree')
    diagnostics = read(public/'ordination_diagnostics.json')
    if diagnostics['group_order'] != ['real', *METHODS]:
        raise ValueError('Wrong coordinate cohort order')
    categories, counts, cells_out, assignment_hashes = {}, [], [], {}
    with np.load(private/'ordination_private.npz', allow_pickle=False) as saved:
        coords = saved['coordinates'].copy()
        if coords.shape != (10358, 2) or not np.isfinite(coords).all():
            raise ValueError('Invalid saved coordinates')
        for method, root in zip(METHODS, ROOTS):
            folder = root/'final/00'
            pin(folder/'COMPLETE.json', manifest['inputs_sha256'][str(folder/'COMPLETE.json')])
            complete = read(folder/'COMPLETE.json')
            if complete['status'] != 'complete' or complete['stage'] != 'final' or complete['task']['repeat'] != 0:
                raise ValueError('Expected completed final fit 0')
            for name in ('cells.csv', 'metrics.json', 'predictions.npz', 'scores.csv'):
                pin(folder/name, complete['outputs'][name])
            if complete['outputs']['predictions.npz'] != manifest['inputs_sha256'][str(folder/'predictions.npz')]:
                raise ValueError('Scores differ from original ordination inputs')
            metrics = read(folder/'metrics.json')
            if (metrics['alpha'], metrics['bins'], metrics['repeat']) != (.05, 20, 0):
                raise ValueError('Unexpected calibration configuration')
            if (metrics['n_cal_p'], metrics['n_cal_q']) != (2002, 2089) or not metrics['same_sample_calibration_evaluation']:
                raise ValueError('Unexpected calibration sample roles')
            table = pd.read_csv(folder/'cells.csv', float_precision='round_trip').sort_values('bin')
            edges = np.linspace(0., 2., 21)
            if table.bin.tolist() != list(range(20)) or not np.array_equal(table.left, edges[:-1]) or not np.array_equal(table.right, edges[1:]):
                raise ValueError('Expected original fixed-width score cells')
            cell_categories = classify(table.c2_lower, table.c2_upper)
            cells_out.append(table.assign(method=method, repeat=0,
                category=np.array(CATEGORIES)[cell_categories]))
            identities = pd.read_csv(folder/'scores.csv', dtype=str, keep_default_na=False)
            with np.load(folder/'predictions.npz', allow_pickle=False) as prediction:
                for side, count in (('p', 2002), ('q', 2089)):
                    rows = identities[identities.role.eq('test') & identities.distribution.eq(side.upper())].copy()
                    rows['array_row'] = rows.array_row.astype(int)
                    rows = rows.sort_values('array_row')
                    key = 'real' if side == 'p' else method
                    if rows.array_row.tolist() != list(range(count)) or not np.array_equal(rows[IDENTITY].to_numpy(str), saved[key+'_identity']):
                        raise ValueError('Sample identities differ from the saved coordinates')
                    scores = prediction['test_'+side]
                    if scores.shape != (count,) or not np.isfinite(scores).all() or (scores < 0).any() or (scores > 2).any():
                        raise ValueError('Invalid RDR scores')
                    if not np.array_equal(rows.rdr.astype(float).to_numpy(), scores):
                        raise ValueError('Score table and prediction ordering differ')
                    bins = np.minimum(np.searchsorted(edges, scores, side='right')-1, 19)
                    histogram = np.bincount(bins, minlength=20)
                    if not np.array_equal(histogram, table['cal_count_'+side]) or not np.array_equal(histogram, table['eval_count_'+side]):
                        raise ValueError('Prediction-to-calibration-cell counts differ')
                    labels = cell_categories[bins]
                    categories[method+'_'+side] = labels
                    assignment_hashes[method+'_'+side] = hashlib.sha256(labels.tobytes()).hexdigest()
                    for number, name in enumerate(CATEGORIES):
                        total = int(np.sum(labels == number))
                        counts.append(dict(method=method, distribution=side.upper(), repeat=0,
                            category=name, count=total, fraction=total/count))
    return coords, diagnostics, categories, pd.DataFrame(counts), pd.concat(cells_out), inputs, assignment_hashes


def plot(coords, diagnostics, categories, output):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fractions = 100*np.array(diagnostics['displayed_corrected_inertia_fractions'])
    limits = [(float(coords[:, k].min()), float(coords[:, k].max())) for k in range(2)]
    limits = [(a-.04*(b-a), b+.04*(b-a)) for a, b in limits]
    cmap, norm = ListedColormap(COLORS), BoundaryNorm([-.5, .5, 1.5, 2.5], 3)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), sharex=True, sharey=True, constrained_layout=True)
    for index, method in enumerate(METHODS):
        for row, (side, xy) in enumerate((('p', coords[:2002]),
                ('q', coords[2002+index*2089:2002+(index+1)*2089]))):
            ax = axes[row, index]
            sequence = np.random.default_rng(20261014+row).permutation(len(xy))
            artist = ax.scatter(xy[sequence, 0], xy[sequence, 1],
                c=categories[method+'_'+side][sequence], cmap=cmap, norm=norm,
                s=10, alpha=.85, linewidths=0, rasterized=True)
            ax.set(xlim=limits[0], ylim=limits[1], xlabel=f'PCo 1 ({fractions[0]:.3f}%)')
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.15)
            if row == 0:
                ax.set_title(SHORT[index])
    axes[0, 0].set_ylabel(f'Real P\nPCo 2 ({fractions[1]:.3f}%)')
    axes[1, 0].set_ylabel(f'Generated Q\nPCo 2 ({fractions[1]:.3f}%)')
    colorbar = fig.colorbar(artist, ax=axes, orientation='horizontal', shrink=.86,
        pad=.06, fraction=.05, ticks=[0, 1, 2])
    colorbar.ax.set_xticklabels(['Entirely below 1\nGenerated-mass excess',
        'Contains 1\nNo direction resolved', 'Entirely above 1\nGenerated-mass deficit'])
    colorbar.set_label('Score-cell RDR: nominal 95% C.2 Clopper–Pearson intervals · final fit 0', labelpad=9)
    for extension in ('png', 'pdf'):
        fig.savefig(output/('pcoa_cp_categories.'+extension), dpi=200, bbox_inches='tight')
    plt.close(fig)


def build(private, public, output):
    if output.exists():
        raise FileExistsError('Use a fresh output directory')
    coords, diagnostics, categories, counts, cells, inputs, assignment_hashes = load(private, public)
    inputs[str(Path(__file__).resolve())] = sha(__file__)
    output.mkdir(parents=True)
    plot(coords, diagnostics, categories, output)
    counts.to_csv(output/'category_counts.csv', index=False)
    cells.to_csv(output/'calibration_cells.csv', index=False)
    (output/'source').mkdir()
    shutil.copy2(__file__, output/'source/four_generator_pcoa_cp.py')
    (output/'README.md').write_text('''# Four-generator PCoA colored by calibration CP intervals

![Discrete CP interval categories](pcoa_cp_categories.png)

This adds categorical colors to the existing joint Bray–Curtis PCoA. All
10,358 observations, coordinates, axis limits, and the Lingoes correction are
unchanged. Real P is the top row; generated Q is the bottom row.

Each point uses its pair-specific **final fit 0** score to enter one of the
original 20 fixed-width bins on [0,2]. The saved bin's **C.2 Clopper–Pearson
interval** determines the color: orange if its upper bound is below 1, blue
if its lower bound is above 1, and gray if it contains 1 (including endpoints).
Orange denotes generated-mass excess in that score cell; blue denotes a
deficit. Gray means the interval does not resolve a direction, not equality.

These are the same nominal 95% intervals used in the fit-0 calibration panel:
the CP tail allocation is 0.05/(4*20), simultaneous over 20 cells within one
fit, not across methods or initializations. They describe cell-average RDR
2P(A)/(P(A)+Q(A)), not confidence intervals for individual true RDR values.
The assessment observations also supplied the cell counts; interpretation
remains retrospective and nominal under the original sampling limitations.
No intervals or models were refitted and no interval bounds were averaged.
The preceding continuous-color plot instead uses the five-fit mean score.

The displayed plane contains only 0.3121% of Lingoes-corrected inertia.
Projected overlap does not establish equality of the full distributions.

[PDF](pcoa_cp_categories.pdf) · [Category counts](category_counts.csv) ·
[Saved calibration cells](calibration_cells.csv) ·
[Rendering source](source/four_generator_pcoa_cp.py) · [Manifest](manifest.json)

The manifest pins input hashes and SHA-256 hashes of the eight ordered uint8
category vectors (0 below, 1 contains, 2 above). Row-level coordinates and
identities remain in the original private ordination archive.
''')
    for name, digest in inputs.items():
        if sha(name) != digest:
            raise ValueError('Input changed while rendering: '+name)
    receipt = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
        methods=list(METHODS), fit_repeat=0, alpha=.05, bins=20,
        cp_tail_probability=.05/80, unique_observations=10358,
        displayed_observations=4*(2002+2089), coordinate_refit=False,
        interval_refit=False, input_ordination=str(private), categories=list(CATEGORIES),
        category_vector_sha256=assignment_hashes, inputs_sha256=inputs,
        files_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps(dict(status='complete', output=str(output), panels=8, calibration_cells=80)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private', type=Path, default=BASE/'four_generator_pcoa_20261014')
    parser.add_argument('--ordination-public', type=Path,
        default=REPOSITORY/'results/AGP/four_generator_pcoa_20261014')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.private.resolve(), args.ordination_public.resolve(), args.output.resolve())
