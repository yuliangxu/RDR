#!/usr/bin/env python3
"""Verified final-score image panels and fixed-cell C.1/C.2 plots; no refitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/celeba-final-figures-mpl')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.calibration import bin_scores
from utils.model_selection import calibration_diagnostics


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sealed(path):
    if sha(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError(f'Sealed record changed: {path}')
    return read(path)


def same_cells(saved, recomputed):
    if len(saved) != len(recomputed):
        raise ValueError('Cell count changed')
    for original, new in zip(saved, recomputed):
        if original.keys() != new.keys():
            raise ValueError('Cell schema changed')
        for key in original:
            a, b = original[key], new[key]
            if a is None or b is None:
                equal = a is b
            else:
                equal = np.isclose(a, b, rtol=1e-10, atol=1e-12)
            if not equal:
                raise ValueError(f'Cell diagnostic differs: bin {original["bin"]}, {key}')


def verify_run(run):
    completion = sealed(run / 'COMPLETE.json')
    protocol = sealed(run / 'protocol.json')
    if completion['protocol_sha256'] != sha(run / 'protocol.json'):
        raise ValueError('Final protocol does not match completion')
    for name, digest in completion['files'].items():
        if sha(run / name) != digest:
            raise ValueError(f'Final report artifact changed: {name}')
    for name, digest in protocol['files'].items():
        if sha(run / name) != digest:
            raise ValueError(f'Frozen final source/input changed: {name}')
    for name in ('utils/calibration.py', 'utils/model_selection.py'):
        if sha(ROOT / name) != protocol['files']['source/' + name]:
            raise ValueError(f'Use the unchanged calibration implementation: {name}')
    data_record = sealed(run / 'evaluation_data.json')
    if (sha(run / 'evaluation_data.json') != completion['evaluation_data_sha256']
            or sha(run / 'data/evaluation_metadata.json') != data_record['metadata_sha256']
            or sha(run / 'freeze.json') != data_record['freeze_sha256']):
        raise ValueError('Final data lineage changed')
    metadata = read(run / 'data/evaluation_metadata.json')
    results = {}
    input_hashes = {}
    for task in protocol['tasks']:
        directory = run / 'evaluation' / task['representation'] / task['branch'] / task['candidate'] / f"repeat_{task['repeat']:02d}"
        receipt = read(directory / 'COMPLETE.json')
        expected_identity = [task[k] for k in ('representation', 'branch', 'candidate', 'repeat')]
        if (receipt['task_identity'] != expected_identity
                or receipt['protocol_sha256'] != sha(run / 'protocol.json')
                or set(receipt['files']) != {'predictions.npz', 'metrics.json', 'cells.json'}):
            raise ValueError('Evaluation task receipt differs')
        for name, digest in receipt['files'].items():
            path = directory / name
            if sha(path) != digest:
                raise ValueError(f'Evaluation artifact changed: {path}')
            input_hashes[str(path)] = digest
        with np.load(directory / 'predictions.npz', allow_pickle=False) as archive:
            predictions = {key: archive[key] for key in archive.files}
        for role in ('test_calibration', 'test_evaluation'):
            for side, source in (('p', 'real'), ('q', task['branch'])):
                if len(predictions[f'{role}_{side}']) != metadata['roles'][role][source]['count']:
                    raise ValueError('Prediction/manifest count differs')
        _, cells = calibration_diagnostics(*[predictions[f'{role}_{side}']
            for role in ('test_calibration', 'test_evaluation') for side in ('p', 'q')],
            alpha=protocol['config']['ci_alpha'], bins=protocol['config']['bins'])
        same_cells(read(directory / 'cells.json'), cells)
        key = (task['representation'], task['branch'], task['repeat'])
        if key in results:
            raise ValueError('Duplicate selected-model evaluation')
        results[key] = dict(task=task, cells=cells, predictions=predictions)
    if len(results) != completion['evaluations'] or len(results) != 20:
        raise ValueError('Expected all 20 completed evaluations')
    return protocol, metadata, results, input_hashes


def save(fig, output, stem):
    fig.savefig(output / f'{stem}.png', dpi=160, bbox_inches='tight')
    fig.savefig(output / f'{stem}.pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)
    return stem


def plot_ci(output, level, repeat, results, alpha):
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1, 1]}, layout='constrained')
    edges = np.linspace(0, 2, 21)
    centers = (edges[:-1] + edges[1:]) / 2
    for column, branch in enumerate(('lower', 'upper')):
        cells = results[level, branch, repeat]['cells']
        ax, calibration_ax, evaluation_ax = axes[:, column]
        low = np.array([r['c2_lower'] for r in cells])
        high = np.array([r['c2_upper'] for r in cells])
        ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step='post',
                        color='#8cb9d5', alpha=.6, label='C.2: simultaneous across cells')
        ax.stairs(low, edges, color='#28739c', lw=.7)
        ax.stairs(high, edges, color='#28739c', lw=.7)
        available = np.array([not r['c1_unavailable'] for r in cells])
        estimate = np.array([np.nan if r['calibrated_rdr'] is None else r['calibrated_rdr'] for r in cells])
        neural = np.array([np.nan if r['neural_mean'] is None else r['neural_mean'] for r in cells])
        ax.errorbar(centers[available], estimate[available],
                    yerr=[estimate[available]-np.array([r['c1_lower'] for r in cells])[available],
                          np.array([r['c1_upper'] for r in cells])[available]-estimate[available]],
                    fmt='none', ecolor='#b75d23', capsize=3, lw=1.5, label='C.1: marginal, when available')
        ax.stairs(estimate, edges, baseline=None, color='#152b3c', label='Calibration cell estimate')
        ax.scatter(centers, neural, s=20, marker='o', facecolors='none', edgecolors='#b42363',
                   label='Evaluation neural cell mean', zorder=4)
        if np.any(~available):
            ax.scatter(centers[~available], np.full(np.sum(~available), -.04), marker='x', color='#b75d23',
                       label='C.1 unavailable', clip_on=False, zorder=5)
        ax.plot([0, 2], [0, 2], ':', color='.5', lw=.8)
        ax.axhline(1, color='.5', ls='--', lw=.8)
        ax.set(title=f'P versus Q_{"l" if branch == "lower" else "u"}',
               ylim=(-.08, 2.05), xlim=(0, 2), ylabel='Cell-average RDR')
        ax.grid(alpha=.15)
        for count_ax, prefix, title in ((calibration_ax, 'cal', 'Calibration count'),
                                        (evaluation_ax, 'eval', 'Evaluation count')):
            for offset, side, color, label in ((-.018, 'p', '#367c98', 'P'), (.018, 'q', '#c48a39', 'Q')):
                count_ax.bar(centers+offset, [r[f'{prefix}_count_{side}'] for r in cells],
                             width=.033, color=color, label=label)
            count_ax.set_yscale('symlog', linthresh=1)
            count_ax.set(ylabel=title)
            count_ax.grid(axis='y', alpha=.15)
            count_ax.legend(fontsize=8, loc='upper center', ncol=2, frameon=False)
        evaluation_ax.set(xlabel='Fixed neural-score bin on [0, 2]')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f'CelebA {level}: selected JS model, repeat {repeat:02d}\n'
                 f'Nominal {100*(1-alpha):g}% cell-average intervals; retrospective IID-image reference\n'
                 'C.2 simultaneity is within each fitted comparison, not across repetitions or models', fontsize=11)
    return save(fig, output, f'ci_{level}_repeat_{repeat:02d}')


def select_rows(scores, low, high, include_high, maximum, seed, target=None):
    scores = np.asarray(scores)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 2)):
        raise ValueError('Invalid RDR scores')
    pool = np.flatnonzero((scores >= low) & ((scores <= high) if include_high else (scores < high)))
    rng = np.random.default_rng(seed)
    if target is None:
        selected = np.sort(rng.choice(pool, min(maximum, len(pool)), replace=False))
    else:
        # Seeded tie ordering; nearest targets reproduce the historical panel rule.
        randomized = rng.permutation(pool)
        selected = randomized[np.argsort(np.abs(scores[randomized]-target), kind='stable')[:maximum]]
    return selected, len(pool)


def montage(images, indices, columns, capacity):
    rows = (capacity + columns - 1) // columns
    canvas = np.full((rows * 66 + 2, columns * 66 + 2, 3), 245, dtype=np.uint8)
    for n, index in enumerate(indices):
        row, column = divmod(n, columns)
        canvas[2+row*66:66+row*66, 2+column*66:66+column*66] = images[index].transpose(1, 2, 0)
    return canvas


def image_panels(output, level, branch, result, images, manifests, seed):
    scores = result['predictions']
    image_records, bin_records = [], []
    stems = []
    for style in ('historical_ranges', 'all_bins'):
        specs = [(0., .5, False, 0.), (.5, 1.5, False, 1.), (1.5, 2., True, 2.)] if style == 'historical_ranges' else [
            (i/10, (i+1)/10, i == 19, None) for i in range(20)]
        # Use exactly the CI edges, including floating-point endpoint convention.
        if style == 'all_bins':
            edges = np.linspace(0, 2, 21)
            specs = [(edges[i], edges[i+1], i == 19, None) for i in range(20)]
        if style == 'historical_ranges':
            fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
            capacity, columns = 40, 10
        else:
            fig, axes = plt.subplots(20, 2, figsize=(10, 23), layout='constrained')
            capacity, columns = 4, 4
        for j, (low, high, inclusive, target) in enumerate(specs):
            for side_number, (side, source) in enumerate((('p', 'real'), ('q', branch))):
                values = scores[f'test_evaluation_{side}']
                selected, size = select_rows(values, low, high, inclusive, capacity,
                                             seed + j * 17 + side_number, target)
                ax = axes[side_number, j] if style == 'historical_ranges' else axes[j, side_number]
                ax.imshow(montage(images[source], selected, columns, capacity))
                interval = f'[{low:g}, {high:g}{"]" if inclusive else ")"}'
                label = 'P (real)' if side == 'p' else f'Q_{"l" if branch == "lower" else "u"} (generated)'
                ax.set_title(f'{label}: {interval} | {size:,} available; {len(selected)} shown', fontsize=10 if style == 'historical_ranges' else 8)
                if not size:
                    ax.text(.5, .5, 'No evaluation images in this bin', transform=ax.transAxes,
                            ha='center', va='center', fontsize=9, color='.35')
                ax.axis('off')
                base = dict(representation=level, branch=branch, repeat=0, panel=style, bin=j,
                            left=float(low), right=float(high), right_inclusive=bool(inclusive),
                            source=source, available_count=size, shown_count=len(selected),
                            target=target, sampling_seed=seed + j*17 + side_number)
                bin_records.append(base)
                for rank, index in enumerate(selected):
                    row = manifests[source].iloc[int(index)]
                    image_records.append({**base, 'display_rank': rank, 'manifest_row': int(index),
                                          'source_id': str(row.source_id), 'rdr': float(values[index])})
        rule = ('Up to 40 images nearest score 0, 1, or 2 within each range; seeded ties' if style == 'historical_ranges'
                else 'Up to 4 uniformly sampled images per source per fixed bin; empty bins retained')
        fig.suptitle(f'CelebA {level}: P versus Q_{"l" if branch == "lower" else "u"}; selected JS model, repeat 00\n'
                     f'{rule}\nFinal evaluation images only; example counts do not represent population prevalence', fontsize=12)
        stems.append(save(fig, output, f'images_{level}_{branch}_{style}'))
    return stems, image_records, bin_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026100301)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new figure directory to preserve earlier exports')
    protocol, metadata, results, hashes = verify_run(run)
    print('Verified and independently recomputed all 20 final cell tables.', flush=True)
    output.mkdir(parents=True)
    images, manifests = {}, {}
    for source in ('real', 'lower', 'upper'):
        receipt = metadata['arrays']['pixel']['test_evaluation'][source]
        path = run / receipt['path']
        if sha(path) != receipt['sha256']:
            raise ValueError('Final image array changed')
        values = np.load(path, mmap_mode='r', allow_pickle=False)
        if list(values.shape) != receipt['shape'] or values.dtype != np.uint8:
            raise ValueError('Final image schema changed')
        item = metadata['roles']['test_evaluation'][source]
        manifest_path = run / item['path']
        if sha(manifest_path) != item['receipt']['sha256'] or receipt['manifest_sha256'] != sha(manifest_path):
            raise ValueError('Image rows do not match the locked manifest')
        manifests[source] = pd.read_csv(manifest_path)
        images[source] = values
        if len(values) != len(manifests[source]):
            raise ValueError('Image/manifest lengths differ')
        hashes[str(path)] = receipt['sha256']
        hashes[str(manifest_path)] = item['receipt']['sha256']
    ci_stems = []
    for level in ('feature', 'pixel'):
        for repeat in range(protocol['config']['repeats']):
            ci_stems.append(plot_ci(output, level, repeat, results, protocol['config']['ci_alpha']))
    image_stems, selected_rows, counts = [], [], []
    for i, (level, branch) in enumerate((('feature','lower'), ('feature','upper'), ('pixel','lower'), ('pixel','upper'))):
        stems, records, inventory = image_panels(output, level, branch, results[level, branch, 0], images, manifests, args.seed+i*1000)
        image_stems.extend(stems)
        selected_rows.extend(records)
        counts.extend(inventory)
    pd.DataFrame(selected_rows).to_csv(output / 'selected_images.csv', index=False)
    pd.DataFrame(counts).to_csv(output / 'image_bin_counts.csv', index=False)
    pd.DataFrame([{**row, 'representation': level, 'branch': branch, 'repeat': repeat}
                  for (level, branch, repeat), result in results.items() for row in result['cells']]).to_csv(output / 'cell_intervals.csv', index=False)
    lines = ['# Selected CelebA models: image examples and calibration intervals', '',
             '**Complete: 20 verified cell tables, 10 CI figures, and 8 image-panel figures (PNG and PDF).**', '',
             'The selected models use JS with slope 0.5 (feature) or 2 (pixel). '
             'Repeat 00 is a fixed illustration choice, not the best-performing repetition. '
             'All five repetitions have separate CI figures; intervals are not averaged across different fitted score partitions.', '',
             'C.1 intervals are nominal 95% marginal intervals; C.2 bands are nominal 95% simultaneous across '
             'the 20 cells of one fixed fitted comparison, not across all models or repetitions. '
             'Calibration uses 19,906 P and 20,000 Q images; evaluation uses 19,923 P and 20,000 Q. '
             'CIs target population cell-average RDR. They are not confidence intervals for individual images '
             'or for the plotted evaluation neural means. Historical data reuse and identity clustering '
             'retain the retrospective, nominal IID-image interpretation.', '',
             'Empty calibration cells have C.2 [0,2] and no cell estimate; unavailable C.1 intervals are marked '
             'with crosses. Count panels show the sparsity. Empty evaluation bins retain explicit empty image panels.', '',
             '## Primary illustrations: repeat 00', '']
    for level in ('feature', 'pixel'):
        stem = f'ci_{level}_repeat_00'
        lines += [f'### {level.capitalize()} CIs', '', f'![{level} CIs]({stem}.png)', '', f'[Paper PDF]({stem}.pdf)', '']
    for level, branch in (('feature','lower'), ('feature','upper'), ('pixel','lower'), ('pixel','upper')):
        stem = f'images_{level}_{branch}_historical_ranges'
        fine = f'images_{level}_{branch}_all_bins'
        lines += [f'### {level.capitalize()} / P versus Q_{branch}', '',
                  f'![Score-range examples]({stem}.png)', '',
                  f'[Historical-format PDF]({stem}.pdf) · [All 20 bins PNG]({fine}.png) · [All 20 bins PDF]({fine}.pdf)', '']
    lines += ['## All CI repetitions', '']
    lines += [f'- [{stem}]({stem}.pdf)' for stem in ci_stems]
    lines += ['', '## Sampling and reproduction', '',
              'Historical-format panels use [0,0.5), [0.5,1.5), [1.5,2] and up to 40 images '
              'nearest 0, 1, or 2 in each source/range, with seeded tie breaking. These broad illustration '
              'ranges are distinct from the 20 equal-width CI bins. All-bin panels sample up to four '
              'images uniformly without replacement per source/cell. No images are borrowed from '
              'neighboring bins or calibration data. Available/shown counts are explicit; examples are '
              'not prevalence estimates. Both formats use the same final evaluation pixels and score ordering.', '',
              '[Selected image IDs, scores and seeds](selected_images.csv) · [All bin counts](image_bin_counts.csv) · [All 400 cell intervals](cell_intervals.csv)', '',
              'Render into a new output directory using the saved source:', '', '```bash',
              f'OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python3 -B {output}/source/experiments/CelebA/final_figures.py \\',
              f'  --run {run} --output /tmp/celeba-final-figures-replay --seed {args.seed}', '```', '',
              '[Original final numeric assessment](../RESULTS.md) is unchanged; this supplement has a separate completion receipt.', '']
    (output / 'FIGURES.md').write_text('\n'.join(lines))
    sources = {}
    for name in ('experiments/CelebA/final_figures.py','utils/calibration.py','utils/model_selection.py'):
        target = output / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
        sources[name] = sha(target)
    artifacts = {str(p.relative_to(output)):sha(p) for p in output.iterdir() if p.is_file()}
    record = dict(complete=True, source_run=str(run), final_completion_sha256=sha(run/'COMPLETE.json'),
                  seed=args.seed, illustration_repeat=0, verified_evaluations=20,
                  ci_figures=len(ci_stems), image_figures=len(image_stems), sources=sources,
                  input_hashes=hashes, artifacts=artifacts)
    path = output / 'COMPLETE.json'
    path.write_text(json.dumps(record, indent=2)+'\n')
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    print(json.dumps({k:record[k] for k in ('complete','verified_evaluations','ci_figures','image_figures')}), flush=True)


if __name__ == '__main__':
    main()
