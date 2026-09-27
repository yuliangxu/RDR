#!/usr/bin/env python3
"""Append adaptive residual-MLP CIs to the sealed equal-width AGP report."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils.calibration import bin_scores, calibrate_merged_counts

DEFAULT_REPORT = Path('/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def plot(out, cells, elementary, protocol):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/agp-residual-adaptive-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    edges = np.r_[cells.left, cells.right.iloc[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, method in zip(axes[:2], ('c2', 'c1')):
        low, high = (cells[f'{method}_{bound}'].to_numpy().copy() for bound in ('lower', 'upper'))
        ax.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(low, 2), np.repeat(high, 2),
                        color='#88b9d9', alpha=.65, label='Interval')
        estimate = cells.estimate.to_numpy().copy()
        estimate[cells.c1_empty.to_numpy()] = np.nan
        ax.stairs(estimate, edges, color='#182f44', label='Cell estimate')
        ax.plot([0, 2], [0, 2], ':', color='.5', lw=.8)
        ax.axhline(1, ls='--', color='.5', lw=.8)
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel='Frozen residual-MLP RDR score',
               ylabel='Cell-average RDR', title='C.2 simultaneous' if method == 'c2'
               else 'C.1 joint multiplier bootstrap')
        ax.legend(fontsize=8)
    centers = (elementary.left + elementary.right) / 2
    width = (elementary.right - elementary.left) * .4
    axes[2].bar(centers - width / 2, elementary.k_p, width=width, label='P: observed AGP')
    axes[2].bar(centers + width / 2, elementary.k_q, width=width, label='Q: ICFM')
    for i, edge in enumerate(edges[1:-1]):
        axes[2].axvline(edge, color='#922b21', ls='--', lw=1,
                        label='Adaptive boundary' if i == 0 else None)
    axes[2].set(xlim=(0, 2), yscale='log', xlabel='Frozen residual-MLP RDR score',
                ylabel='Count in elementary bin (log)',
                title=f'Full test: {protocol["n"]:,} P / {protocol["m"]:,} Q')
    axes[2].legend(fontsize=8)
    fig.suptitle('AGP real vs ICFM: residual MLP, nominal 95% adaptive cell intervals\n'
                 f'{protocol["run_id"]}; h={protocol["min_count"]}, {len(cells)} regions, '
                 f'{protocol["n_candidates"]} candidates protected', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / 'residual_mlp_adaptive_ci.png', dpi=180)
    fig.savefig(out / 'residual_mlp_adaptive_ci.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--min-count', type=int, default=20)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    root = args.report_dir.resolve()
    allowed = Path('/tmp') if args.preflight else Path('/cwork/yx306/RDR')
    if not root.is_relative_to(allowed) or not root.is_dir():
        raise ValueError(f'Use an existing sealed report directory below {allowed}')
    name = f'adaptive_h{args.min_count}'
    out = root / name
    backup = root / 'report_revisions' / f'before_{name}'
    if out.exists() or backup.exists():
        raise ValueError('This adaptive analysis already exists; refusing to overwrite it')
    completed = json.loads((root / 'result.json').read_text())
    if completed['status'] != 'complete':
        raise ValueError('Original report is incomplete')
    for filename, digest in completed['outputs'].items():
        if sha(root / filename) != digest:
            raise ValueError(f'Original report hash mismatch: {filename}')
    original = json.loads((root / 'protocol.json').read_text())
    if original['merged'] or original['run_id'] != completed['run_id']:
        raise ValueError('Expected the original residual equal-width report')
    old_report = (root / 'report.md').read_text()
    if old_report.count('## Reproduction') != 1:
        raise ValueError('Expected one reproduction section')
    scores = pd.read_csv(root / 'test_scores.csv', keep_default_na=False,
                         dtype={'sample_id': str, 'subject_id': str, 'row_sha256': str},
                         float_precision='round_trip')
    elementary = pd.read_csv(root / 'cell_intervals.csv', float_precision='round_trip')
    edges = np.asarray(original['edges'])
    if set(scores.role) != {'test'} or set(scores.distribution) != {'P', 'Q'}:
        raise ValueError('Unexpected scoring roles or distributions')
    if not np.array_equal(bin_scores(scores.rdr.to_numpy(), edges), scores.bin_index):
        raise ValueError('Score bins disagree with frozen boundaries')
    kp, kq = (np.bincount(scores.loc[scores.distribution.eq(d), 'bin_index'], minlength=len(edges)-1)
              for d in ('P', 'Q'))
    n, m = int(kp.sum()), int(kq.sum())
    if (n, m) != (original['n_p'], original['n_q']):
        raise ValueError('P/Q counts changed')
    if not np.array_equal(kp, elementary.k_p) or not np.array_equal(kq, elementary.k_q):
        raise ValueError('Original elementary counts disagree with frozen scores')
    calibrated = calibrate_merged_counts(kp, kq, n, m, alpha=.05,
        min_count=args.min_count, bootstrap_repetitions=10000, seed=20260920)
    candidates = pd.DataFrame(calibrated['candidates'])
    candidates.insert(0, 'candidate_index', np.arange(len(candidates)))
    candidates['left'] = edges[candidates.start]
    candidates['right'] = edges[candidates.stop]
    candidates['right_closed'] = candidates.stop.eq(len(edges)-1)
    candidates['n_p_calibration'], candidates['n_q_calibration'] = n, m
    cells = candidates.iloc[calibrated['selected_indices']].copy()
    cells.insert(0, 'region_index', np.arange(len(cells)))
    lookup = pd.DataFrame({'bin_index': np.arange(len(kp)), 'left': edges[:-1],
        'right': edges[1:], 'k_p': kp, 'k_q': kq,
        'region_index': calibrated['elementary_to_merged']})
    attachments = scores.copy()
    attachments['region_index'] = calibrated['elementary_to_merged'][scores.bin_index.to_numpy()]
    attachments = attachments.merge(cells, on='region_index', validate='many_to_one')
    protocol = {**calibrated['metadata'], 'analysis': 'agp_residual_mlp_adaptive_ci',
        'run_id': original['run_id'], 'epoch': original['epoch'],
        'model_sha256': original['model_sha256'], 'scores_sha256': sha(root / 'test_scores.csv'),
        'original_result_sha256': sha(root / 'result.json'), 'edges': edges.tolist(),
        'selection': original['selection'], 'model_retrained': False, 'retrospective': True,
        'merge_rule': 'left-to-right; close after both P and Q reach h; merge remaining tail backward',
        'c1': 'asymptotic simultaneous joint Gaussian multiplier over all positive-SE candidates',
        'c2': 'simultaneous Clopper-Pearson probability bounds with alpha/(4J) per tail',
        'target': '2P(A)/(P(A)+Q(A)) for selected score regions; cell-average RDR',
        'sampling': original['sampling']}

    out.mkdir()
    cells.to_csv(out / 'cell_intervals.csv', index=False)
    candidates.to_csv(out / 'candidate_intervals.csv', index=False)
    lookup.to_csv(out / 'elementary_to_merged.csv', index=False)
    attachments.to_csv(out / 'test_intervals.csv', index=False)
    np.save(out / 'bootstrap_maxima.npy', calibrated['bootstrap_maxima'])
    save_json(out / 'protocol.json', protocol)
    for filename in ('experiments/AGP/AGP_residual_adaptive_ci.py', 'utils/calibration.py', 'utils/__init__.py'):
        target = out / 'source' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / filename, target)
    plot(out, cells, elementary, protocol)

    section = [f'## Data-adaptive confidence intervals, h={args.min_count}', '',
        f'![Residual MLP adaptive intervals]({name}/residual_mlp_adaptive_ci.png)', '',
        f'The same frozen residual MLP and {n:,} P / {m:,} Q scores give **{len(cells)} adaptive regions**. '
        f'Starting from the same 20 elementary bins, merge adjacent bins from left to right until each region '
        f'has at least **{args.min_count} P and {args.min_count} Q** observations; merge any insufficient final tail '
        'into the preceding region. The histogram retains the elementary bins, with dashed lines at the selected boundaries.', '',
        f'Both adaptive methods protect all **{len(candidates)} contiguous elementary-bin unions**, before selecting regions. '
        f'C.1 uses a joint Gaussian-multiplier bootstrap with 10,000 repetitions, seed 20260920, and critical value '
        f'**{protocol["critical_value"]:.6f}**; it is asymptotically simultaneous. '
        f'C.2 uses Clopper-Pearson probability limits with tail probability '
        f'0.05/(4*{len(candidates)}) = **{protocol["c2_tail_probability"]:.9g}**. '
        'This family-wide correction accounts for count-dependent region selection under the respective sampling assumptions. '
        'The retrospective and repeated-subject limitations stated above still apply.', '',
        '| Adaptive score region | P | Q | Cell-average estimate | C.1 joint | C.2 simultaneous |',
        '|---|---:|---:|---:|---|---|']
    for row in cells.itertuples():
        section.append(f'| [{row.left:.1f}, {row.right:.1f}{"]" if row.right_closed else ")"} | '
            f'{row.k_p} | {row.k_q} | {row.estimate:.4f} | [{row.c1_lower:.4f}, {row.c1_upper:.4f}] | '
            f'[{row.c2_lower:.4f}, {row.c2_upper:.4f}] |')
    section.extend(['',
        'These intervals describe averages over the merged regions. Wide merged regions can conceal the sparse middle-score '
        'mismatch, so tight adaptive intervals do not establish that individual neural scores are calibrated. '
        'Their widths are not directly comparable with the equal-width intervals above: the target regions change, '
        'and adaptive C.1 is simultaneous while the original C.1 is marginal.', '',
        f'[PDF]({name}/residual_mlp_adaptive_ci.pdf) · [Selected regions]({name}/cell_intervals.csv) · '
        f'[All candidates]({name}/candidate_intervals.csv) · [Protocol]({name}/protocol.json)', ''])
    updated_report = old_report.replace('# Residual AGP MLP: equal-width confidence intervals',
        '# Residual AGP MLP: equal-width and adaptive confidence intervals', 1)
    updated_report = updated_report.replace('## Reproduction', '\n'.join(section) + '\n## Reproduction', 1)
    updated_report += ('\nThen append the adaptive analysis to that reproduced report:\n\n```bash\n'
        f'python3 -B {root}/{name}/source/experiments/AGP/AGP_residual_adaptive_ci.py \\\n'
        f'  --report-dir {root}_rerun --min-count {args.min_count}\n```\n')
    backup.mkdir(parents=True)
    for filename in ('report.md', 'result.json'):
        shutil.copyfile(root / filename, backup / filename)
    (root / 'report.md.tmp').write_text(updated_report)
    (root / 'report.md.tmp').replace(root / 'report.md')
    completed.setdefault('adaptive_analyses', {})[name] = {
        'min_count': args.min_count, 'n_regions': len(cells), 'n_candidates': len(candidates)}
    completed['outputs'] = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                            if p.is_file() and p != root / 'result.json'}
    save_json(root / 'result.json.tmp', completed)
    (root / 'result.json.tmp').replace(root / 'result.json')
    print(json.dumps({'status': 'complete', 'report': str(root / 'report.md'),
        'min_count': args.min_count, 'n_regions': len(cells),
        'critical_value': protocol['critical_value']}))


if __name__ == '__main__':
    main()
