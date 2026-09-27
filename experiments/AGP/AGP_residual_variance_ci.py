#!/usr/bin/env python3
"""Add within-cell true-RDR variance bounds to the residual AGP CI report."""

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
from utils.calibration import variance_upper_bound

DEFAULT_REPORT = Path('/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_table(path):
    return pd.read_csv(path, float_precision='round_trip', keep_default_na=False,
                       dtype={'sample_id': str, 'subject_id': str, 'row_sha256': str})


def augment(table):
    result = table.copy()
    for method in ('c1', 'c2'):
        bound = variance_upper_bound(table[f'{method}_lower'], table[f'{method}_upper'])
        result[f'{method}_variance_upper'] = bound
        result[f'{method}_sd_upper'] = np.sqrt(bound)
    return result


def plot(out, cells, title, c1_label, run_id):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/agp-residual-variance-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    edges = np.r_[cells.left, cells.right.iloc[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, method in zip(axes, ('c2', 'c1')):
        bound = cells[f'{method}_variance_upper'].to_numpy()
        ax.fill_between(np.repeat(edges, 2)[1:-1], 0, np.repeat(bound, 2),
                        color='#88b9d9', alpha=.45, label='Variance confidence set [0, upper]')
        ax.stairs(bound, edges, color='#182f44', linewidth=1.5, label='Upper confidence bound')
        ax.axhline(1, color='.5', ls=':', lw=1, label='Universal maximum variance: 1')
        if method == 'c1' and cells.c1_unavailable.any():
            missing = cells.c1_unavailable.to_numpy()
            ax.scatter(((edges[:-1] + edges[1:]) / 2)[missing], bound[missing],
                       marker='x', color='#922b21', zorder=5, label='C.1 fallback: upper = 1')
        ax.set(xlim=(0, 2), ylim=(0, 1.06), xlabel='Frozen residual-MLP RDR score',
               ylabel='Within-cell true RDR variance',
               title='C.2 simultaneous' if method == 'c2' else c1_label)
        ax.legend(fontsize=8, loc='upper right' if len(cells) == 2 else 'lower center')
    fig.suptitle(f'AGP residual MLP: nominal 95% variance upper bounds\n{title}; {run_id}', fontsize=12)
    fig.tight_layout()
    fig.savefig(out / 'variance_upper_bounds.png', dpi=180)
    fig.savefig(out / 'variance_upper_bounds.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def report_section(name, cells, c1_label):
    location = f'variance_bounds/{name}'
    text = ['### Within-cell variance upper bounds', '',
        f'![Within-cell variance bounds: {name}]({location}/variance_upper_bounds.png)', '',
        'Each plotted band is the confidence set $[0,U_v]$ for the population variance '
        r'$\operatorname{Var}_M(r_0(X)\mid X\in A)$; the curve is its upper endpoint. '
        'It is neither a variance estimate nor a range for individual true RDR values. '
        'The CSV table retains the original mean-RDR CIs and adds variance and standard-deviation upper bounds. '
        'Numbers below are rounded; CSV files retain full precision.', '',
        f'| Score region | P | Q | C.1 variance upper ({c1_label}) | C.2 variance upper (simultaneous) | C.1 SD upper | C.2 SD upper |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in cells.itertuples():
        text.append(f'| [{row.left:.1f}, {row.right:.1f}{"]" if row.right_closed else ")"} | '
            f'{row.k_p} | {row.k_q} | {row.c1_variance_upper:.6f} | {row.c2_variance_upper:.6f} | '
            f'{row.c1_sd_upper:.6f} | {row.c2_sd_upper:.6f} |')
    text += ['', f'[PDF]({location}/variance_upper_bounds.pdf) · '
        f'[Mean CIs and variance bounds]({location}/cell_intervals.csv) · '
        f'[Test/calibration attachments]({location}/test_intervals.csv)', '']
    if name == 'equal_width':
        full = int(np.count_nonzero(cells.c2_variance_upper == 1))
        text += [f'C.2 gives the uninformative upper bound 1 in **{full} of {len(cells)}** elementary cells. '
            'The endpoint cells have substantially smaller bounds. The C.1-unavailable cell [1.8,1.9) '
            'retains its conservative mean interval [0,2], giving variance upper bound 1 and SD upper bound 1.', '']
    else:
        text += ['The broad merged regions have small upper bounds because their population mean RDRs '
            'are constrained near 0 or 2. This limits mixture-weighted average dispersion within each region; '
            'rare observations can still have very different true RDR values. Merging changes the variance target, '
            'so these bounds cannot be interpreted as improved estimation of the elementary-cell variances.', '',
            f'[All 210 candidate mean CIs and variance bounds]({location}/candidate_intervals.csv)', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    root = args.report_dir.resolve()
    allowed = Path('/tmp') if args.preflight else Path('/cwork/yx306/RDR')
    if not root.is_relative_to(allowed) or not root.is_dir():
        raise ValueError(f'Use an existing completed residual CI report below {allowed}')
    out = root / 'variance_bounds'
    backup = root / 'report_revisions/before_variance_bounds'
    if out.exists() or backup.exists():
        raise ValueError('Variance analysis already exists; refusing to overwrite it')
    completed = json.loads((root / 'result.json').read_text())
    if completed['status'] != 'complete':
        raise ValueError('Original report is incomplete')
    for filename, digest in completed['outputs'].items():
        if sha(root / filename) != digest:
            raise ValueError(f'Original report hash mismatch: {filename}')
    original = json.loads((root / 'protocol.json').read_text())
    adaptive = json.loads((root / 'adaptive_h20/protocol.json').read_text())
    if original['run_id'] != adaptive['run_id'] or original['model_sha256'] != adaptive['model_sha256']:
        raise ValueError('Partitions must use the same frozen residual model')
    if original['scores_sha256'] != sha(root / 'test_scores.csv') or original['scores_sha256'] != adaptive['scores_sha256']:
        raise ValueError('Partitions must use the same frozen scores')
    if original['alpha'] != .05 or adaptive['alpha'] != .05 or adaptive['min_count'] != 20:
        raise ValueError('Expected the existing nominal 95% equal-width and h=20 analyses')
    old_report = (root / 'report.md').read_text()
    markers = ('![Residual MLP equal-width intervals]', '## Data-adaptive confidence intervals, h=20', '## Reproduction')
    if any(old_report.count(marker) != 1 for marker in markers):
        raise ValueError('Unexpected report structure')

    out.mkdir()
    inputs = {}
    tables = {}
    summary = {}
    for name, source, key in (('equal_width', root, 'bin_index'),
                              ('adaptive_h20', root / 'adaptive_h20', 'region_index')):
        folder = out / name
        folder.mkdir()
        cells = read_table(source / 'cell_intervals.csv')
        rows = read_table(source / 'test_intervals.csv')
        augmented_cells, augmented_rows = augment(cells), augment(rows)
        for method in ('c1', 'c2'):
            for suffix in ('variance_upper', 'sd_upper'):
                column = f'{method}_{suffix}'
                expected = rows[key].map(augmented_cells.set_index(key)[column])
                if not np.array_equal(expected.to_numpy(), augmented_rows[column].to_numpy()):
                    raise ValueError(f'Inconsistent row attachment: {name}/{column}')
        augmented_cells.to_csv(folder / 'cell_intervals.csv', index=False)
        augmented_rows.to_csv(folder / 'test_intervals.csv', index=False)
        for filename in ('cell_intervals.csv', 'test_intervals.csv', 'protocol.json'):
            inputs[str((source / filename).relative_to(root))] = sha(source / filename)
        if name == 'adaptive_h20':
            candidate_path = source / 'candidate_intervals.csv'
            candidates = augment(read_table(candidate_path))
            candidates.to_csv(folder / 'candidate_intervals.csv', index=False)
            inputs[str(candidate_path.relative_to(root))] = sha(candidate_path)
        title = '20 equal-width score cells' if name == 'equal_width' else 'h=20; 2 adaptive score regions'
        c1_label = 'C.1 marginal' if name == 'equal_width' else 'C.1 asymptotic simultaneous'
        plot(folder, augmented_cells, title, c1_label, original['run_id'])
        tables[name] = augmented_cells
        summary[name] = {'n_cells': len(cells), 'n_test_calibration_attachments': len(rows),
            'c1_unavailable': int(cells.c1_unavailable.sum()),
            'c1_uninformative_upper_bounds': int((augmented_cells.c1_variance_upper == 1).sum()),
            'c2_uninformative_upper_bounds': int((augmented_cells.c2_variance_upper == 1).sum())}

    for filename in ('experiments/AGP/AGP_residual_variance_ci.py',
                     'utils/calibration.py', 'utils/__init__.py',
                     'tests/test_variance_bound.py',
                     'experiments/JRSSB/CI/agent_CI.md'):
        target = out / 'source' / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / filename, target)
    protocol = {'analysis': 'within_cell_true_rdr_variance_upper_confidence_bounds',
        'target': 'Var_M(r0(X) | X in A), M=(P+Q)/2, r0=2p/(p+q)',
        'variance_range_given_mean': '[0, theta*(2-theta)]',
        'variance_upper': 'c*(2-c), c=clip(1, mean_CI_lower, mean_CI_upper)',
        'variance_confidence_set': '[0, variance_upper]', 'sd_upper': 'sqrt(variance_upper)',
        'alpha': .05, 'model_retrained': False, 'mean_intervals_recomputed': False,
        'run_id': original['run_id'], 'epoch': original['epoch'],
        'n_p': original['n_p'], 'n_q': original['n_q'],
        'model_sha256': original['model_sha256'], 'scores_sha256': original['scores_sha256'],
        'original_result_sha256': sha(root / 'result.json'), 'input_sha256': inputs,
        'coverage': {'equal_width_c1': 'asymptotic marginal per fixed cell',
            'equal_width_c2': 'simultaneous across 20 fixed cells under independent IID sampling',
            'adaptive_h20_c1': 'asymptotic simultaneous over all 210 candidates before selection',
            'adaptive_h20_c2': 'simultaneous over all 210 candidates before selection under independent IID sampling',
            'additional_alpha_cost': 0,
            'joint_across_separate_analyses': False},
        'whole_space_mean_1_variance_upper': 1.0, 'fallback_mean_0_2_variance_upper': 1.0,
        'sampling': original['sampling'], 'retrospective': True,
        'notes': 'Bounds on within-cell population dispersion, not point estimates, sampling variances, or individual-RDR ranges.',
        'summary': summary}
    save_json(out / 'protocol.json', protocol)
    introduction = '\n'.join(['## Variance target and confidence guarantee', '',
        r'For $M=(P+Q)/2$ and $r_0=2p/(p+q)$, write '
        r'$\theta_A=\mathbb E_M[r_0(X)\mid X\in A]$ and '
        r'$v_A=\operatorname{Var}_M(r_0(X)\mid X\in A)$. '
        'These targets require positive population mixture mass; zero observed counts do not prove zero population mass. '
        r'Since $0\le r_0\le2$, $r_0^2\le2r_0$, so $0\le v_A\le\theta_A(2-\theta_A)$. '
        'For each saved mean interval $[L_A,U_A]$, compute', '',
        '$$', r'c_A=\min\{\max\{1,L_A\},U_A\},\qquad '
        r'U_{v,A}=c_A(2-c_A)=\max_{t\in[L_A,U_A]}t(2-t),\qquad '
        r'v_A\in[0,U_{v,A}].', '$$', '',
        r'The standard-deviation upper bound is $\sqrt{U_{v,A}}$. These are bounds for the true RDR variation '
        'within a score cell, distinct from the standard error of its estimated mean. '
        'Cell counts alone cannot identify this variance: even an exact mean of 1 is compatible with variance anywhere '
        'from 0 to 1. In particular, both a whole-space mean interval [1,1] and a fallback interval [0,2] give '
        'variance upper bound 1. A positive-mass cell whose mean interval is [0,0] or [2,2] gives upper bound 0.', '',
        'Whenever a mean interval covers its target, the derived variance and SD bounds hold on that same event; '
        'no additional error allocation is needed. Equal-width C.1 remains asymptotically marginal; '
        'equal-width C.2 remains simultaneous across 20 cells. Adaptive C.1 remains asymptotically simultaneous, '
        'and adaptive C.2 retains protection across all 210 candidates before h=20 selection. '
        'No joint 95% claim across these separate analyses or methods is made. '
        'The AGP repeated-subject and retrospective sampling limitations carry over unchanged.', '',
        'The upper bound can remain positive with arbitrarily precise mean estimation, because within-cell heterogeneity '
        'is not identified by the mean. A small variance bound constrains mixture-weighted dispersion; it does not cover '
        'each individual true RDR or justify a normal mean-plus/minus-SD interval. Neural-score variation is not used '
        'as a substitute for true-RDR variation. In particular, a narrow neural-score bin does not constrain '
        'the true RDR to that bin.', '',
        '[Method and derivation](variance_bounds/source/experiments/JRSSB/CI/agent_CI.md#upper-confidence-bounds-for-true-within-cell-rdr-variance) · '
        '[Variance protocol](variance_bounds/protocol.json)', '', '## Equal-width confidence intervals', ''])
    updated = old_report.replace('# Residual AGP MLP: equal-width and adaptive confidence intervals',
        '# Residual AGP MLP: mean confidence intervals and variance upper bounds', 1)
    updated = updated.replace(markers[0], introduction + '\n' + markers[0], 1)
    updated = updated.replace(markers[1], report_section('equal_width', tables['equal_width'], 'marginal')
        + '\n' + markers[1], 1)
    updated = updated.replace(markers[2], report_section('adaptive_h20', tables['adaptive_h20'], 'asymptotic simultaneous')
        + '\n' + markers[2], 1)
    updated += ('\nFinally derive the variance bounds from those saved mean intervals:\n\n```bash\n'
        f'python3 -B {root}/variance_bounds/source/experiments/AGP/AGP_residual_variance_ci.py \\\n'
        f'  --report-dir {root}_rerun\n```\n')

    for filename, digest in completed['outputs'].items():
        if sha(root / filename) != digest:
            raise ValueError(f'An original artifact changed during generation: {filename}')
    backup.mkdir(parents=True)
    for filename in ('report.md', 'result.json'):
        shutil.copyfile(root / filename, backup / filename)
    (root / 'report.md.tmp').write_text(updated)
    (root / 'report.md.tmp').replace(root / 'report.md')
    completed['variance_bounds'] = summary
    completed['outputs'] = {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*'))
                            if p.is_file() and p != root / 'result.json'}
    save_json(root / 'result.json.tmp', completed)
    (root / 'result.json.tmp').replace(root / 'result.json')
    print(json.dumps({'status': 'complete', 'report': str(root / 'report.md'), 'variance_bounds': summary}))


if __name__ == '__main__':
    main()
