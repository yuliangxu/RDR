#!/usr/bin/env python3
"""Plot original C.1/C.2 equal-width intervals for the frozen residual AGP MLP."""

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
from utils.calibration import bin_scores, calibrate_counts

DEFAULT_SOURCE = Path('/cwork/yx306/RDR/JRSSB/CI/agp_architecture_20260925')
DEFAULT_OUTPUT = Path('/cwork/yx306/RDR/JRSSB/CI/agp_residual_equal_width_ci_20260925')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def plot(out, cells, run_id, epoch):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/agp-residual-equal-width-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    edges = np.r_[cells.left, cells.right.iloc[-1]]
    centers = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, method in zip(axes[:2], ('c2', 'c1')):
        low, high = (cells[f'{method}_{bound}'].to_numpy().copy() for bound in ('lower', 'upper'))
        if method == 'c1':
            missing = cells.c1_unavailable.to_numpy()
            low[missing], high[missing] = np.nan, np.nan
            if missing.any():
                ax.scatter(centers[missing], np.ones(missing.sum()), marker='x', color='.5',
                           label='C.1 unavailable ([0,2] in CSV)')
        # Duplicate each bin's endpoints so a missing band does not erase its neighbor.
        ax.fill_between(np.repeat(edges, 2)[1:-1], np.repeat(low, 2), np.repeat(high, 2),
                        color='#88b9d9', alpha=.65, label='Interval')
        estimate = cells.estimate.to_numpy().copy()
        estimate[cells.c1_empty.to_numpy()] = np.nan
        ax.stairs(estimate, edges, color='#182f44', label='Cell estimate')
        ax.plot([0, 2], [0, 2], ':', color='.5', lw=.8)
        ax.axhline(1, ls='--', color='.5', lw=.8)
        ax.set(xlim=(0, 2), ylim=(0, 2), xlabel='Frozen residual-MLP RDR score',
               ylabel='Cell-average RDR',
               title='C.2 simultaneous' if method == 'c2' else 'C.1 marginal, available cells')
        ax.legend(fontsize=8)
    width = np.diff(edges) * .4
    axes[2].bar(centers - width / 2, cells.k_p, width=width, label='P: observed AGP')
    axes[2].bar(centers + width / 2, cells.k_q, width=width, label='Q: ICFM')
    axes[2].set(xlim=(0, 2), yscale='log', xlabel='Frozen residual-MLP RDR score',
                ylabel='Test/calibration count (log)',
                title=f'Full test: {cells.n_p_calibration.iloc[0]:,} P / {cells.n_q_calibration.iloc[0]:,} Q')
    axes[2].legend(fontsize=8)
    fig.suptitle('AGP real vs ICFM: residual MLP, nominal 95% equal-width cell intervals\n'
                 f'{run_id}, epoch {epoch}; 20 fixed bins of width 0.1', fontsize=13)
    fig.tight_layout()
    fig.savefig(out / 'residual_mlp_equal_width_ci.png', dpi=180)
    fig.savefig(out / 'residual_mlp_equal_width_ci.pdf', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output_dir.resolve()
    allowed = Path('/tmp') if args.preflight else Path('/cwork/yx306/RDR')
    if not out.is_relative_to(allowed) or out.exists():
        raise ValueError(f'Use a fresh output directory below {allowed}')
    completed = json.loads((source / 'result.json').read_text())
    if completed['status'] != 'complete':
        raise ValueError('Source experiment is incomplete')
    def verified(name):
        path = source / name
        if sha(path) != completed['outputs'][name]:
            raise ValueError(f'Source hash mismatch: {name}')
        return path
    selection = json.loads(verified('selection.json').read_text())
    run_id = selection['by_method']['residual_p30']
    run = Path('runs') / run_id
    training = json.loads(verified(str(run / 'training.json')).read_text())
    model_path = verified(str(run / 'model.pt'))
    score_path = verified(str(run / 'test_scores.csv'))
    manifest_path = verified('split_manifest.csv')
    if sha(model_path) != selection['model_sha256'][run_id]:
        raise ValueError('Selected checkpoint changed')
    dtypes = {'sample_id': str, 'subject_id': str, 'row_sha256': str}
    scores = pd.read_csv(score_path, dtype=dtypes, keep_default_na=False, float_precision='round_trip')
    manifest = pd.read_csv(manifest_path, dtype=dtypes, keep_default_na=False)
    expected = manifest[manifest.role.eq('test')]
    keys = ['distribution', 'sample_id', 'subject_id', 'source_file', 'source_row', 'row_sha256', 'role']
    if set(scores.role) != {'test'} or len(scores) != len(expected):
        raise ValueError('Unexpected scoring role/count')
    if len(scores.merge(expected, on=keys, validate='one_to_one')) != len(scores):
        raise ValueError('Test scores disagree with manifest')
    edges = np.linspace(0, 2, 21)
    if not np.array_equal(scores.bin_index, bin_scores(scores.rdr.to_numpy(), edges)):
        raise ValueError('Saved score bins disagree with fixed boundaries')
    kp, kq = (np.bincount(scores.loc[scores.distribution.eq(d), 'bin_index'], minlength=20) for d in ('P', 'Q'))
    n, m = int(kp.sum()), int(kq.sum())
    if (n, m) != (2002, 2089):
        raise ValueError('Unexpected P/Q test counts')
    intervals = calibrate_counts(kp, kq, n, m, alpha=.05)
    cells = pd.DataFrame({'bin_index': np.arange(20), 'left': edges[:-1], 'right': edges[1:],
        'right_closed': np.arange(20) == 19, 'n_p_calibration': n, 'n_q_calibration': m,
        'k_p': kp, 'k_q': kq, 'p_hat': kp / n, 'q_hat': kq / m, **intervals})
    out.mkdir(parents=True)
    shutil.copyfile(score_path, out / 'test_scores.csv')
    shutil.copyfile(verified('selection.json'), out / 'source_selection.json')
    shutil.copyfile(verified(str(run / 'training.json')), out / 'training.json')
    for name in ('experiments/AGP/AGP_residual_equal_width_ci.py', 'utils/calibration.py', 'utils/__init__.py'):
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    cells.to_csv(out / 'cell_intervals.csv', index=False)
    scores.merge(cells, on='bin_index', validate='many_to_one').to_csv(out / 'test_intervals.csv', index=False)
    protocol = {'analysis': 'agp_residual_mlp_original_equal_width_ci', 'source': str(source),
        'run_id': run_id, 'epoch': training['best_epoch'], 'source_result_sha256': sha(source / 'result.json'),
        'model_sha256': sha(model_path), 'scores_sha256': sha(score_path),
        'selection': 'within-residual minimum validation Brier, from archived selection.json',
        'n_p': n, 'n_q': m, 'edges': edges.tolist(), 'alpha': .05,
        'c1': 'original marginal delta-method interval; unavailable cells marked and stored as [0,2]',
        'c2': 'original simultaneous Clopper-Pearson probability bounds across K=20 fixed bins',
        'c2_tail_probability': .05 / (4 * 20), 'merged': False, 'model_retrained': False,
        'retrospective': True, 'sampling': 'nominal observation-level; repeated P subjects not IID-adjusted'}
    save_json(out / 'protocol.json', protocol)
    plot(out, cells, run_id, training['best_epoch'])
    table = ['| Score region | P | Q | Estimate | C.1 marginal | C.2 simultaneous |',
             '|---|---:|---:|---:|---|---|']
    for row in cells.itertuples():
        c1 = 'Unavailable ([0,2])' if row.c1_unavailable else f'[{row.c1_lower:.4f}, {row.c1_upper:.4f}]'
        table.append(f'| [{row.left:.1f}, {row.right:.1f}{"]" if row.right_closed else ")"} | '
            f'{row.k_p} | {row.k_q} | {row.estimate:.4f} | {c1} | [{row.c2_lower:.4f}, {row.c2_upper:.4f}] |')
    text = ['# Residual AGP MLP: equal-width confidence intervals', '',
        f'Frozen **{run_id}**, epoch **{training["best_epoch"]}**, selected by validation Brier within the residual architecture. '
        f'The same **{n:,} P / {m:,} Q** test/calibration scores are reused.', '',
        '![Residual MLP equal-width intervals](residual_mlp_equal_width_ci.png)', '',
        'The layout follows the original AGP plot: C.2 simultaneous intervals, C.1 marginal intervals, and P/Q counts. '
        'The 20 score intervals are [0,0.1), ..., [1.9,2]. They are fixed numeric boundaries with no merging. '
        'C.2 uses tail probability 0.05/(4*20) = 0.000625; C.1 uses the marginal normal critical value. '
        'These are the original fixed-bin algorithms, distinct from the earlier joint-bootstrap and 210-candidate adjustment.', '',
        r'The horizontal axis is the frozen neural score, and each step spans its score bin. '
        r'The vertical estimate is $2(k_P/n_P)/(k_P/n_P+k_Q/n_Q)$. '
        r'Intervals target $2P(A_j)/(P(A_j)+Q(A_j))$, the cell-average RDR. '
        'They do not cover every individual RDR. Empty estimates are omitted; unavailable C.1 cells are marked with crosses '
        'and retain [0,2] in the CSV. Zero count bars are absent on the logarithmic axis.', '',
        'These remain retrospective nominal intervals: repeated observations within subjects, historical preprocessing, '
        'generator provenance limitations, and previous inspection of the cohort prevent a fresh unconditional IID-coverage claim. '
        'Sparse middle bins produce wide intervals.', '',
        '[PDF](residual_mlp_equal_width_ci.pdf) · [Cell table](cell_intervals.csv) · [Protocol](protocol.json)', '',
        *table, '', '## Reproduction', '', 'Use a fresh output directory:', '', '```bash',
        f'python3 -B {out}/source/experiments/AGP/AGP_residual_equal_width_ci.py \\',
        f'  --source {source} --output-dir {out}_rerun', '```', '']
    (out / 'report.md').write_text('\n'.join(text))
    save_json(out / 'result.json', {'status': 'complete', 'run_id': run_id,
        'n_cells': 20, 'n_c1_unavailable': int(cells.c1_unavailable.sum()),
        'outputs': {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}})
    print(json.dumps({'status': 'complete', 'output': str(out), 'run_id': run_id,
                      'n_c1_unavailable': int(cells.c1_unavailable.sum())}))


if __name__ == '__main__':
    main()
