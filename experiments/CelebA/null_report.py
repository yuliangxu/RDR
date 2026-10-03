"""Final-split diagnostics for freshly fitted, fixed-setting CelebA null models.

The driver verifies provenance and completion receipts before supplying rows.
This module reports those rows; it never selects settings or seals completion.
"""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np


METRICS = (
    'brier', 'local_gap', 'supported_mass', 'c2_distance', 'c2_width',
    'middle_mass', 'middle_supported_mass', 'middle_supported_fraction',
    'middle_local_gap', 'rdr_mean_p', 'rdr_mean_q', 'rdr_mse_one',
    'rdr_mae_one', 'rdr_rmse_one', 'excess_brier_true',
    'empirical_brier_excess', 'h2_variational', 'h2_plugin',
    'h2_clip_rate_p', 'h2_clip_rate_q',
)
REQUIRED = ('brier', 'supported_mass', 'rdr_mse_one', 'rdr_mae_one',
            'rdr_rmse_one', 'rdr_mean_p', 'rdr_mean_q',
            'excess_brier_true', 'empirical_brier_excess')
IDENTITY = ('representation', 'source', 'repeat')
DEFINITION = ('candidate', 'loss', 'output_alpha')
LABELS = {
    'brier': 'Balanced Brier', 'local_gap': 'Local absolute Gap',
    'supported_mass': 'Supported mass', 'c2_distance': 'C.2 distance',
    'c2_width': 'C.2 width', 'middle_mass': 'Middle mass',
    'middle_supported_fraction': 'Middle supported fraction',
    'middle_local_gap': 'Middle local Gap', 'rdr_mean_p': 'Mean RDR on A',
    'rdr_mean_q': 'Mean RDR on B', 'rdr_mse_one': 'RDR MSE against 1',
    'rdr_mae_one': 'RDR MAE against 1', 'rdr_rmse_one': 'RDR RMSE against 1',
    'excess_brier_true': 'RDR MSE / 4',
    'empirical_brier_excess': 'Held-out Brier minus 0.25',
    'h2_variational': 'Variational Hellinger diagnostic',
    'h2_plugin': 'Plug-in Hellinger diagnostic',
    'h2_clip_rate_p': 'Diagnostic clip rate A',
    'h2_clip_rate_q': 'Diagnostic clip rate B',
}
SOURCE_LABELS = {'lower': 'Q_l', 'upper': 'Q_u', 'real': 'Real P'}


def _identity(row):
    return tuple(row[key] for key in IDENTITY)


def _label(representation, source):
    label = SOURCE_LABELS.get(source, source)
    return f'{representation} / {label} A vs B'


def _validate(protocol, rows, failures):
    repeats = protocol['config']['repeats']
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('Invalid registered repetition count')
    tasks, definitions, planned = {}, {}, {}
    for task in protocol['tasks']:
        key = _identity(task)
        repeat = task['repeat']
        if isinstance(repeat, bool) or not isinstance(repeat, int) or not 0 <= repeat < repeats:
            raise ValueError('Invalid registered repetition')
        if key in tasks:
            raise ValueError(f'Duplicate registered task: {key}')
        tasks[key] = task
        group = key[:2]
        definition = tuple(task[field] for field in DEFINITION)
        if definitions.setdefault(group, definition) != definition:
            raise ValueError(f'Inconsistent fixed settings: {group}')
        planned.setdefault(group, []).append(repeat)
    if not tasks or any(sorted(values) != list(range(repeats)) for values in planned.values()):
        raise ValueError('Registered comparisons must include every repetition')
    seen = set()
    for row in rows:
        key = _identity(row)
        if key in seen:
            raise ValueError(f'Duplicate completed fit: {key}')
        seen.add(key)
        if key not in tasks or any(row[field] != tasks[key][field] for field in DEFINITION):
            raise ValueError(f'Completed fit outside registered tasks: {key}')
        for metric in METRICS:
            value = row.get(metric)
            if value is None:
                if metric in REQUIRED or (metric == 'local_gap' and row.get('supported_mass', 0) > 0):
                    raise ValueError(f'Missing required metric: {metric}')
                continue
            if isinstance(value, (bool, str)) or not math.isfinite(float(value)):
                raise ValueError(f'Invalid report metric: {metric}')
            if metric in ('h2_variational', 'h2_plugin'):
                continue  # Neither fitted-score diagnostic is constrained to be nonnegative.
            low = -.25 if metric == 'empirical_brier_excess' else 0
            high = (2 if metric in ('local_gap', 'middle_local_gap', 'c2_distance',
                                    'c2_width', 'rdr_mean_p', 'rdr_mean_q')
                    else .75 if metric == 'empirical_brier_excess'
                    else .25 if metric == 'excess_brier_true' else 1)
            if not low - 1e-12 <= value <= high + 1e-12:
                raise ValueError(f'Report metric outside its range: {metric}')
        for actual, expected in ((row['excess_brier_true'], row['rdr_mse_one'] / 4),
                                 (row['empirical_brier_excess'], row['brier'] - .25),
                                 (row['rdr_rmse_one'] ** 2, row['rdr_mse_one'])):
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-10):
                raise ValueError('Inconsistent null error metrics')
    failed = set()
    for failure in failures:
        key = _identity(failure)
        if key not in tasks or key in seen or key in failed:
            raise ValueError(f'Unregistered, completed, or duplicate failed fit: {key}')
        failed.add(key)
    return tasks, seen, failed


def _aggregate(protocol, rows, failures):
    expected = protocol['config']['repeats']
    summaries = []
    groups = dict.fromkeys((t['representation'], t['source']) for t in protocol['tasks'])
    for representation, source in groups:
        group = [r for r in rows if (r['representation'], r['source']) == (representation, source)]
        task = next(t for t in protocol['tasks']
                    if (t['representation'], t['source']) == (representation, source))
        failed = sum((f['representation'], f['source']) == (representation, source) for f in failures)
        complete = sorted(r['repeat'] for r in group) == list(range(expected))
        summary = {key: task[key] for key in ('representation', 'source') + DEFINITION}
        summary.update(repeats=len(group), expected_repeats=expected, failed_repeats=failed,
                       pending_repeats=expected - len(group) - failed, complete=complete)
        for metric in METRICS:
            values = [r[metric] for r in group if r.get(metric) is not None]
            available = complete and len(values) == expected
            summary[metric + '_available_repeats'] = len(values)
            summary[metric + '_mean'] = float(np.mean(values)) if available else None
            summary[metric + '_sd'] = float(np.std(values, ddof=1)) if available and expected > 1 else None
        summaries.append(summary)
    return summaries


def _csv(path, rows, preferred):
    fields = list(preferred) + sorted(set().union(*(r.keys() for r in rows)) - set(preferred))
    fields = [key for key in fields if not any(isinstance(r.get(key), (dict, list)) for r in rows)]
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _format(summary, metric):
    mean, sd = summary[metric + '_mean'], summary[metric + '_sd']
    if mean is None:
        return 'Unavailable'
    return f'{mean:.6f} ({sd:.6f})' if sd is not None else f'{mean:.6f} (SD unavailable)'


def _state(summary):
    if summary['complete']:
        return f"Complete ({summary['repeats']}/{summary['expected_repeats']})"
    return (f"{summary['repeats']}/{summary['expected_repeats']} complete; "
            f"{summary['failed_repeats']} failed; {summary['pending_repeats']} pending")


def _table(summaries, metrics):
    headers = ['Comparison', 'Fit status'] + [LABELS[metric] for metric in metrics]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for summary in summaries:
        values = [_label(summary['representation'], summary['source']), _state(summary)]
        values += [_format(summary, metric) for metric in metrics]
        lines.append('| ' + ' | '.join(values) + ' |')
    return lines


def _counts(output):
    path = output / 'data/null_data.json'
    if not path.is_file():
        return ['Exact A/B role counts are pending data preparation; see `data/null_data.json`.']
    metadata = json.loads(path.read_text())
    lines = ['| Role | Source | Parent images | A images | B images | A identities | B identities |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for role, sources in metadata['roles'].items():
        for source, item in sources.items():
            a, b = item['sides']['p'], item['sides']['q']
            lines.append(f"| {role} | {SOURCE_LABELS.get(source, source)} | {item['parent_count']} | "
                         f"{a['count']} | {b['count']} | {a.get('identity_count') or '—'} | "
                         f"{b.get('identity_count') or '—'} |")
    return lines


def _plots(output, summaries, rows):
    if not rows:
        return []
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/celeba-null-report-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    colors = {'feature': '#246b9e', 'pixel': '#b75728'}
    for ax, metric, reference in zip(axes, ('brier', 'rdr_mse_one'), (.25, 0)):
        for x, summary in enumerate(summaries):
            values = sorted((r for r in rows if (r['representation'], r['source']) ==
                             (summary['representation'], summary['source'])), key=lambda r: r['repeat'])
            color = colors.get(summary['representation'], '#333333')
            for row in values:
                offset = .24 * (row['repeat'] / max(1, summary['expected_repeats'] - 1) - .5)
                ax.scatter(x + offset, row[metric], color=color, s=30, alpha=.75, zorder=3)
            if summary[metric + '_mean'] is not None:
                ax.errorbar(x, summary[metric + '_mean'], yerr=summary[metric + '_sd'],
                            color='black', fmt='D', markersize=5, capsize=4, zorder=4)
        ax.axhline(reference, color='#555555', linestyle='--', linewidth=1,
                   label=f'Population null optimum: {reference:g}')
        labels = [f"{r['representation']}\n{SOURCE_LABELS.get(r['source'], r['source'])} A vs B" +
                  (f"\n{r['failed_repeats']} failed" if r['failed_repeats'] else
                   f"\n{r['pending_repeats']} pending" if not r['complete'] else '') for r in summaries]
        ax.set(xticks=range(len(summaries)), xticklabels=labels, ylabel=LABELS[metric],
               title=LABELS[metric])
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='best', fontsize=8)
    fig.suptitle('CelebA same-source null: selected JS settings\n'
                 'Dots: training repetitions; diamonds/bars: complete-group mean ± SD (not CIs)')
    paths = []
    for suffix in ('png', 'pdf'):
        name = f'null_summary.{suffix}'
        fig.savefig(output / name, dpi=180, bbox_inches='tight',
                    metadata={'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None)
        paths.append(name)
    plt.close(fig)
    return paths


def _ci_plot(output, protocol, summaries, rows):
    # These cell files have already been receipt-verified by the caller. Never
    # read an unverified/missing task merely because it has a plausible path.
    completed = {(r['representation'], r['source']) for r in rows if r['repeat'] == 0}
    cells_by_group = {}
    for group in completed:
        path = output / 'evaluation' / group[0] / group[1] / 'repeat_00/cells.json'
        if path.is_file():
            cells_by_group[group] = json.loads(path.read_text())
    if not cells_by_group:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    nrows = math.ceil(len(summaries) / 2)
    fig, axes = plt.subplots(nrows, 2, figsize=(12, nrows * 3.5), squeeze=False,
                             constrained_layout=True)
    handles, labels = [], []
    for ax, summary in zip(axes.flat, summaries):
        group = summary['representation'], summary['source']
        ax.set(title=_label(*group), xlim=(0, 2), ylim=(-.08, 2.05),
               xlabel='Fixed neural-score bin', ylabel='Cell-average RDR')
        cells = cells_by_group.get(group)
        if cells is None:
            ax.text(.5, .5, 'Repeat 00 unavailable', transform=ax.transAxes, ha='center')
            continue
        edges = np.array([cell['left'] for cell in cells] + [cells[-1]['right']])
        centers = (edges[:-1] + edges[1:]) / 2
        lower, upper = (np.array([cell[key] for cell in cells]) for key in ('c2_lower', 'c2_upper'))
        ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]],
                        step='post', color='#8cb9d5', alpha=.6, label='C.2 simultaneous cells')
        available = np.array([not cell['c1_unavailable'] for cell in cells])
        estimate = np.array([np.nan if cell['calibrated_rdr'] is None else cell['calibrated_rdr']
                             for cell in cells])
        neural = np.array([np.nan if cell['neural_mean'] is None else cell['neural_mean'] for cell in cells])
        c1lower = np.array([cell['c1_lower'] if cell['c1_lower'] is not None else np.nan for cell in cells])
        c1upper = np.array([cell['c1_upper'] if cell['c1_upper'] is not None else np.nan for cell in cells])
        ax.errorbar(centers[available], estimate[available],
                    yerr=[estimate[available] - c1lower[available], c1upper[available] - estimate[available]],
                    fmt='none', ecolor='#b75d23', capsize=3, label='C.1 marginal when available')
        ax.stairs(estimate, edges, baseline=None, color='#152b3c', label='Calibration cell estimate')
        ax.scatter(centers, neural, s=20, facecolors='none', edgecolors='#b42363',
                   label='Evaluation neural cell mean')
        ax.axhline(1, color='#333333', linestyle='--', linewidth=1, label='Null truth = 1')
        if np.any(~available):
            ax.scatter(centers[~available], np.full(np.sum(~available), -.04), marker='x',
                       color='#b75d23', label='C.1 unavailable', clip_on=False)
        ax.grid(alpha=.15)
        handles, labels = ax.get_legend_handles_labels()
    for ax in list(axes.flat)[len(summaries):]:
        ax.axis('off')
    if len(summaries) % 2:
        list(axes.flat)[-1].legend(handles, labels, loc='center', frameon=False, fontsize=9)
    else:
        axes[0, 0].legend(handles, labels, loc='best', fontsize=6)
    confidence = 100 * (1 - protocol['config'].get('ci_alpha', .05))
    fig.suptitle(f'CelebA same-source null: fixed illustration repeat 00\n'
                 f'Nominal {confidence:g}% cell-average intervals; image-level reference, not empirical coverage\n'
                 'C.2 simultaneity is within one comparison; intervals are not averaged over repetitions')
    paths = []
    for suffix in ('png', 'pdf'):
        name = f'null_ci_repeat_00.{suffix}'
        fig.savefig(output / name, dpi=180, bbox_inches='tight',
                    metadata={'CreationDate': None, 'ModDate': None} if suffix == 'pdf' else None)
        paths.append(name)
    plt.close(fig)
    return paths


def build_report(output: Path, protocol: dict, rows: list, failures=None):
    """Render verified completed fits and explicitly distinguish failure/pending.

    Only final-evaluation summaries belong in ``rows``. Calibration observations
    construct the intervals separately and are never pooled into primary risks.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, failures = list(rows), list(failures or [])
    tasks, seen, failed = _validate(protocol, rows, failures)
    summaries = _aggregate(protocol, rows, failures)
    _csv(output / 'per_fit.csv', rows, IDENTITY + DEFINITION + METRICS)
    _csv(output / 'per_comparison.csv', summaries, ('representation', 'source') + DEFINITION)
    plots = _plots(output, summaries, rows)
    ci_plots = _ci_plot(output, protocol, summaries, rows)
    status = dict(complete=len(seen) == len(tasks), completed_fits=len(seen),
                  expected_fits=len(tasks), failed_fits=len(failed),
                  pending_fits=len(tasks) - len(seen) - len(failed),
                  scope='selected_settings_fresh_null_fits_final_evaluation',
                  smoke=bool(protocol.get('smoke', protocol['config'].get('smoke', False))))
    lines = ['# CelebA expanded-data selected-model null experiments', '',
             f"**Status: {'complete' if status['complete'] else 'incomplete'}; "
             f"{len(seen)}/{len(tasks)} verified fits; {len(failed)} failed; "
             f"{status['pending_fits']} pending.**", '']
    if status['smoke']:
        lines += ['**Smoke validation: plumbing only, not scientific evidence.**', '']
    lines += ['Each comparison fits a fresh network to disjoint A/B subsets of the same source. '
              'The settings selected in the non-null study are fixed: JS loss, '
              'r = 2 sigmoid(0.5 z) for the feature MLP and r = 2 sigmoid(2 z) for the pixel CNN. '
              'The baseline architectures and checkpoint rule are retained; no null-specific tuning '
              'or exact-one prediction substitution is used.', '',
              'The five experiment families are Q_l features, Q_u features, Q_l pixels, '
              'Q_u pixels, and real P pixels. Each has five registered training repetitions. '
              'A/B membership and the six data roles are fixed across repetitions. '
              'The SD therefore describes initialization/training-order variation conditional on these data, '
              'not uncertainty across independently resampled datasets.', '',
              '## Final-evaluation results', '',
              'All primary metrics below use the final evaluation role. The separate final calibration '
              'role constructs score-cell intervals. A and B each receive weight one half even when '
              'their image counts differ. Tables show mean (sample SD) over every registered repetition; '
              'an incomplete family has no aggregate. Individual completed fits remain in `per_fit.csv`.', '',
              'Under the same-source null, the population RDR is 1 and the optimal balanced Brier risk '
              'is 0.25. RDR MSE is the balanced empirical average of (r − 1)^2. Its quarter estimates '
              'population excess Brier risk. Held-out Brier minus 0.25 is a different finite-sample '
              'quantity and can be negative; it is not truncated.', '',
              *_table(summaries, ('brier', 'local_gap', 'rdr_mse_one', 'supported_mass')), '',
              *_table(summaries, ('rdr_mean_p', 'rdr_mean_q', 'rdr_mae_one', 'rdr_rmse_one')), '',
              *_table(summaries, ('excess_brier_true', 'empirical_brier_excess')), '',
              '## Local diagnostics', '',
              'Local Gap and C.2 distance average over supported score cells; C.2 width uses all '
              'evaluation mass, including cells unsupported by calibration. Middle diagnostics refer to '
              'the same middle-score region as the main selected-model assessment; unavailable '
              'diagnostics are not replaced by zero or averaged over fewer repetitions. '
              'Intervals target cell-average RDR, not an image-specific RDR. Nominal image-level '
              'intervals and this fixed-data repetition study do not establish empirical coverage, '
              'especially when real images share identities.', '',
              *_table(summaries, ('c2_distance', 'c2_width', 'middle_mass',
                                 'middle_supported_fraction', 'middle_local_gap')), '',
              '## Historical Hellinger diagnostics', '',
              'These auxiliary final-evaluation summaries reuse the historical midpoint-Hellinger '
              'formulas for comparability. The variational value is a finite learned-objective '
              'diagnostic: it can be negative and is neither a nonnegative distance estimate by '
              'construction nor a validity certificate. The plug-in identity can also be negative '
              'for imperfect learned scores. Values are not truncated to zero. '
              'Only these legacy Hellinger calculations clip scores to [1e-6, 2 − 1e-6]; '
              'the clip rates are reported. Primary RDR errors, Brier, and calibration diagnostics '
              'use unmodified scores; JS training applies no such score clipping. No bootstrap '
              'intervals are computed in this study.', '',
              *_table(summaries, ('h2_variational', 'h2_plugin', 'h2_clip_rate_p', 'h2_clip_rate_q'))]
    if failures:
        lines += ['', '## Failed fits', '']
        for failure in failures:
            message = str(failure.get('message', failure.get('error', 'See FAILED.json'))).replace('\n', ' ')
            lines.append(f"- {_label(failure['representation'], failure['source'])}, "
                         f"repeat {failure['repeat']:02d}: {failure.get('type', 'Failure')}: {message}")
        lines += ['', 'Failed fits are separate from pending tasks; no complete-family mean is reported '
                  'when any registered repetition has failed or is missing.']
    lines += ['', '## Exact split accounting', '', *_counts(output), '',
              'The parent study is retrospective: some historical design/test observations were '
              'previously inspected. Repartitioning those data does not create an untouched '
              'prospective test set. Generated examples are disjoint by their registered seeds; '
              'real-data A/B allocation separates identities within each role.', '']
    if plots:
        lines += ['![Null balanced Brier and RDR error](null_summary.png)', '',
                  '[Paper PDF](null_summary.pdf). Dots show individual completed repetitions. '
                  'Black diamonds and bars show the complete-group mean and SD; the bars are not '
                  'confidence intervals. Dashed lines mark the population null optima.', '']
    if ci_plots:
        lines += ['![Null cell-average calibration intervals](null_ci_repeat_00.png)', '',
                  '[CI paper PDF](null_ci_repeat_00.pdf). Repeat 00 is the fixed illustration; '
                  'the dashed horizontal line is the population null truth, 1. C.2 simultaneity '
                  'is within each comparison, not across the five families or repetitions. '
                  'Unavailable C.1 cells are explicit; no intervals are averaged across repetitions.', '']
    lines += ['[Per-fit metrics](per_fit.csv) · [Per-comparison metrics](per_comparison.csv) · '
              '[Frozen protocol](protocol.json) · [A/B data provenance](data/null_data.json)', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    status['report_files'] = ['RESULTS.md', 'per_fit.csv', 'per_comparison.csv'] + plots + ci_plots
    (output / 'report_status.json').write_text(json.dumps(status, indent=2) + '\n')
    return status
