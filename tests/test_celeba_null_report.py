"""Null reporting keeps failures visible and signed finite-sample diagnostics."""
import csv
import hashlib
import json

import pytest

from experiments.CelebA import null_report as report


def protocol(repeats=5):
    groups = [('feature', 'lower'), ('feature', 'upper'), ('pixel', 'lower'),
              ('pixel', 'upper'), ('pixel', 'real')]
    tasks = [dict(representation=level, source=source, repeat=repeat,
                  candidate='baseline_js_a0p5' if level == 'feature' else 'baseline_js_a2',
                  loss='js', output_alpha=.5 if level == 'feature' else 2)
             for level, source in groups for repeat in range(repeats)]
    return {'config': {'repeats': repeats, 'ci_alpha': .05}, 'tasks': tasks}


def rows():
    return [{**task, 'brier': .249 + .001 * task['repeat'], 'local_gap': .02,
             'supported_mass': 1., 'c2_distance': .003, 'c2_width': .04,
             'middle_mass': .9, 'middle_supported_fraction': 1., 'middle_local_gap': .02,
             'rdr_mean_p': 1.01, 'rdr_mean_q': 1.01, 'rdr_mse_one': .0004,
             'rdr_mae_one': .02, 'rdr_rmse_one': .02, 'excess_brier_true': .0001,
             'empirical_brier_excess': .249 + .001 * task['repeat'] - .25,
             'h2_variational': -.001, 'h2_plugin': -.005,
             'h2_clip_rate_p': 0., 'h2_clip_rate_q': 0.}
            for task in protocol()['tasks']]


@pytest.fixture
def no_plots(monkeypatch):
    monkeypatch.setattr(report, '_plots', lambda *args: [])
    monkeypatch.setattr(report, '_ci_plot', lambda *args: [])


def test_complete_and_signed_diagnostics_preserved(tmp_path, no_plots):
    status = report.build_report(tmp_path, protocol(), rows())
    assert status['complete'] and status['completed_fits'] == 25
    assert status['failed_fits'] == status['pending_fits'] == 0
    with (tmp_path / 'per_comparison.csv').open() as stream:
        summaries = list(csv.DictReader(stream))
    assert len(summaries) == 5
    assert all(float(row['h2_plugin_mean']) == pytest.approx(-.005) for row in summaries)
    assert float(summaries[0]['brier_mean']) == pytest.approx(.251)
    assert float(summaries[0]['brier_sd']) == pytest.approx(.001581138830084191)
    text = (tmp_path / 'RESULTS.md').read_text()
    assert 'not uncertainty across independently resampled datasets' in text
    assert 'can be negative' in text
    assert 'not truncated to zero' in text
    assert 'separate final calibration' in text
    assert not (tmp_path / 'COMPLETE.json').exists()


def test_failed_fit_is_not_pending_or_partial_mean(tmp_path, no_plots):
    failed = {**protocol()['tasks'][-1], 'type': 'FloatingPointError', 'message': 'nonfinite objective'}
    status = report.build_report(tmp_path, protocol(), rows()[:-1], [failed])
    assert not status['complete']
    assert status['failed_fits'] == 1 and status['pending_fits'] == 0
    with (tmp_path / 'per_comparison.csv').open() as stream:
        incomplete = list(csv.DictReader(stream))[-1]
    assert incomplete['brier_mean'] == incomplete['rdr_mse_one_mean'] == ''
    text = (tmp_path / 'RESULTS.md').read_text()
    assert '4/5 complete; 1 failed; 0 pending' in text
    assert 'FloatingPointError: nonfinite objective' in text


def test_empty_and_pending_report_registers_every_family(tmp_path, no_plots):
    status = report.build_report(tmp_path, protocol(), [])
    assert status['completed_fits'] == 0 and status['pending_fits'] == 25
    with (tmp_path / 'per_comparison.csv').open() as stream:
        assert len(list(csv.DictReader(stream))) == 5
    assert '0/5 complete; 0 failed; 5 pending' in (tmp_path / 'RESULTS.md').read_text()


@pytest.mark.parametrize('kind', ['duplicate', 'unregistered', 'setting', 'nonfinite', 'derived'])
def test_malformed_completed_rows_rejected(tmp_path, no_plots, kind):
    values = rows()
    if kind == 'duplicate':
        values.append(values[0])
    elif kind == 'unregistered':
        values[0]['source'] = 'other'
    elif kind == 'setting':
        values[0]['output_alpha'] = 4
    elif kind == 'nonfinite':
        values[0]['brier'] = float('nan')
    else:
        values[0]['excess_brier_true'] = .2
    with pytest.raises(ValueError):
        report.build_report(tmp_path, protocol(), values)


def test_failed_row_must_be_missing_registered_task(tmp_path, no_plots):
    with pytest.raises(ValueError, match='completed'):
        report.build_report(tmp_path, protocol(), rows(), [protocol()['tasks'][0]])


def test_missing_diagnostic_not_silently_averaged(tmp_path, no_plots):
    values = rows()
    values[0]['middle_local_gap'] = None
    report.build_report(tmp_path, protocol(), values)
    summaries = report._aggregate(protocol(), values, [])
    assert summaries[0]['middle_local_gap_mean'] is None
    assert summaries[0]['middle_local_gap_available_repeats'] == 4
    assert summaries[0]['brier_mean'] is not None


def test_exact_counts_are_visible(tmp_path, no_plots):
    (tmp_path / 'data').mkdir()
    metadata = {'roles': {'test_evaluation': {'real': {'parent_count': 19,
                'sides': {'p': {'count': 10, 'identity_count': 3},
                          'q': {'count': 9, 'identity_count': 2}}}}}}
    (tmp_path / 'data/null_data.json').write_text(json.dumps(metadata))
    report.build_report(tmp_path, protocol(), rows())
    assert '| test_evaluation | Real P | 19 | 10 | 9 | 3 | 2 |' in (tmp_path / 'RESULTS.md').read_text()


def test_summary_and_ci_figures_created_from_completed_rows(tmp_path):
    pytest.importorskip('matplotlib')
    cells = [dict(left=i / 2, right=(i + 1) / 2, c2_lower=.8, c2_upper=1.2,
                  calibrated_rdr=1., neural_mean=1., c1_lower=.9, c1_upper=1.1,
                  c1_unavailable=False) for i in range(4)]
    directory = tmp_path / 'evaluation/feature/lower/repeat_00'
    directory.mkdir(parents=True)
    (directory / 'cells.json').write_text(json.dumps(cells))
    # A plausible but unverified directory must never be opened by CI plotting.
    ignored = tmp_path / 'evaluation/pixel/real/repeat_00'
    ignored.mkdir(parents=True)
    (ignored / 'cells.json').write_text('not json')
    values = rows()[:5]
    status = report.build_report(tmp_path, protocol(), values)
    for name in ('null_summary.png', 'null_summary.pdf', 'null_ci_repeat_00.png', 'null_ci_repeat_00.pdf'):
        assert (tmp_path / name).stat().st_size > 0
        assert name in status['report_files']
    before = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
              for name in status['report_files'] + ['report_status.json']}
    report.build_report(tmp_path, protocol(), values)
    after = {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in before}
    assert before == after
