"""Report fixed-architecture loss/activation selection, without test scoring."""
import json

from experiments.CelebA.selection_report import (
    METRICS, _aggregate, _csv, _plots, _split_accounting, _table, _validate,
)


def build_report(output, protocol, rows, selected, accepted_selection=None):
    cfg = protocol['config']
    _validate(rows)
    summaries = _aggregate(rows, cfg['repeats'], 'selection')
    _csv(output / 'per_fit.csv', rows, ('representation', 'branch', 'candidate', 'repeat') + METRICS)
    _csv(output / 'per_candidate.csv', summaries, ('representation', 'branch', 'candidate', 'repeats'))
    displayed_selection = selected or accepted_selection
    plots = _plots(output, summaries, cfg, displayed_selection,
                   selection_label='accepted setting' if accepted_selection and not selected else 'baseline selection')
    failures = []
    for task in protocol['tasks']:
        directory = output / 'fits' / task['representation'] / task['branch'] / task['candidate'] / f"repeat_{task['repeat']:02d}"
        if (directory / 'FAILED.json').exists() and not (directory / 'COMPLETE.json').exists():
            failures.append({**task, **json.loads((directory / 'FAILED.json').read_text())})
    complete = len(rows) == len(protocol['tasks']) and selected is not None
    status = dict(complete=complete, baseline_fits=len(rows), expected_baseline_fits=len(protocol['tasks']),
                  scope='loss_and_activation_selection_only', test_assets_loaded=False,
                  architecture_selection=False, final_assessment=False, failed_fits=len(failures))
    lines = ['# CelebA expanded-data loss and activation selection', '',
             f"**Status: {'complete' if complete else 'incomplete'}; {len(rows)}/{len(protocol['tasks'])} verified fits.**", '',
             'Fixed baseline architectures; four losses, four sigmoid slopes, five paired repetitions. '
             'Separate selections for both P/Q pairs and both representations. The output is r=2 sigmoid(alpha z).', '',
             'Checkpoints minimize stopping-set balanced Brier. Selection uses the paired one-SE Brier shortlist, '
             'then minimum supported local absolute Gap, with at least 0.99 supported mass in every repetition. '
             'Calibration and evaluation rows for selection are separate. P and Q each have weight one half.', '',
             'Hellinger backward uses an algebraic exponential scale factor and float64 gradient norms. '
             'The factor is accounted for during gradient clipping; the objective and ratio parameterization '
             'are unchanged, with no ratio clipping.', '',
             'Architecture sensitivity and final calibration/evaluation are outside this run. '
             'Final image/feature arrays remain unopened by this model-selection workflow.', '',
             '## Accepted settings' if accepted_selection and not selected else '## Selected settings', '',
             '| Representation / pair | Loss | Sigmoid slope |', '| --- | --- | --- |']
    for level in cfg['levels']:
        for branch in cfg['branches']:
            choice = (displayed_selection or {}).get('selections', {}).get(f'{level}/{branch}', {}).get('chosen', {})
            lines.append(f"| {level} / {branch} | {choice.get('loss', 'Pending')} | {choice.get('output_alpha', 'Pending')} |")
    if accepted_selection and not selected:
        lines += ['', 'These settings were accepted by the user for final assessment after reviewing the '
                  'Brier-shortlist/local-Gap recommendations. The pixel/lower recommendation excluded the '
                  'incomplete Hellinger/slope-4 candidate. Acceptance does not mark the 319/320 selection grid complete.', '',
                  '[Separate final assessment](../final_evaluation_expanded_20261003/RESULTS.md).']
    if failures:
        lines += ['', '## Failed fits and figure caption', '',
                  '“Failed (k/n)” denotes k failed fits among n planned repetitions. '
                  'The five-repeat aggregate is omitted; successful repetitions are retained, '
                  'but are not averaged as if the candidate had completed. This is not a running or pending experiment.', '']
        for failure in failures:
            lines.append(f"- {failure['representation']} / P versus Q_{failure['branch']}, "
                         f"{failure['loss']}, sigmoid slope {failure['output_alpha']:g}, "
                         f"repeat {failure['repeat']:02d}: `{failure.get('type')}: {failure.get('message')}`.")
        lines += ['', 'The recorded failure is numerical: the training objective became nonfinite '
                  '(NaN or infinity), triggering the explicit finite-value check. '
                  'The Hellinger objective contains inverse-square-root ratio terms, which can become '
                  'very large when predicted RDR approaches zero. Gradient scaling does not guarantee '
                  'a finite forward objective. The saved log does not establish the exact intermediate '
                  'quantity that first became nonfinite, so this mechanism is an explanation of the risk, '
                  'not a confirmed root-cause trace.', '',
                  '**Paper caption:** Validation model comparison over four losses and four sigmoid slopes, '
                  'using five paired training repetitions for each P/Q pair and representation. '
                  'Cells show mean balanced Brier or supported local absolute Gap. Red boxes indicate '
                  'the accepted settings. For pixel P versus Q_l with Hellinger loss and slope 4, '
                  'one of five fits terminated at epoch 7 because the training objective was nonfinite; '
                  'the aggregate is therefore marked Failed (1/5).']
    lines += ['', '## Exact split counts', '', *_split_accounting(output, protocol), '',
              '## Validation results', '', 'Entries show mean (SD) over the five registered repetitions. '
              'Incomplete candidates are not averaged over fewer runs.', '',
              *_table(summaries, ('brier', 'local_gap', 'supported_mass')), '',
              *_table(summaries, ('middle_mass', 'middle_supported_fraction', 'middle_local_gap'))]
    for stem in plots:
        lines += ['', f'![Brier and local Gap]({stem}.png)']
    lines += ['', 'Repetitions measure initialization/training-order variation conditional on these data. '
              'Local score-cell intervals remain nominal image-level diagnostics. Previously inspected '
              'historical data make this a retrospective study.', '',
              '[Per-fit metrics](per_fit.csv) · [Per-candidate metrics](per_candidate.csv) · '
              '[Protocol](protocol.json) · [Selected settings](baseline_selection.json)', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines))
    (output / 'report_status.json').write_text(json.dumps(status, indent=2) + '\n')
    return status
