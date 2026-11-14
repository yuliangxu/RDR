#!/usr/bin/env python3
"""Replace only JS in the frozen AGP seven-metric summary with test JS(P,Q).

Replays the five already selected checkpoints per generator on their original
full test arrays. Uses each saved transform and hash-checked inference source;
never trains, selects, or fits a transform. Requires numpy, torch and
threadpoolctl. --verify is archive-free;
--verify-archives also checks every private input hash.
"""
from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import types

sys.dont_write_bytecode = True
from js_pq import evaluate_js_pq

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = Path('/cwork/yx306/RDR/AGP')
BASELINE = ROOT / 'results/AGP/seven_global_metrics_20261026'
RETAINED = ROOT / 'results/AGP/retained_mask_comparison_20261013'
DEFAULT_OUTPUT = ROOT / 'results/AGP/seven_global_metrics_js_pq_20261113'
LABELS = {
    'bernoulli_mixture_icfm': 'Mixture + ICFM',
    'autoregressive_icfm': 'NADE + ICFM',
    'midasim_mask_icfm': 'MIDASim mask + ICFM',
    'midasim': 'Full MIDASim',
}
TOKENS = {
    'bernoulli_mixture_icfm': 'bernoulli_mixture_icfm',
    'autoregressive_icfm': 'autoregressive_icfm',
    'midasim_mask_icfm': 'icfm_zero_20261007',
    'midasim': 'midasim_full_20261004',
}
REPLAY_ATOL = 1e-10
OLD_JS = ('js_p_m_variational_mean', 'js_p_m_variational_sd')
NEW_JS = ('js_p_q_variational_mean', 'js_p_q_variational_sd')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_csv(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def close(a, b, tolerance=1e-12):
    if not math.isfinite(float(a)) or not math.isclose(float(a), float(b), rel_tol=0, abs_tol=tolerance):
        raise ValueError(f'Inconsistent value: {a} versus {b}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Inputs:
    def __init__(self):
        self.hashes = {}

    def check(self, path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f'Input hash mismatch: {path}')
        if str(path) in self.hashes and self.hashes[str(path)] != actual:
            raise ValueError(f'Input changed during evaluation: {path}')
        self.hashes[str(path)] = actual
        return actual

    def recheck(self):
        for path, expected in list(self.hashes.items()):
            self.check(path, expected)


def load_public(inputs):
    for folder, names in ((BASELINE, ('manifest.json', 'global_metrics.csv')),
                          (RETAINED, ('manifest.json', 'metrics.csv', 'selection_ranking.csv'))):
        for name in names:
            inputs.check(folder / name)
    baseline_manifest, retained_manifest = read_json(BASELINE/'manifest.json'), read_json(RETAINED/'manifest.json')
    require(baseline_manifest['status'] == retained_manifest['status'] == 'complete', 'Incomplete source package')
    inputs.check(ROOT/baseline_manifest['source_script'], baseline_manifest['source_script_sha256'])
    inputs.check(BASELINE/'global_metrics.csv', baseline_manifest['files_sha256']['global_metrics.csv'])
    for name in ('metrics.csv', 'selection_ranking.csv'):
        inputs.check(RETAINED/name, retained_manifest['files_sha256'][name])
    inputs.check(RETAINED/'metrics.csv', baseline_manifest['sources_sha256'][str((RETAINED/'metrics.csv').relative_to(ROOT))])
    baseline = read_csv(BASELINE/'global_metrics.csv')
    require([r['method'] for r in baseline] == list(LABELS), 'Unexpected baseline generator order')
    ranking = [r for r in read_csv(RETAINED/'selection_ranking.csv') if r['selected'] == 'True']
    require(len(ranking) == 4 and {r['method'] for r in ranking} == set(LABELS), 'Expected four selected recipes')
    recipes = {r['method']: {k: (float(r[k]) if k == 'output_alpha' else r[k])
                           for k in ('candidate', 'loss', 'output_alpha')} for r in ranking}
    final = [r for r in read_csv(RETAINED/'metrics.csv') if r['study'] == 'final']
    require(len(final) == 20, 'Expected exactly 20 final-fit rows')
    for method in LABELS:
        rows = [r for r in final if r['method'] == method]
        require(len(rows) == 5 and {int(r['repeat']) for r in rows} == set(range(5)), 'Expected final repeats 0..4')
        require(all((int(r['n_p']), int(r['n_q'])) == (2002, 2089) for r in rows), 'Incorrect source test counts')
        b = next(r for r in baseline if r['method'] == method)
        close(b[OLD_JS[0]], statistics.mean(float(r['test_js_p_m_variational']) for r in rows))
        close(b[OLD_JS[1]], statistics.stdev(float(r['test_js_p_m_variational']) for r in rows))
    return baseline, retained_manifest, recipes, {(r['method'], int(r['repeat'])): r for r in final}


def archived_root(method, manifest):
    paths = [Path(p) for p in manifest['inputs_sha256']
             if p.endswith('/final/00/scores.csv') and TOKENS[method] in Path(p).parts]
    require(len(paths) == 1, f'Ambiguous archived score root for {method}')
    return paths[0].parents[2]


def frozen_workflow(root, protocol, inputs):
    import numpy as np
    import torch

    # Execute the original inference definitions verbatim from the sealed AST.
    # This avoids importing training-only dependencies (some mask snapshots
    # lack their empty utils/__init__.py). No archived source file is modified.
    relative = 'experiments/AGP/selected_workflow.py'
    inputs.check(root/'source'/relative, protocol['source_sha256'][relative])
    path = root/'source/experiments/AGP/selected_workflow.py'
    names = {'DeepLogitMLP', '_clr', 'transform_ilr'}
    definitions = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    require({node.name for node in definitions} == names and len(definitions) == 3, 'Missing original inference definitions')
    namespace = dict(np=np, torch=torch, nn=torch.nn)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return types.SimpleNamespace(**namespace)


def evaluate_fits(archive_root, retained_manifest, recipes, references, inputs):
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    records, identities = [], []
    real_test_hash = None
    for method in LABELS:
        original_root = archived_root(method, retained_manifest)
        root = Path(archive_root) / original_root.relative_to(ARCHIVE)

        def original_check(path):
            original = original_root / path.relative_to(root)
            return inputs.check(path, retained_manifest['inputs_sha256'][str(original)])

        original_check(root/'COMPLETE.json')
        seal = read_json(root/'COMPLETE.json')
        require(seal['status'] == 'complete' and seal['final_fits'] == 5, f'Incomplete final-fit seal: {root}')
        inputs.check(root/'selection/freeze.json', seal['selection_freeze_sha256'])
        freeze = read_json(root/'selection/freeze.json')
        require(freeze['status'] == 'frozen' and freeze['test_scores_read'] is False, 'Invalid selection freeze')
        inputs.check(root/'selection/selection.json', freeze['outputs']['selection/selection.json'])
        require(read_json(root/'selection/selection.json')['chosen'] == recipes[method], 'Frozen recipe mismatch')
        inputs.check(root/'protocol.json', seal['protocol_sha256'])
        protocol = read_json(root/'protocol.json')
        inputs.check(root/'config.json', protocol['config_sha256'])
        cfg = read_json(root/'config.json')
        workflow = frozen_workflow(root, protocol, inputs)
        arrays = {}
        array_paths = {}
        for side, count in (('p', 2002), ('q', 2089)):
            spec = protocol['roles']['final']['arrays']['test_'+side]
            if side == 'p':
                if real_test_hash is None:
                    real_test_hash = spec['sha256']
                require(spec['sha256'] == real_test_hash, 'Real test cohorts differ between generators')
            path = root/spec['path']
            inputs.check(path, spec['sha256'])
            a = np.load(path, allow_pickle=False)
            require(list(a.shape) == spec['shape'] == [count, 614] and str(a.dtype) == spec['dtype'], 'Incorrect full test array')
            arrays[side], array_paths[side] = a, str(path)
        torch.set_num_threads(cfg['threads'])
        torch.use_deterministic_algorithms(True)
        for repeat in range(5):
            folder = root/'final'/f'{repeat:02d}'
            original_check(folder/'COMPLETE.json')
            marker = read_json(folder/'COMPLETE.json')
            inputs.check(folder/'COMPLETE.json', seal['assessment_completions'][f'final/{repeat:02d}/COMPLETE.json'])
            require(marker['status'] == 'complete' and marker['stage'] == 'final'
                    and marker['task'] == dict(recipes[method], repeat=repeat), 'Final checkpoint identity mismatch')
            for field, path in (('config_sha256', root/'config.json'), ('protocol_sha256', root/'protocol.json'),
                                ('selection_freeze_sha256', root/'selection/freeze.json')):
                inputs.check(path, marker[field])
            for name in ('scores.csv', 'predictions.npz', 'model.pt', 'transform.npz'):
                inputs.check(folder/name, marker['outputs'][name])
            original_check(folder/'scores.csv')
            scores = {'p': [], 'q': []}
            order = {'p': [], 'q': []}
            for row in read_csv(folder/'scores.csv'):
                require(row['role'] == 'test' and row['distribution'] in ('P', 'Q'), 'Unexpected score role or source')
                side = row['distribution'].lower()
                scores[side].append(float(row['rdr']))
                order[side].append(int(row['array_row']))
            for side, count in (('p', 2002), ('q', 2089)):
                require(order[side] == list(range(count)), 'Score row alignment/count mismatch')
            with np.load(folder/'predictions.npz', allow_pickle=False) as predictions:
                require(set(predictions.files) == {'test_p', 'test_q'}, 'Unexpected prediction archive fields')
                for side in ('p', 'q'):
                    require(np.array_equal(predictions['test_'+side], scores[side]), 'CSV/NPZ predictions differ')
            checkpoint = torch.load(folder/'model.pt', map_location='cpu', weights_only=True)
            require(checkpoint['task'] == marker['task'] and checkpoint['config'] == cfg, 'Checkpoint task/config mismatch')
            model = workflow.DeepLogitMLP(613, cfg['hidden_dimension'], cfg['blocks'])
            model.load_state_dict(checkpoint['model'], strict=True)
            model.eval()
            with np.load(folder/'transform.npz', allow_pickle=False) as saved:
                state = {key: saved[key] for key in saved.files}
            logits, errors = {}, {}
            with threadpool_limits(limits=cfg['threads']), torch.inference_mode():
                for side in ('p', 'q'):
                    # Original inference uses a batch of at most 4096 per source.
                    x = torch.as_tensor(workflow.transform_ilr(arrays[side], state))
                    t = float(marker['task']['output_alpha']) * model(x).double()
                    logits[side] = t.numpy()
                    errors[side] = float(np.max(np.abs((2*torch.sigmoid(t)).numpy()-scores[side])))
                    require(errors[side] <= REPLAY_ATOL, f'Checkpoint replay disagrees with archive: {folder} {side}: {errors[side]}')
            values = evaluate_js_pq(logits['p'], logits['q'], representation='logits')
            # Independent torch BCE-with-logits evaluation uses separate means.
            tp, tq = (torch.from_numpy(logits[s]) for s in ('p', 'q'))
            bce = float(.5*torch.nn.functional.binary_cross_entropy_with_logits(tp, torch.ones_like(tp))
                        + .5*torch.nn.functional.binary_cross_entropy_with_logits(tq, torch.zeros_like(tq)))
            close(values['js_p_q_variational'], math.log(2)-bce)
            archived = evaluate_js_pq(scores['p'], scores['q'])
            ref = references[method, repeat]
            for side in ('p', 'q'):
                close(math.fsum(scores[side])/len(scores[side]), ref['mean_rdr_'+side])
            endpoints = {side: sum(v in (0, 2) for v in scores[side]) for side in ('p', 'q')}
            records.append(dict(method=method, label=LABELS[method], study='final', repeat=repeat,
                selected_candidate=marker['task']['candidate'], selected_loss=marker['task']['loss'],
                output_alpha=marker['task']['output_alpha'], **values,
                torch_balanced_bce=bce, archived_rdr_js_p_q_variational=archived['js_p_q_variational'],
                checkpoint_minus_archived_js=values['js_p_q_variational']-archived['js_p_q_variational'],
                old_test_js_p_m_variational=ref['test_js_p_m_variational'],
                archived_endpoint_count_p=endpoints['p'], archived_endpoint_count_q=endpoints['q'],
                replay_max_abs_rdr_error_p=errors['p'], replay_max_abs_rdr_error_q=errors['q'],
                score_path=str(folder/'scores.csv'), score_sha256=sha(folder/'scores.csv'),
                checkpoint_path=str(folder/'model.pt'), checkpoint_sha256=sha(folder/'model.pt'),
                transform_path=str(folder/'transform.npz'), transform_sha256=sha(folder/'transform.npz'),
                completion_path=str(folder/'COMPLETE.json'), completion_sha256=sha(folder/'COMPLETE.json')))
            identities.append(dict(method=method, repeat=repeat, task=marker['task'],
                checkpoint=str(folder/'model.pt'), checkpoint_sha256=sha(folder/'model.pt'),
                selection_freeze=str(root/'selection/freeze.json'), selection_freeze_sha256=marker['selection_freeze_sha256'],
                protocol=str(root/'protocol.json'), workflow=str(root/'source/experiments/AGP/selected_workflow.py'),
                test_array_paths=array_paths, original_training=checkpoint['training'],
                logits_sha256={s: hashlib.sha256(logits[s].astype('<f8').tobytes()).hexdigest() for s in ('p', 'q')},
                numerical_input='float64 scaled logits from frozen float32 CPU network and saved ILR transform'))
            print(f'{method} final/{repeat:02d}: JS(P,Q)={values["js_p_q_variational"]:.12f}; '
                  f'n_P=2002 n_Q=2089; replay error={max(errors.values()):.3g}', flush=True)
    return records, identities, dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)


def assemble(baseline, per_fit):
    summary, comparison = [], []
    require(len(per_fit) == 20, 'Expected exactly 20 fit estimates')
    for b in baseline:
        method = b['method']
        fits = [r for r in per_fit if r['method'] == method]
        require(len(fits) == 5 and {int(r['repeat']) for r in fits} == set(range(5)), 'Wrong final-fit set')
        require(all((int(r['n_p']), int(r['n_q'])) == (2002, 2089) for r in fits), 'Wrong full test counts')
        values = [float(r['js_p_q_variational']) for r in fits]
        avg, sd = statistics.mean(values), statistics.stdev(values)
        # Separate explicit ddof=1 check of sample SD, with all fits retained.
        close(sd, math.sqrt(math.fsum((v-avg)**2 for v in values)/4))
        new = {NEW_JS[OLD_JS.index(k)] if k in OLD_JS else k:
               (avg if k == OLD_JS[0] else sd) if k in OLD_JS else v for k, v in b.items()}
        summary.append(new)
        comparison.append(dict(method=method, label=LABELS[method],
            old_js_p_m_variational_mean=b[OLD_JS[0]], old_js_p_m_variational_sd=b[OLD_JS[1]],
            new_js_p_q_variational_mean=avg, new_js_p_q_variational_sd=sd,
            pq_minus_pm_mean=avg-float(b[OLD_JS[0]]), fits=5, units='nats'))
    return summary, comparison


def markdown_table(rows, decimals):
    def fmt(value):
        return f'{float(value):.{decimals}f}'
    def pair(row, mean, sd):
        return fmt(row[mean]) + ' ± ' + fmt(row[sd])
    lines = ['| Generator | FD ↓ | PRD max F1 ↑ | Prevalence MAE ↓ | Richness KS ↓ | Shannon KS ↓ | JS(P,Q), nats ↓ | Spearman correlation RMSE ↓ |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        cells = [LABELS[r['method']], fmt(r['fd_sqrt']), pair(r, 'prd_max_f1_sqrt_mean', 'prd_max_f1_sqrt_sd'),
                 fmt(r['prevalence_mae']), fmt(r['richness_ks']), fmt(r['shannon_ks']),
                 pair(r, *NEW_JS), fmt(r['spearman_rmse'])]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def report_text(rows, comparison, decimals):
    comparison_lines = ['| Generator | Old JS(P,M) mean | New JS(P,Q) mean | Difference |', '|---|---:|---:|---:|']
    for r in comparison:
        comparison_lines.append('| ' + ' | '.join([r['label']] +
            [f'{float(r[k]):.{decimals}f}' for k in ('old_js_p_m_variational_mean', 'new_js_p_q_variational_mean', 'pq_minus_pm_mean')]) + ' |')
    return ('# Four-generator AGP seven-metric summary with JS(P,Q)\n\n'
        + markdown_table(rows, decimals) + '\n'
        'All metrics use the original full test cohorts: 2,002 P and 2,089 Q per generator. '
        'FD and PRD use the original square-root abundance features. All six non-JS metrics '
        'are copied with their original CSV cell strings. PRD shows the original five-quantizer '
        'mean ± sample SD; JS shows mean ± sample SD over the five selected RDR fits (denominator 4).\n\n'
        '`JS_PQ_hat = 0.5 mean_P(log(r_hat)) + 0.5 mean_Q(log(2-r_hat))`, in nats, '
        'with equal source weights despite unequal counts. Scaled checkpoint logits `t` use '
        '`log(r_hat)=log(2)-softplus(-t)` and `log(2-r_hat)=log(2)-softplus(t)`. '
        'Every frozen fit is evaluated, including chi-square-trained NADE. No training, '
        'reselection, transform fitting, clipping, subsampling, or truncation of negative estimates occurs.\n\n'
        'The checkpoint replay recovers the one exact endpoint in archived Full MIDASim repeat 4 '
        '(P score rounded to 2). Float64 log evaluation uses the original float32 network outputs '
        'and saved ILR transforms. Per-fit CSV records replay agreement and the difference from '
        'direct evaluation of archived RDR scores. Original JS(P,M) files remain unchanged.\n\n'
        + '\n'.join(comparison_lines) + '\n\n'
        'These are fitted held-out variational estimates, not exact population divergences. '
        'Initialization SD on fixed observations is not a sampling confidence interval. '
        'The assessment is retrospective; finite-sample estimates need not be population lower bounds.\n\n'
        'Reproduce from the private HPC archive into a fresh destination:\n\n'
        '```bash\npython -B experiments/AGP/seven_global_metrics_pq.py \\\n'
        '  --output results/AGP/seven_global_metrics_js_pq_reproduction\n```\n\n'
        'Check saved results without private artifacts:\n\n'
        '```bash\npython -B experiments/AGP/seven_global_metrics_pq.py \\\n'
        '  --verify results/AGP/seven_global_metrics_js_pq_20261113\n```\n\n'
        '`--verify-archives` additionally rehashes private inputs; full reproduction replays '
        'the checkpoints. Private arrays, row identifiers, logits and checkpoints are not copied '
        'into this output package.\n')


def verify_repository_inputs(output, manifest):
    """Use declared exact input snapshots when historical metadata has drifted."""
    snapshots = manifest.get('repository_input_snapshots', {})
    require(set(snapshots) <= set(manifest['repository_inputs_sha256']), 'Unknown snapshot input')
    for relative, expected in manifest['repository_inputs_sha256'].items():
        path = Path(output)/snapshots[relative] if relative in snapshots else ROOT/relative
        require(sha(path) == expected, f'Repository input hash mismatch: {relative}')


def verify(output, verify_archives=False):
    output = Path(output)
    manifest = read_json(output/'manifest.json')
    require(manifest['status'] == 'complete', 'Incomplete package')
    for name, expected in manifest['files_sha256'].items():
        require(sha(output/name) == expected, f'Output hash mismatch: {name}')
    verify_repository_inputs(output, manifest)
    per_fit = read_csv(output/'per_fit_js_p_q.csv')
    expected_summary, expected_comparison = assemble(read_csv(BASELINE/'global_metrics.csv'), per_fit)
    for name, expected in (('global_metrics.csv', expected_summary), ('js_target_comparison.csv', expected_comparison)):
        actual = read_csv(output/name)
        require(len(actual) == len(expected), f'Incorrect row count: {name}')
        for a, b in zip(actual, expected):
            require(set(a) == set(b), f'Incorrect fields: {name}')
            for key, value in b.items():
                # This checks exact preservation of every non-JS CSV cell.
                if isinstance(value, str):
                    require(a[key] == value, f'Changed source cell: {name} {key}')
                else:
                    close(a[key], value)
    for row in per_fit:
        close(row['js_p_q_variational'], .5*float(row['mean_log_r_p'])+.5*float(row['mean_log_two_minus_r_q']))
        close(row['js_p_q_variational'], math.log(2)-float(row['torch_balanced_bce']))
        require(max(float(row['replay_max_abs_rdr_error_'+s]) for s in ('p', 'q')) <= REPLAY_ATOL, 'Replay error exceeds tolerance')
    for name, digits in (('table.md', 6), ('table_3dp.md', 3)):
        require((output/name).read_text() == report_text(expected_summary, expected_comparison, digits), f'Table mismatch: {name}')
    if verify_archives:
        for path, expected in manifest['archive_inputs_sha256'].items():
            require(sha(path) == expected, f'Archive hash mismatch: {path}')
    print(f'Verified 20 final fits, full test counts, balanced BCE identity, ddof=1 SD, '
          f'unchanged non-JS cells, tables and hashes: {output}')


def generate(output, archive_root):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a fresh output directory to preserve previous packages')
    inputs = Inputs()
    baseline, retained, recipes, references = load_public(inputs)
    for path in (Path(__file__), Path(__file__).with_name('js_pq.py'),
                 ROOT/'tests/test_agp_js_pq.py', ROOT/'tests/test_agp_seven_global_metrics_pq.py',
                 ROOT/'experiments/AGP/seven_global_metrics.py'):
        inputs.check(path)
    per_fit, identities, environment = evaluate_fits(archive_root, retained, recipes, references, inputs)
    rows, comparison = assemble(baseline, per_fit)
    inputs.recheck()
    output.mkdir(parents=True)
    retained_relative = str((RETAINED/'manifest.json').relative_to(ROOT))
    snapshot_relative = 'source_snapshot/retained_manifest.json'
    snapshot = output/snapshot_relative
    snapshot.parent.mkdir()
    snapshot.write_bytes((RETAINED/'manifest.json').read_bytes())
    require(sha(snapshot) == inputs.hashes[str((RETAINED/'manifest.json').resolve())], 'Snapshot input changed')
    for name, values in (('per_fit_js_p_q.csv', per_fit), ('global_metrics.csv', rows), ('js_target_comparison.csv', comparison)):
        write_csv(output/name, values)
    for name, digits in (('table.md', 6), ('table_3dp.md', 3)):
        (output/name).write_text(report_text(rows, comparison, digits))
    public, private = {}, {}
    for path, digest in inputs.hashes.items():
        p = Path(path)
        if p.is_relative_to(ROOT):
            public[str(p.relative_to(ROOT))] = digest
        else:
            private[path] = digest
    manifest = dict(status='complete', kind='four_generator_full_test_seven_metric_js_p_q',
        created_utc=datetime.now(timezone.utc).isoformat(),
        repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        repository_worktree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip()),
        code_provenance='Repository HEAD plus exact hashes of working scripts; new scripts may be untracked.',
        methods=list(LABELS), n_p=2002, n_q_per_generator=2089, final_fits_per_generator=5,
        new_model_fits=0, reselections=0, transform_fits=0, metric_count=7,
        baseline='results/AGP/seven_global_metrics_20261026/global_metrics.csv',
        unchanged_non_js_fields=[k for k in baseline[0] if k not in OLD_JS],
        js=dict(target='JS(P,Q)', M='(P+Q)/2', rdr='r_hat approximates 2p/(p+q)',
            formula='0.5*mean_P(log(r_hat)) + 0.5*mean_Q(log(2-r_hat))',
            log_r_from_scaled_logit='log(2)-softplus(-t)', log_two_minus_r_from_scaled_logit='log(2)-softplus(t)',
            logit='t=output_alpha*model(transformed_x); r_hat=2*sigmoid(t)',
            identity='JS_PQ_hat=log(2)-balanced_BCE_with_logits(t)',
            units='nats', logarithms='natural', source_weights=dict(P=.5, Q=.5),
            sd='sample standard deviation across all five frozen fits; ddof=1, denominator=4',
            negative_estimates='retained without truncation', empirical_upper_bound=math.log(2)),
        numerical_handling=dict(primary='stable scaled-logit evaluation for every selected checkpoint',
            source_loading='Exact DeepLogitMLP, _clr, transform_ilr AST definitions from hash-sealed selected_workflow.py; training imports not executed. Empty utils/__init__.py absent in mask source snapshots is not needed for inference.',
            arithmetic='float64 softplus/log1p, math.fsum source means; frozen float32 model inference on CPU',
            preprocessing='Original saved ILR state applied by hash-checked archived selected_workflow.py; no transform refit',
            archived_endpoints_p=sum(r['archived_endpoint_count_p'] for r in per_fit),
            archived_endpoints_q=sum(r['archived_endpoint_count_q'] for r in per_fit),
            endpoint_recovery='Full MIDASim final/04 has one P=2; evaluated from finite checkpoint logit',
            clipping=False, discarded_fits=0, subsampling=False,
            replay_absolute_tolerance=REPLAY_ATOL,
            maximum_rdr_replay_error=max(r['replay_max_abs_rdr_error_'+s] for r in per_fit for s in ('p', 'q')),
            maximum_js_difference_from_archived_rdr=max(abs(r['checkpoint_minus_archived_js']) for r in per_fit)),
        verification=dict(analytic_and_stability_tests='tests/test_agp_js_pq.py',
            summary_preservation_and_fit_count_tests='tests/test_agp_seven_global_metrics_pq.py',
            independent_bce='torch.nn.functional.binary_cross_entropy_with_logits evaluated anew on full test logits',
            exactly_five_final_fits_per_generator=True, test_counts_per_fit=[2002, 2089],
            sample_sd_denominator=4, other_six_metrics='all non-JS baseline CSV cell strings preserved exactly',
            checkpoint_predictions='compared to hash-sealed scores.csv and predictions.npz, with exact CSV/NPZ alignment'),
        limitations=['Fitted variational test estimates, not exact population divergences.',
            'Five-fit SD is initialization variation conditional on fixed cohorts, not sampling uncertainty.',
            'Retrospective assessment; finite-sample estimates are not guaranteed population lower bounds.',
            'Portable verification checks saved arithmetic and repository hashes; full replay requires private HPC artifacts.'],
        environment=environment, selected_recipes=recipes, fit_identities=identities,
        repository_input_snapshots={retained_relative: snapshot_relative},
        repository_inputs_sha256=public, archive_inputs_sha256=private,
        files_sha256={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False)+'\n')
    verify(output)
    for row in rows:
        print(f'{LABELS[row["method"]]}: {row[NEW_JS[0]]:.12f} ± {row[NEW_JS[1]]:.12f} nats')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive-root', type=Path, default=ARCHIVE)
    parser.add_argument('--verify', type=Path, help='Verify saved arithmetic and repository/output hashes')
    parser.add_argument('--verify-archives', action='store_true', help='Also rehash private archives when verifying')
    args = parser.parse_args()
    if args.verify:
        verify(args.verify, args.verify_archives)
    else:
        generate(args.output, args.archive_root)
