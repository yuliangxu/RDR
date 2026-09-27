"""Scientific invariants for the AGP architecture and retrospective CI workflow."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import helmert
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import AGP_architecture_diagnostics as workflow


class ArchitectureDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_archived_ilr_initializer_is_exact_for_both_baselines(self):
        basis = helmert(614, full=False)
        expected = workflow.composition.make_model(614, 'ilr', 20260918, basis)
        inputs = torch.tensor(np.random.default_rng(27).normal(size=(9, 613)), dtype=torch.float32)
        for method in ('baseline_p5', 'baseline_p30'):
            model = workflow.make_model(614, method, 20260918, basis)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 21793)
            self.assertEqual(list(model.state_dict()), list(expected.state_dict()))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected.state_dict()[name]), name)
            self.assertTrue(torch.equal(model(inputs), expected(inputs)))

    def test_deep_control_has_exactly_matched_parameters_and_only_skips_differ(self):
        basis = helmert(614, full=False)
        deep = workflow.make_model(614, 'deep_p30', 19, basis)
        residual = workflow.make_model(614, 'residual_p30', 19, basis)
        self.assertEqual(list(deep.state_dict()), list(residual.state_dict()))
        self.assertEqual(sum(p.numel() for p in deep.parameters()), 177793)
        self.assertEqual(sum(p.numel() for p in residual.parameters()), 177793)
        for name, value in deep.state_dict().items():
            self.assertTrue(torch.equal(value, residual.state_dict()[name]), name)
        inputs = torch.tensor(np.random.default_rng(17).normal(size=(7, 613)), dtype=torch.float32)
        self.assertFalse(torch.equal(deep(inputs), residual(inputs)))
        for block in residual.modules():
            if isinstance(block, workflow.TwoLayerBlock):
                block.residual = False
        self.assertTrue(torch.equal(deep(inputs), residual(inputs)))
        wide = workflow.make_model(614, 'wide_p30', 19, basis)
        self.assertEqual(sum(p.numel() for p in wide.parameters()), 111745)
        for model in (wide, deep, residual):
            scores = model(inputs)
            self.assertEqual(tuple(scores.shape), (7, 1))
            self.assertTrue(bool(((scores > 0) & (scores <= 2)).all()))
            self.assertEqual(model.model[-1].alpha, 2.)
            self.assertEqual(model.model[-1].scale, 2.)

    def test_all_architectures_fit_without_accessing_test_values(self):
        class ForbiddenTestArray:
            def __array__(self, *args, **kwargs):
                raise AssertionError('Model fitting accessed a test array')
        generator = np.random.default_rng(88)
        arrays = {(d, role): generator.normal(size=(n, 3)).astype(np.float32)
                  for d, role, n in [('P', 'train', 5), ('Q', 'train', 7),
                                     ('P', 'validation', 3), ('Q', 'validation', 4)]}
        arrays.update({(d, 'test'): ForbiddenTestArray() for d in ('P', 'Q')})
        args = SimpleNamespace(min_epochs=1, max_epochs=3, patience=3)
        for method in workflow.METHODS:
            with self.subTest(method=method), contextlib.redirect_stdout(io.StringIO()):
                model = workflow.make_model(4, method, 14, helmert(4, full=False))
                history, training = workflow.composition.fit_model(model, arrays, args)
                self.assertEqual(training['best_validation_loss'], history.validation_loss.min())
                self.assertEqual(training['best_epoch'], int(history.loc[history.validation_loss.idxmin(), 'epoch']))

    def test_midpoint_loss_uses_separate_means_with_unequal_sizes(self):
        model = torch.nn.Identity()
        p = torch.tensor([[.2], [.8], [1.7]])
        q = torch.tensor([[.1], [.3], [.4], [1.1], [1.8]])
        expected = .5 * p.rsqrt().mean() + .25 * p.sqrt().mean() + .25 * q.sqrt().mean() - 1
        actual = workflow.composition.midpoint_loss(model, p, q)
        self.assertEqual(float(actual), float(expected))
        repeated = workflow.composition.midpoint_loss(model, p, q.repeat_interleave(4, dim=0))
        self.assertAlmostEqual(float(actual), float(repeated), places=7)
        pooled = .5 * p.rsqrt().mean() + .5 * torch.cat((p, q)).sqrt().mean() - 1
        self.assertGreater(abs(float(actual - pooled)), .001)

    def test_validation_selection_ignores_train_and_test_and_breaks_ties(self):
        values = pd.DataFrame([
            {'run_id': 'baseline_s1', 'method': 'baseline', 'role': 'validation', 'balanced_brier': .15},
            {'run_id': 'wide_s2', 'method': 'wide', 'role': 'validation', 'balanced_brier': .08},
            {'run_id': 'wide_s1', 'method': 'wide', 'role': 'validation', 'balanced_brier': .08},
            {'run_id': 'deep_s1', 'method': 'deep', 'role': 'validation', 'balanced_brier': .13},
        ])
        expected = workflow.select_models(values)
        adversarial = values.assign(role='test', balanced_brier=-np.inf)
        training = values.assign(role='train', balanced_brier=np.nan)
        actual = workflow.select_models(pd.concat([training, values.iloc[::-1], adversarial], ignore_index=True))
        for key in ('selected_run_id', 'value', 'by_method', 'test_scoring_started_at_selection'):
            self.assertEqual(actual[key], expected[key])
        self.assertEqual(actual['selected_run_id'], 'wide_s1')
        self.assertEqual(actual['by_method']['wide'], 'wide_s1')
        with self.assertRaisesRegex(ValueError, 'Invalid validation selection'):
            workflow.select_models(adversarial)

    def _toy_scores(self):
        parts = []
        for distribution, counts in [('P', 3 + np.arange(20) % 4), ('Q', 7 + np.arange(20) % 3)]:
            for index, count in enumerate(counts):
                values = np.repeat(workflow.EDGES[index], count)
                values[-1] = (2. if index == 19 else np.nextafter(workflow.EDGES[index + 1], workflow.EDGES[index]))
                parts.append(pd.DataFrame({'distribution': distribution, 'role': 'test', 'rdr': values,
                                           'bin_index': np.searchsorted(workflow.EDGES[1:-1], values, side='right')}))
        return pd.concat(parts, ignore_index=True)

    def test_ci_counts_endpoints_normalization_and_candidate_attachments(self):
        scores = self._toy_scores()
        spec = {'run_id': 'toy', 'method': 'wide_p30', 'seed': 1}
        config = {'alpha': .05, 'min_count': 10, 'bootstrap_repetitions': 128, 'seed': 13}
        cells, candidates, attached, summary, calibrated = workflow.merged_intervals(scores, spec, config)
        self.assertEqual(len(candidates), 210)
        self.assertEqual(calibrated['metadata']['c2_tail_probability'], .05 / 840)
        self.assertEqual(len(attached), len(scores))
        self.assertEqual(calibrated['metadata']['n'], int(scores.distribution.eq('P').sum()))
        self.assertEqual(calibrated['metadata']['m'], int(scores.distribution.eq('Q').sum()))
        self.assertEqual(cells.k_p.sum(), calibrated['metadata']['n'])
        self.assertEqual(cells.k_q.sum(), calibrated['metadata']['m'])
        self.assertEqual(cells.left.iloc[0], 0.)
        self.assertEqual(cells.right.iloc[-1], 2.)
        assert_array_equal(cells.right.to_numpy()[:-1], cells.left.to_numpy()[1:])
        self.assertTrue((cells.k_p >= 10).all() and (cells.k_q >= 10).all())
        self.assertEqual(attached.loc[attached.rdr.eq(2), 'bin_index'].unique().tolist(), [len(cells) - 1])
        self.assertEqual(attached.loc[attached.rdr.eq(0), 'bin_index'].unique().tolist(), [0])
        for cell in cells.itertuples():
            rows = attached.loc[attached.bin_index.eq(cell.bin_index)]
            p, q = (rows.loc[rows.distribution.eq(d), 'rdr'].to_numpy() for d in ('P', 'Q'))
            n, m = calibrated['metadata']['n'], calibrated['metadata']['m']
            total = len(p) / n + len(q) / m
            self.assertAlmostEqual(cell.estimate, 2 * (len(p) / n) / total)
            self.assertAlmostEqual(cell.mean_model_score, (p.sum() / n + q.sum() / m) / total)
            self.assertAlmostEqual(cell.mixture_mass, total / 2)
            self.assertEqual(rows.candidate_index.unique().tolist(), [cell.candidate_index])
            for prefix in ('c1', 'c2'):
                for bound in ('lower', 'upper'):
                    name = prefix + '_' + bound
                    assert_allclose(rows[name], getattr(cell, name), atol=0, rtol=0)
            self.assertEqual(int(candidates.loc[cell.candidate_index, 'k_p']), len(p))
        self.assertAlmostEqual(cells.mixture_mass.sum(), 1.)
        whole = candidates[candidates.whole_space].iloc[0]
        self.assertEqual((whole.c1_lower, whole.c1_upper, whole.c2_lower, whole.c2_upper), (1., 1., 1., 1.))
        self.assertEqual(summary['n_regions'], len(cells))
        with self.assertRaisesRegex(ValueError, 'only frozen test'):
            workflow.merged_intervals(scores.assign(role='validation'), spec, config)

    def test_region_statistics_do_not_use_pooled_row_denominators(self):
        scores = self._toy_scores()
        spec = {'run_id': 'toy', 'method': 'baseline_p5', 'seed': 1}
        config = {'alpha': .05, 'min_count': 1, 'bootstrap_repetitions': 32, 'seed': 13}
        cells = workflow.merged_intervals(scores, spec, config)[0]
        repeated = pd.concat([scores, *[scores[scores.distribution.eq('Q')]] * 3], ignore_index=True)
        duplicate_cells = workflow.merged_intervals(repeated, spec, config)[0]
        columns = ['estimate', 'mean_model_score', 'calibration_gap', 'mixture_mass']
        assert_allclose(cells[columns], duplicate_cells[columns], rtol=0, atol=1e-15)
        self.assertGreater(np.max(np.abs(cells.estimate - 2 * cells.k_p / (cells.k_p + cells.k_q))), .1)

    def _prepared_fixture(self, out):
        (out / 'cache').mkdir()
        (out / 'runs').mkdir()
        basis = helmert(4, full=False)
        torch.save({'dimension': 4, 'basis': basis}, out / 'transform.pt')
        spec = {'run_id': 'wide_p30_s1', 'method': 'wide_p30', 'seed': 1, 'epsilon': 1e-6,
                'primary': True, 'patience': 30}
        protocol = {'candidate_specs': [spec], 'config': {'min_epochs': 1, 'max_epochs': 3, 'preflight': True}}
        workflow.save_json(out / 'protocol.json', protocol)
        manifest = []
        generator = np.random.default_rng(718)
        for distribution, role, count in [('P', 'train', 5), ('Q', 'train', 7),
                                          ('P', 'validation', 3), ('Q', 'validation', 4)]:
            np.save(out / 'cache' / f'{distribution}_{role}.npy', generator.normal(size=(count, 3)).astype(np.float32))
            manifest.extend({'distribution': distribution, 'role': role, 'sample_id': f'{distribution}_{role}_{i}'}
                            for i in range(count))
        for distribution in ('P', 'Q'):
            np.save(out / 'cache' / f'{distribution}_test.npy', np.full((3, 4), np.nan))
        pd.DataFrame(manifest).to_csv(out / 'split_manifest.csv', index=False)
        outputs = [out / 'protocol.json', out / 'transform.pt', out / 'split_manifest.csv',
                   *sorted((out / 'cache').glob('*.npy'))]
        workflow.save_json(out / 'prepared.json', {'outputs': {str(p.relative_to(out)): workflow.sha256(p) for p in outputs}})
        return out / 'runs' / spec['run_id']

    def test_fit_stage_never_loads_test_arrays_and_seals_fit_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            run = self._prepared_fixture(out)
            original_load = np.load
            loaded = []
            def audited_load(path, *args, **kwargs):
                loaded.append(Path(path).name)
                if Path(path).name.endswith('_test.npy'):
                    raise AssertionError('Fit stage loaded test values')
                return original_load(path, *args, **kwargs)
            with patch.object(workflow.np, 'load', side_effect=audited_load), contextlib.redirect_stdout(io.StringIO()):
                workflow.fit_one(out, 0)
            self.assertEqual(set(loaded), {'P_train.npy', 'Q_train.npy', 'P_validation.npy', 'Q_validation.npy'})
            result = workflow.verify_result(run)
            self.assertFalse(result['test_scored'])
            self.assertEqual(set(workflow.read_frame(run / 'fit_scores.csv').role), {'train', 'validation'})
            self.assertFalse((run / 'test_scores.csv').exists())
            with (run / 'model.pt').open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                workflow.fit_one(out, 0)

    def test_changed_prepared_input_is_rejected_before_model_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            run = self._prepared_fixture(out)
            with (out / 'cache' / 'Q_train.npy').open('ab') as handle:
                handle.write(b'changed')
            with patch.object(workflow, 'make_model', side_effect=AssertionError('Should not construct a model')), \
                 self.assertRaisesRegex(ValueError, 'Prepared artifact changed: cache/Q_train.npy'):
                workflow.fit_one(out, 0)
            self.assertFalse(run.exists())


if __name__ == '__main__':
    unittest.main()
