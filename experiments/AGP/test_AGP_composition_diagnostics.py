"""Scientific invariants for matched AGP compositional-input experiments."""

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

# Experiment entrypoints also support direct invocation from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import AGP_composition_diagnostics as workflow


class CompositionDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.arrays = {
            ('P', 'train'): np.array([[.2, .3, .1, .4], [.1, .2, .4, .3]]),
            ('Q', 'train'): np.array([[.5, .2, .2, .1], [.3, .2, .1, .4], [.1, .1, .2, .6]]),
            ('P', 'validation'): np.array([[0., .5, 0., .5]]),
            ('Q', 'validation'): np.array([[0., 0., 1., 0.]]),
            ('P', 'test'): np.array([[1., 0., 0., 0.]]),
            ('Q', 'test'): np.array([[0., 1., 0., 0.]]),
        }
        self.epsilon = 1e-6

    def test_train_only_equal_distribution_preprocessing(self):
        state = workflow.fit_transform(self.arrays, 'logcontrast', self.epsilon)
        p = workflow.clr(self.arrays['P', 'train'], self.epsilon)
        q = workflow.clr(self.arrays['Q', 'train'], self.epsilon)
        expected_center = (p.mean(0) + q.mean(0)) / 2
        assert_allclose(state['center'], expected_center, atol=1e-15)
        self.assertGreater(np.linalg.norm(expected_center - np.vstack([p, q]).mean(0)), 1e-3)
        changed = dict(self.arrays)
        for key in changed:
            if key[1] != 'train':
                changed[key] = np.full_like(changed[key], np.nan)
        changed['Q', 'train'] = np.repeat(changed['Q', 'train'], 4, axis=0)
        repeated = workflow.fit_transform(changed, 'logcontrast', self.epsilon)
        assert_allclose(repeated['center'], state['center'], atol=1e-15)
        self.assertAlmostEqual(repeated['scale'], state['scale'], places=14)

    def test_scale_invariance_with_zeros_and_ilr_reconstruction(self):
        basis = helmert(4, full=False)
        state = workflow.fit_transform(self.arrays, 'ilr', self.epsilon, basis)
        x = np.array([[0., 3., 1., 0.], [.1, .3, .4, .2]])
        z = workflow.transform(x, state)
        assert_allclose(workflow.transform(x * np.array([[13.7], [.004]]), state), z, atol=2e-7)
        # Recover the smoothed composition, not the unsmoothed zero-containing row.
        clr_recovered = (z.astype(float) @ basis) * state['scale'] + state['center']
        positive = np.exp(clr_recovered - clr_recovered.max(1, keepdims=True))
        recovered = positive / positive.sum(1, keepdims=True)
        target = (workflow.close(x) + self.epsilon) / (1 + 4 * self.epsilon)
        assert_allclose(recovered, target, rtol=1e-6, atol=1e-8)

    def test_matching_initial_functions_across_all_logratio_coordinates(self):
        phylo = np.array([[.5, .5, -.5, -.5],
                          [1/np.sqrt(2), -1/np.sqrt(2), 0., 0.],
                          [0., 0., 1/np.sqrt(2), -1/np.sqrt(2)]])
        bases = {'ilr': helmert(4, full=False), 'philr': phylo, 'logcontrast': None}
        x = np.vstack(list(self.arrays.values()))
        first_layers, outputs = [], []
        for method, basis in bases.items():
            state = workflow.fit_transform(self.arrays, method, self.epsilon, basis)
            inputs = torch.as_tensor(workflow.transform(x, state))
            model = workflow.make_model(4, method, seed=317, basis=basis)
            with torch.no_grad():
                first_layers.append(model.model[0](inputs).numpy())
                outputs.append(model(inputs).numpy())
        for first, result in zip(first_layers[1:], outputs[1:]):
            assert_allclose(first, first_layers[0], rtol=1e-5, atol=1e-6)
            assert_allclose(result, outputs[0], rtol=1e-6, atol=1e-7)

    def test_contrast_constraint_survives_optimization(self):
        layer = workflow.ContrastLinear(4, 3)
        optimizer = torch.optim.AdamW(layer.parameters(), lr=.01, weight_decay=.01)
        x = torch.tensor([[1., 2., 3., 4.], [2., 0., -1., 3.]])
        for _ in range(3):
            optimizer.zero_grad()
            layer(x).square().mean().backward()
            optimizer.step()
        assert_allclose(layer.effective_weight().detach().sum(1).numpy(), 0, atol=1e-7)
        assert_allclose(layer(x).detach().numpy(), layer(x + 4.7).detach().numpy(), atol=1e-6)

    def test_unequal_sample_diagnostics_and_replication_invariance(self):
        p = np.array([.12, .18, 1.85])
        q = np.array([.16, 1.82, 1.84, 1.86, 1.88, 1.89])
        spec = {'run_id': 'toy', 'method': 'raw', 'seed': 1, 'epsilon': 0., 'primary': True}
        training = {'best_epoch': 1, 'epochs_run': 1}
        def summarize(q_values):
            scores = pd.DataFrame({'distribution': ['P'] * len(p) + ['Q'] * len(q_values),
                                   'role': 'validation', 'rdr': np.r_[p, q_values]})
            return workflow.summarize(scores, spec, training)
        cells, metrics = summarize(q)
        low = cells.loc[cells.bin_index.eq(1)].iloc[0]
        self.assertEqual((low.k_p, low.k_q), (2, 1))
        self.assertAlmostEqual(low.cell_rdr, 1.6)
        self.assertAlmostEqual(low.mean_model_score, .152)
        self.assertAlmostEqual(low.mixture_mass, 5/12)
        self.assertNotAlmostEqual(low.mean_model_score, (.12 + .18 + .16)/3)
        expected_brier = np.sum((1-p/2)**2)/(2*len(p)) + np.sum((q/2)**2)/(2*len(q))
        self.assertAlmostEqual(metrics.balanced_brier.iloc[0], expected_brier)
        self.assertAlmostEqual(cells.mixture_mass.sum(), 1.)
        duplicate_cells, duplicate_metrics = summarize(np.repeat(q, 5))
        columns = ['mixture_mass', 'mean_model_score', 'cell_rdr', 'calibration_gap']
        assert_allclose(cells[columns], duplicate_cells[columns], equal_nan=True, atol=1e-15)
        for name in ['balanced_brier', 'empirical_bin_absolute_gap', 'empirical_bin_rmse', 'auc']:
            self.assertAlmostEqual(metrics[name].iloc[0], duplicate_metrics[name].iloc[0])

    def test_fit_ignores_test_arrays(self):
        class ForbiddenTestArray:
            def __array__(self, *args, **kwargs):
                raise AssertionError('Training touched test values')
        arrays = {key: value.astype(np.float32) for key, value in self.arrays.items()}
        for distribution in ('P', 'Q'):
            arrays[distribution, 'test'] = ForbiddenTestArray()
        args = SimpleNamespace(max_epochs=3, min_epochs=1, patience=3)
        model = workflow.make_model(4, 'raw', seed=128)
        with contextlib.redirect_stdout(io.StringIO()):
            history, training = workflow.fit_model(model, arrays, args)
        self.assertAlmostEqual(training['best_validation_loss'], history.validation_loss.min(), places=12)

    def test_changed_source_data_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            manifest = pd.DataFrame({'distribution': ['P'], 'role': ['train'], 'sample_id': ['s1']})
            manifest.to_csv(source / 'split_manifest.csv', index=False)
            prior = {'seed': 1, 'p_policy': 'keep-all', 'data_audit': {'input_sha256': {'data': 'old'}}}
            (source / 'protocol.json').write_text(json.dumps(prior))
            output = root / 'output'
            argv = ['AGP_composition_diagnostics.py', '--preflight', '--source-fit', str(source),
                    '--output-dir', str(output), '--max-epochs', '1', '--min-epochs', '1']
            with patch.object(sys, 'argv', argv), \
                 patch.object(workflow, 'verify_result', return_value={'status': 'complete'}), \
                 patch.object(workflow, 'prepare_712_data', return_value=(self.arrays, manifest, {'input_sha256': {'data': 'new'}})), \
                 self.assertRaisesRegex(ValueError, 'Source data changed'):
                workflow.main()
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
