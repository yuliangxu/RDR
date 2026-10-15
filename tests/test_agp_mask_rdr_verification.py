"""Mutation tests for the new mask-RDR archive verification boundary."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('mask_rdr_verifier', ROOT/'experiments/AGP/verify_mask_rdr.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class MaskRdrVerification(unittest.TestCase):
    def test_hash_gate_rejects_mutation_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'source.py').write_text('original')
            original = audit.sha(root/'source.py')
            audit.hashes(root, {'source.py': original})
            (root/'source.py').write_text('modified')
            with self.assertRaisesRegex(AssertionError, 'changed'):
                audit.hashes(root, {'source.py': original})
            with self.assertRaisesRegex(AssertionError, 'Invalid receipt'):
                audit.hashes(root, {'../source.py': original})
            with self.assertRaisesRegex(AssertionError, 'Invalid receipt'):
                audit.hashes(root, {str(root/'source.py'): original})

    def test_null_distribution_counts_as_separate_subject_role(self):
        frame = pd.DataFrame(dict(original_distribution=['P','P'], subject_id=['a','a'],
                                  role=['train','train'], distribution=['P','Q']))
        with self.assertRaisesRegex(AssertionError, 'crosses'):
            audit.real_subject_audit(frame)
        frame['subject_id'] = ['a','b']
        audit.real_subject_audit(frame)
        frame.loc[1, 'subject_id'] = ''
        with self.assertRaisesRegex(AssertionError, 'Missing'):
            audit.real_subject_audit(frame)

    def test_subject_cannot_cross_training_and_assessment(self):
        frame = pd.DataFrame(dict(original_distribution=['P','P'], subject_id=['a','a'],
                                  role=['train','test'], distribution=['P','P']))
        with self.assertRaisesRegex(AssertionError, 'crosses'):
            audit.real_subject_audit(frame)

    def test_grid_uses_correct_loss_slope_and_repeat(self):
        config = dict(losses=['hellinger','kl','chisq','js'], output_alphas=[.5,1.,2.,4.], repeats=5)
        tasks = audit.selection_tasks(config)
        self.assertEqual(len(tasks), 80)
        self.assertEqual(len({(x['candidate'], x['repeat']) for x in tasks}), 80)
        self.assertIn(dict(candidate='js_a0.5', loss='js', output_alpha=.5, repeat=4), tasks)
        self.assertIn(dict(candidate='chisq_a4', loss='chisq', output_alpha=4., repeat=0), tasks)

    def test_masked_decode_is_shift_invariant_and_preserves_structural_zeros(self):
        masks = np.array([[True,False,True],[False,True,False]],bool)
        latent = np.array([[0.,0.,np.log(3.)],[0.,12.,0.]],np.float32)
        decoded = audit.independent_decode(latent,masks)
        np.testing.assert_allclose(decoded, [[.25,0.,.75],[0.,1.,0.]],atol=2e-8)
        shifted = latent.copy()
        shifted[masks] += 5
        np.testing.assert_allclose(audit.independent_decode(shifted,masks),decoded,atol=4e-8)
        self.assertTrue(np.array_equal(decoded>0,masks))
        latent[0,1] = 1
        with self.assertRaisesRegex(AssertionError,'leave mask support'):
            audit.independent_decode(latent,masks)

    def test_independent_mask_replay_tracks_empty_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'bernoulli_mixture.json').write_text(json.dumps(dict(schema=dict(
                dimension=3,active=[0,2],always_one=[]))))
            np.savez(directory/'bernoulli_mixture.npz',weights=np.array([1.]),probabilities=np.array([[.1,.2]]))
            masks,n,rejected = audit.replay_masks(directory,'bernoulli_mixture',30,117,
                                                  dict(batch_size=7,max_draw_multiplier=100))
            self.assertGreater(rejected,0)
            self.assertEqual(n-rejected,30)
            self.assertTrue(masks.any(1).all())
            self.assertFalse(masks[:,1].any())
            repeated,nn,rr = audit.replay_masks(directory,'bernoulli_mixture',30,117,
                                               dict(batch_size=7,max_draw_multiplier=100))
            np.testing.assert_array_equal(masks,repeated)
            self.assertEqual((n,rejected),(nn,rr))

    def test_only_successful_fresh_receipt_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary)/'verification.json'
            with self.assertRaisesRegex(AssertionError, 'successful'):
                audit.write_receipt(receipt, {'status':'failed'})
            self.assertFalse(receipt.exists())
            audit.write_receipt(receipt, {'status':'passed'})
            with self.assertRaisesRegex(AssertionError, 'fresh'):
                audit.write_receipt(receipt, {'status':'passed'})
            self.assertEqual(audit.read(receipt), {'status':'passed'})

    def _fixture(self, root):
        """Tiny final cohort: production import/identity rules, reduced dimensions."""
        analysis, parent, comparison = root/'analysis', root/'parent', root/'comparison'
        source = comparison/'generated/bootstrap_icfm/test.npy'
        source.parent.mkdir(parents=True)
        p = np.array([[.25,.75],[1.,0.]], np.float32)
        q64 = np.array([[.1,.9],[0.,1.],[.6,.4]], np.float64)
        np.save(source, q64)
        records, arrays, counts = [], {}, {}
        for folder in (analysis/'data/final', parent/'data/final'):
            folder.mkdir(parents=True)
        for side, values in [('p',p), ('q',q64.astype(np.float32))]:
            array_path = Path('data/final')/f'test_{side}.npy'
            np.save(analysis/array_path,values)
            np.save(parent/array_path,values)
            arrays['test_'+side] = dict(path=str(array_path),sha256=audit.sha(analysis/array_path))
            counts['test_'+side] = dict(rows=len(values),subjects=len(values) if side=='p' else None)
            for row, value in enumerate(values):
                records.append(dict(distribution=side.upper(), role='test', array_row=row,
                    sample_id=f'{side}:{row}',subject_id=f's:{row}' if side=='p' else '',
                    source_file='original.csv' if side=='p' else str(source),source_row=row,
                    row_sha256=hashlib.sha256(value.astype('<f4').tobytes()).hexdigest(),
                    original_role='test',original_distribution=side.upper()))
        frame = pd.DataFrame(records)
        frame.to_csv(analysis/'data/final/manifest.csv',index=False)
        frame.to_csv(parent/'data/final/manifest.csv',index=False)
        entry = dict(manifest=dict(path='data/final/manifest.csv'),arrays=arrays,counts=counts,
                     role_counts=dict(test=dict(p=2,q=3)))
        data = dict(studies=dict(final=entry))
        return analysis, data, 'bootstrap_icfm', parent, data, comparison, root/'generated'

    def _check(self, fixture):
        return audit.data_audit(*fixture,role_sizes={'final':dict(test_p=2,test_q=3)},feature_dimension=2)

    def test_exact_source_cast_and_real_identity_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            raw, manifests, counts, receipts = self._check(fixture)
            self.assertEqual(counts['final']['test_p'], dict(rows=2, subjects=2))
            self.assertEqual(counts['final']['test_q'], dict(rows=3, subjects=0))
            self.assertEqual(len(receipts), 1)

    def test_changed_observed_bytes_rejected_even_with_new_row_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            analysis = fixture[0]
            path = analysis/'data/final/test_p.npy'
            values = np.load(path)
            values[0] = [0.,1.]
            np.save(path,values)
            manifest = analysis/'data/final/manifest.csv'
            frame = audit.rows(manifest)
            frame.loc[0,'row_sha256'] = hashlib.sha256(values[0].astype('<f4').tobytes()).hexdigest()
            frame.to_csv(manifest,index=False)
            with self.assertRaisesRegex(AssertionError,'Original P/null cache'):
                self._check(fixture)

    def test_generated_source_must_be_original_comparison_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            manifest = fixture[0]/'data/final/manifest.csv'
            frame = audit.rows(manifest)
            frame.loc[frame.distribution.eq('Q'),'source_file'] = str(Path(temporary)/'unrelated.npy')
            frame.to_csv(manifest,index=False)
            with self.assertRaisesRegex(AssertionError,'original mask-comparison'):
                self._check(fixture)

    def test_generated_bytes_cannot_change_under_unchanged_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            path = fixture[5]/'generated/bootstrap_icfm/test.npy'
            values = np.load(path)
            values[0] = [.2,.8]
            np.save(path,values)
            with self.assertRaisesRegex(AssertionError,'Generated import/support'):
                self._check(fixture)

    def test_real_sample_id_change_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            manifest = fixture[0]/'data/final/manifest.csv'
            frame = audit.rows(manifest)
            frame.loc[0,'sample_id'] = 'relabelled'
            frame.to_csv(manifest,index=False)
            with self.assertRaisesRegex(AssertionError,'Observed identity changed'):
                self._check(fixture)


if __name__ == '__main__':
    unittest.main()
