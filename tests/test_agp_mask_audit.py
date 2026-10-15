"""Independent numerical checks for the mask comparison verifier."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

PATH = Path(__file__).resolve().parents[1] / 'experiments/AGP/verify_mask_comparison.py'
SPEC = importlib.util.spec_from_file_location('agp_mask_audit', PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditNumerics(unittest.TestCase):
    def test_auc_ties_and_source_orientation(self):
        self.assertEqual(audit.rank_auc([1,1], [0,0]), 1.)
        self.assertEqual(audit.rank_auc([0,0], [1,1]), 0.)
        self.assertEqual(audit.rank_auc([.5,.5], [.5]), .5)
        self.assertEqual(audit.rank_auc([0,1], [1,2]), .125)

    def test_ecdf_ties_and_unequal_sizes(self):
        self.assertEqual(audit.ks_statistic([0,0,1], [0,1,1]), 1/3)
        self.assertEqual(audit.ks_statistic([0,0], [1,1,1]), 1.)
        self.assertEqual(audit.ks_statistic([0,1], [0,0,1,1]), 0.)

    def test_log_loss_is_source_balanced(self):
        expected = -.5 * (np.log(.8) + np.log(.7))
        self.assertAlmostEqual(audit.balanced_log_loss([.8]*5,[.3]*2), expected)
        with self.assertRaisesRegex(ValueError, 'outside'):
            audit.balanced_log_loss([1.1], [.5])

    def test_min_delta_checkpoint_and_patience(self):
        history = pd.DataFrame({'epoch':[0,1,2,3], 'valid_nll':[3.,2.,1.99995,2.1]})
        self.assertEqual(audit.checkpoint_choice(history,'epoch',1e-4,2), (1,2.,2))
        with self.assertRaisesRegex(ValueError,'past declared patience'):
            audit.checkpoint_choice(history,'epoch',1e-4,1)

    def test_independent_fidelity_and_constant_pairs(self):
        p = np.array([[1,0,0],[0,1,0],[.5,.5,0]],float)
        result = audit.fidelity(p,p,p,p)
        for key in ('richness_ks','shannon_ks','prevalence_mae','mean_abundance_mae',
                    'occurrence_correlation_mae','mask_novel_fraction'):
            self.assertEqual(result[key], 0.)
        self.assertEqual(result['occurrence_variable_taxa_training'], 2)
        self.assertEqual(result['occurrence_pairs_compared'], 1)
        q = np.array([[1,0,0],[.8,0,.2]])
        result = audit.fidelity(p,q,p,p)
        self.assertEqual(result['occurrence_pairs_compared'],0)
        self.assertTrue(np.isnan(result['occurrence_correlation_mae']))
        self.assertEqual(result['mask_novel_fraction'],.5)

    def test_seeded_sampling_rejects_empty_and_preserves_donors(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            full = np.array([[1,0],[0,1],[1,1]],bool)
            schema = audit.schema(full)
            np.savez(folder/'bootstrap.npz', masks=full)
            audit.write(folder/'bootstrap.json', {'schema':schema})
            cfg = {'batch_size':3,'max_draw_multiplier':100}
            masks,n,r = audit.replay_masks(folder,'bootstrap',10,117,cfg)
            self.assertEqual((n,r),(10,0))
            self.assertTrue(all(row.tobytes() in {x.tobytes() for x in full} for row in masks))
            np.savez(folder/'bernoulli_mixture.npz', weights=np.array([1.]), probabilities=np.array([[.2,.3]]))
            audit.write(folder/'bernoulli_mixture.json', {'schema':schema})
            masks,n,r = audit.replay_masks(folder,'bernoulli_mixture',30,111,cfg)
            self.assertGreater(r,0)
            self.assertEqual(n-r,30)
            self.assertTrue(masks.any(1).all())
            replay,nn,rr = audit.replay_masks(folder,'bernoulli_mixture',30,111,cfg)
            np.testing.assert_array_equal(masks,replay)
            self.assertEqual((n,r),(nn,rr))

    def test_seal_pins_artifacts_excludes_live_logs_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder/'logs').mkdir()
            (folder/'logs/job.out').write_text('running')
            (folder/'audit_123.log').write_text('audit still printing')
            (folder/'result.csv').write_text('a,b\n1,2\n')
            audit.write(folder/'verification.json', {'status':'passed'})
            audit.seal(folder, {'status':'passed'})
            receipt = audit.read(folder/'COMPLETE.json')
            self.assertEqual(set(receipt['outputs_sha256']), {'result.csv','verification.json'})
            audit.hashes(folder,receipt['outputs_sha256'])
            (folder/'result.csv').write_text('a,b\n1,3\n')
            with self.assertRaisesRegex(ValueError,'changed'):
                audit.hashes(folder,receipt['outputs_sha256'])

    def _legacy_live_log_receipt(self, folder):
        (folder/'result.csv').write_text('a,b\n1,2\n')
        (folder/'audit_123.log').write_text('before seal\n')
        audit.write(folder/'verification.json', {'status':'passed'})
        audit.seal(folder, {'status':'passed'})
        receipt = audit.read(folder/'COMPLETE.json')
        receipt['outputs_sha256']['audit_123.log'] = audit.sha(folder/'audit_123.log')
        audit.write(folder/'COMPLETE.json', receipt)
        (folder/'audit_123.log').write_text('before seal\nafter seal\n')

    def test_amend_live_log_preserves_receipts_and_scientific_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._legacy_live_log_receipt(folder)
            original = (folder/'COMPLETE.json').read_bytes()
            verification = (folder/'verification.json').read_bytes()
            result = audit.amend_live_log_seal(folder)
            self.assertEqual((folder/'COMPLETE.json').read_bytes(), original)
            self.assertEqual((folder/'verification.json').read_bytes(), verification)
            self.assertEqual(result['original_completion_sha256'],audit.sha(folder/'COMPLETE.json'))
            self.assertEqual(set(result['admissible_log_mismatches']),{'audit_123.log'})
            self.assertFalse(result['numerical_audit_rerun'])
            self.assertEqual(set(result['outputs_sha256']),{'COMPLETE.json','verification.json','result.csv'})
            audit.hashes(folder,result['outputs_sha256'])
            with self.assertRaisesRegex(ValueError,'already exists'):
                audit.amend_live_log_seal(folder)

    def test_amendment_rejects_nonlog_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._legacy_live_log_receipt(folder)
            original = (folder/'COMPLETE.json').read_bytes()
            (folder/'result.csv').write_text('a,b\n1,3\n')
            with self.assertRaisesRegex(ValueError,'Non-log artifact changed'):
                audit.amend_live_log_seal(folder)
            self.assertFalse((folder/'COMPLETE_v2.json').exists())
            self.assertEqual((folder/'COMPLETE.json').read_bytes(),original)

    def test_amendment_rejects_modified_numerical_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self._legacy_live_log_receipt(folder)
            audit.write(folder/'verification.json',{'status':'passed','changed':True})
            with self.assertRaisesRegex(ValueError,'verification is missing or changed'):
                audit.amend_live_log_seal(folder)
            self.assertFalse((folder/'COMPLETE_v2.json').exists())


if __name__ == '__main__':
    unittest.main()
