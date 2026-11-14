"""Public-artifact checks for preserving the AGP summary during JS replacement."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/AGP'))
import seven_global_metrics_pq as report


class SummaryReplacementTests(unittest.TestCase):
    def setUp(self):
        self.baseline = report.read_csv(report.BASELINE / 'global_metrics.csv')
        # Include negative fitted estimates; all five must affect both summaries.
        self.values = [-0.2, -0.1, 0.0, 0.1, 0.2]
        self.fits = [dict(method=method, repeat=i, n_p=2002, n_q=2089,
                          js_p_q_variational=value)
                     for method in report.LABELS for i, value in enumerate(self.values)]

    def test_preserve_all_non_js_cells_and_use_sample_sd(self):
        rows, comparison = report.assemble(self.baseline, self.fits)
        self.assertEqual(len(rows), 4)
        for original, new, delta in zip(self.baseline, rows, comparison):
            self.assertEqual(set(new), (set(original) - set(report.OLD_JS)) | set(report.NEW_JS))
            for key, cell in original.items():
                if key not in report.OLD_JS:
                    self.assertEqual(new[key], cell, (original['method'], key))
            self.assertEqual(new['js_p_q_variational_mean'], 0)
            # Sum of squared deviations is .10, divided by n-1=4.
            self.assertAlmostEqual(new['js_p_q_variational_sd'], math.sqrt(.1/4), places=15)
            self.assertEqual(delta['old_js_p_m_variational_mean'], original[report.OLD_JS[0]])

    def test_missing_fit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '20 fit estimates'):
            report.assemble(self.baseline, self.fits[:-1])

    def test_duplicate_repeat_is_rejected(self):
        self.fits[4]['repeat'] = 3
        with self.assertRaisesRegex(ValueError, 'final-fit set'):
            report.assemble(self.baseline, self.fits)

    def test_equalized_cohorts_are_rejected(self):
        self.fits[0]['n_q'] = 2002
        with self.assertRaisesRegex(ValueError, 'full test counts'):
            report.assemble(self.baseline, self.fits)

    def test_saved_input_snapshot_verifies_without_historical_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            snapshot = output/'source_snapshot/retained_manifest.json'
            snapshot.parent.mkdir()
            snapshot.write_text('{"frozen_input": true}\n')
            logical = 'results/AGP/nonexistent_historical_package/manifest.json'
            manifest = dict(repository_inputs_sha256={logical: report.sha(snapshot)},
                            repository_input_snapshots={logical: 'source_snapshot/retained_manifest.json'})
            report.verify_repository_inputs(output, manifest)
            snapshot.write_text('{"frozen_input": false}\n')
            with self.assertRaisesRegex(ValueError, 'input hash mismatch'):
                report.verify_repository_inputs(output, manifest)


if __name__ == '__main__':
    unittest.main()
