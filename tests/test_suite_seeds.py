"""Regression coverage for synthetic train/validation/test stream separation."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from write_results_summary import read_and_check


class SuiteSeedTests(unittest.TestCase):
    def check_plan(self, override):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            command = [sys.executable, str(ROOT / 'scripts/run_suite.py'), '--data-dir', str(root / 'data'),
                       '--output-dir', str(root / 'output'), '--gpus', '0', '--dry-run']
            if override is not None:
                config = root / 'override.json'
                config.write_text(json.dumps(override))
                command += ['--config', str(config)]
            plan = json.loads(subprocess.check_output(command, text=True, cwd=root))
            self.assertEqual(len(plan['jobs']), 26)
            seed = 42 if override is None else override['seed']
            for job in plan['jobs']:
                if job['stage'] == 'nmse':
                    values = job['command']
                    test_seed = int(values[values.index('--seed') + 1])
                    self.assertNotIn(test_seed, (seed, seed + 100000))
                    self.assertEqual(test_seed, seed + 200000)
                else:
                    self.assertNotIn('--seed', job['command'])
            self.assertFalse((root / 'output').exists())

    def test_default_csi_test_stream_is_separate(self):
        self.check_plan(None)

    def test_custom_training_seed_keeps_separate_test_stream(self):
        self.check_plan({'seed': 7})

    def rejected_report(self, protocol, message):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / 'qualitative').mkdir()
            (root / 'qualitative/manifest.json').write_text('{}')
            (root / 'latency.json').write_text('{}')
            report = {'protocol': 'full_mfnet_100_epochs',
                      'training_modules_match_current_ast': {'runner.py': True},
                      'training': {'dmce_stage2': {'best_checkpoint_sha256': 'expected'}},
                      'csi_nmse_protocol': protocol}
            (root / 'results.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, message):
                read_and_check(root)

    def test_full_report_rejects_reused_csi_random_stream(self):
        self.rejected_report({'independent_test_seed': False, 'samples_per_snr': 1000}, 'independently seeded')

    def test_report_rejects_different_csi_checkpoint(self):
        self.rejected_report({'independent_test_seed': True, 'samples_per_snr': 1000,
                              'checkpoint_sha256': 'wrong'}, 'CSI NMSE checkpoint')


if __name__ == '__main__':
    unittest.main()
