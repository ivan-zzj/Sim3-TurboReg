"""Protocol regressions: scale/translation convention and failed-pair denominator."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from generate_scale_variant_dataset import compute_sim3_from_gt
from evaluate_sim3 import errors


class ProtocolTests(unittest.TestCase):
    def test_scaling_source_and_target(self):
        T = np.eye(4)
        T[:3, 3] = [1, 2, 3]
        x = np.array([2., 3., 4., 1.])
        y = T @ x
        for a in [0.25, 0.5, 2, 4]:
            xs = x.copy()
            xs[:3] *= a
            np.testing.assert_allclose(compute_sim3_from_gt(T, a, 'source') @ xs, y)
            ys = y.copy()
            ys[:3] *= a
            np.testing.assert_allclose(compute_sim3_from_gt(T, a, 'target') @ x, ys)

    def test_joint_metrics_and_units(self):
        gt = np.eye(4)
        pred = np.eye(4)
        pred[:3, :3] *= 2
        pred[:3, 3] = [0.2, 0, 0]
        e = errors(pred, gt, 2)
        self.assertAlmostEqual(e['log_scale_error'], np.log(2))
        self.assertAlmostEqual(e['re_deg'], 0)
        self.assertAlmostEqual(e['te'], 0.1)
        self.assertGreater(e['log_scale_error'], 0.1)  # Cannot pass on RE/TE alone.
        pred[0, 1] = 0.5
        with self.assertRaises(ValueError):
            errors(pred, gt)

    def test_missing_and_invalid_predictions_count_as_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            np.savetxt(d / 'gt.txt', np.eye(4))
            records = [dict(pair_id=str(i), gt='gt.txt', corr='unused.txt') for i in range(3)]
            (d / 'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            predictions = [dict(pair_id='0', transformation=np.eye(4).tolist(), runtime_ms=1),
                           dict(pair_id='2', transformation=np.eye(4).tolist(), runtime_ms=-1)]
            (d / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
            subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_sim3.py'),
                            '--manifest', str(d / 'manifest.jsonl'), '--predictions', str(d / 'predictions.jsonl'),
                            '--method', 'external', '--output', str(d / 'result')], check=True, capture_output=True)
            summary = json.loads((d / 'result/summary.json').read_text())
            self.assertEqual(summary['total'], 3)
            self.assertEqual(summary['successes'], 1)
            self.assertEqual(summary['errors'], 2)
            self.assertAlmostEqual(summary['rr'], 1/3)

    def test_generator_is_stable_across_descriptor_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            for desc, suffix in [('fpfh', ''), ('fcgf', '_fcgf')]:
                scene = d / 'input' / '3DMatch' / ('all_' + desc) / 'scene'
                scene.mkdir(parents=True)
                np.savetxt(scene / f'pair@corr{suffix}.txt', np.tile(np.arange(6), (3,1)))
                np.savetxt(scene / f'pair@GTmat{suffix}.txt', np.eye(4))
                subprocess.run([sys.executable, str(ROOT / 'scripts/generate_scale_variant_dataset.py'),
                                '--input_dir', str(d / 'input'), '--output_dir', str(d / 'output'),
                                '--desc', desc], check=True, capture_output=True)
            a = json.loads((d / 'output/3DMatch/all_fpfh/manifest.jsonl').read_text())
            b = json.loads((d / 'output/3DMatch/all_fcgf/manifest.jsonl').read_text())
            self.assertEqual(a['applied_scale'], b['applied_scale'])


if __name__ == '__main__':
    unittest.main()
