"""Risks specific to paper unknown-scale experiments, independent of old RR tests."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_benchmark', ROOT / 'scripts/teaser_sim3_benchmark.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
b.np = np
CONFIG = json.loads((ROOT / 'configs/teaser_sim3_protocol.json').read_text())


class PaperProtocolTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.random.default_rng(17).normal(size=(1000, 3))

    def test_exact_metric_direction_and_units(self):
        gt = np.eye(4)
        gt[:3, :3] *= 2
        pred = np.eye(4)
        pred[:3, :3] = 3 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        pred[:3, 3] = [3, 4, 0]
        e = b.errors(pred, gt)
        self.assertAlmostEqual(e['scale_error'], 1)
        self.assertAlmostEqual(e['rotation_error_deg'], 90)
        self.assertAlmostEqual(e['translation_error'], 5)
        for bad in [np.zeros((4, 4)), np.diag([-1, 1, 1, 1]), np.diag([1, 2, 3, 1]), np.full((4, 4), np.nan)]:
            with self.assertRaises(ValueError):
                b.errors(bad, gt)

    def test_generated_constraints_and_shared_precision(self):
        for pct in CONFIG['outlier_percent']:
            t = b.make_trial(self.vertices, CONFIG, 0, pct, 2)
            same = b.make_trial(self.vertices, CONFIG, 0, pct, 2)
            for key in t:
                np.testing.assert_array_equal(t[key], same[key])
            s, R, translation = b.decompose(t['gt'])
            self.assertTrue(1 <= s <= 5)
            self.assertLessEqual(np.linalg.norm(translation), 1)
            self.assertEqual(t['src'].shape, (100, 3))
            self.assertTrue(np.all(t['src'] >= 0) and np.all(t['src'] <= 1))
            self.assertEqual(len(set(t['sample_indices'])), 100)
            self.assertEqual(len(set(t['outlier_indices'])), pct)
            self.assertLessEqual(np.linalg.norm(t['noise'], axis=1).max(), CONFIG['beta'])
            self.assertLessEqual(np.linalg.norm(t['dst'][t['outlier_indices']], axis=1).max(), 5 + 1e-6)
            inliers = np.ones(100, dtype=bool)
            inliers[t['outlier_indices']] = False
            residual = t['dst'] - (s * t['src'].astype(float) @ R.T + translation)
            np.testing.assert_allclose(residual[inliers], t['noise'][inliers], atol=5e-7)
            np.testing.assert_array_equal(t['src'].astype(np.float64), t['src'])
            self.assertEqual(b.input_hash(t['src'], t['dst']), b.input_hash(t['src'].astype(float), t['dst'].astype(float)))
        other = b.make_trial(self.vertices, CONFIG, 0, 60, 2, 'validation')
        self.assertFalse(np.array_equal(other['src'], b.make_trial(self.vertices, CONFIG, 0, 60, 2)['src']))

    def test_rejection_noise_and_uniform_volume(self):
        rng = np.random.default_rng(4)
        # Small beta forces the rejection branch, unlike the almost untruncated paper setting.
        noise = b.bounded_noise(rng, 2000, 1, .8)
        self.assertLessEqual(np.linalg.norm(noise, axis=1).max(), .8)
        self.assertGreater(len(np.unique(np.linalg.norm(noise, axis=1))), 1900)
        samples = b.ball(rng, 30000, 5)
        # For uniform volume in 3D E[r^2] = 3 radius^2 / 5.
        self.assertAlmostEqual(float(np.mean(np.sum(samples**2, axis=1))), 15, delta=.2)
        np.testing.assert_allclose(samples.mean(axis=0), 0, atol=.06)

    def test_resume_tail_duplicate_and_saved_input_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'results.jsonl'
            row = dict(trial_id='test', method='Ours', status='failed', transformation=None)
            path.write_text(json.dumps(row) + '\n' + '{"partial"')
            with self.assertRaises(ValueError):
                b.read_records(path)
            self.assertEqual(b.read_records(path, repair=True), [row])
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            path.write_text(path.read_text() * 2)
            with self.assertRaises(ValueError):
                b.read_records(path)
            trial = b.make_trial(self.vertices, CONFIG, 0, 60, 0)
            saved = Path(tmp) / 'trial.npz'
            b.saved_trial(saved, trial)
            trial['src'][0, 0] += 1
            with self.assertRaises(ValueError):
                b.saved_trial(saved, trial)

    def test_summary_keeps_failures_and_marks_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            config = dict(CONFIG, objects=['bunny'], repeats=2, outlier_percent=[60])
            b.atomic_json(out / 'run.json', dict(config=config, contract_sha256='test'))
            rows = []
            for rep in range(2):
                for method in b.METHODS:
                    failed = method == 'Ours' and rep == 1
                    rows.append(dict(trial_id=f'bunny-o60-r{rep:03d}', object='bunny',
                        outlier_percent=60, method=method, status='failed' if failed else 'ok',
                        input_sha256=str(rep), contract_sha256='test', runtime_s=.01,
                        **{k: None if failed else .1 for k in b.METRICS[:3]}))
            (out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            b.summarize(out)
            result = json.loads((out / 'summary.json').read_text())
            self.assertTrue(result['complete'])
            self.assertEqual(result['groups'][-1]['method'], 'Ours')
            self.assertEqual(result['groups'][-1]['failures'], 1)
            self.assertEqual(result['groups'][-1]['scale_error_mean_all'], '+inf')
            self.assertAlmostEqual(result['groups'][-1]['scale_error_median_finite'], .1)
            (out / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows[:-1]))
            b.summarize(out)
            result = json.loads((out / 'summary.json').read_text())
            self.assertFalse(result['complete'])
            self.assertEqual(result['groups'][-1]['missing'], 1)
            self.assertEqual(result['groups'][-1]['scale_error_mean_all'], 'incomplete')


class BackendContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        b.configure_runtime(CONFIG)

    def test_both_solvers_recover_unknown_scale_without_gt(self):
        src = np.random.default_rng(891).normal(size=(100, 3)).astype(np.float32)
        options = json.loads((ROOT / 'configs/sim3.json').read_text())
        for scale in [.5, 2, 4.5]:
            gt = np.eye(4)
            gt[:3, :3] = scale * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
            gt[:3, 3] = [.1, -.3, .2]
            trial = dict(src=src, dst=(src @ gt[:3, :3].T + gt[:3, 3]).astype(np.float32), gt=gt)
            results = [b.solve(method, trial, CONFIG, options) for method in b.METHODS]
            self.assertEqual(results[0]['input_sha256'], results[1]['input_sha256'])
            for row in results:
                self.assertEqual(row['status'], 'ok', row)
                self.assertLess(row['scale_error'], 1e-4)
                self.assertLess(row['rotation_error_deg'], .05)
                self.assertLess(row['translation_error'], 1e-4)

    def test_degenerate_backend_failure_is_not_identity_success(self):
        options = json.loads((ROOT / 'configs/sim3.json').read_text())
        trial = dict(src=np.zeros((100, 3), np.float32), dst=np.zeros((100, 3), np.float32), gt=np.eye(4))
        row = b.solve('Ours', trial, CONFIG, options)
        self.assertEqual(row['status'], 'failed')
        self.assertIsNone(row['transformation'])
        self.assertIsNone(row['scale_error'])
        self.assertGreaterEqual(row['runtime_s'], 0)


if __name__ == '__main__':
    unittest.main()
