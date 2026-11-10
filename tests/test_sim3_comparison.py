import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'build/sim3_completion'))
import sim3_comparison as solver

class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(936725)
        self.x=self.rng.uniform(size=(100,3))
        self.R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        self.t=np.array([.3,-.5,.8])

    def test_unknown_scale_rotation_translation(self):
        for method in ['ransac_1k','ransac_1min','fgr_sim3']:
            for scale in [.25,1.,4.]:
                y=scale*self.x@self.R.T+self.t
                before=y.copy()
                result=solver.solve(self.x,y,method,.0554,12)
                np.testing.assert_allclose(result['transformation'][:3,:3],scale*self.R,atol=1e-8)
                np.testing.assert_allclose(result['transformation'][:3,3],self.t,atol=1e-8)
                np.testing.assert_array_equal(y,before)

    def test_ransac_noisy_outliers_and_determinism(self):
        y=2*self.x@self.R.T+self.t+self.rng.normal(0,.002,self.x.shape)
        y[30:]=self.rng.uniform(-5,5,(70,3))
        for method in ['ransac_1k','ransac_1min']:
            result=solver.solve(self.x,y,method,.0554,42)
            second=solver.solve(self.x,y,method,.0554,42)
            np.testing.assert_allclose(result['transformation'][:3,:3],2*self.R,atol=.02)
            np.testing.assert_array_equal(result['transformation'],second['transformation'])
            self.assertEqual(result['num_inliers'],30)
            if method=='ransac_1k': self.assertLessEqual(result['iterations'],1000)

    def test_timeout_and_degenerate_contract(self):
        y=self.rng.uniform(-5,5,self.x.shape)
        with self.assertRaises(RuntimeError): solver.solve(self.x,y,'ransac_1min',.0554,12,1e-12)
        for method in ['ransac_1k','fgr_sim3']:
            with self.assertRaises(RuntimeError): solver.solve(np.zeros((20,3)),np.zeros((20,3)),method,.0554)

    def test_fgr_not_given_inliers(self):
        y=2*self.x@self.R.T+self.t+self.rng.normal(0,.002,self.x.shape)
        y[70:]=self.rng.uniform(-5,5,(30,3))
        result=solver.solve(self.x,y,'fgr_sim3',.0554)
        np.testing.assert_allclose(result['transformation'][:3,:3],2*self.R,atol=.02)
        self.assertEqual(result['iterations'],64)

if __name__=='__main__': unittest.main()
