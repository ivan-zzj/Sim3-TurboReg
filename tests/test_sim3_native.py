"""Real API correctness and failure contracts of the small-cloud CPU backend."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'bindings'))
import turboreg_native as native

class NativeTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(421)
        self.src=self.rng.uniform(size=(100,3))
        self.R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
        self.t=np.array([.1,-.3,.2])
    def test_exact_similarity_and_input_preservation(self):
        for scale in [.2,1.,4.,8.]:
            a=self.src.copy(); b=scale*a@self.R.T+self.t
            before=b.copy()
            r=native.register_sim3(a,b,{})
            T=r['transformation']
            np.testing.assert_allclose(T[:3,:3],scale*self.R,atol=1e-9)
            np.testing.assert_allclose(T[:3,3],self.t,atol=1e-9)
            self.assertEqual(r['num_inliers'],100)
            np.testing.assert_array_equal(a,self.src);np.testing.assert_array_equal(b,before)
    def test_noisy_outliers_and_uniform_refit(self):
        b=2*self.src@self.R.T+self.t+self.rng.normal(0,.002,(100,3))
        b[40:]=self.rng.uniform(-4,4,(60,3))
        for uniform in [False,True]:
            r=native.register_sim3(self.src,b,{'uniform_refinement':uniform,'refinement_iterations':3})
            np.testing.assert_allclose(r['transformation'][:3,:3],2*self.R,atol=.01)
            self.assertEqual(r['num_inliers'],40)
    def test_invalid_inputs_and_options(self):
        for a,b,o in [(self.src[:2],self.src[:2],{}),(self.src,self.src[:3],{}),
                      (np.full((10,3),np.nan),np.zeros((10,3)),{}),
                      (self.src,self.src,{'k_neighbors':0}),
                      (self.src,self.src,{'metric':'bad'}),
                      (self.src,self.src,{'unknown_option':1})]:
            with self.assertRaises((ValueError,RuntimeError)): native.register_sim3(a,b,o)
    def test_degenerate_inputs_fail(self):
        for a in [np.zeros((10,3)),np.arange(30.).reshape(10,3)]:
            with self.assertRaises((ValueError,RuntimeError)):
                native.register_sim3(a,2*a,{'reject_degenerate':True})
    def test_three_points_and_float32(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
        b=2*a@self.R.T+self.t
        r=native.register_sim3(a,b,{'reject_degenerate':True})
        np.testing.assert_allclose(r['transformation'][:3,:3],2*self.R,atol=1e-9)
    def test_dense_fallback_and_overflow(self):
        for n in [128,129]:
            a=self.rng.uniform(size=(n,3));b=2*a@self.R.T+self.t
            r=native.register_sim3(a,b,{'max_points':n,'use_second_order':True})
            np.testing.assert_allclose(r['transformation'][:3,:3],2*self.R,atol=1e-9)
        huge=self.src*1e308
        with self.assertRaises((ValueError,RuntimeError)):
            native.register_sim3(huge,huge,{})
    def test_limits_and_determinism(self):
        b=2*self.src@self.R.T+self.t
        a=native.register_sim3(self.src,b,{'max_points':50,'use_second_order':True,'graph_iterations':2})
        c=native.register_sim3(self.src,b,{'max_points':50,'use_second_order':True,'graph_iterations':2})
        self.assertEqual(a['num_used'],50)
        self.assertEqual(a['num_input'],100)
        np.testing.assert_array_equal(a['transformation'],c['transformation'])

if __name__=='__main__': unittest.main()
