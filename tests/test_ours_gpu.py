"""End-to-end contract for the CUDA optimization, also exercising CPU fallback."""
import unittest
import torch
import turboreg_gpu as backend

class OursGpuContract(unittest.TestCase):
    def options(self,n=100):
        o=backend.Sim3Options()
        o.max_points=n;o.num_pivots=32;o.k_neighbors=5;o.graph_iterations=1
        o.refinement_iterations=1;o.nms_radius=0;o.use_second_order=False
        o.scale_min=.01;o.scale_max=20;o.inlier_threshold=.025
        return o

    def cloud(self,n=100,device='cuda'):
        g=torch.Generator().manual_seed(923)
        a=torch.randn(n,3,generator=g).to(device)
        R=torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]],device=device)
        t=torch.tensor([.3,-.2,.7],device=device)
        return a,R,t

    def test_modes_neighbor_limits_scales_and_input_preservation(self):
        for device in ['cpu','cuda']:
            a,R,t=self.cloud(150,device);original=a.clone()
            for mode in ['hist','median','union_median']:
                for k in [1,5,128,129]:
                    for s in [.05,2.,12.]:
                        with self.subTest(device=device,mode=mode,k=k,scale=s):
                            o=self.options(150);o.local_scale_mode=mode;o.k_neighbors=k
                            dst=s*a@R.T+t;copy=dst.clone()
                            result=backend.register_sim3(a,dst,o)
                            torch.testing.assert_close(result.transformation[:3,:3],s*R,rtol=1e-4,atol=1e-4)
                            torch.testing.assert_close(result.transformation[:3,3],t,rtol=1e-4,atol=1e-4)
                            self.assertEqual(result.num_inliers,150)
                            self.assertTrue(torch.equal(a,original) and torch.equal(dst,copy))

    def test_minimal_duplicates_noncontiguous_and_float64(self):
        o=self.options()
        for dtype in [torch.float32,torch.float64]:
            tri=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],device='cuda',dtype=dtype)
            for a in [tri,torch.cat([tri,tri])]:
                self.assertEqual(backend.register_sim3(a,2*a,o).num_inliers,len(a))
            a,R,t=self.cloud();a=a.T.contiguous().T.to(dtype)
            self.assertEqual(backend.register_sim3(a,2*a,o).num_inliers,len(a))

    def test_nondefault_stream_and_outliers(self):
        with torch.cuda.stream(torch.cuda.Stream()):
            a,R,t=self.cloud();dst=1.5*a@R.T+t
            dst[60:]=8*torch.randn_like(dst[60:])
            o=self.options();o.k_neighbors=20;o.num_pivots=128
            r=backend.register_sim3(a,dst,o)
            self.assertAlmostEqual(r.scale,1.5,places=4)
            self.assertEqual(r.num_inliers,60)
        torch.cuda.synchronize()

    def test_invalid_and_degenerate_inputs(self):
        a,_,_=self.cloud();o=self.options()
        bad=a.clone();bad[0,0]=float('nan')
        line=torch.zeros(6,3,device='cuda');line[:,0]=torch.arange(6,device='cuda')
        for x,y in [(a,a[:-1]),(a[:2],a[:2]),(bad,a),(line,2*line),
                    (torch.zeros_like(a),torch.zeros_like(a))]:
            with self.assertRaises(RuntimeError):backend.register_sim3(x,y,o)

    def test_optional_graph_and_refinement_modes(self):
        a,R,t=self.cloud();dst=2*a@R.T+t
        for second in [False,True]:
            for tsc in [False,True]:
                o=self.options();o.use_second_order=second;o.use_tsc=tsc
                o.graph_iterations=2;o.refinement_iterations=3
                r=backend.register_sim3(a,dst,o)
                self.assertEqual(r.num_inliers,len(a))
                self.assertAlmostEqual(r.scale,2.,places=4)

if __name__=='__main__':unittest.main()
