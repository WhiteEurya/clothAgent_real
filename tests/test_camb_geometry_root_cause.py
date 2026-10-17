import unittest
import cv2
import numpy as np
from cloth_agent.experiments.camb_geometry_root_cause import optical_z_in_color,robust_plane
class RootCauseTests(unittest.TestCase):
    def test_depth_to_color_optical_z(self):
        r=cv2.Rodrigues(np.array([.008,-.005,.007]))[0];t=np.array([15.,.2,-.12])
        pd=np.array([[-150.,50.,600.],[200.,-70.,650.]])
        pc=pd@r.T+t
        ray=pc/pc[:,2:]
        np.testing.assert_allclose(optical_z_in_color(pd[:,2],ray,r,t),pc[:,2],atol=1e-10)
    def test_robust_plane_rejects_flying_points(self):
        rng=np.random.default_rng(0);xy=rng.uniform(-100,100,(1000,2))
        z=.03*xy[:,0]-.02*xy[:,1]+600
        z[:15]+=1500
        b,n,res=robust_plane(np.c_[xy,z])
        np.testing.assert_allclose(b,[.03,-.02,600],atol=1e-8)
if __name__=='__main__':unittest.main()
