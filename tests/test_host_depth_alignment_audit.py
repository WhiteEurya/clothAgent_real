import unittest
import numpy as np
from cloth_agent.experiments.host_depth_alignment_audit import align_native,sample

class AlignmentTests(unittest.TestCase):
    def test_holes_and_identity(self):
        k=np.array([[10.,0,5],[0,10,5],[0,0,1]])
        a=align_native(np.ones((12,12)),k,k,np.eye(4),(12,12))
        self.assertAlmostEqual(a['rgb_z_m'][5,5],1.)
        np.testing.assert_allclose(a['rgb_xyz_m'][5,5],[0,0,1])
        empty=align_native(np.zeros((12,12)),k,k,np.eye(4),(12,12))
        self.assertFalse(np.isfinite(empty['rgb_z_m']).any());self.assertTrue((empty['source_index']==-1).all())
    def test_depth_axis_is_not_color_axis(self):
        k=np.array([[10.,0,5],[0,10,5],[0,0,1]]);t=np.eye(4);t[2,3]=.1
        a=align_native(np.ones((12,12)),k,k,t,(12,12))
        self.assertAlmostEqual(a['source_native_z_m'][5,5],1.)
        self.assertAlmostEqual(a['rgb_z_m'][5,5],1.1)
        self.assertAlmostEqual(a['legacy_xyz_m'][5,5,2],1.)
    def test_collision_nearer_native_depth_wins(self):
        kd=np.eye(3);kc=np.array([[.1,0,5],[0,.1,5],[0,0,1]])
        a=align_native(np.array([[2.,1.]]),kd,kc,np.eye(4),(12,12))
        self.assertEqual(a['source_native_z_m'][5,5],1.)
        self.assertEqual(a['source_index'][5,5],1)
    def test_inverse_transform_recovers_source_depth(self):
        theta=.1;t=np.eye(4);t[:3,:3]=[[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]];t[:3,3]=[.015,.002,.003]
        k=np.array([[30.,0,10],[0,30,10],[0,0,1]]);a=align_native(np.full((24,24),.6),k,k,t,(24,24));valid=np.isfinite(a['rgb_z_m'])
        back=(a['rgb_xyz_m'][valid]-t[:3,3])@t[:3,:3]
        np.testing.assert_allclose(back[:,2],a['source_native_z_m'][valid],atol=1e-12)
    def test_invalid_rigid_transform_rejected(self):
        t=np.eye(4);t[0,0]=2
        with self.assertRaises(ValueError):align_native(np.ones((3,3)),np.eye(3),np.eye(3),t,(3,3))
    def test_sampling_does_not_fill_holes(self):
        xyz=np.ones((10,10,3));xyz[5,5]=np.nan
        self.assertIsNone(sample(xyz,5,5))
        self.assertIsNone(sample(xyz,0,0))

if __name__=='__main__':unittest.main()
