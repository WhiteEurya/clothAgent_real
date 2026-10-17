import unittest
import numpy as np
import cv2
import pyrealsense2 as rs
from cloth_agent.experiments.camb_cross_stream_audit import transform
from cloth_agent.experiments.apriltag_geometry_audit import project_rgb

class CrossStreamTests(unittest.TestCase):
    def test_transform_and_project_match_sdk(self):
        r=cv2.Rodrigues(np.array([.007,-.005,-.006]))[0]
        t=np.array([.015,.0002,-.00012])
        metadata={'extrinsics':{'depth_to_rgb':dict(rotation_column_major=r.flatten(order='F').tolist(),translation_m=t.tolist())}}
        T=transform(metadata,'rgb')
        ext=rs.extrinsics();ext.rotation=r.flatten(order='F').tolist();ext.translation=t.tolist()
        intr=rs.intrinsics();intr.width=1280;intr.height=720
        intr.fx=907.;intr.fy=906.;intr.ppx=646.;intr.ppy=381.;intr.model=rs.distortion.inverse_brown_conrady;intr.coeffs=[0.]*5
        K=np.array([[907.,0,646.],[0,906.,381.],[0,0,1.]])
        points=np.array([[-.2,.05,.6],[.2,-.1,.65]])
        projected=project_rgb(T,points,K,np.zeros(5))
        expected=[rs.rs2_project_point_to_pixel(intr,rs.rs2_transform_point_to_point(ext,p.tolist())) for p in points]
        np.testing.assert_allclose(projected,expected,atol=.0001)
    def test_stereo_triangulation_depth_frame(self):
        k=np.array([[648.,0,640.],[0,648.,355.],[0,0,1.]])
        T1=np.eye(4);T2=np.eye(4);T2[0,3]=-.05
        points=np.array([[-.1,.05,.6],[.15,.1,.65],[0.,-.03,.55]])
        uv1=project_rgb(T1,points,k,np.zeros(5));uv2=project_rgb(T2,points,k,np.zeros(5))
        h=cv2.triangulatePoints(k@T1[:3],k@T2[:3],uv1.T,uv2.T)
        np.testing.assert_allclose((h[:3]/h[3:]).T,points,atol=1e-10)
if __name__=='__main__':unittest.main()
