import numpy as np
import pytest
from cloth_agent.experiments.cloth_moving_release_trial import image_down_direction

def test_direction_projects_down_without_horizontal_pixel_drift():
    K=np.array([[700.,0,640],[0,700,360],[0,0,1]])
    T=np.diag([1.,-1.,-1.,1.]);T[:3,3]=[500,300,600]
    anchor=np.array([530,280,200.])
    d,points=image_down_direction(dict(intrinsics=K,world_from_camera_mm=T),anchor)
    target=anchor+np.r_[60*d,0]
    def project(p):
        c=np.linalg.solve(T,np.r_[p,1])[:3];h=K@c;return h[:2]/h[2]
    delta=project(target)-project(anchor)
    assert abs(delta[0])<1e-8 and delta[1]>0
    assert target[2]==anchor[2] and np.isclose(np.linalg.norm(target-anchor),60)

def test_behind_camera_rejected():
    with pytest.raises(ValueError,match='behind camera'):
        image_down_direction(dict(intrinsics=np.eye(3),world_from_camera_mm=np.eye(4)),[0,0,-1])
