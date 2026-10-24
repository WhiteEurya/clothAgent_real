from pathlib import Path
import numpy as np
from cloth_agent.dual_arm.observation import Observation
from cloth_agent.experiments import host_grasp_geometry_audit_v2 as a


def observation(size=45):
    y,x=np.mgrid[:size,:size]
    xyz=np.stack((x.astype(float),y.astype(float),np.full_like(x,20,dtype=float)),axis=2)
    return Observation(Path('unused'),{},xyz)


def test_exact_depth_gate_missing_and_range():
    o=observation();p=[22,22]
    assert a.depth_stats(o,p)['passed']
    o.xyz[22,22]=np.nan
    s=a.depth_stats(o,p);assert not s['passed'] and s['valid_count']==24 and not s['center_valid']
    o=observation();o.xyz[20,20,2]=28.001
    assert not a.depth_stats(o,p)['passed']
    o.xyz[20,20,2]=28
    assert a.depth_stats(o,p)['passed']


def test_world_plane_fit_removes_true_slope_not_step():
    y,x=np.mgrid[-2:3,-2:3];worldx=x.ravel()*.7+500;worldy=y.ravel()*.8+100
    z=3*worldx+2*worldy-1500
    m,p,r=a.fit_plane(np.c_[worldx,worldy,z])
    assert m['raw_z_range_mm']>8 and m['residual_rmse_mm']<1e-8
    assert abs(m['a']-3)<1e-8 and abs(m['b']-2)<1e-8
    step=np.where(x.ravel()>=0,20.,0.)
    m,p,r=a.fit_plane(np.c_[worldx,worldy,step])
    assert m['residual_rmse_mm']>1 and len(r)==25
    # Robust weights do not hide the outlier in reported all-point residuals.
    z=z.copy();z[0]+=40
    m,p,r=a.fit_plane(np.c_[worldx,worldy,z])
    assert m['max_absolute_residual_mm']>30


def test_connectivity_rejects_depth_step():
    o=observation();o.xyz[:,24:,2]+=20;rgb=np.zeros((45,45,3),np.uint8)
    assert not a.continuity_path(o,rgb,[22,22],[25,22],5)['connected']
    assert a.continuity_path(o,rgb,[22,22],[23,22],5)['connected']


def test_snap_missing_center_never_fabricates_original_xy():
    o=observation();o.xyz[22,22]=np.nan;rgb=np.zeros((45,45,3),np.uint8)
    rows,candidates=a.snap_search(o,rgb,[22,22],[35,22],7,'DEPTH_REJECTED')
    assert [r['radius_px'] for r in rows]==[3,5,10,15]
    assert rows[0]['found'] and rows[0]['selected_pixel_displacement']==1
    assert rows[0]['world_xy_displacement_mm'] is None
    assert rows[0]['original_center_world_z_delta_mm'] is None
    assert rows[0]['original_patch_median_proxy_xy_displacement_mm'] is not None
    assert rows[0]['unchanged_pb']==[35,22]
    assert a.snap_search(o,rgb,[22,22],[35,22],11,'SEPARATE_IK_FAILURE')==([],[])


def test_existing_pass_nearest_is_original():
    o=observation();rgb=np.zeros((45,45,3),np.uint8)
    rows,_=a.snap_search(o,rgb,[22,22],[35,22],1,'EXECUTED_REFERENCE')
    assert all(r['selected_pixel']==[22,22] and r['selected_pixel_displacement']==0 for r in rows)
