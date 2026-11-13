import numpy as np
from cloth_agent.dual_cloud_render import surface_raster,texture_surface,small_enclosed_holes,align_secondary_color


def plane():
    y,x=np.indices((25,25));return np.stack([x-12,y-12,np.full_like(x,100)],-1).astype(float)


def test_surface_interpolates_appearance_without_changing_input():
    xyz=plane();original=xyz.copy();valid=np.ones(xyz.shape[:2],bool)
    result=surface_raster(xyz,valid,[0,0],100,supersample=1)
    assert np.array_equal(xyz,original)
    assert np.allclose(result[320,320],[0,0,100])
    assert np.allclose(result[320,321],[100/520,0,100],atol=1e-5)
    assert np.isnan(result[0,0]).all()


def test_mesh_does_not_bridge_missing_native_pixels_or_depth_step():
    xyz=plane();valid=np.ones(xyz.shape[:2],bool);valid[:,11:14]=False
    result=surface_raster(xyz,valid,[0,0],100,supersample=1)
    assert np.isnan(result[320,320]).all()
    valid[:]=True;xyz[:,13:,2]=150
    result=surface_raster(xyz,valid,[0,0],100,supersample=1)
    assert np.isnan(result[320,322]).all()


def test_texture_uses_original_rgb_and_does_not_mutate_world():
    y,x=np.indices((30,30));rgb=np.stack([x,y,x+y],-1).astype('uint8')
    world=np.array([[[10.5,12.5,100.],[np.nan,np.nan,np.nan]]]);copy=world.copy()
    result,valid,uv=texture_surface(world,rgb,np.eye(4),np.diag([100.,100.,1.]))
    assert np.allclose(world,copy,equal_nan=True)
    assert np.allclose(uv[0,0],[10.5,12.5])
    assert valid.tolist()==[[True,False]] and result[0,0,2]==23


def test_color_hole_completion_excludes_exterior_and_large_unknown_regions():
    valid=np.ones((30,30),bool);valid[:3]=False;valid[6:8,6:8]=False;valid[15:25,15:25]=False
    fill=small_enclosed_holes(valid,16)
    assert fill.sum()==4 and fill[6,6] and not fill[16,16] and not fill[0,0]


def test_gain_only_alignment_preserves_black_and_checks_current_frame():
    secondary=np.full((200,200,3),200,np.uint8);secondary[40:160,40:160]=0
    primary=np.rint(secondary*.8).astype(np.uint8);y,x=np.indices((200,200));xyz=np.stack([x,y,np.ones_like(x)],-1).astype(float)
    result,report=align_secondary_color(primary,secondary,xyz,xyz)
    assert report['accepted'] and np.array_equal(result,primary)
    assert report['offset_rgb']==[0,0,0]


def test_upper_cloth_requires_an_interior_not_a_depth_noise_fringe():
    from cloth_agent.dual_cloud_render import upper_surface_patch
    shape=(100,100);valid=np.ones(shape,bool)
    world=np.zeros((*shape,3));higher=world.copy();higher[:,:,2]=20
    primary=np.zeros(shape,bool);primary[30:70,30:70]=True
    secondary=np.zeros(shape,bool);secondary[28:72,28:72]=True
    assert not upper_surface_patch([valid,valid],[higher,world],[secondary,primary]).any()
    secondary[35:65,70:95]=True
    patch=upper_surface_patch([valid,valid],[higher,world],[secondary,primary])
    assert patch[50,85] and not patch[50,50]
    higher[:,:,2]=8
    assert not upper_surface_patch([valid,valid],[higher,world],[secondary,primary]).any()
