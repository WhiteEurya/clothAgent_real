import numpy as np
import pytest
from cloth_agent.dual_rgb_mosaic import render

def camera(z=10.):
 y,x=np.indices((21,21));xyz=np.stack([(x-10)*z/10,(y-10)*z/10,np.full_like(x,z)],-1).astype(float)
 rgb=np.stack([x*10,y*10,np.full_like(x,80)],-1).astype(np.uint8)
 return dict(rgb=rgb,K=np.array([[10.,0,10],[0,10,10],[0,0,1]]),world_from_color=np.eye(4),xyz=xyz)

def test_rgb_is_unchanged_by_missing_depth_and_never_fabricates_xyz():
 a=camera();b=camera();args=dict(bounds_xy_mm=[[-5,-5],[5,5]],plane_z_mm=10,size=11)
 good=render([a,b],**args);a['xyz'][:]=np.nan
 missing=render([a,b],**args)
 np.testing.assert_array_equal(missing['rgb'],good['rgb'])
 assert missing['appearance_valid'].all() and not missing['depth_valid'].any()
 assert np.isnan(missing['world_xyz_mm']).all() # no silent other-camera substitution
 assert (missing['source_camera']==0).all()

def test_raised_surface_uses_original_pixel_depth_not_display_plane():
 a=camera(6);result=render([a,camera()],bounds_xy_mm=[[-5,-5],[5,5]],plane_z_mm=10,size=11)
 np.testing.assert_allclose(result['source_color_xy'][5,8],[13,10])
 np.testing.assert_array_equal(result['source_xy'][5,8],[13,10])
 np.testing.assert_allclose(result['world_xyz_mm'][5,8],a['xyz'][10,13])
 assert result['world_xyz_mm'][5,8,2]==6

def test_mosaic_view_separates_visible_rgb_from_invalid_action(tmp_path):
 from PIL import Image
 from types import SimpleNamespace
 from cloth_agent.experiments.unfold_dual_topdown import RGBMosaicView
 from cloth_agent.dual_arm.observation import DepthUnavailableError
 import json
 a=camera();a['xyz'][10,10]=np.nan;r=render([a,camera()],bounds_xy_mm=[[-5,-5],[5,5]],plane_z_mm=10,size=11)
 # Explicitly retain a display pixel with no supported depth.
 r['world_xyz_mm'][5,5]=np.nan
 for name in ('source_camera','source_color_xy','source_xy','world_xyz_mm','appearance_valid'):np.save(tmp_path/(name+'.npy'),r[name])
 Image.fromarray(r['rgb']).save(tmp_path/'planning_rgb.png');(tmp_path/'view.json').write_text(json.dumps({'mm_per_pixel':1}))
 source=SimpleNamespace(xyz=a['xyz'],meta={'camera_serial':'left','observation_id':'test'})
 v=RGBMosaicView(source,tmp_path,[source,source]);assert not v.unknown_mask[5,5]
 with pytest.raises(DepthUnavailableError,match='display plane'):v.sample([5,5],None)
 p=v.sample([8,5],None);np.testing.assert_array_equal(p,a['xyz'][10,13]);assert v.reference([8,5])['original_color_pixel_xy']==[13.,10.]


def test_third_camera_fills_uncovered_region_with_its_native_depth():
 a=camera();b=camera();zed=camera()
 # Wrists have a narrower RGB field, but still share the same coordinates.
 for c in (a,b):c['rgb']=c['rgb'][5:16,5:16];c['K']=c['K'].copy();c['K'][:2,2]-=5
 r=render([a,b,zed],bounds_xy_mm=[[-8,-8],[8,8]],plane_z_mm=10,size=17)
 assert r['source_camera'][8,8]==0
 y,x=np.nonzero(r['source_camera']==2)
 assert len(x)>0
 for row,col in zip(y,x):
  nx,ny=r['source_xy'][row,col]
  np.testing.assert_allclose(r['world_xyz_mm'][row,col],zed['xyz'][ny,nx])


def test_graphcut_updates_color_and_measured_source_together(monkeypatch):
 import cv2
 class Seam:
  def find(self, images, corners, masks):
   h,w=images[0].shape[:2]
   masks[0]=cv2.UMat(np.zeros((h,w),np.uint8))
   masks[1]=cv2.UMat(np.full((h,w),255,np.uint8))
 monkeypatch.setattr(cv2,'detail_GraphCutSeamFinder',lambda *a:Seam())
 a=camera(10);b=camera(6)
 r=render([a,b],bounds_xy_mm=[[-5,-5],[5,5]],plane_z_mm=10,size=21,seam_mode='graphcut')
 assert r['report']['seam']['status']=='APPLIED'
 assert (r['source_camera']==1).all()
 for y,x in zip(*np.nonzero(r['depth_valid'])):
  nx,ny=r['source_xy'][y,x]
  np.testing.assert_allclose(r['world_xyz_mm'][y,x],b['xyz'][ny,nx],atol=1e-6)
 a['xyz'][:]=np.nan;b['xyz'][:]=np.nan
 missing=render([a,b],bounds_xy_mm=[[-5,-5],[5,5]],plane_z_mm=10,size=21,seam_mode='graphcut')
 np.testing.assert_array_equal(missing['rgb'],r['rgb'])
 assert not missing['depth_valid'].any()


def test_graphcut_failure_preserves_priority_coverage(monkeypatch):
 import cv2
 class Seam:
  def find(self,*args):raise cv2.error('test seam failure')
 monkeypatch.setattr(cv2,'detail_GraphCutSeamFinder',lambda *a:Seam())
 args=dict(bounds_xy_mm=[[-5,-5],[5,5]],plane_z_mm=10,size=21)
 expected=render([camera(),camera()],**args)
 actual=render([camera(),camera()],**args,seam_mode='graphcut')
 assert actual['report']['seam']['status']=='PRIORITY_FALLBACK'
 np.testing.assert_array_equal(actual['rgb'],expected['rgb'])
 np.testing.assert_array_equal(actual['source_camera'],expected['source_camera'])


def test_feature_alignment_reduces_image_error_and_reports_bounded_transform():
 import cv2
 from cloth_agent.dual_rgb_mosaic import _feature_align_secondary
 rng=np.random.default_rng(17);image=np.full((240,260,3),[165,132,72],np.uint8)
 for _ in range(180):
  center=tuple(rng.integers([10,10],[250,230]));color=tuple(map(int,rng.integers(20,240,3)))
  cv2.circle(image,center,int(rng.integers(2,7)),color,-1)
 valid=np.ones(image.shape[:2],bool);yy,xx=np.indices(valid.shape);uv=np.stack([xx,yy],-1).astype(np.float32)
 motion=np.array([[1.,-.012,9.],[.012,1.,-7.]],np.float32)
 secondary=cv2.warpAffine(image,motion,(image.shape[1],image.shape[0]),borderValue=(24,30,39))
 secondary_valid=cv2.warpAffine((valid*255).astype(np.uint8),motion,(image.shape[1],image.shape[0]),flags=cv2.INTER_NEAREST)>0
 secondary_uv=cv2.warpAffine(uv,motion,(image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR,borderValue=(-1,-1))
 aligned,report=_feature_align_secondary((image,valid,uv),(secondary,secondary_valid,secondary_uv))
 assert report['status']=='APPLIED' and report['inliers']>=8 and report['p95_residual_px']<4
 overlap=valid&aligned[1]
 before=np.mean(np.abs(image[overlap].astype(float)-secondary[overlap].astype(float)))
 after=np.mean(np.abs(image[overlap].astype(float)-aligned[0][overlap].astype(float)))
 assert after<before*.35
 assert np.isfinite(aligned[2][overlap]).all()


def test_feature_feather_retains_original_measured_source_mapping(monkeypatch):
 import cloth_agent.dual_rgb_mosaic as mosaic
 def accepted(primary,secondary):return secondary,{'status':'APPLIED','inliers':20,'p95_residual_px':1.}
 monkeypatch.setattr(mosaic,'_feature_align_secondary',accepted)
 a=camera(10);b=camera(6)
 result=render([a,b],bounds_xy_mm=[[-5,-5],[5,5]],plane_z_mm=10,size=21,seam_mode='feature_feather')
 assert result['report']['seam']['status']=='APPLIED'
 for y,x in zip(*np.nonzero(result['depth_valid'])):
  owner=result['source_camera'][y,x];nx,ny=result['source_xy'][y,x]
  expected=(a,b)[owner]['xyz'][ny,nx]
  np.testing.assert_allclose(result['world_xyz_mm'][y,x],expected,atol=1e-6)


def test_feature_graphcut_uses_all_three_sources_and_preserves_shifted_uv(monkeypatch):
 import cv2
 import cloth_agent.dual_rgb_mosaic as mosaic
 def align(primary,secondary):
  rgb,valid,uv=secondary
  uv=uv.copy();uv[:,:,0]+=1
  return (rgb,valid,uv),{'status':'APPLIED'}
 class Seam:
  def find(self,images,corners,masks):
   assert len(images)==3
   h,w=images[0].shape[:2]
   for i in range(3):masks[i]=cv2.UMat(np.full((h,w),255 if i==2 else 0,np.uint8))
 monkeypatch.setattr(mosaic,'_feature_align_secondary',align)
 monkeypatch.setattr(cv2,'detail_GraphCutSeamFinder',lambda *a:Seam())
 cams=[camera(),camera(),camera(6)]
 args=dict(bounds_xy_mm=[[-5,-5],[5,5]],plane_z_mm=10,size=21,primary=1,seam_mode='feature_graphcut')
 result=render(cams,**args)
 assert (result['source_camera']==2).all()
 assert result['report']['seam']['optimized_sources']==[0,1,2]
 for y,x in zip(*np.nonzero(result['depth_valid'])):
  nx,ny=result['source_xy'][y,x]
  np.testing.assert_allclose(result['world_xyz_mm'][y,x],cams[2]['xyz'][ny,nx])
 for c in cams:c['xyz'][:]=np.nan
 missing=render(cams,**args)
 np.testing.assert_array_equal(missing['rgb'],result['rgb'])
 assert not missing['depth_valid'].any()
