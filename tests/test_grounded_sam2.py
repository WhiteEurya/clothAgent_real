import pytest
import numpy as np
from types import SimpleNamespace
from PIL import Image
from cloth_agent.grounded_sam2 import project_mask,merge_masks


def test_original_rgb_projection_inverts_depth_scale_without_mutation():
    mask=np.zeros((20,20),bool);mask[5:10,8:12]=True
    raw=np.array([[[9.,6.,100.],[2.,6.,100.],[np.nan,np.nan,np.nan]]])
    world=raw*1.02;copy=world.copy()
    result=project_mask(mask,world,np.eye(4),np.diag([100.,100.,1.]),1.02,np.zeros(3))
    assert result.tolist()==[[True,False,False]]
    assert np.allclose(world,copy,equal_nan=True)


def test_merge_respects_appearance_owner_instead_of_union():
    owner=np.array([[0,1,-1]])
    a=np.array([[True,True,True]]);b=np.array([[False,False,True]])
    assert merge_masks([a,b],owner).tolist()==[[1,0,0]]


def test_planner_routes_new_profile_and_reports_failure(tmp_path,monkeypatch):
    from cloth_agent.experiments.unfold_skill_planner import segment_observation
    from cloth_agent import grounded_sam2
    from cloth_agent.contour_normal_grasp import NormalGraspError
    import pytest
    before=SimpleNamespace(image=tmp_path/'rgb.png',segmentation_profile='grounded_sam2')
    expected=np.ones((3,3),np.uint8)
    monkeypatch.setattr(grounded_sam2,'segment_view',lambda view:(expected,{'method':'new'}))
    assert segment_observation(before)[0] is expected
    def fail(view):raise ValueError('no clothing')
    monkeypatch.setattr(grounded_sam2,'segment_view',fail)
    with pytest.raises(NormalGraspError,match='no clothing'):segment_observation(before)


@pytest.mark.parametrize('count',[2,3])
def test_rgb_mosaic_uses_color_owner_mapping_without_depth(tmp_path,monkeypatch,count):
    import hashlib,json
    from cloth_agent import grounded_sam2
    owner=np.zeros((12,12),np.int8);owner[:,6:]=1
    if count==3:owner[:,10:]=2
    yy,xx=np.indices(owner.shape)
    xy=np.stack([xx+2,yy+3],axis=2).astype(np.float32)
    images=[]
    for i in range(count):
        path=tmp_path/f'source_{i}.png'
        Image.fromarray(np.full((20,20,3),i,np.uint8)).save(path)
        images.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (tmp_path/'view.json').write_text(json.dumps(dict(mode='RGB_FIRST_DUAL_CAMERA' if count==2 else 'RGB_FIRST_MULTI_CAMERA',source_images=images)))
    np.save(tmp_path/'source_camera.npy',owner)
    np.save(tmp_path/'source_color_xy.npy',xy)
    np.save(tmp_path/'appearance_valid.npy',np.ones(owner.shape,bool))
    # No legacy surface Z, calibration, or usable measured depth is needed for a mask.
    depth=np.full((12,12,3),np.nan);np.save(tmp_path/'world_xyz_mm.npy',depth)
    Image.fromarray(np.zeros((12,12,3),np.uint8)).save(tmp_path/'rgb.png')
    calls=[]
    def segment(image,output,crop):
        calls.append(Path(image).name)
        mask=np.zeros((20,20),bool)
        if Path(image).name=='source_0.png':mask[5:13,4:12]=True
        else:mask[7:11,8:12]=True
        return mask,{}
    from pathlib import Path
    monkeypatch.setattr(grounded_sam2,'segment_file',segment)
    before=SimpleNamespace(directory=tmp_path,image=tmp_path/'rgb.png',sources=[None]*count)
    mask,report=grounded_sam2.segment_view(before)
    raw=np.asarray(Image.open(tmp_path/'grounded_sam2/projected_raw_mask.png'))>0
    assert raw[2,2] and not raw[2,6] and raw[4,6]
    assert calls==[f'source_{i}.png' for i in range(count)]
    assert report['depth_modified'] is False
    np.testing.assert_array_equal(np.load(tmp_path/'world_xyz_mm.npy'),depth)
