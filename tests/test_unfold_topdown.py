from types import SimpleNamespace
import json
import numpy as np
from PIL import Image
from cloth_agent.experiments import unfold_topdown as td
from cloth_agent.experiments import partitioned_unfold as unfold
from cloth_agent.pixel_mapping_visualization import PixelMappingView


def test_world_projection_highest_sample_wins_and_keeps_unknown():
    rgb=np.array([[[255,0,0],[0,255,0],[0,0,255]]],dtype=np.uint8)
    xyz=np.array([[[0,0,5],[0,0,20],[100,100,0]]],float)
    v=PixelMappingView(rgb,xyz,np.ones((1,3),bool),frame_label='WORLD XY TOP-DOWN',show_xy_grid=False,max_points=None)
    x,y=np.rint(v.xy_to_panel([0,0])).astype(int)
    assert np.array_equal(np.asarray(v.xy_image)[y,x],[0,255,0])
    x,y=np.rint(v.xy_to_panel([50,50])).astype(int)
    assert np.array_equal(np.asarray(v.xy_image)[y,x],[24,30,39])


def test_topdown_reaches_actual_selection_payload(tmp_path,monkeypatch):
    from cloth_agent.experiments import unfold_skill_planner as skills
    h,w=100,140;yy,xx=np.indices((h,w))
    image=tmp_path/'rgb.png';Image.new('RGB',(w,h),'white').save(image)
    (tmp_path/'goal').mkdir();Image.new('RGB',(w,h)).save(tmp_path/'goal/target.png')
    config=tmp_path/'topdown.json';config.write_text(json.dumps(dict(camera_serial='test',image_size_wh=[w,h],table_polygon_px=[[0,0],[w-1,0],[w-1,h-1],[0,h-1]])))
    monkeypatch.setattr(td,'CONFIG',config)
    before=SimpleNamespace(image=image,xyz=np.stack([xx,yy,np.zeros_like(xx)],axis=-1).astype(float),meta={'camera_serial':'test'})
    folder=tmp_path/'round';folder.mkdir()
    def direct(folder,body,*args):
        content=body['input'][0]['content']
        assert sum(c['type']=='input_image' for c in content)==2
        assert '均在第一张俯视图上输出' in body['instructions']
        assert next(c['image_url'] for c in content if c['type']=='input_image')==td.io.image_data(folder.parent/'topdown/planning_rgb.png')
        return dict(decision='ABSTAIN',skill=None,region_name='',evidence='offline test',grasp_px=None,target_px=None,boundary_point_px=None,chord_endpoints_px=[],pull_distance_mm=10),{}
    unfold.plan_round(tmp_path,folder,before,{},1,direct)
    assert json.loads((folder/'topdown/view.json').read_text())['status']=='AVAILABLE'
    saved=json.loads((folder/'skill_selection/request_payload.json').read_text())
    assert len([c for c in saved['input'][0]['content'] if c['type']=='input_image'])==2


def test_unavailable_projection_is_explicit_not_fabricated(tmp_path):
    report=td.reconstruct(SimpleNamespace(),tmp_path)
    assert report['status']=='UNAVAILABLE'
    assert not (tmp_path/'topdown/planning_rgb.png').exists()


def test_sparse_view_maps_contact_and_angle_without_coverage_gate(tmp_path):
    import pytest
    from cloth_agent.dual_arm.observation import Observation
    from cloth_agent.dual_arm.geometry import DualArmError
    h,w=100,140;yy,xx=np.indices((h,w))
    image=tmp_path/'rgb.png';Image.new('RGB',(w,h),'white').save(image)
    xyz=np.full((h,w,3),np.nan)
    xyz[40:49,60:69]=np.stack([xx[40:49,60:69],yy[40:49,60:69],np.full((9,9),50)],axis=-1)
    T=np.eye(4);T[:2,:2]=[[0,-1],[1,0]]
    source=Observation(tmp_path,dict(camera_serial='test',world_from_camera_mm=T.tolist(),intrinsics=[[100,0,70],[0,100,50],[0,0,1]]),xyz)
    config=dict(camera_serial='test',image_size_wh=[w,h],table_polygon_px=[[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    view=td.primary(source,tmp_path/'view',config=config)
    report=json.loads((view.directory/'view.json').read_text())
    assert report['status']=='AVAILABLE' and report['global_depth_coverage_gate'] is False
    assert report['valid_points']/(w*h)<.01
    y,x=np.argwhere(np.all(view.source_xy==[64,44],axis=2))[0]
    p=[int(x),int(y)]
    assert view.source_pixel(p)==[64,44]
    assert np.allclose(view.sample(p,None),[64,44,50])
    raw=view.raw_plan(dict(decision='ACTION',grasp_px=p,target_px=p,grasp_angle_deg=0))
    assert raw['grasp_px']==[64,44] and raw['grasp_angle_deg']==pytest.approx(90)
    with pytest.raises(DualArmError,match='no measured source'):
        view.sample([1,1],None)


def test_every_post_action_observation_is_reconstructed(tmp_path,monkeypatch):
    events=[]
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    before=SimpleNamespace(image=image,directory=tmp_path,xyz=np.zeros((100,100,3)))
    runtime=SimpleNamespace(observe=lambda path:events.append('observe') or before,
        execute=lambda *args:events.append('execute') or dict(status='EXECUTED',physical_execution=True),close=lambda:None)
    monkeypatch.setattr(td,'primary',lambda *args:events.append('reconstruct'))
    monkeypatch.setattr(unfold,'prepare_action',lambda *args:'source')
    plan=dict(decision='ACTION',grasp_px=[40,40],target_px=[60,40],is_smoothing=False,grasp_angle_deg=None)
    result=unfold.episode(runtime,tmp_path,max_actions=1,request=lambda *args:(plan,{}))
    assert result['physical_actions']==1
    assert events==['observe','reconstruct','execute','observe','reconstruct']
