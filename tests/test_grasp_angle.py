import json
from types import SimpleNamespace
import threading
import numpy as np
import pytest
from PIL import Image
from scipy.spatial.transform import Rotation
from cloth_agent import grasp_angle as angle
from cloth_agent.dual_arm.geometry import DualArmError


@pytest.mark.parametrize('image_angle',[0,30,90,145,180])
def test_projected_opening_axis_matches_image_angle(image_angle):
    k=np.array([[700,0,640],[0,700,360],[0,0,1.]])
    camera=np.eye(4);camera[:3,:3]=Rotation.from_euler('xyz',[180,0,25],degrees=True).as_matrix();camera[2,3]=1000
    xyz=np.array([40.,50.,30.]);pixel=angle.project(xyz,k,camera)
    chosen,delta=angle.image_angle_to_rpy(image_angle,pixel,xyz,[178,2,15],k,camera,[0,1,0])
    axis=Rotation.from_euler('xyz',chosen,degrees=True).apply([0,1,0]);axis[2]=0
    a=angle.project(xyz-30*axis,k,camera);b=angle.project(xyz+30*axis,k,camera)
    actual=np.degrees(np.arctan2(*(b-a)[::-1]))%180
    assert ((actual-image_angle+90)%180-90)==pytest.approx(0,abs=1e-6)
    assert abs(delta)<=90
    # Premultiplying world yaw must retain downward tilt.
    assert Rotation.from_euler('xyz',chosen,degrees=True).as_matrix()[2,2]==pytest.approx(
        Rotation.from_euler('xyz',[178,2,15],degrees=True).as_matrix()[2,2])


@pytest.mark.parametrize('accept',[True,False])
def test_operator_review_is_required_and_recorded(tmp_path,accept,monkeypatch):
    def no_model(*args, **kwargs):pytest.fail("manual angle selection must not invoke a model")
    monkeypatch.setattr("cloth_agent.experiments.discovery_model.CodexMultimodal",no_model)
    image=tmp_path/'rgb.png';Image.new('RGB',(640,480),'purple').save(image)
    k=np.array([[500,0,320],[0,500,240],[0,0,1.]])
    camera=np.diag([1.,-1.,-1.,1.]);camera[2,3]=1000
    runtime=SimpleNamespace(camera=SimpleNamespace(intrinsics=k),cancel=threading.Event(),
        config=SimpleNamespace(arms={'left':SimpleNamespace(tcp_offset=np.zeros(6))}))
    obs=SimpleNamespace(image=image,meta={'world_from_camera_mm':camera.tolist()})
    points=[dict(name='start',pixel_xy=[320,240],world_xyz_mm=[0,0,30]),
            dict(name='target',pixel_xy=[400,240],world_xyz_mm=[150,0,30])]
    reviews=[]
    def review(proposal, render, grasp_pixel):
        reviews.append(proposal)
        im, delta = render(75.)
        assert im.size == (640,480)
        return 75. if accept else None
    folder=tmp_path/'angle'
    if not accept:
        with pytest.raises(KeyboardInterrupt):angle.choose_angle(runtime,obs,points,[180,0,0],folder,review=review)
        assert not json.loads((folder/'decision.json').read_text())['operator_accepted']
    else:
        value=angle.choose_angle(runtime,obs,points,[180,0,0],folder,review=review)
        assert len(reviews)==1 and np.isfinite(value).all()
        decision=json.loads((folder/'decision.json').read_text())
        assert decision['operator_accepted']
        assert decision['selected_angle_image_deg']==75.
        assert decision['initial_angle_image_deg']==pytest.approx(90.)
        assert decision['manually_adjusted']
        assert decision['model_calls']==0
        assert not (folder/'model').exists()
        expected,_=angle.image_angle_to_rpy(75.,[320,240],[0,0,30],[180,0,0],k,camera,[0,1,0])
        assert value==pytest.approx(expected)
        assert (folder/'preview.png').exists()


def test_centered_scaled_image_click_coordinates():
    assert angle.display_pixel(100,100,(1000,800),(800,600),(1600,1200))==[0,0]
    assert angle.display_pixel(500,400,(1000,800),(800,600),(1600,1200))==[800,600]
    assert angle.display_pixel(99,400,(1000,800),(800,600),(1600,1200)) is None
    assert angle.display_pixel(900,400,(1000,800),(800,600),(1600,1200)) is None


@pytest.mark.parametrize('manual',[False,True])
def test_combined_selection_one_review_and_reselection(tmp_path,monkeypatch,manual):
    image=tmp_path/'rgb.png';Image.new('RGB',(640,480),'purple').save(image)
    camera=np.diag([1.,-1.,-1.,1.]);camera[2,3]=1000
    runtime=SimpleNamespace(camera=SimpleNamespace(intrinsics=np.array([[500,0,320],[0,500,240],[0,0,1.]])),
        cancel=threading.Event(),config=SimpleNamespace(limits={'max_surface_spread_mm':8},
        arms={'left':SimpleNamespace(tcp_offset=np.zeros(6))}))
    def sample(pixel,spread):
        if pixel==[1,1]:raise DualArmError('invalid depth')
        return np.array([pixel[0]-320,240-pixel[1],30.])
    obs=SimpleNamespace(image=image,meta={'world_from_camera_mm':camera.tolist()},sample=sample)
    monkeypatch.setattr(angle,'confirm_preview',lambda *a,**kw:pytest.fail('second window must not open'))
    seen=[]
    def review(proposal,render,grasp_pixel,*,selection):
        seen.append('one_window');assert not selection.ready
        selection.manual_grasp=manual
        assert render(0)[0].size==(640,480)
        with pytest.raises(DualArmError):selection.click([1,1])
        assert selection.points==[None,None]
        assert selection.click([320,240]);assert not selection.ready
        selection.click([322,240]);assert not selection.ready
        selection.edit(1);selection.click([420,240]);assert selection.ready
        assert grasp_pixel()==[320,240]
        selection.edit(0);assert not selection.ready
        selection.click([330,250]);assert selection.ready
        assert render(45)[0].size==(640,480)
        return 0.  # Zero degrees is an accepted selection, not cancellation.
    points,chosen,manual_grasp=angle.select_points_and_angle(runtime,obs,[180,0,0],tmp_path/'angle',review=review)
    assert seen==['one_window']
    assert manual_grasp==manual
    assert points[0]['pixel_xy']==[330,250] and points[1]['pixel_xy']==[420,240]
    decision=json.loads((tmp_path/'angle/decision.json').read_text())
    assert decision['operator_accepted'] and decision['selected_angle_image_deg']==0.
    assert chosen==pytest.approx(decision['rpy_world_deg'])


@pytest.mark.parametrize('action',['codex_smooth','random_reset'])
def test_codex_smoothing_selection_needs_no_manual_points(tmp_path,action):
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100),'purple').save(image)
    runtime=SimpleNamespace(camera=SimpleNamespace(intrinsics=np.eye(3)),
        config=SimpleNamespace(limits={'max_surface_spread_mm':8},arms={'left':SimpleNamespace(tcp_offset=np.zeros(6))}))
    obs=SimpleNamespace(image=image,meta={'world_from_camera_mm':np.eye(4).tolist()})
    points,rpy,mode=angle.select_points_and_angle(runtime,obs,[180,0,0],tmp_path/'angle',
        review=lambda *a,**kw:action)
    assert points is None and mode==action
    assert json.loads((tmp_path/'angle/grasp_control.json').read_text())['operator_accepted']
