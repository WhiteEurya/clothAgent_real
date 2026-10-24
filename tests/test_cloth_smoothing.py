import json
import threading
import numpy as np
from types import SimpleNamespace
import pytest
from PIL import Image
from cloth_agent import cloth_smoothing as smooth
from cloth_agent.dual_arm.geometry import DualArmError


def proposal(decision='ACT',progress='INITIAL',confidence=.95):
    return dict(decision=decision,progress=progress,confidence=confidence,assessment='visible cloth',
                reason='edge pull',grasp_px=[100,100],target_px=[140,100],
                opening_angle_image_deg=90,expected_change='local spread')


def setup(tmp_path,monkeypatch,responses,*,geometry_failure=False,execution_failure=False):
    image=tmp_path/'rgb.png';Image.new('RGB',(200,200),'purple').save(image)
    obs=SimpleNamespace(image=image,xyz=np.zeros((200,200,3)))
    calls=[];requests=[]
    class Backend:
        model='fake'
        def invoke(self,prompt,images,schema):
            requests.append(images)
            return dict(text=json.dumps(responses.pop(0)))
    def prepare(*args):
        if geometry_failure:raise DualArmError('depth invalid')
        return 'source'
    monkeypatch.setattr(smooth,'prepare_step',prepare)
    def execute(*args):
        calls.append('execute')
        if execution_failure:raise DualArmError('motion fault')
    monkeypatch.setattr(smooth,'execute_program',execute)
    def capture(folder):
        folder.mkdir();Image.new('RGB',(200,200),'blue').save(folder/'rgb.png')
        calls.append('capture');return SimpleNamespace(image=folder/'rgb.png',xyz=np.zeros((200,200,3)))
    runtime=SimpleNamespace(cancel=threading.Event(),capture_current=capture,
                            move_to_perception=lambda p:calls.append('home'))
    recorder=SimpleNamespace(check=lambda:None,stage=lambda _:None)
    return runtime,recorder,obs,Backend(),calls,requests


@pytest.mark.parametrize('response,status',[(proposal(confidence=.66),'UNCERTAIN'),
    (proposal('DONE'),'MODEL_REPORTED_DONE'),(proposal('STOP'),'MODEL_STOP')])
def test_stops_without_dispatch(tmp_path,monkeypatch,response,status):
    r,rec,obs,b,calls,requests=setup(tmp_path,monkeypatch,[response])
    result=smooth.smooth_cloth(r,rec,obs,tmp_path/'smooth',[180,0,0],backend=b)
    assert result['status']==status and result['completed_actions']==0
    assert calls==[]


@pytest.mark.parametrize('progress,status',[('IMPROVED','MODEL_REPORTED_DONE'),
    ('NO_PROGRESS','NO_PROGRESS'),('WORSE','REASSESSMENT_STOP'),('SLIPPED','REASSESSMENT_STOP')])
def test_reobserves_and_never_blindly_repeats(tmp_path,monkeypatch,progress,status):
    r,rec,obs,b,calls,requests=setup(tmp_path,monkeypatch,[proposal(),proposal('DONE',progress)])
    result=smooth.smooth_cloth(r,rec,obs,tmp_path/'smooth',[180,0,0],backend=b)
    assert result['status']==status and result['completed_actions']==1
    assert calls==['execute','home','capture']
    assert len(requests[0])==1 and len(requests[1])==2
    assert requests[1][0]!=obs.image and requests[1][1]==obs.image
    assert not r.open_at_home_on_return


def test_step_limit_has_final_observation_and_assessment(tmp_path,monkeypatch):
    r,rec,obs,b,calls,requests=setup(tmp_path,monkeypatch,[proposal(),proposal(progress='IMPROVED')])
    result=smooth.smooth_cloth(r,rec,obs,tmp_path/'smooth',[180,0,0],max_steps=1,backend=b)
    assert result['status']=='STEP_LIMIT' and result['completed_actions']==1
    assert len(requests)==2


def test_bad_depth_rejects_without_moving(tmp_path,monkeypatch):
    r,rec,obs,b,calls,requests=setup(tmp_path,monkeypatch,[proposal(),proposal()],geometry_failure=True)
    result=smooth.smooth_cloth(r,rec,obs,tmp_path/'smooth',[180,0,0],backend=b)
    assert result['status']=='GEOMETRY_REJECTED' and not calls


def test_motion_fault_does_not_try_another_action(tmp_path,monkeypatch):
    r,rec,obs,b,calls,requests=setup(tmp_path,monkeypatch,[proposal()],execution_failure=True)
    with pytest.raises(DualArmError):smooth.smooth_cloth(r,rec,obs,tmp_path/'smooth',[180,0,0],backend=b)
    assert calls==['execute'] and len(requests)==1
    assert json.loads((tmp_path/'smooth/result.json').read_text())['status']=='FAILED'


@pytest.mark.parametrize('distance,height,allowed',[(25,0,True),(26,0,False),(20,6,False)])
def test_local_smoothing_hard_limits(tmp_path,monkeypatch,distance,height,allowed):
    import numpy as np
    image=tmp_path/'rgb.png';Image.new('RGB',(200,200),'purple').save(image)
    coords=iter([np.array([0.,0.,30.]),np.array([distance,0.,30.+height])])
    obs=SimpleNamespace(image=image,validate_for=lambda *a,**kw:None,sample=lambda *a:next(coords),
                        meta={'world_from_camera_mm':np.eye(4).tolist()})
    runtime=SimpleNamespace(camera=SimpleNamespace(intrinsics=np.eye(3)),
        config=SimpleNamespace(limits={'max_surface_spread_mm':8},arms={'left':SimpleNamespace(tcp_offset=np.zeros(6))}))
    seen=[]
    monkeypatch.setattr(smooth,'image_angle_to_rpy',lambda *a:(np.array([180,0,0]),0.))
    monkeypatch.setattr(smooth,'draw_preview',lambda *a:None)
    monkeypatch.setattr(smooth,'automatic_program',lambda *a,**kw:seen.append(kw) or 'source')
    if allowed:
        assert smooth.prepare_step(runtime,obs,proposal(),[180,0,0],-15,tmp_path)=='source'
        assert seen[0]['lift_height']==15
    else:
        with pytest.raises(DualArmError):smooth.prepare_step(runtime,obs,proposal(),[180,0,0],-15,tmp_path)
        assert not seen


def test_model_context_reports_actual_depth_and_configured_offset():
    y,x=np.mgrid[:20,:30]
    xyz=np.stack([x*.7,y*.8,np.ones_like(x)*30],axis=-1)
    xyz[0,0]=np.nan
    context=smooth.observation_context(SimpleNamespace(xyz=xyz),-12.)
    assert context['measured_depth_available']
    assert context['configured_motion']['grasp_offset_mm']==-12.
    assert context['approx_world_xy_mm_per_pixel']['image_x']==pytest.approx(.7)
    assert context['approx_world_xy_mm_per_pixel']['image_y']==pytest.approx(.8)
    assert 0<context['valid_depth_fraction']<1
    assert all(np.isfinite(s['world_xyz_mm']).all() for s in context['sparse_measured_geometry'])


def test_geometric_replan_happens_before_any_motion(tmp_path,monkeypatch):
    r,rec,obs,b,calls,requests=setup(tmp_path,monkeypatch,[proposal(),proposal(),proposal('DONE','IMPROVED')])
    attempts=[]
    def prepare(*args):
        attempts.append(1)
        assert not calls
        if len(attempts)==1:raise DualArmError('33.6 mm exceeds 25 mm')
        return 'revised source'
    monkeypatch.setattr(smooth,'prepare_step',prepare)
    result=smooth.smooth_cloth(r,rec,obs,tmp_path/'smooth',[180,0,0],backend=b)
    assert result['completed_actions']==1 and result['status']=='MODEL_REPORTED_DONE'
    assert len(attempts)==2 and calls==['execute','home','capture']
    assert result['steps'][0]['geometry_rejections'][0]['error']=='33.6 mm exceeds 25 mm'
    assert 'validation_error' not in result['steps'][0]
    assert (tmp_path/'smooth/step_0000/revised_proposal.json').exists()
