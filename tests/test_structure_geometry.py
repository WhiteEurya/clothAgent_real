from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from cloth_agent import geometry_report as g
from cloth_agent import restricted_context_one_shot as r


def fixture():
    rgb=np.full((120,160,3),[180,120,200],dtype=np.uint8)
    y,x=np.indices(rgb.shape[:2]);xyz=np.stack([x,y,x*.5],axis=-1).astype(float)
    coords=[(50,90),(50,60),(40,50),(40,30),(110,30),(110,50),(100,60),(100,90)]
    points={k:dict(visibility='VISIBLE',px=list(p),evidence='visible boundary') for k,p in zip(g.LANDMARKS,coords)}
    return rgb,xyz,points


def test_chords_are_anatomical_and_xy_distinct_from_xyz():
    rgb,xyz,points=fixture();report=g.measure_structure(rgb,xyz,points)
    hem=report['segments'][-1]
    assert hem['endpoints']==['right_hem','left_hem']
    assert hem['xy_mm']==50
    assert hem['xyz_mm']==pytest.approx(55.9)
    assert len(report['segments'])==8


def test_hidden_identity_and_depth_holes_never_get_imputed():
    rgb,xyz,points=fixture();points['right_hem'].update(visibility='UNCERTAIN',px=None)
    xyz[58:63,48:53]=np.nan
    report=g.measure_structure(rgb,xyz,points)
    assert report['points']['right_hem']['world_mm'] is None
    assert report['segments'][-1]['xy_mm'] is None
    assert report['segments'][-2]['xyz_mm'] is None
    assert report['segments'][0]['xy_mm'] is None
    assert report['segments'][3]['xy_mm']==70


def test_structure_precedes_action_and_is_measured_in_payload(tmp_path,monkeypatch):
    rgb,xyz,points=fixture()
    goal=tmp_path/'goal';goal.mkdir();Image.fromarray(rgb).save(goal/'target.png')
    folder=tmp_path/'round';(folder/'before').mkdir(parents=True)
    Image.fromarray(rgb).save(folder/'before/rgb.png')
    before=SimpleNamespace(image=folder/'before/rgb.png',xyz=xyz,meta={})
    monkeypatch.setattr(g,'REFERENCE',folder/'before')
    from cloth_agent.dual_arm.observation import Observation
    monkeypatch.setattr(Observation,'load',lambda _:before)
    monkeypatch.setattr(g,'reference',lambda _:dict(source=str(folder/'before')))
    calls=[]
    def direct(dest,body,*args):
        calls.append((dest,body))
        if len(calls)==1:
            return dict(current=points,reference=points),dict(wall_latency_s=2.)
        assert 'xy_ratio_to_reference' in str(body)
        return dict(decision='ACTION',grasp_px=[52,88],target_px=[65,88],motion_type='lift_pull_place',
                    grasp_angle_deg=90,short_reason='抓清晰下摆',is_smoothing=False),dict(wall_latency_s=3.)
    _,metrics=r.plan_round(tmp_path,folder,before,{},10,direct)
    assert len(calls)==2 and calls[0][0]==folder/'structure_grounding' and calls[1][0]==folder
    assert metrics['wall_latency_s']==5 and metrics['action_model_wall_s']==3
    assert (folder/'structure_current.png').exists()
    report=r.io.read(folder/'structure_report.json')
    assert report['current']['segments'][-1]['xy_ratio_to_reference']==1
    assert r.io.read(folder/'request_manifest.json')['model_calls']==2
