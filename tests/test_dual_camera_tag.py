from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from cloth_agent.dual_camera_tag import rigid,fit_camera_transform,compare_tag,summarize,estimate_tag


def transform(xyz, rpy):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',rpy,degrees=True).as_matrix();t[:3,3]=xyz
    return t


def test_different_camera_coordinates_align_after_transform():
    ab=transform([350,-40,25],[3,20,170]);a=transform([40,20,650],[15,5,30]);b=np.linalg.inv(ab)@a
    assert np.linalg.norm(a[:3,3]-b[:3,3])>1
    fit=fit_camera_transform([(a,b)]*3)
    assert np.allclose(fit,ab)
    row=compare_tag(a,b,fit)
    assert row['position_error_mm']<1e-8 and row['orientation_error_deg']<1e-8


def test_held_out_error_not_refitted_away():
    ab=transform([200,10,30],[0,10,180])
    pairs=[]
    for x in [0,20,40]:
        a=transform([x,50,600],[15,10,0]);pairs.append((a,np.linalg.inv(ab)@a))
    fixed=fit_camera_transform(pairs)
    a=transform([100,70,700],[20,5,0]);b=np.linalg.inv(ab)@a
    b[:3,3]+=fixed[:3,:3].T@np.array([3,4,0])
    row=compare_tag(a,b,fixed)
    assert row['delta_B_minus_A_mm']==pytest.approx([3,4,0])
    assert summarize([row])['position_rmse_mm']==pytest.approx(5)


def test_invalid_and_empty():
    t=np.eye(4);t[0,0]=-1
    with pytest.raises(ValueError):rigid(t)
    with pytest.raises(ValueError):fit_camera_transform([])
    assert summarize([])['status']=='UNKNOWN'


def test_single_tag_pose_metric_and_corner_convention():
    k=np.array([[900.,0,640],[0,900,360],[0,0,1]])
    h=24.;obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]])
    expected=transform([30,-20,550],[160,20,10])
    rv=cv2.Rodrigues(expected[:3,:3])[0]
    corners=cv2.projectPoints(obj,rv,expected[:3,3],k,None)[0].reshape(4,2)
    d=SimpleNamespace(tag_id=0,hamming=0,decision_margin=100,corners=corners)
    detector=SimpleNamespace(detect=lambda _: [d])
    result=estimate_tag(np.zeros((720,1280),np.uint8),k,detector,0,48)
    assert np.allclose(result['camera_from_tag_mm'],expected,atol=1e-6)
    detector.detect=lambda _: [d,d]
    with pytest.raises(ValueError,match='Expected one'):estimate_tag(None,k,detector,0,48)


def test_wrong_id_is_not_used():
    detector=SimpleNamespace(detect=lambda _: [SimpleNamespace(tag_id=5)])
    with pytest.raises(ValueError,match='saw 0'):estimate_tag(None,np.eye(3),detector,0,48)


def test_capture_flow_uses_disjoint_validation_and_closes_cameras(tmp_path,monkeypatch):
    import importlib.util,json,sys
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('tag_compare_cli',Path(__file__).parents[1]/'scripts/compare_dual_camera_tag.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cameras=[]
    class FakeCamera:
        def __init__(self,serial,*args):
            self.a=serial=='A';self.count=0;self.closed=False;self.k=np.eye(3);self.metadata={'serial':serial};cameras.append(self)
        def read(self):
            self.count+=1
            value=1 if self.a else (2 if self.count<=3 else 3)
            im=np.full((10,10,3),value,np.uint8)
            return im,im,{'host_received_monotonic_s':1.,'frame_number':self.count}
        def close(self):self.closed=True
    def detect(gray,*args):
        value=int(gray[0,0]);t=transform([0 if value==1 else (-100 if value==2 else -95),0,600],[0,0,0])
        return {'camera_from_tag_mm':t.tolist(),'corners_rectified_px':[[1,1],[8,1],[8,8],[1,8]]}
    monkeypatch.setattr(module,'Camera',FakeCamera)
    monkeypatch.setattr(module,'estimate_tag',detect)
    monkeypatch.setitem(sys.modules,'pupil_apriltags',SimpleNamespace(Detector=lambda **kw:None))
    out=tmp_path/'run'
    assert module.main(['--serial-a','A','--serial-b','B','--tag-id','0','--tag-size-mm','48',
                        '--fit-samples','3','--validation-samples','2','--interval','.000001',
                        '--no-preview','--output',str(out)])==0
    result=json.loads((out/'summary.json').read_text())
    assert result['fit_count']==3 and result['validation']['count']==2
    assert result['validation']['position_rmse_mm']==pytest.approx(5)
    assert all(c.closed for c in cameras)
    records=[json.loads(l) for l in (out/'observations.jsonl').read_text().splitlines()]
    assert [r['phase'] for r in records]==['FIT']*3+['VALIDATION']*2
