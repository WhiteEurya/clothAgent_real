import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import patch
import numpy as np
from cloth_agent.experiments import registration_field as rf
from cloth_agent.experiments.apriltag_geometry_audit import project_rgb
from cloth_agent.experiments.apriltag_multiboard_audit import measured_board
from cloth_agent.experiments.registration_field_report import metric,generate

def scene():
    k=np.array([[800.,0,640.],[0,800.,360.],[0,0,1.]])
    profiles={name:dict(K=k.tolist(),coeffs=[0.]*5,width=1280,height=720) for name in ('rgb','depth','ir1','ir2')}
    extrinsics={}
    transforms={}
    for name,x in (('rgb',.015),('depth',0),('ir1',0),('ir2',-.05)):
        t=np.eye(4);t[0,3]=x;transforms[name]=t
        extrinsics['depth_to_'+name]=dict(rotation_column_major=np.eye(3).flatten(order='F').tolist(),translation_m=[x,0,0])
    meta=dict(profiles=profiles,depth_scale=.001,extrinsics=extrinsics)
    board,_=measured_board(142);tags={l:{} for l in ('rgb','ir1','ir2')}
    for i,obj in board['corners'].items():
        points=obj+np.array([0,0,.6])
        for label in tags:tags[label][i]=project_rgb(transforms[label],points,k,np.zeros(5))
    depth=np.full((3,720,1280),600,np.uint16)
    return meta,tags,depth

class RegistrationFieldTests(unittest.TestCase):
    def test_both_methods_preserve_measured_residual(self):
        m,t,d=scene()
        t['rgb']={i:p+np.array([3.,-2.]) for i,p in t['rgb'].items()}
        rows=rf.calculate_corners(m,t,d,'C-M',1,1)
        self.assertEqual(len(rows),64)
        for r in rows:
            for method in ('direct','stereo'):
                self.assertTrue(r[method+'_valid'])
                self.assertAlmostEqual(r[method+'_du'],-3,places=6)
                self.assertAlmostEqual(r[method+'_dv'],2,places=6)
    def test_motion_uses_two_dimensional_norm_and_missing_rejected(self):
        a={i:np.zeros((4,2)) for i in range(4)}
        b={i:np.ones((4,2))*.8 for i in range(4)}
        self.assertGreater(rf.motion_rms(a,b),1)
        self.assertIsNone(rf.motion_rms(a,{}))
    def test_depth_missing_and_border_do_not_wrap(self):
        m,t,d=scene();d[:]=0
        rows=rf.calculate_corners(m,t,d,'C-M',1,1)
        self.assertFalse(any(r['direct_valid'] for r in rows))
        self.assertTrue(all(r['stereo_valid'] for r in rows))
        t['ir1'][0][0]=[-1,-1]
        rows=rf.calculate_corners(m,t,d,'C-M',1,1)
        self.assertEqual(rows[0]['direct_reason'],'DEPTH_OUT_OF_BOUNDS')
    def test_duplicate_ids_rejected(self):
        tag=SimpleNamespace(tag_id=0,hamming=0,decision_margin=100,corners=np.zeros((4,2)))
        detector=SimpleNamespace(detect=lambda *a,**kw:[tag,tag])
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):rf.detect(detector,np.zeros((10,10),np.uint8))
    def test_motion_rejected_not_counted_and_next_repeat_accepted(self):
        m,t,d=scene();rows=rf.calculate_corners(m,t,d,'C-M',1,1)
        g=dict(center=[640,396],normal=[0,0,1],inplane_deg=0,rgb_tags=16)
        rejected=rf.finalize_repeat(rows,[.2,1.01],g,None,m,'C-M')
        self.assertFalse(rejected['accepted']);self.assertTrue(all(r['repeat_status']=='REJECTED' for r in rows))
        again=copy.deepcopy(rows)
        accepted=rf.finalize_repeat(again,[.2,.3],g,None,m,'C-M')
        self.assertTrue(accepted['accepted'])
        self.assertEqual(metric(rows,'direct')['corners'],0)
        self.assertEqual(metric(again,'direct')['corners'],64)
    def test_position_and_orientation_gates(self):
        m,t,d=scene();rows=rf.calculate_corners(m,t,d,'C-M',1,1)
        g=dict(center=[640,396],normal=[0,0,1],inplane_deg=0,rgb_tags=16)
        ref=dict(g,inplane_deg=20)
        self.assertIn('BOARD_ORIENTATION_CHANGED',rf.finalize_repeat(rows,[0],g,ref,m,'C-M')['reasons'])
        self.assertIn('OUTSIDE_PLACEMENT_TARGET',rf.finalize_repeat(rows,[0],g,None,m,'L-M')['reasons'])
    def test_unsupported_calibration_rejected(self):
        m,_,_=scene();m['profiles']['rgb']['coeffs'][0]=.1
        with self.assertRaises(ValueError):rf.validate_metadata(m)
    def test_capture_restores_emitter_on_failure(self):
        class Camera:
            def __init__(self):self.states=[];self.restored=False
            def emitter(self,v):self.states.append(v)
            def read(self):raise RuntimeError('disconnected')
            def restore_emitter(self):self.restored=True;return dict(verified=True)
        camera=Camera()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                rf.capture_pair(camera,Path(tmp),threading.Event(),lambda x:None,lambda x:None)
            self.assertTrue(camera.restored)
            self.assertTrue((Path(tmp)/'emitter_restoration.json').exists())
    def test_cancel_restores_emitter(self):
        class Camera:
            restored=False
            def emitter(self,v):pass
            def restore_emitter(self):self.restored=True;return dict(verified=True)
        camera=Camera();event=threading.Event();event.set()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InterruptedError):rf.capture_pair(camera,Path(tmp),event,lambda x:None,lambda x:None)
            self.assertTrue(camera.restored)
    def test_session_retries_without_advancing_repeat_then_completes(self):
        meta,_,_=scene()
        class UI:
            def __init__(self):
                self.cancel=threading.Event();self.ready=threading.Event();self.waiting=False
                self.placement='L-M';self.session='';self.ended=None
            def update(self,*args):
                if self.waiting:self.ready.set()
            def preview(self,*args):pass
            def finished(self,*args):self.ended=args
        class Camera:
            closed=False
            def __init__(self,*args):pass
            def start(self):return meta
            def close(self):Camera.closed=True;return dict(verified=True)
        ui=UI()
        evidence={key:np.zeros((10,10),np.uint8) for key in ('on_first','on_last','off_first','off_last','post_restore','ir1','ir2')}
        def corners(m,t,d,placement,repeat,attempt):
            return [dict(placement=placement,repeat=repeat,attempt=attempt,tag_id=i//4,corner_id=i%4,
                rgb_u=600+i,rgb_v=400,stereo_valid=True,direct_valid=True,stereo_du=1.,stereo_dv=2.,
                direct_du=1.,direct_dv=2.,stereo_error_px=5**.5,direct_error_px=5**.5) for i in range(24)]
        def geometry(*args):
            x,y=rf.TARGETS[ui.placement]
            return dict(center=[x*1280,y*720],normal=[0,0,1],inplane_deg=0,rgb_tags=16)
        with tempfile.TemporaryDirectory() as tmp, patch.object(rf,'NativeCamera',Camera),              patch.object(rf,'capture_pair',return_value=(evidence,np.zeros((3,10,10)))),              patch.object(rf,'detect',return_value={}), patch.object(rf,'calculate_corners',side_effect=corners),              patch.object(rf,'board_geometry',side_effect=geometry),              patch.object(rf,'motion_rms',side_effect=[1.1]+[0.]*27),              patch('pupil_apriltags.Detector',return_value=None), patch.object(rf.fcntl,'flock'):
            args=SimpleNamespace(output=Path(tmp),repeats=2,max_attempts=4,config=Path('unused'))
            session=rf.run_session(args,ui)
            manifest=json.loads((session/'session.json').read_text())
            self.assertEqual(manifest['status'],'COMPLETE')
            self.assertTrue(Camera.closed)
            left=[a for a in manifest['attempts'] if a['placement']=='L-M']
            self.assertEqual([a['repeat'] for a in left],[1,1,2])
            self.assertEqual([a['accepted'] for a in left],[False,True,True])
            self.assertEqual(len(manifest['attempts']),7)

    def test_partial_report_excludes_rejected_and_is_complete(self):
        m,t,d=scene();rows=rf.calculate_corners(m,t,d,'C-M',1,1)
        g=dict(center=[640,396],normal=[0,0,1],inplane_deg=0,rgb_tags=16)
        result=rf.finalize_repeat(rows,[.1],g,None,m,'C-M')
        bad=copy.deepcopy(rows)
        for r in bad:r.update(repeat_status='REJECTED',direct_error_px=1000,stereo_error_px=1000)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            rf.save(p/'session.json',dict(status='INTERRUPTED',attempts=[dict(placement='C-M',**result)]))
            rf.save(p/'camera.json',m);rf.append_rows(p/'all_samples.csv',rows+bad)
            summary=generate(p)
            self.assertEqual(summary['C-M']['valid_repeats'],1)
            self.assertLess(summary['C-M']['methods']['direct']['p95_error_px'],.001)
            for name in ('summary.md','summary.json','residual_field.png','median_vectors.png','repeat_summary.csv'):
                self.assertTrue((p/name).exists())
if __name__=='__main__':unittest.main()
