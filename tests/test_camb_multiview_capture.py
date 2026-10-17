import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from cloth_agent.experiments import camb_multiview_capture as m

class Widget:
    def __init__(self,*a,**kw):self.callbacks={};self.started=False
    def pack(self,*a,**kw):pass
    def title(self,*a):pass
    def config(self,*a,**kw):pass
    def bind(self,key,callback):self.callbacks[key]=callback
    def protocol(self,*a):pass
    def update_idletasks(self):pass
    def update(self):
        if not self.started:self.started=True;self.callbacks['<Return>']()
    def destroy(self):pass

class Camera:
    def __init__(self,fail=False):self.closed=False;self.fail=fail;self.emitter_values=[]
    def start(self):return dict(robot_connected=False)
    def emitter(self,v):self.emitter_values.append(v)
    def read(self):
        if self.fail:raise RuntimeError('simulated camera failure')
        return dict(rgb=np.zeros((24,32,3),np.uint8),ir1=np.zeros((24,32),np.uint8),ir2=np.zeros((24,32),np.uint8),timestamps=dict(rgb=1,ir1=1,ir2=1))
    def close(self):self.closed=True;return dict(verified=True)

class CaptureTests(unittest.TestCase):
    def trial(self,fail):
        with tempfile.TemporaryDirectory() as d:
            camera=Camera(fail);counter=iter(np.arange(0,100,.5))
            args=SimpleNamespace(output=Path(d),config=Path('unused'),duration=2.,interval=.5,board_span_mm=142.)
            with patch.object(m,'NativeCamera',return_value=camera),patch.multiple(m.tk,Tk=Widget,Label=Widget,Button=Widget),patch.object(m.ImageTk,'PhotoImage',return_value=None),patch.object(m.fcntl,'flock'),patch.object(m.time,'monotonic',side_effect=lambda:float(next(counter))):
                result=m.run(args)
            p=next(Path(d).iterdir());s=json.loads((p/'session.json').read_text())
            self.assertTrue(camera.closed);self.assertEqual(camera.emitter_values,[0])
            self.assertTrue((p/'restoration.json').exists())
            if fail:self.assertEqual(result,1);self.assertEqual(s['status'],'FAILED')
            else:
                self.assertEqual(result,0);self.assertGreater(len(s['frames']),0)
                self.assertTrue((p/'frame_0000/ir2.png').exists());self.assertFalse(s['calibration_written'])
    def test_success_saves_synchronized_triplets_and_restores(self):self.trial(False)
    def test_failure_restores(self):self.trial(True)
    def test_guide_bounds(self):self.assertEqual(m.guide(90,90),m.GUIDES[-1])

if __name__=='__main__':unittest.main()
