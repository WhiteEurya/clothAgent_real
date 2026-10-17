import csv,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from cloth_agent.experiments.registration_height_audit import compare,check_camera,HeightUI

class HeightAuditTests(unittest.TestCase):
    def session(self,p,z,du,rejected=False):
        p.mkdir();(p/'session.json').write_text(json.dumps(dict(status='COMPLETE')))
        (p/'camera.json').write_text(json.dumps(dict(serial='test',profiles={},extrinsics={},depth_scale=.001,depth_capture_options={})))
        rows=[]
        for place in ('L-M','C-M','R-M'):
            for repeat in (1,2):
                rows.append(dict(placement=place,repeat=repeat,repeat_status='VALID',rgb_u=300,rgb_v=350,native_depth_z_mm=z,direct_valid=True,stereo_valid=True,direct_du=du,direct_dv=1,stereo_du=du,stereo_dv=1,direct_error_px=2,stereo_error_px=2))
        if rejected:rows.append(dict(rows[0],repeat_status='REJECTED',direct_du=999,stereo_du=999))
        with (p/'all_samples.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def test_comparison_measured_height_and_rejection_exclusion(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.session(p/'a',600,1);self.session(p/'b',552,2,True)
            result=compare(p/'a',p/'b',p/'out',50)
            self.assertEqual(len(result['common_tiles']),2)
            for tile in result['common_tiles']:
                self.assertEqual(tile['delta_z_mm'],-48)
                self.assertEqual(tile['delta_du'],1)
            self.assertTrue((p/'out/height_comparison.png').exists())
    def test_changed_calibration_blocked(self):
        with self.assertRaisesRegex(ValueError,'profiles'):check_camera(dict(profiles={'K':1}),dict(profiles={'K':2}))
    def test_proxy_defers_finish_and_forwards_control_state(self):
        calls=[];ui=SimpleNamespace(waiting=False,update=lambda *a:calls.append(a))
        proxy=HeightUI(ui,50);proxy.waiting=True;proxy.update('请将板摆好',0,10);proxy.finished('COMPLETE','path','')
        self.assertTrue(ui.waiting);self.assertIn('稳固垫块',calls[0][0]);self.assertEqual(proxy.end,('COMPLETE','path',''))

if __name__=='__main__':unittest.main()
