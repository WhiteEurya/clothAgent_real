import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
from tests.test_apriltag_diagnostic import synthetic_scene
from cloth_agent.experiments import apriltag_geometry_audit as audit
from cloth_agent.dual_arm.observation import Observation

class GeometryAuditTests(unittest.TestCase):
    def run_scene(self, bias=0, missing=False):
        board=audit.board_config(audit.DEFAULT_BOARD,audit.A4_SCALE)
        rgb,depth,k,detector,pose=synthetic_scene(board,depth_bias=bias)
        for tag in detector.detect():
            tag.center=audit.project_rgb(pose,[board['corners'][tag.tag_id].mean(axis=0)],k,np.zeros(5))[0]
        yy,xx=np.indices(depth.shape)
        cloud=np.stack(((xx-k[0,2])/k[0,0]*depth,(yy-k[1,2])/k[1,1]*depth,depth),axis=-1)*1000
        if missing: cloud[:]=np.nan
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            Image.fromarray(rgb).save(folder/'rgb.png')
            obs=Observation(folder,{'world_from_camera_mm':np.eye(4).tolist()},cloud)
            result=audit.analyze(obs,k,{'distortion_model':'none','distortion_coefficients':[0]*5},board,detector,10,'center')
            audit.save_outputs(folder,obs,result)
            for name in ('results.csv','summary.md','rgb_residuals.png','depth_visualization.png','xy_error_field.png'):
                self.assertTrue((folder/name).exists())
            return result

    def test_known_geometry(self):
        r=self.run_scene()
        self.assertEqual(r['status'],'MEASURED')
        self.assertEqual(r['accepted_tags'],16)
        self.assertLess(r['regions']['all']['xy_error_mm']['max'],.8)
        self.assertLess(r['regions']['all']['abs_z_error_mm']['max'],.3)

    def test_depth_bias_detected(self):
        r=self.run_scene(.012)
        self.assertAlmostEqual(r['regions']['all']['signed_z_error_mm']['median'],12,delta=.3)

    def test_missing_depth_not_zero_error(self):
        r=self.run_scene(missing=True)
        self.assertEqual(r['status'],'NO_VALID_DEPTH')
        self.assertIsNone(r['regions']['all']['xy_error_mm']['median'])

    def test_unknown_distortion_rejected(self):
        with self.assertRaises(ValueError):
            audit.distortion_coefficients(dict(distortion_model='inverse_brown_conrady',distortion_coefficients=[.1,0,0,0,0]))

    def test_readonly_joint_feedback(self):
        calls=[]
        arm=SimpleNamespace(get_servo_angle=lambda **kw:(calls.append(kw) or (0,[0]*7)))
        config=SimpleNamespace(arms={'right':SimpleNamespace(axis=7)})
        snapshot=audit.readonly_snapshot(arm,config,[1]*6)()
        self.assertEqual(len(calls),1)
        self.assertEqual(snapshot['right']['joints'],[0]*7)

if __name__=='__main__':unittest.main()
