"""Evaluator controls, not simulated manipulation outcomes."""
import unittest
import cv2
import numpy as np
from cloth_agent.cloth_harness.cv_harness.outcome_observer import observe
from cloth_agent.cloth_harness.runtime.interfaces import Observation,Action

class FixedEvaluatorControls(unittest.TestCase):
    def image(self):
        rgb=np.zeros((320,320,3),dtype=np.uint8)
        rng=np.random.default_rng(71)
        texture=np.array([180,90,200])+rng.integers(-20,21,(150,130,1))
        rgb[80:230,90:220]=texture.astype(np.uint8)
        return rgb
    def obs(self,rgb,camera='fixed'):
        return Observation(rgb,None,0.,camera,provenance={'joints_deg':{'camera':[0.,0.]}})
    def evaluate(self,before,after,camera='fixed'):
        return observe(self.obs(before),Action((110,110),(120,110),None),self.obs(after,camera),{'status':'EXECUTED','completed_commands':9})[0]
    def test_identical_image_zero_area_not_success(self):
        rgb=self.image();o=self.evaluate(rgb,rgb)
        self.assertEqual(o.coverage['delta_ratio'],0.)
        self.assertEqual(o.coverage['status'],'UNCALIBRATED')
        self.assertEqual(o.execution['grasp_status'],'UNKNOWN')
        self.assertEqual(o.local_change['status'],'UNKNOWN')
    def test_rigid_translation_does_not_increase_area(self):
        before=self.image();after=cv2.warpAffine(before,np.float32([[1,0,12],[0,1,7]]),(320,320))
        o=self.evaluate(before,after)
        self.assertEqual(o.coverage['delta_ratio'],0.)
        self.assertEqual(o.motion['status'],'OK')
        self.assertLess(o.motion['shape_change'],.025)
        self.assertAlmostEqual(o.motion['translation_px'],np.hypot(12,7),delta=1.)
    def test_changed_camera_is_unknown(self):
        o=self.evaluate(self.image(),self.image(),camera='other')
        self.assertIsNone(o.coverage['delta_ratio'])
        self.assertEqual(o.motion['status'],'UNKNOWN')

if __name__=='__main__':unittest.main()
