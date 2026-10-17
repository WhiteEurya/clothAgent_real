import unittest
import cv2
import numpy as np
from cloth_agent.experiments.camb_intrinsics_fit import fit,heldout_error
from cloth_agent.experiments.apriltag_geometry_audit import board_config,DEFAULT_BOARD,A4_SCALE

class IntrinsicsFitTests(unittest.TestCase):
    def test_recovers_intrinsics_and_scores_disjoint_tags(self):
        board=board_config(DEFAULT_BOARD,A4_SCALE)
        objects=np.concatenate([board['corners'][i] for i in range(16)])
        k=np.array([[880.,0,641.],[0,890.,365.],[0,0,1.]])
        views=[]
        for i in range(15):
            r=np.array([.4*np.sin(i),.45*np.cos(i),.1*np.sin(i/2)])
            t=np.array([.07*np.sin(i/3),.08*np.cos(i/2),.55+.07*np.sin(i/5)])
            pixels=cv2.projectPoints(objects,r,t,k,np.zeros(5))[0].reshape(-1,2)
            views.append(dict(object=objects,pixels=pixels,ids=list(range(16)),image=str(i)))
        guess=np.array([[908.,0,646.],[0,907.,381.],[0,0,1.]])
        result=fit(views[:12],guess,[1280,720],False)
        np.testing.assert_allclose(result['K'],k,atol=.02)
        held=heldout_error(views[12:],np.asarray(result['K']),np.asarray(result['dist']))
        self.assertLess(held['rms_px'],.001)
        bad=heldout_error(views[12:],guess,np.zeros(5))
        self.assertGreater(bad['rms_px'],held['rms_px']+.1)

if __name__=='__main__':unittest.main()
