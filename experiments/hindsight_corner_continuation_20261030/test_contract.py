"""Offline gate and code-boundary checks; no model requests or robot imports."""
import unittest
import run

class ContractTest(unittest.TestCase):
    def setUp(self):
        self.item=dict(id='C00',status='PROVISIONAL',goal='Reveal(RS)',before_state='SUSPECTED_OCCLUDED',after_state='EXPOSED',before_score=2,identity_score=2,transition_score=2,semantic_side_score=2)
    def test_semantic_goal_cannot_drift(self):
        self.assertTrue(run.eligible(self.item))
        for patch in [dict(goal=None),dict(goal='Reveal(seam)'),dict(before_state='VISIBLE'),dict(identity_score=1),dict(after_state='UNKNOWN')]:
            self.assertFalse(run.eligible({**self.item,**patch}))
    def test_ranking_does_not_use_surprise(self):
        a={**self.item,'intention':'reveal sleeve','unexpected':False}
        b={**self.item,'intention':'unrelated','unexpected':True}
        self.assertEqual(run.rank(a),run.rank(b))
        stronger={**self.item,'status':'CONFIRMED',**{k:3 for k in ['before_score','identity_score','transition_score','semantic_side_score']}}
        self.assertLess(run.rank(stronger),run.rank(self.item))
    def test_skill_has_no_file_or_robot_import(self):
        for body in ['import os','import cloth_agent','open("/tmp/x")','eval("1")','import numpy as np\nnp.load("x.npy")']:
            with self.assertRaises(ValueError):run.code_check(body+'\ndef generate_action(rgb, target_corner):\n    return {}\n')
        run.code_check('def generate_action(rgb, target_corner):\n    return VISION_MODEL(rgb, FROZEN_PROMPT)\n')

if __name__=='__main__':unittest.main()
