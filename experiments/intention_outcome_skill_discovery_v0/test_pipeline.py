import unittest
from run import event_eligible,rank,check_corners
from contracts import GOALS
from policy import checked_compile
class Test(unittest.TestCase):
    def setUp(self):
        self.c={'goal':'Reveal(RS)','classification':'PROVISIONAL','before_state':'SUSPECTED_OCCLUDED','after_state':'EXPOSED','evidence_score':2}
        self.m={'goal':'Reveal(RS)','classification':'UNEXPECTED_OBSERVED','unexpected_evidence':'PROVISIONAL','benefit_related_to_goal':True,'absence_of_mention_only':False}
    def test_only_unexpected_goal_benefit_passes(self):
        self.assertTrue(event_eligible(self.c,self.m))
        for patch in [{'classification':'EXPECTED_OBSERVED'},{'classification':'INDETERMINATE'},{'absence_of_mention_only':True},{'benefit_related_to_goal':False},{'goal':'Reveal(LS)'}]:
            self.assertFalse(event_eligible(self.c,{**self.m,**patch}))
        for patch in [{'before_state':'VISIBLE'},{'classification':'NO_REVEAL'},{'evidence_score':1}]:
            self.assertFalse(event_eligible({**self.c,**patch},self.m))
    def test_four_endpoint_coverage(self):
        check_corners([{'goal':g} for g in GOALS])
        with self.assertRaises(AssertionError):check_corners([{'goal':'Reveal(RS)'}]*4)
    def test_generated_skill_boundaries(self):
        checked_compile('def generate_action(rgb,target_corner):\n return VISION_MODEL(rgb,FROZEN_PROMPT)')
        with self.assertRaises(ValueError):checked_compile('import os\ndef generate_action(rgb,target_corner):\n return {}')
if __name__=='__main__':unittest.main()
