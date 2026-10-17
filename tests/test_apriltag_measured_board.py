import unittest
import numpy as np
from cloth_agent.experiments.apriltag_multiboard_audit import measured_board

class MeasuredBoardTests(unittest.TestCase):
    def test_black_outer_span_includes_three_gaps(self):
        board,meta=measured_board(142.)
        points=np.concatenate(list(board['corners'].values()))
        self.assertAlmostEqual(np.ptp(points[:,0])*1000,142.)
        self.assertAlmostEqual(np.ptp(points[:,1])*1000,142.)
        self.assertAlmostEqual(3*board['pitch_mm']+board['tag_size_mm'],142.)
        self.assertAlmostEqual(board['tag_size_mm'],33.326530612244895)
    def test_invalid_size_rejected(self):
        for v in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):measured_board(v)

if __name__=='__main__':unittest.main()
