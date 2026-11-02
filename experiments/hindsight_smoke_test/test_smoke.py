import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cv_select import run_id,stable,mask
import numpy as np
class SmokeTests(unittest.TestCase):
 def test_run_is_not_episode(self):
  a={'source':'reports/run_a/episode_0001/round_1'};b={'source':'reports/run_a/episode_0002/round_1'}
  self.assertEqual(run_id(a),run_id(b))
 def test_seed_stable(self):self.assertEqual(stable('input'),stable('input'))
 def test_foreground_not_semantic(self):
  a=np.zeros((270,480,3),np.uint8);a[70:200,170:300]=[180,80,160];m,s=mask(a);self.assertGreater(m.sum(),0);self.assertNotIn('endpoint',s)
 def test_hard_budget_stops_before_api(self):
  import tempfile
  from unittest.mock import patch
  from transport import call
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for i in range(20):
    p=root/'calls'/str(i)/'attempt_0';p.mkdir(parents=True);(p/'started.json').write_text('{}')
   with patch('transport.subprocess.Popen') as api:
    with self.assertRaisesRegex(RuntimeError,'budget 20'):call(root,'blocked','instruction',{},[],{'type':'object'})
    api.assert_not_called()
 def test_no_robot_entry(self):
  s=Path(__file__).with_name('run.py').read_text();self.assertNotIn('connect(',s);self.assertNotIn('--execute',s)
if __name__=='__main__':unittest.main()
