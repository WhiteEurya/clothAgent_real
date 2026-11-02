import unittest,tempfile,json
from pathlib import Path
from PIL import Image
from mining import scan
from run import rank
class PipelineTests(unittest.TestCase):
 def test_executed_and_duplicates_unknown_intention(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);out=r/'output'
   for name,physical in [('a',True),('copy',True),('unexecuted',False)]:
    f=r/'reports'/name;f.mkdir(parents=True)
    for folder,color in [('before','red'),('after','blue')]:
     (f/folder).mkdir();Image.new('RGB',(20,20),color).save(f/folder/'rgb.png')
    (f/'native_plan.json').write_text(json.dumps([{'kind':'gripper','data':{'action':'close'}}]))
    (f/'execution.json').write_text(json.dumps({'status':'EXECUTED','physical_execution':physical,'completed_commands':1}))
    (f/'states.jsonl').write_text('{}\n')
   rows=scan(r,out);self.assertEqual(len(rows),1);self.assertEqual(rows[0]['intent'],{'status':'UNKNOWN'});self.assertEqual(len(rows[0]['duplicate_sources']),1)
 def test_ranking_not_case_identity(self):
  def c(i,label,mismatch):return {'id':i,'effect':{'classification':label,'confidence':'HIGH'},'comparison':{'mismatch':mismatch}}
  a=c('Xzzz','CONFIRMED_SEMANTIC_REVEAL','UNPLANNED');b=c('Xaaa','SUSPECTED_REVEAL','UNPLANNED')
  self.assertLess(rank(a),rank(b))
 def test_prediction_absent_observer(self):
  from contracts import OBS_PROMPT
  self.assertNotIn('expected_change',OBS_PROMPT)
  code=Path(__file__).with_name('validate.py').read_text();callsite=code.split("observed=call(root,'validation_blind_observer'",1)[1].split("record['blind_observed_effect']",1)[0]
  self.assertNotIn("'prediction'",callsite);self.assertNotIn("'skill'",callsite)
if __name__=='__main__':unittest.main()
