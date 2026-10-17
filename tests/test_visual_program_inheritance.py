import json
from pathlib import Path
import pytest
from cloth_agent.experiments.visual_program_inheritance import parse,PIDS

def program():return json.loads(Path('reports/visual_program_S1_S2/visual_program.json').read_text())
def values(items):return {p:dict(value=v,confidence=.9) for p,v in zip(PIDS,items)}
def test_zero_shot_exact_only():
 assert parse('S1','zero_shot',program())['prediction']=='S1'
 with pytest.raises(ValueError):parse('Maybe S1 or S2','zero_shot',program())
def test_inheritance_host_s2():
 p=parse(json.dumps(values(['YES','NO','YES','YES'])),'inheritance',program())
 assert p['prediction']=='S2'
def test_unknown_propagates():
 assert parse(json.dumps(values(['YES','YES','UNKNOWN','YES'])),'inheritance',program())['prediction']=='UNKNOWN'
def test_forbid_direct_node_output():
 v=values(['YES','YES','NO','YES']);v['node']='S1'
 with pytest.raises(ValueError):parse(json.dumps(v),'inheritance',program())
def test_no_confidence_threshold():
 v=values(['YES','YES','NO','YES']);v[PIDS[0]]['confidence']=.01
 assert parse(json.dumps(v),'inheritance',program())['prediction']=='S1'
def test_invalid_confidence_rejected():
 v=values(['YES','YES','NO','YES']);v[PIDS[0]]['confidence']=float('nan')
 with pytest.raises(ValueError):parse(json.dumps(v),'inheritance',program())
