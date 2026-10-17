import math,json
from pathlib import Path
import pytest
from cloth_agent.experiments.qwen_predicate_diagnosis import angle,parse,SIMPLE,PIDS
@pytest.mark.parametrize('c,h',[([0,0],[0,10]),([0,10],[0,0]),([0,0],[10,0]),([0,0],[-10,4]),([684,611],[680,60])])
def test_rotation_axis_points_down_without_mirror(c,h):
 a=math.radians(angle(c,h));dx=h[0]-c[0];dy=h[1]-c[1]
 assert abs(math.cos(a)*dx+math.sin(a)*dy)<1e-8
 assert -math.sin(a)*dx+math.cos(a)*dy>0
 assert math.cos(a)**2+math.sin(a)**2==pytest.approx(1)
def test_degenerate_orientation_rejected():
 with pytest.raises(ValueError):angle([0,0],[0,0])
def test_atomic_json_only():
 assert parse('{"value":"YES","confidence":0.95}')['status']=='OK'
 assert parse('{"value":"UNKNOWN","confidence":0.0}')['value']=='UNKNOWN'
 assert parse('{"value":"NO","confidence":0.9,"reason":"hidden"}')['status']=='PARSE_ERROR'
def test_errors_never_unknown():
 r=parse('YES');assert r['value']=='PARSE_ERROR'
 assert parse('{"value":"NO","confidence":1.2}')['value']=='PARSE_ERROR'
def test_frozen_original_program_not_reworded():
 program=json.loads(Path('reports/visual_program_S1_S2/visual_program.json').read_text())
 original={p['id']:p['question'] for p in program['predicates']}
 assert all(original[p]!=SIMPLE[p] for p in PIDS)
