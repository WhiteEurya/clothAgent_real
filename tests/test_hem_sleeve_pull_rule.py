import json
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from cloth_agent import hem_sleeve_pull_rule as rule
from cloth_agent.cloth_harness.runtime import live


def test_disabled_leaves_prompt_exactly_unchanged(tmp_path):
    assert rule.apply(tmp_path,'original')=='original'
    (tmp_path/rule.CONFIG).write_text(json.dumps({'enabled':False}))
    assert rule.apply(tmp_path,'original')=='original'


def test_rule_is_per_episode_and_uses_only_friction_trigger(tmp_path):
    (tmp_path/rule.CONFIG).write_text(json.dumps({'enabled':True}))
    value=rule.apply(tmp_path,'original')
    assert value.startswith('original')
    assert '展开下摆' in value
    assert '如果目前的袖子可以被摩擦力拉开,则抓对向' in value
    for phrase in ['叠在衣身上层','类似衣袖','分支','代理点']:
        assert phrase not in value
    assert rule.apply(tmp_path/'other','original')=='original'


def test_live_applies_rule_without_schema_or_call_count_change(tmp_path):
    (tmp_path/'goal').mkdir();Image.new('RGB',(100,100)).save(tmp_path/'goal/target.png')
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    (tmp_path/rule.CONFIG).write_text(json.dumps({'enabled':True}));calls=[]
    folder=tmp_path/'round';folder.mkdir()
    def direct(f,body,headers,timeout):
        calls.append(body)
        return {'decision':'ABSTAIN','grasp_px':None,'target_px':None,'grasp_angle_deg':None,'motion_type':None,'is_smoothing':False,'action_purpose':None,'expected_stop_state':None,'short_reason':'unknown'}, {'wall_latency_s':1}
    live.plan_round(tmp_path,folder,SimpleNamespace(image=image),{},10,direct)
    assert len(calls)==1 and rule.RULE in calls[0]['instructions']
    assert calls[0]['text']['format']['schema']==live.SCHEMA


def test_enabled_removes_legacy_opposite_triggers(tmp_path):
    from cloth_agent.restricted_context_one_shot import PROMPT
    (tmp_path/rule.CONFIG).write_text(json.dumps({'enabled':True,'rgb_only':True}))
    result=rule.apply(tmp_path,PROMPT)
    assert '若目标袖子或下摆角不可见' not in result
    assert '通过拉已展开的对侧间接展开' not in result
    assert '叠在衣身上层' not in result
    assert result.count('如果目前的袖子可以被摩擦力拉开,则抓对向')==1
    assert 'action_purpose' in result and '只能抓当前图中可见的衣袖或者下摆' in result
