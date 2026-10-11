import json
from pathlib import Path
import pytest
from cloth_agent.codex_loop import loop
from cloth_agent.codex_iteration_memory import FIELDS
from tests.test_codex_loop import FakeRuntime


class Arm7Fake(FakeRuntime):
    active_arms=('right',)
    planning_policy='只使用 arm7（right）单臂整理并折叠衣服，按顺序参考图完成。'
    from cloth_agent.codex_grasp_task import TASK_CONTRACT
    motion_contract=TASK_CONTRACT.replace('10 mm','15 mm')


def reference(tmp_path):
    p=tmp_path/'initial.png';p.write_bytes(b'initial demo');return [(p,'initial demo')]


def answer(output, code, state='辨认衣领，动作效果待下一轮验证'):
    output.mkdir()
    memory={k:state for k in FIELDS}
    (output/'answer.txt').write_text(json.dumps(dict(code=code,memory=memory),ensure_ascii=False))
    return {'status':'AGENT_FINISHED'}


def test_unfold_memory_and_comparison_reach_next_iteration(tmp_path):
    runtime=Arm7Fake();seen=[]
    code='motion.grasp(right=[.5,0,.05,3.14,0,0])\nmotion.move_tcp(right=[.5,.1,.1,3.14,0,0])'
    def call(prompt,output,images,**kwargs):
        assert kwargs['read_only'] is True
        seen.append((prompt,images))
        return answer(output,code if len(seen)==1 else 'motion.complete()')
    result=loop('unfold',runtime,tmp_path/'run',unfold=True,remember_iterations=True,
                references=reference(tmp_path),model_call=call)
    assert result['status']=='MODEL_REPORTED_COMPLETE'
    assert len(runtime.executions)==1 and runtime.executions[0][0]==code
    assert len(seen[0][1])==2 and len(seen[1][1])==3
    assert seen[1][1][-1]==seen[0][1][0]
    assert '上一轮动作前的 RGB' in seen[1][0]
    assert '辨认衣领' in seen[1][0] and 'EXECUTED' in seen[1][0]
    for prompt,_ in seen:
        assert '按顺序参考图完成' not in prompt
        assert '衣物已整齐叠好' not in prompt and '已叠好时' not in prompt
        assert '恢复参考图的展开状态' in prompt
    history=[json.loads(l) for l in (tmp_path/'run/iteration_history.jsonl').read_text().splitlines()]
    assert len(history)==2 and history[-1]['outcome']['status']=='MODEL_REPORTED_COMPLETE'
    assert json.loads((tmp_path/'run/round_002/memory_input.json').read_text())==history[0]
    assert json.loads((tmp_path/'run/iteration_memory.json').read_text())==history[-1]


@pytest.mark.parametrize('bad',[
    'motion.complete()',
    json.dumps({'code':'motion.complete()','memory':{}}),
    json.dumps({'code':'import os','memory':{k:'x' for k in FIELDS}}),
    json.dumps({'code':'motion.complete()','memory':{k:'x'*3000 for k in FIELDS}}),
])
def test_invalid_memory_or_motion_never_dispatches(tmp_path,bad):
    runtime=Arm7Fake()
    def call(prompt,output,**kwargs):
        output.mkdir();(output/'answer.txt').write_text(bad);return {'status':'AGENT_FINISHED'}
    result=loop('unfold',runtime,tmp_path/'run',unfold=True,remember_iterations=True,
                references=reference(tmp_path),model_call=call)
    assert result['status']=='FAILED' and not runtime.executions
    assert not (tmp_path/'run/iteration_memory.json').exists()


def test_memory_survives_ik_repair_without_fabricating_new_observation(tmp_path):
    class Retry(Arm7Fake):
        def execute(self,*args):
            super().execute(*args)
            return dict(status='PLANNING_FAILED',physical_execution=False,failure_kind='IK_FAILED',
                        ik_failure={'phase':'grasp_goal_ik','command_index':1,'robot':'arm7'})
    runtime=Retry();seen=[]
    def call(prompt,output,images,**kwargs):
        seen.append((prompt,images));return answer(output,
            'motion.grasp(right=[.5,0,.05,3.14,0,0])\nmotion.move_tcp(right=[.5,.1,.1,3.14,0,0])' if len(seen)==1 else 'motion.blocked(reason="IK unavailable")')
    result=loop('unfold',runtime,tmp_path/'run',unfold=True,remember_iterations=True,max_rounds=2,
                references=reference(tmp_path),model_call=call)
    assert result['status']=='ROUND_LIMIT' and len(runtime.captures)==1
    assert seen[1][1]==[]
    assert '辨认衣领' in seen[1][0] and '未执行' in seen[1][0]
    assert '展开顺序' in seen[1][0] and '折叠顺序' not in seen[1][0]


def test_arm7_unfold_starts_at_15mm_and_folding_keeps_offset(tmp_path):
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.codex_motion import parse_code
    class HeightRuntime(Arm7Fake):
        grasp_offset_mm=Arm7Runtime.grasp_offset_mm
        def execute(self,source,before,folder):
            effective=Arm7Runtime.adjust_motion(self,source,folder)
            self.grasp_z=parse_code(effective,allow_grasp=True)[0][1]['right'][2]
            return super().execute(source,before,folder)
    code='motion.grasp(right=[.5,0,.05,3.14,0,0])\nmotion.move_tcp(right=[.5,.1,.1,3.14,0,0])'
    for unfold,z in [(True,.035),(False,.035)]:
        runtime=HeightRuntime();prompts=[]
        def call(prompt,output,**kwargs):
            prompts.append(prompt);output.mkdir();(output/'answer.txt').write_text(code)
            return {'status':'AGENT_FINISHED'}
        dest=tmp_path/str(unfold)
        result=loop('test',runtime,dest,unfold=unfold,max_rounds=1,
                    references=reference(tmp_path),model_call=call)
        assert result['status']=='ROUND_LIMIT'
        assert runtime.grasp_z==pytest.approx(z)
        if unfold:
            assert '本轮宿主额外下降 15 mm' in prompts[0]
            assert '额外降低 15 mm' in prompts[0]
            policy=json.loads((dest/'round_001/grasp_height_policy.json').read_text())
            assert policy['offset_mm']==15
