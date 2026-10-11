import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from cloth_agent.codex_loop import loop
from cloth_agent.codex_motion import parse_code


class FakeRuntime:
    physical=False
    calibration_id='test'
    def __init__(self,status='EXECUTED'):
        self.status=status;self.captures=[];self.executions=[];self.closed=False;self.stopped=False
        self.events=[]
    def observe(self,path):
        path=Path(path);path.mkdir();(path/'rgb.png').write_bytes(b'fake')
        self.captures.append(path);self.events.append('obs')
        return SimpleNamespace(directory=path,image=path/'rgb.png')
    def execute(self,source,before,folder):
        assert (folder/'motion.py').read_text()==source
        self.events.append('execute');self.executions.append((source,before.directory))
        return {'status':self.status,'physical_execution':False}
    def close(self):self.closed=True
    def stop(self):self.stopped=True


def model(runtime,prompts,answers=None):
    def call(prompt,output,**kwargs):
        assert output.name=='planner'
        runtime.events.append('planning');prompts.append(prompt);output.mkdir()
        answer=answers.pop(0) if answers else 'motion.gripper(arm="left",action="open")'
        (output/'answer.txt').write_text(answer)
        return {'status':'AGENT_FINISHED'}
    return call


def test_one_observation_and_one_model_call_per_round_without_memory(tmp_path):
    runtime=FakeRuntime();prompts=[]
    memory=tmp_path/'experience.jsonl';memory.write_text('not even valid JSON: OLD_EXPERIENCE')
    result=loop('fold',runtime,tmp_path/'session',max_rounds=2,memory=memory,model_call=model(runtime,prompts))
    assert result['status']=='ROUND_LIMIT' and result['rounds']==2
    assert runtime.events==['obs','planning','execute']*2
    assert runtime.closed and len(prompts)==2
    assert all('OLD_EXPERIENCE' not in p and 'lesson' not in p and 'next_step' not in p for p in prompts)
    assert memory.read_text()=='not even valid JSON: OLD_EXPERIENCE'
    assert not list((tmp_path/'session').rglob('reflection*'))
    assert not list((tmp_path/'session').rglob('after'))


def test_execution_fault_never_retries_or_calls_model_again(tmp_path):
    runtime=FakeRuntime('EXECUTION_FAILED')
    result=loop('fold',runtime,tmp_path/'session',model_call=model(runtime,[]))
    assert result['status']=='EXECUTION_FAILED'
    assert runtime.events==['obs','planning','execute'] and runtime.closed
    assert not (tmp_path/'session/experience.jsonl').exists()


@pytest.mark.parametrize('status',['PLANNING_FAILED','PLANNING_INCOMPLETE'])
def test_planning_failure_retries_from_new_observation_without_review(tmp_path,status):
    runtime=FakeRuntime(status)
    result=loop('fold',runtime,tmp_path/'session',max_rounds=2,model_call=model(runtime,[]))
    assert result['status']=='ROUND_LIMIT' and len(runtime.executions)==2
    assert runtime.events==['obs','planning','execute']*2


def test_ik_failure_returns_full_feedback_and_reuses_observation_until_execution(tmp_path):
    failure=dict(status='PLANNING_FAILED',failure_kind='IK_FAILED',physical_execution=False,
        error='PyRoki IK rejected target',ik_failure=dict(command_index=3,robot='arm6',phase='rrt_goal_ik',
        target_base_mm_deg=[700,-85,146,180,0,-77],candidate={'distance_mm':2.1,'angle_deg':.8}),
        failure_context={'direct_failure':'full original stopping sweep report'},
        rejected_motion_source='HOST_ADJUSTED_CODE')
    class RetryRuntime(FakeRuntime):
        def execute(self,*args):
            super().execute(*args)
            return failure if len(self.executions)==1 else dict(status='EXECUTED',physical_execution=True)
    runtime=RetryRuntime();prompts=[];attached=[]
    first='motion.move_tcp(left=[.7,-.085,.146,3.14,0,-1.3])'
    second='motion.move_tcp(left=[.7,-.085,.11,3.14,0,-1.3])'
    inner=model(runtime,prompts,[first,second,'motion.complete()'])
    def repair_model(prompt,**kwargs):
        attached.append(kwargs['images'])
        return inner(prompt,**kwargs)
    result=loop('fold',runtime,tmp_path/'run',model_call=repair_model)
    assert result['status']=='MODEL_REPORTED_COMPLETE'
    assert runtime.events==['obs','planning','execute','planning','execute','obs','planning']
    assert runtime.executions[0][1]==runtime.executions[1][1]
    assert first in prompts[1] and 'HOST_ADJUSTED_CODE' in prompts[1]
    assert 'full original stopping sweep report' in prompts[1]
    assert '只修正导致失败的代码' in prompts[1]
    assert '保留未受失败影响的抓点、折叠顺序和动作意图' in prompts[1]
    assert '先查看 RGB' not in prompts[1] and '只规划本轮的一小步' not in prompts[1]
    assert attached[0] and attached[1]==[] and attached[2]
    retry=tmp_path/'run/round_002'
    assert (retry/'before').resolve()==runtime.captures[0].resolve()
    assert json.loads((retry/'replanning_feedback.json').read_text())['outcome']==failure
    assert 'observation_s' not in json.loads((retry/'timing.json').read_text())['seconds']
    assert 'gpt_code_repair_s' in json.loads((retry/'timing.json').read_text())['seconds']
    assert not list((tmp_path/'run').rglob('reflection*'))


def test_failure_details_still_printed(tmp_path,capsys):
    class FailedRuntime(FakeRuntime):
        def execute(self,*args):
            return dict(status='PLANNING_FAILED',error='node budget exhausted',physical_execution=False,
                        failure_context={'command_index':3,'command':'move_tcp'})
    runtime=FailedRuntime()
    loop('fold',runtime,tmp_path/'run',max_rounds=1,model_call=model(runtime,[]))
    printed=capsys.readouterr().out
    assert '失败原因：node budget exhausted' in printed
    assert '"command_index": 3' in printed and 'outcome.json' in printed
    assert '下一步：' not in printed and '反思：' not in printed


def test_ordered_references_reach_only_planning(tmp_path):
    runtime=FakeRuntime();calls=[]
    references=[]
    for i in range(5):
        p=tmp_path/f'goal{i}.png';p.write_bytes(f'reference-{i}'.encode())
        references.append((p,f'step {i}'))
    def call(prompt,output,images,**kwargs):
        calls.append((prompt,list(images)));output.mkdir()
        assert output.name=='planner'
        (output/'answer.txt').write_text('motion.complete()')
        return {'status':'AGENT_FINISHED'}
    result=loop('fold',runtime,tmp_path/'run',model_call=call,references=references)
    assert result['status']=='MODEL_REPORTED_COMPLETE' and len(calls)==1
    assert len(calls[0][1])==6
    assert [p.read_bytes() for p in calls[0][1][1:]]==[p.read_bytes() for p,_ in references]
    assert '自主选择单臂或双臂' in calls[0][0]
    assert '一次移动一臂' in calls[0][0]
    assert runtime.executions==[]
    assert json.loads((tmp_path/'run/round_001/outcome.json').read_text())['physical_execution'] is False


@pytest.mark.parametrize('source,status',[
    ('motion.complete()','MODEL_REPORTED_COMPLETE'),
])
def test_terminal_code_does_not_dispatch(tmp_path,source,status):
    runtime=FakeRuntime()
    result=loop('fold',runtime,tmp_path/'run',model_call=model(runtime,[],[source]))
    assert result['status']==status and result['rounds']==1
    assert runtime.events==['obs','planning'] and runtime.closed


@pytest.mark.parametrize('source',[
    'motion.complete(reason="done")',
    'motion.gripper(arm="left",action="open")\nmotion.complete()',
])
def test_invalid_completion_code_never_executes(tmp_path,source):
    runtime=FakeRuntime()
    result=loop('fold',runtime,tmp_path/'run',model_call=model(runtime,[],[source]))
    assert result['status']=='FAILED'
    assert runtime.executions==[] and runtime.stopped


def test_completion_is_loop_only():
    with pytest.raises(ValueError):parse_code('motion.complete()')
    assert parse_code('motion.complete()',allow_complete=True)==[('complete',{})]


def test_missing_reference_fails_before_observation(tmp_path):
    runtime=FakeRuntime()
    with pytest.raises(ValueError,match='reference image is missing'):
        loop('fold',runtime,tmp_path/'run',references=[(tmp_path/'missing.png','goal')])
    assert runtime.captures==[] and runtime.executions==[]


def test_round_timing_saved_on_success_and_capture_failure(tmp_path):
    runtime=FakeRuntime()
    result=loop('fold',runtime,tmp_path/'success',max_rounds=1,model_call=model(runtime,[]))
    timing=json.loads((tmp_path/'success/round_001/timing.json').read_text())
    assert timing['status']=='EXECUTED'
    for key in ('observation_s','gpt_planning_s','motion_pipeline_s','round_total_s'):
        assert timing['seconds'][key]>=0
    assert result['elapsed_s']>=timing['seconds']['round_total_s']
    class BrokenCamera(FakeRuntime):
        def observe(self,path):raise RuntimeError('No device connected')
    result=loop('fold',BrokenCamera(),tmp_path/'failure',max_rounds=1)
    timing=json.loads((tmp_path/'failure/round_001/timing.json').read_text())
    assert result['status']=='FAILED' and timing['status']=='RuntimeError'
    assert timing['seconds']['observation_s']>=0
    assert 'gpt_planning_s' not in timing['seconds']


def test_default_loop_continues_past_twenty_and_blocked_until_complete(tmp_path):
    runtime=FakeRuntime();prompts=[]
    answers=['motion.blocked(reason="no usable depth")']*21+['motion.complete()']
    result=loop('fold',runtime,tmp_path/'run',model_call=model(runtime,prompts,answers))
    assert result['status']=='MODEL_REPORTED_COMPLETE' and result['rounds']==22
    assert len(runtime.captures)==22 and not runtime.executions
    assert 'no usable depth' in prompts[1]
    assert runtime.closed
