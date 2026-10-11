import json
import pytest
from cloth_agent.codex_loop import loop
from tests.test_codex_unfold_memory import Arm7Fake, reference


def telemetry(folder,state='stop',position=5,**extra):
    feedback=dict(state=state,position_pulse=position,mechanical_grasp_detected=state=='grasp',
                  usable_for_completion=True,read_errors=[],error_code=0)
    feedback.update(extra)
    (folder/'gripper_003.json').write_text(json.dumps(dict(arm='right',action='close',feedback=dict(
        command_result=0,feedback=feedback,completion=dict(status='COMPLETED',target_position_pulse=5,position_tolerance_pulse=5,before_command={'position_pulse':840})))))


@pytest.mark.parametrize('remember',[False,True])
def test_unfold_empty_feedback_does_not_retry_or_deepen(tmp_path,remember):
    from cloth_agent.codex_arm7_runtime import Arm7Runtime
    from cloth_agent.codex_motion import parse_code
    class Runtime(Arm7Fake):
        def __init__(self):super().__init__();self.heights=[]
        def execute(self,source,before,folder):
            effective=Arm7Runtime.adjust_motion(self,source,folder)
            self.heights.append(parse_code(effective,allow_grasp=True)[0][1]['right'][2])
            telemetry(folder,'stop',8)
            return dict(status='EXECUTED',physical_execution=True)
    runtime=Runtime();prompts=[]
    def call(prompt,output,**kwargs):
        prompts.append(prompt);output.mkdir()
        code='motion.grasp(right=[.5,0,.05,3.14,0,0])\nmotion.move_tcp(right=[.5,.1,.1,3.14,0,0])'
        if remember:
            from cloth_agent.codex_iteration_memory import FIELDS
            code=json.dumps(dict(code=code,memory={k:'preserved memory' for k in FIELDS}))
        (output/'answer.txt').write_text(code)
        return dict(status='AGENT_FINISHED')
    result=loop('unfold',runtime,tmp_path/'run',unfold=True,remember_iterations=remember,
                max_rounds=3,references=reference(tmp_path),model_call=call)
    assert result['status']=='ROUND_LIMIT'
    assert runtime.heights==pytest.approx([.035]*3)
    assert len(prompts)==3
    assert not list((tmp_path/'run').glob('round_*/grasp_retry*.json'))


@pytest.mark.parametrize('state,position,extra,expected',[
    ('stop',5,{},25),('grasp',30,{},20),('grasp',836,{},20),
    ('stop',20,{},20),('stop',5,{'read_errors':['failed']},20),
    ('stop',5,{'mechanical_grasp_detected':True},20),
])
def test_feedback_does_not_deepen_ambiguous_grasps(tmp_path,state,position,extra,expected):
    from cloth_agent.unfold_grasp_retry import retry_policy
    telemetry(tmp_path,state,position,**extra)
    result=retry_policy(tmp_path,'right',dict(status='EXECUTED',physical_execution=True),20)
    assert result['next_offset_mm']==expected


@pytest.mark.parametrize('status',['PLANNING_FAILED','BLOCKED','EXECUTION_FAILED'])
def test_no_completed_grasp_keeps_offset(tmp_path,status):
    from cloth_agent.unfold_grasp_retry import retry_policy
    telemetry(tmp_path)
    result=retry_policy(tmp_path,'right',dict(status=status,physical_execution=False),10)
    assert result['next_offset_mm']==10


@pytest.mark.parametrize('state,position,expected',[
    ('stop',5,'EXECUTED'),('grasp',30,'EXECUTED'),('stop',20,'EXECUTED')])
def test_dispatch_continues_after_completed_close(tmp_path,monkeypatch,state,position,expected):
    from types import SimpleNamespace as NS
    from cloth_agent.codex_native_runtime import NativeRuntime
    import cloth_agent.codex_native_runtime as native
    telemetry(tmp_path,state,position)
    response=json.loads((tmp_path/'gripper_003.json').read_text())['feedback']
    (tmp_path/'gripper_003.json').unlink()
    actions=[]
    def gripper(action):
        actions.append(action)
        return response
    runtime=NativeRuntime.__new__(NativeRuntime)
    runtime.prepared=True
    runtime.grasp_offset_mm=20
    runtime.arrival=lambda target:None
    runtime.stop=lambda: {'stopped':True}
    runtime.config=NS(arms={'right':NS(gripper=NS(gripper_speed=100,gripper_completion_timeout_s=20))})
    runtime.coordinator=NS(connections={'right':NS(gripper=gripper)},event=lambda *a,**kw:None)
    monkeypatch.setattr(native,'wait_operation',lambda c,command,*args:command())
    operations=[('gripper',dict(arm='right',action=a,joints={'right':[0]*7})) for a in ('close','open')]
    result=runtime._execute_operations(operations,tmp_path)
    assert result['status']==expected
    assert actions==(['close','open'] if expected=='EXECUTED' else ['close'])
    assert result['completed_commands']==(2 if expected=='EXECUTED' else 1 if expected=='GRASP_EMPTY' else 0)


