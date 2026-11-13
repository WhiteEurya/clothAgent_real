import json
from pathlib import Path
import pytest
from cloth_agent.experiments.unfold_batch import run_batch


@pytest.mark.parametrize('save_goal', [True, False])
def test_batch_captures_goal_once_and_reuses_it(tmp_path, save_goal):
    calls = []
    def execute(cmd):
        calls.append(cmd)
        output = Path(cmd[-1]); output.mkdir()
        if len(calls) == 1 and save_goal:
            (output/'goal').mkdir()
            (output/'goal/target.png').write_bytes(b'original goal')
        (output/'episode_summary.json').write_text(json.dumps({'status':'MODEL_REPORTED_DONE'}))
        return 0
    root = tmp_path/'batch'
    code = run_batch(root, 6, ['--execute', '--capture-goal', '--goal-image=stale.png'], execute)
    assert code == (0 if save_goal else 1)
    assert len(calls) == (6 if save_goal else 1)
    assert '--capture-goal' in calls[0]
    for cmd in calls[1:]:
        assert '--capture-goal' not in cmd
        assert '--goal-image=stale.png' not in cmd
        assert Path(cmd[cmd.index('--goal-image')+1]) == Path(calls[0][-1])/'goal/target.png'


@pytest.mark.parametrize('stop', [None,'RESET_FAILED','EXECUTION_FAILED','PLANNER_UNAVAILABLE','INTERRUPTED','ERROR'])
def test_batch_stops_only_on_physical_or_runtime_failure(tmp_path,stop):
    calls=[]
    def execute(cmd):
        calls.append(cmd)
        output=Path(cmd[-1]);output.mkdir()
        status=stop if stop and len(calls)==2 else ('MODEL_REPORTED_DONE' if len(calls)%2 else 'ACTION_LIMIT')
        (output/'episode_summary.json').write_text(json.dumps(dict(status=status,physical_actions=3)))
        return 0
    root=tmp_path/'batch'
    code=run_batch(root,6,['--execute','--goal-image','target.png'],execute)
    assert len(calls)==(2 if stop else 6)
    assert code==(1 if stop else 0)
    assert len({c[-1] for c in calls})==len(calls)
    assert all('--no-reset' not in c and '--goal-image' in c for c in calls)
    report=json.loads((root/'batch_summary.json').read_text())
    assert report['status']==('STOPPED' if stop else 'COMPLETED')


@pytest.mark.parametrize('status',['ABSTAIN','PLANNING_REJECTED','ACTION_LIMIT'])
def test_inconclusive_run_is_recorded_and_next_reset_starts(tmp_path,status):
    calls=[]
    def execute(cmd):
        calls.append(cmd);output=Path(cmd[-1]);output.mkdir()
        (output/'episode_summary.json').write_text(json.dumps(dict(status=status,physical_actions=0)))
        return 0
    root=tmp_path/'batch'
    assert run_batch(root,3,['--execute','--goal-image','target.png'],execute)==0
    report=json.loads((root/'batch_summary.json').read_text())
    assert len(calls)==3 and report['status']=='COMPLETED'
    assert all(row['outcome_class']=='INCONCLUSIVE' for row in report['runs'])


def test_interrupt_never_restarts(tmp_path):
    def execute(cmd):raise KeyboardInterrupt()
    assert run_batch(tmp_path/'batch',6,['--execute'],execute)==130
    assert json.loads((tmp_path/'batch/batch_summary.json').read_text())['status']=='INTERRUPTED'


def test_missing_summary_never_restarts(tmp_path):
    calls=[]
    def execute(cmd):calls.append(cmd);return 0
    assert run_batch(tmp_path/'batch',6,['--execute'],execute)==1
    assert len(calls)==1

@pytest.mark.parametrize('stop', [False, True])
def test_continue_batch_resumes_current_then_resets_remaining(tmp_path, stop):
    calls=[]
    parent=tmp_path/'original_run_03'
    def execute(cmd):
        calls.append(cmd)
        output=Path(cmd[-1]);output.mkdir()
        status='EXECUTION_FAILED' if stop else 'MODEL_REPORTED_DONE'
        (output/'episode_summary.json').write_text(json.dumps(dict(status=status)))
        return 1 if stop else 0
    root=tmp_path/'continuation'
    code=run_batch(root,6,['--execute','--goal-image','saved.png'],execute,start_index=3,continue_from=parent)
    assert len(calls)==(1 if stop else 4)
    assert '--continue-from' in calls[0]
    assert all('--continue-from' not in c and '--no-reset' not in c for c in calls[1:])
    report=json.loads((root/'batch_summary.json').read_text())
    assert [r['run'] for r in report['runs']]==([3] if stop else [3,4,5,6])
    assert code==(1 if stop else 0)

@pytest.mark.parametrize('status,start',[('HUMAN_REPORTED_DONE',5),('ABSTAIN',5)])
def test_batch_continuation_skips_human_confirmed_run(tmp_path,status,start):
    from cloth_agent.experiments.unfold_batch import batch_continuation
    run=tmp_path/'run04';run.mkdir()
    (tmp_path/'batch_summary.json').write_text(json.dumps(dict(status='STOPPED',run_limit=6,runs=[dict(run=4,output=str(run),status=status)])))
    (run/'protocol.json').write_text(json.dumps(dict(config='scene.json',max_actions=20)))
    begin,limit,args=batch_continuation(run)
    assert (begin,limit)==(start,6)
    assert '--capture-goal' not in args

def test_manual_setup_before_every_run_and_no_reset(tmp_path):
    order=[]
    def confirm(prompt):
        order.append('confirm')
        return 'y'
    def execute(cmd):
        assert '--no-reset' in cmd
        order.append('execute')
        output=Path(cmd[-1]);output.mkdir()
        (output/'episode_summary.json').write_text(json.dumps(dict(status='ACTION_LIMIT')))
        return 0
    root=tmp_path/'manual'
    assert run_batch(root,3,['--execute'],execute,manual_reset=True,confirm=confirm)==0
    assert order==['confirm','execute']*3
    report=json.loads((root/'batch_summary.json').read_text())
    assert report['automatic_reset'] is False
    assert report['manual_setup_confirmations']==[1,2,3]

@pytest.mark.parametrize('answer',['','n','yes','EOF'])
def test_manual_decline_or_eof_does_not_start_robot(tmp_path,answer):
    def confirm(prompt):
        if answer=='EOF':raise EOFError()
        return answer
    def execute(cmd):pytest.fail('must not start robot')
    assert run_batch(tmp_path/'manual',2,['--execute'],execute,manual_reset=True,confirm=confirm)==1

def test_manual_execution_error_does_not_start_next_run(tmp_path):
    confirmations=[]
    def confirm(prompt):confirmations.append(prompt);return 'y'
    def execute(cmd):
        output=Path(cmd[-1]);output.mkdir()
        (output/'episode_summary.json').write_text(json.dumps(dict(status='EXECUTION_FAILED')))
        return 1
    assert run_batch(tmp_path/'manual',3,['--execute'],execute,manual_reset=True,confirm=confirm)==1
    assert len(confirmations)==1


def test_missing_summary_reports_child_exit_code(tmp_path):
    root=tmp_path/'batch'
    assert run_batch(root,1,['--execute'],lambda cmd:2)==1
    report=json.loads((root/'batch_summary.json').read_text())
    assert report['status']=='ERROR'
    assert report['child_exit_code']==2
    assert 'FileNotFoundError' not in report['reason']
    assert 'episode_summary.json' in report['missing_episode_summary']


@pytest.mark.parametrize('runs',[1,3])
def test_continue_from_accepts_run_count_and_inherits_manual_camera_settings(tmp_path,monkeypatch,runs):
    from cloth_agent.experiments import unfold_batch as batch
    parent=tmp_path/'old_batch';parent.mkdir();source=parent/'run01';source.mkdir()
    (source/'protocol.json').write_text(json.dumps(dict(config='scene.json',max_actions=20,
        camera_mode='dual-wrist',fusion_config='wrist.json',topdown_config='topdown.json')))
    (source/'episode_summary.json').write_text(json.dumps(dict(status='ABSTAIN',physical_actions=16,
        steps=[dict(round=16,execution=dict(status='PLANNING_FAILED',physical_execution=False))])))
    (parent/'batch_summary.json').write_text(json.dumps(dict(manual_reset=True)))
    captured=[]
    def run(root,count,args,**kwargs):
        captured.append((count,args,kwargs));return 0
    monkeypatch.setattr(batch,'run_batch',run)
    assert batch.main(['--continue-from',str(source),'--runs',str(runs),'--execute','--output',str(tmp_path/'new')])==0
    count,args,opts=captured[0]
    assert count==runs and opts['continue_from']==source and opts['manual_reset'] is True
    assert args[args.index('--goal-image')+1]==str(source/'goal/target.png')
    assert args[args.index('--fixed-wrist-config')+1]=='wrist.json'


def test_manual_continuation_prompt_preserves_current_garment(tmp_path):
    prompts=[];commands=[]
    def confirm(text):prompts.append(text);return 'y'
    def execute(cmd):
        commands.append(cmd);out=Path(cmd[-1]);out.mkdir()
        (out/'episode_summary.json').write_text(json.dumps(dict(status='MODEL_REPORTED_DONE')))
        return 0
    assert run_batch(tmp_path/'batch',2,['--execute'],execute,continue_from=tmp_path/'source',manual_reset=True,confirm=confirm)==0
    assert '保留当前衣服状态' in prompts[0] and '请手动摆好衣服' in prompts[1]
    assert '--continue-from' in commands[0] and '--continue-from' not in commands[1]
    assert all('--no-reset' in c for c in commands)
