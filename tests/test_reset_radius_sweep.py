from types import SimpleNamespace
from pathlib import Path
import pytest
from cloth_agent import unfold_experiment_policy as p
from cloth_agent.experiments import reset_radius_sweep as sweep


def test_streak_and_done_priority():
    n=0
    for v in ['UNCHANGED','WORSE','UNCERTAIN']:n=p.next_streak(n,v,True)
    assert n==3
    assert p.stop_reason('ACTION',n,3,3,12)=='NO_IMPROVEMENT'
    assert p.stop_reason('ABSTAIN',n,3,3,12)=='NO_IMPROVEMENT'
    assert p.stop_reason('DONE',n,3,3,12) is None
    assert p.next_streak(n,'IMPROVED',True)==0
    assert p.next_streak(0,None,True)==0
    assert p.stop_reason('ACTION',0,3,12,12)=='ROUND_LIMIT'
    assert p.stop_reason('DONE',0,3,12,12) is None


def test_invalid_trials_do_not_make_zero_success():
    rows=[dict(d_mm=20,outcome='FAILURE'),dict(d_mm=20,outcome='FAILURE'),dict(d_mm=20,outcome='INVALID')]
    assert not p.radius_summary(rows,20)['zero_success_stop']
    rows.append(dict(d_mm=20,outcome='FAILURE'))
    assert p.radius_summary(rows,20)['zero_success_stop']
    assert p.radius_summary(rows,20)['invalid_trials']==1
    for reason in ('SAFETY_STOP','USER_STOP','ABSTAIN'):
        assert p.classify(dict(finished_at='now',termination_reason=reason))=='INVALID'
    assert p.classify(dict(finished_at='now',termination_reason='DONE'),operator_stopped=True)=='INVALID'
    assert p.classify(dict(finished_at='now',termination_reason='DONE',cleanup_error='error'))=='INVALID'


@pytest.mark.parametrize('all_success',[False,True])
def test_sweep_without_hardware(tmp_path,monkeypatch,all_success):
    goal=tmp_path/'goal.png';goal.write_bytes(b'goal')
    config=tmp_path/'config.json';config.write_text('{}')
    args=SimpleNamespace(output=tmp_path/'sweep',goal_image=goal,config=config,port=8775,
                         wait_before_first=True,max_consecutive_invalid=3)
    monkeypatch.setattr(sweep.batch,'serve',lambda *a:SimpleNamespace(shutdown=lambda:None,server_close=lambda:None))
    waits=[]
    transition=sweep.Control.transition
    def ready(c,phase,episode=None):
        transition(c,phase,episode)
        if phase=='WAITING_READY':
            waits.append(c.episode);c.ready.set()
    monkeypatch.setattr(sweep.Control,'transition',ready)
    commands=[]
    def launch(cmd,log,control):
        commands.append(cmd)
        d=int(cmd[cmd.index('--initialize-center-radius-mm')+1])
        folder=Path(cmd[cmd.index('--output')+1]);folder.mkdir()
        assert '--uncertain-is-no-improvement' in cmd
        assert cmd[cmd.index('--max-rounds')+1]=='13'
        assert cmd[cmd.index('--max-actions')+1]=='12'
        assert '--automatic-top5' not in cmd
        if all_success:reason='DONE'
        else:
            reason='SAFETY_STOP' if len(commands)==1 else 'DONE' if len(commands)==2 else 'NO_IMPROVEMENT'
        sweep.base.save(folder/'episode_summary.json',dict(finished_at='now',termination_reason=reason,
             total_rounds=4,physical_actions=3,episode_wall_s=120))
        return 0
    monkeypatch.setattr(sweep.base,'launch_episode',launch)
    sweep.run(args)
    data=sweep.base.json.loads((args.output/'sweep_results.json').read_text())
    if all_success:
        assert len(commands)==21
        assert waits==[0]  # initial confirmation only; DONE advances across radii automatically
        assert data['stop_reason']=='RADIUS_LIMIT_80_MM_NO_ZERO_SUCCESS'
    else:
        assert len(commands)==7
        assert waits==[0,1,3,4,5,6]  # no wait after successful attempt 2
        assert data['stop_reason']=='ZERO_SUCCESS_AT_30_MM'
        assert data['radii'][0]['success_rate']==1/3
        assert data['radii'][0]['invalid_trials']==1
        assert data['radii'][1]['success_rate']==0
    assert (args.output/'index.html').is_file()
    assert (args.output/'trials.csv').is_file()
