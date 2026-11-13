from cloth_agent.experiments import unfold_console as console
import json
import subprocess
import sys


def test_failures_are_condensed_and_point_to_full_log(tmp_path,capsys):
    failures=[dict(role='target_px',pixel=[10+i,20],error='no measured source depth') for i in range(43)]
    console.failure('候选失败',str(failures),tmp_path,failures)
    output=capsys.readouterr().out
    assert '43 个候选' in output and '缺少实测深度' in output
    assert str(tmp_path/'skill_failure.json') in output
    assert '[52, 20]' not in output
    assert '\x1b' not in output


def test_summary_does_not_count_undispatched_or_partial_actions_as_completed(tmp_path,capsys):
    result=dict(status='INTERRUPTED',physical_actions=2,steps=[
        dict(execution=dict(status='EXECUTED',physical_execution=True)),
        dict(execution=dict(status='EXECUTION_FAILED',physical_execution=True)),
        dict(execution=dict(status='NOT_DISPATCHED'))])
    console.summary(tmp_path,result,120)
    output=capsys.readouterr().out
    assert '完成展开动作 1 次' in output and '2.0 分钟' in output
    assert 'INTERRUPTED' in output and 'episode_summary.json' in output
    assert 'physical_actions' not in output


def test_native_and_python_noise_go_to_log_but_ui_and_restore_work(tmp_path):
    script='''
import sys,os,io,contextlib
from pathlib import Path
from cloth_agent.experiments import unfold_console as c
root=Path(sys.argv[1]);c.open_session(root)
try:
    print('ROBOT_IP noisy SDK',flush=True)
    os.write(2,b'native warning retained\\n')
    with contextlib.redirect_stdout(io.StringIO()):
        print('[gripper] open: completion confirmed',file=sys.__stdout__,flush=True)
    c.emit('测试','用户可读状态')
    try:
        with c.stage('模拟异常','motion'):raise ValueError('simulated')
    except ValueError:pass
finally:c.close_session()
print('restored stdout')
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True,check=True)
    assert 'ROBOT_IP' not in result.stdout and 'native warning' not in result.stderr
    assert '用户可读状态' in result.stdout and 'restored stdout' in result.stdout
    assert '阶段中断' in result.stdout
    log=(tmp_path/'runtime.log').read_text()
    assert 'ROBOT_IP' in log and 'native warning retained' in log and '[gripper]' in log


def test_actual_cli_events_stream_without_reasoning_or_partial_action(tmp_path):
    script='''
import json,sys,time
from pathlib import Path
from cloth_agent.experiments import unfold_console as c
p=Path(sys.argv[1]);c.open_session(p)
try:
    events=p/'events.jsonl';c.watch_model(events)
    with c.stage('离线模拟模型请求','model'):
        with events.open('w') as f:
            for event in [{'type':'turn.started'},{'type':'item.completed','item':{'type':'reasoning','text':'private text'}},{'type':'item.completed','item':{'type':'agent_message','text':'{"decision":"ACTION"}'}},{'type':'turn.completed'}]:
                f.write(json.dumps(event)+'\\n');f.flush();c.poll_model()
finally:c.close_session()
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True,check=True)
    assert '模型正在处理图像' in result.stdout and '模型请求已完成' in result.stdout
    assert 'private text' not in result.stdout and '"decision"' not in result.stdout
    assert '预计约' in result.stdout
