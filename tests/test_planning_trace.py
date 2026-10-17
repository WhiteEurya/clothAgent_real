import json
from pathlib import Path

from cloth_agent.planning_trace import TraceCollector, PREFIX
from cloth_agent.codex_astra_pipeline import run
from tests.test_codex_astra_pipeline import fake_codex


def entry(phase='ORIENT',step=1):
    e=dict(step=step,phase=phase,goal='inspect observation',working_set=dict(files=['rgb.png'],symbols=[]),
           known=['image supplied'],unknown=['grasp point'],hypothesis=[],
           next_action=dict(type='READ',target='rgb.png',purpose='locate garment'),
           expected_result='visible garment edge',confidence='MEDIUM')
    if phase=='DONE':e.update(solution='candidate ready',validated_hypotheses=[],rejected_hypotheses=[],risks=['not executed'])
    return e


def event(e,ident='one',kind='agent_message'):
    return json.dumps(dict(type='item.completed',item=dict(id=ident,type=kind,text=PREFIX+json.dumps(e))))


def test_only_explicit_summaries_and_global_sequence(tmp_path):
    events=tmp_path/'events';events.touch()
    aggregate=tmp_path/'planning_trace.jsonl'
    c=TraceCollector(events,aggregate,tmp_path/'local1',1,'session')
    c.start();c.consume(event(entry()),1);c.consume(event(entry()),2)
    c.consume(event(entry(),'reasoning','reasoning'),3)
    c.consume(event(entry('DONE',2),'two'),4);status=c.close()
    assert status['complete'] and status['entries']==2
    c=TraceCollector(events,aggregate,tmp_path/'local2',2,'session')
    c.start();c.consume(event(entry()),1);c.close()
    rows=[json.loads(x) for x in aggregate.read_text().splitlines()]
    assert [r['step'] for r in rows]==[1,2,3]
    assert [r['model_step'] for r in rows]==[1,2,1]
    assert [r['iteration'] for r in rows]==[1,1,2]
    assert rows[1]['source_event_line']==4


def test_malformed_and_missing_summary_are_not_fabricated(tmp_path):
    events=tmp_path/'events';events.touch()
    c=TraceCollector(events,tmp_path/'all',tmp_path/'local',1)
    c.start()
    c.consume(event({'phase':'ORIENT'}),1)
    c.consume(json.dumps(dict(type='item.completed',item=dict(id='final',type='agent_message',text='motion.complete()'))),2)
    status=c.close()
    assert status['rejected']==1 and not status['complete']
    assert (tmp_path/'all').read_text()==''


def test_trace_written_before_model_exits_without_changing_answer(tmp_path):
    aggregate=tmp_path/'planning_trace.jsonl'
    binary=tmp_path/'fake_codex'
    binary.write_text('''#!/usr/bin/env python3
import json,sys,time
from pathlib import Path
prompt=sys.stdin.read()
assert 'PLANNING_TRACE' in prompt
print(json.dumps({'type':'thread.started','thread_id':'same-session'}),flush=True)
print(%r,flush=True)
trace=Path(%r)
for _ in range(200):
 if trace.exists() and trace.stat().st_size:break
 time.sleep(.01)
else:sys.exit(7)
print(%r,flush=True)
Path(sys.argv[sys.argv.index('--output-last-message')+1]).write_text('motion.complete()')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':4}}),flush=True)
''' % (event(entry()),str(aggregate),event(entry('DONE',2),'done')))
    binary.chmod(0o755)
    result=run('plan',workspace=tmp_path,output=tmp_path/'round_001/planner',binary=str(binary),
               read_only=True,trace_output=aggregate,trace_iteration=1,timeout_s=5)
    assert result['status']=='AGENT_FINISHED' and result['planning_trace']['complete']
    assert (tmp_path/'round_001/planner/answer.txt').read_text()=='motion.complete()'
    assert json.loads(aggregate.read_text().splitlines()[0])['session_id']=='same-session'


def test_trace_io_failure_does_not_fail_model(tmp_path,fake_codex):
    result=run('plan',workspace=tmp_path,output=tmp_path/'round_001/planner',binary=fake_codex,
               read_only=True,trace_output=tmp_path,trace_iteration=1)
    assert result['status']=='AGENT_FINISHED'
    assert 'planning_trace_error' in result and not result['planning_trace']['complete']
    assert (tmp_path/'round_001/planner/answer.txt').read_text().startswith('motion.move_joints')
