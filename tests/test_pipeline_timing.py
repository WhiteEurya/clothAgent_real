import json
from pathlib import Path

import pytest

from cloth_agent.pipeline_timing import PipelineTiming, active_timing, timed_stage


def test_nested_timing_preserves_results_iteration_and_exclusive_time(tmp_path,monkeypatch):
    import cloth_agent.pipeline_timing as module
    now=[0.0]
    monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    @timed_stage('child')
    def child():
        now[0]+=3
        return 42
    with PipelineTiming(tmp_path/'timing') as recorder:
        with recorder.span('parent',iteration=2):
            now[0]+=2
            assert child()==42
            now[0]+=5
    r=json.loads((tmp_path/'timing/summary.json').read_text())
    parent,child=r['spans']
    assert parent['duration_s']==10 and parent['exclusive_s']==7
    assert child['parent_id']==parent['id'] and child['duration_s']==3
    assert child['details']['iteration']==2
    assert active_timing() is None
    timeline=json.loads((tmp_path/'timing/host_timeline.json').read_text())
    assert sum(r['duration_s'] for r in timeline)==10
    assert [r['stage'] for r in timeline]==['parent','child','parent']
    assert (tmp_path/'timing/host_spans.csv').is_file()


def test_errors_interrupts_and_retries_remain_distinct(tmp_path):
    @timed_stage('attempt')
    def attempt(error):
        if error:raise error
    with pytest.raises(KeyboardInterrupt):
        with PipelineTiming(tmp_path/'timing') as recorder:
            with recorder.span('run'):
                with pytest.raises(ValueError):attempt(ValueError('failed'))
                attempt(None)
                attempt(KeyboardInterrupt())
    spans=json.loads((tmp_path/'timing/summary.json').read_text())['spans']
    assert [r['status'] for r in spans]==['INTERRUPTED','ERROR','RETURNED','INTERRUPTED']
    assert active_timing() is None
    assert len((tmp_path/'timing/events.jsonl').read_text().splitlines())==8


def test_logging_failure_does_not_interrupt_work(tmp_path,monkeypatch):
    with PipelineTiming(tmp_path/'timing') as recorder:
        original=Path.open
        def fail(path,*args,**kwargs):
            if path.name=='events.jsonl':raise OSError('disk failure')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path,'open',fail)
        with recorder.span('work'):
            answer=42
    assert answer==42 and recorder.logging_error=='disk failure'


def test_original_cli_opt_in_and_existing_directory_protection(tmp_path,monkeypatch):
    from cloth_agent import fold_exploration_pipeline as pipeline
    seen=[]
    def run(args):
        seen.append((args.real,active_timing() is not None))
        return 0
    monkeypatch.setattr(pipeline,'_run_main',run)
    assert pipeline.main(['--max-iterations','1'])==0
    assert pipeline.main(['--max-iterations','1','--timing-output',str(tmp_path/'timing')])==0
    assert seen==[(False,False),(False,True)]
    with pytest.raises(FileExistsError):
        pipeline.main(['--timing-output',str(tmp_path/'timing')])
    assert len(seen)==2


def test_summary_links_remote_response_timing_to_call_stage(tmp_path):
    debug=tmp_path/'trace';debug.mkdir()
    rows=[{'type':'stream_event','event':{'type':'message_start','message':{}},
           '_cloth_timing':{'elapsed_s':2}},
          {'type':'stream_event','event':{'type':'message_stop'},
           '_cloth_timing':{'elapsed_s':5}}]
    (debug/'claude_stdout.txt').write_text('\n'.join(json.dumps(r) for r in rows))
    with PipelineTiming(tmp_path/'timing') as recorder:
        with recorder.span('remote.invoke',debug_dir=str(debug),usage_stage='visual_planning',iteration=3):
            pass
    calls=json.loads((tmp_path/'timing/claude_calls.json').read_text())
    assert calls[0]['stage']=='visual_planning' and calls[0]['iteration']==3
    assert calls[0]['events']['rounds'][0]['emission_s']==3


def test_host_operations_and_milestones_use_actual_clock(tmp_path, monkeypatch):
    from cloth_agent.pipeline_timing import timed_call
    import cloth_agent.pipeline_timing as module
    now=[0.0]
    monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    def capture(value):
        now[0]+=2
        return value
    with PipelineTiming(tmp_path/'timing') as recorder:
        recorder.mark('iteration','starting iteration 1',iteration=1)
        with recorder.span('iteration',iteration=1):
            assert timed_call('camera.capture',capture,42)==42
            recorder.mark('evaluation','evaluation completed',iteration=1)
    report=json.loads((tmp_path/'timing/summary.json').read_text())
    assert report['milestones'][1]['elapsed_s']==2
    assert report['spans'][1]['details']['iteration']==1
    assert (tmp_path/'timing/milestones.csv').is_file()
