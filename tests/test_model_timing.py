import json
import time

from cloth_agent.experiments.model_timing import EventTimeline, summarize


def test_event_observer_handles_partial_lines_and_excludes_content(tmp_path):
    path=tmp_path/'events.jsonl';path.write_text('')
    observer=EventTimeline(path,time.monotonic());observer.start()
    with path.open('a') as stream:
        stream.write('{"type":"turn.started"');stream.flush();time.sleep(.03)
        stream.write('}\n');stream.write(json.dumps({'type':'item.completed','item':{'type':'reasoning','text':'DO NOT RECORD'}})+'\n')
        stream.write(json.dumps({'type':'turn.completed','usage':{'output_tokens':10}})+'\n');stream.flush()
    events=observer.finish()
    assert [e['type'] for e in events]==['turn.started','item.completed','turn.completed']
    assert 'DO NOT RECORD' not in json.dumps(events)
    assert not observer.error
    summary=summarize(events,1.)
    assert summary['turn_window_s']>=0
    assert summarize([],1.)['turn_window_s'] is None
