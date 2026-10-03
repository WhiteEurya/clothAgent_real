import json

from cloth_agent.harness.planner_profile import analyze_events


def test_profile_uses_remote_times_and_retains_schema_retry():
    def e(t, **data):
        return json.dumps(dict(data,_cloth_timing={'elapsed_s':t}))
    rows=[e(10,type='stream_event',event={'type':'message_start','message':{}}),
          e(12,type='stream_event',event={'type':'content_block_start','index':0,'content_block':{'type':'tool_use','name':'StructuredOutput'}}),
          e(20,type='stream_event',event={'type':'content_block_stop','index':0}),
          e(20,type='assistant',message={'content':[{'type':'tool_use','id':'one','name':'StructuredOutput','input':{'selected_reference':{'reference_id':'R048'}}}]}),
          e(21,type='user',message={'content':[{'type':'tool_result','tool_use_id':'one','is_error':True,'content':'too long'}]}),
          e(22,type='stream_event',event={'type':'message_stop'}),
          e(30,type='stream_event',event={'type':'message_start','message':{}}),
          e(35,type='stream_event',event={'type':'message_stop'})]
    result=analyze_events('\n'.join(rows))
    assert result['rounds'][0]['emission_s']==12
    assert result['rounds'][0]['blocks'][0]['elapsed_s']==8
    assert result['rounds'][1]['wait_before_first_event_s']==8
    assert result['tool_calls'][0]['result_wait_s']==1
    assert result['tool_calls'][0]['error']=='too long'


def test_missing_producer_timestamps_are_not_invented():
    assert analyze_events(json.dumps({'type':'assistant','received_elapsed_s':900}))['rounds']==[]


def test_entrypoint_preserves_original_call_and_backend_callback(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from cloth_agent.harness import planner_profile as module
    manifest=tmp_path/'manifest.json';manifest.write_text('{}')
    case={'prompt':'original prompt','system_prompt':'original system','schema':{'type':'object'},
          'context_files':{'manifest.json':'{}'},'source':{'decision_id':'saved'},
          'original_budget':{'max_turns':16,'image_edit_limit':6}}
    received={}
    class Backend:
        def __init__(self,**kwargs):
            assert kwargs['record_event_timing'] is True
            self.last_timings={'total_s':0.1}
        def invoke(self,**kwargs):
            received.update(kwargs)
            self.progress_callback('call','started',None,image_count=4)
            trace=kwargs['debug_dir'];trace.mkdir()
            (trace/'claude_stdout.txt').write_text(json.dumps({
                'type':'stream_event','event':{'type':'message_start','message':{}},
                '_cloth_timing':{'elapsed_s':1}}))
            return SimpleNamespace(stdout=json.dumps({'selected_reference':{'reference_id':'R001'}}))
    monkeypatch.setattr(module,'RemoteClaudeBackend',Backend)
    monkeypatch.setattr(module,'load_case',lambda *args:(case,[]))
    monkeypatch.setattr(module,'validate_plan',lambda *args:{'valid':True})
    assert module.main(['--output',str(tmp_path/'result'),'--manifest',str(manifest)])==0
    for key in ('prompt','system_prompt','schema','context_files'):
        assert received[key]==case[key]
    assert received['max_turns']==16 and received['image_edit_limit']==6
    profile=json.loads((tmp_path/'result/profile.json').read_text())
    assert len(profile['events']['rounds'])==1
