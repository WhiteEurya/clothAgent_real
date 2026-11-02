import io
import json
import threading
from types import SimpleNamespace
from rax_cloth import progress


def test_countdown_resume_and_partial_events(tmp_path,monkeypatch):
    monkeypatch.setattr(progress.time,'monotonic',lambda:100.)
    monkeypatch.setattr(progress.time,'time',lambda:200.)
    output=io.StringIO();events=tmp_path/'events.jsonl';events.write_text('')
    relay=SimpleNamespace(lock=threading.Lock(),max_calls=32,state={'calls_reserved':6,'requests':[]})
    p=progress.Progress(tmp_path,10,output);p.start(events,relay,1000,1200)
    e=json.dumps({'type':'item.started','item':{'type':'mcp_tool_call','tool':'observe'}})
    events.write_text(e[:20]);p.tick()
    assert '开始 MCP' not in output.getvalue()
    with events.open('a') as f:f.write(e[20:]+'\n')
    p.tick()
    assert '开始 MCP：observe' in output.getvalue()
    assert '已用 05:00' in output.getvalue() and '距时间上限 15:00' in output.getvalue()
    assert '调用 6/32' in output.getvalue()
    assert json.loads((tmp_path/'progress.json').read_text())['remaining_seconds']==900
    p.close()


def test_waiting_model_heartbeat_and_no_raw_payload(tmp_path,monkeypatch):
    monkeypatch.setattr(progress.time,'monotonic',lambda:100.)
    monkeypatch.setattr(progress.time,'time',lambda:200.)
    events=tmp_path/'events.jsonl';events.write_text(json.dumps({'type':'item.completed','item':{
        'type':'mcp_tool_call','tool':'observe','result':{'image':'SECRET_IMAGE_PAYLOAD'}}})+'\n')
    relay=SimpleNamespace(lock=threading.Lock(),max_calls=32,state={'calls_reserved':1,'requests':[
        {'id':0,'status':'reserved','phase':'response_headers','started_unix_s':155}]})
    out=io.StringIO();p=progress.Progress(tmp_path,10,out);p.start(events,relay,700,600);p.tick()
    assert '等待响应头' in out.getvalue() and '00:45' in out.getvalue()
    assert 'SECRET_IMAGE_PAYLOAD' not in out.getvalue()
    count=len(out.getvalue());p.tick();assert len(out.getvalue())==count
    p.close()


def test_terminal_controls_and_commands_are_shortened(tmp_path):
    out=io.StringIO();p=progress.Progress(tmp_path,stream=out)
    p.event({'type':'item.started','item':{'type':'command_execution','command':'python test.py\n'+'x'*10000}})
    assert 'python test.py' in out.getvalue() and 'xxxx' not in out.getvalue()
    assert progress.short('\x1b[31mhello\nworld')=='hello world'
    assert (tmp_path/'progress.jsonl').is_file()


def test_original_operator_requests_are_visible_without_auto_reply(tmp_path):
    folder=tmp_path/'operator'/'requests';folder.mkdir(parents=True)
    request=folder/'012345abcdef.json'
    row={'id':'012345abcdef','status':'pending','message':'Please reset shirt','kind':'reset'}
    request.write_text(json.dumps(row))
    out=io.StringIO();p=progress.Progress(tmp_path,stream=out)
    p.operator_requests(tmp_path)
    assert 'Please reset shirt' in out.getvalue()
    assert '--request-id 012345abcdef' in out.getvalue()
    assert not (folder/'012345abcdef.reply.json').exists()
    first=out.getvalue();p.operator_requests(tmp_path);assert out.getvalue()==first
    row['status']='expired';request.write_text(json.dumps(row))
    p.operator_requests(tmp_path)
    assert '已超时' in out.getvalue()
