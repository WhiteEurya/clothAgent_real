import copy
import json
import urllib.request
from urllib.error import HTTPError
import pytest
from cloth_agent.experiments.replay_live_graph import LiveGraph, projection
from cloth_agent.experiments import graph_policy_replay as pr
from tests.test_graph_policy_replay import policy, Model, Runtime


def test_live_read_only_and_persistent_snapshot(tmp_path):
    graph=dict(nodes=[dict(node_id='S0',task_context='</script>cloth')],edges=[])
    original=copy.deepcopy(graph)
    live=LiveGraph(tmp_path,0)
    try:
        live.update(graph,current='S0',phase='MATCHED')
        with urllib.request.urlopen(live.url+'/state.json') as response:
            assert json.load(response)['current']=='S0'
        with pytest.raises(HTTPError):urllib.request.urlopen(live.url+'/../secret')
        with pytest.raises(HTTPError):urllib.request.urlopen(urllib.request.Request(live.url,data=b'execute'))
        live.update(graph,phase='STOPPED',result='NO_REPLAYABLE_EDGE')
    finally:live.close()
    assert graph==original
    text=(tmp_path/'live_graph/index.html').read_text()
    assert '\\u003c/script>' in text
    assert json.loads((tmp_path/'live_graph/state.json').read_text())['phase']=='STOPPED'


def test_growth_new_node_and_shared_action_outcomes(tmp_path):
    s=dict(nodes={'S0':dict(node=dict(task_context='start'),status='EXISTING')},actions={})
    live=LiveGraph(tmp_path,0)
    try:
        live.update(s,current='S0')
        s['nodes']['P1']=dict(node=dict(task_context='new'),status='PROVISIONAL')
        s['actions']['a']=dict(source_node='S0',nominal_target='S0',preferred=True,outcomes={'S0':{},'P1':{}})
        live.update(s,current='P1',phase='MATCHED')
        live.update(s,phase='STOPPED')
        assert live.state['new_nodes']==['P1']
        assert len(live.state['events'])==1
        assert len(live.state['actions'])==1
        assert live.state['actions'][0]['targets']==['S0','P1']
    finally:live.close()


def test_replay_updates_do_not_advance_current_until_rematch(tmp_path,monkeypatch):
    from cloth_agent.experiments import replay_live_graph as lg
    events=[]
    class Sink:
        state={}
        def update(self,graph,**kwargs):
            self.state.update(kwargs);events.append(copy.deepcopy(self.state))
        def close(self):pass
    monkeypatch.setattr(lg,'create_live_graph',lambda *a:Sink())
    path,_=policy(tmp_path)
    result=pr.replay(path,tmp_path/'run',Runtime(),Model(),live_graph_port=0)
    selected=next(e for e in events if e['phase']=='GROUNDING')
    assert selected['current']=='N0' and selected['target']=='N1'
    executing=next(e for e in events if e['phase']=='EXECUTING')
    assert executing['current']=='N0'
    assert events[-1]['current']=='N1'
    assert events[-1]['phase']=='STOPPED' and events[-1]['result']==result['status']


def test_online_growth_publishes_new_node(tmp_path):
    from tests.test_online_graph_growth import Requests, matched, match
    path,g=policy(tmp_path)
    req=Requests([matched('N0'),match('NO_MATCH'),match('NO_MATCH')],node=g['nodes'][1])
    pr.online_replay(path,tmp_path/'run',Runtime(),req,max_steps=1,live_graph_port=0)
    s=json.loads((tmp_path/'run/live_graph/state.json').read_text())
    assert len(s['new_nodes'])==1 and s['current']==s['new_nodes'][0]
    a=s['actions'][0]
    assert a['targets']==['N1',s['current']]
    assert s['phase']=='STOPPED'


@pytest.mark.parametrize('unresolved',[False,True])
def test_handle_recovery_is_display_only_and_links_only_confirmed_rematch(tmp_path,monkeypatch,unresolved):
    from cloth_agent.experiments import replay_live_graph as lg
    from tests.test_robustness_recovery import ChangingRuntime, GateModel, runner, matched, SOURCE
    events=[]
    real_update=lg.LiveGraph.update
    def capture(self,graph,**kwargs):
        real_update(self,graph,**kwargs)
        events.append(copy.deepcopy(self.state))
    monkeypatch.setattr(lg.LiveGraph,'update',capture)
    path,graph=policy(tmp_path);original=path.read_bytes()
    runtime=ChangingRuntime()
    run,_,_=runner(runtime,answer='motion.blocked("NO_ROBUST_ACTION")' if unresolved else SOURCE)
    result=pr.replay(path,tmp_path/'run',runtime,
        GateModel([matched('N0'),matched('N0'),matched('N1')]),
        fallback_fn=run,max_recovery_attempts=1,live_graph_port=0)
    recovering=next(e for e in events if e['recoveries'])
    r=recovering['recoveries'][0]
    assert r['origin_node']=='N0' and r['origin_edge']=='E0'
    assert r['status']=='PLANNING' and r['matched_node'] is None
    assert recovering['current']=='N0'
    final=events[-1]
    assert final['new_nodes']==[]
    assert {n['id'] for n in final['nodes']}=={n['node_id'] for n in graph['nodes']}
    assert final['recoveries'][0]['matched_node']==(None if unresolved else 'N0')
    assert final['recoveries'][0]['status']==('RECOVERY_UNRESOLVED' if unresolved else 'REMATCHED')
    assert final['recoveries'][0]['redispatched'] is (not unresolved)
    assert final['result']==result['status']
    assert path.read_bytes()==original


def test_recovery_display_failure_cannot_interrupt_replay(tmp_path):
    from cloth_agent.experiments.robustness_recovery import HandleRecovery
    def broken_display(events):raise RuntimeError('browser disconnected')
    recovery=HandleRecovery(tmp_path,on_update=broken_display)
    recovery.snapshot()
    assert json.loads((tmp_path/'robustness_recovery/audit.json').read_text())['attempts']==[]


def test_observation_images_are_exported_and_only_named_assets_are_served(tmp_path):
    from PIL import Image
    ref=tmp_path/'reference.png';current=tmp_path/'current.png'
    Image.new('RGB',(20,30),'red').save(ref)
    Image.new('RGB',(30,20),'blue').save(current)
    graph=dict(nodes=[dict(node_id='S0',representative_observations=['o1'])],edges=[])
    original=copy.deepcopy(graph)
    live=LiveGraph(tmp_path/'run',0)
    try:
        live.update(graph,observations={'o1':{'rgb':{'path':str(ref)}}})
        reference=live.state['nodes'][0]['images'][0]
        live.update(graph,current='S0',phase='MATCHED',observation=str(current),
            recoveries=[dict(id='r1',before={'path':str(ref)},after=None)])
        images=live.state['nodes'][0]['images']
        assert [i['label'] for i in images]==['最近匹配观测','节点参考 · o1']
        assert images[1]==reference and images[0]['src']!=reference['src']
        assert live.state['recoveries'][0]['images'][0]['src']==reference['src']
        assert live.state['recoveries'][0]['images'][1]['src'] is None
        with urllib.request.urlopen(live.url+'/'+reference['src']) as response:
            assert response.headers['Content-Type']=='image/jpeg'
            assert response.read()==(live.folder/reference['src']).read_bytes()
        for path in ['/images/','/images/../state.json','/images/%2e%2e/state.json','/'+str(ref),'/images/unknown.jpg']:
            with pytest.raises(HTTPError):urllib.request.urlopen(live.url+path)
        assert len(live.image_cache)==2
        live.update(graph,phase='EXECUTING')
        assert live.state['nodes'][0]['images']==images
        assert len(live.image_cache)==2
    finally:live.close()
    assert graph==original
    assert (live.folder/reference['src']).exists()  # Relative assets work with the offline HTML.


def test_missing_reference_does_not_break_graph_display(tmp_path):
    graph=dict(nodes=[dict(node_id='S0',representative_observations=['missing'])],edges=[])
    live=LiveGraph(tmp_path,0)
    try:
        live.update(graph,observations={'missing':{'rgb':{'path':str(tmp_path/'gone.jpg')}}},current='S0')
        assert live.state['current']=='S0'
        assert live.state['nodes'][0]['images']==[dict(label='节点参考 · missing',src=None)]
        assert (live.folder/'index.html').exists()
    finally:live.close()
