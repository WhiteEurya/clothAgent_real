"""Retrieval integration tests use no robot, GPU or external model service."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest
from cloth_agent.experiments import retrieval_matcher as rm
from cloth_agent.experiments.visual_retrieval_index import PROTOCOL,graph_references,validate_memory
from cloth_agent.experiments.graph_policy import frozen,evidence_file,Requests
from cloth_agent.experiments.graph_policy_schema import MATCH
from cloth_agent.experiments.arm6_traj_replay import matching_graph
from cloth_agent.experiments.graph_policy_replay import PROMPTS


@pytest.fixture
def gallery(tmp_path):
    observations={};nodes=[]
    for i in range(12):
        p=tmp_path/f'o{i}.png';Image.new('RGB',(20,20),(i*10,50,80)).save(p)
        observations[f'o{i}']=dict(rgb=evidence_file(p))
        if i%2==0:nodes.append(dict(node_id=f'n{i//2}',representative_observations=[],task_context='cloth boundary',
            observation_condition=dict(positive_evidence=['exposed seam'],negative_evidence=[],uncertainty_conditions=[])))
        nodes[-1]['representative_observations'].append(f'o{i}')
    tp=tmp_path/'trajectory.json';trajectory=dict(observations=observations);frozen(tp,trajectory)
    graph=dict(protocol='arm6-archive-graph-v1',nodes=nodes,edges=[],source_trajectory=evidence_file(tp),terminal_node_ids=[])
    gp=tmp_path/'graph.json';frozen(gp,graph)
    root=tmp_path/'memory';root.mkdir();(root/'reference.index').write_bytes(b'fake-index-not-used-by-unit-test')
    memory=dict(protocol=PROTOCOL,graph_path=str(gp),graph_sha256=rm.file_hash(gp),trajectory=graph['source_trajectory'],
        images=graph_references(graph,trajectory),artifacts={'reference.index':rm.file_hash(root/'reference.index')})
    frozen(root/'memory.json',memory)
    query=tmp_path/'current.png';Image.new('RGB',(20,20),'red').save(query)
    return root,gp,graph,trajectory,query


class Worker:
    instances=[]
    def __init__(self,*a,**kw):self.closed=False;self.calls=[];self.instances.append(self)
    def query(self,image,top_k,exclude_ids=()):
        self.calls.append((image,top_k,exclude_ids))
        return dict(ranking=[dict(image_id=f'o{i}',node_id=f'n{i//2}',similarity=1-i*.01) for i in range(top_k)],
            query_encode_calls=1,reference_encode_calls_runtime=0,timing={'dino_forward_s':.01,'ann_search_s':.0001})
    def close(self):self.closed=True


class Backend:
    model='offline-test'
    def __init__(self,status='MATCH',bad=False):self.calls=[];self.status=status;self.bad=bad
    def invoke(self,prompt,images,schema):
        self.calls.append((prompt,images,schema))
        if self.bad:return dict(text='bad json')
        payload=json.loads(prompt.rsplit('\n',1)[1]);self.payload=payload
        nodes=payload['payload']['nodes'];nid=nodes[0]['node_id'] if self.status=='MATCH' else ''
        self.response=dict(status=self.status,node_id=nid,candidate_nodes=[nid] if nid else [],confidence=.95,
            matched_evidence=['exposed seam'] if nid else [],violated_conditions=[],unresolved_conditions=[])
        return dict(text=json.dumps(self.response),tool_calls=0,visual_tool_calls=0,completed_turns=1)


def make_requests(tmp_path,backend,max_calls=5):
    return Requests(tmp_path,backend,max_calls=max_calls,schemas={'node_match':MATCH},
        prompts={'node_match':PROMPTS['node_match']},common='Only match visible cloth, no actions.')


def test_top_ten_group_by_real_node_and_use_one_budgeted_request(gallery,tmp_path):
    root,gp,graph,trajectory,query=gallery;backend=Backend();requests=make_requests(tmp_path/'model',backend)
    matcher=rm.RetrievalMatcher(root,gp,graph,trajectory,tmp_path/'worker',worker_factory=Worker)
    metrics={};original=copy.deepcopy(graph)
    value=matcher.match(matching_graph(graph),trajectory,SimpleNamespace(image=query),requests,tmp_path/'match',metrics,
        pending=dict(source_node='FORBIDDEN_SOURCE',target_node='FORBIDDEN_TARGET'))
    assert value['node_id']=='n0' and len(backend.calls)==1 and requests.model_calls==1
    prompt,images,_=backend.calls[0]
    assert len(images)==11 and images[0]==query
    nodes=backend.payload['payload']['nodes']
    assert len(nodes)==5 and all(len(n['representative_observations'])==2 for n in nodes)
    assert all(n['node_id'].startswith('n') for n in nodes)
    assert 'FORBIDDEN' not in prompt and 'edges' not in backend.payload['payload']
    assert graph==original and metrics['retrieval_queries']==1
    assert requests.metrics()['completed_model_turns']==1
    matcher.close();assert matcher.worker.closed


@pytest.mark.parametrize('status',['AMBIGUOUS','NO_MATCH'])
def test_uncertain_results_do_not_fallback_or_force_match(gallery,tmp_path,status):
    root,gp,graph,trajectory,query=gallery;backend=Backend(status);req=make_requests(tmp_path/'model',backend)
    matcher=rm.RetrievalMatcher(root,gp,graph,trajectory,tmp_path/'worker',worker_factory=Worker)
    value=matcher.match(matching_graph(graph),trajectory,SimpleNamespace(image=query),req,tmp_path/'match',{})
    assert value['status']==status and not value['node_id'] and len(backend.calls)==1
    attempts=json.loads((tmp_path/'match/matching_attempts.json').read_text())['attempts']
    assert attempts[0]['decision']==('UNKNOWN' if status=='NO_MATCH' else status)


def test_bad_gpt_output_is_not_retried_and_shared_budget_is_enforced(gallery,tmp_path):
    root,gp,graph,trajectory,query=gallery;backend=Backend(bad=True);req=make_requests(tmp_path/'model',backend,max_calls=1)
    matcher=rm.RetrievalMatcher(root,gp,graph,trajectory,tmp_path/'worker',worker_factory=Worker)
    with pytest.raises(ValueError):matcher.match(matching_graph(graph),trajectory,SimpleNamespace(image=query),req,tmp_path/'first',{})
    assert len(backend.calls)==req.model_calls==1
    backend.bad=False
    with pytest.raises(RuntimeError,match='budget'):matcher.match(matching_graph(graph),trajectory,SimpleNamespace(image=query),req,tmp_path/'second',{})
    assert len(backend.calls)==1


def test_unknown_candidate_node_rejected_before_gpt(gallery,tmp_path):
    root,gp,graph,trajectory,query=gallery;backend=Backend();req=make_requests(tmp_path/'model',backend)
    class BadWorker(Worker):
        def query(self,*a):
            value=super().query(*a);value['ranking'][0]['node_id']='invented';return value
    matcher=rm.RetrievalMatcher(root,gp,graph,trajectory,tmp_path/'worker',worker_factory=BadWorker)
    with pytest.raises(ValueError,match='ANN'):matcher.match(matching_graph(graph),trajectory,SimpleNamespace(image=query),req,tmp_path/'match',{})
    assert not backend.calls


def test_mismatched_graph_and_changed_index_rejected(gallery,tmp_path):
    root,gp,graph,trajectory,query=gallery
    other=tmp_path/'other.json';other.write_text('{}')
    with pytest.raises(ValueError,match='selected graph'):validate_memory(root,other)
    (root/'reference.index').write_bytes(b'changed')
    with pytest.raises(ValueError,match='modified'):validate_memory(root,gp)


def test_gallery_requires_unique_node_membership(gallery):
    _,_,graph,trajectory,_=gallery
    graph['nodes'][1]['representative_observations'].append('o0')
    with pytest.raises(ValueError,match='multiple'):graph_references(graph,trajectory)


def test_retrieval_cli_fails_before_robot_connection_if_worker_cannot_start(gallery,tmp_path,monkeypatch):
    from cloth_agent.experiments import graph_policy_replay as cli,arm6_traj_replay as replay
    from cloth_agent import gamepad_collect_runtime
    root,gp,graph,trajectory,query=gallery
    graph['replay_unavailable_edges']={}
    monkeypatch.setattr(replay,'compile_templates',lambda *a,**kw:(graph,trajectory,{}))
    monkeypatch.setattr(rm,'worker_python',lambda _: '/missing-python')
    seen=[]
    def failure(*a,**kw):seen.append('worker');raise RuntimeError('worker unavailable')
    monkeypatch.setattr(rm,'RetrievalMatcher',failure)
    def forbidden(*a):raise AssertionError('robot must not connect')
    monkeypatch.setattr(gamepad_collect_runtime,'CollectionRuntime',forbidden)
    out=tmp_path/'run'
    assert cli.main(['--graph',str(gp),'--arm6-expanded-graph','--matcher','retrieval_gpt_rerank',
        '--retrieval-memory',str(root),'--execute','--no-live-graph','--output',str(out)])==1
    result=json.loads((out/'result.json').read_text())
    assert seen==['worker'] and result['physical_actions']==0 and 'worker unavailable' in result['error']


def test_arm6_routes_to_retrieval_and_closes_worker(gallery,tmp_path,monkeypatch):
    from cloth_agent.experiments import graph_policy_replay as cli,arm6_traj_replay as replay,replay_live_graph
    from cloth_agent import gamepad_collect_runtime
    root,gp,graph,trajectory,query=gallery;graph['replay_unavailable_edges']={};seen=[]
    monkeypatch.setattr(replay,'compile_templates',lambda *a,**kw:(graph,trajectory,{}))
    class Matcher:
        def __init__(self,*a,**kw):seen.append('worker_started')
        def match(self,*a,**kw):
            seen.append('retrieval_match')
            return dict(status='MATCH',node_id='n0',candidate_nodes=['n0'],confidence=.95,
                matched_evidence=['seam'],violated_conditions=[],unresolved_conditions=[])
        def close(self):seen.append('worker_closed')
    monkeypatch.setattr(rm,'RetrievalMatcher',Matcher);monkeypatch.setattr(rm,'worker_python',lambda _: '/fake-python')
    def runtime(*a):
        seen.append('robot_started')
        return SimpleNamespace(observe=lambda *a:SimpleNamespace(image=query),close=lambda:seen.append('robot_closed'))
    monkeypatch.setattr(gamepad_collect_runtime,'CollectionRuntime',runtime)
    monkeypatch.setattr(replay_live_graph,'create_live_graph',lambda *a:SimpleNamespace(update=lambda *a,**kw:None,close=lambda:None))
    monkeypatch.setattr(replay,'topology_match',lambda *a,**kw:pytest.fail('old matcher must not run'))
    out=tmp_path/'run'
    assert cli.main(['--graph',str(gp),'--arm6-expanded-graph','--matcher','retrieval_gpt_rerank',
        '--retrieval-memory',str(root),'--execute','--no-live-graph','--output',str(out)])==0
    assert seen==['worker_started','robot_started','retrieval_match','robot_closed','worker_closed']
    result=json.loads((out/'result.json').read_text())
    assert result['status']=='NO_OUTGOING_EDGE' and result['physical_actions']==0


def test_retrieval_arguments_require_memory_and_arm6(tmp_path):
    from cloth_agent.experiments import graph_policy_replay as cli
    with pytest.raises(SystemExit):cli.main(['--graph',str(tmp_path/'absent.json'),'--matcher','retrieval_gpt_rerank'])


def test_explicit_worker_python_preserves_virtualenv_symlink(tmp_path):
    base=tmp_path/'base-python';base.write_text('base')
    entry=tmp_path/'venv/bin/python';entry.parent.mkdir(parents=True);entry.symlink_to(base)
    assert rm.worker_python(entry)==str(entry)
    assert rm.worker_python(entry)!=str(entry.resolve())
