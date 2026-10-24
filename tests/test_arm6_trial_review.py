import json
import threading
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.experiments.arm6_trial_review import fixed_route


def test_route_rejects_disconnected_unsupported_and_repeated_actions():
    graph={'edges':[dict(edge_id='a',source_node='x',target_node='y'),
                    dict(edge_id='b',source_node='y',target_node='z'),
                    dict(edge_id='c',source_node='q',target_node='r')]}
    templates={key:{} for key in ['a','b','c']}
    assert [e['edge_id'] for e in fixed_route(graph,templates,'a,b',2)]==['a','b']
    for ids,limit in [('a,c',2),('a,a',2),('a,b',1),('missing',1)]:
        with pytest.raises(ValueError):fixed_route(graph,templates,ids,limit)
    with pytest.raises(ValueError):fixed_route(graph,{'a':{}},'a,b',2)


@pytest.mark.parametrize('review,cancel_review,expected,executions',[
    (False,False,'ROUTE_STATE_MISMATCH',0),
    (True,True,'INTERRUPTED',0),
    (True,False,'GOAL_REACHED',1),
])
@pytest.mark.parametrize('live_start',[False,True])
def test_route_keeps_model_evidence_requires_review_and_stops_at_goal(tmp_path,monkeypatch,review,cancel_review,expected,executions,live_start):
    from cloth_agent.experiments import arm6_traj_replay as replay
    from cloth_agent.experiments import graph_policy_replay as cli
    from cloth_agent.experiments import arm6_trial_review as ui
    from cloth_agent.experiments import arm6_live_start as placement
    from cloth_agent.experiments import arm6_replay_media as media
    from cloth_agent.experiments import replay_live_graph,discovery_model
    from cloth_agent import gamepad_collect_runtime,gamepad_collect
    from cloth_agent.experiments.graph_policy import evidence_file
    from cloth_agent.experiments.discovery_data import save
    rgb=tmp_path/'rgb.png';Image.new('RGB',(40,40),'blue').save(rgb)
    fresh_rgb=tmp_path/'fresh.png';Image.new('RGB',(40,40),'green').save(fresh_rgb)
    nodes=[dict(node_id=n,representative_observations=[o]) for n,o in [('x','o1'),('y','o2')]]
    edge=dict(edge_id='a',source_node='x',target_node='y',seed_action={'source_observation':'o1'},action_family='pull',action_template={})
    graph=dict(protocol=replay.PROTOCOL,nodes=nodes,edges=[edge,dict(edge_id='b',source_node='y',target_node='x')],
               terminal_node_ids=[],replay_unavailable_edges={})
    trajectory={'observations':{o:{'rgb':evidence_file(rgb)} for o in ['o1','o2']}}
    templates={'a':dict(reference_start_px=[10,10],reference_target_px=[20,20]),'b':{}}
    path=tmp_path/'graph.json';path.write_text('{}')
    monkeypatch.setattr(cli,'load_frozen',lambda _:graph)
    monkeypatch.setattr(replay,'compile_templates',lambda *a,**kw:(graph,trajectory,templates))
    seen=[]
    runtime=SimpleNamespace(cancel=threading.Event(),observe=lambda *a:SimpleNamespace(image=rgb),
        arrival=lambda *a:None,move_to_perception=lambda *a:None,close=lambda:None,
        stop=lambda:seen.append('stop'),_execute_operations=lambda *a:(seen.append('execute') or {'status':'EXECUTED'}))
    monkeypatch.setattr(gamepad_collect_runtime,'CollectionRuntime',lambda *a:runtime)
    monkeypatch.setattr(replay_live_graph,'create_live_graph',lambda *a:SimpleNamespace(update=lambda *a,**kw:None,close=lambda:None))
    model_match=dict(status='MATCH',node_id='y',candidate_nodes=['y'],confidence=.95,
                     matched_evidence=['model prediction'],violated_conditions=[],unresolved_conditions=[])
    match_calls=[]
    def match(*a,**kw):
        match_calls.append(a[2].image)
        return dict(model_match)
    monkeypatch.setattr(replay,'topology_match',match)
    class Requests:
        def __init__(self,*a,**kw):pass
        def ask(self,*a,**kw):
            seen.append(('grounding_image',a[2][-1][1]))
            return dict(grasp_px=None,target_px=None,confidence=.2)
        def metrics(self):return {}
    monkeypatch.setattr(replay,'Requests',Requests)
    monkeypatch.setattr(discovery_model,'CodexMultimodal',lambda *a:object())
    class Recorder:
        def __init__(self,*a):pass
        def start(self):pass
        def stage(self,*a):pass
        def check(self):pass
        def close(self):pass
    monkeypatch.setattr(gamepad_collect,'TrajectoryRecorder',Recorder)
    monkeypatch.setattr(replay,'prepare_motion',lambda *a:([],{}))
    monkeypatch.setattr(media,'render_video',lambda *a:None)
    monkeypatch.setattr(media,'render_sequence_video',lambda *a:None)
    def approve(current,reference,proposed,template,node,match,folder,**kw):
        if cancel_review:raise KeyboardInterrupt('operator cancelled')
        seen.append(('review',node));save(folder/'operator_review.json',{'approved':True})
        if proposed is None:return None
        return dict(proposed,grasp_px=[10,10],target_px=[20,20])
    monkeypatch.setattr(ui,'review_step',approve)
    def place(*a):
        return SimpleNamespace(image=fresh_rgb),dict(model_match,node_id='x',candidate_nodes=['x'])
    monkeypatch.setattr(placement,'prepare_live_start',place)
    output=tmp_path/'run'
    args=['--graph',str(path),'--arm6-expanded-graph','--arm6-route','a','--execute','--no-live-graph','--output',str(output)]
    if review:args+=['--arm6-review']
    if live_start and review:args+=['--arm6-live-start']
    cli.main(args)
    result=json.loads((output/'result.json').read_text())
    assert result['status']==expected,result
    assert result['physical_actions']==executions
    assert seen.count('execute')==executions
    if live_start and review:
        assert ('grounding_image',fresh_rgb) in seen
        assert len(match_calls)==executions  # no duplicate initial recognition
    if executions:
        assert ('review','x') in seen and ('review','y') in seen
        assert result['completion_basis']=='operator_confirmed'
        assert result['operator_assisted'] and len(result['operator_reviews'])==2
        raw=json.loads((output/'step_0000/grounding.json').read_text())
        chosen=json.loads((output/'step_0000/grounding_operator_selected.json').read_text())
        assert raw==dict(grasp_px=None,target_px=None,confidence=.2)
        assert chosen['confidence']==.2 and chosen['grasp_px']==[10,10]
        assert json.loads((output/'step_0000/node_match.json').read_text())['node_id']==('x' if live_start else 'y')
