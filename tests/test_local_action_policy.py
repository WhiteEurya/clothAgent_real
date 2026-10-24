from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from cloth_agent.experiments.local_action_memory import crop_box,query_boxes,distinct_actions,encode_batches
from cloth_agent.experiments.local_action_matcher import validate_proposal


def test_crops_keep_anchors_and_grid_covers_image_boundaries():
    for size in [(1280,720),(40,30)]:
        for point in [(0,0),(size[0]-1,size[1]-1),(size[0]//2,size[1]//2)]:
            box=crop_box(size,point,.22)
            x,y=point;a,b,c,d=box
            assert 0<=a<=x<c<=size[0] and 0<=b<=y<d<=size[1]
        boxes=query_boxes(size,[.22,.44])
        assert len(boxes)==len(set(boxes))
        covered=np.zeros(size[::-1],dtype=bool)
        for a,b,c,d in boxes:covered[b:d,a:c]=True
        assert covered.all()
    with pytest.raises(ValueError):crop_box((1280,720),(-1,30),.2)


def test_rotations_and_scales_do_not_duplicate_action_shortlist():
    crops=[dict(edge_id='a',crop_id='a0'),dict(edge_id='a',crop_id='a90'),dict(edge_id='b',crop_id='b0')]
    boxes=[[0,0,10,10],[20,20,40,40]]
    result=distinct_actions([[.8,.7],[.95,.85]],[[0,2],[1,2]],boxes,crops,3)
    assert [r['edge_id'] for r in result]==['a','b']
    assert result[0]['query_box_xyxy']==boxes[1] and result[0]['reference_crop']['crop_id']=='a90'
    assert [r['edge_id'] for r in distinct_actions([[.9,.8]],[[0,2]],boxes,crops,3,['a'])]==['b']


def proposal(**kw):
    return dict(dict(decision='ACT',edge_id='a',confidence=.6,grasp_px=[20,20],target_px=[80,80],
                     local_evidence=['exposed edge'],expected_change='uncover lower layer',
                     uncertainties=[],reason='local coverage correspondence',previous_effect='INITIAL',completion_evidence=[]),**kw)


def test_selection_requires_retrieved_local_correspondence_not_arbitrary_pixels():
    rank=[dict(edge_id='a',query_box_xyxy=[10,10,40,40])]
    validate_proposal(proposal(),rank,(100,100))
    for bad in [dict(edge_id='other'),dict(grasp_px=[50,50]),dict(target_px=[101,80]),
                dict(local_evidence=[]),dict(target_px=[20,20]),dict(grasp_px=None)]:
        with pytest.raises(Exception):validate_proposal(proposal(**bad),rank,(100,100))
    done=proposal(decision='DONE',edge_id='',grasp_px=None,target_px=None,completion_evidence=['sleeves and hem exposed'])
    validate_proposal(done,rank,(100,100))
    with pytest.raises(ValueError):validate_proposal(dict(done,completion_evidence=[]),rank,(100,100))


def test_batched_query_embedding_shape_and_timing():
    class Encoder:
        def encode(self,images):return np.ones((len(images),4)),.1,.2
    vectors,timing=encode_batches(Encoder(),[None]*35,16)
    assert vectors.shape==(35,4)
    assert timing['dino_forward_s']==pytest.approx(.6)


def test_cancel_review_never_plans_or_dispatches(tmp_path,monkeypatch):
    from cloth_agent.experiments import arm6_local_action_policy as policy
    from cloth_agent.experiments import arm6_traj_replay as replay
    called=[]
    def cancel(*a):raise KeyboardInterrupt('operator stopped')
    monkeypatch.setattr(policy,'reviewed_anchors',cancel)
    monkeypatch.setattr(replay,'prepare_motion',lambda *a:called.append('plan'))
    runtime=SimpleNamespace(cancel=threading.Event(),_execute_operations=lambda *a:called.append('execute'))
    media=SimpleNamespace(phase=lambda *a:None)
    with pytest.raises(KeyboardInterrupt):
        policy.dispatch_reviewed(runtime,None,proposal(),{},tmp_path,media)
    assert not called


def test_memory_pairs_exact_demo_after_not_merged_node_representative(tmp_path,monkeypatch):
    from PIL import Image
    from cloth_agent.experiments import arm6_traj_replay as replay
    from cloth_agent.experiments.local_action_memory import export,load_library
    from cloth_agent.experiments.graph_policy import evidence_file
    before=tmp_path/'before.png';after=tmp_path/'after.png';wrong=tmp_path/'wrong.png'
    for p,color in [(before,'blue'),(after,'green'),(wrong,'red')]:Image.new('RGB',(80,60),color).save(p)
    graphfile=tmp_path/'graph.json';graphfile.write_text('{}')
    edge=dict(edge_id='a',seed_step_id='s',seed_action={'source_observation':'before'},
              action_family='inward move to expose sleeve',action_template={},observed_outcome='layer revealed',seed_outcome='PROGRESS')
    graph=dict(edges=[edge],replay_unavailable_edges={},nodes=[{'representative_observations':['wrong']}])
    trajectory=dict(steps=[dict(step_id='s',session='20261019_test',before_observation='before',after_observation='after')],
        observations={k:dict(rgb=evidence_file(p)) for k,p in [('before',before),('after',after),('wrong',wrong)]})
    templates={'a':dict(reference_start_px=[20,20],reference_target_px=[50,40])}
    monkeypatch.setattr(replay,'compile_templates',lambda *a,**kw:(graph,trajectory,templates))
    result=export(graphfile,tmp_path/'memory')
    library=load_library(tmp_path/'memory')
    assert result['actions']==1 and result['crops']==8
    assert library['actions']['a']['after']['sha256']==evidence_file(after)['sha256']
    assert library['actions']['a']['action_family']=='inward move to expose sleeve'


def test_offline_entry_never_constructs_robot(tmp_path,monkeypatch):
    import json
    from PIL import Image
    from cloth_agent import gamepad_collect_runtime
    from cloth_agent.experiments import arm6_local_action_policy as policy
    from cloth_agent.experiments.arm6_replay_media import ReplayMedia
    image=tmp_path/'image.png';Image.new('RGB',(100,100),'blue').save(image)
    memory=tmp_path/'memory';memory.mkdir()
    (memory/'library.json').write_text('{}');(memory/'index_manifest.json').write_text('{}')
    monkeypatch.setattr(policy,'load_library',lambda *a:{'actions':{}})
    def forbidden(*a):raise AssertionError('offline must never connect robot')
    monkeypatch.setattr(gamepad_collect_runtime,'CollectionRuntime',forbidden)
    monkeypatch.setattr(policy,'retrieve',lambda *a,**kw:{'ranking':[]})
    monkeypatch.setattr(ReplayMedia,'finish',lambda *a:None)
    assert policy.main(['--library',str(memory),'--image',str(image),'--retrieval-only','--output',str(tmp_path/'out')])==0
    result=json.loads((tmp_path/'out/result.json').read_text())
    assert result['physical_actions']==0 and result['robot_calls']==0
    assert result['graph_state_matching'] is False


def test_planning_failure_never_dispatches(tmp_path,monkeypatch):
    from cloth_agent.experiments import arm6_local_action_policy as policy
    from cloth_agent.experiments import arm6_traj_replay as replay
    calls=[]
    monkeypatch.setattr(policy,'reviewed_anchors',lambda *a:{k:proposal()[k] for k in ['grasp_px','target_px','confidence']})
    def fail(*a):raise ValueError('depth invalid')
    monkeypatch.setattr(replay,'prepare_motion',fail)
    runtime=SimpleNamespace(cancel=threading.Event(),_execute_operations=lambda *a:calls.append('execute'))
    with pytest.raises(ValueError,match='depth invalid'):
        policy.dispatch_reviewed(runtime,None,proposal(),{'template':{}},tmp_path,SimpleNamespace(phase=lambda *a:None))
    assert not calls
