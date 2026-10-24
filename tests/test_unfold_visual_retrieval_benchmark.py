"""Offline protocol checks; no model service or robot hardware is contacted."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cloth_agent.experiments import unfold_visual_retrieval_benchmark as b


@pytest.fixture
def memory(tmp_path):
    images=[];nodes=[];observations={}
    for i in range(24):
        path=tmp_path/f'{i}.png';Image.new('RGB',(32,24),(i, i*3, i*7)).save(path)
        oid=f'o{i:03d}';group=f'n{i//2:03d}'
        images.append(dict(image_id=oid,path=str(path),sha256=b.file_hash(path),pixel_sha256=b.pixel_hash(path),
                           proxy_group=group,sources=[dict(session=f's{i%2}')]))
        observations[oid]=dict(rgb=dict(path=str(path),sha256=b.file_hash(path)))
        if i%2==0:nodes.append(dict(node_id=group,task_context='cloth boundary',representative_observations=[],
            observation_condition=dict(positive_evidence=['visible fold'],negative_evidence=[],uncertainty_conditions=[])))
        nodes[-1]['representative_observations'].append(oid)
    vectors=np.eye(24,dtype='float32')
    return dict(images=images),b.build_index(vectors),dict(nodes=nodes,edges=[]),dict(observations=observations)


class Encoder:
    calls=0
    def encode(self,image):
        self.calls+=1
        vector=np.zeros((1,24),dtype='float32');vector[0,image.getpixel((0,0))[0]]=1
        return vector,0.,0.


class Backend:
    model='fake-offline'
    def reset(self):self.calls=0;self.reference_loading_s=0.;self.query_transport_s=0.;self.model_wall_s=0.
    def invoke(self,prompt,images,schema):
        self.calls+=1;self.prompt=prompt;self.images=images
        payload=json.loads(prompt.rsplit('\n',1)[-1]);self.payload=payload
        key=payload['payload']['nodes'][0]['node_id']
        return dict(text=json.dumps(dict(status='MATCH',node_id=key,candidate_nodes=[key],
            matched_evidence=['visible fold'],violated_conditions=[],unresolved_conditions=[],confidence=.95)),
            tool_calls=0,visual_tool_calls=0,completed_turns=1)


def test_retrieval_excludes_self_and_exact_pixels(memory):
    data,index,_,_=memory;query=data['images'][0]
    # A different ID with identical pixels must also be excluded.
    data['images'][1]['pixel_sha256']=query['pixel_sha256']
    ranking=b.retrieve(index,np.eye(24,dtype='float32')[:1],data['images'],query,20)
    assert len(ranking)==20
    assert not {'o000','o001'}&{r['image_id'] for r in ranking}
    assert b.positives(data['images'],query)==set()


def test_proxy_recall_eligibility_and_cross_session(memory,tmp_path):
    data,index,_,_=memory;encoder=Encoder()
    rows=b.accuracy(data,index,encoder,tmp_path)
    assert encoder.calls==24
    assert all(len(r['positive_ids'])==1 for r in rows)
    assert all(r['cross_session_positive_ids']==r['positive_ids'] for r in rows)
    assert all(r['query_id'] not in {x['image_id'] for x in r['ranking']} for r in rows)


@pytest.mark.parametrize('mode',b.MODES)
def test_trial_encoding_and_matcher_contract(memory,tmp_path,mode):
    data,index,graph,trajectory=memory;encoder=Encoder();backend=Backend()
    row=b.trial(mode,data['images'][0],data,index,encoder,graph,trajectory,backend,tmp_path/mode)
    assert row['status']=='OK',row['error']
    assert encoder.calls==(0 if mode=='current_global_match' else 1)
    assert row['total_s']==pytest.approx(sum(row[s] for s in b.STAGES))
    if mode=='retrieval_only':assert backend.calls==0
    else:
        assert backend.calls==1
        assert backend.prompt.startswith(b.COMMON+'\n'+b.PROMPTS['node_match']+'\n')
        assert backend.images[0]==Path(data['images'][0]['path'])
        assert Path(data['images'][0]['path']) not in backend.images[1:]
        assert len(backend.images)==(11 if mode=='retrieval_gpt_rerank' else 24)
        assert 'edges' not in backend.payload['payload']
        if mode=='retrieval_gpt_rerank':assert row['match']['best_candidate'] in {r['image_id'] for r in row['ranking']}
        else:
            attempts=json.loads((tmp_path/mode/'matching/matching_attempts.json').read_text())['attempts']
            assert [r['stage'] for r in attempts]==['global']


def test_single_shot_never_repairs_or_retries(memory,tmp_path):
    class Broken(Backend):
        def invoke(self,*a):self.calls+=1;return {'text':'not json'}
    data,index,graph,trajectory=memory;backend=Broken()
    row=b.trial('retrieval_gpt_rerank',data['images'][0],data,index,Encoder(),graph,trajectory,backend,tmp_path)
    assert row['status']=='ERROR' and backend.calls==1


def test_image_timing_preserves_existing_transport(monkeypatch,tmp_path):
    source=tmp_path/'source.png';Image.new('RGB',(32,24),(31,42,53)).save(source)
    def original(self,prompt,images,schema):
        for i,path in enumerate(images):
            with b.discovery_model.Image.open(path) as image:image.convert('RGB').save(tmp_path/f'copy{i}.png')
        return {'text':'{}'}
    monkeypatch.setattr(b.discovery_model.CodexMultimodal,'invoke',original)
    backend=object.__new__(b.TimedBackend);backend.reset()
    backend.invoke('unchanged',[source,source],{})
    assert backend.calls==1 and backend.reference_loading_s>0 and backend.query_transport_s>0
    assert backend.model_wall_s>=0
    assert b.pixel_hash(source)==b.pixel_hash(tmp_path/'copy0.png')==b.pixel_hash(tmp_path/'copy1.png')


def test_report_separates_failures_and_reports_percentiles(memory,tmp_path,monkeypatch):
    data,index,_,_=memory;b.accuracy(data,index,Encoder(),tmp_path)
    b.save(tmp_path/'config.json',dict(modes=b.MODES,repeats=20,warmup=1))
    for i,value in enumerate([1.,2.,9.]):
        row={s:0. for s in b.STAGES};row.update(condition='retrieval_only',mode='retrieval_only',iteration=i,
            status='OK' if i<2 else 'ERROR',error='timeout' if i==2 else '',total_s=value,query_image_decode_s=value)
        b.append_row(tmp_path,row)
    monkeypatch.setattr(b,'plot',lambda *a:None);b.report(tmp_path)
    stats=b.read(tmp_path/'statistics.json')['retrieval_only']
    assert stats['successes']==2 and stats['failures']==1
    assert stats['stages']['total_s']['median']==1.5
    assert stats['stages']['total_s']['p95']==pytest.approx(1.95)
    assert stats['all_attempt_total']['mean']==4.
    assert 'IN PROGRESS' in (tmp_path/'summary.md').read_text()
    assert len((tmp_path/'benchmark.csv').read_text().splitlines())==4


def test_trial_budget_rejects_under_twenty(tmp_path):
    with pytest.raises(ValueError,match='20'):b.run(tmp_path,tmp_path/'run',repeats=19)


def test_schedule_varies_groups_and_is_repeatable(memory):
    data,_,_,_=memory
    schedule=b.stable_query_schedule(data['images'],20,42)
    assert schedule==b.stable_query_schedule(data['images'],20,42)
    assert len(set(schedule))==20
    lookup={r['image_id']:r['proxy_group'] for r in data['images']}
    assert len({lookup[oid] for oid in schedule[:12]})==12


def test_early_stop_is_disclosed_without_rewriting_original_design(memory,tmp_path,monkeypatch):
    data,index,_,_=memory;b.accuracy(data,index,Encoder(),tmp_path)
    config=dict(modes=b.MODES,repeats=20,warmup=1)
    b.frozen(tmp_path/'config.json',config)
    original_hash=b.file_hash(tmp_path/'config.json')
    b.save(tmp_path/'early_stop.json',dict(stop_after_pairs=10,original_planned_repeats=20,
        reason='observed stable large effect',status='STOPPED_AFTER_PAIRED_GROUP'))
    for mode in b.MODES:
        for i in range(10):
            row={s:0. for s in b.STAGES};row.update(condition=mode,mode=mode,iteration=i,
                status='OK',error='',total_s=1.,query_image_decode_s=1.)
            b.append_row(tmp_path,row)
    for size in b.SIZES:
        for i in range(20):
            row={s:0. for s in b.STAGES};row.update(condition=f'ann_scaling_{size}',mode='ann_scaling',iteration=i,
                status='OK',error='',total_s=.001,ann_search_s=.001)
            b.append_row(tmp_path,row)
    monkeypatch.setattr(b,'plot',lambda *a:None);b.report(tmp_path)
    text=(tmp_path/'summary.md').read_text()
    assert 'EXPLORATORY EARLY STOP: 10 completed paired attempts' in text
    assert 'n=10 was not the prespecified experimental design' in text
    assert 'outcome-dependent' in text
    assert 'IN PROGRESS' not in text
    assert b.file_hash(tmp_path/'config.json')==original_hash
    assert b.load_frozen(tmp_path/'config.json')['repeats']==20
    stats=b.read(tmp_path/'statistics.json')
    assert all(stats[m]['stages']['total_s']['n']==10 for m in b.MODES)
    assert all(stats[f'ann_scaling_{s}']['attempts']==20 for s in b.SIZES)
    # Later supplementation must not falsely claim that all twenty are complete.
    for mode in b.MODES:
        row={s:0. for s in b.STAGES};row.update(condition=mode,mode=mode,iteration=10,
            status='OK',error='',total_s=1.,query_image_decode_s=1.)
        b.append_row(tmp_path,row)
    b.report(tmp_path)
    text=(tmp_path/'summary.md').read_text()
    assert 'Supplementation beyond the early-stop point is in progress' in text
    assert 'Run completeness: IN PROGRESS' in text
    assert b.file_hash(tmp_path/'config.json')==original_hash
