import numpy as np
import pytest

from cloth_agent.experiments import local_dense_proposal_v2 as p


def test_dense_descriptors_match_pooling_definition_and_normalization():
    rng=np.random.default_rng(4);grid=rng.normal(size=(5,6,8)).astype(np.float32);valid=np.ones((5,6),bool);valid[-1,-1]=False
    ds=p.descriptors(grid,valid)
    for d in ds.values():assert np.allclose(np.linalg.norm(d,axis=-1),1,atol=1e-5)
    single=p.normalize(grid)
    for row,col in [(0,0),(2,3),(4,5)]:
        for w in (3,9):
            r=w//2;g=single[max(0,row-r):row+r+1,max(0,col-r):col+r+1]
            mask=valid[max(0,row-r):row+r+1,max(0,col-r):col+r+1]
            expected=p.normalize(g[mask].mean(axis=0))
            actual=p.pool(single,valid,w)[row,col]
            assert np.allclose(expected,actual,atol=1e-5)
    assert ds['C'].shape[-1]==16


def test_nms_fixed_original_pixel_radius_and_padding():
    xy=np.array([[[10,10],[30,10],[80,10],[140,10]]],float)
    scores=np.array([[1,.9,.8,.7]]);valid=np.array([[True,True,True,False]])
    peaks=p.nms(scores,xy,valid,10,radius=56)
    assert [c['flat_index'] for c in peaks]==[0,2]
    assert p.nearest_pa(peaks,[80,10],1)==70
    assert p.nearest_pa(peaks,[80,10],5)==0


def test_precision_is_not_recall_and_pending_or_maybe_are_explicit():
    rows=[dict(candidate_id=str(i)) for i in range(5)]
    assert p.precision(rows,{},5)['precision_strict'] is None
    labels={'0':'YES','1':'YES','2':'NO','3':'NO','4':'MAYBE'}
    result=p.precision(rows,labels,5)
    assert result['precision_strict']==.4 and result['precision_upper']==.6
    assert result['precision_decisive']==.5


def test_holdout_excludes_same_trajectory_pixels_and_primary_same_session():
    def step(sid,session,unit,images):return dict(step_id=sid,session=session,trajectory_unit=unit,images=images)
    q=step('q','s1','s1/ep1',{'before':'i1','after':'i2'})
    steps={'q':q,'adjacent':step('adjacent','s1','s1/ep1',{'before':'i3'}),
           'other_episode':step('other_episode','s1','s1/ep2',{'before':'i4'}),
           'cross':step('cross','s2','s2/ep1',{'before':'i5'}),
           'duplicate':step('duplicate','s3','s3/ep1',{'before':'i2'})}
    primary=p.reference_steps(q,list(steps),steps,'cross_session_primary')
    assert [s['step_id'] for s in primary]==['cross']
    secondary=p.reference_steps(q,list(steps),steps,'leave_trajectory_out_secondary')
    assert [s['step_id'] for s in secondary]==['other_episode','cross']


def test_localization_requires_frozen_human_groups_before_features(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'verify',lambda root:{})
    monkeypatch.setattr(p,'ensure_features',lambda *args:pytest.fail('no features before group confirmation'))
    with pytest.raises(ValueError,match='human experience grouping'):p.localize(tmp_path)


def test_candidate_reviews_do_not_fill_experience_groups(tmp_path,monkeypatch):
    from cloth_agent.experiments import local_dense_review as review
    monkeypatch.setattr(review,'summarize',lambda root:None)
    p.save(tmp_path/'manifest.json',{'steps':[]})
    p.save(tmp_path/'experience_groups.draft.json',{'steps':{},'annotator':''})
    p.save(tmp_path/'candidate_labels.json',{'labels':{'c0':'UNREVIEWED'},'annotator':''})
    p.save(tmp_path/'discovery/results.json',{'unique_candidates':{'c0':{}}})
    before=(tmp_path/'experience_groups.draft.json').read_bytes()
    review.apply_review(tmp_path,{'action':'candidate','id':'c0','label':'MAYBE','annotator':'human'})
    assert p.read(tmp_path/'candidate_labels.json')['labels']['c0']=='MAYBE'
    assert (tmp_path/'experience_groups.draft.json').read_bytes()==before
