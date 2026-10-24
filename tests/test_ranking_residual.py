import copy
import threading
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from cloth_agent import action_preference as p, ranking_residual as r, residual_features as f
from cloth_agent import preference_one_shot as scoring


def rows():
    return [dict(id=f'P{i:02d}R', point_id=f'P{i:02d}', Ps=80-i, Pt=50,
        grasp_px=[20+i*3, 30], target_px=[80+i*3, 30], grasp_angle_deg=0,
        point_reason='test', trajectory_reason='test') for i in range(10)]


def features(candidates=None, image='new'):
    candidates = rows() if candidates is None else candidates
    return dict(identity=dict(version=f.VERSION, dimension=f.DIM, weights={'test':'fixed'}),
        image_sha256=image, wall_s=.1,
        features={row['id']:[i/10]+[0.]*(f.DIM-1) for i,row in enumerate(candidates)})


def manifest(state, key='review', image='new', candidates=None):
    candidates = rows() if candidates is None else candidates
    order, context = r.prepare(candidates, state, features(candidates, image))
    return dict(review_id=key, token='secret', alpha=state['alpha'], displayed=order[:10],
                all_candidates=order, ranking_context=context)


def choice(m):
    return dict(review_id=m['review_id'], token=m['token'], decision='SELECT',
        candidate_id=m['displayed'][-2]['id'], scene_unchanged=True,
        ranking_confirmed=True, human_order=[c['id'] for c in m['displayed']][::-1])


def test_zero_residual_exact_base_and_features_ignore_ids_labels():
    tokens=np.random.default_rng(1).normal(size=(1024,384))
    candidates=rows()
    x=f.action_features(tokens,np.array([150,100]),candidates)
    renamed=copy.deepcopy(candidates)
    for c in renamed:c.update(id='new'+c['id'],point_reason='unrelated',trajectory_reason='other')
    y=f.action_features(tokens,np.array([150,100]),renamed)
    assert list(x.values())==list(y.values())
    reverse=copy.deepcopy(candidates[:1]);reverse[0]['target_px']=[10,30]
    assert x[candidates[0]['id']]!=f.action_features(tokens,np.array([150,100]),reverse)[reverse[0]['id']]
    assert all(len(v)==f.DIM for v in x.values())


def test_zero_base_train_once_evaluate_future_not_fitted_round(tmp_path):
    path=tmp_path/'residual.json';state=r.load_state(path,1.5945233392133669)
    m=manifest(state)
    assert [(c['id'],c['S']) for c in m['displayed']]==[(c['id'],c['P']) for c in p.ranked(rows(),state['alpha'])]
    assert all(c['correction']==0 for c in m['displayed'])
    label=choice(m)
    event=r.apply_feedback(path,m,label,tmp_path)
    assert event['evaluation']['before_training']
    assert event['evaluation']['current']['pairwise_accuracy']==0
    assert event['evaluation']['current']['revision']==0
    state=r.load_state(path,state['alpha'])
    assert state['revision']==1 and state['alpha']==m['alpha']
    assert event['training_fit']['replay_loss_after']<event['training_fit']['replay_loss_before']
    assert r.apply_feedback(path,m,label,tmp_path)==event
    assert r.load_state(path,state['alpha'])['revision']==1
    future=manifest(state,key='next',image='unseen_rgb')
    evaluated=r.evaluate(future,label['human_order'])
    assert evaluated['eligible_new_image']
    assert evaluated['current']['revision']==1 and evaluated['previous']['revision']==0
    assert evaluated['delta_vs_previous']>0
    assert event['evaluation']['current']['pairwise_accuracy']==0  # immutable pre-update evidence
    repeated=manifest(state,key='repeat',image='new')
    assert not r.evaluate(repeated,label['human_order'])['eligible_new_image']
    r.write_progress(state,tmp_path)
    assert '尚无后续新状态证据' in (tmp_path/'learning_progress.md').read_text()


def test_reject_stale_duplicate_partial_orders_and_changed_alpha(tmp_path):
    path=tmp_path/'state.json';state=r.load_state(path,1)
    m=manifest(state);label=choice(m)
    for change in ({'human_order':label['human_order'][:-1]},
                   {'human_order':['bad']*10}, {'human_order':None}, {'ranking_confirmed':False},
                   {'candidate_id':'absent'}, {'scene_unchanged':False}, {'token':'old'}):
        with pytest.raises(ValueError):p.validate_choice(dict(label,**change),m)
    assert p.validate_choice(dict(label,decision='STOP',human_order=None),m)['decision']=='STOP'
    r.apply_feedback(path,m,label,tmp_path)
    with pytest.raises(ValueError,match='another ranking'):
        r.apply_feedback(path,m,dict(label,human_order=label['human_order'][::-1]),tmp_path)
    with pytest.raises(ValueError,match='stale'):
        r.apply_feedback(path,dict(m,review_id='stale'),label,tmp_path)
    with pytest.raises(ValueError,match='fixed alpha'):
        r.load_state(path,2)
    bad=features();bad['identity']['weights']={'other':'weights'}
    with pytest.raises(ValueError,match='identity'):
        r.prepare(rows(),r.load_state(path,1),bad)


def test_evidence_failure_prevents_training_and_state_commit(tmp_path,monkeypatch):
    path=tmp_path/'state.json';state=r.load_state(path,1);m=manifest(state)
    saved=path.read_bytes(); original=p.atomic_save
    def fail(path,value):
        if path.name=='pre_update_evaluation.json':raise OSError('disk full')
        original(path,value)
    monkeypatch.setattr(p,'atomic_save',fail)
    monkeypatch.setattr(r,'train',lambda *a:pytest.fail('training before evidence durable'))
    with pytest.raises(OSError,match='disk full'):r.apply_feedback(path,m,choice(m),tmp_path)
    assert path.read_bytes()==saved


def test_checkpoint_state_survives_feedback_artifact_failure(tmp_path,monkeypatch):
    path=tmp_path/'state.json';state=r.load_state(path,1);m=manifest(state);label=choice(m)
    original=p.atomic_save
    def fail(path,value):
        if path.name=='residual_feedback.json':raise OSError('interrupted artifact')
        original(path,value)
    monkeypatch.setattr(p,'atomic_save',fail)
    with pytest.raises(OSError):r.apply_feedback(path,m,label,tmp_path)
    assert r.load_state(path,1)['revision']==1
    monkeypatch.setattr(p,'atomic_save',original)
    event=r.apply_feedback(path,m,label,tmp_path)
    assert event['revision_after']==1 and r.load_state(path,1)['revision']==1


def test_metrics_compare_only_displayed_and_do_not_use_future_labels(tmp_path):
    state=r.load_state(tmp_path/'state.json',1)
    candidates=rows()+[dict(rows()[0],id='outside',Ps=0,Pt=0)]
    m=manifest(state,candidates=candidates)
    assert 'outside' not in {c['id'] for c in m['displayed']}
    ev=r.evaluate(m,[c['id'] for c in m['displayed']])
    assert ev['current']['pair_count']==45 and ev['current']['pairwise_accuracy']==1
    assert ev['current']['top1_agreement'] and ev['current']['kendall_tau']==1
    assert ev['current']['human_top1_rank']==1


def test_group_ties_never_create_pairs_and_metrics_ignore_internal_order(tmp_path):
    path=tmp_path/'state.json';state=r.load_state(path,1);m=manifest(state)
    ids=[c['id'] for c in m['displayed']]
    groups=[ids[:4],ids[4:7],ids[7:]]
    label=dict(choice(m),human_groups=groups)
    parsed=p.validate_choice(label,m)
    assert parsed['human_groups']==groups
    event=r.apply_feedback(path,m,label,tmp_path)
    x,offsets=r._pairs([event])
    assert len(offsets)==4*3+4*3+3*3==33
    e=event['evaluation']['current']
    assert e['pair_count']==33 and e['ignored_tie_pairs']==12
    assert e['pairwise_accuracy']==1
    # Within-group permutations do not change labels, update counts or accuracy.
    flipped=[g[::-1] for g in groups]
    assert r.apply_feedback(path,m,dict(label,human_groups=flipped),tmp_path)==event
    reversed_inside=[i for g in flipped for i in g]
    assert r.metrics(reversed_inside,ids,groups)['pairwise_accuracy']==1
    assert r.metrics(reversed_inside,ids,groups)['top1_agreement']
    assert r.load_state(path,1)['revision']==1


def test_all_tied_records_feedback_but_does_not_train_or_claim_improvement(tmp_path,monkeypatch):
    path=tmp_path/'state.json';state=r.load_state(path,1);m=manifest(state)
    ids=[c['id'] for c in m['displayed']]
    monkeypatch.setattr(r,'train',lambda *a:pytest.fail('all ties must not update or shrink weights'))
    event=r.apply_feedback(path,m,dict(choice(m),human_groups=[ids]),tmp_path)
    assert event['training_fit']['skipped']
    assert event['evaluation']['current']['pairwise_accuracy'] is None
    assert event['evaluation']['delta_vs_base'] is None
    updated=r.load_state(path,1)
    assert updated['checkpoints'][-1]['weights']==state['checkpoints'][-1]['weights']
    r.write_progress(updated,tmp_path)
    assert '尚无后续新状态证据' in (tmp_path/'learning_progress.md').read_text()


@pytest.mark.parametrize('groups',[[],[[]],None,'bad',[[1]], [['P00R']]*10])
def test_invalid_groups_rejected(groups):
    with pytest.raises(ValueError):r.validate_groups(groups,{c['id'] for c in rows()})


@pytest.mark.parametrize('mode',['select','stop','timeout','cancel_after_training'])
def test_choose_train_order_and_execution_locked(tmp_path,monkeypatch,mode):
    (tmp_path/'goal').mkdir();folder=tmp_path/'round_000';folder.mkdir()
    current=tmp_path/'current.png'
    for pth in (current,tmp_path/'goal/target.png'):Image.new('RGB',(150,100),'purple').save(pth)
    alpha=tmp_path/'alpha.json';before=p.load_state(alpha);alpha_bytes=alpha.read_bytes()
    residual=tmp_path/'residual.json';cancel=threading.Event();calls=[]
    monkeypatch.setattr(f,'extract',lambda *a:features(image=p.io.sha(current)))
    selected=rows()[4]
    def wait(folder,m,*a,**kw):
        calls.append('human')
        assert (folder.parent/'prediction_before_feedback.json').exists()
        if mode=='timeout':raise TimeoutError('no choice')
        if mode=='stop':return dict(decision='STOP'),3.
        return dict(choice(m),candidate_id=selected['id']),3.
    monkeypatch.setattr(p,'wait_for_choice',wait)
    original_train=r.train
    def train(*a):
        assert (folder/'pre_update_evaluation.json').exists()
        calls.append('train')
        if mode=='cancel_after_training':cancel.set()
        return original_train(*a)
    monkeypatch.setattr(r,'train',train)
    def run():return scoring.choose_plan(tmp_path,folder,SimpleNamespace(image=current),dict(candidates=rows()),
        SimpleNamespace(cancel=cancel),{},alpha,0,1,residual_state_path=residual)
    if mode=='select':
        plan=run()
        assert plan==p.plan_for(selected)
        assert calls==['human','train']
        assert (tmp_path/'learning_progress.md').exists()
    else:
        with pytest.raises((KeyboardInterrupt,TimeoutError)):run()
    assert alpha.read_bytes()==alpha_bytes
    assert r.load_state(residual,before['alpha'])['revision']==int(mode in ('select','cancel_after_training'))
    assert not (folder/'ranking_after_feedback.json').exists()
