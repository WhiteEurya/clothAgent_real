import numpy as np
import pytest
from cloth_agent.experiments.action_anchor_transfer import norm_point,pixel_point,inside,descriptor_transfer,roma_transfer


def test_pixel_normalization_roundtrip():
    b=[320,390,570,660]
    for p in [[320,390],[407,545],[569,659]]:
        np.testing.assert_allclose(pixel_point(norm_point(p,b),b),p,atol=1e-10)
    assert not inside([570,660],b)


def test_descriptor_known_translation_and_ambiguous_rejection():
    torch=pytest.importorskip('torch')
    # Unique one-hot patch descriptors move one cell right; no image labels involved.
    ref=torch.eye(64).T.reshape(64,8,8);query=torch.roll(ref,1,dims=2)
    b=[0,0,80,80];r=descriptor_transfer(ref,query,[34.5,44.5],b)
    assert r['status']=='OK'
    np.testing.assert_allclose(r['point'],[44.5,44.5])
    flat=torch.ones(64,8,8);r=descriptor_transfer(flat,flat,[34.5,44.5],b)
    assert r['status']=='FAILURE' and r['point'] is None
    assert 'AMBIGUOUS_NONLOCAL_MATCH' in r['failure_reasons']


def test_roma_symmetric_layout_and_low_confidence():
    torch=pytest.importorskip('torch')
    n=32;y,x=torch.meshgrid(torch.linspace(-1+1/n,1-1/n,n),torch.linspace(-1+1/n,1-1/n,n),indexing='ij')
    xy=torch.stack([x,y],-1);shift=torch.tensor([.2,0.])
    warp=torch.cat([torch.cat([xy,xy+shift],-1),torch.cat([xy-shift,xy],-1)],1)
    conf=torch.ones(n,2*n)
    r=roma_transfer(warp,conf,[40,50],[0,0,100,100]);assert r['status']=='OK'
    np.testing.assert_allclose(r['point'],[50,50],atol=1e-4);assert r['cycle_px']<1e-4
    batched=roma_transfer(warp[None],conf[None],[40,50],[0,0,100,100])
    np.testing.assert_allclose(batched['point'],r['point'])
    r=roma_transfer(warp,conf*.1,[40,50],[0,0,100,100]);assert r['status']=='FAILURE' and r['point'] is None


def test_frozen_queries_exclude_reference_and_groundtruth():
    from cloth_agent.experiments.action_anchor_transfer import OUT,load
    m=load(OUT/'manifest.json');inp=load(OUT/'inference_inputs.json')
    assert len(m['queries'])==7
    assert all(q['run']!=m['reference']['run'] for q in m['queries'])
    assert all(q['source_match']['node_id']=='S1' for q in m['queries'])
    assert all(set(q)=={'sample_id','image','sha256'} for q in inp['queries'])
