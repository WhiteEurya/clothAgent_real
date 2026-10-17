import copy
import numpy as np
import pytest
from cloth_agent.experiments.local_node_matcha_e000 import aggregate_dense, evaluate_bank


def test_spatial_preserves_layout_global_does_not():
    x=np.zeros((2,8,8),dtype=np.float32);x[0,:,:4]=1;x[1,:,4:]=1
    a=aggregate_dense(x);b=aggregate_dense(x[:,:,::-1])
    assert a['global_mean']@b['global_mean']==pytest.approx(1)
    assert a['spatial_8x8']@b['spatial_8x8']==pytest.approx(0)
    assert a['spatial_8x8']@a['spatial_8x8']==pytest.approx(1)


def fixture():
    samples=[dict(sample_id=f'{r}_{c}',run_id=r,label=c,node_label=c) for r in ('a','b','c','d') for c in ('S0','S1','OTHER')]
    f=np.eye(3,dtype=np.float32)[[0,1,2]*4]
    folds={r:{'train_sample_ids':[s['sample_id'] for s in samples if s['run_id']!=r],
              'test_sample_ids':[s['sample_id'] for s in samples if s['run_id']==r]} for r in ('a','b','c','d')}
    return samples,f,folds


def test_test_run_cannot_affect_its_calibration():
    s,f,folds=fixture();rows,a=evaluate_bank(f,s,folds)
    f[:3]=f[:3][::-1];s[0]['label']='OTHER'
    _,b=evaluate_bank(f,s,folds)
    assert a[0]['calibration']==b[0]['calibration']
    assert a[0]['threshold_selection']==b[0]['threshold_selection']
    assert all(not x.startswith('a_') for x in rows[0]['bank_S0']+rows[0]['bank_S1'])


def test_changed_split_rejected():
    s,f,folds=fixture();folds['a']['test_sample_ids']=[]
    with pytest.raises(ValueError,match='Split differs'):evaluate_bank(f,s,folds)


def test_invalid_dense_fails():
    with pytest.raises(ValueError):aggregate_dense(np.full((3,8,8),np.nan))
