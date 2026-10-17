import numpy as np
from cloth_agent.experiments.local_node_roma_e000 import dense_scores, run_fold


def fixture():
    samples=[dict(sample_id=f'{r}_{c}',run_id=r,label=c,node_label=c) for r in ('a','b','c','d') for c in ('S0','S1','OTHER')]
    matrix=np.array([[.95 if a['label']==b['label'] else .6 for b in samples] for a in samples])
    return samples,matrix


def test_outer_test_does_not_change_calibration():
    samples,matrix=fixture();_,audit=run_fold(matrix,samples,'a')
    matrix[:3,:]=.99;matrix[:,:3]=.99
    samples[0]['label']='OTHER'
    _,changed=run_fold(matrix,samples,'a')
    assert audit['threshold_selection']==changed['threshold_selection']
    assert audit['calibration']==changed['calibration']
    for row in audit['calibration']:
        banks=row['bank_S0']+row['bank_S1']
        assert all(not x.startswith('a_') for x in banks)
        assert all(x.split('_')[0]!=row['sample_id'].split('_')[0] for x in banks)


def test_dense_valid_confidence_and_coverage():
    warp=np.array([[[0,0,0,0],[.1,.1,.1,.1],[np.nan,0,0,0],[1.2,0,0,0]]])
    result=dense_scores(warp,np.array([[.8,.2,.9,.9]]))
    assert result['valid_match_ratio']==.5
    assert result['high_confidence_match_ratio']==.25
    assert result['score']==.25
    assert result['mean_match_confidence']==.5
    assert result['median_match_confidence']==.5
    assert result['workspace_coverage_A']==1/256


def test_test_bank_excludes_other_and_entire_run():
    samples,matrix=fixture();rows,_=run_fold(matrix,samples,'a')
    for row in rows:
        assert len(row['bank_S0'])==3
        assert len(row['bank_S1'])==3
        assert row['output'] in ('SOURCE','TARGET','UNKNOWN')
        assert all(not x.startswith('a_') for x in row['bank_S0']+row['bank_S1'])


def test_incomplete_pairs_fail_closed():
    import pytest
    samples,matrix=fixture();matrix[0,3]=np.nan
    with pytest.raises(ValueError,match='Missing pair score'):
        run_fold(matrix,samples,'a')
