import numpy as np
from cloth_agent.experiments.local_node_classifier_e000 import evaluate,unit


def test_loo_excludes_query_and_other_from_prototype():
    samples=[dict(sample_id=str(i),run_id=str(i),label=l,node_label=l) for i,l in enumerate(['S0','S0','S1','S1','OTHER'])]
    features=unit(np.array([[1.,0],[1.,0],[0,1.],[0,1.],[1,1.]]))
    rows=evaluate(features,samples)
    assert all(r['correct'] for r in rows[:4])
    assert rows[0]['train_S0']==1 and rows[0]['train_S1']==2
    assert rows[-1]['train_S0']==2 and rows[-1]['train_S1']==2
    assert rows[-1]['prediction']=='TIE'


def test_run_holdout_excludes_all_same_run():
    samples=[dict(sample_id=str(i),run_id=str(i//2),label='S0' if i%2==0 else 'S1',node_label='x') for i in range(6)]
    features=np.array([[1.,0],[0,1.]]*3)
    rows=evaluate(features,samples,grouped=True)
    assert len(rows)==6
    assert all(r['correct'] and r['train_S0']==2 and r['train_S1']==2 for r in rows)
