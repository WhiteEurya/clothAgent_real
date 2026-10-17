import copy
import numpy as np
from cloth_agent.experiments.local_node_rejection_e000 import calibrate,run_fold,metrics
from cloth_agent.experiments.local_node_classifier_e000 import unit


def test_calibration_rejects_other_without_rejecting_clear_known():
    rows=[dict(label='S0',prediction='S0',sim_S0=.99,sim_S1=.7,margin=.29),
          dict(label='S1',prediction='S1',sim_S0=.7,sim_S1=.98,margin=.28),
          dict(label='OTHER',prediction='S1',sim_S0=.6,sim_S1=.9,margin=.3)]
    t=calibrate(rows,'similarity_and_margin')
    assert .9<t['sim_threshold']<.98
    assert t['calibration_correct_accepted']==2


def test_outer_query_features_and_labels_do_not_change_calibration():
    samples=[];features=[]
    for r in range(4):
        for l,f in [('S0',[1.,0]),('S1',[0,1.]),('OTHER',[.7,.7])]:
            samples.append(dict(sample_id=f'{r}/{l}',run_id=str(r),label=l,node_label=l));features.append(f)
    features=unit(np.array(features));rows,a=run_fold(features,samples,'3','similarity_and_margin')
    changed=features.copy();changed[-3:]=unit(np.array([[-1.,0],[0,-1.],[-1,-1]]))
    labels=copy.deepcopy(samples)
    for s in labels[-3:]:s['label']='OTHER'
    _,b=run_fold(changed,labels,'3','similarity_and_margin')
    assert a['threshold_selection']==b['threshold_selection']
    assert a['calibration']==b['calibration']
    assert set(a['train_sample_ids']).isdisjoint(a['test_sample_ids'])


def test_missing_negative_calibration_abstains():
    t=calibrate([dict(label='S0',prediction='S0',sim_S0=.99,sim_S1=.5,margin=.49)],'similarity_only')
    assert t['status']=='INSUFFICIENT_CALIBRATION_REJECT_ALL'
    assert t['sim_threshold']>1
