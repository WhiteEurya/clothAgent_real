import json
from pathlib import Path
import pytest
from cloth_agent.experiments.controlled_outcome_audit import select_pairs,PRIORITY_Z,clean_event,action_difference,gather_sample,model_sample,report
from cloth_agent.experiments.discovery_data import save,file_hash
from cloth_agent.experiments.discovery_schema import validate
from cloth_agent.experiments.outcome_pair_schema import instruction_for
from PIL import Image


def action(z=10,tx=20,gx=0):
    return dict(grasp=[gx,0,z],target=[tx,10,20],raw=[dict(name='move',args=dict(x=gx,y=0,z=z,yaw=0)),dict(name='close_gripper',args={}),dict(name='move',args=dict(x=tx,y=10,z=20,yaw=0)),dict(name='open_gripper',args={})])


def test_pair_selection_is_outcome_blind_and_priority_order():
    ids=list(dict.fromkeys(x for p in PRIORITY_Z for x in p))
    m=dict(samples=[dict(sample_id=s,action=action(i+1),outcome='DO_NOT_USE') for i,s in enumerate(ids)])
    r=dict(appearance_similar_action_different=[dict(samples=list(p),evidence='Stage2',uncertainty='') for p in reversed(PRIORITY_Z)])
    pairs,skipped=select_pairs(m,r)
    assert [p['samples'] for p in pairs]==[list(p) for p in PRIORITY_Z]
    assert all(p['pair_type']=='GRASP_Z_PAIR' for p in pairs) and not skipped
    assert 'DO_NOT_USE' not in json.dumps(pairs)
    assert action_difference(action(),action(15))['grasp_difference_mm']==[0,0,5]


def test_simulated_feedback_not_physical():
    assert clean_event(dict(robot_state=dict(state='simulated'))) is None
    assert clean_event(dict(robot_state=dict(gripper_feedback=dict(simulated=True)))) is None
    e=clean_event(dict(name='close_gripper',success=True,robot_state=dict(gripper_feedback=dict(mechanical_grasp_detected=False,position_pulse=6))))
    assert e['gripper_feedback']['position_pulse']==6
    assert 'acquisition' not in e


def test_evidence_gathering_missing_after_simulation_and_policy(tmp_path):
    image=tmp_path/'before.png';Image.new('RGB',(20,20)).save(image)
    save(tmp_path/'execution.json',dict(physical_execution=True,actual_robot_actions=[dict(name='move',robot_state=dict(state='simulated')),dict(name='close_gripper',success=True,robot_state={})]))
    save(tmp_path/'evaluation.json',dict(grasp_acquisition=dict(status='FAILURE',evidence=['UNCHANGED policy']),next_experiment={'change':'LEAK_FUTURE'}))
    s=dict(sample_id='s',run_id='r',iteration_id='i',source_directory=str(tmp_path),before_rgb=str(image),after_rgb=None,image_hashes={'before':file_hash(image)})
    row=gather_sample(s,dict(action=action()),tmp_path/'stage3a')
    ex=next(x for x in row['evidence'] if x['source_id']=='s.actual_execution')
    assert ex['content']['excluded_simulated_events']==1
    assert len(ex['content']['actual_robot_actions'])==1
    assert any('after' in x for x in row['missing'])
    p=model_sample(row);text=json.dumps(p)
    assert 'LEAK_FUTURE' not in text and 'provenance' not in text
    assert 'UNCHANGED policy' in text and 'fallible_evaluator' in text
    assert 'acquisition' not in row  # Host never labels a miss from UNCHANGED.


def judgment(status):
    return dict(status=status,evidence=[dict(source_id='s.availability',observation='missing')],confidence=.1,limitations=[])


def test_independent_sample_citations_and_unknown_allowed():
    v=dict(covered_sample_ids=['s'],sample_id='s',acquisition=judgment('UNKNOWN'),task_progress=judgment('UNKNOWN'),completion=judgment('UNKNOWN'),side_effect=judgment('UNKNOWN'),execution_reach=judgment('UNKNOWN'),evaluator_agreement='unknown',conflicting_evidence=[])
    validate(v,'outcome_pair','sample',['s'],evidence_ids=['s.availability'])
    v['acquisition']['evidence'][0]['source_id']='invented'
    with pytest.raises(ValueError,match='source_id'):validate(v,'outcome_pair','sample',['s'],evidence_ids=['s.availability'])
    assert 'UNCHANGED does NOT imply empty grasp' in instruction_for('pair')
    assert 'Deeper commanded Z is NOT automatically better' in instruction_for('sample')


def test_unknown_outcomes_cannot_be_called_same():
    from cloth_agent.experiments.outcome_pair_schema import validate_evidence
    v=dict(classification='NO_OUTCOME_DIFFERENCE',acquisition_difference=dict(status='UNKNOWN'),task_progress_difference=dict(status='SAME'))
    with pytest.raises(ValueError,match='sameness'):validate_evidence(v,'pair',['a','b'],[])


def test_unavailable_groups_report_does_not_invent_positive_effect(tmp_path):
    d=tmp_path/'stage3a';d.mkdir()
    m=dict(samples=[],pairs=[])
    r=report(m,tmp_path)
    assert r['status']=='PREPARED'
    text=(d/'report.md').read_text()
    assert 'No supported parameter-effect association' in text
    assert not (d/'pairs.json').exists()


def test_independent_samples_then_pair_runner_and_resume(tmp_path):
    from cloth_agent.experiments.controlled_outcome_audit import run
    from cloth_agent.experiments.decision_state_discovery import Requests
    class Backend:
        def __init__(self):self.payloads=[]
        def invoke(self,prompt,images,schema):
            p=json.loads(prompt.split('Payload:\n')[1].split('\nSchema:\n')[0]);self.payloads.append(p);ids=p['sample_ids']
            def j(status):return dict(status=status,evidence=[dict(source_id=ids[0]+'.availability',observation='fixture')],confidence=.1,limitations=[])
            if len(ids)==1:
                assert 'pair' not in p and 'independent_sample_assessments' not in p
                v=dict(covered_sample_ids=ids,sample_id=ids[0],acquisition=j('UNKNOWN'),task_progress=j('UNKNOWN'),completion=j('UNKNOWN'),side_effect=j('UNKNOWN'),execution_reach=j('UNKNOWN'),evaluator_agreement='unknown',conflicting_evidence=[])
            else:
                assert len(p['independent_sample_assessments'])==2
                v=dict(covered_sample_ids=ids,pair_id=p['pair']['pair_id'],pair_type=p['pair']['pair_type'],before_comparability=j('UNKNOWN'),changed_action_variables=[],acquisition_difference=j('UNKNOWN'),task_progress_difference=j('UNKNOWN'),parameter_effect=j('UNRESOLVED'),visible_state_difference=j('UNKNOWN'),classification='UNRESOLVED',classification_evidence=j('UNKNOWN')['evidence'],evidence_strength='NONE',multiple_actions_feasible=j('UNKNOWN'),rgb_state_aliasing=j('UNKNOWN'),confounders=[],conclusion='fixture',limitations=[])
            return dict(text=json.dumps(v),elapsed_s=.01)
    rows=[dict(sample_id=s,evidence=[dict(source_id=s+'.availability',kind='evidence_inventory',content={})],images=[]) for s in ['a','b']]
    m=dict(samples=rows,pairs=[dict(samples=['a','b'],pair_id='p1',pair_type='GRASP_Z_PAIR')],task_goal={},frozen_taxonomy_hash='frozen',stage2_review_sha256='fixed',protocol='test')
    b=Backend();r=Requests(tmp_path/'stage3a','gpt-6-astra',b,max_calls=3)
    out=run(m,tmp_path,r,workers=2)
    assert r.calls==3 and out['pairs'][0]['assessment']['classification']=='UNRESOLVED'
    run(m,tmp_path,r,workers=2)
    assert r.calls==3 and r.hits==3
    assert not (tmp_path/'stage1').exists() and not (tmp_path/'stage2').exists()
