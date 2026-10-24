import copy
import json
from pathlib import Path

from PIL import Image
import pytest

from cloth_agent.experiments import local_experience_transfer_probe as p


def answer():
    return dict(applicable='YES',grasp_px=[30,30],move_direction_px=[1,0],target_region_px=[50,20,70,40])


def dataset(tmp_path):
    paths=[]
    for i in range(6):
        path=tmp_path/f'{i}.png';Image.new('RGB',(100,100),(i*30,15,20)).save(path);paths.append(str(path))
    positive=dict(applicable='YES',rationale='PRIVATE_LABEL_SENTINEL',regions=[dict(polygon_px=[[20,20],[40,20],[40,40],[20,40]],reference_pa=[30,30],move_direction_px=[1,0])])
    negative=dict(applicable='NOT_APPLICABLE',regions=[],rationale='PRIVATE_LABEL_SENTINEL')
    spec=dict(annotation_status='ASSISTANT_PREANNOTATED_PROXY_NOT_HUMAN_VERIFIED',demo=dict(before=paths[0],after=paths[1],pa=[30,30],target=[70,30]),
              queries=[dict(image=paths[i+2],ground_truth=copy.deepcopy(positive if i%2==0 else negative)) for i in range(4)])
    source=tmp_path/'selection.json';p.save(source,spec);out=tmp_path/'dataset';p.prepare(source,out);return out


def test_conditions_have_exact_inputs_no_truth_or_after_leakage(tmp_path):
    root=dataset(tmp_path);m=p.verify_dataset(root)
    for condition,n in [('A',3),('B',4)]:
        prompt,images=p.inputs_for(root,m,'q01',condition)
        assert len(images)==n
        assert all(path.parent==root/'public' for path in images)
        assert 'PRIVATE_LABEL_SENTINEL' not in prompt
        assert ('demo_after.png' in [path.name for path in images])==(condition=='B')
        assert 'reference_pa' not in prompt and 'polygon_px' not in prompt


def test_strict_response_and_direction_metrics():
    p.validate_answer(answer(),(100,100))
    for change in [dict(move_direction_px=[3,0]),dict(grasp_px=[100,30]),dict(target_region_px=[60,20,50,40]),dict(applicable='NO'),dict(grasp_px=[float('nan'),30])]:
        with pytest.raises(ValueError):p.validate_answer(dict(answer(),**change),(100,100))
    p.validate_answer(dict(applicable='AMBIGUOUS',grasp_px=None,move_direction_px=None,target_region_px=None),(100,100))
    assert p.angular_error([1,0],[0,1])==pytest.approx(90)
    assert p.angular_error([1,0],[-1,0])==pytest.approx(180)
    truth=dict(applicable='YES',regions=[dict(polygon_px=[[20,20],[40,20],[40,40],[20,40]],move_direction_px=[1,0])])
    assert p.score(answer(),truth)['pa_in_region'] is True
    assert p.score(dict(answer(),grasp_px=[10,10]),truth)['angular_error_deg'] is None


def test_exactly_eight_calls_errors_not_retried_and_report_generated(tmp_path,monkeypatch):
    root=dataset(tmp_path);out=tmp_path/'run';calls=[]
    def fake(prompt,images,folder,timeout):
        calls.append(str(folder))
        if len(calls)==3:raise RuntimeError('planned transport failure')
        return answer()
    monkeypatch.setattr(p,'invoke_once',fake)
    p.run(root,out);p.run(root,out)
    assert len(calls)==8
    assert len(list(out.glob('*/started.json')))==8
    assert len(p.read(out/'metrics.json')['rows'])==8
    assert (out/'report.md').exists() and (out/'q01_AB.png').exists()
    counts=p.read(out/'metrics.json')['confusion']
    assert sum(c['ERROR'] for c in counts)==1


def test_labels_cannot_change_after_freeze(tmp_path):
    root=dataset(tmp_path)
    with (root/'private/ground_truth.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='labels changed'):p.verify_dataset(root)


def test_interrupted_call_is_consumed_not_repeated(tmp_path,monkeypatch):
    root=dataset(tmp_path);out=tmp_path/'run';(out/'q01_A').mkdir(parents=True)
    calls=[]
    monkeypatch.setattr(p,'invoke_once',lambda *args:(calls.append(1) or answer()))
    p.run(root,out)
    assert len(calls)==7
    assert p.read(out/'q01_A/result.json')['status']=='INTERRUPTED_NOT_RETRIED'
