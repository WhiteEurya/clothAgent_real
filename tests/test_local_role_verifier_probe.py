import copy
from pathlib import Path

import jsonschema
from PIL import Image
import pytest

from cloth_agent.experiments import local_role_verifier_probe as p


def prepared(tmp_path):
    source=tmp_path/'previous';(source/'dataset/public').mkdir(parents=True)
    p.save(source/'dataset/manifest.json',dict(demo_pa=[300,200],demo_target=[470,180]))
    for i,name in enumerate(['demo_before','demo_after','q01','q03','q04']):
        Image.new('RGB',(800,600),(i*35,40,80)).save(source/f'dataset/public/{name}.png')
    for qid in p.IDS:
        a=dict(applicable='YES',grasp_px=[350,490],move_direction_px=[.6,.8],target_region_px=[400,510,450,560])
        p.save(source/f'run/{qid}_A/result.json',dict(status='OK',answer=a))
    root=tmp_path/'probe';p.prepare(source,root)
    return source,root


def test_freezes_candidates_and_conditions_have_identical_local_images(tmp_path):
    source,root=prepared(tmp_path);m=p.verify(root)
    assert p.snapshot(source)==p.read(root/'private/source_snapshot.json')
    for qid in p.IDS:
        assert m['frozen_candidates'][qid]['answer']==p.read(source/f'run/{qid}_A/result.json')['answer']
        local=p.request_images(root,qid,'L');global_=p.request_images(root,qid,'G')
        assert len(local)==3 and len(global_)==6 and local==global_[:3]
        for _,path in local:
            with Image.open(path) as im:assert im.size==(320,320)
        for c in ('L','G'):
            text=(root/f'requests/{qid}_{c}/prompt.txt').read_text()
            assert 'q04' not in text and 'torso' not in text and 'sleeve' not in text
            assert '用户' not in text and 'expected' not in text


def test_crop_same_native_scale_and_padding():
    im=Image.new('RGB',(400,400),(100,20,30));box=p.crop_box([200,350]);crop=p.crop_native(im,box)
    assert box==[40,190,360,510]
    assert crop.getpixel((160,160))==(100,20,30)
    assert crop.getpixel((160,319))==(75,75,75)


def test_strict_two_field_contract_and_verdict():
    a=dict(same_manipulation_role='YES',same_effect_likely='YES')
    jsonschema.validate(a,p.SCHEMA);assert p.verdict(a)=='ACCEPT'
    assert p.verdict(dict(a,same_effect_likely='UNCERTAIN'))=='UNCERTAIN'
    assert p.verdict(dict(a,same_manipulation_role='NO',same_effect_likely='UNCERTAIN'))=='REJECT'
    with pytest.raises(jsonschema.ValidationError):jsonschema.validate(dict(a,grasp_px=[10,10]),p.SCHEMA)


def test_six_calls_no_repair_no_repeat_and_no_source_changes(tmp_path,monkeypatch):
    source,root=prepared(tmp_path);before=p.snapshot(source);calls=[]
    def fake(prompt,images,folder,timeout):
        calls.append(folder.name)
        if len(calls)==2:raise RuntimeError('test transport error')
        if len(calls)==3:return dict(same_manipulation_role='YES',same_effect_likely='YES',new_pa=[1,2])
        return dict(same_manipulation_role='YES',same_effect_likely='YES')
    monkeypatch.setattr(p,'invoke_once',fake)
    p.run(root);p.run(root)
    assert len(calls)==6 and p.snapshot(source)==before
    metrics=p.read(root/'metrics.json')
    assert metrics['attempted_calls']==6 and metrics['previous_experiment_unchanged']
    assert sum(r['verdict']=='ERROR' for r in metrics['rows'])==2
    assert (root/'report.md').exists() and (root/'LG_results.png').exists()


def test_protocol_or_label_mutation_rejected_before_calls(tmp_path):
    _,root=prepared(tmp_path)
    with (root/'private/evaluation_labels.json').open('a') as f:f.write(' ')
    with pytest.raises(ValueError,match='frozen labels'):p.verify(root)
