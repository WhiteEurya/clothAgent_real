import ast
import json
from pathlib import Path
import pytest
from PIL import Image

from cloth_agent.experiments.discovery_data import extract,save,stage_sample,digest,action_record
from cloth_agent.experiments.decision_state_discovery import Experiment,Requests,snapshot_images,montage,report
from cloth_agent.experiments.discovery_schema import validate


def fixture_run(tmp_path,n=4):
    root=tmp_path/'runs'
    for i in range(n):
        d=root/'run1'/f'iteration_{i+1:03d}'
        (d/'before_raw').mkdir(parents=True)
        (d/'after_raw').mkdir()
        Image.new('RGB',(32,24),(10+i,20,30)).save(d/'before_raw/camera_A_rgb_upright.png')
        Image.new('RGB',(32,24),(10+i,20,32)).save(d/'after_raw/camera_A_rgb_upright.png')
        actions=[dict(name='move',args=dict(x=1.,y=2.,z=3.,yaw=0.)),dict(name='close_gripper',args={}),
                 dict(name='move',args=dict(x=4.,y=6.,z=8.,yaw=0.)),dict(name='open_gripper',args={})]
        save(d/'record.json',dict(planned_step='right_sleeve',mode='FOLD',
            execution_proposal=dict(actions=actions,reasoning='FORBIDDEN_REASONING',success='FORBIDDEN_RESULT'),
            execution=dict(physical_execution=True),evaluation=dict(policy='OUTCOME_SENTINEL')))
    return root


def tax(ids):
    # Artificial schema fixture, never a production cloth taxonomy.
    return dict(factors=[dict(id='fixture_factor',name='fixture',description='fixture',possible_values=['v'],
        decision_relevance='fixture',visual_evidence='fixture',uncertainty_condition='fixture')],
        classes=[dict(id='fixture_class',description='fixture',decision_rule='fixture',representative_sample_ids=ids[:1])],
        hypothesis='fixture',representation='INSUFFICIENT',limitations=['synthetic test only'])


def assign(sid,revised=False):
    return dict(sample_id=sid,factor_values=dict(fixture_factor=dict(value='v',confidence=.5,evidence='fixture')),
        **({'state_class':'fixture_class','action_family':'fixture'} if revised else {'provisional_state_class':'fixture_class','next_action_family_without_historical_action':'fixture'}),
        critical_uncertainty=['fixture'],confidence=.5)


class FakeModel:
    def __init__(self):self.requests=[]
    def invoke(self,prompt,images,schema):
        payload=json.loads(prompt.split('Payload:\n')[1].split('\nSchema:\n')[0])
        ids=payload['sample_ids'];self.requests.append((prompt,list(images),payload))
        assert all(Path(p).is_file() for p in images)
        stage=prompt.split('Stage: ')[1].split('.')[0]
        kind=prompt.split('Operation: ')[1].split('.')[0]
        if stage=='image_only' and kind=='classify':
            v=dict(covered_sample_ids=ids,assignments=[assign(s) for s in ids])
        elif stage=='image_only':
            v=dict(covered_sample_ids=ids,taxonomy=tax(ids),ambiguous_samples=ids[:1],irrelevant_visual_variation=[],unresolved_disagreements=[])
        else:
            v=dict(covered_sample_ids=ids,taxonomy=tax(ids),assignments=[assign(s,True) for s in ids],operations=[],unresolved_disagreements=[])
            if stage=='action':v.update(class_review=[],appearance_different_action_equivalent=[],appearance_similar_action_different=[])
            else:v.update(transition_review=[],research_answers={f'Q{i}':dict(answer='insufficient',evidence_sample_ids=ids,confidence=.1,limitations=['fixture']) for i in range(1,7)},best_supported_representation='insufficient',negative_results=['fixture negative result'])
        return dict(text=json.dumps(v),usage=dict(input_tokens=10,output_tokens=5),elapsed_s=.01)


def experiment(tmp_path,n=4,size=2):
    root=fixture_run(tmp_path,n);m=extract([root]);out=tmp_path/'out';out.mkdir()
    save(out/'manifest.json',m);snapshot_images(m,out)
    backend=FakeModel();requests=Requests(out,'gpt-6-astra',backend)
    return Experiment(m,out,requests,size,7),backend


def test_payload_leakage_and_stage3_full_evidence(tmp_path):
    e,b=experiment(tmp_path)
    for stage in ('image_only','action','outcome'):
        p,images=e.payload(e.ids[:1],stage);s=p['samples'][0]
        assert set(s)==({'sample_id'} if stage=='image_only' else {'sample_id','action'} if stage=='action' else {'sample_id','action','outcome','after_observation_available'})
        text=json.dumps(p)
        assert 'FORBIDDEN_REASONING' not in text and 'FORBIDDEN_RESULT' not in text
        assert ('OUTCOME_SENTINEL' in text)==(stage=='outcome')
        assert len(images)==(2 if stage=='outcome' else 1)
        assert all('run1' not in str(p) for p in p['samples'])
    assert e.samples[e.ids[0]]['action']['normalized']['delta']==[3.,4.,5.]
    assert e.samples[e.ids[0]]['action']['normalized']['garment_relative'] is None


def test_three_stages_multimodal_coverage_versions_and_cache(tmp_path):
    e,b=experiment(tmp_path,7,3)
    e.image_only();v0=(e.output/'stage1/taxonomy_v0.json').read_bytes()
    frozen=e.frozen('stage1/assignments.json')
    assert {a['sample_id'] for a in frozen}==set(e.ids)
    assert set(e.frozen('stage1/aggregate.json')['covered_sample_ids'])==set(e.ids)
    e.reveal('action');e.reveal('outcome')
    assert (e.output/'stage1/taxonomy_v0.json').read_bytes()==v0
    for v in [1,2]:assert (e.output/f'stage{v+1}/taxonomy_v{v}.json').exists()
    assert all(imgs for _,imgs,_ in b.requests)
    for p,images,payload in b.requests:
        if 'Stage: image_only.' in p or 'Stage: action.' in p:assert 'OUTCOME_SENTINEL' not in p
        if 'Stage: image_only.' in p:assert 'normalized' not in json.dumps(payload)
        if 'Operation: aggregate.' not in p:assert len(images)<=3
    r=report(e.manifest,e.output)
    assert r['stages']['stage3']['negative_results']==['fixture negative result']
    count=len(b.requests);e.image_only();e.reveal('action');e.reveal('outcome')
    assert len(b.requests)==count and e.requests.hits>0
    with pytest.raises(ValueError,match='overwrite'):e.freeze('stage1/taxonomy_v0.json',tax([]))


def test_missing_fields_are_stage_specific(tmp_path):
    root=fixture_run(tmp_path)
    dirs=sorted(root.glob('*/iteration_*'))
    (dirs[0]/'before_raw/camera_A_rgb_upright.png').unlink()
    d=json.loads((dirs[1]/'record.json').read_text());d.pop('execution_proposal');save(dirs[1]/'record.json',d)
    d=json.loads((dirs[2]/'record.json').read_text());d.pop('evaluation');d.pop('execution');save(dirs[2]/'record.json',d)
    (dirs[2]/'after_raw/camera_A_rgb_upright.png').unlink()
    m=extract([root]);rows=m['samples']
    assert not rows[0]['eligible']['image_only']
    assert rows[1]['eligible']['image_only'] and not rows[1]['eligible']['action']
    assert rows[2]['eligible']['action'] and not rows[2]['eligible']['outcome']
    assert 'physical_execution_not_confirmed' in rows[2]['issues']
    assert m['stage_counts']==dict(image_only=3,action=2,outcome=1)


def test_cache_keys_depend_on_image_and_taxonomy_and_no_cache(tmp_path):
    e,b=experiment(tmp_path,1)
    p,images=e.payload(e.ids,'image_only')
    r=e.requests
    r.ask('image_only','discover',p,images,'taxA');r.ask('image_only','discover',p,images,'taxA')
    assert len(b.requests)==1
    r.ask('image_only','discover',p,images,'taxB');assert len(b.requests)==2
    Image.new('RGB',(32,24),'white').save(images[0][1])
    r.ask('image_only','discover',p,images,'taxB');assert len(b.requests)==3
    r.no_cache=True;r.ask('image_only','discover',p,images,'taxB');assert len(b.requests)==4


def test_unknown_and_new_class_allowed_but_ids_must_match():
    t=tax(['s']);a=assign('s');a['provisional_state_class']='NEW_CLASS_CANDIDATE'
    v=dict(covered_sample_ids=['s'],assignments=[a]);validate(v,'image_only','classify',['s'],t)
    a['sample_id']='invented'
    with pytest.raises(ValueError,match='assignments'):validate(v,'image_only','classify',['s'],t)


def test_montage_labels_only_ids_and_does_not_modify_images(tmp_path):
    e,b=experiment(tmp_path,1)
    from cloth_agent.experiments.discovery_data import file_hash
    before=file_hash(e.images[e.ids[0]]);p=e.output/'sheet.png'
    montage(e.ids,e.images,p)
    labels=read_json(p.with_suffix('.labels.json'))
    assert labels==dict(labels=e.ids,annotation='sample_id only')
    assert file_hash(e.images[e.ids[0]])==before


def read_json(p):return json.loads(p.read_text())


def test_format_retries_bounded_and_no_fabricated_taxonomy(tmp_path):
    e,b=experiment(tmp_path,1)
    class Bad:
        def invoke(self,*args):return dict(text='{}')
    e.requests.backend=Bad();e.requests.repairs=1
    with pytest.raises(ValueError,match='bounded repair'):e.image_only()
    assert e.requests.calls==2 and not (e.output/'stage1/taxonomy_v0.json').exists()


def test_freeze_hash_detects_tampering(tmp_path):
    e,b=experiment(tmp_path,1);e.image_only()
    save(e.output/'stage1/assignments.json',[])
    with pytest.raises(ValueError,match='tampered'):e.reveal('action')


def test_stage_dependencies_and_dry_run_no_model(tmp_path):
    from cloth_agent.experiments.decision_state_discovery import main
    root=fixture_run(tmp_path,1);out=tmp_path/'dry'
    assert main(['--runs',str(root),'--output',str(out),'--dry-run'])==0
    assert read_json(out/'report.json')['model_calls']==0
    assert not (out/'stage1').exists()
    with pytest.raises(SystemExit):main(['--runs',str(root),'--output',str(out),'--execute'])


def test_experiment_imports_no_robot_runtime():
    forbidden={'robot_api','codex_loop','codex_native_runtime','xarm','pyrealsense2','kinematics'}
    root=Path(__file__).parents[1]/'cloth_agent/experiments'
    for path in root.glob('*.py'):
        # Separate physical Seed/Replay experiment; its compile/import boundary
        # is guarded independently in test_graph_policy.py. Discovery remains offline.
        if path.name in {'graph_policy.py', 'graph_policy_recovery.py'}:continue
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert not set((node.module or '').split('.'))&forbidden
            if isinstance(node,ast.Import):
                for x in node.names:assert not set(x.name.split('.'))&forbidden


def test_isolated_multimodal_command_has_no_host_data_or_session(tmp_path,monkeypatch):
    from cloth_agent.experiments.discovery_model import isolated_command
    import cloth_agent.experiments.discovery_model as module
    monkeypatch.setattr(module.shutil,'which',lambda _: '/usr/bin/bwrap')
    cmd=isolated_command('/opt/fakecodex',tmp_path)
    assert '--tmpfs' in cmd and '--unshare-pid' in cmd
    assert '/home/sja/clothAgent_real' not in cmd and '/mnt/newssd' not in cmd
    assert '--ro-bind' in cmd and '--bind' in cmd


def test_backend_reuses_multimodal_flags_and_sanitizes_pixels(tmp_path,monkeypatch):
    from cloth_agent.experiments.discovery_model import CodexMultimodal
    import cloth_agent.experiments.discovery_model as module
    from PIL.PngImagePlugin import PngInfo
    img=tmp_path/'outcome_success.png';meta=PngInfo();meta.add_text('note','FORBIDDEN_METADATA')
    Image.new('RGB',(8,8),'red').save(img,pnginfo=meta)
    captured=[]
    class Process:
        returncode=0
        def __init__(self,command,stdin,stdout,stderr,**kwargs):
            assert kwargs['env']=={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8'}
            captured.append(command);self.work=Path(command[command.index('--bind')+1]);self.stdout=stdout
        def communicate(self,prompt,timeout):
            assert prompt=='stage1-only'
            with Image.open(self.work/'image_000.png') as im:assert not im.info
            (self.work/'answer.txt').write_text('{}')
            self.stdout.write(json.dumps(dict(type='turn.completed',usage={'input_tokens':1}))+'\n');self.stdout.flush()
    monkeypatch.setattr(module.subprocess,'Popen',Process)
    model=CodexMultimodal(binary='/bin/true')
    result=model.invoke('stage1-only',[img],{'type':'object'})
    command=captured[0]
    assert '--image' in command and command[command.index('--image')+1]=='/work/image_000.png'
    assert 'resume' not in command and '--ephemeral' in command
    assert '--ignore-user-config' in command and 'features.shell_tool=false' in command
    assert str(img) not in command and result['image_count']==1


def test_malformed_records_are_skipped_and_probe_not_fold(tmp_path):
    root=fixture_run(tmp_path,3)
    dirs=sorted(root.glob('*/iteration_*'))
    save(dirs[0]/'record.json',['invalid'])
    d=read_json(dirs[1]/'record.json');d['mode']='ACQUISITION_PROBE';save(dirs[1]/'record.json',d)
    m=extract([root]);assert len(m['selected_sample_ids'])==1
    assert any('invalid_record' in issue for s in m['samples'] for issue in s['issues'])


def test_later_stage_without_physical_evidence_does_not_invent_result(tmp_path):
    e,b=experiment(tmp_path,1);e.image_only()
    for s in e.samples.values():s['eligible']['outcome']=False
    e.reveal('action');count=len(b.requests);e.reveal('outcome')
    assert len(b.requests)==count and read_json(e.output/'stage3/status.json')['status']=='UNAVAILABLE'
    assert not (e.output/'stage3/taxonomy_v2.json').exists()


def test_static_goal_in_every_request_without_historical_leakage(tmp_path):
    e, b = experiment(tmp_path, 4, 2)
    contract = e.manifest['task_contract']
    assert contract['status'] == 'AVAILABLE'
    assert contract['sha256'] == digest(contract['goal'])
    assert 'never wearer anatomy' in contract['goal']['side_convention']
    assert 'lies over/inboard on the torso' in contract['goal']['completion_criterion']
    assert len(contract['sources']) == 3
    e.image_only(); e.reveal('action'); e.reveal('outcome')
    kinds = set()
    for prompt, images, payload in b.requests:
        assert payload['task_goal'] == contract['goal']
        assert 'sources' not in payload['task_goal']
        assert 'cloth_agent/' not in prompt
        assert 'FORBIDDEN_REASONING' not in prompt
        if 'Stage: outcome.' not in prompt:
            assert 'OUTCOME_SENTINEL' not in prompt
        kinds.add(prompt.split('Operation: ')[1].split('.')[0])
    assert kinds == {'discover', 'classify', 'aggregate', 'review'}


def test_goal_change_invalidates_cache_and_tampering_rejected(tmp_path):
    e, b = experiment(tmp_path, 1)
    p, images = e.payload(e.ids, 'image_only')
    e.requests.ask('image_only', 'discover', p, images)
    p = json.loads(json.dumps(p))
    p['task_goal']['objective'] += ' test change'
    e.requests.ask('image_only', 'discover', p, images)
    assert len(b.requests) == 2
    e.manifest['task_contract']['goal']['objective'] += ' tampered'
    with pytest.raises(ValueError, match='hash/subgoal'):
        e.payload(e.ids, 'image_only')


def test_unsupported_goal_no_model_calls(tmp_path):
    e, b = experiment(tmp_path, 1)
    from cloth_agent.experiments.discovery_goal import task_contract
    e.manifest['selected_subgoal'] = 'unreviewed_goal'
    e.manifest['task_contract'] = task_contract('unreviewed_goal')
    with pytest.raises(ValueError, match='Task goal unavailable'):
        e.image_only()
    assert not b.requests


def test_old_output_refused_before_modification(tmp_path):
    from cloth_agent.experiments.decision_state_discovery import main
    root = fixture_run(tmp_path, 1); out = tmp_path/'old'
    save(out/'config.json', {'protocol': 'decision-discovery-v1'})
    save(out/'stage1/taxonomy_v0.json', {'original': True})
    before = {str(p): p.read_bytes() for p in out.rglob('*') if p.is_file()}
    with pytest.raises(SystemExit):
        main(['--runs', str(root), '--output', str(out), '--dry-run'])
    assert before == {str(p): p.read_bytes() for p in out.rglob('*') if p.is_file()}


def test_goal_missing_source_fails_closed_and_static_source_not_imported(tmp_path):
    from cloth_agent.experiments.discovery_goal import task_contract
    assert task_contract('right_sleeve', tmp_path)['status'] == 'UNAVAILABLE'
    left = task_contract('left_sleeve')
    assert left['goal']['objective'] == 'garment-frame left sleeve inward'


def test_execution_focus_changes_only_stage1_instructions(tmp_path):
    e, b = experiment(tmp_path, 4, 2)
    e.image_only(); e.reveal('action'); e.reveal('outcome')
    baseline = list(b.requests)
    b2 = FakeModel()
    e2 = Experiment(e.manifest, e.output, Requests(tmp_path/'variant_cache', 'gpt-6-astra', b2,
                    stage1_focus='execution'), 2, 7)
    e2.image_only(); e2.reveal('action'); e2.reveal('outcome')
    assert len(baseline) == len(b2.requests)
    for (old, images0, payload0), (new, images1, payload1) in zip(baseline, b2.requests):
        assert images0 == images1 and payload0 == payload1
        assert old.split('\nSchema:\n')[1] == new.split('\nSchema:\n')[1]
        if 'Stage: image_only.' in old:
            assert old != new and 'execution' in new
            assert 'OUTCOME_SENTINEL' not in new and 'FORBIDDEN_REASONING' not in new
        else:
            assert old == new


def test_reuse_manifest_never_extracts_and_preserves_exact_dataset(tmp_path, monkeypatch):
    import cloth_agent.experiments.decision_state_discovery as module
    e, b = experiment(tmp_path, 3)
    source = e.output/'manifest.json'
    save(source.with_suffix('.sha256.json'), {'sha256': digest(e.manifest)})
    def forbidden(*args):
        raise AssertionError('must not rescan or select data')
    monkeypatch.setattr(module, 'extract', forbidden)
    out = tmp_path/'reuse'
    argv = ['--runs', *e.manifest['roots'], '--subgoal', 'right_sleeve', '--output', str(out),
            '--reuse-manifest', str(source), '--stage1-focus', 'execution', '--dry-run']
    assert module.main(argv) == 0
    assert read_json(out/'manifest.json') == e.manifest
    assert read_json(out/'manifest.sha256.json')['sha256'] == digest(e.manifest)
    assert read_json(out/'report.json')['model_calls'] == 0
    save(source.with_suffix('.sha256.json'), {'sha256': 'tampered'})
    with pytest.raises(SystemExit):module.main(argv)
