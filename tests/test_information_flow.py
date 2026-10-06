import copy
from types import SimpleNamespace

import pytest
from PIL import Image

from cloth_agent.auto_exploration import VISUAL_PLAN_JSON_SCHEMA
from cloth_agent.harness.common import read_json
from cloth_agent.harness.information_flow import ObservationHost, run_information_flow
from cloth_agent.harness.information_probe import ROOT
from cloth_agent.harness.planning_probe import run_planning_arm
from cloth_agent.harness.policy import PolicyError


@pytest.fixture
def setup_case(tmp_path):
    images = []
    for i in range(3):
        p = tmp_path/f'root_{i}.png'
        im = Image.new('RGB', (40, 60))
        im.putdata([(x*5, y*3, i*60) for y in range(60) for x in range(40)])
        im.save(p)
        images.append(p)
    obs = {'images': [{'image_id':f'image_{i}', 'role':role, 'size':[40,60]}
                       for i,role in enumerate(('clean','overlay','reference'))]}
    case = {'observation':obs, 'context_hash':'shared', 'fold_goal':'left_side',
            'prompt':'Select for current fold only', 'system_prompt':'Visual planner',
            'schema':VISUAL_PLAN_JSON_SCHEMA,
            'context':{'approved_skill_names':[], 'rejected_references':[], 'objective':'fold'},
            'registry':{'binding':'RAW_RGB_HASH_VERIFIED', 'candidates':[
                {'camera':'A', 'candidate_id':'R001', 'pixel_xy':[15,25]}]}}
    artifact = read_json(ROOT/'data/skills/experimental/visual_information.json')
    return case, images, artifact


def info(known=False, sources=None):
    return [{'id':'boundary', 'need':'Identify the task-relevant boundary',
             'status':'KNOWN' if known else 'UNKNOWN',
             'finding':'No visible seam at this resolved boundary' if known else '',
             'missing_information':'' if known else 'Boundary direction is ambiguous in this orientation',
             'source_image_ids':sources if sources is not None else ['image_0']}]


def request(skill='orientation', roi=None):
    return {'gap_id':'boundary', 'skill_id':skill, 'source_image_id':'image_0',
            'roi':roi, 'degrees_clockwise':90 if skill=='orientation' else 0,
            'enlarge':False, 'expected_information_gain':'Resolve the visible boundary orientation'}


def plan():
    return {'garment_observation':'Visible cloth boundary', 'opening_strategy':'Fold inward',
            'confidence':0.7, 'selected_reference':{'camera':'A','reference_id':'R001','reason':'Visible boundary'},
            'motion_intent':'Lift and fold inward', 'expected_observation':'Side narrows',
            'safety_notes':['Ground and check IK before execution']}


def global_result(requests=None, known=False):
    return {'information':info(known), 'observation_requests':[request()] if requests is None else requests}


def decision(status='CANDIDATE', requests=None, sources=None):
    return {'status':status, 'information':info(status=='CANDIDATE', sources),
            'observation_requests':requests or [], 'plan':plan() if status=='CANDIDATE' else None}


class Model:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.requests = []

    def invoke(self, **kwargs):
        self.calls.append({'stage':kwargs['stage']})
        self.requests.append({**kwargs, 'images':list(kwargs['images'])})
        assert self.responses, 'Unexpected extra model invocation'
        return copy.deepcopy(self.responses.pop(0))


def run(tmp_path, setup_case, model, **kwargs):
    case, images, artifact = setup_case
    return run_information_flow(case, images, artifact, tmp_path/'flow', model=model, model_name='test', **kwargs)


def test_default_dispatch_integrates_global_understanding_and_direct_selection(tmp_path, setup_case):
    case, images, artifact = setup_case
    model = Model(global_result(), decision(sources=['image_3']))
    def forbidden(**kwargs):
        raise AssertionError('Old planner must not be invoked after understanding')
    report = run_planning_arm(case, images, artifact, tmp_path/'flow', use_skill=True, model=model,
                              backend=SimpleNamespace(invoke=forbidden), model_name='test')
    assert report['status']=='COMPLETED'
    assert report['model_invocations']==2
    assert [r['stage'] for r in model.requests]==['global_understanding','candidate_selection']
    assert report['separate_binding_calls']==report['standalone_verification_calls']==0
    assert len(model.requests[0]['images'])==3
    assert len(model.requests[1]['images'])==5
    bundle = read_json(tmp_path/'flow/selection_bundle_0.json')
    assert bundle['current_goal']=='left_side'
    assert bundle['information'][0]['status']=='UNKNOWN'  # execution is not semantic proof
    assert bundle['observation_results'][0]['success_check']
    assert bundle['images'][3]['original_image_id']=='image_0'
    assert bundle['images'][3]['to_original']==[0,1,0,-1,0,59]
    assert Image.open(model.requests[1]['images'][3]).size==(60,40)
    assert report['information'][0]['finding'].startswith('No visible seam')  # negative finding can suffice
    assert (tmp_path/'flow/result.json').exists()


def test_no_gap_skips_host_observation(tmp_path, setup_case):
    report = run(tmp_path, setup_case, Model(global_result([],True), decision()))
    assert report['status']=='COMPLETED' and report['host_image_ops']==0
    assert report['model_invocations']==2


def test_unresolvable_gap_exits_before_selection(tmp_path, setup_case):
    report = run(tmp_path, setup_case, Model(global_result([])))
    assert report['status']=='UNKNOWN' and report['stop_reason']=='NO_SUPPORTED_OBSERVATION'
    assert report['model_invocations']==1
    assert not (tmp_path/'flow/result.json').exists()


def test_one_specific_supplement_then_selection(tmp_path, setup_case):
    model = Model(global_result(), decision('NEED_MORE',[request('overlay_occlusion',[.1,.2,.7,.8])]),
                  decision(sources=['image_5']))
    report = run(tmp_path, setup_case, model)
    assert report['status']=='COMPLETED' and report['supplements']==1
    assert report['host_image_ops']==4 and report['model_invocations']==3
    assert len(model.requests[-1]['images'])==7
    assert read_json(tmp_path/'flow/selection_bundle_1.json')['remaining_supplements']==0


@pytest.mark.parametrize('max_supplements,reason',[(0,'SUPPLEMENT_BUDGET_EXHAUSTED'),(1,'REPEATED_OBSERVATION')])
def test_repeated_or_disallowed_supplement_exits_unknown(tmp_path, setup_case, max_supplements, reason):
    report = run(tmp_path, setup_case, Model(global_result(), decision('NEED_MORE',[request()])),
                 max_supplements=max_supplements)
    assert report['status']=='UNKNOWN' and report['stop_reason']==reason
    assert report['model_invocations']==2 and report['host_image_ops']==2
    assert not (tmp_path/'flow/result.json').exists()


def test_second_supplement_is_not_executed(tmp_path, setup_case):
    model = Model(global_result(), decision('NEED_MORE',[request('local_boundary',[.1,.2,.7,.8])]),
                  decision('NEED_MORE',[request('local_boundary',[.2,.3,.8,.9])]))
    report = run(tmp_path, setup_case, model)
    assert report['status']=='UNKNOWN' and report['stop_reason']=='SUPPLEMENT_BUDGET_EXHAUSTED'
    assert report['model_invocations']==3 and report['host_image_ops']==3


def test_budget_preflight_makes_no_partial_batch(tmp_path, setup_case):
    report = run(tmp_path, setup_case, Model(global_result()), max_host_ops=1)
    assert report['status']=='UNKNOWN' and report['stop_reason']=='OBSERVATION_BUDGET_EXHAUSTED'
    assert report['host_image_ops']==0


@pytest.mark.parametrize('fault',['unknown_candidate','dropped_need','fabricated_image','rejected_candidate','false_known','unknown_exit'])
def test_invalid_semantic_transitions_cannot_return_candidate(tmp_path, setup_case, fault):
    result = decision()
    if fault=='unknown_candidate': result['information']=info()
    if fault=='dropped_need': result['information'][0]['id']='different'
    if fault=='fabricated_image': result['information'][0]['source_image_ids']=['image_999']
    if fault=='rejected_candidate': setup_case[0]['context']['rejected_references']=[{'camera':'A','reference_id':'R001'}]
    if fault=='false_known': result['information'][0]['source_image_ids']=[]
    if fault=='unknown_exit': result=decision('UNKNOWN');result['plan']=plan()
    report = run(tmp_path, setup_case, Model(global_result(),result))
    assert report['status']=='FAILED'
    assert not (tmp_path/'flow/result.json').exists()


def test_explicit_selection_unknown_is_terminal(tmp_path, setup_case):
    report = run(tmp_path, setup_case, Model(global_result(),decision('UNKNOWN')))
    assert report['status']=='UNKNOWN' and report['model_invocations']==2
    assert report['information'][0]['missing_information']


def test_host_validates_whole_batch_and_equivalent_crop_repetition(tmp_path, setup_case):
    case, images, artifact = setup_case
    host = ObservationHost(case['observation'],images,artifact,tmp_path/'host')
    good = request('local_boundary',[.1,.2,.7,.8])
    bad = request();bad['source_image_id']='image_missing'
    with pytest.raises(PolicyError,match='unavailable image'):
        host.execute([good,bad],info())
    assert host.ops==0
    assert host.execute([good],info()) is None
    equivalent = request('local_boundary',[.10001,.20001,.69999,.79999])
    assert host.execute([equivalent],info())=='REPEATED_OBSERVATION'
    assert host.ops==1


@pytest.mark.parametrize('fault',['missing_image','empty_finding','no_sources'])
def test_known_prose_relaxation_preserves_evidence_checks(fault):
    from cloth_agent.harness.information_flow import validate_state
    state=info(known=True)
    state[0]['missing_information']='Nothing further needed'
    if fault=='missing_image': state[0]['source_image_ids']=['image_missing']
    if fault=='empty_finding': state[0]['finding']=' '
    if fault=='no_sources': state[0]['source_image_ids']=[]
    with pytest.raises(PolicyError):
        validate_state(state,[{'image_id':'image_0'}])


def test_unknown_information_id_is_still_rejected(tmp_path,setup_case):
    case,images,artifact=setup_case
    host=ObservationHost(case['observation'],images,artifact,tmp_path/'host')
    missing=request();missing['gap_id']='not_in_state'
    with pytest.raises(PolicyError,match='existing information item'):
        host.execute([missing],info(known=True))
    assert host.ops==0


def test_known_orientation_can_request_preparation(tmp_path,setup_case):
    model=Model(global_result(known=True),decision())
    report=run(tmp_path,setup_case,model)
    assert report['status']=='COMPLETED'
    assert len(model.requests[1]['images'])==5


def test_intermediate_crop_rotate_crop_preserves_pixels_and_lineage(tmp_path, setup_case):
    from cloth_agent.image_tools_mcp import transform_point
    case, images, artifact = setup_case
    host = ObservationHost(case['observation'], images, artifact, tmp_path/'chain')
    crop = request('local_boundary', [.1,.2,.7,.8])
    assert host.execute([crop], info()) is None
    first = host.catalog[-1]
    rotate = request()
    rotate['source_image_id'] = first['image_id']
    assert host.execute([rotate], info()) is None
    clean, overlay = [next(i for i in host.catalog if i['image_id'] == id)
                      for id in host.history[-1]['delivered_image_ids']]
    assert clean['original_image_id'] == 'image_0'
    assert overlay['original_image_id'] == 'image_1'
    assert clean['to_original'] == overlay['to_original']
    assert len(overlay['lineage']) == 2
    final = request('local_boundary', [.25,.25,.75,.75])
    final['source_image_id'] = clean['image_id']
    assert host.execute([final], info()) is None
    result = host.catalog[-1]
    assert result['parent_image_id'] == clean['image_id']
    assert result['original_image_id'] == 'image_0'
    assert len(result['lineage']) == 3
    x,y = transform_point(result['to_original'], [0,0])
    with Image.open(result['path']) as actual, Image.open(images[0]) as original:
        assert actual.getpixel((0,0)) == original.getpixel((round(x),round(y)))
    assert all(item['parent_image_id'] is None or any(p['image_id']==item['parent_image_id'] for p in host.catalog)
               for item in host.catalog)


def test_overlay_and_reference_are_valid_sources(tmp_path, setup_case):
    case, images, artifact = setup_case
    host = ObservationHost(case['observation'], images, artifact, tmp_path/'sources')
    req = request('overlay_occlusion', [.1,.2,.7,.8])
    req['source_image_id'] = 'image_1'
    assert host.execute([req], info()) is None
    req = request('local_boundary', [.2,.2,.8,.8])
    req['source_image_id'] = 'image_2'
    assert host.execute([req], info()) is None
    assert host.catalog[-1]['original_image_id'] == 'image_2'
    assert host.catalog[-1]['role'] == 'reference'


def test_every_recipe_intermediate_is_reusable(tmp_path, setup_case):
    case, images, artifact = setup_case
    host = ObservationHost(case['observation'], images, artifact, tmp_path/'intermediate')
    req = request('local_boundary', [.1,.2,.7,.8]); req['enlarge'] = True
    assert host.execute([req], info()) is None
    crop, enlarged = host.catalog[-2:]
    assert crop['operation'] == 'crop_image'
    assert enlarged['parent_image_id'] == crop['image_id']
    follow = request('local_boundary', [.1,.1,.9,.9])
    follow['source_image_id'] = crop['image_id']
    assert host.execute([follow], info()) is None
    assert host.catalog[-1]['parent_image_id'] == crop['image_id']


@pytest.mark.parametrize('enlarge', [False, True])
def test_orientation_crop_rotate_combination(tmp_path, setup_case, enlarge):
    case, images, artifact = setup_case
    host = ObservationHost(case['observation'], images, artifact, tmp_path/'combined')
    req = request('orientation', [.1,.2,.7,.8]); req['enlarge'] = enlarge
    assert host.execute([req], info()) is None
    delivered = [next(i for i in host.catalog if i['image_id']==id)
                 for id in host.history[-1]['delivered_image_ids']]
    assert delivered[0]['to_original'] == delivered[1]['to_original']
    assert [x['operation'] for x in delivered[0]['lineage']] == ['crop_image','rotate_image'] + (['resize_image'] if enlarge else [])
    assert delivered[0]['size'] == ([108,72] if enlarge else [36,24])
    if not enlarge:
        with Image.open(images[0]) as original, Image.open(delivered[0]['path']) as actual:
            expected = original.crop((4,12,28,48)).transpose(Image.Transpose.ROTATE_270)
            assert actual.tobytes() == expected.tobytes()
