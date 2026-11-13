import json
from types import SimpleNamespace
import numpy as np
import pytest
from PIL import Image
from cloth_agent.experiments import partitioned_unfold as unfold
from cloth_agent.experiments.unfold_planning_skills import NAMES
from cloth_agent.contour_normal_grasp import geometry, NormalGraspError


@pytest.fixture(autouse=True)
def identity_planning_view_for_geometry_unit_tests(monkeypatch):
    from cloth_agent.experiments import unfold_topdown as td
    def identity(before, folder):
        before.raw_plan=lambda plan:dict(plan)
        return before
    monkeypatch.setattr(td,'primary',identity)


def selection(skill):
    result=dict(decision='ACTION',skill=skill,region_name='左侧内收区域',evidence='左边界可见',
        boundary_point_px=[60,100],chord_endpoints_px=[[60,60],[60,140]] if skill==NAMES[2] else [],
        pull_distance_mm=40,grasp_px=[70,100],target_px=[40,100],grasp_angle_deg=35.)
    return result


@pytest.mark.parametrize('skill',NAMES)
def test_reveal_skill_uses_farthest_geometry(tmp_path,skill):
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';folder.mkdir()
    rgb=np.full((200,300,3),[180,150,30],dtype=np.uint8)
    rgb[40:160,60:180]=[240,240,240]
    image=tmp_path/'rgb.png';Image.fromarray(rgb).save(image)
    Image.fromarray(rgb).save(tmp_path/'goal/target.png')
    before=SimpleNamespace(image=image,sample=lambda pixel,spread:np.r_[pixel,50.])
    calls=[]
    def direct(folder,body,headers,timeout):
        calls.append(folder.name)
        assert folder.name=='skill_selection', 'No second model review call is allowed'
        assert 'Simple只能抓当前图中可见的衣袖或者下摆' in body['instructions']
        assert '\n只能抓当前图中可见的衣袖或者下摆' not in body['instructions']
        assert '原有部位限制仍有效' not in body['instructions']
        return selection(skill),dict(wall_latency_s=1)
    plan,metrics=unfold.plan_round(tmp_path,folder,before,{},1,direct)
    assert calls==['skill_selection']
    assert not (folder/'skill_review').exists()
    assert plan['decision']=='ACTION'
    assert plan['grasp_angle_deg']==35.
    assert metrics['semantic_review'] is False
    assert metrics['wall_latency_s']==1
    manifest=json.loads((folder/'planning_skills.json').read_text())
    assert manifest['local_cv_executed'] is True
    assert manifest['selected_skill']==skill
    assert manifest['action_source']=='measured_cv_candidate'
    assert manifest['automatic_opposite_grasp'] is True
    proposal=json.loads((folder/'skill_geometry/proposal.json').read_text())
    assert plan['grasp_px']==proposal['geometry']['grasp_px']
    assert plan['target_px']==proposal['geometry']['target_px']
    assert plan['grasp_px'][0]>150
    assert (folder/'skill_geometry/action.png').exists()
    prompt=(folder/'skill_selection/planner_prompt.txt').read_text()
    assert 'Simple只能抓当前图中可见的衣袖或者下摆' in prompt
    from cloth_agent.experiments.unfold_completion import ARM6_DONE_RULE
    assert ARM6_DONE_RULE in prompt
    assert '袖子相对衣身的伸出长度应尽量接近目标图' not in prompt
    assert '完整衣物外轮廓明确可辨' not in prompt
    assert '下摆角不能只是露出' not in prompt
    assert '任一角不可见' not in prompt
    assert '对向袖子' not in prompt
    assert '普通动作优先抓取对向的可见袖子' not in prompt
    assert '优先向中央方向拉回' not in prompt
    assert '明显歪斜时优先通过抓放回正' not in prompt
    assert '不为居中单独操作' in prompt
    assert '正向最远交点' in prompt


def test_chord_uses_arc_midpoint_not_chord_midpoint():
    mask=np.zeros((300,400),np.uint8);mask[50:250,100:300]=1
    # Arc goes around upper left corner, with witness on left side.
    g=geometry(mask,[100,80],chord_endpoints=[[160,50],[100,180]],pull_px=60)
    assert g['origin_px'] != [130.,115.]
    assert g['origin_px']==[100.,85.]
    assert g['arc_length_px']==190


def test_skill_without_boundary_reselects_instead_of_dispatch(tmp_path):
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';folder.mkdir()
    image=tmp_path/'rgb.png';Image.new('RGB',(300,200)).save(image)
    Image.new('RGB',(300,200)).save(tmp_path/'goal/target.png')
    def direct(*args):
        value=selection(NAMES[0]);value.update(grasp_px=None,target_px=None,boundary_point_px=None)
        return value,{}
    plan,metrics=unfold.plan_round(tmp_path,folder,SimpleNamespace(image=image),{},1,direct)
    assert plan['decision']=='ABSTAIN' and metrics['selection_attempts']==3


def test_normal_allows_intermediate_hole_with_supported_far_grasp():
    mask=np.zeros((200,300),np.uint8);mask[40:160,60:180]=1
    mask[95:106,110:120]=0
    g=geometry(mask,[60,100],pull_px=40)
    assert g['intersection_px']==[179.,100.]
    assert g['grasp_px']==[171,100]


def test_unknown_depth_hole_is_not_treated_as_physical_hole():
    mask=np.zeros((200,300),np.uint8);mask[40:160,60:180]=1
    mask[95:106,110:120]=0
    unknown=np.zeros_like(mask,dtype=bool);unknown[95:106,110:120]=True
    g=geometry(mask,[60,100],pull_px=40,unknown_mask=unknown)
    assert g['grasp_px']==[171,100]
    unknown[100,171]=True
    with pytest.raises(NormalGraspError,match='visible segmented material'):
        geometry(mask,[60,100],pull_px=40,unknown_mask=unknown)


def test_no_semantic_review_does_not_bypass_measured_depth():
    from cloth_agent.experiments.unfold_skill_planner import measured_candidate
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    mask=np.zeros((200,300),np.uint8);mask[40:160,60:180]=1
    def missing(*args):raise DepthUnavailableError('missing measured depth')
    before=SimpleNamespace(sample=missing)
    with pytest.raises(NormalGraspError,match='实测深度/距离'):
        measured_candidate(before,mask,selection(NAMES[0]))


@pytest.mark.parametrize('recover',[True,False])
def test_depth_failure_reselects_boundary_without_semantic_review(tmp_path,monkeypatch,recover):
    from cloth_agent.experiments import unfold_skill_planner as skills
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';folder.mkdir()
    rgb=np.full((200,300,3),[180,150,30],np.uint8);rgb[40:160,60:180]=240
    image=tmp_path/'rgb.png';Image.fromarray(rgb).save(image);Image.fromarray(rgb).save(tmp_path/'goal/target.png')
    calls=[]
    def sample(p,spread):
        if not recover or p[0]>150:raise DepthUnavailableError('both cameras lack usable depth at '+str(p))
        return np.r_[p,50.]
    def direct(folder,body,*args):
        calls.append(folder)
        s=selection(NAMES[0]);s['grasp_px']=[170,100]
        if len(calls)>1:
            assert 'both cameras lack usable depth' in body['instructions']
            assert '不要只改变拉动距离' in body['instructions']
            s['boundary_point_px']=[179,100]
        return s,{'wall_latency_s':1}
    plan,metrics=unfold.plan_round(tmp_path,folder,SimpleNamespace(image=image,sample=sample),{},1,direct)
    assert len(calls)==(2 if recover else 3)
    assert plan['decision']==('ACTION' if recover else 'ABSTAIN')
    assert metrics['wall_latency_s']==len(calls)
    assert all(p.name=='skill_selection' for p in calls)
    if recover:assert plan['grasp_px'][0]<100




def test_direct_points_default_uses_one_call_and_no_normal_geometry(tmp_path, monkeypatch):
    from cloth_agent.experiments import unfold_skill_planner as skills
    (tmp_path/'goal').mkdir();folder=tmp_path/'round';folder.mkdir()
    image=tmp_path/'rgb.png';Image.new('RGB',(300,200)).save(image)
    Image.new('RGB',(300,200)).save(tmp_path/'goal/target.png')
    def forbidden(*args, **kwargs):
        raise AssertionError('Direct points must not invoke normal geometry')
    monkeypatch.setattr(skills,'measured_candidate',forbidden)
    calls=[]
    def direct(folder,body,headers,timeout):
        calls.append(1)
        assert 'Simple只能抓当前图中可见的衣袖或者下摆' in body['instructions']
        value=selection(None)
        value.update(grasp_px=[100,100],target_px=[140,100],boundary_point_px=None)
        return value,dict(wall_latency_s=1)
    plan,_=unfold.plan_round(tmp_path,folder,SimpleNamespace(image=image,sample=lambda p,s:np.r_[p,50.]),{},1,direct)
    assert len(calls)==1
    assert plan['grasp_px']==[100,100]
    assert plan['target_px']==[140,100]
    assert json.loads((folder/'planning_skills.json').read_text())['action_source']=='direct_points'


def test_switch_rule_replaces_all_old_criteria_and_can_restore(monkeypatch, tmp_path):
    from cloth_agent.experiments import unfold_planning_skills as refs
    from cloth_agent.experiments import unfold_skill_planner as planner
    monkeypatch.setenv('CLOTH_SKILL_SWITCH','legacy')
    original=refs.append(tmp_path,tmp_path,planner.UNFOLD_PROMPT)+'\n'+planner.SELECT_PROMPT
    assert refs.apply_switch_rule(original)==original
    monkeypatch.setenv('CLOTH_SKILL_SWITCH','layered')
    current=refs.apply_switch_rule(refs.append(tmp_path,tmp_path,planner.UNFOLD_PROMPT)+'\n'+planner.SELECT_PROMPT)
    assert current.count(refs.SWITCH_RULE)==1
    assert '只要发现被隐藏的衣角或袖口，本步优先使用几何skill' not in current
    assert '要展开的边角看不到时，使用切线或割线Skill把它打开' not in current
    assert '其他部分已大致展开' not in current
    assert '要展开的对应衣角或袖口被其他布层覆盖、不可见时，使用切线或割线Skill把它打开' in current
    assert 'Simple只能抓四个边角' not in current
    assert '边缘带稍向布面内侧选点' in current
    assert '袖子最外侧的可见边缘' not in current
    assert '不能仅因大块内收、褶皱、歪斜或角点难辨触发' not in current
    assert '能用 Simple 有效推进展开就用 Simple；没有有效的 Simple 展开动作时才用几何 Skill' not in current
    assert '袖子相对衣身的伸出长度应尽量接近目标图' not in current
    assert '伸出程度已接近目标后，不要仅为消除褶皱继续拉扯' not in current
    assert '衣领必须显露' not in current
    assert '不要求领口撑开，也不要求完整领圈清晰可见' not in current
    assert '仅当能指出目标部位被哪一层布覆盖' not in current
    assert all(old not in current.splitlines() for old in refs.LEGACY_SWITCH_LINES)
    assert '几何Skill优先使用割线garment-chord-grasp' in current
    assert 'Host反馈割线无法生成有效候选' in current
    assert '正向最远交点' in current
    assert '普通动作优先抓取对向的可见袖子' not in current
    assert json.loads((tmp_path/'planning_skills.json').read_text())['switch_policy']=='layered'
    monkeypatch.setenv('CLOTH_SKILL_SWITCH','legacy')
    restored=refs.append(tmp_path,tmp_path,planner.UNFOLD_PROMPT)+'\n'+planner.SELECT_PROMPT
    assert restored==original


def test_concise_skill_body_reaches_request_without_loading_reference(tmp_path, monkeypatch):
    from cloth_agent.experiments import unfold_planning_skills as refs
    monkeypatch.setenv('CLOTH_SKILL_SWITCH', 'layered')
    prompt = refs.append(tmp_path, tmp_path, '')
    assert prompt.count(refs.SWITCH_RULE) == 1
    for name in refs.NAMES:
        raw = (refs.ROOT / 'skills' / name / 'SKILL.md').read_text(encoding='utf-8')
        body = raw.split('\n---', 1)[1].strip()
        assert refs.planning_text(name, raw) == body
        # Only the obsolete switching condition is replaced; geometry stays intact.
        for line in body.splitlines():
            if line.strip() in refs.LEGACY_SWITCH_LINES:
                assert line not in prompt
            elif line.strip():
                assert line in prompt
        reference = refs.ROOT / 'skills' / name / 'references/full_method.md'
        assert reference.is_file()
        assert reference.read_text(encoding='utf-8') not in prompt
    assert '/home/sja/' not in prompt
    assert 'contour_normal_grasp_once' not in prompt
    assert '不能为了得到期望抓点反向选择端点' in prompt
    assert '不为得到预设抓点反复挑选 Q' in prompt


def test_recenter_removed_but_space_preparation_available(tmp_path):
    from cloth_agent.experiments import unfold_skill_planner as sp
    from cloth_agent.experiments import unfold_planning_skills as skills
    prompt = skills.append(tmp_path, tmp_path, sp.UNFOLD_PROMPT)
    combined = prompt + sp.SELECT_PROMPT + sp.ACTION_GUIDANCE
    assert 'garment-recenter' not in combined
    assert '衣物中心到桌面中心' not in combined
    assert 'make-space-for-skill' in combined
    choices = sp.SELECT_SCHEMA['properties']['skill']['anyOf'][0]['enum']
    assert 'garment-recenter' not in choices
    assert 'make-space-for-skill' in choices
    manifest = json.loads((tmp_path/'planning_skills.json').read_text())
    assert all(item['name'] != 'garment-recenter' for item in manifest['skills'])


@pytest.mark.parametrize('desired', [5., 250., 350.])
def test_measured_candidate_uses_image_extent_not_fixed_pull_caps(desired):
    from cloth_agent.experiments import unfold_skill_planner as planner
    import jsonschema
    mask=np.zeros((200,600),np.uint8)
    mask[40:160,40:180]=1
    before=SimpleNamespace(sample=lambda p,spread:np.array([p[0],p[1],50.]))
    chosen=selection(NAMES[0])
    chosen.update(boundary_point_px=[40,100],chord_endpoints_px=[],pull_distance_mm=desired)
    jsonschema.validate(chosen,planner.SELECT_SCHEMA)
    geometry,distance,options=planner.measured_candidate(before,mask,chosen)
    assert geometry['grasp_px']==[171,100]
    # Short requests still need to place beyond the actual cloth edge.
    assert geometry['target_px'][0]>=180
    if desired>180:
        assert abs(distance-desired)<=2
        assert options['pull_px']>180
    assert 0<=geometry['target_px'][0]<600


def test_long_candidate_still_requires_measured_depth():
    from cloth_agent.experiments import unfold_skill_planner as planner
    from cloth_agent.dual_arm.observation import DepthUnavailableError
    mask=np.zeros((200,600),np.uint8);mask[40:160,40:180]=1
    def missing(p,spread):
        raise DepthUnavailableError('missing measured depth')
    chosen=selection(NAMES[0]);chosen.update(boundary_point_px=[40,100],chord_endpoints_px=[],pull_distance_mm=300.)
    with pytest.raises(NormalGraspError,match='实测深度'):
        planner.measured_candidate(SimpleNamespace(sample=missing),mask,chosen)
