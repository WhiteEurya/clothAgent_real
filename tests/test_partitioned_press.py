from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from cloth_agent.experiments import partitioned_press as press, partitioned_unfold as unfold
from cloth_agent.dual_arm.geometry import matrix_pose, pose_matrix, DualArmError
from cloth_agent.experiment import ExperimentRunner


@pytest.mark.parametrize('arm',['left','right'])
def test_closed_jaw_vertical_press_and_single_depth(tmp_path,arm):
    runtime=unfold.configure_runtime(unfold.PartitionedReplayRuntime(unfold.DEFAULT_CONFIG,planning_only=True))
    try:
        joints={k:a.home_joints.copy() for k,a in runtime.config.arms.items()}
        home=matrix_pose(runtime.config.arms[arm].world_from_base @ pose_matrix(runtime.models[arm].forward(joints[arm])))
        surface=home[:3].copy();surface[2]=70
        poses=[np.r_[surface/1000,np.radians(home[3:])],np.r_[(surface+[0,0,50])/1000,np.radians(home[3:])]]
        source=f'motion.grasp(right={poses[0].tolist()!r})\nmotion.move_tcp(right={poses[1].tolist()!r})\n'
        config,program,assignment=runtime.compile(source,SimpleNamespace(meta={'joints_deg':joints}),press_depth_mm=5)
        assert assignment['active_arm']==arm
        contact=runtime.config.arms[arm].world_from_base @ np.r_[assignment['contact_base_mm_deg'][:3],1]
        np.testing.assert_allclose(contact[:3],surface-[0,0,5])
        runner=ExperimentRunner(tmp_path/'preview',config)
        path=runner.workspace/'press.py';path.write_text(program)
        result=runner.preflight(path)
        assert result.error is None
        assert [a['name'] for a in result.actions]==['home','close_gripper','move','move','move','open_gripper','move','home']
        with pytest.raises(DualArmError):runtime.compile(source,SimpleNamespace(meta={'joints_deg':joints}),press_depth_mm=16)
        lateral=source.replace(repr(poses[1].tolist()),repr((poses[1]+[.02,0,0,0,0,0]).tolist()))
        with pytest.raises(DualArmError,match='vertical'):runtime.compile(lateral,SimpleNamespace(meta={'joints_deg':joints}),press_depth_mm=5)
    finally:runtime.close()


@pytest.mark.parametrize('decision,confidence',[('SKIP',.9),('UNCERTAIN',.4),('PRESS',.7)])
def test_no_press_on_no_occlusion_or_uncertainty(tmp_path,decision,confidence):
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    before=SimpleNamespace(image=image,validate_for=lambda *a,**kw:None)
    runtime=SimpleNamespace(config=None,execute=lambda *a,**k:pytest.fail('no dispatch'))
    result=press.run(runtime,before,tmp_path/'press',headers={},timeout=1,depth_mm=5,
        direct=lambda *a:(dict(decision=decision,center_px=[50,50] if decision=='PRESS' else None,confidence=confidence,reason='test'),{}))
    assert result['status']=='SKIPPED'
    assert not result['physical_execution']


def test_reset_proceeds_directly_to_unfold_without_press(tmp_path):
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    events=[]
    def observe(path):events.append(path.name);return SimpleNamespace(directory=path,image=image)
    runtime=SimpleNamespace(observe=observe,close=lambda:None)
    def request(folder,before,context):
        events.append('plan');return {'decision':'DONE'},{}
    result=unfold.episode(runtime,tmp_path,max_actions=1,request=request,
        reset_request=lambda *a:dict(status='EXECUTED',physical_execution=True))
    assert events==['initial_observation','after_reset','plan']
    assert not (tmp_path/'post_reset_press').exists()
    assert result['status']=='MODEL_REPORTED_DONE'
    assert result['physical_actions']==1


def test_press_detection_to_existing_runtime(tmp_path,monkeypatch):
    runtime=unfold.configure_runtime(unfold.PartitionedReplayRuntime(unfold.DEFAULT_CONFIG,planning_only=True))
    image=tmp_path/'rgb.png';Image.new('RGB',(100,100)).save(image)
    joints={k:a.home_joints.copy() for k,a in runtime.config.arms.items()}
    arm=runtime.config.arms['left']
    surface=(arm.world_from_base @ pose_matrix(runtime.models['left'].forward(arm.home_joints)))[:3,3]
    surface[2]=70.
    before=SimpleNamespace(image=image,directory=tmp_path,meta={'joints_deg':joints},
        sample=lambda *a:surface,validate_for=lambda *a,**kw:None)
    calls=[]
    def execute(source,observation,folder,*,press_depth_mm):
        calls.append(press_depth_mm)
        _,_,assignment=runtime.compile(source,observation,press_depth_mm=press_depth_mm)
        assert assignment['active_arm']=='left'
        return dict(status='EXECUTED',physical_execution=True)
    monkeypatch.setattr(runtime,'execute',execute)
    try:
        result=press.run(runtime,before,tmp_path/'press',headers={},timeout=1,depth_mm=5,
            direct=lambda *a:(dict(decision='PRESS',center_px=[50,50],confidence=.9,reason='visible raised fabric'),{}))
        assert result['status']=='EXECUTED'
        assert calls==[5]
        assert (tmp_path/'press/press_point.png').exists()
    finally:runtime.close()
