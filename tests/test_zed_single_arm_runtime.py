import json
from pathlib import Path

import pytest

from cloth_agent.codex_dual_runtime import DualArmRuntime, ZedSingleArmRuntime
from cloth_agent.dual_arm.geometry import DualArmError


def commissioned_config(tmp_path):
    source = Path('config/dual_arm.zed_dual_xarm7_20261103.json')
    raw = json.loads(source.read_text())
    raw['collision_geometry_verified'] = True
    raw['servo_commissioned'] = True
    raw['safety']['open_tools_and_attachments_verified'] = True
    raw['safety']['controller_watchdog_verified'] = True
    raw['safety']['joint_segment_tracking_verified'] = True
    raw['safety']['validation_id'] = 'test'
    path = tmp_path / source.name
    path.write_text(json.dumps(raw))
    return path


@pytest.mark.parametrize('active,inactive', [('left', 'right'), ('right', 'left')])
def test_zed_single_arm_loads_static_other_and_no_extra_controller(tmp_path, active, inactive):
    runtime = ZedSingleArmRuntime(commissioned_config(tmp_path), active)
    try:
        assert runtime.active_arms == (active,)
        assert set(runtime.config.static_arm_obstacles) == {inactive}
        assert runtime.config.raw['execution_policy']['inactive_arm'] == inactive
        assert runtime.config.collision_envelopes_disabled_arms == frozenset({inactive})
    finally:
        runtime.close()


def test_zed_single_arm_rejects_inactive_commands(tmp_path):
    runtime = ZedSingleArmRuntime(commissioned_config(tmp_path), 'left')
    try:
        initial = {k: a.home_joints.copy() for k, a in runtime.config.arms.items()}
        with pytest.raises(DualArmError, match='inactive arm'):
            runtime.plan('motion.move_joints(right=[0,0,0,0,0,0,0])', initial)
    finally:
        runtime.close()


def test_current_zed_config_cannot_enter_legacy_dual_runtime(tmp_path):
    with pytest.raises(DualArmError, match='active-arm'):
        DualArmRuntime(commissioned_config(tmp_path))


@pytest.mark.parametrize('active', ['left', 'right'])
def test_zed_seven_joint_plan_reaches_segment_proof(tmp_path, monkeypatch, active):
    import numpy as np
    from cloth_agent.dual_arm.motion.home import pack, unpack
    runtime = ZedSingleArmRuntime(commissioned_config(tmp_path), active)
    initial = {arm: cfg.home_joints.copy() for arm,cfg in runtime.config.arms.items()}
    seen=[]
    def segments(start, end):
        assert start['left'].shape == start['right'].shape == (7,)
        assert end['left'].shape == end['right'].shape == (7,)
        seen.append((start,end))
        return []
    monkeypatch.setattr(runtime.paths,'joint_path',lambda start,end,**kwargs:[start,end])
    monkeypatch.setattr(runtime,'_segments',segments)
    try:
        unpacked=unpack(pack(initial),runtime.config)
        np.testing.assert_allclose(unpacked['left'],initial['left'])
        np.testing.assert_allclose(unpacked['right'],initial['right'])
        target=np.radians(initial[active]+.01).tolist()
        runtime.plan(f'motion.move_joints({active}={target})',initial)
        assert len(seen)==1
        with pytest.raises(DualArmError,match='axis count'):
            runtime.plan(f'motion.move_joints({active}={target[:6]})',initial)
    finally:
        runtime.close()


def test_planning_only_loads_pending_config_but_cannot_use_hardware(tmp_path):
    path=Path('config/dual_arm.zed_dual_xarm7_20261103.json')
    runtime=ZedSingleArmRuntime(path,'right',planning_only=True)
    try:
        assert runtime.physical is False
        assert runtime.config.raw['servo_commissioned'] is False
        with pytest.raises(DualArmError,match='cannot connect'):
            runtime.observe(tmp_path)
        with pytest.raises(DualArmError,match='cannot capture'):
            runtime.capture_current(tmp_path)
        with pytest.raises(DualArmError,match='cannot execute'):
            runtime.execute('',None,tmp_path)
        with pytest.raises(DualArmError,match='cannot dispatch'):
            runtime._execute_operations([],tmp_path)
    finally:
        runtime.close()


@pytest.mark.parametrize('source', [
    'motion.grasp(right=[.2,-.1,.009,3.14,0,0])\nmotion.move_tcp(right=[.4,-.1,.1,3.14,0,0])',
    'motion.move_tcp(right=[.2,-.1,.009,3.14,0,0])\nmotion.gripper(arm="right",action="close")',
])
def test_zed_execution_preserves_grounded_height_without_legacy_lowering(tmp_path,source):
    runtime=ZedSingleArmRuntime(Path('config/dual_arm.zed_dual_xarm7_20261103.json'),
                                'right',planning_only=True)
    try:
        assert runtime.adjust_motion(source,tmp_path)==source
        policy=json.loads((tmp_path/'grasp_height_policy.json').read_text())
        assert policy['offset_mm']==0
        assert policy['changes']==[]
        assert '降低 10 mm' not in runtime.motion_contract
        assert runtime.config.raw['astra_grasp_height']['mode']=='source_surface'
    finally:runtime.close()
