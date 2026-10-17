from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest

from cloth_agent.dual_arm.config import DualConfig
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.robot_capabilities import build_capabilities,render_capabilities,_arm6_downward
from cloth_agent.codex_loop import loop
from .test_codex_loop import FakeRuntime,model

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def report():
    config=DualConfig.load(ROOT/'config/dual_arm.local.json',root=ROOT,homing_only=True)
    models={k:ArmModel(a) for k,a in config.arms.items()}
    return config,models,build_capabilities(config,models)


def test_generated_limits_and_coordinates_match_models(report):
    config,models,data=report
    for k,arm in data['arms'].items():
        assert arm['world_from_base_mm']==config.arms[k].world_from_base.tolist()
        assert arm['tcp_offset_mm_deg']==config.arms[k].tcp_offset.tolist()
        assert np.allclose([j['lower_deg'] for j in arm['joints']],np.degrees(models[k].lower))
    bound=data['arms']['left']['downward_outer_bound']
    assert bound['shoulder_to_wrist_max_mm']==pytest.approx(640.6454105)
    assert bound['table'][1]['max_horizontal_radius_mm']==pytest.approx(715.8457,abs=.01)
    assert data['arms']['right']['downward_outer_bound'] is None
    assert data['arms']['left']['workspace_base_mm']['min'][0] is None
    text=render_capabilities(data)
    assert '|---:|---:|\n| 0 |' in text
    assert '外边界，不是保证可达的内工作区' in text
    assert 'grasp 给观测原始高度' not in text
    assert 'arm6/arm7 单臂 15 mm' not in text
    assert '直接检查输入的 TCP 目标' in text


def test_formula_not_reused_after_tcp_geometry_change(report):
    config,_,original=report
    arm=deepcopy(config.arms['left']);arm.tcp_offset[0]=10
    m=ArmModel(arm)
    assert _arm6_downward(m) is None
    models={k:ArmModel(a) for k,a in config.arms.items()};models['left']=m
    changed=deepcopy(config);changed.arms['left']=arm
    assert build_capabilities(changed,models)['fingerprint']!=original['fingerprint']


def test_capabilities_reach_initial_plan_and_local_grasp_repair(report,tmp_path):
    _,_,data=report
    runtime=FakeRuntime()
    runtime.capability_report=lambda:data
    runtime.execute=lambda *a:dict(status='PLANNING_FAILED',failure_kind='IK_FAILED',physical_execution=False,
        error='grasp rejected',ik_failure=dict(phase='grasp_goal_ik',robot='arm6',command_index=1))
    prompts=[]
    result=loop('fold',runtime,tmp_path/'run',model_call=model(runtime,prompts,
        ['motion.gripper(arm="left",action="open")','motion.blocked(reason="no suitable local grasp")','motion.complete()']))
    assert result['status']=='MODEL_REPORTED_COMPLETE'
    assert all(data['fingerprint'][:16] in p for p in prompts)
    assert '原观测目录' in prompts[1] and '允许依据同一保存观测局部修正失败抓点' in prompts[1]
    assert '先查看 RGB' not in prompts[1]
    assert json.loads((tmp_path/'run/robot_capabilities.json').read_text())==data
    assert (tmp_path/'run/robot_capabilities.md').is_file()
