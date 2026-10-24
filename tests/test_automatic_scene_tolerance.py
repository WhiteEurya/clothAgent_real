from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
from cloth_agent import preference_one_shot as s
from cloth_agent.dual_arm.geometry import DualArmError


def test_automatic_local_tolerance_keeps_large_change_guard(tmp_path):
    a=np.zeros((400,400,3),dtype=np.uint8)+100
    before=SimpleNamespace(image=tmp_path/'a.png',meta={'calibration_id':'test'})
    fresh=SimpleNamespace(image=tmp_path/'b.png',meta=before.meta)
    Image.fromarray(a).save(before.image)
    for delta,passes in [(15,True),(25,True),(26,False),(60,False)]:
        b=a.copy();b[170:231,170:231]+=delta;Image.fromarray(b).save(fresh.image)
        if passes:s.verify_scene(before,fresh,{'grasp_px':[200,200]},tmp_path,automatic=True)
        else:
            with pytest.raises(DualArmError,match='scene changed'):
                s.verify_scene(before,fresh,{'grasp_px':[200,200]},tmp_path,automatic=True)
    Image.fromarray(a+20).save(fresh.image)
    s.verify_scene(before,fresh,{'grasp_px':[200,200]},tmp_path,automatic=True)
    Image.fromarray(a+26).save(fresh.image)
    with pytest.raises(DualArmError,match='scene changed'):
        s.verify_scene(before,fresh,{'grasp_px':[200,200]},tmp_path,automatic=True)


def test_real_stationary_edge_pair_passes_relaxed_local_gate(tmp_path):
    root=Path('reports/reset_radius_sweep_20261023_215146_471137083/episode_0001/round_004')
    if not root.exists():pytest.skip('local observation pair unavailable')
    from cloth_agent.dual_arm.observation import Observation
    before=Observation.load(root/'before');fresh=Observation.load(root/'pre_execution_observation')
    s.verify_scene(before,fresh,s.io.read(root/'plan.json'),tmp_path,automatic=True)
    audit=s.io.read(tmp_path/'pre_execution_scene_check.json')
    assert audit['passed'] and audit['local_grasp_rgb_q95']==15
    assert audit['global_rgb_q95'] is None
    assert audit['global_check_enabled'] is False
    assert audit['local_threshold']==25


def test_background_changes_do_not_block_automatic_but_block_human(tmp_path):
    a=np.full((400,400,3),100,dtype=np.uint8)
    b=np.full_like(a,200);b[170:231,170:231]=100
    before=SimpleNamespace(image=tmp_path/'a.png',meta={'calibration_id':'test'})
    fresh=SimpleNamespace(image=tmp_path/'b.png',meta=before.meta)
    Image.fromarray(a).save(before.image);Image.fromarray(b).save(fresh.image)
    s.verify_scene(before,fresh,{'grasp_px':[200,200]},tmp_path,automatic=True)
    with pytest.raises(DualArmError,match='scene changed'):
        s.verify_scene(before,fresh,{'grasp_px':[200,200]},tmp_path)


def test_saved_side_view_global_boundary_no_longer_rejects(tmp_path):
    root=Path('reports/reset_radius_sweep_20261024_135445_649238735/episode_0001/round_010/depth_recovery/views/view_00')
    if not root.exists():pytest.skip('local observation pair unavailable')
    from cloth_agent.dual_arm.observation import Observation
    s.verify_scene(Observation.load(root/'before'),Observation.load(root/'pre_execution_observation'),
                   s.io.read(root/'plan.json'),tmp_path,automatic=True)
    audit=s.io.read(tmp_path/'pre_execution_scene_check.json')
    assert audit['passed'] and audit['local_grasp_rgb_q95']==23
