import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import yaml

from cloth_agent import table_target_depth as m
from cloth_agent.dual_arm.geometry import DualArmError
from cloth_agent.dual_arm.observation import Observation, DepthUnavailableError
from cloth_agent.experiments import trajectory_memory_full_episode_v1 as shared
from cloth_agent.experiments import one_shot_goal_full_episode_v1 as one_shot


@pytest.fixture
def scene(tmp_path):
    K = np.asarray(yaml.safe_load(m.INTRINSICS_FILE.read_text())["K"])
    y, x = np.mgrid[:720, :1280]
    # Known support at world Z=30; an elevated garment at Z=60.
    z = np.full((720, 1280), 30.)
    z[200:450, 450:750] = 60.
    depth = 650.-z
    xyz = np.stack(((x-K[0, 2])*depth/K[0, 0],
                    -(y-K[1, 2])*depth/K[1, 1], z), axis=2)
    rgb = np.full((720, 1280, 3), [45, 42, 50], dtype=np.uint8)
    rgb[200:450, 450:750] = [190, 130, 220]
    Image.fromarray(rgb).save(tmp_path/"rgb.png")
    T = np.diag([1., -1., -1., 1.]); T[2, 3] = 650.
    meta = dict(camera_serial="317222073552", world_from_camera_mm=T.tolist())
    return Observation(tmp_path, meta, xyz)


def sample(scene, target=(800, 650)):
    return m.sample_target(scene, target, [600, 300], 8, scene.directory/"target_depth.json")


def test_missing_target_uses_true_plane_and_leaves_cloud_unchanged(scene):
    scene.xyz[645:656, 795:806] = np.nan
    saved = scene.xyz.copy()
    result = sample(scene)
    assert result[2] == pytest.approx(30, abs=1e-5)
    assert np.array_equal(saved, scene.xyz, equal_nan=True)
    audit = json.loads((scene.directory/"target_depth.json").read_text())
    assert audit["source"] == "TABLE_PLANE_ESTIMATE"
    assert audit["measured_depth"] is False and audit["pixel_unchanged"]
    # No zero-height substitution and no cloud-wide hole filling.
    assert np.isnan(scene.xyz[650, 800]).all()


def test_measured_target_is_identical_and_does_not_fit_plane(scene, monkeypatch):
    expected = scene.sample([800, 650], 8)
    monkeypatch.setattr(m, "_estimate", lambda *a: pytest.fail("unexpected fallback"))
    assert np.array_equal(sample(scene), expected)


def test_raw_discontinuity_is_not_bypassed(scene, monkeypatch):
    scene.xyz[650, 800, 2] += 10
    monkeypatch.setattr(m, "_estimate", lambda *a: pytest.fail("unexpected fallback"))
    with pytest.raises(DualArmError, match="discontinuity"):
        sample(scene)
    assert json.loads((scene.directory/"target_depth.json").read_text())["status"] == "REJECT"


@pytest.mark.parametrize("target", [(600, 350), (755, 350), (1100, 650)])
def test_cloth_hole_near_cloth_and_outside_roi_rejected(scene, target):
    x, y = target
    scene.xyz[y-3:y+4, x-3:x+4] = np.nan
    with pytest.raises(DualArmError, match="outside bare-table ROI|near foreground"):
        sample(scene, target)


def test_matching_dark_garment_is_ambiguous(scene):
    Image.new("RGB", (1280, 720), (45, 42, 50)).save(scene.image)
    scene.xyz[645:656, 795:806] = np.nan
    with pytest.raises(DualArmError, match="garment/table appearance ambiguous"):
        sample(scene)


def test_large_hole_rejected(scene):
    scene.xyz[610:691, 760:841] = np.nan
    with pytest.raises(DualArmError, match="hole exceeds"):
        sample(scene)


def test_remaining_depth_cannot_contradict_plane(scene):
    scene.xyz[645:656, 795:806] = np.nan
    scene.xyz[650, 800] = [100., -100., 70.]
    with pytest.raises(DualArmError, match="existing target depth contradicts"):
        sample(scene)


def test_calibration_mismatch_rejected(scene):
    scene.xyz[645:656, 795:806] = np.nan
    scene.meta["camera_serial"] = "other"
    with pytest.raises(DualArmError, match="unsupported camera"):
        sample(scene)


def test_missing_grasp_stops_before_target_fallback(tmp_path):
    def fail_sample(*args):
        raise DepthUnavailableError("missing Pa")
    observation = SimpleNamespace(validate_for=lambda *a, **k: None, sample=fail_sample)
    runtime = SimpleNamespace(config=SimpleNamespace(limits={"max_surface_spread_mm": 8}))
    with pytest.raises(DepthUnavailableError, match="missing Pa"):
        shared.prepare_action(runtime, observation,
            {"grasp_px": [10, 10], "target_px": [20, 20]}, tmp_path,
            target_sampler=lambda *a: pytest.fail("Pa must not use a table estimate"))


def test_one_shot_preserves_target_adapter_and_pregrasp_orientation(monkeypatch):
    def prepare(*args, **kwargs):
        assert kwargs == {"target_sampler": m.project_target,
                          "orientation_resolver": one_shot.one_shot_orientation.resolve,
                          "check_grasp_height_spread": False, "smoothing": False}
        return "existing checks"
    monkeypatch.setattr(one_shot, "prepare_collection_action", prepare)
    assert one_shot.prepare_action(None, None, None, None) == "existing checks"


def test_plane_fallback_still_reaches_existing_ik_rejection(scene):
    scene.xyz[395:406, 795:806] = np.nan
    scene.meta.update(calibration_id="test", synthetic=False,
                      joints_deg={"left": [0]*6, "right": [0]*7})
    calls = []
    def reject_ik(*args):
        calls.append("existing_ik")
        raise DualArmError("existing IK rejected")
    runtime = SimpleNamespace(
        config=SimpleNamespace(raw={"calibration_id": "test"},
            limits={"max_surface_spread_mm": 8},
            arms={"left": SimpleNamespace(world_from_base=np.eye(4), home_joints=[0]*6)}),
        models={"left": SimpleNamespace(forward=lambda q: [400, 0, 500, 180, 0, 0])},
        coordinator=SimpleNamespace(snapshots=lambda: {"left": {"joints": [0]*6}}),
        _inverse_target=reject_ik)
    with pytest.raises(DualArmError, match="existing IK rejected"):
        one_shot.prepare_action(runtime, scene,
            {"decision": "ACTION", "grasp_px": [700, 350], "target_px": [800, 400],
             "motion_type": "lift_pull_place", "grasp_angle_deg": None,
             "short_reason": "test target fallback with unchanged orientation"}, scene.directory)
    assert calls == ["existing_ik"]
    points = json.loads((scene.directory/"points.json").read_text())
    assert points[0]["measured_depth"] is True
    assert points[1]["depth_source"] == "TABLE_PLANE_ESTIMATE"


def test_saved_real_pb_hole_regression(tmp_path):
    path = Path("reports/one_shot_goal_full_episode_v1_20261020_225322_882924401/round_008/before")
    if not path.is_dir():
        pytest.skip("local real episode fixture unavailable")
    observation = Observation.load(path)
    with pytest.raises(DepthUnavailableError):
        observation.sample([794, 683], 8)
    result = m.sample_target(observation, [794, 683], [710, 628], 8, tmp_path/"audit.json")
    assert 20 < result[2] < 40
    assert json.loads((tmp_path/"audit.json").read_text())["source"] == "TABLE_PLANE_ESTIMATE"


def test_project_target_never_samples_pb_even_with_large_hole(scene,monkeypatch):
    scene.xyz[610:710,760:841]=np.nan
    monkeypatch.setattr(scene,'sample',lambda *a:pytest.fail('Pb must not sample measured depth'))
    result=m.project_target(scene,[800,650],[600,300],8,scene.directory/'project.json')
    assert result[2]==pytest.approx(30,abs=1e-5)
    audit=json.loads((scene.directory/'project.json').read_text())
    assert audit['target_depth_sampled'] is False and audit['local_support_required'] is False
    assert np.isnan(scene.xyz[650,800]).all()


def test_project_target_still_rejects_outside_table_roi(scene):
    with pytest.raises(DualArmError,match='outside bare-table ROI'):
        m.project_target(scene,[1200,650],[600,300],8,scene.directory/'project.json')


def test_latest_outside_support_pb_now_uses_plane(tmp_path):
    path=Path('reports/reset_radius_sweep_20261023_210122_281488834/episode_0002/round_006/pre_execution_observation')
    if not path.is_dir():pytest.skip('local real observation unavailable')
    obs=Observation.load(path)
    assert not np.isfinite(obs.xyz[689,458]).all()
    result=m.project_target(obs,[458,689],[492,634],8,tmp_path/'audit.json')
    assert np.isfinite(result).all() and 20<result[2]<40
    assert json.loads((tmp_path/'audit.json').read_text())['status']=='PASS'


def test_projection_allows_bare_table_near_cloth_but_not_cloth_itself(scene):
    result=m.project_target(scene,[755,350],[600,300],8,scene.directory/'project.json')
    assert result[2]==pytest.approx(30,abs=1e-5)
    audit=json.loads((scene.directory/'project.json').read_text())
    assert audit['target_foreground_clearance_px']==0
    assert audit['plane_reference_clearance_px']==24
    with pytest.raises(DualArmError,match='pixel itself'):
        m.project_target(scene,[600,350],[600,300],8,scene.directory/'project.json')


def test_saved_false_foreground_clearance_rejection_now_passes(tmp_path):
    path=Path('reports/reset_radius_sweep_20261024_112209_083531285/episode_0001/round_003/pre_execution_observation')
    if not path.is_dir():pytest.skip('local real observation unavailable')
    obs=Observation.load(path)
    result=m.project_target(obs,[289,382],[372,399],8,tmp_path/'audit.json')
    assert np.isfinite(result).all()
    audit=json.loads((tmp_path/'audit.json').read_text())
    assert audit['status']=='PASS' and audit['target_depth_sampled'] is False
    assert audit['target_inside_table_roi'] is True
    assert audit['target_table_rgb_distance']<audit['target_table_rgb_threshold']


def test_user_expanded_roi_accepts_saved_right_target(tmp_path):
    path=Path('reports/reset_radius_sweep_20261024_122641_082985880/episode_0001/round_004/pre_execution_observation')
    if not path.is_dir():pytest.skip('local observation unavailable')
    obs=Observation.load(path)
    result=m.project_target(obs,[1014,401],[934,432],8,tmp_path/'audit.json')
    assert np.isfinite(result).all()
    audit=json.loads((tmp_path/'audit.json').read_text())
    assert audit['status']=='PASS' and audit['target_inside_table_roi']
