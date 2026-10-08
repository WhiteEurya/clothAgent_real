"""Physical-engine queries against real URDFs, no xArm SDK or hardware access."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip(
    "fcl", reason="install requirements-collision.txt for stage 1-3 tests"
)

from cloth_agent.dual_arm.collision import CollisionChecker, CollisionScene
from cloth_agent.dual_arm.collision.audit import model_audit, validate_evidence
from cloth_agent.dual_arm.collision.calibration import template
from cloth_agent.dual_arm.geometry import DualArmError


@pytest.fixture
def scene():
    return CollisionScene(template(synthetic=True))


def check(scene):
    return CollisionChecker(scene).check(scene.initial["left"], scene.initial["right"])


def test_actual_joint_order_core_meshes_and_historical_fk_are_audited():
    report = model_audit()
    assert len(report["joint_order"]) == 13
    assert report["joint_order"][6] == "right/joint1"
    for arm in report["arms"].values():
        assert arm["core_collision_complete"] and arm["joint_order_matches"]
        assert all(m["exists"] for row in arm["links"] for m in row["meshes"])
    assert not report["arms"]["left"]["native_gripper_collision_present"]
    assert report["arms"]["right"]["native_gripper_collision_present"]
    assert report["arms"]["left"]["saved_fk_pass_2mm_1deg"]
    assert not report["arms"]["right"]["saved_fk_pass_2mm_1deg"]
    assert report["base_transform"]["base_origin_distance_m"] == pytest.approx(
        1.1263282679
    )
    assert report["base_transform"]["tcp_distance_at_saved_home_m"] != pytest.approx(
        1.1263
    )
    assert not report["physical_acceptance"]


def test_incomplete_real_attachment_template_never_reports_safe():
    with pytest.raises((DualArmError, TypeError)):
        CollisionScene(template())


def test_synthetic_safe_state_includes_all_pair_classes(scene):
    result = check(scene)
    assert result["safe"] and not result["collision"]
    assert result["checked_pairs"] > 150 and result["distance_units"] == "m"
    pairs = {frozenset((a.name, b.name)) for a, b in scene.pairs()}
    assert frozenset(("cam_a", "cam_b")) in pairs
    assert frozenset(("left_gripper", "right_gripper")) in pairs
    assert frozenset(("cam_a", "right/link5/0")) in pairs
    assert frozenset(("left/link1/0", "left/link5/0")) in pairs
    assert frozenset(("cam_a", "table")) in pairs
    assert frozenset(("left_gripper", "table")) in pairs
    assert not result["execution_authorized"]
    points = np.asarray(result["nearest_points_m"])
    assert np.linalg.norm(points[0] - points[1]) == pytest.approx(
        result["min_distance"], abs=1e-6
    )


def test_camera_pose_composes_optical_and_housing_frames_and_tracks_wrist(scene):
    q_a, q_b = scene.initial["left"].copy(), scene.initial["right"].copy()
    row = next(a for a in scene.raw["attachments"] if a["name"] == "cam_a")
    row["optical_from_housing_m"][2][3] = -0.012
    modified = CollisionScene(scene.raw)
    first = modified.transforms(q_a, q_b)["cam_a"]
    q_a[5] += 0.5
    actual = modified.transforms(q_a, q_b)["cam_a"]
    model = modified.models["left"]
    flange = model.frame_at(np.degrees(q_a), "link_eef")
    flange[:3, 3] /= 1000
    expected = (
        np.asarray(modified.raw["arms"]["left"]["world_from_base_m"])
        @ flange
        @ np.asarray(row["link_from_optical_m"])
        @ np.asarray(row["optical_from_housing_m"])
    )
    assert actual == pytest.approx(expected)
    assert not np.allclose(actual, first)


def test_arbitrary_cross_arm_exclusions_are_forbidden(scene):
    raw = copy.deepcopy(scene.raw)
    raw["disabled_pairs"] = [["cam_a", "cam_b"]]
    with pytest.raises(DualArmError, match="forbidden"):
        CollisionScene(raw)


def test_only_direct_link_neighbors_and_fixed_mount_contacts_are_excluded(scene):
    for pair, reason in scene.exclusions.items():
        a, b = [next(s for s in scene.shapes if s.name == n) for n in pair]
        assert a.arm == b.arm  # No obstacle base exemption in this demo.
        assert {a.name, b.name} != {"cam_a", "left_gripper"}
        assert reason


@pytest.mark.parametrize(
    "delta,collision,safe",
    [(0.06, False, True), (0.005, False, False), (-0.005, True, False)],
)
def test_fcl_safe_near_and_overlapping_boxes_return_metric_distances(
    scene, delta, collision, safe
):
    # Isolate a camera pair high above both actual URDF robots. Same check()
    # method still checks every robot, self, tool and table pair.
    raw = copy.deepcopy(scene.raw)
    for row, x in zip(
        [a for a in raw["attachments"] if a["kind"] == "camera"], [0, 0.09 + delta]
    ):
        arm = row["arm"]
        model = scene.models[arm]
        flange = model.frame_at(np.degrees(scene.initial[arm]), "link_eef")
        flange[:3, 3] /= 1000
        world_flange = np.asarray(raw["arms"][arm]["world_from_base_m"]) @ flange
        target = np.eye(4)
        target[:3, 3] = [x, 0, 3]
        row["link_from_optical_m"] = (np.linalg.inv(world_flange) @ target).tolist()
        row["optical_from_housing_m"] = np.eye(4).tolist()
    result = check(CollisionScene(raw))
    assert result["collision"] == collision and result["safe"] == safe
    if delta < 0.01:
        assert set(result["closest_pair"]) == {"cam_a", "cam_b"}
        assert result["min_distance"] == pytest.approx(max(0, delta), abs=1e-7)


def test_containment_in_solid_geometry_is_collision(scene):
    raw = copy.deepcopy(scene.raw)
    center = scene.transforms(scene.initial["left"], scene.initial["right"])["cam_a"]
    raw["obstacles"].append(
        {
            "name": "enclosing_box",
            "size_m": [0.8, 0.8, 0.8],
            "world_from_box_m": center.tolist(),
        }
    )
    result = check(CollisionScene(raw))
    assert result["collision"] and not result["safe"]
    assert any(
        set(r["pair"]) == {"cam_a", "enclosing_box"} for r in result["unsafe_pairs"]
    )


def test_nonadjacent_self_collision_is_not_ignored(scene):
    # Fixed folded posture exercises actual xArm6 meshes, independently of
    # cross-arm and table collisions.
    checker = CollisionChecker(scene)
    q = [
        -0.2892795590224484,
        1.4914258936713178,
        -0.41756276480415533,
        -0.43633467637204015,
        -1.0685413383363036,
        2.0791256044553674,
    ]
    result = checker.check(q, scene.initial["right"])
    assert not result["safe"]
    assert any(
        r["collision"] and set(r["pair"]) == {"left/link_base/0", "left/link4/0"}
        for r in result["unsafe_pairs"]
    )


@pytest.mark.parametrize(
    "names",
    [
        ("left_gripper", "right_gripper"),
        ("cam_a", "right/link5/0"),
        ("left/link5/0", "right/link5/0"),
        ("cam_b", "table"),
    ],
)
def test_cross_arm_and_table_solid_overlap_detected(scene, names):
    # Place one body on the other in world coordinates, preserving the full
    # production pair set. This tests FCL transforms as well as pair inclusion.
    q_a, q_b = scene.initial["left"], scene.initial["right"]
    frames = scene.transforms(q_a, q_b)
    shape = next(s for s in scene.shapes if s.name == names[0])
    parent = frames[shape.name] @ np.linalg.inv(shape.local)
    shape.local = np.linalg.inv(parent) @ frames[names[1]]
    result = CollisionChecker(scene).check(q_a, q_b)
    assert result["collision"] and not result["safe"]
    assert any(
        r["collision"] and set(r["pair"]) == set(names) for r in result["unsafe_pairs"]
    )


def test_native_gripper_meshes_require_explicit_envelope_subtree(scene):
    assert len(scene.gripper_mesh_replacements["right"]) == 7
    raw = copy.deepcopy(scene.raw)
    del raw["arms"]["right"]["native_gripper_root"]
    with pytest.raises(DualArmError, match="no coverage"):
        CollisionScene(raw)
    raw["arms"]["right"]["native_gripper_root"] = "world"
    with pytest.raises(DualArmError, match="fixed below the flange"):
        CollisionScene(raw)


def test_safety_distance_and_geometry_uncertainty_change_safe_result(scene):
    original = check(scene)
    raw = copy.deepcopy(scene.raw)
    raw["safety_distance_m"] = original["min_distance"] + 0.001
    result = check(CollisionScene(raw))
    assert not result["collision"] and not result["safe"]
    raw = copy.deepcopy(scene.raw)
    raw["geometry_uncertainty_m"]["left"] = 0.04
    assert not check(CollisionScene(raw))["safe"]


@pytest.mark.parametrize(
    "change",
    [
        "missing_camera",
        "missing_gripper",
        "missing_table",
        "wrong_units",
        "negative_box",
        "camera_optical_missing",
    ],
)
def test_missing_or_bad_scene_data_fails_closed(scene, change):
    raw = copy.deepcopy(scene.raw)
    if change == "missing_camera":
        raw["attachments"] = [a for a in raw["attachments"] if a["name"] != "cam_b"]
    elif change == "missing_gripper":
        raw["attachments"] = [
            a for a in raw["attachments"] if a["name"] != "left_gripper"
        ]
    elif change == "missing_table":
        raw["obstacles"] = []
    elif change == "wrong_units":
        raw["units"] = "mm_deg"
    elif change == "negative_box":
        raw["attachments"][0]["size_m"][0] = -0.1
    else:
        raw["attachments"][0]["optical_from_housing_m"] = None
    with pytest.raises((DualArmError, TypeError)):
        CollisionScene(raw)


def test_joint_count_nonfinite_and_limits_rejected(scene):
    checker = CollisionChecker(scene)
    for q in ([0] * 7, [float("nan")] * 6, [100] * 6):
        with pytest.raises(DualArmError):
            checker.check(q, scene.initial["right"])


def evidence_for(scene):
    rows = {}
    for key, model in scene.models.items():
        pose = model.forward(np.degrees(scene.initial[key]))
        rows[key] = {
            "serial": scene.raw["arms"][key]["serial"],
            "q_rad": scene.initial[key].tolist(),
            "tcp_base_m_rad": [*(pose[:3] / 1000), *np.radians(pose[3:])],
        }
    tcps = scene.tcp_world(scene.initial["left"], scene.initial["right"])
    distance = float(np.linalg.norm(tcps["left"][:3, 3] - tcps["right"][:3, 3]))
    return {
        "units": "m_rad",
        "capture_id": "synthetic-test",
        "captured_at": "2026-10-08T00:00:00Z",
        "arms": rows,
        "distances": [
            {
                "label": "TCP origins",
                "a": {"arm": "left", "frame": "tcp", "point_m": [0, 0, 0]},
                "b": {"arm": "right", "frame": "tcp", "point_m": [0, 0, 0]},
                "measured_m": distance,
                "tolerance_m": 0.002,
            }
        ],
    }


def test_same_endpoint_distance_and_fk_evidence_detect_bad_calibration(scene):
    evidence = evidence_for(scene)
    assert validate_evidence(scene, evidence)["numerical_consistency"]
    evidence["distances"][0]["measured_m"] = 0.45
    report = validate_evidence(scene, evidence)
    assert not report["numerical_consistency"] and not report["physical_acceptance"]
    assert report["distances"][0]["error_m"] > 0.1
    evidence = evidence_for(scene)
    evidence["arms"]["right"]["tcp_base_m_rad"][0] += 0.005
    assert not validate_evidence(scene, evidence)["fk"]["right"]["pass"]


def test_viser_joint_vectors_boxes_and_closest_pair_update_without_sdk(
    scene, monkeypatch
):
    import viser.extras

    from cloth_agent.dual_arm.collision.viewer import CollisionViewer

    class Handle:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def on_update(self, fn):
            self.callback = fn
            return fn

        def remove(self):
            pass

    class Api:
        def __getattr__(self, name):
            return lambda *a, **kw: Handle(value=kw.get("initial_value"), **kw)

    class Robot:
        def __init__(self, *a, **kw):
            pass

        def update_cfg(self, q):
            self.cfg = q.copy()

    monkeypatch.setattr(viser.extras, "ViserUrdf", Robot)
    viewer = CollisionViewer(SimpleNamespace(scene=Api(), gui=Api()), scene)
    before = np.asarray(viewer.handles["cam_a"].position).copy()
    viewer.q["left"][5] += 0.2
    viewer.update()
    assert not np.allclose(viewer.handles["cam_a"].position, before)
    assert len(viewer.robots["left"].cfg) == 6
    assert len(viewer.robots["right"].cfg) == 8  # 7 arm joints + visual drive_joint
    assert viewer.robots["right"].cfg[:7] == pytest.approx(scene.initial["right"])
    assert viewer.last_result["closest_pair"]
    assert "OFFLINE" in viewer.status.content
