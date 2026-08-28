from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from cloth_agent.collar_lift_retreat import (
    CollarLiftRetreatError,
    collar_motion_payload_to_proposal,
    invoke_claude_collar_motion_planner,
    invoke_claude_collar_selector,
    insert_runtime_shake_open_after_y0_center,
    validate_claude_collar_motion_proposal,
    validate_collar_selection_payload,
    validate_controller_with_auto_far_x_repair,
    validate_grounded_collar_grasp_feasibility,
)
from cloth_agent.config import RobotConfig, SafetyError, WorkspaceBounds
from cloth_agent.experiment import validate_experiment_source
from cloth_agent.free_exploration import exploration_source, validate_global_exploration_payload


def _robot_config() -> RobotConfig:
    return RobotConfig(
        robot_ip="127.0.0.1",
        boundaries=WorkspaceBounds(
            x_min=350,
            x_max=800,
            y_min=-300,
            y_max=170,
            z_min=6,
            z_max=400,
        ),
        init_joints_deg=(0, 0, 0, 0, 0, 0, 0),
        init_pose_mm_deg=(500, 0, 280, 180, 0, 0),
        orientation_roll_deg=180,
        orientation_pitch_deg=0,
        workspace_margin_mm=1,
        lower_z_margin_mm=0,
    )


def _valid_claude_actions() -> list[dict[str, object]]:
    """One arbitrary Claude candidate used only to exercise validators."""

    return [
        {"name": "move", "args": {"x": 500.0, "y": 40.0, "z": 109.0, "yaw": 0.0}},
        {"name": "open_gripper", "args": {}},
        {"name": "move", "args": {"x": 500.0, "y": 40.0, "z": 27.5, "yaw": 0.0}},
        {"name": "close_gripper", "args": {}},
        {"name": "move", "args": {"x": 500.0, "y": 40.0, "z": 395.0, "yaw": 0.0}},
        {"name": "move", "args": {"x": 500.0, "y": 0.0, "z": 395.0, "yaw": 0.0}},
        {"name": "move", "args": {"x": 500.0, "y": 0.0, "z": 395.0, "yaw": 0.0}},
        {"name": "move", "args": {"x": 700.0, "y": 0.0, "z": 395.0, "yaw": 0.0}},
        {"name": "move", "args": {"x": 600.0, "y": 0.0, "z": 295.0, "yaw": 0.0}},
        {"name": "move", "args": {"x": 490.0, "y": 0.0, "z": 185.0, "yaw": 0.0}},
        {"name": "move", "args": {"x": 380.0, "y": 0.0, "z": 75.0, "yaw": 0.0}},
        {"name": "open_gripper", "args": {}},
        {"name": "move", "args": {"x": 380.0, "y": 0.0, "z": 110.0, "yaw": 0.0}},
        {"name": "home", "args": {}},
    ]


def _valid_compact_motion_payload() -> dict[str, object]:
    return {
        "garment_observation": "The collar rim is graspable.",
        "reveal_strategy": "Center at Y=0, move high toward +X, then descend gradually.",
        "expected_observation": "The shirt spreads during the retreat.",
        "confidence": 0.72,
        "safety_notes": ["Controller IK remains authoritative."],
        "yaw_deg": 0.0,
        "approach_xyz_mm": [500.0, 40.0, 109.0],
        "lift_xyz_mm": [500.0, 40.0, 395.0],
        "y0_center_xyz_mm": [500.0, 0.0, 395.0],
        "pretransport_lower_xyz_mm": [500.0, 0.0, 395.0],
        "far_transport_xyz_mm": [700.0, 0.0, 395.0],
        "descent_xyz_mm": [
            [600.0, 0.0, 295.0],
            [490.0, 0.0, 185.0],
            [380.0, 0.0, 75.0],
        ],
        "retract_xyz_mm": [380.0, 0.0, 110.0],
    }


def _surface_measurement(
    *,
    xyz: tuple[float, float, float] = (500.0, 40.0, 30.0),
    table_z_mm: float = 5.0,
) -> dict[str, object]:
    return {
        "valid": True,
        "base_xyz_median_mm": list(xyz),
        "base_xyz_p10_mm": [xyz[0], xyz[1], xyz[2] - 0.5],
        "base_xyz_p90_mm": [xyz[0], xyz[1], xyz[2] + 0.5],
        "base_z_p90_minus_p10_mm": 1.0,
        "table_z_median_mm": table_z_mm,
    }


def _grasp_height_plan() -> dict[str, object]:
    return validate_grounded_collar_grasp_feasibility(
        measurement=_surface_measurement(),
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=_robot_config(),
    )


def test_collar_selection_accepts_claude_selected_camera_a_fabric_pixel() -> None:
    selection = validate_collar_selection_payload(
        {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.88,
            "evidence": ["A visible curved neckline rim is present."],
            "reference_correspondence": ["The neck opening matches the flat reference topology."],
            "molmo_guided_region_evidence": ["Molmo, RGB, depth, and reference agree on the collar neighborhood."],
            "grasp_point_evidence": ["The independently selected point is dark collar fabric away from the tag."],
            "molmo_relation": "USES_MOLMO_COLLAR_REGION",
            "reason": "Point lies just inside collar fabric, away from the neck hole.",
        },
        image_width=1280,
        image_height=720,
    )

    assert selection.camera == "A"
    assert selection.pixel_xy == (520, 180)
    assert selection.neck_side_opposition_cosine == pytest.approx(0.8804711)

    with pytest.raises(CollarLiftRetreatError, match="not on the neck side"):
        validate_collar_selection_payload(
            {
                **selection.as_dict(),
                "pixel_xy": [448, 458],
                "neck_label_pixel_xy": [434, 422],
                "torso_landmark_pixel_xy": [550, 360],
                "neck_side_opposition_cosine": 0.12,
            },
            image_width=1280,
            image_height=720,
        )

    lower_confidence = validate_collar_selection_payload(
        {
            **selection.as_dict(),
            "pixel_xy": [520, 180],
            "confidence": 0.62,
        },
        image_width=1280,
        image_height=720,
    )
    assert lower_confidence.confidence == pytest.approx(0.62)

    with pytest.raises(CollarLiftRetreatError, match="below required"):
        validate_collar_selection_payload(
            {
                **selection.as_dict(),
                "pixel_xy": [520, 180],
                "confidence": 0.62,
            },
            image_width=1280,
            image_height=720,
            min_confidence=0.65,
        )


def test_collar_selection_can_fail_closed_when_neckline_is_not_visible() -> None:
    selection = validate_collar_selection_payload(
        {
            "status": "NOT_FOUND",
            "camera": None,
            "pixel_xy": None,
            "neck_label_pixel_xy": None,
            "torso_landmark_pixel_xy": None,
            "neck_side_opposition_cosine": None,
            "confidence": 0.2,
            "evidence": ["The neckline is fully occluded."],
            "reference_correspondence": ["Reference topology cannot be matched in the fold."],
            "molmo_guided_region_evidence": ["Molmo did not provide an accepted collar region."],
            "grasp_point_evidence": ["No nearby on-mask fabric candidate has valid depth."],
            "molmo_relation": "MOLMO_COLLAR_UNAVAILABLE",
            "reason": "No Camera A collar point can be selected without guessing.",
        },
        image_width=1280,
        image_height=720,
    )
    assert selection.status == "NOT_FOUND"
    assert selection.pixel_xy is None


def test_collar_selector_uses_claude_strict_schema_subset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    image_dir = run_dir / "results" / "perception"
    image_dir.mkdir(parents=True)
    image = image_dir / "camera_0_A.png"
    image.write_bytes(b"image")
    camera_b = image_dir / "camera_1_B.png"
    camera_b.write_bytes(b"image")
    reference = image_dir / "camera_A_flat_reference.png"
    reference.write_bytes(b"image")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "cloth_agent.collar_lift_retreat.shutil.which", lambda _: "/usr/bin/claude"
    )

    def fake_run(command, **kwargs):
        seen["command"] = command
        payload = {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.9,
            "evidence": ["Visible collar rim."],
            "reference_correspondence": ["Reference neckline topology matches."],
            "molmo_guided_region_evidence": ["Molmo plus RGB and depth identify the collar neighborhood."],
            "grasp_point_evidence": ["The chosen point is nearby collar fabric but not the Molmo pixel."],
            "molmo_relation": "USES_MOLMO_COLLAR_REGION",
            "reason": "The point is inside collar fabric.",
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"structured_output": payload}),
            stderr="",
        )

    monkeypatch.setattr("cloth_agent.collar_lift_retreat.subprocess.run", fake_run)

    selection, log = invoke_claude_collar_selector(
        [image, camera_b, reference],
        run_dir=run_dir,
        image_width=1280,
        image_height=720,
        molmo_manifest={
            "status": "READY",
            "anchors": [
                {"type": "collar", "camera": "A", "pixel_xy": [520, 180], "confidence": 0.9}
            ],
            "axis_references": {"A": {"top_pixel_xy": [520, 180], "bottom_pixel_xy": [600, 650]}},
            "views": [],
        },
        reference_anchors={
            "axis_reference": {"top_pixel_xy": [598, 212], "bottom_pixel_xy": [653, 629]},
            "anchors": [
                {"name": "neckline", "selected_pixel_xy": [598, 212], "description": "neckline"}
            ],
        },
    )

    command = seen["command"]
    assert isinstance(command, list)
    schema = json.loads(command[command.index("--json-schema") + 1])
    schema_text = json.dumps(schema)
    assert "prefixItems" not in schema_text
    pixel_array = schema["properties"]["pixel_xy"]["anyOf"][0]
    assert pixel_array["items"] == {"type": "integer", "minimum": 0}
    assert "reference_correspondence" in schema["required"]
    assert "molmo_guided_region_evidence" in schema["required"]
    assert "grasp_point_evidence" in schema["required"]
    assert "neck_label_pixel_xy" in schema["required"]
    assert "torso_landmark_pixel_xy" in schema["required"]
    assert "neck_side_opposition_cosine" in schema["required"]
    assert str(reference) in log["prompt"]
    assert str(camera_b) in log["prompt"]
    assert "Optional supporting files (read only if useful)" in log["prompt"]
    assert "Do not use Camera B" in log["prompt"]
    assert "A clearly visible neck-opening contour is NOT required" in log["prompt"]
    assert "Treat every Molmo point as a fallible hypothesis" in log["prompt"]
    assert "You may search outside the immediate label neighborhood only when" in log["prompt"]
    assert "CORRECTS_MOLMO_COLLAR_WITH_NECK_LABEL" in log["prompt"]
    assert selection.pixel_xy == (520, 180)


def test_collar_selector_supports_no_molmo_reference_only_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    image_dir = run_dir / "results" / "perception"
    image_dir.mkdir(parents=True)
    image = image_dir / "camera_0_A.png"
    image.write_bytes(b"image")
    reference = image_dir / "camera_A_flat_reference.png"
    reference.write_bytes(b"reference")
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "cloth_agent.collar_lift_retreat.shutil.which", lambda _: "/usr/bin/claude"
    )

    def fake_run(command, **kwargs):
        seen["command"] = command
        payload = {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.9,
            "evidence": ["RGB and reference identify collar fabric."],
            "reference_correspondence": ["Reference topology matches."],
            "molmo_guided_region_evidence": ["No Molmo result was used."],
            "grasp_point_evidence": ["Point is on collar fabric."],
            "molmo_relation": "MOLMO_COLLAR_UNAVAILABLE",
            "reason": "The collar is localized from RGB-D and reference only.",
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"structured_output": payload}),
            stderr="",
        )

    monkeypatch.setattr("cloth_agent.collar_lift_retreat.subprocess.run", fake_run)
    selection, log = invoke_claude_collar_selector(
        [image, reference],
        run_dir=run_dir,
        image_width=1280,
        image_height=720,
        molmo_manifest=None,
        reference_anchors={"axis_reference": {}, "anchors": []},
    )
    assert selection.status == "SELECTED"
    assert "Molmo is intentionally unavailable" in log["prompt"]
    assert "MOLMO_COLLAR_UNAVAILABLE" in log["prompt"]


def test_claude_chosen_collar_motion_preserves_grounded_grasp_and_task_shape() -> None:
    robot = _robot_config()
    selection = validate_collar_selection_payload(
        {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.72,
            "evidence": ["Visible doubled collar rim."],
            "reference_correspondence": ["Reference neckline topology matches."],
            "molmo_guided_region_evidence": ["Molmo plus current RGB/depth identify the collar neighborhood."],
            "grasp_point_evidence": ["The selected point is nearby dark collar fabric away from the tag."],
            "molmo_relation": "USES_MOLMO_COLLAR_REGION",
            "reason": "Point lies inside the collar fabric.",
        },
        image_width=1280,
        image_height=720,
    )
    actions = _valid_claude_actions()
    proposal = insert_runtime_shake_open_after_y0_center(validate_global_exploration_payload(
        {
            "selected_grasp": {
                "camera": "A",
                "pixel_xy": [520, 180],
                "reason": "collar rim",
            },
            "garment_observation": "The collar rim is graspable.",
            "reveal_strategy": "Lift high, move toward +X, and gradually lay down.",
            "confidence": 0.72,
            "actions": actions,
            "expected_observation": "The shirt spreads during the descending retreat.",
            "safety_notes": ["All Claude-selected waypoints require validation."],
        },
        max_actions=14,
    ))

    contract = validate_claude_collar_motion_proposal(
        proposal,
        selection=selection,
        grasp_height_plan=_grasp_height_plan(),
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=robot,
    )

    assert contract["descent_waypoint_count"] == 3
    assert contract["claude_chosen_y0_center"]["y"] == pytest.approx(0.0)
    assert contract["claude_chosen_pretransport_lower"]["z"] == pytest.approx(395.0)
    assert contract["claude_chosen_far_transport"]["x"] == pytest.approx(700.0)
    assert contract["claude_chosen_release"]["x"] == pytest.approx(380.0)
    assert all(
        metric["descent_retreat_ratio"] == pytest.approx(1.0)
        for metric in contract["descent_leg_metrics"]
    )
    source = exploration_source(proposal)
    assert source.index("shake_open()") > source.index("move(500.0, 0.0, 395.0, 0.0)")
    assert source.index("move(700.0, 0.0, 395.0, 0.0)") > source.index("shake_open()")
    assert source.index("move(500.0, 0.0, 395.0, 0.0)", source.index("shake_open()")) < source.index(
        "move(700.0, 0.0, 395.0, 0.0)"
    )
    validate_experiment_source(source)


def test_claude_collar_motion_rejects_non_descending_retreat() -> None:
    robot = _robot_config()
    selection = validate_collar_selection_payload(
        {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.72,
            "evidence": ["Visible collar rim."],
            "reference_correspondence": ["Reference neckline topology matches."],
            "molmo_guided_region_evidence": ["Molmo plus current RGB/depth identify the collar neighborhood."],
            "grasp_point_evidence": ["The selected point is nearby dark collar fabric away from the tag."],
            "molmo_relation": "USES_MOLMO_COLLAR_REGION",
            "reason": "Point lies inside collar fabric.",
        },
        image_width=1280,
        image_height=720,
    )
    actions = _valid_claude_actions()
    actions[8]["args"]["x"] = actions[7]["args"]["x"] + 5.0
    proposal = insert_runtime_shake_open_after_y0_center(validate_global_exploration_payload(
        {
            "selected_grasp": {
                "camera": "A",
                "pixel_xy": [520, 180],
                "reason": "collar rim",
            },
            "garment_observation": "The collar rim is graspable.",
            "reveal_strategy": "Attempt a gradual laydown.",
            "confidence": 0.72,
            "actions": actions,
            "expected_observation": "The shirt spreads.",
            "safety_notes": ["Validate every waypoint."],
        },
        max_actions=14,
    ))

    with pytest.raises(CollarLiftRetreatError, match="backward toward -X"):
        validate_claude_collar_motion_proposal(
            proposal,
            selection=selection,
            grasp_height_plan=_grasp_height_plan(),
            table_plane_abc=[0.0, 0.0, 5.0],
            robot_config=robot,
        )


def test_claude_collar_motion_requires_y0_center_and_near_ceiling_height() -> None:
    selection = validate_collar_selection_payload(
        {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.72,
            "evidence": ["Visible collar rim."],
            "reference_correspondence": ["Reference neckline topology matches."],
            "molmo_guided_region_evidence": ["Molmo plus current RGB/depth identify the collar neighborhood."],
            "grasp_point_evidence": ["The selected point is nearby dark collar fabric away from the tag."],
            "molmo_relation": "USES_MOLMO_COLLAR_REGION",
            "reason": "Point lies inside collar fabric.",
        },
        image_width=1280,
        image_height=720,
    )
    not_centered = _valid_compact_motion_payload()
    not_centered["y0_center_xyz_mm"] = [500.0, 12.0, 395.0]
    proposal = collar_motion_payload_to_proposal(
        not_centered,
        selection=selection,
        grasp_target_xyz_mm=_grasp_height_plan()["target_xyz_mm"],
    )
    with pytest.raises(CollarLiftRetreatError, match="center.*Y=0"):
        validate_claude_collar_motion_proposal(
            proposal,
            selection=selection,
            grasp_height_plan=_grasp_height_plan(),
            table_plane_abc=[0.0, 0.0, 5.0],
            robot_config=_robot_config(),
        )

    too_low = _valid_compact_motion_payload()
    for name in ("lift_xyz_mm", "y0_center_xyz_mm"):
        point = list(too_low[name])
        point[2] = 200.0
        too_low[name] = point
    too_low["pretransport_lower_xyz_mm"] = [500.0, 0.0, 200.0]
    too_low["far_transport_xyz_mm"] = [700.0, 0.0, 200.0]
    proposal = collar_motion_payload_to_proposal(
        too_low,
        selection=selection,
        grasp_target_xyz_mm=_grasp_height_plan()["target_xyz_mm"],
    )
    with pytest.raises(CollarLiftRetreatError, match="high but controller-reachable"):
        validate_claude_collar_motion_proposal(
            proposal,
            selection=selection,
            grasp_height_plan=_grasp_height_plan(),
            table_plane_abc=[0.0, 0.0, 5.0],
            robot_config=_robot_config(),
        )


def test_pretransport_lowering_precedes_transport_and_retreat_is_visible() -> None:
    selection = validate_collar_selection_payload(
        {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.72,
            "evidence": ["Visible collar fabric."],
            "reference_correspondence": ["Reference supports the collar region."],
            "molmo_guided_region_evidence": ["Molmo and depth locate the collar region."],
            "grasp_point_evidence": ["The selected fabric point avoids the tag."],
            "molmo_relation": "USES_MOLMO_COLLAR_REGION",
            "reason": "Collar fabric is graspable.",
        },
        image_width=1280,
        image_height=720,
    )

    moves_x_while_lowering = _valid_compact_motion_payload()
    moves_x_while_lowering["pretransport_lower_xyz_mm"] = [515.0, 0.0, 395.0]
    proposal = collar_motion_payload_to_proposal(
        moves_x_while_lowering,
        selection=selection,
        grasp_target_xyz_mm=_grasp_height_plan()["target_xyz_mm"],
    )
    with pytest.raises(CollarLiftRetreatError, match="stay in place"):
        validate_claude_collar_motion_proposal(
            proposal,
            selection=selection,
            grasp_height_plan=_grasp_height_plan(),
            table_plane_abc=[0.0, 0.0, 5.0],
            robot_config=_robot_config(),
        )

    changes_z_during_transport = _valid_compact_motion_payload()
    changes_z_during_transport["far_transport_xyz_mm"] = [700.0, 0.0, 340.0]
    proposal = collar_motion_payload_to_proposal(
        changes_z_during_transport,
        selection=selection,
        grasp_target_xyz_mm=_grasp_height_plan()["target_xyz_mm"],
    )
    with pytest.raises(CollarLiftRetreatError, match="preserve the pre-transport"):
        validate_claude_collar_motion_proposal(
            proposal,
            selection=selection,
            grasp_height_plan=_grasp_height_plan(),
            table_plane_abc=[0.0, 0.0, 5.0],
            robot_config=_robot_config(),
        )


def test_motion_rejects_high_air_release_even_when_descent_is_monotonic() -> None:
    selection = validate_collar_selection_payload(
        {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.72,
            "evidence": ["Visible collar fabric."],
            "reference_correspondence": ["Reference supports the collar region."],
            "molmo_guided_region_evidence": ["Molmo and depth locate the collar region."],
            "grasp_point_evidence": ["The selected point avoids the tag."],
            "molmo_relation": "USES_MOLMO_COLLAR_REGION",
            "reason": "Collar fabric is graspable.",
        },
        image_width=1280,
        image_height=720,
    )
    payload = _valid_compact_motion_payload()
    payload["descent_xyz_mm"] = [
        [600.0, 0.0, 295.0],
        [500.0, 0.0, 195.0],
        [400.0, 0.0, 95.0],
    ]
    proposal = collar_motion_payload_to_proposal(
        payload,
        selection=selection,
        grasp_target_xyz_mm=_grasp_height_plan()["target_xyz_mm"],
    )
    with pytest.raises(CollarLiftRetreatError, match="final release is still too high"):
        validate_claude_collar_motion_proposal(
            proposal,
            selection=selection,
            grasp_height_plan=_grasp_height_plan(),
            table_plane_abc=[0.0, 0.0, 5.0],
            robot_config=_robot_config(),
        )

    tiny_first_retreat = _valid_compact_motion_payload()
    tiny_first_retreat["descent_xyz_mm"][0] = [695.0, 0.0, 250.0]
    proposal = collar_motion_payload_to_proposal(
        tiny_first_retreat,
        selection=selection,
        grasp_target_xyz_mm=_grasp_height_plan()["target_xyz_mm"],
    )
    with pytest.raises(CollarLiftRetreatError, match="at least 10 mm"):
        validate_claude_collar_motion_proposal(
            proposal,
            selection=selection,
            grasp_height_plan=_grasp_height_plan(),
            table_plane_abc=[0.0, 0.0, 5.0],
            robot_config=_robot_config(),
        )

    wrong_xz_ratio = _valid_compact_motion_payload()
    wrong_xz_ratio["descent_xyz_mm"][0] = [680.0, 0.0, 250.0]
    proposal = collar_motion_payload_to_proposal(
        wrong_xz_ratio,
        selection=selection,
        grasp_target_xyz_mm=_grasp_height_plan()["target_xyz_mm"],
    )
    with pytest.raises(CollarLiftRetreatError, match="balanced descent/retreat ratio"):
        validate_claude_collar_motion_proposal(
            proposal,
            selection=selection,
            grasp_height_plan=_grasp_height_plan(),
            table_plane_abc=[0.0, 0.0, 5.0],
            robot_config=_robot_config(),
        )


def test_grounded_collar_feasibility_rejects_unreachable_flat_collar() -> None:
    feasible = validate_grounded_collar_grasp_feasibility(
        measurement=_surface_measurement(xyz=(500.0, 0.0, 30.0)),
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=_robot_config(),
    )
    assert feasible["valid"] is True
    assert feasible["target_xyz_mm"] == pytest.approx([500.0, 0.0, 27.0])
    assert feasible["achieved_compression_mm"] == pytest.approx(3.0)

    with pytest.raises(CollarLiftRetreatError, match="no legal engaged grasp"):
        validate_grounded_collar_grasp_feasibility(
            measurement=_surface_measurement(
                xyz=(368.8, 24.2, 0.63),
                table_z_mm=0.9,
            ),
            table_plane_abc=[0.0, 0.0, 0.9],
            robot_config=_robot_config(),
        )


def test_grounded_collar_feasibility_rejects_x_boundary_before_motion_planning() -> None:
    with pytest.raises(CollarLiftRetreatError, match="outside the safe Cartesian workspace"):
        validate_grounded_collar_grasp_feasibility(
            measurement=_surface_measurement(xyz=(349.0, 0.0, 30.0)),
            table_plane_abc=[0.0, 0.0, 5.0],
            robot_config=_robot_config(),
        )


def test_motion_replan_prompt_returns_numeric_choice_to_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    image_dir = run_dir / "results" / "perception"
    image_dir.mkdir(parents=True)
    image = image_dir / "camera_0_A.png"
    image.write_bytes(b"image")
    selection = validate_collar_selection_payload(
        {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.72,
            "evidence": ["Visible collar rim."],
            "reference_correspondence": ["Reference neckline topology matches."],
            "molmo_guided_region_evidence": ["Molmo plus current RGB/depth identify the collar neighborhood."],
            "grasp_point_evidence": ["The selected point is nearby dark collar fabric away from the tag."],
            "molmo_relation": "USES_MOLMO_COLLAR_REGION",
            "reason": "Point lies inside collar fabric.",
        },
        image_width=1280,
        image_height=720,
    )
    payload = _valid_compact_motion_payload()
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "cloth_agent.collar_lift_retreat.shutil.which", lambda _: "/usr/bin/claude"
    )

    def fake_run(command, **kwargs):
        seen["command"] = command
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"structured_output": payload}),
            stderr="",
        )

    monkeypatch.setattr("cloth_agent.collar_lift_retreat.subprocess.run", fake_run)
    previous = dict(payload)
    returned_payload, log = invoke_claude_collar_motion_planner(
        run_dir=run_dir,
        selection=selection,
        grasp_height_plan=_grasp_height_plan(),
        table_plane_abc=[0.0, 0.0, 5.0],
        robot_config=_robot_config(),
        previous_proposal=previous,
        validation_error=(
            "SafetyError: controller IK rejected action 6 segment sample 21/28 "
            "pose=[652.3, 109.8, 333.9, 178.3, 3.6, 170.5], code=10"
        ),
    )

    proposal = collar_motion_payload_to_proposal(
        returned_payload,
        selection=selection,
        grasp_target_xyz_mm=_grasp_height_plan()["target_xyz_mm"],
    )
    assert proposal.actions[6]["name"] == "shake_open"
    assert proposal.actions[7]["args"]["x"] == pytest.approx(500.0)
    assert proposal.actions[7]["args"]["z"] == pytest.approx(395.0)
    assert proposal.actions[8]["args"]["x"] == pytest.approx(700.0)
    prompt = log["prompt"]
    assert "runtime-authoritative grasp TCP XYZ" in prompt
    assert "Do not choose or revise the grasp Z" in prompt
    assert "Do not use a fixed canned far-X" in prompt
    assert "balanced descent/retreat ratio" in prompt
    assert "controller IK rejected action 6" in prompt
    assert "do not request or inspect files" in prompt
    assert "do not return an action DSL" in prompt
    command = log["command"]
    assert command[command.index("--tools") + 1] == ""
    assert "--add-dir" not in command
    schema = json.loads(command[command.index("--json-schema") + 1])
    assert "actions" not in schema["properties"]
    assert "grasp_xyz_mm" not in schema["properties"]
    assert schema["properties"]["descent_xyz_mm"]["minItems"] == 3
    assert schema["properties"]["descent_xyz_mm"]["maxItems"] == 3
    assert schema["properties"]["y0_center_xyz_mm"]["maxItems"] == 3
    assert schema["properties"]["pretransport_lower_xyz_mm"]["maxItems"] == 3


def test_controller_ik_far_x_failure_is_repaired_without_another_claude_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = validate_collar_selection_payload(
        {
            "status": "SELECTED",
            "camera": "A",
            "pixel_xy": [520, 180],
            "neck_label_pixel_xy": [540, 200],
            "torso_landmark_pixel_xy": [600, 400],
            "neck_side_opposition_cosine": 0.88,
            "confidence": 0.72,
            "evidence": ["Collar fabric is visible."],
            "reference_correspondence": ["Reference supports the collar region."],
            "molmo_guided_region_evidence": [
                "Molmo and current depth locate the collar region."
            ],
            "grasp_point_evidence": [
                "The selected point avoids the Molmo pixel and sewn tag."
            ],
            "molmo_relation": "USES_MOLMO_COLLAR_REGION",
            "reason": "Nearby collar fabric is graspable.",
        },
        image_width=1280,
        image_height=720,
    )
    proposal = collar_motion_payload_to_proposal(
        _valid_compact_motion_payload(),
        selection=selection,
        grasp_target_xyz_mm=_grasp_height_plan()["target_xyz_mm"],
    )
    tested_far_x: list[float] = []

    def fake_controller(arm, config, actions):
        far_x = float(actions[8]["args"]["x"])
        tested_far_x.append(far_x)
        if far_x > 680.0:
            raise SafetyError(
                "controller IK rejected action 9 segment sample 3/7 "
                f"pose=[{far_x}, 0, 350, 180, 0, 0], code=10"
            )
        return {"status": "IK_ACCEPTED", "far_x_mm": far_x}

    monkeypatch.setattr(
        "cloth_agent.collar_lift_retreat._controller_trajectory_with_arm",
        fake_controller,
    )

    repaired, validation, repair = validate_controller_with_auto_far_x_repair(
        _robot_config(),
        proposal,
        arm=object(),
    )

    assert repair is not None
    assert repair["status"] == "AUTO_REPAIRED"
    selected_far_x = float(repaired.actions[8]["args"]["x"])
    assert 671.0 <= selected_far_x <= 680.0
    assert validation["far_x_mm"] == pytest.approx(selected_far_x)
    assert tested_far_x[0] == pytest.approx(700.0)
    assert any(value <= 680.0 for value in tested_far_x[1:])
    assert repaired.actions[8]["args"]["z"] == proposal.actions[8]["args"]["z"]
    repaired_descent_x = [float(repaired.actions[index]["args"]["x"]) for index in (9, 10, 11)]
    repaired_descent_z = [float(repaired.actions[index]["args"]["z"]) for index in (9, 10, 11)]
    prior_x, prior_z = selected_far_x, float(repaired.actions[8]["args"]["z"])
    for point_x, point_z in zip(repaired_descent_x, repaired_descent_z):
        assert prior_x - point_x == pytest.approx(prior_z - point_z)
        prior_x, prior_z = point_x, point_z
