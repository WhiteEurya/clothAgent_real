from copy import deepcopy
import json

from PIL import Image

from cloth_agent.trajectory_memory import prepare_trajectory_memory


def physical_attempt(root, *, iteration=1):
    before = root / f"iteration_{iteration}" / "camera_0_A.png"
    before.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30), "pink").save(before)
    actions = [
        {"name": "move", "args": {"x": 500, "y": 40, "z": 90, "yaw": 15}},
        {"name": "open_gripper", "args": {}},
        {"name": "move", "args": {"x": 500, "y": 40, "z": 30, "yaw": 15}},
        {"name": "close_gripper", "args": {}},
        {"name": "move", "args": {"x": 500, "y": 40, "z": 60, "yaw": 15}},
        {"name": "move", "args": {"x": 520, "y": 50, "z": 60, "yaw": 15}},
    ]
    actual = deepcopy(actions)
    for a in actual:
        a["success"] = True
        if a["name"] == "move":
            a["actual_ee_pose"] = [a["args"][k] + .1 for k in ("x", "y", "z")] + [180, 0, 0]
        a["robot_state"] = {"secret": "private_robot"}
    model = deepcopy(actions)
    model[2]["args"]["z"] = 32
    return {"iteration": iteration, "planned_step": "left_sleeve", "mode": "FOLD",
        "proposal": {"actions": model}, "execution_proposal": {"actions": actions},
        "execution": {"physical_execution": True, "execution_completed": True,
            "actual_robot_actions": actual},
        "before_images": [str(before)], "after_images": [],
        "planning_diagnostics": {"selected_reference_validation": {
            "camera": "A", "reference_id": "R001", "pixel_xy": [10, 20]}},
        "evaluation": {"grasp_acquisition": {"status": "FAILURE", "evidence": ["Sleeve stayed on table"]},
            "next_experiment": {"keep": ["jaw direction"], "change": ["contact location"]}}}


def test_preserves_proposed_validated_and_measured_trajectory(tmp_path):
    row = physical_attempt(tmp_path)
    memory, images = prepare_trajectory_memory([row], "left_sleeve", tmp_path, tmp_path / "memory")
    attempt = memory["previous_physical_attempt"]
    assert attempt["model_proposed"]["actions"][2]["commanded_offset_from_contact_mm"] == [0, 0, 2]
    assert attempt["host_validated"]["actions"][2]["commanded_offset_from_contact_mm"] == [0, 0, 0]
    assert attempt["execution_log"]["actions"][4]["measured_offset_from_contact_mm"] == [.1, .1, 30.1]
    assert attempt["execution_log"]["actions"][5]["commanded_offset_from_contact_mm"] == [20, 10, 30]
    assert attempt["grasp_in_before_image"]["pixel_xy"] == [9, 10]
    assert Image.open(images[0]).size == (30, 40)
    text = json.dumps(memory)
    assert str(tmp_path) not in text
    assert "private_robot" not in text
    assert "actual_ee_pose" not in text
    assert '"x"' not in text
    assert json.loads((tmp_path / "memory/trajectory_memory.json").read_text()) == memory


def test_selects_physical_attempt_beyond_recent_failures_and_excludes_other_steps(tmp_path):
    row = physical_attempt(tmp_path)
    failures = [{"iteration": i, "planned_step": "left_sleeve", "mode": "PLANNING_FAILURE",
        "planning_failure": {"fold_command_sent": False}} for i in range(2, 12)]
    inherited = {**row, "iteration": 15, "inherited_lesson": True}
    unrelated = {**row, "iteration": 16, "planned_step": "right_sleeve"}
    memory, _ = prepare_trajectory_memory([row, *failures, inherited, unrelated],
        "left_sleeve", tmp_path, tmp_path / "memory")
    assert memory["previous_physical_attempt"]["iteration"] == 1
    assert memory["latest_attempt"]["iteration"] == 11
    assert memory["latest_attempt"]["execution_status"] == "NOT_PHYSICALLY_EXECUTED"


def test_failed_or_unfinished_action_is_not_a_completed_trajectory(tmp_path):
    row = physical_attempt(tmp_path)
    row["execution"]["execution_completed"] = False
    row["execution"]["actual_robot_actions"] = row["execution"]["actual_robot_actions"][:4]
    row["execution"]["actual_robot_actions"][-1].update(success=False, error="RobotExecutionError: closure stalled")
    memory, _ = prepare_trajectory_memory([row], "left_sleeve", tmp_path, tmp_path / "memory")
    attempt = memory["previous_physical_attempt"]
    assert attempt["execution_status"] == "PHYSICAL_INCOMPLETE_OR_UNKNOWN"
    assert attempt["host_validated"]["action_count"] == 6
    assert attempt["execution_log"]["action_count"] == 4
    assert attempt["execution_log"]["actions"][-1]["completion"] == "ATTEMPTED_NOT_CONFIRMED"
    assert attempt["execution_log"]["actions"][-1]["error_type"] == "RobotExecutionError"


def test_simulation_and_rejected_proposals_are_not_physical_experience(tmp_path):
    row = physical_attempt(tmp_path)
    row["execution"]["physical_execution"] = False
    memory, images = prepare_trajectory_memory([row], "left_sleeve", tmp_path, tmp_path / "memory")
    assert memory["previous_physical_attempt"] is None
    assert images == []
    assert memory["latest_attempt"]["execution_status"] == "NOT_PHYSICALLY_EXECUTED"
    del row["execution"]
    row["planning_failure"] = {"fold_command_sent": False}
    memory, _ = prepare_trajectory_memory([row], "left_sleeve", tmp_path, tmp_path / "memory")
    assert memory["previous_physical_attempt"] is None


def test_missing_log_is_unknown_even_with_physical_flag(tmp_path):
    row = physical_attempt(tmp_path)
    row["execution"] = {"physical_execution": True, "requested_robot_actions": row["proposal"]["actions"]}
    memory, _ = prepare_trajectory_memory([row], "left_sleeve", tmp_path, tmp_path / "memory")
    assert memory["previous_physical_attempt"] is None
    assert memory["latest_attempt"]["execution_status"] == "PHYSICAL_INCOMPLETE_OR_UNKNOWN"


def test_missing_mutable_and_external_images_are_not_replaced_with_current_rgb(tmp_path):
    run = tmp_path / "run"
    row = physical_attempt(run)
    external = tmp_path / "camera_0_A.png"
    mutable = run / "workspace/perception_views/camera_0_A.png"
    mutable.parent.mkdir(parents=True)
    for path in [external, mutable]:
        Image.new("RGB", (40, 30), "white").save(path)
    row["before_images"] = [str(external), str(mutable), str(run / "missing/camera_0_A.png")]
    memory, images = prepare_trajectory_memory([row], "left_sleeve", run, run / "memory")
    assert images == []
    assert memory["previous_physical_attempt"]["grasp_in_before_image"] is None
    assert all(e["status"] == "UNAVAILABLE" for e in memory["previous_physical_attempt"]["images"])


def test_grasp_snapshots_preserve_capture_caveats(tmp_path):
    row = physical_attempt(tmp_path)
    snapshots = {}
    for role in ("before_lift", "after_close", "after_lift"):
        path = tmp_path / f"camera_A_grasp_{role}.png"
        Image.new("RGB", (40, 30), "pink").save(path)
        snapshots[role] = {"status": "CAPTURED", "image": str(path), "action_index": 4}
    row["recording"] = {"grasp_snapshots": snapshots}
    memory, images = prepare_trajectory_memory([row], "left_sleeve", tmp_path, tmp_path / "memory")
    assert len(images) == 4
    evidence = {e["role"]: e for e in memory["previous_physical_attempt"]["images"]}
    assert "may be during closure" in evidence["before_lift"]["capture_note"]
    assert "subsequent motion" in evidence["after_lift"]["capture_note"]
    assert evidence["after_close"]["source_action_index"] == 4


def test_other_step_or_only_inherited_history_gives_no_memory(tmp_path):
    row = physical_attempt(tmp_path)
    assert prepare_trajectory_memory([row], "right_sleeve", tmp_path, tmp_path / "memory") == (None, [])
    row["inherited_lesson"] = True
    assert prepare_trajectory_memory([row], "left_sleeve", tmp_path, tmp_path / "memory") == (None, [])
