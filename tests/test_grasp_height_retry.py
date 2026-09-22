from copy import deepcopy

import pytest

from cloth_agent.grasp_height_retry import (
    retry_eligibility, height_options, compile_height_retry, validate_locked_retry,
)
from cloth_agent.free_exploration import ExplorationProposal
from cloth_agent.fold_exploration_pipeline import build_parser


def record():
    def move(x, z):
        return {"name": "move", "args": {"x": x, "y": 0., "z": z, "yaw": 0.}}
    actions = [move(300, 100), {"name": "open_gripper", "args": {}}, move(300, 20),
        {"name": "close_gripper", "args": {}}, move(300, 60), move(400, 60), move(400, 20),
        {"name": "open_gripper", "args": {}}, move(400, 100), {"name": "home", "args": {}}]
    proposal = ExplorationProposal(garment_observation="shirt", reveal_strategy="fold sleeve", confidence=.9,
        actions=tuple(actions), expected_observation="folded sleeve", safety_notes=("safety",),
        selected_grasp={"camera": "A", "pixel_xy": [2, 3]})
    return {"record_id": "parent", "iteration": 1, "planned_step": "left_sleeve", "mode": "FOLD",
        "execution_proposal": proposal.as_dict(),
        "execution": {"execution_completed": True, "physical_execution": True,
            "actual_robot_actions": [{**deepcopy(a), "success": True} for a in actions]},
        "evaluation": {"grasp_acquisition": {"status": "FAILURE"}, "earliest_failure_stage": "ACQUISITION",
                       "perception_comparison": {"status": "UNCHANGED"}},
        "planning_diagnostics": {"grasp_height_resolution": {"camera": "A", "pixel_xy": [2, 3],
            "resolved_grasp_xy_mm": [300, 0], "resolved_grasp_z_mm": 20,
            "resolution": {"valid": True, **resolution()}}}}


def resolution():
    return {"surface_xyz_mm": [300, 0, 23], "lower_z_mm": 5,
            "minimum_compression_mm": .75, "maximum_compression_mm": 5}


def test_failure_schedules_an_experiment_without_claiming_shallow_cause():
    row = record()
    original = deepcopy(row)
    followup = retry_eligibility(row)
    assert followup["status"] == "SCHEDULED"
    assert followup["causal_claim"] == "NONE"
    assert followup["maximum_secondary_attempts"] == 1
    assert row == original


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(height_retry={"parent_iteration": 1}),
    lambda r: r.update(inherited_lesson=True),
    lambda r: r.update(status="INTERRUPTED"),
    lambda r: r.update(mode="PLANNING_FAILURE"),
    lambda r: r["evaluation"].update(grasp_acquisition={"status": "UNKNOWN"}),
    lambda r: r["evaluation"].update(grasp_acquisition={"status": "SUCCESS"}),
    lambda r: r["execution"].update(execution_completed=False),
    lambda r: r["execution"].update(physical_execution=False),
    lambda r: r["execution"].update(robot_errors=["controller failed"]),
    lambda r: r["execution"]["actual_robot_actions"].pop(),
    lambda r: r["execution"]["actual_robot_actions"][0].update(success=False),
    lambda r: r["execution"]["actual_robot_actions"][2]["args"].update(z=20.5),
    lambda r: r["planning_diagnostics"]["grasp_height_resolution"].pop("pixel_xy"),
    lambda r: r.update(grasp_execution_experience={"observed_result": {"acquisition": "SUCCESS"}}),
    lambda r: r.update(grasp_execution_experience={"observed_result": {"contact_alignment": "MISALIGNED"}}),
    lambda r: r.update(supervisor_after={"trajectory_decision": "STOP"}),
])
def test_unsafe_unresolved_nonphysical_or_secondary_attempts_do_not_schedule(mutate):
    row = record()
    mutate(row)
    assert retry_eligibility(row)["status"] == "NOT_SCHEDULED"
    with pytest.raises(ValueError):
        height_options(row, step_mm=1)


def test_disabled_and_parser_options():
    assert retry_eligibility(record(), enabled=False)["status"] == "NOT_SCHEDULED"
    args = build_parser().parse_args([])
    assert args.grasp_height_retry_step_mm == 1
    assert args.no_grasp_height_retry is False
    args = build_parser().parse_args(["--no-grasp-height-retry", "--grasp-height-retry-step-mm", "0.5"])
    assert args.grasp_height_retry_step_mm == .5 and args.no_grasp_height_retry


@pytest.mark.parametrize("step", [0, -1, 3.1, float("nan"), float("inf"), True])
def test_invalid_step_rejected_before_execution(step):
    with pytest.raises(ValueError, match="step"):
        height_options(record(), step_mm=step)


def test_only_contact_z_changes_and_relative_surface_is_used():
    row = record()
    # A stale proposal is not the source of the command.
    row["execution_proposal"]["actions"][2]["args"]["z"] = 18
    choices = height_options(row, step_mm=1)
    assert choices["options"]["DEEPER"]["descent_below_surface_mm"] == 4
    assert choices["options"]["DEEPER"]["commanded_z_mm"] == pytest.approx(19)
    proposal, metadata = compile_height_retry(row, choices)
    assert proposal.actions[2]["args"]["z"] == pytest.approx(19)
    assert metadata["parent_iteration"] == 1
    assert metadata["causal_claim"] == "UNTESTED_HYPOTHESIS"
    validate_locked_retry(proposal.actions, choices, metadata)
    for i, action in enumerate(proposal.actions):
        if i != 2:
            assert action == choices["locked_actions"][i]
    altered = deepcopy(list(proposal.actions))
    altered[4]["args"]["z"] += 1
    with pytest.raises(ValueError, match="another trajectory"):
        validate_locked_retry(altered, choices, metadata)


@pytest.mark.parametrize("mutate,blocked", [
    (lambda r: r.update(maximum_compression_mm=3), "DEEPER"),
    (lambda r: r.update(lower_z_mm=20), "DEEPER"),
    (lambda r: r.update(minimum_compression_mm=3), "SHALLOWER"),
])
def test_existing_compression_and_floor_limits_are_not_expanded(mutate, blocked):
    row = record()
    mutate(row["planning_diagnostics"]["grasp_height_resolution"]["resolution"])
    choices = height_options(row, step_mm=1)
    assert blocked not in choices["options"]
    assert blocked in choices["blocked_options"]
    if blocked == "DEEPER":
        with pytest.raises(ValueError, match="unavailable"):
            compile_height_retry(row, choices)


def test_lift_floor_and_explicit_depth_evidence():
    row = record()
    row["execution"]["actual_robot_actions"][4]["args"]["z"] = 50
    choices = height_options(row, step_mm=1)
    assert "SHALLOWER" in choices["blocked_options"]
    row["grasp_execution_experience"] = {"observed_result": {"depth_interpretation": "TOO_DEEP"}}
    assert not height_options(row, step_mm=1)["options"]


def test_explicit_too_deep_uses_shallower_command():
    row = record()
    row["grasp_execution_experience"] = {"observed_result": {"depth_interpretation": "TOO_DEEP"}}
    choices = height_options(row, step_mm=1)
    proposal, metadata = compile_height_retry(row, choices)
    assert metadata["choice"] == "SHALLOWER"
    assert proposal.actions[2]["args"]["z"] == 21
    validate_locked_retry(proposal.actions, choices, metadata)


@pytest.mark.parametrize("status", ["CHANGED", "UNCOMPARABLE"])
def test_changed_scene_does_not_replay_old_target(status):
    row = record()
    row["evaluation"]["perception_comparison"]["status"] = status
    assert retry_eligibility(row)["status"] == "NOT_SCHEDULED"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, None])
def test_invalid_geometry_cannot_generate_a_command(value):
    row = record()
    row["planning_diagnostics"]["grasp_height_resolution"]["resolution"]["lower_z_mm"] = value
    with pytest.raises(ValueError):
        height_options(row, step_mm=1)


def test_model_selects_nondefault_depth_and_preserves_other_commands():
    from cloth_agent.grasp_height_retry import compile_model_height_retry
    row = record()
    proposal, choices, metadata = compile_model_height_retry(row,
        {"contact_z_mm": 18.5, "reason": "Test 1.5 mm deeper"})
    assert metadata["delta_commanded_z_mm"] == -1.5
    assert metadata["decision_authority"] == "Claude"
    validate_locked_retry(proposal.actions, choices, metadata)


@pytest.mark.parametrize('z', [17., 20., float('nan'), True])
def test_model_invalid_depth_is_rejected_without_clamping(z):
    from cloth_agent.grasp_height_retry import compile_model_height_retry
    with pytest.raises(ValueError):
        compile_model_height_retry(record(), {"contact_z_mm": z, "reason": "test"})


def test_model_can_choose_shallower_without_host_direction_override():
    from cloth_agent.grasp_height_retry import compile_model_height_retry
    _, _, metadata = compile_model_height_retry(record(),
        {"contact_z_mm": 21.5, "reason": "Test shallower"})
    assert metadata["choice"] == "SHALLOWER"


def test_retry_images_exclude_derived_images_and_deduplicate_content(tmp_path):
    from cloth_agent.grasp_height_retry import select_height_retry_images
    def file(group, name, data):
        p = tmp_path / group / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return str(p)
    before = file('before', 'camera_A_rgb_upright.png', b'rgb')
    after = file('after', 'camera_A_rgb_upright.png', b'rgb')
    derived = file('before', 'camera_A_height_map_heatmap.png', b'heatmap')
    raw = file('before', 'camera_0_A.png', b'raw')
    holds = [file('hold', f'camera_A_grasp_{phase}.png', phase.encode())
             for phase in ('before_lift', 'after_close', 'after_lift')]
    images, catalog = select_height_retry_images({'before_images': [derived, raw, before],
        'after_images': [after], 'observer_images_hold_check': holds})
    assert len(images) == 4
    assert len(catalog) == 5
    assert catalog[0]['image_index'] == catalog[1]['image_index'] == 0
    assert images[0].name == 'camera_A_rgb_upright.png'


def test_retry_images_fallback_and_missing_evidence(tmp_path):
    from cloth_agent.grasp_height_retry import select_height_retry_images
    raw = tmp_path / 'camera_0_A.png'
    raw.write_bytes(b'rgb')
    images, catalog = select_height_retry_images({'before_images': [str(raw)],
        'after_images': None, 'observer_images_hold_check': []})
    assert images == [raw]
    assert [c['role'] for c in catalog] == ['before_perception_rgb']
