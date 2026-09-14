from __future__ import annotations

import json
from pathlib import Path

from cloth_agent.fold_exploration_viser import (
    _claude_input_groups,
    _planning_images_from_prompt,
    _run_root,
    _debug_markdown, _workspace_markdown, _iter_images,
)


def test_remote_failure_manifest_and_workspace_are_visible_without_trajectory(tmp_path):
    iteration = tmp_path / "iteration_001"
    diagnostics = iteration / "planning_attempt_fold_01" / "job"
    diagnostics.mkdir(parents=True)
    rgb = _touch_image(iteration / "before_raw" / "camera_A_rgb_upright.png")
    world = _touch_image(diagnostics / "workspace_base_xy.png")
    (diagnostics / "pixel_motion_invocation.json").write_text(json.dumps({
        "stage": "pixel_motion", "status": "FAILED", "evidence_images": [str(rgb)]}))
    (diagnostics / "workspace_diagnostics.json").write_text(json.dumps({
        "status": "REJECTED", "moves": [{"action_index": 6, "target": "pixel",
            "upright_pixel_xy": [9, 10], "base_xyz_mm": [550, 80, 57],
            "error": "outside left/right boundaries", "lateral": {"signed_clearance_mm": [80, -20]}}]}))
    groups = _claude_input_groups(iteration, tmp_path)
    assert groups["planning_attempt_fold_01/pixel_motion"] == [rgb]
    assert _iter_images(iteration)[0] == world
    summary = _workspace_markdown(iteration)
    assert "#6" in summary and "-20" in summary and "REJECTED" in summary


def test_timing_panel_handles_partial_json_and_nested_durations(tmp_path):
    (tmp_path / "debug_events.jsonl").write_text(json.dumps({
        "timestamp": "2026-09-14T00:00:00+00:00", "elapsed_s": 12.5,
        "stage": "remote-planner", "message": "remote_claude: measured",
        "fields": {"duration_s": 8.2}}) + '\n{"unfinished":')
    panel = _debug_markdown(tmp_path)
    assert "8.200s" in panel and "12.5s" in panel
    assert "do not sum" in panel


def _touch_image(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    return path.resolve()


def test_planning_images_are_limited_to_declared_garment_section(
    tmp_path: Path,
) -> None:
    rgb = _touch_image(tmp_path / "camera_A_rgb_upright.png")
    overlay = _touch_image(tmp_path / "camera_A_rxxx_overlay_upright.png")
    unrelated = _touch_image(tmp_path / "molmo_debug.png")
    prompt = (
        "Molmo debug path mentioned earlier: "
        f"{unrelated}\n\n"
        "Garment images to inspect:\n"
        f"- RGB: {rgb}\n"
        f"- OVERLAY: {overlay}\n\n"
        "When the canonical upright Camera-A RGB is supplied, use it."
    )

    assert _planning_images_from_prompt(prompt) == [rgb, overlay]


def test_claude_input_groups_follow_actual_stage_manifests(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    source = run_root / "results" / "fold_exploration" / "stamp"
    iteration = source / "iteration_001"
    iteration.mkdir(parents=True)
    (source / "summary.json").write_text(
        json.dumps({"run_dir": str(run_root)}),
        encoding="utf-8",
    )

    planning_rgb = _touch_image(iteration / "before_raw" / "camera_A_rgb_upright.png")
    planning_overlay = _touch_image(
        iteration / "before_raw" / "camera_A_rxxx_overlay_upright.png"
    )
    planning_prompt = (
        "Garment images to inspect:\n"
        f"- RGB: {planning_rgb}\n"
        f"- OVERLAY: {planning_overlay}\n\n"
        "When the canonical upright Camera-A RGB is supplied, use it."
    )
    (iteration / "planning_diagnostics.json").write_text(
        json.dumps({"visual_plan_result": {"prompt": planning_prompt}}),
        encoding="utf-8",
    )

    supervisor_a = _touch_image(run_root / "results" / "perception" / "camera_0_A.png")
    supervisor_b = _touch_image(run_root / "results" / "perception" / "camera_1_B.png")
    context = run_root / "results" / "fold_supervisor" / "context_test"
    context.mkdir(parents=True)
    evidence = context / "04_evidence_manifest.json"
    evidence.write_text(
        json.dumps(
            {
                "images": [
                    str(supervisor_a.relative_to(run_root)),
                    str(supervisor_b.relative_to(run_root)),
                ],
                "rollout_video_contact_sheets": [],
            }
        ),
        encoding="utf-8",
    )
    (iteration / "supervisor_before.json").write_text(
        json.dumps(
            {
                "context_bundle": {
                    "read_order": [str(evidence.relative_to(run_root))]
                }
            }
        ),
        encoding="utf-8",
    )

    evaluation_image = _touch_image(
        iteration / "rollout_recording" / "contact_sheet.png"
    )
    (iteration / "claude_evaluation_result.json").write_text(
        json.dumps({"prompt": f"Evidence:\n- {evaluation_image}"}),
        encoding="utf-8",
    )

    assert _run_root(source.resolve()) == run_root.resolve()
    groups = _claude_input_groups(iteration.resolve(), run_root.resolve())

    assert groups["planning_stage1"] == [planning_rgb, planning_overlay]
    assert groups["supervisor_before"] == [supervisor_a, supervisor_b]
    assert groups["evaluation"] == [evaluation_image]
