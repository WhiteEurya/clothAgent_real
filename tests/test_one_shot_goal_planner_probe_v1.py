import json
from pathlib import Path
import pytest
from PIL import Image

from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as p


def plan(decision="ACTION"):
    return dict(decision=decision, grasp_px=[20, 30] if decision == "ACTION" else None,
                target_px=[70, 40] if decision == "ACTION" else None,
                motion_type="lift_pull_place" if decision == "ACTION" else None,
                short_reason="拉开可见折边。")


def test_payload_has_only_rgb_pair_and_no_tools_history_or_original(tmp_path):
    image = tmp_path/"rgb.png"
    Image.new("RGB", (100, 80), "purple").save(image)
    payload = p.payload(image, image, [100, 80])
    assert payload["model"] == "gpt-6-astra"
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["tools"] == [] and payload["tool_choice"] == "none"
    assert payload["text"]["format"]["strict"]
    assert len(payload["input"]) == 1
    assert len([c for c in payload["input"][0]["content"] if c["type"] == "input_image"]) == 2
    assert "previous_response_id" not in payload
    assert "original_plan" not in json.dumps(payload)
    assert p.GRIPPER_BOUNDARY_PROMPT in payload["instructions"]


def test_prompt_uses_existing_user_base_frame_boundary_without_new_limits():
    root = Path(p.__file__).resolve().parents[2]
    measured = json.loads((root/"data/robot/xarm_boundaries_new.json").read_text())["boundary_mm"]
    configured = json.loads((root/"config/dual_arm.local.json").read_text())["arms"]["left"]["workspace"]
    ys = sorted(point[1] for point in measured["lateral_points_mm"])
    assert ys == [configured["min_mm"][1], configured["max_mm"][1]]
    assert measured["z_min"] == configured["min_mm"][2]
    for value in [*ys, measured["z_min"]]:
        assert str(value) in p.GRIPPER_BOUNDARY_PROMPT
    assert configured["min_mm"][0] is configured["max_mm"][0] is None
    assert configured["max_mm"][2] is None
    assert "基座坐标" in p.GRIPPER_BOUNDARY_PROMPT
    assert "X 未设限" in p.GRIPPER_BOUNDARY_PROMPT


def test_json_semantics_and_coordinate_bounds():
    for decision in ("ACTION", "DONE", "ABSTAIN"):
        p.validate_plan(plan(decision), [100, 80])
    for invalid in (dict(plan(), grasp_px=None), dict(plan(), target_px=[100, 20]),
                    dict(plan(), short_reason="第一句。第二句。"), dict(plan("DONE"), motion_type="lift_pull_place")):
        with pytest.raises((ValueError, p.jsonschema.ValidationError)):
            p.validate_plan(invalid, [100, 80])


def test_one_response_no_tool_output():
    response = dict(id="response_1", status="completed", output=[
        dict(type="message", content=[dict(type="output_text", text=json.dumps(plan()))])])
    events = [dict(type="response.created", response=dict(id="response_1")),
              dict(type="response.completed", response=response)]
    assert json.loads(p.parse_events(events)[1]) == plan()
    with pytest.raises(ValueError):
        p.parse_events(events+[dict(type="response.created", response=dict(id="response_2"))])
    with pytest.raises(ValueError):
        p.parse_events(events+[dict(type="response.output_item.added", item=dict(type="function_call"))])


def test_transport_failure_is_not_retried_and_cannot_repeat_query(tmp_path, monkeypatch):
    (tmp_path/"request_payload.json").write_text("{}")
    meta = dict(query_id="test", image_size_px=[100, 80], request_payload_sha256=p.sha(tmp_path/"request_payload.json"))
    calls = []
    class Response:
        status_code = 503
        headers = {}
        text = "offline synthetic failure"
        def __enter__(self): return self
        def __exit__(self, *args): pass
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def mount(self, prefix, adapter):
            assert adapter.max_retries.total == 0
        def post(self, *args, **kwargs):
            calls.append(kwargs)
            assert kwargs["allow_redirects"] is False
            return Response()
    monkeypatch.setattr(p.requests, "Session", Session)
    result = p.call_once(tmp_path, meta, {})
    assert len(calls) == 1 and result["status"] == "FAILED"
    assert result["model_response_count"] == 0
    with pytest.raises(FileExistsError):
        p.call_once(tmp_path, meta, {})
    assert len(calls) == 1
