from __future__ import annotations

import json
from pathlib import Path

from scripts.claude_condition_ablation import (
    CONDITIONS,
    QUESTIONS,
    _condition_assets,
    _condition_description,
    _candidate_from_payload,
    _json_from_cli,
    _sample_segments,
    _schema,
    _validate_response,
)


def test_candidate_payload_is_normalized() -> None:
    candidate = _candidate_from_payload(
        {"selected_grasp": {"camera": "A", "pixel_xy": [620.4, 544.6], "reason": "raised fold"}}
    )
    assert candidate == {"camera": "A", "pixel_xy": [620, 545], "reason": "raised fold"}
    assert _candidate_from_payload({"selected_grasp": {"camera": "C", "pixel_xy": [1, 2]}}) is None


def test_condition_assets_follow_information_ladder(tmp_path: Path) -> None:
    extracted = {}
    for name in (
        "before_A_rgb",
        "after_A_rgb",
        "before_B_rgb",
        "after_B_rgb",
        "during_A_rgb",
        "during_B_rgb",
        "A_height",
        "B_height",
    ):
        path = tmp_path / f"{name}.png"
        path.write_bytes(b"placeholder")
        extracted[name] = str(path)
    bundle = {"extracted": extracted}
    assert [p.name for p in _condition_assets(bundle, "A")] == ["before_A_rgb.png", "after_A_rgb.png"]
    assert len(_condition_assets(bundle, "B")) == 4
    assert len(_condition_assets(bundle, "C")) == 6
    assert len(_condition_assets(bundle, "D")) == 8
    assert "action metadata" in _condition_description("D")


def test_response_validation_requires_all_ten_questions() -> None:
    response = {
        "segment_id": "segment_01",
        "condition": "A",
        "answers": {
            q["id"]: {"label": "UNKNOWN", "confidence": 0.5, "evidence": "not observable"}
            for q in QUESTIONS
        },
        "overall_notes": "Static evidence is insufficient for motion questions.",
    }
    validated = _validate_response(response, segment_id="segment_01", condition="A")
    assert set(validated["answers"]) == {q["id"] for q in QUESTIONS}
    assert validated["answers"]["Q1"]["label"] == "UNKNOWN"


def test_response_validation_rejects_missing_question() -> None:
    response = {
        "segment_id": "segment_01",
        "condition": "A",
        "answers": {},
        "overall_notes": "missing",
    }
    try:
        _validate_response(response, segment_id="segment_01", condition="A")
    except Exception as exc:
        assert "Q1" in str(exc)
    else:
        raise AssertionError("missing question should fail validation")


def test_schema_has_fixed_conditions_and_questions() -> None:
    schema = _schema()
    assert schema["properties"]["condition"]["enum"] == list(CONDITIONS)
    assert schema["properties"]["answers"]["required"] == [q["id"] for q in QUESTIONS]


def test_cli_json_envelope_is_unwrapped() -> None:
    payload = {"segment_id": "s", "condition": "A"}
    wrapped = json.dumps({"structured_output": payload})
    assert _json_from_cli(wrapped) == payload
    assert _json_from_cli("prefix\n" + json.dumps(payload) + "\nsuffix") == payload
