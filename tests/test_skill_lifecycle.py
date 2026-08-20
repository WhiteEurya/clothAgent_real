from __future__ import annotations

import json
from pathlib import Path

from cloth_agent.skill_lifecycle import SkillStore


def _proposal(**overrides):
    payload = {
        "operation": "create",
        "name": "edge-release",
        "purpose": "Release a supported edge without collapsing the garment.",
        "guidance": (
            "Use only after the edge response is visually supported; preserve the "
            "validated workspace margin, check IK feasibility, and release gradually."
        ),
        "rationale": "The same edge response was reproduced in two completed trials.",
        "evidence": ["before/after visible edge displacement in iterations 1 and 2"],
        "confidence": 0.9,
    }
    payload.update(overrides)
    return SkillStore.parse_update(payload)


def test_approved_create_is_persisted_and_prompted(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    proposal = _proposal()
    review = store.review_and_apply(proposal)

    assert review is not None
    assert review.approved is True
    assert review.status == "APPROVED"
    assert (tmp_path / "skills" / "approved.json").is_file()
    assert "edge-release" in store.prompt()
    assert (tmp_path / "skills" / "proposals.jsonl").read_text()
    assert (tmp_path / "skills" / "reviews.jsonl").read_text()


def test_modify_increments_version_and_replaces_skill(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    assert store.review_and_apply(_proposal())
    proposal = _proposal(
        operation="modify",
        base_skill="edge-release",
        guidance=(
            "Use the validated edge only after a workspace check and IK check; "
            "release while descending and verify the cloth remains supported."
        ),
        rationale="The second trial showed that a slower release prevents rebound.",
        evidence=["iteration 2 after-state shows stable tabletop contact"],
    )
    review = store.review_and_apply(proposal)

    assert review is not None and review.approved is True
    assert review.activated_skill["version"] == 2
    raw = json.loads((tmp_path / "skills" / "approved.json").read_text())
    matches = [item for item in raw["skills"] if item["name"] == "edge-release"]
    assert len(matches) == 1
    assert matches[0]["version"] == 2


def test_reviewer_rejects_duplicate_and_unsafe_proposals(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    assert store.review_and_apply(_proposal())

    duplicate = _proposal(
        name="edge-release-new",
        purpose="Release a supported edge without collapsing the garment.",
        guidance=(
            "Use only after the edge response is visually supported; preserve the "
            "validated workspace margin, check IK feasibility, and release gradually."
        ),
    )
    duplicate_review = store.review_and_apply(duplicate)
    assert duplicate_review is not None
    assert duplicate_review.approved is False
    assert duplicate_review.status == "REJECTED"

    unsafe = _proposal(
        name="unsafe-edge",
        guidance="Call set_position directly after checking the workspace.",
    )
    unsafe_review = store.review_and_apply(unsafe)
    assert unsafe_review is not None
    assert unsafe_review.approved is False
    assert "low-level" in unsafe_review.reason


def test_unreviewed_proposals_never_enter_prompt(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    proposal = _proposal(confidence=0.2)
    review = store.review_and_apply(proposal)

    assert review is not None and review.approved is False
    assert "edge-release" not in store.prompt()


def test_skill_name_separators_are_canonicalized_and_audited() -> None:
    proposal = _proposal(name="Grasp_Acquisition_Check")

    assert proposal is not None
    assert proposal.name == "grasp-acquisition-check"
    assert proposal.requested_name == "Grasp_Acquisition_Check"
    assert proposal.as_dict()["requested_name"] == "Grasp_Acquisition_Check"
