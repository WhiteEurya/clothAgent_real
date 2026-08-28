from __future__ import annotations

import json
from pathlib import Path

import pytest

from cloth_agent.skill_lifecycle import RunSkillLedger, SkillStore
from cloth_agent.skills import available_skill_names, skill_prompt


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


def test_flatten_garment_is_a_builtin_system_skill() -> None:
    assert "flatten-garment" in available_skill_names()
    assert "farthest safe X" in skill_prompt()


def test_shake_skills_are_builtin_and_available_to_auto_exploration(tmp_path: Path) -> None:
    names = available_skill_names()
    assert "collar-full-shake" in names
    assert "direct-shake" in names
    prompt = skill_prompt()
    assert "### Skill: collar-full-shake" in prompt
    assert "### Skill: direct-shake" in prompt

    store = SkillStore(tmp_path / "skills")
    assert {skill.name for skill in store.approved()} >= {
        "collar-full-shake",
        "direct-shake",
    }


def test_builtin_skill_modify_writes_external_patch_only(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    proposal = _proposal(
        operation="modify",
        name="flatten-garment",
        base_skill="flatten-garment",
        purpose="Refine the built-in flatten procedure from a completed trial.",
        guidance=(
            "Use the current visual evidence, preserve workspace and IK margins, "
            "and release in a controlled way."
        ),
        rationale="A completed trial suggests a possible procedural refinement.",
        evidence=["iteration 3 before/after evidence"],
    )

    review = store.review_and_apply(proposal)

    assert review is not None
    assert review.status == "PATCH_PROPOSED"
    assert review.approved is False
    assert review.patch_path is not None
    assert Path(review.patch_path).is_file()
    assert not (tmp_path / "skills" / "approved.json").is_file()
    assert "farthest safe X" in store.prompt()
    patch = json.loads(Path(review.patch_path).read_text(encoding="utf-8"))
    assert patch["apply_policy"] == "external_review_only"
    assert patch["applied"] is False


def test_externally_approved_builtin_patch_overlays_prompt(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    guidance = (
        "Use the current visual evidence, preserve workspace and IK margins, "
        "and release only after the separated layer reaches clean table."
    )
    review = store.review_and_apply(
        _proposal(
            operation="modify",
            name="flatten-garment",
            base_skill="flatten-garment",
            purpose="Refine the built-in flatten procedure from reviewed evidence.",
            guidance=guidance,
            rationale="Reviewed before/after evidence supports this refinement.",
            evidence=["iteration 3 before/after evidence"],
        )
    )
    assert review is not None and review.patch_path is not None

    activated = store.approve_patch(
        review.patch_path,
        reviewer="operator",
        note="Reviewed against the saved before/after evidence.",
    )

    assert activated.name == "flatten-garment"
    assert activated.version == 2
    assert activated.source.startswith("external-patch:")
    assert guidance in store.prompt()
    assert "farthest safe X" not in store.prompt()
    manifest = json.loads((tmp_path / "skills" / "approved_patches.json").read_text())
    assert manifest["patches"][0]["reviewer"] == "operator"
    assert len(manifest["patches"][0]["sha256"]) == 64


def test_approved_patch_fails_closed_after_tampering(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    review = store.review_and_apply(
        _proposal(
            operation="modify",
            name="flatten-garment",
            base_skill="flatten-garment",
            purpose="Refine the built-in flatten procedure from reviewed evidence.",
            guidance=(
                "Use current evidence, preserve workspace and IK margins, and "
                "release only after the layer reaches clean table."
            ),
            rationale="Reviewed before/after evidence supports this refinement.",
            evidence=["iteration 3 before/after evidence"],
        )
    )
    assert review is not None and review.patch_path is not None
    store.approve_patch(
        review.patch_path,
        reviewer="operator",
        note="Reviewed against the saved before/after evidence.",
    )
    patch_path = Path(review.patch_path)
    patch_path.write_text(patch_path.read_text() + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        store.prompt()


def test_run_skill_ledger_defers_global_activation_until_finalize(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    ledger = RunSkillLedger(tmp_path / "run_workspace")
    proposal = _proposal()

    review = ledger.stage_skill_update(proposal, iteration=3, source="evaluation")

    assert review is not None
    assert review.status == "RUN_LOCAL_PENDING"
    assert not (tmp_path / "skills" / "approved.json").exists()
    assert "edge-release" in ledger.prompt_appendix()

    synthesis = ledger.finalize(store)

    assert synthesis["candidate_count"] == 1
    assert synthesis["skill_group_count"] == 1
    assert synthesis["outcome_summary"] == {
        "IMPROVED": 0,
        "NEUTRAL": 0,
        "REGRESSED": 0,
    }
    assert synthesis["persistent_reviews"][0]["review"]["approved"] is True
    assert "edge-release" in store.prompt()
    assert (tmp_path / "run_workspace" / "run_skill_lifecycle" / "synthesis.json").is_file()


def test_run_skill_ledger_combines_same_skill_evidence_before_finalize(tmp_path: Path) -> None:
    store = SkillStore(tmp_path / "skills")
    ledger = RunSkillLedger(tmp_path / "run_workspace")
    first = _proposal(
        rationale="The first completed action supported the edge response.",
        evidence=["iteration 1 edge moved"],
    )
    second = _proposal(
        operation="modify",
        name="edge-release",
        base_skill="edge-release",
        purpose="Release a supported edge without collapsing the garment.",
        guidance=(
            "Use the validated edge only after a workspace check and IK check; "
            "release while descending and verify the cloth remains supported."
        ),
        rationale="A second action showed that a slower release prevents rebound.",
        evidence=["iteration 2 after-state is stable"],
        confidence=0.95,
    )

    ledger.stage_skill_update(first, iteration=1)
    ledger.stage_skill_update(second, iteration=2)
    synthesis = ledger.finalize(store)

    candidate = synthesis["synthesized"][0]["proposal"]
    assert synthesis["synthesized"][0]["source_count"] == 2
    assert "iteration 1 edge moved" in candidate["evidence"]
    assert "iteration 2 after-state is stable" in candidate["evidence"]
    assert synthesis["persistent_reviews"][0]["review"]["approved"] is True
