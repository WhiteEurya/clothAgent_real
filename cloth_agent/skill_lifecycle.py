"""Versioned skill proposals, deterministic review, and persistence.

Skills are prompt guidance only.  This module deliberately never emits robot
commands and never activates an unreviewed proposal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from difflib import SequenceMatcher
from typing import Any, Iterable

from .skills import SKILL_REGISTRY, SkillSpec, skill_prompt


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", value.lower()))


def _similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    jaccard = len(a & b) / max(1, len(a | b))
    sequence = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return max(jaccard, sequence)


@dataclass(frozen=True)
class SkillProposal:
    operation: str
    name: str
    purpose: str
    guidance: str
    rationale: str
    evidence: tuple[str, ...]
    base_skill: str | None = None
    confidence: float = 0.0
    requested_name: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "operation": self.operation,
            "name": self.name,
            "purpose": self.purpose,
            "guidance": self.guidance,
            "rationale": self.rationale,
            "evidence": list(self.evidence),
            "base_skill": self.base_skill,
            "confidence": self.confidence,
        }
        if self.requested_name is not None and self.requested_name != self.name:
            payload["requested_name"] = self.requested_name
        return payload


@dataclass(frozen=True)
class SkillReview:
    status: str
    approved: bool
    reason: str
    similar_skills: tuple[dict[str, Any], ...] = ()
    activated_skill: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "approved": self.approved,
            "reason": self.reason,
            "similar_skills": [dict(item) for item in self.similar_skills],
            "activated_skill": self.activated_skill,
        }


class SkillStore:
    """Persistent approved skill library with proposal/review audit logs."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.approved_path = self.root / "approved.json"
        self.proposals_path = self.root / "proposals.jsonl"
        self.reviews_path = self.root / "reviews.jsonl"

    def _write_json(self, path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _append(self, path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def approved(self) -> tuple[SkillSpec, ...]:
        merged = dict(SKILL_REGISTRY)
        if self.approved_path.is_file():
            raw = json.loads(self.approved_path.read_text(encoding="utf-8"))
            for item in raw.get("skills", []):
                skill = SkillSpec(
                    name=str(item["name"]),
                    purpose=str(item["purpose"]),
                    guidance=str(item["guidance"]),
                    version=int(item.get("version", 1)),
                    source=str(item.get("source", "reviewed")),
                )
                merged[skill.name] = skill
        return tuple(merged[name] for name in sorted(merged))

    def prompt(self) -> str:
        dynamic = tuple(skill for skill in self.approved() if skill.name not in SKILL_REGISTRY)
        overrides = tuple(skill for skill in self.approved() if skill.name in SKILL_REGISTRY and skill.source != "builtin")
        return skill_prompt(dynamic + overrides)

    @staticmethod
    def parse_update(value: Any) -> SkillProposal | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("skill_update must be an object")
        required = {"operation", "name", "purpose", "guidance", "rationale", "evidence", "confidence"}
        missing = required.difference(value)
        unknown = set(value).difference(required | {"base_skill"})
        if missing or unknown:
            raise ValueError(f"skill_update fields invalid; missing={sorted(missing)} unknown={sorted(unknown)}")
        operation = str(value["operation"]).strip().lower()
        if operation not in {"create", "modify"}:
            raise ValueError("skill_update.operation must be create or modify")
        requested_name = str(value["name"]).strip()
        # Claude occasionally emits a Python-style identifier or title-cased
        # label even though the skill contract uses lowercase kebab-case.
        # Normalize only harmless separators/case; punctuation and malformed
        # names still fail closed and are sent through the normal audit path.
        name = re.sub(r"[\s_]+", "-", requested_name.lower())
        name = re.sub(r"-+", "-", name).strip("-")
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+){0,5}", name):
            raise ValueError("skill_update.name must be lowercase hyphenated text")
        evidence = value["evidence"]
        if not isinstance(evidence, list) or not evidence or len(evidence) > 8:
            raise ValueError("skill_update.evidence must contain 1-8 entries")
        if any(not isinstance(item, str) or not item.strip() for item in evidence):
            raise ValueError("skill_update.evidence entries must be non-empty strings")
        for field in ("purpose", "guidance", "rationale"):
            if not isinstance(value[field], str) or not value[field].strip():
                raise ValueError(f"skill_update.{field} must be a non-empty string")
        confidence = float(value["confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("skill_update.confidence must be between 0 and 1")
        return SkillProposal(
            operation=operation,
            name=name,
            purpose=str(value["purpose"]).strip(),
            guidance=str(value["guidance"]).strip(),
            rationale=str(value["rationale"]).strip(),
            evidence=tuple(str(item).strip() for item in evidence),
            base_skill=(
                re.sub(
                    r"-+",
                    "-",
                    re.sub(r"[\s_]+", "-", str(value["base_skill"]).strip().lower()),
                ).strip("-")
                if value.get("base_skill")
                else None
            ),
            confidence=confidence,
            requested_name=(requested_name if requested_name != name else None),
        )

    def review_and_apply(self, proposal: SkillProposal | None) -> SkillReview | None:
        if proposal is None:
            return None
        self._append(self.proposals_path, {"created_at": _now(), "proposal": proposal.as_dict()})
        existing = self.approved()
        by_name = {skill.name: skill for skill in existing}
        errors: list[str] = []
        if not proposal.purpose or not proposal.guidance or not proposal.rationale:
            errors.append("purpose, guidance, and rationale must be non-empty")
        guidance_compact = re.sub(r"[\s_-]+", "", proposal.guidance.lower())
        forbidden = (
            "setposition",
            "setservoangle",
            "xarm",
            "sdkcall",
            "subprocess",
            "pythoncode",
            "jointangle",
            "coordinate",
        )
        if any(term in guidance_compact for term in forbidden):
            errors.append("skill guidance contains low-level robot/API instructions")
        safety_words = _tokens(proposal.guidance)
        if not ({"workspace", "ik", "release"} & safety_words):
            errors.append("skill guidance must include at least one safety constraint")
        if proposal.operation == "modify" and proposal.base_skill not in by_name:
            errors.append(f"base_skill {proposal.base_skill!r} is not in the approved library")
        if (
            proposal.operation == "modify"
            and proposal.base_skill in by_name
            and proposal.name != proposal.base_skill
        ):
            errors.append("modify proposals must keep the base skill name")
        if proposal.operation == "create" and proposal.name in by_name:
            errors.append(f"skill {proposal.name!r} already exists; use modify")
        similar: list[dict[str, Any]] = []
        candidate_text = f"{proposal.name} {proposal.purpose} {proposal.guidance}"
        for skill in existing:
            if proposal.operation == "modify" and skill.name == proposal.base_skill:
                continue
            score = _similarity(candidate_text, f"{skill.name} {skill.purpose} {skill.guidance}")
            if score >= 0.68:
                similar.append({"name": skill.name, "version": skill.version, "similarity": round(score, 3)})
        if proposal.operation == "create" and similar:
            errors.append("candidate is too similar to an approved skill; modify that skill instead")
        if proposal.confidence < 0.55:
            errors.append("evidence confidence is below the activation threshold 0.55")
        if errors:
            review = SkillReview(
                status="REJECTED" if any("similar" in item or "already exists" in item for item in errors) else "NEEDS_REVIEW",
                approved=False,
                reason="; ".join(errors),
                similar_skills=tuple(similar),
            )
            self._append(self.reviews_path, {"created_at": _now(), "proposal": proposal.as_dict(), "review": review.as_dict()})
            return review

        previous = by_name.get(proposal.name) if proposal.operation == "modify" else None
        version = (previous.version + 1) if previous is not None else 1
        activated = SkillSpec(
            name=proposal.name,
            purpose=proposal.purpose,
            guidance=proposal.guidance,
            version=version,
            source="reviewed",
        )
        saved = [asdict(skill) for skill in existing if skill.name != activated.name]
        saved.append(asdict(activated))
        self._write_json(self.approved_path, {"updated_at": _now(), "skills": sorted(saved, key=lambda item: item["name"])})
        review = SkillReview(
            status="APPROVED",
            approved=True,
            reason="proposal passed schema, safety, evidence, and duplicate checks",
            similar_skills=tuple(similar),
            activated_skill=asdict(activated),
        )
        self._append(self.reviews_path, {"created_at": _now(), "proposal": proposal.as_dict(), "review": review.as_dict()})
        return review
