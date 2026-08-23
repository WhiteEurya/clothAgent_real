"""Versioned skill proposals, deterministic review, and persistence.

Skills are prompt guidance only.  This module deliberately never emits robot
commands and never activates an unreviewed proposal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
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
    patch_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "approved": self.approved,
            "reason": self.reason,
            "similar_skills": [dict(item) for item in self.similar_skills],
            "activated_skill": self.activated_skill,
            "patch_path": self.patch_path,
        }


class SkillStore:
    """Persistent approved skill library with proposal/review audit logs."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.approved_path = self.root / "approved.json"
        self.approved_patches_path = self.root / "approved_patches.json"
        self.patch_approvals_path = self.root / "patch_approvals.jsonl"
        self.proposals_path = self.root / "proposals.jsonl"
        self.reviews_path = self.root / "reviews.jsonl"
        self.patches_dir = self.root / "patches"

    def _write_json(self, path: Path, payload: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _append(self, path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _patch_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.patches_dir / path
        resolved = path.expanduser().resolve()
        patches_dir = self.patches_dir.resolve()
        if resolved.parent != patches_dir:
            raise ValueError(f"approved skill patch must be directly inside {patches_dir}")
        if not resolved.is_file():
            raise ValueError(f"approved skill patch does not exist: {resolved}")
        return resolved

    def _validated_patch(
        self,
        path: Path,
        current: SkillSpec,
        *,
        expected_sha256: str | None = None,
    ) -> SkillProposal:
        if expected_sha256 is not None and self._sha256(path) != expected_sha256:
            raise ValueError(f"approved skill patch failed SHA-256 validation: {path.name}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError(f"unsupported skill patch schema: {path.name}")
        if raw.get("patch_type") != "system-skill-guidance-patch":
            raise ValueError(f"invalid skill patch type: {path.name}")
        if raw.get("apply_policy") != "external_review_only":
            raise ValueError(f"skill patch is not external-review-only: {path.name}")
        proposal = self.parse_update(raw.get("proposal"))
        if proposal is None:
            raise ValueError(f"skill patch has no proposal: {path.name}")
        base_skill = str(raw.get("base_skill", ""))
        if base_skill not in SKILL_REGISTRY:
            raise ValueError(f"skill patch does not target a system skill: {path.name}")
        if (
            proposal.operation != "modify"
            or proposal.name != base_skill
            or proposal.base_skill != base_skill
        ):
            raise ValueError(f"skill patch proposal does not match its base skill: {path.name}")
        if current.name != base_skill:
            raise ValueError(f"skill patch approval order is invalid for {path.name}")
        if int(raw.get("base_version", -1)) != current.version:
            raise ValueError(
                f"skill patch base version mismatch for {path.name}: "
                f"expected {current.version}, got {raw.get('base_version')}"
            )
        return proposal

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
                # Built-in/system skills are immutable at runtime.  A reviewed
                # override is kept only as an external patch proposal and is
                # never allowed to shadow the authoritative registry here.
                if skill.name in SKILL_REGISTRY and skill.source != "builtin":
                    continue
                merged[skill.name] = skill
        if self.approved_patches_path.is_file():
            manifest = json.loads(self.approved_patches_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1:
                raise ValueError("unsupported approved skill patch manifest schema")
            entries = manifest.get("patches", [])
            if not isinstance(entries, list):
                raise ValueError("approved skill patch manifest patches must be a list")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("approved skill patch manifest entry must be an object")
                patch_file = str(entry.get("patch_file", ""))
                if not patch_file or Path(patch_file).name != patch_file:
                    raise ValueError("approved skill patch manifest contains an invalid patch_file")
                path = self._patch_path(patch_file)
                base_skill = str(entry.get("base_skill", ""))
                current = merged.get(base_skill)
                if current is None or base_skill not in SKILL_REGISTRY:
                    raise ValueError(f"approved patch targets unknown system skill: {base_skill!r}")
                proposal = self._validated_patch(
                    path,
                    current,
                    expected_sha256=str(entry.get("sha256", "")),
                )
                merged[base_skill] = SkillSpec(
                    name=base_skill,
                    purpose=proposal.purpose,
                    guidance=proposal.guidance,
                    version=current.version + 1,
                    source=f"external-patch:{patch_file}",
                )
        return tuple(merged[name] for name in sorted(merged))

    def approve_patch(
        self,
        patch_path: str | Path,
        *,
        reviewer: str,
        note: str,
    ) -> SkillSpec:
        """Approve a system-skill patch through an explicit external action."""

        reviewer = str(reviewer).strip()
        note = str(note).strip()
        if not reviewer:
            raise ValueError("patch reviewer must be non-empty")
        if not note:
            raise ValueError("patch approval note must be non-empty")
        path = self._patch_path(patch_path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        base_skill = str(raw.get("base_skill", ""))
        current = {skill.name: skill for skill in self.approved()}.get(base_skill)
        if current is None or base_skill not in SKILL_REGISTRY:
            raise ValueError(f"skill patch does not target an approved system skill: {base_skill!r}")
        self._validated_patch(path, current)
        digest = self._sha256(path)
        if self.approved_patches_path.is_file():
            manifest = json.loads(self.approved_patches_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != 1 or not isinstance(manifest.get("patches"), list):
                raise ValueError("approved skill patch manifest is invalid")
        else:
            manifest = {"schema_version": 1, "patches": []}
        if any(
            entry.get("patch_file") == path.name or entry.get("sha256") == digest
            for entry in manifest["patches"]
        ):
            raise ValueError(f"skill patch is already approved: {path.name}")
        approved_at = _now()
        entry = {
            "base_skill": base_skill,
            "patch_file": path.name,
            "sha256": digest,
            "approved_at": approved_at,
            "reviewer": reviewer,
            "note": note,
        }
        manifest["updated_at"] = approved_at
        manifest["patches"].append(entry)
        self._write_json(self.approved_patches_path, manifest)
        self._append(self.patch_approvals_path, entry)
        return {skill.name: skill for skill in self.approved()}[base_skill]

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

        if proposal.operation == "modify" and proposal.base_skill in SKILL_REGISTRY:
            self.patches_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            patch_path = self.patches_dir / f"{stamp}_{proposal.base_skill}.json"
            self._write_json(
                patch_path,
                {
                    "schema_version": 1,
                    "patch_type": "system-skill-guidance-patch",
                    "base_skill": proposal.base_skill,
                    "base_version": by_name[proposal.base_skill].version,
                    "created_at": _now(),
                    "proposal": proposal.as_dict(),
                    "apply_policy": "external_review_only",
                    "applied": False,
                },
            )
            review = SkillReview(
                status="PATCH_PROPOSED",
                approved=False,
                reason=(
                    "system/builtin skill is immutable; wrote an external patch "
                    "file for separate review and application"
                ),
                similar_skills=tuple(similar),
                patch_path=str(patch_path),
            )
            self._append(
                self.reviews_path,
                {
                    "created_at": _now(),
                    "proposal": proposal.as_dict(),
                    "review": review.as_dict(),
                },
            )
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


class RunSkillLedger:
    """Collect provisional skill knowledge without mutating the global library.

    A run may contain contradictory or low-confidence observations.  The ledger
    therefore keeps proposals and their provisional reviews under the run's
    workspace.  Only :meth:`finalize` may forward a synthesized proposal to the
    persistent :class:`SkillStore`.
    """

    def __init__(self, run_workspace: Path):
        self.root = Path(run_workspace).expanduser().resolve() / "run_skill_lifecycle"
        self.root.mkdir(parents=True, exist_ok=True)
        self.experience_path = self.root / "run_experience.jsonl"
        self.candidates_path = self.root / "skill_candidates.jsonl"
        self.reviews_path = self.root / "skill_reviews.jsonl"
        self.synthesis_path = self.root / "synthesis.json"

    @staticmethod
    def _append(path: Path, payload: Any) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def append_experience(self, experience: dict[str, Any]) -> None:
        self._append(self.experience_path, experience)

    def stage_skill_update(
        self,
        proposal: SkillProposal | None,
        *,
        iteration: int | None = None,
        source: str = "evaluation",
    ) -> SkillReview | None:
        """Store a candidate and return a run-local, non-activating review."""

        if proposal is None:
            return None
        entry = {
            "created_at": _now(),
            "iteration": iteration,
            "source": source,
            "proposal": proposal.as_dict(),
        }
        self._append(self.candidates_path, entry)
        review = SkillReview(
            status="RUN_LOCAL_PENDING",
            approved=False,
            reason=(
                "candidate retained in this run for evidence aggregation; "
                "global skill activation is deferred until run finalization"
            ),
        )
        self._append(
            self.reviews_path,
            {
                "created_at": _now(),
                "iteration": iteration,
                "source": source,
                "proposal": proposal.as_dict(),
                "review": review.as_dict(),
            },
        )
        return review

    def prompt_appendix(self, limit: int = 8) -> str:
        """Return provisional run lessons for the next planner prompt."""

        entries = self._read_jsonl(self.candidates_path)[-max(1, limit):]
        if not entries:
            return ""
        lines = [
            "### Run-local skill candidates (provisional; do not treat as globally approved)",
            "These candidates are evidence from the current run. Compare them against later outcomes; "
            "they will be synthesized only when the run ends.",
        ]
        for entry in entries:
            proposal = entry.get("proposal", {})
            if not isinstance(proposal, dict):
                continue
            lines.append(
                f"- iteration {entry.get('iteration', '?')}, "
                f"{proposal.get('operation', '?')} {proposal.get('name', '?')}: "
                f"{proposal.get('purpose', '')} Guidance: {proposal.get('guidance', '')}"
            )
        return "\n".join(lines)

    def finalize(self, persistent_store: SkillStore) -> dict[str, Any]:
        """Synthesize run candidates and only then update persistent skills."""

        entries = self._read_jsonl(self.candidates_path)
        experiences = self._read_jsonl(self.experience_path)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for entry in entries:
            proposal = entry.get("proposal", {})
            if isinstance(proposal, dict) and proposal.get("name"):
                grouped.setdefault(str(proposal["name"]), []).append(entry)

        persisted: list[dict[str, Any]] = []
        synthesized: list[dict[str, Any]] = []
        existing_names = {skill.name for skill in persistent_store.approved()}
        for name, group in sorted(grouped.items()):
            proposals = [item["proposal"] for item in group if isinstance(item.get("proposal"), dict)]
            if not proposals:
                continue
            # Prefer the highest-confidence proposal and merge independent
            # evidence/rationale from the same run into its final candidate.
            chosen = max(
                proposals,
                key=lambda value: float(value.get("confidence", 0.0)),
            )
            evidence: list[str] = []
            rationale_parts: list[str] = []
            for value in proposals:
                for item in value.get("evidence", []):
                    if isinstance(item, str) and item not in evidence and len(evidence) < 8:
                        evidence.append(item)
                rationale = value.get("rationale")
                if isinstance(rationale, str) and rationale not in rationale_parts:
                    rationale_parts.append(rationale)
            candidate = dict(chosen)
            candidate["evidence"] = evidence or list(chosen.get("evidence", []))[:8]
            candidate["rationale"] = (
                f"Run-level synthesis from {len(proposals)} provisional observation(s). "
                + " ".join(rationale_parts[:3])
            )
            if name not in existing_names:
                candidate["operation"] = "create"
                candidate.pop("base_skill", None)
            synthesized.append({"name": name, "source_count": len(proposals), "proposal": candidate})
            try:
                parsed = SkillStore.parse_update(candidate)
                review = persistent_store.review_and_apply(parsed)
                persisted.append(
                    {
                        "name": name,
                        "review": review.as_dict() if review is not None else None,
                    }
                )
            except Exception as exc:
                persisted.append(
                    {
                        "name": name,
                        "review": {
                            "status": "FINALIZE_ERROR",
                            "approved": False,
                            "reason": f"{type(exc).__name__}: {exc}",
                        },
                    }
                )

        summary = {
            "schema_version": 1,
            "completed_at": _now(),
            "experience_count": len(experiences),
            "candidate_count": len(entries),
            "skill_group_count": len(grouped),
            "outcome_summary": {
                status: sum(
                    1
                    for item in experiences
                    if isinstance(item.get("evaluation"), dict)
                    and (item["evaluation"].get("task_progress") or {}).get("status")
                    == status
                )
                for status in ("IMPROVED", "NEUTRAL", "REGRESSED")
            },
            "synthesized": synthesized,
            "persistent_reviews": persisted,
        }
        temporary = self.synthesis_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.synthesis_path)
        return summary
