"""Small reusable procedural skills exposed to Claude's garment planner.

Skills in this module provide reasoning guidance only.  They never choose
coordinates or execute robot commands; Claude still emits the concrete
``move``/gripper actions and the normal runtime safety gates remain in charge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    name: str
    purpose: str
    guidance: str

    def prompt(self) -> str:
        return f"### Skill: {self.name}\nPurpose: {self.purpose}\nGuidance:\n{self.guidance}"


LAYDOWN_SKILL = SkillSpec(
    name="laydown",
    purpose=(
        "Convert a garment held from a useful lifting anchor into a more spread "
        "tabletop configuration for easier perception and manipulation."
    ),
    guidance=(
        "Use this skill only when you believe the current grasp is a useful anchor. "
        "This is a quasi-static maneuver, not a fling: let the garment hang under "
        "gravity, retreat the grasp point away from the hanging garment while "
        "gradually descending, allow the previously forward-facing hanging surface "
        "to become the upward-facing tabletop surface, then release in a controlled "
        "way. Choose retreat direction, retreat distance, descent profile, "
        "intermediate waypoints, yaw, and release height from the current geometry "
        "and workspace. Avoid dropping a bundled garment from high above the table, "
        "dragging the grasp across deposited cloth, unnecessary twisting, releasing "
        "while most of the garment remains vertically bundled, or leaving the "
        "validated workspace. The skill supplies procedure, never fixed coordinates."
    ),
)


SKILL_REGISTRY: dict[str, SkillSpec] = {LAYDOWN_SKILL.name: LAYDOWN_SKILL}


def available_skill_names() -> tuple[str, ...]:
    return tuple(sorted(SKILL_REGISTRY))


def skill_prompt() -> str:
    """Return all currently available skill guidance for a Claude prompt."""

    return "\n\n".join(SKILL_REGISTRY[name].prompt() for name in sorted(SKILL_REGISTRY))


def validate_skill_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in SKILL_REGISTRY:
        raise ValueError(
            f"unknown skill {name!r}; available skills: {', '.join(available_skill_names())}"
        )
    return normalized
