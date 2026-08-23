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
    version: int = 1
    source: str = "builtin"

    def prompt(self) -> str:
        return (
            f"### Skill: {self.name} (v{self.version})\n"
            f"Purpose: {self.purpose}\nGuidance:\n{self.guidance}"
        )


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


FLATTEN_SKILL = SkillSpec(
    name="flatten-garment",
    purpose=(
        "Progressively open and flatten a garment by validating the grasped "
        "structure, establishing a useful hanging configuration, transporting "
        "in a verified opening direction, and laying down only while the scene "
        "shows measurable improvement."
    ),
    guidance=(
        "Follow this staged workflow:\n"
        "1. Acquire a validated garment structure: choose a visible, liftable "
        "anchor whose local RGB-D measurement, height, and depth consistency "
        "support a real garment grasp; do not proceed from an unsupported or "
        "table-only point.\n"
        "2. Establish a hanging configuration if useful: lift to a validated "
        "safe height and confirm that the grasped fabric is supported and can "
        "hang without dragging or colliding. If the lift does not reveal a "
        "supported structure, stop and re-plan.\n"
        "3. Transport toward a verified opening direction: infer the direction "
        "from the current garment geometry and before/after evidence, then use "
        "validated intermediate waypoints with stable orientation and workspace "
        "margin. The farthest safe X direction is allowed only when the scene "
        "verifies that it opens the garment; do not assume it is always correct.\n"
        "4. Continue only while overlap decreases or coverage increases: after "
        "each meaningful transport segment, compare the garment footprint, "
        "visible area, layer overlap, and height relief. Stop, release, or "
        "re-plan when the move is not improving those measures; never continue "
        "a blind pull.\n"
        "5. Lay down progressively: retreat while descending through validated "
        "intermediate heights, release quasi-statically on a clean surface, and "
        "avoid dropping, flinging, twisting, or dragging the loaded gripper. "
        "The skill supplies this decision sequence, never fixed coordinates or "
        "robot API calls."
    ),
)


SKILL_REGISTRY: dict[str, SkillSpec] = {
    LAYDOWN_SKILL.name: LAYDOWN_SKILL,
    FLATTEN_SKILL.name: FLATTEN_SKILL,
}


def available_skill_names() -> tuple[str, ...]:
    return tuple(sorted(SKILL_REGISTRY))


def skill_prompt(extra_skills: tuple[SkillSpec, ...] | list[SkillSpec] | None = None) -> str:
    """Return built-in guidance plus approved dynamic skills for a Claude prompt."""

    merged = dict(SKILL_REGISTRY)
    for skill in extra_skills or ():
        merged[skill.name] = skill
    return "\n\n".join(merged[name].prompt() for name in sorted(merged))


def validate_skill_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized not in SKILL_REGISTRY:
        raise ValueError(
            f"unknown skill {name!r}; available skills: {', '.join(available_skill_names())}"
        )
    return normalized
