"""Compiler-only snapshots of existing conditional experience and approved skills."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from ..skill_lifecycle import SkillStore
from ..skills import SKILL_REGISTRY
from .common import digest, read_json


def collect_knowledge(project_root):
    root = Path(project_root).resolve()
    sources, issues = [], []
    rules_path = root / "data/fold_experience/rules.json"
    if rules_path.is_file():
        data = read_json(rules_path)
        if data.get("schema_version") != 1 or not isinstance(data.get("rules"), dict):
            raise ValueError("Unsupported conditional experience library")
        sources.append({"kind": "conditional_experience", "path": str(rules_path),
                        "content_hash": digest(data), "content": data})
    else:
        issues.append({"code": "experience_library_missing", "path": str(rules_path)})

    # Resolve the same built-ins and approved external patches as the planner.
    # Reading approved.json alone can incorrectly activate stale built-in overrides.
    skills_root = root / "data/skills"
    if skills_root.is_dir():
        skills = [asdict(skill) for skill in SkillStore(skills_root).approved()]
    else:
        skills = [asdict(skill) for _, skill in sorted(SKILL_REGISTRY.items())]
        issues.append({"code": "skill_library_missing", "path": str(skills_root)})
    sources.append({"kind": "approved_skills", "path": str(skills_root),
                    "content_hash": digest(skills), "content": skills})
    return {"schema_version": 1, "scope": "compiler_only_unverified_prior_knowledge",
            "sources": sources, "issues": issues}
