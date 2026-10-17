"""Static task semantics, read as data without importing the robot policy.

These definitions are goals, never observation classes or inferred run outcomes.
Only the small, reviewed sleeve-task mapping is supported for now.
"""
from __future__ import annotations

import ast
from pathlib import Path

from .discovery_data import digest, file_hash


def task_contract(subgoal, repository=None):
    root = Path(repository) if repository else Path(__file__).resolve().parents[2]
    unavailable = dict(status='UNAVAILABLE', goal=None, sha256=None, sources=[],
                       reason='No reviewed static task definition for this subgoal')
    if subgoal not in ('left_sleeve', 'right_sleeve'):
        return unavailable
    sources = []

    def source(relative, symbol, node, text):
        sources.append(dict(file=relative, symbol=symbol, line=node.lineno,
                            file_sha256=file_hash(root / relative), excerpt=text))

    try:
        frame_path = 'cloth_agent/fold_frame.py'
        frame_tree = ast.parse((root / frame_path).read_text())
        frame_node = next(n for n in frame_tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'FRAME_RULE' for t in n.targets))
        frame = ast.literal_eval(frame_node.value)
        # Only static identity convention; omit runtime hints, pixels and references.
        convention = frame.split('In displayed pixels', 1)[0].strip()
        if 'without mirroring; never wearer anatomy.' not in convention:
            raise ValueError('unrecognized garment-frame definition')
        source(frame_path, 'FRAME_RULE', frame_node, convention)

        policy_path = 'cloth_agent/fold_exploration_pipeline.py'
        tree = ast.parse((root / policy_path).read_text())
        steps_node = next(n for n in tree.body if isinstance(n, ast.AnnAssign)
                          and isinstance(n.target, ast.Name) and n.target.id == 'FOLD_STEPS')
        step = next(s for s in ast.literal_eval(steps_node.value) if s['id'] == subgoal)
        source(policy_path, 'FOLD_STEPS', steps_node, step)
        rubrics = [n for n in ast.walk(tree) if isinstance(n, ast.Constant)
                   and isinstance(n.value, str) and n.value.startswith('Sleeve completion rubric:')]
        if len(rubrics) != 1:
            raise ValueError('missing or ambiguous static completion rubric')
        source(policy_path, 'Sleeve completion rubric', rubrics[0], rubrics[0].value)
        goal = dict(subgoal=subgoal, objective=step['label'],
                    side_convention=convention, completion_criterion=rubrics[0].value,
                    interpretation='This is the desired task condition, not a claim that any sample satisfies it. '
                    'Determine garment orientation from each before RGB; if identity is occluded or ambiguous, '
                    'report uncertainty. No grasp point, trajectory, or state taxonomy is prescribed.')
        return dict(status='AVAILABLE', goal=goal, sha256=digest(goal), sources=sources,
                    provenance_scope='Static task contract from the repository at extraction time; '
                    'not a reconstruction of every historical prompt. No action/evaluation records used.')
    except (OSError, ValueError, SyntaxError, StopIteration, KeyError, TypeError) as exc:
        return dict(unavailable, reason=f'Static task definition unavailable: {exc}')


def model_task_goal(manifest):
    contract = manifest.get('task_contract', {})
    goal = contract.get('goal')
    if contract.get('status') != 'AVAILABLE' or not goal:
        raise ValueError('Task goal unavailable; create a new dry-run with a supported static subgoal definition')
    if goal.get('subgoal') != manifest['selected_subgoal'] or digest(goal) != contract.get('sha256'):
        raise ValueError('Task goal hash/subgoal mismatch')
    return goal
