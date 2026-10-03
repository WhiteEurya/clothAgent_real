"""Observation policy for the live planner; legacy isolated preparation helper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import uuid

from PIL import Image

from .pipeline_timing import timed_stage


def load_observation_skill(path):
    from .harness.information_probe import compact_skill
    artifact = json.loads(Path(path).read_text())
    compact_skill(artifact)  # Validate before real observation starts.
    return artifact


def inline_observation_policy(artifact):
    """Supply methods in the original visual call, without a binding call."""
    from .harness.information_probe import compact_skill
    return (
        'OBSERVATION POLICY EXPERIMENT (inline, same planning conversation):\n'
        'During your existing full-image inspection, identify only information gaps needed for '
        'the current task. Use the methods below to resolve those gaps with the existing cloth_image '
        'tools. Do not produce a separate observation plan or request a separate observer. '
        'If current images already answer the question, continue directly to normal selection. '
        'Otherwise call observe_information with information_need, method, image_id, box, scale, '
        'degrees_clockwise, success_check and on_insufficient copied/adapted from the relevant skill. '
        'Methods: local_boundary executes a source-pixel crop and optional display enlargement '
        'together (box required, scale>=1, degrees_clockwise=0); orientation rotates a selected '
        'source (box=null, scale=1); inspect_view returns a selected existing clean/reference view '
        '(box=null, scale=1, degrees_clockwise=0). To resolve overlay occlusion, choose the clean '
        'source explicitly. Choose only the method needed for the current gap; these are not a fixed '
        'sequence. Bind crop and display scale together when both are already known, rather than '
        'spending separate model turns dispatching those edits. Do not guess a tighter ROI before '
        'seeing evidence that justifies it. Low-level tools remain available for unsupported methods. '
        'Bind the exact source image and crop/rotation/resize parameters in the tool call; '
        'the Host executes it and returns the actual image in this conversation. Inspect those '
        'returned pixels, apply success_check and on_insufficient, and continue the original planning '
        'from that evidence. Reuse already inspected views and resolved findings; do not restart '
        'full-image interpretation merely because an edit completed. Reinspect only when evidence '
        'is insufficient or contradictory. Verify collar/hem direction after rotation rather than '
        'assuming a requested angle establishes orientation. Preserve source IDs and coordinate '
        'transforms, using map_point when needed; crops use tool pixel coordinates, not normalized ROI. '
        'A supported negative finding can satisfy an information need; missing visibility means '
        'UNKNOWN, not absence. Do not repeat edits that cannot resolve the missing information. '
        'If a required selection remains unsupported, follow the original insufficient-evidence '
        'contract without inventing a candidate. These skills are methods, not current scene facts. '
        'The original task, final response schema, image edit budget and safety constraints still apply.\n'
        + json.dumps(compact_skill(artifact), ensure_ascii=False, separators=(',', ':'))
    )


@timed_stage('observation_skill.prepare_views')
def prepare_skill_observations(*, images, context, instructions, artifact, root,
                               ssh_host, timeout_s, model=None):
    from .harness.model import RuntimeClaude
    from .harness.observation_ab import prepare_with_skill
    output = Path(root)/'observation_skill'/uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=False)
    (output/'skill_snapshot.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+'\n')
    report = {'status': 'RUNNING', 'downstream': 'original RemoteFoldClient._visual_plan and original grounding',
              'skill_sha256': hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest(),
              'max_total_visual_edits': 6, 'no_robot_execution': True}
    start = time.monotonic()
    try:
        catalog = []
        for index, path in enumerate(images):
            name = Path(path).name.lower()
            role = ('clean' if name == 'camera_a_rgb_upright.png' else
                    'overlay' if name == 'camera_a_rxxx_overlay_upright.png' else 'reference_or_hint')
            with Image.open(path) as im:
                size = list(im.size)
            catalog.append({'image_id': f'image_{index}', 'role': role, 'size': size,
                            'reference_kind': 'current' if role in ('clean', 'overlay') else name})
        if sum(r['role'] == 'clean' for r in catalog) != 1 or sum(r['role'] == 'overlay' for r in catalog) != 1:
            raise ValueError('Observation skill requires one canonical current RGB/overlay pair')
        case = {'fold_goal': context['objective'], 'prompt': instructions,
                'system_prompt': 'Original fold visual selection; current RGB governs semantic targets.',
                'context': context, 'registry': context.get('locally_executable_reference_ids'),
                'observation': {'images': catalog}}
        observer = RuntimeClaude(backend='remote', ssh_host=ssh_host, model=model,
                                 timeout_s=timeout_s, max_turns=4)
        paths, views, metrics = prepare_with_skill(case, images, artifact, observer, output/'prepare',
                                                  timeout=timeout_s, max_edits=6)
        # Keep original root paths/indices stable. Added views are for visual
        # selection only; the later pixel-motion call retains original roots.
        paths = list(images) + paths[len(images):]
        bundle = {'images': views,
                  'authority': 'Host-verified image transformations only, not scene conclusions or proof of success.',
                  'instruction': 'Use useful prepared views before requesting redundant edits. Original Rxxx IDs remain authoritative. '
                                 'Additional image IDs are local to this visual call; later grounding uses original roots.',
                  'remaining_visual_edit_budget': 6-metrics['host_image_ops']}
        (output/'prepared_images.json').write_text(json.dumps(
            {'paths': [str(p) for p in paths], **bundle}, ensure_ascii=False, indent=2)+'\n')
        report.update(status='PREPARED', **metrics, observer_calls=observer.calls,
                      original_images=len(images), derived_images=len(paths)-len(images))
        return paths, bundle, report['host_image_ops']
    except Exception as exc:
        report.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_s'] = time.monotonic()-start
        (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
