"""Reference-only hem landmarks; never modify observation or geometry inputs."""
import base64
import hashlib
import json
import os
from pathlib import Path


REGISTRY = Path(__file__).resolve().parents[2] / 'data/reference/hem_annotations/registry.json'


def apply_goal_annotation(root, folder, body):
    """Replace only a byte-identical registered goal in the outgoing request."""
    goal = Path(root) / 'goal/target.png'
    report = {'applied': False, 'reason': 'unregistered_goal'}
    if os.environ.get('CLOTH_GOAL_HEM_ANNOTATION', '1') == '0':
        report['reason'] = 'disabled'
    elif goal.is_file() and REGISTRY.is_file():
        raw = goal.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        report['source_sha256'] = digest
        entry = json.loads(REGISTRY.read_text(encoding='utf-8')).get(digest)
        if entry:
            original = 'data:image/png;base64,' + base64.b64encode(raw).decode()
            # Both current pipelines use current, goal, then optional history.
            # Require the second image to match; never replace by position alone.
            images = [item for message in body.get('input', [])
                      for item in message.get('content', [])
                      if item.get('type') == 'input_image']
            asset = REGISTRY.parent / entry['image']
            report['reason'] = 'goal_payload_mismatch'
            if len(images) >= 2 and images[1].get('image_url') == original:
                if asset.is_file():
                    annotated = asset.read_bytes()
                    images[1]['image_url'] = 'data:image/png;base64,' + base64.b64encode(annotated).decode()
                    (Path(folder) / 'annotated_goal.png').write_bytes(annotated)
                    report.update(applied=True, reason='registered_reference',
                                  asset=str(asset), landmarks=entry['landmarks'],
                                  annotated_sha256=hashlib.sha256(annotated).hexdigest())
                else:
                    report['reason'] = 'annotation_asset_missing'
    (Path(folder) / 'goal_annotation.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report
