"""World-aligned goal silhouette for single-request reposition decisions."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from cloth_agent.garment_chord_reorient import NAME as CHORD_NAME
CHORD_SKILL = Path(__file__).resolve().parents[2]/'skills'/CHORD_NAME/'SKILL.md'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def project(points, metadata):
    points=np.asarray(points,float)
    center=np.asarray(metadata['center_world_xy_mm'],float)
    return (points-center)*np.array([1.,-1.])/float(metadata['mm_per_pixel'])+320

def attach(root, folder, before, body):
    from .one_shot_goal_planner_probe_v1 import image_data
    goal=Path(root)/'goal/target.png'
    source=goal.with_suffix('.boundary.json')
    if not source.exists() or getattr(before,'plan_coordinates',None)!='topdown':
        return False
    data=json.loads(source.read_text(encoding='utf-8'))
    if data['goal_sha256']!=digest(goal):
        raise ValueError('goal boundary belongs to a different target image')
    metadata=json.loads((Path(before.directory)/'view.json').read_text(encoding='utf-8'))
    points=project(data['world_xy_mm'],metadata)
    rgb=np.array(Image.open(before.image).convert('RGB'))
    cv2.polylines(rgb,[np.rint(points).astype(np.int32)],True,(30,240,60),1,cv2.LINE_AA)
    output=Path(folder)/'goal_boundary_overlay.png';Image.fromarray(rgb).save(output)
    # Replace the existing current image; no additional view or model request.
    for item in body['input'][0]['content']:
        if item.get('type')=='input_image':
            item['image_url']=image_data(output);break
    body['instructions']+='\n绿色轮廓是展开目标在当前世界坐标中的位置参考，不是当前衣物边缘，也不是硬边界。纠偏仅通过 garment-chord-reorient 计算，不用 Simple 或自由选点回拉代替。'
    body['instructions']+='\n【割线几何纠偏Skill；独立于覆盖层揭开】\n'+CHORD_SKILL.read_text(encoding='utf-8').split('## 当前双 Arm7 规划',1)[1]
    (Path(folder)/'goal_boundary_reference.json').write_text(json.dumps(dict(
        source=str(source),goal_sha256=data['goal_sha256'],
        chord_skill_sha256=digest(CHORD_SKILL),boundary_pixels=points.tolist(),image=str(output),coordinate_frame='world_xy_mm',
        replaces_current_rgb=True),ensure_ascii=False,indent=2),encoding='utf-8')
    return True
