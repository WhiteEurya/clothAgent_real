#!/usr/bin/env python3
"""Read-only offline dual-arm geometry preview from an existing live snapshot.
No robot connection or execution capability. Capsules are provisional, not
validated installed geometry. Trajectory progress is not execution timing.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cloth_agent.dual_arm.setup import mesh_capsules
from cloth_agent.dual_arm.kinematics import ArmModel
from cloth_agent.dual_arm.geometry import segment_distance, pose_error, pose_matrix


def generate(directory, tool_radius):
    snapshot = json.loads((directory/'snapshot.json').read_text())
    raw = json.loads((directory/'installation_config.json').read_text())
    models = {}
    warnings = ['仅离线几何预览，不能用于下发动作。',
                '底座变换来源：'+str(raw.get('base_transform_provenance',{}).get('source','未记录'))+'；此预览不提供实机安全认证。',
                f'连杆包络来自 URDF；工具半径 {tool_radius:g} mm 为可视化占位，未测量相机、支架和线缆。',
                '未加入桌面/环境、同臂自碰撞、跟踪和停车余量；进度不是实际执行时间。']
    checks = {}
    for k, arm in raw['arms'].items():
        capsules = mesh_capsules(ROOT/arm['urdf'], arm['axis'], tool_radius_mm=tool_radius,
                                 tcp_offset=arm['tcp_offset_mm_deg'])
        cfg = SimpleNamespace(arm_id=k, axis=arm['axis'], urdf=ROOT/arm['urdf'],raw=arm,
                              capsules=capsules, tcp_offset=np.array(arm['tcp_offset_mm_deg']),
                              world_from_base=np.array(arm['world_from_base_mm']))
        models[k] = ArmModel(cfg)
        row = snapshot['arms'][k]
        error = pose_error(models[k].forward(row['joints_deg']), row['tcp_pose_mm_deg'])
        checks[k] = {'model_vs_controller_tcp_mm_deg':list(error),
                     'controller_state':row['state'],'controller_errors':row['errors'],
                     'tcp_offset_matches':bool(np.allclose(row['tcp_offset_mm_deg'],arm['tcp_offset_mm_deg'],atol=.1))}
        if row['errors'] != [0,0] or row['state'] not in (0,2):
            warnings.append(f"{k}: 控制器 state={row['state']}, errors={row['errors']}；不可执行。")
        if error[0] > 2 or error[1] > 2 or not checks[k]['tcp_offset_matches']:
            warnings.append(f'{k}: 模型/TCP 配置与反馈不一致，需核对。')
    start={k:np.array(row['joints_deg']) for k,row in snapshot['arms'].items()}
    home={k:np.array(a['home_joints_deg']) for k,a in raw['arms'].items()}
    def frame(q):
        arms={}
        geometry={k:m.capsules(q[k]) for k,m in models.items()}
        for k,m in models.items():
            tcp=(m.config.world_from_base@pose_matrix(m.forward(q[k])))[:3,3]
            arms[k]={'tcp':tcp.tolist(),'capsules':[{'name':c.name,'a':c.start.tolist(),'b':c.end.tolist(),'r':c.radius} for c in geometry[k]]}
        distances=[(float(segment_distance(a.start,a.end,b.start,b.end)-a.radius-b.radius),a.name,b.name)
                   for a in geometry['left'] for b in geometry['right']]
        distance,a,b=min(distances)
        return {'arms':arms,'clearance_mm':distance,'closest_pair':[a,b]}
    routes={};summary={}
    for name,order in [('同步回 Home',[('left','right')]),('arm6 先回',[('left',),('right',)]),('arm7 先回',[('right',),('left',)])]:
        current={k:q.copy() for k,q in start.items()};frames=[]
        for group in order:
            target={k:home[k] if k in group else q for k,q in current.items()}
            for u in np.linspace(0,1,51):
                s=10*u**3-15*u**4+6*u**5
                frames.append(frame({k:q+(target[k]-q)*s for k,q in current.items()}))
            current={k:q.copy() for k,q in target.items()}
        minimum=min(frames,key=lambda f:f['clearance_mm'])
        routes[name]=frames
        summary[name]={'minimum_sampled_capsule_clearance_mm':minimum['clearance_mm'],
                       'closest_pair':minimum['closest_pair'],
                       'status':'MODEL_CLEARANCE_VIOLATION' if minimum['clearance_mm']<=raw['limits']['clearance_mm'] else 'SAMPLED_ONLY_NOT_CERTIFIED',
                       'frames':len(frames)}
    report={'physical_execution':False,'warnings':warnings,'checks':checks,'current':frame(start),'home':frame(home),'routes':summary,
            'source':'snapshot.json + installation_config.json','tool_placeholder_radius_mm':tool_radius}
    (directory/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    payload={'warnings':warnings,'routes':routes,'summary':summary}
    template=(ROOT/'scripts/dual_home_snapshot_template.html').read_text()
    (directory/'preview.html').write_text(template.replace('__DATA__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))
    print(json.dumps({'directory':str(directory),'checks':checks,'routes':summary,'current_clearance_mm':report['current']['clearance_mm'],'home_clearance_mm':report['home']['clearance_mm']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--tool-radius-mm',type=float,default=45,help='UNMEASURED display placeholder')
    a=p.parse_args()
    if not 0<a.tool_radius_mm<172: p.error('placeholder radius must be between 0 and 172 mm')
    generate(a.directory,a.tool_radius_mm)
