"""Embed versioned, local planning skill references into the direct vision request."""
import hashlib
import os
import json
from pathlib import Path
from .discovery_data import save

ROOT=Path(__file__).resolve().parents[2]
NAMES=('cloth-contour-normal-grasp','garment-tangent-grasp','garment-chord-grasp')
SPACE_PREPARATION = 'make-space-for-skill'



SWITCH_RULE = 'Simple只能抓当前图中可见的衣袖或者下摆。不能抓领口、肩部或衣身内部，不猜隐藏点。要展开的对应衣角或袖口被其他布层覆盖、不可见时，使用切线或割线Skill把它打开。几何Skill优先使用割线garment-chord-grasp；仅当无法可靠选择同一可见边界段的两个端点，或Host反馈割线无法生成有效候选时，才使用切线garment-tangent-grasp。'
LEGACY_SWITCH_LINES = (
    'Simple只能抓当前图中可见的衣袖或者下摆。不能抓领口、肩部或衣身内部，不猜隐藏点。要展开的对应衣角或袖口被其他布层覆盖、不可见时，使用切线或割线Skill把它打开。',
    'Simple只能抓四个边角（两个下摆角和两个袖口）；要展开的对应边角可见时用Simple，看不到时使用切线或割线Skill把它打开。',
    'Simple只能抓四个边角（两个下摆角和两个袖口）。仅当要展开的对应衣角或袖口被其他布层覆盖、不可见时使用几何 Skill；对应边角可见时用 Simple。一次未揭开且该边角仍被覆盖时可继续使用 Skill；不能仅因大块内收、褶皱、歪斜或角点难辨触发。',
    'Simple只能抓四个边角（两个下摆角和两个袖口）。默认使用 Simple；其他部分已大致展开、仍剩明显的大块折入或覆盖时使用几何 Skill。该折叠仍在就可继续使用，不因局部褶皱或角点难辨使用 Skill。',
    'Simple只能抓四个边角（两个下摆角和两个袖口）。能用 Simple 有效推进展开就用 Simple；没有有效的 Simple 展开动作时才用几何 Skill。',
    'Simple只能抓四个边角（两个下摆角和两个袖口）。默认使用 Simple；其他部分已大致展开，而剩余衣角或袖口仍被明显布层覆盖时，优先使用几何 Skill 揭开该覆盖，不要反复拉已经展开的部位。一次未揭开时，仍按当前覆盖状态选择 Skill；不因局部褶皱使用 Skill。',
    'Simple只能抓四个边角（两个下摆角和两个袖口）；要展开的边角看不到时，使用切线或割线Skill把它打开。',
    '- 条件：要展开的下摆角或袖口不可见；已可见时使用 Simple，整体已展开则 DONE。',
    '默认直接输出抓点和放点；仅当衣角或袖口明显被其他布层压住时使用。',
    '先确定本步要展开的目标部位：外露可抓时用普通 oneshot；被布层覆盖、无法直接抓取时使用本技能。不要因别处有可见角而回避被覆盖的目标，也不要仅因普通褶皱或内收使用技能。',
    '先根据目标图选择本步需要展开的部位。该部位外露时直接抓放；被布层覆盖、无法直接抓取时使用几何skill。不要用别处可见的角替代被覆盖的目标，也不要仅因褶皱或内收启动skill。',
)

def switch_policy():
    value=os.environ.get('CLOTH_SKILL_SWITCH','layered')
    if value not in ('layered','legacy'):
        raise ValueError('CLOTH_SKILL_SWITCH must be layered or legacy')
    return value


def apply_switch_rule(prompt):
    """Replace only switching criteria; legacy restores the original prompt."""
    if switch_policy()=='legacy':
        return prompt
    lines=prompt.splitlines();result=[];inserted=False
    for line in lines:
        if line.strip() in (*LEGACY_SWITCH_LINES,SWITCH_RULE):
            if not inserted:
                result.append(SWITCH_RULE);inserted=True
        else:
            result.append(line)
    return '\n'.join(result)


def planning_text(name, raw):
    """Inject the complete skill body; omit only YAML discovery metadata."""
    if raw.startswith('---\n'):
        return raw.split('\n---', 1)[1].strip()
    return raw.strip()


def append(root, folder, prompt):
    entries=[];sections=[]
    for name in NAMES:
        path=ROOT/'skills'/name/'SKILL.md'
        raw=path.read_text(encoding='utf-8')
        text=planning_text(name,raw)
        entries.append(dict(name=name,source=str(path),source_sha256=hashlib.sha256(raw.encode()).hexdigest(),planning_text=text))
        sections.append(f'【规划参考 skill：{name}】\n{text}')
    instructions="""
【衣物展开 planning skills：覆盖层揭开】
默认直接输出抓点和放点；仅当衣角或袖口明显被其他布层压住时使用。
使用skill时Codex定位相关可见边界，Host独立CV计算内法线与正向最远交点，生成抓放点；不做候选语义复核。
不调用旧Arm6执行入口。当前双Arm7的深度、分区和运动检查保持有效。
"""
    combined=apply_switch_rule(prompt+'\n'+instructions+'\n\n'.join(sections))
    folder=Path(folder)
    save(folder/'planning_skills.json',dict(mode='cv_geometry_direct',switch_policy=switch_policy(),semantic_review=False,skills=entries,
                                          local_cv_executed=False))
    (folder/'planner_prompt.txt').write_text(combined,encoding='utf-8')
    return combined


def workspace_prompt(runtime, before, view, folder):
    """Project the executor's configured planes into this observation's pixels."""
    import numpy as np
    from cloth_agent.dual_arm.geometry import pose_matrix
    raw = runtime.config.raw
    report = dict(config=str(runtime.config_path), arms={})
    lines = ['【当前执行器工作边界】抓点决定执行臂；抓点、放置点及中间路径都须满足同一臂的边界。以下是配置范围，不代表已通过深度、IK或碰撞检查。']
    view_file = Path(view.directory)/'view.json'
    if getattr(view, 'plan_coordinates', None) != 'topdown' or not view_file.exists():
        for name, arm in raw['arms'].items():
            report['arms'][name] = arm['workspace']
        save(Path(folder)/'planning_workspace.json', report)
        return '\n'.join(lines) + '\n基座坐标范围（毫米；不是像素；null表示该侧未配置界限）：' + json.dumps(report['arms'], ensure_ascii=False)
    metadata = json.loads(view_file.read_text(encoding='utf-8'))
    scale = metadata['mm_per_pixel']; cx, cy = metadata['center_world_xy_mm']
    # Homogeneous [pixel_x,pixel_y,world_z,1] -> world coordinates.
    from PIL import Image
    width,height=Image.open(view.image).size
    offset=1 if metadata.get('mode','').startswith('RGB_FIRST_') else 0
    center_x,center_y=(width-offset)/2,(height-offset)/2
    world = np.array([[scale,0,0,cx-center_x*scale], [0,-scale,0,cy+center_y*scale],
                      [0,0,1,0], [0,0,0,1]], float)
    lines.append('以下u、v是当前俯视图像素，z是实测world高度（毫米），不是图像深度猜测；抓放点和路径不得越过对应执行臂的工作边界。')
    tcp = {name:(arm.world_from_base @ pose_matrix(runtime.models[name].forward(before.meta['joints_deg'][name])))[:3,3]
           for name,arm in runtime.config.arms.items()}
    axis=tcp['right'][:2]-tcp['left'][:2];axis=axis/np.linalg.norm(axis)
    middle=(tcp['right'][:2]+tcp['left'][:2])/2
    routing=axis@world[:2,:];routing[3]-=axis@middle
    lines.append(f'按抓点分区：{routing[0]:.6f}*u + {routing[1]:.6f}*v + {routing[3]:.6f} >= 0 用right，否则用left；放置时不换臂。')
    report.update(pixel_to_world=world.tolist(),right_partition_ge_zero=routing.tolist())
    for name,arm in raw['arms'].items():
        workspace=arm['workspace'];planes=list(workspace.get('halfspaces_mm',[]))
        for index in range(3):
            for side,sign in [('min',-1),('max',1)]:
                bound=workspace[side+'_mm'][index]
                if bound is not None:
                    n=np.zeros(3);n[index]=sign
                    planes.append([*n,sign*bound])
        transform=np.linalg.inv(np.asarray(arm['world_from_base_mm']))@world
        projected=[]
        for nx,ny,nz,limit in planes:
            coefficients=np.array([nx,ny,nz])@transform[:3,:]
            row=[*coefficients[:3],float(limit-coefficients[3])]
            projected.append(row)
            norm=max(abs(row[0]),abs(row[1]))
            display=np.array(row)/(norm if norm>1e-10 else 1)
            terms=[f'{c:.6f}*{v}' for c,v in zip(display[:3],['u','v','z']) if abs(c)>1e-10]
            lines.append(f'{name}: '+ ' + '.join(terms)+f' <= {display[3]:.6f}')
        report['arms'][name]=dict(workspace=workspace,pixel_world_z_halfspaces=projected)
    save(Path(folder)/'planning_workspace.json',report)
    return '\n'.join(lines)
