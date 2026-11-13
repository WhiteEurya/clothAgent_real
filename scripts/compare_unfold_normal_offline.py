"""Paired offline visual-planning comparison; never opens cameras or robots."""
from pathlib import Path
import sys,json,copy,shutil,hashlib
from concurrent.futures import ThreadPoolExecutor,as_completed
import numpy as np
from PIL import Image,ImageDraw,ImageFont
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
from cloth_agent.experiments.unfold_skill_planner import _run_once,segment
from cloth_agent.experiments.unfold_planning_skills import append
from cloth_agent.experiments.unfold_dual_topdown import DualPrimaryView,sources
from cloth_agent.dual_arm.observation import Observation
from cloth_agent import restricted_context_one_shot as validator

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/arm7_unfold_20261105_204512_527642'
OUT=ROOT/'reports/unfold_normal_ablation_20261105'
COMMON='''你是衣物展开的单步规划器。第一张是当前双相机topdown，第二张是目标图。
仅依据这两张图独立判断，不提供历史。目标是主要衣身、袖子、下摆的展开形态接近目标，允许整体平移旋转和普通小皱纹。不要为了追求像素外形一致而拉动已展开的衣物。
若已达到目标主要展开形态，输出DONE；无法提出有依据的一步动作时ABSTAIN；有明确待展开区域时ACTION。
只能使用lift_pull_place，保持Home夹爪姿态、持布不旋转，不做抚平。普通动作Pa在可见布料，Pb在当前可见桌面，实测XY幅度10–150mm；像素不是毫米，不猜隐藏点。抓点在哪侧由哪台机械臂单独执行，另一臂静止。运动保护由Host负责，本次仅离线规划，不执行。
说明动作要改善哪一处折叠或遮挡，理由简短。不得把目标图的像素坐标照搬到当前图。
'''
FREE='''本组自由选择有助于上述目标的可见抓点Pa和放置点Pb，不要求经过对侧法线，不必先找一个隐藏衣角才能选择动作。
输出grasp_px、target_px两点及schema要求的简短目的。ACTION时motion_type=lift_pull_place，grasp_angle_deg=null，is_smoothing=false。非ACTION时动作相关字段为null、is_smoothing=false。'''
NORMAL='''本组仅使用所附三个对侧法线skill。使用条件是当前图看不到对应衣角；选该区域邻近可见边界Q，Host由法线最远对侧交点生成抓点并继续向外拉。不能直接选择最终抓放点。无适用不可见衣角时，根据目标决定DONE或ABSTAIN。'''

def task(index,arm,headers):
    case=OUT/f'round_{index:03d}';folder=case/arm;folder.mkdir()
    image=case/'current.png';goal=OUT/'goal.png'
    body=io.payload(image,goal,list(Image.open(image).size));body['instructions']=COMMON
    b=Observation.load(json.loads((SOURCE/f'round_{index:03d}/observation_used.json').read_text())['directory'])
    # Saved top-down mappings, exactly matching this run's RGB; no recapture.
    view=DualPrimaryView(b,SOURCE/f'round_{index:03d}/topdown',sources(b))
    # Keep diagnostic writes out of the source run.
    maps=folder/'topdown';maps.mkdir()
    for name in ['planning_rgb.png','world_xyz_mm.npy','source_xy.npy','source_camera.npy']:
        shutil.copyfile(view.directory/name,maps/name)
    view=DualPrimaryView(b,maps,sources(b))
    if arm=='free':
        body['instructions']+=FREE;body['text']['format'].update(name='free_unfold_pair',schema=validator.SCHEMA)
        plan,metrics=direct_once(folder/'request',body,headers,300)
        plan=validator.validate(plan,Image.open(image).size,False)
    else:
        body['instructions']=append(ROOT,folder,COMMON+NORMAL)
        plan,metrics=_run_once(ROOT,folder,view,body,headers,300,direct_once,{})
    result=dict(round=index,condition=arm,plan=plan,latency_s=metrics['wall_latency_s'],
        model=io.MODEL,effort=io.EFFORT,robot_calls=0)
    if plan['decision']=='ACTION':
        try:
            a,z=[view.sample(plan[k],None) for k in ('grasp_px','target_px')]
            result['measured_xy_distance_mm']=float(np.linalg.norm(np.asarray(a)[:2]-np.asarray(z)[:2]))
            result['depth_status']='AVAILABLE'
        except Exception as exc:result['depth_status']=str(exc)
    io.save(folder/'result.json',result)
    io.overlay(image,plan,folder/'action.png',f'ROUND {index} | {arm.upper()} | OFFLINE ONLY')
    return result

def main():
    OUT.mkdir(exist_ok=False);shutil.copyfile(SOURCE/'goal/target.png',OUT/'goal.png')
    for i in [6,7,8]:
        case=OUT/f'round_{i:03d}';case.mkdir();shutil.copyfile(SOURCE/f'round_{i:03d}/topdown/planning_rgb.png',case/'current.png')
    protocol=dict(source_run=str(SOURCE),rounds=[6,7,8],conditions=['free','normal'],model=io.MODEL,effort=io.EFFORT,
        requests=6,repeats_per_case=1,history=False,semantic_review=False,robot_calls=0,
        common_prompt=COMMON,goal_sha256=hashlib.sha256((OUT/'goal.png').read_bytes()).hexdigest(),
        limitations='Exploratory comparison of full decision rules (normal eligibility plus geometry), not isolated causal test or physical effectiveness validation.')
    io.save(OUT/'protocol.json',protocol);headers=io.auth_headers();results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures={pool.submit(task,i,a,headers):(i,a) for i in [6,7,8] for a in ['free','normal']}
        for f in as_completed(futures):
            i,a=futures[f]
            try:r=f.result()
            except Exception as exc:r=dict(round=i,condition=a,error=f'{type(exc).__name__}: {exc}',robot_calls=0)
            results.append(r);io.save(OUT/'results.json',results);print(json.dumps(r,ensure_ascii=False),flush=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    canvas=Image.new('RGB',(1280,1980),'#18202c');d=ImageDraw.Draw(canvas)
    for row,i in enumerate([6,7,8]):
        for col,a in enumerate(['free','normal']):
            f=OUT/f'round_{i:03d}'/a/'action.png'
            if f.exists():canvas.paste(Image.open(f),(col*640,row*660+20))
            d.text((col*640+10,row*660),f'Round {i}: {a}',fill='white',font=font)
    canvas.save(OUT/'comparison.jpg')
if __name__=='__main__':main()
