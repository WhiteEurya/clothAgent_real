"""Read saved camera observations; independent vision trials, no robot calls."""
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
import copy,json,hashlib,sys,random
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as io
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
from cloth_agent.experiments.unfold_skill_planner import SELECT_SCHEMA
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'reports/arm7_unfold_20261106_013638_820130'
OBS=RUN/'initial_observation';TOP=RUN/'round_000/topdown'
GOAL=ROOT/'data/reference/unfold_goal_20261106_004747'
VIEWS={'zed':(OBS/'rgb.png',GOAL/'observation/rgb.png'),
       'realsense_color':(OBS/'realsense/color_rgb.png',GOAL/'observation/realsense/color_rgb.png'),
       'fused':(TOP/'planning_rgb.png',GOAL/'goal.png'),
       'fused_unmatched':(TOP/'unmatched_rgb.png',GOAL/'topdown/unmatched_rgb.png'),
       'zed_topdown':(TOP/'zed_rgb.png',GOAL/'topdown/zed_rgb.png')}
POINT={'anyOf':[dict(type='array',items=dict(type='integer'),minItems=2,maxItems=2),dict(type='null')]}
PART=dict(type='object',additionalProperties=False,properties=dict(name=dict(type='string',enum=['collar','left_sleeve','right_sleeve','left_hem_corner','right_hem_corner']),status=dict(type='string',enum=['VISIBLE','OCCLUDED','UNCERTAIN','OUT_OF_FRAME']),pixel=POINT,evidence=dict(type='string')),required=['name','status','pixel','evidence'])
SCHEMA=dict(type='object',additionalProperties=False,properties=dict(parts=dict(type='array',items=PART,minItems=5,maxItems=5),occlusion_description=dict(type='string'),image_limitations=dict(type='string'),garment_truncated=dict(type='boolean')),required=['parts','occlusion_description','image_limitations','garment_truncated'])
RECOGNITION='''第一张是当前衣物，第二张是同一衣物的展开目标。只识别，不规划动作。分别报告领口、左右袖口、左右下摆角的可见性；左右按目标衣物自身对应部位，无法确定身份就填UNCERTAIN。VISIBLE表示能辨认该部位身份，不是只看到任意布边；OCCLUDED需要说明覆盖关系，轮廓不清不等于被覆盖。像素用第一张图坐标，无法定位填null。说明可见覆盖关系及图像限制，不猜测实验者想要的结论。五个部位各报告一次。'''
PLANNING='''第一张是当前衣物，第二张是展开目标。选择一步lift_pull_place使衣物接近目标。先确定要展开的部位：部位外露可抓时skill=null并输出最终grasp_px/target_px；部位被布层覆盖无法直接抓时，选择几何skill并输出相关可见边界boundary_point_px，最终两点填null。不要把附近折边当成目标部位，也不要仅因褶皱或内收使用skill。轮廓定位用cloth-contour-normal-grasp，局部切线用garment-tangent-grasp，同一边界两端清楚可用garment-chord-grasp并给chord_endpoints_px。Host沿内法线取正向最远交点。普通分支boundary_point_px=null、chord_endpoints_px=[]。pull_distance_mm为10–150；像素均用第一张图，不猜隐藏点。不要求必须使用skill。无法提出可靠动作则ABSTAIN，已展开则DONE；非ACTION的skill与各点为null，chord_endpoints_px=[]。region_name/evidence说明目标部位和选择依据。本次只离线判断，不执行，不声称已经通过深度或运动验证。'''

def main():
 out=ROOT/'reports'/('visibility_ablation_'+datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
 jobs=[(view,task,i+1) for view in ['zed','realsense_color','fused'] for task in ['recognition','planning'] for i in range(3)]
 jobs += [(view,'recognition',i+1) for view in ['fused_unmatched','zed_topdown'] for i in range(2)]
 random.Random(602).shuffle(jobs)
 protocol=dict(source=str(RUN),requests=len(jobs),model=io.MODEL,effort=io.EFFORT,max_concurrent=2,robot_calls=0,history=False,retries=0,recognition_prompt=RECOGNITION,planning_prompt=PLANNING,
    limitations=['Single garment and folded state; no causal color/size isolation','Native views include perspective; targets use matching view','Standardized planning prompt is not identical to production; compare earlier identical-production repeats separately'],
    views={k:{'current':str(a),'goal':str(b),'current_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'goal_sha256':hashlib.sha256(b.read_bytes()).hexdigest()} for k,(a,b) in VIEWS.items()})
 io.save(out/'protocol.json',protocol);print('OUTPUT',str(out),flush=True);headers=io.auth_headers()
 def trial(view,task,index):
  current,goal=VIEWS[view];folder=out/f'{view}_{task}_{index}'
  body=io.payload(current,goal,Image.open(current).size)
  body['instructions']=RECOGNITION if task=='recognition' else PLANNING
  body['text']['format'].update(name='garment_visibility' if task=='recognition' else 'unfold_selection',schema=SCHEMA if task=='recognition' else SELECT_SCHEMA)
  row=dict(view=view,task=task,repeat=index)
  try:
   value,metrics=direct_once(folder,body,headers,300)
   row.update(status='COMPLETED',result=value,latency_s=metrics['wall_latency_s'])
  except Exception as exc:row.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
  return row
 rows=[]
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(trial,*job) for job in jobs]
  for f in as_completed(futures):
   row=f.result();rows.append(row);io.save(out/'results.json',rows)
   brief={k:row[k] for k in ['view','task','repeat','status']}
   if row['status']=='COMPLETED':
    value=row['result']
    brief.update(skill=value.get('skill'),decision=value.get('decision'),parts={p['name']:p['status'] for p in value.get('parts',[])},latency_s=row['latency_s'])
   else:brief['error']=row['error']
   print(json.dumps(brief,ensure_ascii=False),flush=True)
 print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
