"""Camera-only capture and two independent planning requests; no robot SDK."""
import sys,json,fcntl
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from PIL import Image
from cloth_agent.dual_arm.observation import create_camera
from experiments.rsi_exploration_01.run import io
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
root=Path('reports')/('current_state_ab_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
root.mkdir(parents=True)
print('OUTPUT='+str(root),flush=True)
with open('/tmp/clothagent_codex_dual_arm.lock','w') as lease:
 fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
 raw=json.loads(Path('config/dual_arm.local.json').read_text())
 camera=create_camera(SimpleNamespace(raw=raw))
 try:
  camera.start()
  for _ in range(30):rgb,depth=camera.read()
  Image.fromarray(rgb).save(root/'current.png')
 finally:camera.stop()
 io.save(root/'capture.json',{'captured_at':datetime.now().isoformat(),'camera_serial':raw['camera']['serial'],'robot_connection':False,'robot_motion':False})
schema=io.obj({'analysis':io.S,'grasp_px':io.POINT,'target_px':io.POINT})
common='仅依据所附当前RGB，独立完成任务，不访问历史或Skill。接口：单次抓取、提起、牵拉并放下；Pa为抓点，Pb为落点，使用原图整数像素坐标，原点左上，x向右、y向下。夹爪保持Home方向。此为离线提案，不执行动作。输出analysis和grasp_px(Pa)、target_px(Pb)。'
tasks={'A':'当前目标是展开这件衣服。请选择下一次操作。','B':'请先分析当前最需要改变的布料形变。为了实现这一变化，哪些布料区域需要发生相对运动？然后选择一个能够产生这种相对运动的抓取与牵拉动作。'}
def ask(key):
 value=io.request(root/key,common,tasks[key],[root/'current.png'],schema)
 plan={'decision':'ACTION','grasp_px':value['grasp_px'],'target_px':value['target_px'],'motion_type':'lift_pull_place','grasp_angle_deg':None,'short_reason':value['analysis']}
 wire.validate_plan(plan,list(Image.open(root/'current.png').size),require_grasp_rotation=True,require_single_sentence=False)
 wire.overlay(root/'current.png',plan,root/(key+'_action.png'),key)
 print(key+' '+json.dumps(value,ensure_ascii=False),flush=True)
 return value
with ThreadPoolExecutor(max_workers=2) as pool:
 results=dict(zip(tasks,pool.map(ask,tasks)))
io.save(root/'results.json',results)
canvas=Image.new('RGB',(2560,720))
for i,key in enumerate(tasks):canvas.paste(Image.open(root/(key+'_action.png')).resize((1280,720)),(i*1280,0))
canvas.save(root/'comparison.png')
(root/'report.md').write_text('# 同一当前图像的A/B规划\n\n'+ '\n\n'.join('## '+k+'\n'+json.dumps(v,ensure_ascii=False,indent=2) for k,v in results.items())+'\n\n各一次独立gpt-6-astra/high请求；无历史Skill；未执行动作，未核验物理效果或完整Host可执行性。\n')
