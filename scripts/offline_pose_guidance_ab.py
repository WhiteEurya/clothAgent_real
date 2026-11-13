"""Fixed-image offline prompt comparison. No robot/camera objects are constructed."""
import argparse,copy,json,time,sys,base64,io,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw
from cloth_agent.experiments import one_shot_goal_planner_probe_v1 as wire
from cloth_agent.experiments.one_shot_goal_full_episode_v1 import direct_once
from cloth_agent.grounded_sam2 import segment_file,close_worker

GUIDANCE='对照目标图，在能有效展开的候选动作中，优先选择同时减小衣物整体位置与朝向偏差的动作，避免持续加剧偏移或旋转；已充分展开时，不为精确对齐继续操作。以下CV量仅为参考，不是执行门禁，轮廓形变可能来自折叠而非整体旋转。'
SOURCE=Path('reports/arm7_unfold_20261108_160630_770586/round_007')
GOAL=Path('reports/target_capture_20261107T163224250153Z')

def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def world(mask,meta):
 y,x=np.nonzero(mask);return (np.column_stack([x,y])-320)*meta['mm_per_pixel']*[1,-1]+meta['center_world_xy_mm']
def raster(points):
 out=np.zeros((256,256),np.uint8);p=np.rint(points*65+128).astype(int);good=(p>=0).all(1)&(p<256).all(1);p=p[good];out[p[:,1],p[:,0]]=1
 return cv2.morphologyEx(out,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
def metrics(mask,goal,cm,gm):
 a=world(mask,cm);b=world(goal,gm)
 if min(len(a),len(b))<100:raise ValueError('insufficient mask pixels')
 ca=a.mean(0);cb=b.mean(0);scale=np.sqrt(len(b))*gm['mm_per_pixel']
 an=(a-ca)/(np.sqrt(len(a))*cm['mm_per_pixel']);bn=(b-cb)/scale
 target=raster(bn);scores=[]
 for angle in range(-90,91):
  t=np.deg2rad(angle);r=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
  rotated=raster(an@r.T);scores.append(float((rotated&target).sum()/(rotated|target).sum()))
 best=int(np.argmax(scores));return dict(normalized_center_offset_world_xy=((ca-cb)/scale).round(3).tolist(),normalization='sqrt(target mask world area)',correction_angle_world_ccw_deg=best-90,iou_before=round(scores[90],3),iou_best=round(scores[best],3),iou_gain=round(scores[best]-scores[90],3),caveat='轮廓匹配估计，非语义衣身角度；折叠和对称可造成歧义，搜索限于±90度',angles=list(range(-90,91)),scores=scores)

def prepare(out):
 out.mkdir(parents=True,exist_ok=False)
 body=json.loads((SOURCE/'skill_selection/request_payload.json').read_text())
 images=[x for x in body['input'][0]['content'] if x.get('type')=='input_image']
 for name,item in zip(['current','goal','previous'],images):
  (out/f'{name}.png').write_bytes(base64.b64decode(item['image_url'].split(',',1)[1]))
 started=time.perf_counter()
 try:
  mask,_=segment_file(out/'current.png',out/'current_mask.png')
  goal,_=segment_file(out/'goal.png',out/'goal_mask.png')
 finally:close_worker()
 m=metrics(mask,goal,json.loads((SOURCE/'topdown/view.json').read_text()),json.loads((GOAL/'topdown/view.json').read_text()))
 m['offline_cv_wall_s_including_model_start']=time.perf_counter()-started;save(out/'cv_metrics.json',m)
 for name,ms in [('current',mask),('goal',goal)]:
  rgb=np.array(Image.open(out/f'{name}.png').convert('RGB'));contours,_=cv2.findContours(ms.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(rgb,contours,-1,(0,255,0),1);Image.fromarray(rgb).save(out/f'{name}_boundary.png')
 for c in 'ABC':
  request=copy.deepcopy(body)
  if c!='A':request['instructions']+='\n'+GUIDANCE
  if c=='C':request['instructions']+='\nCV参考（世界X向右，Y对应图像向上；角度表示将当前轮廓转向目标的校正量）：'+json.dumps({k:v for k,v in m.items() if k not in ['scores','angles','offline_cv_wall_s_including_model_start']},ensure_ascii=False)
  save(out/f'prepared_{c}.json',request);(out/f'prompt_{c}.txt').write_text(request['instructions'],encoding='utf-8')
 save(out/'protocol.json',dict(source=str(SOURCE),conditions={'A':'saved original request','B':'A + general guidance','C':'B + CV estimates'},schedule=['A1','B1','C1','C2','B2','A2'],model=body['model'],reasoning=body['reasoning'],images=len(images),robot_calls=0,retries=0,production_changed=False,image_sha256=[hashlib.sha256(base64.b64decode(x['image_url'].split(',',1)[1])).hexdigest() for x in images],limitations=['2 requests per condition; exploratory only','existing previous-before image held constant','no execution validation or physical outcomes']))
 print('PREPARED',out,flush=True)

def run(out):
 headers=wire.auth_headers();protocol=json.loads((out/'protocol.json').read_text());rows=[]
 for name in protocol['schedule']:
  f=out/name
  try:
   output,m=direct_once(f,json.loads((out/f'prepared_{name[0]}.json').read_text()),headers,300)
   save(f/'selection.json',output);row=dict(name=name,seconds=m['wall_latency_s'],selection=output)
   image=Image.open(out/'current.png').convert('RGB');d=ImageDraw.Draw(image)
   a=output.get('grasp_px');b=output.get('target_px');c=output.get('boundary_point_px')
   if a and b:
    d.line([tuple(a),tuple(b)],fill='yellow',width=3)
    for p,col in [(a,'red'),(b,'cyan')]:d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=col)
   elif c:d.ellipse((c[0]-5,c[1]-5,c[0]+5,c[1]+5),outline='yellow',width=2)
   image.save(f/'selection.png')
  except Exception as e:row=dict(name=name,error=f'{type(e).__name__}: {e}')
  rows.append(row);save(out/'results.json',rows);print(json.dumps(row,ensure_ascii=False),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 (prepare if a.mode=='prepare' else run)(a.output)
