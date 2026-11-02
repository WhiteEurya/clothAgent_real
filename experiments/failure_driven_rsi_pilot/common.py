"""Shared offline helpers; never imports any robot controller."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,hashlib,json,math,os,shutil,subprocess,time
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/mnt/newssd/sja/clothAgent_real')
OUT=ROOT/'experiments/failure_driven_rsi_pilot'
SEED='failure_driven_rsi_pilot_20261027'
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def freeze(name,paths):save(OUT/'freezes'/f'{name}.json',{'at':utc(),'files':{str(p.relative_to(OUT)):sha(p) for p in paths}})
def verify_freezes():
 bad=[]
 for f in (OUT/'freezes').glob('*.json'):
  for rel,s in read(f)['files'].items():
   if not (OUT/rel).exists() or sha(OUT/rel)!=s:bad.append(rel)
 if bad:raise RuntimeError('Frozen file changed: '+repr(bad))
def key(s):return hashlib.sha256((SEED+'|'+s).encode()).hexdigest()
def obj(p):return {'type':'object','properties':p,'required':list(p),'additionalProperties':False}
S={'type':'string'};SS={'type':'array','items':S}
BASE='''你是独立离线实验模型。仅分析本次附带图文，不调用工具、文件、网络、技能、历史会话或记忆，不执行动作。区分OBSERVED直接观察、INFERRED图像推断、UNKNOWN未知。不得虚构抓层、滑脱、隐藏部位、真实意图或替代动作效果。只输出指定JSON。'''
INTERFACE='''任务：使桌面上的T-shirt展开并铺平，减少可见折叠和重叠。
基础接口：单臂lift_pull_place(Pa,Pb)。Pa/Pb为操作前RGB整数像素[x,y]，原点左上，x右增、y下增，图像1280×720。Pa是抓取点，Pb是移动和落下的目标。命令序列为接近、下降、闭合、提起、水平移动、下降、分阶段释放、撤回观察。历史数值以本案例实际执行参数为准。
没有可信输入说明历史Planner的局部意图，必须将真实意图标UNKNOWN；可以单列INFERRED操作可能想改变什么。EXECUTED只表示命令完成。机械抓取检测不等于确认布料/层数。过程相机随手腕运动，视角变化不能直接解释为布料运动，遮挡和抽帧间隔保留UNKNOWN。'''

def annotate(src,a,b,dest):
 im=Image.open(src).convert('RGB');d=ImageDraw.Draw(im)
 if a is not None and b is not None:
  d.line([tuple(a),tuple(b)],fill='#ffde20',width=4)
  for p,t,col in [(a,'Pa','#ff4444'),(b,'Pb','#00ffff')]:
   x,y=p;d.ellipse((x-7,y-7,x+7,y+7),outline=col,width=3);d.text((x+10,y-18),t,fill=col)
 dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest)
