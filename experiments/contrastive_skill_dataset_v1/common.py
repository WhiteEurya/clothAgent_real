"""Isolated offline dataset helpers. No robot or policy modules imported."""
from pathlib import Path
from datetime import datetime, timezone
import argparse,hashlib,json,math,os,shutil,subprocess,time,threading
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/mnt/newssd/sja/clothAgent_real')
OUT=ROOT/'experiments/contrastive_skill_dataset_v1'
SEED='contrastive_effects_20261028_v1'
def utc():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False))
def freeze(name,paths):save(OUT/'freezes'/f'{name}.json',{'at':utc(),'files':{str(p.relative_to(OUT)):sha(p) for p in paths}})
def verify_freezes():
 for f in (OUT/'freezes').glob('*.json'):
  for p,s in read(f)['files'].items():
   if sha(OUT/p)!=s:raise ValueError('Frozen artifact changed: '+p)
def key(s):return hashlib.sha256((SEED+'|'+s).encode()).hexdigest()
def obj(p):return {'type':'object','properties':p,'required':list(p),'additionalProperties':False}
S={'type':'string'};SS={'type':'array','items':S};NUM={'type':'number'}
def enum(*xs):return {'type':'string','enum':list(xs)}
BASE='你是独立离线视觉证据标注模型。只使用本次附图和事实；不调用工具、技能、文件、网络、记忆或历史会话。不生成操作策略，不评价未执行动作。区分直接可见事实、视觉推断与UNKNOWN。只输出指定JSON。'
