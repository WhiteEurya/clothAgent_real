from pathlib import Path
import json,hashlib,math
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'reports/corner_reveal_mining_20261029'
source=ROOT/'reports/offline_failure_mining_20261029/index.json'
d=json.loads(source.read_text());rows=[]
for r in d['runs']:
 for s in r['steps']:
  if s['valid_pair']:rows.append(dict(s,run=r['path'],termination=r['termination']))
rows.sort(key=lambda s:(s['timestamp'],s['source']))
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for i,s in enumerate(rows):
 s['candidate_id']=f'E{i+1:03d}'
 for side in ['before','after']:
  assert hashlib.sha256((ROOT/s[side]).read_bytes()).hexdigest()==s[side+'_sha256']
for page in range(math.ceil(len(rows)/20)):
 group=rows[page*20:(page+1)*20];sheet=Image.new('RGB',(1200,200*math.ceil(len(group)/2)),'white');draw=ImageDraw.Draw(sheet)
 for i,s in enumerate(group):
  x=(i%2)*600;y=(i//2)*200
  draw.text((x+4,y+2),f"{s['candidate_id']}  BEFORE | AFTER  {s['step']}",font=font,fill='black')
  for j,side in enumerate(['before','after']):
   im=Image.open(ROOT/s[side]).convert('RGB');im.thumbnail((298,170));sheet.paste(im,(x+j*300,y+26))
 sheet.save(OUT/'overview'/f'page_{page+1:02d}.jpg',quality=93)
(OUT/'inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
(OUT/'protocol.json').write_text(json.dumps({'source_index':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'verified_pairs':len(rows),'ordering':'chronological, no area filter or rank','screen':'visual before/after overview of every verified pair, then original-resolution review of promising transitions','endpoints':'wearer LS/RS cuff-opening edge midpoint; LH/RH true hem-side seam junction; no current-image left/right renaming','success':'target occluded by garment before and identifiable/exposed after; other endpoints may worsen','status':'candidates pending human confirmation, not skill training ground truth','physical_actions':0,'policy_changes':0},ensure_ascii=False,indent=2))
print('pairs',len(rows),'pages',math.ceil(len(rows)/20))
