from pathlib import Path
from PIL import Image,ImageDraw
import json
SRC=Path('reports/minimal_one_shot_6iter_20261028_165427_789294/round_002');OUT=Path('experiments/nonlocal_unfolding_verification_01')
for name,folder in [('action',SRC/'execution_images'),('return',SRC/'return_perception/execution_images')]:
 rows=[json.loads(l) for l in (folder/'frames.jsonl').read_text().splitlines()]
 for start in range(0,len(rows),30):
  canvas=Image.new('RGB',(1600,1025),'white');d=ImageDraw.Draw(canvas)
  for j,r in enumerate(rows[start:start+30]):
   im=Image.open(folder/r['file']).convert('RGB');im.thumbnail((320,180));x=(j%5)*320;y=(j//5)*170
   im.thumbnail((310,145));canvas.paste(im,(x,y+23));d.text((x+3,y+3),f"{name} {r['file']} t={r['t_s']:.2f}s",fill='black')
  canvas.save(OUT/f'{name}_frames_{start:03d}.jpg')
 print(name,len(rows),'size',Image.open(folder/rows[0]['file']).size)
