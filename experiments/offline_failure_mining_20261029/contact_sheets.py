from index import *
from PIL import ImageDraw
D=read(OUT/'index.json');dest=OUT/'inspection';dest.mkdir(exist_ok=True)
for j,r in enumerate(x for x in D['runs'] if x['screen']['0.02']['windows']):
 w=r['screen']['0.02']['first_nonprogress'];start=next(i for i,s in enumerate(r['steps']) if s['id']==w['ids'][0]);seq=r['steps'][max(0,start-1):start+4]
 sheet=Image.new('RGB',(1000,320*len(seq)), 'white');draw=ImageDraw.Draw(sheet)
 for i,s in enumerate(seq):
  for k,side in enumerate(['before','after']):
   im=Image.open(ROOT/s[side]).convert('RGB');im.thumbnail((490,285));sheet.paste(im,(k*500,i*320+30))
  a=s['action'];delta=s['evaluation']['coverage']['delta_ratio'] if s['evaluation'] else None
  draw.text((4,i*320+4),f"{s['step']} delta={delta} {a['grasp_px']} -> {a['target_px']}",fill='black')
 sheet.save(dest/f'candidate_{j}.jpg');print(j,r['path'])
