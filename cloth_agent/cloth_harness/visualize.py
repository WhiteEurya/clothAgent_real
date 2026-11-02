"""Additional local diagnostics from saved CV outputs; no new estimates or models."""
import html,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .cv_harness.segmentation import segment

def enrich(root):
    root=Path(root);page=(root/'index.html').read_text()
    for f in root.glob('*/outcome.json'):
        out=json.loads(f.read_text());folder=f.parent;m=out['motion']
        details='<details><summary>测量数值与质量原因</summary><pre>'+html.escape(json.dumps({'coverage':out['coverage'],'motion':m,'quality':out['quality']},ensure_ascii=False,indent=2))+'</pre></details>'
        if m['status']=='OK':
            before=np.asarray(Image.open(folder/'before.png').convert('RGB'));after=np.asarray(Image.open(folder/'after.png').convert('RGB'));mb,_=segment(before);ma,_=segment(after)
            warped=cv2.warpAffine(mb.astype('uint8'),np.array(m['matrix']),ma.shape[::-1],flags=cv2.INTER_NEAREST).astype(bool)
            canvas=(after*.45).astype('uint8');canvas[ma&~warped]=[0,255,100];canvas[warped&~ma]=[255,60,70]
            contours,_=cv2.findContours(warped.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(canvas,contours,-1,(80,180,255),2)
            Image.fromarray(canvas).save(folder/'aligned_difference.png')
            details+='<details><summary>查看去除估计整体运动后的轮廓差异</summary><p>蓝线：变换后的前轮廓；绿：新出现的投影区域；红：消失区域。颜色不代表好坏，不是材料对应。</p><img style="width:65%" src="'+folder.name+'/aligned_difference.png"></details>'
        page=page.replace(f'<h2>{folder.name}</h2>',f'<h2>{folder.name}</h2>'+details)
    (root/'index.html').write_text(page)

if __name__=='__main__':
    import sys
    enrich(sys.argv[1])
