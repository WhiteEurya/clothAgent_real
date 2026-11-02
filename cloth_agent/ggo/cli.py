"""Offline annotation and V0/V1 evaluation. Never imports a robot runtime."""
import argparse
from dataclasses import asdict
import html
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from cloth_agent.dual_arm.observation import Observation, sha
from .observer import LANDMARKS, Settings, observe, contour_points


def save(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))


def annotate(reference,current,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    ref,cur=Observation.load(reference),Observation.load(current)
    shutil.copyfile(ref.image,out/'reference.png');shutil.copyfile(cur.image,out/'current.png')
    template=dict(reference={k:None for k in LANDMARKS},current={k:None for k in ('N','AL','AR')},
                  provenance='human manual pixel annotations',reference_rgb_sha256=sha(ref.image),current_rgb_sha256=sha(cur.image))
    save(out/'annotation_template.json',template)
    page='''<!doctype html><meta charset="utf-8"><title>GGO 上半身锚点标注</title>
<style>body{font:18px system-ui;margin:24px}canvas{max-width:95vw;border:1px solid #aaa}button,select{font:inherit;margin:8px}</style>
<h1>GGO V0/V1 标注</h1><p>参考图标 N领口中心、AL/AR左右腋下、SL/SR左右袖口外端、HL/HR左右下摆角；当前图只标 N/AL/AR。左右沿衣服自身方向延续，不能按屏幕方向交换。点在对应部位布面一侧；认不准点设为未知，不为了配准补全。</p>
<select id="frame"><option value="reference">参考图</option><option value="current">当前图</option></select><select id="point"></select><button id="unknown">设为未知</button><button id="download">下载 annotations.json</button><p id="status"></p><canvas id="canvas"></canvas><pre id="values"></pre>
<script>const data=__DATA__;const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d'),frame=document.getElementById('frame'),point=document.getElementById('point');let img=new Image();
function draw(){ctx.drawImage(img,0,0);for(const [name,p] of Object.entries(data[frame.value])){if(!p)continue;ctx.beginPath();ctx.arc(p[0],p[1],5,0,7);ctx.fillStyle='red';ctx.fill();ctx.font='18px sans-serif';ctx.strokeStyle='black';ctx.lineWidth=3;ctx.strokeText(name,p[0]+8,p[1]);ctx.fillStyle='white';ctx.fillText(name,p[0]+8,p[1]);}document.getElementById('values').textContent=JSON.stringify(data,null,2);document.getElementById('status').textContent='点击标注：'+frame.value+' '+point.value;}
function load(){point.innerHTML=Object.keys(data[frame.value]).map(k=>'<option>'+k+'</option>').join('');img.onload=()=>{canvas.width=img.width;canvas.height=img.height;draw();};img.src=frame.value+'.png';}
frame.onchange=load;point.onchange=draw;canvas.onclick=e=>{const r=canvas.getBoundingClientRect();data[frame.value][point.value]=[Math.max(0,Math.min(canvas.width-1,Math.round((e.clientX-r.left)*canvas.width/r.width))),Math.max(0,Math.min(canvas.height-1,Math.round((e.clientY-r.top)*canvas.height/r.height)))];draw();};document.getElementById('unknown').onclick=()=>{data[frame.value][point.value]=null;draw();};document.getElementById('download').onclick=()=>{let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='annotations.json';a.click();};load();</script>'''
    (out/'index.html').write_text(page.replace('__DATA__',json.dumps(template)))
    print(out/'index.html')


def render(out,report,artifacts,cur,ref,annotations):
    out=Path(out)
    for key,obs in [('current',cur),('reference',ref)]:
        im=Image.open(obs.image).convert('RGB');draw=ImageDraw.Draw(im)
        for name,p in annotations[key].items():
            if p is not None:
                x,y=p;draw.ellipse((x-5,y-5,x+5,y+5),fill='red');draw.text((x+8,y),name,fill='white',stroke_width=2,stroke_fill='black')
        im.save(out/(key+'_anchors.png'))
    if 'current_grid' in artifacts:
        shape=artifacts['current_grid'].shape
        region_image=np.full((*shape,3),30,np.uint8)
        colors={'LEFT_SLEEVE':(80,180,230),'RIGHT_SLEEVE':(220,140,70),'LEFT_LOWER_BODY':(110,210,120),'RIGHT_LOWER_BODY':(210,110,180)}
        for name,color in colors.items():region_image[artifacts['region_'+name]]=color
        rim=Image.fromarray(region_image).resize((shape[1]*2,shape[0]*2));rd=ImageDraw.Draw(rim)
        for name in colors:
            ys,xs=np.where(artifacts['region_'+name])
            if len(xs):rd.text((max(0,int(xs.mean()*2)-40),int(ys.mean()*2)),name,fill='white',stroke_width=1,stroke_fill='black')
        rim.save(out/'reference_regions.png')
        img=np.full((*shape,3),30,np.uint8)
        img[~artifacts['observed_grid']]=[75,75,75];img[artifacts['current_grid']]=[155,100,180]
        if 'missing' in artifacts:img[artifacts['missing']]=[240,80,70]
        if 'excess' in artifacts:img[artifacts['excess']]=[50,140,230]
        contours,_=cv2.findContours(artifacts['current_grid'].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,contours,-1,(40,255,100),1)
        if 'aligned_reference' in artifacts:
            contours,_=cv2.findContours(artifacts['aligned_reference'].astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            for c in contours:
                pts=c[:,0,:]
                for i in range(0,len(pts)-1,10):cv2.polylines(img,[pts[i:i+5]],False,(255,240,100),1)
        im=Image.fromarray(img);im.resize((im.width*2,im.height*2)).save(out/'metric_overlay.png')
        # Projection for human display only: reference/missing points lie on fitted table plane.
        im=Image.open(cur.image).convert('RGB');draw=ImageDraw.Draw(im)
        if 'aligned_reference' in artifacts and 'world_from_camera_mm' in cur.meta:
            import yaml
            kpath=Path('config/calibration/dual_arm_working_20261008/camA_intrinsics.yaml')
            if kpath.exists():
                K=np.array(yaml.safe_load(kpath.read_text())['K']);camera=np.linalg.inv(np.array(cur.meta.get('corrected_world_from_camera_affine_mm',cur.meta['world_from_camera_mm'])))
                table=report['table'];basis=np.array(table['basis_columns']);origin=np.array(table['origin_world_mm'])
                def project(xy):
                    world=np.c_[xy,np.zeros(len(xy))]@basis.T+origin
                    cam=np.c_[world,np.ones(len(world))]@camera.T;uv=cam[:,:3]@K.T
                    return uv[:,:2]/uv[:,2,None]
                xy=contour_points(artifacts['aligned_reference'],artifacts['origin_mm'],report['grid']['resolution_mm'])
                uv=project(xy)
                for i in range(0,len(uv)-5,10):draw.line([tuple(p) for p in uv[i:i+5]],fill='yellow',width=2)
                ys,xs=np.where(artifacts['missing']);xy=np.c_[xs,ys]*report['grid']['resolution_mm']+artifacts['origin_mm']+report['grid']['resolution_mm']/2
                for x,y in project(xy[::3]):
                    if 0<=x<im.width and 0<=y<im.height:draw.ellipse((x-1,y-1,x+1,y+1),fill='#ff6655')
                draw.text((15,15),'TABLE-PLANE PROJECTION: yellow=reference; red=missing (not hidden grasp targets)',fill='white',stroke_width=2,stroke_fill='black')
                im.save(out/'rgb_overlay.png')
    page=['<!doctype html><meta charset="utf-8"><title>GGO V0/V1</title><style>body{font:17px system-ui;max-width:1400px;margin:25px auto}img{max-width:100%}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #aaa}pre{white-space:pre-wrap}.pair{display:grid;grid-template-columns:1fr 1fr;gap:15px}</style><h1>GGO V0/V1：上半身对齐与区域偏差</h1>',
        '<p>无 GPT 调用、无机器人动作。锚点来源：'+html.escape(str(report['annotation_provenance']))+'</p>',
        '<p>配准：'+html.escape(report['registration']['status'])+'；主要异常：'+html.escape(str(report['deformation'].get('primary_region')))+'</p>',
        '<p>黄虚线：刚体对齐参考；绿线：当前轮廓；红色：容差后缺失；蓝色：超出参考；灰色：缺少实测覆盖。RGB图投影位于桌面平面，衣服抬高处会有视差，指标以毫米俯视网格为准。</p>',
        '<div class="pair"><img src="reference_anchors.png"><img src="current_anchors.png"></div>']
    if (out/'geometry_assessment.txt').exists():
        page.append('<h2>Geometry Assessment（可作为 GPT 输入）</h2><pre>'+html.escape((out/'geometry_assessment.txt').read_text())+'</pre><p><a href="geometry_assessment.json">结构化报告</a> · <a href="body_cross_sections.csv">逐截面 CSV</a></p>')
    for f in ('body_cross_sections.png','reference_regions.png','metric_overlay.png','rgb_overlay.png'):
        if (out/f).exists():page.append('<h2>'+f+'</h2><img src="'+f+'">')
    page.append('<table><tr><th>区域</th><th>缺失比例</th><th>缺失 mm²</th><th>最大块 mm²</th><th>轮廓偏差 mm</th><th>状态</th></tr>')
    for k,v in report['deformation'].get('regions',{}).items():
        page.append('<tr><td>'+k+'</td>'+''.join('<td>'+html.escape(str(v.get(n)))+'</td>' for n in ('missing_ratio','missing_area_mm2','largest_missing_component_mm2','contour_deviation_mm','state'))+'</tr>')
    page.append('</table><details><summary>完整测量、配准与不确定性</summary><pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre></details>')
    (out/'index.html').write_text('\n'.join(page))


def run(reference,current,annotations,out,settings=None):
    began=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    ref,cur=Observation.load(reference),Observation.load(current);ann=json.loads(Path(annotations).read_text())
    if ref.meta['calibration_id']!=cur.meta['calibration_id']:raise ValueError('reference/current calibration IDs differ')
    for key,obs in [('reference',ref),('current',cur)]:
        if ann.get(key+'_rgb_sha256')!=sha(obs.image):raise ValueError(key+' annotation RGB hash mismatch')
    cfg=Settings(**json.loads(Path(settings).read_text())) if settings else Settings()
    report,artifacts=observe(np.asarray(Image.open(ref.image).convert('RGB')),ref.xyz,np.asarray(Image.open(cur.image).convert('RGB')),cur.xyz,ann,cfg)
    report['sources']=dict(reference=str(ref.directory),current=str(cur.directory),reference_sha256=ref.meta['sha256'],current_sha256=cur.meta['sha256'])
    report['compute_wall_s']=time.perf_counter()-began;save(out/'report.json',report);save(out/'annotations.json',ann)
    from .assessment import write_outputs
    write_outputs(out,report)
    np.savez_compressed(out/'geometry.npz',**artifacts);render(out,report,artifacts,cur,ref,ann)
    print(json.dumps(dict(report=str(out/'index.html'),registration=report['registration'],deformation_status=report['deformation']['status'],primary_region=report['deformation'].get('primary_region'),seconds=report['compute_wall_s']),ensure_ascii=False))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['annotate','run'])
    parser.add_argument('--reference',required=True);parser.add_argument('--current',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--annotations');parser.add_argument('--settings');args=parser.parse_args()
    if args.mode=='annotate':annotate(args.reference,args.current,args.out)
    else:
        if not args.annotations:parser.error('run requires --annotations')
        run(args.reference,args.current,args.annotations,args.out,args.settings)

if __name__=='__main__':main()
