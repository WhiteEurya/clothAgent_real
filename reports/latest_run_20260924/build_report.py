from pathlib import Path
import json,csv,shutil,importlib.util,zipfile,xml.etree.ElementTree as ET
from PIL import Image,ImageDraw,ImageFont
root=Path('/home/sja/clothAgent_real');out=Path(__file__).parent
review=root/'results/process_review_20260924';molmo=root/'results/molmo_review_20260924'
font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
f=lambda n:ImageFont.truetype(font,n)
slides=[]; notes=[]
def slide(title,lines,imgs=()):
 im=Image.new('RGB',(1920,1080),'#ffffff');d=ImageDraw.Draw(im)
 d.text((65,45),title,font=f(48),fill='#000000')
 y=125
 for line in lines:
  # wrap by rendered width
  s=''
  for ch in line:
   if d.textlength(s+ch,font=f(27))>1770:
    d.text((70,y),s,font=f(27),fill='#000000');y+=40;s=''
   s+=ch
  d.text((70,y),s,font=f(27),fill='#000000');y+=44
 if imgs:
  top=max(y+20,290);w=1760//len(imgs)
  for i,(p,label) in enumerate(imgs):
   x=75+i*w;d.text((x,top),label,font=f(24),fill='#000000')
   pic=Image.open(p).convert('RGB');pic.thumbnail((w-24,980-top-42));im.paste(pic,(x+(w-24-pic.width)//2,top+45))
 d.text((65,1025),f'ClothAgent | run 20260924T014204289928777Z | {len(slides)+1:02}',font=f(19),fill='#000000')
 p=out/f'slide_{len(slides)+1:02}.png';im.save(p);slides.append(p);notes.append({'title':title,'text':lines,'image_sources':[str(p) for p,l in imgs]})
def flow(title, subtitle, boxes, bottom):
 slide(title,[subtitle])
 p=slides[-1];im=Image.open(p);d=ImageDraw.Draw(im)
 # two rows, three nodes each; arrows follow the sequence
 coords=[(80,285),(700,285),(1320,285),(1320,600),(700,600),(80,600)]
 for i,(head,body) in enumerate(boxes):
  x,y=coords[i];d.rectangle((x,y,x+510,y+200),outline='black',width=3)
  d.text((x+22,y+24),head,font=f(32),fill='black')
  for j,line in enumerate(body.split('\n')):d.text((x+22,y+88+j*38),line,font=f(25),fill='black')
  if i<len(boxes)-1:
   if i<2:a=(x+515,y+100);b=(coords[i+1][0]-10,y+100)
   elif i==2:a=(x+255,y+205);b=(x+255,590)
   else:a=(x-5,y+100);b=(coords[i+1][0]+520,y+100)
   d.line([a,b],fill='black',width=3)
   dx=b[0]-a[0];dy=b[1]-a[1]
   if dx:d.polygon([b,(b[0]-(12 if dx>0 else -12),b[1]-7),(b[0]-(12 if dx>0 else -12),b[1]+7)],fill='black')
   else:d.polygon([b,(b[0]-7,b[1]-12),(b[0]+7,b[1]-12)],fill='black')
 d.text((80,880),bottom,font=f(27),fill='black');im.save(p)
 notes[-1]['flow_nodes']=boxes;notes[-1]['flow_note']=bottom
rows=[]
for p in sorted(review.glob('*_record.json')):
 x=json.loads(p.read_text());h=(x.get('host_compilation') or {}).get('grasp_height_resolution') or {};m=h.get('measurement') or {};ev=x.get('evaluation') or {};n=int(p.name[:2]);g=x.get('experience_generation') or {}
 rows.append({'segment':1 if n<=7 else 2,'iteration':x['iteration'],'step':x.get('planned_step'),'record_status':x.get('status'),'reference':h.get('selected_reference_id'),'surface_z_mm':(m.get('base_xyz_median_mm') or [None]*3)[2],'requested_z_mm':h.get('requested_grasp_z_mm'),'resolved_z_mm':h.get('resolved_grasp_z_mm'),'z_rewritten':h.get('z_rewritten'),'acquisition':(ev.get('grasp_acquisition') or {}).get('status'),'progress':(ev.get('task_progress') or {}).get('status'),'experience':g.get('status')})
with (out/'intermediate_variables.csv').open('w',encoding='utf-8-sig',newline='') as fp:
 wr=csv.DictWriter(fp,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
slide('我们要讲什么故事',['无需额外训练任务策略，无需 simulator / sim-to-real，','用预训练模型 + 真实反馈 + 执行约束，构建有效的 harness policy。','当前：单臂折衣已有完成案例；下一步验证稳定性与泛化。'],[(review/'01_00.png','起始状态'),(review/'15_04.png','最终状态（仍有起皱）')])
flow('算法流程：观测—规划—执行—反馈','Harness 将模型决策与机器人执行组织成闭环。',[('1 观测','RGB-D 采集\n衣物分割与三维定位'),('2 阶段判断','识别当前折叠阶段\n读取历史经验'),('3 选点与规划','Claude 从 Rxxx 候选中选点\nMolmo 仅辅助袖子定位'),('4 执行检查','解析 XYZ 与抓取高度\n工作空间 / IK 检查'),('5 执行与拍照','抓取、抬升、搬运、释放\n保存过程证据'),('6 评估与重试','比较前后状态、更新经验\n继续下一步或重新规划')], '评估结果与经验返回下一轮规划；规则更新必须通过校验。')
last=rows[-1]
slide('中间过程：以最后一轮下摆对折为例',[f'选点 {last["reference"]} → 表面 Z {last["surface_z_mm"]:.1f} mm → 抓取 Z {last["resolved_z_mm"]:.1f} mm。','候选点由程序生成，Claude 选择目标；执行后通过图像评估结果。'],[(review/'15_01.png','Rxxx 候选点'),(review/'15_02.png','过程取证'),(review/'15_04.png','执行后')])
slide('最新 run：结果与视频',['15 轮记录、13 段执行录像；重启后的 8 轮结束时，系统判为 COMPLETE。','32× 视频约 36 秒；仅拼接执行录像，不含规划等待。','新增 15 条尝试记录，成功写入 1 条条件规则；9 次经验生成 / 校验失败。','当前局限：折叠仍有起皱、运行偏慢；单次完成不代表稳定成功率。'],[(review/'08_00.png','右袖阶段'),(review/'14_00.png','右侧身阶段'),(review/'15_04.png','最终状态')])
slide('当前进度与下一步',['已完成：单臂观测—规划—执行—评估闭环，并保存视频与中间变量。','下一步 1：跑通双臂 pipeline，实现固定、牵拉与折叠协同。','下一步 2：实现衣物展平，衔接“凌乱衣物 → 展平 → 折叠”。','验证指标：重复成功率、展平 / 折叠质量、耗时与人工干预次数。'])
spec=importlib.util.spec_from_file_location('builder',root/'scripts/build_fold_report_ppt.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.pptx_from_images(slides,out/'latest_run_report.pptx')
Image.open(slides[0]).save(out/'latest_run_report.pdf',save_all=True,append_images=[Image.open(p) for p in slides[1:]],resolution=144)
shutil.copyfile(root/'results/run_videos/fold_20260924T014204289928777Z/full_run_32x_final.mp4',out/'full_run_32x.mp4')
(out/'slide_sources.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2))
(out/'README.md').write_text('# 最新 run 汇报材料\n\nlatest_run_report.pptx：5 页图片式幻灯片；文字不可逐项编辑。\nlatest_run_report.pdf：同版 PDF。\nfull_run_32x.mp4：32倍速执行视频，独立播放，未内嵌。\nintermediate_variables.csv：15轮结构化中间变量。\nslide_sources.json：内容及图片来源。\nslide_*.png：可单独插入其他PPT。\n')
with zipfile.ZipFile(out/'latest_run_report.pptx') as z:
 for n in z.namelist():
  if n.endswith('.xml') or n.endswith('.rels'):ET.fromstring(z.read(n))
print('Created',len(slides),'slides; XML valid')
