from pathlib import Path
import zipfile,xml.etree.ElementTree as E
from xml.sax.saxutils import escape
from PIL import Image
out=Path(__file__).parent;root=out.parents[1];review=root/'results/process_review_20260924';ppt=out/'latest_run_report.pptx'
with zipfile.ZipFile(ppt) as z:data={n:z.read(n) for n in z.namelist()}
p='http://schemas.openxmlformats.org/presentationml/2006/main';a='http://schemas.openxmlformats.org/drawingml/2006/main';r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
def em(v):return int(v*6350)
def box(id,text,x,y,w,h,size=25,border=False):
 paras=''.join(f'<a:p><a:r><a:rPr lang="zh-CN" sz="{size*100}"><a:solidFill><a:srgbClr val="000000"/></a:solidFill><a:latin typeface="Arial"/><a:ea typeface="Microsoft YaHei"/></a:rPr><a:t>{escape(t)}</a:t></a:r><a:endParaRPr lang="zh-CN" sz="{size*100}"/></a:p>' for t in text.split('\n'))
 return f'<p:sp><p:nvSpPr><p:cNvPr id="{id}" name="Text {id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x="{em(x)}" y="{em(y)}"/><a:ext cx="{em(w)}" cy="{em(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/><a:ln w="12700">'+('<a:solidFill><a:srgbClr val="000000"/></a:solidFill>' if border else '<a:noFill/>')+f'</a:ln></p:spPr><p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{paras}</p:txBody></p:sp>'
def pic(id,rid,path,x,y,w,h):
 iw,ih=Image.open(path).size;scale=min(w/iw,h/ih);ww=iw*scale;hh=ih*scale;x+=(w-ww)/2;y+=(h-hh)/2
 return f'<p:pic><p:nvPicPr><p:cNvPr id="{id}" name="Photo"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="{em(x)}" y="{em(y)}"/><a:ext cx="{em(ww)}" cy="{em(hh)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'
titles=['我们要讲什么故事','算法流程：观测—规划—执行—反馈','中间过程：下摆对折','最新 run：结果与视频','当前进度与下一步']
texts=[['无需额外训练任务策略，无需 simulator / sim-to-real。','预训练模型 + 真实反馈 + 执行约束 → 有效的 harness policy。','单臂已有完成案例；稳定性与泛化仍需验证。'],[],['R005 → 表面 Z 45.0 mm → 抓取 Z 37.0 mm。','程序生成候选点，Claude 选目标；执行后由视觉评估结果。'],['15 轮记录、13 段录像；第二段 8 轮后系统判为 COMPLETE。','新增 15 条记录，仅 1 次规则写入；9 次经验生成 / 校验失败。','视频约 36 秒（32×），不含规划等待；仍有起皱，稳定性待验证。'],['已完成：单臂观测—规划—执行—评估闭环。','下一步 1：跑通双臂 pipeline，实现固定、牵拉与折叠协同。','下一步 2：展平衣物，衔接“凌乱衣物 → 展平 → 折叠”。','验证：重复成功率、展平 / 折叠质量、耗时、人工干预次数。']]
for i in range(1,6):
 shapes=box(10,titles[i-1],65,45,1800,85,34)
 for j,t in enumerate(texts[i-1]):shapes+=box(11+j,t,70,150+j*65,1780,60,23)
 rels=[f'<Relationship Id="rId1" Type="{r}/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'];timing=''
 if i==2:
  nodes=[('1 观测','RGB-D、分割、三维定位'),('2 阶段判断','当前阶段与历史经验'),('3 选点规划','Claude + Rxxx；袖子用 Molmo'),('4 动作落地','XYZ / 抓取 Z / 工作空间 / IK'),('5 执行取证','抓取、抬升、搬运、释放'),('6 评估反馈','比较前后、更新经验、重试')]
  coords=[(70,260),(710,260),(1350,260),(1350,610),(710,610),(70,610)]
  for j,((title,body),(x,y)) in enumerate(zip(nodes,coords)):
   shapes+=box(30+j,title+'\n\n'+body,x,y,490,210,22,True)
   if j<5:shapes+=box(40+j,'→' if j<2 else ('↓' if j==2 else '←'),x+515 if j<2 else (x+200 if j==2 else x-110),y+70 if j!=2 else y+240,100,90,36)
  shapes+=box(50,'评估与经验返回下一轮；命令到位不等于抓取成功。',70,910,1780,70,23)
 if i in (1,3):
  images=[('01_00.png','起始状态'),('15_04.png','最终状态（仍有起皱）')] if i==1 else [('15_01.png','Rxxx 候选点'),('15_02.png','过程取证'),('15_04.png','执行后')]
  width=1760/len(images)
  for j,(name,label) in enumerate(images):
   path=review/name;rid=f'rId{10+j}';media=f'editable_{i}_{j}.png';data['ppt/media/'+media]=path.read_bytes();rels.append(f'<Relationship Id="{rid}" Type="{r}/image" Target="../media/{media}"/>')
   shapes+=box(60+j,label,75+j*width,380,width-20,50,20)+pic(70+j,rid,path,75+j*width,435,width-20,550)
 if i==4:
  old=E.fromstring(data['ppt/slides/slide4.xml']);movie=next(e for e in old.findall(f'.//{{{p}}}pic') if e.find(f'.//{{{a}}}videoFile') is not None)
  shapes+=E.tostring(movie,encoding='unicode');t=old.find(f'{{{p}}}timing');timing=E.tostring(t,encoding='unicode')
  for rid,typ,target in [('rId3',r+'/video','full_run_32x.mp4'),('rId4','http://schemas.microsoft.com/office/2007/relationships/media','full_run_32x.mp4'),('rId5',r+'/image','video_poster.png')]:rels.append(f'<Relationship Id="{rid}" Type="{typ}" Target="../media/{target}"/>')
  shapes+=box(80,'点击画面播放',70,350,1600,50,20)
 shapes+=box(90,f'ClothAgent | 2026-09-24 run | {i}/5',65,1020,1700,40,12)
 xml=f'<p:sld xmlns:p="{p}" xmlns:a="{a}" xmlns:r="{r}"><p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill><a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{shapes}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>{timing}</p:sld>'
 E.fromstring(xml);data[f'ppt/slides/slide{i}.xml']=xml.encode();data[f'ppt/slides/_rels/slide{i}.xml.rels']=('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(rels)+'</Relationships>').encode()
with zipfile.ZipFile(out/'editable.tmp','w',zipfile.ZIP_DEFLATED) as z:
 for n,b in data.items():z.writestr(n,b)
(out/'editable.tmp').replace(ppt)
print('5 slides converted to editable text and shapes; embedded video preserved')
