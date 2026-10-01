from pathlib import Path
import json
base=Path(__file__).parent
# Load only helpers; do not rewrite the existing five editable slides.
source=(base/'make_editable.py').read_text();exec(source[:source.index('titles=')])
run=Path('/mnt/newssd/sja/clothAgent_real/runs/2026-09-24/fold_20260924T014204289928777Z')
tooldir=run/'results/fold_exploration/20260924T031106388377Z/iteration_008/claude_image_tools/visual_planning_7ae40ebded93'
per=run/'results/perception/center_20260924T052651802452Z'
history=json.loads((tooldir/'images/inspection_history.json').read_text())
crops=[Path(x['path']) for x in history['images'] if x.get('operation')]
slides=[('过程 1：Claude 实际调用图像工具',['Read manifest → list_images → view_image → crop_image → resize_image → map_point → StructuredOutput','日志示例：crop_image(image_0, [500,480,720,710])；resize_image(scale=2.5)。','以下是该次工具会话保存的图像；工具调用记录不等于模型理解正确。'],[(tooldir/'images/image_0.png','模型查看的原图')]+[(x,'工具生成局部视图') for x in crops[-2:]]),('过程 2：抓取前的候选点与 Molmo 提示',['Rxxx 是程序生成的候选点；Molmo 提供袖子区域提示，Claude 决定最终抓点。','第二段 iter1 的执行参考点为 R097；后续侧身和下摆不调用 Molmo。'],[(review/'08_00.png','抓取前 RGB'),(review/'08_01.png','Rxxx 标点图'),(root/'results/molmo_review_20260924/4_processed_image.png','Molmo 右袖提示')]),('过程 3：深度信息与表面高度',['展示同次感知保存的高度热图：由 RGB-D 和标定转换得到，并非原始相机距离图。','末轮 R005：表面 Z≈45.0 mm，抓取 Z≈37.0 mm；坐标在机器人基座系。'],[(per/'camera_0_A.png','对应 RGB'),(per/'camera_A_height_map_heatmap.png','局部高度热图'),(per/'camera_A_height_map_heatmap_global.png','全局高度热图')]),('过程 4：抓取取证与最终评估',['末轮下摆对折：比较抬升前后布料形变，再结合最终同视角结果判断。','腕部相机会随机器人移动；仅凭画面放大不能证明抓住。'],[(review/'15_02.png','过程取证帧 1'),(review/'15_03.png','过程取证帧 2'),(review/'15_04.png','释放后结果')])]
for i,(title,lines,imgs) in enumerate(slides,6):
 shapes=box(10,title,65,45,1800,90,32)
 for j,t in enumerate(lines):shapes+=box(11+j,t,70,145+j*65,1780,65,20)
 rels=[f'<Relationship Id="rId1" Type="{r}/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'];w=1760/len(imgs)
 for j,(path,label) in enumerate(imgs):
  rid=f'rId{10+j}';media=f'process_{i}_{j}.png';data['ppt/media/'+media]=path.read_bytes();rels.append(f'<Relationship Id="{rid}" Type="{r}/image" Target="../media/{media}"/>')
  shapes+=box(40+j,label,75+j*w,375,w-20,55,18)+pic(50+j,rid,path,75+j*w,435,w-20,555)
 shapes+=box(90,f'ClothAgent | 2026-09-24 run | {i}/9',65,1020,1700,40,12)
 data[f'ppt/slides/slide{i}.xml']=f'<p:sld xmlns:p="{p}" xmlns:a="{a}" xmlns:r="{r}"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr/>{shapes}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>'.encode()
 data[f'ppt/slides/_rels/slide{i}.xml.rels']=('<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'+''.join(rels)+'</Relationships>').encode()
pr=E.fromstring(data['ppt/presentation.xml']);ids=pr.find(f'{{{p}}}sldIdLst');rels=E.fromstring(data['ppt/_rels/presentation.xml.rels']);ct=E.fromstring(data['[Content_Types].xml'])
for i in range(6,10):
 E.SubElement(ids,f'{{{p}}}sldId',{'id':str(300+i),f'{{{r}}}id':f'rIdProcess{i}'})
 E.SubElement(rels,'{http://schemas.openxmlformats.org/package/2006/relationships}Relationship',{'Id':f'rIdProcess{i}','Type':r+'/slide','Target':f'slides/slide{i}.xml'})
 E.SubElement(ct,'{http://schemas.openxmlformats.org/package/2006/content-types}Override',{'PartName':f'/ppt/slides/slide{i}.xml','ContentType':'application/vnd.openxmlformats-officedocument.presentationml.slide+xml'})
data['ppt/presentation.xml']=E.tostring(pr);data['ppt/_rels/presentation.xml.rels']=E.tostring(rels);data['[Content_Types].xml']=E.tostring(ct)
for i in range(1,6):data[f'ppt/slides/slide{i}.xml']=data[f'ppt/slides/slide{i}.xml'].replace(f'{i}/5'.encode(),f'{i}/9'.encode())
for n,b in data.items():
 if n.endswith(('.xml','.rels')):E.fromstring(b)
target=out/'latest_run_report_with_process.pptx'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in data.items():z.writestr(n,b)
(out/'process_slide_sources.json').write_text(json.dumps([{'title':t,'images':[str(x) for x,l in imgs]} for t,lines,imgs in slides],ensure_ascii=False,indent=2))
print(target, '9 slides, editable text, video retained')
