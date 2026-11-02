"""Package unchanged D2/I1 RGBs and concise notes into a PDF."""
from pathlib import Path
import json,textwrap
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/corner_recoverability_20261029'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
W,H=1654,2339
pages=[]
def font(n):return ImageFont.truetype(FONT,n)
def page(title,subtitle):
 im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im)
 d.text((80,65),title,font=font(52),fill='#18334a');d.text((80,145),subtitle,font=font(26),fill='#556477')
 d.line((80,205,W-80,205),fill='#c7d4df',width=2)
 return im,d
def paragraph(d,text,y,size=32,width=1450):
 line=''
 for ch in text:
  if ch=='\n' or d.textlength(line+ch,font=font(size))>width:
   d.text((85,y),line,font=font(size),fill='#25374a');y+=size+20;line='' if ch=='\n' else ch
  else:line+=ch
 if line:d.text((85,y),line,font=font(size),fill='#25374a');y+=size+20
 return y+25
im,d=page('D2 与 I1：原始图片与简要总结','Corner Recoverability 离线核验 | 2 条轨迹 · 6 个历史动作 · 8 个状态')
y=265
for title,body in [
 ('资料范围','本PDF仅整理 D2 和 I1。每条轨迹包含第0、1、2步动作前图，以及第2步动作后图。图片保持原始内容，不叠加角点、动作箭头或裁剪。为排版等比例缩放。'),
 ('D2：需要核验画面边界','来源：2026-10-27 01:40 开始的 sleeve_hem_batch，episode_0001。原运行记录 DONE，但 DONE 不能直接视为四个语义端点已明确可见。部分下摆靠近或越过画面下边界，应区分出视野与被布层遮挡。'),
 ('I1：需要核验材料身份','来源：2026-10-22 21:17 开始的 one_shot_goal_full_episode_v1。后续包含人工中断。袖部和下摆发生折回，普通轮廓尖角不能直接当作袖口中点或真实下摆交角。'),
 ('当前结论','本轮尚未确认这两条轨迹构成“一个角显露、另一个角隐藏，随后四端同时恢复”的案例。模型标注只是待审核提案，不能当作真值。原图应优先用于核验角点身份及可见性。'),
 ('回报计算原则','H=2，γ=0.9。只有真实后续状态明确达到四端同时可见，且首次到达时间明确时，才能给出折扣回报。不确定为 UNKNOWN；不足两步的未成功后缀为 CENSORED，不直接赋零。'),
 ('执行范围','只整理历史资料；没有新增机器人动作，也没有修改控制代码或正式策略。完整数学定义、标注与审核页面见同目录 review.html。')]:
 d.text((85,y),title,font=font(36),fill='#18334a');y+=60;y=paragraph(d,body,y,29)
pages.append(im)
manifest=json.loads((OUT/'manifest.json').read_text())
for run in manifest:
 if run['id'] not in ['D2','I1']:continue
 for start in [0,2]:
  im,d=page(run['id']+' 原始图像：'+str(start)+'—'+str(start+1),'按时间顺序排列；坐标标注和动作箭头均未叠加')
  for slot,state in enumerate(run['states'][start:start+2]):
   y=245+slot*950
   label=state['id']+' | '+state['round']+' | '+('操作前' if state['role']=='before' else '操作后')
   d.text((80,y),label,font=font(35),fill='#18334a')
   with Image.open(ROOT/state['image']) as raw:
    rgb=raw.convert('RGB');rgb.thumbnail((W-160,840),Image.Resampling.LANCZOS)
    im.paste(rgb,((W-rgb.width)//2,y+65))
  d.text((80,H-125),'来源：'+run['source_run'].split('/')[1],font=font(20),fill='#556477')
  pages.append(im)
for i,im in enumerate(pages):ImageDraw.Draw(im).text((W-160,H-65),f'{i+1} / {len(pages)}',font=font(24),fill='#556477')
path=OUT/'D2_I1_original_images_summary.pdf'
pages[0].save(path,'PDF',resolution=200,save_all=True,append_images=pages[1:],title='D2 and I1 - Original Images and Offline Summary',author='Codex')
print(path);print('pages',len(pages),'bytes',path.stat().st_size)
pages[1].resize((827,1170)).save('/tmp/d2_i1_pdf_preview.jpg')
