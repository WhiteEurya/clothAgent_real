from pathlib import Path
import zipfile,subprocess,xml.etree.ElementTree as E
from PIL import Image,ImageDraw,ImageFont
out=Path(__file__).parent
ppt=out/'latest_run_report.pptx';video=out/'full_run_32x.mp4';poster=out/'video_poster.png'
subprocess.run(['ffmpeg','-v','error','-y','-ss','0.5','-i',str(video),'-frames:v','1',str(poster)],check=True)
# Clear the old three-image row, leaving the result summary above it.
im=Image.open(out/'slide_04.png');d=ImageDraw.Draw(im);d.rectangle((0,345,1919,1005),fill='white')
d.text((70,355),'32× 全过程视频 · 点击画面播放',font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',27),fill='black')
im.save(out/'slide_04.png')
with zipfile.ZipFile(ppt) as z:data={n:z.read(n) for n in z.namelist()}
data['ppt/media/image4.png']=(out/'slide_04.png').read_bytes()
p='http://schemas.openxmlformats.org/presentationml/2006/main';a='http://schemas.openxmlformats.org/drawingml/2006/main';r='http://schemas.openxmlformats.org/officeDocument/2006/relationships'
xml=data['ppt/slides/slide4.xml'].decode()
shape=f'''<p:pic><p:nvPicPr><p:cNvPr id="3" name="Full run 32x"><a:hlinkClick r:id="" action="ppaction://media"/></p:cNvPr><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr><a:videoFile r:link="rId3"/><p:extLst><p:ext uri="{{DAA4B4D4-6D71-4841-9C94-3DE7FCFB2F80}}"><p14:media xmlns:p14="http://schemas.microsoft.com/office/powerpoint/2010/main" r:embed="rId4"/></p:ext></p:extLst></p:nvPr></p:nvPicPr><p:blipFill><a:blip r:embed="rId5"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="2921000" y="2603500"/><a:ext cx="6350000" cy="3571875"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic>'''
xml=xml.replace('</p:spTree>',shape+'</p:spTree>')
timing='''<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst><p:video><p:cMediaNode vol="80000"><p:cTn id="2" fill="hold" display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="3"/></p:tgtEl></p:cMediaNode></p:video></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>'''
xml=xml.replace('</p:sld>',timing+'</p:sld>');data['ppt/slides/slide4.xml']=xml.encode()
rel=data['ppt/slides/_rels/slide4.xml.rels'].decode().replace('</Relationships>',f'<Relationship Id="rId3" Type="{r}/video" Target="../media/full_run_32x.mp4"/><Relationship Id="rId4" Type="http://schemas.microsoft.com/office/2007/relationships/media" Target="../media/full_run_32x.mp4"/><Relationship Id="rId5" Type="{r}/image" Target="../media/video_poster.png"/></Relationships>');data['ppt/slides/_rels/slide4.xml.rels']=rel.encode()
data['[Content_Types].xml']=data['[Content_Types].xml'].decode().replace('</Types>','<Default Extension="mp4" ContentType="video/mp4"/></Types>').encode();data['ppt/media/full_run_32x.mp4']=video.read_bytes();data['ppt/media/video_poster.png']=poster.read_bytes()
for n,b in data.items():
 if n.endswith(('.xml','.rels')):E.fromstring(b)
tmp=out/'embedded.tmp'
with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in data.items():z.writestr(n,b)
tmp.replace(ppt)
# PDF shows the poster, since PDF cannot play this video.
preview=im.copy();frame=Image.open(poster);frame.thumbnail((1000,563));preview.paste(frame,(460,410));preview.save(out/'slide_04_preview.png')
images=[Image.open(out/f'slide_{i:02}.png') if i!=4 else preview for i in range(1,6)]
images[0].save(out/'latest_run_report.pdf',save_all=True,append_images=images[1:],resolution=144)
print('Embedded MP4 verified:',len(data['ppt/media/full_run_32x.mp4']),'bytes; 5 slides retained')
