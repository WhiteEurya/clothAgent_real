"""Export every visible case and expanded appendix using system ReportLab."""
from pathlib import Path
import json,html
import itertools
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image,Table,TableStyle,PageBreak,KeepTogether
from reportlab.lib.pagesizes import A3,landscape
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]/'reports/hypothesis_bank_positive_images_20261102'
def main():
    pdfmetrics.registerFont(TTFont('CJK','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
    pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    styles=getSampleStyleSheet()
    for name,size,leading in [('TitleCN',25,34),('HeadingCN',18,25),('BodyCN',11,16),('SmallCN',8,12),('NoteCN',10,15)]:
        styles.add(ParagraphStyle(name,fontName='CJK',fontSize=size,leading=leading,wordWrap='CJK',spaceAfter=5,textColor=colors.HexColor('#193047')))
    styles['NoteCN'].backColor=colors.HexColor('#fff2d9');styles['NoteCN'].borderPadding=5
    def para(s,style='BodyCN'):
        # DroidSansFallback contains CJK but not Latin/digits; explicitly mix fonts.
        groups=itertools.groupby(str(s),lambda c:32<=ord(c)<127 or c in '→·•–—')
        markup=''.join('<font name="'+('Latin' if latin else 'CJK')+'">'+html.escape(''.join(chars))+'</font>' for latin,chars in groups)
        return Paragraph(markup,styles[style])
    assets=ROOT/'pdf_assets';assets.mkdir(exist_ok=True)
    def jpg(path):
        dest=assets/(path.parent.name+'_'+path.stem+'.jpg')
        if not dest.exists():
            with PILImage.open(path) as image:image.convert('RGB').save(dest,quality=90,optimize=True)
        return str(dest)
    es=json.loads((ROOT/'positive_cases.json').read_text());byid={e['id']:e for e in es}
    ordered=[byid[x] for x in ['X0518','X0311','X0331']]+[e for e in es if e['id'] not in ['X0518','X0311','X0331']]
    assert len(ordered)==61
    goals={'Reveal(LS)':'穿着者左袖口','Reveal(RS)':'穿着者右袖口','Reveal(LH)':'穿着者左下摆角','Reveal(RH)':'穿着者右下摆角'}
    page_w,page_h=landscape(A3);margin=30;usable=page_w-2*margin;pic_width=(usable-14)/2
    pdf=ROOT/'full_positive_image_report.pdf'
    story=[para('正向变化图片报告 · 完整版','TitleCN'),
           para('61 个案例｜全部 Before / After｜历史 Pa → Pb｜原始意图与观察证据','HeadingCN'),Spacer(1,15),
           para('本文件完整收录网页中的全部案例与展开附录。来自532条已完成分析：2例袖口显露候选、1例可能涉及外部工具遮挡的显露、58次端部可见性改善（66个端点事件）。'),
           para('重要：确认的Corner Reveal为0例。这里展示正向可见性观察，不将暂定证据写成已验证动作成功，也不据此计算机器人成功率。','NoteCN'),
           para('所有干净Before/After保持完整画幅。黄色箭头只出现在独立的历史动作图上。LS/RS及LH/RH按穿着者左右定义，与画面左右可能相反。'),
           para('原始记录中的水平距离、索引抬升参数不等于实测布料位移或离台高度；抓取材料身份、夹持层数和成功获得状态仍可能未知。'),
           para('本次仅重新排版已有报告，没有新增AI分析、修改标签、合并假设或执行机器人。'),
           para('报告顺序：X0518、X0311 → X0331（外部遮挡混淆）→ 其余58个可见性改善案例。'),
           para('原网页版及逐例JSON证据保留在同目录 review.html 与 evidence/ 下。'),PageBreak()]
    for i,e in enumerate(ordered):
        rid=e['id'];d=ROOT/'evidence'/rid;a=e['action']
        category='袖口显露候选 / PROVISIONAL' if i<2 else ('外部遮挡混淆 / 非确认自折叠恢复' if i==2 else '仅可见性改善 / 非确认角点恢复')
        title=f'{i+1:02d}/61  {rid} · '+ ' / '.join(goals[c['goal']] for c in e['corners'])
        story += [para(title,'HeadingCN'),para(category,'NoteCN'),para(e['source'].replace('/mnt/newssd/sja/clothAgent_real/reports/',''),'SmallCN')]
        images=Table([[para('操作前 · 干净原图'),para('操作后 · 干净原图')],
                      [Image(jpg(d/'before.png'),width=pic_width,height=pic_width*720/1280),Image(jpg(d/'after.png'),width=pic_width,height=pic_width*720/1280)]],
                     colWidths=[usable/2]*2)
        images.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('RIGHTPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
        story += [images,Spacer(1,5),para('原本想做：'+e['original_intention'].get('short_reason','原意图缺失')),
                  para(f'实际命令：Pa {a.get("grasp_px")} → Pb {a.get("target_px")}；记录水平距离 {a.get("distance_mm",0):.1f} mm；索引抬升 {a.get("lift_height_mm","未知")} mm。')]
        for c in e['corners']:
            story += [para(goals[c['goal']]+'｜已有分析记录的正向变化：'+c['after_evidence']),
                      para(c['change']+' / '+c['evidence_level']+'｜'+c['alternative_explanation'],'NoteCN')]
        # Full expanded webpage detail, not a summary or omitted appendix.
        details=[para('Pa → Pb 标注与伴随变化','BodyCN')]+[para('• '+v,'SmallCN') for v in e['whole_garment_changes']]
        action_width=260
        table=Table([[Image(jpg(d/'action.png'),width=action_width,height=action_width*720/1280),details]],colWidths=[action_width+12,usable-action_width-12])
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story += [table,para(f'完整结构化来源：evidence/{rid}/record.json','SmallCN')]
        if i<len(ordered)-1:story.append(PageBreak())
    def footer(canvas,doc):
        canvas.saveState();canvas.setFont('CJK',8);canvas.setFillColor(colors.HexColor('#61748a'))
        canvas.drawString(margin,14,'完整正向变化图片报告｜暂定视觉证据，不是已验证成功率')
        canvas.setFont('Latin',8);canvas.drawRightString(page_w-margin,14,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(str(pdf),pagesize=landscape(A3),leftMargin=margin,rightMargin=margin,topMargin=24,bottomMargin=28,
                          title='完整正向变化图片报告：61个案例',author='Cloth experiment report')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(json.dumps({'pdf':str(pdf),'cases':len(ordered),'size_bytes':pdf.stat().st_size},ensure_ascii=False))

if __name__=='__main__':main()
