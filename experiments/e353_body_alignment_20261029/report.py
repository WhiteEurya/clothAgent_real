from pathlib import Path
import json,html
R=Path('/mnt/newssd/sja/clothAgent_real/reports/e353_body_alignment_20261029');a=json.loads((R/'analysis_corrected.json').read_text());m=json.loads((R/'alignment.json').read_text());titles=['记录的动作意图','Before可见结构','After可见结构','衣身对齐的含义','共同运动','相对形变','实际发生的变化','物理机制一','物理机制二','其他机制','如何区分机制','结论']
body='<h1>E353：先解释变化，再讨论机制</h1><p>独立会话，仅原始E353记录和图像处理结果；未传入聊天历史或以前的机制解释。第一次轮廓混入背景，已修正并重新独立分析，原请求保留。</p>'
for f,t in [('before.png','Before'),('after.png','After'),('aligned_overlay.png','印花锚定的刚体对齐，半透明叠加'),('aligned_contours_corrected.png','青色：对齐前图轮廓；紫色：后图轮廓'),('body_anchor_fit.png','胸前印花锚点（不代表整衣刚性）')]:body+='<h2>'+t+'</h2><a href="'+f+'"><img src="'+f+'"></a>'
md='# E353 独立物理分析\n\n'
for (k,v),t in zip(a.items(),titles):body+='<h2>'+t+'</h2><p>'+html.escape(v).replace('\n','<br>')+'</p>';md+='## '+t+'\n\n'+v+'\n\n'
body+='<h2>对齐测量</h2><pre>'+html.escape(json.dumps(m,ensure_ascii=False,indent=2))+'</pre>'
(R/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E353 body alignment</title><style>body{max-width:1280px;margin:30px auto;font:19px sans-serif;line-height:1.6}img{width:100%}pre{white-space:pre-wrap}</style>'+body)
(R/'report.md').write_text(md)
print(R/'index.html')
