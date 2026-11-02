from pathlib import Path
import json,html
R=Path('/mnt/newssd/sja/clothAgent_real/reports/e353_intention_outcome_20261029');a=json.loads((R/'analysis.json').read_text())
titles={'intended_object':'1. 意图：它以为抓住什么','intended_target_and_prediction':'意图：希望改变什么','actual_contact':'2. 实际接触','actual_grasp':'实际夹持','actual_transmitted_motion':'实际牵动范围','observed_target_region':'3. 原目标区域的变化','observed_unplanned_regions':'未计划区域的变化','matched_expectations':'符合预期','unfulfilled_or_unverified_expectations':'未发生或尚未证实的预期','unexpected_benefit':'意外收益','unrecognized_relationship':'此前未意识到的关系','testable_new_hypothesis':'新可检验假设','hypothesis_prediction':'假设预测','hypothesis_falsifier':'反证条件','evidence_limits':'证据边界','conclusion':'结论'}
body='<h1>E353：Intention–Outcome Mismatch</h1><p>独立新会话；只输入原始决策、真实前后图、实际执行参数、过程帧与夹爪反馈。未输入聊天历史、先前分析、配准结果或专家规则。</p>'
for n,t in [('before.png','Before'),('recorded_action.png','原始 Pa/Pb'),('after.png','After'),('process_1.jpg','过程帧 15–29.4秒'),('process_2.jpg','过程帧 32–44秒')]:body+='<h2>'+t+'</h2><a href="'+n+'"><img src="'+n+'"></a>'
md='# E353：意图、实际操作与观察结果\n\n本报告来自独立新会话，不包含此前解释。\n\n'
for k,t in titles.items():v=a[k];body+='<h2>'+t+'</h2><p>'+html.escape(v).replace('\n','<br>')+'</p>';md+='## '+t+'\n\n'+v+'\n\n'
body+='<h2>过程帧原图</h2>'
for f in json.loads((R/'process_frames.json').read_text()):body+='<p><a href="'+f['image']+'">'+str(round(f['frame_t_s'],2))+' 秒完整图片</a></p>'
body+='<p><a href="calls/jpeg_independent_intention_outcome/input.txt">隔离会话完整输入</a> · <a href="analysis.json">原始输出</a></p>'
(R/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>E353 intention vs outcome</title><style>body{max-width:1280px;margin:30px auto;font:19px sans-serif;line-height:1.6}img{width:100%}</style>'+body)
(R/'report.md').write_text(md);print(R/'index.html')
