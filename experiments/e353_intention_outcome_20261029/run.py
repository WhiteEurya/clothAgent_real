from pathlib import Path
import json,sys,shutil,hashlib
from PIL import Image,ImageDraw
ROOT=Path('/mnt/newssd/sja/clothAgent_real');S=ROOT/'reports/controlled_ablation_20261026/robot_v2/runs/episode_0007/round_004';R=ROOT/'reports/e353_intention_outcome_20261029';R.mkdir(exist_ok=True)
for src,dst in [('before/rgb.png','before.png'),('after/rgb.png','after.png'),('plan.json','plan.json'),('geometry.json','geometry.json'),('execution.json','execution.json')]:shutil.copyfile(S/src,R/dst)
fs=[json.loads(l) for l in (S/'execution_images/frames.jsonl').read_text().splitlines()];states=[json.loads(l) for l in (S/'states.jsonl').read_text().splitlines()]
times=[15,19.5,21,24.5,27,29.5,32,35,37.5,40,41.5,44]
selected=[]
for t in times:
 f=min(fs,key=lambda x:abs(x['t_s']-t));v=min(states,key=lambda x:abs(x['t_s']-f['t_s']));dst='frame_'+f['file'];shutil.copyfile(S/'execution_images'/f['file'],R/dst)
 selected.append(dict(image=dst,frame_t_s=f['t_s'],feedback_t_s=v['t_s'],tcp_mm_deg=v['world_tcp_mm_deg'],gripper_pulse=v['gripper_pulse']))
for sheet in range(2):
 canvas=Image.new('RGB',(1280,1170),'white');d=ImageDraw.Draw(canvas)
 for i,row in enumerate(selected[sheet*6:sheet*6+6]):
  x=(i%2)*640;y=(i//2)*390;im=Image.open(R/row['image']).convert('RGB');im.thumbnail((640,360));canvas.paste(im,(x,y+30));d.text((x+5,y+7),f"t={row['frame_t_s']:.2f}s Z={row['tcp_mm_deg'][2]:.1f}mm gripper={row['gripper_pulse']}",fill='black')
 canvas.save(R/f'process_{sheet+1}.jpg',quality=93)
im=Image.open(R/'before.png').convert('RGB');d=ImageDraw.Draw(im);p=json.loads((R/'plan.json').read_text());x,y=p['grasp_px'],p['target_px'];d.line([tuple(x),tuple(y)],fill='yellow',width=4)
for q,c,lab in [(x,'lime','Pa'),(y,'red','Pb')]:d.ellipse((q[0]-6,q[1]-6,q[0]+6,q[1]+6),fill=c);d.text((q[0]+8,q[1]-10),lab,fill=c)
im.save(R/'recorded_action.png')
g=json.loads((S/'gripper_004.json').read_text())['feedback'];compact={'final_feedback':g['feedback'],'completion':{k:g['completion'].get(k) for k in ['status','reason','duration_s','target_position_pulse']},'meaning':'Mechanical gripper state is not a semantic material identity label.'}
payload={'task':'严格区分Intended Manipulation、Actual Manipulation、Observed Effect，分析Intention–Outcome Mismatch。不事后修改意图，不预设正确抓点规则。最后仅从原始意图和本次真实结果提出一个新的可检验操作假设。不要重点做运动配准或先总结抓点方向规律。','image_order':['clean Before','clean After','Before with logged Pa/Pb (annotation not actual material identity)','chronological process sheet 1: row-major','chronological process sheet 2: row-major','full-resolution close-period frame','full-resolution lift-period frame','full-resolution transfer-period frame'],'original_decision':p,'actual_parameters':json.loads((R/'geometry.json').read_text()),'execution':json.loads((R/'execution.json').read_text()),'gripper_evidence':compact,'process_frames':selected,'caveats':['Camera is wrist-mounted; moving viewpoints and occlusion may hide gripper contact. Pixel motion across process frames is not material motion.','Frame/feedback nearest-time association saved; not exact synchronous correspondence.','Original planner right-side wording may be image-side, not anatomical side. Preserve ambiguity.','UNKNOWN required for unverified contact identity, layers, or exact material correspondence.','Expected effects explicitly recorded must be separated from plausible interpretation.','Unexpected benefit means absent from recorded local intention, not unknowable internal thoughts.']}
s={'type':'string'};fields={k:s for k in ['intended_object','intended_target_and_prediction','actual_contact','actual_grasp','actual_transmitted_motion','observed_target_region','observed_unplanned_regions','matched_expectations','unfulfilled_or_unverified_expectations','unexpected_benefit','unrecognized_relationship','testable_new_hypothesis','hypothesis_prediction','hypothesis_falsifier','evidence_limits','conclusion']};schema={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
images=[R/n for n in ['before.png','after.png','recorded_action.png','process_1.jpg','process_2.jpg',selected[2]['image'],selected[5]['image'],selected[7]['image']]]
manifest={'source':str(S),'chat_history_included':False,'prior_analysis_included':False,'alignment_results_included':False,'images':[{'path':str(x.relative_to(R)),'sha256':hashlib.sha256(x.read_bytes()).hexdigest()} for x in images],'frame_selection':'Nearest frames to preset temporal samples covering approach, closure, lift, transfer, release; no selection by outcome.'};(R/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));(R/'process_frames.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
sys.path.insert(0,str(ROOT/'reports/gcpe_v01_20261027'));import engine;engine.OUT=R
result=engine.call('independent_intention_outcome','你在全新隔离会话中分析一次真实机器人经历。只使用提供的原始决策、执行记录和图像，不读取其他文件，不调用工具。中文回答。把规划器语义判断当成信念而非事实；引用具体过程帧时间支持实际夹持判断。无法看到接触就明确UNKNOWN。不要为规划器事后补充未记录的目标。不要引入任何已知专家规则。',payload,images,schema,'independent_intention_outcome',['E353'])
engine.save(R/'analysis.json',engine.require(result));print(json.dumps(result['output'],ensure_ascii=False,indent=2))
