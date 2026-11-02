from pathlib import Path
import sys,json
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/mnt/newssd/sja/clothAgent_real');R=ROOT/'reports/hindsight_skill_discovery_20261029_01';sys.path.insert(0,str(ROOT/'reports/gcpe_v01_20261027'));import engine;engine.OUT=R
frames=[json.loads(l) for l in (R/'live_execution/execution_images/frames.jsonl').read_text().splitlines()]
selected=[frames[i] for i in np.linspace(0,len(frames)-1,12).astype(int)]
for j in range(2):
 im=Image.new('RGB',(1280,1170),'white');d=ImageDraw.Draw(im)
 for i,f in enumerate(selected[j*6:j*6+6]):
  q=Image.open(R/'live_execution/execution_images'/f['file']).convert('RGB');q.thumbnail((640,360));x=i%2*640;y=i//2*390;im.paste(q,(x,y+30));d.text((x+5,y+7),f"t={f['t_s']:.2f}s",fill='black')
 im.save(R/f'process_{j+1}.jpg',quality=92)
engine.save(R/'process_selected_frames.json',selected)
st={'type':'string'};fields={k:st for k in ['target_identity_before','manipulation_evidence','observed_target_effect','other_changes','prediction_comparison','counterevidence','experience_comparison','limitations','next_test_suggestion','conclusion']};fields['outcome']={'type':'string','enum':['CONFIRMED_REVEAL','NO_REVEAL','UNCERTAIN']};fields['hypothesis_support']={'type':'string','enum':['SUPPORTED_THIS_TRIAL','NOT_SUPPORTED_THIS_TRIAL','INCONCLUSIVE']};schema={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
payload={'task':'只评估一次hindsight skill discovery实验。比较实际与执行前冻结预测，判断是否有意识实现未直接操作袖部显露；不继续动作、不升级Skill。','image_order':['fresh Before','After at original camera pose','process frames chronological sheet1','process frames chronological sheet2'],'frozen_prediction':engine.read(R/'prediction_before_execution.json')['proposal'],'actual_geometry':engine.read(R/'planned_action_01/geometry.json'),'execution':engine.read(R/'live_execution/execution.json'),'limitations':['Wrist camera moves; process pixel motion is not material motion.','Controller completed != semantic grasp verified.','Single sequential trial after online01, not randomized independent reset.','History comparison restricted to supplied frozen proposal; do not infer unseen facts.','Mechanism support and target reveal are separate; reveal alone does not prove mechanism.','Do not downgrade target reveal because other areas wrinkle.','If before/after semantic identity insufficient retain UNCERTAIN.']}
r=engine.call('isolated_outcome_analysis','你在独立新会话中进行真实结果分析。只使用当前实验的原始图、日志和提前保存的预测。中文输出，区分观察、推断和机制，不读取文件或调用工具。',payload,[R/'before_capture/observation/rgb.png',R/'live_execution/after/rgb.png',R/'process_1.jpg',R/'process_2.jpg'],schema,'observer',['hindsight01']);engine.save(R/'outcome.json',engine.require(r));print(json.dumps(r['output'],ensure_ascii=False,indent=2))
