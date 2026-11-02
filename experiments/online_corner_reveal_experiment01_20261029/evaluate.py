from pathlib import Path
import sys,json
ROOT=Path('/mnt/newssd/sja/clothAgent_real');sys.path.insert(0,str(ROOT/'reports/gcpe_v01_20261027'))
import engine
R=ROOT/'reports/online_corner_reveal_experiment01_20261029';engine.OUT=R
s={'type':'string'};fields={k:s for k in ['before_identity_evidence','after_identity_evidence','observed_changes','prediction_comparison','e353_comparison','limitations','next_hypothesis']};fields['outcome']={'type':'string','enum':['CONFIRMED_REVEAL','NO_REVEAL','UNCERTAIN']}
schema={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
payload={'image_order':['Before','After release and return to original observation pose'],'prediction':engine.read(R/'prediction_before_execution.json')['proposal'],'execution':engine.read(R/'live_execution/execution.json'),'prior_E353':'A side pull was followed by right cuff reappearance, mechanism and grasp-side selection unverified. No established generalization.','user_before_fact':'User explicitly prepared garment-semantic right cuff self-occluded; front-facing garment right is image-left, but independently inspect identity.','criterion':'Only target semantic right cuff reveal, other curls do not negate reveal. A newly seen edge alone is not true cuff proof. If initial identity/occlusion or After identity insufficient mark UNCERTAIN. No claims of physical grasp success solely from controller completion.'}
x=engine.call('after_evaluation','Analyze actual Before and After in Chinese; no tools. Distinguish observed contour from hidden topology. Assess frozen prediction, report one of specified outcomes. No new actions; suggest a testable next hypothesis only, do not deploy rules.',payload,[R/'before_capture/observation/rgb.png',R/'live_execution/after/rgb.png'],schema,'observer',['experiment01'])
engine.save(R/'outcome.json',engine.require(x))
print(json.dumps(x['output'],ensure_ascii=False,indent=2))
