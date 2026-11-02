from pathlib import Path
import sys,json,hashlib,time
from PIL import Image
import numpy as np
ROOT=Path('/mnt/newssd/sja/clothAgent_real')
sys.path.insert(0,str(ROOT/'reports/gcpe_v01_20261027'))
import engine
sys.path.insert(0,str(ROOT/'experiments/corner_reveal_induction_20261029'))
from candidate_skill import generate_action
OUT=ROOT/'reports/online_corner_reveal_experiment01_20261029'
engine.OUT=OUT
im=OUT/'before_capture/observation/rgb.png'
g=generate_action(np.asarray(Image.open(im).convert('RGB')),'RS')
engine.save(OUT/'geometry_candidates.json',g)
s={'type':'string'}
pair={'type':'array','items':{'type':'integer'},'minItems':2,'maxItems':2}
fields={k:s for k in ['semantic_mapping','before_evidence','uncertainty','reason','predicted_deformation','falsification','direction_description','purpose']}
fields.update(decision={'type':'string','enum':['EXPLORATORY_ACTION','ABSTAIN']},grasp_px=pair,target_px=pair,grasp_angle_deg={'type':'integer','minimum':0,'maximum':180},before_occlusion_visually_confirmed={'type':'boolean'})
schema={'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}
payload={'image_order':['fresh stationary Before RGB 1280x720'],'task':'One experimental action to reveal garment-semantic RIGHT cuff. User prepared a self-occluded right cuff, new state; no correct direction or grasp supplied. This is not full flattening.','prior_evidence':'E353: one actual side pull was followed by renewed visibility of previously inconspicuous right cuff. Causal mechanism, grasp-side selector and transverse-axis generalization remain unverified. No historical coordinates supplied.','geometry_hypotheses':g,'interface':'Pa/Pb original RGB pixels, gripper angle image degrees 0..180. Existing lift-pull-place, offline original depth/workspace/IK/collision checks afterward; maximum XY displacement 150 mm. Choose a visible reliable grasp and visible table target; do not claim safety from RGB. Lift configuration will be saved separately.','interpretation':'PCA is visible shape axis, NOT proven semantic garment axis. The candidate generator ABSTAINS. Any selected action is an experimental hypothesis, not learned autonomous policy success. You may choose a newly grounded action instead of geometric candidates. If ABSTAIN use [0,0] coordinates and state why.'}
r=engine.call('new_state_proposal_retry_transport','Analyze only the supplied fresh image. Output Chinese reasoning and explicit uncertainty. Do not access tools/files. Infer garment orientation from visible neckline/seams, never assume image horizontal is garment transverse. Choose at most one testable reveal hypothesis; predict relative cloth deformation and observable falsifiers before After exists. Do not assert hidden layer topology. No retries.',payload,[im],schema,'exploration_proposal',['experiment01'])
p=engine.require(r)
engine.save(OUT/'prediction_before_execution.json',{'saved_at_unix':time.time(),'before_sha256':engine.sha(im),'physical_actions':0,'proposal':p,'model_result':str(OUT/'calls/jpeg_new_state_proposal_retry_transport/result.json'),'requires_human_execution_confirmation':True,'maximum_actions':1})
engine.overlay({},dict(decision='ACTION' if p['decision']=='EXPLORATORY_ACTION' else 'ABSTAIN',grasp_px=p['grasp_px'],target_px=p['target_px']),OUT/'unused.png') if False else None
print(json.dumps(p,ensure_ascii=False,indent=2))
