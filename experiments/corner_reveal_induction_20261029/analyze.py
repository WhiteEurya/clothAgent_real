"""Before-only proposal evaluation and report; no real robot calls."""
import json, math, hashlib, base64, html
from pathlib import Path
import cv2
import numpy as np
from candidate_skill import generate_action, observable_geometry
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/corner_reveal_induction_20261029';OUT.mkdir(exist_ok=True)
records=json.loads((ROOT/'reports/corner_reveal_mining_20261029/candidates.json').read_text())['success_candidates']
results=[];checks=[]
for record in records:
    eid=record['event_id'];source=ROOT/record['source']
    # Copy only physical action fields; planner language is not evidence.
    dispatch=json.loads((source/'dispatch_intent.json').read_text())
    action={k:dispatch['plan'][k] for k in ['grasp_px','target_px','motion_type','grasp_angle_deg']}
    execution=json.loads((source/'execution.json').read_text())
    assert execution['physical_execution'] and execution['status']=='EXECUTED'
    geometry=json.loads((source/'geometry.json').read_text())
    action.update(distance_mm=geometry['distance_mm'],lift_height_mm=geometry['primitive']['lift_height_mm'])
    rgb=cv2.cvtColor(cv2.imread(str(source/'before/rgb.png')),cv2.COLOR_BGR2RGB)
    prediction=generate_action(rgb,'RS')
    mask,points,c,v,u,q,bounds=observable_geometry(rgb)
    grasp=np.array(action['grasp_px'],float);delta=np.array(action['target_px'])-grasp
    fraction=abs(delta@u)/np.linalg.norm(delta)
    errors=[]
    for proposal in prediction['hypothesis_proposals']:
        pu=np.array(proposal['direction_xy_unit']); angle=math.degrees(math.acos(np.clip(delta@pu/np.linalg.norm(delta),-1,1)))
        errors.append({'grasp_error_px':float(np.linalg.norm(grasp-proposal['grasp_px'])),'direction_error_deg':angle})
    result={'event_id':eid,'source':record['source'],'actual_action':action,
            'before_sha256':hashlib.sha256((source/'before/rgb.png').read_bytes()).hexdigest(),
            'after_sha256':hashlib.sha256((source/'after/rgb.png').read_bytes()).hexdigest(),
            'assessment':'UNCERTAIN_PREEXISTING_EXPOSURE' if eid=='E153' else 'PROVISIONAL_REVEAL_SUPPORTED',
            'human_confirmed':False,'quantitative':{'visible_cross_axis_unit':u.tolist(),
            'cross_axis_pull_fraction':float(fraction),'angle_to_cross_axis_deg':math.degrees(math.acos(fraction)),
            'outward_relative_to_shape_centroid':bool(((grasp-c)@u)*(delta@u)>0),
            'robust_cross_span_px':float(bounds[1]-bounds[0]),
            'command_pixel_distance_over_cross_span':float(np.linalg.norm(delta)/(bounds[1]-bounds[0]))},
            'before_only_proposals':prediction,'posthoc_comparison_not_a_selection_rule':errors}
    results.append(result)
    # Rotation equivariance of the unordered proposal set, not physical success.
    h,w=rgb.shape[:2];rot=np.rot90(rgb).copy();pr=generate_action(rot,'RS')
    transform=lambda p:np.array([p[1],w-1-p[0]])
    pairs=prediction['hypothesis_proposals']; rp=pr['hypothesis_proposals']; assert len(pairs)==len(rp) and len(pairs)>0
    err=max(min(np.linalg.norm(transform(a['grasp_px'])-b['grasp_px']) for b in rp) for a in pairs)
    terr=max(min(np.linalg.norm(transform(a['target_px'])-b['target_px']) for b in rp) for a in pairs)
    assert err<2 and terr<2,(eid,err,terr)
    padded=np.pad(rgb,((37,23),(51,29),(0,0)));pt=generate_action(padded,'RS')['hypothesis_proposals']
    pe=max(min(np.linalg.norm(np.array(a['grasp_px'])+[51,37]-b['grasp_px']) for b in pt) for a in pairs)
    assert pe<2
    checks.append({'case':eid,'rotation90_grasp_error_px':float(err),'rotation90_target_error_px':float(terr),'translation_grasp_error_px':float(pe),'meaning':'image-coordinate equivariance only; not physical generalization'})
    # Actual vs candidate overlays are derivatives; originals linked separately.
    overlay=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(overlay,contours,-1,(110,190,110),1)
    for p in pairs:
        a=tuple(np.rint(p['grasp_px']).astype(int));b=tuple(np.rint(p['target_px']).astype(int));cv2.arrowedLine(overlay,a,b,(230,200,30),2,tipLength=.12)
    cv2.arrowedLine(overlay,tuple(action['grasp_px']),tuple(action['target_px']),(0,70,255),4,tipLength=.12)
    center=tuple(np.rint(c).astype(int));end=tuple(np.rint(c+120*u).astype(int));cv2.line(overlay,center,end,(255,255,255),2)
    cv2.putText(overlay,'RED: recorded | CYAN: hypothesis proposals | WHITE: PCA cross axis',(20,32),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    cv2.imwrite(str(OUT/(eid+'_geometry.png')),overlay)
assert generate_action(np.zeros((100,100,3),np.uint8),'RS')['decision']=='ABSTAIN'
assert generate_action(np.zeros((100,100,3),np.uint8),'LH')['decision']=='ABSTAIN'
checks.extend([{'case':'blank','result':'ABSTAIN'},{'case':'unsupported_hem','result':'ABSTAIN'}])
(OUT/'analysis.json').write_text(json.dumps({'cases':results,'checks':checks,'physical_actions':0,'formal_policy_changes':0},ensure_ascii=False,indent=2))
print(json.dumps({'measurements':[{k:r[k] for k in ['event_id','quantitative','posthoc_comparison_not_a_selection_rule']} for r in results],'checks':checks},ensure_ascii=False,indent=2))
